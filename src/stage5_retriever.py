import hashlib
import random
import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from rank_bm25 import BM25Okapi
from src.chem_utils import SharedSemanticEmbedder
from src.stage1_mention_detection import MentionDetector
from src.stage2_entity_linker import EntityLinker
from src.stage3_canonical_graph import CanonicalGraphBuilder

# Minimal English stopword set used by the BM25 tokenizer.
# Keeps punctuation-stripped terms (e.g. "YBCO" from "YBCO,") while
# filtering high-frequency function words that inflate IDF denominators.
_BM25_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "to", "for", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "shall",
    "with", "at", "by", "from", "on", "as", "it", "its", "this", "that",
    "these", "those", "not", "no", "but", "if", "so", "than", "into",
    "about", "also", "up", "out", "can", "we", "our", "their", "which",
}

def _bm25_tokenize(text: str) -> List[str]:
    """
    Tokenize for BM25Okapi.
    Lowercases, strips leading/trailing punctuation from each token,
    and removes English stopwords. This ensures "YBCO," → "ybco" and
    matches corpus tokens correctly (fix #3 — baseline handicap).
    """
    tokens = []
    for tok in text.lower().split():
        tok = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", tok)
        if tok and tok not in _BM25_STOPWORDS:
            tokens.append(tok)
    return tokens


class RetrievalPipeline:
    """
    Stage 5 Retrieval & RRF Fusion Engine.

    Channels
    --------
    Dense     : LSA/TruncatedSVD semantic embeddings (SharedSemanticEmbedder)
    BM25      : True BM25Okapi term-weighting (rank_bm25), punctuation-stripped tokenizer
    Entity-ID : Salience-ranked by confidence × salience; returns [] on NIL

    Conditions
    ----------
    0 : Random       — hashlib-sha256-seeded, process-stable
    1 : Dense only
    2 : BM25 only    — true BM25Okapi
    3 : BM25 + Dense — equal-weight RRF (strongest non-entity baseline)
    4 : Entity-linked only
    5 : Weighted RRF — entity_weight × entity + 1 × BM25 + 1 × Dense
        entity_weight is set at construction time (default 3.0).
        It should be chosen on a held-out split, not on the eval set.

    Limitations
    -----------
    - Dense channel is LSA, not a neural embedder; treat as a weak baseline.
    - Entity channel never abstains on the QA set (100% link rate); see
      linking_accuracy() for per-query correctness.
    - Corpus, queries and aliases are fully synthetic.
    """

    def __init__(
        self,
        tagged_corpus: List[Dict],
        entity_linker: EntityLinker,
        graph_builder: CanonicalGraphBuilder,
        embedder: Optional[SharedSemanticEmbedder] = None,
        k_ent: int = 20,
        entity_weight: float = 3.0,
        neural_embedder: Optional[Any] = None,
        gate_threshold: float = 10.0,
    ):
        self.corpus = tagged_corpus
        self.k_ent = k_ent
        self.entity_weight = entity_weight
        self.gate_threshold = gate_threshold
        self.corpus_dict = {p["passage_id"]: p for p in tagged_corpus}
        self.passage_ids = [p["passage_id"] for p in tagged_corpus]
        self.linker = entity_linker
        self.builder = graph_builder
        self.embedder = embedder
        self.neural_embedder = neural_embedder
        self.detector = MentionDetector(
            alias_dict=graph_builder.alias_dict, embedder=embedder
        )

        corpus_texts = [
            f"{p.get('title','')} {p.get('abstract','')} {p.get('text','')}"
            for p in tagged_corpus
        ]

        # BM25Okapi index with punctuation-stripped tokenizer (fix #3 baseline handicap)
        tokenized_corpus = [_bm25_tokenize(t) for t in corpus_texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)

        # Dense LSA/SVD matrix
        if self.embedder and self.embedder.is_fitted:
            self.dense_matrix = self.embedder.transform(corpus_texts)
        else:
            self.dense_matrix = np.zeros((len(corpus_texts), 128), dtype=np.float32)

        # Neural Dense matrix (e.g. SentenceTransformer all-MiniLM-L6-v2)
        if self.neural_embedder is not None:
            self.neural_dense_matrix = self.neural_embedder.encode(
                corpus_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
            )
        else:
            self.neural_dense_matrix = None

        # Multi-candidate expansion mapping (fixes alias polysemy ambiguity)
        self.surface_to_candidates: Dict[str, List[str]] = {}
        for ent_id, info in graph_builder.alias_dict.items():
            names = [info.get("primary_name", "")]
            for cat, a_list in info.get("categories", {}).items():
                names.extend(a_list)
            for n in names:
                n_low = n.lower().strip()
                if n_low:
                    if n_low not in self.surface_to_candidates:
                        self.surface_to_candidates[n_low] = []
                    if ent_id not in self.surface_to_candidates[n_low]:
                        self.surface_to_candidates[n_low].append(ent_id)

        # Inverted index from canonical entity to passages for fast O(1) multi-candidate retrieval
        self.entity_to_passages: Dict[str, List[Tuple[str, float]]] = {}
        for p in tagged_corpus:
            pid = p["passage_id"]
            for ent_id, salience in p.get("tagged_entities", {}).items():
                if salience > 0.0:
                    if ent_id not in self.entity_to_passages:
                        self.entity_to_passages[ent_id] = []
                    self.entity_to_passages[ent_id].append((pid, salience))

    # ------------------------------------------------------------------
    # Individual retrieval channels
    # ------------------------------------------------------------------

    def retrieve_dense(self, query: str, top_k: int = 15) -> List[Tuple[str, float]]:
        """Dense semantic retrieval via LSA/SVD dot-product similarity."""
        if not (self.embedder and self.embedder.is_fitted):
            return []
        q_emb = self.embedder.embed_text(query)
        sims = np.dot(self.dense_matrix, q_emb)
        ranked = np.argsort(sims)[::-1][:top_k]
        return [(self.passage_ids[i], float(sims[i])) for i in ranked if sims[i] > 0]

    def retrieve_neural(self, query: str, top_k: int = 15) -> List[Tuple[str, float]]:
        """Real Neural Dense semantic retrieval (e.g. SentenceTransformer all-MiniLM-L6-v2)."""
        if self.neural_dense_matrix is None or self.neural_embedder is None:
            return []
        q_emb = self.neural_embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = np.dot(self.neural_dense_matrix, q_emb)
        ranked = np.argsort(sims)[::-1][:top_k]
        return [(self.passage_ids[i], float(sims[i])) for i in ranked if sims[i] > 0]

    def retrieve_bm25(self, query: str, top_k: int = 15) -> List[Tuple[str, float]]:
        """True BM25Okapi retrieval with punctuation-stripped tokenizer."""
        tokens = _bm25_tokenize(query)
        scores = self.bm25_index.get_scores(tokens)
        ranked = np.argsort(scores)[::-1][:top_k]
        return [(self.passage_ids[i], float(scores[i])) for i in ranked if scores[i] > 0]

    def retrieve_entity_id(
        self,
        query: str,
        top_k: int = 15,
        return_metadata: bool = False,
        use_multi_candidate: bool = True,
    ):
        """
        Entity-ID retrieval channel with multi-candidate entity expansion.

        When a query mention matches an ambiguous surface form that maps to k entities in the
        alias dictionary (e.g. NaFePO4 polymorphs), 1-best selection reduces to a 1/k coin flip.
        Multi-candidate expansion retrieves passages tagged with ANY of the k candidates,
        weighting each candidate equally by (confidence / k).
        For unambiguous queries (k=1), this is identically equivalent to 1-best retrieval.

        Returns
        -------
        If return_metadata=False:
            results : List[Tuple[str, float]], empty on NIL/low-confidence
        If return_metadata=True:
            (results, link_meta) : Tuple[List[Tuple[str, float]], Dict]
        """
        query_mentions = self.detector.extract_mentions(query)

        candidates: List[str] = []
        confidence = 0.0
        best_canonical = "NIL"

        if query_mentions:
            best_canonical, confidence = self.linker.link_mention(query_mentions[0])
            for m in query_mentions:
                sf = m.get("surface_form", "").lower().strip()
                c_list = self.surface_to_candidates.get(sf, [])
                for cid in c_list:
                    if cid not in candidates:
                        candidates.append(cid)

        if not candidates:
            # Fallback 1: direct alias-dictionary text match
            resolved = self.builder.map_mention_to_canonical(query)
            if resolved != "NIL":
                best_canonical = resolved
                candidates.append(resolved)
                confidence = max(confidence, 0.85)

        if not candidates and best_canonical != "NIL":
            candidates.append(best_canonical)

        if confidence <= 0.0 and candidates:
            confidence = 0.85

        is_linked = len(candidates) > 0 and confidence >= self.linker.tau
        link_meta = {
            "canonical_id": best_canonical if best_canonical != "NIL" else (candidates[0] if candidates else "NIL"),
            "candidates": candidates,
            "num_candidates": len(candidates),
            "is_ambiguous": len(candidates) > 1,
            "confidence": round(confidence, 4),
            "linked": is_linked,
        }

        if not link_meta["linked"]:
            return ([], link_meta) if return_metadata else []

        scored = []
        if use_multi_candidate:
            k = len(candidates)
            weight_per_candidate = confidence / k
            p_scores: Dict[str, float] = {}
            for cand in candidates:
                for pid, sal in self.entity_to_passages.get(cand, []):
                    p_scores[pid] = p_scores.get(pid, 0.0) + weight_per_candidate * sal
            scored = list(p_scores.items())
        else:
            cid = link_meta["canonical_id"]
            p_scores = {}
            for pid, sal in self.entity_to_passages.get(cid, []):
                p_scores[pid] = confidence * sal
            scored = list(p_scores.items())

        if not scored:
            return ([], link_meta) if return_metadata else []

        # Score-descending; break ties by passage_id for determinism
        # (confidence is constant per query; salience ∈ {1.0, 1.5})
        scored.sort(key=lambda x: (-x[1], x[0]))
        results = scored[:top_k]
        return (results, link_meta) if return_metadata else results

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    def rrf_fusion(
        self,
        channel_rankings: List[List[Tuple[str, float]]],
        k: int = 60,
        channel_weights: Optional[List[float]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Weighted Reciprocal Rank Fusion:
        RRF(d) = Σ_r  w_r / (k + rank(d, r))
        Skips empty channels so a NIL entity channel doesn't zero-score others.
        """
        rrf_scores: Dict[str, float] = {}
        for ch_idx, ranking in enumerate(channel_rankings):
            if not ranking:
                continue
            w = (
                channel_weights[ch_idx]
                if channel_weights and ch_idx < len(channel_weights)
                else 1.0
            )
            for rank_idx, (pid, _) in enumerate(ranking, 1):
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + w / (k + rank_idx)
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # Condition dispatch
    # ------------------------------------------------------------------

    def retrieve_condition(
        self, query: str, condition: int, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Execute one of 6 retrieval conditions.
        Conditions 0–4 return List[Tuple[str,float]].
        Condition 5 uses self.entity_weight set at construction time.
        """
        if condition == 0:
            # hashlib seed: stable across Python processes (hash() is salted per-process)
            shuffled = list(self.passage_ids)
            seed = int(hashlib.sha256(query.encode("utf-8")).hexdigest(), 16) % (2**32)
            rng = random.Random(seed)
            rng.shuffle(shuffled)
            return [(pid, 1.0 / (i + 1)) for i, pid in enumerate(shuffled[:top_k])]

        elif condition == 1:
            return self.retrieve_dense(query, top_k)

        elif condition == 2:
            return self.retrieve_bm25(query, top_k)

        elif condition == 3:
            # Equal-weight BM25 + Dense: strongest non-entity baseline
            r_bm25 = self.retrieve_bm25(query, top_k=20)
            r_dense = self.retrieve_dense(query, top_k=20)
            return self.rrf_fusion([r_bm25, r_dense], k=60)[:top_k]

        elif condition == 4:
            return self.retrieve_entity_id(query, top_k)

        elif condition == 5:
            # Weighted RRF: entity channel boosted by self.entity_weight.
            # Weight must be chosen on a held-out split; see run_pipeline.py.
            r_bm25 = self.retrieve_bm25(query, top_k=20)
            r_dense = self.retrieve_dense(query, top_k=20)
            r_ent = self.retrieve_entity_id(query, top_k=20)
            return self.rrf_fusion(
                [r_bm25, r_dense, r_ent],
                k=60,
                channel_weights=[1.0, 1.0, self.entity_weight],
            )[:top_k]

        elif condition == 6:
            # Neural Dense only (e.g. all-MiniLM-L6-v2)
            return self.retrieve_neural(query, top_k)

        elif condition == 7:
            # BM25 + Neural Dense (Neural Hybrid Baseline)
            r_bm25 = self.retrieve_bm25(query, top_k=20)
            r_neural = self.retrieve_neural(query, top_k=20)
            return self.rrf_fusion([r_bm25, r_neural], k=60)[:top_k]

        elif condition == 8:
            # Full Neural System (Entity-linked + BM25 + Neural Dense)
            r_bm25 = self.retrieve_bm25(query, top_k=20)
            r_neural = self.retrieve_neural(query, top_k=20)
            r_ent = self.retrieve_entity_id(query, top_k=20)
            return self.rrf_fusion(
                [r_bm25, r_neural, r_ent],
                k=60,
                channel_weights=[1.0, 1.0, self.entity_weight],
            )[:top_k]

        elif condition == 9:
            # Condition 9: Gated Entity RAG (Classical LSA)
            # A mismatch gate monitors lexical (BM25) search confidence.
            # If an entity is linked AND BM25 top score <= gate_threshold:
            #   High divergence / alias mismatch detected!
            #   Activates entity-aware channel (Dense + Entity) to avoid dilution from failing BM25 terms.
            # Else:
            #   Standard hybrid retrieval (BM25 + Dense).
            r_bm25 = self.retrieve_bm25(query, top_k=20)
            top_bm25_score = r_bm25[0][1] if r_bm25 else 0.0
            r_dense = self.retrieve_dense(query, top_k=20)
            r_ent, link_meta = self.retrieve_entity_id(query, top_k=20, return_metadata=True)
            
            gate_triggered = bool(link_meta["linked"] and top_bm25_score <= self.gate_threshold)
            if gate_triggered:
                return self.rrf_fusion([r_dense, r_ent], k=60, channel_weights=[1.0, 2.0])[:top_k]
            else:
                return self.rrf_fusion([r_bm25, r_dense], k=60)[:top_k]

        elif condition == 10:
            # Condition 10: Gated Neural System (all-MiniLM-L6-v2)
            r_bm25 = self.retrieve_bm25(query, top_k=20)
            top_bm25_score = r_bm25[0][1] if r_bm25 else 0.0
            r_neural = self.retrieve_neural(query, top_k=20)
            r_ent, link_meta = self.retrieve_entity_id(query, top_k=20, return_metadata=True)
            
            gate_triggered = bool(link_meta["linked"] and top_bm25_score <= self.gate_threshold)
            if gate_triggered:
                return self.rrf_fusion([r_neural, r_ent], k=60, channel_weights=[1.0, 2.0])[:top_k]
            else:
                return self.rrf_fusion([r_bm25, r_neural], k=60)[:top_k]

        else:
            raise ValueError(f"Unknown condition index: {condition}")

    def retrieve(
        self, query: str, condition: int = 5, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Convenience wrapper for retrieve_condition."""
        return self.retrieve_condition(query=query, condition=condition, top_k=top_k)
