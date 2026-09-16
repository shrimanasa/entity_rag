import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.chem_utils import SharedSemanticEmbedder
from src.stage1_mention_detection import MentionDetector
from src.stage2_entity_linker import EntityLinker
from src.stage3_canonical_graph import CanonicalGraphBuilder

class RetrievalPipeline:
    """
    Stage 5 Retrieval & RRF Fusion Engine:
    3 Channels: Real Dense SVD Embeddings, BM25, Entity-ID (Salience-ranked, NIL returns [])
    RRF Fusion (filtered non-empty channels)
    6 Conditions:
      0: Random
      1: Dense only
      2: BM25 only
      3: BM25 + Dense (baseline)
      4: Entity-linked only
      5: Entity-linked + BM25 + Dense (full system)
    """
    def __init__(self, tagged_corpus: List[Dict], entity_linker: EntityLinker, graph_builder: CanonicalGraphBuilder, embedder: Optional[SharedSemanticEmbedder] = None, k_ent: int = 20):
        self.corpus = tagged_corpus
        self.k_ent = k_ent
        self.corpus_dict = {p["passage_id"]: p for p in tagged_corpus}
        self.passage_ids = [p["passage_id"] for p in tagged_corpus]
        self.linker = entity_linker
        self.builder = graph_builder
        self.embedder = embedder
        self.detector = MentionDetector(alias_dict=graph_builder.alias_dict, embedder=embedder)

        corpus_texts = [f"{p.get('title','')} {p.get('abstract','')} {p.get('text','')}" for p in tagged_corpus]

        # BM25 / TF-IDF setup
        self.bm25_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        self.bm25_matrix = self.bm25_vectorizer.fit_transform(corpus_texts)

        # Shared Real Dense LSA / SVD Matrix
        if self.embedder and self.embedder.is_fitted:
            self.dense_matrix = self.embedder.transform(corpus_texts)
        else:
            self.dense_matrix = np.zeros((len(corpus_texts), 128), dtype=np.float32)

    def retrieve_dense(self, query: str, top_k: int = 15) -> List[Tuple[str, float]]:
        """Real Dense semantic retrieval channel."""
        if not (self.embedder and self.embedder.is_fitted):
            return []
            
        q_emb = self.embedder.embed_text(query)
        sims = np.dot(self.dense_matrix, q_emb)
        ranked_indices = np.argsort(sims)[::-1][:top_k]
        return [(self.passage_ids[i], float(sims[i])) for i in ranked_indices if sims[i] > 0]

    def retrieve_bm25(self, query: str, top_k: int = 15) -> List[Tuple[str, float]]:
        """BM25 / lexical retrieval channel."""
        q_vec = self.bm25_vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.bm25_matrix)[0]
        ranked_indices = np.argsort(sims)[::-1][:top_k]
        return [(self.passage_ids[i], float(sims[i])) for i in ranked_indices if sims[i] > 0]

    def retrieve_entity_id(self, query: str, top_k: int = 15) -> List[Tuple[str, float]]:
        """
        Entity-ID retrieval channel:
        Links query mention -> canonical ID -> ranks matching passages by confidence * salience.
        Returns [] (empty list) on NIL or failure to avoid ghost candidate injection.
        """
        query_mentions = self.detector.extract_mentions(query)
        
        canonical_id = "NIL"
        confidence = 0.0

        if query_mentions:
            canonical_id, confidence = self.linker.link_mention(query_mentions[0])
            
        if canonical_id == "NIL":
            # Fallback to direct text alias matching
            resolved = self.builder.map_mention_to_canonical(query)
            if resolved != "NIL":
                canonical_id = resolved
                confidence = 0.85

        if canonical_id == "NIL" or confidence < self.linker.tau:
            return []

        scored_passages = []
        for p in self.corpus:
            tagged = p.get("tagged_entities", {})
            salience = tagged.get(canonical_id, 0.0)
            if salience > 0.0:
                score = confidence * salience
                scored_passages.append((p["passage_id"], score))

        if not scored_passages:
            return []

        scored_passages.sort(key=lambda x: x[1], reverse=True)
        return scored_passages[:top_k]

    def rrf_fusion(self, channel_rankings: List[List[Tuple[str, float]]], k: int = 60, channel_weights: Optional[List[float]] = None) -> List[Tuple[str, float]]:
        """
        Reciprocal Rank Fusion over active non-empty channels:
        RRF(d) = sum_r ( w_r / (k + rank(d, r)) )
        """
        rrf_scores: Dict[str, float] = {}
        
        for ch_idx, ranking in enumerate(channel_rankings):
            if not ranking:
                continue
            w = channel_weights[ch_idx] if channel_weights and ch_idx < len(channel_weights) else 1.0
            for rank_idx, (pid, score) in enumerate(ranking, 1):
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (w / (k + rank_idx))

        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    def retrieve_condition(self, query: str, condition: int, top_k: int = 10) -> List[Tuple[str, float]]:
        """Executes retrieval for one of the 6 conditions."""
        if condition == 0:  # Random
            shuffled = list(self.passage_ids)
            rng = random.Random(abs(hash(query)))
            rng.shuffle(shuffled)
            return [(pid, 1.0 / (i + 1)) for i, pid in enumerate(shuffled[:top_k])]

        elif condition == 1:  # Dense only
            return self.retrieve_dense(query, top_k)

        elif condition == 2:  # BM25 only
            return self.retrieve_bm25(query, top_k)

        elif condition == 3:  # BM25 + Dense (Baseline)
            r_dense = self.retrieve_dense(query, top_k=20)
            r_bm25 = self.retrieve_bm25(query, top_k=20)
            return self.rrf_fusion([r_bm25, r_dense], k=60)[:top_k]

        elif condition == 4:  # Entity-linked only
            return self.retrieve_entity_id(query, top_k)

        elif condition == 5:  # Entity-linked + BM25 + Dense (Full System)
            r_dense = self.retrieve_dense(query, top_k=20)
            r_bm25 = self.retrieve_bm25(query, top_k=20)
            r_ent = self.retrieve_entity_id(query, top_k=20)
            return self.rrf_fusion([r_bm25, r_dense, r_ent], k=60)[:top_k]

        else:
            raise ValueError(f"Unknown condition index: {condition}")

    def retrieve(self, query: str, condition: int = 5, top_k: int = 10) -> List[Tuple[str, float]]:
        """Convenience wrapper for retrieve_condition."""
        return self.retrieve_condition(query=query, condition=condition, top_k=top_k)
