import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.chem_utils import formula_exact_match, token_jaccard_similarity, SharedSemanticEmbedder
from src.stage0_ground_truth import load_alias_dictionary

class EntityLinker:
    """
    Stage 2 Entity Linker:
    Combines 3 real independent signals:
    score(m, e) = alpha * lexical + beta * embedding + gamma * context
    Uses authentic corpus passages to build distinctive entity centroids.
    """
    def __init__(self, alias_dict_path: str = "data/alias_dictionary.json",
                 corpus_path: str = "data/corpus.json",
                 embedder: Optional[SharedSemanticEmbedder] = None,
                 alpha: float = 0.40, beta: float = 0.35, gamma: float = 0.25, tau: float = 0.50):
        self.alias_dict = load_alias_dictionary(alias_dict_path)
        self.corpus_path = corpus_path
        self.embedder = embedder
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        
        self.entity_centroids: Dict[str, np.ndarray] = {}
        self.entity_context_profiles: Dict[str, str] = {}
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self._initialize_profiles()

    def _initialize_profiles(self):
        # 1. Aggregate authentic corpus passages per entity
        entity_corpus_texts: Dict[str, List[str]] = {ent_id: [] for ent_id in self.alias_dict.keys()}
        if os.path.exists(self.corpus_path):
            try:
                with open(self.corpus_path, "r", encoding="utf-8") as f:
                    corpus = json.load(f)
                    for p in corpus:
                        cid = p.get("canonical_entity")
                        if cid in entity_corpus_texts:
                            p_text = f"{p.get('title','')} {p.get('abstract','')} {p.get('text','')}"
                            entity_corpus_texts[cid].append(p_text)
            except Exception:
                pass

        # 2. Construct authentic, distinctive profile per entity
        corpus_texts = []
        for ent_id, info in self.alias_dict.items():
            all_aliases = []
            for cat, aliases in info.get("categories", {}).items():
                all_aliases.extend(aliases)
                
            passages_str = " ".join(entity_corpus_texts.get(ent_id, []))
            profile_text = f"{info.get('primary_name','')} {' '.join(all_aliases)} {passages_str}".strip()
            
            self.entity_context_profiles[ent_id] = profile_text
            corpus_texts.append(profile_text)
            
            if self.embedder and self.embedder.is_fitted:
                self.entity_centroids[ent_id] = self.embedder.embed_text(profile_text)
            else:
                vec = np.ones(self.embedder.n_components if self.embedder else 128, dtype=np.float32)
                self.entity_centroids[ent_id] = vec / np.linalg.norm(vec)

        self.vectorizer.fit(corpus_texts)
        self.entity_tfidf_mats = {
            ent_id: self.vectorizer.transform([self.entity_context_profiles[ent_id]])
            for ent_id in self.alias_dict.keys()
        }

    def compute_lexical_signal(self, surface_form: str, entity_id: str) -> float:
        ent_info = self.alias_dict.get(entity_id, {})
        if not ent_info:
            return 0.0

        if formula_exact_match(surface_form, ent_info.get("primary_name", "")):
            return 1.0

        max_sim = 0.0
        for cat, aliases in ent_info.get("categories", {}).items():
            for alias in aliases:
                if formula_exact_match(surface_form, alias):
                    return 1.0
                if surface_form.lower() == alias.lower():
                    return 1.0
                sim = token_jaccard_similarity(surface_form, alias)
                if sim > max_sim:
                    max_sim = sim
                    
        return max_sim

    def compute_embedding_signal(self, mention_embedding: np.ndarray, entity_id: str) -> float:
        centroid = self.entity_centroids.get(entity_id)
        if centroid is None or mention_embedding is None:
            return 0.0
            
        dot = float(np.dot(mention_embedding, centroid))
        return max(0.0, min(1.0, dot))

    def compute_context_signal(self, context_window: str, entity_id: str) -> float:
        if not context_window or entity_id not in self.entity_tfidf_mats:
            return 0.0
            
        ctx_mat = self.vectorizer.transform([context_window])
        ent_mat = self.entity_tfidf_mats[entity_id]
        sim = cosine_similarity(ctx_mat, ent_mat)[0][0]
        return float(sim)

    def score_mention_entity_fast(self, lex_score: float, emb_score: float, ctx_score: float) -> float:
        return (self.alpha * lex_score) + (self.beta * emb_score) + (self.gamma * ctx_score)

    def link_mention(self, mention: Dict) -> Tuple[str, float]:
        sf = mention.get("surface_form", "")
        ctx = mention.get("context_window", "")
        emb = mention.get("embedding")
        if emb is None and self.embedder and self.embedder.is_fitted:
            emb = self.embedder.embed_text(f"{sf} {ctx}")

        best_entity = "NIL"
        best_score = 0.0

        for ent_id in self.alias_dict.keys():
            lex = self.compute_lexical_signal(sf, ent_id)
            emb_sc = self.compute_embedding_signal(emb, ent_id) if emb is not None else 0.0
            ctx_sc = self.compute_context_signal(ctx, ent_id)
            sc = self.score_mention_entity_fast(lex, emb_sc, ctx_sc)
            if sc > best_score:
                best_score = sc
                best_entity = ent_id

        if best_score < self.tau:
            return "NIL", best_score
            
        return best_entity, best_score

def grid_search_hyperparameters(dev_pairs: List[Dict], embedder: Optional[SharedSemanticEmbedder] = None, alias_dict_path: str = "data/alias_dictionary.json", corpus_path: str = "data/corpus.json") -> Dict:
    print("=== Stage 2: Entity Linker Grid Search ===")
    
    alpha_vals = [0.3, 0.5, 0.7]
    beta_vals = [0.2, 0.4]
    gamma_vals = [0.1, 0.3]
    tau_vals = [0.3, 0.5, 0.7]

    base_linker = EntityLinker(alias_dict_path=alias_dict_path, corpus_path=corpus_path, embedder=embedder)

    # Pre-calculate signal triples (lex, emb, ctx) for all dev pairs across all entities using real embeddings
    precalculated_signals = []
    entities = list(base_linker.alias_dict.keys())

    for pair in dev_pairs:
        m1 = pair["mention_1"]
        ctx_window = f"{m1} material property synthesis"
        m1_emb = embedder.embed_text(f"{m1} {ctx_window}") if (embedder and embedder.is_fitted) else None
        
        item_signals = {}
        for ent_id in entities:
            lex = base_linker.compute_lexical_signal(m1, ent_id)
            emb_sc = base_linker.compute_embedding_signal(m1_emb, ent_id) if m1_emb is not None else 0.0
            ctx_sc = base_linker.compute_context_signal(ctx_window, ent_id)
            item_signals[ent_id] = (lex, emb_sc, ctx_sc)
        precalculated_signals.append((pair, item_signals))

    best_f1 = -1.0
    best_params = {}

    for a in alpha_vals:
        for b in beta_vals:
            for g in gamma_vals:
                total_w = a + b + g
                norm_a, norm_b, norm_g = a / total_w, b / total_w, g / total_w
                
                for t in tau_vals:
                    base_linker.alpha = norm_a
                    base_linker.beta = norm_b
                    base_linker.gamma = norm_g
                    base_linker.tau = t
                    
                    tp, fp, fn = 0, 0, 0
                    for pair, item_signals in precalculated_signals:
                        gt_target = pair["canonical_id"]
                        
                        best_ent = "NIL"
                        best_sc = 0.0
                        for ent_id, (lex, emb_sc, ctx_sc) in item_signals.items():
                            sc = base_linker.score_mention_entity_fast(lex, emb_sc, ctx_sc)
                            if sc > best_sc:
                                best_sc = sc
                                best_ent = ent_id
                                
                        if best_sc < t:
                            best_ent = "NIL"
                            
                        if pair["same_entity"] == 1:
                            if best_ent == gt_target:
                                tp += 1
                            else:
                                fn += 1
                        else:
                            if best_ent != "NIL":
                                fp += 1

                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

                    if f1 > best_f1:
                        best_f1 = f1
                        best_params = {
                            "alpha": round(norm_a, 4),
                            "beta": round(norm_b, 4),
                            "gamma": round(norm_g, 4),
                            "tau": round(t, 4),
                            "f1": round(f1, 4),
                            "precision": round(prec, 4),
                            "recall": round(rec, 4)
                        }

    print(f"--> Grid Search Optimal Parameters: α={best_params['alpha']}, β={best_params['beta']}, γ={best_params['gamma']}, τ={best_params['tau']}")
    print(f"--> Best Linking Dev F1: {best_params['f1'] * 100:.2f}% (P: {best_params['precision']*100:.1f}%, R: {best_params['recall']*100:.1f}%)\n")
    return best_params
