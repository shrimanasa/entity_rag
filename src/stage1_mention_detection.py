import os
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
from src.chem_utils import AliasMatcher, SharedSemanticEmbedder

class MentionDetector:
    """
    Stage 1 Mention Detector:
    Extracts material entity mentions, ±20-token context windows, and real semantic dense embeddings.
    """
    def __init__(self, alias_dict: Optional[Dict] = None, embedder: Optional[SharedSemanticEmbedder] = None, alias_dict_path: str = "data/alias_dictionary.json"):
        if alias_dict is None:
            if os.path.exists(alias_dict_path):
                with open(alias_dict_path, "r", encoding="utf-8") as f:
                    alias_dict = json.load(f)
            else:
                alias_dict = {}
                
        self.matcher = AliasMatcher(alias_dict)
        self.embedder = embedder
        self.embedding_dim = embedder.n_components if embedder else 128

    def extract_mentions(self, text: str) -> List[Dict]:
        mentions = self.matcher.extract_mentions(text)
        
        for m in mentions:
            embed_input = f"{m['surface_form']} {m['context_window']}"
            if self.embedder and self.embedder.is_fitted:
                m["embedding"] = self.embedder.embed_text(embed_input)
            else:
                vec = np.ones(self.embedding_dim, dtype=np.float32)
                m["embedding"] = vec / np.linalg.norm(vec)
                
        return mentions

    def get_contextual_embedding(self, surface_form: str, context_window: str) -> np.ndarray:
        embed_input = f"{surface_form} {context_window}"
        if self.embedder and self.embedder.is_fitted:
            return self.embedder.embed_text(embed_input)
        vec = np.ones(self.embedding_dim, dtype=np.float32)
        return vec / np.linalg.norm(vec)

    def process_corpus(self, corpus: List[Dict]) -> List[Dict]:
        processed_corpus = []
        for passage in corpus:
            full_text = f"{passage['title']} {passage['abstract']} {passage['text']}"
            mentions = self.extract_mentions(full_text)
            p_copy = dict(passage)
            p_copy["full_text"] = full_text
            p_copy["extracted_mentions"] = mentions
            processed_corpus.append(p_copy)
            
        return processed_corpus
