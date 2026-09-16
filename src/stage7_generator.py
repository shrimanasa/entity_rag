import re
import string
from typing import Dict, List, Optional, Tuple

class RAGAnswerGenerator:
    """
    Stage 7 Downstream QA Answer Generator.
    Synthesizes answer responses from top-k retrieved passages for a given query.
    Supports neural/extractive context synthesis and optional transformer pipelines.
    """
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name
        self.pipeline = None
        if model_name:
            try:
                from transformers import pipeline
                self.pipeline = pipeline("text2text-generation", model=model_name)
            except Exception as e:
                print(f"Notice: Transformer model '{model_name}' not loaded ({e}). Using robust context-synthesis.")

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join(text.split()).strip()

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def synthesize_from_passages(self, query: str, passages: List[Dict], max_passages: int = 3) -> Dict:
        """
        Synthesizes an answer using top retrieved passages.
        If no passages provided, returns an insufficient context response.
        """
        if not passages:
            return {
                "answer": "Insufficient context retrieved to answer the query.",
                "cited_passages": [],
                "confidence": 0.0
            }

        top_passages = passages[:max_passages]
        cited_pids = [p["passage_id"] for p in top_passages]

        candidate_sentences = []
        for p in top_passages:
            full_text = f"{p.get('abstract', '')} {p.get('text', '')}".strip()
            sents = self._split_sentences(full_text)
            for s in sents:
                candidate_sentences.append((s, p["passage_id"]))

        if not candidate_sentences:
            return {
                "answer": "Insufficient context retrieved to answer the query.",
                "cited_passages": cited_pids,
                "confidence": 0.0
            }

        stopwords = {
            'the', 'of', 'and', 'a', 'in', 'to', 'is', 'was', 'for', 'with', 'by',
            'at', 'on', 'an', 'under', 'regarding', 'novel', 'experimental', 'methodology',
            'characterization', 'functional', 'analysis', 'testing', 'operational', 'performance',
            'microstructural', 'evolution', 'study', 'investigation', 'properties', 'behavior'
        }
        
        q_tokens = [w.lower().strip(string.punctuation) for w in query.split()]
        q_keywords = [w for w in q_tokens if w and w not in stopwords]

        property_markers = {'exhibits', 'revealed', 'transition', 'temperature', 'tc', 'structure', 'lattice',
                            'perovskite', 'density', 'superconducting', 'band', 'conductivity', 'doping',
                            'measured', 'observed', 'synthesis', 'critical', 'group', 'symmetry'}

        scored_sentences: List[Tuple[float, str, str]] = []
        for sent, pid in candidate_sentences:
            sent_lower = sent.lower()
            s_tokens = set(w.strip(string.punctuation) for w in sent_lower.split())

            keyword_hits = sum(1 for kw in q_keywords if kw in sent_lower)
            marker_hits = sum(1 for pm in property_markers if pm in s_tokens)
            has_numbers = bool(re.search(r'\b\d+(?:\.\d+)?\b', sent))

            score = (keyword_hits * 2.5) + (marker_hits * 1.5) + (2.0 if has_numbers else 0.0)
            
            if any(term in sent_lower for term in ['exhibits', 'revealed', 'confirmed', 'characterized', 'transition']):
                score += 3.0

            scored_sentences.append((score, sent, pid))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        best_sentence = scored_sentences[0][1]
        answer = best_sentence.rstrip('.') + '.'
        
        return {
            "answer": answer,
            "cited_passages": cited_pids,
            "confidence": min(1.0, scored_sentences[0][0] / 10.0)
        }

    def generate(self, query: str, passages: List[Dict], max_passages: int = 3) -> Dict:
        """
        Unified generation endpoint. Uses transformer if available, otherwise extractive synthesis.
        """
        if self.pipeline and passages:
            context = " ".join([p.get('text', '') for p in passages[:max_passages]])
            prompt = f"question: {query} context: {context}"
            try:
                out = self.pipeline(prompt, max_length=128, clean_up_tokenization_spaces=True)
                return {
                    "answer": out[0]['generated_text'],
                    "cited_passages": [p["passage_id"] for p in passages[:max_passages]],
                    "confidence": 0.9
                }
            except Exception:
                pass

        return self.synthesize_from_passages(query, passages, max_passages=max_passages)
