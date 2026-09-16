import re
import string
from typing import Dict, List, Tuple, Any
import numpy as np
from src.stage7_generator import RAGAnswerGenerator
from src.stage5_retriever import RetrievalPipeline

def normalize_text(text: str) -> str:
    """Standard QA text normalization: lowercase, strip punctuation, remove articles and excess whitespace."""
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())

def compute_token_f1(pred: str, target: str) -> Tuple[float, float, float]:
    """Computes token-level precision, recall, and F1 between prediction and ground truth."""
    pred_tokens = normalize_text(pred).split()
    target_tokens = normalize_text(target).split()

    if not pred_tokens or not target_tokens:
        return 0.0, 0.0, 0.0

    common_tokens = set(pred_tokens) & set(target_tokens)
    if not common_tokens:
        return 0.0, 0.0, 0.0

    # Count overlaps accounting for frequency
    pred_count = sum(1 for t in pred_tokens if t in common_tokens)
    target_count = sum(1 for t in target_tokens if t in common_tokens)

    precision = pred_count / len(pred_tokens)
    recall = target_count / len(target_tokens)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def extract_key_facts(text: str) -> List[str]:
    """Extracts high-salience factual tokens (temperatures, numbers, chemical terms)."""
    facts = []
    # Match numbers with units (e.g. 92 K, 39 K, 140 K, 10 GPa)
    units = re.findall(r'\b\d+(?:\.\d+)?\s*(?:k|gpa|ev|kpa|s/cm|t)\b', text.lower())
    facts.extend(units)
    # Match space groups or crystal structures
    structures = re.findall(r'\b(pmmm|perovskite|tetragonal|orthorhombic|hexagonal|cubic|monoclinic)\b', text.lower())
    facts.extend(structures)
    return facts

def compute_entity_property_fidelity(pred: str, target: str) -> float:
    """Evaluates whether key physical properties present in ground truth are preserved in generated answer."""
    target_facts = extract_key_facts(target)
    if not target_facts:
        return 1.0 if normalize_text(pred) in normalize_text(target) or normalize_text(target) in normalize_text(pred) else 0.0
    
    pred_lower = pred.lower()
    hits = sum(1 for f in target_facts if f in pred_lower)
    return hits / len(target_facts)

class Stage7Evaluator:
    """
    Evaluates downstream end-to-end RAG QA performance:
    Compares Baseline RAG (Condition 3: BM25 + Dense) vs Full System RAG (Condition 5: Entity-Aware + BM25 + Dense).
    """
    def __init__(self, corpus_dict: Dict[str, Dict]):
        self.corpus_dict = corpus_dict
        self.generator = RAGAnswerGenerator()

    def evaluate_pipeline(self, labeled_qa: List[Dict], pipeline: RetrievalPipeline) -> Dict:
        """
        Runs QA generation on retrieved passages for Condition 3 and Condition 5,
        evaluates against ground_truth_answer, and stratifies across divergence buckets.
        """
        retrieval_results: Dict[str, Dict[int, List[str]]] = {}
        bucket_mapping: Dict[str, str] = {}

        for item in labeled_qa:
            qid = item["query_id"]
            query = item["query"]
            bucket_mapping[qid] = item["bucket"]

            # Retrieve top-5 for condition 3 (Baseline) and condition 5 (Full System)
            c3_res = pipeline.retrieve_condition(query, condition=3, top_k=5)
            c5_res = pipeline.retrieve_condition(query, condition=5, top_k=5)

            retrieval_results[qid] = {
                3: [p for p, _ in c3_res],
                5: [p for p, _ in c5_res]
            }

        return self.evaluate_qa_dataset(labeled_qa, retrieval_results, bucket_mapping)

    def evaluate_qa_dataset(
        self,
        qa_dataset: List[Dict],
        retrieval_results: Dict[str, Dict[int, List[str]]],
        bucket_mapping: Dict[str, str]
    ) -> Dict:
        """
        Runs QA generation on retrieved passages for Condition 3 and Condition 5,
        and stratifies metrics across the 3 alias-mismatch buckets.
        """
        bucket_eval: Dict[str, Dict[str, List[float]]] = {
            "No-mismatch": {"c3_f1": [], "c5_f1": [], "c3_fidelity": [], "c5_fidelity": []},
            "Partial-mismatch": {"c3_f1": [], "c5_f1": [], "c3_fidelity": [], "c5_fidelity": []},
            "Full alias-mismatch": {"c3_f1": [], "c5_f1": [], "c3_fidelity": [], "c5_fidelity": []}
        }

        generated_samples = []

        for item in qa_dataset:
            qid = item["query_id"]
            query = item["query"]
            gt_answer = item["ground_truth_answer"]
            bucket = bucket_mapping.get(qid, "No-mismatch")

            # Condition 3 (BM25 + Dense Baseline)
            c3_pids = retrieval_results.get(qid, {}).get(3, [])
            c3_passages = [self.corpus_dict[pid] for pid in c3_pids if pid in self.corpus_dict]
            c3_gen = self.generator.generate(query, c3_passages, max_passages=3)
            _, _, c3_f1 = compute_token_f1(c3_gen["answer"], gt_answer)
            c3_fid = compute_entity_property_fidelity(c3_gen["answer"], gt_answer)

            # Condition 5 (Full Entity-Aware System)
            c5_pids = retrieval_results.get(qid, {}).get(5, [])
            c5_passages = [self.corpus_dict[pid] for pid in c5_pids if pid in self.corpus_dict]
            c5_gen = self.generator.generate(query, c5_passages, max_passages=3)
            _, _, c5_f1 = compute_token_f1(c5_gen["answer"], gt_answer)
            c5_fid = compute_entity_property_fidelity(c5_gen["answer"], gt_answer)

            bucket_eval[bucket]["c3_f1"].append(c3_f1)
            bucket_eval[bucket]["c5_f1"].append(c5_f1)
            bucket_eval[bucket]["c3_fidelity"].append(c3_fid)
            bucket_eval[bucket]["c5_fidelity"].append(c5_fid)

            if len(generated_samples) < 10 or (bucket == "Full alias-mismatch" and len(generated_samples) < 15):
                generated_samples.append({
                    "query_id": qid,
                    "query": query,
                    "bucket": bucket,
                    "ground_truth": gt_answer,
                    "condition_3_answer": c3_gen["answer"],
                    "condition_5_answer": c5_gen["answer"],
                    "c3_f1": round(c3_f1, 4),
                    "c5_f1": round(c5_f1, 4)
                })

        # Aggregate summary and statistical significance bootstrap (1000 resamples)
        summary = {}
        for b, data in bucket_eval.items():
            n = len(data["c3_f1"])
            if n == 0:
                continue
            mean_c3_f1 = float(np.mean(data["c3_f1"]))
            mean_c5_f1 = float(np.mean(data["c5_f1"]))
            mean_c3_fid = float(np.mean(data["c3_fidelity"]))
            mean_c5_fid = float(np.mean(data["c5_fidelity"]))

            # Paired bootstrap on F1 delta (C5 - C3)
            deltas = np.array(data["c5_f1"]) - np.array(data["c3_f1"])
            rng = np.random.RandomState(42)
            bs_deltas = []
            for _ in range(1000):
                sample_idx = rng.choice(n, size=n, replace=True)
                bs_deltas.append(np.mean(deltas[sample_idx]))
            
            ci_lower = float(np.percentile(bs_deltas, 2.5))
            ci_upper = float(np.percentile(bs_deltas, 97.5))
            significant = bool(ci_lower > 0.0)

            summary[b] = {
                "count": n,
                "baseline_f1": round(mean_c3_f1, 4),
                "full_system_f1": round(mean_c5_f1, 4),
                "f1_delta": round(mean_c5_f1 - mean_c3_f1, 4),
                "f1_ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
                "significant": significant,
                "baseline_fidelity": round(mean_c3_fid, 4),
                "full_system_fidelity": round(mean_c5_fid, 4),
                "fidelity_delta": round(mean_c5_fid - mean_c3_fid, 4)
            }

        return {
            "summary": summary,
            "samples": generated_samples
        }
