import os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratch", "matplotlib"))
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from src.stage5_retriever import RetrievalPipeline

def compute_recall_at_k(retrieved_pids: List[str], ground_truth_pid: str, k: int = 10) -> float:
    return 1.0 if ground_truth_pid in retrieved_pids[:k] else 0.0

def compute_mrr(retrieved_pids: List[str], ground_truth_pid: str) -> float:
    for idx, pid in enumerate(retrieved_pids, 1):
        if pid == ground_truth_pid:
            return 1.0 / idx
    return 0.0

def compute_ndcg_at_k(retrieved_pids: List[str], ground_truth_pid: str, k: int = 10) -> float:
    for idx, pid in enumerate(retrieved_pids[:k], 1):
        if pid == ground_truth_pid:
            return 1.0 / math.log2(idx + 1)
    return 0.0

def paired_bootstrap_significance_test(scores_cond5: List[float], scores_cond3: List[float],
                                         n_resamples: int = 1000, alpha: float = 0.05) -> Dict:
    """
    Paired bootstrap resampling (1000 resamples).
    Delta = condition 5 - condition 3.
    Significant only if 95% CI excludes zero.
    """
    n = len(scores_cond5)
    if n == 0 or len(scores_cond3) != n:
        return {"mean_delta": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "significant": False}

    rng = np.random.RandomState(42)
    deltas = []
    
    for _ in range(n_resamples):
        indices = rng.choice(n, size=n, replace=True)
        sample_5 = [scores_cond5[i] for i in indices]
        sample_3 = [scores_cond3[i] for i in indices]
        deltas.append(np.mean(sample_5) - np.mean(sample_3))

    mean_delta = float(np.mean(deltas))
    ci_lower = float(np.percentile(deltas, (alpha / 2) * 100))
    ci_upper = float(np.percentile(deltas, (1 - alpha / 2) * 100))
    significant = bool(ci_lower > 0.0 or ci_upper < 0.0)

    return {
        "mean_delta": round(mean_delta, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "significant": significant
    }

class Evaluator:
    """
    Stage 6 Evaluator & Decisive Plot Generator:
    Metrics: Recall@10, MRR, nDCG@10 x 3 buckets x 6 conditions.
    Paired bootstrap significance test (1000 resamples).
    Generates decisive plot: Alias divergence vs Recall@10.
    """
    def __init__(self, pipeline: RetrievalPipeline):
        self.pipeline = pipeline

    def evaluate_dataset(self, labeled_qa: List[Dict]) -> Tuple[Dict, Dict]:
        conditions = [0, 1, 2, 3, 4, 5]
        buckets = ["No-mismatch", "Partial-mismatch", "Full alias-mismatch"]

        raw_scores: Dict[str, Dict[int, Dict[str, List[float]]]] = {
            b: {c: {"recall10": [], "mrr": [], "ndcg10": []} for c in conditions}
            for b in buckets
        }

        for qa in labeled_qa:
            bucket = qa["bucket"]
            gt_pid = qa["evidence_passage_id"]
            q_text = qa["query"]

            for c in conditions:
                retrieved = self.pipeline.retrieve_condition(q_text, condition=c, top_k=10)
                r_pids = [item[0] for item in retrieved]

                rec10 = compute_recall_at_k(r_pids, gt_pid, k=10)
                mrr_val = compute_mrr(r_pids, gt_pid)
                ndcg10 = compute_ndcg_at_k(r_pids, gt_pid, k=10)

                raw_scores[bucket][c]["recall10"].append(rec10)
                raw_scores[bucket][c]["mrr"].append(mrr_val)
                raw_scores[bucket][c]["ndcg10"].append(ndcg10)

        table_results: Dict[str, Dict[int, Dict[str, float]]] = {
            b: {c: {} for c in conditions} for b in buckets
        }

        for b in buckets:
            for c in conditions:
                rec_list = raw_scores[b][c]["recall10"]
                mrr_list = raw_scores[b][c]["mrr"]
                ndcg_list = raw_scores[b][c]["ndcg10"]

                table_results[b][c]["Recall@10"] = round(float(np.mean(rec_list)) if rec_list else 0.0, 4)
                table_results[b][c]["MRR"] = round(float(np.mean(mrr_list)) if mrr_list else 0.0, 4)
                table_results[b][c]["nDCG@10"] = round(float(np.mean(ndcg_list)) if ndcg_list else 0.0, 4)

        # Bootstrap Significance Tests for Delta = Cond 5 - Cond 3 per bucket across metrics
        bootstrap_results = {}
        for b in buckets:
            bootstrap_results[b] = {
                "Recall@10": paired_bootstrap_significance_test(raw_scores[b][5]["recall10"], raw_scores[b][3]["recall10"]),
                "MRR": paired_bootstrap_significance_test(raw_scores[b][5]["mrr"], raw_scores[b][3]["mrr"]),
                "nDCG@10": paired_bootstrap_significance_test(raw_scores[b][5]["ndcg10"], raw_scores[b][3]["ndcg10"])
            }

        return table_results, bootstrap_results

    def generate_decisive_plot(self, table_results: Dict, output_path: str = "results/decisive_plot.png"):
        """
        Produces the decisive plot:
        x-axis: Alias-divergence bucket
        y-axis: MRR / Recall@10
        One line per condition (0: Random, 1: Dense, 2: BM25, 3: BM25+Dense, 4: Entity-only, 5: Full system)
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        buckets = ["No-mismatch", "Partial-mismatch", "Full alias-mismatch"]
        condition_labels = {
            0: "0: Random",
            1: "1: Dense only",
            2: "2: BM25 only",
            3: "3: BM25 + Dense (Baseline)",
            4: "4: Entity-linked only",
            5: "5: Full System (Entity + BM25 + Dense)"
        }
        condition_colors = {
            0: "#7f7f7f",
            1: "#1f77b4",
            2: "#ff7f0e",
            3: "#2ca02c",
            4: "#9467bd",
            5: "#d62728"
        }
        condition_markers = {0: 'x', 1: 's', 2: '^', 3: 'o', 4: 'd', 5: 'D'}

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

        # Subplot 1: Recall@10
        for c in [0, 1, 2, 3, 4, 5]:
            y_vals = [table_results[b][c]["Recall@10"] for b in buckets]
            axes[0].plot(buckets, y_vals, label=condition_labels[c],
                         color=condition_colors[c], marker=condition_markers[c],
                         linewidth=2.5 if c in (3, 5) else 1.5,
                         linestyle='-' if c == 5 else '--')

        axes[0].set_title("Alias Divergence vs. Recall@10", fontsize=12, fontweight='bold')
        axes[0].set_xlabel("Alias-Divergence Bucket", fontsize=11, fontweight='bold')
        axes[0].set_ylabel("Recall@10", fontsize=11, fontweight='bold')
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].grid(True, linestyle=':', alpha=0.6)

        # Subplot 2: MRR (Rank Quality Recovery)
        for c in [0, 1, 2, 3, 4, 5]:
            y_vals = [table_results[b][c]["MRR"] for b in buckets]
            axes[1].plot(buckets, y_vals, label=condition_labels[c],
                         color=condition_colors[c], marker=condition_markers[c],
                         linewidth=2.5 if c in (3, 5) else 1.5,
                         linestyle='-' if c == 5 else '--')

        axes[1].set_title("Alias Divergence vs. MRR (Rank Recovery)", fontsize=12, fontweight='bold')
        axes[1].set_xlabel("Alias-Divergence Bucket", fontsize=11, fontweight='bold')
        axes[1].set_ylabel("Mean Reciprocal Rank (MRR)", fontsize=11, fontweight='bold')
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].grid(True, linestyle=':', alpha=0.6)

        plt.suptitle("When Does Entity-Aware Retrieval Earn Its Keep?", fontsize=14, fontweight='bold', y=0.98)
        axes[1].legend(fontsize=9, loc="lower left", framealpha=0.9)
        plt.tight_layout()

        plt.savefig(output_path)
        plt.close()
        print(f"--> Decisive Plot saved to {output_path}")

def run_stage6_evaluation(labeled_qa: List[Dict], pipeline: RetrievalPipeline) -> Tuple[Dict, Dict]:
    print("=== Stage 6: Stratified Evaluation & Bootstrap Significance Testing ===")
    evaluator = Evaluator(pipeline)
    table_results, bootstrap_results = evaluator.evaluate_dataset(labeled_qa)

    print("\n--- RESULTS TABLE (Recall@10 / MRR / nDCG@10) ---")
    for b, conds in table_results.items():
        print(f"\nBucket: [{b}]")
        print(f"{'Condition':<40} | {'Recall@10':<10} | {'MRR':<10} | {'nDCG@10':<10}")
        print("-" * 80)
        for c, metrics in conds.items():
            print(f"Cond {c}: {metrics['Recall@10']:<10.4f} | {metrics['MRR']:<10.4f} | {metrics['nDCG@10']:<10.4f}")

    print("\n--- PAIRED BOOTSTRAP SIGNIFICANCE TEST (1000 Resamples, Δ = Cond 5 - Cond 3) ---")
    for b, metrics_bs in bootstrap_results.items():
        mrr_bs = metrics_bs["MRR"]
        sig_str = "SIGNIFICANT (p < 0.05)" if mrr_bs["significant"] else "NOT Significant (CI includes 0)"
        print(f"Bucket: [{b:<20}] MRR Δ = {mrr_bs['mean_delta']:+.4f} | 95% CI: [{mrr_bs['ci_lower']:+.4f}, {mrr_bs['ci_upper']:+.4f}] -> {sig_str}")

    evaluator.generate_decisive_plot(table_results)
    return table_results, bootstrap_results

if __name__ == '__main__':
    pass
