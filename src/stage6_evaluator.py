import os
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratch", "matplotlib"),
)
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Set, Any
from src.stage5_retriever import RetrievalPipeline


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Bootstrap significance
# ---------------------------------------------------------------------------

def paired_bootstrap_significance_test(
    scores_a: List[float],
    scores_b: List[float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
) -> Dict:
    """
    Paired bootstrap resampling.
    Delta = scores_a - scores_b (typically cond5 - baseline).
    Significant only if 95% CI strictly excludes zero (two-sided).
    """
    n = len(scores_a)
    if n == 0 or len(scores_b) != n:
        return {"mean_delta": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "significant": False}

    rng = np.random.RandomState(42)
    deltas = []
    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        deltas.append(
            float(np.mean([scores_a[i] for i in idx]) - np.mean([scores_b[i] for i in idx]))
        )

    mean_delta = float(np.mean(deltas))
    ci_lower = float(np.percentile(deltas, (alpha / 2) * 100))
    ci_upper = float(np.percentile(deltas, (1 - alpha / 2) * 100))
    significant = bool(ci_lower > 0.0 or ci_upper < 0.0)

    return {
        "mean_delta": round(mean_delta, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "significant": significant,
    }


# ---------------------------------------------------------------------------
# Entity weight sweep (fix #2 & #3 — dev-only, preserves pipeline state)
# ---------------------------------------------------------------------------

def sweep_entity_weight(
    labeled_qa: List[Dict],
    pipeline: RetrievalPipeline,
    weights: List[float],
    split_indices: Optional[List[int]] = None,
    test_indices: Optional[List[int]] = None,
) -> Dict[str, Dict[float, float]]:
    """
    Evaluates Condition 5 MRR on:
      - Full alias-mismatch dev half (N=33)
      - Overall dev half (N=234, strictly excluding all test queries)
      - Full alias-mismatch held-out test half (N=32, true held out)
      - Overall test half (N=234)
    across candidate entity weights.
    Crucially saves and restores pipeline.entity_weight to avoid state mutation bugs.
    """
    orig_weight = pipeline.entity_weight
    try:
        full_qa = [qa for qa in labeled_qa if qa["bucket"] == "Full alias-mismatch"]
        if split_indices is not None:
            full_dev = [full_qa[i] for i in split_indices if i < len(full_qa)]
        else:
            full_dev = [full_qa[i] for i in range(len(full_qa)) if i % 2 == 0]

        if test_indices is not None:
            full_test = [full_qa[i] for i in test_indices if i < len(full_qa)]
        else:
            full_test = [full_qa[i] for i in range(len(full_qa)) if i % 2 == 1]

        # Quarantined non-test queries (strictly zero overlap with held-out test split)
        test_queries_set = set(q["query"] for q in full_test)
        non_test_qa = [q for q in labeled_qa if q["query"] not in test_queries_set]
        overall_dev = [q for idx, q in enumerate(non_test_qa) if idx % 2 == 0]
        overall_test = [q for idx, q in enumerate(non_test_qa) if idx % 2 == 1]

        full_dev_results = {}
        overall_dev_results = {}
        full_test_results = {}
        overall_test_results = {}

        for w in weights:
            pipeline.entity_weight = w

            # 1. Full-bucket dev MRR (N=33)
            mrrs_full_dev = [
                compute_mrr([x[0] for x in pipeline.retrieve_condition(qa["query"], condition=5, top_k=10)], qa["evidence_passage_id"])
                for qa in full_dev
            ]
            full_dev_results[w] = round(float(np.mean(mrrs_full_dev)) if mrrs_full_dev else 0.0, 4)

            # 2. Overall dev MRR (N=234, clean partition)
            mrrs_overall_dev = [
                compute_mrr([x[0] for x in pipeline.retrieve_condition(qa["query"], condition=5, top_k=10)], qa["evidence_passage_id"])
                for qa in overall_dev
            ]
            overall_dev_results[w] = round(float(np.mean(mrrs_overall_dev)) if mrrs_overall_dev else 0.0, 4)

            # 3. Full-bucket test MRR (N=32, true held out)
            mrrs_full_test = [
                compute_mrr([x[0] for x in pipeline.retrieve_condition(qa["query"], condition=5, top_k=10)], qa["evidence_passage_id"])
                for qa in full_test
            ]
            full_test_results[w] = round(float(np.mean(mrrs_full_test)) if mrrs_full_test else 0.0, 4)

            # 4. Overall test MRR (N=234)
            mrrs_overall_test = [
                compute_mrr([x[0] for x in pipeline.retrieve_condition(qa["query"], condition=5, top_k=10)], qa["evidence_passage_id"])
                for qa in overall_test
            ]
            overall_test_results[w] = round(float(np.mean(mrrs_overall_test)) if mrrs_overall_test else 0.0, 4)

        return {
            "full_mismatch_dev": full_dev_results,
            "overall_dev": overall_dev_results,
            "full_mismatch_test": full_test_results,
            "overall_test": overall_test_results,
        }
    finally:
        pipeline.entity_weight = orig_weight


def sweep_gate_threshold(
    labeled_qa: List[Dict],
    pipeline: RetrievalPipeline,
    thresholds: List[float],
    dev_indices: Optional[List[int]] = None,
    test_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Evaluates mismatch gate thresholds strictly on held-out dev split.
    Sweeps BM25 score thresholds to conditionally trigger entity retrieval
    when lexical search indicates alias mismatch.
    """
    orig_th = pipeline.gate_threshold
    try:
        full_qa = [qa for qa in labeled_qa if qa["bucket"] == "Full alias-mismatch"]
        if dev_indices is not None:
            full_dev = [full_qa[i] for i in dev_indices if i < len(full_qa)]
        else:
            full_dev = [full_qa[i] for i in range(len(full_qa)) if i % 2 == 0]

        if test_indices is not None:
            full_test = [full_qa[i] for i in test_indices if i < len(full_qa)]
        else:
            full_test = [full_qa[i] for i in range(len(full_qa)) if i % 2 == 1]

        test_queries_set = set(q["query"] for q in full_test)
        non_test_qa = [q for q in labeled_qa if q["query"] not in test_queries_set]
        overall_dev = [q for idx, q in enumerate(non_test_qa) if idx % 2 == 0]
        overall_test = [q for idx, q in enumerate(non_test_qa) if idx % 2 == 1]

        dev_sweep: Dict[float, Dict[str, float]] = {}
        test_sweep: Dict[float, Dict[str, float]] = {}
        best_th = 0.0
        best_dev_mrr = -1.0

        for th in thresholds:
            pipeline.gate_threshold = th

            # 1. Full-bucket dev MRR (N=33)
            mrrs_full_dev = [
                compute_mrr([x[0] for x in pipeline.retrieve_condition(qa["query"], condition=9, top_k=10)], qa["evidence_passage_id"])
                for qa in full_dev
            ]
            # 2. Overall dev MRR (N=234)
            mrrs_overall_dev = [
                compute_mrr([x[0] for x in pipeline.retrieve_condition(qa["query"], condition=9, top_k=10)], qa["evidence_passage_id"])
                for qa in overall_dev
            ]
            fd_mrr = round(float(np.mean(mrrs_full_dev)) if mrrs_full_dev else 0.0, 4)
            od_mrr = round(float(np.mean(mrrs_overall_dev)) if mrrs_overall_dev else 0.0, 4)
            dev_sweep[th] = {"full_dev_mrr": fd_mrr, "overall_dev_mrr": od_mrr}

            # 3. Held-out test curves (strictly reported, never used for selection)
            mrrs_full_test = [
                compute_mrr([x[0] for x in pipeline.retrieve_condition(qa["query"], condition=9, top_k=10)], qa["evidence_passage_id"])
                for qa in full_test
            ]
            mrrs_overall_test = [
                compute_mrr([x[0] for x in pipeline.retrieve_condition(qa["query"], condition=9, top_k=10)], qa["evidence_passage_id"])
                for qa in overall_test
            ]
            ft_mrr = round(float(np.mean(mrrs_full_test)) if mrrs_full_test else 0.0, 4)
            ot_mrr = round(float(np.mean(mrrs_overall_test)) if mrrs_overall_test else 0.0, 4)
            test_sweep[th] = {"full_test_mrr": ft_mrr, "overall_test_mrr": ot_mrr}

            # Select threshold that maximizes dev overall MRR
            if od_mrr > best_dev_mrr:
                best_dev_mrr = od_mrr
                best_th = th

        return {
            "best_threshold": best_th,
            "dev_sweep": dev_sweep,
            "test_sweep": test_sweep,
        }
    finally:
        pipeline.gate_threshold = orig_th


# ---------------------------------------------------------------------------
# Linking accuracy & dictionary ambiguity ceiling (fix #4)
# ---------------------------------------------------------------------------

def compute_linking_accuracy(labeled_qa: List[Dict], pipeline: RetrievalPipeline) -> Dict:
    """
    Evaluates per-query entity linking accuracy and explicitly measures the
    candidate-set ambiguity ceiling (how many queries use an alias that maps
    to >1 canonical entity in the dictionary).
    """
    alias_dict = pipeline.builder.alias_dict
    surface_to_entities: Dict[str, Set[str]] = {}
    for ent_id, info in alias_dict.items():
        aliases = [info.get("primary_name", "")]
        for cat, names in info.get("categories", {}).items():
            aliases.extend(names)
        for a in set(aliases):
            a_low = a.lower().strip()
            if a_low:
                surface_to_entities.setdefault(a_low, set()).add(ent_id)

    buckets = ["No-mismatch", "Partial-mismatch", "Full alias-mismatch"]
    stats: Dict[str, Dict] = {
        b: {
            "total": 0,
            "linked": 0,
            "correct_1best": 0,
            "candidate_contains_gold": 0,
            "ambiguous_queries": 0,
            "correct_when_unambiguous": 0,
            "total_unambiguous": 0,
            "correct_when_ambiguous": 0,
            "total_ambiguous": 0,
        }
        for b in buckets
    }

    for qa in labeled_qa:
        bucket = qa["bucket"]
        gt_canonical = qa.get("canonical_entity", "")
        _, link_meta = pipeline.retrieve_entity_id(qa["query"], top_k=1, return_metadata=True)

        mentions = pipeline.detector.extract_mentions(qa["query"])
        candidates: Set[str] = set()
        for m in mentions:
            sf = m["surface_form"].lower().strip()
            candidates.update(surface_to_entities.get(sf, set()))

        if not candidates:
            # Fallback direct map check
            mapped = pipeline.builder.map_mention_to_canonical(qa["query"])
            if mapped != "NIL":
                candidates.add(mapped)

        is_ambiguous = len(candidates) > 1
        cand_has_gold = gt_canonical in candidates
        is_1best_correct = (link_meta["canonical_id"] == gt_canonical)

        b_stat = stats[bucket]
        b_stat["total"] += 1
        if link_meta["linked"]:
            b_stat["linked"] += 1
            if is_1best_correct:
                b_stat["correct_1best"] += 1

        if cand_has_gold:
            b_stat["candidate_contains_gold"] += 1

        if is_ambiguous:
            b_stat["ambiguous_queries"] += 1
            b_stat["total_ambiguous"] += 1
            if is_1best_correct:
                b_stat["correct_when_ambiguous"] += 1
        else:
            b_stat["total_unambiguous"] += 1
            if is_1best_correct:
                b_stat["correct_when_unambiguous"] += 1

    summary = {}
    for b in buckets:
        t = stats[b]["total"]
        lk = stats[b]["linked"]
        cor = stats[b]["correct_1best"]
        ceil = stats[b]["candidate_contains_gold"]
        amb = stats[b]["ambiguous_queries"]
        t_unamb = stats[b]["total_unambiguous"]
        c_unamb = stats[b]["correct_when_unambiguous"]
        t_amb = stats[b]["total_ambiguous"]
        c_amb = stats[b]["correct_when_ambiguous"]

        summary[b] = {
            "total": t,
            "linked": lk,
            "link_rate": round(lk / t, 4) if t else 0.0,
            "accuracy_1best": round(cor / t, 4) if t else 0.0,
            "candidate_set_recall": round(ceil / t, 4) if t else 0.0,
            "ambiguous_rate": round(amb / t, 4) if t else 0.0,
            "acc_unambiguous": round(c_unamb / t_unamb, 4) if t_unamb else 0.0,
            "acc_ambiguous": round(c_amb / t_amb, 4) if t_amb else 0.0,
        }

    # Overall aggregate
    all_t = sum(stats[b]["total"] for b in buckets)
    all_lk = sum(stats[b]["linked"] for b in buckets)
    all_cor = sum(stats[b]["correct_1best"] for b in buckets)
    all_ceil = sum(stats[b]["candidate_contains_gold"] for b in buckets)
    all_amb = sum(stats[b]["ambiguous_queries"] for b in buckets)
    all_tunamb = sum(stats[b]["total_unambiguous"] for b in buckets)
    all_cunamb = sum(stats[b]["correct_when_unambiguous"] for b in buckets)
    all_tamb = sum(stats[b]["total_ambiguous"] for b in buckets)
    all_camb = sum(stats[b]["correct_when_ambiguous"] for b in buckets)

    summary["Overall"] = {
        "total": all_t,
        "linked": all_lk,
        "link_rate": round(all_lk / all_t, 4) if all_t else 0.0,
        "accuracy_1best": round(all_cor / all_t, 4) if all_t else 0.0,
        "candidate_set_recall": round(all_ceil / all_t, 4) if all_t else 0.0,
        "ambiguous_rate": round(all_amb / all_t, 4) if all_t else 0.0,
        "acc_unambiguous": round(all_cunamb / all_tunamb, 4) if all_tunamb else 0.0,
        "acc_ambiguous": round(all_camb / all_tamb, 4) if all_tamb else 0.0,
    }
    return summary


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Stage 6 Stratified Evaluator & Decisive Plot Generator.
    Evaluates retrieval metrics across all buckets, specifically including
    the true held-out test split of Full alias-mismatch and Overall (N=500).
    """

    def __init__(self, pipeline: RetrievalPipeline):
        self.pipeline = pipeline

    def evaluate_dataset(
        self,
        labeled_qa: List[Dict],
        weight_test_indices: Optional[List[int]] = None,
    ) -> Tuple[Dict, Dict]:
        # Conditions to evaluate: 0-5 standard, 9 (gated); 6-8, 10 if neural dense embedder loaded
        conditions = [0, 1, 2, 3, 4, 5, 9]
        if self.pipeline.neural_embedder is not None:
            conditions.extend([6, 7, 8, 10])

        # Separate Full-mismatch items
        full_qa = [qa for qa in labeled_qa if qa["bucket"] == "Full alias-mismatch"]
        if weight_test_indices is not None:
            full_test_qa = [full_qa[i] for i in weight_test_indices if i < len(full_qa)]
        else:
            full_test_qa = [full_qa[i] for i in range(len(full_qa)) if i % 2 == 1]

        evaluation_subsets = {
            "No-mismatch": [qa for qa in labeled_qa if qa["bucket"] == "No-mismatch"],
            "Partial-mismatch": [qa for qa in labeled_qa if qa["bucket"] == "Partial-mismatch"],
            "Full alias-mismatch (Held-Out Test)": full_test_qa,
            "Full alias-mismatch (All)": full_qa,
            "Overall (N=500)": labeled_qa,
        }

        raw: Dict[str, Dict[int, Dict[str, List[float]]]] = {
            s_name: {c: {"recall10": [], "mrr": [], "ndcg10": []} for c in conditions}
            for s_name in evaluation_subsets.keys()
        }

        # Precompute and cache retrieval rankings once per unique query and condition
        query_retrieval_cache: Dict[Tuple[str, int], List[str]] = {}
        for qa in labeled_qa:
            q_text = qa["query"]
            for c in conditions:
                if (q_text, c) not in query_retrieval_cache:
                    retrieved = self.pipeline.retrieve_condition(q_text, condition=c, top_k=10)
                    query_retrieval_cache[(q_text, c)] = [item[0] for item in retrieved]

        for s_name, qa_list in evaluation_subsets.items():
            for qa in qa_list:
                gt_pid = qa["evidence_passage_id"]
                q_text = qa["query"]
                for c in conditions:
                    r_pids = query_retrieval_cache[(q_text, c)]
                    raw[s_name][c]["recall10"].append(compute_recall_at_k(r_pids, gt_pid))
                    raw[s_name][c]["mrr"].append(compute_mrr(r_pids, gt_pid))
                    raw[s_name][c]["ndcg10"].append(compute_ndcg_at_k(r_pids, gt_pid))

        # Aggregate means
        table: Dict[str, Dict[int, Dict[str, float]]] = {
            s_name: {c: {} for c in conditions} for s_name in evaluation_subsets.keys()
        }
        for s_name in evaluation_subsets.keys():
            for c in conditions:
                for metric_key, out_key in [("recall10", "Recall@10"), ("mrr", "MRR"), ("ndcg10", "nDCG@10")]:
                    vals = raw[s_name][c][metric_key]
                    table[s_name][c][out_key] = round(float(np.mean(vals)) if vals else 0.0, 4)

        # Bootstrap significance tests
        bootstrap: Dict[str, Dict] = {}
        for s_name in evaluation_subsets.keys():
            bootstrap[s_name] = {
                "vs_cond3": {
                    "Recall@10": paired_bootstrap_significance_test(
                        raw[s_name][5]["recall10"], raw[s_name][3]["recall10"]
                    ),
                    "MRR": paired_bootstrap_significance_test(
                        raw[s_name][5]["mrr"], raw[s_name][3]["mrr"]
                    ),
                    "nDCG@10": paired_bootstrap_significance_test(
                        raw[s_name][5]["ndcg10"], raw[s_name][3]["ndcg10"]
                    ),
                },
                "vs_cond1": {
                    "MRR": paired_bootstrap_significance_test(
                        raw[s_name][5]["mrr"], raw[s_name][1]["mrr"]
                    ),
                },
                "cond9_vs_cond3": {
                    "Recall@10": paired_bootstrap_significance_test(
                        raw[s_name][9]["recall10"], raw[s_name][3]["recall10"]
                    ),
                    "MRR": paired_bootstrap_significance_test(
                        raw[s_name][9]["mrr"], raw[s_name][3]["mrr"]
                    ),
                    "nDCG@10": paired_bootstrap_significance_test(
                        raw[s_name][9]["ndcg10"], raw[s_name][3]["ndcg10"]
                    ),
                },
                "cond9_vs_cond1": {
                    "MRR": paired_bootstrap_significance_test(
                        raw[s_name][9]["mrr"], raw[s_name][1]["mrr"]
                    ),
                },
                "cond9_vs_cond5": {
                    "MRR": paired_bootstrap_significance_test(
                        raw[s_name][9]["mrr"], raw[s_name][5]["mrr"]
                    ),
                },
            }
            if 6 in conditions:
                bootstrap[s_name]["vs_neural_dense"] = {
                    "MRR": paired_bootstrap_significance_test(
                        raw[s_name][5]["mrr"], raw[s_name][6]["mrr"]
                    ),
                }
            if 8 in conditions and 7 in conditions:
                bootstrap[s_name]["cond8_vs_cond7"] = {
                    "Recall@10": paired_bootstrap_significance_test(
                        raw[s_name][8]["recall10"], raw[s_name][7]["recall10"]
                    ),
                    "MRR": paired_bootstrap_significance_test(
                        raw[s_name][8]["mrr"], raw[s_name][7]["mrr"]
                    ),
                    "nDCG@10": paired_bootstrap_significance_test(
                        raw[s_name][8]["ndcg10"], raw[s_name][7]["ndcg10"]
                    ),
                }
            if 10 in conditions and 7 in conditions:
                bootstrap[s_name]["cond10_vs_cond7"] = {
                    "Recall@10": paired_bootstrap_significance_test(
                        raw[s_name][10]["recall10"], raw[s_name][7]["recall10"]
                    ),
                    "MRR": paired_bootstrap_significance_test(
                        raw[s_name][10]["mrr"], raw[s_name][7]["mrr"]
                    ),
                    "nDCG@10": paired_bootstrap_significance_test(
                        raw[s_name][10]["ndcg10"], raw[s_name][7]["ndcg10"]
                    ),
                }
                bootstrap[s_name]["cond10_vs_cond6"] = {
                    "MRR": paired_bootstrap_significance_test(
                        raw[s_name][10]["mrr"], raw[s_name][6]["mrr"]
                    ),
                }

        return table, bootstrap

    def generate_decisive_plot(
        self, table_results: Dict, output_path: str = "results/decisive_plot.png"
    ):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        buckets = ["No-mismatch", "Partial-mismatch", "Full alias-mismatch (Held-Out Test)", "Overall (N=500)"]
        condition_labels = {
            0: "0: Random",
            1: "1: Dense LSA",
            2: "2: BM25",
            3: "3: BM25 + Dense LSA (Baseline)",
            4: "4: Entity-only",
            5: "5: Full System (Fixed)",
            6: "6: Neural Dense (MiniLM)",
            7: "7: BM25 + Neural Dense (Neural Baseline)",
            8: "8: Full Neural System (Fixed)",
            9: "9: Gated Entity RAG",
            10: "10: Gated Neural System",
        }
        condition_colors = {
            0: "#7f7f7f", 1: "#1f77b4", 2: "#ff7f0e",
            3: "#2ca02c", 4: "#9467bd", 5: "#d62728",
            6: "#17becf", 7: "#bcbd22", 8: "#e377c2",
            9: "#8c564b", 10: "#2b5c8f",
        }
        condition_markers = {0: "x", 1: "s", 2: "^", 3: "o", 4: "d", 5: "D", 6: "P", 7: "v", 8: "*", 9: "o", 10: "p"}

        conds_to_plot = [0, 1, 2, 3, 4, 5, 9]
        if 7 in table_results.get("No-mismatch", {}):
            conds_to_plot.append(7)
        if 8 in table_results.get("No-mismatch", {}):
            conds_to_plot.append(8)
        if 10 in table_results.get("No-mismatch", {}):
            conds_to_plot.append(10)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

        for ax, metric in zip(axes, ["Recall@10", "MRR"]):
            for c in conds_to_plot:
                y_vals = [table_results.get(b, {}).get(c, {}).get(metric, 0.0) for b in buckets]
                ax.plot(
                    ["No-mismatch", "Partial", "Full (Held-Out)", "Overall"],
                    y_vals,
                    label=condition_labels.get(c, f"Cond {c}"),
                    color=condition_colors.get(c, "#333333"),
                    marker=condition_markers.get(c, "o"),
                    linewidth=2.5 if c in (3, 5, 9) else 1.5,
                    linestyle="-" if c in (5, 9) else "--",
                )
            ax.set_title(f"Alias Divergence vs. {metric}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Stratification Subset", fontsize=11, fontweight="bold")
            ax.set_ylabel(metric, fontsize=11, fontweight="bold")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, linestyle=":", alpha=0.6)

        plt.suptitle(
            "When Does Entity-Aware Retrieval Earn Its Keep? (With Held-Out Test & Mismatch Gate)",
            fontsize=14, fontweight="bold", y=0.98,
        )
        axes[1].legend(fontsize=8, loc="lower left", framealpha=0.9)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"--> Decisive Plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_stage6_evaluation(
    labeled_qa: List[Dict],
    pipeline: RetrievalPipeline,
    weight_sweep: Optional[Dict] = None,
    gate_sweep: Optional[Dict] = None,
    weight_dev_indices: Optional[List[int]] = None,
    weight_test_indices: Optional[List[int]] = None,
) -> Tuple[Dict, Dict, Dict, Dict, Optional[Dict]]:
    """
    Run the full Stage 6 evaluation.
    Takes pre-computed weight_sweep and gate_sweep from run_pipeline without re-mutating pipeline state.
    Evaluates both Full-bucket held-out test split (N=32) and overall (N=500).
    """
    print("=== Stage 6: Stratified Evaluation & Bootstrap Significance Testing ===")
    evaluator = Evaluator(pipeline)

    if weight_sweep is None:
        sweep_weights = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0]
        weight_sweep = sweep_entity_weight(
            labeled_qa, pipeline, sweep_weights, split_indices=weight_dev_indices
        )

    if gate_sweep is None:
        sweep_thresholds = [6.0, 7.0, 8.0, 9.0, 10.0, 12.0]
        gate_sweep = sweep_gate_threshold(
            labeled_qa, pipeline, sweep_thresholds, dev_indices=weight_dev_indices, test_indices=weight_test_indices
        )

    print(f"--> Using selected entity_weight = {pipeline.entity_weight}")
    print(f"--> Using selected gate_threshold = {pipeline.gate_threshold}")

    # --- Linking accuracy ---
    print("\n--- Entity Linking Accuracy & Candidate-Set Recall ---")
    linking_accuracy = compute_linking_accuracy(labeled_qa, pipeline)
    for b, s in linking_accuracy.items():
        cand_recall = s.get("candidate_set_recall", s.get("ambiguity_ceiling", 0.0))
        print(
            f"  [{b:<25}] 1-best Acc: {s['accuracy_1best']*100:.1f}% | "
            f"Candidate-Set Recall: {cand_recall*100:.1f}% | "
            f"Ambiguous Queries: {s['ambiguous_rate']*100:.1f}%"
        )

    # --- Main evaluation ---
    print("\n--- Results Table (Recall@10 / MRR / nDCG@10) ---")
    table_results, bootstrap_results = evaluator.evaluate_dataset(
        labeled_qa, weight_test_indices=weight_test_indices
    )

    for b, cond_dict in table_results.items():
        print(f"\nSubset: [{b}]")
        print(f"{'Cond':<6} | {'Recall@10':<10} | {'MRR':<10} | {'nDCG@10':<10}")
        print("-" * 46)
        for c, m in cond_dict.items():
            print(f"Cond {c}: {m['Recall@10']:<10.4f} | {m['MRR']:<10.4f} | {m['nDCG@10']:<10.4f}")

    print("\n--- Bootstrap Significance (1000 resamples: Cond5 vs Cond3, Cond9 vs Cond3, Cond8 vs Cond7) ---")
    for b, b_data in bootstrap_results.items():
        bs3 = b_data["vs_cond3"]["MRR"]
        bs1 = b_data["vs_cond1"]["MRR"]
        bs9_3 = b_data.get("cond9_vs_cond3", {}).get("MRR", {})
        sig3 = "SIG" if bs3["significant"] else "ns"
        sig1 = "SIG" if bs1["significant"] else "ns"
        sig9_3 = "SIG" if bs9_3.get("significant", False) else "ns"
        line = (
            f"[{b:<35}] "
            f"vs Cond3: Δ={bs3['mean_delta']:+.4f} CI[{bs3['ci_lower']:+.4f},{bs3['ci_upper']:+.4f}] {sig3}  |  "
            f"vs Cond1: Δ={bs1['mean_delta']:+.4f} CI[{bs1['ci_lower']:+.4f},{bs1['ci_upper']:+.4f}] {sig1}"
        )
        if bs9_3:
            line += f"  |  Cond9 vs 3: Δ={bs9_3['mean_delta']:+.4f} CI[{bs9_3['ci_lower']:+.4f},{bs9_3['ci_upper']:+.4f}] {sig9_3}"
        if "cond8_vs_cond7" in b_data:
            bs87 = b_data["cond8_vs_cond7"]["MRR"]
            sig87 = "SIG" if bs87["significant"] else "ns"
            line += f"  |  Cond8 vs 7: Δ={bs87['mean_delta']:+.4f} CI[{bs87['ci_lower']:+.4f},{bs87['ci_upper']:+.4f}] {sig87}"
        print(line)

    evaluator.generate_decisive_plot(table_results)
    return table_results, bootstrap_results, linking_accuracy, weight_sweep, gate_sweep


if __name__ == "__main__":
    pass
