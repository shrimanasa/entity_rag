import os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "scratch", "matplotlib"))
import csv
import json
from typing import Dict, List, Tuple
import numpy as np
from src.chem_utils import SharedSemanticEmbedder
from src.stage0_ground_truth import run_stage0_audit, load_alias_dictionary
from src.stage2_entity_linker import EntityLinker, grid_search_hyperparameters
from src.stage3_canonical_graph import run_stage3_graph_construction
from src.stage4_alias_labeler import run_stage4_divergence_labeling
from src.stage5_retriever import RetrievalPipeline
from src.stage6_evaluator import run_stage6_evaluation
from src.stage7_evaluator import Stage7Evaluator

import argparse

from src.report_generator import (
    format_findings_report_md,
    generate_metrics_table_md,
    format_qa_results_markdown,
    sync_readme,
)

def run_end_to_end_pipeline(skip_neural: bool = False):
    os.makedirs("results", exist_ok=True)
    print("================================================================================")
    print("      RESEARCH SYSTEM: WHEN DOES ENTITY-AWARE RETRIEVAL EARN ITS KEEP?         ")
    print("================================================================================\n")

    # 1. Load Data
    with open("data/corpus.json", "r", encoding="utf-8") as f:
        corpus = json.load(f)
    corpus_dict = {p["passage_id"]: p for p in corpus}

    with open("data/qa_dataset.json", "r", encoding="utf-8") as f:
        qa_dataset = json.load(f)
    alias_dict = load_alias_dictionary("data/alias_dictionary.json")

    dev_pairs = []
    with open("data/ground_truth_pairs.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dev_pairs.append({
                "mention_1": row["mention_1"],
                "mention_2": row["mention_2"],
                "same_entity": int(row["same_entity"]),
                "canonical_id": row["canonical_id"]
            })

    # 2. Fit Shared Semantic Embedder (TruncatedSVD / LSA) Inductively on Corpus & Profiles
    # (Strictly zero evaluation query text to eliminate transductive leakage)
    print("=== Initializing Shared Semantic Embedder (LSA / TruncatedSVD) ===")
    training_texts = []
    for p in corpus:
        training_texts.append(f"{p.get('title','')} {p.get('abstract','')} {p.get('text','')}")
    for ent_id, info in alias_dict.items():
        all_aliases = []
        for cat, names in info.get("categories", {}).items():
            all_aliases.extend(names)
        training_texts.append(f"{info.get('primary_name','')} {' '.join(all_aliases)} synthesis property crystal bandgap superconductor cathode catalyst")

    shared_embedder = SharedSemanticEmbedder(n_components=128, random_state=42)
    shared_embedder.fit(training_texts)
    print(f"--> Fitted SharedSemanticEmbedder (n_components=128, random_state=42) on {len(training_texts)} text units (inductive).\n")

    # 3. Stage 0 Coverage Audit & Cohen's Kappa
    cov_report = run_stage0_audit(
        audit_csv="data/cluster_audit_200.csv",
        dict_path="data/alias_dictionary.json"
    )

    # 4. Stage 2 Grid Search using Shared Embedder
    best_grid_params = grid_search_hyperparameters(
        dev_pairs=dev_pairs,
        embedder=shared_embedder,
        alias_dict_path="data/alias_dictionary.json"
    )

    # 5. Stage 3 Canonical Graph Construction & Corpus Tagging
    builder, tagged_corpus = run_stage3_graph_construction(corpus, alias_dict_path="data/alias_dictionary.json")

    # 6. Stage 4 Alias-Divergence QA Pair Labeling
    labeled_qa, bucket_counts = run_stage4_divergence_labeling(qa_dataset, corpus, builder)

    # 7. Initialize Entity Linker & Retrieval Pipeline with Shared Embedder
    linker = EntityLinker(
        alias_dict_path="data/alias_dictionary.json",
        embedder=shared_embedder,
        alpha=best_grid_params["alpha"],
        beta=best_grid_params["beta"],
        gamma=best_grid_params["gamma"],
        tau=best_grid_params["tau"],
        include_corpus_passages=False,  # Leak-free: no evaluation evidence text in entity profiles
    )

    # Optional real neural dense embedder (SentenceTransformer all-MiniLM-L6-v2)
    neural_model = None
    if not skip_neural:
        try:
            from sentence_transformers import SentenceTransformer
            print("\n=== Initializing Neural Dense Embedder (SentenceTransformer all-MiniLM-L6-v2) ===")
            neural_model = SentenceTransformer("all-MiniLM-L6-v2", revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
            print("--> Loaded all-MiniLM-L6-v2 neural embedder successfully.\n")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SentenceTransformer('all-MiniLM-L6-v2'): {e}.\n"
                "To run the pipeline strictly offline or without neural dependencies, pass --skip-neural."
            ) from e
    else:
        print("\n--> --skip-neural flag passed: skipping Neural Dense conditions (6, 7, 8).\n")

    # Held-Out Weight Selection:
    # Full alias-mismatch dev split = even indices (N=33)
    # Full alias-mismatch test split = odd indices (N=32)
    # Overall dev split = even indices (N=250)
    full_bucket_qa = [qa for qa in labeled_qa if qa["bucket"] == "Full alias-mismatch"]
    weight_dev_idx  = [i for i in range(len(full_bucket_qa)) if i % 2 == 0]
    weight_test_idx = [i for i in range(len(full_bucket_qa)) if i % 2 == 1]
    sweep_weights   = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]

    pipeline = RetrievalPipeline(
        tagged_corpus=tagged_corpus,
        entity_linker=linker,
        graph_builder=builder,
        embedder=shared_embedder,
        neural_embedder=neural_model,
        entity_weight=1.0,
    )

    from src.stage6_evaluator import sweep_entity_weight, sweep_gate_threshold
    print("\n=== Held-Out Entity Weight Selection ===")
    weight_sweep_data = sweep_entity_weight(labeled_qa, pipeline, sweep_weights, split_indices=weight_dev_idx, test_indices=weight_test_idx)
    dev_sweep_full = weight_sweep_data["full_mismatch_dev"]
    dev_sweep_overall = weight_sweep_data["overall_dev"]

    # Select entity weight on Overall Dev MRR, since bucket is unknown at query time
    best_weight = max(dev_sweep_overall, key=dev_sweep_overall.get)
    pipeline.entity_weight = best_weight
    print(f"--> Candidate weights swept: {sweep_weights}")
    print(f"--> Dev Full-mismatch MRR: {dev_sweep_full}")
    print(f"--> Dev Overall MRR:       {dev_sweep_overall}")
    print(f"--> Selected entity_weight = {best_weight} (Dev Overall MRR = {dev_sweep_overall[best_weight]:.4f}, Dev Full MRR = {dev_sweep_full[best_weight]:.4f})\n")

    # Mismatch Gate Calibration (Stage 5b)
    print("\n=== Stage 5b: Mismatch Gate Calibration ===")
    sweep_thresholds = [6.0, 7.0, 8.0, 9.0, 10.0, 12.0]
    gate_sweep_data = sweep_gate_threshold(
        labeled_qa, pipeline, sweep_thresholds, dev_indices=weight_dev_idx, test_indices=weight_test_idx
    )
    best_gate_th = gate_sweep_data["best_threshold"]
    pipeline.gate_threshold = best_gate_th
    print(f"--> Candidate gate thresholds swept: {sweep_thresholds}")
    print(f"--> Dev Gate Sweep: {gate_sweep_data['dev_sweep']}")
    print(f"--> Selected gate_threshold = {best_gate_th}\n")

    # 8. Stage 6 Evaluation & Significance Testing
    table_results, bootstrap_results, linking_accuracy, weight_sweep, gate_sweep = run_stage6_evaluation(
        labeled_qa,
        pipeline,
        weight_sweep=weight_sweep_data,
        gate_sweep=gate_sweep_data,
        weight_dev_indices=weight_dev_idx,
        weight_test_indices=weight_test_idx,
    )

    # 9. Stage 7 Downstream QA Answer Evaluation
    print("\n=== Stage 7: Downstream QA Generation & Answer Evaluation ===")
    stage7 = Stage7Evaluator(corpus_dict=corpus_dict)
    qa_results = stage7.evaluate_pipeline(labeled_qa, pipeline)

    print("\n--- STAGE 7 QA EVALUATION RESULTS ---")
    print(f"{'Bucket':<25} | {'Count':<6} | {'Baseline F1':<12} | {'Entity RAG F1':<14} | {'F1 Δ':<8} | {'Significant'}")
    print("-" * 80)
    for b, stats in qa_results["summary"].items():
        sig_str = "YES (p < 0.05)" if stats["significant"] else "NO"
        print(f"{b:<25} | {stats['count']:<6} | {stats['baseline_f1']:<12.4f} | {stats['full_system_f1']:<14.4f} | {stats['f1_delta']:<+8.4f} | {sig_str}")

    # 10. Stage 8 Held-Out Alias Split Experiment (Open-World Evaluation)
    print("\n=== Stage 8: Held-Out Alias Split Experiment (Open-World Evaluation) ===")
    from scripts.held_out_alias_experiment import run_held_out_alias_experiment
    held_out_report = run_held_out_alias_experiment()

    # 11. Save Artifacts & Reports
    with open("results/coverage_report.json", "w", encoding="utf-8") as f:
        json.dump(cov_report, f, indent=2)

    with open("results/grid_search_results.json", "w", encoding="utf-8") as f:
        json.dump(best_grid_params, f, indent=2)

    with open("results/linking_accuracy.json", "w", encoding="utf-8") as f:
        json.dump(linking_accuracy, f, indent=2)

    with open("results/weight_sweep.json", "w", encoding="utf-8") as f:
        # Convert float keys to strings for JSON compatibility
        json.dump({str(k): v for k, v in weight_sweep.items()}, f, indent=2)

    with open("results/gate_sweep.json", "w", encoding="utf-8") as f:
        json.dump(gate_sweep_data, f, indent=2)

    with open("results/results_table.json", "w", encoding="utf-8") as f:
        json.dump(table_results, f, indent=2)

    with open("results/qa_results.json", "w", encoding="utf-8") as f:
        json.dump(qa_results, f, indent=2)

    markdown_report = format_findings_report_md(
        table_results=table_results,
        bootstrap_results=bootstrap_results,
        coverage_report=cov_report,
        grid_params=best_grid_params,
        linking_accuracy=linking_accuracy,
        weight_sweep=weight_sweep,
        gate_sweep=gate_sweep_data,
        held_out_alias_report=held_out_report,
        qa_results=qa_results,
    )
    with open("results/findings_report.md", "w", encoding="utf-8") as f:
        f.write(markdown_report)

    # Synchronize README.md dynamically with exact numbers
    sync_readme(
        readme_path="README.md",
        table_results=table_results,
        bootstrap_results=bootstrap_results,
        weight_sweep=weight_sweep,
        gate_sweep=gate_sweep_data,
        held_out_alias_report=held_out_report,
        linking_accuracy=linking_accuracy,
        qa_results=qa_results,
    )

    # results_table.md: metrics table only (findings_report.md has the full write-up)
    table_only_md = "# Stage 5–6: Retrieval Metrics Table\n\n" + generate_metrics_table_md(table_results)
    with open("results/results_table.md", "w", encoding="utf-8") as f:
        f.write(table_only_md)

    qa_report_md = "# Stage 7: Downstream Question Answering Evaluation\n\n" + format_qa_results_markdown(qa_results)
    with open("results/qa_evaluation_report.md", "w", encoding="utf-8") as f:
        f.write(qa_report_md)

    print("\n================================================================================")
    print("                     PIPELINE EXECUTION COMPLETE!                              ")
    print("Deliverables generated:")
    print("  - Stage 0 Coverage Report:   results/coverage_report.json")
    print("  - Stage 2 Grid Search:       results/grid_search_results.json")
    print("  - Entity Linking Accuracy:   results/linking_accuracy.json")
    print("  - Entity Weight Sweep:       results/weight_sweep.json")
    print("  - Mismatch Gate Sweep:       results/gate_sweep.json")
    print("  - Stage 5-6 Results Table:   results/results_table.json & results/results_table.md")
    print("  - Stage 7 QA Evaluation:     results/qa_results.json & results/qa_evaluation_report.md")
    print("  - Decisive Figure:           results/decisive_plot.png")
    print("  - Final Write-up:            results/findings_report.md")
    print("  - Synchronized README:       README.md")
    print("================================================================================")

def parse_args():
    parser = argparse.ArgumentParser(description="Run Entity-Aware RAG benchmark pipeline")
    parser.add_argument("--skip-neural", action="store_true", help="Skip neural dense embedder (Conditions 6, 7, 8)")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run_end_to_end_pipeline(skip_neural=args.skip_neural)
