"""
Held-Out Alias Split Experiment (Open-World / Incomplete Dictionary Evaluation).

Audits H1 under realistic conditions where queries contain aliases absent
from the curated knowledge graph dictionary (out-of-dictionary / OOD surface forms).
Withholds a random 50% of aliases across entities (preserving primary names),
trains the LSA embedder strictly on the retained vocabulary without leak,
uses mention detector extraction for linking, and reports bootstrap CIs vs Cond 1 & Cond 3.
"""

import os
import sys
import copy
import json
import random
import numpy as np
from typing import Dict, List, Tuple

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stage5_retriever import RetrievalPipeline
from src.stage2_entity_linker import EntityLinker
from src.stage3_canonical_graph import run_stage3_graph_construction
from src.stage4_alias_labeler import run_stage4_divergence_labeling
from src.chem_utils import SharedSemanticEmbedder
from src.stage6_evaluator import (
    compute_mrr,
    compute_recall_at_k,
    compute_ndcg_at_k,
    paired_bootstrap_significance_test,
)


def run_held_out_alias_experiment(
    corpus_path: str = "data/corpus.json",
    qa_path: str = "data/qa_dataset.json",
    alias_dict_path: str = "data/alias_dictionary.json",
    output_path: str = "results/held_out_alias_experiment.json",
) -> Dict:
    print("================================================================================")
    print("            HELD-OUT ALIAS SPLIT EXPERIMENT (OPEN-WORLD EVALUATION)             ")
    print("================================================================================\n")

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_dataset = json.load(f)
    with open(alias_dict_path, "r", encoding="utf-8") as f:
        full_alias_dict = json.load(f)

    # 1. Closed-World Inductive LSA Embedder (full dictionary, zero query text)
    full_train_texts = [f"{p.get('title','')} {p.get('abstract','')} {p.get('text','')}" for p in corpus]
    for ent_id, info in full_alias_dict.items():
        all_aliases = []
        for cat, names in info.get("categories", {}).items():
            all_aliases.extend(names)
        full_train_texts.append(
            f"{info.get('primary_name','')} {' '.join(all_aliases)} synthesis property crystal bandgap superconductor cathode catalyst"
        )
    embedder_full = SharedSemanticEmbedder(n_components=128, random_state=42)
    embedder_full.fit(full_train_texts)

    builder_full, tagged_corpus_full = run_stage3_graph_construction(corpus, alias_dict_path=alias_dict_path)
    labeled_qa, _ = run_stage4_divergence_labeling(qa_dataset, corpus, builder_full)
    linker_full = EntityLinker(
        alias_dict_path=alias_dict_path,
        embedder=embedder_full,
        alpha=0.5, beta=0.3333, gamma=0.1667, tau=0.3,
        include_corpus_passages=False,
    )
    pipeline_full = RetrievalPipeline(
        tagged_corpus=tagged_corpus_full,
        entity_linker=linker_full,
        graph_builder=builder_full,
        embedder=embedder_full,
        entity_weight=0.5,
    )

    # 2. Build Incomplete Dictionary: True Random 50% Withheld Across Non-Primary Aliases
    rng = random.Random(42)
    held_out_dict = copy.deepcopy(full_alias_dict)
    held_out_aliases_set = set()
    total_non_primary = 0
    withheld_count = 0

    for ent_id, info in held_out_dict.items():
        cats = info.get("categories", {})
        for cat_name, a_list in list(cats.items()):
            if not a_list:
                continue
            total_non_primary += len(a_list)
            shuffled = list(a_list)
            rng.shuffle(shuffled)
            k_withhold = max(1, int(round(len(shuffled) * 0.5))) if len(shuffled) > 1 else (1 if rng.random() < 0.5 else 0)
            withheld = shuffled[:k_withhold]
            retained = shuffled[k_withhold:]
            for a in withheld:
                held_out_aliases_set.add(a.lower().strip())
            cats[cat_name] = retained
            withheld_count += len(withheld)

    print(f"--> Withheld {withheld_count}/{total_non_primary} aliases ({withheld_count/total_non_primary*100:.1f}%) from dictionary.")

    tmp_dict_path = "/tmp/held_out_alias_dict.json"
    with open(tmp_dict_path, "w", encoding="utf-8") as f:
        json.dump(held_out_dict, f, indent=2)

    # Inductive LSA embedder fit STRICTLY on retained dictionary (zero withheld aliases, zero queries)
    held_train_texts = [f"{p.get('title','')} {p.get('abstract','')} {p.get('text','')}" for p in corpus]
    for ent_id, info in held_out_dict.items():
        all_aliases = []
        for cat, names in info.get("categories", {}).items():
            all_aliases.extend(names)
        held_train_texts.append(
            f"{info.get('primary_name','')} {' '.join(all_aliases)} synthesis property crystal bandgap superconductor cathode catalyst"
        )
    embedder_held = SharedSemanticEmbedder(n_components=128, random_state=42)
    embedder_held.fit(held_train_texts)

    builder_held, tagged_corpus_held = run_stage3_graph_construction(corpus, alias_dict_path=tmp_dict_path)
    linker_held = EntityLinker(
        alias_dict_path=tmp_dict_path,
        embedder=embedder_held,
        alpha=0.5, beta=0.3333, gamma=0.1667, tau=0.3,
        include_corpus_passages=False,
    )
    pipeline_held = RetrievalPipeline(
        tagged_corpus=tagged_corpus_held,
        entity_linker=linker_held,
        graph_builder=builder_held,
        embedder=embedder_held,
        entity_weight=0.5,
    )

    # Stratify queries: check if extracted query mention matches any held-out alias
    held_out_qa = []
    in_dict_qa = []
    for q in labeled_qa:
        mentions = pipeline_full.detector.extract_mentions(q["query"])
        contains_withheld = False
        if mentions:
            for m in mentions:
                if m["surface_form"].lower().strip() in held_out_aliases_set:
                    contains_withheld = True
                    break
        else:
            q_low = q["query"].lower()
            if any(a in q_low for a in held_out_aliases_set):
                contains_withheld = True
        if contains_withheld:
            held_out_qa.append(q)
        else:
            in_dict_qa.append(q)

    # 3. Evaluate Linking Accuracy using the Detector's Extracted Mention
    def eval_linking(pipeline_obj, query_list):
        correct = 0
        nil = 0
        wrong = 0
        for q in query_list:
            mentions = pipeline_obj.detector.extract_mentions(q["query"])
            if mentions:
                res, conf = pipeline_obj.linker.link_mention(mentions[0])
            else:
                mapped = pipeline_obj.builder.map_mention_to_canonical(q["query"])
                res, conf = (mapped, 0.85) if mapped != "NIL" else ("NIL", 0.0)
            gt = q.get("canonical_entity", "")
            if conf >= pipeline_obj.linker.tau and res == gt:
                correct += 1
            elif res == "NIL" or conf < pipeline_obj.linker.tau:
                nil += 1
            else:
                wrong += 1
        n = len(query_list)
        return {
            "total": n,
            "accuracy": round(correct / n, 4) if n else 0.0,
            "nil_rate": round(nil / n, 4) if n else 0.0,
            "wrong_rate": round(wrong / n, 4) if n else 0.0,
        }

    linking_results = {
        "queries_with_in_dict_aliases": {
            "full_dict": eval_linking(pipeline_full, in_dict_qa),
            "held_out_dict": eval_linking(pipeline_held, in_dict_qa),
        },
        "queries_with_held_out_aliases": {
            "full_dict": eval_linking(pipeline_full, held_out_qa),
            "held_out_dict": eval_linking(pipeline_held, held_out_qa),
        },
    }

    # 4. Evaluate Retrieval on Full-Mismatch Queries
    full_mismatch_held = [q for q in held_out_qa if q["bucket"] == "Full alias-mismatch"]
    full_mismatch_in = [q for q in in_dict_qa if q["bucket"] == "Full alias-mismatch"]

    def eval_retrieval(pipeline_obj, query_list, cond):
        mrrs = []
        recalls = []
        for q in query_list:
            r = [x[0] for x in pipeline_obj.retrieve_condition(q["query"], condition=cond, top_k=10)]
            gt = q["evidence_passage_id"]
            mrrs.append(compute_mrr(r, gt))
            recalls.append(compute_recall_at_k(r, gt))
        return {
            "MRR": round(float(np.mean(mrrs)), 4) if mrrs else 0.0,
            "Recall@10": round(float(np.mean(recalls)), 4) if recalls else 0.0,
            "raw_mrrs": mrrs,
            "raw_recalls": recalls,
        }

    # Retrieval stats on held-out full-mismatch
    c1_held = eval_retrieval(pipeline_held, full_mismatch_held, 1)
    c3_held = eval_retrieval(pipeline_held, full_mismatch_held, 3)
    c4_full = eval_retrieval(pipeline_full, full_mismatch_held, 4)
    c4_held = eval_retrieval(pipeline_held, full_mismatch_held, 4)
    c5_full = eval_retrieval(pipeline_full, full_mismatch_held, 5)
    c5_held = eval_retrieval(pipeline_held, full_mismatch_held, 5)

    # Retrieval stats on in-dictionary control full-mismatch
    c1_in = eval_retrieval(pipeline_held, full_mismatch_in, 1)
    c3_in = eval_retrieval(pipeline_held, full_mismatch_in, 3)
    c4_in_held = eval_retrieval(pipeline_held, full_mismatch_in, 4)
    c5_in_held = eval_retrieval(pipeline_held, full_mismatch_in, 5)

    # Bootstrap significance for held-out full-mismatch queries
    bs_held_c5_vs_c3 = paired_bootstrap_significance_test(c5_held["raw_mrrs"], c3_held["raw_mrrs"])
    bs_held_c5_vs_c1 = paired_bootstrap_significance_test(c5_held["raw_mrrs"], c1_held["raw_mrrs"])

    retrieval_results = {
        "full_mismatch_held_out_queries": {
            "count": len(full_mismatch_held),
            "cond1_dense_lsa": {"MRR": c1_held["MRR"], "Recall@10": c1_held["Recall@10"]},
            "cond3_baseline": {"MRR": c3_held["MRR"], "Recall@10": c3_held["Recall@10"]},
            "cond4_entity_only_full_dict": {"MRR": c4_full["MRR"], "Recall@10": c4_full["Recall@10"]},
            "cond4_entity_only_held_dict": {"MRR": c4_held["MRR"], "Recall@10": c4_held["Recall@10"]},
            "cond5_fusion_full_dict": {"MRR": c5_full["MRR"], "Recall@10": c5_full["Recall@10"]},
            "cond5_fusion_held_dict": {"MRR": c5_held["MRR"], "Recall@10": c5_held["Recall@10"]},
            "bootstrap_c5_vs_c3_mrr": bs_held_c5_vs_c3,
            "bootstrap_c5_vs_c1_mrr": bs_held_c5_vs_c1,
        },
        "full_mismatch_in_dict_control_queries": {
            "count": len(full_mismatch_in),
            "cond1_dense_lsa": {"MRR": c1_in["MRR"], "Recall@10": c1_in["Recall@10"]},
            "cond3_baseline": {"MRR": c3_in["MRR"], "Recall@10": c3_in["Recall@10"]},
            "cond4_entity_only_held_dict": {"MRR": c4_in_held["MRR"], "Recall@10": c4_in_held["Recall@10"]},
            "cond5_fusion_held_dict": {"MRR": c5_in_held["MRR"], "Recall@10": c5_in_held["Recall@10"]},
        },
    }

    report = {
        "summary": {
            "total_queries": len(labeled_qa),
            "queries_in_dict": len(in_dict_qa),
            "queries_held_out": len(held_out_qa),
            "full_mismatch_held_out_count": len(full_mismatch_held),
            "full_mismatch_in_dict_count": len(full_mismatch_in),
            "aliases_withheld_count": withheld_count,
            "total_aliases_count": total_non_primary,
            "alias_withhold_fraction": round(withheld_count / total_non_primary, 4) if total_non_primary else 0.5,
        },
        "linking_evaluation": linking_results,
        "retrieval_evaluation": retrieval_results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print Clean Summary
    print(f"Total benchmark queries: {len(labeled_qa)}")
    print(f"  • Queries with fully known aliases:    {len(in_dict_qa)} ({len(in_dict_qa)/len(labeled_qa)*100:.1f}%)")
    print(f"  • Queries with held-out novel aliases: {len(held_out_qa)} ({len(held_out_qa)/len(labeled_qa)*100:.1f}%)")
    print(f"  • Full-mismatch split: {len(full_mismatch_held)} novel vs. {len(full_mismatch_in)} known aliases")

    print("\n--- 1. Entity Linking Accuracy ---")
    in_res = linking_results['queries_with_in_dict_aliases']['held_out_dict']
    held_res = linking_results['queries_with_held_out_aliases']['held_out_dict']
    print(f"  • In-Dictionary Queries:  Accuracy={in_res['accuracy']*100:.1f}%, NIL={in_res['nil_rate']*100:.1f}%, Wrong={in_res['wrong_rate']*100:.1f}%")
    print(f"  • Novel Alias Queries:   Accuracy={held_res['accuracy']*100:.1f}%, NIL={held_res['nil_rate']*100:.1f}%, Wrong={held_res['wrong_rate']*100:.1f}%")

    print(f"\n--- 2. Full-Mismatch Retrieval on Novel Aliases (N = {len(full_mismatch_held)}) ---")
    print(f"  • Cond 1 Dense LSA:                  MRR = {c1_held['MRR']:.4f}")
    print(f"  • Cond 3 Baseline (BM25 + Dense):    MRR = {c3_held['MRR']:.4f}")
    print(f"  • Cond 4 Entity-only (Held Dict):    MRR = {c4_held['MRR']:.4f}")
    print(f"  • Cond 5 Fusion (Held Dict):         MRR = {c5_held['MRR']:.4f}")
    print(f"  • Δ vs. Cond 3: {bs_held_c5_vs_c3['mean_delta']:+.4f} (95% CI: [{bs_held_c5_vs_c3['ci_lower']:+.4f}, {bs_held_c5_vs_c3['ci_upper']:+.4f}])")
    print(f"  • Δ vs. Cond 1: {bs_held_c5_vs_c1['mean_delta']:+.4f} (95% CI: [{bs_held_c5_vs_c1['ci_lower']:+.4f}, {bs_held_c5_vs_c1['ci_upper']:+.4f}])")
    print("================================================================================")
    print(f"Deliverable saved to: {output_path}")
    print("================================================================================")

    return report


if __name__ == "__main__":
    run_held_out_alias_experiment()
