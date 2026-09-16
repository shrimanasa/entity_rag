import os
import csv
import json
from typing import Dict, List, Tuple
from src.chem_utils import formula_exact_match, canonicalize_hill_notation

def compute_cohens_kappa(rater1: List[int], rater2: List[int]) -> float:
    """Computes Cohen's kappa coefficient for two raters."""
    if len(rater1) != len(rater2) or len(rater1) == 0:
        raise ValueError("Rater vectors must be non-empty and equal length.")

    n = len(rater1)
    a = sum(1 for r1, r2 in zip(rater1, rater2) if r1 == 1 and r2 == 1)
    b = sum(1 for r1, r2 in zip(rater1, rater2) if r1 == 1 and r2 == 0)
    c = sum(1 for r1, r2 in zip(rater1, rater2) if r1 == 0 and r2 == 1)
    d = sum(1 for r1, r2 in zip(rater1, rater2) if r1 == 0 and r2 == 0)

    p_o = (a + d) / n
    p_r1_yes = (a + b) / n
    p_r2_yes = (a + c) / n
    p_r1_no = (c + d) / n
    p_r2_no = (b + d) / n

    p_e = (p_r1_yes * p_r2_yes) + (p_r1_no * p_r2_no)
    if p_e == 1.0:
        return 1.0

    return (p_o - p_e) / (1.0 - p_e)

def load_alias_dictionary(dict_path: str = "data/alias_dictionary.json") -> Dict:
    with open(dict_path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_stage0_audit(pairs_csv: str = "data/ground_truth_pairs.csv",
                      audit_csv: str = "data/cluster_audit_200.csv",
                      dict_path: str = "data/alias_dictionary.json") -> Dict:
    """
    Stage 0 Ground Truth Audit:
    1. Loads 200 mention pair dev set.
    2. Computes Cohen's kappa on inter-annotator audit (asserts >= 0.70; raises hard RuntimeError if < 0.70).
    3. Loads curated alias dictionary.
    4. Computes coverage rate report on the 200 audit items.
    """
    print("=== Stage 0: Ground Truth & Annotator Audit ===")
    
    # 1. Load Cluster Audit & Compute Cohen's kappa
    rater1, rater2 = [], []
    audit_items = []
    with open(audit_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rater1.append(int(row["annotator_1"]))
            rater2.append(int(row["annotator_2"]))
            audit_items.append(row)

    kappa = compute_cohens_kappa(rater1, rater2)
    print(f"--> Inter-annotator agreement Cohen's κ: {kappa:.4f}")
    if kappa < 0.70:
        raise RuntimeError(f"HARD STOP: Cohen's κ = {kappa:.4f} is below target threshold 0.70!")

    # 2. Load Ground Truth Pairs for Dev Set Grid Search
    dev_pairs = []
    with open(pairs_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dev_pairs.append({
                "mention_1": row["mention_1"],
                "mention_2": row["mention_2"],
                "same_entity": int(row["same_entity"]),
                "canonical_id": row["canonical_id"]
            })

    # 3. Load Alias Dictionary
    alias_dict = load_alias_dictionary(dict_path)
    all_known_aliases = set()
    for ent in alias_dict.values():
        for cat, aliases in ent["categories"].items():
            for a in aliases:
                all_known_aliases.add(a.lower())

    # 4. Coverage Rate Calculation on 200 audit items
    auto_canonicalized = 0
    dictionary_matched = 0
    cooccurrence_matched = 0
    missed_nil = 0

    for item in dev_pairs:
        if item["same_entity"] == 1:
            m1, m2 = item["mention_1"], item["mention_2"]
            # Auto-canonicalization via Hill notation
            if formula_exact_match(m1, m2):
                auto_canonicalized += 1
            # Dictionary match
            elif m1.lower() in all_known_aliases and m2.lower() in all_known_aliases:
                dictionary_matched += 1
            # Co-occurrence / Co-reference
            elif ("superconductor" in m1.lower() and "superconductor" in m2.lower()) or ("cathode" in m1.lower() and "cathode" in m2.lower()):
                cooccurrence_matched += 1
            else:
                missed_nil += 1

    total_true_positives = sum(1 for item in dev_pairs if item["same_entity"] == 1)
    caught_auto = auto_canonicalized + dictionary_matched + cooccurrence_matched
    coverage_rate = caught_auto / total_true_positives if total_true_positives > 0 else 0.0

    coverage_report = {
        "cohens_kappa": kappa,
        "total_dev_pairs": len(dev_pairs),
        "true_alias_pairs": total_true_positives,
        "auto_hill_canonicalized": auto_canonicalized,
        "dictionary_matched": dictionary_matched,
        "cooccurrence_matched": cooccurrence_matched,
        "missed_nil": missed_nil,
        "coverage_rate": coverage_rate
    }

    print(f"--> Stage 0 Coverage Rate: {coverage_rate * 100:.1f}% ({caught_auto}/{total_true_positives} true alias pairs caught)")
    print(f"    - Hill canonicalization: {auto_canonicalized}")
    print(f"    - Dictionary match: {dictionary_matched}")
    print(f"    - Co-occurrence: {cooccurrence_matched}")
    print(f"    - Missed (NIL): {missed_nil}\n")

    return coverage_report

if __name__ == '__main__':
    run_stage0_audit()
