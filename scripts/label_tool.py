import os
import sys
import csv
import json
import argparse
from typing import List, Tuple, Dict

def compute_cohens_kappa(rater1: List[int], rater2: List[int]) -> float:
    """
    Computes Cohen's kappa coefficient for two raters.
    kappa = (p_o - p_e) / (1 - p_e)
    """
    if len(rater1) != len(rater2) or len(rater1) == 0:
        raise ValueError("Rater vectors must be non-empty and equal length.")

    n = len(rater1)
    # Count confusion matrix
    # Categories: 0 and 1
    a = sum(1 for r1, r2 in zip(rater1, rater2) if r1 == 1 and r2 == 1) # Both yes
    b = sum(1 for r1, r2 in zip(rater1, rater2) if r1 == 1 and r2 == 0) # R1 yes, R2 no
    c = sum(1 for r1, r2 in zip(rater1, rater2) if r1 == 0 and r2 == 1) # R1 no, R2 yes
    d = sum(1 for r1, r2 in zip(rater1, rater2) if r1 == 0 and r2 == 0) # Both no

    p_o = (a + d) / n

    p_r1_yes = (a + b) / n
    p_r2_yes = (a + c) / n
    p_r1_no = (c + d) / n
    p_r2_no = (b + d) / n

    p_e = (p_r1_yes * p_r2_yes) + (p_r1_no * p_r2_no)

    if p_e == 1.0:
        return 1.0

    kappa = (p_o - p_e) / (1.0 - p_e)
    return kappa

def run_labeling_cli(output_csv: str):
    print("=== Material Entity Mention Labeling CLI Tool ===")
    print("Label pairs: 1 for Same Canonical Entity, 0 for Different Entities.\n")

    sample_pairs = [
        ("YBCO", "yttrium barium copper oxide", 1),
        ("LiFePO4", "lithium iron phosphate", 1),
        ("MoS2", "molybdenite", 1),
        ("TiO2", "rutile", 1),
        ("YBCO", "LiFePO4", 0),
        ("123 cuprate", "YBCO-123", 1),
        ("triphylite", "LFP cathode", 1),
        ("titania", "anatase", 1),
        ("MAPbI3", "methylammonium lead iodide", 1),
        ("Bi2Te3", "bismuth telluride", 1),
        ("STO", "strontium titanate", 1),
        ("FeSe", "iron selenide", 1),
        ("Ba-122", "BaFe2As2", 1),
        ("Nb3Sn", "niobium-tin alloy", 1),
        ("MoS2", "TiO2", 0),
        ("YBCO", "Bi2Te3", 0),
        ("LiFePO4", "FeSe", 0),
        ("MAPI", "MAPbI3", 1),
        ("A15 superconductor", "Nb3Sn wire", 1),
        ("tausonite", "STO substrate", 1),
    ]

    records = []
    for idx, (m1, m2, default_label) in enumerate(sample_pairs, 1):
        print(f"[{idx}/{len(sample_pairs)}] Pair: '{m1}' vs '{m2}'")
        resp = input(f"  Same Entity? (1/0, default={default_label}): ").strip()
        if resp in ('0', '1'):
            label = int(resp)
        else:
            label = default_label
        records.append({'mention_1': m1, 'mention_2': m2, 'annotator_1': label, 'annotator_2': default_label})

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['mention_1', 'mention_2', 'annotator_1', 'annotator_2'])
        writer.writeheader()
        writer.writerows(records)

    print(f"\nSaved {len(records)} annotations to {output_csv}")

def verify_ground_truth(pairs_csv: str, audit_csv: str) -> float:
    """Loads double annotations and computes Cohen's kappa. Asserts kappa >= 0.7."""
    if not os.path.exists(pairs_csv):
        raise FileNotFoundError(f"Ground truth pairs file not found: {pairs_csv}")
    if not os.path.exists(audit_csv):
        raise FileNotFoundError(f"Cluster audit file not found: {audit_csv}")

    r1, r2 = [], []
    with open(audit_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            r1.append(int(row['annotator_1']))
            r2.append(int(row['annotator_2']))

    kappa = compute_cohens_kappa(r1, r2)
    print(f"--> Inter-annotator agreement Cohen's κ: {kappa:.4f}")

    if kappa < 0.7:
        raise RuntimeError(f"HARD STOP: Cohen's κ = {kappa:.4f} is below target threshold 0.70! Labeling is too noisy.")

    return kappa

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Human Labeling and Cohen's Kappa Verification Tool")
    parser.add_argument('--mode', choices=['interactive', 'verify'], default='verify', help="CLI mode")
    parser.add_argument('--pairs-csv', default='data/ground_truth_pairs.csv', help="Path to ground truth pairs CSV")
    parser.add_argument('--audit-csv', default='data/cluster_audit_200.csv', help="Path to 200-cluster audit CSV")

    args = parser.parse_args()

    if args.mode == 'interactive':
        run_labeling_cli(args.pairs_csv)
    else:
        verify_ground_truth(args.pairs_csv, args.audit_csv)
