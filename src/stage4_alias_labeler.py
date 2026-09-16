import json
from typing import Dict, List, Tuple
from src.chem_utils import calculate_lexical_divergence
from src.stage3_canonical_graph import CanonicalGraphBuilder

class AliasDivergenceLabeler:
    """
    Stage 4 Alias-Divergence Stratification:
    Evaluates lexical divergence between query and evidence passage:
      - No-mismatch: lex_sim >= 0.70
      - Partial-mismatch: 0.20 <= lex_sim < 0.70
      - Full alias-mismatch: lex_sim < 0.20 AND same canonical entity
    """
    def __init__(self, graph_builder: CanonicalGraphBuilder):
        self.builder = graph_builder

    def label_qa_pair(self, qa_pair: Dict, corpus_dict: Dict[str, Dict]) -> Dict:
        query_text = qa_pair["query"]
        ev_id = qa_pair["evidence_passage_id"]
        ev_passage = corpus_dict.get(ev_id, {})
        ev_text = f"{ev_passage.get('title','')} {ev_passage.get('abstract','')} {ev_passage.get('text','')}"

        # Lexical containment similarity
        lex_sim = calculate_lexical_divergence(query_text, ev_text)

        # Gold ground truth canonical entity
        gold_canonical = qa_pair.get("canonical_entity", "NIL")
        ev_canonical = ev_passage.get("canonical_entity", "NIL")
        same_gold_canonical = (gold_canonical != "NIL" and gold_canonical == ev_canonical)

        # System-detected entity from query string alone
        system_query_canonical = self.builder.map_mention_to_canonical(query_text)
        system_match = (system_query_canonical != "NIL" and system_query_canonical == ev_canonical)

        # Ground truth evaluation bucketing
        if lex_sim >= 0.70:
            bucket = "No-mismatch"
        elif 0.20 <= lex_sim < 0.70:
            bucket = "Partial-mismatch"
        else: # lex_sim < 0.20
            if same_gold_canonical:
                bucket = "Full alias-mismatch"
            else:
                bucket = "Partial-mismatch"

        labeled = dict(qa_pair)
        labeled["lexical_similarity"] = round(lex_sim, 4)
        labeled["gold_canonical_id"] = gold_canonical
        labeled["system_query_canonical_id"] = system_query_canonical
        labeled["evidence_canonical_id"] = ev_canonical
        labeled["same_canonical_id"] = same_gold_canonical
        labeled["system_detected_match"] = system_match
        labeled["bucket"] = bucket
        return labeled

    def label_qa_dataset(self, qa_dataset: List[Dict], corpus: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
        corpus_dict = {p["passage_id"]: p for p in corpus}
        labeled_dataset = []
        bucket_counts = {"No-mismatch": 0, "Partial-mismatch": 0, "Full alias-mismatch": 0}

        for qa in qa_dataset:
            labeled = self.label_qa_pair(qa, corpus_dict)
            labeled_dataset.append(labeled)
            bucket_counts[labeled["bucket"]] += 1

        return labeled_dataset, bucket_counts

def run_stage4_divergence_labeling(qa_dataset: List[Dict], corpus: List[Dict], graph_builder: CanonicalGraphBuilder) -> Tuple[List[Dict], Dict[str, int]]:
    print("=== Stage 4: Alias-Divergence QA Pair Labeling ===")
    labeler = AliasDivergenceLabeler(graph_builder)
    labeled_dataset, bucket_counts = labeler.label_qa_dataset(qa_dataset, corpus)

    system_detected_count = sum(1 for item in labeled_dataset if item.get("system_detected_match", False))
    print(f"--> Stratified {len(labeled_dataset)} QA Pairs into 3 Divergence Buckets:")
    print(f"    - No-mismatch (lex_sim ≥ 0.70): {bucket_counts['No-mismatch']}")
    print(f"    - Partial-mismatch (0.20 ≤ lex_sim < 0.70): {bucket_counts['Partial-mismatch']}")
    print(f"    - Full alias-mismatch (lex_sim < 0.20 & same canonical ID): {bucket_counts['Full alias-mismatch']}")
    print(f"    - System Query-Mention Detection Match Rate: {system_detected_count / len(labeled_dataset) * 100:.1f}%\n")

    return labeled_dataset, bucket_counts
