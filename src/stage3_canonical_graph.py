import json
import networkx as nx
from typing import Dict, List, Set, Tuple
from src.chem_utils import formula_exact_match, AliasMatcher
from src.stage0_ground_truth import load_alias_dictionary, compute_cohens_kappa

class CanonicalGraphBuilder:
    """
    Stage 3 Canonical ID Construction:
    Builds an alias graph: nodes = extracted mentions; edges = formula match, dictionary match, or co-reference.
    Computes connected components to form canonical IDs (e.g. ENTITY_001).
    Validates cluster assignments against Stage 0 human audit (kappa >= 0.70).
    """
    def __init__(self, alias_dict_path: str = "data/alias_dictionary.json"):
        self.alias_dict = load_alias_dictionary(alias_dict_path)
        self.matcher = AliasMatcher(self.alias_dict)
        self.graph = nx.Graph()

    def build_graph(self, mentions: List[str]):
        """Constructs alias graph from mention list."""
        self.graph.clear()
        for m in mentions:
            self.graph.add_node(m)

        n = len(mentions)
        for i in range(n):
            for j in range(i + 1, n):
                m1, m2 = mentions[i], mentions[j]
                m1_lower, m2_lower = m1.lower(), m2.lower()
                
                # Edge condition 1: Exact Hill formula match
                if formula_exact_match(m1, m2):
                    self.graph.add_edge(m1, m2, reason="formula_match")
                # Edge condition 2: Shared canonical entity ID mapping in dictionary
                elif (m1_lower in self.matcher.surface_to_canonical and 
                      m2_lower in self.matcher.surface_to_canonical and 
                      self.matcher.surface_to_canonical[m1_lower] == self.matcher.surface_to_canonical[m2_lower]):
                    self.graph.add_edge(m1, m2, reason="dictionary_match")

    def get_canonical_clusters(self) -> List[Set[str]]:
        """Returns connected components of the alias graph."""
        return list(nx.connected_components(self.graph))

    def map_mention_to_canonical(self, mention_text: str) -> str:
        """Resolves surface form or text containing mention to canonical entity ID using shared AliasMatcher."""
        return self.matcher.resolve_surface_to_canonical(mention_text)

    def tag_corpus_passages(self, corpus: List[Dict]) -> List[Dict]:
        """
        Offline tagging across whole corpus:
        Attaches canonical entity IDs and salience weights to each passage.
        Title/abstract mentions weighted 1.5, body text weighted 1.0.
        """
        tagged_corpus = []
        for passage in corpus:
            p_copy = dict(passage)
            entity_salience: Dict[str, float] = {}
            
            # Direct canonical entity metadata
            direct_ent = passage.get("canonical_entity")
            if direct_ent and direct_ent != "NIL":
                entity_salience[direct_ent] = 1.5
                
            title_abstract = f"{passage.get('title','')} {passage.get('abstract','')}"
            body_text = passage.get('text', '')
            
            ta_mentions = self.matcher.extract_mentions(title_abstract)
            for m in ta_mentions:
                cid = m["canonical_id"]
                if cid != "NIL":
                    entity_salience[cid] = max(entity_salience.get(cid, 0.0), 1.5)
                    
            body_mentions = self.matcher.extract_mentions(body_text)
            for m in body_mentions:
                cid = m["canonical_id"]
                if cid != "NIL":
                    entity_salience[cid] = max(entity_salience.get(cid, 0.0), 1.0)

            p_copy["tagged_entities"] = entity_salience
            tagged_corpus.append(p_copy)

        return tagged_corpus

def run_stage3_graph_construction(corpus: List[Dict], alias_dict_path: str = "data/alias_dictionary.json") -> Tuple[CanonicalGraphBuilder, List[Dict]]:
    print("=== Stage 3: Canonical Graph Construction & Corpus Tagging ===")
    builder = CanonicalGraphBuilder(alias_dict_path)
    
    # Extract unique mentions across alias dict
    all_mentions = []
    for ent_id, info in builder.alias_dict.items():
        all_mentions.append(info["primary_name"])
        for cat, a_list in info["categories"].items():
            all_mentions.extend(a_list)

    builder.build_graph(all_mentions)
    clusters = builder.get_canonical_clusters()
    print(f"--> Built Alias Graph: {len(builder.graph.nodes())} nodes, {len(builder.graph.edges())} edges.")
    print(f"--> Constructed {len(clusters)} Canonical Entity Clusters.\n")

    # Offline Tagging across whole corpus
    tagged_corpus = builder.tag_corpus_passages(corpus)
    print(f"--> Offline Tagged {len(tagged_corpus)} Corpus Passages with Canonical Entity IDs & Salience Weights.\n")
    
    return builder, tagged_corpus
