import os
import json
from typing import Dict, List, Optional
from src.chem_utils import SharedSemanticEmbedder
from src.stage0_ground_truth import load_alias_dictionary
from src.stage2_entity_linker import EntityLinker
from src.stage3_canonical_graph import CanonicalGraphBuilder, run_stage3_graph_construction
from src.stage5_retriever import RetrievalPipeline
from src.stage7_generator import RAGAnswerGenerator

class EntityAwareRAG:
    """
    High-level Production API for Entity-Aware Materials-Science RAG.
    Connects entity detection, canonical graph linking, multi-channel hybrid retrieval, and answer generation.
    """
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, "data")
        self.data_dir = data_dir

        self.corpus: List[Dict] = []
        self.corpus_dict: Dict[str, Dict] = {}
        self.alias_dict: Dict = {}
        self.embedder: Optional[SharedSemanticEmbedder] = None
        self.linker: Optional[EntityLinker] = None
        self.graph_builder: Optional[CanonicalGraphBuilder] = None
        self.retriever: Optional[RetrievalPipeline] = None
        self.generator = RAGAnswerGenerator()
        self.is_initialized = False

    def initialize(self):
        """Loads data, fits embedder, builds canonical graph, and initializes retrieval pipeline."""
        corpus_file = os.path.join(self.data_dir, "corpus.json")
        alias_file = os.path.join(self.data_dir, "alias_dictionary.json")
        qa_file = os.path.join(self.data_dir, "qa_dataset.json")

        with open(corpus_file, "r") as f:
            self.corpus = json.load(f)
        self.corpus_dict = {p["passage_id"]: p for p in self.corpus}

        self.alias_dict = load_alias_dictionary(alias_file)

        # Load QA for fitting shared embedder universe
        qa_dataset = []
        if os.path.exists(qa_file):
            with open(qa_file, "r") as f:
                qa_dataset = json.load(f)

        # 1. Initialize & fit shared semantic embedder
        self.embedder = SharedSemanticEmbedder(n_components=128, random_state=42)
        embedder_texts = [f"{p.get('title','')} {p.get('abstract','')} {p.get('text','')}" for p in self.corpus]
        embedder_texts += [f"{info['primary_name']} {info.get('categories', {}).get('common_trade_names', [])}" for info in self.alias_dict.values()]
        embedder_texts += [q['query'] for q in qa_dataset]
        self.embedder.fit(embedder_texts)

        # 2. Build canonical graph and tag corpus
        self.graph_builder, tagged_corpus = run_stage3_graph_construction(self.corpus, alias_dict_path=alias_file)

        # 3. Entity linker with optimal tuned parameters
        self.linker = EntityLinker(
            alias_dict_path=alias_file,
            embedder=self.embedder,
            alpha=0.375,
            beta=0.5,
            gamma=0.125,
            tau=0.3
        )

        # 4. Retrieval pipeline
        self.retriever = RetrievalPipeline(
            tagged_corpus=tagged_corpus,
            entity_linker=self.linker,
            graph_builder=self.graph_builder,
            embedder=self.embedder
        )

        self.is_initialized = True
        return self

    def ask(self, query: str, top_k: int = 5) -> Dict:
        """
        Executes end-to-end Entity-Aware RAG on a user query:
        1. Detects entity mentions
        2. Resolves canonical identity
        3. Retrieves evidence passages via Entity-Aware Hybrid fusion
        4. Synthesizes final answer
        """
        if not self.is_initialized:
            self.initialize()

        # Mention detection & resolution
        mentions = self.retriever.detector.extract_mentions(query)
        canonical_id = "NIL"
        confidence = 0.0

        if mentions:
            canonical_id, confidence = self.linker.link_mention(mentions[0])
        if canonical_id == "NIL":
            resolved = self.graph_builder.map_mention_to_canonical(query)
            if resolved != "NIL":
                canonical_id = resolved
                confidence = 0.85

        entity_info = self.alias_dict.get(canonical_id, {})

        # Condition 5: Full Entity-Aware Hybrid System
        c5_scored = self.retriever.retrieve(query, condition=5, top_k=top_k)
        c5_passages = [self.corpus_dict[pid] for pid, _ in c5_scored if pid in self.corpus_dict]

        # Condition 3: Standard BM25 + Dense Baseline (for comparison)
        c3_scored = self.retriever.retrieve(query, condition=3, top_k=top_k)
        c3_passages = [self.corpus_dict[pid] for pid, _ in c3_scored if pid in self.corpus_dict]

        # Generate Answers
        gen_c5 = self.generator.generate(query, c5_passages, max_passages=3)
        gen_c3 = self.generator.generate(query, c3_passages, max_passages=3)

        return {
            "query": query,
            "detected_mentions": [m["surface_form"] for m in mentions],
            "canonical_entity_id": canonical_id,
            "canonical_primary_name": entity_info.get("primary_name", "Unknown"),
            "known_aliases": entity_info.get("categories", {}),
            "linking_confidence": round(confidence, 4),
            "answer": gen_c5["answer"],
            "cited_passages": [p["passage_id"] for p in c5_passages[:3]],
            "baseline_answer": gen_c3["answer"],
            "baseline_cited_passages": [p["passage_id"] for p in c3_passages[:3]],
            "top_passages": c5_passages
        }
