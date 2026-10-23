import unittest
import json
from src.chem_utils import SharedSemanticEmbedder
from src.stage0_ground_truth import load_alias_dictionary
from src.stage2_entity_linker import EntityLinker
from src.stage3_canonical_graph import CanonicalGraphBuilder, run_stage3_graph_construction
from src.stage5_retriever import RetrievalPipeline

class TestRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("data/corpus.json") as f:
            cls.corpus = json.load(f)[:50]
        cls.alias_dict = load_alias_dictionary("data/alias_dictionary.json")

        cls.embedder = SharedSemanticEmbedder(n_components=16, random_state=42)
        texts = [f"{p.get('title','')} {p.get('text','')}" for p in cls.corpus]
        cls.embedder.fit(texts)

        cls.builder, cls.tagged_corpus = run_stage3_graph_construction(cls.corpus, "data/alias_dictionary.json")
        cls.linker = EntityLinker(
            alias_dict_path="data/alias_dictionary.json",
            embedder=cls.embedder,
            alpha=0.375,
            beta=0.5,
            gamma=0.125,
            tau=0.3
        )
        cls.pipeline = RetrievalPipeline(
            tagged_corpus=cls.tagged_corpus,
            entity_linker=cls.linker,
            graph_builder=cls.builder,
            embedder=cls.embedder
        )

    def test_dense_retrieval(self):
        results = self.pipeline.retrieve_dense("superconducting transition temperature", top_k=5)
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 5)
        if results:
            self.assertIsInstance(results[0][0], str)
            self.assertIsInstance(results[0][1], float)

    def test_bm25_retrieval(self):
        results = self.pipeline.retrieve_bm25("YBCO perovskite structure", top_k=5)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test_entity_id_retrieval(self):
        # Known entity in corpus
        results = self.pipeline.retrieve_entity_id("YBCO superconductor", top_k=5)
        self.assertIsInstance(results, list)

        # Gibberish/unseen query should return [] (no ghost candidate injection)
        nil_results = self.pipeline.retrieve_entity_id("xyzqwerty nonexistent term", top_k=5)
        self.assertEqual(nil_results, [])

    def test_rrf_fusion(self):
        ranking1 = [("P_001", 0.9), ("P_002", 0.8), ("P_003", 0.7)]
        ranking2 = [("P_002", 0.95), ("P_001", 0.85), ("P_004", 0.6)]
        fused = self.pipeline.rrf_fusion([ranking1, ranking2], k=60)
        
        self.assertGreater(len(fused), 0)
        pids = [pid for pid, _ in fused]
        # P_001 and P_002 both appear at top ranks in both lists, so they must lead
        self.assertIn(pids[0], ["P_001", "P_002"])

    def test_all_conditions(self):
        query = "What is the critical temperature of YBCO?"
        for c in [0, 1, 2, 3, 4, 5, 9]:
            res = self.pipeline.retrieve_condition(query, condition=c, top_k=5)
            self.assertIsInstance(res, list)
            self.assertLessEqual(len(res), 5)

if __name__ == '__main__':
    unittest.main()
