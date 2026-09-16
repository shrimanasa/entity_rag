import unittest
import os
import json
from src.stage0_ground_truth import compute_cohens_kappa, load_alias_dictionary, run_stage0_audit
from src.stage1_mention_detection import MentionDetector
from src.stage2_entity_linker import EntityLinker
from src.stage3_canonical_graph import CanonicalGraphBuilder, run_stage3_graph_construction
from src.stage4_alias_labeler import run_stage4_divergence_labeling
from src.chem_utils import SharedSemanticEmbedder

class TestEntityPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.alias_dict_path = "data/alias_dictionary.json"
        cls.corpus_path = "data/corpus.json"
        cls.qa_path = "data/qa_dataset.json"

        with open(cls.corpus_path) as f:
            cls.corpus = json.load(f)[:20]  # small subset for rapid tests
        with open(cls.qa_path) as f:
            cls.qa_dataset = json.load(f)[:20]

        cls.alias_dict = load_alias_dictionary(cls.alias_dict_path)
        cls.embedder = SharedSemanticEmbedder(n_components=8, random_state=42)
        sample_texts = [p.get("text", "") for p in cls.corpus] + [q["query"] for q in cls.qa_dataset]
        cls.embedder.fit(sample_texts)

    def test_cohens_kappa(self):
        r1 = [1, 1, 0, 0, 1]
        r2 = [1, 1, 0, 0, 1]
        self.assertAlmostEqual(compute_cohens_kappa(r1, r2), 1.0)

        r3 = [1, 0, 1, 0]
        r4 = [0, 1, 0, 1]
        self.assertLess(compute_cohens_kappa(r3, r4), 0.0)

    def test_mention_detector(self):
        detector = MentionDetector(alias_dict=self.alias_dict, embedder=self.embedder)
        mentions = detector.extract_mentions("Characterization of YBCO under extreme conditions.")
        self.assertGreater(len(mentions), 0)
        self.assertIn("YBCO", [m["surface_form"] for m in mentions])

    def test_entity_linker(self):
        linker = EntityLinker(
            alias_dict_path=self.alias_dict_path,
            embedder=self.embedder,
            alpha=0.375,
            beta=0.5,
            gamma=0.125,
            tau=0.3
        )
        cid, conf = linker.link_mention({"surface_form": "YBCO", "context_window": "superconductor transition temperature"})
        self.assertEqual(cid, "ENTITY_001")
        self.assertGreater(conf, 0.3)

    def test_canonical_graph_builder(self):
        builder = CanonicalGraphBuilder(self.alias_dict_path)
        builder.build_graph(["YBCO", "123 cuprate", "YBa2Cu3O7", "LiFePO4"])
        clusters = builder.get_canonical_clusters()
        
        # YBCO, 123 cuprate, and YBa2Cu3O7 belong to the same canonical cluster
        ybco_cluster = None
        for c in clusters:
            if "YBCO" in c:
                ybco_cluster = c
                break
        self.assertIsNotNone(ybco_cluster)
        self.assertIn("YBa2Cu3O7", ybco_cluster)
        self.assertNotIn("LiFePO4", ybco_cluster)

    def test_stage4_divergence_labeling(self):
        builder = CanonicalGraphBuilder(self.alias_dict_path)
        labeled_qa, counts = run_stage4_divergence_labeling(self.qa_dataset, self.corpus, builder)
        self.assertEqual(len(labeled_qa), len(self.qa_dataset))
        for item in labeled_qa:
            self.assertIn(item["bucket"], ["No-mismatch", "Partial-mismatch", "Full alias-mismatch"])

if __name__ == '__main__':
    unittest.main()
