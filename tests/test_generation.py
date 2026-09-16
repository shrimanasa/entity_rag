import unittest
from src.stage7_generator import RAGAnswerGenerator
from src.stage7_evaluator import (
    normalize_text,
    compute_token_f1,
    extract_key_facts,
    compute_entity_property_fidelity
)

class TestGeneration(unittest.TestCase):
    def setUp(self):
        self.generator = RAGAnswerGenerator()

    def test_normalize_text(self):
        raw = "The critical transition temperature (Tc) of YBa2Cu3O7 is 92 K!"
        norm = normalize_text(raw)
        self.assertEqual(norm, "critical transition temperature tc of yba2cu3o7 is 92 k")

    def test_compute_token_f1(self):
        target = "YBa2Cu3O7 exhibits critical transition temperature Tc of 92 K."
        pred_identical = "YBa2Cu3O7 exhibits critical transition temperature Tc of 92 K."
        p, r, f1 = compute_token_f1(pred_identical, target)
        self.assertAlmostEqual(f1, 1.0)

        pred_partial = "Transition temperature is 92 K."
        p, r, f1 = compute_token_f1(pred_partial, target)
        self.assertGreater(f1, 0.0)
        self.assertLess(f1, 1.0)

        pred_disjoint = "Completely unrelated text without any matching words."
        p, r, f1 = compute_token_f1(pred_disjoint, target)
        self.assertAlmostEqual(f1, 0.0)

    def test_extract_key_facts_and_fidelity(self):
        gt = "YBa2Cu3O7 exhibits critical transition temperature Tc of 92 K, orthorhombic perovskite-like structure."
        facts = extract_key_facts(gt)
        self.assertIn("92 k", facts)
        self.assertIn("orthorhombic", facts)
        self.assertIn("perovskite", facts)

        pred_good = "High-resolution testing confirmed orthorhombic perovskite with Tc of 92 K."
        fid_good = compute_entity_property_fidelity(pred_good, gt)
        self.assertAlmostEqual(fid_good, 1.0)

        pred_bad = "No property details mentioned."
        fid_bad = compute_entity_property_fidelity(pred_bad, gt)
        self.assertAlmostEqual(fid_bad, 0.0)

    def test_generator_synthesis(self):
        query = "What is the critical temperature of YBCO?"
        passages = [
            {
                "passage_id": "P_001",
                "title": "Synthesis of YBCO",
                "abstract": "We report synthesis of YBCO.",
                "text": "Detailed characterization revealed critical transition temperature Tc of 92 K, orthorhombic perovskite-like structure, Pmmm space group."
            }
        ]
        res = self.generator.synthesize_from_passages(query, passages)
        self.assertIn("92 K", res["answer"])
        self.assertIn("P_001", res["cited_passages"])

    def test_insufficient_context(self):
        res = self.generator.synthesize_from_passages("Query about nonexistent material", [])
        self.assertEqual(res["answer"], "Insufficient context retrieved to answer the query.")
        self.assertEqual(res["cited_passages"], [])

if __name__ == '__main__':
    unittest.main()
