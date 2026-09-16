import unittest
from src.chem_utils import (
    parse_chemical_formula,
    canonicalize_hill_notation,
    formula_exact_match,
    token_jaccard_similarity,
    AliasMatcher,
    SharedSemanticEmbedder
)

class TestChemUtils(unittest.TestCase):
    def test_parse_chemical_formula(self):
        res1 = parse_chemical_formula("YBa2Cu3O7")
        self.assertEqual(res1, {"Y": 1.0, "Ba": 2.0, "Cu": 3.0, "O": 7.0})

        res2 = parse_chemical_formula("TiO2")
        self.assertEqual(res2, {"Ti": 1.0, "O": 2.0})

        res3 = parse_chemical_formula("LiFePO4")
        self.assertEqual(res3, {"Li": 1.0, "Fe": 1.0, "P": 1.0, "O": 4.0})

        # Invalid strings
        self.assertEqual(parse_chemical_formula("superconductor"), {})
        self.assertEqual(parse_chemical_formula(""), {})
        self.assertEqual(parse_chemical_formula(None), {})

    def test_canonicalize_hill_notation(self):
        # Non-carbon formulas: strictly alphabetical by element symbol
        self.assertEqual(canonicalize_hill_notation("YBa2Cu3O7"), "Ba2Cu3O7Y")
        self.assertEqual(canonicalize_hill_notation("TiO2"), "O2Ti")
        self.assertEqual(canonicalize_hill_notation("LiFePO4"), "FeLiO4P")

        # Carbon containing formula: C first, H second, then alphabetical
        c_formula = canonicalize_hill_notation("CH3NH3PbI3")  # MAPbI3
        self.assertTrue(c_formula.startswith("C"))

        # Non-formula returns empty string
        self.assertEqual(canonicalize_hill_notation("rutile phase"), "")

    def test_formula_exact_match(self):
        self.assertTrue(formula_exact_match("YBa2Cu3O7", "Ba2Cu3O7Y"))
        self.assertTrue(formula_exact_match("TiO2", "O2Ti"))
        self.assertFalse(formula_exact_match("TiO2", "MoS2"))

    def test_token_jaccard_similarity(self):
        sim1 = token_jaccard_similarity("YBCO superconductor", "YBCO superconductor")
        self.assertAlmostEqual(sim1, 1.0)

        sim2 = token_jaccard_similarity("YBCO thin film", "YBCO bulk pellet")
        self.assertGreater(sim2, 0.0)
        self.assertLess(sim2, 1.0)

        sim3 = token_jaccard_similarity("YBCO", "LiFePO4")
        self.assertAlmostEqual(sim3, 0.0)

    def test_shared_semantic_embedder(self):
        texts = [
            "YBCO is a high-temperature superconductor with Tc of 92 K.",
            "LiFePO4 is an olivine cathode material for lithium-ion batteries.",
            "MoS2 transition metal dichalcogenide monolayer."
        ]
        embedder = SharedSemanticEmbedder(n_components=2, random_state=42)
        embedder.fit(texts)
        self.assertTrue(embedder.is_fitted)

        vec = embedder.embed_text("superconductor transition temperature")
        self.assertEqual(vec.shape, (2,))

    def test_alias_matcher(self):
        alias_dict = {
            "ENTITY_001": {
                "primary_name": "YBa2Cu3O7",
                "categories": {
                    "abbreviations": ["YBCO", "123 cuprate"]
                }
            }
        }
        matcher = AliasMatcher(alias_dict)
        mentions = matcher.extract_mentions("High-resolution testing of YBCO showed critical transition.")
        self.assertEqual(len(mentions), 1)
        self.assertEqual(mentions[0]["surface_form"], "YBCO")
        self.assertEqual(mentions[0]["canonical_id"], "ENTITY_001")

if __name__ == '__main__':
    unittest.main()
