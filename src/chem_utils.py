import re
from typing import Dict, List, Set, Tuple

# Valid Periodic Table chemical element symbols
PERIODIC_TABLE = {
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
}

def parse_chemical_formula(formula: str) -> Dict[str, float]:
    """
    Parses a chemical formula string into element counts.
    Handles standard formulas like YBa2Cu3O7, LiFePO4, MoS2, TiO2, Bi2Te3, MAPbI3.
    Strictly verifies that parsed symbols are real chemical elements and cover the entire string.
    Returns dictionary mapping element symbol -> float count, or {} if not a formula.
    """
    if not formula or not isinstance(formula, str):
        return {}
        
    formula = formula.strip()
    if ' ' in formula or len(formula) > 60:
        return {}

    # Strip sub-stoichiometry tags like -x, -d, +x, +d
    cleaned = re.sub(r'[-\+][xX\delta\dd]', '', formula)
    cleaned = re.sub(r'[\s\(\)\[\]_]', '', cleaned)
    
    # Match Element + optional integer/decimal count
    pattern = r'([A-Z][a-z]?)([0-9]*(?:\.[0-9]+)?)'
    matches = re.findall(pattern, cleaned)
    
    if not matches:
        return {}
    
    counts: Dict[str, float] = {}
    total_parsed_len = 0
    
    for elem, count_str in matches:
        if elem not in PERIODIC_TABLE:
            return {}  # Not a valid chemical element
            
        try:
            count = float(count_str) if count_str else 1.0
        except ValueError:
            return {}
            
        counts[elem] = counts.get(elem, 0.0) + count
        total_parsed_len += len(elem) + len(count_str)
        
    # Ensure full coverage of the cleaned string without residual English characters
    if total_parsed_len != len(cleaned):
        return {}
        
    return counts

def canonicalize_hill_notation(formula: str) -> str:
    """
    Converts a chemical formula into Hill system canonical notation.
    If Carbon (C) is present: C first, then H, then all other elements alphabetically.
    If Carbon is NOT present: all elements in alphabetical order.
    Returns empty string "" if formula cannot be strictly parsed as a chemical formula.
    """
    counts = parse_chemical_formula(formula)
    if not counts:
        return ""
        
    elements = list(counts.keys())
    if 'C' in elements:
        sorted_elems = ['C']
        if 'H' in elements:
            sorted_elems.append('H')
        remaining = sorted([e for e in elements if e not in ('C', 'H')])
        sorted_elems.extend(remaining)
    else:
        sorted_elems = sorted(elements)
        
    res = []
    for elem in sorted_elems:
        cnt = counts[elem]
        cnt_str = f"{int(cnt)}" if cnt == int(cnt) else f"{cnt}"
        res.append(f"{elem}{cnt_str if cnt != 1 else ''}")
        
    return "".join(res)

def formula_exact_match(formula1: str, formula2: str) -> bool:
    """Checks if two formula strings resolve to the exact same canonical Hill notation."""
    c1 = canonicalize_hill_notation(formula1)
    c2 = canonicalize_hill_notation(formula2)
    if not c1 or not c2:
        return False
    return c1.upper() == c2.upper()

def token_jaccard_similarity(text1: str, text2: str) -> float:
    """Computes token-level Jaccard similarity between two texts."""
    tokens1: Set[str] = set(re.findall(r'\b\w+\b', text1.lower()))
    tokens2: Set[str] = set(re.findall(r'\b\w+\b', text2.lower()))
    
    if not tokens1 or not tokens2:
        return 0.0
        
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return intersection / union if union > 0 else 0.0

def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes standard edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def normalized_levenshtein_similarity(s1: str, s2: str) -> float:
    """Computes 1 - (levenshtein_distance / max_len)."""
    str1, str2 = s1.lower(), s2.lower()
    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(str1, str2)
    return 1.0 - (dist / max_len)

def calculate_lexical_divergence(query: str, evidence: str) -> float:
    """
    Stage 4 metric:
    Computes query-to-evidence token containment overlap.
    """
    tokens_q = set(re.findall(r'\b\w+\b', query.lower()))
    tokens_ev = set(re.findall(r'\b\w+\b', evidence.lower()))
    
    if not tokens_q:
        return 0.0
        
    overlap_containment = len(tokens_q.intersection(tokens_ev)) / len(tokens_q)
    return overlap_containment

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

class SharedSemanticEmbedder:
    """
    Unified LSA / TruncatedSVD Semantic Embedder.
    Fit ONCE on the shared universe (corpus passages + entity profiles + queries),
    then reused deterministically across all stages (Stage 1, Stage 2, Stage 5).
    """
    def __init__(self, n_components: int = 128, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
        self.svd = TruncatedSVD(n_components=n_components, algorithm='arpack', random_state=random_state)
        self.is_fitted = False

    def fit(self, texts: List[str]):
        tfidf_mat = self.vectorizer.fit_transform(texts)
        self.svd.fit(tfidf_mat)
        self.is_fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("SharedSemanticEmbedder must be fitted before transform() is called.")
        tfidf_mat = self.vectorizer.transform(texts)
        dense_mat = self.svd.transform(tfidf_mat)
        # Unit L2 normalization for clean cosine similarity via dot products
        norms = np.linalg.norm(dense_mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return dense_mat / norms

    def embed_text(self, text: str) -> np.ndarray:
        return self.transform([text])[0]

class AliasMatcher:
    """
    Unified Word-Boundary Regex and Canonical Mapping Engine.
    Used identically across Stage 1 (NER), Stage 3 (Graph), and Stage 4 (Labeler).
    """
    def __init__(self, alias_dict: Dict):
        self.alias_dict = alias_dict
        self.surface_to_canonical: Dict[str, str] = {}
        all_aliases = []

        for ent_id, info in alias_dict.items():
            if "primary_name" in info:
                pname = info["primary_name"]
                self.surface_to_canonical[pname.lower()] = ent_id
                all_aliases.append(pname)
            for cat, names in info.get("categories", {}).items():
                for name in names:
                    self.surface_to_canonical[name.lower()] = ent_id
                    all_aliases.append(name)

        # Sort unique aliases by length descending so longer compound phrases match before short acronyms
        unique_aliases = sorted(list(set(all_aliases)), key=lambda x: len(x), reverse=True)
        
        patterns = []
        for a in unique_aliases:
            if len(a.strip()) >= 2:
                patterns.append(r'\b' + re.escape(a) + r'\b')

        self.combined_pattern = re.compile("|".join(patterns), re.IGNORECASE) if patterns else None

    def extract_mentions(self, text: str) -> List[Dict]:
        if not self.combined_pattern:
            return []
        
        tokens = text.split()
        mentions = []
        for match in self.combined_pattern.finditer(text):
            surface_form = match.group(0)
            char_start, char_end = match.span()
            
            pre_text = text[:char_start]
            token_idx = len(pre_text.split())
            ctx_start = max(0, token_idx - 20)
            ctx_end = min(len(tokens), token_idx + 20 + 1)
            context_window = " ".join(tokens[ctx_start:ctx_end])
            
            canonical_id = self.surface_to_canonical.get(surface_form.lower(), "NIL")
            
            mentions.append({
                "surface_form": surface_form,
                "char_span": (char_start, char_end),
                "context_window": context_window,
                "canonical_id": canonical_id
            })
        return mentions

    def resolve_surface_to_canonical(self, text: str) -> str:
        """Resolves text or surface form to canonical entity ID."""
        text_lower = text.strip().lower()
        if text_lower in self.surface_to_canonical:
            return self.surface_to_canonical[text_lower]

        # Extract mentions with word boundaries
        extracted = self.extract_mentions(text)
        if extracted:
            return extracted[0]["canonical_id"]

        # Check Hill formula exact match
        for ent_id, info in self.alias_dict.items():
            primary = info["primary_name"]
            if formula_exact_match(text, primary):
                return ent_id

        return "NIL"
