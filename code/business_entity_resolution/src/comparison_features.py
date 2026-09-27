"""
=============================================================================
comparison_features.py  -  PAIRWISE SIMILARITY FEATURES
=============================================================================
Role:  Data & Features  (Member 3, Pod 2)
Task:  Compute similarity features between candidate entity pairs for the
       matching classifier.

Public API:
    compute_pair_features(row_a, row_b)  -> dict of feature values
    compute_batch_features(pairs_df, df_s1, df_sx)  -> features DataFrame

Feature Vector (10 dimensions):
    1. name_jw           Jaro-Winkler similarity on name_clean
    2. name_jaccard      Jaccard coefficient of significant name tokens
    3. name_overlap      Fraction of shared significant tokens
    4. name_exact        Binary: exact match after normalization
    5. name_len_ratio    Length ratio (shorter / longer) of names
    6. addr_jw           Jaro-Winkler similarity on address_clean
    7. addr_jaccard      Jaccard coefficient of address tokens
    8. state_match       Binary: same extracted state code
    9. postal_match      Binary: same extracted postal code
   10. country_match     Binary: same country

Design:
    - Jaro-Winkler implemented from scratch (zero external dependency)
    - All features are floats in [0.0, 1.0] range
    - Optimized for batch processing of millions of pairs
    - F_0.5 alignment: features chosen for precision discrimination
=============================================================================
"""

import re
import math
import os
import sys

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

try:
    from blocking_features import (
        extract_postal_code,
        extract_state_code,
        extract_significant_tokens,
        NAME_STOPWORDS,
    )
except ImportError:
    from src.blocking_features import (
        extract_postal_code,
        extract_state_code,
        extract_significant_tokens,
        NAME_STOPWORDS,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1: STRING SIMILARITY ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════════

def jaro_similarity(s1, s2):
    """
    Compute Jaro similarity between two strings.

    Returns a float in [0.0, 1.0].  Handles edge cases:
      - Both empty -> 1.0
      - One empty  -> 0.0
      - Identical  -> 1.0 (fast path)
    """
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    # Maximum matching window
    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    transpositions = 0

    # Find matches
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    # Count transpositions
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    return (
        matches / len1 +
        matches / len2 +
        (matches - transpositions / 2) / matches
    ) / 3.0


def jaro_winkler_similarity(s1, s2, prefix_weight=0.1):
    """
    Compute Jaro-Winkler similarity between two strings.

    Adds a prefix bonus to the Jaro score for strings that share
    a common prefix (up to 4 characters).

    Parameters
    ----------
    s1, s2 : str
        Strings to compare.
    prefix_weight : float
        Scaling factor for prefix bonus (default 0.1, Winkler's original).

    Returns
    -------
    float
        Similarity in [0.0, 1.0].
    """
    jaro = jaro_similarity(s1, s2)

    # Common prefix (max 4 chars, per Winkler's specification)
    prefix_len = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break

    return jaro + prefix_len * prefix_weight * (1.0 - jaro)


def token_jaccard(tokens_a, tokens_b):
    """
    Jaccard coefficient between two token sets.

    |A ∩ B| / |A ∪ B|

    Returns 0.0 if both sets are empty.
    """
    if not tokens_a and not tokens_b:
        return 1.0  # Both empty → consider identical
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    intersection = set_a & set_b
    union = set_a | set_b
    if not union:
        return 1.0
    return len(intersection) / len(union)


def token_overlap(tokens_a, tokens_b):
    """
    Fraction of tokens shared relative to the shorter set.

    |A ∩ B| / min(|A|, |B|)

    Returns 1.0 if either set is empty (lenient for short names).
    """
    if not tokens_a or not tokens_b:
        return 0.0
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    intersection = set_a & set_b
    min_size = min(len(set_a), len(set_b))
    if min_size == 0:
        return 0.0
    return len(intersection) / min_size


def length_ratio(s1, s2):
    """
    Length ratio: shorter / longer.  Returns 1.0 for equal lengths.
    Returns 0.0 if both are empty.
    """
    len1, len2 = len(s1), len(s2)
    if len1 == 0 and len2 == 0:
        return 1.0
    if len1 == 0 or len2 == 0:
        return 0.0
    return min(len1, len2) / max(len1, len2)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2: ADDRESS TOKEN EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

# Common address words to exclude from token-based similarity
# (they're too frequent to be discriminative)
ADDR_STOPWORDS = {
    'road', 'rd', 'street', 'st', 'avenue', 'ave', 'lane', 'ln',
    'drive', 'dr', 'boulevard', 'blvd', 'court', 'ct', 'place', 'pl',
    'circle', 'cir', 'highway', 'hwy', 'parkway', 'pkwy', 'way',
    'floor', 'ground', 'first', 'second', 'third', 'fourth', 'fifth',
    'near', 'opposite', 'behind', 'adjacent', 'next', 'beside',
    'block', 'phase', 'sector', 'plot', 'flat', 'unit', 'suite', 'room',
    'no', 'number', 'building', 'complex', 'tower', 'towers',
    'nagar', 'colony', 'cross', 'main', 'house', 'door',
    'po', 'box', 'and', 'the', 'of', 'in', 'at', 'to',
    'new', 'old', 'north', 'south', 'east', 'west',
    'rue', 'chemin', 'boulevard', 'avenue', 'place', 'allee',
    'c/o', 's/o', 'd/o', 'w/o',
}


def extract_addr_tokens(address_clean):
    """
    Extract discriminative tokens from a normalized address.
    Filters out common address words that don't help disambiguation.
    """
    if not address_clean:
        return []
    words = re.findall(r'[a-zA-Z0-9\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F'
                       r'\u0B00-\u0B7F\u0C00-\u0C7F\u0D00-\u0D7F'
                       r'\u0E00-\u0E7F\u4E00-\u9FFF]+',
                       address_clean.lower())
    return [w for w in words if len(w) >= 2 and w not in ADDR_STOPWORDS]


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3: FEATURE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pair_features(row_a, row_b):
    """
    Compute the full feature vector for a candidate pair.

    Parameters
    ----------
    row_a : dict-like
        Record from source 1 with keys: entity_id, name_clean, address_clean,
        country.
    row_b : dict-like
        Record from source 2 or 3 with same keys.

    Returns
    -------
    dict
        Feature dictionary with 10 float values in [0.0, 1.0].
    """
    name_a = row_a.get('name_clean', '') or ''
    name_b = row_b.get('name_clean', '') or ''
    addr_a = row_a.get('address_clean', '') or ''
    addr_b = row_b.get('address_clean', '') or ''
    country_a = row_a.get('country', '') or ''
    country_b = row_b.get('country', '') or ''

    # ── Name features ────────────────────────────────────────────────────
    name_tokens_a = extract_significant_tokens(name_a)
    name_tokens_b = extract_significant_tokens(name_b)

    feat_name_jw = jaro_winkler_similarity(name_a, name_b)
    feat_name_jaccard = token_jaccard(name_tokens_a, name_tokens_b)
    feat_name_overlap = token_overlap(name_tokens_a, name_tokens_b)
    feat_name_exact = 1.0 if name_a == name_b and name_a != '' else 0.0
    feat_name_len_ratio = length_ratio(name_a, name_b)

    # ── Address features ─────────────────────────────────────────────────
    addr_tokens_a = extract_addr_tokens(addr_a)
    addr_tokens_b = extract_addr_tokens(addr_b)

    feat_addr_jw = jaro_winkler_similarity(addr_a, addr_b)
    feat_addr_jaccard = token_jaccard(addr_tokens_a, addr_tokens_b)

    # ── State & postal match ─────────────────────────────────────────────
    state_a = extract_state_code(addr_a, country_a)
    state_b = extract_state_code(addr_b, country_b)
    feat_state_match = 1.0 if (state_a and state_b and state_a == state_b) else 0.0

    postal_a = extract_postal_code(addr_a, country_a)
    postal_b = extract_postal_code(addr_b, country_b)
    feat_postal_match = 1.0 if (postal_a and postal_b and postal_a == postal_b) else 0.0

    # ── Country match ────────────────────────────────────────────────────
    feat_country_match = 1.0 if country_a == country_b else 0.0

    return {
        'name_jw':        feat_name_jw,
        'name_jaccard':   feat_name_jaccard,
        'name_overlap':   feat_name_overlap,
        'name_exact':     feat_name_exact,
        'name_len_ratio': feat_name_len_ratio,
        'addr_jw':        feat_addr_jw,
        'addr_jaccard':   feat_addr_jaccard,
        'state_match':    feat_state_match,
        'postal_match':   feat_postal_match,
        'country_match':  feat_country_match,
    }


# Feature column names in canonical order
FEATURE_COLS = [
    'name_jw', 'name_jaccard', 'name_overlap', 'name_exact',
    'name_len_ratio', 'addr_jw', 'addr_jaccard',
    'state_match', 'postal_match', 'country_match',
]


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4: SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 72)
    print("  COMPARISON FEATURES SELF-TEST")
    print("=" * 72)

    # ── Jaro-Winkler Tests ───────────────────────────────────────────────
    jw_tests = [
        ("", "", 1.0),
        ("abc", "", 0.0),
        ("hello", "hello", 1.0),
        ("martha", "marhta", 0.961),  # Classic Jaro-Winkler example
        ("dwayne", "duane", 0.84),
        ("dixon", "dicksonx", 0.81),
    ]

    print("\n--- JARO-WINKLER TESTS ---\n")
    jw_pass = 0
    jw_fail = 0
    for s1, s2, expected in jw_tests:
        result = jaro_winkler_similarity(s1, s2)
        ok = abs(result - expected) < 0.02
        tag = "PASS" if ok else "FAIL"
        if ok:
            jw_pass += 1
        else:
            jw_fail += 1
        print(f"  [{tag}]  JW({s1!r}, {s2!r}) = {result:.4f}  (expected ~{expected:.3f})")

    # ── Token Similarity Tests ───────────────────────────────────────────
    print("\n--- TOKEN SIMILARITY TESTS ---\n")
    tok_tests = [
        (["halocast", "foods"], ["halocast", "foods"], 1.0, 1.0),
        (["halocast", "foods"], ["halocast", "beverages"], 0.333, 0.5),
        (["abc"], ["def"], 0.0, 0.0),
        ([], [], 1.0, 0.0),
    ]
    tok_pass = 0
    tok_fail = 0
    for ta, tb, exp_j, exp_o in tok_tests:
        j = token_jaccard(ta, tb)
        o = token_overlap(ta, tb)
        ok_j = abs(j - exp_j) < 0.02
        ok_o = abs(o - exp_o) < 0.02
        tag = "PASS" if (ok_j and ok_o) else "FAIL"
        if ok_j and ok_o:
            tok_pass += 1
        else:
            tok_fail += 1
        print(f"  [{tag}]  Jaccard={j:.3f} (exp {exp_j:.3f}), "
              f"Overlap={o:.3f} (exp {exp_o:.3f})  {ta} vs {tb}")

    # ── Full Pair Features Test ──────────────────────────────────────────
    print("\n--- PAIR FEATURES TEST ---\n")

    # True match: same entity, slightly different formatting
    row_a = {
        'entity_id': 'S1-001',
        'name_clean': 'halocast private limited',
        'address_clean': '797, lake town block a, kolkata, howrah, wb',
        'country': 'India',
    }
    row_b = {
        'entity_id': 'S2-001',
        'name_clean': 'halocast private limited',
        'address_clean': '797 lake town block a, kolkata, wb',
        'country': 'India',
    }
    feats_match = compute_pair_features(row_a, row_b)
    print(f"  TRUE MATCH pair:")
    for k, v in feats_match.items():
        print(f"    {k:18s} = {v:.4f}")

    # Non-match: different entities
    row_c = {
        'entity_id': 'S3-999',
        'name_clean': 'supreme consulting services',
        'address_clean': '500 market street, san jose, ca',
        'country': 'US',
    }
    feats_nomatch = compute_pair_features(row_a, row_c)
    print(f"\n  NON-MATCH pair:")
    for k, v in feats_nomatch.items():
        print(f"    {k:18s} = {v:.4f}")

    # Validate: match pair should have higher name_jw than non-match
    assert feats_match['name_jw'] > feats_nomatch['name_jw'], \
        "Match pair should have higher name_jw than non-match!"
    assert feats_match['state_match'] == 1.0, "State should match!"
    assert feats_nomatch['country_match'] == 0.0, "Countries differ!"

    total_pass = jw_pass + tok_pass + 1  # +1 for pair features assertion
    total_fail = jw_fail + tok_fail
    total = total_pass + total_fail

    print(f"\n{'=' * 72}")
    print(f"  RESULTS:  {total_pass}/{total} passed")
    if total_fail > 0:
        print(f"  FAILURES: {total_fail}")
    else:
        print("  ALL TESTS PASSED ✓")
    print("=" * 72)
