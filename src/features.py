"""
Feature extraction and data loading module for Business Entity Resolution.

This module provides:
1. `PairFeatureExtractor`: Extracts pairwise numerical features between two records
   (name1, address1) and (name2, address2) using string similarities (RapidFuzz, Jaccard),
   numerical overlap, and exact match flags.
2. `load_entity_dict`: Reads any source TSV file using sep='\t' and stores records in a
   dictionary {entity_id: (business_name, business_address)} for fast in-memory lookup.
"""

import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from rapidfuzz import distance, fuzz


def _extract_word_tokens(text: Optional[str]) -> Set[str]:
    """Extract word tokens in lowercase from text."""
    if not text:
        return set()
    return set(re.findall(r"\w+", text.lower()))


def _extract_numbers(text: Optional[str]) -> Set[str]:
    """Extract normalized house / street numbers from text."""
    if not text:
        return set()
    # Normalize leading zeros so '05' matches '5'
    return {num.lstrip("0") or "0" for num in re.findall(r"\d+", text)}


def compute_word_token_jaccard(text1: Optional[str], text2: Optional[str]) -> float:
    """Compute word token Jaccard similarity between two address strings."""
    tokens1 = _extract_word_tokens(text1)
    tokens2 = _extract_word_tokens(text2)
    union_len = len(tokens1 | tokens2)
    if union_len == 0:
        str1 = text1.strip().lower() if text1 else ""
        str2 = text2.strip().lower() if text2 else ""
        return 1.0 if str1 == str2 and str1 != "" else 0.0
    return float(len(tokens1 & tokens2) / union_len)


def compute_numerical_overlap(text1: Optional[str], text2: Optional[str]) -> float:
    """Check if house or street numbers in address1 match numbers in address2."""
    nums1 = _extract_numbers(text1)
    nums2 = _extract_numbers(text2)
    if not nums1 or not nums2:
        return 0.0
    return 1.0 if bool(nums1 & nums2) else 0.0


def _parse_record(record: Any) -> Tuple[str, str]:
    """Robustly parse a record into (name, address) strings."""
    if record is None:
        return "", ""
    if isinstance(record, (tuple, list)):
        name = str(record[0]) if len(record) > 0 and record[0] is not None else ""
        addr = str(record[1]) if len(record) > 1 and record[1] is not None else ""
    elif isinstance(record, dict):
        name = record.get("business_name") or record.get("name") or ""
        addr = record.get("business_address") or record.get("address") or ""
        name = str(name) if name is not None else ""
        addr = str(addr) if addr is not None else ""
    elif hasattr(record, "business_name") and hasattr(record, "business_address"):
        name = str(record.business_name or "")
        addr = str(record.business_address or "")
    elif hasattr(record, "name") and hasattr(record, "address"):
        name = str(record.name or "")
        addr = str(record.address or "")
    else:
        name = str(record)
        addr = ""

    # Sanitize float-str representations of NaN
    if name.lower() == "nan":
        name = ""
    if addr.lower() == "nan":
        addr = ""
    return name, addr


class PairFeatureExtractor:
    """
    Extracts numerical features from a pair of business records (name1, address1)
    and (name2, address2).

    Features computed:
    - RapidFuzz fuzz.token_sort_ratio for names
    - RapidFuzz fuzz.token_set_ratio for names
    - RapidFuzz distance.JaroWinkler.similarity for names
    - Word token Jaccard similarity for addresses
    - Numerical overlap: check if house or street numbers in address1 match numbers in address2
    - Exact match boolean flags for name and address
    """

    CANONICAL_FEATURES = [
        "name_token_sort_ratio",
        "name_token_set_ratio",
        "name_jaro_winkler_similarity",
        "address_token_jaccard",
        "numerical_overlap",
        "name_exact_match",
        "address_exact_match",
    ]

    def __init__(
        self,
        record1: Optional[Any] = None,
        record2: Optional[Any] = None,
        canonical_only: bool = False,
    ):
        """
        Initialize the feature extractor.
        Optionally pre-binds record1 and record2.
        """
        self.record1 = record1
        self.record2 = record2
        self.canonical_only = canonical_only

    def extract_features(
        self,
        record1: Optional[Any] = None,
        record2: Optional[Any] = None,
        *args,
        canonical_only: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Extract numerical features from two records (name1, address1) and (name2, address2).

        Can be called with:
        - extractor.extract_features((name1, address1), (name2, address2))
        - extractor.extract_features(name1, address1, name2, address2)
        - extractor.extract_features()  # if record1, record2 passed at __init__

        Returns:
            dict: Numerical feature dictionary.
        """
        # Allow static / class-level invocation: PairFeatureExtractor.extract_features(rec1, rec2)
        if not isinstance(self, PairFeatureExtractor):
            extractor = PairFeatureExtractor()
            return extractor.extract_features(self, record1, *args, canonical_only=canonical_only, **kwargs)

        if canonical_only is None:
            canonical_only = self.canonical_only

        # Unpack arguments
        if record1 is None and record2 is None:
            if self.record1 is not None and self.record2 is not None:
                rec1, rec2 = self.record1, self.record2
            else:
                rec1, rec2 = ("", ""), ("", "")
        elif record1 is not None and record2 is not None:
            if len(args) == 2:
                # Called as extract_features(name1, address1, name2, address2)
                rec1 = (record1, record2)
                rec2 = (args[0], args[1])
            else:
                rec1, rec2 = record1, record2
        elif record1 is not None and record2 is None:
            rec1, rec2 = record1, kwargs.get("record2", ("", ""))
        else:
            rec1, rec2 = ("", ""), ("", "")

        name1, address1 = _parse_record(rec1)
        name2, address2 = _parse_record(rec2)

        # 1. RapidFuzz fuzz.token_sort_ratio and fuzz.token_set_ratio for names
        name_token_sort = float(fuzz.token_sort_ratio(name1, name2))
        name_token_set = float(fuzz.token_set_ratio(name1, name2))

        # 2. RapidFuzz distance.JaroWinkler.similarity for names
        name_jaro_winkler = float(distance.JaroWinkler.similarity(name1, name2))

        # 3. Word token Jaccard similarity for addresses
        address_jaccard = float(compute_word_token_jaccard(address1, address2))

        # 4. Numerical overlap: check if house or street numbers in address1 match numbers in address2
        num_overlap = float(compute_numerical_overlap(address1, address2))

        # 5. Exact match boolean flags for name and address
        name_exact = 1.0 if name1.strip().lower() == name2.strip().lower() else 0.0
        address_exact = 1.0 if address1.strip().lower() == address2.strip().lower() else 0.0

        if canonical_only:
            return {
                "name_token_sort_ratio": name_token_sort,
                "name_token_set_ratio": name_token_set,
                "name_jaro_winkler_similarity": name_jaro_winkler,
                "address_token_jaccard": address_jaccard,
                "numerical_overlap": num_overlap,
                "name_exact_match": name_exact,
                "address_exact_match": address_exact,
            }

        # Include canonical keys plus standard aliases for universal compatibility
        return {
            # Canonical features
            "name_token_sort_ratio": name_token_sort,
            "name_token_set_ratio": name_token_set,
            "name_jaro_winkler_similarity": name_jaro_winkler,
            "address_token_jaccard": address_jaccard,
            "numerical_overlap": num_overlap,
            "name_exact_match": name_exact,
            "address_exact_match": address_exact,
            # Name aliases
            "token_sort_ratio": name_token_sort,
            "token_set_ratio": name_token_set,
            "fuzz_token_sort_ratio": name_token_sort,
            "fuzz_token_set_ratio": name_token_set,
            "name_jaro_winkler": name_jaro_winkler,
            "jaro_winkler_similarity": name_jaro_winkler,
            "jaro_winkler": name_jaro_winkler,
            # Address aliases
            "address_jaccard": address_jaccard,
            "word_token_jaccard": address_jaccard,
            "address_word_token_jaccard": address_jaccard,
            # Exact match aliases
            "exact_match_name": name_exact,
            "exact_match_address": address_exact,
            # Number overlap aliases
            "number_overlap": num_overlap,
            "num_overlap": num_overlap,
        }

    def extract(self, *args, **kwargs) -> Dict[str, float]:
        """Alias for extract_features."""
        return self.extract_features(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> Dict[str, float]:
        """Calling the instance invokes extract_features."""
        return self.extract_features(*args, **kwargs)

    def extract_feature_vector(self, record1: Optional[Any] = None, record2: Optional[Any] = None, *args, **kwargs) -> List[float]:
        """Return the canonical features as a numerical list/vector."""
        feats = self.extract_features(record1, record2, *args, canonical_only=True, **kwargs)
        return [feats[k] for k in self.CANONICAL_FEATURES]

    def extract_batch(
        self, pairs: List[Tuple[Any, Any]], canonical_only: bool = True
    ) -> List[Dict[str, float]]:
        """Extract features for a list of (record1, record2) pairs."""
        return [
            self.extract_features(rec1, rec2, canonical_only=canonical_only)
            for rec1, rec2 in pairs
        ]


def load_entity_dict(tsv_path: Union[str, os.PathLike], **kwargs) -> Dict[str, Tuple[str, str]]:
    """
    Reads any source TSV file using sep='\t' and stores records in a dictionary
    {entity_id: (business_name, business_address)} for fast in-memory lookup.

    Parameters
    ----------
    tsv_path : str or Path
        Path to the TSV file (e.g. train_source1.tsv, test_source2.tsv).
    **kwargs :
        Optional arguments passed directly to pd.read_csv (e.g. nrows, chunksize).

    Returns
    -------
    dict
        Dictionary mapping entity_id to (business_name, business_address).
    """
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False, **kwargs)
    df = df.fillna("")

    cols = list(df.columns)
    id_col = "entity_id" if "entity_id" in cols else cols[0]
    name_col = "business_name" if "business_name" in cols else (cols[1] if len(cols) > 1 else id_col)
    addr_col = "business_address" if "business_address" in cols else (cols[2] if len(cols) > 2 else name_col)

    return dict(zip(df[id_col], zip(df[name_col], df[addr_col])))
