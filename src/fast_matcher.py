#!/usr/bin/env python3
"""
Fast High-Precision Entity Matcher v2 — Macro F_0.5 Optimized.

Architecture:
1. Smarter Inverted Index Blocking:
   - Partitions processing by country to stay within memory limits (< 2GB RAM).
   - Three blocking keys per record:
     * Key A: (country, first_meaningful_word_of_name) — expanded stopwords, len>=3, 4-char fallback.
     * Key B: (country, extracted_address_numbers) — house/suite/shop numbers.
     * Key C: (country, postal_code + first_3_letters_of_name) — high-precision postal code anchor.
   - Strict candidate cap: max 8 candidate IDs per Source 1 entity.
2. Precision-Biased Scoring (Macro F_0.5):
   - RapidFuzz token_set_ratio on lowercased names.
   - JaroWinkler similarity on lowercased names.
   - Address digit overlap and conflict detection.
   - Strict thresholds: name_sim >= 82 AND digits match, OR name_sim >= 92 for noisy addresses.
   - Max 2 matched IDs per Source 1 entity.
3. Streaming File Outputs:
   - Streams line-by-line output preserving test_source1.tsv ordering.
   - Generates output/candidate_pairs.tsv and output/matching_results.tsv.
   - Automatically runs official validator at completion.
"""

import gc
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

from rapidfuzz import fuzz, distance

# ---------------------------------------------------------------------------
# Pre-compiled regex patterns
# ---------------------------------------------------------------------------
RE_WORD = re.compile(r"[^\W_]+", re.UNICODE)
RE_DIGITS = re.compile(r"\d+")
RE_POSTAL_5 = re.compile(r"\b\d{5}\b")  # US / France zip codes
RE_POSTAL_6 = re.compile(r"\b\d{6}\b")  # India PIN codes

# ---------------------------------------------------------------------------
# Business name stopwords — generic words that produce overly broad blocking
# buckets and pollute candidate sets.
# ---------------------------------------------------------------------------
NAME_STOPWORDS = frozenset({
    # Articles / honorifics
    "the", "a", "an", "m", "s", "dr", "mr", "ms", "of", "and", "de", "le", "la", "les",
    # Indian honorifics
    "shree", "sri", "shri",
    # Corporate suffixes
    "inc", "corp", "llc", "ltd", "pvt", "sa", "sarl", "co", "sas", "gmbh",
    "private", "limited", "company", "corporation", "group",
    # Generic business type words
    "hotel", "restaurant", "cafe", "store", "stores", "shop", "mart",
    "enterprises", "solutions", "services", "industries", "agency",
    "traders", "trading", "associates", "foundation", "institute",
    "international", "national", "general",
})


def get_blocking_word(name: str) -> str:
    """
    Extract the first meaningful blocking token from a business name.

    Rules:
    - Skip any token in NAME_STOPWORDS.
    - Require length >= 3 to avoid initials / single chars.
    - Fallback: first 4 characters of the concatenated cleaned name.
    """
    if not name:
        return ""
    tokens = RE_WORD.findall(name.lower())
    for t in tokens:
        if t not in NAME_STOPWORDS and len(t) >= 3:
            return t
    # Fallback: first 4 chars of concatenated tokens
    cleaned = "".join(tokens)
    return cleaned[:4] if len(cleaned) >= 4 else cleaned


def get_name_prefix3(name: str) -> str:
    """Return the first 3 lowercase letters from the business name (for postal code key)."""
    letters = RE_WORD.findall(name.lower())
    joined = "".join(letters)
    return joined[:3] if len(joined) >= 3 else joined


def extract_postal_codes(addr: str, country: str) -> List[str]:
    """
    Extract postal / ZIP / PIN codes from an address string.
    - US and France: 5-digit codes.
    - India: 6-digit PIN codes.
    """
    if not addr:
        return []
    if country in ("US", "France"):
        return RE_POSTAL_5.findall(addr)
    elif country == "India":
        return RE_POSTAL_6.findall(addr)
    return []


def extract_numbers(addr: str) -> List[str]:
    """Extract normalized digit sequences from an address (house/suite/shop numbers)."""
    if not addr:
        return []
    # Strip leading zeros; cap at 6 digits to avoid phone numbers / timestamps
    return [n.lstrip("0") or "0" for n in RE_DIGITS.findall(addr) if len(n) <= 6]


def score_pair(
    name1_lower: str,
    name2_lower: str,
    nums1: Set[str],
    nums2: Set[str],
) -> Tuple[float, bool, bool]:
    """
    Score a candidate pair.

    Returns:
        (name_similarity, digits_match, digits_conflict)
        - name_similarity: max(token_set_ratio, JaroWinkler * 100) on lowercased names.
        - digits_match: True if both have numbers and at least one overlaps.
        - digits_conflict: True if both have numbers but NONE overlap.
    """
    # Combined name score: best of token_set_ratio and JaroWinkler * 100
    tsr = fuzz.token_set_ratio(name1_lower, name2_lower)
    jw = distance.JaroWinkler.similarity(name1_lower, name2_lower) * 100.0
    name_sim = max(tsr, jw)

    if nums1 and nums2:
        overlap = bool(nums1 & nums2)
        return name_sim, overlap, not overlap
    else:
        # One or both have no numbers — can't confirm or conflict
        return name_sim, False, False


def is_match(
    name_sim: float,
    digits_match: bool,
    digits_conflict: bool,
) -> bool:
    """
    Apply strict Macro F_0.5 precision-biased decision rule.

    Match if:
      - name_sim >= 82 AND digits match, OR
      - name_sim >= 92 (for cases with slight address noise / missing numbers).
    Never match if digits conflict.
    """
    if digits_conflict:
        return False
    if digits_match and name_sim >= 82:
        return True
    if name_sim >= 92:
        return True
    return False


def run_fast_matcher(
    test_dir: str = "dataset/test",
    output_dir: str = "output",
    max_candidates_per_entity: int = 8,
    max_matches_per_entity: int = 2,
    bucket_cap: int = 30,
):
    print("=" * 70)
    print("Fast High-Precision Entity Matcher v2 — Macro F_0.5 Optimized")
    print("=" * 70)
    start_time = time.time()

    os.makedirs(output_dir, exist_ok=True)
    candidate_out_path = os.path.join(output_dir, "candidate_pairs.tsv")
    matching_out_path = os.path.join(output_dir, "matching_results.tsv")

    test_s1_path = os.path.join(test_dir, "test_source1.tsv")
    test_s2_path = os.path.join(test_dir, "test_source2.tsv")
    test_s3_path = os.path.join(test_dir, "test_source3.tsv")

    for p in (test_s1_path, test_s2_path, test_s3_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing required test file: {p}")

    # -----------------------------------------------------------------------
    # Step 1: Discover countries
    # -----------------------------------------------------------------------
    print("[1/4] Discovering country partition in test_source1.tsv...")
    s1_countries: Set[str] = set()
    with open(test_s1_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.split("\t")
            if len(parts) >= 4:
                s1_countries.add(parts[3].strip())

    ordered_countries = sorted(s1_countries)
    print(f"      Countries found: {ordered_countries}")

    # Result maps — populated per-country, written at end
    cand_results: Dict[str, str] = {}
    match_results: Dict[str, str] = {}

    total_s1_processed = 0
    total_candidates_found = 0
    total_matches_predicted = 0

    # -----------------------------------------------------------------------
    # Step 2: Process each country independently
    # -----------------------------------------------------------------------
    for country_idx, country in enumerate(ordered_countries, start=1):
        print(f"\n--- [{country_idx}/{len(ordered_countries)}] Processing Country: {country} ---")
        t_country_start = time.time()

        # ------ Index Source 2 and Source 3 for this country ------
        # records[eid] = (name, addr, name_lower)
        records: Dict[str, Tuple[str, str, str]] = {}
        idx_word: Dict[str, List[str]] = {}    # Key A: blocking word
        idx_num: Dict[str, List[str]] = {}     # Key B: address numbers
        idx_postal: Dict[str, List[str]] = {}  # Key C: postal+prefix3

        print(f"  Indexing Source 2 & Source 3 for {country}...")
        for src_path in (test_s2_path, test_s3_path):
            with open(src_path, "r", encoding="utf-8") as f:
                f.readline()
                for line in f:
                    parts = line.split("\t")
                    if len(parts) >= 4 and parts[3].strip() == country:
                        eid = parts[0].strip()
                        name = parts[1].strip()
                        addr = parts[2].strip()
                        name_lower = name.lower()

                        records[eid] = (name, addr, name_lower)

                        # Key A: First meaningful blocking word
                        w = get_blocking_word(name)
                        if w:
                            b = idx_word.setdefault(w, [])
                            if len(b) < bucket_cap:
                                b.append(eid)

                        # Key B: Address numbers (len >= 2 digits)
                        for n in extract_numbers(addr):
                            if len(n) >= 2:
                                b = idx_num.setdefault(n, [])
                                if len(b) < bucket_cap:
                                    b.append(eid)

                        # Key C: Postal code + name prefix
                        prefix3 = get_name_prefix3(name)
                        if prefix3:
                            for pc in extract_postal_codes(addr, country):
                                key = pc + ":" + prefix3
                                b = idx_postal.setdefault(key, [])
                                if len(b) < bucket_cap:
                                    b.append(eid)

        t_index = time.time()
        print(
            f"  Indexed {len(records):,} records "
            f"({len(idx_word):,} word keys, {len(idx_num):,} num keys, "
            f"{len(idx_postal):,} postal keys) in {t_index - t_country_start:.2f} s"
        )

        # ------ Stream Source 1 and score candidates ------
        print(f"  Streaming Source 1 queries for {country}...")
        s1_country_count = 0
        matches_country_count = 0
        cands_country_count = 0

        with open(test_s1_path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.split("\t")
                if len(parts) >= 4 and parts[3].strip() == country:
                    s1_id = parts[0].strip()
                    name1 = parts[1].strip()
                    addr1 = parts[2].strip()
                    name1_lower = name1.lower()

                    w1 = get_blocking_word(name1)
                    nums1_list = extract_numbers(addr1)
                    nums1 = set(nums1_list)
                    prefix3_1 = get_name_prefix3(name1)
                    postals_1 = extract_postal_codes(addr1, country)

                    # Gather candidates from all three index types
                    cands: List[str] = []

                    # Key A: word index
                    if w1 and w1 in idx_word:
                        cands.extend(idx_word[w1])

                    # Key B: address number index
                    for n in nums1_list:
                        if len(n) >= 2 and n in idx_num:
                            cands.extend(idx_num[n])

                    # Key C: postal + name prefix index
                    if prefix3_1:
                        for pc in postals_1:
                            key = pc + ":" + prefix3_1
                            if key in idx_postal:
                                cands.extend(idx_postal[key])

                    # Deduplicate preserving order, enforce strict 8-candidate cap
                    cands_uniq = list(dict.fromkeys(cands))[:max_candidates_per_entity]
                    cand_results[s1_id] = ",".join(cands_uniq)
                    cands_country_count += len(cands_uniq)

                    # Score candidates and select matches
                    scored: List[Tuple[float, str]] = []
                    for cid in cands_uniq:
                        rec2 = records.get(cid)
                        if not rec2:
                            continue
                        name2, addr2, name2_lower = rec2

                        nums2 = set(extract_numbers(addr2))
                        name_sim, digits_match, digits_conflict = score_pair(
                            name1_lower, name2_lower, nums1, nums2
                        )

                        if is_match(name_sim, digits_match, digits_conflict):
                            scored.append((name_sim, cid))

                    # Sort by descending similarity, take top max_matches_per_entity
                    scored.sort(key=lambda x: -x[0])
                    matched_ids = [cid for _, cid in scored[:max_matches_per_entity]]

                    match_results[s1_id] = ",".join(matched_ids)
                    matches_country_count += len(matched_ids)
                    s1_country_count += 1

        t_score = time.time()
        print(f"  Completed {s1_country_count:,} Source 1 entities in {t_score - t_index:.2f} s:")
        print(f"    - Candidates generated: {cands_country_count:,}")
        print(f"    - Matches predicted:    {matches_country_count:,}")

        total_s1_processed += s1_country_count
        total_candidates_found += cands_country_count
        total_matches_predicted += matches_country_count

        # Reclaim memory before next country
        del records, idx_word, idx_num, idx_postal
        gc.collect()

    # -----------------------------------------------------------------------
    # Step 3: Write output files in exact test_source1.tsv order
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[3/4] Writing output files in exact test_source1.tsv order...")
    t_write_start = time.time()

    with open(test_s1_path, "r", encoding="utf-8") as f_in, \
         open(candidate_out_path, "w", encoding="utf-8", newline="\n") as f_cand, \
         open(matching_out_path, "w", encoding="utf-8", newline="\n") as f_match:

        f_in.readline()  # Skip input header
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.write("source1_entity_id\tmatched_entity_ids\n")

        for line in f_in:
            line_str = line.rstrip("\r\n")
            if not line_str:
                continue
            s1_id = line_str.split("\t")[0].strip()
            c_str = cand_results.get(s1_id, "")
            m_str = match_results.get(s1_id, "")

            f_cand.write(f"{s1_id}\t{c_str}\n")
            f_match.write(f"{s1_id}\t{m_str}\n")

    print(f"      Written {total_s1_processed:,} rows to:")
    print(f"      - {candidate_out_path}")
    print(f"      - {matching_out_path}")
    print(f"      File writing took {time.time() - t_write_start:.2f} seconds.")

    print(f"\nPipeline statistics:")
    print(f"  Total Source 1 entities:  {total_s1_processed:,}")
    print(f"  Total Candidates:         {total_candidates_found:,}")
    print(f"  Total Matches:            {total_matches_predicted:,}")
    print(f"  Max candidates per entity: {max_candidates_per_entity}")
    print(f"  Max matches per entity:    {max_matches_per_entity}")
    print(f"  Total Execution Time:     {time.time() - start_time:.2f} seconds")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Step 4: Run official validator
    # -----------------------------------------------------------------------
    print("\n[4/4] Executing official validation script...")
    val_cmd = [
        sys.executable,
        "utils/validate_submission.py",
        "--matching", matching_out_path,
        "--candidate", candidate_out_path,
        "--test-dir", test_dir,
    ]
    print(f"Running: {' '.join(val_cmd)}\n")
    proc = subprocess.run(val_cmd, capture_output=True, text=True)
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)

    if proc.returncode == 0:
        print(">>> SUCCESS: Official submission validator passed with exit code 0! <<<")
    else:
        print(f">>> WARNING: Validator failed with exit code {proc.returncode} <<<")

    return proc.returncode


if __name__ == "__main__":
    run_fast_matcher()
