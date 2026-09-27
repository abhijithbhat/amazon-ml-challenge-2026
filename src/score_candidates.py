#!/usr/bin/env python3
"""
=============================================================================
src/score_candidates.py  -  VERIFIED 0.708 DIRECT CANDIDATE SCORER
=============================================================================
Amazon ML Challenge 2026: Multi-Source Business Entity Resolution

Verified Architecture & Decision Rules (Leaderboard Macro F0.5 = 0.708):
1. Candidate Loading / Streaming:
   - Loads pre-computed scored candidate pairs from cache (output/scored_pairs_cache.pkl)
     or streams model inference across output/candidate_pairs.tsv.
2. 3-Tier Cascaded Evaluation:
   - Tier 1: Fast C-level RapidFuzz quick-ratio pre-filter (<45 rejected unless confirmed digits).
   - Tier 2: Restricted exact-anchor bypass (prob = 1.0) requiring:
       * s1_nc == v_nc
       * len(s1_nc) >= 6 (prevents common short acronym over-merges)
       * addr_jw >= 0.80
       * non-empty significant digit overlap
   - Tier 3: 10-dimensional pairwise comparison feature extraction scored with trained
     Random Forest classifier.
3. Decision Rules:
   - Claude's 3-Way Digit Conflict Guard:
     * 'confirm':  Shared significant digits -> prob >= 0.78 (US/India) / 0.88 (France)
     * 'absent':   Landmark address with no digits -> prob >= 0.86 (US/India) / 0.92 (France)
     * 'conflict': Contradictory street numbers -> STRICTLY SUPPRESSED (prob = 0.0)
   - Preserves complete multi-vendor clusters (no artificial tail pruning or delta gaps).
4. Global Greedy 1-to-1 Disjoint Assignment:
   - Sorts surviving pairs globally by probability descending.
   - Greedily assigns each vendor record to at most ONE Source 1 entity.
5. TSV Generation & Verification:
   - Writes output/matching_results.tsv in exact S1 sequence (1,732,544 rows).
   - Singletons are emitted with empty strings ("").
   - Automatically executes utils/validate_submission.py with --check-ids.
=============================================================================
"""

import argparse
import io
import os
import pickle
import re
import subprocess
import sys
import time
import unicodedata
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Force unbuffered UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

# Fast C-level similarity via RapidFuzz
try:
    from rapidfuzz.fuzz import quick_ratio as _rf_quick_ratio
except (ImportError, AttributeError):
    try:
        from rapidfuzz.fuzz import QRatio as _rf_quick_ratio
    except (ImportError, AttributeError):
        from rapidfuzz.fuzz import ratio as _rf_quick_ratio

try:
    from rapidfuzz.distance.JaroWinkler import similarity as _rf_jaro_winkler
except (ImportError, AttributeError):
    try:
        from rapidfuzz.distance import jaro_winkler as _rf_jw_mod
        _rf_jaro_winkler = getattr(_rf_jw_mod, "similarity", None)
    except (ImportError, AttributeError):
        _rf_jaro_winkler = None

# Domain normalization and pairwise feature extraction
try:
    from normalization import normalize_name, normalize_address
    from comparison_features import compute_pair_features, FEATURE_COLS, jaro_winkler_similarity
except ImportError:
    from src.normalization import normalize_name, normalize_address
    from src.comparison_features import compute_pair_features, FEATURE_COLS, jaro_winkler_similarity

_DIGIT_RE = re.compile(r"\d+")


def calc_addr_jw(s1: str, s2: str) -> float:
    """Fast Jaro-Winkler string similarity in [0.0, 1.0]."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if _rf_jaro_winkler is not None:
        return float(_rf_jaro_winkler(s1, s2))
    return float(jaro_winkler_similarity(s1, s2))


def extract_significant_digits(text: str) -> Set[str]:
    """Extract significant digits (house numbers, postal codes, unit numbers).
    Filters out trivial single digits ('0', '1', '2') to prevent false matches."""
    if not text:
        return set()
    raw = set(_DIGIT_RE.findall(text))
    return {d for d in raw if d not in {"0", "1", "2"}}


def get_digit_status(digits_a: Set[str], digits_b: Set[str]) -> str:
    """
    Claude's 3-way digit state for address comparison:
      - 'absent':   Landmark address with no digits — score normally with model
      - 'confirm':  Shared significant digits (anchors the address)
      - 'conflict': Both have digits, but zero overlap (different house/street numbers)
    """
    if not digits_a or not digits_b:
        return "absent"
    if digits_a & digits_b:
        return "confirm"
    return "conflict"


# ── Accent-Folding & French Legal Suffix Cleaning ────────────────────────────
_FRENCH_LEGAL_SUFFIXES = re.compile(
    r"\b(?:sarl|sas|sasu|sa|eurl|sci|snc|earl|gaec|gie|scop|scic)\b"
)


def fold_accents_and_clean(text: str) -> str:
    """
    Fast accent-folding normalizer applied to all name and address strings
    before pairwise feature extraction.
    """
    if not text:
        return text
    nfkd = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in nfkd if unicodedata.category(ch) != "Mn")
    stripped = _FRENCH_LEGAL_SUFFIXES.sub("", stripped)
    return re.sub(r"\s+", " ", stripped).strip()


def is_tight_country(country: str) -> bool:
    """Check if country belongs to French/Belgian open set requiring tighter thresholds."""
    if not country:
        return False
    c_up = str(country).upper().strip()
    return c_up in ("FRANCE", "FR", "BE", "BELGIUM") or "FR" in c_up


def passes_threshold(prob: float, digit_status: str, country: str, threshold: float = 0.78, fallback_threshold: float = 0.86) -> bool:
    """
    Empirically verified optimal decision boundary (Val Macro F0.5 = 0.9538):
    - 'conflict': Strictly suppressed to 0.0 (prevents chain store false merges)
    - France: prob >= 0.88 ('confirm') or prob >= 0.92 ('absent')
    - US/India: prob >= threshold (0.78) or prob >= fallback_threshold (0.86)
    """
    if digit_status == "conflict":
        return False

    is_tight = is_tight_country(country)
    eff_confirm = 0.88 if is_tight else threshold
    eff_absent = 0.92 if is_tight else fallback_threshold

    if digit_status == "confirm":
        return prob >= eff_confirm
    elif digit_status == "absent":
        return prob >= eff_absent
    return False


def load_model_bundle(model_path: Optional[str] = None) -> Tuple[object, float, List[str]]:
    """Load trained model checkpoint bundle."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        model_path,
        "trained_model.pkl",
        "model.pkl",
        "output/trained_model.pkl",
        os.path.join(base_dir, "output", "trained_model.pkl"),
        os.path.join(base_dir, "trained_model.pkl"),
    ]
    selected_path = None
    for p in candidates:
        if p and os.path.isfile(p):
            selected_path = p
            break

    if not selected_path:
        raise FileNotFoundError(f"No trained model found! Looked in: {[c for c in candidates if c]}")

    print(f"Loading trained model bundle from: {selected_path}")
    with open(selected_path, "rb") as f:
        bundle = pickle.load(f)

    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"Invalid model bundle format in {selected_path}")

    model = bundle["model"]
    optimal_threshold = float(bundle.get("optimal_threshold", 0.83))
    feature_cols = bundle.get("feature_cols", FEATURE_COLS)

    if hasattr(model, "n_jobs"):
        model.n_jobs = -1

    print(f"  - Model Architecture: {type(model).__name__}")
    print(f"  - Feature Columns ({len(feature_cols)}): {feature_cols}")
    if "precision" in bundle and "f05" in bundle:
        print(f"  - Training Precision: {bundle['precision']*100:.2f}%, F_0.5: {bundle['f05']:.4f}")

    return model, optimal_threshold, feature_cols


def run_direct_scoring(
    candidate_file: str = "output/candidate_pairs.tsv",
    matching_file: str = "output/matching_results.tsv",
    test_dir: str = "dataset/test",
    model_path: Optional[str] = None,
    threshold: float = 0.78,
    fallback_threshold: float = 0.86,
    batch_size: int = 25_000,
    cache_file: Optional[str] = "output/scored_pairs_cache.pkl",
    skip_validator: bool = False,
    check_ids: bool = True,
):
    t_global_start = time.time()
    print("=" * 76)
    print("  VERIFIED 0.708 DIRECT CANDIDATE SCORER")
    print("=" * 76)

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    for path in [s1_path, s2_path, s3_path]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required test dataset not found: {path}")

    # Read S1 ordered IDs from test_source1.tsv
    print(f"Reading reference S1 entity IDs from: {s1_path}...")
    s1_ordered_ids: List[str] = []
    with open(s1_path, "r", encoding="utf-8") as f:
        f.readline()  # header
        for line in f:
            if line.strip():
                s1_ordered_ids.append(line.split("\t", 1)[0].strip())
    print(f"  Total Reference Source 1 Entities: {len(s1_ordered_ids):,}")

    all_scored_pairs: List[Tuple[float, str, str, str, str]] = []

    # ── Fast-Path: Load pre-computed scored candidate pairs from cache ────────
    use_cache = bool(cache_file and os.path.isfile(cache_file))
    if use_cache:
        print("\n" + "=" * 76)
        print("  STAGE 1-4: LOADING PRE-COMPUTED SCORED PAIRS CACHE")
        print("=" * 76)
        t0 = time.time()
        print(f"Loading cached scored candidate pairs from: {cache_file}...")
        with open(cache_file, "rb") as f:
            all_scored_pairs = pickle.load(f)
        print(f"  Loaded {len(all_scored_pairs):,} scored candidate pairs in {time.time()-t0:.2f}s.")
    else:
        if not os.path.isfile(candidate_file):
            raise FileNotFoundError(f"Candidate file not found: {candidate_file}")

        # ── 1. Load Trained Model Bundle ─────────────────────────────────────
        model, _, feature_cols = load_model_bundle(model_path)

        # ── 2. Scan Candidate Pairs & Collect Needed Vendor IDs ──────────────
        print("\n" + "=" * 76)
        print("  STAGE 1: SCANNING CANDIDATE PAIRS & EXTRACTING VENDOR IDs")
        print("=" * 76)
        t0 = time.time()

        needed_vendor_ids: Set[str] = set()
        total_pairs_count = 0
        s1_candidates_map: List[Tuple[str, List[str]]] = []

        print(f"Reading candidate pairs from: {candidate_file}...")
        with open(candidate_file, "r", encoding="utf-8") as f:
            f.readline()  # header
            for line in f:
                line_str = line.rstrip("\r\n")
                if not line_str:
                    continue
                parts = line_str.split("\t", 1)
                s1_id = parts[0].strip()
                cands_str = parts[1].strip() if len(parts) > 1 else ""

                if cands_str:
                    cand_list = [c.strip() for c in cands_str.split(",") if c.strip()]
                    needed_vendor_ids.update(cand_list)
                    total_pairs_count += len(cand_list)
                    s1_candidates_map.append((s1_id, cand_list))
                else:
                    s1_candidates_map.append((s1_id, []))

        print(f"  Done in {time.time()-t0:.2f}s:")
        print(f"    - Total S1 Entities in candidate file: {len(s1_candidates_map):,}")
        print(f"    - Total Candidate Pairs to evaluate:  {total_pairs_count:,}")
        print(f"    - Unique Vendor IDs required:         {len(needed_vendor_ids):,}")

        # ── 3. Load S1 Metadata ──────────────────────────────────────────────
        print("\n" + "=" * 76)
        print("  STAGE 2: LOADING SOURCE 1 REFERENCE METADATA")
        print("=" * 76)
        t0 = time.time()

        s1_lookup: Dict[str, Tuple[str, str, str, Set[str]]] = {}
        for chunk in pd.read_csv(s1_path, sep="\t", chunksize=100_000, dtype=str):
            chunk = chunk.fillna("")
            eids = chunk["entity_id"].tolist()
            names = chunk["business_name"].tolist()
            addrs = chunk["business_address"].tolist()
            countries = chunk["country"].tolist()

            norm_names = [fold_accents_and_clean(normalize_name(n, c)) for n, c in zip(names, countries)]
            norm_addrs = [fold_accents_and_clean(normalize_address(a, c)) for a, c in zip(addrs, countries)]

            for eid, nc, ac, c in zip(eids, norm_names, norm_addrs, countries):
                digits = extract_significant_digits(ac)
                s1_lookup[eid] = (nc, ac, c.strip().upper(), digits)

        print(f"  Loaded {len(s1_lookup):,} Source 1 records in {time.time()-t0:.2f}s.")

        # ── 4. Load Filtered Vendor Records ──────────────────────────────────
        print("\n" + "=" * 76)
        print("  STAGE 3: LOADING FILTERED VENDOR METADATA")
        print("=" * 76)
        t0 = time.time()

        vendor_lookup: Dict[str, Tuple[str, str, str, Set[str]]] = {}

        def load_vendor_source(file_path: str, label: str):
            t_v = time.time()
            matched_records = 0
            total_scanned = 0
            print(f"Filtering {label} ({os.path.basename(file_path)})...")

            for chunk in pd.read_csv(file_path, sep="\t", chunksize=100_000, dtype=str):
                chunk = chunk.fillna("")
                total_scanned += len(chunk)

                mask = chunk["entity_id"].isin(needed_vendor_ids)
                filtered = chunk[mask]
                if filtered.empty:
                    continue

                matched_records += len(filtered)
                v_eids = filtered["entity_id"].tolist()
                v_names = filtered["business_name"].tolist()
                v_addrs = filtered["business_address"].tolist()
                v_countries = filtered["country"].tolist()

                norm_names = [fold_accents_and_clean(normalize_name(n, c)) for n, c in zip(v_names, v_countries)]
                norm_addrs = [fold_accents_and_clean(normalize_address(a, c)) for a, c in zip(v_addrs, v_countries)]

                for eid, nc, ac, c in zip(v_eids, norm_names, norm_addrs, v_countries):
                    digits = extract_significant_digits(ac)
                    vendor_lookup[eid] = (nc, ac, c.strip().upper(), digits)

            print(f"  - {label}: Retained {matched_records:,} / {total_scanned:,} rows in {time.time()-t_v:.2f}s.")

        load_vendor_source(s2_path, "Source 2")
        load_vendor_source(s3_path, "Source 3")
        print(f"  Total Active Vendor Lookup Records: {len(vendor_lookup):,} in {time.time()-t0:.2f}s.")

        # ── 5. 3-Tier Cascaded Evaluation & Batch Model Scoring ──────────────
        print("\n" + "=" * 76)
        print("  STAGE 4: 3-TIER CASCADED EVALUATION & BATCH MODEL SCORING")
        print("=" * 76)
        t0 = time.time()

        pairs_evaluated = 0
        tier1_rejected = 0
        tier2_exact_bypassed = 0
        tier3_model_scored = 0

        batch_s1: List[str] = []
        batch_vx: List[str] = []
        batch_digit_status: List[str] = []
        batch_s1_country: List[str] = []
        batch_features: List[List[float]] = []

        def flush_batch():
            nonlocal tier3_model_scored
            if not batch_features:
                return

            X = np.asarray(batch_features, dtype=np.float32)
            probs = model.predict_proba(X)[:, 1]
            tier3_model_scored += len(probs)

            for s1, vx, d_status, prob, s1c in zip(
                batch_s1, batch_vx, batch_digit_status, probs, batch_s1_country
            ):
                prob = float(prob)
                all_scored_pairs.append((prob, s1, vx, d_status, s1c))

            batch_s1.clear()
            batch_vx.clear()
            batch_digit_status.clear()
            batch_s1_country.clear()
            batch_features.clear()

        last_log_time = time.time()

        for s1_id, cand_list in s1_candidates_map:
            if not cand_list:
                continue

            s1_info = s1_lookup.get(s1_id)
            if not s1_info:
                continue
            s1_nc, s1_ac, s1_country, s1_digits = s1_info

            for vx_id in cand_list:
                pairs_evaluated += 1
                v_info = vendor_lookup.get(vx_id)
                if not v_info:
                    continue
                v_nc, v_ac, v_country, v_digits = v_info

                digit_status = get_digit_status(s1_digits, v_digits)

                # Tier 1: Fast C-Level Pre-Filter Gate
                sim_score = _rf_quick_ratio(s1_nc, v_nc)
                if sim_score < 45.0 and digit_status != "confirm":
                    tier1_rejected += 1
                    continue

                # Tier 2: Restricted Exact-Anchor Bypass
                if (
                    s1_nc == v_nc
                    and len(s1_nc) >= 6
                    and bool(s1_digits & v_digits)
                    and calc_addr_jw(s1_ac, v_ac) >= 0.80
                ):
                    tier2_exact_bypassed += 1
                    all_scored_pairs.append((1.0, s1_id, vx_id, "confirm", s1_country))
                    continue

                # Tier 3: Pairwise Features & Model Scoring
                row_a = {"entity_id": s1_id, "name_clean": s1_nc, "address_clean": s1_ac, "country": s1_country}
                row_b = {"entity_id": vx_id, "name_clean": v_nc, "address_clean": v_ac, "country": v_country}

                feat_dict = compute_pair_features(row_a, row_b)
                feat_vec = [feat_dict[c] for c in feature_cols]

                batch_s1.append(s1_id)
                batch_vx.append(vx_id)
                batch_digit_status.append(digit_status)
                batch_s1_country.append(s1_country)
                batch_features.append(feat_vec)

                if len(batch_features) >= batch_size:
                    flush_batch()

            if time.time() - last_log_time >= 10.0:
                last_log_time = time.time()
                pct = (pairs_evaluated / total_pairs_count) * 100 if total_pairs_count > 0 else 0.0
                print(
                    f"    ... evaluated {pairs_evaluated:,} / {total_pairs_count:,} pairs ({pct:.1f}%) | "
                    f"Model Scored: {tier3_model_scored:,} | "
                    f"Scored Pairs: {len(all_scored_pairs):,}"
                )

        flush_batch()

        t_eval = time.time() - t0
        rate = pairs_evaluated / t_eval if t_eval > 0 else 0
        print(f"\n  Candidate Evaluation Complete in {t_eval:.2f}s ({rate:.0f} pairs/s):")
        print(f"    - Total Pairs Evaluated:         {pairs_evaluated:,}")
        print(f"    - Tier 1 Pre-Filter Rejected:    {tier1_rejected:,}")
        print(f"    - Tier 2 Exact Matches Bypassed: {tier2_exact_bypassed:,}")
        print(f"    - Tier 3 Pairs Model Scored:     {tier3_model_scored:,}")
        print(f"    - Total Scored Pairs Collected:  {len(all_scored_pairs):,}")

        # Dump cache if path specified
        if cache_file:
            print(f"\nDumping intermediate scored pairs cache to: {cache_file}...")
            os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(all_scored_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ── 6. Candidate Filtering via 0.708 Baseline Rules ───────────────────────
    print("\n" + "=" * 76)
    print("  STAGE 5: FILTERING CANDIDATES (VERIFIED 0.708 THRESHOLDS)")
    print("=" * 76)
    t0 = time.time()

    surviving_pairs: List[Tuple[float, str, str]] = []
    conflict_suppressed_count = 0

    for prob, s1, vx, d_stat, country in all_scored_pairs:
        if d_stat == "conflict":
            conflict_suppressed_count += 1
            continue

        if passes_threshold(prob, d_stat, country, threshold=threshold, fallback_threshold=fallback_threshold):
            surviving_pairs.append((prob, s1, vx))

    print(f"  - Conflicting Street Digits Suppressed: {conflict_suppressed_count:,}")
    print(f"  - Pairs Surviving Decision Thresholds:  {len(surviving_pairs):,} in {time.time()-t0:.2f}s.")

    # ── 7. Global Greedy 1-to-1 Disjoint Assignment ──────────────────────────
    print("\n" + "=" * 76)
    print("  STAGE 6: GLOBAL GREEDY 1-TO-1 DISJOINT ASSIGNMENT")
    print("=" * 76)
    t0 = time.time()

    surviving_pairs.sort(key=lambda x: x[0], reverse=True)

    assigned_vendors: Set[str] = set()
    matches_by_s1: Dict[str, List[str]] = defaultdict(list)

    for prob, s1_id, vx_id in surviving_pairs:
        if vx_id in assigned_vendors:
            continue
        assigned_vendors.add(vx_id)
        matches_by_s1[s1_id].append(vx_id)

    total_s1 = len(s1_ordered_ids)
    matched_s1 = len(matches_by_s1)
    singletons = total_s1 - matched_s1
    total_assigned_vendors = len(assigned_vendors)

    print(f"  Assignment Complete in {time.time()-t0:.2f}s:")
    print(f"    - Total Rows:                           {total_s1:,}")
    print(f"    - S1 Entities with Matches:             {matched_s1:,} ({matched_s1/total_s1*100:.2f}%)")
    print(f"    - Predicted Singletons (empty):         {singletons:,} ({singletons/total_s1*100:.2f}%)")
    print(f"    - Total Matched Vendors Assigned:       {total_assigned_vendors:,}")

    # ── 8. Generate matching_results.tsv ─────────────────────────────────────
    print("\n" + "=" * 76)
    print("  STAGE 7: WRITING OFFICIAL SUBMISSION FILE")
    print("=" * 76)
    t0 = time.time()

    os.makedirs(os.path.dirname(matching_file) or ".", exist_ok=True)
    print(f"Writing matching results to: {matching_file}")

    with open(matching_file, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_ordered_ids:
            matched = matches_by_s1.get(s1_id, [])
            if matched:
                matched_str = ",".join(sorted(set(matched)))
            else:
                matched_str = ""
            f.write(f"{s1_id}\t{matched_str}\n")

    print(f"  Wrote {total_s1:,} lines to {matching_file} in {time.time()-t0:.2f}s.")

    # ── 9. Official Submission Verification ──────────────────────────────────
    if skip_validator:
        print("\n  [Skipping official submission validator as requested]")
    else:
        print("\n" + "=" * 76)
        print("  STAGE 8: OFFICIAL SUBMISSION VALIDATOR")
        print("=" * 76)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        validator_path = os.path.join(base_dir, "utils", "validate_submission.py")
        if not os.path.isfile(validator_path):
            validator_path = "utils/validate_submission.py"
        if not os.path.isfile(validator_path):
            print(f"  Warning: Validator script not found at {validator_path}")
        else:
            cmd = [
                sys.executable,
                validator_path,
                "--matching",
                matching_file,
                "--candidate",
                candidate_file,
                "--test-dir",
                test_dir,
            ]
            if check_ids:
                cmd.append("--check-ids")
            print(f"Executing: {' '.join(cmd)}\n")
            res = subprocess.run(cmd)
            if res.returncode == 0:
                print("\n  >>> SUBMISSION INTEGRITY: ALL VALIDATION CHECKS PASSED (EXIT 0) <<<")
            else:
                print(f"\n  Validator returned non-zero exit code: {res.returncode}", file=sys.stderr)
                sys.exit(res.returncode)

    t_total = time.time() - t_global_start
    print("\n" + "=" * 76)
    print(f"  DIRECT SCORER COMPLETE (Total: {t_total:.1f}s / {t_total/60:.1f} minutes)")
    print("=" * 76)


def main():
    parser = argparse.ArgumentParser(
        description="Verified 0.708 Direct Candidate Scorer (Amazon ML Challenge 2026)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--candidate-file",
        default="output/candidate_pairs.tsv",
        help="Path to pre-computed candidate_pairs.tsv",
    )
    parser.add_argument(
        "--matching-file",
        default="output/matching_results.tsv",
        help="Path to output matching_results.tsv",
    )
    parser.add_argument(
        "--test-dir",
        default="dataset/test",
        help="Directory containing test_source1.tsv, test_source2.tsv, and test_source3.tsv",
    )
    parser.add_argument(
        "--model",
        default="trained_model.pkl",
        help="Path to trained model pickle bundle (or model.pkl)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.78,
        help="Baseline threshold for digit-confirmed matches (US/India)",
    )
    parser.add_argument(
        "--fallback-threshold",
        type=float,
        default=0.86,
        help="Baseline threshold for landmark/no-digit matches (US/India)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25_000,
        help="Batch size for model feature prediction",
    )
    parser.add_argument(
        "--cache-file",
        default="output/scored_pairs_cache.pkl",
        help="Path to pre-computed scored candidate pairs pickle cache (set empty string to bypass cache)",
    )
    parser.add_argument(
        "--skip-validator",
        action="store_true",
        help="Skip running utils/validate_submission.py at completion",
    )
    parser.add_argument(
        "--check-ids",
        action="store_true",
        default=True,
        help="Pass --check-ids to utils/validate_submission.py",
    )

    args = parser.parse_args()

    cache_path = args.cache_file if args.cache_file and args.cache_file.strip() else None

    run_direct_scoring(
        candidate_file=args.candidate_file,
        matching_file=args.matching_file,
        test_dir=args.test_dir,
        model_path=args.model,
        threshold=args.threshold,
        fallback_threshold=args.fallback_threshold,
        batch_size=args.batch_size,
        cache_file=cache_path,
        skip_validator=args.skip_validator,
        check_ids=args.check_ids,
    )


if __name__ == "__main__":
    main()
