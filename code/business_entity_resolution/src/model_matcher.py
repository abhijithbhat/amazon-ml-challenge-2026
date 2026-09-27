#!/usr/bin/env python3
"""
=============================================================================
src/model_matcher.py  -  HIGH-PRECISION TEST INFERENCE PIPELINE
=============================================================================
Amazon ML Challenge 2026: Multi-Source Business Entity Resolution

Pipeline Architecture:
1. S1 Inverted Index:
   - Normalizes test_source1 records (clean_name, clean_address)
   - Builds country-partitioned blocking index using get_blocking_keys
2. Streaming Vendor Scoring:
   - Streams test_source2.tsv and test_source3.tsv in 100k-row chunks
   - Matches vendor blocking keys against S1 blocking buckets within the same country
   - Extracts 10-D pairwise similarity features via compute_pair_features
   - Scores candidate pairs in batches via model.predict_proba(X)[:, 1]
3. Greedy 1-to-1 Assignment & Singleton Gating:
   - Only pairs with predicted probability >= optimal_threshold are considered
   - Enforces 1-to-1 assignment: each S2 or S3 entity can belong to at most
     ONE S1 entity (highest probability wins)
   - S1 entities without candidates clearing the threshold remain empty singletons (Score = 1.0)
4. Output Generation:
   - output/matching_results.tsv (scored on official leaderboard)
   - output/candidate_pairs.tsv (blocking candidates, superset of matches)
5. Verification:
   - Executes utils/validate_submission.py to guarantee submission passes.
=============================================================================
"""

import argparse
import io
import os
import pickle
import subprocess
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Force unbuffered UTF-8 standard output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

# Robust imports supporting direct script execution and package execution
try:
    from normalization import normalize_name, normalize_address
    from blocking_features import get_blocking_keys
    from comparison_features import compute_pair_features, FEATURE_COLS
except ImportError:
    from src.normalization import normalize_name, normalize_address
    from src.blocking_features import get_blocking_keys
    from src.comparison_features import compute_pair_features, FEATURE_COLS


def load_model_bundle(model_path: Optional[str] = None) -> Tuple[object, float, List[str]]:
    """
    Load trained model checkpoint bundle.
    Supports trained_model.pkl, model.pkl, and output/trained_model.pkl.
    """
    candidates = [
        model_path,
        "trained_model.pkl",
        "model.pkl",
        "output/trained_model.pkl",
    ]
    selected_path = None
    for p in candidates:
        if p and os.path.isfile(p):
            selected_path = p
            break

    if not selected_path:
        raise FileNotFoundError(
            f"No trained model found! Looked in: {[c for c in candidates if c]}"
        )

    print(f"Loading trained model bundle from: {selected_path}")
    with open(selected_path, "rb") as f:
        bundle = pickle.load(f)

    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"Invalid model bundle format in {selected_path}")

    model = bundle["model"]
    optimal_threshold = float(bundle.get("optimal_threshold", 0.85))
    feature_cols = bundle.get("feature_cols", FEATURE_COLS)

    print(f"  - Model Architecture: {type(model).__name__}")
    print(f"  - Decision Threshold: {optimal_threshold:.2f}")
    print(f"  - Feature Columns ({len(feature_cols)}): {feature_cols}")
    if "precision" in bundle and "f05" in bundle:
        print(f"  - Training Precision: {bundle['precision']*100:.2f}%, F_0.5: {bundle['f05']:.4f}")

    return model, optimal_threshold, feature_cols


def run_test_inference(
    test_dir: str = "dataset/test",
    output_dir: str = "output",
    model_path: Optional[str] = None,
    threshold_override: Optional[float] = None,
    sample_size: Optional[int] = None,
    chunk_size: int = 100_000,
    max_bucket_size: int = 25,
    max_candidates_per_s1: int = 10,
    inference_batch_size: int = 25_000,
    skip_validator: bool = False,
):
    """
    Execute high-precision test inference pipeline end-to-end.
    """
    t_global_start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    for path, name in [(s1_path, "test_source1.tsv"), (s2_path, "test_source2.tsv"), (s3_path, "test_source3.tsv")]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required test dataset file not found: {path}")

    # ── 1. Load Trained Model ────────────────────────────────────────────────
    print("=" * 76)
    print("  STAGE 1: LOADING TRAINED MODEL & HYPERPARAMETERS")
    print("=" * 76)
    model, optimal_threshold, feature_cols = load_model_bundle(model_path)
    if threshold_override is not None:
        print(f"  * Overriding decision threshold to: {threshold_override:.2f}")
        optimal_threshold = threshold_override

    # ── 2. Build S1 Reference Inverted Index ──────────────────────────────────
    print("\n" + "=" * 76)
    print("  STAGE 2: BUILDING SOURCE 1 NORMALIZED INVERTED BLOCKING INDEX")
    print("=" * 76)
    t0 = time.time()

    print(f"Loading all S1 reference entity IDs from: {s1_path}...")
    s1_ordered_ids: List[str] = []
    with open(s1_path, "r", encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            line_str = line.rstrip("\r\n")
            if not line_str:
                continue
            s1_id = line_str.split("\t", 1)[0].strip()
            if s1_id:
                s1_ordered_ids.append(s1_id)
    print(f"  Loaded {len(s1_ordered_ids):,} total S1 entity IDs.")

    # Compact lookup: eid -> (clean_name, clean_address, country)
    s1_lookup: Dict[str, Tuple[str, str, str]] = {}
    # Country-partitioned blocking index: country -> block_key -> list of s1_ids
    s1_block_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

    total_s1_records = 0
    active_s1_ids: Set[str] = set()

    for chunk in pd.read_csv(s1_path, sep="\t", chunksize=chunk_size, dtype=str):
        chunk = chunk.fillna("")
        eids = chunk["entity_id"].tolist()
        names = chunk["business_name"].tolist()
        addrs = chunk["business_address"].tolist()
        countries = chunk["country"].tolist()

        # Normalize names and addresses
        norm_names = [normalize_name(n, c) for n, c in zip(names, countries)]
        norm_addrs = [normalize_address(a, c) for a, c in zip(addrs, countries)]

        for eid, nc, ac, c in zip(eids, norm_names, norm_addrs, countries):
            total_s1_records += 1
            if sample_size and total_s1_records > sample_size:
                continue

            active_s1_ids.add(eid)
            s1_lookup[eid] = (nc, ac, c)

            # Generate multi-pass blocking keys
            row_dict = {"country": c, "name_clean": nc, "address_clean": ac}
            keys = get_blocking_keys(row_dict)
            c_buckets = s1_block_index[c]
            for k in keys:
                c_buckets[k].append(eid)

        if total_s1_records % 500_000 == 0 or (sample_size and total_s1_records >= sample_size):
            print(f"    ... processed {total_s1_records:,} S1 rows ({time.time()-t0:.1f}s)")
        if sample_size and total_s1_records >= sample_size:
            print(f"    (Sample limit of {sample_size:,} active S1 entities reached)")
            break

    total_keys = sum(len(buckets) for buckets in s1_block_index.values())
    print(f"  Done in {time.time()-t0:.1f}s:")
    print(f"    - Total S1 entities loaded: {len(s1_ordered_ids):,}")
    print(f"    - Active indexed S1 entities: {len(active_s1_ids):,}")
    print(f"    - Partitioned countries: {list(s1_block_index.keys())}")
    print(f"    - Total blocking index buckets: {total_keys:,}")

    # ── 3. Stream Vendor Sources & Perform Greedy Matching ───────────────────
    print("\n" + "=" * 76)
    print("  STAGE 3: STREAMING VENDOR SOURCES & GREEDY 1-TO-1 SCORING")
    print("=" * 76)

    # candidates_by_s1: s1_id -> set of candidate vendor IDs
    candidates_by_s1: Dict[str, Set[str]] = defaultdict(set)

    # best_s1_for_vendor: vx_eid -> (best_s1_eid, max_prob)
    # Strictly enforces 1-to-1 constraint: each vendor record belongs to at most ONE S1 entity
    best_s1_for_vendor: Dict[str, Tuple[str, float]] = {}

    total_pairs_scored = 0
    total_candidates_found = 0

    def process_vendor_file(vendor_path: str, vendor_label: str):
        nonlocal total_pairs_scored, total_candidates_found
        t_vendor_start = time.time()
        print(f"\n  Starting stream: {vendor_label} ({os.path.basename(vendor_path)})...")

        chunk_idx = 0
        vendor_records_seen = 0
        vendor_pairs_scored = 0
        vendor_matches_retained = 0

        # Feature batch accumulators
        batch_pairs: List[Tuple[str, str]] = []
        batch_features: List[List[float]] = []

        def flush_batch():
            nonlocal batch_pairs, batch_features, vendor_pairs_scored, total_pairs_scored, vendor_matches_retained
            if not batch_features:
                return

            X = np.asarray(batch_features, dtype=np.float32)
            probs = model.predict_proba(X)[:, 1]
            scored_count = len(probs)
            vendor_pairs_scored += scored_count
            total_pairs_scored += scored_count

            for (s1_eid, vx_eid), prob in zip(batch_pairs, probs):
                if prob >= optimal_threshold:
                    # Greedy 1-to-1 assignment: highest probability wins
                    curr = best_s1_for_vendor.get(vx_eid)
                    if curr is None or prob > curr[1]:
                        best_s1_for_vendor[vx_eid] = (s1_eid, float(prob))
                        vendor_matches_retained += 1

            batch_pairs.clear()
            batch_features.clear()

        for chunk in pd.read_csv(vendor_path, sep="\t", chunksize=chunk_size, dtype=str):
            chunk_idx += 1
            chunk = chunk.fillna("")

            v_eids = chunk["entity_id"].tolist()
            v_names = chunk["business_name"].tolist()
            v_addrs = chunk["business_address"].tolist()
            v_countries = chunk["country"].tolist()
            vendor_records_seen += len(v_eids)

            # Fast list normalization
            norm_names = [normalize_name(n, c) for n, c in zip(v_names, v_countries)]
            norm_addrs = [normalize_address(a, c) for a, c in zip(v_addrs, v_countries)]

            for vx_eid, nc, ac, c in zip(v_eids, norm_names, norm_addrs, v_countries):
                c_buckets = s1_block_index.get(c)
                if not c_buckets:
                    continue

                row_b = {"entity_id": vx_eid, "name_clean": nc, "address_clean": ac, "country": c}
                keys = get_blocking_keys(row_b)

                hit_s1: Set[str] = set()
                for k in keys:
                    if k in c_buckets:
                        bucket = c_buckets[k]
                        if len(bucket) <= max_bucket_size:
                            hit_s1.update(bucket)

                if not hit_s1:
                    continue

                for s1_eid in hit_s1:
                    # Adaptive candidate cap: keep at most max_candidates_per_s1 per S1 entity
                    cands = candidates_by_s1[s1_eid]
                    if len(cands) < max_candidates_per_s1:
                        cands.add(vx_eid)
                        total_candidates_found += 1

                    # Compute pairwise features
                    s1_nc, s1_ac, s1_c = s1_lookup[s1_eid]
                    row_a = {"entity_id": s1_eid, "name_clean": s1_nc, "address_clean": s1_ac, "country": s1_c}

                    feat_dict = compute_pair_features(row_a, row_b)
                    feat_vec = [feat_dict[col] for col in feature_cols]

                    batch_pairs.append((s1_eid, vx_eid))
                    batch_features.append(feat_vec)

                    if len(batch_features) >= inference_batch_size:
                        flush_batch()

            # Periodic progress update every chunk
            elapsed = time.time() - t_vendor_start
            rate = vendor_records_seen / elapsed if elapsed > 0 else 0
            print(
                f"    ... chunk {chunk_idx:>2}: {vendor_records_seen:,} rows processed "
                f"({rate:.0f} rows/s) | Scored: {vendor_pairs_scored:,} pairs | Assigned: {len(best_s1_for_vendor):,}"
            )

            # In sample mode, stop after scanning enough chunks to verify candidate extraction
            if sample_size and chunk_idx >= 5:
                print(f"    (Sample mode: stopping after {chunk_idx} chunks for verification)")
                break

        # Flush any remaining pairs in buffer
        flush_batch()

        t_vendor_elapsed = time.time() - t_vendor_start
        print(
            f"  {vendor_label} Complete in {t_vendor_elapsed:.1f}s:\n"
            f"    - Total records streamed: {vendor_records_seen:,}\n"
            f"    - Total pairs scored:     {vendor_pairs_scored:,}\n"
            f"    - Running 1-to-1 matches: {len(best_s1_for_vendor):,}"
        )

    # Stream Source 2 and Source 3
    process_vendor_file(s2_path, "Source 2")
    process_vendor_file(s3_path, "Source 3")

    # ── 4. Invert 1-to-1 Vendor Assignments to S1 Match Lists ────────────────
    print("\n" + "=" * 76)
    print("  STAGE 4: COMPILING GREEDY 1-TO-1 MATCHES & SINGLETONS")
    print("=" * 76)
    matches_by_s1: Dict[str, List[str]] = defaultdict(list)
    for vx_eid, (s1_eid, prob) in best_s1_for_vendor.items():
        matches_by_s1[s1_eid].append(vx_eid)

    s1_with_matches = len(matches_by_s1)
    total_test_s1 = len(s1_ordered_ids)
    singletons = total_test_s1 - s1_with_matches

    print(f"  Total Test S1 Entities:     {total_test_s1:,}")
    print(f"  S1 Entities with Matches:   {s1_with_matches:,} ({s1_with_matches / total_test_s1 * 100:.2f}%)")
    print(f"  S1 Singletons (no matches): {singletons:,} ({singletons / total_test_s1 * 100:.2f}%)")
    print(f"  Total Assigned Vendor IDs:  {len(best_s1_for_vendor):,}")

    # ── 5. Write Submission Files ────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("  STAGE 5: GENERATING OFFICIAL SUBMISSION TSVs")
    print("=" * 76)
    matching_file = os.path.join(output_dir, "matching_results.tsv")
    candidate_file = os.path.join(output_dir, "candidate_pairs.tsv")
    t_write = time.time()

    print(f"Writing matching results to:  {matching_file}")
    with open(matching_file, "w", encoding="utf-8") as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_ordered_ids:
            matched_list = matches_by_s1.get(s1_id, [])
            if matched_list:
                matched_str = ",".join(sorted(set(matched_list)))
            else:
                matched_str = ""
            f_match.write(f"{s1_id}\t{matched_str}\n")

    print(f"Writing candidate pairs to:   {candidate_file}")
    with open(candidate_file, "w", encoding="utf-8") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_ordered_ids:
            cands = set(candidates_by_s1.get(s1_id, set()))
            # Mandatory rule: every matched ID must appear in candidates
            matched = set(matches_by_s1.get(s1_id, []))
            all_cands = cands | matched
            if all_cands:
                cand_str = ",".join(sorted(all_cands))
            else:
                cand_str = ""
            f_cand.write(f"{s1_id}\t{cand_str}\n")

    print(f"Submission files successfully written in {time.time()-t_write:.2f}s.")

    # ── 6. Official Submission Verification ──────────────────────────────────
    if skip_validator:
        print("\n  [Skipping official submission validator as requested]")
    else:
        print("\n" + "=" * 76)
        print("  STAGE 6: OFFICIAL SUBMISSION VALIDATOR")
        print("=" * 76)
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
            print(f"Executing: {' '.join(cmd)}\n")
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.stdout:
                print(res.stdout)
            if res.stderr:
                print(res.stderr, file=sys.stderr)

            if res.returncode == 0:
                print("\n  >>> SUBMISSION INTEGRITY: ALL VALIDATION CHECKS PASSED (EXIT 0) <<<")
            else:
                print(f"\n  Validator returned non-zero exit code: {res.returncode}", file=sys.stderr)
                sys.exit(res.returncode)

    print("\n" + "=" * 76)
    print(f"  PIPELINE EXECUTION COMPLETE (Total: {time.time()-t_global_start:.1f}s)")
    print("=" * 76)


def main():
    parser = argparse.ArgumentParser(
        description="High-Precision Test Inference Pipeline (Amazon ML Challenge 2026)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--test-dir",
        default="dataset/test",
        help="Directory containing test_source1.tsv, test_source2.tsv, and test_source3.tsv",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to save matching_results.tsv and candidate_pairs.tsv",
    )
    parser.add_argument(
        "--model",
        default="trained_model.pkl",
        help="Path to trained model pickle bundle (or model.pkl)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional override for classifier probability decision threshold",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Quick verification sample: number of active S1 entities to score (e.g. 5000)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run inference over all 1,732,544 test S1 entities",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Chunk size for streaming vendor source TSVs",
    )
    parser.add_argument(
        "--max-bucket-size",
        type=int,
        default=25,
        help="Maximum size of blocking bucket to prevent runaway generic token explosions",
    )
    parser.add_argument(
        "--max-candidates-per-s1",
        type=int,
        default=10,
        help="Adaptive candidate cap: keep at most N candidate IDs per Source 1 entity",
    )
    parser.add_argument(
        "--skip-validator",
        action="store_true",
        help="Skip running utils/validate_submission.py at completion",
    )

    args = parser.parse_args()

    sample_val = None if args.full else args.sample

    run_test_inference(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_path=args.model,
        threshold_override=args.threshold,
        sample_size=sample_val,
        chunk_size=args.chunk_size,
        max_bucket_size=args.max_bucket_size,
        max_candidates_per_s1=args.max_candidates_per_s1,
        skip_validator=args.skip_validator,
    )


if __name__ == "__main__":
    main()
