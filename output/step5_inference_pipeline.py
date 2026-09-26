"""
=============================================================================
step5_inference_pipeline.py  -  END-TO-END INFERENCE & SUBMISSION GENERATOR
=============================================================================
Role:  Data & Features (Pod 2, Member 3)
Task:  Apply normalization, blocking, and the trained matching classifier to
       produce the official competition submission files:
         - matching_results.tsv (scored on leaderboard)
         - candidate_pairs.tsv (blocking candidates)

Usage:
    # Quick sample test (e.g. 5,000 S1 entities):
    python step5_inference_pipeline.py --sample 5000

    # Full test set run:
    python step5_inference_pipeline.py --full

Validation:
    python utils-20260925T171059Z-1-001/utils/validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir dataset-20260925T160811Z-1-001/dataset/test
=============================================================================
"""

import os
import sys
import io
import time
import pickle
import argparse
import pandas as pd
from collections import defaultdict

# Force UTF-8 and line-buffered stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

from normalization import normalize_name, normalize_address
from blocking_features import get_blocking_keys
from comparison_features import compute_pair_features, FEATURE_COLS

# ── Paths ──
TEST_DIR = r"c:\Users\ASUS\Downloads\konachiwa\dataset-20260925T160811Z-1-001\dataset\test"
OUTPUT_DIR = r"c:\Users\ASUS\Downloads\konachiwa\output"
MODEL_PATH = r"c:\Users\ASUS\Downloads\konachiwa\trained_model.pkl"

S1_FILE = os.path.join(TEST_DIR, "test_source1.tsv")
S2_FILE = os.path.join(TEST_DIR, "test_source2.tsv")
S3_FILE = os.path.join(TEST_DIR, "test_source3.tsv")


def run_pipeline(sample_size=None):
    print("=" * 72)
    print("  STEP 5: INFERENCE PIPELINE & SUBMISSION GENERATION")
    print("=" * 72)
    
    t_start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Trained Model Bundle
    print(f"\n[1/5] Loading trained model from {MODEL_PATH}...")
    if not os.path.isfile(MODEL_PATH):
        print(f"      ERROR: {MODEL_PATH} not found. Run step4_train_matching_model.py first.")
        sys.exit(1)

    with open(MODEL_PATH, 'rb') as f:
        bundle = pickle.load(f)

    model = bundle['model']
    threshold = bundle['optimal_threshold']
    print(f"      Loaded model successfully.")
    print(f"      Decision threshold: {threshold:.2f} (Tuned for F_0.5 = {bundle['f05']:.4f})")
    print(f"      Expected precision: {bundle['precision']*100:.2f}%")

    # 2. Load & Normalize S1 Reference
    print(f"\n[2/5] Loading and normalizing test Source 1...")
    t0 = time.time()
    if sample_size:
        print(f"      (Sample mode: processing first {sample_size:,} S1 entities)")
        df_s1 = pd.read_csv(S1_FILE, sep='\t', nrows=sample_size, dtype=str).fillna('')
    else:
        print(f"      (Full mode: processing all S1 entities)")
        df_s1 = pd.read_csv(S1_FILE, sep='\t', dtype=str).fillna('')

    df_s1['name_clean'] = [
        normalize_name(n, c) for n, c in zip(df_s1['business_name'], df_s1['country'])
    ]
    df_s1['address_clean'] = [
        normalize_address(a, c) for a, c in zip(df_s1['business_address'], df_s1['country'])
    ]

    s1_lookup = {}
    s1_block_index = defaultdict(list)
    s1_ordered_ids = list(df_s1['entity_id'])

    for _, row in df_s1.iterrows():
        eid = row['entity_id']
        s1_lookup[eid] = {
            'entity_id': eid,
            'name_clean': row['name_clean'],
            'address_clean': row['address_clean'],
            'country': row['country'],
        }
        keys = get_blocking_keys(row)
        for k in keys:
            s1_block_index[k].append(eid)

    print(f"      Loaded & normalized {len(df_s1):,} S1 rows in {time.time()-t0:.2f}s")
    print(f"      Indexed {len(s1_block_index):,} blocking keys.")

    # 3. Stream S2 & S3 and Classify Candidates
    print(f"\n[3/5] Streaming vendor sources (S2 & S3) to score candidate matches...")
    
    # Store candidates and matches per S1 entity
    candidates_by_s1 = defaultdict(list)
    matches_by_s1 = defaultdict(list)

    def process_vendor(source_path, vendor_label):
        t_vendor = time.time()
        print(f"      Streaming {vendor_label} ({os.path.basename(source_path)})...")
        chunk_size = 100_000
        vendor_pairs_scored = 0
        vendor_matches_found = 0

        for chunk_idx, chunk in enumerate(pd.read_csv(source_path, sep='\t', chunksize=chunk_size, dtype=str)):
            chunk = chunk.fillna('')
            chunk['name_clean'] = [
                normalize_name(n, c) for n, c in zip(chunk['business_name'], chunk['country'])
            ]
            chunk['address_clean'] = [
                normalize_address(a, c) for a, c in zip(chunk['business_address'], chunk['country'])
            ]

            # Find candidates hitting S1 blocking keys
            candidate_batch = []
            for _, r in chunk.iterrows():
                vx_eid = r['entity_id']
                keys = get_blocking_keys(r)
                hit_s1 = set()
                for k in keys:
                    if k in s1_block_index:
                        bucket = s1_block_index[k]
                        if len(bucket) <= 100:
                            hit_s1.update(bucket)

                for s1_eid in hit_s1:
                    row_a = s1_lookup[s1_eid]
                    row_b = {
                        'entity_id': vx_eid,
                        'name_clean': r['name_clean'],
                        'address_clean': r['address_clean'],
                        'country': r['country'],
                    }
                    candidate_batch.append((s1_eid, vx_eid, row_a, row_b))

            if not candidate_batch:
                continue

            # Compute features for candidate batch
            X_batch = []
            batch_pairs = []
            for s1_eid, vx_eid, row_a, row_b in candidate_batch:
                feats = compute_pair_features(row_a, row_b)
                X_batch.append([feats[col] for col in FEATURE_COLS])
                batch_pairs.append((s1_eid, vx_eid))
                candidates_by_s1[s1_eid].append(vx_eid)

            # Classify batch
            probs = model.predict_proba(X_batch)[:, 1]
            vendor_pairs_scored += len(batch_pairs)

            for (s1_eid, vx_eid), p in zip(batch_pairs, probs):
                if p >= threshold:
                    matches_by_s1[s1_eid].append(vx_eid)
                    vendor_matches_found += 1

            if (chunk_idx + 1) % 10 == 0:
                print(f"        ... processed {chunk_idx + 1} chunks ({vendor_pairs_scored:,} pairs scored, {vendor_matches_found:,} matches)")

            # If running in sample mode, stop after scanning enough chunks to verify
            if sample_size and chunk_idx >= 5:
                print(f"        (Sample mode: stopping after {chunk_idx + 1} chunks)")
                break

        print(f"      {vendor_label} complete in {time.time()-t_vendor:.2f}s: "
              f"{vendor_pairs_scored:,} candidate pairs scored, {vendor_matches_found:,} matches found.")

    process_vendor(S2_FILE, "Source 2")
    process_vendor(S3_FILE, "Source 3")

    # 4. Generate Output TSV Files
    print(f"\n[4/5] Formatting submission files according to strict competition schema...")
    t0 = time.time()

    matching_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    candidate_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

    # Write matching_results.tsv
    with open(matching_path, 'w', encoding='utf-8') as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        match_count = 0
        singleton_count = 0
        for s1_id in s1_ordered_ids:
            matched = sorted(set(matches_by_s1.get(s1_id, [])))
            if matched:
                match_str = ",".join(matched)
                match_count += 1
            else:
                match_str = ""
                singleton_count += 1
            f_match.write(f"{s1_id}\t{match_str}\n")

    # Write candidate_pairs.tsv
    with open(candidate_path, 'w', encoding='utf-8') as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_ordered_ids:
            cands = sorted(set(candidates_by_s1.get(s1_id, [])))
            # Ground truth rule: matches must be a subset of candidates
            matched = set(matches_by_s1.get(s1_id, []))
            all_cands = sorted(set(cands) | matched)
            cand_str = ",".join(all_cands) if all_cands else ""
            f_cand.write(f"{s1_id}\t{cand_str}\n")

    print(f"      Generated matching_results.tsv:")
    print(f"        - Path: {matching_path}")
    print(f"        - S1 entities with matches: {match_count:,}")
    print(f"        - S1 singletons (no matches): {singleton_count:,}")
    print(f"      Generated candidate_pairs.tsv:")
    print(f"        - Path: {candidate_path}")
    print(f"      Files written in {time.time()-t0:.2f}s")

    # 5. Run Submission Validator
    print(f"\n[5/5] Running official submission validator (validate_submission.py)...")
    validator_script = r"c:\Users\ASUS\Downloads\konachiwa\utils-20260925T171059Z-1-001\utils\validate_submission.py"
    if os.path.isfile(validator_script):
        cmd = (
            f"python \"{validator_script}\" "
            f"--matching \"{matching_path}\" "
            f"--candidate \"{candidate_path}\" "
            f"--test-dir \"{TEST_DIR}\""
        )
        print(f"      Executing: {cmd}\n")
        import subprocess
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print(res.stderr)
        if res.returncode == 0:
            print("      VALIDATION SUCCESS: All competition submission checks PASSED! ✓")
        else:
            print("      VALIDATION WARNINGS / ISSUES (see output above)")
    else:
        print(f"      Validator script not found at {validator_script}")

    print("\n" + "=" * 72)
    print(f"  INFERENCE PIPELINE COMPLETE ({time.time()-t_start:.1f}s total)")
    print("=" * 72)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run entity resolution inference pipeline.")
    parser.add_argument('--sample', type=int, default=5000, help="Number of S1 rows for sample run (default: 5000)")
    parser.add_argument('--full', action='store_true', help="Run on full test set")
    args = parser.parse_args()

    sample_val = None if args.full else args.sample
    run_pipeline(sample_size=sample_val)
