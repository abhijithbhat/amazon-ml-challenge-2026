"""
step3_validate_blocking.py
===========================
Evaluate candidate generation (blocking) quality using real ground truth labels.

Metrics evaluated:
  1. Candidate Reduction Ratio:
     (1 - Candidates / Total Possible Cartesian Pairs) -> Target > 99.9%
  2. Recall / Coverage:
     % of true matches from train_ground_truth.tsv captured in the candidate set.
  3. Average candidates per S1 entity:
     Target < 20 candidates per S1 entity (keeps downstream inference fast).
"""

import sys, io, os, time
import pandas as pd
from collections import defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from normalization import normalize_name, normalize_address
from blocking_features import get_blocking_keys

BASE = r"c:\Users\ASUS\Downloads\konachiwa\dataset-20260925T160811Z-1-001\dataset\train"
PATH_S1 = os.path.join(BASE, "train_source1.tsv")
PATH_S2 = os.path.join(BASE, "train_source2.tsv")
PATH_GT = os.path.join(BASE, "train_ground_truth.tsv")

SAMPLE_SIZE = 10_000

print("=" * 72)
print("  STEP 3: EVALUATE BLOCKING QUALITY ON REAL GROUND TRUTH")
print("=" * 72)

# Load S1 sample
print(f"Loading {SAMPLE_SIZE:,} rows from Train S1...")
df_s1 = pd.read_csv(PATH_S1, sep='\t', nrows=SAMPLE_SIZE, dtype=str).fillna('')
s1_ids = set(df_s1['entity_id'])

# Load Ground Truth for this sample
print("Loading Ground Truth matching labels...")
df_gt = pd.read_csv(PATH_GT, sep='\t', dtype=str).fillna('')
df_gt = df_gt[df_gt['source1_entity_id'].isin(s1_ids)]

gt_pairs = set()
for _, row in df_gt.iterrows():
    s1_id = row['source1_entity_id']
    m_ids = [m.strip() for m in row['matched_entity_ids'].split(',') if m.strip()]
    for mid in m_ids:
        gt_pairs.add((s1_id, mid))

print(f"  Sample S1 entities: {len(s1_ids):,}")
print(f"  True match pairs in sample: {len(gt_pairs):,}")

# Load S2 sample (50,000 rows to find candidate matches)
print("\nLoading 50,000 rows from Train S2...")
df_s2 = pd.read_csv(PATH_S2, sep='\t', nrows=50_000, dtype=str).fillna('')

# Normalize both samples
print("Normalizing S1 and S2 samples...")
df_s1['name_clean'] = [normalize_name(n, c) for n, c in zip(df_s1['business_name'], df_s1['country'])]
df_s1['address_clean'] = [normalize_address(a, c) for a, c in zip(df_s1['business_address'], df_s1['country'])]

df_s2['name_clean'] = [normalize_name(n, c) for n, c in zip(df_s2['business_name'], df_s2['country'])]
df_s2['address_clean'] = [normalize_address(a, c) for a, c in zip(df_s2['business_address'], df_s2['country'])]

# Build inverted blocking index from S2
print("Building blocking index from S2...")
t0 = time.time()
block_index = defaultdict(list)
for idx, row in df_s2.iterrows():
    keys = get_blocking_keys(row)
    for k in keys:
        block_index[k].append(row['entity_id'])
t1 = time.time()
print(f"  Indexed {len(df_s2):,} S2 records into {len(block_index):,} buckets in {t1-t0:.2f}s")

# Query candidates for each S1 entity
print("Querying candidate matches for S1 entities...")
t0 = time.time()
candidate_pairs = set()
candidates_per_s1 = []

for idx, row in df_s1.iterrows():
    s1_id = row['entity_id']
    keys = get_blocking_keys(row)
    s2_candidates = set()
    for k in keys:
        if k in block_index:
            bucket = block_index[k]
            # Avoid huge runaway buckets (precision rule)
            if len(bucket) <= 100:
                s2_candidates.update(bucket)
    for c_id in s2_candidates:
        candidate_pairs.add((s1_id, c_id))
    candidates_per_s1.append(len(s2_candidates))

t1 = time.time()
print(f"  Generated {len(candidate_pairs):,} candidate pairs in {t1-t0:.2f}s")

# Metrics
total_possible = len(df_s1) * len(df_s2)
reduction_ratio = (1.0 - len(candidate_pairs) / total_possible) * 100
avg_candidates = sum(candidates_per_s1) / len(candidates_per_s1)

# Ground truth recall in the S2 sample
s2_ids_in_sample = set(df_s2['entity_id'])
relevant_gt = {p for p in gt_pairs if p[1] in s2_ids_in_sample}
captured_gt = candidate_pairs.intersection(relevant_gt)
recall = (len(captured_gt) / len(relevant_gt) * 100) if relevant_gt else 0.0

print("\n" + "=" * 72)
print("  BLOCKING QUALITY METRICS")
print("=" * 72)
print(f"  Cartesian Product (Brute Force): {total_possible:,} pairs")
print(f"  Generated Candidates:           {len(candidate_pairs):,} pairs")
print(f"  Reduction Ratio:                 {reduction_ratio:.4f}%")
print(f"  Avg Candidates per S1:           {avg_candidates:.1f}")
print(f"  True Matches in S2 Sample:       {len(relevant_gt):,}")
print(f"  True Matches Captured in Blocks: {len(captured_gt):,}")
if relevant_gt:
    print(f"  Candidate Recall / Coverage:     {recall:.2f}%")
print("=" * 72)
