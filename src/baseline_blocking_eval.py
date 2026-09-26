"""
baseline_blocking_eval.py
==========================
Comprehensive baseline evaluation of the existing blocking system.

Loads FULL training S1, S2, S3 and ground truth.
Measures: recall, avg/P95/max candidates, runtime, memory.
Analyzes missed matches by failure cause.
Identifies which blocking branch caused each hit.
"""

import sys, io, os, time, tracemalloc, re
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add src/ for imports
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from normalization import normalize_name, normalize_address
from blocking_features import get_blocking_keys, extract_postal_code, extract_state_code, extract_significant_tokens

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

# ── Paths (relative to project root which is inside student_resource/) ──
TRAIN_DIR = os.path.join('..', 'dataset', 'train')

PATH_S1 = os.path.join(TRAIN_DIR, 'train_source1.tsv')
PATH_S2 = os.path.join(TRAIN_DIR, 'train_source2.tsv')
PATH_S3 = os.path.join(TRAIN_DIR, 'train_source3.tsv')
PATH_GT = os.path.join(TRAIN_DIR, 'train_ground_truth.tsv')

# Verify paths
for p in [PATH_S1, PATH_S2, PATH_S3, PATH_GT]:
    if not os.path.exists(p):
        print(f"ERROR: File not found: {p}")
        print(f"  Resolved to: {os.path.abspath(p)}")
        sys.exit(1)

print("=" * 80)
print("  BASELINE BLOCKING EVALUATION — FULL TRAINING DATA")
print("=" * 80)

# ── CONFIGURATION ──
# For initial baseline, use a meaningful sample to get results in reasonable time
S1_LIMIT = 50000
S2_LIMIT = 200000
S3_LIMIT = 200000
BUCKET_CAP = 100  # matches existing implementation

# ═══════════════════════════════════════════════════════════════════════════════
#  1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

tracemalloc.start()
t_start = time.time()

print("\n[1] Loading data...")
t0 = time.time()
df_s1 = pd.read_csv(PATH_S1, sep='\t', nrows=S1_LIMIT, dtype=str).fillna('')
print(f"  S1: {len(df_s1):,} rows  ({time.time()-t0:.1f}s)")

t0 = time.time()
df_s2 = pd.read_csv(PATH_S2, sep='\t', nrows=S2_LIMIT, dtype=str).fillna('')
print(f"  S2: {len(df_s2):,} rows  ({time.time()-t0:.1f}s)")

t0 = time.time()
df_s3 = pd.read_csv(PATH_S3, sep='\t', nrows=S3_LIMIT, dtype=str).fillna('')
print(f"  S3: {len(df_s3):,} rows  ({time.time()-t0:.1f}s)")

t0 = time.time()
df_gt = pd.read_csv(PATH_GT, sep='\t', dtype=str).fillna('')
print(f"  GT: {len(df_gt):,} rows  ({time.time()-t0:.1f}s)")

# Parse ground truth - vectorized (2.2M rows is too slow for iterrows)
s1_ids = set(df_s1['entity_id'])
print("  Parsing ground truth (vectorized)...")
t0 = time.time()
df_gt_filtered = df_gt[df_gt['source1_entity_id'].isin(s1_ids)].copy()
df_gt_filtered = df_gt_filtered[df_gt_filtered['matched_entity_ids'].str.len() > 0]
# Explode comma-separated IDs
df_gt_exploded = df_gt_filtered.assign(
    matched_id=df_gt_filtered['matched_entity_ids'].str.split(',')
).explode('matched_id')
df_gt_exploded['matched_id'] = df_gt_exploded['matched_id'].str.strip()
df_gt_exploded = df_gt_exploded[df_gt_exploded['matched_id'].str.len() > 0]

gt_pairs = set(zip(df_gt_exploded['source1_entity_id'], df_gt_exploded['matched_id']))
gt_by_s1 = defaultdict(set)
for s1_id, mid in gt_pairs:
    gt_by_s1[s1_id].add(mid)
print(f"  GT parsing: {time.time()-t0:.1f}s")

print(f"\n  S1 entities: {len(s1_ids):,}")
print(f"  Total true-match pairs: {len(gt_pairs):,}")
print(f"  S1 entities with matches: {len(gt_by_s1):,}")

# Separate S2 and S3 ground truth
s2_ids_in_data = set(df_s2['entity_id'])
s3_ids_in_data = set(df_s3['entity_id'])
gt_s2_pairs = {p for p in gt_pairs if p[1] in s2_ids_in_data}
gt_s3_pairs = {p for p in gt_pairs if p[1] in s3_ids_in_data}
gt_unreachable = gt_pairs - gt_s2_pairs - gt_s3_pairs

print(f"  GT pairs with S2 match in loaded data: {len(gt_s2_pairs):,}")
print(f"  GT pairs with S3 match in loaded data: {len(gt_s3_pairs):,}")
print(f"  GT pairs unreachable (matched ID not in loaded S2/S3): {len(gt_unreachable):,}")

# ═══════════════════════════════════════════════════════════════════════════════
#  2. NORMALIZE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[2] Normalizing...")
t0 = time.time()

for df, label in [(df_s1, 'S1'), (df_s2, 'S2'), (df_s3, 'S3')]:
    t_n = time.time()
    df['name_clean'] = [normalize_name(n, c) for n, c in zip(df['business_name'], df['country'])]
    df['address_clean'] = [normalize_address(a, c) for a, c in zip(df['business_address'], df['country'])]
    print(f"  {label}: normalized {len(df):,} rows in {time.time()-t_n:.1f}s")

print(f"  Total normalization time: {time.time()-t0:.1f}s")

# ═══════════════════════════════════════════════════════════════════════════════
#  3. BUILD BLOCKING INDEX (S2 + S3)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[3] Building blocking index from S2 + S3...")
t0 = time.time()

block_index = defaultdict(list)
for df_target, label in [(df_s2, 'S2'), (df_s3, 'S3')]:
    t_b = time.time()
    records = df_target.to_dict('records')
    for row in records:
        keys = get_blocking_keys(row)
        eid = row['entity_id']
        for k in keys:
            block_index[k].append(eid)
    print(f"  {label}: indexed {len(df_target):,} rows in {time.time()-t_b:.1f}s")

print(f"  Total buckets: {len(block_index):,}")
bucket_sizes = [len(v) for v in block_index.values()]
print(f"  Bucket size stats: mean={np.mean(bucket_sizes):.1f}, "
      f"median={np.median(bucket_sizes):.0f}, "
      f"P95={np.percentile(bucket_sizes, 95):.0f}, "
      f"max={max(bucket_sizes):,}")
big_buckets = sum(1 for s in bucket_sizes if s > BUCKET_CAP)
print(f"  Buckets > {BUCKET_CAP} (will be SKIPPED): {big_buckets:,} "
      f"({100*big_buckets/len(bucket_sizes):.2f}%)")

# ═══════════════════════════════════════════════════════════════════════════════
#  4. QUERY CANDIDATES FOR S1
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[4] Querying candidates for S1 entities...")
t0 = time.time()

candidate_pairs = set()
candidates_per_s1 = []
# Track which blocking key type matched each candidate
hit_sources = defaultdict(set)

s1_records = df_s1.to_dict('records')
for i, row in enumerate(s1_records):
    s1_id = row['entity_id']
    keys = get_blocking_keys(row)
    s_candidates = set()
    for k in keys:
        if k in block_index:
            bucket = block_index[k]
            if len(bucket) <= BUCKET_CAP:
                for c_id in bucket:
                    s_candidates.add(c_id)
                    if '_ST_PRE3_' in k:
                        hit_sources[(s1_id, c_id)].add('state+prefix3')
                    elif '_ST_' in k and '_ADDR_' not in k:
                        hit_sources[(s1_id, c_id)].add('state+token')
                    elif '_PIN_' in k:
                        hit_sources[(s1_id, c_id)].add('pin+prefix')
                    elif '_ADDR_' in k:
                        hit_sources[(s1_id, c_id)].add('state+addr_token')
                    elif '_TOK_' in k:
                        hit_sources[(s1_id, c_id)].add('country+token_fallback')
    for c_id in s_candidates:
        candidate_pairs.add((s1_id, c_id))
    candidates_per_s1.append(len(s_candidates))

    if (i + 1) % 10000 == 0:
        elapsed = time.time() - t0
        print(f"  ... {i+1:,} / {len(s1_records):,}  ({elapsed:.1f}s)")

t_query = time.time() - t0
print(f"  Query complete: {len(candidate_pairs):,} candidate pairs in {t_query:.1f}s")

# ═══════════════════════════════════════════════════════════════════════════════
#  5. COMPUTE METRICS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  BASELINE BLOCKING METRICS")
print("=" * 80)

cand_arr = np.array(candidates_per_s1)
avg_cand = cand_arr.mean()
p95_cand = np.percentile(cand_arr, 95)
max_cand = cand_arr.max()
p99_cand = np.percentile(cand_arr, 99)
zero_cand = (cand_arr == 0).sum()

print(f"\n  Candidate set statistics:")
print(f"    Total candidate pairs:  {len(candidate_pairs):,}")
print(f"    Avg candidates / S1:    {avg_cand:.2f}")
print(f"    Median candidates / S1: {np.median(cand_arr):.0f}")
print(f"    P95 candidates / S1:    {p95_cand:.0f}")
print(f"    P99 candidates / S1:    {p99_cand:.0f}")
print(f"    Max candidates / S1:    {max_cand:,}")
print(f"    S1 with 0 candidates:   {zero_cand:,} ({100*zero_cand/len(cand_arr):.1f}%)")

# Recall
reachable_gt = gt_s2_pairs | gt_s3_pairs
captured = candidate_pairs & reachable_gt
recall = len(captured) / len(reachable_gt) * 100 if reachable_gt else 0

print(f"\n  Recall (ground truth coverage):")
print(f"    Reachable GT pairs:     {len(reachable_gt):,}")
print(f"    Captured by blocking:   {len(captured):,}")
print(f"    RECALL:                 {recall:.2f}%")
print(f"    Missed:                 {len(reachable_gt) - len(captured):,}")

# S2 vs S3 breakdown
captured_s2 = candidate_pairs & gt_s2_pairs
captured_s3 = candidate_pairs & gt_s3_pairs
recall_s2 = len(captured_s2) / len(gt_s2_pairs) * 100 if gt_s2_pairs else 0
recall_s3 = len(captured_s3) / len(gt_s3_pairs) * 100 if gt_s3_pairs else 0
print(f"\n  Breakdown by source:")
print(f"    S2 recall: {recall_s2:.2f}%  ({len(captured_s2):,}/{len(gt_s2_pairs):,})")
print(f"    S3 recall: {recall_s3:.2f}%  ({len(captured_s3):,}/{len(gt_s3_pairs):,})")

# Country breakdown
print(f"\n  Breakdown by country:")
for country in ['US', 'India', 'France']:
    country_s1 = set(df_s1[df_s1['country'] == country]['entity_id'])
    country_gt = {p for p in reachable_gt if p[0] in country_s1}
    country_cap = captured & country_gt
    if country_gt:
        c_recall = len(country_cap) / len(country_gt) * 100
        print(f"    {country}: {c_recall:.2f}% recall ({len(country_cap):,}/{len(country_gt):,})")

# Runtime & memory
t_total = time.time() - t_start
current_mem, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"\n  Runtime:")
print(f"    Total:       {t_total:.1f}s")
print(f"    Query phase: {t_query:.1f}s")
print(f"\n  Memory:")
print(f"    Current: {current_mem / 1024**2:.1f} MB")
print(f"    Peak:    {peak_mem / 1024**2:.1f} MB")

# ═══════════════════════════════════════════════════════════════════════════════
#  6. CANDIDATE EXPLOSION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  CANDIDATE EXPLOSION ANALYSIS")
print("=" * 80)

thresholds = [10, 20, 50, 100, 200, 500, 1000]
for th in thresholds:
    count = (cand_arr > th).sum()
    if count > 0:
        print(f"  S1 entities with > {th} candidates: {count:,}")

top_indices = np.argsort(candidates_per_s1)[-10:][::-1]
print(f"\n  Top 10 entities by candidate count:")
for i in top_indices:
    row = df_s1.iloc[i]
    print(f"    {row['entity_id']} ({row['country']}): {candidates_per_s1[i]:,} candidates")
    print(f"      Name: {row['business_name'][:60]}")

# ═══════════════════════════════════════════════════════════════════════════════
#  7. MISSED MATCH FAILURE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  MISSED MATCH FAILURE ANALYSIS")
print("=" * 80)

missed = reachable_gt - captured

s2_lookup = df_s2.set_index('entity_id')
s3_lookup = df_s3.set_index('entity_id')
s1_lookup = df_s1.set_index('entity_id')

failure_causes = Counter()
failure_examples = defaultdict(list)
MAX_EXAMPLES = 5

def classify_failure(s1_row, target_row):
    causes = []
    s1_country = s1_row.get('country', '')
    t_country = target_row.get('country', '')
    
    if s1_country != t_country:
        return ['country_mismatch']
    
    s1_name = s1_row.get('name_clean', '')
    t_name = target_row.get('name_clean', '')
    s1_addr = s1_row.get('address_clean', '')
    t_addr = target_row.get('address_clean', '')
    
    s1_state = extract_state_code(s1_addr, s1_country)
    t_state = extract_state_code(t_addr, t_country)
    s1_pin = extract_postal_code(s1_addr, s1_country)
    t_pin = extract_postal_code(t_addr, t_country)
    s1_tokens = extract_significant_tokens(s1_name)
    t_tokens = extract_significant_tokens(t_name)
    
    # State analysis
    if not s1_state and not t_state:
        causes.append('both_missing_state')
    elif s1_state != t_state:
        if not s1_state or not t_state:
            causes.append('one_missing_state')
        else:
            causes.append('state_mismatch')
    
    # Postal analysis
    if not s1_pin and not t_pin:
        causes.append('both_missing_postal')
    elif s1_pin != t_pin:
        if not s1_pin or not t_pin:
            causes.append('one_missing_postal')
        else:
            causes.append('postal_mismatch')
    
    # Name token analysis
    if not s1_tokens or not t_tokens:
        causes.append('missing_name_tokens')
    else:
        if s1_tokens[0] != t_tokens[0]:
            has_indic_s1 = bool(re.search(r'[\u0900-\u0D7F]', s1_row.get('business_name', '')))
            has_indic_t = bool(re.search(r'[\u0900-\u0D7F]', target_row.get('business_name', '')))
            if has_indic_s1 != has_indic_t:
                causes.append('transliteration_variation')
            elif has_indic_s1 and has_indic_t:
                causes.append('indic_script_variation')
            else:
                overlap = set(s1_tokens) & set(t_tokens)
                if not overlap:
                    causes.append('name_completely_different')
                else:
                    causes.append('name_first_token_mismatch')
        
        s1_pre3 = s1_tokens[0][:3] if s1_tokens else ''
        t_pre3 = t_tokens[0][:3] if t_tokens else ''
        if s1_pre3 != t_pre3:
            causes.append('name_prefix3_mismatch')
    
    if s1_addr and t_addr:
        s1_addr_toks = set(re.findall(r'[a-zA-Z0-9]+', s1_addr.lower()))
        t_addr_toks = set(re.findall(r'[a-zA-Z0-9]+', t_addr.lower()))
        addr_overlap = s1_addr_toks & t_addr_toks
        if len(addr_overlap) < 2:
            causes.append('address_completely_different')
    elif not s1_addr or not t_addr:
        causes.append('missing_address')
    
    if not causes:
        causes.append('bucket_cap_or_blocking_key_collision')
    
    return causes


print(f"\n  Analyzing {min(len(missed), 10000):,} of {len(missed):,} missed pairs...")
analyzed = 0
for s1_id, target_id in list(missed)[:10000]:
    try:
        s1_row = s1_lookup.loc[s1_id]
        if target_id.startswith('S2-'):
            t_row = s2_lookup.loc[target_id]
        else:
            t_row = s3_lookup.loc[target_id]
    except KeyError:
        failure_causes['lookup_error'] += 1
        analyzed += 1
        continue
    
    causes = classify_failure(s1_row, t_row)
    for cause in causes:
        failure_causes[cause] += 1
        if len(failure_examples[cause]) < MAX_EXAMPLES:
            failure_examples[cause].append({
                's1_id': s1_id,
                'target_id': target_id,
                's1_name': str(s1_row.get('business_name', ''))[:60],
                't_name': str(t_row.get('business_name', ''))[:60],
                's1_addr': str(s1_row.get('business_address', ''))[:60],
                't_addr': str(t_row.get('business_address', ''))[:60],
                's1_country': str(s1_row.get('country', '')),
            })
    analyzed += 1

print(f"\n  Failure causes (from {analyzed:,} analyzed missed pairs):")
print(f"  {'Cause':<40s}  {'Count':>8s}  {'%':>6s}")
print(f"  {'-'*40}  {'-'*8}  {'-'*6}")
for cause, count in failure_causes.most_common():
    pct = 100 * count / analyzed if analyzed > 0 else 0
    print(f"  {cause:<40s}  {count:>8,}  {pct:>5.1f}%")

print(f"\n  Example missed pairs by top causes:")
for cause, count in failure_causes.most_common(5):
    print(f"\n  === {cause} ({count:,} occurrences) ===")
    for ex in failure_examples[cause][:3]:
        print(f"    S1 {ex['s1_id']} ({ex['s1_country']}): {ex['s1_name']}")
        print(f"    Target {ex['target_id']}: {ex['t_name']}")
        print(f"    S1 addr: {ex['s1_addr']}")
        print(f"    T  addr: {ex['t_addr']}")
        print()

# ═══════════════════════════════════════════════════════════════════════════════
#  8. BLOCKING BRANCH ATTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("  BLOCKING BRANCH ATTRIBUTION (successful retrievals)")
print("=" * 80)

branch_counts = Counter()
for pair in captured:
    sources = hit_sources.get(pair, set())
    for s in sources:
        branch_counts[s] += 1
    if not sources:
        branch_counts['unknown'] += 1

print(f"\n  {'Branch':<30s}  {'Hits':>10s}  {'%':>6s}")
print(f"  {'-'*30}  {'-'*10}  {'-'*6}")
for branch, count in branch_counts.most_common():
    pct = 100 * count / len(captured) if captured else 0
    print(f"  {branch:<30s}  {count:>10,}  {pct:>5.1f}%")

sole_branch = Counter()
for pair in captured:
    sources = hit_sources.get(pair, set())
    if len(sources) == 1:
        sole_branch[list(sources)[0]] += 1

print(f"\n  Sole-source attribution (pair captured by only one branch):")
for branch, count in sole_branch.most_common():
    pct = 100 * count / len(captured) if captured else 0
    print(f"    {branch:<30s}  {count:>10,}  ({pct:.1f}%)")

print("\n" + "=" * 80)
print("  BASELINE EVALUATION COMPLETE")
print("=" * 80)
