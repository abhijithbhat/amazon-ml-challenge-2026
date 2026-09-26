"""
generate_test_candidates.py
===========================
High-performance streaming candidate generation for the FULL test dataset:
- test_source1.tsv: 1,732,544 rows
- test_source2.tsv: 4,887,273 rows
- test_source3.tsv: 5,082,316 rows

Uses ProcessPoolExecutor (10 workers) for parallel normalization while
streaming chunks to keep memory usage under 3 GB.
Outputs output/candidate_pairs.tsv.
"""

import os
import sys
import time
import math
import shutil
import psutil
import functools
import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

print = functools.partial(print, flush=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_DIR = os.path.dirname(BASE_DIR)

SRC_DIR = os.path.join(BASE_DIR, "src")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
BACKUP_FILE = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv.bak")

TEST_DIR = os.path.join(WORKSPACE_DIR, "dataset", "test")
PATH_TEST_S1 = os.path.join(TEST_DIR, "test_source1.tsv")
PATH_TEST_S2 = os.path.join(TEST_DIR, "test_source2.tsv")
PATH_TEST_S3 = os.path.join(TEST_DIR, "test_source3.tsv")

sys.path.insert(0, SRC_DIR)
from normalization import normalize_name, normalize_address
from blocking_features import get_blocking_keys

BUCKET_CAP = 25
MAX_CANDIDATES = 12  # Primary operating point: K=12
CHUNK_SIZE = 250_000
NUM_WORKERS = 10

BRANCH_WEIGHTS = {
    'ST_PRE3': 1.5,
    'ST': 4.0,
    '2TOK': 4.0,
    'ADDR': 3.5,
    'PIN': 2.0,
    'TOK': 2.0
}

def get_btype(k):
    if '_ST_PRE3_' in k: return 'ST_PRE3'
    if '_ST_' in k: return 'ST'
    if '_2TOK_' in k: return '2TOK'
    if '_ADDR_' in k: return 'ADDR'
    if '_PIN_' in k: return 'PIN'
    if '_TOK_' in k: return 'TOK'
    return 'OTHER'

def _norm_chunk(chunk):
    names = [normalize_name(n, c) for n, c in zip(chunk['business_name'], chunk['country'])]
    addrs = [normalize_address(a, c) for a, c in zip(chunk['business_address'], chunk['country'])]
    chunk['name_clean'] = names
    chunk['address_clean'] = addrs
    return chunk

def parallel_normalize(df, executor):
    chunks = np.array_split(df, NUM_WORKERS)
    results = list(executor.map(_norm_chunk, chunks))
    return pd.concat(results, ignore_index=True)

def main():
    t_start = time.time()
    proc = psutil.Process(os.getpid())
    
    print("=" * 80)
    print("  GENERATING TEST CANDIDATE PAIRS (PRIMARY OPERATING POINT: K=12)")
    print("=" * 80)
    print(f"Target Output Path:  {OUTPUT_FILE}")
    print(f"Operating Config:    K={MAX_CANDIDATES}, BUCKET_CAP={BUCKET_CAP}")
    print(f"Workers:             {NUM_WORKERS}")

    # Confirm and backup existing file if present
    if os.path.exists(OUTPUT_FILE):
        existing_sz = os.path.getsize(OUTPUT_FILE)
        print(f"Existing file found: {existing_sz / (1024**2):.2f} MB")
        print(f"Creating backup at:  {BACKUP_FILE}")
        shutil.copyfile(OUTPUT_FILE, BACKUP_FILE)
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # ── Step 1: Index S2 and S3 ──
        print("\n[1] Indexing Test Target Records (S2 + S3)...")
        block_index = defaultdict(list)
        total_indexed = 0
        t0 = time.time()
        
        for label, path in [('S2', PATH_TEST_S2), ('S3', PATH_TEST_S3)]:
            t_sub = time.time()
            print(f"  Streaming {label} from {os.path.basename(path)}...")
            sub_count = 0
            for chunk in pd.read_csv(path, sep='\t', chunksize=CHUNK_SIZE, dtype=str):
                chunk = chunk.fillna('')
                chunk_norm = parallel_normalize(chunk, executor)
                
                recs = chunk_norm.to_dict('records')
                for row in recs:
                    eid = row['entity_id']
                    for k in get_blocking_keys(row):
                        block_index[k].append(eid)
                        
                sub_count += len(chunk)
                total_indexed += len(chunk)
                mem = proc.memory_info().rss / (1024**2)
                print(f"    ... indexed {sub_count:,} {label} records ({time.time()-t_sub:.1f}s, RAM: {mem:.1f} MB)")
                
            print(f"  {label} indexing complete: {sub_count:,} records in {time.time()-t_sub:.1f}s")
            
        print(f"\nIndex built: {total_indexed:,} records into {len(block_index):,} buckets in {time.time()-t0:.1f}s")
        print(f"Current RAM after indexing: {proc.memory_info().rss / (1024**2):.1f} MB")
        
        # ── Step 2: Query S1 and write directly to TSV with Priority Ranking ──
        print("\n[2] Generating candidates for Test Source 1 with Priority Ranking (K=12)...")
        t0 = time.time()
        total_s1 = 0
        total_pairs = 0
        zero_cands = 0
        cand_counts = []
        peak_ram_mb = proc.memory_info().rss / (1024**2)
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
            out_f.write("source1_entity_id\tcandidate_entity_ids\n")
            
            for chunk in pd.read_csv(PATH_TEST_S1, sep='\t', chunksize=CHUNK_SIZE, dtype=str):
                chunk = chunk.fillna('')
                chunk_norm = parallel_normalize(chunk, executor)
                
                lines = []
                recs = chunk_norm.to_dict('records')
                for row in recs:
                    s1_id = row['entity_id']
                    keys = get_blocking_keys(row)
                    
                    cand_scores = defaultdict(float)
                    cand_branches = defaultdict(set)
                    
                    for k in keys:
                        if k in block_index:
                            b = block_index[k]
                            eff_cap = min(15, BUCKET_CAP) if '_ST_PRE3_' in k else BUCKET_CAP
                            if len(b) <= eff_cap:
                                btype = get_btype(k)
                                w = BRANCH_WEIGHTS.get(btype, 1.0)
                                spec = 1.0 / math.sqrt(len(b))
                                for cid in b:
                                    cand_scores[cid] += w + spec
                                    cand_branches[cid].add(btype)
                                    
                    # Consensus bonus for multi-branch agreement
                    for cid, brs in cand_branches.items():
                        if len(brs) >= 2:
                            cand_scores[cid] += 3.0 * (len(brs) - 1)
                            
                    # Priority rank and truncate to K=12
                    ranked = sorted(cand_scores.keys(), key=lambda c: cand_scores[c], reverse=True)[:MAX_CANDIDATES]
                    
                    n_cands = len(ranked)
                    total_pairs += n_cands
                    if n_cands == 0:
                        zero_cands += 1
                    cand_counts.append(n_cands)
                    
                    cand_str = ",".join(ranked)
                    lines.append(f"{s1_id}\t{cand_str}\n")
                    
                out_f.writelines(lines)
                total_s1 += len(chunk)
                
                cur_ram = proc.memory_info().rss / (1024**2)
                if cur_ram > peak_ram_mb:
                    peak_ram_mb = cur_ram
                    
                elapsed = time.time() - t0
                rate = total_s1 / elapsed
                print(f"  ... processed {total_s1:,} / 1,732,544 S1 entities ({rate:.0f} entities/s, RAM: {cur_ram:.1f} MB, elapsed {elapsed:.1f}s)")
                
    t_total = time.time() - t_start
    carr = np.array(cand_counts)
    out_size_bytes = os.path.getsize(OUTPUT_FILE)
    
    print("\n" + "=" * 80)
    print("  TEST CANDIDATE GENERATION COMPLETE (K=12)")
    print("=" * 80)
    print(f"  Total Test S1 Entities:      {total_s1:,}")
    print(f"  Total Candidate Pairs:       {total_pairs:,}")
    print(f"  Average Candidates / S1:     {carr.mean():.2f}")
    print(f"  Median Candidates:           {np.median(carr):.0f}")
    print(f"  P95 Candidates:              {np.percentile(carr, 95):.0f}")
    print(f"  P99 Candidates:              {np.percentile(carr, 99):.0f}")
    print(f"  Maximum Candidates:          {carr.max():,}")
    print(f"  Zero-Candidate Count:        {zero_cands:,} ({100*zero_cands/total_s1:.2f}%)")
    print(f"  Runtime:                     {t_total:.1f}s ({t_total/60:.2f} min)")
    print(f"  Peak Memory:                 {peak_ram_mb:.1f} MB ({peak_ram_mb/1024:.2f} GB)")
    print(f"  Output File Size:            {out_size_bytes / (1024**2):.2f} MB ({out_size_bytes:,} bytes)")
    print(f"  Output File Path:            {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
