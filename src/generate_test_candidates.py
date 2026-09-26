"""
generate_test_candidates.py
===========================
High-performance, memory-safe streaming candidate generation for the FULL test dataset:
- test_source1.tsv: 1,732,544 rows
- test_source2.tsv: 4,887,273 rows
- test_source3.tsv: 5,082,316 rows

Architecture:
- Country-partitioned execution (France -> US -> India) to guarantee 100% hard country
  isolation while bounding peak memory usage strictly under 1.8 GB.
- ProcessPoolExecutor (10 workers) for fast parallel record normalization & token extraction.
- Preserves exact original sequence of Source 1 entity IDs in output/candidate_pairs.tsv.

Operating Configuration:
- PURE SELECTIVE PRE3 with K=12
- PRE3 admission cap = 8
- PRE3 large bucket threshold = 100 (top 1 candidate via cheap token overlap)
- Other branches cap = 25, large cap = 200 (top 2 candidates)
- Priority candidate ranking:
    * Learned branch reliability weights (state+token: 4.0, 2tok: 4.0, addr: 3.5, etc.)
    * Multi-branch consensus bonus: 3.0 * (n_branches - 1)
    * Inverse bucket specificity score: min(3.0, sum(1 / sqrt(bsize)))
    * Cheap Name Jaccard + token overlap similarity
    * Cheap Address Jaccard + token overlap similarity
    * Truncation to top K=12 candidates per S1 entity
"""

import os
import sys
import gc
import time
import math
import shutil
import psutil
import functools
import re
import io
import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
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
from blocking_features import (
    get_blocking_keys, extract_significant_tokens, ADDR_STOPWORDS
)

BUCKET_CAP = 25
MAX_CANDIDATES = 12  # Primary operating point: K=12
CHUNK_SIZE = 250_000
NUM_WORKERS = 10

BRANCH_WEIGHTS = {
    'state+token': 4.0,
    '2tok_conjunction': 4.0,
    'state+addr_token': 3.5,
    'state+second_token': 2.5,
    'pin+prefix': 2.0,
    'country+token_fallback': 2.0,
    'state+prefix3': 1.5,
    'other': 1.0
}


def _norm_chunk(chunk):
    names = [normalize_name(n, c) for n, c in zip(chunk['business_name'], chunk['country'])]
    addrs = [normalize_address(a, c) for a, c in zip(chunk['business_address'], chunk['country'])]
    chunk['name_clean'] = names
    chunk['address_clean'] = addrs

    # Pre-extract lightweight token tuples in worker processes
    n_toks = [tuple(extract_significant_tokens(n)) for n in names]
    a_toks = [
        tuple(t for t in re.findall(r'[a-zA-Z0-9]+', a.lower()) if len(t) >= 4 and t not in ADDR_STOPWORDS)
        for a in addrs
    ]
    chunk['name_toks'] = n_toks
    chunk['addr_toks'] = a_toks
    return chunk


def parallel_normalize(df, executor):
    chunks = np.array_split(df, NUM_WORKERS)
    results = list(executor.map(_norm_chunk, chunks))
    return pd.concat(results, ignore_index=True)


def main():
    t_start = time.time()
    proc = psutil.Process(os.getpid())

    print("=" * 80)
    print("  GENERATING TEST CANDIDATE PAIRS (PURE SELECTIVE PRE3, K=12)")
    print("=" * 80)
    print(f"Target Output Path:  {OUTPUT_FILE}")
    print(f"Operating Config:    K={MAX_CANDIDATES}, BUCKET_CAP={BUCKET_CAP}")
    print(f"PRE3 Policy:         Admission cap=8, Large cap=100, Top=1")
    print(f"Workers:             {NUM_WORKERS}")

    # Confirm and backup existing file if present
    if os.path.exists(OUTPUT_FILE):
        existing_sz = os.path.getsize(OUTPUT_FILE)
        print(f"Existing file found: {existing_sz / (1024**2):.2f} MB")
        print(f"Preserving backup at: {BACKUP_FILE}")
        shutil.copyfile(OUTPUT_FILE, BACKUP_FILE)

    cand_map = {}  # s1_id -> candidate_string
    peak_ram_mb = proc.memory_info().rss / (1024**2)

    countries = ['France', 'US', 'India']

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for country in countries:
            t_country = time.time()
            print("\n" + "-" * 80)
            print(f"  PROCESSING COUNTRY: {country.upper()}")
            print("-" * 80)

            # ── Step 1: Index Target Records (S2 + S3) for this Country ──
            print(f"[{country}] 1. Indexing Target Records (S2 + S3)...")
            block_index = defaultdict(list)
            target_meta = {}  # eid -> (name_tok_set, addr_tok_set)
            country_target_count = 0
            t0 = time.time()

            for label, path in [('S2', PATH_TEST_S2), ('S3', PATH_TEST_S3)]:
                t_sub = time.time()
                sub_count = 0
                for chunk in pd.read_csv(path, sep='\t', chunksize=CHUNK_SIZE, dtype=str):
                    chunk = chunk[chunk['country'] == country]
                    if len(chunk) == 0:
                        continue
                    chunk = chunk.fillna('')
                    chunk_norm = parallel_normalize(chunk, executor)

                    recs = chunk_norm.to_dict('records')
                    for row in recs:
                        eid = row['entity_id']
                        target_meta[eid] = (set(row['name_toks']), set(row['addr_toks']))
                        for k in get_blocking_keys(row):
                            block_index[k].append(eid)

                    sub_count += len(chunk)
                    country_target_count += len(chunk)

                mem = proc.memory_info().rss / (1024**2)
                if mem > peak_ram_mb: peak_ram_mb = mem
                print(f"  Indexed {sub_count:,} {label} {country} records ({time.time()-t_sub:.1f}s, RAM: {mem:.1f} MB)")

            print(f"[{country}] Index complete: {country_target_count:,} targets into {len(block_index):,} buckets ({time.time()-t0:.1f}s)")

            # ── Step 2: Query S1 for this Country ──
            print(f"\n[{country}] 2. Generating Candidates for S1 entities...")
            t0 = time.time()
            country_s1_count = 0
            country_pairs_count = 0

            for chunk in pd.read_csv(PATH_TEST_S1, sep='\t', chunksize=CHUNK_SIZE, dtype=str):
                chunk = chunk[chunk['country'] == country]
                if len(chunk) == 0:
                    continue
                chunk = chunk.fillna('')
                chunk_norm = parallel_normalize(chunk, executor)

                recs = chunk_norm.to_dict('records')
                for row in recs:
                    s1_id = row['entity_id']
                    s1_name_toks = set(row['name_toks'])
                    s1_addr_toks = set(row['addr_toks'])
                    keys = get_blocking_keys(row)

                    cand_meta = defaultdict(lambda: {'branches': set(), 'spec': 0.0})

                    for k in keys:
                        if k not in block_index:
                            continue
                        bsize = len(block_index[k])

                        # Pure Selective PRE3 configuration
                        if '_ST_PRE3_' in k:
                            btype = 'state+prefix3'
                            eff_cap = 8
                            eff_large_cap = 100
                            top_take = 1
                        else:
                            eff_cap = BUCKET_CAP  # 25
                            eff_large_cap = 200
                            top_take = 2
                            if '_ST_' in k:
                                btype = 'state+token'
                            elif '_2TOK_' in k:
                                btype = '2tok_conjunction'
                            elif '_ADDR_' in k:
                                btype = 'state+addr_token'
                            elif '_PIN_' in k:
                                btype = 'pin+prefix'
                            elif '_TOK_' in k:
                                btype = 'country+token_fallback'
                            else:
                                btype = 'other'

                        if bsize <= eff_cap:
                            spec = 1.0 / math.sqrt(bsize)
                            for cid in block_index[k]:
                                cm = cand_meta[cid]
                                cm['branches'].add(btype)
                                cm['spec'] += spec
                        elif bsize <= eff_large_cap:
                            # Selective large-bucket retrieval: filter candidates by token overlap
                            large_matches = []
                            for cid in block_index[k]:
                                t_cand = target_meta.get(cid)
                                if not t_cand:
                                    continue
                                n_overlap = len(s1_name_toks & t_cand[0])
                                a_overlap = len(s1_addr_toks & t_cand[1])
                                if n_overlap >= 1 or a_overlap >= 2:
                                    large_matches.append((n_overlap * 2.0 + a_overlap, cid))
                            large_matches.sort(reverse=True)
                            for _, cid in large_matches[:top_take]:
                                cm = cand_meta[cid]
                                cm['branches'].add(btype)
                                cm['spec'] += 0.1

                    if not cand_meta:
                        cand_map[s1_id] = ""
                        country_s1_count += 1
                        continue

                    # Score all collected candidates
                    scored_cands = []
                    for cid, meta in cand_meta.items():
                        t_cand = target_meta.get(cid)
                        if not t_cand:
                            continue

                        # Branch reliability weight + consensus bonus
                        b_score = sum(BRANCH_WEIGHTS.get(b, 1.0) for b in meta['branches'])
                        if len(meta['branches']) >= 2:
                            b_score += 3.0 * (len(meta['branches']) - 1)

                        # Bucket specificity score
                        spec_score = min(3.0, meta['spec'])

                        # Cheap Name Similarity (Jaccard + overlap)
                        t_name_toks = t_cand[0]
                        inter_n = len(s1_name_toks & t_name_toks)
                        union_n = len(s1_name_toks | t_name_toks)
                        name_jaccard = inter_n / max(1, union_n)
                        name_score = name_jaccard * 4.0 + inter_n * 1.5

                        # Cheap Address Similarity (Jaccard + overlap)
                        t_addr_toks = t_cand[1]
                        inter_a = len(s1_addr_toks & t_addr_toks)
                        union_a = len(s1_addr_toks | t_addr_toks)
                        addr_jaccard = inter_a / max(1, union_a)
                        addr_score = addr_jaccard * 2.5 + inter_a * 0.8

                        total_score = b_score + spec_score + name_score + addr_score
                        scored_cands.append((total_score, cid))

                    # Rank candidates descending by priority score
                    scored_cands.sort(reverse=True)
                    ranked = [cid for _, cid in scored_cands[:MAX_CANDIDATES]]

                    country_pairs_count += len(ranked)
                    country_s1_count += 1
                    cand_map[s1_id] = ",".join(ranked)

                mem = proc.memory_info().rss / (1024**2)
                if mem > peak_ram_mb: peak_ram_mb = mem
                elapsed = time.time() - t0
                rate = country_s1_count / max(0.1, elapsed)
                print(f"  [{country}] ... processed {country_s1_count:,} S1 entities ({rate:.0f} ent/s, RAM: {mem:.1f} MB)")

            print(f"[{country}] Complete: {country_s1_count:,} S1 entities, {country_pairs_count:,} pairs in {time.time()-t_country:.1f}s")

            # Clean up memory before next country
            del block_index
            del target_meta
            gc.collect()

    # ── Step 3: Write Output TSV in exact original sequence ──
    print("\n" + "=" * 80)
    print("  [3] WRITING FINAL OUTPUT TSV IN EXACT ORIGINAL S1 ORDER")
    print("=" * 80)
    t_write = time.time()
    total_written = 0
    cand_counts = []
    zero_cands = 0
    total_pairs = 0

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        out_f.write("source1_entity_id\tcandidate_entity_ids\n")
        with open(PATH_TEST_S1, 'r', encoding='utf-8') as in_f:
            next(in_f, None)  # skip header
            lines = []
            for line in in_f:
                s1_id = line.split('\t', 1)[0].strip()
                if not s1_id:
                    continue
                c_str = cand_map.get(s1_id, '')
                n_c = len(c_str.split(',')) if c_str else 0
                if n_c == 0:
                    zero_cands += 1
                total_pairs += n_c
                cand_counts.append(n_c)
                lines.append(f"{s1_id}\t{c_str}\n")
                total_written += 1
                if len(lines) >= 100_000:
                    out_f.writelines(lines)
                    lines = []
            if lines:
                out_f.writelines(lines)

    print(f"Wrote {total_written:,} rows in {time.time()-t_write:.1f}s")

    t_total = time.time() - t_start
    carr = np.array(cand_counts)
    out_size_bytes = os.path.getsize(OUTPUT_FILE)

    print("\n" + "=" * 80)
    print("  TEST CANDIDATE GENERATION COMPLETE (PURE SELECTIVE PRE3, K=12)")
    print("=" * 80)
    print(f"  Total Test S1 Entities:      {total_written:,}")
    print(f"  Total Candidate Pairs:       {total_pairs:,}")
    print(f"  Average Candidates / S1:     {carr.mean():.2f}")
    print(f"  Median Candidates:           {np.median(carr):.0f}")
    print(f"  P95 Candidates:              {np.percentile(carr, 95):.0f}")
    print(f"  P99 Candidates:              {np.percentile(carr, 99):.0f}")
    print(f"  Maximum Candidates:          {carr.max():,}")
    print(f"  Zero-Candidate Count:        {zero_cands:,} ({100*zero_cands/total_written:.2f}%)")
    print(f"  Runtime:                     {t_total:.1f}s ({t_total/60:.2f} min)")
    print(f"  Peak Memory:                 {peak_ram_mb:.1f} MB ({peak_ram_mb/1024:.2f} GB)")
    print(f"  Output File Size:            {out_size_bytes / (1024**2):.2f} MB ({out_size_bytes:,} bytes)")
    print(f"  Output File Path:            {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
