"""
step2_validate_normalization.py
================================
Validate normalization.py on REAL data from all sources.

Strategy:
  - Load first 50K rows from each source
  - Apply normalize_name() and normalize_address()
  - Show before/after comparisons for spot-checking
  - Report stats: how many rows changed, avg length change, etc.
  - Check for any crashes (NaN, empty, unicode, etc.)

This is NOT the full pipeline — it's a quality gate before we commit.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import time
import pandas as pd
from normalization import normalize_name, normalize_address, extract_landmarks

# ── Paths ──
BASE = r"c:\Users\ASUS\Downloads\konachiwa\dataset-20260925T160811Z-1-001\dataset"
TRAIN = os.path.join(BASE, "train")
TEST  = os.path.join(BASE, "test")

SOURCES = {
    'Train S1': os.path.join(TRAIN, 'train_source1.tsv'),
    'Train S2': os.path.join(TRAIN, 'train_source2.tsv'),
    'Train S3': os.path.join(TRAIN, 'train_source3.tsv'),
    'Test S1':  os.path.join(TEST,  'test_source1.tsv'),
    'Test S2':  os.path.join(TEST,  'test_source2.tsv'),
    'Test S3':  os.path.join(TEST,  'test_source3.tsv'),
}

SAMPLE_SIZE = 50_000  # rows per source

print("=" * 72)
print("  STEP 2: VALIDATE NORMALIZATION ON REAL DATA")
print("=" * 72)

for source_name, path in SOURCES.items():
    if not os.path.exists(path):
        print(f"\n  [{source_name}] FILE NOT FOUND: {path}")
        continue

    print(f"\n{'─' * 72}")
    print(f"  {source_name}  ({os.path.basename(path)})")
    print(f"{'─' * 72}")

    # Load sample
    t0 = time.time()
    df = pd.read_csv(path, sep='\t', nrows=SAMPLE_SIZE, dtype=str)
    t1 = time.time()
    print(f"  Loaded {len(df):,} rows in {t1-t0:.1f}s")
    print(f"  Columns: {list(df.columns)}")

    # Fill NaN
    df['business_name'] = df['business_name'].fillna('')
    df['business_address'] = df['business_address'].fillna('')
    df['country'] = df['country'].fillna('')

    # Country distribution
    country_dist = df['country'].value_counts()
    print(f"  Countries: {dict(country_dist)}")

    # Apply normalization
    t0 = time.time()
    df['name_clean'] = [
        normalize_name(n, c) for n, c in zip(df['business_name'], df['country'])
    ]
    df['address_clean'] = [
        normalize_address(a, c) for a, c in zip(df['business_address'], df['country'])
    ]
    t1 = time.time()
    print(f"  Normalized {len(df):,} rows in {t1-t0:.1f}s "
          f"({len(df)/(t1-t0):,.0f} rows/sec)")

    # ── Name Stats ──
    name_changed = (df['business_name'].str.lower() != df['name_clean']).sum()
    name_empty = (df['name_clean'] == '').sum()
    print(f"\n  NAME STATS:")
    print(f"    Changed after normalization: {name_changed:,} / {len(df):,} "
          f"({100*name_changed/len(df):.1f}%)")
    print(f"    Empty after normalization:   {name_empty:,}")

    # ── Address Stats ──
    addr_changed = (df['business_address'].str.lower() != df['address_clean']).sum()
    addr_empty = (df['address_clean'] == '').sum()
    print(f"\n  ADDRESS STATS:")
    print(f"    Changed after normalization: {addr_changed:,} / {len(df):,} "
          f"({100*addr_changed/len(df):.1f}%)")
    print(f"    Empty after normalization:   {addr_empty:,}")

    # ── Landmark stats (India only) ──
    india_mask = df['country'] == 'India'
    india_count = india_mask.sum()
    if india_count > 0:
        lm_counts = [
            len(extract_landmarks(a))
            for a in df.loc[india_mask, 'business_address']
        ]
        lm_with = sum(1 for c in lm_counts if c > 0)
        print(f"\n  LANDMARK STATS (India, n={india_count:,}):")
        print(f"    Addresses with landmarks: {lm_with:,} ({100*lm_with/india_count:.1f}%)")

    # ── Show 10 random before/after examples ──
    print(f"\n  SAMPLE BEFORE/AFTER (10 random rows):")
    sample = df.sample(min(10, len(df)), random_state=42)
    for _, row in sample.iterrows():
        print(f"\n    [{row['country']}]")
        print(f"      Name:     {row['business_name']!r}")
        print(f"      Clean:    {row['name_clean']!r}")
        print(f"      Address:  {row['business_address']!r}")
        print(f"      Clean:    {row['address_clean']!r}")

    # ── Check for potential problems ──
    print(f"\n  QUALITY CHECKS:")

    # Check: no name became longer than original * 2 (sign of runaway expansion)
    orig_lens = df['business_name'].str.len()
    clean_lens = df['name_clean'].str.len()
    ratio = clean_lens / orig_lens.clip(lower=1)
    exploded = (ratio > 2.0).sum()
    print(f"    Names that grew >2x: {exploded:,}")

    # Check: no address totally destroyed (clean is empty but original wasn't)
    destroyed = ((df['business_address'] != '') & (df['address_clean'] == '')).sum()
    print(f"    Addresses destroyed (non-empty -> empty): {destroyed:,}")

    # Check: commas preserved
    orig_commas = df['business_address'].str.count(',').sum()
    clean_commas = df['address_clean'].str.count(',').sum()
    print(f"    Commas preserved: {clean_commas:,} / {orig_commas:,}")

print(f"\n{'=' * 72}")
print("  VALIDATION COMPLETE")
print(f"{'=' * 72}")
