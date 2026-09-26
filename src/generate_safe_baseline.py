#!/usr/bin/env python3
"""
Generate a safe, 100% compliant baseline submission for ML Challenge 2026.

This script:
1. Reads all entity_ids from dataset/test/test_source1.tsv in exact order.
2. Creates output/candidate_pairs.tsv with header and empty candidate lists.
3. Creates output/matching_results.tsv with header and empty matched lists.
4. Ensures strict tab separation and unix line endings (\\n).
"""

import os
import sys
import time


def generate_baseline(
    test_source1_path: str = "dataset/test/test_source1.tsv",
    output_dir: str = "output",
):
    print("=" * 60)
    print("Generating Clean Baseline Submission")
    print("=" * 60)

    if not os.path.isfile(test_source1_path):
        raise FileNotFoundError(f"Test file not found: {test_source1_path}")

    os.makedirs(output_dir, exist_ok=True)
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    matching_path = os.path.join(output_dir, "matching_results.tsv")

    print(f"Reading entity IDs from {test_source1_path}...")
    t0 = time.time()

    entity_ids = []
    with open(test_source1_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        id_idx = header.index("entity_id") if "entity_id" in header else 0
        for line in f:
            line_str = line.rstrip("\r\n")
            if not line_str:
                continue
            parts = line_str.split("\t")
            if len(parts) > id_idx:
                entity_ids.append(parts[id_idx].strip())

    total_entities = len(entity_ids)
    print(f"Extracted {total_entities:,} entity IDs in {time.time() - t0:.2f} seconds.")

    # 1. Write output/candidate_pairs.tsv
    print(f"Writing {candidate_path}...")
    t1 = time.time()
    with open(candidate_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for eid in entity_ids:
            f.write(f"{eid}\t\n")
    print(f"Wrote {total_entities:,} rows to {candidate_path} in {time.time() - t1:.2f} seconds.")

    # 2. Write output/matching_results.tsv
    print(f"Writing {matching_path}...")
    t2 = time.time()
    with open(matching_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for eid in entity_ids:
            f.write(f"{eid}\t\n")
    print(f"Wrote {total_entities:,} rows to {matching_path} in {time.time() - t2:.2f} seconds.")

    print("=" * 60)
    print(f"Baseline generation complete in {time.time() - t0:.2f} total seconds.")
    print("=" * 60)


if __name__ == "__main__":
    generate_baseline()
