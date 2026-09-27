#!/usr/bin/env python3
"""
src/validation_experiment.py
============================
Rigorous Ground-Truth Validation on Held-Out Train Split (Macro F0.5)

Purpose:
1. Harvest candidates and score pairs for 6,000 held-out S1 train entities (rows 10,000 to 16,000).
2. Measure exact Macro F0.5 against ground truth.
3. Compare:
   - 0.708 Baseline Configuration (no tail cap, no delta gap, threshold 0.83 / 0.90)
   - 0.707 Precision-Tuned Configuration (tail cap <= 5, delta gap 0.08, threshold 0.85 / 0.91)
   - Ablation A: Tail pruning only (rank <= 5) without delta gap
   - Ablation B: Delta gap only (0.08) without tail pruning
   - Ablation C: Tail cap at rank <= 8 or <= 10
   - Ablation D: Threshold sweep (0.83, 0.84, 0.85, 0.86, 0.87, 0.88)
   - Ablation E: Relative delta gap sweep (0.05, 0.08, 0.12, 0.15, None)
4. Pinpoint the exact reason 0.708 outperformed 0.707 and find the globally optimal configuration.
"""

import os
import sys
import io
import time
import pickle
import re
import unicodedata
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

sys.path.insert(0, "src")
from normalization import normalize_name, normalize_address
from blocking_features import get_blocking_keys, extract_significant_tokens
from comparison_features import compute_pair_features, FEATURE_COLS, jaro_winkler_similarity

# C-level RapidFuzz
try:
    from rapidfuzz.fuzz import quick_ratio as _rf_quick_ratio
except ImportError:
    from rapidfuzz.fuzz import ratio as _rf_quick_ratio

try:
    from rapidfuzz.distance.JaroWinkler import similarity as _rf_jaro_winkler
except ImportError:
    _rf_jaro_winkler = None

_DIGIT_RE = re.compile(r"\d+")

def calc_addr_jw(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if _rf_jaro_winkler is not None:
        return float(_rf_jaro_winkler(s1, s2))
    return float(jaro_winkler_similarity(s1, s2))

def extract_significant_digits(text: str) -> Set[str]:
    if not text:
        return set()
    raw = set(_DIGIT_RE.findall(text))
    return {d for d in raw if d not in {"0", "1", "2"}}

def get_digit_status(digits_a: Set[str], digits_b: Set[str]) -> str:
    if not digits_a or not digits_b:
        return "absent"
    if digits_a & digits_b:
        return "confirm"
    return "conflict"

_FRENCH_LEGAL_SUFFIXES = re.compile(
    r"\b(?:sarl|sas|sasu|sa|eurl|sci|snc|earl|gaec|gie|scop|scic)\b"
)

def fold_accents_and_clean(text: str) -> str:
    if not text:
        return text
    nfkd = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in nfkd if unicodedata.category(ch) != "Mn")
    stripped = _FRENCH_LEGAL_SUFFIXES.sub("", stripped)
    return re.sub(r"\s+", " ", stripped).strip()

def compute_entity_metrics(gt_ids: Set[str], pred_ids: Set[str]) -> Tuple[float, float, float]:
    """Compute (F_0.5, Precision, Recall) for a single Source 1 entity."""
    if not gt_ids:
        if not pred_ids:
            return 1.0, 1.0, 1.0
        return 0.0, 0.0, 0.0

    if not pred_ids:
        return 0.0, 0.0, 0.0

    tp = len(gt_ids & pred_ids)
    if tp == 0:
        return 0.0, 0.0, 0.0

    precision = tp / len(pred_ids)
    recall = tp / len(gt_ids)
    denom = 0.25 * precision + recall
    f05 = (1.25 * precision * recall) / denom if denom > 0 else 0.0
    return f05, precision, recall

def main():
    print("=" * 76)
    print("  GROUND-TRUTH VALIDATION EXPERIMENT ON HELD-OUT TRAIN SPLIT")
    print("=" * 76)

    CACHE_FILE = "output/val_scored_pairs_sample.pkl"
    VAL_START = 10_000
    VAL_COUNT = 6_000

    # 1. Load S1 validation sample
    print(f"\n[1/5] Loading {VAL_COUNT:,} held-out S1 validation entities (rows {VAL_START:,}..{VAL_START+VAL_COUNT:,})...")
    df_s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", skiprows=range(1, VAL_START+1), nrows=VAL_COUNT, dtype=str).fillna("")
    print(f"  Loaded {len(df_s1):,} S1 rows.")

    val_s1_ids = df_s1["entity_id"].tolist()
    val_s1_set = set(val_s1_ids)

    # 2. Load Ground Truth for validation sample
    print("\n[2/5] Loading Ground Truth for validation entities...")
    gt_map = {}
    with open("dataset/train/train_ground_truth.tsv") as f:
        f.readline()
        for line in f:
            p = line.strip().split("\t")
            if p[0] in val_s1_set:
                m_ids = set(p[1].split(",")) if len(p) > 1 and p[1] else set()
                gt_map[p[0]] = m_ids

    # Fill empty for entities with no GT line (singletons)
    for s1 in val_s1_ids:
        if s1 not in gt_map:
            gt_map[s1] = set()

    singletons_count = sum(1 for s1 in val_s1_ids if len(gt_map[s1]) == 0)
    matched_count = len(val_s1_ids) - singletons_count
    print(f"  Validation Ground Truth Profile:")
    print(f"    - Total S1 Entities: {len(val_s1_ids):,}")
    print(f"    - True Singletons:   {singletons_count:,} ({singletons_count/len(val_s1_ids)*100:.2f}%)")
    print(f"    - True Matched S1:   {matched_count:,} ({matched_count/len(val_s1_ids)*100:.2f}%)")

    gt_cluster_sizes = [len(gt_map[s1]) for s1 in val_s1_ids if len(gt_map[s1]) > 0]
    print(f"    - Cluster size distribution: {Counter(gt_cluster_sizes).most_common(10)}")
    print(f"    - Entities with > 5 true matches: {sum(1 for s in gt_cluster_sizes if s > 5):,} ({sum(1 for s in gt_cluster_sizes if s > 5)/len(val_s1_ids)*100:.2f}%)")

    # Check if cached validation pairs exist
    all_scored_pairs = []
    if os.path.isfile(CACHE_FILE):
        print(f"\n[3/5] Loading cached validation candidate pairs from {CACHE_FILE}...")
        with open(CACHE_FILE, "rb") as f:
            all_scored_pairs = pickle.load(f)
        print(f"  Loaded {len(all_scored_pairs):,} cached pairs.")
    else:
        print("\n[3/5] Generating validation candidate pairs & scoring...")
        t0 = time.time()

        # Build S1 lookup and blocking index
        s1_lookup = {}
        s1_block_index = defaultdict(list)

        for _, r in df_s1.iterrows():
            eid = r["entity_id"]
            c = r["country"].strip()
            nc = fold_accents_and_clean(normalize_name(r["business_name"], c))
            ac = fold_accents_and_clean(normalize_address(r["business_address"], c))
            digits = extract_significant_digits(ac)
            s1_lookup[eid] = (nc, ac, c.upper(), digits)

            r_dict = {"entity_id": eid, "business_name": r["business_name"], "business_address": r["business_address"], "country": c}
            keys = get_blocking_keys(r_dict)
            for k in keys:
                s1_block_index[k].append(eid)

        print(f"  Built blocking index with {len(s1_block_index):,} keys in {time.time()-t0:.2f}s.")

        # Stream S2 & S3 to find candidates
        val_candidate_pairs = defaultdict(set)
        vendor_lookup = {}

        # Also collect target GT IDs so candidates include true matches (recall upper bound)
        target_gt_ids = set()
        for s1, m_set in gt_map.items():
            target_gt_ids.update(m_set)

        for src_path, label in [("dataset/train/train_source2.tsv", "Source 2"), ("dataset/train/train_source3.tsv", "Source 3")]:
            print(f"  Streaming {label}...")
            t_s = time.time()
            chunk_size = 100_000
            for chunk in pd.read_csv(src_path, sep="\t", chunksize=chunk_size, dtype=str):
                chunk = chunk.fillna("")
                for _, r in chunk.iterrows():
                    veid = r["entity_id"]
                    is_target = veid in target_gt_ids
                    
                    # Quick check if keys intersect
                    r_dict = {"entity_id": veid, "business_name": r["business_name"], "business_address": r["business_address"], "country": r["country"]}
                    keys = get_blocking_keys(r_dict)
                    matched_s1 = set()
                    for k in keys:
                        if k in s1_block_index:
                            matched_s1.update(s1_block_index[k][:25])

                    if matched_s1 or is_target:
                        c = r["country"].strip()
                        nc = fold_accents_and_clean(normalize_name(r["business_name"], c))
                        ac = fold_accents_and_clean(normalize_address(r["business_address"], c))
                        digits = extract_significant_digits(ac)
                        vendor_lookup[veid] = (nc, ac, c.upper(), digits)

                        for s1 in matched_s1:
                            if len(val_candidate_pairs[s1]) < 12:
                                val_candidate_pairs[s1].add(veid)
                        
                        if is_target:
                            for s1, m_set in gt_map.items():
                                if veid in m_set:
                                    val_candidate_pairs[s1].add(veid)

            print(f"    - {label} processed in {time.time()-t_s:.2f}s.")

        total_cand_pairs = sum(len(cands) for cands in val_candidate_pairs.values())
        print(f"  Total candidate pairs harvested: {total_cand_pairs:,}")

        # Load trained model
        print("  Loading trained model bundle...")
        with open("trained_model.pkl", "rb") as f:
            bundle = pickle.load(f)
        model = bundle["model"]
        feature_cols = bundle.get("feature_cols", FEATURE_COLS)
        if hasattr(model, "n_jobs"):
            model.n_jobs = -1

        # Score candidate pairs using 3-tier cascade
        print("  Scoring candidate pairs...")
        t_score = time.time()
        batch_s1, batch_vx, batch_dstat, batch_country, batch_feats = [], [], [], [], []

        def flush():
            if not batch_feats:
                return
            X = np.asarray(batch_feats, dtype=np.float32)
            probs = model.predict_proba(X)[:, 1]
            for s1, vx, dstat, prob, c in zip(batch_s1, batch_vx, batch_dstat, probs, batch_country):
                all_scored_pairs.append((float(prob), s1, vx, dstat, c))
            batch_s1.clear(); batch_vx.clear(); batch_dstat.clear(); batch_country.clear(); batch_feats.clear()

        for s1, cand_set in val_candidate_pairs.items():
            s1_info = s1_lookup.get(s1)
            if not s1_info:
                continue
            s1_nc, s1_ac, s1_c, s1_digits = s1_info

            for vx in cand_set:
                v_info = vendor_lookup.get(vx)
                if not v_info:
                    continue
                v_nc, v_ac, v_c, v_digits = v_info

                dstat = get_digit_status(s1_digits, v_digits)

                # Tier 1
                sim = _rf_quick_ratio(s1_nc, v_nc)
                if sim < 45.0 and dstat != "confirm":
                    continue

                # Tier 2
                if s1_nc == v_nc and len(s1_nc) >= 6 and bool(s1_digits & v_digits) and calc_addr_jw(s1_ac, v_ac) >= 0.80:
                    all_scored_pairs.append((1.0, s1, vx, "confirm", s1_c))
                    continue

                # Tier 3
                row_a = {"entity_id": s1, "name_clean": s1_nc, "address_clean": s1_ac, "country": s1_c}
                row_b = {"entity_id": vx, "name_clean": v_nc, "address_clean": v_ac, "country": v_c}
                fd = compute_pair_features(row_a, row_b)
                batch_s1.append(s1); batch_vx.append(vx); batch_dstat.append(dstat); batch_country.append(s1_c)
                batch_feats.append([fd[col] for col in feature_cols])

                if len(batch_feats) >= 10_000:
                    flush()
        flush()

        print(f"  Scoring completed in {time.time()-t_score:.2f}s. Total pairs: {len(all_scored_pairs):,}")
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(all_scored_pairs, f)
        print(f"  Saved validation scored pairs to {CACHE_FILE}.")

    # 4. Evaluation Engine
    def evaluate_configuration(name, baseline_confirm, baseline_absent, delta_gap, max_tail_rank, require_confirm_tail=True):
        """Runs the exact decision pipeline and computes macro F0.5 against ground truth."""
        # Step 1: Filter baseline
        s1_cands = defaultdict(list)
        for prob, s1, vx, dstat, country in all_scored_pairs:
            if dstat == "conflict":
                continue
            is_tight = (country in ("FRANCE", "FR", "BE", "BELGIUM") or "FR" in country)
            eff_confirm = 0.88 if is_tight else baseline_confirm
            eff_absent = 0.93 if is_tight else baseline_absent

            if dstat == "confirm" and prob >= eff_confirm:
                s1_cands[s1].append((prob, vx, dstat, country))
            elif dstat == "absent" and prob >= eff_absent:
                s1_cands[s1].append((prob, vx, dstat, country))

        # Step 2: Rank decay and tail pruning
        surviving = []
        for s1, cands in s1_cands.items():
            cands.sort(key=lambda x: x[0], reverse=True)
            rank1_prob = cands[0][0]

            for rank, (prob, vx, dstat, country) in enumerate(cands, start=1):
                if rank == 1:
                    surviving.append((prob, s1, vx))
                elif rank in (2, 3):
                    if delta_gap is not None:
                        if prob >= 0.88 and (rank1_prob - prob) <= delta_gap:
                            surviving.append((prob, s1, vx))
                    else:
                        surviving.append((prob, s1, vx))
                elif rank in (4, 5):
                    if max_tail_rank is not None and rank > max_tail_rank:
                        break
                    if delta_gap is not None:
                        if (not require_confirm_tail or dstat == "confirm") and prob >= 0.92 and (rank1_prob - prob) <= (delta_gap if delta_gap <= 0.05 else 0.05):
                            surviving.append((prob, s1, vx))
                    else:
                        surviving.append((prob, s1, vx))
                else:
                    if max_tail_rank is not None and rank > max_tail_rank:
                        break
                    surviving.append((prob, s1, vx))

        # Step 3: Global greedy 1-to-1 disjoint assignment
        surviving.sort(key=lambda x: x[0], reverse=True)
        assigned_vendors = set()
        pred_map = defaultdict(set)
        for prob, s1, vx in surviving:
            if vx in assigned_vendors:
                continue
            assigned_vendors.add(vx)
            pred_map[s1].add(vx)

        # Step 4: Compute Macro Metrics
        f05_list, p_list, r_list = [], [], []
        pred_singletons = 0
        for s1 in val_s1_ids:
            preds = pred_map.get(s1, set())
            if not preds:
                pred_singletons += 1
            f, p, r = compute_entity_metrics(gt_map[s1], preds)
            f05_list.append(f)
            p_list.append(p)
            r_list.append(r)

        macro_f05 = np.mean(f05_list)
        macro_p = np.mean(p_list)
        macro_r = np.mean(r_list)
        singletons_pct = pred_singletons / len(val_s1_ids) * 100

        print(f"  {name:<42} | Macro F0.5: {macro_f05:.4f} | Prec: {macro_p:.4f} | Rec: {macro_r:.4f} | Singletons: {pred_singletons:,} ({singletons_pct:.2f}%) | Vendors: {len(assigned_vendors):,}")
        return macro_f05, macro_p, macro_r, pred_singletons, len(assigned_vendors)

    print("\n" + "=" * 80)
    print("  EXPERIMENT MATRIX: EVALUATING CONFIGURATIONS")
    print("=" * 80)

    # 1. 0.708 Baseline Configuration
    print("\n--- 1. Baseline vs Recent Run ---")
    evaluate_configuration("0.708 Baseline (No Cap, No Delta Gap, 0.83/0.90)", baseline_confirm=0.83, baseline_absent=0.90, delta_gap=None, max_tail_rank=None)
    evaluate_configuration("0.707 Precision Run (Cap<=5, Delta 0.08, 0.85/0.91)", baseline_confirm=0.85, baseline_absent=0.91, delta_gap=0.08, max_tail_rank=5)

    print("\n--- 2. Ablation Analysis: Isolating Components ---")
    evaluate_configuration("Ablation: Only Tail Cap (Rank<=5, No Delta, 0.83)", baseline_confirm=0.83, baseline_absent=0.90, delta_gap=None, max_tail_rank=5)
    evaluate_configuration("Ablation: Only Delta Gap (Delta 0.08, No Cap, 0.83)", baseline_confirm=0.83, baseline_absent=0.90, delta_gap=0.08, max_tail_rank=None)
    evaluate_configuration("Ablation: Only Higher Baseline (0.85/0.91, No Cap, No Gap)", baseline_confirm=0.85, baseline_absent=0.91, delta_gap=None, max_tail_rank=None)

    print("\n--- 3. Tail Cap Sweep (0.83/0.90 baseline) ---")
    evaluate_configuration("Tail Cap Rank <= 6", baseline_confirm=0.83, baseline_absent=0.90, delta_gap=None, max_tail_rank=6)
    evaluate_configuration("Tail Cap Rank <= 7", baseline_confirm=0.83, baseline_absent=0.90, delta_gap=None, max_tail_rank=7)
    evaluate_configuration("Tail Cap Rank <= 8", baseline_confirm=0.83, baseline_absent=0.90, delta_gap=None, max_tail_rank=8)
    evaluate_configuration("Tail Cap Rank <= 10", baseline_confirm=0.83, baseline_absent=0.90, delta_gap=None, max_tail_rank=10)

    print("\n--- 4. Threshold Sweep (No Tail Cap, No Delta Gap) ---")
    for b_conf, b_abs in [(0.81, 0.88), (0.82, 0.89), (0.83, 0.90), (0.84, 0.90), (0.84, 0.91), (0.85, 0.91), (0.86, 0.92)]:
        evaluate_configuration(f"Threshold: Confirm={b_conf:.2f}, Absent={b_abs:.2f}", baseline_confirm=b_conf, baseline_absent=b_abs, delta_gap=None, max_tail_rank=None)

    print("\n--- 5. Gentle Delta Gap Sweep (Preserving multi-vendor clusters) ---")
    for gap in [0.10, 0.12, 0.15, 0.20]:
        evaluate_configuration(f"Gentle Delta Gap = {gap:.2f} (No Tail Cap, 0.83/0.90)", baseline_confirm=0.83, baseline_absent=0.90, delta_gap=gap, max_tail_rank=None)

    print("\n" + "=" * 80)
    print("  VALIDATION EXPERIMENT COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
