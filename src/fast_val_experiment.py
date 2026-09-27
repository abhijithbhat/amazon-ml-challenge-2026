#!/usr/bin/env python3
"""
src/fast_val_experiment.py
==========================
Fast Ground-Truth Validation on 3,000 Held-Out Train Split Entities.
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
    t_global_start = time.time()
    print("=" * 76)
    print("  FAST GROUND-TRUTH VALIDATION EXPERIMENT (3,000 HELD-OUT S1 ENTITIES)")
    print("=" * 76)

    VAL_START = 10_000
    VAL_COUNT = 3_000

    # 1. Load S1 validation sample
    print(f"\n[1/5] Loading {VAL_COUNT:,} held-out S1 validation entities (rows {VAL_START:,}..{VAL_START+VAL_COUNT:,})...")
    df_s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", skiprows=range(1, VAL_START+1), nrows=VAL_COUNT, dtype=str).fillna("")
    print(f"  Loaded {len(df_s1):,} S1 rows.")

    val_s1_ids = df_s1["entity_id"].tolist()
    val_s1_set = set(val_s1_ids)

    # 2. Load Ground Truth for validation sample
    print("\n[2/5] Loading Ground Truth for validation entities...")
    gt_map = {}
    target_vendor_ids = set()
    with open("dataset/train/train_ground_truth.tsv") as f:
        f.readline()
        for line in f:
            p = line.strip().split("\t")
            if p[0] in val_s1_set:
                m_ids = set(p[1].split(",")) if len(p) > 1 and p[1] else set()
                gt_map[p[0]] = m_ids
                target_vendor_ids.update(m_ids)

    for s1 in val_s1_ids:
        if s1 not in gt_map:
            gt_map[s1] = set()

    singletons_count = sum(1 for s1 in val_s1_ids if len(gt_map[s1]) == 0)
    matched_count = len(val_s1_ids) - singletons_count
    print(f"  Validation Ground Truth Profile:")
    print(f"    - Total S1 Entities:     {len(val_s1_ids):,}")
    print(f"    - True Singletons:       {singletons_count:,} ({singletons_count/len(val_s1_ids)*100:.2f}%)")
    print(f"    - True Matched Entities: {matched_count:,} ({matched_count/len(val_s1_ids)*100:.2f}%)")
    print(f"    - Target Vendor IDs:     {len(target_vendor_ids):,}")

    cluster_sizes = [len(gt_map[s1]) for s1 in val_s1_ids if len(gt_map[s1]) > 0]
    c_counts = Counter(cluster_sizes)
    print("    - Top cluster sizes:", c_counts.most_common(8))
    print(f"    - Clusters > 5 matches:  {sum(1 for s in cluster_sizes if s > 5):,} ({sum(1 for s in cluster_sizes if s > 5)/len(val_s1_ids)*100:.2f}%)")

    # 3. Stream S2 & S3 to collect Target Vendor Records + Lookalike Sample
    print("\n[3/5] Streaming Source 2 & Source 3 for target records...")
    t0 = time.time()
    vendor_lookup = {}
    extra_neg_ids = set()

    for path, label in [("dataset/train/train_source2.tsv", "Source 2"), ("dataset/train/train_source3.tsv", "Source 3")]:
        t_src = time.time()
        for chunk in pd.read_csv(path, sep="\t", chunksize=250_000, usecols=["entity_id", "business_name", "business_address", "country"], dtype=str):
            chunk = chunk.fillna("")
            m = chunk[chunk["entity_id"].isin(target_vendor_ids)]
            for _, r in m.iterrows():
                veid = r["entity_id"]
                c = r["country"].strip()
                nc = fold_accents_and_clean(normalize_name(r["business_name"], c))
                ac = fold_accents_and_clean(normalize_address(r["business_address"], c))
                digits = extract_significant_digits(ac)
                vendor_lookup[veid] = (nc, ac, c.upper(), digits)
            
            # Also grab up to 1,500 negative lookalikes from chunk
            if len(extra_neg_ids) < 3_000:
                sample_neg = chunk.iloc[:200]
                for _, r in sample_neg.iterrows():
                    veid = r["entity_id"]
                    extra_neg_ids.add(veid)
                    c = r["country"].strip()
                    nc = fold_accents_and_clean(normalize_name(r["business_name"], c))
                    ac = fold_accents_and_clean(normalize_address(r["business_address"], c))
                    digits = extract_significant_digits(ac)
                    vendor_lookup[veid] = (nc, ac, c.upper(), digits)

        print(f"  {label} streamed in {time.time()-t_src:.2f}s (retained {len(vendor_lookup):,} total vendors).")

    # Normalize S1
    s1_lookup = {}
    s1_prefix_index = defaultdict(list)
    for _, r in df_s1.iterrows():
        eid = r["entity_id"]
        c = r["country"].strip()
        nc = fold_accents_and_clean(normalize_name(r["business_name"], c))
        ac = fold_accents_and_clean(normalize_address(r["business_address"], c))
        digits = extract_significant_digits(ac)
        s1_lookup[eid] = (nc, ac, c.upper(), digits)
        pre = nc[:4].lower() if len(nc) >= 4 else nc.lower()
        s1_prefix_index[pre].append(eid)

    # Invert vendor index by prefix to generate hard negative pairs
    vendor_prefix_index = defaultdict(list)
    for veid, (v_nc, v_ac, v_c, v_digits) in vendor_lookup.items():
        pre = v_nc[:4].lower() if len(v_nc) >= 4 else v_nc.lower()
        vendor_prefix_index[pre].append(veid)

    # Build candidate pairs: True matches + Prefix sharing negatives + general negatives
    s1_candidates = defaultdict(set)
    for s1, true_matches in gt_map.items():
        s1_candidates[s1].update(true_matches)
        s1_nc, s1_ac, s1_c, s1_digits = s1_lookup[s1]
        pre = s1_nc[:4].lower() if len(s1_nc) >= 4 else s1_nc.lower()
        for veid in vendor_prefix_index.get(pre, [])[:8]:
            if len(s1_candidates[s1]) < 12:
                s1_candidates[s1].add(veid)

    total_pairs = sum(len(cands) for cands in s1_candidates.values())
    print(f"  Harvested {total_pairs:,} total candidate pairs for scoring.")

    # 4. Score candidate pairs with trained model
    print("\n[4/5] Scoring candidate pairs with trained model...")
    t_score = time.time()
    with open("trained_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_cols = bundle.get("feature_cols", FEATURE_COLS)
    if hasattr(model, "n_jobs"):
        model.n_jobs = -1

    all_scored_pairs = []
    batch_s1, batch_vx, batch_dstat, batch_country, batch_feats = [], [], [], [], []

    def flush():
        if not batch_feats:
            return
        X = np.asarray(batch_feats, dtype=np.float32)
        probs = model.predict_proba(X)[:, 1]
        for s1, vx, dstat, prob, c in zip(batch_s1, batch_vx, batch_dstat, probs, batch_country):
            all_scored_pairs.append((float(prob), s1, vx, dstat, c))
        batch_s1.clear(); batch_vx.clear(); batch_dstat.clear(); batch_country.clear(); batch_feats.clear()

    for s1, cand_set in s1_candidates.items():
        s1_nc, s1_ac, s1_c, s1_digits = s1_lookup[s1]
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
    print(f"  Scored {len(all_scored_pairs):,} pairs in {time.time()-t_score:.2f}s.")

    # 5. Evaluate Decision Rules Matrix
    print("\n" + "=" * 80)
    print("  [5/5] EXACT MACRO F0.5 EVALUATION MATRIX")
    print("=" * 80)

    def evaluate(name, b_conf, b_abs, delta_gap, max_tail_rank, req_confirm_tail=True):
        s1_cands = defaultdict(list)
        for prob, s1, vx, dstat, country in all_scored_pairs:
            if dstat == "conflict":
                continue
            is_tight = (country in ("FRANCE", "FR", "BE", "BELGIUM") or "FR" in country)
            eff_conf = 0.88 if is_tight else b_conf
            eff_abs = 0.93 if is_tight else b_abs

            if dstat == "confirm" and prob >= eff_conf:
                s1_cands[s1].append((prob, vx, dstat))
            elif dstat == "absent" and prob >= eff_abs:
                s1_cands[s1].append((prob, vx, dstat))

        surviving = []
        for s1, cands in s1_cands.items():
            cands.sort(key=lambda x: x[0], reverse=True)
            r1 = cands[0][0]
            for rank, (prob, vx, dstat) in enumerate(cands, start=1):
                if rank == 1:
                    surviving.append((prob, s1, vx))
                elif rank in (2, 3):
                    if delta_gap is not None:
                        if prob >= 0.88 and (r1 - prob) <= delta_gap:
                            surviving.append((prob, s1, vx))
                    else:
                        surviving.append((prob, s1, vx))
                elif rank in (4, 5):
                    if max_tail_rank is not None and rank > max_tail_rank:
                        break
                    if delta_gap is not None:
                        if (not req_confirm_tail or dstat == "confirm") and prob >= 0.92 and (r1 - prob) <= (delta_gap if delta_gap <= 0.05 else 0.05):
                            surviving.append((prob, s1, vx))
                    else:
                        surviving.append((prob, s1, vx))
                else:
                    if max_tail_rank is not None and rank > max_tail_rank:
                        break
                    surviving.append((prob, s1, vx))

        surviving.sort(key=lambda x: x[0], reverse=True)
        assigned = set()
        pred_map = defaultdict(set)
        for prob, s1, vx in surviving:
            if vx in assigned:
                continue
            assigned.add(vx)
            pred_map[s1].add(vx)

        f05_list, p_list, r_list = [], [], []
        pred_sing = 0
        for s1 in val_s1_ids:
            preds = pred_map.get(s1, set())
            if not preds:
                pred_sing += 1
            f, p, r = compute_entity_metrics(gt_map[s1], preds)
            f05_list.append(f)
            p_list.append(p)
            r_list.append(r)

        mf = np.mean(f05_list)
        mp = np.mean(p_list)
        mr = np.mean(r_list)
        print(f"  {name:<45} | Macro F0.5: {mf:.4f} | Prec: {mp:.4f} | Rec: {mr:.4f} | Sing: {pred_sing/len(val_s1_ids)*100:.2f}% | Vendors: {len(assigned):,}")
        return mf

    print("\n--- Benchmark 1: 0.708 Baseline vs 0.707 Precision-Tuning ---")
    score_0708 = evaluate("0.708 Baseline (No Cap, No Delta, 0.83/0.90)", 0.83, 0.90, None, None)
    score_0707 = evaluate("0.707 Precision (Cap<=5, Delta 0.08, 0.85/0.91)", 0.85, 0.91, 0.08, 5)

    print("\n--- Benchmark 2: Ablation Analysis (Why did 0.707 drop?) ---")
    evaluate("Ablation 1: Only Tail Cap (Rank<=5, No Delta, 0.83)", 0.83, 0.90, None, 5)
    evaluate("Ablation 2: Only Delta Gap (Delta 0.08, No Cap, 0.83)", 0.83, 0.90, 0.08, None)
    evaluate("Ablation 3: Only Baseline Shift (0.85/0.91, No Cap, No Gap)", 0.85, 0.91, None, None)

    print("\n--- Benchmark 3: Tail Cap Sweep (0.83/0.90) ---")
    for cap in [6, 7, 8, 9, 10, None]:
        evaluate(f"Tail Cap Rank <= {str(cap):<4} (No Delta Gap, 0.83/0.90)", 0.83, 0.90, None, cap)

    print("\n--- Benchmark 4: Threshold Grid Search (No Tail Cap, No Delta Gap) ---")
    best_score = 0.0
    best_cfg = None
    for conf in [0.81, 0.82, 0.83, 0.84, 0.85, 0.86]:
        for abs_th in [0.88, 0.90, 0.91, 0.92]:
            s = evaluate(f"Grid: Confirm={conf:.2f}, Absent={abs_th:.2f}", conf, abs_th, None, None)
            if s > best_score:
                best_score = s
                best_cfg = (conf, abs_th)

    print("\n" + "=" * 80)
    print(f"  VALIDATION SUMMARY:")
    print(f"    - 0.708 Baseline Validation Score: {score_0708:.4f}")
    print(f"    - 0.707 Run Validation Score:     {score_0707:.4f}")
    print(f"    - Best Configuration on Val:      Confirm={best_cfg[0]:.2f}, Absent={best_cfg[1]:.2f} (Score: {best_score:.4f})")
    print(f"    - Delta vs Baseline:              {best_score - score_0708:+.4f}")
    print(f"  Total Runtime: {time.time()-t_global_start:.1f}s")
    print("=" * 80)

if __name__ == "__main__":
    main()
