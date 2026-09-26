"""
=============================================================================
step4_feature_matrix.py  -  FEATURE MATRIX + CLASSIFIER TRAINING
=============================================================================
Role:  Data & Features  (Member 3, Pod 2)
Task:  End-to-end pipeline that:
       1. Loads & normalizes all 3 training sources
       2. Generates candidate pairs via blocking
       3. Computes 10-dimensional similarity features for each pair
       4. Labels pairs using ground truth
       5. Trains a Random Forest classifier (F_0.5 optimized)
       6. Saves the trained model and feature matrix

Usage:
    python step4_feature_matrix.py

Output:
    - feature_matrix.pkl       : labeled feature matrix (DataFrame)
    - trained_model.pkl        : trained sklearn RandomForest model
    - Console report with precision/recall/F0.5 scores
=============================================================================
"""

import os
import sys
import io
import time
import pickle
import re
import numpy as np
import pandas as pd
from collections import defaultdict

# Local imports
from normalization import normalize_name, normalize_address
from blocking_features import (
    get_blocking_keys,
    extract_postal_code,
    extract_state_code,
    extract_significant_tokens,
)
from comparison_features import (
    compute_pair_features,
    FEATURE_COLS,
)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DATA_DIR = os.path.join('dataset', 'train')
S1_FILE = os.path.join(DATA_DIR, 'train_source1.tsv')
S2_FILE = os.path.join(DATA_DIR, 'train_source2.tsv')
S3_FILE = os.path.join(DATA_DIR, 'train_source3.tsv')
GT_FILE = os.path.join(DATA_DIR, 'train_ground_truth.tsv')

MAX_BUCKET_SIZE = 100    # Cap bucket size to prevent memory blowup
NEG_POS_RATIO   = 5      # Negative-to-positive ratio for balanced training
RANDOM_SEED     = 42


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1: LOAD & NORMALIZE DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_normalize(filepath, source_label):
    """Load a source TSV and add name_clean + address_clean columns."""
    t0 = time.time()
    print(f"\n  Loading {source_label} from {filepath}...")

    df = pd.read_csv(filepath, sep='\t', dtype=str, keep_default_na=False)
    print(f"    Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    t1 = time.time()
    df['name_clean'] = [
        normalize_name(n, c) for n, c in zip(df['business_name'], df['country'])
    ]
    df['address_clean'] = [
        normalize_address(a, c) for a, c in zip(df['business_address'], df['country'])
    ]
    print(f"    Normalized in {time.time()-t1:.1f}s")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2: BUILD BLOCKING INDEX
# ═══════════════════════════════════════════════════════════════════════════════

def build_blocking_index(df, label):
    """
    Build an inverted index: blocking_key -> list of row indices.
    Caps each bucket at MAX_BUCKET_SIZE to prevent blowup.
    """
    t0 = time.time()
    print(f"\n  Building blocking index for {label}...")

    index = defaultdict(list)
    for idx, row in df.iterrows():
        keys = get_blocking_keys(row)
        for key in keys:
            if len(index[key]) < MAX_BUCKET_SIZE:
                index[key].append(idx)

    n_keys = len(index)
    n_entries = sum(len(v) for v in index.values())
    print(f"    {n_keys:,} distinct keys, {n_entries:,} total entries "
          f"in {time.time()-t0:.1f}s")

    return index


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3: GENERATE CANDIDATE PAIRS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_candidates(df_s1, df_sx, index_s1, index_sx, sx_label):
    """
    Generate candidate pairs between S1 and Sx by finding shared blocking keys.

    Returns set of (s1_entity_id, sx_entity_id) tuples.
    """
    t0 = time.time()
    print(f"\n  Generating candidates: S1 × {sx_label}...")

    candidates = set()

    # For each blocking key in S1's index, find matching keys in Sx's index
    shared_keys = set(index_s1.keys()) & set(index_sx.keys())
    print(f"    {len(shared_keys):,} shared blocking keys")

    for key in shared_keys:
        s1_indices = index_s1[key]
        sx_indices = index_sx[key]

        for i in s1_indices:
            s1_eid = df_s1.at[i, 'entity_id']
            for j in sx_indices:
                sx_eid = df_sx.at[j, 'entity_id']
                candidates.add((s1_eid, sx_eid))

    print(f"    {len(candidates):,} unique candidate pairs in {time.time()-t0:.1f}s")
    return candidates


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4: LOAD GROUND TRUTH & LABEL PAIRS
# ═══════════════════════════════════════════════════════════════════════════════

def load_ground_truth():
    """
    Parse ground truth into a set of (s1_entity_id, matched_entity_id) pairs.
    """
    print(f"\n  Loading ground truth from {GT_FILE}...")
    gt = pd.read_csv(GT_FILE, sep='\t', dtype=str, keep_default_na=False)

    positive_pairs = set()
    s1_to_matches = {}  # for quick lookup

    for _, row in gt.iterrows():
        s1_eid = row['source1_entity_id']
        matched_ids = [x.strip() for x in row['matched_entity_ids'].split(',') if x.strip()]
        s1_to_matches[s1_eid] = set(matched_ids)
        for m_eid in matched_ids:
            positive_pairs.add((s1_eid, m_eid))

    print(f"    {len(gt):,} S1 entities with matches")
    print(f"    {len(positive_pairs):,} total positive pairs")

    return positive_pairs, s1_to_matches


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 5: BUILD FEATURE MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(candidates, positive_pairs, df_s1, df_sx,
                         sx_label, neg_pos_ratio=NEG_POS_RATIO):
    """
    Build labeled feature matrix from candidate pairs.

    Strategy for balanced training (F_0.5 alignment):
    - All positive pairs that appear in candidates are included
    - Negative pairs are sampled at `neg_pos_ratio` × positives
    """
    t0 = time.time()
    print(f"\n  Building feature matrix for S1 × {sx_label}...")

    # Build entity_id -> row index lookups
    s1_lookup = {eid: idx for idx, eid in zip(df_s1.index, df_s1['entity_id'])}
    sx_lookup = {eid: idx for idx, eid in zip(df_sx.index, df_sx['entity_id'])}

    # Classify candidates into positive and negative
    pos_candidates = []
    neg_candidates = []

    for s1_eid, sx_eid in candidates:
        if (s1_eid, sx_eid) in positive_pairs:
            pos_candidates.append((s1_eid, sx_eid))
        else:
            neg_candidates.append((s1_eid, sx_eid))

    n_pos = len(pos_candidates)
    n_neg_target = min(len(neg_candidates), n_pos * neg_pos_ratio)

    print(f"    Positive candidates found: {n_pos:,}")
    print(f"    Negative candidates total: {len(neg_candidates):,}")
    print(f"    Negative sample target:    {n_neg_target:,}")

    # Sample negatives
    rng = np.random.RandomState(RANDOM_SEED)
    if n_neg_target < len(neg_candidates):
        neg_sample_idx = rng.choice(len(neg_candidates), size=n_neg_target, replace=False)
        neg_sampled = [neg_candidates[i] for i in neg_sample_idx]
    else:
        neg_sampled = neg_candidates

    # Compute features for all selected pairs
    all_pairs = pos_candidates + neg_sampled
    labels = [1] * n_pos + [0] * len(neg_sampled)

    features_list = []
    skipped = 0

    for i, (s1_eid, sx_eid) in enumerate(all_pairs):
        if i > 0 and i % 50000 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            print(f"    ... processed {i:,}/{len(all_pairs):,} pairs "
                  f"({rate:.0f} pairs/sec)")

        s1_idx = s1_lookup.get(s1_eid)
        sx_idx = sx_lookup.get(sx_eid)

        if s1_idx is None or sx_idx is None:
            skipped += 1
            continue

        row_a = df_s1.loc[s1_idx]
        row_b = df_sx.loc[sx_idx]

        feats = compute_pair_features(row_a, row_b)
        feats['s1_entity_id'] = s1_eid
        feats['sx_entity_id'] = sx_eid
        feats['label'] = labels[i]
        features_list.append(feats)

    if skipped > 0:
        print(f"    Skipped {skipped} pairs (entity not found in source)")

    feat_df = pd.DataFrame(features_list)
    elapsed = time.time() - t0
    print(f"    Feature matrix: {len(feat_df):,} rows × "
          f"{len(feat_df.columns)} cols in {elapsed:.1f}s")

    return feat_df


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 6: TRAIN CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

def train_classifier(feat_df):
    """
    Train a Random Forest classifier optimized for F_0.5.

    Returns (model, classification_report_str, threshold).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (
        precision_score, recall_score, fbeta_score,
        classification_report, confusion_matrix
    )

    print("\n  Training Random Forest classifier...")
    t0 = time.time()

    X = feat_df[FEATURE_COLS].values
    y = feat_df['label'].values

    print(f"    Training set: {len(X):,} samples")
    print(f"    Positive: {(y == 1).sum():,}  Negative: {(y == 0).sum():,}")
    print(f"    Pos/Neg ratio: 1:{(y == 0).sum() / max((y == 1).sum(), 1):.1f}")

    # Stratified 3-fold cross-validation for evaluation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

    cv_precision = []
    cv_recall = []
    cv_f05 = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        p = precision_score(y_val, y_pred, zero_division=0)
        r = recall_score(y_val, y_pred, zero_division=0)
        f05 = fbeta_score(y_val, y_pred, beta=0.5, zero_division=0)

        cv_precision.append(p)
        cv_recall.append(r)
        cv_f05.append(f05)

        print(f"    Fold {fold+1}: Precision={p:.4f}  Recall={r:.4f}  F0.5={f05:.4f}")

    print(f"\n    CV Mean:  Precision={np.mean(cv_precision):.4f}  "
          f"Recall={np.mean(cv_recall):.4f}  F0.5={np.mean(cv_f05):.4f}")

    # Train final model on all data
    final_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    final_rf.fit(X, y)

    y_pred_all = final_rf.predict(X)
    report = classification_report(y, y_pred_all, target_names=['Non-Match', 'Match'])
    cm = confusion_matrix(y, y_pred_all)

    print(f"\n    Final model (trained on all data):")
    print(f"    {report}")
    print(f"    Confusion Matrix:")
    print(f"      {cm}")

    # Feature importance
    print(f"\n    Feature Importance:")
    importances = final_rf.feature_importances_
    for fname, imp in sorted(zip(FEATURE_COLS, importances),
                              key=lambda x: -x[1]):
        bar = '█' * int(imp * 50)
        print(f"      {fname:18s}  {imp:.4f}  {bar}")

    elapsed = time.time() - t0
    print(f"\n    Training completed in {elapsed:.1f}s")

    return final_rf, report


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  STEP 4: FEATURE MATRIX & CLASSIFIER TRAINING")
    print("=" * 72)

    overall_t0 = time.time()

    # ── Step 1: Load & Normalize ─────────────────────────────────────────
    print("\n" + "─" * 72)
    print("  PHASE 1: Load & Normalize Data")
    print("─" * 72)

    df_s1 = load_and_normalize(S1_FILE, "S1")
    df_s2 = load_and_normalize(S2_FILE, "S2")
    df_s3 = load_and_normalize(S3_FILE, "S3")

    # ── Step 2: Build Blocking Indices ───────────────────────────────────
    print("\n" + "─" * 72)
    print("  PHASE 2: Build Blocking Indices")
    print("─" * 72)

    idx_s1 = build_blocking_index(df_s1, "S1")
    idx_s2 = build_blocking_index(df_s2, "S2")
    idx_s3 = build_blocking_index(df_s3, "S3")

    # ── Step 3: Generate Candidate Pairs ─────────────────────────────────
    print("\n" + "─" * 72)
    print("  PHASE 3: Generate Candidate Pairs")
    print("─" * 72)

    cands_s1_s2 = generate_candidates(df_s1, df_s2, idx_s1, idx_s2, "S2")
    cands_s1_s3 = generate_candidates(df_s1, df_s3, idx_s1, idx_s3, "S3")

    all_candidates = cands_s1_s2 | cands_s1_s3
    print(f"\n    Total unique candidates: {len(all_candidates):,}")

    # ── Step 4: Load Ground Truth ────────────────────────────────────────
    print("\n" + "─" * 72)
    print("  PHASE 4: Load Ground Truth & Label Pairs")
    print("─" * 72)

    positive_pairs, s1_to_matches = load_ground_truth()

    # Check recall of blocking on ground truth
    gt_found = sum(1 for p in positive_pairs if p in all_candidates)
    gt_recall = gt_found / len(positive_pairs) if positive_pairs else 0
    print(f"\n    Blocking recall on GT: {gt_found:,}/{len(positive_pairs):,} "
          f"= {gt_recall:.4f} ({gt_recall*100:.2f}%)")

    # ── Step 5: Build Feature Matrix ─────────────────────────────────────
    print("\n" + "─" * 72)
    print("  PHASE 5: Compute Pairwise Features")
    print("─" * 72)

    feat_s2 = build_feature_matrix(
        cands_s1_s2, positive_pairs, df_s1, df_s2, "S2"
    )
    feat_s3 = build_feature_matrix(
        cands_s1_s3, positive_pairs, df_s1, df_s3, "S3"
    )

    feat_all = pd.concat([feat_s2, feat_s3], ignore_index=True)
    print(f"\n    Combined feature matrix: {len(feat_all):,} rows")

    # Save feature matrix
    feat_all.to_pickle('feature_matrix.pkl')
    print(f"    Saved to feature_matrix.pkl")

    # ── Step 6: Train Classifier ─────────────────────────────────────────
    print("\n" + "─" * 72)
    print("  PHASE 6: Train & Evaluate Classifier")
    print("─" * 72)

    model, report = train_classifier(feat_all)

    # Save model
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"\n    Model saved to trained_model.pkl")

    # ── Summary ──────────────────────────────────────────────────────────
    total_time = time.time() - overall_t0
    print("\n" + "=" * 72)
    print(f"  PIPELINE COMPLETE  ({total_time:.0f}s total)")
    print(f"  Feature Matrix:  {len(feat_all):,} rows × {len(FEATURE_COLS)} features")
    print(f"  Positives: {(feat_all['label']==1).sum():,}")
    print(f"  Negatives: {(feat_all['label']==0).sum():,}")
    print(f"  Model: RandomForest (300 trees, depth 12)")
    print(f"  Outputs: feature_matrix.pkl, trained_model.pkl")
    print("=" * 72)


if __name__ == '__main__':
    main()
