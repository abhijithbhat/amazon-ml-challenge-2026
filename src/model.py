"""
Model training and inference module for Business Entity Resolution.

This module provides:
1. `train_model`: Trains a binary LightGBM classifier using ground truth matches
   (Label = 1) and random non-matching pairs (Label = 0) with a 3:1 negative ratio,
   using `PairFeatureExtractor` from `src.features`, and saves the model to `model.pkl`.
2. `score_candidates`: Inference function that checks for `candidate_pairs_file`.
   If missing, prints 'Waiting for candidate_pairs.tsv from Tejas.'
   If present, scores each candidate pair, retains matches with probability >= threshold
   (default 0.80), formats results per source1_entity_id, and saves to the output file.
"""

import os
import pickle
import random
import time
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from lightgbm import LGBMClassifier

try:
    from src.features import PairFeatureExtractor, load_entity_dict
except ImportError:
    from features import PairFeatureExtractor, load_entity_dict


def load_training_pairs(
    ground_truth_path: str = "dataset/train/train_ground_truth.tsv",
    max_positive_pairs: Optional[int] = 30000,
    random_seed: int = 42,
) -> Tuple[List[Tuple[str, str]], Set[str], Set[str], Dict[str, Set[str]]]:
    """
    Load ground truth matches: every pair (source1_id, matched_id) is a positive example.

    Returns:
        pos_pairs: List of (source1_id, matched_id)
        needed_s1: Set of all unique source1 IDs
        needed_s23: Set of all unique matched S2/S3 IDs
        s1_matches: Dict mapping source1_id to set of its true matched IDs
    """
    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"Ground truth file not found at: {ground_truth_path}")

    pos_pairs: List[Tuple[str, str]] = []
    needed_s1: Set[str] = set()
    needed_s23: Set[str] = set()
    s1_matches: Dict[str, Set[str]] = {}

    with open(ground_truth_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        for line in f:
            line_str = line.rstrip("\r\n")
            if not line_str:
                continue
            parts = line_str.split("\t")
            if len(parts) >= 2 and parts[1].strip():
                s1_id = parts[0].strip()
                matches = [m.strip() for m in parts[1].split(",") if m.strip()]
                if not matches:
                    continue
                if s1_id not in s1_matches:
                    s1_matches[s1_id] = set()
                s1_matches[s1_id].update(matches)

                for m in matches:
                    pos_pairs.append((s1_id, m))
                    needed_s1.add(s1_id)
                    needed_s23.add(m)
                    if max_positive_pairs and len(pos_pairs) >= max_positive_pairs:
                        break
            if max_positive_pairs and len(pos_pairs) >= max_positive_pairs:
                break

    return pos_pairs, needed_s1, needed_s23, s1_matches


def load_training_records(
    source_dir: str = "dataset/train",
    needed_s1: Optional[Set[str]] = None,
    needed_s23: Optional[Set[str]] = None,
    max_neg_pool_size: int = 100000,
) -> Tuple[Dict[str, Tuple[str, str]], List[Tuple[str, Tuple[str, str]]]]:
    """
    Load needed entity records from train source files and build a negative candidate pool.

    Returns:
        records: Dict mapping entity_id to (business_name, business_address)
        neg_pool: List of (entity_id, (business_name, business_address)) for S2/S3
    """
    records: Dict[str, Tuple[str, str]] = {}
    neg_pool: List[Tuple[str, Tuple[str, str]]] = []

    # Stream Source 1
    s1_path = os.path.join(source_dir, "train_source1.tsv")
    if os.path.exists(s1_path):
        with open(s1_path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\r\n").split("\t")
            id_idx = header.index("entity_id") if "entity_id" in header else 0
            name_idx = header.index("business_name") if "business_name" in header else 1
            addr_idx = header.index("business_address") if "business_address" in header else 2
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) > max(id_idx, name_idx, addr_idx):
                    eid = parts[id_idx].strip()
                    if needed_s1 is None or eid in needed_s1:
                        records[eid] = (parts[name_idx].strip(), parts[addr_idx].strip())
                        if needed_s1 and len(records) >= len(needed_s1):
                            break

    # Stream Source 2 and Source 3
    for s_name in ("train_source2.tsv", "train_source3.tsv"):
        s_path = os.path.join(source_dir, s_name)
        if not os.path.exists(s_path):
            continue
        with open(s_path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\r\n").split("\t")
            id_idx = header.index("entity_id") if "entity_id" in header else 0
            name_idx = header.index("business_name") if "business_name" in header else 1
            addr_idx = header.index("business_address") if "business_address" in header else 2
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) > max(id_idx, name_idx, addr_idx):
                    eid = parts[id_idx].strip()
                    rec = (parts[name_idx].strip(), parts[addr_idx].strip())
                    if needed_s23 is None or eid in needed_s23:
                        records[eid] = rec
                    elif len(neg_pool) < max_neg_pool_size:
                        neg_pool.append((eid, rec))

    return records, neg_pool


def build_training_dataset(
    pos_pairs: List[Tuple[str, str]],
    records: Dict[str, Tuple[str, str]],
    neg_pool: List[Tuple[str, Tuple[str, str]]],
    s1_matches: Dict[str, Set[str]],
    negative_ratio: int = 3,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build training dataset (X, y) with positive examples (Label = 1) and
    random non-matching records from Source 2 or Source 3 (Label = 0).
    """
    rng = random.Random(random_seed)
    extractor = PairFeatureExtractor()

    X: List[List[float]] = []
    y: List[int] = []

    neg_pool_len = len(neg_pool)
    if neg_pool_len == 0:
        raise ValueError("Negative pool is empty. Cannot generate negative examples.")

    for s1_id, matched_id in pos_pairs:
        if s1_id not in records or matched_id not in records:
            continue
        rec_s1 = records[s1_id]
        rec_matched = records[matched_id]

        # 1. Positive pair (Label = 1)
        feat_pos = extractor.extract_feature_vector(rec_s1, rec_matched)
        X.append(feat_pos)
        y.append(1)

        # 2. For every positive pair, randomly pick 3 non-matching records (Label = 0)
        true_m = s1_matches.get(s1_id, set())
        negs_sampled = 0
        attempts = 0
        while negs_sampled < negative_ratio and attempts < 30:
            attempts += 1
            rand_eid, rand_rec = neg_pool[rng.randint(0, neg_pool_len - 1)]
            if rand_eid not in true_m:
                feat_neg = extractor.extract_feature_vector(rec_s1, rand_rec)
                X.append(feat_neg)
                y.append(0)
                negs_sampled += 1

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train_model(
    ground_truth_path: str = "dataset/train/train_ground_truth.tsv",
    source_dir: str = "dataset/train",
    model_save_path: str = "model.pkl",
    max_positive_pairs: Optional[int] = 30000,
    random_seed: int = 42,
) -> LGBMClassifier:
    """
    Train a LightGBM binary classifier using ground truth matches and negative sampling,
    then save the trained model to model.pkl.
    """
    print("=" * 60)
    print("Starting Model Training Pipeline")
    print("=" * 60)
    t0 = time.time()

    # Step 1: Load ground truth matches
    print(f"[1/4] Loading ground truth matches from {ground_truth_path}...")
    pos_pairs, needed_s1, needed_s23, s1_matches = load_training_pairs(
        ground_truth_path=ground_truth_path,
        max_positive_pairs=max_positive_pairs,
        random_seed=random_seed,
    )
    print(f"      Loaded {len(pos_pairs)} positive pairs across {len(needed_s1)} S1 entities.")

    # Step 2: Load records for pairs + negative pool
    print(f"[2/4] Streaming source files from {source_dir}...")
    records, neg_pool = load_training_records(
        source_dir=source_dir,
        needed_s1=needed_s1,
        needed_s23=needed_s23,
    )
    print(f"      Loaded {len(records)} entity records and {len(neg_pool)} negative candidates.")

    # Step 3: Extract features and build dataset (Label = 1 and Label = 0)
    print("[3/4] Extracting features with PairFeatureExtractor (3:1 negative ratio)...")
    X, y = build_training_dataset(
        pos_pairs=pos_pairs,
        records=records,
        neg_pool=neg_pool,
        s1_matches=s1_matches,
        negative_ratio=3,
        random_seed=random_seed,
    )
    print(f"      Dataset ready: X shape={X.shape}, Positives={np.sum(y == 1)}, Negatives={np.sum(y == 0)}")

    # Step 4: Train LightGBM classifier
    print("[4/4] Training LightGBM classifier (LGBMClassifier)...")
    clf = LGBMClassifier(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_seed,
        n_jobs=1,
        verbose=-1,
    )
    clf.fit(X, y)

    # Calculate training accuracy
    preds = clf.predict_proba(X)[:, 1]
    train_acc = np.mean((preds >= 0.5) == y)
    print(f"      Model trained successfully! Train accuracy: {train_acc:.4%}")

    # Save trained model to model.pkl
    with open(model_save_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"      Model saved to: {model_save_path}")
    print(f"Training completed in {time.time() - t0:.2f} seconds.")
    print("=" * 60)

    return clf


def _load_lookup_records_for_ids(
    entity_ids: Set[str],
    data_dirs: Optional[List[str]] = None,
) -> Dict[str, Tuple[str, str]]:
    """
    Stream source files across test and train directories to retrieve
    (business_name, business_address) for the requested entity IDs.
    """
    if data_dirs is None:
        data_dirs = ["dataset/test", "dataset/train"]

    found: Dict[str, Tuple[str, str]] = {}
    remaining = set(entity_ids)

    for d in data_dirs:
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".tsv") or "ground_truth" in fname:
                continue
            fpath = os.path.join(d, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                header = f.readline().rstrip("\r\n").split("\t")
                id_idx = header.index("entity_id") if "entity_id" in header else 0
                name_idx = header.index("business_name") if "business_name" in header else 1
                addr_idx = header.index("business_address") if "business_address" in header else 2
                for line in f:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) > max(id_idx, name_idx, addr_idx):
                        eid = parts[id_idx].strip()
                        if eid in remaining:
                            found[eid] = (parts[name_idx].strip(), parts[addr_idx].strip())
                            remaining.remove(eid)
                            if not remaining:
                                return found
    return found


def score_candidates(
    candidate_pairs_file: str,
    output_file: str = "output_matching_results.tsv",
    threshold: float = 0.80,
    model_path: str = "model.pkl",
) -> Optional[str]:
    """
    Inference function to score candidate pairs and produce final matches.

    Rules:
    - Checks if candidate_pairs_file exists.
    - If it does NOT exist: prints 'Waiting for candidate_pairs.tsv from Tejas.'
    - If it does exist:
        - Scores each candidate pair using the trained LightGBM model.
        - Retains matches where probability >= threshold (default 0.80).
        - Formats the results per source1_entity_id.
        - Saves to output_file (and also output_matching_results.tsv if specified).
    """
    # Check if candidate_pairs_file exists
    if not os.path.exists(candidate_pairs_file):
        print("Waiting for candidate_pairs.tsv from Tejas.")
        return None

    print(f"Scoring candidates from {candidate_pairs_file} with threshold >= {threshold}...")

    # Load model
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found. Training model first...")
        train_model(model_save_path=model_path)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Read candidate pairs file
    # Format: source1_entity_id \t candidate_entity_ids (comma-separated or single)
    s1_order: List[str] = []
    candidates_by_s1: Dict[str, List[str]] = {}
    all_needed_ids: Set[str] = set()

    with open(candidate_pairs_file, "r", encoding="utf-8") as f:
        header_line = f.readline().rstrip("\r\n")
        for line in f:
            line_str = line.rstrip("\r\n")
            if not line_str:
                continue
            parts = line_str.split("\t")
            s1_id = parts[0].strip()
            if s1_id not in candidates_by_s1:
                s1_order.append(s1_id)
                candidates_by_s1[s1_id] = []
            all_needed_ids.add(s1_id)

            if len(parts) > 1 and parts[1].strip():
                cands = [c.strip() for c in parts[1].split(",") if c.strip()]
                candidates_by_s1[s1_id].extend(cands)
                all_needed_ids.update(cands)

    print(f"Loaded {len(s1_order)} Source 1 entities with candidate lists from {candidate_pairs_file}.")

    # Load entity records for needed IDs
    print(f"Loading entity records for {len(all_needed_ids)} unique entities...")
    entity_records = _load_lookup_records_for_ids(all_needed_ids)
    print(f"Retrieved {len(entity_records)} records.")

    # Score candidates
    extractor = PairFeatureExtractor()
    matched_results: Dict[str, List[str]] = {}

    # Flatten pairs for efficient batch inference
    flat_pairs: List[Tuple[str, str]] = []
    flat_features: List[List[float]] = []

    for s1_id in s1_order:
        matched_results[s1_id] = []
        rec1 = entity_records.get(s1_id, ("", ""))
        for cand_id in candidates_by_s1.get(s1_id, []):
            rec2 = entity_records.get(cand_id, ("", ""))
            flat_pairs.append((s1_id, cand_id))
            flat_features.append(extractor.extract_feature_vector(rec1, rec2))

    if flat_features:
        print(f"Scoring {len(flat_features)} candidate pairs with LightGBM...")
        X_score = np.array(flat_features, dtype=np.float32)
        probs = model.predict_proba(X_score)[:, 1]

        # Filter by threshold
        for (s1_id, cand_id), prob in zip(flat_pairs, probs):
            if prob >= threshold:
                matched_results[s1_id].append(cand_id)

    # Ensure parent directory of output_file exists
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Format results per source1_entity_id and write to output_file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_order:
            matches_str = ",".join(matched_results.get(s1_id, []))
            f.write(f"{s1_id}\t{matches_str}\n")

    # If output_file is not output_matching_results.tsv, also save to output_matching_results.tsv
    # to guarantee compatibility with all test scripts
    if output_file != "output_matching_results.tsv":
        with open("output_matching_results.tsv", "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for s1_id in s1_order:
                matches_str = ",".join(matched_results.get(s1_id, []))
                f.write(f"{s1_id}\t{matches_str}\n")

    total_matches = sum(len(m) for m in matched_results.values())
    print(f"Inference complete! Saved {total_matches} matches across {len(s1_order)} entities to {output_file}.")
    return output_file


if __name__ == "__main__":
    # Train the model when run directly
    model = train_model(
        ground_truth_path="dataset/train/train_ground_truth.tsv",
        source_dir="dataset/train",
        model_save_path="model.pkl",
    )

    # Attempt to score candidate pairs if available
    cand_candidates = [
        "candidate_pairs.tsv",
        "output/candidate_pairs.tsv",
    ]
    cand_file = next((f for f in cand_candidates if os.path.exists(f)), "candidate_pairs.tsv")

    out_file = "output/matching_results.tsv" if os.path.isdir("output") else "output_matching_results.tsv"
    score_candidates(cand_file, output_file=out_file, threshold=0.80)
