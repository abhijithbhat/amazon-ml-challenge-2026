#!/usr/bin/env python3
"""
Official Macro F_0.5 Metric Evaluator for Business Entity Resolution.
Amazon ML Challenge 2026.

Evaluation Metric (Official Rules):
1. Macro F_0.5 Score: Precision-heavy metric calculated per S1 entity,
   then averaged across all S1 entities in the ground truth:
   F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
2. Singletons (Ground truth is empty):
   - Predicting empty  -> Score = 1.0
   - Predicting any ID -> Score = 0.0
3. Non-Singletons (Ground truth is non-empty):
   - Predicting empty  -> Score = 0.0
   - Otherwise compute standard precision, recall, and F_0.5.
4. Macro-average is computed across all Source 1 entities present in the
   ground truth file.

CLI Usage:
    python3 src/evaluate.py <path_to_ground_truth.tsv> <path_to_matching_results.tsv>
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, FrozenSet, Optional, Set, Tuple


EMPTY_FROZENSET: FrozenSet[str] = frozenset()


def compute_entity_metrics(
    gt_ids: Set[str],
    pred_ids: Set[str],
) -> Tuple[float, float, float]:
    """
    Compute (F_0.5, Precision, Recall) for a single Source 1 entity.

    Rules:
    - Singletons (gt_ids is empty):
        * If pred_ids is empty -> Score = 1.0, Precision = 1.0, Recall = 1.0
        * If pred_ids is non-empty -> Score = 0.0, Precision = 0.0, Recall = 0.0
    - Non-Singletons (gt_ids is non-empty):
        * If pred_ids is empty -> Score = 0.0, Precision = 0.0, Recall = 0.0
        * If pred_ids is non-empty:
            tp = |gt_ids & pred_ids|
            precision = tp / |pred_ids|
            recall = tp / |gt_ids|
            F_0.5 = (1.25 * precision * recall) / (0.25 * precision + recall)
            (if denominator is 0, F_0.5 is 0.0)

    Returns:
        (f05, precision, recall)
    """
    if not gt_ids:
        # Singleton entity
        if not pred_ids:
            return 1.0, 1.0, 1.0
        return 0.0, 0.0, 0.0

    # Non-singleton entity
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


def load_predictions_dict(pred_path: str) -> Dict[str, FrozenSet[str]]:
    """
    Load matching predictions TSV into a memory-efficient mapping:
        {source1_id: frozenset(matched_ids)}

    Handles headers, trailing tabs, empty match lists, and whitespace.
    """
    if not os.path.isfile(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    predictions: Dict[str, FrozenSet[str]] = {}
    with open(pred_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_clean = line.rstrip("\r\n")
            if not line_clean:
                continue

            s1, has_tab, rest = line_clean.partition("\t")
            s1 = s1.strip()

            # Skip header if present
            if line_num == 1 and ("entity_id" in s1.lower() or "source1" in s1.lower()):
                continue

            if not s1:
                continue

            rest_clean = rest.strip()
            if not rest_clean:
                predictions[s1] = EMPTY_FROZENSET
            else:
                matches = frozenset(
                    m.strip() for m in rest_clean.split(",") if m.strip()
                )
                predictions[s1] = matches

    return predictions


def evaluate_predictions(
    ground_truth_path: str,
    predictions_path: str,
    verbose: bool = False,
    quiet: bool = False,
) -> Dict[str, float]:
    """
    Evaluate predicted entity matches against ground truth.

    Streams the ground truth file to evaluate all Source 1 entities present
    in the ground truth, computing official Macro F_0.5 score along with
    diagnostic metrics.

    Returns:
        Dictionary of computed evaluation metrics.
    """
    if not os.path.isfile(ground_truth_path):
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")

    t0 = time.time()
    predictions = load_predictions_dict(predictions_path)
    t_load_pred = time.time() - t0

    # Running accumulators across all Source 1 entities in ground truth
    total_gt_entities = 0
    total_f05 = 0.0
    total_precision = 0.0
    total_recall = 0.0

    # Singleton tracking
    singletons_count = 0
    singletons_correct = 0

    # Non-singleton tracking
    non_singletons_count = 0
    non_singletons_f05 = 0.0
    non_singletons_prec = 0.0
    non_singletons_rec = 0.0

    # Micro counts for non-singletons
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Coverage tracking
    preds_found_in_gt = 0
    gt_missing_in_preds = 0

    with open(ground_truth_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_clean = line.rstrip("\r\n")
            if not line_clean:
                continue

            s1, has_tab, rest = line_clean.partition("\t")
            s1 = s1.strip()

            # Skip header if present
            if line_num == 1 and ("entity_id" in s1.lower() or "source1" in s1.lower()):
                continue

            if not s1:
                continue

            rest_clean = rest.strip()
            gt_ids = (
                frozenset(m.strip() for m in rest_clean.split(",") if m.strip())
                if rest_clean
                else EMPTY_FROZENSET
            )

            # Retrieve predictions for s1 (defaults to empty set if unpredicted)
            if s1 in predictions:
                preds_found_in_gt += 1
                pred_ids = predictions[s1]
            else:
                gt_missing_in_preds += 1
                pred_ids = EMPTY_FROZENSET

            f05, prec, rec = compute_entity_metrics(gt_ids, pred_ids)

            total_gt_entities += 1
            total_f05 += f05
            total_precision += prec
            total_recall += rec

            if not gt_ids:
                singletons_count += 1
                if f05 == 1.0:
                    singletons_correct += 1
            else:
                non_singletons_count += 1
                non_singletons_f05 += f05
                non_singletons_prec += prec
                non_singletons_rec += rec

                tp = len(gt_ids & pred_ids)
                fp = len(pred_ids - gt_ids)
                fn = len(gt_ids - pred_ids)
                total_tp += tp
                total_fp += fp
                total_fn += fn

    if total_gt_entities == 0:
        raise ValueError("Ground truth file contains no valid Source 1 entities.")

    # Calculate Macro averages across all Source 1 entities in ground truth
    macro_f05 = total_f05 / total_gt_entities
    macro_precision = total_precision / total_gt_entities
    macro_recall = total_recall / total_gt_entities

    # Singleton metrics
    singleton_accuracy = (
        singletons_correct / singletons_count if singletons_count > 0 else 0.0
    )

    # Non-singleton macro averages
    non_singleton_macro_f05 = (
        non_singletons_f05 / non_singletons_count if non_singletons_count > 0 else 0.0
    )
    non_singleton_macro_prec = (
        non_singletons_prec / non_singletons_count if non_singletons_count > 0 else 0.0
    )
    non_singleton_macro_rec = (
        non_singletons_rec / non_singletons_count if non_singletons_count > 0 else 0.0
    )

    # Micro metrics
    micro_prec = (
        total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    )
    micro_rec = (
        total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    )
    micro_denom = 0.25 * micro_prec + micro_rec
    micro_f05 = (
        (1.25 * micro_prec * micro_rec) / micro_denom if micro_denom > 0 else 0.0
    )

    total_time = time.time() - t0
    extra_predictions = len(predictions) - preds_found_in_gt

    results = {
        "macro_f05": macro_f05,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "total_ground_truth_entities": total_gt_entities,
        "total_predicted_entities": len(predictions),
        "singletons_count": singletons_count,
        "singletons_correct": singletons_correct,
        "singleton_accuracy": singleton_accuracy,
        "non_singletons_count": non_singletons_count,
        "non_singleton_macro_f05": non_singleton_macro_f05,
        "non_singleton_macro_precision": non_singleton_macro_prec,
        "non_singleton_macro_recall": non_singleton_macro_rec,
        "micro_tp": total_tp,
        "micro_fp": total_fp,
        "micro_fn": total_fn,
        "micro_precision": micro_prec,
        "micro_recall": micro_rec,
        "micro_f05": micro_f05,
        "gt_missing_in_predictions": gt_missing_in_preds,
        "extra_predicted_entities": extra_predictions,
        "evaluation_time_sec": total_time,
    }

    if quiet:
        print(f"{macro_f05:.6f}")
        return results

    # Print formatted evaluation report
    print("=" * 72)
    print("  Amazon ML Challenge 2026 — Official Entity Resolution Evaluator")
    print("=" * 72)
    print(f"Ground Truth:  {ground_truth_path}")
    print(f"Predictions:   {predictions_path}")
    print(f"Entities in Ground Truth:  {total_gt_entities:,}")
    print(f"Entities in Predictions:   {len(predictions):,}")

    if gt_missing_in_preds > 0:
        print(
            f"\n  [!] Warning: {gt_missing_in_preds:,} / {total_gt_entities:,} "
            f"entities from ground truth were missing in predictions "
            f"(evaluated as empty predictions)."
        )
    if extra_predictions > 0:
        print(
            f"  [!] Note: {extra_predictions:,} predicted entities were not "
            f"in the ground truth file."
        )

    print("\n--- Singletons (Ground Truth is Empty) ---")
    print(
        f"  Total Singletons:     {singletons_count:,} "
        f"({singletons_count / total_gt_entities * 100:.2f}% of GT)"
    )
    print(
        f"  Correctly Predicted:  {singletons_correct:,} "
        f"({singleton_accuracy * 100:.2f}% accuracy)"
    )
    print(f"  False Merges (FP):    {singletons_count - singletons_correct:,}")

    print("\n--- Non-Singletons (Ground Truth Has Matches) ---")
    print(
        f"  Total Non-Singletons: {non_singletons_count:,} "
        f"({non_singletons_count / total_gt_entities * 100:.2f}% of GT)"
    )
    print(f"  Macro F_0.5:          {non_singleton_macro_f05:.6f}")
    print(f"  Macro Precision:      {non_singleton_macro_prec:.6f}")
    print(f"  Macro Recall:         {non_singleton_macro_rec:.6f}")

    if verbose:
        print("\n--- Micro Linkage Statistics ---")
        print(f"  True Positives (TP):  {total_tp:,}")
        print(f"  False Positives (FP): {total_fp:,}")
        print(f"  False Negatives (FN): {total_fn:,}")
        print(f"  Micro Precision:      {micro_prec:.6f}")
        print(f"  Micro Recall:         {micro_rec:.6f}")
        print(f"  Micro F_0.5:          {micro_f05:.6f}")

    print("=" * 72)
    print(f"  OFFICIAL MACRO F_0.5 SCORE: {macro_f05:.6f} ({macro_f05 * 100:.2f}%)")
    print("=" * 72)
    print(f"Evaluation completed in {total_time:.2f}s (predictions parsed in {t_load_pred:.2f}s)\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Amazon ML Challenge 2026: Official Macro F_0.5 Entity Resolution Evaluator.\n"
            "Calculates the official macro-averaged F_0.5 score across all Source 1 entities."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "ground_truth",
        help="Path to ground truth TSV file (e.g., dataset/train/train_ground_truth.tsv)",
    )
    parser.add_argument(
        "predictions",
        help="Path to matching predictions TSV file (e.g., output/matching_results.tsv)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Display detailed micro-level TP/FP/FN linkage metrics.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only print the raw numeric Macro F_0.5 score (ideal for scripting).",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Output evaluation results as a JSON string.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.ground_truth):
        sys.stderr.write(f"Error: Ground truth file not found: {args.ground_truth}\n")
        sys.exit(1)

    if not os.path.exists(args.predictions):
        sys.stderr.write(f"Error: Predictions file not found: {args.predictions}\n")
        sys.exit(1)

    try:
        results = evaluate_predictions(
            ground_truth_path=args.ground_truth,
            predictions_path=args.predictions,
            verbose=args.verbose,
            quiet=args.quiet or args.json_output,
        )
        if args.json_output:
            print(json.dumps(results, indent=2))
    except Exception as e:
        sys.stderr.write(f"Error during evaluation: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
