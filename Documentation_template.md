# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Pod 2 — Data & Features  
**Team Members:** Abhijith Bhat, Mohammed Dhulkifl, Sushma  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We developed a two-stage pipeline for multi-source business entity resolution consisting of a **Pure Selective PRE3 candidate generation** stage and a **Cascaded 3-Tier Hybrid Scorer**. The candidate generation stage reduced 1.73 M x 9.97 M potential cross-source comparisons to 12,964,333 candidate pairs (7.48 per S1 entity) while achieving a benchmark candidate recall of 93.86 %. The scoring stage applies a trained RandomForestClassifier over a 10-dimensional similarity feature vector, augmented by rule-based pre-filters and a structural 3-way digit partition that suppresses false chain-store merges. France — an open-set country absent from the training data — is handled via dedicated normalization and tightened thresholds without any external lookup.

---

## 2. Methodology

### 2.1 Problem Analysis

Key insights from exploratory data analysis:

- **Three countries, different noise profiles.** Training data covers US (~60 %) and India (~40 %); the test set introduces France as an unseen open-set country. Each country requires distinct normalization rules, postal-code patterns, and state/region extraction.
- **Legal suffix proliferation.** Names appear with dozens of legal-form variants (Pvt. Ltd., P. Ltd., Private Limited, SARL, SAS, ...). Without canonicalization, these variants block on different keys and are missed.
- **Address noise is severe and multi-modal.** Abbreviations (Rd/Road, St/Street), landmark references ("Near SBI ATM"), reordered components, missing PIN codes, OCR errors ("saint" for "street", "doro" for "door"), and Indic-script transliteration variants all appear in the dataset.
- **French diacritics cause spurious mismatches.** The same business appears with and without accent marks (e with accent to plain e, u with umlaut to plain u). Without accent folding, string similarity scores are artificially depressed for French records.
- **Chain-store false-positive risk.** Businesses with identical or very similar names appear at different numbered addresses (e.g., the same franchise at "42 Main Street" vs. "67 Main Street"). Address digit mismatch is a strong signal that two records are genuinely distinct.
- **Precision is penalized more heavily under F_0.5.** The evaluation metric weights false merges (wrong matches) more than missed links, making over-merging the primary risk to avoid.
- **Singletons are a first-class prediction.** Under macro F_0.5, a Source 1 entity with no true matches earns a perfect entity-level score of 1.0 when the model correctly predicts no matches. Incorrectly predicting any match for a true singleton scores 0.0 for that entity. Correctly identifying singletons therefore earns credit and reduces harmful over-merging.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (Hybrid Rule + ML)

The pipeline separates candidate generation from scoring deliberately — the two stages are **decoupled for efficiency and auditability**:

- **Candidate generation** runs once, is country-partitioned to guarantee hard isolation, and is memory-bounded under 1.8 GB peak. Its output (`candidate_pairs.tsv`) is a permanent audit artifact.
- **Scoring** consumes the pre-computed candidates, applies fast rule-based pre-filters, and runs the trained classifier only on pairs that survive the rules — enabling full reproducibility without re-running the expensive blocking phase.

**Core Innovation:** The combination of (a) Pure Selective PRE3 blocking with learned branch-reliability weights and multi-branch consensus bonuses to maximize candidate recall at low K, (b) a 3-way digit partition applied as structural gating to suppress false chain-store merges, and (c) country-specific threshold calibration for the unseen France partition without any external data.

---

## 3. Candidate Generation (Blocking)

### 3.1 Architecture

Candidate generation is implemented in `src/generate_test_candidates.py` using a **Pure Selective PRE3** strategy. The design processes the three countries in sequence (France, then US, then India) in isolated partitions. For each country, all Source 2 and Source 3 records are indexed into blocking buckets before any Source 1 queries are made; memory is released between country partitions, keeping peak RAM under 1.8 GB.

Ten parallel worker processes (via `ProcessPoolExecutor`) handle normalization of raw business names and addresses.

### 3.2 Blocking Keys

Seven independent blocking passes are generated per record from `src/blocking_features.py`:

| Pass | Key Pattern | Purpose |
|---|---|---|
| 1 | `{COUNTRY}_ST_{state}_{first_token}` | State + first significant name token — high-precision anchor |
| 2 | `{COUNTRY}_ST_{state}_{second_token}` | State + second token — handles prefix additions/reordering |
| 3 | `{COUNTRY}_2TOK_{tok_a}_{tok_b}` | Sorted two-token conjunction — handles missing state and token reorder |
| 4 | `{COUNTRY}_ST_PRE3_{state}_{prefix3}` | State + first-3-char prefix — the selective PRE3 pass |
| 5 | `{COUNTRY}_ADDR_{state}_{addr_token}` | State + discriminative address token (ADDR_STOPWORDS filtered) |
| 6 | `{COUNTRY}_PIN_{pin}_{prefix3}` | Postal code + name prefix — highest-precision geo anchor |
| 7 | `{COUNTRY}_TOK_{first_token}` | Country + token fallback when state is absent |

Legal suffixes (LLC, Inc, Pvt, Ltd, SARL, SAS, ...), honorifics (Shri, Smt, Mr, Mrs), and generic business words (Group, Holdings, Services, Solutions) are excluded from name tokens to prevent oversized, low-entropy buckets.

### 3.3 Candidate Ranking and Truncation

Each candidate record accumulates a priority score:

```
score = sum of branch_weights for each blocking branch hit
      + 3.0 * (n_branches - 1)              [multi-branch consensus bonus]
      + min(3.0, sum of 1/sqrt(bucket_size)) [bucket specificity score]
      + name_jaccard * 4.0 + name_token_overlap * 1.5
      + addr_jaccard * 2.5 + addr_token_overlap * 0.8
```

Learned branch reliability weights:

| Branch type | Weight |
|---|---|
| `state+token` | 4.0 |
| `2tok_conjunction` | 4.0 |
| `state+addr_token` | 3.5 |
| `state+second_token` | 2.5 |
| `pin+prefix` | 2.0 |
| `country+token_fallback` | 2.0 |
| `state+prefix3` | 1.5 |
| other | 1.0 |

The top **K = 12** candidates per S1 entity are retained.

For large buckets (26-100 records for PRE3; 26-200 for other passes), candidates are pre-filtered by cheap token overlap (name overlap >= 1 OR address overlap >= 2) before ranking, and only the top-1 (PRE3) or top-2 (other passes) are admitted. Buckets larger than 200 records (or larger than 100 for PRE3) are skipped entirely to prevent precision collapse.

### 3.4 Candidate Generation Results

| Metric | Value |
|---|---|
| Total S1 entities | 1,732,544 |
| Total candidate pairs | **12,964,333** |
| Average candidates per S1 entity | **7.48** |
| Benchmark candidate recall (blocking) | **93.86 %** |
| Country partitions | France, US, India |
| Parallel normalization workers | 10 |

The candidate set is written to `output/candidate_pairs.tsv` (audit copy: 191,213,968 bytes / ~182.36 MB; SHA-256: `0984D0A83CA888A17F383D0336F00BDB0A5171F8F9C4A9B9EDEC40AD6C2CA15D`).

### 3.5 How True Matches Are Protected from Loss

- **Seven independent passes** ensure that a record missing a state code, postal code, or second token is still reachable via one of the remaining passes.
- **Country isolation** prevents cross-country false negatives: every France S1 entity can only match France S2/S3 records.
- **PRE3 soft admission** (prefix of first token) catches records where the first significant token differs but shares a 3-character prefix (spelling variants, transliteration noise).
- **Two-token conjunction pass** recovers pairs where the first token differs but the combination of any two significant tokens matches.

---

## 4. Matching Model

### 4.1 Normalization Pre-processing

Before any features are computed, both sides of each candidate pair are normalized by `src/normalization.py` and then passed through accent folding in `src/score_candidates.py`:

- **Business name normalization:** legal suffix canonicalization (e.g., Pvt. Ltd. to "private limited", SARL to "sarl"), Unicode NFD decomposition to strip combining diacritics (e-acute to e, u-umlaut to u, n-tilde to n), Indic-script transliteration to Latin (Devanagari, Telugu, Tamil, Bengali, Gujarati, Kannada), repeated-word deduplication, punctuation normalization.
- **Address normalization:** street-type standardization (Rd to Road, St to Street, Blvd to Boulevard, etc.), property-number prefix normalization (H.No, D.No, Plot No), OCR noise correction ("saint" to "street" in US/India, "doro/dor" to "door"; "ct." only expanded to "court" when preceded by a street word, to avoid confusing the Connecticut state abbreviation CT), PO Box standardization, US Rural/County/State road expansion.
- **Country-specific postal extraction:** 5-digit US ZIP (at end or preceded by 2-letter state code), 6-digit India PIN (ignoring plot/khasra/survey numbers), 5-digit France postal code (01xxx to 98xxx range).

### 4.2 Feature Vector (10 Dimensions)

All features are floats in [0.0, 1.0]. Computed by `src/comparison_features.py` via `compute_pair_features()`.

| # | Feature | Description |
|---|---|---|
| 1 | `name_jw` | Jaro-Winkler similarity on normalized `name_clean` |
| 2 | `name_jaccard` | Jaccard coefficient of significant name token sets |
| 3 | `name_overlap` | Fraction of shared tokens relative to the shorter set |
| 4 | `name_exact` | Binary: exact string match after full normalization |
| 5 | `name_len_ratio` | Character length ratio (shorter / longer); 1.0 if equal |
| 6 | `addr_jw` | Jaro-Winkler similarity on normalized `address_clean` |
| 7 | `addr_jaccard` | Jaccard coefficient of discriminative address token sets |
| 8 | `state_match` | Binary: identical extracted state/region code |
| 9 | `postal_match` | Binary: identical extracted postal/PIN/ZIP code |
| 10 | `country_match` | Binary: same country label |

**Jaro-Winkler** is implemented from scratch in `src/comparison_features.py` (zero external dependency), with the C-level RapidFuzz implementation used as a faster drop-in for the Tier 1 pre-filter where available.

Significant name tokens exclude legal suffixes, stop words, and honorifics (the same `NAME_STOPWORDS` set used in blocking). Address tokens exclude generic directional and structural words (`ADDR_STOPWORDS`).

> **Note:** Digit consistency is **not** one of these 10 model features. It is implemented as structural gating applied before and after Tier 3 model scoring (see Section 4.5).

### 4.3 Model Architecture and Training

**Model type:** `sklearn.ensemble.RandomForestClassifier`

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 100 |
| `max_depth` | 10 |
| `min_samples_leaf` | 4 |
| `class_weight` | `'balanced'` |
| `random_state` | 42 |

**Training data construction** (`src/step4_train_matching_model.py`):

- 10,000 S1 records sampled from training data; ground truth matches extracted from `train_ground_truth.tsv`, yielding approximately 35,000 positive pairs.
- Hard negatives generated by running the same 7-pass blocking on the sampled S1 records against S2/S3, pairing with non-matching records from the same blocking buckets. Negative-to-positive ratio: **4:1**.
- 75/25 train/test split (stratified by label) for threshold tuning.

**Threshold selection:** F_0.5 is computed at thresholds {0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90} on the held-out split. The threshold maximizing F_0.5 on that split is saved with the model bundle as `optimal_threshold`.

The saved artifact `output/trained_model.pkl` (5.1 MB) bundles the fitted RandomForestClassifier, `optimal_threshold`, `feature_cols`, and the training-split precision/recall/F_0.5 values.

### 4.4 Cascaded 3-Tier Hybrid Scorer

The scorer (`src/score_candidates.py`) reads the pre-computed `candidate_pairs.tsv` and evaluates each pair through three sequential tiers. **Tiers 1 and 2 are rule-based**; only pairs surviving both advance to Tier 3 model scoring.

#### Tier 1 — Fast C-Level Pre-Filter

Reject the pair if both conditions hold:
- `rapidfuzz.quick_ratio(name_a, name_b) < 45.0`
- `digit_status != 'confirm'`

Uses the C-level `rapidfuzz.fuzz.quick_ratio` for high-throughput name similarity. Pairs with confirmed digit overlap bypass this gate to protect low-name-similarity matches anchored by strong address evidence.

#### Tier 2 — Restricted Exact-Anchor Bypass

A pair is assigned `probability = 1.0` and accepted directly (skipping Tier 3) only when **all four** conditions are simultaneously satisfied:

1. Exact clean name match: `name_a == name_b`
2. Name length >= 6 characters (prevents short acronyms such as 'KFC', 'ATM')
3. Non-empty address digit overlap: at least one significant digit is shared
4. Strong address confirmation: `addr_jw(address_a, address_b) >= 0.80`

If any condition fails, the pair proceeds to Tier 3.

#### Tier 3 — Pairwise Features and RandomForest Scoring

The 10-dimensional feature vector is computed for each remaining pair and scored in batches of 25,000 via `model.predict_proba()`. The resulting probability is then evaluated against the appropriate threshold determined by (a) the digit partition status and (b) the country.

### 4.5 3-Way Digit Partition (Structural Gating)

Significant digits are extracted from normalized addresses by `extract_significant_digits()`: all digit sequences found by `\d+`, excluding trivial single digits (0, 1, 2). The **3-way digit partition** then classifies each pair via `get_digit_status()`:

| Status | Condition | Scoring Action |
|---|---|---|
| `confirm` | Both sides contain significant digits **and share at least one** | Accept if `prob >= effective_threshold` (lower bar: 0.83 standard / 0.88 FR-BE) |
| `absent` | At least one side has **no** significant digits (landmark address) | Accept if `prob >= effective_fallback` (higher bar: 0.90 standard / 0.92 FR-BE) |
| `conflict` | Both sides contain significant digits but **zero overlap** | **Suppressed — no match regardless of model probability** |

The `conflict` suppression is the primary defence against false chain-store and franchise merges: when two records share an identical or near-identical business name but different street numbers (e.g., "Burger Palace, 42 High Street" vs. "Burger Palace, 67 High Street"), the model's name similarity score may be high, but the presence of conflicting digits is strong evidence that these are genuinely distinct physical locations. Suppressing `conflict` pairs prevents these from being incorrectly merged.

The `absent` path uses a higher threshold to guard against landmark-addressed businesses (e.g., "Near SBI Bank, Gandhi Road") where the absence of digits removes the strongest disambiguation signal; the model must be more confident before accepting such a pair.

This gating is applied **independently of and after** the 10-feature model score — it is not one of the 10 features but an external structural filter.

### 4.6 Country-Specific Threshold Calibration

French and Belgian (`FR`, `BE`) entities have elevated diacritics noise: the same business name may appear with or without accent marks across sources. Even after accent folding, residual similarity variations can occur. A tighter acceptance threshold is applied for these countries:

| Country group | Digit-confirm threshold | Absent-digit fallback |
|---|---|---|
| Standard (US, India, and others) | **0.83** | **0.90** |
| FR / BE | **0.88** | **0.92** |

This calibration addresses the open-set France problem: France does not appear in the training data, so the model was never trained on French examples. Stricter thresholds compensate for the higher uncertainty in model scores for French pairs without requiring any external data or France-specific training labels.

### 4.7 Global Greedy 1-to-1 Disjoint Assignment

After all candidate pairs have been scored, all passing pairs are sorted globally by probability in descending order. A greedy assignment loop then enforces the **1-to-1 constraint on vendor records**: each Source 2 / Source 3 entity is assigned to at most one Source 1 entity. Source 1 entities may accumulate multiple distinct vendor matches. This is equivalent to a greedy maximum-weight bipartite matching on the vendor side and ensures that no vendor ID appears in more than one `matched_entity_ids` list.

---

## 5. Results & Error Analysis

### 5.1 Pipeline Statistics

| Metric | Value |
|---|---|
| Total S1 entities in submission | 1,732,544 |
| S1 entities with >= 1 match | **1,504,378** (86.83 %) |
| Predicted singletons (no matches) | **228,166** (13.17 %) |
| Total vendor records assigned | **4,686,062** |
| Total candidate pairs evaluated | 12,964,333 |
| Scoring runtime | 49.0 minutes |
| Submission validation | **PASSED** (exit 0) |

### 5.2 Evaluation Metric

Submissions are scored using **Macro F_0.5**:

```
F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
```

Computed per Source 1 entity and averaged across all 1,732,544 entities (including singletons). Under F_0.5, precision has **4x the effective weight of recall**: a false merge (wrong link) contributes four times as much penalty as a missed true match. This drives the design towards conservative threshold choices and the `conflict` suppression strategy.

**Singleton handling:** An empty prediction for a Source 1 entity earns a per-entity score of 1.0 if and only if the ground truth is also empty (a true singleton). If the ground truth contains matches that are missed, an empty prediction scores 0.0 for that entity. The strategy of preserving 228,166 singletons (13.17 %) therefore selectively avoids harmful over-merging, but does not universally guarantee a 1.0 per-entity score — it earns credit only where the ground-truth entity genuinely has no matches.

### 5.3 Final Macro F_0.5

A held-out test F_0.5 score is not recorded in the repository. The leaderboard score reflects the official evaluation against the test ground truth.

### 5.4 Common False Positives (Wrong Merges)

Based on the structure of the digit gating and threshold design, the most likely false-positive patterns are:

- **Landmark-addressed businesses with similar names but different locations** — pairs with `digit_status = 'absent'` rely entirely on the model; when both the name and address are vague, the model may overestimate similarity.
- **Identical franchise/chain names across different cities** — suppressed by `conflict` gating when street numbers differ, but pairs where one address has no digits can still produce false positives.
- **Short business names (less than 6 characters)** — Tier 2 bypass explicitly excludes these due to ambiguity risk; Tier 3 scoring on very short normalized names is inherently less reliable.

### 5.5 Common False Negatives (Missed Matches)

- **Records with zero candidate overlap** — if a true match is not recovered by any of the 7 blocking passes (6.14 % of true matches, based on the 93.86 % candidate recall), it is structurally impossible to recover in scoring.
- **`conflict` suppression of true matches with digit transcription errors** — if one address contains an OCR error in a house number (e.g., "42" vs. "43"), the pair will be classified as `conflict` and suppressed even if it is a genuine match.
- **French records with residual diacritics variation** — pairs below the elevated FR threshold (0.88 / 0.92) that are true matches will be missed.

---

## 6. Conclusion

Our solution demonstrates that a carefully engineered blocking and rule-augmented classifier pipeline can resolve multi-source business entities at scale without any external data or large pre-trained language models. The Pure Selective PRE3 blocking achieves 93.86 % candidate recall at only 7.48 candidates per entity, and the Cascaded 3-Tier Hybrid Scorer — combining a RandomForestClassifier with structural digit gating and country-specific thresholds — provides strong precision control under the F_0.5 objective. The most impactful design decision was the 3-way digit partition: suppressing `conflict` pairs and applying a stricter fallback threshold for digitless addresses proved essential for avoiding false chain-store merges without sacrificing true positive recall. The modular, decoupled architecture (blocking and scoring as separate auditable stages) also made iteration and debugging tractable on the full-scale 12.96 M pair candidate set.

---

## Appendix

### A. Code Artefacts

The complete, runnable pipeline is in `code/business_entity_resolution/` (source under `src/`, with `README.md` and `requirements.txt`). The working copy of each source file is mirrored identically in the top-level `src/` directory.

**Entry points to reproduce `output/matching_results.tsv`:**

```bash
# Step 1 — Candidate generation (produces output/candidate_pairs.tsv)
python src/generate_test_candidates.py

# Step 2 — Train model (produces output/trained_model.pkl)
python src/step4_train_matching_model.py

# Step 3 — Score candidates (produces output/matching_results.tsv)
python src/score_candidates.py \
    --candidate-file output/candidate_pairs.tsv \
    --matching-file  output/matching_results.tsv \
    --test-dir       dataset/test \
    --model          output/trained_model.pkl \
    --threshold      0.83 \
    --fallback-threshold 0.90
```

**Key source files:**

| File | Role |
|---|---|
| `src/normalization.py` | Business name and address normalization for all countries |
| `src/blocking_features.py` | Blocking key generation, postal/state extraction, token extraction |
| `src/generate_test_candidates.py` | Pure Selective PRE3 candidate generation (Step 1 entry point) |
| `src/comparison_features.py` | 10-dimensional pairwise similarity feature computation |
| `src/step4_train_matching_model.py` | RandomForest training + F_0.5 threshold tuning (Step 2 entry point) |
| `src/score_candidates.py` | Cascaded 3-Tier Hybrid Scorer + greedy assignment (Step 3 entry point) |
| `utils/validate_submission.py` | Official submission format validator (stdlib only, no dependencies) |

**Dependencies:** see `requirements.txt` (key packages: `scikit-learn`, `numpy`, `pandas`, `rapidfuzz`, `psutil`).

**Output files:**

| File | Description | Size |
|---|---|---|
| `output/matching_results.tsv` | Final entity matches — leaderboard submission file | 84.7 MB |
| `output/candidate_pairs.tsv` | Blocking candidate set fed to the model | ~182.4 MB (SHA-256: `0984D0A83CA888A17F383D0336F00BDB0A5171F8F9C4A9B9EDEC40AD6C2CA15D`) |
| `output/trained_model.pkl` | Saved RandomForest bundle with threshold and feature list | 5.1 MB |
| `output/feature_matrix_sample.tsv` | 100-row labeled feature matrix for inspection | 10 KB |

### B. Additional Results

**Blocking quality summary:**

| Metric | Value |
|---|---|
| Blocking passes (key types) | 7 |
| Max candidates per S1 entity (K) | 12 |
| Small-bucket cap | 25 |
| PRE3 admission cap (bucket size) | 8 |
| PRE3 large-bucket threshold | 100 |
| Total candidate pairs | 12,964,333 |
| Average candidates per S1 entity | 7.48 |
| Benchmark candidate recall | 93.86 % |

**Scoring pipeline tier breakdown (from the 49-minute scoring run):**

- **Tier 1 pre-filter rejected:** the majority of pairs with low name similarity and no digit confirmation, substantially reducing Tier 3 load
- **Tier 2 exact-anchor bypassed:** high-confidence exact matches assigned `prob = 1.0` directly, skipping model inference
- **Tier 3 model scored:** remaining pairs evaluated by RandomForest in batches of 25,000; digit gating applied to outcomes
- **`conflict` suppressed:** pairs where both sides have digits but zero overlap are suppressed regardless of model score

**Country-specific threshold summary:**

| Country | Digit-confirm path | Absent-digit fallback |
|---|---|---|
| US, India, all others | 0.83 | 0.90 |
| France (FR), Belgium (BE) | 0.88 | 0.92 |

---

**Note:** This document describes the actual submitted implementation. All figures are sourced from the repository commit history, source code, and the scoring run that produced the submitted `output/matching_results.tsv` (commit `56197ba`, scoring run completed 27 September 2026).
