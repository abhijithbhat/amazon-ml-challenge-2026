# Amazon ML Challenge 2026: Multi-Source Business Entity Resolution
## Team Solution Reproduction Guide

**Role:** Data Normalization & Reproducibility Lead (Dul)  
**Task:** End-to-End Pipeline Execution, Candidate Generation, and Scoring  

---

### Table of Contents
1. [Overview & Architecture](#1-overview--architecture)
2. [Input and Output Data Locations](#2-input-and-output-data-locations)
3. [Complete Step-by-Step Execution Sequence](#3-complete-step-by-step-execution-sequence)
   - [Step 1: Environment Setup](#step-1--environment-setup)
   - [Step 2: Candidate Generation (Blocking)](#step-2--candidate-generation-blocking)
   - [Step 3: Direct Scoring (Inference)](#step-3--direct-scoring-inference)
   - [Step 4: Output Validation](#step-4--output-validation)
4. [Methodology & Normalization Details](#4-methodology--normalization-details)
5. [Reproducibility Sign-off](#5-reproducibility-sign-off)

---

### 1. Overview & Architecture

This package contains the complete, production-grade Entity Resolution pipeline for the Amazon ML Challenge 2026. The solution links business records across 3 independent sources (`Source 1`, `Source 2`, and `Source 3`) across three countries (`US`, `India`, and `France`).

Key pipeline stages:
1. **Conservative, Information-Preserving Normalization (`src/normalization.py`):**
   - Country-aware address cleaning and state/region code normalization.
   - Intelligent legal suffix standardization (US, Indian, and French legal forms).
   - Indic-script preservation and multilingual legal form mapping (Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati).
   - Landmark extraction and disambiguation (preserving Saint vs. Street).
   - Reversal of synthetic corruption patterns (`... saint` to `street`, OCR typo normalization).
2. **Selective Multi-Pass Candidate Blocking (`src/generate_test_candidates.py`):**
   - Hard country isolation (`France` $\to$ `US` $\to$ `India`).
   - Pure selective PRE3 blocking with multi-branch consensus scoring.
   - Caps candidates to $K \le 12$ high-quality candidates per entity while maintaining $>98\%$ recall.
   - Peak RAM strictly bounded under 1.8 GB.
3. **High-Speed Direct Scorer (`src/score_candidates.py`):**
   - 3-Tier Cascaded Evaluation:
     - Tier 1: Fast C-level RapidFuzz quick-ratio pre-filter.
     - Tier 2: Strict exact-anchor bypass for high-confidence identical entities.
     - Tier 3: 10-dimensional pairwise feature extraction scored with trained Random Forest classifier.
   - 3-Way Digit Conflict Guard (`confirm`, `absent`, `conflict`) to suppress chain-store false merges.
   - Global greedy 1-to-1 disjoint vendor assignment to maximize Macro $F_{0.5}$ and protect singletons.
4. **Official Format Validation (`utils/validate_submission.py`):**
   - Guarantees 100% adherence to submission specifications (tab-separated, exact ordering, valid vendor IDs).

---

### 2. Input and Output Data Locations

All scripts expect the standard directory layout from the repository root:

```
<project_root>/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv       # S1 reference training entities
│   │   ├── train_source2.tsv       # S2 vendor training entities
│   │   ├── train_source3.tsv       # S3 vendor training entities
│   │   └── train_ground_truth.tsv  # S1 to S2/S3 ground truth mappings
│   └── test/
│       ├── test_source1.tsv        # S1 test reference (1,732,544 rows)
│       ├── test_source2.tsv        # S2 test vendor pool (4,887,273 rows)
│       └── test_source3.tsv        # S3 test vendor pool (5,082,316 rows)
├── output/
│   ├── trained_model.pkl           # Pre-trained Random Forest model bundle
│   ├── candidate_pairs.tsv         # Blocking candidates (~12.96M pairs across S1)
│   └── matching_results.tsv        # Final scored matches for leaderboard upload
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # Pipeline source code
│       ├── requirements.txt        # Pinned dependencies
│       └── README.md               # This execution guide
├── utils/
│   └── validate_submission.py      # Official submission validator
└── requirements.txt                # Root-level pinned dependencies
```

---

### 3. Complete Step-by-Step Execution Sequence

#### Step 1 — Environment Setup

Create a clean Python 3.10+ virtual environment and install all pinned dependencies:

```bash
# 1. Create a fresh virtual environment
python3 -m venv .venv

# 2. Activate the virtual environment
# On Linux / macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.venv\Scripts\activate.bat

# 3. Upgrade pip and install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Verified dependencies in `requirements.txt`:
```
joblib==1.6.0
lightgbm==4.7.0
numpy==2.5.3
pandas==3.0.6
psutil==7.2.2
rapidfuzz==3.14.6
scikit-learn==1.9.1
scipy==1.18.1
```

---

#### Step 2 — Candidate Generation (Blocking)

Run the multi-process candidate generation engine to build `output/candidate_pairs.tsv`:

```bash
python3 src/generate_test_candidates.py
```

- **Runtime:** ~15–20 minutes (using 10 parallel worker processes).
- **RAM Footprint:** Strictly $< 1.8$ GB peak.
- **Output:** `output/candidate_pairs.tsv` containing exactly 1,732,544 rows in the original sequence of `test_source1.tsv`.

---

#### Step 3 — Direct Scoring (Inference)

Score the candidate pairs using the trained model, applying 3-way digit anchoring and greedy 1-to-1 assignment:

```bash
python3 src/score_candidates.py
```

*Command-Line Options (all default to optimal challenge settings):*
```bash
python3 src/score_candidates.py \
    --candidate-file output/candidate_pairs.tsv \
    --matching-file output/matching_results.tsv \
    --test-dir dataset/test \
    --threshold 0.83 \
    --fallback-threshold 0.90 \
    --batch-size 25000
```

- **Runtime:** ~35–45 minutes for full 12.96M candidate evaluation.
- **Output:** `output/matching_results.tsv` (tab-separated, 1,732,544 rows).

---

#### Step 4 — Output Validation

Validate the resulting output files against the official competition requirements:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Expected result:
```
ML Challenge 2026 — submission validator
  test dir: dataset/test
  required S1 entities: 1732544
  matching_results.tsv: 1732544 rows
  candidate_pairs.tsv: 1732544 rows

PASS — no blocking issues found. Safe to submit.
```

---

### 4. Methodology & Normalization Details

#### High-Precision Normalization Rules
1. **Conservative Name Normalization:**
   - Legal suffix standardization (e.g. `Pvt Ltd`, `Private Limited`, `LLC`, `Inc`, `Corp`, `SARL`, `SAS`).
   - Multilingual Indic legal forms: maps Hindi, Tamil (`எல்எல்பி`), Telugu (`ఎల్‌ఎల్‌పీ`), Kannada, Bengali, and Gujarati legal suffixes.
   - Synthetic noise removal: extracts trade names from `DBA:`, `formerly known as`, `fka`, strips URL domain suffixes (`.com`, `.org`, `.net`, `.in`), and removes commercial honorifics (`M/s`, `Shri`, `Dr`).
2. **Address & Landmark Standardization:**
   - Street-type abbreviation expansion (`St` $\to$ `street`, `Rd` $\to$ `road`, `Ave` $\to$ `avenue`).
   - Saint vs. Street disambiguation: preserves `Saint Louis`, `St. Albans`, `St. Francis`, while expanding corrupted end-of-street `... saint` to `street`.
   - Connecticut (`CT`) state preservation: ensures `ct` is only expanded to `court` when preceded by a street name token, preventing state code destruction.
   - Landmark preservation and preposition standardization (`opp.` $\to$ `opposite`, `nr.` $\to$ `near`, `behind`).
   - Standardized unit prefixes (`door no`, `plot no`, `h no`, `flat no`).
3. **Macro $F_{0.5}$ Alignment:**
   - Singletons (S1 entities without true matches) receive empty strings in `matched_entity_ids` (earning 1.0 per singleton).
   - Address digit conflict suppression prevents false merges of chain stores sharing identical names at different physical addresses.

---

### 5. Reproducibility Sign-off

- [x] **Pinned Dependencies:** All packages pinned in `requirements.txt` with zero extraneous dependencies.
- [x] **Clean Paths:** Zero hard-coded absolute paths; all paths resolve dynamically relative to project root.
- [x] **Module Safety:** Explicit `sys.path` and fallback import blocks prevent any `ModuleNotFoundError`.
- [x] **Full Pipeline Tested:** Candidate generation, scoring, and submission validation complete without error.
- [x] **Format Compliance:** Validated with official `utils/validate_submission.py` (Exit Code 0, PASS).
