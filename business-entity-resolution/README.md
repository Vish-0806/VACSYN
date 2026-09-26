# Business Entity Resolution Challenge
**Amazon ML Challenge 2026**

---

## 1. Problem Overview

In large-scale commercial platforms, business entity identity data arrives from multiple independent sources with inconsistent, noisy, or incomplete information. 

The goal of this challenge is to build a scalable Machine Learning pipeline that resolves entity records from **Source 2 (S2)** and **Source 3 (S3)** against **Source 1 (S1)** (the deduplicated reference source). A Source 1 entity may match zero (singleton), one, or multiple entities from Source 2 and Source 3.

The solution is evaluated using **Macro $F_{0.5}$ score** across all Source 1 entities, which weights precision $2\times$ over recall and penalizes incorrect merges.

---

## 2. Pipeline Architecture

```text
Raw Source TSVs (S1, S2, S3)
            │
            ▼
    [ Preprocessing ]       <-- Unicode normalization (Indic/Latin), address cleaning, house # extraction
            │
            ▼
       [ Blocking ]         <-- Country partitioning, multi-signal indexing, sparse TF-IDF
            │
            ▼
   Candidate Pairs (TSV)    <-- High-recall candidate set (Contract 1)
            │
            ▼
 [ Feature Engineering ]    <-- Pairwise string similarities (name, address, house #, source flag)
            │
            ▼
     [ LightGBM Model ]     <-- Gradient-boosted pair classification & probability estimation (Contract 2)
            │
            ▼
      [ Thresholding ]      <-- Macro F0.5 optimization, singleton protection, multi-match filtering
            │
            ▼
Final Submission (TSV)      <-- matching_results.tsv (Contract 3) & candidate_pairs.tsv
```

---

## 3. Repository Structure

```text
business-entity-resolution/
├── dataset/                        # Dataset directory (raw TSV data symlinked or mounted)
│   ├── train/                      # Training sources: train_source1, 2, 3, train_ground_truth
│   └── test/                       # Test sources: test_source1, 2, 3
├── analysis/                       # Data profiling, EDA artifacts, and empirical distribution reports
├── notebooks/                      # Exploratory and prototyping notebooks
│   ├── 01_eda.ipynb                # Exploratory Data Analysis & profiling
│   ├── 02_blocking.ipynb           # Candidate blocking experiments & recall analysis
│   ├── 03_features.ipynb           # Feature engineering & distribution inspection
│   ├── 04_model.ipynb              # Model training, hard negative analysis & threshold tuning
│   └── 05_evaluation.ipynb         # Macro F0.5 evaluation & error diagnostics
├── src/
│   └── business_entity_resolution/ # Core modular Python package
│       ├── __init__.py             # Package initializer
│       ├── pipeline.py             # End-to-end integration and orchestration layer
│       ├── preprocessing/          # Normalization and address parsing
│       │   ├── __init__.py
│       │   ├── normalize.py        # Multilingual business name normalization
│       │   └── address.py          # Address standardization & component extraction
│       ├── blocking/               # Candidate generation & search space reduction
│       │   ├── __init__.py
│       │   ├── exact.py            # Exact & prefix blocking keys
│       │   ├── tfidf.py            # Sparse TF-IDF candidate retrieval
│       │   ├── fuzzy.py            # Fuzzy & token overlap blocking
│       │   └── candidate_generator.py # Primary blocking interface & candidate pair generator
│       ├── features/               # Pairwise feature extraction
│       │   ├── __init__.py
│       │   ├── name_features.py    # Name similarity metrics (Jaccard, n-grams, edit distance)
│       │   ├── address_features.py # Address similarity metrics (tokens, house #, street)
│       │   └── pair_features.py    # Feature matrix compilation for LightGBM
│       ├── model/                  # Classification & decision layer
│       │   ├── __init__.py
│       │   ├── train.py            # LightGBM model training with hard negative handling
│       │   ├── predict.py          # Batch probability scoring
│       │   └── threshold.py        # Macro F0.5 threshold search & entity mapping
│       └── evaluation/             # Metrics and validation
│           ├── __init__.py
│           ├── metrics.py          # Official macro F0.5 and entity-level scorers
│           ├── validation.py       # Stratified cross-validation splitting
│           └── error_analysis.py   # Diagnostics on false merges and missed links
├── output/                         # Competition submission output files
│   ├── matching_results.tsv        # Final predicted matches (scored on leaderboard)
│   └── candidate_pairs.tsv         # Candidate set fed into matching model
├── Documentation_template.md       # Solution methodology document for submission
├── README.md                       # Project documentation and developer guide
└── requirements.txt                # Pinned project dependencies
```

---

## 4. Team Ownership & Isolation

To enable parallel, conflict-free development across 4 team members, responsibilities are strictly partitioned by module boundaries:

| Member | Primary Ownership | Controlled Directories & Files | Contract Responsibility |
| :--- | :--- | :--- | :--- |
| **Member 1** | **Preprocessing + Blocking** | `src/preprocessing/`<br>`src/blocking/`<br>`notebooks/02_blocking.ipynb` | Produces candidate pairs matching **Contract 1** with recall $\ge 97\%$. |
| **Member 2** | **Features + Model** | `src/features/`<br>`src/model/`<br>`notebooks/03_features.ipynb`<br>`notebooks/04_model.ipynb` | Consumes Contract 1; outputs probability scores matching **Contract 2**. |
| **Member 3** | **Evaluation** | `src/evaluation/`<br>`notebooks/05_evaluation.ipynb` | Implements validation splits and macro $F_{0.5}$ metric scoring. |
| **Member 4** | **Integration + Submission** | `src/pipeline.py`<br>`output/`<br>`README.md`<br>`requirements.txt`<br>`Documentation_template.md` | Orchestrates pipeline, runs validator, and formats submission TSVs. |

### Module Modification Rules
- **Member 1** must NOT modify `features/`, `model/`, `evaluation/`, or `pipeline.py`.
- **Member 2** must NOT modify `preprocessing/`, `blocking/`, or `evaluation/`.
- **Member 3** must NOT modify `preprocessing/`, `blocking/`, `features/`, or `model/`.
- **Member 4** manages interfaces, end-to-end execution, and packaging.

---

## 5. Interface Contracts

Each subsystem communicates strictly through typed tabular contracts:

### Contract 1 — Candidate Generation (`candidate_pairs.tsv`)
Produced by `blocking/candidate_generator.py`:
- `s1_entity_id`: String ID of the Source 1 reference entity (e.g., `S1-00001`).
- `candidate_entity_id`: String ID of the candidate match from Source 2 or 3 (e.g., `S2-00047`, `S3-00812`).
- `candidate_source`: String identifier of candidate origin (`S2` or `S3`).

### Contract 2 — Model Prediction
Produced by `model/predict.py`:
- `s1_entity_id`: Source 1 entity ID.
- `candidate_entity_id`: Candidate entity ID.
- `match_probability`: Numeric float in $[0.0, 1.0]$.

### Contract 3 — Final Matching Result (`matching_results.tsv`)
Produced by `model/threshold.py` and written to `output/matching_results.tsv`:
- `source1_entity_id`: Exact Source 1 ID (must include every entity in `test_source1.tsv`).
- `matched_entity_ids`: Comma-separated list of matched S2/S3 IDs, or empty string for singletons.

---

## 6. Git Branching Strategy

```text
main
 ├── member1-blocking      # Feature branch for preprocessing and blocking
 ├── member2-model         # Feature branch for feature engineering and LightGBM
 ├── member3-evaluation    # Feature branch for metrics and cross-validation
 └── member4-integration   # Branch for pipeline orchestration & submissions
```

---

## 7. Critical Constraints & Engineering Guidelines

1. **No External Data:** External APIs, databases, commercial ER services, and web lookups are strictly prohibited.
2. **Memory Efficiency:** The dataset exceeds 24 million records. Streaming, chunked processing (100k - 250k rows), and sparse matrix representations must be used.
3. **Preserve Multilingual Scripts:** Source 2 and Source 3 contain over 450,000 records in native Indic scripts (Hindi, Tamil, etc.). Do not apply destructive ASCII-stripping.
4. **Domain Shift Preparedness:** The test set introduces **France** (14.4% of records) which does not appear in the training data. Features must remain country-agnostic.
5. **Strict Country Partitioning:** True matches never cross country borders. Hard blocking on `country` is required.
6. **Code Standards:** Python 3.10+, complete type hints, docstrings, `pathlib.Path` for file handling, and Python `logging` instead of `print()`.
