# Phase 2.6: Real-Data Smoke Test Report

## Overview

**Date**: Executed on current branch `features-ml`  
**Objective**: Small real-data smoke test using actual training dataset  
**Status**: ✅ **PASSED**

---

## Dataset

**Path**: `C:\Users\Vishal S Naik\MyProjects\VACSYN\student_resource\dataset\train\extracted\student_resource\dataset\train`

| File | Size | Rows |
|------|------|------|
| `train_source1.tsv` | 210 MB | 2,206,821 |
| `train_source2.tsv` | 466 MB | 5,034,616 |
| `train_source3.tsv` | 480 MB | 5,285,603 |
| `train_ground_truth.tsv` | 121 MB | 2,206,821 (S1 entities with matches) |

---

## Smoke Test Configuration

| Parameter | Value |
|-----------|-------|
| S1 entities sampled | 500 (from 2,083,574 with ground truth) |
| Candidate budget per S1 | 40 |
| Validation ratio | 0.2 (20%) |
| Random seed | 42 |
| LightGBM estimators | 100 |
| Early stopping rounds | 50 |

---

## Execution Results

### 1. Data Loading
- Ground truth loaded: **2,083,574** S1 entities with matches
- S1 entities sampled: **500** (random, seed=42)
- S1 data loaded: **500 rows**
- S2 entities needed: **894** (+ 500 distractors = 1,394 loaded)
- S3 entities needed: **965** (+ 500 distractors = 1,465 loaded)

### 2. Candidate Generation
- **Candidate pairs generated**: **15,939**
- M1 `CandidateGenerator` used without modification
- Country-isolated blocking confirmed

### 3. Label Assignment
| Metric | Count | Percentage |
|--------|-------|------------|
| Total candidates | 15,939 | 100% |
| Positive (label=1) | 1,854 | 11.63% |
| Negative (label=0) | 14,085 | 88.37% |

- Labels assigned using `assign_labels_to_candidates()` 
- Only existing candidate pairs labeled
- No arbitrary negatives generated

### 3. Entity-Level Train/Validation Split
| Split | S1 Entities | Candidate Rows |
|-------|-------------|----------------|
| Train | 400 | 12,738 |
| Validation | 100 | 3,201 |
| **Total** | **500** | **15,939** |

- ✅ No S1 entity appears in both splits (verified disjoint)
- ✅ All candidate rows for an S1 stay together
- ✅ Deterministic with `random_state=42`

### 4. Feature Assembly
- Feature dataframe shape: **(15,939, 21)**
- Identity columns (3): `s1_entity_id`, `candidate_entity_id`, `candidate_source`
- **Model features (18)**: ✅ **EXACTLY 18 FEATURES CONFIRMED**

| Feature | Type |
|---------|------|
| `name_token_jaccard` | Continuous [0,1] |
| `name_token_overlap_count` | Count |
| `name_char_ngram_cosine` | Continuous [0,1] |
| `name_edit_similarity` | Continuous [0,1] |
| `name_length_ratio` | Continuous [0,1] |
| `name_exact_match` | Binary {0,1} |
| `name_first_token_match` | Binary {0,1} |
| `address_token_jaccard` | Continuous [0,1] |
| `address_token_overlap_count` | Count |
| `address_house_number_match` | Binary {0,1} |
| `address_edit_similarity` | Continuous [0,1] |
| `address_length_ratio` | Continuous [0,1] |
| `address_missing` | Binary {0,1} |
| `country_match` | Binary {0,1} |
| `source_is_s2` | Binary {0,1} |
| `source_is_s3` | Binary {0,1} |
| `name_length_ratio_context` | Continuous [0,1] |
| `address_length_ratio_context` | Continuous [0,1] |

### 5. LightGBM Training
| Parameter | Value |
|-----------|-------|
| Training samples | 12,738 |
| Validation samples | 3,201 |
| Features | 18 |
| Objective | Binary classification |
| Metric | Binary logloss |
| Estimators | 100 |
| Early stopping | 50 rounds |
| Learning rate | 0.05 |
| Random state | 42 |

**Result**: ✅ **TRAINING SUCCESSFUL**

| Metric | Value |
|--------|-------|
| Features used | 18 |
| Best iteration | 100 |
| Validation binary_logloss | 0.00761563 |
| Early stopping | Not triggered |
| Model artifact | Saved & loaded successfully |
| Model features | 18 |

---

## Verification Checklist

| Check | Status |
|-------|--------|
| 500 S1 entities sampled from ground truth | ✅ |
| M1 CandidateGenerator used without modification | ✅ |
| Labels assigned using `assign_labels_to_candidates()` | ✅ |
| No Cartesian products / arbitrary negatives | ✅ |
| Entity-level split with `random_state=42` | ✅ |
| No S1 leakage (train ∩ val = ∅) | ✅ |
| Phase 1 feature assembly used (`build_pair_features`) | ✅ |
| Feature DF: 3 identity + 18 model columns | ✅ |
| Exactly 18 model features confirmed | ✅ |
| LightGBM training succeeded | ✅ |
| Model saved & loaded successfully | ✅ |
| No memory/runtime errors | ✅ |

---

## Summary

| Metric | Value |
|--------|-------|
| **SMOKE TEST STATUS** | **PASSED** |
| **REAL MODEL TRAINED** | **YES** (on 500-S1 sample) |
| **PRODUCTION MODEL** | **NO** (this is a smoke test only) |

---

## Notes

- This is a **smoke test only** — not a full production model
- Full production training would require the complete dataset (~2.2M S1 entities)
- No production model artifact was created/overwritten
- No changes to `predict.py`, `threshold.py`, `pipeline.py`, or M1/M3/M4 code
- No hyperparameter tuning performed
- No new dependencies added

---

## Next Steps (Not in Scope)

1. Full dataset training (2.2M S1 entities)
2. Hyperparameter tuning
2. Prediction implementation (`predict.py`)
3. Threshold optimization (`threshold.py`)
4. End-to-end pipeline integration
5. Official submission validation