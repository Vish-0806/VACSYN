# Phase 2: Training-Label Preparation + LightGBM Training - Complete

## Summary

All Phase 2 M2 training tasks completed and validated.

| Component | Status | Tests |
|-----------|--------|-------|
| **train.py** | ✅ Complete | 33 new tests |
| **Label preparation** | ✅ Complete | 11 tests |
| **Entity-level split** | ✅ Complete | 6 tests |
| **Feature contract** | ✅ Complete | 5 tests |
| **LightGBM training** | ✅ Complete | 11 tests |
| **Integration test** | ✅ Complete | Full pipeline verified |

---

## Files Created/Modified

### Source Code
| File | Status | Lines |
|------|--------|-------|
| `src/business_entity_resolution/model/train.py` | ✅ Implemented | +197 lines |

### Tests
| File | Tests | Status |
|------|-------|--------|
| `tests/test_training.py` | 33 | ✅ All pass |

### Scripts
| File | Purpose |
|------|---------|
| `scripts/verify_pipeline.py` | End-to-end integration test |

---

## New Functions in `train.py`

### 1. `prepare_ground_truth_mapping(path: str) -> Dict[str, List[str]]`
Loads ground truth TSV into dict mapping S1 ID → list of matched S2/S3 IDs.

### 2. `assign_labels_to_candidates(pairs_df, gt_mapping) -> pd.DataFrame`
Assigns binary labels (0/1) to **existing candidate pairs only**:
- `label = 1` if `candidate_entity_id ∈ ground_truth[s1_entity_id]`
- `label = 0` otherwise
- **Never generates new pairs** - only labels existing M1 candidate pairs

### 3. `entity_level_split(pairs_df, val_ratio=0.2, random_state=42) -> (train_df, val_df)`
Entity-level train/validation split with **no S1 leakage**:
- All candidate pairs for a given S1 go entirely to train OR validation
- `train_s1_ids ∩ val_s1_ids = ∅`
- Deterministic with `random_state=42`

### 4. `train_lightgbm_model(...) -> lgb.Booster`
Full LightGBM binary classification training:
- Validates exactly 18 model features
- Excludes identity columns (`s1_entity_id`, `candidate_entity_id`, `candidate_source`)
- Validates binary labels (0/1)
- Optional validation data with early stopping
- Saves model to `models/lightgbm_model.txt` if path provided

### 5. `extract_model_features(df) -> pd.DataFrame`
Extracts exactly 18 model features, drops identity columns and label.

---

## Exact 18 Model Features (Validated)

| Category | Features |
|----------|----------|
| **Name (7)** | `name_token_jaccard`, `name_token_overlap_count`, `name_char_ngram_cosine`, `name_edit_similarity`, `name_length_ratio`, `name_exact_match`, `name_first_token_match` |
| **Address (6)** | `address_token_jaccard`, `address_token_overlap_count`, `address_house_number_match`, `address_edit_similarity`, `address_length_ratio`, `address_missing` |
| **Pair/Context (5)** | `country_match`, `source_is_s2`, `source_is_s3`, `name_length_ratio_context`, `address_length_ratio_context` |

**Excluded from model**: `s1_entity_id`, `candidate_entity_id`, `candidate_source`, `label`

---

## LightGBM Configuration (Default)

```python
DEFAULT_LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
    "force_col_wise": True,
}
```

---

## Validation Results

### Test Suite
| Test | Result |
|------|--------|
| **Full test suite** | 200 passed, 1 skipped |
| **New training tests** | 33 passed |
| **Integration test** | ✅ Full pipeline works |

### Integration Test Results
```
=== FULL PIPELINE TEST PASSED ===
Generated 1000 candidate pairs
Labeled 1000 candidates: Positive: 25, Negative: 975
Train: 800 rows, 40 S1 entities
Val: 200 rows, 10 S1 entities
Entity split verified: no S1 overlap
Built features: (1000, 21)
Model features: (1000, 18)
Feature columns verified!
Training LightGBM model...
Training until validation scores don't improve for 50 rounds
[20] valid's binary_logloss: 0.0583035
Model trained: 18 features, best_iteration=20
Model saved and loaded successfully! Features: 18
```

---

## Quality Checks Verified

| Check | Result |
|-------|--------|
| Exactly 18 model features | ✅ |
| Identity columns excluded | ✅ |
| Labels excluded from features | ✅ |
| No NaN/Inf in features | ✅ |
| Labels are binary 0/1 | ✅ |
| Entity-level split: no S1 leakage | ✅ |
| Deterministic splits (seed=42) | ✅ |
| Model saves & loads successfully | ✅ |
| All 18 feature names exact match | ✅ |

---

## Real Dataset Status

| Item | Status |
|------|--------|
| **REAL MODEL TRAINED** | **NO** |
| **REASON** | Required dataset (`student_resource/dataset/train/train_ground_truth.tsv`) not available locally |
| **Implementation** | Complete with synthetic validation |
| **Artifact created** | Only during synthetic test (temp dir) |

---

## No Changes To (M2 Boundary Respected)

✅ **Not modified:**
- `src/business_entity_resolution/preprocessing/`
- `src/business_entity_resolution/blocking/`
- `src/business_entity_resolution/evaluation/`
- `src/business_entity_resolution/model/predict.py`
- `src/business_entity_resolution/model/threshold.py`
- `src/business_entity_resolution/pipeline.py`
- M1 feature files (`name_features.py`, `address_features.py`, `pair_features.py`)

---

## Git Summary

```bash
Modified:  src/business_entity_resolution/model/train.py (+197 lines)
Created:   tests/test_training.py (33 tests)
Created:   scripts/verify_pipeline.py
```

---

## Next Phase (Not Started)

**Phase 3 (M2)**: Prediction & Thresholding
- Implement `predict_match_probabilities()` in `predict.py`
- Implement `find_optimal_threshold()` in `threshold.py`
- Implement `apply_entity_thresholds()` in `threshold.py`
- End-to-end evaluation with macro F0.5