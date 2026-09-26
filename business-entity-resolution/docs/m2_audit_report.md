# M2 Feature Engineering - Read-Only Audit Report

## Summary
**Branch**: `features-ml`  
**Latest Commit**: `e25743b`  
**Test Suite**: 167 passed, 1 skipped (2.78s)

---

## 1. Feature Extraction

### 1.1 Implementation Status: **COMPLETE & TESTED**

| File | Status | Features Implemented |
|------|--------|---------------------|
| `src/business_entity_resolution/features/name_features.py` | ✅ Complete | 7/7 |
| `src/business_entity_resolution/features/address_features.py` | ✅ Complete | 6/6 |
| `src/business_entity_resolution/features/pair_features.py` | ✅ Complete | 5/5 + Assembly |

### 1.2 Exact Finalized Feature Names

**Name (7)** — `src/business_entity_resolution/features/name_features.py:58`
- `name_token_jaccard`
- `name_token_overlap_count`
- `name_char_ngram_cosine`
- `name_edit_similarity`
- `name_length_ratio`
- `name_exact_match`
- `name_first_token_match`

**Address (6)** — `src/business_entity_resolution/features/address_features.py:28`
- `address_token_jaccard`
- `address_token_overlap_count`
- `address_house_number_match`
- `address_edit_similarity`
- `address_length_ratio`
- `address_missing`

**Pair/Context (5)** — `src/business_entity_resolution/features/pair_features.py:30`
- `country_match`
- `source_is_s2`
- `source_is_s3`
- `name_length_ratio_context`
- `address_length_ratio_context`

**Assembly** — `src/business_entity_resolution/features/pair_features.py:164`
- `PairFeatureExtractor.extract_features()` — produces 21-column DataFrame (3 identity + 18 model features)
- `build_pair_features()` — high-level API accepting DataFrames

### 1.3 Verified Implementation Details
- All 3 modules use **frozen M1 preprocessing helpers** exclusively (`normalize_business_name`, `tokenize_business_name`, `normalize_address`, `extract_house_number`, `tokenize_address`)
- No local normalization/tokenization duplication
- Deterministic output verified (bitwise identical on repeated runs)
- Zero NaN/Inf across all features
- All 18 model features within expected bounds

---

## 2. Model Training

### 2.1 Implementation Status: **STUB ONLY**

**File**: `src/business_entity_resolution/model/train.py`

```python
def train_lightgbm_model(
    train_features_df: pd.DataFrame,
    labels: pd.Series,
    val_features_df: Optional[pd.DataFrame] = None,
    val_labels: Optional[pd.Series] = None,
    params: Optional[Dict[str, Any]] = None,
    model_save_path: Optional[Path] = None,
) -> Any:
    raise NotImplementedError("LightGBM model training is not implemented yet.")
```

**Status**: 
- ✅ Function signature defined
- ❌ **No implementation** — raises `NotImplementedError`
- ❌ No training entry point executed
- ❌ No label handling implemented
- ❌ No training has been executed

---

## 3. Prediction

### 3.1 Implementation Status: **STUB ONLY**

**File**: `src/business_entity_resolution/model/predict.py`

```python
def predict_match_probabilities(
    model: Any,
    features_df: pd.DataFrame,
    batch_size: int = 100_000,
) -> pd.DataFrame:
    raise NotImplementedError("Batch probability prediction is not implemented yet.")
```

**Contract** (from docstring):
- **Input**: `features_df` with identifier columns + 18 model features
- **Output**: DataFrame with `s1_entity_id`, `candidate_entity_id`, `match_probability`
- **Batch behavior**: `batch_size=100_000` parameter defined but not implemented

**Pipeline integration** (`pipeline.py:437-441`):
```python
if model_path is not None:
    raise NotImplementedError(
        "M2 model prediction and feature extraction interfaces are not yet integrated. "
        "Model scoring is blocked until Member 2 finishes implementation."
    )
```

---

## 4. Thresholding

### 4.1 Implementation Status: **STUB ONLY**

**File**: `src/business_entity_resolution/model/threshold.py`

```python
def find_optimal_threshold(
    predictions_df: pd.DataFrame,
    ground_truth_mapping: Dict[str, List[str]],
    threshold_range: Tuple[float, float, float] = (0.2, 0.9, 0.02),
) -> float:
    raise NotImplementedError("Optimal threshold search is not implemented yet.")

def apply_entity_thresholds(
    predictions_df: pd.DataFrame,
    threshold: float,
    all_s1_ids: List[str],
) -> pd.DataFrame:
    raise NotImplementedError("Entity threshold application is not implemented yet.")
```

**Contract** (from docstrings):
- `find_optimal_threshold`: Grid search over `[0.2, 0.9)` step 0.02, maximize macro F0.5
- `apply_entity_thresholds`: Filter predictions, group by S1, singleton preservation

---

## 5. Model Artifact

### 5.1 Status: **NO ARTIFACT EXISTS**

| Search Pattern | Found |
|----------------|-------|
| `*.pkl` | ❌ |
| `*.model` | ❌ |
| `*.bin` | ❌ |
| `*.joblib` | ❌ |
| `*.txt` (model) | ❌ |

Only `requirements.txt` found. No trained model artifact exists in repository.

---

## 6. Tests

### 6.1 Actual Test Execution

**Command**: 
```bash
cd business-entity-resolution && PYTHONPATH=src python -m pytest tests/ --tb=no -q
```

**Result**: **167 passed, 1 skipped** (2.78s)

### 6.2 Breakdown by Module

| Test Module | Tests | Status |
|-------------|-------|--------|
| `test_name_features.py` | 24 | ✅ All pass |
| `test_address_features.py` | 23 | ✅ All pass |
| `test_pair_features.py` | 19 | ✅ All pass |
| `test_feature_builder.py` | 24 | ✅ All pass |
| `test_feature_validation.py` | 29 | ✅ All pass |
| `test_address.py` | 10 | ✅ All pass |
| `test_address_blocking.py` | 3 | ✅ All pass |
| `test_candidate_generator.py` | 3 | ✅ All pass |
| `test_contracts.py` | 2 | ✅ All pass |
| `test_exact.py` | 5 | ✅ All pass |
| `test_fuzzy.py` | 5 | ✅ All pass |
| `test_normalize.py` | 7 | ✅ All pass |
| `test_pipeline.py` | 9 | ✅ All pass |
| `test_tfidf.py` | 4 | ✅ All pass |
| `test_candidate_validation.py` | 1 | ⏭ Skipped (requires dataset) |

**Total**: 168 collected, 167 passed, 1 skipped

---

## 7. Remaining Blockers/Dependencies

### 7.1 Factual Blockers (Missing Implementation)

| Component | Blocker | Evidence |
|-----------|---------|----------|
| **LightGBM Training** | `train_lightgbm_model()` raises `NotImplementedError` | `train.py:50` |
| **Batch Prediction** | `predict_match_probabilities()` raises `NotImplementedError` | `predict.py:45` |
| **Threshold Optimization** | `find_optimal_threshold()` raises `NotImplementedError` | `threshold.py:39` |
| **Threshold Application** | `apply_entity_thresholds()` raises `NotImplementedError` | `threshold.py:66` |
| **Model Artifact** | No `.pkl`/`.model`/`.bin`/`.joblib` file exists | `Get-ChildItem` search |
| **Pipeline ML Integration** | `run_pipeline()` raises `NotImplementedError` when `model_path` provided | `pipeline.py:437-441` |

### 7.2 Untested but Not Missing

| Component | Status | Notes |
|-----------|--------|-------|
| `CandidateGenerator` | ✅ Tested | 3 tests pass; empirical recall test skipped (needs dataset) |
| `format_candidate_pairs_submission` | ✅ Tested | 9 pipeline tests pass |
| `format_matching_results` | ✅ Tested | 9 pipeline tests pass |
| `validate_match_candidate_consistency` | ✅ Tested | 9 pipeline tests pass |
| `generate_candidates_chunked` | ✅ Tested | 9 pipeline tests pass |
| `run_submission_validator` | ⚠️ Partial | Validator file exists; requires test dataset |

### 7.3 Dependency Status

| Dependency | Available | Notes |
|------------|-----------|-------|
| `rapidfuzz` | ✅ | Used for Levenshtein |
| `lightgbm` | ✅ | In `requirements.txt` |
| `pandas`, `numpy`, `scikit-learn` | ✅ | Core dependencies |
| Training dataset (`train_ground_truth.tsv`) | ❌ Local | Path `c:/VACSYN/student_resource/dataset/train/` not present locally |

---

## Conclusion

| Layer | Status | Completion |
|-------|--------|------------|
| Feature Engineering (M2) | ✅ **COMPLETE** | 18 features implemented, tested, validated |
| Feature Assembly | ✅ **COMPLETE** | 21-column DataFrame, deterministic |
| Model Training (M2) | ❌ **NOT STARTED** | Stub only |
| Prediction/Inference | ❌ **NOT STARTED** | Stub only |
| Threshold Optimization | ❌ **NOT STARTED** | Stub only |
| Model Artifact | ❌ **NONE** | No file exists |
| End-to-End Pipeline | ⚠️ **PARTIAL** | Candidate gen works; ML scoring blocked |

**Next Required Work**: Implement `train_lightgbm_model()`, `predict_match_probabilities()`, `find_optimal_threshold()`, `apply_entity_thresholds()`, then execute training on full dataset.