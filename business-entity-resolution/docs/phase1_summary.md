# Phase 1: Business Entity Resolution - Feature Engineering Complete

## Overview
All five Phase 1 M2 feature engineering tasks completed and validated.

| Task | Component | Features | File | Tests | Status |
|------|-----------|----------|------|-------|--------|
| 1.1 | Business Name Features | 7 | `name_features.py` | 24 | ✅ CLOSED |
| 1.2 | Address Features | 6 | `address_features.py` | 23 | ✅ CLOSED |
| 1.3 | Pair/Context Features | 5 | `pair_features.py` | 19 | ✅ CLOSED |
| 1.4 | Feature Assembly | 18 (combined) | `pair_features.py` | 24 | ✅ CLOSED |
| 1.5 | Feature Validation | N/A | `test_feature_validation.py` | 29 | ✅ CLOSED |

**Total Model Features: 18** (7 name + 6 address + 5 pair/context)

---

## Task 1.1: Business Name Features

**File**: `src/business_entity_resolution/features/name_features.py`

**Function**: `compute_name_similarity_features(name1, name2) -> Dict[str, float]`

**7 Features**:
1. `name_token_jaccard` - Token set Jaccard similarity
2. `name_token_overlap_count` - Token intersection count
3. `name_char_ngram_cosine` - Average of 2-gram & 3-gram cosine similarity
4. `name_edit_similarity` - Normalized Levenshtein similarity (rapidfuzz)
5. `name_length_ratio` - Min/max normalized name length ratio
6. `name_exact_match` - Exact normalized name equality (0/1)
7. `name_first_token_match` - First token equality (0/1)

**M1 Helpers Used**: `normalize_business_name()`, `tokenize_business_name()`

**Tests**: 24 passed (Unicode/Indic, bounds, edge cases, missing values)

---

## Task 1.2: Address Features

**File**: `src/business_entity_resolution/features/address_features.py`

**Function**: `compute_address_similarity_features(address1, address2) -> Dict[str, float]`

**6 Features**:
1. `address_token_jaccard` - Token set Jaccard (M1 tokenizes with stopword filtering)
2. `address_token_overlap_count` - Token intersection count
3. `address_house_number_match` - Exact house number equality (0/1)
4. `address_edit_similarity` - Normalized Levenshtein similarity
5. `address_length_ratio` - Min/max normalized address length ratio
6. `address_missing` - Missing address indicator (0/1)

**M1 Helpers Used**: `normalize_address()`, `extract_house_number()`, `tokenize_address()`

**Tests**: 23 passed (abbreviations, Unicode, missing addresses, house numbers)

---

## Task 1.3: Pair/Context Features

**File**: `src/business_entity_resolution/features/pair_features.py`

**Function**: `compute_pair_context_features(s1_country, candidate_country, candidate_source, s1_business_name, candidate_business_name, s1_address, candidate_address) -> Dict[str, float]`

**5 Features**:
1. `country_match` - Country equality (0/1)
2. `source_is_s2` - Candidate source is S2 (0/1)
3. `source_is_s3` - Candidate source is S3 (0/1)
4. `name_length_ratio_context` - Name length ratio using M1 normalization
5. `address_length_ratio_context` - Address length ratio using M1 normalization

**M1 Helpers Used**: `normalize_business_name()`, `normalize_address()`

**Tests**: 19 passed (S2/S3 exclusivity, Unicode, bounds, missing values)

---

## Task 1.4: Feature Assembly

**File**: `src/business_entity_resolution/features/pair_features.py`

**APIs**:
- `PairFeatureExtractor.extract_features(candidate_pairs_df, s1_records, s2_records, s3_records) -> pd.DataFrame`
- `build_pair_features(candidate_pairs, s1_data, candidate_data) -> pd.DataFrame`

**Output**: 21 columns (3 identity + 18 model features) in exact specified order

**Implementation**: Calls the three feature modules - no duplicate logic

**Architecture Decision**: Kept in `pair_features.py` (not separate `feature_builder.py`) because existing exports and interfaces already reside there.

**Tests**: 24 passed (schema, identity preservation, S2/S3 lookup, values match direct calls)

---

## Task 1.5: Feature Validation

**File**: `tests/test_feature_validation.py`

**Validation Summary** (61 candidate pairs on synthetic data):

| Check | Result |
|-------|--------|
| Schema (18 model + 3 identity) | ✅ |
| All features finite (0 NaN/Inf) | ✅ |
| Range validation (all [0,1] bounded) | ✅ |
| Leakage audit (no ground truth params) | ✅ |
| Candidate-pair integrity | ✅ |
| Determinism | ✅ |
| Performance | 2,533 pairs/sec |

**Key Observations**:
- `country_match` = 1.0 constant (expected: M1 country-isolated blocking)
- `address_missing` highly skewed (96.7% = 0.0)
- No label leakage in feature functions or matrix
- Row-wise implementation parallelizable for production scale

---

## Architecture Summary

### M1 Preprocessing Helpers (Frozen)
```python
from business_entity_resolution.preprocessing import (
    normalize_business_name,   # Unicode NFKC, lowercase, punctuation handling
    tokenize_business_name,    # Tokenization with min_length=2
    normalize_address,         # Abbreviation expansion, Unicode preservation
    extract_house_number,      # Multi-country house number extraction
    tokenize_address,          # Stopword-filtered tokenization
)
```

### Feature Modules (No Cross-Dependencies)
```
name_features.py     → compute_name_similarity_features()
address_features.py  → compute_address_similarity_features()
pair_features.py     → compute_pair_context_features()
                     → PairFeatureExtractor.extract_features()
                     → build_pair_features()
```

### Data Flow
```
S1/S2/S3 DataFrames
    ↓
CandidateGenerator (M1) → candidate_pairs.tsv (s1_entity_id, candidate_entity_id, candidate_source)
    ↓
PairFeatureExtractor.extract_features() / build_pair_features()
    ↓
Feature DataFrame (21 columns) → LightGBM
```

---

## Test Results

| Test Module | Tests | Status |
|-------------|-------|--------|
| `test_name_features.py` | 24 | ✅ |
| `test_address_features.py` | 23 | ✅ |
| `test_pair_features.py` | 19 | ✅ |
| `test_feature_builder.py` | 24 | ✅ |
| `test_feature_validation.py` | 29 | ✅ |
| **Project-wide** | **167** | ✅ (1 skipped) |

---

## Files Created/Modified

### Source Code
| File | Status |
|------|--------|
| `src/business_entity_resolution/features/name_features.py` | ✅ Implemented |
| `src/business_entity_resolution/features/address_features.py` | ✅ Implemented |
| `src/business_entity_resolution/features/pair_features.py` | ✅ Implemented (Tasks 1.3 + 1.4) |

### Tests
| File | Tests | Status |
|------|-------|--------|
| `tests/test_name_features.py` | 24 | ✅ |
| `tests/test_address_features.py` | 23 | ✅ |
| `tests/test_pair_features.py` | 19 | ✅ |
| `tests/test_feature_builder.py` | 24 | ✅ |
| `tests/test_feature_validation.py` | 29 | ✅ |

### Documentation
| File | Purpose |
|------|---------|
| `docs/task_1_1_final_verification.md` | Task 1.1 verification |
| `docs/task_1_2_status.md` | Task 1.2 verification |
| `docs/task_1_3_status.md` | Task 1.3 verification |
| `docs/task_1_4_status.md` | Task 1.4 verification (incl. architecture decision) |
| `docs/task_1_4_implementation_audit.md` | Architecture deviation audit |
| `docs/task_1_5_validation_report.md` | Task 1.5 comprehensive validation |

---

## Dependencies Used (Existing)

| Package | Purpose |
|---------|---------|
| `rapidfuzz` | Levenshtein distance (already in requirements.txt) |
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn`, `scipy` | ML utilities |
| `lightgbm` | Gradient boosting (future use) |

**No new dependencies added.**

---

## Prohibited Changes Respected

✅ No modifications to:
- `src/business_entity_resolution/preprocessing/`
- `src/business_entity_resolution/blocking/`
- `src/business_entity_resolution/model/`
- `src/business_entity_resolution/evaluation/`

✅ No:
- Embeddings/LLMs/external APIs
- Country-specific logic
- Candidate generation logic
- Dense global similarity matrices
- Training/prediction/thresholding code

---

## Production Readiness Notes

1. **Scalability**: Row-wise feature computation (~2,500 pairs/sec single-threaded) parallelizable by candidate-pair chunks
2. **Determinism**: Bitwise identical output on repeated runs
3. **Memory**: Minimal - only materializes feature rows, not full source datasets
4. **Missing Data**: Robust handling of None/empty addresses and names
5. **Unicode/Indic**: Preserved through NFKC normalization

---

## Next Phase (Not Started)

**Phase 2**: Model Training & Evaluation
- Training dataset construction (sampling negatives)
- LightGBM training with cross-validation
- Threshold selection (macro F0.5)
- Error analysis
- Prediction pipeline

---

## Git Summary

**Commits**:
1. Task 1.1: `feat: implement name_features.py with 7 similarity features using frozen M1 preprocessing helpers`
2. Task 1.2: `feat: implement address_features.py with 6 similarity features using frozen M1 preprocessing helpers`
3. Task 1.3: `feat: implement pair_features.py with 5 context features using frozen M1 helpers`
4. Task 1.4: `feat: implement feature assembly in PairFeatureExtractor extracting 18 features from 3 modules`
5. Task 1.4 docs: `docs: record Task 1.4 architecture decision (assembly in pair_features.py) and close task`
6. Task 1.5: Validation test file and report (uncommitted)

**All Phase 1 Tasks Complete and Closed.**