# Task 1.1: Business Name Feature Engineering - Status Report

## Task Summary
**Phase 1, Task 1.1** - Implement business-name feature engineering module (`src/business_entity_resolution/features/name_features.py`)

## Status: ✅ COMPLETED

## Summary
Successfully implemented the business-name feature engineering module using the frozen M1 preprocessing helpers. All 24 unit tests pass, plus all 63 existing project tests pass (1 skipped).

---

## Files Modified/Created

| File | Status |
|------|--------|
| `src/business_entity_resolution/features/name_features.py` | **Modified** - Rewritten to use M1 helpers |
| `tests/test_name_features.py` | **Modified** - Updated to test 7 required features |

---

## Functions Implemented

### Public API
- `compute_name_similarity_features(name1, name2) -> Dict[str, float]`
  - Main entry point for computing all 7 name similarity features
  - Uses frozen M1 preprocessing helpers

### Private Helpers
- `_char_ngrams(text, n) -> Dict[str, int]` - Character n-gram frequency dictionary
- `_cosine_similarity(vec1, vec2) -> float` - Cosine similarity with fast-path for identical vectors

---

## Exact Seven Final Feature Names

The function returns exactly these seven features:

| Feature | Description | Range |
|---------|-------------|-------|
| `name_token_jaccard` | Token set Jaccard similarity | [0, 1] |
| `name_token_overlap_count` | Token intersection count | [0, ∞) |
| `name_char_ngram_cosine` | Average of 2-gram & 3-gram cosine | [0, 1] |
| `name_edit_similarity` | Normalized Levenshtein similarity | [0, 1] |
| `name_length_ratio` | Min/max normalized name length | [0, 1] |
| `name_exact_match` | Exact normalized name equality | {0, 1} |
| `name_first_token_match` | First token equality | {0, 1} |

---

## M1 Preprocessing Helpers Reused

The implementation **exclusively uses** the frozen M1 helpers (no local duplication):

```python
from business_entity_resolution.preprocessing import (
    normalize_business_name,
    tokenize_business_name,
)
```

| M1 Helper | Usage |
|-----------|-------|
| `normalize_business_name(name, strip_legal=False)` | Normalizes both names for exact match, edit similarity, length ratio, char n-grams |
| `tokenize_business_name(name, min_length=2)` | Tokenizes for Jaccard, overlap count, first token match |

**Key decisions:**
- `strip_legal=False` by default (legal suffixes retained as tokens)
- `min_length=2` for tokenization (matches M1 default)
- No local fallback normalization/tokenization

---

## Character 2/3-gram Implementation

```python
# 2-gram cosine
char_2gram_sim = _cosine_similarity(_char_ngrams(norm1, 2), _char_ngrams(norm2, 2))
# 3-gram cosine
char_3gram_sim = _cosine_similarity(_char_ngrams(norm1, 3), _char_ngrams(norm2, 3))
# Combined feature
name_char_ngram_cosine = (char_2gram_sim + char_3gram_sim) / 2.0
```

- Unicode-safe (operates on normalized Unicode strings)
- Sparse dictionary representation (no dense matrices)
- Fast-path optimization: identical vectors return exactly 1.0
- Handles strings shorter than n gracefully (returns 0.0 or 1.0 appropriately)

---

## Edit Similarity Implementation

```python
if norm1 and norm2:
    max_len = max(len(norm1), len(norm2))
    edit_dist = Levenshtein.distance(norm1, norm2)
    edit_similarity = 1.0 - (edit_dist / max_len)
elif not norm1 and not norm2:
    edit_similarity = 1.0
else:
    edit_similarity = 0.0
```

- Uses `rapidfuzz.distance.Levenshtein` (existing project dependency)
- Normalized: `1 - distance / max(len(a), len(b))`
- Identical strings → 1.0
- Empty/empty → 1.0, empty/non-empty → 0.0

---

## Missing Value Handling

| Input | Behavior |
|-------|----------|
| `None` | Treated as empty string by M1 normalizer |
| `""` / `"   "` | Returns empty normalized string |
| NaN (float) | Converted to string by M1 (`str(nan)` → `"nan"`), then normalized |

All features return **finite** values (no NaN, no ±Inf) for all input combinations.

---

## Unicode/Indic Validation

✅ **Verified working:**
- Devanagari script: `"अपोलो हॉस्पिटल"` → tokens `["अपोलो", "हॉस्पिटल"]`
- Mixed Latin/Indic: `"Apollo हॉस्पिटल"` → tokens `["apollo", "हॉस्पिटल"]`
- All 7 features compute correctly for Indic names
- No character loss during normalization (NFKC preserves combining marks)

---

## Tests Executed

### New/Updated Tests (24 tests in `test_name_features.py`)
| Test | Status |
|------|--------|
| `test_exact_match_case_insensitive` | ✅ |
| `test_punctuation_whitespace_variation` | ✅ |
| `test_unicode_preservation` | ✅ |
| `test_identical_token_sets_jaccard` | ✅ |
| `test_disjoint_token_sets_jaccard` | ✅ |
| `test_partial_token_overlap` | ✅ |
| `test_edit_similarity_identical` | ✅ |
| `test_edit_similarity_different` | ✅ |
| `test_first_token_match` | ✅ |
| `test_first_token_mismatch` | ✅ |
| `test_first_token_empty` | ✅ |
| `test_length_ratio_identical` | ✅ |
| `test_length_ratio_empty` | ✅ |
| `test_both_empty` | ✅ |
| `test_none_inputs` | ✅ |
| `test_no_nan_inf` | ✅ |
| `test_bounded_features` | ✅ |
| `test_char_ngram_cosine` | ✅ |
| `test_short_strings` | ✅ |
| `test_single_char_vs_multi_char` | ✅ |
| `test_indic_script_feature_computation` | ✅ |
| `test_mixed_script_names` | ✅ |
| `test_feature_names_exact` | ✅ |
| `test_legal_suffix_handling` | ✅ |

### Full Project Test Suite
- **63 passed, 1 skipped** (all existing tests continue to pass)

---

## Git Status

```
Modified: src/business_entity_resolution/features/name_features.py
Modified: tests/test_name_features.py
```

No changes to:
- `src/business_entity_resolution/preprocessing/`
- `src/business_entity_resolution/blocking/`
- `src/business_entity_resolution/features/address_features.py`
- `src/business_entity_resolution/features/pair_features.py`
- `src/business_entity_resolution/model/`
- `src/business_entity_resolution/evaluation/`

---

## Assumptions

1. M1 `normalize_business_name` with `strip_legal=False` is the correct default for feature computation
2. M1 `tokenize_business_name` with `min_length=2` is the correct tokenization
3. Empty/empty token sets → Jaccard = 0.0 (not 1.0) - consistent with "no shared tokens"
4. `rapidfuzz` is the accepted Levenshtein implementation (already in requirements.txt)

---

## Unresolved Issues

None. Task 1.1 is complete and validated.

---

## Next Steps

Task 1.1 is **complete**. Ready for Task 1.2 (Address Features) or Task 1.3 (Pair/Context Features) as per project plan.

**DO NOT PROCEED** - Stop after Task 1.1 as instructed.