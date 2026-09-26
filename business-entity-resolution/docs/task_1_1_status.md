# Task 1.1: Business Name Feature Engineering - Status Report

## Task Summary
**Phase 1, Task 1.1** - Implement business-name feature engineering module (`src/business_entity_resolution/features/name_features.py`)

## Status: BLOCKED - Missing M1 Interface

### Blocking Issue
The frozen M1 preprocessing helpers required by this task are **not implemented**:

| Function | Location | Status |
|----------|----------|--------|
| `normalize_business_name(name, strip_legal=False)` | `src/business_entity_resolution/preprocessing/normalize.py:25` | `NotImplementedError` |
| `tokenize_name(name, min_length=2)` | `src/business_entity_resolution/preprocessing/normalize.py:62` | `NotImplementedError` |

**Note**: Task references `tokenize_business_name` but actual function name is `tokenize_name`.

Both functions contain only stub implementations with `TODO` comments and `raise NotImplementedError("...")`.

### Task Requirement Violation
The task states: *"M1 preprocessing is COMPLETE and FROZEN"* and *"Reuse the frozen M1 business-name preprocessing helpers."*

This is **not true** - M1 preprocessing is incomplete.

### Action Taken
1. ✅ Created `src/business_entity_resolution/features/name_features.py` with full implementation
2. ✅ Created test file `tests/test_name_features.py` with comprehensive test coverage
3. ❌ **Cannot validate** - tests fail because M1 helpers raise `NotImplementedError`
4. ❌ **Cannot proceed** per strict instructions: *"If a required interface is missing: STOP. Report the exact missing interface. Do not guess."*

### Files Created/Modified
| File | Status |
|------|--------|
| `src/business_entity_resolution/features/name_features.py` | Modified (implementation added) |
| `tests/test_name_features.py` | Created (new test file) |

### Implementation Details (Ready for Validation)
The implementation includes all 7 required features:
1. `name_token_jaccard` - Token set Jaccard similarity
2. `name_token_overlap_count` - Token intersection count
3. `name_char_ngram_cosine` - Combined 2-gram/3-gram cosine similarity (exposed as `name_char_2gram_similarity` + `name_char_3gram_similarity`)
4. `name_edit_similarity` - Normalized Levenshtein similarity using `rapidfuzz`
5. `name_length_ratio` - Min/max length ratio
6. `name_exact_match` - Exact normalized name match
7. `name_first_token_match` - First token equality

**Dependencies**: Uses `rapidfuzz` (already in `requirements.txt`) for Levenshtein distance.

### Required Resolution
M1 team must implement:
```python
# preprocessing/normalize.py
def normalize_business_name(name: Optional[str], strip_legal: bool = False) -> str:
    # Unicode NFKC, lowercase, punctuation handling, whitespace normalization
    ...

def tokenize_name(name: Optional[str], min_length: int = 2) -> List[str]:
    # Split on whitespace, filter by min_length
    ...
```

Once M1 provides working helpers, Task 1.1 can be completed and validated.