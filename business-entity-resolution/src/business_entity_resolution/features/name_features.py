"""
Name similarity feature extraction.

Responsible for extracting similarity signals between two business names:
- Token Jaccard similarity.
- Token overlap count.
- Character 2/3-gram cosine similarity.
- Normalized edit similarity (Levenshtein ratio).
- Exact normalized-name match indicator.
- First-token match indicator.
- Name character length ratio.

Ownership: Member 2 (Features & Model).
"""

import logging
from typing import Dict, List, Optional, Set

from rapidfuzz.distance import Levenshtein
from business_entity_resolution.preprocessing import (
    normalize_business_name,
    tokenize_business_name,
)

logger = logging.getLogger(__name__)


def _char_ngrams(text: str, n: int) -> Dict[str, int]:
    """Generate character n-gram frequency dictionary."""
    if len(text) < n:
        return {}
    return {text[i:i+n]: 1 for i in range(len(text) - n + 1)}


def _cosine_similarity(vec1: Dict[str, int], vec2: Dict[str, int]) -> float:
    """Compute cosine similarity between two frequency dictionaries."""
    if not vec1 and not vec2:
        return 1.0
    if not vec1 or not vec2:
        return 0.0

    # Fast path for identical vectors to avoid floating point drift
    if vec1 == vec2:
        return 1.0

    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum(vec1[k] * vec2[k] for k in intersection)

    norm1 = sum(v * v for v in vec1.values()) ** 0.5
    norm2 = sum(v * v for v in vec2.values()) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return numerator / (norm1 * norm2)


def compute_name_similarity_features(
    name1: Optional[str],
    name2: Optional[str],
) -> Dict[str, float]:
    """
    Compute comprehensive name similarity metrics for a candidate pair.

    Uses frozen M1 preprocessing helpers for normalization and tokenization.

    Args:
        name1: Reference S1 business name.
        name2: Candidate S2/S3 business name.

    Returns:
        Dictionary of exactly seven numeric feature values:
        - name_token_jaccard: float
        - name_token_overlap_count: float
        - name_char_ngram_cosine: float
        - name_edit_similarity: float
        - name_length_ratio: float
        - name_exact_match: float (0.0 or 1.0)
        - name_first_token_match: float (0.0 or 1.0)
    """
    # Normalize using frozen M1 helper
    norm1 = normalize_business_name(name1, strip_legal=False)
    norm2 = normalize_business_name(name2, strip_legal=False)

    # Tokenize using frozen M1 helper
    tokens1 = tokenize_business_name(name1, min_length=2)
    tokens2 = tokenize_business_name(name2, min_length=2)

    set1: Set[str] = set(tokens1)
    set2: Set[str] = set(tokens2)

    intersection = set1 & set2
    union = set1 | set2

    if union:
        token_jaccard = len(intersection) / len(union)
    else:
        token_jaccard = 0.0

    token_overlap_count = float(len(intersection))

    # Character n-gram cosine: average of 2-gram and 3-gram cosine
    char_2gram_sim = _cosine_similarity(_char_ngrams(norm1, 2), _char_ngrams(norm2, 2))
    char_3gram_sim = _cosine_similarity(_char_ngrams(norm1, 3), _char_ngrams(norm2, 3))
    char_ngram_cosine = (char_2gram_sim + char_3gram_sim) / 2.0

    # Normalized edit similarity using Levenshtein distance
    if norm1 and norm2:
        max_len = max(len(norm1), len(norm2))
        edit_dist = Levenshtein.distance(norm1, norm2)
        edit_similarity = 1.0 - (edit_dist / max_len)
    elif not norm1 and not norm2:
        edit_similarity = 1.0
    else:
        edit_similarity = 0.0

    # Exact normalized name match
    exact_match = 1.0 if norm1 == norm2 and norm1 else 0.0

    # First token match
    first_token1 = tokens1[0] if tokens1 else ""
    first_token2 = tokens2[0] if tokens2 else ""
    first_token_match = 1.0 if first_token1 and first_token1 == first_token2 else 0.0

    # Name length ratio
    len1 = len(norm1)
    len2 = len(norm2)
    if len1 == 0 and len2 == 0:
        length_ratio = 1.0
    elif len1 == 0 or len2 == 0:
        length_ratio = 0.0
    else:
        length_ratio = min(len1, len2) / max(len1, len2)

    return {
        "name_token_jaccard": token_jaccard,
        "name_token_overlap_count": token_overlap_count,
        "name_char_ngram_cosine": char_ngram_cosine,
        "name_edit_similarity": edit_similarity,
        "name_length_ratio": length_ratio,
        "name_exact_match": exact_match,
        "name_first_token_match": first_token_match,
    }