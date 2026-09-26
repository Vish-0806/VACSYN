"""
Name similarity feature extraction.

Responsible for extracting similarity signals between two business names:
- Token Jaccard similarity.
- Token overlap count.
- Character 2-gram similarity.
- Character 3-gram similarity.
- Normalized edit similarity (Levenshtein ratio).
- Exact normalized-name match indicator.
- First-token match indicator.
- Name character length ratio and absolute difference.

Ownership: Member 2 (Features & Model).
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def compute_name_similarity_features(
    name1: Optional[str],
    name2: Optional[str],
) -> Dict[str, float]:
    """
    Compute comprehensive name similarity metrics for a candidate pair.

    Args:
        name1: Reference S1 business name.
        name2: Candidate S2/S3 business name.

    Returns:
        Dictionary of numeric feature values:
        - name_token_jaccard: float
        - name_token_overlap_count: float
        - name_char_2gram_similarity: float
        - name_char_3gram_similarity: float
        - name_edit_similarity: float
        - name_exact_match: float (0.0 or 1.0)
        - name_first_token_match: float (0.0 or 1.0)
        - name_length_ratio: float
        - name_length_diff: float

    TODO:
        Implement string similarity algorithms (token set operations, n-gram bigram/trigram Dice, rapidfuzz ratio).
    """
    raise NotImplementedError("Name similarity features calculation is not implemented yet.")
