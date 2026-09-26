"""
Address similarity feature extraction.

Responsible for extracting geographic and address signals between two records:
- Address token Jaccard similarity.
- Address token overlap count.
- Character-level similarity.
- Edit similarity (Levenshtein).
- House/building number match indicator.
- Street and locality similarity.
- Address character length ratio and difference.
- Missing-address indicator flag.

Ownership: Member 2 (Features & Model).
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def compute_address_similarity_features(
    address1: Optional[str],
    address2: Optional[str],
) -> Dict[str, float]:
    """
    Compute address similarity metrics for a candidate pair.

    Args:
        address1: Reference S1 address string.
        address2: Candidate S2/S3 address string (may be missing/empty).

    Returns:
        Dictionary of numeric feature values:
        - address_token_jaccard: float
        - address_token_overlap_count: float
        - address_char_similarity: float
        - address_edit_similarity: float
        - house_number_exact_match: float (-1.0 if missing, 0.0 if mismatch, 1.0 if match)
        - street_similarity: float
        - address_length_ratio: float
        - address_length_diff: float
        - is_candidate_address_missing: float (0.0 or 1.0)

    TODO:
        Implement address feature calculations handling missing addresses gracefully.
    """
    raise NotImplementedError("Address similarity features calculation is not implemented yet.")
