"""
Address similarity feature extraction.

Responsible for extracting geographic and address signals between two records:
- Address token Jaccard similarity.
- Address token overlap count.
- House/building number match indicator.
- Normalized edit similarity (Levenshtein).
- Address character length ratio.
- Missing-address indicator flag.

Ownership: Member 2 (Features & Model).
"""

import logging
from typing import Dict, Optional

from rapidfuzz.distance import Levenshtein
from business_entity_resolution.preprocessing import (
    normalize_address,
    extract_house_number,
    tokenize_address,
)

logger = logging.getLogger(__name__)


def compute_address_similarity_features(
    address1: Optional[str],
    address2: Optional[str],
) -> Dict[str, float]:
    """
    Compute address similarity metrics for a candidate pair.

    Uses frozen M1 preprocessing helpers for normalization, tokenization, and house number extraction.

    Args:
        address1: Reference S1 address string.
        address2: Candidate S2/S3 address string (may be missing/empty).

    Returns:
        Dictionary of exactly six numeric feature values:
        - address_token_jaccard: float
        - address_token_overlap_count: float
        - address_house_number_match: float (0.0 or 1.0)
        - address_edit_similarity: float
        - address_length_ratio: float
        - address_missing: float (0.0 or 1.0)
    """
    # Normalize using frozen M1 helper
    norm1 = normalize_address(address1)
    norm2 = normalize_address(address2)

    # Tokenize using frozen M1 helper
    tokens1 = tokenize_address(address1)
    tokens2 = tokenize_address(address2)

    # Extract house numbers using frozen M1 helper
    house_num1 = extract_house_number(address1)
    house_num2 = extract_house_number(address2)

    # 1. Token Jaccard similarity
    if tokens1 or tokens2:
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        if union:
            token_jaccard = len(intersection) / len(union)
        else:
            token_jaccard = 0.0
        token_overlap_count = float(len(intersection))
    else:
        token_jaccard = 0.0
        token_overlap_count = 0.0

    # 2. House number match
    if house_num1 is not None and house_num2 is not None:
        house_number_match = 1.0 if house_num1 == house_num2 else 0.0
    else:
        house_number_match = 0.0

    # 3. Normalized edit similarity
    if norm1 and norm2:
        max_len = max(len(norm1), len(norm2))
        edit_dist = Levenshtein.distance(norm1, norm2)
        edit_similarity = 1.0 - (edit_dist / max_len)
    elif not norm1 and not norm2:
        edit_similarity = 1.0
    else:
        edit_similarity = 0.0

    # 4. Address length ratio
    len1 = len(norm1)
    len2 = len(norm2)
    if len1 == 0 and len2 == 0:
        length_ratio = 1.0
    elif len1 == 0 or len2 == 0:
        length_ratio = 0.0
    else:
        length_ratio = min(len1, len2) / max(len1, len2)

    # 5. Address missing indicator
    missing = 1.0 if (not norm1 or not norm2) else 0.0

    return {
        "address_token_jaccard": token_jaccard,
        "address_token_overlap_count": token_overlap_count,
        "address_house_number_match": house_number_match,
        "address_edit_similarity": edit_similarity,
        "address_length_ratio": length_ratio,
        "address_missing": missing,
    }