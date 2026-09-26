"""
Pairwise Feature Assembly Layer.

Responsible for combining:
- Business name similarity features
- Address similarity features
- Country and candidate source context (S2 vs S3 indicator)
- Entity-level aggregated signals

Output is designed directly for LightGBM gradient boosting ingestion.

Ownership: Member 2 (Features & Model).
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd

from business_entity_resolution.preprocessing import (
    normalize_business_name,
    normalize_address,
)

logger = logging.getLogger(__name__)


def compute_pair_context_features(
    s1_country: Optional[str],
    candidate_country: Optional[str],
    candidate_source: Optional[str],
    s1_business_name: Optional[str],
    candidate_business_name: Optional[str],
    s1_address: Optional[str],
    candidate_address: Optional[str],
) -> Dict[str, float]:
    """
    Compute pair/context features for a candidate pair.

    Uses frozen M1 normalization helpers for name and address length ratios.

    Args:
        s1_country: S1 record country.
        candidate_country: Candidate record country.
        candidate_source: Candidate source ("S2" or "S3").
        s1_business_name: S1 business name.
        candidate_business_name: Candidate business name.
        s1_address: S1 address.
        candidate_address: Candidate address.

    Returns:
        Dictionary of exactly five numeric feature values:
        - country_match: float (0.0 or 1.0)
        - source_is_s2: float (0.0 or 1.0)
        - source_is_s3: float (0.0 or 1.0)
        - name_length_ratio_context: float [0, 1]
        - address_length_ratio_context: float [0, 1]
    """
    # 1. Country match - case-insensitive, whitespace-normalized comparison
    if s1_country and candidate_country:
        s1_ctry = str(s1_country).strip().lower()
        cand_ctry = str(candidate_country).strip().lower()
        country_match = 1.0 if s1_ctry == cand_ctry else 0.0
    else:
        country_match = 0.0

    # 2. Source indicators
    source_str = str(candidate_source).strip().upper() if candidate_source else ""
    source_is_s2 = 1.0 if source_str == "S2" else 0.0
    source_is_s3 = 1.0 if source_str == "S3" else 0.0

    # 3. Name length ratio context
    norm_s1_name = normalize_business_name(s1_business_name, strip_legal=False)
    norm_cand_name = normalize_business_name(candidate_business_name, strip_legal=False)
    len1 = len(norm_s1_name)
    len2 = len(norm_cand_name)
    if len1 == 0 and len2 == 0:
        name_length_ratio_context = 1.0
    elif len1 == 0 or len2 == 0:
        name_length_ratio_context = 0.0
    else:
        name_length_ratio_context = min(len1, len2) / max(len1, len2)

    # 4. Address length ratio context
    norm_s1_addr = normalize_address(s1_address)
    norm_cand_addr = normalize_address(candidate_address)
    len1 = len(norm_s1_addr)
    len2 = len(norm_cand_addr)
    if len1 == 0 and len2 == 0:
        address_length_ratio_context = 1.0
    elif len1 == 0 or len2 == 0:
        address_length_ratio_context = 0.0
    else:
        address_length_ratio_context = min(len1, len2) / max(len1, len2)

    return {
        "country_match": country_match,
        "source_is_s2": source_is_s2,
        "source_is_s3": source_is_s3,
        "name_length_ratio_context": name_length_ratio_context,
        "address_length_ratio_context": address_length_ratio_context,
    }


class PairFeatureExtractor:
    """
    Coordinates extraction of feature vectors for candidate pairs.
    """

    def __init__(self) -> None:
        """Initialize feature extractor."""
        pass

    def extract_features(
        self,
        candidate_pairs_df: pd.DataFrame,
        s1_records: Dict[str, Dict[str, Any]],
        s2_records: Dict[str, Dict[str, Any]],
        s3_records: Dict[str, Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Extract tabular feature DataFrame for candidate pairs.

        Args:
            candidate_pairs_df: DataFrame with ['s1_entity_id', 'candidate_entity_id', 'candidate_source'].
            s1_records: Lookup mapping for S1 records.
            s2_records: Lookup mapping for S2 records.
            s3_records: Lookup mapping for S3 records.

        Returns:
            DataFrame containing pair identifiers and numeric feature columns ready for LightGBM.
        """
        # This will be implemented in a later task
        raise NotImplementedError("Feature extraction pipeline is not implemented yet.")


def build_pair_features(
    candidate_pairs: pd.DataFrame,
    s1_data: Any,
    candidate_data: Any,
) -> pd.DataFrame:
    """
    Build ML-ready feature DataFrame for candidate pairs.

    Args:
        candidate_pairs: DataFrame containing candidate pairs.
        s1_data: Reference records.
        candidate_data: Candidate pool records.

    Returns:
        DataFrame with feature columns suitable for model training or inference.
    """
    # This will be implemented in a later task
    raise NotImplementedError("build_pair_features function is not implemented yet.")