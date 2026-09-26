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

from .name_features import compute_name_similarity_features
from .address_features import compute_address_similarity_features

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


def _compute_all_features_for_pair(
    s1_record: Dict[str, Any],
    candidate_record: Dict[str, Any],
    candidate_source: str,
) -> Dict[str, float]:
    """
    Compute all 18 features for a single candidate pair.

    Args:
        s1_record: S1 entity record with keys: entity_id, business_name, business_address, country
        candidate_record: Candidate entity record (S2 or S3) with same keys
        candidate_source: "S2" or "S3"

    Returns:
        Dictionary of all 18 features
    """
    # Name features (7)
    name_features = compute_name_similarity_features(
        s1_record.get("business_name"),
        candidate_record.get("business_name"),
    )

    # Address features (6)
    address_features = compute_address_similarity_features(
        s1_record.get("business_address"),
        candidate_record.get("business_address"),
    )

    # Pair/context features (5)
    pair_features = compute_pair_context_features(
        s1_record.get("country"),
        candidate_record.get("country"),
        candidate_source,
        s1_record.get("business_name"),
        candidate_record.get("business_name"),
        s1_record.get("business_address"),
        candidate_record.get("business_address"),
    )

    # Merge all features
    all_features = {}
    all_features.update(name_features)
    all_features.update(address_features)
    all_features.update(pair_features)

    return all_features


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
        # Validate required columns
        required_cols = ["s1_entity_id", "candidate_entity_id", "candidate_source"]
        for col in required_cols:
            if col not in candidate_pairs_df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Define the expected feature column order
        feature_order = [
            # Identity columns
            "s1_entity_id",
            "candidate_entity_id",
            "candidate_source",
            # Name features (7)
            "name_token_jaccard",
            "name_token_overlap_count",
            "name_char_ngram_cosine",
            "name_edit_similarity",
            "name_length_ratio",
            "name_exact_match",
            "name_first_token_match",
            # Address features (6)
            "address_token_jaccard",
            "address_token_overlap_count",
            "address_house_number_match",
            "address_edit_similarity",
            "address_length_ratio",
            "address_missing",
            # Pair/context features (5)
            "country_match",
            "source_is_s2",
            "source_is_s3",
            "name_length_ratio_context",
            "address_length_ratio_context",
        ]

        results = []

        for _, row in candidate_pairs_df.iterrows():
            s1_id = row["s1_entity_id"]
            cand_id = row["candidate_entity_id"]
            cand_source = row["candidate_source"]

            # Retrieve S1 record
            if s1_id not in s1_records:
                raise KeyError(f"S1 entity ID not found: {s1_id}")
            s1_record = s1_records[s1_id]

            # Retrieve candidate record based on source
            if cand_source == "S2":
                if cand_id not in s2_records:
                    raise KeyError(f"S2 entity ID not found: {cand_id}")
                candidate_record = s2_records[cand_id]
            elif cand_source == "S3":
                if cand_id not in s3_records:
                    raise KeyError(f"S3 entity ID not found: {cand_id}")
                candidate_record = s3_records[cand_id]
            else:
                raise ValueError(f"Invalid candidate_source: {cand_source}. Expected 'S2' or 'S3'.")

            # Compute all features
            features = _compute_all_features_for_pair(s1_record, candidate_record, cand_source)

            # Build output row
            output_row = {
                "s1_entity_id": s1_id,
                "candidate_entity_id": cand_id,
                "candidate_source": cand_source,
            }
            output_row.update(features)
            results.append(output_row)

        # Create DataFrame with deterministic column order
        df = pd.DataFrame(results, columns=feature_order)

        # Validate no NaN/Inf in feature columns
        feature_cols = feature_order[3:]  # Skip identity columns
        for col in feature_cols:
            if df[col].isna().any():
                raise ValueError(f"NaN detected in feature column: {col}")
            if (df[col] == float('inf')).any() or (df[col] == float('-inf')).any():
                raise ValueError(f"Inf detected in feature column: {col}")

        return df


def build_pair_features(
    candidate_pairs: pd.DataFrame,
    s1_data: Any,
    candidate_data: Any,
) -> pd.DataFrame:
    """
    Build ML-ready feature DataFrame for candidate pairs.

    Args:
        candidate_pairs: DataFrame containing candidate pairs with columns
            ['s1_entity_id', 'candidate_entity_id', 'candidate_source'].
        s1_data: S1 records DataFrame with columns
            ['entity_id', 'business_name', 'business_address', 'country'].
        candidate_data: Candidate pool records DataFrame with columns
            ['entity_id', 'business_name', 'business_address', 'country', 'source'].

    Returns:
        DataFrame with feature columns suitable for model training or inference.
    """
    # Convert S1 records to lookup dict
    s1_records = {}
    for _, row in s1_data.iterrows():
        s1_records[row["entity_id"]] = {
            "entity_id": row["entity_id"],
            "business_name": row.get("business_name"),
            "business_address": row.get("business_address"),
            "country": row.get("country"),
        }

    # Convert candidate data to lookup dicts by source
    s2_records = {}
    s3_records = {}
    for _, row in candidate_data.iterrows():
        record = {
            "entity_id": row["entity_id"],
            "business_name": row.get("business_name"),
            "business_address": row.get("business_address"),
            "country": row.get("country"),
        }
        source = row.get("source", "").strip().upper()
        if source == "S2":
            s2_records[row["entity_id"]] = record
        elif source == "S3":
            s3_records[row["entity_id"]] = record
        else:
            logger.warning(f"Unknown candidate source: {source} for entity {row['entity_id']}")

    # Use PairFeatureExtractor to build features
    extractor = PairFeatureExtractor()
    return extractor.extract_features(candidate_pairs, s1_records, s2_records, s3_records)