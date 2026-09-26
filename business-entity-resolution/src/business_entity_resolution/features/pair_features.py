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

logger = logging.getLogger(__name__)


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

        TODO:
            1. Iterate over candidate pairs in batches.
            2. Compute name features via compute_name_similarity_features.
            3. Compute address features via compute_address_similarity_features.
            4. Append candidate source indicator (is_source2, is_source3).
            5. Return clean numeric feature matrix.
        """
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

    TODO:
        Implement feature compilation pipeline.
    """
    raise NotImplementedError("build_pair_features function is not implemented yet.")
