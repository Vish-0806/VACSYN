"""
Threshold Optimization and Decision Layer.

Responsible for:
- Finding the global or per-country probability threshold that maximizes macro F0.5 score.
- Singleton protection: ensuring low-confidence entities correctly receive empty match lists.
- Multi-match filtering: allowing multiple high-probability candidates per S1 entity.
- Precision vs. Recall tradeoff calibration (macro F0.5 weights precision 2x over recall).

Ownership: Member 2 (Features & Model).
"""

import logging
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from ..evaluation.metrics import compute_macro_f05

logger = logging.getLogger(__name__)


def find_optimal_threshold(
    predictions_df: pd.DataFrame,
    ground_truth_mapping: Dict[str, List[str]],
    threshold_range: Tuple[float, float, float] = (0.2, 0.9, 0.02),
) -> float:
    """
    Search grid of probability thresholds to maximize validation macro F0.5 score.

    Args:
        predictions_df: DataFrame with ['s1_entity_id', 'candidate_entity_id', 'match_probability'].
        ground_truth_mapping: Mapping from s1_entity_id to list of true matching IDs.
        threshold_range: (min_threshold, max_threshold, step).

    Returns:
        Optimal threshold value yielding maximum macro F0.5.

    Raises:
        ValueError: If required columns are missing, threshold_range is invalid,
                    or no valid thresholds found.
    """
    # Validate required columns
    required_cols = ["s1_entity_id", "candidate_entity_id", "match_probability"]
    for col in required_cols:
        if col not in predictions_df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Validate threshold_range
    if len(threshold_range) != 3:
        raise ValueError(f"threshold_range must be a tuple of (min, max, step), got {threshold_range}")
    min_thresh, max_thresh, step = threshold_range
    if not (0 <= min_thresh <= 1 and 0 <= max_thresh <= 1 and step > 0):
        raise ValueError(f"Invalid threshold_range: min/max must be in [0,1], step > 0, got {threshold_range}")
    if min_thresh > max_thresh:
        raise ValueError(f"threshold_range min ({min_thresh}) > max ({max_thresh})")

    if len(predictions_df) == 0:
        raise ValueError("predictions_df is empty")

    # Generate threshold grid
    thresholds = np.arange(min_thresh, max_thresh + step / 2, step)
    thresholds = np.round(thresholds, 10)  # Avoid floating point issues

    best_threshold = None
    best_score = -1.0

    # For each threshold, build predictions dict and evaluate
    for thresh in thresholds:
        # Filter predictions by threshold
        filtered = predictions_df[predictions_df["match_probability"] >= thresh]

        # Build predictions dict: s1_entity_id -> list of candidate_entity_id
        predictions_dict = {}
        for s1_id in ground_truth_mapping.keys():
            s1_preds = filtered[filtered["s1_entity_id"] == s1_id]["candidate_entity_id"].tolist()
            # Deduplicate while preserving order
            seen = set()
            unique_preds = []
            for cid in s1_preds:
                if cid not in seen:
                    seen.add(cid)
                    unique_preds.append(cid)
            predictions_dict[s1_id] = unique_preds

        # Evaluate using M3 macro F0.5
        metrics = compute_macro_f05(predictions_dict, ground_truth_mapping)
        score = metrics.get("macro_f05", 0.0)

        if score > best_score:
            best_score = score
            best_threshold = thresh
        # Tie-breaking: lower threshold wins (deterministic)
        elif score == best_score and best_threshold is not None and thresh < best_threshold:
            best_threshold = thresh

    if best_threshold is None:
        raise ValueError("No valid threshold found in range")

    return float(best_threshold)


def apply_entity_thresholds(
    predictions_df: pd.DataFrame,
    threshold: float,
    all_s1_ids: List[str],
) -> pd.DataFrame:
    """
    Apply decision threshold to scored candidate pairs to produce final match mappings.

    Ensures every Source 1 entity appears exactly once in the final result,
    assigning an empty match string to singletons.

    Args:
        predictions_df: DataFrame with ['s1_entity_id', 'candidate_entity_id', 'match_probability'].
        threshold: Chosen decision boundary.
        all_s1_ids: Complete set of required Source 1 entity IDs.

    Returns:
        DataFrame with columns:
        - source1_entity_id
        - matched_entity_ids (comma-separated S2/S3 IDs, or empty string)

    Raises:
        ValueError: If threshold not in [0, 1], required columns missing,
                    or all_s1_ids is not a list.
    """
    # Validate threshold
    if not (0 <= threshold <= 1):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    # Validate required columns
    required_cols = ["s1_entity_id", "candidate_entity_id", "match_probability"]
    for col in required_cols:
        if col not in predictions_df.columns:
            raise ValueError(f"Missing required column: {col}")

    if not isinstance(all_s1_ids, (list, tuple, set)):
        raise ValueError(f"all_s1_ids must be a list/tuple/set, got {type(all_s1_ids).__name__}")

    # Handle empty inputs
    if len(all_s1_ids) == 0:
        return pd.DataFrame(columns=["source1_entity_id", "matched_entity_ids"])

    # Filter predictions by threshold
    filtered = predictions_df[predictions_df["match_probability"] >= threshold].copy()

    # Remove duplicate (s1_entity_id, candidate_entity_id) pairs
    filtered.drop_duplicates(subset=["s1_entity_id", "candidate_entity_id"], inplace=True)

    # Group by s1_entity_id and collect candidate_entity_ids
    grouped = filtered.groupby("s1_entity_id")["candidate_entity_id"].apply(list).to_dict()

    # Build output for all required S1 entities
    rows = []
    for s1_id in all_s1_ids:
        if not s1_id or (isinstance(s1_id, float) and np.isnan(s1_id)):
            continue
        s1_str = str(s1_id).strip()
        if not s1_str:
            continue

        candidates = grouped.get(s1_str, [])
        # Deduplicate preserving order
        seen = set()
        unique_candidates = []
        for cid in candidates:
            cid_str = str(cid).strip()
            if cid_str and cid_str not in seen:
                seen.add(cid_str)
                unique_candidates.append(cid_str)

        matched_str = ",".join(unique_candidates) if unique_candidates else ""
        rows.append({
            "source1_entity_id": s1_str,
            "matched_entity_ids": matched_str,
        })

    return pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])
