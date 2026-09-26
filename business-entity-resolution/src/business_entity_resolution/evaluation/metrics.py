"""
Official Challenge Metrics Calculation Module.

Implements the official macro F0.5 score:
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

Metric Rules:
- Evaluated per Source 1 entity and then macro-averaged across ALL S1 entities.
- Singletons: If true matches = empty and predicted = empty -> F0.5 = 1.0.
- False merges on singletons: If true matches = empty and predicted != empty -> F0.5 = 0.0.
- Precision weighted 2x over recall (beta = 0.5).

Ownership: Member 3 (Evaluation).
"""

import logging
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


def compute_entity_f05(predicted_ids: Set[str], true_ids: Set[str]) -> float:
    """
    Compute F0.5 score for a single Source 1 entity.

    Args:
        predicted_ids: Set of predicted matching S2/S3 entity IDs.
        true_ids: Set of ground truth matching S2/S3 entity IDs.

    Returns:
        F0.5 score between 0.0 and 1.0.

    TODO:
        - Handle singleton case: true_ids is empty -> 1.0 if predicted_ids is empty, else 0.0.
        - Handle missed match case: predicted_ids is empty and true_ids is not -> 0.0.
        - Calculate Precision = |true & pred| / |pred|.
        - Calculate Recall = |true & pred| / |true|.
        - Return (1.25 * P * R) / (0.25 * P + R) if (0.25 * P + R) > 0 else 0.0.
    """
    raise NotImplementedError("Entity F0.5 calculation is not implemented yet.")


def compute_macro_f05(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
) -> Dict[str, float]:
    """
    Compute macro-averaged F0.5, Precision, and Recall across all S1 entities.

    Args:
        predictions: Mapping from s1_entity_id to list of predicted match IDs.
        ground_truth: Mapping from s1_entity_id to list of true match IDs.

    Returns:
        Dictionary containing:
        - macro_f05: float
        - macro_precision: float
        - macro_recall: float
        - singleton_accuracy: float
        - multi_match_f05: float

    TODO:
        Iterate over all ground truth S1 entities, evaluate per-entity metrics, and average.
    """
    raise NotImplementedError("Macro F0.5 calculation is not implemented yet.")
