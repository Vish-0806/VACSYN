"""
Machine Learning Model Module.

Responsible for LightGBM model training, match probability inference,
and macro F0.5 precision-biased threshold optimization.

Ownership: Member 2 (Features & Model).
"""

from .train import train_lightgbm_model
from .predict import predict_match_probabilities
from .threshold import find_optimal_threshold, apply_entity_thresholds

__all__ = [
    "train_lightgbm_model",
    "predict_match_probabilities",
    "find_optimal_threshold",
    "apply_entity_thresholds",
]
