"""
Model Inference and Probability Prediction Module.

Responsible for:
- Loading serialized LightGBM model.
- Predicting pairwise match probabilities for test candidate pairs.
- Emitting standard prediction tuples: (s1_entity_id, candidate_entity_id, match_probability).

Ownership: Member 2 (Features & Model).
"""

import logging
from pathlib import Path
from typing import Any, Union
import pandas as pd
import lightgbm as lgb

from .train import extract_model_features

logger = logging.getLogger(__name__)


def _load_model(model: Union[lgb.Booster, str, Path]) -> lgb.Booster:
    """
    Load LightGBM model from Booster object or file path.
    
    Args:
        model: LightGBM Booster object or path to model file.
        
    Returns:
        Loaded LightGBM Booster object.
    """
    if isinstance(model, lgb.Booster):
        return model
    if isinstance(model, (str, Path)):
        path = Path(model)
        if not path.exists():
            raise TypeError(
                f"model must be a LightGBM Booster object or a path to an existing model file, "
                f"got non-existent path: {path}"
            )
        return lgb.Booster(model_file=str(path))
    raise TypeError(
        f"model must be a LightGBM Booster object or a path to a model file, "
        f"got {type(model).__name__}"
    )


def predict_match_probabilities(
    model: Any,
    features_df: pd.DataFrame,
    batch_size: int = 100_000,
) -> pd.DataFrame:
    """
    Predict match probabilities for candidate pairs.

    Interface Contract:
        Returns DataFrame with columns:
        - s1_entity_id
        - candidate_entity_id
        - match_probability (float between 0.0 and 1.0)

    Args:
        model: Trained LightGBM Booster or path to model file.
        features_df: Feature DataFrame containing identifier columns and model features.
        batch_size: Inference batch size for streaming execution.

    Returns:
        DataFrame with ['s1_entity_id', 'candidate_entity_id', 'match_probability'].
    """
    # Validate batch_size
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    # Validate required identifier columns
    required_id_cols = ["s1_entity_id", "candidate_entity_id"]
    for col in required_id_cols:
        if col not in features_df.columns:
            raise ValueError(f"Missing required identifier column: {col}")

    # Handle empty input
    if len(features_df) == 0:
        return pd.DataFrame(
            columns=["s1_entity_id", "candidate_entity_id", "match_probability"]
        )

    # Load model if path provided
    booster = _load_model(model)

    # Extract the 18 model features using existing function (validates columns)
    model_features = extract_model_features(features_df)

    # Preserve identifier columns in original order
    s1_ids = features_df["s1_entity_id"]
    cand_ids = features_df["candidate_entity_id"]

    # Predict in batches to avoid memory issues
    n_rows = len(model_features)
    probabilities = []

    for start_idx in range(0, n_rows, batch_size):
        end_idx = min(start_idx + batch_size, n_rows)
        batch = model_features.iloc[start_idx:end_idx]
        batch_probs = booster.predict(batch, num_iteration=booster.best_iteration)
        probabilities.extend(batch_probs)

    # Ensure probabilities are in [0, 1] and convert to float
    probabilities = [float(max(0.0, min(1.0, p))) for p in probabilities]

    # Return DataFrame with exact column order, preserving input row order
    return pd.DataFrame({
        "s1_entity_id": s1_ids.values,
        "candidate_entity_id": cand_ids.values,
        "match_probability": probabilities,
    })
