"""
Model Training Module.

Responsible for:
- Constructing training datasets by joining candidate pairs with ground truth labels.
- Handling hard negatives generated during blocking.
- Configuring LightGBM binary classification parameters.
- Training the gradient-boosted decision tree model.
- Saving trained model artifacts.

Ownership: Member 2 (Features & Model).
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


def train_lightgbm_model(
    train_features_df: pd.DataFrame,
    labels: pd.Series,
    val_features_df: Optional[pd.DataFrame] = None,
    val_labels: Optional[pd.Series] = None,
    params: Optional[Dict[str, Any]] = None,
    model_save_path: Optional[Path] = None,
) -> Any:
    """
    Train LightGBM binary classification model on candidate pair features.

    Args:
        train_features_df: Feature matrix for training pairs.
        labels: Binary labels (1 = true match, 0 = non-match).
        val_features_df: Optional validation feature matrix.
        val_labels: Optional validation labels.
        params: LightGBM hyperparameter overrides.
        model_save_path: Path to write serialized model artifact.

    Returns:
        Trained LightGBM Booster object.

    TODO:
        - Setup LightGBM Dataset objects.
        - Configure binary objective, balanced sampling or positive weight if needed.
        - Fit model with early stopping on validation split.
        - Persist trained model artifact.
    """
    raise NotImplementedError("LightGBM model training is not implemented yet.")
