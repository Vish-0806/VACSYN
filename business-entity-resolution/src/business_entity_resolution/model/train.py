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
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import lightgbm as lgb

logger = logging.getLogger(__name__)

# Exact 18 model feature names expected by the model
MODEL_FEATURES = [
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

# Identity columns that should NOT be passed to the model
IDENTITY_COLUMNS = ["s1_entity_id", "candidate_entity_id", "candidate_source"]

# Default LightGBM parameters
DEFAULT_LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
    "force_col_wise": True,
}


def prepare_ground_truth_mapping(ground_truth_path: str) -> Dict[str, List[str]]:
    """
    Load ground truth mapping from TSV file.

    Args:
        ground_truth_path: Path to train_ground_truth.tsv file.

    Returns:
        Dictionary mapping s1_entity_id to list of matched S2/S3 entity IDs.
    """
    mapping = {}
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        # Skip header
        f.readline()
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            s1_id, _, rest = line.partition("\t")
            if rest:
                matched = [m.strip() for m in rest.split(",") if m.strip()]
                if matched:
                    mapping[s1_id] = matched
    return mapping


def assign_labels_to_candidates(
    candidate_pairs_df: pd.DataFrame,
    ground_truth_mapping: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Assign binary labels to existing candidate pairs using ground truth.

    For each candidate pair (s1_entity_id, candidate_entity_id), assign:
    - label = 1 if candidate_entity_id is in ground_truth[s1_entity_id]
    - label = 0 otherwise

    Only labels EXISTING candidate pairs. Does NOT generate new pairs.

    Args:
        candidate_pairs_df: DataFrame with columns ['s1_entity_id', 'candidate_entity_id', 'candidate_source'].
        ground_truth_mapping: Dict mapping s1_entity_id to list of true matching S2/S3 IDs.

    Returns:
        DataFrame with original columns plus 'label' column (0 or 1).
    """
    # Validate required columns
    required_cols = ["s1_entity_id", "candidate_entity_id", "candidate_source"]
    for col in required_cols:
        if col not in candidate_pairs_df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Validate candidate_source values
    invalid_sources = set(candidate_pairs_df["candidate_source"].unique()) - {"S2", "S3"}
    if invalid_sources:
        raise ValueError(f"Invalid candidate_source values: {invalid_sources}. Expected only 'S2' or 'S3'.")

    # Create a copy to avoid mutating the original
    result = candidate_pairs_df.copy()

    # Assign labels
    def get_label(row) -> int:
        s1_id = row["s1_entity_id"]
        cand_id = row["candidate_entity_id"]
        true_matches = ground_truth_mapping.get(s1_id, [])
        return 1 if cand_id in true_matches else 0

    result["label"] = result.apply(get_label, axis=1)

    return result


def entity_level_split(
    candidate_pairs_df: pd.DataFrame,
    val_ratio: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split candidate pairs into train and validation at the entity level.

    All candidate pairs for a given S1 entity go entirely to train OR validation.
    No S1 entity appears in both splits.

    Args:
        candidate_pairs_df: DataFrame with candidate pairs including 's1_entity_id'.
        val_ratio: Proportion of S1 entities to allocate to validation.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df) DataFrames.
    """
    if "s1_entity_id" not in candidate_pairs_df.columns:
        raise ValueError("Missing required column: s1_entity_id")

    # Get unique S1 entity IDs
    s1_ids = candidate_pairs_df["s1_entity_id"].unique()

    # Deterministic shuffle
    import numpy as np
    rng = np.random.RandomState(random_state)
    shuffled_ids = rng.permutation(s1_ids)

    # Split
    n_val = int(len(shuffled_ids) * val_ratio)
    val_s1_ids = set(shuffled_ids[:n_val])
    train_s1_ids = set(shuffled_ids[n_val:])

    # Split candidate pairs
    train_df = candidate_pairs_df[candidate_pairs_df["s1_entity_id"].isin(train_s1_ids)].copy()
    val_df = candidate_pairs_df[candidate_pairs_df["s1_entity_id"].isin(val_s1_ids)].copy()

    return train_df, val_df


def _validate_features_df(df: pd.DataFrame, name: str) -> None:
    """Validate that a feature DataFrame has the correct schema."""
    missing_features = set(MODEL_FEATURES) - set(df.columns)
    if missing_features:
        raise ValueError(f"{name}: Missing required model features: {sorted(missing_features)}")

    extra_features = set(df.columns) - set(MODEL_FEATURES) - set(IDENTITY_COLUMNS) - {"label"}
    if extra_features:
        raise ValueError(f"{name}: Unexpected columns (not in model features or identity): {sorted(extra_features)}")

    # Check for NaN/Inf in model features
    for col in MODEL_FEATURES:
        if col in df.columns:
            if df[col].isna().any():
                raise ValueError(f"{name}: NaN detected in feature column '{col}'")
            if (df[col] == float('inf')).any() or (df[col] == float('-inf')).any():
                raise ValueError(f"{name}: Inf detected in feature column '{col}'")


def _validate_labels(labels: pd.Series, name: str) -> None:
    """Validate that labels are binary 0/1."""
    if not labels.isin([0, 1]).all():
        raise ValueError(f"{name}: Labels must be binary (0 or 1). Found other values.")


def train_lightgbm_model(
    train_features_df: pd.DataFrame,
    labels: pd.Series,
    val_features_df: Optional[pd.DataFrame] = None,
    val_labels: Optional[pd.Series] = None,
    params: Optional[Dict[str, Any]] = None,
    model_save_path: Optional[Path] = None,
) -> lgb.Booster:
    """
    Train LightGBM binary classification model on candidate pair features.

    Args:
        train_features_df: Feature matrix for training pairs (18 model features only).
        labels: Binary labels (1 = true match, 0 = non-match) for training pairs.
        val_features_df: Optional validation feature matrix (18 model features only).
        val_labels: Optional validation labels.
        params: LightGBM hyperparameter overrides.
        model_save_path: Path to write serialized model artifact.

    Returns:
        Trained LightGBM Booster object.
    """
    # Validate inputs
    _validate_features_df(train_features_df, "train_features_df")
    _validate_labels(labels, "labels")

    if len(train_features_df) != len(labels):
        raise ValueError(
            f"train_features_df length ({len(train_features_df)}) "
            f"does not match labels length ({len(labels)})"
        )

    # Merge with validation data if provided
    if val_features_df is not None:
        _validate_features_df(val_features_df, "val_features_df")
        _validate_labels(val_labels, "val_labels")
        if len(val_features_df) != len(val_labels):
            raise ValueError(
                f"val_features_df length ({len(val_features_df)}) "
                f"does not match val_labels length ({len(val_labels)})"
            )

    # Prepare LightGBM datasets
    train_data = lgb.Dataset(
        train_features_df[MODEL_FEATURES],
        label=labels,
        free_raw_data=False,
    )

    val_data = None
    if val_features_df is not None and val_labels is not None:
        val_data = lgb.Dataset(
            val_features_df[MODEL_FEATURES],
            label=val_labels,
            reference=train_data,
            free_raw_data=False,
        )

    # Merge default params with user overrides
    lgbm_params = DEFAULT_LGBM_PARAMS.copy()
    if params:
        lgbm_params.update(params)

    # Train model
    callbacks = [lgb.early_stopping(stopping_rounds=50)] if val_data is not None else []

    logger.info("Training LightGBM model with %d training samples...", len(train_features_df))
    model = lgb.train(
        lgbm_params,
        train_data,
        valid_sets=[val_data] if val_data is not None else None,
        valid_names=["valid"] if val_data is not None else None,
        callbacks=callbacks,
    )

    # Save model if path provided
    if model_save_path is not None:
        model_save_path = Path(model_save_path)
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_save_path))
        logger.info("Model saved to %s", model_save_path)

    logger.info("Training completed. Best iteration: %d", model.best_iteration)
    return model


def extract_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only the 18 model features from a feature DataFrame.

    Drops identity columns and label if present.

    Args:
        df: Feature DataFrame with identity columns and/or label.

    Returns:
        DataFrame with only the 18 model feature columns.
    """
    missing = set(MODEL_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required model features: {sorted(missing)}")

    extra = set(df.columns) - set(MODEL_FEATURES) - set(IDENTITY_COLUMNS) - {"label"}
    if extra:
        raise ValueError(f"Unexpected columns (not in model features or identity): {sorted(extra)}")

    return df[MODEL_FEATURES].copy()