"""
Tests for predict_match_probabilities function.
"""

import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import pytest

from src.business_entity_resolution.model.train import (
    MODEL_FEATURES,
    train_lightgbm_model,
)
from src.business_entity_resolution.model.predict import predict_match_probabilities


def _create_dummy_model(tmp_path: Path) -> lgb.Booster:
    """Create a minimal trained model for testing."""
    n_samples = 200
    np.random.seed(42)
    
    # Create dummy feature matrix with 18 features
    X = pd.DataFrame(
        np.random.rand(n_samples, len(MODEL_FEATURES)),
        columns=MODEL_FEATURES,
    )
    y = pd.Series(np.random.randint(0, 2, n_samples))
    
    model = train_lightgbm_model(
        train_features_df=X,
        labels=y,
        params={"n_estimators": 10, "verbose": -1},
    )
    
    model_path = tmp_path / "test_model.txt"
    model.save_model(str(model_path))
    return model, model_path


def _create_features_df(n_rows: int, add_extra_cols: bool = False) -> pd.DataFrame:
    """Create a features DataFrame with required columns."""
    np.random.seed(123)
    data = {
        "s1_entity_id": [f"S1-{i:05d}" for i in range(n_rows)],
        "candidate_entity_id": [f"S2-{i:05d}" for i in range(n_rows)],
    }
    for feat in MODEL_FEATURES:
        data[feat] = np.random.rand(n_rows)
    
    df = pd.DataFrame(data)
    
    if add_extra_cols:
        df["extra_col"] = "extra"
    
    return df


class TestPredictMatchProbabilities:
    """Tests for predict_match_probabilities function."""

    def test_normal_prediction(self):
        """Test normal prediction with a loaded Booster."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model, _ = _create_dummy_model(tmp_path)
            features_df = _create_features_df(100)
            
            result = predict_match_probabilities(model, features_df, batch_size=50)
            
            assert len(result) == 100
            assert list(result.columns) == ["s1_entity_id", "candidate_entity_id", "match_probability"]
            assert result["s1_entity_id"].tolist() == features_df["s1_entity_id"].tolist()
            assert result["candidate_entity_id"].tolist() == features_df["candidate_entity_id"].tolist()
            assert (result["match_probability"] >= 0.0).all()
            assert (result["match_probability"] <= 1.0).all()

    def test_prediction_with_model_path(self):
        """Test prediction with model file path instead of Booster."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _, model_path = _create_dummy_model(tmp_path)
            features_df = _create_features_df(50)
            
            result = predict_match_probabilities(model_path, features_df, batch_size=25)
            
            assert len(result) == 50
            assert list(result.columns) == ["s1_entity_id", "candidate_entity_id", "match_probability"]
            assert (result["match_probability"] >= 0.0).all()
            assert (result["match_probability"] <= 1.0).all()

    def test_batching(self):
        """Test that batching works correctly with different batch sizes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model, _ = _create_dummy_model(tmp_path)
            features_df = _create_features_df(250)
            
            # Test with batch_size smaller than dataset
            result1 = predict_match_probabilities(model, features_df, batch_size=100)
            result2 = predict_match_probabilities(model, features_df, batch_size=50)
            result3 = predict_match_probabilities(model, features_df, batch_size=1000)
            
            # All should produce same results
            pd.testing.assert_series_equal(
                result1["match_probability"], result2["match_probability"]
            )
            pd.testing.assert_series_equal(
                result1["match_probability"], result3["match_probability"]
            )

    def test_empty_input(self):
        """Test empty input returns correct empty DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model, _ = _create_dummy_model(tmp_path)
            
            empty_df = pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id"] + MODEL_FEATURES)
            result = predict_match_probabilities(model, empty_df, batch_size=100)
            
            assert len(result) == 0
            assert list(result.columns) == ["s1_entity_id", "candidate_entity_id", "match_probability"]

    def test_invalid_batch_size_zero(self):
        """Test that batch_size <= 0 raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model, _ = _create_dummy_model(tmp_path)
            features_df = _create_features_df(10)
            
            with pytest.raises(ValueError, match="batch_size must be positive"):
                predict_match_probabilities(model, features_df, batch_size=0)
            
            with pytest.raises(ValueError, match="batch_size must be positive"):
                predict_match_probabilities(model, features_df, batch_size=-1)

    def test_missing_s1_entity_id(self):
        """Test missing s1_entity_id raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model, _ = _create_dummy_model(tmp_path)
            features_df = _create_features_df(10)
            features_df = features_df.drop(columns=["s1_entity_id"])
            
            with pytest.raises(ValueError, match="Missing required identifier column: s1_entity_id"):
                predict_match_probabilities(model, features_df)

    def test_missing_candidate_entity_id(self):
        """Test missing candidate_entity_id raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model, _ = _create_dummy_model(tmp_path)
            features_df = _create_features_df(10)
            features_df = features_df.drop(columns=["candidate_entity_id"])
            
            with pytest.raises(ValueError, match="Missing required identifier column: candidate_entity_id"):
                predict_match_probabilities(model, features_df)

    def test_missing_model_features(self):
        """Test missing model features raises ValueError from extract_model_features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model, _ = _create_dummy_model(tmp_path)
            features_df = _create_features_df(10)
            features_df = features_df.drop(columns=[MODEL_FEATURES[0]])
            
            with pytest.raises(ValueError, match="Missing required model features"):
                predict_match_probabilities(model, features_df)

    def test_probability_range(self):
        """Test all probabilities are in [0.0, 1.0]."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model, _ = _create_dummy_model(tmp_path)
            features_df = _create_features_df(1000)
            
            result = predict_match_probabilities(model, features_df, batch_size=100)
            
            assert (result["match_probability"] >= 0.0).all()
            assert (result["match_probability"] <= 1.0).all()

    def test_row_order_preservation(self):
        """Test that input row order is preserved in output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model, _ = _create_dummy_model(tmp_path)
            features_df = _create_features_df(50)
            
            # Shuffle the input
            shuffled = features_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
            result = predict_match_probabilities(model, shuffled, batch_size=25)
            
            # Order should match shuffled input
            assert result["s1_entity_id"].tolist() == shuffled["s1_entity_id"].tolist()
            assert result["candidate_entity_id"].tolist() == shuffled["candidate_entity_id"].tolist()

    def test_extra_columns_raises(self):
        """Test that extra columns in input raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model, _ = _create_dummy_model(tmp_path)
            features_df = _create_features_df(50, add_extra_cols=True)
            
            with pytest.raises(ValueError, match="Unexpected columns"):
                predict_match_probabilities(model, features_df, batch_size=25)

    def test_invalid_model_type(self):
        """Test that invalid model type raises TypeError."""
        features_df = _create_features_df(10)
        
        with pytest.raises(TypeError, match="model must be a LightGBM Booster object or a path"):
            predict_match_probabilities("not_a_model_or_path", features_df)
        
        with pytest.raises(TypeError, match="model must be a LightGBM Booster object or a path"):
            predict_match_probabilities(123, features_df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])