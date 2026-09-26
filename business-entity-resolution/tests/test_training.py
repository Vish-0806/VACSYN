"""
Tests for M2 Phase 2: Training label preparation, entity-level splitting, and LightGBM training.
"""

import tempfile
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.business_entity_resolution.model.train import (
    prepare_ground_truth_mapping,
    assign_labels_to_candidates,
    entity_level_split,
    train_lightgbm_model,
    extract_model_features,
    MODEL_FEATURES,
    IDENTITY_COLUMNS,
)
from src.business_entity_resolution.features.pair_features import (
    PairFeatureExtractor,
    build_pair_features,
)


class TestGroundTruthMapping:
    """Tests for ground truth loading."""

    def test_parse_ground_truth_file(self):
        """Test parsing ground truth TSV format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("S1-001\tS2-001,S3-001\n")
            f.write("S1-002\tS2-002\n")
            f.write("S1-003\t\n")
            f.write("S1-004\tS3-003,S3-004,S3-005\n")
            temp_path = f.name

        try:
            mapping = prepare_ground_truth_mapping(temp_path)
            assert mapping == {
                "S1-001": ["S2-001", "S3-001"],
                "S1-002": ["S2-002"],
                "S1-004": ["S3-003", "S3-004", "S3-005"],
            }
            # S1-003 has no matches, should not be in mapping
            assert "S1-003" not in mapping
        finally:
            Path(temp_path).unlink()

    def test_empty_file(self):
        """Test empty ground truth file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            temp_path = f.name

        try:
            mapping = prepare_ground_truth_mapping(temp_path)
            assert mapping == {}
        finally:
            Path(temp_path).unlink()


class TestLabelAssignment:
    """Tests for label assignment to candidate pairs."""

    def setup_method(self):
        """Set up test data."""
        self.ground_truth = {
            "S1-001": ["S2-001", "S3-001"],
            "S1-002": ["S2-002"],
            "S1-003": ["S3-003", "S3-004"],
        }

        self.candidate_pairs = pd.DataFrame([
            {"s1_entity_id": "S1-001", "candidate_entity_id": "S2-001", "candidate_source": "S2"},
            {"s1_entity_id": "S1-001", "candidate_entity_id": "S3-001", "candidate_source": "S3"},
            {"s1_entity_id": "S1-001", "candidate_entity_id": "S2-002", "candidate_source": "S2"},
            {"s1_entity_id": "S1-002", "candidate_entity_id": "S2-002", "candidate_source": "S2"},
            {"s1_entity_id": "S1-002", "candidate_entity_id": "S3-002", "candidate_source": "S3"},
            {"s1_entity_id": "S1-003", "candidate_entity_id": "S3-003", "candidate_source": "S3"},
            {"s1_entity_id": "S1-003", "candidate_entity_id": "S3-004", "candidate_source": "S3"},
            {"s1_entity_id": "S1-003", "candidate_entity_id": "S3-005", "candidate_source": "S3"},
            {"s1_entity_id": "S1-004", "candidate_entity_id": "S2-003", "candidate_source": "S2"},
        ])

    def test_positive_candidates_receive_label_1(self):
        """True matches should get label 1."""
        result = assign_labels_to_candidates(self.candidate_pairs, self.ground_truth)

        # Check positive matches
        pos = result[result["label"] == 1]
        assert len(pos) == 5  # 2 + 1 + 2 matches

        # Verify specific matches
        match = result[
            (result["s1_entity_id"] == "S1-001") &
            (result["candidate_entity_id"] == "S2-001")
        ]
        assert match["label"].values[0] == 1

    def test_non_ground_truth_candidates_receive_label_0(self):
        """Non-matches should get label 0."""
        result = assign_labels_to_candidates(self.candidate_pairs, self.ground_truth)

        neg = result[result["label"] == 0]
        assert len(neg) == 4  # 1 + 1 + 2 non-matches

        # Verify specific non-matches
        non_match = result[
            (result["s1_entity_id"] == "S1-001") &
            (result["candidate_entity_id"] == "S2-002")
        ]
        assert non_match["label"].values[0] == 0

    def test_only_existing_candidates_labeled(self):
        """Only existing candidate pairs are labeled; no new pairs generated."""
        original_len = len(self.candidate_pairs)
        result = assign_labels_to_candidates(self.candidate_pairs, self.ground_truth)
        assert len(result) == original_len

    def test_multiple_s1_entities(self):
        """Multiple S1 entities work correctly."""
        result = assign_labels_to_candidates(self.candidate_pairs, self.ground_truth)
        s1_ids = result["s1_entity_id"].unique()
        assert set(s1_ids) == {"S1-001", "S1-002", "S1-003", "S1-004"}

    def test_multiple_s2_s3_matches(self):
        """Multiple S2/S3 matches for same S1 work correctly."""
        result = assign_labels_to_candidates(self.candidate_pairs, self.ground_truth)
        s1_001_matches = result[
            (result["s1_entity_id"] == "S1-001") &
            (result["label"] == 1)
        ]["candidate_entity_id"].tolist()
        assert set(s1_001_matches) == {"S2-001", "S3-001"}

    def test_s1_with_no_ground_truth_gets_all_zeros(self):
        """S1 with no ground truth gets all label 0."""
        result = assign_labels_to_candidates(self.candidate_pairs, self.ground_truth)
        s1_004 = result[result["s1_entity_id"] == "S1-004"]
        assert (s1_004["label"] == 0).all()

    def test_preserves_original_columns(self):
        """Original columns preserved plus label."""
        result = assign_labels_to_candidates(self.candidate_pairs, self.ground_truth)
        expected_cols = ["s1_entity_id", "candidate_entity_id", "candidate_source", "label"]
        assert list(result.columns) == expected_cols

    def test_invalid_candidate_source_raises(self):
        """Invalid candidate_source raises error."""
        bad_pairs = pd.DataFrame([
            {"s1_entity_id": "S1-001", "candidate_entity_id": "S2-001", "candidate_source": "S1"},
        ])
        with pytest.raises(ValueError, match="Invalid candidate_source"):
            assign_labels_to_candidates(bad_pairs, self.ground_truth)

    def test_missing_required_column_raises(self):
        """Missing required column raises error."""
        bad_pairs = pd.DataFrame([
            {"s1_entity_id": "S1-001", "candidate_source": "S2"},
        ])
        with pytest.raises(ValueError, match="Missing required column"):
            assign_labels_to_candidates(bad_pairs, self.ground_truth)


class TestEntityLevelSplit:
    """Tests for entity-level train/validation splitting."""

    def setup_method(self):
        """Create candidate pairs with multiple rows per S1 entity."""
        self.candidate_pairs = pd.DataFrame([
            # S1-001: 3 candidates
            {"s1_entity_id": "S1-001", "candidate_entity_id": "S2-001", "candidate_source": "S2", "label": 1},
            {"s1_entity_id": "S1-001", "candidate_entity_id": "S3-001", "candidate_source": "S3", "label": 1},
            {"s1_entity_id": "S1-001", "candidate_entity_id": "S2-002", "candidate_source": "S2", "label": 0},
            # S1-002: 2 candidates
            {"s1_entity_id": "S1-002", "candidate_entity_id": "S2-003", "candidate_source": "S2", "label": 1},
            {"s1_entity_id": "S1-002", "candidate_entity_id": "S3-002", "candidate_source": "S3", "label": 0},
            # S1-003: 1 candidate
            {"s1_entity_id": "S1-003", "candidate_entity_id": "S2-004", "candidate_source": "S2", "label": 0},
            # S1-004: 2 candidates
            {"s1_entity_id": "S1-004", "candidate_entity_id": "S2-005", "candidate_source": "S2", "label": 0},
            {"s1_entity_id": "S1-004", "candidate_entity_id": "S3-003", "candidate_source": "S3", "label": 1},
        ])

    def test_same_s1_never_in_both_splits(self):
        """Same S1 entity never appears in both train and validation."""
        train_df, val_df = entity_level_split(self.candidate_pairs, val_ratio=0.5, random_state=42)

        train_s1 = set(train_df["s1_entity_id"].unique())
        val_s1 = set(val_df["s1_entity_id"].unique())
        assert train_s1.isdisjoint(val_s1), f"S1 overlap: {train_s1 & val_s1}"

    def test_all_rows_for_s1_stay_together(self):
        """All candidate rows for an S1 stay together in one split."""
        train_df, val_df = entity_level_split(self.candidate_pairs, val_ratio=0.5, random_state=42)

        # Check each S1 entity's rows are all in one split
        for s1_id in self.candidate_pairs["s1_entity_id"].unique():
            s1_rows = self.candidate_pairs[self.candidate_pairs["s1_entity_id"] == s1_id]
            train_count = len(train_df[train_df["s1_entity_id"] == s1_id])
            val_count = len(val_df[val_df["s1_entity_id"] == s1_id])
            assert train_count == len(s1_rows) or val_count == len(s1_rows), \
                f"S1 {s1_id} split across train ({train_count}) and val ({val_count})"

    def test_deterministic_with_random_state(self):
        """Split is deterministic with same random_state."""
        train1, val1 = entity_level_split(self.candidate_pairs, val_ratio=0.25, random_state=42)
        train2, val2 = entity_level_split(self.candidate_pairs, val_ratio=0.25, random_state=42)

        pd.testing.assert_frame_equal(train1.sort_values("s1_entity_id").reset_index(drop=True),
                                      train2.sort_values("s1_entity_id").reset_index(drop=True))
        pd.testing.assert_frame_equal(val1.sort_values("s1_entity_id").reset_index(drop=True),
                                      val2.sort_values("s1_entity_id").reset_index(drop=True))

    def test_different_random_state_gives_different_split(self):
        """Different random_state gives different split."""
        train1, val1 = entity_level_split(self.candidate_pairs, val_ratio=0.25, random_state=42)
        train2, val2 = entity_level_split(self.candidate_pairs, val_ratio=0.25, random_state=123)

        # At least one S1 should be in different split
        train1_s1 = set(train1["s1_entity_id"].unique())
        train2_s1 = set(train2["s1_entity_id"].unique())
        # With 4 S1 entities and 25% val_ratio, splits should differ
        assert train1_s1 != train2_s1 or set(val1["s1_entity_id"].unique()) != set(val2["s1_entity_id"].unique())

    def test_train_plus_val_equals_original(self):
        """Train + validation rows equal original candidate rows."""
        train_df, val_df = entity_level_split(self.candidate_pairs, val_ratio=0.25, random_state=42)
        assert len(train_df) + len(val_df) == len(self.candidate_pairs)

    def test_missing_s1_column_raises(self):
        """Missing s1_entity_id column raises error."""
        bad_df = pd.DataFrame([{"candidate_entity_id": "S2-001", "candidate_source": "S2"}])
        with pytest.raises(ValueError, match="Missing required column: s1_entity_id"):
            entity_level_split(bad_df)


class TestFeatureContract:
    """Tests for feature matrix contract."""

    def setup_method(self):
        """Create sample feature DataFrame."""
        self.sample_features = pd.DataFrame({
            "s1_entity_id": ["S1-001", "S1-002"],
            "candidate_entity_id": ["S2-001", "S2-002"],
            "candidate_source": ["S2", "S2"],
            "label": [1, 0],
            **{f: [0.5, 0.3] for f in MODEL_FEATURES},
        })

    def test_extract_model_features_returns_18(self):
        """extract_model_features returns exactly 18 features."""
        features = extract_model_features(self.sample_features)
        assert list(features.columns) == MODEL_FEATURES
        assert len(features.columns) == 18

    def test_identity_columns_excluded(self):
        """Identity columns are excluded from model features."""
        features = extract_model_features(self.sample_features)
        for col in IDENTITY_COLUMNS:
            assert col not in features.columns

    def test_label_excluded_from_features(self):
        """Label column excluded from model features."""
        features = extract_model_features(self.sample_features)
        assert "label" not in features.columns

    def test_missing_feature_raises(self):
        """Missing feature column raises clear error."""
        bad_df = self.sample_features.drop(columns=["name_token_jaccard"])
        with pytest.raises(ValueError, match="Missing required model features"):
            extract_model_features(bad_df)

    def test_extra_column_raises(self):
        """Unexpected extra column raises clear error."""
        bad_df = self.sample_features.copy()
        bad_df["extra_column"] = [1, 2]
        with pytest.raises(ValueError, match="Unexpected columns"):
            extract_model_features(bad_df)


class TestLightGBMTraining:
    """Tests for LightGBM training."""

    def setup_method(self):
        """Create minimal training data."""
        np.random.seed(42)
        n = 100
        self.train_features = pd.DataFrame({
            f: np.random.rand(n) for f in MODEL_FEATURES
        })
        self.train_labels = pd.Series(np.random.randint(0, 2, n))

        self.val_features = pd.DataFrame({
            f: np.random.rand(20) for f in MODEL_FEATURES
        })
        self.val_labels = pd.Series(np.random.randint(0, 2, 20))

    def test_training_returns_booster(self):
        """Training returns a LightGBM Booster object."""
        model = train_lightgbm_model(
            train_features_df=self.train_features,
            labels=self.train_labels,
            val_features_df=self.val_features,
            val_labels=self.val_labels,
            params={"n_estimators": 10, "verbose": -1},
        )
        assert isinstance(model, lgb.Booster)

    def test_training_without_validation(self):
        """Training works without validation data."""
        model = train_lightgbm_model(
            train_features_df=self.train_features,
            labels=self.train_labels,
            params={"n_estimators": 10, "verbose": -1},
        )
        assert isinstance(model, lgb.Booster)

    def test_training_uses_exactly_18_features(self):
        """Training uses exactly 18 features."""
        model = train_lightgbm_model(
            train_features_df=self.train_features,
            labels=self.train_labels,
            params={"n_estimators": 10, "verbose": -1},
        )
        assert model.num_feature() == 18

    def test_feature_order_preserved(self):
        """Feature order matches MODEL_FEATURES list."""
        model = train_lightgbm_model(
            train_features_df=self.train_features,
            labels=self.train_labels,
            params={"n_estimators": 10, "verbose": -1},
        )
        assert model.feature_name() == MODEL_FEATURES

    def test_model_saving(self):
        """Model can be saved and loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.txt"
            model = train_lightgbm_model(
                train_features_df=self.train_features,
                labels=self.train_labels,
                params={"n_estimators": 10, "verbose": -1},
                model_save_path=model_path,
            )

            assert model_path.exists()
            assert model_path.stat().st_size > 0

            # Verify it can be loaded
            loaded_model = lgb.Booster(model_file=str(model_path))
            assert isinstance(loaded_model, lgb.Booster)
            assert loaded_model.num_feature() == 18

    def test_feature_validation_catches_nan(self):
        """NaN in features raises error."""
        bad_features = self.train_features.copy()
        bad_features.loc[0, "name_token_jaccard"] = np.nan
        with pytest.raises(ValueError, match="NaN detected"):
            train_lightgbm_model(bad_features, self.train_labels)

    def test_feature_validation_catches_inf(self):
        """Inf in features raises error."""
        bad_features = self.train_features.copy()
        bad_features.loc[0, "name_token_jaccard"] = float('inf')
        with pytest.raises(ValueError, match="Inf detected"):
            train_lightgbm_model(bad_features, self.train_labels)

    def test_label_validation_catches_non_binary(self):
        """Non-binary labels raise error."""
        bad_labels = pd.Series([0, 1, 2, 0, 1])
        bad_features = self.train_features.iloc[:5]
        with pytest.raises(ValueError, match="Labels must be binary"):
            train_lightgbm_model(bad_features, bad_labels)

    def test_length_mismatch_raises(self):
        """Feature/label length mismatch raises error."""
        with pytest.raises(ValueError, match="does not match labels length"):
            train_lightgbm_model(self.train_features.iloc[:10], self.train_labels.iloc[:5])

    def test_missing_feature_raises(self):
        """Missing feature column raises error."""
        bad_features = self.train_features.drop(columns=["name_token_jaccard"])
        with pytest.raises(ValueError, match="Missing required model features"):
            train_lightgbm_model(bad_features, self.train_labels)


class TestIntegrationWithFeatureAssembly:
    """Integration tests with feature assembly layer."""

    def test_build_pair_features_then_extract_model_features(self):
        """Full pipeline: candidate pairs -> features -> model features."""
        # Create minimal data
        s1_df = pd.DataFrame([
            {"entity_id": "S1-1", "business_name": "Test Corp", "business_address": "123 Main St", "country": "US"},
            {"entity_id": "S1-2", "business_name": "Acme Inc", "business_address": "456 Oak Ave", "country": "US"},
        ])
        s2_df = pd.DataFrame([
            {"entity_id": "S2-1", "business_name": "Test Corp", "business_address": "123 Main St", "country": "US", "source": "S2"},
            {"entity_id": "S2-2", "business_name": "Acme Corp", "business_address": "456 Oak Ave", "country": "US", "source": "S2"},
        ])
        s3_df = pd.DataFrame([
            {"entity_id": "S3-1", "business_name": "Test Corp", "business_address": "123 Main St", "country": "US", "source": "S3"},
        ])

        pairs = pd.DataFrame([
            {"s1_entity_id": "S1-1", "candidate_entity_id": "S2-1", "candidate_source": "S2"},
            {"s1_entity_id": "S1-1", "candidate_entity_id": "S3-1", "candidate_source": "S3"},
            {"s1_entity_id": "S1-2", "candidate_entity_id": "S2-2", "candidate_source": "S2"},
        ])

        # Build features
        feature_df = build_pair_features(pairs, s1_df, pd.concat([s2_df, s3_df]))

        # Extract model features
        model_features = extract_model_features(feature_df)

        assert list(model_features.columns) == MODEL_FEATURES
        assert len(model_features) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])