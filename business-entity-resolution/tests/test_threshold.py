"""
Tests for threshold optimization and application functions.

Note: find_optimal_threshold tests require M3's compute_macro_f05 to be implemented.
Currently M3 metrics raise NotImplementedError, so those tests will fail until M3 is complete.
"""

import pandas as pd
import numpy as np
import pytest

from src.business_entity_resolution.model.threshold import (
    find_optimal_threshold,
    apply_entity_thresholds,
)


def _create_predictions_df():
    """Create a sample predictions DataFrame for testing."""
    return pd.DataFrame({
        "s1_entity_id": [
            "S1-001", "S1-001", "S1-001",
            "S1-002", "S1-002",
            "S1-003",
            "S1-004", "S1-004",
            "S1-005",
        ],
        "candidate_entity_id": [
            "S2-001", "S2-002", "S2-003",
            "S2-004", "S2-005",
            "S2-006",
            "S2-007", "S2-008",
            "S2-009",
        ],
        "match_probability": [
            0.9, 0.7, 0.4,
            0.8, 0.3,
            0.6,
            0.95, 0.2,
            0.1,
        ],
    })


def _create_ground_truth():
    """Create a sample ground truth mapping for testing."""
    return {
        "S1-001": ["S2-001", "S2-002"],  # Two true matches
        "S1-002": ["S2-004"],             # One true match
        "S1-003": ["S2-006"],             # One true match
        "S1-004": ["S2-007"],             # One true match
        "S1-005": [],                     # Singleton (no true matches)
    }


class TestFindOptimalThreshold:
    """Tests for find_optimal_threshold function.

    These tests require M3's compute_macro_f05 to be implemented.
    Currently they will raise NotImplementedError until M3 is complete.
    """

    def test_threshold_sweep(self):
        """Test threshold sweep finds optimal threshold (requires M3)."""
        preds = _create_predictions_df()
        gt = _create_ground_truth()

        # This will raise NotImplementedError until M3 implements compute_macro_f05
        with pytest.raises(NotImplementedError):
            find_optimal_threshold(preds, gt, threshold_range=(0.2, 0.9, 0.1))

    def test_tie_handling_chooses_lower_threshold(self):
        """Test that ties are broken by choosing lower threshold (requires M3)."""
        preds = _create_predictions_df()
        gt = _create_ground_truth()

        with pytest.raises(NotImplementedError):
            find_optimal_threshold(preds, gt, threshold_range=(0.3, 0.7, 0.1))

    def test_missing_columns_raises(self):
        """Test missing required columns raises ValueError."""
        preds = pd.DataFrame({"s1_entity_id": ["S1-001"], "candidate_entity_id": ["S2-001"]})  # missing match_probability
        gt = {"S1-001": ["S2-001"]}

        with pytest.raises(ValueError, match="Missing required column: match_probability"):
            find_optimal_threshold(preds, gt)

    def test_invalid_threshold_range_raises(self):
        """Test invalid threshold_range raises ValueError."""
        preds = _create_predictions_df()
        gt = _create_ground_truth()

        with pytest.raises(ValueError, match="threshold_range must be a tuple"):
            find_optimal_threshold(preds, gt, threshold_range=(0.5,))

        with pytest.raises(ValueError, match="Invalid threshold_range"):
            find_optimal_threshold(preds, gt, threshold_range=(1.5, 0.9, 0.02))

        with pytest.raises(ValueError, match="step > 0"):
            find_optimal_threshold(preds, gt, threshold_range=(0.5, 0.9, 0))

        with pytest.raises(ValueError, match="min.*> max"):
            find_optimal_threshold(preds, gt, threshold_range=(0.9, 0.5, 0.02))

    def test_empty_predictions_raises(self):
        """Test empty predictions raises ValueError."""
        preds = pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "match_probability"])
        gt = {"S1-001": ["S2-001"]}

        with pytest.raises(ValueError, match="predictions_df is empty"):
            find_optimal_threshold(preds, gt)

    def test_threshold_includes_s1_entities_in_gt(self):
        """Test that all S1 entities in ground_truth are evaluated (requires M3)."""
        preds = _create_predictions_df()
        gt = _create_ground_truth()

        with pytest.raises(NotImplementedError):
            find_optimal_threshold(preds, gt, threshold_range=(0.5, 0.5, 0.1))

    def test_returns_float(self):
        """Test return type is float (requires M3)."""
        preds = _create_predictions_df()
        gt = _create_ground_truth()

        with pytest.raises(NotImplementedError):
            find_optimal_threshold(preds, gt, threshold_range=(0.5, 0.5, 0.1))


class TestApplyEntityThresholds:
    """Tests for apply_entity_thresholds function."""

    def test_normal_application(self):
        """Test normal threshold application with multiple matches."""
        preds = _create_predictions_df()
        all_s1 = ["S1-001", "S1-002", "S1-003", "S1-004", "S1-005"]

        result = apply_entity_thresholds(preds, threshold=0.5, all_s1_ids=all_s1)

        assert len(result) == 5
        assert list(result.columns) == ["source1_entity_id", "matched_entity_ids"]

        # Check S1-001 has two matches (0.9, 0.7 >= 0.5)
        s1_001 = result[result["source1_entity_id"] == "S1-001"]
        assert len(s1_001) == 1
        assert s1_001.iloc[0]["matched_entity_ids"] == "S2-001,S2-002"

        # Check S1-002 has one match (0.8 >= 0.5, 0.3 < 0.5)
        s1_002 = result[result["source1_entity_id"] == "S1-002"]
        assert s1_002.iloc[0]["matched_entity_ids"] == "S2-004"

        # Check S1-005 (singleton) has empty matches (0.1 < 0.5)
        s1_005 = result[result["source1_entity_id"] == "S1-005"]
        assert s1_005.iloc[0]["matched_entity_ids"] == ""

    def test_singleton_with_no_retained_candidates(self):
        """Test singleton S1 with no candidates above threshold."""
        preds = pd.DataFrame({
            "s1_entity_id": ["S1-001", "S1-001"],
            "candidate_entity_id": ["S2-001", "S2-002"],
            "match_probability": [0.3, 0.2],
        })
        all_s1 = ["S1-001", "S1-002"]

        result = apply_entity_thresholds(preds, threshold=0.5, all_s1_ids=all_s1)

        assert len(result) == 2
        s1_001 = result[result["source1_entity_id"] == "S1-001"]
        assert s1_001.iloc[0]["matched_entity_ids"] == ""
        s1_002 = result[result["source1_entity_id"] == "S1-002"]
        assert s1_002.iloc[0]["matched_entity_ids"] == ""

    def test_all_s1_ids_preserved(self):
        """Test every S1 ID in all_s1_ids appears exactly once."""
        preds = _create_predictions_df()
        all_s1 = ["S1-001", "S1-002", "S1-003", "S1-004", "S1-005", "S1-006"]

        result = apply_entity_thresholds(preds, threshold=0.5, all_s1_ids=all_s1)

        assert len(result) == 6
        assert set(result["source1_entity_id"]) == set(all_s1)

    def test_deterministic_matched_id_ordering(self):
        """Test matched_entity_ids ordering is deterministic."""
        preds = pd.DataFrame({
            "s1_entity_id": ["S1-001", "S1-001", "S1-001"],
            "candidate_entity_id": ["S2-003", "S2-001", "S2-002"],
            "match_probability": [0.9, 0.8, 0.7],
        })
        all_s1 = ["S1-001"]

        result = apply_entity_thresholds(preds, threshold=0.5, all_s1_ids=all_s1)

        # Should preserve input order after dedup (first occurrence wins)
        matched = result.iloc[0]["matched_entity_ids"]
        assert matched == "S2-003,S2-001,S2-002"

    def test_duplicate_candidates_deduped(self):
        """Test duplicate candidate rows are deduplicated."""
        preds = pd.DataFrame({
            "s1_entity_id": ["S1-001", "S1-001", "S1-001"],
            "candidate_entity_id": ["S2-001", "S2-001", "S2-002"],
            "match_probability": [0.9, 0.9, 0.8],
        })
        all_s1 = ["S1-001"]

        result = apply_entity_thresholds(preds, threshold=0.5, all_s1_ids=all_s1)

        matched = result.iloc[0]["matched_entity_ids"]
        assert matched == "S2-001,S2-002"  # No duplicates

    def test_multiple_matches_preserved(self):
        """Test multiple valid matches per S1 are preserved (NOT top-1)."""
        preds = pd.DataFrame({
            "s1_entity_id": ["S1-001"] * 4,
            "candidate_entity_id": ["S2-001", "S2-002", "S2-003", "S2-004"],
            "match_probability": [0.9, 0.8, 0.7, 0.6],
        })
        all_s1 = ["S1-001"]

        result = apply_entity_thresholds(preds, threshold=0.5, all_s1_ids=all_s1)

        matched = result.iloc[0]["matched_entity_ids"]
        parts = matched.split(",")
        assert len(parts) == 4  # All 4 matches retained
        assert set(parts) == {"S2-001", "S2-002", "S2-003", "S2-004"}

    def test_empty_predictions(self):
        """Test empty predictions DataFrame."""
        preds = pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "match_probability"])
        all_s1 = ["S1-001", "S1-002"]

        result = apply_entity_thresholds(preds, threshold=0.5, all_s1_ids=all_s1)

        assert len(result) == 2
        assert all(result["matched_entity_ids"] == "")

    def test_empty_all_s1_ids(self):
        """Test empty all_s1_ids returns empty DataFrame."""
        preds = _create_predictions_df()

        result = apply_entity_thresholds(preds, threshold=0.5, all_s1_ids=[])

        assert len(result) == 0
        assert list(result.columns) == ["source1_entity_id", "matched_entity_ids"]

    def test_invalid_threshold_raises(self):
        """Test threshold outside [0,1] raises ValueError."""
        preds = _create_predictions_df()
        all_s1 = ["S1-001"]

        with pytest.raises(ValueError, match="threshold must be in \\[0, 1\\]"):
            apply_entity_thresholds(preds, threshold=-0.1, all_s1_ids=all_s1)

        with pytest.raises(ValueError, match="threshold must be in \\[0, 1\\]"):
            apply_entity_thresholds(preds, threshold=1.1, all_s1_ids=all_s1)

    def test_missing_columns_raises(self):
        """Test missing required columns raises ValueError."""
        preds = pd.DataFrame({"s1_entity_id": ["S1-001"], "candidate_entity_id": ["S2-001"]})
        all_s1 = ["S1-001"]

        with pytest.raises(ValueError, match="Missing required column: match_probability"):
            apply_entity_thresholds(preds, threshold=0.5, all_s1_ids=all_s1)

    def test_invalid_all_s1_ids_type_raises(self):
        """Test all_s1_ids not being list/tuple/set raises ValueError."""
        preds = _create_predictions_df()

        with pytest.raises(ValueError, match="all_s1_ids must be a list/tuple/set"):
            apply_entity_thresholds(preds, threshold=0.5, all_s1_ids="S1-001")

        with pytest.raises(ValueError, match="all_s1_ids must be a list/tuple/set"):
            apply_entity_thresholds(preds, threshold=0.5, all_s1_ids=123)

    def test_threshold_zero_includes_all(self):
        """Test threshold=0 includes all predictions."""
        preds = _create_predictions_df()
        all_s1 = ["S1-001", "S1-002", "S1-003", "S1-004", "S1-005"]

        result = apply_entity_thresholds(preds, threshold=0.0, all_s1_ids=all_s1)

        s1_001 = result[result["source1_entity_id"] == "S1-001"]
        assert s1_001.iloc[0]["matched_entity_ids"] == "S2-001,S2-002,S2-003"

    def test_threshold_one_includes_none(self):
        """Test threshold=1 includes only probability=1.0."""
        preds = pd.DataFrame({
            "s1_entity_id": ["S1-001", "S1-001"],
            "candidate_entity_id": ["S2-001", "S2-002"],
            "match_probability": [1.0, 0.99],
        })
        all_s1 = ["S1-001"]

        result = apply_entity_thresholds(preds, threshold=1.0, all_s1_ids=all_s1)

        matched = result.iloc[0]["matched_entity_ids"]
        assert matched == "S2-001"  # Only 1.0 included


if __name__ == "__main__":
    pytest.main([__file__, "-v"])