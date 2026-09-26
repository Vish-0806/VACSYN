"""
Tests for name similarity feature extraction.
"""

import math
import pytest

from src.business_entity_resolution.features.name_features import (
    compute_name_similarity_features,
)


class TestComputeNameSimilarityFeatures:
    def test_exact_match_case_insensitive(self):
        features = compute_name_similarity_features("Apollo Hospital", "apollo hospital")
        assert features["name_exact_match"] == 1.0
        assert features["name_token_jaccard"] == 1.0
        assert features["name_edit_similarity"] == 1.0
        assert features["name_length_ratio"] == 1.0
        assert features["name_char_ngram_cosine"] == 1.0
        assert features["name_first_token_match"] == 1.0

    def test_punctuation_whitespace_variation(self):
        features = compute_name_similarity_features(
            "Apollo Hospital, Ltd.", "Apollo Hospital Ltd"
        )
        # M1 normalizer handles punctuation consistently
        assert features["name_exact_match"] == 1.0
        assert features["name_token_jaccard"] == 1.0

    def test_unicode_preservation(self):
        hindi1 = "अपोलो हॉस्पिटल"
        hindi2 = "अपोलो हॉस्पिटल"
        features = compute_name_similarity_features(hindi1, hindi2)
        assert features["name_exact_match"] == 1.0
        assert features["name_token_jaccard"] == 1.0
        assert features["name_char_ngram_cosine"] == 1.0
        assert features["name_edit_similarity"] == 1.0
        assert features["name_length_ratio"] == 1.0
        assert features["name_first_token_match"] == 1.0

    def test_identical_token_sets_jaccard(self):
        features = compute_name_similarity_features("Apollo Hospital", "Hospital Apollo")
        assert features["name_token_jaccard"] == 1.0
        assert features["name_token_overlap_count"] == 2.0

    def test_disjoint_token_sets_jaccard(self):
        features = compute_name_similarity_features("Apollo Hospital", "Delta Clinic")
        assert features["name_token_jaccard"] == 0.0
        assert features["name_token_overlap_count"] == 0.0

    def test_partial_token_overlap(self):
        features = compute_name_similarity_features(
            "Apollo Hospital Delhi", "Apollo Clinic Mumbai"
        )
        # Tokens: apollo, hospital, delhi vs apollo, clinic, mumbai
        # Intersection: {apollo} = 1, Union: {apollo, hospital, delhi, clinic, mumbai} = 5
        assert features["name_token_jaccard"] == 1.0 / 5.0
        assert features["name_token_overlap_count"] == 1.0

    def test_edit_similarity_identical(self):
        features = compute_name_similarity_features("Apollo", "Apollo")
        assert features["name_edit_similarity"] == 1.0

    def test_edit_similarity_different(self):
        features = compute_name_similarity_features("Apollo", "Delta")
        assert 0.0 <= features["name_edit_similarity"] < 1.0

    def test_first_token_match(self):
        features = compute_name_similarity_features("Apollo Hospital", "Apollo Clinic")
        assert features["name_first_token_match"] == 1.0

    def test_first_token_mismatch(self):
        features = compute_name_similarity_features("Apollo Hospital", "Delta Hospital")
        assert features["name_first_token_match"] == 0.0

    def test_first_token_empty(self):
        features = compute_name_similarity_features("", "Apollo")
        assert features["name_first_token_match"] == 0.0

    def test_length_ratio_identical(self):
        features = compute_name_similarity_features("Apollo", "Apollo")
        assert features["name_length_ratio"] == 1.0

    def test_length_ratio_empty(self):
        features = compute_name_similarity_features("", "Apollo")
        assert features["name_length_ratio"] == 0.0

    def test_both_empty(self):
        features = compute_name_similarity_features("", "")
        assert features["name_exact_match"] == 0.0
        assert features["name_token_jaccard"] == 0.0  # Both empty token sets -> 0.0
        assert features["name_edit_similarity"] == 1.0
        assert features["name_length_ratio"] == 1.0
        assert features["name_char_ngram_cosine"] == 1.0
        assert features["name_first_token_match"] == 0.0
        assert features["name_token_overlap_count"] == 0.0

    def test_none_inputs(self):
        features = compute_name_similarity_features(None, None)
        assert features["name_exact_match"] == 0.0
        assert features["name_token_jaccard"] == 0.0
        assert features["name_edit_similarity"] == 1.0
        assert features["name_length_ratio"] == 1.0
        assert features["name_char_ngram_cosine"] == 1.0
        assert features["name_first_token_match"] == 0.0
        assert features["name_token_overlap_count"] == 0.0

    def test_no_nan_inf(self):
        test_cases = [
            ("Apollo Hospital", "Delta Clinic"),
            ("", ""),
            (None, "Apollo"),
            ("अपोलो", "अपोलो"),
            ("A", "B"),
            ("Apollo Hospital, Ltd.", "Apollo Hospital Ltd"),
        ]
        for name1, name2 in test_cases:
            features = compute_name_similarity_features(name1, name2)
            for key, value in features.items():
                assert not math.isnan(value), f"NaN in {key} for ({name1}, {name2})"
                assert not math.isinf(value), f"Inf in {key} for ({name1}, {name2})"

    def test_bounded_features(self):
        features = compute_name_similarity_features("Apollo Hospital", "Delta Clinic")
        bounded_keys = [
            "name_exact_match",
            "name_token_jaccard",
            "name_char_ngram_cosine",
            "name_edit_similarity",
            "name_first_token_match",
            "name_length_ratio",
        ]
        for key in bounded_keys:
            assert 0.0 <= features[key] <= 1.0, f"{key} out of bounds: {features[key]}"

    def test_char_ngram_cosine(self):
        features = compute_name_similarity_features("Apollo", "Apolo")
        assert 0.0 < features["name_char_ngram_cosine"] < 1.0

    def test_short_strings(self):
        features = compute_name_similarity_features("A", "B")
        assert features["name_edit_similarity"] == 0.0
        assert features["name_length_ratio"] == 1.0
        assert 0.0 <= features["name_char_ngram_cosine"] <= 1.0

    def test_single_char_vs_multi_char(self):
        features = compute_name_similarity_features("A", "Apollo")
        assert features["name_length_ratio"] == 1.0 / 6.0

    def test_indic_script_feature_computation(self):
        """Verify Indic script names don't crash and produce valid features."""
        hindi_name = "अपोलो हॉस्पिटल"
        english_name = "Apollo Hospital"
        features = compute_name_similarity_features(hindi_name, english_name)
        # All features should be finite
        for key, value in features.items():
            assert not math.isnan(value), f"NaN in {key}"
            assert not math.isinf(value), f"Inf in {key}"

    def test_mixed_script_names(self):
        """Verify mixed script names work correctly."""
        mixed1 = "Apollo हॉस्पिटल"
        mixed2 = "Apollo हॉस्पिटल"
        features = compute_name_similarity_features(mixed1, mixed2)
        assert features["name_exact_match"] == 1.0
        assert features["name_token_jaccard"] == 1.0

    def test_feature_names_exact(self):
        """Verify exactly seven required feature names are returned."""
        features = compute_name_similarity_features("Apollo Hospital", "Delta Clinic")
        expected_keys = {
            "name_token_jaccard",
            "name_token_overlap_count",
            "name_char_ngram_cosine",
            "name_edit_similarity",
            "name_length_ratio",
            "name_exact_match",
            "name_first_token_match",
        }
        assert set(features.keys()) == expected_keys

    def test_legal_suffix_handling(self):
        """Verify M1 legal suffix stripping is NOT applied by default."""
        features = compute_name_similarity_features(
            "Apollo Hospital Ltd", "Apollo Hospital"
        )
        # Without strip_legal=True, "ltd" token should remain
        assert features["name_exact_match"] == 0.0
        # But token overlap should capture shared tokens
        assert features["name_token_overlap_count"] >= 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])