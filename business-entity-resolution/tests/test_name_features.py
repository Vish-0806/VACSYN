"""
Tests for name similarity feature extraction.
"""

import math
import pytest

from src.business_entity_resolution.features.name_features import (
    compute_name_similarity_features,
    normalize_name,
    tokenize_name,
)


class TestNormalizeName:
    def test_none_input(self):
        assert normalize_name(None) == ""

    def test_empty_string(self):
        assert normalize_name("") == ""

    def test_whitespace_only(self):
        assert normalize_name("   ") == ""

    def test_case_normalization(self):
        assert normalize_name("Apollo Hospital") == "apollo hospital"
        assert normalize_name("APOLLO HOSPITAL") == "apollo hospital"

    def test_punctuation_handling(self):
        assert normalize_name("Apollo Hospital, Ltd.") == "apollo hospital ltd"
        assert normalize_name("Apollo Hospital Ltd") == "apollo hospital ltd"

    def test_whitespace_normalization(self):
        assert normalize_name("Apollo   Hospital") == "apollo hospital"
        assert normalize_name("  Apollo Hospital  ") == "apollo hospital"

    def test_ampersand_normalization(self):
        assert normalize_name("Smith & Jones") == "smith and jones"

    def test_unicode_preservation(self):
        hindi_name = "अपोलो हॉस्पिटल"
        assert normalize_name(hindi_name) == hindi_name

    def test_mixed_script(self):
        mixed = "Apollo हॉस्पिटल"
        result = normalize_name(mixed)
        assert "apollo" in result
        assert "हॉस्पिटल" in result

    def test_numbers_preserved(self):
        assert normalize_name("Apollo 24/7 Hospital") == "apollo 24 7 hospital"


class TestTokenizeName:
    def test_empty_input(self):
        assert tokenize_name("") == []
        assert tokenize_name(None) == []

    def test_basic_tokenization(self):
        assert tokenize_name("Apollo Hospital") == ["apollo", "hospital"]

    def test_unicode_tokens(self):
        tokens = tokenize_name("अपोलो हॉस्पिटल")
        assert tokens == ["अपोलो", "हॉस्पिटल"]

    def test_punctuation_removed(self):
        assert tokenize_name("Apollo, Hospital!") == ["apollo", "hospital"]


class TestComputeNameSimilarityFeatures:
    def test_exact_match_case_insensitive(self):
        features = compute_name_similarity_features("Apollo Hospital", "apollo hospital")
        assert features["name_exact_match"] == 1.0
        assert features["name_token_jaccard"] == 1.0
        assert features["name_edit_similarity"] == 1.0
        assert features["name_length_ratio"] == 1.0

    def test_punctuation_whitespace_variation(self):
        features = compute_name_similarity_features(
            "Apollo Hospital, Ltd.", "Apollo Hospital Ltd"
        )
        assert features["name_exact_match"] == 1.0
        assert features["name_token_jaccard"] == 1.0

    def test_unicode_preservation(self):
        hindi1 = "अपोलो हॉस्पिटल"
        hindi2 = "अपोलो हॉस्पिटल"
        features = compute_name_similarity_features(hindi1, hindi2)
        assert features["name_exact_match"] == 1.0
        assert features["name_token_jaccard"] == 1.0

    def test_identical_token_sets_jaccard(self):
        features = compute_name_similarity_features("Apollo Hospital", "Hospital Apollo")
        assert features["name_token_jaccard"] == 1.0
        assert features["name_token_overlap_count"] == 2.0

    def test_disjoint_token_sets_jaccard(self):
        features = compute_name_similarity_features("Apollo Hospital", "Delta Clinic")
        assert features["name_token_jaccard"] == 0.0
        assert features["name_token_overlap_count"] == 0.0

    def test_partial_token_overlap(self):
        features = compute_name_similarity_features("Apollo Hospital Delhi", "Apollo Clinic Mumbai")
        assert features["name_token_jaccard"] == 1.0 / 4.0
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
        assert features["name_token_jaccard"] == 1.0
        assert features["name_edit_similarity"] == 1.0
        assert features["name_length_ratio"] == 1.0

    def test_none_inputs(self):
        features = compute_name_similarity_features(None, None)
        assert features["name_exact_match"] == 0.0
        assert features["name_token_jaccard"] == 1.0
        assert features["name_edit_similarity"] == 1.0
        assert features["name_length_ratio"] == 1.0

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
            "name_char_2gram_similarity",
            "name_char_3gram_similarity",
            "name_edit_similarity",
            "name_first_token_match",
            "name_length_ratio",
        ]
        for key in bounded_keys:
            assert 0.0 <= features[key] <= 1.0, f"{key} out of bounds: {features[key]}"

    def test_char_ngram_similarity(self):
        features = compute_name_similarity_features("Apollo", "Apolo")
        assert 0.0 < features["name_char_2gram_similarity"] < 1.0
        assert 0.0 < features["name_char_3gram_similarity"] < 1.0

    def test_short_strings(self):
        features = compute_name_similarity_features("A", "B")
        assert features["name_edit_similarity"] == 0.0
        assert features["name_length_ratio"] == 1.0

    def test_single_char_vs_multi_char(self):
        features = compute_name_similarity_features("A", "Apollo")
        assert features["name_length_ratio"] == 1.0 / 6.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])