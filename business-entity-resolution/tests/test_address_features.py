"""
Tests for address similarity feature extraction.
"""

import math
import pytest

from src.business_entity_resolution.features.address_features import (
    compute_address_similarity_features,
)


class TestComputeAddressSimilarityFeatures:
    def test_exact_match(self):
        """Test identical addresses produce perfect similarity."""
        addr = "1795 Westchester Drive, High Point, NC"
        features = compute_address_similarity_features(addr, addr)
        assert features["address_token_jaccard"] == 1.0
        assert features["address_token_overlap_count"] > 0.0
        assert features["address_house_number_match"] == 1.0
        assert features["address_edit_similarity"] == 1.0
        assert features["address_length_ratio"] == 1.0
        assert features["address_missing"] == 0.0

    def test_punctuation_whitespace_normalization(self):
        """Test that punctuation and whitespace differences are normalized."""
        addr1 = "1795 Westchester Drive, High Point, NC"
        addr2 = "1795 Westchester Drive High Point NC"
        features = compute_address_similarity_features(addr1, addr2)
        assert features["address_token_jaccard"] == 1.0
        assert features["address_house_number_match"] == 1.0

    def test_token_overlap(self):
        """Test partial token overlap."""
        addr1 = "105 Elm Street, Morganton, NC"
        addr2 = "105 Elm Street, Charlotte, NC"
        features = compute_address_similarity_features(addr1, addr2)
        assert 0.0 < features["address_token_jaccard"] < 1.0
        assert features["address_token_overlap_count"] >= 3.0  # 105, elm, street
        assert features["address_house_number_match"] == 1.0

    def test_disjoint_token_sets(self):
        """Test completely different addresses."""
        addr1 = "105 Elm Street, Morganton, NC"
        addr2 = "914 Pierpont Avenue, Cleveland, OH"
        features = compute_address_similarity_features(addr1, addr2)
        assert features["address_token_jaccard"] == 0.0
        assert features["address_token_overlap_count"] == 0.0
        assert features["address_house_number_match"] == 0.0

    def test_identical_house_numbers(self):
        """Test identical house numbers."""
        features = compute_address_similarity_features(
            "1795 Westchester Drive, High Point, NC",
            "1795 Main Street, Raleigh, NC"
        )
        assert features["address_house_number_match"] == 1.0

    def test_different_house_numbers(self):
        """Test different house numbers."""
        features = compute_address_similarity_features(
            "105 Elm Street, Morganton, NC",
            "914 Pierpont Avenue, Cleveland, OH"
        )
        assert features["address_house_number_match"] == 0.0

    def test_one_missing_house_number(self):
        """Test one address has house number, other doesn't."""
        features = compute_address_similarity_features(
            "105 Elm Street, Morganton, NC",
            "Near SBI ATM, MG Road, Bengaluru"
        )
        assert features["address_house_number_match"] == 0.0

    def test_both_missing_house_numbers(self):
        """Test neither address has a house number."""
        features = compute_address_similarity_features(
            "Near SBI ATM, MG Road, Bengaluru",
            "Opposite Bus Stand, Main Market, Jaipur"
        )
        assert features["address_house_number_match"] == 0.0

    def test_one_missing_address(self):
        """Test one address is missing/empty."""
        features = compute_address_similarity_features(
            "105 Elm Street, Morganton, NC",
            ""
        )
        assert features["address_token_jaccard"] == 0.0
        assert features["address_token_overlap_count"] == 0.0
        assert features["address_house_number_match"] == 0.0
        assert features["address_edit_similarity"] == 0.0
        assert features["address_length_ratio"] == 0.0
        assert features["address_missing"] == 1.0

    def test_both_missing_addresses(self):
        """Test both addresses are missing/empty."""
        features = compute_address_similarity_features("", "")
        assert features["address_token_jaccard"] == 0.0
        assert features["address_token_overlap_count"] == 0.0
        assert features["address_house_number_match"] == 0.0
        assert features["address_edit_similarity"] == 1.0  # Both empty -> 1.0
        assert features["address_length_ratio"] == 1.0
        assert features["address_missing"] == 1.0

    def test_none_addresses(self):
        """Test None addresses are handled."""
        features = compute_address_similarity_features(None, None)
        assert features["address_token_jaccard"] == 0.0
        assert features["address_token_overlap_count"] == 0.0
        assert features["address_house_number_match"] == 0.0
        assert features["address_edit_similarity"] == 1.0
        assert features["address_length_ratio"] == 1.0
        assert features["address_missing"] == 1.0

    def test_edit_similarity(self):
        """Test edit similarity for similar but different addresses."""
        features = compute_address_similarity_features(
            "105 Elm Street, Morganton, NC",
            "105 Elm Street, Morganton, N Carolina"
        )
        assert 0.0 < features["address_edit_similarity"] < 1.0
        assert features["address_token_jaccard"] > 0.0

    def test_length_ratio(self):
        """Test length ratio for different length addresses."""
        features = compute_address_similarity_features(
            "105 Elm Street",
            "105 Elm Street, Morganton, Burke County, NC"
        )
        assert 0.0 < features["address_length_ratio"] < 1.0

    def test_address_missing_behavior(self):
        """Test address_missing flag behavior."""
        # Both present
        features = compute_address_similarity_features("105 Elm St", "105 Elm St")
        assert features["address_missing"] == 0.0

        # One missing
        features = compute_address_similarity_features("105 Elm St", "")
        assert features["address_missing"] == 1.0

        features = compute_address_similarity_features("", "105 Elm St")
        assert features["address_missing"] == 1.0

        # Both missing
        features = compute_address_similarity_features("", "")
        assert features["address_missing"] == 1.0

    def test_unicode_address(self):
        """Test Unicode (French) address."""
        addr1 = "15 Rue de l'Étoile, 75008 Paris, Île-de-France"
        addr2 = "15 Rue de l'Étoile, 75008 Paris, Île-de-France"
        features = compute_address_similarity_features(addr1, addr2)
        assert features["address_token_jaccard"] == 1.0
        assert features["address_house_number_match"] == 1.0

    def test_indic_script_address(self):
        """Test Indic-script address."""
        addr1 = "एम जी रोड, बेंगलुरु, कर्नाटक"
        addr2 = "एम जी रोड, बेंगलुरु, कर्नाटक"
        features = compute_address_similarity_features(addr1, addr2)
        assert features["address_token_jaccard"] == 1.0

    def test_mixed_script_address(self):
        """Test mixed-script address."""
        addr1 = "105 MG Road, Bengaluru, Karnataka"
        addr2 = "105 MG Road, Bengaluru, Karnataka"
        features = compute_address_similarity_features(addr1, addr2)
        assert features["address_token_jaccard"] == 1.0
        assert features["address_house_number_match"] == 1.0

    def test_short_address(self):
        """Test very short addresses."""
        features = compute_address_similarity_features("105 Elm St", "105 Elm St")
        assert features["address_token_jaccard"] == 1.0
        assert features["address_edit_similarity"] == 1.0

    def test_no_nan_inf(self):
        """Test no NaN or Inf values for various inputs."""
        test_cases = [
            ("105 Elm St", "105 Elm St"),
            ("", ""),
            (None, "105 Elm St"),
            ("एम जी रोड", "एम जी रोड"),
            ("15 Rue de l'Étoile", "15 Rue de l'Étoile"),
            ("A", "B"),
        ]
        for addr1, addr2 in test_cases:
            features = compute_address_similarity_features(addr1, addr2)
            for key, value in features.items():
                assert not math.isnan(value), f"NaN in {key} for ({addr1}, {addr2})"
                assert not math.isinf(value), f"Inf in {key} for ({addr1}, {addr2})"

    def test_bounded_features(self):
        """Test all bounded features are in [0, 1]."""
        features = compute_address_similarity_features("105 Elm St", "914 Pierpont Ave")
        bounded_keys = [
            "address_token_jaccard",
            "address_house_number_match",
            "address_edit_similarity",
            "address_length_ratio",
            "address_missing",
        ]
        for key in bounded_keys:
            assert 0.0 <= features[key] <= 1.0, f"{key} out of bounds: {features[key]}"

    def test_feature_names_exact(self):
        """Test exactly six required feature names are returned."""
        features = compute_address_similarity_features("105 Elm St", "914 Pierpont Ave")
        expected_keys = {
            "address_token_jaccard",
            "address_token_overlap_count",
            "address_house_number_match",
            "address_edit_similarity",
            "address_length_ratio",
            "address_missing",
        }
        assert set(features.keys()) == expected_keys

    def test_abbreviation_expansion_in_tokens(self):
        """Test that abbreviation expansion affects tokenization."""
        addr1 = "105 ELM ST, MORGANTON, NC"
        addr2 = "105 Elm Street, Morganton, NC"
        features = compute_address_similarity_features(addr1, addr2)
        # M1 normalizer expands ST -> street, so tokens should match
        assert features["address_token_jaccard"] == 1.0

    def test_reordered_address_components(self):
        """Test reordered address components."""
        addr1 = "OH, Columbus, 5559 Orville Avenue"
        addr2 = "5559 Orville Avenue, Columbus, OH"
        features = compute_address_similarity_features(addr1, addr2)
        # Tokens should be the same after normalization
        assert features["address_token_jaccard"] == 1.0
        assert features["address_house_number_match"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])