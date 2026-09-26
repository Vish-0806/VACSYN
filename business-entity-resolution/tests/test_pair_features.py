"""
Tests for pair/context feature extraction.
"""

import math
import pytest

from src.business_entity_resolution.features.pair_features import (
    compute_pair_context_features,
)


class TestComputePairContextFeatures:
    def test_matching_countries(self):
        """Test country match returns 1.0 for same country."""
        features = compute_pair_context_features(
            s1_country="US",
            candidate_country="US",
            candidate_source="S2",
            s1_business_name="Apollo Hospital",
            candidate_business_name="Apollo Hospital",
            s1_address="105 Elm St",
            candidate_address="105 Elm St",
        )
        assert features["country_match"] == 1.0

    def test_non_matching_countries(self):
        """Test country match returns 0.0 for different countries."""
        features = compute_pair_context_features(
            s1_country="US",
            candidate_country="IN",
            candidate_source="S2",
            s1_business_name="Apollo Hospital",
            candidate_business_name="Apollo Hospital",
            s1_address="105 Elm St",
            candidate_address="105 Elm St",
        )
        assert features["country_match"] == 0.0

    def test_country_case_whitespace_variation(self):
        """Test country comparison handles case and whitespace."""
        features = compute_pair_context_features(
            s1_country="  us  ",
            candidate_country="US",
            candidate_source="S2",
            s1_business_name="Apollo Hospital",
            candidate_business_name="Apollo Hospital",
            s1_address="105 Elm St",
            candidate_address="105 Elm St",
        )
        assert features["country_match"] == 1.0

        features = compute_pair_context_features(
            s1_country="Us",
            candidate_country="us",
            candidate_source="S2",
            s1_business_name="Apollo Hospital",
            candidate_business_name="Apollo Hospital",
            s1_address="105 Elm St",
            candidate_address="105 Elm St",
        )
        assert features["country_match"] == 1.0

    def test_source_is_s2(self):
        """Test source_is_s2 returns 1.0 for S2 source."""
        features = compute_pair_context_features(
            s1_country="US",
            candidate_country="US",
            candidate_source="S2",
            s1_business_name="Apollo Hospital",
            candidate_business_name="Apollo Hospital",
            s1_address="105 Elm St",
            candidate_address="105 Elm St",
        )
        assert features["source_is_s2"] == 1.0
        assert features["source_is_s3"] == 0.0

    def test_source_is_s3(self):
        """Test source_is_s3 returns 1.0 for S3 source."""
        features = compute_pair_context_features(
            s1_country="US",
            candidate_country="US",
            candidate_source="S3",
            s1_business_name="Apollo Hospital",
            candidate_business_name="Apollo Hospital",
            s1_address="105 Elm St",
            candidate_address="105 Elm St",
        )
        assert features["source_is_s3"] == 1.0
        assert features["source_is_s2"] == 0.0

    def test_s2_s3_exclusivity(self):
        """Test S2 and S3 indicators are mutually exclusive."""
        features_s2 = compute_pair_context_features(
            s1_country="US", candidate_country="US", candidate_source="S2",
            s1_business_name="A", candidate_business_name="A",
            s1_address="A", candidate_address="A",
        )
        features_s3 = compute_pair_context_features(
            s1_country="US", candidate_country="US", candidate_source="S3",
            s1_business_name="A", candidate_business_name="A",
            s1_address="A", candidate_address="A",
        )
        features_other = compute_pair_context_features(
            s1_country="US", candidate_country="US", candidate_source="S1",
            s1_business_name="A", candidate_business_name="A",
            s1_address="A", candidate_address="A",
        )

        # S2
        assert features_s2["source_is_s2"] == 1.0
        assert features_s2["source_is_s3"] == 0.0

        # S3
        assert features_s3["source_is_s3"] == 1.0
        assert features_s3["source_is_s2"] == 0.0

        # Other source
        assert features_other["source_is_s2"] == 0.0
        assert features_other["source_is_s3"] == 0.0

    def test_identical_name_lengths(self):
        """Test name length ratio is 1.0 for identical lengths."""
        features = compute_pair_context_features(
            s1_country="US", candidate_country="US", candidate_source="S2",
            s1_business_name="Apollo Hospital",
            candidate_business_name="Apollo Hospital",
            s1_address="105 Elm St",
            candidate_address="105 Elm St",
        )
        assert features["name_length_ratio_context"] == 1.0

    def test_different_name_lengths(self):
        """Test name length ratio for different lengths."""
        features = compute_pair_context_features(
            s1_country="US", candidate_country="US", candidate_source="S2",
            s1_business_name="Apollo",
            candidate_business_name="Apollo Hospital",
            s1_address="105 Elm St",
            candidate_address="105 Elm St",
        )
        assert 0.0 < features["name_length_ratio_context"] < 1.0

    def test_empty_name_handling(self):
        """Test name length ratio handles empty names."""
        features = compute_pair_context_features(
            s1_country="US", candidate_country="US", candidate_source="S2",
            s1_business_name="",
            candidate_business_name="Apollo Hospital",
            s1_address="105 Elm St",
            candidate_address="105 Elm St",
        )
        assert features["name_length_ratio_context"] == 0.0

        features = compute_pair_context_features(
            s1_country="US", candidate_country="US", candidate_source="S2",
            s1_business_name="",
            candidate_business_name="",
            s1_address="105 Elm St",
            candidate_address="105 Elm St",
        )
        assert features["name_length_ratio_context"] == 1.0

    def test_identical_address_lengths(self):
        """Test address length ratio is 1.0 for identical lengths."""
        features = compute_pair_context_features(
            s1_country="US", candidate_country="US", candidate_source="S2",
            s1_business_name="Apollo Hospital",
            candidate_business_name="Apollo Hospital",
            s1_address="105 Elm Street, Morganton, NC",
            candidate_address="105 Elm Street, Morganton, NC",
        )
        assert features["address_length_ratio_context"] == 1.0

    def test_different_address_lengths(self):
        """Test address length ratio for different lengths."""
        features = compute_pair_context_features(
            s1_country="US", candidate_country="US", candidate_source="S2",
            s1_business_name="Apollo Hospital",
            candidate_business_name="Apollo Hospital",
            s1_address="105 Elm St",
            candidate_address="105 Elm Street, Morganton, Burke County, NC",
        )
        assert 0.0 < features["address_length_ratio_context"] < 1.0

    def test_empty_address_handling(self):
        """Test address length ratio handles empty addresses."""
        features = compute_pair_context_features(
            s1_country="US", candidate_country="US", candidate_source="S2",
            s1_business_name="Apollo Hospital",
            candidate_business_name="Apollo Hospital",
            s1_address="",
            candidate_address="105 Elm St",
        )
        assert features["address_length_ratio_context"] == 0.0

        features = compute_pair_context_features(
            s1_country="US", candidate_country="US", candidate_source="S2",
            s1_business_name="Apollo Hospital",
            candidate_business_name="Apollo Hospital",
            s1_address="",
            candidate_address="",
        )
        assert features["address_length_ratio_context"] == 1.0

    def test_none_handling(self):
        """Test None values are handled safely."""
        features = compute_pair_context_features(
            s1_country=None,
            candidate_country=None,
            candidate_source=None,
            s1_business_name=None,
            candidate_business_name=None,
            s1_address=None,
            candidate_address=None,
        )
        assert features["country_match"] == 0.0
        assert features["source_is_s2"] == 0.0
        assert features["source_is_s3"] == 0.0
        assert features["name_length_ratio_context"] == 1.0
        assert features["address_length_ratio_context"] == 1.0

    def test_unicode_indic_name_handling(self):
        """Test Unicode/Indic script names don't crash."""
        hindi_name = "अपोलो हॉस्पिटल"
        features = compute_pair_context_features(
            s1_country="IN", candidate_country="IN", candidate_source="S2",
            s1_business_name=hindi_name,
            candidate_business_name=hindi_name,
            s1_address="105 MG Road",
            candidate_address="105 MG Road",
        )
        assert features["country_match"] == 1.0
        assert features["name_length_ratio_context"] == 1.0

    def test_unicode_address_handling(self):
        """Test Unicode addresses don't crash."""
        french_addr = "15 Rue de l'Étoile, 75008 Paris"
        features = compute_pair_context_features(
            s1_country="FR", candidate_country="FR", candidate_source="S3",
            s1_business_name="Apollo Hospital",
            candidate_business_name="Apollo Hospital",
            s1_address=french_addr,
            candidate_address=french_addr,
        )
        assert features["country_match"] == 1.0
        assert features["address_length_ratio_context"] == 1.0

    def test_feature_names_exact(self):
        """Test exactly 5 required feature names are returned."""
        features = compute_pair_context_features(
            s1_country="US", candidate_country="US", candidate_source="S2",
            s1_business_name="Apollo Hospital",
            candidate_business_name="Apollo Hospital",
            s1_address="105 Elm St",
            candidate_address="105 Elm St",
        )
        expected_keys = {
            "country_match",
            "source_is_s2",
            "source_is_s3",
            "name_length_ratio_context",
            "address_length_ratio_context",
        }
        assert set(features.keys()) == expected_keys

    def test_no_nan_inf(self):
        """Test no NaN or Inf values for various inputs."""
        test_cases = [
            ("US", "US", "S2", "Apollo Hospital", "Apollo Hospital", "105 Elm St", "105 Elm St"),
            ("", "", "", "", "", "", ""),
            (None, None, None, None, None, None, None),
            ("IN", "IN", "S3", "अपोलो हॉस्पिटल", "अपोलो हॉस्पिटल", "एम जी रोड", "एम जी रोड"),
            ("FR", "FR", "S2", "Apollo Hospital", "Apollo Hospital", "15 Rue de l'Étoile", "15 Rue de l'Étoile"),
        ]
        for case in test_cases:
            features = compute_pair_context_features(*case)
            for key, value in features.items():
                assert not math.isnan(value), f"NaN in {key} for {case}"
                assert not math.isinf(value), f"Inf in {key} for {case}"

    def test_bounded_features(self):
        """Test all features are in their expected bounds."""
        features = compute_pair_context_features(
            s1_country="US", candidate_country="IN", candidate_source="S2",
            s1_business_name="Short",
            candidate_business_name="Very Long Business Name",
            s1_address="105 Elm St",
            candidate_address="105 Elm Street, Morganton, Burke County, NC",
        )
        # Binary features
        assert features["country_match"] in (0.0, 1.0)
        assert features["source_is_s2"] in (0.0, 1.0)
        assert features["source_is_s3"] in (0.0, 1.0)
        # Ratio features
        assert 0.0 <= features["name_length_ratio_context"] <= 1.0
        assert 0.0 <= features["address_length_ratio_context"] <= 1.0

    def test_mixed_case_source(self):
        """Test source handling with mixed case."""
        features = compute_pair_context_features(
            s1_country="US", candidate_country="US", candidate_source="s2",
            s1_business_name="A", candidate_business_name="A",
            s1_address="A", candidate_address="A",
        )
        assert features["source_is_s2"] == 1.0

        features = compute_pair_context_features(
            s1_country="US", candidate_country="US", candidate_source="S3",
            s1_business_name="A", candidate_business_name="A",
            s1_address="A", candidate_address="A",
        )
        assert features["source_is_s3"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])