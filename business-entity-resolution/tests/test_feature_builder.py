"""
Tests for feature assembly / feature builder.
"""

import math
import pandas as pd
import pytest

from src.business_entity_resolution.features.pair_features import (
    PairFeatureExtractor,
    build_pair_features,
    compute_pair_context_features,
)


class TestPairFeatureExtractor:
    """Tests for PairFeatureExtractor.extract_features"""

    def setup_method(self):
        """Set up test data."""
        self.s1_records = {
            "S1-001": {
                "entity_id": "S1-001",
                "business_name": "Apollo Hospital",
                "business_address": "105 Elm Street, Morganton, NC",
                "country": "US",
            },
            "S1-002": {
                "entity_id": "S1-002",
                "business_name": "Acme Corp",
                "business_address": "500 Market St, Suite 400, San Francisco, CA",
                "country": "US",
            },
            "S1-003": {
                "entity_id": "S1-003",
                "business_name": "अपोलो हॉस्पिटल",
                "business_address": "एम जी रोड, बेंगलुरु, कर्नाटक",
                "country": "IN",
            },
        }

        self.s2_records = {
            "S2-001": {
                "entity_id": "S2-001",
                "business_name": "Apollo Hospital",
                "business_address": "105 Elm St, Morganton, NC",
                "country": "US",
            },
            "S2-002": {
                "entity_id": "S2-002",
                "business_name": "Acme Corporation",
                "business_address": "500 Market Street, San Francisco, CA",
                "country": "US",
            },
        }

        self.s3_records = {
            "S3-001": {
                "entity_id": "S3-001",
                "business_name": "Apollo Hospital",
                "business_address": "105 Elm Street, Morganton, Burke County, NC",
                "country": "US",
            },
            "S3-002": {
                "entity_id": "S3-002",
                "business_name": "Different Company",
                "business_address": "999 Unknown Ave, Nowhere, NY",
                "country": "US",
            },
        }

        self.candidate_pairs = pd.DataFrame([
            {"s1_entity_id": "S1-001", "candidate_entity_id": "S2-001", "candidate_source": "S2"},
            {"s1_entity_id": "S1-001", "candidate_entity_id": "S3-001", "candidate_source": "S3"},
            {"s1_entity_id": "S1-002", "candidate_entity_id": "S2-002", "candidate_source": "S2"},
            {"s1_entity_id": "S1-003", "candidate_entity_id": "S3-002", "candidate_source": "S3"},
        ])

        self.extractor = PairFeatureExtractor()

    def test_correct_feature_count(self):
        """Test that output has exactly 21 columns (3 identity + 18 features)."""
        result = self.extractor.extract_features(
            self.candidate_pairs, self.s1_records, self.s2_records, self.s3_records
        )
        assert len(result.columns) == 21
        assert len(result) == 4

    def test_exact_column_names(self):
        """Test exact column names match expected."""
        result = self.extractor.extract_features(
            self.candidate_pairs, self.s1_records, self.s2_records, self.s3_records
        )
        expected_columns = [
            "s1_entity_id",
            "candidate_entity_id",
            "candidate_source",
            "name_token_jaccard",
            "name_token_overlap_count",
            "name_char_ngram_cosine",
            "name_edit_similarity",
            "name_length_ratio",
            "name_exact_match",
            "name_first_token_match",
            "address_token_jaccard",
            "address_token_overlap_count",
            "address_house_number_match",
            "address_edit_similarity",
            "address_length_ratio",
            "address_missing",
            "country_match",
            "source_is_s2",
            "source_is_s3",
            "name_length_ratio_context",
            "address_length_ratio_context",
        ]
        assert list(result.columns) == expected_columns

    def test_exact_column_order(self):
        """Test column order is deterministic and matches spec."""
        result = self.extractor.extract_features(
            self.candidate_pairs, self.s1_records, self.s2_records, self.s3_records
        )
        expected_order = [
            "s1_entity_id",
            "candidate_entity_id",
            "candidate_source",
            "name_token_jaccard",
            "name_token_overlap_count",
            "name_char_ngram_cosine",
            "name_edit_similarity",
            "name_length_ratio",
            "name_exact_match",
            "name_first_token_match",
            "address_token_jaccard",
            "address_token_overlap_count",
            "address_house_number_match",
            "address_edit_similarity",
            "address_length_ratio",
            "address_missing",
            "country_match",
            "source_is_s2",
            "source_is_s3",
            "name_length_ratio_context",
            "address_length_ratio_context",
        ]
        assert list(result.columns) == expected_order

    def test_identity_columns_preserved(self):
        """Test identity columns are preserved exactly."""
        result = self.extractor.extract_features(
            self.candidate_pairs, self.s1_records, self.s2_records, self.s3_records
        )
        pd.testing.assert_series_equal(
            result["s1_entity_id"],
            self.candidate_pairs["s1_entity_id"],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            result["candidate_entity_id"],
            self.candidate_pairs["candidate_entity_id"],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            result["candidate_source"],
            self.candidate_pairs["candidate_source"],
            check_names=False,
        )

    def test_s2_candidate_lookup(self):
        """Test S2 candidate lookup works correctly."""
        result = self.extractor.extract_features(
            self.candidate_pairs, self.s1_records, self.s2_records, self.s3_records
        )
        # First row: S1-001 -> S2-001 (should match closely)
        row0 = result.iloc[0]
        assert row0["s1_entity_id"] == "S1-001"
        assert row0["candidate_entity_id"] == "S2-001"
        assert row0["candidate_source"] == "S2"
        assert row0["source_is_s2"] == 1.0
        assert row0["source_is_s3"] == 0.0
        # Names are identical -> high similarity
        assert row0["name_exact_match"] == 1.0
        assert row0["name_token_jaccard"] == 1.0

    def test_s3_candidate_lookup(self):
        """Test S3 candidate lookup works correctly."""
        result = self.extractor.extract_features(
            self.candidate_pairs, self.s1_records, self.s2_records, self.s3_records
        )
        # Second row: S1-001 -> S3-001 (should match closely)
        row1 = result.iloc[1]
        assert row1["s1_entity_id"] == "S1-001"
        assert row1["candidate_entity_id"] == "S3-001"
        assert row1["candidate_source"] == "S3"
        assert row1["source_is_s3"] == 1.0
        assert row1["source_is_s2"] == 0.0
        assert row1["name_exact_match"] == 1.0

    def test_mixed_s2_s3_candidates(self):
        """Test mixed S2 and S3 candidates in same pair set."""
        result = self.extractor.extract_features(
            self.candidate_pairs, self.s1_records, self.s2_records, self.s3_records
        )
        sources = result["candidate_source"].tolist()
        assert "S2" in sources
        assert "S3" in sources
        assert result["source_is_s2"].sum() == 2  # Two S2 candidates
        assert result["source_is_s3"].sum() == 2  # Two S3 candidates

    def test_multiple_candidates_for_one_s1(self):
        """Test multiple candidate pairs for the same S1 entity."""
        # S1-001 has two candidates: S2-001 and S3-001
        result = self.extractor.extract_features(
            self.candidate_pairs, self.s1_records, self.s2_records, self.s3_records
        )
        s1_001_rows = result[result["s1_entity_id"] == "S1-001"]
        assert len(s1_001_rows) == 2

    def test_feature_values_match_direct_calls(self):
        """Test feature values match direct calls to the three feature modules."""
        from src.business_entity_resolution.features.name_features import compute_name_similarity_features
        from src.business_entity_resolution.features.address_features import compute_address_similarity_features

        result = self.extractor.extract_features(
            self.candidate_pairs, self.s1_records, self.s2_records, self.s3_records
        )

        # Check first row: S1-001 vs S2-001
        row = result.iloc[0]
        s1_rec = self.s1_records["S1-001"]
        cand_rec = self.s2_records["S2-001"]

        # Name features
        name_direct = compute_name_similarity_features(s1_rec["business_name"], cand_rec["business_name"])
        for k, v in name_direct.items():
            assert abs(row[k] - v) < 1e-10, f"Mismatch in {k}: {row[k]} vs {v}"

        # Address features
        addr_direct = compute_address_similarity_features(s1_rec["business_address"], cand_rec["business_address"])
        for k, v in addr_direct.items():
            assert abs(row[k] - v) < 1e-10, f"Mismatch in {k}: {row[k]} vs {v}"

        # Pair features
        pair_direct = compute_pair_context_features(
            s1_rec["country"], cand_rec["country"], "S2",
            s1_rec["business_name"], cand_rec["business_name"],
            s1_rec["business_address"], cand_rec["business_address"]
        )
        for k, v in pair_direct.items():
            assert abs(row[k] - v) < 1e-10, f"Mismatch in {k}: {row[k]} vs {v}"

    def test_no_candidate_rows_dropped(self):
        """Test no candidate rows are silently dropped."""
        result = self.extractor.extract_features(
            self.candidate_pairs, self.s1_records, self.s2_records, self.s3_records
        )
        assert len(result) == len(self.candidate_pairs)

    def test_no_duplicate_rows_introduced(self):
        """Test no duplicate rows are introduced."""
        result = self.extractor.extract_features(
            self.candidate_pairs, self.s1_records, self.s2_records, self.s3_records
        )
        # Check for duplicate identity columns
        id_cols = ["s1_entity_id", "candidate_entity_id", "candidate_source"]
        duplicates = result[id_cols].duplicated().sum()
        assert duplicates == 0

    def test_missing_s1_entity_raises(self):
        """Test missing S1 entity raises clear error."""
        bad_pairs = pd.DataFrame([
            {"s1_entity_id": "S1-999", "candidate_entity_id": "S2-001", "candidate_source": "S2"},
        ])
        with pytest.raises(KeyError, match="S1 entity ID not found: S1-999"):
            self.extractor.extract_features(bad_pairs, self.s1_records, self.s2_records, self.s3_records)

    def test_missing_s2_entity_raises(self):
        """Test missing S2 entity raises clear error."""
        bad_pairs = pd.DataFrame([
            {"s1_entity_id": "S1-001", "candidate_entity_id": "S2-999", "candidate_source": "S2"},
        ])
        with pytest.raises(KeyError, match="S2 entity ID not found: S2-999"):
            self.extractor.extract_features(bad_pairs, self.s1_records, self.s2_records, self.s3_records)

    def test_missing_s3_entity_raises(self):
        """Test missing S3 entity raises clear error."""
        bad_pairs = pd.DataFrame([
            {"s1_entity_id": "S1-001", "candidate_entity_id": "S3-999", "candidate_source": "S3"},
        ])
        with pytest.raises(KeyError, match="S3 entity ID not found: S3-999"):
            self.extractor.extract_features(bad_pairs, self.s1_records, self.s2_records, self.s3_records)

    def test_invalid_candidate_source_raises(self):
        """Test invalid candidate_source raises clear error."""
        bad_pairs = pd.DataFrame([
            {"s1_entity_id": "S1-001", "candidate_entity_id": "S2-001", "candidate_source": "S1"},
        ])
        with pytest.raises(ValueError, match="Invalid candidate_source: S1"):
            self.extractor.extract_features(bad_pairs, self.s1_records, self.s2_records, self.s3_records)

    def test_missing_address_handling(self):
        """Test missing addresses in records."""
        s1_records = {
            "S1-001": {
                "entity_id": "S1-001",
                "business_name": "Apollo Hospital",
                "business_address": "",
                "country": "US",
            },
        }
        s2_records = {
            "S2-001": {
                "entity_id": "S2-001",
                "business_name": "Apollo Hospital",
                "business_address": "105 Elm St, Morganton, NC",
                "country": "US",
            },
        }
        s3_records = {}
        pairs = pd.DataFrame([{"s1_entity_id": "S1-001", "candidate_entity_id": "S2-001", "candidate_source": "S2"}])

        result = self.extractor.extract_features(pairs, s1_records, s2_records, s3_records)
        assert result.iloc[0]["address_missing"] == 1.0
        assert result.iloc[0]["address_token_jaccard"] == 0.0

    def test_unicode_indic_names(self):
        """Test Unicode/Indic names work correctly."""
        result = self.extractor.extract_features(
            self.candidate_pairs, self.s1_records, self.s2_records, self.s3_records
        )
        # Row 3: S1-003 (Indic) vs S3-002 (different)
        row3 = result[result["s1_entity_id"] == "S1-003"].iloc[0]
        assert row3["name_token_jaccard"] >= 0.0
        assert not math.isnan(row3["name_char_ngram_cosine"])

    def test_deterministic_output(self):
        """Test output is deterministic across multiple calls."""
        result1 = self.extractor.extract_features(
            self.candidate_pairs, self.s1_records, self.s2_records, self.s3_records
        )
        result2 = self.extractor.extract_features(
            self.candidate_pairs, self.s1_records, self.s2_records, self.s3_records
        )
        pd.testing.assert_frame_equal(result1, result2)

    def test_no_nan_inf_in_features(self):
        """Test no NaN or Inf in any feature column."""
        result = self.extractor.extract_features(
            self.candidate_pairs, self.s1_records, self.s2_records, self.s3_records
        )
        feature_cols = [c for c in result.columns if c not in ["s1_entity_id", "candidate_entity_id", "candidate_source"]]
        for col in feature_cols:
            assert not result[col].isna().any(), f"NaN in {col}"
            assert not (result[col] == float('inf')).any(), f"+Inf in {col}"
            assert not (result[col] == float('-inf')).any(), f"-Inf in {col}"


class TestBuildPairFeatures:
    """Tests for build_pair_features function."""

    def setup_method(self):
        """Set up test data as DataFrames."""
        self.s1_data = pd.DataFrame([
            {"entity_id": "S1-001", "business_name": "Apollo Hospital", "business_address": "105 Elm St, Morganton, NC", "country": "US"},
            {"entity_id": "S1-002", "business_name": "Acme Corp", "business_address": "500 Market St, San Francisco, CA", "country": "US"},
        ])

        self.candidate_data = pd.DataFrame([
            {"entity_id": "S2-001", "business_name": "Apollo Hospital", "business_address": "105 Elm St, Morganton, NC", "country": "US", "source": "S2"},
            {"entity_id": "S2-002", "business_name": "Acme Corporation", "business_address": "500 Market St, San Francisco, CA", "country": "US", "source": "S2"},
            {"entity_id": "S3-001", "business_name": "Apollo Hospital", "business_address": "105 Elm St, Morganton, NC", "country": "US", "source": "S3"},
        ])

        self.candidate_pairs = pd.DataFrame([
            {"s1_entity_id": "S1-001", "candidate_entity_id": "S2-001", "candidate_source": "S2"},
            {"s1_entity_id": "S1-001", "candidate_entity_id": "S3-001", "candidate_source": "S3"},
            {"s1_entity_id": "S1-002", "candidate_entity_id": "S2-002", "candidate_source": "S2"},
        ])

    def test_build_pair_features_works(self):
        """Test build_pair_features produces correct output."""
        result = build_pair_features(self.candidate_pairs, self.s1_data, self.candidate_data)
        assert len(result) == 3
        assert len(result.columns) == 21
        assert "name_token_jaccard" in result.columns
        assert "address_house_number_match" in result.columns
        assert "country_match" in result.columns

    def test_s3_candidates_included(self):
        """Test S3 candidates are correctly included."""
        result = build_pair_features(self.candidate_pairs, self.s1_data, self.candidate_data)
        s3_rows = result[result["candidate_source"] == "S3"]
        assert len(s3_rows) == 1
        assert s3_rows.iloc[0]["source_is_s3"] == 1.0


class TestMissingColumns:
    """Tests for missing required columns in input."""

    def test_missing_s1_entity_id_column(self):
        extractor = PairFeatureExtractor()
        pairs = pd.DataFrame([{"candidate_entity_id": "S2-001", "candidate_source": "S2"}])
        with pytest.raises(ValueError, match="Missing required column: s1_entity_id"):
            extractor.extract_features(pairs, {}, {}, {})

    def test_missing_candidate_entity_id_column(self):
        extractor = PairFeatureExtractor()
        pairs = pd.DataFrame([{"s1_entity_id": "S1-001", "candidate_source": "S2"}])
        with pytest.raises(ValueError, match="Missing required column: candidate_entity_id"):
            extractor.extract_features(pairs, {}, {}, {})

    def test_missing_candidate_source_column(self):
        extractor = PairFeatureExtractor()
        pairs = pd.DataFrame([{"s1_entity_id": "S1-001", "candidate_entity_id": "S2-001"}])
        with pytest.raises(ValueError, match="Missing required column: candidate_source"):
            extractor.extract_features(pairs, {}, {}, {})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])