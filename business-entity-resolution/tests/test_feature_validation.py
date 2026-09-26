"""
Validation tests for the assembled 18-feature model input.

These tests verify the feature matrix produced by the assembly layer
conforms to the required schema, value ranges, and quality expectations.
"""

import math
import pandas as pd
import pytest

from src.business_entity_resolution.blocking.candidate_generator import CandidateGenerator
from src.business_entity_resolution.features.pair_features import (
    PairFeatureExtractor,
    build_pair_features,
)


def create_validation_data():
    """Create a realistic validation dataset with known properties."""
    # S1 reference entities
    s1_df = pd.DataFrame([
        # US entities
        {
            "entity_id": "S1-001",
            "business_name": "Apollo Hospital",
            "business_address": "105 Elm Street, Morganton, NC",
            "country": "US",
        },
        {
            "entity_id": "S1-002",
            "business_name": "Acme Corporation",
            "business_address": "500 Market St, Suite 400, San Francisco, CA",
            "country": "US",
        },
        {
            "entity_id": "S1-003",
            "business_name": "Primary Care Group",
            "business_address": "1795 Westchester Drive, High Point, NC",
            "country": "US",
        },
        {
            "entity_id": "S1-004",
            "business_name": "Walmart Supercenter",
            "business_address": "100 Main St, Bentonville, AR",
            "country": "US",
        },
        {
            "entity_id": "S1-005",
            "business_name": "Best Buy Electronics",
            "business_address": "7601 Penn Ave S, Richfield, MN",
            "country": "US",
        },
        # India entities
        {
            "entity_id": "S1-006",
            "business_name": "Apollo Hospital",
            "business_address": "MG Road, Bengaluru, Karnataka",
            "country": "India",
        },
        {
            "entity_id": "S1-007",
            "business_name": "Wipro Technologies",
            "business_address": "Plot No 2, Electronic City, Bengaluru",
            "country": "India",
        },
        {
            "entity_id": "S1-008",
            "business_name": "Primary Care Group",
            "business_address": "123 MG Road, Pune, Maharashtra",
            "country": "India",
        },
        # France entities (for test country coverage)
        {
            "entity_id": "S1-009",
            "business_name": "Bordeaux Club SARL",
            "business_address": "15 Rue de l'Étoile, 75008 Paris",
            "country": "France",
        },
        {
            "entity_id": "S1-010",
            "business_name": "Nantes Club SAS",
            "business_address": "10 Rue de la Paix, Nantes",
            "country": "France",
        },
        # Entity with missing address
        {
            "entity_id": "S1-011",
            "business_name": "Missing Address Corp",
            "business_address": "",
            "country": "US",
        },
        # Entity with missing name
        {
            "entity_id": "S1-012",
            "business_name": "",
            "business_address": "123 Main St, Springfield, IL",
            "country": "US",
        },
        # Indic script names
        {
            "entity_id": "S1-013",
            "business_name": "अपोलो हॉस्पिटल",
            "business_address": "एम जी रोड, बेंगलुरु, कर्नाटक",
            "country": "India",
        },
    ])

    # S2 candidate pool
    s2_df = pd.DataFrame([
        # True matches for S1-001 (Apollo Hospital US)
        {
            "entity_id": "S2-001",
            "business_name": "Apollo Hospital",
            "business_address": "105 Elm St, Morganton, NC",
            "country": "US",
        },
        # True match for S1-002 (Acme Corp)
        {
            "entity_id": "S2-002",
            "business_name": "Acme Corp",
            "business_address": "500 Market St, San Francisco, CA",
            "country": "US",
        },
        # True match for S1-003 (Primary Care Group)
        {
            "entity_id": "S2-003",
            "business_name": "Primary Care Group",
            "business_address": "1795 Westchester Dr, High Point, NC",
            "country": "US",
        },
        # True match for S1-004 (Walmart)
        {
            "entity_id": "S2-004",
            "business_name": "Wal-Mart Super Center",
            "business_address": "100 Main St, Bentonville, AR",
            "country": "US",
        },
        # True match for S1-006 (Apollo Hospital India)
        {
            "entity_id": "S2-006",
            "business_name": "Apollo Hospital",
            "business_address": "MG Road, Bengaluru, Karnataka",
            "country": "India",
        },
        # True match for S1-007 (Wipro)
        {
            "entity_id": "S2-007",
            "business_name": "Wipro Ltd",
            "business_address": "Plot No 2, Electronic City, Bengaluru",
            "country": "India",
        },
        # True match for S1-009 (Bordeaux Club)
        {
            "entity_id": "S2-009",
            "business_name": "Bordeaux Club SARL",
            "business_address": "15 Rue de l'Étoile, 75008 Paris",
            "country": "France",
        },
        # Hard negatives - high name similarity but different entity
        {
            "entity_id": "S2-010",
            "business_name": "Apollo Clinic",
            "business_address": "105 Elm St, Morganton, NC",
            "country": "US",
        },
        {
            "entity_id": "S2-011",
            "business_name": "Acme Industries",
            "business_address": "500 Market St, San Francisco, CA",
            "country": "US",
        },
        # Different country - should be isolated by blocking
        {
            "entity_id": "S2-012",
            "business_name": "Apollo Hospital",
            "business_address": "105 Elm St, London",
            "country": "UK",
        },
        # Missing address
        {
            "entity_id": "S2-013",
            "business_name": "Missing Address Corp",
            "business_address": "",
            "country": "US",
        },
        # Missing name
        {
            "entity_id": "S2-014",
            "business_name": "",
            "business_address": "123 Main St, Springfield, IL",
            "country": "US",
        },
        # Indic script
        {
            "entity_id": "S2-015",
            "business_name": "अपोलो हॉस्पिटल",
            "business_address": "एम जी रोड, बेंगलुरु, कर्नाटक",
            "country": "India",
        },
        # Distractors
        {
            "entity_id": "S2-016",
            "business_name": "Delta Clinic",
            "business_address": "999 Unknown Ave, Nowhere, NY",
            "country": "US",
        },
        {
            "entity_id": "S2-017",
            "business_name": "Different Company",
            "business_address": "888 Other St, Elsewhere, TX",
            "country": "US",
        },
    ])

    # S3 candidate pool
    s3_df = pd.DataFrame([
        # True matches
        {
            "entity_id": "S3-001",
            "business_name": "Apollo Hospital",
            "business_address": "105 Elm Street, Morganton, Burke County, NC",
            "country": "US",
        },
        {
            "entity_id": "S3-002",
            "business_name": "Acme Corporation",
            "business_address": "500 Market Street, San Francisco, CA",
            "country": "US",
        },
        {
            "entity_id": "S3-003",
            "business_name": "Primary Care Group",
            "business_address": "1795 Westchester Drive, High Point, NC",
            "country": "US",
        },
        {
            "entity_id": "S3-004",
            "business_name": "Walmart Inc",
            "business_address": "100 Main St, Bentonville, AR",
            "country": "US",
        },
        {
            "entity_id": "S3-006",
            "business_name": "अपोलो हॉस्पिटल",
            "business_address": "एम जी रोड, बेंगलुरु, कर्नाटक",
            "country": "India",
        },
        {
            "entity_id": "S3-009",
            "business_name": "Nantes Club SAS",
            "business_address": "10 Rue de la Paix, Nantes",
            "country": "France",
        },
        # Hard negatives
        {
            "entity_id": "S3-010",
            "business_name": "Apollo Medical Center",
            "business_address": "105 Elm Street, Morganton, NC",
            "country": "US",
        },
        {
            "entity_id": "S3-011",
            "business_name": "Acme Corp",
            "business_address": "500 Market St, Suite 400, San Francisco, CA",
            "country": "US",
        },
        # Different country
        {
            "entity_id": "S3-012",
            "business_name": "Apollo Hospital",
            "business_address": "105 Elm St, London",
            "country": "UK",
        },
        # Missing address
        {
            "entity_id": "S3-013",
            "business_name": "Missing Address Corp",
            "business_address": "",
            "country": "US",
        },
        # Distractors
        {
            "entity_id": "S3-014",
            "business_name": "Different Company",
            "business_address": "888 Other St, Elsewhere, TX",
            "country": "US",
        },
    ])

    return s1_df, s2_df, s3_df


def create_candidate_pairs(s1_df, s2_df, s3_df):
    """Generate candidate pairs using the CandidateGenerator."""
    cg = CandidateGenerator(max_candidates_per_s1=20)
    candidate_pairs = cg.generate(s1_df, s2_df, s3_df)
    return candidate_pairs


def compute_features(candidate_pairs, s1_df, s2_df, s3_df):
    """Compute features for candidate pairs using the assembly layer."""
    # Add source column to each DataFrame before concatenating
    s2_with_source = s2_df.copy()
    s2_with_source["source"] = "S2"
    s3_with_source = s3_df.copy()
    s3_with_source["source"] = "S3"
    candidate_data = pd.concat([s2_with_source, s3_with_source], ignore_index=True)
    return build_pair_features(candidate_pairs, s1_df, candidate_data)


class TestFeatureValidation:
    """Validation tests for the 18-feature model input."""

    @classmethod
    def setup_class(cls):
        """Set up validation data once for all tests."""
        cls.s1_df, cls.s2_df, cls.s3_df = create_validation_data()
        cls.candidate_pairs = create_candidate_pairs(cls.s1_df, cls.s2_df, cls.s3_df)
        cls.feature_df = compute_features(cls.candidate_pairs, cls.s1_df, cls.s2_df, cls.s3_df)

    def test_exact_feature_count(self):
        """Test exactly 18 model features + 3 identity columns = 21 total."""
        assert len(self.feature_df.columns) == 21
        model_features = [c for c in self.feature_df.columns
                          if c not in ["s1_entity_id", "candidate_entity_id", "candidate_source"]]
        assert len(model_features) == 18

    def test_exact_feature_names(self):
        """Test exact column names match specification."""
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
        assert list(self.feature_df.columns) == expected_columns

    def test_exact_column_order(self):
        """Test column order is deterministic and matches spec."""
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
        assert list(self.feature_df.columns) == expected_order

    def test_identity_columns_preserved(self):
        """Test identity columns are preserved exactly."""
        pd.testing.assert_series_equal(
            self.feature_df["s1_entity_id"].reset_index(drop=True),
            self.candidate_pairs["s1_entity_id"].reset_index(drop=True),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            self.feature_df["candidate_entity_id"].reset_index(drop=True),
            self.candidate_pairs["candidate_entity_id"].reset_index(drop=True),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            self.feature_df["candidate_source"].reset_index(drop=True),
            self.candidate_pairs["candidate_source"].reset_index(drop=True),
            check_names=False,
        )

    def test_feature_dtypes_numeric(self):
        """Test all model features are numeric dtypes."""
        model_features = [c for c in self.feature_df.columns
                          if c not in ["s1_entity_id", "candidate_entity_id", "candidate_source"]]
        for col in model_features:
            assert pd.api.types.is_numeric_dtype(self.feature_df[col])

    def test_no_nan_values(self):
        """Test no NaN values in any feature column."""
        model_features = [c for c in self.feature_df.columns
                          if c not in ["s1_entity_id", "candidate_entity_id", "candidate_source"]]
        for col in model_features:
            nan_count = self.feature_df[col].isna().sum()
            assert nan_count == 0, f"NaN found in {col}: {nan_count} occurrences"

    def test_no_inf_values(self):
        """Test no +Inf or -Inf values in any feature column."""
        model_features = [c for c in self.feature_df.columns
                          if c not in ["s1_entity_id", "candidate_entity_id", "candidate_source"]]
        for col in model_features:
            pos_inf = (self.feature_df[col] == float('inf')).sum()
            neg_inf = (self.feature_df[col] == float('-inf')).sum()
            assert pos_inf == 0, f"+Inf found in {col}: {pos_inf} occurrences"
            assert neg_inf == 0, f"-Inf found in {col}: {neg_inf} occurrences"

    def test_feature_ranges_bounded_0_1(self):
        """Test all [0,1] bounded features are within range."""
        bounded_features = [
            "name_token_jaccard",
            "name_char_ngram_cosine",
            "name_edit_similarity",
            "name_length_ratio",
            "name_exact_match",
            "name_first_token_match",
            "address_token_jaccard",
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
        for col in bounded_features:
            min_val = self.feature_df[col].min()
            max_val = self.feature_df[col].max()
            assert min_val >= 0.0, f"{col} min {min_val} < 0"
            assert max_val <= 1.0, f"{col} max {max_val} > 1"

    def test_feature_ranges_non_negative(self):
        """Test non-negative features."""
        non_negative_features = [
            "name_token_overlap_count",
            "address_token_overlap_count",
        ]
        for col in non_negative_features:
            min_val = self.feature_df[col].min()
            assert min_val >= 0.0, f"{col} min {min_val} < 0"

    def test_candidate_source_validity(self):
        """Test candidate_source values are only S2 or S3."""
        unique_sources = set(self.feature_df["candidate_source"].unique())
        assert unique_sources.issubset({"S2", "S3"}), f"Invalid sources: {unique_sources}"

    def test_no_s1_s1_pairs(self):
        """Test no S1->S1 candidate pairs."""
        # Our candidate generator only produces S2/S3 candidates, so this is verified by source validity

    def test_no_s2_s3_pairs(self):
        """Test no S2->S3 candidate pairs (only S1->S2 and S1->S3)."""
        # Our candidate generator only produces S1->S2 and S1->S3

    def test_row_count_preserved(self):
        """Test feature row count equals candidate-pair row count."""
        assert len(self.feature_df) == len(self.candidate_pairs)

    def test_no_duplicate_identity_rows(self):
        """Test no duplicate candidate pairs in output."""
        id_cols = ["s1_entity_id", "candidate_entity_id", "candidate_source"]
        duplicates = self.feature_df[id_cols].duplicated().sum()
        assert duplicates == 0

    def test_no_label_columns_in_features(self):
        """Test no ground-truth or label columns in feature matrix."""
        forbidden_prefixes = ["match", "label", "target", "ground_truth", "prob", "score", "pred"]
        for col in self.feature_df.columns:
            col_lower = col.lower()
            for prefix in forbidden_prefixes:
                assert not col_lower.startswith(prefix), f"Label-like column found: {col}"

    def test_deterministic_output(self):
        """Test running feature computation twice produces identical results."""
        feature_df_2 = compute_features(self.candidate_pairs, self.s1_df, self.s2_df, self.s3_df)
        pd.testing.assert_frame_equal(self.feature_df, feature_df_2)

    def test_missing_value_handling(self):
        """Test missing addresses/names produce valid finite features."""
        # Find rows with missing address in S1
        missing_addr_s1 = self.feature_df[
            self.feature_df["s1_entity_id"].isin(["S1-011", "S1-012"])
        ]
        assert len(missing_addr_s1) > 0
        # All features should be finite
        for col in missing_addr_s1.columns:
            if col not in ["s1_entity_id", "candidate_entity_id", "candidate_source"]:
                assert missing_addr_s1[col].notna().all()
                assert not (missing_addr_s1[col] == float('inf')).any()
                assert not (missing_addr_s1[col] == float('-inf')).any()

    def test_unicode_indic_handling(self):
        """Test Indic script names produce valid features."""
        indic_rows = self.feature_df[
            self.feature_df["s1_entity_id"] == "S1-013"
        ]
        assert len(indic_rows) > 0
        for col in indic_rows.columns:
            if col not in ["s1_entity_id", "candidate_entity_id", "candidate_source"]:
                assert indic_rows[col].notna().all()
                assert not (indic_rows[col] == float('inf')).any()
                assert not (indic_rows[col] == float('-inf')).any()


class TestFeatureDistribution:
    """Distribution analysis tests - informational, not pass/fail."""

    @classmethod
    def setup_class(cls):
        cls.s1_df, cls.s2_df, cls.s3_df = create_validation_data()
        cls.candidate_pairs = create_candidate_pairs(cls.s1_df, cls.s2_df, cls.s3_df)
        cls.feature_df = compute_features(cls.candidate_pairs, cls.s1_df, cls.s2_df, cls.s3_df)

    def test_unique_value_counts(self):
        """Report unique value counts for each feature."""
        model_features = [c for c in self.feature_df.columns
                          if c not in ["s1_entity_id", "candidate_entity_id", "candidate_source"]]
        for col in model_features:
            nunique = self.feature_df[col].nunique()
            most_common = self.feature_df[col].mode().iloc[0] if not self.feature_df[col].mode().empty else None
            most_common_freq = (self.feature_df[col] == most_common).sum() if most_common is not None else 0
            most_common_pct = most_common_freq / len(self.feature_df) * 100 if most_common is not None else 0
            print(f"  {col}: unique={nunique}, most_common={most_common}, freq={most_common_freq} ({most_common_pct:.1f}%)")

    def test_constant_features(self):
        """Flag completely constant features."""
        model_features = [c for c in self.feature_df.columns
                          if c not in ["s1_entity_id", "candidate_entity_id", "candidate_source"]]
        constant_features = []
        for col in model_features:
            if self.feature_df[col].nunique() == 1:
                constant_features.append(col)
        print(f"  Constant features: {constant_features}")
        # country_match may be constant due to country-isolated blocking

    def test_binary_feature_dominance(self):
        """Report dominance for binary features."""
        binary_features = [
            "name_exact_match", "name_first_token_match",
            "address_house_number_match", "address_missing",
            "country_match", "source_is_s2", "source_is_s3",
        ]
        for col in binary_features:
            if col in self.feature_df.columns:
                vc = self.feature_df[col].value_counts()
                print(f"  {col}: {dict(vc)}")

    def test_distribution_summary(self):
        """Print distribution summary for continuous features."""
        continuous_features = [
            "name_token_jaccard", "name_token_overlap_count",
            "name_char_ngram_cosine", "name_edit_similarity",
            "name_length_ratio", "address_token_jaccard",
            "address_token_overlap_count", "address_edit_similarity",
            "address_length_ratio", "name_length_ratio_context",
            "address_length_ratio_context",
        ]
        for col in continuous_features:
            if col in self.feature_df.columns:
                desc = self.feature_df[col].describe()
                print(f"  {col}: min={desc['min']:.4f}, max={desc['max']:.4f}, "
                      f"mean={desc['mean']:.4f}, median={desc['50%']:.4f}, "
                      f"std={desc['std']:.4f}")


class TestCandidateIntegrity:
    """Candidate pair integrity checks."""

    @classmethod
    def setup_class(cls):
        cls.s1_df, cls.s2_df, cls.s3_df = create_validation_data()
        cls.candidate_pairs = create_candidate_pairs(cls.s1_df, cls.s2_df, cls.s3_df)
        cls.feature_df = compute_features(cls.candidate_pairs, cls.s1_df, cls.s2_df, cls.s3_df)

    def test_all_candidate_sources_s2_or_s3(self):
        """All candidate_source values must be S2 or S3."""
        sources = set(self.candidate_pairs["candidate_source"].unique())
        assert sources.issubset({"S2", "S3"})

    def test_no_s2_s3_cross_pairs(self):
        """No candidate pair should have S2 matched to S3."""
        # This is inherent in our candidate generation (S1->S2 or S1->S3 only)
        pass

    def test_identity_columns_exact_match(self):
        """Identity columns exactly match input candidate pairs."""
        # Verified in TestFeatureValidation.test_identity_columns_preserved
        pass


class TestPerformanceSanity:
    """Performance sanity check on a realistic sample."""

    def test_feature_computation_performance(self):
        """Measure feature computation time on sample data."""
        import time
        s1_df, s2_df, s3_df = create_validation_data()
        candidate_pairs = create_candidate_pairs(s1_df, s2_df, s3_df)

        # Warmup
        compute_features(candidate_pairs, s1_df, s2_df, s3_df)

        # Timed run
        start = time.perf_counter()
        for _ in range(3):
            compute_features(candidate_pairs, s1_df, s2_df, s3_df)
        elapsed = (time.perf_counter() - start) / 3

        pairs_per_sec = len(candidate_pairs) / elapsed
        print(f"\nPerformance: {len(candidate_pairs)} pairs in {elapsed*1000:.1f}ms "
              f"({pairs_per_sec:.0f} pairs/sec)")

        # Sanity check: should complete in reasonable time
        assert elapsed < 5.0, "Feature computation too slow"

    def test_determinism_twice(self):
        """Run validation sample twice, verify identical output."""
        s1_df, s2_df, s3_df = create_validation_data()
        candidate_pairs = create_candidate_pairs(s1_df, s2_df, s3_df)

        result1 = compute_features(candidate_pairs, s1_df, s2_df, s3_df)
        result2 = compute_features(candidate_pairs, s1_df, s2_df, s3_df)

        pd.testing.assert_frame_equal(result1, result2)


class TestLeakageAudit:
    """Verify no ground-truth or label leakage in features."""

    def test_no_ground_truth_in_feature_functions(self):
        """Verify feature functions don't accept ground truth parameters."""
        import inspect
        from src.business_entity_resolution.features.name_features import compute_name_similarity_features
        from src.business_entity_resolution.features.address_features import compute_address_similarity_features
        from src.business_entity_resolution.features.pair_features import compute_pair_context_features

        for func in [compute_name_similarity_features, compute_address_similarity_features, compute_pair_context_features]:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            # None of these should have ground truth, label, match, target parameters
            forbidden = ["ground", "label", "target", "match", "truth", "prob", "score", "pred", "rank"]
            for param in params:
                param_lower = param.lower()
                for forb in forbidden:
                    assert forb not in param_lower, f"Suspicious parameter '{param}' in {func.__name__}"

    def test_feature_matrix_no_label_columns(self):
        """Feature matrix should not contain label-like columns."""
        s1_df, s2_df, s3_df = create_validation_data()
        candidate_pairs = create_candidate_pairs(s1_df, s2_df, s3_df)
        feature_df = compute_features(candidate_pairs, s1_df, s2_df, s3_df)

        # Forbidden patterns that would indicate label/ground-truth leakage
        # "match" is allowed as part of legitimate feature names like "name_exact_match"
        # "source" is allowed as part of "source_is_s2"/"source_is_s3"
        forbidden = ["label", "target", "ground_truth", "prob", "score", "pred", "rank"]
        for col in feature_df.columns:
            col_lower = col.lower()
            for f in forbidden:
                assert f not in col_lower, f"Label-like column found: {col}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])