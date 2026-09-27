"""
Unit and integration tests for distributed training shard processing.

Tests:
1. Deterministic shard assignment.
2. Shard 0 vs Shard 1 mutual exclusivity and completeness (no overlap, every S1 in exactly one shard).
3. Candidate labeling accuracy against ground truth.
4. Completeness and exact naming of all 18 model features.
5. Final schema compliance and absence of NaN / Inf values.
"""

import pandas as pd
import pytest

from business_entity_resolution.analysis.process_training_shard import (
    get_shard,
    EXPECTED_FEATURE_COLUMNS,
    FINAL_SCHEMA_COLUMNS,
    validate_training_shard,
    build_candidate_record_lookups,
)
from business_entity_resolution.features.pair_features import PairFeatureExtractor


class TestDeterministicSharding:
    """Tests for MD5 deterministic S1 sharding logic."""

    def test_deterministic_assignment(self):
        """Verify hash is deterministic across multiple evaluations."""
        test_ids = ["S1-001", "S1-925783039", "S1-773889195", "S1-abc-xyz"]
        for sid in test_ids:
            shard_a = get_shard(sid)
            shard_b = get_shard(sid)
            assert shard_a == shard_b, f"Sharding not deterministic for {sid}"
            assert shard_a in (0, 1), f"Shard out of range for {sid}: {shard_a}"

    def test_no_overlap_and_completeness(self):
        """Verify partition is mutually exclusive and exhaustive across sample entity IDs."""
        sample_ids = [f"S1-{i:07d}" for i in range(1, 2001)]
        shard_0_set = {sid for sid in sample_ids if get_shard(sid) == 0}
        shard_1_set = {sid for sid in sample_ids if get_shard(sid) == 1}

        # 1. No overlap
        overlap = shard_0_set.intersection(shard_1_set)
        assert len(overlap) == 0, f"Overlapping IDs found between shard 0 and 1: {overlap}"

        # 2. Complete coverage
        union = shard_0_set.union(shard_1_set)
        assert union == set(sample_ids), "Some IDs were omitted from both shards"

        # 3. Approximately balanced 50/50 split (allow +/- 10% on 2000 samples)
        ratio_0 = len(shard_0_set) / len(sample_ids)
        assert 0.40 <= ratio_0 <= 0.60, f"Sharding split is biased: {ratio_0:.2%}"


class TestCandidateLabelingAndFeatures:
    """Tests for feature assembly and ground-truth labeling contracts."""

    @pytest.fixture
    def synthetic_data(self):
        s1_df = pd.DataFrame([
            {"entity_id": "S1-001", "business_name": "Apollo Clinic", "business_address": "123 Main St, New York, NY", "country": "US"},
            {"entity_id": "S1-002", "business_name": "Bharat Petrol", "business_address": "MG Road, Bengaluru, Karnataka", "country": "IN"},
        ])
        s2_df = pd.DataFrame([
            {"entity_id": "S2-001", "business_name": "Apollo Clinic LLC", "business_address": "123 Main St, New York, NY", "country": "US", "source": "S2"},
            {"entity_id": "S2-002", "business_name": "Random Corp", "business_address": "456 Elm St, Dallas, TX", "country": "US", "source": "S2"},
        ])
        s3_df = pd.DataFrame([
            {"entity_id": "S3-001", "business_name": "Bharat Petroleum", "business_address": "MG Road, Bangalore, KA", "country": "IN", "source": "S3"},
        ])
        candidate_pairs_df = pd.DataFrame([
            {"s1_entity_id": "S1-001", "candidate_entity_id": "S2-001", "candidate_source": "S2"},
            {"s1_entity_id": "S1-001", "candidate_entity_id": "S2-002", "candidate_source": "S2"},
            {"s1_entity_id": "S1-002", "candidate_entity_id": "S3-001", "candidate_source": "S3"},
        ])
        ground_truth = {
            "S1-001": {"S2-001"},  # S2-001 is true match, S2-002 is false match (negative)
            "S1-002": {"S3-001"},  # S3-001 is true match
        }
        return s1_df, s2_df, s3_df, candidate_pairs_df, ground_truth

    def test_feature_assembly_and_18_features(self, synthetic_data):
        s1_df, s2_df, s3_df, cand_df, _ = synthetic_data
        s1_rec, s2_rec, s3_rec = build_candidate_record_lookups(cand_df, s1_df, s2_df, s3_df)

        extractor = PairFeatureExtractor()
        features_df = extractor.extract_features(cand_df, s1_rec, s2_rec, s3_rec)

        # Verify all 18 features exist
        for col in EXPECTED_FEATURE_COLUMNS:
            assert col in features_df.columns, f"Feature column missing: {col}"
            assert not features_df[col].isna().any(), f"NaN in feature: {col}"
            assert not (features_df[col] == float("inf")).any(), f"Inf in feature: {col}"

    def test_ground_truth_labeling(self, synthetic_data):
        s1_df, s2_df, s3_df, cand_df, ground_truth = synthetic_data
        s1_rec, s2_rec, s3_rec = build_candidate_record_lookups(cand_df, s1_df, s2_df, s3_df)

        extractor = PairFeatureExtractor()
        features_df = extractor.extract_features(cand_df, s1_rec, s2_rec, s3_rec)

        # Assign labels
        labels = []
        for s1_id, cand_id in zip(features_df["s1_entity_id"], features_df["candidate_entity_id"]):
            true_mids = ground_truth.get(str(s1_id).strip(), set())
            labels.append(1 if str(cand_id).strip() in true_mids else 0)

        features_df["label"] = labels

        # Pair 1: S1-001 and S2-001 -> match (1)
        # Pair 2: S1-001 and S2-002 -> negative (0)
        # Pair 3: S1-002 and S3-001 -> match (1)
        expected_labels = [1, 0, 1]
        assert features_df["label"].tolist() == expected_labels

    def test_final_schema_and_validation(self, synthetic_data):
        s1_df, s2_df, s3_df, cand_df, ground_truth = synthetic_data
        s1_rec, s2_rec, s3_rec = build_candidate_record_lookups(cand_df, s1_df, s2_df, s3_df)

        extractor = PairFeatureExtractor()
        features_df = extractor.extract_features(cand_df, s1_rec, s2_rec, s3_rec)
        features_df["label"] = [1, 0, 1]

        final_df = features_df[FINAL_SCHEMA_COLUMNS]
        assert list(final_df.columns) == FINAL_SCHEMA_COLUMNS
        assert len(final_df.columns) == 22  # 3 ids + 18 features + 1 label

        # Check validation
        target_shard = get_shard("S1-001")
        # Filter final_df to only entities with that shard for the validation test
        valid_s1_ids = {sid for sid in s1_df["entity_id"] if get_shard(sid) == target_shard}
        shard_df = final_df[final_df["s1_entity_id"].isin(valid_s1_ids)]

        stats = validate_training_shard(shard_df, target_shard, ground_truth, valid_s1_ids)
        assert stats["shard_correctness"] is True
        assert stats["no_duplicates"] is True
        assert stats["no_missing_labels"] is True
        assert stats["features_complete"] is True
