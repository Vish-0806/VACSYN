"""
Unit and integration tests for M4 Integration & Submission Pipeline.

Covers:
1. Candidate aggregation (pairwise -> comma-separated list).
2. Zero-candidate singleton preservation.
3. Duplicate candidate removal.
4. Complete S1 entity preservation.
5. Prediction subset consistency validation (matched ⊆ candidate).
6. Matching output formatting (source1_entity_id, matched_entity_ids).
7. Candidate output formatting (source1_entity_id, candidate_entity_ids).
8. S1 chunked candidate generation.
9. Official submission validator compliance.
"""

from pathlib import Path
import pandas as pd
import pytest

from business_entity_resolution.pipeline import (
    format_candidate_pairs_submission,
    format_matching_results,
    validate_match_candidate_consistency,
    generate_candidates_chunked,
    run_submission_validator,
    MATCHING_HEADER,
    CANDIDATE_HEADER,
)


def test_candidate_aggregation():
    """Test 1: Multiple candidates for the same S1 become comma-separated."""
    pairs_df = pd.DataFrame([
        {"s1_entity_id": "S1_A", "candidate_entity_id": "S2_1", "candidate_source": "S2"},
        {"s1_entity_id": "S1_A", "candidate_entity_id": "S3_4", "candidate_source": "S3"},
        {"s1_entity_id": "S1_B", "candidate_entity_id": "S2_8", "candidate_source": "S2"},
    ])
    all_s1 = ["S1_A", "S1_B"]

    sub_df = format_candidate_pairs_submission(pairs_df, all_s1)
    res = dict(zip(sub_df["source1_entity_id"], sub_df["candidate_entity_ids"]))

    assert res["S1_A"] == "S2_1,S3_4"
    assert res["S1_B"] == "S2_8"


def test_zero_candidates_singleton():
    """Test 2: Zero candidates produce an empty string, not a missing row."""
    pairs_df = pd.DataFrame([
        {"s1_entity_id": "S1_B", "candidate_entity_id": "S2_1", "candidate_source": "S2"},
    ])
    all_s1 = ["S1_A", "S1_B"]

    sub_df = format_candidate_pairs_submission(pairs_df, all_s1)
    res = dict(zip(sub_df["source1_entity_id"], sub_df["candidate_entity_ids"]))

    assert res["S1_A"] == ""
    assert res["S1_B"] == "S2_1"


def test_duplicate_candidate_removal():
    """Test 3: Internal duplicates for an S1 entity are removed preserving order."""
    pairs_df = pd.DataFrame([
        {"s1_entity_id": "S1_A", "candidate_entity_id": "S2_1", "candidate_source": "S2"},
        {"s1_entity_id": "S1_A", "candidate_entity_id": "S2_1", "candidate_source": "S2"},
        {"s1_entity_id": "S1_A", "candidate_entity_id": "S3_4", "candidate_source": "S3"},
    ])
    all_s1 = ["S1_A"]

    sub_df = format_candidate_pairs_submission(pairs_df, all_s1)
    assert sub_df.iloc[0]["candidate_entity_ids"] == "S2_1,S3_4"


def test_every_s1_preserved():
    """Test 4: Every S1 in the master reference list is guaranteed to appear."""
    all_s1 = ["S1_A", "S1_B", "S1_C"]
    pairs_df = pd.DataFrame([
        {"s1_entity_id": "S1_A", "candidate_entity_id": "S2_1", "candidate_source": "S2"},
        {"s1_entity_id": "S1_C", "candidate_entity_id": "S3_9", "candidate_source": "S3"},
    ])

    sub_df = format_candidate_pairs_submission(pairs_df, all_s1)
    assert len(sub_df) == 3
    assert list(sub_df["source1_entity_id"]) == ["S1_A", "S1_B", "S1_C"]
    assert sub_df.iloc[1]["candidate_entity_ids"] == ""


def test_prediction_subset_validation():
    """Test 5: Predictions must be a subset of the candidate set."""
    # Valid case
    cand_df = pd.DataFrame([
        {"source1_entity_id": "S1_A", "candidate_entity_ids": "S2_1,S3_4"},
        {"source1_entity_id": "S1_B", "candidate_entity_ids": "S2_2"},
    ])
    matching_valid = pd.DataFrame([
        {"source1_entity_id": "S1_A", "matched_entity_ids": "S2_1"},
        {"source1_entity_id": "S1_B", "matched_entity_ids": ""},
    ])
    assert validate_match_candidate_consistency(matching_valid, cand_df) is True

    # Invalid case: S2_99 is not in S1_A's candidate set
    matching_invalid = pd.DataFrame([
        {"source1_entity_id": "S1_A", "matched_entity_ids": "S2_99"},
        {"source1_entity_id": "S1_B", "matched_entity_ids": ""},
    ])
    with pytest.raises(ValueError) as excinfo:
        validate_match_candidate_consistency(matching_invalid, cand_df)
    assert "S2_99" in str(excinfo.value)
    assert "S1_A" in str(excinfo.value)


def test_matching_output_formatting():
    """Test 6: matching_results.tsv schema and singleton handling."""
    predictions = {
        "S1_1": ["S2_10", "S3_20"],
        "S1_2": [],  # Singleton
        "S1_3": "S2_30,S2_30",  # Duplicate candidate test
    }
    all_s1 = ["S1_1", "S1_2", "S1_3", "S1_4"]

    matching_df = format_matching_results(predictions, all_s1)
    assert list(matching_df.columns) == MATCHING_HEADER
    assert len(matching_df) == 4

    res = dict(zip(matching_df["source1_entity_id"], matching_df["matched_entity_ids"]))
    assert res["S1_1"] == "S2_10,S3_20"
    assert res["S1_2"] == ""
    assert res["S1_3"] == "S2_30"
    assert res["S1_4"] == ""


def test_candidate_output_formatting():
    """Test 7: candidate_pairs.tsv schema and ordering."""
    pairs_df = pd.DataFrame([
        {"s1_entity_id": "S1_X", "candidate_entity_id": "S2_100", "candidate_source": "S2"},
    ])
    all_s1 = ["S1_X", "S1_Y"]

    cand_df = format_candidate_pairs_submission(pairs_df, all_s1)
    assert list(cand_df.columns) == CANDIDATE_HEADER
    assert len(cand_df) == 2
    assert cand_df.iloc[0]["candidate_entity_ids"] == "S2_100"
    assert cand_df.iloc[1]["candidate_entity_ids"] == ""


def test_chunked_s1_processing():
    """Test 8: S1 queries are processed in multiple chunks without loss."""
    # Create 5 synthetic S1 entities across US and INDIA
    s1_df = pd.DataFrame([
        {"entity_id": "S1-1", "business_name": "Walmart Supercenter", "business_address": "100 Main St", "country": "US"},
        {"entity_id": "S1-2", "business_name": "Walmart Store", "business_address": "100 Main St", "country": "US"},
        {"entity_id": "S1-3", "business_name": "Target Express", "business_address": "200 State St", "country": "US"},
        {"entity_id": "S1-4", "business_name": "Apollo Pharmacy", "business_address": "12-34 Road No 36", "country": "INDIA"},
        {"entity_id": "S1-5", "business_name": "Apollo Clinic", "business_address": "12-34 Jubilee Hills", "country": "INDIA"},
    ])

    s2_df = pd.DataFrame([
        {"entity_id": "S2-10", "business_name": "Walmart Inc", "business_address": "100 Main St", "country": "US"},
        {"entity_id": "S2-40", "business_name": "Apollo Hospital", "business_address": "12-34 Road No 36", "country": "INDIA"},
    ])

    s3_df = pd.DataFrame([
        {"entity_id": "S3-100", "business_name": "Target Store", "business_address": "200 State St", "country": "US"},
    ])

    # Run with small chunk_size=2 to force multiple chunks
    cands_chunked = generate_candidates_chunked(
        s1_df=s1_df,
        s2_df=s2_df,
        s3_df=s3_df,
        chunk_size=2,
        max_candidates_per_s1=10,
    )

    assert not cands_chunked.empty
    assert list(cands_chunked.columns) == ["s1_entity_id", "candidate_entity_id", "candidate_source"]

    # Verify that S1-1 in US matched S2-10
    s1_1_matches = cands_chunked[cands_chunked["s1_entity_id"] == "S1-1"]
    assert "S2-10" in s1_1_matches["candidate_entity_id"].values

    # Verify that S1-4 in INDIA matched S2-40
    s1_4_matches = cands_chunked[cands_chunked["s1_entity_id"] == "S1-4"]
    assert "S2-40" in s1_4_matches["candidate_entity_id"].values


def test_official_validator_compliance(tmp_path: Path):
    """Test 9: Verify generated files pass student_resource/utils/validate_submission.py."""
    # Create synthetic test dataset in tmp_path
    test_dir = tmp_path / "test"
    test_dir.mkdir()

    s1_content = "entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-001\tAcme Inc\t100 Main St\tUS\nS1-002\tBeta Corp\t200 Elm St\tUS\n"
    (test_dir / "test_source1.tsv").write_text(s1_content, encoding="utf-8")

    s2_content = "entity_id\tbusiness_name\tbusiness_address\tcountry\nS2-001\tAcme\t100 Main St\tUS\n"
    (test_dir / "test_source2.tsv").write_text(s2_content, encoding="utf-8")

    s3_content = "entity_id\tbusiness_name\tbusiness_address\tcountry\nS3-001\tBeta\t200 Elm St\tUS\n"
    (test_dir / "test_source3.tsv").write_text(s3_content, encoding="utf-8")

    # Generate compliant candidate and matching files
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    all_s1 = ["S1-001", "S1-002"]

    # Candidates: S1-001 has S2-001, S1-002 has S3-001
    cand_pairs = pd.DataFrame([
        {"s1_entity_id": "S1-001", "candidate_entity_id": "S2-001", "candidate_source": "S2"},
        {"s1_entity_id": "S1-002", "candidate_entity_id": "S3-001", "candidate_source": "S3"},
    ])
    cand_sub = format_candidate_pairs_submission(cand_pairs, all_s1)
    cand_path = out_dir / "candidate_pairs.tsv"
    cand_sub.to_csv(cand_path, sep="\t", index=False)

    # Matching: S1-001 matched S2-001, S1-002 is singleton
    match_preds = {"S1-001": ["S2-001"], "S1-002": []}
    match_sub = format_matching_results(match_preds, all_s1)
    match_path = out_dir / "matching_results.tsv"
    match_sub.to_csv(match_path, sep="\t", index=False)

    # Consistency check
    assert validate_match_candidate_consistency(match_sub, cand_sub) is True

    # Run official validator
    code = run_submission_validator(
        matching_path=match_path,
        candidate_path=cand_path,
        test_dir=test_dir,
        check_ids=True,
    )
    assert code == 0, "Submission validator returned non-zero exit code"
