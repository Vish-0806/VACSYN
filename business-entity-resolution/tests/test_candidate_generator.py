"""
Unit and integration tests for Composite Candidate Generator.
"""

from pathlib import Path
import pandas as pd
import pytest

from business_entity_resolution.blocking.candidate_generator import (
    CandidateGenerator,
    generate_candidates,
)


def test_candidate_generator_empty():
    cg = CandidateGenerator()
    s1 = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
    s2 = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
    s3 = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])

    res = cg.generate(s1, s2, s3)
    assert list(res.columns) == ["s1_entity_id", "candidate_entity_id", "candidate_source"]
    assert len(res) == 0


def test_candidate_generator_multi_signal_and_sources(tmp_path: Path):
    s1_df = pd.DataFrame([
        {
            "entity_id": "S1-1",
            "business_name": "Walmart Supercenter",
            "business_address": "100 Main St, Bentonville",
            "country": "US",
        },
        {
            "entity_id": "S1-2",
            "business_name": "Apollo Pharmacy",
            "business_address": "12-34 Jubilee Hills, Hyderabad",
            "country": "INDIA",
        },
        {
            "entity_id": "S1-3",
            "business_name": "Cafe de Paris",
            "business_address": "15 Rue de Rivoli, Paris",
            "country": "FRANCE",
        },
    ])

    s2_df = pd.DataFrame([
        {
            "entity_id": "S2-10",
            "business_name": "Wal-Mart Super Center",  # Fuzzy / TF-IDF variation
            "business_address": "100 Main Street",
            "country": "US",
        },
        {
            "entity_id": "S2-20",
            "business_name": "Apollo Hospital & Pharmacy",
            "business_address": "12-34 Road No 36 Jubilee Hills",
            "country": "INDIA",
        },
    ])

    s3_df = pd.DataFrame([
        {
            "entity_id": "S3-100",
            "business_name": "Walmart Inc",
            "business_address": "100 Main St",
            "country": "US",
        },
        {
            "entity_id": "S3-300",
            "business_name": "Bistro Cafe de Paris",
            "business_address": "15 Rue de Rivoli",
            "country": "FRANCE",
        },
        # Candidate with mismatched country (US name in India)
        {
            "entity_id": "S3-999",
            "business_name": "Walmart Supercenter",
            "business_address": "100 Main St",
            "country": "INDIA",
        },
    ])

    cg = CandidateGenerator(max_candidates_per_s1=10)
    cand_pairs = cg.generate(s1_df, s2_df, s3_df)

    assert list(cand_pairs.columns) == ["s1_entity_id", "candidate_entity_id", "candidate_source"]
    assert len(cand_pairs) > 0

    s1_1_cands = cand_pairs[cand_pairs["s1_entity_id"] == "S1-1"]
    assert "S2-10" in s1_1_cands["candidate_entity_id"].values
    assert "S3-100" in s1_1_cands["candidate_entity_id"].values
    # Check that candidate source is correctly tracked
    assert s1_1_cands[s1_1_cands["candidate_entity_id"] == "S2-10"]["candidate_source"].iloc[0] == "S2"
    assert s1_1_cands[s1_1_cands["candidate_entity_id"] == "S3-100"]["candidate_source"].iloc[0] == "S3"

    # Cross-country isolation: S3-999 is in INDIA, must never match S1-1 in US
    assert "S3-999" not in s1_1_cands["candidate_entity_id"].values

    # Check India matching
    s1_2_cands = cand_pairs[cand_pairs["s1_entity_id"] == "S1-2"]
    assert "S2-20" in s1_2_cands["candidate_entity_id"].values
    assert s1_2_cands[s1_2_cands["candidate_entity_id"] == "S2-20"]["candidate_source"].iloc[0] == "S2"

    # Check France matching
    s1_3_cands = cand_pairs[cand_pairs["s1_entity_id"] == "S1-3"]
    assert "S3-300" in s1_3_cands["candidate_entity_id"].values

    # Test file output with generate_candidates
    out_tsv = tmp_path / "candidate_pairs.tsv"
    res_file = generate_candidates(s1_df, s2_df, s3_df, output_path=out_tsv)
    assert out_tsv.exists()

    loaded = pd.read_csv(out_tsv, sep="\t")
    assert len(loaded) == len(cand_pairs)
    assert list(loaded.columns) == ["s1_entity_id", "candidate_entity_id", "candidate_source"]


def test_candidate_budget_cap():
    s1_df = pd.DataFrame([
        {
            "entity_id": "S1-1",
            "business_name": "Acme Store",
            "business_address": "123 Market St",
            "country": "US",
        }
    ])

    # 15 potential matching candidates
    s2_records = [
        {
            "entity_id": f"S2-{i}",
            "business_name": f"Acme Store {i}",
            "business_address": f"123 Market St Suite {i}",
            "country": "US",
        }
        for i in range(15)
    ]
    s2_df = pd.DataFrame(s2_records)

    cg = CandidateGenerator(max_candidates_per_s1=5)
    cand_pairs = cg.generate(s1_df, s2_df, None)

    # Budget must not exceed 5
    assert len(cand_pairs) <= 5
