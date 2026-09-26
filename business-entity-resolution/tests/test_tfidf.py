"""
Unit and integration tests for Sparse TF-IDF Blocker and CountrySparseTFIDFBlocker.
"""

import pytest
from business_entity_resolution.blocking.tfidf import (
    SparseTFIDFBlocker,
    CountrySparseTFIDFBlocker,
)


def test_sparse_tfidf_empty():
    blocker = SparseTFIDFBlocker()
    mat = blocker.fit_transform_corpus([], [])
    assert mat.shape == (0, 0)
    res = blocker.query_top_k(["test"], top_k=5)
    assert res == {0: []}
    cand_ids = blocker.query_candidate_ids(["test"])
    assert cand_ids == [[]]


def test_sparse_tfidf_typo_fuzzy_retrieval():
    corpus_texts = [
        "walmart supercenter",
        "target store",
        "best buy electronics",
        "apollo pharmacy",
        "home depot hardware",
    ]
    candidate_ids = ["c1", "c2", "c3", "c4", "c5"]

    blocker = SparseTFIDFBlocker(ngram_range=(3, 4), min_df=1)
    blocker.fit_transform_corpus(corpus_texts, candidate_ids)

    # Queries with typos/variations
    queries = [
        "wal-mart super center",  # Similar to c1
        "appolo pharmacie",        # Similar to c4
        "completely unrelated xyz" # No match
    ]

    cands = blocker.query_candidate_ids(queries, top_k=3, min_similarity=0.25)

    assert "c1" in cands[0]
    assert "c4" in cands[1]
    assert len(cands[2]) == 0 or "c1" not in cands[2]


def test_sparse_tfidf_top_k_ordering():
    corpus_texts = [
        "starbucks coffee company",
        "starbucks reserve roastery",
        "peets coffee and tea",
        "dunkin donuts coffee",
    ]
    candidate_ids = ["sb1", "sb2", "peet", "dunk"]

    blocker = SparseTFIDFBlocker(ngram_range=(3, 4), min_df=1)
    blocker.fit_transform_corpus(corpus_texts, candidate_ids)

    results = blocker.query_top_k(["starbucks coffee"], top_k=2, min_similarity=0.2)
    top_matches = results[0]
    assert len(top_matches) <= 2
    # Verify that the scores are sorted descending
    if len(top_matches) > 1:
        assert top_matches[0][1] >= top_matches[1][1]


def test_country_sparse_tfidf_partitioning():
    country_blocker = CountrySparseTFIDFBlocker(ngram_range=(3, 4), top_k=5, min_similarity=0.2)

    # US candidates
    country_blocker.fit_country_corpus(
        country="US",
        texts=["walgreens store 101", "cvs pharmacy 202"],
        candidate_ids=["us_1", "us_2"],
    )

    # India candidates
    country_blocker.fit_country_corpus(
        country="INDIA",
        texts=["medplus pharmacy bangalore", "apollo pharmacy hyderabad"],
        candidate_ids=["in_1", "in_2"],
    )

    # Query US
    us_cands = country_blocker.query("US", ["walgreens store"])
    assert "us_1" in us_cands[0]
    assert "in_1" not in us_cands[0]
    assert "in_2" not in us_cands[0]

    # Query India
    in_cands = country_blocker.query("INDIA", ["apollo pharmacy"])
    assert "in_2" in in_cands[0]
    assert "us_1" not in in_cands[0]

    # Query Unseen Country (e.g. France before fit)
    fr_cands = country_blocker.query("FRANCE", ["boulangerie patisserie"])
    assert fr_cands == [[]]
