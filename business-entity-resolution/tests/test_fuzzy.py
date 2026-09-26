"""
Unit tests and candidate recall validation for scalable name-token blocking (Stage 5).

Tests:
1. NameTokenBlocker inverted indexing and query retrieval.
2. Exclusion of generic corporate and trade stop terms.
3. Rarest-first (IDF-style) token ranking.
4. Token overlap thresholding and edit distance filtering.
5. Empirical recall and candidate count measurement on training sample.
"""

import sys
from pathlib import Path

# Add src to sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from business_entity_resolution.preprocessing.normalize import tokenize_business_name
from business_entity_resolution.blocking.fuzzy import (
    NameTokenBlocker,
    generate_token_overlap_candidates,
    filter_candidates_by_edit_distance,
    GENERIC_NAME_TERMS,
)


def test_generic_term_filtering():
    blocker = NameTokenBlocker(max_token_doc_freq=10)
    # Generic terms must not be indexed
    tokens = ["primary", "care", "group", "llc", "pediatric"]
    blocker.add_entity("S2-001", tokens, "US")

    # 'care', 'group', 'llc' are in GENERIC_NAME_TERMS
    assert "US::care" not in blocker.inverted_index
    assert "US::group" not in blocker.inverted_index
    assert "US::llc" not in blocker.inverted_index

    # 'primary' and 'pediatric' are indexed
    assert "US::primary" in blocker.inverted_index
    assert "US::pediatric" in blocker.inverted_index
    assert "S2-001" in blocker.inverted_index["US::primary"]


def test_rarest_first_querying():
    blocker = NameTokenBlocker(max_token_doc_freq=100, max_candidates_per_entity=5)

    # Add multiple entities with common token 'apex'
    blocker.add_entity("S2-001", ["apex", "summit"], "US")
    blocker.add_entity("S2-002", ["apex", "logistics"], "US")
    blocker.add_entity("S2-003", ["apex", "ventures"], "US")
    blocker.add_entity("S2-004", ["apex", "partners"], "US")

    # 'summit' appears once (rarest), 'apex' appears 4 times
    cands = blocker.query_candidates(["apex", "summit"], "US")
    # S2-001 matches both, and 'summit' gives highest priority
    assert cands[0] == "S2-001"
    assert len(cands) <= 5


def test_token_overlap_candidates():
    index = {
        "orville": ["S2-101", "S2-102"],
        "avenue": ["S2-101", "S2-103"],
        "market": ["S2-102"],
    }
    # Require at least 2 shared tokens
    cands_2 = generate_token_overlap_candidates({"orville", "avenue"}, index, min_shared_tokens=2)
    assert cands_2 == ["S2-101"]

    # Require at least 1 shared token
    cands_1 = generate_token_overlap_candidates({"orville", "market"}, index, min_shared_tokens=1)
    assert "S2-102" in cands_1
    assert "S2-101" in cands_1


def test_edit_distance_filtering():
    candidate_texts = {
        "S2-001": "holloway peak seafood",
        "S2-002": "holloway peak seafood inc",
        "S2-003": "random completely different",
    }
    filtered = filter_candidates_by_edit_distance(
        "holloway peak seafood", ["S2-001", "S2-002", "S2-003"], candidate_texts, threshold=0.7
    )
    assert "S2-001" in filtered
    assert "S2-002" in filtered
    assert "S2-003" not in filtered


def test_empirical_name_token_recall_on_training_sample():
    """
    Measure candidate recall and candidate set size of NameTokenBlocker on training sample.
    """
    train_gt_path = Path("student_resource/dataset/train/train_ground_truth.tsv")
    train_s1_path = Path("student_resource/dataset/train/train_source1.tsv")
    train_s2_path = Path("student_resource/dataset/train/train_source2.tsv")

    if not train_gt_path.exists() or not train_s1_path.exists() or not train_s2_path.exists():
        print("Dataset files not found locally, skipping dataset sample test.")
        return

    # Sample 500 S1 records with known S2 matches
    needed_s1 = {}
    needed_s2_ids = set()
    true_pairs = []

    with open(train_gt_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            s1_id, _, rest = line.rstrip("\r\n").partition("\t")
            if rest:
                s2_mids = [m for m in rest.split(",") if m.startswith("S2-")]
                if s2_mids:
                    needed_s1[s1_id] = s2_mids
                    for m in s2_mids:
                        needed_s2_ids.add(m)
                        true_pairs.append((s1_id, m))
                    if len(needed_s1) >= 500:
                        break

    # Build NameTokenBlocker from S2 sample records
    blocker = NameTokenBlocker(max_token_doc_freq=500, max_candidates_per_entity=50)
    s2_count = 0

    with open(train_s2_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts[0] in needed_s2_ids:
                s2_count += 1
                cid, name, addr, cty = parts[0], parts[1], parts[2], parts[3]
                tokens = tokenize_business_name(name)
                blocker.add_entity(cid, tokens, cty)
                if s2_count >= len(needed_s2_ids):
                    break

    # Load S1 sample records and query
    s1_records = {}
    with open(train_s1_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts[0] in needed_s1:
                s1_records[parts[0]] = (parts[1], parts[2], parts[3])
                if len(s1_records) >= len(needed_s1):
                    break

    hits = 0
    total_candidates_generated = 0
    total_true = len(true_pairs)

    for s1_id, true_s2 in true_pairs:
        if s1_id in s1_records:
            name, addr, cty = s1_records[s1_id]
            q_tokens = tokenize_business_name(name)
            cands = blocker.query_candidates(q_tokens, cty)
            total_candidates_generated += len(cands)
            if true_s2 in cands:
                hits += 1

    recall = (hits / total_true) * 100 if total_true > 0 else 0
    avg_cands = total_candidates_generated / len(true_pairs) if true_pairs else 0

    print(f"Name Token Blocking Results (Sample size: {total_true} pairs):")
    print(f"  Candidate Recall: {recall:.2f}% ({hits}/{total_true})")
    print(f"  Avg candidates per query: {avg_cands:.2f}")

    # Name token blocking should achieve >80% recall on this sample
    assert recall >= 80.0


if __name__ == "__main__":
    test_generic_term_filtering()
    test_rarest_first_querying()
    test_token_overlap_candidates()
    test_edit_distance_filtering()
    test_empirical_name_token_recall_on_training_sample()
    print("All Stage 5 name token blocking tests passed successfully!")
