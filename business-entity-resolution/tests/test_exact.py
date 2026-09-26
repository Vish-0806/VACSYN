"""
Unit tests and recall validation for deterministic exact blocking (Stage 4).

Tests:
1. Key generation functions (exact name, prefix, house number, address tokens).
2. Country isolation (no cross-country blocking keys).
3. Inverted index storage and retrieval.
4. Maximum bucket size protection against oversized generic terms.
5. Independent evaluation of each exact blocking strategy on sample data.
6. Empirical recall and candidate count measurement.
"""

import sys
from pathlib import Path

# Add src to sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from business_entity_resolution.blocking.exact import (
    generate_exact_name_key,
    generate_name_prefix_key,
    generate_house_number_key,
    generate_address_token_keys,
    ExactIndex,
    block_by_exact_name,
    block_by_name_prefix,
    block_by_house_number,
    block_by_address_tokens,
)


def test_key_generators():
    # Exact name
    k_name = generate_exact_name_key("acme logistics", "US")
    assert k_name == "US::name::acme logistics"
    assert generate_exact_name_key("", "US") == ""
    assert generate_exact_name_key("acme", "") == ""

    # Name prefix
    k_pref = generate_name_prefix_key("acme logistics", "US", prefix_len=4)
    assert k_pref == "US::pref4::acme"
    k_short = generate_name_prefix_key("abc", "India", prefix_len=5)
    assert k_short == "INDIA::pref5::abc"

    # House number
    k_house = generate_house_number_key("1795", "US")
    assert k_house == "US::house::1795"
    assert generate_house_number_key(None, "US") is None
    assert generate_house_number_key("", "US") is None

    # Address tokens
    tokens = ["westchester", "drive", "point"]
    k_addr = generate_address_token_keys(tokens, "US")
    assert "US::addr::westchester" in k_addr
    assert "US::addr::point" in k_addr
    assert len(k_addr) == 3


def test_country_partition_isolation():
    # Identical business name in different countries must yield distinct keys
    k_us = generate_exact_name_key("metro hospital", "US")
    k_in = generate_exact_name_key("metro hospital", "India")
    k_fr = generate_exact_name_key("metro hospital", "France")
    assert k_us != k_in
    assert k_us != k_fr
    assert k_in != k_fr


def test_bucket_size_capping():
    # If a generic term appears more than max_bucket_size times, drop/cap it
    index = ExactIndex(max_bucket_size=3)
    key = "US::addr::street"
    index.add(key, "S2-001")
    index.add(key, "S2-002")
    index.add(key, "S2-003")
    assert len(index.get(key)) == 3

    # Exceeding threshold drops bucket to prevent candidate explosion
    index.add(key, "S2-004")
    assert len(index.get(key)) == 0


def test_independent_blocking_strategies_on_mock_dataset():
    # Setup candidate pool
    s2_records = [
        {"id": "S2-001", "name": "acme supply corp", "house": "105", "tokens": {"elm", "street", "morganton"}, "country": "US"},
        {"id": "S2-002", "name": "acme distribution", "house": "2100", "tokens": {"cameron", "drive"}, "country": "US"},
        {"id": "S2-003", "name": "sunshine dental", "house": "797", "tokens": {"lake", "town", "kolkata"}, "country": "India"},
        {"id": "S2-004", "name": "bordeaux club sarl", "house": "175", "tokens": {"roosevelt", "bordeaux"}, "country": "France"},
    ]

    # Build indexes
    name_idx = ExactIndex(max_bucket_size=100)
    prefix_idx = ExactIndex(max_bucket_size=100)
    house_idx = ExactIndex(max_bucket_size=100)
    addr_idx = ExactIndex(max_bucket_size=100)

    for r in s2_records:
        cid, cty = r["id"], r["country"]
        name_idx.add(generate_exact_name_key(r["name"], cty), cid)
        prefix_idx.add(generate_name_prefix_key(r["name"], cty, prefix_len=4), cid)
        house_idx.add(generate_house_number_key(r["house"], cty), cid)
        addr_idx.add_many(generate_address_token_keys(r["tokens"], cty), cid)

    # Query 1: Exact Name Match
    cands_name = block_by_exact_name("acme supply corp", "US", name_idx)
    assert "S2-001" in cands_name
    assert "S2-002" not in cands_name

    # Query 2: Name Prefix Match (prefix 'acme')
    cands_prefix = block_by_name_prefix("acme holdings", "US", prefix_idx, prefix_len=4)
    assert "S2-001" in cands_prefix
    assert "S2-002" in cands_prefix  # both start with 'acme'
    assert "S2-003" not in cands_prefix  # wrong country and different name

    # Query 3: House Number Match
    cands_house = block_by_house_number("105", "US", house_idx)
    assert "S2-001" in cands_house

    # Query 4: Address Tokens Match
    cands_addr = block_by_address_tokens({"elm", "greensboro"}, "US", addr_idx)
    assert "S2-001" in cands_addr


def test_empirical_recall_on_training_sample():
    """
    Evaluate candidate recall on a sample of 1,000 true training pairs.
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

    # Load S2 sample records into index
    name_idx = ExactIndex(max_bucket_size=500)
    prefix_idx = ExactIndex(max_bucket_size=500)
    s2_count = 0

    with open(train_s2_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts[0] in needed_s2_ids:
                s2_count += 1
                cid, name, addr, cty = parts[0], parts[1].lower().strip(), parts[2].lower().strip(), parts[3]
                name_idx.add(generate_exact_name_key(name, cty), cid)
                prefix_idx.add(generate_name_prefix_key(name, cty, prefix_len=5), cid)
                if s2_count >= len(needed_s2_ids):
                    break

    # Load S1 sample records and query
    s1_records = {}
    with open(train_s1_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts[0] in needed_s1:
                s1_records[parts[0]] = (parts[1].lower().strip(), parts[2].lower().strip(), parts[3])
                if len(s1_records) >= len(needed_s1):
                    break

    # Measure exact name recall & prefix recall
    exact_hits = 0
    prefix_hits = 0
    total_true = len(true_pairs)

    for s1_id, true_s2 in true_pairs:
        if s1_id in s1_records:
            name, addr, cty = s1_records[s1_id]
            # Exact name
            cands_exact = block_by_exact_name(name, cty, name_idx)
            if true_s2 in cands_exact:
                exact_hits += 1
            # Prefix 5
            cands_pref = block_by_name_prefix(name, cty, prefix_idx, prefix_len=5)
            if true_s2 in cands_pref:
                prefix_hits += 1

    exact_recall = (exact_hits / total_true) * 100 if total_true > 0 else 0
    prefix_recall = (prefix_hits / total_true) * 100 if total_true > 0 else 0

    print(f"Sample Exact Blocking Results (Total Pairs: {total_true}):")
    print(f"  Exact Name Recall: {exact_recall:.2f}% ({exact_hits}/{total_true})")
    print(f"  Prefix-5 Recall:   {prefix_recall:.2f}% ({prefix_hits}/{total_true})")

    assert exact_hits > 0
    assert prefix_hits >= exact_hits


if __name__ == "__main__":
    test_key_generators()
    test_country_partition_isolation()
    test_bucket_size_capping()
    test_independent_blocking_strategies_on_mock_dataset()
    test_empirical_recall_on_training_sample()
    print("All Stage 4 exact blocking tests passed successfully!")
