"""
Contract and interface validation tests for Member 1 (Preprocessing and Blocking).
"""

import inspect
import sys
from pathlib import Path

# Add src to sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

def test_preprocessing_imports_and_signatures():
    from business_entity_resolution.preprocessing import (
        normalize_business_name,
        tokenize_business_name,
        tokenize_name,
        extract_legal_suffixes,
        normalize_address,
        extract_house_number,
        tokenize_address,
    )
    
    # Check callable
    for fn in [
        normalize_business_name,
        tokenize_business_name,
        tokenize_name,
        extract_legal_suffixes,
        normalize_address,
        extract_house_number,
        tokenize_address,
    ]:
        assert callable(fn), f"{fn} must be callable"

    # Check signatures
    sig_norm_name = inspect.signature(normalize_business_name)
    assert "name" in sig_norm_name.parameters
    assert "strip_legal" in sig_norm_name.parameters

    sig_tok_name = inspect.signature(tokenize_business_name)
    assert "name" in sig_tok_name.parameters

    sig_norm_addr = inspect.signature(normalize_address)
    assert "address" in sig_norm_addr.parameters

    sig_house_num = inspect.signature(extract_house_number)
    assert "address" in sig_house_num.parameters

    sig_tok_addr = inspect.signature(tokenize_address)
    assert "address" in sig_tok_addr.parameters


def test_blocking_imports_and_signatures():
    from business_entity_resolution.blocking import (
        CandidateGenerator,
        generate_candidates,
    )
    from business_entity_resolution.blocking.exact import (
        generate_exact_name_key,
        generate_name_prefix_key,
        generate_house_number_key,
        generate_address_token_keys,
    )
    from business_entity_resolution.blocking.tfidf import SparseTFIDFBlocker
    from business_entity_resolution.blocking.fuzzy import (
        generate_token_overlap_candidates,
        filter_candidates_by_edit_distance,
    )

    for fn in [
        generate_candidates,
        generate_exact_name_key,
        generate_name_prefix_key,
        generate_house_number_key,
        generate_address_token_keys,
        generate_token_overlap_candidates,
        filter_candidates_by_edit_distance,
    ]:
        assert callable(fn), f"{fn} must be callable"

    # Check CandidateGenerator class
    cg = CandidateGenerator(max_candidates_per_s1=25)
    assert hasattr(cg, "generate")
    assert callable(cg.generate)
    
    # Check SparseTFIDFBlocker class
    blocker = SparseTFIDFBlocker()
    assert hasattr(blocker, "fit_transform_corpus")
    assert hasattr(blocker, "query_top_k")


if __name__ == "__main__":
    print("Testing preprocessing imports and signatures...")
    test_preprocessing_imports_and_signatures()
    print("Testing blocking imports and signatures...")
    test_blocking_imports_and_signatures()
    print("All Stage 1 contract checks passed successfully!")
