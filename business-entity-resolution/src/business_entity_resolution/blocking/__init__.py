"""
Blocking and Candidate Generation Module.

Responsible for reducing the NxM comparison space to a high-recall, manageable candidate set
using country partitioning, multi-signal indexing, sparse TF-IDF, and candidate aggregation.

Ownership: Member 1 (Preprocessing & Blocking).
"""

from .candidate_generator import CandidateGenerator, generate_candidates
from .exact import (
    ExactIndex,
    AddressBlocker,
    generate_exact_name_key,
    generate_name_prefix_key,
    generate_house_number_key,
    generate_address_token_keys,
    generate_house_number_name_prefix_key,
)
from .fuzzy import (
    NameTokenBlocker,
    generate_token_overlap_candidates,
    filter_candidates_by_edit_distance,
)
from .tfidf import SparseTFIDFBlocker, CountrySparseTFIDFBlocker

__all__ = [
    "CandidateGenerator",
    "generate_candidates",
    "ExactIndex",
    "AddressBlocker",
    "generate_exact_name_key",
    "generate_name_prefix_key",
    "generate_house_number_key",
    "generate_address_token_keys",
    "generate_house_number_name_prefix_key",
    "NameTokenBlocker",
    "generate_token_overlap_candidates",
    "filter_candidates_by_edit_distance",
    "SparseTFIDFBlocker",
    "CountrySparseTFIDFBlocker",
]
