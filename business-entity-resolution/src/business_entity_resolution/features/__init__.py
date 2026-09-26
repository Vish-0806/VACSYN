"""
Feature Engineering Module.

Responsible for extracting similarity features across entity names, addresses, and
source metadata for candidate pairs.

Ownership: Member 2 (Features & Model).
"""

from .name_features import compute_name_similarity_features
from .address_features import compute_address_similarity_features
from .pair_features import PairFeatureExtractor, build_pair_features

__all__ = [
    "compute_name_similarity_features",
    "compute_address_similarity_features",
    "PairFeatureExtractor",
    "build_pair_features",
]
