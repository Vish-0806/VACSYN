"""
Blocking and Candidate Generation Module.

Responsible for reducing the NxM comparison space to a high-recall, manageable candidate set
using country partitioning, multi-signal indexing, sparse TF-IDF, and candidate aggregation.

Ownership: Member 1 (Preprocessing & Blocking).
"""

from .candidate_generator import CandidateGenerator, generate_candidates

__all__ = [
    "CandidateGenerator",
    "generate_candidates",
]
