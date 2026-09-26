"""
Evaluation and Validation Module.

Responsible for official competition metric calculation (macro F0.5 per S1 entity),
local cross-validation splitting, blocking recall assessment, and error diagnostics.

Ownership: Member 3 (Evaluation).
"""

from .metrics import compute_macro_f05, compute_entity_f05
from .validation import create_validation_split, evaluate_pipeline_on_validation
from .error_analysis import perform_error_analysis

__all__ = [
    "compute_macro_f05",
    "compute_entity_f05",
    "create_validation_split",
    "evaluate_pipeline_on_validation",
    "perform_error_analysis",
]
