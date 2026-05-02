"""BP+GA vs MaxSAT comparison module.

Contains:
- bp_core: Core BP algorithm and GA optimization
- batch_comparison: Batch comparison across multiple graph instances
- scale_comparison: Scale comparison
"""

from .bp_core import (
    run_bp_analysis,
    find_best_defense_bp,
    find_best_defense_maxsat,
    compute_objective_bp_style,
    MAXHS_BIN,
    FIXED_E_PROBS,
)

__all__ = [
    "run_bp_analysis",
    "find_best_defense_bp",
    "find_best_defense_maxsat",
    "compute_objective_bp_style",
    "MAXHS_BIN",
    "FIXED_E_PROBS",
]
