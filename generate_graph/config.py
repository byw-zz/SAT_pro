"""Shared generation configuration.

Single source of truth for constants that must stay identical across the
graph generators and every analysis/comparison entry point.
"""

# Fixed 24-level exploit-success-probability grid. Every module that samples
# E_prob draws from this same set, so results stay comparable across scripts.
FIXED_E_PROBS = [
    0.02, 0.05, 0.10, 0.12, 0.15, 0.18,
    0.20, 0.25, 0.30, 0.32, 0.35, 0.38,
    0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
    0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
]
