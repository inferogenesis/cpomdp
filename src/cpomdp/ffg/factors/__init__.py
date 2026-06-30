"""Factor node implementations, one tier per file.

Tier 1 (linear-Gaussian) lives in ``linear_gaussian.py``: the observation,
transition, and coupling nodes. Higher tiers (conjugate-exponential, second-order
Gaussianization) stay deferred seams (ADR-012; issue #21).
"""
