"""Factor node implementations, one tier per file.

Tier 1 (linear-Gaussian) lives in ``linear_gaussian.py``: the transition node. The
coupling node is the graph edge, ``Coupling`` (ADR-073), and a node's observation is
the flat model's sensor (ADR-071). Higher tiers (conjugate-exponential, second-order
Gaussianization) stay deferred seams (ADR-012, issue #21).
"""
