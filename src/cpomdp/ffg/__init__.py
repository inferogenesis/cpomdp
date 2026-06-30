"""Forney-style factor graph (FFG) message passing, canonical-Gaussian form.

Owned, from-scratch JAX — no RxInfer on this path (ADR-012). Variables are
wires, factors are nodes. ``message`` carries the ``CanonicalGaussian`` payload;
``factors`` holds the tier-1 linear-Gaussian nodes; ``chain.ChainBackend`` is the
linear-chain (Kalman-identical) backend and ``graph.CouplingGraph`` the branching
tree. Reached via these submodules — not re-exported here.
"""
