"""Forney-style factor graph (FFG) message passing, canonical-Gaussian form.

Owned, from-scratch JAX, with no RxInfer on this path (ADR-012). Variables are wires
and factors are nodes. The private ``_message`` holds the ``CanonicalGaussian``
payload. ``factors`` holds the tier-1 linear-Gaussian nodes. The private
``_chain.ChainBackend`` is the linear-chain (Kalman-identical) backend and
``graph.CouplingGraph`` the branching tree. The construction names are re-exported from
``cpomdp``, not from here.
"""
