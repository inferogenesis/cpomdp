"""Builders more than one test module shares.

A helper moves here once a second module needs it verbatim. A helper only one module
uses stays in that module.
"""

import jax.numpy as jnp

from cpomdp import Belief, CouplingGraph


def unit_prior(graph: CouplingGraph) -> Belief:
    """N(0, I) over a graph's joint state."""
    n = sum(graph.dims)
    return Belief(jnp.zeros(n), jnp.eye(n))
