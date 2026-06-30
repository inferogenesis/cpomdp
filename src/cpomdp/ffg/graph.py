"""A tree of Gaussian-coupled variables and the structure that wires it together.

Nodes are integer indices ``0..N-1``, each with a dimension. A ``Coupling`` edge links
a parent node to a child node through a linear-Gaussian factor and a time-constant, and
the edges must form a tree rooted at a chosen node. ``CouplingGraph`` gathers the
nodes, edges, and per-node observations and checks, on construction, that the wiring is
a well-formed rooted tree with dimension-consistent factors.

The graph carries no names: a node is only its index, so the same structure can stand
for any model. It holds structure, not inference — building one validates the wiring;
the message passing that reads it lives elsewhere.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import jax.numpy as jnp
from numpy.typing import ArrayLike

from cpomdp.ffg.factors.linear_gaussian import GaussianCoupling, GaussianObservation
from cpomdp.ffg.message import CanonicalGaussian
from cpomdp.types import Belief

__all__ = ["Coupling", "CouplingGraph"]


@dataclass(frozen=True)
class Coupling:
    """A directed edge from a parent node to a child node: ``child = W·parent + noise``.

    ``parent`` and ``child`` are node indices, oriented so the parent is the endpoint
    nearer the tree's root. ``factor`` is the ``GaussianCoupling`` holding this edge's
    ``W`` (shape ``(dim[child], dim[parent])``) and its noise covariance. ``tau`` is a
    time-constant carried alongside the edge; it is metadata and does not affect the
    factor.
    """

    parent: int
    child: int
    factor: GaussianCoupling
    tau: float


class CouplingGraph:
    """A rooted tree of Gaussian-coupled variables.

    The ``N`` nodes are indexed ``0..N-1`` with dimensions ``dims``. ``couplings`` are
    the tree edges, directed away from ``root``, and ``observations`` maps a node index
    to the ``GaussianObservation`` attached to it. Construction validates the wiring and
    raises if it is malformed.

    Args:
        root: index of the node the tree is rooted at.
        dims: ``dims[i]`` is the dimension of node ``i``; its length is the node count.
        couplings: the tree edges — one per non-root node, each that node's only parent.
        observations: maps a node index to the ``GaussianObservation`` on it.

    Raises:
        ValueError: if ``dims`` is empty or non-positive; if ``root``, an edge, or an
            observation references an out-of-range node; if an edge's factor ``W`` is
            not ``(dim[child], dim[parent])``; or if the edges do not form a tree rooted
            at ``root`` — the root has a parent, a node has two parents or none, or the
            edges contain a cycle.
    """

    def __init__(
        self,
        root: int,
        dims: Sequence[int],
        couplings: Sequence[Coupling],
        observations: Mapping[int, GaussianObservation],
    ) -> None:
        self.root = int(root)
        self.dims = tuple(int(d) for d in dims)
        self.couplings = tuple(couplings)
        self.observations = dict(observations)
        self._validate()

    def _validate(self) -> None:
        n = len(self.dims)
        if n == 0 or any(d <= 0 for d in self.dims):
            raise ValueError(f"dims must be non-empty positive ints, got {self.dims}")
        if not 0 <= self.root < n:
            raise ValueError(f"root {self.root} out of range for {n} nodes")

        for edge in self.couplings:
            for idx in (edge.parent, edge.child):
                if not 0 <= idx < n:
                    raise ValueError(f"coupling references unknown node {idx}")
            expected = (self.dims[edge.child], self.dims[edge.parent])
            actual = tuple(edge.factor.coupling.shape)
            if actual != expected:
                raise ValueError(
                    f"coupling {edge.parent}->{edge.child}: factor W shape {actual} "
                    f"!= expected {expected} (dim[child], dim[parent])"
                )
        for idx in self.observations:
            if not 0 <= idx < n:
                raise ValueError(f"observation references unknown node {idx}")

        # The edges must form a tree rooted at `root`: each non-root node has exactly
        # one parent, the root has none, and every node walks up to the root.
        children = [edge.child for edge in self.couplings]
        if self.root in children:
            raise ValueError(f"root {self.root} must not be a child of any coupling")
        if len(children) != len(set(children)):
            raise ValueError("a node has more than one parent — not a tree")
        if set(children) != set(range(n)) - {self.root}:
            raise ValueError("every non-root node must have exactly one parent")

        parent_of = {edge.child: edge.parent for edge in self.couplings}
        for start in range(n):
            node, seen = start, set()
            while node != self.root:
                if node in seen:
                    raise ValueError("couplings contain a cycle — not a tree")
                seen.add(node)
                node = parent_of[node]

    def infer(self, prior: Belief, readings: dict[int, ArrayLike]) -> Belief:
        """Compute the marginal belief at the root from a prior and per-node readings.

        Each reading becomes a message about its node; those messages are passed up the
        tree through the couplings and combined at the root with the prior, giving the
        root's posterior over every reading. Only the root is converted to and from
        moment form — once to lift the prior in, once to read the result out — while
        every message in between stays in canonical form.

        Args:
            prior: the belief on the root node, taken as its prior.
            readings: maps a node index to that node's observation; each such node must
                carry a ``GaussianObservation``.

        Returns:
            The marginal belief at the root.
        """

        def combine(acc, key, msg):
            """Add ``msg`` to ``acc[key]``, or start the slot with it if absent."""
            return acc[key] + msg if key in acc else msg

        def depth(node: int) -> int:
            """The number of edges from ``node`` up to the root."""
            hops = 0
            while node != self.root:
                node = parent_edge[node].parent
                hops += 1
            return hops

        # Lift the moment-form prior into a canonical message on the root.
        prior_precision = jnp.linalg.inv(prior.cov)  # Λ₀ = Σ⁻¹
        prior_msg = CanonicalGaussian._unchecked(
            prior_precision, prior_precision @ prior.mean
        )  # h₀ = Λ₀μ; invariant-preserving lift of a validated Belief — no re-validate

        # Seed each observed node with its reading's message, then fold in the prior at
        # the root (which may already hold the root's own observation).
        acc = {
            node: self.observations[node].message(reading)
            for node, reading in readings.items()
        }
        acc[self.root] = combine(acc, self.root, prior_msg)

        # Pass messages up to the root, deepest nodes first so every child folds into a
        # node before that node is itself sent up to its parent.
        parent_edge = {edge.child: edge for edge in self.couplings}
        order = sorted(parent_edge, key=depth, reverse=True)
        for node in order:
            if node not in acc:  # an unobserved leaf has nothing to send up
                continue
            edge = parent_edge[node]
            # Summarise everything known at `node` onto its parent, eliminating `node`.
            up = edge.factor.message_to_parent(acc[node])
            acc[edge.parent] = combine(acc, edge.parent, up)

        # Read the accumulated root message back into moment form.
        mean, cov = acc[self.root].to_moment()  # Σ = Λ⁻¹, μ = Λ⁻¹h
        return Belief(mean=mean, cov=cov)
