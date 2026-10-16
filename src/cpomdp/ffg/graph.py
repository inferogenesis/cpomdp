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
from types import MappingProxyType

import jax.numpy as jnp
from numpy.typing import ArrayLike

from cpomdp.ffg._message import CanonicalGaussian
from cpomdp.ffg.factors.linear_gaussian import (
    GaussianCoupling,
    ObservationFactor,
)
from cpomdp.types import Belief

__all__ = ["Coupling", "CouplingGraph"]


@dataclass(frozen=True)
class Coupling:
    """A directed edge from a parent node to a child node: ``child = W·parent + noise``.

    ``parent`` and ``child`` are node indices, oriented so the parent is the endpoint
    nearer the tree's root. ``factor`` is the
    [`GaussianCoupling`][cpomdp.GaussianCoupling] holding this edge's ``W`` (shape
    ``(dim[child], dim[parent])``) and its noise covariance. ``tau`` is a
    time-constant carried alongside the edge; it is metadata and does not affect the
    factor.

    ``efe_relevant`` is a modeller's *declaration* that this edge carries information
    the instrumental epistemic depends on — a physics call, not a structural one (in
    chemotaxis the gradient rides receptor->CheA->CheY, while CheA->CheB methylation is
    observed and coupled yet gradient-blind). A carry partition (ADR-016) that severs a
    flagged edge drops the cross-temporal covariance it holds, breaking the integration
    of that information about a slow latent; every graph EFE entry point refuses such a
    partition (ADR-018). Default ``False``; it does not affect the factor or the filter.
    """

    parent: int
    child: int
    factor: GaussianCoupling
    tau: float
    efe_relevant: bool = False


class CouplingGraph:
    """A rooted tree of Gaussian-coupled variables.

    The ``N`` nodes are indexed ``0..N-1`` with dimensions ``dims``. ``couplings`` are
    the tree edges, directed away from ``root``, and ``observations`` maps a node index
    to the observation factor attached to it (fixed
    [`GaussianObservation`][cpomdp.GaussianObservation] or state-dependent
    [`CallableGaussianObservation`][cpomdp.CallableGaussianObservation]). Construction
    validates the wiring and raises if it is malformed.

    Args:
        root: index of the node the tree is rooted at.
        dims: ``dims[i]`` is the dimension of node ``i``; its length is the node count.
        couplings: the tree edges — one per non-root node, each that node's only parent.
        observations: maps a node index to its ``ObservationFactor`` (fixed or R(x)).

    Raises:
        ValueError: if ``dims`` is empty or non-positive; if ``root``, an edge, or an
            observation references an out-of-range node; if an edge's factor ``W`` is
            not ``(dim[child], dim[parent])``; if a sensor's ``C`` is not as wide as the
            node it observes; or if the edges do not form a tree rooted at ``root`` —
            the root has a parent, a node has two parents or none, or the edges contain
            a cycle.

    The four fields are read-only once built. Construction is where the wiring is
    checked, so a graph that needs other wiring is a new graph.
    """

    def __init__(
        self,
        root: int,
        dims: Sequence[int],
        couplings: Sequence[Coupling],
        observations: Mapping[int, ObservationFactor],
    ) -> None:
        self._root = int(root)
        self._dims = tuple(int(d) for d in dims)
        self._couplings = tuple(couplings)
        self._observations = dict(observations)
        self._validate()

    @property
    def root(self) -> int:
        """The index of the node the tree is rooted at."""
        return self._root

    @property
    def dims(self) -> tuple[int, ...]:
        """``dims[i]`` is the dimension of node ``i``."""
        return self._dims

    @property
    def couplings(self) -> tuple[Coupling, ...]:
        """The tree edges, directed away from the root."""
        return self._couplings

    @property
    def observations(self) -> Mapping[int, ObservationFactor]:
        """Each observed node's sensor, keyed by node index. A read-only view."""
        return MappingProxyType(self._observations)

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
            actual = tuple(edge.factor.coupling_matrix.shape)
            if actual != expected:
                raise ValueError(
                    f"coupling {edge.parent}->{edge.child}: factor W shape {actual} "
                    f"!= expected {expected} (dim[child], dim[parent])"
                )
        for idx, factor in self.observations.items():
            if not 0 <= idx < n:
                raise ValueError(f"observation references unknown node {idx}")
            width = factor.observation_matrix.shape[1]
            if width != self.dims[idx]:
                raise ValueError(
                    f"observation on node {idx}: sensor C width {width} "
                    f"!= node dimension {self.dims[idx]}"
                )

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

    def infer(self, prior: Belief, readings: Mapping[int, ArrayLike]) -> Belief:
        """Compute the marginal belief at the root from a prior and per-node readings.

        Each reading becomes a message about its node; those messages are passed up the
        tree through the couplings and combined at the root with the prior, giving the
        root's posterior over every reading. Only the root is converted to and from
        moment form — once to lift the prior in, once to read the result out — while
        every message in between stays in canonical form.

        Args:
            prior: the belief on the root node, taken as its prior.
            readings: maps a node index to that node's observation; each such node must
                carry a fixed [`GaussianObservation`][cpomdp.GaussianObservation]
                (static inference has no predicted mean to linearize a state-dependent
                ``R(x)`` at).

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
            node: self.observations[node]._message(reading)
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
            up = edge.factor._message_to_parent(acc[node])
            acc[edge.parent] = combine(acc, edge.parent, up)

        # Read the accumulated root message back into moment form.
        mean, cov = acc[self.root].to_moment()  # Σ = Λ⁻¹, μ = Λ⁻¹h
        return Belief(mean=mean, cov=cov)

    def _infer_all(
        self, seeds: Mapping[int, CanonicalGaussian]
    ) -> dict[int, CanonicalGaussian]:
        """Every node's exact marginal by two-pass belief propagation over the tree.

        Where ``infer`` collects to the root and returns only that one marginal, this
        adds a downward *distribute* pass so **every** node's marginal comes back — the
        cheap, structure-exploiting alternative to a dense joint solve. Each seed is a
        node's already-formed canonical message (its local prior + evidence, combined by
        the caller); marginalisation stays in canonical form, so no node is inverted on
        this path.

        Args:
            seeds: a canonical message per node — the node's local information. Unlike
                ``infer``'s raw ``readings``, these are ready-made ``CanonicalGaussian``
                messages, not observations still needing a factor.

        Returns:
            A ``CanonicalGaussian`` marginal per node index; call ``.to_moment()`` on
            any one for its ``(mean, cov)``.
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

        parent_edge = {edge.child: edge for edge in self.couplings}
        order = sorted(parent_edge, key=depth, reverse=True)  # deepest-first

        # Collect: fold each subtree up into its parent (as ``infer``), but cache every
        # edge's upward message for the distribute pass. ``below[node]`` ends holding
        # the node's own seed plus everything its children sent up.
        below = dict(seeds)
        up_msg = {}
        for node in order:
            edge = parent_edge[node]
            up = edge.factor._message_to_parent(below[node])
            up_msg[node] = up
            below[edge.parent] = combine(below, edge.parent, up)

        # Distribute: push each parent's finished marginal back down, shallowest-first
        # so a parent is done before its children. Divide out the child's own upward
        # message first (``- up_msg[node]``) so its evidence does not double-count.
        marginals = {self.root: below[self.root]}
        for node in reversed(order):
            edge = parent_edge[node]
            down = edge.factor._message_to_child(marginals[edge.parent] - up_msg[node])
            marginals[node] = combine(below, node, down)
        return marginals
