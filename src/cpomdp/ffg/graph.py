"""A tree of Gaussian-coupled variables and the structure that wires it together.

Nodes are integer indices ``0..N-1``, each with a dimension. A ``Coupling`` edge links
a parent node to a child node through a linear-Gaussian factor, and the edges must form
a tree rooted at a chosen node. ``CouplingGraph`` gathers the nodes, edges, and per-node
sensors and checks, on construction, that the wiring is a well-formed rooted tree with
dimension-consistent factors.

The graph carries no names. A node is only its index, so the same structure can stand
for any model. Its one public inference is the static collect to the root, ``infer``.
A filter over time is a ``CouplingGraphBackend``.
"""

import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import SupportsIndex

import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float64
from numpy.typing import ArrayLike

from cpomdp.ffg._message import CanonicalGaussian, likelihood_message
from cpomdp.ffg.factors.linear_gaussian import GaussianCoupling
from cpomdp.observation import ObservationModel
from cpomdp.types import Belief

__all__ = ["Coupling", "CouplingGraph"]


@dataclass(frozen=True)
class Coupling:
    """A directed edge from a parent node to a child node: ``child = W·parent + noise``.

    ``parent`` and ``child`` are node indices, oriented so the parent is the endpoint
    nearer the tree's root. ``factor`` is the
    [`GaussianCoupling`][cpomdp.GaussianCoupling] holding this edge's ``W`` (shape
    ``(dim[child], dim[parent])``) and its noise covariance. A node's timescale is
    its [`GaussianTransition`][cpomdp.GaussianTransition], so an edge carries none
    (ADR-072).

    ``efe_relevant`` is a modeller's *declaration* that this edge carries information
    the instrumental epistemic depends on — a physics call, not a structural one (in
    chemotaxis the gradient rides receptor->CheA->CheY, while CheA->CheB methylation is
    observed and coupled yet gradient-blind). A carry partition (ADR-016) that severs a
    flagged edge drops the cross-temporal covariance it holds, breaking the integration
    of that information about a slow latent; every graph EFE entry point refuses such a
    partition (ADR-018). Default ``False``; it does not affect the factor or the filter.

    Raises:
        TypeError: if ``parent`` or ``child`` is not an int, ``factor`` is not a
            ``GaussianCoupling``, or ``efe_relevant`` is not a bool.
    """

    parent: int
    child: int
    factor: GaussianCoupling
    efe_relevant: bool = False

    def __post_init__(self) -> None:
        """Coerce the node indices to ``int`` and check every field's type."""
        object.__setattr__(self, "parent", _as_int(self.parent, "parent"))
        object.__setattr__(self, "child", _as_int(self.child, "child"))
        if not isinstance(self.factor, GaussianCoupling):
            raise TypeError(
                f"factor must be a GaussianCoupling, got {type(self.factor).__name__}"
            )
        if not isinstance(self.efe_relevant, bool):
            raise TypeError(
                f"efe_relevant must be a bool, got {type(self.efe_relevant).__name__}"
            )


class CouplingGraph:
    """A rooted tree of Gaussian-coupled variables.

    The ``N`` nodes are indexed ``0..N-1`` with dimensions ``dims``. ``couplings`` are
    the tree edges, directed away from ``root``, and ``observations`` maps a node index
    to the sensor on it. A node takes the sensor a flat model takes: a
    [`FixedSensor`][cpomdp.FixedSensor], a state-dependent
    [`CallableSensor`][cpomdp.CallableSensor], or any other
    [`ObservationModel`][cpomdp.ObservationModel]. Construction validates the wiring
    and raises if it is malformed.

    Args:
        root: index of the node the tree is rooted at.
        dims: ``dims[i]`` is the dimension of node ``i``; its length is the node count.
        couplings: the tree edges — one per non-root node, each that node's only parent.
        observations: maps a node index to the ``ObservationModel`` that reads it.

    Raises:
        TypeError: if ``root`` or a dimension is not an int, an edge is not a
            ``Coupling``, or an observation is not an ``ObservationModel``.
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
        observations: Mapping[int, ObservationModel],
    ) -> None:
        self._root = _as_int(root, "root")
        self._dims = tuple(_as_int(dim, f"dims[{i}]") for i, dim in enumerate(dims))
        self._couplings = tuple(couplings)
        self._observations = MappingProxyType(dict(observations))
        self._validate()
        self._sensor_at_zero = self._linearize_sensors_at_zero()
        self._parent_edge = {edge.child: edge for edge in self._couplings}
        # Deepest first, so a child folds into its parent before the parent moves up.
        self._collect_order = tuple(
            sorted(self._parent_edge, key=self._depth, reverse=True)
        )

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
    def observations(self) -> Mapping[int, ObservationModel]:
        """Each observed node's sensor, keyed by node index. A read-only view."""
        return self._observations

    def _validate(self) -> None:
        n = len(self.dims)
        if n == 0 or any(d <= 0 for d in self.dims):
            raise ValueError(f"dims must be non-empty positive ints, got {self.dims}")
        if not 0 <= self.root < n:
            raise ValueError(f"root {self.root} out of range for {n} nodes")

        for edge in self.couplings:
            if not isinstance(edge, Coupling):
                raise TypeError(
                    f"couplings must hold Coupling edges, got {type(edge).__name__}"
                )
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
        for idx, sensor in self.observations.items():
            if not 0 <= idx < n:
                raise ValueError(f"observation references unknown node {idx}")
            if not isinstance(sensor, ObservationModel):
                raise TypeError(
                    f"the observation on node {idx} must be an ObservationModel such "
                    f"as FixedSensor or CallableSensor, got {type(sensor).__name__}"
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

    def _depth(self, node: int) -> int:
        """The number of edges from ``node`` up to the root."""
        hops = 0
        while node != self.root:
            node = self._parent_edge[node].parent
            hops += 1
        return hops

    def _linearize_sensors_at_zero(
        self,
    ) -> dict[int, tuple[Float64[Array, "m n"], Float64[Array, "m m"]]]:
        """Each sensor's ``(C, R)`` at its node's zero state, width-checked once here.

        ``C`` is read through ``linearize``, the one method the protocol and the flat
        model share. The pair is a fixed sensor's own ``(C, R)``, which the fixed filter
        path reuses every step, and a state-dependent sensor's ``(C, R(0))``.
        """
        at_zero = {}
        for node, sensor in self.observations.items():
            observation_matrix, observation_noise = sensor.linearize(
                jnp.zeros(self.dims[node])
            )  # C, R
            width = observation_matrix.shape[1]
            if width != self.dims[node]:
                raise ValueError(
                    f"observation on node {node}: sensor C width {width} "
                    f"!= node dimension {self.dims[node]}"
                )
            at_zero[node] = (observation_matrix, observation_noise)
        return at_zero

    def _linearized_sensor(
        self, node: int, state: ArrayLike | None = None
    ) -> tuple[Float64[Array, "m n"], Float64[Array, "m m"]]:
        """Node ``node``'s sensor ``(C, R)`` at ``state``, or at zero when ``None``.

        ``None`` reads the pair front-loaded at construction. It is exact for a fixed
        sensor and the representative ``R(0)`` for a state-dependent one.
        """
        if state is None:
            return self._sensor_at_zero[node]
        return self.observations[node].linearize(state)

    def _observation_message(
        self, node: int, reading: ArrayLike, state: ArrayLike | None = None
    ) -> CanonicalGaussian:
        """A reading's message into the node it observes, ``R`` taken at ``state``.

        A fixed sensor's message ignores ``state``, so the fixed paths pass ``None``
        and read the front-loaded ``(C, R)``. A state-dependent sensor needs the
        predicted mean of its node.
        """
        observation_matrix, observation_noise = self._linearized_sensor(node, state)
        return likelihood_message(observation_matrix, observation_noise, reading)

    def _check_prior(self, prior: Belief) -> None:
        """Refuse a prior that is not a ``Belief`` over the root node's state."""
        if not isinstance(prior, Belief):
            raise TypeError(f"prior must be a Belief, got {type(prior).__name__}")
        root_dim = self.dims[self.root]
        if prior.mean.shape != (root_dim,):
            raise ValueError(
                f"prior must be over root node {self.root}'s {root_dim}-D state, "
                f"got a mean of shape {prior.mean.shape}"
            )

    def _check_readings(self, readings: Mapping[int, ArrayLike]) -> None:
        """Refuse a reading static inference cannot fold in, naming its node."""
        for node, reading in readings.items():
            if node not in self.observations:
                raise ValueError(
                    f"node {node} has no sensor, so it cannot take a reading. The "
                    f"observed nodes are {sorted(self.observations)}."
                )
            if not self.observations[node].is_fixed:
                raise ValueError(
                    f"node {node} carries a state-dependent sensor, and static "
                    "inference has no predicted mean to evaluate R(x) at. Run the "
                    "graph through a CouplingGraphBackend."
                )
            width = self._sensor_at_zero[node][0].shape[0]
            shape = np.shape(reading)
            if shape != (width,):
                raise ValueError(
                    f"the reading on node {node} must have shape ({width},), "
                    f"got {shape}"
                )

    def infer(self, prior: Belief, readings: Mapping[int, ArrayLike]) -> Belief:
        """Compute the marginal belief at the root from a prior and per-node readings.

        Each reading becomes a message about its node; those messages are passed up the
        tree through the couplings and combined at the root with the prior, giving the
        root's posterior over every reading. Only the root is converted to and from
        moment form — once to lift the prior in, once to read the result out — while
        every message in between stays in canonical form.

        Args:
            prior: the belief on the root node, taken as its prior.
            readings: maps a node index to that node's observation. Each such node must
                carry a fixed sensor, since static inference has no predicted mean to
                evaluate a state-dependent ``R(x)`` at.

        Returns:
            The marginal belief at the root.

        Raises:
            TypeError: if ``prior`` is not a ``Belief``.
            ValueError: if ``prior`` is not over the root's state, or a reading is on
                a node with no sensor, on a node whose sensor is state-dependent, or
                not as wide as its sensor.
        """
        self._check_prior(prior)
        self._check_readings(readings)

        # Lift the moment-form prior into a canonical message on the root.
        prior_precision = jnp.linalg.inv(prior.cov)  # Λ₀ = Σ⁻¹
        prior_msg = CanonicalGaussian._unchecked(
            prior_precision, prior_precision @ prior.mean
        )  # h₀ = Λ₀μ; invariant-preserving lift of a validated Belief — no re-validate

        # Seed each observed node with its reading's message, then fold in the prior at
        # the root (which may already hold the root's own observation).
        acc = {
            node: self._observation_message(node, reading)
            for node, reading in readings.items()
        }
        acc[self.root] = _combine(acc, self.root, prior_msg)

        # Pass messages up to the root, deepest nodes first.
        for node in self._collect_order:
            if node not in acc:  # an unobserved leaf has nothing to send up
                continue
            edge = self._parent_edge[node]
            # Summarise everything known at `node` onto its parent, eliminating `node`.
            up = edge.factor._message_to_parent(acc[node])
            acc[edge.parent] = _combine(acc, edge.parent, up)

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
        # Collect: fold each subtree up into its parent (as ``infer``), but cache every
        # edge's upward message for the distribute pass. ``below[node]`` ends holding
        # the node's own seed plus everything its children sent up.
        below = dict(seeds)
        up_msg = {}
        for node in self._collect_order:
            edge = self._parent_edge[node]
            up = edge.factor._message_to_parent(below[node])
            up_msg[node] = up
            below[edge.parent] = _combine(below, edge.parent, up)

        # Distribute: push each parent's finished marginal back down, shallowest-first
        # so a parent is done before its children. Divide out the child's own upward
        # message first (``- up_msg[node]``) so its evidence does not double-count.
        marginals = {self.root: below[self.root]}
        for node in reversed(self._collect_order):
            edge = self._parent_edge[node]
            down = edge.factor._message_to_child(marginals[edge.parent] - up_msg[node])
            marginals[node] = _combine(below, node, down)
        return marginals


def _combine(
    acc: Mapping[int, CanonicalGaussian], key: int, msg: CanonicalGaussian
) -> CanonicalGaussian:
    """Add ``msg`` to ``acc[key]``, or start the slot with it if absent."""
    return acc[key] + msg if key in acc else msg


def _as_int(value: object, name: str) -> int:
    """``value`` as an ``int``. A bool and a non-integer are refused."""
    if isinstance(value, bool) or not isinstance(value, SupportsIndex):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return operator.index(value)
