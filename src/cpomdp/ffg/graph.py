"""A tree of Gaussian-coupled variables and the structure that wires it together.

Nodes are integer indices ``0..N-1``, each with a dimension. A ``Coupling`` edge links
a parent node to a child node through a linear-Gaussian factor, and the edges must form
a tree rooted at a chosen node. ``CouplingGraph`` gathers the nodes, edges, and per-node
sensors and checks, on construction, that the wiring is a well-formed rooted tree with
dimension-consistent factors.

The graph carries no names. A node is only its index, so the same structure can stand
for any model. Its one public inference is the static collect to the root,
``infer_root``.
A filter over time is a ``CouplingGraphBackend``.
"""

import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, SupportsIndex

import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float64
from numpy.typing import ArrayLike, NDArray

from cpomdp._validation import validate_covariance
from cpomdp.ffg._message import CanonicalGaussian, likelihood_message
from cpomdp.observation import ObservationModel
from cpomdp.types import Belief

__all__ = ["Coupling", "CouplingGraph"]


@dataclass(frozen=True, init=False)
class Coupling:
    """A directed edge from a parent node to a child node: ``child = W·parent + noise``.

    ``parent`` and ``child`` are node indices, oriented so the parent is the endpoint
    nearer the tree's root. The edge is the linear-Gaussian factor
    ``N(child; W·parent, Q)``:

    - ``coupling_matrix`` — W, shape ``(dim[child], dim[parent])``. It maps the parent's
      state onto the child's and need not be square.
    - ``coupling_noise`` — Q, shape ``(dim[child], dim[child])``, positive-definite
      (it is inverted).

    Where a [`GaussianTransition`][cpomdp.GaussianTransition] steps one node in time,
    an edge couples two nodes within a slice. The maths is the same. A node's timescale
    is its transition, so an edge carries none (ADR-072).

    ``efe_relevant`` is a modeller's *declaration* that this edge carries information
    the instrumental epistemic depends on — a physics call, not a structural one (in
    chemotaxis the gradient rides receptor->CheA->CheY, while CheA->CheB methylation is
    observed and coupled yet gradient-blind). A carry partition (ADR-016) that severs a
    flagged edge drops the cross-temporal covariance it holds, breaking the integration
    of that information about a slow latent; every graph EFE entry point refuses such a
    partition (ADR-018). Default ``False``; it does not affect the factor or the filter.

    Raises:
        TypeError: if ``parent`` or ``child`` is not an int, or ``efe_relevant`` is not
            a bool.
        ValueError: if ``coupling_matrix`` is not 2-D, or ``coupling_noise`` is not a
            positive-definite matrix as tall as ``coupling_matrix``.
    """

    parent: int
    child: int
    coupling_matrix: Float64[Array, "c p"]  # W: child-rows × parent-cols
    coupling_noise: Float64[Array, "c c"]  # Q: child × child, positive-definite
    efe_relevant: bool

    def __init__(
        self,
        parent: int,
        child: int,
        coupling_matrix: ArrayLike,
        *,
        coupling_noise: ArrayLike,
        efe_relevant: bool = False,
    ) -> None:
        object.__setattr__(self, "parent", _as_int(parent, "parent"))
        object.__setattr__(self, "child", _as_int(child, "child"))
        object.__setattr__(
            self, "coupling_matrix", jnp.asarray(coupling_matrix, dtype=float)
        )
        object.__setattr__(
            self, "coupling_noise", jnp.asarray(coupling_noise, dtype=float)
        )
        object.__setattr__(self, "efe_relevant", _as_bool(efe_relevant))
        self._validate()

    def _validate(self) -> None:
        coupling_matrix = self.coupling_matrix  # W
        coupling_noise = self.coupling_noise  # Q
        if coupling_matrix.ndim != 2:
            raise ValueError(
                f"coupling_matrix must be 2-D (c, p), got shape {coupling_matrix.shape}"
            )
        # Q is inverted in the message, so it must be positive-definite.
        validate_covariance(coupling_noise, "coupling_noise", require_definite=True)
        child_dim = coupling_matrix.shape[0]
        if coupling_noise.shape != (child_dim, child_dim):
            raise ValueError(
                f"coupling_noise must be {child_dim}x{child_dim} to match the "
                f"{child_dim}-row coupling_matrix, got shape {coupling_noise.shape}"
            )

    def _message_to_parent(self, child_message: CanonicalGaussian) -> CanonicalGaussian:
        """Summarise what a child's belief says about the parent: eliminate the child.

        The coupling is the joint Gaussian over ``z = [parent, child]``::

            Λ_J = [[ WᵀQ⁻¹W, −WᵀQ⁻¹ ],     h_J = 0   (a pure coupling has no bias)
                   [ −Q⁻¹W,    Q⁻¹   ]]

        The upward message:

        1. Folds ``child_message`` into the *child* block — its precision into the
           bottom-right ``c×c`` of ``Λ_J``, its potential into the trailing ``c`` of
           ``h_J`` (a block add during construction, *not* ``__add__``).
        2. Marginalizes the child out, leaving the message on the p-D parent.

        This is the mirror of ``GaussianTransition._predict`` (which folds into the
        parent block and eliminates the parent, emitting downward onto the child);
        here we fold into the child block and eliminate the child, emitting upward.

        Args:
            child_message: the incoming belief on the c-D child, as a
                ``CanonicalGaussian``.

        Returns:
            A ``CanonicalGaussian`` over the p-D parent.
        """
        coupling_matrix = self.coupling_matrix  # W
        coupling_noise = self.coupling_noise  # Q
        c, p = coupling_matrix.shape  # W is (child, parent)
        noise_precision = jnp.linalg.inv(coupling_noise)  # Q⁻¹
        noise_weighted_coupling = noise_precision @ coupling_matrix  # Q⁻¹W

        # The incoming message is on the CHILD, so it folds into the child block —
        # the mirror of predict, where the message folds into the parent (state) block.
        parent_block = coupling_matrix.T @ noise_weighted_coupling  # WᵀQ⁻¹W
        child_block = noise_precision + child_message.precision  # Q⁻¹ + message

        precision = jnp.block(
            [
                [parent_block, -noise_weighted_coupling.T],
                [-noise_weighted_coupling, child_block],
            ]
        )
        # No bias and no parent message → the parent potential is zero; the child
        # slot carries the incoming message's potential.
        potential = jnp.concatenate([jnp.zeros(p), child_message.potential])

        joint = CanonicalGaussian._unchecked(precision, potential)
        return joint.marginalize(over=range(p, p + c))  # eliminate child, keep parent

    def _message_to_child(self, parent_message: CanonicalGaussian) -> CanonicalGaussian:
        """Push a parent's belief down the edge onto the child: eliminate the parent.

        The distribute-pass mirror of ``_message_to_parent``. Over the same joint on
        ``z = [parent, child]``, but here the *parent* is known, so its message folds
        into the *parent* block of ``Λ_J`` and the parent is marginalized out, leaving
        the message on the c-D child.

        Structurally this is ``GaussianTransition._predict`` (fold the incoming belief
        into the source block, eliminate the source, emit onto the target) with a
        non-square ``W`` and no control shift — a pure coupling carries no bias. So on
        its own the downward message is a full child belief, landing in moment form on
        ``mean = W·μ_parent`` and ``cov = W·Σ_parent·Wᵀ + Q``.

        Args:
            parent_message: the incoming belief on the p-D parent, as a
                ``CanonicalGaussian``.

        Returns:
            A ``CanonicalGaussian`` over the c-D child.
        """
        coupling_matrix = self.coupling_matrix  # W
        coupling_noise = self.coupling_noise  # Q
        c, p = coupling_matrix.shape  # W is (child, parent)

        noise_precision = jnp.linalg.inv(coupling_noise)  # Q⁻¹
        noise_weighted_coupling = noise_precision @ coupling_matrix  # Q⁻¹W

        parent_block = (
            coupling_matrix.T @ noise_weighted_coupling + parent_message.precision
        )  # WᵀQ⁻¹W + message
        child_block = noise_precision

        precision = jnp.block(
            [
                [parent_block, -noise_weighted_coupling.T],
                [-noise_weighted_coupling, child_block],
            ]
        )

        potential = jnp.concatenate([parent_message.potential, jnp.zeros(c)])

        joint = CanonicalGaussian._unchecked(precision, potential)
        return joint.marginalize(over=range(p))


class CouplingGraph:
    """A rooted tree of Gaussian-coupled variables.

    The ``N`` nodes are indexed ``0..N-1`` with dimensions ``dims``. ``couplings`` are
    the tree edges, each directed from parent to child, and the root is the one node no
    edge points to. ``observation_models`` maps a node index to the sensor on it. A node
    takes the sensor a flat model takes: a
    [`FixedSensor`][cpomdp.FixedSensor], a state-dependent
    [`CallableSensor`][cpomdp.CallableSensor], or any other
    [`ObservationModel`][cpomdp.ObservationModel]. Construction validates the wiring
    and raises if it is malformed.

    Args:
        dims: ``dims[i]`` is the dimension of node ``i``; its length is the node count.
        couplings: the tree edges — one per non-root node, each that node's only parent.
            The node left without a parent is the root.
        observation_models: maps a node index to the ``ObservationModel`` that reads
            it.

    Raises:
        TypeError: if a dimension is not an int, an edge is not a ``Coupling``, or an
            observation model is not an ``ObservationModel``.
        ValueError: if ``dims`` is empty or non-positive; if an edge or an observation
            model references an out-of-range node; if an edge's ``coupling_matrix`` is
            not ``(dim[child], dim[parent])``; if a sensor's ``C`` is not as wide as the
            node it observes; or if the edges do not form a tree — a node has two
            parents, more than one node has none, or the edges contain a cycle.

    The four fields are read-only once built. Construction is where the wiring is
    checked, so a graph that needs other wiring is a new graph.
    """

    def __init__(
        self,
        dims: Sequence[int],
        couplings: Sequence[Coupling],
        observation_models: Mapping[int, ObservationModel],
    ) -> None:
        self._dims = tuple(_as_int(dim, f"dims[{i}]") for i, dim in enumerate(dims))
        self._couplings = tuple(couplings)
        self._observation_models = dict(observation_models)
        self._validate()
        self._root = self._find_root()
        self._sensor_at_zero = self._linearize_sensors_at_zero()
        self._parent_edge = {edge.child: edge for edge in self._couplings}
        # Deepest first, so a child folds into its parent before the parent moves up.
        self._collect_order = tuple(
            sorted(self._parent_edge, key=self._depth, reverse=True)
        )

    @property
    def root(self) -> int:
        """The one node no coupling points to, which the tree hangs from."""
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
    def observation_models(self) -> Mapping[int, ObservationModel]:
        """Each observed node's sensor, keyed by node index. A read-only view."""
        return MappingProxyType(self._observation_models)

    def _validate(self) -> None:
        n = len(self.dims)
        if n == 0 or any(d <= 0 for d in self.dims):
            raise ValueError(f"dims must be non-empty positive ints, got {self.dims}")

        for edge in self.couplings:
            if not isinstance(edge, Coupling):
                raise TypeError(
                    f"couplings must hold Coupling edges, got {type(edge).__name__}"
                )
            for idx in (edge.parent, edge.child):
                if not 0 <= idx < n:
                    raise ValueError(f"coupling references unknown node {idx}")
            expected = (self.dims[edge.child], self.dims[edge.parent])
            actual = tuple(edge.coupling_matrix.shape)
            if actual != expected:
                raise ValueError(
                    f"coupling {edge.parent}->{edge.child}: coupling_matrix shape "
                    f"{actual} != expected {expected} (dim[child], dim[parent])"
                )
        for idx, sensor in self.observation_models.items():
            if not 0 <= idx < n:
                raise ValueError(f"an observation model references unknown node {idx}")
            if not isinstance(sensor, ObservationModel):
                raise TypeError(
                    f"the observation model on node {idx} must be an ObservationModel "
                    f"such as FixedSensor or CallableSensor, got "
                    f"{type(sensor).__name__}"
                )

    def _find_root(self) -> int:
        """The one node no edge points to, once the edges are checked to form a tree.

        A tree has one parent per node but the root, and every node walks up to it.
        """
        node_count = len(self.dims)
        children = [edge.child for edge in self.couplings]
        if len(children) != len(set(children)):
            raise ValueError("a node has more than one parent — not a tree")
        without_parent = sorted(set(range(node_count)) - set(children))
        if not without_parent:
            raise ValueError(
                "every node has a parent, so the couplings contain a cycle — not a tree"
            )
        if len(without_parent) > 1:
            raise ValueError(
                f"nodes {without_parent} have no parent, and a tree has one root"
            )
        (root,) = without_parent

        parent_of = {edge.child: edge.parent for edge in self.couplings}
        for start in range(node_count):
            node, seen = start, set()
            while node != root:
                if node in seen:
                    raise ValueError("couplings contain a cycle — not a tree")
                seen.add(node)
                node = parent_of[node]
        return root

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
        for node, sensor in self.observation_models.items():
            observation_matrix, observation_noise = sensor.linearize(
                jnp.zeros(self.dims[node])
            )  # C, R
            width = observation_matrix.shape[1]
            if width != self.dims[node]:
                raise ValueError(
                    f"the observation model on node {node}: sensor C width {width} "
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
        return self.observation_models[node].linearize(state)

    def _observation_message(
        self, node: int, reading: ArrayLike, state: ArrayLike | None = None
    ) -> CanonicalGaussian:
        """A reading's message into the node it observes, ``R`` taken at ``state``.

        A fixed sensor's message ignores ``state``, so the fixed paths pass ``None``
        and read the front-loaded ``(C, R)``. A state-dependent sensor needs the
        predicted mean of its node.

        Raises:
            ValueError: If ``R`` at ``state`` is not positive-definite.
        """
        observation_matrix, observation_noise = self._linearized_sensor(node, state)
        if state is not None:
            # R at the predicted mean is inverted in the message, so it must be
            # positive-definite there. Skipped under a trace, like every such check.
            validate_covariance(
                observation_noise,
                f"node {node}'s observation_noise at its predicted mean",
                require_definite=True,
            )
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
        if not isinstance(readings, Mapping):
            raise TypeError(
                "readings must map a node index to its reading, got "
                f"{type(readings).__name__}"
            )
        for node, reading in readings.items():
            if node not in self.observation_models:
                raise ValueError(
                    f"node {node!r} has no sensor, so it cannot take a reading. The "
                    f"observed nodes are {sorted(self.observation_models)}."
                )
            if not self.observation_models[node].is_fixed:
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

    def infer_root(self, readings: Mapping[int, ArrayLike], prior: Belief) -> Belief:
        """Compute the marginal belief at the root from per-node readings and a prior.

        Each reading becomes a message about its node; those messages are passed up the
        tree through the couplings and combined at the root with the prior, giving the
        root's posterior over every reading. Only the root is converted to and from
        moment form — once to lift the prior in, once to read the result out — while
        every message in between stays in canonical form.

        The reading comes first and the belief second, the order
        [`InferenceBackend.infer_states`][cpomdp.InferenceBackend.infer_states] takes.

        Args:
            readings: maps a node index to that node's observation. Each such node must
                carry a fixed sensor, since static inference has no predicted mean to
                evaluate a state-dependent ``R(x)`` at.
            prior: the belief on the root node, taken as its prior.

        Returns:
            The marginal belief at the root.

        Raises:
            TypeError: if ``readings`` is not a mapping or ``prior`` is not a
                ``Belief``.
            ValueError: if ``prior`` is not over the root's state, or a reading is on
                a node with no sensor, on a node whose sensor is state-dependent, or
                not as wide as its sensor.
        """
        self._check_readings(readings)
        self._check_prior(prior)

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
            up = edge._message_to_parent(acc[node])
            acc[edge.parent] = _combine(acc, edge.parent, up)

        # Read the accumulated root message back into moment form.
        mean, cov = acc[self.root].to_moment()  # Σ = Λ⁻¹, μ = Λ⁻¹h
        return Belief(mean=mean, cov=cov)

    def _infer_all(
        self, seeds: Mapping[int, CanonicalGaussian]
    ) -> dict[int, CanonicalGaussian]:
        """Every node's exact marginal by two-pass belief propagation over the tree.

        Where ``infer_root`` collects to the root and returns only that one marginal,
        this adds a downward *distribute* pass so **every** node's marginal comes back —
        the cheap, structure-exploiting alternative to a dense joint solve. Each seed is
        a node's already-formed canonical message (its local prior + evidence, combined
        by the caller); marginalisation stays in canonical form, so no node is inverted
        on this path.

        Args:
            seeds: a canonical message per node — the node's local information. Unlike
                ``infer_root``'s raw ``readings``, these are ready-made
                ``CanonicalGaussian`` messages, not readings still needing a sensor.

        Returns:
            A ``CanonicalGaussian`` marginal per node index; call ``.to_moment()`` on
            any one for its ``(mean, cov)``.
        """
        # Collect: fold each subtree up into its parent (as ``infer_root``), but cache
        # every edge's upward message for the distribute pass. ``below[node]`` ends
        # holding the node's own seed plus everything its children sent up.
        below = dict(seeds)
        up_msg = {}
        for node in self._collect_order:
            edge = self._parent_edge[node]
            up = edge._message_to_parent(below[node])
            up_msg[node] = up
            below[edge.parent] = _combine(below, edge.parent, up)

        # Distribute: push each parent's finished marginal back down, shallowest-first
        # so a parent is done before its children. Divide out the child's own upward
        # message first (``- up_msg[node]``) so its evidence does not double-count.
        marginals = {self.root: below[self.root]}
        for node in reversed(self._collect_order):
            edge = self._parent_edge[node]
            down = edge._message_to_child(marginals[edge.parent] - up_msg[node])
            marginals[node] = _combine(below, node, down)
        return marginals


def _combine(
    acc: Mapping[int, CanonicalGaussian], key: int, msg: CanonicalGaussian
) -> CanonicalGaussian:
    """Add ``msg`` to ``acc[key]``, or start the slot with it if absent."""
    return acc[key] + msg if key in acc else msg


def _as_bool(value: object) -> bool:
    """``efe_relevant`` as a ``bool``. A NumPy bool is stored as one."""
    if not isinstance(value, bool | np.bool_):
        raise TypeError(f"efe_relevant must be a bool, got {type(value).__name__}")
    return bool(value)


def _scalar_of(array: NDArray[Any]) -> object:
    """A 0-d array's one element, as the NumPy scalar it holds."""
    return array[()]


def _as_int(value: object, name: str) -> int:
    """``value`` as an ``int``. A bool and a non-integral number are refused."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = _scalar_of(value)
    if isinstance(value, float | np.floating) and float(value).is_integer():
        return int(value)
    if isinstance(value, bool) or not isinstance(value, SupportsIndex):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return operator.index(value)
