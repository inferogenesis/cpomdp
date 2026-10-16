"""Branching-FFG inference backend: the coupling tree as a recursive Gaussian filter.

``ChainBackend`` runs the canonical-form message algebra on a *chain*; this runs it on
a *branching* ``CouplingGraph`` whose nodes each carry their own dynamics. It is the
``InferenceBackend`` a factorised model is driven through — one recursive filter step
over the whole network under an action (issue #25).

The composition is driven relaxation (ADR-017): each node evolves on its own timescale
*and* is driven by its structural parent within the slice. The exact ``[[all]]`` carry
is the *joint* over every node (ADR-016) — purity forces it (the protocol is prior-in /
posterior-out, and ``Agent`` feeds the returned belief back as next step's prior), and
it makes the recursion exact. A chosen node's belief is then a pure slice of that joint
(``marginal``).
"""

import operator
from collections.abc import Mapping, Sequence
from typing import SupportsIndex

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float64
from numpy.typing import ArrayLike

from cpomdp.backends.base import _validate_step_inputs
from cpomdp.ffg._message import CanonicalGaussian
from cpomdp.ffg.factors.linear_gaussian import GaussianTransition
from cpomdp.ffg.graph import Coupling, CouplingGraph
from cpomdp.observation import ObservationModel
from cpomdp.types import Belief, LinearGaussianModel

__all__ = ["CouplingGraphBackend", "IncompatibleLinearizationError"]


class IncompatibleLinearizationError(ValueError):
    """The flattened-Kalman route can't reproduce this backend's linearization.

    ``to_flat_model`` encodes couplings as pseudo-observations, so the flat Kalman
    linearizes ``R`` at the pre-coupling predicted mean μ⁻, while the FFG linearizes at
    the coupling-resolved μ⁺ (issue #27, ADR-019). With a state-dependent ``R(x)`` and a
    mean-shifting coupling, μ⁻ ≠ μ⁺ and no fixed flat model reproduces the filter. The
    test suite checks this backend against a NumPy filter that reads ``R`` at μ⁺, and
    against [`CallableSensor`][cpomdp.CallableSensor] on a single node.
    """


@jax.tree_util.register_pytree_node_class
class _JointObservation:
    """The graph's per-node sensors read as one sensor over the joint state.

    Satisfies the ``ObservationModel`` protocol, so the flat model a backend hands out
    carries the graph's state-dependence instead of a noise block frozen at one
    representative state. Each node's factor is linearized at that node's slice of the
    joint mean — the same point the backend's own filter uses.

    ``C`` is constant, so it rides as a leaf; the per-node factors are leaves in turn
    (each is a pytree), and the column spans are static.
    """

    def __init__(
        self,
        observation_matrix: jax.Array,
        factors: tuple[ObservationModel, ...],
        blocks: tuple[tuple[int, int], ...],
    ) -> None:
        self.observation_matrix = observation_matrix
        self.factors = factors
        self.blocks = blocks
        self.is_fixed = all(f.is_fixed for f in factors)

    def linearize(
        self, x: ArrayLike
    ) -> tuple[Float64[jax.Array, "m n"], Float64[jax.Array, "m m"]]:
        """Local ``(C, R(x))`` — constant ``C``, each node's ``R`` at its own block."""
        x = jnp.asarray(x, dtype=float)
        blocks = [
            factor.linearize(x[lo:hi])[1]
            for factor, (lo, hi) in zip(self.factors, self.blocks, strict=True)
        ]
        if not blocks:
            return self.observation_matrix, jnp.zeros((0, 0))
        return self.observation_matrix, jax.scipy.linalg.block_diag(*blocks)

    def gaussianize(
        self, x: ArrayLike, sigma: Float64[jax.Array, "n n"]
    ) -> tuple[
        Float64[jax.Array, "m"], Float64[jax.Array, "m m"], Float64[jax.Array, "m m"]
    ]:
        """Linear ingredients ``(C·x, C·Σ·Cᵀ + R(x), R(x))``."""
        x = jnp.asarray(x, dtype=float)
        observation_matrix, observation_noise = self.linearize(x)
        return (
            observation_matrix @ x,
            observation_matrix @ sigma @ observation_matrix.T + observation_noise,
            observation_noise,
        )

    def tree_flatten(self):
        """Children: ``(C, factors)``; aux: the column spans."""
        return (self.observation_matrix, self.factors), self.blocks

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "_JointObservation":
        """Rebuild without re-deriving ``is_fixed`` from traced leaves."""
        observation_matrix, factors = children
        obj = object.__new__(cls)
        obj.observation_matrix = observation_matrix
        obj.factors = factors
        obj.blocks = aux_data
        obj.is_fixed = all(getattr(f, "is_fixed", True) for f in factors)
        return obj


class CouplingGraphBackend:
    """FFG message-passing inference on a *branching* linear-Gaussian tree.

    Implements the [`InferenceBackend`][cpomdp.InferenceBackend] protocol for a
    [`CouplingGraph`][cpomdp.CouplingGraph] whose nodes each carry their own temporal
    dynamics. Constructed once from the graph and the per-node transitions, then
    advances a *joint* belief over every node one step at a time (prior in, posterior
    out). Read a single node back with ``marginal``.

    Args:
        graph: the rooted tree — the structural couplings (the within-slice drive
            ``child = W·parent + noise``) and the per-node sensors.
        transitions: one [`GaussianTransition`][cpomdp.GaussianTransition] (A_i, Q_i)
            per node, indexed by node, so ``transitions[i]`` is node ``i``'s own
            dynamics. Each ``Q_i`` must be positive-*definite*, because the
            information form inverts it. A flat
            [`LinearGaussianModel`][cpomdp.LinearGaussianModel] accepts a
            semi-definite Q.
        prior: the belief over the joint state before any reading, a mean of length
            ``sum(graph.dims)``. It is ``model.prior``, so an [`Agent`][cpomdp.Agent]
            built on this backend starts from it, and ``to_flat_model`` carries it.
        control_matrix: B, shape ``(n, p)`` with ``n = sum(graph.dims)``, mapping an
            action into the joint state. ``None`` for a pure filtering model.
        partition: clusters of node indices that together hold every node once. The
            filter carries the correlation within a cluster to the next step and
            drops the correlation between clusters (ADR-016). ``None`` is one cluster
            of every node, the exact filter. All singletons take a cheaper
            node-by-node pass. A prior passed in is cut the same way before it is
            used. ``severed_mass`` measures what a cut drops.

    An ``observation`` passed to ``infer_states`` is the readings of the observed nodes
    stacked in **ascending node-index order**, each node contributing its sensor's rows.
    ``stack_observations`` builds it from readings keyed by node.

    Raises:
        TypeError: If ``prior`` is not a [`Belief`][cpomdp.Belief].
        ValueError: If there is not one transition per node sized to it, if
            ``prior`` or ``control_matrix`` is not over the joint state, or if
            ``partition`` misses a node, repeats one or names one outside the graph.
    """

    def __init__(
        self,
        graph: CouplingGraph,
        transitions: Sequence[GaussianTransition],
        *,
        prior: Belief,
        control_matrix: ArrayLike | None = None,
        partition: Sequence[Sequence[int]] | None = None,
    ) -> None:
        """Validate the wiring and front-load the data-independent work (ADR-002).

        Everything here depends only on the graph and the transitions, never on a
        per-step observation / prior / action, so it is built once and reused across
        every ``infer_states`` call.
        """
        self._validate_transitions(graph, transitions)

        self._graph = graph
        self._transitions = tuple(transitions)
        self._offsets = tuple(int(o) for o in np.cumsum([0, *graph.dims]))
        self._n_states = self._offsets[-1]  # n, the joint state's width
        self._prior = self._validate_prior(prior)
        self._control = self._coerce_control(control_matrix)
        self._partition = self._resolve_partition(partition)
        # Fully-factored (every cluster a singleton) → the carry keeps the belief
        # block-diagonal, so the within-slice solve is a tree and cheap two-pass BP
        # (``_infer_all``) replaces the dense joint solve (ADR-016 energy lever).
        self._is_factored = all(len(cluster) == 1 for cluster in self._partition)
        self._is_exact = len(self._partition) == 1  # one cluster severs nothing

        # Front-loaded factor tier — built once, reused every step (ADR-002).
        self._transition = self._build_transition()
        self._structural_precision = self._assemble_structural_precision()
        self._obs_layout, self._n_observations = self._build_observation_layout()
        self._sensor_is_fixed = all(
            o.is_fixed for o in self._graph.observation_models.values()
        )
        self._model = self._build_model()
        self._partition_mask = self._build_partition_mask()
        # node -> its partition cluster index (ADR-018 admissibility diagnostic).
        self._cluster_of = {
            node: cid for cid, cluster in enumerate(self._partition) for node in cluster
        }

    @staticmethod
    def _validate_transitions(
        graph: CouplingGraph, transitions: Sequence[GaussianTransition]
    ) -> None:
        """Check there is one transition per node, each sized to its node."""
        node_count = len(graph.dims)
        if len(transitions) != node_count:
            raise ValueError(
                f"expected one transition per node ({node_count}), "
                f"got {len(transitions)}"
            )
        for node, transition in enumerate(transitions):
            state_dim = transition.dynamics_matrix.shape[0]
            if state_dim != graph.dims[node]:
                raise ValueError(
                    f"transition for node {node} has state dim {state_dim}, "
                    f"but node {node} has dim {graph.dims[node]}"
                )

    def _validate_prior(self, prior: Belief) -> Belief:
        """Check the prior is a ``Belief`` over the whole joint state."""
        if not isinstance(prior, Belief):
            raise TypeError(f"prior must be a Belief, got {type(prior).__name__}")
        if prior.mean.shape != (self._n_states,):
            raise ValueError(
                f"prior must be over the {self._n_states}-D joint state (the sum of "
                f"the node dimensions), got a mean of shape {prior.mean.shape}"
            )
        return prior

    def _resolve_partition(
        self, partition: Sequence[Sequence[int]] | None
    ) -> tuple[tuple[int, ...], ...]:
        """Default and validate the carry partition (ADR-016).

        ``None`` means the trivial single cluster over every node (the exact full-joint
        carry). Otherwise ``partition`` must be a true partition of the node set: every
        node in exactly one cluster, no node repeated, no node missing.
        """
        node_count = len(self._graph.dims)
        if partition is None:
            return (tuple(range(node_count)),)
        clusters = tuple(
            tuple(_partition_node(node) for node in cluster) for cluster in partition
        )
        seen: set[int] = set()
        for node in (n for cluster in clusters for n in cluster):
            if not 0 <= node < node_count:
                raise ValueError(
                    f"partition node {node} out of range for {node_count} nodes"
                )
            if node in seen:
                raise ValueError(
                    f"partition node {node} appears in more than one cluster"
                )
            seen.add(node)
        missing = set(range(node_count)) - seen
        if missing:
            raise ValueError(f"partition does not cover node(s) {sorted(missing)}")
        return clusters

    def _coerce_control(self, control_matrix: ArrayLike | None) -> jax.Array | None:
        """Coerce the control matrix B to a float array and shape-check it."""
        if control_matrix is None:
            return None
        matrix = jnp.asarray(control_matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != self._n_states:
            raise ValueError(
                f"control_matrix must be a 2-D matrix with {self._n_states} rows "
                f"(the joint state dim), got shape {matrix.shape}"
            )
        return matrix

    def _build_transition(self) -> GaussianTransition:
        """The network's temporal edges as one block-diagonal transition (F, Q)."""
        force = jax.scipy.linalg.block_diag(
            *[t.dynamics_matrix for t in self._transitions]
        )
        force_noise = jax.scipy.linalg.block_diag(
            *[t.dynamics_noise for t in self._transitions]
        )
        return GaussianTransition(force, dynamics_noise=force_noise)

    def _build_observation_layout(
        self,
    ) -> tuple[tuple[tuple[int, int, int], ...], int]:
        """The observed nodes and each one's slice of the stacked observation vector.

        Returns ``(layout, n_observations)`` where ``layout`` is a tuple of
        ``(node, lo, hi)``: reading rows ``lo:hi`` of the stacked observation belong to
        ``node``. Ascending node order is the stacking order a caller must pass.
        """
        if not self._graph.observation_models:
            raise ValueError(
                "no node of the graph carries a sensor, so the filter has no reading "
                "to fold in. Give at least one node a FixedSensor or CallableSensor."
            )
        layout, cursor = [], 0
        for node in sorted(self._graph.observation_models):
            width = self._graph._linearized_sensor(node)[0].shape[0]
            layout.append((node, cursor, cursor + width))
            cursor += width
        return tuple(layout), cursor

    def stack_observations(
        self, readings: Mapping[int, ArrayLike]
    ) -> Float64[jax.Array, "m"]:
        """One step's readings, keyed by node, as the vector ``infer_states`` takes.

        The vector holds each observed node's reading in ascending node order. A dict
        iterates in insertion order, so readings of one width stacked by hand in the
        wrong order pass every shape check and filter each node against another's.

        Args:
            readings: one reading per observed node, keyed by node, the form
                ``CouplingGraph.infer`` takes. A scalar is a one-row reading.

        Returns:
            The stacked observation, shape ``(model.n_observations,)``.

        Raises:
            ValueError: If an observed node has no reading, a reading names a node
                with no sensor, or a reading's width is not its sensor's row count.
        """
        observed = [node for node, _lo, _hi in self._obs_layout]
        missing = sorted(set(observed) - set(readings))
        if missing:
            raise ValueError(f"no reading for observed node(s) {missing}")
        unobserved = sorted(set(readings) - set(observed))
        if unobserved:
            raise ValueError(
                f"node(s) {unobserved} carry no sensor, so they take no reading. "
                f"The observed nodes are {observed}."
            )
        parts = []
        for node, lo, hi in self._obs_layout:
            reading = jnp.atleast_1d(jnp.asarray(readings[node], dtype=float))
            if reading.shape != (hi - lo,):
                raise ValueError(
                    f"the reading on node {node} has shape {reading.shape}, and its "
                    f"sensor reads {hi - lo} value(s)"
                )
            parts.append(reading)
        return jnp.concatenate(parts) if parts else jnp.zeros(0)

    def _embedded_sensor_rows(self) -> list[jax.Array]:
        """Each observed node's C embedded into the joint state, in stacking order.

        Row-block ``k`` reads observed node ``self._obs_layout[k]`` out of the joint
        state, its C placed at the node's columns. Shared by the backend's model and
        ``to_flat_model``.
        """
        rows = []
        for node, _lo, _hi in self._obs_layout:
            observation_matrix = self._graph._linearized_sensor(node)[0]  # C
            embedded = jnp.zeros((observation_matrix.shape[0], self._n_states))
            embedded = embedded.at[:, self._block(node)].set(observation_matrix)
            rows.append(embedded)
        return rows

    def _fixed_noise_blocks(self) -> list[jax.Array]:
        """Each observed node's constant R, in stacking order. Fixed sensors only.

        Read through the graph's ``_linearized_sensor``, so a sensor that implements
        only the protocol works. A state-dependent sensor has no constant to give, and
        the model then reads R at its prior mean instead (ADR-064).
        """
        return [
            self._graph._linearized_sensor(node)[1]
            for node, _lo, _hi in self._obs_layout
        ]

    def _build_model(self) -> LinearGaussianModel:
        """The flat model of the real interface, which also shape-checks each step.

        ``infer_states`` reuses the shared per-step validator (``backends.base``) every
        backend uses, so a caller gets the same errors here as from ``KalmanBackend``.
        It carries the *real* stacked observations (not the structural
        pseudo-observations ``to_flat_model`` adds) and the backend's prior.

        A graph with any state-dependent sensor gets a ``_JointObservation`` and no
        declared ``observation_noise``, so the model's ``R`` follows the state and its
        nominal value is ``R`` at the prior mean, by the rule every flat model follows.
        """
        rows = self._embedded_sensor_rows()
        observation_matrix = (
            jnp.vstack(rows) if rows else jnp.zeros((0, self._n_states))
        )
        if not self._sensor_is_fixed:
            factors = tuple(
                self._graph.observation_models[node]
                for node, _lo, _hi in self._obs_layout
            )
            spans = tuple(
                (self._offsets[node], self._offsets[node + 1])
                for node, _lo, _hi in self._obs_layout
            )
            observation_model = _JointObservation(observation_matrix, factors, spans)
            observation_noise = None  # R at the prior mean (ADR-064)
        else:
            noise_blocks = self._fixed_noise_blocks()
            observation_model = None
            observation_noise = (
                jax.scipy.linalg.block_diag(*noise_blocks)
                if noise_blocks
                else jnp.zeros((0, 0))
            )
        return LinearGaussianModel(
            dynamics_matrix=self._transition.dynamics_matrix,
            observation_matrix=observation_matrix,
            dynamics_noise=self._transition.dynamics_noise,
            observation_noise=None
            if observation_model is not None
            else observation_noise,
            prior=self._prior,
            control_matrix=self._control,
            observation_model=observation_model,
        )

    def _block(self, node: int) -> slice:
        """The span of the stacked joint state that belongs to ``node``.

        The backend stacks every node's state into one vector — node 0's entries, then
        node 1's, and so on — so a node owns a contiguous run of slots. ``_block(i)``
        returns ``slice(offset_i, offset_{i+1})``, the index a caller uses to read or
        write node ``i``'s block of the joint mean / covariance / precision. Example:
        ``dims = (2, 1, 2)`` → node 1 is ``slice(2, 3)``, node 2 is ``slice(3, 5)``.
        """
        return slice(self._offsets[node], self._offsets[node + 1])

    def _assemble_structural_precision(self) -> jax.Array:
        """The fixed structural-coupling precision Λ_struct (ADR-017).

        Each edge adds its canonical coupling block at the parent/child offsets; a
        shared node accumulates a contribution from every incident edge (information
        form is additive), so any node degree works.
        """
        precision = jnp.zeros((self._n_states, self._n_states))
        for edge in self._graph.couplings:
            W = edge.coupling_matrix  # (child_dim, parent_dim)
            Q_inv = jnp.linalg.inv(edge.coupling_noise)
            weighted = Q_inv @ W  # Q⁻¹W   (reused twice)
            parent, child = self._block(edge.parent), self._block(edge.child)
            precision = precision.at[parent, parent].add(W.T @ weighted)  # WᵀQ⁻¹W
            precision = precision.at[child, child].add(Q_inv)  # Q⁻¹
            precision = precision.at[parent, child].add(-weighted.T)  # −WᵀQ⁻¹
            precision = precision.at[child, parent].add(-weighted)  # −Q⁻¹W
        return precision

    def _observation_messages(
        self, observation: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Scatter each node's observation message into the joint (Λ_obs, h_obs).

        The per-step twin of ``_assemble_structural_precision``: each observed node's
        reading is a canonical message on that node (``CᵀR⁻¹C``, ``CᵀR⁻¹y``), built
        from the sensor's front-loaded ``(C, R)``. Each lands at its node's offset. It
        reads ``y``, so it runs per step, unlike the front-loaded structural precision.
        """
        precision = jnp.zeros((self._n_states, self._n_states))
        potential = jnp.zeros(self._n_states)
        for node, lo, hi in self._obs_layout:
            message = self._graph._observation_message(node, observation[lo:hi])
            block = self._block(node)
            precision = precision.at[block, block].add(message.precision)
            potential = potential.at[block].add(message.potential)
        return precision, potential

    def _observation_messages_at(
        self, observation: jax.Array, predicted_mean: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Scatter each node's observation message, ``R`` linearized at μ⁺ (issue #27).

        The state-dependent twin of ``_observation_messages``: a node with a
        state-dependent sensor evaluates its ``R`` at that node's block of the predicted
        mean ``μ⁺``. A fixed sensor's ``linearize`` ignores it, so a mixed graph works.
        This is where the chosen action reaches the covariance. ``μ⁺`` moves with the
        action, so ``R(μ⁺)`` does too, and the epistemic term stops being
        action-invariant (ADR-003).
        """
        precision = jnp.zeros((self._n_states, self._n_states))
        potential = jnp.zeros(self._n_states)
        for node, lo, hi in self._obs_layout:
            block = self._block(node)
            # A fixed sensor ignores the state and a state-dependent one evaluates R at
            # μ⁺[block], so one call serves both.
            message = self._graph._observation_message(
                node, observation[lo:hi], predicted_mean[block]
            )
            precision = precision.at[block, block].add(message.precision)
            potential = potential.at[block].add(message.potential)
        return precision, potential

    def _build_partition_mask(self) -> jax.Array:
        """The carry mask: 1 on within-cluster precision blocks, 0 between clusters.

        A partition (ADR-016) keeps the joint *within* a cluster and severs the
        correlation *between* clusters at the time boundary only. This static mask is
        that block-sparsity; ``infer_states`` multiplies the carried joint precision by
        it before ``to_moment``. The trivial ``[[all nodes]]`` partition is all ones, so
        nothing is severed and the carry stays exact.

        Built in NumPy (mutable slice-assignment) then frozen — front-loaded, never in
        the hot path.
        """
        mask = np.zeros((self._n_states, self._n_states))
        for cluster in self._partition:
            for a in cluster:
                for b in cluster:
                    mask[self._block(a), self._block(b)] = 1.0
        return jnp.asarray(mask)

    def infer_states(
        self,
        observation: ArrayLike,
        prior: Belief,
        action: ArrayLike | None = None,
    ) -> Belief:
        """Advance the joint belief by one filter step over the tree.

        Validate the inputs, form the control shift ``b = control_matrix @ action``,
        then run the driven-relaxation pipeline: cut the prior by the partition, lift
        it into canonical form, ``_predict`` through the block-diagonal per-node
        dynamics (the temporal edges), add the structural coupling precision and the
        per-node observation messages (the within-slice update), and ``to_moment`` the
        joint posterior back.

        Args:
            observation: the observed nodes' readings, stacked in ascending node-index
                order, shape ``(model.n_observations,)``. ``stack_observations``
                builds it from readings keyed by node.
            prior: the current *joint* belief over all nodes. Never mutated.
            action: the action just taken, shape ``(p,)``; required iff the model has a
                control matrix, ``None`` for pure filtering.

        Returns:
            The posterior *joint* belief over all nodes; slice one node out with
            ``marginal``.
        """
        observation, action = _validate_step_inputs(
            self._model, observation, prior, action
        )
        if self._is_factored:
            # Fully-factored carry: skip the dense joint solve, run tree BP instead.
            return self._infer_states_factored(observation, prior, action)
        precision, potential = self._assemble_unchecked(observation, prior, action)
        mean, cov = CanonicalGaussian._unchecked(
            precision, potential
        ).to_moment()  # exact joint: Σ = Λ⁻¹, μ = Λ⁻¹h
        factored_cov, _severed = self._carry(cov)
        return Belief(mean=mean, cov=factored_cov)

    def _infer_states_factored(
        self, observation: jax.Array, prior: Belief, action: jax.Array | None
    ) -> Belief:
        """One filter step via tree BP — the cheap fully-factored path.

        For a singleton partition the carried belief is block-diagonal, so the
        within-slice problem is a tree: each node is seeded per-node (its own predicted
        block + observation) and ``CouplingGraph._infer_all`` returns every node's exact
        marginal in O(tree) small solves, never forming or inverting the dense joint
        (ADR-016). Only each node's own diagonal block of the prior is used. That is
        the cut ``_factor`` makes on the dense path, read without forming it.

        A factor whose linearization point is a coupling-resolved marginal — a
        state-dependent ``R(x)`` today, a nonlinear sensor ``C(x)`` later — needs it
        evaluated at the predicted mean μ⁺, which the per-node predicted seeds do not
        yet carry (couplings only resolve inside BP). So it takes one extra pass: run
        ``_infer_all`` on the *R-free* predicted seeds to read each node's μ⁺ marginal,
        linearize there, then seed and BP again (ADR-019). The pass-1 seeds carry no
        likelihood, so every point is the R-free predictive prior — non-circular, and
        exact in exactly two passes (not a truncated fixed point). Still O(tree), no
        dense solve — the extra pass is the attributable dual-effect cost (RFC-001).
        """
        control_term = self._control_shift(action)  # b = B·action over the joint state
        predicted = {}
        for node in range(len(self._graph.dims)):
            block = self._block(node)
            node_precision = jnp.linalg.inv(prior.cov[block, block])  # Λ_i = Σ_i⁻¹
            node_prior = CanonicalGaussian._unchecked(
                node_precision, node_precision @ prior.mean[block]
            )
            predicted[node] = self._transitions[node]._predict(
                node_prior, control_term[block]
            )

        if self._sensor_is_fixed:
            obs_precision, obs_potential = self._observation_messages(observation)
        else:
            # Pass 1: μ⁺ marginals from the R-free predicted seeds (couplings via BP, no
            # observation) — the frozen point every R(μ⁺) is linearized at (ADR-019).
            predicted_marginals = self._graph._infer_all(predicted)
            mu_plus = jnp.concatenate(
                [
                    predicted_marginals[node].to_moment()[0]
                    for node in range(len(self._graph.dims))
                ]
            )
            obs_precision, obs_potential = self._observation_messages_at(
                observation, mu_plus
            )

        seeds = {}
        for node in range(len(self._graph.dims)):
            block = self._block(node)
            seeds[node] = CanonicalGaussian._unchecked(
                predicted[node].precision + obs_precision[block, block],
                predicted[node].potential + obs_potential[block],
            )
        marginals = self._graph._infer_all(seeds)
        moments = [marginals[node].to_moment() for node in range(len(self._graph.dims))]
        mean = jnp.concatenate([node_mean for node_mean, _ in moments])
        cov = jax.scipy.linalg.block_diag(*[node_cov for _, node_cov in moments])
        return Belief(mean=mean, cov=cov)

    def severed_mass(
        self,
        observation: ArrayLike,
        prior: Belief,
        action: ArrayLike | None = None,
    ) -> Float64[jax.Array, ""]:
        """The covariance one step's carry drops under this partition (ADR-016).

        The norm of the between-cluster *covariance* blocks the carry zeros: how much
        cross-cluster correlation the partition drops at the time boundary, the
        approximation cost of the cut. A covariance magnitude, not bits and not a rate.
        ``0.0`` for the full-joint ``[[all]]`` partition (exact), growing as the cut
        severs more correlation. The per-node marginals this slice are unaffected. The
        carry only drops what is *carried forward* (ADR-017).

        A 0-d array, so it composes under ``jit``, ``vmap`` and ``lax.scan``. A per-run
        profile is a scan over ``infer_states`` and this, and ``float(...)`` reads one
        value on the host. It takes the dense joint solve, separately from
        ``infer_states``, so its cost is its own.
        """
        precision, potential = self._assemble_posterior(observation, prior, action)
        _mean, cov = CanonicalGaussian._unchecked(precision, potential).to_moment()
        _factored, severed = self._carry(cov)
        return severed

    def _assemble_posterior(
        self, observation: ArrayLike, prior: Belief, action: ArrayLike | None
    ) -> tuple[jax.Array, jax.Array]:
        """The within-slice joint posterior in information form (Λ, h), *pre-carry*.

        The driven-relaxation update (ADR-017): validate the inputs, then delegate to
        ``_assemble_unchecked``. Shared by ``infer_states`` (which factorises +
        moment-forms it) and ``severed_mass`` (which measures the mass a partition
        would sever), so the two can never drift.
        """
        observation, action = _validate_step_inputs(
            self._model, observation, prior, action
        )
        return self._assemble_unchecked(observation, prior, action)

    def _assemble_unchecked(
        self, observation: jax.Array, prior: Belief, action: jax.Array | None
    ) -> tuple[jax.Array, jax.Array]:
        """The within-slice joint posterior (Λ, h) without input validation.

        The pure numerical core under the validated surfaces. Cut the prior by the
        partition and lift it to canonical, ``_predict`` through the block-diagonal
        per-node dynamics (the temporal edges), then add the structural coupling
        precision and the per-node observation messages (the within-slice update).
        ChainBackend sums two terms; the tree's within-slice couplings are the third.
        """
        if self._sensor_is_fixed:
            precision, potential = self._predicted_precision(prior, action)
            obs_precision, obs_potential = self._observation_messages(observation)
            return precision + obs_precision, potential + obs_potential
        # state-dependent: need μ⁺ to linearize R
        precision, potential = self._predicted_precision(prior, action)
        predicted_mean, _ = CanonicalGaussian._unchecked(
            precision, potential
        ).to_moment()  # μ⁺
        obs_precision, obs_potential = self._observation_messages_at(
            observation, predicted_mean
        )
        return precision + obs_precision, potential + obs_potential

    def _predicted_precision(
        self, prior: Belief, action: jax.Array | None
    ) -> tuple[jax.Array, jax.Array]:
        """The joint (Λ, h) after dynamics + structural couplings, *before* observing.

        The predict side of a step: cut the joint prior as the partition carries it,
        lift it to canonical, ``_predict`` through the block-diagonal per-node dynamics
        under ``action`` (the temporal edges), then add the structural coupling
        precision — but no observation.
        ``_assemble_unchecked`` folds the observation on top; ``predicted_belief``
        moment-forms it as Σ⁺ (the EFE reads the info gain as the drop from this to the
        post-observation covariance).
        """
        control_term = self._control_shift(action)
        prior_precision = jnp.linalg.inv(self._factor(prior.cov))  # Λ₀ = Σ⁻¹
        prior_msg = CanonicalGaussian._unchecked(
            prior_precision,
            prior_precision @ prior.mean,  # h₀ = Λ₀μ
        )
        predicted = self._transition._predict(prior_msg, control_term)  # → Λ⁻, h⁻
        return predicted.precision + self._structural_precision, predicted.potential

    def _factor(self, cov: jax.Array) -> jax.Array:
        """``cov`` with its between-cluster blocks zeroed, as the partition cuts it.

        The one cut both ends of a step make (ADR-016). ``_predicted_precision``
        applies it to the prior coming in and ``_carry`` to the posterior going out,
        so every surface filters from the belief the node-by-node pass sees. The
        within-cluster blocks are untouched. The full-joint ``[[all]]`` partition
        returns ``cov`` itself and costs nothing per step.
        """
        if self._is_exact:
            return cov
        return cov * self._partition_mask

    def _carry(self, cov: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Factor the carried joint *covariance* and measure what it severs (ADR-016).

        The pure primitive behind both surfaces: ``_factor`` the posterior at the time
        boundary, and return it alongside the severed mass as a jnp scalar (jit / grad /
        vmap / scan safe, no host sync here). Factoring the covariance leaves the
        per-cluster diagonal blocks untouched, so every node's marginal this slice stays
        exact (ADR-017). Only the cross-cluster correlation carried forward is dropped.
        The full-joint ``[[all]]`` partition keeps every block, so the severed mass is
        exactly zero and the exact endpoint stays byte-identical.
        """
        factored = self._factor(cov)
        squared = jnp.sum((cov - factored) ** 2)
        # sqrt has no gradient at 0, and the full joint severs exactly 0. The inner
        # where keeps the untaken branch finite, so its gradient is 0, not NaN.
        nonzero = squared > 0.0
        severed = jnp.where(nonzero, jnp.sqrt(jnp.where(nonzero, squared, 1.0)), 0.0)
        return factored, severed

    def _control_shift(self, action: jax.Array | None) -> jax.Array:
        """The control shift ``b = B·action`` (zero when the model has no control)."""
        control_matrix = self._control
        if control_matrix is None:
            return jnp.zeros(self._n_states)
        assert action is not None  # _validate_step_inputs guarantees this
        return control_matrix @ action  # b = B·action

    def marginal(self, node: int, belief: Belief) -> Belief:
        """The marginal belief at a single ``node`` — a pure slice of the joint belief.

        The joint carried by ``infer_states`` makes every node exact, so a chosen node's
        belief is a slice, no re-inference (issue #25: the target latent need not be the
        root).

        Raises:
            ValueError: If ``node`` is not a node of the graph.
        """
        block = self._block(self._checked_node(node))
        return Belief(mean=belief.mean[block], cov=belief.cov[block, block])

    def block(self, node: int) -> range:
        """The state indices node ``node`` occupies in the joint (a public ``_block``).

        Lets a caller (e.g. the EFE selector) turn a node index into the joint-state
        block the epistemic term targets — ``info_node`` node → ``block`` (issue #26).

        Raises:
            ValueError: If ``node`` is not a node of the graph.
        """
        node = self._checked_node(node)
        return range(self._offsets[node], self._offsets[node + 1])

    def _checked_node(self, node: int) -> int:
        """``node``, refused unless it indexes a node of the graph.

        A negative index would otherwise wrap or slice to an empty block, and an
        epistemic term aimed at an empty block is silently zero.
        """
        node_count = len(self._graph.dims)
        if not 0 <= node < node_count:
            raise ValueError(
                f"node {node} is not a node of this graph, whose nodes are "
                f"0..{node_count - 1}"
            )
        return node

    def severed_efe_edges(self) -> tuple[Coupling, ...]:
        """The EFE-relevant edges this carry partition cuts (ADR-018 diagnostic).

        Each ``efe_relevant`` coupling whose parent and child fall in different clusters
        — so the carry drops the cross-temporal covariance it holds, breaking the
        integration of that information about the targeted latent. Non-empty means
        every EFE entry point must refuse this partition; empty for the exact
        ``[[all]]`` carry and for any cut that only severs unflagged edges. It is a
        structural read and forms no covariance.
        """
        return tuple(
            edge
            for edge in self._graph.couplings
            if edge.efe_relevant
            and self._cluster_of[edge.parent] != self._cluster_of[edge.child]
        )

    @property
    def graph(self) -> CouplingGraph:
        """The graph this backend filters over, fixed at construction."""
        return self._graph

    @property
    def transitions(self) -> tuple[GaussianTransition, ...]:
        """Each node's transition, indexed by node, fixed at construction."""
        return self._transitions

    @property
    def model(self) -> LinearGaussianModel:
        """The flat ``LinearGaussianModel`` of the *real* interface (the backend model).

        Block-diagonal dynamics (F, Q), the real sensors and the control: the joint
        state as one linear-Gaussian model, with the sizes an [`Agent`][cpomdp.Agent]
        reads. The ``Agent`` takes its own ``model`` from this, so passing the backend
        is enough (issue #26). Read the stacked sensor C as
        ``model.observation_matrix`` and R at a mean as
        ``model.observation_noise_at(mean)``. A fixed graph's model returns the R it
        front-loaded. A state-dependent graph's model reads each node's R at that
        node's block of the mean, the point the filter uses.

        It omits the structural couplings, so it is not a filter for the graph.
        ``to_flat_model()`` adds them as pseudo-observations for the oracle
        cross-check.
        """
        return self._model

    def predicted_belief(
        self, prior: Belief, action: ArrayLike | None = None
    ) -> Belief:
        """The joint belief after dynamics + structural couplings, before observing.

        The predict side of one step (Σ⁺, μ⁺): cut the joint ``prior`` as the partition
        carries it, push it through the per-node dynamics under ``action`` and fold in
        the structural couplings, but add no observation. This is the covariance the
        EFE reads — a candidate action's epistemic value is the info gain from this Σ⁺
        to the post-observation covariance (issue #26). Pure (no host-side
        validation), so it rides ``jit`` / ``vmap`` over a grid of candidate actions.
        """
        coerced = None if action is None else jnp.asarray(action, dtype=float)
        precision, potential = self._predicted_precision(prior, coerced)
        mean, cov = CanonicalGaussian._unchecked(precision, potential).to_moment()
        return Belief(mean=mean, cov=cov)

    def to_flat_model(self) -> LinearGaussianModel:
        """The tree flattened into one dense ``LinearGaussianModel`` (the oracle route).

        The temporal edges become the block-diagonal transition (F, Q); the real
        observations and the structural couplings stack into one sensor, each coupling
        an always-zero pseudo-observation ``child − W·parent ~ N(0, Q_struct)``. Running
        [`KalmanBackend`][cpomdp.KalmanBackend] or ``RxInferBackend`` on this
        reproduces this backend's filter exactly. It is the independent cross-check.
        Pad a step's readings with ``flat_observation`` first. The returned model
        carries the backend's prior.

        With couplings present a state-dependent ``R(x)`` is a *category error* for this
        route (``IncompatibleLinearizationError`` — the flat Kalman linearizes at μ⁻,
        the FFG at μ⁺). Coupling-free, μ⁻ = μ⁺, so the emitted model carries the graph's
        own state-dependent sensor and stays faithful; it is a state-dependent flat
        model, not a fixed one, and no fixed one exists.

        The refusal reads the sensor's *declared* state-dependence. Whether ``R`` really
        varies over the means a policy can reach is a property of the noise function and
        the reachable set, which this cannot decide — ``cpomdp.diagnostics.probe_model``
        samples it.
        """
        if not self._sensor_is_fixed and self._graph.couplings:
            # A mean-shifting coupling makes μ⁺ ≠ μ⁻, so no fixed flat model
            # reproduces R(μ⁺) -- inherent, not a missing feature.
            raise IncompatibleLinearizationError(
                "Cannot flatten a state-dependent R(x) model with couplings: a "
                "mean-shifting coupling makes the prediction mean μ⁺ differ from "
                "the prior mean μ⁻, so no fixed linear-Gaussian model reproduces "
                "R(μ⁺). This reads the sensor's declared state-dependence; a "
                "declared R(x) that ignores the state is constant in fact and does "
                "flatten — see cpomdp.diagnostics.probe_model."
            )
        if not self._sensor_is_fixed:
            # Coupling-free R(x): μ⁻ = μ⁺, so handing the graph's own sensor to a flat
            # backend reproduces this filter, and this model is the backend's own.
            return self._model
        rows = self._embedded_sensor_rows()
        noise_blocks = self._fixed_noise_blocks()
        for edge in self._graph.couplings:  # child − W·parent ~ N(0, Q_struct)
            coupling = edge.coupling_matrix  # W
            child_dim = coupling.shape[0]
            row = jnp.zeros((child_dim, self._n_states))
            row = row.at[:, self._block(edge.child)].set(jnp.eye(child_dim))
            row = row.at[:, self._block(edge.parent)].set(-coupling)
            rows.append(row)
            noise_blocks.append(edge.coupling_noise)
        return LinearGaussianModel(
            dynamics_matrix=self._transition.dynamics_matrix,
            observation_matrix=jnp.vstack(rows),
            dynamics_noise=self._transition.dynamics_noise,
            observation_noise=jax.scipy.linalg.block_diag(*noise_blocks),
            prior=self._prior,
            control_matrix=self._control,
        )

    def flat_observation(self, observation: ArrayLike) -> jax.Array:
        """Pad a step's readings with the structural pseudo-observations' zeros.

        ``to_flat_model`` stacks the real observations then the structural couplings, so
        a flat backend consumes ``[readings, zeros(n_structural)]``.
        """
        observation = jnp.asarray(observation, dtype=float)
        n_structural = sum(
            edge.coupling_matrix.shape[0] for edge in self._graph.couplings
        )
        return jnp.concatenate([observation, jnp.zeros(n_structural)])


def _partition_node(node: object) -> int:
    """A partition's node index as an ``int``. A bool and a fraction are refused."""
    if isinstance(node, float | np.floating) and float(node).is_integer():
        return int(node)
    if isinstance(node, bool) or not isinstance(node, SupportsIndex):
        raise TypeError(f"partition node {node!r} must be an int")
    return operator.index(node)
