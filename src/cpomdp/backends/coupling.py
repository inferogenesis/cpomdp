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
(``marginal`` / ``readout``).
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from cpomdp.backends.base import validate_step_inputs
from cpomdp.ffg.factors.linear_gaussian import GaussianTransition
from cpomdp.ffg.graph import CouplingGraph
from cpomdp.ffg.message import CanonicalGaussian
from cpomdp.types import Belief, LinearGaussianModel


class CouplingGraphBackend:
    """FFG message-passing inference on a *branching* linear-Gaussian tree.

    Implements the ``InferenceBackend`` protocol for a ``CouplingGraph`` whose nodes
    each carry their own temporal dynamics. Constructed once from the graph and the
    per-node transitions, then advances a *joint* belief over every node one step at a
    time (prior in, posterior out). Read a single node back with ``marginal`` /
    ``readout``.

    Args:
        graph: the rooted tree — the structural couplings (the within-slice drive
            ``child = W·parent + noise``) and the per-node observations.
        transitions: one ``GaussianTransition`` (A_i, Q_i) per node, indexed by node,
            so ``transitions[i]`` is node ``i``'s own dynamics. Each ``Q_i`` must be
            positive-*definite* (the information form inverts it), the same divergence
            from moment-form Kalman that ``ChainBackend`` carries.
        control: B, shape ``(n_total, p)``, mapping an action into the joint state;
            ``None`` for a pure filtering model. ``n_total = sum(graph.dims)``.
        readout_node: the node ``readout`` returns; defaults to ``graph.root``. The
            latent of interest need not be the root (issue #25).

    An ``observation`` passed to ``infer_states`` is the readings of the observed nodes
    stacked in **ascending node-index order**, each node contributing its sensor's rows.
    """

    def __init__(
        self,
        graph: CouplingGraph,
        transitions: Sequence[GaussianTransition],
        *,
        control: ArrayLike | None = None,
        readout_node: int | None = None,
        partition: Sequence[Sequence[int]] | None = None,
    ) -> None:
        """Validate the wiring and front-load the data-independent work (ADR-002).

        Everything here depends only on the graph and the transitions, never on a
        per-step observation / prior / action, so it is built once and reused across
        every ``infer_states`` call.
        """
        self._validate_transitions(graph, transitions)

        self.graph = graph
        self.dims = graph.dims
        self.transitions = tuple(transitions)
        self._offsets = tuple(int(o) for o in np.cumsum([0, *graph.dims]))
        self.n_total = self._offsets[-1]
        self.readout_node = self._resolve_readout_node(readout_node)
        self._control = self._coerce_control(control)
        self._partition = self._resolve_partition(partition)

        # Front-loaded factor tier — built once, reused every step (ADR-002).
        self._transition = self._build_transition()
        self._structural_precision = self._assemble_structural_precision()
        self._obs_layout, self.n_observations = self._build_observation_layout()
        self._flat_model = self._build_validation_model()
        self._partition_mask = self._build_partition_mask()

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
            state_dim = transition.dynamics.shape[0]
            if state_dim != graph.dims[node]:
                raise ValueError(
                    f"transition for node {node} has state dim {state_dim}, "
                    f"but node {node} has dim {graph.dims[node]}"
                )

    def _resolve_readout_node(self, readout_node: int | None) -> int:
        """Default the readout node to the root and check it is in range."""
        node = self.graph.root if readout_node is None else int(readout_node)
        if not 0 <= node < len(self.dims):
            raise ValueError(
                f"readout_node {node} out of range for {len(self.dims)} nodes"
            )
        return node

    def _resolve_partition(
        self, partition: Sequence[Sequence[int]] | None
    ) -> tuple[tuple[int, ...], ...]:
        """Default and validate the carry partition (ADR-016).

        ``None`` means the trivial single cluster over every node (the exact full-joint
        carry). Otherwise ``partition`` must be a true partition of the node set: every
        node in exactly one cluster, no node repeated, no node missing.
        """
        node_count = len(self.dims)
        if partition is None:
            return (tuple(range(node_count)),)
        clusters = tuple(tuple(int(node) for node in cluster) for cluster in partition)
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

    def _coerce_control(self, control: ArrayLike | None) -> jax.Array | None:
        """Coerce the control matrix B to a float array and shape-check it."""
        if control is None:
            return None
        matrix = jnp.asarray(control, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != self.n_total:
            raise ValueError(
                f"control must be a 2-D matrix with {self.n_total} rows "
                f"(the joint state dim), got shape {matrix.shape}"
            )
        return matrix

    def _build_transition(self) -> GaussianTransition:
        """The network's temporal edges as one block-diagonal transition (F, Q)."""
        force = jax.scipy.linalg.block_diag(*[t.dynamics for t in self.transitions])
        force_noise = jax.scipy.linalg.block_diag(
            *[t.dynamics_noise for t in self.transitions]
        )
        return GaussianTransition(force, force_noise)

    def _build_observation_layout(
        self,
    ) -> tuple[tuple[tuple[int, int, int], ...], int]:
        """The observed nodes and each one's slice of the stacked observation vector.

        Returns ``(layout, n_observations)`` where ``layout`` is a tuple of
        ``(node, lo, hi)``: reading rows ``lo:hi`` of the stacked observation belong to
        ``node``. Ascending node order is the stacking order a caller must pass.
        """
        layout, cursor = [], 0
        for node in sorted(self.graph.observations):
            width = self.graph.observations[node].sensor_model.shape[0]
            layout.append((node, cursor, cursor + width))
            cursor += width
        return tuple(layout), cursor

    def _real_observation_blocks(self) -> tuple[list[jax.Array], list[jax.Array]]:
        """Each observed node's sensor embedded into the joint state, and its noise.

        Row-block ``k`` reads observed node ``self._obs_layout[k]`` out of the joint
        state (its C placed at the node's columns) with that node's R. Shared by the
        validation model and ``to_flat_model``.
        """
        rows, noise_blocks = [], []
        for node, _lo, _hi in self._obs_layout:
            observation = self.graph.observations[node]
            embedded = jnp.zeros((observation.sensor_model.shape[0], self.n_total))
            embedded = embedded.at[:, self._block(node)].set(observation.sensor_model)
            rows.append(embedded)
            noise_blocks.append(observation.sensor_noise)
        return rows, noise_blocks

    def _build_validation_model(self) -> LinearGaussianModel:
        """A flat model so ``validate_step_inputs`` shape-checks inputs identically.

        ``infer_states`` reuses the shared per-step validator (``backends.base``) every
        backend uses, so a caller gets the same errors here as from ``KalmanBackend``.
        The validator reads only the model's dims, so this carries the *real* stacked
        observations (not the structural pseudo-observations ``to_flat_model`` adds);
        the prior is an unused placeholder.
        """
        rows, noise_blocks = self._real_observation_blocks()
        sensor_model = jnp.vstack(rows) if rows else jnp.zeros((0, self.n_total))
        if noise_blocks:
            sensor_noise = jax.scipy.linalg.block_diag(*noise_blocks)
        else:
            sensor_noise = jnp.zeros((0, 0))
        return LinearGaussianModel(
            dynamics=self._transition.dynamics,
            sensor_model=sensor_model,
            dynamics_noise=self._transition.dynamics_noise,
            sensor_noise=sensor_noise,
            prior=Belief(jnp.zeros(self.n_total), jnp.eye(self.n_total)),
            control=self._control,
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
        precision = jnp.zeros((self.n_total, self.n_total))
        for edge in self.graph.couplings:
            W = edge.factor.coupling  # (child_dim, parent_dim)
            Q_inv = jnp.linalg.inv(edge.factor.coupling_noise)
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
        ``GaussianObservation.message(y_node)`` is a canonical message on that node
        (``CᵀR⁻¹C``, ``CᵀR⁻¹y``); place each at the node's offset. Data-dependent — it
        reads ``y`` — unlike the front-loaded structural precision.
        """
        precision = jnp.zeros((self.n_total, self.n_total))
        potential = jnp.zeros(self.n_total)
        for node, lo, hi in self._obs_layout:
            message = self.graph.observations[node].message(observation[lo:hi])
            block = self._block(node)
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
        mask = np.zeros((self.n_total, self.n_total))
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

        Validate the inputs, form the control shift ``b = control @ action``, then run
        the driven-relaxation pipeline: lift the joint prior into canonical form,
        ``predict`` through the block-diagonal per-node dynamics (the temporal edges),
        add the structural coupling precision and the per-node observation messages (the
        within-slice update), and ``to_moment`` the joint posterior back.

        Args:
            observation: the observed nodes' readings, stacked in ascending node-index
                order, shape ``(n_observations,)``.
            prior: the current *joint* belief over all nodes. Never mutated.
            action: the action just taken, shape ``(p,)``; required iff the model has a
                control matrix, ``None`` for pure filtering.

        Returns:
            The posterior *joint* belief over all nodes; slice one node out with
            ``marginal`` / ``readout``.
        """
        precision, potential = self._assemble_posterior(observation, prior, action)
        factored_precision, _severed = self._carry(precision)
        mean, cov = CanonicalGaussian._unchecked(
            factored_precision, potential
        ).to_moment()  # Σ = Λ⁻¹, μ = Λ⁻¹h
        return Belief(mean=mean, cov=cov)

    def partition_error(
        self,
        observation: ArrayLike,
        prior: Belief,
        action: ArrayLike | None = None,
    ) -> float:
        """The severed mass a step under this partition drops (ADR-016 diagnostic).

        The norm of the between-cluster precision blocks the carry zeros — how much
        coupling information the partition throws away at the time boundary, the
        approximation cost of the cut. An *information-form magnitude*, not bits and not
        a rate. ``0.0`` for the full-joint ``[[all]]`` partition (exact), growing as the
        cut severs more coupling.

        This is the eager convenience surface: it forces a host ``float`` and so is not
        itself jit-able. For a per-run profile with no host syncs, stack the
        ``_carry`` scalar inside a traced rollout instead.
        """
        precision, _potential = self._assemble_posterior(observation, prior, action)
        _factored, severed = self._carry(precision)
        return float(severed)

    def rollout(
        self,
        prior: Belief,
        observations: ArrayLike,
        actions: ArrayLike | None = None,
    ) -> tuple[Belief, jax.Array]:
        """Filter a whole sequence, profiling the severed mass at each step (ADR-016).

        One traced ``lax.scan`` pass with no per-step host syncs: the joint belief is
        the scan carry, and each step emits its posterior and the severed mass its carry
        drops. This is the per-run diagnostic a mutable per-step field could not give —
        the profile is produced *inside* the traced rollout, not read off ``self``.

        Args:
            prior: the joint belief the run starts from.
            observations: the stacked readings per step, shape ``(T, n_observations)``.
            actions: the per-step actions, shape ``(T, p)``, required iff the model has
                a control matrix; ``None`` for pure filtering.

        Returns:
            ``(beliefs, severed_masses)`` — the time-stacked posteriors (a ``Belief``
            with a leading time axis: ``mean`` ``(T, n_total)``, ``cov``
            ``(T, n_total, n_total)``) and the length-``T`` severed-mass profile. The
            full-joint ``[[all]]`` partition profiles all-zero.
        """
        observations = jnp.asarray(observations, dtype=float)
        if observations.ndim != 2 or observations.shape[1] != self.n_observations:
            raise ValueError(
                f"observations must have shape (T, {self.n_observations}), "
                f"got {observations.shape}"
            )

        if self._control is None:
            if actions is not None:
                raise ValueError("actions given, but this model has no control matrix")

            def step(belief: Belief, observation: jax.Array):
                return self._instrumented_step(belief, observation, None)

            _final, outputs = jax.lax.scan(step, prior, observations)
            return outputs

        if actions is None:
            raise ValueError("this model has a control matrix; actions are required")
        actions = jnp.asarray(actions, dtype=float)
        if actions.shape[0] != observations.shape[0]:
            raise ValueError(
                f"actions has {actions.shape[0]} steps, but observations "
                f"has {observations.shape[0]}"
            )

        def controlled_step(belief: Belief, step_inputs: tuple[jax.Array, jax.Array]):
            observation, action = step_inputs
            return self._instrumented_step(belief, observation, action)

        _final, outputs = jax.lax.scan(controlled_step, prior, (observations, actions))
        return outputs

    def _instrumented_step(
        self, belief: Belief, observation: jax.Array, action: jax.Array | None
    ) -> tuple[Belief, tuple[Belief, jax.Array]]:
        """One traced filter step for ``rollout``: the ``lax.scan`` body.

        The pure numerical step (``_assemble_unchecked`` → ``_carry`` → ``to_moment``)
        packaged as ``(carry, output)`` for ``lax.scan``: the joint posterior is both
        the next carry and, with the severed scalar, the emitted per-step output.
        """
        precision, potential = self._assemble_unchecked(observation, belief, action)
        factored_precision, severed = self._carry(precision)
        mean, cov = CanonicalGaussian._unchecked(
            factored_precision, potential
        ).to_moment()
        posterior = Belief(mean=mean, cov=cov)
        return posterior, (posterior, severed)

    def _assemble_posterior(
        self, observation: ArrayLike, prior: Belief, action: ArrayLike | None
    ) -> tuple[jax.Array, jax.Array]:
        """The within-slice joint posterior in information form (Λ, h), *pre-carry*.

        The driven-relaxation update (ADR-017): validate the inputs, then delegate to
        ``_assemble_unchecked``. Shared by ``infer_states`` (which factorises +
        moment-forms it) and ``partition_error`` (which measures the mass a partition
        would sever), so the two can never drift.
        """
        observation, action = validate_step_inputs(
            self._flat_model, observation, prior, action
        )
        return self._assemble_unchecked(observation, prior, action)

    def _assemble_unchecked(
        self, observation: jax.Array, prior: Belief, action: jax.Array | None
    ) -> tuple[jax.Array, jax.Array]:
        """The within-slice joint posterior (Λ, h) without input validation.

        The pure numerical core, so a traced rollout (``rollout``) can call it inside a
        ``lax.scan`` body where the host-side ``validate_step_inputs`` cannot run. Lift
        the joint prior to canonical, ``predict`` through the block-diagonal per-node
        dynamics (the temporal edges), then add the structural coupling precision and
        the per-node observation messages (the within-slice update). ChainBackend sums
        two terms; the tree's within-slice couplings are the third.
        """
        control_term = self._control_shift(action)
        prior_precision = jnp.linalg.inv(prior.cov)  # Λ₀ = Σ⁻¹
        prior_msg = CanonicalGaussian._unchecked(
            prior_precision,
            prior_precision @ prior.mean,  # h₀ = Λ₀μ
        )
        predicted = self._transition.predict(prior_msg, control_term)  # → Λ⁻, h⁻
        obs_precision, obs_potential = self._observation_messages(observation)
        precision = predicted.precision + self._structural_precision + obs_precision
        potential = predicted.potential + obs_potential
        return precision, potential

    def _carry(self, precision: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Factor the carried joint precision and measure what it severs (ADR-016).

        The pure primitive behind both surfaces: zero the between-cluster precision
        blocks (keep the mask's within-cluster ones) at the time boundary, and return
        the factored precision alongside the severed mass as a jnp scalar (jit / grad /
        vmap / scan safe — no host sync here). The full-joint ``[[all]]`` mask keeps
        every block, so the factored precision is unchanged and the severed mass is
        exactly zero (the exact endpoint stays byte-identical).
        """
        severed = precision * (1.0 - self._partition_mask)
        factored = precision - severed
        return factored, jnp.linalg.norm(severed)

    def _control_shift(self, action: jax.Array | None) -> jax.Array:
        """The control shift ``b = B·action`` (zero when the model has no control)."""
        control = self._control
        if control is None:
            return jnp.zeros(self.n_total)
        assert action is not None  # validate_step_inputs guarantees this
        return control @ action  # b = B·action

    def marginal(self, node: int, belief: Belief) -> Belief:
        """The marginal belief at a single ``node`` — a pure slice of the joint belief.

        The joint carried by ``infer_states`` makes every node exact, so a chosen node's
        belief is a slice, no re-inference (issue #25: the target latent need not be the
        root).
        """
        block = self._block(node)
        return Belief(mean=belief.mean[block], cov=belief.cov[block, block])

    def readout(self, belief: Belief) -> Belief:
        """The marginal at ``readout_node`` (the root by default)."""
        return self.marginal(self.readout_node, belief)

    def to_flat_model(self) -> LinearGaussianModel:
        """The tree flattened into one dense ``LinearGaussianModel`` (the oracle route).

        The temporal edges become the block-diagonal transition (F, Q); the real
        observations and the structural couplings stack into one sensor, each coupling
        an always-zero pseudo-observation ``child − W·parent ~ N(0, Q_struct)``. Running
        ``KalmanBackend`` or ``RxInferBackend`` on this reproduces this backend's filter
        exactly — the independent cross-check, and what the Phase-3 demo contrasts the
        native FFG against. Pad a step's readings with ``flat_observation`` first; the
        returned prior is an unused placeholder (pass the real prior per step).
        """
        rows, noise_blocks = self._real_observation_blocks()
        for edge in self.graph.couplings:  # child − W·parent ~ N(0, Q_struct)
            coupling = edge.factor.coupling  # W
            child_dim = coupling.shape[0]
            row = jnp.zeros((child_dim, self.n_total))
            row = row.at[:, self._block(edge.child)].set(jnp.eye(child_dim))
            row = row.at[:, self._block(edge.parent)].set(-coupling)
            rows.append(row)
            noise_blocks.append(edge.factor.coupling_noise)
        return LinearGaussianModel(
            dynamics=self._transition.dynamics,
            sensor_model=jnp.vstack(rows),
            dynamics_noise=self._transition.dynamics_noise,
            sensor_noise=jax.scipy.linalg.block_diag(*noise_blocks),
            prior=Belief(jnp.zeros(self.n_total), jnp.eye(self.n_total)),
            control=self._control,
        )

    def flat_observation(self, observation: ArrayLike) -> jax.Array:
        """Pad a step's readings with the structural pseudo-observations' zeros.

        ``to_flat_model`` stacks the real observations then the structural couplings, so
        a flat backend consumes ``[readings, zeros(n_structural)]``.
        """
        observation = jnp.asarray(observation, dtype=float)
        n_structural = sum(
            edge.factor.coupling.shape[0] for edge in self.graph.couplings
        )
        return jnp.concatenate([observation, jnp.zeros(n_structural)])
