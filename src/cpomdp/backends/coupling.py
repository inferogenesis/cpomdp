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

        # Front-loaded factor tier — built once, reused every step (ADR-002).
        self._transition = self._build_transition()
        self._structural_precision = self._assemble_structural_precision()
        self._obs_layout, self.n_observations = self._build_observation_layout()
        self._flat_model = self._build_validation_model()

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
        observation, action = validate_step_inputs(
            self._flat_model, observation, prior, action
        )
        control = self._control
        if control is None:
            control_term = jnp.zeros(self.n_total)
        else:
            assert action is not None  # validate_step_inputs guarantees this
            control_term = control @ action  # b = B·action

        prior_precision = jnp.linalg.inv(prior.cov)  # Λ₀ = Σ⁻¹
        prior_msg = CanonicalGaussian._unchecked(
            prior_precision,
            prior_precision @ prior.mean,  # h₀ = Λ₀μ
        )
        predicted = self._transition.predict(prior_msg, control_term)  # → Λ⁻, h⁻
        obs_precision, obs_potential = self._observation_messages(observation)

        # The driven-relaxation update, summed in information form (ADR-017):
        # predict (temporal) + structural couplings + observations. ChainBackend adds
        # two terms; the tree's within-slice couplings are the third.
        posterior_precision = (
            predicted.precision + self._structural_precision + obs_precision
        )
        posterior_potential = predicted.potential + obs_potential

        mean, cov = CanonicalGaussian._unchecked(
            posterior_precision, posterior_potential
        ).to_moment()  # Σ = Λ⁻¹, μ = Λ⁻¹h
        return Belief(mean=mean, cov=cov)

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
