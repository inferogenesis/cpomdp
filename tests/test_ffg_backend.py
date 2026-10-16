"""The branching FFG as a recursive filter: ``CouplingGraphBackend`` (issue #25).

``test_ffg_tree.py`` pins the *static* within-slice collect
(``CouplingGraph.infer_root`` — a root prior + readings -> the root marginal). This file
pins the *temporal* recursion that turns that structure into an ``InferenceBackend``:
each node carries its own dynamics (driven relaxation — ADR-017), the whole tree steps
forward under an action, and the exact ``[[all]]`` filter is the joint-precision solve
(predict through the block-diagonal per-node dynamics, add the structural coupling +
observation precision blocks, solve).

Oracle: a plain-NumPy driven-relaxation filter — per step ``predict`` (block-diagonal
``F = blkdiag(A_i)``, ``Q = blkdiag(Q_i)``) then a joint-precision ``update`` that folds
in the structural couplings and the observations, sharing no code with the
canonical-form math under test. The keystone bar is atol 1e-7 on *every* node's marginal
(not just the root), over multi-step sequences with and without control.
"""

import copy
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _support import unit_prior
from cpomdp.agent import Agent
from cpomdp.backends.base import InferenceBackend
from cpomdp.backends.coupling import (
    CouplingGraphBackend,
    IncompatibleLinearizationError,
)
from cpomdp.backends.kalman import KalmanBackend
from cpomdp.backends.rxinfer import RxInferBackend
from cpomdp.ffg.factors.linear_gaussian import GaussianTransition
from cpomdp.ffg.graph import Coupling, CouplingGraph
from cpomdp.observation import CallableSensor, FixedSensor
from cpomdp.types import Belief, LinearGaussianModel


def _spd(rng, n):
    """A random n x n symmetric positive-definite matrix (NumPy, independent)."""
    a = rng.standard_normal((n, n))
    return a @ a.T + n * np.eye(n)


def _block_diag(blocks):
    """Lay square blocks down the diagonal of a single zero matrix (NumPy)."""
    total = sum(b.shape[0] for b in blocks)
    out = np.zeros((total, total))
    offset = 0
    for b in blocks:
        size = b.shape[0]
        out[offset : offset + size, offset : offset + size] = b
        offset += size
    return out


def _driven_relaxation_oracle(
    dims,
    edges,
    obs_specs,
    dyn,
    prior_mean,
    prior_cov,
    control_matrix,
    obs_seq,
    action_seq,
):
    """Exact N-step driven-relaxation filter via the joint precision — plain NumPy.

    The independent ground truth for the dynamic branching FFG (ADR-016/017). Per step:

    - **predict** each node by its own dynamics, ``mu- = F mu + B a`` and
      ``Sig- = F Sig F^T + Q`` with ``F = blkdiag(A_i)``, ``Q = blkdiag(Q_i)``;
    - **update** by assembling the joint precision
      ``Lam = inv(Sig-) + Lam_struct + sum_node C^T R^-1 C`` and potential
      ``h = inv(Sig-) mu- + sum_node C^T R^-1 y``, then read back ``Sig = inv(Lam)``
      and ``mu = Sig h``.

    ``Lam_struct`` is the data-independent structural-coupling precision: each edge
    ``(p, c, W, Qs)`` contributes ``W^T Qs^-1 W`` to the parent block, ``Qs^-1`` to the
    child block, and ``-W^T Qs^-1`` / ``-Qs^-1 W`` to the cross blocks — the same
    ``[[W^T Q^-1 W, -W^T Q^-1], [-Q^-1 W, Q^-1]]`` a ``Coupling`` carries.

    Returns ``(states, offs)`` where ``states[t] = (mu, Sig)`` is the full joint at step
    ``t`` and ``offs`` locates each node's block.
    """
    offs = np.cumsum([0, *dims])
    dim = int(offs[-1])

    def blk(i):
        return slice(int(offs[i]), int(offs[i + 1]))

    force = _block_diag([np.asarray(a, float) for a, _ in dyn])  # F = blkdiag(A_i)
    proc = _block_diag([np.asarray(q, float) for _, q in dyn])  # Q = blkdiag(Q_i)

    lam_struct = np.zeros((dim, dim))
    for parent, child, w, qs in edges:
        w = np.asarray(w, float)
        qs_inv = np.linalg.inv(np.asarray(qs, float))
        lam_struct[blk(parent), blk(parent)] += w.T @ qs_inv @ w
        lam_struct[blk(child), blk(child)] += qs_inv
        lam_struct[blk(parent), blk(child)] += -w.T @ qs_inv
        lam_struct[blk(child), blk(parent)] += -qs_inv @ w

    mu = np.asarray(prior_mean, float).copy()
    sig = np.asarray(prior_cov, float).copy()
    observed = sorted(obs_specs)
    states = []
    for step, y_stacked in enumerate(obs_seq):
        shift = np.zeros(dim)
        if control_matrix is not None:
            shift = np.asarray(control_matrix, float) @ np.asarray(
                action_seq[step], float
            )
        mu_pred = force @ mu + shift
        sig_pred = force @ sig @ force.T + proc
        pred_precision = np.linalg.inv(sig_pred)

        lam = pred_precision + lam_struct
        h = pred_precision @ mu_pred
        cursor = 0
        for node in observed:
            c_mat, r_mat = (np.asarray(z, float) for z in obs_specs[node])
            m = c_mat.shape[0]
            y = np.asarray(y_stacked, float)[cursor : cursor + m]
            cursor += m
            r_inv = np.linalg.inv(r_mat)
            lam[blk(node), blk(node)] += c_mat.T @ r_inv @ c_mat
            h[blk(node)] += c_mat.T @ r_inv @ y

        sig = np.linalg.inv(lam)
        mu = sig @ h
        states.append((mu.copy(), sig.copy()))
    return states, offs


def _predicted_joint_oracle(
    dims, edges, dyn, prior_mean, prior_cov, control_matrix, action
):
    """The joint after predict + structural couplings, BEFORE observing — plain NumPy.

    The pre-observation ``(mu+, Sig+)`` the EFE reads (issue #26): predict each node by
    its own dynamics, then fold in the structural-coupling precision, but add no
    observation. Independent of the canonical-form backend.
    """
    offs = np.cumsum([0, *dims])
    dim = int(offs[-1])

    def blk(i):
        return slice(int(offs[i]), int(offs[i + 1]))

    force = _block_diag([np.asarray(a, float) for a, _ in dyn])  # F = blkdiag(A_i)
    proc = _block_diag([np.asarray(q, float) for _, q in dyn])  # Q = blkdiag(Q_i)

    lam_struct = np.zeros((dim, dim))
    for parent, child, w, qs in edges:
        w = np.asarray(w, float)
        qs_inv = np.linalg.inv(np.asarray(qs, float))
        lam_struct[blk(parent), blk(parent)] += w.T @ qs_inv @ w
        lam_struct[blk(child), blk(child)] += qs_inv
        lam_struct[blk(parent), blk(child)] += -w.T @ qs_inv
        lam_struct[blk(child), blk(parent)] += -qs_inv @ w

    shift = np.zeros(dim)
    if control_matrix is not None:
        shift = np.asarray(control_matrix, float) @ np.asarray(action, float)
    mu_pred = force @ np.asarray(prior_mean, float) + shift
    sig_pred = force @ np.asarray(prior_cov, float) @ force.T + proc
    pred_precision = np.linalg.inv(sig_pred)

    lam = pred_precision + lam_struct  # predict + structural couplings, NO observation
    sig = np.linalg.inv(lam)
    mu = sig @ (pred_precision @ mu_pred)
    return mu, sig


def _build(
    dims,
    edges,
    obs_specs,
    dyn,
    *,
    control_matrix=None,
    partition=None,
):
    """Build a ``CouplingGraphBackend`` (graph + per-node transitions) for a spec."""
    couplings = tuple(
        Coupling(parent, child, w, coupling_noise=q) for parent, child, w, q in edges
    )
    observation_models = {
        n: FixedSensor(c, observation_noise=r) for n, (c, r) in obs_specs.items()
    }
    graph = CouplingGraph(
        dims=dims, couplings=couplings, observation_models=observation_models
    )
    transitions = tuple(GaussianTransition(a, dynamics_noise=q) for a, q in dyn)
    kwargs = {"control_matrix": control_matrix}
    if partition is not None:
        kwargs["partition"] = partition
    return CouplingGraphBackend(graph, transitions, prior=unit_prior(graph), **kwargs)


# The Phase-5 demo tree, now dynamic: shared CheA (root, 0) feeds fast CheY-P (1, the
# degree-3 hub) and slow CheB (2); CheY-P feeds two motors (3, 4). Each node relaxes on
# its own timescale (a_i) — CheB slow (near 1), motors fast.
_CHEMOTAXIS_DIMS = (1, 1, 1, 1, 1)
_CHEMOTAXIS_EDGES = [
    (0, 1, [[0.8]], [[0.05]]),
    (0, 2, [[0.6]], [[0.05]]),
    (1, 3, [[1.0]], [[0.03]]),
    (1, 4, [[1.0]], [[0.03]]),
]
_CHEMOTAXIS_OBS = {
    2: (np.eye(1), [[0.10]]),
    3: (np.eye(1), [[0.10]]),
    4: (np.eye(1), [[0.10]]),
}
_CHEMOTAXIS_DYN = [
    ([[0.7]], [[0.10]]),  # CheA
    ([[0.5]], [[0.08]]),  # CheY-P (fast)
    ([[0.98]], [[0.02]]),  # CheB (slow)
    ([[0.4]], [[0.05]]),  # motor A (fast)
    ([[0.4]], [[0.05]]),  # motor B (fast)
]


def _check_sequence(
    backend,
    dims,
    edges,
    obs_specs,
    dyn,
    prior,
    obs_seq,
    *,
    control_matrix=None,
    action_seq=None,
    atol=1e-7,
):
    """Drive the backend over ``obs_seq`` and check each node's marginal vs oracle."""
    states, offs = _driven_relaxation_oracle(
        dims,
        edges,
        obs_specs,
        dyn,
        prior.mean,
        prior.cov,
        control_matrix,
        obs_seq,
        action_seq,
    )
    belief = prior
    for step, y in enumerate(obs_seq):
        action = None if action_seq is None else action_seq[step]
        belief = backend.infer_states(y, belief, action)
        mu_o, sig_o = states[step]
        for node in range(len(dims)):
            node_belief = backend.marginal(node, belief)
            lo, hi = int(offs[node]), int(offs[node + 1])
            np.testing.assert_allclose(
                np.asarray(node_belief.mean), mu_o[lo:hi], atol=atol
            )
            np.testing.assert_allclose(
                np.asarray(node_belief.cov), sig_o[lo:hi, lo:hi], atol=atol
            )
    return belief


class TestKeystoneDrivenRelaxation:
    """The exact ``[[all]]`` filter matches the independent joint-precision oracle."""

    def test_chemotaxis_tree_sequence_no_control(self):
        rng = np.random.default_rng(1)
        backend = _build(
            _CHEMOTAXIS_DIMS, _CHEMOTAXIS_EDGES, _CHEMOTAXIS_OBS, _CHEMOTAXIS_DYN
        )
        prior = Belief(mean=np.zeros(5), cov=np.eye(5) * 2.0)
        obs_seq = [rng.standard_normal(3) for _ in range(8)]
        _check_sequence(
            backend,
            _CHEMOTAXIS_DIMS,
            _CHEMOTAXIS_EDGES,
            _CHEMOTAXIS_OBS,
            _CHEMOTAXIS_DYN,
            prior,
            obs_seq,
        )

    def test_chemotaxis_tree_sequence_with_control(self):
        rng = np.random.default_rng(2)
        # A scalar action drives CheA (the sensed signal) — a thin control (p < n).
        control = np.array([[1.0], [0.0], [0.0], [0.0], [0.0]])
        backend = _build(
            _CHEMOTAXIS_DIMS,
            _CHEMOTAXIS_EDGES,
            _CHEMOTAXIS_OBS,
            _CHEMOTAXIS_DYN,
            control_matrix=control,
        )
        prior = Belief(mean=np.zeros(5), cov=np.eye(5) * 2.0)
        obs_seq = [rng.standard_normal(3) for _ in range(8)]
        action_seq = [rng.standard_normal(1) for _ in range(8)]
        _check_sequence(
            backend,
            _CHEMOTAXIS_DIMS,
            _CHEMOTAXIS_EDGES,
            _CHEMOTAXIS_OBS,
            _CHEMOTAXIS_DYN,
            prior,
            obs_seq,
            control_matrix=control,
            action_seq=action_seq,
        )

    def test_multidim_nonsquare_couplings(self):
        # A 2-D root, a non-square coupling, and an internal unobserved node (1).
        rng = np.random.default_rng(3)
        dims = (2, 1, 2, 2)
        edges = [
            (0, 1, rng.standard_normal((1, 2)), _spd(rng, 1)),
            (1, 2, rng.standard_normal((2, 1)), _spd(rng, 2)),
            (0, 3, rng.standard_normal((2, 2)), _spd(rng, 2)),
        ]
        obs = {
            2: (rng.standard_normal((1, 2)), _spd(rng, 1)),
            3: (np.eye(2), _spd(rng, 2)),
        }
        dyn = [
            (rng.standard_normal((2, 2)) * 0.2, _spd(rng, 2)),
            (rng.standard_normal((1, 1)) * 0.2, _spd(rng, 1)),
            (rng.standard_normal((2, 2)) * 0.2, _spd(rng, 2)),
            (rng.standard_normal((2, 2)) * 0.2, _spd(rng, 2)),
        ]
        backend = _build(dims, edges, obs, dyn)
        prior = Belief(mean=rng.standard_normal(7), cov=_spd(rng, 7))
        obs_seq = [
            np.concatenate([rng.standard_normal(1), rng.standard_normal(2)])
            for _ in range(6)
        ]
        _check_sequence(backend, dims, edges, obs, dyn, prior, obs_seq)

    def test_star_topology_high_degree(self):
        # A degree-6 hub: the root feeds six observed children. "Any node degree works"
        # (ADR-015 / the additive information form) as a checked claim, not just the
        # chemotaxis degree-3 hub.
        rng = np.random.default_rng(7)
        k = 6
        dims = (1,) * (k + 1)
        edges = [(0, c, [[rng.uniform(0.5, 1.0)]], [[0.05]]) for c in range(1, k + 1)]
        obs = {c: (np.eye(1), [[0.1]]) for c in range(1, k + 1)}
        dyn = [([[0.6]], [[0.08]])] + [([[0.4]], [[0.05]]) for _ in range(k)]
        backend = _build(dims, edges, obs, dyn)
        prior = Belief(mean=np.zeros(k + 1), cov=np.eye(k + 1) * 2.0)
        obs_seq = [rng.standard_normal(k) for _ in range(5)]
        _check_sequence(backend, dims, edges, obs, dyn, prior, obs_seq)


class TestCarryPartition:
    """The carry partition (ADR-016): the off-diagonal precision block-sparsity kept
    across the time boundary. The safety-net gate first — the trivial single-cluster
    partition ``[[all nodes]]`` must reproduce the exact #25 path byte-for-byte, so
    adding the ``partition`` axis cannot perturb the exact endpoint.
    """

    def test_full_joint_partition_matches_unpartitioned(self):
        # [[all nodes]] zeros no between-cluster blocks -> identical to no partition.
        all_nodes = [list(range(len(_CHEMOTAXIS_DIMS)))]
        partitioned = _build(
            _CHEMOTAXIS_DIMS,
            _CHEMOTAXIS_EDGES,
            _CHEMOTAXIS_OBS,
            _CHEMOTAXIS_DYN,
            partition=all_nodes,
        )
        exact = _build(
            _CHEMOTAXIS_DIMS, _CHEMOTAXIS_EDGES, _CHEMOTAXIS_OBS, _CHEMOTAXIS_DYN
        )
        rng = np.random.default_rng(0)
        prior = Belief(mean=np.zeros(5), cov=np.eye(5))
        obs_seq = [rng.standard_normal(3) for _ in range(4)]
        belief_p, belief_e = prior, prior
        for y in obs_seq:
            belief_p = partitioned.infer_states(y, belief_p)
            belief_e = exact.infer_states(y, belief_e)
            np.testing.assert_array_equal(
                np.asarray(belief_p.mean), np.asarray(belief_e.mean)
            )
            np.testing.assert_array_equal(
                np.asarray(belief_p.cov), np.asarray(belief_e.cov)
            )

    def test_severed_mass_zero_for_full_joint_positive_for_cut(self):
        # The severed-mass diagnostic (ADR-016): the full-joint carry drops no coupling
        # (0.0), a singleton carry severs every between-node block (positive mass).
        full = _build(
            _CHEMOTAXIS_DIMS, _CHEMOTAXIS_EDGES, _CHEMOTAXIS_OBS, _CHEMOTAXIS_DYN
        )
        singletons = [[n] for n in range(len(_CHEMOTAXIS_DIMS))]
        cut = _build(
            _CHEMOTAXIS_DIMS,
            _CHEMOTAXIS_EDGES,
            _CHEMOTAXIS_OBS,
            _CHEMOTAXIS_DYN,
            partition=singletons,
        )
        rng = np.random.default_rng(1)
        prior = Belief(mean=np.zeros(5), cov=np.eye(5))
        y = rng.standard_normal(3)
        assert full.severed_mass(y, prior) == pytest.approx(0.0)
        assert cut.severed_mass(y, prior) > 0.0

    def test_within_slice_marginals_exact_under_any_partition(self):
        # ADR-017: the carry factors only what crosses the *time* boundary. Within a
        # single step every node's marginal must equal the exact full-joint marginal —
        # a cut drops correlation carried forward, not this slice's accuracy.
        exact = _build(
            _CHEMOTAXIS_DIMS, _CHEMOTAXIS_EDGES, _CHEMOTAXIS_OBS, _CHEMOTAXIS_DYN
        )
        cut = _build(
            _CHEMOTAXIS_DIMS,
            _CHEMOTAXIS_EDGES,
            _CHEMOTAXIS_OBS,
            _CHEMOTAXIS_DYN,
            partition=[[0, 1, 3, 4], [2]],
        )
        rng = np.random.default_rng(4)
        prior = Belief(mean=np.zeros(5), cov=np.eye(5))
        y = rng.standard_normal(3)
        belief_exact = exact.infer_states(y, prior)
        belief_cut = cut.infer_states(y, prior)
        for node in range(len(_CHEMOTAXIS_DIMS)):
            me = exact.marginal(node, belief_exact)
            mc = cut.marginal(node, belief_cut)
            np.testing.assert_allclose(
                np.asarray(mc.mean), np.asarray(me.mean), atol=1e-10
            )
            np.testing.assert_allclose(
                np.asarray(mc.cov), np.asarray(me.cov), atol=1e-10
            )

    def test_singleton_partition_marginals_via_bp_match_exact(self):
        # A singleton partition takes the cheap two-pass BP path (_is_factored). From a
        # block-diagonal prior its per-node marginals must equal the exact full-joint
        # marginals — BP computes the same numbers as the dense solve, without forming
        # or inverting the dense joint.
        exact = _build(
            _CHEMOTAXIS_DIMS, _CHEMOTAXIS_EDGES, _CHEMOTAXIS_OBS, _CHEMOTAXIS_DYN
        )
        singletons = _build(
            _CHEMOTAXIS_DIMS,
            _CHEMOTAXIS_EDGES,
            _CHEMOTAXIS_OBS,
            _CHEMOTAXIS_DYN,
            partition=[[n] for n in range(len(_CHEMOTAXIS_DIMS))],
        )
        assert singletons._is_factored  # the BP path is actually engaged
        rng = np.random.default_rng(9)
        prior = Belief(mean=np.zeros(5), cov=np.eye(5))
        y = rng.standard_normal(3)
        bp_belief = singletons.infer_states(y, prior)
        exact_belief = exact.infer_states(y, prior)
        for node in range(len(_CHEMOTAXIS_DIMS)):
            got = singletons.marginal(node, bp_belief)
            want = exact.marginal(node, exact_belief)
            np.testing.assert_allclose(
                np.asarray(got.mean), np.asarray(want.mean), atol=1e-7
            )
            np.testing.assert_allclose(
                np.asarray(got.cov), np.asarray(want.cov), atol=1e-7
            )

    def test_filter_sequence_profiles_severed_mass_over_a_run(self):
        # The per-run severed-mass profile (ADR-016): one traced pass over a sequence
        # returns the stacked posterior moments and a length-T severed-mass profile.
        # The full-joint run severs nothing at every step; a singleton cut severs at
        # each.
        full = _build(
            _CHEMOTAXIS_DIMS, _CHEMOTAXIS_EDGES, _CHEMOTAXIS_OBS, _CHEMOTAXIS_DYN
        )
        singletons = [[n] for n in range(len(_CHEMOTAXIS_DIMS))]
        cut = _build(
            _CHEMOTAXIS_DIMS,
            _CHEMOTAXIS_EDGES,
            _CHEMOTAXIS_OBS,
            _CHEMOTAXIS_DYN,
            partition=singletons,
        )
        rng = np.random.default_rng(2)
        prior = Belief(mean=np.zeros(5), cov=np.eye(5))
        obs_seq = np.stack([rng.standard_normal(3) for _ in range(4)])  # (T=4, 3)

        means, covs, severed = cut.filter_sequence(obs_seq, prior)
        assert np.asarray(severed).shape == (4,)
        assert np.asarray(means).shape == (4, 5)
        assert np.asarray(covs).shape == (4, 5, 5)
        assert np.all(np.asarray(severed) > 0.0)

        _means, _covs, severed_full = full.filter_sequence(obs_seq, prior)
        np.testing.assert_allclose(np.asarray(severed_full), 0.0, atol=1e-12)

        # The trajectory must equal iterating infer_states by hand — the sequence can't
        # compute anything the single-step path doesn't.
        belief = prior
        for step, y in enumerate(obs_seq):
            belief = cut.infer_states(y, belief)
            np.testing.assert_allclose(
                np.asarray(means[step]), np.asarray(belief.mean), atol=1e-10
            )
            np.testing.assert_allclose(
                np.asarray(covs[step]), np.asarray(belief.cov), atol=1e-10
            )

    def test_filter_sequence_returns_moments_not_a_belief(self):
        # A Belief with a leading time axis broke Belief's invariant, and the backend's
        # own `marginal` refused it.
        backend = _build(
            _CHEMOTAXIS_DIMS, _CHEMOTAXIS_EDGES, _CHEMOTAXIS_OBS, _CHEMOTAXIS_DYN
        )
        prior = Belief(mean=np.zeros(5), cov=np.eye(5))
        outputs = backend.filter_sequence(np.zeros((2, 3)), prior)
        assert not any(isinstance(output, Belief) for output in outputs)

    def test_filter_sequence_refuses_a_prior_over_the_wrong_state(self):
        backend = _build(
            _CHEMOTAXIS_DIMS, _CHEMOTAXIS_EDGES, _CHEMOTAXIS_OBS, _CHEMOTAXIS_DYN
        )
        with pytest.raises(ValueError, match="5-D state"):
            backend.filter_sequence(
                np.zeros((2, 3)), Belief(mean=np.zeros(2), cov=np.eye(2))
            )

    def test_filter_sequence_refuses_an_action_of_the_wrong_width(self):
        backend = _build(
            _CHEMOTAXIS_DIMS,
            _CHEMOTAXIS_EDGES,
            _CHEMOTAXIS_OBS,
            _CHEMOTAXIS_DYN,
            control_matrix=np.array([[1.0], [0.0], [0.0], [0.0], [0.0]]),
        )
        prior = Belief(mean=np.zeros(5), cov=np.eye(5))
        with pytest.raises(ValueError, match="action must be a 1-D vector of length 1"):
            backend.filter_sequence(np.zeros((2, 3)), prior, np.zeros((2, 2)))

    def test_filter_sequence_refuses_an_empty_run(self):
        backend = _build(
            _CHEMOTAXIS_DIMS, _CHEMOTAXIS_EDGES, _CHEMOTAXIS_OBS, _CHEMOTAXIS_DYN
        )
        prior = Belief(mean=np.zeros(5), cov=np.eye(5))
        with pytest.raises(ValueError, match="at least one step"):
            backend.filter_sequence(np.zeros((0, 3)), prior)

    def test_filter_sequence_takes_its_arguments_in_infer_states_order(self):
        # Readings first, then the belief, then the actions, on both methods.
        backend = _build(
            _CHEMOTAXIS_DIMS, _CHEMOTAXIS_EDGES, _CHEMOTAXIS_OBS, _CHEMOTAXIS_DYN
        )
        prior = Belief(mean=np.zeros(5), cov=np.eye(5))
        means, _covs, _severed = backend.filter_sequence(
            observations=np.ones((1, 3)), prior=prior, actions=None
        )
        np.testing.assert_allclose(
            np.asarray(means[0]),
            np.asarray(backend.infer_states(observation=np.ones(3), prior=prior).mean),
            atol=1e-10,
        )


class TestPredictedBelief:
    """B1 (issue #26): the pre-observation joint the EFE reads — predict + structural
    couplings, before the real observation. Checked against an independent NumPy oracle.
    """

    def test_predicted_belief_matches_pre_observation_oracle(self):
        control = np.array([[1.0], [0.0], [0.0], [0.0], [0.0]])  # action drives CheA
        backend = _build(
            _CHEMOTAXIS_DIMS,
            _CHEMOTAXIS_EDGES,
            _CHEMOTAXIS_OBS,
            _CHEMOTAXIS_DYN,
            control_matrix=control,
        )
        prior = Belief(mean=np.array([0.2, -0.1, 0.3, 0.0, 0.1]), cov=np.eye(5) * 1.5)
        action = np.array([0.7])

        predicted = backend.predicted_belief(prior, action)
        mu_o, sig_o = _predicted_joint_oracle(
            _CHEMOTAXIS_DIMS,
            _CHEMOTAXIS_EDGES,
            _CHEMOTAXIS_DYN,
            prior.mean,
            prior.cov,
            control,
            action,
        )
        np.testing.assert_allclose(np.asarray(predicted.mean), mu_o, atol=1e-7)
        np.testing.assert_allclose(np.asarray(predicted.cov), sig_o, atol=1e-7)


class TestProtocolAndReadout:
    def test_is_inference_backend(self):
        backend = _build(
            _CHEMOTAXIS_DIMS, _CHEMOTAXIS_EDGES, _CHEMOTAXIS_OBS, _CHEMOTAXIS_DYN
        )
        assert isinstance(backend, InferenceBackend)

    @pytest.mark.parametrize("second_way", ["readout", "readout_node"])
    def test_marginal_is_the_one_way_to_read_a_node(self, second_way):
        # `readout(belief)` was `marginal(readout_node, belief)` under a second name,
        # with a constructor argument that did nothing else.
        backend = _build(
            _CHEMOTAXIS_DIMS, _CHEMOTAXIS_EDGES, _CHEMOTAXIS_OBS, _CHEMOTAXIS_DYN
        )
        assert not hasattr(backend, second_way)

    def _one_node_backend(self, prior, control_matrix=None):
        graph = CouplingGraph(
            dims=(2,),
            couplings=(),
            observation_models={
                0: FixedSensor([[1.0, 0.0]], observation_noise=[[0.1]])
            },
        )
        transition = GaussianTransition(np.eye(2), dynamics_noise=np.eye(2) * 0.1)
        return CouplingGraphBackend(
            graph, (transition,), prior=prior, control_matrix=control_matrix
        )

    def test_control_matrix_over_the_wrong_rows_is_refused_by_name(self):
        with pytest.raises(ValueError, match="control_matrix must be a 2-D matrix"):
            self._one_node_backend(
                Belief(mean=np.zeros(2), cov=np.eye(2)), control_matrix=[[1.0]]
            )

    def test_prior_starts_the_agent_and_the_flat_model(self):
        # Both used to start from an N(0, I) placeholder the caller never chose.
        prior = Belief(mean=[0.5, -0.5], cov=np.eye(2) * 3.0)
        backend = self._one_node_backend(prior)
        assert backend.model.prior is prior
        assert backend.to_flat_model().prior is prior
        assert Agent.from_backend(backend).belief is prior

    def test_prior_that_is_not_a_belief_is_refused(self):
        with pytest.raises(TypeError, match="prior must be a Belief"):
            self._one_node_backend(np.zeros(2))

    def test_prior_over_the_wrong_dimension_is_refused(self):
        with pytest.raises(ValueError, match="2-D joint state"):
            self._one_node_backend(Belief(mean=[0.0], cov=[[1.0]]))

    def test_a_graph_with_no_sensor_is_refused_by_name(self):
        # It used to fail with a zero-size reduction inside the model's noise check.
        graph = CouplingGraph(
            dims=(1, 1),
            couplings=(Coupling(0, 1, [[1.0]], coupling_noise=[[0.5]]),),
            observation_models={},
        )
        transition = GaussianTransition([[0.9]], dynamics_noise=[[0.1]])
        with pytest.raises(ValueError, match="no node of the graph carries a sensor"):
            CouplingGraphBackend(
                graph, (transition, transition), prior=unit_prior(graph)
            )

    @pytest.mark.parametrize(
        "round_trip",
        [copy.deepcopy, lambda backend: pickle.loads(pickle.dumps(backend))],
        ids=["deepcopy", "pickle"],
    )
    def test_survives_a_copy_and_a_pickle(self, round_trip):
        # An agent reaches a multiprocessing worker by pickle, backend and graph both.
        backend = _build(
            _CHEMOTAXIS_DIMS, _CHEMOTAXIS_EDGES, _CHEMOTAXIS_OBS, _CHEMOTAXIS_DYN
        )
        prior = Belief(mean=np.zeros(5), cov=np.eye(5) * 2.0)
        reading = np.array([0.4, -0.2, 0.1])
        copied = round_trip(backend)
        np.testing.assert_array_equal(
            copied.infer_states(reading, prior).mean,
            backend.infer_states(reading, prior).mean,
        )

    def test_prior_not_mutated(self):
        backend = _build(
            _CHEMOTAXIS_DIMS, _CHEMOTAXIS_EDGES, _CHEMOTAXIS_OBS, _CHEMOTAXIS_DYN
        )
        prior = Belief(mean=np.zeros(5), cov=np.eye(5) * 2.0)
        before = np.asarray(prior.cov).copy()
        backend.infer_states(np.zeros(3), prior)
        np.testing.assert_array_equal(np.asarray(prior.cov), before)


class TestOneNamePerQuantity:
    """Each size the backend knows has one home: the model it hands out, or its graph.

    The backend carried `n_total`, `dims` and `n_observations` beside `model.n_states`,
    `graph.dims` and `model.n_observations`, the same numbers under other names.
    """

    def test_sizes_are_read_off_the_model_and_the_graph(self):
        backend = _build(
            _CHEMOTAXIS_DIMS, _CHEMOTAXIS_EDGES, _CHEMOTAXIS_OBS, _CHEMOTAXIS_DYN
        )
        assert backend.model.n_states == 5
        assert backend.model.n_observations == 3
        assert backend.graph.dims == _CHEMOTAXIS_DIMS

    @pytest.mark.parametrize("duplicate", ["n_total", "dims", "n_observations"])
    def test_the_backend_carries_no_second_name_for_a_size(self, duplicate):
        backend = _build(
            _CHEMOTAXIS_DIMS, _CHEMOTAXIS_EDGES, _CHEMOTAXIS_OBS, _CHEMOTAXIS_DYN
        )
        assert not hasattr(backend, duplicate)


def _noise_rising_off_zero(x, params):
    """``R(x) = 0.5 + x₀²``, so ``R(0)`` and ``R`` anywhere else differ."""
    return jnp.array([[0.5 + x[0] ** 2]])


class TestObservationChannel:
    """C is read off the model and R at a mean, so no accessor hands out ``R(0)``.

    The backend's `observation_model` returned `(C, R)` with R linearized at zero,
    beside `model.observation_model`, which is a sensor object.
    """

    def test_c_is_the_model_s_observation_matrix(self):
        backend = _build(
            _CHEMOTAXIS_DIMS, _CHEMOTAXIS_EDGES, _CHEMOTAXIS_OBS, _CHEMOTAXIS_DYN
        )
        expected = np.zeros((3, 5))
        expected[[0, 1, 2], [2, 3, 4]] = 1.0  # nodes 2, 3, 4 in ascending order
        np.testing.assert_array_equal(
            np.asarray(backend.model.observation_matrix), expected
        )

    def test_r_is_read_at_the_mean_it_is_asked_about(self):
        backend = _build_rx(
            (2,),
            [],
            {0: (np.array([[1.0, 0.0]]), _noise_rising_off_zero, ())},
            [(np.eye(2), np.eye(2) * 0.1)],
        )
        np.testing.assert_allclose(
            np.asarray(backend.observation_noise_at([3.0, 3.0])), [[9.5]]
        )

    def test_no_tuple_freezes_r_at_zero(self):
        backend = _build_rx(
            (2,),
            [],
            {0: (np.array([[1.0, 0.0]]), _noise_rising_off_zero, ())},
            [(np.eye(2), np.eye(2) * 0.1)],
        )
        assert not hasattr(backend, "observation_model")


class TestStackObservations:
    """Readings keyed by node, the format `CouplingGraph.infer` takes, stack here.

    The stacked vector puts each observed node's reading in ascending node order, a
    rule that lived in a private layout. Stacking `{3: motor, 2: cheB}` in its
    insertion order gave each node the other's reading and passed every shape check.
    """

    def _chemotaxis(self):
        return _build(
            _CHEMOTAXIS_DIMS, _CHEMOTAXIS_EDGES, _CHEMOTAXIS_OBS, _CHEMOTAXIS_DYN
        )

    def test_readings_stack_in_node_order_whatever_the_dict_order(self):
        stacked = self._chemotaxis().stack_observations({4: [0.4], 2: [0.2], 3: [0.3]})
        np.testing.assert_array_equal(np.asarray(stacked), [0.2, 0.3, 0.4])

    def test_a_scalar_is_a_one_row_reading(self):
        stacked = self._chemotaxis().stack_observations({3: 0.3, 4: 0.4, 2: 0.2})
        np.testing.assert_array_equal(np.asarray(stacked), [0.2, 0.3, 0.4])

    def test_the_stack_filters_like_the_vector_it_replaces(self):
        backend = self._chemotaxis()
        prior = Belief(mean=np.zeros(5), cov=np.eye(5))
        by_node = backend.infer_states(
            backend.stack_observations({4: 0.4, 3: 0.3, 2: 0.2}), prior
        )
        by_hand = backend.infer_states([0.2, 0.3, 0.4], prior)
        np.testing.assert_array_equal(
            np.asarray(by_node.mean), np.asarray(by_hand.mean)
        )

    def test_a_missing_observed_node_is_refused_by_name(self):
        with pytest.raises(
            ValueError, match=r"no reading for observed node\(s\) \[3\]"
        ):
            self._chemotaxis().stack_observations({2: 0.2, 4: 0.4})

    def test_a_reading_on_an_unobserved_node_is_refused_by_name(self):
        with pytest.raises(ValueError, match=r"node\(s\) \[0\] carry no sensor"):
            self._chemotaxis().stack_observations({0: 0.0, 2: 0.2, 3: 0.3, 4: 0.4})

    def test_a_reading_of_the_wrong_width_is_refused_by_name(self):
        rng = np.random.default_rng(0)
        backend = _build(
            (1, 2),
            [(0, 1, rng.standard_normal((2, 1)), _spd(rng, 2))],
            {1: (np.eye(2), _spd(rng, 2))},
            [([[0.5]], [[0.1]]), (np.eye(2) * 0.5, np.eye(2) * 0.1)],
        )
        with pytest.raises(ValueError, match=r"node 1 .* reads 2 value"):
            backend.stack_observations({1: 0.5})


class TestDefiniteNoise:
    """The graph path inverts each node's Q and each edge's noise, and says so.

    The flat model accepts a semi-definite Q. The graph's refusal of one blamed "a
    noiseless or degenerate sensor", an object the caller had not passed.
    """

    def test_a_semi_definite_node_noise_names_the_inversion(self):
        with pytest.raises(
            ValueError,
            match="dynamics_noise must be positive-definite because it is inverted",
        ) as refusal:
            GaussianTransition(np.eye(2), dynamics_noise=np.diag([1e-4, 0.0]))
        assert "sensor" not in str(refusal.value)

    def test_a_semi_definite_edge_noise_names_the_inversion(self):
        with pytest.raises(
            ValueError,
            match="coupling_noise must be positive-definite because it is inverted",
        ) as refusal:
            Coupling(0, 1, [[1.0]], coupling_noise=[[0.0]])
        assert "sensor" not in str(refusal.value)


class TestNodeLookup:
    """A node outside the graph is refused by name, never read as an empty block.

    `block(-1)` used to return an empty range, which aims an epistemic term at nothing,
    and `block(5)` on five nodes raised a bare `IndexError`.
    """

    @pytest.mark.parametrize("node", [-1, 5])
    def test_block_refuses_a_node_outside_the_graph(self, node):
        backend = _build(
            _CHEMOTAXIS_DIMS, _CHEMOTAXIS_EDGES, _CHEMOTAXIS_OBS, _CHEMOTAXIS_DYN
        )
        with pytest.raises(ValueError, match=rf"node {node} .* 0\.\.4"):
            backend.block(node)

    @pytest.mark.parametrize("node", [-1, 5])
    def test_marginal_refuses_a_node_outside_the_graph(self, node):
        backend = _build(
            _CHEMOTAXIS_DIMS, _CHEMOTAXIS_EDGES, _CHEMOTAXIS_OBS, _CHEMOTAXIS_DYN
        )
        prior = Belief(mean=np.zeros(5), cov=np.eye(5))
        with pytest.raises(ValueError, match=rf"node {node} .* 0\.\.4"):
            backend.marginal(node, prior)


class TestTransforms:
    def test_jit_grad_vmap_through_infer_states(self):
        backend = _build(
            _CHEMOTAXIS_DIMS, _CHEMOTAXIS_EDGES, _CHEMOTAXIS_OBS, _CHEMOTAXIS_DYN
        )
        prior = Belief(mean=jnp.zeros(5), cov=jnp.eye(5) * 2.0)

        def root_mean(y):
            return backend.marginal(0, backend.infer_states(y, prior)).mean

        y = jnp.array([1.25, 1.15, 0.95])
        np.testing.assert_allclose(
            np.asarray(jax.jit(root_mean)(y)), np.asarray(root_mean(y)), atol=1e-10
        )
        grad = jax.grad(lambda yy: root_mean(yy).sum())(y)
        assert bool(jnp.all(jnp.isfinite(grad)))

        ys = jnp.array([[1.25, 1.15, 0.95], [0.1, -0.2, 0.3], [0.0, 0.0, 0.0]])
        batched = jax.vmap(root_mean)(ys)
        expected = jnp.stack([root_mean(one) for one in ys])
        np.testing.assert_allclose(
            np.asarray(batched), np.asarray(expected), atol=1e-10
        )

    def test_jit_grad_through_filter_sequence_scan(self):
        # The traced scan path: the severed-mass profile is produced *inside* the trace
        # (no host syncs), so filter_sequence must jit and differentiate (ADR-012 gate).
        cut = _build(
            _CHEMOTAXIS_DIMS,
            _CHEMOTAXIS_EDGES,
            _CHEMOTAXIS_OBS,
            _CHEMOTAXIS_DYN,
            partition=[[0, 1, 3, 4], [2]],
        )
        prior = Belief(mean=jnp.zeros(5), cov=jnp.eye(5) * 2.0)
        obs_seq = jnp.array([[1.25, 1.15, 0.95], [0.1, -0.2, 0.3], [0.0, 0.0, 0.0]])

        def total_severed(ys):
            _means, _covs, severed = cut.filter_sequence(ys, prior)
            return severed.sum()

        np.testing.assert_allclose(
            np.asarray(jax.jit(total_severed)(obs_seq)),
            np.asarray(total_severed(obs_seq)),
            atol=1e-10,
        )
        grad = jax.grad(total_severed)(obs_seq)
        assert bool(jnp.all(jnp.isfinite(grad)))

    def test_jit_grad_vmap_through_bp_path(self):
        # The fully-factored (singleton) path runs two-pass BP, a static schedule over
        # traced data, so it must jit / grad / vmap like the dense path (ADR-012 gate).
        backend = _build(
            _CHEMOTAXIS_DIMS,
            _CHEMOTAXIS_EDGES,
            _CHEMOTAXIS_OBS,
            _CHEMOTAXIS_DYN,
            partition=[[n] for n in range(len(_CHEMOTAXIS_DIMS))],
        )
        prior = Belief(mean=jnp.zeros(5), cov=jnp.eye(5) * 2.0)

        def root_mean(y):
            return backend.marginal(0, backend.infer_states(y, prior)).mean

        y = jnp.array([1.25, 1.15, 0.95])
        np.testing.assert_allclose(
            np.asarray(jax.jit(root_mean)(y)), np.asarray(root_mean(y)), atol=1e-10
        )
        grad = jax.grad(lambda yy: root_mean(yy).sum())(y)
        assert bool(jnp.all(jnp.isfinite(grad)))

        ys = jnp.array([[1.25, 1.15, 0.95], [0.1, -0.2, 0.3], [0.0, 0.0, 0.0]])
        batched = jax.vmap(root_mean)(ys)
        expected = jnp.stack([root_mean(one) for one in ys])
        np.testing.assert_allclose(
            np.asarray(batched), np.asarray(expected), atol=1e-10
        )


def _check_against_flat(backend, flat_backend, prior, obs_seq, *, action_seq=None):
    """Drive the backend and a flat backend on ``to_flat_model`` in lockstep.

    The flat backend consumes the padded observation (real readings + the structural
    couplings' zeros); both carry the joint belief, so they must agree step for step.
    """
    bel_b = bel_f = prior
    for step, y in enumerate(obs_seq):
        action = None if action_seq is None else action_seq[step]
        bel_b = backend.infer_states(y, bel_b, action)
        bel_f = flat_backend.infer_states(backend.flat_observation(y), bel_f, action)
        np.testing.assert_allclose(
            np.asarray(bel_b.mean), np.asarray(bel_f.mean), atol=1e-7
        )
        np.testing.assert_allclose(
            np.asarray(bel_b.cov), np.asarray(bel_f.cov), atol=1e-7
        )


class TestFlatModelCrossCheck:
    """The filter matches KalmanBackend / RxInferBackend on ``to_flat_model``.

    A structural coupling ``child = W·parent + noise`` is a within-slice
    pseudo-observation ``child − W·parent ≈ 0``, so the branching tree flattens into one
    ``LinearGaussianModel`` that the moment-form Kalman path and RxInfer's engine filter
    — two independent oracles for the whole temporal recursion (ADR-016).
    """

    def test_matches_kalman_on_flat_model(self):
        rng = np.random.default_rng(4)
        backend = _build(
            _CHEMOTAXIS_DIMS, _CHEMOTAXIS_EDGES, _CHEMOTAXIS_OBS, _CHEMOTAXIS_DYN
        )
        flat = KalmanBackend(backend.to_flat_model())
        prior = Belief(mean=np.zeros(5), cov=np.eye(5) * 2.0)
        obs_seq = [rng.standard_normal(3) for _ in range(6)]
        _check_against_flat(backend, flat, prior, obs_seq)

    def test_matches_kalman_on_flat_model_with_control(self):
        rng = np.random.default_rng(5)
        control = np.array([[1.0], [0.0], [0.0], [0.0], [0.0]])
        backend = _build(
            _CHEMOTAXIS_DIMS,
            _CHEMOTAXIS_EDGES,
            _CHEMOTAXIS_OBS,
            _CHEMOTAXIS_DYN,
            control_matrix=control,
        )
        flat = KalmanBackend(backend.to_flat_model())
        prior = Belief(mean=np.zeros(5), cov=np.eye(5) * 2.0)
        obs_seq = [rng.standard_normal(3) for _ in range(6)]
        action_seq = [rng.standard_normal(1) for _ in range(6)]
        _check_against_flat(backend, flat, prior, obs_seq, action_seq=action_seq)

    @pytest.mark.rxinfer
    def test_matches_rxinfer_on_flat_model(self):
        rng = np.random.default_rng(6)
        backend = _build(
            _CHEMOTAXIS_DIMS, _CHEMOTAXIS_EDGES, _CHEMOTAXIS_OBS, _CHEMOTAXIS_DYN
        )
        flat = RxInferBackend(backend.to_flat_model())
        prior = Belief(mean=np.zeros(5), cov=np.eye(5) * 2.0)
        obs_seq = [rng.standard_normal(3) for _ in range(3)]
        _check_against_flat(backend, flat, prior, obs_seq)


# --- State-dependent sensing R(x) on the FFG backend (issue #27, Phase 2) -------


def _rx_noise(x, params):
    """Node-local ``R(x) = R0·(1 + gain·xᵀx)`` — PD, sharpens as the node nears 0.

    Shared model spec (like ``obs_specs``' C/R), not part of the filter algebra under
    test — the oracles below drive an independent NumPy recursion around it.
    """
    return params["R0"] * (1.0 + params["gain"] * jnp.dot(x, x))


def _rx_relaxation_oracle(
    dims, edges, rx_obs, dyn, prior_mean, prior_cov, control_matrix, obs_seq, action_seq
):
    """Exact N-step driven-relaxation filter with state-dependent ``R(μ⁺)`` — NumPy.

    Like ``_driven_relaxation_oracle``, but each observed node's noise is evaluated at
    that node's block of the *coupling-included* predicted mean μ⁺ — the FFG
    linearization point (issue #27, ADR-003 broke here) — and only then folded like a
    fixed R. So each step is predict -> couplings -> μ⁺ -> R(μ⁺) -> update.
    ``rx_obs[node] = (C, noise_fn, params)``.
    """
    offs = np.cumsum([0, *dims])
    dim = int(offs[-1])

    def blk(i):
        return slice(int(offs[i]), int(offs[i + 1]))

    force = _block_diag([np.asarray(a, float) for a, _ in dyn])  # F = blkdiag(A_i)
    proc = _block_diag([np.asarray(q, float) for _, q in dyn])  # Q = blkdiag(Q_i)

    lam_struct = np.zeros((dim, dim))
    for parent, child, w, qs in edges:
        w = np.asarray(w, float)
        qs_inv = np.linalg.inv(np.asarray(qs, float))
        lam_struct[blk(parent), blk(parent)] += w.T @ qs_inv @ w
        lam_struct[blk(child), blk(child)] += qs_inv
        lam_struct[blk(parent), blk(child)] += -w.T @ qs_inv
        lam_struct[blk(child), blk(parent)] += -qs_inv @ w

    mu = np.asarray(prior_mean, float).copy()
    sig = np.asarray(prior_cov, float).copy()
    observed = sorted(rx_obs)
    states = []
    for step, y_stacked in enumerate(obs_seq):
        shift = np.zeros(dim)
        if control_matrix is not None:
            shift = np.asarray(control_matrix, float) @ np.asarray(
                action_seq[step], float
            )
        mu_pred = force @ mu + shift
        sig_pred = force @ sig @ force.T + proc
        pred_precision = np.linalg.inv(sig_pred)

        # μ⁺ = predict + structural couplings, BEFORE observing (the R eval point).
        lam_plus = pred_precision + lam_struct
        h_plus = pred_precision @ mu_pred
        mu_plus = np.linalg.inv(lam_plus) @ h_plus

        lam = lam_plus.copy()
        h = h_plus.copy()
        cursor = 0
        for node in observed:
            c_mat, noise_fn, params = rx_obs[node]
            c_mat = np.asarray(c_mat, float)
            m = c_mat.shape[0]
            y = np.asarray(y_stacked, float)[cursor : cursor + m]
            cursor += m
            r_mat = np.asarray(noise_fn(mu_plus[blk(node)], params), float)  # R(μ⁺)
            r_inv = np.linalg.inv(r_mat)
            lam[blk(node), blk(node)] += c_mat.T @ r_inv @ c_mat
            h[blk(node)] += c_mat.T @ r_inv @ y

        sig = np.linalg.inv(lam)
        mu = sig @ h
        states.append((mu.copy(), sig.copy()))
    return states, offs


def _build_rx(dims, edges, rx_obs, dyn, *, control_matrix=None, partition=None):
    """Build a ``CouplingGraphBackend`` with state-dependent (callable) observations."""
    couplings = tuple(
        Coupling(parent, child, w, coupling_noise=q) for parent, child, w, q in edges
    )
    observation_models = {
        node: CallableSensor(c, fn, prm) for node, (c, fn, prm) in rx_obs.items()
    }
    graph = CouplingGraph(
        dims=dims, couplings=couplings, observation_models=observation_models
    )
    transitions = tuple(GaussianTransition(a, dynamics_noise=q) for a, q in dyn)
    kwargs = {"control_matrix": control_matrix}
    if partition is not None:
        kwargs["partition"] = partition
    return CouplingGraphBackend(graph, transitions, prior=unit_prior(graph), **kwargs)


# A branching R(x) spec: the chemotaxis tree, but the three observed nodes each sense
# through a state-dependent (node-local) noise. The action drives the root (CheA).
_RX_OBS = {
    2: (np.eye(1), _rx_noise, {"R0": np.array([[0.10]]), "gain": 0.4}),
    3: (np.eye(1), _rx_noise, {"R0": np.array([[0.10]]), "gain": 0.4}),
    4: (np.eye(1), _rx_noise, {"R0": np.array([[0.10]]), "gain": 0.4}),
}
_RX_CONTROL = [[1.0], [0.0], [0.0], [0.0], [0.0]]  # B: action drives CheA (root)


def _goes_negative(state, params):
    """R(x) = 1 - x0², a covariance only for |x0| < 1."""
    return jnp.array([[1.0 - state[0] ** 2]])


def test_an_r_that_is_not_a_covariance_at_the_predicted_mean_is_refused():
    # ChainBackend refuses the same sensor. A finite but wrong belief was returned.
    backend = _build_rx(
        (1,),
        (),
        {0: (np.eye(1), _goes_negative, None)},
        ((np.eye(1), np.eye(1) * 0.05),),
    )
    far = Belief(mean=[2.0], cov=[[0.1]])
    with pytest.raises(ValueError, match="positive-definite"):
        backend.infer_states(jnp.array([0.0]), far)


class TestStateDependentSensing:
    """R(x) on the FFG backend: the chosen action moves the covariance (dual effect).

    The gate is two-sided. The fixed path must stay byte-identical (a constant noise_fn
    reproduces the fixed backend); the R(x) path must fold ``R`` evaluated at the
    coupling-included predicted mean μ⁺ — checked single-node against the trusted flat
    ``KalmanBackend(CallableSensor)`` and branching against an independent NumPy filter.
    """

    def test_constant_noise_reduces_to_fixed_backend(self):
        # The reduction: gain=0 (R ≡ R0) must reproduce the fixed-R backend step for
        # step, so nothing downstream can tell a constant R(x) from a fixed sensor.
        rng = np.random.default_rng(0)
        const = {
            n: (c, _rx_noise, {"R0": np.asarray(r), "gain": 0.0})
            for n, (c, r) in _CHEMOTAXIS_OBS.items()
        }
        rx = _build_rx(
            _CHEMOTAXIS_DIMS,
            _CHEMOTAXIS_EDGES,
            const,
            _CHEMOTAXIS_DYN,
            control_matrix=_RX_CONTROL,
        )
        fixed = _build(
            _CHEMOTAXIS_DIMS,
            _CHEMOTAXIS_EDGES,
            _CHEMOTAXIS_OBS,
            _CHEMOTAXIS_DYN,
            control_matrix=_RX_CONTROL,
        )
        prior = Belief(mean=np.zeros(5), cov=np.eye(5) * 2.0)
        belief_rx = belief_fixed = prior
        for _ in range(5):
            y = rng.standard_normal(3)
            a = rng.standard_normal(1)
            belief_rx = rx.infer_states(y, belief_rx, a)
            belief_fixed = fixed.infer_states(y, belief_fixed, a)
            np.testing.assert_allclose(
                np.asarray(belief_rx.mean), np.asarray(belief_fixed.mean), atol=1e-10
            )
            np.testing.assert_allclose(
                np.asarray(belief_rx.cov), np.asarray(belief_fixed.cov), atol=1e-10
            )

    def test_single_node_rx_matches_kalman_callable_sensor(self):
        # Single node, no couplings: μ⁺ = the predict mean, so the FFG's R(μ⁺) coincides
        # with the trusted flat KalmanBackend(CallableSensor)'s R(μ⁻). atol 1e-7.
        rng = np.random.default_rng(1)
        A = np.array([[1.0, 0.1], [0.0, 1.0]])
        Q = np.eye(2) * 0.1
        C = np.array([[1.0, 0.0]])
        B = np.array([[1.0], [0.0]])
        params = {"R0": np.array([[0.5]]), "gain": 0.3}
        rx = _build_rx(
            (2,), [], {0: (C, _rx_noise, params)}, [(A, Q)], control_matrix=B
        )
        model = LinearGaussianModel(
            dynamics_matrix=A,
            dynamics_noise=Q,
            prior=Belief(mean=np.zeros(2), cov=np.eye(2)),
            control_matrix=B,
            observation_model=CallableSensor(C, _rx_noise, params),
        )
        flat = KalmanBackend(model)
        belief_rx = belief_flat = Belief(mean=np.zeros(2), cov=np.eye(2))
        for _ in range(6):
            y = rng.standard_normal(1)
            a = rng.standard_normal(1)
            belief_rx = rx.infer_states(y, belief_rx, a)
            belief_flat = flat.infer_states(y, belief_flat, a)
            np.testing.assert_allclose(
                np.asarray(belief_rx.mean), np.asarray(belief_flat.mean), atol=1e-7
            )
            np.testing.assert_allclose(
                np.asarray(belief_rx.cov), np.asarray(belief_flat.cov), atol=1e-7
            )

    def test_a_prior_away_from_the_origin_builds_and_flattens(self):
        # The flat model reads R at the prior mean from the graph's own sensor, so a
        # representative R(0) beside it would be a second value for one field.
        params = {"R0": np.array([[0.5]]), "gain": 0.3}
        graph = CouplingGraph(
            dims=(2,),
            couplings=(),
            observation_models={0: CallableSensor([[1.0, 0.0]], _rx_noise, params)},
        )
        prior = Belief(mean=[1.0, 2.0], cov=np.eye(2))
        backend = CouplingGraphBackend(
            graph,
            (GaussianTransition(np.eye(2), dynamics_noise=np.eye(2) * 0.1),),
            prior=prior,
        )
        flat = backend.to_flat_model()
        np.testing.assert_array_equal(
            flat.observation_noise, _rx_noise(prior.mean, params)
        )

    def test_branching_rx_matches_numpy_oracle(self):
        # Couplings + R(x) together: the branching backend must match the independent
        # NumPy filter evaluating R at the coupling-included μ⁺ each step. atol 1e-7.
        rng = np.random.default_rng(2)
        rx = _build_rx(
            _CHEMOTAXIS_DIMS,
            _CHEMOTAXIS_EDGES,
            _RX_OBS,
            _CHEMOTAXIS_DYN,
            control_matrix=_RX_CONTROL,
        )
        prior = Belief(mean=np.zeros(5), cov=np.eye(5) * 2.0)
        obs_seq = [rng.standard_normal(3) for _ in range(6)]
        action_seq = [rng.standard_normal(1) for _ in range(6)]
        states, offs = _rx_relaxation_oracle(
            _CHEMOTAXIS_DIMS,
            _CHEMOTAXIS_EDGES,
            _RX_OBS,
            _CHEMOTAXIS_DYN,
            prior.mean,
            prior.cov,
            _RX_CONTROL,
            obs_seq,
            action_seq,
        )
        belief = prior
        for step, y in enumerate(obs_seq):
            belief = rx.infer_states(y, belief, action_seq[step])
            mu_o, sig_o = states[step]
            for node in range(5):
                nb = rx.marginal(node, belief)
                lo, hi = int(offs[node]), int(offs[node + 1])
                np.testing.assert_allclose(np.asarray(nb.mean), mu_o[lo:hi], atol=1e-7)
                np.testing.assert_allclose(
                    np.asarray(nb.cov), sig_o[lo:hi, lo:hi], atol=1e-7
                )

    def test_predicted_belief_is_R_independent(self):
        # μ⁺/Σ⁺ are the pre-observation joint: R never enters the predict, so the R(x)
        # backend's predicted_belief equals the independent predict+couplings oracle.
        rx = _build_rx(
            _CHEMOTAXIS_DIMS,
            _CHEMOTAXIS_EDGES,
            _RX_OBS,
            _CHEMOTAXIS_DYN,
            control_matrix=_RX_CONTROL,
        )
        prior = Belief(mean=np.arange(5, dtype=float), cov=np.eye(5) * 1.5)
        action = np.array([0.7])
        mu_o, sig_o = _predicted_joint_oracle(
            _CHEMOTAXIS_DIMS,
            _CHEMOTAXIS_EDGES,
            _CHEMOTAXIS_DYN,
            prior.mean,
            prior.cov,
            _RX_CONTROL,
            action,
        )
        got = rx.predicted_belief(prior, action)
        np.testing.assert_allclose(np.asarray(got.mean), mu_o, atol=1e-7)
        np.testing.assert_allclose(np.asarray(got.cov), sig_o, atol=1e-7)

    def test_rx_infer_states_is_jittable(self):
        rx = _build_rx(
            (2,),
            [],
            {
                0: (
                    np.array([[1.0, 0.0]]),
                    _rx_noise,
                    {"R0": np.array([[0.5]]), "gain": 0.3},
                )
            },
            [(np.array([[1.0, 0.1], [0.0, 1.0]]), np.eye(2) * 0.1)],
            control_matrix=[[1.0], [0.0]],
        )
        prior = Belief(mean=np.zeros(2), cov=np.eye(2))
        eager = rx.infer_states(np.array([0.3]), prior, np.array([0.5]))
        jitted = jax.jit(rx.infer_states)(jnp.array([0.3]), prior, jnp.array([0.5]))
        np.testing.assert_allclose(
            np.asarray(jitted.mean), np.asarray(eager.mean), atol=1e-10
        )

    def test_predicted_belief_vmaps_over_actions(self):
        # Phase 3 needs Σ⁺/μ⁺ vmapped over a candidate-action grid; the R(x) backend's
        # predicted_belief must ride vmap (predict is R-independent, so this is safe).
        rx = _build_rx(
            _CHEMOTAXIS_DIMS,
            _CHEMOTAXIS_EDGES,
            _RX_OBS,
            _CHEMOTAXIS_DYN,
            control_matrix=_RX_CONTROL,
        )
        prior = Belief(mean=np.zeros(5), cov=np.eye(5))
        actions = jnp.linspace(-2.0, 2.0, 7)[:, None]
        means = jax.vmap(lambda a: rx.predicted_belief(prior, a).mean)(actions)
        assert means.shape == (7, 5)
        assert bool(jnp.all(jnp.isfinite(means)))

    def test_factored_rx_matches_numpy_oracle_one_step(self):
        # Factored (singleton) partition + R(x): the two-pass linearization (pass 1
        # reads μ⁺ marginals, pass 2 folds R(μ⁺)) must match the NumPy R(μ⁺) filter.
        # Single step from a block-diagonal prior (factored marginals exact), atol 1e-7.
        singletons = [[0], [1], [2], [3], [4]]
        rx = _build_rx(
            _CHEMOTAXIS_DIMS,
            _CHEMOTAXIS_EDGES,
            _RX_OBS,
            _CHEMOTAXIS_DYN,
            control_matrix=_RX_CONTROL,
            partition=singletons,
        )
        prior = Belief(mean=np.zeros(5), cov=np.eye(5))  # block-diagonal
        y, a = np.array([0.4, -0.2, 0.3]), np.array([0.6])
        states, offs = _rx_relaxation_oracle(
            _CHEMOTAXIS_DIMS,
            _CHEMOTAXIS_EDGES,
            _RX_OBS,
            _CHEMOTAXIS_DYN,
            prior.mean,
            prior.cov,
            _RX_CONTROL,
            [y],
            [a],
        )
        belief = rx.infer_states(y, prior, a)
        mu_o, sig_o = states[0]
        for node in range(5):
            nb = rx.marginal(node, belief)
            lo, hi = int(offs[node]), int(offs[node + 1])
            np.testing.assert_allclose(np.asarray(nb.mean), mu_o[lo:hi], atol=1e-7)
            np.testing.assert_allclose(
                np.asarray(nb.cov), sig_o[lo:hi, lo:hi], atol=1e-7
            )

    def test_factored_constant_noise_is_a_no_op_extra_pass(self):
        # The affine no-op invariant: with a constant noise_fn (constant Jacobian) the
        # extra μ⁺ pass changes nothing, so the factored R(x) path reproduces the
        # factored fixed path bit-for-bit — the pass is free when R doesn't vary.
        rng = np.random.default_rng(3)
        singletons = [[0], [1], [2], [3], [4]]
        const = {
            n: (c, _rx_noise, {"R0": np.asarray(r), "gain": 0.0})
            for n, (c, r) in _CHEMOTAXIS_OBS.items()
        }
        rx = _build_rx(
            _CHEMOTAXIS_DIMS,
            _CHEMOTAXIS_EDGES,
            const,
            _CHEMOTAXIS_DYN,
            control_matrix=_RX_CONTROL,
            partition=singletons,
        )
        fixed = _build(
            _CHEMOTAXIS_DIMS,
            _CHEMOTAXIS_EDGES,
            _CHEMOTAXIS_OBS,
            _CHEMOTAXIS_DYN,
            control_matrix=_RX_CONTROL,
            partition=singletons,
        )
        belief_rx = belief_fixed = Belief(mean=np.zeros(5), cov=np.eye(5))
        for _ in range(4):
            y, a = rng.standard_normal(3), rng.standard_normal(1)
            belief_rx = rx.infer_states(y, belief_rx, a)
            belief_fixed = fixed.infer_states(y, belief_fixed, a)
            np.testing.assert_allclose(
                np.asarray(belief_rx.mean), np.asarray(belief_fixed.mean), atol=1e-10
            )
            np.testing.assert_allclose(
                np.asarray(belief_rx.cov), np.asarray(belief_fixed.cov), atol=1e-10
            )

    def test_to_flat_model_rejects_coupled_state_dependent_sensor(self):
        # The theorem: with couplings, R(x) at μ⁺ is unreachable by the pseudo-obs flat
        # route (flat Kalman linearizes at μ⁻). It must raise the typed category error,
        # never silently freeze R — the guard that stops a future "make it return
        # something" edit from reintroducing a wrong oracle.
        rx = _build_rx(
            _CHEMOTAXIS_DIMS,
            _CHEMOTAXIS_EDGES,
            _RX_OBS,
            _CHEMOTAXIS_DYN,
            control_matrix=_RX_CONTROL,
        )
        with pytest.raises(IncompatibleLinearizationError):
            rx.to_flat_model()

    def test_rx_backend_still_shape_checks_observations(self):
        # The shared validator stays wired: a wrong-length reading is rejected, R(x) or
        # not (the state-dependent path must not bypass _validate_step_inputs).
        rx = _build_rx(
            _CHEMOTAXIS_DIMS,
            _CHEMOTAXIS_EDGES,
            _RX_OBS,
            _CHEMOTAXIS_DYN,
            control_matrix=_RX_CONTROL,
        )
        prior = Belief(mean=np.zeros(5), cov=np.eye(5))
        with pytest.raises(ValueError, match="observation"):
            rx.infer_states(np.zeros(2), prior, np.array([0.0]))  # need 3 readings


# --- one sensor family: a graph node takes the flat model's sensor (ADR-073) -------


class _UnstoredSensor:
    """A fixed `ObservationModel` that keeps no `observation_matrix` attribute.

    The protocol promises `is_fixed`, `linearize` and `gaussianize` and nothing else, so
    a graph that read `C` off an attribute would refuse a sensor the flat model takes.
    """

    is_fixed = True

    def __init__(self, gain, noise):
        self._gain = jnp.asarray([[gain]], dtype=float)
        self._noise = jnp.asarray([[noise]], dtype=float)

    def linearize(self, x):
        return self._gain, self._noise

    def gaussianize(self, x, sigma):
        c, r = self.linearize(x)
        return c @ x, c @ sigma @ c.T + r, r


def _one_node_graph_backend(sensor):
    """A single scalar node with a control, observed by ``sensor``."""
    graph = CouplingGraph(dims=(1,), couplings=(), observation_models={0: sensor})
    return CouplingGraphBackend(
        graph,
        (GaussianTransition([[0.9]], dynamics_noise=[[0.2]]),),
        prior=Belief(mean=[0.3], cov=[[1.5]]),
        control_matrix=[[1.0]],
    )


def _one_node_flat_backend(sensor):
    """The same node as a flat model, observed by the same sensor object."""
    gain, _noise = sensor.linearize(jnp.zeros(1))
    return KalmanBackend(
        LinearGaussianModel(
            dynamics_matrix=[[0.9]],
            observation_matrix=gain,
            dynamics_noise=[[0.2]],
            prior=Belief(mean=[0.3], cov=[[1.5]]),
            control_matrix=[[1.0]],
            observation_model=sensor,
        )
    )


class TestOneSensorFamily:
    @pytest.mark.parametrize(
        "sensor",
        [
            FixedSensor([[2.0]], observation_noise=[[0.4]]),
            CallableSensor([[2.0]], _rx_noise, {"R0": np.array([[0.4]]), "gain": 0.3}),
            _UnstoredSensor(2.0, 0.4),
        ],
        ids=["fixed", "callable", "protocol-only"],
    )
    def test_one_sensor_object_serves_a_flat_model_and_a_graph_node(self, sensor):
        # One node and no couplings, so the graph's R(μ⁺) and the flat R(μ⁻) agree.
        graph, flat = _one_node_graph_backend(sensor), _one_node_flat_backend(sensor)
        rng = np.random.default_rng(71)
        graph_belief = flat_belief = Belief(mean=[0.3], cov=[[1.5]])
        for _ in range(5):
            y, a = rng.standard_normal(1), rng.standard_normal(1)
            graph_belief = graph.infer_states(y, graph_belief, a)
            flat_belief = flat.infer_states(y, flat_belief, a)
            np.testing.assert_allclose(graph_belief.mean, flat_belief.mean, atol=1e-10)
            np.testing.assert_allclose(graph_belief.cov, flat_belief.cov, atol=1e-10)
