"""The branching FFG as a recursive filter: ``CouplingGraphBackend`` (issue #25).

``test_ffg_tree.py`` pins the *static* within-slice collect (``CouplingGraph.infer`` —
a root prior + readings -> the root marginal). This file pins the *temporal* recursion
that turns that structure into an ``InferenceBackend``: each node carries its own
dynamics (driven relaxation — ADR-017), the whole tree steps forward under an action,
and the exact ``[[all]]`` filter is the joint-precision solve (predict through the
block-diagonal per-node dynamics, add the structural coupling + observation precision
blocks, solve).

Oracle: a plain-NumPy driven-relaxation filter — per step ``predict`` (block-diagonal
``F = blkdiag(A_i)``, ``Q = blkdiag(Q_i)``) then a joint-precision ``update`` that folds
in the structural couplings and the observations, sharing no code with the
canonical-form math under test. The keystone bar is atol 1e-7 on *every* node's marginal
(not just the root), over multi-step sequences with and without control.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp.backends.base import InferenceBackend
from cpomdp.backends.coupling import CouplingGraphBackend
from cpomdp.backends.kalman import KalmanBackend
from cpomdp.backends.rxinfer import RxInferBackend
from cpomdp.ffg.factors.linear_gaussian import (
    GaussianCoupling,
    GaussianObservation,
    GaussianTransition,
)
from cpomdp.ffg.graph import Coupling, CouplingGraph
from cpomdp.types import Belief


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
    dims, edges, obs_specs, dyn, prior_mean, prior_cov, control, obs_seq, action_seq
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
    ``[[W^T Q^-1 W, -W^T Q^-1], [-Q^-1 W, Q^-1]]`` a ``GaussianCoupling`` carries.

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
        if control is not None:
            shift = np.asarray(control, float) @ np.asarray(action_seq[step], float)
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


def _build(
    dims, edges, obs_specs, dyn, *, control=None, readout_node=None, partition=None
):
    """Build a ``CouplingGraphBackend`` (graph + per-node transitions) for a spec."""
    couplings = tuple(
        Coupling(parent, child, GaussianCoupling(w, q), 1.0)
        for parent, child, w, q in edges
    )
    observations = {n: GaussianObservation(c, r) for n, (c, r) in obs_specs.items()}
    graph = CouplingGraph(
        root=0, dims=dims, couplings=couplings, observations=observations
    )
    transitions = tuple(GaussianTransition(a, q) for a, q in dyn)
    kwargs = {"control": control, "readout_node": readout_node}
    if partition is not None:
        kwargs["partition"] = partition
    return CouplingGraphBackend(graph, transitions, **kwargs)


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
    control=None,
    action_seq=None,
    atol=1e-7,
):
    """Drive the backend over ``obs_seq`` and check each node's marginal vs oracle."""
    states, offs = _driven_relaxation_oracle(
        dims, edges, obs_specs, dyn, prior.mean, prior.cov, control, obs_seq, action_seq
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
            control=control,
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
            control=control,
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


class TestProtocolAndReadout:
    def test_is_inference_backend(self):
        backend = _build(
            _CHEMOTAXIS_DIMS, _CHEMOTAXIS_EDGES, _CHEMOTAXIS_OBS, _CHEMOTAXIS_DYN
        )
        assert isinstance(backend, InferenceBackend)

    def test_readout_defaults_to_root(self):
        backend = _build(
            _CHEMOTAXIS_DIMS, _CHEMOTAXIS_EDGES, _CHEMOTAXIS_OBS, _CHEMOTAXIS_DYN
        )
        prior = Belief(mean=np.zeros(5), cov=np.eye(5) * 2.0)
        posterior = backend.infer_states(np.zeros(3), prior)
        root_belief = backend.readout(posterior)
        node0 = backend.marginal(0, posterior)
        np.testing.assert_allclose(root_belief.mean, node0.mean)
        np.testing.assert_allclose(root_belief.cov, node0.cov)

    def test_prior_not_mutated(self):
        backend = _build(
            _CHEMOTAXIS_DIMS, _CHEMOTAXIS_EDGES, _CHEMOTAXIS_OBS, _CHEMOTAXIS_DYN
        )
        prior = Belief(mean=np.zeros(5), cov=np.eye(5) * 2.0)
        before = np.asarray(prior.cov).copy()
        backend.infer_states(np.zeros(3), prior)
        np.testing.assert_array_equal(np.asarray(prior.cov), before)


class TestTransforms:
    def test_jit_grad_vmap_through_infer_states(self):
        backend = _build(
            _CHEMOTAXIS_DIMS, _CHEMOTAXIS_EDGES, _CHEMOTAXIS_OBS, _CHEMOTAXIS_DYN
        )
        prior = Belief(mean=jnp.zeros(5), cov=jnp.eye(5) * 2.0)

        def root_mean(y):
            return backend.readout(backend.infer_states(y, prior)).mean

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
            control=control,
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
