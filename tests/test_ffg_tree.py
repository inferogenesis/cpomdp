"""Branching factor-graph inference: messages combining at a shared node (Phase 4).

The chain (``test_ffg_chain.py``) is the degenerate FFG — every node has degree 2.
The shared ``CheA`` network ADR-012 targets has a node of degree > 2: one latent
feeding two branches. This file pins the first genuinely-branching inference — the
marginal at a shared parent computed by *combining* the upward message from each
branch (``GaussianCoupling.message_to_parent``) with the prior. The combination is
the factor product ``CanonicalGaussian.__add__``; summing three messages at one node
is the operation a chain never needs.

Oracle: the full joint over ``[parent, child_1, child_2]`` built and conditioned on
both branch observations in plain NumPy — never the canonical-form math under test.
The two children are correlated *through* the shared parent (off-diagonal
``W₁ P₀ W₂ᵀ`` blocks), which is exactly the structure message passing has to respect.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp.ffg.factors.linear_gaussian import GaussianCoupling, GaussianObservation
from cpomdp.ffg.graph import Coupling, CouplingGraph
from cpomdp.ffg.message import CanonicalGaussian
from cpomdp.types import Belief


def _spd(rng, n):
    """A random n x n symmetric positive-definite matrix (NumPy, independent)."""
    a = rng.standard_normal((n, n))
    return a @ a.T + n * np.eye(n)


def _belief_as_canonical(mean, cov):
    """Moment-form (mean, cov) -> its canonical message (NumPy, independent)."""
    precision = np.linalg.inv(cov)
    return CanonicalGaussian(precision, precision @ mean)


class TestSharedNodeTwoBranches:
    @pytest.mark.parametrize(("p", "c1", "c2"), [(1, 1, 1), (2, 1, 1), (2, 2, 1)])
    def test_marginal_matches_full_joint_oracle(self, p, c1, c2):
        # Minimal branching graph: a shared parent X feeds two children, each
        # observed directly. X's marginal = prior + the upward message from each
        # branch — a degree-3 node, the thing a chain cannot be.
        rng = np.random.default_rng(300 + 100 * p + 10 * c1 + c2)
        W1 = rng.standard_normal((c1, p))
        W2 = rng.standard_normal((c2, p))
        Q1, Q2 = _spd(rng, c1), _spd(rng, c2)
        R1, R2 = _spd(rng, c1), _spd(rng, c2)
        m0, P0 = rng.standard_normal(p), _spd(rng, p)
        y1, y2 = rng.standard_normal(c1), rng.standard_normal(c2)

        # --- FFG: combine the prior with each branch's upward message ---
        prior = _belief_as_canonical(m0, P0)
        msg1 = GaussianCoupling(W1, Q1).message_to_parent(
            GaussianObservation(np.eye(c1), R1).message(y1)
        )
        msg2 = GaussianCoupling(W2, Q2).message_to_parent(
            GaussianObservation(np.eye(c2), R2).message(y2)
        )
        out_mean, out_cov = (prior + msg1 + msg2).to_moment()

        # --- Oracle: full joint over [X, Z1, Z2], condition on both readings ---
        mean_j = np.concatenate([m0, W1 @ m0, W2 @ m0])
        cov_j = np.block(
            [
                [P0, P0 @ W1.T, P0 @ W2.T],
                [W1 @ P0, W1 @ P0 @ W1.T + Q1, W1 @ P0 @ W2.T],
                [W2 @ P0, W2 @ P0 @ W1.T, W2 @ P0 @ W2.T + Q2],
            ]
        )
        H = np.block(
            [
                [np.zeros((c1, p)), np.eye(c1), np.zeros((c1, c2))],
                [np.zeros((c2, p)), np.zeros((c2, c1)), np.eye(c2)],
            ]
        )
        R = np.block([[R1, np.zeros((c1, c2))], [np.zeros((c2, c1)), R2]])
        y = np.concatenate([y1, y2])
        gain = cov_j @ H.T @ np.linalg.inv(H @ cov_j @ H.T + R)
        mean_post = mean_j + gain @ (y - H @ mean_j)
        cov_post = (np.eye(p + c1 + c2) - gain @ H) @ cov_j

        np.testing.assert_allclose(out_mean, mean_post[:p], atol=1e-8)
        np.testing.assert_allclose(out_cov, cov_post[:p, :p], atol=1e-8)

    def test_branch_message_order_does_not_matter(self):
        # The shared node just sums messages, and addition is commutative — so the
        # belief is independent of the order branches (and the prior) are combined.
        rng = np.random.default_rng(7)
        p = 2
        W1, W2 = rng.standard_normal((1, p)), rng.standard_normal((1, p))
        Q1, Q2 = _spd(rng, 1), _spd(rng, 1)
        R1, R2 = _spd(rng, 1), _spd(rng, 1)
        m0, P0 = rng.standard_normal(p), _spd(rng, p)
        y1, y2 = rng.standard_normal(1), rng.standard_normal(1)

        prior = _belief_as_canonical(m0, P0)
        msg1 = GaussianCoupling(W1, Q1).message_to_parent(
            GaussianObservation(np.eye(1), R1).message(y1)
        )
        msg2 = GaussianCoupling(W2, Q2).message_to_parent(
            GaussianObservation(np.eye(1), R2).message(y2)
        )
        a_mean, a_cov = (prior + msg1 + msg2).to_moment()
        b_mean, b_cov = (msg2 + (msg1 + prior)).to_moment()
        np.testing.assert_allclose(a_mean, b_mean, atol=1e-12)
        np.testing.assert_allclose(a_cov, b_cov, atol=1e-12)


# --- CouplingGraph.infer: collect-to-root vs an independent moment-form oracle ----


def _tree_marginal_oracle(root, dims, edges, obs_specs, m0, P0, readings):
    """Root marginal via the full moment-form joint — independent of message passing.

    Builds the joint mean/covariance over every node by a root-outward forward pass (a
    child's block has mean ``W·parent``, covariance ``W Σ_pp Wᵀ + Q``, and cross-
    covariance ``W Σ_pk`` to every node already placed), conditions on each reading in
    turn, and returns the root block. Pure NumPy — no canonical-form / message maths.
    """
    offs = np.cumsum([0, *dims])
    dim = int(offs[-1])

    def blk(i):
        return slice(int(offs[i]), int(offs[i + 1]))

    mu = np.zeros(dim)
    sig = np.zeros((dim, dim))
    mu[blk(root)] = m0
    sig[blk(root), blk(root)] = P0

    children: dict[int, list] = {}
    for parent, child, w, q in edges:
        children.setdefault(parent, []).append(
            (child, np.asarray(w, float), np.asarray(q, float))
        )

    placed, frontier = [root], [root]
    while frontier:
        parent = frontier.pop(0)
        for child, w, q in children.get(parent, []):
            mu[blk(child)] = w @ mu[blk(parent)]
            sig[blk(child), blk(child)] = w @ sig[blk(parent), blk(parent)] @ w.T + q
            for placed_node in placed:
                cross = w @ sig[blk(parent), blk(placed_node)]
                sig[blk(child), blk(placed_node)] = cross
                sig[blk(placed_node), blk(child)] = cross.T
            placed.append(child)
            frontier.append(child)

    for node, (c_mat, r_mat) in obs_specs.items():
        if node not in readings:
            continue
        c_mat, r_mat = np.asarray(c_mat, float), np.asarray(r_mat, float)
        h = np.zeros((c_mat.shape[0], dim))
        h[:, blk(node)] = c_mat
        gain = sig @ h.T @ np.linalg.inv(h @ sig @ h.T + r_mat)
        mu = mu + gain @ (np.asarray(readings[node], float) - h @ mu)
        sig = (np.eye(dim) - gain @ h) @ sig

    return mu[blk(root)], sig[blk(root), blk(root)]


def _check_infer(root, dims, edges, obs_specs, m0, P0, readings, atol=1e-8):
    """Build the graph, run ``infer``, and check its marginal against the oracle."""
    couplings = tuple(
        Coupling(parent, child, GaussianCoupling(w, q), 1.0)
        for parent, child, w, q in edges
    )
    observations = {n: GaussianObservation(c, r) for n, (c, r) in obs_specs.items()}
    graph = CouplingGraph(
        root=root, dims=dims, couplings=couplings, observations=observations
    )
    m0, P0 = np.asarray(m0, float), np.asarray(P0, float)
    out = graph.infer(Belief(mean=m0, cov=P0), readings)
    mu_o, cov_o = _tree_marginal_oracle(root, dims, edges, obs_specs, m0, P0, readings)
    np.testing.assert_allclose(np.asarray(out.mean), mu_o, atol=atol)
    np.testing.assert_allclose(np.asarray(out.cov), cov_o, atol=atol)
    return out


class TestCouplingGraphInfer:
    @pytest.mark.parametrize(
        ("p", "c1", "c2"), [(1, 1, 1), (2, 1, 1), (2, 2, 1), (1, 2, 2)]
    )
    def test_depth1_two_branches(self, p, c1, c2):
        # Root with two observed children — square and non-square couplings.
        rng = np.random.default_rng(400 + 10 * p + c1 + c2)
        edges = [
            (0, 1, rng.standard_normal((c1, p)), _spd(rng, c1)),
            (0, 2, rng.standard_normal((c2, p)), _spd(rng, c2)),
        ]
        obs = {1: (np.eye(c1), _spd(rng, c1)), 2: (np.eye(c2), _spd(rng, c2))}
        readings = {1: rng.standard_normal(c1), 2: rng.standard_normal(c2)}
        _check_infer(
            0, (p, c1, c2), edges, obs, rng.standard_normal(p), _spd(rng, p), readings
        )

    def test_depth2_chain_internal_node_unobserved(self):
        # 0 -> 1 -> 2, observe only the leaf: node 1's slot is created mid-collect, the
        # case the combine helper's create-or-add branch exists for.
        rng = np.random.default_rng(7)
        edges = [
            (0, 1, rng.standard_normal((1, 1)), _spd(rng, 1)),
            (1, 2, rng.standard_normal((1, 1)), _spd(rng, 1)),
        ]
        _check_infer(
            0,
            (1, 1, 1),
            edges,
            {2: (np.eye(1), _spd(rng, 1))},
            rng.standard_normal(1),
            _spd(rng, 1),
            {2: rng.standard_normal(1)},
        )

    def test_mixed_tree_with_internal_observation(self):
        # Deep branch 0->1->2 and shallow branch 0->3; an observation sits on the
        # INTERNAL node 1, so evidence enters mid-tree and must merge with its child's
        # message before going up.
        rng = np.random.default_rng(11)
        edges = [
            (0, 1, rng.standard_normal((1, 2)), _spd(rng, 1)),
            (1, 2, rng.standard_normal((2, 1)), _spd(rng, 2)),
            (0, 3, rng.standard_normal((1, 2)), _spd(rng, 1)),
        ]
        obs = {
            1: (np.eye(1), _spd(rng, 1)),
            2: (np.eye(2), _spd(rng, 2)),
            3: (np.eye(1), _spd(rng, 1)),
        }
        readings = {
            1: rng.standard_normal(1),
            2: rng.standard_normal(2),
            3: rng.standard_normal(1),
        }
        _check_infer(
            0, (2, 1, 2, 1), edges, obs, rng.standard_normal(2), _spd(rng, 2), readings
        )

    def test_observation_at_root(self):
        # The root carries its own observation, which must combine with the prior.
        rng = np.random.default_rng(13)
        edges = [(0, 1, rng.standard_normal((1, 2)), _spd(rng, 1))]
        obs = {
            0: (rng.standard_normal((2, 2)), _spd(rng, 2)),
            1: (np.eye(1), _spd(rng, 1)),
        }
        readings = {0: rng.standard_normal(2), 1: rng.standard_normal(1)}
        _check_infer(
            0, (2, 1), edges, obs, rng.standard_normal(2), _spd(rng, 2), readings
        )

    def test_underdetermined_observation(self):
        # A 2-D node seen through a single scalar row — rank-deficient evidence the
        # upward message must still carry correctly.
        rng = np.random.default_rng(17)
        edges = [(0, 1, rng.standard_normal((2, 2)), _spd(rng, 2))]
        _check_infer(
            0,
            (2, 2),
            edges,
            {1: (rng.standard_normal((1, 2)), _spd(rng, 1))},
            rng.standard_normal(2),
            _spd(rng, 2),
            {1: rng.standard_normal(1)},
        )

    def test_unobserved_leaf_contributes_nothing(self):
        # An extra leaf with no reading must not change the root marginal.
        rng = np.random.default_rng(29)
        edges = [
            (0, 1, rng.standard_normal((1, 1)), _spd(rng, 1)),
            (0, 2, rng.standard_normal((1, 1)), _spd(rng, 1)),
        ]
        obs = {1: (np.eye(1), _spd(rng, 1)), 2: (np.eye(1), _spd(rng, 1))}
        _check_infer(
            0,
            (1, 1, 1),
            edges,
            obs,
            rng.standard_normal(1),
            _spd(rng, 1),
            {1: rng.standard_normal(1)},  # node 2 is an unobserved leaf
        )

    def test_single_node_with_reading(self):
        # N=1: just the root, no couplings — a plain measurement update.
        rng = np.random.default_rng(23)
        _check_infer(
            0,
            (1,),
            [],
            {0: (np.eye(1), _spd(rng, 1))},
            rng.standard_normal(1),
            _spd(rng, 1),
            {0: rng.standard_normal(1)},
        )

    def test_empty_readings_returns_prior(self):
        # No evidence -> the root marginal is exactly the prior, lifted and read back.
        rng = np.random.default_rng(19)
        w, q = rng.standard_normal((1, 2)), _spd(rng, 1)
        graph = CouplingGraph(
            0, (2, 1), (Coupling(0, 1, GaussianCoupling(w, q), 1.0),), {}
        )
        m0, P0 = rng.standard_normal(2), _spd(rng, 2)
        out = graph.infer(Belief(mean=m0, cov=P0), {})
        np.testing.assert_allclose(np.asarray(out.mean), m0, atol=1e-8)
        np.testing.assert_allclose(np.asarray(out.cov), P0, atol=1e-8)

    def test_reading_without_observation_factor_raises(self):
        # A reading for a node that has no observation factor is a usage error.
        graph = CouplingGraph(
            0, (1, 1), (Coupling(0, 1, GaussianCoupling([[1.0]], [[0.5]]), 1.0),), {}
        )
        with pytest.raises(KeyError):
            graph.infer(Belief(mean=np.zeros(1), cov=np.eye(1)), {1: np.zeros(1)})

    def test_jit_grad_vmap_through_infer(self):
        # The collect is static structure over traced array data, so it jits, vmaps over
        # a batch of readings, and the root marginal is differentiable w.r.t. a reading.
        graph = CouplingGraph(
            0,
            (2, 1),
            (
                Coupling(
                    0,
                    1,
                    GaussianCoupling(jnp.array([[1.5, -0.5]]), jnp.array([[0.3]])),
                    1.0,
                ),
            ),
            {1: GaussianObservation(jnp.eye(1), jnp.array([[0.2]]))},
        )
        prior = Belief(mean=jnp.zeros(2), cov=jnp.eye(2))

        def root_mean(y):
            return graph.infer(prior, {1: y}).mean

        y = jnp.array([0.7])
        np.testing.assert_allclose(
            np.asarray(jax.jit(root_mean)(y)), np.asarray(root_mean(y)), atol=1e-10
        )
        grad = jax.grad(lambda yy: root_mean(yy).sum())(y)
        assert bool(jnp.all(jnp.isfinite(grad)))

        ys = jnp.array([[0.7], [-1.2], [0.0]])  # a batch of readings
        batched = jax.vmap(root_mean)(ys)
        expected = jnp.stack([root_mean(one) for one in ys])
        np.testing.assert_allclose(
            np.asarray(batched), np.asarray(expected), atol=1e-10
        )


class TestCouplingGraphValidation:
    def _w(self, c, p):
        return GaussianCoupling(np.zeros((c, p)) + 0.5, np.eye(c) * 0.3)

    def test_valid_tree_builds(self):
        g = CouplingGraph(
            0,
            (2, 1, 1),
            (Coupling(0, 1, self._w(1, 2), 1.0), Coupling(0, 2, self._w(1, 2), 1.0)),
            {1: GaussianObservation(np.eye(1), np.eye(1))},
        )
        assert g.root == 0
        assert g.dims == (2, 1, 1)

    def test_rejects_empty_dims(self):
        with pytest.raises(ValueError, match="positive"):
            CouplingGraph(0, (), (), {})

    def test_rejects_root_out_of_range(self):
        with pytest.raises(ValueError, match="root"):
            CouplingGraph(5, (1, 1), (Coupling(0, 1, self._w(1, 1), 1.0),), {})

    def test_rejects_root_as_child(self):
        with pytest.raises(ValueError, match="child"):
            CouplingGraph(
                0,
                (1, 1, 1),
                (
                    Coupling(1, 0, self._w(1, 1), 1.0),
                    Coupling(0, 2, self._w(1, 1), 1.0),
                ),
                {},
            )

    def test_rejects_two_parents(self):
        with pytest.raises(ValueError, match="more than one parent"):
            CouplingGraph(
                0,
                (1, 1, 1),
                (
                    Coupling(0, 1, self._w(1, 1), 1.0),
                    Coupling(2, 1, self._w(1, 1), 1.0),
                ),
                {},
            )

    def test_rejects_cycle(self):
        with pytest.raises(ValueError, match="cycle"):
            CouplingGraph(
                0,
                (1, 1, 1),
                (
                    Coupling(1, 2, self._w(1, 1), 1.0),
                    Coupling(2, 1, self._w(1, 1), 1.0),
                ),
                {},
            )

    def test_rejects_factor_dim_mismatch(self):
        # W is (1,1) but the edge needs (dim[child], dim[parent]) = (1, 2).
        with pytest.raises(ValueError, match="factor W shape"):
            CouplingGraph(0, (2, 1), (Coupling(0, 1, self._w(1, 1), 1.0),), {})

    def test_rejects_observation_on_unknown_node(self):
        with pytest.raises(ValueError, match="observation references unknown node"):
            CouplingGraph(
                0,
                (1, 1),
                (Coupling(0, 1, self._w(1, 1), 1.0),),
                {9: GaussianObservation(np.eye(1), np.eye(1))},
            )
