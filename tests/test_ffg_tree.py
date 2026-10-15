"""Branching factor-graph inference: messages combining at a shared node (Phase 4).

The chain (``test_ffg_chain.py``) is the degenerate FFG — every node has degree 2.
The shared ``CheA`` network ADR-012 targets has a node of degree > 2: one latent
feeding two branches. This file pins the first genuinely-branching inference — the
marginal at a shared parent computed by *combining* the upward message from each
branch (``GaussianCoupling._message_to_parent``) with the prior. The combination is
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

from cpomdp import CallableSensor, FixedSensor, KalmanBackend, LinearGaussianModel
from cpomdp.ffg._message import CanonicalGaussian, likelihood_message
from cpomdp.ffg.factors.linear_gaussian import GaussianCoupling
from cpomdp.ffg.graph import Coupling, CouplingGraph
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
        msg1 = GaussianCoupling(W1, coupling_noise=Q1)._message_to_parent(
            likelihood_message(jnp.eye(c1), jnp.asarray(R1), y1)
        )
        msg2 = GaussianCoupling(W2, coupling_noise=Q2)._message_to_parent(
            likelihood_message(jnp.eye(c2), jnp.asarray(R2), y2)
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
        msg1 = GaussianCoupling(W1, coupling_noise=Q1)._message_to_parent(
            likelihood_message(jnp.eye(1), jnp.asarray(R1), y1)
        )
        msg2 = GaussianCoupling(W2, coupling_noise=Q2)._message_to_parent(
            likelihood_message(jnp.eye(1), jnp.asarray(R2), y2)
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
        Coupling(parent, child, GaussianCoupling(w, coupling_noise=q), 1.0)
        for parent, child, w, q in edges
    )
    observations = {
        n: FixedSensor(c, observation_noise=r) for n, (c, r) in obs_specs.items()
    }
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
            0, (2, 1), (Coupling(0, 1, GaussianCoupling(w, coupling_noise=q), 1.0),), {}
        )
        m0, P0 = rng.standard_normal(2), _spd(rng, 2)
        out = graph.infer(Belief(mean=m0, cov=P0), {})
        np.testing.assert_allclose(np.asarray(out.mean), m0, atol=1e-8)
        np.testing.assert_allclose(np.asarray(out.cov), P0, atol=1e-8)

    def test_reading_without_observation_factor_raises(self):
        # A reading for a node that has no sensor is a usage error.
        graph = CouplingGraph(
            0,
            (1, 1),
            (Coupling(0, 1, GaussianCoupling([[1.0]], coupling_noise=[[0.5]]), 1.0),),
            {},
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
                    GaussianCoupling(
                        jnp.array([[1.5, -0.5]]), coupling_noise=jnp.array([[0.3]])
                    ),
                    1.0,
                ),
            ),
            {1: FixedSensor(jnp.eye(1), observation_noise=jnp.array([[0.2]]))},
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


# --- CouplingGraph._infer_all: collect + distribute vs the dense joint solve -------


def _joint_all_marginals_oracle(dims, edges, seeds):
    """Every node's marginal via one dense joint solve — independent of the BP maths.

    Assemble the joint precision (block-diagonal seed precisions + each coupling's
    canonical block) and joint potential (stacked seed potentials), invert once, and
    slice each node's block. Pure NumPy — shares nothing with the two-pass messages.
    """
    offs = np.cumsum([0, *dims])
    dim = int(offs[-1])

    def blk(i):
        return slice(int(offs[i]), int(offs[i + 1]))

    lam = np.zeros((dim, dim))
    h = np.zeros(dim)
    for node, (precision, potential) in seeds.items():
        lam[blk(node), blk(node)] += np.asarray(precision, float)
        h[blk(node)] += np.asarray(potential, float)
    for parent, child, w, q in edges:
        w, q = np.asarray(w, float), np.asarray(q, float)
        q_inv = np.linalg.inv(q)
        weighted = q_inv @ w  # Q⁻¹W
        lam[blk(parent), blk(parent)] += w.T @ weighted
        lam[blk(child), blk(child)] += q_inv
        lam[blk(parent), blk(child)] += -weighted.T
        lam[blk(child), blk(parent)] += -weighted
    sig = np.linalg.inv(lam)
    mu = sig @ h
    return {n: (mu[blk(n)], sig[blk(n), blk(n)]) for n in range(len(dims))}


def _check_infer_all(dims, edges, seeds, atol=1e-8):
    """Build the graph, run _infer_all, check *every* node's marginal vs the oracle."""
    couplings = tuple(
        Coupling(parent, child, GaussianCoupling(w, coupling_noise=q), 1.0)
        for parent, child, w, q in edges
    )
    graph = CouplingGraph(root=0, dims=dims, couplings=couplings, observations={})
    seed_msgs = {
        n: CanonicalGaussian(precision, h) for n, (precision, h) in seeds.items()
    }
    marginals = graph._infer_all(seed_msgs)
    oracle = _joint_all_marginals_oracle(dims, edges, seeds)
    for node in range(len(dims)):
        out_mean, out_cov = marginals[node].to_moment()
        np.testing.assert_allclose(np.asarray(out_mean), oracle[node][0], atol=atol)
        np.testing.assert_allclose(np.asarray(out_cov), oracle[node][1], atol=atol)


class TestCouplingGraphInferAll:
    """The distribute pass: _infer_all returns *every* node's marginal (collect alone
    gives only the root), matching an independent dense joint solve.
    """

    def test_depth1_two_branches_all_marginals(self):
        # Root 0 -> children 1, 2, every node seeded. The children come only from the
        # downward pass — collect leaves them holding just their own seed.
        rng = np.random.default_rng(500)
        dims = (1, 1, 1)
        edges = [
            (0, 1, rng.standard_normal((1, 1)), _spd(rng, 1)),
            (0, 2, rng.standard_normal((1, 1)), _spd(rng, 1)),
        ]
        seeds = {n: (_spd(rng, 1), rng.standard_normal(1)) for n in range(3)}
        _check_infer_all(dims, edges, seeds)

    def test_depth2_mixed_dims_nonsquare(self):
        # 0 -> 1 -> 2 and 0 -> 3: an internal node with a child, non-square couplings,
        # mixed node dims — the downward message must thread through node 1 to node 2.
        rng = np.random.default_rng(501)
        dims = (2, 1, 2, 1)
        edges = [
            (0, 1, rng.standard_normal((1, 2)), _spd(rng, 1)),
            (1, 2, rng.standard_normal((2, 1)), _spd(rng, 2)),
            (0, 3, rng.standard_normal((1, 2)), _spd(rng, 1)),
        ]
        seeds = {
            n: (_spd(rng, dims[n]), rng.standard_normal(dims[n])) for n in range(4)
        }
        _check_infer_all(dims, edges, seeds)


class TestCouplingGraphValidation:
    def _w(self, c, p):
        return GaussianCoupling(np.zeros((c, p)) + 0.5, coupling_noise=np.eye(c) * 0.3)

    def test_valid_tree_builds(self):
        g = CouplingGraph(
            0,
            (2, 1, 1),
            (Coupling(0, 1, self._w(1, 2), 1.0), Coupling(0, 2, self._w(1, 2), 1.0)),
            {1: FixedSensor(np.eye(1), observation_noise=np.eye(1))},
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
                {9: FixedSensor(np.eye(1), observation_noise=np.eye(1))},
            )

    def test_rejects_sensor_narrower_than_its_node(self):
        # A one-column C on a 2-D node would broadcast into both columns of the
        # node's block and read the sum of its components.
        with pytest.raises(ValueError, match="sensor C width 1"):
            CouplingGraph(
                0,
                (2, 1),
                (Coupling(0, 1, self._w(1, 2), 1.0),),
                {0: FixedSensor(np.eye(1), observation_noise=np.eye(1))},
            )

    @pytest.mark.parametrize("field", ["root", "dims", "couplings", "observations"])
    def test_wiring_cannot_be_reassigned_after_validation(self, field):
        # Construction is the only place the wiring is checked, so a field written
        # afterwards would reach the backend unvalidated.
        graph = CouplingGraph(
            0,
            (1, 1),
            (Coupling(0, 1, self._w(1, 1), 1.0),),
            {1: FixedSensor(np.eye(1), observation_noise=np.eye(1))},
        )
        with pytest.raises(AttributeError):
            setattr(graph, field, getattr(graph, field))

    def test_observations_cannot_gain_a_sensor_after_validation(self):
        # A sensor added to node 7 of a two-node graph would skip the range check.
        sensor = FixedSensor(np.eye(1), observation_noise=np.eye(1))
        graph = CouplingGraph(
            0, (1, 1), (Coupling(0, 1, self._w(1, 1), 1.0),), {1: sensor}
        )
        with pytest.raises(TypeError):
            # Writing to it is the test, so ty's refusal is expected.
            graph.observations[7] = sensor  # ty: ignore[invalid-assignment]

    def test_rejects_an_observation_that_is_not_a_sensor(self):
        # A bare matrix used to construct and fail at the first filter step.
        with pytest.raises(TypeError, match=r"node 1.*FixedSensor.*CallableSensor"):
            CouplingGraph(
                0,
                (1, 1),
                (Coupling(0, 1, self._w(1, 1), 1.0),),
                {1: np.eye(1)},  # ty: ignore[invalid-argument-type]
            )

    def test_static_inference_refuses_a_reading_on_a_state_dependent_node(self):
        # R(x) needs the predicted mean, which a single static collect does not have.
        sensor = CallableSensor(
            np.eye(1), lambda x, p: p * (1.0 + x @ x)[None, None], 0.5
        )
        graph = CouplingGraph(
            0, (1, 1), (Coupling(0, 1, self._w(1, 1), 1.0),), {1: sensor}
        )
        with pytest.raises(ValueError, match=r"node 1.*CouplingGraphBackend"):
            graph.infer(Belief(mean=np.zeros(1), cov=np.eye(1)), {1: np.zeros(1)})


# --- CouplingGraph.infer vs the normal backend on the hand-flattened joint ---


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


def _flattened_kalman_root(root, dims, edges, obs_specs, m0, p0, readings):
    """Root marginal via a flattened `LinearGaussianModel` through `KalmanBackend`.

    The "normal backend" route the demo contrasts against the graph: assemble the full
    joint prior over every node (a root-outward forward pass), stack each reading's
    `(C, R) into one tall update places at that node's block, run a single Kalman
    step (identity dynamics, zero process noise, so predict is a no-op), and marginalise
    the posterior back down to the root block. Independent of the message-passing under
    test *and* of the moment-form oracle's manual conditioning.
    """
    offs = np.cumsum([0, *dims])
    dim = int(offs[-1])

    def blk(i):
        return slice(int(offs[i]), int(offs[i + 1]))

    mu = np.zeros(dim)
    sig = np.zeros((dim, dim))
    mu[blk(root)] = m0
    sig[blk(root), blk(root)] = p0

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

    c_rows, r_blocks, y_parts = [], [], []
    for node, (c_mat, r_mat) in obs_specs.items():
        if node not in readings:
            continue
        c_mat = np.asarray(c_mat, float)
        full = np.zeros((c_mat.shape[0], dim))
        full[:, blk(node)] = c_mat
        c_rows.append(full)
        r_blocks.append(np.asarray(r_mat, float))
        y_parts.append(np.asarray(readings[node], float))

    model = LinearGaussianModel(
        dynamics_matrix=np.eye(dim),
        observation_matrix=np.vstack(c_rows),
        dynamics_noise=np.zeros((dim, dim)),
        observation_noise=_block_diag(r_blocks),
        prior=Belief(mean=mu, cov=sig),
    )
    posterior = KalmanBackend(model).infer_states(
        np.concatenate(y_parts), Belief(mu, sig)
    )
    pm, pc = np.asarray(posterior.mean), np.asarray(posterior.cov)
    return pm[blk(root)], pc[blk(root), blk(root)]


class TestFlattenedKalmanParity:
    """`CouplingGraph.infer` vs the normal backend on the hand-flattened joint.

    The difference Phase 5 demonstrates is representational, not numerical: a branching
    model the chain backend cannot draw still yields the *same* posterior a flattened
    `LinearGaussianModel` does through `KalmanBackend`. These pin that identity
    directly (graph vs the normal backend) on the demo's chemotaxis tree and a spread
    of harder topologies.
    """

    def _check(self, root, dims, edges, obs_specs, m0, p0, readings, atol=1e-7):
        couplings = tuple(
            Coupling(p, c, GaussianCoupling(w, coupling_noise=q), 1.0)
            for p, c, w, q in edges
        )
        observations = {
            n: FixedSensor(c, observation_noise=r) for n, (c, r) in obs_specs.items()
        }
        graph = CouplingGraph(
            root=root, dims=dims, couplings=couplings, observations=observations
        )
        m0, p0 = np.asarray(m0, float), np.asarray(p0, float)
        out = graph.infer(Belief(mean=m0, cov=p0), readings)
        km, kc = _flattened_kalman_root(root, dims, edges, obs_specs, m0, p0, readings)
        np.testing.assert_allclose(np.asarray(out.mean), km, atol=atol)
        np.testing.assert_allclose(np.asarray(out.cov), kc, atol=atol)

    def test_chemotaxis_tree(self):
        # The depth-2 demo tree: a shared CheY-P (degree-3) feeding two motors plus a
        # slow CheB branch; the hidden root CheA is recovered through structure alone.
        chea, chey, cheb, motor_a, motor_b = 0, 1, 2, 3, 4
        edges = [
            (chea, chey, [[0.8]], [[0.05]]),
            (chea, cheb, [[0.6]], [[0.05]]),
            (chey, motor_a, [[1.0]], [[0.03]]),
            (chey, motor_b, [[1.0]], [[0.03]]),
        ]
        obs = {n: (np.eye(1), [[0.10]]) for n in (cheb, motor_a, motor_b)}
        readings = {motor_a: [1.25], motor_b: [1.15], cheb: [0.95]}
        self._check(chea, (1, 1, 1, 1, 1), edges, obs, [0.0], [[4.0]], readings)

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_varied_topologies(self, seed):
        # Non-square couplings, multi-dim nodes, an unobserved internal node (1), and a
        # 2-D root: the normal-backend route must agree wherever the graph does.
        rng = np.random.default_rng(seed)
        edges = [
            (0, 1, rng.standard_normal((1, 2)), _spd(rng, 1)),
            (1, 2, rng.standard_normal((2, 1)), _spd(rng, 2)),
            (0, 3, rng.standard_normal((2, 2)), _spd(rng, 2)),
        ]
        obs = {
            2: (rng.standard_normal((1, 2)), _spd(rng, 1)),
            3: (np.eye(2), _spd(rng, 2)),
        }
        readings = {2: rng.standard_normal(1), 3: rng.standard_normal(2)}
        self._check(
            0, (2, 1, 2, 2), edges, obs, rng.standard_normal(2), _spd(rng, 2), readings
        )
