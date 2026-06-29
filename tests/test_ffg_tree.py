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

import numpy as np
import pytest

from cpomdp.ffg.factors.linear_gaussian import GaussianCoupling, GaussianObservation
from cpomdp.ffg.message import CanonicalGaussian


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
