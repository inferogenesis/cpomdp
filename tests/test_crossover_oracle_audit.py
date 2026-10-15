"""Records what the crossover demo's NumPy oracle returns at H=7, before it changes.

`examples/ffg/crossover.py::_numpy_score` is the independent kernel the headline
`H* = 7` is checked against. Its per-step epistemic term calls `np.linalg.slogdet` and
keeps only the log-magnitude, so a covariance block with an even number of negative
eigenvalues returns a finite, plausible number instead of NaN. The shipped kernel does
not: `_efe_step` and `_state_info_gain` route through `_logdet_pd`, which tests positive
definiteness by Cholesky. The two routines were documented as differing, never audited.

Guarding the oracle is a precondition, not a change of arithmetic, so it must leave
every value here untouched. The anchors below make that checkable rather than asserted.
They are measured off the unguarded path and recorded first, so the guard cannot be
introduced and blessed in one step. An anchor carries no theory: it says the number did
not move, nothing about whether it was right to begin with.

`crossover.check()` gates the same numbers but enumerates ~150k policies and is marked
slow, so it is deselected on pull requests. Scoring two known policies costs a backend
build and fourteen propagation steps, which keeps the headline gated on every run.

`TestOracleRejectsNonPd` checks the route and not the guard: that the oracle reaches
`diagnostics._host_logdet_pd` for both halves of its epistemic term, and that the kernel
rejects the same matrix. `tests/test_diagnostics.py::TestLogdetPd` tests the guard.
"""

import crossover
import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp.efe import _logdet_pd as kernel_logdet_pd

# Declared bars; full justification in warrant_numbers.md.
# The guard under audit adds a positive-definiteness precondition ahead of an unchanged
# `slogdet`, so on a positive-definite rollout the expected movement is zero and a bar
# at the ULP would hold. This sits about three orders above it (4.3e-9 absolute at
# G ~ 425), matching the atol = 1e-9 the demo already trusts for shipped-against-oracle
# agreement, so cross-platform BLAS variation cannot flake it while a real change in the
# epistemic term cannot hide under it.
ORACLE_RTOL = 1e-11
# The flip margin is a difference of two values near 425, so it inherits ~8.5e-9 of
# absolute slack from the pair above. This is that slack, rounded up.
MARGIN_ATOL = 1e-8

# Measured on the unguarded oracle at H = 7 (float64, x64 enabled).
ANCHOR_WALK = 425.163110098734
ANCHOR_REACH = 425.3151092512748
ANCHOR_MARGIN = -0.15199915254078178


@pytest.fixture(scope="module")
def oracle_scores():
    """`(walk, reach)` from the NumPy oracle at H*, computed once for the module."""
    horizon = crossover.FLIP_H
    return (
        crossover._numpy_score(crossover._walk(horizon)),
        crossover._numpy_score(crossover._reach(horizon)),
    )


class TestOracleAnchors:
    """The oracle's H=7 scores, recorded so a guard cannot move them unnoticed."""

    def test_walk_score(self, oracle_scores):
        walk, _ = oracle_scores
        np.testing.assert_allclose(walk, ANCHOR_WALK, rtol=ORACLE_RTOL)

    def test_reach_score(self, oracle_scores):
        _, reach = oracle_scores
        np.testing.assert_allclose(reach, ANCHOR_REACH, rtol=ORACLE_RTOL)

    def test_flip_margin(self, oracle_scores):
        # ΔG(7) as the ledger registers it: G(walk) − G(reach), relative size 3.6e−4.
        walk, reach = oracle_scores
        np.testing.assert_allclose(walk - reach, ANCHOR_MARGIN, atol=MARGIN_ATOL)

    def test_walk_wins_at_flip_horizon(self, oracle_scores):
        # The direction the headline rests on, asserted with no tolerance in it:
        # whatever the anchors drift to, the oracle must rank the walk first at H*.
        walk, reach = oracle_scores
        assert walk < reach

    def test_horizon_under_audit_is_the_registered_one(self):
        # The anchors are H-specific. If FLIP_H moves, they are stale, not passing.
        assert crossover.FLIP_H == 7


# diag(-1, -2): determinant +2, two negative eigenvalues. Every routine that reads a
# determinant sign, or discards it, calls this matrix fine. It is the smallest witness
# that separates a Cholesky guard from a sign shortcut.
NOT_PD = np.array([[-1.0, 0.0], [0.0, -2.0]])


class TestOracleRejectsNonPd:
    """The oracle's log-det keeps the sign, and is wired into the score.

    `tests/test_rollout_hygiene.py::TestSlogdetSignGuarded` makes the same demand of the
    shipped kernel and of `diagnostics.epistemic_value`. The demo's oracle is the third
    route and was never held to it. Two routes that both discard the sign agree with
    each other and are both wrong, the one failure a two-route check cannot see.
    """

    def test_oracle_logdet_is_nan_on_non_pd(self):
        # The name as the oracle binds it, so an import swapped back to a bare slogdet
        # fails here. What the guard does on its own is `TestLogdetPd`'s subject.
        assert np.isnan(crossover._host_logdet_pd(NOT_PD))

    def test_kernel_logdet_is_nan_on_non_pd(self):
        assert bool(jnp.isnan(kernel_logdet_pd(jnp.asarray(NOT_PD))))

    def test_oracle_logdet_matches_the_kernel_on_pd(self):
        # Two routines, one value. The host guard calls slogdet behind a Cholesky; the
        # kernel reads its answer off a Cholesky factor under jit. Point the demo at
        # `efe._logdet_pd` and this stops being a check.
        matrix = np.array([[2.0, 0.5], [0.5, 1.5]])
        oracle = crossover._host_logdet_pd(matrix)
        kernel = float(kernel_logdet_pd(jnp.asarray(matrix)))
        np.testing.assert_allclose(oracle, kernel, rtol=1e-12)

    def test_score_routes_both_epistemic_terms_through_the_guard(self, monkeypatch):
        # A guard the score does not call is decoration. The epistemic needs a log-det
        # of the predicted block and one of the posterior block, so a scored H-step
        # policy must reach it twice per step. One call per step means half the term is
        # still bare.
        calls = []
        guarded = crossover._host_logdet_pd

        def counting(matrix):
            calls.append(matrix)
            return guarded(matrix)

        monkeypatch.setattr(crossover, "_host_logdet_pd", counting)
        crossover._numpy_score(crossover._walk(crossover.FLIP_H))
        assert len(calls) == 2 * crossover.FLIP_H


class TestFlipSeparation:
    """The flip margin against the error the declared conditioning ceiling allows.

    `H* = 7` rests on `G(walk) < G(reach)`, a bare inequality between two computed
    floats. Nothing said how far apart they had to be, so the claim was delivered with
    no bar behind it while the conditioning bars sat one function away unused.

    `COND_CEILING` is already declared and already gated in
    `tests/test_rollout_hygiene.py`. A float64 solve at condition number `k` carries
    relative error near `k * eps`, so it states an error on each `G` and, doubled, on
    their difference. Asserting the separation against that bound is the flip measured
    against something stated, not a threshold chosen to fit it.
    """

    def test_the_bound_is_the_declared_ceiling_propagated(self):
        bound = crossover.flip_margin_error(425.0, -425.0)
        expected = 2 * 425.0 * crossover.COND_CEILING * np.finfo(float).eps
        np.testing.assert_allclose(bound, expected, rtol=1e-12)

    def test_the_bound_scales_with_the_larger_magnitude(self):
        # Relative error, so a pair ten times larger carries ten times the slack.
        assert crossover.flip_margin_error(4250.0, 1.0) == pytest.approx(
            10 * crossover.flip_margin_error(425.0, 1.0)
        )

    def test_the_margin_clears_the_bound(self, oracle_scores):
        walk, reach = oracle_scores
        assert abs(walk - reach) > crossover.flip_margin_error(walk, reach)

    def test_the_bound_refuses_a_near_tie(self):
        # The bar bites. Two values apart by less than the propagated error are not
        # separated, whatever the sign of their difference says.
        walk = 425.0
        reach = walk + crossover.flip_margin_error(walk, walk) / 2
        assert not abs(walk - reach) > crossover.flip_margin_error(walk, reach)
