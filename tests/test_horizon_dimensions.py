"""The horizon surface is agnostic in the state, observation and action dimensions.

Every other test of the H-step rollout runs at ``p = 1`` with a scalar action set, so
nothing currently fails if a change to the rollout, the crossover statistic or the
enumerated search starts assuming a one-dimensional action. The arithmetic is already
shape-driven. This file pins that so the assumption cannot creep back in.

The fixtures make every dimension distinct. On the coupled backend, ``n = 6``,
``m = 3``, ``p = 2``, ``H = 4``. On the flat pair, ``n = 4``, ``m = 3``, ``p = 2``. A
transposed matrix or an assumed-square one cannot survive that, and neither can an axis
mix-up between the horizon and the action width.

The sensor is state-dependent on purpose. Under a fixed sensor the whole covariance
path is policy-independent, so the epistemic contrast would be identically zero and the
crossover checks would pass on a degenerate model. ``R(x)`` is what makes the epistemic
term move with the action, and here it is keyed on the *second* position axis so both
action components do work.

This file pins the machinery. It claims no paper number. Whether *this* fixture's walk
overtakes its reach is beside the point. The registered crossover is measured on the
corridor model in ``tests/test_crossover.py``. The statistic and the finder agree with
each other, and both thread a p-dimensional policy end to end.
"""

import cue_maze
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cpomdp.crossover
from _support import unit_prior
from cpomdp.backends.coupling import CouplingGraphBackend
from cpomdp.crossover import CrossoverStatistic, crossover_horizon, crossover_statistic
from cpomdp.diagnostics import rollout_conditioning
from cpomdp.efe import (
    _ffg_efe_step,
    policy_efe,
    policy_efe_ffg,
    policy_efe_ffg_trace,
)
from cpomdp.enumeration import EnumeratedEfeSearch
from cpomdp.ffg.factors.linear_gaussian import GaussianTransition
from cpomdp.ffg.graph import CouplingGraph
from cpomdp.observation import CallableSensor
from cpomdp.types import Belief, LinearGaussianModel

CONTEXT, ARENA = cue_maze.CONTEXT, cue_maze.ARENA
N_DIMS = 2  # the arena's spatial axes, which fix m = 3 and p = 2
CONTEXT_DIM = 2  # wider than the task needs, so n = 6 differs from every other axis
# The cue sits one step along axis 1. ``cue_maze``'s own default puts it two steps out,
# which the default action set reaches and this file's unit-step ``_walk`` does not: the
# info channel would then never sharpen and the mechanism check below would measure
# nothing. ``cue_maze.best_reachable_noise`` is the check for exactly that.
CUE = np.array([0.0, 1.0])
MAX_H = 4  # the finder's scan bracket. Each horizon is a fresh trace, so keep it short.


# --- fixtures ----------------------------------------------------------------------
# The scene is ``examples/ffg/cue_maze.py`` at ``n_dims = 2``, not a copy of it. Only
# the shapes are this file's own concern, so the model it pins is the one the examples
# actually build.
_cue_noise = cue_maze.cue_noise
_CUE_PARAMS = cue_maze.sensor_params(N_DIMS, CUE)
_ARENA_C = cue_maze.observation_matrix(N_DIMS)


def _coupled_backend():
    """A two-node FFG at n=6, m=3, p=2: context (2-D) drives a 4-D arena.

    The action drives the arena's position block, which the joint state puts behind the
    context. A control matrix built on the wrong offset moves the wrong states. The
    context is widened past what the task needs so ``n`` differs from every other axis.
    """
    return cue_maze.build_maze(N_DIMS, cue=CUE, context_dim=CONTEXT_DIM)


def _coupled_belief():
    """Known position at the origin, and a goal belief mirroring a wrong context."""
    return cue_maze.start_belief(N_DIMS, context_dim=CONTEXT_DIM)


def _pref():
    """Observe zero displacement. The info channel is weighted at about zero."""
    return cue_maze.preference(N_DIMS)


def _policy(steps, horizon):
    """Pad a list of (p,) steps out to ``horizon`` with stays."""
    rows = list(steps[:horizon]) + [[0.0, 0.0]] * max(0, horizon - len(steps))
    return jnp.asarray(np.array(rows, dtype=float))


def _walk(horizon):
    """Detour onto the cue along axis 1, come back, then close on the goal.

    It parks on the same square as ``_reach``, two steps later. Same endpoint,
    different route. That makes the contrast a race between a one-off detour cost and a
    per-step sensing benefit. A walk that stopped short would instead accumulate a
    constant pragmatic penalty for ever, and Δc would grow rather than decay.
    """
    return _policy(
        [[0.0, 1.0], [0.0, -1.0], [-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]], horizon
    )


def _reach(horizon):
    """Straight down axis 0 onto the goal, then park."""
    return _policy([[-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]], horizon)


def _flat_pair():
    """An equivalent (single-node FFG backend, flat model) pair at n=4, m=3, p=2.

    With no couplings and a whole-state info block the FFG rollout must agree with
    the flat
    ``policy_efe``, so this is a cross-implementation check at ``p > 1``. The FFG
    predicts through the precision form, the flat model through ``AΣAᵀ + Q``.
    """
    a = np.eye(4) + 0.05 * np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    q = np.diag([0.1, 0.1, 0.1, 0.1])
    b = np.zeros((4, 2))
    b[0, 0] = 1.0
    b[1, 1] = 1.0
    sensor = CallableSensor(_ARENA_C, _cue_noise, _CUE_PARAMS)
    graph = CouplingGraph(
        root=0, dims=(4,), couplings=(), observation_models={0: sensor}
    )
    backend = CouplingGraphBackend(
        graph,
        (GaussianTransition(a, dynamics_noise=q),),
        prior=unit_prior(graph),
        control_matrix=b,
    )
    model = LinearGaussianModel(
        dynamics_matrix=a,
        observation_matrix=_ARENA_C,
        dynamics_noise=q,
        observation_noise=np.eye(
            3
        ),  # placeholder under a callable sensor, ignored there
        prior=Belief(mean=np.zeros(4), cov=np.eye(4)),
        control_matrix=b,
        observation_model=CallableSensor(_ARENA_C, _cue_noise, _CUE_PARAMS),
    )
    return backend, model


def _action_set():
    """Stay, or step by one along either axis in either direction, at p = 2."""
    return cue_maze.axis_action_set(N_DIMS, magnitudes=(1.0,))


# --- the FFG rollout carries n, m and p on independent axes ------------------------
class TestFfgRolloutAtMultiDimAction:
    def test_trace_shapes_track_each_dimension_separately(self):
        backend, belief, pref = _coupled_backend(), _coupled_belief(), _pref()
        info_block = tuple(backend.block(CONTEXT))
        trace = policy_efe_ffg_trace(
            backend, belief, _walk(4), pref, info_block=info_block
        )
        assert trace.g.shape == (4,)  # H
        assert trace.mu_pred.shape == (4, 6)  # H, n
        assert trace.sigma_pred.shape == (4, 6, 6)
        assert trace.sigma_post.shape == (4, 6, 6)
        assert trace.s.shape == (4, 3, 3)  # H, m. Not n, and not p.

    def test_sums_equal_the_summed_rollout(self):
        backend, belief, pref = _coupled_backend(), _coupled_belief(), _pref()
        info_block = tuple(backend.block(CONTEXT))
        for h in (1, 2, 4):
            g, parts = policy_efe_ffg(
                backend, belief, _walk(h), pref, info_block=info_block
            )
            trace = policy_efe_ffg_trace(
                backend, belief, _walk(h), pref, info_block=info_block
            )
            np.testing.assert_array_equal(g, jnp.sum(trace.g))
            np.testing.assert_array_equal(parts["pragmatic"], jnp.sum(trace.pragmatic))
            np.testing.assert_array_equal(parts["epistemic"], jnp.sum(trace.epistemic))

    def test_h1_reduces_to_one_ffg_step(self):
        backend, belief, pref = _coupled_backend(), _coupled_belief(), _pref()
        info_block = tuple(backend.block(CONTEXT))
        policy = _walk(1)
        g, parts = policy_efe_ffg(backend, belief, policy, pref, info_block=info_block)

        predicted = backend.predicted_belief(belief, policy[0])
        observation_matrix, _ = backend.observation_model
        g_ref, parts_ref = _ffg_efe_step(
            predicted.mean,
            predicted.cov,
            observation_matrix,
            backend.observation_noise_at(predicted.mean),
            pref.goal,
            pref.precision,
            info_block,
        )
        np.testing.assert_array_equal(g, g_ref)
        np.testing.assert_array_equal(parts["pragmatic"], parts_ref["pragmatic"])
        np.testing.assert_array_equal(parts["epistemic"], parts_ref["epistemic"])

    def test_action_reaches_every_control_axis(self):
        # A guard on the fixture itself: if only one action component moved the state,
        # the rest of this file would pass at p = 1 in disguise.
        backend, belief = _coupled_backend(), _coupled_belief()
        base = np.asarray(backend.predicted_belief(belief, jnp.zeros(2)).mean)
        for axis in range(2):
            step = np.zeros(2)
            step[axis] = 1.0
            moved = np.asarray(backend.predicted_belief(belief, jnp.asarray(step)).mean)
            assert not np.allclose(moved, base)

    def test_grad_and_vmap_carry_the_action_width(self):
        backend, belief, pref = _coupled_backend(), _coupled_belief(), _pref()
        info_block = tuple(backend.block(CONTEXT))

        def score(policy):
            return policy_efe_ffg(backend, belief, policy, pref, info_block=info_block)[
                0
            ]

        grad = jax.grad(score)(_walk(4))
        assert grad.shape == (4, 2)  # H, p
        assert np.all(np.isfinite(np.asarray(grad)))

        batched = jax.jit(jax.vmap(score))(jnp.stack([_walk(4), _reach(4)]))
        assert batched.shape == (2,)
        assert np.all(np.isfinite(np.asarray(batched)))

    def test_conditioning_diagnostics_accept_the_wider_rollout(self):
        backend, belief, pref = _coupled_backend(), _coupled_belief(), _pref()
        info_block = tuple(backend.block(CONTEXT))
        trace = policy_efe_ffg_trace(
            backend, belief, _walk(4), pref, info_block=info_block
        )
        rc = rollout_conditioning(trace)
        assert rc.all_positive_definite
        assert rc.cond_sigma_pred.shape == (4,)
        assert rc.cond_s.shape == (4,)


# --- the flat and FFG rollouts still agree once the action is wider ----------------
class TestFfgReducesToFlatAtMultiDimAction:
    def test_no_coupling_whole_state_matches_policy_efe(self):
        backend, model = _flat_pair()
        belief = Belief(mean=[0.4, -0.2, 0.1, 0.3], cov=np.diag([0.7, 0.5, 0.4, 0.6]))
        pref = _pref()
        for h in (2, 3):
            policy = _policy([[0.5, -0.4], [-0.3, 0.2], [0.1, 0.6]], h)
            g, parts = policy_efe_ffg(
                backend, belief, policy, pref, info_block=range(4)
            )
            g_ref, parts_ref = policy_efe(model, belief, policy, pref)
            np.testing.assert_allclose(float(g), float(g_ref), atol=1e-9)
            np.testing.assert_allclose(
                float(parts["pragmatic"]), float(parts_ref["pragmatic"]), atol=1e-9
            )
            np.testing.assert_allclose(
                float(parts["epistemic"]), float(parts_ref["epistemic"]), atol=1e-9
            )


# --- the crossover statistic contrasts p-dimensional policies ----------------------
class TestCrossoverAtMultiDimAction:
    def test_delta_g_is_the_difference_of_the_two_rollouts(self):
        backend, belief, pref = _coupled_backend(), _coupled_belief(), _pref()
        info_block = tuple(backend.block(CONTEXT))
        for h in (1, 3, 5):
            stat = crossover_statistic(
                backend, belief, _walk(h), _reach(h), pref, info_block=info_block
            )
            g_walk = policy_efe_ffg(
                backend, belief, _walk(h), pref, info_block=info_block
            )[0]
            g_reach = policy_efe_ffg(
                backend, belief, _reach(h), pref, info_block=info_block
            )[0]
            assert stat.horizon == h
            np.testing.assert_allclose(
                float(stat.delta_g), float(g_walk) - float(g_reach), atol=1e-9
            )
            np.testing.assert_array_equal(
                stat.delta_g, stat.delta_c - stat.delta_epsilon
            )

    def test_contrast_is_antisymmetric_in_its_two_policies(self):
        # Swapping walk and reach negates every component, so a shared buffer giving the
        # same sign twice is caught here. A policy argument silently reused for both
        # would give zero, which this test does *not* catch on its own (0 == -0); the
        # non-zero assertion below and the difference-of-rollouts test above do.
        backend, belief, pref = _coupled_backend(), _coupled_belief(), _pref()
        info_block = tuple(backend.block(CONTEXT))
        forward = crossover_statistic(
            backend, belief, _walk(4), _reach(4), pref, info_block=info_block
        )
        reversed_ = crossover_statistic(
            backend, belief, _reach(4), _walk(4), pref, info_block=info_block
        )
        np.testing.assert_allclose(
            float(forward.delta_g), -float(reversed_.delta_g), atol=1e-12
        )
        np.testing.assert_allclose(
            float(forward.delta_epsilon), -float(reversed_.delta_epsilon), atol=1e-12
        )
        assert float(forward.delta_g) != 0.0  # else the negation above is vacuous

    def test_horizon_finder_returns_the_first_negative_the_statistic_reports(self):
        # The finder's contract at p = 2. It is checked against the statistic it is
        # defined over rather than against a tuned geometry. Whatever the sign pattern,
        # H* is the first horizon where ΔG < 0, and None when there is none.
        backend, belief, pref = _coupled_backend(), _coupled_belief(), _pref()
        info_block = tuple(backend.block(CONTEXT))
        for walk_of, reach_of in ((_walk, _reach), (_reach, _walk)):
            scan = [
                crossover_statistic(
                    backend,
                    belief,
                    walk_of(h),
                    reach_of(h),
                    pref,
                    info_block=info_block,
                ).walk_wins
                for h in range(1, MAX_H + 1)
            ]
            want = next((h for h, flipped in enumerate(scan, 1) if flipped), None)
            got = crossover_horizon(
                backend,
                belief,
                walk_of,
                reach_of,
                pref,
                info_block=info_block,
                max_horizon=MAX_H,
            )
            assert got == want
        # Antisymmetry guarantees the two directions cannot both be flip-free, so the
        # loop covers a real H* and a real None. This fixture's ΔG holds one sign right
        # across the bracket, so the H* it finds is 1 and no horizon is ever skipped
        # past. The *first* in `min{H : ΔG(H) < 0}` is pinned by the test below.

    def test_the_finder_skips_the_horizons_before_the_first_flip(self, monkeypatch):
        # A finder that only ever evaluated H = 1 passes every other test in this file,
        # because no fixture here changes sign mid-bracket. Script the sign pattern
        # instead of tuning a geometry to produce one: the statistic is stubbed, so what
        # is under test is the scan and nothing else.
        pattern = {1: +1.0, 2: +0.5, 3: -0.2, 4: -1.0}
        seen = []

        def fake_statistic(_backend, _belief, walk, _reach, _pref, *, info_block):
            del info_block
            horizon = int(np.asarray(walk).shape[0])
            seen.append(horizon)
            delta_g = jnp.asarray(pattern[horizon])
            return CrossoverStatistic(horizon, jnp.asarray(0.0), delta_g, delta_g)

        monkeypatch.setattr(cpomdp.crossover, "crossover_statistic", fake_statistic)
        got = crossover_horizon(
            _coupled_backend(),
            _coupled_belief(),
            _walk,
            _reach,
            _pref(),
            info_block=(0,),
            max_horizon=MAX_H,
        )
        assert got == 3  # the *first* negative, not merely some negative
        assert seen == [1, 2, 3]  # scanned upward and stopped, rather than sweeping

    def test_the_two_halves_of_the_mechanism_are_both_live(self):
        # The detour buys information, and the reach's pragmatic advantage decays as
        # the horizon lengthens. Δε > 0 is the sharper here. The cue sits off the goal
        # axis, so the pull is bought by action dim 1 alone. A rollout that dropped the
        # second action component would read Δε ≈ 0 rather than fail a shape check.
        backend, belief, pref = _coupled_backend(), _coupled_belief(), _pref()
        info_block = tuple(backend.block(CONTEXT))
        near = crossover_statistic(
            backend, belief, _walk(4), _reach(4), pref, info_block=info_block
        )
        far = crossover_statistic(
            backend, belief, _walk(8), _reach(8), pref, info_block=info_block
        )
        assert float(near.delta_epsilon) > 0.1
        assert float(far.delta_c) < float(near.delta_c)

    def test_no_crossover_when_the_two_policies_are_identical(self):
        # A control on the finder at p = 2: ΔG(H) = 0 for all H, so there is no flip and
        # None comes back rather than a spurious H*.
        backend, belief, pref = _coupled_backend(), _coupled_belief(), _pref()
        info_block = tuple(backend.block(CONTEXT))
        h_star = crossover_horizon(
            backend, belief, _walk, _walk, pref, info_block=info_block, max_horizon=3
        )
        assert h_star is None

    def test_mismatched_horizons_are_rejected(self):
        backend, belief, pref = _coupled_backend(), _coupled_belief(), _pref()
        info_block = tuple(backend.block(CONTEXT))
        with np.testing.assert_raises(ValueError):
            crossover_statistic(
                backend, belief, _walk(3), _reach(2), pref, info_block=info_block
            )


# --- the enumerated search enumerates p-dimensional sequences ----------------------
class TestEnumeratedSearchAtMultiDimAction:
    def test_certificate_and_argmin_shape(self):
        backend, belief, pref = _coupled_backend(), _coupled_belief(), _pref()
        action_set = _action_set()
        search = EnumeratedEfeSearch.over_backend(
            backend, action_set, info_block=tuple(backend.block(CONTEXT)), horizon=3
        )
        result = search.evaluate(belief, pref)
        assert search.certificate.complete
        assert search.certificate.expected == action_set.size**3  # 125
        assert result.g.shape == (125,)
        assert result.best_policy.shape == (3, 2)  # H, p
        assert np.all(np.isfinite(np.asarray(result.g)))

    def test_every_enumerated_policy_is_a_declared_sequence(self):
        # The enumeration must vary the *whole* action vector. Varying only its first
        # component would be the bug. All |A|^H sequences appear, each row a declared
        # member.
        backend = _coupled_backend()
        action_set = _action_set()
        search = EnumeratedEfeSearch.over_backend(
            backend, action_set, info_block=tuple(backend.block(CONTEXT)), horizon=2
        )
        policies = np.asarray(search.policies)
        assert policies.shape == (25, 2, 2)  # |A|^H, H, p
        members = {tuple(a) for a in np.asarray(action_set.actions)}
        assert {tuple(row) for row in policies.reshape(-1, 2)} == members
        assert len({tuple(p.ravel()) for p in policies}) == 25


def _priced_joint_width(action_set, horizon: int, n_dims: int, context_dim: int) -> int:
    """The joint width `enumeration_cost` charged for, recovered from its estimate.

    `per_policy = 8·(H·p + 4·joint²)`, so the width is the only unknown left once the
    policy count and the action width are known. Reading it back is what lets the test
    compare against the real graph rather than against a repeated formula.
    """
    policies, _, peak_gib = cue_maze.enumeration_cost(
        action_set, horizon, n_dims, context_dim=context_dim
    )
    per_policy = peak_gib * 1024**3 / policies
    joint_squared = ((per_policy / 8) - horizon * action_set.action_dim) / 4
    return round(joint_squared**0.5)


class TestEnumerationCostPricesTheRealJoint:
    """The memory estimate is quadratic in the joint width, and overshooting the live
    ceiling kills the whole session. So the width it charges for has to be the width the
    graph actually builds, at every `(n_dims, context_dim)` the maze offers.

    `n_dims` is read off the action set and cannot drift. `context_dim` cannot be:
    nothing in an action set knows how wide the context node is, so the caller carries
    it across by hand. That is exactly the kind of constant that goes stale.
    """

    @pytest.mark.parametrize(
        ("n_dims", "context_dim"),
        [(1, 1), (2, 1), (2, 2), (3, 1), (2, 4)],
    )
    def test_the_priced_width_is_the_backend_s_own(self, n_dims, context_dim):
        action_set = cue_maze.axis_action_set(n_dims)
        backend = cue_maze.build_maze(n_dims, context_dim=context_dim)
        priced = _priced_joint_width(action_set, 2, n_dims, context_dim)
        assert priced == backend.n_total

    def test_a_wider_context_is_charged_for(self):
        # The regression itself: a hardcoded `1 + 2·n_dims` returns the same estimate at
        # every context width, so a `context_dim=2` sweep is priced as a `context_dim=1`
        # one and under-reads by (6/5)² on the dominant term.
        action_set = cue_maze.axis_action_set(2)
        narrow = cue_maze.enumeration_cost(action_set, 2, 2, context_dim=1)[2]
        wide = cue_maze.enumeration_cost(action_set, 2, 2, context_dim=2)[2]
        assert wide > narrow
