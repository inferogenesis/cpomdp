"""The epistemic/pragmatic crossover statistic, checked against its H=1 anchors.

The crossover asks: as the horizon grows, does the *walk* (detour, then exploit)
overtake the *reach* (head straight for the goal)? The statistic contrasts the two
policies' summed EFE components: the accumulated epistemic pull Δε, the accumulated
pragmatic gradient Δc, and ΔG = Δc − Δε = G(walk) − G(reach), the difference the planner
minimises. ΔG < 0 is the crossover — the walk's information now outweighs its
goal-distance cost. H* = min{H : ΔG(H) < 0}.

At H = 1 the statistic must reduce to the anchors read on the two-node coupled-tree
T-maze (``examples/ffg/epistemic_dissociation_figure.py``, Result 4), with the cue off
the prior path (``CUE_DETOUR_X``):

- Δε(1) = the epistemic pull = 1.72 nats (node-restricted, the CONTEXT marginal).
- Δc(1) = the pragmatic gradient = 4.49 nats.
- ΔG(1) = 4.49 − 1.72 = +2.77 nats > 0 — the reach wins at H = 1, which is what forces
  H* > 1 and makes the crossover an arithmetic question rather than a hope.

This file fixes the statistic, the sign convention, reach/walk as declared members of a
versioned action set, and the anchor magnitudes, and checks them against the H=1
reduction (threshold rationale in warrant_numbers.md). The anchor scan carries no
observation draw, so these are exact facts, not sampled ones.
"""

import epistemic_dissociation_figure as demo
import jax.numpy as jnp
import numpy as np

from cpomdp.crossover import CrossoverStatistic, crossover_horizon, crossover_statistic
from cpomdp.enumeration import FiniteActionSet
from cpomdp.selection import Preference

# Paper 1 Result 4 anchors (nats), measured on the dissociation model at CUE_DETOUR_X.
# Exact facts (the scan carries no draw); pinned so the pre-registration is falsifiable.
PULL_ANCHOR = 1.723213  # Δε(1), node-restricted (CONTEXT marginal)
GRADIENT_ANCHOR = 4.491010  # Δc(1)
CROSSOVER_ANCHOR = 2.767797  # ΔG(1) = gradient − pull > 0 (reach wins at H=1)
WHOLE_STATE_PULL = (
    2.416572  # the same contrast with a whole-state info block (not 1.72)
)
ANCHOR_TOL = 1e-4

# The reach/walk pair as declared members of a coarse, versioned action set (a
# superset of the two anchor actions). "Declared" so the pair is a property of the
# model, not of whichever two the sweep happens to surface.
CROSSOVER_ACTION_SET = FiniteActionSet(
    [[-2.0], [-1.0], [0.0], [1.0], [2.0]], version="crossover-v1"
)


def _setup():
    backend = demo.build_backend(epistemic_alive=True, cue_x=demo.CUE_DETOUR_X)
    belief = demo.start_belief()
    pref = Preference(
        goal=[0.0, 0.0],
        precision=[[demo.GOAL_PRECISION, 0.0], [0.0, demo.INFO_PRECISION]],
    )
    info_block = tuple(backend.block(demo.CONTEXT))
    return backend, belief, pref, info_block


def _anchor_actions():
    """(a_sense, a_myopic) — argmax ε and argmin G over the demo's action grid."""
    scan = demo._boundary_scan(alive=True, cue_x=demo.CUE_DETOUR_X)
    a_sense = float(scan["grid"][int(np.argmax(scan["epistemic"]))])
    a_myopic = float(scan["grid"][int(np.argmin(scan["total"]))])
    return a_sense, a_myopic


def _constant(action, horizon):
    return jnp.full((horizon, 1), action)


# --- the H=1 reduction to Paper 1's anchors ----------------------------------------
class TestH1CollapsesToAnchors:
    def test_reduces_to_pull_and_gradient(self):
        backend, belief, pref, info_block = _setup()
        a_sense, a_myopic = _anchor_actions()
        stat = crossover_statistic(
            backend,
            belief,
            _constant(a_sense, 1),
            _constant(a_myopic, 1),
            pref,
            info_block=info_block,
        )
        assert stat.horizon == 1
        got = (float(stat.delta_epsilon), float(stat.delta_c), float(stat.delta_g))
        want = (PULL_ANCHOR, GRADIENT_ANCHOR, CROSSOVER_ANCHOR)
        np.testing.assert_allclose(got, want, atol=ANCHOR_TOL)

    def test_whole_state_target_reads_the_2_42_number(self):
        # The same contrast aimed at the whole state reads 2.42, not the headline
        # node-restricted 1.72 — the ledger records both so they are never confused.
        backend, belief, pref, _ = _setup()
        a_sense, a_myopic = _anchor_actions()
        whole = tuple(range(backend.n_total))
        stat = crossover_statistic(
            backend,
            belief,
            _constant(a_sense, 1),
            _constant(a_myopic, 1),
            pref,
            info_block=whole,
        )
        np.testing.assert_allclose(
            float(stat.delta_epsilon), WHOLE_STATE_PULL, atol=ANCHOR_TOL
        )


# --- the sign convention the H>1 assertions rest on --------------------------------
class TestSignConvention:
    def test_reach_wins_at_h1(self):
        # ΔG(1) > 0: at the opening step the pragmatic gradient outweighs the pull, so
        # the planner still prefers the reach. This is what makes H* > 1.
        backend, belief, pref, info_block = _setup()
        a_sense, a_myopic = _anchor_actions()
        stat = crossover_statistic(
            backend,
            belief,
            _constant(a_sense, 1),
            _constant(a_myopic, 1),
            pref,
            info_block=info_block,
        )
        assert float(stat.delta_g) > 0
        assert not stat.walk_wins

    def test_pull_and_gradient_are_both_positive(self):
        # Moving to the cue both buys information (Δε > 0) and costs goal-distance
        # (Δc > 0). The crossover is a race between two positive quantities.
        backend, belief, pref, info_block = _setup()
        a_sense, a_myopic = _anchor_actions()
        stat = crossover_statistic(
            backend,
            belief,
            _constant(a_sense, 1),
            _constant(a_myopic, 1),
            pref,
            info_block=info_block,
        )
        assert float(stat.delta_epsilon) > 0
        assert float(stat.delta_c) > 0

    def test_delta_g_is_gradient_minus_pull_exactly(self):
        # The sign convention, structurally: ΔG = Δc − Δε (pragmatic a cost, epistemic a
        # value). Byte-exact, since ΔG is defined that way.
        backend, belief, pref, info_block = _setup()
        a_sense, a_myopic = _anchor_actions()
        stat = crossover_statistic(
            backend,
            belief,
            _constant(a_sense, 2),
            _constant(a_myopic, 2),
            pref,
            info_block=info_block,
        )
        np.testing.assert_array_equal(stat.delta_g, stat.delta_c - stat.delta_epsilon)


# --- the reach/walk pair are declared members, not whatever the sweep surfaced -----
class TestReachWalkDeclaredMembers:
    def test_anchor_actions_are_members_of_the_declared_set(self):
        a_sense, a_myopic = _anchor_actions()
        members = [float(a[0]) for a in np.asarray(CROSSOVER_ACTION_SET.actions)]
        assert a_sense in members
        assert a_myopic in members

    def test_argmax_eps_and_argmin_g_land_on_the_anchors(self):
        # Over the coarse declared set (not just the fine grid), argmax ε still picks
        # the cue-ward sense action and argmin G still picks the prior-ward myopic
        # action. So the pair is stable under this set refinement (a D3 falsifier).
        backend, belief, pref, info_block = _setup()
        a_sense, a_myopic = _anchor_actions()
        from cpomdp.efe import _ffg_efe_step

        observation_matrix, _ = backend.observation_model
        eps, gees = [], []
        for action in np.asarray(CROSSOVER_ACTION_SET.actions):
            predicted = backend.predicted_belief(belief, jnp.asarray(action))
            noise = backend.observation_noise_at(predicted.mean)
            g, parts = _ffg_efe_step(
                predicted.mean,
                predicted.cov,
                observation_matrix,
                noise,
                pref.goal,
                pref.precision,
                info_block,
            )
            eps.append(float(parts["epistemic"]))
            gees.append(float(g))
        members = [float(a[0]) for a in np.asarray(CROSSOVER_ACTION_SET.actions)]
        assert members[int(np.argmax(eps))] == a_sense
        assert members[int(np.argmin(gees))] == a_myopic


# --- H* is defined (measured only at the sweep), and the finder's logic holds ------
class TestCrossoverHorizon:
    def test_no_crossover_when_walk_equals_reach(self):
        # A control on the finder: identical policies give ΔG(H) = 0 for all H (never
        # < 0), so there is no crossover and the finder returns None, not a spurious H*.
        backend, belief, pref, info_block = _setup()
        a_sense, _ = _anchor_actions()

        def same(h):
            return _constant(a_sense, h)

        h_star = crossover_horizon(
            backend, belief, same, same, pref, info_block=info_block, max_horizon=3
        )
        assert h_star is None

    def test_mismatched_horizons_are_rejected(self):
        backend, belief, pref, info_block = _setup()
        a_sense, a_myopic = _anchor_actions()
        with np.testing.assert_raises(ValueError):
            crossover_statistic(
                backend,
                belief,
                _constant(a_sense, 2),
                _constant(a_myopic, 1),
                pref,
                info_block=info_block,
            )


# --- D3 reproducibility: the anchors are exact facts, not seed-dependent draws ------
class TestReproducibility:
    def test_anchors_are_deterministic(self):
        # The anchor scan carries no observation draw, so recomputing the statistic
        # gives bit-identical Δε, Δc, ΔG, not a noise-gated coincidence.
        backend, belief, pref, info_block = _setup()
        a_sense, a_myopic = _anchor_actions()
        first = crossover_statistic(
            backend,
            belief,
            _constant(a_sense, 1),
            _constant(a_myopic, 1),
            pref,
            info_block=info_block,
        )
        second = crossover_statistic(
            backend,
            belief,
            _constant(a_sense, 1),
            _constant(a_myopic, 1),
            pref,
            info_block=info_block,
        )
        np.testing.assert_array_equal(first.delta_epsilon, second.delta_epsilon)
        np.testing.assert_array_equal(first.delta_c, second.delta_c)
        np.testing.assert_array_equal(first.delta_g, second.delta_g)


def test_crossover_statistic_is_a_value():
    backend, belief, pref, info_block = _setup()
    a_sense, a_myopic = _anchor_actions()
    stat = crossover_statistic(
        backend,
        belief,
        _constant(a_sense, 1),
        _constant(a_myopic, 1),
        pref,
        info_block=info_block,
    )
    assert isinstance(stat, CrossoverStatistic)
