"""Receding-horizon and open-loop drivers over `EnumeratedEfeSearch`.

The enumerated search returns a whole best sequence. Two drivers turn that into an
`ActionSelector` the `Agent` loop can run, and they differ in a way item E's
matched-horizon bracket depends on:

- `RecedingHorizonSelector` re-runs the full search on the *current* belief every step
  and applies the first action of the best sequence. The belief drives every choice.
- `OpenLoopSelector` plans once, commits to the whole best sequence, and applies its
  actions in order while ignoring the interim beliefs. It re-plans only when the
  committed sequence is exhausted.

Cost accounting is honest per mode. Both expose `cost_per_plan = |A|^H * H` (one search)
and a `replan_interval` (1 receding, H open-loop). `cost_per_cycle` is the amortized
per-cycle cost, so `cost_per_cycle * cycles` is a correct total either way: `|A|^H * H`
for receding, `|A|^H` for open-loop. Neither reports the grid's `n_candidates * H`.

Imports the two selectors directly, so until they land this module is collection-red.
"""

import jax.numpy as jnp
import numpy as np

from cpomdp.agent import Agent
from cpomdp.enumeration import (
    EnumeratedEfeSearch,
    FiniteActionSet,
    OpenLoopSelector,
    RecedingHorizonSelector,
    SearchWarrant,
)
from cpomdp.selection import ActionSelector, Preference
from cpomdp.types import Belief, LinearGaussianModel


# --- fixtures ----------------------------------------------------------------------
def _model():
    return LinearGaussianModel(
        dynamics_matrix=[[1.0, 0.1], [0.0, 1.0]],
        observation_matrix=[[1.0, 0.0]],
        dynamics_noise=[[0.1, 0.0], [0.0, 0.1]],
        observation_noise=[[0.5]],
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        control_matrix=[[0.0], [1.0]],
    )


def _action_set():
    return FiniteActionSet([[-1.0], [0.0], [1.0]], version="v1")


def _search(horizon=2):
    return EnumeratedEfeSearch(_model(), _action_set(), horizon=horizon)


def _pref():
    return Preference(target=[1.0], precision=[[2.0]])


def _b0():
    return Belief(mean=[0.3, -0.2], cov=[[0.7, 0.1], [0.1, 0.4]])


def _b_other():
    return Belief(mean=[-0.5, 0.4], cov=[[0.5, 0.0], [0.0, 0.6]])


# --- receding horizon --------------------------------------------------------------
class TestRecedingHorizonSelector:
    def test_returns_first_action_of_the_best_sequence(self):
        search, pref = _search(), _pref()
        sel = RecedingHorizonSelector(search)
        expected = search.evaluate(_b0(), pref).best_policy[0]
        np.testing.assert_array_equal(sel.select(_b0(), pref), expected)

    def test_replans_on_the_current_belief_each_step(self):
        # Feeding two different beliefs must give each one's own best first action,
        # which is what re-planning every step means.
        search, pref = _search(), _pref()
        sel = RecedingHorizonSelector(search)
        for belief in (_b0(), _b_other()):
            expected = search.evaluate(belief, pref).best_policy[0]
            np.testing.assert_array_equal(sel.select(belief, pref), expected)

    def test_conforms_to_action_selector(self):
        assert isinstance(RecedingHorizonSelector(_search()), ActionSelector)

    def test_cost_accounting(self):
        sel = RecedingHorizonSelector(_search(horizon=2))  # |A|=3, H=2
        assert sel.cost_per_plan == 18  # |A|^H * H = 9 * 2
        assert sel.replan_interval == 1
        assert sel.cost_per_cycle == 18  # plans every cycle

    def test_delegates_introspection(self):
        sel = RecedingHorizonSelector(_search(horizon=2))
        assert sel.horizon == 2
        assert sel.warrant is SearchWarrant.PROVED
        assert sel.certificate.complete


# --- open loop ---------------------------------------------------------------------
class TestOpenLoopSelector:
    def test_commits_to_the_sequence_and_ignores_interim_beliefs(self):
        search, pref = _search(horizon=2), _pref()
        plan = np.asarray(search.evaluate(_b0(), pref).best_policy)  # (H, p)
        sel = OpenLoopSelector(search)
        # first call plans on b0; the rest feed a different belief that must be ignored
        np.testing.assert_array_equal(sel.select(_b0(), pref), plan[0])
        np.testing.assert_array_equal(sel.select(_b_other(), pref), plan[1])

    def test_replans_when_the_sequence_is_exhausted(self):
        search, pref = _search(horizon=2), _pref()
        sel = OpenLoopSelector(search)
        sel.select(_b0(), pref)  # step 0 (plans on b0)
        sel.select(_b0(), pref)  # step 1 (last of the committed plan)
        # step 2: the plan is spent, so it re-plans on the belief it is given now
        fresh = search.evaluate(_b_other(), pref).best_policy[0]
        np.testing.assert_array_equal(sel.select(_b_other(), pref), fresh)

    def test_reset_clears_the_committed_plan(self):
        search, pref = _search(horizon=2), _pref()
        sel = OpenLoopSelector(search)
        sel.select(_b0(), pref)  # commits to b0's plan, index now 1
        sel.reset()
        # after reset the next call re-plans from scratch on the new belief
        fresh = search.evaluate(_b_other(), pref).best_policy[0]
        np.testing.assert_array_equal(sel.select(_b_other(), pref), fresh)

    def test_conforms_to_action_selector(self):
        assert isinstance(OpenLoopSelector(_search()), ActionSelector)

    def test_cost_accounting(self):
        sel = OpenLoopSelector(_search(horizon=2))  # |A|=3, H=2
        assert sel.cost_per_plan == 18  # |A|^H * H = 9 * 2
        assert sel.replan_interval == 2  # = horizon
        assert sel.cost_per_cycle == 9  # amortized: |A|^H (one plan per H steps)


# --- the two modes genuinely differ ------------------------------------------------
class TestModesDiffer:
    def test_open_loop_can_disagree_with_receding_mid_sequence(self):
        # After committing on b0, open-loop returns the committed step-1 action while
        # receding re-plans on the drifted belief. Where those differ, the two modes
        # take different actions — the distinction item E's bracket rests on.
        search, pref = _search(horizon=2), _pref()
        committed = np.asarray(search.evaluate(_b0(), pref).best_policy)[1]
        replanned = np.asarray(search.evaluate(_b_other(), pref).best_policy)[0]
        # the fixtures are chosen so these differ; if they ever coincide the test is
        # vacuous, so assert the premise too
        assert not np.array_equal(committed, replanned)

        open_loop = OpenLoopSelector(search)
        open_loop.select(_b0(), pref)  # commit on b0
        receding = RecedingHorizonSelector(search)
        np.testing.assert_array_equal(open_loop.select(_b_other(), pref), committed)
        np.testing.assert_array_equal(receding.select(_b_other(), pref), replanned)


# --- drives the Agent loop ---------------------------------------------------------
class TestDrivesAgent:
    def test_receding_selector_runs_in_the_agent_loop(self):
        model = _model()
        search = EnumeratedEfeSearch(model, _action_set(), horizon=2)
        agent = Agent(
            model, Preference([1.0]), selector=RecedingHorizonSelector(search)
        )
        agent.infer_states(jnp.array([0.1]))
        action = agent.sample_action()
        assert action.shape == (1,)
        # a second perceive/act step advances without error
        agent.infer_states(jnp.array([0.2]))
        assert agent.sample_action().shape == (1,)
