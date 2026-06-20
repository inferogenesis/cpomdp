"""Workstream B / B4 — EFESelector.horizon (constant-action lookahead).

`EFESelector` gains `horizon: int = 1`:

- at H=1 it runs the literal current path (the existing `test_efe_selector.py` is the
  unmodified gate; here `TestHorizonOneUnchanged` also pins explicit-1 == default,
  byte-identical);
- at H>1 the candidate family is **constant-action policies** — each grid action
  repeated H times — scored by `policy_efe`; `select` returns the **first** (= constant)
  action of the best policy (receding-horizon: apply first, re-plan). This picks the
  best *constant* action, NOT the best *sequence* — the honest caveat.

`cost_per_cycle = n_candidates * horizon` (attributable step-evals, RFC-001). `horizon`
threads `ObservationGoal` -> the Agent-built selector; default 1 = no behaviour change.

Until B4 lands the `horizon` kwarg / fields / `cost_per_cycle` don't exist, so the new
tests fail (TypeError / AttributeError) — that is the build cue. The gate beyond this
file: `test_efe_selector.py` must pass UNMODIFIED.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp.agent import Agent
from cpomdp.efe import expected_free_energy, policy_efe
from cpomdp.observation import CallableSensor
from cpomdp.selection import EFESelector, ObservationGoal, Preference
from cpomdp.types import Belief, LinearGaussianModel


def _well_noise(x, params):
    pos = x[0]
    falloff = 1.0 - jnp.exp(
        -((pos - params["beacon"]) ** 2) / (2.0 * params["width"] ** 2)
    )
    return jnp.array([[params["r_lo"] + (params["r_hi"] - params["r_lo"]) * falloff]])


def _model():
    # 1-D single integrator with an R(x) beacon at 1.5 (the corridor; p = 1).
    sensor = CallableSensor(
        sensor_model=[[1.0]],
        noise_fn=_well_noise,
        noise_params={
            "beacon": jnp.array(1.5),
            "width": jnp.array(0.6),
            "r_lo": jnp.array(0.05),
            "r_hi": jnp.array(0.8),
        },
    )
    return LinearGaussianModel(
        dynamics=[[1.0]],
        sensor_model=[[1.0]],
        dynamics_noise=[[0.05]],
        sensor_noise=[[0.3]],
        prior=Belief(mean=[0.0], cov=[[0.5]]),
        control=[[1.0]],
        observation=sensor,
    )


def _belief():
    return Belief(mean=[1.0], cov=[[0.5]])


def _pref():
    return Preference(goal=[0.0], precision=[[0.4]])


class TestHorizonConstruction:
    def test_default_horizon_is_one(self):
        sel = EFESelector(_model(), n_candidates=11, action_bounds=(-2.0, 2.0))
        assert sel.horizon == 1
        assert sel.cost_per_cycle == 11  # n_candidates * 1

    def test_cost_per_cycle_scales_with_horizon(self):
        sel = EFESelector(
            _model(), n_candidates=11, action_bounds=(-2.0, 2.0), horizon=3
        )
        assert sel.horizon == 3
        assert sel.cost_per_cycle == 33  # n_candidates * horizon

    def test_horizon_below_one_raises(self):
        with np.testing.assert_raises(ValueError):
            EFESelector(_model(), n_candidates=11, action_bounds=(-2.0, 2.0), horizon=0)


class TestHorizonOneUnchanged:
    """horizon=1 is the current path — explicit-1 equals the default, bit-for-bit."""

    def test_explicit_h1_equals_default(self):
        model, belief, pref = _model(), _belief(), _pref()
        default = EFESelector(model, n_candidates=21, action_bounds=(-3.0, 3.0))
        h1 = EFESelector(model, n_candidates=21, action_bounds=(-3.0, 3.0), horizon=1)
        np.testing.assert_array_equal(
            default.select(belief, pref), h1.select(belief, pref)
        )


class TestHorizonSelect:
    def test_h2_chooses_the_constant_action_minimising_policy_efe(self):
        # The H=2 choice must be the grid action whose constant-action policy minimises
        # the rollout EFE — exercises tile -> policy_efe -> argmin -> first-action.
        model, belief, pref = _model(), _belief(), _pref()
        n = 21
        sel = EFESelector(model, n_candidates=n, action_bounds=(-3.0, 3.0), horizon=2)
        chosen = sel.select(belief, pref)

        def g_of(c):
            policy = jnp.broadcast_to(c[None, :], (2, 1))  # constant action, H=2
            return float(policy_efe(model, belief, policy, pref)[0])

        cands = jnp.linspace(-3.0, 3.0, n)[:, None]
        g_min = min(g_of(c) for c in cands)
        np.testing.assert_allclose(g_of(chosen), g_min, atol=1e-9)

    def test_select_returns_single_action_not_a_sequence(self):
        sel = EFESelector(
            _model(), n_candidates=21, action_bounds=(-3.0, 3.0), horizon=2
        )
        chosen = sel.select(_belief(), _pref())
        assert chosen.shape == (1,)  # the first/constant action, not an (H, p) sequence


class TestObservationGoalHorizon:
    def test_default_horizon_is_one(self):
        assert ObservationGoal([0.0], (-2.0, 2.0)).horizon == 1

    def test_carries_horizon(self):
        assert ObservationGoal([0.0], (-2.0, 2.0), horizon=4).horizon == 4

    def test_horizon_below_one_raises(self):
        with np.testing.assert_raises(ValueError):
            ObservationGoal([0.0], (-2.0, 2.0), horizon=0)


class TestAgentThreadsHorizon:
    def test_agent_builds_selector_with_goal_horizon(self):
        model = _model()
        agent = Agent(
            model, ObservationGoal([0.0], (-3.0, 3.0), n_candidates=11, horizon=3)
        )
        sel = agent._selector
        assert isinstance(sel, EFESelector)
        assert sel.horizon == 3
        assert sel.cost_per_cycle == 33

    def test_default_goal_gives_horizon_one_selector(self):
        model = _model()
        agent = Agent(model, ObservationGoal([0.0], (-3.0, 3.0), n_candidates=11))
        sel = agent._selector
        assert isinstance(sel, EFESelector)
        assert sel.horizon == 1


class TestEFESelectorValidation:
    def test_rejects_too_few_candidates(self):
        with pytest.raises(ValueError, match="at least 2"):
            EFESelector(_model(), n_candidates=1, action_bounds=(-2.0, 2.0))

    def test_rejects_inverted_bounds(self):
        with pytest.raises(ValueError, match="lo < hi"):
            EFESelector(_model(), n_candidates=11, action_bounds=(2.0, -2.0))

    def test_rejects_multidimensional_action(self):
        # EFESelector's grid is 1-D; p>1 must error clearly, not crash cryptically.
        m2 = LinearGaussianModel(
            dynamics=[[1.0, 0.0], [0.0, 1.0]],
            sensor_model=[[1.0, 0.0]],
            dynamics_noise=[[0.1, 0.0], [0.0, 0.1]],
            sensor_noise=[[0.3]],
            prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
            control=[[1.0, 0.0], [0.0, 1.0]],  # p = 2
        )
        with pytest.raises(ValueError, match=r"1-D action grid|p=1|p>1"):
            EFESelector(m2, n_candidates=11, action_bounds=(-2.0, 2.0))

    def test_nan_scoring_candidate_does_not_win(self):
        # A non-PD R(x) at reachable states scores those candidates NaN; the selector
        # must pick a finite-scoring action, not the NaN one (the nan-safe argmin).
        def half_neg(x, params):
            return jnp.array([[1.0 - x[0]]])  # PD at probe x=0; non-PD where x[0] > 1

        belief = Belief(mean=[0.0], cov=[[0.5]])
        model = LinearGaussianModel(
            dynamics=[[1.0]],
            sensor_model=[[1.0]],
            dynamics_noise=[[0.05]],
            sensor_noise=[[0.3]],
            prior=belief,
            control=[[1.0]],
            observation=CallableSensor([[1.0]], half_neg, {}),
        )
        sel = EFESelector(model, n_candidates=21, action_bounds=(-3.0, 3.0))
        chosen = sel.select(belief, _pref())
        g = float(expected_free_energy(model, belief, chosen, _pref())[0])
        assert jnp.isfinite(g)  # a NaN-scoring candidate did not win
