"""Workstream B / B5 — the double-integrator demo: H=1 is myopic, H=2 acts.

The plant is a position-velocity double integrator. The action moves **velocity**; the
sensor observes **position**. So one step of action changes velocity but not yet the
observed position — at H=1 the predicted observation `o⁺ = C·μ⁺` is identical for every
action, so `G` is exactly action-flat (the kernel cannot see the delayed consequence).
At H=2 the rollout carries the velocity change into position, the observation moves, and
the selector picks a sensible action toward the goal. This retires the myopia caveat at
DECISIONS.md:614-619 empirically.

No new src code here — the behaviour emerges from `policy_efe` (B2) and
`EFESelector.horizon` (B4); this is the demonstration that the seam works.
"""

import jax.numpy as jnp
import numpy as np

from cpomdp.efe import expected_free_energy, policy_efe
from cpomdp.selection import EFESelector, Preference
from cpomdp.types import Belief, LinearGaussianModel


def _double_integrator():
    # state = [position, velocity]; A advances position by velocity; control moves
    # velocity only ([[0], [1]]); C observes position only ([[1, 0]]).
    return LinearGaussianModel(
        dynamics=[[1.0, 1.0], [0.0, 1.0]],
        sensor_model=[[1.0, 0.0]],
        dynamics_noise=[[0.05, 0.0], [0.0, 0.05]],
        sensor_noise=[[0.2]],
        prior=Belief(mean=[0.0, 0.0], cov=[[0.3, 0.0], [0.0, 0.3]]),
        control=[[0.0], [1.0]],
    )


_BELIEF = Belief(mean=[0.0, 0.0], cov=[[0.3, 0.0], [0.0, 0.3]])
_GOAL = Preference(goal=[1.0], precision=[[1.0]])  # prefer to observe position = 1


class TestDoubleIntegratorHorizon:
    def test_h1_is_exactly_action_flat(self):
        # The action moves velocity, not yet position, so o⁺ = C·μ⁺ is identical across
        # actions — one-step G is flat to machine precision (genuine myopia).
        model = _double_integrator()
        actions = jnp.linspace(-2.0, 2.0, 21)[:, None]
        gs = jnp.array(
            [expected_free_energy(model, _BELIEF, a, _GOAL)[0] for a in actions]
        )
        assert float(jnp.max(gs) - jnp.min(gs)) < 1e-9

    def test_h2_picks_a_positive_action_toward_the_goal(self):
        # Goal is position +1 from a standstill at 0, so the agent must raise velocity:
        # a positive action. H=1 could not see this; H=2 can.
        model = _double_integrator()
        sel = EFESelector(model, n_candidates=41, action_bounds=(-2.0, 2.0), horizon=2)
        assert float(sel.select(_BELIEF, _GOAL)[0]) > 0.0

    def test_h2_choice_matches_brute_force_oracle(self):
        # The H=2 choice is the constant action minimising the 2-step rollout EFE over a
        # fine independent grid (policy_efe is the B3-verified rollout).
        model = _double_integrator()
        sel = EFESelector(model, n_candidates=41, action_bounds=(-2.0, 2.0), horizon=2)
        chosen = float(sel.select(_BELIEF, _GOAL)[0])

        fine = np.linspace(-2.0, 2.0, 161)
        gs = [
            float(policy_efe(model, _BELIEF, jnp.array([[a], [a]]), _GOAL)[0])
            for a in fine
        ]
        oracle = float(fine[int(np.argmin(gs))])
        assert abs(chosen - oracle) <= 4.0 / 40  # within the selector's grid spacing
