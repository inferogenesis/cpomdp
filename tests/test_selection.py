"""Preference value type and the LQRSelector pass-through adapter."""

import numpy as np
import pytest

from cpomdp.control import LQRController
from cpomdp.selection import (
    ActionSelector,
    LQRSelector,
    ObservationGoal,
    Preference,
    StateGoal,
)
from cpomdp.types import Belief, LinearGaussianModel

# A double-integrator point mass, matching the control-test plant: state =
# [position, velocity], one force on the velocity. Reused to build a controller.
DYNAMICS = [[1.0, 0.1], [0.0, 1.0]]
CONTROL = [[0.0], [0.1]]
GOAL_PRECISION = [[1.0, 0.0], [0.0, 1.0]]
EFFORT_PENALTY = [[0.1]]


def _point_mass_model():
    return LinearGaussianModel(
        dynamics=DYNAMICS,
        sensor_model=[[1.0, 0.0]],
        dynamics_noise=[[1e-4, 0.0], [0.0, 1e-4]],
        sensor_noise=[[1e-2]],
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        control=CONTROL,
    )


def _controller():
    return LQRController(
        _point_mass_model(),
        goal_precision=GOAL_PRECISION,
        effort_penalty=EFFORT_PENALTY,
    )


class TestPreference:
    def test_stores_and_coerces_goal(self):
        p = Preference(goal=[1.0, 2.0])
        assert isinstance(p.goal, np.ndarray) or hasattr(p.goal, "shape")
        np.testing.assert_array_equal(p.goal, [1.0, 2.0])

    def test_precision_defaults_to_identity(self):
        p = Preference(goal=[1.0, 2.0])
        np.testing.assert_array_equal(p.precision, np.eye(2))

    def test_accepts_an_explicit_precision(self):
        p = Preference(goal=[0.0, 0.0], precision=[[2.0, 0.0], [0.0, 3.0]])
        np.testing.assert_array_equal(p.precision, [[2.0, 0.0], [0.0, 3.0]])

    def test_rejects_non_1d_goal(self):
        with pytest.raises(ValueError, match="1-D"):
            Preference(goal=[[1.0]])

    def test_rejects_asymmetric_precision(self):
        with pytest.raises(ValueError, match="symmetric"):
            Preference(goal=[0.0, 0.0], precision=[[1.0, 0.2], [0.9, 1.0]])

    def test_rejects_negative_definite_precision(self):
        # A negative-definite "precision" makes the EFE pragmatic cost negative and
        # inverts the agent's behaviour — reject it as a covariance/precision.
        with pytest.raises(ValueError, match="positive-semi-definite"):
            Preference(goal=[0.0], precision=[[-1.0]])

    def test_rejects_precision_shape_mismatch(self):
        with pytest.raises(ValueError, match="match"):
            Preference(goal=[0.0, 0.0], precision=[[1.0]])


class TestLQRSelector:
    def test_satisfies_the_action_selector_protocol(self):
        selector = LQRSelector(_controller())
        assert isinstance(selector, ActionSelector)

    def test_select_is_a_faithful_pass_through(self):
        controller = _controller()
        selector = LQRSelector(controller)
        belief = Belief(mean=[0.3, -0.2], cov=[[1.0, 0.0], [0.0, 1.0]])
        pref = Preference(goal=[1.0, 0.0])

        np.testing.assert_array_equal(
            selector.select(belief, pref),
            controller.action(belief.mean, pref.goal),
        )


# The public typed objectives. The whole point of the sum type is that an illegal
# state (state knobs on an obs goal, or vice versa) is *unrepresentable* — passing
# the wrong field is a TypeError at construction, never a runtime guard the Agent
# has to remember to write. StateGoal -> LQR (state space); ObservationGoal -> EFE
# (observation space); the Agent dispatches on which type it is given.


class TestStateGoal:
    def test_stores_and_coerces_target(self):
        g = StateGoal([1.0, 2.0])
        np.testing.assert_array_equal(g.target, [1.0, 2.0])

    def test_precision_defaults_to_identity(self):
        g = StateGoal([1.0, 2.0])
        np.testing.assert_array_equal(g.precision, np.eye(2))

    def test_carries_precision_and_effort(self):
        g = StateGoal([0.0], precision=[[2.0]], effort=[[0.5]])
        np.testing.assert_array_equal(g.precision, [[2.0]])
        np.testing.assert_array_equal(g.effort, [[0.5]])

    def test_effort_defaults_to_none(self):
        # p is unknown at construction, so effort stays None and the Agent fills
        # the identity when it builds the controller.
        assert StateGoal([0.0]).effort is None

    def test_rejects_non_1d_target(self):
        with pytest.raises(ValueError, match="1-D"):
            StateGoal([[1.0]])

    def test_rejects_asymmetric_precision(self):
        # The precision must still be validated as a covariance — regression for a
        # validate_covariance call that was once swallowed into the 1-D guard.
        with pytest.raises(ValueError, match="symmetric"):
            StateGoal([0.0, 0.0], precision=[[1.0, 0.2], [0.9, 1.0]])

    def test_rejects_efe_search_knobs(self):
        # Unrepresentable: a state goal has no action-search config.
        with pytest.raises(TypeError):
            StateGoal([0.0], action_bounds=(-1.0, 1.0))  # ty: ignore[unknown-argument]


class TestObservationGoal:
    def test_stores_target_and_search_config(self):
        g = ObservationGoal([0.5], (-2.0, 2.0), n_candidates=15)
        np.testing.assert_array_equal(g.target, [0.5])
        assert g.action_bounds == (-2.0, 2.0)
        assert g.n_candidates == 15

    def test_precision_defaults_to_identity(self):
        g = ObservationGoal([0.0], (-1.0, 1.0))
        np.testing.assert_array_equal(g.precision, np.eye(1))

    def test_n_candidates_defaults_to_21(self):
        assert ObservationGoal([0.0], (-1.0, 1.0)).n_candidates == 21

    def test_rejects_non_1d_target(self):
        with pytest.raises(ValueError, match="1-D"):
            ObservationGoal([[0.0]], (-1.0, 1.0))

    def test_rejects_inverted_action_bounds(self):
        with pytest.raises(ValueError, match="bound"):
            ObservationGoal([0.0], (1.0, -1.0))

    def test_rejects_too_few_candidates(self):
        with pytest.raises(ValueError, match="candidate"):
            ObservationGoal([0.0], (-1.0, 1.0), n_candidates=1)

    def test_rejects_lqr_effort_knob(self):
        # Unrepresentable: an observation goal carries no LQR effort weight.
        with pytest.raises(TypeError):
            ObservationGoal([0.0], (-1.0, 1.0), effort=[[1.0]])  # ty: ignore[unknown-argument]
