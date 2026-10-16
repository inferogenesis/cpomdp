"""Preference value type and the LQRSelector pass-through adapter."""

import numpy as np
import pytest

from cpomdp.control import LQRController
from cpomdp.selection import (
    ActionSelector,
    EFESelector,
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
        dynamics_matrix=DYNAMICS,
        observation_matrix=[[1.0, 0.0]],
        dynamics_noise=[[1e-4, 0.0], [0.0, 1e-4]],
        observation_noise=[[1e-2]],
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        control_matrix=CONTROL,
    )


def _controller():
    return LQRController(
        _point_mass_model(),
        precision=GOAL_PRECISION,
        effort=EFFORT_PENALTY,
    )


class TestPreference:
    def test_stores_and_coerces_target(self):
        p = Preference(target=[1.0, 2.0])
        assert hasattr(p.target, "shape")
        np.testing.assert_array_equal(p.target, [1.0, 2.0])

    def test_calls_its_vector_target_as_the_goals_do(self):
        # StateGoal, ObservationGoal and LQRController.action call it target too.
        with pytest.raises(TypeError):
            Preference(goal=[1.0])  # ty: ignore[missing-argument, unknown-argument]

    def test_precision_defaults_to_identity(self):
        p = Preference(target=[1.0, 2.0])
        np.testing.assert_array_equal(p.precision, np.eye(2))

    def test_accepts_an_explicit_precision(self):
        p = Preference(target=[0.0, 0.0], precision=[[2.0, 0.0], [0.0, 3.0]])
        np.testing.assert_array_equal(p.precision, [[2.0, 0.0], [0.0, 3.0]])

    def test_rejects_non_1d_goal(self):
        with pytest.raises(ValueError, match="1-D"):
            Preference(target=[[1.0]])

    def test_rejects_asymmetric_precision(self):
        with pytest.raises(ValueError, match="symmetric"):
            Preference(target=[0.0, 0.0], precision=[[1.0, 0.2], [0.9, 1.0]])

    def test_rejects_negative_definite_precision(self):
        # A negative-definite "precision" makes the EFE pragmatic cost negative and
        # inverts the agent's behaviour — reject it as a covariance/precision.
        with pytest.raises(ValueError, match="positive-semi-definite"):
            Preference(target=[0.0], precision=[[-1.0]])

    def test_rejects_precision_shape_mismatch(self):
        with pytest.raises(ValueError, match="match"):
            Preference(target=[0.0, 0.0], precision=[[1.0]])


class TestLQRSelector:
    def test_satisfies_the_action_selector_protocol(self):
        selector = LQRSelector(_controller())
        assert isinstance(selector, ActionSelector)

    def test_select_is_a_faithful_pass_through(self):
        controller = _controller()
        selector = LQRSelector(controller)
        belief = Belief(mean=[0.3, -0.2], cov=[[1.0, 0.0], [0.0, 1.0]])
        pref = Preference(target=[1.0, 0.0])

        np.testing.assert_array_equal(
            selector.select(belief, pref),
            controller.action(belief.mean, pref.target),
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

    @pytest.mark.parametrize(
        ("effort", "match"),
        [
            ([1.0], "square"),
            ([[1.0, 0.2], [0.9, 1.0]], "symmetric"),
            ([[0.0]], "positive-definite"),
        ],
    )
    def test_rejects_an_effort_the_controller_cannot_invert(self, effort, match):
        # It used to construct, and the Agent raised later naming effort_penalty.
        with pytest.raises(ValueError, match=match):
            StateGoal([0.0], effort=effort)

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
    def test_action_bounds_is_keyword_only(self):
        # StateGoal already keyword-onlys everything after its first argument, so this
        # was the odd one out in a public pair. The two positional slots also transpose
        # without raising when the target is 2-D and increasing, since (lo, hi) then
        # passes the target's own shape check and the target passes lo < hi.
        with pytest.raises(TypeError, match="positional"):
            # Calling it wrongly is the test, so ty's arity errors are expected.
            ObservationGoal(  # ty: ignore[missing-argument]
                [1.0],
                (-2.0, 2.0),  # ty: ignore[too-many-positional-arguments]
            )

    def test_stores_target_and_search_config(self):
        g = ObservationGoal([0.5], action_bounds=(-2.0, 2.0), n_candidates=15)
        np.testing.assert_array_equal(g.target, [0.5])
        assert g.action_bounds == (-2.0, 2.0)
        assert g.n_candidates == 15

    def test_precision_defaults_to_identity(self):
        g = ObservationGoal([0.0], action_bounds=(-1.0, 1.0))
        np.testing.assert_array_equal(g.precision, np.eye(1))

    def test_n_candidates_defaults_to_21(self):
        assert ObservationGoal([0.0], action_bounds=(-1.0, 1.0)).n_candidates == 21

    def test_rejects_non_1d_target(self):
        with pytest.raises(ValueError, match="1-D"):
            ObservationGoal([[0.0]], action_bounds=(-1.0, 1.0))

    def test_rejects_inverted_action_bounds(self):
        with pytest.raises(ValueError, match="bound"):
            ObservationGoal([0.0], action_bounds=(1.0, -1.0))

    def test_rejects_too_few_candidates(self):
        with pytest.raises(ValueError, match="candidate"):
            ObservationGoal([0.0], action_bounds=(-1.0, 1.0), n_candidates=1)

    def test_rejects_lqr_effort_knob(self):
        # Unrepresentable: an observation goal carries no LQR effort weight.
        with pytest.raises(TypeError):
            ObservationGoal([0.0], action_bounds=(-1.0, 1.0), effort=[[1.0]])  # ty: ignore[unknown-argument]


# The action grid. Three classes took (action_bounds, n_candidates) and wrote the same
# check three times, with Python's `<` on whatever the bounds were. Two sequences
# compare element by element and stop at the first difference, so per-axis bounds with
# the second axis inverted passed, and JAX's linspace failed later in its own words.

_MODEL_1D = LinearGaussianModel(
    dynamics_matrix=[[1.0]],
    observation_matrix=[[1.0]],
    dynamics_noise=[[0.05]],
    observation_noise=[[0.3]],
    prior=Belief(mean=[0.0], cov=[[0.5]]),
    control_matrix=[[1.0]],
)


def _goal(action_bounds, n_candidates=21):
    return ObservationGoal(
        [0.5], action_bounds=action_bounds, n_candidates=n_candidates
    )


def _efe(action_bounds, n_candidates=21):
    return EFESelector(
        _MODEL_1D, action_bounds=action_bounds, n_candidates=n_candidates
    )


@pytest.mark.parametrize("build", [_goal, _efe], ids=["ObservationGoal", "EFESelector"])
class TestActionGrid:
    def test_per_axis_bounds_are_refused_for_a_one_dimensional_grid(self, build):
        with pytest.raises(ValueError, match="two scalars"):
            build(([-1.0, 5.0], [1.0, -5.0]))

    def test_a_bound_that_is_not_a_pair_is_refused(self, build):
        with pytest.raises(ValueError, match=r"pair \(lo, hi\)"):
            build((1.0,))

    @pytest.mark.parametrize("bounds", [(-1.0, np.inf), (np.nan, 1.0)])
    def test_a_bound_that_is_not_finite_is_refused(self, build, bounds):
        with pytest.raises(ValueError, match="finite"):
            build(bounds)

    def test_inverted_bounds_are_refused(self, build):
        with pytest.raises(ValueError, match="lo < hi"):
            build((1.0, -1.0))

    @pytest.mark.parametrize("n_candidates", [1, 2.5])
    def test_a_count_that_cannot_search_is_refused(self, build, n_candidates):
        with pytest.raises((ValueError, TypeError), match="n_candidates"):
            build((-1.0, 1.0), n_candidates)


# The horizon was checked twice with a bare `< 1`. A float passed, and failed later
# inside `range` or `jnp.repeat`, far from the argument that caused it.


@pytest.mark.parametrize(
    "build",
    [
        lambda horizon: ObservationGoal(
            [0.5], action_bounds=(-1.0, 1.0), horizon=horizon
        ),
        lambda horizon: EFESelector(
            _MODEL_1D, action_bounds=(-1.0, 1.0), horizon=horizon
        ),
    ],
    ids=["ObservationGoal", "EFESelector"],
)
class TestHorizon:
    def test_a_horizon_below_one_is_refused(self, build):
        with pytest.raises(ValueError, match="horizon must be at least 1"):
            build(0)

    def test_a_fractional_horizon_is_refused(self, build):
        with pytest.raises(TypeError, match="horizon must be an integer"):
            build(2.5)

    def test_an_integer_of_any_kind_is_held_as_an_int(self, build):
        assert type(build(np.int64(2)).horizon) is int


def test_the_goal_stores_its_bounds_as_two_floats():
    bounds = _goal((np.float32(-1), 2)).action_bounds
    assert bounds == (-1.0, 2.0)
    assert all(type(bound) is float for bound in bounds)


def test_the_grid_selector_defaults_to_the_goals_candidate_count():
    assert EFESelector(_MODEL_1D, action_bounds=(-1.0, 1.0)).n_candidates == 21


def test_the_grid_is_readable_as_the_actions_searched():
    selector = _efe((-1.0, 1.0), n_candidates=5)
    np.testing.assert_array_equal(
        selector.candidates, [[-1.0], [-0.5], [0.0], [0.5], [1.0]]
    )
    assert selector.n_candidates == selector.candidates.shape[0]
