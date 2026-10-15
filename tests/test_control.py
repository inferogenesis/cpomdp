import numpy as np
import pytest
import scipy.linalg

from _support import (
    CONTROL,
    DYNAMICS,
    EFFORT_PENALTY,
    GOAL_PRECISION,
    brute_force_plan,
    numpy_kalman_covariances,
    point_mass_model,
    point_mass_schedule,
)
from cpomdp.backends.kalman import KalmanBackend
from cpomdp.control import (
    KalmanSchedule,
    LQRController,
    LQRSchedule,
    kalman_schedule,
    lqr_schedule,
)
from cpomdp.types import Belief, LinearGaussianModel

# Inside this module the terse letters a/b/qc/rc are local scalars for the
# Riccati/gain hand-math below, NOT the role-named public API
# (dynamics/control/goal_precision/effort_penalty). The library deliberately spells
# those out to avoid the Q/R collision (ADR-003); here, where we're transcribing
# the textbook DARE formula to check it line-for-line, the letters keep the
# matrix algebra readable and carry no API meaning.


def _scipy_gain(dynamics_matrix, control_matrix, goal_precision, effort_penalty):
    """L∞ via scipy's Schur-based DARE solver — the independent oracle.

    scipy returns the cost-to-go P, not the gain, so we derive the gain with the
    same closing formula LQRController uses: L∞ = (Rc + BᵀPB)⁻¹(BᵀPA). Because
    scipy reaches P by Schur decomposition rather than value iteration, a
    transpose or orientation bug in our loop can't survive in both.
    """
    a = np.asarray(dynamics_matrix, dtype=float)
    b = np.asarray(control_matrix, dtype=float)
    qc = np.asarray(goal_precision, dtype=float)
    rc = np.asarray(effort_penalty, dtype=float)
    p = scipy.linalg.solve_discrete_are(a, b, qc, rc)
    return np.linalg.solve(rc + b.T @ p @ b, b.T @ p @ a)


class TestLQRGain:
    def test_gain_matches_scipy_dare(self):
        # The core oracle: our hand-rolled fixed-point iteration must agree with
        # scipy's independent Schur solve. Disagreement = the bug is ours.
        controller = LQRController(
            point_mass_model(),
            goal_precision=GOAL_PRECISION,
            effort_penalty=EFFORT_PENALTY,
        )
        expected = _scipy_gain(DYNAMICS, CONTROL, GOAL_PRECISION, EFFORT_PENALTY)
        np.testing.assert_allclose(controller.gain, expected, atol=1e-8)

    def test_gain_has_shape_p_by_n(self):
        controller = LQRController(
            point_mass_model(),
            goal_precision=GOAL_PRECISION,
            effort_penalty=EFFORT_PENALTY,
        )
        # one action, two states
        assert controller.gain.shape == (1, 2)

    def test_closed_loop_is_stable(self):
        # The real test that the gain is right in sign AND magnitude: the
        # closed-loop dynamics (A - B·L∞) must be stable, i.e. every eigenvalue
        # strictly inside the unit circle. A sign-flipped gain would push the
        # eigenvalues out and this would fail loudly.
        controller = LQRController(
            point_mass_model(),
            goal_precision=GOAL_PRECISION,
            effort_penalty=EFFORT_PENALTY,
        )
        a = np.asarray(DYNAMICS)
        b = np.asarray(CONTROL)
        closed_loop = a - b @ controller.gain
        eigvals = np.linalg.eigvals(closed_loop)
        assert np.all(np.abs(eigvals) < 1.0)


class TestLQRAction:
    def test_action_pushes_toward_goal(self):
        # From the origin with a target at position +1, the force must be
        # positive (accelerate toward the goal). A dropped minus sign flips this.
        controller = LQRController(
            point_mass_model(),
            goal_precision=GOAL_PRECISION,
            effort_penalty=EFFORT_PENALTY,
        )
        action = controller.action(mean=np.array([0.0, 0.0]), goal=np.array([1.0, 0.0]))
        assert action[0] > 0

    def test_zero_error_gives_zero_action(self):
        # Sitting exactly on an equilibrium goal, the controller asks for nothing.
        controller = LQRController(
            point_mass_model(),
            goal_precision=GOAL_PRECISION,
            effort_penalty=EFFORT_PENALTY,
        )
        action = controller.action(mean=np.array([1.0, 0.0]), goal=np.array([1.0, 0.0]))
        np.testing.assert_allclose(action, [0.0], atol=1e-12)

    def test_rejects_wrong_shape_goal(self):
        controller = LQRController(
            point_mass_model(),
            goal_precision=GOAL_PRECISION,
            effort_penalty=EFFORT_PENALTY,
        )
        with pytest.raises(ValueError, match="goal"):
            controller.action(mean=np.array([0.0, 0.0]), goal=np.array([1.0, 0.0, 0.0]))


class TestLQRValidation:
    def test_rejects_model_without_control(self):
        model = LinearGaussianModel(
            dynamics_matrix=DYNAMICS,
            observation_matrix=[[1.0, 0.0]],
            dynamics_noise=[[1e-4, 0.0], [0.0, 1e-4]],
            observation_noise=[[1e-2]],
            prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        )  # no control matrix
        with pytest.raises(ValueError, match="control matrix"):
            LQRController(
                model, goal_precision=GOAL_PRECISION, effort_penalty=EFFORT_PENALTY
            )

    def test_rejects_wrong_goal_precision_shape(self):
        with pytest.raises(ValueError, match="goal_precision"):
            LQRController(
                point_mass_model(),
                goal_precision=[[1.0]],
                effort_penalty=EFFORT_PENALTY,
            )

    def test_rejects_wrong_effort_penalty_shape(self):
        with pytest.raises(ValueError, match="effort_penalty"):
            LQRController(
                point_mass_model(),
                goal_precision=GOAL_PRECISION,
                effort_penalty=[[1.0, 0.0], [0.0, 1.0]],
            )

    def test_rejects_asymmetric_goal_precision(self):
        # An off-diagonal typo: symmetric on shape, asymmetric in value. Without
        # the check this silently yields a non-symmetric cost-to-go and a wrong
        # gain — the hardest failure to trace in a control loop.
        with pytest.raises(ValueError, match="symmetric"):
            LQRController(
                point_mass_model(),
                goal_precision=[[1.0, 0.5], [-0.5, 1.0]],
                effort_penalty=EFFORT_PENALTY,
            )

    def test_rejects_indefinite_effort_penalty(self):
        # effort_penalty is inverted against in the gain solve; a zero (singular)
        # cost must fail loudly, not blow up mid-recursion.
        with pytest.raises(ValueError, match="positive-definite"):
            LQRController(
                point_mass_model(),
                goal_precision=GOAL_PRECISION,
                effort_penalty=[[0.0]],
            )

    def test_rejects_negative_semidefinite_goal_precision(self):
        with pytest.raises(ValueError, match="positive-semi-definite"):
            LQRController(
                point_mass_model(),
                goal_precision=[[-1.0, 0.0], [0.0, 1.0]],
                effort_penalty=EFFORT_PENALTY,
            )

    def test_raises_when_not_converged(self):
        # max_iter too small to reach the fixed point -> a loud failure, not a
        # silently-wrong frozen gain. Mirrors the Kalman steady-state guard.
        with pytest.raises(RuntimeError, match="converge"):
            LQRController(
                point_mass_model(),
                goal_precision=GOAL_PRECISION,
                effort_penalty=EFFORT_PENALTY,
                max_iter=1,
            )


# --- the finite-horizon schedule ------------------------------------------------------


class TestLQRSchedule:
    def test_one_gain_per_step_and_one_cost_per_remaining_horizon(self):
        schedule = point_mass_schedule(6)
        assert schedule.horizon == 6
        assert schedule.gains.shape == (6, 1, 2)
        assert schedule.cost_to_go.shape == (7, 2, 2)
        # Nothing is charged after the last action: the terminal cost is zero.
        np.testing.assert_array_equal(schedule.cost_to_go[0], np.zeros((2, 2)))
        np.testing.assert_array_equal(schedule.first_gain, schedule.gains[0])

    def test_the_one_step_gain_is_the_static_regulator(self):
        a, b = np.asarray(DYNAMICS), np.asarray(CONTROL)
        qc, rc = np.asarray(GOAL_PRECISION), np.asarray(EFFORT_PENALTY)
        expected = np.linalg.solve(rc + b.T @ qc @ b, b.T @ qc @ a)
        np.testing.assert_allclose(
            point_mass_schedule(1).first_gain, expected, rtol=1e-14
        )

    @pytest.mark.parametrize("horizon", [1, 2, 5, 12])
    def test_the_first_gain_is_the_brute_force_optimum_first_action(self, horizon):
        x0 = np.array([0.8, -0.3])
        expected_action, expected_cost = brute_force_plan(
            DYNAMICS, CONTROL, GOAL_PRECISION, EFFORT_PENALTY, horizon, x0
        )
        schedule = point_mass_schedule(horizon)
        np.testing.assert_allclose(
            -schedule.first_gain @ x0, expected_action, rtol=1e-11, atol=1e-13
        )
        np.testing.assert_allclose(
            x0 @ schedule.cost_to_go[horizon] @ x0, expected_cost, rtol=1e-11
        )

    def test_later_gains_are_the_shorter_schedules_first_gains(self):
        # The gain applied with j steps remaining depends on j alone, so the tail of
        # a long schedule is a shorter schedule. gains[k] has H − k steps remaining.
        long = point_mass_schedule(8)
        for horizon in (1, 3, 8):
            np.testing.assert_array_equal(
                long.gains[8 - horizon], point_mass_schedule(horizon).first_gain
            )

    def test_the_first_gain_converges_to_the_steady_state_gain_and_is_not_it(self):
        steady = LQRController(
            point_mass_model(),
            goal_precision=GOAL_PRECISION,
            effort_penalty=EFFORT_PENALTY,
        ).gain
        mismatch = [
            float(np.abs(point_mass_schedule(h).first_gain - steady).max())
            for h in (1, 5, 20)
        ]
        # Shrinks with H and is not zero at any of them: an unmatched comparison
        # reads this as an error that fades with the horizon.
        assert mismatch[0] > mismatch[1] > mismatch[2] > 1e-6
        np.testing.assert_allclose(
            point_mass_schedule(400).first_gain, steady, atol=1e-10
        )

    def test_a_horizon_below_one_is_refused(self):
        with pytest.raises(ValueError, match="horizon"):
            point_mass_schedule(0)

    def test_the_costs_are_validated_as_the_controller_validates_them(self):
        with pytest.raises(ValueError, match="symmetric"):
            lqr_schedule(
                point_mass_model(),
                goal_precision=[[1.0, 0.5], [-0.5, 1.0]],
                effort_penalty=EFFORT_PENALTY,
                horizon=3,
            )
        with pytest.raises(ValueError, match="control matrix"):
            lqr_schedule(
                LinearGaussianModel(
                    dynamics_matrix=DYNAMICS,
                    observation_matrix=[[1.0, 0.0]],
                    dynamics_noise=[[1e-4, 0.0], [0.0, 1e-4]],
                    observation_noise=[[1e-2]],
                    prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
                ),
                goal_precision=GOAL_PRECISION,
                effort_penalty=EFFORT_PENALTY,
                horizon=3,
            )


# --- the Kalman schedule -------------------------------------------------------------


class TestKalmanSchedule:
    def test_matches_the_textbook_recursion(self):
        schedule = kalman_schedule(point_mass_model(), horizon=5)
        gains, covs = numpy_kalman_covariances(np.eye(2), 5)
        np.testing.assert_allclose(schedule.gains, gains, rtol=1e-12)
        np.testing.assert_allclose(schedule.covariances, covs, rtol=1e-12)

    def test_matches_the_backend_step_by_step(self):
        # The backend's posterior covariance after k readings, and its gain read off
        # the mean's response to a unit change in the reading.
        model = point_mass_model()
        backend = KalmanBackend(model)
        schedule = kalman_schedule(model, horizon=4)
        belief, action = model.prior, np.array([0.3])
        for k in range(4):
            reading = np.array([0.5 * k])
            folded = backend.infer_states(reading, belief, action)
            nudged = backend.infer_states(reading + 1.0, belief, action)
            np.testing.assert_allclose(
                schedule.gains[k][:, 0], nudged.mean - folded.mean, atol=1e-12
            )
            np.testing.assert_allclose(
                schedule.covariances[k + 1], folded.cov, rtol=1e-12
            )
            belief = folded

    def test_starts_from_the_prior_given_or_the_models(self):
        model = point_mass_model()
        assert np.array_equal(
            kalman_schedule(model, horizon=2).covariances[0], model.prior.cov
        )
        sharper = Belief(mean=[0.0, 0.0], cov=[[0.1, 0.0], [0.0, 0.1]])
        assert np.array_equal(
            kalman_schedule(model, horizon=2, prior=sharper).covariances[0],
            sharper.cov,
        )


# --- the dual pair -------------------------------------------------------------------


class TestTheDualPair:
    """The control schedule and the filter schedule are called and read alike."""

    def test_both_take_the_horizon_by_keyword(self):
        model = point_mass_model()
        with pytest.raises(TypeError):
            kalman_schedule(model, 3)  # ty: ignore[missing-argument, too-many-positional-arguments]
        with pytest.raises(TypeError):
            lqr_schedule(  # ty: ignore[missing-argument]
                model,
                3,  # ty: ignore[too-many-positional-arguments]
                goal_precision=GOAL_PRECISION,
                effort_penalty=EFFORT_PENALTY,
            )

    def test_both_are_schedules_that_report_their_horizon(self):
        model = point_mass_model()
        plan = lqr_schedule(
            model,
            horizon=4,
            goal_precision=GOAL_PRECISION,
            effort_penalty=EFFORT_PENALTY,
        )
        estimate = kalman_schedule(model, horizon=4)
        assert isinstance(plan, LQRSchedule)
        assert isinstance(estimate, KalmanSchedule)
        assert plan.horizon == estimate.horizon == 4
