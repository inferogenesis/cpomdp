import jax
import numpy as np
import pytest
import scipy.linalg

from cpomdp.backends.kalman import KalmanBackend
from cpomdp.control import (
    CertaintyEquivalentController,
    ControlBracket,
    LQRController,
    closed_loop_cost,
    control_bracket,
    control_efficiency,
    finite_horizon_lqr,
    full_information_cost,
    kalman_schedule,
    optimal_cost,
)
from cpomdp.harness import World
from cpomdp.resolution import EXACT, Bar, Bounded
from cpomdp.types import Belief, LinearGaussianModel

# Inside this module the terse letters a/b/qc/rc are local scalars for the
# Riccati/gain hand-math below, NOT the role-named public API
# (dynamics/control/goal_precision/effort_penalty). The library deliberately spells
# those out to avoid the Q/R collision (ADR-003); here, where we're transcribing
# the textbook DARE formula to check it line-for-line, the letters keep the
# matrix algebra readable and carry no API meaning.

# A double-integrator point mass: state = [position, velocity], a force moves the
# velocity, velocity moves the position. dt small enough to be well-conditioned.
# This is both a controllable system (so a steady-state gain exists) and exactly
# the plant the 2-D reaching demo will use, so the oracle here guards the demo too.
DT = 0.1
DYNAMICS = [[1.0, DT], [0.0, 1.0]]
CONTROL = [[0.0], [DT]]
GOAL_PRECISION = [[1.0, 0.0], [0.0, 1.0]]
EFFORT_PENALTY = [[0.1]]


def _point_mass_model():
    # The noise/sensor fields are required to build a model but don't enter the
    # LQR solve at all — control selection reads only dynamics + control.
    return LinearGaussianModel(
        dynamics_matrix=DYNAMICS,
        observation_matrix=[[1.0, 0.0]],  # observe position
        dynamics_noise=[[1e-4, 0.0], [0.0, 1e-4]],
        observation_noise=[[1e-2]],
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        control_matrix=CONTROL,
    )


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
            _point_mass_model(),
            goal_precision=GOAL_PRECISION,
            effort_penalty=EFFORT_PENALTY,
        )
        expected = _scipy_gain(DYNAMICS, CONTROL, GOAL_PRECISION, EFFORT_PENALTY)
        np.testing.assert_allclose(controller.gain, expected, atol=1e-8)

    def test_gain_has_shape_p_by_n(self):
        controller = LQRController(
            _point_mass_model(),
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
            _point_mass_model(),
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
            _point_mass_model(),
            goal_precision=GOAL_PRECISION,
            effort_penalty=EFFORT_PENALTY,
        )
        action = controller.action(mean=np.array([0.0, 0.0]), goal=np.array([1.0, 0.0]))
        assert action[0] > 0

    def test_zero_error_gives_zero_action(self):
        # Sitting exactly on an equilibrium goal, the controller asks for nothing.
        controller = LQRController(
            _point_mass_model(),
            goal_precision=GOAL_PRECISION,
            effort_penalty=EFFORT_PENALTY,
        )
        action = controller.action(mean=np.array([1.0, 0.0]), goal=np.array([1.0, 0.0]))
        np.testing.assert_allclose(action, [0.0], atol=1e-12)

    def test_rejects_wrong_shape_goal(self):
        controller = LQRController(
            _point_mass_model(),
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
                _point_mass_model(),
                goal_precision=[[1.0]],
                effort_penalty=EFFORT_PENALTY,
            )

    def test_rejects_wrong_effort_penalty_shape(self):
        with pytest.raises(ValueError, match="effort_penalty"):
            LQRController(
                _point_mass_model(),
                goal_precision=GOAL_PRECISION,
                effort_penalty=[[1.0, 0.0], [0.0, 1.0]],
            )

    def test_rejects_asymmetric_goal_precision(self):
        # An off-diagonal typo: symmetric on shape, asymmetric in value. Without
        # the check this silently yields a non-symmetric cost-to-go and a wrong
        # gain — the hardest failure to trace in a control loop.
        with pytest.raises(ValueError, match="symmetric"):
            LQRController(
                _point_mass_model(),
                goal_precision=[[1.0, 0.5], [-0.5, 1.0]],
                effort_penalty=EFFORT_PENALTY,
            )

    def test_rejects_indefinite_effort_penalty(self):
        # effort_penalty is inverted against in the gain solve; a zero (singular)
        # cost must fail loudly, not blow up mid-recursion.
        with pytest.raises(ValueError, match="positive-definite"):
            LQRController(
                _point_mass_model(),
                goal_precision=GOAL_PRECISION,
                effort_penalty=[[0.0]],
            )

    def test_rejects_negative_semidefinite_goal_precision(self):
        with pytest.raises(ValueError, match="positive-semi-definite"):
            LQRController(
                _point_mass_model(),
                goal_precision=[[-1.0, 0.0], [0.0, 1.0]],
                effort_penalty=EFFORT_PENALTY,
            )

    def test_raises_when_not_converged(self):
        # max_iter too small to reach the fixed point -> a loud failure, not a
        # silently-wrong frozen gain. Mirrors the Kalman steady-state guard.
        with pytest.raises(RuntimeError, match="converge"):
            LQRController(
                _point_mass_model(),
                goal_precision=GOAL_PRECISION,
                effort_penalty=EFFORT_PENALTY,
                max_iter=1,
            )


# --- the finite-horizon schedule ------------------------------------------------------


def _brute_force_plan(dynamics, control, goal_precision, effort_penalty, horizon, x0):
    """The H-step open-loop optimum as a stacked least-squares problem.

    Stacks x_1..x_H = Φ·x0 + Γ·U with a cost on every arrived-at state and every
    action, and solves the normal equations for U. No Riccati recursion anywhere,
    so agreement with the schedule is evidence about the schedule.
    """
    a = np.asarray(dynamics, dtype=float)
    b = np.asarray(control, dtype=float)
    n, p = b.shape
    powers = [np.linalg.matrix_power(a, k) for k in range(horizon + 1)]
    phi = np.vstack([powers[k] for k in range(1, horizon + 1)])
    gamma = np.zeros((horizon * n, horizon * p))
    for i in range(horizon):
        for j in range(i + 1):
            gamma[i * n : (i + 1) * n, j * p : (j + 1) * p] = powers[i - j] @ b
    stage = np.kron(np.eye(horizon), np.asarray(goal_precision, dtype=float))
    effort = np.kron(np.eye(horizon), np.asarray(effort_penalty, dtype=float))
    hessian = gamma.T @ stage @ gamma + effort
    linear = gamma.T @ stage @ phi @ x0
    actions = -np.linalg.solve(hessian, linear)
    cost = x0 @ phi.T @ stage @ phi @ x0 - linear @ np.linalg.solve(hessian, linear)
    return actions[:p], cost


def _schedule(horizon):
    return finite_horizon_lqr(
        _point_mass_model(),
        goal_precision=GOAL_PRECISION,
        effort_penalty=EFFORT_PENALTY,
        horizon=horizon,
    )


class TestFiniteHorizonSchedule:
    def test_one_gain_per_step_and_one_cost_per_remaining_horizon(self):
        schedule = _schedule(6)
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
        np.testing.assert_allclose(_schedule(1).first_gain, expected, rtol=1e-14)

    @pytest.mark.parametrize("horizon", [1, 2, 5, 12])
    def test_the_first_gain_is_the_brute_force_optimum_first_action(self, horizon):
        x0 = np.array([0.8, -0.3])
        expected_action, expected_cost = _brute_force_plan(
            DYNAMICS, CONTROL, GOAL_PRECISION, EFFORT_PENALTY, horizon, x0
        )
        schedule = _schedule(horizon)
        np.testing.assert_allclose(
            -schedule.first_gain @ x0, expected_action, rtol=1e-11, atol=1e-13
        )
        np.testing.assert_allclose(
            x0 @ schedule.cost_to_go[horizon] @ x0, expected_cost, rtol=1e-11
        )

    def test_later_gains_are_the_shorter_schedules_first_gains(self):
        # The gain applied with j steps remaining depends on j alone, so the tail of
        # a long schedule is a shorter schedule. gains[k] has H − k steps remaining.
        long = _schedule(8)
        for horizon in (1, 3, 8):
            np.testing.assert_array_equal(
                long.gains[8 - horizon], _schedule(horizon).first_gain
            )

    def test_the_first_gain_converges_to_the_steady_state_gain_and_is_not_it(self):
        steady = LQRController(
            _point_mass_model(),
            goal_precision=GOAL_PRECISION,
            effort_penalty=EFFORT_PENALTY,
        ).gain
        mismatch = [
            float(np.abs(_schedule(h).first_gain - steady).max()) for h in (1, 5, 20)
        ]
        # Shrinks with H and is not zero at any of them: an unmatched comparison
        # reads this as an error that fades with the horizon.
        assert mismatch[0] > mismatch[1] > mismatch[2] > 1e-6
        np.testing.assert_allclose(_schedule(400).first_gain, steady, atol=1e-10)

    def test_a_horizon_below_one_is_refused(self):
        with pytest.raises(ValueError, match="horizon"):
            _schedule(0)

    def test_the_costs_are_validated_as_the_controller_validates_them(self):
        with pytest.raises(ValueError, match="symmetric"):
            finite_horizon_lqr(
                _point_mass_model(),
                goal_precision=[[1.0, 0.5], [-0.5, 1.0]],
                effort_penalty=EFFORT_PENALTY,
                horizon=3,
            )
        with pytest.raises(ValueError, match="control matrix"):
            finite_horizon_lqr(
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


# --- the full-information floor -------------------------------------------------------


GOAL = np.array([1.0, 0.0])  # an equilibrium: zero velocity holds the position
PROCESS_NOISE = np.array([[1e-4, 0.0], [0.0, 1e-4]])


def _sampled_full_information_cost(schedule, goal, prior_mean, prior_cov, draws):
    """The H-step cost with the state known, averaged over sampled trajectories.

    Rolls the dynamics forward under the schedule with a numpy generator, charging
    the stage cost on every arrived-at state and every action. Nothing here reads
    ``cost_to_go``, so agreement is evidence about the closed form.
    """
    rng = np.random.default_rng(0)
    a, b = np.asarray(DYNAMICS), np.asarray(CONTROL)
    qc, rc = np.asarray(GOAL_PRECISION), np.asarray(EFFORT_PENALTY)
    state = rng.multivariate_normal(prior_mean, prior_cov, size=draws)
    cost = np.zeros(draws)
    for gain in np.asarray(schedule.gains):
        action = -(state - goal) @ gain.T
        state = state @ a.T + action @ b.T
        state += rng.multivariate_normal(np.zeros(2), PROCESS_NOISE, size=draws)
        deviation = state - goal
        cost += np.einsum("bi,ij,bj->b", deviation, qc, deviation)
        cost += np.einsum("bi,ij,bj->b", action, rc, action)
    return cost.mean(), cost.std(ddof=1) / np.sqrt(draws)


class TestFullInformationCost:
    def test_the_schedule_carries_what_it_was_built_from(self):
        model = _point_mass_model()
        schedule = finite_horizon_lqr(
            model,
            goal_precision=GOAL_PRECISION,
            effort_penalty=EFFORT_PENALTY,
            horizon=3,
        )
        assert schedule.model is model
        np.testing.assert_array_equal(schedule.goal_precision, GOAL_PRECISION)
        np.testing.assert_array_equal(schedule.effort_penalty, EFFORT_PENALTY)

    def test_matches_a_sampled_rollout_with_the_state_known(self):
        schedule = _schedule(6)
        prior = Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]])
        sampled, standard_error = _sampled_full_information_cost(
            schedule, GOAL, np.zeros(2), np.eye(2), draws=20_000
        )
        closed = full_information_cost(schedule, goal=GOAL, prior=prior)
        assert abs(closed - sampled) < 4 * standard_error

    def test_without_noise_it_is_the_terminal_quadratic_alone(self):
        # A known start and no process noise: the whole cost is x0ᵀ P_H x0, which the
        # brute-force plan already pins.
        model = LinearGaussianModel(
            dynamics_matrix=DYNAMICS,
            observation_matrix=[[1.0, 0.0]],
            dynamics_noise=[[0.0, 0.0], [0.0, 0.0]],
            observation_noise=[[1e-2]],
            prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
            control_matrix=CONTROL,
        )
        schedule = finite_horizon_lqr(
            model,
            goal_precision=GOAL_PRECISION,
            effort_penalty=EFFORT_PENALTY,
            horizon=5,
        )
        x0 = np.array([0.8, -0.3])
        _, expected = _brute_force_plan(
            DYNAMICS, CONTROL, GOAL_PRECISION, EFFORT_PENALTY, 5, x0
        )
        known = Belief(mean=x0, cov=np.zeros((2, 2)))
        assert full_information_cost(
            schedule, goal=np.zeros(2), prior=known
        ) == pytest.approx(expected, rel=1e-12)

    def test_the_prior_covariance_adds_its_trace_against_the_terminal_cost(self):
        schedule = _schedule(4)
        cov = np.array([[0.5, 0.1], [0.1, 0.2]])
        spread = full_information_cost(
            schedule, goal=GOAL, prior=Belief(mean=[0.0, 0.0], cov=cov)
        )
        known = full_information_cost(
            schedule, goal=GOAL, prior=Belief(mean=[0.0, 0.0], cov=np.zeros((2, 2)))
        )
        assert spread - known == pytest.approx(
            np.trace(np.asarray(schedule.cost_to_go[4]) @ cov), rel=1e-12
        )

    def test_the_goal_is_a_shift_of_the_prior_mean(self):
        schedule = _schedule(4)
        at_goal = full_information_cost(
            schedule, goal=GOAL, prior=Belief(mean=[0.3, 0.2], cov=np.eye(2))
        )
        regulated = full_information_cost(
            schedule,
            goal=np.zeros(2),
            prior=Belief(mean=np.array([0.3, 0.2]) - GOAL, cov=np.eye(2)),
        )
        assert at_goal == pytest.approx(regulated, rel=1e-14)

    def test_the_prior_defaults_to_the_models(self):
        schedule = _schedule(4)
        assert full_information_cost(schedule, goal=GOAL) == full_information_cost(
            schedule, goal=GOAL, prior=schedule.model.prior
        )

    def test_a_goal_the_dynamics_cannot_hold_is_refused(self):
        # A moving target: a nonzero velocity is not an equilibrium, so the shifted
        # problem is not a regulator and the closed form does not describe it.
        with pytest.raises(ValueError, match="equilibrium"):
            full_information_cost(_schedule(3), goal=[1.0, 0.5])

    def test_a_goal_of_the_wrong_shape_is_refused(self):
        with pytest.raises(ValueError, match="goal"):
            full_information_cost(_schedule(3), goal=[1.0])

    def test_a_prior_over_another_state_is_refused(self):
        with pytest.raises(ValueError, match="prior"):
            full_information_cost(
                _schedule(3), goal=GOAL, prior=Belief(mean=[0.0], cov=[[1.0]])
            )


# --- the certainty-equivalent controller and its closed-loop cost ---------------------


OBSERVATION = np.array([[1.0, 0.0]])
OBSERVATION_NOISE = np.array([[1e-2]])


def _numpy_kalman_covariances(prior_cov, horizon):
    """The textbook covariance recursion, kept apart from the library's kernel."""
    a, c = np.asarray(DYNAMICS), OBSERVATION
    gains, covs = [], [np.asarray(prior_cov, dtype=float)]
    for _ in range(horizon):
        pred = a @ covs[-1] @ a.T + PROCESS_NOISE
        innovation = c @ pred @ c.T + OBSERVATION_NOISE
        gain = pred @ c.T @ np.linalg.inv(innovation)
        gains.append(gain)
        covs.append((np.eye(2) - gain @ c) @ pred)
    return np.stack(gains), np.stack(covs)


def _sampled_closed_loop_cost(controller_gains, filter_gains, goal, draws):
    """The cost of acting on a filtered estimate, averaged over sampled runs.

    The first action is chosen from the prior mean, each reading arrives after the
    action that preceded it, and the estimate folds it in with the supplied gain.
    Nothing here propagates a second moment.
    """
    rng = np.random.default_rng(1)
    a, b, c = np.asarray(DYNAMICS), np.asarray(CONTROL), OBSERVATION
    qc, rc = np.asarray(GOAL_PRECISION), np.asarray(EFFORT_PENALTY)
    state = rng.multivariate_normal(np.zeros(2), np.eye(2), size=draws)
    estimate = np.zeros((draws, 2))
    cost = np.zeros(draws)
    for gain, filter_gain in zip(controller_gains, filter_gains, strict=True):
        action = -(estimate - goal) @ np.asarray(gain).T
        state = state @ a.T + action @ b.T
        state += rng.multivariate_normal(np.zeros(2), PROCESS_NOISE, size=draws)
        deviation = state - goal
        cost += np.einsum("bi,ij,bj->b", deviation, qc, deviation)
        cost += np.einsum("bi,ij,bj->b", action, rc, action)
        reading = state @ c.T + rng.normal(
            0.0, np.sqrt(OBSERVATION_NOISE[0, 0]), (draws, 1)
        )
        predicted = estimate @ a.T + action @ b.T
        estimate = predicted + (reading - predicted @ c.T) @ np.asarray(filter_gain).T
    return cost.mean(), cost.std(ddof=1) / np.sqrt(draws)


class TestKalmanSchedule:
    def test_matches_the_textbook_recursion(self):
        schedule = kalman_schedule(_point_mass_model(), horizon=5)
        gains, covs = _numpy_kalman_covariances(np.eye(2), 5)
        np.testing.assert_allclose(schedule.gains, gains, rtol=1e-12)
        np.testing.assert_allclose(schedule.covariances, covs, rtol=1e-12)

    def test_matches_the_backend_step_by_step(self):
        # The backend's posterior covariance after k readings, and its gain read off
        # the mean's response to a unit change in the reading.
        model = _point_mass_model()
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
        model = _point_mass_model()
        assert np.array_equal(
            kalman_schedule(model, horizon=2).covariances[0], model.prior.cov
        )
        sharper = Belief(mean=[0.0, 0.0], cov=[[0.1, 0.0], [0.0, 0.1]])
        assert np.array_equal(
            kalman_schedule(model, horizon=2, prior=sharper).covariances[0],
            sharper.cov,
        )


class TestClosedLoopCost:
    def test_one_step_by_hand(self):
        # With one step the filter never acts: the action comes from the prior mean,
        # the state is charged where it lands, and the noise adds its trace.
        plan = _schedule(1)
        a, b = np.asarray(DYNAMICS), np.asarray(CONTROL)
        qc, rc = np.asarray(GOAL_PRECISION), np.asarray(EFFORT_PENALTY)
        gain = np.asarray(plan.first_gain)
        offset = np.array([0.0, 0.0]) - GOAL
        landed = (a - b @ gain) @ offset
        expected = (
            landed @ qc @ landed
            + np.trace(qc @ (a @ np.eye(2) @ a.T + PROCESS_NOISE))
            + offset @ gain.T @ rc @ gain @ offset
        )
        cost = closed_loop_cost(
            plan, kalman_schedule(plan.model, horizon=1).gains, goal=GOAL
        )
        assert cost == pytest.approx(expected, rel=1e-12)

    def test_matches_a_sampled_run_of_the_same_information_pattern(self):
        plan = _schedule(6)
        filter_gains, _ = _numpy_kalman_covariances(np.eye(2), 6)
        sampled, standard_error = _sampled_closed_loop_cost(
            np.asarray(plan.gains), filter_gains, GOAL, draws=20_000
        )
        closed = closed_loop_cost(plan, filter_gains, goal=GOAL)
        assert abs(closed - sampled) < 4 * standard_error

    def test_the_certainty_equivalent_cost_sits_above_the_floor(self):
        plan = _schedule(6)
        controller = CertaintyEquivalentController(plan, goal=GOAL)
        assert controller.expected_cost() > full_information_cost(plan, goal=GOAL)

    def test_the_steady_state_gain_costs_more_than_the_matched_schedule(self):
        # The mismatch from unit one, priced: applying the forever gain at every
        # step of a short plan is not optimal for that plan.
        plan = _schedule(3)
        steady = LQRController(
            plan.model, goal_precision=GOAL_PRECISION, effort_penalty=EFFORT_PENALTY
        ).gain
        gains = kalman_schedule(plan.model, horizon=3).gains
        matched = closed_loop_cost(plan, gains, goal=GOAL)
        unmatched = closed_loop_cost(
            plan, gains, goal=GOAL, controller_gains=np.repeat(steady[None], 3, 0)
        )
        assert unmatched > matched > full_information_cost(plan, goal=GOAL)

    def test_a_filter_gain_per_reading_is_required(self):
        plan = _schedule(3)
        with pytest.raises(ValueError, match="filter_gains"):
            closed_loop_cost(plan, np.zeros((2, 2, 1)), goal=GOAL)
        with pytest.raises(ValueError, match="controller_gains"):
            closed_loop_cost(
                plan,
                np.zeros((3, 2, 1)),
                goal=GOAL,
                controller_gains=np.zeros((2, 1, 2)),
            )


class TestCertaintyEquivalentController:
    def test_acts_on_the_estimate_with_the_steps_gain(self):
        plan = _schedule(4)
        controller = CertaintyEquivalentController(plan, goal=GOAL)
        mean = np.array([0.4, -0.1])
        for step in range(4):
            np.testing.assert_array_equal(
                controller.action(step, mean), -plan.gains[step] @ (mean - GOAL)
            )

    def test_refuses_a_goal_the_dynamics_cannot_hold(self):
        with pytest.raises(ValueError, match="equilibrium"):
            CertaintyEquivalentController(_schedule(2), goal=[1.0, 0.5])

    def test_its_sampled_cost_through_the_harness_matches_its_expected_cost(self):
        # The world emits a reading after each action and the filter folds it before
        # the next: the pattern the closed form assumes, driven for real.
        plan = _schedule(5)
        model = plan.model
        controller = CertaintyEquivalentController(plan, goal=GOAL)
        qc, rc = np.asarray(GOAL_PRECISION), np.asarray(EFFORT_PENALTY)
        rng = np.random.default_rng(2)
        keys = jax.random.split(jax.random.PRNGKey(3), 150)
        costs = []
        for key in keys:
            world = World(
                model, initial_state=rng.multivariate_normal(np.zeros(2), np.eye(2))
            )
            backend, belief, cost = KalmanBackend(model), model.prior, 0.0
            for step, step_key in enumerate(jax.random.split(key, 5)):
                action = controller.action(step, belief.mean)
                reading = world.step(action, step_key)
                deviation = np.asarray(world.state) - GOAL
                cost += deviation @ qc @ deviation + float(action @ rc @ action)
                belief = backend.infer_states(reading, belief, action)
            costs.append(cost)
        costs = np.asarray(costs)
        standard_error = costs.std(ddof=1) / np.sqrt(len(costs))
        assert abs(controller.expected_cost() - costs.mean()) < 4 * standard_error


# --- the optimum by separation, and the bracket it closes -----------------------------


EPS = np.finfo(float).eps


def _point_mass_schedule(horizon, *, observation_noise=1e-2, prior_cov=None):
    model = LinearGaussianModel(
        dynamics_matrix=DYNAMICS,
        observation_matrix=OBSERVATION,
        dynamics_noise=PROCESS_NOISE,
        observation_noise=[[observation_noise]],
        prior=Belief(
            mean=[0.0, 0.0], cov=np.eye(2) if prior_cov is None else prior_cov
        ),
        control_matrix=CONTROL,
    )
    return finite_horizon_lqr(
        model,
        goal_precision=GOAL_PRECISION,
        effort_penalty=EFFORT_PENALTY,
        horizon=horizon,
    )


def _exact_filter_cost(plan, prior=None):
    gains = kalman_schedule(plan.model, plan.horizon, prior=prior).gains
    return closed_loop_cost(plan, gains, goal=GOAL, prior=prior)


class TestOptimalCost:
    @pytest.mark.parametrize("horizon", [1, 2, 5, 12])
    def test_agrees_with_the_moment_propagated_cost_to_machine_precision(self, horizon):
        # Two closed forms that share nothing past the Riccati recursions. Their
        # agreement is the whole of the fixed-noise signature.
        plan = _schedule(horizon)
        by_separation = optimal_cost(plan, goal=GOAL)
        by_moments = _exact_filter_cost(plan)
        assert abs(by_separation - by_moments) <= 64 * EPS * by_moments

    def test_is_the_floor_plus_the_estimate_penalty_by_hand(self):
        horizon = 4
        plan = _schedule(horizon)
        b, rc = np.asarray(CONTROL), np.asarray(EFFORT_PENALTY)
        _, covs = _numpy_kalman_covariances(np.eye(2), horizon)
        penalty = 0.0
        for k in range(horizon):
            remaining = np.asarray(GOAL_PRECISION) + np.asarray(
                plan.cost_to_go[horizon - k - 1]
            )
            gain = np.asarray(plan.gains[k])
            penalty += np.trace(gain.T @ (rc + b.T @ remaining @ b) @ gain @ covs[k])
        expected = full_information_cost(plan, goal=GOAL) + penalty
        assert optimal_cost(plan, goal=GOAL) == pytest.approx(expected, rel=1e-13)

    def test_a_worse_sensor_costs_more_and_a_known_state_costs_the_floor(self):
        sharp = _point_mass_schedule(5, observation_noise=1e-4)
        blunt = _point_mass_schedule(5, observation_noise=1.0)
        floor = full_information_cost(sharp, goal=GOAL)
        assert floor < optimal_cost(sharp, goal=GOAL) < optimal_cost(blunt, goal=GOAL)

    def test_reads_the_prior_it_is_given(self):
        plan = _schedule(3)
        prior = Belief(mean=[0.3, -0.2], cov=[[2.0, 0.1], [0.1, 0.5]])
        by_moments = _exact_filter_cost(plan, prior=prior)
        assert optimal_cost(plan, goal=GOAL, prior=prior) == pytest.approx(
            by_moments, rel=1e-13
        )
        assert optimal_cost(plan, goal=GOAL) != pytest.approx(by_moments, rel=1e-6)


class TestControlBracket:
    def test_its_ends_are_the_two_closed_forms(self):
        plan = _schedule(5)
        bracket = control_bracket(plan, goal=GOAL)
        assert bracket.floor == full_information_cost(plan, goal=GOAL)
        assert bracket.ceiling == _exact_filter_cost(plan)

    def test_its_width_is_the_price_of_inference(self):
        plan = _schedule(5)
        bracket = control_bracket(plan, goal=GOAL)
        assert bracket.width == bracket.ceiling - bracket.floor
        assert bracket.width > 0.0
        penalty = optimal_cost(plan, goal=GOAL) - full_information_cost(plan, goal=GOAL)
        assert bracket.width == pytest.approx(penalty, abs=64 * EPS * bracket.ceiling)

    def test_its_width_does_not_move_with_where_the_state_starts(self):
        # The offset from the goal is charged at both ends and cancels: only the
        # spread, which the filter has to work down, sets the price of inference.
        plan = _schedule(5)
        near = control_bracket(plan, goal=GOAL)
        far = control_bracket(
            plan, goal=GOAL, prior=Belief(mean=[5.0, -3.0], cov=np.eye(2))
        )
        assert far.ceiling > near.ceiling
        assert far.width == pytest.approx(near.width, abs=64 * EPS * far.ceiling)

    def test_its_width_grows_with_the_prior_spread(self):
        plan = _schedule(5)
        tight = control_bracket(plan, goal=GOAL)
        loose = control_bracket(
            plan, goal=GOAL, prior=Belief(mean=[0.0, 0.0], cov=4.0 * np.eye(2))
        )
        assert loose.width > tight.width

    def test_refuses_what_its_two_closed_forms_refuse(self):
        with pytest.raises(ValueError, match="equilibrium"):
            control_bracket(_schedule(2), goal=[1.0, 0.5])
        with pytest.raises(ValueError, match="prior"):
            control_bracket(
                _schedule(2), goal=GOAL, prior=Belief(mean=[0.0], cov=[[1.0]])
            )


class TestControlEfficiency:
    def test_a_certainty_equivalent_agent_reads_zero_exactly(self):
        bracket = control_bracket(_schedule(5), goal=GOAL)
        eta = control_efficiency(bracket, Bounded(value=bracket.ceiling, bar=EXACT))
        assert eta.value == 0.0
        assert eta.bar == EXACT

    def test_an_agent_that_knows_the_state_reads_one(self):
        bracket = control_bracket(_schedule(5), goal=GOAL)
        eta = control_efficiency(bracket, Bounded(value=bracket.floor, bar=EXACT))
        assert eta.value == pytest.approx(1.0, abs=64 * EPS)

    def test_positions_the_agent_within_the_bracket_and_scales_its_bar(self):
        bracket = control_bracket(_schedule(5), goal=GOAL)
        agent = Bounded(
            value=bracket.ceiling - 0.3 * bracket.width,
            bar=Bar(common_mode=0.02, own=0.05),
        )
        eta = control_efficiency(bracket, agent)
        assert eta.value == pytest.approx(0.3, abs=1e-12)
        assert eta.bar.common_mode == pytest.approx(-0.02 / bracket.width)
        assert eta.bar.own == pytest.approx(0.05 / bracket.width)

    def test_a_sampled_certainty_equivalent_run_reads_zero_within_its_floor(self):
        # The stated floor is the sampled bar over the width; zero is claimed to it.
        plan = _schedule(5)
        bracket = control_bracket(plan, goal=GOAL)
        gains = kalman_schedule(plan.model, 5).gains
        mean, standard_error = _sampled_closed_loop_cost(
            plan.gains, gains, GOAL, draws=20_000
        )
        eta = control_efficiency(
            bracket,
            Bounded(value=mean, bar=Bar(common_mode=0.0, own=4 * standard_error)),
        )
        assert abs(eta.value) <= eta.bar.total
        assert eta.bar.total < 0.5  # a floor the sampled bar cannot make vacuous

    def test_refuses_a_bracket_of_no_width(self):
        with pytest.raises(ValueError, match="width"):
            control_efficiency(
                ControlBracket(floor=1.0, ceiling=1.0),
                Bounded(value=1.0, bar=EXACT),
            )
