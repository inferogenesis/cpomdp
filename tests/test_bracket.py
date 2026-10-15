"""The control bracket, `cpomdp._lab.bracket`: the costs a controller sits between."""

import jax
import numpy as np
import pytest

from _support import (
    CONTROL,
    DYNAMICS,
    EFFORT_PENALTY,
    GOAL_PRECISION,
    OBSERVATION,
    OBSERVATION_NOISE,
    PROCESS_NOISE,
    brute_force_plan,
    lqr_schedule,
    numpy_kalman_covariances,
    point_mass_model,
)
from cpomdp._lab.bracket import (
    CertaintyEquivalentController,
    ControlBracket,
    closed_loop_cost,
    control_bracket,
    control_efficiency,
    full_information_cost,
    optimal_cost,
)
from cpomdp._lab.harness import World
from cpomdp._lab.resolution import EXACT, Bar, Bounded
from cpomdp.backends.kalman import KalmanBackend
from cpomdp.control import LQRController, finite_horizon_lqr, kalman_schedule
from cpomdp.types import Belief, LinearGaussianModel

# The samplers below keep the terse letters a/b/c/qc/rc of the textbook recursions.
# They are local scalars, not the role-named API, which spells them out to avoid the
# Q/R collision (ADR-003).


# --- the full-information floor -------------------------------------------------------


GOAL = np.array([1.0, 0.0])  # an equilibrium: zero velocity holds the position


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
        model = point_mass_model()
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
        schedule = lqr_schedule(6)
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
        _, expected = brute_force_plan(
            DYNAMICS, CONTROL, GOAL_PRECISION, EFFORT_PENALTY, 5, x0
        )
        known = Belief(mean=x0, cov=np.zeros((2, 2)))
        assert full_information_cost(
            schedule, goal=np.zeros(2), prior=known
        ) == pytest.approx(expected, rel=1e-12)

    def test_the_prior_covariance_adds_its_trace_against_the_terminal_cost(self):
        schedule = lqr_schedule(4)
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
        schedule = lqr_schedule(4)
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
        schedule = lqr_schedule(4)
        assert full_information_cost(schedule, goal=GOAL) == full_information_cost(
            schedule, goal=GOAL, prior=schedule.model.prior
        )

    def test_a_goal_the_dynamics_cannot_hold_is_refused(self):
        # A moving target: a nonzero velocity is not an equilibrium, so the shifted
        # problem is not a regulator and the closed form does not describe it.
        with pytest.raises(ValueError, match="equilibrium"):
            full_information_cost(lqr_schedule(3), goal=[1.0, 0.5])

    def test_a_goal_of_the_wrong_shape_is_refused(self):
        with pytest.raises(ValueError, match="goal"):
            full_information_cost(lqr_schedule(3), goal=[1.0])

    def test_a_prior_over_another_state_is_refused(self):
        with pytest.raises(ValueError, match="prior"):
            full_information_cost(
                lqr_schedule(3), goal=GOAL, prior=Belief(mean=[0.0], cov=[[1.0]])
            )


# --- the certainty-equivalent controller and its closed-loop cost ---------------------


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


class TestClosedLoopCost:
    def test_one_step_by_hand(self):
        # With one step the filter never acts: the action comes from the prior mean,
        # the state is charged where it lands, and the noise adds its trace.
        plan = lqr_schedule(1)
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
        plan = lqr_schedule(6)
        filter_gains, _ = numpy_kalman_covariances(np.eye(2), 6)
        sampled, standard_error = _sampled_closed_loop_cost(
            np.asarray(plan.gains), filter_gains, GOAL, draws=20_000
        )
        closed = closed_loop_cost(plan, filter_gains, goal=GOAL)
        assert abs(closed - sampled) < 4 * standard_error

    def test_the_certainty_equivalent_cost_sits_above_the_floor(self):
        plan = lqr_schedule(6)
        controller = CertaintyEquivalentController(plan, goal=GOAL)
        assert controller.expected_cost() > full_information_cost(plan, goal=GOAL)

    def test_the_steady_state_gain_costs_more_than_the_matched_schedule(self):
        # The mismatch from unit one, priced: applying the forever gain at every
        # step of a short plan is not optimal for that plan.
        plan = lqr_schedule(3)
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
        plan = lqr_schedule(3)
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
        plan = lqr_schedule(4)
        controller = CertaintyEquivalentController(plan, goal=GOAL)
        mean = np.array([0.4, -0.1])
        for step in range(4):
            np.testing.assert_array_equal(
                controller.action(step, mean), -plan.gains[step] @ (mean - GOAL)
            )

    def test_refuses_a_goal_the_dynamics_cannot_hold(self):
        with pytest.raises(ValueError, match="equilibrium"):
            CertaintyEquivalentController(lqr_schedule(2), goal=[1.0, 0.5])

    def test_its_sampled_cost_through_the_harness_matches_its_expected_cost(self):
        # The world emits a reading after each action and the filter folds it before
        # the next: the pattern the closed form assumes, driven for real.
        plan = lqr_schedule(5)
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
        plan = lqr_schedule(horizon)
        by_separation = optimal_cost(plan, goal=GOAL)
        by_moments = _exact_filter_cost(plan)
        assert abs(by_separation - by_moments) <= 64 * EPS * by_moments

    def test_is_the_floor_plus_the_estimate_penalty_by_hand(self):
        horizon = 4
        plan = lqr_schedule(horizon)
        b, rc = np.asarray(CONTROL), np.asarray(EFFORT_PENALTY)
        _, covs = numpy_kalman_covariances(np.eye(2), horizon)
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
        plan = lqr_schedule(3)
        prior = Belief(mean=[0.3, -0.2], cov=[[2.0, 0.1], [0.1, 0.5]])
        by_moments = _exact_filter_cost(plan, prior=prior)
        assert optimal_cost(plan, goal=GOAL, prior=prior) == pytest.approx(
            by_moments, rel=1e-13
        )
        assert optimal_cost(plan, goal=GOAL) != pytest.approx(by_moments, rel=1e-6)


class TestControlBracket:
    def test_its_ends_are_the_two_closed_forms(self):
        plan = lqr_schedule(5)
        bracket = control_bracket(plan, goal=GOAL)
        assert bracket.floor == full_information_cost(plan, goal=GOAL)
        assert bracket.ceiling == _exact_filter_cost(plan)

    def test_its_width_is_the_price_of_inference(self):
        plan = lqr_schedule(5)
        bracket = control_bracket(plan, goal=GOAL)
        assert bracket.width == bracket.ceiling - bracket.floor
        assert bracket.width > 0.0
        penalty = optimal_cost(plan, goal=GOAL) - full_information_cost(plan, goal=GOAL)
        assert bracket.width == pytest.approx(penalty, abs=64 * EPS * bracket.ceiling)

    def test_its_width_does_not_move_with_where_the_state_starts(self):
        # The offset from the goal is charged at both ends and cancels: only the
        # spread, which the filter has to work down, sets the price of inference.
        plan = lqr_schedule(5)
        near = control_bracket(plan, goal=GOAL)
        far = control_bracket(
            plan, goal=GOAL, prior=Belief(mean=[5.0, -3.0], cov=np.eye(2))
        )
        assert far.ceiling > near.ceiling
        assert far.width == pytest.approx(near.width, abs=64 * EPS * far.ceiling)

    def test_its_width_grows_with_the_prior_spread(self):
        plan = lqr_schedule(5)
        tight = control_bracket(plan, goal=GOAL)
        loose = control_bracket(
            plan, goal=GOAL, prior=Belief(mean=[0.0, 0.0], cov=4.0 * np.eye(2))
        )
        assert loose.width > tight.width

    def test_refuses_what_its_two_closed_forms_refuse(self):
        with pytest.raises(ValueError, match="equilibrium"):
            control_bracket(lqr_schedule(2), goal=[1.0, 0.5])
        with pytest.raises(ValueError, match="prior"):
            control_bracket(
                lqr_schedule(2), goal=GOAL, prior=Belief(mean=[0.0], cov=[[1.0]])
            )


class TestControlEfficiency:
    def test_a_certainty_equivalent_agent_reads_zero_exactly(self):
        bracket = control_bracket(lqr_schedule(5), goal=GOAL)
        eta = control_efficiency(bracket, Bounded(value=bracket.ceiling, bar=EXACT))
        assert eta.value == 0.0
        assert eta.bar == EXACT

    def test_an_agent_that_knows_the_state_reads_one(self):
        bracket = control_bracket(lqr_schedule(5), goal=GOAL)
        eta = control_efficiency(bracket, Bounded(value=bracket.floor, bar=EXACT))
        assert eta.value == pytest.approx(1.0, abs=64 * EPS)

    def test_positions_the_agent_within_the_bracket_and_scales_its_bar(self):
        bracket = control_bracket(lqr_schedule(5), goal=GOAL)
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
        plan = lqr_schedule(5)
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
