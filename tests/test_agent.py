import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp._lab.degraded import DiagonalCovarianceBackend, WrongFixedRBackend
from cpomdp._lab.harness import ScoredAgent
from cpomdp.agent import Agent
from cpomdp.backends.kalman import KalmanBackend
from cpomdp.control import LQRController
from cpomdp.ffg.chain import ChainBackend
from cpomdp.observation import CallableSensor
from cpomdp.selection import (
    EFESelector,
    LQRSelector,
    ObservationGoal,
    Preference,
    StateGoal,
)
from cpomdp.types import Belief, LinearGaussianModel

# The double-integrator point mass from test_control: state = [position, velocity],
# a force moves the velocity, velocity moves the position. Observe position only,
# so the filter must infer velocity through the off-diagonal coupling while the
# controller drives both — a real LQG loop, not a toy.
DT = 0.1
DYNAMICS = np.array([[1.0, DT], [0.0, 1.0]])
CONTROL = np.array([[0.0], [DT]])
SENSOR_MODEL = np.array([[1.0, 0.0]])
GOAL = np.array([1.0, 0.0])  # reach position 1, at rest


def _reaching_model() -> LinearGaussianModel:
    return LinearGaussianModel(
        dynamics_matrix=DYNAMICS,
        observation_matrix=SENSOR_MODEL,
        dynamics_noise=[[1e-6, 0.0], [0.0, 1e-6]],
        observation_noise=[[1e-2]],
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        control_matrix=CONTROL,
    )


# A 2D point mass: state = [px, py, vx, vy], with two independent force inputs
# (fx pushes vx, fy pushes vy) and position-only sensing. This is the p=2
# multi-actuator stress case: the controller gain is now (2, 4), so a
# transposed or axis-swapped gain — invisible at p=1 — would steer the wrong
# axis. The asymmetric goal (reach px=+1, py=-1) means a swap can't pass by
# coincidence: the two channels must stay distinct end-to-end.
DYNAMICS_2D = np.array(
    [
        [1.0, 0.0, DT, 0.0],
        [0.0, 1.0, 0.0, DT],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
CONTROL_2D = np.array([[0.0, 0.0], [0.0, 0.0], [DT, 0.0], [0.0, DT]])
SENSOR_MODEL_2D = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
GOAL_2D = np.array([1.0, -1.0, 0.0, 0.0])  # reach (+1, -1), at rest


def _reaching_model_2d() -> LinearGaussianModel:
    return LinearGaussianModel(
        dynamics_matrix=DYNAMICS_2D,
        observation_matrix=SENSOR_MODEL_2D,
        dynamics_noise=np.eye(4) * 1e-6,
        observation_noise=np.eye(2) * 1e-2,
        prior=Belief(mean=[0.0, 0.0, 0.0, 0.0], cov=np.eye(4)),
        control_matrix=CONTROL_2D,
    )


def _run_closed_loop(agent, true_state, steps, rng=None):
    """Drive the true plant with the agent's own actions; return the final state.

    Each step is the full perceive -> act -> advance cycle. The agent sees only
    `obs` (a noisy position reading), never `true_state` directly. If `rng` is
    given, sensor and process noise are injected — otherwise the plant is clean.

    The plant matrices are read off the agent's own model, so this drives a
    p=1 single-force point mass and a p=2 multi-actuator plant identically.
    """
    A, B, C = (
        agent.model.dynamics_matrix,
        agent.model.control_matrix,
        agent.model.observation_matrix,
    )
    true_state = np.asarray(true_state, dtype=float)
    for _ in range(steps):
        obs = C @ true_state
        if rng is not None:
            obs = obs + rng.normal(0.0, 0.1, size=obs.shape)
        agent.infer_states(obs)  # perceive
        action = agent.sample_action()  # act
        true_state = A @ true_state + B @ action  # advance the plant
        if rng is not None:
            true_state = true_state + rng.normal(0.0, 1e-3, size=true_state.shape)
    return true_state


def test_goal_is_equilibrium() -> None:
    np.testing.assert_allclose(DYNAMICS @ GOAL, GOAL)
    np.testing.assert_allclose(DYNAMICS_2D @ GOAL_2D, GOAL_2D)


def test_reaches_goal_noiseless() -> None:
    agent = Agent(_reaching_model(), StateGoal(GOAL))
    final_state = _run_closed_loop(
        agent, [0.0, 0.0], steps=200
    )  # <-- the helper returns it
    np.testing.assert_allclose(final_state, GOAL, atol=1e-2)
    np.testing.assert_allclose(
        agent.belief.mean, final_state, atol=1e-2
    )  # Check model thinks it is where it is


def test_reaches_goal_under_noise() -> None:
    agent = Agent(_reaching_model(), StateGoal(GOAL))
    final_state = _run_closed_loop(
        agent, [0.0, 0.0], steps=200, rng=np.random.default_rng(0)
    )
    np.testing.assert_allclose(agent.belief.mean, final_state, atol=0.1)


def test_reaches_goal_multi_actuator() -> None:
    # The end-to-end p=2 check: two independent actuators, position-only
    # sensing. Drives the full agent -> controller -> gain stack with a
    # non-scalar action and an asymmetric goal, so a mis-oriented (transposed
    # or axis-swapped) gain would steer the wrong axis and never converge.
    agent = Agent(_reaching_model_2d(), StateGoal(GOAL_2D))
    final_state = _run_closed_loop(agent, [0.0, 0.0, 0.0, 0.0], steps=300)

    assert agent.sample_action().shape == (2,)  # genuinely a 2-actuator action
    np.testing.assert_allclose(final_state, GOAL_2D, atol=1e-2)
    np.testing.assert_allclose(agent.belief.mean, final_state, atol=1e-2)


def test_perceive_only_agent_cannot_act():
    # A goal-less agent on a control-free model: a pure tracker.
    model = LinearGaussianModel(
        dynamics_matrix=[[1.0, DT], [0.0, 1.0]],
        observation_matrix=SENSOR_MODEL,
        dynamics_noise=[[1e-6, 0.0], [0.0, 1e-6]],
        observation_noise=[[1e-2]],
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        # no control_matrix= → no action channel
    )
    agent = Agent(model)  # no goal

    # perceive works: folding in an observation returns a Belief and sharpens it
    belief = agent.infer_states([0.5])

    assert belief.cov[0, 0] < model.prior.cov[0, 0]  # uncertainty dropped

    # but it cannot act — no objective to steer toward
    with pytest.raises(ValueError, match="objective"):
        agent.sample_action()


def test_a_model_and_a_backend_cannot_both_be_passed():
    # Two inputs could disagree, and the Agent would plan on one model and filter on
    # the other. Each front door takes one input, so there is nothing to disagree.
    model = _reaching_model()
    with pytest.raises(TypeError):
        Agent(model, StateGoal(GOAL), backend=KalmanBackend(model))  # ty: ignore[unknown-argument]


def test_from_backend_plans_under_the_backends_model():
    backend = KalmanBackend(_reaching_model(), steady_state=True)
    agent = Agent.from_backend(backend, StateGoal(GOAL))
    assert agent.model is backend.model
    assert agent._backend is backend  # white-box: the filter is the one passed


def test_model_is_read_only():
    # The selector and controller were built from the model, so swapping it would
    # leave them planning under the old one while agent.model reported the new.
    agent = Agent(_reaching_model(), StateGoal(GOAL))
    with pytest.raises(AttributeError):
        agent.model = _reaching_model()  # ty: ignore[invalid-assignment]


def test_belief_assignment_is_validated():
    agent = Agent(_reaching_model(), StateGoal(GOAL))
    with pytest.raises(TypeError, match="Belief"):
        agent.belief = [1.0, 2.0]  # ty: ignore[invalid-assignment]
    with pytest.raises(ValueError, match="2-D state"):
        agent.belief = Belief(mean=[0.0], cov=[[1.0]])


def test_belief_assignment_restarts_filtering():
    agent = Agent(_reaching_model(), StateGoal(GOAL))
    restart = Belief(mean=[0.5, 0.0], cov=np.eye(2))
    agent.belief = restart
    assert agent.belief is restart
    assert agent.infer_states([0.5]).mean[0] == pytest.approx(0.5, abs=1e-3)


def test_kalman_backend_model_is_read_only():
    backend = KalmanBackend(_reaching_model())
    with pytest.raises(AttributeError):
        backend.model = _reaching_model()  # ty: ignore[invalid-assignment]
    with pytest.raises(AttributeError):
        backend.steady_state = True  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    "build",
    [
        ScoredAgent,
        lambda m: LQRController(m, goal_precision=np.eye(2), effort_penalty=np.eye(1)),
        ChainBackend,
        lambda m: WrongFixedRBackend(m, magnitude=0.5),
        lambda m: DiagonalCovarianceBackend(KalmanBackend(m)),
    ],
    ids=["ScoredAgent", "LQRController", "ChainBackend", "WrongFixedR", "Diagonal"],
)
def test_every_holder_of_a_model_keeps_it_read_only(build):
    # Each builds a filter, a gain or a factor from the model at construction.
    holder = build(_reaching_model())
    with pytest.raises(AttributeError):
        holder.model = _reaching_model()


def _damped_model() -> LinearGaussianModel:
    # A = 0.9 holds only the origin still: any other state decays toward it.
    return LinearGaussianModel(
        dynamics_matrix=[[0.9]],
        observation_matrix=[[1.0]],
        dynamics_noise=[[1e-6]],
        observation_noise=[[1e-2]],
        prior=Belief(mean=[0.0], cov=[[1.0]]),
        control_matrix=[[1.0]],
    )


def test_state_goal_off_equilibrium_is_refused():
    # Unrefused, the regulator settles at x = 0.8432 for a target of 1.
    with pytest.raises(ValueError, match="equilibrium"):
        Agent(_damped_model(), StateGoal([1.0]))


def test_state_goal_at_equilibrium_is_accepted():
    agent = Agent(_damped_model(), StateGoal([0.0]))
    assert agent.sample_action().shape == (1,)


# ============================================================================
# Phase 5: regime dispatch — the sensor type picks the selector.
#
#   fixed sensor    + state goal=     -> LQRSelector  (state tracking, v0.2 path)
#   callable sensor + obs preference= -> EFESelector  (information-seeking, v0.3)
#
# A Preference with selector= skips the dispatch and uses that selector. Every other
# quadrant raises: a state goal on a callable sensor would need converting through C;
# an obs preference on a fixed sensor is output regulation (deferred). The model is
# the 1-D single-integrator corridor the EFESelector tests use (μ⁺ = μ + a,
# observe position) with a precision-well sensor sharp at a beacon.
# ============================================================================


def _well_noise(x, params):
    """R(x): sharp (low) near the beacon, foggy (high) away. Module-level (jit-safe)."""
    pos = x[0]
    falloff = 1.0 - jnp.exp(
        -((pos - params["beacon"]) ** 2) / (2.0 * params["width"] ** 2)
    )
    return jnp.array([[params["r_lo"] + (params["r_hi"] - params["r_lo"]) * falloff]])


_WELL_PARAMS = {
    "beacon": jnp.array(1.5),
    "width": jnp.array(0.6),
    "r_lo": jnp.array(0.05),
    "r_hi": jnp.array(0.8),
}


def _corridor_model(*, fixed, control_matrix=True):
    """1-D single integrator; fixed or precision-well sensor; optional control."""
    sensor = (
        None
        if fixed
        else CallableSensor(
            observation_matrix=[[1.0]], noise_fn=_well_noise, noise_params=_WELL_PARAMS
        )
    )
    return LinearGaussianModel(
        dynamics_matrix=[[1.0]],
        observation_matrix=[[1.0]],
        dynamics_noise=[[0.05]],
        observation_noise=[[0.3]],
        prior=Belief(mean=[0.0], cov=[[0.5]]),
        control_matrix=[[1.0]] if control_matrix else None,
        observation_model=sensor,
    )


_OBS_GOAL = ObservationGoal(
    [0.0], action_bounds=(-3.0, 3.0), precision=[[0.4]]
)  # observe position 0
_OBS_PREFERENCE = Preference(
    [0.0], [[0.4]]
)  # the same wish, for a selector of one's own
_BOUNDS = (-3.0, 3.0)


class _Hold:
    """A custom selector the Agent cannot inspect: it always rests."""

    def select(self, belief, preference):
        return jnp.zeros(1)


class TestRegimeDispatch:
    """The Agent wires the selector that matches the objective's domain (Phase 5).

    The objective's *type* is the dispatch key: a StateGoal goes to the LQR path
    (state space), an ObservationGoal to the EFE path (observation space). The
    illegal-state smell is gone by construction — there is one objective slot, and
    the two variants can't be conflated — so the only guards left are genuine
    objective/model compatibility checks.
    """

    def test_state_goal_on_fixed_sensor_uses_lqr_selector(self):
        agent = Agent(_reaching_model(), StateGoal(GOAL))
        assert isinstance(agent.selector, LQRSelector)

    def test_observation_goal_on_callable_sensor_uses_efe_selector(self):
        agent = Agent(_corridor_model(fixed=False), _OBS_GOAL)
        assert isinstance(agent.selector, EFESelector)

    def test_explicit_selector_is_used_verbatim(self):
        # selector= takes a Preference, and the selector it is given is the one used.
        model = _corridor_model(fixed=False)
        sel = EFESelector(model, n_candidates=11, action_bounds=_BOUNDS)
        agent = Agent(model, _OBS_PREFERENCE, selector=sel)
        assert agent.selector is sel
        agent.infer_states([0.1])
        np.testing.assert_array_equal(
            agent.sample_action(), sel.select(agent.belief, _OBS_PREFERENCE)
        )

    def test_explicit_selector_chooses_by_efe_on_a_fixed_sensor(self):
        # The Agent declines an ObservationGoal on a fixed sensor. A selector built by
        # hand chooses by EFE there anyway, the collapse regime EFESelector tests use.
        model = _corridor_model(fixed=True)
        sel = EFESelector(model, n_candidates=11, action_bounds=_BOUNDS)
        agent = Agent(model, _OBS_PREFERENCE, selector=sel)
        assert agent.selector is sel

    def test_a_goal_beside_a_selector_is_refused(self):
        # The goal's box went unread: declared (-0.1, 0.1), the agent returned 1.5.
        model = _corridor_model(fixed=False)
        narrow = ObservationGoal([1.5], action_bounds=(-0.1, 0.1))
        sel = EFESelector(model, n_candidates=7, action_bounds=_BOUNDS)
        with pytest.raises(TypeError, match="Preference"):
            Agent(model, narrow, selector=sel)

    def test_a_state_goal_beside_a_selector_is_refused_before_any_riccati_solve(self):
        # No stabilising gain exists here, and the Agent used to solve for one it then
        # discarded, raising RuntimeError before it looked at the selector.
        model = LinearGaussianModel(
            dynamics_matrix=np.eye(2),
            observation_matrix=[[1.0, 0.0]],
            dynamics_noise=np.eye(2) * 1e-6,
            observation_noise=[[1e-2]],
            prior=Belief(mean=[0.0, 0.0], cov=np.eye(2)),
            control_matrix=[[0.0], [1.0]],
        )
        with pytest.raises(TypeError, match="Preference"):
            Agent(model, StateGoal([0.0, 0.0]), selector=_Hold())

    def test_a_selector_without_an_objective_is_refused(self):
        # It used to be dropped, leaving a tracker whose sample_action never named it.
        model = _corridor_model(fixed=False)
        sel = EFESelector(model, n_candidates=11, action_bounds=_BOUNDS)
        with pytest.raises(TypeError, match="Preference"):
            Agent(model, selector=sel)

    def test_a_preference_without_a_selector_is_refused(self):
        with pytest.raises(TypeError, match="selector="):
            Agent(_corridor_model(fixed=False), _OBS_PREFERENCE)

    def test_from_backend_takes_the_same_two_forms(self):
        model = _corridor_model(fixed=False)
        sel = EFESelector(model, n_candidates=11, action_bounds=_BOUNDS)
        agent = Agent.from_backend(KalmanBackend(model), _OBS_PREFERENCE, selector=sel)
        assert agent.selector is sel
        with pytest.raises(TypeError, match="Preference"):
            Agent.from_backend(KalmanBackend(model), _OBS_GOAL, selector=sel)

    def test_a_selector_needs_a_model_that_can_act(self):
        model = _corridor_model(fixed=False, control_matrix=False)
        with pytest.raises(ValueError, match="control matrix"):
            Agent(model, _OBS_PREFERENCE, selector=_Hold())

    def test_efe_agent_completes_a_perceive_act_cycle(self):
        # The EFE path wires end-to-end: perceive (Kalman) then act (EFE selection).
        agent = Agent(_corridor_model(fixed=False), _OBS_GOAL)
        agent.infer_states([0.1])
        action = agent.sample_action()
        assert action.shape == (1,)

    # --- compatibility guards: legitimate objective/model mismatches raise ---

    def test_state_goal_on_callable_sensor_raises(self):
        # A state goal on a state-dependent sensor would need converting through C;
        # raise instead of silently converting — use an ObservationGoal here.
        with pytest.raises(ValueError, match=r"ObservationGoal|observation"):
            Agent(_corridor_model(fixed=False), StateGoal([0.0]))

    def test_observation_goal_without_control_raises(self):
        with pytest.raises(ValueError, match="control"):
            Agent(_corridor_model(fixed=False, control_matrix=False), _OBS_GOAL)

    def test_observation_goal_on_fixed_sensor_raises(self):
        # Output regulation (obs goal on a fixed sensor) is deferred; raise rather
        # than hand back a myopic H=1 selector when LQR is available.
        with pytest.raises(ValueError, match=r"regulation|selector|StateGoal"):
            Agent(_corridor_model(fixed=True), _OBS_GOAL)

    def test_observation_goal_wrong_obs_dim_raises(self):
        # A 2-D obs goal on a model that observes m=1 must fail at construction, not
        # crash cryptically deep in the EFE kernel at sample_action.
        with pytest.raises(ValueError, match=r"observation dimension|length 1"):
            Agent(
                _corridor_model(fixed=False),
                ObservationGoal([0.0, 0.0], action_bounds=(-3.0, 3.0)),
            )

    def test_a_preference_of_the_wrong_length_for_an_efe_selector_is_refused(self):
        # EFESelector reads the preference as an observation, here of length 1. A
        # state-length target used to fail inside the first select's dot_general.
        model = _reaching_model()
        sel = EFESelector(model, n_candidates=11, action_bounds=_BOUNDS, horizon=2)
        with pytest.raises(ValueError, match="observation, a vector of length 1"):
            Agent(model, Preference(GOAL), selector=sel)

    def test_a_preference_of_the_wrong_length_for_lqr_is_refused(self):
        model = _reaching_model()
        sel = LQRSelector(
            LQRController(model, goal_precision=jnp.eye(2), effort_penalty=jnp.eye(1))
        )
        with pytest.raises(ValueError, match="state, a vector of length 2"):
            Agent(model, Preference([1.0]), selector=sel)

    def test_lqr_selector_takes_a_state_preference(self):
        model = _reaching_model()
        controller = LQRController(
            model, goal_precision=jnp.eye(2), effort_penalty=jnp.eye(1)
        )
        agent = Agent(model, Preference(GOAL), selector=LQRSelector(controller))
        np.testing.assert_array_equal(
            agent.sample_action(), controller.action(model.prior.mean, GOAL)
        )

    def test_qs_aliases_belief(self):
        # pymdp muscle-memory: agent.qs is a read-only alias of agent.belief.
        agent = Agent(_corridor_model(fixed=False), _OBS_GOAL)
        assert agent.qs is agent.belief
        agent.infer_states([0.5])
        assert agent.qs is agent.belief  # follows the reassignment

    def test_unknown_objective_type_raises(self):
        # The objective must be a StateGoal or ObservationGoal — nothing else.
        with pytest.raises(TypeError, match=r"StateGoal|ObservationGoal|objective"):
            Agent(_reaching_model(), "reach the goal")  # ty: ignore[invalid-argument-type]


# --- the horizon an ObservationGoal is searched over --------------------------------
#
# On the double integrator a push changes velocity, and velocity reaches the observed
# position a step later. At horizon 1 every candidate scores the same G, and the
# argmin's first candidate, a full push away from the goal, won the tie. A noise that
# reads velocity breaks the tie at horizon 1 without the target, so the scores moving
# is not enough. The target has to move them.


def _sharp_near_one(x, params):
    """R(x) sharpest at position 1, so the noise reads only what the sensor reads."""
    return jnp.array([[0.01 + (x[0] - 1.0) ** 2]])


def _sharp_at_rest(x, params):
    """R(x) sharpest at zero velocity, a state a push changes at once."""
    return jnp.array([[0.01 + x[1] ** 2]])


def _double_integrator(noise_fn, dynamics=DYNAMICS):
    return LinearGaussianModel(
        dynamics_matrix=dynamics,
        observation_matrix=SENSOR_MODEL,
        dynamics_noise=np.eye(2) * 1e-6,
        prior=Belief(mean=[0.0, 0.0], cov=np.eye(2)),
        control_matrix=CONTROL,
        observation_model=CallableSensor(SENSOR_MODEL, noise_fn, ()),
    )


_REACH_ONE = ObservationGoal([1.0], action_bounds=(-2.0, 2.0))


def _horizon(agent):
    """The horizon of the EFE selector the Agent built."""
    assert isinstance(agent.selector, EFESelector)
    return agent.selector.horizon


class TestDerivedHorizon:
    def test_a_push_two_steps_from_the_reading_is_searched_over_two(self):
        agent = Agent(_double_integrator(_sharp_near_one), _REACH_ONE)
        assert _horizon(agent) == 2
        agent.infer_states([0.0])
        assert float(agent.sample_action()[0]) > 0.0  # toward the goal at +1

    @pytest.mark.parametrize("noise_fn", [_sharp_near_one, _sharp_at_rest])
    def test_a_horizon_too_short_to_reach_the_reading_is_refused(self, noise_fn):
        goal = ObservationGoal([1.0], action_bounds=(-2.0, 2.0), horizon=1)
        with pytest.raises(ValueError, match="horizon=2"):
            Agent(_double_integrator(noise_fn), goal)

    def test_at_horizon_one_a_noise_that_reads_velocity_chooses_alone(self):
        # C·B = 0, so the reading cannot move in one step. The noise can.
        model = _double_integrator(_sharp_at_rest)
        selector = EFESelector(model, n_candidates=21, action_bounds=(-2.0, 2.0))
        up, down = (
            selector.select(model.prior, Preference([target])) for target in (1.0, -5.0)
        )
        np.testing.assert_array_equal(up, down)

    def _first_pushes(self, precision):
        pushes = []
        for target in (1.0, -5.0):
            goal = ObservationGoal(
                [target], action_bounds=(-2.0, 2.0), precision=precision
            )
            agent = Agent(_double_integrator(_sharp_at_rest), goal)
            assert _horizon(agent) == 2
            agent.infer_states([0.0])
            pushes.append(float(agent.sample_action()[0]))
        return pushes

    def test_noise_that_reads_velocity_does_not_stand_in_for_the_target(self):
        up, down = self._first_pushes(precision=[[100.0]])
        assert up > 0.0 > down

    def test_a_weak_preference_loses_to_the_information_term(self):
        # Read, the target still competes. At precision 1 the agent rests at the prior.
        up, down = self._first_pushes(precision=None)
        assert up == pytest.approx(0.0, abs=1e-12)
        assert down == pytest.approx(0.0, abs=1e-12)

    def test_a_single_integrator_keeps_horizon_one(self):
        agent = Agent(_corridor_model(fixed=False), _OBS_GOAL)
        assert _horizon(agent) == 1

    def test_an_action_that_never_reaches_the_reading_is_refused(self):
        # Velocity never feeds position.
        model = _double_integrator(_sharp_near_one, dynamics=np.eye(2))
        with pytest.raises(ValueError, match="within 2 steps"):
            Agent(model, _REACH_ONE)

    def test_a_refusal_names_the_horizon_it_scored(self):
        model = _double_integrator(_sharp_near_one, dynamics=np.eye(2))
        goal = ObservationGoal([1.0], action_bounds=(-2.0, 2.0), horizon=5)
        with pytest.raises(ValueError, match="at horizon 5,"):
            Agent(model, goal)


class TestSelectorAccess:
    def test_the_selector_is_read_only(self):
        agent = Agent(_corridor_model(fixed=False), _OBS_GOAL)
        with pytest.raises(AttributeError):
            agent.selector = None  # ty: ignore[invalid-assignment]

    def test_a_tracker_has_no_selector(self):
        tracker = Agent(_corridor_model(fixed=False, control_matrix=False))
        assert tracker.selector is None

    def test_printing_the_agent_shows_the_horizon_it_chose(self):
        agent = Agent(_double_integrator(_sharp_near_one), _REACH_ONE)
        assert repr(agent) == (
            "Agent(objective=ObservationGoal, selector=EFESelector(horizon=2, "
            "n_candidates=21, action_bounds=(-2.0, 2.0)), backend=KalmanBackend)"
        )

    def test_the_lqr_path_prints_the_gain_it_solved_for(self):
        printed = repr(Agent(_reaching_model(), StateGoal(GOAL)))
        assert printed.startswith(
            "Agent(objective=StateGoal, selector=LQRSelector(controller=LQRController("
            "gain=[["
        )
        assert printed.endswith(")), backend=KalmanBackend)")

    def test_a_selector_without_a_repr_prints_its_name(self):
        agent = Agent(_corridor_model(fixed=False), _OBS_PREFERENCE, selector=_Hold())
        assert repr(agent) == (
            "Agent(objective=Preference, selector=_Hold, backend=KalmanBackend)"
        )


def test_state_goal_path_is_byte_identical_to_lqr():
    # Regression (hard constraint): the typed-objective refactor must not perturb
    # the v0.2 LQR path by even a ULP. The sampled action equals a standalone
    # LQRController's for the same belief and goal, exactly (not allclose).
    model = _reaching_model()
    agent = Agent(model, StateGoal(GOAL))
    agent.infer_states([0.3])  # advance the belief off the prior
    action = agent.sample_action()
    expected = LQRController(
        model, goal_precision=jnp.eye(2), effort_penalty=jnp.eye(1)
    ).action(agent.belief.mean, jnp.asarray(GOAL, dtype=float))
    np.testing.assert_array_equal(action, expected)


# ============================================================================
# The v0.3 payoff: under state-dependent R(x), an information-seeking EFE agent
# drives its belief covariance far below an LQR baseline — by seeking the
# high-precision beacon the LQR agent can't perceive. The comparison is fair via
# a *covariance-only replay*: the Kalman cov recursion is observation-independent,
# so "the LQR path's uncertainty under the SAME R(x)" is fully determined by the
# LQR agent's predicted-mean (μ⁻) sequence — no noise, fully deterministic. This
# isolates "the path won, not the model".
# ============================================================================


def _run_recording(agent, start_pos, steps):
    """Noise-free closed loop recording the (μ⁻, cov, position) traces per step.

    Sibling of `_run_closed_loop` (which returns only the final state). The EFE leg
    needs the per-step belief covariance; the LQR leg needs the μ⁻ sequence the
    filter linearized R at, for the replay baseline.
    """
    model = agent.model
    a_mat = np.asarray(model.dynamics_matrix)
    b_mat = None if model.control_matrix is None else np.asarray(model.control_matrix)
    c_mat = np.asarray(model.observation_matrix)
    p = 0 if b_mat is None else b_mat.shape[1]
    true = np.asarray(start_pos, dtype=float)
    last_action = np.zeros(p)
    mu_minus, cov_trace, positions = [], [], []
    for _ in range(steps):
        mean = np.asarray(agent.belief.mean)
        control = b_mat @ last_action if b_mat is not None else 0.0
        mu_minus.append(a_mat @ mean + control)
        agent.infer_states(c_mat @ true)  # perceive (noise-free)
        cov_trace.append(np.asarray(agent.belief.cov))
        action = np.asarray(agent.sample_action())
        true = a_mat @ true + (b_mat @ action if b_mat is not None else 0.0)
        last_action = action
        positions.append(float(true[0]))
    return mu_minus, cov_trace, positions, np.asarray(agent.belief.mean)


def _replay_cov_trace(model, mu_minus, noise_fn, params):
    """Roll the Kalman covariance recursion along a μ⁻ sequence under R(x).

    Mirrors `_gain_and_posterior_cov` in NumPy (an independent oracle). The cov
    recursion never touches observation *values*, so this faithfully reconstructs
    the covariance a path would accumulate under the state-dependent noise.
    """
    a_mat = np.asarray(model.dynamics_matrix)
    c_mat = np.asarray(model.observation_matrix)
    q_mat = np.asarray(model.dynamics_noise)
    n = a_mat.shape[0]
    cov = np.asarray(model.prior.cov, dtype=float)
    trace = []
    for mu in mu_minus:
        cov_pred = a_mat @ cov @ a_mat.T + q_mat
        r_mat = np.asarray(noise_fn(jnp.asarray(mu), params), dtype=float)
        s = c_mat @ cov_pred @ c_mat.T + r_mat
        gain = cov_pred @ c_mat.T @ np.linalg.inv(s)
        cov = (np.eye(n) - gain @ c_mat) @ cov_pred
        trace.append(cov)
    return trace


def test_efe_agent_reaches_lower_uncertainty_than_lqr():
    # With a weak preference the epistemic drive dominates: the EFE agent seeks and
    # holds the precision beacon (it does NOT reach the observe-0 goal — that's the
    # information drive, not a bug), so under the SAME R(x) its belief ends far more
    # certain than the LQR baseline, which sits in the fog at its state goal.
    steps = 15
    efe = Agent(
        _corridor_model(fixed=False),
        ObservationGoal(
            [0.0], action_bounds=(-3.0, 3.0), precision=[[0.4]], n_candidates=61
        ),
    )
    _, efe_cov, efe_pos, _ = _run_recording(efe, [0.0], steps)

    lqr = Agent(_corridor_model(fixed=True), StateGoal([0.0]))
    lqr_mu, _, _, lqr_mean = _run_recording(lqr, [0.0], steps)
    lqr_cov = _replay_cov_trace(
        _corridor_model(fixed=False), lqr_mu, _well_noise, _WELL_PARAMS
    )

    efe_final = float(np.trace(efe_cov[-1]))
    lqr_final = float(np.trace(lqr_cov[-1]))
    assert efe_final < 0.5 * lqr_final  # the payoff: markedly more certain under R(x)
    assert max(efe_pos) > 1.0  # mechanism: it sought the beacon, off the goal
    assert abs(float(lqr_mean[0])) < 0.05  # the LQR baseline does reach its state goal
