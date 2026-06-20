import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp.agent import Agent
from cpomdp.control import LQRController
from cpomdp.observation import CallableSensor
from cpomdp.selection import EFESelector, LQRSelector, ObservationGoal, StateGoal
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
        dynamics=DYNAMICS,
        sensor_model=SENSOR_MODEL,
        dynamics_noise=[[1e-6, 0.0], [0.0, 1e-6]],
        sensor_noise=[[1e-2]],
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        control=CONTROL,
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
        dynamics=DYNAMICS_2D,
        sensor_model=SENSOR_MODEL_2D,
        dynamics_noise=np.eye(4) * 1e-6,
        sensor_noise=np.eye(2) * 1e-2,
        prior=Belief(mean=[0.0, 0.0, 0.0, 0.0], cov=np.eye(4)),
        control=CONTROL_2D,
    )


def _run_closed_loop(agent, true_state, steps, rng=None):
    """Drive the true plant with the agent's own actions; return the final state.

    Each step is the full perceive -> act -> advance cycle. The agent sees only
    `obs` (a noisy position reading), never `true_state` directly. If `rng` is
    given, sensor and process noise are injected — otherwise the plant is clean.

    The plant matrices are read off the agent's own model, so this drives a
    p=1 single-force point mass and a p=2 multi-actuator plant identically.
    """
    A, B, C = agent.model.dynamics, agent.model.control, agent.model.sensor_model
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
        dynamics=[[1.0, DT], [0.0, 1.0]],
        sensor_model=SENSOR_MODEL,
        dynamics_noise=[[1e-6, 0.0], [0.0, 1e-6]],
        sensor_noise=[[1e-2]],
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        # no control= → no action channel
    )
    agent = Agent(model)  # no goal

    # perceive works: folding in an observation returns a Belief and sharpens it
    belief = agent.infer_states([0.5])

    assert belief.cov[0, 0] < model.prior.cov[0, 0]  # uncertainty dropped

    # but it cannot act — no objective to steer toward
    with pytest.raises(ValueError, match="objective"):
        agent.sample_action()


# ============================================================================
# Phase 5: regime dispatch — the sensor type picks the selector.
#
#   fixed sensor    + state goal=     -> LQRSelector  (state tracking, v0.2 path)
#   callable sensor + obs preference= -> EFESelector  (information-seeking, v0.3)
#
# An explicit selector= overrides the dispatch verbatim. Every other quadrant
# raises: a state goal on a callable sensor would need converting through C; an
# obs preference on a fixed sensor is output regulation (deferred). The model is
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


def _corridor_model(*, fixed, control=True):
    """1-D single integrator; fixed or precision-well sensor; optional control."""
    sensor = (
        None
        if fixed
        else CallableSensor(
            sensor_model=[[1.0]], noise_fn=_well_noise, noise_params=_WELL_PARAMS
        )
    )
    return LinearGaussianModel(
        dynamics=[[1.0]],
        sensor_model=[[1.0]],
        dynamics_noise=[[0.05]],
        sensor_noise=[[0.3]],
        prior=Belief(mean=[0.0], cov=[[0.5]]),
        control=[[1.0]] if control else None,
        observation=sensor,
    )


_OBS_GOAL = ObservationGoal([0.0], (-3.0, 3.0), precision=[[0.4]])  # observe position 0
_BOUNDS = (-3.0, 3.0)


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
        assert isinstance(agent._selector, LQRSelector)  # white-box: dispatch wiring

    def test_observation_goal_on_callable_sensor_uses_efe_selector(self):
        agent = Agent(_corridor_model(fixed=False), _OBS_GOAL)
        assert isinstance(agent._selector, EFESelector)

    def test_explicit_selector_is_used_verbatim(self):
        # An explicit selector= overrides the dispatched selector, used as given.
        sel = EFESelector(
            _corridor_model(fixed=False), n_candidates=11, action_bounds=_BOUNDS
        )
        agent = Agent(_corridor_model(fixed=False), _OBS_GOAL, selector=sel)
        assert agent._selector is sel

    def test_explicit_selector_rescues_fixed_sensor_observation_goal(self):
        # The escape hatch: an ObservationGoal on a fixed sensor raises under auto-
        # dispatch (output regulation, deferred), but an explicit selector= opts
        # into H=1 EFE on a fixed sensor — the collapse regime EFESelector tests use.
        sel = EFESelector(
            _corridor_model(fixed=True), n_candidates=11, action_bounds=_BOUNDS
        )
        agent = Agent(_corridor_model(fixed=True), _OBS_GOAL, selector=sel)
        assert agent._selector is sel

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
            Agent(_corridor_model(fixed=False, control=False), _OBS_GOAL)

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
                _corridor_model(fixed=False), ObservationGoal([0.0, 0.0], (-3.0, 3.0))
            )

    def test_unknown_objective_type_raises(self):
        # The objective must be a StateGoal or ObservationGoal — nothing else.
        with pytest.raises(TypeError, match=r"StateGoal|ObservationGoal|objective"):
            Agent(_reaching_model(), "reach the goal")  # ty: ignore[invalid-argument-type]


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
    a_mat = np.asarray(model.dynamics)
    b_mat = None if model.control is None else np.asarray(model.control)
    c_mat = np.asarray(model.sensor_model)
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
    a_mat = np.asarray(model.dynamics)
    c_mat = np.asarray(model.sensor_model)
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
        ObservationGoal([0.0], (-3.0, 3.0), precision=[[0.4]], n_candidates=61),
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
