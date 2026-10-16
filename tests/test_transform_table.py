"""The transform table on the backends page, pinned row by row.

A row that says a name traces is run here under ``jit``, ``vmap`` and ``grad`` against
its eager answer. A row that says a name does not trace is shown failing the way the
page says it fails, so a change in either direction lands with the table edited.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _support import unit_prior
from cpomdp import (
    Agent,
    Belief,
    CallableDynamicsNoise,
    CallableSensor,
    Coupling,
    CouplingGraph,
    CouplingGraphBackend,
    EFESelector,
    FFGEFESelector,
    FixedSensor,
    GaussianTransition,
    KalmanBackend,
    LinearGaussianModel,
    LQRController,
    LQRSelector,
    Preference,
    StateGoal,
    expected_free_energy,
    probe_model,
)
from cpomdp.efe import policy_efe
from cpomdp.enumeration import (
    EnumeratedEFESearch,
    FiniteActionSet,
    OpenLoopSelector,
    RecedingHorizonSelector,
)

OBSERVATIONS = jnp.array([[0.2], [0.5], [-0.3]])
ACTION = jnp.array([0.1])


def _sharp_near_one(x, params):
    return jnp.array([[0.01 + (x[0] - 1.0) ** 2]])


def _fixed_model(*, dynamics_noise=1e-6):
    return LinearGaussianModel(
        [[1.0, 0.1], [0.0, 1.0]],
        control_matrix=[[0.0], [0.1]],
        observation_matrix=[[1.0, 0.0]],
        dynamics_noise=jnp.eye(2) * dynamics_noise,
        observation_noise=[[1e-2]],
        prior=Belief(mean=[0.0, 0.0], cov=jnp.eye(2)),
    )


def _rx_model():
    return LinearGaussianModel(
        [[1.0, 0.1], [0.0, 1.0]],
        control_matrix=[[0.0], [0.1]],
        observation_matrix=[[1.0, 0.0]],
        dynamics_noise=jnp.eye(2) * 1e-6,
        observation_model=CallableSensor([[1.0, 0.0]], _sharp_near_one, ()),
        prior=Belief(mean=[0.0, 0.0], cov=jnp.eye(2)),
    )


def _diffuses_away_from_zero(x, params):
    return jnp.eye(2) * (1e-4 + 1e-2 * x[0] ** 2)


def _qx_model():
    return LinearGaussianModel(
        [[1.0, 0.1], [0.0, 1.0]],
        control_matrix=[[0.0], [0.1]],
        observation_matrix=[[1.0, 0.0]],
        dynamics_noise_model=CallableDynamicsNoise(_diffuses_away_from_zero, ()),
        observation_noise=[[1e-2]],
        prior=Belief(mean=[0.5, 0.0], cov=jnp.eye(2)),
    )


def _graph_backend():
    graph = CouplingGraph(
        dims=(1, 1),
        couplings=(Coupling(0, 1, [[0.8]], coupling_noise=[[0.05]]),),
        observation_models={1: FixedSensor([[1.0]], observation_noise=[[0.1]])},
    )
    transitions = (
        GaussianTransition([[0.7]], dynamics_noise=[[0.1]]),
        GaussianTransition([[0.5]], dynamics_noise=[[0.08]]),
    )
    return CouplingGraphBackend(
        graph, transitions, prior=unit_prior(graph), control_matrix=[[1.0], [0.0]]
    )


def _assert_infer_states_traces(backend, prior):
    eager = [backend.infer_states(o, prior, ACTION).mean for o in OBSERVATIONS]

    jitted = jax.jit(backend.infer_states)
    for observation, expected in zip(OBSERVATIONS, eager, strict=True):
        np.testing.assert_allclose(jitted(observation, prior, ACTION).mean, expected)

    batched = jax.vmap(lambda o: backend.infer_states(o, prior, ACTION).mean)
    np.testing.assert_allclose(batched(OBSERVATIONS), np.stack(eager))

    def first_mean(observation):
        return backend.infer_states(observation, prior, ACTION).mean[0]

    gradient = jax.grad(first_mean)(OBSERVATIONS[0])
    assert np.all(np.isfinite(gradient))
    assert float(jnp.abs(gradient).max()) > 0.0


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: KalmanBackend(_fixed_model()), id="per-step"),
        pytest.param(
            lambda: KalmanBackend(_fixed_model(dynamics_noise=1e-2), steady_state=True),
            id="steady-state",
        ),
        pytest.param(lambda: KalmanBackend(_rx_model()), id="per-step-R(x)"),
        pytest.param(lambda: KalmanBackend(_qx_model()), id="per-step-Q(x)"),
    ],
)
def test_kalman_infer_states_traces(build):
    backend = build()
    _assert_infer_states_traces(backend, backend.model.prior)


def test_graph_backend_infer_states_traces():
    backend = _graph_backend()
    prior = unit_prior(backend.graph)
    _assert_infer_states_traces(backend, prior)


def test_the_efe_functions_trace():
    model, preference = _rx_model(), Preference([1.0])

    def one_step(action):
        return expected_free_energy(model, model.prior, action, preference).g

    def rollout(policy):
        return policy_efe(model, model.prior, policy, preference).g

    actions = jnp.array([[-1.0], [0.0], [1.0]])
    policies = jnp.repeat(actions[:, None, :], 3, axis=1)
    for score, batch in ((one_step, actions), (rollout, policies)):
        eager = np.array([float(score(entry)) for entry in batch])
        np.testing.assert_allclose(jax.vmap(jax.jit(score))(batch), eager)
        assert np.all(np.isfinite(jax.grad(score)(batch[0])))


def _selectors():
    model = _rx_model()
    preference = Preference([1.0])
    search = EnumeratedEFESearch(
        model, FiniteActionSet([[-1.0], [0.0], [1.0]], version="v1"), horizon=2
    )
    graph = _graph_backend()
    controller = LQRController(_fixed_model(), precision=jnp.eye(2), effort=jnp.eye(1))
    return [
        pytest.param(
            EFESelector(model, n_candidates=7, action_bounds=(-2.0, 2.0), horizon=2),
            model.prior,
            preference,
            True,
            id="EFESelector",
        ),
        pytest.param(
            FFGEFESelector(
                graph, info_block=range(2), n_candidates=5, action_bounds=(-1.0, 1.0)
            ),
            unit_prior(graph.graph),
            preference,
            True,
            id="FFGEFESelector",
        ),
        pytest.param(
            LQRSelector(controller),
            controller.model.prior,
            Preference([1.0, 0.0]),
            False,
            id="LQRSelector",
        ),
        pytest.param(
            RecedingHorizonSelector(search),
            model.prior,
            preference,
            True,
            id="RecedingHorizonSelector",
        ),
    ]


@pytest.mark.parametrize(
    ("selector", "belief", "preference", "is_argmin"), _selectors()
)
def test_a_pure_selector_traces(selector, belief, preference, is_argmin):
    eager = selector.select(belief, preference)
    np.testing.assert_allclose(jax.jit(selector.select)(belief, preference), eager)

    def first_action(mean):
        return selector.select(Belief(mean, belief.cov), preference)[0]

    means = jnp.stack([belief.mean, belief.mean + 0.1])
    assert jax.vmap(first_action)(means).shape == (2,)

    gradient = jax.grad(first_action)(belief.mean)
    if is_argmin:
        np.testing.assert_array_equal(gradient, 0.0)
    else:
        np.testing.assert_allclose(gradient, -selector.controller.gain[0])


def test_model_belief_and_preference_pass_as_arguments():
    model = _fixed_model()

    @jax.jit
    def posterior_mean(model, belief, preference):
        posterior = KalmanBackend(model).infer_states(OBSERVATIONS[0], belief, ACTION)
        return posterior.mean - preference.target

    preference = Preference([1.0, 0.0])
    expected = (
        KalmanBackend(model).infer_states(OBSERVATIONS[0], model.prior, ACTION).mean
        - preference.target
    )
    np.testing.assert_allclose(posterior_mean(model, model.prior, preference), expected)


def test_a_jitted_agent_restarts_from_the_belief_it_was_traced_with():
    model = _fixed_model()
    agent = Agent(model, StateGoal([1.0, 0.0]))
    jitted = jax.jit(agent.infer_states)
    jitted(OBSERVATIONS[0])
    from_prior = KalmanBackend(model).infer_states(
        OBSERVATIONS[1], model.prior, jnp.zeros(1)
    )
    np.testing.assert_allclose(jitted(OBSERVATIONS[1]).mean, from_prior.mean)
    with pytest.raises(jax.errors.UnexpectedTracerError):
        agent.infer_states(OBSERVATIONS[2])


def test_a_jitted_sample_action_raises_before_it_keeps_the_action():
    # It reads the action on the host to refuse a NaN, so the trace stops there and
    # the agent keeps no tracer.
    agent = Agent(_fixed_model(), StateGoal([1.0, 0.0]))
    with pytest.raises(jax.errors.TracerBoolConversionError):
        jax.jit(agent.sample_action)()
    agent.infer_states(OBSERVATIONS[0])


def test_a_jitted_open_loop_selector_leaks_its_plan():
    model = _rx_model()
    search = EnumeratedEFESearch(
        model, FiniteActionSet([[-1.0], [0.0], [1.0]], version="v1"), horizon=2
    )
    selector, preference = OpenLoopSelector(search), Preference([1.0])
    jax.jit(selector.select)(model.prior, preference)
    with pytest.raises(jax.errors.UnexpectedTracerError):
        selector.select(model.prior, preference)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda scale: KalmanBackend(
                _fixed_model(dynamics_noise=1e-2 * scale), steady_state=True
            ),
            id="steady-state KalmanBackend",
        ),
        pytest.param(
            lambda scale: LQRController(
                _fixed_model(),
                precision=jnp.eye(2) * scale,
                effort=jnp.eye(1),
            ),
            id="LQRController",
        ),
        pytest.param(
            lambda scale: Agent(
                _fixed_model(), StateGoal([1.0, 0.0], precision=jnp.eye(2) * scale)
            ),
            id="Agent",
        ),
    ],
)
def test_front_loaded_construction_refuses_a_trace(build):
    with pytest.raises(jax.errors.ConcretizationTypeError):
        jax.jit(lambda scale: build(scale) and 0.0)(1.0)


def test_probe_model_runs_on_the_host():
    model = _rx_model()
    with pytest.raises(jax.errors.TracerArrayConversionError):
        jax.jit(lambda shift: probe_model(model, model.prior, [ACTION + shift]))(0.0)
