import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp._lab.degraded import (
    DiagonalCovarianceBackend,
    WrongFixedRBackend,
)
from cpomdp.backends.base import InferenceBackend
from cpomdp.backends.kalman import KalmanBackend
from cpomdp.dynamics import CallableProcessNoise
from cpomdp.observation import CallableSensor
from cpomdp.types import Belief, LinearGaussianModel

# The double integrator again: position is observed, velocity is inferred through the
# off-diagonal, so the posterior covariance is genuinely correlated and zeroing it costs
# something measurable.
DT = 0.1
DYNAMICS = np.array([[1.0, DT], [0.0, 1.0]])
CONTROL = np.array([[0.0], [DT]])
SENSOR = np.array([[1.0, 0.0]])


def _model(*, observation_noise=None, observation_model=None) -> LinearGaussianModel:
    return LinearGaussianModel(
        dynamics_matrix=DYNAMICS,
        observation_matrix=SENSOR,
        dynamics_noise=[[1e-3, 0.0], [0.0, 1e-3]],
        observation_noise=[[1e-2]] if observation_noise is None else observation_noise,
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        control_matrix=CONTROL,
        observation_model=observation_model,
    )


def _noise_at(mean, params):
    return jnp.array([[1e-2]]) * (1.0 + jnp.sum(jnp.asarray(mean) ** 2))


def _process_noise_at(x, params):
    return jnp.eye(2) * 1e-3 * (1.0 + jnp.sum(jnp.asarray(x) ** 2))


def _sensed_model() -> LinearGaussianModel:
    return _model(observation_model=CallableSensor(SENSOR, _noise_at, ()))


def _run(backend, steps: int = 6) -> Belief:
    belief = backend.model.prior
    for step in range(steps):
        belief = backend.infer_states([0.1 * step], belief, [1.0])
    return belief


# --- the wrapper keeps the model it was built for -----------------------------------


def test_wrong_fixed_r_reports_the_model_it_was_built_for():
    model = _model()
    assert WrongFixedRBackend(model, magnitude=0.5).model is model


def test_diagonal_covariance_reports_the_model_it_was_built_for():
    model = _model()
    assert DiagonalCovarianceBackend(KalmanBackend(model)).model is model


def test_both_satisfy_the_backend_protocol():
    model = _model()
    assert isinstance(WrongFixedRBackend(model, magnitude=0.5), InferenceBackend)
    assert isinstance(DiagonalCovarianceBackend(KalmanBackend(model)), InferenceBackend)


def test_each_names_its_degradation():
    model = _model()
    assert WrongFixedRBackend(model, magnitude=0.5).degradation == "WrongFixedR"
    assert (
        DiagonalCovarianceBackend(KalmanBackend(model)).degradation
        == "DiagonalCovarianceOnly"
    )


# --- WrongFixedR filters on a scaled R, checked against the model that carries it ----


def test_wrong_fixed_r_matches_a_filter_over_the_scaled_model():
    scaled = _model(observation_noise=[[1.5e-2]])
    degraded = _run(WrongFixedRBackend(_model(), magnitude=0.5))
    oracle = _run(KalmanBackend(scaled))
    assert np.allclose(np.asarray(degraded.mean), np.asarray(oracle.mean), atol=0)
    assert np.allclose(np.asarray(degraded.cov), np.asarray(oracle.cov), atol=0)


def test_wrong_fixed_r_differs_from_exact_inference():
    exact = _run(KalmanBackend(_model()))
    degraded = _run(WrongFixedRBackend(_model(), magnitude=0.5))
    assert not np.allclose(np.asarray(exact.cov), np.asarray(degraded.cov))


def test_a_zero_magnitude_on_a_fixed_sensor_reproduces_exact_inference():
    exact = _run(KalmanBackend(_model()))
    degraded = _run(WrongFixedRBackend(_model(), magnitude=0.0))
    assert np.allclose(np.asarray(exact.mean), np.asarray(degraded.mean), atol=0)
    assert np.allclose(np.asarray(exact.cov), np.asarray(degraded.cov), atol=0)


def test_a_state_dependent_sensor_is_replaced_by_the_fixed_fallback():
    sensed = _sensed_model()
    degraded = _run(WrongFixedRBackend(sensed, magnitude=0.0))
    oracle = _run(KalmanBackend(_model()))  # the same model with R(x) dropped
    assert np.allclose(np.asarray(degraded.cov), np.asarray(oracle.cov), atol=0)


def test_an_omitted_noise_falls_back_to_the_sensors_r_at_the_prior_mean():
    # Off the origin, so R at the prior mean is not the R(0) the sensor probes itself
    # at, and the fallback reads the one the model derived.
    prior = Belief(mean=[1.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]])
    sensed = LinearGaussianModel(
        dynamics_matrix=DYNAMICS,
        observation_matrix=SENSOR,
        dynamics_noise=[[1e-3, 0.0], [0.0, 1e-3]],
        prior=prior,
        control_matrix=CONTROL,
        observation_model=CallableSensor(SENSOR, _noise_at, ()),
    )
    frozen = LinearGaussianModel(
        dynamics_matrix=DYNAMICS,
        observation_matrix=SENSOR,
        dynamics_noise=[[1e-3, 0.0], [0.0, 1e-3]],
        observation_noise=_noise_at(prior.mean, ()),
        prior=prior,
        control_matrix=CONTROL,
    )
    degraded = _run(WrongFixedRBackend(sensed, magnitude=0.0))
    oracle = _run(KalmanBackend(frozen))
    assert np.allclose(np.asarray(degraded.mean), np.asarray(oracle.mean), atol=0)
    assert np.allclose(np.asarray(degraded.cov), np.asarray(oracle.cov), atol=0)


def test_a_state_dependent_sensor_makes_a_zero_magnitude_a_real_degradation():
    sensed = _sensed_model()
    exact = _run(KalmanBackend(sensed))
    degraded = _run(WrongFixedRBackend(sensed, magnitude=0.0))
    assert not np.allclose(np.asarray(exact.cov), np.asarray(degraded.cov))


def test_the_scored_model_keeps_its_state_dependent_sensor():
    sensed = _sensed_model()
    backend = WrongFixedRBackend(sensed, magnitude=0.5)
    assert backend.model.observation_model is sensed.observation_model


def test_state_dependent_process_noise_reaches_the_substituted_filter():
    # R is what WrongFixedR replaces. Q(x) is not, so dropping it would be a second,
    # unnamed degradation, and the filter would move for a reason nothing declares.
    process = CallableProcessNoise(_process_noise_at, ())
    model = LinearGaussianModel(
        dynamics_matrix=DYNAMICS,
        observation_matrix=SENSOR,
        dynamics_noise=[[1e-3, 0.0], [0.0, 1e-3]],
        observation_noise=[[1e-2]],
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        control_matrix=CONTROL,
        dynamics_noise_model=process,
    )
    fixed_process = _model()
    assert not np.allclose(
        np.asarray(_run(KalmanBackend(model)).cov),
        np.asarray(_run(KalmanBackend(fixed_process)).cov),
    )  # Q(x) moves this model, so the check below can see it go missing

    scaled = LinearGaussianModel(
        dynamics_matrix=DYNAMICS,
        observation_matrix=SENSOR,
        dynamics_noise=[[1e-3, 0.0], [0.0, 1e-3]],
        observation_noise=[[1.5e-2]],
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        control_matrix=CONTROL,
        dynamics_noise_model=process,
    )
    degraded = _run(WrongFixedRBackend(model, magnitude=0.5))
    assert np.allclose(
        np.asarray(degraded.cov), np.asarray(_run(KalmanBackend(scaled)).cov), atol=0
    )


def test_a_magnitude_that_empties_the_noise_does_not_construct():
    with pytest.raises(ValueError, match="observation_noise"):
        WrongFixedRBackend(_model(), magnitude=-1.0)


# --- DiagonalCovarianceOnly drops the correlation, and it propagates -----------------


def test_the_posterior_covariance_comes_back_diagonal():
    posterior = _run(DiagonalCovarianceBackend(KalmanBackend(_model())), steps=1)
    assert float(posterior.cov[0, 1]) == 0.0
    assert float(posterior.cov[1, 0]) == 0.0


def test_the_exact_filter_leaves_a_correlation_to_drop():
    posterior = _run(KalmanBackend(_model()), steps=1)
    assert abs(float(posterior.cov[0, 1])) > 1e-6


def test_dropping_the_correlation_moves_the_variances_it_fed():
    exact = _run(KalmanBackend(_model()))
    degraded = _run(DiagonalCovarianceBackend(KalmanBackend(_model())))
    assert not np.allclose(
        np.diag(np.asarray(exact.cov)), np.diag(np.asarray(degraded.cov))
    )


def test_dropping_the_correlation_moves_the_mean():
    exact = _run(KalmanBackend(_model()))
    degraded = _run(DiagonalCovarianceBackend(KalmanBackend(_model())))
    assert not np.allclose(np.asarray(exact.mean), np.asarray(degraded.mean))


def test_a_one_dimensional_state_has_no_correlation_to_drop():
    scalar = LinearGaussianModel(
        dynamics_matrix=[[0.9]],
        observation_matrix=[[1.0]],
        dynamics_noise=[[1e-3]],
        observation_noise=[[1e-2]],
        prior=Belief(mean=[0.0], cov=[[1.0]]),
        control_matrix=[[1.0]],
    )
    exact = _run(KalmanBackend(scalar))
    degraded = _run(DiagonalCovarianceBackend(KalmanBackend(scalar)))
    assert np.allclose(np.asarray(exact.cov), np.asarray(degraded.cov), atol=0)


def test_the_wrapper_stacks_on_a_frozen_gain_filter():
    frozen = KalmanBackend(_model(), steady_state=True)
    stacked = DiagonalCovarianceBackend(frozen)
    posterior = _run(stacked, steps=1)
    assert stacked.model is frozen.model
    assert float(posterior.cov[0, 1]) == 0.0
