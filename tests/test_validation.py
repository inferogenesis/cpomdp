"""The refusals shared across consumers: fixed noise, and a target the dynamics hold."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp._validation import equilibrium_drift, require_fixed_noise, state_dependent
from cpomdp.dynamics import CallableProcessNoise
from cpomdp.observation import CallableSensor, FixedSensor
from cpomdp.types import Belief, LinearGaussianModel


def _rising_noise(x, params):
    return jnp.array([[1.0 + x[0] ** 2]])


def _model(**noise_models):
    return LinearGaussianModel(
        dynamics_matrix=[[0.9]],
        observation_matrix=[[1.0]],
        dynamics_noise=None if "dynamics_noise_model" in noise_models else [[0.5]],
        observation_noise=None if "observation_model" in noise_models else [[1.0]],
        prior=Belief(mean=[0.0], cov=[[10.0]]),
        **noise_models,
    )


def test_a_model_with_no_noise_objects_passes():
    require_fixed_noise(_model(), "unused")


def test_a_fixed_noise_object_passes():
    sensor = FixedSensor([[1.0]], observation_noise=[[1.0]])
    require_fixed_noise(_model(observation_model=sensor), "unused")


@pytest.mark.parametrize(
    ("slot", "noise_model"),
    [
        ("observation_model", CallableSensor([[1.0]], _rising_noise, ())),
        ("dynamics_noise_model", CallableProcessNoise(_rising_noise, ())),
    ],
)
def test_a_state_dependent_slot_is_named_beside_the_reason(slot, noise_model):
    with pytest.raises(
        ValueError, match=f"^the model's {slot} is state-dependent; the reason$"
    ):
        require_fixed_noise(_model(**{slot: noise_model}), "the reason")


def test_the_sensor_is_named_first_when_both_vary():
    model = _model(
        observation_model=CallableSensor([[1.0]], _rising_noise, ()),
        dynamics_noise_model=CallableProcessNoise(_rising_noise, ()),
    )
    with pytest.raises(ValueError, match="observation_model"):
        require_fixed_noise(model, "the reason")


def test_an_absent_or_fixed_noise_object_is_not_state_dependent():
    assert state_dependent(None) is None
    assert state_dependent(FixedSensor([[1.0]], observation_noise=[[1.0]])) is None


@pytest.mark.parametrize(
    "noise_model",
    [
        CallableSensor([[1.0]], _rising_noise, ()),
        CallableProcessNoise(_rising_noise, ()),
    ],
)
def test_a_state_dependent_noise_object_comes_back(noise_model):
    assert state_dependent(noise_model) is noise_model


def test_a_slot_left_out_is_not_checked():
    # The full-information cost observes the state exactly and never reads R, so a
    # state-dependent sensor is no reason for it to refuse.
    model = _model(observation_model=CallableSensor([[1.0]], _rising_noise, ()))
    require_fixed_noise(model, "unused", slots=("dynamics_noise_model",))


def test_a_state_the_dynamics_hold_has_no_drift():
    assert equilibrium_drift([[1.0, 0.1], [0.0, 1.0]], [1.0, 0.0]) is None


def test_a_state_the_dynamics_move_reports_where_it_goes():
    drift = equilibrium_drift([[0.9]], [1.0])
    assert drift is not None
    np.testing.assert_allclose(drift, [0.9])


def test_equilibrium_drift_is_skipped_under_a_trace():
    seen = []

    def traced(state):
        seen.append(equilibrium_drift([[0.9]], state))
        return state

    jax.jit(traced)(jnp.ones(1))
    assert seen == [None]
