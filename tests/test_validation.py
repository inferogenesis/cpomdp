"""The one refusal every fixed-noise consumer shares."""

import jax.numpy as jnp
import pytest

from cpomdp._validation import require_fixed_noise
from cpomdp.dynamics import CallableProcessNoise
from cpomdp.observation import CallableSensor, FixedSensor
from cpomdp.types import Belief, LinearGaussianModel


def _rising_noise(x, params):
    return jnp.array([[1.0 + x[0] ** 2]])


def _model(**noise_models):
    return LinearGaussianModel(
        dynamics_matrix=[[0.9]],
        observation_matrix=[[1.0]],
        dynamics_noise=[[0.5]],
        observation_noise=[[1.0]],
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


def test_a_slot_left_out_is_not_checked():
    # The full-information cost observes the state exactly and never reads R, so a
    # state-dependent sensor is no reason for it to refuse.
    model = _model(observation_model=CallableSensor([[1.0]], _rising_noise, ()))
    require_fixed_noise(model, "unused", slots=("dynamics_noise_model",))
