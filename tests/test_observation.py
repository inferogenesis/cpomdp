"""FixedSensor: the constant-(C, R) observation model."""

import dataclasses
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp.observation import CallableSensor, FixedSensor, ObservationModel


def test_fixed_sensor_observation_noise_is_keyword_only():
    # C is (m, n) and R is (m, m), so whenever m == n the two are interchangeable by
    # shape. Only R is content-checked, and a square symmetric C passes that check, so
    # a transposed pair builds a sensor whose Jacobian is the noise. Keywords remove
    # the route; a 1-D sensor is the common case where the shapes coincide.
    with pytest.raises(TypeError, match="positional"):
        # Calling it wrongly is the test, so ty's arity errors are expected.
        FixedSensor(  # ty: ignore[missing-argument]
            jnp.eye(2),
            jnp.eye(2) * 0.1,  # ty: ignore[too-many-positional-arguments]
        )


def test_fixed_sensor_linearize_returns_stored_matrices():
    C = jnp.array([[1.0, 0.0]])
    R = jnp.array([[0.5]])
    sensor = FixedSensor(C, observation_noise=R)

    C_out, R_out = sensor.linearize(jnp.array([3.0, 7.0]))

    assert jnp.allclose(C_out, C)
    assert jnp.allclose(R_out, R)


def test_fixed_sensor_is_state_independent():
    sensor = FixedSensor(jnp.array([[1.0, 0.0]]), observation_noise=jnp.array([[0.5]]))

    C0, R0 = sensor.linearize(jnp.array([0.0, 0.0]))
    C1, R1 = sensor.linearize(jnp.array([99.0, -99.0]))

    assert jnp.allclose(C0, C1)
    assert jnp.allclose(R0, R1)
    assert sensor.is_fixed is True


def test_fixed_sensor_satisfies_protocol():
    sensor = FixedSensor(jnp.array([[1.0]]), observation_noise=jnp.array([[1.0]]))

    assert isinstance(sensor, ObservationModel)


def test_fixed_sensor_rejects_non_2d_observation_matrix():
    with pytest.raises(ValueError, match="2-D"):
        FixedSensor(jnp.array([1.0, 0.0]), observation_noise=jnp.array([[0.5]]))


def test_fixed_sensor_rejects_mismatched_dims():
    C = jnp.array([[1.0, 0.0], [0.0, 1.0]])  # m=2
    R = jnp.array([[0.5]])  # m=1
    with pytest.raises(ValueError, match="match the 2-D observation"):
        FixedSensor(C, observation_noise=R)


def test_FixedSensor_survives_a_flatten_unflatten_round_trip():
    sensor = FixedSensor(jnp.array([[1.0]]), observation_noise=jnp.array([[1.0]]))
    leaves, treedef = jax.tree_util.tree_flatten(sensor)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(restored, FixedSensor)
    np.testing.assert_array_equal(
        restored.observation_matrix, sensor.observation_matrix
    )
    np.testing.assert_array_equal(restored.observation_noise, sensor.observation_noise)


# --- CallableSensor: state-dependent noise R(x), constant C (Phase 2a) ---
# noise_fn is module-level (NOT a lambda) so it is hashable by a stable identity
# and rides in the pytree's static aux without breaking jit caching.
def _well_noise(x, params):
    """R(x): low (sharp) near params['beacon'], rising to r_hi away from it."""
    pos = x[0]
    falloff = 1.0 - jnp.exp(
        -((pos - params["beacon"]) ** 2) / (2.0 * params["width"] ** 2)
    )
    r = params["r_lo"] + (params["r_hi"] - params["r_lo"]) * falloff
    return jnp.array([[r]])


def _well_params():
    return {
        "beacon": jnp.array(1.5),
        "width": jnp.array(0.6),
        "r_lo": jnp.array(0.05),
        "r_hi": jnp.array(0.8),
    }


def _callable_sensor():
    return CallableSensor(
        observation_matrix=[[1.0]], noise_fn=_well_noise, noise_params=_well_params()
    )


class TestCallableSensor:
    def test_R_varies_with_state_C_does_not(self):
        s = _callable_sensor()
        c0, r0 = s.linearize(jnp.array([1.5]))  # at the beacon (sharp)
        c1, r1 = s.linearize(jnp.array([6.0]))  # far away (blurry)
        np.testing.assert_array_equal(c0, c1)  # C constant
        assert float(r0[0, 0]) < float(r1[0, 0])  # sharper at the beacon
        assert s.is_fixed is False

    def test_satisfies_protocol(self):
        assert isinstance(_callable_sensor(), ObservationModel)

    def test_gaussianize_is_linear_moment_match(self):
        s = _callable_sensor()
        x, sigma = jnp.array([0.7]), jnp.array([[2.0]])
        o, big_s, r_out = s.gaussianize(x, sigma)
        c, r = s.linearize(x)
        np.testing.assert_allclose(o, c @ x, atol=1e-12)
        np.testing.assert_allclose(big_s, c @ sigma @ c.T + r, atol=1e-12)
        np.testing.assert_array_equal(r_out, r)  # conditional noise passed through

    def test_pytree_round_trip_preserves_noise_fn(self):
        s = _callable_sensor()
        leaves, treedef = jax.tree_util.tree_flatten(s)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(restored, CallableSensor)
        _, r = restored.linearize(jnp.array([1.5]))
        _, r_ref = s.linearize(jnp.array([1.5]))
        np.testing.assert_array_equal(r, r_ref)  # noise_fn survived in aux

    def test_grad_wrt_params_flows(self):
        # params is a pytree LEAF, so grad of R(x) w.r.t. params is finite.
        def scalar(params):
            s = CallableSensor(
                observation_matrix=[[1.0]], noise_fn=_well_noise, noise_params=params
            )
            _, r = s.linearize(jnp.array([0.7]))
            return jnp.trace(r)

        grads = jax.tree_util.tree_leaves(jax.grad(scalar)(_well_params()))
        assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads)
        assert any(float(jnp.abs(g)) > 0 for g in grads)  # actually depends on params

    def test_rejects_bad_noise_shape(self):
        def bad_noise(x, params):
            return jnp.array([1.0])  # 1-D, not (m, m)

        with pytest.raises(ValueError, match="noise_fn"):
            CallableSensor(
                observation_matrix=[[1.0]], noise_fn=bad_noise, noise_params={}
            )


# --- one name for the state, and the protocol takes it by position ------------------


@pytest.mark.parametrize(
    ("method", "arguments"),
    [
        (ObservationModel.linearize, ("state",)),
        (ObservationModel.gaussianize, ("mean", "cov")),
    ],
)
def test_the_protocol_takes_its_arguments_by_position(method, arguments):
    # An implementer may name them anything, so no caller passes them by name.
    _, *parameters = inspect.signature(method).parameters.values()
    assert tuple(parameter.name for parameter in parameters) == arguments
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_ONLY for parameter in parameters
    )


@pytest.mark.parametrize(
    "sensor",
    [FixedSensor([[1.0]], observation_noise=[[0.5]]), _callable_sensor()],
    ids=["fixed", "callable"],
)
def test_the_sensors_name_the_state_as_the_protocol_does(sensor):
    for method, arguments in (
        (sensor.linearize, ("state",)),
        (sensor.gaussianize, ("mean", "cov")),
    ):
        parameters = inspect.signature(method).parameters.values()
        assert tuple(parameter.name for parameter in parameters) == arguments
        assert all(
            parameter.kind is inspect.Parameter.POSITIONAL_ONLY
            for parameter in parameters
        )


# 0.5.0 declared `is_fixed` a plain attribute, so a subclass could set it per instance.
# A type checker now refuses that assignment, and treats a subclass that leaves the
# protocol's methods unimplemented as abstract, so ty's errors here are expected. Both
# classes still run.
def test_a_subclass_may_set_is_fixed_on_the_instance():
    class FlagOnInstance(ObservationModel):
        def __init__(self, is_fixed):
            self.is_fixed = is_fixed  # ty: ignore[invalid-assignment]

    assert FlagOnInstance(True).is_fixed is True  # ty: ignore[call-non-callable]


def test_a_frozen_dataclass_subclass_may_take_is_fixed_as_a_field():
    @dataclasses.dataclass(frozen=True)
    class FlagAsField(ObservationModel):
        is_fixed: bool

    assert FlagAsField(is_fixed=False).is_fixed is False  # ty: ignore[call-non-callable]
