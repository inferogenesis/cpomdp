"""CallableDynamicsNoise: state-dependent dynamics noise Q(x).

The dynamics-side twin of CallableSensor. ``noise_params`` is a pytree leaf (grad-able)
and ``noise_fn`` is static aux. Unlike CallableSensor it cannot probe its own
``(n, n)`` shape at construction, since it has no dimension. The model validates that
when it knows ``n``. See test_types.py for the model-level shape rejection.
"""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp.dynamics import CallableDynamicsNoise, DynamicsNoiseModel


def _well_dynamics_noise(state, params):
    """Q(state): low near 0, growing with the position squared."""
    return jnp.array([[params["base"] + params["slope"] * state[0] ** 2]])


def _params():
    return {"base": jnp.array(0.05), "slope": jnp.array(0.4)}


def _dynamics_noise_model():
    return CallableDynamicsNoise(noise_fn=_well_dynamics_noise, noise_params=_params())


class TestCallableDynamicsNoise:
    def test_noise_varies_with_the_state(self):
        dynamics_noise_model = _dynamics_noise_model()
        noise_at_origin = dynamics_noise_model.noise_at(jnp.array([0.0]))
        noise_away = dynamics_noise_model.noise_at(jnp.array([2.0]))
        assert float(noise_at_origin[0, 0]) < float(noise_away[0, 0])
        assert dynamics_noise_model.is_fixed is False

    def test_it_takes_its_arguments_by_the_names_callable_sensor_uses(self):
        dynamics_noise_model = _dynamics_noise_model()
        assert dynamics_noise_model.noise_fn is _well_dynamics_noise
        assert dynamics_noise_model.noise_params.keys() == {"base", "slope"}

    def test_a_state_given_as_a_list_is_read_as_an_array(self):
        np.testing.assert_array_equal(
            _dynamics_noise_model().noise_at([2.0]),
            _dynamics_noise_model().noise_at(jnp.array([2.0])),
        )

    def test_satisfies_the_protocol(self):
        assert isinstance(_dynamics_noise_model(), DynamicsNoiseModel)

    def test_a_plain_matrix_does_not_satisfy_the_protocol(self):
        # A constant Q goes in dynamics_noise. The protocol is for a Q(x).
        assert not isinstance(jnp.eye(2), DynamicsNoiseModel)

    def test_a_noise_fn_that_is_not_callable_is_refused(self):
        with pytest.raises(TypeError, match="noise_fn"):
            # Passing the wrong type is the test, so ty's argument error is expected.
            CallableDynamicsNoise(
                noise_fn=0.5,  # ty: ignore[invalid-argument-type]
                noise_params=None,
            )

    def test_pytree_round_trip_preserves_noise_fn(self):
        dynamics_noise_model = _dynamics_noise_model()
        leaves, treedef = jax.tree_util.tree_flatten(dynamics_noise_model)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(restored, CallableDynamicsNoise)
        np.testing.assert_array_equal(
            restored.noise_at(jnp.array([1.0])),
            dynamics_noise_model.noise_at(jnp.array([1.0])),
        )

    def test_grad_wrt_noise_params_flows(self):
        def scalar(params):
            return jnp.trace(
                CallableDynamicsNoise(_well_dynamics_noise, params).noise_at(
                    jnp.array([1.0])
                )
            )

        grads = jax.tree_util.tree_leaves(jax.grad(scalar)(_params()))
        assert all(bool(jnp.all(jnp.isfinite(grad))) for grad in grads)


def test_the_protocol_takes_the_state_by_position():
    # An implementer may name the argument anything, so no caller passes it by name.
    _, state = inspect.signature(DynamicsNoiseModel.noise_at).parameters.values()
    assert state.kind is inspect.Parameter.POSITIONAL_ONLY
