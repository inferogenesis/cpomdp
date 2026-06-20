import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp.observation import FixedSensor
from cpomdp.types import Belief, LinearGaussianModel


class TestBeliefs:
    def test_valid_belief_stores_what_you_passed(self):
        b = Belief(mean=[1.0, 2.0], cov=[[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_equal(b.mean, [1.0, 2.0])
        np.testing.assert_array_equal(b.cov, [[1.0, 0.0], [0.0, 1.0]])

    def test_coerce_lists_to_float_arrays(self):
        b = Belief(mean=[0, 1], cov=[[1, 0], [0, 1]])
        assert isinstance(b.mean, jax.Array)
        assert b.mean.dtype == jnp.float64

    def test_rejects_mean_not_1D(self):
        with pytest.raises(ValueError, match="1-D"):
            Belief(mean=[[0.0]], cov=[[1.0]])

    def test_rejects_nonfinite_mean(self):
        # A NaN/Inf mean (e.g. from a degenerate upstream step) must be rejected, not
        # silently propagated as a NaN belief.
        with pytest.raises(ValueError, match="finite"):
            Belief(mean=[0.0, float("nan")], cov=[[1.0, 0.0], [0.0, 1.0]])

    def test_rejects_cov_not_2D(self):
        with pytest.raises(ValueError, match="2-D"):
            Belief(mean=[0.0], cov=[1.0])

    def test_reject_shape_mismatch(self):
        with pytest.raises(ValueError, match="match"):
            Belief(mean=[0.0, 0.0], cov=[[1.0]])

    def test_rejects_asymmetric_cov(self):
        with pytest.raises(ValueError, match="symmetric"):
            Belief(mean=[0.0, 0.0], cov=[[1.0, 0.2], [0.9, 1.0]])

    def test_rejects_indefinite_cov(self):
        # eigenvalues (-1, 3): a plausible "correlation larger than the variances"
        # typo. Without the PSD check the Kalman update yields a negative variance.
        with pytest.raises(ValueError, match="positive-semi-definite"):
            Belief(mean=[0.0, 0.0], cov=[[1.0, 2.0], [2.0, 1.0]])

    def test_accepts_semidefinite_cov(self):
        # A degenerate (zero-variance) but valid covariance must still construct.
        Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 0.0]])

    def test_ndim_reports_state_dimension(self):
        assert Belief(mean=[0.0], cov=[[1.0]]).ndim == 1
        assert Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]).ndim == 2


# A valid 2-state / 1-observation model, reused as the baseline. Each rejection
# test overrides exactly one field with a bad value, so the test isolates one
# validation branch.
def _valid_kwargs(**overrides):
    kwargs = {
        "dynamics": [[1.0, 0.1], [0.0, 1.0]],  # 2x2  (n=2)
        "sensor_model": [[1.0, 0.0]],  # 1x2  (m=1)
        "dynamics_noise": [[0.1, 0.0], [0.0, 0.1]],  # 2x2
        "sensor_noise": [[1.0]],  # 1x1
        "prior": Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
    }
    kwargs.update(overrides)
    return kwargs


class TestLinearGaussianModels:
    def test_valid_model_constructs(self):
        m = LinearGaussianModel(**_valid_kwargs())
        np.testing.assert_array_equal(m.dynamics, [[1.0, 0.1], [0.0, 1.0]])
        assert m.n_states == 2
        assert m.n_observations == 1

    def test_control_is_optional(self):
        m = LinearGaussianModel(**_valid_kwargs())
        assert m.control is None
        assert m.B is None
        assert m.n_controls == 0

    def test_with_control(self):
        m = LinearGaussianModel(**_valid_kwargs(control=[[0.0], [1.0]]))  # 2x1 (p=1)
        np.testing.assert_array_equal(m.control, [[0.0], [1.0]])
        np.testing.assert_array_equal(m.B, [[0.0], [1.0]])
        assert m.n_controls == 1

    def test_letter_aliases_map_to_role_names(self):
        m = LinearGaussianModel(**_valid_kwargs())
        np.testing.assert_array_equal(m.A, m.dynamics)
        np.testing.assert_array_equal(m.C, m.sensor_model)
        np.testing.assert_array_equal(m.Q, m.dynamics_noise)
        np.testing.assert_array_equal(m.R, m.sensor_noise)

    def test_rejects_non_square_dynamics(self):
        with pytest.raises(ValueError, match="square"):
            LinearGaussianModel(
                **_valid_kwargs(dynamics=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            )

    def test_rejects_sensor_model_wrong_columns(self):
        with pytest.raises(ValueError, match="columns"):
            LinearGaussianModel(**_valid_kwargs(sensor_model=[[1.0, 0.0, 0.0]]))

    def test_rejects_dynamics_noise_wrong_size(self):
        with pytest.raises(ValueError, match="dynamics_noise"):
            LinearGaussianModel(**_valid_kwargs(dynamics_noise=[[1.0]]))

    def test_rejects_sensor_noise_wrong_size(self):
        with pytest.raises(ValueError, match="sensor_noise"):
            LinearGaussianModel(**_valid_kwargs(sensor_noise=[[1.0, 0.0], [0.0, 1.0]]))

    def test_rejects_indefinite_dynamics_noise(self):
        with pytest.raises(ValueError, match="positive-semi-definite"):
            LinearGaussianModel(
                **_valid_kwargs(dynamics_noise=[[-1.0, 0.0], [0.0, -1.0]])
            )

    def test_rejects_negative_sensor_noise(self):
        # sensor_noise is positive-DEFINITE (not just PSD): the epistemic term inverts
        # it, so a negative or zero R is rejected.
        with pytest.raises(ValueError, match="positive-definite"):
            LinearGaussianModel(**_valid_kwargs(sensor_noise=[[-0.5]]))

    def test_rejects_noiseless_sensor_noise(self):
        # A noiseless R=0 sends the EFE information gain to +inf and silently
        # collapses action selection — reject it as not positive-definite.
        with pytest.raises(ValueError, match="positive-definite"):
            LinearGaussianModel(**_valid_kwargs(sensor_noise=[[0.0]]))

    def test_rejects_control_wrong_rows(self):
        with pytest.raises(ValueError, match="control"):
            LinearGaussianModel(**_valid_kwargs(control=[[0.0], [0.0], [0.0]]))  # 3x1

    def test_rejects_prior_wrong_dimension(self):
        bad_prior = Belief(mean=[0.0, 0.0, 0.0], cov=np.eye(3))  # 3-D, but n=2
        with pytest.raises(ValueError, match="prior"):
            LinearGaussianModel(**_valid_kwargs(prior=bad_prior))

    def test_rejects_prior_not_a_belief(self):
        with pytest.raises(TypeError, match="Belief"):
            LinearGaussianModel(**_valid_kwargs(prior=[0.0, 0.0]))

    def test_observation_defaults_to_none(self):
        # No observation given -> fixed sensor defined by sensor_model/sensor_noise
        # (the v0.2 semantics). None is the canonical "fixed" case.
        assert LinearGaussianModel(**_valid_kwargs()).observation is None

    def test_accepts_an_observation_model(self):
        sensor = FixedSensor([[1.0, 0.0]], [[1.0]])
        m = LinearGaussianModel(**_valid_kwargs(observation=sensor))
        assert m.observation is sensor

    def test_rejects_observation_not_an_observation_model(self):
        with pytest.raises(TypeError, match="ObservationModel"):
            LinearGaussianModel(**_valid_kwargs(observation="not a sensor"))


class TestPytreeRegistration:
    def test_belief_flattens_to_its_two_arrays(self):
        b = Belief(mean=[1.0, 2.0], cov=[[1.0, 0.0], [0.0, 1.0]])
        leaves = jax.tree_util.tree_leaves(b)
        assert len(leaves) == 2

    def test_belief_survives_a_flatten_unflatten_round_trip(self):
        b = Belief(mean=[1.0, 2.0], cov=[[2.0, 0.5], [0.5, 3.0]])
        leaves, treedef = jax.tree_util.tree_flatten(b)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(restored, Belief)
        np.testing.assert_array_equal(restored.mean, b.mean)
        np.testing.assert_array_equal(restored.cov, b.cov)

    def test_unflatten_does_not_re_validate(self):
        # JAX rebuilds from leaves under jit/vmap/grad, where leaves are tracers;
        # the rebuild path must not run the symmetry/shape checks. An asymmetric
        # cov that __init__ would reject must pass straight through unflatten.
        asymmetric = jnp.array([[1.0, 0.9], [0.2, 1.0]])
        rebuilt = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(Belief(mean=[0.0, 0.0], cov=jnp.eye(2))),
            [jnp.zeros(2), asymmetric],
        )
        np.testing.assert_array_equal(rebuilt.cov, asymmetric)

    def test_vmap_maps_over_a_batch_of_beliefs(self):
        b1 = Belief(mean=[1.0, 0.0], cov=jnp.eye(2))
        b2 = Belief(mean=[0.0, 1.0], cov=jnp.eye(2))
        batch = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), b1, b2)
        means = jax.vmap(lambda b: b.mean)(batch)
        assert means.shape == (2, 2)
        np.testing.assert_array_equal(means, [[1.0, 0.0], [0.0, 1.0]])

    def test_model_round_trips_and_keeps_control_none(self):
        m = LinearGaussianModel(**_valid_kwargs())
        leaves, treedef = jax.tree_util.tree_flatten(m)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(restored, LinearGaussianModel)
        assert restored.control is None
        np.testing.assert_array_equal(restored.dynamics, m.dynamics)
        np.testing.assert_array_equal(restored.prior.mean, m.prior.mean)

    def test_model_round_trips_with_control(self):
        m = LinearGaussianModel(**_valid_kwargs(control=[[0.0], [1.0]]))
        leaves, treedef = jax.tree_util.tree_flatten(m)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        np.testing.assert_array_equal(restored.control, m.control)

    def test_model_round_trips_with_an_observation(self):
        # observation is a nullable child like control: a FixedSensor recurses
        # into its own array leaves and is rebuilt on unflatten.
        sensor = FixedSensor([[1.0, 0.0]], [[0.5]])
        m = LinearGaussianModel(**_valid_kwargs(observation=sensor))
        leaves, treedef = jax.tree_util.tree_flatten(m)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(restored.observation, FixedSensor)
        c_out, r_out = restored.observation.linearize(jnp.zeros(2))
        np.testing.assert_array_equal(c_out, sensor.sensor_model)
        np.testing.assert_array_equal(r_out, sensor.sensor_noise)
