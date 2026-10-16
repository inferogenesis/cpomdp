import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp.dynamics import CallableProcessNoise
from cpomdp.observation import CallableSensor, FixedSensor
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
        "dynamics_matrix": [[1.0, 0.1], [0.0, 1.0]],  # 2x2  (n=2)
        "observation_matrix": [[1.0, 0.0]],  # 1x2  (m=1)
        "dynamics_noise": [[0.1, 0.0], [0.0, 0.1]],  # 2x2
        "observation_noise": [[1.0]],  # 1x1
        "prior": Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
    }
    kwargs.update(overrides)
    return kwargs


class TestLinearGaussianModels:
    def test_valid_model_constructs(self):
        m = LinearGaussianModel(**_valid_kwargs())
        np.testing.assert_array_equal(m.dynamics_matrix, [[1.0, 0.1], [0.0, 1.0]])
        assert m.n_states == 2
        assert m.n_observations == 1

    def test_control_is_optional(self):
        m = LinearGaussianModel(**_valid_kwargs())
        assert m.control_matrix is None
        assert m.B is None
        assert m.n_controls == 0

    def test_with_control(self):
        m = LinearGaussianModel(
            **_valid_kwargs(control_matrix=[[0.0], [1.0]])
        )  # 2x1 (p=1)
        np.testing.assert_array_equal(m.control_matrix, [[0.0], [1.0]])
        np.testing.assert_array_equal(m.B, [[0.0], [1.0]])
        assert m.n_controls == 1

    def test_matrices_after_dynamics_are_keyword_only(self):
        # Two same-shaped matrices in adjacent positional slots is the shape that builds
        # a wrong model in silence: only the covariances are content-checked, so a
        # transposed pair passes whenever the maps are square and symmetric. Keywords
        # close the route instead of trying to detect the mistake after the fact.
        k = _valid_kwargs()
        with pytest.raises(TypeError, match="positional"):
            # Making the call wrongly is the point, so ty's arity errors are expected.
            LinearGaussianModel(  # ty: ignore[missing-argument]
                k["dynamics_matrix"],
                k["observation_matrix"],  # ty: ignore[too-many-positional-arguments]
                k["dynamics_noise"],
                k["observation_noise"],
                k["prior"],
            )

    def test_letter_aliases_map_to_role_names(self):
        m = LinearGaussianModel(**_valid_kwargs())
        np.testing.assert_array_equal(m.A, m.dynamics_matrix)
        np.testing.assert_array_equal(m.C, m.observation_matrix)
        np.testing.assert_array_equal(m.Q, m.dynamics_noise)
        np.testing.assert_array_equal(m.R, m.observation_noise)

    def test_rejects_non_square_dynamics(self):
        with pytest.raises(ValueError, match="square"):
            LinearGaussianModel(
                **_valid_kwargs(dynamics_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            )

    def test_rejects_observation_matrix_wrong_columns(self):
        with pytest.raises(ValueError, match="columns"):
            LinearGaussianModel(**_valid_kwargs(observation_matrix=[[1.0, 0.0, 0.0]]))

    def test_rejects_dynamics_noise_wrong_size(self):
        with pytest.raises(ValueError, match="dynamics_noise"):
            LinearGaussianModel(**_valid_kwargs(dynamics_noise=[[1.0]]))

    def test_rejects_observation_noise_wrong_size(self):
        with pytest.raises(ValueError, match="observation_noise"):
            LinearGaussianModel(
                **_valid_kwargs(observation_noise=[[1.0, 0.0], [0.0, 1.0]])
            )

    def test_rejects_indefinite_dynamics_noise(self):
        with pytest.raises(ValueError, match="positive-semi-definite"):
            LinearGaussianModel(
                **_valid_kwargs(dynamics_noise=[[-1.0, 0.0], [0.0, -1.0]])
            )

    def test_rejects_negative_observation_noise(self):
        # observation_noise is positive-DEFINITE (not just PSD): the epistemic term
        # inverts
        # it, so a negative or zero R is rejected.
        with pytest.raises(ValueError, match="positive-definite"):
            LinearGaussianModel(**_valid_kwargs(observation_noise=[[-0.5]]))

    def test_rejects_noiseless_observation_noise(self):
        # A noiseless R=0 sends the EFE information gain to +inf and silently
        # collapses action selection — reject it as not positive-definite.
        with pytest.raises(
            ValueError,
            match="positive-definite because a noiseless sensor sends the information "
            "gain to infinity",
        ):
            LinearGaussianModel(**_valid_kwargs(observation_noise=[[0.0]]))

    def test_rejects_control_wrong_rows(self):
        with pytest.raises(ValueError, match="control"):
            LinearGaussianModel(
                **_valid_kwargs(control_matrix=[[0.0], [0.0], [0.0]])
            )  # 3x1

    def test_rejects_prior_wrong_dimension(self):
        bad_prior = Belief(mean=[0.0, 0.0, 0.0], cov=np.eye(3))  # 3-D, but n=2
        with pytest.raises(ValueError, match="prior"):
            LinearGaussianModel(**_valid_kwargs(prior=bad_prior))

    def test_rejects_prior_not_a_belief(self):
        with pytest.raises(TypeError, match="Belief"):
            LinearGaussianModel(**_valid_kwargs(prior=[0.0, 0.0]))

    def test_observation_defaults_to_none(self):
        # No observation given -> fixed sensor defined by the observation_matrix
        # and observation_noise fields
        # (the v0.2 semantics). None is the canonical "fixed" case.
        assert LinearGaussianModel(**_valid_kwargs()).observation_model is None

    def test_accepts_an_observation_model(self):
        sensor = FixedSensor([[1.0, 0.0]], observation_noise=[[1.0]])
        m = LinearGaussianModel(**_valid_kwargs(observation_model=sensor))
        assert m.observation_model is sensor

    def test_rejects_observation_not_an_observation_model(self):
        with pytest.raises(TypeError, match="ObservationModel"):
            LinearGaussianModel(**_valid_kwargs(observation_model="not a sensor"))


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
        assert restored.control_matrix is None
        np.testing.assert_array_equal(restored.dynamics_matrix, m.dynamics_matrix)
        np.testing.assert_array_equal(restored.prior.mean, m.prior.mean)

    def test_model_round_trips_with_control(self):
        m = LinearGaussianModel(**_valid_kwargs(control_matrix=[[0.0], [1.0]]))
        leaves, treedef = jax.tree_util.tree_flatten(m)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        np.testing.assert_array_equal(restored.control_matrix, m.control_matrix)

    def test_model_round_trips_with_an_observation(self):
        # observation is a nullable child like control: a FixedSensor recurses
        # into its own array leaves and is rebuilt on unflatten.
        sensor = FixedSensor([[1.0, 0.0]], observation_noise=[[1.0]])
        m = LinearGaussianModel(**_valid_kwargs(observation_model=sensor))
        leaves, treedef = jax.tree_util.tree_flatten(m)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(restored.observation_model, FixedSensor)
        c_out, r_out = restored.observation_model.linearize(jnp.zeros(2))
        np.testing.assert_array_equal(c_out, sensor.observation_matrix)
        np.testing.assert_array_equal(r_out, sensor.observation_noise)


def _flat_noise(mean, params):
    return jnp.array([[1.0]])


class _AgreeingFixedProcessNoise:
    """A DynamicsNoise whose constant equals the baseline model's plain field."""

    is_fixed = True

    def noise_at(self, x):
        return jnp.array([[0.1, 0.0], [0.0, 0.1]])


class _DisagreeingFixedProcessNoise:
    is_fixed = True

    def noise_at(self, x):
        return jnp.array([[0.9, 0.0], [0.0, 0.9]])


class TestACarriedModelObjectAgreesWithThePlainFields:
    """One model, one story: an object restating a plain field must restate it exactly.

    The consumers split otherwise. The Kalman filter and the world read the plain fields
    when the object says it is fixed, while the EFE kernel reads the object whenever one
    is present, so a disagreement changes the planner's numbers and nobody's else's.
    """

    def test_a_fixed_sensor_with_a_different_noise_does_not_construct(self):
        sensor = FixedSensor([[1.0, 0.0]], observation_noise=[[0.5]])
        with pytest.raises(ValueError, match="observation_noise"):
            LinearGaussianModel(**_valid_kwargs(observation_model=sensor))

    def test_a_fixed_sensor_with_a_different_matrix_does_not_construct(self):
        sensor = FixedSensor([[0.0, 1.0]], observation_noise=[[1.0]])
        with pytest.raises(ValueError, match="observation_matrix"):
            LinearGaussianModel(**_valid_kwargs(observation_model=sensor))

    def test_an_agreeing_fixed_sensor_constructs(self):
        sensor = FixedSensor([[1.0, 0.0]], observation_noise=[[1.0]])
        assert (
            LinearGaussianModel(
                **_valid_kwargs(observation_model=sensor)
            ).observation_model
            is sensor
        )

    def test_a_callable_sensor_with_a_different_matrix_does_not_construct(self):
        sensor = CallableSensor([[0.0, 1.0]], _flat_noise, ())
        with pytest.raises(ValueError, match="observation_matrix"):
            LinearGaussianModel(**_valid_kwargs(observation_model=sensor))

    def test_a_callable_sensor_with_a_wrong_shaped_noise_does_not_construct(self):
        def wide(mean, params):
            return jnp.eye(2)

        # CallableSensor probes its own noise_fn and refuses this before the model is
        # even asked. The model's probe is for a protocol implementation that does not
        # self-check, so this pins the earliest of the two guards.
        with pytest.raises(ValueError, match=r"\(1, 1\)"):
            LinearGaussianModel(
                **_valid_kwargs(
                    observation_model=CallableSensor([[1.0, 0.0]], wide, ())
                )
            )

    def test_an_agreeing_callable_sensor_constructs(self):
        sensor = CallableSensor([[1.0, 0.0]], _flat_noise, ())
        LinearGaussianModel(**_valid_kwargs(observation_model=sensor))

    def test_a_fixed_process_noise_with_a_different_value_does_not_construct(self):
        with pytest.raises(ValueError, match="dynamics_noise"):
            LinearGaussianModel(
                **_valid_kwargs(dynamics_noise_model=_DisagreeingFixedProcessNoise())
            )

    def test_an_agreeing_fixed_process_noise_constructs(self):
        LinearGaussianModel(
            **_valid_kwargs(dynamics_noise_model=_AgreeingFixedProcessNoise())
        )


def _rising_noise(mean, params):
    return jnp.array([[1.0 + mean[0] ** 2]])


# R(x) is 5 here and 1 at the origin, where CallableSensor probes itself, so a value
# read at the wrong point shows.
_OFF_ORIGIN_PRIOR = Belief(mean=[2.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]])


def _without_observation_noise(**overrides):
    kwargs = _valid_kwargs(**overrides)
    del kwargs["observation_noise"]
    return kwargs


class TestAnOmittedObservationNoiseIsReadFromTheSensor:
    """A sensor states the channel once. The plain field is its R at the prior mean."""

    def test_a_state_dependent_sensor_supplies_its_noise_at_the_prior_mean(self):
        sensor = CallableSensor([[1.0, 0.0]], _rising_noise, ())
        model = LinearGaussianModel(
            **_without_observation_noise(
                observation_model=sensor, prior=_OFF_ORIGIN_PRIOR
            )
        )
        np.testing.assert_array_equal(model.observation_noise, [[5.0]])

    def test_omitting_it_builds_the_model_that_passing_it_at_the_prior_mean_does(self):
        sensor = CallableSensor([[1.0, 0.0]], _rising_noise, ())
        omitted = LinearGaussianModel(
            **_without_observation_noise(
                observation_model=sensor, prior=_OFF_ORIGIN_PRIOR
            )
        )
        passed = LinearGaussianModel(
            **_valid_kwargs(
                observation_model=sensor,
                prior=_OFF_ORIGIN_PRIOR,
                observation_noise=[[5.0]],
            )
        )
        omitted_leaves, omitted_structure = jax.tree_util.tree_flatten(omitted)
        passed_leaves, passed_structure = jax.tree_util.tree_flatten(passed)
        assert omitted_structure == passed_structure
        for a, b in zip(omitted_leaves, passed_leaves, strict=True):
            np.testing.assert_array_equal(a, b)

    def test_the_derived_noise_survives_a_flatten_unflatten_round_trip(self):
        sensor = CallableSensor([[1.0, 0.0]], _rising_noise, ())
        model = LinearGaussianModel(
            **_without_observation_noise(
                observation_model=sensor, prior=_OFF_ORIGIN_PRIOR
            )
        )
        leaves, treedef = jax.tree_util.tree_flatten(model)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        np.testing.assert_array_equal(restored.observation_noise, [[5.0]])

    def test_a_fixed_sensor_supplies_its_own_noise(self):
        sensor = FixedSensor([[1.0, 0.0]], observation_noise=[[0.5]])
        model = LinearGaussianModel(
            **_without_observation_noise(observation_model=sensor)
        )
        np.testing.assert_array_equal(model.observation_noise, [[0.5]])

    def test_with_no_sensor_it_is_still_required(self):
        with pytest.raises(TypeError, match="observation_noise"):
            LinearGaussianModel(**_without_observation_noise())

    def test_passed_beside_a_state_dependent_sensor_it_is_kept_and_not_warned_on(self):
        # The field is then the nominal R a fixed-noise consumer substitutes. One taken
        # somewhere other than the prior mean is a choice, not a disagreement.
        sensor = CallableSensor([[1.0, 0.0]], _rising_noise, ())
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model = LinearGaussianModel(
                **_valid_kwargs(observation_model=sensor, prior=_OFF_ORIGIN_PRIOR)
            )
        np.testing.assert_array_equal(model.observation_noise, [[1.0]])

    def test_a_sensor_of_the_wrong_kind_is_named_rather_than_asked_for_a_noise(self):
        with pytest.raises(TypeError, match="ObservationModel"):
            LinearGaussianModel(
                **_without_observation_noise(observation_model="not a sensor")
            )

    def test_a_prior_of_the_wrong_kind_is_named_rather_than_probed(self):
        sensor = CallableSensor([[1.0, 0.0]], _rising_noise, ())
        with pytest.raises(TypeError, match="Belief"):
            LinearGaussianModel(
                **_without_observation_noise(observation_model=sensor, prior=[2.0, 0.0])
            )


class TestTheObservationNoiseAtAState:
    """``observation_noise_at`` is the one read of R at a state, with or without a
    sensor, so no consumer branches on which kind of sensor the model carries.
    """

    def test_with_no_sensor_it_is_the_declared_noise_everywhere(self):
        model = LinearGaussianModel(**_valid_kwargs())
        np.testing.assert_array_equal(model.observation_noise_at([3.0, -1.0]), [[1.0]])

    def test_a_fixed_sensor_gives_its_own_noise_everywhere(self):
        sensor = FixedSensor([[1.0, 0.0]], observation_noise=[[0.5]])
        model = LinearGaussianModel(
            **_without_observation_noise(observation_model=sensor)
        )
        np.testing.assert_array_equal(model.observation_noise_at([3.0, -1.0]), [[0.5]])

    def test_a_state_dependent_sensor_is_read_at_the_state_asked_about(self):
        sensor = CallableSensor([[1.0, 0.0]], _rising_noise, ())
        model = LinearGaussianModel(
            **_without_observation_noise(
                observation_model=sensor, prior=_OFF_ORIGIN_PRIOR
            )
        )
        np.testing.assert_allclose(model.observation_noise_at([3.0, 0.0]), [[10.0]])
        np.testing.assert_allclose(model.observation_noise_at([0.0, 0.0]), [[1.0]])

    def test_it_traces_under_jit(self):
        sensor = CallableSensor([[1.0, 0.0]], _rising_noise, ())
        model = LinearGaussianModel(
            **_without_observation_noise(observation_model=sensor)
        )
        noise_at = jax.jit(model.observation_noise_at)
        np.testing.assert_allclose(noise_at(jnp.array([3.0, 0.0])), [[10.0]])


def _rising_process_noise(x, params):
    return jnp.diag(jnp.array([1.0 + x[0] ** 2, 0.1]))


def _without_dynamics_noise(**overrides):
    kwargs = _valid_kwargs(**overrides)
    del kwargs["dynamics_noise"]
    return kwargs


class TestAnOmittedDynamicsNoiseIsReadFromTheProcessNoise:
    """The same rule for Q. The plain field is the process Q at the prior mean."""

    def test_a_state_dependent_process_noise_supplies_its_q_at_the_prior_mean(self):
        process = CallableProcessNoise(_rising_process_noise, ())
        model = LinearGaussianModel(
            **_without_dynamics_noise(
                dynamics_noise_model=process, prior=_OFF_ORIGIN_PRIOR
            )
        )
        np.testing.assert_array_equal(model.dynamics_noise, [[5.0, 0.0], [0.0, 0.1]])

    def test_omitting_it_builds_the_model_that_passing_it_at_the_prior_mean_does(self):
        process = CallableProcessNoise(_rising_process_noise, ())
        omitted = LinearGaussianModel(
            **_without_dynamics_noise(
                dynamics_noise_model=process, prior=_OFF_ORIGIN_PRIOR
            )
        )
        passed = LinearGaussianModel(
            **_valid_kwargs(
                dynamics_noise_model=process,
                prior=_OFF_ORIGIN_PRIOR,
                dynamics_noise=[[5.0, 0.0], [0.0, 0.1]],
            )
        )
        omitted_leaves, omitted_structure = jax.tree_util.tree_flatten(omitted)
        passed_leaves, passed_structure = jax.tree_util.tree_flatten(passed)
        assert omitted_structure == passed_structure
        for a, b in zip(omitted_leaves, passed_leaves, strict=True):
            np.testing.assert_array_equal(a, b)

    def test_a_fixed_process_noise_supplies_its_own_q(self):
        model = LinearGaussianModel(
            **_without_dynamics_noise(dynamics_noise_model=_AgreeingFixedProcessNoise())
        )
        np.testing.assert_array_equal(model.dynamics_noise, [[0.1, 0.0], [0.0, 0.1]])

    def test_with_no_process_noise_object_it_is_still_required(self):
        with pytest.raises(TypeError, match="dynamics_noise"):
            LinearGaussianModel(**_without_dynamics_noise())

    def test_passed_beside_a_state_dependent_q_it_is_kept_and_not_warned_on(self):
        process = CallableProcessNoise(_rising_process_noise, ())
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model = LinearGaussianModel(
                **_valid_kwargs(dynamics_noise_model=process, prior=_OFF_ORIGIN_PRIOR)
            )
        np.testing.assert_array_equal(model.dynamics_noise, [[0.1, 0.0], [0.0, 0.1]])

    def test_a_process_noise_of_the_wrong_kind_is_named_rather_than_asked_for_a_q(self):
        with pytest.raises(TypeError, match="DynamicsNoise"):
            LinearGaussianModel(
                **_without_dynamics_noise(dynamics_noise_model="not a noise")
            )

    def test_both_noises_can_be_omitted_together(self):
        kwargs = _without_dynamics_noise(
            observation_model=CallableSensor([[1.0, 0.0]], _rising_noise, ()),
            dynamics_noise_model=CallableProcessNoise(_rising_process_noise, ()),
            prior=_OFF_ORIGIN_PRIOR,
        )
        del kwargs["observation_noise"]
        model = LinearGaussianModel(**kwargs)
        np.testing.assert_array_equal(model.observation_noise, [[5.0]])
        np.testing.assert_array_equal(model.dynamics_noise, [[5.0, 0.0], [0.0, 0.1]])
