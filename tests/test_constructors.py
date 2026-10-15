import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import ArrayLike

from cpomdp._lab.constructors import CORRECT, ConstructorSet, ModelSpec, Perturbation
from cpomdp.dynamics import CallableProcessNoise
from cpomdp.observation import CallableSensor, FixedSensor
from cpomdp.types import LinearGaussianModel

DT = 0.1
CONTROL = [[0.0], [DT]]


def _spec(
    *,
    control_matrix: ArrayLike | None = CONTROL,
    version: str = "spec-v1",
    observation_model=None,
    dynamics_noise_model=None,
) -> ModelSpec:
    return ModelSpec(
        dynamics_matrix=[[1.0, DT], [0.0, 1.0]],
        observation_matrix=[[1.0, 0.0]],
        dynamics_noise=[[1e-4, 0.0], [0.0, 1e-4]],
        observation_noise=[[1e-2]],
        prior_mean=[0.0, 0.0],
        prior_cov=[[1.0, 0.0], [0.0, 1.0]],
        control_matrix=control_matrix,
        observation_model=observation_model,
        dynamics_noise_model=dynamics_noise_model,
        version=version,
    )


def _leaves(model: LinearGaussianModel) -> set[int]:
    return {
        id(model.dynamics_matrix),
        id(model.observation_matrix),
        id(model.dynamics_noise),
        id(model.observation_noise),
        id(model.prior.mean),
        id(model.prior.cov),
    }


# --- the spec builds models, and every build is its own object ----------------------


def test_build_returns_a_model_carrying_the_declared_parameters():
    model = _spec().build()
    assert isinstance(model, LinearGaussianModel)
    assert np.allclose(np.asarray(model.dynamics_matrix), [[1.0, DT], [0.0, 1.0]])
    assert np.allclose(np.asarray(model.observation_noise), [[1e-2]])
    assert np.allclose(np.asarray(model.prior.mean), [0.0, 0.0])


def test_two_builds_are_different_objects():
    spec = _spec()
    assert spec.build() is not spec.build()


def test_two_builds_share_no_array():
    spec = _spec()
    left, right = spec.build(), spec.build()  # both held: a freed id can be reused
    assert not _leaves(left) & _leaves(right)


def test_two_builds_are_equal_by_value():
    spec = _spec()
    left, right = spec.build(), spec.build()
    assert np.array_equal(np.asarray(left.dynamics_matrix), np.asarray(right.A))
    assert np.array_equal(np.asarray(left.observation_noise), np.asarray(right.R))


def test_a_control_free_spec_builds_a_control_free_model():
    model = _spec(control_matrix=None).build()
    assert model.control_matrix is None
    assert model.n_controls == 0


def test_the_spec_needs_a_version():
    with pytest.raises(ValueError, match="declared and versioned"):
        _spec(version="")


# --- perturbations are data, applied as a relative change ---------------------------


def test_correct_changes_nothing():
    spec = _spec()
    plain, corrected = spec.build(), spec.build(CORRECT)
    assert np.array_equal(
        np.asarray(plain.dynamics_matrix), np.asarray(corrected.dynamics_matrix)
    )


def test_a_perturbation_scales_the_named_parameter():
    spec = _spec()
    perturbed = spec.build(Perturbation("noisy", "observation_noise", 0.5))
    assert np.allclose(np.asarray(perturbed.observation_noise), [[1.5e-2]])


def test_a_perturbation_leaves_every_other_parameter_alone():
    spec = _spec()
    perturbed = spec.build(Perturbation("noisy", "observation_noise", 0.5))
    plain = spec.build()
    assert np.array_equal(
        np.asarray(perturbed.dynamics_matrix), np.asarray(plain.dynamics_matrix)
    )
    assert np.array_equal(np.asarray(perturbed.prior.cov), np.asarray(plain.prior.cov))


def test_a_zero_magnitude_on_a_named_axis_is_the_identity():
    spec = _spec()
    perturbed = spec.build(Perturbation("flat", "dynamics_matrix", 0.0))
    assert np.array_equal(
        np.asarray(perturbed.dynamics_matrix), np.asarray(spec.build().A)
    )


def test_a_control_matrix_perturbation_applies():
    spec = _spec()
    perturbed = spec.build(Perturbation("weak", "control_matrix", -0.5))
    assert np.allclose(np.asarray(perturbed.control_matrix), [[0.0], [DT * 0.5]])


def test_perturbing_an_absent_control_matrix_does_not_build():
    spec = _spec(control_matrix=None)
    with pytest.raises(ValueError, match="no control_matrix"):
        spec.build(Perturbation("weak", "control_matrix", -0.5))


def test_an_unknown_parameter_does_not_construct():
    with pytest.raises(ValueError, match="not a perturbable parameter"):
        Perturbation("odd", "prior_mean", 0.5)


def test_a_perturbation_needs_a_name():
    with pytest.raises(ValueError, match="needs a name"):
        Perturbation("", "dynamics_noise", 0.5)


def test_a_named_parameter_needs_a_magnitude_that_leaves_a_covariance_valid():
    spec = _spec()
    with pytest.raises(ValueError, match="observation_noise"):
        spec.build(Perturbation("void", "observation_noise", -1.0))


def test_correct_names_no_parameter():
    assert CORRECT.parameter is None
    assert CORRECT.magnitude == 0.0


def test_an_unperturbed_entry_may_not_name_a_magnitude():
    with pytest.raises(ValueError, match="perturbs nothing"):
        Perturbation("odd", None, 0.5)


# --- the set is declared, versioned, and pinned by a test ---------------------------


def test_a_set_reports_its_names_in_declaration_order():
    declared = ConstructorSet(
        (CORRECT, Perturbation("noisy", "observation_noise", 0.5)), version="cross-v1"
    )
    assert declared.names == ("correct", "noisy")
    assert declared.size == 2


def test_a_set_needs_a_version():
    with pytest.raises(ValueError, match="declared and versioned"):
        ConstructorSet((CORRECT,), version="")


def test_a_set_refuses_a_duplicate_name():
    with pytest.raises(ValueError, match="duplicate"):
        ConstructorSet(
            (
                CORRECT,
                Perturbation("noisy", "observation_noise", 0.5),
                Perturbation("noisy", "dynamics_noise", 0.5),
            ),
            version="cross-v1",
        )


def test_a_set_with_nothing_unperturbed_does_not_construct():
    with pytest.raises(ValueError, match="unperturbed"):
        ConstructorSet(
            (Perturbation("noisy", "observation_noise", 0.5),), version="cross-v1"
        )


def test_building_a_set_pairs_every_name_with_its_model():
    declared = ConstructorSet(
        (CORRECT, Perturbation("noisy", "observation_noise", 0.5)), version="cross-v1"
    )
    built = declared.build_all(_spec())
    assert tuple(name for name, _ in built) == declared.names
    assert np.allclose(np.asarray(built[0][1].observation_noise), [[1e-2]])
    assert np.allclose(np.asarray(built[1][1].observation_noise), [[1.5e-2]])


def test_every_model_a_set_builds_is_its_own_object():
    declared = ConstructorSet(
        (CORRECT, Perturbation("same", "observation_noise", 0.0)), version="cross-v1"
    )
    left, right = (model for _, model in declared.build_all(_spec()))
    assert left is not right
    assert not _leaves(left) & _leaves(right)


# --- a scale the filter would not read is refused, not built ------------------------
#
# A state-dependent sensor supplies its own (C, R) and a state-dependent process noise
# its own Q, so scaling the matrix each stands in for reaches nothing. Left unguarded
# the cell builds what CORRECT builds and reports a label saying otherwise.


def _noise_at(mean, params):
    return jnp.array([[1e-2]]) * (1.0 + jnp.sum(jnp.asarray(mean) ** 2))


def _process_at(x, params):
    return jnp.eye(2) * 1e-3 * (1.0 + jnp.sum(jnp.asarray(x) ** 2))


class _FixedProcessNoise:
    """A DynamicsNoise that does not vary; nothing shipped implements the fixed case."""

    is_fixed = True

    def noise_at(self, x):
        return jnp.eye(2) * 1e-3


@pytest.mark.parametrize("parameter", ["observation_matrix", "observation_noise"])
def test_a_scale_the_state_dependent_sensor_shadows_is_refused(parameter):
    spec = _spec(observation_model=CallableSensor([[1.0, 0.0]], _noise_at, ()))
    with pytest.raises(ValueError, match="does not read on this spec"):
        spec.build(Perturbation("shadowed", parameter, -0.5))


def test_a_scale_the_state_dependent_process_noise_shadows_is_refused():
    spec = _spec(dynamics_noise_model=CallableProcessNoise(_process_at, ()))
    with pytest.raises(ValueError, match="does not read on this spec"):
        spec.build(Perturbation("shadowed", "dynamics_noise", -0.5))


@pytest.mark.parametrize("parameter", ["dynamics_matrix", "control_matrix"])
def test_a_parameter_the_filter_still_reads_is_not_refused(parameter):
    spec = _spec(observation_model=CallableSensor([[1.0, 0.0]], _noise_at, ()))
    built = spec.build(Perturbation("read", parameter, -0.5))
    assert np.allclose(
        np.asarray(getattr(built, parameter)),
        np.asarray(getattr(_spec(), parameter)) * 0.5,
    )


def test_correct_still_builds_on_a_state_dependent_spec():
    spec = _spec(observation_model=CallableSensor([[1.0, 0.0]], _noise_at, ()))
    assert spec.build(CORRECT).observation_model is spec.observation_model


@pytest.mark.parametrize("parameter", ["observation_matrix", "observation_noise"])
def test_a_scale_a_fixed_sensor_restates_is_refused(parameter):
    # A fixed sensor restates the plain fields and the model refuses the two
    # disagreeing, so the scaled cell could never build. Refusing at the spec names
    # the situation instead of failing deep inside LinearGaussianModel.
    sensor = FixedSensor([[1.0, 0.0]], observation_noise=[[1e-2]])
    spec = _spec(observation_model=sensor)
    with pytest.raises(ValueError, match="restating"):
        spec.build(Perturbation("read", parameter, -0.5))


def test_a_scale_a_fixed_process_noise_restates_is_refused():
    spec = _spec(dynamics_noise_model=_FixedProcessNoise())
    with pytest.raises(ValueError, match="restating"):
        spec.build(Perturbation("read", "dynamics_noise", -0.5))


def test_correct_still_builds_with_an_agreeing_fixed_sensor():
    sensor = FixedSensor([[1.0, 0.0]], observation_noise=[[1e-2]])
    spec = _spec(observation_model=sensor)
    assert spec.build(CORRECT).observation_model is sensor


# --- a noise left to its object -------------------------------------------------------
#
# The model derives an omitted noise from its object at the prior mean. The spec holds
# the None and passes it through, so the rule lives in one place.


# Off the origin, where the model probes Q(x) itself. _noise_at and _process_at double
# here, so a noise read at the wrong point shows.
_OFF_ORIGIN = [1.0, 0.0]


def _spec_leaving(field: str, **noise_objects) -> ModelSpec:
    return ModelSpec(
        dynamics_matrix=[[1.0, DT], [0.0, 1.0]],
        observation_matrix=[[1.0, 0.0]],
        dynamics_noise=(
            None if field == "dynamics_noise" else [[1e-4, 0.0], [0.0, 1e-4]]
        ),
        observation_noise=None if field == "observation_noise" else [[1e-2]],
        prior_mean=_OFF_ORIGIN,
        prior_cov=[[1.0, 0.0], [0.0, 1.0]],
        control_matrix=CONTROL,
        version="spec-v1",
        **noise_objects,
    )


def test_a_spec_may_leave_the_observation_noise_to_its_sensor():
    sensor = CallableSensor([[1.0, 0.0]], _noise_at, ())
    spec = _spec_leaving("observation_noise", observation_model=sensor)
    assert spec.observation_noise is None
    np.testing.assert_allclose(spec.build().observation_noise, [[2e-2]])


def test_a_spec_may_leave_the_dynamics_noise_to_its_process_noise():
    process = CallableProcessNoise(_process_at, ())
    spec = _spec_leaving("dynamics_noise", dynamics_noise_model=process)
    assert spec.dynamics_noise is None
    np.testing.assert_allclose(spec.build().dynamics_noise, np.eye(2) * 2e-3)


@pytest.mark.parametrize("field", ["observation_noise", "dynamics_noise"])
def test_a_spec_leaving_a_noise_with_no_object_fails_at_its_first_build(field):
    spec = _spec_leaving(field)
    with pytest.raises(TypeError, match=f"{field} is required"):
        spec.build()


@pytest.mark.parametrize(
    ("field", "noise_objects"),
    [
        (
            "observation_noise",
            {"observation_model": CallableSensor([[1.0, 0.0]], _noise_at, ())},
        ),
        (
            "observation_noise",
            {
                "observation_model": FixedSensor(
                    [[1.0, 0.0]], observation_noise=[[1e-2]]
                )
            },
        ),
        (
            "dynamics_noise",
            {"dynamics_noise_model": CallableProcessNoise(_process_at, ())},
        ),
        ("dynamics_noise", {"dynamics_noise_model": _FixedProcessNoise()}),
    ],
    ids=["state-dependent-R", "fixed-R", "state-dependent-Q", "fixed-Q"],
)
def test_perturbing_a_noise_the_spec_leaves_out_does_not_build(field, noise_objects):
    spec = _spec_leaving(field, **noise_objects)
    with pytest.raises(ValueError, match=f"declares no {field}"):
        spec.build(Perturbation("noisy", field, 0.5))


# --- the spec compares and hashes rather than raising -------------------------------


def test_two_specs_compare_without_raising():
    assert (_spec() == _spec()) is False
    assert _spec() != _spec()


def test_a_spec_is_hashable():
    assert len({_spec(), _spec()}) == 2
