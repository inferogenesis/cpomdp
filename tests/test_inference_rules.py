import numpy as np
import pytest

from cpomdp._lab.constructors import (
    EXACT_INFERENCE,
    InferenceKind,
    InferenceRule,
    InferenceSet,
)
from cpomdp._lab.degraded import DiagonalCovarianceBackend, WrongFixedRBackend
from cpomdp.backends.kalman import KalmanBackend
from cpomdp.types import Belief, LinearGaussianModel

DT = 0.1


def _model() -> LinearGaussianModel:
    return LinearGaussianModel(
        dynamics_matrix=[[1.0, DT], [0.0, 1.0]],
        observation_matrix=[[1.0, 0.0]],
        dynamics_noise=[[1e-3, 0.0], [0.0, 1e-3]],
        observation_noise=[[1e-2]],
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        control_matrix=[[0.0], [DT]],
    )


def _posterior(backend, steps: int = 4) -> Belief:
    belief = backend.model.prior
    for step in range(steps):
        belief = backend.infer_states([0.1 * step], belief, [1.0])
    return belief


# --- the four declared kinds, and what each builds ----------------------------------


def test_the_declared_kinds_are_the_four_the_cross_names():
    assert {kind.value for kind in InferenceKind} == {
        "exact",
        "FrozenGain",
        "WrongFixedR",
        "DiagonalCovarianceOnly",
    }


def test_exact_builds_a_per_step_filter():
    backend = EXACT_INFERENCE.build(_model())
    assert isinstance(backend, KalmanBackend)
    assert backend.steady_state is False


def test_frozen_gain_builds_a_steady_state_filter():
    backend = InferenceRule("frozen", InferenceKind.FROZEN_GAIN).build(_model())
    assert isinstance(backend, KalmanBackend)
    assert backend.steady_state is True


def test_wrong_fixed_r_builds_the_substituted_filter_at_its_magnitude():
    model = _model()
    rule = InferenceRule("loud", InferenceKind.WRONG_FIXED_R, magnitude=0.5)
    built = rule.build(model)
    assert isinstance(built, WrongFixedRBackend)
    direct = WrongFixedRBackend(model, magnitude=0.5)
    assert np.allclose(
        np.asarray(_posterior(built).cov), np.asarray(_posterior(direct).cov), atol=0
    )


def test_diagonal_covariance_builds_the_wrapper():
    backend = InferenceRule("flat", InferenceKind.DIAGONAL_COVARIANCE_ONLY).build(
        _model()
    )
    assert isinstance(backend, DiagonalCovarianceBackend)
    assert float(_posterior(backend, steps=1).cov[0, 1]) == 0.0


@pytest.mark.parametrize("kind", list(InferenceKind))
def test_every_kind_builds_a_backend_over_the_model_it_was_given(kind):
    model = _model()
    rule = InferenceRule("cell", kind, magnitude=0.5 if _takes_magnitude(kind) else 0.0)
    assert rule.build(model).model is model


def _takes_magnitude(kind: InferenceKind) -> bool:
    return kind is InferenceKind.WRONG_FIXED_R


# --- a rule is data, and only one kind reads a magnitude ----------------------------


def test_exact_names_the_exact_kind_and_no_magnitude():
    assert EXACT_INFERENCE.kind is InferenceKind.EXACT
    assert EXACT_INFERENCE.magnitude == 0.0


def test_a_rule_needs_a_name():
    with pytest.raises(ValueError, match="needs a name"):
        InferenceRule("", InferenceKind.EXACT)


@pytest.mark.parametrize(
    "kind",
    [
        InferenceKind.EXACT,
        InferenceKind.FROZEN_GAIN,
        InferenceKind.DIAGONAL_COVARIANCE_ONLY,
    ],
)
def test_a_magnitude_on_a_kind_that_cannot_read_one_does_not_construct(kind):
    with pytest.raises(ValueError, match="reads no magnitude"):
        InferenceRule("odd", kind, magnitude=0.5)


def test_a_magnitude_that_empties_the_noise_fails_when_it_is_built():
    rule = InferenceRule("void", InferenceKind.WRONG_FIXED_R, magnitude=-1.0)
    with pytest.raises(ValueError, match="observation_noise"):
        rule.build(_model())


# --- the set is declared, versioned and pinned by a test ----------------------------


def test_a_set_reports_its_names_in_declaration_order():
    declared = InferenceSet(
        (EXACT_INFERENCE, InferenceRule("frozen", InferenceKind.FROZEN_GAIN)),
        version="inference-v1",
    )
    assert declared.names == ("exact", "frozen")
    assert declared.size == 2


def test_a_set_needs_a_version():
    with pytest.raises(ValueError, match="declared and versioned"):
        InferenceSet((EXACT_INFERENCE,), version="")


def test_a_set_refuses_a_duplicate_name():
    with pytest.raises(ValueError, match="duplicate"):
        InferenceSet(
            (
                EXACT_INFERENCE,
                InferenceRule("frozen", InferenceKind.FROZEN_GAIN),
                InferenceRule("frozen", InferenceKind.DIAGONAL_COVARIANCE_ONLY),
            ),
            version="inference-v1",
        )


def test_a_set_with_no_exact_rule_does_not_construct():
    with pytest.raises(ValueError, match="exact"):
        InferenceSet(
            (InferenceRule("frozen", InferenceKind.FROZEN_GAIN),),
            version="inference-v1",
        )


def test_building_a_set_pairs_every_name_with_its_backend():
    declared = InferenceSet(
        (
            EXACT_INFERENCE,
            InferenceRule("flat", InferenceKind.DIAGONAL_COVARIANCE_ONLY),
        ),
        version="inference-v1",
    )
    model = _model()
    built = declared.build_all(model)
    assert tuple(name for name, _ in built) == declared.names
    assert all(backend.model is model for _, backend in built)
    assert isinstance(built[0][1], KalmanBackend)
    assert isinstance(built[1][1], DiagonalCovarianceBackend)


def test_the_cells_of_a_set_do_not_agree_with_each_other():
    declared = InferenceSet(
        (
            EXACT_INFERENCE,
            InferenceRule("loud", InferenceKind.WRONG_FIXED_R, magnitude=2.0),
            InferenceRule("flat", InferenceKind.DIAGONAL_COVARIANCE_ONLY),
        ),
        version="inference-v1",
    )
    built = declared.build_all(_model())
    covariances = [np.asarray(_posterior(backend).cov) for _, backend in built]
    assert not np.allclose(covariances[0], covariances[1])
    assert not np.allclose(covariances[0], covariances[2])
    assert not np.allclose(covariances[1], covariances[2])
