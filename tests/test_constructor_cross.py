"""The cross of the two declared axes, and the certificate that it was enumerated.

Every separation Paper 2 reports is a comparison between two cells of one cross. What
makes that a decision rather than a survey is that the cross is finite, declared, and
visited in full. `|model axis| x |inference axis|` is a product, so the certificate
carries both axes and both versions: 12 is 3 x 4 and 2 x 6, and a bare count says which
neither.

The count is asserted rather than assumed. A cell dropped by a filter leaves a shorter
run in which every remaining cell still passes, and nothing but the count says so.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp._lab.constructors import (
    CORRECT,
    EXACT_INFERENCE,
    ConstructorSet,
    InferenceKind,
    InferenceRule,
    InferenceSet,
    ModelSpec,
    Perturbation,
)
from cpomdp._lab.scoring import build_cross
from cpomdp.enumeration import IncompleteEnumerationError
from cpomdp.observation import CallableSensor
from warrantlib import ProductCompletenessCertificate, Warrant

DT = 0.1


def _spec(version: str = "spec-v1") -> ModelSpec:
    return ModelSpec(
        dynamics_matrix=[[1.0, DT], [0.0, 1.0]],
        observation_matrix=[[1.0, 0.0]],
        dynamics_noise=[[1e-4, 0.0], [0.0, 1e-4]],
        observation_noise=[[1e-2]],
        prior_mean=[0.0, 0.0],
        prior_cov=[[1.0, 0.0], [0.0, 1.0]],
        control_matrix=[[0.0], [DT]],
        version=version,
    )


def _models(version: str = "models-v1") -> ConstructorSet:
    return ConstructorSet(
        (
            CORRECT,
            Perturbation("noisy_sensor", "observation_noise", 0.5),
            Perturbation("noisy_process", "dynamics_noise", 0.5),
        ),
        version=version,
    )


def _rules(version: str = "rules-v1") -> InferenceSet:
    return InferenceSet(
        (
            EXACT_INFERENCE,
            InferenceRule("frozen_gain", InferenceKind.FROZEN_GAIN),
            InferenceRule("wrong_r", InferenceKind.WRONG_FIXED_R, magnitude=0.5),
            InferenceRule("diagonal", InferenceKind.DIAGONAL_COVARIANCE_ONLY),
        ),
        version=version,
    )


def test_the_cross_visits_the_product_of_the_two_axes():
    models, rules = _models(), _rules()
    cross = build_cross(_spec(), models, rules)
    assert len(cross.cells) == models.size * rules.size == 12


def test_the_cells_run_model_major_in_declaration_order():
    models, rules = _models(), _rules()
    cross = build_cross(_spec(), models, rules)
    assert tuple(
        (cell.model_name, cell.inference_name) for cell in cross.cells
    ) == tuple(
        (model_name, rule_name)
        for model_name in models.names
        for rule_name in rules.names
    )


def test_every_cell_infers_over_the_model_its_own_row_built():
    cross = build_cross(_spec(), _models(), _rules())
    for cell in cross.cells:
        assert cell.inference_backend.model is cell.model


def test_a_row_shares_one_model_across_its_rules():
    # The axes are independent: one perturbation is one model, inferred four ways. A
    # model rebuilt per cell would make the inference axis look like a model axis under
    # any test comparing arrays by identity.
    cross = build_cross(_spec(), _models(), _rules())
    first_row = [cell for cell in cross.cells if cell.model_name == "correct"]
    assert len({id(cell.model) for cell in first_row}) == 1


def test_rows_do_not_share_a_model():
    cross = build_cross(_spec(), _models(), _rules())
    per_row = {cell.model_name: id(cell.model) for cell in cross.cells}
    assert len(set(per_row.values())) == len(per_row)


def test_the_perturbation_reaches_the_model_it_names():
    cross = build_cross(_spec(), _models(), _rules())
    by_row = {cell.model_name: cell.model for cell in cross.cells}
    assert np.allclose(np.asarray(by_row["correct"].observation_noise), [[1e-2]])
    assert np.allclose(np.asarray(by_row["noisy_sensor"].observation_noise), [[1.5e-2]])


# --- the certificate ----------------------------------------------------------------


def test_the_certificate_decides_the_cross():
    cross = build_cross(_spec(), _models(), _rules())
    certificate = cross.certificate
    assert isinstance(certificate, ProductCompletenessCertificate)
    assert certificate.warrant is Warrant.PROVED
    assert certificate.expected == certificate.visited == 12
    assert certificate.domain_declared
    assert certificate.complete


def test_the_certificate_names_both_axes_and_both_versions():
    # A bare 12 is 3 x 4 and 2 x 6. Which axes were crossed, and which version of each,
    # is what makes two certificates over different crosses tellable apart.
    cross = build_cross(_spec(), _models(version="m-v9"), _rules(version="r-v2"))
    axes = {axis.name: axis for axis in cross.certificate.axes}
    assert set(axes) == {"model", "inference"}
    assert (axes["model"].size, axes["model"].version) == (3, "m-v9")
    assert (axes["inference"].size, axes["inference"].version) == (4, "r-v2")


def test_a_cross_that_visits_fewer_cells_than_it_declared_is_refused(monkeypatch):
    # The count is a real loop carry, not `len(cells)`, so a cell lost on the way out is
    # a failure here rather than a shorter run in which everything still passes.
    models = _models()
    monkeypatch.setattr(type(models), "size", property(lambda self: 4))
    with pytest.raises(IncompleteEnumerationError, match="sample"):
        build_cross(_spec(), models, _rules())


def test_the_error_names_both_declared_versions(monkeypatch):
    models = _models(version="m-v9")
    monkeypatch.setattr(type(models), "size", property(lambda self: 4))
    with pytest.raises(IncompleteEnumerationError, match="m-v9"):
        build_cross(_spec(), models, _rules(version="r-v2"))


def test_a_one_by_one_cross_is_a_legal_product():
    # A ladder or a seed list is a product of one axis. Nothing here needs two.
    models = ConstructorSet((CORRECT,), version="m-v1")
    rules = InferenceSet((EXACT_INFERENCE,), version="r-v1")
    cross = build_cross(_spec(), models, rules)
    assert cross.certificate.expected == 1
    assert cross.certificate.warrant is Warrant.PROVED


# --- a cell that cannot be built -----------------------------------------------------


def _state_dependent_spec() -> ModelSpec:
    return ModelSpec(
        dynamics_matrix=[[1.0, DT], [0.0, 1.0]],
        dynamics_noise=[[1e-4, 0.0], [0.0, 1e-4]],
        prior_mean=[0.0, 0.0],
        prior_cov=[[1.0, 0.0], [0.0, 1.0]],
        control_matrix=[[0.0], [DT]],
        observation_model=CallableSensor(
            observation_matrix=[[1.0, 0.0]],
            noise_fn=lambda x, _p: jnp.array([[1e-2 * (1 + x[0] ** 2)]]),
            noise_params=(),
        ),
        version="spec-rx",
    )


def test_a_cell_that_cannot_be_built_names_its_row():
    # A frozen gain has no constant fixed point under state-dependent noise. The axes
    # name the rule that refused and cannot name the row, since they were handed a model
    # rather than the perturbation that produced one. On a cross that is half the
    # address, and the caller has a dozen cells to search.
    rules = InferenceSet(
        (EXACT_INFERENCE, InferenceRule("frozen_gain", InferenceKind.FROZEN_GAIN)),
        version="rules-v1",
    )
    models = ConstructorSet(
        (CORRECT, Perturbation("noisy_process", "dynamics_noise", 0.5)),
        version="models-v1",
    )
    with pytest.raises(ValueError, match="row 'correct'"):
        build_cross(_state_dependent_spec(), models, rules)


def test_the_refusal_keeps_the_type_the_axis_raised():
    # `except ValueError` at a call site has to keep working, and a RuntimeError from a
    # recursion that did not converge is a different failure from a refused model.
    rules = InferenceSet(
        (EXACT_INFERENCE, InferenceRule("frozen_gain", InferenceKind.FROZEN_GAIN)),
        version="rules-v1",
    )
    with pytest.raises(ValueError, match="row") as refused:
        build_cross(
            _state_dependent_spec(),
            ConstructorSet((CORRECT,), version="models-v1"),
            rules,
        )
    assert isinstance(refused.value.__cause__, ValueError)


def test_each_cell_knows_whether_its_axes_sit_at_the_reference():
    cross = build_cross(_spec(), _models(), _rules())
    flags = {
        (cell.model_name, cell.inference_name): (
            cell.model_is_correct,
            cell.inference_is_exact,
        )
        for cell in cross.cells
    }
    assert flags[("correct", "exact")] == (True, True)
    assert flags[("noisy_sensor", "exact")] == (False, True)
    assert flags[("correct", "frozen_gain")] == (True, False)
    assert flags[("noisy_process", "diagonal")] == (False, False)
