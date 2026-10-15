"""Whether ``E[F] = H(p*) + D₁ + D₂`` closes, with both sides measured separately."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp._lab.additivity import (
    AdditivityCheck,
    AdditivityResidual,
    EntropyEstimate,
    GaussianEntropy,
    MonteCarloEntropy,
    variational_free_energy,
)
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
from cpomdp._lab.scoring import (
    build_cross,
    gaussian_kl,
    observation_predictive,
    predicted_belief,
)
from cpomdp.backends.kalman import KalmanBackend
from cpomdp.observation import CallableSensor
from cpomdp.types import Belief, LinearGaussianModel

DT = 0.1


def _spec() -> ModelSpec:
    return ModelSpec(
        dynamics_matrix=[[1.0, DT], [0.0, 1.0]],
        observation_matrix=[[1.0, 0.0]],
        dynamics_noise=[[1e-4, 0.0], [0.0, 1e-4]],
        observation_noise=[[1e-2]],
        prior_mean=[0.0, 0.0],
        prior_cov=[[1.0, 0.0], [0.0, 1.0]],
        control_matrix=[[0.0], [DT]],
        version="spec-v1",
    )


CROSS = build_cross(
    _spec(),
    ConstructorSet(
        (CORRECT, Perturbation("noisy_sensor", "observation_noise", 0.5)),
        version="models-v1",
    ),
    InferenceSet(
        (
            EXACT_INFERENCE,
            InferenceRule("wrong_r", InferenceKind.WRONG_FIXED_R, magnitude=0.5),
            InferenceRule("diagonal", InferenceKind.DIAGONAL_COVARIANCE_ONLY),
        ),
        version="rules-v1",
    ),
)
BELIEF = Belief(mean=[0.4, -0.2], cov=[[0.5, 0.1], [0.1, 0.3]])
ACTION = np.array([0.7])


def _cell(model_name, inference_name):
    (cell,) = [
        c
        for c in CROSS.cells
        if (c.model_name, c.inference_name) == (model_name, inference_name)
    ]
    return cell


def _log_predictive(model, belief, action, reading):
    mean, cov = observation_predictive(model, belief, action)
    shift = np.asarray(reading) - mean
    return -0.5 * (
        len(mean) * np.log(2 * np.pi)
        + np.linalg.slogdet(cov)[1]
        + shift @ np.linalg.solve(cov, shift)
    )


# --- the free energy, directly -----------------------------------------------------


@pytest.mark.parametrize("reading", [[0.1], [0.9], [-1.3]])
def test_an_exact_posterior_makes_the_free_energy_the_negative_log_evidence(reading):
    # F = −ln p(y) + KL(q ‖ p(x|y)), and the second term is zero for the exact
    # posterior. The negative log evidence is the predictive's density, computed
    # here without any term of F.
    model = _cell("correct", "exact").model
    posterior = KalmanBackend(model).infer_states(np.array(reading), BELIEF, ACTION)
    measured = variational_free_energy(
        posterior.mean,
        posterior.cov,
        model,
        predicted_belief(model, BELIEF, ACTION),
        reading,
    )
    assert measured == pytest.approx(
        -_log_predictive(model, BELIEF, ACTION, reading), rel=1e-12
    )


def test_a_degraded_posterior_adds_exactly_its_divergence_from_the_exact_one():
    cell = _cell("correct", "wrong_r")
    reading = np.array([0.6])
    degraded = cell.inference_backend.infer_states(reading, BELIEF, ACTION)
    exact = KalmanBackend(cell.model).infer_states(reading, BELIEF, ACTION)
    measured = variational_free_energy(
        degraded.mean,
        degraded.cov,
        cell.model,
        predicted_belief(cell.model, BELIEF, ACTION),
        reading,
    )
    expected = -_log_predictive(cell.model, BELIEF, ACTION, reading) + gaussian_kl(
        degraded.mean, degraded.cov, exact.mean, exact.cov
    )
    assert measured == pytest.approx(expected, rel=1e-12)


def test_the_free_energy_is_vectorised_over_readings():
    model = _cell("correct", "exact").model
    readings = np.array([[0.1], [0.9], [-1.3]])
    backend = KalmanBackend(model)
    posteriors = [backend.infer_states(r, BELIEF, ACTION) for r in readings]
    batched = variational_free_energy(
        np.stack([np.asarray(p.mean) for p in posteriors]),
        posteriors[0].cov,
        model,
        predicted_belief(model, BELIEF, ACTION),
        readings,
    )
    singles = [
        variational_free_energy(
            p.mean, p.cov, model, predicted_belief(model, BELIEF, ACTION), r
        )
        for p, r in zip(posteriors, readings, strict=True)
    ]
    np.testing.assert_allclose(batched, singles, rtol=1e-14)


def _rising_noise(mean, params):
    return jnp.array([[1e-2 * (1.0 + mean[0] ** 2)]])


def test_the_free_energy_refuses_a_state_dependent_sensor():
    # Under R(x) the plain field is the noise at one state, and the likelihood term
    # would price every reading against it.
    sensed = LinearGaussianModel(
        dynamics_matrix=[[1.0, DT], [0.0, 1.0]],
        observation_matrix=[[1.0, 0.0]],
        dynamics_noise=[[1e-4, 0.0], [0.0, 1e-4]],
        prior=BELIEF,
        control_matrix=[[0.0], [DT]],
        observation_model=CallableSensor([[1.0, 0.0]], _rising_noise, ()),
    )
    with pytest.raises(ValueError, match="observation_model is state-dependent"):
        variational_free_energy(BELIEF.mean, BELIEF.cov, sensed, BELIEF, [0.1])


# --- the entropy, estimated in one place --------------------------------------------


def test_the_closed_form_entropy_of_a_scalar_gaussian():
    estimate = GaussianEntropy().estimate([0.0], [[2.0]], jax.random.PRNGKey(0))
    assert estimate.value == pytest.approx(0.5 * np.log(2 * np.pi * np.e * 2.0))
    assert estimate.bar == 0.0


def test_the_sampling_estimator_agrees_with_the_closed_form_within_its_bar():
    mean, cov = np.array([0.3, -1.0]), np.array([[1.5, 0.2], [0.2, 0.8]])
    closed = GaussianEntropy().estimate(mean, cov, jax.random.PRNGKey(0))
    sampled = MonteCarloEntropy(samples=20_000).estimate(
        mean, cov, jax.random.PRNGKey(3)
    )
    assert sampled.bar > 0.0
    assert abs(sampled.value - closed.value) <= sampled.bar


# --- the four-term bound -------------------------------------------------------------


def _residual(**overrides) -> AdditivityResidual:
    terms = {
        "free_energy": 1.008,
        "free_energy_bar": 0.01,
        "entropy": 0.6,
        "entropy_bar": 0.0,
        "misspecification": 0.3,
        "misspecification_bar": 0.0,
        "inference_gap": 0.1,
        "inference_gap_bar": 0.0,
    }
    terms.update(overrides)
    return AdditivityResidual(**terms)


def test_the_bound_is_the_sum_of_all_four_bars():
    residual = _residual(
        free_energy_bar=0.01,
        entropy_bar=0.002,
        misspecification_bar=0.0003,
        inference_gap_bar=0.00004,
    )
    assert residual.bound == pytest.approx(0.01234)


def test_omitting_the_free_energy_bar_lets_a_real_failure_read_as_closure():
    # The worked case. A residual of 0.008 sits inside δ_F = 0.01 and the check
    # closes. Drop δ_F, as the three-term bound does, and the same residual has no
    # bar to sit inside and is a failure, though nothing about the measurement
    # changed. Under-bounding cuts the other way too: the same three-term bound
    # would pass a residual it has no evidence about, once any other bar exists.
    four_terms = _residual()
    assert four_terms.residual == pytest.approx(0.008)
    assert four_terms.closes
    three_terms = _residual(free_energy_bar=0.0, entropy_bar=0.005)
    assert three_terms.residual == pytest.approx(0.008)
    assert not three_terms.closes


# --- the check, on real cells --------------------------------------------------------


def _check_step(cell, check, key=0):
    truth = _spec().build()
    return check.check_step(
        truth,
        BELIEF,
        cell,
        BELIEF,
        BELIEF,
        ACTION,
        jax.random.PRNGKey(key),
    )


@pytest.mark.parametrize(
    "cell_name",
    [
        ("correct", "exact"),
        ("noisy_sensor", "exact"),
        ("correct", "diagonal"),
        ("noisy_sensor", "wrong_r"),
    ],
)
def test_the_accounting_closes_on_every_kind_of_cell(cell_name):
    check = AdditivityCheck(GaussianEntropy(), samples=20_000)
    residual = _check_step(_cell(*cell_name), check)
    assert residual.closes
    assert residual.free_energy_bar > 0.0
    assert residual.entropy_bar == 0.0
    assert residual.bound == residual.free_energy_bar


def test_the_check_closes_with_a_sampling_entropy_estimator_too():
    check = AdditivityCheck(MonteCarloEntropy(samples=20_000), samples=20_000)
    residual = _check_step(_cell("noisy_sensor", "wrong_r"), check)
    assert residual.closes
    assert residual.bound == pytest.approx(
        residual.free_energy_bar + residual.entropy_bar
    )


class _BiasedEntropy:
    """A closed form nudged by a twentieth of a nat, with a bar that admits nothing."""

    def estimate(self, mean, cov, key):
        exact = GaussianEntropy().estimate(mean, cov, key)
        return EntropyEstimate(value=exact.value + 0.05, bar=0.0)


def test_a_wrong_entropy_fails_to_close():
    # The check discriminates: with twenty thousand draws δ_F is well under the
    # bias, so a mis-estimated term reads as a residual outside the bound.
    check = AdditivityCheck(_BiasedEntropy(), samples=20_000)
    residual = _check_step(_cell("correct", "exact"), check)
    assert residual.free_energy_bar < 0.05
    assert not residual.closes
