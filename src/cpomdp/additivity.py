"""Whether the three-term accounting closes: ``E[F] = H(p*) + D₁ + D₂``.

The scoring harness returns the two divergences and never touches the entropy of
the true process. This module is the one place that entropy is estimated, and it
exists to check the identity the harness rests on rather than to score anything.

The check measures each side separately. The left side is the cell filter's
variational free energy, computed directly from the belief it holds after a reading
and the model's joint at that reading, then averaged over readings drawn from the
true predictive. The right side is the entropy from a supplied estimator plus the two
divergences the harness computes in closed form. Nothing on one side is derived from
the other, so agreement is evidence about the accounting and not a restatement of it.

The residual carries a four-term bound, ``δ_F + δ_H + δ₁ + δ₂``. The measured free
energy is itself an average under ``p*`` and carries its own bar. Leaving ``δ_F`` out
under-bounds the residual and lets a real failure of closure read as within tolerance
(``research/warrant_ledger.md`` section 4).
"""

import math
from dataclasses import dataclass
from typing import Protocol

import jax
import numpy as np
from jaxtyping import Array
from numpy.typing import ArrayLike

from cpomdp._validation import require_fixed_noise
from cpomdp.backends.kalman import KalmanBackend
from cpomdp.scoring import (
    CrossCell,
    StepUpdate,
    _affine_in_observation,
    inference_gap_step,
    misspecification_step,
    observation_predictive,
    predicted_belief,
)
from cpomdp.types import Belief, LinearGaussianModel

__all__ = [
    "AdditivityCheck",
    "AdditivityResidual",
    "EntropyEstimate",
    "EntropyEstimator",
    "GaussianEntropy",
    "MonteCarloEntropy",
    "variational_free_energy",
]


@dataclass(frozen=True)
class EntropyEstimate:
    """An entropy in nats and the bar it is known to.

    Args:
        value: The estimate.
        bar: How far the estimate may sit from the entropy. ``0.0`` for a closed form.
    """

    value: float
    bar: float


class EntropyEstimator(Protocol):
    """Anything that estimates the entropy of a Gaussian and says how well."""

    def estimate(self, mean: ArrayLike, cov: ArrayLike, key: Array) -> EntropyEstimate:
        """``H(N(mean, cov))`` in nats, with its bar.

        Args:
            mean: The Gaussian's mean, shape ``(m,)``.
            cov: Its covariance, shape ``(m, m)``.
            key: A JAX PRNG key, for an estimator that samples.
        """
        ...


@dataclass(frozen=True)
class GaussianEntropy:
    """The closed form ``½ · ln det(2πe · cov)``, with no bar."""

    def estimate(self, mean: ArrayLike, cov: ArrayLike, key: Array) -> EntropyEstimate:
        """The entropy of the Gaussian, exactly.

        Args:
            mean: The Gaussian's mean, unused. The entropy does not depend on it.
            cov: Its covariance, shape ``(m, m)``.
            key: Unused, since nothing is sampled.

        Returns:
            The entropy with a bar of ``0.0``.
        """
        cov = np.asarray(cov, dtype=float)
        dimension = cov.shape[0]
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise ValueError("the covariance must be positive definite")
        return EntropyEstimate(
            value=0.5 * (dimension * math.log(2 * math.pi * math.e) + float(logdet)),
            bar=0.0,
        )


@dataclass(frozen=True)
class MonteCarloEntropy:
    """``−mean log p(y_i)`` over draws from the Gaussian, with a standard-error bar.

    The estimator a check would have to use if ``p*`` had no closed form. It is here
    so that the closed form is not the only estimator ever wired in, and so the
    check's bar is exercised by something that has one.

    Args:
        samples: How many draws.
        sigma_multiplier: The bar is this many standard errors. A stated bar, not a
            proved one.
    """

    samples: int
    sigma_multiplier: float = 3.0

    def estimate(self, mean: ArrayLike, cov: ArrayLike, key: Array) -> EntropyEstimate:
        """The entropy by sampling, with a ``sigma_multiplier`` standard-error bar.

        Args:
            mean: The Gaussian's mean, shape ``(m,)``.
            cov: Its covariance, shape ``(m, m)``.
            key: A JAX PRNG key.

        Returns:
            The estimate and its bar.
        """
        mean = np.asarray(mean, dtype=float)
        cov = np.asarray(cov, dtype=float)
        draws = np.asarray(
            jax.random.multivariate_normal(key, mean, cov, shape=(self.samples,))
        )
        log_density = _gaussian_log_density(draws, mean, cov)
        return EntropyEstimate(
            value=float(-log_density.mean()),
            bar=self.sigma_multiplier
            * float(log_density.std(ddof=1) / math.sqrt(self.samples)),
        )


def variational_free_energy(
    posterior_mean: ArrayLike,
    posterior_cov: ArrayLike,
    model: LinearGaussianModel,
    prior: Belief,
    observation: ArrayLike,
) -> np.ndarray:
    """``F = E_q[ln q(x) − ln p(y | x) − ln p(x)]`` for a Gaussian ``q`` and joint.

    Computed term by term from the belief and the joint, never through the identity
    this module checks. ``q`` is ``N(posterior_mean, posterior_cov)``, ``p(x)`` is
    ``prior``, and ``p(y | x) = N(y; C · x, R)`` under ``model``. Every expectation
    is one of a Gaussian log-density under a Gaussian, so each is a trace and a
    quadratic form.

    Vectorised over readings: ``posterior_mean`` and ``observation`` may carry a
    leading batch axis, with one covariance shared by the batch.

    Args:
        posterior_mean: ``q``'s mean, shape ``(n,)`` or ``(N, n)``.
        posterior_cov: ``q``'s covariance, shape ``(n, n)``.
        model: The model the joint is built from, with fixed noise.
        prior: The ``p(x)`` the joint conditions, the step's predicted belief.
        observation: The reading, shape ``(m,)`` or ``(N, m)``.

    Returns:
        The free energy in nats, a scalar array or one per reading.

    Raises:
        ValueError: If the model carries state-dependent noise. Under ``R(x)`` the
            plain ``observation_noise`` is the noise at one state, and the likelihood
            term would price every reading against it.
    """
    require_fixed_noise(model, "the free energy here is a closed form for fixed noise")
    mean_q = np.atleast_2d(np.asarray(posterior_mean, dtype=float))
    cov_q = np.asarray(posterior_cov, dtype=float)
    readings = np.atleast_2d(np.asarray(observation, dtype=float))
    observation_matrix = np.asarray(model.observation_matrix, dtype=float)  # C
    observation_noise = np.asarray(model.observation_noise, dtype=float)  # R
    prior_mean = np.asarray(prior.mean, dtype=float)
    prior_cov = np.asarray(prior.cov, dtype=float)
    dimension = cov_q.shape[0]

    negative_entropy = -0.5 * (
        dimension * math.log(2 * math.pi * math.e) + np.linalg.slogdet(cov_q)[1]
    )
    # −E_q[ln p(x)]: the cross-entropy of q against the prior.
    prior_precision = np.linalg.inv(prior_cov)
    prior_term = 0.5 * (
        dimension * math.log(2 * math.pi)
        + np.linalg.slogdet(prior_cov)[1]
        + np.trace(prior_precision @ cov_q)
        + np.einsum(
            "bi,ij,bj->b", mean_q - prior_mean, prior_precision, mean_q - prior_mean
        )
    )
    # −E_q[ln p(y | x)]: the reading against the belief pushed through the sensor.
    noise_precision = np.linalg.inv(observation_noise)
    innovation = readings - mean_q @ observation_matrix.T
    likelihood_term = 0.5 * (
        observation_matrix.shape[0] * math.log(2 * math.pi)
        + np.linalg.slogdet(observation_noise)[1]
        + np.trace(noise_precision @ observation_matrix @ cov_q @ observation_matrix.T)
        + np.einsum("bi,ij,bj->b", innovation, noise_precision, innovation)
    )
    free_energy = negative_entropy + prior_term + likelihood_term
    return free_energy[0] if np.ndim(observation) == 1 else free_energy


@dataclass(frozen=True)
class AdditivityResidual:
    """One step's closure check: both sides, every bar, and whether they meet.

    Args:
        free_energy: ``E_{y∼p*}[F]``, measured directly.
        free_energy_bar: Its bar, ``δ_F``.
        entropy: ``H(p*(y | u))`` from the estimator.
        entropy_bar: Its bar, ``δ_H``.
        misspecification: ``D₁`` in closed form.
        misspecification_bar: Its bar, ``δ₁``. ``0.0`` at ``EXACT``.
        inference_gap: ``D₂`` in closed form.
        inference_gap_bar: Its bar, ``δ₂``. ``0.0`` at ``EXACT``.
    """

    free_energy: float
    free_energy_bar: float
    entropy: float
    entropy_bar: float
    misspecification: float
    misspecification_bar: float
    inference_gap: float
    inference_gap_bar: float

    @property
    def residual(self) -> float:
        """``E[F] − (H(p*) + D₁ + D₂)``."""
        return self.free_energy - (
            self.entropy + self.misspecification + self.inference_gap
        )

    @property
    def bound(self) -> float:
        """``δ_F + δ_H + δ₁ + δ₂``, all four."""
        return (
            self.free_energy_bar
            + self.entropy_bar
            + self.misspecification_bar
            + self.inference_gap_bar
        )

    @property
    def closes(self) -> bool:
        """Whether the residual sits within the four-term bound."""
        return abs(self.residual) <= self.bound


@dataclass(frozen=True)
class AdditivityCheck:
    """Measures both sides of ``E[F] = H(p*) + D₁ + D₂`` for one step of a cell.

    Separate from the evaluator on purpose. The evaluator's terms never need an
    entropy, and this is the only object that estimates one.

    Args:
        entropy_estimator: What estimates ``H(p*(y | u))``, and to what bar.
        samples: How many readings to draw from the true predictive for ``E[F]``.
        sigma_multiplier: ``δ_F`` is this many standard errors of the sample mean. A
            stated bar, not a proved one, and it is printed as such.
    """

    entropy_estimator: EntropyEstimator
    samples: int
    sigma_multiplier: float = 3.0

    def check_step(
        self,
        truth: LinearGaussianModel,
        truth_belief: Belief,
        cell: CrossCell,
        model_belief: Belief,
        agent_belief: Belief,
        action: ArrayLike | None,
        key: Array,
        *,
        misspecification_bar: float = 0.0,
        inference_gap_bar: float = 0.0,
    ) -> AdditivityResidual:
        """Both sides of the identity for one step, from the beliefs before it.

        Args:
            truth: ``p*``.
            truth_belief: The exact filter's belief under ``truth`` before the step.
            cell: The model and filter under check.
            model_belief: The exact filter's belief under the cell's model.
            agent_belief: The cell filter's own belief.
            action: The action applied over the step.
            key: A JAX PRNG key, split between the entropy and the readings.
            misspecification_bar: ``δ₁``, for a ``D₁`` not at ``EXACT``.
            inference_gap_bar: ``δ₂``, on the same terms.

        Returns:
            The residual with every term and bar it was built from.
        """
        key_entropy, key_readings = jax.random.split(key)
        true_mean, true_cov = observation_predictive(truth, truth_belief, action)
        exact_model = KalmanBackend(cell.model)

        def agent_update(reading: np.ndarray) -> Belief:
            return cell.inference_backend.infer_states(reading, agent_belief, action)

        def exact_update(reading: np.ndarray) -> Belief:
            return exact_model.infer_states(reading, model_belief, action)

        entropy = self.entropy_estimator.estimate(true_mean, true_cov, key_entropy)
        free_energy, free_energy_bar = self._measured_free_energy(
            agent_update,
            cell.model,
            model_belief,
            action,
            true_mean,
            true_cov,
            key_readings,
        )
        return AdditivityResidual(
            free_energy=free_energy,
            free_energy_bar=free_energy_bar,
            entropy=entropy.value,
            entropy_bar=entropy.bar,
            misspecification=misspecification_step(
                truth, truth_belief, cell.model, model_belief, action
            ),
            misspecification_bar=misspecification_bar,
            inference_gap=inference_gap_step(
                agent_update, exact_update, true_mean, true_cov
            ),
            inference_gap_bar=inference_gap_bar,
        )

    def _measured_free_energy(
        self,
        agent_update: StepUpdate,
        model: LinearGaussianModel,
        model_belief: Belief,
        action: ArrayLike | None,
        true_mean: np.ndarray,
        true_cov: np.ndarray,
        key: Array,
    ) -> tuple[float, float]:
        """``E_{y∼p*}[F]`` by sampling, and its ``sigma_multiplier`` standard-error bar.

        The cell's step is read as an affine map once, checked, and then applied to
        every draw at once. ``F`` at each draw is the direct formula.
        """
        step = _affine_in_observation(agent_update, true_mean, "agent_update")
        readings = np.asarray(
            jax.random.multivariate_normal(
                key, true_mean, true_cov, shape=(self.samples,)
            )
        )
        posterior_means = step.mean + (readings - step.centre) @ step.gain.T
        values = variational_free_energy(
            posterior_means,
            step.cov,
            model,
            predicted_belief(model, model_belief, action),
            readings,
        )
        return (
            float(values.mean()),
            self.sigma_multiplier * float(values.std(ddof=1) / math.sqrt(self.samples)),
        )


def _gaussian_log_density(
    points: np.ndarray, mean: np.ndarray, cov: np.ndarray
) -> np.ndarray:
    """``ln N(points; mean, cov)`` for a batch of points."""
    precision = np.linalg.inv(cov)
    shift = points - mean
    return -0.5 * (
        cov.shape[0] * math.log(2 * math.pi)
        + np.linalg.slogdet(cov)[1]
        + np.einsum("bi,ij,bj->b", shift, precision, shift)
    )
