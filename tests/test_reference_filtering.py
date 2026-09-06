import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp.reference.filtering import condition
from cpomdp.reference.likelihood import (
    FixedNoiseLikelihood,
    StateDependentNoiseLikelihood,
)
from cpomdp.reference.quadrature import GridDensity, QuadratureGrid, gaussian_on


def quadratic_noise(states, params):
    """R(x) = R0 + kappa * x1^2, one 1x1 covariance per state."""
    r0, kappa = params
    return (r0 + kappa * states[:, :1] ** 2)[:, :, None]


def kalman_update(mean, cov, c, r, y):
    """The closed-form measurement update, the oracle this module is checked on."""
    mean, cov, c, r, y = (
        np.atleast_1d(mean),
        np.atleast_2d(cov),
        np.atleast_2d(c),
        np.atleast_2d(r),
        np.atleast_1d(y),
    )
    innovation_cov = c @ cov @ c.T + r  # S
    gain = cov @ c.T @ np.linalg.inv(innovation_cov)  # K
    return (
        mean + gain @ (y - c @ mean),
        (np.eye(len(mean)) - gain @ c) @ cov,
        innovation_cov,
    )


# --- the merge gate: agreement with the closed-form Kalman posterior ----------------


class TestAgreementWithKalman:
    def test_one_dimensional_update(self):
        grid = QuadratureGrid(lower=[-12.0], upper=[12.0], counts=[24001])
        posterior = condition(
            gaussian_on(grid, 0.0, 1.0),
            FixedNoiseLikelihood([[1.0]], observation_noise=[[1.0]]),
            [2.0],
        )
        expected_mean, expected_cov, _ = kalman_update(0.0, 1.0, 1.0, 1.0, 2.0)

        np.testing.assert_allclose(posterior.mean, expected_mean, atol=1e-10)
        np.testing.assert_allclose(posterior.cov, expected_cov, atol=1e-10)

    def test_two_dimensional_update_with_a_partial_observation(self):
        # One channel observing a two-state latent, with correlation in the prior.
        # The unobserved dimension is only corrected through that correlation, which
        # is the part a filter can get right on the diagonal and wrong off it.
        grid = QuadratureGrid(
            lower=[-13.0, -13.0], upper=[13.0, 13.0], counts=[401, 401]
        )
        prior_mean = np.array([0.2, -0.5])
        prior_cov = np.array([[1.4, 0.6], [0.6, 0.9]])
        c, r, y = np.array([[1.0, 0.0]]), np.array([[0.3]]), np.array([1.7])

        posterior = condition(
            gaussian_on(grid, prior_mean, prior_cov),
            FixedNoiseLikelihood(c, observation_noise=r),
            y,
        )
        expected_mean, expected_cov, _ = kalman_update(prior_mean, prior_cov, c, r, y)

        np.testing.assert_allclose(posterior.mean, expected_mean, atol=1e-8)
        np.testing.assert_allclose(posterior.cov, expected_cov, atol=1e-7)

    def test_two_channels_with_correlated_noise(self):
        grid = QuadratureGrid(
            lower=[-12.0, -12.0], upper=[12.0, 12.0], counts=[401, 401]
        )
        prior_mean = np.array([0.0, 0.0])
        prior_cov = np.array([[2.0, -0.4], [-0.4, 1.1]])
        c = np.array([[1.0, 0.5], [0.0, 1.0]])
        r = np.array([[0.5, 0.15], [0.15, 0.8]])
        y = np.array([1.0, -0.6])

        posterior = condition(
            gaussian_on(grid, prior_mean, prior_cov),
            FixedNoiseLikelihood(c, observation_noise=r),
            y,
        )
        expected_mean, expected_cov, _ = kalman_update(prior_mean, prior_cov, c, r, y)

        np.testing.assert_allclose(posterior.mean, expected_mean, atol=1e-8)
        np.testing.assert_allclose(posterior.cov, expected_cov, atol=1e-7)

    def test_the_posterior_is_the_kalman_gaussian_and_not_merely_its_moments(self):
        # Matching two moments is weaker than matching the distribution. Under fixed
        # R the exact posterior *is* Gaussian, so the divergence is zero, and that is
        # the claim the merge gate rests on.
        grid = QuadratureGrid(lower=[-14.0], upper=[14.0], counts=[28001])
        posterior = condition(
            gaussian_on(grid, 0.3, 1.2),
            FixedNoiseLikelihood([[1.0]], observation_noise=[[0.7]]),
            [1.5],
        )
        expected_mean, expected_cov, _ = kalman_update(0.3, 1.2, 1.0, 0.7, 1.5)
        kalman = gaussian_on(grid, expected_mean, expected_cov).normalise()

        assert abs(float(kalman.kl_to(posterior))) < 1e-12


# --- the evidence falls out of the normaliser ---------------------------------------


class TestEvidence:
    def test_the_normaliser_increment_is_the_log_evidence(self):
        # p(y) = N(y; C mu, S) in closed form. Getting it for free is why `condition`
        # is allowed to normalise: nothing is discarded, it is only moved.
        grid = QuadratureGrid(lower=[-14.0], upper=[14.0], counts=[28001])
        prior = gaussian_on(grid, 0.3, 1.2).normalise()
        posterior = condition(
            prior, FixedNoiseLikelihood([[1.0]], observation_noise=[[0.7]]), [1.5]
        )

        _, _, innovation_cov = kalman_update(0.3, 1.2, 1.0, 0.7, 1.5)
        expected = float(
            -0.5
            * (
                np.log(2 * np.pi * innovation_cov[0, 0])
                + (1.5 - 0.3) ** 2 / innovation_cov[0, 0]
            )
        )
        increment = float(posterior.log_normaliser - prior.log_normaliser)
        np.testing.assert_allclose(increment, expected, atol=1e-10)

    def test_the_posterior_is_normalised_whatever_the_prior_was(self):
        grid = QuadratureGrid(lower=[-10.0], upper=[10.0], counts=[8001])
        unnormalised = GridDensity(grid, gaussian_on(grid, 0.0, 1.0).log_density + 3.7)
        posterior = condition(
            unnormalised,
            FixedNoiseLikelihood([[1.0]], observation_noise=[[1.0]]),
            [1.0],
        )
        np.testing.assert_allclose(float(posterior.log_mass), 0.0, atol=1e-12)


# --- the case the whole thing exists for --------------------------------------------


class TestStateDependentNoise:
    def test_the_posterior_leaves_the_gaussian_family(self):
        # R(x) = 0.1 + x^2 with a unit prior and y = 2. The exact posterior is skewed
        # and bimodal: the reading is explained either by the state being out there,
        # or by the state sitting where the sensor happens to be noisy. No Gaussian
        # holds both, at any resolution.
        grid = QuadratureGrid(lower=[-14.0], upper=[14.0], counts=[56001])
        posterior = condition(
            gaussian_on(grid, 0.0, 1.0),
            StateDependentNoiseLikelihood(
                [[1.0]],
                observation_noise_fn=quadratic_noise,
                observation_noise_params=(0.1, 1.0),
            ),
            [2.0],
        )
        mean = float(posterior.mean[0])
        variance = float(posterior.cov[0, 0])
        third = float(posterior.expectation((np.asarray(grid.nodes)[:, 0] - mean) ** 3))
        skewness = third / variance**1.5

        assert abs(skewness) > 1.0

        moment_matched = gaussian_on(grid, mean, variance).normalise()
        assert float(moment_matched.kl_to(posterior)) > 0.1

    def test_it_agrees_with_the_fixed_path_when_the_noise_is_constant(self):
        # The state-dependent route reduces to the fixed one at zero curvature, so a
        # sign or shape error in the batched decomposition shows up here rather than
        # only in the regime with no oracle.
        grid = QuadratureGrid(lower=[-12.0], upper=[12.0], counts=[12001])
        prior = gaussian_on(grid, 0.4, 1.1)
        fixed = condition(
            prior, FixedNoiseLikelihood([[1.0]], observation_noise=[[0.6]]), [1.3]
        )
        varying = condition(
            prior,
            StateDependentNoiseLikelihood(
                [[1.0]],
                observation_noise_fn=quadratic_noise,
                observation_noise_params=(0.6, 0.0),
            ),
            [1.3],
        )
        np.testing.assert_allclose(varying.log_density, fixed.log_density, atol=1e-12)


# --- plumbing -----------------------------------------------------------------------


def test_the_posterior_stays_on_the_priors_grid():
    grid = QuadratureGrid(lower=[-8.0], upper=[8.0], counts=[801])
    posterior = condition(
        gaussian_on(grid, 0.0, 1.0),
        FixedNoiseLikelihood([[1.0]], observation_noise=[[1.0]]),
        [0.5],
    )
    assert posterior.grid.same_lattice_as(grid)


def test_an_observation_of_the_wrong_length_is_refused():
    grid = QuadratureGrid(lower=[-8.0], upper=[8.0], counts=[801])
    with pytest.raises(ValueError, match="length 1"):
        condition(
            gaussian_on(grid, 0.0, 1.0),
            FixedNoiseLikelihood([[1.0]], observation_noise=[[1.0]]),
            [0.5, 0.5],
        )


def test_conditioning_survives_a_jit_boundary():
    grid = QuadratureGrid(lower=[-10.0], upper=[10.0], counts=[2001])
    prior = gaussian_on(grid, 0.0, 1.0)
    likelihood = FixedNoiseLikelihood([[1.0]], observation_noise=[[0.9]])
    jitted = jax.jit(lambda p, lik: condition(p, lik, jnp.array([1.1])).mean)
    np.testing.assert_allclose(
        jitted(prior, likelihood), condition(prior, likelihood, [1.1]).mean, atol=1e-12
    )
