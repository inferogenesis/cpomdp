import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp.reference.likelihood import (
    FixedNoiseLikelihood,
    GaussianChannel,
    ObservationLikelihood,
    StateDependentNoiseLikelihood,
)


def quadratic_noise(states, params):
    """R(x) = R0 + kappa * x1^2, one 1x1 covariance per state."""
    r0, kappa = params
    return (r0 + kappa * states[:, :1] ** 2)[:, :, None]


def _log_normal(residual, var):
    return -0.5 * np.log(2.0 * np.pi * var) - residual**2 / (2.0 * var)


# --- the fixed-noise path -----------------------------------------------------------


class TestFixedNoiseLikelihood:
    def test_matches_the_scalar_gaussian_by_hand(self):
        likelihood = FixedNoiseLikelihood([[1.0]], observation_noise=[[0.4]])
        states = np.array([[-1.0], [0.0], [2.5]])
        np.testing.assert_allclose(
            likelihood.log_likelihood([1.2], states),
            _log_normal(1.2 - states[:, 0], 0.4),
            rtol=1e-12,
        )

    def test_matches_a_correlated_multivariate_gaussian_by_hand(self):
        # Two observation channels with correlated noise, so the log-determinant and
        # the quadratic form both carry off-diagonal terms. A whitening that dropped
        # the correlation would pass a diagonal case and fail here.
        c = np.array([[1.0, 0.0], [0.5, 1.0]])
        r = np.array([[0.6, 0.2], [0.2, 0.9]])
        likelihood = FixedNoiseLikelihood(c, observation_noise=r)
        states = np.array([[0.3, -0.7], [1.1, 2.0]])
        y = np.array([0.4, -0.2])

        residuals = y - states @ c.T
        sign, log_det = np.linalg.slogdet(r)
        expected = -0.5 * (
            2 * np.log(2 * np.pi)
            + log_det
            + np.einsum("ni,ij,nj->n", residuals, np.linalg.inv(r), residuals)
        )
        assert sign == 1.0
        np.testing.assert_allclose(
            likelihood.log_likelihood(y, states), expected, rtol=1e-11
        )

    def test_it_integrates_to_one_over_the_observation(self):
        # A likelihood is a density in y, not in x. Nothing else in the suite pins the
        # normalising constant, and a sign slip in the log-determinant would leave
        # every ratio right and every absolute value wrong.
        likelihood = FixedNoiseLikelihood([[1.0]], observation_noise=[[0.4]])
        state = np.array([[0.5]])
        readings = np.linspace(-10.0, 10.0, 4001)
        values = np.exp(
            [float(likelihood.log_likelihood([y], state)[0]) for y in readings]
        )
        np.testing.assert_allclose(np.trapezoid(values, readings), 1.0, atol=1e-9)

    def test_factors_the_noise_once_at_construction(self):
        likelihood = FixedNoiseLikelihood([[1.0, 0.0]], observation_noise=[[4.0]])
        np.testing.assert_allclose(likelihood.noise_cholesky, [[2.0]])
        np.testing.assert_allclose(float(likelihood.log_det_noise), np.log(4.0))

    def test_reports_itself_fixed(self):
        likelihood = FixedNoiseLikelihood([[1.0]], observation_noise=[[1.0]])
        assert likelihood.is_fixed
        assert isinstance(likelihood, ObservationLikelihood)
        assert isinstance(likelihood, GaussianChannel)

    def test_refuses_a_singular_noise(self):
        # The density inverts R, so a noiseless sensor is +inf, not a sharp reading.
        with pytest.raises(ValueError, match="positive-definite"):
            FixedNoiseLikelihood([[1.0]], observation_noise=[[0.0]])

    def test_refuses_a_noise_that_does_not_match_the_matrix(self):
        with pytest.raises(ValueError, match="to match"):
            FixedNoiseLikelihood(
                [[1.0, 0.0]], observation_noise=[[1.0, 0.0], [0.0, 1.0]]
            )

    def test_refuses_a_non_matrix_observation_map(self):
        with pytest.raises(ValueError, match="2-D"):
            FixedNoiseLikelihood([1.0], observation_noise=[[1.0]])

    def test_refuses_an_observation_of_the_wrong_length(self):
        # The broadcast trap: without the check a length-1 reading against a 2-D
        # prediction gives a confident density over the wrong thing.
        likelihood = FixedNoiseLikelihood(np.eye(2), observation_noise=np.eye(2))
        with pytest.raises(ValueError, match="length 2"):
            likelihood.log_likelihood([0.0], np.zeros((3, 2)))

    def test_refuses_states_of_the_wrong_width(self):
        likelihood = FixedNoiseLikelihood([[1.0, 0.0]], observation_noise=[[1.0]])
        with pytest.raises(ValueError, match="N x 2"):
            likelihood.log_likelihood([0.0], np.zeros((3, 1)))


# --- the state-dependent path -------------------------------------------------------


class TestStateDependentNoiseLikelihood:
    def test_matches_the_scalar_gaussian_with_the_noise_varying(self):
        likelihood = StateDependentNoiseLikelihood(
            [[1.0]],
            observation_noise_fn=quadratic_noise,
            observation_noise_params=(0.1, 1.0),
        )
        states = np.array([[-1.4], [0.0], [0.975], [2.0]])
        variances = 0.1 + states[:, 0] ** 2
        np.testing.assert_allclose(
            likelihood.log_likelihood([2.0], states),
            _log_normal(2.0 - states[:, 0], variances),
            rtol=1e-12,
        )

    def test_agrees_with_the_fixed_path_when_the_noise_does_not_vary(self):
        # Two independent routes to the same number: one factors R once, the other
        # decomposes a stack of identical matrices. They should not disagree.
        states = np.linspace(-3.0, 3.0, 101)[:, None]
        fixed = FixedNoiseLikelihood([[1.0]], observation_noise=[[0.7]])
        varying = StateDependentNoiseLikelihood(
            [[1.0]],
            observation_noise_fn=quadratic_noise,
            observation_noise_params=(0.7, 0.0),
        )
        np.testing.assert_allclose(
            varying.log_likelihood([0.3], states),
            fixed.log_likelihood([0.3], states),
            rtol=1e-12,
        )

    def test_reports_itself_not_fixed(self):
        likelihood = StateDependentNoiseLikelihood(
            [[1.0]],
            observation_noise_fn=quadratic_noise,
            observation_noise_params=(1.0, 1.0),
        )
        assert not likelihood.is_fixed
        assert isinstance(likelihood, ObservationLikelihood)
        assert isinstance(likelihood, GaussianChannel)

    def test_refuses_a_noise_function_returning_the_wrong_shape(self):
        def one_matrix_for_everyone(states, params):
            return jnp.eye(1)

        likelihood = StateDependentNoiseLikelihood(
            [[1.0]], observation_noise_fn=one_matrix_for_everyone
        )
        with pytest.raises(ValueError, match="one covariance per state"):
            likelihood.log_likelihood([0.0], np.zeros((4, 1)))

    def test_refuses_a_noise_that_is_not_callable(self):
        with pytest.raises(TypeError, match="callable"):
            StateDependentNoiseLikelihood(
                [[1.0]],
                observation_noise_fn=np.eye(1),  # ty: ignore[invalid-argument-type]
            )

    def test_is_differentiable_in_the_noise_parameters(self):
        # The reason params are a leaf rather than baked into the function.
        states = np.linspace(-2.0, 2.0, 51)[:, None]
        likelihood = StateDependentNoiseLikelihood(
            [[1.0]], observation_noise_fn=quadratic_noise
        )

        def total(params):
            return jnp.sum(
                StateDependentNoiseLikelihood(
                    likelihood.observation_matrix,
                    observation_noise_fn=quadratic_noise,
                    observation_noise_params=params,
                ).log_likelihood([1.0], states)
            )

        gradient = jax.grad(total)((0.5, 1.0))
        assert all(bool(jnp.isfinite(g)) for g in gradient)
        assert any(abs(float(g)) > 0.0 for g in gradient)


# --- pytree behaviour ---------------------------------------------------------------


class TestPytree:
    def test_the_fixed_likelihood_survives_a_jit_boundary(self):
        likelihood = FixedNoiseLikelihood([[1.0]], observation_noise=[[0.4]])
        states = np.linspace(-2.0, 2.0, 33)[:, None]
        jitted = jax.jit(lambda lik: lik.log_likelihood(jnp.array([0.7]), states))
        np.testing.assert_allclose(
            jitted(likelihood), likelihood.log_likelihood([0.7], states), rtol=1e-12
        )

    def test_the_state_dependent_likelihood_survives_a_jit_boundary(self):
        likelihood = StateDependentNoiseLikelihood(
            [[1.0]],
            observation_noise_fn=quadratic_noise,
            observation_noise_params=(0.2, 0.8),
        )
        states = np.linspace(-2.0, 2.0, 33)[:, None]
        jitted = jax.jit(lambda lik: lik.log_likelihood(jnp.array([0.7]), states))
        np.testing.assert_allclose(
            jitted(likelihood), likelihood.log_likelihood([0.7], states), rtol=1e-12
        )


def test_a_fixed_likelihood_refuses_states_of_the_wrong_shape():
    likelihood = FixedNoiseLikelihood([[1.0, 0.0]], observation_noise=[[0.5]])
    with pytest.raises(ValueError, match=r"states must be a 2-D array"):
        likelihood.observation_noise_at(jnp.zeros(5))
    with pytest.raises(ValueError, match=r"states must be a 2-D array"):
        likelihood.observation_noise_at(jnp.zeros((5, 3)))
    assert likelihood.observation_noise_at(jnp.zeros((5, 2))).shape == (5, 1, 1)
