"""The merge gate: the grid filter against the closed-form Kalman posterior.

The reference filter may not import the filter it is evidence about, so the
comparison lives here rather than in `src`. `KalmanBackend` is the oracle because
under a fixed `R` the Kalman filter *is* the exact Bayesian filter, not an
approximation of it.
"""

import numpy as np
import pytest

from cpomdp.backends.kalman import KalmanBackend
from cpomdp.reference.filtering import condition, filter_sequence, predict
from cpomdp.reference.likelihood import FixedNoiseLikelihood
from cpomdp.reference.quadrature import QuadratureGrid, gaussian_on
from cpomdp.reference.transition import LinearGaussianKernel
from cpomdp.types import Belief, LinearGaussianModel

A, C, Q, R = [[0.9]], [[1.0]], [[0.15]], [[0.3]]
PRIOR = Belief(mean=[0.0], cov=[[1.0]])
OBSERVATIONS = [[0.6], [1.4], [0.2], [-0.5], [1.1]]


def scalar_model(control_matrix=None):
    """The model both filters are run on, built the same way for each."""
    return LinearGaussianModel(
        dynamics_matrix=A,
        observation_matrix=C,
        dynamics_noise=Q,
        observation_noise=R,
        control_matrix=control_matrix,
        prior=PRIOR,
    )


def scalar_pieces(control_matrix=None):
    """The reference filter's two halves for that same model."""
    return (
        LinearGaussianKernel(A, dynamics_noise=Q, control_matrix=control_matrix),
        FixedNoiseLikelihood(C, observation_noise=R),
    )


# --- one prediction ------------------------------------------------------------------


class TestPredict:
    def test_matches_the_kalman_time_update(self):
        # mu- = A mu, Sigma- = A Sigma A' + Q, in closed form.
        grid = QuadratureGrid(lower=[-16.0], upper=[16.0], counts=[3201])
        kernel = LinearGaussianKernel([[0.9]], dynamics_noise=[[0.2]])
        predicted = predict(
            gaussian_on(grid, 0.4, 1.1),
            kernel.log_transition(grid.nodes, grid.nodes),
        )
        np.testing.assert_allclose(predicted.mean, [0.9 * 0.4], atol=1e-9)
        np.testing.assert_allclose(predicted.cov, [[0.9 * 1.1 * 0.9 + 0.2]], atol=1e-9)

    def test_a_control_moves_the_predicted_mean(self):
        grid = QuadratureGrid(lower=[-16.0], upper=[16.0], counts=[3201])
        kernel = LinearGaussianKernel(
            [[1.0]], dynamics_noise=[[0.2]], control_matrix=[[0.5]]
        )
        predicted = predict(
            gaussian_on(grid, 0.0, 1.0),
            kernel.log_transition(grid.nodes, grid.nodes, [2.0]),
        )
        np.testing.assert_allclose(predicted.mean, [1.0], atol=1e-9)

    def test_prediction_preserves_mass_and_does_not_normalise(self):
        # An exact time update moves mass without creating or destroying it, so a
        # box wide enough leaves the normaliser where it was. That is what makes a
        # shortfall on a tighter box readable as truncation rather than as the step.
        grid = QuadratureGrid(lower=[-16.0], upper=[16.0], counts=[3201])
        kernel = LinearGaussianKernel([[0.9]], dynamics_noise=[[0.2]])
        prior = gaussian_on(grid, 0.4, 1.1).normalise()
        predicted = predict(prior, kernel.log_transition(grid.nodes, grid.nodes))

        np.testing.assert_allclose(float(predicted.log_mass), 0.0, atol=1e-10)
        np.testing.assert_allclose(
            float(predicted.log_normaliser), float(prior.log_normaliser), rtol=1e-12
        )

    def test_a_box_too_small_for_the_spread_shows_up_after_the_step(self):
        # Prediction is where truncation first bites: the transition widens the
        # support the prior was sized for. A tight box loses mass here, and the
        # unnormalised result is what lets the caller see it.
        grid = QuadratureGrid(lower=[-2.0], upper=[2.0], counts=[801])
        kernel = LinearGaussianKernel([[1.0]], dynamics_noise=[[1.0]])
        prior = gaussian_on(grid, 0.0, 1.0).normalise()
        predicted = predict(prior, kernel.log_transition(grid.nodes, grid.nodes))
        assert float(predicted.log_mass) < -0.05

    def test_refuses_a_matrix_that_does_not_match_the_grid(self):
        grid = QuadratureGrid(lower=[-4.0], upper=[4.0], counts=[101])
        with pytest.raises(ValueError, match="101x101"):
            predict(gaussian_on(grid, 0.0, 1.0), np.zeros((101, 50)))


# --- the whole recursion --------------------------------------------------------------


class TestAgreementWithKalmanOverATrajectory:
    def test_every_posterior_matches_the_kalman_run(self):
        grid = QuadratureGrid(lower=[-14.0], upper=[14.0], counts=[2801])
        kernel, likelihood = scalar_pieces()

        posteriors = filter_sequence(
            gaussian_on(grid, 0.0, 1.0), kernel, likelihood, OBSERVATIONS
        )

        backend = KalmanBackend(scalar_model())
        belief = PRIOR
        for step, observation in enumerate(OBSERVATIONS):
            belief = backend.infer_states(observation, belief)
            np.testing.assert_allclose(posteriors[step].mean, belief.mean, atol=1e-8)
            np.testing.assert_allclose(posteriors[step].cov, belief.cov, atol=1e-8)

    def test_every_step_holds_the_same_absolute_bar(self):
        # This does not test a growth *ratio*, and an earlier version claimed to. The
        # per-step disagreements run at float epsilon, around 1e-15, so a ratio
        # between them is noise and asserts nothing. One fixed bar is what is
        # available: a systematic leak in prediction, or a wrongly reused transition
        # matrix, accumulates across five steps and breaks it. Measured worst is
        # ~1.6e-15, so the bar below has three orders of headroom and still fires on
        # a per-step shift of 1e-11.
        grid = QuadratureGrid(lower=[-14.0], upper=[14.0], counts=[2801])
        kernel, likelihood = scalar_pieces()
        posteriors = filter_sequence(
            gaussian_on(grid, 0.0, 1.0), kernel, likelihood, OBSERVATIONS
        )

        backend = KalmanBackend(scalar_model())
        belief = PRIOR
        errors = []
        for step, observation in enumerate(OBSERVATIONS):
            belief = backend.infer_states(observation, belief)
            errors.append(abs(float(posteriors[step].mean[0] - belief.mean[0])))

        assert max(errors) < 5e-12, errors
        # The last step is not systematically worse than the first, which is the part
        # a leak would break. Stated as a bound rather than a ratio, for the reason
        # above.
        assert errors[-1] < 5e-12

    def test_a_driven_run_matches_the_kalman_run(self):
        grid = QuadratureGrid(lower=[-14.0], upper=[14.0], counts=[2801])
        kernel, likelihood = scalar_pieces(control_matrix=[[0.4]])
        actions = [[1.0], [-0.5], [0.0], [2.0], [1.0]]

        posteriors = filter_sequence(
            gaussian_on(grid, 0.0, 1.0), kernel, likelihood, OBSERVATIONS, actions
        )

        backend = KalmanBackend(scalar_model(control_matrix=[[0.4]]))
        belief = PRIOR
        for step, observation in enumerate(OBSERVATIONS):
            belief = backend.infer_states(observation, belief, actions[step])
            np.testing.assert_allclose(posteriors[step].mean, belief.mean, atol=1e-8)
            np.testing.assert_allclose(posteriors[step].cov, belief.cov, atol=1e-8)

    def test_the_run_returns_one_posterior_per_observation(self):
        grid = QuadratureGrid(lower=[-10.0], upper=[10.0], counts=[801])
        kernel, likelihood = scalar_pieces()
        posteriors = filter_sequence(
            gaussian_on(grid, 0.0, 1.0), kernel, likelihood, OBSERVATIONS
        )
        assert len(posteriors) == len(OBSERVATIONS)
        assert all(p.grid.same_lattice_as(grid) for p in posteriors)

    def test_the_normaliser_accumulates_the_run_evidence(self):
        # Each step adds its own log p(y_t | y_<t), so the last posterior's
        # normaliser less the prior's is the log evidence for the whole sequence.
        grid = QuadratureGrid(lower=[-14.0], upper=[14.0], counts=[2801])
        kernel, likelihood = scalar_pieces()
        prior = gaussian_on(grid, 0.0, 1.0).normalise()
        posteriors = filter_sequence(prior, kernel, likelihood, OBSERVATIONS)

        stepwise = 0.0
        belief = prior
        for observation in OBSERVATIONS:
            predicted = predict(belief, kernel.log_transition(grid.nodes, grid.nodes))
            belief = condition(predicted, likelihood, observation)
            stepwise += float(belief.log_normaliser - predicted.log_normaliser)

        total = float(posteriors[-1].log_normaliser - prior.log_normaliser)
        np.testing.assert_allclose(total, stepwise, atol=1e-10)

    def test_refuses_actions_that_do_not_match_the_observations(self):
        grid = QuadratureGrid(lower=[-10.0], upper=[10.0], counts=[401])
        kernel, likelihood = scalar_pieces(control_matrix=[[1.0]])
        with pytest.raises(ValueError, match="one entry per observation"):
            filter_sequence(
                gaussian_on(grid, 0.0, 1.0),
                kernel,
                likelihood,
                OBSERVATIONS,
                [[1.0], [1.0]],
            )
