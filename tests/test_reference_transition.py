import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp._lab.reference.quadrature import QuadratureGrid
from cpomdp._lab.reference.transition import LinearGaussianKernel, TransitionKernel


def _log_normal(residual, var):
    return -0.5 * np.log(2.0 * np.pi * var) - residual**2 / (2.0 * var)


class TestLinearGaussianKernel:
    def test_matches_the_scalar_gaussian_by_hand(self):
        kernel = LinearGaussianKernel([[0.9]], dynamics_noise=[[0.2]])
        origins = np.array([[-1.0], [0.5]])
        destinations = np.array([[0.0], [1.0], [2.0]])

        expected = _log_normal(
            destinations[:, 0][:, None] - 0.9 * origins[:, 0][None, :], 0.2
        )
        np.testing.assert_allclose(
            kernel.log_transition(destinations, origins), expected, rtol=1e-12
        )

    def test_is_indexed_destination_major(self):
        # The orientation the prediction contracts over. A transposed kernel is a
        # different, plausible-looking filter, so the layout is pinned rather than
        # left to the shape happening to work out.
        kernel = LinearGaussianKernel([[1.0]], dynamics_noise=[[1.0]])
        got = kernel.log_transition(np.zeros((3, 1)), np.zeros((5, 1)))
        assert got.shape == (3, 5)

    def test_each_row_of_the_kernel_integrates_to_one_over_destinations(self):
        # p(x' | x) is a density in x', so mass leaving any origin is conserved.
        # This is what makes the prediction mass-preserving, and it is the property a
        # sign slip in the log-determinant would break without moving any mode.
        grid = QuadratureGrid(lower=[-14.0], upper=[14.0], counts=[5601])
        kernel = LinearGaussianKernel([[0.85]], dynamics_noise=[[0.35]])
        origins = np.array([[-2.0], [0.0], [1.3]])
        transition = kernel.log_transition(grid.nodes, origins)
        for column in range(origins.shape[0]):
            np.testing.assert_allclose(
                float(grid.integrate(jnp.exp(transition[:, column]))), 1.0, atol=1e-10
            )

    def test_matches_a_correlated_multivariate_transition_by_hand(self):
        a = np.array([[0.9, 0.1], [0.0, 0.8]])
        q = np.array([[0.4, 0.12], [0.12, 0.25]])
        kernel = LinearGaussianKernel(a, dynamics_noise=q)
        origins = np.array([[0.3, -0.7], [1.1, 2.0]])
        destinations = np.array([[0.0, 0.0], [1.0, -1.0], [2.0, 0.5]])

        means = origins @ a.T
        differences = destinations[:, None, :] - means[None, :, :]
        _, log_det = np.linalg.slogdet(q)
        quadratic = np.einsum(
            "mni,ij,mnj->mn", differences, np.linalg.inv(q), differences
        )
        expected = -0.5 * (2 * np.log(2 * np.pi) + log_det + quadratic)
        np.testing.assert_allclose(
            kernel.log_transition(destinations, origins), expected, rtol=1e-11
        )

    def test_the_control_shifts_every_destination_mean(self):
        kernel = LinearGaussianKernel(
            [[1.0]], dynamics_noise=[[0.5]], control_matrix=[[2.0]]
        )
        origins = np.array([[0.0], [1.0]])
        destinations = np.array([[0.0], [3.0]])

        driven = kernel.log_transition(destinations, origins, [1.5])
        undriven = kernel.log_transition(destinations - 3.0, origins, [0.0])
        np.testing.assert_allclose(driven, undriven, rtol=1e-12)

    def test_reports_itself_fixed(self):
        kernel = LinearGaussianKernel([[1.0]], dynamics_noise=[[1.0]])
        assert kernel.is_fixed
        assert isinstance(kernel, TransitionKernel)

    def test_refuses_a_deterministic_transition(self):
        # No density against Lebesgue measure. Substituting a small Q on the caller's
        # behalf would pick their discretisation error for them.
        with pytest.raises(ValueError, match="positive-definite"):
            LinearGaussianKernel([[1.0]], dynamics_noise=[[0.0]])

    def test_refuses_a_non_square_dynamics_matrix(self):
        with pytest.raises(ValueError, match="square"):
            LinearGaussianKernel([[1.0, 0.0]], dynamics_noise=[[1.0]])

    def test_refuses_a_noise_that_does_not_match_the_dynamics(self):
        with pytest.raises(ValueError, match="to match dynamics_matrix"):
            LinearGaussianKernel(np.eye(2), dynamics_noise=[[1.0]])

    def test_refuses_a_control_matrix_that_does_not_reach_the_state(self):
        with pytest.raises(ValueError, match="mapping an action"):
            LinearGaussianKernel(
                np.eye(2), dynamics_noise=np.eye(2), control_matrix=[[1.0]]
            )

    def test_refuses_an_action_when_there_is_no_control_matrix(self):
        kernel = LinearGaussianKernel([[1.0]], dynamics_noise=[[1.0]])
        with pytest.raises(ValueError, match="takes no action"):
            kernel.log_transition(np.zeros((2, 1)), np.zeros((2, 1)), [1.0])

    def test_refuses_a_missing_action_when_there_is_a_control_matrix(self):
        kernel = LinearGaussianKernel(
            [[1.0]], dynamics_noise=[[1.0]], control_matrix=[[1.0]]
        )
        with pytest.raises(ValueError, match="requires an action"):
            kernel.log_transition(np.zeros((2, 1)), np.zeros((2, 1)))

    def test_refuses_an_action_of_the_wrong_length(self):
        kernel = LinearGaussianKernel(
            [[1.0]], dynamics_noise=[[1.0]], control_matrix=[[1.0]]
        )
        with pytest.raises(ValueError, match="length 1"):
            kernel.log_transition(np.zeros((2, 1)), np.zeros((2, 1)), [1.0, 2.0])

    def test_refuses_states_of_the_wrong_width(self):
        kernel = LinearGaussianKernel(np.eye(2), dynamics_noise=np.eye(2))
        with pytest.raises(ValueError, match="destinations must be an N x 2"):
            kernel.log_transition(np.zeros((3, 1)), np.zeros((3, 2)))
        with pytest.raises(ValueError, match="origins must be an N x 2"):
            kernel.log_transition(np.zeros((3, 2)), np.zeros((3, 1)))


class TestPytree:
    def test_survives_a_jit_boundary_without_a_control_matrix(self):
        kernel = LinearGaussianKernel([[0.9]], dynamics_noise=[[0.3]])
        nodes = np.linspace(-2.0, 2.0, 17)[:, None]
        jitted = jax.jit(lambda k: k.log_transition(nodes, nodes))
        np.testing.assert_allclose(
            jitted(kernel), kernel.log_transition(nodes, nodes), rtol=1e-12
        )

    def test_survives_a_jit_boundary_with_a_control_matrix(self):
        kernel = LinearGaussianKernel(
            [[0.9]], dynamics_noise=[[0.3]], control_matrix=[[1.0]]
        )
        nodes = np.linspace(-2.0, 2.0, 17)[:, None]
        jitted = jax.jit(lambda k: k.log_transition(nodes, nodes, jnp.array([0.4])))
        np.testing.assert_allclose(
            jitted(kernel), kernel.log_transition(nodes, nodes, [0.4]), rtol=1e-12
        )
