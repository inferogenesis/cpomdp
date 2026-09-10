"""The transition seam is general, checked rather than asserted.

PR-7 requires the filter to be written against a general transition kernel rather
than a hard-coded linear-Gaussian one. Every other kernel test uses
`LinearGaussianKernel`, so a fixed-noise assumption leaking into `predict` or
`filter_sequence` would go unnoticed. The double below is defined here, in numpy,
outside the package, and the filter has to run it unchanged.

State-dependent process noise is the case it is written in, because that is the one
the seam is claimed to admit and no shipped class provides (issue #56). Nothing here
makes `Q(x)` a supported surface: the double is a test fixture, not an export.
"""

import numpy as np
import pytest

from cpomdp.reference.filtering import filter_sequence, predict
from cpomdp.reference.likelihood import FixedNoiseLikelihood
from cpomdp.reference.quadrature import QuadratureGrid, gaussian_on
from cpomdp.reference.transition import LinearGaussianKernel, TransitionKernel


class QuadraticProcessNoiseKernel:
    """A scalar test double: ``x' ~ N(a·x, q0 + curvature·x²)``.

    Implements the protocol and nothing else. Plain numpy, no pytree registration and
    no validation, which is the point: whatever the filter needs from a kernel is
    what this has, and if it needs more than the protocol says, this fails.

    ``at_destination`` reads the noise at ``x'`` instead of ``x``. That is not a
    transition density, and the test below is what says so.
    """

    is_fixed = False

    def __init__(self, dynamics, base_noise, curvature, *, at_destination=False):
        self.dynamics = dynamics
        self.base_noise = base_noise
        self.curvature = curvature
        self.at_destination = at_destination

    def log_transition(self, destinations, origins, action=None):
        destination = np.asarray(destinations)[:, 0][:, None]
        origin = np.asarray(origins)[:, 0][None, :]
        variance = self.base_noise + self.curvature * (
            destination**2 if self.at_destination else origin**2
        )
        residual = destination - self.dynamics * origin
        return -0.5 * (np.log(2 * np.pi * variance) + residual**2 / variance)


def central_moment(density, order):
    x = np.asarray(density.grid.nodes)[:, 0]
    return float(density.expectation((x - float(density.mean[0])) ** order))


# --- the double is a kernel, and a correct one --------------------------------------


def test_the_double_satisfies_the_protocol():
    kernel = QuadraticProcessNoiseKernel(0.9, 0.2, 0.5)
    assert isinstance(kernel, TransitionKernel)
    assert not kernel.is_fixed


def test_at_zero_curvature_it_is_the_shipped_kernel():
    # Anchors the double against the one path with an oracle. Without this, a bug in
    # the fixture would read as a bug in the seam.
    grid = QuadratureGrid(lower=[-6.0], upper=[6.0], counts=[301])
    double = QuadraticProcessNoiseKernel(0.9, 0.2, 0.0)
    shipped = LinearGaussianKernel([[0.9]], dynamics_noise=[[0.2]])
    np.testing.assert_allclose(
        double.log_transition(grid.nodes, grid.nodes),
        shipped.log_transition(grid.nodes, grid.nodes),
        rtol=1e-12,
    )


# --- why the noise is read at the departed state ------------------------------------


def test_reading_the_noise_at_the_origin_gives_a_transition_density():
    # Every row integrates to one over destinations, which is what makes it a
    # conditional density and what makes prediction mass-preserving.
    grid = QuadratureGrid(lower=[-20.0], upper=[20.0], counts=[8001])
    kernel = QuadraticProcessNoiseKernel(0.9, 0.2, 0.5)
    transition = np.exp(
        kernel.log_transition(grid.nodes, np.array([[-1.5], [0.0], [2.0]]))
    )
    for column in range(3):
        np.testing.assert_allclose(
            float(grid.integrate(transition[:, column])), 1.0, atol=1e-9
        )


def test_reading_it_at_the_destination_does_not():
    # The convention is forced, not chosen. N(x'; ax, Q(x')) is a perfectly ordinary
    # expression that is not a transition density, and nothing downstream would
    # report the difference: prediction would quietly create or destroy mass.
    grid = QuadratureGrid(lower=[-20.0], upper=[20.0], counts=[8001])
    kernel = QuadraticProcessNoiseKernel(0.9, 0.2, 0.5, at_destination=True)
    transition = np.exp(kernel.log_transition(grid.nodes, np.array([[0.0]])))
    assert abs(float(grid.integrate(transition[:, 0])) - 1.0) > 0.1


# --- the filter runs it unchanged ---------------------------------------------------


def test_predict_accepts_a_kernel_it_has_never_seen():
    grid = QuadratureGrid(lower=[-16.0], upper=[16.0], counts=[2801])
    prior = gaussian_on(grid, 0.0, 1.0).normalise()
    varying = predict(
        prior,
        QuadraticProcessNoiseKernel(1.0, 0.2, 0.6).log_transition(
            grid.nodes, grid.nodes
        ),
    )
    fixed = predict(
        prior,
        LinearGaussianKernel([[1.0]], dynamics_noise=[[0.2]]).log_transition(
            grid.nodes, grid.nodes
        ),
    )

    # Prediction still moves mass without creating it, since each row is a density.
    np.testing.assert_allclose(float(varying.log_mass), 0.0, atol=1e-7)

    # State-dependent noise costs box, the same box is not as good under it. The
    # prior's far tail diffuses at a rate the near mass never sees: at x = 6 this
    # kernel's own spread is 4.7, so it reaches past an edge a fixed Q of 0.2 never
    # approaches. Sizing a box from the belief alone is not enough once Q varies.
    assert float(varying.log_mass) < float(fixed.log_mass)
    assert abs(float(fixed.log_mass)) < 1e-10


def test_state_dependent_process_noise_predicts_out_of_the_gaussian_family():
    # States that diffuse at different rates predict forward into a mixture of
    # differently-scaled Gaussians. A scale mixture is symmetric here, so skewness
    # says nothing and the tails are the tell: excess kurtosis is strictly positive,
    # and no covariance recursion reproduces that.
    grid = QuadratureGrid(lower=[-16.0], upper=[16.0], counts=[2801])
    prior = gaussian_on(grid, 0.0, 1.0).normalise()

    varying = predict(
        prior,
        QuadraticProcessNoiseKernel(1.0, 0.2, 0.6).log_transition(
            grid.nodes, grid.nodes
        ),
    ).normalise()
    fixed = predict(
        prior,
        LinearGaussianKernel([[1.0]], dynamics_noise=[[0.2]]).log_transition(
            grid.nodes, grid.nodes
        ),
    ).normalise()

    excess = central_moment(varying, 4) / central_moment(varying, 2) ** 2 - 3.0
    assert excess > 0.1

    fixed_excess = central_moment(fixed, 4) / central_moment(fixed, 2) ** 2 - 3.0
    assert abs(fixed_excess) < 1e-6


def test_the_recursion_accepts_it_too_and_the_noise_changes_the_answer():
    grid = QuadratureGrid(lower=[-14.0], upper=[14.0], counts=[2001])
    likelihood = FixedNoiseLikelihood([[1.0]], observation_noise=[[0.3]])
    observations = [[0.6], [1.4], [-0.5]]
    prior = gaussian_on(grid, 0.0, 1.0).normalise()

    varying = filter_sequence(
        prior, QuadraticProcessNoiseKernel(0.9, 0.15, 0.5), likelihood, observations
    )
    fixed = filter_sequence(
        prior,
        LinearGaussianKernel([[0.9]], dynamics_noise=[[0.15]]),
        likelihood,
        observations,
    )

    assert len(varying) == len(observations)
    assert all(abs(float(p.log_mass)) < 1e-12 for p in varying)
    # If the seam were ignoring the kernel it was handed, these would coincide.
    assert abs(float(varying[-1].cov[0, 0]) - float(fixed[-1].cov[0, 0])) > 1e-3


def test_a_kernel_of_the_wrong_shape_is_still_refused():
    # Generality is not permissiveness. The filter takes any kernel and no wrong one.
    grid = QuadratureGrid(lower=[-6.0], upper=[6.0], counts=[201])
    kernel = QuadraticProcessNoiseKernel(0.9, 0.2, 0.5)
    with pytest.raises(ValueError, match="201x201"):
        predict(
            gaussian_on(grid, 0.0, 1.0),
            kernel.log_transition(grid.nodes[:50], grid.nodes),
        )
