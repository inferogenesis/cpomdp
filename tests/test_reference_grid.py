import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import erf

from cpomdp.reference.quadrature import GridDensity, QuadratureGrid, gaussian_on

SQRT2 = np.sqrt(2.0)


def _log_normal(x, mu, var):
    """log N(x; mu, var) for a 1-D column of nodes."""
    return -0.5 * np.log(2.0 * np.pi * var) - (x - mu) ** 2 / (2.0 * var)


def _gaussian_kl(mu_p, var_p, mu_q, var_q):
    """Closed-form D_KL[N(mu_p, var_p) || N(mu_q, var_q)] in one dimension."""
    return (
        0.5 * np.log(var_q / var_p) + (var_p + (mu_p - mu_q) ** 2) / (2.0 * var_q) - 0.5
    )


# --- the lattice --------------------------------------------------------------------


class TestQuadratureGrid:
    def test_node_count_is_the_product_of_the_counts(self):
        grid = QuadratureGrid(lower=[-1.0, 0.0], upper=[1.0, 2.0], counts=[5, 7])
        assert grid.size == 35
        assert grid.nodes.shape == (35, 2)
        assert grid.weights.shape == (35,)
        assert grid.ndim == 2

    def test_the_last_axis_varies_fastest(self):
        # The weight outer-product is built in the same order as the meshgrid ravel.
        # If the two ever disagree, every integral is silently permuted, so the node
        # order is pinned here rather than left implicit.
        grid = QuadratureGrid(lower=[0.0, 10.0], upper=[1.0, 11.0], counts=[2, 3])
        np.testing.assert_allclose(
            grid.nodes,
            [[0, 10], [0, 10.5], [0, 11], [1, 10], [1, 10.5], [1, 11]],
        )

    def test_weights_sum_to_the_volume_of_the_box(self):
        grid = QuadratureGrid(lower=[-1.0, 0.0], upper=[1.0, 3.0], counts=[9, 11])
        np.testing.assert_allclose(float(grid.weights.sum()), 6.0)

    def test_endpoint_weights_are_half(self):
        grid = QuadratureGrid(lower=[0.0], upper=[1.0], counts=[5])
        h = 0.25
        np.testing.assert_allclose(grid.weights, [h / 2, h, h, h, h / 2])

    def test_integrating_the_constant_one_gives_the_volume(self):
        grid = QuadratureGrid(lower=[-2.0], upper=[3.0], counts=[17])
        np.testing.assert_allclose(
            float(grid.integrate(jnp.ones(grid.size))), 5.0, rtol=1e-12
        )

    def test_integrate_carries_trailing_axes(self):
        grid = QuadratureGrid(lower=[0.0], upper=[1.0], counts=[9])
        values = jnp.ones((grid.size, 2, 2))
        assert grid.integrate(values).shape == (2, 2)

    def test_spacing_and_box_report_what_was_declared(self):
        grid = QuadratureGrid(lower=[-1.0, 0.0], upper=[1.0, 1.0], counts=[5, 3])
        assert grid.spacing == (0.5, 0.5)
        assert grid.box == ((-1.0, 1.0), (0.0, 1.0))

    def test_same_lattice_compares_box_and_resolution(self):
        a = QuadratureGrid(lower=[0.0], upper=[1.0], counts=[5])
        assert a.same_lattice_as(QuadratureGrid([0.0], [1.0], [5]))
        assert not a.same_lattice_as(QuadratureGrid([0.0], [1.0], [7]))
        assert not a.same_lattice_as(QuadratureGrid([0.0], [2.0], [5]))

    def test_rejects_a_single_node_axis(self):
        with pytest.raises(ValueError, match="at least 2"):
            QuadratureGrid(lower=[0.0], upper=[1.0], counts=[1])

    def test_rejects_an_empty_or_inverted_axis(self):
        with pytest.raises(ValueError, match="must exceed lower"):
            QuadratureGrid(lower=[1.0], upper=[1.0], counts=[5])
        with pytest.raises(ValueError, match="must exceed lower"):
            QuadratureGrid(lower=[1.0], upper=[0.0], counts=[5])

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="one entry per axis"):
            QuadratureGrid(lower=[0.0, 0.0], upper=[1.0], counts=[5, 5])

    def test_rejects_no_axes(self):
        with pytest.raises(ValueError, match="at least one axis"):
            QuadratureGrid(lower=[], upper=[], counts=[])


# --- convergence --------------------------------------------------------------------


def test_refinement_converges_at_least_at_the_trapezoid_rate():
    # A Gaussian truncated at one standard deviation, where the density at the box
    # edge is far from zero and the endpoint correction dominates. The bound is
    # one-sided: on a box wide enough for the tails to vanish the same rule is
    # spectrally accurate, so a test pinned to h^2 exactly would fail on the grids
    # this substrate is actually used with.
    exact = erf(1.0 / SQRT2)
    errors = []
    for count in (51, 101, 201):
        grid = QuadratureGrid(lower=[-1.0], upper=[1.0], counts=[count])
        values = np.exp(_log_normal(np.asarray(grid.nodes)[:, 0], 0.0, 1.0))
        errors.append(abs(float(grid.integrate(values)) - exact))

    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert errors[0] / errors[1] >= 3.0
    assert errors[1] / errors[2] >= 3.0


# --- what a density does and does not do to itself ----------------------------------


class TestNormalisation:
    def test_construction_does_not_normalise(self):
        grid = QuadratureGrid(lower=[-6.0], upper=[6.0], counts=[601])
        density = GridDensity(grid, _log_normal(np.asarray(grid.nodes)[:, 0], 0, 1) + 3)
        np.testing.assert_allclose(float(density.log_mass), 3.0, atol=1e-9)
        np.testing.assert_allclose(float(density.log_normaliser), 0.0)

    def test_normalise_moves_the_mass_onto_the_normaliser(self):
        grid = QuadratureGrid(lower=[-6.0], upper=[6.0], counts=[601])
        raw = GridDensity(grid, _log_normal(np.asarray(grid.nodes)[:, 0], 0, 1) + 3)
        normalised = raw.normalise()
        np.testing.assert_allclose(float(normalised.log_mass), 0.0, atol=1e-12)
        np.testing.assert_allclose(float(normalised.log_normaliser), 3.0, atol=1e-9)

    def test_normalising_twice_adds_nothing_the_second_time(self):
        grid = QuadratureGrid(lower=[-6.0], upper=[6.0], counts=[601])
        once = gaussian_on(grid, 0.0, 1.0).normalise()
        twice = once.normalise()
        np.testing.assert_allclose(
            float(twice.log_normaliser), float(once.log_normaliser), atol=1e-12
        )

    def test_a_box_too_small_shows_up_as_a_mass_deficit(self):
        # The one measurement the substrate exists to keep visible. Two standard
        # deviations clip 4.55% of a unit Gaussian, and the grid says so rather than
        # dividing it away.
        # The tolerance is the trapezoid rule's own error at this spacing, which
        # the density's slope at the box edge sets and which no widening removes.
        # It sits five orders below the deficit being measured.
        grid = QuadratureGrid(lower=[-2.0], upper=[2.0], counts=[4001])
        clipped = gaussian_on(grid, 0.0, 1.0)
        np.testing.assert_allclose(
            float(jnp.exp(clipped.log_mass)), erf(2.0 / SQRT2), atol=1e-7
        )
        np.testing.assert_allclose(
            float(clipped.normalise().log_normaliser),
            np.log(erf(2.0 / SQRT2)),
            atol=1e-7,
        )


# --- moments ------------------------------------------------------------------------


class TestMoments:
    def test_gaussian_moments_match_the_closed_form(self):
        grid = QuadratureGrid(lower=[-8.0], upper=[10.0], counts=[901])
        density = gaussian_on(grid, 1.0, 2.0)
        np.testing.assert_allclose(density.mean, [1.0], atol=1e-10)
        np.testing.assert_allclose(density.cov, [[2.0]], atol=1e-10)

    def test_moments_are_right_on_an_unnormalised_density(self):
        grid = QuadratureGrid(lower=[-8.0], upper=[10.0], counts=[901])
        scaled = GridDensity(
            grid, _log_normal(np.asarray(grid.nodes)[:, 0], 1.0, 2.0) - 4.2
        )
        np.testing.assert_allclose(scaled.mean, [1.0], atol=1e-10)
        np.testing.assert_allclose(scaled.cov, [[2.0]], atol=1e-10)

    def test_a_bimodal_density_gets_its_own_moments(self):
        # Nothing here may presume a Gaussian. A two-component mixture has moments in
        # closed form and is not one, so it separates a working quadrature from a
        # rule that happens to be exact for the Gaussian it was built around.
        grid = QuadratureGrid(lower=[-12.0], upper=[12.0], counts=[2401])
        x = np.asarray(grid.nodes)[:, 0]
        weights, means, variances = (0.3, 0.7), (-3.0, 2.0), (0.5, 1.5)
        mixture = sum(
            w * np.exp(_log_normal(x, m, v))
            for w, m, v in zip(weights, means, variances, strict=True)
        )
        density = GridDensity(grid, np.log(mixture))

        expected_mean = sum(w * m for w, m in zip(weights, means, strict=True))
        expected_var = (
            sum(
                w * (v + m**2)
                for w, m, v in zip(weights, means, variances, strict=True)
            )
            - expected_mean**2
        )
        np.testing.assert_allclose(density.mean, [expected_mean], atol=1e-9)
        np.testing.assert_allclose(density.cov, [[expected_var]], atol=1e-8)

    def test_two_dimensional_moments_including_the_correlation(self):
        grid = QuadratureGrid(
            lower=[-12.0, -12.0], upper=[12.0, 12.0], counts=[241, 241]
        )
        cov = np.array([[2.0, 0.6], [0.6, 1.0]])
        mean = np.array([0.4, -0.3])
        centred = np.asarray(grid.nodes) - mean
        precision = np.linalg.inv(cov)
        quadratic = np.einsum("ni,ij,nj->n", centred, precision, centred)
        density = GridDensity(grid, -0.5 * quadratic)

        np.testing.assert_allclose(density.mean, mean, atol=1e-8)
        np.testing.assert_allclose(density.cov, cov, atol=1e-7)

    def test_expectation_carries_a_matrix_integrand(self):
        grid = QuadratureGrid(lower=[-6.0], upper=[6.0], counts=[601])
        density = gaussian_on(grid, 0.0, 1.0)
        got = density.expectation(jnp.ones((grid.size, 2, 3)))
        assert got.shape == (2, 3)
        np.testing.assert_allclose(got, np.ones((2, 3)), atol=1e-9)

    def test_expectation_and_integrate_measure_different_things(self):
        # Same convention, same contraction, different measure. Integrating the
        # constant one gives the box's volume; expecting it gives one. Anything that
        # collapsed the two would have to break one of these.
        grid = QuadratureGrid(lower=[-6.0], upper=[6.0], counts=[601])
        density = gaussian_on(grid, 0.0, 1.0)
        ones = jnp.ones(grid.size)
        np.testing.assert_allclose(float(grid.integrate(ones)), 12.0, rtol=1e-12)
        np.testing.assert_allclose(float(density.expectation(ones)), 1.0, atol=1e-12)

    def test_expectation_folds_in_the_normalisation_the_grid_does_not(self):
        # The division a caller reaching for the grid directly would have to
        # remember. On a clipped box the mass is not one, so forgetting it is a
        # scaled answer rather than an error.
        grid = QuadratureGrid(lower=[-1.5], upper=[1.5], counts=[1201])
        clipped = gaussian_on(grid, 0.0, 1.0)
        squares = np.asarray(grid.nodes)[:, 0] ** 2
        by_hand = float(grid.integrate(np.exp(clipped.log_density) * squares))
        np.testing.assert_allclose(
            float(clipped.expectation(jnp.asarray(squares))),
            by_hand / float(jnp.exp(clipped.log_mass)),
            rtol=1e-12,
        )


# --- divergence ---------------------------------------------------------------------


class TestKullbackLeibler:
    def test_matches_the_closed_form_gaussian_divergence(self):
        grid = QuadratureGrid(lower=[-12.0], upper=[12.0], counts=[2401])
        p = gaussian_on(grid, 0.5, 1.0)
        q = gaussian_on(grid, -0.4, 2.5)
        np.testing.assert_allclose(
            float(p.kl_to(q)), _gaussian_kl(0.5, 1.0, -0.4, 2.5), atol=1e-9
        )

    def test_is_not_symmetric(self):
        grid = QuadratureGrid(lower=[-12.0], upper=[12.0], counts=[2401])
        p = gaussian_on(grid, 0.5, 1.0)
        q = gaussian_on(grid, -0.4, 2.5)
        assert not np.isclose(float(p.kl_to(q)), float(q.kl_to(p)))

    def test_a_density_against_itself_is_zero(self):
        grid = QuadratureGrid(lower=[-10.0], upper=[10.0], counts=[801])
        p = gaussian_on(grid, 0.2, 1.3)
        np.testing.assert_allclose(float(p.kl_to(p)), 0.0, atol=1e-12)

    def test_scaling_either_argument_changes_nothing(self):
        grid = QuadratureGrid(lower=[-10.0], upper=[10.0], counts=[801])
        p = gaussian_on(grid, 0.2, 1.3)
        q = gaussian_on(grid, 0.0, 1.0)
        scaled_p = GridDensity(grid, p.log_density + 2.0)
        scaled_q = GridDensity(grid, q.log_density - 5.0)
        np.testing.assert_allclose(
            float(scaled_p.kl_to(scaled_q)), float(p.kl_to(q)), atol=1e-12
        )

    def test_a_node_where_p_vanishes_contributes_nothing(self):
        # 0*log 0 is the limit, not the arithmetic. Without the guard this is NaN and
        # every divergence involving a compactly supported p returns NaN.
        grid = QuadratureGrid(lower=[-4.0], upper=[4.0], counts=[801])
        x = np.asarray(grid.nodes)[:, 0]
        log_p = np.where(np.abs(x) <= 1.0, 0.0, -np.inf)
        p = GridDensity(grid, log_p)
        q = gaussian_on(grid, 0.0, 1.0)
        value = float(p.kl_to(q))
        assert np.isfinite(value)
        assert value > 0.0

    def test_a_node_where_q_vanishes_and_p_does_not_is_infinite(self):
        grid = QuadratureGrid(lower=[-4.0], upper=[4.0], counts=[801])
        x = np.asarray(grid.nodes)[:, 0]
        p = gaussian_on(grid, 0.0, 1.0)
        q = GridDensity(grid, np.where(np.abs(x) <= 1.0, 0.0, -np.inf))
        assert float(p.kl_to(q)) == np.inf

    def test_refuses_a_density_on_another_lattice(self):
        p = gaussian_on(QuadratureGrid([-5.0], [5.0], [201]), 0.0, 1.0)
        q = gaussian_on(QuadratureGrid([-5.0], [5.0], [401]), 0.0, 1.0)
        with pytest.raises(ValueError, match="same lattice"):
            p.kl_to(q)


# --- construction-time refusals -----------------------------------------------------


class TestDensityValidation:
    def test_rejects_a_length_that_does_not_match_the_grid(self):
        grid = QuadratureGrid(lower=[0.0], upper=[1.0], counts=[5])
        with pytest.raises(ValueError, match="to match the grid"):
            GridDensity(grid, np.zeros(4))

    def test_rejects_values_that_are_not_a_vector(self):
        grid = QuadratureGrid(lower=[0.0], upper=[1.0], counts=[5])
        with pytest.raises(ValueError, match="1-D vector"):
            GridDensity(grid, np.zeros((5, 1)))

    def test_rejects_nan(self):
        grid = QuadratureGrid(lower=[0.0], upper=[1.0], counts=[3])
        with pytest.raises(ValueError, match="no NaN"):
            GridDensity(grid, [0.0, np.nan, 0.0])

    def test_rejects_positive_infinity(self):
        grid = QuadratureGrid(lower=[0.0], upper=[1.0], counts=[3])
        with pytest.raises(ValueError, match=r"no \+inf"):
            GridDensity(grid, [0.0, np.inf, 0.0])

    def test_accepts_negative_infinity(self):
        # A vanishing density is ordinary, and a compactly supported reference
        # posterior is made of these.
        grid = QuadratureGrid(lower=[0.0], upper=[1.0], counts=[3])
        GridDensity(grid, [-np.inf, 0.0, -np.inf])


# --- pytree behaviour ---------------------------------------------------------------


class TestPytree:
    def test_a_grid_round_trips_through_flatten(self):
        grid = QuadratureGrid(lower=[-1.0, 0.0], upper=[1.0, 2.0], counts=[5, 7])
        leaves, aux = grid.tree_flatten()
        rebuilt = QuadratureGrid.tree_unflatten(aux, leaves)
        assert rebuilt.box == grid.box
        assert rebuilt.counts == grid.counts
        np.testing.assert_array_equal(rebuilt.nodes, grid.nodes)
        np.testing.assert_array_equal(rebuilt.weights, grid.weights)

    def test_a_density_round_trips_through_flatten(self):
        grid = QuadratureGrid(lower=[-6.0], upper=[6.0], counts=[201])
        density = gaussian_on(grid, 0.0, 1.0).normalise()
        rebuilt = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(density),
            jax.tree_util.tree_leaves(density),
        )
        np.testing.assert_array_equal(rebuilt.log_density, density.log_density)
        np.testing.assert_array_equal(rebuilt.log_normaliser, density.log_normaliser)
        assert rebuilt.grid.box == grid.box

    def test_moments_survive_a_jit_boundary(self):
        grid = QuadratureGrid(lower=[-8.0], upper=[8.0], counts=[801])
        density = gaussian_on(grid, 0.5, 1.4)
        jitted = jax.jit(lambda d: (d.mean, d.cov))
        mean, cov = jitted(density)
        np.testing.assert_allclose(mean, density.mean, atol=1e-12)
        np.testing.assert_allclose(cov, density.cov, atol=1e-12)

    def test_the_divergence_is_differentiable_in_the_density(self):
        # The reference filter feeds a divergence to things that differentiate it, so
        # the guard against vanishing nodes has to be a `where` rather than a mask
        # applied after the fact, which would leak a NaN through the gradient.
        grid = QuadratureGrid(lower=[-6.0], upper=[6.0], counts=[401])
        q = gaussian_on(grid, 0.0, 1.0)

        def divergence(log_density):
            return GridDensity(grid, log_density).kl_to(q)

        gradient = jax.grad(divergence)(gaussian_on(grid, 0.3, 1.0).log_density)
        assert bool(jnp.isfinite(gradient).all())


# --- a Gaussian on the lattice --------------------------------------------------------


class TestGaussianOn:
    def test_a_scalar_mean_and_variance_render_the_one_dimensional_gaussian(self):
        grid = QuadratureGrid(lower=[-6.0], upper=[6.0], counts=[241])
        density = gaussian_on(grid, 0.7, 1.9)
        np.testing.assert_allclose(
            density.log_density,
            _log_normal(np.asarray(grid.nodes)[:, 0], 0.7, 1.9),
            rtol=1e-14,
        )
        assert float(density.log_normaliser) == 0.0

    def test_a_correlated_gaussian_has_unit_mass_and_the_declared_moments(self):
        grid = QuadratureGrid(lower=[-9.0, -9.0], upper=[9.0, 9.0], counts=[181, 181])
        mean = np.array([0.4, -1.1])
        cov = np.array([[2.0, 0.9], [0.9, 1.5]])
        density = gaussian_on(grid, mean, cov)
        np.testing.assert_allclose(float(jnp.exp(density.log_mass)), 1.0, atol=1e-9)
        np.testing.assert_allclose(density.mean, mean, atol=1e-9)
        np.testing.assert_allclose(density.cov, cov, atol=1e-9)

    def test_it_traces(self):
        grid = QuadratureGrid(lower=[-6.0], upper=[6.0], counts=[121])

        @jax.jit
        def render(mean, var):
            return gaussian_on(grid, mean, var).log_density

        np.testing.assert_allclose(
            render(0.3, 1.2), gaussian_on(grid, 0.3, 1.2).log_density
        )

    def test_a_mean_of_the_wrong_length_is_refused(self):
        grid = QuadratureGrid(lower=[-1.0, -1.0], upper=[1.0, 1.0], counts=[3, 3])
        with pytest.raises(ValueError, match="mean"):
            gaussian_on(grid, [0.0], [[1.0, 0.0], [0.0, 1.0]])

    def test_a_covariance_of_the_wrong_shape_is_refused(self):
        grid = QuadratureGrid(lower=[-1.0, -1.0], upper=[1.0, 1.0], counts=[3, 3])
        with pytest.raises(ValueError, match="cov"):
            gaussian_on(grid, [0.0, 0.0], [[1.0]])

    def test_a_sharp_gaussian_is_a_gaussian_and_not_a_degenerate_sensor(self):
        # The sensor validator floors definiteness at 1e-8 because something inverts
        # the noise. Nothing inverts a belief, so a variance below that floor renders.
        grid = QuadratureGrid(lower=[-1e-3], upper=[1e-3], counts=[2001])
        density = gaussian_on(grid, 0.0, 1e-9)
        np.testing.assert_allclose(float(jnp.exp(density.log_mass)), 1.0, atol=1e-9)

    def test_a_zero_variance_direction_is_refused(self):
        grid = QuadratureGrid(lower=[-1.0, -1.0], upper=[1.0, 1.0], counts=[3, 3])
        with pytest.raises(ValueError, match="no density on the lattice"):
            gaussian_on(grid, [0.0, 0.0], [[1.0, 0.0], [0.0, 0.0]])

    def test_an_indefinite_covariance_is_refused(self):
        grid = QuadratureGrid(lower=[-1.0, -1.0], upper=[1.0, 1.0], counts=[3, 3])
        with pytest.raises(ValueError, match="definite"):
            gaussian_on(grid, [0.0, 0.0], [[1.0, 2.0], [2.0, 1.0]])
