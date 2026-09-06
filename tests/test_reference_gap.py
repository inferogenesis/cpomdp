"""The averaged inference gap: calibration, a closed form, and what it reports.

These are ordinary two-valued oracle assertions, outside the warrant vocabulary by
construction: nothing here decides a declared claim. The engine carries no warrant on
arrival (ADR-052), and R6 is where a claim about this quantity gets registered.
"""

import numpy as np
import pytest

from cpomdp.reference.gap import averaged_inference_gap
from cpomdp.reference.likelihood import (
    FixedNoiseLikelihood,
    StateDependentNoiseLikelihood,
)
from cpomdp.reference.quadrature import GridDensity, QuadratureGrid, gaussian_on

PRIOR_MEAN, PRIOR_VAR = 0.3, 0.8


def kalman_rule(noise, prior_mean=PRIOR_MEAN, prior_var=PRIOR_VAR):
    """A scalar Kalman update with `noise` plugged in, as an approximate-posterior rule.

    The one-rung stand-in for the ladder. It reads its noise from the argument rather
    than from the model, which is exactly the freedom the rungs will differ over.
    """
    gain = prior_var / (prior_var + noise)

    def rule(prior, observation):
        mean = prior_mean + gain * (float(np.asarray(observation)[0]) - prior_mean)
        return gaussian_on(prior.grid, mean, (1.0 - gain) * prior_var)

    return rule


def averaged_gaussian_gap(true_noise, plugin_noise, prior_var=PRIOR_VAR):
    """Closed form for E_y[KL(q ‖ p)] when both are Gaussian.

    Under a fixed R the exact posterior is Gaussian, so a wrong-R filter differs from
    it only in gain and variance and the whole functional is available in closed form.
    The mean term is quadratic in `y - mu`, whose expectation under p* is S.
    """
    gain = prior_var / (prior_var + true_noise)
    plugin_gain = prior_var / (prior_var + plugin_noise)
    exact_var = (1.0 - gain) * prior_var
    approx_var = (1.0 - plugin_gain) * prior_var
    innovation_var = prior_var + true_noise  # S
    return (
        0.5 * np.log(exact_var / approx_var)
        + (approx_var + (plugin_gain - gain) ** 2 * innovation_var) / (2 * exact_var)
        - 0.5
    )


def quadratic_noise(states, params):
    """R(x) = R0 + kappa * x^2, one 1x1 covariance per state."""
    r0, kappa = params
    return (r0 + kappa * states[:, :1] ** 2)[:, :, None]


# --- the instrument reads zero where zero is known -----------------------------------


def test_an_exact_filter_under_fixed_noise_has_no_gap():
    # The calibration. Under a fixed R the Kalman posterior *is* the exact Bayesian
    # posterior, so the gap is zero by the structure of the problem, not by tuning.
    states = QuadratureGrid(lower=[-12.0], upper=[12.0], counts=[4001])
    observations = QuadratureGrid(lower=[-14.0], upper=[14.0], counts=[401])
    noise = 0.5

    measured = averaged_inference_gap(
        gaussian_on(states, PRIOR_MEAN, PRIOR_VAR),
        FixedNoiseLikelihood([[1.0]], observation_noise=[[noise]]),
        kalman_rule(noise),
        observations,
    )
    assert abs(measured.value) < 1e-10
    np.testing.assert_allclose(measured.predictive_mass, 1.0, atol=1e-9)


# --- and the right number where the answer is known ----------------------------------


@pytest.mark.parametrize("plugin_noise", [0.25, 0.8, 2.0])
def test_a_wrong_fixed_noise_matches_the_closed_form(plugin_noise):
    # The whole functional against an oracle, not just its pieces: the y-average, the
    # divergence and the predictive weighting all have to be right together.
    states = QuadratureGrid(lower=[-14.0], upper=[14.0], counts=[5601])
    observations = QuadratureGrid(lower=[-16.0], upper=[16.0], counts=[801])
    true_noise = 0.5

    measured = averaged_inference_gap(
        gaussian_on(states, PRIOR_MEAN, PRIOR_VAR),
        FixedNoiseLikelihood([[1.0]], observation_noise=[[true_noise]]),
        kalman_rule(plugin_noise),
        observations,
    )
    np.testing.assert_allclose(
        measured.value, averaged_gaussian_gap(true_noise, plugin_noise), rtol=1e-6
    )


def test_the_gap_vanishes_only_where_the_rule_is_right():
    states = QuadratureGrid(lower=[-14.0], upper=[14.0], counts=[2801])
    observations = QuadratureGrid(lower=[-16.0], upper=[16.0], counts=[401])
    likelihood = FixedNoiseLikelihood([[1.0]], observation_noise=[[0.5]])

    for plugin in (0.3, 0.5, 0.9):
        measured = averaged_inference_gap(
            gaussian_on(states, PRIOR_MEAN, PRIOR_VAR),
            likelihood,
            kalman_rule(plugin),
            observations,
        )
        assert measured.value >= -1e-12
        assert (measured.value > 1e-6) == (plugin != 0.5)


# --- the conventions -----------------------------------------------------------------


def test_the_direction_is_reverse():
    # KL(q ‖ p), not KL(p ‖ q). The two differ, and only one is the declared figure.
    # The asymmetry is what makes the convention checkable at all.
    states = QuadratureGrid(lower=[-14.0], upper=[14.0], counts=[4001])
    observations = QuadratureGrid(lower=[-16.0], upper=[16.0], counts=[401])
    measured = averaged_inference_gap(
        gaussian_on(states, PRIOR_MEAN, PRIOR_VAR),
        FixedNoiseLikelihood([[1.0]], observation_noise=[[0.5]]),
        kalman_rule(2.0),
        observations,
    )
    reverse = averaged_gaussian_gap(0.5, 2.0)
    forward = averaged_gaussian_gap(2.0, 0.5)
    assert abs(measured.value - reverse) < abs(measured.value - forward)


def test_the_prior_is_normalised_internally():
    # p* has to be a density for predictive_mass to mean anything, and the value must
    # not depend on how the caller happened to scale the belief they passed in.
    states = QuadratureGrid(lower=[-12.0], upper=[12.0], counts=[2801])
    observations = QuadratureGrid(lower=[-14.0], upper=[14.0], counts=[301])
    likelihood = FixedNoiseLikelihood([[1.0]], observation_noise=[[0.5]])
    rule = kalman_rule(1.5)

    normalised = gaussian_on(states, PRIOR_MEAN, PRIOR_VAR).normalise()
    scaled = GridDensity(states, normalised.log_density + 4.3)

    from_normalised = averaged_inference_gap(normalised, likelihood, rule, observations)
    from_scaled = averaged_inference_gap(scaled, likelihood, rule, observations)

    np.testing.assert_allclose(from_scaled.value, from_normalised.value, rtol=1e-12)
    np.testing.assert_allclose(
        from_scaled.predictive_mass, from_normalised.predictive_mass, rtol=1e-12
    )


# --- what the diagnostics report -----------------------------------------------------


def test_a_tight_observation_box_is_reported_not_hidden():
    states = QuadratureGrid(lower=[-14.0], upper=[14.0], counts=[2801])
    likelihood = FixedNoiseLikelihood([[1.0]], observation_noise=[[0.5]])
    rule = kalman_rule(2.0)

    wide = averaged_inference_gap(
        gaussian_on(states, PRIOR_MEAN, PRIOR_VAR),
        likelihood,
        rule,
        QuadratureGrid(lower=[-16.0], upper=[16.0], counts=[401]),
    )
    tight = averaged_inference_gap(
        gaussian_on(states, PRIOR_MEAN, PRIOR_VAR),
        likelihood,
        rule,
        QuadratureGrid(lower=[-1.0], upper=[1.6], counts=[401]),
    )

    np.testing.assert_allclose(wide.predictive_mass, 1.0, atol=1e-9)
    assert tight.predictive_mass < 0.75
    # The value still reads as an expectation, of a conditional nobody asked for. The
    # mass is what says so, which is why it comes back beside the number.
    assert tight.value < wide.value


def test_a_state_box_too_small_for_an_extreme_reading_is_reported():
    # The blind spot `predictive_mass` cannot see. A wide observation box with a
    # narrow state box drags the exact posterior to the edge on the far readings,
    # where the joint integrates to far less than p*(y). Those readings carry almost
    # no weight, so the predictive mass still reads one and only the edge ratio says
    # anything is wrong.
    likelihood = FixedNoiseLikelihood([[1.0]], observation_noise=[[0.5]])
    rule = kalman_rule(2.0)
    observations = QuadratureGrid(lower=[-30.0], upper=[30.0], counts=[401])

    roomy = averaged_inference_gap(
        gaussian_on(QuadratureGrid([-40.0], [40.0], [4001]), PRIOR_MEAN, PRIOR_VAR),
        likelihood,
        rule,
        observations,
    )
    cramped = averaged_inference_gap(
        gaussian_on(QuadratureGrid([-9.0], [9.0], [4001]), PRIOR_MEAN, PRIOR_VAR),
        likelihood,
        rule,
        observations,
    )

    np.testing.assert_allclose(cramped.predictive_mass, 1.0, atol=1e-9)
    assert roomy.worst_edge_ratio < 1e-6
    assert cramped.worst_edge_ratio > 0.5


def test_a_state_box_wide_enough_reports_a_negligible_edge():
    states = QuadratureGrid(lower=[-14.0], upper=[14.0], counts=[2801])
    observations = QuadratureGrid(lower=[-10.0], upper=[10.0], counts=[201])
    measured = averaged_inference_gap(
        gaussian_on(states, PRIOR_MEAN, PRIOR_VAR),
        FixedNoiseLikelihood([[1.0]], observation_noise=[[0.5]]),
        kalman_rule(2.0),
        observations,
    )
    assert measured.worst_edge_ratio < 1e-9
    assert measured.worst_edge_ratio >= 0.0


def test_the_divergences_are_returned_per_observation():
    # Where the gap lives in y is a property of the gap. A wrong-gain filter is worst
    # on the readings furthest from the prior mean, since that is where the two
    # posteriors' means separate.
    states = QuadratureGrid(lower=[-14.0], upper=[14.0], counts=[2801])
    observations = QuadratureGrid(lower=[-8.0], upper=[8.0], counts=[201])
    measured = averaged_inference_gap(
        gaussian_on(states, PRIOR_MEAN, PRIOR_VAR),
        FixedNoiseLikelihood([[1.0]], observation_noise=[[0.5]]),
        kalman_rule(2.0),
        observations,
    )

    assert measured.divergences.shape == (observations.size,)
    nearest = int(np.argmin(np.abs(np.asarray(observations.nodes)[:, 0] - PRIOR_MEAN)))
    assert float(measured.divergences[nearest]) == pytest.approx(
        float(measured.divergences.min()), abs=1e-9
    )
    assert float(measured.divergences[0]) > float(measured.divergences[nearest])


def test_a_rule_answering_on_another_lattice_is_refused():
    states = QuadratureGrid(lower=[-8.0], upper=[8.0], counts=[401])
    other = QuadratureGrid(lower=[-8.0], upper=[8.0], counts=[201])

    def wrong_lattice(prior, observation):
        return gaussian_on(other, 0.0, 1.0)

    with pytest.raises(ValueError, match="prior's lattice"):
        averaged_inference_gap(
            gaussian_on(states, PRIOR_MEAN, PRIOR_VAR),
            FixedNoiseLikelihood([[1.0]], observation_noise=[[0.5]]),
            wrong_lattice,
            QuadratureGrid(lower=[-10.0], upper=[10.0], counts=[51]),
        )


# --- the case with no closed form ----------------------------------------------------


def test_state_dependent_noise_opens_a_gap_no_constant_closes():
    # Under R(x) the exact posterior is not Gaussian, so every constant-noise rule has
    # a positive gap. Sweeping the plug-in shows no choice drives it to zero, which is
    # the structural-versus-quadrature distinction as a number. The sweep exhibits;
    # it does not prove a universal, and nothing here claims one.
    states = QuadratureGrid(lower=[-6.0], upper=[6.0], counts=[4001])
    observations = QuadratureGrid(lower=[-14.0], upper=[14.0], counts=[401])
    likelihood = StateDependentNoiseLikelihood(
        [[1.0]],
        observation_noise_fn=quadratic_noise,
        observation_noise_params=(0.4, 0.6),
    )
    prior = gaussian_on(states, PRIOR_MEAN, PRIOR_VAR)

    gaps = [
        averaged_inference_gap(
            prior, likelihood, kalman_rule(plugin), observations
        ).value
        for plugin in (0.4, 0.6, 0.8, 1.0, 1.4)
    ]
    assert min(gaps) > 1e-4
