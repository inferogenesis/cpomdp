"""The rule ladder: five rungs and the declared set they sit in.

Every rung is checked against the closed-form Kalman posterior where that is the exact
filter, and against the averaged gap where it is not. The two iterating rungs are also
read against a scalar transcription of the scheme and against the exact posterior's
own gradient at their fixed point. The ladder itself is checked the way the other
declared sets are: versioned, non-empty, unique, and holding its top.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp.reference.gap import Void, averaged_inference_gap
from cpomdp.reference.ladder import (
    BELIEF_SMOOTHED_RUNG,
    CONVERGENCE_TOLERANCE,
    EXACT_RUNG,
    ITERATED_RUNG,
    ITERATION_BUDGET,
    LADDER,
    PLUG_IN_RUNG,
    SINGLE_STEP_RUNG,
    RuleLadder,
    Rung,
    RungKind,
    iterated_update,
)
from cpomdp.reference.likelihood import (
    FixedNoiseLikelihood,
    StateDependentNoiseLikelihood,
)
from cpomdp.reference.quadrature import GridDensity, QuadratureGrid, gaussian_on

PLUG_IN = PLUG_IN_RUNG
BELIEF_SMOOTHED = BELIEF_SMOOTHED_RUNG
GAUSSIAN_RUNGS = [PLUG_IN_RUNG, SINGLE_STEP_RUNG, ITERATED_RUNG, BELIEF_SMOOTHED_RUNG]
GAUSSIAN_IDS = [rung.name for rung in GAUSSIAN_RUNGS]
ITERATING_RUNGS = [SINGLE_STEP_RUNG, ITERATED_RUNG]
ITERATING_IDS = [rung.name for rung in ITERATING_RUNGS]

PRIOR_MEAN, PRIOR_VAR = 0.3, 0.8
STATES = QuadratureGrid(lower=[-14.0], upper=[14.0], counts=[1601])
OBSERVATIONS = QuadratureGrid(lower=[-12.0], upper=[12.0], counts=[401])


def quadratic_noise(states, params):
    """R(x) = R0 + kappa * x1^2, one 1x1 covariance per state."""
    r0, kappa = params
    return (r0 + kappa * states[:, :1] ** 2)[:, :, None]


def quartic_noise(states, params):
    """R(x) = R0 + kappa * x1^4, whose average is not fixed by a density's moments."""
    r0, kappa = params
    return (r0 + kappa * states[:, :1] ** 4)[:, :, None]


def sine_noise(states, params):
    """R(x) = 1.5 + 0.5 sin(x1), the declared family that can exhaust the budget."""
    return (1.5 + 0.5 * jnp.sin(states[:, :1]))[:, :, None]


def bimodal_prior():
    """An equal mixture of two unit-variance Gaussians at ±2, on the state grid."""
    x = STATES.nodes[:, 0]
    log_density = jnp.logaddexp(-0.5 * (x - 2.0) ** 2, -0.5 * (x + 2.0) ** 2)
    return GridDensity(STATES, log_density).normalise()


def scalar_scheme(observation, prior_mean, prior_var, r0, kappa, budget, tolerance):
    """(35) with the r3 block removed, scalar, for R = r0 + kappa x^2 and h = x.

    The oracle the vector code is read against in one dimension: the same equations
    written out by hand with the analytic slope 2 kappa x, so an inexact derivative in
    the rung would show up here as a different iterate.
    """
    estimate, taken = prior_mean, 0
    while taken < budget:
        taken += 1
        noise, slope = r0 + kappa * estimate**2, 2.0 * kappa * estimate
        residual = observation - estimate
        score = (
            -(residual / noise) + (1.0 - residual**2 / noise) / (2.0 * noise) * slope
        )
        steered = 1.0 + residual / (2.0 * noise) * slope
        step = ((estimate - prior_mean) / prior_var + score) / (
            1.0 / prior_var + steered**2 / noise
        )
        estimate -= step
        if abs(step) < tolerance * math.sqrt(prior_var):
            break
    noise, slope = r0 + kappa * estimate**2, 2.0 * kappa * estimate
    fisher = 1.0 / noise + slope**2 / (2.0 * noise**2)
    return estimate, 1.0 / (1.0 / prior_var + fisher), taken


def kalman_update(mean, cov, c, r, y):
    """The closed-form measurement update, the oracle every Gaussian rung is read on."""
    mean, cov, c, r, y = (
        np.atleast_1d(mean),
        np.atleast_2d(cov),
        np.atleast_2d(c),
        np.atleast_2d(r),
        np.atleast_1d(y),
    )
    innovation_cov = c @ cov @ c.T + r  # S
    gain = cov @ c.T @ np.linalg.inv(innovation_cov)  # K
    return mean + gain @ (y - c @ mean), (np.eye(len(mean)) - gain @ c) @ cov


def kalman_rule(noise):
    """The hand-rolled stand-in the gap tests used before the ladder existed."""
    gain = PRIOR_VAR / (PRIOR_VAR + noise)

    def rule(prior, observation):
        mean = PRIOR_MEAN + gain * (float(np.asarray(observation)[0]) - PRIOR_MEAN)
        return gaussian_on(prior.grid, mean, (1.0 - gain) * PRIOR_VAR)

    return rule


# --- the merge gate, part one: agreement with the closed-form Kalman posterior ------


@pytest.mark.parametrize("rung", GAUSSIAN_RUNGS, ids=GAUSSIAN_IDS)
def test_a_gaussian_rung_is_the_kalman_update_under_a_fixed_noise(rung):
    likelihood = FixedNoiseLikelihood([[1.0]], observation_noise=[[0.5]])
    prior = gaussian_on(STATES, PRIOR_MEAN, PRIOR_VAR)
    belief = rung.build(likelihood)(prior, [1.7])
    mean, cov = kalman_update(PRIOR_MEAN, PRIOR_VAR, 1.0, 0.5, 1.7)

    assert isinstance(belief, GridDensity)
    assert belief.grid.same_lattice_as(prior.grid)
    np.testing.assert_allclose(np.asarray(belief.mean), mean, atol=1e-12)
    np.testing.assert_allclose(np.asarray(belief.cov), cov, atol=1e-12)


@pytest.mark.parametrize("rung", GAUSSIAN_RUNGS, ids=GAUSSIAN_IDS)
def test_a_gaussian_rung_agrees_with_kalman_on_a_two_dimensional_state(rung):
    c = np.array([[1.0, 0.5]])
    r = np.array([[0.4]])
    prior_mean = np.array([0.2, -0.3])
    prior_cov = np.array([[0.5, 0.1], [0.1, 0.3]])
    states = QuadratureGrid(lower=[-8.0, -8.0], upper=[8.0, 8.0], counts=[321, 321])
    likelihood = FixedNoiseLikelihood(c, observation_noise=r)
    prior = gaussian_on(states, prior_mean, prior_cov)

    belief = rung.build(likelihood)(prior, [0.9])
    mean, cov = kalman_update(prior_mean, prior_cov, c, r, 0.9)

    assert isinstance(belief, GridDensity)
    np.testing.assert_allclose(np.asarray(belief.mean), mean, atol=1e-10)
    np.testing.assert_allclose(np.asarray(belief.cov), cov, atol=1e-10)


def test_the_plug_in_rung_reads_the_noise_at_the_prior_mean():
    r0, kappa = 0.5, 1.0
    likelihood = StateDependentNoiseLikelihood(
        [[1.0]],
        observation_noise_fn=quadratic_noise,
        observation_noise_params=(r0, kappa),
    )
    prior = gaussian_on(STATES, PRIOR_MEAN, PRIOR_VAR)
    belief = PLUG_IN.build(likelihood)(prior, [1.7])
    mean, cov = kalman_update(
        PRIOR_MEAN, PRIOR_VAR, 1.0, r0 + kappa * PRIOR_MEAN**2, 1.7
    )

    assert isinstance(belief, GridDensity)
    np.testing.assert_allclose(np.asarray(belief.mean), mean, atol=1e-12)
    np.testing.assert_allclose(np.asarray(belief.cov), cov, atol=1e-12)


def test_the_belief_smoothed_rung_reads_the_noise_averaged_under_the_prior():
    r0, kappa = 0.5, 1.0
    likelihood = StateDependentNoiseLikelihood(
        [[1.0]],
        observation_noise_fn=quadratic_noise,
        observation_noise_params=(r0, kappa),
    )
    prior = gaussian_on(STATES, PRIOR_MEAN, PRIOR_VAR)
    belief = BELIEF_SMOOTHED.build(likelihood)(prior, [1.7])
    # E[R0 + kappa x^2] under N(mu, var) in closed form.
    mean, cov = kalman_update(
        PRIOR_MEAN, PRIOR_VAR, 1.0, r0 + kappa * (PRIOR_MEAN**2 + PRIOR_VAR), 1.7
    )

    assert isinstance(belief, GridDensity)
    np.testing.assert_allclose(np.asarray(belief.mean), mean, atol=1e-10)
    np.testing.assert_allclose(np.asarray(belief.cov), cov, atol=1e-10)


def test_the_belief_smoothed_rung_averages_under_the_density_it_is_handed():
    r0, kappa = 0.5, 0.05
    likelihood = StateDependentNoiseLikelihood(
        [[1.0]],
        observation_noise_fn=quartic_noise,
        observation_noise_params=(r0, kappa),
    )
    prior = bimodal_prior()
    prior_mean, prior_cov = (np.asarray(m) for m in prior.moments)
    belief = BELIEF_SMOOTHED.build(likelihood)(prior, [1.7])

    noise_at_nodes = likelihood.observation_noise_at(STATES.nodes)
    under_prior = float(prior.expectation(noise_at_nodes)[0, 0])
    under_moments = float(
        gaussian_on(STATES, prior_mean, prior_cov).expectation(noise_at_nodes)[0, 0]
    )
    assert under_prior != pytest.approx(under_moments, rel=1e-2)

    mean, cov = kalman_update(prior_mean, prior_cov, 1.0, under_prior, 1.7)
    assert isinstance(belief, GridDensity)
    np.testing.assert_allclose(np.asarray(belief.mean), mean, atol=1e-10)
    np.testing.assert_allclose(np.asarray(belief.cov), cov, atol=1e-10)


# --- the two iterating rungs, read against a scalar transcription of the scheme -----


@pytest.mark.parametrize(
    ("budget", "tolerance"), [(1, 0.0), (ITERATION_BUDGET, CONVERGENCE_TOLERANCE)]
)
def test_the_scheme_matches_its_scalar_transcription_at_either_budget(
    budget, tolerance
):
    r0, kappa = 0.5, 1.0
    likelihood = StateDependentNoiseLikelihood(
        [[1.0]],
        observation_noise_fn=quadratic_noise,
        observation_noise_params=(r0, kappa),
    )
    update = iterated_update(
        likelihood,
        [PRIOR_MEAN],
        [[PRIOR_VAR]],
        [1.7],
        budget=budget,
        tolerance=tolerance,
    )
    mean, var, taken = scalar_scheme(
        1.7, PRIOR_MEAN, PRIOR_VAR, r0, kappa, budget, tolerance
    )

    assert update.iterations == taken
    assert update.converged is (budget > 1)
    np.testing.assert_allclose(np.asarray(update.mean), [mean], atol=1e-13)
    np.testing.assert_allclose(np.asarray(update.cov), [[var]], atol=1e-13)


def test_the_single_step_rung_is_one_step_of_the_scheme():
    r0, kappa = 0.5, 1.0
    likelihood = StateDependentNoiseLikelihood(
        [[1.0]],
        observation_noise_fn=quadratic_noise,
        observation_noise_params=(r0, kappa),
    )
    prior = gaussian_on(STATES, PRIOR_MEAN, PRIOR_VAR)
    belief = SINGLE_STEP_RUNG.build(likelihood)(prior, [1.7])
    mean, var, _ = scalar_scheme(1.7, PRIOR_MEAN, PRIOR_VAR, r0, kappa, 1, 0.0)

    assert isinstance(belief, GridDensity)
    np.testing.assert_allclose(np.asarray(belief.mean), [mean], atol=1e-10)
    np.testing.assert_allclose(np.asarray(belief.cov), [[var]], atol=1e-10)


def test_the_iterated_rung_settles_where_the_exact_posterior_is_flat():
    likelihood = StateDependentNoiseLikelihood(
        [[1.0]],
        observation_noise_fn=quadratic_noise,
        observation_noise_params=(0.5, 1.0),
    )
    update = iterated_update(
        likelihood,
        [PRIOR_MEAN],
        [[PRIOR_VAR]],
        [1.7],
        budget=ITERATION_BUDGET,
        tolerance=CONVERGENCE_TOLERANCE,
    )

    def log_posterior(x):
        prior = -0.5 * (x[0] - PRIOR_MEAN) ** 2 / PRIOR_VAR
        return prior + likelihood.log_likelihood([1.7], x[None, :])[0]

    assert update.converged
    assert 1 < update.iterations < ITERATION_BUDGET
    assert abs(float(jax.grad(log_posterior)(update.mean)[0])) < 1e-9
    assert abs(float(update.mean[0]) - PRIOR_MEAN) > 0.1


def test_the_iterated_rung_answers_void_when_the_budget_is_spent():
    # ADR-058's declared cell: the bounded periodic family at spread 0.30, read nine
    # predictive spreads off its prior mean, needs 124 steps at the tolerance.
    likelihood = StateDependentNoiseLikelihood([[1.0]], observation_noise_fn=sine_noise)
    prior_mean, prior_var = 1.0, 0.30**2
    predictive = math.sqrt(prior_var + 1.5 + 0.5 * math.sin(prior_mean))
    reading = [prior_mean + 9.0 * predictive]
    prior = gaussian_on(STATES, prior_mean, prior_var)

    answer = ITERATED_RUNG.build(likelihood)(prior, reading)
    assert answer == Void(
        iterations=ITERATION_BUDGET,
        detail=f"budget of {ITERATION_BUDGET} spent above the tolerance",
    )

    settled = iterated_update(
        likelihood, [prior_mean], [[prior_var]], reading, budget=200, tolerance=1e-12
    )
    assert settled.converged
    assert settled.iterations == 124


def test_the_single_step_rung_never_answers_void():
    likelihood = StateDependentNoiseLikelihood([[1.0]], observation_noise_fn=sine_noise)
    prior_mean, prior_var = 1.0, 0.30**2
    predictive = math.sqrt(prior_var + 1.5 + 0.5 * math.sin(prior_mean))
    prior = gaussian_on(STATES, prior_mean, prior_var)

    belief = SINGLE_STEP_RUNG.build(likelihood)(prior, [prior_mean + 9.0 * predictive])
    assert isinstance(belief, GridDensity)


@pytest.mark.parametrize("rung", ITERATING_RUNGS, ids=ITERATING_IDS)
def test_an_iterating_rung_refuses_a_second_observation_channel(rung):
    likelihood = FixedNoiseLikelihood(
        [[1.0, 0.0], [0.0, 1.0]], observation_noise=[[0.5, 0.0], [0.0, 0.5]]
    )
    with pytest.raises(ValueError, match="one observation channel"):
        rung.build(likelihood)


def test_the_exact_rung_is_the_exact_posterior():
    likelihood = StateDependentNoiseLikelihood(
        [[1.0]],
        observation_noise_fn=quadratic_noise,
        observation_noise_params=(0.5, 1.0),
    )
    prior = gaussian_on(STATES, PRIOR_MEAN, PRIOR_VAR)
    belief = EXACT_RUNG.build(likelihood)(prior, [1.7])
    exact = GridDensity(
        STATES,
        prior.log_density + likelihood.log_likelihood([1.7], STATES.nodes),
    )

    assert isinstance(belief, GridDensity)
    assert float(exact.kl_to(belief)) < 1e-12
    assert float(belief.kl_to(exact)) < 1e-12


def test_a_rung_refuses_an_observation_of_the_wrong_length():
    likelihood = FixedNoiseLikelihood([[1.0]], observation_noise=[[0.5]])
    prior = gaussian_on(STATES, PRIOR_MEAN, PRIOR_VAR)
    with pytest.raises(ValueError, match="observation must be a 1-D vector"):
        PLUG_IN.build(likelihood)(prior, [1.0, 2.0])


# --- what each rung reports through the gap ----------------------------------------


@pytest.mark.parametrize("rung", GAUSSIAN_RUNGS, ids=GAUSSIAN_IDS)
def test_a_gaussian_rung_closes_the_gap_under_a_fixed_noise(rung):
    likelihood = FixedNoiseLikelihood([[1.0]], observation_noise=[[0.5]])
    gap = averaged_inference_gap(
        gaussian_on(STATES, PRIOR_MEAN, PRIOR_VAR),
        likelihood,
        rung.build(likelihood),
        OBSERVATIONS,
    )
    assert gap.value < 1e-10
    assert gap.voided_mass == 0.0


def test_the_plug_in_rung_reproduces_the_hand_rolled_rule_under_a_varying_noise():
    r0, kappa = 0.5, 1.0
    likelihood = StateDependentNoiseLikelihood(
        [[1.0]],
        observation_noise_fn=quadratic_noise,
        observation_noise_params=(r0, kappa),
    )
    prior = gaussian_on(STATES, PRIOR_MEAN, PRIOR_VAR)
    by_rung = averaged_inference_gap(
        prior, likelihood, PLUG_IN.build(likelihood), OBSERVATIONS
    )
    by_hand = averaged_inference_gap(
        prior, likelihood, kalman_rule(r0 + kappa * PRIOR_MEAN**2), OBSERVATIONS
    )
    assert by_rung.value > 1e-3
    assert by_rung.value == pytest.approx(by_hand.value, rel=1e-9)


def test_the_belief_smoothed_rung_reproduces_the_hand_rolled_rule_at_its_own_noise():
    r0, kappa = 0.5, 1.0
    likelihood = StateDependentNoiseLikelihood(
        [[1.0]],
        observation_noise_fn=quadratic_noise,
        observation_noise_params=(r0, kappa),
    )
    prior = gaussian_on(STATES, PRIOR_MEAN, PRIOR_VAR)
    by_rung = averaged_inference_gap(
        prior, likelihood, BELIEF_SMOOTHED.build(likelihood), OBSERVATIONS
    )
    by_hand = averaged_inference_gap(
        prior,
        likelihood,
        kalman_rule(r0 + kappa * (PRIOR_MEAN**2 + PRIOR_VAR)),
        OBSERVATIONS,
    )
    assert by_rung.value > 1e-3
    assert by_rung.value == pytest.approx(by_hand.value, rel=1e-9)


@pytest.mark.slow
def test_the_exact_rung_closes_the_gap_where_no_gaussian_rung_can():
    likelihood = StateDependentNoiseLikelihood(
        [[1.0]],
        observation_noise_fn=quadratic_noise,
        observation_noise_params=(0.5, 1.0),
    )
    prior = gaussian_on(STATES, PRIOR_MEAN, PRIOR_VAR)
    exact = averaged_inference_gap(
        prior, likelihood, EXACT_RUNG.build(likelihood), OBSERVATIONS
    )
    gaussian = [
        averaged_inference_gap(prior, likelihood, rung.build(likelihood), OBSERVATIONS)
        for rung in GAUSSIAN_RUNGS
    ]
    assert exact.value < 1e-10
    assert all(gap.value > 1e-3 for gap in gaussian)
    # The plug-in and smoothed rungs read different noise, so they report different
    # gaps. Which is smaller is R7's question and is not asserted here.
    by_name = dict(zip(GAUSSIAN_IDS, (gap.value for gap in gaussian), strict=True))
    assert by_name["plug-in"] != pytest.approx(by_name["belief-smoothed"], rel=1e-3)


# --- the merge gate, part two: the reported gap does not move under o -> λo ---------


def scaled_observations(scale):
    """The observation box in the rescaled units, node for node."""
    return QuadratureGrid(
        lower=[scale * OBSERVATIONS.box[0][0]],
        upper=[scale * OBSERVATIONS.box[0][1]],
        counts=[OBSERVATIONS.size],
    )


@pytest.mark.parametrize(
    "rung", [*GAUSSIAN_RUNGS, EXACT_RUNG], ids=[*GAUSSIAN_IDS, "exact"]
)
@pytest.mark.parametrize("scale", [0.25, 4.0])
def test_the_reported_gap_is_invariant_to_the_observation_units(rung, scale):
    r0, kappa = 0.5, 1.0
    prior = gaussian_on(STATES, PRIOR_MEAN, PRIOR_VAR)
    native = StateDependentNoiseLikelihood(
        [[1.0]],
        observation_noise_fn=quadratic_noise,
        observation_noise_params=(r0, kappa),
    )
    rescaled = StateDependentNoiseLikelihood(
        [[scale]],
        observation_noise_fn=quadratic_noise,
        observation_noise_params=(scale**2 * r0, scale**2 * kappa),
    )

    at_native = averaged_inference_gap(prior, native, rung.build(native), OBSERVATIONS)
    at_scale = averaged_inference_gap(
        prior, rescaled, rung.build(rescaled), scaled_observations(scale)
    )

    assert at_scale.predictive_mass == pytest.approx(at_native.predictive_mass, 1e-12)
    assert at_scale.value == pytest.approx(at_native.value, abs=1e-12)


# --- what a rung asks of the likelihood it is handed --------------------------------


class LaplaceLikelihood:
    """A pointwise-evaluable likelihood that is not a Gaussian channel."""

    is_fixed = True

    def __init__(self, scale):
        self.scale = scale

    def log_likelihood(self, observation, states):
        residual = jnp.asarray(observation, dtype=float)[0] - states[:, 0]
        return -jnp.abs(residual) / self.scale - jnp.log(2.0 * self.scale)


def test_the_exact_rung_takes_any_pointwise_likelihood():
    likelihood = LaplaceLikelihood(0.7)
    prior = gaussian_on(STATES, PRIOR_MEAN, PRIOR_VAR)
    belief = EXACT_RUNG.build(likelihood)(prior, [1.7])
    exact = GridDensity(
        STATES, prior.log_density + likelihood.log_likelihood([1.7], STATES.nodes)
    )
    assert isinstance(belief, GridDensity)
    assert float(exact.kl_to(belief)) < 1e-12


@pytest.mark.parametrize("rung", GAUSSIAN_RUNGS, ids=GAUSSIAN_IDS)
def test_a_gaussian_rung_refuses_a_likelihood_that_is_not_a_channel(rung):
    with pytest.raises(TypeError, match="reads a GaussianChannel"):
        rung.build(LaplaceLikelihood(0.7))


def undefined_noise(states, params):
    """A noise the scheme cannot take a step against."""
    return jnp.full((states.shape[0], 1, 1), jnp.nan)


def test_the_iterated_rung_says_when_a_step_was_not_finite():
    likelihood = StateDependentNoiseLikelihood(
        [[1.0]], observation_noise_fn=undefined_noise
    )
    prior = gaussian_on(STATES, PRIOR_MEAN, PRIOR_VAR)
    answer = ITERATED_RUNG.build(likelihood)(prior, [1.7])
    assert isinstance(answer, Void)
    assert answer.iterations == 1
    assert "not finite after 1 iterations" in answer.detail
    assert "budget" not in answer.detail


# --- the ladder is a declared set ---------------------------------------------------


def test_a_rung_needs_a_name():
    with pytest.raises(ValueError, match="needs a name"):
        Rung(name="", kind=RungKind.PLUG_IN)


def test_the_ladder_holds_its_top():
    ladder = RuleLadder(rungs=(PLUG_IN, EXACT_RUNG), version="test-1")
    assert ladder.names == ("plug-in", "exact")
    assert ladder.size == 2


def test_the_ladder_is_versioned():
    with pytest.raises(ValueError, match="declared and versioned"):
        RuleLadder(rungs=(PLUG_IN, EXACT_RUNG), version="")


def test_the_ladder_is_not_empty():
    with pytest.raises(ValueError, match="at least one member"):
        RuleLadder(rungs=(), version="test-1")


def test_the_ladder_refuses_a_duplicate_name():
    twice = Rung(name="plug-in", kind=RungKind.EXACT)
    with pytest.raises(ValueError, match="duplicate name"):
        RuleLadder(rungs=(PLUG_IN, twice), version="test-1")


def test_the_ladder_refuses_a_set_with_no_exact_reference():
    with pytest.raises(ValueError, match="include EXACT_RUNG"):
        RuleLadder(rungs=(PLUG_IN,), version="test-1")


def test_build_all_pairs_every_rung_with_its_name_in_order():
    likelihood = FixedNoiseLikelihood([[1.0]], observation_noise=[[0.5]])
    prior = gaussian_on(STATES, PRIOR_MEAN, PRIOR_VAR)
    ladder = RuleLadder(rungs=(EXACT_RUNG, BELIEF_SMOOTHED, PLUG_IN), version="test-1")

    built = ladder.build_all(likelihood)

    assert tuple(name for name, _ in built) == ("exact", "belief-smoothed", "plug-in")
    for _, rule in built:
        belief = rule(prior, [1.7])
        assert isinstance(belief, GridDensity)
        assert belief.grid.same_lattice_as(prior.grid)


def test_the_declared_ladder_is_the_five_rungs_in_order():
    assert LADDER.version == "v1"
    assert LADDER.names == (
        "plug-in",
        "modified-single-step",
        "modified-iterated",
        "belief-smoothed",
        "exact",
    )
    assert tuple(rung.kind for rung in LADDER.rungs) == tuple(RungKind)


def test_every_declared_rung_answers_a_fixed_channel():
    likelihood = FixedNoiseLikelihood([[1.0]], observation_noise=[[0.5]])
    prior = gaussian_on(STATES, PRIOR_MEAN, PRIOR_VAR)
    for _, rule in LADDER.build_all(likelihood):
        belief = rule(prior, [1.7])
        assert isinstance(belief, GridDensity)
        assert belief.grid.same_lattice_as(prior.grid)


# --- the noise a Gaussian rung reads ------------------------------------------------


def test_a_fixed_likelihood_reports_its_noise_at_every_state():
    likelihood = FixedNoiseLikelihood([[1.0, 0.0]], observation_noise=[[0.5]])
    noise = likelihood.observation_noise_at(jnp.zeros((5, 2)))
    assert noise.shape == (5, 1, 1)
    np.testing.assert_array_equal(np.asarray(noise), 0.5)


def test_a_state_dependent_likelihood_reports_its_noise_at_each_state():
    likelihood = StateDependentNoiseLikelihood(
        [[1.0]],
        observation_noise_fn=quadratic_noise,
        observation_noise_params=(0.5, 2.0),
    )
    states = jnp.array([[0.0], [1.0], [-2.0]])
    noise = likelihood.observation_noise_at(states)
    np.testing.assert_allclose(np.asarray(noise)[:, 0, 0], [0.5, 2.5, 8.5])


def test_a_noise_function_of_the_wrong_shape_is_refused():
    def flat_noise(states, params):
        return jnp.full((states.shape[0],), 0.5)

    likelihood = StateDependentNoiseLikelihood([[1.0]], observation_noise_fn=flat_noise)
    with pytest.raises(ValueError, match="one covariance per state"):
        likelihood.observation_noise_at(jnp.zeros((3, 1)))
