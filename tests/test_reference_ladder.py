"""The rule ladder: two rungs and the declared set they sit in.

Every rung is checked against the closed-form Kalman posterior where that is the exact
filter, and against the averaged gap where it is not. The ladder itself is checked the
way the other declared sets are: versioned, non-empty, unique, and holding its top.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp.reference.gap import averaged_inference_gap
from cpomdp.reference.ladder import EXACT_RUNG, RuleLadder, Rung, RungKind
from cpomdp.reference.likelihood import (
    FixedNoiseLikelihood,
    StateDependentNoiseLikelihood,
)
from cpomdp.reference.quadrature import GridDensity, QuadratureGrid, gaussian_on

PLUG_IN = Rung(name="plug-in", kind=RungKind.PLUG_IN)

PRIOR_MEAN, PRIOR_VAR = 0.3, 0.8
STATES = QuadratureGrid(lower=[-14.0], upper=[14.0], counts=[1601])
OBSERVATIONS = QuadratureGrid(lower=[-12.0], upper=[12.0], counts=[401])


def quadratic_noise(states, params):
    """R(x) = R0 + kappa * x1^2, one 1x1 covariance per state."""
    r0, kappa = params
    return (r0 + kappa * states[:, :1] ** 2)[:, :, None]


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


def test_the_plug_in_rung_is_the_kalman_update_under_a_fixed_noise():
    likelihood = FixedNoiseLikelihood([[1.0]], observation_noise=[[0.5]])
    prior = gaussian_on(STATES, PRIOR_MEAN, PRIOR_VAR)
    belief = PLUG_IN.build(likelihood)(prior, [1.7])
    mean, cov = kalman_update(PRIOR_MEAN, PRIOR_VAR, 1.0, 0.5, 1.7)

    assert isinstance(belief, GridDensity)
    assert belief.grid.same_lattice_as(prior.grid)
    np.testing.assert_allclose(np.asarray(belief.mean), mean, atol=1e-12)
    np.testing.assert_allclose(np.asarray(belief.cov), cov, atol=1e-12)


def test_the_plug_in_rung_agrees_with_kalman_on_a_two_dimensional_state():
    c = np.array([[1.0, 0.5]])
    r = np.array([[0.4]])
    prior_mean = np.array([0.2, -0.3])
    prior_cov = np.array([[0.5, 0.1], [0.1, 0.3]])
    states = QuadratureGrid(lower=[-8.0, -8.0], upper=[8.0, 8.0], counts=[321, 321])
    likelihood = FixedNoiseLikelihood(c, observation_noise=r)
    prior = gaussian_on(states, prior_mean, prior_cov)

    belief = PLUG_IN.build(likelihood)(prior, [0.9])
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


def test_the_plug_in_rung_closes_the_gap_under_a_fixed_noise():
    likelihood = FixedNoiseLikelihood([[1.0]], observation_noise=[[0.5]])
    gap = averaged_inference_gap(
        gaussian_on(STATES, PRIOR_MEAN, PRIOR_VAR),
        likelihood,
        PLUG_IN.build(likelihood),
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
    plug_in = averaged_inference_gap(
        prior, likelihood, PLUG_IN.build(likelihood), OBSERVATIONS
    )
    assert exact.value < 1e-10
    assert plug_in.value > 1e-3


# --- the merge gate, part two: the reported gap does not move under o -> λo ---------


def scaled_observations(scale):
    """The observation box in the rescaled units, node for node."""
    return QuadratureGrid(
        lower=[scale * OBSERVATIONS.box[0][0]],
        upper=[scale * OBSERVATIONS.box[0][1]],
        counts=[OBSERVATIONS.size],
    )


@pytest.mark.parametrize("rung", [PLUG_IN, EXACT_RUNG], ids=["plug-in", "exact"])
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
    ladder = RuleLadder(rungs=(EXACT_RUNG, PLUG_IN), version="test-1")

    built = ladder.build_all(likelihood)

    assert tuple(name for name, _ in built) == ("exact", "plug-in")
    for _, rule in built:
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
