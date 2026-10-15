"""Node-targeted epistemic value for the FFG EFE (issue #26, Phase A).

The epistemic term generalises from whole-state info gain (the observation-space
½(ln det S − ln det R) the kernel computes today) to info gain about a *chosen* latent's
marginal, in state space: ½(ln det Σ⁺[info_block] − ln det Σ_post[info_block]).
This file pins
the kernel — first the full-state anchor (it must reproduce the observation-space
number), then the node-restricted case.
"""

import jax.numpy as jnp
import numpy as np

from cpomdp.efe import _ffg_efe_step, _state_info_gain, expected_free_energy
from cpomdp.selection import Preference
from cpomdp.types import Belief, LinearGaussianModel


def _spd(rng, n):
    """A random n x n symmetric positive-definite matrix."""
    a = rng.standard_normal((n, n))
    return a @ a.T + n * np.eye(n)


def test_full_state_info_gain_equals_observation_space_form():
    # info_block = the WHOLE state: the state-space info gain must reproduce today's
    # observation-space epistemic ½(ln det S − ln det R), with S = C·Σ⁺·Cᵀ + R.
    rng = np.random.default_rng(0)
    n, m = 4, 2
    sigma_pred = jnp.asarray(_spd(rng, n))  # Σ⁺, the predicted joint covariance
    observation_matrix = jnp.asarray(rng.standard_normal((m, n)))  # C
    observation_noise = jnp.asarray(_spd(rng, m))  # R

    # The observation-space number the current kernel computes.
    s = observation_matrix @ sigma_pred @ observation_matrix.T + observation_noise  # S
    _, logdet_s = np.linalg.slogdet(s)
    _, logdet_r = np.linalg.slogdet(observation_noise)
    expected = 0.5 * (logdet_s - logdet_r)

    # The new state-space route, restricted to the whole state (all indices).
    got = _state_info_gain(
        sigma_pred, observation_matrix, observation_noise, info_block=range(n)
    )
    np.testing.assert_allclose(float(got), expected, atol=1e-10)


def test_node_info_gain_matches_independent_moment_update():
    # Restrict info_block to ONE node's block, then check against Σ_post computed
    # the *other*
    # way — the Kalman moment update Σ_post = Σ⁺ − K·C·Σ⁺ (the function uses info form).
    rng = np.random.default_rng(1)
    n, m = 5, 2
    sigma_pred = _spd(rng, n)  # Σ⁺
    observation_matrix = rng.standard_normal((m, n))  # C
    observation_noise = _spd(rng, m)  # R
    info_block = [1, 2]  # a node occupying state indices 1..2

    # Independent (moment-form) posterior covariance, then the block log-det drop.
    s = observation_matrix @ sigma_pred @ observation_matrix.T + observation_noise
    gain = sigma_pred @ observation_matrix.T @ np.linalg.inv(s)  # K
    sigma_post = sigma_pred - gain @ observation_matrix @ sigma_pred
    block = np.ix_(info_block, info_block)
    _, logdet_pred = np.linalg.slogdet(sigma_pred[block])
    _, logdet_post = np.linalg.slogdet(sigma_post[block])
    expected = 0.5 * (logdet_pred - logdet_post)

    got = _state_info_gain(
        jnp.asarray(sigma_pred),
        jnp.asarray(observation_matrix),
        jnp.asarray(observation_noise),
        info_block=info_block,
    )
    np.testing.assert_allclose(float(got), expected, atol=1e-9)


def test_full_state_gain_matches_expected_free_energy_epistemic():
    # The production tie: on a real model, the whole-state info gain must reproduce the
    # epistemic that `expected_free_energy` actually returns (Σ⁺ = A·Σ·Aᵀ + Q).
    dynamics = np.array([[1.0, 0.1], [0.0, 1.0]])  # A
    dynamics_noise = np.array([[0.1, 0.0], [0.0, 0.1]])  # Q
    observation_matrix = np.array([[1.0, 0.0]])  # C
    observation_noise = np.array([[0.5]])  # R
    model = LinearGaussianModel(
        dynamics_matrix=dynamics,
        observation_matrix=observation_matrix,
        dynamics_noise=dynamics_noise,
        observation_noise=observation_noise,
        prior=Belief(mean=[0.0, 0.0], cov=np.eye(2)),
        control_matrix=[[0.0], [1.0]],
    )
    belief = Belief(mean=[0.3, -0.2], cov=[[0.7, 0.1], [0.1, 0.4]])
    action = jnp.array([0.5])
    preference = Preference(target=[1.0], precision=[[2.0]])

    _, parts = expected_free_energy(model, belief, action, preference)

    sigma = np.asarray(belief.cov)
    sigma_pred = dynamics @ sigma @ dynamics.T + dynamics_noise  # Σ⁺
    got = _state_info_gain(
        jnp.asarray(sigma_pred),
        jnp.asarray(observation_matrix),
        jnp.asarray(observation_noise),
        info_block=range(2),
    )
    np.testing.assert_allclose(float(got), float(parts["epistemic"]), atol=1e-9)


def test_ffg_efe_step_reduces_to_expected_free_energy():
    # B2 (issue #26): with no structural couplings and info_block = the whole state, the
    # FFG EFE step must reproduce expected_free_energy — same pragmatic, epistemic, G.
    dynamics = np.array([[1.0, 0.1], [0.0, 1.0]])  # A
    dynamics_noise = np.array([[0.1, 0.0], [0.0, 0.1]])  # Q
    observation_matrix = np.array([[1.0, 0.0]])  # C
    observation_noise = np.array([[0.5]])  # R
    control = np.array([[0.0], [1.0]])  # B
    model = LinearGaussianModel(
        dynamics_matrix=dynamics,
        observation_matrix=observation_matrix,
        dynamics_noise=dynamics_noise,
        observation_noise=observation_noise,
        prior=Belief(mean=[0.0, 0.0], cov=np.eye(2)),
        control_matrix=control,
    )
    belief = Belief(mean=[0.3, -0.2], cov=[[0.7, 0.1], [0.1, 0.4]])
    action = jnp.array([0.5])
    preference = Preference(target=[1.0], precision=[[2.0]])

    g_ref, parts_ref = expected_free_energy(model, belief, action, preference)

    # The predicted moments the FFG step is handed (no couplings: μ⁺ = Aμ + Ba,
    # Σ⁺ = AΣAᵀ + Q).
    mu_plus = dynamics @ np.asarray(belief.mean) + control @ action
    sigma_plus = dynamics @ np.asarray(belief.cov) @ dynamics.T + dynamics_noise
    g, parts = _ffg_efe_step(
        jnp.asarray(mu_plus),
        jnp.asarray(sigma_plus),
        jnp.asarray(observation_matrix),
        jnp.asarray(observation_noise),
        preference.target,
        preference.precision,
        info_block=range(2),
    )
    np.testing.assert_allclose(float(g), float(g_ref), atol=1e-9)
    np.testing.assert_allclose(
        float(parts["pragmatic"]), float(parts_ref["pragmatic"]), atol=1e-9
    )
    np.testing.assert_allclose(
        float(parts["epistemic"]), float(parts_ref["epistemic"]), atol=1e-9
    )
