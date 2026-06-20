"""Workstream B / B1 — the `_efe_step` extraction, proven inert.

efe.py's one-step kernel is being split (Fowler "Extract Function") into a private
`_efe_step` that the H-step rollout (`policy_efe`, B2) will `lax.scan` over. B1's whole
job is to make that split **arithmetically inert**: the public `expected_free_energy`
keeps returning a byte-identical `(G, parts)`, and `_efe_step` additionally surfaces the
three intermediates the rollout consumes — `(μ⁺, Σ⁺, S)` — with **no `C`** (the rollout
fetches its own `C` only where it propagates; see the build plan Phase 3 / B1).

Target: `_efe_step(model, mu, sigma, control, action, goal, precision)` returns an
internal `_EfeStep` (g, pragmatic, epistemic, mu_pred, sigma_pred, s) — the public
split plus the moments B2 propagates.

Two locks live here:

- `_frozen_efe` is a verbatim snapshot of the pre-extraction kernel arithmetic. An inert
  extraction reproduces it to the ULP — asserted with `assert_array_equal`, NOT
  `allclose` (the project's byte-identical discipline, RFC-001). Robust to
  XLA drift: both sides move together, so only a real arithmetic change trips it. Covers
  all three branches the extraction moves: fixed (`None`), `R(x)`, `Q(x)`.
- `TestEfeStepContract` pins the signature, the `_EfeStep` fields, the byte-identical
  `(G, parts)` the wrapper exposes, and the three intermediates' values (vs independent
  NumPy) — including a shape check that `.s` is `S` (m×m), not `C` (m×n).

This module imports `_efe_step` directly, so until B1 lands it is **collection-red** —
the `ImportError` naming `_efe_step` is the build cue. The gate BEYOND this file: the
entire existing `test_efe.py` must pass UNMODIFIED.
"""

import jax.numpy as jnp
import numpy as np

from cpomdp.dynamics import CallableProcessNoise
from cpomdp.efe import _efe_step, expected_free_energy
from cpomdp.observation import CallableSensor
from cpomdp.selection import Preference
from cpomdp.types import Belief, LinearGaussianModel


# --- fixtures: one model per branch the extraction touches ------------------------
def _model(observation=None):
    # 2-state, 1-observation, 1-action, controllable (mirrors test_efe.py).
    return LinearGaussianModel(
        dynamics=[[1.0, 0.1], [0.0, 1.0]],
        sensor_model=[[1.0, 0.0]],
        dynamics_noise=[[0.1, 0.0], [0.0, 0.1]],
        sensor_noise=[[0.5]],
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        control=[[0.0], [1.0]],
        observation=observation,
    )


def _belief():
    return Belief(mean=[0.3, -0.2], cov=[[0.7, 0.1], [0.1, 0.4]])


def _obs_preference():
    return Preference(goal=[1.0], precision=[[2.0]])


def _state_noise(x, params):
    return jnp.array([[params["base"] + params["slope"] * x[1] ** 2]])


def _callable_model():
    sensor = CallableSensor(
        sensor_model=[[1.0, 0.0]],
        noise_fn=_state_noise,
        noise_params={"base": jnp.array(0.2), "slope": jnp.array(0.5)},
    )
    return _model(observation=sensor)


def _q_well(x, params):
    return jnp.array([[params["base"] + params["slope"] * x[0] ** 2]])


def _internal_q_model():
    pn = CallableProcessNoise(
        q_fn=_q_well, q_params={"base": jnp.array(0.05), "slope": jnp.array(0.4)}
    )
    return LinearGaussianModel(
        dynamics=[[1.0]],
        sensor_model=[[1.0]],
        dynamics_noise=[[0.1]],
        sensor_noise=[[0.3]],
        prior=Belief(mean=[0.0], cov=[[0.2]]),
        control=[[1.0]],
        process_noise=pn,
    )


def _cases():
    """The three kernel branches B1 must leave bit-identical: fixed, R(x), Q(x)."""
    return [
        ("fixed_none", _model(), _belief(), _obs_preference(), jnp.array([0.4])),
        (
            "callable_rx",
            _callable_model(),
            _belief(),
            _obs_preference(),
            jnp.array([0.4]),
        ),
        (
            "internal_qx",
            _internal_q_model(),
            Belief(mean=[0.0], cov=[[0.2]]),
            Preference(goal=[0.0], precision=[[1.0]]),
            jnp.array([0.7]),
        ),
    ]


def _frozen_efe(model, belief, action, preference):
    """Verbatim snapshot of efe.py's kernel arithmetic (lines 147-204), pre-B1.

    The "before" half of the characterization test: an inert `_efe_step` extraction
    must make `expected_free_energy` reproduce this bit-for-bit. Do NOT refactor this
    to call the new kernel — its whole value is being a frozen, independent copy.
    """
    control = model.control
    assert control is not None  # mirrors the kernel arithmetic past the control guard
    action = jnp.asarray(action, dtype=float)
    mu, sigma = belief.mean, belief.cov

    mu_pred = model.A @ mu + control @ action
    process_q = (
        model.Q
        if model.process_noise is None
        else model.process_noise.noise_at(mu_pred)
    )
    sigma_pred = model.A @ sigma @ model.A.T + process_q

    if model.observation is None:
        sensor_model, sensor_noise = model.C, model.R
        o_pred = sensor_model @ mu_pred
        pred_obs_cov = sensor_model @ sigma_pred @ sensor_model.T + sensor_noise
    else:
        o_pred, pred_obs_cov, sensor_noise = model.observation.gaussianize(
            mu_pred, sigma_pred
        )

    goal, precision = preference.goal, preference.precision
    residual = o_pred - goal
    pragmatic_mean = 0.5 * residual @ precision @ residual
    pragmatic_var = 0.5 * jnp.trace(precision @ pred_obs_cov)
    pragmatic = pragmatic_mean + pragmatic_var

    _, logdet_pred_obs = jnp.linalg.slogdet(pred_obs_cov)
    _, logdet_noise = jnp.linalg.slogdet(sensor_noise)
    epistemic = 0.5 * (logdet_pred_obs - logdet_noise)

    g = pragmatic - epistemic
    return g, {"pragmatic": pragmatic, "epistemic": epistemic}


class TestExtractionInert:
    """The wrapper stays byte-identical to the frozen pre-B1 kernel (all branches).

    Green once `_efe_step` exists and is inert; goes red if the refactor changes any
    arithmetic. If it trips, the extraction was not inert — stop and fix.
    """

    def test_wrapper_is_byte_identical_to_frozen_kernel(self):
        for name, model, belief, pref, action in _cases():
            g, parts = expected_free_energy(model, belief, action, pref)
            g_frozen, parts_frozen = _frozen_efe(model, belief, action, pref)
            np.testing.assert_array_equal(g, g_frozen, err_msg=f"G non-inert: {name}")
            np.testing.assert_array_equal(
                parts["pragmatic"],
                parts_frozen["pragmatic"],
                err_msg=f"pragmatic non-inert: {name}",
            )
            np.testing.assert_array_equal(
                parts["epistemic"],
                parts_frozen["epistemic"],
                err_msg=f"epistemic non-inert: {name}",
            )


class TestEfeStepContract:
    """The rollout's per-step kernel — returns the moments B2 will propagate."""

    def test_returns_g_parts_and_three_intermediates_no_C(self):
        model, belief, pref = _model(), _belief(), _obs_preference()
        action = jnp.asarray([0.4], dtype=float)

        step = _efe_step(
            model,
            belief.mean,
            belief.cov,
            model.control,
            action,
            pref.goal,
            pref.precision,
        )

        # the (G, parts) the wrapper exposes must be exactly these — bit-for-bit.
        g_ref, parts_ref = expected_free_energy(model, belief, action, pref)
        np.testing.assert_array_equal(step.g, g_ref)
        np.testing.assert_array_equal(step.pragmatic, parts_ref["pragmatic"])
        np.testing.assert_array_equal(step.epistemic, parts_ref["epistemic"])

        # shapes pin the contract — and prove `.s` is S (m×m), NOT C (m×n):
        # for this model n=2, m=1, so C would be (1, 2) and S is (1, 1).
        assert step.mu_pred.shape == (2,)
        assert step.sigma_pred.shape == (2, 2)
        assert step.s.shape == (1, 1)

        # the three intermediates are the real moments B2 consumes (vs NumPy).
        a_mat = np.asarray(model.dynamics)
        b_mat = np.asarray(model.control)
        q_mat = np.asarray(model.dynamics_noise)
        c_mat = np.asarray(model.sensor_model)
        r_mat = np.asarray(model.sensor_noise)
        mu = np.asarray(belief.mean)
        sigma = np.asarray(belief.cov)
        a = np.asarray(action, dtype=float)
        sigma_pred_ref = a_mat @ sigma @ a_mat.T + q_mat
        np.testing.assert_allclose(step.mu_pred, a_mat @ mu + b_mat @ a, atol=1e-12)
        np.testing.assert_allclose(step.sigma_pred, sigma_pred_ref, atol=1e-12)
        np.testing.assert_allclose(
            step.s, c_mat @ sigma_pred_ref @ c_mat.T + r_mat, atol=1e-12
        )

    def test_matches_wrapper_on_state_dependent_sensor(self):
        # The gaussianize branch must also surface a valid S and a bit-identical G.
        model, belief, pref = _callable_model(), _belief(), _obs_preference()
        action = jnp.asarray([0.4], dtype=float)

        step = _efe_step(
            model,
            belief.mean,
            belief.cov,
            model.control,
            action,
            pref.goal,
            pref.precision,
        )
        g_ref, _ = expected_free_energy(model, belief, action, pref)
        np.testing.assert_array_equal(step.g, g_ref)
        assert step.s.shape == (1, 1)
