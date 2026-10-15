"""Numerical hygiene of the H-step rollout, read from `policy_efe_trace`.

One `S` inversion at H=1 is benign; the re-inversions under `R(x)` against a
monotonically contracting `Σ_post` are where precision quietly leaks. These checks
apply the condition-number discipline to the rollout itself, reading the per-step
matrices the trace already surfaces — the assertions live on the host, outside any
`jit`/`vmap`, which is the only place they can raise.

The locks:

- `TestRolloutConditioning`: per-step `cond(Σ⁺)`, `cond(S)`, `cond(Σ_post)` stay under a
  declared ceiling and `Σ_post`'s smallest eigenvalue stays above a declared floor, and
  every matrix is positive definite (PD) across the horizon. `rollout_conditioning`
  computes it on the host from the trace.
- `TestSlogdetSignGuarded`: the log-determinant guards do NOT discard the sign — a
  non-positive-definite matrix yields NaN, not a plausible-looking number, in both the
  kernel's `_logdet_pd` and the host oracle `epistemic_value`.
- `TestX64Enabled`: `jax_enable_x64` is on, so nothing silently downcasts to float32
  inside the scan (invisible at H=1, fatal at H=3).

Imports `rollout_conditioning`/`RolloutConditioning`, so until they land this module is
collection-red — the `ImportError` naming `rollout_conditioning` is the build cue.
"""

import jax.numpy as jnp
import numpy as np

from cpomdp.diagnostics import (
    RolloutConditioning,
    epistemic_value,
    rollout_conditioning,
)
from cpomdp.dynamics import CallableProcessNoise
from cpomdp.efe import _logdet_pd, policy_efe_trace
from cpomdp.observation import CallableSensor
from cpomdp.selection import Preference
from cpomdp.types import Belief, LinearGaussianModel

# Declared bars; full justification in warrant_numbers.md.
# float64 loses ~half its significant digits by a condition number of ~1e8; the healthy
# fixtures here sit near 6, so this ceiling flags real degradation, not benign models.
COND_CEILING = 1e8
# A covariance eigenvalue below this signals a near-singular contraction; the fixtures'
# Σ_post stays near 0.1.
MIN_EIG_FLOOR = 1e-9


# --- fixtures: one model per branch the rollout exercises (mirror test_policy_efe) ---
def _model(observation_model=None):
    return LinearGaussianModel(
        dynamics_matrix=[[1.0, 0.1], [0.0, 1.0]],
        observation_matrix=[[1.0, 0.0]],
        dynamics_noise=[[0.1, 0.0], [0.0, 0.1]],
        observation_noise=[[0.5]],
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        control_matrix=[[0.0], [1.0]],
        observation_model=observation_model,
    )


def _belief():
    return Belief(mean=[0.3, -0.2], cov=[[0.7, 0.1], [0.1, 0.4]])


def _obs_preference():
    return Preference(target=[1.0], precision=[[2.0]])


def _state_noise(x, params):
    return jnp.array([[params["base"] + params["slope"] * x[1] ** 2]])


def _callable_model():
    sensor = CallableSensor(
        observation_matrix=[[1.0, 0.0]],
        noise_fn=_state_noise,
        noise_params={"base": jnp.array(0.2), "slope": jnp.array(0.5)},
    )
    return _model(observation_model=sensor)


def _q_well(x, params):
    return jnp.array([[params["base"] + params["slope"] * x[0] ** 2]])


def _internal_q_model():
    pn = CallableProcessNoise(
        q_fn=_q_well, q_params={"base": jnp.array(0.05), "slope": jnp.array(0.4)}
    )
    return LinearGaussianModel(
        dynamics_matrix=[[1.0]],
        observation_matrix=[[1.0]],
        dynamics_noise=[[0.1]],
        observation_noise=[[0.3]],
        prior=Belief(mean=[0.0], cov=[[0.2]]),
        control_matrix=[[1.0]],
        dynamics_noise_model=pn,
    )


def _cases():
    """(name, model, belief, pref) — fixed sensor, R(x), Q(x). All have p = 1."""
    return [
        ("fixed", _model(), _belief(), _obs_preference()),
        ("rx", _callable_model(), _belief(), _obs_preference()),
        (
            "qx",
            _internal_q_model(),
            Belief(mean=[0.0], cov=[[0.2]]),
            Preference(target=[0.0], precision=[[1.0]]),
        ),
    ]


def _policy(values):
    return jnp.array([[v] for v in values])


class TestRolloutConditioning:
    """Per-step conditioning of the rollout stays within declared bars, all PD."""

    def test_healthy_fixtures_within_bars(self):
        for name, model, belief, pref in _cases():
            trace = policy_efe_trace(model, belief, _policy([0.4, -0.2, 0.1]), pref)
            rc = rollout_conditioning(trace)
            assert isinstance(rc, RolloutConditioning)
            assert rc.all_positive_definite, f"{name}: a rollout matrix is not PD"
            assert np.all(rc.cond_sigma_pred < COND_CEILING), f"{name}: cond(Σ⁺)"
            assert np.all(rc.cond_s < COND_CEILING), f"{name}: cond(S)"
            assert np.all(rc.cond_sigma_post < COND_CEILING), f"{name}: cond(Σ_post)"
            assert np.all(rc.min_eig_sigma_post > MIN_EIG_FLOOR), f"{name}: min eig"

    def test_per_step_shapes(self):
        model, belief, pref = _model(), _belief(), _obs_preference()
        rc = rollout_conditioning(
            policy_efe_trace(model, belief, _policy([0.4, -0.2, 0.1]), pref)
        )
        for column in (
            rc.cond_sigma_pred,
            rc.cond_s,
            rc.cond_sigma_post,
            rc.min_eig_sigma_post,
        ):
            assert column.shape == (3,)

    def test_flags_a_near_singular_step(self):
        # A trace whose last Σ_post is near-singular must trip the floor, proving the
        # check bites, not passes everything. Built by hand off a real trace.
        model, belief, pref = _model(), _belief(), _obs_preference()
        trace = policy_efe_trace(model, belief, _policy([0.4, -0.2]), pref)
        bad_post = np.array(trace.sigma_post)  # writable (jax arrays are read-only)
        bad_post[-1] = np.array([[1e-15, 0.0], [0.0, 1e-15]])
        degraded = trace._replace(sigma_post=jnp.asarray(bad_post))
        rc = rollout_conditioning(degraded)
        assert not np.all(rc.min_eig_sigma_post > MIN_EIG_FLOOR)


class TestSlogdetSignGuarded:
    """A non-PD matrix yields NaN, not a plausible log-det (the sign is kept)."""

    def test_kernel_logdet_is_nan_on_non_pd(self):
        # diag(-1, -2): det = +2 > 0, two negative eigenvalues. A determinant-sign
        # shortcut sees the positive det and returns a finite log|det|, which is wrong.
        # _logdet_pd uses Cholesky, so it catches the non-PD matrix and returns NaN.
        not_pd = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        sign_shortcut = np.linalg.slogdet(np.asarray(not_pd))[1]
        assert np.isfinite(sign_shortcut)  # the sign shortcut is fooled
        assert bool(jnp.isnan(_logdet_pd(not_pd)))  # the Cholesky guard is not

    def test_kernel_logdet_correct_on_pd(self):
        # ln det(2·I₂) = ln 4.
        got = float(_logdet_pd(2.0 * jnp.eye(2)))
        np.testing.assert_allclose(got, np.log(4.0), atol=1e-12)

    def test_oracle_epistemic_value_is_nan_on_non_pd_noise(self):
        cov = np.eye(2)
        c = np.array([[1.0, 0.0]])
        non_pd_r = np.array([[-0.5]])  # a "noise" that is not a covariance
        assert np.isnan(epistemic_value(cov, c, non_pd_r))


class TestX64Enabled:
    """Importing cpomdp enables x64 — no silent float32 downcast inside the scan."""

    def test_x64_is_on(self):
        # x64 shows up as float64 being the default float dtype; without it a scalar
        # array is float32 — the silent downcast this guards against.
        assert jnp.zeros(1).dtype == jnp.float64
