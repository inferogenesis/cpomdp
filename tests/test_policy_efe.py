"""Workstream B / B2-B3 — the H-step rollout `policy_efe`.

`policy_efe(model, belief, policy, preference)` sums the per-step EFE along a
`policy` (shape `(H, p)`) via `lax.scan`, propagating the belief **predict-only**
between steps:

- the mean carries forward as the prediction, `μ_next = μ⁺` — the innovation
  `y − Cμ⁺` has zero expectation (no real future observation);
- the covariance contracts by the Kalman update `Σ_post = Σ⁺ − Σ⁺Cᵀ S⁻¹ C Σ⁺`,
  computed inline from the `(Σ⁺, S)` `_efe_step` already returns plus the `C` the
  rollout fetches itself (`model.C` fixed, else `linearize(μ⁺)[0]`), then
  symmetrized as a PSD guard.

The signature is deliberately `horizon`-free: `H` is `policy.shape[0]` (which
`lax.scan` uses as its static trip count), so a separate kwarg would be a
redundant second source of truth. The `horizon` knob and the constant-action
tiling live one level up in `EFESelector` (B4).

Three locks:

- `TestH1ReducesToOneStepExact` (B2): at H=1 the rollout is **byte-identical** to
  `expected_free_energy` (`assert_array_equal`) — the propagation is computed but
  unused, so `G` is bit-for-bit the one-step value.
- `TestMatchesNumpyOracle` (B3): `_numpy_policy_efe` is an independent plain-NumPy
  rollout (no `lax.scan`, no efe import); `policy_efe` matches it to `1e-9` at
  H=2,3 under fixed sensor, `R(x)`, and `Q(x)` — this drives the propagation math.
- `TestTransforms` / `TestPropagationPSD` (B3): `jit` / `vmap`-over-policies /
  `grad`-over-policy survive; the propagated covariance stays PSD each step.

Imports `policy_efe` directly, so until B2 lands this module is collection-red —
the `ImportError` naming `policy_efe` is the build cue.
"""

import jax
import jax.numpy as jnp
import numpy as np

from cpomdp.dynamics import CallableProcessNoise
from cpomdp.efe import expected_free_energy, policy_efe
from cpomdp.observation import CallableSensor
from cpomdp.selection import Preference
from cpomdp.types import Belief, LinearGaussianModel


# --- fixtures: one model per branch the rollout exercises -------------------------
def _model(observation=None):
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
    """(name, model, belief, pref) — fixed sensor, R(x), Q(x). All have p = 1."""
    return [
        ("fixed", _model(), _belief(), _obs_preference()),
        ("rx", _callable_model(), _belief(), _obs_preference()),
        (
            "qx",
            _internal_q_model(),
            Belief(mean=[0.0], cov=[[0.2]]),
            Preference(goal=[0.0], precision=[[1.0]]),
        ),
    ]


def _policy(values):
    """A (H, 1) policy from a list of scalar actions (p = 1 for every fixture)."""
    return jnp.array([[v] for v in values])


def _numpy_policy_efe(model, belief, policy, goal, precision):
    """Independent NumPy rollout — plain loop, no `lax.scan`, no efe import.

    Mirrors `policy_efe`'s math in a separate library and code path, so agreement is an
    independent confirmation. Returns `(G, covs)`: the summed EFE and the per-step
    propagated covariances (for the PSD check).
    """
    a_mat = np.asarray(model.dynamics)
    b_mat = np.asarray(model.control)
    g = np.asarray(goal, dtype=float)
    lam = np.asarray(precision, dtype=float)
    mu = np.asarray(belief.mean, dtype=float)
    sigma = np.asarray(belief.cov, dtype=float)

    g_total = 0.0
    covs = []
    for a in np.asarray(policy, dtype=float):
        mu_pred = a_mat @ mu + b_mat @ a
        if model.process_noise is None:
            q = np.asarray(model.dynamics_noise)
        else:
            q = np.asarray(model.process_noise.noise_at(mu_pred))
        sigma_pred = a_mat @ sigma @ a_mat.T + q

        if model.observation is None:
            c = np.asarray(model.sensor_model)
            r = np.asarray(model.sensor_noise)
        else:
            c_arr, r_arr = model.observation.linearize(mu_pred)
            c, r = np.asarray(c_arr), np.asarray(r_arr)
        o_pred = c @ mu_pred
        s = c @ sigma_pred @ c.T + r

        resid = o_pred - g
        pragmatic = 0.5 * resid @ lam @ resid + 0.5 * np.trace(lam @ s)
        epistemic = 0.5 * (np.linalg.slogdet(s)[1] - np.linalg.slogdet(r)[1])
        g_total += pragmatic - epistemic

        # predict-only propagation: mean = μ⁺, cov = Σ⁺ − Σ⁺Cᵀ S⁻¹ C Σ⁺ (symmetrized).
        p_xo = sigma_pred @ c.T
        sigma_post = sigma_pred - p_xo @ np.linalg.solve(s, p_xo.T)
        sigma_post = 0.5 * (sigma_post + sigma_post.T)
        covs.append(sigma_post)
        mu, sigma = mu_pred, sigma_post
    return g_total, covs


class TestH1ReducesToOneStepExact:
    """At H=1 the rollout is byte-identical to the one-step kernel (the locked seam)."""

    def test_h1_is_bit_identical_to_expected_free_energy(self):
        action = jnp.array([0.4])  # p = 1 for every fixture
        for name, model, belief, pref in _cases():
            g_roll, parts_roll = policy_efe(model, belief, action[None, :], pref)
            g_one, parts_one = expected_free_energy(model, belief, action, pref)
            np.testing.assert_array_equal(g_roll, g_one, err_msg=f"G H1: {name}")
            np.testing.assert_array_equal(
                parts_roll["pragmatic"],
                parts_one["pragmatic"],
                err_msg=f"pragmatic H1: {name}",
            )
            np.testing.assert_array_equal(
                parts_roll["epistemic"],
                parts_one["epistemic"],
                err_msg=f"epistemic H1: {name}",
            )


class TestMatchesNumpyOracle:
    """Multi-step rollout matches the independent NumPy oracle (drives the math)."""

    def test_oracle_anchored_to_one_step_kernel_at_h1(self):
        # The oracle's per-step EFE must match the canonical one-step kernel, so that
        # matching policy_efe to it at H>1 is meaningful (not two wrongs agreeing).
        action = jnp.array([0.4])
        for name, model, belief, pref in _cases():
            g_ref, _ = _numpy_policy_efe(
                model, belief, action[None, :], pref.goal, pref.precision
            )
            g_one = float(expected_free_energy(model, belief, action, pref)[0])
            np.testing.assert_allclose(g_ref, g_one, atol=1e-9, err_msg=name)

    def test_multistep_matches_oracle(self):
        for name, model, belief, pref in _cases():
            for values in ([0.4, -0.2], [0.4, -0.2, 0.1]):
                policy = _policy(values)
                g = policy_efe(model, belief, policy, pref)[0]
                g_ref, _ = _numpy_policy_efe(
                    model, belief, policy, pref.goal, pref.precision
                )
                np.testing.assert_allclose(
                    g, g_ref, atol=1e-9, err_msg=f"{name} H{len(values)}"
                )


class TestTransforms:
    """The `lax.scan` rollout composes under jit / vmap-over-policies / grad."""

    def test_jit_agrees_with_eager(self):
        model, belief, pref = _model(), _belief(), _obs_preference()
        policy = _policy([0.4, -0.2])
        eager = policy_efe(model, belief, policy, pref)[0]
        jitted = jax.jit(lambda p: policy_efe(model, belief, p, pref)[0])(policy)
        np.testing.assert_allclose(jitted, eager, atol=1e-12)

    def test_vmap_over_policies(self):
        model, belief, pref = _model(), _belief(), _obs_preference()
        policies = jnp.stack(
            [_policy([0.4, -0.2]), _policy([-1.0, 0.5]), _policy([0.0, 0.0])]
        )
        batched = jax.vmap(lambda p: policy_efe(model, belief, p, pref)[0])(policies)
        per = jnp.array([policy_efe(model, belief, p, pref)[0] for p in policies])
        np.testing.assert_allclose(batched, per, atol=1e-12)

    def test_grad_over_policy_runs(self):
        model, belief, pref = _model(), _belief(), _obs_preference()
        grad = jax.grad(lambda p: policy_efe(model, belief, p, pref)[0])(
            _policy([0.4, -0.2])
        )
        assert grad.shape == (2, 1)
        assert bool(jnp.all(jnp.isfinite(grad)))


class TestPropagationPSD:
    """The predict-only contraction stays PSD each step (math soundness).

    Checked on the oracle; `policy_efe`'s G matching the oracle (above) confirms it
    implements the same propagation, so a PSD-breaking bug there would diverge the G.
    """

    def test_oracle_propagated_covs_are_psd(self):
        for name, model, belief, pref in _cases():
            _, covs = _numpy_policy_efe(
                model, belief, _policy([0.8, -0.5, 0.3]), pref.goal, pref.precision
            )
            for t, cov in enumerate(covs):
                eigs = np.linalg.eigvalsh(cov)
                assert float(eigs.min()) > -1e-9, f"{name} step {t} not PSD: {eigs}"
