"""`policy_efe_trace` — the per-step trace from the `policy_efe` rollout.

`policy_efe_trace(model, belief, policy, preference)` returns the `lax.scan` `ys`
that `policy_efe` computes but sums-and-discards: per-step `(g, pragmatic, epistemic,
μ⁺, Σ⁺, Σ_post, S)` as a `PolicyEfeTrace` NamedTuple of arrays stacked along a leading
H axis. It is a diagnostic surface; it is NOT on the `EFESelector` hot path — the
selector still calls `policy_efe(...)[0]`, whose `ys` stays the three lean scalars, so
no H×n×n covariance is stacked under its `vmap`.

Naming follows `efe.py`'s convention: `⁺` marks the one-step *prediction* (after the
dynamics and action, before any observation), so `mu_pred`/`sigma_pred` are μ⁺/Σ⁺;
`sigma_post` is Σ_post, the covariance after the predict-only Kalman contraction; `s`
is the innovation covariance S.

The locks:

- `TestTraceSumsEqualScalars`: `jnp.sum` of each traced scalar column is
  **byte-identical** (`assert_array_equal`) to `policy_efe`'s returned scalar, across
  fixed / R(x) / Q(x) at H=1,2,3 — proof it is the same arithmetic, not a second impl.
- `TestTraceMatchesNumpyOracle`: the per-step moments and split match an independent
  plain-NumPy rollout (`_numpy_policy_efe_trace`, no `lax.scan`, no efe import) at 1e-9,
  with the stacked shapes pinned (the `(H, 1, 1)` `s` proving it is S, not C).
- `TestTraceH1MatchesEfeStep`: at H=1 the traced predict moments equal `_efe_step`'s
  fields byte-for-byte, pinning the field→symbol mapping.
- `TestTraceTransforms` / `TestTraceIsPytree`: jit / vmap-over-policies / grad survive,
  and the NamedTuple is a pytree with seven leaves (no registration needed).

Imports `policy_efe_trace`/`PolicyEfeTrace` directly, so until they land this module is
collection-red — the `ImportError` naming `policy_efe_trace` is the build cue.
"""

import jax
import jax.numpy as jnp
import numpy as np

from cpomdp.dynamics import CallableDynamicsNoise
from cpomdp.efe import PolicyEfeTrace, _efe_step, policy_efe, policy_efe_trace
from cpomdp.observation import CallableSensor
from cpomdp.selection import Preference
from cpomdp.types import Belief, LinearGaussianModel


# --- fixtures: one model per branch the rollout exercises (mirror test_policy_efe) ---
def _model(observation_model=None):
    return LinearGaussianModel(
        dynamics_matrix=[[1.0, 0.1], [0.0, 1.0]],
        observation_matrix=[[1.0, 0.0]] if observation_model is None else None,
        dynamics_noise=[[0.1, 0.0], [0.0, 0.1]],
        observation_noise=[[0.5]] if observation_model is None else None,
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        control_matrix=[[0.0], [1.0]],
        observation_model=observation_model,
    )


def _belief():
    return Belief(mean=[0.3, -0.2], cov=[[0.7, 0.1], [0.1, 0.4]])


def _obs_preference():
    return Preference(goal=[1.0], precision=[[2.0]])


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
    dynamics_noise_model = CallableDynamicsNoise(
        noise_fn=_q_well,
        noise_params={"base": jnp.array(0.05), "slope": jnp.array(0.4)},
    )
    return LinearGaussianModel(
        dynamics_matrix=[[1.0]],
        observation_matrix=[[1.0]],
        observation_noise=[[0.3]],
        prior=Belief(mean=[0.0], cov=[[0.2]]),
        control_matrix=[[1.0]],
        dynamics_noise_model=dynamics_noise_model,
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


def _numpy_policy_efe_trace(model, belief, policy, goal, precision):
    """Independent NumPy rollout returning the full per-step trace — no `lax.scan`.

    A second copy of the rollout math in plain NumPy (a different library and code
    path), extended to record every per-step quantity `policy_efe_trace` surfaces, so
    the moment/split agreement is an independent confirmation. Returns a dict of arrays
    stacked along a leading H axis: `g, pragmatic, epistemic, mu_pred (μ⁺),
    sigma_pred (Σ⁺), sigma_post (Σ_post), s (S)`.
    """
    a_mat = np.asarray(model.dynamics_matrix)
    b_mat = np.asarray(model.control_matrix)
    g = np.asarray(goal, dtype=float)
    lam = np.asarray(precision, dtype=float)
    mu = np.asarray(belief.mean, dtype=float)
    sigma = np.asarray(belief.cov, dtype=float)

    cols = {
        k: []
        for k in (
            "g",
            "pragmatic",
            "epistemic",
            "mu_pred",
            "sigma_pred",
            "sigma_post",
            "s",
        )
    }
    for a in np.asarray(policy, dtype=float):
        mu_pred = a_mat @ mu + b_mat @ a
        if model.dynamics_noise_model is None:
            q = np.asarray(model.dynamics_noise)
        else:
            q = np.asarray(model.dynamics_noise_model.noise_at(mu_pred))
        sigma_pred = a_mat @ sigma @ a_mat.T + q

        if model.observation_model is None:
            c = np.asarray(model.observation_matrix)
            r = np.asarray(model.observation_noise)
        else:
            c_arr, r_arr = model.observation_model.linearize(mu_pred)
            c, r = np.asarray(c_arr), np.asarray(r_arr)
        o_pred = c @ mu_pred
        s = c @ sigma_pred @ c.T + r

        resid = o_pred - g
        pragmatic = 0.5 * resid @ lam @ resid + 0.5 * np.trace(lam @ s)
        epistemic = 0.5 * (np.linalg.slogdet(s)[1] - np.linalg.slogdet(r)[1])

        # predict-only propagation: mean = μ⁺, cov = Σ_post = Σ⁺ − Σ⁺Cᵀ S⁻¹ C Σ⁺ (symm).
        p_xo = sigma_pred @ c.T
        sigma_post = sigma_pred - p_xo @ np.linalg.solve(s, p_xo.T)
        sigma_post = 0.5 * (sigma_post + sigma_post.T)

        cols["g"].append(pragmatic - epistemic)
        cols["pragmatic"].append(pragmatic)
        cols["epistemic"].append(epistemic)
        cols["mu_pred"].append(mu_pred)
        cols["sigma_pred"].append(sigma_pred)
        cols["sigma_post"].append(sigma_post)
        cols["s"].append(s)
        mu, sigma = mu_pred, sigma_post
    return {k: np.stack(v) for k, v in cols.items()}


class TestTraceSumsEqualScalars:
    """The central lock: summed trace columns are byte-identical to `policy_efe`."""

    def test_sums_equal_policy_efe_scalars(self):
        for name, model, belief, pref in _cases():
            for values in ([0.4], [0.4, -0.2], [0.4, -0.2, 0.1]):  # H = 1, 2, 3
                policy = _policy(values)
                trace = policy_efe_trace(model, belief, policy, pref)
                efe = policy_efe(model, belief, policy, pref)
                tag = f"{name} H{len(values)}"
                np.testing.assert_array_equal(
                    jnp.sum(trace.g), efe.g, err_msg=f"G {tag}"
                )
                np.testing.assert_array_equal(
                    jnp.sum(trace.pragmatic),
                    efe.pragmatic,
                    err_msg=f"pragmatic {tag}",
                )
                np.testing.assert_array_equal(
                    jnp.sum(trace.epistemic),
                    efe.epistemic,
                    err_msg=f"epistemic {tag}",
                )


class TestTraceMatchesNumpyOracle:
    """Per-step moments and split match the independent NumPy oracle; shapes pinned."""

    def test_moments_match_oracle(self):
        for name, model, belief, pref in _cases():
            for values in ([0.4, -0.2], [0.4, -0.2, 0.1]):  # H = 2, 3
                policy = _policy(values)
                trace = policy_efe_trace(model, belief, policy, pref)
                ref = _numpy_policy_efe_trace(
                    model, belief, policy, pref.goal, pref.precision
                )
                for field in ref:
                    np.testing.assert_allclose(
                        getattr(trace, field),
                        ref[field],
                        atol=1e-9,
                        err_msg=f"{name} H{len(values)} {field}",
                    )

    def test_stacked_shapes(self):
        # fixed case: n = 2, m = 1. The (H, 1, 1) `s` proves it is S, not C (1x2).
        model, belief, pref = _model(), _belief(), _obs_preference()
        policy = _policy([0.4, -0.2, 0.1])  # H = 3
        trace = policy_efe_trace(model, belief, policy, pref)
        assert trace.g.shape == (3,)
        assert trace.pragmatic.shape == (3,)
        assert trace.epistemic.shape == (3,)
        assert trace.mu_pred.shape == (3, 2)
        assert trace.sigma_pred.shape == (3, 2, 2)
        assert trace.sigma_post.shape == (3, 2, 2)
        assert trace.s.shape == (3, 1, 1)


class TestTraceH1MatchesEfeStep:
    """At H=1 the traced predict moments equal `_efe_step`'s fields (the field map)."""

    def test_h1_moments_are_bit_identical_to_efe_step(self):
        action = jnp.array([0.4])
        for name, model, belief, pref in _cases():
            trace = policy_efe_trace(model, belief, action[None, :], pref)
            step = _efe_step(
                model,
                belief.mean,
                belief.cov,
                model.control_matrix,
                action,
                pref.goal,
                pref.precision,
            )
            np.testing.assert_array_equal(
                trace.mu_pred[0], step.mu_pred, err_msg=f"mu_pred {name}"
            )
            np.testing.assert_array_equal(
                trace.sigma_pred[0], step.sigma_pred, err_msg=f"sigma_pred {name}"
            )
            np.testing.assert_array_equal(trace.s[0], step.s, err_msg=f"s {name}")

    def test_h1_sigma_post_is_the_kalman_contraction(self):
        # Σ_post = Σ⁺ − Σ⁺Cᵀ S⁻¹ C Σ⁺, symmetrized — from _efe_step's moments.
        model, belief, pref = _model(), _belief(), _obs_preference()
        action = jnp.array([0.4])
        trace = policy_efe_trace(model, belief, action[None, :], pref)
        step = _efe_step(
            model,
            belief.mean,
            belief.cov,
            model.control_matrix,
            action,
            pref.goal,
            pref.precision,
        )
        c = np.asarray(model.observation_matrix)
        sp = np.asarray(step.sigma_pred)
        p_xo = sp @ c.T
        expected = sp - p_xo @ np.linalg.solve(np.asarray(step.s), p_xo.T)
        expected = 0.5 * (expected + expected.T)
        np.testing.assert_allclose(trace.sigma_post[0], expected, atol=1e-12)


class TestTraceTransforms:
    """The trace rollout composes under jit / vmap-over-policies / grad."""

    def test_jit_agrees_with_eager(self):
        model, belief, pref = _model(), _belief(), _obs_preference()
        policy = _policy([0.4, -0.2])
        eager = policy_efe_trace(model, belief, policy, pref)
        jitted = jax.jit(lambda p: policy_efe_trace(model, belief, p, pref))(policy)
        for field in eager._fields:
            np.testing.assert_allclose(
                getattr(jitted, field),
                getattr(eager, field),
                atol=1e-12,
                err_msg=field,
            )

    def test_vmap_over_policies(self):
        model, belief, pref = _model(), _belief(), _obs_preference()
        policies = jnp.stack(
            [_policy([0.4, -0.2]), _policy([-1.0, 0.5]), _policy([0.0, 0.0])]
        )
        batched = jax.vmap(lambda p: policy_efe_trace(model, belief, p, pref))(policies)
        # leading batch axis on every field, ahead of the H axis
        assert batched.g.shape == (3, 2)
        assert batched.sigma_post.shape == (3, 2, 2, 2)
        for i, p in enumerate(policies):
            per = policy_efe_trace(model, belief, p, pref)
            np.testing.assert_allclose(batched.g[i], per.g, atol=1e-12)
            np.testing.assert_allclose(
                batched.sigma_post[i], per.sigma_post, atol=1e-12
            )

    def test_grad_over_policy_runs(self):
        model, belief, pref = _model(), _belief(), _obs_preference()
        grad = jax.grad(lambda p: jnp.sum(policy_efe_trace(model, belief, p, pref).g))(
            _policy([0.4, -0.2])
        )
        assert grad.shape == (2, 1)
        assert bool(jnp.all(jnp.isfinite(grad)))


class TestTraceIsPytree:
    """The NamedTuple is a JAX pytree with seven leaves (survives scan/jit/vmap)."""

    def test_seven_leaves(self):
        model, belief, pref = _model(), _belief(), _obs_preference()
        trace = policy_efe_trace(model, belief, _policy([0.4, -0.2]), pref)
        assert isinstance(trace, PolicyEfeTrace)
        assert len(jax.tree_util.tree_leaves(trace)) == 7
