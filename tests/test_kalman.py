import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp.backends.kalman import (
    KalmanBackend,
    _gain_and_posterior_cov,
    _posterior_mean,
)
from cpomdp.dynamics import CallableProcessNoise
from cpomdp.observation import CallableSensor
from cpomdp.types import Belief, LinearGaussianModel

# Scalar linear-Gaussian setup, matching the Phase-0 spike.
# A/C/Q/R here are just terse local scalars for the hand-math below, NOT the
# control-theory letters the public API deliberately renames away from
# (dynamics/sensor_model/dynamics_noise/sensor_noise). Inside one short test
# module the letters keep the scalar Kalman recursion readable; they carry no
# API meaning.
A, C, Q, R = 0.9, 1.0, 0.5, 1.0
PRIOR_MEAN, PRIOR_VAR = 0.0, 10.0
OBSERVATIONS = [1.2, 0.8, 1.5, 2.1, 1.9, 2.4, 2.0, 1.7]


def _scalar_model():
    return LinearGaussianModel(
        dynamics=[[A]],
        sensor_model=[[C]],
        dynamics_noise=[[Q]],
        sensor_noise=[[R]],
        prior=Belief(mean=[PRIOR_MEAN], cov=[[PRIOR_VAR]]),
    )


def test_collinear_sensor_with_pd_noise_gives_finite_posterior():
    # Two collinear sensor rows make C·Σ·Cᵀ rank-deficient, but a positive-definite R
    # keeps the innovation S = C·Σ·Cᵀ + R nonsingular, so the gain solve stays finite
    # (no NaN). The R-must-be-positive-definite construction check guarantees this.
    model = LinearGaussianModel(
        dynamics=[[1.0, 0.0], [0.0, 1.0]],
        sensor_model=[[1.0, 0.0], [1.0, 0.0]],  # collinear rows -> rank-1 C·Σ·Cᵀ
        dynamics_noise=[[0.1, 0.0], [0.0, 0.1]],
        sensor_noise=[[0.2, 0.0], [0.0, 0.2]],  # PD
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
    )
    post = KalmanBackend(model).infer_states(jnp.array([3.0, 3.0]), model.prior)
    assert bool(jnp.all(jnp.isfinite(post.mean)))
    assert bool(jnp.all(jnp.isfinite(post.cov)))


def _independent_scalar_filter(ys):
    """A dead-simple scalar Kalman filter (no matrix ops) used as the oracle.

    Because it shares no code with KalmanBackend, a transpose or gain-formula
    bug in the matrix implementation would make the two disagree.
    """
    m, v = PRIOR_MEAN, PRIOR_VAR
    out = []
    for y in ys:
        m_pred = A * m
        v_pred = A * v * A + Q
        s = C * v_pred * C + R
        k = v_pred * C / s
        m = m_pred + k * (y - C * m_pred)
        v = (1 - k * C) * v_pred
        out.append((m, v))
    return out


class TestKalmanBackend:
    def test_single_step_matches_hand_computation(self):
        model = _scalar_model()
        post = KalmanBackend(model).infer_states(np.array([1.2]), model.prior)
        # Derive the expected posterior instead of hardcoding a 16-digit literal
        # nobody can verify by eye: predict (prior mean 0), then one update.
        var_pred = A * PRIOR_VAR * A + Q  # 0.9·10·0.9 + 0.5 = 8.6
        gain = var_pred / (var_pred + R)  # 8.6 / 9.6
        expected_mean = gain * 1.2  # mean_pred is 0
        expected_cov = (1 - gain) * var_pred
        np.testing.assert_allclose(post.mean, [expected_mean], rtol=1e-12)
        np.testing.assert_allclose(post.cov, [[expected_cov]], rtol=1e-12)

    def test_matches_independent_scalar_filter(self):
        model = _scalar_model()
        kf = KalmanBackend(model)
        belief = model.prior
        for y, (mean_exp, var_exp) in zip(
            OBSERVATIONS, _independent_scalar_filter(OBSERVATIONS), strict=True
        ):
            belief = kf.infer_states(np.array([y]), belief)
            np.testing.assert_allclose(belief.mean, [mean_exp], rtol=1e-12)
            np.testing.assert_allclose(belief.cov, [[var_exp]], rtol=1e-12)

    def test_final_step_matches_rxinfer_oracle(self):
        """Anchor the recursion against a *truly external* oracle: RxInfer.jl."""
        rxinfer_final_mean = 1.679270599888
        rxinfer_final_var = 0.467784044120

        model = _scalar_model()
        kf = KalmanBackend(model)
        belief = model.prior
        for y in OBSERVATIONS:
            belief = kf.infer_states(np.array([y]), belief)

        np.testing.assert_allclose(belief.mean, [rxinfer_final_mean], rtol=1e-11)
        np.testing.assert_allclose(belief.cov, [[rxinfer_final_var]], rtol=1e-11)

    def test_filtering_reduces_uncertainty(self):
        model = _scalar_model()
        post = KalmanBackend(model).infer_states(np.array([1.2]), model.prior)
        assert post.cov[0, 0] < model.prior.cov[0, 0]

    def test_covariance_converges_to_steady_state(self):
        # The covariance recursion is data-independent (the ADR-002 front-loading
        # premise): after enough steps it stops changing, which is exactly what
        # lets the steady-state gain be precomputed once.
        model = _scalar_model()
        kf = KalmanBackend(model)
        belief = model.prior
        covs = []
        for _ in range(100):
            belief = kf.infer_states(np.array([0.0]), belief)
            covs.append(belief.cov)
        # Once converged, consecutive covariances are identical to ~machine
        # precision, so atol=1e-12 asserts "stopped changing", not "close".
        np.testing.assert_allclose(covs[-1], covs[-2], atol=1e-12)

    def test_does_not_mutate_prior(self):
        # Beliefs are immutable values — infer_states must return a NEW belief,
        # never mutate the arrays inside the prior it was given.
        model = _scalar_model()
        prior = model.prior
        mean_before, cov_before = prior.mean.copy(), prior.cov.copy()
        KalmanBackend(model).infer_states(np.array([1.2]), prior)
        np.testing.assert_array_equal(prior.mean, mean_before)
        np.testing.assert_array_equal(prior.cov, cov_before)

    def test_multivariate_position_velocity_step(self):
        # 2-D state [position, velocity], observe position only. This is the test
        # scalar (1x1) models CAN'T provide: at 1x1 a transpose is a no-op and
        # there are no off-diagonals, so matrix-orientation and cross-covariance
        # bugs go undetected. Here, observing position must reduce VELOCITY
        # uncertainty through the off-diagonal coupling.
        #
        # dynamics_noise = 0 keeps the arithmetic hand-checkable:
        #   predicted cov = A·I·Aᵀ = [[2, 1], [1, 1]]
        #   gain          = [[2/3], [1/3]]
        #   posterior mean = [2/3, 1/3];  posterior cov = [[2/3, 1/3], [1/3, 2/3]]
        model = LinearGaussianModel(
            dynamics=[[1.0, 1.0], [0.0, 1.0]],  # position advances by velocity
            sensor_model=[[1.0, 0.0]],  # observe position only
            dynamics_noise=[[0.0, 0.0], [0.0, 0.0]],
            sensor_noise=[[1.0]],
            prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        )
        post = KalmanBackend(model).infer_states(np.array([1.0]), model.prior)
        np.testing.assert_allclose(post.mean, [2 / 3, 1 / 3], rtol=1e-12)
        np.testing.assert_allclose(
            post.cov, [[2 / 3, 1 / 3], [1 / 3, 2 / 3]], rtol=1e-12
        )
        # the payoff: position observation sharpened the velocity estimate
        assert post.cov[1, 1] < model.prior.cov[1, 1]


def _control_model():
    return LinearGaussianModel(
        dynamics=[[1.0]],
        sensor_model=[[1.0]],
        dynamics_noise=[[0.5]],
        sensor_noise=[[1.0]],
        prior=Belief(mean=[0.0], cov=[[10.0]]),
        control=[[1.0]],
    )


class TestKalmanControl:
    def test_requires_action_when_model_has_control(self):
        model = _control_model()
        with pytest.raises(ValueError, match="action"):
            KalmanBackend(model).infer_states(np.array([1.0]), model.prior)

    def test_action_shifts_prediction(self):
        model = _control_model()
        kf = KalmanBackend(model)
        unpushed = kf.infer_states(np.array([1.0]), model.prior, action=np.array([0.0]))
        pushed = kf.infer_states(np.array([1.0]), model.prior, action=np.array([5.0]))
        # A +5 action enters the predicted mean, then the update pulls partly
        # back toward the (unchanged) observation. Direction must be UP (a sign
        # error in the control term would flip this and != would never notice).
        assert pushed.mean[0] > unpushed.mean[0]
        # And the surviving shift is exactly the gain-attenuated action:
        # (1 - gain) * 5, with gain from the predicted variance.
        var_pred = 1.0 * 10.0 * 1.0 + 0.5  # dynamics·var·dynamics + dyn_noise
        gain = var_pred / (var_pred + 1.0)  # sensor_noise = 1.0
        expected_shift = (1 - gain) * 5.0
        np.testing.assert_allclose(
            pushed.mean[0] - unpushed.mean[0], expected_shift, rtol=1e-12
        )


def _full_filter_converged_belief(model, steps=200):
    """Run the full per-step filter long enough to reach steady state."""
    kf = KalmanBackend(model)
    belief = model.prior
    for _ in range(steps):
        belief = kf.infer_states(np.array([0.0]), belief)
    return belief


class TestKalmanSteadyState:
    def test_frozen_cov_matches_converged_full_filter(self):
        # The whole point of front-loading: the covariance precomputed at
        # construction must equal what the full per-step filter converges to.
        # This is the test that catches a broken convergence loop (e.g. one that
        # returns after a single iteration) — it would freeze the wrong cov.
        model = _scalar_model()
        converged = _full_filter_converged_belief(model).cov
        steady = KalmanBackend(model, steady_state=True)
        frozen = steady.infer_states(np.array([0.0]), model.prior).cov
        np.testing.assert_allclose(frozen, converged, atol=1e-10)

    def test_steady_state_cov_is_constant(self):
        # In steady-state mode the returned covariance is frozen: the same every
        # step, regardless of the incoming belief.
        model = _scalar_model()
        kf = KalmanBackend(model, steady_state=True)
        first = kf.infer_states(np.array([1.0]), model.prior)
        second = kf.infer_states(np.array([5.0]), first)
        np.testing.assert_array_equal(first.cov, second.cov)

    def test_mean_matches_full_filter_once_converged(self):
        # Once the full filter has converged its gain equals K∞, so from the same
        # belief both modes produce the same mean update.
        model = _scalar_model()
        converged = _full_filter_converged_belief(model)
        full = KalmanBackend(model)
        steady = KalmanBackend(model, steady_state=True)
        y = np.array([1.3])
        np.testing.assert_allclose(
            steady.infer_states(y, converged).mean,
            full.infer_states(y, converged).mean,
            atol=1e-10,
        )

    def test_raises_when_not_converged(self):
        # max_iter too small to reach the fixed point -> a loud failure, not a
        # silently-wrong frozen gain.
        with pytest.raises(RuntimeError, match="converge"):
            KalmanBackend(_scalar_model(), steady_state=True, max_iter=1)


class TestJitReady:
    def test_kernels_jitted_step_matches_eager_backend(self):
        # The whole step rebuilt from the jit-compiled kernels must reproduce the
        # eager backend exactly — proving the hot path is pure and traceable.
        model = _scalar_model()
        eager = KalmanBackend(model).infer_states(jnp.array([1.2]), model.prior)

        @jax.jit
        def step(observation, prior_mean, prior_cov):
            gain, cov_post = _gain_and_posterior_cov(
                model.dynamics,
                model.sensor_model,
                model.dynamics_noise,
                model.sensor_noise,
                prior_cov,
            )
            mean_post = _posterior_mean(
                model.dynamics,
                model.sensor_model,
                prior_mean,
                jnp.zeros(model.n_states),
                gain,
                observation,
            )
            return mean_post, cov_post

        mean_post, cov_post = step(jnp.array([1.2]), model.prior.mean, model.prior.cov)
        np.testing.assert_allclose(mean_post, eager.mean, rtol=1e-12)
        np.testing.assert_allclose(cov_post, eager.cov, rtol=1e-12)

    def test_gain_kernel_vmaps_over_a_batch_of_covariances(self):
        # vmap is the payoff the migration buys: one filter, many beliefs at once.
        model = _scalar_model()
        covs = jnp.array([[[1.0]], [[5.0]], [[10.0]]])
        gains, cov_posts = jax.vmap(
            lambda c: _gain_and_posterior_cov(
                model.dynamics,
                model.sensor_model,
                model.dynamics_noise,
                model.sensor_noise,
                c,
            )
        )(covs)
        assert gains.shape == (3, 1, 1)
        assert cov_posts.shape == (3, 1, 1)


# --- state-dependent sensor noise R(x) in the filter (Part A) ---------------
# A CallableSensor carries R(x); the filter must evaluate it at the PREDICTED
# mean μ⁻ (matching the EFE kernel's linearization point), keeping the fixed
# path byte-identical. Noise functions are module-level (jit-safe, hashable by
# identity), like the corridor fixtures.


def _const_noise(x, params):
    """R(x) that ignores x — a constant. For the callable==fixed reduction check."""
    return params["R"]


def _quad_noise(x, params):
    """R(x) = base + scale·position² — always positive, varies with the state."""
    return jnp.array([[params["base"] + params["scale"] * x[0] ** 2]])


_QUAD = {"base": jnp.array(0.2), "scale": jnp.array(0.5)}


def _numpy_rx_filter(
    model, observations, noise_fn, params, actions=None, r_point="pred"
):
    """Independent NumPy R(x)-plug-in Kalman filter; shares no code with the backend.

    R is evaluated at the predicted mean μ⁻ (``r_point="pred"``) — the point the
    backend must use — or at the incoming (prior) mean (``r_point="prior"``), the
    wrong point, so a test can assert the backend picked the right one.
    """
    a_mat = np.asarray(model.dynamics)
    c_mat = np.asarray(model.sensor_model)
    q_mat = np.asarray(model.dynamics_noise)
    b_mat = None if model.control is None else np.asarray(model.control)
    mean = np.asarray(model.prior.mean, dtype=float)
    cov = np.asarray(model.prior.cov, dtype=float)
    n = mean.shape[0]
    out = []
    for t, y in enumerate(observations):
        if b_mat is None:
            mean_pred = a_mat @ mean
        else:
            assert actions is not None  # a model with control must be given actions
            mean_pred = a_mat @ mean + b_mat @ np.asarray(actions[t], dtype=float)
        r_at = mean_pred if r_point == "pred" else mean
        r_mat = np.asarray(noise_fn(jnp.asarray(r_at), params), dtype=float)
        cov_pred = a_mat @ cov @ a_mat.T + q_mat
        s = c_mat @ cov_pred @ c_mat.T + r_mat
        gain = cov_pred @ c_mat.T @ np.linalg.inv(s)
        mean = mean_pred + gain @ (np.asarray(y, dtype=float) - c_mat @ mean_pred)
        cov = (np.eye(n) - gain @ c_mat) @ cov_pred
        out.append((mean.copy(), cov.copy()))
    return out


def _callable_scalar_model(noise_fn, params, *, sensor_noise=None, control=None):
    return LinearGaussianModel(
        dynamics=[[0.9]],
        sensor_model=[[1.0]],
        dynamics_noise=[[0.5]],
        sensor_noise=[[1.0]] if sensor_noise is None else sensor_noise,
        prior=Belief(mean=[0.0], cov=[[10.0]]),
        control=control,
        observation=CallableSensor(
            sensor_model=[[1.0]], noise_fn=noise_fn, noise_params=params
        ),
    )


class TestKalmanCallableSensor:
    def test_constant_callable_reduces_to_fixed_filter(self):
        # Safety net — green BEFORE and AFTER the change. A CallableSensor whose R
        # ignores x and equals the model's fixed sensor_noise must filter exactly
        # like the fixed-sensor model. Guards the gating and the fixed hot path.
        r0 = [[1.0]]
        fixed = LinearGaussianModel(
            dynamics=[[0.9]],
            sensor_model=[[1.0]],
            dynamics_noise=[[0.5]],
            sensor_noise=r0,
            prior=Belief(mean=[0.0], cov=[[10.0]]),
        )
        callable_model = _callable_scalar_model(
            _const_noise, {"R": jnp.array(r0)}, sensor_noise=r0
        )
        kf_fixed, kf_call = KalmanBackend(fixed), KalmanBackend(callable_model)
        b_fixed, b_call = fixed.prior, callable_model.prior
        for y in OBSERVATIONS:
            b_fixed = kf_fixed.infer_states(np.array([y]), b_fixed)
            b_call = kf_call.infer_states(np.array([y]), b_call)
            np.testing.assert_array_equal(b_call.mean, b_fixed.mean)
            np.testing.assert_array_equal(b_call.cov, b_fixed.cov)

    def test_matches_numpy_rx_oracle_scalar(self):
        # RED until Part A: the current filter uses the fixed sensor_noise (1.0),
        # not R(μ⁻). The independent oracle evaluates R at the predicted mean.
        model = _callable_scalar_model(_quad_noise, _QUAD)
        kf = KalmanBackend(model)
        belief = model.prior
        oracle = _numpy_rx_filter(
            model, [[y] for y in OBSERVATIONS], _quad_noise, _QUAD
        )
        for y, (mean_exp, cov_exp) in zip(OBSERVATIONS, oracle, strict=True):
            belief = kf.infer_states(np.array([y]), belief)
            np.testing.assert_allclose(belief.mean, mean_exp, rtol=1e-10)
            np.testing.assert_allclose(belief.cov, cov_exp, rtol=1e-10)

    def test_matches_numpy_rx_oracle_2d_state(self):
        # 2-D state, 1-D obs: exercises the gain/cov orientation (K·C is 2x2) that
        # the scalar case can't, with R varying through the position component.
        model = LinearGaussianModel(
            dynamics=[[1.0, 1.0], [0.0, 1.0]],
            sensor_model=[[1.0, 0.0]],
            dynamics_noise=[[1e-3, 0.0], [0.0, 1e-3]],
            sensor_noise=[[1.0]],
            prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
            observation=CallableSensor(
                sensor_model=[[1.0, 0.0]], noise_fn=_quad_noise, noise_params=_QUAD
            ),
        )
        kf = KalmanBackend(model)
        belief = model.prior
        obs = [[1.0], [1.3], [0.7], [1.6]]
        oracle = _numpy_rx_filter(model, obs, _quad_noise, _QUAD)
        for y, (mean_exp, cov_exp) in zip(obs, oracle, strict=True):
            belief = kf.infer_states(np.array(y), belief)
            np.testing.assert_allclose(belief.mean, mean_exp, rtol=1e-10)
            np.testing.assert_allclose(belief.cov, cov_exp, rtol=1e-10)

    def test_evaluates_R_at_predicted_mean_not_prior(self):
        # The discriminator (perception-side mirror of test_detours_toward_the_beacon):
        # a control input carries μ⁻ far from the prior mean into a different R
        # regime. The filter must use R(μ⁻), not R(prior.mean).
        model = _callable_scalar_model(_quad_noise, _QUAD, control=[[1.0]])
        action = np.array([3.0])  # μ⁻ = 0 + 3 = 3 -> R(3)=4.7 vs R(0)=0.2
        belief = KalmanBackend(model).infer_states(np.array([2.0]), model.prior, action)
        at_pred = _numpy_rx_filter(
            model, [[2.0]], _quad_noise, _QUAD, actions=[action], r_point="pred"
        )[0]
        at_prior = _numpy_rx_filter(
            model, [[2.0]], _quad_noise, _QUAD, actions=[action], r_point="prior"
        )[0]
        np.testing.assert_allclose(belief.cov, at_pred[1], rtol=1e-10)  # R(μ⁻)
        assert not np.allclose(belief.cov, at_prior[1])  # ... NOT R(prior.mean)

    def test_steady_state_with_callable_sensor_raises(self):
        # R(x) has no state-independent Riccati fixed point, so the steady-state
        # gain can't be precomputed — refuse it loudly rather than freeze a wrong gain.
        model = _callable_scalar_model(_quad_noise, _QUAD)
        with pytest.raises(ValueError, match="steady"):
            KalmanBackend(model, steady_state=True)


# --- state-dependent process noise Q(x) in the filter (the R(x) dual) --------
# CallableProcessNoise carries Q(x); the filter's covariance predict must evaluate
# it at the predicted mean μ⁻ (the same point the EFE kernel uses), gated so the
# fixed path stays byte-identical.


def _const_process(x, params):
    """Q(x) that ignores x — a constant. For the callable==fixed reduction check."""
    return params["Q"]


def _quad_process(x, params):
    """Q(x) = base·(1 + scale·position²) — PSD, grows with the state."""
    return params["base"] * (1.0 + params["scale"] * x[0] ** 2)


_QPROC = {"base": jnp.array([[0.05]]), "scale": jnp.array(0.4)}


def _numpy_qx_filter(model, observations, q_fn, q_params, actions=None, q_point="pred"):
    """Independent NumPy Q(x)-plug-in Kalman filter (fixed R); shares no backend code.

    Q is evaluated at the predicted mean μ⁻ (``q_point="pred"``) — the point the
    backend must use — or at the incoming mean (``q_point="prior"``), the wrong one.
    """
    a_mat = np.asarray(model.dynamics)
    c_mat = np.asarray(model.sensor_model)
    r_mat = np.asarray(model.sensor_noise)
    b_mat = None if model.control is None else np.asarray(model.control)
    mean = np.asarray(model.prior.mean, dtype=float)
    cov = np.asarray(model.prior.cov, dtype=float)
    n = mean.shape[0]
    out = []
    for t, y in enumerate(observations):
        if b_mat is None:
            mean_pred = a_mat @ mean
        else:
            assert actions is not None  # a model with control must be given actions
            mean_pred = a_mat @ mean + b_mat @ np.asarray(actions[t], dtype=float)
        q_at = mean_pred if q_point == "pred" else mean
        q_mat = np.asarray(q_fn(jnp.asarray(q_at), q_params), dtype=float)
        cov_pred = a_mat @ cov @ a_mat.T + q_mat
        s = c_mat @ cov_pred @ c_mat.T + r_mat
        gain = cov_pred @ c_mat.T @ np.linalg.inv(s)
        mean = mean_pred + gain @ (np.asarray(y, dtype=float) - c_mat @ mean_pred)
        cov = (np.eye(n) - gain @ c_mat) @ cov_pred
        out.append((mean.copy(), cov.copy()))
    return out


def _callable_q_scalar_model(q_fn, q_params, *, dynamics_noise=None, control=None):
    return LinearGaussianModel(
        dynamics=[[0.9]],
        sensor_model=[[1.0]],
        dynamics_noise=[[0.5]] if dynamics_noise is None else dynamics_noise,
        sensor_noise=[[1.0]],
        prior=Belief(mean=[0.0], cov=[[10.0]]),
        control=control,
        process_noise=CallableProcessNoise(q_fn, q_params),
    )


class TestKalmanCallableProcessNoise:
    def test_constant_callable_process_reduces_to_fixed_filter(self):
        # Safety net — green BEFORE and AFTER. A CallableProcessNoise whose Q ignores
        # x and equals the model's fixed dynamics_noise must filter exactly like the
        # fixed model. Guards the gating and the fixed hot path.
        q0 = [[0.5]]
        fixed = LinearGaussianModel(
            dynamics=[[0.9]],
            sensor_model=[[1.0]],
            dynamics_noise=q0,
            sensor_noise=[[1.0]],
            prior=Belief(mean=[0.0], cov=[[10.0]]),
        )
        callable_model = _callable_q_scalar_model(
            _const_process, {"Q": jnp.array(q0)}, dynamics_noise=q0
        )
        kf_fixed, kf_call = KalmanBackend(fixed), KalmanBackend(callable_model)
        b_fixed, b_call = fixed.prior, callable_model.prior
        for y in OBSERVATIONS:
            b_fixed = kf_fixed.infer_states(np.array([y]), b_fixed)
            b_call = kf_call.infer_states(np.array([y]), b_call)
            np.testing.assert_array_equal(b_call.mean, b_fixed.mean)
            np.testing.assert_array_equal(b_call.cov, b_fixed.cov)

    def test_matches_numpy_qx_oracle_scalar(self):
        # RED until the Q(x) fix: the current predict uses the fixed dynamics_noise
        # (0.5), not Q(μ⁻). The oracle evaluates Q at the predicted mean.
        model = _callable_q_scalar_model(_quad_process, _QPROC)
        kf = KalmanBackend(model)
        belief = model.prior
        oracle = _numpy_qx_filter(
            model, [[y] for y in OBSERVATIONS], _quad_process, _QPROC
        )
        for y, (mean_exp, cov_exp) in zip(OBSERVATIONS, oracle, strict=True):
            belief = kf.infer_states(np.array([y]), belief)
            np.testing.assert_allclose(belief.mean, mean_exp, rtol=1e-10)
            np.testing.assert_allclose(belief.cov, cov_exp, rtol=1e-10)

    def test_matches_numpy_qx_oracle_2d_state(self):
        # 2-D state, 1-D obs: exercises the A·Σ·Aᵀ + Q(μ⁻) predict where Q is a full
        # 2x2 growing with the position component.
        q2d = {"base": jnp.eye(2) * 0.05, "scale": jnp.array(0.4)}
        model = LinearGaussianModel(
            dynamics=[[1.0, 1.0], [0.0, 1.0]],
            sensor_model=[[1.0, 0.0]],
            dynamics_noise=[[1e-3, 0.0], [0.0, 1e-3]],
            sensor_noise=[[1.0]],
            prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
            process_noise=CallableProcessNoise(_quad_process, q2d),
        )
        kf = KalmanBackend(model)
        belief = model.prior
        obs = [[1.0], [1.3], [0.7], [1.6]]
        oracle = _numpy_qx_filter(model, obs, _quad_process, q2d)
        for y, (mean_exp, cov_exp) in zip(obs, oracle, strict=True):
            belief = kf.infer_states(np.array(y), belief)
            np.testing.assert_allclose(belief.mean, mean_exp, rtol=1e-10)
            np.testing.assert_allclose(belief.cov, cov_exp, rtol=1e-10)

    def test_evaluates_Q_at_predicted_mean_not_prior(self):
        # Discriminator: a control input carries μ⁻ far from the prior mean into a
        # different Q regime. The predict must use Q(μ⁻), not Q(prior.mean).
        model = _callable_q_scalar_model(_quad_process, _QPROC, control=[[1.0]])
        action = np.array([3.0])  # μ⁻ = 0 + 3 = 3 -> Q(3) >> Q(0)
        belief = KalmanBackend(model).infer_states(np.array([2.0]), model.prior, action)
        at_pred = _numpy_qx_filter(
            model, [[2.0]], _quad_process, _QPROC, actions=[action], q_point="pred"
        )[0]
        at_prior = _numpy_qx_filter(
            model, [[2.0]], _quad_process, _QPROC, actions=[action], q_point="prior"
        )[0]
        np.testing.assert_allclose(belief.cov, at_pred[1], rtol=1e-10)  # Q(μ⁻)
        assert not np.allclose(belief.cov, at_prior[1])  # ... NOT Q(prior.mean)

    def test_steady_state_with_callable_process_noise_raises(self):
        # Q(x) breaks the constant-recursion fixed point just like R(x) does.
        model = _callable_q_scalar_model(_quad_process, _QPROC)
        with pytest.raises(ValueError, match="steady"):
            KalmanBackend(model, steady_state=True)
