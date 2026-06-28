"""FFG chain backend = the Kalman path on a chain (v0.4 Phase 2 keystone, ADR-012).

Gaussian belief propagation on a linear chain *is* the Kalman filter, so this is the
phase's keystone gate: ``ChainBackend`` (the canonical-form message algebra wired into
the ``InferenceBackend`` protocol) must match the trusted ``KalmanBackend`` path on a
chain topology. The gate is *numerical* identity, not literal bit-for-bit: the FFG
runs in information form and inverts/re-inverts where Kalman stays in moment form, so
ADR-012's "byte-identity" wording is read as "tight numerical identity" (atol 1e-7).

Two independent oracles guard the claim:

- ``KalmanBackend`` (per-step mode) — the ADR-012 keystone oracle, the path the FFG
  must reproduce step for step.
- a dead-simple scalar Kalman filter in plain NumPy (``_independent_scalar_filter``) —
  shares no code with either backend, so a transpose/gain/order bug that happened to
  afflict both matrix paths still gets caught.

RED until ``cpomdp.ffg.chain.ChainBackend`` is implemented.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp.backends.base import InferenceBackend
from cpomdp.backends.kalman import KalmanBackend
from cpomdp.dynamics import CallableProcessNoise
from cpomdp.ffg.chain import ChainBackend
from cpomdp.observation import CallableSensor
from cpomdp.types import Belief, LinearGaussianModel

# Scalar chain, matching test_kalman's setup so the two gates speak the same model.
# A/C/Q/R are terse local scalars for the hand-math, NOT the renamed public fields.
A, C, Q, R = 0.9, 1.0, 0.5, 1.0
PRIOR_MEAN, PRIOR_VAR = 0.0, 10.0
OBSERVATIONS = [1.2, 0.8, 1.5, 2.1, 1.9, 2.4, 2.0, 1.7]


def _spd(rng, n):
    """A random n x n symmetric positive-definite matrix (NumPy, independent)."""
    a = rng.standard_normal((n, n))
    return a @ a.T + n * np.eye(n)


def _scalar_model():
    return LinearGaussianModel(
        dynamics=[[A]],
        sensor_model=[[C]],
        dynamics_noise=[[Q]],
        sensor_noise=[[R]],
        prior=Belief(mean=[PRIOR_MEAN], cov=[[PRIOR_VAR]]),
    )


def _random_chain_model(rng, n, m, *, with_control=False):
    """A well-conditioned fixed-matrix chain model of the given dims."""
    control = rng.standard_normal((n, 1)) if with_control else None
    return LinearGaussianModel(
        dynamics=0.5 * rng.standard_normal((n, n)),
        sensor_model=rng.standard_normal((m, n)),
        dynamics_noise=_spd(rng, n),
        sensor_noise=_spd(rng, m),
        prior=Belief(mean=rng.standard_normal(n), cov=_spd(rng, n)),
        control=control,
    )


def _independent_scalar_filter(ys):
    """A dead-simple scalar Kalman filter (no matrix ops), the third-party oracle."""
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


# --- the keystone: ChainBackend vs the Kalman path ----------------------------


class TestChainMatchesKalman:
    def test_single_step_scalar(self):
        model = _scalar_model()
        kalman = KalmanBackend(model).infer_states([1.2], model.prior)
        chain = ChainBackend(model).infer_states([1.2], model.prior)
        np.testing.assert_allclose(chain.mean, kalman.mean, atol=1e-7)
        np.testing.assert_allclose(chain.cov, kalman.cov, atol=1e-7)

    def test_scalar_sequence_matches_independent_filter(self):
        # Third-party oracle: the FFG chain reproduces a from-scratch scalar filter.
        model = _scalar_model()
        backend = ChainBackend(model)
        belief = model.prior
        for y, (m_exp, v_exp) in zip(
            OBSERVATIONS, _independent_scalar_filter(OBSERVATIONS), strict=True
        ):
            belief = backend.infer_states([y], belief)
            np.testing.assert_allclose(belief.mean, [m_exp], atol=1e-7)
            np.testing.assert_allclose(belief.cov, [[v_exp]], atol=1e-7)

    @pytest.mark.parametrize(("n", "m"), [(1, 1), (2, 1), (3, 2), (4, 3)])
    def test_sequence_matches_kalman(self, n, m):
        # The keystone, multi-dim: prior fed forward each step, both backends lockstep.
        rng = np.random.default_rng(n * 10 + m)
        model = _random_chain_model(rng, n, m)
        kalman, chain = KalmanBackend(model), ChainBackend(model)
        k_belief = c_belief = model.prior
        for _ in range(6):
            y = rng.standard_normal(m)
            k_belief = kalman.infer_states(y, k_belief)
            c_belief = chain.infer_states(y, c_belief)
            np.testing.assert_allclose(c_belief.mean, k_belief.mean, atol=1e-7)
            np.testing.assert_allclose(c_belief.cov, k_belief.cov, atol=1e-7)

    @pytest.mark.parametrize(("n", "m"), [(2, 1), (3, 2)])
    def test_sequence_matches_kalman_with_control(self, n, m):
        # The control shift b = control @ action must thread through predict.
        rng = np.random.default_rng(500 + n * 10 + m)
        model = _random_chain_model(rng, n, m, with_control=True)
        kalman, chain = KalmanBackend(model), ChainBackend(model)
        k_belief = c_belief = model.prior
        for _ in range(6):
            y, action = rng.standard_normal(m), rng.standard_normal(1)
            k_belief = kalman.infer_states(y, k_belief, action)
            c_belief = chain.infer_states(y, c_belief, action)
            np.testing.assert_allclose(c_belief.mean, k_belief.mean, atol=1e-7)
            np.testing.assert_allclose(c_belief.cov, k_belief.cov, atol=1e-7)


# --- protocol conformance + the shared trust boundary -------------------------


class TestChainBackendContract:
    def test_satisfies_inference_backend_protocol(self):
        assert isinstance(ChainBackend(_scalar_model()), InferenceBackend)

    def test_does_not_mutate_prior(self):
        model = _scalar_model()
        prior = model.prior
        ChainBackend(model).infer_states([1.2], prior)
        np.testing.assert_array_equal(prior.mean, [PRIOR_MEAN])
        np.testing.assert_array_equal(prior.cov, [[PRIOR_VAR]])

    def test_rejects_wrong_observation_shape(self):
        model = _scalar_model()
        with pytest.raises(ValueError, match="observation"):
            ChainBackend(model).infer_states([1.0, 2.0], model.prior)

    def test_requires_action_when_model_has_control(self):
        rng = np.random.default_rng(0)
        model = _random_chain_model(rng, 2, 1, with_control=True)
        with pytest.raises(ValueError, match="control"):
            ChainBackend(model).infer_states(rng.standard_normal(1), model.prior)


# --- scope: the one durable rejection -----------------------------------------


class TestChainBackendScope:
    def test_rejects_deterministic_transition(self):
        # Q = 0 has no information form (the transition factor inverts Q).
        model = LinearGaussianModel(
            dynamics=[[A]],
            sensor_model=[[C]],
            dynamics_noise=[[0.0]],
            sensor_noise=[[R]],
            prior=Belief(mean=[0.0], cov=[[1.0]]),
        )
        with pytest.raises(ValueError, match="positive-definite"):
            ChainBackend(model)


# --- R(x)/Q(x) parity with KalmanBackend (Phase 2.5, ADR-012 amendment) -------
# ChainBackend linearizes a state-dependent side at μ⁻ exactly like KalmanBackend
# does (ADR-008), so it is gated directly against that already-oracle-proven path
# (test_kalman.py) rather than re-deriving a third independent oracle here.


def _quad_noise(x, params):
    """R(x) = base + scale·position² — always positive, varies with the state."""
    return jnp.array([[params["base"] + params["scale"] * x[0] ** 2]])


_QUAD = {"base": jnp.array(0.2), "scale": jnp.array(0.5)}


def _quad_process(x, params):
    """Q(x) = base·(1 + scale·position²) — PSD, grows with the state."""
    return params["base"] * (1.0 + params["scale"] * x[0] ** 2)


_QPROC = {"base": jnp.array([[0.05]]), "scale": jnp.array(0.4)}


def _callable_sensor_scalar_model(*, control=None):
    return LinearGaussianModel(
        dynamics=[[A]],
        sensor_model=[[C]],
        dynamics_noise=[[Q]],
        sensor_noise=[[R]],
        prior=Belief(mean=[PRIOR_MEAN], cov=[[PRIOR_VAR]]),
        control=control,
        observation=CallableSensor(
            sensor_model=[[C]], noise_fn=_quad_noise, noise_params=_QUAD
        ),
    )


def _callable_process_scalar_model(*, control=None):
    return LinearGaussianModel(
        dynamics=[[A]],
        sensor_model=[[C]],
        dynamics_noise=[[Q]],
        sensor_noise=[[R]],
        prior=Belief(mean=[PRIOR_MEAN], cov=[[PRIOR_VAR]]),
        control=control,
        process_noise=CallableProcessNoise(_quad_process, _QPROC),
    )


class TestChainCallableSensorParity:
    def test_constant_callable_reduces_to_fixed_chain(self):
        # Safety net, green before and after: a CallableSensor whose R ignores x and
        # equals the model's fixed sensor_noise must filter exactly like the
        # fixed-sensor model. Guards the gating and the fixed hot path.
        r0 = [[R]]
        fixed = _scalar_model()
        callable_model = LinearGaussianModel(
            dynamics=[[A]],
            sensor_model=[[C]],
            dynamics_noise=[[Q]],
            sensor_noise=r0,
            prior=Belief(mean=[PRIOR_MEAN], cov=[[PRIOR_VAR]]),
            observation=CallableSensor(
                sensor_model=[[C]],
                noise_fn=lambda x, p: jnp.array(r0),
                noise_params=None,
            ),
        )
        cf_fixed, cf_call = ChainBackend(fixed), ChainBackend(callable_model)
        b_fixed, b_call = fixed.prior, callable_model.prior
        for y in OBSERVATIONS:
            b_fixed = cf_fixed.infer_states([y], b_fixed)
            b_call = cf_call.infer_states([y], b_call)
            np.testing.assert_allclose(b_call.mean, b_fixed.mean, atol=1e-7)
            np.testing.assert_allclose(b_call.cov, b_fixed.cov, atol=1e-7)

    def test_matches_kalman_scalar_sequence(self):
        model = _callable_sensor_scalar_model()
        kalman, chain = KalmanBackend(model), ChainBackend(model)
        k_belief = c_belief = model.prior
        for y in OBSERVATIONS:
            k_belief = kalman.infer_states([y], k_belief)
            c_belief = chain.infer_states([y], c_belief)
            np.testing.assert_allclose(c_belief.mean, k_belief.mean, atol=1e-7)
            np.testing.assert_allclose(c_belief.cov, k_belief.cov, atol=1e-7)

    def test_matches_kalman_2d_state(self):
        # 2-D state, 1-D obs: exercises the gain/cov orientation the scalar case
        # can't, with R varying through the position component.
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
        kalman, chain = KalmanBackend(model), ChainBackend(model)
        k_belief = c_belief = model.prior
        for y in [[1.0], [1.3], [0.7], [1.6]]:
            k_belief = kalman.infer_states(y, k_belief)
            c_belief = chain.infer_states(y, c_belief)
            np.testing.assert_allclose(c_belief.mean, k_belief.mean, atol=1e-7)
            np.testing.assert_allclose(c_belief.cov, k_belief.cov, atol=1e-7)

    def test_evaluates_R_at_predicted_mean_matching_kalman(self):
        # A control input carries μ⁻ far from the prior mean, into a different R
        # regime — guards ChainBackend actually threading control_term into μ⁻
        # before linearizing, not just the no-control case above.
        model = _callable_sensor_scalar_model(control=[[1.0]])
        action = np.array([3.0])  # μ⁻ = 0 + 3 = 3 -> R(3)=4.7 vs R(0)=0.2
        kalman_belief = KalmanBackend(model).infer_states([2.0], model.prior, action)
        chain_belief = ChainBackend(model).infer_states([2.0], model.prior, action)
        np.testing.assert_allclose(chain_belief.mean, kalman_belief.mean, atol=1e-7)
        np.testing.assert_allclose(chain_belief.cov, kalman_belief.cov, atol=1e-7)


class TestChainCallableProcessNoiseParity:
    def test_constant_callable_reduces_to_fixed_chain(self):
        q0 = [[Q]]
        fixed = _scalar_model()
        callable_model = LinearGaussianModel(
            dynamics=[[A]],
            sensor_model=[[C]],
            dynamics_noise=q0,
            sensor_noise=[[R]],
            prior=Belief(mean=[PRIOR_MEAN], cov=[[PRIOR_VAR]]),
            process_noise=CallableProcessNoise(lambda x, p: jnp.array(q0), None),
        )
        cf_fixed, cf_call = ChainBackend(fixed), ChainBackend(callable_model)
        b_fixed, b_call = fixed.prior, callable_model.prior
        for y in OBSERVATIONS:
            b_fixed = cf_fixed.infer_states([y], b_fixed)
            b_call = cf_call.infer_states([y], b_call)
            np.testing.assert_allclose(b_call.mean, b_fixed.mean, atol=1e-7)
            np.testing.assert_allclose(b_call.cov, b_fixed.cov, atol=1e-7)

    def test_matches_kalman_scalar_sequence(self):
        model = _callable_process_scalar_model()
        kalman, chain = KalmanBackend(model), ChainBackend(model)
        k_belief = c_belief = model.prior
        for y in OBSERVATIONS:
            k_belief = kalman.infer_states([y], k_belief)
            c_belief = chain.infer_states([y], c_belief)
            np.testing.assert_allclose(c_belief.mean, k_belief.mean, atol=1e-7)
            np.testing.assert_allclose(c_belief.cov, k_belief.cov, atol=1e-7)

    def test_matches_kalman_2d_state(self):
        # 2x2 Q growing with the position component — exercises A·Σ·Aᵀ + Q(μ⁻).
        q2d = {"base": jnp.eye(2) * 0.05, "scale": jnp.array(0.4)}
        model = LinearGaussianModel(
            dynamics=[[1.0, 1.0], [0.0, 1.0]],
            sensor_model=[[1.0, 0.0]],
            dynamics_noise=[[1e-3, 0.0], [0.0, 1e-3]],
            sensor_noise=[[1.0]],
            prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
            process_noise=CallableProcessNoise(_quad_process, q2d),
        )
        kalman, chain = KalmanBackend(model), ChainBackend(model)
        k_belief = c_belief = model.prior
        for y in [[1.0], [1.3], [0.7], [1.6]]:
            k_belief = kalman.infer_states(y, k_belief)
            c_belief = chain.infer_states(y, c_belief)
            np.testing.assert_allclose(c_belief.mean, k_belief.mean, atol=1e-7)
            np.testing.assert_allclose(c_belief.cov, k_belief.cov, atol=1e-7)

    def test_evaluates_Q_at_predicted_mean_matching_kalman(self):
        model = _callable_process_scalar_model(control=[[1.0]])
        action = np.array([3.0])  # μ⁻ = 0 + 3 = 3 -> Q(3) >> Q(0)
        kalman_belief = KalmanBackend(model).infer_states([2.0], model.prior, action)
        chain_belief = ChainBackend(model).infer_states([2.0], model.prior, action)
        np.testing.assert_allclose(chain_belief.mean, kalman_belief.mean, atol=1e-7)
        np.testing.assert_allclose(chain_belief.cov, kalman_belief.cov, atol=1e-7)


# --- jit / grad / vmap smoke (ADR-012 gates on a new inference entry point) ----


class TestChainBackendTransforms:
    def test_jit_matches_eager(self):
        model = _scalar_model()
        backend = ChainBackend(model)
        y = jnp.array([1.2])
        eager = backend.infer_states(y, model.prior).mean
        jitted = jax.jit(lambda yy: backend.infer_states(yy, model.prior).mean)(y)
        np.testing.assert_allclose(jitted, eager, atol=1e-10)

    def test_grad_is_finite(self):
        model = _scalar_model()
        backend = ChainBackend(model)

        def posterior_mean(y):
            return backend.infer_states(y, model.prior).mean.sum()

        grad = jax.grad(posterior_mean)(jnp.array([1.2]))
        assert bool(jnp.all(jnp.isfinite(grad)))

    def test_vmap_over_observations(self):
        model = _scalar_model()
        backend = ChainBackend(model)
        ys = jnp.array([[0.5], [1.0], [1.5]])
        means = jax.vmap(lambda y: backend.infer_states(y, model.prior).mean)(ys)
        assert means.shape == (3, 1)
        assert bool(jnp.all(jnp.isfinite(means)))
