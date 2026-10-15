"""Tier-1 linear-Gaussian factor nodes (v0.4 Phase 2, DECISIONS.md ADR-012).

Oracle-first: these pin ``GaussianObservation`` and ``GaussianTransition`` before
they exist, so the file is RED until ``cpomdp.ffg.factors.linear_gaussian`` lands.
Each oracle is an independent path — the moment-form measurement update and the
moment-form predict, computed in plain NumPy — never the canonical-form math
under test.

The decomposition these factors implement (one Kalman step = Phase 1 ops):

- ``GaussianObservation(C, R).message(y)`` -> ``CanonicalGaussian`` on x, the
  information form ``(CᵀR⁻¹C, CᵀR⁻¹y)``. The measurement *update* is then
  ``belief + message`` (``CanonicalGaussian.__add__``).
- ``GaussianTransition(A, Q).predict(message, control_term)`` -> the *predict*
  step: build the joint over ``[x, x']``, fold the message into the x block,
  marginalize x out. Moment form: ``AΣAᵀ+Q``, ``Aμ+b``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp.ffg.factors.linear_gaussian import (
    CallableGaussianObservation,
    GaussianCoupling,
    GaussianObservation,
    GaussianTransition,
)
from cpomdp.ffg.message import CanonicalGaussian


def _spd(rng, n):
    """A random n x n symmetric positive-definite matrix (NumPy, independent)."""
    a = rng.standard_normal((n, n))
    return a @ a.T + n * np.eye(n)


# Module-level noise_fns for the callable-observation tests: a closure/lambda is
# hashable only by identity and would defeat jit caching (mirrors CallableSensor's
# contract — all tunables live in params, the function stays static aux).


def _constant_noise(x, params):
    """R(x) = R0 for every x — the reduction back to a fixed observation noise."""
    return params["R0"]


def _scaled_noise(x, params):
    """R(x) = R0·(1 + gain·xᵀx): a PD noise that sharpens/blurs with |x|."""
    scale = 1.0 + params["gain"] * jnp.dot(x, x)
    return params["R0"] * scale


def _belief_as_canonical(mean, cov):
    """Moment-form (mean, cov) -> its canonical message (NumPy, independent)."""
    precision = np.linalg.inv(cov)
    return CanonicalGaussian(precision, precision @ mean)


# --- Observation factor: the measurement update -------------------------------


class TestGaussianObservation:
    def test_observation_noise_is_keyword_only(self):
        # Same pair as FixedSensor, same reason: (m, n) against (m, m) is only
        # distinguishable when m != n, and the factor validates just the noise.
        with pytest.raises(TypeError, match="positional"):
            # Calling it wrongly is the test, so ty's arity errors are expected.
            GaussianObservation(  # ty: ignore[missing-argument]
                jnp.eye(2),
                jnp.eye(2) * 0.1,  # ty: ignore[too-many-positional-arguments]
            )

    def test_stores_coerced_arrays(self):
        fac = GaussianObservation([[1.0, 0.0]], observation_noise=[[2.0]])
        assert isinstance(fac.observation_matrix, jax.Array)
        np.testing.assert_array_equal(fac.observation_matrix, [[1.0, 0.0]])
        np.testing.assert_array_equal(fac.observation_noise, [[2.0]])

    def test_rejects_singular_observation_noise(self):
        # R is inverted in the message, so a singular R is rejected at construction.
        with pytest.raises(ValueError, match="positive-definite"):
            GaussianObservation([[1.0]], observation_noise=[[0.0]])

    def test_rejects_observation_noise_shape_mismatch(self):
        # C is 1xn (m=1) but R is 2x2 — R must be m x m.
        with pytest.raises(ValueError, match="match"):
            GaussianObservation(
                [[1.0, 0.0]], observation_noise=[[1.0, 0.0], [0.0, 1.0]]
            )

    def test_message_is_information_form_of_likelihood(self):
        rng = np.random.default_rng(0)
        n, m = 3, 2
        C = rng.standard_normal((m, n))
        R = _spd(rng, m)
        y = rng.standard_normal(m)
        msg = GaussianObservation(C, observation_noise=R).message(y)
        Rinv = np.linalg.inv(R)
        np.testing.assert_allclose(msg.precision, C.T @ Rinv @ C, atol=1e-10)
        np.testing.assert_allclose(msg.potential, C.T @ Rinv @ y, atol=1e-10)

    @pytest.mark.parametrize(("n", "m"), [(1, 1), (2, 1), (3, 2), (4, 3)])
    def test_update_matches_moment_form_measurement_update(self, n, m):
        # Oracle: the standard Kalman *update* (no prediction), moment form, NumPy.
        rng = np.random.default_rng(n * 10 + m)
        C = rng.standard_normal((m, n))
        R = _spd(rng, m)
        mean = rng.standard_normal(n)
        cov = _spd(rng, n)
        y = rng.standard_normal(m)
        gain = cov @ C.T @ np.linalg.inv(C @ cov @ C.T + R)
        mean_post = mean + gain @ (y - C @ mean)
        cov_post = (np.eye(n) - gain @ C) @ cov

        post = _belief_as_canonical(mean, cov) + GaussianObservation(
            C, observation_noise=R
        ).message(y)
        out_mean, out_cov = post.to_moment()
        np.testing.assert_allclose(out_mean, mean_post, atol=1e-8)
        np.testing.assert_allclose(out_cov, cov_post, atol=1e-8)

    def test_jit_and_grad_through_message(self):
        fac = GaussianObservation(
            [[1.0, 0.5], [0.0, 1.0]], observation_noise=[[1.0, 0.0], [0.0, 1.0]]
        )
        y = jnp.array([1.0, -1.0])
        eager = fac.message(y).potential
        jitted = jax.jit(lambda yy: fac.message(yy).potential)(y)
        np.testing.assert_allclose(jitted, eager, atol=1e-12)
        grad = jax.grad(lambda yy: fac.message(yy).potential.sum())(y)
        assert bool(jnp.all(jnp.isfinite(grad)))


# --- Callable observation factor: state-dependent R(x) (issue #27, Phase 1) ----


class TestCallableGaussianObservation:
    """The FFG twin of ``CallableSensor``: constant C, ``noise_fn(x, params) -> R(x)``.

    Oracle-first, and the *fixed* factor is the oracle: a constant ``noise_fn`` must
    reproduce ``GaussianObservation`` exactly (the reduction), and the state-dependent
    case is pinned against the moment-form Kalman update at the plug-in ``R(x)``. This
    is the primitive that lets the chosen action move the covariance (the dual effect,
    ADR-014 finding #1); without it the FFG epistemic is action-invariant (ADR-003).
    """

    def test_stores_coerced_arrays(self):
        params = {"R0": jnp.array([[2.0]]), "gain": 0.5}
        fac = CallableGaussianObservation([[1.0, 0.0]], _scaled_noise, params)
        assert isinstance(fac.observation_matrix, jax.Array)
        np.testing.assert_array_equal(fac.observation_matrix, [[1.0, 0.0]])
        assert fac.noise_fn is _scaled_noise

    def test_rejects_non_pd_noise_at_probe(self):
        # noise_fn is probed once at construction; a singular R(x) is rejected there —
        # the construction-time analogue of the fixed factor's positive-definite check.
        params = {"R0": jnp.array([[0.0]]), "gain": 0.0}
        with pytest.raises(ValueError, match="positive-definite"):
            CallableGaussianObservation([[1.0]], _constant_noise, params)

    def test_rejects_noise_shape_mismatch(self):
        # C is 1xn (m=1) but noise_fn returns 2x2 — R(x) must be m x m.
        params = {"R0": jnp.eye(2), "gain": 0.0}
        with pytest.raises(ValueError, match="covariance"):
            CallableGaussianObservation([[1.0, 0.0]], _constant_noise, params)

    def test_constant_noise_fn_matches_fixed_observation(self):
        # The reduction: with noise_fn ≡ R0 the message equals the fixed factor's, at
        # ANY plug-in state — the "reduces to fixed" gate the whole feature rides on.
        rng = np.random.default_rng(0)
        n, m = 3, 2
        C = rng.standard_normal((m, n))
        R0 = _spd(rng, m)
        y = rng.standard_normal(m)
        params = {"R0": jnp.asarray(R0), "gain": 0.0}
        callable_fac = CallableGaussianObservation(C, _constant_noise, params)
        fixed = GaussianObservation(C, observation_noise=R0)
        states = (np.zeros(n), rng.standard_normal(n), 3.0 * rng.standard_normal(n))
        for state in states:
            msg = callable_fac.message(y, jnp.asarray(state))
            ref = fixed.message(y)
            np.testing.assert_allclose(msg.precision, ref.precision, atol=1e-10)
            np.testing.assert_allclose(msg.potential, ref.potential, atol=1e-10)

    def test_message_is_information_form_at_plugin_state(self):
        # Λ = CᵀR(x)⁻¹C, h = CᵀR(x)⁻¹y with R evaluated at the given plug-in state.
        rng = np.random.default_rng(1)
        n, m = 2, 2
        C = rng.standard_normal((m, n))
        R0 = _spd(rng, m)
        y = rng.standard_normal(m)
        state = rng.standard_normal(n)
        params = {"R0": jnp.asarray(R0), "gain": 0.3}
        fac = CallableGaussianObservation(C, _scaled_noise, params)
        r_at = R0 * (1.0 + 0.3 * float(state @ state))
        r_inv = np.linalg.inv(r_at)
        msg = fac.message(y, jnp.asarray(state))
        np.testing.assert_allclose(msg.precision, C.T @ r_inv @ C, atol=1e-9)
        np.testing.assert_allclose(msg.potential, C.T @ r_inv @ y, atol=1e-9)

    @pytest.mark.parametrize(("n", "m"), [(1, 1), (2, 1), (3, 2)])
    def test_update_matches_moment_form_measurement_update(self, n, m):
        # Oracle: the Kalman *update* (no prediction) at the plug-in R(x), moment form.
        rng = np.random.default_rng(n * 10 + m)
        C = rng.standard_normal((m, n))
        R0 = _spd(rng, m)
        mean = rng.standard_normal(n)
        cov = _spd(rng, n)
        y = rng.standard_normal(m)
        state = rng.standard_normal(n)  # where R is evaluated (the predicted mean μ⁺)
        r_at = R0 * (1.0 + 0.2 * float(state @ state))
        gain = cov @ C.T @ np.linalg.inv(C @ cov @ C.T + r_at)
        mean_post = mean + gain @ (y - C @ mean)
        cov_post = (np.eye(n) - gain @ C) @ cov

        params = {"R0": jnp.asarray(R0), "gain": 0.2}
        fac = CallableGaussianObservation(C, _scaled_noise, params)
        post = _belief_as_canonical(mean, cov) + fac.message(y, jnp.asarray(state))
        out_mean, out_cov = post.to_moment()
        np.testing.assert_allclose(out_mean, mean_post, atol=1e-8)
        np.testing.assert_allclose(out_cov, cov_post, atol=1e-8)

    def test_message_varies_with_state(self):
        # The dual effect in the primitive: a state-dependent noise_fn gives a
        # *different* message precision at two states — the covariance the action moves
        # (which ADR-003 broke for a fixed sensor).
        C = np.array([[1.0, 0.0]])
        params = {"R0": jnp.array([[1.0]]), "gain": 0.5}
        fac = CallableGaussianObservation(C, _scaled_noise, params)
        near = fac.message(np.array([0.0]), jnp.zeros(2)).precision  # R small ⇒ sharp
        far = fac.message(np.array([0.0]), jnp.array([3.0, 0.0])).precision  # R big
        assert float(near[0, 0]) > float(far[0, 0])

    def test_linearize_returns_C_and_R_at_state(self):
        # The seam the backend/selector reads R(μ⁺) from: linearize(x) -> (C, R(x)).
        C = np.array([[1.0, 0.0], [0.0, 1.0]])
        params = {"R0": jnp.eye(2), "gain": 0.25}
        fac = CallableGaussianObservation(C, _scaled_noise, params)
        state = jnp.array([1.0, -1.0])
        out_C, out_R = fac.linearize(state)
        np.testing.assert_allclose(out_C, C, atol=1e-12)
        np.testing.assert_allclose(out_R, np.eye(2) * (1.0 + 0.25 * 2.0), atol=1e-12)

    def test_pytree_roundtrip_keeps_params_as_leaf(self):
        # noise_params is a traced leaf (so EFE is grad-able w.r.t. it — sensor
        # learning); noise_fn is static aux. Flatten/unflatten must preserve both.
        params = {"R0": jnp.array([[2.0]]), "gain": 0.5}
        fac = CallableGaussianObservation([[1.0, 0.0]], _scaled_noise, params)
        leaves, treedef = jax.tree_util.tree_flatten(fac)
        assert any(np.asarray(leaf).shape == (1, 1) for leaf in leaves)  # R0 is a leaf
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert rebuilt.noise_fn is _scaled_noise
        np.testing.assert_array_equal(
            rebuilt.observation_matrix, fac.observation_matrix
        )

    def test_jit_and_grad_through_message(self):
        params = {"R0": jnp.array([[1.0]]), "gain": 0.5}
        fac = CallableGaussianObservation([[1.0, 0.0]], _scaled_noise, params)
        y = jnp.array([0.7])
        state = jnp.array([1.2, -0.3])
        eager = fac.message(y, state).potential
        jitted = jax.jit(lambda s: fac.message(y, s).potential)(state)
        np.testing.assert_allclose(jitted, eager, atol=1e-12)
        # grad w.r.t. the noise params (the sensor-learning gradient must be finite).
        grad = jax.grad(
            lambda g: (
                CallableGaussianObservation(
                    [[1.0, 0.0]], _scaled_noise, {"R0": params["R0"], "gain": g}
                )
                .message(y, state)
                .potential.sum()
            )
        )(0.5)
        assert bool(jnp.isfinite(grad))


# --- Transition factor: the predict step --------------------------------------


class TestGaussianTransition:
    def test_stores_coerced_arrays(self):
        fac = GaussianTransition([[1.0]], dynamics_noise=[[2.0]])
        assert isinstance(fac.dynamics_matrix, jax.Array)
        np.testing.assert_array_equal(fac.dynamics_matrix, [[1.0]])

    def test_dynamics_noise_is_keyword_only(self):
        # A and Q are both (n, n) at every dimension, and only Q is content-checked, so
        # a transposed pair builds a factor whose predict step is quietly wrong. Unlike
        # the sensor pair this needs no square-shape coincidence to go silent.
        with pytest.raises(TypeError, match="positional"):
            # Calling it wrongly is the test, so ty's arity errors are expected.
            GaussianTransition(  # ty: ignore[missing-argument]
                jnp.eye(2),
                jnp.eye(2) * 0.1,  # ty: ignore[too-many-positional-arguments]
            )

    def test_from_ou_parameters_after_tau_are_keyword_only(self):
        # tau and dt are both positive scalars in the same time unit, and nothing in
        # the body can tell them apart: A = exp(-dt/tau) is finite either way.
        with pytest.raises(TypeError, match="positional"):
            # Calling it wrongly is the test, so ty's arity errors are expected.
            GaussianTransition.from_ou(  # ty: ignore[missing-argument]
                2.0,
                0.5,  # ty: ignore[too-many-positional-arguments]
                0.1,
            )

    def test_rejects_singular_process_noise(self):
        # Q is inverted in the joint, so a singular Q is rejected at construction.
        with pytest.raises(ValueError, match="positive-definite"):
            GaussianTransition([[1.0]], dynamics_noise=[[0.0]])

    def test_rejects_nonsquare_dynamics(self):
        with pytest.raises(ValueError, match="square"):
            GaussianTransition([[1.0, 0.0]], dynamics_noise=[[1.0]])

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_predict_matches_moment_form(self, n):
        # Oracle: moment-form predict, NumPy. cov_pred = AΣAᵀ+Q, mean_pred = Aμ.
        rng = np.random.default_rng(100 + n)
        A = rng.standard_normal((n, n))
        Q = _spd(rng, n)
        mean = rng.standard_normal(n)
        cov = _spd(rng, n)
        cov_pred = A @ cov @ A.T + Q
        mean_pred = A @ mean

        pred = GaussianTransition(A, dynamics_noise=Q).predict(
            _belief_as_canonical(mean, cov)
        )
        out_mean, out_cov = pred.to_moment()
        np.testing.assert_allclose(out_mean, mean_pred, atol=1e-7)
        np.testing.assert_allclose(out_cov, cov_pred, atol=1e-7)

    @pytest.mark.parametrize(
        ("tau", "var", "dt"), [(0.05, 1.0, 0.01), (9.9, 2.0, 0.01), (0.05, 0.5, 0.2)]
    )
    def test_from_ou_exact_discretisation(self, tau, var, dt):
        # Exact OU discretisation (ADR-017): A_i = exp(-dt/τ), Q_i = Σ_stat (1 − A²).
        t = GaussianTransition.from_ou(tau, stationary_var=var, dt=dt)
        a = np.exp(-dt / tau)
        np.testing.assert_allclose(np.asarray(t.dynamics_matrix), [[a]], atol=1e-12)
        np.testing.assert_allclose(
            np.asarray(t.dynamics_noise), [[var * (1.0 - a * a)]], atol=1e-12
        )

    def test_from_ou_preserves_stationary_variance(self):
        # The property oracle: the discrete stationary variance P = Q/(1−A²) must equal
        # the requested Σ_stat at any τ, dt (a wrong Q, e.g. Σ(1−A), fails this).
        for tau, var, dt in [(0.05, 1.7, 0.01), (9.9, 0.4, 0.01), (9.9, 0.4, 2.0)]:
            t = GaussianTransition.from_ou(tau, stationary_var=var, dt=dt)
            a = float(np.asarray(t.dynamics_matrix)[0, 0])
            q = float(np.asarray(t.dynamics_noise)[0, 0])
            np.testing.assert_allclose(q / (1.0 - a * a), var, atol=1e-10)

    def test_from_ou_well_conditioned_across_stiff_timescales(self):
        # Fast (τ=0.05s) and slow (τ=9.9s) on ONE dt both give A∈(0,1) and a PD Q (it
        # constructs) — no blow-up at the stiff slow mode (exact, not Euler; ADR-017).
        for tau in (0.05, 9.9):
            t = GaussianTransition.from_ou(
                tau, stationary_var=1.0, dt=0.01
            )  # builds ⇒ Q is PD
            assert 0.0 < float(np.asarray(t.dynamics_matrix)[0, 0]) < 1.0

    def test_predict_applies_control_term(self):
        # The control shifts the predicted mean by b; the covariance is unchanged.
        rng = np.random.default_rng(7)
        n = 2
        A = rng.standard_normal((n, n))
        Q = _spd(rng, n)
        mean = rng.standard_normal(n)
        cov = _spd(rng, n)
        b = rng.standard_normal(n)
        mean_pred = A @ mean + b
        cov_pred = A @ cov @ A.T + Q

        pred = GaussianTransition(A, dynamics_noise=Q).predict(
            _belief_as_canonical(mean, cov), control_term=b
        )
        out_mean, out_cov = pred.to_moment()
        np.testing.assert_allclose(out_mean, mean_pred, atol=1e-7)
        np.testing.assert_allclose(out_cov, cov_pred, atol=1e-7)

    def test_jit_and_grad_through_predict(self):
        fac = GaussianTransition(
            [[1.0, 0.1], [0.0, 1.0]], dynamics_noise=[[1.0, 0.0], [0.0, 1.0]]
        )
        msg = CanonicalGaussian([[2.0, 0.0], [0.0, 2.0]], [1.0, 0.0])
        eager = fac.predict(msg).potential
        jitted = jax.jit(
            lambda h: fac.predict(CanonicalGaussian(msg.precision, h)).potential
        )(msg.potential)
        np.testing.assert_allclose(jitted, eager, atol=1e-10)
        grad = jax.grad(lambda b: fac.predict(msg, control_term=b).potential.sum())(
            jnp.zeros(2)
        )
        assert bool(jnp.all(jnp.isfinite(grad)))


# --- Coupling factor: the upward (child -> parent) message ---------------------


class TestGaussianCoupling:
    def test_stores_coerced_arrays(self):
        fac = GaussianCoupling([[2.0, -1.0]], coupling_noise=[[0.3]])
        assert isinstance(fac.coupling_matrix, jax.Array)
        np.testing.assert_array_equal(fac.coupling_matrix, [[2.0, -1.0]])
        np.testing.assert_array_equal(fac.coupling_noise, [[0.3]])

    def test_coupling_noise_is_keyword_only(self):
        # A scalar edge's W and Q are both (1, 1), and only Q is content-checked, so a
        # swapped pair builds an edge whose messages are quietly wrong.
        with pytest.raises(TypeError, match="positional"):
            # Calling it wrongly is the test, so ty's arity errors are expected.
            GaussianCoupling(  # ty: ignore[missing-argument]
                [[0.05]],
                [[0.9]],  # ty: ignore[too-many-positional-arguments]
            )

    def test_rejects_singular_coupling_noise(self):
        # Q is inverted in the message, so a singular Q is rejected at construction.
        with pytest.raises(ValueError, match="positive-definite"):
            GaussianCoupling([[1.0]], coupling_noise=[[0.0]])

    def test_rejects_coupling_noise_shape_mismatch(self):
        # W is 1x2 (c=1) but Q is 2x2 — Q must be c x c.
        with pytest.raises(ValueError, match="match"):
            GaussianCoupling([[1.0, 0.0]], coupling_noise=[[1.0, 0.0], [0.0, 1.0]])

    def test_accepts_nonsquare_coupling(self):
        # The defining difference from GaussianTransition: a structural coupling's W
        # maps parent -> child and need NOT be square. The very shape the transition
        # factor rejects ("square") must construct cleanly here.
        fac = GaussianCoupling([[1.0, 0.0]], coupling_noise=[[1.0]])
        assert fac.coupling_matrix.shape == (1, 2)

    @pytest.mark.parametrize(("p", "c"), [(1, 1), (2, 1), (1, 2), (3, 2)])
    def test_message_to_parent_matches_moment_form(self, p, c):
        # Oracle: build the moment-form joint over [parent, child] under the coupling,
        # condition on a direct reading of the child, read back the parent marginal —
        # all in NumPy, never the canonical-form math under test. Covers square and
        # both non-square directions (parent bigger, child bigger).
        rng = np.random.default_rng(200 + 10 * p + c)
        W = rng.standard_normal((c, p))
        Q = _spd(rng, c)
        m0 = rng.standard_normal(p)
        P0 = _spd(rng, p)
        R = _spd(rng, c)
        y = rng.standard_normal(c)

        mean_j = np.concatenate([m0, W @ m0])
        cov_j = np.block([[P0, P0 @ W.T], [W @ P0, W @ P0 @ W.T + Q]])
        H = np.hstack([np.zeros((c, p)), np.eye(c)])  # the reading sees the child block
        gain = cov_j @ H.T @ np.linalg.inv(H @ cov_j @ H.T + R)
        mean_post = mean_j + gain @ (y - H @ mean_j)
        cov_post = (np.eye(p + c) - gain @ H) @ cov_j
        parent_mean, parent_cov = mean_post[:p], cov_post[:p, :p]

        child_msg = GaussianObservation(np.eye(c), observation_noise=R).message(y)
        up = GaussianCoupling(W, coupling_noise=Q).message_to_parent(child_msg)
        out_mean, out_cov = (_belief_as_canonical(m0, P0) + up).to_moment()

        np.testing.assert_allclose(out_mean, parent_mean, atol=1e-8)
        np.testing.assert_allclose(out_cov, parent_cov, atol=1e-8)

    @pytest.mark.parametrize(("p", "c"), [(1, 1), (2, 1), (1, 2), (3, 2)])
    def test_message_to_child_matches_moment_form(self, p, c):
        # Oracle: a parent belief pushed *down* through child = W·parent + noise(Q)
        # lands on N(W μ_p, W Σ_p Wᵀ + Q) — the mirror of predict (fold the parent in,
        # eliminate the parent, emit on the child), done in NumPy moment form. Unlike
        # the upward message, the downward message is a full child belief on its own,
        # so it moment-forms directly. Square and both non-square directions.
        rng = np.random.default_rng(300 + 10 * p + c)
        W = rng.standard_normal((c, p))
        Q = _spd(rng, c)
        mp = rng.standard_normal(p)
        Pp = _spd(rng, p)

        mean_child = W @ mp
        cov_child = W @ Pp @ W.T + Q

        parent_msg = _belief_as_canonical(mp, Pp)
        down = GaussianCoupling(W, coupling_noise=Q).message_to_child(parent_msg)
        out_mean, out_cov = down.to_moment()

        np.testing.assert_allclose(out_mean, mean_child, atol=1e-8)
        np.testing.assert_allclose(out_cov, cov_child, atol=1e-8)

    def test_jit_and_grad_through_message_to_parent(self):
        # non-square W (1x2)
        coupling = GaussianCoupling([[1.5, -0.5]], coupling_noise=[[0.3]])
        m = CanonicalGaussian([[2.0]], [1.0])  # a message on the 1-D child
        eager = coupling.message_to_parent(m).potential
        jitted = jax.jit(
            lambda h: (
                coupling.message_to_parent(CanonicalGaussian(m.precision, h)).potential
            )
        )(m.potential)
        np.testing.assert_allclose(jitted, eager, atol=1e-10)
        grad = jax.grad(
            lambda h: coupling.message_to_parent(
                CanonicalGaussian(m.precision, h)
            ).potential.sum()
        )(m.potential)
        assert bool(jnp.all(jnp.isfinite(grad)))
