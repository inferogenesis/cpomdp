"""One-step EFE: independent NumPy oracle, the collapse property, and transforms.

``_numpy_efe`` recomputes G in pure NumPy via a DIFFERENT code path from the JAX
kernel in efe.py — they share no code, so agreement is an independent confirmation
the kernel's algebra is right (same discipline as test_kalman / test_rxinfer).

The locked definition these tests pin (observation-space cross-entropy pragmatic
MINUS state info-gain epistemic, G minimised) is documented in efe.py's module
docstring and DECISIONS.md ADR-005. These tests do NOT prove the *choice* of
pragmatic form is correct — that is rfcs/004's job; the fixed-sensor collapse here
is satisfied by all three candidate forms.
"""

import jax
import jax.numpy as jnp
import numpy as np

from cpomdp.efe import expected_free_energy
from cpomdp.selection import Preference
from cpomdp.types import Belief, LinearGaussianModel


def _model(observation=None):
    # 2-state, 1-observation, 1-action, controllable.
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
    # Observation-space goal: m = 1 here, so goal is (1,), precision (1, 1).
    return Preference(goal=[1.0], precision=[[2.0]])


def _numpy_efe(model, belief, action, goal, precision):
    """Independent NumPy recomputation of (G, pragmatic, epistemic).

    Deliberately mirrors efe.py's math in a separate library (NumPy) and a separate
    code path — no import of the kernel, no shared helpers.
    """
    A = np.asarray(model.dynamics)
    B = np.asarray(model.control)
    Q = np.asarray(model.dynamics_noise)
    C = np.asarray(model.sensor_model)
    R = np.asarray(model.sensor_noise)
    mu = np.asarray(belief.mean)
    sigma = np.asarray(belief.cov)
    a = np.asarray(action, dtype=float)
    g = np.asarray(goal, dtype=float)
    lam = np.asarray(precision, dtype=float)

    mu_pred = A @ mu + B @ a
    sigma_pred = A @ sigma @ A.T + Q
    o_pred = C @ mu_pred
    s = C @ sigma_pred @ C.T + R

    resid = o_pred - g
    pragmatic = 0.5 * resid @ lam @ resid + 0.5 * np.trace(lam @ s)
    epistemic = 0.5 * (np.linalg.slogdet(s)[1] - np.linalg.slogdet(R)[1])
    return pragmatic - epistemic, pragmatic, epistemic


class TestAgainstNumpyOracle:
    def test_matches_numpy_oracle(self):
        model, belief, pref = _model(), _belief(), _obs_preference()
        action = jnp.array([0.4])

        g, parts = expected_free_energy(model, belief, action, pref)
        g_ref, prag_ref, epi_ref = _numpy_efe(
            model, belief, action, pref.goal, pref.precision
        )

        np.testing.assert_allclose(g, g_ref, atol=1e-10)
        np.testing.assert_allclose(parts["pragmatic"], prag_ref, atol=1e-10)
        np.testing.assert_allclose(parts["epistemic"], epi_ref, atol=1e-10)

    def test_components_are_nonnegative(self):
        # epistemic = ½ ln(det S / det R) ≥ 0 (S ⪰ R); pragmatic ≥ 0 (Λ ⪰ 0).
        _, parts = expected_free_energy(
            _model(), _belief(), jnp.array([0.4]), _obs_preference()
        )
        assert parts["epistemic"] >= 0.0
        assert parts["pragmatic"] >= 0.0

    def test_raises_without_control(self):
        control_free = LinearGaussianModel(
            dynamics=[[1.0]],
            sensor_model=[[1.0]],
            dynamics_noise=[[0.1]],
            sensor_noise=[[0.5]],
            prior=Belief(mean=[0.0], cov=[[1.0]]),
        )
        pref = Preference(goal=[0.0], precision=[[1.0]])
        with np.testing.assert_raises(ValueError):
            expected_free_energy(
                control_free, control_free.prior, jnp.array([0.0]), pref
            )


class TestCollapseUnderFixedSensor:
    # Under a fixed sensor, Σ⁺/S/R don't depend on the action, so the epistemic
    # term is constant across actions and G's argmin is driven entirely by the
    # pragmatic term (ADR-003 made executable). NB this collapse holds for ALL
    # three candidate pragmatic forms — it does NOT prove the form choice (rfcs/004).
    def test_epistemic_is_action_invariant(self):
        model, belief, pref = _model(), _belief(), _obs_preference()
        actions = jnp.array([[-1.0], [0.0], [0.5], [2.0]])

        epistemics = jnp.array(
            [
                expected_free_energy(model, belief, a, pref)[1]["epistemic"]
                for a in actions
            ]
        )
        np.testing.assert_allclose(epistemics, epistemics[0], atol=1e-12)

    def test_argmin_g_equals_argmin_pragmatic(self):
        model, belief, pref = _model(), _belief(), _obs_preference()
        actions = jnp.array([[-1.0], [0.0], [0.5], [2.0]])

        gs, prags = [], []
        for a in actions:
            g, parts = expected_free_energy(model, belief, a, pref)
            gs.append(g)
            prags.append(parts["pragmatic"])
        assert int(jnp.argmin(jnp.array(gs))) == int(jnp.argmin(jnp.array(prags)))


class TestTransforms:
    def test_jit_agrees_with_eager(self):
        model, belief, pref = _model(), _belief(), _obs_preference()
        action = jnp.array([0.4])

        eager, _ = expected_free_energy(model, belief, action, pref)
        jitted = jax.jit(lambda a: expected_free_energy(model, belief, a, pref)[0])(
            action
        )
        np.testing.assert_allclose(jitted, eager, atol=1e-12)

    def test_vmap_over_candidate_actions(self):
        model, belief, pref = _model(), _belief(), _obs_preference()
        actions = jnp.array([[-1.0], [0.0], [0.5], [2.0]])

        batched = jax.vmap(lambda a: expected_free_energy(model, belief, a, pref)[0])(
            actions
        )
        per_action = jnp.array(
            [expected_free_energy(model, belief, a, pref)[0] for a in actions]
        )
        np.testing.assert_allclose(batched, per_action, atol=1e-12)

    def test_grad_over_action_runs(self):
        model, belief, pref = _model(), _belief(), _obs_preference()
        grad = jax.grad(lambda a: expected_free_energy(model, belief, a, pref)[0])(
            jnp.array([0.4])
        )
        assert grad.shape == (1,)
        assert bool(jnp.all(jnp.isfinite(grad)))
