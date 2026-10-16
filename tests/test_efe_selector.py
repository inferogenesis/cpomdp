"""EFESelector: greedy (H=1) EFE action selection over a front-loaded candidate grid.

The oracle is an independent NumPy brute force over a STRICTLY FINER, independently
built action grid — so the test exercises whether the selector's grid is dense
enough, not a grid compared against itself.

The collapse test here is the corrected one: under a fixed sensor the selector picks
the one-step-PRAGMATIC argmin (epistemic is action-invariant), which is NOT the
infinite-horizon LQR action. H=1 greedy EFE ≠ LQR — asserted explicitly so nobody
"fixes" it into the wrong equality.
"""

import jax
import jax.numpy as jnp
import numpy as np

from cpomdp.control import LQRController
from cpomdp.observation import CallableSensor
from cpomdp.selection import ActionSelector, EFESelector, Preference
from cpomdp.types import Belief, LinearGaussianModel


def _well_noise(x, params):
    """R(x): sharp (low) near the beacon, foggy (high) away. Module-level (jit-safe)."""
    pos = x[0]
    falloff = 1.0 - jnp.exp(
        -((pos - params["beacon"]) ** 2) / (2.0 * params["width"] ** 2)
    )
    return jnp.array([[params["r_lo"] + (params["r_hi"] - params["r_lo"]) * falloff]])


def _corridor_model(*, fixed=False):
    # 1-D single integrator: μ⁺ = μ + a, observe position. Beacon (sharp sensor) at 1.5.
    sensor = (
        None
        if fixed
        else CallableSensor(
            observation_matrix=[[1.0]],
            noise_fn=_well_noise,
            noise_params={
                "beacon": jnp.array(1.5),
                "width": jnp.array(0.6),
                "r_lo": jnp.array(0.05),
                "r_hi": jnp.array(0.8),
            },
        )
    )
    return LinearGaussianModel(
        dynamics_matrix=[[1.0]],
        observation_matrix=[[1.0]],
        dynamics_noise=[[0.05]],
        observation_noise=[[0.3]],  # fixed-sensor fallback / fixed-model noise
        prior=Belief(mean=[0.0], cov=[[0.5]]),
        control_matrix=[[1.0]],
        observation_model=sensor,
    )


_PREF = Preference(target=[0.0], precision=[[0.4]])  # obs-space: prefer to observe 0


def _numpy_g(model, belief, action, goal, precision):
    """Independent NumPy (G, pragmatic) — reads R(μ⁺) via the sensor, no kernel code."""
    a_mat, b_mat = np.asarray(model.dynamics_matrix), np.asarray(model.control_matrix)
    q_mat = np.asarray(model.dynamics_noise)
    mu, sigma = np.asarray(belief.mean), np.asarray(belief.cov)
    a = np.asarray(action, dtype=float)
    mu_pred = a_mat @ mu + b_mat @ a
    sigma_pred = a_mat @ sigma @ a_mat.T + q_mat
    if model.observation_model is None:
        c_mat, r_mat = (
            np.asarray(model.observation_matrix),
            np.asarray(model.observation_noise),
        )
    else:
        c_jax, r_jax = model.observation_model.linearize(mu_pred)
        c_mat, r_mat = np.asarray(c_jax), np.asarray(r_jax)
    o = c_mat @ mu_pred
    s = c_mat @ sigma_pred @ c_mat.T + r_mat
    g, lam = np.asarray(goal, dtype=float), np.asarray(precision, dtype=float)
    resid = o - g
    pragmatic = 0.5 * resid @ lam @ resid + 0.5 * np.trace(lam @ s)
    epistemic = 0.5 * (np.linalg.slogdet(s)[1] - np.linalg.slogdet(r_mat)[1])
    return pragmatic - epistemic, pragmatic


def _brute_force(model, belief, preference, lo, hi, n_points, key="g"):
    """argmin of the NumPy G (or pragmatic) over a fine independent 1-D grid."""
    grid = np.linspace(lo, hi, n_points)
    scores = [
        _numpy_g(
            model, belief, np.array([action]), preference.target, preference.precision
        )
        for action in grid
    ]
    part = 0 if key == "g" else 1
    return grid[int(np.argmin([score[part] for score in scores]))]


class TestEFESelector:
    def test_satisfies_action_selector_protocol(self):
        sel = EFESelector(_corridor_model(), n_candidates=21, action_bounds=(-3.0, 3.0))
        assert isinstance(sel, ActionSelector)

    def test_control_free_model_raises(self):
        control_free = LinearGaussianModel(
            dynamics_matrix=[[1.0]],
            observation_matrix=[[1.0]],
            dynamics_noise=[[0.1]],
            observation_noise=[[0.3]],
            prior=Belief(mean=[0.0], cov=[[1.0]]),
        )
        try:
            EFESelector(control_free, n_candidates=21, action_bounds=(-3.0, 3.0))
        except ValueError:
            return
        raise AssertionError("EFESelector should reject a control-free model")

    def test_n_candidates_is_the_per_cycle_eval_count(self):
        sel = EFESelector(_corridor_model(), n_candidates=21, action_bounds=(-3.0, 3.0))
        assert sel.n_candidates == 21  # the attributable per-cycle cost

    def test_select_matches_brute_force_over_a_finer_grid(self):
        model = _corridor_model()
        belief = Belief(mean=[1.0], cov=[[0.5]])
        sel = EFESelector(model, n_candidates=41, action_bounds=(-3.0, 3.0))
        chosen = float(sel.select(belief, _PREF)[0])
        fine = _brute_force(model, belief, _PREF, -3.0, 3.0, 4001, key="g")
        spacing = 6.0 / 40  # the selector's grid spacing
        assert abs(chosen - fine) <= spacing  # dense enough to land on the optimum

    def test_jit_select_agrees_with_eager(self):
        model = _corridor_model()
        belief = Belief(mean=[1.0], cov=[[0.5]])
        sel = EFESelector(model, n_candidates=41, action_bounds=(-3.0, 3.0))
        eager = sel.select(belief, _PREF)
        jitted = jax.jit(sel.select)(belief, _PREF)
        np.testing.assert_allclose(jitted, eager, atol=1e-12)

    def test_under_fixed_sensor_picks_one_step_pragmatic_not_lqr(self):
        # Epistemic is action-invariant under a fixed sensor, so the H=1 selector
        # picks the one-step-PRAGMATIC argmin. That is NOT the infinite-horizon LQR
        # action — assert both: equals the pragmatic argmin, and differs from LQR.
        model = _corridor_model(fixed=True)
        belief = Belief(mean=[1.0], cov=[[0.5]])  # off the goal so the actions differ
        sel = EFESelector(model, n_candidates=41, action_bounds=(-3.0, 3.0))
        chosen = float(sel.select(belief, _PREF)[0])

        prag = _brute_force(model, belief, _PREF, -3.0, 3.0, 4001, key="pragmatic")
        assert abs(chosen - prag) <= 6.0 / 40  # == one-step pragmatic argmin

        # The one-step pragmatic argmin is DEADBEAT (drive the residual to 0 in one
        # step → a ≈ -1.0). Infinite-horizon LQR is gradual — it spreads the
        # correction over the horizon. They only coincide as effort → 0 (LQR
        # → deadbeat); a *balanced* penalty puts LQR in its characteristic gradual
        # regime, which is where H=1 EFE ≠ H=∞ LQR is visible. Do not lower the
        # penalty back toward 0 — that collapses the very distinction being asserted.
        lqr = float(
            LQRController(model, precision=[[1.0]], effort=[[1.0]]).action(
                belief.mean, jnp.array([0.0])
            )[0]
        )
        assert abs(chosen - lqr) > 0.2  # ... and NOT the LQR action

    def test_detours_toward_the_beacon(self):
        # At the goal observation (μ=0), pragmatic alone says "don't move" (a≈0); the
        # epistemic term pulls the chosen action toward the sharp-sensing beacon (+1.5).
        model = _corridor_model()
        belief = Belief(mean=[0.0], cov=[[0.5]])
        sel = EFESelector(model, n_candidates=61, action_bounds=(-3.0, 3.0))
        efe_action = float(sel.select(belief, _PREF)[0])
        prag = _brute_force(model, belief, _PREF, -3.0, 3.0, 4001, key="pragmatic")
        assert efe_action > prag + 0.1  # detours toward the beacon


# Each select used to re-trace the scoring under an eager vmap, and at horizon > 1 the
# rollout's scan recompiled every call, about 180 ms a step whatever the grid. The
# scorer is now compiled once, on the first select, and a selector traces its kernel
# once however many beliefs it sees.


def _counting(monkeypatch, name):
    """Patch ``cpomdp.selection.<name>`` to count the times it is traced."""
    import cpomdp.selection as selection

    traced = []
    original = getattr(selection, name)

    def counted(*args, **kwargs):
        traced.append(name)
        return original(*args, **kwargs)

    monkeypatch.setattr(selection, name, counted)
    return traced


_BELIEFS = [Belief(mean=[position], cov=[[0.5]]) for position in (0.0, 0.4, -0.7)]


class TestCompiledOnce:
    def test_a_rollout_horizon_traces_its_rollout_once(self, monkeypatch):
        traced = _counting(monkeypatch, "policy_efe")
        selector = EFESelector(_corridor_model(), action_bounds=(-3.0, 3.0), horizon=3)
        for belief in _BELIEFS:
            selector.select(belief, _PREF)
        assert len(traced) == 1

    def test_one_step_traces_its_kernel_once(self, monkeypatch):
        traced = _counting(monkeypatch, "expected_free_energy")
        selector = EFESelector(_corridor_model(), action_bounds=(-3.0, 3.0))
        for belief in _BELIEFS:
            selector.select(belief, _PREF)
        assert len(traced) == 1

    def test_the_choice_is_the_argmin_of_the_scores(self):
        selector = EFESelector(_corridor_model(), action_bounds=(-3.0, 3.0), horizon=2)
        for belief in _BELIEFS:
            scores = np.asarray(selector._scores(belief, _PREF))
            chosen = selector.candidates[int(np.argmin(scores))]
            np.testing.assert_array_equal(selector.select(belief, _PREF), chosen)
