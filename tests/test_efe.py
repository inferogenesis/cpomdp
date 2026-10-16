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

import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp.dynamics import CallableDynamicsNoise
from cpomdp.efe import (
    EFEResult,
    PolicyEFETrace,
    expected_free_energy,
    policy_efe,
    policy_efe_trace,
)
from cpomdp.observation import CallableSensor, FixedSensor
from cpomdp.selection import ObservationGoal, Preference, StateGoal
from cpomdp.types import Belief, LinearGaussianModel


def _model(observation_model=None):
    # 2-state, 1-observation, 1-action, controllable.
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
    # Observation-space goal: m = 1 here, so goal is (1,), precision (1, 1).
    return Preference(target=[1.0], precision=[[2.0]])


def _numpy_efe(model, belief, action, goal, precision):
    """Independent NumPy recomputation of (G, pragmatic, epistemic).

    Deliberately mirrors efe.py's math in a separate library (NumPy) and a separate
    code path — no import of the kernel, no shared helpers.
    """
    A = np.asarray(model.dynamics_matrix)
    B = np.asarray(model.control_matrix)
    Q = np.asarray(model.dynamics_noise)
    C = np.asarray(model.observation_matrix)
    R = np.asarray(model.observation_noise)
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

        efe = expected_free_energy(model, belief, action, pref)
        g_ref, prag_ref, epi_ref = _numpy_efe(
            model, belief, action, pref.target, pref.precision
        )

        np.testing.assert_allclose(efe.g, g_ref, atol=1e-10)
        np.testing.assert_allclose(efe.pragmatic, prag_ref, atol=1e-10)
        np.testing.assert_allclose(efe.epistemic, epi_ref, atol=1e-10)

    def test_components_are_nonnegative(self):
        # epistemic = ½ ln(det S / det R) ≥ 0 (S ⪰ R); pragmatic ≥ 0 (Λ ⪰ 0).
        efe = expected_free_energy(
            _model(), _belief(), jnp.array([0.4]), _obs_preference()
        )
        assert efe.epistemic >= 0.0
        assert efe.pragmatic >= 0.0

    def test_raises_without_control(self):
        control_free = LinearGaussianModel(
            dynamics_matrix=[[1.0]],
            observation_matrix=[[1.0]],
            dynamics_noise=[[0.1]],
            observation_noise=[[0.5]],
            prior=Belief(mean=[0.0], cov=[[1.0]]),
        )
        pref = Preference(target=[0.0], precision=[[1.0]])
        with np.testing.assert_raises(ValueError):
            expected_free_energy(
                control_free, control_free.prior, jnp.array([0.0]), pref
            )


class TestTheResultRecord:
    """Every EFE function returns one record, read by name as the search results are."""

    def test_the_one_step_result_names_its_terms(self):
        result = expected_free_energy(
            _model(), _belief(), jnp.array([0.4]), _obs_preference()
        )
        assert isinstance(result, EFEResult)
        np.testing.assert_allclose(
            result.g, result.pragmatic - result.epistemic, atol=1e-12
        )

    def test_the_record_does_not_index_or_unpack(self):
        # A position named G on one function could name another field on the next.
        result = expected_free_energy(
            _model(), _belief(), jnp.array([0.4]), _obs_preference()
        )
        with pytest.raises(TypeError):
            result[0]  # ty: ignore[not-subscriptable]
        with pytest.raises(TypeError):
            _g, _pragmatic, _epistemic = result  # ty: ignore[not-iterable]

    def test_numpy_refuses_to_read_the_record_as_an_array(self):
        # np.argmin on the record would rank three stacked fields or a 0-d object, so
        # either way it returned an index that is not a candidate's.
        model, belief, pref = _model(), _belief(), _obs_preference()
        actions = jnp.array([[-1.0], [0.0], [0.5]])
        batch = jax.vmap(lambda a: expected_free_energy(model, belief, a, pref))(
            actions
        )
        with pytest.raises(TypeError, match=r"batch\.g"):
            np.argmin(batch)
        assert int(jnp.argmin(batch.g)) in range(3)

    def test_a_jitted_call_returns_the_whole_record(self):
        model, belief, pref = _model(), _belief(), _obs_preference()
        jitted = jax.jit(lambda a: expected_free_energy(model, belief, a, pref))
        result = jitted(jnp.array([0.4]))
        assert isinstance(result, EFEResult)
        assert result.g.shape == ()

    def test_the_rollout_returns_the_same_record(self):
        model, belief, pref = _model(), _belief(), _obs_preference()
        one_step = expected_free_energy(model, belief, jnp.array([0.4]), pref)
        rollout = policy_efe(model, belief, jnp.array([[0.4]]), pref)
        assert isinstance(rollout, EFEResult)
        for field in dataclasses.fields(EFEResult):
            np.testing.assert_allclose(
                getattr(rollout, field.name),
                getattr(one_step, field.name),
                atol=1e-12,
            )

    def test_the_trace_leads_with_the_same_fields(self):
        trace_fields = [field.name for field in dataclasses.fields(PolicyEFETrace)]
        result_fields = [field.name for field in dataclasses.fields(EFEResult)]
        assert trace_fields[:3] == result_fields

    def test_a_gradient_is_taken_of_g_with_the_record_as_its_aux(self):
        model, belief, preference = _model(), _belief(), _obs_preference()
        action = jnp.array([0.4])

        def scored(candidate):
            result = expected_free_energy(model, belief, candidate, preference)
            return result.g, result

        (value, record), gradient = jax.value_and_grad(scored, has_aux=True)(action)
        assert isinstance(record, EFEResult)
        assert value is record.g
        np.testing.assert_array_equal(
            gradient,
            jax.grad(lambda candidate: scored(candidate)[0])(action),
        )

    def test_a_vmap_batches_every_field(self):
        model, belief, pref = _model(), _belief(), _obs_preference()
        actions = jnp.array([[-1.0], [0.0], [0.5], [2.0]])
        batched = jax.vmap(lambda a: expected_free_energy(model, belief, a, pref))(
            actions
        )
        assert isinstance(batched, EFEResult)
        assert batched.pragmatic.shape == batched.epistemic.shape == (4,)


class TestTheGoalAUserHolds:
    """An ``ObservationGoal`` hands the scoring functions its preference by name."""

    def test_the_goal_carries_the_preference_it_is_scored_against(self):
        goal = ObservationGoal([1.0], action_bounds=(-1.0, 1.0), precision=[[2.0]])
        assert isinstance(goal.preference, Preference)
        np.testing.assert_array_equal(goal.preference.target, goal.target)
        np.testing.assert_array_equal(goal.preference.precision, goal.precision)
        model, belief, action = _model(), _belief(), jnp.array([0.4])
        np.testing.assert_array_equal(
            expected_free_energy(model, belief, action, goal.preference).g,
            expected_free_energy(model, belief, action, _obs_preference()).g,
        )

    @pytest.mark.parametrize(
        ("score", "action"),
        [
            (expected_free_energy, jnp.array([0.4])),
            (policy_efe, jnp.array([[0.4], [0.1]])),
            (policy_efe_trace, jnp.array([[0.4], [0.1]])),
        ],
    )
    def test_a_whole_goal_is_refused_naming_its_preference(self, score, action):
        goal = ObservationGoal([1.0], action_bounds=(-1.0, 1.0))
        with pytest.raises(TypeError, match=r"Pass its \.preference") as refusal:
            score(_model(), _belief(), action, goal)
        # The selectors and searches reach the same check, so it names no caller.
        assert score.__name__ not in str(refusal.value)


class TestWhatIsNotAPreference:
    """Anything but a ``Preference`` is refused with how to build one."""

    def _score(self, preference):
        return expected_free_energy(_model(), _belief(), jnp.array([0.4]), preference)

    def test_a_state_goal_is_refused_with_a_preference_over_readings(self):
        with pytest.raises(TypeError, match=r"Preference\(C @ target\)") as refusal:
            self._score(StateGoal([1.0, 0.0]))
        assert "target is a state" in str(refusal.value)
        assert ".preference" not in str(refusal.value)  # a StateGoal has none

    @pytest.mark.parametrize(
        "bare",
        [jnp.array([1.0]), np.array([1.0]), [1.0], ([1.0], [[2.0]])],
        ids=["jax-array", "numpy-array", "list", "goal-precision-pair"],
    )
    def test_a_bare_goal_vector_is_refused_with_how_to_wrap_it(self, bare):
        with pytest.raises(
            TypeError, match=r"Preference\(reading, precision\)"
        ) as refusal:
            self._score(bare)
        assert "ArrayImpl" not in str(refusal.value)

    def test_anything_else_is_refused_by_its_type(self):
        with pytest.raises(TypeError, match="a value of type NoneType"):
            self._score(None)
        look_alike = SimpleNamespace(goal=jnp.array([1.0]), precision=jnp.eye(1))
        with pytest.raises(TypeError, match="a value of type SimpleNamespace"):
            self._score(look_alike)

    def test_a_fault_inside_a_preference_surfaces_as_itself(self):
        missing_its_goal = object.__new__(Preference)
        object.__setattr__(missing_its_goal, "precision", jnp.eye(1))
        with pytest.raises(AttributeError):
            self._score(missing_its_goal)


class TestCollapseUnderFixedSensor:
    # Under a fixed sensor, Σ⁺/S/R don't depend on the action, so the epistemic
    # term is constant across actions and G's argmin is driven entirely by the
    # pragmatic term (ADR-003 made executable). NB this collapse holds for ALL
    # three candidate pragmatic forms — it does NOT prove the form choice (rfcs/004).
    def test_epistemic_is_action_invariant(self):
        model, belief, pref = _model(), _belief(), _obs_preference()
        actions = jnp.array([[-1.0], [0.0], [0.5], [2.0]])

        epistemics = jnp.array(
            [expected_free_energy(model, belief, a, pref).epistemic for a in actions]
        )
        np.testing.assert_allclose(epistemics, epistemics[0], atol=1e-12)

    def test_argmin_g_equals_argmin_pragmatic(self):
        model, belief, pref = _model(), _belief(), _obs_preference()
        actions = jnp.array([[-1.0], [0.0], [0.5], [2.0]])

        gs, prags = [], []
        for a in actions:
            efe = expected_free_energy(model, belief, a, pref)
            gs.append(efe.g)
            prags.append(efe.pragmatic)
        assert int(jnp.argmin(jnp.array(gs))) == int(jnp.argmin(jnp.array(prags)))


class TestTransforms:
    def test_jit_agrees_with_eager(self):
        model, belief, pref = _model(), _belief(), _obs_preference()
        action = jnp.array([0.4])

        eager = expected_free_energy(model, belief, action, pref).g
        jitted = jax.jit(lambda a: expected_free_energy(model, belief, a, pref).g)(
            action
        )
        np.testing.assert_allclose(jitted, eager, atol=1e-12)

    def test_vmap_over_candidate_actions(self):
        model, belief, pref = _model(), _belief(), _obs_preference()
        actions = jnp.array([[-1.0], [0.0], [0.5], [2.0]])

        batched = jax.vmap(lambda a: expected_free_energy(model, belief, a, pref).g)(
            actions
        )
        per_action = jnp.array(
            [expected_free_energy(model, belief, a, pref).g for a in actions]
        )
        np.testing.assert_allclose(batched, per_action, atol=1e-12)

    def test_grad_over_action_runs(self):
        model, belief, pref = _model(), _belief(), _obs_preference()
        grad = jax.grad(lambda a: expected_free_energy(model, belief, a, pref).g)(
            jnp.array([0.4])
        )
        assert grad.shape == (1,)
        assert bool(jnp.all(jnp.isfinite(grad)))


# --- Phase 2a: state-dependent sensing breaks the collapse, through the kernel ---
def _state_noise(x, params):
    """R(x) grows with velocity^2. Depends on x[1] (velocity) — the component the
    action moves in this double-integrator — so R(μ⁺) is genuinely action-dependent
    (R on x[0]/position would be flat one-step, since the action doesn't move it)."""
    return jnp.array([[params["base"] + params["slope"] * x[1] ** 2]])


def _callable_model():
    sensor = CallableSensor(
        observation_matrix=[[1.0, 0.0]],
        noise_fn=_state_noise,
        noise_params={"base": jnp.array(0.2), "slope": jnp.array(0.5)},
    )
    return _model(observation_model=sensor)


class TestCallableSensorBreaksCollapse:
    def test_epistemic_varies_across_actions(self):
        # The DUAL of TestCollapseUnderFixedSensor: with R(μ⁺) action-dependent, the
        # epistemic term is no longer flat across actions — the collapse is broken.
        model, belief, pref = _callable_model(), _belief(), _obs_preference()
        actions = jnp.array([[-1.0], [0.0], [0.5], [2.0]])
        epis = jnp.array(
            [expected_free_energy(model, belief, a, pref).epistemic for a in actions]
        )
        assert float(jnp.max(epis) - jnp.min(epis)) > 1e-6

    def test_grad_wrt_sensor_params_through_the_kernel(self):
        belief, pref = _belief(), _obs_preference()
        action = jnp.array([0.4])

        def efe_of_params(params):
            sensor = CallableSensor([[1.0, 0.0]], _state_noise, params)
            return expected_free_energy(
                _model(observation_model=sensor), belief, action, pref
            ).g

        grads = jax.tree_util.tree_leaves(
            jax.grad(efe_of_params)({"base": jnp.array(0.2), "slope": jnp.array(0.5)})
        )
        assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads)

    def test_nonpd_state_noise_gives_nan_epistemic_not_finite(self):
        # A CallableSensor whose R(x) is non-PD at the evaluated state has no real
        # ½ln det; the epistemic term must be NaN (caught downstream by the nan-safe
        # argmin), NOT a plausible-but-wrong finite value — slogdet's sign is kept.
        def neg_noise(x, params):
            # PD at the x=0 construction probe ([[1.0]]); non-PD at the μ⁺ the rollout
            # reaches here (velocity 0.2 ⇒ [[-1.0]]) — the runtime case the probe
            # cannot catch, where the kept slogdet sign must yield NaN.
            return jnp.array([[1.0 - 10.0 * x[1]]])

        model = _model(observation_model=CallableSensor([[1.0, 0.0]], neg_noise, {}))
        epi = expected_free_energy(
            model, _belief(), jnp.array([0.4]), _obs_preference()
        ).epistemic
        assert bool(jnp.isnan(epi))


class TestPrecisionControlsBalance:
    """The preference precision Λ is the explore/exploit knob — no hand-weighted split.

    Weak Λ lets the epistemic term win (seek the sharp-sensing beacon); sharp Λ makes
    the goal dominate (pursue it). Pins the bacillus demo's "one knob, the spectrum"
    as a real, reproducible capability of the public API.
    """

    def test_weak_precision_explores_sharp_precision_exploits(self):
        def beacon_noise(x, params):  # sharp (low R) at the beacon (1.5), foggy away
            sharp = jnp.exp(-((x[0] - 1.5) ** 2) / (2 * 0.6**2))
            return jnp.array([[0.05 + 0.8 * (1.0 - sharp)]])

        model = LinearGaussianModel(
            dynamics_matrix=[[1.0]],
            control_matrix=[[1.0]],
            dynamics_noise=[[0.05]],
            prior=Belief(mean=[0.0], cov=[[0.5]]),
            observation_model=CallableSensor([[1.0]], beacon_noise, {}),
        )
        belief = model.prior
        actions = jnp.linspace(-3.0, 3.0, 121)

        def chosen(precision):  # argmin-G action over the grid at this precision
            g = jnp.array(
                [
                    expected_free_energy(
                        model, belief, jnp.array([a]), Preference([0.0], [[precision]])
                    ).g
                    for a in actions
                ]
            )
            return float(actions[jnp.argmin(g)])

        weak = chosen(0.02)  # weak Λ -> seek the beacon (~ +1.5)
        sharp = chosen(3.0)  # sharp Λ -> pursue the goal (~ 0)
        assert weak > 1.0  # explores: heads to the beacon
        assert sharp < 0.5  # exploits: stays near the goal
        assert weak > sharp  # weakening Λ shifts toward exploration


class TestGaussianizeDispatch:
    def test_none_fast_path_matches_equivalent_fixed_sensor(self):
        # observation_model=None (inline fast path) and an equivalent FixedSensor
        # (routed through gaussianize) must give a byte-identical G — the dispatch is
        # behaviour-preserving on the linear case.
        belief, pref = _belief(), _obs_preference()
        action = jnp.array([0.4])
        g_none = expected_free_energy(_model(), belief, action, pref).g
        g_fixed = expected_free_energy(
            _model(
                observation_model=FixedSensor([[1.0, 0.0]], observation_noise=[[0.5]])
            ),
            belief,
            action,
            pref,
        ).g
        np.testing.assert_array_equal(g_none, g_fixed)


# --- Phase 2b: RFC-004 form-proof — the kernel is the FULL form (decomposition b),
# not mean-only, not the forbidden mix. NB the fixed-sensor collapse and the
# NumPy-oracle test above are NON-discriminating negative controls: they pass for
# all three forms. Discrimination needs a state-dependent sensor (here) so S(a)
# moves. The clean argmin "straddled-S flip" (vary S while holding R FIXED) is
# cleaner under the internal-Q regime (2d) — see TestStraddledSFlip there; in the
# R(x) regime S and R co-vary, so here we discriminate by value + the variance
# penalty + an independent Monte-Carlo cross-check.
def _ramp_noise(x, params):
    """Asymmetric R(x) = base·exp(rate·x[0]) (always > 0) so S differs across ±a."""
    return jnp.array([[params["base"] * jnp.exp(params["rate"] * x[0])]])


def _ramp_model():
    # Single integrator: μ⁺ = μ + a, o⁺ = position, so a=±1 gives a TIED mean term.
    sensor = CallableSensor(
        observation_matrix=[[1.0]],
        noise_fn=_ramp_noise,
        noise_params={"base": jnp.array(0.5), "rate": jnp.array(0.4)},
    )
    return LinearGaussianModel(
        dynamics_matrix=[[1.0]],
        dynamics_noise=[[0.1]],
        prior=Belief(mean=[0.0], cov=[[0.4]]),
        control_matrix=[[1.0]],
        observation_model=sensor,
    )


_RAMP_BELIEF = Belief(mean=[0.0], cov=[[0.4]])
_RAMP_PREF = Preference(target=[0.0], precision=[[1.0]])


def _form_components(model, belief, action):
    """Recompute the three rival EFE forms in NumPy (independent of the kernel)."""
    a = np.asarray(action, dtype=float)
    mu_pred = (
        np.asarray(model.dynamics_matrix) @ np.asarray(belief.mean)
        + np.asarray(model.control_matrix) @ a
    )
    sigma_pred = np.asarray(model.dynamics_matrix) @ np.asarray(
        belief.cov
    ) @ np.asarray(model.dynamics_matrix).T + np.asarray(model.dynamics_noise)
    c, r = model.observation_model.linearize(mu_pred)
    c, r = np.asarray(c), np.asarray(r)
    o = c @ mu_pred
    s = c @ sigma_pred @ c.T + r
    g, lam = np.asarray(_RAMP_PREF.target), np.asarray(_RAMP_PREF.precision)
    resid = o - g
    mean_term = 0.5 * resid @ lam @ resid
    var_term = 0.5 * np.trace(lam @ s)
    info_gain = 0.5 * (np.linalg.slogdet(s)[1] - np.linalg.slogdet(r)[1])
    h_qo = 0.5 * np.linalg.slogdet(2 * np.pi * np.e * s)[1]  # entropy of Q(o)
    full_prag = mean_term + var_term
    return {
        "o": o,
        "s": s,
        "mean_term": mean_term,
        "var_term": var_term,
        "h_qo": h_qo,
        "g_full": full_prag - info_gain,
        "g_forbidden": (full_prag - info_gain) - h_qo,  # KL-risk paired with −info-gain
    }


class TestFormProof:
    def test_pragmatic_carries_variance_penalty_not_mean_only(self):
        # a=±1 tie the mean term, but the kernel's pragmatic differs — so it carries
        # the ½tr(ΛS) term that mean-only drops.
        model = _ramp_model()
        a1, a2 = jnp.array([1.0]), jnp.array([-1.0])
        c1 = _form_components(model, _RAMP_BELIEF, a1)
        c2 = _form_components(model, _RAMP_BELIEF, a2)
        np.testing.assert_allclose(c1["mean_term"], c2["mean_term"], atol=1e-9)  # tied
        p1 = float(expected_free_energy(model, _RAMP_BELIEF, a1, _RAMP_PREF).pragmatic)
        p2 = float(expected_free_energy(model, _RAMP_BELIEF, a2, _RAMP_PREF).pragmatic)
        assert abs(p1 - p2) > 1e-3  # NOT tied -> not mean-only

    def test_pragmatic_matches_monte_carlo_cross_entropy(self):
        # MC of E_{o~N(o⁺,S)}[½(o−g)ᵀΛ(o−g)] == the kernel pragmatic (the EXPECTATION,
        # full form), NOT the point value ½(o⁺−g)ᵀΛ(o⁺−g) (mean-only). MC proves the
        # FORMULA; the analytic NumPy oracle (above) proves the implementation.
        model, action = _ramp_model(), jnp.array([0.6])
        p_kernel = float(
            expected_free_energy(model, _RAMP_BELIEF, action, _RAMP_PREF).pragmatic
        )
        c = _form_components(model, _RAMP_BELIEF, action)
        o, s = c["o"], c["s"]
        g, lam = np.asarray(_RAMP_PREF.target), np.asarray(_RAMP_PREF.precision)
        rng = np.random.default_rng(0)
        samples = rng.multivariate_normal(o, s, size=200_000)
        diff = samples - g
        mc = float(np.mean(0.5 * np.einsum("ni,ij,nj->n", diff, lam, diff)))
        np.testing.assert_allclose(p_kernel, mc, rtol=0.02)  # MC tolerance
        assert abs(p_kernel - float(c["mean_term"])) > 1e-3  # != mean-only point value

    def test_kernel_g_is_full_not_forbidden_mix(self):
        # Kernel G == full G; the forbidden mix (KL-risk − info-gain) differs by
        # EXACTLY H[Q(o)] — the double-counted predicted-observation entropy.
        model, action = _ramp_model(), jnp.array([0.6])
        g_kernel = float(
            expected_free_energy(model, _RAMP_BELIEF, action, _RAMP_PREF).g
        )
        c = _form_components(model, _RAMP_BELIEF, action)
        np.testing.assert_allclose(g_kernel, c["g_full"], atol=1e-9)  # kernel == full
        assert abs(g_kernel - float(c["g_forbidden"])) > 1e-3  # != forbidden mix
        np.testing.assert_allclose(
            c["g_full"] - c["g_forbidden"],
            c["h_qo"],
            atol=1e-9,  # gap == H[Q(o)]
        )


# --- Phase 2d: internal dynamics noise Q(μ⁺) breaks the collapse FROM THE INSIDE,
# with the observation noise R held FIXED (RFC-001 Section 8: the binding constraint
# lives in internal processing, not the sensor). dynamics_noise REPLACES the fixed
# dynamics_noise matrix when set (mirrors observation). Q is evaluated at μ⁺.
def _q_well(x, params):
    """Internal dynamics noise Q(x), low near 0 and growing with position²."""
    return jnp.array([[params["base"] + params["slope"] * x[0] ** 2]])


def _internal_q_model():
    dynamics_noise_model = CallableDynamicsNoise(
        noise_fn=_q_well,
        noise_params={"base": jnp.array(0.05), "slope": jnp.array(0.4)},
    )
    return LinearGaussianModel(
        dynamics_matrix=[[1.0]],
        observation_matrix=[[1.0]],
        observation_noise=[[0.3]],  # R is FIXED (no observation_model)
        prior=Belief(mean=[0.0], cov=[[0.2]]),
        control_matrix=[[1.0]],
        dynamics_noise_model=dynamics_noise_model,
    )


_INTERNAL_BELIEF = Belief(mean=[0.0], cov=[[0.2]])
_INTERNAL_PREF = Preference(target=[0.0], precision=[[1.0]])


class TestInternalProcessNoise:
    def test_epistemic_reenters_via_internal_Q_with_R_fixed(self):
        # The internal dual of 2a: R is constant, yet epistemic varies across actions
        # because Q(μ⁺) — and so Σ⁺ and S — depend on the action.
        model = _internal_q_model()
        actions = jnp.array([[-1.0], [0.0], [0.5], [2.0]])
        epis = jnp.array(
            [
                expected_free_energy(
                    model, _INTERNAL_BELIEF, a, _INTERNAL_PREF
                ).epistemic
                for a in actions
            ]
        )
        assert float(jnp.max(epis) - jnp.min(epis)) > 1e-6

    def test_Q_evaluated_at_mu_pred_not_mu(self):
        # REQUIRED guard: Q must be evaluated at μ⁺ = Aμ + Ba (action-dependent), not
        # at μ. Cross-check the kernel epistemic against BOTH numpy conventions.
        model = _internal_q_model()
        action = jnp.array([1.3])
        epi = float(
            expected_free_energy(
                model, _INTERNAL_BELIEF, action, _INTERNAL_PREF
            ).epistemic
        )
        a_mat, b_mat = np.array([[1.0]]), np.array([[1.0]])
        c_mat, r_mat = np.array([[1.0]]), np.array([[0.3]])
        mu, sigma, a = np.array([0.0]), np.array([[0.2]]), np.array([1.3])
        mu_pred = a_mat @ mu + b_mat @ a
        params = {"base": 0.05, "slope": 0.4}

        def epi_with(q):
            sp = a_mat @ sigma @ a_mat.T + np.asarray(q)
            s = c_mat @ sp @ c_mat.T + r_mat
            return 0.5 * (np.linalg.slogdet(s)[1] - np.linalg.slogdet(r_mat)[1])

        np.testing.assert_allclose(epi, epi_with(_q_well(mu_pred, params)), atol=1e-9)
        assert abs(epi - epi_with(_q_well(mu, params))) > 1e-3  # NOT Q(μ)

    def test_fixed_path_unchanged_when_dynamics_noise_none(self):
        # dynamics_noise=None must give a byte-identical Σ⁺/G to the matrix path.
        base = LinearGaussianModel(
            dynamics_matrix=[[1.0]],
            observation_matrix=[[1.0]],
            dynamics_noise=[[0.1]],
            observation_noise=[[0.3]],
            prior=Belief(mean=[0.0], cov=[[0.2]]),
            control_matrix=[[1.0]],
        )
        action = jnp.array([0.7])
        g_ref = _numpy_efe(
            base,
            _INTERNAL_BELIEF,
            action,
            _INTERNAL_PREF.target,
            _INTERNAL_PREF.precision,
        )[0]
        g_kernel = expected_free_energy(
            base, _INTERNAL_BELIEF, action, _INTERNAL_PREF
        ).g
        np.testing.assert_allclose(g_kernel, g_ref, atol=1e-10)

    def test_grad_wrt_Q_params(self):
        action = jnp.array([0.6])

        def efe_of_params(params):
            dynamics_noise_model = CallableDynamicsNoise(_q_well, params)
            model = LinearGaussianModel(
                dynamics_matrix=[[1.0]],
                observation_matrix=[[1.0]],
                observation_noise=[[0.3]],
                prior=Belief(mean=[0.0], cov=[[0.2]]),
                control_matrix=[[1.0]],
                dynamics_noise_model=dynamics_noise_model,
            )
            return expected_free_energy(
                model, _INTERNAL_BELIEF, action, _INTERNAL_PREF
            ).g

        grads = jax.tree_util.tree_leaves(
            jax.grad(efe_of_params)({"base": jnp.array(0.05), "slope": jnp.array(0.4)})
        )
        assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads)


# The clean straddled-S flip — R FIXED, S varied via Q(μ⁺), so the full form picks
# S=1/Λ while the forbidden mix picks S=2/Λ (opposite argmins). This is the regime
# where the flip's math is honest (R does not co-vary); see the 2b deviation note.
def _q_ramp(x, params):
    """Asymmetric Q(x) = base + slope·x[0] so S straddles 1/Λ and 2/Λ across ±a."""
    return jnp.array([[params["base"] + params["slope"] * x[0]]])


def _flip_model():
    dynamics_noise_model = CallableDynamicsNoise(
        _q_ramp, {"base": jnp.array(1.2), "slope": jnp.array(-0.5)}
    )
    return LinearGaussianModel(
        dynamics_matrix=[[1.0]],
        observation_matrix=[[1.0]],
        observation_noise=[[0.2]],
        prior=Belief(mean=[0.0], cov=[[0.1]]),
        control_matrix=[[1.0]],
        dynamics_noise_model=dynamics_noise_model,
    )


class TestStraddledSFlip:
    def test_full_picks_low_S_forbidden_picks_high_S(self):
        model = _flip_model()
        belief = Belief(mean=[0.0], cov=[[0.1]])
        pref = Preference(target=[0.0], precision=[[1.0]])
        a_low, a_high = jnp.array([1.0]), jnp.array([-1.0])  # S≈1/Λ, S≈2/Λ; tied mean

        def s_of(action):
            a = np.asarray(action, dtype=float)
            mu_pred = np.array([[1.0]]) @ np.array([0.0]) + np.array([[1.0]]) @ a
            q = np.asarray(model.dynamics_noise_model.noise_at(mu_pred))
            sp = np.array([[1.0]]) @ np.array([[0.1]]) @ np.array([[1.0]]).T + q
            s = np.array([[1.0]]) @ sp @ np.array([[1.0]]).T + np.array([[0.2]])
            return float(s[0, 0])

        def forbidden_g(action):
            gf = float(expected_free_energy(model, belief, action, pref).g)
            return gf - 0.5 * np.log(2 * np.pi * np.e * s_of(action))  # gf − H[Q(o)]

        g_low = float(expected_free_energy(model, belief, a_low, pref).g)
        g_high = float(expected_free_energy(model, belief, a_high, pref).g)
        assert g_low < g_high  # full (kernel) prefers S=1/Λ
        assert forbidden_g(a_high) < forbidden_g(a_low)  # forbidden flips to S=2/Λ
