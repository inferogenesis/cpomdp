"""The H-step EFE rollout over a branching FFG backend, and its two oracle reductions.

``policy_efe_ffg`` is the FFG counterpart of ``policy_efe``: it drives ``_ffg_efe_step``
over the backend's predicted joint each step, aiming the epistemic at a node's block
(``info_block``) rather than the whole observation. ``policy_efe_ffg_trace`` keeps
the whole
per-step record instead of summing it, so the crossover statistic and the rollout
diagnostics can read the per-step pragmatic/epistemic split and the covariance path.

Two oracles pin it, mirroring the flat rollout's:

- **H=1 reduces to ``_ffg_efe_step``** — one rollout step is one single-step FFG EFE, so
  the summed scalars are byte-identical to a hand-run ``_ffg_efe_step`` on the backend's
  ``predicted_belief``. This holds *with* couplings.
- **No couplings + whole-state info block reduces to ``policy_efe``** — a single-node
  backend is the flat linear-Gaussian model, so the FFG rollout must agree with
  ``policy_efe`` step for step, under both a fixed sensor and ``R(x)``. Agreement is
  numerical (``allclose``), not byte-identical: the FFG predicts through the precision
  form (``to_moment``) while the flat rollout predicts through ``AΣAᵀ + Q``.

Collection-red until the two functions land: the import of ``policy_efe_ffg`` /
``policy_efe_ffg_trace`` is the build signal.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _support import unit_prior
from cpomdp.backends.coupling import CouplingGraphBackend
from cpomdp.efe import (
    EfeResult,
    PolicyEfeTrace,
    _ffg_efe_step,
    policy_efe,
    policy_efe_ffg,
    policy_efe_ffg_trace,
)
from cpomdp.ffg.factors.linear_gaussian import (
    CallableGaussianObservation,
    GaussianCoupling,
    GaussianObservation,
    GaussianTransition,
)
from cpomdp.ffg.graph import Coupling, CouplingGraph
from cpomdp.observation import CallableSensor
from cpomdp.selection import ObservationGoal, Preference
from cpomdp.types import Belief, LinearGaussianModel


# --- fixtures ----------------------------------------------------------------------
def _rx_noise(x, params):
    """Node-local R(x) = R0·(1 + gain·xᵀx) — sharpest as the node nears 0."""
    return params["R0"] * (1.0 + params["gain"] * jnp.dot(x, x))


def _single_node_pair(*, state_dependent):
    """An equivalent (FFG backend, flat model) pair — one node, no couplings.

    With no couplings μ⁺ = μ⁻, so the FFG rollout with the whole-state info block
    must match
    the flat ``policy_efe`` rollout. ``state_dependent`` swaps the fixed sensor for the
    matched R(x) sensor on both sides.
    """
    a = np.array([[1.0, 0.1], [0.0, 1.0]])  # A
    q = np.array([[0.1, 0.0], [0.0, 0.1]])  # Q
    c = np.array([[1.0, 0.0]])  # C
    b = np.array([[1.0], [0.0]])  # B drives observed state[0] (non-flat G)
    r0 = np.array([[0.5]])  # R0
    params = {"R0": r0, "gain": 0.4}
    if state_dependent:
        obs_ffg = CallableGaussianObservation(c, _rx_noise, params)
        obs_flat = CallableSensor(c, _rx_noise, params)
    else:
        obs_ffg = GaussianObservation(c, observation_noise=r0)
        obs_flat = None
    graph = CouplingGraph(root=0, dims=(2,), couplings=(), observations={0: obs_ffg})
    backend = CouplingGraphBackend(
        graph,
        (GaussianTransition(a, dynamics_noise=q),),
        prior=unit_prior(graph),
        control_matrix=b,
    )
    model = LinearGaussianModel(
        dynamics_matrix=a,
        observation_matrix=c,
        dynamics_noise=q,
        observation_noise=r0,  # placeholder under a callable sensor; ignored there
        prior=Belief(mean=np.zeros(2), cov=np.eye(2)),
        control_matrix=b,
        observation_model=obs_flat,
    )
    return backend, model


def _coupled_backend():
    """A genuine two-node FFG: node 0 drives node 1 via a coupling; observe node 1."""
    graph = CouplingGraph(
        root=0,
        dims=(1, 1),
        couplings=(
            Coupling(0, 1, GaussianCoupling([[0.8]], coupling_noise=[[0.05]]), 1.0),
        ),
        observations={1: GaussianObservation([[1.0]], observation_noise=[[0.1]])},
    )
    transitions = (
        GaussianTransition([[0.7]], dynamics_noise=[[0.1]]),
        GaussianTransition([[0.5]], dynamics_noise=[[0.08]]),
    )
    return CouplingGraphBackend(
        graph, transitions, prior=unit_prior(graph), control_matrix=[[1.0], [0.0]]
    )


def _coupled_belief():
    return Belief(mean=[0.2, -0.1], cov=[[0.7, 0.1], [0.1, 0.4]])


def _single_belief():
    return Belief(mean=[0.3, -0.2], cov=[[0.7, 0.1], [0.1, 0.4]])


def _pref():
    return Preference(goal=[0.5], precision=[[2.0]])


def _policy(values):
    return jnp.asarray([[v] for v in values], dtype=float)


# --- the summed rollout and the trace share one arithmetic -------------------------
class TestFfgSumsEqualScalars:
    """``policy_efe_ffg``'s summed scalars equal the trace columns summed, bitwise."""

    def test_matches_across_horizons(self):
        backend, belief, pref = _coupled_backend(), _coupled_belief(), _pref()
        info_block = tuple(backend.block(0))
        for h in (1, 2, 3):
            policy = _policy([0.4, -0.3, 0.1][:h])
            efe = policy_efe_ffg(backend, belief, policy, pref, info_block=info_block)
            assert isinstance(efe, EfeResult)
            trace = policy_efe_ffg_trace(
                backend, belief, policy, pref, info_block=info_block
            )
            np.testing.assert_array_equal(efe.g, jnp.sum(trace.g))
            np.testing.assert_array_equal(efe.pragmatic, jnp.sum(trace.pragmatic))
            np.testing.assert_array_equal(efe.epistemic, jnp.sum(trace.epistemic))


# --- oracle 1: H=1 is one _ffg_efe_step (holds with couplings) ---------------------
class TestFfgH1ReducesToFfgEfeStep:
    def test_scalars_byte_identical_to_single_step(self):
        backend, belief, pref = _coupled_backend(), _coupled_belief(), _pref()
        info_block = tuple(backend.block(0))
        policy = _policy([0.6])
        efe = policy_efe_ffg(backend, belief, policy, pref, info_block=info_block)

        predicted = backend.predicted_belief(belief, policy[0])
        observation_noise = backend.observation_noise_at(predicted.mean)
        observation_matrix, _ = backend.observation_model
        efe_ref = _ffg_efe_step(
            predicted.mean,
            predicted.cov,
            observation_matrix,
            observation_noise,
            pref.goal,
            pref.precision,
            info_block,
        )
        np.testing.assert_array_equal(efe.g, efe_ref.g)
        np.testing.assert_array_equal(efe.pragmatic, efe_ref.pragmatic)
        np.testing.assert_array_equal(efe.epistemic, efe_ref.epistemic)

    def test_trace_moments_match_predicted_belief(self):
        backend, belief, pref = _coupled_backend(), _coupled_belief(), _pref()
        info_block = tuple(backend.block(0))
        policy = _policy([0.6])
        trace = policy_efe_ffg_trace(
            backend, belief, policy, pref, info_block=info_block
        )
        predicted = backend.predicted_belief(belief, policy[0])
        np.testing.assert_array_equal(trace.predicted_mean[0], predicted.mean)
        np.testing.assert_array_equal(trace.predicted_cov[0], predicted.cov)


# --- oracle 2: no couplings + whole state is policy_efe ----------------------------
class TestFfgNoCouplingReducesToPolicyEfe:
    def _check(self, *, state_dependent):
        backend, model = _single_node_pair(state_dependent=state_dependent)
        belief, info_block = _single_belief(), range(2)
        pref = Preference(goal=[1.0], precision=[[2.0]])
        for h in (2, 3):
            policy = _policy([0.5, -0.4, 0.2][:h])
            efe = policy_efe_ffg(backend, belief, policy, pref, info_block=info_block)
            efe_ref = policy_efe(model, belief, policy, pref)
            np.testing.assert_allclose(float(efe.g), float(efe_ref.g), atol=1e-9)
            np.testing.assert_allclose(
                float(efe.pragmatic), float(efe_ref.pragmatic), atol=1e-9
            )
            np.testing.assert_allclose(
                float(efe.epistemic), float(efe_ref.epistemic), atol=1e-9
            )

    def test_fixed_sensor(self):
        self._check(state_dependent=False)

    def test_state_dependent_sensor(self):
        # The R(μ⁺) contraction is exercised on both paths; μ⁻ = μ⁺ (no couplings) makes
        # them agree despite the state-dependence.
        self._check(state_dependent=True)


# --- the trace is the flat PolicyEfeTrace, at the joint shapes ---------------------
class TestFfgTraceShape:
    def test_field_shapes(self):
        backend, belief, pref = _coupled_backend(), _coupled_belief(), _pref()
        info_block = tuple(backend.block(0))
        policy = _policy([0.4, -0.3, 0.1])
        trace = policy_efe_ffg_trace(
            backend, belief, policy, pref, info_block=info_block
        )
        assert trace.g.shape == (3,)
        assert trace.pragmatic.shape == (3,)
        assert trace.epistemic.shape == (3,)
        assert trace.predicted_mean.shape == (3, 2)  # n_total = 2
        assert trace.predicted_cov.shape == (3, 2, 2)
        assert trace.posterior_cov.shape == (3, 2, 2)
        assert trace.predicted_observation_cov.shape == (
            3,
            1,
            1,
        )  # m = 1 (one observed channel)

    def test_is_policy_efe_trace_pytree(self):
        backend, belief, pref = _coupled_backend(), _coupled_belief(), _pref()
        info_block = tuple(backend.block(0))
        policy = _policy([0.4, -0.3])
        trace = policy_efe_ffg_trace(
            backend, belief, policy, pref, info_block=info_block
        )
        assert isinstance(trace, PolicyEfeTrace)
        assert len(jax.tree_util.tree_leaves(trace)) == 7

    def test_posterior_cov_is_tighter_than_predicted_cov(self):
        # Observing can only shrink uncertainty: Σ_post ≼ Σ⁺ (trace of the drop ≥ 0).
        backend, belief, pref = _coupled_backend(), _coupled_belief(), _pref()
        info_block = tuple(backend.block(0))
        policy = _policy([0.4, -0.3])
        trace = policy_efe_ffg_trace(
            backend, belief, policy, pref, info_block=info_block
        )
        drop = np.asarray(trace.predicted_cov) - np.asarray(trace.posterior_cov)
        for step in drop:
            assert np.trace(step) >= -1e-12


# --- composes under the transforms, with the backend held as a closure constant ----
class TestFfgTraceTransforms:
    def test_jit_agrees_with_eager(self):
        backend, belief, pref = _coupled_backend(), _coupled_belief(), _pref()
        info_block = tuple(backend.block(0))
        policy = _policy([0.4, -0.2])
        run = jax.jit(
            lambda b, p: policy_efe_ffg_trace(
                backend, b, p, pref, info_block=info_block
            )
        )
        eager = policy_efe_ffg_trace(
            backend, belief, policy, pref, info_block=info_block
        )
        jitted = run(belief, policy)
        np.testing.assert_allclose(jitted.g, eager.g, atol=1e-12)
        np.testing.assert_allclose(
            jitted.posterior_cov, eager.posterior_cov, atol=1e-12
        )

    def test_vmap_over_policies_adds_a_batch_axis(self):
        backend, belief, pref = _coupled_backend(), _coupled_belief(), _pref()
        info_block = tuple(backend.block(0))
        policies = jnp.stack([_policy([0.4, -0.2]), _policy([-0.3, 0.5])])  # (2, 2, 1)
        batched = jax.vmap(
            lambda p: policy_efe_ffg_trace(
                backend, belief, p, pref, info_block=info_block
            )
        )(policies)
        assert batched.g.shape == (2, 2)
        assert batched.predicted_cov.shape == (2, 2, 2, 2)

    def test_grad_of_summed_g_is_finite(self):
        backend, belief, pref = _coupled_backend(), _coupled_belief(), _pref()
        info_block = tuple(backend.block(0))
        policy = _policy([0.4, -0.2])
        grad = jax.grad(
            lambda p: jnp.sum(
                policy_efe_ffg_trace(backend, belief, p, pref, info_block=info_block).g
            )
        )(policy)
        assert np.all(np.isfinite(np.asarray(grad)))
        assert grad.shape == (2, 1)


# --- the goal a user holds ---------------------------------------------------------
@pytest.mark.parametrize("score", [policy_efe_ffg, policy_efe_ffg_trace])
def test_a_whole_goal_is_refused_naming_its_preference(score):
    backend, belief = _coupled_backend(), _coupled_belief()
    goal = ObservationGoal([0.5], action_bounds=(-1.0, 1.0), precision=[[2.0]])
    info_block = tuple(backend.block(0))
    policy = _policy([0.4])
    with pytest.raises(TypeError, match=r"Pass its \.preference") as refusal:
        score(backend, belief, policy, goal, info_block=info_block)
    assert score.__name__ not in str(refusal.value)
    np.testing.assert_array_equal(
        score(backend, belief, policy, goal.preference, info_block=info_block).g,
        score(backend, belief, policy, _pref(), info_block=info_block).g,
    )
