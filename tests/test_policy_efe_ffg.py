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

from _support import unit_prior
from cpomdp.backends.coupling import CouplingGraphBackend
from cpomdp.efe import (
    PolicyEfeTrace,
    _ffg_efe_step,
    policy_efe,
    policy_efe_ffg,
    policy_efe_ffg_trace,
)
from cpomdp.ffg.factors.linear_gaussian import GaussianTransition
from cpomdp.ffg.graph import Coupling, CouplingGraph
from cpomdp.observation import CallableSensor, FixedSensor
from cpomdp.selection import Preference
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
        obs_ffg = CallableSensor(c, _rx_noise, params)
        obs_flat = CallableSensor(c, _rx_noise, params)
    else:
        obs_ffg = FixedSensor(c, observation_noise=r0)
        obs_flat = None
    graph = CouplingGraph(
        root=0, dims=(2,), couplings=(), observation_models={0: obs_ffg}
    )
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
        couplings=(Coupling(0, 1, [[0.8]], coupling_noise=[[0.05]]),),
        observation_models={1: FixedSensor([[1.0]], observation_noise=[[0.1]])},
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
            g, parts = policy_efe_ffg(
                backend, belief, policy, pref, info_block=info_block
            )
            trace = policy_efe_ffg_trace(
                backend, belief, policy, pref, info_block=info_block
            )
            np.testing.assert_array_equal(g, jnp.sum(trace.g))
            np.testing.assert_array_equal(parts["pragmatic"], jnp.sum(trace.pragmatic))
            np.testing.assert_array_equal(parts["epistemic"], jnp.sum(trace.epistemic))


# --- oracle 1: H=1 is one _ffg_efe_step (holds with couplings) ---------------------
class TestFfgH1ReducesToFfgEfeStep:
    def test_scalars_byte_identical_to_single_step(self):
        backend, belief, pref = _coupled_backend(), _coupled_belief(), _pref()
        info_block = tuple(backend.block(0))
        policy = _policy([0.6])
        g, parts = policy_efe_ffg(backend, belief, policy, pref, info_block=info_block)

        predicted = backend.predicted_belief(belief, policy[0])
        observation_noise = backend.observation_noise_at(predicted.mean)
        observation_matrix, _ = backend.observation_model
        g_ref, parts_ref = _ffg_efe_step(
            predicted.mean,
            predicted.cov,
            observation_matrix,
            observation_noise,
            pref.goal,
            pref.precision,
            info_block,
        )
        np.testing.assert_array_equal(g, g_ref)
        np.testing.assert_array_equal(parts["pragmatic"], parts_ref["pragmatic"])
        np.testing.assert_array_equal(parts["epistemic"], parts_ref["epistemic"])

    def test_trace_moments_match_predicted_belief(self):
        backend, belief, pref = _coupled_backend(), _coupled_belief(), _pref()
        info_block = tuple(backend.block(0))
        policy = _policy([0.6])
        trace = policy_efe_ffg_trace(
            backend, belief, policy, pref, info_block=info_block
        )
        predicted = backend.predicted_belief(belief, policy[0])
        np.testing.assert_array_equal(trace.mu_pred[0], predicted.mean)
        np.testing.assert_array_equal(trace.sigma_pred[0], predicted.cov)


# --- oracle 2: no couplings + whole state is policy_efe ----------------------------
class TestFfgNoCouplingReducesToPolicyEfe:
    def _check(self, *, state_dependent):
        backend, model = _single_node_pair(state_dependent=state_dependent)
        belief, info_block = _single_belief(), range(2)
        pref = Preference(goal=[1.0], precision=[[2.0]])
        for h in (2, 3):
            policy = _policy([0.5, -0.4, 0.2][:h])
            g, parts = policy_efe_ffg(
                backend, belief, policy, pref, info_block=info_block
            )
            g_ref, parts_ref = policy_efe(model, belief, policy, pref)
            np.testing.assert_allclose(float(g), float(g_ref), atol=1e-9)
            np.testing.assert_allclose(
                float(parts["pragmatic"]), float(parts_ref["pragmatic"]), atol=1e-9
            )
            np.testing.assert_allclose(
                float(parts["epistemic"]), float(parts_ref["epistemic"]), atol=1e-9
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
        assert trace.mu_pred.shape == (3, 2)  # n_total = 2
        assert trace.sigma_pred.shape == (3, 2, 2)
        assert trace.sigma_post.shape == (3, 2, 2)
        assert trace.s.shape == (3, 1, 1)  # m = 1 (one observed channel)

    def test_is_policy_efe_trace_pytree(self):
        backend, belief, pref = _coupled_backend(), _coupled_belief(), _pref()
        info_block = tuple(backend.block(0))
        policy = _policy([0.4, -0.3])
        trace = policy_efe_ffg_trace(
            backend, belief, policy, pref, info_block=info_block
        )
        assert isinstance(trace, PolicyEfeTrace)
        assert len(jax.tree_util.tree_leaves(trace)) == 7

    def test_sigma_post_is_tighter_than_sigma_pred(self):
        # Observing can only shrink uncertainty: Σ_post ≼ Σ⁺ (trace of the drop ≥ 0).
        backend, belief, pref = _coupled_backend(), _coupled_belief(), _pref()
        info_block = tuple(backend.block(0))
        policy = _policy([0.4, -0.3])
        trace = policy_efe_ffg_trace(
            backend, belief, policy, pref, info_block=info_block
        )
        drop = np.asarray(trace.sigma_pred) - np.asarray(trace.sigma_post)
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
        np.testing.assert_allclose(jitted.sigma_post, eager.sigma_post, atol=1e-12)

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
        assert batched.sigma_pred.shape == (2, 2, 2, 2)

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
