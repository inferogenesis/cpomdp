"""The exhaustive enumerated search, scoring on an FFG backend (the scorer seam).

``EnumeratedEfeSearch`` scores every ``A^H`` policy. On a flat model it scores with
``policy_efe``; ``over_backend`` scores with ``policy_efe_ffg`` on a coupling graph,
aimed at a node's block. The enumeration, the certificate, and the cost are shared —
only the per-policy scoring changes.

The oracle mirrors the rollout's: on a coupling-free single-node backend with the
whole-state info block the FFG search reduces to the flat search, both under a fixed
sensor
and under ``R(x)`` — same argmin policy, same ``G`` vector at ``allclose`` (the FFG path
predicts through the precision form, the flat through ``AΣAᵀ + Q``). A coupled-tree
smoke confirms it runs where the flat search cannot.

Collection-red until ``over_backend`` lands.
"""

import jax.numpy as jnp
import numpy as np

from _support import unit_prior
from cpomdp.backends.coupling import CouplingGraphBackend
from cpomdp.enumeration import EnumeratedEfeSearch, FiniteActionSet, SearchWarrant
from cpomdp.ffg.factors.linear_gaussian import GaussianTransition
from cpomdp.ffg.graph import Coupling, CouplingGraph
from cpomdp.observation import CallableSensor, FixedSensor
from cpomdp.selection import Preference
from cpomdp.types import Belief, LinearGaussianModel


# --- fixtures (mirror tests/test_policy_efe_ffg.py) --------------------------------
def _rx_noise(x, params):
    return params["R0"] * (1.0 + params["gain"] * jnp.dot(x, x))


def _single_node_pair(*, state_dependent):
    """An equivalent (FFG backend, flat model) pair — one node, no couplings."""
    a = np.array([[1.0, 0.1], [0.0, 1.0]])
    q = np.array([[0.1, 0.0], [0.0, 0.1]])
    c = np.array([[1.0, 0.0]])
    b = np.array([[1.0], [0.0]])  # drives observed state[0], so G varies with action
    r0 = np.array([[0.5]])
    params = {"R0": r0, "gain": 0.4}
    if state_dependent:
        obs_ffg = CallableSensor(c, _rx_noise, params)
        obs_flat = CallableSensor(c, _rx_noise, params)
    else:
        obs_ffg = FixedSensor(c, observation_noise=r0)
        obs_flat = None
    graph = CouplingGraph(dims=(2,), couplings=(), observation_models={0: obs_ffg})
    backend = CouplingGraphBackend(
        graph,
        (GaussianTransition(a, dynamics_noise=q),),
        prior=unit_prior(graph),
        control_matrix=b,
    )
    model = LinearGaussianModel(
        dynamics_matrix=a,
        observation_matrix=None if state_dependent else c,
        dynamics_noise=q,
        observation_noise=None if state_dependent else r0,
        prior=Belief(mean=np.zeros(2), cov=np.eye(2)),
        control_matrix=b,
        observation_model=obs_flat,
    )
    return backend, model


def _coupled_backend():
    graph = CouplingGraph(
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


def _action_set():
    return FiniteActionSet([[-1.0], [0.0], [1.0]], version="v1")


def _single_belief():
    return Belief(mean=[0.3, -0.2], cov=[[0.7, 0.1], [0.1, 0.4]])


def _pref():
    return Preference(goal=[1.0], precision=[[2.0]])


# --- oracle: no couplings + whole-state info block reduces to the flat search ------
class TestOverBackendReducesToFlat:
    def _check(self, *, state_dependent):
        backend, model = _single_node_pair(state_dependent=state_dependent)
        action_set, belief, pref = _action_set(), _single_belief(), _pref()
        for h in (2, 3):
            flat = EnumeratedEfeSearch(model, action_set, horizon=h)
            ffg = EnumeratedEfeSearch.over_backend(
                backend, action_set, info_block=range(2), horizon=h
            )
            r_flat = flat.evaluate(belief, pref)
            r_ffg = ffg.evaluate(belief, pref)
            np.testing.assert_allclose(
                np.asarray(r_ffg.g), np.asarray(r_flat.g), atol=1e-9
            )
            np.testing.assert_array_equal(r_ffg.best_policy, r_flat.best_policy)

    def test_fixed_sensor(self):
        self._check(state_dependent=False)

    def test_state_dependent_sensor(self):
        self._check(state_dependent=True)


# --- the certificate / cost / warrant are the shared enumeration, unchanged --------
class TestOverBackendSharedAccounting:
    def test_certificate_cost_and_warrant(self):
        backend, _ = _single_node_pair(state_dependent=False)
        ffg = EnumeratedEfeSearch.over_backend(
            backend, _action_set(), info_block=range(2), horizon=2
        )
        assert ffg.warrant is SearchWarrant.PROVED
        assert ffg.certificate.complete
        assert ffg.n_policies == 9  # 3^2
        assert ffg.cost_per_cycle == 18  # |A|^H * H = 9 * 2

    def test_rejects_an_action_set_of_other_n_controls(self):
        backend, _ = _single_node_pair(state_dependent=False)  # p = 1
        wrong = FiniteActionSet([[-1.0, 0.0], [1.0, 0.0]], version="v1")  # p = 2
        with np.testing.assert_raises(ValueError):
            EnumeratedEfeSearch.over_backend(
                backend, wrong, info_block=range(2), horizon=2
            )


# --- it runs on the coupled tree, where the flat search cannot ---------------------
class TestOverBackendOnCoupledTree:
    def test_runs_and_enumerates_varying_sequences(self):
        backend = _coupled_backend()
        belief = Belief(mean=[0.2, -0.1], cov=[[0.7, 0.1], [0.1, 0.4]])
        pref = Preference(goal=[0.5], precision=[[2.0]])
        info_block = tuple(backend.block(0))
        ffg = EnumeratedEfeSearch.over_backend(
            backend, _action_set(), info_block=info_block, horizon=2
        )
        result = ffg.evaluate(belief, pref)
        assert result.best_policy.shape == (2, 1)
        assert result.g.shape == (9,)
        assert ffg.certificate.complete
