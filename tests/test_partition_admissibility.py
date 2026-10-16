"""Partition admissibility under EFE (ADR-018, issue #27 Phase 4).

A carry partition (ADR-016) that severs an EFE-load-bearing edge silently collapses the
instrumental epistemic: the covariance path from the sensed state to the latent whose
ambiguity the agent reduces is cut, so the epistemic term dies while a drift eyeball
still passes. Relevance is a *declared* physics property — the modeller flags the
information-bearing edges (gradient rides receptor→CheA→CheY; methylation carries none),
because a structural derivation would wrongly flag CheA→CheB (observed and coupled, yet
gradient-blind). So a per-edge ``efe_relevant`` flag on ``Coupling`` drives the guard.
It fires on every EFE path (the selector, the rollouts and the searches), never on pure
filtering.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _support import unit_prior
from cpomdp.agent import Agent
from cpomdp.backends.base import InadmissiblePartitionError
from cpomdp.backends.coupling import CouplingGraphBackend
from cpomdp.efe import policy_efe_ffg, policy_efe_ffg_trace
from cpomdp.enumeration import ChunkedEFESearch, EnumeratedEFESearch, FiniteActionSet
from cpomdp.ffg.factors.linear_gaussian import GaussianTransition
from cpomdp.ffg.graph import Coupling, CouplingGraph
from cpomdp.observation import CallableSensor, FixedSensor
from cpomdp.selection import FFGEFESelector, ObservationGoal, Preference
from cpomdp.types import Belief


def _sharp_near_zero(state, params):
    """R(x) sharpest at zero, so the action moves the information term."""
    return params["r0"] * (1.0 + jnp.dot(state, state)) * jnp.eye(1)


def _hub_graph(*, fast_relevant=True, slow_relevant=False, fast_sensor=None):
    # hub 0 -> fast 1 (gradient path) and slow 2 (methylation-like); observe 1 and 2.
    if fast_sensor is None:
        fast_sensor = FixedSensor([[1.0]], observation_noise=[[0.1]])
    return CouplingGraph(
        dims=(1, 1, 1),
        couplings=(
            Coupling(
                0,
                1,
                [[0.8]],
                coupling_noise=[[0.05]],
                efe_relevant=fast_relevant,
            ),
            Coupling(
                0,
                2,
                [[0.6]],
                coupling_noise=[[0.05]],
                efe_relevant=slow_relevant,
            ),
        ),
        observation_models={
            1: fast_sensor,
            2: FixedSensor([[1.0]], observation_noise=[[0.1]]),
        },
    )


def _backend(graph, partition=None):
    transitions = (
        GaussianTransition([[0.7]], dynamics_noise=[[0.1]]),
        GaussianTransition([[0.5]], dynamics_noise=[[0.08]]),
        GaussianTransition([[0.9]], dynamics_noise=[[0.02]]),
    )
    return CouplingGraphBackend(
        graph,
        transitions,
        prior=unit_prior(graph),
        control_matrix=[[1.0], [0.0], [0.0]],
        partition=partition,
    )


def _selector(backend, info_block=(0, 1, 2)):
    return FFGEFESelector(
        backend, info_block=info_block, n_candidates=11, action_bounds=(-2.0, 2.0)
    )


def test_coupling_carries_efe_relevant_flag():
    # The declared-relevance seam: an edge can be flagged EFE-load-bearing; default off.
    flagged = Coupling(0, 1, [[1.0]], coupling_noise=[[0.1]], efe_relevant=True)
    assert flagged.efe_relevant is True
    plain = Coupling(0, 1, [[1.0]], coupling_noise=[[0.1]])
    assert plain.efe_relevant is False


def test_selector_rejects_partition_severing_efe_edge():
    # {0,2}/{1} cuts the flagged fast edge 0->1 — the gradient epistemic would die, so
    # building the EFE selector must raise rather than silently collapse it (ADR-018).
    backend = _backend(_hub_graph(), partition=[[0, 2], [1]])
    with pytest.raises(InadmissiblePartitionError, match="sever"):
        _selector(backend)


def test_selector_admits_methylation_cut():
    # {0,1}/{2} cuts only the *unflagged* slow edge 0->2 — gradient path intact, so the
    # cut is admissible (the natural chemotaxis {fast+CheA}/{slow} partition).
    backend = _backend(_hub_graph(), partition=[[0, 1], [2]])
    _selector(backend)  # must not raise


def test_selector_admits_full_joint():
    # The exact [[all]] carry severs nothing, so it is trivially admissible.
    _selector(_backend(_hub_graph()))


def test_backend_reports_severed_efe_edges():
    # The diagnostic the selector's policy reads: which EFE-relevant edges a cut severs.
    cut = _backend(_hub_graph(), partition=[[0, 2], [1]]).severed_efe_edges()
    assert {(e.parent, e.child) for e in cut} == {(0, 1)}
    assert _backend(_hub_graph(), partition=[[0, 1], [2]]).severed_efe_edges() == ()


def test_pure_filtering_partition_is_not_guarded():
    # The guard is EFE-only: a cutting partition constructs and *filters* fine (no
    # epistemic to protect), so pure tracking on any partition is never blocked.
    backend = _backend(_hub_graph(), partition=[[0, 2], [1]])
    prior = Belief(mean=np.zeros(3), cov=np.eye(3))
    belief = backend.infer_states(np.array([0.3, -0.1]), prior, np.array([0.0]))
    assert np.asarray(belief.mean).shape == (3,)


def test_agent_rejects_inadmissible_partition():
    # The Agent builds the FFG selector, so the guard reaches the user-facing path too.
    # It builds one only where the action moves the information term.
    sensor = CallableSensor([[1.0]], _sharp_near_zero, {"r0": jnp.array(0.1)})
    backend = _backend(_hub_graph(fast_sensor=sensor), partition=[[0, 2], [1]])
    with pytest.raises(InadmissiblePartitionError):
        Agent.from_backend(
            backend, ObservationGoal([0.5, 0.5], action_bounds=(-2.0, 2.0))
        )


def test_selector_still_selects_on_admissible_partition():
    # After admission, the selector actually works on the factored partition.
    backend = _backend(_hub_graph(), partition=[[0, 1], [2]])
    selector = _selector(backend)
    belief = Belief(mean=np.zeros(3), cov=np.eye(3))
    preference = Preference(target=[0.5, 0.5], precision=np.eye(2))
    action = selector.select(belief, preference)
    assert np.asarray(action).shape == (1,)


# The rollouts and searches the selector sits beside take the same target, and used
# to skip both checks: a severing partition or a block JAX clamps scored as PROVED.

_SEVERING = [[0, 2], [1]]  # cuts the flagged hub -> fast edge


def _actions():
    return FiniteActionSet([[-1.0], [0.0], [1.0]], version="v1")


def test_searches_reject_partition_severing_efe_edge():
    backend = _backend(_hub_graph(), partition=_SEVERING)
    for search in (EnumeratedEFESearch, ChunkedEFESearch):
        with pytest.raises(InadmissiblePartitionError, match="sever"):
            search.over_backend(backend, _actions(), info_block=range(3), horizon=2)


def test_rollouts_reject_partition_severing_efe_edge():
    backend = _backend(_hub_graph(), partition=_SEVERING)
    belief, policy = unit_prior(_hub_graph()), jnp.zeros((2, 1))
    preference = Preference(jnp.zeros(2))
    for rollout in (policy_efe_ffg, policy_efe_ffg_trace):
        with pytest.raises(InadmissiblePartitionError, match="sever"):
            rollout(backend, belief, policy, preference, info_block=range(3))


@pytest.mark.parametrize(
    ("info_block", "match"),
    [([], "empty"), ([0, 0], "repeats"), ([99], "outside"), ([-1], "outside")],
)
def test_info_block_must_name_distinct_joint_indices(info_block, match):
    backend = _backend(_hub_graph())
    with pytest.raises(ValueError, match=match):
        EnumeratedEFESearch.over_backend(
            backend, _actions(), info_block=info_block, horizon=2
        )
    with pytest.raises(ValueError, match=match):
        policy_efe_ffg(
            backend,
            unit_prior(_hub_graph()),
            jnp.zeros((2, 1)),
            Preference(jnp.zeros(2)),
            info_block=info_block,
        )
    with pytest.raises(ValueError, match=match):
        _selector(backend, info_block)
