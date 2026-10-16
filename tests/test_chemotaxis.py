"""The E. coli chemotaxis generative model as a native FFG (Phase 3).

Pins `examples/ffg/chemotaxis_model.py`: the receptor-driven CheA hub feeding a fast
CheY -> motor branch and a slow CheR/CheB methylation branch, with
per-node Ornstein-Uhlenbeck dynamics discretised on a single clock. Correctness is the
established bar: the built model's filtering matches an independent `KalmanBackend` on
the flattened joint (`to_flat_model`) at atol 1e-7. The timescale separation (slow
methylation persists, fast kinase relaxes) is a checked property, not an assumption.
"""

import numpy as np
import pytest
from chemotaxis_model import (
    CHEA,
    CHEB,
    CHEY,
    MOTOR_A,
    MOTOR_B,
    chemotaxis_ffg,
)

from _support import unit_prior
from cpomdp.backends.coupling import CouplingGraphBackend
from cpomdp.backends.kalman import KalmanBackend
from cpomdp.backends.rxinfer import RxInferBackend
from cpomdp.efe import _state_info_gain
from cpomdp.types import Belief


def test_topology_matches_the_chemotaxis_tree():
    # CheA(0) hub -> CheY(1) fast + CheB(2) slow; CheY -> motorA(3), motorB(4); the
    # methylation node and the two motors are observed (the RxInfer oracle's topology).
    graph, transitions = chemotaxis_ffg(dt=0.01)
    assert len(graph.dims) == 5
    assert len(transitions) == 5
    assert graph.root == CHEA
    edges = {(c.parent, c.child) for c in graph.couplings}
    assert edges == {(CHEA, CHEY), (CHEA, CHEB), (CHEY, MOTOR_A), (CHEY, MOTOR_B)}
    assert set(graph.observation_models) == {CHEB, MOTOR_A, MOTOR_B}


def test_timescale_separation_is_present():
    # The slow methylation node persists across a step (A ~ 1); the fast kinase/motor
    # nodes relax noticeably (A well below 1). This ~separation is what makes the
    # methylation carry cut nearly free.
    _graph, transitions = chemotaxis_ffg(dt=0.01)
    a = [float(np.asarray(t.dynamics_matrix)[0, 0]) for t in transitions]
    assert a[CHEB] > 0.99  # slow methylation: long memory
    assert a[CHEA] < 0.95  # fast kinase: relaxes each step
    assert a[CHEY] < 0.95


def test_filtering_matches_flat_kalman():
    # The built model's filtering equals an independent KalmanBackend on the flattened
    # joint (structural couplings as within-slice pseudo-observations), step for step.
    graph, transitions = chemotaxis_ffg(dt=0.01)
    backend = CouplingGraphBackend(graph, transitions, prior=unit_prior(graph))
    flat = KalmanBackend(backend.to_flat_model())

    rng = np.random.default_rng(0)
    prior = Belief(mean=np.zeros(5), cov=np.eye(5))
    obs_seq = [rng.standard_normal(backend.n_observations) for _ in range(6)]

    belief = flat_belief = prior
    for y in obs_seq:
        belief = backend.infer_states(y, belief)
        flat_belief = flat.infer_states(backend.flat_observation(y), flat_belief)
        np.testing.assert_allclose(
            np.asarray(belief.mean), np.asarray(flat_belief.mean), atol=1e-7
        )
        np.testing.assert_allclose(
            np.asarray(belief.cov), np.asarray(flat_belief.cov), atol=1e-7
        )


@pytest.mark.rxinfer
def test_filtering_matches_rxinfer_on_flat_model():
    # The external oracle: RxInfer (Julia) on the flattened joint must reproduce the
    # native backend's filtering, through completely separate machinery.
    graph, transitions = chemotaxis_ffg(dt=0.01)
    backend = CouplingGraphBackend(graph, transitions, prior=unit_prior(graph))
    oracle = RxInferBackend(backend.to_flat_model())

    rng = np.random.default_rng(1)
    prior = Belief(mean=np.zeros(5), cov=np.eye(5))
    obs_seq = [rng.standard_normal(backend.n_observations) for _ in range(4)]

    belief = oracle_belief = prior
    for y in obs_seq:
        belief = backend.infer_states(y, belief)
        oracle_belief = oracle.infer_states(backend.flat_observation(y), oracle_belief)
        np.testing.assert_allclose(
            np.asarray(belief.mean), np.asarray(oracle_belief.mean), atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(belief.cov), np.asarray(oracle_belief.cov), atol=1e-6
        )


def test_methylation_is_the_cheapest_node_to_sever():
    # The partition tie-in on the real model (ADR-016): the slow, weakly-driven
    # methylation node (CheB) is the *cheapest* single node to decouple at the carry —
    # severing it drops less cross-cluster correlation than severing any faster node.
    # (It is the cheapest cut, not a free one: the severed mass is a within-slice
    # covariance, which the fast/slow τ separation does not drive to zero.)
    graph, transitions = chemotaxis_ffg(dt=0.01)
    prior = Belief(mean=np.zeros(5), cov=np.eye(5))
    y = np.array([0.4, -0.2, 0.3])

    def sever_one(node):
        partition = [[n for n in range(5) if n != node], [node]]
        backend = CouplingGraphBackend(
            graph, transitions, prior=unit_prior(graph), partition=partition
        )
        return backend.partition_error(y, prior)

    exact = CouplingGraphBackend(
        graph, transitions, prior=unit_prior(graph)
    ).partition_error(y, prior)
    assert exact == pytest.approx(0.0)  # the full carry drops nothing

    cheb = sever_one(CHEB)
    assert cheb > 0.0
    for faster in (CHEA, CHEY, MOTOR_A, MOTOR_B):
        assert cheb < sever_one(faster)


def test_epistemic_can_target_the_hidden_hub():
    # #26 Phase C (ADR-014 #3): the factored epistemic can aim at a *hidden* latent.
    # CheA (the hub) is never directly observed — the sensors read the motors and CheB —
    # yet observing those downstream children reduces CheA's uncertainty through the
    # couplings. That info gain is positive, isolable, and distinct from the whole-state
    # info gain the flat observation-space EFE is limited to.
    graph, transitions = chemotaxis_ffg(dt=0.01)
    backend = CouplingGraphBackend(graph, transitions, prior=unit_prior(graph))
    observation_matrix, observation_noise = backend.observation_model
    prior = Belief(mean=np.zeros(5), cov=np.eye(5))
    sigma_pred = backend.predicted_belief(prior).cov  # Σ⁺

    def info_gain(info_block):
        return float(
            _state_info_gain(
                sigma_pred, observation_matrix, observation_noise, info_block
            )
        )

    assert CHEA not in backend.graph.observation_models  # the hub is a hidden latent
    chea = info_gain(backend.block(CHEA))
    whole = info_gain(range(5))
    assert chea > 0.0  # learnable through the couplings, though never observed directly
    assert not np.isclose(chea, whole)  # targeting restricts to that node's marginal
