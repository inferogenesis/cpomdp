"""Tests for the abstraction wall (backends/base.py).

The point of these is not any one backend's behaviour but *swappability*: two
unrelated classes satisfy InferenceBackend purely by shape, with no shared base.
If both pass isinstance, the wall holds and engines are interchangeable.
"""

import numpy as np

from _support import unit_prior
from cpomdp import diagnostics
from cpomdp.backends import base
from cpomdp.backends.base import GraphBackend, InferenceBackend
from cpomdp.backends.coupling import CouplingGraphBackend
from cpomdp.backends.kalman import KalmanBackend
from cpomdp.ffg.factors.linear_gaussian import GaussianObservation, GaussianTransition
from cpomdp.ffg.graph import CouplingGraph
from cpomdp.types import Belief, LinearGaussianModel


class _PassthroughBackend:
    """A trivial fake backend: returns the prior unchanged. It inherits nothing
    and knows nothing about InferenceBackend — it conforms by structure alone
    (a matching ``infer_states`` and the ``model`` it was built from)."""

    def __init__(self, model: LinearGaussianModel) -> None:
        self.model = model

    def infer_states(
        self,
        observation: np.ndarray,
        prior: Belief,
        action: np.ndarray | None = None,
    ) -> Belief:
        return prior


def _scalar_model():
    return LinearGaussianModel(
        dynamics_matrix=[[0.9]],
        observation_matrix=[[1.0]],
        dynamics_noise=[[0.5]],
        observation_noise=[[1.0]],
        prior=Belief(mean=[0.0], cov=[[10.0]]),
    )


def test_fake_backend_satisfies_protocol():
    assert isinstance(_PassthroughBackend(_scalar_model()), InferenceBackend)


def test_kalman_backend_satisfies_protocol():
    assert isinstance(KalmanBackend(_scalar_model()), InferenceBackend)


def _one_node_graph_backend():
    graph = CouplingGraph(
        root=0,
        dims=(1,),
        couplings=(),
        observations={0: GaussianObservation([[1.0]], observation_noise=[[0.5]])},
    )
    transitions = (GaussianTransition([[0.9]], dynamics_noise=[[0.1]]),)
    return CouplingGraphBackend(graph, transitions, prior=unit_prior(graph))


def test_a_graph_backend_satisfies_the_graph_protocol():
    assert isinstance(_one_node_graph_backend(), GraphBackend)


def test_a_flat_backend_is_not_a_graph_backend():
    # It has no nodes to aim an epistemic term at, so the Agent dispatches it flat.
    assert not isinstance(KalmanBackend(_scalar_model()), GraphBackend)


def test_one_protocol_describes_a_graph_backend():
    # `EfeBackend` named the protocol for one consumer, and `probe_model` restated part
    # of it as `ProbeBackend`.
    assert not hasattr(base, "EfeBackend")
    assert not hasattr(diagnostics, "ProbeBackend")
