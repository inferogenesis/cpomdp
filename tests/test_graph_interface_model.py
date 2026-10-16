"""A coupled graph backend's model is its interface, not its plant.

`CouplingGraphBackend.model` carries the sizes, prior, sensor and control the backend
reads, and leaves out the couplings. A flat consumer handed it would filter, plan or
regulate a plant without them, with nothing raised. Each model-taking constructor
refuses it and names the route that keeps the couplings.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _support import unit_prior
from cpomdp import (
    Agent,
    Coupling,
    CouplingGraph,
    CouplingGraphBackend,
    EFESelector,
    FixedSensor,
    GaussianTransition,
    KalmanBackend,
    LQRController,
    StateGoal,
)
from cpomdp.control import kalman_schedule, lqr_schedule
from cpomdp.diagnostics import probe_model
from cpomdp.enumeration import ChunkedEFESearch, EnumeratedEFESearch, FiniteActionSet


def _backend(*, coupled):
    """Two scalar nodes joined by an edge, or one two-state node with none."""
    if coupled:
        graph = CouplingGraph(
            dims=(1, 1),
            couplings=(Coupling(0, 1, [[0.8]], coupling_noise=[[0.05]]),),
            observation_models={
                0: FixedSensor([[1.0]], observation_noise=[[0.1]]),
                1: FixedSensor([[1.0]], observation_noise=[[0.1]]),
            },
        )
        transitions = (
            GaussianTransition([[0.9]], dynamics_noise=[[0.1]]),
            GaussianTransition([[0.9]], dynamics_noise=[[0.1]]),
        )
    else:
        graph = CouplingGraph(
            dims=(2,),
            couplings=(),
            observation_models={
                0: FixedSensor(np.eye(2), observation_noise=np.eye(2) * 0.1)
            },
        )
        transitions = (
            GaussianTransition(np.eye(2) * 0.9, dynamics_noise=np.eye(2) * 0.1),
        )
    return CouplingGraphBackend(
        graph, transitions, prior=unit_prior(graph), control_matrix=[[1.0], [0.0]]
    )


_ACTIONS = FiniteActionSet([[-1.0], [0.0], [1.0]], version="v1")

_CONSUMERS = {
    "KalmanBackend": lambda model: KalmanBackend(model),
    "Agent": lambda model: Agent(model, StateGoal(jnp.zeros(2))),
    "EFESelector": lambda model: EFESelector(model, action_bounds=(-1.0, 1.0)),
    "EnumeratedEFESearch": lambda model: EnumeratedEFESearch(
        model, _ACTIONS, horizon=1
    ),
    "ChunkedEFESearch": lambda model: ChunkedEFESearch(model, _ACTIONS, horizon=1),
    "LQRController": lambda model: LQRController(
        model, precision=np.eye(2), effort=np.eye(1)
    ),
    "lqr_schedule": lambda model: lqr_schedule(
        model, horizon=2, precision=np.eye(2), effort=np.eye(1)
    ),
    "kalman_schedule": lambda model: kalman_schedule(model, horizon=2),
    "probe_model": lambda model: probe_model(model, model.prior, [jnp.zeros(1)]),
}


@pytest.mark.parametrize("consumer", sorted(_CONSUMERS))
def test_a_coupled_graphs_model_is_refused_by_name(consumer):
    model = _backend(coupled=True).model
    with pytest.raises(TypeError, match="couplings"):
        _CONSUMERS[consumer](model)


@pytest.mark.parametrize("consumer", sorted(_CONSUMERS))
def test_an_uncoupled_graphs_model_is_its_plant(consumer):
    # With no couplings, nothing is left out.
    _CONSUMERS[consumer](_backend(coupled=False).model)


def test_the_flat_model_with_its_couplings_is_taken():
    KalmanBackend(_backend(coupled=True).to_flat_model())


def test_a_state_goal_on_a_coupled_graph_is_refused():
    # LQR on the interface model would regulate a plant without the couplings.
    with pytest.raises(TypeError, match="couplings"):
        Agent.from_backend(_backend(coupled=True), StateGoal(jnp.zeros(2)))
