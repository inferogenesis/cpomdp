"""A coupled graph backend's model is its interface, not its plant.

`CouplingGraphBackend.model` carries the sizes, prior, sensor and control the backend
reads, and leaves out the couplings. A flat consumer handed it would filter, plan or
regulate a plant without them, with nothing raised. Each model-taking constructor
refuses it and names the route that keeps the couplings.
"""

import dataclasses
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _support import unit_prior
from cpomdp import (
    Agent,
    CallableSensor,
    Coupling,
    CouplingGraph,
    CouplingGraphBackend,
    EFESelector,
    FixedSensor,
    GaussianTransition,
    KalmanBackend,
    LinearGaussianModel,
    LQRController,
    Preference,
    StateGoal,
)
from cpomdp.control import kalman_schedule, lqr_schedule
from cpomdp.diagnostics import probe_model
from cpomdp.efe import expected_free_energy, policy_efe, policy_efe_trace
from cpomdp.enumeration import (
    ChunkedEFESearch,
    EnumeratedEFESearch,
    FiniteActionSet,
    RecedingHorizonSelector,
)


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
    "expected_free_energy": lambda model: expected_free_energy(
        model, model.prior, jnp.zeros(1), Preference(jnp.zeros(model.n_observations))
    ),
    "policy_efe": lambda model: policy_efe(
        model,
        model.prior,
        jnp.zeros((2, 1)),
        Preference(jnp.zeros(model.n_observations)),
    ),
    "policy_efe_trace": lambda model: policy_efe_trace(
        model,
        model.prior,
        jnp.zeros((2, 1)),
        Preference(jnp.zeros(model.n_observations)),
    ),
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


def test_the_mark_survives_a_pytree_rebuild():
    # tree_map, device_put and a jit boundary all rebuild the model from its leaves.
    rebuilt = jax.tree.map(jnp.asarray, _backend(coupled=True).model)
    with pytest.raises(TypeError, match="couplings"):
        KalmanBackend(rebuilt)


def test_a_jitted_score_on_the_marked_model_is_refused_at_trace():
    model = _backend(coupled=True).model
    preference = Preference(jnp.zeros(model.n_observations))
    with pytest.raises(TypeError, match="couplings"):
        jax.jit(lambda m: expected_free_energy(m, m.prior, jnp.zeros(1), preference).g)(
            model
        )


def test_the_rxinfer_oracle_refuses_a_coupled_graphs_model():
    # Checked before Julia starts, so this runs without the rxinfer extra.
    from cpomdp.backends.rxinfer import RxInferBackend

    with pytest.raises(TypeError, match="couplings"):
        RxInferBackend(_backend(coupled=True).model)


def test_the_mark_survives_replacing_the_prior():
    # The model's own docstring moves a prior with dataclasses.replace.
    model = _backend(coupled=True).model
    moved = dataclasses.replace(model, prior=model.prior)
    with pytest.raises(TypeError, match="couplings"):
        KalmanBackend(moved)


def test_the_chain_backend_refuses_a_coupled_graphs_model():
    from cpomdp.ffg._chain import ChainBackend

    with pytest.raises(TypeError, match="couplings"):
        ChainBackend(_backend(coupled=True).model)


def _coupled_backend(coupling_noise, partition=None):
    graph = CouplingGraph(
        dims=(1, 1),
        couplings=(Coupling(0, 1, [[0.8]], coupling_noise=[[coupling_noise]]),),
        observation_models={
            0: CallableSensor([[1.0]], _sharp_near_zero, ()),
            1: FixedSensor([[1.0]], observation_noise=[[0.1]]),
        },
    )
    transitions = (
        GaussianTransition([[0.9]], dynamics_noise=[[0.1]]),
        GaussianTransition([[0.9]], dynamics_noise=[[0.1]]),
    )
    return CouplingGraphBackend(
        graph,
        transitions,
        prior=unit_prior(graph),
        control_matrix=[[1.0], [0.0]],
        partition=partition,
    )


def _sharp_near_zero(state, params):
    return jnp.array([[0.05 + state[0] ** 2]])


def _driver_over(backend):
    return RecedingHorizonSelector(
        EnumeratedEFESearch.over_backend(backend, _ACTIONS, horizon=1)
    )


def test_a_driver_over_a_graph_with_other_couplings_is_refused():
    # The interface models match. The couplings, which the score reads, do not.
    agent_graph, other_graph = _coupled_backend(0.05), _coupled_backend(0.5)
    with pytest.raises(ValueError, match="another"):
        Agent.from_backend(
            agent_graph, Preference(jnp.zeros(2)), selector=_driver_over(other_graph)
        )


def test_a_driver_over_a_graph_with_another_carry_partition_is_refused():
    # The partition cuts covariance in the predict step the score rolls through.
    agent_graph = _coupled_backend(0.05)
    other_graph = _coupled_backend(0.05, partition=[[0], [1]])
    with pytest.raises(ValueError, match="another"):
        Agent.from_backend(
            agent_graph, Preference(jnp.zeros(2)), selector=_driver_over(other_graph)
        )


def test_a_carry_partition_written_in_another_order_is_the_same_plant():
    agent_graph = _coupled_backend(0.05, partition=[[0], [1]])
    selector = _driver_over(_coupled_backend(0.05, partition=[[1], [0]]))
    agent = Agent.from_backend(agent_graph, Preference(jnp.zeros(2)), selector=selector)
    assert agent.selector is selector


def test_a_graph_backend_reads_back_its_carry_partition():
    assert _coupled_backend(0.05).partition == ((0, 1),)
    assert _coupled_backend(0.05, partition=[[1], [0]]).partition == ((1,), (0,))


def test_a_driver_over_an_equal_copy_of_the_graph_is_taken():
    agent_graph, same_graph = _coupled_backend(0.05), _coupled_backend(0.05)
    selector = _driver_over(same_graph)
    agent = Agent.from_backend(agent_graph, Preference(jnp.zeros(2)), selector=selector)
    assert agent.selector is selector


def test_a_custom_selector_is_not_checked_whatever_it_calls_model():
    # A misspecification ablation plans under a nominal plant on purpose (ADR-080).
    true_model = _backend(coupled=False).to_flat_model()
    nominal = dataclasses.replace(true_model, dynamics_noise=np.eye(2) * 0.5)

    class NominalPlanner:
        model = nominal

        def select(self, belief, preference):
            return jnp.zeros(1)

    Agent(true_model, Preference(jnp.zeros(2)), selector=NominalPlanner())


def _coupled_backend_with_leaf_noise(leaf_noise):
    graph = CouplingGraph(
        dims=(1, 1),
        couplings=(Coupling(0, 1, [[0.8]], coupling_noise=[[0.05]]),),
        observation_models={
            0: CallableSensor([[1.0]], _sharp_near_zero, ()),
            1: FixedSensor([[1.0]], observation_noise=[[leaf_noise]]),
        },
    )
    transitions = (
        GaussianTransition([[0.9]], dynamics_noise=[[0.1]]),
        GaussianTransition([[0.9]], dynamics_noise=[[0.1]]),
    )
    return CouplingGraphBackend(
        graph, transitions, prior=unit_prior(graph), control_matrix=[[1.0], [0.0]]
    )


def test_a_driver_over_a_graph_with_another_leaf_sensor_is_refused():
    # The interface model stacks every node's sensor, so a leaf's noise is compared.
    agent_graph = _coupled_backend_with_leaf_noise(0.1)
    other_graph = _coupled_backend_with_leaf_noise(0.5)
    with pytest.raises(ValueError, match="another"):
        Agent.from_backend(
            agent_graph, Preference(jnp.zeros(2)), selector=_driver_over(other_graph)
        )


class _OpaqueGraph:
    """A graph backend the Agent cannot see the couplings of."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_an_opaque_graph_backend_is_matched_by_identity():
    # Its couplings cannot be read, so only the same backend object passes.
    first = cast("CouplingGraphBackend", _OpaqueGraph(_coupled_backend(0.05)))
    second = cast("CouplingGraphBackend", _OpaqueGraph(_coupled_backend(0.05)))
    selector = _driver_over(first)
    assert (
        Agent.from_backend(first, Preference(jnp.zeros(2)), selector=selector).selector
        is selector
    )
    with pytest.raises(ValueError, match="same graph backend"):
        Agent.from_backend(second, Preference(jnp.zeros(2)), selector=selector)


def test_a_flat_selector_is_refused_on_a_coupled_graph_agent():
    # It would score the agent's plant without the edge the agent filters with.
    agent_graph = _coupled_backend(0.05)
    flat = LinearGaussianModel(
        dynamics_matrix=agent_graph.model.dynamics_matrix,
        observation_matrix=agent_graph.model.observation_matrix,
        dynamics_noise=agent_graph.model.dynamics_noise,
        prior=agent_graph.model.prior,
        control_matrix=agent_graph.model.control_matrix,
        observation_model=agent_graph.model.observation_model,
    )
    selector = EFESelector(flat, action_bounds=(-1.0, 1.0))
    with pytest.raises(ValueError, match="another"):
        Agent.from_backend(agent_graph, Preference(jnp.zeros(2)), selector=selector)


def test_the_research_wrappers_refuse_a_coupled_graphs_model():
    # Each rebuilds the model field by field, which would drop the mark.
    from cpomdp._lab.degraded import WrongFixedRBackend
    from cpomdp._lab.harness import World

    model = _backend(coupled=True).model
    with pytest.raises(TypeError, match="couplings"):
        WrongFixedRBackend(model, magnitude=0.1)
    with pytest.raises(TypeError, match="couplings"):
        World(model)


def test_a_driver_over_a_graph_with_another_efe_flag_is_refused():
    # efe_relevant changes what the score aims at, so it is part of what is compared.
    def flagged(relevant):
        graph = CouplingGraph(
            dims=(1, 1),
            couplings=(
                Coupling(0, 1, [[0.8]], coupling_noise=[[0.05]], efe_relevant=relevant),
            ),
            observation_models={
                0: CallableSensor([[1.0]], _sharp_near_zero, ()),
                1: FixedSensor([[1.0]], observation_noise=[[0.1]]),
            },
        )
        transitions = (
            GaussianTransition([[0.9]], dynamics_noise=[[0.1]]),
            GaussianTransition([[0.9]], dynamics_noise=[[0.1]]),
        )
        return CouplingGraphBackend(
            graph, transitions, prior=unit_prior(graph), control_matrix=[[1.0], [0.0]]
        )

    with pytest.raises(ValueError, match="another"):
        Agent.from_backend(
            flagged(False),
            Preference(jnp.zeros(2)),
            selector=_driver_over(flagged(True)),
        )


def test_a_scored_agent_refuses_a_coupled_graph_backend():
    from cpomdp._lab.harness import ScoredAgent

    with pytest.raises(TypeError, match="couplings"):
        ScoredAgent.from_backend(_backend(coupled=True))


@pytest.mark.parametrize("rule_name", ["exact", "frozen_gain", "wrong_r", "diagonal"])
def test_every_inference_cell_refuses_a_coupled_graphs_model(rule_name):
    from cpomdp._lab.constructors import InferenceKind, InferenceRule

    kind = {
        "exact": InferenceKind.EXACT,
        "frozen_gain": InferenceKind.FROZEN_GAIN,
        "wrong_r": InferenceKind.WRONG_FIXED_R,
        "diagonal": InferenceKind.DIAGONAL_COVARIANCE_ONLY,
    }[rule_name]
    magnitude = 0.1 if kind is InferenceKind.WRONG_FIXED_R else 0.0
    rule = InferenceRule(name=rule_name, kind=kind, magnitude=magnitude)
    with pytest.raises(TypeError, match="couplings"):
        rule.build(_backend(coupled=True).model)
