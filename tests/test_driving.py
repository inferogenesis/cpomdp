import inspect

import jax
import numpy as np
import pytest

from cpomdp._lab.harness import (
    SEVERED_CONTROL_LOOP,
    DrivenRun,
    ExogenousActionSequence,
    ScoredAgent,
    World,
    drive,
)
from cpomdp.types import Belief, LinearGaussianModel

DT = 0.1
CONTROL = np.array([[0.0], [DT]])
SENSOR = np.array([[1.0, 0.0]])


def _model(*, damping: float = 1.0) -> LinearGaussianModel:
    return LinearGaussianModel(
        dynamics_matrix=[[1.0, DT], [0.0, damping]],
        observation_matrix=SENSOR,
        dynamics_noise=[[1e-4, 0.0], [0.0, 1e-4]],
        observation_noise=[[1e-2]],
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        control_matrix=CONTROL,
    )


def _sequence(steps: int = 5) -> ExogenousActionSequence:
    return ExogenousActionSequence(
        [[1.0], [-1.0], [0.5], [0.0], [-0.5]][:steps], version="drive-v1"
    )


def _agents(*names: str) -> dict[str, ScoredAgent]:
    return {
        name: ScoredAgent(_model(damping=0.9 - 0.1 * i)) for i, name in enumerate(names)
    }


# --- the declared sequence ----------------------------------------------------------


def test_a_sequence_reports_its_length_and_action_dimension():
    sequence = _sequence()
    assert sequence.horizon == 5
    assert sequence.action_dim == 1


def test_a_sequence_needs_a_version():
    with pytest.raises(ValueError, match="declared and versioned"):
        ExogenousActionSequence([[1.0]], version="")


def test_an_empty_sequence_does_not_construct():
    with pytest.raises(ValueError, match="at least one action"):
        ExogenousActionSequence(np.zeros((0, 1)), version="drive-v1")


def test_a_one_dimensional_sequence_does_not_construct():
    with pytest.raises(ValueError, match="2-D"):
        ExogenousActionSequence([1.0, -1.0], version="drive-v1")


# --- one world, one observation stream, every agent on it ---------------------------


def test_the_run_records_one_observation_and_one_state_per_step():
    run = drive(World(_model()), _agents("a"), _sequence(), jax.random.PRNGKey(0))
    assert np.asarray(run.observations).shape == (5, 1)
    assert np.asarray(run.states).shape == (5, 2)


def test_the_world_advances_once_per_step_not_once_per_agent():
    one = drive(World(_model()), _agents("a"), _sequence(), jax.random.PRNGKey(0))
    three = drive(
        World(_model()), _agents("a", "b", "c"), _sequence(), jax.random.PRNGKey(0)
    )
    assert np.array_equal(np.asarray(one.states), np.asarray(three.states))
    assert np.array_equal(np.asarray(one.observations), np.asarray(three.observations))


def test_each_agent_folded_exactly_the_recorded_observations():
    agents = _agents("a", "b")
    run = drive(World(_model()), agents, _sequence(), jax.random.PRNGKey(3))
    replayed = ScoredAgent(_model(damping=0.9))
    for observation, action in zip(
        np.asarray(run.observations), np.asarray(_sequence().actions), strict=True
    ):
        replayed.driven_step(observation, action)
    assert np.array_equal(
        np.asarray(run.beliefs["a"][-1].mean), np.asarray(replayed.belief.mean)
    )
    assert np.array_equal(
        np.asarray(run.beliefs["a"][-1].cov), np.asarray(replayed.belief.cov)
    )


def test_every_agent_gets_one_belief_per_step():
    run = drive(World(_model()), _agents("a", "b"), _sequence(), jax.random.PRNGKey(0))
    assert set(run.beliefs) == {"a", "b"}
    assert all(len(trajectory) == 5 for trajectory in run.beliefs.values())


def test_agents_with_different_models_end_apart():
    run = drive(World(_model()), _agents("a", "b"), _sequence(), jax.random.PRNGKey(0))
    assert not np.allclose(
        np.asarray(run.beliefs["a"][-1].mean), np.asarray(run.beliefs["b"][-1].mean)
    )


def test_the_final_belief_is_the_last_of_each_trajectory():
    run = drive(World(_model()), _agents("a", "b"), _sequence(), jax.random.PRNGKey(0))
    for name, trajectory in run.beliefs.items():
        assert run.final_beliefs[name] is trajectory[-1]


def test_the_same_key_reproduces_the_run():
    left = drive(World(_model()), _agents("a"), _sequence(), jax.random.PRNGKey(11))
    right = drive(World(_model()), _agents("a"), _sequence(), jax.random.PRNGKey(11))
    assert np.array_equal(np.asarray(left.observations), np.asarray(right.observations))


def test_a_different_key_moves_the_run():
    left = drive(World(_model()), _agents("a"), _sequence(), jax.random.PRNGKey(11))
    right = drive(World(_model()), _agents("a"), _sequence(), jax.random.PRNGKey(12))
    assert not np.allclose(
        np.asarray(left.observations), np.asarray(right.observations)
    )


def test_a_run_with_no_agents_still_produces_the_trajectory():
    run = drive(World(_model()), {}, _sequence(), jax.random.PRNGKey(0))
    assert np.asarray(run.states).shape == (5, 2)
    assert run.beliefs == {}


def test_an_action_dimension_the_world_cannot_take_does_not_run():
    wide = ExogenousActionSequence([[1.0, 0.0]], version="drive-v1")
    with pytest.raises(ValueError, match="action dimension"):
        drive(World(_model()), _agents("a"), wide, jax.random.PRNGKey(0))


# --- the declaration travels with the numbers ---------------------------------------


def test_the_run_carries_the_version_of_the_sequence_that_drove_it():
    run = drive(World(_model()), _agents("a"), _sequence(), jax.random.PRNGKey(0))
    assert run.action_sequence_version == "drive-v1"


def test_the_run_carries_the_severed_control_loop_declaration():
    run = drive(World(_model()), _agents("a"), _sequence(), jax.random.PRNGKey(0))
    assert run.control_loop is SEVERED_CONTROL_LOOP


def test_the_declaration_states_both_what_was_chosen_and_what_it_costs():
    assert "cuts the control loop" in SEVERED_CONTROL_LOOP.statement
    assert "inference gap" in SEVERED_CONTROL_LOOP.contested_by


def test_the_declaration_cannot_be_left_off_a_run():
    control_loop = inspect.signature(DrivenRun).parameters["control_loop"]
    assert control_loop.default is inspect.Parameter.empty
