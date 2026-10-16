"""Which candidate a grid selector returns when the scores do not single one out.

Two cases. When candidates tie exactly, the action does not reach anything the scores
read, and the selector should do the least: the candidate nearest zero. It used to take
the first, the lower bound, which on a double integrator at horizon 1 pushed at full
effort away from any goal. When no candidate scores a finite G, there is no argmin at
all. The selectors used to return the first candidate, which looks decisive. They
return NaN now, and the Agent refuses it.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _support import unit_prior
from cpomdp.agent import Agent
from cpomdp.backends.coupling import CouplingGraphBackend
from cpomdp.ffg.factors.linear_gaussian import GaussianTransition
from cpomdp.ffg.graph import CouplingGraph
from cpomdp.observation import CallableSensor, FixedSensor
from cpomdp.selection import EFESelector, FfgEfeSelector, Preference
from cpomdp.types import Belief, LinearGaussianModel

# A push moves the velocity, and position is what the sensor reads. At horizon 1 the
# action reaches neither the predicted reading nor its noise, so every candidate ties.
DOUBLE_INTEGRATOR = [[1.0, 0.1], [0.0, 1.0]]
PUSH_ON_VELOCITY = [[0.0], [0.1]]
READ_POSITION = [[1.0, 0.0]]


def _sharp_near_one(x, params):
    """R(x) sharpest at position 1, a function of what the sensor already reads."""
    return jnp.array([[0.01 + (x[0] - 1.0) ** 2]])


def _goes_negative(x, params):
    """R(x) = 1 - x0², a covariance only for |x0| < 1."""
    return jnp.array([[1.0 - x[0] ** 2]])


def _double_integrator(noise_fn=_sharp_near_one):
    return LinearGaussianModel(
        dynamics_matrix=DOUBLE_INTEGRATOR,
        observation_matrix=READ_POSITION,
        dynamics_noise=np.eye(2) * 1e-6,
        prior=Belief(mean=[0.0, 0.0], cov=np.eye(2)),
        control_matrix=PUSH_ON_VELOCITY,
        observation_model=CallableSensor(READ_POSITION, noise_fn, ()),
    )


def _graph_double_integrator(observation):
    graph = CouplingGraph(dims=(2,), couplings=(), observation_models={0: observation})
    return CouplingGraphBackend(
        graph,
        (GaussianTransition(DOUBLE_INTEGRATOR, dynamics_noise=np.eye(2) * 1e-6),),
        prior=unit_prior(graph),
        control_matrix=PUSH_ON_VELOCITY,
    )


class TestATieGoesToTheCandidateNearestZero:
    def test_the_flat_selector_rests_when_the_push_cannot_reach_the_reading(self):
        model = _double_integrator()
        selector = EFESelector(model, action_bounds=(-2.0, 2.0))
        scores = np.asarray(selector._scores(model.prior, Preference([1.0])))
        assert np.all(scores == scores[0])  # the premise: an exact tie
        np.testing.assert_allclose(
            selector.select(model.prior, Preference([1.0])), [0.0], atol=1e-12
        )

    def test_without_zero_on_the_grid_one_of_the_two_nearest_wins(self):
        model = _double_integrator()
        selector = EFESelector(model, action_bounds=(-1.0, 1.0), n_candidates=4)
        chosen = float(selector.select(model.prior, Preference([1.0]))[0])
        assert abs(chosen) == pytest.approx(1.0 / 3.0)

    def test_the_graph_selector_rests_too(self):
        backend = _graph_double_integrator(
            FixedSensor(READ_POSITION, observation_noise=[[0.1]])
        )
        selector = FfgEfeSelector(backend, action_bounds=(-2.0, 2.0))
        np.testing.assert_allclose(
            selector.select(backend.model.prior, Preference([1.0])), [0.0], atol=1e-12
        )

    def test_a_score_that_differs_still_wins_wherever_it_sits(self):
        # A single integrator reads the push at once, so the scores differ and the
        # tie-break never runs. The goal at the upper bound pulls the choice there.
        model = LinearGaussianModel(
            dynamics_matrix=[[1.0]],
            observation_matrix=[[1.0]],
            dynamics_noise=[[0.05]],
            prior=Belief(mean=[0.0], cov=[[0.5]]),
            control_matrix=[[1.0]],
            observation_model=CallableSensor([[1.0]], _sharp_near_one, ()),
        )
        selector = EFESelector(model, action_bounds=(-1.0, 1.0), n_candidates=5)
        np.testing.assert_array_equal(
            selector.select(model.prior, Preference([1.0], [[100.0]])), [1.0]
        )


# R(x) = 1 - x0² at x0 = 2 is no covariance anywhere on the grid, so every G is NaN.
_PAST_THE_EDGE = Belief(mean=[2.0, 0.0], cov=np.eye(2) * 0.1)


class TestNoFiniteScoreGivesNoAction:
    def test_the_flat_selector_returns_nan(self):
        model = _double_integrator(_goes_negative)
        selector = EFESelector(model, action_bounds=(-1.0, 1.0), n_candidates=3)
        scores = selector._scores(_PAST_THE_EDGE, Preference([0.0]))
        assert bool(jnp.isnan(scores).all())  # the premise
        assert bool(jnp.isnan(selector.select(_PAST_THE_EDGE, Preference([0.0]))).all())

    def test_the_graph_selector_returns_nan(self):
        backend = _graph_double_integrator(
            CallableSensor(READ_POSITION, _goes_negative, None)
        )
        selector = FfgEfeSelector(backend, action_bounds=(-1.0, 1.0), n_candidates=3)
        assert bool(jnp.isnan(selector.select(_PAST_THE_EDGE, Preference([0.0]))).all())

    def test_one_finite_score_is_enough(self):
        # At x0 = 1.2 only the push toward zero keeps R(x) a covariance at μ⁺.
        model = LinearGaussianModel(
            dynamics_matrix=[[1.0]],
            observation_matrix=[[1.0]],
            dynamics_noise=[[1e-3]],
            prior=Belief(mean=[0.0], cov=[[0.5]]),
            control_matrix=[[1.0]],
            observation_model=CallableSensor([[1.0]], _goes_negative, ()),
        )
        selector = EFESelector(model, action_bounds=(-0.5, 0.5), n_candidates=3)
        near_edge = Belief(mean=[1.2], cov=[[0.01]])
        np.testing.assert_array_equal(
            selector.select(near_edge, Preference([0.0])), [-0.5]
        )

    def test_the_agent_refuses_an_action_that_is_not_a_number(self):
        model = _double_integrator(_goes_negative)
        selector = EFESelector(model, action_bounds=(-1.0, 1.0), n_candidates=3)
        agent = Agent(model, Preference([0.0]), selector=selector)
        agent.belief = _PAST_THE_EDGE
        with pytest.raises(ValueError, match="no candidate scored a finite"):
            agent.sample_action()
        agent.belief = model.prior  # the refused action was never remembered
        agent.infer_states([0.0])
