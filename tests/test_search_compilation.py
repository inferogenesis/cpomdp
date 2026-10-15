"""Each exhaustive search compiles once, and the compiled scores are the eager ones.

A plan used to re-trace and recompile the H-step scan on every call. That fixed cost
landed on every plan and `cost_per_plan` did not count it. Each search now builds its
jitted scorer at construction and takes the belief and the preference as arguments, so
a second plan reuses the first one's program.

- `TestTracesOnce`: two plans from different beliefs and preferences trace the rollout
  step once, on both searches over both scorers, and through a driver.
- `TestEagerIdentity`: the compiled scores equal an eager `vmap` of the same scorer
  under `==`. The published numbers came from the eager path, so a moved bit is a
  finding.
"""

import jax
import numpy as np
import pytest

from _support import unit_prior
from cpomdp import efe
from cpomdp.backends.coupling import CouplingGraphBackend
from cpomdp.efe import policy_efe, policy_efe_ffg
from cpomdp.enumeration import (
    ChunkedEfeSearch,
    EnumeratedEfeSearch,
    FiniteActionSet,
    RecedingHorizonSelector,
)
from cpomdp.ffg.factors.linear_gaussian import (
    GaussianCoupling,
    GaussianObservation,
    GaussianTransition,
)
from cpomdp.ffg.graph import Coupling, CouplingGraph
from cpomdp.observation import CallableSensor
from cpomdp.selection import Preference
from cpomdp.types import Belief, LinearGaussianModel


def _model():
    """Two states, the first read through a noise that sharpens near +1."""
    sensor = CallableSensor(
        observation_matrix=[[1.0, 0.0]],
        noise_fn=lambda x, p: p["base"] + p["sharp"] * (x[0] - 1.0) ** 2 * np.eye(1),
        noise_params={"base": 0.05 * np.eye(1), "sharp": 4.0},
    )
    return LinearGaussianModel(
        dynamics_matrix=[[1.0, 0.1], [0.0, 1.0]],
        observation_matrix=[[1.0, 0.0]],
        dynamics_noise=[[0.1, 0.0], [0.0, 0.1]],
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        control_matrix=[[1.0], [0.5]],
        observation_model=sensor,
    )


def _backend():
    """Two scalar nodes, the second observed and driven by the first."""
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


def _actions():
    return FiniteActionSet([[-1.0], [0.0], [1.0]], version="v1")


FIRST = (
    Belief(mean=[0.3, -0.2], cov=[[0.7, 0.1], [0.1, 0.4]]),
    Preference(goal=[1.0], precision=[[2.0]]),
)
SECOND = (
    Belief(mean=[-0.5, 0.4], cov=[[0.5, 0.0], [0.0, 0.6]]),
    Preference(goal=[-0.5], precision=[[3.0]]),
)

# Each case: a search, and the per-step kernel its rollout traces through.
SEARCHES = {
    "front-loaded flat": (
        lambda: EnumeratedEfeSearch(_model(), _actions(), horizon=3),
        "_efe_step",
    ),
    "chunked flat": (
        lambda: ChunkedEfeSearch(_model(), _actions(), horizon=3, chunk=5),
        "_efe_step",
    ),
    "front-loaded graph": (
        lambda: EnumeratedEfeSearch.over_backend(
            _backend(), _actions(), info_block=[0], horizon=3
        ),
        "_ffg_efe_step",
    ),
    "chunked graph": (
        lambda: ChunkedEfeSearch.over_backend(
            _backend(), _actions(), info_block=[0], horizon=3, chunk=5
        ),
        "_ffg_efe_step",
    ),
}


def _count_traces(monkeypatch, kernel: str) -> list[None]:
    """Record one entry each time ``kernel`` runs in Python, which is once per trace."""
    traces: list[None] = []
    real = getattr(efe, kernel)

    def counted(*args, **kwargs):
        traces.append(None)
        return real(*args, **kwargs)

    monkeypatch.setattr(efe, kernel, counted)
    return traces


class TestTracesOnce:
    @pytest.mark.parametrize("case", SEARCHES)
    def test_a_second_plan_reuses_the_first_ones_program(self, monkeypatch, case):
        build, kernel = SEARCHES[case]
        search = build()
        traces = _count_traces(monkeypatch, kernel)
        search.evaluate(*FIRST)
        traced = len(traces)
        search.evaluate(*SECOND)
        assert traced > 0
        assert len(traces) == traced

    def test_a_driver_plans_on_the_compiled_search(self, monkeypatch):
        selector = RecedingHorizonSelector(SEARCHES["front-loaded flat"][0]())
        traces = _count_traces(monkeypatch, "_efe_step")
        selector.select(*FIRST)
        traced = len(traces)
        selector.select(*SECOND)
        assert len(traces) == traced


class TestEagerIdentity:
    @pytest.mark.parametrize("belief_and_preference", [FIRST, SECOND])
    def test_flat_scores_match_the_eager_vmap(self, belief_and_preference):
        belief, preference = belief_and_preference
        model = _model()
        search = EnumeratedEfeSearch(model, _actions(), horizon=3)
        eager = jax.vmap(
            lambda policy: policy_efe(model, belief, policy, preference)[0]
        )(search.policies)
        compiled = search.evaluate(belief, preference).g
        np.testing.assert_array_equal(np.asarray(compiled), np.asarray(eager))

    @pytest.mark.parametrize("belief_and_preference", [FIRST, SECOND])
    def test_graph_scores_match_the_eager_vmap(self, belief_and_preference):
        belief, preference = belief_and_preference
        backend = _backend()
        search = EnumeratedEfeSearch.over_backend(
            backend, _actions(), info_block=[0], horizon=3
        )
        eager = jax.vmap(
            lambda policy: policy_efe_ffg(
                backend, belief, policy, preference, info_block=[0]
            )[0]
        )(search.policies)
        compiled = search.evaluate(belief, preference).g
        np.testing.assert_array_equal(np.asarray(compiled), np.asarray(eager))
