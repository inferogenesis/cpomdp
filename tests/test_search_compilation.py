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
- `TestTransforms`: `evaluate` composes with `jax.jit` and `jax.vmap` and returns the
  whole result, certificate included, on both searches.
- `TestCompiledLoop`: a receding-horizon loop compiles under `jax.lax.scan` on either
  search, and a miscount still fails when the chunked count is only known at run time.
"""

import jax
import jax.numpy as jnp
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
    IncompleteEnumerationError,
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


def _assert_same_result(transformed, eager):
    """Every field equal, the certificate included."""
    assert type(transformed) is type(eager)
    np.testing.assert_array_equal(transformed.best_policy, eager.best_policy)
    assert float(transformed.best_g) == float(eager.best_g)
    assert int(transformed.best_index) == int(eager.best_index)
    assert transformed.certificate == eager.certificate


class TestTransforms:
    @pytest.mark.parametrize("case", SEARCHES)
    def test_a_jitted_evaluate_returns_the_whole_result(self, case):
        search = SEARCHES[case][0]()
        _assert_same_result(jax.jit(search.evaluate)(*FIRST), search.evaluate(*FIRST))

    @pytest.mark.parametrize("case", SEARCHES)
    def test_evaluate_maps_over_beliefs(self, case):
        # The belief is built inside the mapped function, because `Belief` refuses a
        # batched mean at construction.
        search = SEARCHES[case][0]()
        first_belief, preference = FIRST
        means = jnp.stack([first_belief.mean, SECOND[0].mean])
        batched = jax.vmap(
            lambda mean: search.evaluate(
                Belief(mean=mean, cov=first_belief.cov), preference
            )
        )(means)
        for row, mean in enumerate(means):
            eager = search.evaluate(Belief(mean=mean, cov=first_belief.cov), preference)
            np.testing.assert_array_equal(batched.best_policy[row], eager.best_policy)
            assert int(batched.best_index[row]) == int(eager.best_index)
            assert batched.certificate == eager.certificate


class TestCompiledLoop:
    @pytest.mark.parametrize("case", SEARCHES)
    def test_a_receding_horizon_loop_compiles_under_scan(self, case):
        selector = RecedingHorizonSelector(SEARCHES[case][0]())
        first_belief, preference = FIRST

        def step(mean, _):
            belief = Belief(mean=mean, cov=first_belief.cov)
            action = selector.select(belief, preference)
            return mean.at[0].add(0.5 * action[0]), action

        _, compiled = jax.jit(lambda mean: jax.lax.scan(step, mean, length=3))(
            first_belief.mean
        )
        mean, eager = first_belief.mean, []
        for _ in range(3):
            mean, action = step(mean, None)
            eager.append(action)
        np.testing.assert_array_equal(compiled, jnp.stack(eager))
        certificate = selector.certificate
        assert certificate is not None
        assert certificate.complete

    def test_a_chunked_miscount_fails_eagerly_and_compiled(self, monkeypatch):
        # A masking bug that drops one lane. Eagerly the count is checked on the host.
        # Compiled, it is checked when the program runs, and the run fails.
        real = ChunkedEfeSearch._scan_blocks

        def one_short(search, *args, **kwargs):
            argmin, best_policy, extra = real(search, *args, **kwargs)
            return argmin._replace(visited=argmin.visited - 1), best_policy, extra

        monkeypatch.setattr(ChunkedEfeSearch, "_scan_blocks", one_short)
        search = SEARCHES["chunked flat"][0]()
        with pytest.raises(
            IncompleteEnumerationError, match="visited 26 of an expected"
        ):
            search.evaluate(*FIRST)
        with pytest.raises(Exception, match="visited 26 of an expected"):
            jax.block_until_ready(jax.jit(search.evaluate)(*FIRST).best_g)
