"""FFG-aware EFE action selector (issue #26, Phase B4).

The selector vmaps the FFG EFE step over the candidate action grid and argmins. Its
anchor: on a single-node model (no structural couplings) with the whole-state info
block, it
must pick the same action as the existing ``EFESelector`` on the equivalent flat model.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _support import unit_prior
from cpomdp.agent import Agent
from cpomdp.backends.coupling import CouplingGraphBackend
from cpomdp.efe import _ffg_efe_step
from cpomdp.ffg.factors.linear_gaussian import GaussianTransition
from cpomdp.ffg.graph import Coupling, CouplingGraph
from cpomdp.observation import CallableSensor, FixedSensor
from cpomdp.selection import (
    EFESelector,
    FfgEfeSelector,
    ObservationGoal,
    Preference,
)
from cpomdp.types import Belief, LinearGaussianModel


def _rx_noise(x, params):
    """Node-local R(x) = R0·(1 + gain·xᵀx) — sharpest as the node nears 0."""
    return params["R0"] * (1.0 + params["gain"] * jnp.dot(x, x))


def _decoupled_noise(s, params):
    """Two channels: a fixed-R goal channel and a state-dependent info channel.

    ``diag([R_goal, R0·(1 + gain·sᵀs)])`` — the goal channel's noise is constant (it
    carries the pragmatic pull), the info channel sharpens near ``s = 0`` (it carries
    the action-dependent epistemic). The block-diagonal keeps them independent.
    """
    r_info = params["R0"] * (1.0 + params["gain"] * jnp.dot(s, s))
    return jnp.diag(jnp.stack([jnp.asarray(params["R_goal"], dtype=float), r_info]))


def test_ffg_selector_reduces_to_efe_selector_single_node():
    # One node, no couplings, whole-state info block: the FFG selector must pick the
    # same
    # grid action as EFESelector on the equivalent flat LinearGaussianModel.
    dynamics = np.array([[1.0, 0.1], [0.0, 1.0]])  # A
    dynamics_noise = np.array([[0.1, 0.0], [0.0, 0.1]])  # Q
    observation_matrix = np.array([[1.0, 0.0]])  # C
    observation_noise = np.array([[0.5]])  # R
    control = np.array([[1.0], [0.0]])  # B — drives observed state[0], so G is non-flat

    graph = CouplingGraph(
        dims=(2,),
        couplings=(),
        observation_models={
            0: FixedSensor(observation_matrix, observation_noise=observation_noise)
        },
    )
    backend = CouplingGraphBackend(
        graph,
        (GaussianTransition(dynamics, dynamics_noise=dynamics_noise),),
        prior=unit_prior(graph),
        control_matrix=control,
    )
    model = LinearGaussianModel(
        dynamics_matrix=dynamics,
        observation_matrix=observation_matrix,
        dynamics_noise=dynamics_noise,
        observation_noise=observation_noise,
        prior=Belief(mean=np.zeros(2), cov=np.eye(2)),
        control_matrix=control,
    )

    belief = Belief(mean=[0.3, -0.2], cov=[[0.7, 0.1], [0.1, 0.4]])
    preference = Preference(goal=[1.0], precision=[[2.0]])
    bounds, k = (-2.0, 2.0), 21

    ffg = FfgEfeSelector(
        backend, info_block=range(2), n_candidates=k, action_bounds=bounds
    )
    efe = EFESelector(model, n_candidates=k, action_bounds=bounds)

    np.testing.assert_allclose(
        np.asarray(ffg.select(belief, preference)),
        np.asarray(efe.select(belief, preference)),
        atol=1e-7,
    )


def _two_node_backend():
    # CheA(0) -> CheY(1); observe CheY; the action drives CheA (the root).
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


@pytest.mark.parametrize(
    ("control_matrix", "match"),
    [
        (None, "control matrix"),
        # The search named must be the one that takes a backend.
        ([[1.0, 0.0], [0.0, 1.0]], r"p=1\).*EnumeratedEfeSearch\.over_backend"),
    ],
)
def test_selector_refuses_a_backend_it_cannot_search(control_matrix, match):
    # Unrefused, a control-free backend returned the grid's first action, and p = 2
    # constructed and failed at the first select.
    template = _two_node_backend()
    backend = CouplingGraphBackend(
        template.graph,
        template.transitions,
        prior=unit_prior(template.graph),
        control_matrix=control_matrix,
    )
    with pytest.raises(ValueError, match=match):
        FfgEfeSelector(
            backend, info_block=range(2), n_candidates=5, action_bounds=(-1.0, 1.0)
        )


def test_agent_on_ffg_backend_acts_via_ffg_selector():
    # B5 (issue #26): a user drives a branching FFG as an Agent. info_node aims the
    # epistemic at a chosen node; the Agent routes to the FFG EFE selector, and the
    # perceive -> act cycle completes.
    backend = _two_node_backend()
    agent = Agent.from_backend(
        backend,
        ObservationGoal([0.5], action_bounds=(-2.0, 2.0), info_node=0),  # info: CheA
    )
    agent.infer_states([0.3])  # observe CheY
    action = agent.sample_action()

    assert np.asarray(action).shape == (1,)
    assert isinstance(agent.selector, FfgEfeSelector)


def test_the_graph_selector_prints_its_horizon_and_target():
    agent = Agent.from_backend(
        _two_node_backend(),
        ObservationGoal([0.5], action_bounds=(-2.0, 2.0), info_node=0),
    )
    assert repr(agent.selector) == (
        "FfgEfeSelector(horizon=1, n_candidates=21, action_bounds=(-2.0, 2.0), "
        "info_block=[0])"
    )


def test_agent_ffg_whole_state_when_info_target_none():
    # info_node=None on the FFG backend still routes to the FFG selector, aimed at the
    # whole state (the default epistemic).
    backend = _two_node_backend()
    agent = Agent.from_backend(
        backend,
        ObservationGoal([0.5], action_bounds=(-2.0, 2.0)),  # no info_node
    )
    assert isinstance(agent.selector, FfgEfeSelector)
    assert tuple(agent.selector._target) == (0, 1)  # whole joint state


def test_info_target_on_flat_backend_raises():
    # info_node aims at an FFG node; it is meaningless without a branching backend.
    model = LinearGaussianModel(
        dynamics_matrix=[[1.0]],
        observation_matrix=[[1.0]],
        dynamics_noise=[[0.1]],
        observation_noise=[[0.2]],
        prior=Belief(mean=[0.0], cov=[[1.0]]),
        control_matrix=[[1.0]],
    )
    with np.testing.assert_raises(ValueError):
        Agent(model, ObservationGoal([0.5], action_bounds=(-2.0, 2.0), info_node=0))


def test_ffg_agent_matches_kalman_efe_agent_end_to_end():
    # B6 gate (issue #26): the whole FFG EFE Agent path — single node, info_node=None,
    # so the epistemic is whole-state — must reproduce a KalmanBackend + EFESelector
    # Agent step for step. The non-negotiable "v0.3 behaviour is not broken" check.
    dynamics = np.array([[1.0, 0.1], [0.0, 1.0]])  # A
    dynamics_noise = np.array([[0.1, 0.0], [0.0, 0.1]])  # Q
    observation_matrix = np.array([[1.0, 0.0]])  # C
    observation_noise = np.array([[0.5]])  # R
    control = np.array([[1.0], [0.0]])  # B — drives the observed state[0]
    prior = Belief(mean=np.zeros(2), cov=np.eye(2))
    model = LinearGaussianModel(
        dynamics_matrix=dynamics,
        observation_matrix=observation_matrix,
        dynamics_noise=dynamics_noise,
        observation_noise=observation_noise,
        prior=prior,
        control_matrix=control,
    )
    graph = CouplingGraph(
        dims=(2,),
        couplings=(),
        observation_models={
            0: FixedSensor(observation_matrix, observation_noise=observation_noise)
        },
    )
    backend = CouplingGraphBackend(
        graph,
        (GaussianTransition(dynamics, dynamics_noise=dynamics_noise),),
        prior=unit_prior(graph),
        control_matrix=control,
    )
    bounds, k = (-2.0, 2.0), 21

    ffg_agent = Agent.from_backend(
        backend, ObservationGoal([0.5], action_bounds=bounds, n_candidates=k)
    )
    ref_agent = Agent(
        model,
        ObservationGoal([0.5], action_bounds=bounds, n_candidates=k),
        selector=EFESelector(model, n_candidates=k, action_bounds=bounds),
    )

    for y in ([0.3], [-0.1], [0.2]):
        ffg_agent.infer_states(y)
        ref_agent.infer_states(y)
        np.testing.assert_allclose(
            np.asarray(ffg_agent.sample_action()),
            np.asarray(ref_agent.sample_action()),
            atol=1e-7,
        )


# --- Action-driven epistemic: state-dependent sensing in the selector (Phase 3) -


def _score_components(backend, belief, preference, candidates, info_block):
    """``(pragmatic, epistemic)`` per candidate — the selector's kernel, exposed.

    Mirrors ``FfgEfeSelector.select``'s per-candidate work (predict → ``R(μ⁺)`` →
    ``_ffg_efe_step``) but returns the two ``G`` components, so a test can compare the
    full-``G`` argmin against a pragmatic-only one at the *same* ``R(μ⁺)``.
    """
    observation_matrix, _ = backend.observation_model  # C (constant)

    def comp(action):
        predicted = backend.predicted_belief(belief, action)  # μ⁺, Σ⁺
        observation_noise = backend.observation_noise_at(predicted.mean)  # R(μ⁺)
        _, parts = _ffg_efe_step(
            predicted.mean,
            predicted.cov,
            observation_matrix,
            observation_noise,
            preference.goal,
            preference.precision,
            info_block,
        )
        return parts["pragmatic"], parts["epistemic"]

    return jax.vmap(comp)(candidates)


def _single_node_rx_backend(params):
    A = np.array([[1.0, 0.1], [0.0, 1.0]])  # drives state[0], which C observes
    Q = np.eye(2) * 0.1
    C = np.array([[1.0, 0.0]])
    B = np.array([[1.0], [0.0]])
    graph = CouplingGraph(
        dims=(2,),
        couplings=(),
        observation_models={0: CallableSensor(C, _rx_noise, params)},
    )
    backend = CouplingGraphBackend(
        graph,
        (GaussianTransition(A, dynamics_noise=Q),),
        prior=unit_prior(graph),
        control_matrix=B,
    )
    return backend, A, Q, C, B


class TestStateDependentSelection:
    """The dual effect in the selector: ``R(μ⁺)`` moves with the action, so the
    epistemic term is action-dependent and bends the choice (ADR-003 broke, ADR-019)."""

    def test_rx_epistemic_is_action_dependent(self):
        # R(μ⁺) moves with the action, so the epistemic term varies across candidates —
        # the dual effect. A fixed sensor's epistemic is flat across actions (ADR-003).
        params = {"R0": np.array([[0.5]]), "gain": 2.0}
        rx, A, Q, C, B = _single_node_rx_backend(params)
        belief = Belief(mean=[0.5, -0.2], cov=[[0.7, 0.1], [0.1, 0.4]])
        pref = Preference(goal=[1.0], precision=[[2.0]])
        cands = jnp.linspace(-2.0, 2.0, 9)[:, None]
        _, epi = _score_components(rx, belief, pref, cands, range(2))
        assert float(jnp.max(epi) - jnp.min(epi)) > 1e-6  # varies with action

        graph_fx = CouplingGraph(
            dims=(2,),
            couplings=(),
            observation_models={0: FixedSensor(C, observation_noise=params["R0"])},
        )
        fx = CouplingGraphBackend(
            graph_fx,
            (GaussianTransition(A, dynamics_noise=Q),),
            prior=unit_prior(graph_fx),
            control_matrix=B,
        )
        _, epi_fx = _score_components(fx, belief, pref, cands, range(2))
        np.testing.assert_allclose(np.asarray(epi_fx), float(epi_fx[0]), atol=1e-9)

    def test_single_node_rx_selector_matches_flat_efe_selector(self):
        # Trusted oracle: single node (no couplings, μ⁺ = μ⁻), whole-state info block.
        # The FFG R(x) selector must pick the same action as EFESelector on the
        # equivalent flat CallableSensor model, atol 1e-7 (the R(x) analogue of B4).
        params = {"R0": np.array([[0.5]]), "gain": 0.4}
        rx, A, Q, C, B = _single_node_rx_backend(params)
        model = LinearGaussianModel(
            dynamics_matrix=A,
            dynamics_noise=Q,
            prior=Belief(mean=np.zeros(2), cov=np.eye(2)),
            control_matrix=B,
            observation_model=CallableSensor(C, _rx_noise, params),
        )
        belief = Belief(mean=[0.3, -0.2], cov=[[0.7, 0.1], [0.1, 0.4]])
        pref = Preference(goal=[1.0], precision=[[2.0]])
        bounds, k = (-2.0, 2.0), 21
        ffg = FfgEfeSelector(
            rx, info_block=range(2), n_candidates=k, action_bounds=bounds
        )
        efe = EFESelector(model, n_candidates=k, action_bounds=bounds)
        np.testing.assert_allclose(
            np.asarray(ffg.select(belief, pref)),
            np.asarray(efe.select(belief, pref)),
            atol=1e-7,
        )

    def test_rx_selector_bends_the_choice(self):
        # The T-Maze-shaped decoupling (ADR-014 #3): a *fixed-R goal channel* carries
        # the pragmatic pull (toward the goal) and a *separate R(x) info channel* holds
        # the action-dependent epistemic (sharpest near 0). With the goal precision
        # zeroed on the info channel the two un-entangle, so the epistemic genuinely
        # moves the decision: the full-G argmin differs from the pragmatic-only one, and
        # the selector returns the full-G action. (A single entangled channel would not:
        # the pragmatic risk and the epistemic there share one covariance.)
        params = {"R_goal": 5.0, "R0": 0.15, "gain": 8.0}  # noisy goal, sharp info
        graph = CouplingGraph(
            dims=(1,),
            couplings=(),
            observation_models={
                0: CallableSensor([[1.0], [1.0]], _decoupled_noise, params)
            },  # both channels read s
        )
        rx = CouplingGraphBackend(
            graph,
            (GaussianTransition([[1.0]], dynamics_noise=[[0.1]]),),
            prior=unit_prior(graph),
            control_matrix=[[1.0]],
        )
        belief = Belief(mean=[0.0], cov=[[1.0]])
        # goal 3.0 on the fixed channel, ~0 weight on the info channel (decoupling).
        pref = Preference(goal=[3.0, 0.0], precision=[[0.15, 0.0], [0.0, 1e-4]])
        bounds, k = (-4.0, 4.0), 41
        cands = jnp.linspace(-4.0, 4.0, k)[:, None]
        prag, epi = _score_components(rx, belief, pref, cands, range(1))
        full_action = cands[int(jnp.argmin(prag - epi))]
        prag_action = cands[int(jnp.argmin(prag))]
        assert not np.isclose(
            float(full_action[0]), float(prag_action[0])
        )  # the epistemic genuinely moved the decision, away from the pragmatic goal
        sel = FfgEfeSelector(
            rx, info_block=range(1), n_candidates=k, action_bounds=bounds
        )
        np.testing.assert_allclose(
            np.asarray(sel.select(belief, pref)), np.asarray(full_action), atol=1e-7
        )
