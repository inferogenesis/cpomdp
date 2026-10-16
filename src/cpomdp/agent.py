"""The ``Agent`` façade: a stateful perceive/act loop over a ``LinearGaussianModel``."""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float64
from numpy.typing import ArrayLike

from cpomdp._validation import (
    concrete,
    equilibrium_drift,
    state_dependent,
)
from cpomdp.backends.base import GraphBackend, InferenceBackend
from cpomdp.backends.coupling import CouplingGraphBackend
from cpomdp.backends.kalman import KalmanBackend
from cpomdp.control import LQRController
from cpomdp.enumeration import OpenLoopSelector, RecedingHorizonSelector
from cpomdp.selection import (
    ActionSelector,
    EFESelector,
    FFGEFESelector,
    LQRSelector,
    ObservationGoal,
    Preference,
    StateGoal,
)
from cpomdp.types import (
    Belief,
    LinearGaussianModel,
    is_graph_interface,
    refuse_graph_interface,
)

__all__ = ["Agent"]

# Each built-in selector reads its Preference in one space. LQR regulates the state.
# The EFE selectors and the search drivers score a predicted observation.
_STATE_SPACE_SELECTORS = (LQRSelector,)
_OBSERVATION_SPACE_SELECTORS = (
    EFESelector,
    FFGEFESelector,
    RecedingHorizonSelector,
    OpenLoopSelector,
)

_Objective = StateGoal | ObservationGoal | Preference | None


def _require_held_target(
    model: LinearGaussianModel, target: Float64[Array, "n"], name: str
) -> None:
    """Refuse a target the dynamics move at zero action, where LQR settles short."""
    moved = equilibrium_drift(model.dynamics_matrix, target)
    if moved is not None:
        raise ValueError(
            f"{name} {target} is not an equilibrium of the dynamics: at zero action "
            f"it moves to {moved}. LQR regulates the offset from the target and "
            "charges for effort, so it would settle short of a target the dynamics "
            "do not hold."
        )


def _preference_for_selector(
    backend: InferenceBackend, objective: _Objective, selector: ActionSelector
) -> Preference:
    """The ``Preference`` a selector of the caller's own chooses toward, checked.

    A goal configures the selector the Agent builds. Beside ``selector=`` it would be
    a second description of the search, and the selector's would win unread.
    """
    if not isinstance(objective, Preference):
        given = "no objective" if objective is None else type(objective).__name__
        raise TypeError(
            "selector= takes a Preference, the target and precision the selector "
            f"chooses toward. Got {given}. A StateGoal or ObservationGoal configures "
            "the selector the Agent builds, so it goes without selector=."
        )
    model = backend.model
    if model.control_matrix is None:
        raise ValueError("selector= needs a model with a control matrix to act with.")
    if _scores_another_plant(selector, backend):
        raise ValueError(
            f"this {type(selector).__name__} was built for another model or graph. The "
            "Agent filters under its own, so build the selector from the same model, "
            "or over the same graph backend."
        )
    if isinstance(selector, _STATE_SPACE_SELECTORS):
        space, length = "a state", model.n_states
    elif isinstance(selector, _OBSERVATION_SPACE_SELECTORS):
        space, length = "an observation", model.n_observations
    else:
        return objective  # a custom selector reads its preference in its own space
    if objective.target.shape != (length,):
        raise ValueError(
            f"{type(selector).__name__} reads the Preference's target as {space}, a "
            f"vector of length {length}, got shape {objective.target.shape}"
        )
    if isinstance(selector, _STATE_SPACE_SELECTORS):
        _require_held_target(model, objective.target, "the Preference's target")
    return objective


def _same_model(first: LinearGaussianModel, second: LinearGaussianModel) -> bool:
    """Whether two models are the same plant: one structure and equal arrays.

    A model rebuilt from the same config is the same plant, so this compares values.
    A sensor or noise function compares by identity, since two functions cannot be
    compared by what they compute. A traced leaf cannot be read, so it counts as equal.
    """
    if first is second:
        return True
    first_leaves, first_structure = jax.tree_util.tree_flatten(first)
    second_leaves, second_structure = jax.tree_util.tree_flatten(second)
    if first_structure != second_structure:
        return False
    for first_leaf, second_leaf in zip(first_leaves, second_leaves, strict=True):
        first_value, second_value = concrete(first_leaf), concrete(second_leaf)
        if first_value is None or second_value is None:
            continue
        if not np.array_equal(first_value, second_value):
            return False
    return True


_BUILT_IN_SELECTORS = (*_STATE_SPACE_SELECTORS, *_OBSERVATION_SPACE_SELECTORS)
_GRAPH_SCORED_SELECTORS = (FFGEFESelector, RecedingHorizonSelector, OpenLoopSelector)


def _known_equal(first: ArrayLike, second: ArrayLike) -> bool:
    """Whether two arrays are known to be equal. A traced one cannot be read, so not."""
    first_value, second_value = concrete(first), concrete(second)
    if first_value is None or second_value is None:
        return False
    return bool(np.array_equal(first_value, second_value))


def _same_couplings(first: CouplingGraphBackend, second: CouplingGraphBackend) -> bool:
    """Whether two graph backends' edges join the same nodes with equal factors."""
    first_edges, second_edges = first.graph.couplings, second.graph.couplings
    return len(first_edges) == len(second_edges) and all(
        (a.parent, a.child, a.efe_relevant) == (b.parent, b.child, b.efe_relevant)
        and _known_equal(a.coupling_matrix, b.coupling_matrix)
        and _known_equal(a.coupling_noise, b.coupling_noise)
        for a, b in zip(first_edges, second_edges, strict=True)
    )


def _scores_another_plant(selector: ActionSelector, backend: InferenceBackend) -> bool:
    """Whether a built-in selector scores a different plant from ``backend``'s.

    A custom selector is not checked (ADR-080). A graph-scored selector compares the
    backend it scores on: the same object passes, and a ``CouplingGraphBackend`` with an
    equal interface model and equal couplings passes too, since the couplings are part
    of the plant and the interface model leaves them out. A coupling that cannot be
    read under a trace counts as different. Any other graph backend passes on an equal
    model when that model is its plant, and only as the same object when it is not.
    """
    if not isinstance(selector, _BUILT_IN_SELECTORS):
        return False
    graph = selector.backend if isinstance(selector, _GRAPH_SCORED_SELECTORS) else None
    if graph is None:
        # A flat score on a coupled graph leaves out the couplings the agent filters by.
        if is_graph_interface(backend.model):
            return True
        return not _same_model(selector.model, backend.model)
    if graph is backend:
        return False
    if not (
        isinstance(graph, CouplingGraphBackend)
        and isinstance(backend, CouplingGraphBackend)
    ):
        if is_graph_interface(graph.model) or is_graph_interface(backend.model):
            return True  # its couplings cannot be read, so only identity passes
        return not _same_model(graph.model, backend.model)
    return not (
        _same_model(graph.model, backend.model) and _same_couplings(graph, backend)
    )


def _lqr_path(
    model: LinearGaussianModel, objective: StateGoal
) -> tuple[Preference, LQRSelector]:
    """The preference and the LQR selector a ``StateGoal`` asks for."""
    # A state-dependent sensor would mean converting the goal through C (deferred).
    if state_dependent(model.observation_model) is not None:
        raise ValueError(
            "a StateGoal needs a fixed sensor. A state-dependent sensor would require "
            "converting the goal through C. Pass an ObservationGoal."
        )
    n, p = model.n_states, model.n_controls
    target = jnp.asarray(objective.target, dtype=float)
    if target.shape != (n,):
        raise ValueError(
            f"StateGoal target must be a 1-D vector of length {n} (the state "
            f"dimension), got shape {target.shape}"
        )
    _require_held_target(model, target, "StateGoal target")
    effort = jnp.eye(p) if objective.effort is None else objective.effort
    controller = LQRController(model, precision=objective.precision, effort=effort)
    return Preference(target, objective.precision), LQRSelector(controller)


def _information_can_move(model: LinearGaussianModel) -> bool:
    """Whether a noise depends on the state, so the action can change what is learned.

    With both noises fixed the posterior covariance is the same whatever the agent
    does, and the information term of the expected free energy is flat (ADR-003).
    """
    noises = (model.observation_model, model.dynamics_noise_model)
    return any(state_dependent(noise) is not None for noise in noises)


def _efe_path(
    backend: InferenceBackend, objective: ObservationGoal
) -> tuple[Preference, EFESelector | FFGEFESelector]:
    """The preference and the EFE selector an ``ObservationGoal`` asks for."""
    model = backend.model
    if model.control_matrix is None:
        raise ValueError(
            "an ObservationGoal needs a model with a control matrix to act."
        )
    is_graph = isinstance(backend, GraphBackend)
    if not _information_can_move(model):
        # A StateGoal on a coupled graph is refused, so a graph user is pointed at the
        # EFE selector.
        way_out = (
            "Pass a Preference with selector=FFGEFESelector(...) to choose by EFE "
            "anyway."
            if is_graph
            else "Pass a StateGoal for the LQR path, or a Preference with "
            "selector=EFESelector(...) to choose by EFE anyway."
        )
        raise ValueError(
            "an ObservationGoal needs a noise that depends on the state, and both "
            "noises are fixed here. The information term is then the same for every "
            f"action, and choosing by EFE is output regulation (deferred). {way_out}"
        )
    if objective.info_node is not None and not is_graph:
        raise ValueError(
            "info_node aims the epistemic at an FFG node. It needs a "
            "CouplingGraphBackend, not a flat backend."
        )
    m = model.n_observations
    target = jnp.asarray(objective.target, dtype=float)
    if target.shape != (m,):
        raise ValueError(
            f"ObservationGoal target must be a 1-D vector of length {m} (the "
            f"observation dimension), got shape {target.shape}"
        )
    preference = objective.preference
    if isinstance(backend, GraphBackend):
        return preference, _graph_efe_selector(backend, objective)
    selector = _reaching_efe_selector(model, objective, preference)
    sensor_is_fixed = (
        model.observation_model is None or model.observation_model.is_fixed
    )
    if sensor_is_fixed and selector.horizon > 1:
        raise ValueError(
            f"only Q(x) depends on the state here, and the target first moves the "
            f"scores at horizon {selector.horizon}. Q(x) is claimed at horizon 1 only "
            f"(#56). To score it anyway, pass a Preference with "
            f"selector=EFESelector(model, ..., horizon={selector.horizon})."
        )
    return preference, selector


def _graph_efe_selector(
    backend: GraphBackend, objective: ObservationGoal
) -> FFGEFESelector:
    """The EFE selector for an ObservationGoal on a branching backend (issue #26).

    ``info_node`` names the latent *node* whose marginal info gain is the
    epistemic value. ``None`` aims at the whole joint state. The selector resolves
    the node to its joint-state block.
    """
    if objective.horizon not in (None, 1):
        raise ValueError(
            "H-step EFE on the FFG backend is deferred, so ObservationGoal.horizon "
            "must be 1 with a CouplingGraphBackend."
        )
    return FFGEFESelector(
        backend,
        info_node=objective.info_node,
        n_candidates=objective.n_candidates,
        action_bounds=objective.action_bounds,
    )


def _reaching_efe_selector(
    model: LinearGaussianModel, objective: ObservationGoal, preference: Preference
) -> EFESelector:
    """The EFE selector for ``objective``, at a horizon that reads its target.

    The target enters ``G`` only through the predicted reading. Until a push reaches the
    reading, the candidates differ by their noise terms alone, or not at all, and the
    argmin chooses without regard to the goal. The probe differentiates every
    candidate's score with respect to the target, once, at the prior, and keeps the
    first horizon where the derivatives differ. It adds nothing per cycle (DECISIONS.md
    ADR-068).
    """

    def at(horizon: int) -> EFESelector:
        return EFESelector(
            model,
            n_candidates=objective.n_candidates,
            action_bounds=objective.action_bounds,
            horizon=horizon,
        )

    def reads_target(selector: EFESelector) -> bool:
        def scores(target: Float64[Array, "m"]) -> Float64[Array, "k"]:
            moved = Preference(target, preference.precision)
            return selector._scores(model.prior, moved)

        sensitivity = np.asarray(jax.jacfwd(scores)(preference.target))  # (k, m)
        finite = sensitivity[np.isfinite(sensitivity).all(axis=1)]
        if finite.shape[0] < 2:
            return True  # a NaN score is a different failure, reported elsewhere
        spread = float(np.ptp(finite, axis=0).max())
        return spread > 1e-12 * max(1.0, float(np.abs(finite).max()))

    # A push that has not reached the reading within n steps never will, so the search
    # stops at the state dimension.
    longest = model.n_states
    if objective.horizon is None:
        for horizon in range(1, longest + 1):
            selector = at(horizon)
            if reads_target(selector):
                return selector
        scored = f"within {longest} steps, the state dimension"
    else:
        selector = at(objective.horizon)
        if reads_target(selector):
            return selector
        longer = range(objective.horizon + 1, longest + 1)
        shortest = next((h for h in longer if reads_target(at(h))), None)
        if shortest is not None:
            raise ValueError(
                f"the target does not move the expected free energy at horizon "
                f"{objective.horizon}: a push first reaches the reading after "
                f"{shortest} steps, so the agent would choose its action without "
                f"regard to the goal. Pass horizon={shortest} or more, or leave "
                f"horizon unset."
            )
        scored = f"at horizon {objective.horizon}"
        if objective.horizon < longest:
            scored += f" or any longer one up to {longest}, the state dimension"
    raise ValueError(
        f"the target does not move the expected free energy {scored}, so the agent "
        f"cannot steer toward it. The control matrix does not reach the reading."
    )


def _short_repr(selector: ActionSelector | None) -> str:
    """A selector's own repr where it defines one, and its class name otherwise."""
    if selector is None:
        return "None"
    if type(selector).__repr__ is object.__repr__:
        return type(selector).__name__
    return repr(selector)


class Agent:
    """A continuous active-inference agent — the continuous sibling of pymdp's Agent.

    The backend and action selector underneath are pure: belief in, belief out;
    belief + preference in, action out. The ``Agent`` is the one *stateful* piece
    — it owns the
    current ``belief`` (the continuous analog of pymdp's ``qs``) and carries it
    forward across calls, so you drive it in the same perceive → act loop pymdp
    users already know::

        agent = Agent(model, StateGoal(target))
        belief = agent.infer_states(observation)   # perceive
        action = agent.sample_action()             # act

    The agent remembers the action it last sampled and feeds it to its own
    predict step on the next ``infer_states`` — so you never thread actions back
    in by hand, and a perceive-only model (no control matrix) needs no action at
    all. This mirrors the LQG loop exactly: the filter predicts with the action
    that was actually applied to the plant between observations.

    The vocabulary maps onto pymdp's like this:

    | pymdp (discrete) | cpomdp (continuous) | role |
    | --- | --- | --- |
    | ``Agent`` | ``Agent`` | the stateful façade |
    | ``qs`` | ``belief`` | posterior over the state |
    | ``infer_states`` | ``infer_states`` | fold an observation in |
    | ``sample_action`` | ``sample_action`` | choose an action |
    | ``A`` | ``observation_matrix``, ``observation_noise`` | the likelihood |
    | ``B`` | ``dynamics_matrix``, ``control_matrix``, ``dynamics_noise`` | transition |
    | ``C`` | ``objective`` | a StateGoal, an ObservationGoal, or a Preference |
    | ``D`` | ``model.prior`` | belief before any observation |
    | ``infer_policies`` | folded into ``sample_action`` | score the actions |

    The model has no letter aliases. Control theory's ``A`` is pymdp's ``B``, and its
    ``C`` is pymdp's ``A``, so each field goes by its role name only.

    One honest difference from pymdp: ``sample_action`` is **deterministic** here,
    not a draw from a policy posterior. With a fixed linear-Gaussian sensor and fixed
    dynamics noise the EFE-minimising action is the LQR optimum (ADR-003), so it
    returns the single best action. The name keeps the pymdp muscle-memory; the
    behaviour is exact.

    An agent built without an ``objective`` is a pure tracker: ``infer_states``
    works, but ``sample_action`` raises — there is nothing to act toward.

    The ``Agent`` is eager. It keeps the belief and the last action on itself, so
    ``jax.jit``, ``vmap`` and ``grad`` do not apply to its methods. A jitted
    ``infer_states`` filters every observation against the belief it was traced with,
    so the belief never advances, and the next eager call raises
    ``UnexpectedTracerError``. To compile a loop, call a backend's ``infer_states``
    and a selector's ``select`` inside ``jax.lax.scan``, as the API page shows.
    """

    def __init__(
        self,
        model: LinearGaussianModel,
        objective: StateGoal | ObservationGoal | Preference | None = None,
        *,
        selector: ActionSelector | None = None,
    ) -> None:
        """Build an agent over ``model``, optionally one that can act.

        The objective says what the agent pursues, and one place says how it searches.
        A goal carries its own search and the Agent builds the selector from it. A
        ``StateGoal`` steers in state space via LQR (it needs a fixed sensor). An
        ``ObservationGoal`` seeks a preferred observation via EFE, over the goal's
        horizon or, left unset, the shortest at which the target moves the scores. It
        needs a control matrix, and a sensor noise ``R(x)`` or dynamics noise ``Q(x)``
        that depends on the state, on a flat model and a graph alike. To choose with a
        selector of your own, pass a [`Preference`][cpomdp.Preference] and
        ``selector=``. The selector then carries the search. Omit the objective for a
        perceive-only tracker.

        The agent filters with a per-step [`KalmanBackend`][cpomdp.KalmanBackend]
        built from ``model``. For any other engine, use ``from_backend``.

        Args:
            model: The linear-Gaussian generative model the agent perceives and
                (with an objective) acts under. Its ``prior`` becomes the starting
                belief.
            objective: What the agent pursues. A ``StateGoal`` (a state to reach,
                carrying the LQR weights) or an ``ObservationGoal`` (an observation
                to prefer, carrying the action search). A ``Preference`` only beside
                ``selector=``. ``None`` builds a pure tracker, which perceives and
                never acts.
            selector: An [`ActionSelector`][cpomdp.ActionSelector] of your own, for a
                search no goal describes: EFE with both noises fixed, a
                varying-sequence search driver, an ``LQRSelector`` over a controller
                of your own, or a custom rule. It needs a ``Preference`` as the
                objective, in the space the selector reads. That is an observation
                for the EFE selectors and the search drivers, and a state for
                ``LQRSelector``.

        Raises:
            ValueError: If the objective and model are incompatible. That covers a
                ``StateGoal`` on a state-dependent sensor, an ``ObservationGoal`` on a
                control-free model, and an ``ObservationGoal`` on a model whose noises
                are both fixed (output regulation, deferred). Also raised for a target
                of the wrong length for the space it is read in, a target the
                dynamics do not hold at zero action on the LQR path, an
                ``ObservationGoal`` horizon too short for its target to move the
                scores, and ``selector=`` with a control-free model.
            TypeError: If ``objective`` is not a ``StateGoal``, ``ObservationGoal``
                or ``Preference``, if a goal comes with ``selector=``, or if a
                ``Preference`` or ``selector=`` comes without the other. Also raised
                for a coupled ``CouplingGraphBackend``'s model, which leaves out the
                couplings.
        """
        refuse_graph_interface(model, "Agent")
        self._setup(KalmanBackend(model), objective, selector)

    @classmethod
    def from_backend(
        cls,
        backend: InferenceBackend,
        objective: StateGoal | ObservationGoal | Preference | None = None,
        *,
        selector: ActionSelector | None = None,
    ) -> "Agent":
        """Build an agent that filters with ``backend`` and plans under its model.

        For an engine other than the default per-step Kalman filter: a steady-state
        [`KalmanBackend`][cpomdp.KalmanBackend], the RxInfer oracle, or a
        [`CouplingGraphBackend`][cpomdp.CouplingGraphBackend], which has no model of
        its own to pass. The agent's model is ``backend.model``, so the planner and
        the filter cannot work from different models.

        Args:
            backend: The inference engine, any
                [`InferenceBackend`][cpomdp.InferenceBackend].
            objective: What the agent pursues, as for the constructor.
            selector: A selector of your own, beside a ``Preference``, as for the
                constructor.

        Returns:
            The agent.

        Raises:
            ValueError: If the objective and the backend's model are incompatible, as
                for the constructor.
            TypeError: On an objective and ``selector=`` that do not pair, as for the
                constructor, or on a ``StateGoal`` with a coupled graph backend, whose
                model leaves out the couplings LQR would need.
        """
        agent = cls.__new__(cls)
        agent._setup(backend, objective, selector)
        return agent

    def _setup(
        self,
        backend: InferenceBackend,
        objective: _Objective,
        selector: ActionSelector | None,
    ) -> None:
        """Hold the backend, and the selector and preference the objective asks for."""
        model = backend.model
        self._backend = backend
        self._objective = objective
        self._belief = model.prior
        self._selector: ActionSelector | None = None
        self._preference: Preference | None = None
        self._last_action: Float64[Array, "p"] | None = None

        if selector is not None:
            self._preference = _preference_for_selector(backend, objective, selector)
            self._selector = selector
        elif objective is None:
            return  # a perceive-only tracker: nothing to act toward
        elif isinstance(objective, StateGoal):
            if is_graph_interface(model):
                raise TypeError(
                    "a StateGoal on a coupled CouplingGraphBackend would regulate with "
                    "LQR on a model that leaves out the couplings. Pass a Preference "
                    "with selector=FFGEFESelector(...), or an ObservationGoal if a "
                    "sensor's noise depends on the state."
                )
            self._preference, self._selector = _lqr_path(model, objective)
        elif isinstance(objective, ObservationGoal):
            self._preference, self._selector = _efe_path(backend, objective)
        elif isinstance(objective, Preference):
            raise TypeError(
                "a Preference carries no search, so the Agent cannot build a selector "
                "for it. Pass selector= with it, or a StateGoal or ObservationGoal "
                "for the Agent to build one."
            )
        else:
            raise TypeError(
                "objective must be a StateGoal, an ObservationGoal, or a Preference "
                f"beside selector=, got {type(objective).__name__}."
            )
        self._last_action = jnp.zeros(model.n_controls)

    @property
    def model(self) -> LinearGaussianModel:
        """The model the agent plans and filters under, read-only.

        It is the backend's model. The selector and any controller were built from it
        at construction, so acting under a different model needs a new ``Agent``.
        """
        return self._backend.model

    @property
    def belief(self) -> Belief:
        """The current posterior over the state, replaced by each ``infer_states``.

        Assigning a [`Belief`][cpomdp.Belief] over the model's state restarts
        filtering from it. Anything else is refused, since the next step would fail
        on it or filter from the wrong state.
        """
        return self._belief

    @belief.setter
    def belief(self, belief: Belief) -> None:
        if not isinstance(belief, Belief):
            raise TypeError(f"belief must be a Belief, got {type(belief).__name__}")
        n = self.model.n_states
        if belief.mean.shape != (n,):
            raise ValueError(
                f"belief must be over the model's {n}-D state, got a mean of shape "
                f"{belief.mean.shape}"
            )
        self._belief = belief

    @property
    def qs(self) -> Belief:
        """Alias for ``belief`` (read-only) — pymdp muscle-memory ``agent.qs``.

        The name is pymdp's, carried over for familiarity; the object is a Gaussian
        [`Belief`][cpomdp.Belief], not a categorical posterior. Prefer ``belief``.
        """
        return self.belief

    @property
    def selector(self) -> ActionSelector | None:
        """The selector that chooses the agent's actions, read-only.

        ``None`` for a tracker. For an ``ObservationGoal`` with no horizon set,
        ``selector.horizon`` is the horizon read off the model, and
        ``selector.cost_per_cycle`` is what each ``sample_action`` costs.
        """
        return self._selector

    def __repr__(self) -> str:
        """The objective, the selector and the backend, as built."""
        objective = None if self._objective is None else type(self._objective).__name__
        return (
            f"Agent(objective={objective}, selector={_short_repr(self._selector)}, "
            f"backend={type(self._backend).__name__})"
        )

    def infer_states(self, observation: ArrayLike) -> Belief:
        """Fold one observation into the belief and return the updated belief.

        The agent's current ``belief`` goes in as the prior and the posterior
        comes back out and is stored — that reassignment *is* the recursive
        filter, advanced one step. The belief is never mutated in place; each call
        replaces it with a fresh ``Belief``.

        No action is passed in: the agent supplies its own last sampled action to
        the predict step (zero before the first ``sample_action``), and a
        perceive-only model carries no action at all. This is the action actually
        applied to the plant since the previous observation — exactly what the
        Kalman predict step needs.

        Args:
            observation: The latest sensor reading, shape ``(m,)``.

        Returns:
            The updated belief (also stored on ``self.belief``).

        Raises:
            ValueError: On a shape mismatch in ``observation``, which the backend
                enforces.
        """
        observation = jnp.asarray(observation, dtype=float)
        self._belief = self._backend.infer_states(
            observation, self._belief, self._last_action
        )
        return self._belief

    def sample_action(self) -> Float64[Array, "p"]:
        """The action the agent's selector chooses for the current belief.

        Delegates to the agent's ``ActionSelector``, handing it the current belief
        and the objective's ``Preference``. For a ``StateGoal`` under a fixed
        sensor that selection is exactly the LQR optimum, ``-L∞·(mean − target)``
        with ``L∞`` the steady-state gain solved at construction (ADR-003). That is
        one matrix-vector product. For an ``ObservationGoal`` it is the
        EFE-minimising action over the front-loaded candidate grid. Deterministic,
        not a sample, and it takes no ``rng_key`` (see the class docstring). Policy
        evaluation is folded in here, so there is no separate ``infer_policies``
        step. On the EFE path its per-cycle cost reads back as
        ``agent.selector.cost_per_cycle``. The chosen action is remembered so the next
        ``infer_states`` predicts with it.

        Returns:
            The action, shape ``(p,)``.

        Raises:
            ValueError: If this is a perceive-only agent, built without an
                ``objective``, with nothing to act toward. Also raised when the
                selector returns NaN, which the grid selectors do when no candidate
                scores a finite ``G``. The action is then not remembered. The check
                is one host read of the ``p`` action entries per cycle.
        """
        if self._selector is None:
            raise ValueError(
                "this Agent has no objective, so it can only perceive. Pass a "
                "StateGoal or an ObservationGoal to Agent(...), or a Preference with "
                "selector=, to enable sample_action()."
            )
        assert self._preference is not None  # set with _selector; narrows the type
        action = self._selector.select(self.belief, self._preference)
        if bool(jnp.isnan(action).any()):
            raise ValueError(
                "the selector returned a non-finite action at this belief. For an "
                "EFE selector, no candidate scored a finite expected free energy, "
                "which a state-dependent R(x) or Q(x) that is not a covariance at "
                "the predicted states does."
            )
        self._last_action = action
        return action
