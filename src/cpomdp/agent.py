"""The ``Agent`` façade: a stateful perceive/act loop over a ``LinearGaussianModel``."""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float64
from numpy.typing import ArrayLike

from cpomdp._validation import equilibrium_drift
from cpomdp.backends.base import GraphBackend, InferenceBackend
from cpomdp.backends.kalman import KalmanBackend
from cpomdp.control import LQRController
from cpomdp.enumeration import OpenLoopSelector, RecedingHorizonSelector
from cpomdp.selection import (
    ActionSelector,
    EFESelector,
    FfgEfeSelector,
    LQRSelector,
    ObservationGoal,
    Preference,
    StateGoal,
)
from cpomdp.types import Belief, LinearGaussianModel

__all__ = ["Agent"]

# Each built-in selector reads its Preference in one space. LQR regulates the state.
# The EFE selectors and the search drivers score a predicted observation.
_STATE_SPACE_SELECTORS = (LQRSelector,)
_OBSERVATION_SPACE_SELECTORS = (
    EFESelector,
    FfgEfeSelector,
    RecedingHorizonSelector,
    OpenLoopSelector,
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

        sensitivity = np.asarray(jax.jacfwd(scores)(preference.goal))  # (k, m)
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
    | ``C`` | ``objective`` | a StateGoal or ObservationGoal |
    | ``D`` | ``model.prior`` | belief before any observation |

    One honest difference from pymdp: ``sample_action`` is **deterministic** here,
    not a draw from a policy posterior. With a fixed linear-Gaussian sensor and fixed
    process noise the EFE-minimising action is the LQR optimum (ADR-003), so it
    returns the single best action. The name keeps the pymdp muscle-memory; the
    behaviour is exact.

    An agent built without an ``objective`` is a pure tracker: ``infer_states``
    works, but ``sample_action`` raises — there is nothing to act toward.

    The ``Agent`` is eager. It keeps the belief and the last action on itself, so
    ``jax.jit``, ``vmap`` and ``grad`` do not apply to its methods. A jitted
    ``infer_states`` bakes in the belief it was traced with and returns the same
    posterior on every call. To compile a loop, call a backend's ``infer_states``
    and a selector's ``select`` inside ``jax.lax.scan``, as the API page shows.
    """

    def __init__(
        self,
        model: LinearGaussianModel,
        objective: ObservationGoal | StateGoal | None = None,
        *,
        selector: ActionSelector | None = None,
    ) -> None:
        """Build an agent over ``model``, optionally one that can act.

        The objective's *type* selects the regime: a ``StateGoal`` steers in state
        space via LQR (it needs a fixed sensor); an ``ObservationGoal`` seeks a
        preferred observation via EFE, over the goal's horizon or, left unset, the
        shortest at which the target moves the scores (it needs a control matrix, and a
        state-dependent sensor unless an explicit ``selector`` is given). Omit the
        objective for a perceive-only tracker.

        The agent filters with a per-step [`KalmanBackend`][cpomdp.KalmanBackend]
        built from ``model``. For any other engine, use ``from_backend``.

        Args:
            model: The linear-Gaussian generative model the agent perceives and
                (with an objective) acts under. Its ``prior`` becomes the starting
                belief.
            objective: What the agent pursues — a ``StateGoal`` (a state to reach,
                carrying the LQR weights) or an ``ObservationGoal`` (an observation
                to prefer, carrying the action-search config). ``None`` builds a
                pure tracker that perceives but cannot act.
            selector: An explicit ``ActionSelector`` overriding the one the
                objective would dispatch — the escape hatch for regimes the
                automatic dispatch declines (e.g. EFE on a fixed sensor). It reads
                the objective's target in the objective's own space, so a built-in
                EFE selector needs an ``ObservationGoal`` and ``LQRSelector`` a
                ``StateGoal``.

        Raises:
            ValueError: If the objective and model are incompatible — a
                ``StateGoal`` on a state-dependent sensor, an ``ObservationGoal``
                on a control-free model, or an ``ObservationGoal`` on a fixed
                sensor without a ``selector`` (output regulation, deferred); or if
                the ``StateGoal`` target is not a 1-D vector of length ``n``; or if a
                built-in ``selector`` reads its target in the other space from the
                objective; or if the LQR path is given a ``StateGoal`` target the
                dynamics do not hold at zero action. Also raised when an
                ``ObservationGoal``'s horizon is too short for its target to move the
                scores.
            TypeError: If ``objective`` is neither a ``StateGoal`` nor an
                ``ObservationGoal``.
        """
        self._setup(KalmanBackend(model), objective, selector)

    @classmethod
    def from_backend(
        cls,
        backend: InferenceBackend,
        objective: ObservationGoal | StateGoal | None = None,
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
            selector: An explicit ``ActionSelector``, as for the constructor.

        Returns:
            The agent.

        Raises:
            ValueError: If the objective and the backend's model are incompatible, as
                for the constructor.
            TypeError: If ``objective`` is neither a ``StateGoal`` nor an
                ``ObservationGoal``.
        """
        agent = cls.__new__(cls)
        agent._setup(backend, objective, selector)
        return agent

    def _setup(
        self,
        backend: InferenceBackend,
        objective: ObservationGoal | StateGoal | None,
        selector: ActionSelector | None,
    ) -> None:
        """Hold the backend and dispatch the objective to a selector."""
        model = backend.model
        self._backend = backend
        self._objective = objective
        self._belief = model.prior
        sensor_is_fixed = (
            model.observation_model is None or model.observation_model.is_fixed
        )

        if objective is None:
            # perceive-only tracker: nothing to act toward.
            self._controller: LQRController | None = None
            self._goal: Float64[Array, "n"] | None = None
            self._last_action: Float64[Array, "p"] | None = None
            self._selector: ActionSelector | None = None
            self._preference: Preference | None = None
        elif isinstance(objective, StateGoal):
            # state-space goal -> LQR. Needs a fixed sensor; a state-dependent sensor
            # would mean converting the goal through C (deferred), so guard it.
            if not sensor_is_fixed:
                raise ValueError(
                    "a StateGoal needs a fixed sensor; a state-dependent sensor would "
                    "require converting the goal through C — pass an ObservationGoal."
                )
            if isinstance(selector, _OBSERVATION_SPACE_SELECTORS):
                raise ValueError(
                    f"{type(selector).__name__} scores its target as an observation, "
                    "and a StateGoal's target is a state. Pass an ObservationGoal "
                    "with the observation to seek, or omit selector= for the LQR path."
                )
            n, p = model.n_states, model.n_controls
            self._goal = jnp.asarray(objective.target, dtype=float)
            if self._goal.shape != (n,):
                raise ValueError(
                    f"StateGoal target must be a 1-D vector of length {n} (the state "
                    f"dimension), got shape {self._goal.shape}"
                )
            if selector is None or isinstance(selector, _STATE_SPACE_SELECTORS):
                moved = equilibrium_drift(model.dynamics_matrix, self._goal)
                if moved is not None:
                    raise ValueError(
                        f"StateGoal target {self._goal} is not an equilibrium of the "
                        f"dynamics: at zero action it moves to {moved}. LQR regulates "
                        "the offset from the target and charges for effort, so it "
                        "would settle short of a target the dynamics do not hold."
                    )
            effort = jnp.eye(p) if objective.effort is None else objective.effort
            self._controller = LQRController(
                model, goal_precision=objective.precision, effort_penalty=effort
            )
            self._preference = Preference(self._goal, objective.precision)
            self._selector = (
                selector if selector is not None else LQRSelector(self._controller)
            )
            self._last_action = jnp.zeros(p)
        elif isinstance(objective, ObservationGoal):
            # observation-space goal -> EFE. Needs control to act; on a fixed sensor it
            # is output regulation (deferred), so require an explicit selector there.
            if model.control_matrix is None:
                raise ValueError(
                    "an ObservationGoal needs a model with a control matrix to act."
                )
            if isinstance(selector, _STATE_SPACE_SELECTORS):
                raise ValueError(
                    f"{type(selector).__name__} regulates the state toward its target, "
                    "and an ObservationGoal's target is an observation. Pass a "
                    "StateGoal for the LQR path, or an EFE selector."
                )
            is_ffg = isinstance(self._backend, GraphBackend)
            if sensor_is_fixed and selector is None and not is_ffg:
                raise ValueError(
                    "an ObservationGoal on a fixed sensor is output regulation "
                    "(deferred); pass a StateGoal for the LQR path, or "
                    "selector=EFESelector(...) to opt into H=1 EFE."
                )
            if objective.info_node is not None and not is_ffg:
                raise ValueError(
                    "info_node aims the epistemic at an FFG node; it needs a "
                    "CouplingGraphBackend, not a flat backend."
                )
            m = model.n_observations
            obs_target = jnp.asarray(objective.target, dtype=float)
            if obs_target.shape != (m,):
                raise ValueError(
                    f"ObservationGoal target must be a 1-D vector of length {m} (the "
                    f"observation dimension), got shape {obs_target.shape}"
                )
            self._controller = None
            self._goal = None
            self._preference = Preference(obs_target, objective.precision)
            if selector is not None:
                self._selector = selector
            elif is_ffg:
                self._selector = self._build_ffg_selector(objective)
            else:
                self._selector = _reaching_efe_selector(
                    model, objective, self._preference
                )
            self._last_action = jnp.zeros(model.n_controls)

        else:
            raise TypeError(
                f"objective must be a StateGoal or ObservationGoal, "
                f"got {type(objective).__name__}."
            )

    def _build_ffg_selector(self, objective: ObservationGoal) -> FfgEfeSelector:
        """The EFE selector for an ObservationGoal on a branching backend (issue #26).

        ``info_node`` names the latent *node* whose marginal info gain is the
        epistemic value; ``None`` aims at the whole joint state. Translate that node to
        its joint-state block and hand it to the ``FfgEfeSelector``.
        """
        backend = self._backend
        assert isinstance(backend, GraphBackend)  # only built for the FFG path
        if objective.horizon not in (None, 1):
            raise ValueError(
                "H-step EFE on the FFG backend is deferred; ObservationGoal.horizon "
                "must be 1 with a CouplingGraphBackend."
            )
        if objective.info_node is None:
            info_block = range(backend.model.n_states)  # whole-state info gain
        else:
            try:
                info_block = backend.block(objective.info_node)
            except ValueError as error:
                raise ValueError(f"info_node {objective.info_node}: {error}") from error
        return FfgEfeSelector(
            backend,
            info_block=info_block,
            n_candidates=objective.n_candidates,
            action_bounds=objective.action_bounds,
        )

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
        sensor that selection is exactly the LQR optimum, ``-L∞·(mean − goal)`` —
        one matrix-vector product, front-loaded at construction (ADR-003); for an
        ``ObservationGoal`` it is the EFE-minimising action over the front-loaded
        candidate grid. Deterministic, not a sample, and it takes no ``rng_key`` (see
        the class docstring). There is no separate ``infer_policies`` step — policy
        evaluation is folded in here. On the EFE path its per-cycle cost reads back as
        ``agent.selector.cost_per_cycle``. The chosen action is remembered so the next
        ``infer_states`` predicts with it.

        Returns:
            The action, shape ``(p,)``.

        Raises:
            ValueError: If this is a perceive-only agent (built without an
                ``objective``) — there is nothing to act toward.
        """
        if self._selector is None:
            raise ValueError(
                "this Agent has no objective, so it can only perceive; pass a "
                "StateGoal(...) or ObservationGoal(...) to Agent(...) to enable "
                "sample_action()."
            )
        assert self._preference is not None  # set with _selector; narrows the type
        self._last_action = self._selector.select(self.belief, self._preference)
        return self._last_action
