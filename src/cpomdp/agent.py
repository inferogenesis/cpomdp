"""The ``Agent`` façade: a stateful perceive/act loop over a ``LinearGaussianModel``."""

import jax.numpy as jnp
from jaxtyping import Array, Float64
from numpy.typing import ArrayLike

from cpomdp.backends.base import InferenceBackend
from cpomdp.backends.kalman import KalmanBackend
from cpomdp.control import LQRController
from cpomdp.selection import (
    ActionSelector,
    EFESelector,
    LQRSelector,
    ObservationGoal,
    Preference,
    StateGoal,
)
from cpomdp.types import Belief, LinearGaussianModel

__all__ = ["Agent"]


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

    ==================  ============================  ==============================
    pymdp (discrete)    cpomdp (continuous)           role
    ==================  ============================  ==============================
    ``Agent``           ``Agent``                     the stateful façade
    ``qs``              ``belief``                    posterior over the state
    ``infer_states``    ``infer_states``              fold an observation in
    ``sample_action``   ``sample_action``             choose an action
    ``C``               ``objective``                 a StateGoal or ObservationGoal
    ``D``               ``model.prior``               belief before any observation
    ==================  ============================  ==============================

    One honest difference from pymdp: ``sample_action`` is **deterministic** here,
    not a draw from a policy posterior. For a fixed linear-Gaussian sensor the
    EFE-minimising action is the LQR optimum (ADR-003), so it returns the single
    best action. The name keeps the pymdp muscle-memory; the behaviour is exact.

    An agent built without an ``objective`` is a pure tracker: ``infer_states``
    works, but ``sample_action`` raises — there is nothing to act toward.
    """

    def __init__(
        self,
        model: LinearGaussianModel,
        objective: ObservationGoal | StateGoal | None = None,
        *,
        selector: ActionSelector | None = None,
        backend: InferenceBackend | None = None,
    ) -> None:
        """Build an agent over ``model``, optionally one that can act.

        The objective's *type* selects the regime: a ``StateGoal`` steers in state
        space via LQR (it needs a fixed sensor); an ``ObservationGoal`` seeks a
        preferred observation via one-step EFE (it needs a control matrix, and a
        state-dependent sensor unless an explicit ``selector`` is given). Omit the
        objective for a perceive-only tracker.

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
                automatic dispatch declines (e.g. EFE on a fixed sensor).
            backend: The inference engine. Defaults to a per-step
                ``KalmanBackend``; pass any ``InferenceBackend`` (e.g. a
                steady-state Kalman or the RxInfer oracle) to swap engines.

        Raises:
            ValueError: If the objective and model are incompatible — a
                ``StateGoal`` on a state-dependent sensor, an ``ObservationGoal``
                on a control-free model, or an ``ObservationGoal`` on a fixed
                sensor without a ``selector`` (output regulation, deferred); or if
                the ``StateGoal`` target is not a 1-D vector of length ``n``.
            TypeError: If ``objective`` is neither a ``StateGoal`` nor an
                ``ObservationGoal``.
        """
        self.model = model
        self.belief = model.prior
        self._backend = backend if backend is not None else KalmanBackend(model)
        sensor_is_fixed = model.observation is None or model.observation.is_fixed

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
            n, p = model.n_states, model.n_controls
            self._goal = jnp.asarray(objective.target, dtype=float)
            if self._goal.shape != (n,):
                raise ValueError(
                    f"StateGoal target must be a 1-D vector of length {n} (the state "
                    f"dimension), got shape {self._goal.shape}"
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
            if model.control is None:
                raise ValueError(
                    "an ObservationGoal needs a model with a control matrix to act."
                )
            if sensor_is_fixed and selector is None:
                raise ValueError(
                    "an ObservationGoal on a fixed sensor is output regulation "
                    "(deferred); pass a StateGoal for the LQR path, or "
                    "selector=EFESelector(...) to opt into H=1 EFE."
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
            self._preference = Preference(
                jnp.asarray(objective.target, dtype=float), objective.precision
            )
            self._selector = (
                selector
                if selector is not None
                else EFESelector(
                    model,
                    n_candidates=objective.n_candidates,
                    action_bounds=objective.action_bounds,
                    horizon=objective.horizon,
                )
            )
            self._last_action = jnp.zeros(model.n_controls)

        else:
            raise TypeError(
                f"objective must be a StateGoal or ObservationGoal, "
                f"got {type(objective).__name__}."
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
            ValueError: On a shape mismatch in ``observation`` (enforced by the
                backend; see ``validate_step_inputs``).
        """
        observation = jnp.asarray(observation, dtype=float)
        self.belief = self._backend.infer_states(
            observation, self.belief, self._last_action
        )
        return self.belief

    def sample_action(self) -> Float64[Array, "p"]:
        """The action the agent's selector chooses for the current belief.

        Delegates to the agent's ``ActionSelector``, handing it the current belief
        and the objective's ``Preference``. For a ``StateGoal`` under a fixed
        sensor that selection is exactly the LQR optimum, ``-L∞·(mean − goal)`` —
        one matrix-vector product, front-loaded at construction (ADR-003); for an
        ``ObservationGoal`` it is the one-step EFE-minimising action over the
        front-loaded candidate grid. Deterministic, not a sample (see the class
        docstring). The chosen action is remembered so the next ``infer_states``
        predicts with it.

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
