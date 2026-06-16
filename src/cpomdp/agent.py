"""The ``Agent`` façade: a stateful perceive/act loop over a ``LinearGaussianModel``."""

import jax.numpy as jnp
from jaxtyping import Array, Float64
from numpy.typing import ArrayLike

from cpomdp.backends.base import InferenceBackend
from cpomdp.backends.kalman import KalmanBackend
from cpomdp.control import LQRController
from cpomdp.types import Belief, LinearGaussianModel

__all__ = ["Agent"]


class Agent:
    """A continuous active-inference agent — the continuous sibling of pymdp's Agent.

    The backend and controller underneath are pure: belief in, belief out; mean
    in, action out. The ``Agent`` is the one *stateful* piece — it owns the
    current ``belief`` (the continuous analog of pymdp's ``qs``) and carries it
    forward across calls, so you drive it in the same perceive → act loop pymdp
    users already know::

        agent = Agent(model, goal=target)
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
    ``C``               ``goal`` + ``goal_precision``  preferred state + precision
    ``D``               ``model.prior``               belief before any observation
    ==================  ============================  ==============================

    One honest difference from pymdp: ``sample_action`` is **deterministic** here,
    not a draw from a policy posterior. For a fixed linear-Gaussian sensor the
    EFE-minimising action is the LQR optimum (ADR-003), so it returns the single
    best action. The name keeps the pymdp muscle-memory; the behaviour is exact.

    An agent built without a ``goal`` is a pure tracker: ``infer_states`` works,
    but ``sample_action`` raises — there is nothing to act toward.
    """

    def __init__(
        self,
        model: LinearGaussianModel,
        *,
        goal: ArrayLike | None = None,
        goal_precision: ArrayLike | None = None,
        effort_penalty: ArrayLike | None = None,
        backend: InferenceBackend | None = None,
    ) -> None:
        """Build an agent over ``model``, optionally one that can act.

        Args:
            model: The linear-Gaussian generative model the agent perceives and
                (if given a goal) acts under. Its ``prior`` becomes the starting
                belief.
            goal: The preferred state to steer toward, shape ``(n,)``. Omit it for
                a perceive-only tracker. Must be an equilibrium the dynamics can
                hold at zero action (see ``LQRController.action``).
            goal_precision: How sharply the goal is preferred, an ``(n, n)``
                matrix; defaults to the identity. Ignored without a ``goal``.
            effort_penalty: How much action costs, a ``(p, p)`` matrix; defaults
                to the identity. Ignored without a ``goal``.
            backend: The inference engine. Defaults to a per-step
                ``KalmanBackend``; pass any ``InferenceBackend`` (e.g. a
                steady-state Kalman or the RxInfer oracle) to swap engines.

        Raises:
            ValueError: If ``goal_precision``/``effort_penalty`` are given without
                a ``goal``, or ``goal`` is not a 1-D vector of length ``n``.
        """
        self.model = model
        self.belief = model.prior
        self._backend = backend if backend is not None else KalmanBackend(model)

        if goal is None:
            # perceive-only tracker: preferences without a goal are meaningless,
            # so flag them rather than silently ignoring them.
            if goal_precision is not None or effort_penalty is not None:
                raise ValueError(
                    "goal_precision/effort_penalty were given but goal is None; "
                    "preferences need a goal to act toward."
                )
            self._controller: LQRController | None = None
            self._goal: Float64[Array, "n"] | None = None
            self._last_action: Float64[Array, "p"] | None = None
        else:
            # acting agent — build the front-loaded controller now.
            n, p = model.n_states, model.n_controls
            self._goal = jnp.asarray(goal, dtype=float)
            if self._goal.shape != (n,):
                raise ValueError(
                    f"goal must be a 1-D vector of length {n} (the state "
                    f"dimension), got shape {self._goal.shape}"
                )
            goal_precision = jnp.eye(n) if goal_precision is None else goal_precision
            effort_penalty = jnp.eye(p) if effort_penalty is None else effort_penalty
            self._controller = LQRController(
                model, goal_precision=goal_precision, effort_penalty=effort_penalty
            )
            # No action applied yet — the first predict step coasts on zero
            # control, then sample_action overwrites this each step.
            self._last_action = jnp.zeros(p)

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
        """The action that best drives the current belief toward the goal.

        Reads the current belief mean and returns the LQR-optimal action,
        ``-L∞·(mean − goal)`` — one matrix-vector product, since the controller
        was front-loaded at construction. Deterministic, not a sample (see the
        class docstring). The chosen action is remembered so the next
        ``infer_states`` predicts with it.

        Returns:
            The action, shape ``(p,)``.

        Raises:
            ValueError: If this is a perceive-only agent (built without a
                ``goal``) — there is nothing to act toward.
        """
        if self._controller is None:
            raise ValueError(
                "this Agent has no goal, so it can only perceive; pass goal=... to "
                "Agent(...) to enable sample_action()."
            )
        assert self._goal is not None  # set together with _controller; narrows the type
        self._last_action = self._controller.action(self.belief.mean, self._goal)
        return self._last_action
