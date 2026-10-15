"""The control bracket: the two costs every controller under one plan sits between.

Private research apparatus (DECISIONS.md ADR-069). It stays out of the public surface
until its leg under state-dependent noise lands, since that leg changes the signatures.

The lower cost is what the plan costs when the state is known exactly at every step.
The upper one is the best any controller can do when it has to work the state out from
readings. The gap between them is the price of not knowing, what the sensor fails to
deliver. That gap is the number to report. Either end on its own is a cost in units
nothing calibrates.

Under fixed noise the upper end is settled. Take the gains the full-information plan
would use and apply them to the filtered estimate instead of the state. That is the
certainty-equivalent controller, and the separation principle says nothing that infers
the state does better. Its cost is written down two ways that share nothing but the two
Riccati recursions: once by propagating how the state and the estimate spread together,
once as the floor plus what the estimate's error costs at each step. The two agreeing to
machine precision is the signature of the fixed-noise regime, and it is what makes the
upper end a closed form rather than a measurement.

An agent's cost is then read as a position inside the bracket. Zero means it did exactly
as well as certainty equivalence. One means it did as well as knowing the state. An
agent that can move its own sensor to where the readings are sharper can sit above zero,
and that is the effect the bracket exists to measure. The bar on the agent's cost,
divided by the width, is the floor below which its position cannot be told from zero.
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Float64
from numpy.typing import ArrayLike

from cpomdp._lab.resolution import Bar, Bounded
from cpomdp._validation import equilibrium_drift, require_fixed_noise
from cpomdp.control import (
    _PRICES_FIXED_NOISE,
    FiniteHorizonLQR,
    _prior_over,
    kalman_schedule,
)
from cpomdp.types import Belief, LinearGaussianModel

__all__ = [
    "CertaintyEquivalentController",
    "ControlBracket",
    "closed_loop_cost",
    "control_bracket",
    "control_efficiency",
    "full_information_cost",
    "optimal_cost",
]


def _held_goal(model: LinearGaussianModel, goal: ArrayLike) -> Float64[Array, "n"]:
    """``goal`` as an array, refused unless the dynamics hold it at zero action.

    The plan regulates the offset from the goal, and the offset obeys the model's
    dynamics only when ``A · goal = goal``.
    """
    n = model.n_states
    goal = jnp.asarray(goal, dtype=float)
    if goal.shape != (n,):
        raise ValueError(
            f"goal must be a 1-D vector of length {n} (the state dimension), got "
            f"shape {goal.shape}"
        )
    held = equilibrium_drift(model.dynamics_matrix, goal)
    if held is not None:
        raise ValueError(
            "goal is not an equilibrium of the dynamics: at zero action it moves to "
            f"{held}. The plan regulates the offset from the goal, and the offset only "
            "obeys the same dynamics when the goal holds still."
        )
    return goal


def closed_loop_cost(
    schedule: FiniteHorizonLQR,
    filter_gains: ArrayLike,
    *,
    goal: ArrayLike,
    controller_gains: ArrayLike | None = None,
    prior: Belief | None = None,
) -> float:
    """The expected cost of acting on a linear estimate, in closed form.

    Action ``k`` is ``−L_k · (estimate − goal)``, chosen from the estimate after ``k``
    readings; the first is chosen from the prior mean. The state then moves, is
    charged where it lands, and is read, and the estimate folds the reading in with
    ``K_{k+1}``. Every map is linear and every disturbance Gaussian, so the joint
    second moment of ``(state, estimate)`` about the goal propagates exactly and the
    cost is a sum of traces against it. Nothing is sampled.

    Any data-independent gain sequence prices this way: the exact filter's, a frozen
    one's, or a degraded one's. ``L_k`` defaults to the plan's own schedule, and can be
    replaced to price a gain the plan did not choose, as the steady-state gain applied
    at every step of a short plan.

    Args:
        schedule: The plan, with the model and costs it was built from.
        filter_gains: ``K_{k+1}`` for each step, shape ``(H, n, m)``.
        goal: The state steered toward, shape ``(n,)``, held by the dynamics.
        controller_gains: ``L_k`` for each step, shape ``(H, p, n)``. Defaults to
            ``schedule.gains``.
        prior: Where the state starts and what the estimate starts as. Defaults to the
            model's prior.

    Returns:
        The expected cost in the units of ``goal_precision`` and ``effort_penalty``.

    Raises:
        ValueError: If either noise depends on the state, if ``goal`` is not a vector
            of length ``n`` the dynamics hold, if ``prior`` is not over the model's
            state, or if either gain sequence is not one per step of the plan.
    """
    model = schedule.model
    require_fixed_noise(model, _PRICES_FIXED_NOISE)
    goal = _held_goal(model, goal)
    prior = _prior_over(model, prior)
    n, m, p, horizon = (
        model.n_states,
        model.n_observations,
        model.n_controls,
        (schedule.horizon),
    )
    filter_gains = jnp.asarray(filter_gains, dtype=float)
    if filter_gains.shape != (horizon, n, m):
        raise ValueError(
            f"filter_gains must be one (n, m) gain per step, shape "
            f"{(horizon, n, m)}, got {filter_gains.shape}"
        )
    controller_gains = (
        schedule.gains
        if controller_gains is None
        else jnp.asarray(controller_gains, dtype=float)
    )
    if controller_gains.shape != (horizon, p, n):
        raise ValueError(
            f"controller_gains must be one (p, n) gain per step, shape "
            f"{(horizon, p, n)}, got {controller_gains.shape}"
        )
    dynamics_matrix = model.dynamics_matrix  # A
    assert model.control_matrix is not None  # the schedule was built on it
    control_matrix = model.control_matrix  # B
    observation_matrix = model.observation_matrix  # C
    goal_precision, effort_penalty = schedule.goal_precision, schedule.effort_penalty

    # Second moment of (state − goal, estimate − goal). The estimate starts at the
    # prior mean with no spread of its own; the state starts spread around it.
    offset = prior.mean - goal  # d
    certain = jnp.outer(offset, offset)
    moment = jnp.block([[prior.cov + certain, certain], [certain, certain]])
    disturbance = jnp.block(
        [
            [model.dynamics_noise, jnp.zeros((n, m))],
            [jnp.zeros((m, n)), model.observation_noise],
        ]
    )
    identity = jnp.eye(n)
    cost = 0.0
    for controller_gain, filter_gain in zip(
        controller_gains, filter_gains, strict=True
    ):
        # uᵀ R u with u = −L · (estimate − goal)
        cost += jnp.trace(
            controller_gain.T @ effort_penalty @ controller_gain @ moment[n:, n:]
        )
        pushed = dynamics_matrix - control_matrix @ controller_gain  # A − B L
        # state ← A·state − B L·estimate + w
        # estimate ← K C A·state + (A − B L − K C A)·estimate + K C·w + K·v
        corrected = filter_gain @ observation_matrix @ dynamics_matrix  # K C A
        transition = jnp.block(
            [
                [dynamics_matrix, -control_matrix @ controller_gain],
                [corrected, pushed - corrected],
            ]
        )
        entry = jnp.block(
            [
                [identity, jnp.zeros((n, m))],
                [filter_gain @ observation_matrix, filter_gain],
            ]
        )
        moment = transition @ moment @ transition.T + entry @ disturbance @ entry.T
        cost += jnp.trace(goal_precision @ moment[:n, :n])
    return float(cost)


@dataclass(frozen=True)
class CertaintyEquivalentController:
    """Acts on the estimate as if it were the state: ``−L_k · (mean − goal)``.

    The plan's gains are the ones a controller that saw the state would use. Applying
    them to a filtered mean instead is certainty equivalence, and under fixed noise
    it is optimal: the separation principle says no controller that has to infer the
    state does better. ``expected_cost`` is ``J_CE``, priced by ``closed_loop_cost``
    with the exact filter's gains.

    Args:
        schedule: The plan, with the model and costs it was built from.
        goal: The state steered toward, shape ``(n,)``, held by the dynamics.

    Raises:
        ValueError: If ``goal`` is not a vector of length ``n`` the dynamics hold.
    """

    schedule: FiniteHorizonLQR
    goal: Float64[Array, "n"]

    def __init__(self, schedule: FiniteHorizonLQR, *, goal: ArrayLike) -> None:
        object.__setattr__(self, "schedule", schedule)
        object.__setattr__(self, "goal", _held_goal(schedule.model, goal))

    def action(self, step: int, mean: ArrayLike) -> Float64[Array, "p"]:
        """The action at ``step`` of the plan from the current estimate.

        Args:
            step: Which step of the plan this is, from ``0``.
            mean: The belief mean after ``step`` readings, shape ``(n,)``.

        Returns:
            ``−gains[step] · (mean − goal)``, shape ``(p,)``.
        """
        return -self.schedule.gains[step] @ (jnp.asarray(mean, dtype=float) - self.goal)

    def expected_cost(self, prior: Belief | None = None) -> float:
        """``J_CE``: this controller's expected cost with the exact filter.

        Args:
            prior: Where the state starts and what the filter starts from. Defaults
                to the model's prior.
        """
        gains = kalman_schedule(
            self.schedule.model, self.schedule.horizon, prior=prior
        ).gains
        return closed_loop_cost(self.schedule, gains, goal=self.goal, prior=prior)


def full_information_cost(
    schedule: FiniteHorizonLQR, *, goal: ArrayLike, prior: Belief | None = None
) -> float:
    """``J_lower``: the expected cost of the plan when the state is observed exactly.

    The floor of the control bracket. No controller that has to infer the state from
    readings can do better, since the state itself is the most it could know. With
    ``d`` the prior mean's offset from the goal, ``Σ₀`` the prior covariance and
    ``Q_w`` the process noise::

        J_lower = dᵀ P_H d + tr(P_H Σ₀) + Σ_{j=0}^{H−1} tr((Q + P_j) Q_w)

    The first two terms are the terminal quadratic averaged over where the state
    starts. The sum is what the process noise adds at each step, priced at the cost
    of everything that remains from there.

    The goal has to be a state the dynamics hold at zero action, ``A · goal = goal``,
    since the plan regulates the offset from it and the offset only obeys the same
    dynamics when the goal does not drift.

    Args:
        schedule: The plan, with the model and costs it was built from.
        goal: The state the plan steers toward, shape ``(n,)``.
        prior: Where the state starts, as a Gaussian. Defaults to the model's prior.

    Returns:
        The expected cost in the units of ``goal_precision`` and ``effort_penalty``.

    Raises:
        ValueError: If the model's process noise depends on the state, if ``goal`` is
            not a vector of length ``n`` the dynamics hold, or if ``prior`` is not
            over the model's state.
    """
    model = schedule.model
    require_fixed_noise(model, _PRICES_FIXED_NOISE, slots=("dynamics_noise_model",))
    goal = _held_goal(model, goal)
    prior = _prior_over(model, prior)

    offset = prior.mean - goal  # d
    terminal = schedule.cost_to_go[schedule.horizon]  # P_H
    from_start = offset @ terminal @ offset + jnp.trace(terminal @ prior.cov)
    # W_j = Q + P_j for j = 0..H−1: what remains after the step whose noise it prices
    remaining = schedule.goal_precision + schedule.cost_to_go[:-1]
    from_noise = jnp.einsum("jab,ba->", remaining, model.dynamics_noise)
    return float(from_start + from_noise)


def optimal_cost(
    schedule: FiniteHorizonLQR, *, goal: ArrayLike, prior: Belief | None = None
) -> float:
    """``J*``: the least any controller that infers the state from readings can pay.

    The separation principle prices it without propagating a state. The plan's gains
    are optimal whatever the estimate, and acting on an estimate instead of the state
    costs, at each step, the estimate's error charged at what an action error costs
    from there. With ``Σ_k`` the exact filter's error covariance of the estimate action
    ``k`` is chosen from, and ``W_k = Q + P_{H−k−1}`` what remains after the step::

        J* = J_lower + Σ_{k=0}^{H−1} tr(L_kᵀ (R + Bᵀ W_k B) L_k Σ_k)

    ``closed_loop_cost`` with the exact filter's gains reaches the same number by
    propagating the joint second moment of state and estimate. The two share no
    arithmetic past the two Riccati recursions, so their agreement to machine
    precision is the fixed-noise signature: certainty equivalence is optimal, and no
    use of the readings beats it.

    Args:
        schedule: The plan, with the model and costs it was built from.
        goal: The state the plan steers toward, shape ``(n,)``, held by the dynamics.
        prior: Where the state starts and what the filter starts from. Defaults to the
            model's prior.

    Returns:
        The expected cost in the units of ``goal_precision`` and ``effort_penalty``.

    Raises:
        ValueError: If either noise depends on the state, if ``goal`` is not a vector
            of length ``n`` the dynamics hold, or if ``prior`` is not over the model's
            state.
    """
    model = schedule.model
    floor = full_information_cost(schedule, goal=goal, prior=prior)
    covariances = kalman_schedule(model, schedule.horizon, prior=prior).covariances
    assert model.control_matrix is not None  # the schedule was built on it
    control_matrix, effort_penalty = model.control_matrix, schedule.effort_penalty
    # W_k = Q + P_{H−k−1}: what remains after step k, read in step order
    remaining = schedule.goal_precision + schedule.cost_to_go[-2::-1]
    penalty = 0.0
    for gain, after, error in zip(
        schedule.gains, remaining, covariances[:-1], strict=True
    ):
        action_cost = effort_penalty + control_matrix.T @ after @ control_matrix
        penalty += jnp.trace(gain.T @ action_cost @ gain @ error)
    return float(floor + penalty)


@dataclass(frozen=True)
class ControlBracket:
    """The two costs every controller under one plan sits between.

    The floor is the plan with the state observed exactly. The ceiling is the best
    any controller that has to infer the state from readings can do, which under fixed
    noise is the certainty-equivalent controller with the exact filter. Their
    difference is what the readings fail to deliver: the price of inference under this
    plan. That width is the object to report. Either end alone is a number in cost
    units that nothing calibrates.

    Args:
        floor: ``J_lower``, from ``full_information_cost``.
        ceiling: ``J_CE``, from ``closed_loop_cost`` with the exact filter's gains.
    """

    floor: float
    ceiling: float

    @property
    def width(self) -> float:
        """``ceiling − floor``, the price of inference under the plan."""
        return self.ceiling - self.floor


def control_bracket(
    schedule: FiniteHorizonLQR, *, goal: ArrayLike, prior: Belief | None = None
) -> ControlBracket:
    """The floor and ceiling of a plan, both in closed form.

    Args:
        schedule: The plan, with the model and costs it was built from.
        goal: The state the plan steers toward, shape ``(n,)``, held by the dynamics.
        prior: Where the state starts and what the filter starts from. Defaults to the
            model's prior.

    Raises:
        ValueError: Whatever ``full_information_cost`` and ``closed_loop_cost`` refuse.
    """
    floor = full_information_cost(schedule, goal=goal, prior=prior)
    gains = kalman_schedule(schedule.model, schedule.horizon, prior=prior).gains
    ceiling = closed_loop_cost(schedule, gains, goal=goal, prior=prior)
    return ControlBracket(floor=floor, ceiling=ceiling)


def control_efficiency(bracket: ControlBracket, agent_cost: Bounded) -> Bounded:
    """``η_ctrl``: how much of the price of inference an agent recovers.

    ``(ceiling − agent_cost) / width``, with the agent's bar scaled by the width. Zero
    for a certainty-equivalent controller under fixed noise. Positive when an agent
    knows more than the fixed-noise filter can, as one that steers its own sensor
    does. Negative when it pays more than certainty equivalence would. Every term is
    scored under the plan's own model, so the number is within-model and needs no
    reference.

    Both ends of the bracket are closed forms, so the result's bar is the agent's
    alone, divided by the width. That bar is the floor below which ``η_ctrl`` cannot
    be told from zero, and a claim of zero is a claim to it.

    Args:
        bracket: The plan's floor and ceiling.
        agent_cost: The agent's expected cost under the same plan, with its bar.

    Returns:
        ``η_ctrl`` with its bar: ``common_mode`` is the agent's negated and divided by
        the width, since the agent's cost enters negated; ``own`` is the agent's
        divided by the width.

    Raises:
        ValueError: If the bracket's width is not positive, since there is then no
            price of inference to normalise by.
    """
    width = bracket.width
    if width <= 0.0:
        raise ValueError(
            f"the bracket's width must be positive to normalise by, got {width!r}"
        )
    return Bounded(
        value=(bracket.ceiling - agent_cost.value) / width,
        bar=Bar(
            common_mode=-agent_cost.bar.common_mode / width,
            own=agent_cost.bar.own / width,
        ),
    )
