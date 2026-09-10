"""Quadratic regulation: the action-side dual of the Kalman filter.

``LQRController`` solves the control Riccati equation to its fixed point once and
applies the steady-state gain forever after. ``finite_horizon_lqr`` runs the same
backward recursion a declared number of steps and keeps every gain it produces, which
is the schedule an ``H``-step plan applies and the gain a receding-horizon planner at
horizon ``H`` applies at every step. The two agree in the limit and at no finite ``H``.
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Float64
from numpy.typing import ArrayLike

from cpomdp.backends.kalman import _gain_and_posterior_cov
from cpomdp.resolution import Bar, Bounded
from cpomdp.types import Belief, LinearGaussianModel

__all__ = [
    "CertaintyEquivalentController",
    "ControlBracket",
    "FiniteHorizonLQR",
    "KalmanSchedule",
    "LQRController",
    "closed_loop_cost",
    "control_bracket",
    "control_efficiency",
    "finite_horizon_lqr",
    "full_information_cost",
    "kalman_schedule",
    "optimal_cost",
]


def _validate_cost(
    matrix: Float64[Array, "dim dim"], name: str, *, require_definite: bool
) -> None:
    """Symmetry + (semi)definiteness check for a preference cost matrix.

    ``goal_precision`` and ``effort_penalty`` are user input handed in once at
    construction — the trust boundary — so unlike a per-step belief covariance
    they're checked in full here. (``types._validate_covariance`` skips
    definiteness on purpose because it runs on every filter step; this runs once,
    so it doesn't have to.) Both failure modes this catches are the
    silently-wrong-in-a-loop kind that are hardest to trace downstream: a
    non-symmetric matrix (an off-diagonal typo) quietly yields a non-symmetric
    cost-to-go and a wrong gain, and a singular or indefinite ``effort_penalty``
    — which the gain solve inverts against — blows up or returns garbage.

    Args:
        matrix: The already-shape-checked cost matrix.
        name: Field name for error messages.
        require_definite: ``True`` for ``effort_penalty`` (must be positive-
            *definite*, since it is inverted against); ``False`` for
            ``goal_precision`` (positive-*semi*-definite is enough).
    """
    if not jnp.allclose(matrix, matrix.T):
        raise ValueError(f"{name} must be symmetric.")
    if require_definite:
        # Cholesky succeeds iff (symmetric) positive-definite — the standard test.
        # JAX fills the result with NaNs instead of raising when it fails, so the
        # NaN check is what flags a singular or indefinite matrix.
        if bool(jnp.isnan(jnp.linalg.cholesky(matrix)).any()):
            raise ValueError(
                f"{name} must be positive-definite — the gain solve inverts "
                "against it — but it is singular or indefinite."
            )
    else:
        eigvals = jnp.linalg.eigvalsh(matrix)  # symmetric ⇒ real eigenvalues
        tol = 1e-8 * max(1.0, float(jnp.abs(eigvals).max()))
        if eigvals.min() < -tol:
            raise ValueError(
                f"{name} must be positive-semi-definite, but its smallest "
                f"eigenvalue is {eigvals.min():.3g}."
            )


def _validated_costs(
    model: LinearGaussianModel, goal_precision: ArrayLike, effort_penalty: ArrayLike
) -> tuple[Float64[Array, "n n"], Float64[Array, "p p"]]:
    """The two cost matrices as arrays, checked against the model they regulate.

    Raises:
        ValueError: If the model has no ``control_matrix``, or a cost matrix does
            not match the state/action dimensions, is not symmetric, or fails its
            definiteness requirement (``goal_precision`` PSD, ``effort_penalty``
            PD).
    """
    if model.control_matrix is None:
        raise ValueError(
            "LQR needs an action channel: the model has no control matrix, "
            "so there is nothing to act with."
        )
    goal_precision = jnp.asarray(goal_precision, dtype=float)
    effort_penalty = jnp.asarray(effort_penalty, dtype=float)
    n, p = model.n_states, model.n_controls
    if goal_precision.shape != (n, n):
        raise ValueError(
            f"goal_precision must be {n}x{n} to match the {n}-D state, "
            f"got shape {goal_precision.shape}"
        )
    if effort_penalty.shape != (p, p):
        raise ValueError(
            f"effort_penalty must be {p}x{p} to match the {p}-D action, "
            f"got shape {effort_penalty.shape}"
        )
    _validate_cost(goal_precision, "goal_precision", require_definite=False)
    _validate_cost(effort_penalty, "effort_penalty", require_definite=True)
    return goal_precision, effort_penalty


def _riccati_step(
    remaining: Float64[Array, "n n"],
    dynamics_matrix: Float64[Array, "n n"],
    control_matrix: Float64[Array, "n p"],
    effort_penalty: Float64[Array, "p p"],
) -> tuple[Float64[Array, "n n"], Float64[Array, "p n"]]:
    """One backward Bellman step: the cost-to-go before acting, and the gain that acts.

    ``remaining`` is ``W``, what the state an action arrives at will cost from there
    on, stage cost included. Minimising ``(Ax + Bu)ᵀ W (Ax + Bu) + uᵀ R u`` over
    ``u`` gives::

        L = (R + Bᵀ W B)⁻¹ (Bᵀ W A)
        P = Aᵀ W A − (Aᵀ W B) L

    ``(R + Bᵀ W B)`` is solved against rather than inverted, for the same reason the
    filter solves against its innovation covariance.
    """
    cross = dynamics_matrix.T @ remaining @ control_matrix  # Aᵀ W B  (n×p)
    # curvature of the action cost, the dual of the Kalman innovation covariance (p×p)
    inner = effort_penalty + control_matrix.T @ remaining @ control_matrix
    gain = jnp.linalg.solve(inner, cross.T)  # L  (p×n)
    cost_to_go = dynamics_matrix.T @ remaining @ dynamics_matrix - cross @ gain  # P
    return cost_to_go, gain


@dataclass(frozen=True)
class FiniteHorizonLQR:
    """The gain schedule and cost-to-go of an ``H``-step regulator.

    The stage cost is charged on the state each action arrives at, and nothing is
    charged after the last one: the terminal cost is zero. That is the sum a
    receding-horizon planner scores over its lookahead, so ``first_gain`` is the gain
    such a planner applies at every step, and it is what a comparison against one
    has to use. ``LQRController.gain`` is its limit as ``H`` grows and differs from it
    at every finite ``H``, by an amount that shrinks with ``H`` and reads as an error
    when the horizons are not matched.

    The schedule carries the model and the two costs it was built from, so anything
    priced against it reads the same ``Q``, ``R`` and noise the recursion did.

    Args:
        model: The model the schedule regulates.
        goal_precision: The stage cost on the state it was built with, ``Q``.
        effort_penalty: The stage cost on the action it was built with, ``R``.
        gains: ``gains[k]`` is the gain applied at step ``k`` of the plan, with
            ``H − k`` steps remaining, shape ``(H, p, n)``. The action is
            ``−gains[k] · state``.
        cost_to_go: ``cost_to_go[j]`` is ``P_j``, the matrix of the optimal cost
            ``stateᵀ · P_j · state`` with ``j`` steps remaining, before acting, shape
            ``(H + 1, n, n)``. ``cost_to_go[0]`` is zero.
    """

    model: LinearGaussianModel
    goal_precision: Float64[Array, "n n"]
    effort_penalty: Float64[Array, "p p"]
    gains: Float64[Array, "H p n"]
    cost_to_go: Float64[Array, "H+1 n n"]

    @property
    def horizon(self) -> int:
        """``H``, the number of steps planned."""
        return int(self.gains.shape[0])

    @property
    def first_gain(self) -> Float64[Array, "p n"]:
        """The gain with all ``H`` steps remaining, the receding-horizon gain."""
        return self.gains[0]


def finite_horizon_lqr(
    model: LinearGaussianModel,
    *,
    goal_precision: ArrayLike,
    effort_penalty: ArrayLike,
    horizon: int,
) -> FiniteHorizonLQR:
    """Run the control Riccati recursion backward over ``horizon`` steps.

    The same Bellman step ``LQRController`` iterates to a fixed point, run a declared
    number of times from a zero terminal cost and with every gain kept. With ``j``
    steps remaining and ``W = goal_precision + P_{j−1}`` the cost of what remains::

        L_j = (effort_penalty + Bᵀ W B)⁻¹ (Bᵀ W A)
        P_j = Aᵀ W A − (Aᵀ W B) L_j

    starting from ``P_0 = 0``. At ``horizon = 1`` the gain is the one-step regulator
    ``(effort_penalty + Bᵀ Q B)⁻¹ Bᵀ Q A``. The Control page of the API reference
    opens the same account in plain terms.

    Args:
        model: The linear-Gaussian model to act in. Must carry a ``control_matrix``.
        goal_precision: The stage cost on the state, an ``(n, n)`` matrix. (LQR's
            ``Q``.)
        effort_penalty: The stage cost on the action, a ``(p, p)`` matrix. (LQR's
            ``R``.)
        horizon: ``H``, how many steps the plan covers. At least one.

    Returns:
        The schedule, with ``gains[k]`` the gain at step ``k`` of the plan and
        ``cost_to_go[j]`` the cost matrix with ``j`` steps remaining.

    Raises:
        ValueError: If ``horizon`` is below one, or on any of the cost and model
            conditions ``LQRController`` refuses.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    goal_precision, effort_penalty = _validated_costs(
        model, goal_precision, effort_penalty
    )
    dynamics_matrix = model.dynamics_matrix  # A
    assert model.control_matrix is not None  # refused by _validated_costs
    control_matrix = model.control_matrix  # B

    costs = [jnp.zeros_like(goal_precision)]
    gains = []
    for _ in range(horizon):
        cost_to_go, gain = _riccati_step(
            goal_precision + costs[-1], dynamics_matrix, control_matrix, effort_penalty
        )
        costs.append(cost_to_go)
        gains.append(gain)
    # Built with the fewest steps remaining first; the plan applies them the other
    # way round.
    return FiniteHorizonLQR(
        model=model,
        goal_precision=goal_precision,
        effort_penalty=effort_penalty,
        gains=jnp.stack(gains[::-1]),
        cost_to_go=jnp.stack(costs),
    )


def _require_fixed(model: LinearGaussianModel, slot: str) -> None:
    """Refuse a noise model that varies with the state; the closed forms price fixed."""
    carried = getattr(model, slot)
    if carried is not None and not carried.is_fixed:
        raise ValueError(
            f"the model's {slot} is state-dependent; the closed form prices a fixed "
            f"noise"
        )


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
    held = model.dynamics_matrix @ goal
    if not jnp.allclose(
        held, goal, rtol=0.0, atol=1e-12 * max(1.0, float(jnp.abs(goal).max()))
    ):
        raise ValueError(
            "goal is not an equilibrium of the dynamics: at zero action it moves to "
            f"{held}. The plan regulates the offset from the goal, and the offset only "
            "obeys the same dynamics when the goal holds still."
        )
    return goal


def _prior_over(model: LinearGaussianModel, prior: Belief | None) -> Belief:
    """``prior``, or the model's, refused unless it is over the model's state."""
    prior = model.prior if prior is None else prior
    if prior.mean.shape != (model.n_states,):
        raise ValueError(
            f"prior is over {prior.mean.shape[0]} states and the model has "
            f"{model.n_states}"
        )
    return prior


@dataclass(frozen=True)
class KalmanSchedule:
    """The exact filter's gains and error covariances over ``H`` readings.

    Both are data-independent, so they can be written down before a reading exists.
    The indexing follows the plan's: action ``k`` is chosen from the belief after
    ``k`` readings, and reading ``k + 1`` arrives after it.

    Args:
        gains: ``gains[k]`` folds reading ``k + 1`` into the estimate, shape
            ``(H, n, m)``.
        covariances: ``covariances[k]`` is the error covariance of the estimate action
            ``k`` is chosen from, shape ``(H + 1, n, n)``. ``covariances[0]`` is the
            prior's.
    """

    gains: Float64[Array, "H n m"]
    covariances: Float64[Array, "H+1 n n"]


def kalman_schedule(
    model: LinearGaussianModel, horizon: int, prior: Belief | None = None
) -> KalmanSchedule:
    """The exact filter's gain and covariance at each of ``horizon`` readings.

    The covariance recursion the per-step ``KalmanBackend`` runs, run ahead of time
    and kept, since under fixed noise it never sees a reading.

    Args:
        model: The model the filter runs under, with fixed noise.
        horizon: How many readings, at least one.
        prior: The belief before the first reading. Defaults to the model's prior.

    Returns:
        The gains and covariances, indexed as the plan indexes its steps.

    Raises:
        ValueError: If ``horizon`` is below one, if either noise depends on the state,
            or if ``prior`` is not over the model's state.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    _require_fixed(model, "observation_model")
    _require_fixed(model, "dynamics_noise_model")
    prior = _prior_over(model, prior)
    gains, covariances = [], [prior.cov]
    for _ in range(horizon):
        gain, cov_post = _gain_and_posterior_cov(
            model.dynamics_matrix,
            model.observation_matrix,
            model.dynamics_noise,
            model.observation_noise,
            covariances[-1],
        )
        gains.append(gain)
        covariances.append(cov_post)
    return KalmanSchedule(gains=jnp.stack(gains), covariances=jnp.stack(covariances))


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
    _require_fixed(model, "observation_model")
    _require_fixed(model, "dynamics_noise_model")
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
    _require_fixed(model, "dynamics_noise_model")
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


class LQRController:
    """Steady-state LQR action selection — the action-side dual of the filter.

    Where the Kalman filter front-loads perception (solve the estimation Riccati
    once for the steady-state gain ``K∞``, then ``mean += K∞·prediction_error``),
    this front-loads action: solve the dual *control* Riccati once for ``L∞``,
    then ``action = -L∞·(mean − goal)``. Both gains are data-independent, both are
    computed at construction, and together they are LQG (see RESEARCH.md).

    The load-bearing claim (ADR-003) is that LQR *is* active inference here, not a
    substitute for it. For a fixed linear-Gaussian sensor the covariance recursion
    is control-independent, so Expected Free Energy's epistemic term is identical
    for every action and drops out of the argmin; EFE-minimising selection reduces
    to its pragmatic term, and the pragmatic term under a Gaussian preference is a
    quadratic cost whose optimum is exactly LQR. The epistemic term only re-enters
    once sensing depends on the state or action — out of scope for v0.1.

    The two cost matrices are named for the preference they encode, not by LQR's
    traditional ``Q``/``R`` — those letters already mean the noise covariances on
    the model (``dynamics_noise``/``observation_noise``), the exact collision ADR-003
    warns about. The names are the same across the whole library: an ``Agent``
    hands these straight through to its controller.

    Args:
        model: The linear-Gaussian model to act in. Must carry a ``control_matrix``
            matrix — there is nothing to act with otherwise.
        goal_precision: How sharply the agent prefers the goal, an ``(n, n)``
            matrix. It is exactly the precision of the Gaussian preference centred
            at the goal — ``exp(−½(state−goal)ᵀ·goal_precision·(state−goal))`` —
            so heavier ``goal_precision`` buys a more aggressive controller.
            (LQR's ``Q``.)
        effort_penalty: How much action costs, a ``(p, p)`` matrix. Heavier
            ``effort_penalty`` buys a gentler controller. (LQR's ``R``.)
        tol: Absolute tolerance on successive cost-to-go iterates; convergence is
            declared when they stop moving by more than this.
        max_iter: Iteration cap before the Riccati recursion is declared to have
            failed to converge.

    Raises:
        ValueError: If the model has no ``control_matrix``, or a cost matrix does
            not match the state/action dimensions, is not symmetric, or fails its
            definiteness requirement (``goal_precision`` PSD, ``effort_penalty``
            PD).
        RuntimeError: If the control Riccati does not converge within ``max_iter``
            — typically because ``(dynamics, control)`` is not stabilisable.
    """

    def __init__(
        self,
        model: LinearGaussianModel,
        *,
        goal_precision: ArrayLike,
        effort_penalty: ArrayLike,
        tol: float = 1e-12,
        max_iter: int = 1000,
    ) -> None:
        self.model = model
        self._goal_precision, self._effort_penalty = _validated_costs(
            model, goal_precision, effort_penalty
        )
        self._gain = self._converge_to_steady_state(tol, max_iter)

    @property
    def gain(self) -> Float64[Array, "p n"]:
        """The steady-state feedback gain L∞, shape (p, n)."""
        return self._gain

    def action(self, mean: ArrayLike, goal: ArrayLike) -> Float64[Array, "p"]:
        """The action that drives the estimated state toward ``goal``.

        One matrix-vector product, ``-L∞·(mean − goal)`` — all the work was
        front-loaded into ``L∞`` at construction, so there is no optimisation in
        the loop. The ``mean − goal`` shift turns the regulator (which drives its
        state to zero) into a controller that drives the state to ``goal``.

        Args:
            mean: The current belief mean — the best estimate of the state,
                shape ``(n,)``.
            goal: The state to steer toward, shape ``(n,)``. It must be an
                equilibrium the dynamics can hold at zero action; aim at a
                non-equilibrium and a steady-state offset is left behind.

        Returns:
            The action, shape ``(p,)``.

        Raises:
            ValueError: If ``goal`` is not a 1-D vector of length ``n``.
        """
        # self._gain : (p, n) L∞;  mean, goal : (n,);  returns (p,)
        mean = jnp.asarray(mean, dtype=float)
        goal = jnp.asarray(goal, dtype=float)
        if goal.shape != (self.model.n_states,):
            raise ValueError(
                f"goal must be a 1-D vector of length {self.model.n_states} "
                f"(the state dimension), got shape {goal.shape}"
            )
        return -self._gain @ (mean - goal)

    def _converge_to_steady_state(
        self, tol: float, max_iter: int
    ) -> Float64[Array, "p n"]:
        """Iterate the control Riccati recursion to its fixed point for ``L∞``.

        The exact dual of ``KalmanBackend._converge_to_steady_state``. The filter
        iterates a *covariance* forward until it stops moving; this iterates a
        *cost-to-go* — the matrix ``P`` of the quadratic value function
        ``V(state) = stateᵀ·P·state`` — until it stops moving. Starting from
        ``goal_precision``, each step applies Bellman's equation::

            P ← goal_precision + Aᵀ P A − (Aᵀ P B)(effort_penalty + Bᵀ P B)⁻¹(Bᵀ P A)

        "the cost from here = what I pay now + the cost from wherever the dynamics
        carry me, minus what acting optimally buys back." For a stabilisable
        ``(A, B)`` this converges to the unique fixed point ``P∞`` (the solution
        of the discrete algebraic Riccati equation), from which the steady-state
        gain follows::

            L∞ = (effort_penalty + Bᵀ P∞ B)⁻¹ (Bᵀ P∞ A)

        (A=dynamics, B=control.) The ``(effort_penalty + Bᵀ P B)`` term is solved
        against with ``jnp.linalg.solve`` rather than inverted explicitly, for the
        same numerical reason the filter solves against its innovation covariance.

        Returns:
            The steady-state gain ``L∞``, shape ``(p, n)``.

        Raises:
            RuntimeError: If the recursion has not converged within ``max_iter``.
        """
        dynamics_matrix = self.model.dynamics_matrix  # A  (n×n)
        assert self.model.control_matrix is not None  # refused by _validated_costs
        control_matrix = self.model.control_matrix  # B  (n×p)
        # P, starting at the running state cost (n×n)
        cost_to_go = self._goal_precision

        for _ in range(max_iter):
            # pay now, plus what the dynamics carry forward net of what acting buys back
            carried, _ = _riccati_step(
                cost_to_go, dynamics_matrix, control_matrix, self._effort_penalty
            )
            next_cost_to_go = self._goal_precision + carried

            if jnp.allclose(cost_to_go, next_cost_to_go, atol=tol, rtol=0.0):
                cost_to_go = next_cost_to_go
                break
            cost_to_go = next_cost_to_go
        else:
            raise RuntimeError(
                f"control Riccati did not converge in {max_iter} iterations; "
                "(dynamics, control) may not be stabilisable, so no steady-state "
                "gain exists."
            )

        _, gain = _riccati_step(
            cost_to_go, dynamics_matrix, control_matrix, self._effort_penalty
        )
        return gain  # L∞  (p×n)
