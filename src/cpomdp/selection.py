"""Action selection: the seam between a belief+preference and an action."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float64
from numpy.typing import ArrayLike

from cpomdp._validation import validate_covariance
from cpomdp.backends.base import EfeBackend
from cpomdp.control import LQRController
from cpomdp.efe import (
    _checked_info_block,
    _ffg_efe_step,
    expected_free_energy,
    policy_efe,
)
from cpomdp.enumeration import SearchWarrant
from cpomdp.types import Belief, LinearGaussianModel

__all__ = [
    "ActionSelector",
    "EFESelector",
    "FfgEfeSelector",
    "LQRSelector",
    "ObservationGoal",
    "Preference",
    "StateGoal",
]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class Preference:
    """What the agent wants: a goal and how sharply it is preferred.

    Single-mode for v0.3 — one Gaussian preference. The disjunctive *mixture*
    case (visit one of several goals) is RFC-002, deferred; this type is the seam
    that a mixture ``Preference`` plugs into.

    ``precision`` is unused by [`LQRSelector`][cpomdp.LQRSelector] (it is baked into
    the controller's Riccati solve at construction); it is carried here for the EFE
    pragmatic term added in Phase 1A.
    """

    goal: Float64[Array, "n"]
    precision: Float64[Array, "n n"]

    def __init__(self, goal: ArrayLike, precision: ArrayLike | None = None) -> None:
        goal = jnp.asarray(goal, dtype=float)
        object.__setattr__(self, "goal", goal)
        n = goal.shape[0]
        object.__setattr__(
            self,
            "precision",
            jnp.eye(n) if precision is None else jnp.asarray(precision, dtype=float),
        )
        self._validate()

    def _validate(self) -> None:
        if self.goal.ndim != 1:
            raise ValueError(f"goal must be a 1-D vector, got shape {self.goal.shape}")
        validate_covariance(self.precision, "precision")
        n = self.goal.shape[0]
        if self.precision.shape != (n, n):
            raise ValueError(
                f"precision must be {n}x{n} to match the {n}-D goal, "
                f"got shape {self.precision.shape}"
            )

    def tree_flatten(self):
        """Leaves: (goal, precision); no static aux. Lets jit/vmap take a Preference."""
        return (self.goal, self.precision), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Rebuild without re-validating — the leaves may be tracers."""
        goal, precision = children
        obj = object.__new__(cls)
        object.__setattr__(obj, "goal", goal)
        object.__setattr__(obj, "precision", precision)
        return obj


@runtime_checkable
class ActionSelector(Protocol):
    """Chooses an action from a belief and a preference.

    The abstraction wall for action selection: [`LQRSelector`][cpomdp.LQRSelector] is
    the fixed-sensor case (EFE collapses to LQR, ADR-003);
    [`EFESelector`][cpomdp.EFESelector] is the state-dependent one.
    [`Agent`][cpomdp.Agent] depends only on this, never on a concrete selector.
    """

    def select(self, belief: Belief, preference: Preference) -> Float64[Array, "p"]:
        """The action to take given the current ``belief`` and ``preference``."""
        ...


@dataclass(frozen=True)
class LQRSelector:
    """Adapts an LQRController to the ActionSelector interface.

    A thin wrapper: it owns no control logic, it just forwards to the
    front-loaded controller, unpacking the belief's mean and the preference's
    goal. EFE collapses to LQR under a fixed sensor (ADR-003), so for that regime
    this *is* the action selector.
    """

    controller: LQRController

    def select(self, belief: Belief, preference: Preference) -> Float64[Array, "p"]:
        """Forward to the controller: ``action(belief.mean, preference.goal)``.

        No control logic of its own — the Riccati solve was front-loaded into the
        controller at construction. This just unpacks the belief and preference
        into the controller's ``(mean, goal)`` signature.
        """
        return self.controller.action(belief.mean, preference.goal)


def _require_scalar_control(
    control_matrix: Float64[Array, "n p"] | None, selector: str
) -> None:
    """Refuse a model a one-dimensional action grid cannot search."""
    if control_matrix is None:
        raise ValueError(
            f"{selector} needs a model with a control matrix; an action has no "
            "effect on a control-free (pure-tracking) model."
        )
    p = control_matrix.shape[1]
    if p != 1:
        raise ValueError(
            f"{selector} searches a 1-D action grid (p=1); got p={p}. "
            "For p>1 use EnumeratedEfeSearch over a declared FiniteActionSet "
            "(cpomdp.enumeration), or pass a custom selector."
        )


class EFESelector:
    """EFE action selection over a front-loaded candidate grid, horizon-aware.

    At ``horizon = 1`` (default) it minimises one-step ``G`` over the grid. At
    ``horizon > 1`` it scores **constant-action** policies (each grid action held for H
    steps) via ``policy_efe`` and returns the first action of the best one
    (receding-horizon). Per-cycle cost is a single attributable number,
    ``cost_per_cycle = n_candidates * horizon``.

    Built directly, the selector uses the ``horizon`` it is given. On a model where a
    push takes more than one step to reach the reading, ``horizon = 1`` leaves the
    target out of every comparison. The candidates then differ by their noise alone, or
    tie and the first wins. The ``Agent`` reads the horizon for an ``ObservationGoal``
    off the model and refuses one too short for the target to move the scores.

    Honest caveat: ``horizon`` selects the best *constant* action, not the best
    *sequence*. A genuinely sequential epistemic policy — move to sense, then exploit —
    needs a varying sequence the constant-action family cannot express, so at H > 1 the
    selector can still look myopic-ish on such tasks. True varying-sequence search is
    ``EnumeratedEfeSearch`` in ``cpomdp.enumeration``, which enumerates every length-H
    sequence of a declared finite action set (ADR-031).
    """

    def __init__(
        self,
        model: LinearGaussianModel,
        *,
        n_candidates: int,
        action_bounds: tuple[float, float],
        horizon: int = 1,
    ) -> None:
        _require_scalar_control(model.control_matrix, "EFESelector")
        lo, hi = action_bounds
        if not lo < hi:
            raise ValueError(
                f"action_bounds must be (lo, hi) with lo < hi, got {action_bounds}"
            )
        if n_candidates < 2:
            raise ValueError(
                f"n_candidates must be at least 2 to search, got {n_candidates}"
            )
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self._model = model
        self._horizon = horizon
        self._candidates = jnp.linspace(lo, hi, n_candidates)[:, None]

    @staticmethod
    def _argmin(g: Float64[Array, "k"]) -> Array:
        # A NaN-scoring candidate must not silently win: map NaN -> +inf so it loses
        # the search (jnp.argmin otherwise treats NaN as the minimum).
        return jnp.argmin(jnp.where(jnp.isnan(g), jnp.inf, g))

    def select(self, belief: Belief, preference: Preference) -> Float64[Array, "p"]:
        """The grid action minimising ``G`` over the horizon (the per-cycle work).

        At ``horizon = 1`` one ``vmap`` of the one-step kernel + ``argmin``. At
        ``horizon > 1`` one ``vmap`` of ``policy_efe`` over the constant-action policies
        + ``argmin``, returning the first (= constant) action of the best policy.
        """
        g = self._scores(belief, preference)
        return self._candidates[self._argmin(g)]  # first (= constant) action

    def _scores(self, belief: Belief, preference: Preference) -> Float64[Array, "k"]:
        """``G`` for every candidate over the horizon, in grid order."""
        if self._horizon == 1:
            return jax.vmap(
                lambda a: expected_free_energy(self._model, belief, a, preference)[0]
            )(self._candidates)
        # H>1: each candidate becomes a constant-action policy (held for H steps).
        policies = jnp.repeat(self._candidates[:, None, :], self._horizon, axis=1)
        return jax.vmap(
            lambda pol: policy_efe(self._model, belief, pol, preference)[0]
        )(policies)

    @property
    def n_candidates(self) -> int:
        """The per-cycle EFE-evaluation count — attributable work (RFC-001)."""
        return self._candidates.shape[0]

    @property
    def horizon(self) -> int:
        """The lookahead depth — constant-action steps scored per candidate."""
        return self._horizon

    @property
    def cost_per_cycle(self) -> int:
        """Per-cycle step-evals = n_candidates * horizon."""
        return self.n_candidates * self._horizon

    @property
    def warrant(self) -> SearchWarrant:
        """``CORROBORATED`` — the grid samples a continuum, it does not decide it.

        The counterpart to ``EnumeratedEfeSearch.warrant`` (``PROVED``): the two search
        families print in different vocabulary so a sampled result is never read as a
        decided one (standing rule 6; ADR-031).
        """
        return SearchWarrant.CORROBORATED

    def __repr__(self) -> str:
        """The horizon and the grid, the numbers that set what each cycle costs."""
        lo, hi = float(self._candidates[0, 0]), float(self._candidates[-1, 0])
        return (
            f"EFESelector(horizon={self._horizon}, n_candidates={self.n_candidates}, "
            f"action_bounds=({lo}, {hi}))"
        )


class FfgEfeSelector:
    """EFE action selection over a branching FFG backend (issue #26).

    The FFG counterpart of [`EFESelector`][cpomdp.EFESelector]: instead of
    ``_efe_step`` on a flat model it ``vmap``s ``_ffg_efe_step`` over the candidate
    action grid, reading ``μ⁺``/``Σ⁺``
    from the backend's ``predicted_belief`` (structural couplings folded in) and aiming
    the epistemic term at ``info_block`` — a joint-state block (a node's block via
    ``backend.block`` for ``info_node``, or the whole state). One-step
    (``horizon = 1``); the H-step FFG rollout is a later seam. Conforms to
    [`ActionSelector`][cpomdp.ActionSelector].
    """

    def __init__(
        self,
        backend: EfeBackend,
        *,
        info_block: Sequence[int],
        n_candidates: int,
        action_bounds: tuple[float, float],
    ) -> None:
        _require_scalar_control(backend.model.control_matrix, "FfgEfeSelector")
        lo, hi = action_bounds
        if not lo < hi:
            raise ValueError(
                f"action_bounds must be (lo, hi) with lo < hi, got {action_bounds}"
            )
        if n_candidates < 2:
            raise ValueError(
                f"n_candidates must be at least 2 to search, got {n_candidates}"
            )
        # ADR-018 admissibility, and a block the epistemic can be aimed at. Enforced on
        # the EFE paths; pure filtering on any partition is never blocked.
        target = _checked_info_block(backend, info_block)
        # Front-load (ADR-002): the candidate grid, the info block, the constant C.
        # R is *not* front-loaded — it may be state-dependent, so it is read per
        # candidate at μ⁺ via ``observation_noise_at`` (issue #27); a fixed sensor
        # returns the same constant each time.
        self._backend = backend
        self._target = target
        self._observation_matrix, _ = backend.observation_model  # C (constant)
        self._candidates = jnp.linspace(lo, hi, n_candidates)[:, None]

    @staticmethod
    def _argmin(g: Float64[Array, "k"]) -> Array:
        # A NaN-scoring candidate must not silently win: map NaN -> +inf so it loses.
        return jnp.argmin(jnp.where(jnp.isnan(g), jnp.inf, g))

    def select(self, belief: Belief, preference: Preference) -> Float64[Array, "p"]:
        """The grid action minimising ``G`` (the per-cycle work).

        For each candidate: predict the joint (``predicted_belief`` → ``μ⁺``, ``Σ⁺``),
        score with ``_ffg_efe_step``, ``argmin`` over the grid. ``R`` is re-read at
        ``μ⁺`` per candidate — constant under a fixed sensor, action-dependent under
        ``R(x)`` (the dual effect). This is the RFC-001 hot path; keep it lean.
        """

        def g_of(action: Float64[Array, "p"]) -> Float64[Array, ""]:
            predicted = self._backend.predicted_belief(belief, action)  # μ⁺, Σ⁺
            observation_noise = self._backend.observation_noise_at(
                predicted.mean
            )  # R(μ⁺)
            g, _ = _ffg_efe_step(
                predicted.mean,
                predicted.cov,
                self._observation_matrix,
                observation_noise,
                preference.goal,
                preference.precision,
                self._target,
            )
            return g

        g = jax.vmap(g_of)(self._candidates)
        return self._candidates[self._argmin(g)]

    @property
    def n_candidates(self) -> int:
        """The per-cycle EFE-evaluation count — attributable work (RFC-001)."""
        return self._candidates.shape[0]

    @property
    def horizon(self) -> int:
        """One-step; the H-step FFG rollout is a deferred seam."""
        return 1

    @property
    def cost_per_cycle(self) -> int:
        """Per-cycle step-evals = n_candidates (horizon is 1)."""
        return self.n_candidates

    def __repr__(self) -> str:
        """The horizon, the grid and the joint-state block the epistemic aims at."""
        lo, hi = float(self._candidates[0, 0]), float(self._candidates[-1, 0])
        return (
            f"FfgEfeSelector(horizon={self.horizon}, n_candidates={self.n_candidates}, "
            f"action_bounds=({lo}, {hi}), info_block={list(self._target)})"
        )


@dataclass(frozen=True, init=False)
class StateGoal:
    """A state-space objective: reach a target state (the LQR / fixed-sensor regime).

    The complete spec for the state-tracking path - the target plus the LQR cost
    weights it implies. ``precision`` is LQR's state weight Q; ``effort`` is its
    action weight R, left None here because the action dimension p isn't known
    until the Agent pairs this with a model (the Agent fills the identity). The
    Agent dispatches a StateGoal to an LQRSelector. Not a pytree - construction-
    time only; the Agent extracts a Preference for the selector.

    The target must hold still under the model's dynamics at zero action,
    ``A · target = target``. LQR regulates the offset from the target and charges for
    effort, so a target the dynamics move would take a constant push it never pays
    in full, and the state would settle short. The Agent refuses such a target.
    """

    target: Float64[Array, "n"]
    precision: Float64[Array, "n n"]
    effort: Float64[Array, "p p"] | None

    def __init__(self, target: ArrayLike, *, precision=None, effort=None) -> None:
        target = jnp.asarray(target, dtype=float)
        object.__setattr__(self, "target", target)
        n = target.shape[0]
        object.__setattr__(
            self,
            "precision",
            jnp.eye(n) if precision is None else jnp.asarray(precision, dtype=float),
        )
        object.__setattr__(
            self,
            "effort",
            None if effort is None else jnp.asarray(effort, dtype=float),
        )
        self._validate()

    def _validate(self) -> None:
        if self.target.ndim != 1:
            raise ValueError(
                f"target must be a 1-D vector, got shape {self.target.shape}"
            )
        validate_covariance(self.precision, "precision")
        n = self.target.shape[0]
        if self.precision.shape != (n, n):
            raise ValueError(
                f"precision must be {n}x{n} to match the {n}-D target, "
                f"got shape {self.precision.shape}"
            )


@dataclass(frozen=True, init=False)
class ObservationGoal:
    """An observation-space objective: prefer to observe a target (the EFE regime).

    The complete spec for the information-seeking path - the preferred observation, how
    sharply it is preferred (``precision``), and the action-search config the
    EFESelector front-loads: ``action_bounds`` is the action box, ``n_candidates`` its
    resolution, ``horizon`` its lookahead depth. Left as ``None``, the horizon is the
    shortest at which the target moves the scores, which the Agent reads off the model
    (ADR-068). On a double integrator observed in position that is 2.
    ``agent.selector.horizon`` reads it back. ``info_node`` optionally aims the
    epistemic term at a single latent *node* (info gain about that node's marginal, the
    factored-EFE regime on a branching backend, issue #26); ``None`` = whole-state info
    gain (the default, unchanged behaviour). The Agent dispatches an ObservationGoal to
    an EFESelector. Not a pytree - construction-time only; the Agent extracts a
    Preference.
    """

    target: Float64[Array, "m"]
    precision: Float64[Array, "m m"]
    action_bounds: tuple[float, float]
    n_candidates: int
    horizon: int | None
    info_node: int | None

    def __init__(
        self,
        target,
        *,
        action_bounds,
        precision=None,
        n_candidates=21,
        horizon=None,
        info_node=None,
    ) -> None:
        target = jnp.asarray(target, dtype=float)
        object.__setattr__(self, "target", target)
        m = target.shape[0]
        object.__setattr__(
            self,
            "precision",
            jnp.eye(m) if precision is None else jnp.asarray(precision, dtype=float),
        )
        object.__setattr__(self, "action_bounds", action_bounds)
        object.__setattr__(self, "n_candidates", n_candidates)
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(
            self, "info_node", None if info_node is None else int(info_node)
        )
        self._validate()

    def _validate(self) -> None:
        if self.target.ndim != 1:
            raise ValueError(
                f"target must be a 1-D vector, got shape {self.target.shape}"
            )
        validate_covariance(self.precision, "precision")
        m = self.target.shape[0]
        if self.precision.shape != (m, m):
            raise ValueError(
                f"precision must be {m}x{m} to match the {m}-D observation goal, "
                f"got shape {self.precision.shape}"
            )
        lo, hi = self.action_bounds
        if not lo < hi:
            raise ValueError(
                f"action_bounds must be (lo, hi) with lo < hi, got {self.action_bounds}"
            )
        if self.n_candidates < 2:
            raise ValueError(
                f"n_candidates must be at least 2 to search, got {self.n_candidates}"
            )
        if self.horizon is not None and self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.info_node is not None and self.info_node < 0:
            raise ValueError(
                f"info_node must be a non-negative node index or None, "
                f"got {self.info_node}"
            )
