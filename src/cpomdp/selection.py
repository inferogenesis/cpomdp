"""Action selection: the seam between a belief+preference and an action."""

import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Protocol, SupportsIndex, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float64
from numpy.typing import ArrayLike

from cpomdp._validation import refuse_graph_interface, validate_covariance
from cpomdp.backends.base import GraphBackend
from cpomdp.control import LQRController, _validate_cost
from cpomdp.efe import (
    _ffg_efe_step,
    _info_block_of,
    _info_nodes,
    _target_and_precision,
    expected_free_energy,
    policy_efe,
)
from cpomdp.types import Belief, LinearGaussianModel
from warrantlib import Warrant

__all__ = [
    "ActionSelector",
    "EFESelector",
    "FFGEFESelector",
    "LQRSelector",
    "ObservationGoal",
    "Preference",
    "StateGoal",
]

_DEFAULT_N_CANDIDATES = 21
"""How many actions a grid holds when nothing says otherwise, on goal and selector."""


def _checked_action_bounds(action_bounds: tuple[float, float]) -> tuple[float, float]:
    """``action_bounds`` as two finite floats, the lower first.

    The grid is one-dimensional, so each bound is a scalar. Python's ``<`` on two
    sequences stops at the first element that differs, so a pair of per-axis bounds
    with its second axis inverted would pass a plain ``lo < hi``.
    """
    try:
        lo, hi = action_bounds
    except (TypeError, ValueError):
        raise ValueError(
            f"action_bounds must be a pair (lo, hi), got {action_bounds!r}"
        ) from None
    if np.ndim(lo) != 0 or np.ndim(hi) != 0:
        raise ValueError(
            "action_bounds must be two scalars, since the action grid is 1-D (p=1), "
            f"got {action_bounds!r}"
        )
    lo, hi = float(lo), float(hi)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f"action_bounds must be finite, got ({lo}, {hi})")
    if not lo < hi:
        raise ValueError(
            f"action_bounds must be (lo, hi) with lo < hi, got ({lo}, {hi})"
        )
    return lo, hi


def _checked_n_candidates(n_candidates: int) -> int:
    """``n_candidates`` as an int of at least 2, the total number of grid actions."""
    try:
        count = operator.index(n_candidates)
    except TypeError:
        raise TypeError(
            f"n_candidates must be an integer, got {n_candidates!r}"
        ) from None
    if count < 2:
        raise ValueError(f"n_candidates must be at least 2 to search, got {count}")
    return count


def _checked_horizon(horizon: int) -> int:
    """``horizon`` as an int of at least 1, the steps each candidate is held for."""
    try:
        steps = operator.index(horizon)
    except TypeError:
        raise TypeError(f"horizon must be an integer, got {horizon!r}") from None
    if steps < 1:
        raise ValueError(f"horizon must be at least 1, got {steps}")
    return steps


def _stored_info_node(
    info_node: SupportsIndex | Sequence[SupportsIndex] | None,
) -> int | tuple[int, ...] | None:
    """``info_node`` as an ``ObservationGoal`` keeps it: an int, a tuple, or None."""
    if info_node is None:
        return None
    nodes = _info_nodes(info_node)
    single = not isinstance(info_node, Sequence) or isinstance(info_node, str)
    return nodes[0] if single else nodes


def _candidate_grid(
    action_bounds: tuple[float, float], n_candidates: int
) -> Float64[Array, "k 1"]:
    """The evenly spaced actions a grid searches, shape ``(n_candidates, 1)``."""
    lo, hi = _checked_action_bounds(action_bounds)
    return jnp.linspace(lo, hi, _checked_n_candidates(n_candidates))[:, None]


_TIE_TOLERANCE = 1e-11
"""Scores within this of the best, relative to it, tie. The difference is roundoff.

Ten times the ``1e-12`` the graph backend leaves on a tie of size one, and far below
any difference a candidate's action makes.
"""


def _grid_choice(
    candidates: Float64[Array, "k p"],
    distance_from_zero: Float64[Array, "k"],
    g: Float64[Array, "k"],
) -> Float64[Array, "p"]:
    """The candidate with the lowest finite ``G``, or NaN when no score is finite.

    Among candidates whose scores tie, the one nearest zero wins. A tie means the action
    reaches nothing the scores read, so the least action is the honest choice. Two
    tied candidates equally near zero, such as ``±h`` on a grid without zero, give
    the point between them. Either alone would push the same way every step. The
    graph backend's canonical form leaves roundoff of about ``1e-12`` on such a tie,
    so scores within ``_TIE_TOLERANCE`` of the best count as tied. Jit-safe, and two
    more passes over ``k`` scores than a plain argmin.
    """
    scores = jnp.where(jnp.isfinite(g), g, jnp.inf)  # NaN or inf never wins
    best = jnp.min(scores)
    tied = scores <= best + _TIE_TOLERANCE * jnp.maximum(1.0, jnp.abs(best))
    distance = jnp.where(tied, distance_from_zero, jnp.inf)
    closest = jnp.min(distance)
    nearest = distance <= closest + _TIE_TOLERANCE * jnp.maximum(1.0, closest)
    choice = jnp.sum(jnp.where(nearest[:, None], candidates, 0.0), axis=0) / jnp.sum(
        nearest
    )
    return jnp.where(jnp.isfinite(best), choice, jnp.nan)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class Preference:
    """What the agent wants: a target and how sharply it is preferred.

    Every selector's ``select`` reads one. The [`Agent`][cpomdp.Agent] builds it from a
    goal's ``target`` and ``precision``, or takes it as the objective beside a selector
    of your own, in the space that selector reads.

    It is one Gaussian preference, single-mode since v0.3. The disjunctive *mixture*
    case, visiting one of several goals, is RFC-002 and deferred. A mixture
    ``Preference`` plugs in here.

    ``precision`` is unused by [`LQRSelector`][cpomdp.LQRSelector], whose controller
    bakes its own precision into the Riccati solve at construction. The EFE selectors
    read it in the pragmatic term.
    """

    target: Float64[Array, "n"]
    precision: Float64[Array, "n n"]

    def __init__(self, target: ArrayLike, precision: ArrayLike | None = None) -> None:
        target = jnp.asarray(target, dtype=float)
        if target.ndim != 1:
            raise ValueError(f"target must be a 1-D vector, got shape {target.shape}")
        object.__setattr__(self, "target", target)
        n = target.shape[0]
        object.__setattr__(
            self,
            "precision",
            jnp.eye(n) if precision is None else jnp.asarray(precision, dtype=float),
        )
        self._validate()

    def _validate(self) -> None:
        validate_covariance(self.precision, "precision")
        n = self.target.shape[0]
        if self.precision.shape != (n, n):
            raise ValueError(
                f"precision must be {n}x{n} to match the {n}-D target, "
                f"got shape {self.precision.shape}"
            )

    def tree_flatten(self):
        """Leaves: (target, precision), and no static aux. jit and vmap take one."""
        return (self.target, self.precision), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Rebuild without re-validating — the leaves may be tracers."""
        target, precision = children
        obj = object.__new__(cls)
        object.__setattr__(obj, "target", target)
        object.__setattr__(obj, "precision", precision)
        return obj


@runtime_checkable
class ActionSelector(Protocol):
    """Chooses an action from a belief and a preference.

    The abstraction wall for action selection. [`LQRSelector`][cpomdp.LQRSelector] is
    the fixed-noise case, a fixed sensor and fixed dynamics noise, where EFE collapses
    to LQR (ADR-003). [`EFESelector`][cpomdp.EFESelector] is the state-dependent one.
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
    target. EFE collapses to LQR under a fixed sensor and fixed dynamics noise
    (ADR-003), so for that regime this *is* the action selector.
    """

    controller: LQRController

    def select(self, belief: Belief, preference: Preference) -> Float64[Array, "p"]:
        """Forward to the controller: ``action(belief.mean, preference.target)``.

        No control logic of its own — the Riccati solve was front-loaded into the
        controller at construction. This just unpacks the belief and preference
        into the controller's ``(mean, target)`` signature.
        """
        return self.controller.action(belief.mean, preference.target)

    @property
    def model(self) -> LinearGaussianModel:
        """The model the controller's gain was solved for."""
        return self.controller.model


def _require_scalar_control(
    control_matrix: Float64[Array, "n p"] | None, selector: str, search: str
) -> None:
    """Refuse a model a one-dimensional action grid cannot search.

    ``search`` is where a caller with more than one action dimension goes instead.
    """
    if control_matrix is None:
        raise ValueError(
            f"{selector} needs a model with a control matrix; an action has no "
            "effect on a control-free (pure-tracking) model."
        )
    p = control_matrix.shape[1]
    if p != 1:
        raise ValueError(
            f"{selector} searches a 1-D action grid (p=1); got p={p}. "
            f"For p>1 use {search} with a declared FiniteActionSet "
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
    tie. The ``Agent`` reads the horizon for an ``ObservationGoal`` off the model and
    refuses one too short for the target to move the scores.

    Candidates whose scores tie to within roundoff go to the one nearest zero. When no
    candidate scores a finite ``G``, ``select`` returns NaN rather than an edge of the
    box. A state-dependent noise that is not a covariance at the predicted states does
    that, and [`Agent`][cpomdp.Agent] refuses the NaN.

    Honest caveat: ``horizon`` selects the best *constant* action, not the best
    *sequence*. A genuinely sequential epistemic policy — move to sense, then exploit —
    needs a varying sequence the constant-action family cannot express, so at H > 1 the
    selector can still look myopic-ish on such tasks. True varying-sequence search is
    ``EnumeratedEFESearch`` in ``cpomdp.enumeration``, which enumerates every length-H
    sequence of a declared finite action set (ADR-031).
    """

    def __init__(
        self,
        model: LinearGaussianModel,
        *,
        action_bounds: tuple[float, float],
        n_candidates: int = _DEFAULT_N_CANDIDATES,
        horizon: int = 1,
    ) -> None:
        """Front-load the candidate grid over the action box.

        Args:
            model: The model the scores are computed under. It needs a control matrix
                with one action dimension.
            action_bounds: ``(lo, hi)``, the action box the grid spans, two finite
                scalars with ``lo < hi``.
            n_candidates: How many evenly spaced actions the grid holds, in total. At
                least 2.
            horizon: How many steps each candidate is held and scored over. At least
                1.

        Raises:
            TypeError: If ``model`` is a coupled graph backend's model, which leaves
                out the couplings.
            ValueError: If the model has no control matrix or more than one action
                dimension, if ``action_bounds`` is not two finite scalars in order, if
                ``n_candidates`` is below 2, or if ``horizon`` is below 1.
            TypeError: If ``n_candidates`` or ``horizon`` is not an integer.
        """
        refuse_graph_interface(model, "EFESelector")
        _require_scalar_control(
            model.control_matrix, "EFESelector", "EnumeratedEFESearch"
        )
        candidates = _candidate_grid(action_bounds, n_candidates)
        horizon = _checked_horizon(horizon)
        self._model = model
        self._horizon = horizon
        self._candidates = candidates
        self._distance_from_zero = jnp.linalg.norm(candidates, axis=1)
        # H>1: each candidate becomes a constant-action policy (held for H steps).
        self._policies = jnp.repeat(candidates[:, None, :], horizon, axis=1)
        # Compiled once, on first use. Belief and Preference are pytree arguments, so
        # every later call reuses the program (ADR-002).
        self._scores = jax.jit(self._score_grid)
        self._choose = jax.jit(self._score_and_choose)

    def select(self, belief: Belief, preference: Preference) -> Float64[Array, "p"]:
        """The grid action minimising ``G`` over the horizon (the per-cycle work).

        At ``horizon = 1`` one ``vmap`` of the one-step kernel + ``argmin``. At
        ``horizon > 1`` one ``vmap`` of ``policy_efe`` over the constant-action policies
        + ``argmin``, returning the first (= constant) action of the best policy. NaN
        when no candidate scores a finite ``G``. The first call compiles the work and
        later calls reuse it.
        """
        _target_and_precision(preference)  # refused here, not inside the compiled call
        return self._choose(belief, preference)

    def _score_and_choose(
        self, belief: Belief, preference: Preference
    ) -> Float64[Array, "p"]:
        g = self._score_grid(belief, preference)
        return _grid_choice(self._candidates, self._distance_from_zero, g)

    def _score_grid(
        self, belief: Belief, preference: Preference
    ) -> Float64[Array, "k"]:
        """``G`` for every candidate over the horizon, in grid order."""
        if self._horizon == 1:
            return jax.vmap(
                lambda a: expected_free_energy(self._model, belief, a, preference).g
            )(self._candidates)
        return jax.vmap(lambda pol: policy_efe(self._model, belief, pol, preference).g)(
            self._policies
        )

    @property
    def candidates(self) -> Float64[Array, "k p"]:
        """The actions the grid searches, shape ``(n_candidates, p)``, lowest first."""
        return self._candidates

    @property
    def n_candidates(self) -> int:
        """How many actions the grid holds in total, each scored once per cycle."""
        return self._candidates.shape[0]

    @property
    def model(self) -> LinearGaussianModel:
        """The model every candidate is scored under."""
        return self._model

    @property
    def horizon(self) -> int:
        """How many steps each candidate's one action is held and scored over."""
        return self._horizon

    @property
    def cost_per_cycle(self) -> int:
        """Per-cycle step-evals = n_candidates * horizon."""
        return self.n_candidates * self._horizon

    @property
    def warrant(self) -> Warrant:
        """``CORROBORATED`` — the grid samples a continuum, it does not decide it.

        The counterpart to ``EnumeratedEFESearch.warrant`` (``PROVED``): the two search
        families print in different vocabulary so a sampled result is never read as a
        decided one (ADR-031).
        """
        return Warrant.CORROBORATED

    def __repr__(self) -> str:
        """The horizon and the grid, the numbers that set what each cycle costs."""
        lo, hi = float(self._candidates[0, 0]), float(self._candidates[-1, 0])
        return (
            f"EFESelector(horizon={self._horizon}, n_candidates={self.n_candidates}, "
            f"action_bounds=({lo}, {hi}))"
        )


class FFGEFESelector:
    """EFE action selection over a branching FFG backend (issue #26).

    The FFG counterpart of [`EFESelector`][cpomdp.EFESelector]: instead of
    ``_efe_step`` on a flat model it ``vmap``s ``_ffg_efe_step`` over the candidate
    action grid, reading ``μ⁺``/``Σ⁺``
    from the backend's ``predicted_belief`` (structural couplings folded in) and aiming
    the epistemic term at ``info_node``, one node or several, or the whole joint state
    when it is ``None``. One-step
    (``horizon = 1``); the H-step FFG rollout is a later seam. Conforms to
    [`ActionSelector`][cpomdp.ActionSelector].
    """

    def __init__(
        self,
        backend: GraphBackend,
        *,
        action_bounds: tuple[float, float],
        n_candidates: int = _DEFAULT_N_CANDIDATES,
        info_node: SupportsIndex | Sequence[SupportsIndex] | None = None,
    ) -> None:
        """Front-load the candidate grid, after checking the backend and the target.

        Args:
            backend: The branching backend whose predicted joint belief is scored.
            action_bounds: ``(lo, hi)``, the action box the grid spans, two finite
                scalars with ``lo < hi``.
            n_candidates: How many evenly spaced actions the grid holds, in total. At
                least 2.
            info_node: The node whose state the information term is about. ``None``
                takes the whole joint state, as on an
                [`ObservationGoal`][cpomdp.ObservationGoal].

        Raises:
            InadmissiblePartitionError: If the backend's carry partition severs an
                EFE-load-bearing edge.
            ValueError: If the backend has no control matrix or more than one action
                dimension, if ``action_bounds`` is not two finite scalars in order,
                if ``n_candidates`` is below 2, or if ``info_node`` is not a node of
                the graph.
            TypeError: If ``n_candidates`` is not an integer.
        """
        _require_scalar_control(
            backend.model.control_matrix,
            "FFGEFESelector",
            "EnumeratedEFESearch.over_backend",
        )
        candidates = _candidate_grid(action_bounds, n_candidates)
        # ADR-018 admissibility, enforced on the EFE paths. Pure filtering on any
        # partition is never blocked.
        info_block = _info_block_of(backend, info_node)
        # Front-load (ADR-002): the candidate grid, the info block, the constant C.
        # R is *not* front-loaded — it may be state-dependent, so it is read per
        # candidate at μ⁺ via ``model.observation_noise_at`` (issue #27). A fixed
        # sensor returns the same constant each time.
        self._backend = backend
        self._info_node = _stored_info_node(info_node)
        self._info_block = info_block
        self._observation_matrix = backend.model.observation_matrix  # C (constant)
        self._candidates = candidates
        self._distance_from_zero = jnp.linalg.norm(candidates, axis=1)
        self._choose = jax.jit(self._score_and_choose)  # compiled once, on first use

    def select(self, belief: Belief, preference: Preference) -> Float64[Array, "p"]:
        """The grid action minimising ``G`` (the per-cycle work).

        For each candidate: predict the joint (``predicted_belief`` → ``μ⁺``, ``Σ⁺``),
        score with ``_ffg_efe_step``, ``argmin`` over the grid. ``R`` is re-read at
        ``μ⁺`` per candidate. It is constant under a fixed sensor and moves with the
        action under ``R(x)``, the dual effect. This is the RFC-001 hot path. Ties
        and scores that are not finite go as for
        [`EFESelector`][cpomdp.EFESelector], and the first call compiles the work.
        """
        _target_and_precision(preference)  # refused here, not inside the compiled call
        return self._choose(belief, preference)

    def _score_and_choose(
        self, belief: Belief, preference: Preference
    ) -> Float64[Array, "p"]:
        def g_of(action: Float64[Array, "p"]) -> Float64[Array, ""]:
            predicted = self._backend.predicted_belief(belief, action)  # μ⁺, Σ⁺
            observation_noise = self._backend.model.observation_noise_at(
                predicted.mean
            )  # R(μ⁺)
            return _ffg_efe_step(
                predicted.mean,
                predicted.cov,
                self._observation_matrix,
                observation_noise,
                preference.target,
                preference.precision,
                self._info_block,
            ).g

        g = jax.vmap(g_of)(self._candidates)
        return _grid_choice(self._candidates, self._distance_from_zero, g)

    @property
    def candidates(self) -> Float64[Array, "k p"]:
        """The actions the grid searches, shape ``(n_candidates, p)``, lowest first."""
        return self._candidates

    @property
    def n_candidates(self) -> int:
        """How many actions the grid holds in total, each scored once per cycle."""
        return self._candidates.shape[0]

    @property
    def model(self) -> LinearGaussianModel:
        """The backend's model, whose sizes, sensor and control the score reads."""
        return self._backend.model

    @property
    def horizon(self) -> int:
        """One-step; the H-step FFG rollout is a deferred seam."""
        return 1

    @property
    def cost_per_cycle(self) -> int:
        """Per-cycle step-evals = n_candidates (horizon is 1)."""
        return self.n_candidates

    @property
    def warrant(self) -> Warrant:
        """``CORROBORATED``, as for [`EFESelector`][cpomdp.EFESelector].

        The grid samples a continuum of actions and decides nothing about it.
        """
        return Warrant.CORROBORATED

    def __repr__(self) -> str:
        """The horizon, the grid and the node the epistemic aims at."""
        lo, hi = float(self._candidates[0, 0]), float(self._candidates[-1, 0])
        return (
            f"FFGEFESelector(horizon={self.horizon}, n_candidates={self.n_candidates}, "
            f"action_bounds=({lo}, {hi}), info_node={self._info_node})"
        )


@dataclass(frozen=True, init=False)
class StateGoal:
    """A state-space objective: reach a target state (the LQR / fixed-sensor regime).

    The complete spec for the state-tracking path, the target and the LQR cost
    weights. ``precision`` is the state-cost weight and ``effort`` the action-cost
    weight, under the names [`LQRController`][cpomdp.LQRController] takes. ``effort``
    is left None here because the action dimension is unknown until the Agent pairs
    the goal with a model. The Agent then fills the identity and builds an
    LQRSelector. Not a pytree. The Agent reads it once at construction and keeps a
    Preference for the selector.

    The target must hold still under the model's dynamics at zero action,
    ``A · target = target`` with ``A`` the dynamics matrix. LQR regulates the offset
    from the target and charges for effort, so a target the dynamics move would take a
    constant push it never pays in full, and the state would settle short. The Agent
    refuses such a target.
    """

    target: Float64[Array, "n"]
    precision: Float64[Array, "n n"]
    effort: Float64[Array, "p p"] | None

    def __init__(
        self,
        target: ArrayLike,
        *,
        precision: ArrayLike | None = None,
        effort: ArrayLike | None = None,
    ) -> None:
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
        if self.effort is not None:
            # The controller inverts against it, so it must be positive-definite. Its
            # size waits for the model, where the Agent checks it.
            if self.effort.ndim != 2 or self.effort.shape[0] != self.effort.shape[1]:
                raise ValueError(
                    f"effort must be a square 2-D matrix, got shape {self.effort.shape}"
                )
            _validate_cost(self.effort, "effort", require_definite=True)


@dataclass(frozen=True, init=False)
class ObservationGoal:
    """An observation-space objective: prefer to observe a target (the EFE regime).

    The complete spec for the information-seeking path: the preferred observation, how
    sharply it is preferred (``precision``), and the grid search the Agent builds from
    it. ``action_bounds`` is the action box, two finite scalars. ``n_candidates`` is how
    many actions the grid holds in total.

    ``horizon`` is how many steps each candidate is held and scored over. A candidate
    is one action repeated, so a longer horizon looks further along a constant push.
    It never scores a plan whose action changes, such as stepping to a cue and then
    committing. That search is ``cpomdp.enumeration.EnumeratedEFESearch``, passed to
    the Agent as a selector beside a [`Preference`][cpomdp.Preference]. Left as
    ``None``, the horizon is the shortest at which the target moves the scores, which
    the Agent reads off the model (ADR-068). On a double integrator observed in
    position that is 2. ``agent.selector.horizon`` reads it back.

    ``info_node`` optionally aims the epistemic term at a latent *node*, or at several
    as their joint marginal (the factored-EFE regime on a branching backend, issue
    #26). ``None`` means whole-state info gain. The Agent dispatches an
    ObservationGoal to an EFESelector, or to an FFGEFESelector on a branching backend.
    Not a pytree. The Agent reads it once at construction and keeps its
    ``preference``, which it hands to the EFE functions.
    """

    target: Float64[Array, "m"]
    precision: Float64[Array, "m m"]
    action_bounds: tuple[float, float]
    n_candidates: int
    horizon: int | None
    info_node: int | tuple[int, ...] | None

    def __init__(
        self,
        target: ArrayLike,
        *,
        action_bounds: tuple[float, float],
        precision: ArrayLike | None = None,
        n_candidates: int = _DEFAULT_N_CANDIDATES,
        horizon: int | None = None,
        info_node: SupportsIndex | Sequence[SupportsIndex] | None = None,
    ) -> None:
        target = jnp.asarray(target, dtype=float)
        object.__setattr__(self, "target", target)
        m = target.shape[0]
        object.__setattr__(
            self,
            "precision",
            jnp.eye(m) if precision is None else jnp.asarray(precision, dtype=float),
        )
        object.__setattr__(self, "action_bounds", _checked_action_bounds(action_bounds))
        object.__setattr__(self, "n_candidates", _checked_n_candidates(n_candidates))
        object.__setattr__(
            self, "horizon", None if horizon is None else _checked_horizon(horizon)
        )
        object.__setattr__(self, "info_node", _stored_info_node(info_node))
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
        nodes = (
            ()
            if self.info_node is None
            else (self.info_node,)
            if isinstance(self.info_node, int)
            else self.info_node
        )
        if any(node < 0 for node in nodes):
            raise ValueError(
                f"info_node must be non-negative node indices or None, "
                f"got {self.info_node}"
            )

    @cached_property
    def preference(self) -> Preference:
        """The target and precision, as the preference a score is taken against.

        The [`Preference`][cpomdp.Preference] that
        [`expected_free_energy`][cpomdp.expected_free_energy] and the rollouts in
        ``cpomdp.efe`` take. The search settings stay on the goal. Built once, since
        the goal cannot change.
        """
        return Preference(self.target, self.precision)
