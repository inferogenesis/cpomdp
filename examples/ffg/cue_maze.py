"""The cue maze, in as many spatial dimensions as you ask for.

A hidden context decides which of two goals pays. The agent's prior points at the wrong
one. A cue somewhere else in the arena reads the context sharply. The reading only
works while the agent is standing on it, so learning the truth costs a detour away from
where the agent currently believes the payoff is.

None of that is one-dimensional. This module builds it for any `n_dims >= 1`:

- **1-D** is the corridor the crossover result was measured on. Both goals and the cue
  sit on one axis, so the detour is a there-and-back walk.
- **2-D** puts the cue *perpendicular* to the line between the goals. The detour becomes
  a shape rather than a reversal, and a multi-step plan is legible on screen.
- **3-D and up** work the same way and are built the same way. Whether they are
  *affordable* is a separate question, and `enumeration_cost` answers it before you
  commit to a sweep.

The graph is one rooted tree in every dimension. Node 0 is the hidden context, a scalar.
Node 1 is the arena: the agent's position followed by its belief about where the
paying goal is. A coupling drives the goal belief from the context, and a sensor on
node 1 reads displacement, so the context is only ever seen through the coupling.

Nothing here is cpomdp API. It is a model built *with* cpomdp, in the same spirit as
`chemotaxis_model.py`. `tests/test_horizon_dimensions.py` builds its arena from it; no
demo does yet.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from cpomdp import (
    CallableGaussianObservation,
    Coupling,
    CouplingGraph,
    CouplingGraphBackend,
    GaussianCoupling,
    GaussianObservation,
    GaussianTransition,
    Preference,
)
from cpomdp.enumeration import FiniteActionSet
from cpomdp.types import Belief

CONTEXT, ARENA = 0, 1  # node indices

# --- geometry -----------------------------------------------------------------------
# Axis 0 carries the whole context. The two goals differ along it and nothing else, so
# it is the only direction the context speaks about. Every other axis is free space,
# spendable on a detour without moving toward or away from either goal.
#
# These are the corridor's published distances, and every dimension uses them. In one
# dimension `build_maze(1)` reproduces the model the crossover was measured on
# element for element: `epistemic_dissociation_figure.build_backend(cue_x=CUE_DETOUR_X)`
# agrees on A, Q, C, B, dims, the coupling and its noise, and on the whole R(x) surface.
#
#   **The frozen twins do not agree, and must not be swapped.** `epistemic_alive=False`
#   here freezes R at `_far_away()`, giving `diag([200, 200])` at n=1. The dissociation
#   demo freezes at the *prior position*, giving `diag([200, 58.68])`. Using this one as
#   the crossover's frozen-R control hands the control a 3.4x sharper info channel than
#   the registered one.
#
#   **Goals at ±4 were tried and reverted.** With the default step of 2 the reachable
#   lattice is even, so a goal at ±3 is one the agent parks a step short of rather than
#   landing on, and the animation's closing beat reads as "nearly arrived". Moving the
#   goals to ±4 fixes that and is exactly the kind of presentation tweak this directory
#   is allowed. It was measured before being kept, and it did not survive: at ±4 the
#   pragmatic stakes rise enough that the detour stops paying and the crossover
#   disappears entirely through H = 8. So the geometry stayed at ±3, where the crossover
#   is real, and the agent parking one step short is left as it is.
GOAL_DISTANCE = 3.0  # the paying goal sits at +GOAL_DISTANCE on axis 0 ...
PRIOR_GOAL = -3.0  # ... and the prior points at the other one
CUE_ON_AXIS = 1.0  # the corridor cue: off the prior path, deliberately
CUE_OFF_AXIS = 2.0  # the cue's perpendicular displacement, where there is an axis 1


def goal_distance(n_dims: int) -> float:
    """How far along axis 0 the goals sit. The same in every dimension. See the note.

    Takes `n_dims` and ignores it, matching `cue_position` and `true_goal` so callers
    can pass the arena width to all three without checking which ones care.
    """
    del n_dims  # constant by design, not by omission
    return GOAL_DISTANCE


# --- the sensor ---------------------------------------------------------------------
# `n_dims` commit channels, one per axis. Each carries the pragmatic pull toward
# wherever the agent believes the goal is. They are deliberately dull and
# action-invariant, so never trap the agent at the cue. One info channel on axis 0, the
# only axis the context can be seen through, sharp only near the cue. The epistemic
# term therefore depends on the action.
R_LO, R_HI, R_WIDTH = 0.02, 200.0, 1.2
R_GOAL = 200.0

# --- process and coupling noise (strictly positive; the information form inverts) ----
Q_CONTEXT, Q_POSITION, Q_GOAL_BELIEF = 1e-2, 1e-4, 1e-2
COUPLE_Q_POSITION = 1e3  # the coupling leaves position essentially free ...
COUPLE_Q_GOAL = 1e-2  # ... and pins the goal belief to the context

# --- prior --------------------------------------------------------------------------
PRIOR_COV_CONTEXT = 5.0  # loosely known: the agent has to detour to learn it
PRIOR_COV_POSITION = 0.05  # the agent knows where it is (dead reckoning)
PRIOR_COV_GOAL = 5.0  # the goal belief inherits the context's uncertainty

# --- objective ----------------------------------------------------------------------
GOAL_PRECISION = 0.6  # Λ on the commit channels
INFO_PRECISION = 1e-4  # ~0 on the info channel: decouples pragmatic from epistemic


def cue_position(n_dims: int) -> np.ndarray:
    """Where the cue sits, given how much room the arena has.

    In one dimension there is nowhere to put it but the goal axis, so it sits on the
    far side of the start from the prior. From 2-D up it leaves that axis. The detour
    then costs pure displacement rather than ground toward one goal or the other.

    It also has to sit somewhere the agent can actually stand. `R(x)` is sharp within
    about `R_WIDTH` of the cue, and an axis action set reaches a lattice, so a cue
    placed between lattice points is one no policy can sense from. That produces a null
    that looks exactly like "information is never worth the detour". It is nothing of
    the kind. `best_reachable_noise` is the check. This default keeps it on the lattice.
    """
    cue = np.zeros(n_dims)
    if n_dims == 1:
        cue[0] = CUE_ON_AXIS
    else:
        cue[1] = CUE_OFF_AXIS  # purely perpendicular, and on the lattice
    return cue


def axis_action_set(
    n_dims: int,
    *,
    magnitudes: tuple[float, ...] = (1.0, 2.0),
    version: str | None = None,
) -> FiniteActionSet:
    """Stay put, or step by one of `magnitudes` along one axis, in either direction.

    The canonical repertoire for a grid arena, and the reason the enumeration stays
    affordable as dimensions are added: it grows as `2·n·|magnitudes| + 1` rather than
    as a full product grid. In one dimension with the default magnitudes it is exactly
    `{0, ±1, ±2}`, the set the corridor result was measured on.

    Args:
        n_dims: how many spatial axes the arena has.
        magnitudes: the step sizes offered along each axis, in each direction.
        version: the label the completeness certificate reports. Defaults to one derived
            from the arguments, so a set that changes cannot silently keep its name.

    Returns:
        The `FiniteActionSet`, with `stay` first and then axis by axis.
    """
    if n_dims < 1:
        raise ValueError(f"n_dims must be at least 1, got {n_dims}")
    if not magnitudes:
        raise ValueError("magnitudes must offer at least one step size")

    actions = [np.zeros(n_dims)]
    for axis in range(n_dims):
        for magnitude in magnitudes:
            for sign in (+1.0, -1.0):
                step = np.zeros(n_dims)
                step[axis] = sign * magnitude
                actions.append(step)
    label = version or f"axis-{n_dims}d-{'-'.join(f'{m:g}' for m in magnitudes)}"
    return FiniteActionSet([a.tolist() for a in actions], version=label)


def enumeration_cost(
    action_set: FiniteActionSet,
    horizon: int,
    n_dims: int | None = None,
    *,
    context_dim: int = 1,
) -> tuple[int, int, float]:
    """`(policies, step_evals, peak_gib)` for one exhaustive sweep at this horizon.

    Call it before the sweep. The memory figure is the one that actually bites.
    `evaluate` `vmap`s the rollout across the *whole* policy set at once, so every
    policy simultaneously holds a predicted covariance, a posterior and an innovation.
    Adding a dimension multiplies the policy count by `((2(n+1)|m|+1) / (2n|m|+1))^H`,
    gentle for one extra axis and brutal for three.

    The estimate counts four live buffers of the dominant `n x n` carries, which is
    roughly what XLA needs to hold input and output at each stage. **Measured against a
    real run it under-reads by about 1.6x** (it predicted 4.06 GiB for a search that
    actually used ~6.7 GB), because XLA keeps more intermediates alive than that. Treat
    the number as a floor: multiply by 1.6 and leave headroom before comparing it
    against the live ceiling from `free -g`. Overshooting kills the whole WSL session.

    Args:
        action_set: the repertoire being enumerated.
        horizon: how many steps deep.
        n_dims: spatial dimensions. Defaults to the action set's own `n_controls`,
            which is what `axis_action_set` builds it from, so this one cannot drift.
        context_dim: how wide the context node is, matching `build_maze`'s argument of
            the same name. The joint is `context_dim + 2·n_dims` and the estimate is
            quadratic in it, so leaving this defaulted against a two-wide context
            under-reads by 1.44x on top of the 1.6x floor above. Nothing here can read
            it off the action set. The caller has to carry it across.

    Returns:
        The policy count, the step-evaluation count, and a peak-memory estimate in GiB.
    """
    if n_dims is None:
        n_dims = action_set.n_controls
    policies = action_set.size**horizon
    joint = context_dim + 2 * n_dims
    per_policy = 8 * (horizon * action_set.n_controls + 4 * joint * joint)
    return policies, policies * horizon, policies * per_policy / 1024**3


def best_reachable_noise(
    action_set: FiniteActionSet, n_dims: int, horizon: int, *, cue=None
) -> float:
    """The sharpest info-channel noise a reachable position gives, over `horizon` steps.

    A cue the action lattice cannot land on is a cue no policy can read. A sweep over
    that model returns "no crossover" for a reason that has nothing to do with the
    objective. Compare the result against `R_LO`. If it is not close, the task is
    unsensable by construction and any null it produces is an artefact of the geometry.

    Enumerates reachable *positions* rather than policies, so it stays cheap.
    """
    params = sensor_params(n_dims, cue)
    steps = np.asarray(action_set.actions)
    reachable = {tuple(np.zeros(n_dims))}
    for _ in range(horizon):
        reachable |= {
            tuple(np.round(np.asarray(point) + step, 9))
            for point in reachable
            for step in steps
        }
    padded = [
        jnp.asarray(np.concatenate([np.asarray(point), np.zeros(n_dims)]))
        for point in reachable
    ]
    return min(
        float(np.asarray(cue_noise(state, params))[n_dims, n_dims]) for state in padded
    )


def cue_noise(state, params):
    """`R`: dull commit channels, plus one info channel keyed on the agent's position.

    `state` is the arena node, `[position (n), goal_belief (n)]`. The falloff reads the
    agent's distance from the cue and nothing else. Far from the cue the info channel is
    as dull as the commit channels. On the cue it is sharp -- R.
    """
    n_dims = params["n_dims"]
    position = state[:n_dims]
    gap = position - params["cue"]
    falloff = 1.0 - jnp.exp(-jnp.sum(gap**2) / (2.0 * params["width"] ** 2))
    r_info = params["r_lo"] + (params["r_hi"] - params["r_lo"]) * falloff
    commit = jnp.full((n_dims,), params["r_goal"])
    return jnp.diag(jnp.concatenate([commit, r_info[None]]))


def sensor_params(n_dims: int, cue: np.ndarray | None = None) -> dict:
    """The `cue_noise` parameter bundle for an arena of this many dimensions."""
    return {
        "n_dims": n_dims,
        "cue": jnp.asarray(cue_position(n_dims) if cue is None else cue),
        "r_lo": R_LO,
        "r_hi": R_HI,
        "width": R_WIDTH,
        "r_goal": R_GOAL,
    }


def observation_matrix(n_dims: int) -> np.ndarray:
    """`C`: a commit row per axis, plus an info row repeating axis 0.

    Each commit row reads `goal_belief[i] - position[i]`. Only axis 0 can carry context
    information, because the two goals differ along axis 0 and nowhere else, so pointing
    an info channel down any other axis would cost a row and read nothing.

    **The repeated row is the mechanism, not an oversight.** `C` is rank `n_dims` over
    `n_dims + 1` rows at every dimension, and it has to be: the two rows read the same
    functional through different noise, row 0 through a fixed `R_GOAL` and row `n_dims`
    through `r_info(x)`. That is what holds the pragmatic term action-invariant while
    the epistemic term rides the action. Merge them into one full-rank row carrying the
    parallel precision and the posterior is preserved to 1e-18, the epistemic term to
    1e-15, and the pragmatic term moves by 30 to 60 nats per step — enough to flip the
    argmin cue-ward at every horizon and drag `H*` from 7 to 1.

    The cost of the deficiency is that `R` is not recoverable from a posterior here,
    only its parallel precision. Nothing in this model infers `R`; it is specified. The
    sibling guard in `crossover_horizon_figure._require_full_row_rank` is about a
    construction that does depend on that recovery, and it would refuse this `C`.
    """
    rows = np.zeros((n_dims + 1, 2 * n_dims))
    for axis in range(n_dims):
        rows[axis, axis] = -1.0  # -position[axis]
        rows[axis, n_dims + axis] = +1.0  # +goal_belief[axis]
    rows[n_dims, 0] = -1.0  # the info channel repeats axis 0 ...
    rows[n_dims, n_dims] = +1.0  # ... through its own R(x)
    return rows


def build_maze(
    n_dims: int = 2,
    *,
    epistemic_alive: bool = True,
    cue: np.ndarray | None = None,
    context_dim: int = 1,
) -> CouplingGraphBackend:
    """The cue maze as one rooted tree, in `n_dims` spatial dimensions.

    Args:
        n_dims: how many spatial axes the arena has. The joint state is
            `context_dim + 2·n_dims` wide: the context, the position, and the belief
            about where the goal is.
        epistemic_alive: with `R(x)` the info channel sharpens near the cue and the
            epistemic term moves with the action. Frozen at its value far from the cue,
            the term goes flat and the agent reduces to a pure goal-chaser, which is the
            control this demo's sibling uses.
        cue: where the cue sits. Defaults to `cue_position(n_dims)`.
        context_dim: how wide the context node is. One scalar says all the task needs,
            and that is the default. A wider context is numerically inert here — every
            EFE component is bit-identical across widths — so it exists only to exercise
            the node-shape bookkeeping against a graph whose two nodes differ in width.
            Inert in the *answer*, not in the *cost*: it widens the joint, and
            `enumeration_cost` is quadratic in that. Pass it there too.

    Returns:
        A `CouplingGraphBackend` with an `n_dims`-wide control on the position block.
    """
    if n_dims < 1:
        raise ValueError(f"n_dims must be at least 1, got {n_dims}")
    if context_dim < 1:
        raise ValueError(f"context_dim must be at least 1, got {context_dim}")

    arena_dim = 2 * n_dims
    params = sensor_params(n_dims, cue)
    observed = observation_matrix(n_dims)
    fixed_noise = np.asarray(
        cue_noise(
            jnp.asarray(np.zeros(arena_dim)), sensor_params(n_dims, _far_away(n_dims))
        )
    )
    arena_sensor = (
        CallableGaussianObservation(observed, cue_noise, params)
        if epistemic_alive
        else GaussianObservation(observed, observation_noise=fixed_noise)
    )

    # W: the context drives the goal belief on axis 0 and nothing else. A wider context
    # drives one goal-belief axis each, so the extra axes ride along without speaking.
    coupling_weights = np.zeros((arena_dim, context_dim))
    for axis in range(min(context_dim, n_dims)):
        coupling_weights[n_dims + axis, axis] = 1.0
    coupling_noise = np.diag([COUPLE_Q_POSITION] * n_dims + [COUPLE_Q_GOAL] * n_dims)
    context_to_arena = Coupling(
        parent=CONTEXT,
        child=ARENA,
        factor=GaussianCoupling(
            coupling_matrix=coupling_weights.tolist(),
            coupling_noise=coupling_noise.tolist(),
        ),
        tau=1.0,
        efe_relevant=True,  # the covariance path the instrumental epistemic rides
    )
    graph = CouplingGraph(
        root=CONTEXT,
        dims=(context_dim, arena_dim),
        couplings=(context_to_arena,),
        observations={ARENA: arena_sensor},
    )
    transitions = (
        GaussianTransition(  # a near-static context
            np.eye(context_dim).tolist(),
            dynamics_noise=(Q_CONTEXT * np.eye(context_dim)).tolist(),
        ),
        GaussianTransition(
            np.eye(arena_dim).tolist(),
            dynamics_noise=np.diag(
                [Q_POSITION] * n_dims + [Q_GOAL_BELIEF] * n_dims
            ).tolist(),
        ),
    )
    # B: the action drives position, which is the arena node's first block, and the
    # joint state puts the context in front of it.
    control_matrix = np.zeros((context_dim + arena_dim, n_dims))
    for axis in range(n_dims):
        control_matrix[context_dim + axis, axis] = 1.0
    return CouplingGraphBackend(
        graph,
        transitions,
        prior=start_belief(n_dims, context_dim=context_dim),
        control_matrix=control_matrix.tolist(),
    )


def _far_away(n_dims: int) -> np.ndarray:
    """A cue no start state is near, for freezing the sensor at its dull far value."""
    return np.full(n_dims, 1e3)


def start_belief(n_dims: int = 2, *, context_dim: int = 1) -> Belief:
    """The joint prior: known position at the origin, and a goal belief that is wrong.

    The goal belief points at the arm that does *not* pay, mirrored from a context the
    agent has guessed wrong. Chasing it takes an agent to the wrong place. Only
    resolving the context changes its mind.

    Args:
        n_dims: how many spatial axes the arena has.
        context_dim: how wide the context node is, matching `build_maze`. Every axis
            past the first starts at zero and stays uninformative.
    """
    wrong_arm = -goal_distance(n_dims)
    mean = np.zeros(context_dim + 2 * n_dims)
    mean[0] = wrong_arm  # the context, guessed wrong
    mean[context_dim + n_dims] = wrong_arm  # the goal belief mirrors that guess
    cov = np.diag(
        [PRIOR_COV_CONTEXT] * context_dim
        + [PRIOR_COV_POSITION] * n_dims
        + [PRIOR_COV_GOAL] * n_dims
    )
    return Belief(mean=mean.tolist(), cov=cov.tolist())


def preference(n_dims: int = 2) -> Preference:
    """Observe zero displacement, weighted on the commit channels only.

    The info channel gets a precision of about zero. The pragmatic term cannot leak
    into it. Anything the agent gains by standing on the cue is epistemic.
    """
    weights = [GOAL_PRECISION] * n_dims + [INFO_PRECISION]
    return Preference(goal=[0.0] * (n_dims + 1), precision=np.diag(weights).tolist())


def true_goal(n_dims: int = 2) -> np.ndarray:
    """Where the paying goal is: `goal_distance(n_dims)` on axis 0, origin elsewhere."""
    goal = np.zeros(n_dims)
    goal[0] = goal_distance(n_dims)
    return goal
