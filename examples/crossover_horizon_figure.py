"""R(x) and a long enough H bring curiosity back to a linear-Gaussian agent.

Under a *fixed* linear-Gaussian sensor and fixed process noise the epistemic term of
expected free energy is a constant across policies, so it cannot change any decision and
the agent reduces to LQR (the collapse `efe_collapse_figure.py` draws). A
state-dependent `R(x)` breaks that: the action reaches the noise, and curiosity becomes
possible again. Possible is not the same as decisive.

An open plane. The agent wants to be at a goal whose location it does not know. Its
prior points at `x*`. The goal is five units away from there. A beacon sits off the
straight line to `x*`, and standing near it makes the channel that reads the goal
sharp, so a detour is the only way to find out where the goal actually is. The detour
costs ground.

The animation runs the same world once per planning horizon. At a short horizon the
agent walks straight to the place it already believed in and settles there. It never
checks. Somewhere in the sweep it stops doing that and goes to the beacon first, finds
out it was wrong, and then goes to the real goal. Nothing changed but `H`: same model,
same prior, same sensor, same two candidate plans, same expected free energy.

**What the numbers on the figures are.** Expected free energy splits in two,
`G = c − ε`: a pragmatic cost `c`, how far the observations a plan expects sit from the
ones the agent prefers, and an epistemic value `ε`, how much it expects that plan to
teach it. Lower `G` is better, so a plan is worth taking when what it teaches outweighs
what it costs. Everything reported here is a *difference between the two candidate
plans*, detour minus direct:

    Δε = ε(detour) − ε(direct)      the extra information the detour buys
    Δc = c(detour) − c(direct)      the extra goal cost it pays for it
    ΔG = Δc − Δε = G(detour) − G(direct)

The difference is the whole decision. Whatever the two plans have in common cancels out
of it, so the absolute `G` of either one says nothing about which gets picked. All three
are in nats. `ΔG > 0` means the direct plan wins. `ΔG < 0` means the detour does. It is
positive at short horizons, negative at long ones, and crosses zero exactly once.

**What the planning horizon is.** `H` is how many steps of a plan the agent adds up when
it scores that plan. It is not how far it can see. The whole world is visible at every
`H`. What changes is how much of a plan's consequence is inside the window when the
agent commits to its next single step. At `H = 2` the detour cannot even reach the
beacon inside the window, so the information it would buy never appears on the balance
sheet at all. `Δε` there is under 0.01 nats.

**Why the horizon changes the answer.** The epistemic pull is *flat*. Once the horizon
is long enough to reach the beacon at all it sits at 6.67 nats and barely moves,
because sensing once is worth what sensing once is worth. What moves is the pragmatic
gradient. After the detour the sharpened goal belief lowers the expected cost of the
goal channel on *every remaining step*, and that saving accumulates until it covers the
one-off cost of walking off course. A constant pull outlasts a decaying gradient.
Nothing grows past anything.

**The control that makes it attributable.** A beacon-and-goal task on a plane looks like
the discrete cue tasks readers already know, so the whole sweep is re-run with `R` held
at a constant matched to the direct path, everything else identical. Curiosity then
never pays at any horizon in range. What freezing `R` zeroes is `Δε`, not `ε`. With a
fixed noise the covariance recursion never consults the action, so both plans carry the
identical covariance sequence, and whatever either one learns cancels out of the
difference. Each plan's own `ε` stays nonzero and grows with `H` (0.005 nats at `H = 2`,
0.042 at `H = 16`). The collapse ADR-003 records is action-invariance, not absence.
Both the state-dependent sensor and the horizon are load-bearing. Neither produces the
crossing alone.

Where the crossing lands belongs to these particular numbers. The shape carries over.

**This is not the registered `H*`.** That number comes from a complete `|A|^H`
enumeration over a declared action set on the two-node coupled cue tree, with the
epistemic term restricted to the context node (`Δε = 1.72` nats at `H = 1`), and it
carries a completeness certificate. Here two named plans are scored on a flat
four-dimensional chain with a whole-state epistemic term (`Δε = 6.67` nats), and nothing
is searched. The two land on the same integer. Nothing follows from that.

`--check` asserts what the figures claim and prints it, with no plotting deps.
Everything asserted is open-loop, so none of it takes a seed. The animated runs take
one, named in the caption::

    uv run --no-sync python examples/crossover_horizon_figure.py --check
    uv run --extra examples python examples/crossover_horizon_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import gallery
import jax
import jax.numpy as jnp
import numpy as np

from cpomdp.backends.kalman import KalmanBackend
from cpomdp.crossover import CrossoverStatistic
from cpomdp.efe import policy_efe
from cpomdp.observation import CallableSensor
from cpomdp.selection import Preference
from cpomdp.types import Belief, LinearGaussianModel

# --- the scene ----------------------------------------------------------------------
# The agent starts at the origin. Its prior says the goal is at BELIEVED_GOAL. The
# goal is really at BELIEVED_GOAL + TRUE_OFFSET. BEACON sits well off the straight
# line between the two, so visiting it buys no ground toward either.
BELIEVED_GOAL = np.array([7.0, 0.0])  # x*
TRUE_OFFSET = np.array([0.0, -5.0])  # g_true, nonzero by construction
BEACON = np.array([1.0, 3.0])  # x_b
SPEED = 1.25  # the furthest one step moves the agent

# --- the sensor ---------------------------------------------------------------------
# Two channels over the augmented state, four rows over four state dimensions, so C is
# square and invertible.
#
#   commit  o_c = p − g   how far the agent is from where the goal actually is.
#                         Constant noise. This is the channel the preference is on.
#   beacon  o_b = g       a direct reading of the latent goal, through a noise that is
#                         sharp only near x_b. This is the channel information lives on,
#                         and it carries essentially no preference weight.
#
# Keeping the two apart makes the demo about the horizon. If the preference sat on the
# same channel whose noise sharpens, then standing at the beacon would cut the expected
# goal cost directly, in one step, and the plans would flip the moment the detour
# became reachable. Geometry would decide it, and looking ahead would not enter.
R_NEAR = 0.02  # on the beacon: the goal is legible
R_DULL = 12000.0  # off it, and on the commit channel throughout: it is not
WELL_WIDTH = 1.0  # λ, how far the legible region reaches

# --- process noise and prior --------------------------------------------------------
Q_POSITION = 1e-3  # the agent tracks its own position by dead reckoning
Q_GOAL = 1e-4  # the goal is static, so this only keeps the recursion well posed
PRIOR_COV_POSITION = 0.05  # it knows where it is ...
PRIOR_COV_GOAL = 16.0  # ... and not where the goal is

# --- the objective ------------------------------------------------------------------
GOAL_PRECISION = 1.0  # Λ on the commit rows: observe zero displacement
BEACON_PRECISION = 1e-4  # ~0 on the beacon rows, so nothing pragmatic leaks onto them

# --- the sweep and the two agents ----------------------------------------------------
H_MAX = 16  # how far the horizon sweep runs
MYOPIC_H = 2  # the near-sighted agent, comfortably below the crossing
FARSIGHTED_H = 14  # the far-sighted one, comfortably above it
MONOTONE_TOL = 1e-9  # how far the frozen twin may dip and still count as monotone
# Flatness is a per-step bound, not a bound on the range. The pull drifts down at a
# constant ~0.0026 nats per horizon, so any range test passes or fails on H_MAX rather
# than on the model: at FLAT_PULL = 0.05 across the whole window the same claim would
# start failing from H_MAX = 23 with nothing about the agent having changed.
FLAT_PULL_PER_STEP = 0.01  # how far Δε may move between consecutive horizons
PULL_LEVEL = 6.67  # the level the plateau holds at, which the prose quotes
PULL_LEVEL_TOL = 0.05  # and how far off it the mean may sit
MYOPIC_PULL_MAX = 0.01  # Δε below the beacon's reach, where the prose says ~nothing

CONTEXT_DIMS = 2  # the plane
STATE_DIMS = 2 * CONTEXT_DIMS  # (position, goal)


def observation_matrix() -> np.ndarray:
    """`C`: a commit row per axis reading `p − g`, then a row per axis reading `g`.

    Square and invertible, so the sensor carries no redundant channel. Position stays
    observable through the pair, since `p = o_c + o_b`.
    """
    rows = np.zeros((STATE_DIMS, STATE_DIMS))
    for axis in range(CONTEXT_DIMS):
        rows[axis, axis] = +1.0  # +position[axis] ...
        rows[axis, CONTEXT_DIMS + axis] = -1.0  # ... − goal[axis]
        rows[CONTEXT_DIMS + axis, CONTEXT_DIMS + axis] = 1.0  # the beacon channel
    return rows


def beacon_noise(state, params):
    """`R(x)`: dull commit rows, plus beacon rows keyed on the agent's position.

    The falloff reads the distance from the beacon and nothing else, so `R` depends on
    the position block alone. That is the block the control moves. That dependence is
    the whole coupling: the action chooses where the noise is read, and through it the
    posterior.

    Module-level and `jnp`-only so it can ride in a ``CallableSensor``'s static aux.
    """
    gap = state[:CONTEXT_DIMS] - params["beacon"]
    falloff = 1.0 - jnp.exp(-jnp.sum(gap**2) / (2.0 * params["width"] ** 2))
    sharpness = params["r_near"] + (params["r_far"] - params["r_near"]) * falloff
    commit = jnp.full((CONTEXT_DIMS,), params["r_commit"])
    return jnp.diag(jnp.concatenate([commit, jnp.full((CONTEXT_DIMS,), sharpness)]))


def sensor_params() -> dict:
    """The `beacon_noise` parameter bundle for this scene."""
    return {
        "beacon": jnp.asarray(BEACON),
        "r_near": R_NEAR,
        "r_far": R_DULL,
        "r_commit": R_DULL,
        "width": WELL_WIDTH,
    }


def build_model(*, frozen_noise: float | None = None) -> LinearGaussianModel:
    """The augmented plane, live or with its beacon channel frozen.

    Args:
        frozen_noise: if given, the constant the beacon rows of `R` are pinned at, and
            the sensor becomes an ordinary fixed one. The `None` default is the live
            `R(x)`.

    Returns:
        The `LinearGaussianModel`. A = I, control on the position block only.
    """
    dynamics_matrix = np.eye(
        STATE_DIMS
    )  # A: position drifts nowhere, the goal is static
    control_matrix = np.vstack(
        [np.eye(CONTEXT_DIMS), np.zeros((CONTEXT_DIMS, CONTEXT_DIMS))]
    )
    dynamics_noise = np.diag(  # Q
        [Q_POSITION] * CONTEXT_DIMS + [Q_GOAL] * CONTEXT_DIMS
    )
    observed = observation_matrix()  # C
    _require_full_row_rank(observed)
    # The live sensor carries C and R(x) itself. The frozen twin states both plainly.
    live = frozen_noise is None
    pinned = R_DULL if frozen_noise is None else frozen_noise
    frozen = np.diag([R_DULL] * CONTEXT_DIMS + [pinned] * CONTEXT_DIMS)
    return LinearGaussianModel(
        dynamics_matrix=dynamics_matrix.tolist(),
        observation_matrix=None if live else observed.tolist(),
        dynamics_noise=dynamics_noise.tolist(),
        observation_noise=None if live else frozen.tolist(),
        prior=start_belief(),
        control_matrix=control_matrix.tolist(),
        observation_model=(
            CallableSensor(observed, beacon_noise, sensor_params()) if live else None
        ),
    )


def _require_full_row_rank(observed: np.ndarray) -> None:
    """Refuse to build *this* model with a sensor that carries a redundant channel.

    The frozen twin here is pinned by matching a posterior along the direct path, and
    without full row rank only `R`'s action on the row space of `C` is identified. The
    covariance off that subspace is then free, so the constant the twin is pinned at
    stops being determined by the thing it is meant to reproduce.

    Local to this construction, not a rule about sensors. A redundant row is the right
    design wherever two channels are wanted on one functional with different noise —
    `ffg/cue_maze.py` does exactly that on purpose, and would fail this check. Cheap
    enough to pay at every construction, and construction is not the loop.
    """
    rank = int(np.linalg.matrix_rank(observed))
    if rank < observed.shape[0]:
        raise ValueError(
            f"the sensor C must have full row rank. Got rank {rank} for "
            f"{observed.shape[0]} channels over {observed.shape[1]} states"
        )


def start_belief() -> Belief:
    """Known position at the origin. A goal belief centred on `x*`, and wrong.

    The prior over the offset `g − x*` is therefore zero-mean with covariance
    `PRIOR_COV_GOAL`, and the truth sits 1.25 standard deviations away from it.
    """
    mean = np.concatenate([np.zeros(CONTEXT_DIMS), BELIEVED_GOAL])
    cov = np.diag([PRIOR_COV_POSITION] * CONTEXT_DIMS + [PRIOR_COV_GOAL] * CONTEXT_DIMS)
    return Belief(mean=mean.tolist(), cov=cov.tolist())


def preference() -> Preference:
    """Observe zero displacement, weighted on the commit rows only.

    The beacon rows get a precision of about zero, so nothing the agent gains by
    standing at the beacon is pragmatic. The contrast between the two terms is then a
    matter of construction rather than of interpretation.
    """
    weights = [GOAL_PRECISION] * CONTEXT_DIMS + [BEACON_PRECISION] * CONTEXT_DIMS
    return Preference(target=[0.0] * STATE_DIMS, precision=np.diag(weights).tolist())


def true_goal() -> np.ndarray:
    """Where the goal actually is. `x* + g_true`, which the prior does not know."""
    return BELIEVED_GOAL + TRUE_OFFSET


# --- the two plans -------------------------------------------------------------------
# Both are open-loop and constant-target: a fixed list of waypoints, walked at SPEED.
# Neither consults an observation, so the margin between them carries no seed.
def _walk(start, waypoints, horizon: int) -> np.ndarray:
    """Step from `start` through `waypoints` in order, `SPEED` at a time, `horizon` up.

    The waypoint list is walked greedily: each step closes on the first target not yet
    reached, so a plan is fully determined by its waypoints and its length.
    """
    position = np.asarray(start, dtype=float).copy()
    queue = [np.asarray(w, dtype=float) for w in waypoints]
    moves = []
    for _ in range(horizon):
        while len(queue) > 1 and np.linalg.norm(queue[0] - position) < 1e-9:
            queue.pop(0)
        gap = queue[0] - position
        distance = float(np.linalg.norm(gap))
        step = gap if distance <= SPEED else gap / distance * SPEED
        moves.append(step)
        position = position + step
    return np.asarray(moves)


def direct_policy(horizon: int) -> np.ndarray:
    """Drive at `x*`, the place the prior already believes in, and hold."""
    return _walk(np.zeros(CONTEXT_DIMS), [BELIEVED_GOAL], horizon)


def detour_policy(horizon: int) -> np.ndarray:
    """Route via the beacon, then on to `x*`. The plan that buys information."""
    return _walk(np.zeros(CONTEXT_DIMS), [BEACON, BELIEVED_GOAL], horizon)


def beacon_arrival() -> int:
    """The step the detour first stands on the beacon. The geometry's own number.

    Worth reporting beside the crossing, because they are different facts. A crossing
    that landed here would say only that the detour had become reachable in time.
    """
    return int(np.ceil(float(np.linalg.norm(BEACON)) / SPEED))


# --- the margin ---------------------------------------------------------------------
def margin(horizon: int, *, frozen_noise: float | None = None) -> CrossoverStatistic:
    """`ΔG(H)` and its split into an epistemic pull and a pragmatic gradient.

    Sign convention borrowed from `cpomdp.crossover` so it cannot drift: the detour
    plays the walk and the direct plan the reach. That gives
    `Δε = ε(detour) − ε(direct)`, `Δc = c(detour) − c(direct)` and
    `ΔG = Δc − Δε = G(detour) − G(direct)`. Negative is the crossover. The rollout is
    the flat one (`policy_efe`), since this model is a single augmented chain with no
    couplings.
    """
    model = build_model(frozen_noise=frozen_noise)
    goal, belief = preference(), start_belief()
    detour = policy_efe(model, belief, jnp.asarray(detour_policy(horizon)), goal)
    direct = policy_efe(model, belief, jnp.asarray(direct_policy(horizon)), goal)
    delta_epsilon = detour.epistemic - direct.epistemic
    delta_c = detour.pragmatic - direct.pragmatic
    return CrossoverStatistic(horizon, delta_epsilon, delta_c, delta_c - delta_epsilon)


def margin_curve(
    max_horizon: int = H_MAX, *, frozen_noise: float | None = None
) -> list[CrossoverStatistic]:
    """`ΔG(H)` for every horizon in `1..max_horizon`."""
    return [margin(h, frozen_noise=frozen_noise) for h in range(1, max_horizon + 1)]


def first_crossing(curve: list[CrossoverStatistic]) -> int | None:
    """The first horizon at which the detour wins, or `None` if it never does.

    Deliberately not named `crossover_horizon`. That name belongs to
    `cpomdp.crossover`, where it means `H*` over an enumerated search, and this is a
    scan of two named plans over a precomputed curve.
    """
    return next((stat.horizon for stat in curve if stat.walk_wins), None)


def sign_changes(values) -> int:
    """How many times a swept margin changes sign."""
    signs = np.sign(np.asarray(values, dtype=float))
    return int(np.sum(signs[1:] != signs[:-1]))


def frozen_reference_noise(horizon: int = H_MAX) -> float:
    """The constant the twin pins `R`'s beacon rows at.

    The trajectory-average of the live beacon noise along the *direct* path, over the
    swept range. That is the path taken below the crossing. That path never comes near
    the beacon, so the average sits just under the dull far-field value, which is the
    point: the twin is the sensor the near-sighted agent actually experiences, held
    still.

    It averages `pinned_noise` rather than walking the path a second time. Sampling
    `R(x)` twice invites the two samplers to drift apart in which state they hand it,
    and today they would disagree silently: `beacon_noise` reads the position block
    alone, so a wrong goal block is invisible until the well ever keys on one.
    """
    return float(np.mean(pinned_noise(direct_policy(horizon))))


def pinned_noise(policy: np.ndarray) -> np.ndarray:
    """`R(μ⁻_k)` along a plan. The noise the sensor presents at each predicted mean.

    Two plans that pin different sequences cannot both be reproduced by one constant,
    so the frozen twin is a control rather than a restatement.
    """
    params = sensor_params()
    position = np.zeros(CONTEXT_DIMS)
    readings = []
    for step in policy:
        position = position + step  # μ⁻ = A·μ + B·a, and A = I on the position block
        state = jnp.asarray(np.concatenate([position, BELIEVED_GOAL]))
        readings.append(float(np.asarray(beacon_noise(state, params))[-1, -1]))
    return np.asarray(readings)


# --- the closed loop the animation draws ---------------------------------------------
def simulate(horizon: int, *, seed: int, n_steps: int = 12) -> dict:
    """One agent through the scene. It chooses its plan once, then tracks its belief.

    The choice between the two plans is made once, at the start, at this agent's
    horizon. That is the object the margin is about, and the only place the horizon
    enters. Execution is closed-loop: every step draws a real observation through
    `R(x)`, filters it, and steers at wherever the agent believes the goal is. So an
    agent that senses can act on what it learned, and one that does not keeps walking
    at a belief that never moved.

    A rolling re-choice is a different and equally honest agent, and it does not show
    this: from a point already part-way along, the detour is cheap enough that a
    near-sighted agent takes it too. The margin is stated from the prior, so the run
    that illustrates it commits from the prior.

    Returns:
        The track, and the goal belief aligned with it. Index `k` is the belief the
        agent holds standing at `track[k]`, with index 0 the prior.
    """
    model = build_model()
    backend = KalmanBackend(model)
    belief = start_belief()
    goal, truth = preference(), true_goal()

    def score(policy):
        return float(policy_efe(model, belief, jnp.asarray(policy), goal).g)

    detours = score(detour_policy(horizon)) < score(direct_policy(horizon))

    rng = np.random.default_rng(seed)
    position = np.zeros(CONTEXT_DIMS)
    sensed = not detours  # the beacon leg, once done, stays done
    track = [position.copy()]
    guesses = [np.asarray(belief.mean)[CONTEXT_DIMS:].copy()]
    spreads = [np.asarray(belief.cov)[CONTEXT_DIMS:, CONTEXT_DIMS:].copy()]

    for _ in range(n_steps):
        if not sensed:
            action = _walk(position, [BEACON], 1)[0]
            sensed = bool(np.linalg.norm(position + action - BEACON) < 1e-9)
        else:
            action = _walk(position, [np.asarray(belief.mean)[CONTEXT_DIMS:]], 1)[0]
        position = position + action
        state = jnp.asarray(np.concatenate([position, truth]))
        noise = np.asarray(beacon_noise(state, sensor_params()))
        draw = np.linalg.cholesky(noise) @ rng.standard_normal(STATE_DIMS)
        observation = observation_matrix() @ np.concatenate([position, truth]) + draw
        belief = backend.infer_states(
            jnp.asarray(observation), belief, action=jnp.asarray(action)
        )
        track.append(position.copy())
        guesses.append(np.asarray(belief.mean)[CONTEXT_DIMS:].copy())
        spreads.append(np.asarray(belief.cov)[CONTEXT_DIMS:, CONTEXT_DIMS:].copy())

    return {
        "horizon": horizon,
        "detours": detours,
        "track": np.asarray(track),
        "guesses": np.asarray(guesses),
        "spreads": np.asarray(spreads),
    }


# --- the gate ------------------------------------------------------------------------
def check() -> None:
    """Assert what the figures claim, and print the sweep they are read off.

    Everything here is open-loop and planning-level, so none of it takes a seed. The
    animated runs do, and their seed is named in the caption rather than pinned here.
    """
    live = margin_curve()
    frozen_at = frozen_reference_noise()
    frozen = margin_curve(frozen_noise=frozen_at)
    crossing = first_crossing(live)
    delta = np.array([float(stat.delta_g) for stat in live])
    frozen_delta = np.array([float(stat.delta_g) for stat in frozen])
    frozen_pull = np.array([abs(float(stat.delta_epsilon)) for stat in frozen])
    reached = beacon_arrival()
    pull = np.array([float(stat.delta_epsilon) for stat in live[reached - 1 :]])
    direct_pins = pinned_noise(direct_policy(H_MAX))
    detour_pins = pinned_noise(detour_policy(H_MAX))

    _print_sweep(live, frozen, crossing, frozen_at, direct_pins, detour_pins)
    # The three ways the sweep can leave a verdict with nothing to read off. Every
    # verdict below slices the curve either side of the crossing, or from the beacon's
    # first reachable step, and an empty slice reaches numpy as a bare reduction error.
    # Report the geometry that caused it instead. Retune the scene constants and one of
    # these is how you find out.
    if crossing is None:
        raise AssertionError(
            f"no crossing in H = 1..{H_MAX}: ΔG never turns negative "
            f"(smallest {delta.min():+.2f} nats)"
        )
    if crossing == 1:
        raise AssertionError(
            f"the crossing sits at H = 1, so no horizon below it is in the sweep and "
            f"the claims about what happens below it read off nothing "
            f"(ΔG(1) = {delta[0]:+.2f} nats)"
        )
    if reached > H_MAX:
        raise AssertionError(
            f"the detour first stands on the beacon at step {reached}, past the end of "
            f"the H = 1..{H_MAX} sweep, so the epistemic pull is never measured"
        )
    results = [
        (
            "the two agents in the animation choose differently, and the horizon is "
            "the only thing that differs between them",
            not live[MYOPIC_H - 1].walk_wins and live[FARSIGHTED_H - 1].walk_wins,
            f"H = {MYOPIC_H} takes the direct plan (ΔG = "
            f"{delta[MYOPIC_H - 1]:+.2f} nats), H = {FARSIGHTED_H} the detour "
            f"(ΔG = {delta[FARSIGHTED_H - 1]:+.2f}). Both choices are read off the "
            "same prior, so neither takes a draw",
        ),
        (
            "the direct plan wins at every horizon below the crossing",
            bool(np.all(delta[: crossing - 1] > 0)),
            f"H = 1..{crossing - 1}, smallest ΔG = "
            f"{delta[: crossing - 1].min():+.2f} nats",
        ),
        (
            "the detour wins at every horizon from the crossing to the end of the "
            "sweep",
            bool(np.all(delta[crossing - 1 :] < 0)),
            f"H = {crossing}..{H_MAX}, largest ΔG = "
            f"{delta[crossing - 1 :].max():+.2f} nats",
        ),
        (
            "the margin changes sign exactly once, so there is one crossing to point "
            "at",
            sign_changes(delta) == 1,
            f"ΔG({crossing - 1}) = {delta[crossing - 2]:+.2f} → ΔG({crossing}) = "
            f"{delta[crossing - 1]:+.2f}. The detour first reaches the beacon at step "
            f"{reached}, so the crossing is not it becoming reachable",
        ),
        (
            "the epistemic pull is flat, so what moves is the pragmatic term",
            float(np.max(np.abs(np.diff(pull)))) < FLAT_PULL_PER_STEP
            and abs(float(pull.mean()) - PULL_LEVEL) < PULL_LEVEL_TOL,
            f"Δε holds at {pull.mean():.2f} nats from H = {reached} to H = {H_MAX}, "
            f"moving {float(np.max(np.abs(np.diff(pull)))):.4f} per step",
        ),
        (
            "below the beacon's reach the detour buys essentially nothing, so the "
            "information it would find never enters the balance sheet",
            abs(float(live[MYOPIC_H - 1].delta_epsilon)) < MYOPIC_PULL_MAX,
            f"Δε(H = {MYOPIC_H}) = {float(live[MYOPIC_H - 1].delta_epsilon):.4f} nats, "
            f"and the detour first stands on the beacon at step {reached}",
        ),
        (
            "the two plans read genuinely different noise, which is why any of this "
            "happens",
            float(np.max(np.abs(direct_pins - detour_pins))) > 1.0,
            f"R along the detour reaches {detour_pins.min():.2f}. Along the direct "
            f"plan it never falls below {direct_pins.min():.0f}",
        ),
        (
            "freeze R and the detour never wins at any horizon in range",
            sign_changes(frozen_delta) == 0
            and bool(frozen_delta.min() > 0)
            and _is_monotone(frozen_delta),
            f"frozen ΔG stays in [{frozen_delta.min():+.2f}, "
            f"{frozen_delta.max():+.2f}], never falling and never crossing. It climbs "
            f"to {frozen_delta.max():+.2f} and then saturates, both plans having run "
            f"out of ground to differ over",
        ),
        (
            "freezing R zeroes the epistemic contrast, rather than shrinking it",
            float(frozen_pull.max()) == 0.0,
            f"largest |Δε| under a frozen sensor is {frozen_pull.max():.1f}. The "
            "covariance recursion stops consulting the action, so both plans carry "
            "the identical covariance sequence. Neither plan's own ε is zero, and "
            "both grow with H. What collapses is the difference",
        ),
    ]

    print()
    for label, passed, detail in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}\n         {detail}")
    print(
        f"\nThe crossing sits at H = {crossing}: below it curiosity does not pay for "
        f"itself, above it it does.\nThat number belongs to these parameters and "
        f"nothing else. What travels is the shape.\n"
        f"\nNot the registered H*. That one is an exhaustive search over a declared "
        f"action set on the\ncoupled cue tree, with the epistemic term restricted to "
        f"the context node. This is two named\nplans on a flat chain, whole-state, "
        f"searching nothing. The integers coincide and mean\nnothing by it."
    )
    for label, passed, _ in results:
        assert passed, label


def _is_monotone(values: np.ndarray) -> bool:
    """Whether a swept margin never turns back on itself."""
    return bool(np.all(np.diff(values) >= -MONOTONE_TOL))


def _print_sweep(live, frozen, crossing, frozen_at, direct_pins, detour_pins) -> None:
    """The sweep table the assertions are read off, printed before the verdicts."""
    print(
        f"ΔG(H) = G(detour, H) − G(direct, H) in nats, and its frozen-R twin "
        f"(R pinned at {frozen_at:.0f})\n"
    )
    print(
        f"  {'H':>2} {'Δε (pull)':>11} {'Δc (gradient)':>14} {'ΔG (margin)':>12}"
        f" {'ΔG frozen':>11}   the better plan"
    )
    for stat, twin in zip(live, frozen, strict=True):
        mark = "  <-- the crossing" if stat.horizon == crossing else ""
        print(
            f"  {stat.horizon:>2} {float(stat.delta_epsilon):>11.4f} "
            f"{float(stat.delta_c):>14.4f} {float(stat.delta_g):>12.4f} "
            f"{float(twin.delta_g):>11.3f}   "
            f"{'detour' if stat.walk_wins else 'direct':<8}{mark}"
        )
    print(
        f"\n  R(μ⁻) along the direct plan: {direct_pins.min():.0f}..."
        f"{direct_pins.max():.0f}    along the detour: {detour_pins.min():.2f}..."
        f"{detour_pins.max():.0f}"
    )


# --- rendering ------------------------------------------------------------------------
# The science above is frozen. Everything below is presentation. Drawn in the gallery's
# idiom, with the same palette and the same body glyph the other demos use. Nothing here
# feeds back.
BG, INK = gallery.PAPER.bg, gallery.PAPER.ink
GRID, FAINT = gallery.PAPER.grid, gallery.PAPER.faint
BEACON_C = gallery.BLUE  # the beacon and the region it makes legible
TRUE_C = gallery.VERMILLION  # where the goal actually is
GUESS_C = gallery.ORANGE  # where the agent believes it is
SIGMA_C = gallery.SKY  # how sure it is about that
MYOPIC_C = "#9AA0A6"  # grey: the near-sighted agent
FARSIGHTED_C = gallery.GREEN  # green: the far-sighted one
MARGIN_C = gallery.G_C  # the margin itself
XLIM, YLIM = (-1.9, 9.6), (-6.8, 4.8)
SEED = 11  # the animated runs' only seed, named in both captions
N_STEPS = 12  # long enough to sense, turn, and arrive

MYOPIC = gallery.BacillusStyle(body=MYOPIC_C, ink=INK, length=0.74, width=0.35)
FARSIGHTED = gallery.BacillusStyle(body=FARSIGHTED_C, ink=INK, length=0.74, width=0.35)

# The animation is a sweep. The same agent runs the same world once per horizon, and
# the ladder on the right fills in as it goes. Consecutive horizons bracket the
# crossing, so the row where "direct" becomes "detour" arrives while the reader is
# watching rather than being announced.
H_LADDER = (2, 4, 6, 7, 10, 14)
DIRECT_STEPS = 7  # a direct run is at the goal it believed in by here ...
DETOUR_STEPS = 12  # ... and a detour needs this many to sense, turn, and arrive
TRAVEL_FRAMES = 2  # eased frames between one step and the next
RUN_END_HOLD = 8  # a beat on the outcome before the next horizon starts
FPS = 14

_FIELD_CACHE: list = [None]  # one pass over R(x). Every frame reads it


def _sharpness_field(resolution: int = 180):
    """Where the goal is legible: `−ln R_beacon` sampled over the plane.

    Returned with the contour levels it should be drawn at. They start one nat above
    the dull floor rather than at it, so the far field is left unfilled: with a well
    this narrow, linear levels across the whole range put every band inside a radius of
    less than one and wash the rest of the panel in the base colour, which reads as
    "somewhat legible everywhere" and is the opposite of the truth.
    """
    xs = np.linspace(*XLIM, resolution)
    ys = np.linspace(*YLIM, resolution)
    params = sensor_params()
    field = np.empty((resolution, resolution))
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            state = jnp.asarray([x, y, 0.0, 0.0])
            field[j, i] = -np.log(float(beacon_noise(state, params)[-1, -1]))
    levels = np.linspace(-np.log(R_DULL) + 1.0, -np.log(R_NEAR), 7)
    return xs, ys, field, levels


def _heading(track: np.ndarray, upto: float) -> np.ndarray:
    """Which way the body points at a fractional step: the move it is making."""
    here = min(int(np.floor(upto)), len(track) - 1)
    step = np.asarray(track[min(here + 1, len(track) - 1)]) - np.asarray(track[here])
    if float(np.hypot(*step)) > 1e-6:
        return step
    if here >= 1:
        return np.asarray(track[here]) - np.asarray(track[here - 1])
    return np.array([1.0, 0.0])


def _at(run: dict, upto: float) -> dict:
    """The agent's drawable state at a fractional step, interpolated between two."""
    here = min(int(np.floor(upto)), len(run["track"]) - 1)
    nxt = min(here + 1, len(run["track"]) - 1)
    blend = float(upto - here)
    return {
        "position": np.asarray(
            [
                gallery.lerp(run["track"][here][k], run["track"][nxt][k], blend)
                for k in range(CONTEXT_DIMS)
            ]
        ),
        "guess": np.asarray(
            [
                gallery.lerp(run["guesses"][here][k], run["guesses"][nxt][k], blend)
                for k in range(CONTEXT_DIMS)
            ]
        ),
        "spread": run["spreads"][here],
        "trail": run["track"][: here + 1],
        "heading": _heading(run["track"], upto),
    }


def _ladder_runs(seed: int = SEED) -> list[dict]:
    """One closed-loop run per rung of the ladder, with the margin that decided it.

    Each run is drawn only for as long as it is still doing something: a direct run has
    arrived where it believed the goal was within `DIRECT_STEPS`, and after that it only
    jitters against a belief that never moved.
    """
    runs = []
    for horizon in H_LADDER:
        run = simulate(horizon, seed=seed, n_steps=DETOUR_STEPS)
        run["stat"] = margin(horizon)
        run["drawn_steps"] = DETOUR_STEPS if run["detours"] else DIRECT_STEPS
        runs.append(run)
    return runs


def _sweep_frames(runs: list[dict]) -> list[tuple[int, float]]:
    """`(rung, fractional step)` for every frame: each run walked, then held."""
    frames: list[tuple[int, float]] = []
    for rung, run in enumerate(runs):
        frames.append((rung, 0.0))
        for step in range(run["drawn_steps"]):
            frames.extend(
                (rung, step + gallery.ease(t / TRAVEL_FRAMES))
                for t in range(1, TRAVEL_FRAMES + 1)
            )
        frames.extend((rung, float(run["drawn_steps"])) for _ in range(RUN_END_HOLD))
    return frames


def _draw_ladder(ax, runs: list[dict], rung: int) -> None:
    """The right-hand column: the horizons tried so far, and this one's arithmetic.

    Static furniture, so the animation never has to stop for it to be read. Rows the
    sweep has not reached yet are left as dashes rather than filled in faintly. The row
    where `direct` becomes `detour` is the whole point, and showing it early gives it
    away.

    Every number is a *difference between the two plans*, detour minus direct, in nats.
    The difference is what decides the choice. Whatever the two plans share cancels out
    of it, so the absolute EFE of either one is not worth the space.
    """
    from matplotlib.patches import Rectangle

    ax.set_facecolor(BG)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    run = runs[rung]
    stat = run["stat"]
    colour = FARSIGHTED_C if run["detours"] else MYOPIC_C

    ax.text(
        0.0,
        0.995,
        "same model, same prior, same sensor.",
        fontsize=8.4,
        color=FAINT,
        ha="left",
        va="top",
    )
    ax.text(
        0.0,
        0.945,
        "Only the planning horizon H changes.",
        fontsize=8.4,
        color=FAINT,
        ha="left",
        va="top",
    )
    ax.text(
        0.0,
        0.885,
        f"H = {run['horizon']}",
        fontsize=20,
        fontweight="bold",
        color=colour,
        ha="left",
        va="top",
    )
    ax.text(
        0.30,
        0.845,
        f"it adds up the next {run['horizon']} steps\nof a plan before taking one",
        fontsize=8.2,
        color=FAINT,
        ha="left",
        va="top",
    )

    top, row_height = 0.735, 0.055
    ax.text(0.02, top, "H", fontsize=8.2, color=FAINT, ha="right", va="bottom")
    ax.text(0.12, top, "it picks", fontsize=8.2, color=FAINT, ha="left", va="bottom")
    ax.text(0.99, top, "ΔG  (nats)", fontsize=8.2, color=FAINT, ha="right", va="bottom")
    ax.plot([0.0, 1.0], [top - 0.012] * 2, color=GRID, lw=1.0)

    for index, other in enumerate(runs):
        y = top - 0.048 - index * row_height
        seen = index <= rung
        shade = (FARSIGHTED_C if other["detours"] else MYOPIC_C) if seen else FAINT
        weight = "bold" if index == rung else "normal"
        if index == rung:
            ax.add_patch(
                Rectangle(
                    (-0.03, y - 0.015),
                    1.06,
                    row_height * 0.82,
                    facecolor=colour,
                    alpha=0.13,
                    lw=0,
                    zorder=0,
                )
            )
        ax.text(
            0.02,
            y,
            str(other["horizon"]),
            fontsize=9.4,
            color=shade,
            ha="right",
            va="baseline",
            fontweight=weight,
            family="monospace",
        )
        ax.text(
            0.12,
            y,
            ("detour" if other["detours"] else "direct") if seen else "—",
            fontsize=9.4,
            color=shade,
            ha="left",
            va="baseline",
            fontweight=weight,
        )
        ax.text(
            0.99,
            y,
            f"{float(other['stat'].delta_g):+.1f}" if seen else "—",
            fontsize=9.4,
            color=shade,
            ha="right",
            va="baseline",
            fontweight=weight,
            family="monospace",
        )

    split = top - 0.048 - len(runs) * row_height - 0.055
    ax.plot([0.0, 1.0], [split + 0.052] * 2, color=GRID, lw=1.0)
    ax.text(
        0.0,
        split + 0.014,
        "how that score splits, detour − direct, in nats:",
        fontsize=8.2,
        color=FAINT,
        ha="left",
        va="baseline",
    )
    # Signed differences, so the labels have to stay neutral: past the crossing Δc goes
    # negative and "what the detour costs" becomes false.
    rows = (
        ("Δε  epistemic — what it learns", stat.delta_epsilon, gallery.EPISTEMIC_C),
        ("Δc  pragmatic — goal cost", stat.delta_c, gallery.PRAGMATIC_C),
        ("ΔG  = Δc − Δε", stat.delta_g, MARGIN_C),
    )
    for index, (label, value, tint) in enumerate(rows):
        y = split - 0.028 - index * row_height
        bold = "bold" if index == 2 else "normal"
        ax.text(
            0.0,
            y,
            label,
            fontsize=9.0,
            color=tint,
            ha="left",
            va="baseline",
            fontweight=bold,
        )
        ax.text(
            0.99,
            y,
            f"{float(value):+.2f}",
            fontsize=9.4,
            color=tint,
            ha="right",
            va="baseline",
            fontweight=bold,
            family="monospace",
        )
    verdict = (
        "ΔG < 0, so the detour is the better plan"
        if run["detours"]
        else "ΔG > 0, so going straight is the better plan"
    )
    ax.text(
        0.0,
        split - 0.028 - 3.1 * row_height,
        verdict,
        fontsize=9.2,
        color=colour,
        ha="left",
        va="baseline",
        fontweight="bold",
    )


def _draw_world(ax, field, *, show_prior: bool) -> None:
    """The parts of the scene that never move: the legible region and the goal."""
    from matplotlib.patches import Circle, Ellipse

    xs, ys, sharpness, levels = field
    truth = true_goal()

    ax.set_facecolor(BG)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID)

    ax.contourf(xs, ys, sharpness, levels=levels, cmap="Blues", alpha=0.55, zorder=0)
    ax.add_patch(
        Circle(
            gallery.xy(BEACON),
            2.0 * WELL_WIDTH,
            facecolor="none",
            edgecolor=BEACON_C,
            lw=1.0,
            ls=(0, (3, 3)),
            alpha=0.6,
            zorder=1,
        )
    )
    ax.plot(*BEACON, marker="v", ms=11, color=BEACON_C, zorder=7)
    ax.text(
        BEACON[0],
        BEACON[1] + 0.55,
        "beacon",
        color=BEACON_C,
        fontsize=8.6,
        ha="center",
        va="bottom",
        fontweight="bold",
    )

    if show_prior:
        # The prior's 2-sigma spread, drawn as a dashed outline and capped: at 4 units
        # of standard deviation the real thing is wider than the panel, and filling it
        # floods every other mark. The cap is on the drawing, never on the belief.
        width, height, angle = gallery.covariance_ellipse(
            np.asarray(start_belief().cov)[CONTEXT_DIMS:, CONTEXT_DIMS:],
            max_diameter=7.0,
        )
        ax.add_patch(
            Ellipse(
                gallery.xy(BELIEVED_GOAL),
                width,
                height,
                angle=angle,
                facecolor="none",
                edgecolor=SIGMA_C,
                lw=1.2,
                ls=(0, (2, 2)),
                alpha=0.9,
                zorder=1,
            )
        )

    ax.plot(*truth, marker="*", ms=22, color=TRUE_C, zorder=7)
    ax.text(
        truth[0] + 0.75,
        truth[1] - 0.1,
        "the goal",
        color=TRUE_C,
        fontsize=8.6,
        ha="left",
        va="center",
        fontweight="bold",
    )
    ax.plot(0.0, 0.0, marker="o", ms=5, color=INK, zorder=7)
    ax.text(0.42, -0.5, "start", color=FAINT, fontsize=8.0, ha="left", va="top")


def _draw_agent(ax, state: dict, colour: str, style, dash) -> None:
    """One agent: where it is, where it thinks the goal is, and how sure it is."""
    from matplotlib.patches import Ellipse

    width, height, angle = gallery.covariance_ellipse(state["spread"], max_diameter=6.4)
    shared = {"angle": angle, "zorder": 2}
    ax.add_patch(
        Ellipse(state["guess"], width, height, facecolor=SIGMA_C, alpha=0.09, **shared)
    )
    ax.add_patch(
        Ellipse(
            state["guess"],
            width,
            height,
            facecolor="none",
            edgecolor=SIGMA_C,
            lw=1.3,
            ls=(0, (4, 3)),
            alpha=0.95,
            **shared,
        )
    )
    ax.plot(
        *state["guess"],
        marker="D",
        ms=9,
        markerfacecolor="none",
        markeredgecolor=GUESS_C,
        markeredgewidth=2.0,
        zorder=6,
    )
    # Up and to the left of the diamond, so that once the belief resolves onto the goal
    # the label does not land on top of the goal's own.
    ax.text(
        state["guess"][0] - 0.5,
        state["guess"][1] + 0.5,
        "its guess",
        color=GUESS_C,
        fontsize=8.0,
        ha="right",
        va="bottom",
        fontweight="bold",
    )

    trail = np.asarray(state["trail"])
    if len(trail) > 1:
        ax.plot(
            trail[:, 0], trail[:, 1], color=colour, lw=2.0, alpha=0.9, ls=dash, zorder=4
        )
        ax.plot(
            trail[:, 0],
            trail[:, 1],
            ls="none",
            marker="o",
            ms=2.6,
            color=colour,
            alpha=0.9,
            zorder=5,
        )
    gallery.draw_bacillus(ax, state["position"], state["heading"], 0.0, style)


TITLE = "R(x) and a long enough H bring curiosity back to a linear-Gaussian agent"


def _sweep_figure(runs: list[dict], rung: int, upto: float):
    """One frame: the world with this horizon's agent, and the ladder beside it."""
    import matplotlib.pyplot as plt

    run = runs[rung]
    detours = run["detours"]
    fig, (world, ladder) = plt.subplots(
        1, 2, figsize=(10.8, 5.0), width_ratios=(1.6, 1.0), dpi=96
    )
    fig.patch.set_facecolor(BG)
    fig.suptitle(TITLE, fontsize=13.0, fontweight="bold", color=INK, y=0.965)
    _draw_world(world, _FIELD_CACHE[0], show_prior=False)
    _draw_agent(
        world,
        _at(run, upto),
        FARSIGHTED_C if detours else MYOPIC_C,
        FARSIGHTED if detours else MYOPIC,
        "solid" if detours else (0, (4, 2)),
    )
    world.set_title(
        f"one agent, one world, scoring plans {run['horizon']} steps deep",
        fontsize=11.0,
        fontweight="bold",
        color=INK,
        pad=7,
    )
    world.set_xlabel(
        "H is how far ahead it scores, not how far it can see: the whole world is "
        "visible\nat every H, and only the first H steps of a plan count toward that "
        "plan's score.",
        fontsize=8.2,
        color=FAINT,
        labelpad=6,
    )
    _draw_ladder(ladder, runs, rung)
    fig.subplots_adjust(left=0.025, right=0.965, top=0.855, bottom=0.115, wspace=0.14)
    return fig, plt


def render_animation(runs: list[dict], out_path: Path) -> Path:
    """The flagship: the same world once per horizon, until the choice flips.

    This is the memory-hungry step, so frames are drawn and released one at a time
    rather than holding a couple of hundred matplotlib figures open at once.
    """
    gallery.use_headless_backend()
    frames = []
    for rung, upto in _sweep_frames(runs):
        fig, plt = _sweep_figure(runs, rung, upto)
        frames.append(gallery.figure_frame(fig))
        plt.close(fig)
    return gallery.write_gif(
        frames, out_path, fps=FPS, hold_seconds=1.4, quantize_colors=140
    )


def _draw_scene(ax, myopic: dict, farsighted: dict) -> None:
    """Panel A of the still: both runs finished, overlaid on the world they shared."""
    _draw_world(ax, _FIELD_CACHE[0], show_prior=True)
    ax.plot(
        *BELIEVED_GOAL,
        marker="D",
        ms=9,
        markerfacecolor="none",
        markeredgecolor=GUESS_C,
        markeredgewidth=2.0,
        zorder=7,
    )
    ax.text(
        BELIEVED_GOAL[0] - 0.35,
        BELIEVED_GOAL[1] + 0.35,
        "where the prior\nsaid it was",
        color=GUESS_C,
        fontsize=8.0,
        ha="right",
        va="bottom",
        fontweight="bold",
    )
    for run, colour, style, dash in (
        (myopic, MYOPIC_C, MYOPIC, (0, (4, 2))),
        (farsighted, FARSIGHTED_C, FARSIGHTED, "solid"),
    ):
        track = run["track"]
        ax.plot(
            track[:, 0],
            track[:, 1],
            color=colour,
            lw=2.0,
            alpha=0.9,
            ls=dash,
            zorder=4,
            label=f"plans {run['horizon']} ahead",
        )
        ax.plot(
            track[:, 0],
            track[:, 1],
            ls="none",
            marker="o",
            ms=2.6,
            color=colour,
            alpha=0.9,
            zorder=5,
        )
        gallery.draw_bacillus(
            ax, track[-1], _heading(track, len(track) - 1), 0.0, style
        )
    ax.set_title(
        "A  one world, two horizons",
        fontsize=11.5,
        fontweight="bold",
        color=INK,
        pad=8,
        loc="left",
    )
    ax.legend(loc="lower left", fontsize=8.2, framealpha=0.9)
    ax.set_xlabel(
        f"{MYOPIC_H} steps ahead: settle where the prior said, without looking\n"
        f"{FARSIGHTED_H} steps ahead: read the beacon first, then go to the real one",
        fontsize=8.4,
        color=FAINT,
        labelpad=7,
    )


def _draw_margin(ax, curve, crossing: int | None) -> None:
    """Panel B of the still: the margin, its split, and where it crosses."""
    horizons = np.array([stat.horizon for stat in curve])
    delta = np.array([float(stat.delta_g) for stat in curve])
    pull = np.array([float(stat.delta_epsilon) for stat in curve])
    gradient = np.array([float(stat.delta_c) for stat in curve])

    ax.axhline(0.0, color=INK, lw=1.2, zorder=4)
    ax.plot(
        horizons,
        gradient,
        color=gallery.PRAGMATIC_C,
        lw=1.6,
        ls=(0, (5, 2)),
        marker="^",
        ms=4,
        alpha=0.85,
        label="Δc  pragmatic gradient",
    )
    ax.plot(
        horizons,
        pull,
        color=gallery.EPISTEMIC_C,
        lw=1.6,
        ls=(0, (1, 1.6)),
        marker="s",
        ms=4,
        alpha=0.85,
        label="Δε  epistemic pull (flat)",
    )
    ax.plot(
        horizons,
        delta,
        color=MARGIN_C,
        lw=2.8,
        marker="o",
        ms=5,
        label="ΔG = Δc − Δε  the margin",
    )
    for horizon, colour in ((MYOPIC_H, MYOPIC_C), (FARSIGHTED_H, FARSIGHTED_C)):
        ax.axvline(horizon, color=colour, lw=1.6, alpha=0.7, zorder=1)
        ax.text(
            horizon,
            74.0,
            f"H = {horizon}",
            color=colour,
            fontsize=8.6,
            ha="center",
            va="top",
            fontweight="bold",
        )
    if crossing is not None:
        ax.annotate(
            f"the crossing, at H = {crossing}:\nabove it, curiosity pays",
            xy=(crossing, delta[crossing - 1]),
            xytext=(crossing + 1.2, 38.0),
            color=MARGIN_C,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="top",
            arrowprops={"arrowstyle": "->", "color": MARGIN_C, "lw": 1.3},
        )
    ax.set_title(
        "B  what the two agents were choosing between",
        fontsize=11.5,
        fontweight="bold",
        color=INK,
        pad=8,
        loc="left",
    )
    ax.set_xlabel("planning horizon H")
    ax.set_ylabel("nats")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=8.2, framealpha=0.9)


def _draw_control(ax, curve, frozen_at: float) -> None:
    """Panel C of the still: the same sweep with `R` frozen. No crossing, ever."""
    horizons = np.array([stat.horizon for stat in curve])
    delta = np.array([float(stat.delta_g) for stat in curve])
    closest = int(delta.argmin())

    ax.axhline(0.0, color=INK, lw=1.2, zorder=4)
    ax.plot(
        horizons,
        delta,
        color=MYOPIC_C,
        lw=2.8,
        ls=(0, (6, 2)),
        marker="D",
        ms=5,
        label="ΔG  frozen-R twin",
    )
    ax.fill_between(horizons, 0.0, delta, color=MYOPIC_C, alpha=0.12, lw=0)
    ax.annotate(
        f"the closest it ever comes to\nzero is {delta.min():+.2f} nats, at H = "
        f"{horizons[closest]}",
        xy=(horizons[closest], delta[closest]),
        xytext=(horizons[0] + 1.6, -38.0),
        color=INK,
        fontsize=8.8,
        ha="left",
        va="top",
        arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.1},
    )
    ax.text(
        float(horizons.mean()) + 0.5,
        -112.0,
        "no crossing at any horizon.\nWith R held still, curiosity never pays",
        color=MYOPIC_C,
        fontsize=9.6,
        ha="center",
        va="center",
        fontweight="bold",
    )
    ax.set_title(
        "C  the control: R frozen",
        fontsize=11.5,
        fontweight="bold",
        color=INK,
        pad=8,
        loc="left",
    )
    ax.set_xlabel("planning horizon H")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=8.2, framealpha=0.9)
    ax.text(
        0.97,
        0.04,
        f"R pinned at {frozen_at:.0f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.0,
        color=FAINT,
        family="monospace",
    )


def render_still(myopic: dict, farsighted: dict, out_path: Path) -> Path:
    """The companion: the same two runs, plus the margin that decided them."""
    gallery.use_headless_backend()
    import matplotlib.pyplot as plt

    live = margin_curve()
    frozen_at = frozen_reference_noise()
    frozen = margin_curve(frozen_noise=frozen_at)
    crossing = first_crossing(live)

    fig, (scene, flip, control_matrix) = plt.subplots(
        1, 3, figsize=(15.4, 5.2), width_ratios=(1.05, 1.35, 1.0)
    )
    fig.patch.set_facecolor(BG)
    fig.suptitle(TITLE, fontsize=13.5, fontweight="bold", color=INK, y=0.975)
    _draw_scene(scene, myopic, farsighted)
    _draw_margin(flip, live, crossing)
    _draw_control(control_matrix, frozen, frozen_at)
    for ax in (flip, control_matrix):
        ax.set_facecolor(BG)
        ax.set_ylim(-165.0, 82.0)
    control_matrix.set_yticklabels([])

    fig.text(
        0.5,
        0.018,
        f"ΔG(H) = G(detour, H) − G(direct, H), in nats. ΔG < 0 means the detour is "
        f"the better plan. Panels B and C are deterministic. The plans there are "
        f"open-loop, so nothing in them takes a draw. Panel A shows one closed-loop "
        f"run per agent at seed {SEED}.",
        ha="center",
        va="bottom",
        fontsize=8.4,
        color=FAINT,
        style="italic",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    gallery.save_figure(fig, out_path, dpi=190, facecolor=BG)
    plt.close(fig)
    return out_path


def main() -> None:
    """``--check`` runs the gate. Otherwise render the animation and its companion."""
    if "--check" in sys.argv:
        check()
        return
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = Path(args[0]) if args else Path("docs/assets/crossover_horizon.gif")

    _FIELD_CACHE[0] = _sharpness_field()
    runs = _ladder_runs()
    animation = render_animation(runs, out)
    still = render_still(
        simulate(MYOPIC_H, seed=SEED, n_steps=N_STEPS),
        simulate(FARSIGHTED_H, seed=SEED, n_steps=N_STEPS),
        out.with_suffix(".png"),
    )
    print(f"wrote {animation}\nwrote {still}")


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    main()
