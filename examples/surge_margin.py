"""Anti-surge margin on a compressor, where the flow meter's noise is `R(Q)`.

An applied case for state-dependent sensing, based on my reading of Sathish, Khader,
Sayma and Shah, "Centrifugal Compressor Design and Surge Simulation for Active Inference
Based Control", ASME Turbo Expo 2024, GT2024-124905 (doi:10.1115/GT2024-124905).
Nothing here is a claim about what that paper did.

A centrifugal compressor must not be run below its surge line, so anti-surge control
holds a margin above it. Flow is measured here with a differential-pressure element,
and a DP element obeys `Q = k·√ΔP` with roughly flat transmitter error. Propagate that
through the square root and the variance of the flow measurement is

    R(Q) = e_ref²·Q_ref⁴ / Q²

The noise on the flow reading depends on the flow being read. Nobody modelled that in.
It is what a DP meter does, and it is why such a meter has a turndown limit at all.

The element type sets the sign, and the sign decides the result. A Coriolis or
ultrasonic meter holds roughly a constant percentage of rate, so `R ∝ Q²` and the
epistemic term pushes the other way. A meter with constant absolute error has `R'= 0`
and no epistemic term at all. `docs/guides/surge_margin_derivation.md` carries the
derivation, the sign table and the one calibration constant.

Two agents differ in one line: a fixed-`R` twin frozen at the operating point (70% of
design flow), and an `R(x)` agent given the meter's real noise. The collapse the twin
exhibits is a property of *the agent's generative model*, which is linear-Gaussian with
additive control here, and not of the plant. An EKF linearised on a nonlinear plant has
action-dependent Jacobians and does not collapse this way, whatever its `R` does.

`--check` reports eleven registered falsifiers, with no plotting deps:

- dissociation: the twin's epistemic term is dead-flat across the valve grid, the
  `R(x)` agent's is not. Koudahl, Kouw and de Vries proved the flat case for
  fixed-noise linear Gaussian state space models (Entropy 23(12):1565, 2021,
  doi:10.3390/e23121565). ADR-003 records it here.
- standoff: the twin's `argmin G` is exactly `argmin pragmatic`, so its operating point
  is whatever preference dictates and nothing else. The `R(x)` agent's sits further
  from surge, held off by the ambiguity term.
- meter quality: over an 8:1 ladder in meter quality the `R(x)` standoff moves
  monotonically while the twin's does not move at all. The surge control line is an
  inference outcome under `R(Q)` and a hand-set constant under fixed `R`.
- slope: `ℓ'(m) = d/dm log R` from autodiff matches the closed form `−2/Q`, so this
  sensor's inference-gap coefficient is `c₂ = (ℓ'/2)² = 1/Q²`.
- robustness: the `R(x)` agent stays held off the twin's standoff across a 10:1 sweep
  of the economic weight `Λ`, so the *direction* of the offset survives that knob.
- sensor probe: the library's `probe_model` says the sensor earns its keep over the
  reachable set.
- stationarity law: every grid argmin lands within one grid step of the root of
  `Λ(m − g) = −ℓ'(m)·(κ + ΛR)/2`, with κ the Kalman gain.
- closed loop: the library's own `Agent`, run noise-free at the default meter, settles
  within a grid step of the law with κ at its steady state, and the twin on its
  preference.
- settled offset: by the law, every rung of the meter ladder settles clear of the
  preference, and each worse meter settles nearer it.
- Coriolis sign: a Coriolis meter with the same accuracy at design flow has
  `ℓ' = +2/Q`, and puts the agent below its preference, first step and settled.
- Coriolis loop: the library's own `Agent` with that meter settles where the law puts
  it, past the surge line.

**The standoff has a closed form.** Setting `dG/dm` to zero puts the offset from
preference at exactly `κ/(ΛQ) + R/Q` for this meter, with κ the Kalman gain. The first
part is epistemic. The second is pragmatic, because the preference is over readings and
a reading is quieter at higher flow. `Λ` prices margin against information and I chose
it, so it sets the scale of the first part. Sweeping it 10:1 moves the standoff by more
than the meter ladder does, which `--check` prints and asserts beside the ladder. A
worse meter lowers κ, which shrinks the first part, and raises `R`, which grows the
second. The first wins, and rows 3 and 9 assert the net direction. The existence of the
offset and its direction are what belong to the meter.

The ladders measure the first EFE step, taken from a prior whose σ is 2% of design
flow. κ depends on the meter only through `P⁻/R`, so from a fixed prior a meter twice as
bad and a prior four times as tight give almost the same first step. Only the small
pragmatic part tells them apart. The closed loop separates
them, because there P⁻ comes from the filter's steady state and so from the meter
itself. A settled loop holds a tighter belief than the prior, so its κ is smaller and it
sits much nearer the preference than its first step aimed. `--check` prints both.

**The direction is not the engineering one, and that is the interesting part.** A worse
meter moves this agent *closer* to surge, because a meter that resolves nothing offers
no information to go and get, so the economic term wins. Real surge margin is set by the
cost of being wrong, which is a pragmatic quantity. The ambiguity term is not a safety
factor, and this demo is the cleanest way to see that it is not.

The bare command renders a four-panel figure:

- TOP: EFE and its split across the valve grid for the fixed-`R` twin and the DP
  meter, at a shared y-scale with the surge line marked. The twin's flat epistemic line
  against the DP meter's rising one, and the `G` argmin moving off the pragmatic argmin.
- BOTTOM LEFT: the same for a Coriolis meter. Its epistemic line rises toward surge, and
  the `G` argmin lands past the surge line.
- BOTTOM RIGHT: standoff against meter quality, first step and settled, with the
  library `Agent`'s closed-loop point on the settled curve and the Coriolis meter's
  settled margin below the surge line. The DP curves' height is set by `Λ` through the
  law, so read their shape and not their height.

Nothing here is a citable number. See `examples/README.md`.

Run:
    uv run --no-sync python examples/surge_margin.py --check
    uv run --extra examples python examples/surge_margin.py
Output (bare command): docs/assets/surge_margin.png
"""

import math
from functools import cache, partial
from itertools import pairwise
from pathlib import Path

import gallery
import jax
import jax.numpy as jnp
import numpy as np

from cpomdp import Agent, EFESelector, ObservationGoal
from cpomdp.diagnostics import probe_model
from cpomdp.efe import expected_free_energy
from cpomdp.observation import CallableSensor
from cpomdp.selection import Preference
from cpomdp.types import Belief, LinearGaussianModel
from warrantlib import CheckReport, Outcome, Tier, Warrant, check_summary

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "surge_margin.png"

# Every flow quantity is a percentage of design volumetric flow. No Am³/h anywhere:
# the demo is scale-free once the meter is calibrated at Q_REF, and mixing units is
# how the calibration constant stops meaning what the derivation says it means.
#
# The operating point is illustrative and describes no machine. A real surge line comes
# off a measured performance map at a speed line, and nothing here was read off one.
Q_REF = 100.0  # design volumetric flow
Q_SURGE = 55.0  # surge-line flow at the operating speed line
Q_FLOOR = 5.0  # clamp on Q inside both meters' R(Q), each docstring says why
E_REF = 0.002  # relative flow uncertainty at Q_REF, the one calibration constant

# The latent is the surge margin m = Q - Q_SURGE, so the agent reasons in the quantity
# an anti-surge controller actually holds. The valve acts on it additively.
DYNAMICS_MATRIX = [[1.0]]  # A: margin persists
CONTROL_MATRIX = [[1.0]]  # B: an anti-surge valve step moves flow additively
OBSERVATION_MATRIX = [[1.0]]  # C: the meter reads flow
DYNAMICS_NOISE = [[0.01]]

OPERATING_MARGIN = 15.0  # m at the operating point (Q = 70%), illustrative
MARGIN_VAR = 4.0  # prior var on the margin, so sigma is 2% of design flow

# Economics: recycle costs money, so the preferred observation sits close to surge.
# Lambda prices a unit of margin against a nat of information. It is a policy choice,
# not a physical constant. It is not the only knob outside the meter derivation:
# MARGIN_VAR sets how much one observation can teach, and that scales the epistemic
# pull as much as Lambda does.
GOAL_MARGIN = 5.0
GOAL_PRECISION = 0.002

# The valve grid. Its resolution is also the bar the monotonicity row is read against,
# so both come from this one line. It reaches 15% of design flow past the surge line.
# `_interior_argmin` refuses a minimum on the edge, and a grid that stopped short of
# surge would act as an interlock nobody declared, deciding where the agent may go.
ACTIONS = jnp.linspace(-30.0, 20.0, 600)
GRID_STEP = float(ACTIONS[1] - ACTIONS[0])
PROBE_SAMPLES = 25  # actions `probe_model` samples, evenly strided over ACTIONS

# An 8:1 ladder in meter quality, anchored on E_REF at the good end. Doubling rather
# than a fine sweep, so each rung's standoff move clears the grid resolution by a wide
# margin rather than by one cell. It is a ladder in e_ref and not in turndown: eight
# times the error at design flow is what the base meter shows at 1/sqrt(8) of it.
METER_QUALITIES = (E_REF, 2 * E_REF, 4 * E_REF, 8 * E_REF)

# A 10:1 sweep of the economic weight, to price the meter ladder against a knob that
# is not traceable to the meter derivation. This moves the standoff further than
# meter quality does, which is the point of printing it beside the ladder.
GOAL_PRECISIONS = (
    GOAL_PRECISION / 4,
    GOAL_PRECISION / 2,
    GOAL_PRECISION,
    GOAL_PRECISION * 2.5,
)


def _dp_meter_noise(x, params):
    """`R(m)` for a DP flow meter: variance falls as the square of flow.

    Module-level (jit-safe) so it can ride in ``CallableSensor``'s static aux, with all
    tunables in ``params``. A lambda would hash by identity and defeat the cache.

    ``Q`` is clamped at ``floor`` well below the surge line. The expression diverges at
    ``Q = 0``, and a non-positive-definite ``R`` does not raise: it surfaces as a NaN at
    action selection, which is harder to read than an exception.

    Args:
        x: the predicted mean, whose single entry is the surge margin ``m``.
        params: ``e_ref``, ``q_ref``, ``q_surge`` and ``floor``.

    Returns:
        The 1x1 observation-noise matrix — R.
    """
    flow = jnp.maximum(params["q_surge"] + x[0], params["floor"])
    variance = params["e_ref"] ** 2 * params["q_ref"] ** 4 / flow**2
    return jnp.array([[variance]])


def _coriolis_meter_noise(x, params):
    """`R(m)` for a Coriolis meter: constant percentage of rate, so `R` grows as `Q²`.

    It reads the DP meter's parameters, so the two agree at design flow, where both are
    ``e_ref`` relative. Below design flow the DP meter gets worse and this one gets
    better, which is the whole difference between them.

    The clamp has its own reason here. Unclamped, ``R`` reaches zero at ``Q = 0``. That
    claims a perfect reading and makes the epistemic term infinite. An agent would steer
    into deep surge to collect it.

    Args:
        x: the predicted mean, whose single entry is the surge margin ``m``.
        params: ``e_ref``, ``q_ref``, ``q_surge`` and ``floor``.

    Returns:
        The 1x1 observation-noise matrix — R.
    """
    flow = jnp.maximum(params["q_surge"] + x[0], params["floor"])
    variance = (params["e_ref"] * flow) ** 2
    return jnp.array([[variance]])


METERS = {"dp": _dp_meter_noise, "coriolis": _coriolis_meter_noise}


def _meter_params(e_ref: float) -> dict:
    """The meter's noise parameters at one relative uncertainty."""
    return {
        "e_ref": e_ref,
        "q_ref": Q_REF,
        "q_surge": Q_SURGE,
        "floor": Q_FLOOR,
    }


def _frozen_noise(e_ref: float, meter: str = "dp") -> float:
    """`R` at the operating point (Q = 70%), which is what the fixed twin is given."""
    params = _meter_params(e_ref)
    return float(METERS[meter](jnp.array([OPERATING_MARGIN]), params)[0, 0])


def _model(
    e_ref: float = E_REF, *, meter: str = "dp", live: bool = True
) -> LinearGaussianModel:
    """The compressor model, its meter live or frozen at the operating point.

    Args:
        e_ref: the meter's relative flow uncertainty at ``Q_REF``.
        meter: a key of ``METERS``. The DP meter unless told otherwise.
        live: ``True`` for the ``R(x)`` agent, ``False`` for the fixed-`R` twin.

    Returns:
        The model. The live one reads its channel off a
        [`CallableSensor`][cpomdp.CallableSensor], and its ``observation_noise`` is that
        sensor's `R` at the prior mean, the operating point. The twin is handed the same
        value as a constant.
    """
    sensor = None
    if live:
        sensor = CallableSensor(OBSERVATION_MATRIX, METERS[meter], _meter_params(e_ref))
    return LinearGaussianModel(
        DYNAMICS_MATRIX,
        observation_matrix=None if live else OBSERVATION_MATRIX,
        dynamics_noise=DYNAMICS_NOISE,
        observation_noise=None if live else [[_frozen_noise(e_ref, meter)]],
        prior=_belief(),
        control_matrix=CONTROL_MATRIX,
        observation_model=sensor,
    )


def _belief() -> Belief:
    """The operating-point belief every sweep starts from."""
    return Belief(mean=[OPERATING_MARGIN], cov=[[MARGIN_VAR]])


def _goal(precision: float = GOAL_PRECISION) -> Preference:
    """The economic preference, in observation space: run close to surge.

    Args:
        precision: the economic weight — Λ. Defaults to the shipped value.

    Returns:
        The preference over observed surge margin.
    """
    return Preference(target=[GOAL_MARGIN], precision=[[precision]])


def _sweep(
    e_ref: float = E_REF,
    *,
    live: bool = True,
    precision: float = GOAL_PRECISION,
    meter: str = "dp",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pragmatic, epistemic and G over the valve grid, for one meter and one `Λ`.

    Args:
        e_ref: the meter's relative flow uncertainty at ``Q_REF``.
        live: ``True`` for the ``R(x)`` agent, ``False`` for the fixed-`R` twin.
        precision: the economic weight — Λ.
        meter: a key of ``METERS``.

    Returns:
        Three arrays, ``(pragmatic, epistemic, g)``, aligned with ``ACTIONS``.
    """
    return _cached_sweep(float(e_ref), bool(live), float(precision), meter)


@cache
def _cached_sweep(
    e_ref: float, live: bool, precision: float, meter: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One vmapped sweep per setting, computed once per process.

    The rows and the printout ask for the same settings more than once, and the ladders
    share their first rung with the default. Fifteen distinct settings cover the whole
    check, one of them the Coriolis meter's. `gallery.efe_sweep` loops eagerly, which
    suits the tens of actions it was written for and not the 600 here.
    """
    model = _model(e_ref, live=live, meter=meter)
    belief, goal = _belief(), _goal(precision)
    efe = jax.vmap(
        lambda action: expected_free_energy(model, belief, action[None], goal)
    )(ACTIONS)
    return (
        np.asarray(efe.pragmatic),
        np.asarray(efe.epistemic),
        np.asarray(efe.g),
    )


def _interior_argmin(values: np.ndarray) -> int:
    """The grid index of the minimum, refusing one on the edge of the valve grid.

    A minimum on the edge is the grid's limit and not the agent's optimum. Every row
    that reads a standoff would pass on it without saying so.

    Raises:
        ValueError: if the minimum sits on the first or last grid action.
    """
    index = int(np.argmin(values))
    if index in (0, len(values) - 1):
        raise ValueError(
            f"the minimum sits on the valve grid's edge, at action "
            f"{float(ACTIONS[index]):g}. Widen ACTIONS before reading it as a standoff"
        )
    return index


def _standoff(
    e_ref: float = E_REF,
    *,
    live: bool = True,
    precision: float = GOAL_PRECISION,
    meter: str = "dp",
) -> float:
    """Where one EFE step from the prior aims: `argmin G`, in % above surge."""
    _, _, g = _sweep(e_ref, live=live, precision=precision, meter=meter)
    return OPERATING_MARGIN + float(ACTIONS[_interior_argmin(g)])


def _pragmatic_standoff(
    e_ref: float = E_REF, *, live: bool = True, precision: float = GOAL_PRECISION
) -> float:
    """Where preference alone would put the valve, ignoring the ambiguity term."""
    pragmatic, _, _ = _sweep(e_ref, live=live, precision=precision)
    return OPERATING_MARGIN + float(ACTIONS[_interior_argmin(pragmatic)])


def _log_noise(margin, params, meter="dp"):
    """`ℓ(m) = log R(m)`, the scalar the gap work's coefficients are written in.

    ``meter`` is a key of ``METERS``. ``jax.grad`` passes it through undifferentiated,
    so the law's kernel and the DP slope below share this one function.
    """
    return jnp.log(METERS[meter](jnp.array([margin]), params)[0, 0])


#: The DP meter's `ℓ'(m)` over a whole grid in one traced call, for row 4. Built once
#: at import: a `jax.grad` constructed per call retraces per call, which costs more
#: than the sweep it checks.
_log_noise_slopes = jax.jit(jax.vmap(jax.grad(_log_noise), in_axes=(0, None)))


def _log_noise_slope(margin: float, e_ref: float = E_REF) -> float:
    """The DP meter's `ℓ'(m) = d/dm log R(m)` at one margin, by autodiff through `R`."""
    return float(_log_noise_slopes(jnp.array([margin]), _meter_params(e_ref))[0])


# --- the stationarity law -----------------------------------------------------------
# One EFE step moves only the predicted mean. With m the margin a valve step aims for
# and P⁻ the predicted variance, the two parts of G are
#
#     pragmatic(m) = ½Λ·[(m − g)² + P⁻ + R(m)]
#     epistemic(m) = ½·ln(1 + P⁻/R(m))
#
# Setting dG/dm to zero, with ℓ' = R'/R and the Kalman gain κ = P⁻/(P⁻ + R), gives
#
#     Λ·(m − g) = −ℓ'(m)·(κ(m) + Λ·R(m)) / 2
#
# For the DP meter ℓ' = −2/Q, so the offset from preference is exactly
#
#     m − g = κ/(ΛQ) + R/Q
#
# The first part is epistemic. The second is pragmatic: the preference is over readings,
# a noisier reading costs expected pragmatic value, and the reading is quieter at higher
# flow. As the meter worsens, κ falls and R rises, so the two parts pull opposite ways.
# One step from the prior κ is near one and the first part carries all but a few
# percent. A settled loop's κ is small enough at the worse meters that the second part
# matters. Λ sets the scale of the first. κ depends on the meter only through P⁻/R, and
# R scales as e_ref², so one step from a fixed prior a meter twice as bad and a prior
# four times as tight give almost the same standoff. Only the pragmatic part, which
# depends on R alone, tells them apart, and one step from the prior it is small.
#
# The sign of the offset is the sign of −ℓ', since κ + ΛR is positive. A Coriolis meter
# has ℓ' = +2/Q, so the same law puts it on the surge side of its preference.
PREDICTED_VAR = MARGIN_VAR + DYNAMICS_NOISE[0][0]  # P⁻, one step from the prior

# Stops short of the clamp at Q_FLOOR. Below about m = −45 the Coriolis residual has a
# false root, and at the clamp's kink the gradient of jnp.maximum is split in half.
LAW_BRACKET = (-Q_SURGE / 2, Q_REF - Q_SURGE)


def _stationarity_residual(
    margin: float,
    e_ref: float,
    precision: float,
    predicted_var: float,
    meter: str = "dp",
) -> float:
    """`dG/dm` at one target margin, from the closed form above.

    Written from the meter's own `R` and its autodiff slope, and from nothing in the
    library's EFE kernel, so a root of it is an independent oracle for the grid argmin.

    Args:
        margin: the margin a valve step aims for — m.
        e_ref: the meter's relative flow uncertainty at ``Q_REF``.
        precision: the economic weight — Λ.
        predicted_var: the predicted variance of the margin — P⁻.
        meter: a key of ``METERS``.

    Returns:
        `Λ(m − g) + ℓ'(m)·(κ + ΛR)/2`, which is zero at the standoff.
    """
    return float(
        _residual_kernel(
            margin, _meter_params(e_ref), precision, predicted_var, meter=meter
        )
    )


@partial(jax.jit, static_argnames="meter")
def _residual_kernel(margin, params, precision, predicted_var, *, meter):
    """The traced body of `_stationarity_residual`.

    Jitted because a bisection calls it hundreds of times, and eager dispatch of three
    small ops per call costs more than the sweep the root is checked against. The meter
    is static, so each one compiles once.
    """
    noise = METERS[meter](jnp.array([margin]), params)[0, 0]
    gain = predicted_var / (predicted_var + noise)  # κ
    slope = jax.grad(_log_noise)(margin, params, meter)  # ℓ'
    return precision * (margin - GOAL_MARGIN) + 0.5 * slope * (gain + precision * noise)


def _bisect(residual, low: float, high: float, steps: int = 60) -> float:
    """A root of ``residual`` on ``[low, high]``, running from negative to positive.

    Raises:
        ValueError: unless ``residual`` is negative at ``low`` and positive at ``high``.
    """
    if not residual(low) < 0.0 < residual(high):
        raise ValueError(f"[{low}, {high}] does not bracket a root of the residual")
    for _ in range(steps):
        middle = 0.5 * (low + high)
        if residual(middle) < 0.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _law_standoff(
    e_ref: float = E_REF,
    precision: float = GOAL_PRECISION,
    predicted_var: float = PREDICTED_VAR,
    meter: str = "dp",
) -> float:
    """The margin where `dG/dm = 0`, one step from the prior unless told otherwise.

    The root is searched for over ``LAW_BRACKET``, which holds it for every setting the
    demo visits, with either meter.
    """
    return _cached_law_standoff(
        float(e_ref), float(precision), float(predicted_var), meter
    )


@cache
def _cached_law_standoff(
    e_ref: float, precision: float, predicted_var: float, meter: str
) -> float:
    """One bisection per setting, computed once per process."""
    return _bisect(
        lambda margin: _stationarity_residual(
            margin, e_ref, precision, predicted_var, meter
        ),
        *LAW_BRACKET,
    )


# --- the closed loop ----------------------------------------------------------------
# A loop that has run for a while holds a tighter belief than the prior, so its κ is
# smaller and it settles nearer the preference than its first step aimed. At steady
# state the scalar filter's predicted variance solves P⁻² − qP⁻ − qR = 0 at the margin
# the loop holds, and that margin solves the law at that P⁻. Iterating the two finds
# the fixed point.
LOOP_STEPS = 40  # the default meter's belief settles inside twenty steps
LOOP_TAIL = 10  # the steps averaged, since a grid action dithers by a cell


def _steady_predicted_var(noise: float) -> float:
    """The scalar filter's steady-state P⁻ at one `R`, the positive root above."""
    process_noise = DYNAMICS_NOISE[0][0]  # q
    return 0.5 * (
        process_noise + math.sqrt(process_noise**2 + 4.0 * process_noise * noise)
    )


def _settled_law(
    e_ref: float = E_REF, precision: float = GOAL_PRECISION, meter: str = "dp"
) -> float:
    """Where the closed loop settles by the law, with P⁻ at its steady state there.

    Raises:
        RuntimeError: if the fixed-point iteration does not converge.
    """
    return _cached_settled_law(float(e_ref), float(precision), meter)


@cache
def _cached_settled_law(e_ref: float, precision: float, meter: str) -> float:
    """One fixed-point iteration per setting, computed once per process."""
    margin = _law_standoff(e_ref, precision, meter=meter)
    for _ in range(100):
        noise = float(METERS[meter](jnp.array([margin]), _meter_params(e_ref))[0, 0])
        settled = _law_standoff(
            e_ref, precision, _steady_predicted_var(noise), meter=meter
        )
        if abs(settled - margin) < 1e-10:
            return settled
        margin = settled
    raise RuntimeError(f"the steady-state law did not converge at e_ref = {e_ref}")


@cache
def _closed_loop(live: bool, meter: str = "dp") -> float:
    """Where the library's own `Agent` settles at the default accuracy, run noise-free.

    The plant follows the agent's model and the meter reads true, so the run is
    deterministic. The belief variance follows the filter's recursion and the margin
    goes wherever each step aims. A noisy run fluctuates about this point. Only the
    default accuracy runs here: a worse meter has a small gain, its variance takes
    hundreds of steps to settle, and the law covers it.

    Args:
        live: ``True`` for the ``R(x)`` agent, ``False`` for the fixed-`R` twin.
        meter: a key of ``METERS``.

    Returns:
        The mean margin over the last ``LOOP_TAIL`` steps.
    """
    model = _model(live=live, meter=meter)
    bounds = (float(ACTIONS[0]), float(ACTIONS[-1]))
    if live:
        objective = ObservationGoal(
            [GOAL_MARGIN],
            precision=[[GOAL_PRECISION]],
            action_bounds=bounds,
            n_candidates=len(ACTIONS),
        )
        agent = Agent(model, objective)
    else:
        # The Agent declines EFE on a fixed sensor, because the epistemic term is dead
        # there. The twin brings its own selector over the same grid.
        selector = EFESelector(model, n_candidates=len(ACTIONS), action_bounds=bounds)
        agent = Agent(
            model, Preference([GOAL_MARGIN], [[GOAL_PRECISION]]), selector=selector
        )
    margin, trace = OPERATING_MARGIN, []
    for _ in range(LOOP_STEPS):
        agent.infer_states(jnp.array([margin]))
        margin += float(agent.sample_action()[0])
        trace.append(margin)
    return float(np.mean(trace[-LOOP_TAIL:]))


def _panel(ax, prag, epi, g, title, note):
    """This demo's EFE panel: the shared one, labelled for a valve step."""
    gallery.draw_efe_panel(
        ax,
        ACTIONS,
        prag,
        epi,
        g,
        title=title,
        note=note,
        xlabel="anti-surge valve step  (% of design flow)",
    )


# --- the measurements ---------------------------------------------------------------
def _dissociation() -> tuple[float, float]:
    """Epistemic swing across the valve grid, live meter and frozen twin."""
    _, live_epistemic, _ = _sweep()
    _, frozen_epistemic, _ = _sweep(live=False)
    return float(np.ptp(live_epistemic)), float(np.ptp(frozen_epistemic))


def _standoffs() -> tuple[float, float, float]:
    """The live standoff, the twin's standoff, and the twin's pragmatic argmin."""
    return _standoff(), _standoff(live=False), _pragmatic_standoff(live=False)


def _meter_ladder() -> tuple[tuple[float, float, float], ...]:
    """`(e_ref, live standoff, frozen standoff)` at each rung of the meter ladder."""
    return tuple(
        (e_ref, _standoff(e_ref), _standoff(e_ref, live=False))
        for e_ref in METER_QUALITIES
    )


def _economic_ladder() -> tuple[tuple[float, float, float], ...]:
    """`(Λ, live standoff, frozen standoff)` at each rung of the economic-weight sweep.

    The control against the meter ladder. `Λ` is not traceable to the meter
    derivation, so its effect on the standoff is what says whether the meter ladder's
    numbers mean anything on their own. They do not.
    """
    return tuple(
        (
            precision,
            _standoff(precision=precision),
            _standoff(live=False, precision=precision),
        )
        for precision in GOAL_PRECISIONS
    )


def _law_agreement() -> tuple[tuple[str, float, float, float, float], ...]:
    """`(meter, e_ref, Λ, grid standoff, law root)` at every live setting visited.

    The two ladders share their default rung, so seven DP settings rather than eight,
    and the Coriolis meter adds its default.
    """
    settings = dict.fromkeys(
        [("dp", e_ref, GOAL_PRECISION) for e_ref in METER_QUALITIES]
        + [("dp", E_REF, precision) for precision in GOAL_PRECISIONS]
        + [("coriolis", E_REF, GOAL_PRECISION)]
    )
    return tuple(
        (
            meter,
            e_ref,
            precision,
            _standoff(e_ref, precision=precision, meter=meter),
            _law_standoff(e_ref, precision, meter=meter),
        )
        for meter, e_ref, precision in settings
    )


def _coriolis() -> tuple[float, float, float]:
    """The Coriolis meter at the default: first step, settled law, and the `Agent` loop.

    The first step is the grid argmin, which row 7 checks against the law's root.
    """
    return (
        _standoff(meter="coriolis"),
        _settled_law(meter="coriolis"),
        _closed_loop(True, "coriolis"),
    )


def _settled_ladder() -> tuple[tuple[float, float, float], ...]:
    """`(e_ref, first step, settled)` by the law at each rung of the meter ladder."""
    return tuple(
        (e_ref, _law_standoff(e_ref), _settled_law(e_ref)) for e_ref in METER_QUALITIES
    )


def _slope_deviation() -> tuple[float, float]:
    """Worst `|autodiff ℓ' − (−2/Q)|` over the grid, and `c₂` at the operating point."""
    margins = OPERATING_MARGIN + jnp.asarray(ACTIONS)
    autodiff = np.asarray(_log_noise_slopes(margins, _meter_params(E_REF)))
    closed_form = -2.0 / (Q_SURGE + np.asarray(margins))
    worst = float(np.max(np.abs(autodiff - closed_form)))
    slope = _log_noise_slope(OPERATING_MARGIN)
    return worst, (slope / 2.0) ** 2


@cache
def _probe_report():
    """The library's own verdict on whether the sensor earns its keep over the grid."""
    stride = len(ACTIONS) // PROBE_SAMPLES
    actions = [jnp.array([a]) for a in np.asarray(ACTIONS)[::stride]]
    return probe_model(_model(), _belief(), actions)


# --- the registered falsifiers ------------------------------------------------------
def falsifiers() -> tuple[CheckReport, ...]:
    """The registered falsifiers as reports. The sixth is the library's sensor probe.

    Every row here is read off a sweep over a *continuous* valve range, which is a
    finite grid over an infinite domain. That samples, so no row is ``PROVED`` at any
    grid resolution and none carries a ``Provenance``. These are showcase rows, and
    none of them is in ``research/warrant_ledger.md``.

    Returns:
        One report per registered falsifier, in the order the printout uses.
    """
    live_swing, frozen_swing = _dissociation()
    live_standoff, twin_standoff, twin_pragmatic = _standoffs()
    ladder = _meter_ladder()
    economic = _economic_ladder()
    worst_slope, c2 = _slope_deviation()
    report = _probe_report()
    agreement = _law_agreement()
    coriolis_first, coriolis_settled, coriolis_loop = _coriolis()
    settled = _settled_ladder()
    loop_live, loop_twin = _closed_loop(True), _closed_loop(False)

    live_margins = [m for _, m, _ in ladder]
    frozen_margins = [m for _, _, m in ladder]
    drops = [a - b for a, b in pairwise(live_margins)]
    frozen_spread = max(frozen_margins) - min(frozen_margins)
    law_misses = [abs(grid - law) for *_, grid, law in agreement]
    _, first_step, settled_default = settled[0]  # METER_QUALITIES opens on E_REF
    settled_margins = [margin for _, _, margin in settled]
    settled_drops = [a - b for a, b in pairwise(settled_margins)]
    settled_offset = settled_default - GOAL_MARGIN
    # Guarded, so a regression in the settled law that puts it on the preference reaches
    # row 9 as a FIRED row rather than as a ZeroDivisionError out of this function.
    overshoot = (
        (first_step - GOAL_MARGIN) / settled_offset if settled_offset > 0 else math.inf
    )

    def dissociation_fails() -> Outcome:
        """Fires unless the twin is flat and the live meter is not."""
        if frozen_swing < 1e-9 and live_swing > 1e-3:
            return Outcome.NOT_TRIGGERED
        return Outcome.FIRED

    def standoff_collapses() -> Outcome:
        """Fires unless the twin sits on preference and the live agent is held off."""
        twin_on_preference = abs(twin_standoff - twin_pragmatic) <= GRID_STEP
        held_off = live_standoff > twin_standoff + GRID_STEP
        return (
            Outcome.NOT_TRIGGERED if twin_on_preference and held_off else Outcome.FIRED
        )

    def ladder_not_monotone() -> Outcome:
        """Fires unless every rung moves the live standoff and none moves the twin's.

        The bar is the sweep's own resolution: a standoff that moved by less than one
        grid cell did not measurably move. Nothing is fitted to make this pass.
        """
        if frozen_spread > 1e-9:
            return Outcome.FIRED
        return (
            Outcome.NOT_TRIGGERED
            if all(drop > GRID_STEP for drop in drops)
            else Outcome.FIRED
        )

    def slope_misses_closed_form() -> Outcome:
        """Fires if autodiff and `−2/Q` disagree beyond float noise."""
        return Outcome.NOT_TRIGGERED if worst_slope < 1e-12 else Outcome.FIRED

    def offset_is_a_lambda_artefact() -> Outcome:
        """Fires if any economic weight leaves the R(x) agent no further from surge.

        The meter ladder's absolute numbers are not a result, since Λ moves them
        further than meter quality does. What the demo claims is the direction of the
        offset, and this is the row that claims it.
        """
        held_off = all(live > frozen + GRID_STEP for _, live, frozen in economic)
        return Outcome.NOT_TRIGGERED if held_off else Outcome.FIRED

    def sensor_flattens() -> Outcome:
        """Fires if the state-dependent sensor does not earn its keep over the grid."""
        if report.flattens or not report.epistemic_varies:
            return Outcome.FIRED
        return Outcome.NOT_TRIGGERED

    def standoff_leaves_its_law() -> Outcome:
        """Fires if any grid argmin sits more than one grid step from the law's root.

        A grid argmin of a smooth single-minimum curve lies within one cell of the
        true minimum, so one grid step is the sweep's own resolution and nothing is
        fitted to it.
        """
        return Outcome.NOT_TRIGGERED if max(law_misses) <= GRID_STEP else Outcome.FIRED

    def loop_leaves_the_law() -> Outcome:
        """Fires if the library's own loop settles away from where the law puts it.

        The `R(x)` agent has to settle within a grid step of the law at the default
        meter, and the twin within a grid step of its preference.
        """
        agrees = abs(loop_live - settled_default) <= GRID_STEP
        twin_on_preference = abs(loop_twin - GOAL_MARGIN) <= GRID_STEP
        return Outcome.NOT_TRIGGERED if agrees and twin_on_preference else Outcome.FIRED

    def settled_ladder_loses_the_offset() -> Outcome:
        """Fires if any settled rung loses the offset or the ladder stops falling.

        By the law, every rung of the meter ladder has to settle more than a grid step
        clear of the preference, and each worse meter more than a grid step nearer it.
        That is the first step's direction, held once the loop has settled. The settled
        margins are roots of the law and not read off the sweep. The grid step is the
        bar anyway because it is the agent's resolution: a loop acting on this grid
        holds its margin only to within a cell, so a settled difference smaller than
        that could not show up in the loop the law describes.
        """
        clear = all(margin - GOAL_MARGIN > GRID_STEP for margin in settled_margins)
        falling = all(drop > GRID_STEP for drop in settled_drops)
        return Outcome.NOT_TRIGGERED if clear and falling else Outcome.FIRED

    def coriolis_keeps_the_sign() -> Outcome:
        """Fires unless a Coriolis meter puts the agent on the surge side of preference.

        Its `ℓ' = +2/Q` against the DP meter's `−2/Q`, so the law flips the offset. The
        first step and the settled margin both have to land more than a grid step
        below the preference.
        """
        flipped = (
            coriolis_first < GOAL_MARGIN - GRID_STEP
            and coriolis_settled < GOAL_MARGIN - GRID_STEP
        )
        return Outcome.NOT_TRIGGERED if flipped else Outcome.FIRED

    def coriolis_stops_short_of_surge() -> Outcome:
        """Fires unless the library's Coriolis loop settles past the surge line.

        The library's own `Agent` has to settle within a grid step of the law, and
        more than a grid step below zero margin.
        """
        agrees = abs(coriolis_loop - coriolis_settled) <= GRID_STEP
        past_surge = coriolis_loop < -GRID_STEP
        return Outcome.NOT_TRIGGERED if agrees and past_surge else Outcome.FIRED

    quality_ratio = METER_QUALITIES[-1] / METER_QUALITIES[0]
    return (
        CheckReport(
            name="1. epistemic term does not dissociate",
            check_id="surge_margin.epistemic_does_not_dissociate",
            warrant=Warrant.CORROBORATED,
            outcome=dissociation_fails(),
            tier=Tier.COMPUTED,
            detail=(
                f"epistemic swing over the valve grid: live meter {live_swing:.4f} "
                f"nats, frozen twin {frozen_swing:.0e} nats (ADR-003 collapse). "
                f"Sampled on {len(ACTIONS)} grid actions"
            ),
        ),
        CheckReport(
            name="2. standoff is preference alone",
            check_id="surge_margin.standoff_is_preference_alone",
            warrant=Warrant.CORROBORATED,
            outcome=standoff_collapses(),
            tier=Tier.COMPUTED,
            detail=(
                f"twin aims at m = {twin_standoff:.2f}% and its pragmatic argmin is "
                f"m = {twin_pragmatic:.2f}%, the same cell. The R(x) agent aims "
                f"at m = {live_standoff:.2f}%"
            ),
        ),
        CheckReport(
            name="3. standoff ignores meter quality",
            check_id="surge_margin.standoff_ignores_meter_quality",
            warrant=Warrant.CORROBORATED,
            outcome=ladder_not_monotone(),
            tier=Tier.BOUNDED,
            detail=(
                f"over an {quality_ratio:.0f}:1 ladder in meter quality the R(x) "
                f"standoff falls "
                f"{live_margins[0]:.2f}% -> {live_margins[-1]:.2f}%, every rung moving "
                f"more than the grid step {GRID_STEP:.4f}% (smallest {min(drops):.3f}%)"
                f". The twin does not move: spread {frozen_spread:.0e}%"
            ),
        ),
        CheckReport(
            name="4. log-noise slope misses closed form",
            check_id="surge_margin.log_noise_slope_misses_closed_form",
            warrant=Warrant.CORROBORATED,
            outcome=slope_misses_closed_form(),
            tier=Tier.EXACT,
            detail=(
                f"autodiff ℓ'(m) vs −2/Q over {len(ACTIONS)} grid points: worst "
                f"deviation {worst_slope:.1e}, so c₂ = (ℓ'/2)² = {c2:.3e} at the "
                "operating point. A property of the sensor, not a measured gap"
            ),
        ),
        CheckReport(
            name="5. the offset is an economic-weight artefact",
            check_id="surge_margin.offset_is_an_artefact_of_the_economic_weight",
            warrant=Warrant.CORROBORATED,
            outcome=offset_is_a_lambda_artefact(),
            tier=Tier.BOUNDED,
            detail=(
                f"over Λ = {economic[0][0]:g} .. {economic[-1][0]:g} the R(x) standoff "
                f"spans {min(m for _, m, _ in economic):.2f}% .. "
                f"{max(m for _, m, _ in economic):.2f}% and stays clear of the twin by "
                f"more than the grid step {GRID_STEP:.4f}%. Λ moves the standoff "
                "further than meter quality does, so only the direction is a result"
            ),
        ),
        CheckReport(
            name="6. sensor flattens over reachable set",
            check_id="surge_margin.sensor_flattens_over_reachable_set",
            warrant=Warrant.CORROBORATED,
            outcome=sensor_flattens(),
            tier=Tier.COMPUTED,
            detail=(
                f"probe_model over {report.n_samples} reachable means: R spread "
                f"{report.observation_noise_spread:.3e}, epistemic "
                f"{report.epistemic_range[0]:.3f} .. "
                f"{report.epistemic_range[1]:.3f} nats"
            ),
        ),
        CheckReport(
            name="7. standoff leaves its stationarity law",
            check_id="surge_margin.standoff_leaves_its_stationarity_law",
            warrant=Warrant.CORROBORATED,
            outcome=standoff_leaves_its_law(),
            tier=Tier.BOUNDED,
            detail=(
                f"worst |grid argmin − law root| over {len(agreement)} settings of "
                f"both ladders and the Coriolis meter: {max(law_misses):.3f}%, "
                f"against the grid step {GRID_STEP:.4f}%. Λ(m − g) = −ℓ'(κ + ΛR)/2 "
                "places the standoff to the sweep's resolution"
            ),
        ),
        CheckReport(
            name="8. closed loop leaves the settled law",
            check_id="surge_margin.closed_loop_leaves_the_settled_law",
            warrant=Warrant.CORROBORATED,
            outcome=loop_leaves_the_law(),
            tier=Tier.BOUNDED,
            detail=(
                f"library Agent, noise-free closed loop at e_ref = {E_REF}: settles at "
                f"m = {loop_live:.2f}% against the law's {settled_default:.2f}%, and "
                f"the twin at {loop_twin:.2f}% against its preference "
                f"{GOAL_MARGIN:.2f}%, both within the grid step {GRID_STEP:.4f}%"
            ),
        ),
        CheckReport(
            name="9. settled ladder loses the offset",
            check_id="surge_margin.settled_ladder_loses_the_offset",
            warrant=Warrant.CORROBORATED,
            outcome=settled_ladder_loses_the_offset(),
            tier=Tier.BOUNDED,
            detail=(
                f"by the law the {quality_ratio:.0f}:1 ladder settles "
                f"{settled_margins[0]:.2f}% -> {settled_margins[-1]:.2f}%, every rung "
                f"clear of the preference and every step down larger than the grid "
                f"step {GRID_STEP:.4f}% (smallest {min(settled_drops):.3f}%). The "
                f"first step's offset from preference is {overshoot:.1f}x the settled "
                "one"
            ),
        ),
        CheckReport(
            name="10. Coriolis offset keeps the DP sign",
            check_id="surge_margin.coriolis_offset_keeps_the_dp_sign",
            warrant=Warrant.CORROBORATED,
            outcome=coriolis_keeps_the_sign(),
            tier=Tier.BOUNDED,
            detail=(
                f"a Coriolis meter, {E_REF * 100:.1f}% at design flow like the DP one: "
                f"the first step aims at m = {coriolis_first:.2f}% and the law settles "
                f"it at m = {coriolis_settled:.2f}%, both below the preference "
                f"{GOAL_MARGIN:.0f}% by more than the grid step {GRID_STEP:.4f}%. "
                "ℓ' = +2/Q flips the offset"
            ),
        ),
        CheckReport(
            name="11. Coriolis loop settles short of surge",
            check_id="surge_margin.coriolis_loop_settles_short_of_surge",
            warrant=Warrant.CORROBORATED,
            outcome=coriolis_stops_short_of_surge(),
            tier=Tier.BOUNDED,
            detail=(
                f"library Agent with the Coriolis meter, noise-free closed loop: "
                f"settles at m = {coriolis_loop:.2f}% against the law's "
                f"{coriolis_settled:.2f}%, past the surge line by more than the grid "
                f"step {GRID_STEP:.4f}%. With the DP meter it settles at "
                f"{loop_live:.2f}%"
            ),
        ),
    )


def _print_falsifiers(reports: tuple[CheckReport, ...]) -> None:
    """Print the registered falsifiers, then the run's counts.

    Prover and tier print as separate columns because they answer separate questions.
    Every row here samples a continuous range, so the prover column reads the same
    down the table. The tier column is what varies.
    """
    print(f"\nRegistered falsifiers ({len(reports)} registered):")
    print(f"   {'':<44} {'outcome':<15} {'prover':<13} {'tier':<9} why")
    for report in reports:
        prover = report.warrant.value if report.warrant else "—"
        print(
            f"   {report.name:<44} {report.outcome.value:<15} "
            f"{prover:<13} {report.tier.value:<9} {report.detail}"
        )
    print(f"\n{check_summary(reports)}")


def check() -> None:
    """Report the registered falsifiers and assert none of them fired.

    Plotting-free: matplotlib is imported inside ``main``, so this runs in the base
    test environment without the ``examples`` extra.

    Raises:
        AssertionError: if any registered falsifier fired, or if the meter ladder
            spans as much as the `Λ` ladder, which would make the printout's framing
            of the two wrong.
    """
    print("Anti-surge margin, DP flow meter unless stated, R(Q) = e_ref²·Q_ref⁴/Q²")
    print(
        f"   surge line Q_s = {Q_SURGE:.0f}%, design flow Q_ref = {Q_REF:.0f}%, "
        f"e_ref = {E_REF} ({E_REF * 100:.1f}% at design flow)"
    )
    print(
        f"   operating margin m = {OPERATING_MARGIN:.0f}%, economic preference "
        f"m = {GOAL_MARGIN:.0f}%, Λ = {GOAL_PRECISION}\n"
    )

    ladder = _meter_ladder()
    print("Where the R(x) agent aims, and where it settles (the headline):")
    print(
        f"   {'e_ref':>8} {'% at design':>12} {'first step':>11} {'law':>8} "
        f"{'settles':>9} {'fixed twin':>11}"
    )
    for e_ref, live, frozen in ladder:
        print(
            f"   {e_ref:>8.4f} {e_ref * 100:>11.1f}% {live:>10.2f}% "
            f"{_law_standoff(e_ref):>7.2f}% {_settled_law(e_ref):>8.2f}% "
            f"{frozen:>10.2f}%"
        )
    print(
        "   first step: the grid argmin. law: its closed-form root. settles: the law "
        "with κ\n   at its steady state."
    )
    print(
        f"   library Agent in closed loop at e_ref = {E_REF}: settles at "
        f"{_closed_loop(True):.2f}%, the twin at {_closed_loop(False):.2f}%"
    )
    print(
        "   -> a worse meter runs CLOSER to surge, first step and settled alike, which "
        "rows\n      3 and 9 assert. The offset is κ/(ΛQ) + R/Q. A worse meter lowers "
        "the gain κ,\n      which shrinks the first part, and raises R, which grows "
        "the second. The first\n      wins. A settled loop holds a tighter belief, so "
        "its κ is smaller and it sits\n      nearer the preference. The ambiguity term "
        "rewards resolvable uncertainty, and\n      it is not a safety factor.\n"
    )

    economic = _economic_ladder()
    meter_span = max(m for _, m, _ in ladder) - min(m for _, m, _ in ladder)
    economic_span = max(m for _, m, _ in economic) - min(m for _, m, _ in economic)
    print("The same standoff against the economic weight Λ (the control):")
    print(f"   {'Λ':>8} {'':>12} {'R(x) agent':>12} {'law':>8} {'fixed twin':>12}")
    for precision, live, frozen in economic:
        print(
            f"   {precision:>8.5f} {'':>12} {live:>11.2f}% "
            f"{_law_standoff(precision=precision):>7.2f}% {frozen:>11.2f}%"
        )
    print(
        f"   -> Λ spans {economic_span:.2f}% of design flow against the meter "
        f"ladder's {meter_span:.2f}%.\n      The law says why: the epistemic part of "
        "the offset is κ/(ΛQ), so Λ sets its\n      scale. Only the direction belongs "
        "to the meter, and row 5 is what claims it.\n"
    )
    assert economic_span > meter_span, (
        f"the meter ladder spans {meter_span:.2f}%, at least the Λ ladder's "
        f"{economic_span:.2f}%, so the printout's framing no longer holds"
    )

    coriolis_first, coriolis_settled, coriolis_loop = _coriolis()
    print(
        f"The same agent with a Coriolis meter, {E_REF * 100:.1f}% at design flow, "
        "R = (e_ref·Q)²:"
    )
    print(
        f"   first step m = {coriolis_first:.2f}%, settles m = {coriolis_settled:.2f}% "
        f"by the law, library Agent in closed loop {coriolis_loop:.2f}%"
    )
    print(
        "   -> the offset flips sign. ℓ' = +2/Q, so the information term pulls toward "
        "low\n      flow, where this meter reads best, and the loop settles past the "
        "surge line.\n      The ambiguity term can push the agent into surge.\n"
    )

    print(_probe_report().summary())

    reports = falsifiers()
    _print_falsifiers(reports)
    print(
        "   these are showcase numbers, not citable facts. Derivation: "
        "docs/guides/surge_margin_derivation.md"
    )

    fired = [report.name for report in reports if report.outcome is Outcome.FIRED]
    assert not fired, f"registered falsifiers fired: {fired}"


def render(out_path: Path) -> Path:
    """Draw the four-panel figure and write it.

    Three EFE panels through the shared gallery helper at one y-scale: the fixed-`R`
    twin's flat epistemic line, the DP meter's rising one, and the Coriolis meter's
    falling one. The fourth panel is the meter-quality ladder, the first step against
    where the loop settles, with the Coriolis meter's settled margin below the surge
    line.

    Args:
        out_path: where the PNG goes.

    Returns:
        The path written.
    """
    gallery.use_headless_backend()
    import matplotlib.pyplot as plt

    twin_prag, twin_epi, twin_g = _sweep(live=False)
    live_prag, live_epi, live_g = _sweep()
    coriolis_prag, coriolis_epi, coriolis_g = _sweep(meter="coriolis")
    ladder = _meter_ladder()

    fig, ((twin_ax, live_ax), (coriolis_ax, right)) = plt.subplots(
        2, 2, figsize=(13.0, 11.0)
    )
    fig.patch.set_facecolor(gallery.PAPER.bg)

    _panel(
        twin_ax,
        twin_prag,
        twin_epi,
        twin_g,
        "fixed R, frozen at the operating point",
        "epistemic dead-flat, so argmin G is argmin pragmatic:\nthe control line is "
        "whatever preference says",
    )
    _panel(
        live_ax,
        live_prag,
        live_epi,
        live_g,
        f"DP meter, R(Q) live, e_ref = {E_REF}",
        "epistemic rises away from surge and holds\nthe valve off the economic optimum",
    )
    _panel(
        coriolis_ax,
        coriolis_prag,
        coriolis_epi,
        coriolis_g,
        f"Coriolis meter, R(Q) live, e_ref = {E_REF}",
        "epistemic rises toward surge and pulls\nthe valve past the surge line",
    )
    ylim = _shared_ylim(twin_g, live_g, coriolis_g, twin_epi, live_epi, coriolis_epi)
    for ax in (twin_ax, live_ax, coriolis_ax):
        ax.set_ylim(ylim)
        _mark_surge(ax)

    e_refs = [e for e, _, _ in ladder]
    right.plot(
        e_refs,
        [m for _, m, _ in ladder],
        "--o",
        color=gallery.EPISTEMIC_C,
        mfc="white",
        lw=1.6,
        label="DP meter, first step",
    )
    right.plot(
        e_refs,
        [_settled_law(e_ref) for e_ref in e_refs],
        "-o",
        color=gallery.EPISTEMIC_C,
        lw=2.2,
        label="DP meter, settled (law)",
    )
    right.plot(
        [E_REF],
        [_closed_loop(True)],
        "*",
        color=gallery.VERMILLION,
        ms=13,
        label="library Agent, DP meter, closed loop",
    )
    right.plot(
        e_refs,
        [m for _, _, m in ladder],
        "-s",
        color=gallery.PRAGMATIC_C,
        lw=2.2,
        label="fixed R twin",
    )
    right.plot(
        [E_REF],
        [_settled_law(meter="coriolis")],
        "D",
        color=gallery.VERMILLION,
        ms=8,
        label="Coriolis meter, settled (law)",
    )
    right.axhline(0.0, color=gallery.VERMILLION, lw=1.4, ls="--", zorder=1)
    right.annotate(
        "surge",
        xy=(1.0, 0.0),
        xycoords=("axes fraction", "data"),
        xytext=(-4, 4),
        textcoords="offset points",
        ha="right",
        color=gallery.VERMILLION,
        fontsize=9,
        fontweight="bold",
    )
    right.set_xscale("log")
    right.set_xticks(e_refs)
    right.set_xticklabels([f"{e * 100:g}%" for e in e_refs])
    # The log locator would otherwise print its own decade minors beside the four
    # rungs, which reads as data points that are not there.
    right.set_xticks([], minor=True)
    right.set_xlabel("meter relative uncertainty at design flow")
    right.set_ylabel("standoff above surge  (% of design flow)")
    right.set_title(
        "the control line moves, or it does not", fontsize=12, fontweight="bold"
    )
    right.annotate(
        "a worse DP meter runs CLOSER to surge, first step and settled alike,\n"
        "and a Coriolis meter settles past it: the ambiguity term is not a\n"
        "safety factor. Read the shape, not the height: Λ sets the height",
        xy=(0.5, -0.30),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#333333",
    )
    right.grid(True, alpha=0.25)
    # Between the surge line and the twin's line, the one band no curve crosses.
    right.legend(
        loc="lower right", bbox_to_anchor=(1.0, 0.1), fontsize=8.5, framealpha=0.9
    )

    fig.tight_layout()
    return gallery.save_figure(fig, out_path, dpi=150, tight=True)


def _shared_ylim(*series) -> tuple[float, float]:
    """One y-range across the EFE panels, so the flat line reads as flat.

    The headroom is uneven on purpose. The epistemic line runs along the top of the
    data, and the shared legend sits upper right, so without room above it the legend
    covers the one curve the panels exist to compare.
    """
    lo = min(float(np.min(s)) for s in series)
    hi = max(float(np.max(s)) for s in series)
    span = hi - lo
    return lo - 0.08 * span, hi + 0.45 * span


def _mark_surge(ax) -> None:
    """Draw the surge line on a valve-sweep panel."""
    surge_action = -OPERATING_MARGIN
    ax.axvline(surge_action, color=gallery.VERMILLION, lw=1.4, ls="--", zorder=1)
    ax.annotate(
        "surge",
        xy=(surge_action, ax.get_ylim()[1]),
        xytext=(3, -11),
        textcoords="offset points",
        color=gallery.VERMILLION,
        fontsize=9,
        fontweight="bold",
    )


def main() -> None:
    """``--check`` runs the gate. Otherwise render the figure."""
    gallery.figure_main(render, str(OUT), check=check)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    main()
