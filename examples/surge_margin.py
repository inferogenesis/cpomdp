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

`--check` reports six registered falsifiers, with no plotting deps:

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

**Where the standoff sits is a `Λ` artefact and is not a result.** `Λ` prices margin
against information and I chose it. Sweeping it 10:1 moves the standoff by more than
the meter ladder does, which `--check` prints and asserts beside the ladder so the two
cannot be read as comparable. The existence of the offset and its direction are what
survive.

Every standoff here is one EFE step from the prior, whose σ is 2% of design flow. That
is where the first decision aims. It is not where a closed-loop agent settles, and the
prior's width scales it as much as `Λ` does.

**The direction is not the engineering one, and that is the interesting part.** A worse
meter moves this agent *closer* to surge, because a meter that resolves nothing offers
no information to go and get, so the economic term wins. Real surge margin is set by the
cost of being wrong, which is a pragmatic quantity. The ambiguity term is not a safety
factor, and this demo is the cleanest way to see that it is not.

The bare command renders a three-panel figure:

- LEFT and MIDDLE: EFE and its split across the valve grid, one agent each at a shared
  y-scale, surge line marked. The twin's flat epistemic line against the `R(x)`
  agent's rising one, and the `G` argmin moving off the pragmatic argmin.
- RIGHT: standoff against meter quality. A flat line (fixed `R`) against a curve. The
  vertical position of that curve is set by `Λ`, so read its shape and not its height.

Nothing here is a citable number. See `examples/README.md`.

Run:
    uv run --no-sync python examples/surge_margin.py --check
    uv run --extra examples python examples/surge_margin.py
Output (bare command): docs/assets/surge_margin.png
"""

from functools import cache
from itertools import pairwise
from pathlib import Path

import gallery
import jax
import jax.numpy as jnp
import numpy as np

from cpomdp.diagnostics import probe_model
from cpomdp.efe import expected_free_energy
from cpomdp.observation import CallableSensor
from cpomdp.selection import Preference
from cpomdp.types import Belief, LinearGaussianModel
from cpomdp.warrant import CheckReport, Outcome, Tier, Warrant, check_summary

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "surge_margin.png"

# Every flow quantity is a percentage of design volumetric flow. No Am³/h anywhere:
# the demo is scale-free once the meter is calibrated at Q_REF, and mixing units is
# how the calibration constant stops meaning what the derivation says it means.
#
# The operating point is illustrative and describes no machine. A real surge line comes
# off a measured performance map at a speed line, and nothing here was read off one.
Q_REF = 100.0  # design volumetric flow
Q_SURGE = 55.0  # surge-line flow at the operating speed line
Q_FLOOR = 5.0  # clamp on Q inside R(Q), see _dp_meter_noise
E_REF = 0.002  # relative flow uncertainty at Q_REF, the one calibration constant

# The latent is the surge margin m = Q - Q_SURGE, so the agent reasons in the quantity
# an anti-surge controller actually holds. The valve acts on it additively.
DYNAMICS_MATRIX = [[1.0]]  # A: margin persists
CONTROL_MATRIX = [[1.0]]  # B: an anti-surge valve step moves flow additively
OBSERVATION_MATRIX = [[1.0]]  # C: the DP meter reads flow
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
# so both come from this one line.
ACTIONS = jnp.linspace(-14.0, 20.0, 400)
GRID_STEP = float(ACTIONS[1] - ACTIONS[0])

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


def _meter_params(e_ref: float) -> dict:
    """The meter's noise parameters at one relative uncertainty."""
    return {
        "e_ref": e_ref,
        "q_ref": Q_REF,
        "q_surge": Q_SURGE,
        "floor": Q_FLOOR,
    }


def _frozen_noise(e_ref: float) -> float:
    """`R` at the operating point (Q = 70%), which is what the fixed twin is given."""
    params = _meter_params(e_ref)
    return float(_dp_meter_noise(jnp.array([OPERATING_MARGIN]), params)[0, 0])


def _model(e_ref: float = E_REF, *, live: bool = True) -> LinearGaussianModel:
    """The compressor model, its meter live or frozen at the operating point.

    Args:
        e_ref: the meter's relative flow uncertainty at ``Q_REF``.
        live: ``True`` for the ``R(x)`` agent, ``False`` for the fixed-`R` twin.

    Returns:
        The model. Both variants carry the same nominal ``observation_noise``. The live
        one overrides it through [`CallableSensor`][cpomdp.CallableSensor].
    """
    sensor = None
    if live:
        sensor = CallableSensor(
            OBSERVATION_MATRIX, _dp_meter_noise, _meter_params(e_ref)
        )
    return LinearGaussianModel(
        DYNAMICS_MATRIX,
        observation_matrix=OBSERVATION_MATRIX,
        dynamics_noise=DYNAMICS_NOISE,
        # Required and positive-definite even when a callable sensor supplies the live
        # R. For the twin this frozen value IS the sensor.
        observation_noise=[[_frozen_noise(e_ref)]],
        prior=Belief(mean=[OPERATING_MARGIN], cov=[[MARGIN_VAR]]),
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
    return Preference(goal=[GOAL_MARGIN], precision=[[precision]])


def _sweep(
    e_ref: float = E_REF, *, live: bool = True, precision: float = GOAL_PRECISION
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pragmatic, epistemic and G over the valve grid, for one meter and one `Λ`.

    Args:
        e_ref: the meter's relative flow uncertainty at ``Q_REF``.
        live: ``True`` for the ``R(x)`` agent, ``False`` for the fixed-`R` twin.
        precision: the economic weight — Λ.

    Returns:
        Three arrays, ``(pragmatic, epistemic, g)``, aligned with ``ACTIONS``.
    """
    return _cached_sweep(float(e_ref), bool(live), float(precision))


@cache
def _cached_sweep(
    e_ref: float, live: bool, precision: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One vmapped sweep per setting, computed once per process.

    The rows and the printout ask for the same settings more than once, and the ladders
    share their first rung with the default. Fourteen distinct settings cover the
    whole check. `gallery.efe_sweep` loops eagerly, which suits the tens of actions it
    was written for and not the 400 here.
    """
    model, belief, goal = _model(e_ref, live=live), _belief(), _goal(precision)
    g, parts = jax.vmap(
        lambda action: expected_free_energy(model, belief, action[None], goal)
    )(ACTIONS)
    return (
        np.asarray(parts["pragmatic"]),
        np.asarray(parts["epistemic"]),
        np.asarray(g),
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
    e_ref: float = E_REF, *, live: bool = True, precision: float = GOAL_PRECISION
) -> float:
    """Where one EFE step from the prior aims: `argmin G`, in % above surge."""
    _, _, g = _sweep(e_ref, live=live, precision=precision)
    return OPERATING_MARGIN + float(ACTIONS[_interior_argmin(g)])


def _pragmatic_standoff(
    e_ref: float = E_REF, *, live: bool = True, precision: float = GOAL_PRECISION
) -> float:
    """Where preference alone would put the valve, ignoring the ambiguity term."""
    pragmatic, _, _ = _sweep(e_ref, live=live, precision=precision)
    return OPERATING_MARGIN + float(ACTIONS[_interior_argmin(pragmatic)])


def _log_noise(margin, params):
    """`ℓ(m) = log R(m)`, the scalar the gap work's coefficients are written in."""
    return jnp.log(_dp_meter_noise(jnp.array([margin]), params)[0, 0])


#: `ℓ'(m)` over a whole grid in one traced call. Built once at import: a `jax.grad`
#: constructed per call retraces per call, which costs more than the sweep it checks.
_log_noise_slopes = jax.jit(jax.vmap(jax.grad(_log_noise), in_axes=(0, None)))


def _log_noise_slope(margin: float, e_ref: float = E_REF) -> float:
    """`ℓ'(m) = d/dm log R(m)` at one margin, by autodiff through the demo's own `R`."""
    return float(_log_noise_slopes(jnp.array([margin]), _meter_params(e_ref))[0])


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


def _slope_deviation() -> tuple[float, float]:
    """Worst `|autodiff ℓ' − (−2/Q)|` over the grid, and `c₂` at the operating point."""
    margins = OPERATING_MARGIN + jnp.asarray(ACTIONS)
    autodiff = np.asarray(_log_noise_slopes(margins, _meter_params(E_REF)))
    closed_form = -2.0 / (Q_SURGE + np.asarray(margins))
    worst = float(np.max(np.abs(autodiff - closed_form)))
    slope = _log_noise_slope(OPERATING_MARGIN)
    return worst, (slope / 2.0) ** 2


@cache
def _sensor_report():
    """The library's own verdict on whether the sensor earns its keep over the grid."""
    actions = [jnp.array([a]) for a in np.asarray(ACTIONS)[::16]]
    return probe_model(_model(), _belief(), actions)


# --- the registered falsifiers ------------------------------------------------------
def falsifiers() -> tuple[CheckReport, ...]:
    """The six registered falsifiers as reports, the last being the sensor probe.

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
    report = _sensor_report()

    live_margins = [m for _, m, _ in ladder]
    frozen_margins = [m for _, _, m in ladder]
    drops = [a - b for a, b in pairwise(live_margins)]
    frozen_spread = max(frozen_margins) - min(frozen_margins)

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
                f"{report.noise_spread:.3e}, epistemic {report.epistemic_range[0]:.3f}"
                f" .. {report.epistemic_range[1]:.3f} nats"
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
    print("Anti-surge margin under a DP flow meter, R(Q) = e_ref²·Q_ref⁴/Q²")
    print(
        f"   surge line Q_s = {Q_SURGE:.0f}%, design flow Q_ref = {Q_REF:.0f}%, "
        f"e_ref = {E_REF} ({E_REF * 100:.1f}% at design flow)"
    )
    print(
        f"   operating margin m = {OPERATING_MARGIN:.0f}%, economic preference "
        f"m = {GOAL_MARGIN:.0f}%, Λ = {GOAL_PRECISION}\n"
    )

    ladder = _meter_ladder()
    print("One EFE step from the prior, against meter quality (the headline):")
    print(f"   {'e_ref':>8} {'% at design':>12} {'R(x) agent':>12} {'fixed twin':>12}")
    for e_ref, live, frozen in ladder:
        print(f"   {e_ref:>8.4f} {e_ref * 100:>11.1f}% {live:>11.2f}% {frozen:>11.2f}%")
    print(
        "   -> a worse meter runs CLOSER to surge. The ambiguity term rewards "
        "resolvable\n      uncertainty, and it is not a safety factor.\n"
    )

    economic = _economic_ladder()
    meter_span = max(m for _, m, _ in ladder) - min(m for _, m, _ in ladder)
    economic_span = max(m for _, m, _ in economic) - min(m for _, m, _ in economic)
    print("The same standoff against the economic weight Λ (the control):")
    print(f"   {'Λ':>8} {'':>12} {'R(x) agent':>12} {'fixed twin':>12}")
    for precision, live, frozen in economic:
        print(f"   {precision:>8.5f} {'':>12} {live:>11.2f}% {frozen:>11.2f}%")
    print(
        f"   -> Λ spans {economic_span:.2f}% of design flow against the meter "
        f"ladder's {meter_span:.2f}%.\n      Where the standoff SITS is a Λ artefact. "
        "Only the direction of the offset\n      is a result, and row 5 is what "
        "claims it.\n"
    )
    assert economic_span > meter_span, (
        f"the meter ladder spans {meter_span:.2f}%, at least the Λ ladder's "
        f"{economic_span:.2f}%, so the printout's framing no longer holds"
    )

    print(_sensor_report().summary())

    reports = falsifiers()
    _print_falsifiers(reports)
    print(
        "   these are showcase numbers, not citable facts. Derivation: "
        "docs/guides/surge_margin_derivation.md"
    )

    fired = [report.name for report in reports if report.outcome is Outcome.FIRED]
    assert not fired, f"registered falsifiers fired: {fired}"


def render(out_path: Path) -> Path:
    """Draw the three-panel figure and write it.

    Two EFE panels through the shared gallery helper, one per agent, so the flat
    epistemic line and the rising one sit side by side at the same scale. The third
    panel is the meter-quality ladder.

    Args:
        out_path: where the PNG goes.

    Returns:
        The path written.
    """
    gallery.use_headless_backend()
    import matplotlib.pyplot as plt

    twin_prag, twin_epi, twin_g = _sweep(live=False)
    live_prag, live_epi, live_g = _sweep()
    ladder = _meter_ladder()

    fig, (twin_ax, live_ax, right) = plt.subplots(1, 3, figsize=(17.5, 5.0))
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
        f"R(Q) live, e_ref = {E_REF}",
        "epistemic rises away from surge and holds\nthe valve off the economic optimum",
    )
    for ax in (twin_ax, live_ax):
        ax.set_ylim(_shared_ylim(twin_g, live_g, twin_epi, live_epi))
        _mark_surge(ax)

    e_refs = [e for e, _, _ in ladder]
    right.plot(
        e_refs,
        [m for _, m, _ in ladder],
        "-o",
        color=gallery.EPISTEMIC_C,
        lw=2.2,
        label="R(Q) agent",
    )
    right.plot(
        e_refs,
        [m for _, _, m in ladder],
        "-s",
        color=gallery.PRAGMATIC_C,
        lw=2.2,
        label="fixed R twin",
    )
    right.set_xscale("log")
    right.set_xticks(e_refs)
    right.set_xticklabels([f"{e * 100:g}%" for e in e_refs])
    # The log locator would otherwise print its own decade minors beside the four
    # rungs, which reads as data points that are not there.
    right.set_xticks([], minor=True)
    right.set_xlabel("meter relative uncertainty at design flow")
    right.set_ylabel("one-step standoff above surge  (% of design flow)")
    right.set_title(
        "the control line moves, or it does not", fontsize=12, fontweight="bold"
    )
    right.annotate(
        "a worse meter runs CLOSER to surge:\n"
        "the ambiguity term is not a safety factor.\n"
        "Read the shape, not the height: Λ sets the height",
        xy=(0.5, -0.30),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#333333",
    )
    right.grid(True, alpha=0.25)
    right.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    fig.tight_layout()
    return gallery.save_figure(fig, out_path, dpi=150, tight=True)


def _shared_ylim(*series) -> tuple[float, float]:
    """One y-range across both EFE panels, so the flat line reads as flat.

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
