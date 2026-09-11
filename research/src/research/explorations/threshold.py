"""The threshold `T`, from the reference engine's measured error field.

`research/gate_d4_registration.md`, PRE-REGISTRATION 2026-09-11, fixes everything this
module runs on before it ran: the cell, the lattices, the estimator, how the two biases
combine, the floor on `D` and the predictions. This module is the run. It prints every
number the RESULT entry quotes, and nothing here is a warrant: the field is a
refinement difference and `T` is a declaration derived from it. Run it with
`python -m research.explorations.threshold`.

Three parts, in the order the registration states them.

**The field.** The plug-in gap of `cpomdp.reference` on `d4-family-v1` at the binding
cell, on the declared lattice and on a finer one, at fifteen spreads. The relative
difference is the declared lattice's error, `ε(σ)`. A third lattice at three of the
spreads says whether that estimate has converged. Where `|ε|` is under the roundoff
floor it is read as zero.

**The shift.** An error field enters the D2 fit as a bias on the exponent,
`Cov(v, ε)/Var(v)` with `v = ln σ` uniform over the window. It carries no sample count.

**The threshold.** For each width `D` at or above the floor, the largest edge fraction
`f` whose truncation bias plus the field's shift fits the budget, and the `T` that
gives. `D*` is where `T` is largest.

`cpomdp` is imported inside the function that needs it, as
`research.checks.gap_identity` does, since it is not a declared dependency of this
package.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar

from research.explorations.operating_point import BETA, KAPPA_MIN, LN10
from research.explorations.sigma_max_edge import c2, c4, c6

__all__ = [
    "DECADES_FLOOR",
    "DECLARED",
    "FINE",
    "FINER",
    "ROUNDOFF_FLOOR",
    "SPREADS",
    "ErrorField",
    "FieldPoint",
    "Lattice",
    "Threshold",
    "fitted_shift",
    "measure_field",
    "measure_gap",
    "optimise",
    "series_gap",
    "shift",
    "threshold",
    "truncation_bias",
    "upper_edge",
]

BASE_NOISE = 1.0
"""`R₀` of `d4-family-v1`."""

KAPPA = KAPPA_MIN
"""The binding cell: `T` is evaluated at the `κ` minimising the window width."""

RIDGE_MEAN = math.sqrt(BASE_NOISE / KAPPA)
"""`μ* = √(R₀/κ)`, the prior mean the family derives rather than sweeps."""

SPREADS = tuple(float(s) for s in np.geomspace(0.005, 0.7, 15))
"""The fifteen prior spreads the field is measured at, even in `ln σ`."""

CONVERGENCE_POINTS = (0, 7, 14)
"""Which of `SPREADS` also run on the third lattice: the two ends and the middle."""

ROUNDOFF_FLOOR = 100.0 * 2.0**-52
"""Below this relative size the field is read as zero: roundoff, not discretisation."""

CONVERGENCE_TOLERANCE = 0.10
"""The estimate is converged where replacing fine by finer moves it by under this."""

DECADES_FLOOR = 0.5
"""The declared floor on the window width, one decade in `σ²`."""

DECADES_CEILING = 1.9
"""Where the window's lower edge would leave the measured spreads."""

PREDICTION_SHARE = 0.01
"""The registered prediction: the field's shift is under this share of `β`."""

SUPERSESSION_SHARE = 0.10
"""Above this share of `β` the registered statistical term is superseded."""


@dataclass(frozen=True)
class Lattice:
    """One lattice pair, in the units the registration declares them.

    Attributes:
        state_half_width: the state box's half-width in prior spreads.
        state_nodes: nodes on the state axis.
        observation_half_width: the observation box's half-width in predictive
            spreads, `√(σ² + R(μ*))`.
        observation_nodes: nodes on the observation axis.
    """

    state_half_width: float
    state_nodes: int
    observation_half_width: float
    observation_nodes: int


DECLARED = Lattice(12.0, 1601, 9.0, 401)
"""The lattice `T` is registered against and PR-8's bound certifies."""

FINE = Lattice(18.0, 3201, 13.5, 801)
"""Twice the nodes and boxes half again as wide, which the error is read against."""

FINER = Lattice(27.0, 6401, 20.25, 1601)
"""The same step again, run at three spreads to say whether the estimate converged."""


def quadratic_noise(states, params):
    """`R(x) = R₀ + κ x²`, one `1 x 1` covariance per state. Module-level for `jit`."""
    base, curvature = params
    return (base + curvature * states[:, :1] ** 2)[:, :, None]


def series_gap(spread: float) -> float:
    """The registered expansion `c₂σ² + c₄σ⁴ + c₆σ⁶` at the binding cell."""
    return float(c2(KAPPA) * spread**2 + c4(KAPPA) * spread**4 + c6(KAPPA) * spread**6)


def measure_gap(spread: float, lattice: Lattice) -> tuple[float, float, float]:
    """The plug-in gap on one lattice, with the two checks on the lattice itself.

    Args:
        spread: the prior standard deviation `σ`.
        lattice: which lattice to run on.

    Returns:
        The gap in nats, the predictive mass the observation box caught, and the
        worst edge ratio across the sweep.
    """
    from cpomdp.reference.gap import averaged_inference_gap
    from cpomdp.reference.ladder import PLUG_IN_RUNG
    from cpomdp.reference.likelihood import StateDependentNoiseLikelihood
    from cpomdp.reference.quadrature import QuadratureGrid, gaussian_on

    likelihood = StateDependentNoiseLikelihood(
        [[1.0]],
        observation_noise_fn=quadratic_noise,
        observation_noise_params=(BASE_NOISE, KAPPA),
    )
    state_reach = lattice.state_half_width * spread
    states = QuadratureGrid(
        lower=[RIDGE_MEAN - state_reach],
        upper=[RIDGE_MEAN + state_reach],
        counts=[lattice.state_nodes],
    )
    predictive = math.sqrt(spread**2 + BASE_NOISE + KAPPA * RIDGE_MEAN**2)
    observation_reach = lattice.observation_half_width * predictive
    observations = QuadratureGrid(
        lower=[RIDGE_MEAN - observation_reach],
        upper=[RIDGE_MEAN + observation_reach],
        counts=[lattice.observation_nodes],
    )
    prior = gaussian_on(states, [RIDGE_MEAN], [[spread**2]])
    gap = averaged_inference_gap(
        prior, likelihood, PLUG_IN_RUNG.build(likelihood), observations
    )
    if gap.voided_mass != 0.0:
        raise RuntimeError("the plug-in rung declined a reading, which it cannot")
    return float(gap.value), float(gap.predictive_mass), float(gap.worst_edge_ratio)


@dataclass(frozen=True)
class FieldPoint:
    """The field at one spread.

    Attributes:
        spread: `σ`.
        declared: the gap on the declared lattice.
        fine: the gap on the fine lattice.
        finer: the gap on the finer lattice, or `None` where it was not run.
        predictive_mass: what the declared observation box caught of `p*`.
        worst_edge_ratio: the declared state box's edge ratio.
    """

    spread: float
    declared: float
    fine: float
    finer: float | None
    predictive_mass: float
    worst_edge_ratio: float

    @property
    def raw_error(self) -> float:
        """`(declared − fine)/fine`, before the roundoff floor."""
        return (self.declared - self.fine) / self.fine

    @property
    def error(self) -> float:
        """The field's value: `raw_error`, or zero under the roundoff floor."""
        return 0.0 if abs(self.raw_error) < ROUNDOFF_FLOOR else self.raw_error

    @property
    def converged(self) -> bool | None:
        """Whether the finer lattice moved the estimate by under the tolerance.

        `None` where the finer lattice was not run. Two estimates both under the
        roundoff floor are converged on zero.
        """
        if self.finer is None:
            return None
        against_finer = (self.declared - self.finer) / self.finer
        if abs(self.raw_error) < ROUNDOFF_FLOOR and abs(against_finer) < ROUNDOFF_FLOOR:
            return True
        return abs(self.raw_error - against_finer) <= CONVERGENCE_TOLERANCE * abs(
            against_finer
        )


@dataclass(frozen=True)
class ErrorField:
    """The measured field, read as a function of `v = ln σ` between its points.

    Attributes:
        points: one per spread, in increasing spread.
    """

    points: tuple[FieldPoint, ...]

    def __call__(self, log_spread: float) -> float:
        """`ε` at `ln σ`, linear between the measured spreads and flat beyond them."""
        logs = np.log([p.spread for p in self.points])
        return float(np.interp(log_spread, logs, [p.error for p in self.points]))

    @property
    def knots(self) -> tuple[float, ...]:
        """`ln σ` at the measured spreads, where the interpolant has its kinks."""
        return tuple(math.log(p.spread) for p in self.points)

    @property
    def largest(self) -> float:
        """The largest `|ε|` after the roundoff floor."""
        return max(abs(p.error) for p in self.points)


def measure_field(
    spreads: Sequence[float] = SPREADS,
    convergence_points: Sequence[int] = CONVERGENCE_POINTS,
    declared: Lattice = DECLARED,
    fine: Lattice = FINE,
    finer: Lattice = FINER,
    report: Callable[[str], None] | None = None,
) -> ErrorField:
    """Run the engine on every spread and lattice the registration names.

    Args:
        spreads: the prior spreads, increasing.
        convergence_points: indices into `spreads` that also run on `finer`.
        declared: the lattice the error is of.
        fine: the lattice the error is read against.
        finer: the lattice that says whether that reading converged.
        report: called with one line per spread as it finishes, if given.

    Returns:
        The field.
    """
    points = []
    for index, spread in enumerate(spreads):
        on_declared, mass, edge = measure_gap(spread, declared)
        on_fine, _, _ = measure_gap(spread, fine)
        on_finer = (
            measure_gap(spread, finer)[0] if index in convergence_points else None
        )
        point = FieldPoint(spread, on_declared, on_fine, on_finer, mass, edge)
        points.append(point)
        if report is not None:
            read = "converged" if point.converged else "NOT converged"
            if point.converged is None:
                read = "not checked"
            report(
                f"  sigma={spread:.4f}  gap={on_declared:.6e}  "
                f"eps={point.raw_error:+.3e}  {read}"
            )
    return ErrorField(tuple(points))


def _covariance_slope(
    field: Callable[[float], float],
    lower: float,
    upper: float,
    knots: Sequence[float] = (),
) -> float:
    """`Cov(v, field)/Var(v)` for `v` uniform on `[lower, upper]`.

    `knots` are points where `field` has a kink, handed to the integrator so it does
    not chase them as roundoff.
    """
    width = upper - lower
    mean_point = (lower + upper) / 2
    inside = [k for k in knots if lower < k < upper] or None
    mean_field = quad(field, lower, upper, points=inside, limit=200)[0] / width
    covariance = (
        quad(
            lambda point: (point - mean_point) * (field(point) - mean_field),
            lower,
            upper,
            points=inside,
            limit=200,
        )[0]
        / width
    )
    return covariance / (width**2 / 12)


def shift(field: ErrorField, sigma_max: float, decades: float) -> float:
    """The exponent shift the field leaves on a window, `b_ref`.

    Args:
        field: the measured field.
        sigma_max: the window's upper edge.
        decades: its width.

    Returns:
        `Cov(v, ln(1 + ε))/Var(v)` over `[ln σ_max − D·ln10, ln σ_max]`, signed.
    """
    upper = math.log(sigma_max)
    return _covariance_slope(
        lambda point: math.log1p(field(point)),
        upper - decades * LN10,
        upper,
        knots=field.knots,
    )


def fitted_shift(
    field: ErrorField, sigma_max: float, decades: float, samples: int = 4001
) -> float:
    """The same shift by running the fit, sharing no code with `shift`.

    A pure `σ²` law carrying the field, `ln gap` on `ln σ` by ordinary least squares,
    and the slope read against two.

    Args:
        field: the measured field.
        sigma_max: the window's upper edge.
        decades: its width.
        samples: how many spreads to fit, even in `ln σ`.

    Returns:
        The measured shift, signed.
    """
    upper = math.log(sigma_max)
    points = np.linspace(upper - decades * LN10, upper, samples)
    values = 2.0 * points + np.log1p([field(float(p)) for p in points])
    return float(np.polyfit(points, values, 1)[0] - 2.0)


def truncation_bias(fraction: float, decades: float, sign: float = 1.0) -> float:
    """The bias the sextic left in the fit puts on the exponent, `b_trunc`.

    The correction is `sign·fraction·e^{4u}` with `u = ln σ − ln σ_max` over
    `[−D·ln10, 0]`, so it is exactly `fraction` of the leading term at the top edge.
    `c₆ > 0` at the binding cell, so the registered sign is positive; `−1` is the
    window exploration's sign, kept for the check against it.

    Args:
        fraction: `f`.
        decades: `D`.
        sign: which way the residual bends the curve.

    Returns:
        The bias, signed.
    """
    if fraction <= 0:
        return 0.0
    return _covariance_slope(
        lambda point: math.log1p(sign * fraction * math.exp(4.0 * point)),
        -decades * LN10,
        0.0,
    )


def upper_edge(fraction: float) -> float:
    """`σ_max` under the sextic edge, `σ_max⁴ = f·c₂/|c₆|`, at the binding cell."""
    return float((fraction * c2(KAPPA) / abs(c6(KAPPA))) ** 0.25)


def threshold(fraction: float, decades: float) -> float:
    """`T = c₂^{3/2}·√(f/|c₆|)·10^{−2D}` at the binding cell.

    Equal to `c₂σ_min²`, the leading-order gap at the window's lower edge.
    """
    return float(
        c2(KAPPA) ** 1.5 * math.sqrt(fraction / abs(c6(KAPPA))) * 10.0 ** (-2 * decades)
    )


@dataclass(frozen=True)
class Threshold:
    """What the optimisation settled on.

    Attributes:
        decades: `D*`.
        fraction: `f*`.
        sigma_max: the window's upper edge.
        sigma_min: its lower edge.
        value: `T` in nats.
        truncation: `b_trunc` at the optimum.
        field_shift: `b_ref` at the optimum.
        at_floor: whether `D*` sits on the declared floor.
        fraction_capped: whether `f*` hit the largest fraction the measured spreads
            cover rather than the budget.
    """

    decades: float
    fraction: float
    sigma_max: float
    sigma_min: float
    value: float
    truncation: float
    field_shift: float
    at_floor: bool
    fraction_capped: bool


def _fraction_cap() -> float:
    """The largest `f` whose upper edge stays inside the measured spreads."""
    return float(SPREADS[-1] ** 4 * abs(c6(KAPPA)) / c2(KAPPA))


def _largest_fraction(field: ErrorField, decades: float) -> tuple[float, bool]:
    """`f*(D)`, the largest `f` in the combined budget, and whether it was capped."""
    cap = _fraction_cap()

    def excess(fraction: float) -> float:
        return (
            abs(truncation_bias(fraction, decades))
            + abs(shift(field, upper_edge(fraction), decades))
            - BETA
        )

    if excess(cap) < 0:
        return cap, True
    return brentq(excess, 1e-9, cap, xtol=1e-12), False


def optimise(field: ErrorField) -> Threshold:
    """`D*` and `f*` under the registered rule, and the `T` they give.

    Args:
        field: the measured field.

    Returns:
        The optimum.
    """

    def negative_threshold(decades: float) -> float:
        fraction, _ = _largest_fraction(field, decades)
        return -threshold(fraction, decades)

    found = minimize_scalar(
        negative_threshold,
        bounds=(DECADES_FLOOR, DECADES_CEILING),
        method="bounded",
        options={"xatol": 1e-6},
    )
    decades = float(found.x)
    # A bounded minimiser reports an interior point a step off a binding bound.
    if -negative_threshold(DECADES_FLOOR) >= -float(found.fun):
        decades = DECADES_FLOOR
    fraction, capped = _largest_fraction(field, decades)
    sigma_max = upper_edge(fraction)
    return Threshold(
        decades=decades,
        fraction=fraction,
        sigma_max=sigma_max,
        sigma_min=sigma_max / 10.0**decades,
        value=threshold(fraction, decades),
        truncation=truncation_bias(fraction, decades),
        field_shift=shift(field, sigma_max, decades),
        at_floor=decades == DECADES_FLOOR,
        fraction_capped=capped,
    )


def main() -> None:
    """Measure the field, derive `T`, and print what the RESULT quotes."""
    from research.explorations import c6_window

    print("the sign check: the registered sign differs from the exploration's only")
    print("at second order in f, and agrees with it exactly at that sign")
    for fraction, decades in ((0.02, 1.0), (0.05, 0.5)):
        theirs = c6_window.ols_bias(fraction, decades, 4)
        ours = truncation_bias(fraction, decades, sign=-1.0)
        print(f"  f={fraction} D={decades}: theirs {theirs:+.9f}  ours {ours:+.9f}")
        assert abs(theirs - ours) < 1e-12, (theirs, ours)

    print(f"\nthe field at kappa = {KAPPA}, mu* = {RIDGE_MEAN:.6f}, R0 = {BASE_NOISE}")
    print("  declared lattice: state mu* +/- 12 sigma on 1601 nodes,")
    print("  observation mu* +/- 9 predictive spreads on 401 nodes")
    field = measure_field(report=print)

    print("\nthe lattice itself, at every declared point")
    worst_mass = min(p.predictive_mass for p in field.points)
    worst_edge = max(p.worst_edge_ratio for p in field.points)
    print(f"  smallest predictive mass caught  {worst_mass:.12f}")
    print(f"  largest edge ratio               {worst_edge:.3e}")
    assert worst_mass > 1.0 - 1e-8, worst_mass
    assert worst_edge < 1e-12, worst_edge

    print(
        "\nthe engine against the registered expansion, at the three smallest spreads"
    )
    for point in field.points[:3]:
        series = series_gap(point.spread)
        print(
            f"  sigma={point.spread:.4f}  engine {point.fine:.9e}  "
            f"series {series:.9e}  ratio {point.fine / series:.9f}"
        )
        assert abs(point.fine / series - 1.0) < 1e-3, (point.spread, point.fine, series)

    print("\nconvergence of the error estimate where the third lattice ran")
    for index in CONVERGENCE_POINTS:
        point = field.points[index]
        assert point.finer is not None
        against_finer = (point.declared - point.finer) / point.finer
        print(
            f"  sigma={point.spread:.4f}  vs fine {point.raw_error:+.3e}  "
            f"vs finer {against_finer:+.3e}  "
            f"{'converged' if point.converged else 'NOT converged'}"
        )
        assert point.converged, point
    floored = sum(1 for p in field.points if p.error == 0.0 and p.raw_error != 0.0)
    print(f"  largest |eps| after the floor: {field.largest:.3e}")
    print(
        f"  points read as zero under the roundoff floor: {floored} of {len(SPREADS)}"
    )

    print("\nthe optimisation")
    found = optimise(field)
    print(f"  D* = {found.decades:.4f}  {'(the floor)' if found.at_floor else ''}")
    print(f"  f* = {found.fraction:.6f}  {'(capped)' if found.fraction_capped else ''}")
    print(f"  window sigma in [{found.sigma_min:.6f}, {found.sigma_max:.6f}]")
    print(f"  b_trunc = {found.truncation:+.6f}   b_ref = {found.field_shift:+.3e}")
    assert not found.fraction_capped, "f* left the measured spreads"

    print("\nthe field's shift against the fit, on the chosen window")
    by_fit = fitted_shift(field, found.sigma_max, found.decades)
    print(f"  integral {found.field_shift:+.3e}   fit {by_fit:+.3e}")
    assert abs(by_fit - found.field_shift) <= max(
        0.05 * abs(found.field_shift), 1e-9
    ), (by_fit, found.field_shift)

    print("\nthe prediction")
    largest_shift = max(
        abs(shift(field, upper_edge(f), d))
        for f in np.geomspace(1e-3, _fraction_cap(), 40)
        for d in np.linspace(DECADES_FLOOR, DECADES_CEILING, 15)
    )
    share = largest_shift / BETA
    print(
        f"  largest |b_ref| on any window visited: {largest_shift:.3e} "
        f"= {share:.2%} of beta"
    )
    if share < PREDICTION_SHARE:
        print("  benign, as predicted: the field is carried and does not bind")
    elif share < SUPERSESSION_SHARE:
        print("  prediction MISSED: the term is carried in the constraint and binds")
    else:
        print("  the registered sigma_p is SUPERSEDED: b_ref is the statistical term")

    print("\nT, in nats, at the binding cell")
    for step in (0.0, 0.5, 1.0):
        decades = found.decades + step
        fraction, _ = _largest_fraction(field, decades)
        print(
            f"  D = {decades:.4f}:  f* = {fraction:.6f}  "
            f"T = {threshold(fraction, decades):.6e}"
        )
    gap_at_edge = c2(KAPPA) * found.sigma_min**2
    print(f"  T equals c2 sigma_min^2 = {gap_at_edge:.6e}")
    assert abs(gap_at_edge / found.value - 1.0) < 1e-12


if __name__ == "__main__":
    main()
