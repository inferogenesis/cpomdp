"""Pages 8 to 12 of the derivation: Route B on the four prior-side sums.

The engine computes four sums over the state lattice that depend only on the prior or
on the plug-in rung's Gaussian:

    (a) m_π = Σ_x π            the prior's mass
    (b) Σ_x x π                the numerator of μ̂
    (c) Σ_x (x − μ̂)² π         the numerator of ŝ²
    (d) m_Q = Σ_x q            the plug-in posterior's mass

Each stands for an integral, and Route B bounds the error on each. The integrands are a
Gaussian times a polynomial, so they are entire and every strip is admissible. The strip
is `a = min(10σ, a_cap)`, the one the likelihood integrand needs, kept for all six sums.
Run::

    uv run --no-sync python -m research.discretisation_bound.prior_sums

What is checked:

1. **The four `M`.** Each derivation step in SymPy. `∫π = 1` and
   `E|U − μ*| = σ√(2/π)` by integration, not restated. `|(z − μ̂)²| = (u − μ̂)² + v²`
   through the conjugate. `∫(u − μ̂)²π = σ² + (μ* − μ̂)²`. For (d),
   `a²/σ² + a²/2R₀ − a²/2v_q` is a sum of manifestly non-negative terms once
   `ŝ² ≥ σ²/2` and `R̂ ≥ R₀`.
2. **(d)'s hypothesis.** `ŝ² ≥ σ²/2` is the one place a later bound leans on an earlier
   one, and the tail argument discharges it. Here it is corroborated, not proved, by
   reading the engine's `ŝ²/σ²` at all fifteen spreads.
3. **The table.** Every `M` and every log-error at the crossover and at `σ = 0.7`,
   exact where they are exact. It corrects the table drafted from the scan, and the
   corrections are asserted.
4. **Monotonicity.** Each log-error is nondecreasing in `σ` on both sides of the
   crossover, so `σ = 0.7` bounds all fifteen spreads.
5. **Against exact errors.** Route B at its best strip reproduces the Poisson exponent
   for a Gaussian. On the declared lattice the bounds for (a) and (d) sit above the
   Poisson error. At coarse steps, where the error is visible, all four sit above the
   measured error.
6. **H2 numerically.** Line integrals at all fifteen spreads, against each `M`.
"""

import math

import numpy as np
import sympy
from scipy.integrate import quad

from research.discretisation_bound import cell, prelims, route_b
from research.explorations import threshold

__all__ = [
    "BELOW_CROSSOVER",
    "DRAFTED_AT_LARGEST",
    "SUMS",
    "VISIBLE_STEPS",
    "column_at_largest",
    "engine_variance_ratios",
    "lattice_sum_departure",
    "line_integral_bounds",
    "line_integral_ratios",
    "log_error_bounds",
    "main",
    "mass_integral",
    "mean_absolute_deviation",
    "offset_inflation",
    "plug_in_mass_exponent_residual",
    "poisson_margins",
    "regime_growth",
    "route_b_optimum",
    "second_moment_residual",
    "squared_modulus_residual",
    "visible_trapezoid_errors",
]

SUMS = ("mass", "first_moment", "second_moment", "plug_in_mass")
"""The four sums, (a) to (d) in the notes."""

BELOW_CROSSOVER = tuple(
    spread
    for spread in cell.SPREADS
    if spread * cell.STRIP_IN_SPREADS <= cell.STRIP_CAP
)
"""The declared spreads where the strip is `10σ`. Ten of the fifteen."""

VISIBLE_STEPS = (1.0, 1.25, 1.5)
"""Steps, in prior spreads at `σ = 0.7`, where the trapezoid error is measurable."""

DRAFTED_AT_LARGEST = {
    "first_moment": 67.7,
    "second_moment": 939.0,
    "plug_in_mass": 569.0,
    "rate": 945.0,
}
"""The `σ = 0.7` column of the table drafted from the scan, before this module ran."""

_PRIOR_OFFSET = cell.STATE_HALF_WIDTH
"""`|μ* − μ̂| ≤ 12σ`: `μ̂` is a convex combination of nodes inside the box."""


def _prior(u: sympy.Expr, spread: sympy.Expr) -> sympy.Expr:
    return sympy.exp(-((u - cell.RIDGE_MEAN) ** 2) / (2 * spread**2)) / sympy.sqrt(
        2 * sympy.pi * spread**2
    )


def mass_integral() -> sympy.Expr:
    """`∫π(u) du`, which (a) multiplies by the growth factor.

    Returns:
        The integral. It is one.
    """
    u = sympy.Symbol("u", real=True)
    spread = sympy.Symbol("sigma", positive=True)
    return sympy.simplify(sympy.integrate(_prior(u, spread), (u, -sympy.oo, sympy.oo)))


def mean_absolute_deviation() -> sympy.Expr:
    """`E|U − μ*|` for `U ~ N(μ*, σ²)`, integrated as twice the upper half.

    Returns:
        The deviation, in terms of the symbol `sigma`. It is `σ√(2/π)`.
    """
    u = sympy.Symbol("u", real=True)
    spread = sympy.Symbol("sigma", positive=True)
    upper = sympy.integrate(
        (u - cell.RIDGE_MEAN) * _prior(u, spread), (u, cell.RIDGE_MEAN, sympy.oo)
    )
    return sympy.simplify(2 * upper)


def squared_modulus_residual() -> sympy.Expr:
    """`|(z − μ̂)²|² − ((u − μ̂)² + v²)²`, through `w² · conj(w²)`.

    Both `|(z − μ̂)²|` and `(u − μ̂)² + v²` are non-negative, so a zero here makes them
    equal.

    Returns:
        The residual, expanded. It is zero.
    """
    u, v, estimate = sympy.symbols("u v muhat", real=True)
    squared = (u - estimate + sympy.I * v) ** 2
    return sympy.expand(
        squared * sympy.conjugate(squared) - ((u - estimate) ** 2 + v**2) ** 2
    )


def second_moment_residual() -> sympy.Expr:
    """`∫(u − μ̂)²π(u) du − (σ² + (μ* − μ̂)²)`, for a real `μ̂` held fixed.

    Returns:
        The residual, simplified. It is zero.
    """
    u, estimate = sympy.symbols("u muhat", real=True)
    spread = sympy.Symbol("sigma", positive=True)
    moment = sympy.integrate(
        (u - estimate) ** 2 * _prior(u, spread), (u, -sympy.oo, sympy.oo)
    )
    return sympy.simplify(moment - spread**2 - (cell.RIDGE_MEAN - estimate) ** 2)


def plug_in_mass_exponent_residual() -> sympy.Expr:
    """(d)'s exponent gap, against a sum of terms that are non-negative by inspection.

    Write `ŝ² = σ²/2 + p` and `R̂ = R₀ + r` with `p, r ≥ 0`. Then

        a²/σ² + a²/2R₀ − a²/2v_q = (a²/2)[4p/(σ²(σ² + 2p)) + r/(R₀(R₀ + r))],

    with `v_q` from step 4. Each bracketed term is non-negative, so
    `e^{a²/2v_q} ≤ e^{a²/σ²} e^{a²/2R₀}`.

    Returns:
        The left side minus the right, simplified. It is zero.
    """
    width, spread, base_noise = sympy.symbols("a sigma R0", positive=True)
    variance_excess, noise_excess = sympy.symbols("p r", nonnegative=True)
    prior_variance = spread**2 / 2 + variance_excess  # ŝ²
    noise = base_noise + noise_excess  # R̂
    plug_in = prelims.plug_in_variance(prior_variance, noise)  # v_q
    gap = width**2 / spread**2 + width**2 / (2 * base_noise) - width**2 / (2 * plug_in)
    claimed = (width**2 / 2) * (
        4 * variance_excess / (spread**2 * (spread**2 + 2 * variance_excess))
        + noise_excess / (base_noise * (base_noise + noise_excess))
    )
    return sympy.simplify(gap - claimed)


def offset_inflation(spread: sympy.Expr) -> sympy.Expr:
    """How far (c)'s crude offset bound inflates its `M`, against no offset at all.

    `|μ* − μ̂| ≤ 12σ` turns `σ² + a²` into `145σ² + a²`. The ratio is the cost of not
    assuming `μ̂ ≈ μ*`.

    Args:
        spread: the prior spread `σ`.

    Returns:
        `(145σ² + a²) / (σ² + a²)`, with `a = min(10σ, a_cap)`.
    """
    width = cell.strip_half_width(spread)
    return ((1 + _PRIOR_OFFSET**2) * spread**2 + width**2) / (spread**2 + width**2)


def _bounds_at(spread: sympy.Expr, width: sympy.Expr) -> dict[str, sympy.Expr]:
    growth = sympy.exp(width**2 / (2 * spread**2))
    deviation = spread * sympy.sqrt(2 / sympy.pi)
    return {
        "mass": growth,
        "first_moment": (cell.RIDGE_MEAN + deviation + width) * growth,
        "second_moment": ((1 + _PRIOR_OFFSET**2) * spread**2 + width**2) * growth,
        "plug_in_mass": sympy.exp(
            width**2 / spread**2 + width**2 / (2 * cell.BASE_NOISE)
        ),
    }


def line_integral_bounds(
    spread: sympy.Expr, width: sympy.Expr | None = None
) -> dict[str, sympy.Expr]:
    """H2's `M` for each sum.

    (a) `e^{a²/2σ²}`. (b) `(μ* + σ√(2/π) + a) e^{a²/2σ²}`.
    (c) `(145σ² + a²) e^{a²/2σ²}`.
    (d) `e^{a²/σ²} e^{a²/2R₀}`, conditional on `ŝ² ≥ σ²/2`.

    Args:
        spread: the prior spread `σ`.
        width: the strip's half-width `a`. The derivation's `min(10σ, a_cap)` if not
            given.

    Returns:
        `M` by sum, exact.
    """
    if width is None:
        width = cell.strip_half_width(spread)
    return _bounds_at(spread, width)


def log_error_bounds(
    spread: sympy.Expr, width: sympy.Expr | None = None
) -> dict[str, sympy.Expr]:
    """Route B's bound on each sum at the declared step `h_x = 0.015σ`, in log space.

    Args:
        spread: the prior spread `σ`.
        width: the strip's half-width `a`. The derivation's `min(10σ, a_cap)` if not
            given.

    Returns:
        `ln(2M/(e^{2πa/h_x} − 1))` by sum.
    """
    if width is None:
        width = cell.strip_half_width(spread)
    step = cell.STATE_STEP * spread
    return {
        name: route_b.log_error_bound(bound, width, step)
        for name, bound in line_integral_bounds(spread, width).items()
    }


def regime_growth() -> dict[str, tuple[sympy.Expr, sympy.Expr]]:
    """How each log-error moves with `σ`, on each side of the crossover.

    Below `σ_c` the strip is `10σ`, so `2πa/h_x` is the constant `4000π/3` and only
    `ln M` moves. The first entry is `∂ ln M/∂σ` there.

    Above `σ_c` the strip is `a_cap`. The log-error is `ln 2M − x − ln(1 − e^{−x})`
    with `x = 2πa/h_x`, and the last term grows with `σ` because `x` falls. So it is
    enough that `g = ln M − x` grows. The second entry is
    `g′ − (a/σ²)(400π/3 − k·a/σ)`, with `k = 2` for (d) and 1 otherwise. That
    bracket is at least `400π/3 − 10k > 0` once `a/σ ≤ 10`, which is what `σ ≥ σ_c`
    means.

    Returns:
        By sum, the two expressions. Both must be non-negative for every `σ > 0`.
    """
    spread = sympy.Symbol("sigma", positive=True)
    width = cell.STRIP_CAP
    rate = 2 * sympy.pi * width / (cell.STATE_STEP * spread)
    below = line_integral_bounds(spread, cell.STRIP_IN_SPREADS * spread)
    above = line_integral_bounds(spread, width)
    growth = {}
    for name in SUMS:
        lower = sympy.simplify(
            sympy.diff(sympy.expand_log(sympy.log(below[name]), force=True), spread)
        )
        multiple = 2 if name == "plug_in_mass" else 1
        excess = sympy.diff(sympy.log(above[name]) - rate, spread) - (
            width / spread**2
        ) * (400 * sympy.pi / 3 - multiple * width / spread)
        growth[name] = (lower, sympy.simplify(excess))
    return growth


def route_b_optimum() -> tuple[sympy.Expr, sympy.Expr]:
    """Route B on a Gaussian at its best strip, against the Poisson rate.

    For a normalised Gaussian `M = e^{a²/2s}`, so the bound's exponent is
    `a²/2s − 2πa/h`. Minimised over `a` it is `−2π²s/h²`, which is Poisson's `−β`
    exactly. So the theorem's constants are the sharp ones, and the derivation's
    strip `a = 10σ`, far below the optimum `2πσ²/h = 419σ`, only costs tightness.

    Returns:
        The optimal strip minus `2πs/h`, and the optimal exponent plus `2π²s/h²`. Both
        are zero.
    """
    width, variance, step = sympy.symbols("a s h", positive=True)
    exponent = width**2 / (2 * variance) - 2 * sympy.pi * width / step
    (best,) = sympy.solve(sympy.diff(exponent, width), width)
    return (
        sympy.simplify(best - 2 * sympy.pi * variance / step),
        sympy.simplify(
            exponent.subs(width, best) + 2 * sympy.pi**2 * variance / step**2
        ),
    )


def poisson_margins(spread: sympy.Expr) -> dict[str, float]:
    """Route B's log bound minus the log of the exact Poisson error, on the lattice.

    (a) is a Gaussian of variance `σ²`. (d) is one of variance `v_q`, at least
    `1/(2/σ² + 1/R₀)` under (d)'s hypothesis, and a smaller variance only raises the
    exact error. A positive margin means the bound sits above the truth.

    Args:
        spread: the prior spread `σ`.

    Returns:
        The margin in nats for `mass` and `plug_in_mass`.
    """
    step = cell.STATE_STEP * spread
    bounds = log_error_bounds(spread)
    plug_in_floor = 1 / (2 / spread**2 + 1 / cell.BASE_NOISE)
    exact = {
        "mass": route_b.log_gaussian_worst_error(spread**2, step),
        "plug_in_mass": route_b.log_gaussian_worst_error(plug_in_floor, step),
    }
    return {
        name: float((bounds[name] - error).evalf(30)) for name, error in exact.items()
    }


def engine_variance_ratios() -> list[float]:
    """`ŝ²/σ²` from the engine, at each declared spread on the declared lattice.

    The prior is put on the lattice and normalised as `averaged_inference_gap` does,
    and its moments are the ones the plug-in rung reads.

    Returns:
        One ratio per spread.
    """
    from cpomdp.reference.quadrature import gaussian_on

    ratios = []
    for spread in threshold.SPREADS:
        prior = gaussian_on(
            cell.state_grid(spread), [threshold.RIDGE_MEAN], [[spread**2]]
        ).normalise()
        _, covariance = prior.moments
        ratios.append(float(covariance[0, 0]) / spread**2)
    return ratios


def _gaussian(state, mean, variance):
    return np.exp(-((state - mean) ** 2) / (2 * variance)) / math.sqrt(
        2 * math.pi * variance
    )


def _plug_in_moments(spread: float, observation_offset: float) -> tuple[float, float]:
    noise = float(cell.PLUG_IN_NOISE)
    gain = spread**2 / (spread**2 + noise)
    return (
        threshold.RIDGE_MEAN + gain * observation_offset,
        (1 - gain) * spread**2,
    )


def _off_axis_integrands(spread: float, estimate: float, plug_in: tuple[float, float]):
    mean, variance = plug_in
    return {
        "mass": lambda z: _gaussian(z, threshold.RIDGE_MEAN, spread**2),
        "first_moment": lambda z: z * _gaussian(z, threshold.RIDGE_MEAN, spread**2),
        "second_moment": lambda z: (
            (z - estimate) ** 2 * _gaussian(z, threshold.RIDGE_MEAN, spread**2)
        ),
        "plug_in_mass": lambda z: _gaussian(z, mean, variance),
    }


def line_integral_ratios(spread: float) -> dict[str, float]:
    """The largest `∫|f(u + iv)| du / M` per sum, over lines inside the strip.

    Heights `v ∈ {0, a/2, 0.99a}`. (c) runs at `μ̂ ∈ {μ* − 12σ, μ*, μ* + 12σ}` and (d)
    at a plug-in Gaussian whose observation sits `±9s_p` from `μ*`.

    Args:
        spread: `σ`, one of the declared spreads.

    Returns:
        The worst ratio per sum. At most one if each `M` holds.
    """
    exact_spread = sympy.nsimplify(spread)
    width = float(cell.strip_half_width(exact_spread))
    bounds = {
        name: float(bound) for name, bound in line_integral_bounds(exact_spread).items()
    }
    predictive = float(cell.predictive_spread(exact_spread))
    reach = 40.0 * spread
    worst = dict.fromkeys(SUMS, 0.0)
    for estimate_offset in (-12.0, 0.0, 12.0):
        for observation_offset in (-9.0 * predictive, 9.0 * predictive):
            integrands = _off_axis_integrands(
                spread,
                threshold.RIDGE_MEAN + estimate_offset * spread,
                _plug_in_moments(spread, observation_offset),
            )
            plug_in_mean, _ = _plug_in_moments(spread, observation_offset)
            for fraction in (0.0, 0.5, 0.99):
                height = fraction * width
                for name, integrand in integrands.items():
                    centre = (
                        plug_in_mean if name == "plug_in_mass" else threshold.RIDGE_MEAN
                    )
                    value, _ = quad(
                        lambda u, f=integrand, v=height: float(abs(f(u + 1j * v))),
                        centre - reach,
                        centre + reach,
                        points=[centre],
                        epsabs=0.0,
                        epsrel=1e-12,
                        limit=200,
                    )
                    worst[name] = max(worst[name], value / bounds[name])
    return worst


def visible_trapezoid_errors(step_in_spreads: float) -> dict[str, tuple[float, float]]:
    """Measured trapezoid error and Route B's bound, per sum, at a coarse step.

    `σ = 0.7`, the strip `a_cap`, and an offset that puts no node on `μ*`. (c) runs at
    `μ̂ = μ* + 12σ`, the box's edge, and (d) at the plug-in Gaussian for an observation
    `s_p/2` above `μ*`. Every exact integral is closed form.

    Args:
        step_in_spreads: `h/σ`.

    Returns:
        `(measured, bound)` by sum.
    """
    spread = float(cell.LARGEST_SPREAD)
    step = step_in_spreads * spread
    estimate = threshold.RIDGE_MEAN + 12.0 * spread
    plug_in = _plug_in_moments(
        spread, 0.5 * float(cell.predictive_spread(cell.LARGEST_SPREAD))
    )
    integrands = _off_axis_integrands(spread, estimate, plug_in)
    exact = {
        "mass": 1.0,
        "first_moment": threshold.RIDGE_MEAN,
        "second_moment": spread**2 + (threshold.RIDGE_MEAN - estimate) ** 2,
        "plug_in_mass": 1.0,
    }
    exact_step = sympy.nsimplify(step_in_spreads) * cell.LARGEST_SPREAD
    bounds = line_integral_bounds(cell.LARGEST_SPREAD)
    errors = {}
    for name, integrand in integrands.items():
        bound = route_b.log_error_bound(bounds[name], cell.STRIP_CAP, exact_step)
        measured = abs(
            route_b.lattice_sum(
                integrand,
                threshold.RIDGE_MEAN + 0.37 * step,
                step,
                threshold.RIDGE_MEAN,
                60.0 * spread,
            )
            - exact[name]
        )
        errors[name] = (measured, float(sympy.exp(bound)))
    return errors


def lattice_sum_departure(step_in_spreads: float) -> float:
    """`lattice_sum` on the prior against the Poisson series, at a coarse step.

    The measured side of `visible_trapezoid_errors` is only as good as the lattice sum,
    so it is held to the exact error on the one integrand where that is closed form.

    Args:
        step_in_spreads: `h/σ`.

    Returns:
        The absolute difference between the two errors.
    """
    spread = float(cell.LARGEST_SPREAD)
    step = step_in_spreads * spread
    offset = 0.37 * step
    measured = route_b.lattice_sum(
        lambda state: _gaussian(state, threshold.RIDGE_MEAN, spread**2),
        threshold.RIDGE_MEAN + offset,
        step,
        threshold.RIDGE_MEAN,
        60.0 * spread,
    )
    return abs(
        (measured - 1.0) - route_b.gaussian_lattice_error(spread**2, step, offset)
    )


def column_at_largest(width: sympy.Expr) -> dict[str, float]:
    """The `σ = 0.7` column for a given strip, with `2πa/h_x` beside the four `M`.

    Run at `a = 1.58` it shows where the drafted column came from.

    Args:
        width: the strip's half-width `a`.

    Returns:
        `M` by sum, and `rate` for `2πa/h_x`.
    """
    column = {
        name: float(bound)
        for name, bound in line_integral_bounds(cell.LARGEST_SPREAD, width).items()
    }
    column["rate"] = float(
        2 * sympy.pi * width / (cell.STATE_STEP * cell.LARGEST_SPREAD)
    )
    return column


def _table_row(bounds: dict[str, sympy.Expr], name: str) -> str:
    value = bounds[name]
    log_value = float(sympy.log(value).evalf(30))
    if log_value > 60:
        return f"e^{log_value:.4f}"
    if log_value > 40:
        return f"{math.exp(log_value - 50):.4f} e^50"
    return f"{float(value):.4f}"


def main() -> None:
    """Check pages 8 to 12, print the table and assert every claim in it."""
    assert cell.departure_from_declared() < 1e-15

    print("=== The four M, derived ===")
    assert route_b.gaussian_growth_residual() == 0
    assert mass_integral() == 1
    print("  (a) |pi(u + iv)| = pi(u) e^(v^2/2s^2) and the integral of pi is 1")
    spread = sympy.Symbol("sigma", positive=True)
    assert (
        sympy.simplify(mean_absolute_deviation() - spread * sympy.sqrt(2 / sympy.pi))
        == 0
    )
    print("  (b) E|U - mu*| = sigma sqrt(2/pi), integrated")
    assert squared_modulus_residual() == 0
    assert second_moment_residual() == 0
    print("  (c) |(z - muhat)^2| = (u - muhat)^2 + v^2, moment = s^2 + (mu* - muhat)^2")
    assert plug_in_mass_exponent_residual() == 0
    print("  (d) a^2/s^2 + a^2/2R0 - a^2/2v_q is a sum of non-negative terms")

    below_inflation = offset_inflation(cell.CROSSOVER_SPREAD)
    largest_inflation = offset_inflation(cell.LARGEST_SPREAD)
    print(
        f"  (c)'s 12 sigma offset inflates M by {float(below_inflation):.1f} below"
        f" sigma_c and {float(largest_inflation):.1f} at 0.7"
    )
    assert below_inflation == sympy.Rational(245, 101)
    assert f"{float(largest_inflation):.1f}" == "24.6"

    print("\n=== (d)'s hypothesis, from the engine ===")
    ratios = engine_variance_ratios()
    print(
        f"  shat^2/sigma^2 across the fifteen spreads: {min(ratios):.15f} to"
        f" {max(ratios):.15f}"
    )
    assert min(ratios) > 0.5, ratios
    assert max(abs(ratio - 1.0) for ratio in ratios) < 1e-10, ratios
    print("  shat^2 >= sigma^2/2 with room to spare; corroborated, not proved here")

    print("\n=== The table, a_cap = sqrt(10)/2 ===")
    below = line_integral_bounds(cell.CROSSOVER_SPREAD)
    largest = line_integral_bounds(cell.LARGEST_SPREAD)
    log_below = log_error_bounds(cell.CROSSOVER_SPREAD)
    log_largest = log_error_bounds(cell.LARGEST_SPREAD)
    print(
        f"  {'sum':<15}{'M, sigma <= sigma_c':>20}{'M, sigma = 0.7':>17}"
        f"{'ln err <= sigma_c':>19}{'ln err 0.7':>13}"
    )
    for name in SUMS:
        print(
            f"  {name:<15}{_table_row(below, name):>20}{_table_row(largest, name):>17}"
            f"{float(log_below[name].evalf(30)):>19.3f}"
            f"{float(log_largest[name].evalf(30)):>13.3f}"
        )
    rate_below = (
        2 * sympy.pi * cell.STRIP_CAP / (cell.STATE_STEP * cell.CROSSOVER_SPREAD)
    )
    rate_largest = (
        2 * sympy.pi * cell.STRIP_CAP / (cell.STATE_STEP * cell.LARGEST_SPREAD)
    )
    print(f"  {'2 pi a/h_x':<15}{float(rate_below):>20.3f}{float(rate_largest):>17.3f}")

    growth = sympy.exp(50)
    assert below["mass"] == growth
    assert sympy.simplify(below["second_moment"] - sympy.Rational(49, 8) * growth) == 0
    assert (
        sympy.simplify(below["plug_in_mass"] - sympy.exp(sympy.Rational(405, 4))) == 0
    )
    assert f"{float(below['first_moment'] / growth):.3f}" == "4.870"
    largest_growth = sympy.exp(sympy.Rational(125, 49))
    assert sympy.simplify(largest["mass"] - largest_growth) == 0
    assert (
        sympy.simplify(
            largest["second_moment"] - sympy.Rational(1471, 20) * largest_growth
        )
        == 0
    )
    assert (
        sympy.simplify(largest["plug_in_mass"] - sympy.exp(sympy.Rational(1245, 196)))
        == 0
    )
    assert sympy.simplify(rate_below - 4000 * sympy.pi / 3) == 0
    assert sympy.simplify(rate_largest - 2000 * sympy.sqrt(10) * sympy.pi / 21) == 0
    assert f"{float(largest['first_moment']):.2f}" == "67.97"
    assert f"{float(largest['second_moment']):.1f}" == "942.9"
    assert f"{float(largest['plug_in_mass']):.1f}" == "573.7"

    print("\n=== What the table says about the error ===")
    worst_below = max(float(value.evalf(30)) for value in log_below.values())
    worst_largest = max(float(value.evalf(30)) for value in log_largest.values())
    worst_on_grid = max(
        float(value.evalf(30))
        for spread_value in BELOW_CROSSOVER
        for value in log_error_bounds(spread_value).values()
    )
    print(
        f"  sigma <= sigma_c: largest ln err {worst_below:.3f}, (d), about"
        f" 10^{worst_below / math.log(10):.1f}"
    )
    print(f"    over the ten declared spreads there: {worst_on_grid:.3f}")
    print(
        f"  sigma = 0.7: largest ln err {worst_largest:.3f}, (c), about"
        f" 10^{worst_largest / math.log(10):.1f}"
    )
    assert max(log_below, key=lambda name: float(log_below[name].evalf(30))) == (
        "plug_in_mass"
    )
    assert -4087.0 < worst_below < -4086.0, worst_below
    assert worst_on_grid < -4087.0, worst_on_grid
    assert len(BELOW_CROSSOVER) == 10
    assert max(log_largest, key=lambda name: float(log_largest[name].evalf(30))) == (
        "second_moment"
    )
    assert -939.0 < worst_largest < -938.0, worst_largest
    assert round(worst_below / math.log(10)) == -1775
    assert -408.0 < worst_largest / math.log(10) < -407.0
    print("  'below e^-4087' fails at the crossover; below e^-4086 holds")

    print("\n=== The drafted sigma = 0.7 column, reproduced ===")
    decimal = column_at_largest(sympy.Rational(158, 100))
    exact = column_at_largest(cell.STRIP_CAP)
    for name, drafted in DRAFTED_AT_LARGEST.items():
        print(
            f"  {name:<15} drafted {drafted:<6g} a = 1.58: {decimal[name]:9.3f}"
            f"   a = sqrt(10)/2: {exact[name]:9.3f}"
        )
    assert f"{decimal['first_moment']:.1f}" == "67.7"
    assert round(decimal["second_moment"]) == 939
    assert math.floor(decimal["rate"]) == 945
    assert round(decimal["plug_in_mass"]) == 568
    assert round(exact["plug_in_mass"]) == 574
    print("  (b), (c) and 2 pi a/h_x were computed at a = 1.58; (d) matches neither")

    print("\n=== Every log-error grows with sigma ===")
    for name, (lower, excess) in regime_growth().items():
        assert lower.is_nonnegative, (name, lower)
        assert excess.is_nonnegative, (name, excess)
    assert (400 * sympy.pi / 3 - 20).is_positive
    by_spread = [
        [float(log_error_bounds(value)[name].evalf(30)) for value in cell.SPREADS]
        for name in SUMS
    ]
    assert all(row == sorted(row) for row in by_spread)
    print("  nondecreasing on both sides of the crossover, so sigma = 0.7 is the worst")
    print(f"  worst over the fifteen declared spreads: {max(map(max, by_spread)):.3f}")

    print("\n=== Against exact errors ===")
    best, exponent = route_b_optimum()
    assert best == 0, best
    assert exponent == 0, exponent
    print(
        "  Route B at its best strip, 2 pi s/h, gives Poisson's exponent -2 pi^2 s/h^2"
    )
    margins = [poisson_margins(value) for value in cell.SPREADS]
    smallest = min(min(margin.values()) for margin in margins)
    print(
        f"  declared lattice: Route B is above Poisson by at least {smallest:.0f} nats"
    )
    assert smallest > 0, smallest
    departure = max(lattice_sum_departure(value) for value in VISIBLE_STEPS)
    print(f"  the float lattice sum matches Poisson's exact error to {departure:.1e}")
    assert departure < 1e-14, departure
    for step_in_spreads in VISIBLE_STEPS:
        errors = visible_trapezoid_errors(step_in_spreads)
        cells = ", ".join(
            f"{name} {measured:.2e} <= {bound:.2e}"
            for name, (measured, bound) in errors.items()
        )
        print(f"  h = {step_in_spreads:.2f} sigma: {cells}")
        for name, (measured, bound) in errors.items():
            assert 1e-11 < measured <= bound, (step_in_spreads, name, measured, bound)

    print("\n=== H2 numerically, at all fifteen spreads ===")
    worst_ratio = dict.fromkeys(SUMS, 0.0)
    for value in threshold.SPREADS:
        for name, ratio in line_integral_ratios(value).items():
            worst_ratio[name] = max(worst_ratio[name], ratio)
    for name, ratio in worst_ratio.items():
        print(f"  {name:<15} worst line integral / M = {ratio:.4f}")
        assert ratio <= 1.0, (name, ratio)


if __name__ == "__main__":
    main()
