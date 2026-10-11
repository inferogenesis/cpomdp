"""Pages 5 to 7 of the derivation: the likelihood integrand in the strip.

At a fixed observation `y` the state-axis integrand is

    f_y(x) = π(x) N(y; x, R(x)),       R(x) = R₀ + κx²,

and Route B needs it analytic in a strip `|Im z| < a` (H1) with a bounded line integral
along every horizontal line in it (H2). Run::

    uv run --no-sync python -m research.discretisation_bound.likelihood_strip

What is checked, in the notes' order:

1. **H1.** `R(u + iv) = A + iB` with `A = R₀ + κ(u² − v²)` and `B = 2κuv`. `R` vanishes
   only at `±i√(R₀/κ)`. `A − (R₀ − κv²) = κu²`, so `Re R > 0` on any strip narrower
   than `√(R₀/κ)`, and the principal `R^{−½}` and `1/R` are analytic there.
2. **Factor 1, the prior.** The growth fact, from `route_b`.
3. **Factor 2, the normaliser.** `|R|² − A² = B²`, so `|R|^{−½} ≤ A^{−½}`.
4. **Factor 3, the exponent.** `Q = Re[(y − z)²/2R]` has the closed form the notes give,
   and the completed square makes `Q + v²/2A` a manifest square over `2|R|²`. So
   `Q ≥ −v²/2A`, uniformly in `u` and `y`. `v²/2(R₀ − κv²)` grows with `v`.
5. **The envelope** `C(a) = (2π(R₀ − κa²))^{−½} e^{a²/2(R₀ − κa²)}` grows with `a`,
   and is attained at `u = y = 0`, `v = a`. So it is the exact supremum of
   `|N(y; z, R(z))|` over the strip, not a loose bound on it.
6. **The numbers at the cell**, with `a_cap = √10/2`, and the page-7 figures they
   correct.
7. **Numerically**, at all fifteen spreads: `|N(y; z, R(z))| ≤ C(a)` on a dense grid,
   and `∫|f_y(u + iv)|du ≤ C(a)e^{v²/2σ²}` along lines in the strip. Corroboration
   only. The inequality is proved by 3 and 4.
8. **Route B against a measured error.** At a step coarse enough that the trapezoid
   error on `f_y` is visible, the bound sits above it.
"""

import math

import numpy as np
import sympy
from scipy.integrate import quad

from research.discretisation_bound import cell, route_b
from research.explorations import threshold

__all__ = [
    "PAGE_SEVEN",
    "VISIBLE_STEPS",
    "completed_square_residual",
    "envelope",
    "envelope_attained_residual",
    "envelope_growth_residual",
    "envelope_peak",
    "exponent_floor_residual",
    "exponent_growth_residual",
    "exponent_residual",
    "line_integral_bound",
    "line_integral_ratio",
    "log_route_b_bound",
    "main",
    "modulus_residual",
    "noise_parts_residuals",
    "pole_locations",
    "real_part_floor_residual",
    "visible_trapezoid_error",
]

VISIBLE_STEPS = (1.0, 1.25, 1.5)
"""Steps, in prior spreads at `σ = 0.7`, where the trapezoid error is above roundoff.

The declared step is `0.015σ`, where the error is near `e^{−900}` and nothing can
measure it. These are coarse on purpose.
"""

PAGE_SEVEN = {
    "envelope_at_cap": 2.43,
    "bound_below_crossover": 1.3e22,
    "bound_at_largest": 31.0,
    "rate_below_crossover": 4189.0,
    "rate_at_largest": 945.0,
    "log_error_below_crossover": math.log(2.6e22) - 4189.0,
    "log_error_at_largest": math.log(62.0) - 945.0,
}
"""Page 7's figures as written, computed with `a = 1.58` as a decimal."""


def _strip_symbols():
    u, v, observation = sympy.symbols("u v y", real=True)
    base_noise, curvature = sympy.symbols("R0 kappa", positive=True)
    return u, v, observation, base_noise, curvature


def _noise_parts(u, v, base_noise, curvature):
    real_part = base_noise + curvature * (u**2 - v**2)  # A
    imaginary_part = 2 * curvature * u * v  # B
    return real_part, imaginary_part


def noise_parts_residuals() -> tuple[sympy.Expr, sympy.Expr]:
    """`Re R(u + iv) − A` and `Im R(u + iv) − B`.

    Returns:
        The two residuals, simplified. Both are zero.
    """
    u, v, _, base_noise, curvature = _strip_symbols()
    noise = sympy.expand(base_noise + curvature * (u + sympy.I * v) ** 2)
    real_part, imaginary_part = noise.as_real_imag()
    claimed_real, claimed_imaginary = _noise_parts(u, v, base_noise, curvature)
    return (
        sympy.simplify(real_part - claimed_real),
        sympy.simplify(imaginary_part - claimed_imaginary),
    )


def pole_locations() -> list[sympy.Expr]:
    """Where `R(z) = R₀ + κz²` vanishes.

    Returns:
        `±i√(R₀/κ)`, symbolic in `R₀` and `κ`.
    """
    _, _, _, base_noise, curvature = _strip_symbols()
    state = sympy.Symbol("z")
    return sympy.solve(base_noise + curvature * state**2, state)


def real_part_floor_residual() -> sympy.Expr:
    """`A − (R₀ − κv²) − κu²`, so `A ≥ R₀ − κv²`.

    Returns:
        The residual. It is zero.
    """
    u, v, _, base_noise, curvature = _strip_symbols()
    real_part, _ = _noise_parts(u, v, base_noise, curvature)
    return sympy.simplify(
        real_part - (base_noise - curvature * v**2) - curvature * u**2
    )


def modulus_residual() -> sympy.Expr:
    """`|R|² − A² − B²`, so `|R| ≥ A` and `|R|^{−½} ≤ A^{−½}`.

    Returns:
        The residual. It is zero.
    """
    u, v, _, base_noise, curvature = _strip_symbols()
    noise = sympy.expand(base_noise + curvature * (u + sympy.I * v) ** 2)
    real_part, imaginary_part = _noise_parts(u, v, base_noise, curvature)
    return sympy.simplify(sympy.Abs(noise) ** 2 - real_part**2 - imaginary_part**2)


def _exponent_parts():
    u, v, observation, base_noise, curvature = _strip_symbols()
    real_part, imaginary_part = _noise_parts(u, v, base_noise, curvature)
    offset = observation - u  # d
    closed = (
        real_part * offset**2 - 4 * curvature * u * v**2 * offset - v**2 * real_part
    ) / (2 * (real_part**2 + imaginary_part**2))
    return u, v, observation, base_noise, curvature, real_part, offset, closed


def exponent_residual() -> sympy.Expr:
    """`Re[(y − z)²/2R(z)]` against the notes' `[Ad² − 4κuv²d − v²A] / 2|R|²`.

    SymPy takes the real part of the quotient itself, so the conjugate step the notes
    do by hand is not restated here.

    Returns:
        The residual, simplified. It is zero.
    """
    u, v, observation, base_noise, curvature, _, _, closed = _exponent_parts()
    state = u + sympy.I * v
    exponent = (observation - state) ** 2 / (2 * (base_noise + curvature * state**2))
    real_part, _ = exponent.as_real_imag()
    return sympy.simplify(real_part - closed)


def completed_square_residual() -> sympy.Expr:
    """`Ad² − 4κuv²d − v²A` against `A(d − 2κuv²/A)² − (v²/A)(A² + B²)`.

    Returns:
        The residual, simplified. It is zero.
    """
    u, v, _, base_noise, curvature = _strip_symbols()
    real_part, imaginary_part = _noise_parts(u, v, base_noise, curvature)
    offset = sympy.Symbol("d", real=True)
    numerator = (
        real_part * offset**2 - 4 * curvature * u * v**2 * offset - v**2 * real_part
    )
    square = real_part * (offset - 2 * curvature * u * v**2 / real_part) ** 2
    floor = (v**2 / real_part) * (real_part**2 + imaginary_part**2)
    return sympy.simplify(numerator - (square - floor))


def exponent_floor_residual() -> sympy.Expr:
    """`Q + v²/2A − A(d − 2κuv²/A)² / 2|R|²`.

    Zero makes `Q + v²/2A` a square over a positive number wherever `A > 0`, so
    `Q ≥ −v²/2A`. That holds for every `u` and `y`.

    Returns:
        The residual, simplified. It is zero.
    """
    u, v, _, base_noise, curvature, real_part, offset, closed = _exponent_parts()
    _, imaginary_part = _noise_parts(u, v, base_noise, curvature)
    square = real_part * (offset - 2 * curvature * u * v**2 / real_part) ** 2
    return sympy.simplify(
        closed
        + v**2 / (2 * real_part)
        - square / (2 * (real_part**2 + imaginary_part**2))
    )


def exponent_growth_residual() -> sympy.Expr:
    """`∂/∂v [v²/2(R₀ − κv²)] − vR₀/(R₀ − κv²)²`.

    The derivative is non-negative for `v ≥ 0` below the pole, so the exponent's bound
    is largest at the strip's edge.

    Returns:
        The residual, simplified. It is zero.
    """
    _, v, _, base_noise, curvature = _strip_symbols()
    floor = base_noise - curvature * v**2
    return sympy.simplify(sympy.diff(v**2 / (2 * floor), v) - v * base_noise / floor**2)


def envelope(
    strip_half_width: sympy.Expr,
    base_noise: sympy.Expr = cell.BASE_NOISE,
    curvature: sympy.Expr = cell.CURVATURE,
) -> sympy.Expr:
    """`C(a) = (2π(R₀ − κa²))^{−½} e^{a²/2(R₀ − κa²)}`, factors 2 and 3 at the edge.

    Args:
        strip_half_width: `a`, below `√(R₀/κ)`.
        base_noise: `R₀`.
        curvature: `κ`.

    Returns:
        The envelope.
    """
    floor = base_noise - curvature * strip_half_width**2
    return sympy.exp(strip_half_width**2 / (2 * floor)) / sympy.sqrt(
        2 * sympy.pi * floor
    )


def envelope_growth_residual() -> sympy.Expr:
    """`∂/∂a log C(a)` against `κa/(R₀ − κa²) + aR₀/(R₀ − κa²)²`.

    Both terms are positive below the pole, so `C(10σ) ≤ C(a_cap)` throughout the
    regime where the strip is `10σ`.

    Returns:
        The residual, simplified. It is zero.
    """
    _, _, _, base_noise, curvature = _strip_symbols()
    width = sympy.Symbol("a", positive=True)
    floor = base_noise - curvature * width**2
    derivative = sympy.diff(sympy.log(envelope(width, base_noise, curvature)), width)
    return sympy.simplify(
        derivative - (curvature * width / floor + width * base_noise / floor**2)
    )


def envelope_attained_residual() -> sympy.Expr:
    """`N(0; ia, R(ia)) − C(a)`.

    At `u = 0` the imaginary part `B` vanishes and `|R| = A`. At `y = u` the completed
    square vanishes too. Both of the notes' inequalities are then equalities, so
    `C(a)` is the supremum.

    Returns:
        The residual, simplified. It is zero.
    """
    _, _, _, base_noise, curvature = _strip_symbols()
    width = sympy.Symbol("a", positive=True)
    state = sympy.I * width
    noise = base_noise + curvature * state**2
    density = sympy.exp(-((0 - state) ** 2) / (2 * noise)) / sympy.sqrt(
        2 * sympy.pi * noise
    )
    return sympy.simplify(density - envelope(width, base_noise, curvature))


def line_integral_bound(spread: sympy.Expr) -> sympy.Expr:
    """H2's `M = C(a) e^{a²/2σ²}`, with `a = min(10σ, a_cap)`.

    At `σ_c` this is the notes' bound for every `σ ≤ σ_c`, since there it uses
    `C(a_cap)` for the whole regime.

    Args:
        spread: the prior spread `σ`.

    Returns:
        `M`, exact.
    """
    width = cell.strip_half_width(spread)
    return envelope(width) * sympy.exp(width**2 / (2 * spread**2))


def log_route_b_bound(spread: sympy.Expr) -> sympy.Expr:
    """Route B on `f_y` at the declared state step `h_x = 0.015σ`.

    Args:
        spread: the prior spread `σ`.

    Returns:
        The log of `2M/(e^{2πa/h_x} − 1)`.
    """
    return route_b.log_error_bound(
        line_integral_bound(spread),
        cell.strip_half_width(spread),
        cell.STATE_STEP * spread,
    )


def _likelihood_off_axis(observation, state):
    noise = threshold.BASE_NOISE + threshold.KAPPA * state**2
    return np.exp(-((observation - state) ** 2) / (2 * noise)) / np.sqrt(
        2 * np.pi * noise
    )


def _prior_off_axis(state, spread):
    return np.exp(-((state - threshold.RIDGE_MEAN) ** 2) / (2 * spread**2)) / (
        math.sqrt(2 * math.pi) * spread
    )


def envelope_peak(strip_half_width: float) -> float:
    """The largest `|N(y; z, R(z))| / C(a)` over a dense grid in the strip.

    `u` and `y` each run over `[−20, 20]` and `v` over `[−a, a]`, with the grid
    through `u = y = 0`, `v = ±a`, where the supremum sits.

    Args:
        strip_half_width: `a`, below `√10`.

    Returns:
        The peak ratio. One to roundoff if the envelope is the supremum.
    """
    u = np.linspace(-20.0, 20.0, 401)[:, None, None]
    v = np.linspace(-strip_half_width, strip_half_width, 41)[None, :, None]
    observation = np.linspace(-20.0, 20.0, 401)[None, None, :]
    ratio = np.abs(_likelihood_off_axis(observation, u + 1j * v)) / float(
        envelope(sympy.nsimplify(strip_half_width))
    )
    return float(ratio.max())


def line_integral_ratio(spread: float, height: float, observation: float) -> float:
    """`∫|f_y(u + iv)| du / (C(a) e^{v²/2σ²})`, in complex floats.

    Args:
        spread: `σ`, one of the declared spreads.
        height: `v`, inside the strip.
        observation: `y`.

    Returns:
        The ratio. At most one if H2's bound holds on this line.
    """
    exact_spread = sympy.nsimplify(spread)
    width = float(cell.strip_half_width(exact_spread))
    scale = float(envelope(cell.strip_half_width(exact_spread))) * math.exp(
        height**2 / (2 * spread**2)
    )
    assert abs(height) < width, (height, width)

    def integrand(u: float) -> float:
        state = u + 1j * height
        value = _prior_off_axis(state, spread) * _likelihood_off_axis(
            observation, state
        )
        return float(abs(value)) / scale

    reach = 40.0 * spread
    value, _ = quad(
        integrand,
        threshold.RIDGE_MEAN - reach,
        threshold.RIDGE_MEAN + reach,
        points=[threshold.RIDGE_MEAN],
        epsabs=1e-14,
        epsrel=1e-12,
        limit=200,
    )
    return value


def visible_trapezoid_error(step_in_spreads: float) -> tuple[float, float]:
    """The trapezoid error on `f_y` at a coarse step, and Route B's bound on it.

    `σ = 0.7`, `y = μ* + s_p/2`, and an offset that puts no node on `μ*`. The strip is
    the derivation's `a_cap`.

    Args:
        step_in_spreads: `h/σ`.

    Returns:
        The measured error and the bound.
    """
    spread = float(cell.LARGEST_SPREAD)
    step = step_in_spreads * spread
    observation = threshold.RIDGE_MEAN + 0.5 * float(
        cell.predictive_spread(cell.LARGEST_SPREAD)
    )

    def integrand(state):
        return _prior_off_axis(state, spread) * _likelihood_off_axis(observation, state)

    reach = 40.0 * spread
    exact, _ = quad(
        integrand,
        threshold.RIDGE_MEAN - reach,
        threshold.RIDGE_MEAN + reach,
        points=[threshold.RIDGE_MEAN],
        epsabs=1e-15,
        epsrel=1e-13,
        limit=200,
    )
    measured = abs(
        route_b.lattice_sum(
            integrand,
            threshold.RIDGE_MEAN + 0.37 * step,
            step,
            threshold.RIDGE_MEAN,
            reach,
        )
        - exact
    )
    bound = route_b.log_error_bound(
        line_integral_bound(cell.LARGEST_SPREAD),
        cell.STRIP_CAP,
        sympy.nsimplify(step_in_spreads) * cell.LARGEST_SPREAD,
    )
    return measured, float(sympy.exp(bound))


def main() -> None:
    """Check pages 5 to 7 and assert each claim."""
    assert cell.departure_from_declared() < 1e-15

    print("=== H1: R(z) off the real axis ===")
    assert noise_parts_residuals() == (0, 0)
    print("  R(u + iv) = A + iB, A = R0 + k(u^2 - v^2), B = 2kuv")
    poles = pole_locations()
    print(f"  R vanishes at {sympy.sstr(poles)}")
    _, _, _, base_noise, curvature = _strip_symbols()
    assert set(poles) == {
        sympy.I * sympy.sqrt(base_noise / curvature),
        -sympy.I * sympy.sqrt(base_noise / curvature),
    }
    assert real_part_floor_residual() == 0
    print("  A - (R0 - kv^2) = ku^2, so Re R > 0 wherever v^2 < R0/k")

    print("\n=== H2: the three factors ===")
    assert route_b.gaussian_growth_residual() == 0
    print("  factor 1: |pi(u + iv)| = pi(u) e^(v^2/2s^2)")
    assert modulus_residual() == 0
    print("  factor 2: |R|^2 - A^2 = B^2, so |R|^(-1/2) <= A^(-1/2)")
    assert exponent_residual() == 0
    print("  factor 3: Q = [Ad^2 - 4kuv^2 d - v^2 A] / 2|R|^2")
    assert completed_square_residual() == 0
    assert exponent_floor_residual() == 0
    print("  Q + v^2/2A is a square over 2|R|^2, so Q >= -v^2/2A for every u and y")
    assert exponent_growth_residual() == 0
    print("  v^2/2(R0 - kv^2) grows with v, so the edge of the strip is the worst line")

    print("\n=== The envelope C(a) ===")
    assert envelope_growth_residual() == 0
    print("  d log C/da = ka/(R0 - ka^2) + aR0/(R0 - ka^2)^2 > 0")
    assert envelope_attained_residual() == 0
    print("  N(0; ia, R(ia)) = C(a): both inequalities are equalities at u = y = 0")

    print("\n=== The numbers at the cell, a_cap = sqrt(10)/2 ===")
    floor = cell.BASE_NOISE - cell.CURVATURE * cell.STRIP_CAP**2
    cap = envelope(cell.STRIP_CAP)
    assert floor == sympy.Rational(3, 4)
    assert (
        sympy.simplify(
            cap - sympy.exp(sympy.Rational(5, 3)) / sympy.sqrt(3 * sympy.pi / 2)
        )
        == 0
    )
    below = line_integral_bound(cell.CROSSOVER_SPREAD)
    largest = line_integral_bound(cell.LARGEST_SPREAD)
    rate_below = 2 * sympy.pi * cell.STRIP_IN_SPREADS / cell.STATE_STEP
    rate_largest = (
        2 * sympy.pi * cell.STRIP_CAP / (cell.STATE_STEP * cell.LARGEST_SPREAD)
    )
    log_below = log_route_b_bound(cell.CROSSOVER_SPREAD)
    log_largest = log_route_b_bound(cell.LARGEST_SPREAD)
    print(f"  R0 - k a_cap^2 = {floor}, C(a_cap) = {float(cap):.5f}")
    print(f"  sigma <= sigma_c: M <= {float(below):.4e}", end="")
    print(f", 2 pi a/h_x = {float(rate_below):.3f}")
    print(f"  sigma =  0.7:     M <= {float(largest):.3f}", end="")
    print(f", 2 pi a/h_x = {float(rate_largest):.3f}")
    print(f"  ln bound: {float(log_below):.3f} below sigma_c", end="")
    print(f", {float(log_largest):.3f} at 0.7")
    assert rate_below == 4000 * sympy.pi / 3
    assert sympy.simplify(rate_largest - 2000 * sympy.sqrt(10) * sympy.pi / 21) == 0

    print("\n=== Page 7 as written, against the recomputation ===")
    assert float(below) <= PAGE_SEVEN["bound_below_crossover"]
    assert float(largest) > PAGE_SEVEN["bound_at_largest"]
    decimal_cap = sympy.Rational(158, 100)
    as_written = envelope(decimal_cap) * sympy.exp(
        decimal_cap**2 / (2 * cell.LARGEST_SPREAD**2)
    )
    assert f"{float(envelope(decimal_cap)):.2f}" == "2.43"
    assert f"{float(as_written):.2f}" == "31.05"
    assert float(log_below) > PAGE_SEVEN["log_error_below_crossover"]
    assert float(log_largest) < PAGE_SEVEN["log_error_at_largest"]
    print("  M <= 1.3e22 holds; M <= 31 does not (31.27, and 31.05 with a = 1.58)")
    print("  2.6e22 e^-4189 is below the bound, since 2 pi a/h_x is 4188.79, not 4189")
    print("  62 e^-945 holds")

    print("\n=== Numerically, at all fifteen spreads ===")
    for width in (0.05, 1.0, float(cell.STRIP_CAP)):
        peak = envelope_peak(width)
        print(f"  a = {width:.4f}: peak |N(y; z, R(z))| / C(a) = {peak:.15f}")
        assert peak <= 1.0 + 1e-12, (width, peak)
        assert peak >= 1.0 - 1e-12, (width, peak)
    worst = 0.0
    for spread in threshold.SPREADS:
        width = float(cell.strip_half_width(sympy.nsimplify(spread)))
        predictive = float(cell.predictive_spread(sympy.nsimplify(spread)))
        for fraction in (0.0, 0.5, 0.99):
            for observation in (
                threshold.RIDGE_MEAN - 9 * predictive,
                threshold.RIDGE_MEAN,
                threshold.RIDGE_MEAN + 9 * predictive,
            ):
                ratio = line_integral_ratio(spread, fraction * width, observation)
                worst = max(worst, ratio)
    print(f"  worst line integral against C(a) e^(v^2/2s^2): {worst:.4f}")
    assert worst <= 1.0, worst

    print("\n=== Route B against a measured trapezoid error, sigma = 0.7 ===")
    for step_in_spreads in VISIBLE_STEPS:
        measured, bound = visible_trapezoid_error(step_in_spreads)
        print(f"  h = {step_in_spreads:.2f} sigma: error {measured:.3e}", end="")
        print(f", bound {bound:.3e}")
        assert 1e-11 < measured <= bound, (step_in_spreads, measured, bound)


if __name__ == "__main__":
    main()
