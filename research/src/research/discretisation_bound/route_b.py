"""Route B: the trapezoid rule's error on an integrand analytic in a strip.

Trefethen and Weideman, *The exponentially convergent trapezoidal rule*, SIAM Review
56(3), 385–458, 2014, Theorem 5.1. Let `f` be analytic in the strip `|Im z| < a` and
tend to zero uniformly as `|Re z| → ∞` inside it (H1). Let `∫|f(u + iv)| du ≤ M` along
every line `|v| < a` (H2). Then for every step `h > 0`

    |h Σ_{j∈ℤ} f(x₀ + jh) − ∫f|  ≤  2M / (e^{2πa/h} − 1).

The theorem puts its nodes at `jh`. An offset `x₀` is a translation of `f`, which keeps
both the strip and `M`, so the bound holds at every offset.

It bounds the sum over the whole lattice. The engine sums over a box, and the difference
is the nodes outside it plus the two half-weighted ends. That is the tail argument's
term, and nothing here covers it.

Three things here are not the theorem.

- `gaussian_growth_residual` is the fact every `M` in the derivation is built on: a
  Gaussian of variance `s` moved a height `v` off the axis is multiplied by
  `e^{v²/2s}`.
- `log_gaussian_worst_error` is the trapezoid rule's exact error on a Gaussian, from
  Poisson summation. It is the oracle the bounds are checked against. It is independent
  of the theorem, so a mistake in the theorem's constants shows up against it.
- `lattice_sum` evaluates the theorem's left-hand side in floats. That is only
  meaningful at a step coarse enough for the error to stand above roundoff.
"""

import math
from collections.abc import Callable

import numpy as np
import sympy

__all__ = [
    "gaussian_growth_residual",
    "gaussian_lattice_error",
    "lattice_sum",
    "log_error_bound",
    "log_gaussian_worst_error",
]


def log_error_bound(
    line_integral_bound: sympy.Expr,
    strip_half_width: sympy.Expr,
    step: sympy.Expr,
) -> sympy.Expr:
    """`ln(2M / (e^{2πa/h} − 1))`, the theorem's bound in log space.

    On the declared lattice the bound is near `e^{−4000}`, which underflows every float
    type, so it stays a SymPy expression and is evaluated to as many digits as asked.

    Args:
        line_integral_bound: `M`, the H2 constant.
        strip_half_width: `a`.
        step: `h`, the lattice spacing.

    Returns:
        The log of the bound.
    """
    exponent = 2 * sympy.pi * strip_half_width / step
    return (
        sympy.log(2 * line_integral_bound)
        - exponent
        - sympy.log(1 - sympy.exp(-exponent))
    )


def gaussian_growth_residual() -> sympy.Expr:
    """`Re[−(z − m)²/2s] − [−(u − m)²/2s + v²/2s]`, which is zero.

    With `z = u + iv`, `|e^w| = e^{Re w}` turns this into
    `|N(u + iv; m, s)| = N(u; m, s) · e^{v²/2s}` for every real `m` and `s > 0`.

    Returns:
        The residual, simplified.
    """
    u, v, mean = sympy.symbols("u v m", real=True)
    variance = sympy.Symbol("s", positive=True)
    exponent = -((u + sympy.I * v - mean) ** 2) / (2 * variance)
    real_part, _ = sympy.expand(exponent).as_real_imag()
    claimed = -((u - mean) ** 2) / (2 * variance) + v**2 / (2 * variance)
    return sympy.simplify(real_part - claimed)


def log_gaussian_worst_error(variance: sympy.Expr, step: sympy.Expr) -> sympy.Expr:
    """The log of an upper bound on the trapezoid rule's exact error on a Gaussian.

    Poisson summation gives, for a normalised Gaussian of variance `s` and mean `m`,

        h Σ_j N(x₀ + jh) − 1  =  2 Σ_{n≥1} e^{−βn²} cos(2πn(x₀ − m)/h),
        β = 2π²s/h².

    The worst offset puts a node on the mean, where every cosine is one. Then
    `Σ e^{−βn²} ≤ Σ e^{−βn} = 1/(e^β − 1)`, so the error is at most `2/(e^β − 1)`
    at every offset. To leading order that is `2e^{−β}`, which the worst offset attains.

    Args:
        variance: `s`.
        step: `h`.

    Returns:
        `ln(2/(e^β − 1))`.
    """
    rate = 2 * sympy.pi**2 * variance / step**2
    return sympy.log(2) - rate - sympy.log(1 - sympy.exp(-rate))


def gaussian_lattice_error(variance: float, step: float, offset: float) -> float:
    """The exact error of the trapezoid rule on a normalised Gaussian, in floats.

    The Poisson series of `log_gaussian_worst_error`, summed until its terms vanish.
    It is the oracle `lattice_sum` is validated against, at a step coarse enough for
    the error to be visible.

    Args:
        variance: `s`.
        step: `h`.
        offset: `x₀ − m`, the distance from the mean to a node.

    Returns:
        `h Σ_j N(x₀ + jh) − 1`.
    """
    rate = 2.0 * math.pi**2 * variance / step**2
    terms = []
    for harmonic in range(1, 100):
        term = math.exp(-rate * harmonic**2)
        if term == 0.0:
            break
        terms.append(term * math.cos(2.0 * math.pi * harmonic * offset / step))
    return 2.0 * math.fsum(terms)


def lattice_sum(
    integrand: Callable[[np.ndarray], np.ndarray],
    offset: float,
    step: float,
    centre: float,
    reach: float,
) -> float:
    """`h Σ_j f(x₀ + jh)`, over every node within `reach` of `centre`.

    The theorem's sum runs over all of `ℤ`. Cutting it at `reach` changes nothing in
    floats once the integrand there is below roundoff, and the caller picks `reach` so
    that it is.

    Args:
        integrand: `f`, vectorised over real nodes.
        offset: `x₀`, any one node of the lattice.
        step: `h`.
        centre: where the integrand's mass sits.
        reach: how far either side of `centre` to sum.

    Returns:
        The lattice sum.
    """
    first = math.ceil((centre - reach - offset) / step)
    last = math.floor((centre + reach - offset) / step)
    nodes = offset + step * np.arange(first, last + 1)
    return step * math.fsum(integrand(nodes))
