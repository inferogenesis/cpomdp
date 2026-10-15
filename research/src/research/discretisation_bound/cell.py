"""The cell the derivation runs on, as exact numbers.

`research.explorations.threshold` declares `d4-family-v1` at the binding cell in floats,
since the engine runs in floats. The derivation's constants come out exact on it
(`μ* = √10`, `R₀ − κa² = 3/4`, `2πa/h_x = 4000π/3`), so this holds the same cell in
SymPy. `departure_from_declared` measures how far each exact value sits from the float
the threshold module declares, and every runnable module here asserts it is roundoff
before it prints anything. A change to the declared cell fails there rather than moving
a number in the write-up without anyone seeing.

The strip the trapezoid argument runs in is declared here too. Its width is a choice
the derivation makes, not part of the cell, but every bound downstream reads it.
"""

from typing import TYPE_CHECKING

import sympy

from research.explorations import operating_point, threshold

if TYPE_CHECKING:
    from cpomdp._lab.reference.quadrature import QuadratureGrid

__all__ = [
    "BASE_NOISE",
    "CROSSOVER_SPREAD",
    "CURVATURE",
    "GAP_BAR",
    "LARGEST_SPREAD",
    "OBSERVATION_HALF_WIDTH",
    "OBSERVATION_STEP",
    "PLUG_IN_NOISE",
    "POLE_HEIGHT",
    "RIDGE_MEAN",
    "SPREADS",
    "STATE_HALF_WIDTH",
    "STATE_STEP",
    "STRIP_CAP",
    "STRIP_IN_SPREADS",
    "departure_from_declared",
    "observation_noise",
    "predictive_spread",
    "state_grid",
    "strip_half_width",
]

BASE_NOISE = sympy.Integer(1)
"""`R₀` of `d4-family-v1`."""

CURVATURE = sympy.Rational(1, 10)
"""`κ` at the binding cell."""

RIDGE_MEAN = sympy.sqrt(BASE_NOISE / CURVATURE)
"""`μ* = √(R₀/κ) = √10`, the prior mean the family derives."""

POLE_HEIGHT = sympy.sqrt(BASE_NOISE / CURVATURE)
"""Where `R(z)` vanishes, at `z = ±i√(R₀/κ)`.

The same number as `RIDGE_MEAN` on this family, and a different object. One is a
location on the real axis and the other a distance off it.
"""


def observation_noise(state: sympy.Expr) -> sympy.Expr:
    """`R(x) = R₀ + κx²`, at a real or a complex state.

    Args:
        state: `x`, or `z = u + iv` off the real axis.

    Returns:
        The noise variance there.
    """
    return BASE_NOISE + CURVATURE * state**2


PLUG_IN_NOISE = observation_noise(RIDGE_MEAN)
"""`R̄ = R(μ*) = 2R₀`, the noise the exact quantity's plug-in freezes."""

SPREADS = tuple(
    sympy.Rational(1, 200) * sympy.Integer(140) ** sympy.Rational(step, 14)
    for step in range(15)
)
"""The fifteen prior spreads `0.005 · 140^(j/14)`, which `geomspace` rounds."""

LARGEST_SPREAD = SPREADS[-1]
"""`σ = 7/10`, the spread every bound is worst at."""

GAP_BAR = sympy.Float(operating_point.THRESHOLD, 30) / operating_point.K_MIN
"""`T / k_min`, the most `δ_ref` may be for D1 and D2 to be tests."""

STATE_HALF_WIDTH = sympy.Rational(threshold.DECLARED.state_half_width)
"""The state box's half-width in prior spreads, 12."""

STATE_STEP = 2 * STATE_HALF_WIDTH / (threshold.DECLARED.state_nodes - 1)
"""`h_x / σ = 24/1600 = 3/200`."""

OBSERVATION_HALF_WIDTH = sympy.Rational(threshold.DECLARED.observation_half_width)
"""The observation box's half-width in predictive spreads, 9."""

OBSERVATION_STEP = (
    2 * OBSERVATION_HALF_WIDTH / (threshold.DECLARED.observation_nodes - 1)
)
"""`h_y / s_p = 18/400 = 9/200`."""

STRIP_IN_SPREADS = sympy.Integer(10)
"""The strip's half-width in prior spreads below the cap, the `c` of `a = cσ`."""

STRIP_CAP = POLE_HEIGHT / 2
"""`a_cap = √10/2`, half the distance to the pole, so `R₀ − κa_cap² = 3R₀/4` exactly."""

CROSSOVER_SPREAD = STRIP_CAP / STRIP_IN_SPREADS
"""`σ_c = √10/20`, where `cσ` reaches the cap. Not one of `SPREADS`."""


def predictive_spread(spread: sympy.Expr) -> sympy.Expr:
    """`s_p = √(σ² + R̄)`, the unit the observation box is declared in.

    Args:
        spread: the prior spread `σ`.

    Returns:
        The plug-in predictive spread.
    """
    return sympy.sqrt(spread**2 + PLUG_IN_NOISE)


def strip_half_width(spread: sympy.Expr) -> sympy.Expr:
    """`a(σ) = min(cσ, a_cap)`, the strip every Route B bound runs in.

    Args:
        spread: the prior spread `σ`, a number.

    Returns:
        The strip's half-width `a`.
    """
    return sympy.Min(STRIP_IN_SPREADS * spread, STRIP_CAP)


def state_grid(spread: float) -> "QuadratureGrid":
    """The engine's state lattice at one spread, built as `threshold.measure_gap` does.

    `cpomdp` is imported here rather than at the top, as `research.checks.gap_identity`
    does, since it is not a declared dependency of this package.

    Args:
        spread: the prior spread `σ`.

    Returns:
        The declared `QuadratureGrid`, `μ* ± 12σ` with 1601 nodes.
    """
    from cpomdp._lab.reference.quadrature import QuadratureGrid

    reach = threshold.DECLARED.state_half_width * spread
    return QuadratureGrid(
        lower=[threshold.RIDGE_MEAN - reach],
        upper=[threshold.RIDGE_MEAN + reach],
        counts=[threshold.DECLARED.state_nodes],
    )


def departure_from_declared() -> float:
    """The largest relative gap between an exact value here and the declared float.

    Covers `R₀`, `κ`, `μ*`, all fifteen spreads and the two half-widths. Roundoff is
    about `1e−16`. Anything larger means the declared cell moved.

    Returns:
        The worst relative departure.
    """
    pairs = [
        (BASE_NOISE, threshold.BASE_NOISE),
        (CURVATURE, threshold.KAPPA),
        (RIDGE_MEAN, threshold.RIDGE_MEAN),
        (STATE_HALF_WIDTH, threshold.DECLARED.state_half_width),
        (OBSERVATION_HALF_WIDTH, threshold.DECLARED.observation_half_width),
        *zip(SPREADS, threshold.SPREADS, strict=True),
    ]
    return max(
        abs(float(exact) - declared) / abs(declared) for exact, declared in pairs
    )
