"""Structural pins on the log-ratio series, before any coefficient is derived.

The inference gap is a cumulant difference. With `q` the agent's plug-in Gaussian and
`p(x|y)` the exact posterior::

    KL(q ‖ p(·|y))  =  log E_q[e^W]  −  E_q[W]

where `W` is the log-ratio of the true likelihood to the plug-in one::

    W(x, y)  =  log N(y; x, R(x))  −  log N(y; x, R̄),        R̄ = R(μ)

Everything the expansion of the gap in prior spread rests on is a property of `W`: that
it vanishes when the noise is flat, that it carries no zeroth-order term in spread, and
that its leading term averages away under `q`. This module pins those properties
symbolically, so the derivation of the quartic coefficient has fixed ground to stand on
rather than conventions recalled from memory.

The construction itself lives in ``series_kernel``, which owns `W`, its expansion and
the moment operator. This module owns the questions. ``gap_series`` owns the
coefficients.

**Nothing here computes a gap coefficient.** Not `c₂`, not `c₄`, not `c₆`. The line is
drawn at the first-order term in `σ` and its expectation, which is where the structure
lives and where the arithmetic does not.

Run it::

    uv run --no-sync python -m research.checks.log_ratio_series --check
    uv run --no-sync python -m research.checks.log_ratio_series

Symbolic throughout. No floats, no numerics, no functional form chosen for `R`: the
log-derivatives `l₁..l₆` stay free symbols and `R̄` stays symbolic and is never set to
one. A check that passes only at `R̄ = 1` is a check that has lost a variable.

Every identity reports `PROVED` at `EXACT`, carrying a `SymbolicReduction` that names
where the symbolic setup was hand derived against the analytic problem. A CAS
establishes that one expression equals another and cannot establish that those are the
expressions the claim is about, which is the condition the warrant ledger attaches to
Prover 2 being theorem-grade. A refuted identity reports `FIRED` at `CORROBORATED`:
simplification is incomplete, so a residual the CAS could not reduce is evidence that
something is wrong rather than proof of what.

Where this sits against the numeric checks. ``gap_kernel`` implements the same three
conventions in quadrature (reverse KL, `R` frozen at `R(μ)`, averaged under the true
predictive), and this module is where they are stated rather than coded. T6 and T8
together say the gap carries nothing at `σ⁰` or `σ¹`, so it starts at `σ²`.
``gap_expansion``'s G4a measures the residual after `c₂σ²` scaling as `σ⁴`. The symbolic
side supplies the reason, the numeric side the confirmation.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

import sympy

from research.checks.series_kernel import (
    CONSTRUCTION_SOURCE,
    DELTA,
    EXPANSION_SOURCE,
    L1,
    L2,
    L3,
    L4,
    NU,
    RBAR,
    SIGMA,
    H,
    Z,
    displacement_series,
    gaussian_expectation,
    kalman_gain,
    log_noise_increment,
    log_ratio,
    log_ratio_in_sigma,
    posterior_sd,
    report_condition,
    report_identity,
)
from warrantlib import CheckReport, Outcome, check_summary

__all__ = ["run_checks"]

#: The highest power of `σ` these pins need. `W`'s first-order term is the last
#: statement before the arithmetic starts, and T8 averages it. Nothing here reaches the
#: second order, which is where a coefficient would appear.
STRUCTURAL_ORDER = 1


def check_reciprocal_identity() -> list[CheckReport]:
    """T1: `1/R(x) − 1/R̄ == (e^{−δ} − 1)/R̄`, exactly and with no expansion.

    The identity that lets the whole `R`-dependence be carried by `δ` alone. It is
    exact, so nothing downstream inherits a truncation from it.

    Returns:
        The report, as a one-item list.
    """
    noise = RBAR * sympy.exp(DELTA)  # R(x) = R̄·e^δ, by the definition of δ
    difference = 1 / noise - 1 / RBAR
    claimed = (sympy.exp(-DELTA) - 1) / RBAR
    residual = sympy.simplify(difference - claimed)
    return [
        report_identity(
            name="T1 reciprocal identity",
            check_id="log_ratio_series.reciprocal_identity",
            claim="1/R(x) − 1/R̄ = (e^−δ − 1)/R̄, exact",
            source=CONSTRUCTION_SOURCE,
            residual=residual,
            shown=f"difference − claim = {residual}",
        )
    ]


def check_flat_noise_vanishing() -> list[CheckReport]:
    """T2: `W ≡ 0` at `δ = 0`, for every `ν` and `h`.

    Flat noise is the case where the Kalman filter is exact, so the log-ratio must
    vanish identically rather than to some order. Checked by substitution, not by
    series: an expansion would only show it vanishes to the order expanded.

    Returns:
        The report, as a one-item list.
    """
    flat = sympy.simplify(log_ratio(sympy.Integer(0), H))
    return [
        report_identity(
            name="T2 flat noise",
            check_id="log_ratio_series.flat_noise",
            claim="W = 0 identically at δ = 0, for all ν and h",
            source=CONSTRUCTION_SOURCE,
            residual=flat,
            shown=f"W|(δ=0) = {flat}",
        )
    ]


def check_gain_and_sd_series() -> list[CheckReport]:
    """T3: the gain and the posterior width, expanded in `σ`.

    `K = σ²/R̄ − σ⁴/R̄² + O(σ⁶)` and `√v_q = σ − σ³/(2R̄) + O(σ⁵)`.

    Returns:
        One report per primitive.
    """
    gain = sympy.expand(kalman_gain().series(SIGMA, 0, 6).removeO())
    width = sympy.expand(posterior_sd().series(SIGMA, 0, 5).removeO())
    return [
        report_identity(
            name="T3 gain",
            check_id="log_ratio_series.gain",
            claim="K = σ²/R̄ − σ⁴/R̄² + O(σ⁶)",
            source=EXPANSION_SOURCE,
            residual=gain - (SIGMA**2 / RBAR - SIGMA**4 / RBAR**2),
            shown=f"K = {gain}",
        ),
        report_identity(
            name="T3 width",
            check_id="log_ratio_series.width",
            claim="√v_q = σ − σ³/(2R̄) + O(σ⁵)",
            source=EXPANSION_SOURCE,
            residual=width - (SIGMA - SIGMA**3 / (2 * RBAR)),
            shown=f"√v_q = {width}",
        ),
    ]


def check_displacement_series() -> list[CheckReport]:
    """T4: `h = σz + (ν/R̄)σ² − (z/2R̄)σ³ + O(σ⁴)`, coefficient by coefficient.

    The `σ²` coefficient gets its own report. It is the Kalman shift, it is free of `z`,
    and it is what separates displacement from the prior mean from displacement from the
    posterior mean. A derivation that measures `h` from the wrong centre loses exactly
    this term, so asserting it is non-zero is asserting the two are not interchangeable.

    **What a lost shift costs, measured rather than assumed.** Deleting the `σ²` term
    from `h` and re-running the three suites fires five checks: T4's own `σ²`
    coefficient and its Kalman-shift report here, and `series_kernel`'s K4 at `σ²`, `σ³`
    and `σ⁴`. K4 fires because the mutation moves the truncation path and leaves the
    `series` arm where it was, so the two stop agreeing. ``gap_series`` is blind to it
    and reports twelve clean checks, because at `σ²` the gap reads only the first-order
    term of `W`.

    T4 is still the check that *names* the shift. It is the only one that asserts the
    coefficient is `ν/R̄`, non-zero and free of `z`, so it says what is wrong. K4 reports
    only that two paths disagree. The shift is wrong at `σ⁴`, so this pin is what
    the quartic work inherits.

    Returns:
        One report per coefficient, plus the shift report.
    """
    expansion = displacement_series(3)
    claims = {1: Z, 2: NU / RBAR, 3: -Z / (2 * RBAR)}
    reports = [
        report_identity(
            name=f"T4 displacement σ^{power}",
            check_id=f"log_ratio_series.displacement_sigma{power}",
            claim=f"the coefficient of σ^{power} is {claim}",
            source=EXPANSION_SOURCE,
            residual=expansion.coeff(SIGMA, power) - claim,
            shown=f"{expansion.coeff(SIGMA, power)}",
        )
        for power, claim in claims.items()
    ]
    shift = expansion.coeff(SIGMA, 2)
    reports.append(
        report_condition(
            name="T4 Kalman shift",
            check_id="log_ratio_series.kalman_shift",
            claim="the σ² coefficient is non-zero and free of z",
            source=EXPANSION_SOURCE,
            holds=shift != 0 and sympy.diff(shift, Z) == 0,
            shown=f"{shift}, with d/dz = {sympy.diff(shift, Z)}",
        )
    )
    return reports


def check_increment_coefficients() -> list[CheckReport]:
    """T5: `δ` has no constant term, and its `h^k` coefficients are `l_k/k!`.

    A convention pin rather than a discovery. It exists so that a later change to how
    the log-derivatives are carried breaks a test here instead of surfacing as a factor
    of `2` or `24` inside the hand derivation.

    Returns:
        The constant-term report, then one per order.
    """
    increment = log_noise_increment()
    constant = increment.subs(H, 0)
    reports = [
        report_identity(
            name="T5 δ constant term",
            check_id="log_ratio_series.delta_constant_term",
            claim="δ vanishes at h = 0",
            source=CONSTRUCTION_SOURCE,
            residual=constant,
            shown=f"δ|(h=0) = {constant}",
        )
    ]
    for order, coefficient in enumerate((L1, L2, L3, L4), start=1):
        claim = coefficient / sympy.factorial(order)
        reports.append(
            report_identity(
                name=f"T5 δ h^{order}",
                check_id=f"log_ratio_series.delta_h{order}",
                claim=f"the coefficient of h^{order} is l{order}/{order}!",
                source=CONSTRUCTION_SOURCE,
                residual=increment.coeff(H, order) - claim,
                shown=f"{increment.coeff(H, order)} against {claim}",
            )
        )
    return reports


def check_no_zeroth_order() -> list[CheckReport]:
    """T6: `W` has no `σ⁰` term.

    At zero spread the belief is a point mass at `μ`, so `δ` vanishes and T2 applies.
    The gap therefore starts at first order in `σ` at the earliest, which is what makes
    an expansion in spread meaningful at all.

    Returns:
        The report, as a one-item list.
    """
    zeroth = sympy.simplify(log_ratio_in_sigma(STRUCTURAL_ORDER).coeff(SIGMA, 0))
    return [
        report_identity(
            name="T6 no σ⁰ term",
            check_id="log_ratio_series.no_sigma0_term",
            claim="W carries no zeroth-order term in σ",
            source=EXPANSION_SOURCE,
            residual=zeroth,
            shown=f"[σ⁰] W = {zeroth}",
        )
    ]


def check_first_order_term() -> list[CheckReport]:
    """T7: `[σ¹] W == (l₁z/2)·(ν²/R̄ − 1)`, linear in `z` with no constant part.

    Only `l₁` appears. The higher log-derivatives cannot reach first order because `δ`
    enters through `h`, and `h` is `O(σ)`. That is the structural reason the first-order
    term is the one that averages away, which T8 then shows.

    Returns:
        The value report, the degree report and the constant-term report.
    """
    first = sympy.expand(log_ratio_in_sigma(STRUCTURAL_ORDER).coeff(SIGMA, 1))
    claim = (L1 * Z / 2) * (NU**2 / RBAR - 1)
    degree = sympy.Poly(first, Z).degree()
    return [
        report_identity(
            name="T7 first-order term",
            check_id="log_ratio_series.first_order_term",
            claim="[σ¹] W = (l₁z/2)(ν²/R̄ − 1)",
            source=EXPANSION_SOURCE,
            residual=first - claim,
            shown=f"{sympy.factor(first)}",
        ),
        report_condition(
            name="T7 degree in z",
            check_id="log_ratio_series.degree_in_z",
            claim="[σ¹] W is degree 1 in z",
            source=EXPANSION_SOURCE,
            holds=degree == 1,
            shown=f"degree = {degree}",
        ),
        report_identity(
            name="T7 constant in z",
            check_id="log_ratio_series.constant_in_z",
            claim="[σ¹] W has no z-free part",
            source=EXPANSION_SOURCE,
            residual=first.subs(Z, 0),
            shown=f"[σ¹] W|(z=0) = {sympy.simplify(first.subs(Z, 0))}",
        ),
    ]


def check_first_order_expectation() -> list[CheckReport]:
    """T8: `E_q[[σ¹] W] == 0`.

    The first-order term is odd in `z`, so it averages away and the gap's leading
    behaviour is pushed to second order. This is the last statement before the
    arithmetic starts, and it is where this module stops.

    The moment operator it uses is checked in ``series_kernel`` against the Gaussian
    integral, so the arm this rests on is independent of the identity it is written as.

    Returns:
        The report, as a one-item list.
    """
    first = log_ratio_in_sigma(STRUCTURAL_ORDER).coeff(SIGMA, 1)
    expectation = gaussian_expectation(first)
    return [
        report_identity(
            name="T8 first-order expectation",
            check_id="log_ratio_series.first_order_expectation",
            claim="E_q[[σ¹] W] = 0",
            source=EXPANSION_SOURCE,
            residual=expectation,
            shown=f"E_q[[σ¹] W] = {expectation}",
        )
    ]


def run_checks() -> list[CheckReport]:
    """Run every pin, in the order the derivation needs them.

    Returns:
        Every pin's report.
    """
    stages: Sequence[Callable[[], list[CheckReport]]] = (
        check_reciprocal_identity,
        check_flat_noise_vanishing,
        check_gain_and_sd_series,
        check_displacement_series,
        check_increment_coefficients,
        check_no_zeroth_order,
        check_first_order_term,
        check_first_order_expectation,
    )
    return [report for stage in stages for report in stage()]


def _print_setup() -> None:
    """Print the symbolic objects the pins are about, for the bare run."""
    print("R̄ symbolic, l₁..l₆ free. No family chosen, no numbers.\n")
    print(f"K        = {kalman_gain()}")
    print(f"√v_q     = {posterior_sd()}")
    print(f"δ(h)     = {log_noise_increment()}")
    print(f"h(σ)     = {displacement_series(3)}")
    print(f"W        = {log_ratio(DELTA, H)}")
    print(
        f"[σ¹] W   = "
        f"{sympy.factor(log_ratio_in_sigma(STRUCTURAL_ORDER).coeff(SIGMA, 1))}"
    )
    print(
        "\nNo gap coefficient is computed here. The coefficients live in gap_series, "
        "\nand this module stops at the first order in σ."
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pins and print them.

    Args:
        argv: command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        Zero when every identity holds, one otherwise.
    """
    parser = argparse.ArgumentParser(
        description="Structural pins on the log-ratio series, before any coefficient."
    )
    parser.add_argument("--check", action="store_true", help="run the check suite")
    arguments = parser.parse_args(argv)

    if not arguments.check:
        _print_setup()
        return 0

    reports = run_checks()
    for report in reports:
        print(report)
    print(f"\n{check_summary(reports)}")
    return 1 if any(r.outcome is Outcome.FIRED for r in reports) else 0


if __name__ == "__main__":
    sys.exit(main())
