"""The averaged inference gap in closed form, where a closed form exists.

Under a **fixed** `R` the exact posterior is Gaussian, so the agent's `q` and the truth
differ only in gain and variance and the whole functional collapses to algebra::

    E_{y∼p*}[ KL(q ‖ p(·|y)) ]  =  ½·log(v_p/v_q) + (v_q + (K′−K)²·S) / (2·v_p) − ½

with `K = σ²/(σ²+R)` the exact gain, `K′` the gain a wrong plugged-in `R̂` produces,
`v_p`, `v_q` the two posterior variances, and `S = σ² + R` the innovation variance.

That is the calibration the whole reference filter rests on: it is where the gap is
*known*, so an engine that misses it here is not to be trusted where the answer is
unknown. The identity is proved rather than sampled, in four steps a CAS settles in
under a second each. Integrating the product over `y` in one go does not terminate, and
splitting it is not a workaround: the split is what states the structure. The integrand
is exactly quadratic in the innovation, and the predictive's second moment is `S`.

**What is proved, and what is not.**

- The scalar identity is Prover 2, symbolic, and it holds for every positive `σ²`,
  `R_true` and `R_plug` rather than at sampled values.
- Its **non-negativity is not proved here**. That is Gibbs' inequality, a pen-and-paper
  theorem, and `sympy` returns `None` when asked directly. A check claiming it would be
  claiming the wrong prover.
- The **general observation matrix** case is not proved here either. It holds — the
  exploration measures it against Monte Carlo for a non-square `C` — but by sampling,
  so it is corroborated and no row below asserts it.

**The two engines.** ADR-052 accepted two implementations of this quantity and made a
cross-check the thing that keeps them honest. They do not overlap where this identity
lives: `research.checks.gap_kernel` freezes `R̂ = R(μ)` by construction, so at a
constant `R` its plug-in *is* the truth and its gap is identically zero. It cannot be
driven to a wrong plug-in. Their overlap is the state-dependent case, and that is where
the cross-check below runs.

Run it::

    uv run --no-sync python -m research.checks.gap_identity --check
    uv run --no-sync python -m research.checks.gap_identity
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np
import sympy

from research.checks.gap_kernel import NoiseFamily, Quadrature, core_and_tail
from warrantlib import (
    CheckReport,
    Outcome,
    Provenance,
    SymbolicReduction,
    Tier,
    Warrant,
    check_summary,
)

__all__ = ["PROVENANCE", "closed_form", "main", "run_checks"]

#: Where the form of this identity was established, before anything asserted it. The
#: exploration checks it three ways: scipy quadrature over 45 parameter triples, Monte
#: Carlo for a general observation matrix, and symbolically.
_REGISTERED_REF = "fde795e"

#: The commit these checks were first measured at. Filled in by the commit after the
#: one that landed them, since a commit cannot carry its own hash (ADR-041).
_MEASURED_REF = "98022da"

#: What a CAS cannot supply: that the expressions below are the ones the analytic claim
#: is about. Discharged by the exploration, which reaches the same closed form from a
#: direct quadrature of the definition and never touches this module's algebra.
_CORRESPONDENCE = (
    "research/src/research/explorations/averaged_gap_identity.py: the same closed form "
    "against scipy quadrature of the definition over 45 parameter triples"
)

_ASSUMPTIONS = (
    "scalar state and scalar observation, with a unit observation matrix",
    "R constant in the state, so the exact posterior is Gaussian",
    "the agent differs from the exact filter only in the noise it plugs in",
    "non-negativity is not asserted; that is Gibbs' inequality, prover 1",
)

PROVENANCE = Provenance(
    registered_at=_REGISTERED_REF,
    measured_at=_MEASURED_REF,
    registered="the closed form for the averaged gap under a fixed R, reached by "
    "quadrature of the definition and by Monte Carlo for a general C",
)

#: Free throughout. Nothing here is fitted and no measured number is substituted into
#: any of it, which is what makes agreement with an engine evidence rather than a
#: restatement.
PRIOR_VARIANCE, TRUE_NOISE, PLUGIN_NOISE = sympy.symbols(
    "sigma2 R_true R_plug", positive=True
)
OBSERVATION, PRIOR_MEAN = sympy.symbols("y mu", real=True)


def _pieces() -> tuple[sympy.Expr, ...]:
    """The two gains, the two posterior variances and the innovation variance."""
    innovation_variance = PRIOR_VARIANCE + TRUE_NOISE  # S
    gain = PRIOR_VARIANCE / innovation_variance  # K
    plugin_gain = PRIOR_VARIANCE / (PRIOR_VARIANCE + PLUGIN_NOISE)  # K'
    return (
        gain,
        plugin_gain,
        (1 - gain) * PRIOR_VARIANCE,
        (1 - plugin_gain) * PRIOR_VARIANCE,
        innovation_variance,
    )


def divergence_at_one_observation() -> sympy.Expr:
    """`KL(q ‖ p(·|y))` at a fixed `y`, from the two-Gaussian formula.

    Reverse direction, the agent against the exact posterior. Both posteriors come from
    the same prior and the same reading, and differ only in the noise plugged in.
    """
    gain, plugin_gain, exact_variance, approximate_variance, _ = _pieces()
    exact_mean = PRIOR_MEAN + gain * (OBSERVATION - PRIOR_MEAN)
    approximate_mean = PRIOR_MEAN + plugin_gain * (OBSERVATION - PRIOR_MEAN)
    return (
        sympy.log(sympy.sqrt(exact_variance) / sympy.sqrt(approximate_variance))
        + (approximate_variance + (approximate_mean - exact_mean) ** 2)
        / (2 * exact_variance)
        - sympy.Rational(1, 2)
    )


def closed_form() -> sympy.Expr:
    """The averaged gap, as the identity states it."""
    gain, plugin_gain, exact_variance, approximate_variance, innovation = _pieces()
    return (
        sympy.Rational(1, 2) * sympy.log(exact_variance / approximate_variance)
        + (approximate_variance + (plugin_gain - gain) ** 2 * innovation)
        / (2 * exact_variance)
        - sympy.Rational(1, 2)
    )


def _predictive() -> sympy.Expr:
    """`p*(y)`, the true predictive: `N(y; mu, S)`."""
    innovation = _pieces()[4]
    return sympy.exp(
        -((OBSERVATION - PRIOR_MEAN) ** 2) / (2 * innovation)
    ) / sympy.sqrt(2 * sympy.pi * innovation)


def _proved(name: str, check_id: str, claim: str, shown: str) -> CheckReport:
    """A symbolic identity that reduced to zero. Prover 2, so `PROVED` at `EXACT`."""
    return CheckReport(
        name=name,
        check_id=check_id,
        warrant=Warrant.PROVED,
        outcome=Outcome.NOT_TRIGGERED,
        tier=Tier.EXACT,
        detail=f"PASS — {claim}. got: {shown}",
        evidence=(
            SymbolicReduction(
                claim=claim,
                correspondence=_CORRESPONDENCE,
                assumptions=_ASSUMPTIONS,
            ),
        ),
        provenance=(PROVENANCE,),
    )


def _refuted(name: str, check_id: str, claim: str, shown: str) -> CheckReport:
    """A residual the CAS failed to clear.

    `CORROBORATED`, not `PROVED`, and the asymmetry is deliberate. A residual reduced to
    zero decides the identity. One that fails to reduce decides nothing, since
    simplification is incomplete and a true zero it could not find looks the same from
    here.
    """
    return CheckReport(
        name=name,
        check_id=check_id,
        warrant=Warrant.CORROBORATED,
        outcome=Outcome.FIRED,
        tier=Tier.EXACT,
        detail=f"FAIL — {claim}. got: {shown}",
    )


def _identity(
    name: str, check_id: str, claim: str, residual: sympy.Expr
) -> CheckReport:
    """Report whether `residual` reduces to zero."""
    reduced = sympy.simplify(residual)
    if reduced == 0:
        return _proved(name, check_id, claim, "0")
    return _refuted(name, check_id, claim, str(reduced))


def check_the_integrand_is_quadratic_in_the_innovation() -> CheckReport:
    """`KL(q ‖ p(·|y))` is exactly `A + B·(y − mu)²`, with no higher term.

    This is the step that makes the average tractable, and it is a structural claim
    rather than an approximation: the only place `y` enters is the squared difference
    of the two posterior means, which is quadratic in the innovation by construction.
    """
    gain, plugin_gain, exact_variance, approximate_variance, _ = _pieces()
    constant = (
        sympy.Rational(1, 2) * sympy.log(exact_variance / approximate_variance)
        + approximate_variance / (2 * exact_variance)
        - sympy.Rational(1, 2)
    )
    coefficient = (plugin_gain - gain) ** 2 / (2 * exact_variance)
    residual = sympy.expand(
        divergence_at_one_observation()
        - (constant + coefficient * (OBSERVATION - PRIOR_MEAN) ** 2)
    )
    return _identity(
        "the fixed-y divergence is quadratic in the innovation",
        "gap_identity.integrand_is_quadratic",
        "KL(q ‖ p(·|y)) equals A + B·(y − mu)² exactly",
        residual,
    )


def check_the_predictive_is_a_density() -> CheckReport:
    """`p*` integrates to one over the whole line, symbolically."""
    mass = sympy.integrate(_predictive(), (OBSERVATION, -sympy.oo, sympy.oo))
    return _identity(
        "the true predictive integrates to one",
        "gap_identity.predictive_is_a_density",
        "the integral of p*(y) over the line is one",
        mass - 1,
    )


def check_the_predictive_second_moment() -> CheckReport:
    """The second moment of `p*` about the prior mean is the innovation variance."""
    innovation = _pieces()[4]
    second = sympy.integrate(
        (OBSERVATION - PRIOR_MEAN) ** 2 * _predictive(),
        (OBSERVATION, -sympy.oo, sympy.oo),
    )
    return _identity(
        "the predictive's second moment is the innovation variance",
        "gap_identity.predictive_second_moment",
        "E_{p*}[(y − mu)²] equals sigma2 + R_true",
        second - innovation,
    )


def check_the_closed_form_assembles() -> CheckReport:
    """`A + B·S` is the identity, which is the two steps above combined."""
    gain, plugin_gain, exact_variance, approximate_variance, innovation = _pieces()
    constant = (
        sympy.Rational(1, 2) * sympy.log(exact_variance / approximate_variance)
        + approximate_variance / (2 * exact_variance)
        - sympy.Rational(1, 2)
    )
    coefficient = (plugin_gain - gain) ** 2 / (2 * exact_variance)
    return _identity(
        "the averaged gap assembles from the two moments",
        "gap_identity.closed_form_assembles",
        "A + B·S equals the declared closed form",
        constant + coefficient * innovation - closed_form(),
    )


def check_it_vanishes_where_the_rule_is_right() -> CheckReport:
    """At `R_plug = R_true` the expression is identically zero, not merely small.

    This is what entitles the reference filter's calibration test to assert a hard
    zero rather than a tolerance.
    """
    return _identity(
        "a correct plug-in leaves no gap",
        "gap_identity.vanishes_at_the_truth",
        "the closed form is identically zero at R_plug = R_true",
        closed_form().subs(PLUGIN_NOISE, TRUE_NOISE),
    )


def check_the_direction_is_not_symmetric() -> CheckReport:
    """Reverse and forward are different expressions.

    The pinned convention is reverse, and a convention that made no difference would
    not be worth pinning. Reported as an identity that must *not* reduce to zero, so a
    firing outcome here is the informative one.
    """
    gain, plugin_gain, exact_variance, approximate_variance, innovation = _pieces()
    forward = (
        sympy.Rational(1, 2) * sympy.log(approximate_variance / exact_variance)
        + (exact_variance + (plugin_gain - gain) ** 2 * innovation)
        / (2 * approximate_variance)
        - sympy.Rational(1, 2)
    )
    difference = sympy.simplify(closed_form() - forward)
    claim = "the reverse and forward averaged gaps are different expressions"
    name = "reverse is not forward"
    check_id = "gap_identity.reverse_is_not_forward"
    if difference == 0:
        return _refuted(name, check_id, claim, "they coincide")
    return _proved(name, check_id, claim, "they differ")


#: The state-dependent family the two engines are compared on, and the spreads it is
#: compared at. Declared here rather than borrowed from `gap_kernel.FAMILIES`, which is
#: a registered set that a cross-check has no business extending.
_CROSS_CHECK_FAMILY = NoiseFamily(
    name="1 + x²",
    key="cross_check_quadratic",
    noise=lambda x: 1.0 + x**2,
    prior_mean=1.0,
    unbounded=True,
)
_CROSS_CHECK_SPREADS = (0.10, 0.15, 0.20, 0.30)

#: The bar the two engines are required to agree within. Set well inside the quadrature
#: error either carries, and far inside anything a result would rest on.
_ENGINE_TOLERANCE = 1e-10


def check_the_two_engines_agree() -> CheckReport:
    """ADR-052's obligation: two implementations of one quantity, checked together.

    Run on the state-dependent case, because that is where they overlap. `gap_kernel`
    freezes `R̂ = R(μ)`, so at a constant `R` its gap is identically zero and there is
    nothing to compare.

    Sampling a continuum of spreads, so `CORROBORATED` however many spreads are added.
    """
    from cpomdp.reference.gap import averaged_inference_gap
    from cpomdp.reference.likelihood import StateDependentNoiseLikelihood
    from cpomdp.reference.quadrature import QuadratureGrid, gaussian_on

    centre = _CROSS_CHECK_FAMILY.prior_mean
    plugin = float(_CROSS_CHECK_FAMILY.noise(np.asarray(centre)))
    worst = 0.0
    rows = []
    for spread in _CROSS_CHECK_SPREADS:
        prior_variance = spread**2
        core, tail = core_and_tail(
            _CROSS_CHECK_FAMILY, prior_variance, 12.0, Quadrature()
        )
        kernel_value = core + tail

        states = QuadratureGrid([centre - 14.0], [centre + 14.0], [12001])
        observations = QuadratureGrid([centre - 26.0], [centre + 26.0], [2001])
        prior = gaussian_on(states, centre, prior_variance)
        gain = prior_variance / (prior_variance + plugin)

        def rule(belief, observation, gain=gain, prior_variance=prior_variance):
            mean = centre + gain * (float(np.asarray(observation)[0]) - centre)
            variance = (1.0 - gain) * prior_variance
            return gaussian_on(belief.grid, mean, variance)

        measured = averaged_inference_gap(
            prior,
            StateDependentNoiseLikelihood(
                [[1.0]],
                observation_noise_fn=_quadratic_noise,
                observation_noise_params=(1.0, 1.0),
            ),
            rule,
            observations,
        )
        relative = abs(measured.value - kernel_value) / kernel_value
        worst = max(worst, relative)
        rows.append(f"σ={spread:.2f} rel {relative:.1e}")

    claim = (
        f"gap_kernel and cpomdp.reference agree on R(x) = 1 + x² within "
        f"{_ENGINE_TOLERANCE:.0e}"
    )
    detail = f"{claim}. worst {worst:.2e} over {', '.join(rows)}"
    return CheckReport(
        name="the two engines agree on the state-dependent case",
        check_id="gap_identity.engines_agree",
        warrant=Warrant.CORROBORATED,
        outcome=Outcome.NOT_TRIGGERED if worst < _ENGINE_TOLERANCE else Outcome.FIRED,
        tier=Tier.BOUNDED,
        detail=("PASS — " if worst < _ENGINE_TOLERANCE else "FAIL — ") + detail,
    )


def _quadratic_noise(states, params):
    """`R(x) = R0 + kappa·x²`, one 1x1 covariance per state.

    Module level rather than a closure, so `jit` caches on it by identity.
    """
    r0, kappa = params
    return (r0 + kappa * states[:, :1] ** 2)[:, :, None]


def run_checks() -> list[CheckReport]:
    """Run every identity, then the cross-engine comparison.

    Returns:
        Every check's report.
    """
    return [
        check_the_integrand_is_quadratic_in_the_innovation(),
        check_the_predictive_is_a_density(),
        check_the_predictive_second_moment(),
        check_the_closed_form_assembles(),
        check_it_vanishes_where_the_rule_is_right(),
        check_the_direction_is_not_symmetric(),
        check_the_two_engines_agree(),
    ]


def _print_setup() -> None:
    """Show the identity and what bounds it, without asserting anything."""
    print("The averaged inference gap under a FIXED R, in closed form.\n")
    print(f"  KL(q ‖ p(·|y))   = {sympy.simplify(divergence_at_one_observation())}\n")
    print(f"  E_{{p*}}[ KL ]      = {sympy.simplify(closed_form())}\n")
    print("Hypotheses: scalar state, unit observation matrix, R constant in the state.")
    print("Not proved here: non-negativity (Gibbs, prover 1), and the general")
    print("observation-matrix form (corroborated by Monte Carlo in the exploration).")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the checks and print them.

    Args:
        argv: command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        Zero when every check holds, one otherwise.
    """
    parser = argparse.ArgumentParser(
        description="The averaged inference gap in closed form, under a fixed R."
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
    if any(report.outcome is Outcome.FIRED for report in reports):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
