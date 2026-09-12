"""Route 1's empirical half: what rescaling the observation does to the reported gap.

Q1 of `research/spinello_stilwell_rung.md` asks for the converged estimate and the
reported gap to be invariant under `o -> lambda*o` once the `r3` block is removed, and
not before. `invariance` answered for the estimate. This answers for the gap, which is
what the ladder reports and what no unit choice can move: a divergence between
distributions over the state, averaged under the true predictive.

One worked case, the symbolic half's, at the four unit choices it compares. Two
schemes, the printed (35) and the modification ADR-057 ships, each at a budget of one,
which is rung (36), and at the rung's declared budget and tolerance, which is rung
(35). The exact posterior and the observation box stay in native units throughout,
since the divergence is over the state, and the scale enters only where the scheme
reads the observation. A run that spends the declared budget, or leaves the reals,
answers `Void` and the sweep reports its weight, as the rung does.

The modification at unit scale is also read against the ladder's own two rungs, which
carry the same equations in vector form. Two implementations agreeing is what lets the
scalar one stand in for the rung here, and it is the one check on this module's
wrapper that does not go through the wrapper.

Not the rung, and no warrant. Every claim printed is asserted. Run::

    uv run --no-sync python -m research.spinello_stilwell.gap_rescaling

`cpomdp` is imported inside the functions that need it, as the other probes do, since
it is not a declared dependency of this package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from research.explorations.threshold import DECLARED
from research.spinello_stilwell import scheme
from research.spinello_stilwell.invariance import CASE, SCALES

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from cpomdp.reference.gap import ApproximatePosterior, Void
    from cpomdp.reference.likelihood import StateDependentNoiseLikelihood
    from cpomdp.reference.quadrature import GridDensity, QuadratureGrid

__all__ = [
    "MODIFIED",
    "PRINTED",
    "SINGLE_STEP",
    "Reading",
    "Variant",
    "declared_budget",
    "ladder_gap",
    "main",
    "measure",
    "relative_spread",
    "scheme_rule",
    "sweep",
]


@dataclass(frozen=True)
class Variant:
    """One of the two schemes, by whether it keeps the `r3` block of (35d)."""

    name: str
    log_block: bool


PRINTED = Variant("printed", log_block=True)
MODIFIED = Variant("modified", log_block=False)

SINGLE_STEP = (1, 0.0)
"""Rung (36): one step, so the tolerance is never consulted."""


def declared_budget() -> tuple[int, float]:
    """Rung (35)'s budget and tolerance, read from where ADR-058 declared them."""
    from cpomdp.reference.ladder import CONVERGENCE_TOLERANCE, ITERATION_BUDGET

    return ITERATION_BUDGET, CONVERGENCE_TOLERANCE


@dataclass(frozen=True)
class Reading:
    """One scheme at one scale and one budget, as the gap reports it.

    Attributes:
        variant: which scheme.
        scale: `lambda`.
        budget: the steps allowed.
        gap: the averaged gap in nats. NaN where the scheme answered no reading.
        voided_mass: the predictive weight the scheme declined.
        predictive_mass: what the observation box caught of `p*`.
    """

    variant: str
    scale: float
    budget: int
    gap: float
    voided_mass: float
    predictive_mass: float


def _quadratic_noise(states: Any, params: Any) -> Any:
    """`R(x) = R0 + kappa x^2` as one `1 x 1` covariance per state, for `jit`."""
    base, curvature = params
    return (base + curvature * states[:, :1] ** 2)[:, :, None]


def scheme_rule(
    variant: Variant, scale: float, budget: int, tolerance: float
) -> ApproximatePosterior:
    """The scalar scheme as the callable the gap measures.

    The prior's mean and variance are read off the density it is handed, and the
    belief is rendered back onto its lattice. The scale reaches only the scheme.

    Args:
        variant: printed or modified.
        scale: `lambda`, the factor the scheme multiplies the observation by.
        budget: most steps to take.
        tolerance: the relative step that ends the run.

    Returns:
        ``rule(prior, observation)``, answering `Void` where the run did not settle.
    """
    quadratic = scheme.quadratic_noise(CASE["base_noise"], CASE["curvature"])

    def rule(prior: GridDensity, observation: ArrayLike) -> GridDensity | Void:
        from cpomdp.reference.gap import Void
        from cpomdp.reference.quadrature import gaussian_on

        # The scheme reads the noise once per step, so the count of reads is the
        # step a failure happened on, which the scheme cannot report itself.
        reads = 0

        def noise_at(state: float) -> tuple[float, float]:
            nonlocal reads
            reads += 1
            return quadratic(state)

        mean, cov = prior.moments
        try:
            # One step past the budget, so a run settling on its last allowed step
            # is told apart from one that spent them all: the rung accepts the first
            # and declines the second, and `taken` alone cannot say which happened.
            estimate, variance, taken = scheme.iterate_with(
                float(np.asarray(observation)[0]),
                float(mean[0]),
                float(cov[0, 0]),
                noise_at,
                scale,
                tolerance,
                budget + 1 if budget > 1 else 1,
                log_block=variant.log_block,
            )
        except ZeroDivisionError:
            return Void(
                iterations=reads, detail="the printed curvature met ln(noise) = 0"
            )
        if taken > budget:
            return Void(iterations=budget, detail="budget spent above the tolerance")
        if not (math.isfinite(estimate) and math.isfinite(variance) and variance > 0):
            return Void(iterations=taken, detail="the iterate left the reals")
        return gaussian_on(prior.grid, estimate, variance)

    return rule


def _grids() -> tuple[QuadratureGrid, GridDensity]:
    """The observation box and the prior on its state box, in native units.

    Sized as the registered lattice is, in prior spreads and in predictive spreads
    about the prior mean, so the box is the one the ladder's readings use.
    """
    from cpomdp.reference.quadrature import QuadratureGrid, gaussian_on

    mean, variance = CASE["prior_mean"], CASE["prior_variance"]
    spread = math.sqrt(variance)
    noise_at_mean = CASE["base_noise"] + CASE["curvature"] * mean**2
    predictive = math.sqrt(variance + noise_at_mean)
    states = QuadratureGrid(
        lower=[mean - DECLARED.state_half_width * spread],
        upper=[mean + DECLARED.state_half_width * spread],
        counts=[DECLARED.state_nodes],
    )
    observations = QuadratureGrid(
        lower=[mean - DECLARED.observation_half_width * predictive],
        upper=[mean + DECLARED.observation_half_width * predictive],
        counts=[DECLARED.observation_nodes],
    )
    return observations, gaussian_on(states, [mean], [[variance]])


def _likelihood() -> StateDependentNoiseLikelihood:
    from cpomdp.reference.likelihood import StateDependentNoiseLikelihood

    return StateDependentNoiseLikelihood(
        [[1.0]],
        observation_noise_fn=_quadratic_noise,
        observation_noise_params=(CASE["base_noise"], CASE["curvature"]),
    )


def measure(variant: Variant, scale: float, budget: int, tolerance: float) -> Reading:
    """The reported gap for one scheme at one scale and one budget."""
    from cpomdp.reference.gap import averaged_inference_gap

    observations, prior = _grids()
    gap = averaged_inference_gap(
        prior,
        _likelihood(),
        scheme_rule(variant, scale, budget, tolerance),
        observations,
    )
    return Reading(
        variant=variant.name,
        scale=scale,
        budget=budget,
        gap=float(gap.value),
        voided_mass=float(gap.voided_mass),
        predictive_mass=float(gap.predictive_mass),
    )


def ladder_gap(budget: int) -> float:
    """The ladder's own rung at this budget, on the same grids, in native units."""
    from cpomdp.reference.gap import averaged_inference_gap
    from cpomdp.reference.ladder import ITERATED_RUNG, SINGLE_STEP_RUNG

    rung = SINGLE_STEP_RUNG if budget == 1 else ITERATED_RUNG
    observations, prior = _grids()
    likelihood = _likelihood()
    gap = averaged_inference_gap(
        prior, likelihood, rung.build(likelihood), observations
    )
    assert gap.voided_mass == 0.0
    return float(gap.value)


def sweep() -> tuple[Reading, ...]:
    """Both schemes at both budgets at every scale, in that order."""
    return tuple(
        measure(variant, scale, budget, tolerance)
        for variant in (PRINTED, MODIFIED)
        for budget, tolerance in (SINGLE_STEP, declared_budget())
        for scale in SCALES
    )


def relative_spread(gaps: list[float]) -> float:
    """`(max - min) / |mean|` over the finite entries, or infinity if any is not."""
    if not all(math.isfinite(gap) for gap in gaps):
        return math.inf
    return (max(gaps) - min(gaps)) / abs(sum(gaps) / len(gaps))


def _select(
    readings: tuple[Reading, ...], variant: Variant, budget: int
) -> list[Reading]:
    return [r for r in readings if r.variant == variant.name and r.budget == budget]


def main() -> None:
    """Run the sweep, print it, and assert what the write-up says of it."""
    budget, tolerance = declared_budget()
    print(
        f"the worked case: prior N({CASE['prior_mean']}, {CASE['prior_variance']}), "
        f"R(x) = {CASE['base_noise']} + {CASE['curvature']} x^2, native units"
    )
    print(
        f"rung (36) at budget 1; rung (35) at budget {budget}, tolerance {tolerance:g}"
    )
    print("the reported gap, E_p*[KL(q || p(x|y))] in nats, with the weight declined\n")
    readings = sweep()
    print(f"  {'scheme':>8} {'budget':>6} {'lambda':>7} {'gap':>22} {'voided':>10}")
    for r in readings:
        print(
            f"  {r.variant:>8} {r.budget:>6} {r.scale:>7} {r.gap:>22.15e} "
            f"{r.voided_mass:>10.2e}"
        )

    print("\n=== the modification is unit-free at both budgets ===")
    for at in (1, budget):
        rows = _select(readings, MODIFIED, at)
        assert all(r.voided_mass == 0.0 for r in rows), rows
        spread = relative_spread([r.gap for r in rows])
        print(f"  budget {at:>3}: relative spread across lambda {spread:.2e}")
        assert spread < 1e-10, (at, spread)

    print("\n=== the scalar modification at lambda = 1 is the ladder's rung ===")
    for at in (1, budget):
        scalar = next(r.gap for r in _select(readings, MODIFIED, at) if r.scale == 1.0)
        vector = ladder_gap(at)
        relative = abs(scalar - vector) / abs(vector)
        print(
            f"  budget {at:>3}: scalar {scalar:.15e}  rung {vector:.15e}  "
            f"rel {relative:.1e}"
        )
        assert relative < 1e-8, (at, relative)

    print("\n=== the printed scheme, run to the declared budget ===")
    settled = _select(readings, MODIFIED, budget)[0].gap
    for r in _select(readings, PRINTED, budget):
        if r.voided_mass > 0.0:
            print(f"  lambda {r.scale:>4}: declined {r.voided_mass:.2e} of the weight")
            continue
        relative = abs(r.gap - settled) / abs(settled)
        print(f"  lambda {r.scale:>4}: same fixed point, rel {relative:.1e}")
        assert relative < 1e-8, (r.scale, relative)

    print("\n=== the printed scheme at a budget of one moves with the units ===")
    printed = [r.gap for r in _select(readings, PRINTED, 1)]
    modified = [r.gap for r in _select(readings, MODIFIED, 1)]
    printed_spread = relative_spread(printed)
    modified_spread = relative_spread(modified)
    print(
        f"  printed spread {printed_spread:.2e} against modified {modified_spread:.2e}"
    )
    assert printed_spread > 1e3 * modified_spread, (printed_spread, modified_spread)
    for r in _select(readings, PRINTED, 1):
        print(f"  lambda {r.scale:>4}: gap {r.gap:.15e}  declined {r.voided_mass:.2e}")

    print("\n=== what the modification costs at rung (36), in native units ===")
    at_one = {r.variant: r.gap for r in readings if r.budget == 1 and r.scale == 1.0}
    cost = at_one["printed"] - at_one["modified"]
    print(
        f"  printed {at_one['printed']:.15e}  modified {at_one['modified']:.15e}  "
        f"difference {cost:+.3e} ({cost / at_one['modified']:+.2e} relative)"
    )
    assert abs(cost) > 1e-12 * abs(at_one["modified"]), cost


if __name__ == "__main__":
    main()
