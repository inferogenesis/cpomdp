"""The ladder's first reading: five rungs on `d4-family-v1`, ordered with measured bars.

Pre-registered under D1 of `research/fep_falsification_battery.md` and under Q5 of
`research/spinello_stilwell_rung.md`, both dated 2026-09-12, at `ac9fbc5`. That entry
fixes the cells, the lattices, the bar, the minimum separation, the direction and the
void rule. This module is the run. Nothing here is the D1 test, which runs in PR-9
under a certified bar: every ordering row is `COMPUTED`, its bar measured by
refinement and not certified, and the rows say so.

Five gaps at fifteen spreads on two lattices. A cell's value is its gap on the declared
lattice, and its bar is the refinement difference against the fine one, floored at
roundoff, carried whole as the quantity's own with nothing cancelled between rungs.
Adjacent pairs are resolved through `cpomdp.resolution` at every spread, and one row
per pair reads the orders together: `FIRED` on a resolved `BELOW` anywhere,
`NOT_RESOLVED` where the pair resolves nowhere, `NOT_TRIGGERED` otherwise.

Two rows read route 6 off the same numbers: whether the derivative-of-covariance terms
and the iteration are each visible as a resolved difference at one spread or more.

The plug-in rung's row is the R6 signal PR-8 compares against `T`. It prints the gap
beside `T` at every spread and asserts agreement with the threshold exploration's
engine, since both run one engine on one lattice. It renders no verdict on the gate.

`cpomdp` is imported inside the functions that need it, as the other suites do, since
it is not a declared dependency of this package.

Run it::

    uv run --no-sync python -m research.checks.ladder --check
    uv run --no-sync python -m research.checks.ladder
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from research.explorations.operating_point import THRESHOLD
from research.explorations.threshold import (
    BASE_NOISE,
    DECLARED,
    FINE,
    KAPPA,
    RIDGE_MEAN,
    ROUNDOFF_FLOOR,
    SPREADS,
    Lattice,
    measure_gap,
    quadratic_noise,
)
from warrantlib import (
    AxisDeclaration,
    CheckReport,
    Outcome,
    ProductCompletenessCertificate,
    Provenance,
    Tier,
    Warrant,
    check_summary,
)

if TYPE_CHECKING:
    from cpomdp.reference.gap import ApproximatePosterior
    from cpomdp.reference.quadrature import QuadratureGrid
    from cpomdp.resolution import Bar, Bounded, Order

__all__ = [
    "ENGINE_TOLERANCE",
    "PAIRS",
    "Cell",
    "PairAtSpread",
    "Reading",
    "Sweep",
    "compare",
    "main",
    "measure",
    "outcome_of",
    "run_checks",
]

#: The commit the PRE-REGISTRATION of 2026-09-12 landed in.
_REGISTERED_REF = "ac9fbc5"

#: The commit whose tree produced the numbers, which is the one this suite landed in.
#: Filled by the commit after it, since a commit cannot carry its own hash (ADR-041).
#: Until then the certificate reports CORROBORATED and says why.
_MEASURED_REF: str | None = None

ENGINE_TOLERANCE = 1e-12
"""Relative agreement the plug-in row asks of the threshold exploration's engine."""

PAIRS: tuple[tuple[str, str, str], ...] = (
    ("plug-in", "modified-single-step", "plug_in_to_single_step"),
    ("modified-single-step", "modified-iterated", "single_step_to_iterated"),
    ("modified-iterated", "belief-smoothed", "iterated_to_belief_smoothed"),
    ("belief-smoothed", "exact", "belief_smoothed_to_exact"),
)
"""The adjacent pairs, lower rung first, with the key each row reports under."""

#: The rung whose runs can spend the budget. Its pairs read NOT_RESOLVED at a spread it
#: voids, per the registration, rather than on the conditional value.
_ITERATED = "modified-iterated"


@dataclass(frozen=True)
class Reading:
    """One rung's sweep at one spread on one lattice, as the gap reports it.

    Attributes:
        gap: the averaged gap in nats. NaN where the rung answered no reading.
        predictive_mass: what the observation box caught of `p*`.
        worst_edge_ratio: the state box's edge ratio across the sweep.
        voided_mass: the predictive weight the rung declined.
    """

    gap: float
    predictive_mass: float
    worst_edge_ratio: float
    voided_mass: float


@dataclass(frozen=True)
class Cell:
    """One rung at one spread: the declared reading, the fine one, and the bar.

    Attributes:
        rung: the rung's declared name.
        spread: `σ`.
        declared: the reading on the lattice `T` is registered under.
        fine: the reading on the lattice with twice the nodes and wider boxes.
    """

    rung: str
    spread: float
    declared: Reading
    fine: Reading

    @property
    def value(self) -> float:
        """The gap on the declared lattice, which is what the ordering reads."""
        return self.declared.gap

    @property
    def voided(self) -> bool:
        """Set aside: weight declined above the roundoff floor, or no gap at all."""
        return not math.isfinite(self.declared.gap)

    @property
    def bar(self) -> Bar:
        """The refinement difference, floored at roundoff, as the cell's own bar.

        Nothing is carried as common mode. The five gaps share one reference and a
        common-mode part exists, and claiming it needs the error's shape, which PR-8
        certifies. The conservative reading is the registered one until then.
        """
        from cpomdp.resolution import Bar

        refinement = abs(self.declared.gap - self.fine.gap)
        floor = ROUNDOFF_FLOOR * abs(self.declared.gap)
        return Bar(common_mode=0.0, own=max(refinement, floor))

    @property
    def bounded(self) -> Bounded:
        """The value with its bar, ready to resolve against a neighbour's."""
        from cpomdp.resolution import Bounded

        return Bounded(value=self.value, bar=self.bar)


@dataclass(frozen=True)
class Sweep:
    """Every cell, keyed by rung name in ladder order, each in spread order."""

    cells: dict[str, tuple[Cell, ...]]

    @property
    def names(self) -> tuple[str, ...]:
        """The rungs read, in the order they were read."""
        return tuple(self.cells)

    def read_at_every_spread(self, rung: str) -> bool:
        """Whether the rung produced a finite gap at each spread."""
        return all(math.isfinite(cell.value) for cell in self.cells[rung])


@dataclass(frozen=True)
class PairAtSpread:
    """One adjacent comparison at one spread.

    Attributes:
        spread: `σ`.
        order: how the lower rung's gap sits against the higher rung's.
        difference: lower minus higher, in nats. NaN where a rung voided.
        threshold: what the difference had to exceed. NaN where a rung voided.
        voided: whether the comparison was set aside for a voided reading.
    """

    spread: float
    order: Order
    difference: float
    threshold: float
    voided: bool


def _grids(spread: float, lattice: Lattice) -> tuple[QuadratureGrid, QuadratureGrid]:
    """The state and observation boxes at one spread, sized as the registration says.

    The same construction `research.explorations.threshold.measure_gap` uses, written
    here rather than shared so the plug-in row's agreement with that engine is a
    check on these grids and not a tautology.
    """
    from cpomdp.reference.quadrature import QuadratureGrid

    state_reach = lattice.state_half_width * spread
    predictive = math.sqrt(spread**2 + BASE_NOISE + KAPPA * RIDGE_MEAN**2)
    observation_reach = lattice.observation_half_width * predictive
    return (
        QuadratureGrid(
            lower=[RIDGE_MEAN - state_reach],
            upper=[RIDGE_MEAN + state_reach],
            counts=[lattice.state_nodes],
        ),
        QuadratureGrid(
            lower=[RIDGE_MEAN - observation_reach],
            upper=[RIDGE_MEAN + observation_reach],
            counts=[lattice.observation_nodes],
        ),
    )


def measure(report: Callable[[str], None] | None = None) -> Sweep:
    """Every rung of the declared ladder at every spread, on both lattices.

    Args:
        report: where to print a line per rung as it completes, since the run takes
            minutes. ``None`` prints nothing.

    Returns:
        The sweep.
    """
    from cpomdp.reference.gap import averaged_inference_gap
    from cpomdp.reference.ladder import LADDER
    from cpomdp.reference.likelihood import StateDependentNoiseLikelihood
    from cpomdp.reference.quadrature import gaussian_on

    likelihood = StateDependentNoiseLikelihood(
        [[1.0]],
        observation_noise_fn=quadratic_noise,
        observation_noise_params=(BASE_NOISE, KAPPA),
    )

    def read(rule: ApproximatePosterior, spread: float, lattice: Lattice) -> Reading:
        states, observations = _grids(spread, lattice)
        prior = gaussian_on(states, [RIDGE_MEAN], [[spread**2]])
        gap = averaged_inference_gap(prior, likelihood, rule, observations)
        return Reading(
            gap=float(gap.value),
            predictive_mass=float(gap.predictive_mass),
            worst_edge_ratio=float(gap.worst_edge_ratio),
            voided_mass=float(gap.voided_mass),
        )

    cells: dict[str, tuple[Cell, ...]] = {}
    for name, rule in LADDER.build_all(likelihood):
        cells[name] = tuple(
            Cell(name, spread, read(rule, spread, DECLARED), read(rule, spread, FINE))
            for spread in SPREADS
        )
        if report is not None:
            worst = max(cell.bar.own for cell in cells[name])
            report(f"  {name}: read at {len(SPREADS)} spreads, widest bar {worst:.1e}")
    return Sweep(cells)


def compare(sweep: Sweep, lower: str, higher: str) -> tuple[PairAtSpread, ...]:
    """The lower rung against the higher at every spread, at each difference's error.

    Args:
        sweep: the measured cells.
        lower: the rung predicted to sit above.
        higher: the rung predicted to sit below.

    Returns:
        One comparison per spread. A spread where either rung voided reads
        ``NOT_RESOLVED`` without resolving, per the registration.
    """
    from cpomdp.resolution import Order, resolve

    comparisons = []
    for a, b in zip(sweep.cells[lower], sweep.cells[higher], strict=True):
        if a.voided or b.voided:
            comparisons.append(
                PairAtSpread(a.spread, Order.NOT_RESOLVED, math.nan, math.nan, True)
            )
            continue
        resolution = resolve(a.bounded, b.bounded)
        comparisons.append(
            PairAtSpread(
                a.spread,
                resolution.order,
                resolution.difference.value,
                resolution.threshold,
                False,
            )
        )
    return tuple(comparisons)


def outcome_of(comparisons: Sequence[PairAtSpread]) -> Outcome:
    """The registered reading of one pair across the spreads.

    ``FIRED`` if the pair resolves ``BELOW`` anywhere, ``NOT_RESOLVED`` if it resolves
    nowhere, ``NOT_TRIGGERED`` otherwise.
    """
    from cpomdp.resolution import Order

    orders = [comparison.order for comparison in comparisons]
    if any(order is Order.BELOW for order in orders):
        return Outcome.FIRED
    if all(order is Order.NOT_RESOLVED for order in orders):
        return Outcome.NOT_RESOLVED
    return Outcome.NOT_TRIGGERED


def _letters(comparisons: Sequence[PairAtSpread]) -> str:
    """The orders across the spreads as one string, ``A``, ``B``, ``N`` or ``V``."""
    return "".join("V" if c.voided else c.order.name[0] for c in comparisons)


def _ordering_row(sweep: Sweep, lower: str, higher: str, key: str) -> CheckReport:
    comparisons = compare(sweep, lower, higher)
    letters = _letters(comparisons)
    resolved = [c for c in comparisons if not c.voided and not math.isnan(c.threshold)]
    margin = min(
        (abs(c.difference) / c.threshold for c in resolved if c.threshold > 0.0),
        default=math.nan,
    )
    return CheckReport(
        name=f"{lower} → {higher}: the gap decreases",
        check_id=f"ladder.{key}",
        warrant=Warrant.CORROBORATED,
        outcome=outcome_of(comparisons),
        tier=Tier.COMPUTED,
        detail=(
            f"above at {letters.count('A')}/{len(letters)} spreads, not resolved at "
            f"{letters.count('N')}, below at {letters.count('B')}, voided at "
            f"{letters.count('V')}; smallest |difference|/threshold {margin:.2e}; "
            f"threshold equals the sum of bars throughout; by spread {letters}"
        ),
    )


def _route_6_row(
    sweep: Sweep, lower: str, higher: str, key: str, name: str
) -> CheckReport:
    comparisons = compare(sweep, lower, higher)
    seen = sum(1 for c in comparisons if c.order.name != "NOT_RESOLVED")
    return CheckReport(
        name=name,
        check_id=f"ladder.{key}",
        warrant=Warrant.CORROBORATED,
        outcome=Outcome.NOT_TRIGGERED if seen else Outcome.NOT_RESOLVED,
        tier=Tier.COMPUTED,
        detail=(
            f"{lower} → {higher} resolves at {seen}/{len(comparisons)} spreads, "
            f"by spread {_letters(comparisons)}"
        ),
    )


def _voids_row(sweep: Sweep) -> CheckReport:
    voided = [cell for cell in sweep.cells[_ITERATED] if cell.voided]
    if voided:
        detail = "; ".join(
            f"σ={cell.spread:.4f} voided {cell.declared.voided_mass:.2e}"
            for cell in voided
        )
    else:
        worst = max(cell.declared.voided_mass for cell in sweep.cells[_ITERATED])
        detail = f"no spread voided above the roundoff floor; largest {worst:.1e}"
    return CheckReport(
        name="the iterated rung declines no reading on this family",
        check_id="ladder.iterated_voids",
        warrant=Warrant.CORROBORATED,
        outcome=Outcome.FIRED if voided else Outcome.NOT_TRIGGERED,
        tier=Tier.COMPUTED,
        detail=detail,
    )


def _plug_in_by_threshold_engine(spread: float) -> float:
    """The plug-in gap on the declared lattice, from the threshold exploration."""
    return measure_gap(spread, DECLARED)[0]


def _r6_row(sweep: Sweep, reference: Callable[[float], float]) -> CheckReport:
    cells = sweep.cells["plug-in"]
    worst = max(
        abs(cell.value - reference(cell.spread)) / abs(reference(cell.spread))
        for cell in cells
    )
    above = [cell.spread for cell in cells if cell.value > THRESHOLD]
    first = f"from σ={min(above):.4f}" if above else "at none"
    agrees = worst <= ENGINE_TOLERANCE
    return CheckReport(
        name="the R6 signal: the plug-in gap, read where T is registered",
        check_id="ladder.r6_signal",
        warrant=Warrant.CORROBORATED,
        outcome=Outcome.NOT_TRIGGERED if agrees else Outcome.FIRED,
        tier=Tier.COMPUTED,
        detail=(
            f"agrees with research.explorations.threshold to {worst:.1e} relative; "
            f"exceeds T = {THRESHOLD:.3e} nats at {len(above)}/{len(cells)} spreads, "
            f"{first}; no verdict on the gate, which is PR-8's"
        ),
    )


def _certificate_row(sweep: Sweep) -> CheckReport:
    from cpomdp.reference.ladder import LADDER

    declared = sweep.names == LADDER.names
    visited = sum(1 for rung in sweep.names if sweep.read_at_every_spread(rung))
    complete = declared and visited == LADDER.size
    proved = complete and _MEASURED_REF is not None
    certificate = ProductCompletenessCertificate(
        expected=LADDER.size,
        visited=visited,
        warrant=Warrant.PROVED if proved else Warrant.CORROBORATED,
        axes=(AxisDeclaration(name="rung", size=LADDER.size, version=LADDER.version),),
    )
    if proved:
        provenance: tuple[Provenance, ...] = (
            Provenance(
                registered_at=_REGISTERED_REF,
                measured_at=_MEASURED_REF or _REGISTERED_REF,
                registered=(
                    "the ordering's first reading: cells, lattices, the bar, the "
                    "minimum separation, the direction and the void rule, under D1 "
                    "of the battery"
                ),
            ),
        )
        why = "registered before it was read"
    elif complete:
        provenance = ()
        why = "measured_at is the commit this suite lands in, filled by the next"
    else:
        provenance = ()
        why = "the ladder was not read in full"
    return CheckReport(
        name="every declared rung was read at every spread",
        check_id="ladder.certificate",
        warrant=certificate.warrant,
        outcome=Outcome.NOT_TRIGGERED if complete else Outcome.FIRED,
        tier=Tier.EXACT,
        detail=(
            f"{visited} of {LADDER.size} rungs at {len(SPREADS)} spreads, ladder "
            f"{LADDER.version}; {why}"
        ),
        evidence=(certificate,),
        provenance=provenance,
    )


def run_checks(
    sweep: Sweep | None = None,
    reference: Callable[[float], float] = _plug_in_by_threshold_engine,
) -> list[CheckReport]:
    """Read the sweep against the registration.

    Args:
        sweep: the measured cells. Measured live when omitted.
        reference: the plug-in gap at a spread by another route, for the R6 row.

    Returns:
        Every check's report: four ordering rows, two route 6 rows, the void row,
        the R6 row and the certificate.
    """
    sweep = measure() if sweep is None else sweep
    reports = [_ordering_row(sweep, lower, higher, key) for lower, higher, key in PAIRS]
    reports.append(
        _route_6_row(
            sweep,
            *PAIRS[0][:2],
            "route6_derivative_terms",
            "route 6: the derivative-of-covariance terms are visible",
        )
    )
    reports.append(
        _route_6_row(
            sweep,
            *PAIRS[1][:2],
            "route6_iteration",
            "route 6: the iteration is visible",
        )
    )
    reports.append(_voids_row(sweep))
    reports.append(_r6_row(sweep, reference))
    reports.append(_certificate_row(sweep))
    return reports


def _print_table(sweep: Sweep) -> None:
    """The five gaps with their bars and the four orders, one line per spread."""
    names = sweep.names
    print("σ        " + "  ".join(f"{name[:12]:>24}" for name in names) + "   orders")
    pairs = [compare(sweep, lower, higher) for lower, higher, _ in PAIRS]
    for index, spread in enumerate(SPREADS):
        cells = [sweep.cells[name][index] for name in names]
        values = "  ".join(f"{c.value:.6e} ± {c.bar.own:.1e}" for c in cells)
        orders = " ".join(_letters([pair[index]]) for pair in pairs)
        print(f"{spread:.4f}   {values}   {orders}")
    print(f"\nT = {THRESHOLD:.6e} nats; A above, B below, N not resolved, V voided")


def main(argv: Sequence[str] | None = None) -> int:
    """Measure the ladder and print its table, then run the checks if asked.

    Args:
        argv: command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        Zero when no check fired, one otherwise.
    """
    parser = argparse.ArgumentParser(
        description="The ladder's first reading on d4-family-v1 at the binding cell."
    )
    parser.add_argument("--check", action="store_true", help="run the check suite")
    arguments = parser.parse_args(argv)

    print(f"kappa = {KAPPA}, mu* = {RIDGE_MEAN:.6f}, R0 = {BASE_NOISE}")
    sweep = measure(report=print)
    print()
    _print_table(sweep)
    if not arguments.check:
        return 0

    print()
    reports = run_checks(sweep)
    for report in reports:
        print(report)
    print(f"\n{check_summary(reports)}")
    return 1 if any(report.outcome is Outcome.FIRED for report in reports) else 0


if __name__ == "__main__":
    sys.exit(main())
