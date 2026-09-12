"""What the ladder suite reads off a sweep, and what it may claim.

The reading rules are checked on hand-built sweeps, so they run in milliseconds and
can be driven to every outcome. The live run is checked once for its shape, since
what its rows say is the RESULT the registration is waiting for and is not asserted
here.
"""

import math
from functools import cache

import pytest

from cpomdp.reference.ladder import LADDER
from cpomdp.resolution import Order
from research.checks import ladder
from research.explorations.threshold import ROUNDOFF_FLOOR, SPREADS
from warrantlib import Outcome, ProductCompletenessCertificate, Tier, Warrant

NAMES = LADDER.names
TWO_SPREADS = SPREADS[:2]


def reading(gap, voided_mass=0.0):
    return ladder.Reading(
        gap=gap, predictive_mass=1.0, worst_edge_ratio=0.0, voided_mass=voided_mass
    )


def cell(rung, spread, declared, fine=None, voided_mass=0.0):
    fine = declared if fine is None else fine
    return ladder.Cell(
        rung, spread, reading(declared, voided_mass), reading(fine, voided_mass)
    )


def sweep_of(values, bars=None):
    """A sweep with one value per rung per spread, and a refinement gap per rung.

    `values[rung]` is a tuple over the two spreads. `bars[rung]` widens the fine
    reading away from the declared one by that much, so the bar is under control.
    """
    bars = bars or {}
    return ladder.Sweep(
        {
            rung: tuple(
                cell(rung, spread, value, value + bars.get(rung, 0.0))
                for spread, value in zip(TWO_SPREADS, values[rung], strict=True)
            )
            for rung in NAMES
        }
    )


def decreasing():
    """Five gaps a decade apart at each spread, with negligible bars."""
    return sweep_of({rung: (10.0**-k, 2.0 * 10.0**-k) for k, rung in enumerate(NAMES)})


def by_hand(sweep):
    """The plug-in reading itself, standing in for the threshold engine."""
    return lambda spread: next(
        c.value for c in sweep.cells["plug-in"] if c.spread == spread
    )


# --- the bar ----------------------------------------------------------------------


def test_the_bar_is_the_refinement_difference():
    one = cell("plug-in", 0.1, 1.0e-3, 1.0e-3 + 4.0e-9)
    assert one.bar.common_mode == 0.0
    assert one.bar.own == pytest.approx(4.0e-9)


def test_the_bar_is_floored_at_roundoff():
    one = cell("plug-in", 0.1, 1.0e-3, 1.0e-3)
    assert one.bar.own == pytest.approx(ROUNDOFF_FLOOR * 1.0e-3)


def test_a_cell_is_voided_only_above_the_floor():
    assert not cell("modified-iterated", 0.1, 1.0, voided_mass=1e-15).voided
    assert cell("modified-iterated", 0.1, 1.0, voided_mass=1e-12).voided


# --- the reading of a pair ----------------------------------------------------------


def test_a_decreasing_ladder_reads_above_at_every_spread():
    for lower, higher, _ in ladder.PAIRS:
        comparisons = ladder.compare(decreasing(), lower, higher)
        assert all(c.order is Order.ABOVE for c in comparisons)
        assert ladder.outcome_of(comparisons) is Outcome.NOT_TRIGGERED


def test_a_reversed_pair_fires():
    values = {rung: (10.0**-k, 2.0 * 10.0**-k) for k, rung in enumerate(NAMES)}
    values["modified-iterated"], values["belief-smoothed"] = (
        values["belief-smoothed"],
        values["modified-iterated"],
    )
    comparisons = ladder.compare(
        sweep_of(values), "modified-iterated", "belief-smoothed"
    )
    assert ladder.outcome_of(comparisons) is Outcome.FIRED


def test_overlapping_bars_read_not_resolved():
    # A difference of 1e-3 against bars of 1e-2 on each side.
    values = {rung: (1.0 - 1e-3 * k, 1.0 - 1e-3 * k) for k, rung in enumerate(NAMES)}
    wide = sweep_of(values, bars=dict.fromkeys(NAMES, 1e-2))
    comparisons = ladder.compare(wide, "plug-in", "modified-single-step")
    assert all(c.order is Order.NOT_RESOLVED for c in comparisons)
    assert ladder.outcome_of(comparisons) is Outcome.NOT_RESOLVED
    assert all(c.threshold == pytest.approx(2e-2) for c in comparisons)


def test_the_threshold_is_the_sum_of_bars_since_nothing_is_cancelled():
    comparisons = ladder.compare(decreasing(), "plug-in", "modified-single-step")
    one = decreasing().cells["plug-in"][0].bar.own
    other = decreasing().cells["modified-single-step"][0].bar.own
    assert comparisons[0].threshold == pytest.approx(one + other)


def test_every_comparison_carries_the_sum_of_bars_beside_the_threshold():
    comparisons = ladder.compare(decreasing(), "plug-in", "modified-single-step")
    assert all(c.sum_of_bars == c.threshold for c in comparisons)
    assert ladder.coincidence(comparisons) == (2, 2)


def test_the_coincidence_count_sees_a_threshold_below_the_sum_of_bars():
    # What a common-mode part would do once PR-8 certifies the error's shape: the
    # threshold drops under the sum of bars, and the row has to say at how many
    # spreads it did rather than assert in prose that they coincide.
    apart = ladder.PairAtSpread(0.1, Order.ABOVE, 1e-3, 1e-6, 2e-6, False)
    same = ladder.PairAtSpread(0.2, Order.ABOVE, 1e-3, 2e-6, 2e-6, False)
    voided = ladder.PairAtSpread(
        0.3, Order.NOT_RESOLVED, math.nan, math.nan, math.nan, True
    )
    assert ladder.coincidence([apart, same, voided]) == (1, 2)


def test_a_mixed_pair_survives_where_it_resolves():
    values = {rung: (10.0**-k, 1.0 - 1e-3 * k) for k, rung in enumerate(NAMES)}
    wide_at_second = ladder.Sweep(
        {
            rung: (
                cell(rung, TWO_SPREADS[0], values[rung][0]),
                cell(rung, TWO_SPREADS[1], values[rung][1], values[rung][1] + 1e-2),
            )
            for rung in NAMES
        }
    )
    comparisons = ladder.compare(wide_at_second, "plug-in", "modified-single-step")
    assert [c.order for c in comparisons] == [Order.ABOVE, Order.NOT_RESOLVED]
    assert ladder.outcome_of(comparisons) is Outcome.NOT_TRIGGERED


def test_a_voided_spread_sets_the_iterated_pairs_aside():
    sweep = decreasing()
    cells = dict(sweep.cells)
    first, second = cells["modified-iterated"]
    cells["modified-iterated"] = (
        first,
        cell("modified-iterated", second.spread, second.value, voided_mass=1e-9),
    )
    voided = ladder.Sweep(cells)
    for lower, higher in (
        ("modified-single-step", "modified-iterated"),
        ("modified-iterated", "belief-smoothed"),
    ):
        comparisons = ladder.compare(voided, lower, higher)
        assert [c.order for c in comparisons] == [Order.ABOVE, Order.NOT_RESOLVED]
        assert comparisons[1].voided
        assert math.isnan(comparisons[1].difference)
    untouched = ladder.compare(voided, "plug-in", "modified-single-step")
    assert not any(c.voided for c in untouched)


# --- the rows ----------------------------------------------------------------------


@pytest.fixture
def reports():
    sweep = decreasing()
    return {r.check_id: r for r in ladder.run_checks(sweep, reference=by_hand(sweep))}


def test_the_suite_reports_ten_rows(reports):
    assert len(reports) == 10


def test_every_ordering_row_is_computed_and_corroborated(reports):
    for _, _, key in ladder.PAIRS:
        row = reports[f"ladder.{key}"]
        assert row.tier is Tier.COMPUTED
        assert row.warrant is Warrant.CORROBORATED
        assert row.outcome is Outcome.NOT_TRIGGERED
        assert "threshold equals the sum of bars at 2/2 resolved spreads" in row.detail


def test_route_6_reads_visible_where_a_pair_resolves(reports):
    assert reports["ladder.route6_derivative_terms"].outcome is Outcome.NOT_TRIGGERED
    assert reports["ladder.route6_iteration"].outcome is Outcome.NOT_TRIGGERED


def test_route_6_reads_not_resolved_where_a_pair_never_resolves():
    values = {rung: (1.0 - 1e-3 * k, 1.0 - 1e-3 * k) for k, rung in enumerate(NAMES)}
    wide = sweep_of(values, bars=dict.fromkeys(NAMES, 1e-2))
    rows = {r.check_id: r for r in ladder.run_checks(wide, reference=by_hand(wide))}
    assert rows["ladder.route6_derivative_terms"].outcome is Outcome.NOT_RESOLVED
    assert rows["ladder.route6_iteration"].outcome is Outcome.NOT_RESOLVED


def test_the_void_row_fires_on_a_voided_spread():
    sweep = decreasing()
    cells = dict(sweep.cells)
    first, second = cells["modified-iterated"]
    cells["modified-iterated"] = (
        first,
        cell("modified-iterated", second.spread, second.value, voided_mass=1e-9),
    )
    voided = ladder.Sweep(cells)
    rows = {r.check_id: r for r in ladder.run_checks(voided, reference=by_hand(voided))}
    assert rows["ladder.iterated_voids"].outcome is Outcome.FIRED
    assert "voided 1.00e-09" in rows["ladder.iterated_voids"].detail


def test_the_r6_row_reads_the_reference_once_per_spread():
    sweep = decreasing()
    calls = []

    def counting(spread):
        calls.append(spread)
        return by_hand(sweep)(spread)

    ladder.run_checks(sweep, reference=counting)
    assert calls == list(TWO_SPREADS)


def test_the_lattice_row_holds_on_a_box_that_caught_everything(reports):
    row = reports["ladder.lattice"]
    assert row.outcome is Outcome.NOT_TRIGGERED
    assert row.tier is Tier.COMPUTED
    assert "the same for every rung at a spread" in row.detail


def test_the_lattice_row_fires_on_a_clipped_box():
    sweep = decreasing()
    cells = dict(sweep.cells)
    first, second = cells["belief-smoothed"]
    clipped = ladder.Reading(
        gap=second.value, predictive_mass=0.9, worst_edge_ratio=0.0, voided_mass=0.0
    )
    cells["belief-smoothed"] = (
        first,
        ladder.Cell("belief-smoothed", second.spread, clipped, second.fine),
    )
    rows = {
        r.check_id: r
        for r in ladder.run_checks(ladder.Sweep(cells), reference=by_hand(sweep))
    }
    row = rows["ladder.lattice"]
    assert row.outcome is Outcome.FIRED
    assert "caught 0.9000000000 of p* at worst" in row.detail
    assert f"σ={second.spread:.4f}" in row.detail


def test_the_r6_row_fires_when_the_engines_disagree():
    sweep = decreasing()
    rows = {
        r.check_id: r for r in ladder.run_checks(sweep, reference=lambda spread: 1.5)
    }
    assert rows["ladder.r6_signal"].outcome is Outcome.FIRED


def test_the_r6_row_renders_no_verdict_on_the_gate(reports):
    row = reports["ladder.r6_signal"]
    assert row.outcome is Outcome.NOT_TRIGGERED
    assert "no verdict on the gate" in row.detail


def test_the_certificate_quantifies_over_the_declared_ladder(reports):
    row = reports["ladder.certificate"]
    (certificate,) = row.evidence
    assert isinstance(certificate, ProductCompletenessCertificate)
    assert certificate.expected == certificate.visited == LADDER.size
    (axis,) = certificate.axes
    assert (axis.name, axis.size, axis.version) == ("rung", LADDER.size, LADDER.version)
    assert row.outcome is Outcome.NOT_TRIGGERED
    assert row.tier is Tier.EXACT


def test_the_certificate_is_proved_with_its_provenance(reports):
    row = reports["ladder.certificate"]
    assert row.warrant is Warrant.PROVED
    assert row.provenance == (ladder.PROVENANCE,)
    assert not ladder.PROVENANCE.same_ref


def test_an_unread_rung_leaves_the_certificate_incomplete():
    sweep = decreasing()
    cells = dict(sweep.cells)
    first, _ = cells["exact"]
    cells["exact"] = (first, cell("exact", TWO_SPREADS[1], math.nan, voided_mass=1.0))
    rows = {
        r.check_id: r
        for r in ladder.run_checks(ladder.Sweep(cells), reference=by_hand(sweep))
    }
    row = rows["ladder.certificate"]
    assert row.outcome is Outcome.FIRED
    assert row.warrant is Warrant.CORROBORATED
    (certificate,) = row.evidence
    assert isinstance(certificate, ProductCompletenessCertificate)
    assert certificate.visited == LADDER.size - 1


# --- the live run: its shape, and not what it says --------------------------------


@cache
def _live():
    return tuple(ladder.run_checks())


#: The suite is not in the manifest (its module docstring says why), so the ids are
#: pinned here: one dropped or renamed fails by name on the merge and release path.
RECORDED_IDS = (
    "ladder.belief_smoothed_to_exact",
    "ladder.certificate",
    "ladder.iterated_to_belief_smoothed",
    "ladder.iterated_voids",
    "ladder.lattice",
    "ladder.plug_in_to_single_step",
    "ladder.r6_signal",
    "ladder.route6_derivative_terms",
    "ladder.route6_iteration",
    "ladder.single_step_to_iterated",
)

#: The outcomes the battery's RESULT of 2026-09-12 records. Three orderings fired,
#: and a row that stops firing is a moved result: the battery is amended before this
#: tuple is.
RECORDED_FIRED = (
    "ladder.iterated_to_belief_smoothed",
    "ladder.plug_in_to_single_step",
    "ladder.single_step_to_iterated",
)


@pytest.mark.slow
def test_the_live_run_reports_the_recorded_ids():
    assert tuple(sorted(r.check_id for r in _live())) == RECORDED_IDS


@pytest.mark.slow
def test_the_live_run_reads_as_the_result_records():
    rows = {r.check_id: r for r in _live()}
    fired = tuple(sorted(k for k, r in rows.items() if r.outcome is Outcome.FIRED))
    assert fired == RECORDED_FIRED
    for check_id in set(RECORDED_IDS) - set(RECORDED_FIRED):
        assert rows[check_id].outcome is Outcome.NOT_TRIGGERED, check_id
    assert rows["ladder.certificate"].warrant is Warrant.PROVED
    assert rows["ladder.certificate"].evidence[0].visited == LADDER.size
