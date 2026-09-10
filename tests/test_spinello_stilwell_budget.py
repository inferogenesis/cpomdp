"""Route 5: what truncating a run costs, and what a budget has to be sized against.

`research/spinello_stilwell_rung.md` states Q7 as an argument. This is the measurement.
The module is not the rung, declares no budget, and reports no warrant.

Running the module is checking it, since every claim it prints is asserted inside it.
These call the same functions so a failure names the claim rather than the whole run.
"""

from research.checks.gap_kernel import FAMILIES
from research.spinello_stilwell import budget, invariance


def test_it_runs_and_its_assertions_hold(capsys):
    budget.main()
    assert capsys.readouterr().out


def test_it_reports_no_warrant():
    # The package invariant, stated in its `__init__`. A `run_checks` or a `_SOURCE` is
    # the shape that would let a route be collected as though it had decided something.
    assert not hasattr(budget, "run_checks")
    assert not any(name.endswith("_SOURCE") for name in dir(budget))


class TestTheBudget:
    """Route 5. What truncation costs, and what a declared budget has to cover."""

    def test_both_halves_of_the_posterior_move_when_the_budget_is_cut(self):
        # Q7. `Ū` is evaluated at the current iterate, so the covariance is truncated
        # along with the mean and a gap computed from the pair is wrong twice.
        for case in budget.TRUNCATION_GRID:
            mean_gap, variance_gap = budget.departure_at_budget(case, 1)
            assert mean_gap > 0.0, case
            assert variance_gap > 0.0, case

    def test_the_covariance_error_is_not_bounded_by_the_mean_s(self):
        # The reason `VOID` beats reporting the number with a caveat about the mean.
        departures = [
            budget.departure_at_budget(case, 1) for case in budget.TRUNCATION_GRID
        ]
        assert max(variance for _, variance in departures) > 0.4
        assert any(variance > mean for mean, variance in departures)

    def test_both_halves_settle_as_the_budget_grows(self):
        wider = budget.departure_at_budget(budget.SLOW_CASE, 2)
        tighter = budget.departure_at_budget(budget.SLOW_CASE, 5)
        assert tighter[0] < wider[0]
        assert tighter[1] < wider[1]

    def test_the_declared_slopes_agree_with_a_difference(self):
        # `SLOPES` differentiates a registered declaration by hand, so it can drift
        # from what it differentiates without anything else noticing.
        worst = budget.slopes_match_a_difference((-1.0, 0.0, 0.5, 1.0, 2.0))
        assert all(gap < 1e-6 for gap in worst.values()), worst

    def test_no_declared_cell_needs_more_than_ten_iterations(self):
        # The number a budget declaration reads. The fixed family needs two and the
        # quadratic at the widest spread needs ten, so the spread is what moves it.
        counts = budget.convergence_counts()
        assert max(counts.values()) == 10, counts
        assert counts["constant", 0.06] == 2, counts

    def test_the_box_edge_costs_far_more_than_the_bulk_at_every_curvature(self):
        # The convergence survey reads two *prior* spreads out. The gap runs the rung
        # nine *predictive* spreads out, to the edge of its observation box, and the
        # count follows.
        worst = budget.counts_over_the_operating_range(1e-12)
        assert max(bulk for bulk, _ in worst.values()) == 23, worst
        assert max(edge for _, edge in worst.values()) == 65, worst
        assert all(edge > bulk for bulk, edge in worst.values()), worst

    def test_a_declared_family_exhausts_the_declared_budget_at_the_box_edge(self):
        # The curvature sweep never leaves `1 + kappa x^2`, so it cannot see this. On
        # the bounded oscillating family the iterate crosses several periods of `R`
        # before it settles, and 64 iterations do not get it there.
        worst = budget.counts_over_the_declared_families(1e-12)
        assert worst["sin"] == (10, 124), worst
        assert max(bulk for bulk, _ in worst.values()) == 16, worst
        exhausted = [
            key for key, (_, edge) in worst.items() if edge > budget.DECLARED_BUDGET
        ]
        assert exhausted == ["sin"], worst

    def test_every_declared_family_still_converges_inside_the_probe_budget(self):
        # A count equal to the probe budget would be a run that never settled, and the
        # number read off it would be the budget rather than a measurement.
        worst = budget.counts_over_the_declared_families(1e-12)
        assert all(
            count < invariance.PROBE_BUDGET for cell in worst.values() for count in cell
        ), worst

    def test_the_bounded_family_stops_converging_past_the_box_edge(self):
        # A stable two-cycle, not a slow run. ADR-060 rejects widening the budget for
        # a voided node, and this is why: no budget converges it.
        settled, alternate, back = budget.three_iterates_at(
            FAMILIES["sin"], 0.30, budget.OUTSIDE_THE_BOX, 400
        )
        assert abs(settled - back) < 1e-12
        assert abs(alternate - back) > 1.0

    def test_the_unbounded_family_has_settled_at_the_same_offset(self):
        # The falsifier for the test above: a two-cycle everywhere would mean the
        # iteration was wrong rather than that this family is hard far from its mean.
        states = budget.three_iterates_at(
            FAMILIES["quadratic"], 0.30, budget.OUTSIDE_THE_BOX, 400
        )
        assert max(states) - min(states) < 1e-12, states

    def test_a_tighter_tolerance_costs_iterations_at_every_curvature(self):
        # What 1e-14 buys over 1e-12, in the currency a budget is spent in.
        tight = budget.counts_over_the_operating_range(1e-14)
        loose = budget.counts_over_the_operating_range(1e-12)
        assert all(
            tight[curvature][where] >= loose[curvature][where]
            for curvature in budget.OPERATING_CURVATURES
            for where in (0, 1)
        )
        assert max(bulk for bulk, _ in tight.values()) == 27, tight

    def test_the_two_curvatures_agree_once_both_converge(self):
        # ADR-057's "surgical" in numbers: a difference before convergence and none
        # after, so the deletion moves no fixed point.
        early = budget.modification_cost(budget.SLOW_CASE, 1)
        settled = budget.modification_cost(budget.SLOW_CASE, invariance.PROBE_BUDGET)
        assert early[0] > 0.0
        assert settled[0] < 1e-12
        assert settled[1] < 1e-12
