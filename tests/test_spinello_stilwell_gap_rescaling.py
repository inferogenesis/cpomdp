"""Route 1's empirical half: the reported gap under a rescaled observation.

Running the module is checking it, since every claim it prints is asserted inside it.
The wrapper that turns the scalar scheme into a rule is checked on its own here, on
the two ways it declines a reading.
"""

import pytest

from cpomdp.reference.gap import Void
from cpomdp.reference.quadrature import GridDensity, QuadratureGrid, gaussian_on
from research.spinello_stilwell import gap_rescaling
from research.spinello_stilwell.invariance import CASE

STATES = QuadratureGrid(lower=[-2.0], upper=[4.0], counts=[601])


def prior():
    return gaussian_on(STATES, [CASE["prior_mean"]], [[CASE["prior_variance"]]])


@pytest.mark.slow
def test_it_runs_and_its_assertions_hold(capsys):
    gap_rescaling.main()
    printed = capsys.readouterr().out
    assert "the modification is unit-free at both budgets" in printed
    assert "moves with the units" in printed


def test_it_reports_no_warrant():
    # The package invariant, stated in its `__init__`. A `run_checks` or a `_SOURCE` is
    # the shape that would let a route be collected as though it had decided something.
    assert not hasattr(gap_rescaling, "run_checks")
    assert not any(name.endswith("_SOURCE") for name in dir(gap_rescaling))


def test_the_budget_it_runs_at_is_the_rung_s():
    from cpomdp.reference.ladder import CONVERGENCE_TOLERANCE, ITERATION_BUDGET

    assert gap_rescaling.declared_budget() == (ITERATION_BUDGET, CONVERGENCE_TOLERANCE)


def test_a_run_that_spends_the_budget_answers_void():
    # Tolerance zero can never be met, so a budget of two is spent by construction.
    rule = gap_rescaling.scheme_rule(gap_rescaling.MODIFIED, 1.0, 2, 0.0)
    answer = rule(prior(), [1.7])
    assert isinstance(answer, Void)
    assert answer.iterations == 2


def test_a_run_settling_on_its_last_allowed_step_is_accepted():
    # The rung accepts a run whose last step fell under the tolerance, whatever the
    # count. The stand-in has to read it the same way, so the budget is set to the
    # exact count a settled run takes and then one less.
    _, tolerance = gap_rescaling.declared_budget()
    noise_at = gap_rescaling.scheme.quadratic_noise(
        CASE["base_noise"], CASE["curvature"]
    )
    _, _, settled = gap_rescaling.scheme.iterate_with(
        1.7,
        CASE["prior_mean"],
        CASE["prior_variance"],
        noise_at,
        1.0,
        tolerance,
        200,
        log_block=False,
    )
    assert 1 < settled < 200
    exact = gap_rescaling.scheme_rule(gap_rescaling.MODIFIED, 1.0, settled, tolerance)
    assert isinstance(exact(prior(), [1.7]), GridDensity)
    short = gap_rescaling.scheme_rule(
        gap_rescaling.MODIFIED, 1.0, settled - 1, tolerance
    )
    answer = short(prior(), [1.7])
    assert isinstance(answer, Void)
    assert answer.iterations == settled - 1


def test_a_single_step_never_answers_void():
    rule = gap_rescaling.scheme_rule(
        gap_rescaling.PRINTED, 1.0, *gap_rescaling.SINGLE_STEP
    )
    answer = rule(prior(), [1.7])
    assert isinstance(answer, GridDensity)
    assert answer.grid.same_lattice_as(STATES)


def test_a_settled_run_answers_a_belief_on_the_prior_s_lattice():
    budget, tolerance = gap_rescaling.declared_budget()
    rule = gap_rescaling.scheme_rule(gap_rescaling.MODIFIED, 3.0, budget, tolerance)
    answer = rule(prior(), [1.7])
    assert isinstance(answer, GridDensity)
    assert answer.grid.same_lattice_as(STATES)


def test_the_spread_is_infinite_where_a_reading_is_not_finite():
    assert gap_rescaling.relative_spread([1.0, 1.0, float("nan")]) == float("inf")
    assert gap_rescaling.relative_spread([1.0, 1.0, 1.0]) == 0.0
