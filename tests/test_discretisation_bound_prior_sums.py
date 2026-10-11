"""Pages 8 to 12 of the discretisation-bound hand derivation: the four prior sums.

Every number in the RESULT and CLARIFICATION of 2026-10-11 in
`research/discretisation_bound_hand_derivation.md` is asserted here, so the write-up
cannot drift from the run.

Running the module is checking it: every claim it prints is asserted inside it. These
call the same functions so a failure names the claim rather than the whole run.
"""

import math

import pytest
import sympy

from research.discretisation_bound import (
    cell,
    likelihood_strip,
    prelims,
    prior_sums,
    route_b,
)
from research.explorations import threshold


def _ln(expr):
    return float(expr.evalf(30))


def test_it_runs_and_its_assertions_hold(capsys):
    prior_sums.main()
    printed = capsys.readouterr().out
    assert "'below e^-4087' fails at the crossover; below e^-4086 holds" in printed
    assert "so sigma = 0.7 is the worst" in printed


@pytest.mark.parametrize(
    "module", [cell, route_b, prelims, likelihood_strip, prior_sums]
)
def test_no_module_reports_a_warrant(module):
    # The package docstring says nothing here is the bound. A `run_checks` or a
    # `_SOURCE` is the shape that would let one be collected as a check.
    assert not hasattr(module, "run_checks")
    assert not any(name.endswith("_SOURCE") for name in dir(module))


class TestTheFourM:
    def test_a_the_prior_integrates_to_one(self):
        assert prior_sums.mass_integral() == 1

    def test_b_the_mean_absolute_deviation_is_integrated(self):
        s = sympy.Symbol("sigma", positive=True)
        deviation = prior_sums.mean_absolute_deviation()
        assert sympy.simplify(deviation - s * sympy.sqrt(2 / sympy.pi)) == 0

    def test_c_the_squared_modulus(self):
        assert prior_sums.squared_modulus_residual() == 0

    def test_c_the_second_moment_is_variance_plus_offset(self):
        assert prior_sums.second_moment_residual() == 0

    def test_d_the_exponent_gap_is_a_sum_of_nonnegative_terms(self):
        assert prior_sums.plug_in_mass_exponent_residual() == 0

    def test_c_crude_offset_costs_a_factor_of_2_4_to_24_6(self):
        # Not 145. That is the coefficient on sigma^2, not the factor on M.
        assert prior_sums.offset_inflation(cell.CROSSOVER_SPREAD) == sympy.Rational(
            245, 101
        )
        inflation = prior_sums.offset_inflation(cell.LARGEST_SPREAD)
        assert f"{float(inflation):.1f}" == "24.6"

    def test_d_hypothesis_holds_in_the_engine(self):
        # Corroboration of shat^2 >= sigma^2/2, which the tail argument has to prove.
        ratios = prior_sums.engine_variance_ratios()
        assert len(ratios) == 15
        assert max(abs(r - 1.0) for r in ratios) < 1e-10, ratios


class TestTheTable:
    def test_below_the_crossover(self):
        M = prior_sums.line_integral_bounds(cell.CROSSOVER_SPREAD)
        e50 = sympy.exp(50)
        assert M["mass"] == e50
        assert f"{float(M['first_moment'] / e50):.3f}" == "4.870"
        assert sympy.simplify(M["second_moment"] - sympy.Rational(49, 8) * e50) == 0
        assert (
            sympy.simplify(M["plug_in_mass"] - sympy.exp(sympy.Rational(405, 4))) == 0
        )

    def test_at_the_largest_spread(self):
        M = prior_sums.line_integral_bounds(cell.LARGEST_SPREAD)
        growth = sympy.exp(sympy.Rational(125, 49))
        assert sympy.simplify(M["mass"] - growth) == 0
        assert f"{float(M['mass']):.2f}" == "12.82"
        assert f"{float(M['first_moment']):.2f}" == "67.97"
        assert (
            sympy.simplify(M["second_moment"] - sympy.Rational(1471, 20) * growth) == 0
        )
        assert f"{float(M['second_moment']):.1f}" == "942.9"
        assert f"{float(M['plug_in_mass']):.1f}" == "573.7"

    def test_the_rates(self):
        below = (
            2 * sympy.pi * cell.STRIP_CAP / (cell.STATE_STEP * cell.CROSSOVER_SPREAD)
        )
        largest = (
            2 * sympy.pi * cell.STRIP_CAP / (cell.STATE_STEP * cell.LARGEST_SPREAD)
        )
        assert sympy.simplify(below - 4000 * sympy.pi / 3) == 0
        assert sympy.simplify(largest - 2000 * sympy.sqrt(10) * sympy.pi / 21) == 0
        assert f"{float(below):.2f}" == "4188.79"
        assert f"{float(largest):.2f}" == "946.15"

    @pytest.mark.parametrize(
        ("name", "below", "largest"),
        [
            ("mass", "-4138.097", "-942.907"),
            ("first_moment", "-4136.514", "-941.239"),
            ("second_moment", "-4136.285", "-938.609"),
            ("plug_in_mass", "-4086.847", "-939.106"),
        ],
    )
    def test_the_log_errors(self, name, below, largest):
        assert (
            f"{_ln(prior_sums.log_error_bounds(cell.CROSSOVER_SPREAD)[name]):.3f}"
            == below
        )
        assert (
            f"{_ln(prior_sums.log_error_bounds(cell.LARGEST_SPREAD)[name]):.3f}"
            == largest
        )


class TestTheClaims:
    def test_below_e_minus_4087_fails_at_the_crossover(self):
        # The drafted sentence. It holds only if 2 pi a/h_x is rounded up to 4189.
        worst = max(
            map(_ln, prior_sums.log_error_bounds(cell.CROSSOVER_SPREAD).values())
        )
        assert worst > -4087.0
        assert worst < -4086.0
        assert round(worst / math.log(10)) == -1775

    def test_it_holds_over_the_ten_declared_spreads_below_the_crossover(self):
        assert len(prior_sums.BELOW_CROSSOVER) == 10
        assert f"{float(max(prior_sums.BELOW_CROSSOVER)):.4f}" == "0.1198"
        worst = max(
            _ln(value)
            for spread in prior_sums.BELOW_CROSSOVER
            for value in prior_sums.log_error_bounds(spread).values()
        )
        assert f"{worst:.3f}" == "-4087.379"

    def test_the_largest_spread_is_worst_at_c_and_below_e_minus_938(self):
        bounds = prior_sums.log_error_bounds(cell.LARGEST_SPREAD)
        worst = max(bounds, key=lambda name: _ln(bounds[name]))
        assert worst == "second_moment"
        assert -939.0 < _ln(bounds[worst]) < -938.0
        assert f"{_ln(bounds[worst]) / math.log(10):.1f}" == "-407.6"

    def test_every_log_error_grows_with_the_spread(self):
        for name, (below, above) in prior_sums.regime_growth().items():
            assert below.is_nonnegative, (name, below)
            assert above.is_nonnegative, (name, above)
        assert (400 * sympy.pi / 3 - 20).is_positive

    @pytest.mark.parametrize("name", prior_sums.SUMS)
    def test_the_declared_spreads_agree(self, name):
        row = [_ln(prior_sums.log_error_bounds(s)[name]) for s in cell.SPREADS]
        assert row == sorted(row)


class TestTheDraftedColumn:
    # The CLARIFICATION says where the drafted sigma = 0.7 column came from.

    def test_b_c_and_the_rate_reproduce_at_a_decimal_cap(self):
        column = prior_sums.column_at_largest(sympy.Rational(158, 100))
        drafted = prior_sums.DRAFTED_AT_LARGEST
        assert f"{column['first_moment']:.1f}" == f"{drafted['first_moment']:.1f}"
        assert round(column["second_moment"]) == drafted["second_moment"]
        assert math.floor(column["rate"]) == drafted["rate"]

    def test_d_matches_neither_cap(self):
        decimal = prior_sums.column_at_largest(sympy.Rational(158, 100))
        exact = prior_sums.column_at_largest(cell.STRIP_CAP)
        assert round(decimal["plug_in_mass"]) == 568
        assert round(exact["plug_in_mass"]) == 574
        assert prior_sums.DRAFTED_AT_LARGEST["plug_in_mass"] == 569


class TestAgainstExactErrors:
    def test_route_b_at_its_best_strip_gives_the_poisson_exponent(self):
        best, exponent = prior_sums.route_b_optimum()
        assert best == 0
        assert exponent == 0

    @pytest.mark.parametrize("spread", cell.SPREADS)
    def test_the_bounds_sit_above_poisson_on_the_declared_lattice(self, spread):
        margins = prior_sums.poisson_margins(spread)
        assert min(margins.values()) > 0, margins

    def test_the_smallest_poisson_margin(self):
        smallest = min(
            min(prior_sums.poisson_margins(s).values()) for s in cell.SPREADS
        )
        assert f"{smallest:.0f}" == "34293"

    def test_the_poisson_rate_for_the_prior_mass(self):
        # beta = 2 pi^2 sigma^2 / h_x^2 is the same at every spread.
        s = cell.LARGEST_SPREAD
        log_error = route_b.log_gaussian_worst_error(s**2, cell.STATE_STEP * s)
        beta = 80000 * sympy.pi**2 / 9
        assert sympy.simplify(2 * sympy.pi**2 / cell.STATE_STEP**2 - beta) == 0
        assert f"{float(beta):.0f}" == "87730"
        assert abs(_ln(log_error) - (math.log(2) - float(beta))) < 1e-9

    @pytest.mark.parametrize("h", prior_sums.VISIBLE_STEPS)
    def test_the_lattice_sum_matches_poisson(self, h):
        assert prior_sums.lattice_sum_departure(h) < 1e-14

    @pytest.mark.parametrize("h", prior_sums.VISIBLE_STEPS)
    def test_the_bounds_sit_above_measured_errors(self, h):
        for name, (measured, bound) in prior_sums.visible_trapezoid_errors(h).items():
            assert measured > 1e-11, (name, measured)
            assert measured <= bound, (name, measured, bound)


@pytest.mark.parametrize("spread", threshold.SPREADS)
def test_h2_holds_numerically(spread):
    ratios = prior_sums.line_integral_ratios(spread)
    assert max(ratios.values()) <= 1.0, ratios
