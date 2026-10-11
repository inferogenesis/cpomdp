"""Pages 5 to 7 of the discretisation-bound hand derivation: `f_y` in the strip.

Running the module is checking it: every claim it prints is asserted inside it. These
call the same functions so a failure names the claim rather than the whole run.
"""

import math

import pytest
import sympy

from research.discretisation_bound import cell, likelihood_strip
from research.explorations import threshold


def test_it_runs_and_its_assertions_hold(capsys):
    likelihood_strip.main()
    printed = capsys.readouterr().out
    assert "Q >= -v^2/2A for every u and y" in printed
    assert "both inequalities are equalities at u = y = 0" in printed


class TestH1:
    def test_the_noise_splits_into_a_and_b(self):
        assert likelihood_strip.noise_parts_residuals() == (0, 0)

    def test_the_noise_vanishes_only_at_the_two_poles(self):
        R0, k = sympy.symbols("R0 kappa", positive=True)
        poles = set(likelihood_strip.pole_locations())
        assert poles == {sympy.I * sympy.sqrt(R0 / k), -sympy.I * sympy.sqrt(R0 / k)}

    def test_the_real_part_is_at_least_its_floor(self):
        assert likelihood_strip.real_part_floor_residual() == 0


class TestH2:
    def test_factor_two(self):
        assert likelihood_strip.modulus_residual() == 0

    def test_factor_three_closed_form(self):
        assert likelihood_strip.exponent_residual() == 0

    def test_the_completed_square(self):
        assert likelihood_strip.completed_square_residual() == 0

    def test_the_exponent_floor_is_a_square(self):
        assert likelihood_strip.exponent_floor_residual() == 0

    def test_the_exponent_bound_grows_toward_the_edge(self):
        assert likelihood_strip.exponent_growth_residual() == 0


class TestTheEnvelope:
    def test_it_grows_with_the_strip(self):
        assert likelihood_strip.envelope_growth_residual() == 0

    def test_it_is_attained(self):
        # So C(a) is the supremum of |N(y; z, R(z))| over the strip, and the numeric
        # peak below should touch it rather than sit under it.
        assert likelihood_strip.envelope_attained_residual() == 0

    @pytest.mark.parametrize("a", [0.05, 1.0, float(cell.STRIP_CAP)])
    def test_the_numeric_peak_touches_it(self, a):
        peak = likelihood_strip.envelope_peak(a)
        assert abs(peak - 1.0) < 1e-12, peak


class TestTheNumbersAtTheCell:
    def test_the_envelope_at_the_cap(self):
        C = likelihood_strip.envelope(cell.STRIP_CAP)
        exact = sympy.exp(sympy.Rational(5, 3)) / sympy.sqrt(3 * sympy.pi / 2)
        assert sympy.simplify(C - exact) == 0
        assert f"{float(C):.5f}" == "2.43895"

    def test_m_below_the_crossover(self):
        M = likelihood_strip.line_integral_bound(cell.CROSSOVER_SPREAD)
        assert f"{float(M):.4e}" == "1.2645e+22"

    def test_m_at_the_largest_spread(self):
        M = likelihood_strip.line_integral_bound(cell.LARGEST_SPREAD)
        assert f"{float(M):.2f}" == "31.27"

    def test_the_log_bounds(self):
        below = float(likelihood_strip.log_route_b_bound(cell.CROSSOVER_SPREAD))
        largest = float(likelihood_strip.log_route_b_bound(cell.LARGEST_SPREAD))
        assert f"{below:.3f}" == "-4137.205"
        assert f"{largest:.3f}" == "-942.016"


class TestPageSevenAsWritten:
    # Page 7 was computed with a = 1.58 as a decimal and rounded 4188.79 to 4189. These
    # pin what the CLARIFICATION of 2026-10-11 says about each figure.

    def test_c_at_most_2_43_fails_by_rounding(self):
        C = float(likelihood_strip.envelope(sympy.Rational(158, 100)))
        assert f"{C:.4f}" == "2.4306"
        assert likelihood_strip.PAGE_SEVEN["envelope_at_cap"] < C

    def test_m_below_the_crossover_holds(self):
        M = float(likelihood_strip.line_integral_bound(cell.CROSSOVER_SPREAD))
        assert likelihood_strip.PAGE_SEVEN["bound_below_crossover"] >= M

    def test_m_at_most_31_fails_with_either_cap(self):
        M = float(likelihood_strip.line_integral_bound(cell.LARGEST_SPREAD))
        assert likelihood_strip.PAGE_SEVEN["bound_at_largest"] < M
        a = sympy.Rational(158, 100)
        decimal = likelihood_strip.envelope(a) * sympy.exp(
            a**2 / (2 * cell.LARGEST_SPREAD**2)
        )
        assert float(decimal) > likelihood_strip.PAGE_SEVEN["bound_at_largest"]

    def test_the_bound_below_the_crossover_is_not_under_2_6e22_e_minus_4189(self):
        log_bound = float(likelihood_strip.log_route_b_bound(cell.CROSSOVER_SPREAD))
        assert log_bound > likelihood_strip.PAGE_SEVEN["log_error_below_crossover"]

    def test_the_bound_below_the_crossover_is_3_1e22_e_minus_4189(self):
        log_bound = float(likelihood_strip.log_route_b_bound(cell.CROSSOVER_SPREAD))
        assert f"{math.exp(log_bound + 4189.0):.1e}" == "3.1e+22"

    def test_the_bound_at_the_largest_spread_is_under_62_e_minus_945(self):
        log_bound = float(likelihood_strip.log_route_b_bound(cell.LARGEST_SPREAD))
        assert log_bound < likelihood_strip.PAGE_SEVEN["log_error_at_largest"]


class TestNumerically:
    @pytest.mark.parametrize("spread", threshold.SPREADS)
    def test_the_line_integral_stays_under_its_bound(self, spread):
        a = float(cell.strip_half_width(sympy.nsimplify(spread)))
        for fraction in (0.0, 0.99):
            ratio = likelihood_strip.line_integral_ratio(
                spread, fraction * a, threshold.RIDGE_MEAN
            )
            assert ratio <= 1.0, (spread, fraction, ratio)

    @pytest.mark.parametrize("h", likelihood_strip.VISIBLE_STEPS)
    def test_route_b_sits_above_a_measured_error(self, h):
        measured, bound = likelihood_strip.visible_trapezoid_error(h)
        assert measured > 1e-11, "the error has to be visible for this to test much"
        assert measured <= bound
        assert math.isfinite(bound)
