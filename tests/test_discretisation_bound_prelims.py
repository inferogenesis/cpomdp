"""Pages 1 to 4 of the discretisation-bound hand derivation.

`research/discretisation_bound_hand_derivation.md` transcribes them. None of this is the
bound, and none of it reports a warrant.

Running the module is checking it: every claim it prints is asserted inside it. These
call the same functions so a failure names the claim rather than the whole run.
"""

import pytest
import sympy

from research.discretisation_bound import cell, prelims
from research.explorations import threshold


def test_it_runs_and_its_assertions_hold(capsys):
    prelims.main()
    printed = capsys.readouterr().out
    assert "endpoints exact at all fifteen spreads" in printed
    assert "T/k_min = 5.962e-05 nats" in printed


class TestTheCell:
    def test_the_exact_values_round_to_the_declared_floats(self):
        assert cell.departure_from_declared() < 1e-15

    def test_page_one_constants(self):
        constants = prelims.cell_constants()
        assert constants["ridge_mean"] == sympy.sqrt(10)
        assert constants["plug_in_noise"] == 2 * cell.BASE_NOISE
        assert constants["state_step"] == sympy.Rational(3, 200)
        assert constants["observation_step"] == sympy.Rational(9, 200)

    def test_the_spreads_end_exactly_where_geomspace_does(self):
        assert cell.SPREADS[0] == sympy.Rational(1, 200)
        assert sympy.Rational(7, 10) == cell.LARGEST_SPREAD

    def test_the_cap_is_half_the_pole_height(self):
        # The owner's choice of 2026-10-11: sqrt(10)/2 exactly, so R0 - k a^2 is 3/4.
        assert sympy.sqrt(10) / 2 == cell.STRIP_CAP
        floor = cell.BASE_NOISE - cell.CURVATURE * cell.STRIP_CAP**2
        assert floor == sympy.Rational(3, 4)
        assert sympy.sqrt(10) / 20 == cell.CROSSOVER_SPREAD

    def test_the_crossover_is_not_a_declared_spread(self):
        assert cell.CROSSOVER_SPREAD not in cell.SPREADS


class TestTheEngineLattice:
    @pytest.mark.parametrize("spread", threshold.SPREADS)
    def test_the_weights_are_the_step_halved_at_both_ends(self, spread):
        assert prelims.engine_weight_departure(spread) < 1e-12

    @pytest.mark.parametrize("spread", threshold.SPREADS)
    def test_the_nodes_are_evenly_spaced_to_roundoff(self, spread):
        # linspace computes x0 (1 - t) + xN t, so the interior is not x0 + ih bit for
        # bit. Route B assumes the spacing is exact, and this is the size of the miss.
        departure, endpoints_exact = prelims.engine_node_departure(spread)
        assert endpoints_exact
        assert departure < 1e-9, departure


@pytest.mark.parametrize("spread", [threshold.SPREADS[0], threshold.SPREADS[-1]])
def test_the_end_weight_correction_is_subtracted(spread):
    # The scan's step 1 adds it. CLARIFICATION 2026-10-11 item 8 corrects the sign.
    subtracted, added = prelims.end_weight_residuals(spread)
    assert subtracted < 1e-14, subtracted
    assert added > 1e-4, added


class TestTheSteps:
    def test_the_plug_in_variance_has_the_two_other_forms(self):
        closed, precision = prelims.plug_in_variance_residuals()
        assert closed == 0
        assert precision == 0

    def test_the_plug_in_gaussian_is_the_frozen_noise_posterior(self):
        assert prelims.plug_in_posterior_residual() == 0

    def test_step_five_second_line_is_the_first_substituted(self):
        assert prelims.kl_second_line_residual() == 0

    def test_the_exact_kl_expands_as_written(self):
        assert prelims.exact_kl_residual() == 0


class TestTheTargetAndTheRate:
    def test_the_gap_bar(self):
        assert f"{float(cell.GAP_BAR):.2e}" == "5.96e-05"

    def test_the_roundoff_floor(self):
        assert threshold.ROUNDOFF_FLOOR == 100.0 * 2.0**-52
        assert f"{threshold.ROUNDOFF_FLOOR:.1e}" == "2.2e-14"

    def test_route_b_rate_on_the_state_axis(self):
        rate = prelims.state_rate()
        assert rate == 400 * sympy.pi / 3
        assert round(float(rate)) == 419
