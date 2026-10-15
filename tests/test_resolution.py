"""The error on a difference against a shared reference is not the sum of the bars."""

import math

import numpy as np
import pytest

from cpomdp._lab.resolution import (
    EXACT,
    Bar,
    Bounded,
    Order,
    Resolution,
    difference,
    difference_bar,
    resolution_threshold,
    resolve,
    sum_of_bars,
)
from cpomdp._lab.scoring import gaussian_kl


class TestBar:
    def test_total_is_the_unsigned_common_mode_plus_own(self) -> None:
        assert Bar(common_mode=-0.01, own=0.002).total == pytest.approx(0.012)

    def test_exact_carries_nothing(self) -> None:
        assert EXACT.total == 0.0

    def test_refuses_a_negative_own_bar(self) -> None:
        with pytest.raises(ValueError, match="own"):
            Bar(common_mode=0.0, own=-1e-3)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_refuses_a_non_finite_field(self, bad: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            Bar(common_mode=bad, own=0.0)
        with pytest.raises(ValueError, match="finite"):
            Bar(common_mode=0.0, own=bad)


class TestDifferenceBar:
    def test_shared_reference_cancels_and_own_bars_add(self) -> None:
        a = Bar(common_mode=0.010, own=1e-4)
        b = Bar(common_mode=0.011, own=2e-4)
        d = difference_bar(a, b)
        assert d.common_mode == pytest.approx(-0.001)
        assert d.own == pytest.approx(3e-4)

    def test_opposite_signs_do_not_cancel(self) -> None:
        a = Bar(common_mode=0.01, own=0.0)
        b = Bar(common_mode=-0.01, own=0.0)
        assert resolution_threshold(a, b) == pytest.approx(sum_of_bars(a, b))

    def test_threshold_never_exceeds_the_sum_of_bars(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(200):
            a = Bar(common_mode=rng.normal(), own=abs(rng.normal()))
            b = Bar(common_mode=rng.normal(), own=abs(rng.normal()))
            assert resolution_threshold(a, b) <= sum_of_bars(a, b) + 1e-15

    def test_a_difference_can_be_differenced_again(self) -> None:
        # Three rungs against one reference: the second difference still carries a
        # signed common-mode part against that reference.
        first = difference_bar(Bar(0.010, 1e-4), Bar(0.011, 1e-4))
        second = difference_bar(Bar(0.011, 1e-4), Bar(0.013, 1e-4))
        curvature = difference_bar(first, second)
        assert curvature.common_mode == pytest.approx(0.001)
        assert curvature.own == pytest.approx(4e-4)

    def test_threshold_reads_bars_and_never_values(self) -> None:
        # The threshold is a function of Bar alone, which carries no value field.
        assert not hasattr(Bar(0.0, 0.0), "value")
        assert resolution_threshold(Bar(0.01, 1e-4), Bar(0.01, 1e-4)) == pytest.approx(
            2e-4
        )


class TestBounded:
    def test_refuses_a_non_finite_value(self) -> None:
        with pytest.raises(ValueError, match="value"):
            Bounded(value=math.nan, bar=EXACT)

    def test_difference_carries_the_difference_bar(self) -> None:
        a = Bounded(0.500, Bar(0.010, 1e-4))
        b = Bounded(0.503, Bar(0.010, 1e-4))
        d = difference(a, b)
        assert d.value == pytest.approx(-0.003)
        assert d.bar == difference_bar(a.bar, b.bar)


class TestResolve:
    """The worked case: two rungs, one reference, a tie under the sum of bars."""

    A = Bounded(0.500, Bar(common_mode=0.010, own=1e-4))
    B = Bounded(0.503, Bar(common_mode=0.010, own=1e-4))

    def test_the_two_thresholds_differ_on_the_worked_case(self) -> None:
        result = resolve(self.A, self.B)
        assert isinstance(result, Resolution)
        assert result.threshold == pytest.approx(2e-4)
        assert result.sum_of_bars == pytest.approx(0.0202)

    def test_the_sum_of_bars_would_call_it_a_tie(self) -> None:
        result = resolve(self.A, self.B)
        assert abs(result.difference.value) < result.sum_of_bars

    def test_the_difference_bar_resolves_it(self) -> None:
        assert resolve(self.A, self.B).order is Order.BELOW
        assert resolve(self.B, self.A).order is Order.ABOVE

    def test_within_threshold_is_not_resolved(self) -> None:
        near = Bounded(0.5001, self.A.bar)
        assert resolve(self.A, near).order is Order.NOT_RESOLVED

    def test_exactly_at_threshold_is_not_resolved(self) -> None:
        a = Bounded(0.0, Bar(0.0, 1e-3))
        b = Bounded(2e-3, Bar(0.0, 1e-3))
        assert resolve(a, b).order is Order.NOT_RESOLVED

    def test_two_exact_values_resolve_on_any_difference(self) -> None:
        assert resolve(Bounded(1.0, EXACT), Bounded(1.0 + 1e-15, EXACT)).order is (
            Order.BELOW
        )

    def test_two_equal_exact_values_are_not_resolved(self) -> None:
        assert resolve(Bounded(1.0, EXACT), Bounded(1.0, EXACT)).order is (
            Order.NOT_RESOLVED
        )

    def test_the_difference_is_a_minus_b(self) -> None:
        assert resolve(self.A, self.B).difference == difference(self.A, self.B)


class TestAgainstAGaussianReference:
    """Two divergences read against one reference move together when it moves.

    Two nearby Gaussians are scored against a third by ``gaussian_kl``. The reference's
    mean is then shifted by a small amount, standing in for its discretisation error,
    and both divergences move by nearly the same signed amount. The bars built from
    those shifts show the difference resolving where the sum of bars reads a tie.
    """

    COV = ((1.0,),)
    REFERENCE = (1.0,)
    SHIFT = 1e-2
    RUNGS = ((0.0,), (0.01,))

    def _score(self, reference_mean: tuple[float, ...]) -> tuple[float, float]:
        first, second = (
            gaussian_kl(mean, self.COV, reference_mean, self.COV) for mean in self.RUNGS
        )
        return first, second

    def _bounded(self) -> tuple[Bounded, Bounded]:
        at = self._score(self.REFERENCE)
        moved = self._score((self.REFERENCE[0] + self.SHIFT,))
        first, second = (
            Bounded(value, Bar(common_mode=shifted - value, own=0.0))
            for value, shifted in zip(at, moved, strict=True)
        )
        return first, second

    def test_the_shifts_share_a_sign(self) -> None:
        a, b = self._bounded()
        assert math.copysign(1.0, a.bar.common_mode) == math.copysign(
            1.0, b.bar.common_mode
        )

    def test_the_difference_moves_far_less_than_either_term(self) -> None:
        a, b = self._bounded()
        result = resolve(a, b)
        assert result.threshold < 0.02 * result.sum_of_bars

    def test_the_difference_bar_holds_the_observed_shift(self) -> None:
        a, b = self._bounded()
        at = self._score(self.REFERENCE)
        moved = self._score((self.REFERENCE[0] + self.SHIFT,))
        observed = (moved[0] - moved[1]) - (at[0] - at[1])
        assert abs(observed) <= resolve(a, b).threshold + 1e-15

    def test_resolved_at_the_difference_and_tied_at_the_sum(self) -> None:
        a, b = self._bounded()
        result = resolve(a, b)
        assert result.order is Order.ABOVE
        assert abs(result.difference.value) < result.sum_of_bars
