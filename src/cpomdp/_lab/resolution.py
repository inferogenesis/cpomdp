"""The error on a difference, for quantities scored against one reference.

Two quantities whose bars overlap are a tie under the conservative rule. But when both
are scored against the *same* reference, the reference's own error moves them
together, and the part it moves cancels in their difference. The error on the
difference can then be far smaller than the sum of the two bars, and a comparison that
uses the sum understates what it can resolve (``research/warrant_ledger.md`` section
4).

So a bar here is split by where it came from. Its ``common_mode`` part is the
first-order shift the reference's error puts on the value, carried *signed* and at the
reference's bar, so that two quantities against one reference subtract it in their
difference. Its ``own`` part is the quantity's alone and adds as bars do. A quantity at
``EXACT`` carries neither.

The threshold a comparison must clear is a function of the two bars only. It is
computed by ``resolution_threshold`` from ``Bar`` values that carry no ``value``
field, which is what makes it pre-registered rather than read off the numbers it
judges. ``resolve`` then reports one of three orderings, with ``NOT_RESOLVED`` a
measured tie and neither confirmation nor refutation.
"""

import math
from dataclasses import dataclass
from enum import Enum


def _require_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


@dataclass(frozen=True)
class Bar:
    """A stated bar, split by whether the reference's error is in it.

    Args:
        common_mode: The signed shift the reference's error puts on the value at the
            reference's bar, ``s · δ_ref`` for the value's sensitivity ``s``. Two bars
            against one reference must take the same sign convention for the
            reference's error, since it is the difference of their shifts that
            survives into a difference's bar.
        own: The part of the bar the quantity alone accounts for, ``≥ 0``.

    Raises:
        ValueError: If ``own`` is negative, or if either field is not finite.
    """

    common_mode: float
    own: float

    def __post_init__(self) -> None:
        """Refuse a bar that cannot be summed."""
        _require_finite(self.common_mode, "common_mode")
        _require_finite(self.own, "own")
        if self.own < 0.0:
            raise ValueError(f"own must be non-negative, got {self.own!r}")

    @property
    def total(self) -> float:
        """The bar as a comparison against an unrelated quantity would read it.

        ``|common_mode| + own``.
        """
        return abs(self.common_mode) + self.own


#: The bar of a closed form at machine precision: nothing from the reference, nothing
#: of its own.
EXACT = Bar(common_mode=0.0, own=0.0)


def difference_bar(a: Bar, b: Bar) -> Bar:
    """The bar on ``a − b``, with the shared reference's error cancelled to first order.

    Its ``common_mode`` is ``a.common_mode − b.common_mode``, still signed against the
    same reference so that a difference can itself be differenced. Its ``own`` is
    ``a.own + b.own``, since bars are bounds and bounds add.
    """
    return Bar(common_mode=a.common_mode - b.common_mode, own=a.own + b.own)


def sum_of_bars(a: Bar, b: Bar) -> float:
    """``a.total + b.total``, the conservative rule's threshold, kept for contrast."""
    return a.total + b.total


def resolution_threshold(a: Bar, b: Bar) -> float:
    """What ``|a − b|`` must exceed for the ordering to be reported.

    ``difference_bar(a, b).total``. Reads the two bars and nothing else, so it can be
    declared before either value is.
    """
    return difference_bar(a, b).total


@dataclass(frozen=True)
class Bounded:
    """A value with its bar.

    Args:
        value: What was measured, finite.
        bar: Its bar.

    Raises:
        ValueError: If ``value`` is not finite.
    """

    value: float
    bar: Bar

    def __post_init__(self) -> None:
        """Refuse a value that cannot be differenced."""
        _require_finite(self.value, "value")


def difference(a: Bounded, b: Bounded) -> Bounded:
    """``a − b`` with ``difference_bar(a.bar, b.bar)``."""
    return Bounded(value=a.value - b.value, bar=difference_bar(a.bar, b.bar))


class Order(Enum):
    """How ``a`` sits against ``b`` once the threshold is applied.

    ``BELOW`` and ``ABOVE`` are reported only when ``|a − b|`` strictly exceeds the
    threshold. ``NOT_RESOLVED`` is the third outcome: the two intervals overlap, and
    the ordering is undetermined at these bars. Two ``EXACT`` values that agree also
    read ``NOT_RESOLVED``, since a difference of zero exceeds nothing.
    """

    BELOW = "BELOW"
    ABOVE = "ABOVE"
    NOT_RESOLVED = "NOT RESOLVED"


@dataclass(frozen=True)
class Resolution:
    """One comparison: the difference, the threshold it was held to, and the order.

    ``sum_of_bars`` travels beside ``threshold`` so a report can show what the shared
    reference bought. They coincide when the two ``common_mode`` parts have opposite
    signs, and the threshold is never the larger.

    Args:
        difference: ``a − b``, with its bar.
        threshold: ``resolution_threshold(a.bar, b.bar)``.
        sum_of_bars: ``sum_of_bars(a.bar, b.bar)``.
    """

    difference: Bounded
    threshold: float
    sum_of_bars: float

    @property
    def order(self) -> Order:
        """``a`` against ``b``: below, above, or not resolved at this threshold."""
        gap = self.difference.value
        if abs(gap) <= self.threshold:
            return Order.NOT_RESOLVED
        return Order.BELOW if gap < 0.0 else Order.ABOVE


def resolve(a: Bounded, b: Bounded) -> Resolution:
    """Compare ``a`` with ``b`` at the error on their difference."""
    return Resolution(
        difference=difference(a, b),
        threshold=resolution_threshold(a.bar, b.bar),
        sum_of_bars=sum_of_bars(a.bar, b.bar),
    )
