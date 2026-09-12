"""Every check declares a key, and no two checks declare the same one.

`check_id` is what a manifest declares before a run and what joins one run's report to
the next. Both jobs fail the same way: two checks sharing a key read as one, and a
dropped check reads as a rename. Neither shows up in a count, which is what the pinned
CI strings compare today.

So the suites are run here and their ids are checked as a set. `research/` is not on
`pytest`'s `testpaths`, so this is where the three symbolic modules and the crossover
falsifiers are covered at all. Only `log_ratio_series` is covered on the pull-request
path: `series_kernel` and `gap_series` both carry the sextic expansion, both are marked
`slow`, and the PR job deselects that marker, so their ids are compared on the merge and
release runs. The measured suites' ids are built by the same helper, and that helper is
pinned below on its own.
"""

import re
from functools import cache

import crossover  # examples/ffg, on the path via the root conftest
import pytest

from research.checks import (
    gap_identity,
    gap_series,
    ladder,
    log_ratio_series,
    series_kernel,
)
from research.checks.gap_kernel import SIGMAS, sigma_slug

#: The symbolic suites, keyed by the prefix every id in them must carry, so a check
#: copied between modules cannot keep the wrong namespace.
_SUITES = {
    "series_kernel": series_kernel.run_checks,
    "log_ratio_series": log_ratio_series.run_checks,
    "gap_series": gap_series.run_checks,
    "gap_identity": gap_identity.run_checks,
    "ladder": ladder.run_checks,
}

#: Reading a suite's check ids runs it. `gap_series` derives `c₂`, `c₄` and `c₆`
#: symbolically and costs minutes, and `series_kernel` checks its expansion against
#: `sympy.series` through `σ⁶` and costs minutes too, both far past the threshold
#: `conftest.SLOW_TEST_SECONDS` sets. `log_ratio_series` stops at first order in `σ` and
#: costs under a second, so it is what keeps the uniqueness guard on the PR path.
_SUITE_CASES = [
    # Marked per suite, not per test: log_ratio_series runs in well under a second and
    # is what keeps the uniqueness guard on the pull-request path at all. The other two
    # carry the sextic expansion and cost minutes.
    pytest.param("series_kernel", marks=pytest.mark.slow),
    "log_ratio_series",
    pytest.param("gap_series", marks=pytest.mark.slow),
    # 24s: the symbolic identities settle in about a second, and the cross-engine
    # comparison runs two full quadrature engines at four spreads.
    pytest.param("gap_identity", marks=pytest.mark.slow),
    # Minutes: five rungs at fifteen spreads on two lattices.
    pytest.param("ladder", marks=pytest.mark.slow),
]


@cache
def _ids(module: str) -> tuple[str, ...]:
    """Every id one suite declares, computed once for the whole session.

    The suites are symbolic and `gap_series` costs tens of seconds, so running one per
    assertion would put this module on the slow path three times over.

    Args:
        module: the suite's name, as its ids are prefixed.

    Returns:
        The ids, in the order the suite reports them.
    """
    return tuple(report.check_id for report in _SUITES[module]())


class TestTheSuitesDeclareDistinctKeys:
    @pytest.mark.parametrize("module", _SUITE_CASES)
    def test_no_two_checks_share_a_key(self, module):
        ids = _ids(module)
        assert len(set(ids)) == len(ids), sorted(
            check_id for check_id in set(ids) if ids.count(check_id) > 1
        )

    @pytest.mark.parametrize("module", _SUITE_CASES)
    def test_every_key_is_namespaced_to_its_module(self, module):
        ids = _ids(module)
        assert ids, "a suite reporting nothing passes both checks above by asking none"
        assert all(check_id.startswith(f"{module}.") for check_id in ids)

    @pytest.mark.slow
    def test_the_suites_do_not_share_keys_with_each_other(self):
        seen = [check_id for module in _SUITES for check_id in _ids(module)]
        assert len(set(seen)) == len(seen)

    @pytest.mark.slow
    def test_the_crossover_falsifiers_declare_distinct_keys(self):
        # Five since the extension axis landed as its own falsifier. Derived from the
        # tuple rather than repeated, so adding a row cannot leave the count behind.
        reports = crossover.falsifiers()
        ids = [report.check_id for report in reports]
        assert len(set(ids)) == len(ids) == len(reports)
        assert len(reports) == 5


class TestTheSigmaSlug:
    """A cell's id names its `σ`, so the rendering has to be one-to-one.

    `--sigmas` takes arbitrary floats. A slug that rounds hands two cells one id, and
    the ledger joining on it reads them as one check.
    """

    @pytest.mark.parametrize(
        ("sigma", "expected"),
        [(0.15, "0p15"), (0.06, "0p06"), (1e-05, "1em05"), (0.0301, "0p0301")],
    )
    def test_it_renders_the_value(self, sigma, expected):
        assert sigma_slug(sigma) == expected

    def test_neighbouring_cells_do_not_share_a_slug(self):
        # Rounded to three decimals these were one id, and `--sigmas` accepts both.
        assert sigma_slug(0.0301) != sigma_slug(0.0302)

    def test_it_is_one_to_one_where_rounding_was_not(self):
        grid = [*SIGMAS, 0.0301, 0.0302, 0.1504, 0.15, 1e-05, 0.1 + 0.2, 0.3]
        assert len({sigma_slug(value) for value in grid}) == len(set(grid))

    @pytest.mark.parametrize("sigma", [*SIGMAS, 0.0301, 1e-05, 1e20, 0.1 + 0.2])
    def test_every_slug_is_key_shaped(self, sigma):
        # The id it lands in is validated by `CheckReport`, so a slug that is not a key
        # takes the whole suite down rather than one row.
        assert re.fullmatch(r"[A-Za-z0-9_]+", sigma_slug(sigma))
