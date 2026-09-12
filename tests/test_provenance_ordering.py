"""Every registration ref the check suites declare, checked against the history.

`Provenance` compares two refs for equality. It cannot order them, because ordering is
a property of the commit graph and not of a string. So a registration written after the
number it claims to precede constructs, renders without a marker, and reads exactly like
one written before. This is where that is caught.

The rule: for every declared provenance whose two refs differ, `registered_at` must be
an ancestor of `measured_at`. Same-ref ones are skipped, since they already say in their
own render that history establishes no ordering.

Three sources fail it today. The xfail marks record which and why, rather than
softening the rule until it stops reporting.
"""

import subprocess

import pytest

from research.checks import gap_identity, gap_series, ladder, series_kernel
from research.checks.series_kernel import Source
from warrantlib import Provenance

#: Sources whose registration landed after the measurement that relies on it. The
#: derivation in `research/c4_hand_derivation.md` was committed 2026-08-17, later than
#: the suites citing it (`23f0c47` 2026-08-15, `1888ad4` 2026-08-16). ADR-037 discloses
#: the same ordering for the result these back. Repairing it means re-measuring against
#: the derivation, not editing the refs.
_KNOWN_BACKWARDS = {
    "CONSTRUCTION_SOURCE",
    "EXPANSION_SOURCE",
    "CUMULANT_SOURCE",
}


def _sources():
    """Every declared provenance, as `(name, provenance)` pairs.

    Covers the `Source` records in the check suites and the bare `Provenance` constants
    the demos declare. A registration that only the demo carries is a registration
    nothing checks, which is the gap this module exists to close.
    """
    import crossover

    found = []
    for module in (series_kernel, gap_series):
        for name in dir(module):
            value = getattr(module, name)
            if isinstance(value, Source):
                found.append((name, value.provenance))
    # `gap_identity` declares a bare `Provenance` rather than a `Source`, since every
    # reduction in it rests on the same registration. Walked the same way regardless:
    # what matters is that a declared registration is reachable from here, not which
    # shape carries it.
    for module in (gap_identity, ladder):
        for name in dir(module):
            value = getattr(module, name)
            if isinstance(value, Provenance):
                found.append((f"{module.__name__.rsplit('.', 1)[-1]}.{name}", value))
    for name in dir(crossover):
        value = getattr(crossover, name)
        if isinstance(value, Provenance):
            found.append((name, value))
    return sorted(set(found), key=lambda pair: pair[0])


def _git(*args):
    """Run git in the repository, returning the completed process."""
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=_REPO_ROOT
    )


_REPO_ROOT = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
).stdout.strip()


def _history_is_reachable():
    """Whether this checkout has the commits the refs name.

    A shallow clone has the files and not the graph, so every ancestry query fails for
    a reason that has nothing to do with the claim under test. CI sets
    `fetch-depth: 0`. Without that, every case here skips.
    """
    if not _REPO_ROOT:
        return False
    shallow = _git("rev-parse", "--is-shallow-repository").stdout.strip()
    return shallow == "false"


pytestmark = pytest.mark.skipif(
    not _history_is_reachable(),
    reason="shallow checkout: the commit graph the refs name is not present",
)


def test_the_suites_declare_sources():
    # Guards the collection above. A rename that stops `_sources` finding anything
    # would turn every case below into a silent pass. Floored per source rather than in
    # total: the check suites alone clear 8, so a total-only floor would let the whole
    # demo half vanish and still pass.
    names = {name for name, _ in _sources()}
    assert len(names) >= 12
    from_checks = {n for n in names if n.endswith("_SOURCE")}
    assert len(from_checks) >= 9, from_checks
    assert len(names - from_checks) >= 2, names - from_checks


@pytest.mark.parametrize(
    ("name", "source"), _sources(), ids=[name for name, _ in _sources()]
)
def test_the_registration_ref_exists(name, source):
    found = _git("cat-file", "-e", f"{source.registered_at}^{{commit}}")
    assert found.returncode == 0, (
        f"{name}: registered_at={source.registered_at} is not a commit in "
        "this repository"
    )


@pytest.mark.parametrize(
    ("name", "source"), _sources(), ids=[name for name, _ in _sources()]
)
def test_the_registration_precedes_the_measurement(name, source, request):
    if source.same_ref:
        pytest.skip("one ref: the render already says history orders nothing")
    if name in _KNOWN_BACKWARDS:
        request.node.add_marker(
            pytest.mark.xfail(
                reason=(
                    f"{name}: the hand derivation landed after the suite measuring "
                    "against it, and ADR-037 discloses it."
                ),
                strict=True,
            )
        )
    ancestor = _git(
        "merge-base",
        "--is-ancestor",
        source.registered_at,
        source.measured_at,
    )
    assert ancestor.returncode == 0, (
        f"{name}: registered_at={source.registered_at} is not an ancestor of "
        f"measured_at={source.measured_at}, so the bar was not fixed before the "
        "number it is quoted against"
    )
