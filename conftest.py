"""Pytest configuration.

Put ``examples/`` on the import path so the acceptance tests can import the models that
are built *with* cpomdp (e.g. the chemotaxis network) rather than shipped inside the
installable library. The examples stay plain, runnable scripts; this only affects how
the test process resolves them.

Turn on jax's persistent compilation cache, so repeated runs skip most XLA compiles.

Also police the ``slow`` marker. The rule is a wall-clock one: past
``SLOW_TEST_SECONDS`` a test belongs off the pull-request path. Remembering to apply
it is the part that rots, so the run measures itself and reports what has drifted over.
"""

import os
import sys
from pathlib import Path

# `pytester` runs a pytest inside this one, which is the only way to assert on what a
# run *reports* rather than on what its checks returned. warrantlib's plugin is loaded
# through an entry point, so it is active in the inner run too. Declared here because
# pytest reads `pytest_plugins` from the rootdir conftest alone.
pytest_plugins = ["pytester"]

_examples = Path(__file__).parent / "examples"
sys.path.insert(0, str(_examples))
sys.path.insert(0, str(_examples / "ffg"))
# The builders test modules share live in `tests/_support.py`.
sys.path.insert(0, str(Path(__file__).parent / "tests"))

# Read at jax import time. The zero threshold caches sub-second compiles too.
# The per-version subdirectory is because jax's entry format differs by python
# version (stdlib zstd from 3.14, zlib before) while its keys do not.
_cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
_python = f"py{sys.version_info.major}.{sys.version_info.minor}"
os.environ.setdefault(
    "JAX_COMPILATION_CACHE_DIR", str(_cache_home / "cpomdp-jax" / _python)
)
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")

# The threshold the `slow` marker means, in seconds. Change it here. The marker's
# description in pyproject.toml points at this name rather than repeating the number.
#
# Wall clock is machine-relative, so the rule needs a reference or it produces a
# different verdict per reviewer. Marks are decided on the durations of an isolated
# run on the owner's development machine. A slower box will see the reporter below
# flag tests that are marked correctly by that rule. So will a parallel run, where
# workers contend for cores and every call time inflates. That is the report doing
# its job on the wrong hardware, not a marking bug. Only reproduce it on the
# reference, serially, before adding a marker.
SLOW_TEST_SECONDS = 20.0

# Nodeids under this are warrant checks, collected from the manifest rather than from
# `tests`. See the reporter below for why they are exempt from the threshold.
_WARRANT_MANIFEST = "research/registered_checks.toml"

_over_threshold: list[tuple[str, float]] = []


def pytest_runtest_logreport(report) -> None:
    """Record any unmarked test whose call phase ran past the threshold.

    Only the call phase counts. Setup and teardown are fixture cost, which marking the
    test would not move off the pull-request path anyway.

    Checks collected from the warrant manifest are exempt. They are not in `testpaths`,
    so they never reach the pull-request path and the marker has nothing to move them
    off. Their cost is also the suite's rather than the item's: whichever check runs
    first pays for the whole symbolic derivation and the rest read the result.
    """
    if report.when != "call" or report.duration <= SLOW_TEST_SECONDS:
        return
    if "slow" in report.keywords or report.nodeid.startswith(_WARRANT_MANIFEST):
        return
    _over_threshold.append((report.nodeid, report.duration))


def pytest_sessionfinish(session, exitstatus) -> None:
    """Delete unreadable jax cache entries after the run.

    jax writes an entry with a plain unlocked ``write_bytes``, so two xdist workers
    compiling the same function can interleave and leave the file corrupt. jax never
    deletes such an entry, only warns on every later read. Controller only: the
    workers are finished by the time it runs.
    """
    if hasattr(session.config, "workerinput"):
        return
    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", "")
    if not cache_dir:
        return
    for entry in Path(cache_dir).glob("*-cache"):
        if not _decompresses(entry):
            entry.unlink(missing_ok=True)


def _decompresses(entry: Path) -> bool:
    try:
        from compression.zstd import decompress  # what jax uses from python 3.14
    except ImportError:
        from zlib import decompress
    try:
        decompress(entry.read_bytes())
    except Exception:
        return False
    return True


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    """Report the drift, without failing on it.

    A hard failure here would turn ordinary CI timing noise into a red build, and the
    threshold is a housekeeping rule rather than a correctness one. Reporting keeps it
    visible on the merge and release runs, which are the ones that execute the slow
    tests and so are the only ones that can see the real durations.
    """
    if not _over_threshold:
        return
    terminalreporter.write_sep(
        "=", "unmarked tests over the slow threshold", yellow=True
    )
    for nodeid, duration in sorted(_over_threshold, key=lambda row: -row[1]):
        terminalreporter.write_line(f"  {duration:6.1f}s  {nodeid}")
    terminalreporter.write_line(
        f"  threshold {SLOW_TEST_SECONDS:.0f}s (conftest.SLOW_TEST_SECONDS). Add "
        f"@pytest.mark.slow, or raise the threshold if the rule has moved."
    )
