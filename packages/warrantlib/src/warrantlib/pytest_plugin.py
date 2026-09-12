"""The warrant vocabulary in a pytest run, instead of a column of dots.

Two halves, and they are independent.

**A test reports its checks.** The `record_check` fixture takes `CheckReport`s, and the
run reports them in the vocabulary the check used rather than as a pass or a skip::

    def test_the_coefficient_is_closed_form(record_check):
        record_check(measure_c2())

**A manifest declares what a suite owes.** Set the `warrant_manifest` ini option and
give pytest the file to collect. Every id the manifest declares becomes an item because
the manifest declares it, so a check that stops reporting still has a row and the row
fails naming it, and a check nobody declared fails too::

    pytest research/registered_checks.toml

`--warrant-detail`, or `-vv`, adds each check's own line: its warrant, its tier, the
reason it gives and the refs it was registered at.

The rest of this is why it is built the way it is.

pytest has three outcomes and a check has five. Collapsing the five loses the two the
vocabulary exists to keep apart: a falsifier void by construction and one measured
elsewhere both read as a skip, and a run that survived everything without deciding
anything reads the same as one that decided it all.

So a check's outcome is carried alongside pytest's rather than replacing it.
`pytest_report_teststatus` sets the progress letter and the verbose word, which is how
`xfail` and `xpass` are spelled too, and the run's own accounting is printed at the end
in the vocabulary the checks reported in.

The pytest outcome underneath stays one of its three. `junitxml` branches on that and on
nothing else, so a fourth value writes the row as nothing at all and every CI parser
downstream sees a test that never ran.

Loaded through the ``pytest11`` entry point, so it is active wherever both packages are
installed. ``warrantlib`` itself imports nothing from here, and importing ``warrantlib``
does not import pytest.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pytest

from warrantlib._serialise import report_from_dict, report_to_dict
from warrantlib._vocabulary import CheckReport, Outcome, check_summary
from warrantlib.manifest import Manifest, Suite

if TYPE_CHECKING:
    from _pytest.reports import TestReport
    from _pytest.terminal import TerminalReporter

#: The oldest pytest this plugin runs on. `@pytest.hookimpl(wrapper=True)` arrived in
#: 8.0 and `Config.get_verbosity` in 7.4, and both are on the ordinary path here.
#:
#: The floor is checked rather than only declared. `pytest>=8` sits in the `pytest`
#: extra, and the `pytest11` entry point is read by any pytest that finds this
#: distribution installed, extra or not. An older one would load this module and fail
#: somewhere inside a hook, which reads as a bug in the suite being run.
_MINIMUM_PYTEST = (8, 0)

if pytest.version_tuple[:2] < _MINIMUM_PYTEST:
    wanted = ".".join(str(part) for part in _MINIMUM_PYTEST)
    raise RuntimeError(
        f"warrantlib's pytest plugin needs pytest {wanted} or newer, and this is "
        f"{pytest.__version__}. It is loaded through the pytest11 entry point, which "
        f"every pytest reads, so install `warrantlib[pytest]` to get a version that "
        f"satisfies it."
    )

#: The three outcomes pytest itself has. Spelled out so the table below cannot drift
#: into a fourth, which `junitxml` would write as no row at all.
_PytestOutcome = Literal["passed", "failed", "skipped"]

#: What each outcome becomes in a run: the pytest outcome, the counting category, the
#: progress letter and the verbose word.
#:
#: `NOT_RESOLVED` fails. It ran and the ordering came out undetermined, which is a
#: question the run could not answer rather than one it answered in the claim's favour.
#: The two that never ran skip, because they did.
#:
#: The category is pytest's own rather than a new one, and this is the one place the
#: vocabulary gives ground. A category is what the closing tally counts under, so a
#: fresh one moves every check out of `N passed` into a name no existing tool reads:
#: `assert_outcomes` stops seeing them and `parseoutcomes` cannot split a two-word name.
#: The distinctions survive in the three places a reader meets them anyway. The letter
#: separates `v` from `e` and from `.` as the run goes, `-v` prints the check's own word
#: instead of `PASSED`, and the warrant summary carries the accounting that decides
#: anything. Buying a headline count with a broken tally is the wrong trade.
_STATUS: dict[Outcome, tuple[_PytestOutcome, str, str, str]] = {
    Outcome.NOT_TRIGGERED: ("passed", "passed", ".", "NOT TRIGGERED"),
    Outcome.FIRED: ("failed", "failed", "F", "FIRED"),
    Outcome.NOT_RESOLVED: ("failed", "failed", "?", "NOT RESOLVED"),
    Outcome.NOT_APPLICABLE: ("skipped", "skipped", "v", "NOT APPLICABLE"),
    Outcome.NOT_RUN_HERE: ("skipped", "skipped", "e", "NOT RUN HERE"),
}

#: The outcome a test takes when it recorded several, worst first. A test recording one
#: fired check and four survivors is a test that found a refutation.
_SEVERITY: tuple[Outcome, ...] = (
    Outcome.FIRED,
    Outcome.NOT_RESOLVED,
    Outcome.NOT_TRIGGERED,
    Outcome.NOT_APPLICABLE,
    Outcome.NOT_RUN_HERE,
)

#: What a test recorded, read by the report hook once its call phase is over.
_RECORDS: pytest.StashKey[list[CheckReport]] = pytest.StashKey()

#: The attribute a `TestReport` carries the records on, and the reason they are carried
#: as mappings rather than as `CheckReport` objects.
#:
#: Under `-n auto` a report crosses a process boundary. `_report_to_json` copies the
#: report's `__dict__` wholesale and `TestReport.__init__` puts the extras back, so an
#: attribute survives the trip on one condition: everything in it is JSON. Records that
#: were dataclasses would need a pair of `pytest_report_*_serializable` hooks to
#: survive. Records that are already the wire form need none, and the run behaves the
#: same way with workers as without.
_REPORT_ATTRIBUTE = "warrant_records"

#: The outcome a report's own status was taken from, set only where it was taken. A test
#: that recorded a check and then raised carries records and keeps pytest's verdict, so
#: the status hook cannot re-derive the word from the records: it would relabel a real
#: error as the outcome of a check that survived, and the error would read as a pass.
_STATUS_ATTRIBUTE = "warrant_status"

#: The suite a reconciliation item checked, set on that item's report alone. Those items
#: are not checks and carry no record, so without this the accounting says 70 where
#: pytest says 73, and the reader is left subtracting.
_RECONCILED_ATTRIBUTE = "warrant_reconciled"

#: What a check the manifest lists as refuted becomes, by whether it fired. A registered
#: refutation is a result the run holds, so the check firing is the pass, and anything
#: else is the change that has to be looked at. The words are the status attribute's
#: values for these two, beside the outcome values it carries otherwise.
_REFUTED = "REFUTED"
_NOT_REFUTED = "NOT REFUTED"
_REFUTED_STATUS: dict[str, tuple[_PytestOutcome, str, str, str]] = {
    _REFUTED: ("passed", "passed", "x", _REFUTED),
    _NOT_REFUTED: ("failed", "failed", "F", _NOT_REFUTED),
}

#: The id of a check the manifest lists as refuted, set on its report, so the accounting
#: can name what was held rather than leave a fired count beside a green run.
_REFUTED_ATTRIBUTE = "warrant_refuted"


#: Every suite's reports, run once per session and read by each of its items. Front
#: loaded on purpose: a suite deriving a coefficient symbolically costs tens of seconds,
#: and running it once per declared check would multiply that by the number of checks.
_SUITE_REPORTS: pytest.StashKey[dict[str, list[CheckReport]]] = pytest.StashKey()


def pytest_addoption(parser: pytest.Parser) -> None:
    """Declare the detail flag, and where the manifest lives.

    `--warrant-detail` prints every check's own line. `warrant_manifest` is an ini
    option rather than a flag, because a project's manifest does not move between runs
    and naming it on every invocation is a step to forget.

    Args:
        parser: the option parser.
    """
    parser.getgroup("warrant").addoption(
        "--warrant-detail",
        action="store_true",
        help=(
            "print every check's own line: its outcome, its warrant, its tier, the "
            "reason it gives and the refs it was registered at. `-vv` does the same, "
            "which is pytest's own spelling for more detail than `-v`"
        ),
    )
    parser.addini(
        "warrant_manifest",
        help=(
            "path to a warrant check manifest, relative to the rootdir. The file has "
            "to reach collection as well, so name it in testpaths or on the command "
            "line."
        ),
        default="",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the run's collector, which needs config and the report stream both.

    Args:
        config: the run's configuration.
    """
    config.stash[_SUITE_REPORTS] = {}
    config.pluginmanager.register(_WarrantRun(), "warrant-run")


def pytest_collect_file(
    file_path: Path, parent: pytest.Collector
) -> pytest.Collector | None:
    """Collect the manifest itself, one item per check it declares.

    The items exist because the manifest declares them, not because a suite reported
    them. That is the whole point: a check that stops reporting still has an item, and
    the item fails naming it. A count could only get smaller.

    Args:
        file_path: the file pytest is considering.
        parent: its collector.

    Returns:
        The manifest's collector, or ``None`` for any other file.
    """
    declared = parent.config.getini("warrant_manifest")
    if not declared or file_path != parent.config.rootpath / declared:
        return None
    return _ManifestFile.from_parent(parent, path=file_path)


class _ManifestFile(pytest.File):
    """The manifest as a collectable file: its suites, and the checks each declares."""

    def collect(self) -> Iterator[pytest.Item]:
        """Yield an item per declared check, and one per suite for the other direction.

        Yields:
            The items.
        """
        manifest = Manifest.from_toml(self.path.read_text())
        for suite in manifest.suites:
            for check_id in suite.checks:
                yield _CheckItem.from_parent(
                    self, name=check_id, suite=suite, check_id=check_id
                )
            yield _UndeclaredItem.from_parent(
                self, name=f"{suite.name}::undeclared", suite=suite
            )


def _reports_of(item: pytest.Item, suite: Suite) -> dict[str, CheckReport]:
    """Run a suite once per session and index what it reported by id.

    Args:
        item: the item asking, for the session stash.
        suite: the suite to run.

    Returns:
        Its reports, by check id.
    """
    cache = item.config.stash[_SUITE_REPORTS]
    if suite.name not in cache:
        cache[suite.name] = suite.run()
    return {report.check_id: report for report in cache[suite.name]}


class _CheckItem(pytest.Item):
    """One declared check, which the suite either reported or did not."""

    def __init__(self, *, suite: Suite, check_id: str, **kwargs: Any) -> None:
        """Remember which check of which suite this is.

        Args:
            suite: the suite that declares it.
            check_id: the check.
            **kwargs: what `from_parent` supplies.
        """
        super().__init__(**kwargs)
        self.suite = suite
        self.check_id = check_id
        self.refuted = check_id in suite.refuted

    def runtest(self) -> None:
        """Look the check up in what the suite reported, and record it.

        Raises:
            Failed: if the manifest declares it and the suite did not report it.
        """
        reported = _reports_of(self, self.suite)
        report = reported.get(self.check_id)
        if report is None:
            pytest.fail(
                f"{self.check_id} is registered in the manifest and this run did not "
                f"report it. Either the check was dropped from "
                f"{self.suite.entry_point}, or it was renamed without the manifest "
                f"being rewritten. A suite that reports fewer checks than it declares "
                f"is asking less than it registered to ask.",
                pytrace=False,
            )
        self.stash[_RECORDS] = [report]

    def reportinfo(self) -> tuple[Path, int, str]:
        """Where a failure points, which is the manifest that declared the check.

        Returns:
            The manifest, its first line, and the check's id.
        """
        return self.path, 0, self.check_id


class _UndeclaredItem(pytest.Item):
    """The other direction: a check the suite reported that nobody declared."""

    def __init__(self, *, suite: Suite, **kwargs: Any) -> None:
        """Remember which suite this reconciles.

        Args:
            suite: the suite.
            **kwargs: what `from_parent` supplies.
        """
        super().__init__(**kwargs)
        self.suite = suite

    def runtest(self) -> None:
        """Fail if the suite reported an id the manifest does not carry.

        Raises:
            Failed: naming every undeclared id.
        """
        reported = set(_reports_of(self, self.suite))
        undeclared = sorted(reported - set(self.suite.checks))
        if undeclared:
            pytest.fail(
                f"{self.suite.name} reported checks the manifest does not declare: "
                f"{', '.join(undeclared)}. A new check is registered by rewriting the "
                f"manifest, so that the run before it and the run after it can be told "
                f"apart. A misspelled id arrives here too.",
                pytrace=False,
            )

    def reportinfo(self) -> tuple[Path, int, str]:
        """Where a failure points, which is the manifest that declared the suite.

        Returns:
            The manifest, its first line, and what this item asks.
        """
        return self.path, 0, f"{self.suite.name}: nothing undeclared"


@pytest.fixture
def record_check(request: pytest.FixtureRequest) -> Callable[..., None]:
    """Hand a `CheckReport` to the run, which reports it in the warrant vocabulary.

    A test that records one check takes that check's outcome. A test that records
    several takes the worst of them, and every one reaches the summary.

        def test_the_coefficient_is_closed_form(record_check):
            record_check(measure_c2())

    Args:
        request: the running test, whose stash the records live on.

    Returns:
        A callable taking one `CheckReport`, or several.
    """
    records: list[CheckReport] = []
    request.node.stash[_RECORDS] = records

    def record(*reports: CheckReport) -> None:
        for report in reports:
            if not isinstance(report, CheckReport):
                raise TypeError(
                    f"record_check was given a {type(report).__name__}. It records a "
                    "CheckReport, which is the only thing carrying the outcome and the "
                    "warrant this run reports in."
                )
        records.extend(reports)

    return record


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, TestReport, TestReport]:
    """Move a test's records onto its report, and take the outcome from them.

    The records are attached whatever the phase, so a run that writes them out later has
    the setup and teardown rows too. The outcome is only taken from the call phase: a
    test that recorded a fired check and then raised in teardown has two problems, and
    overwriting the second with the first hides it.

    Args:
        item: the test.
        call: its phase.

    Returns:
        The report, with the records on it.
    """
    report = yield
    if isinstance(item, _UndeclaredItem) and report.when == "call":
        setattr(report, _RECONCILED_ATTRIBUTE, item.suite.name)
    records = item.stash.get(_RECORDS, [])
    if not records:
        return report
    setattr(report, _REPORT_ATTRIBUTE, [report_to_dict(one) for one in records])
    if report.when == "call" and not call.excinfo:
        if getattr(item, "refuted", False):
            _apply_refutation(report, records)
        else:
            _apply_outcome(report, item, records)
    return report


def _apply_refutation(report: TestReport, records: list[CheckReport]) -> None:
    """Hold a registered refutation: the check firing passes, anything else fails.

    Args:
        report: the call phase's report, passing so far.
        records: the one record a manifest item carries.
    """
    (record,) = records
    fired = record.outcome is Outcome.FIRED
    status = _REFUTED if fired else _NOT_REFUTED
    report.outcome = _REFUTED_STATUS[status][0]
    setattr(report, _STATUS_ATTRIBUTE, status)
    setattr(report, _REFUTED_ATTRIBUTE, record.check_id)
    if not fired:
        report.longrepr = (
            f"{record.check_id}: registered as refuted and reported "
            f"{record.outcome.value}: {record.detail}. A refutation that stopped "
            "firing is a change on the same terms as a check that stopped reporting; "
            "if the result moved, the registration is amended before the manifest is."
        )


def _apply_outcome(
    report: TestReport, item: pytest.Item, records: list[CheckReport]
) -> None:
    """Give a passing test the outcome its checks reported, and something to render.

    An outcome on its own is not enough. The terminal reads a skip's reason out of a
    three-part `longrepr` and asserts on its shape, and a failure with nothing to show
    prints as a row with no cause. Both are filled from the checks that decided it, so
    `-rs` and the failure block say which check and why rather than naming the test.

    Args:
        report: the call phase's report, passing so far.
        item: the test, for the location a skip reason carries.
        records: what it recorded.
    """
    worst = _worst(records)
    outcome = _STATUS[worst][0]
    report.outcome = outcome
    setattr(report, _STATUS_ATTRIBUTE, worst.value)
    deciding = [record for record in records if record.outcome is worst]
    reason = "; ".join(f"{record.check_id}: {record.detail}" for record in deciding)
    if outcome == "skipped":
        # The shape `_get_raw_skip_reason` asserts on: path, line, and a reason it
        # strips its own prefix from.
        report.longrepr = (str(item.path), item.location[1] or 0, f"Skipped: {reason}")
    elif outcome == "failed":
        # No leading word. `pytest_report_teststatus` already puts it in the row and in
        # the short summary, and the short summary takes its reason from the first line
        # of this, so repeating it there gives `FIRED ...::gain - FIRED`.
        report.longrepr = reason


def _worst(records: list[CheckReport]) -> Outcome:
    """The outcome a test takes, given everything it recorded.

    Args:
        records: what the test recorded, in any order.

    Returns:
        The most severe outcome present.
    """
    present = {record.outcome for record in records}
    return next(outcome for outcome in _SEVERITY if outcome in present)


def pytest_report_teststatus(
    report: pytest.CollectReport | pytest.TestReport, config: pytest.Config
) -> tuple[str, str, str] | None:
    """Say what a check-carrying report is called, in the vocabulary the check used.

    Only a report whose own outcome came from its checks is relabelled. A test that
    recorded a check and then failed on its own keeps pytest's word for it, because the
    check did not decide that row.

    Args:
        report: the report being rendered.
        config: the run's configuration.

    Returns:
        The category, the letter and the word, or ``None`` to leave the report alone.
    """
    status = getattr(report, _STATUS_ATTRIBUTE, None)
    if status is None:
        return None
    if status in _REFUTED_STATUS:
        _, category, letter, word = _REFUTED_STATUS[status]
    else:
        _, category, letter, word = _STATUS[Outcome(status)]
    return category, letter, word


def _records_of(report: object) -> list[CheckReport]:
    """The reports a test recorded, read back from whatever carried them.

    Args:
        report: a pytest report, which may carry none.

    Returns:
        The records, empty where the report carries none.
    """
    raw = getattr(report, _REPORT_ATTRIBUTE, None)
    if not raw:
        return []
    return [report_from_dict(one) for one in raw]


class _WarrantRun:
    """One run's checks, gathered as they are reported and summarised at the end.

    A plugin object rather than module state, because the hooks that gather and the hook
    that renders need the same store and pytest hands `config` to only some of them.
    """

    def __init__(self) -> None:
        """Start with nothing recorded."""
        self.records: list[CheckReport] = []
        self.reconciled: list[str] = []
        self.refuted: list[str] = []

    def pytest_runtest_logreport(self, report: TestReport) -> None:
        """Gather a report's checks, on the controller and on a worker alike.

        Args:
            report: the report, which may carry none.
        """
        if report.when != "call":
            return
        self.records.extend(_records_of(report))
        suite = getattr(report, _RECONCILED_ATTRIBUTE, None)
        if suite is not None:
            self.reconciled.append(suite)
        refuted = getattr(report, _REFUTED_ATTRIBUTE, None)
        if refuted is not None:
            self.refuted.append(refuted)

    def pytest_terminal_summary(
        self,
        terminalreporter: TerminalReporter,
        exitstatus: int,
        config: pytest.Config,
    ) -> None:
        """Print the run's accounting, in the vocabulary the checks reported in.

        Silent when the run carried no checks, so a suite that uses none of this reads
        exactly as it did before the plugin was installed.

        Args:
            terminalreporter: where the section goes.
            exitstatus: the run's status, unused.
            config: the run's configuration, unused.
        """
        # A reconciliation with no checks behind it still ran and still counts, which
        # is the case a suite is in between being declared by hand and being rewritten.
        # Returning on the records alone would print nothing at all for it.
        if not self.records and not self.reconciled:
            return
        if self.records and (
            config.getoption("warrant_detail") or config.get_verbosity() >= 2
        ):
            # What a suite prints when it is run on its own. A row's outcome reaches
            # the terminal without this and the reason the check gives does not, so a
            # run replacing those suites would keep the verdicts and lose the record.
            terminalreporter.write_sep("=", "warrant checks")
            for record in sorted(self.records, key=lambda one: one.check_id):
                terminalreporter.write_line(str(record))
        terminalreporter.write_sep("=", "warrant summary")
        for line in check_summary(self.records).splitlines():
            terminalreporter.write_line(line)
        if self.refuted:
            # The fired count above includes these, and they passed. Say which, so a
            # reader is not left reconciling a fired count against a green run.
            terminalreporter.write_line(
                "registered as refuted, and held: " + ", ".join(sorted(self.refuted))
            )
        if self.reconciled:
            # Named rather than counted. These are not checks and carry no warrant, so
            # they are absent from the rows above while pytest counts them, and a
            # reader given only a number is left working out which items they were.
            suites = sorted(set(self.reconciled))
            terminalreporter.write_line(
                f"reconciled against the manifest: {', '.join(suites)}"
            )
            terminalreporter.write_line(
                f"   {len(suites)} item{'s' if len(suites) != 1 else ''}, which pytest "
                "counts and the rows above do not"
            )
