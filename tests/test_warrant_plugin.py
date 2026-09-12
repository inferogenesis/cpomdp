"""What a run says, rather than what its checks returned.

Every other test here reads a `CheckReport` back from the function that built it. None
of them can see what pytest then does with it, and that is the whole job of the plugin:
five outcomes reaching a terminal that has three, an accounting line at the end, and the
same result whether or not the run was split across workers.

So these run pytest inside pytest. `pytester` writes a throwaway suite, runs it, and
hands back the outcomes and the output. The plugin loads in the inner run through its
entry point, the same way it will for anyone who installs it.
"""

import pytest

from warrantlib.manifest import MANIFEST_SCHEMA_VERSION

#: Imported by every generated suite. Kept here so a change to the vocabulary breaks
#: one string rather than a dozen.
_PREAMBLE = """
from warrantlib import CheckReport, Outcome, Tier, Warrant

def _report(check_id, outcome, warrant=Warrant.CORROBORATED, name="a check"):
    if outcome in (Outcome.NOT_APPLICABLE, Outcome.NOT_RUN_HERE):
        warrant = None
    return CheckReport(
        name=name,
        check_id=check_id,
        warrant=warrant,
        outcome=outcome,
        tier=Tier.COMPUTED,
        detail="why it reports what it reports",
    )
"""


def _suite(pytester: pytest.Pytester, body: str) -> None:
    """Write a one-file suite that records checks.

    Args:
        pytester: the inner run's workspace.
        body: the test functions, appended to the shared preamble.
    """
    pytester.makepyfile(_PREAMBLE + body)


class TestTheOutcomeAReportTakes:
    """A check's outcome decides the test's, in pytest's own three."""

    def test_a_surviving_check_passes(self, pytester):
        _suite(
            pytester,
            """
def test_it(record_check):
    record_check(_report("s.survived", Outcome.NOT_TRIGGERED))
""",
        )
        pytester.runpytest().assert_outcomes(passed=1)

    def test_a_fired_check_fails(self, pytester):
        # The refutation is the result, and a run that reports it green is the failure
        # this whole vocabulary exists to prevent.
        _suite(
            pytester,
            """
def test_it(record_check):
    record_check(_report("s.fired", Outcome.FIRED))
""",
        )
        pytester.runpytest().assert_outcomes(failed=1)

    def test_an_unresolved_check_fails(self, pytester):
        # It ran and could not decide. That is not the claim surviving.
        _suite(
            pytester,
            """
def test_it(record_check):
    record_check(_report("s.unresolved", Outcome.NOT_RESOLVED))
""",
        )
        pytester.runpytest().assert_outcomes(failed=1)

    @pytest.mark.parametrize("outcome", ["NOT_APPLICABLE", "NOT_RUN_HERE"])
    def test_a_check_that_never_ran_here_skips(self, pytester, outcome):
        _suite(
            pytester,
            f"""
def test_it(record_check):
    record_check(_report("s.unrun", Outcome.{outcome}))
""",
        )
        pytester.runpytest().assert_outcomes(skipped=1)

    def test_a_test_recording_several_takes_the_worst(self, pytester):
        _suite(
            pytester,
            """
def test_it(record_check):
    record_check(
        _report("s.a", Outcome.NOT_TRIGGERED),
        _report("s.b", Outcome.FIRED),
        _report("s.c", Outcome.NOT_TRIGGERED),
    )
""",
        )
        pytester.runpytest().assert_outcomes(failed=1)

    def test_a_test_recording_nothing_is_left_alone(self, pytester):
        # A suite that uses none of this must read exactly as it did before.
        _suite(
            pytester,
            """
def test_it():
    assert True
""",
        )
        result = pytester.runpytest()
        result.assert_outcomes(passed=1)
        assert "warrant summary" not in result.stdout.str()

    def test_a_raising_test_still_fails_on_its_own_error(self, pytester):
        # The check survived and the test around it did not. Taking the check's outcome
        # here reports a crash as a survivor, which is worse than any label being wrong.
        _suite(
            pytester,
            """
def test_it(record_check):
    record_check(_report("s.survived", Outcome.NOT_TRIGGERED))
    raise RuntimeError("the test itself broke")
""",
        )
        result = pytester.runpytest()
        result.assert_outcomes(failed=1)
        result.stdout.fnmatch_lines(["*RuntimeError: the test itself broke*"])

    def test_a_raising_test_is_not_relabelled_in_the_vocabulary(self, pytester):
        # The outcome and the word it prints are set by two different hooks, so the
        # verdict can be right while the row still reads NOT TRIGGERED beside it.
        _suite(
            pytester,
            """
def test_it(record_check):
    record_check(_report("s.survived", Outcome.NOT_TRIGGERED))
    raise RuntimeError("the test itself broke")
""",
        )
        # Scoped to the test's own row. The summary block below it says NOT TRIGGERED
        # and is right to: the check survived, and the test failed around it.
        rows = [
            line
            for line in pytester.runpytest("-v").stdout.lines
            if "::test_it" in line
        ]
        assert rows
        assert not any("NOT TRIGGERED" in row for row in rows)
        assert any("FAILED" in row for row in rows)

    def test_a_check_that_never_ran_still_reaches_the_summary(self, pytester):
        # The record is attached whatever the verdict, so a run's accounting counts a
        # check the test then failed around rather than losing it.
        _suite(
            pytester,
            """
def test_it(record_check):
    record_check(_report("s.void", Outcome.NOT_APPLICABLE))
    raise RuntimeError("the test itself broke")
""",
        )
        result = pytester.runpytest()
        result.assert_outcomes(failed=1)
        result.stdout.fnmatch_lines(["*1 registered, 0 tested here, none fired*"])


class TestTheVocabularyReachesTheTerminal:
    """`PASSED` and `SKIPPED` are not what a falsifier did."""

    @pytest.mark.parametrize(
        ("outcome", "word"),
        [
            ("NOT_TRIGGERED", "NOT TRIGGERED"),
            ("FIRED", "FIRED"),
            ("NOT_RESOLVED", "NOT RESOLVED"),
            ("NOT_APPLICABLE", "NOT APPLICABLE"),
            ("NOT_RUN_HERE", "NOT RUN HERE"),
        ],
    )
    def test_the_verbose_word_is_the_checks_own(self, pytester, outcome, word):
        _suite(
            pytester,
            f"""
def test_it(record_check):
    record_check(_report("s.check", Outcome.{outcome}))
""",
        )
        result = pytester.runpytest("-v")
        result.stdout.fnmatch_lines([f"*{word}*"])

    def test_the_two_that_never_ran_stay_distinct(self, pytester):
        # Both are skips underneath. Collapsing them loses the survivor accounting,
        # which is the reading ADR-029 split them to prevent.
        _suite(
            pytester,
            """
def test_void(record_check):
    record_check(_report("s.void", Outcome.NOT_APPLICABLE))

def test_elsewhere(record_check):
    record_check(_report("s.elsewhere", Outcome.NOT_RUN_HERE))
""",
        )
        output = pytester.runpytest("-v").stdout.str()
        assert "NOT APPLICABLE" in output
        assert "NOT RUN HERE" in output


class TestTheRunSummary:
    """The accounting a reader needs first: registered, tested here, fired."""

    def test_the_summary_counts_the_run(self, pytester):
        _suite(
            pytester,
            """
def test_a(record_check):
    record_check(_report("s.a", Outcome.NOT_TRIGGERED))

def test_b(record_check):
    record_check(_report("s.b", Outcome.NOT_APPLICABLE))

def test_c(record_check):
    record_check(_report("s.c", Outcome.NOT_RUN_HERE))
""",
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(["*3 registered, 1 tested here, none fired*"])

    def test_a_fired_check_reaches_the_summary(self, pytester):
        _suite(
            pytester,
            """
def test_a(record_check):
    record_check(_report("s.a", Outcome.FIRED))
""",
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(["*1 registered, 1 tested here, 1 fired*"])


class TestTheRunSurvivesWorkers:
    """The suite runs under `-n auto`, so a worker boundary is the ordinary path."""

    def test_the_summary_is_the_same_split_across_workers(self, pytester):
        _suite(
            pytester,
            """
def test_a(record_check):
    record_check(_report("s.a", Outcome.NOT_TRIGGERED))

def test_b(record_check):
    record_check(_report("s.b", Outcome.NOT_APPLICABLE))

def test_c(record_check):
    record_check(_report("s.c", Outcome.NOT_TRIGGERED))
""",
        )
        # A report crosses a process boundary as JSON. Records that were dataclasses
        # would arrive as nothing and the summary would pass by counting none.
        line = "*3 registered, 2 tested here, none fired*"
        pytester.runpytest().stdout.fnmatch_lines([line])
        split = pytester.runpytest("-n", "2", "--dist", "worksteal")
        split.stdout.fnmatch_lines([line])


class TestTheFixtureRefusesWhatItCannotRecord:
    def test_something_that_is_not_a_report_is_refused(self, pytester):
        _suite(
            pytester,
            """
def test_it(record_check):
    record_check("s.a: NOT TRIGGERED")
""",
        )
        result = pytester.runpytest()
        result.assert_outcomes(failed=1)
        result.stdout.fnmatch_lines(["*record_check was given a str*"])


#: A suite the inner run can import, writing whichever ids the case needs. The reports
#: it returns are real `CheckReport`s, since the collector reads `check_id` off them.
_SUITE = """
from warrantlib import CheckReport, Outcome, Tier, Warrant

REPORTED = {ids}
FIRED = {fired}

def run_checks():
    return [
        CheckReport(
            name="a check",
            check_id=check_id,
            warrant=Warrant.CORROBORATED,
            outcome=Outcome.FIRED if check_id in FIRED else Outcome.NOT_TRIGGERED,
            tier=Tier.COMPUTED,
            detail="why it reports what it reports",
        )
        for check_id in REPORTED
    ]
"""


def _manifest_suite(pytester, *, declared, reported, fired=(), refuted=()):
    """Write a suite reporting `reported` and a manifest declaring `declared`.

    `fired` names the ids the suite reports as FIRED, and `refuted` the ids the
    manifest declares as a registered refutation.
    """
    pytester.makepyfile(a_suite=_SUITE.format(ids=list(reported), fired=list(fired)))
    checks = "".join(f'  "{check}",\n' for check in declared)
    held = "".join(f'  "{check}",\n' for check in refuted)
    refuted_block = f"refuted = [\n{held}]\n" if refuted else ""
    pytester.makefile(
        ".toml",
        registered_checks=(
            f'schema_version = "{MANIFEST_SCHEMA_VERSION}"\n\n'
            "[suites.a_suite]\n"
            'entry_point = "a_suite:run_checks"\n'
            f"checks = [\n{checks}]\n" + refuted_block
        ),
    )
    # `pythonpath` so the inner run can import the suite the entry point names. In a
    # real project the suite is an installed package and this is already true.
    pytester.makeini(
        "[pytest]\nwarrant_manifest = registered_checks.toml\npythonpath = .\n"
    )
    return "registered_checks.toml"


class TestTheManifestBecomesItems:
    """A declared check is an item because it was declared, not because it ran."""

    def test_every_declared_check_is_collected(self, pytester):
        path = _manifest_suite(
            pytester, declared=["a.one", "a.two"], reported=["a.one", "a.two"]
        )
        result = pytester.runpytest(path, "--collect-only", "-q")
        result.stdout.fnmatch_lines(
            ["*::a.one", "*::a.two", "*::a_suite::undeclared"], consecutive=False
        )

    def test_a_run_reporting_what_it_declared_passes(self, pytester):
        path = _manifest_suite(
            pytester, declared=["a.one", "a.two"], reported=["a.one", "a.two"]
        )
        result = pytester.runpytest(path)
        result.assert_outcomes(passed=3)
        result.stdout.fnmatch_lines(["*2 registered, 2 tested here, none fired*"])

    def test_a_dropped_check_fails_and_names_itself(self, pytester):
        # The failure the pinned counts existed to catch, and could only count.
        path = _manifest_suite(
            pytester, declared=["a.one", "a.two"], reported=["a.one"]
        )
        result = pytester.runpytest(path)
        result.assert_outcomes(passed=2, failed=1)
        result.stdout.fnmatch_lines(["*a.two is registered in the manifest*"])

    def test_a_check_nobody_declared_fails_and_names_itself(self, pytester):
        path = _manifest_suite(
            pytester, declared=["a.one"], reported=["a.one", "a.new"]
        )
        result = pytester.runpytest(path)
        result.assert_outcomes(passed=1, failed=1)
        result.stdout.fnmatch_lines(["*does not declare: a.new*"])

    def test_a_rename_reads_as_a_drop_and_an_addition(self, pytester):
        # The case a count cannot see at all: same number of checks, different checks.
        path = _manifest_suite(pytester, declared=["a.one"], reported=["a.one_renamed"])
        result = pytester.runpytest(path)
        result.assert_outcomes(failed=2)
        output = result.stdout.str()
        assert "a.one is registered in the manifest" in output
        assert "does not declare: a.one_renamed" in output

    def test_a_suite_that_does_not_import_fails_rather_than_reporting_nothing(
        self, pytester
    ):
        path = _manifest_suite(pytester, declared=["a.one"], reported=["a.one"])
        pytester.path.joinpath("a_suite.py").unlink()
        result = pytester.runpytest(path)
        result.assert_outcomes(failed=2)
        result.stdout.fnmatch_lines(["*does not import*"])

    def test_the_suite_runs_once_however_many_checks_it_declares(self, pytester):
        # Front-loading the repo requires: a suite deriving a coefficient symbolically
        # costs tens of seconds, and one run per declared check multiplies that.
        pytester.makepyfile(
            a_suite=_SUITE.format(ids=["a.one", "a.two", "a.three"], fired=[])
            + "\nRUNS = []\n"
            "_original = run_checks\n"
            "def run_checks():\n"
            "    RUNS.append(1)\n"
            "    return _original()\n"
        )
        checks = "".join(f'  "a.{n}",\n' for n in ("one", "two", "three"))
        pytester.makefile(
            ".toml",
            registered_checks=(
                f'schema_version = "{MANIFEST_SCHEMA_VERSION}"\n\n[suites.a_suite]\n'
                'entry_point = "a_suite:run_checks"\n'
                f"checks = [\n{checks}]\n"
            ),
        )
        pytester.makeini(
            "[pytest]\nwarrant_manifest = registered_checks.toml\npythonpath = .\n"
        )
        pytester.makeconftest(
            "def pytest_sessionfinish(session):\n"
            "    import a_suite\n"
            "    assert len(a_suite.RUNS) == 1, a_suite.RUNS\n"
        )
        pytester.runpytest("registered_checks.toml").assert_outcomes(passed=4)

    def test_a_manifest_nothing_points_at_is_left_alone(self, pytester):
        # The ini option is unset, so the file is ordinary and pytest ignores it.
        pytester.makefile(
            ".toml", registered_checks=f'schema_version = "{MANIFEST_SCHEMA_VERSION}"\n'
        )
        _suite(
            pytester,
            """
def test_it():
    assert True
""",
        )
        pytester.runpytest().assert_outcomes(passed=1)

    def test_the_summary_accounts_for_the_reconciliation_items(self, pytester):
        # pytest counts three items and the vocabulary counts two checks. Leaving the
        # reader to subtract is how a correct summary reads like a wrong one.
        path = _manifest_suite(
            pytester, declared=["a.one", "a.two"], reported=["a.one", "a.two"]
        )
        result = pytester.runpytest(path)
        result.assert_outcomes(passed=3)
        result.stdout.fnmatch_lines(
            [
                "*2 registered, 2 tested here, none fired*",
                "*reconciled against the manifest: a_suite*",
                "*1 item, which pytest counts and the rows above do not*",
            ],
            consecutive=False,
        )

    def test_a_suite_declaring_no_checks_still_names_its_reconciliation(self, pytester):
        # The state a suite is in between being declared by hand and being rewritten.
        # Nothing records a check, so a summary keyed on the records alone prints
        # nothing at all, and the one item pytest counted is left unexplained.
        path = _manifest_suite(pytester, declared=[], reported=[])
        result = pytester.runpytest(path)
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(
            [
                "*warrant summary*",
                "*0 registered, 0 tested here, none fired*",
                "*reconciled against the manifest: a_suite*",
            ],
            consecutive=False,
        )

    def test_the_reconciled_suites_are_named_not_counted(self, pytester):
        # A bare count leaves the reader working out which items they were.
        pytester.makepyfile(a_suite=_SUITE.format(ids=["a.one"], fired=[]))
        pytester.makepyfile(b_suite=_SUITE.format(ids=["b.one"], fired=[]))
        pytester.makefile(
            ".toml",
            registered_checks=(
                f'schema_version = "{MANIFEST_SCHEMA_VERSION}"\n\n'
                '[suites.a_suite]\nentry_point = "a_suite:run_checks"\n'
                'checks = [\n  "a.one",\n]\n\n'
                '[suites.b_suite]\nentry_point = "b_suite:run_checks"\n'
                'checks = [\n  "b.one",\n]\n'
            ),
        )
        pytester.makeini(
            "[pytest]\nwarrant_manifest = registered_checks.toml\npythonpath = .\n"
        )
        result = pytester.runpytest("registered_checks.toml")
        # Two checks and two reconciliations. Asserting the outcomes as well, or the
        # line below passes just as well on a run where something failed.
        result.assert_outcomes(passed=4)
        result.stdout.fnmatch_lines(
            ["*reconciled against the manifest: a_suite, b_suite*"]
        )


class TestARegisteredRefutationIsHeld:
    """A check the manifest lists as refuted passes by firing, and fails otherwise."""

    def test_a_declared_refutation_that_fires_passes(self, pytester):
        path = _manifest_suite(
            pytester,
            declared=["a.one", "a.two"],
            reported=["a.one", "a.two"],
            fired=["a.two"],
            refuted=["a.two"],
        )
        result = pytester.runpytest(path, "-v")
        result.assert_outcomes(passed=3)
        result.stdout.fnmatch_lines(
            ["*a::two REFUTED*", "*registered as refuted, and held: a.two*"],
            consecutive=False,
        )

    def test_the_fired_count_still_says_it_fired(self, pytester):
        # The accounting is the checks' own. Holding a refutation changes what pytest
        # counts, not what the check reported.
        path = _manifest_suite(
            pytester,
            declared=["a.one", "a.two"],
            reported=["a.one", "a.two"],
            fired=["a.two"],
            refuted=["a.two"],
        )
        result = pytester.runpytest(path)
        result.stdout.fnmatch_lines(["*2 registered, 2 tested here, 1 fired*"])

    def test_a_declared_refutation_that_stops_firing_fails_by_name(self, pytester):
        path = _manifest_suite(
            pytester,
            declared=["a.one", "a.two"],
            reported=["a.one", "a.two"],
            refuted=["a.two"],
        )
        result = pytester.runpytest(path, "-v")
        result.assert_outcomes(passed=2, failed=1)
        result.stdout.fnmatch_lines(
            [
                "*a::two NOT REFUTED*",
                "*a.two: registered as refuted and reported NOT TRIGGERED*",
            ],
            consecutive=False,
        )

    def test_a_fire_nobody_registered_still_fails(self, pytester):
        path = _manifest_suite(
            pytester,
            declared=["a.one", "a.two"],
            reported=["a.one", "a.two"],
            fired=["a.two"],
        )
        result = pytester.runpytest(path)
        result.assert_outcomes(passed=2, failed=1)

    def test_a_refutation_over_an_undeclared_check_does_not_collect(self, pytester):
        path = _manifest_suite(
            pytester, declared=["a.one"], reported=["a.one"], refuted=["a.two"]
        )
        result = pytester.runpytest(path)
        result.assert_outcomes(errors=1)
        result.stdout.fnmatch_lines(["*declares no such check*"])


class TestTheDetailFlag:
    """The reason a check gives, which the verdict alone does not carry."""

    def test_it_is_off_by_default(self, pytester):
        _suite(
            pytester,
            """
def test_it(record_check):
    record_check(_report("s.a", Outcome.NOT_TRIGGERED))
""",
        )
        assert "warrant checks" not in pytester.runpytest().stdout.str()

    def test_it_prints_each_check_in_full(self, pytester):
        _suite(
            pytester,
            """
def test_it(record_check):
    record_check(_report("s.a", Outcome.NOT_TRIGGERED))
""",
        )
        result = pytester.runpytest("--warrant-detail")
        result.stdout.fnmatch_lines(
            [
                "*warrant checks*",
                "*a check: NOT TRIGGERED (CORROBORATED, tier computed).*",
            ],
            consecutive=False,
        )

    def test_the_order_is_the_id_order_not_the_run_order(self, pytester):
        # Under `-n auto` the reporting order is whatever the workers finish in, so a
        # log nobody can diff against the last one is the alternative.
        _suite(
            pytester,
            """
def test_b(record_check):
    record_check(_report("s.zeta", Outcome.NOT_TRIGGERED, name="zeta"))

def test_a(record_check):
    record_check(_report("s.alpha", Outcome.NOT_TRIGGERED, name="alpha"))
""",
        )
        lines = [
            line
            for line in pytester.runpytest("--warrant-detail").stdout.lines
            if line.startswith(("alpha:", "zeta:"))
        ]
        assert [line.split(":")[0] for line in lines] == ["alpha", "zeta"]

    def test_double_verbose_turns_it_on(self, pytester):
        # pytest's own spelling for more detail than `-v`, so the plugin does not mint
        # a short flag of its own into an alphabet the ecosystem has nearly used up.
        _suite(
            pytester,
            """
def test_it(record_check):
    record_check(_report("s.a", Outcome.NOT_TRIGGERED))
""",
        )
        assert "warrant checks" in pytester.runpytest("-vv").stdout.str()

    def test_single_verbose_does_not(self, pytester):
        _suite(
            pytester,
            """
def test_it(record_check):
    record_check(_report("s.a", Outcome.NOT_TRIGGERED))
""",
        )
        assert "warrant checks" not in pytester.runpytest("-v").stdout.str()


class TestAFiringCheckReadsLikeAFailingTest:
    """No flags. What pytest gives natively for a failure, the check gives too."""

    def test_the_failure_block_carries_the_checks_reason(self, pytester):
        _suite(
            pytester,
            """
def test_it(record_check):
    record_check(_report("s.gain", Outcome.FIRED))
""",
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(
            ["*FAILURES*", "*s.gain: why it reports what it reports*"],
            consecutive=False,
        )

    def test_the_short_summary_does_not_repeat_the_word(self, pytester):
        # The row prefix and the short summary both carry the word already, so putting
        # it at the head of the reason gives `FIRED ...::test_it - FIRED`.
        _suite(
            pytester,
            """
def test_it(record_check):
    record_check(_report("s.gain", Outcome.FIRED))
""",
        )
        rows = [
            line
            for line in pytester.runpytest("-rA").stdout.lines
            if line.startswith("FIRED ")
        ]
        assert rows
        assert not any(row.rstrip().endswith("- FIRED") for row in rows)

    def test_the_accounting_still_prints_when_something_fired(self, pytester):
        _suite(
            pytester,
            """
def test_a(record_check):
    record_check(_report("s.a", Outcome.NOT_TRIGGERED))

def test_b(record_check):
    record_check(_report("s.b", Outcome.FIRED))
""",
        )
        result = pytester.runpytest()
        result.assert_outcomes(passed=1, failed=1)
        result.stdout.fnmatch_lines(["*2 registered, 2 tested here, 1 fired*"])
