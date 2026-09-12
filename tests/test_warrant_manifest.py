"""The declared inventory, and what it catches that a count could not.

A count says a suite got shorter. It cannot say which check left, it is satisfied by a
different check arriving in place of the one that went, and nothing about it fails until
a person notices the number moved. The manifest replaces all three properties, so these
tests are about the two directions it reconciles and about the file surviving a round
trip through the parser that will read it back.

The writer emits TOML by hand, because nothing in the standard library writes it. So the
round-trip tests below matter more than they usually would. They check the emitted
subset against `tomllib` rather than against a reading of the specification.
"""

import tomllib
from dataclasses import dataclass

import pytest

from warrantlib.manifest import (
    MANIFEST_SCHEMA_VERSION,
    Manifest,
    Reconciliation,
    Suite,
    main,
)


@dataclass(frozen=True)
class _Reported:
    """The one field `rewritten` reads off a report."""

    check_id: str


def _suite(name="series_kernel", checks=("a.one", "a.two")):
    return Suite(
        name=name,
        entry_point=f"research.checks.{name}:run_checks",
        checks=tuple(checks),
    )


def _manifest(*suites):
    return Manifest(suites=suites or (_suite(),))


class TestTheFileSurvivesTheParserThatReadsIt:
    """The writer is ours. The reader is `tomllib`, and it is the one that decides."""

    def test_a_manifest_round_trips(self):
        original = _manifest(
            _suite("series_kernel", ("series_kernel.moment_z0", "series_kernel.gain")),
            _suite("gap_series", ("gap_series.c2_closed_form",)),
        )
        assert Manifest.from_toml(original.to_toml()) == original

    def test_what_the_writer_emits_parses(self):
        parsed = tomllib.loads(_manifest().to_toml())
        assert parsed["schema_version"] == MANIFEST_SCHEMA_VERSION
        assert parsed["suites"]["series_kernel"]["checks"] == ["a.one", "a.two"]

    def test_the_file_says_it_is_generated_and_how(self):
        # The whole reason this is TOML rather than JSON. A reader meeting the diff has
        # to be able to tell that editing it by hand loses the edit.
        text = _manifest().to_toml()
        assert "Generated. Do not edit by hand." in text
        assert "python -m warrantlib.manifest" in text

    def test_one_check_per_line(self):
        # A check that stops reporting should be one line of a diff, not a reflowed
        # array where the reviewer has to find the difference inside a paragraph.
        text = _manifest(_suite(checks=("a.one", "a.two", "a.three"))).to_toml()
        assert '  "a.one",\n  "a.two",\n  "a.three",\n' in text

    def test_an_empty_suite_still_round_trips(self):
        # The state a suite is in between being added by hand and being rewritten.
        original = _manifest(_suite(checks=()))
        assert Manifest.from_toml(original.to_toml()) == original


class TestTheWriterRefusesWhatItCannotEmit:
    """It emits a subset. Emitting outside it silently produces a different file."""

    @pytest.mark.parametrize("check", ['a."quoted"', "a.back\\slash", "a.new\nline"])
    def test_a_value_it_cannot_escape_is_refused(self, check):
        with pytest.raises(ValueError, match="manifest writer"):
            _manifest(_suite(checks=(check,))).to_toml()

    @pytest.mark.parametrize("name", ["has space", "has.dot", "", 'has"quote'])
    def test_a_suite_name_that_is_not_a_bare_key_is_refused(self, name):
        with pytest.raises(ValueError, match="bare TOML key"):
            _manifest(_suite(name=name)).to_toml()


class TestReadingRefuses:
    def test_an_unknown_schema_version_is_refused(self):
        text = _manifest().to_toml().replace(f'"{MANIFEST_SCHEMA_VERSION}"', '"2.0"')
        with pytest.raises(ValueError, match="schema_version"):
            Manifest.from_toml(text)

    def test_a_manifest_with_no_suites_is_refused(self):
        # It would reconcile against anything, which is the state it replaces.
        with pytest.raises(ValueError, match="no suites"):
            Manifest.from_toml(f'schema_version = "{MANIFEST_SCHEMA_VERSION}"\n')

    @pytest.mark.parametrize("field", ["entry_point", "checks"])
    def test_a_suite_missing_a_field_is_refused(self, field):
        text = "\n".join(
            line
            for line in _manifest().to_toml().splitlines()
            if not line.startswith(field)
        )
        # `checks` spans several lines; dropping its opening leaves its entries as a
        # parse error rather than a missing field, which is refused either way.
        with pytest.raises((ValueError, tomllib.TOMLDecodeError)):
            Manifest.from_toml(text)


class TestReconciliation:
    """Two directions, and they mean different things."""

    def test_a_run_reporting_exactly_what_was_declared_agrees(self):
        assert _manifest().reconcile(["a.one", "a.two"]).agrees

    def test_a_dropped_check_is_named(self):
        # The failure the pinned counts existed to catch, and could only count.
        result = _manifest().reconcile(["a.one"])
        assert not result.agrees
        assert result.missing == ("a.two",)
        assert result.unexpected == ()
        assert "declared, not reported: a.two" in str(result)

    def test_a_check_nobody_declared_is_named(self):
        result = _manifest().reconcile(["a.one", "a.two", "a.three"])
        assert result.unexpected == ("a.three",)
        assert "reported, not declared: a.three" in str(result)

    def test_a_swap_is_not_a_count(self):
        # Same number of checks, different checks. This is what a count cannot see.
        result = _manifest().reconcile(["a.one", "a.swapped"])
        assert not result.agrees
        assert result.missing == ("a.two",)
        assert result.unexpected == ("a.swapped",)

    def test_order_does_not_matter(self):
        assert _manifest().reconcile(["a.two", "a.one"]).agrees

    def test_an_agreeing_reconciliation_says_so(self):
        agreed = Reconciliation(missing=(), unexpected=())
        assert str(agreed) == "manifest and run agree"


class TestRewriting:
    def test_it_takes_the_ids_the_suite_reports_now(self, monkeypatch):
        suite = _suite(checks=("a.stale",))
        monkeypatch.setattr(
            Suite, "run", lambda self: [_Reported("a.two"), _Reported("a.one")]
        )
        assert _manifest(suite).rewritten().suites[0].checks == ("a.one", "a.two")

    def test_a_suite_reporting_one_id_twice_is_refused(self, monkeypatch):
        # Two checks under one id are one row in any ledger joining on it, and the
        # manifest would declare one where the run has two.
        monkeypatch.setattr(
            Suite, "run", lambda self: [_Reported("a.one"), _Reported("a.one")]
        )
        with pytest.raises(ValueError, match="more than once"):
            _manifest().rewritten()

    def test_the_entry_point_is_carried_through(self, monkeypatch):
        monkeypatch.setattr(Suite, "run", lambda self: [])
        rewritten = _manifest().rewritten()
        assert rewritten.suites[0].entry_point == _suite().entry_point


class TestARegisteredRefutation:
    """A check whose registered result is a refutation is declared, and survives."""

    def _refuting(self, refuted=("a.two",)):
        return Suite(
            name="series_kernel",
            entry_point="research.checks.series_kernel:run_checks",
            checks=("a.one", "a.two"),
            refuted=tuple(refuted),
        )

    def test_it_round_trips(self):
        original = Manifest(suites=(self._refuting(),))
        assert Manifest.from_toml(original.to_toml()) == original
        parsed = tomllib.loads(original.to_toml())
        assert parsed["suites"]["series_kernel"]["refuted"] == ["a.two"]

    def test_a_suite_declaring_none_writes_no_field(self):
        assert "refuted = [" not in _manifest().to_toml()
        assert Manifest.from_toml(_manifest().to_toml()).suites[0].refuted == ()

    def test_it_must_name_a_declared_check(self):
        with pytest.raises(ValueError, match="declares no such check"):
            self._refuting(refuted=("a.three",))

    def test_it_is_declared_once(self):
        with pytest.raises(ValueError, match="more than once"):
            self._refuting(refuted=("a.two", "a.two"))

    def test_the_field_must_be_a_list(self):
        text = _manifest().to_toml() + 'refuted = "a.two"\n'
        with pytest.raises(ValueError, match="not a list"):
            Manifest.from_toml(text)

    def test_a_rewrite_keeps_what_the_run_still_reports(self, monkeypatch):
        # The declaration over an id the run no longer reports leaves with the id,
        # and the dropped line is in the diff the rewrite is reviewed by.
        suite = self._refuting(refuted=("a.two", "a.one"))
        monkeypatch.setattr(
            Suite, "run", lambda self: [_Reported("a.two"), _Reported("a.new")]
        )
        rewritten = Manifest(suites=(suite,)).rewritten().suites[0]
        assert rewritten.checks == ("a.new", "a.two")
        assert rewritten.refuted == ("a.two",)

    def test_relaid_sorts_it(self):
        relaid = Manifest(suites=(self._refuting(("a.two", "a.one")),)).relaid()
        assert relaid.suites[0].refuted == ("a.one", "a.two")


class TestRunningASuite:
    @pytest.mark.parametrize("entry_point", ["research.checks.series_kernel", "", "x:"])
    def test_an_entry_point_that_names_no_callable_is_refused(self, entry_point):
        suite = Suite(name="s", entry_point=entry_point, checks=())
        with pytest.raises(ValueError, match="module:callable"):
            suite.run()

    def test_a_module_that_does_not_import_is_refused(self):
        # A suite that fails to import reports nothing, and a reconciliation reads that
        # as every one of its checks dropping at once.
        suite = Suite(name="s", entry_point="no.such.module:run_checks", checks=())
        with pytest.raises(ValueError, match="does not import"):
            suite.run()

    def test_a_real_suite_runs(self):
        # No count asserted. A number here is the pinned-count habit this whole change
        # removes, and it would need editing every time a check is added.
        suite = Suite(
            name="log_ratio_series",
            entry_point="research.checks.log_ratio_series:run_checks",
            checks=(),
        )
        reports = suite.run()
        assert reports
        assert all(
            report.check_id.startswith("log_ratio_series.") for report in reports
        )


class TestTheCommandLine:
    """What the command calls current has to be current, in every form of the ask."""

    def _seed(self, tmp_path, checks=()):
        path = tmp_path / "registered_checks.toml"
        path.write_text(
            Manifest(
                suites=(
                    Suite(
                        name="log_ratio_series",
                        entry_point="research.checks.log_ratio_series:run_checks",
                        checks=tuple(checks),
                    ),
                )
            ).to_toml()
        )
        return path

    def test_rewriting_declares_what_the_suite_reports(self, tmp_path, capsys):
        path = self._seed(tmp_path)
        assert main([str(path)]) == 0
        assert "18 checks declared" in capsys.readouterr().out
        assert len(Manifest.from_toml(path.read_text()).checks) == 18

    def test_rewriting_a_current_manifest_changes_nothing(self, tmp_path):
        path = self._seed(tmp_path)
        main([str(path)])
        before = path.read_text()
        assert main([str(path)]) == 0
        assert path.read_text() == before

    def test_check_fails_on_a_stale_manifest_and_names_the_drift(
        self, tmp_path, capsys
    ):
        path = self._seed(tmp_path, ["log_ratio_series.gone"])
        assert main(["--check", str(path)]) == 1
        printed = capsys.readouterr().out
        assert "out of date" in printed
        assert "declared, not reported: log_ratio_series.gone" in printed

    def test_check_changes_nothing(self, tmp_path):
        path = self._seed(tmp_path, ["log_ratio_series.gone"])
        before = path.read_text()
        main(["--check", str(path)])
        assert path.read_text() == before

    def test_check_passes_on_a_current_manifest(self, tmp_path):
        path = self._seed(tmp_path)
        main([str(path)])
        assert main(["--check", str(path)]) == 0

    def test_a_file_whose_only_drift_is_layout_is_not_current(self, tmp_path, capsys):
        # The ids agree, so comparing parsed manifests calls this current and the file
        # keeps a layout the writer no longer produces. Printing the reconciliation
        # here says "manifest and run agree" one line under "out of date".
        path = self._seed(tmp_path)
        main([str(path)])
        path.write_text(path.read_text().replace("\n\n[suites.", "\n[suites."))
        assert main(["--check", str(path)]) == 1
        printed = capsys.readouterr().out
        assert "out of date" in printed
        assert "manifest and run agree" not in printed
        assert "layout" in printed


class TestAskingAboutTheLayoutAlone:
    """`--layout-only`, which is the half of the question the file already answers."""

    #: An entry point that raises the moment anything tries to run it. A test that
    #: passes with this in the file has proved no suite was run.
    UNRUNNABLE = "warrantlib.nowhere:absent"

    def _seed(self, tmp_path, checks=("b.two", "a.one")):
        path = tmp_path / "manifest.toml"
        path.write_text(
            Manifest(
                suites=(
                    Suite(
                        name="unrunnable",
                        entry_point=self.UNRUNNABLE,
                        checks=tuple(checks),
                    ),
                )
            )
            .relaid()
            .to_toml()
        )
        return path

    def test_it_runs_no_suite(self, tmp_path, capsys):
        path = self._seed(tmp_path)
        assert main(["--check", "--layout-only", str(path)]) == 0
        assert "layout current" in capsys.readouterr().out

    def test_the_same_file_without_the_flag_cannot_be_answered_at_all(self, tmp_path):
        # The contrast is the point: reaching the ids means importing the suite, and
        # this one does not import. `--layout-only` never gets that far.
        path = self._seed(tmp_path)
        with pytest.raises(ValueError, match="does not import"):
            main(["--check", str(path)])

    def test_a_layout_the_writer_no_longer_emits_is_named(self, tmp_path, capsys):
        path = self._seed(tmp_path)
        path.write_text(path.read_text().replace("\n\n[suites.", "\n[suites."))
        assert main(["--check", "--layout-only", str(path)]) == 1
        printed = capsys.readouterr().out
        assert "out of date" in printed
        assert "layout" in printed

    def test_ids_declared_out_of_order_are_a_stale_layout(self, tmp_path):
        path = self._seed(tmp_path)
        path.write_text(
            path.read_text().replace('"a.one",\n  "b.two",', '"b.two",\n  "a.one",')
        )
        assert main(["--check", "--layout-only", str(path)]) == 1

    def test_writing_says_it_relaid_rather_than_rewrote(self, tmp_path, capsys):
        # The write path is reachable without `--check`, and it is the cheap repair for
        # a file the layout step flags. Printing "rewritten" here would report what only
        # a run of the suites can establish.
        path = self._seed(tmp_path)
        stale = path.read_text().replace("\n\n[suites.", "\n[suites.")
        path.write_text(stale)
        assert main(["--layout-only", str(path)]) == 0
        assert "relaid" in capsys.readouterr().out
        assert path.read_text() != stale
        assert main(["--check", "--layout-only", str(path)]) == 0

    def test_it_does_not_claim_the_declared_ids_are_current(self, tmp_path, capsys):
        # The boundary, stated where someone reading the output would need it. Only the
        # suites can say whether the ids drifted, and this flag does not run them.
        path = self._seed(tmp_path)
        path.write_text(path.read_text().replace("\n\n[suites.", "\n[suites."))
        main(["--check", "--layout-only", str(path)])
        assert "--layout-only does not ask" in capsys.readouterr().out


class TestTheManifestRefusesWhatWouldReadAsChecks:
    """Shapes that parse as TOML and mean something else entirely."""

    def test_a_bare_string_of_checks_is_refused(self):
        # `tuple("a.one")` is five ids, each a single character, each collected as its
        # own item. The same coercion `_sequence` exists to stop in the wire form.
        text = (
            f'schema_version = "{MANIFEST_SCHEMA_VERSION}"\n\n'
            '[suites.a]\nentry_point = "x:y"\nchecks = "a.one"\n'
        )
        with pytest.raises(ValueError, match="checks"):
            Manifest.from_toml(text)

    def test_a_suite_that_is_not_a_table_is_refused(self):
        text = f'schema_version = "{MANIFEST_SCHEMA_VERSION}"\n\n[suites]\na = "x:y"\n'
        with pytest.raises(ValueError, match="entry_point"):
            Manifest.from_toml(text)

    def test_two_suites_declaring_one_id_are_refused(self):
        # `reconcile` compares sets, so one report satisfies both declarations and the
        # two items collide by name. A key shared by two checks is not a key.
        text = (
            f'schema_version = "{MANIFEST_SCHEMA_VERSION}"\n\n'
            '[suites.a]\nentry_point = "x:y"\nchecks = [\n  "shared.id",\n]\n\n'
            '[suites.b]\nentry_point = "x:z"\nchecks = [\n  "shared.id",\n]\n'
        )
        with pytest.raises(ValueError, match=r"shared\.id"):
            Manifest.from_toml(text)

    def test_a_renamed_runner_is_refused_as_the_type_documents(self):
        suite = Suite(
            name="s",
            entry_point="research.checks.log_ratio_series:no_such_runner",
            checks=(),
        )
        with pytest.raises(ValueError, match="no_such_runner"):
            suite.run()
