"""The checks a suite is supposed to report, declared before it runs.

A count cannot say which check went missing. Drop a stage from a suite's runner and the
run is shorter, every remaining check still passes, and the only thing in the way is a
number somebody has to notice changed. That number is satisfied by adding a different
check too, so it does not even say the same checks ran.

A manifest names them. It is generated from the suites themselves, checked in, and read
back before the next run, so a check that stops reporting is named rather than counted.
Regenerating it is a deliberate edit reviewed with the diff, which is what the count was
reaching for.

TOML, so the file says what it is and how to regenerate it without a reader having to
find that out elsewhere (ADR-045). ``tomllib`` reads it. Nothing in the standard library
writes it, so the writer here emits a restricted subset by hand: table headers, quoted
strings, and arrays of quoted strings. A round-trip test parses what it wrote with
``tomllib`` and compares, so the subset is checked against the real parser.

Run ``python -m warrantlib.manifest <path>`` to rewrite one after a suite changes, and
``--check`` to ask whether it is current without touching it. Adding a suite means
adding its entry point by hand, with no checks, and then rewriting.

A check whose registered result is a refutation is listed under ``refuted``. A
falsifier that fired is a result, and a run that holds it there has to say so: with
the declaration, the check firing is what reconciles, and the check not firing fails
by name, since a refutation that vanished is a change on the same terms as a check
that stopped reporting. The list is written by hand after the result is on record,
and a rewrite carries it.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = "1.1"
"""The version of the manifest's form, carried on the file.

Bumped whenever a field is added, removed or renamed. A manifest this version does not
know is refused rather than half-read, on the same terms the report wire form is.
"""

#: The header the writer puts at the top of every manifest. A generated file that does
#: not say so invites someone to edit it by hand and lose the edit on the next run.
_HEADER = """\
# Generated. Do not edit by hand.
#
# Every check a suite is registered to report. A check declared here that stops
# reporting fails the run by name, which is what a count of them could not do.
#
# Rewrite after a suite changes:  python -m warrantlib.manifest <this file>
# Ask whether it is current:      python -m warrantlib.manifest --check <this file>
#
# Adding a suite: add its [suites.<name>] table with an entry_point and no checks,
# then rewrite.
#
# A check whose registered result is a refutation is listed under `refuted`, by hand
# and after the result is on record. It then reconciles by firing, and fails by name
# if it stops. A rewrite keeps the list."""

#: What a bare key may hold, so the writer never has to quote one. Suite names come from
#: the file itself, and anything outside this needs TOML's quoting rules rather than the
#: restricted subset the writer emits.
_BARE_KEY = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
)


@dataclass(frozen=True)
class Suite:
    """One runner and every check id it is registered to report.

    Args:
        name: the suite, as the manifest keys it and as a message names it.
        entry_point: ``module:callable``, returning the suite's reports. Imported when
            the manifest is rewritten, and by anything running the suite from it.
        checks: the ids the suite reported when the manifest was last written, sorted so
            the file's order is the file's own rather than the runner's.
        refuted: the ids among ``checks`` whose registered result is a refutation.
            Declared by hand once the result is on record. A run in which one of them
            fires reconciles, and a run in which it does not fails by name.

    Raises:
        ValueError: if ``refuted`` names an id that is not one of ``checks``, or names
            one twice.
    """

    name: str
    entry_point: str
    checks: tuple[str, ...]
    refuted: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject a refutation declared over a check the suite does not declare."""
        unknown = sorted(set(self.refuted) - set(self.checks))
        if unknown:
            raise ValueError(
                f"suite {self.name!r} lists {', '.join(unknown)} as refuted, and "
                "declares no such check. A refutation is a result a declared check "
                "reported, so the id is registered under checks first."
            )
        repeated = sorted({one for one in self.refuted if self.refuted.count(one) > 1})
        if repeated:
            raise ValueError(
                f"suite {self.name!r} lists {', '.join(repeated)} as refuted more "
                "than once. One declaration per check, so the diff that records a "
                "refutation is one line."
            )

    def run(self) -> list[Any]:
        """Import the entry point and call it.

        Returns:
            Whatever the runner returns, which is its reports.

        Raises:
            ValueError: if the entry point is not ``module:callable``, or the module
                does not import.
        """
        module_name, separator, attribute = self.entry_point.partition(":")
        if not separator or not attribute:
            raise ValueError(
                f"suite {self.name!r} has entry_point={self.entry_point!r}, which is "
                "not module:callable. Without the callable there is nothing to run, "
                "and a module that merely imports reports no checks."
            )
        try:
            module = importlib.import_module(module_name)
        except ImportError as failure:
            raise ValueError(
                f"suite {self.name!r} names {module_name!r}, which does not import: "
                f"{failure}. A suite that cannot be imported reports no checks, and a "
                "reconciliation reads that as every one of them dropped at once."
            ) from failure
        runner = getattr(module, attribute, None)
        if runner is None:
            raise ValueError(
                f"suite {self.name!r} names {attribute!r} in {module_name!r}, which "
                "has no such attribute. A renamed runner is the same failure as a "
                "renamed check and is reported the same way, rather than as the "
                "AttributeError of whoever called it."
            )
        return list(runner())


@dataclass(frozen=True)
class Reconciliation:
    """What a run reported, against what the manifest declared.

    Both directions matter and they mean different things. A declared check with no
    report is a stage that stopped running. A reported check nobody declared is a typo
    in an id, or a check somebody wrote without registering it.

    Args:
        missing: declared, not reported.
        unexpected: reported, not declared.
    """

    missing: tuple[str, ...]
    unexpected: tuple[str, ...]

    @property
    def agrees(self) -> bool:
        """Whether the run reported exactly what was declared."""
        return not self.missing and not self.unexpected

    def __str__(self) -> str:
        """The disagreement as one line per check, or one line saying there is none."""
        if self.agrees:
            return "manifest and run agree"
        lines = [f"declared, not reported: {check}" for check in self.missing]
        lines += [f"reported, not declared: {check}" for check in self.unexpected]
        return "\n".join(lines)


@dataclass(frozen=True)
class Manifest:
    """Every suite a project registers, and every check each one declares.

    Args:
        suites: the suites, in the order the file lists them.
    """

    suites: tuple[Suite, ...]

    @property
    def checks(self) -> tuple[str, ...]:
        """Every declared id across every suite, in file order."""
        return tuple(check for suite in self.suites for check in suite.checks)

    def reconcile(self, reported: Iterable[str]) -> Reconciliation:
        """Compare a run's ids against the declared ones.

        Args:
            reported: the ids a run produced, in any order.

        Returns:
            What each side has that the other does not.
        """
        declared, seen = set(self.checks), set(reported)
        return Reconciliation(
            missing=tuple(sorted(declared - seen)),
            unexpected=tuple(sorted(seen - declared)),
        )

    def rewritten(self) -> Manifest:
        """Run every suite and take the ids it reports now.

        Returns:
            The same suites, carrying their current ids, sorted.

        Raises:
            ValueError: if a suite reports two checks under one id. The manifest would
                declare one where the run has two, and a count of them would agree while
                the checks did not.
        """
        suites = []
        for suite in self.suites:
            ids = [report.check_id for report in suite.run()]
            repeated = sorted({one for one in ids if ids.count(one) > 1})
            if repeated:
                raise ValueError(
                    f"suite {suite.name!r} reports {', '.join(repeated)} more than "
                    "once. Two checks under one id are one row in any ledger joining "
                    "on it, and a manifest cannot declare the difference between them."
                )
            # A refutation declared over an id the run no longer reports leaves with
            # the id. The rewrite is the reviewed half, so the dropped line is in the
            # diff.
            suites.append(
                Suite(
                    name=suite.name,
                    entry_point=suite.entry_point,
                    checks=tuple(ids),
                    refuted=tuple(one for one in suite.refuted if one in ids),
                )
            )
        return Manifest(suites=tuple(suites)).relaid()

    def relaid(self) -> Manifest:
        """The same suites and the same ids, in the order the writer emits them.

        Whether a manifest's *ids* are current is a question only the suites can
        answer. Whether its *layout* is current is answerable from what the file
        already declares, and this is that half on its own.

        Returns:
            The same manifest with every suite's checks sorted.
        """
        return Manifest(
            suites=tuple(
                Suite(
                    name=suite.name,
                    entry_point=suite.entry_point,
                    checks=tuple(sorted(suite.checks)),
                    refuted=tuple(sorted(suite.refuted)),
                )
                for suite in self.suites
            )
        )

    def to_toml(self) -> str:
        """The manifest as the file holds it.

        One id per line, so a check that stopped reporting is one line of a diff.

        Returns:
            The file's contents, header included.

        Raises:
            ValueError: if a suite's name is not a bare key, or any value holds a
                character the restricted writer does not escape.
        """
        blocks = [_HEADER, f'schema_version = "{_quoted(MANIFEST_SCHEMA_VERSION)}"']
        for suite in self.suites:
            if not suite.name or set(suite.name) - _BARE_KEY:
                raise ValueError(
                    f"suite name {suite.name!r} is not a bare TOML key. The writer "
                    "emits letters, digits, underscores and hyphens, and quoting the "
                    "rest correctly is the part of the format it does not implement."
                )
            entries = "".join(f'  "{_quoted(check)}",\n' for check in suite.checks)
            block = (
                f"[suites.{suite.name}]\n"
                f'entry_point = "{_quoted(suite.entry_point)}"\n'
                f"checks = [\n{entries}]"
            )
            if suite.refuted:
                refuted = "".join(f'  "{_quoted(check)}",\n' for check in suite.refuted)
                block += f"\nrefuted = [\n{refuted}]"
            blocks.append(block)
        return "\n\n".join(blocks) + "\n"

    @classmethod
    def from_toml(cls, text: str) -> Manifest:
        """Read a manifest back.

        Args:
            text: the file's contents.

        Returns:
            The manifest.

        Raises:
            ValueError: if the version is not this one, or a suite is missing a field.
        """
        body = tomllib.loads(text)
        version = body.get("schema_version")
        if version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"manifest has schema_version={version!r}, and this is "
                f"{MANIFEST_SCHEMA_VERSION!r}. Reading it would mean guessing which "
                "fields moved, and a reconciliation that guesses names checks as "
                "dropped that were only spelled differently."
            )
        suites = body.get("suites")
        if not isinstance(suites, dict) or not suites:
            raise ValueError(
                "manifest declares no suites. A file declaring nothing reconciles "
                "against anything, which is the state it exists to replace."
            )
        manifest = cls(
            suites=tuple(
                Suite(
                    name=name,
                    entry_point=_field(name, entry, "entry_point"),
                    checks=_ids(name, entry, "checks"),
                    refuted=_ids(name, entry, "refuted", optional=True),
                )
                for name, entry in suites.items()
            )
        )
        declared = manifest.checks
        shared = sorted({one for one in declared if declared.count(one) > 1})
        if shared:
            raise ValueError(
                f"manifest declares {', '.join(shared)} in more than one suite. "
                "`reconcile` compares sets, so one report would satisfy both "
                "declarations, and the two items collide under one name. A key shared "
                "by two checks is not a key."
            )
        return manifest


def _quoted(value: str) -> str:
    """A string the restricted writer can put inside double quotes.

    Args:
        value: the value to write.

    Returns:
        It, unchanged.

    Raises:
        ValueError: if it holds a quote, a backslash or a control character. Escaping
            those is the part of TOML this writer does not implement, and emitting them
            raw produces a file that parses as something else.
    """
    if any(character in value for character in '"\\') or any(
        ord(character) < 0x20 for character in value
    ):
        raise ValueError(
            f"{value!r} cannot be written by this manifest writer. It emits plain "
            "double-quoted strings, so a quote, a backslash or a control character "
            "would end the string early or change what it says."
        )
    return value


def _ids(
    suite: str, entry: Any, name: str, *, optional: bool = False
) -> tuple[str, ...]:
    """Read a suite's list of ids, without coercing what is not a list.

    `tuple` accepts any iterable and a bare string is one, so `checks = "a.one"` would
    declare five ids of one character each and collect five items to match.

    Args:
        suite: the suite, as the message names it.
        entry: its table in the file.
        name: which list.
        optional: whether an absent field reads as an empty list rather than as a
            file this version cannot read.

    Returns:
        The ids.

    Raises:
        ValueError: if the field is absent and not optional, or is not a list.
    """
    if optional and isinstance(entry, dict) and name not in entry:
        return ()
    value = _field(suite, entry, name)
    if not isinstance(value, list):
        raise ValueError(
            f"suite {suite!r} has {name}={value!r}, which is not a list. Converting "
            "whatever arrives is how a bare string becomes one id per character, each "
            "of them collected as a check nobody wrote."
        )
    return tuple(value)


def _field(suite: str, entry: Any, name: str) -> Any:
    """Read a suite's field, or say which suite is missing which.

    Args:
        suite: the suite, as the message names it.
        entry: its table in the file.
        name: the field to read.

    Returns:
        Its value.

    Raises:
        ValueError: if the suite has no such field.
    """
    if not isinstance(entry, dict) or name not in entry:
        raise ValueError(
            f"suite {suite!r} has no {name}. Every field is written when the manifest "
            "is generated, so one missing is a file this version cannot read."
        )
    return entry[name]


def _staleness(difference: Reconciliation, *, layout_only: bool) -> str:
    """Why a manifest is out of date, claiming nothing that was not looked at.

    Args:
        difference: what the declared ids and the reported ones each had alone.
        layout_only: whether the suites were run.

    Returns:
        One line per drifted check, or one line saying where the drift is instead.
    """
    if layout_only:
        return (
            "the file's layout is not what the writer emits. Whether its declared "
            "ids are current is a question --layout-only does not ask."
        )
    # The ids can agree while the file is stale, so saying they agree here reads as a
    # contradiction of the line above it. Name the real reason instead.
    if difference.agrees:
        return "the declared ids are current, and the file's layout is not"
    return str(difference)


def main(argv: Sequence[str] | None = None) -> int:
    """Rewrite a manifest from the suites it names, or say whether it is current.

    Args:
        argv: command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        Zero when the file is current and zero when it was written, since writing
        is the job. ``--check`` returns one on a manifest nobody regenerated, which is
        the convention a formatter's own check mode follows.
    """
    parser = argparse.ArgumentParser(description="Rewrite a warrant check manifest.")
    parser.add_argument("path", type=Path, help="the manifest to rewrite")
    parser.add_argument(
        "--check",
        action="store_true",
        help="report whether it is current, and change nothing",
    )
    parser.add_argument(
        "--layout-only",
        action="store_true",
        help="act on the file's layout alone, running no suite to do it",
    )
    arguments = parser.parse_args(argv)

    # Compared as text rather than as parsed manifests. The declared ids can agree
    # while the file is still stale, because the writer's own layout is part of what is
    # generated: a change to the header or the spacing would otherwise never reach the
    # file, and `--check` would keep passing on a version nothing produces any more.
    current_text = arguments.path.read_text()
    current = Manifest.from_toml(current_text)
    # `--layout-only` answers the half of the question the file already carries. The
    # other half needs the suites, and a runner that reconciles as it runs them has
    # already asked it, so nothing is served by deriving everything twice to ask again.
    rewritten = current.relaid() if arguments.layout_only else current.rewritten()
    wanted_text = rewritten.to_toml()
    difference = current.reconcile(rewritten.checks)
    if wanted_text == current_text:
        scope = "layout current" if arguments.layout_only else "current"
        print(f"{arguments.path}: {scope}, {len(current.checks)} checks declared")
        return 0
    if arguments.check:
        print(f"{arguments.path}: out of date")
        print(_staleness(difference, layout_only=arguments.layout_only))
        return 1
    arguments.path.write_text(wanted_text)
    # A relaid file and a rewritten one are not the same claim. Printing one line for
    # both would let the cheap form report what only the derivation can establish.
    verb = "relaid" if arguments.layout_only else "rewritten"
    print(f"{arguments.path}: {verb}, {len(rewritten.checks)} checks declared")
    print(difference)
    return 0


if __name__ == "__main__":
    sys.exit(main())
