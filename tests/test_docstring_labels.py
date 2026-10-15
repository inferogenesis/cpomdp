"""Docstrings outside ``cpomdp._lab`` cite nothing a user of the package cannot reach.

The build plan is reset at every release, so its phase and item labels die with it. The
research battery's standing rules live in ``research/``, which neither the wheel nor the
site carries. ADR and RFC numbers stay: both records are append-only, and the docs link
DECISIONS.md. ``cpomdp._lab`` is private and is read beside the repository.
"""

import ast
import re
from collections.abc import Iterator
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "cpomdp"

_UNREACHABLE = re.compile(
    r"\bPhase[- ]?[0-9][0-9A-Z]*\b"
    r"|\bitem [A-Z]\b"
    r"|\bG-D\b"
    r"|\bPR-[0-9]"
    r"|standing (?:rule|prohibition)"
    r"|BUILD_PLAN|RESEARCH\.md"
)

_DOCUMENTED = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _public_docstrings() -> Iterator[tuple[str, str]]:
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(SRC)
        if relative.parts[0] == "_lab":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, _DOCUMENTED):
                docstring = ast.get_docstring(node, clean=False)
                if docstring:
                    yield f"{relative}:{getattr(node, 'name', '<module>')}", docstring


def test_the_scan_reads_the_public_modules():
    owners = {where.split(":")[0] for where, _ in _public_docstrings()}
    assert "agent.py" in owners
    assert not any(owner.startswith("_lab") for owner in owners)


def test_no_public_docstring_cites_a_build_plan_label():
    cited = [
        f"{where}: {match.group(0)}"
        for where, docstring in _public_docstrings()
        for match in _UNREACHABLE.finditer(docstring)
    ]
    assert cited == []
