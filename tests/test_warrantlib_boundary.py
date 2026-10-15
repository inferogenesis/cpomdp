"""`warrantlib` stands alone, and what cpomdp re-exports from it is warrantlib's own.

The vocabulary is published as its own distribution with no dependencies, so a check
suite can label its findings without a numerical stack. That property is invisible in
ordinary use: every test in this repository imports cpomdp anyway, so a fresh import of
`cpomdp` inside `warrantlib` would go unnoticed until someone installed it on its own.
The first test below runs the import in a clean interpreter and fails if any `cpomdp`
module is reachable afterwards.

The rest pin the re-exports. cpomdp's top level and `cpomdp.enumeration` name objects
that are defined in `warrantlib`, and a copy rather than a re-export would break every
`isinstance` check that crosses the two.

One more pins the façade. The definitions live in `warrantlib._vocabulary` and
`warrantlib._serialise`, and `warrantlib/__init__.py` re-exports them by hand, so a new
public name added to one and forgotten in the other is invisible: it imports fine from
its own module and is absent from the published surface. The source is read rather than
the imported module, because the imported module cannot tell a definition from an
import.

`warrantlib.pytest_plugin` is outside that rule, also deliberately. Its public names
are pytest hooks and a fixture, which pytest finds through the entry point, and putting
them in `__all__` would publish an import path nobody should use.

cpomdp re-exports `Warrant` alone, the type its selectors and searches report. The
rest of the vocabulary is imported from warrantlib, its one home (ADR-069).
"""

import ast
import subprocess
import sys
from pathlib import Path

import cpomdp
import warrantlib
from cpomdp.enumeration import CompletenessCertificate, SearchWarrant
from warrantlib import _serialise, _vocabulary

_NO_CPOMDP = """
import sys
import warrantlib

leaked = sorted(name for name in sys.modules if name.split(".")[0] == "cpomdp")
if leaked:
    raise SystemExit("warrantlib pulled in " + ", ".join(leaked))
"""

_NO_PYTEST = """
import sys
import warrantlib

if "pytest" in sys.modules:
    raise SystemExit("warrantlib pulled in pytest")
"""


def test_warrantlib_imports_nothing_from_cpomdp():
    completed = subprocess.run(
        [sys.executable, "-c", _NO_CPOMDP], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_warrantlib_imports_no_pytest():
    # The plugin module imports pytest and is reached through the `pytest11` entry
    # point, never from `__init__`. An import added there would put pytest on the
    # dependency list of a distribution whose list is empty on purpose (ADR-039), and
    # nothing else in the suite would notice, because pytest is always installed here.
    completed = subprocess.run(
        [sys.executable, "-c", _NO_PYTEST], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_the_top_level_warrant_is_the_warrantlib_object():
    assert cpomdp.Warrant is warrantlib.Warrant


def test_enumeration_re_exports_the_certificate_and_the_alias():
    assert CompletenessCertificate is warrantlib.CompletenessCertificate
    assert SearchWarrant is warrantlib.Warrant


def _defined_public_names() -> set[str]:
    """Top-level public names the private modules define, read from their source."""
    names = set()
    sources = (Path(_vocabulary.__file__), Path(_serialise.__file__))
    for node in [n for path in sources for n in ast.parse(path.read_text()).body]:
        if isinstance(node, ast.ClassDef | ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
    return {name for name in names if not name.startswith("_")}


def test_the_facade_exports_every_public_vocabulary_name():
    defined = _defined_public_names()
    # A guard on the guard: an empty set would pass the comparison below by asking
    # nothing, exactly as a renamed module would.
    assert len(defined) >= 9
    assert defined == set(warrantlib.__all__)
