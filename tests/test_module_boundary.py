import ast
from pathlib import Path

import pytest

# The seam PR-3 built. Paper 3 reuses it and does not import the three-term evaluator,
# so neither may these, however many modules deep the route runs.
SEAM = ("_lab.harness", "_lab.constructors")

# The evaluator's module, named here before it exists. A test that waits for the
# import to be possible is a test that arrives after the import has been written.
EVALUATOR = "_lab.scoring"

# The reference filter PR-7 builds. It is the independent object the Kalman path is
# compared against, so it may not reach the thing it is a reference for. The
# comparison lives in a test; a reference that imported the filter under test would
# agree with it for reasons that are not evidence.
REFERENCE = "_lab.reference"

# The tree's shared construction-time checker. The reference filter is allowed to
# reach this one module and nothing else first-party, which the two tests below state
# and justify together.
SHARED_VALIDATOR = "_validation"

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "cpomdp"


def _path_of(module: str) -> Path | None:
    """The file backing a module named relative to ``cpomdp``, if there is one."""
    for candidate in (
        PACKAGE.joinpath(*module.split(".")).with_suffix(".py"),
        PACKAGE.joinpath(*module.split("."), "__init__.py"),
    ):
        if candidate.is_file():
            return candidate
    return None


def _imports_of(path: Path) -> set[str]:
    """The cpomdp submodules a file imports, named relative to the package.

    Covers the three forms the tree uses: ``import cpomdp.x``, ``from cpomdp.x import
    y``, and ``from cpomdp import x``. A relative import would be missed, and none
    exists here — ``test_the_tree_imports_absolutely`` is what keeps that true.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found |= {
                alias.name.removeprefix("cpomdp.")
                for alias in node.names
                if alias.name.startswith("cpomdp.")
            }
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module == "cpomdp":
                found |= {alias.name for alias in node.names}
            elif node.module.startswith("cpomdp."):
                found.add(node.module.removeprefix("cpomdp."))
    return found


def _closure(*modules: str) -> set[str]:
    """Every cpomdp module reachable from ``modules`` by following imports.

    ``cpomdp/__init__.py`` is not followed. It re-exports the public surface, so
    following it would make every module depend on every other and the boundary would
    be untestable rather than unbroken.
    """
    seen: set[str] = set()
    queue = list(modules)
    while queue:
        module = queue.pop()
        if module in seen or module == "":
            continue
        seen.add(module)
        path = _path_of(module)
        if path is not None:
            queue.extend(_imports_of(path))
    return seen


# --- the walk finds things, before it is asked to find nothing ----------------------


def test_the_walk_finds_a_direct_import():
    assert "types" in _closure("_lab.harness")


def test_the_walk_follows_an_import_of_an_import():
    # `structure` is the second hop. harness imports cpomdp.types, and types is what
    # names structure. harness does not, so finding it proves the walk carried on past
    # the imports written in the file it started from. The pair is asserted together:
    # if harness ever imports structure directly, the first line fails rather than the
    # second quietly becoming a re-test of a direct import.
    harness = _path_of("_lab.harness")
    assert harness is not None
    assert "structure" not in _imports_of(harness)
    assert "structure" in _closure("_lab.harness")


def test_the_walk_stops_where_the_first_party_imports_do():
    # The shared validators import nothing first-party, and cpomdp.types imports them,
    # so they cannot import back. A walk that over-collected would not come back with
    # one name. The subject differs from the leaf claim below, so a broken walk and a
    # broken boundary fail apart.
    assert _closure("_validation") == {"_validation"}


def test_a_module_outside_the_seam_is_not_dragged_in():
    assert "ffg._chain" not in _closure("_lab.constructors")


# --- the boundary ------------------------------------------------------------------


@pytest.mark.parametrize("module", SEAM)
def test_the_seam_does_not_reach_the_evaluator(module):
    assert EVALUATOR not in _closure(module)


def test_the_seam_does_not_reach_an_action_selector():
    # A ScoredAgent is driven, never choosing. This is a source-level claim about what
    # the seam depends on, not about what a process has loaded: `cpomdp/__init__.py`
    # imports the selectors for its public surface, so `import cpomdp._lab.harness`
    # executes them either way.
    assert "selection" not in _closure(*SEAM)
    assert "enumeration" not in _closure(*SEAM)


def _reference_modules() -> set[str]:
    """Every module under ``cpomdp/_lab/reference/``, named relative to the package.

    Discovered rather than listed, so a module added to the subpackage is inside the
    boundary from the moment it exists.
    """
    root = PACKAGE.joinpath(*REFERENCE.split("."))
    return {
        ".".join(path.relative_to(PACKAGE).with_suffix("").parts).removesuffix(
            ".__init__"
        )
        for path in root.rglob("*.py")
    }


def test_the_walk_finds_the_reference_subpackage():
    assert "_lab.reference.quadrature" in _reference_modules()


@pytest.mark.parametrize(
    "forbidden",
    ["backends.kalman", "backends.base", "selection", "enumeration", EVALUATOR],
)
def test_the_reference_filter_does_not_reach_the_filter_it_references(forbidden):
    assert forbidden not in _closure(*_reference_modules())


def test_validation_is_a_leaf_and_so_carries_nothing_in():
    # What makes the allowance below safe rather than a hole. `_validation` is the
    # tree's shared trust-boundary checker and imports no first-party module, so
    # reaching it couples the reference to nothing. If it ever grows an import, the
    # allowance stops being harmless and this fails before that one does.
    assert _closure(SHARED_VALIDATOR) == {SHARED_VALIDATOR}


def test_the_reference_filter_reaches_nothing_first_party_beyond_the_validator():
    # Stronger than the list above and cheaper to keep true. The moment the reference
    # needs a model type that is a design change, and it should read as one in the
    # diff rather than passing quietly.
    #
    # `_validation` is the one exception. Duplicating a positive-definiteness check
    # to keep a closure empty would trade a real invariant for a cosmetic one, and it
    # carries nothing in — see the test above.
    allowed = _reference_modules() | {SHARED_VALIDATOR}
    assert _closure(*_reference_modules()) <= allowed


def test_the_tree_imports_absolutely():
    relative = [
        path.relative_to(PACKAGE)
        for path in PACKAGE.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.ImportFrom) and node.level > 0
    ]
    assert not relative


def test_the_shared_propagation_is_a_leaf():
    # The error on a difference is built once for the evaluator and for what may not
    # import it. It reaches no first-party module, so any side can take it without
    # taking the evaluator along.
    assert _closure("_lab.resolution") == {"_lab.resolution"}
