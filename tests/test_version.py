"""One version, read where a user meets it: the package, its metadata and the citation.

`pyproject.toml` holds it. `cpomdp.__version__` reads the installed metadata, which
the bug template asks for, and `CITATION.cff` restates it by hand for GitHub and Zenodo.
"""

import re
import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path

import cpomdp

ROOT = Path(__file__).resolve().parents[1]


def _project_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def test_the_package_reports_its_installed_version():
    assert cpomdp.__version__ == metadata.version("cpomdp") == _project_version()


def test_the_citation_names_the_project_version():
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    (cited,) = re.findall(r"^version:\s*\"?([^\"\s]+)\"?\s*$", citation, re.MULTILINE)
    assert cited == _project_version()


def test_a_copy_without_metadata_still_imports():
    script = (
        "from importlib import metadata\n"
        "installed = metadata.version\n"
        "def unmetered(name):\n"
        "    if name == 'cpomdp':\n"
        "        raise metadata.PackageNotFoundError(name)\n"
        "    return installed(name)\n"
        "metadata.version = unmetered\n"
        "import cpomdp\n"
        "print(cpomdp.__version__)\n"
    )
    run = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert run.stdout.strip() == "0+unknown"
