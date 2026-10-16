"""What the built wheel carries, read from the wheel itself.

juliapkg reads a ``juliapkg.json`` at the top of every ``sys.path`` entry and in each
directory directly under one, whether or not that package is imported. A wheel that
put one there would declare its Julia packages to every juliacall user in the
environment (ADR-085).
"""

import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def wheel_names(tmp_path_factory) -> list[str]:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("building the wheel needs uv on PATH")
    out_dir = tmp_path_factory.mktemp("wheel")
    build = subprocess.run(
        [uv, "build", "--wheel", "--package", "cpomdp", "--out-dir", str(out_dir)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        pytest.skip(f"the wheel did not build here: {build.stderr.strip()}")
    (wheel,) = out_dir.glob("cpomdp-*.whl")
    with zipfile.ZipFile(wheel) as archive:
        return archive.namelist()


def test_the_wheel_declares_nothing_to_juliapkg(wheel_names):
    reachable = [
        name
        for name in wheel_names
        if Path(name).name == "juliapkg.json" and name.count("/") <= 1
    ]
    assert reachable == []


def test_the_wheel_carries_the_rxinfer_backend(wheel_names):
    assert "cpomdp/backends/rxinfer.py" in wheel_names
