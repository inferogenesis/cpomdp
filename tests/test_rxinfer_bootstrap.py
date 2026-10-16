"""RxInfer is declared to juliapkg when the backend first starts Julia, and not before.

Julia is not needed here. A stand-in juliapkg records what it is asked to do, and a
stand-in juliacall records when it is imported. CI's `rxinfer` job runs the real path.
"""

import importlib.machinery
import sys
import types

import pytest

from cpomdp.backends import rxinfer

RXINFER_UUID = "86711068-29c9-4ff7-b620-ae75d7495b3d"


class _JuliaCall(types.ModuleType):
    """A juliacall whose ``Main`` records the import and the code Julia is sent."""

    def __init__(self, calls: list) -> None:
        super().__init__("juliacall")
        self.__spec__ = importlib.machinery.ModuleSpec("juliacall", None)
        self._calls = calls

    def __getattr__(self, name: str):
        if name != "Main":
            raise AttributeError(name)
        self._calls.append(("import juliacall",))
        calls = self._calls

        class Main:
            @staticmethod
            def seval(source: str) -> None:
                calls.append(("seval", source.strip().splitlines()[0]))

        return Main


class _JuliaPkg:
    """A juliapkg over an environment that already declares ``declared``."""

    def __init__(self, declared: dict, calls: list) -> None:
        self.deps = types.SimpleNamespace(load_cur_deps=lambda: declared)
        self._calls = calls

    def require_julia(self, compat: str) -> None:
        self._calls.append(("require_julia", compat))

    def add(self, name: str, uuid: str, **spec: str) -> None:
        self._calls.append(("add", name, uuid, spec))

    def resolve(self) -> None:
        self._calls.append(("resolve",))


def _stand_ins(monkeypatch, declared: dict) -> list:
    calls: list = []
    monkeypatch.setitem(sys.modules, "juliapkg", _JuliaPkg(declared, calls))
    monkeypatch.setitem(sys.modules, "juliacall", _JuliaCall(calls))
    return calls


@pytest.fixture(autouse=True)
def _fresh_julia_cache():
    rxinfer._julia.cache_clear()
    yield
    rxinfer._julia.cache_clear()


def test_a_fresh_environment_gets_rxinfer_as_a_range_before_julia_starts(monkeypatch):
    calls = _stand_ins(monkeypatch, declared={})
    rxinfer._julia()
    assert calls[:4] == [
        ("require_julia", "^1.10"),
        ("add", "RxInfer", RXINFER_UUID, {"version": "5.4"}),
        ("resolve",),
        ("import juliacall",),
    ]
    assert calls[4] == ("seval", "using RxInfer")


def test_an_entry_the_environment_already_has_is_left_alone(monkeypatch):
    declared = {
        "julia": "1.11",
        "packages": {"RxInfer": {"uuid": RXINFER_UUID, "version": "=5.5.0"}},
    }
    calls = _stand_ins(monkeypatch, declared=declared)
    rxinfer._julia()
    assert calls[:2] == [("resolve",), ("import juliacall",)]


def test_julia_starts_once_per_process(monkeypatch):
    calls = _stand_ins(monkeypatch, declared={})
    rxinfer._julia()
    rxinfer._julia()
    assert calls.count(("resolve",)) == 1


def test_without_the_extra_the_error_names_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "juliapkg", None)
    with pytest.raises(ImportError, match="rxinfer"):
        rxinfer._julia()


def test_a_missing_juliacall_is_refused_before_anything_is_declared(monkeypatch):
    calls = _stand_ins(monkeypatch, declared={})
    monkeypatch.setitem(sys.modules, "juliacall", None)
    with pytest.raises(ImportError, match="rxinfer"):
        rxinfer._julia()
    assert calls == []
