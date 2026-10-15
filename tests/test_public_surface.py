"""The names ADR-069 promotes to the top level, and one it keeps private, pinned.

The strict docs build resolves `cpomdp.LQRController` through the import whether or not
`__all__` lists it, so the API page alone does not catch a name falling off the surface.
"""

import importlib

import pytest

import cpomdp
from cpomdp.backends.base import InadmissiblePartitionError
from cpomdp.control import LQRController


@pytest.mark.parametrize(
    ("name", "obj"),
    [
        ("LQRController", LQRController),
        ("InadmissiblePartitionError", InadmissiblePartitionError),
    ],
)
def test_a_promoted_name_is_exported_at_the_top_level(name, obj):
    assert name in cpomdp.__all__
    assert getattr(cpomdp, name) is obj


def test_the_chain_backend_is_private():
    # It gives KalmanBackend's numbers at two extra inversions per step, and exists to
    # show the message algebra is the Kalman filter on a chain (ADR-012).
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("cpomdp.ffg.chain")
