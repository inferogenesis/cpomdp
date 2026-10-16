"""The names the top level promotes or renames, pinned so a dropped export fails.

The strict docs build resolves `cpomdp.LQRController` through the import whether or not
`__all__` lists it, so the API page alone does not catch a name falling off the surface.
"""

import pytest

import cpomdp
from cpomdp.backends.base import InadmissiblePartitionError
from cpomdp.control import LQRController
from cpomdp.dynamics import CallableDynamicsNoise, DynamicsNoiseModel


@pytest.mark.parametrize(
    ("name", "obj"),
    [
        ("LQRController", LQRController),
        ("InadmissiblePartitionError", InadmissiblePartitionError),
        ("DynamicsNoiseModel", DynamicsNoiseModel),
        ("CallableDynamicsNoise", CallableDynamicsNoise),
    ],
)
def test_a_promoted_name_is_exported_at_the_top_level(name, obj):
    assert name in cpomdp.__all__
    assert getattr(cpomdp, name) is obj


@pytest.mark.parametrize("name", ["DynamicsNoise", "CallableProcessNoise"])
def test_a_renamed_name_is_gone_from_the_top_level(name):
    assert name not in cpomdp.__all__
    assert not hasattr(cpomdp, name)
