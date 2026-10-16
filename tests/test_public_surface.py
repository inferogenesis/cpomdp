"""The names ADR-069 promotes to the top level, pinned so a dropped export fails.

The strict docs build resolves `cpomdp.LQRController` through the import whether or not
`__all__` lists it, so the API page alone does not catch a name falling off the surface.
"""

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
