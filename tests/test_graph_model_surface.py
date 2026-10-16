"""The graph model shows a user only what a user builds.

The factor classes and `CouplingGraph` are construction API. The message passing behind
them is typed on a private payload and is the hot path the energy work will change, so
none of it is public (ADR-069). A graph node takes the same sensor a flat model does, so
there is no graph twin of `FixedSensor` or `CallableSensor` to learn (ADR-073).
"""

import importlib.util

import pytest

import cpomdp
from cpomdp.ffg.factors import linear_gaussian


@pytest.mark.parametrize(
    ("owner", "member"),
    [
        (cpomdp.GaussianTransition, "predict"),
        (cpomdp.Coupling, "message_to_parent"),
        (cpomdp.Coupling, "message_to_child"),
        (cpomdp.CouplingGraph, "infer_all"),
    ],
)
def test_a_graph_class_shows_no_message_passing(owner, member):
    assert not hasattr(owner, member)


def test_the_message_payload_lives_in_a_private_module():
    assert importlib.util.find_spec("cpomdp.ffg.message") is None
    assert importlib.util.find_spec("cpomdp.ffg._message") is not None


@pytest.mark.parametrize(
    "name", ["GaussianObservation", "CallableGaussianObservation", "ObservationFactor"]
)
def test_the_graph_has_no_sensor_classes_of_its_own(name):
    assert name not in cpomdp.__all__
    assert not hasattr(cpomdp, name)
    assert not hasattr(linear_gaussian, name)


def test_an_edge_is_one_class():
    # The edge holds its matrix and noise itself, so no factor class wraps them.
    assert "GaussianCoupling" not in cpomdp.__all__
    assert not hasattr(cpomdp, "GaussianCoupling")
    assert not hasattr(linear_gaussian, "GaussianCoupling")
