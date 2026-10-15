"""The graph model shows a user only what a user builds.

The factor classes and `CouplingGraph` are construction API. The message passing behind
them is typed on a private payload and is the hot path the energy work will change, so
none of it is public (ADR-069).
"""

import importlib.util

import pytest

import cpomdp


@pytest.mark.parametrize(
    ("owner", "member"),
    [
        (cpomdp.GaussianTransition, "predict"),
        (cpomdp.GaussianCoupling, "message_to_parent"),
        (cpomdp.GaussianCoupling, "message_to_child"),
        (cpomdp.GaussianObservation, "message"),
        (cpomdp.CallableGaussianObservation, "message"),
        (cpomdp.CouplingGraph, "infer_all"),
    ],
)
def test_a_graph_class_shows_no_message_passing(owner, member):
    assert not hasattr(owner, member)


def test_the_message_payload_lives_in_a_private_module():
    assert importlib.util.find_spec("cpomdp.ffg.message") is None
    assert importlib.util.find_spec("cpomdp.ffg._message") is not None
