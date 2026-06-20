"""cpomdp — continuous active inference for Python.

The continuous-state sibling of pymdp. The public API is the stateful
:class:`Agent` façade over a :class:`LinearGaussianModel`, driven in the same
perceive → act loop pymdp users know::

    from cpomdp import Agent, Belief, LinearGaussianModel, StateGoal

    agent = Agent(model, StateGoal(target))
    belief = agent.infer_states(observation)   # perceive
    action = agent.sample_action()             # act

Swap the inference engine via the ``backend=`` argument; :class:`KalmanBackend`
is the default and :class:`InferenceBackend` is the protocol custom backends
implement. The optional RxInfer oracle lives behind the ``rxinfer`` extra —
import it explicitly from ``cpomdp.backends.rxinfer`` so the core stays
Julia-free.
"""

import jax

from cpomdp.agent import Agent
from cpomdp.backends.base import InferenceBackend
from cpomdp.backends.kalman import KalmanBackend
from cpomdp.dynamics import CallableProcessNoise, DynamicsNoise
from cpomdp.efe import expected_free_energy
from cpomdp.observation import CallableSensor, FixedSensor, ObservationModel
from cpomdp.selection import EFESelector, ObservationGoal, Preference, StateGoal
from cpomdp.structure import ModelStructure
from cpomdp.types import Belief, LinearGaussianModel

# Float64 throughout — the oracle matches to 1e-9 and JAX defaults to float32.
# Process-global by necessity; see ADR-004.
jax.config.update("jax_enable_x64", True)

__version__ = "0.3.0"

__all__ = [
    "Agent",
    "Belief",
    "CallableProcessNoise",
    "CallableSensor",
    "DynamicsNoise",
    "EFESelector",
    "FixedSensor",
    "InferenceBackend",
    "KalmanBackend",
    "LinearGaussianModel",
    "ModelStructure",
    "ObservationGoal",
    "ObservationModel",
    "Preference",
    "StateGoal",
    "expected_free_energy",
]
