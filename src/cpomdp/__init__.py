"""cpomdp — continuous active inference for Python.

The continuous-state sibling of pymdp. The public API is the stateful
:class:`Agent` façade over a :class:`LinearGaussianModel`, driven in the same
perceive → act loop pymdp users know::

    from cpomdp import Agent, Belief, LinearGaussianModel

    agent = Agent(model, goal=target)
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
from cpomdp.types import Belief, LinearGaussianModel

# Float64 throughout — the oracle matches to 1e-9 and JAX defaults to float32.
# Process-global by necessity; see ADR-004.
jax.config.update("jax_enable_x64", True)

__version__ = "0.2.0"

__all__ = [
    "Agent",
    "Belief",
    "InferenceBackend",
    "KalmanBackend",
    "LinearGaussianModel",
]
