"""cpomdp — continuous active inference for Python.

The continuous-state sibling of pymdp. The entry point is the stateful
``Agent`` façade over a ``LinearGaussianModel``, driven in the same
perceive → act loop pymdp users know::

    from cpomdp import Agent, Belief, LinearGaussianModel, StateGoal

    agent = Agent(model, StateGoal(target))
    belief = agent.infer_states(observation)   # perceive
    action = agent.sample_action()             # act

``KalmanBackend`` is the default inference engine, and ``Agent.from_backend`` swaps in
another. ``InferenceBackend`` is the protocol a custom backend implements. The optional
RxInfer oracle needs the ``rxinfer`` extra. Import it from ``cpomdp.backends.rxinfer``,
which keeps the core free of Julia.

For a *branching* model, declare a ``CouplingGraph`` whose nodes carry
observation factors (``GaussianObservation``,
``CallableGaussianObservation``) and whose ``Coupling`` edges carry a
``GaussianCoupling``, then run it through a ``CouplingGraphBackend``
with a per-node ``GaussianTransition``. A state-dependent ``R(x)`` on a
coupled node cannot be flattened to a fixed linear-Gaussian model, and asking it to
raises ``IncompatibleLinearizationError`` (ADR-019, ADR-020). For such a backend the
``Agent`` dispatches an ``FfgEfeSelector``, the FFG peer of ``EFESelector``. Pass
``selector=`` to override it.

Whether a state-dependent sensor earns its keep is a question about the states an
action can actually reach: ``probe_model`` samples that reachable set and
reports back a ``SensorReport``.

Importing the package sets ``jax_enable_x64`` for the whole process, so every JAX array
defaults to float64. The filter is checked to 1e-9 against its oracles, which float32
cannot hold (ADR-004).
"""

from importlib import metadata as _metadata

import jax as _jax

from cpomdp.agent import Agent
from cpomdp.backends.base import InadmissiblePartitionError, InferenceBackend
from cpomdp.backends.coupling import (
    CouplingGraphBackend,
    IncompatibleLinearizationError,
)
from cpomdp.backends.kalman import KalmanBackend
from cpomdp.control import LQRController
from cpomdp.diagnostics import SensorReport, probe_model
from cpomdp.dynamics import CallableProcessNoise, DynamicsNoise
from cpomdp.efe import expected_free_energy
from cpomdp.ffg.factors.linear_gaussian import (
    CallableGaussianObservation,
    GaussianCoupling,
    GaussianObservation,
    GaussianTransition,
)
from cpomdp.ffg.graph import Coupling, CouplingGraph
from cpomdp.observation import CallableSensor, FixedSensor, ObservationModel
from cpomdp.selection import (
    ActionSelector,
    EFESelector,
    FfgEfeSelector,
    LQRSelector,
    ObservationGoal,
    Preference,
    StateGoal,
)
from cpomdp.structure import ModelStructure
from cpomdp.types import Belief, LinearGaussianModel
from warrantlib import Warrant

# Float64 throughout — the oracle matches to 1e-9 and JAX defaults to float32.
# Process-global by necessity; see ADR-004.
_jax.config.update("jax_enable_x64", True)

try:
    __version__ = _metadata.version("cpomdp")
except _metadata.PackageNotFoundError:
    __version__ = "0+unknown"
"""The installed version, read from the package metadata `pyproject.toml` sets.

A copy imported without being installed has no metadata and reads ``0+unknown``.
"""

__all__ = [
    "ActionSelector",
    "Agent",
    "Belief",
    "CallableGaussianObservation",
    "CallableProcessNoise",
    "CallableSensor",
    "Coupling",
    "CouplingGraph",
    "CouplingGraphBackend",
    "DynamicsNoise",
    "EFESelector",
    "FfgEfeSelector",
    "FixedSensor",
    "GaussianCoupling",
    "GaussianObservation",
    "GaussianTransition",
    "InadmissiblePartitionError",
    "IncompatibleLinearizationError",
    "InferenceBackend",
    "KalmanBackend",
    "LQRController",
    "LQRSelector",
    "LinearGaussianModel",
    "ModelStructure",
    "ObservationGoal",
    "ObservationModel",
    "Preference",
    "SensorReport",
    "StateGoal",
    "Warrant",
    "expected_free_energy",
    "probe_model",
]
