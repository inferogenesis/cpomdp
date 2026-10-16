"""Process-noise models: state-dependent dynamics noise Q(x).

The internal-noise dual of ``observation_model``. ``DynamicsNoise`` is the seam the EFE
predict step asks for the process-noise covariance at a state; ``CallableProcessNoise``
is the state-dependent case. Per RFC-001 chapter 8 this is where a *binding* precision
constraint lives when it comes from internal processing rather than the sensor.
The kernel evaluates it at ``μ⁺`` — the diffusion of the arrived-at state.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax
from jaxtyping import Array, Float64, PyTree
from numpy.typing import ArrayLike

__all__ = ["CallableProcessNoise", "DynamicsNoise"]


@runtime_checkable
class DynamicsNoise(Protocol):
    """How much the dynamics diffuse per step — the process noise covariance Q(x).

    The EFE predict step asks for ``Q`` at a state. ``is_fixed`` flags the constant
    case (a plain matrix); a state-dependent ``Q(x)`` re-introduces epistemic value
    through the *internal* route (RFC-001 Section8).
    """

    is_fixed: bool

    def noise_at(self, x: ArrayLike) -> Float64[Array, "n n"]:
        """The process-noise covariance ``Q(x)`` at state ``x``."""
        ...


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class CallableProcessNoise:
    """State-dependent process noise ``Q(x) = q_fn(x, q_params)``.

    ``q_fn`` holds the functional *form* (static aux — a callable can't be a traced
    leaf); ``q_params`` holds the *values* it is a function of (a pytree leaf, so EFE
    is grad-able w.r.t. them — process-noise learning). The rule for what goes where:
    a number lives in ``q_params`` if you'd ever want to *fit* it; it's baked into
    ``q_fn`` only if it's structural and never learned. Pass a module-level ``q_fn``
    (a closure/lambda hashes by identity and defeats ``jit`` caching).
    """

    q_fn: Callable
    q_params: PyTree
    is_fixed = False

    def __init__(self, q_fn, q_params) -> None:
        object.__setattr__(self, "q_fn", q_fn)
        object.__setattr__(self, "q_params", q_params)
        self._validate()

    def noise_at(self, x: ArrayLike) -> Float64[Array, "n n"]:
        """Q(x) — the process-noise covariance at state x."""
        return self.q_fn(x, self.q_params)

    def tree_flatten(self) -> tuple[tuple[PyTree], Callable]:
        """Children (traced): ``(q_params,)``; aux (static): ``q_fn``."""
        return (self.q_params,), self.q_fn

    @classmethod
    def tree_unflatten(
        cls, aux_data: Callable, children: tuple
    ) -> "CallableProcessNoise":
        """Rebuild without re-validating — leaves may be tracers."""
        (q_params,) = children
        obj = object.__new__(cls)
        object.__setattr__(obj, "q_fn", aux_data)
        object.__setattr__(obj, "q_params", q_params)
        return obj

    def _validate(self) -> None:
        # Shape/PSD of Q(x) can't be checked here — we don't know n. The MODEL probes
        # dynamics_noise_model.noise_at(zeros(n)) at its own construction, where n
        # is known.
        if not callable(self.q_fn):
            raise TypeError(f"q_fn must be callable, got {type(self.q_fn).__name__}")
