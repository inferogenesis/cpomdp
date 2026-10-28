"""Dynamics noise models: state-dependent dynamics noise Q(x).

The dynamics-side twin of ``observation``. ``DynamicsNoiseModel`` is the seam the EFE
predict step asks for the dynamics-noise covariance at a state, as ``ObservationModel``
is for the sensor. ``CallableDynamicsNoise`` is the state-dependent case. Per RFC-001
chapter 8 this is where a *binding* precision constraint lives when it comes from
internal processing rather than the sensor. The kernel evaluates it at ``μ⁺``, the
diffusion of the arrived-at state.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float64, PyTree
from numpy.typing import ArrayLike

__all__ = ["CallableDynamicsNoise", "DynamicsNoiseModel"]


@runtime_checkable
class DynamicsNoiseModel(Protocol):
    """How much the dynamics diffuse per step, as a covariance ``Q(x)`` at each state.

    The model's ``dynamics_noise_model`` slot takes one, as ``observation_model`` takes
    an [`ObservationModel`][cpomdp.ObservationModel]. A constant ``Q`` needs no object.
    It goes in ``dynamics_noise``, and a plain matrix does not satisfy this protocol. A
    state-dependent ``Q(x)`` brings epistemic value back through the *internal* route
    (RFC-001 chapter 8).
    """

    @property
    def is_fixed(self) -> bool:
        """``True`` when ``Q`` does not vary with the state.

        The filter then keeps its fixed path and reads the model's ``dynamics_noise``,
        which equals this object's ``Q``.
        """
        ...

    def noise_at(self, state: ArrayLike, /) -> Float64[Array, "n n"]:
        """The dynamics-noise covariance ``Q`` at ``state``."""
        ...


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class CallableDynamicsNoise:
    """State-dependent dynamics noise, ``Q(state) = noise_fn(state, noise_params)``.

    The dynamics-side twin of [`CallableSensor`][cpomdp.CallableSensor], with the same
    two arguments. ``noise_fn`` holds the functional *form*. It is static aux, since a
    callable cannot be a traced leaf. ``noise_params`` holds the *values* it is a
    function of, as a pytree leaf, so the EFE is differentiable with respect to them.
    A number belongs in ``noise_params`` if it might ever be fitted, and is baked into
    ``noise_fn`` only if it is structural. Pass a module-level ``noise_fn``. A closure
    or lambda hashes by identity and defeats ``jit`` caching.

    ``noise_fn`` must return a positive-semi-definite ``(n, n)`` covariance at every
    reachable state. The model probes it once at construction, where ``n`` is known.

    Attributes:
        noise_fn: ``(state, noise_params) -> Q``, the covariance at one state.
        noise_params: the pytree of values ``noise_fn`` reads.
    """

    noise_fn: Callable[[Float64[Array, "n"], PyTree], Float64[Array, "n n"]]  # aux
    noise_params: PyTree  # leaf
    is_fixed = False

    def __init__(
        self,
        noise_fn: Callable[[Float64[Array, "n"], PyTree], Float64[Array, "n n"]],
        noise_params: PyTree,
    ) -> None:
        object.__setattr__(self, "noise_fn", noise_fn)
        object.__setattr__(self, "noise_params", noise_params)
        self._validate()

    def noise_at(self, state: ArrayLike, /) -> Float64[Array, "n n"]:
        """``Q`` at ``state``, which is read as a float array first."""
        return self.noise_fn(jnp.asarray(state, dtype=float), self.noise_params)

    def tree_flatten(self) -> tuple[tuple[PyTree], Callable]:
        """Traced children ``(noise_params,)``; aux ``noise_fn``."""
        return (self.noise_params,), self.noise_fn

    @classmethod
    def tree_unflatten(
        cls, aux_data: Callable, children: tuple
    ) -> "CallableDynamicsNoise":
        """Rebuild without re-validating — leaves may be tracers."""
        (noise_params,) = children
        obj = object.__new__(cls)
        object.__setattr__(obj, "noise_fn", aux_data)
        object.__setattr__(obj, "noise_params", noise_params)
        return obj

    def _validate(self) -> None:
        # The shape of Q(x) cannot be checked here without n. The model probes
        # dynamics_noise_model.noise_at(zeros(n)) at its own construction.
        if not callable(self.noise_fn):
            raise TypeError(
                f"noise_fn must be callable, got {type(self.noise_fn).__name__}"
            )
