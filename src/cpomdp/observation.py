"""Observation models: how a hidden state produces a sensor reading.

The ``ObservationModel`` protocol is the seam the EFE core asks for a local
linear-Gaussian ``(C, R)`` about a state. ``FixedSensor`` is the constant case;
``CallableSensor`` lets the noise covariance follow the state.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float64, PyTree
from numpy.typing import ArrayLike

from cpomdp._validation import validate_covariance

__all__ = ["CallableSensor", "FixedSensor", "ObservationModel"]


def _linear_gaussianize(
    observation_matrix: Float64[Array, "m n"],
    observation_noise: Float64[Array, "m m"],
    x: Float64[Array, "n"],
    sigma: Float64[Array, "n n"],
) -> tuple[Float64[Array, "m"], Float64[Array, "m m"]]:
    """Exact predicted-observation moments for a LINEAR sensor: ``(C·x, C·Σ·Cᵀ + R)``.

    Shared by every linear-mean sensor (``FixedSensor``, ``CallableSensor``) so the
    moment-matching lives in one place. A sensor with a nonlinear mean would supply
    its own second-order ``gaussianize`` instead of calling this.
    """
    return (
        observation_matrix @ x,
        observation_matrix @ sigma @ observation_matrix.T + observation_noise,
    )


@runtime_checkable
class ObservationModel(Protocol):
    """How a hidden state produces an observation, as a local linear-Gaussian map.

    The EFE core never assumes a fixed sensor matrix; it asks the observation
    model to linearize itself about a state ``x``, getting back the local
    ``(C, R)`` (the observation Jacobian and the noise covariance there). For a
    fixed sensor these are constant; for a state-dependent sensor they vary.

    Attributes:
        is_fixed: ``True`` when ``(C, R)`` are constant in the state — the
            fixed-sensor fast path. With fixed process noise as well, EFE's
            epistemic term is constant there and EFE collapses to LQR. ``False``
            for a state-dependent sensor.
    """

    is_fixed: bool

    # "linearize" keeps the z (literature spelling) despite the British prose elsewhere.
    def linearize(
        self, x: ArrayLike
    ) -> tuple[Float64[Array, "m n"], Float64[Array, "m m"]]:
        """Local ``(C, R)`` about state ``x`` — the observation Jacobian and noise."""
        ...

    def gaussianize(
        self, x: ArrayLike, sigma: Float64[Array, "n n"]
    ) -> tuple[Float64[Array, "m"], Float64[Array, "m m"], Float64[Array, "m m"]]:
        """Sensor's EFE ingredients ``(o⁺, S, R)`` about belief ``(x, sigma)``.

        The EFE kernel calls this, not ``linearize``: each sensor owns its own
        moment-matching, so the fixed/linear path stays a bare matvec and a
        sensor with a nonlinear mean could do second order without reopening it.
        Returns the predicted-observation mean ``o⁺``, its covariance ``S`` (feeds
        the pragmatic term), and the conditional observation noise ``R`` at ``x``
        (feeds the epistemic ``½(ln det S − ln det R)``) — all computed in one pass.
        """
        ...


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class FixedSensor:
    """A sensor whose (C, R) never change with state.

    ``linearize`` returns the same stored matrices for every ``x``: a fixed
    linear sensor *is* its own linear approximation everywhere. With fixed process
    noise as well, this is the regime where EFE's epistemic term is constant and EFE
    collapses to LQR (DECISIONS.md ADR-003). A state-dependent ``Q(x)`` moves the
    epistemic term even here (ADR-006).

    Attributes:
        observation_matrix: the observation matrix ``C`` (shape ``m x n``), mapping
            the ``n``-D state to the ``m``-D observation mean.
        observation_noise: the observation-noise covariance ``R`` (shape ``m x m``).
    """

    observation_matrix: Float64[Array, "m n"]  # C
    observation_noise: Float64[Array, "m m"]  # R
    is_fixed = True

    def __init__(
        self, observation_matrix: ArrayLike, *, observation_noise: ArrayLike
    ) -> None:
        object.__setattr__(
            self, "observation_matrix", jnp.asarray(observation_matrix, dtype=float)
        )
        object.__setattr__(
            self, "observation_noise", jnp.asarray(observation_noise, dtype=float)
        )
        self._validate()

    def linearize(
        self, x: ArrayLike
    ) -> tuple[Float64[Array, "m n"], Float64[Array, "m m"]]:
        """Return the stored ``(C, R)`` unchanged — the same for every ``x``."""
        return self.observation_matrix, self.observation_noise

    def gaussianize(
        self, x: ArrayLike, sigma: Float64[Array, "n n"]
    ) -> tuple[Float64[Array, "m"], Float64[Array, "m m"], Float64[Array, "m m"]]:
        """Exact linear ingredients ``(C·x, C·Σ·Cᵀ + R, R)``."""
        o_pred, pred_obs_cov = _linear_gaussianize(
            self.observation_matrix,
            self.observation_noise,
            jnp.asarray(x, dtype=float),
            sigma,
        )
        return o_pred, pred_obs_cov, self.observation_noise

    def tree_flatten(self):
        """Leaves: (observation_matrix, observation_noise); no static aux."""
        return (self.observation_matrix, self.observation_noise), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Rebuild without re-validating — leaves may be tracers."""
        observation_matrix, observation_noise = children
        obj = object.__new__(cls)
        object.__setattr__(obj, "observation_matrix", observation_matrix)
        object.__setattr__(obj, "observation_noise", observation_noise)
        return obj

    def _validate(self) -> None:
        if self.observation_matrix.ndim != 2:
            raise ValueError(
                f"observation_matrix must be a 2-D (m x n) matrix, "
                f"got shape {self.observation_matrix.shape}"
            )
        validate_covariance(
            self.observation_noise, "observation_noise", require_definite=True
        )
        m = self.observation_matrix.shape[0]
        if self.observation_noise.shape != (m, m):
            raise ValueError(
                f"observation_noise must be {m}x{m} to match the {m}-D observation, "
                f"got shape {self.observation_noise.shape}"
            )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class CallableSensor:
    """A sensor with state-dependent observation noise ``R(x)`` and constant ``C``.

    The observation map stays linear (constant ``C``); the noise covariance varies
    with the state via ``noise_fn(x, params) -> R(x)``. This breaks the ADR-003
    fixed-sensor collapse: with ``R`` depending on the predicted state ``μ⁺`` (and
    so on the action), the epistemic term is no longer action-invariant — the agent
    can act to reach states where the sensor is sharper. *Mean-exact,
    covariance-plug-in*: ``o⁺ = C·μ⁺`` is exact, while ``R(μ⁺)`` is a plug-in that
    drops the ``½tr(H_R Σ⁺)`` Jensen term — a deliberate first-order choice.

    ``noise_fn`` must return a **positive-definite** ``R(x)`` at every reachable state
    — it is a covariance the epistemic term inverts. A non-PD ``R(x)`` has no real
    ½ln det, so the EFE epistemic term becomes NaN there (surfaced at action
    selection, not silently wrong); this is the runtime analogue of the
    construction-time positive-definite check on a fixed ``observation_noise``.

    ``params`` is a pytree **leaf** (so EFE is grad-able w.r.t. it — sensor
    learning); ``noise_fn`` is **static aux** (a callable cannot be a traced leaf).
    Pass a *module-level* ``noise_fn`` and keep all tunables in ``params``: a
    closure/lambda is hashable only by identity and would defeat ``jit`` caching.
    """

    observation_matrix: Float64[Array, "m n"]  # C (constant) — leaf
    noise_fn: Callable[[Float64[Array, "n"], PyTree], Float64[Array, "m m"]]  # aux
    noise_params: PyTree  # grad-able sensor parameters — leaf
    is_fixed = False

    def __init__(
        self,
        observation_matrix: ArrayLike,
        noise_fn: Callable[[Float64[Array, "n"], PyTree], Float64[Array, "m m"]],
        noise_params: PyTree,
    ) -> None:
        object.__setattr__(
            self, "observation_matrix", jnp.asarray(observation_matrix, dtype=float)
        )
        object.__setattr__(self, "noise_fn", noise_fn)
        object.__setattr__(self, "noise_params", noise_params)
        self._validate()

    def linearize(
        self, x: ArrayLike
    ) -> tuple[Float64[Array, "m n"], Float64[Array, "m m"]]:
        """Local ``(C, R(x))`` — constant ``C``, state-dependent noise."""
        x = jnp.asarray(x, dtype=float)
        return self.observation_matrix, self.noise_fn(x, self.noise_params)

    def gaussianize(
        self, x: ArrayLike, sigma: Float64[Array, "n n"]
    ) -> tuple[Float64[Array, "m"], Float64[Array, "m m"], Float64[Array, "m m"]]:
        """Linear ingredients ``(C·x, C·Σ·Cᵀ + R(x), R(x))`` (mean-exact, R plug-in)."""
        x = jnp.asarray(x, dtype=float)
        r = self.noise_fn(x, self.noise_params)
        o_pred, pred_obs_cov = _linear_gaussianize(self.observation_matrix, r, x, sigma)
        return o_pred, pred_obs_cov, r

    def tree_flatten(
        self,
    ) -> tuple[tuple[Float64[Array, "m n"], PyTree], Callable]:
        """Traced children ``(observation_matrix, noise_params)``; aux ``noise_fn``."""
        return (self.observation_matrix, self.noise_params), self.noise_fn

    @classmethod
    def tree_unflatten(cls, aux_data: Callable, children: tuple) -> "CallableSensor":
        """Rebuild without re-validating — leaves may be tracers."""
        observation_matrix, noise_params = children
        obj = object.__new__(cls)
        object.__setattr__(obj, "observation_matrix", observation_matrix)
        object.__setattr__(obj, "noise_params", noise_params)
        object.__setattr__(obj, "noise_fn", aux_data)
        return obj

    def _validate(self) -> None:
        if self.observation_matrix.ndim != 2:
            raise ValueError(
                f"observation_matrix must be a 2-D (m x n) matrix, "
                f"got shape {self.observation_matrix.shape}"
            )
        m, n = self.observation_matrix.shape
        # Probe noise_fn once, at the trust boundary, to catch shape bugs early.
        r0 = jnp.asarray(self.noise_fn(jnp.zeros(n), self.noise_params))
        if r0.shape != (m, m):
            raise ValueError(
                f"noise_fn(x, params) must return an (m, m)=({m}, {m}) covariance, "
                f"got shape {r0.shape}"
            )
        validate_covariance(r0, "noise_fn(x, params)", require_definite=True)
