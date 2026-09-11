"""Observation likelihoods evaluated pointwise, as ``log p(y | x)`` at every node.

The seam the exact filter conditions on. It asks for a log-density at each state and
takes no view on the shape of the answer, so a sensor whose noise varies with the
state is the same kind of object here as one whose noise does not. That is what lets
the reference filter be exact where a Gaussian filter cannot be.

Deliberately independent of ``cpomdp.observation``. That protocol answers a different
question, handing back a local linear-Gaussian ``(C, R)`` to a filter that will then
assume the posterior is Gaussian. This one evaluates the density itself. Sharing an
implementation between the two would make the reference agree with the filter it is
supposed to be independent evidence about.

``FixedNoiseLikelihood`` factors its noise once at construction, so the fixed-sensor
path costs one triangular solve per call and no decomposition.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from jaxtyping import Array, Float64, PyTree
from numpy.typing import ArrayLike

from cpomdp._validation import validate_covariance

__all__ = [
    "FixedNoiseLikelihood",
    "GaussianChannel",
    "ObservationLikelihood",
    "StateDependentNoiseLikelihood",
]

_LOG_TWO_PI = math.log(2.0 * math.pi)


@runtime_checkable
class ObservationLikelihood(Protocol):
    """``log p(y | x)`` at an array of states, evaluated in one pass.

    Implementations receive every node at once rather than one state at a time. The
    filter conditions on a whole grid, and a per-state call would put a Python loop
    over hundreds of thousands of nodes in the middle of it.

    Attributes:
        is_fixed: ``True`` when the noise does not vary with the state. The filter
            does not branch on this. It is here so a caller can report which regime a
            result came from, the fixed one being where a Gaussian filter is already
            exact and the gap being measured is zero.
    """

    is_fixed: bool

    def log_likelihood(
        self, observation: ArrayLike, states: ArrayLike
    ) -> Float64[Array, "N"]:
        """``log p(observation | x)`` for each row of ``states``, an ``N x n`` array."""
        ...


@runtime_checkable
class GaussianChannel(ObservationLikelihood, Protocol):
    """A linear-mean Gaussian sensor, read in the parts a Gaussian rung needs.

    The exact filter conditions on ``log_likelihood`` alone. The rule ladder's
    Gaussian rungs build a Kalman-shaped update, and that needs the observation
    matrix and the noise at a state rather than a density. This is the seam that
    asks for them, kept apart so the exact rung is not asked.

    Attributes:
        observation_matrix: the observation matrix ``C`` (shape ``m x n``).
    """

    observation_matrix: Float64[Array, "m n"]

    def observation_noise_at(self, states: ArrayLike) -> Float64[Array, "N m m"]:
        """``R(x)`` at each row of ``states``, one ``m x m`` covariance per state."""
        ...


def _residuals(
    observation: ArrayLike,
    states: ArrayLike,
    observation_matrix: Float64[Array, "m n"],
) -> tuple[Float64[Array, "N m"], Float64[Array, "N n"]]:
    """``y - C x`` at every state, with both shapes checked once.

    The observation check closes the silent-broadcast trap that
    ``cpomdp.backends.base.validate_step_inputs`` closes for the Kalman path. A
    length-1 observation against an m-D prediction would broadcast and give a
    confident density over the wrong thing rather than an error.

    Returns the residuals and the coerced states, this being the one place either
    array crosses from what a caller passed to what the density is built on.
    """
    observation = jnp.asarray(observation, dtype=float)
    states = jnp.asarray(states, dtype=float)
    m, n = observation_matrix.shape
    if observation.shape != (m,):
        raise ValueError(
            f"observation must be a 1-D vector of length {m} (the observation "
            f"dimension), got shape {observation.shape}"
        )
    if states.ndim != 2 or states.shape[-1] != n:
        raise ValueError(
            f"states must be an N x {n} array to match observation_matrix, "
            f"got shape {states.shape}"
        )
    return observation - states @ observation_matrix.T, states


def _gaussian_log_density(
    log_det_noise: Float64[Array, "..."],
    whitened_residuals: Float64[Array, "N m"],
) -> Float64[Array, "N"]:
    """Assemble ``log N`` from the log-determinant and the whitened residuals.

    Whitening is the change of variable that turns a correlated vector into one with
    identity covariance: with ``R = L Lᵀ``, the vector ``L⁻¹r`` has covariance ``I``,
    so the quadratic form ``rᵀ R⁻¹ r`` is that vector's squared norm and no inverse
    is ever formed. Both likelihoods reach this point by different routes, and this
    last step is all they share.
    """
    quadratic = jnp.sum(whitened_residuals**2, axis=-1)
    return -0.5 * (
        whitened_residuals.shape[-1] * _LOG_TWO_PI + log_det_noise + quadratic
    )


def _validate_observation_matrix(observation_matrix: Float64[Array, "m n"]) -> None:
    """Refuse anything that is not a 2-D map from state to observation."""
    if observation_matrix.ndim != 2:
        raise ValueError(
            "observation_matrix must be a 2-D m x n matrix, got shape "
            f"{observation_matrix.shape}"
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class FixedNoiseLikelihood:
    """``log N(y; C x, R)`` with ``R`` constant in the state.

    The regime where the Kalman filter is itself the exact Bayesian filter, which is
    why the grid filter is checked here before it is trusted anywhere else.

    The noise is factored once at construction and the factor is what the per-call
    path uses, leaving one triangular solve and a sum of squares in the loop.

    Attributes:
        observation_matrix: the observation matrix ``C`` (shape ``m x n``).
        observation_noise: the observation-noise covariance ``R`` (shape ``m x m``),
            positive-definite because the density inverts it.
        noise_cholesky: the lower Cholesky factor of ``R``, built at construction.
        log_det_noise: ``log det R``, built at construction from that factor.
    """

    observation_matrix: Float64[Array, "m n"]  # C
    observation_noise: Float64[Array, "m m"]  # R
    noise_cholesky: Float64[Array, "m m"]
    log_det_noise: Float64[Array, ""]
    is_fixed = True

    def __init__(
        self, observation_matrix: ArrayLike, *, observation_noise: ArrayLike
    ) -> None:
        """Store ``(C, R)`` and factor ``R`` for the per-call path.

        Args:
            observation_matrix: the observation matrix ``C``.
            observation_noise: the observation-noise covariance ``R``. Keyword-only,
                so a transposed pair cannot construct in silence.

        Raises:
            ValueError: if ``C`` is not 2-D, or if ``R`` is not a positive-definite
                covariance of the size ``C``'s rows call for.
        """
        object.__setattr__(
            self, "observation_matrix", jnp.asarray(observation_matrix, dtype=float)
        )
        object.__setattr__(
            self, "observation_noise", jnp.asarray(observation_noise, dtype=float)
        )
        self._validate()
        cholesky = jnp.linalg.cholesky(self.observation_noise)
        object.__setattr__(self, "noise_cholesky", cholesky)
        object.__setattr__(
            self, "log_det_noise", 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky)))
        )

    def _validate(self) -> None:
        _validate_observation_matrix(self.observation_matrix)
        validate_covariance(
            self.observation_noise, "observation_noise", require_definite=True
        )
        m = self.observation_matrix.shape[0]
        if self.observation_noise.shape != (m, m):
            raise ValueError(
                f"observation_noise must be {m}x{m} to match the {m} rows of "
                f"observation_matrix, got shape {self.observation_noise.shape}"
            )

    def log_likelihood(
        self, observation: ArrayLike, states: ArrayLike
    ) -> Float64[Array, "N"]:
        """``log N(observation; C x, R)`` at each row of ``states``."""
        residuals, _ = _residuals(observation, states, self.observation_matrix)
        whitened_residuals = solve_triangular(
            self.noise_cholesky, residuals.T, lower=True
        ).T
        return _gaussian_log_density(self.log_det_noise, whitened_residuals)

    def observation_noise_at(self, states: ArrayLike) -> Float64[Array, "N m m"]:
        """``R`` once per row of ``states``, since it does not vary with the state.

        Raises:
            ValueError: if ``states`` is not an ``N x n`` array, the check the
                state-dependent channel makes through its residuals.
        """
        states = jnp.asarray(states, dtype=float)
        n = self.observation_matrix.shape[1]
        if states.ndim != 2 or states.shape[1] != n:
            raise ValueError(
                f"states must be a 2-D array of shape (N, {n}), got shape "
                f"{states.shape}"
            )
        return jnp.broadcast_to(
            self.observation_noise, (states.shape[0], *self.observation_noise.shape)
        )

    def tree_flatten(
        self,
    ) -> tuple[tuple[Float64[Array, "..."], ...], None]:
        """Leaves: the two matrices and the two values derived from the noise."""
        return (
            self.observation_matrix,
            self.observation_noise,
            self.noise_cholesky,
            self.log_det_noise,
        ), None

    @classmethod
    def tree_unflatten(
        cls, aux_data: None, children: tuple[Float64[Array, "..."], ...]
    ) -> "FixedNoiseLikelihood":
        """Rebuild from leaves without re-validating or re-factoring."""
        likelihood = object.__new__(cls)
        matrix, noise, cholesky, log_det = children
        object.__setattr__(likelihood, "observation_matrix", matrix)
        object.__setattr__(likelihood, "observation_noise", noise)
        object.__setattr__(likelihood, "noise_cholesky", cholesky)
        object.__setattr__(likelihood, "log_det_noise", log_det)
        return likelihood


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class StateDependentNoiseLikelihood:
    """``log N(y; C x, R(x))``, the case no Gaussian filter represents exactly.

    ``R(x)`` enters the density through the log-determinant as well as the quadratic
    form, and neither is quadratic in ``x``. The posterior is therefore not Gaussian.
    It can be skewed, and it can be multi-modal, which is the object the exact filter
    exists to reach.

    Nothing is factored ahead of time. The noise is a different matrix at every node,
    so the decomposition is per-call by construction. This path is the expensive one
    on purpose, and it is not on any agent's hot path.

    Attributes:
        observation_matrix: the observation matrix ``C`` (shape ``m x n``).
        observation_noise_fn: called as ``fn(states, params)``, returning one
            ``m x m`` covariance per state, shape ``(N, m, m)``. Static aux data,
            since a callable cannot be a traced leaf. Pass a module-level function; a
            closure or a lambda hashes by identity and defeats ``jit`` caching.
        observation_noise_params: the values ``observation_noise_fn`` is a function
            of, carried as a pytree leaf so the density is differentiable in them.
            The split follows ``cpomdp.dynamics.CallableProcessNoise``: a number lives
            in the params if it would ever be swept, and in the function only if it is
            structural.
    """

    observation_matrix: Float64[Array, "m n"]  # C
    observation_noise_fn: Callable
    observation_noise_params: PyTree
    is_fixed = False

    def __init__(
        self,
        observation_matrix: ArrayLike,
        *,
        observation_noise_fn: Callable,
        observation_noise_params: PyTree = None,
    ) -> None:
        """Store ``C`` and the noise function the density will be built with.

        Args:
            observation_matrix: the observation matrix ``C``.
            observation_noise_fn: ``fn(states, params) -> (N, m, m)``.
            observation_noise_params: whatever ``observation_noise_fn`` reads.

        Raises:
            ValueError: if ``C`` is not a 2-D matrix.
            TypeError: if ``observation_noise_fn`` is not callable.
        """
        object.__setattr__(
            self, "observation_matrix", jnp.asarray(observation_matrix, dtype=float)
        )
        object.__setattr__(self, "observation_noise_fn", observation_noise_fn)
        object.__setattr__(self, "observation_noise_params", observation_noise_params)
        self._validate()

    def _validate(self) -> None:
        # R(x)'s shape and definiteness cannot be checked here: the states it is a
        # function of are not known until a call. The Cholesky raises on the first
        # call instead, which at least fails on the nodes actually being used.
        _validate_observation_matrix(self.observation_matrix)
        if not callable(self.observation_noise_fn):
            raise TypeError(
                "observation_noise_fn must be callable, got "
                f"{type(self.observation_noise_fn).__name__}"
            )

    def log_likelihood(
        self, observation: ArrayLike, states: ArrayLike
    ) -> Float64[Array, "N"]:
        """``log N(observation; C x, R(x))`` at each row of ``states``.

        Raises:
            ValueError: if the noise function does not return one ``m x m`` matrix
                per state, from ``observation_noise_at``.
        """
        residuals, states = _residuals(observation, states, self.observation_matrix)
        cholesky = jnp.linalg.cholesky(self.observation_noise_at(states))
        whitened_residuals = jnp.squeeze(
            solve_triangular(cholesky, residuals[..., None], lower=True), axis=-1
        )
        log_det = 2.0 * jnp.sum(
            jnp.log(jnp.diagonal(cholesky, axis1=-2, axis2=-1)), axis=-1
        )
        return _gaussian_log_density(log_det, whitened_residuals)

    def observation_noise_at(self, states: ArrayLike) -> Float64[Array, "N m m"]:
        """``R(x)`` at each row of ``states``, from the noise function.

        Raises:
            ValueError: if the noise function does not return one ``m x m`` matrix
                per state. A wrong shape here broadcasts rather than failing, and
                would give a plausible density built on the wrong noise.
        """
        states = jnp.asarray(states, dtype=float)
        noise = jnp.asarray(
            self.observation_noise_fn(states, self.observation_noise_params),
            dtype=float,
        )
        m = self.observation_matrix.shape[0]
        expected = (states.shape[0], m, m)
        if noise.shape != expected:
            raise ValueError(
                "observation_noise_fn must return one covariance per state, shape "
                f"{expected}, got shape {noise.shape}"
            )
        return noise

    def tree_flatten(self) -> tuple[tuple[Float64[Array, "m n"], PyTree], Callable]:
        """Children (traced): ``(C, params)``. Aux (static): the noise function."""
        return (
            self.observation_matrix,
            self.observation_noise_params,
        ), self.observation_noise_fn

    @classmethod
    def tree_unflatten(
        cls, aux_data: Callable, children: tuple[Float64[Array, "m n"], PyTree]
    ) -> "StateDependentNoiseLikelihood":
        """Rebuild without re-validating — the leaves may be tracers."""
        likelihood = object.__new__(cls)
        observation_matrix, params = children
        object.__setattr__(likelihood, "observation_matrix", observation_matrix)
        object.__setattr__(likelihood, "observation_noise_fn", aux_data)
        object.__setattr__(likelihood, "observation_noise_params", params)
        return likelihood
