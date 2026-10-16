"""Backends that filter worse than the model they are built for, and say how.

Each wraps or replaces the exact filter for one model and keeps ``model`` pointing at
that same model, so a caller reading ``backend.model`` still gets the model the results
are scored under. ``degradation`` names what the filter does differently.

``WrongFixedRBackend`` plugs a fixed observation noise into the filter, scaled away from
the one the model declares or from one the caller hands it.
``DiagonalCovarianceBackend`` drops the covariance's off-diagonal terms after every
step, so the correlation never reaches the next gain.
"""

import jax.numpy as jnp
from numpy.typing import ArrayLike

from cpomdp.backends.base import InferenceBackend
from cpomdp.backends.kalman import KalmanBackend
from cpomdp.types import Belief, LinearGaussianModel, _refuse_graph_interface

__all__ = ["DiagonalCovarianceBackend", "WrongFixedRBackend"]


class WrongFixedRBackend:
    """A filter running on an observation noise the model does not declare.

    The filter uses a fixed ``R`` scaled by ``1 + magnitude``. That ``R`` is the
    model's ``observation_noise`` unless the caller passes one, so a constant taken at
    another operating point is this backend's argument rather than a value on the model.
    A model carrying a state-dependent ``R(x)`` loses it here, so the filter reads a
    constant where the model varies. On such a model a magnitude of zero is already a
    degradation. On a fixed-noise model it reproduces the exact filter.

    ``model`` is the model the agent is scored under, unchanged and still carrying its
    own sensor. Only the filter sees the substitution.
    """

    degradation = "WrongFixedR"

    def __init__(
        self,
        model: LinearGaussianModel,
        *,
        magnitude: float,
        observation_noise: ArrayLike | None = None,
    ) -> None:
        """Build a filter over ``model`` reading a scaled fixed observation noise.

        Args:
            model: The model being scored. Read back as ``model``, untouched.
            magnitude: Relative change to the fixed noise, applied as
                ``value * (1 + magnitude)``.
            observation_noise: The fixed noise ``R`` to scale. Defaults to the model's
                ``observation_noise``, its ``R`` at the prior mean beside a sensor.

        Raises:
            ValueError: If the scaled noise is not a valid covariance, as it is not at
                a magnitude of ``-1`` or below.
        """
        _refuse_graph_interface(model, "WrongFixedRBackend")
        fixed_noise = (
            model.observation_noise
            if observation_noise is None
            else jnp.asarray(observation_noise, dtype=float)
        )
        self._model = model
        self._inner = KalmanBackend(
            LinearGaussianModel(
                dynamics_matrix=model.dynamics_matrix,
                observation_matrix=model.observation_matrix,
                dynamics_noise=model.dynamics_noise,
                observation_noise=fixed_noise * (1.0 + magnitude),
                prior=model.prior,
                control_matrix=model.control_matrix,
                dynamics_noise_model=model.dynamics_noise_model,
                structure=model.structure,
            )
        )

    @property
    def model(self) -> LinearGaussianModel:
        """The model being scored, fixed at construction."""
        return self._model

    def infer_states(
        self,
        observation: ArrayLike,
        prior: Belief,
        action: ArrayLike | None = None,
    ) -> Belief:
        """Advance the belief one step under the substituted observation noise."""
        return self._inner.infer_states(observation, prior, action)


class DiagonalCovarianceBackend:
    """A filter that keeps only the diagonal of every posterior covariance.

    Wraps another backend and zeroes the off-diagonal terms of what it returns. The
    dropped correlation does not reach the next step's prior, so the loss accumulates
    rather than being reapplied from scratch each time.

    ``model`` is the wrapped backend's model, so the wrapper is transparent to anything
    reading which model the results are scored under.
    """

    degradation = "DiagonalCovarianceOnly"

    def __init__(self, backend: InferenceBackend) -> None:
        """Wrap ``backend``, keeping the model it was built from.

        Args:
            backend: The filter to degrade. Any inference backend, including one that
                is itself degraded.
        """
        self._inner = backend

    @property
    def model(self) -> LinearGaussianModel:
        """The wrapped backend's model."""
        return self._inner.model

    def infer_states(
        self,
        observation: ArrayLike,
        prior: Belief,
        action: ArrayLike | None = None,
    ) -> Belief:
        """Advance the belief one step, then drop the covariance's off-diagonals."""
        posterior = self._inner.infer_states(observation, prior, action)
        return Belief(mean=posterior.mean, cov=jnp.diag(jnp.diag(posterior.cov)))
