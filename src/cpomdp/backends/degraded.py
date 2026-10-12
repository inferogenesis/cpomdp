"""Backends that filter worse than the model they are built for, and say how.

Each wraps or replaces the exact filter for one model and keeps ``model`` pointing at
that same model, so a caller reading ``backend.model`` still gets the model the results
are scored under. ``degradation`` names what the filter does differently.

``WrongFixedRBackend`` plugs a fixed observation noise into the filter, scaled away from
the one the model declares. ``DiagonalCovarianceBackend`` drops the covariance's
off-diagonal terms after every step, so the correlation never reaches the next gain.
"""

import jax.numpy as jnp
from numpy.typing import ArrayLike

from cpomdp.backends.base import InferenceBackend
from cpomdp.backends.kalman import KalmanBackend
from cpomdp.types import Belief, LinearGaussianModel

__all__ = ["DiagonalCovarianceBackend", "WrongFixedRBackend"]


class WrongFixedRBackend:
    """A filter running on an observation noise the model does not declare.

    The filter uses ``observation_noise * (1 + magnitude)``, held fixed. A model
    carrying a state-dependent ``R(x)`` loses it here, so the filter reads a constant
    where the model varies. The constant is the model's nominal ``observation_noise``,
    which is ``R`` at the prior mean when the model was built without one. On such a
    model a magnitude of zero is already a degradation. On a fixed-noise model it
    reproduces the exact filter.

    ``model`` is the model the agent is scored under, unchanged and still carrying its
    own sensor. Only the filter sees the substitution.
    """

    degradation = "WrongFixedR"

    def __init__(self, model: LinearGaussianModel, *, magnitude: float) -> None:
        """Build a filter over ``model`` reading a scaled fixed observation noise.

        Args:
            model: The model being scored. Kept as ``self.model``, untouched.
            magnitude: Relative change to the declared ``observation_noise``, applied
                as ``value * (1 + magnitude)``.

        Raises:
            ValueError: If the scaled noise is not a valid covariance, as it is not at
                a magnitude of ``-1`` or below.
        """
        self.model = model
        self._inner = KalmanBackend(
            LinearGaussianModel(
                dynamics_matrix=model.dynamics_matrix,
                observation_matrix=model.observation_matrix,
                dynamics_noise=model.dynamics_noise,
                observation_noise=model.observation_noise * (1.0 + magnitude),
                prior=model.prior,
                control_matrix=model.control_matrix,
                dynamics_noise_model=model.dynamics_noise_model,
                structure=model.structure,
            )
        )

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
        self.model = backend.model

    def infer_states(
        self,
        observation: ArrayLike,
        prior: Belief,
        action: ArrayLike | None = None,
    ) -> Belief:
        """Advance the belief one step, then drop the covariance's off-diagonals."""
        posterior = self._inner.infer_states(observation, prior, action)
        return Belief(mean=posterior.mean, cov=jnp.diag(jnp.diag(posterior.cov)))
