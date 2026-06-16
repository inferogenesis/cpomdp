"""Exact Kalman-filter inference backend (per-step and steady-state modes)."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float64
from numpy.typing import ArrayLike

from cpomdp.backends.base import validate_step_inputs
from cpomdp.types import Belief, LinearGaussianModel

__all__ = ["KalmanBackend"]


@jax.jit
def _gain_and_posterior_cov(
    dynamics: Float64[Array, "n n"],
    sensor_model: Float64[Array, "m n"],
    dynamics_noise: Float64[Array, "n n"],
    sensor_noise: Float64[Array, "m m"],
    prior_cov: Float64[Array, "n n"],
) -> tuple[Float64[Array, "n m"], Float64[Array, "n n"]]:
    """Run one covariance recursion: prior covariance in, ``(gain, cov_post)`` out.

    This is the covariance half of the Kalman step — predict and update applied to
    the *uncertainty* rather than the mean::

        cov_pred = A · prior_cov · Aᵀ + Q       # predict: dynamics inflate it
        S        = C · cov_pred · Cᵀ + R        # prediction-error covariance
        gain     = cov_pred · Cᵀ · S⁻¹          # how far to trust the reading
        cov_post = (I − gain · C) · cov_pred    # update: the reading shrinks it

    (A=dynamics, C=sensor_model, Q=dynamics_noise, R=sensor_noise.) The gain is
    obtained via ``jnp.linalg.solve`` against ``S`` rather than an explicit inverse,
    for numerical stability. Crucially this depends only on the model and
    ``prior_cov``, never on an observation — which is exactly what lets the
    steady-state mode precompute it once and the per-step mode recompute it cheaply
    each step. Pure and ``jit``-compiled, so it also drops into ``vmap``/``grad``.

    Args:
        dynamics: The state-transition matrix A, shape ``(n, n)``.
        sensor_model: The observation matrix C, shape ``(m, n)``.
        dynamics_noise: The process-noise covariance Q, shape ``(n, n)``.
        sensor_noise: The observation-noise covariance R, shape ``(m, m)``.
        prior_cov: The incoming belief's covariance, shape ``(n, n)``.

    Returns:
        ``(gain, cov_post)``: the Kalman gain, shape ``(n, m)``, and the posterior
        covariance, shape ``(n, n)``, for this step.
    """
    cov_pred = dynamics @ prior_cov @ dynamics.T + dynamics_noise
    prediction_error_cov = sensor_model @ cov_pred @ sensor_model.T + sensor_noise
    gain = jnp.linalg.solve(prediction_error_cov, sensor_model @ cov_pred).T
    cov_post = (jnp.eye(dynamics.shape[0]) - gain @ sensor_model) @ cov_pred
    return gain, cov_post


@jax.jit
def _posterior_mean(
    dynamics: Float64[Array, "n n"],
    sensor_model: Float64[Array, "m n"],
    prior_mean: Float64[Array, "n"],
    control_term: Float64[Array, "n"],
    gain: Float64[Array, "n m"],
    observation: Float64[Array, "m"],
) -> Float64[Array, "n"]:
    """The mean half of the Kalman step: predict the mean, then correct it.

    Steps the prior mean through the dynamics (adding the pre-computed
    ``control_term``), then nudges it toward ``observation`` by the gain times the
    prediction error (the "innovation"). Pure and ``jit``-compiled.
    """
    mean_pred = dynamics @ prior_mean + control_term
    prediction_error = observation - sensor_model @ mean_pred
    return mean_pred + gain @ prediction_error


class KalmanBackend:
    """Exact Kalman-filter inference for a LinearGaussianModel.

    Implements the ``InferenceBackend`` protocol: constructed from a model,
    then advances a belief one step at a time (prior in, posterior out) via
    the standard predict/update recursion.

    Two modes:

    - **Per-step (default):** recomputes the Kalman gain and covariance every
      step from the incoming belief. Correct for any linear-Gaussian model,
      including transient (pre-convergence) behaviour. This is the analytic
      oracle the rest of the toolbox is validated against.
    - **Steady-state (``steady_state=True``):** solves the covariance recursion
      *once* at construction to a fixed point, then reuses the frozen gain and
      covariance every step. Cheap (no per-step covariance maths), but only
      valid for time-invariant models with regular complete observations.
      Raises ``RuntimeError`` if the recursion does not converge within ``max_iter``
      (i.e. the model is not stabilisable/detectable).

    Args:
        model: The linear-Gaussian generative model to filter under.
        steady_state: If True, precompute and freeze the steady-state gain.
        tol: Convergence tolerance for the steady-state fixed point (absolute,
            on successive covariances).
        max_iter: Cap on steady-state iterations before giving up.
    """

    def __init__(
        self,
        model: LinearGaussianModel,
        *,
        steady_state: bool = False,
        tol: float = 1e-12,
        max_iter: int = 1000,
    ) -> None:
        self.model = model
        self.steady_state = steady_state
        if steady_state:
            self._steady_gain, self._steady_cov = self._converge_to_steady_state(
                tol, max_iter
            )

    def infer_states(
        self,
        observation: ArrayLike,
        prior: Belief,
        action: ArrayLike | None = None,
    ) -> Belief:
        """Advance the belief by one filter step.

        Runs one predict/update cycle: step the prior through the dynamics
        (applying ``action`` if the model has a control matrix), then correct the
        prediction toward ``observation`` using the Kalman gain. In steady-state
        mode the gain and covariance are the frozen fixed-point values; otherwise
        they are recomputed from ``prior.cov`` on this step.

        The numeric work is delegated to the ``jit``-compiled module kernels
        (``_gain_and_posterior_cov``, ``_posterior_mean``); this method stays the
        eager orchestrator that validates inputs and wraps the result in a
        ``Belief``.

        Args:
            observation: The latest sensor reading, shape ``(m,)``.
            prior: The current belief, treated as this step's previous posterior.
                Never mutated.
            action: The action just taken, shape ``(p,)``. Required iff the model
                has a control matrix; ignored (pass ``None``) for pure filtering.

        Returns:
            The posterior belief — a new ``Belief``; the prior is left untouched.

        Raises:
            ValueError: If ``observation`` is not shape ``(m,)``, ``prior`` is not
                a belief over the model's ``n``-D state, the model has a control
                matrix but ``action`` is ``None``, or ``action`` is not shape
                ``(p,)``. (All enforced in ``_validate_inputs``.)
        """
        model = self.model
        observation, action = validate_step_inputs(model, observation, prior, action)
        control = model.control
        if control is None:
            control_term = jnp.zeros(model.n_states)
        else:
            # validate_step_inputs guarantees a non-None action when control exists
            assert action is not None
            control_term = control @ action

        if self.steady_state:
            gain, cov_post = self._steady_gain, self._steady_cov  # frozen
        else:
            gain, cov_post = _gain_and_posterior_cov(
                model.dynamics,
                model.sensor_model,
                model.dynamics_noise,
                model.sensor_noise,
                prior.cov,
            )

        mean_post = _posterior_mean(
            model.dynamics,
            model.sensor_model,
            prior.mean,
            control_term,
            gain,
            observation,
        )

        return Belief(mean=mean_post, cov=cov_post)

    def _converge_to_steady_state(
        self, tol: float, max_iter: int
    ) -> tuple[Float64[Array, "n m"], Float64[Array, "n n"]]:
        """Iterate the covariance recursion to its fixed point (the steady state).

        Because ``_gain_and_posterior_cov`` is data-independent, feeding its
        output covariance back as the next input traces the very recursion the
        per-step filter would follow — but with no observations required. For a
        time-invariant, stabilisable/detectable model this converges to the unique
        fixed point ``cov∞`` (the solution of the discrete algebraic Riccati
        equation), where ``cov_post == cov``. The gain there is the steady-state
        gain ``K∞`` the filter can then reuse on every step.

        Convergence is declared when successive covariances agree to ``tol``
        (absolute, ``rtol`` 0). At the fixed point ``cov == cov_post``, so the
        returned ``cov`` equals the freshly-computed ``cov_post`` to tolerance.

        Args:
            tol: Absolute tolerance on successive covariances.
            max_iter: Maximum iterations before declaring non-convergence.

        Returns:
            ``(gain, cov)``: the steady-state Kalman gain ``(n, m)`` and
            covariance ``(n, n)`` to freeze for reuse.

        Raises:
            RuntimeError: If the recursion has not converged within ``max_iter`` —
                typically because the model is not time-invariant /
                stabilisable / detectable, so no steady state exists and the full
                per-step filter (``steady_state=False``) is required instead.
        """
        model = self.model
        cov = model.prior.cov
        for _ in range(max_iter):
            gain, cov_post = _gain_and_posterior_cov(
                model.dynamics,
                model.sensor_model,
                model.dynamics_noise,
                model.sensor_noise,
                cov,
            )
            if jnp.allclose(cov, cov_post, atol=tol, rtol=0.0):
                return (
                    gain,
                    cov,
                )  # at the fixed point cov == cov_post, returning either is equivalent.
            cov = cov_post

        raise RuntimeError(
            f"steady-state covariance did not converge in {max_iter} iterations; "
            "the model may not be stabilisable/detectable. "
            "Use steady_state=False for the full per-step filter."
        )
