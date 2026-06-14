import numpy as np

from cpomdp.types import Belief, LinearGaussianModel


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
        observation: np.ndarray,
        prior: Belief,
        action: np.ndarray | None = None,
    ) -> Belief:
        """Advance the belief by one filter step.

        Runs one predict/update cycle: step the prior through the dynamics
        (applying ``action`` if the model has a control matrix), then correct the
        prediction toward ``observation`` using the Kalman gain. In steady-state
        mode the gain and covariance are the frozen fixed-point values; otherwise
        they are recomputed from ``prior.cov`` on this step.

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
        observation, action = self._validate_inputs(observation, prior, action)
        control_term = (
            np.zeros(model.n_states)
            if model.control is None
            else model.control @ action
        )

        if self.steady_state:
            gain, cov_post = self._steady_gain, self._steady_cov  # frozen
        else:
            gain, cov_post = self._gain_and_posterior_cov(prior.cov)

        # Mean half of the Kalman predict step
        mean_pred = model.dynamics @ prior.mean + control_term

        # prediction_error: observation minus predicted observation
        # ("innovation" in Kalman terms). Its covariance is the gain denominator.
        prediction_error = observation - model.sensor_model @ mean_pred

        mean_post = mean_pred + gain @ prediction_error

        return Belief(mean=mean_post, cov=cov_post)

    def _validate_inputs(
        self,
        observation: np.ndarray,
        prior: Belief,
        action: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Coerce and shape-check the per-step inputs at the trust boundary.

        ``LinearGaussianModel`` validates the model once at construction; this
        gives the runtime data the same care, since that is where library users
        actually slip. Owning coercion here keeps it in a single place, so the
        rest of ``infer_states`` can assume clean float arrays.

        The shape checks also close the silent-broadcast trap: a length-1
        observation would otherwise broadcast against the ``m``-D prediction
        error and yield a confident *wrong* belief rather than an error.

        Args:
            observation: Raw sensor reading; any array-like, coerced to float.
            prior: The incoming belief, checked against the model's state dim.
            action: Raw action; any array-like or ``None``. Coerced only when the
                model has a control matrix.

        Returns:
            The coerced ``(observation, action)``. ``action`` is ``None`` for a
            control-free model, otherwise a float array of shape ``(p,)``.

        Raises:
            ValueError: If ``observation`` is not shape ``(m,)``, ``prior`` is not
                over the ``n``-D state, the model needs an action but got
                ``None``, or ``action`` is not shape ``(p,)``.
        """
        model = self.model

        observation = np.asarray(observation, dtype=float)
        m = model.n_observations
        if observation.shape != (m,):
            raise ValueError(
                f"observation must be a 1-D vector of length {m} "
                f"(the observation dimension), got shape {observation.shape}"
            )

        if prior.ndim != model.n_states:
            raise ValueError(
                f"prior must be a belief over the {model.n_states}-D state, "
                f"got a {prior.ndim}-D belief"
            )

        if model.control is None:
            return observation, None

        if action is None:
            raise ValueError(
                "this model has a control matrix; infer_states requires an action"
            )
        action = np.asarray(action, dtype=float)
        p = model.n_controls
        if action.shape != (p,):
            raise ValueError(
                f"action must be a 1-D vector of length {p} "
                f"(the action dimension), got shape {action.shape}"
            )
        return observation, action

    def _gain_and_posterior_cov(
        self, prior_cov: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run one covariance recursion: prior covariance in, ``(gain, cov_post)`` out.

        This is the covariance half of the Kalman step — predict and update
        applied to the *uncertainty* rather than the mean::

            cov_pred = A · prior_cov · Aᵀ + Q       # predict: dynamics inflate it
            S        = C · cov_pred · Cᵀ + R        # prediction-error covariance
            gain     = cov_pred · Cᵀ · S⁻¹          # how far to trust the reading
            cov_post = (I − gain · C) · cov_pred    # update: the reading shrinks it

        (Letters are the model's aliases: A=dynamics, C=sensor_model,
        Q=dynamics_noise, R=sensor_noise.) The gain is obtained via
        ``np.linalg.solve`` against ``S`` rather than an explicit inverse, for
        numerical stability. Crucially this depends only on the model and
        ``prior_cov``, never on an observation — which is exactly what lets the
        steady-state mode precompute it once and the per-step mode recompute it
        cheaply each step.

        Args:
            prior_cov: The incoming belief's covariance, shape ``(n, n)``.

        Returns:
            ``(gain, cov_post)``: the Kalman gain, shape ``(n, m)``, and the
            posterior covariance, shape ``(n, n)``, for this step.
        """
        model = self.model
        cov_pred = model.dynamics @ prior_cov @ model.dynamics.T + model.dynamics_noise
        prediction_error_cov = (
            model.sensor_model @ cov_pred @ model.sensor_model.T + model.sensor_noise
        )
        gain = np.linalg.solve(prediction_error_cov, model.sensor_model @ cov_pred).T
        cov_post = (np.eye(model.n_states) - gain @ model.sensor_model) @ cov_pred

        return gain, cov_post

    def _converge_to_steady_state(
        self, tol: float, max_iter: int
    ) -> tuple[np.ndarray, np.ndarray]:
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
            gain, cov_post = self._gain_and_posterior_cov(cov)
            if np.allclose(cov, cov_post, atol=tol, rtol=0.0):
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
