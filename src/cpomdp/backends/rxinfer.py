"""RxInfer (Julia) inference backend — the correctness oracle for v0.1.

The native KalmanBackend is the default fast path; this re-derives the same
linear-Gaussian filter step through RxInfer.jl's message passing, reached
in-process via juliacall (ADR-001/002). Sharing no code with the NumPy path, it
is the independent check that the fast path's matrix algebra is right.

Starting Julia is slow (import + ``using RxInfer`` + JIT warmup), so it is done
once per process, lazily, on the first backend built — never per inference.
Importing this module does not touch Julia.
"""

from functools import lru_cache
from typing import Any

import numpy as np

from cpomdp.backends.base import validate_step_inputs
from cpomdp.types import Belief, LinearGaussianModel

# One filter step as a one-timestep model: prior on the previous state, one
# transition, one observation; the posterior over x is the Kalman predict+update.
# With a single observation this is the filter, not the smoother (ADR-001).
# MvNormal covers every dimension, including the scalar 1×1 and zero-Q cases.
# control_term arrives pre-computed (B @ action) from Python.
_MODEL_SRC = r"""
@model function cpomdp_kalman_step(y, A, C, Q, R, prior_mean, prior_cov, control_term)
    x_prev ~ MvNormal(mean = prior_mean, covariance = prior_cov)
    x      ~ MvNormal(mean = A * x_prev + control_term, covariance = Q)
    y      ~ MvNormal(mean = C * x, covariance = R)
end

function cpomdp_run_step(y, A, C, Q, R, prior_mean, prior_cov, control_term)
    result = infer(
        model = cpomdp_kalman_step(
            A = A, C = C, Q = Q, R = R,
            prior_mean = prior_mean, prior_cov = prior_cov,
            control_term = control_term,
        ),
        data = (y = y,),
    )
    posterior = result.posteriors[:x]
    return (mean(posterior), cov(posterior))
end
"""


@lru_cache(maxsize=1)
def _julia() -> Any:
    """Load Julia + RxInfer and define the model, once per process.

    juliacall is imported here, not at module top, so the module imports without
    the optional ``rxinfer`` extra installed.
    """
    try:
        from juliacall import Main as jl  # ty:ignore[unresolved-import]
    except ImportError as exc:  # pragma: no cover - only hit without the extra
        raise ImportError(
            "the RxInfer backend needs the optional Julia bridge, which isn't "
            "installed. Add cpomdp's 'rxinfer' extra "
            "(e.g. `pip install 'cpomdp[rxinfer]'`)."
        ) from exc

    jl.seval("using RxInfer")
    jl.seval(_MODEL_SRC)
    return jl


class RxInferBackend:
    """Linear-Gaussian filtering via RxInfer.jl — the oracle backend.

    Satisfies the InferenceBackend protocol: built from a model, advances a
    belief one step at a time. No steady-state mode — that belongs to the native
    fast path; this backend exists for correctness, not speed. The first instance
    built in a process loads the Julia runtime; later ones reuse it.
    """

    def __init__(self, model: LinearGaussianModel) -> None:
        self.model = model
        self._jl = _julia()

    def infer_states(
        self,
        observation: np.ndarray,
        prior: Belief,
        action: np.ndarray | None = None,
    ) -> Belief:
        """Advance the belief one filter step: prior in, posterior out.

        Args:
            observation: Latest sensor reading, shape ``(m,)``.
            prior: Current belief; never mutated.
            action: Action just taken, shape ``(p,)``. Required iff the model has
                a control matrix; pass ``None`` for pure filtering.

        Raises:
            ValueError: On a shape/None mismatch (see ``validate_step_inputs``).
        """
        model = self.model
        observation, action = validate_step_inputs(model, observation, prior, action)
        control_term = (
            np.zeros(model.n_states)
            if model.control is None
            else model.control @ action
        )

        mean_post, cov_post = self._jl.cpomdp_run_step(
            observation,
            model.dynamics,
            model.sensor_model,
            model.dynamics_noise,
            model.sensor_noise,
            prior.mean,
            prior.cov,
            control_term,
        )

        return Belief(mean=np.asarray(mean_post), cov=np.asarray(cov_post))
