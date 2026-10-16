"""RxInfer (Julia) inference backend, the correctness oracle for fixed-noise models.

KalmanBackend is the default. This re-derives its filter step on a fixed-noise model
through RxInfer.jl's message passing, reached in-process via juliacall (ADR-001,
ADR-002). It shares no code with the JAX path, so it is the independent check that the
Kalman path's matrix algebra is right.

Starting Julia is slow (import + ``using RxInfer`` + JIT warmup), so it is done
once per process, lazily, on the first backend built — never per inference.
Importing this module does not touch Julia.
"""

from functools import lru_cache
from typing import Any

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from cpomdp._validation import require_fixed_noise
from cpomdp.backends.base import _validate_step_inputs
from cpomdp.types import Belief, LinearGaussianModel

__all__ = ["RxInferBackend"]

# The juliacall `Main` handle is a runtime-constructed bridge object with no type
# stubs.
_JuliaModule = Any

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

# A small branching graph hardcoded to one topology based on E. coli chemotaxis (the
# CouplingGraph difference demo): a hidden hub `CheA` feeding a fast sub-branch
# `CheY` -> {motorA, motorB} and a slow branch `CheB`, each leaf observed.
# The posterior over `CheA` is the marginal a `CouplingGraph.infer_root` collects up the
# same tree. Same MvNormal idiom as the chain step, so a node of degree > 2 (CheY)
# is the only new thing under test.
@model function cpomdp_chemotaxis_tree(
    y_motorA, y_motorB, y_cheB,
    W_chey, W_cheb, W_motorA, W_motorB,
    Q_chey, Q_cheb, Q_motorA, Q_motorB,
    R_motorA, R_motorB, R_cheB,
    prior_mean, prior_cov,
)
    chea    ~ MvNormal(mean = prior_mean, covariance = prior_cov)
    chey    ~ MvNormal(mean = W_chey * chea, covariance = Q_chey)
    cheb    ~ MvNormal(mean = W_cheb * chea, covariance = Q_cheb)
    motorA  ~ MvNormal(mean = W_motorA * chey, covariance = Q_motorA)
    motorB  ~ MvNormal(mean = W_motorB * chey, covariance = Q_motorB)
    y_motorA    ~ MvNormal(mean = motorA, covariance = R_motorA)
    y_motorB    ~ MvNormal(mean = motorB, covariance = R_motorB)
    y_cheB  ~ MvNormal(mean = cheb, covariance = R_cheB)
end

function cpomdp_run_tree(
    y_motorA, y_motorB, y_cheB,
    W_chey, W_cheb, W_motorA, W_motorB,
    Q_chey, Q_cheb, Q_motorA, Q_motorB,
    R_motorA, R_motorB, R_cheB,
    prior_mean, prior_cov,
)
    result = infer(
        model = cpomdp_chemotaxis_tree(
            W_chey = W_chey, W_cheb = W_cheb, W_motorA = W_motorA, W_motorB = W_motorB,
            Q_chey = Q_chey, Q_cheb = Q_cheb, Q_motorA = Q_motorA, Q_motorB = Q_motorB,
            R_motorA = R_motorA, R_motorB = R_motorB, R_cheB = R_cheB,
            prior_mean = prior_mean, prior_cov = prior_cov,
        ),
        data = (y_motorA = y_motorA, y_motorB = y_motorB, y_cheB = y_cheB),
    )
    posterior = result.posteriors[:chea]
    return (mean(posterior), cov(posterior))
end
"""


@lru_cache(maxsize=1)
def _julia() -> _JuliaModule:
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
    """Linear-Gaussian filtering via RxInfer.jl, the oracle backend.

    Satisfies the InferenceBackend protocol: built from a model, advances a
    belief one step at a time. On a fixed-noise model it gives
    [`KalmanBackend`][cpomdp.KalmanBackend]'s answers through separate machinery.
    It exists for correctness, so it has no steady-state mode. The first instance
    built in a process loads the Julia runtime, and later ones reuse it.

    Fixed noise only. The Julia model takes one ``Q`` and one ``R``, so a model
    carrying a state-dependent ``R(x)`` or ``Q(x)`` raises ``ValueError`` here, before
    Julia starts. Filtering it on the plain fields would hold constant what the model
    varies.
    """

    def __init__(self, model: LinearGaussianModel) -> None:
        require_fixed_noise(
            model,
            "the RxInfer model filters on one fixed Q and R. Use KalmanBackend for "
            "R(x) or Q(x).",
        )
        self._model = model
        self._jl = _julia()

    @property
    def model(self) -> LinearGaussianModel:
        """The model this backend filters under, fixed at construction."""
        return self._model

    def infer_states(
        self,
        observation: ArrayLike,
        prior: Belief,
        action: ArrayLike | None = None,
    ) -> Belief:
        """Advance the belief one filter step: prior in, posterior out.

        Args:
            observation: Latest sensor reading, shape ``(m,)``.
            prior: Current belief; never mutated.
            action: Action just taken, shape ``(p,)``. Required iff the model has
                a control matrix; pass ``None`` for pure filtering.

        Raises:
            ValueError: On a shape/None mismatch (see ``_validate_step_inputs``).
        """
        model = self.model
        observation, action = _validate_step_inputs(model, observation, prior, action)
        control_matrix = model.control_matrix
        if control_matrix is None:
            control_term = jnp.zeros(model.n_states)
        else:
            # _validate_step_inputs guarantees a non-None action when control exists
            assert action is not None
            control_term = control_matrix @ action

        # juliacall speaks numpy, not jax.Array, so coerce every array as it
        # crosses into Julia and coerce the posteriors back on the way out.
        mean_post, cov_post = self._jl.cpomdp_run_step(
            np.asarray(observation),
            np.asarray(model.dynamics_matrix),
            np.asarray(model.observation_matrix),
            np.asarray(model.dynamics_noise),
            np.asarray(model.observation_noise),
            np.asarray(prior.mean),
            np.asarray(prior.cov),
            np.asarray(control_term),
        )

        return Belief(mean=np.asarray(mean_post), cov=np.asarray(cov_post))


def _rxinfer_chemotaxis_tree_root(
    *,
    prior_mean: float,
    prior_var: float,
    w_chey: float,
    w_cheb: float,
    w_motora: float,
    w_motorb: float,
    q_chey: float,
    q_cheb: float,
    q_motora: float,
    q_motorb: float,
    r_motora: float,
    r_motorb: float,
    r_cheb: float,
    y_motora: float,
    y_motorb: float,
    y_cheb: float,
) -> tuple[float, float]:
    """RxInfer's root marginal for the branching difference-demo tree (the oracle).

    Drives the hardcoded `cpomdp_chemotaxis_tree` model (hidden hub `chea` ->
    fast `chey` -> two motor, plus slow `cheb`; the three leaves observed) and
    returns the posterior over `chea` as `(mean, variance)`. Every scalar is lifted
    to the 1x1 / 1-vector form the `MvNormal` model expects. This re-derives, through
    RxInfer's message passing, what `CouplingGraph.infer_root` collects up the same
    tree.

    The first call in a process loads the Julia runtime (see ``_julia``).
    """
    jl = _julia()

    def vec(value: float):
        return np.array([float(value)])

    def mat(value: float):
        return np.array([[float(value)]])

    mean_post, cov_post = jl.cpomdp_run_tree(
        vec(y_motora),
        vec(y_motorb),
        vec(y_cheb),
        mat(w_chey),
        mat(w_cheb),
        mat(w_motora),
        mat(w_motorb),
        mat(q_chey),
        mat(q_cheb),
        mat(q_motora),
        mat(q_motorb),
        mat(r_motora),
        mat(r_motorb),
        mat(r_cheb),
        vec(prior_mean),
        mat(prior_var),
    )
    mean_post = np.asarray(mean_post)
    cov_post = np.asarray(cov_post)
    return float(mean_post[0]), float(cov_post[0, 0])
