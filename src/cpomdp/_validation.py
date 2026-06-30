"""Shared construction-time validators (internal)."""

import numpy as np
from jax.errors import ConcretizationTypeError, TracerArrayConversionError
from jaxtyping import Array, Float64

__all__ = ["validate_covariance", "validate_finite"]


def validate_covariance(
    cov: Float64[Array, "n n"], name: str, *, require_definite: bool = False
) -> None:
    """Square (2-D, n x n) + symmetric + positive-(semi-)definite check.

    Shared by ``Belief.cov``, ``dynamics_noise``, ``sensor_noise`` and
    ``Preference.precision``. ``require_definite=True`` demands positive-*definite*
    (``sensor_noise``: the EFE/Kalman epistemic term inverts it, and a noiseless
    ``R=0`` sends the information gain to ``+inf``); the default is
    positive-*semi*-definite, where a degenerate zero-variance direction is a
    legitimate (if sharp) belief / deterministic noise. Enforced once here at the
    trust boundary: an indefinite matrix (e.g. an off-diagonal correlation larger
    than the variances) is a physically impossible covariance that yields a silent
    negative-variance belief downstream if accepted. The pytree ``tree_unflatten``
    path skips validation, so ``jit``/``vmap``/``grad`` are unaffected — and a model
    rebuilt inside a trace (where ``eigvalsh`` stays abstract) is skipped via the
    ``np.asarray`` guard.
    """
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"{name} must be a square 2-D matrix, got shape {cov.shape}")
    # Concretise once via NumPy: the symmetry and PSD checks both need concrete
    # values (jnp stays abstract under a trace). If `cov` is genuinely traced — a
    # model rebuilt inside jit/grad — np.asarray raises and we skip: validation is a
    # concrete-construction concern, already enforced at the eager build.
    try:
        cov_np = np.asarray(cov, dtype=float)
    except (TracerArrayConversionError, ConcretizationTypeError):
        return
    if not np.allclose(cov_np, cov_np.T):
        raise ValueError(f"{name} must be symmetric.")
    eig = np.linalg.eigvalsh(cov_np)
    tol = 1e-8 * max(1.0, float(np.abs(eig).max()))
    if require_definite:
        if float(eig.min()) <= tol:
            raise ValueError(
                f"{name} must be positive-definite, but its smallest eigenvalue is "
                f"{float(eig.min()):.3g} (a noiseless or degenerate sensor sends the "
                f"information gain to infinity)."
            )
    elif float(eig.min()) < -tol:
        raise ValueError(
            f"{name} must be positive-semi-definite (a covariance), but its "
            f"smallest eigenvalue is {float(eig.min()):.3g}."
        )


def validate_finite(arr: Float64[Array, "n"], name: str) -> None:
    """Reject NaN/Inf entries — a concrete-construction trust-boundary check.

    Skipped under a trace (a value built inside jit/grad), like the PSD check, via the
    ``np.asarray`` guard: validation is enforced at the eager build.
    """
    try:
        finite = bool(np.isfinite(np.asarray(arr, dtype=float)).all())
    except (TracerArrayConversionError, ConcretizationTypeError):
        return
    if not finite:
        raise ValueError(f"{name} must be finite (no NaN/Inf).")
