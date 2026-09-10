"""Shared construction-time validators (internal)."""

from collections.abc import Callable
from typing import Protocol, TypeVar

import numpy as np
from jax.errors import ConcretizationTypeError, TracerArrayConversionError
from jaxtyping import Array, Float64
from numpy.typing import ArrayLike

__all__ = [
    "UNVERSIONED",
    "concrete",
    "validate_covariance",
    "validate_declared",
    "validate_finite",
]

UNVERSIONED = (
    "version must be a non-empty string — the {subject} is declared and versioned"
)


def validate_covariance(
    cov: Float64[Array, "n n"], name: str, *, require_definite: bool = False
) -> None:
    """Square, symmetric and positive-(semi-)definite, checked once at construction.

    ``require_definite=True`` demands positive-*definite* to a floor of ``1e-8``,
    relative to the largest eigenvalue and never below that in absolute terms, for a
    matrix something inverts. The default accepts a zero-variance direction, which
    is a sharp belief or a deterministic noise rather than an error. An indefinite
    matrix is refused either way: accepted, it yields a silent negative-variance
    belief downstream. Skipped under a trace, where the values are abstract. The
    pytree ``tree_unflatten`` path never calls it, so ``jit``, ``vmap`` and ``grad``
    are unaffected.
    """
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"{name} must be a square 2-D matrix, got shape {cov.shape}")
    cov_np = concrete(cov)
    if cov_np is None:
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
    values = concrete(arr)
    if values is not None and not bool(np.isfinite(values).all()):
        raise ValueError(f"{name} must be finite (no NaN/Inf).")


def concrete(arr: ArrayLike) -> np.ndarray | None:
    """``arr`` as a float NumPy array, or ``None`` under a trace.

    A value built inside ``jit`` or ``grad`` is abstract and cannot be inspected.
    Every construction-time check skips it on the same terms: validation is enforced
    at the eager build, where the values are real.
    """
    try:
        return np.asarray(arr, dtype=float)
    except (TracerArrayConversionError, ConcretizationTypeError):
        return None


class _Named(Protocol):
    """Anything a declared set holds: it has a name, and that name identifies it."""

    name: str


_Member = TypeVar("_Member", bound=_Named)


def validate_declared(
    members: tuple[_Member, ...],
    version: str,
    *,
    subject: str,
    contains: Callable[[_Member], bool],
    requirement: str,
) -> None:
    """The checks every declared set shares: versioned, non-empty, unique, complete."""
    if not isinstance(version, str) or not version:
        raise ValueError(UNVERSIONED.format(subject=subject))
    if not members:
        raise ValueError(f"a declared {subject} needs at least one member")
    names = [member.name for member in members]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"duplicate name(s) {duplicates}; a member of a declared {subject} is "
            "identified by its name alone"
        )
    if not [member for member in members if contains(member)]:
        raise ValueError(requirement)
