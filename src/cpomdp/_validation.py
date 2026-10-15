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
    "equilibrium_drift",
    "require_fixed_noise",
    "validate_covariance",
    "validate_declared",
    "validate_finite",
]

UNVERSIONED = (
    "version must be a non-empty string — the {subject} is declared and versioned"
)

_NOISE_SLOTS = ("observation_model", "dynamics_noise_model")


class _CarriesNoise(Protocol):
    """A model with the two optional noise objects, each flagging ``is_fixed``.

    Named structurally because this module imports nothing first-party.
    """

    @property
    def observation_model(self) -> object: ...

    @property
    def dynamics_noise_model(self) -> object: ...


def require_fixed_noise(
    model: _CarriesNoise,
    reason: str,
    *,
    slots: tuple[str, ...] = _NOISE_SLOTS,
) -> None:
    """Refuse a model carrying state-dependent noise in any of ``slots``.

    For a consumer whose arithmetic holds ``R`` and ``Q`` constant. Built over ``R(x)``
    or ``Q(x)`` it would read the plain fields, which are then nominal values, and
    compute against a constant where the model varies (DECISIONS.md ADR-064 for
    ``R``, ADR-065 for ``Q``).

    Args:
        model: The model the consumer was handed.
        reason: What the consumer needs the constant for. It finishes the message.
        slots: The noise objects to check, in order. Both by default. Name one when
            the consumer reads only that noise.

    Raises:
        ValueError: Naming the first slot that carries a state-dependent object.
    """
    for slot in slots:
        carried = getattr(model, slot)
        if carried is not None and not carried.is_fixed:
            raise ValueError(f"the model's {slot} is state-dependent; {reason}")


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
                f"{name} must be positive-definite because it is inverted. Its "
                f"smallest eigenvalue is {float(eig.min()):.3g}."
            )
    elif float(eig.min()) < -tol:
        raise ValueError(
            f"{name} must be positive-semi-definite (a covariance), but its "
            f"smallest eigenvalue is {float(eig.min()):.3g}."
        )


def equilibrium_drift(
    dynamics_matrix: ArrayLike, state: ArrayLike
) -> np.ndarray | None:
    """Where ``state`` moves in one step at zero action, or ``None`` if it stays put.

    A regulator acts on the offset from its target, and the offset obeys the model's
    dynamics only when ``A · target = target``. Under a trace the values cannot be
    inspected, and the check is skipped like every other construction check.
    """
    concrete_dynamics, concrete_state = concrete(dynamics_matrix), concrete(state)
    if concrete_dynamics is None or concrete_state is None:
        return None
    moved = concrete_dynamics @ concrete_state
    atol = 1e-12 * max(1.0, float(np.abs(concrete_state).max()))
    return None if np.allclose(moved, concrete_state, rtol=0.0, atol=atol) else moved


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
