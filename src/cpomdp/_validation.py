"""Shared construction-time validators (internal)."""

from collections.abc import Callable
from typing import Protocol, TypeVar

import numpy as np
from jax.errors import ConcretizationTypeError, TracerArrayConversionError
from jaxtyping import Array, Float64
from numpy.typing import ArrayLike

__all__ = [
    "INFORMATION_FORM_INVERTS",
    "NOISELESS_SENSOR",
    "UNVERSIONED",
    "concrete",
    "equilibrium_drift",
    "is_graph_interface",
    "mark_graph_interface",
    "refuse_graph_interface",
    "require_fixed_noise",
    "state_dependent",
    "validate_covariance",
    "validate_declared",
    "validate_finite",
]

UNVERSIONED = (
    "version must be a non-empty string — the {subject} is declared and versioned"
)

NOISELESS_SENSOR = "a noiseless sensor sends the information gain to infinity"
"""Why a sensor's R must be definite: the epistemic term reads its log-determinant."""

INFORMATION_FORM_INVERTS = "the information form inverts it"
"""Why an FFG factor's noise must be definite: its message holds the inverse."""

_NOISE_SLOTS = ("observation_model", "dynamics_noise_model")

_GRAPH_INTERFACE = "_leaves_out_couplings"


def mark_graph_interface(model: object) -> None:
    """Mark a coupled graph backend's model as its interface rather than its plant.

    The mark is an attribute, not a field, so it adds no leaf. The model's pytree
    carries it in its static aux data, so a rebuilt model keeps it.
    """
    object.__setattr__(model, _GRAPH_INTERFACE, True)


def is_graph_interface(model: object) -> bool:
    """Whether ``model`` is a coupled graph backend's interface model."""
    return bool(getattr(model, _GRAPH_INTERFACE, False))


def refuse_graph_interface(
    model: object, consumer: str, *, remedy: str | None = None
) -> None:
    """Refuse a coupled graph backend's model, which leaves out the couplings.

    ``remedy`` replaces the default advice, for a consumer the user did not hand the
    model to directly.

    Raises:
        TypeError: If ``model`` is a coupled ``CouplingGraphBackend``'s ``model``.
    """
    if is_graph_interface(model):
        if remedy is None:
            remedy = (
                "Pass the backend where one is taken, or backend.to_flat_model() for a "
                "flat model that keeps them."
            )
        raise TypeError(
            f"{consumer} was given a CouplingGraphBackend's model, which leaves out "
            "the graph's couplings, so it would compute on a plant without them. "
            f"{remedy}"
        )


class _FlagsFixed(Protocol):
    """A noise object: an observation model or a dynamics noise model."""

    @property
    def is_fixed(self) -> bool: ...


_NoiseModel = TypeVar("_NoiseModel", bound=_FlagsFixed)


def state_dependent(noise_model: _NoiseModel | None) -> _NoiseModel | None:
    """``noise_model`` if it varies with the state, ``None`` if the noise is constant.

    ``None`` comes back for a model carrying no object, which leaves the plain field as
    the noise, and for an object flagged ``is_fixed``. Every consumer with a fixed-noise
    path makes this test before taking it.
    """
    if noise_model is None or noise_model.is_fixed:
        return None
    return noise_model


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
    or ``Q(x)`` it would read the plain fields, which hold only the value at the prior
    mean, and compute against a constant where the model varies (DECISIONS.md ADR-064
    for ``R``, ADR-065 for ``Q``).

    Args:
        model: The model the consumer was handed.
        reason: What the consumer needs the constant for. It finishes the message.
        slots: The noise objects to check, in order. Both by default. Name one when
            the consumer reads only that noise.

    Raises:
        ValueError: Naming the first slot that carries a state-dependent object.
    """
    for slot in slots:
        if state_dependent(getattr(model, slot)) is not None:
            raise ValueError(f"the model's {slot} is state-dependent; {reason}")


def validate_covariance(
    cov: Float64[Array, "n n"],
    name: str,
    *,
    require_definite: bool = False,
    reason: str | None = None,
) -> None:
    """Square, symmetric and positive-(semi-)definite, checked once at construction.

    ``require_definite=True`` demands positive-*definite* to a floor of ``1e-8``,
    relative to the largest eigenvalue and never below that in absolute terms, for a
    matrix something inverts or factors. The default accepts a zero-variance
    direction, which is a sharp belief or a deterministic noise rather than an error.
    An indefinite matrix is refused either way: accepted, it yields a silent
    negative-variance belief downstream. Skipped under a trace, where the values are
    abstract. The pytree ``tree_unflatten`` path never calls it, so ``jit``, ``vmap``
    and ``grad`` are unaffected.

    ``reason`` says why the caller needs a definite matrix, and the refusal gives it
    after "because". Callers differ, so there is no default and none is printed when
    it is omitted.
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
            because = "" if reason is None else f" because {reason}"
            raise ValueError(
                f"{name} must be positive-definite{because}. Its smallest eigenvalue "
                f"is {float(eig.min()):.3g}."
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
