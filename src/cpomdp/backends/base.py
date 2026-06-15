from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from cpomdp.types import Belief, LinearGaussianModel

__all__ = ["InferenceBackend", "validate_step_inputs"]


@runtime_checkable
class InferenceBackend(Protocol):
    """A swappable inference engine for a ``LinearGaussianModel``.

    A backend is *built from a model*: any expensive, data-independent work
    (front-loading — see DECISIONS.md ADR-002) happens at construction, so the
    per-step ``infer_states`` stays cheap. Each call advances the belief one
    recursive filter step: the current belief goes in as the ``prior`` and the
    updated belief comes back as the posterior.

    The Protocol is structural: any class with a matching ``infer_states`` is a
    backend, with no shared base class. This is the abstraction wall — the
    native Kalman fast path and the RxInfer oracle are interchangeable behind it,
    and neither's implementation (NumPy, juliacall, …) leaks into this signature.
    """

    def infer_states(
        self,
        observation: NDArray[np.float64],
        prior: Belief,
        action: NDArray[np.float64] | None = None,
    ) -> Belief:
        """Advance the belief by one filter step: ``prior`` in, posterior out.

        Given the current belief (``prior``) and a new ``observation`` (plus the
        ``action`` just taken, if the model has a control matrix), return the
        updated belief.
        """
        ...


def validate_step_inputs(
    model: LinearGaussianModel,
    observation: NDArray[np.float64],
    prior: Belief,
    action: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Coerce and shape-check one step's inputs at the trust boundary.

    ``LinearGaussianModel`` validates the model once at construction; this gives
    the per-step runtime data the same care, since that is where library users
    actually slip. Living here (not in any one backend) keeps the native fast
    path and the RxInfer oracle validating *identically* — an oracle that
    accepted inputs the fast path rejected would not be a trustworthy oracle.

    The shape checks also close the silent-broadcast trap: a length-1
    observation would otherwise broadcast against the ``m``-D prediction error
    and yield a confident *wrong* belief rather than an error.

    Args:
        model: The model being filtered under — supplies the expected dims.
        observation: Raw sensor reading; any array-like, coerced to float.
        prior: The incoming belief, checked against the model's state dim.
        action: Raw action; any array-like or ``None``. Coerced only when the
            model has a control matrix.

    Returns:
        The coerced ``(observation, action)``. ``action`` is ``None`` for a
        control-free model, otherwise a float array of shape ``(p,)``.

    Raises:
        ValueError: If ``observation`` is not shape ``(m,)``, ``prior`` is not
            over the ``n``-D state, the model needs an action but got ``None``,
            or ``action`` is not shape ``(p,)``.
    """
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
