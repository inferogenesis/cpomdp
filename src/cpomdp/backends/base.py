"""The ``InferenceBackend`` protocol and the shared per-step input validator."""

from typing import Protocol, runtime_checkable

import jax.numpy as jnp
from jaxtyping import Array, Float64
from numpy.typing import ArrayLike

from cpomdp.ffg.graph import Coupling
from cpomdp.types import Belief, LinearGaussianModel

__all__ = [
    "GraphBackend",
    "InadmissiblePartitionError",
    "InferenceBackend",
]


class InadmissiblePartitionError(ValueError):
    """A carry partition severs an edge the model declares EFE-load-bearing (ADR-018).

    The carry drops the cross-cluster covariance that edge holds across the time
    boundary (the ADR-016 severed mass). That cross-temporal correlation is how
    information about a slow latent integrates from weak per-step observations of the
    fast nodes it couples to; severing an edge on the covariance path from the sensed
    nodes to the targeted latent breaks that integration. The one-step epistemic and the
    behavioural drift can still look right while the *accumulated* information about the
    latent — what the instrumental epistemic rides on — degrades, up to collapse for an
    integration-limited latent. Whether the degradation is mild or total is
    model-specific (measure it with ``CouplingGraphBackend.severed_mass``), so the
    guard refuses conservatively rather than sever and hope. Keep the flagged edge's
    endpoints in one cluster, or, if a check shows the cut is safe, drop its
    ``efe_relevant`` flag.
    """


@runtime_checkable
class InferenceBackend(Protocol):
    """A swappable inference engine for a ``LinearGaussianModel``.

    A backend is *built from a model*: any expensive, data-independent work
    (front-loading — see DECISIONS.md ADR-002) happens at construction, so the
    per-step ``infer_states`` stays cheap. Each call advances the belief one
    recursive filter step: the current belief goes in as the ``prior`` and the
    updated belief comes back as the posterior.

    The Protocol is structural: any class with a matching ``infer_states`` (and the
    ``model`` it was built from) is a backend, with no shared base class. This is the
    abstraction wall — the native Kalman fast path and the RxInfer oracle are
    interchangeable behind it, and neither's implementation (JAX, juliacall, …) leaks
    into this signature.
    """

    @property
    def model(self) -> LinearGaussianModel:
        """The model this backend was built from.

        A backend is built from a model, so it can always hand it back. The
        [`Agent`][cpomdp.Agent] takes its own ``model`` from here. A plain attribute
        satisfies this as well as a read-only property.
        """
        ...

    def infer_states(
        self,
        observation: ArrayLike,
        prior: Belief,
        action: ArrayLike | None = None,
    ) -> Belief:
        """Advance the belief by one filter step: ``prior`` in, posterior out.

        Given the current belief (``prior``) and a new ``observation`` (plus the
        ``action`` just taken, if the model has a control matrix), return the
        updated belief.
        """
        ...


@runtime_checkable
class GraphBackend(InferenceBackend, Protocol):
    """An ``InferenceBackend`` over a graph of nodes, whose joint state stacks them.

    Beyond ``infer_states`` a graph backend exposes the joint predicted before an
    observation (``predicted_belief`` → μ⁺, Σ⁺), the observation noise at a predicted
    mean (``observation_noise_at`` → R), the joint-state block a node occupies
    (``block``) and the EFE-load-bearing edges its carry partition cuts
    (``severed_efe_edges``). The sensor C is ``model.observation_matrix`` and the joint
    width ``model.n_states``, as on every backend.

    The graph EFE paths, ``probe_model`` and the Agent's dispatch read a backend
    through this protocol, never the concrete class (Dependency Inversion).
    ``CouplingGraphBackend`` satisfies it. Like this module, the protocol carries no
    promise (ADR-069).
    """

    def predicted_belief(
        self, prior: Belief, action: ArrayLike | None = None
    ) -> Belief:
        """The joint after dynamics + couplings, before observing (Σ⁺, μ⁺)."""
        ...

    def block(self, node: int) -> range:
        """The joint-state indices node ``node`` occupies (``info_node`` → block)."""
        ...

    def observation_noise_at(self, mean: ArrayLike) -> Float64[Array, "m m"]:
        """The stacked observation noise R at a predicted mean (fixed → constant)."""
        ...

    def severed_efe_edges(self) -> tuple[Coupling, ...]:
        """The EFE-relevant edges this carry partition cuts (ADR-018 diagnostic).

        Non-empty means the partition drops a covariance path the epistemic integrates
        over, so every EFE entry point refuses it; empty for the exact ``[[all]]`` carry
        and for any cut that only severs unflagged edges.
        """
        ...


def _validate_step_inputs(
    model: LinearGaussianModel,
    observation: ArrayLike,
    prior: Belief,
    action: ArrayLike | None,
) -> tuple[Float64[Array, "m"], Float64[Array, "p"] | None]:
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
    observation = jnp.asarray(observation, dtype=float)
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

    if model.control_matrix is None:
        return observation, None

    if action is None:
        raise ValueError(
            "this model has a control matrix; infer_states requires an action"
        )
    action = jnp.asarray(action, dtype=float)
    p = model.n_controls
    if action.shape != (p,):
        raise ValueError(
            f"action must be a 1-D vector of length {p} "
            f"(the action dimension), got shape {action.shape}"
        )
    return observation, action
