"""Core data types: the Gaussian ``Belief`` and its ``LinearGaussianModel``."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float64
from numpy.typing import ArrayLike

from cpomdp._validation import (
    NOISELESS_SENSOR,
    concrete,
    state_dependent,
    validate_covariance,
    validate_finite,
)
from cpomdp.dynamics import DynamicsNoiseModel
from cpomdp.observation import ObservationModel
from cpomdp.structure import ModelStructure

__all__ = ["Belief", "LinearGaussianModel"]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class Belief:
    """A Gaussian belief over a continuous state.

    In active inference an agent never knows the hidden state directly — it holds
    a probability distribution over what the state might be. For the
    linear-Gaussian case that distribution is always a Gaussian, fully described
    by two things:

    - ``mean`` -- the centre, the best single estimate. A 1-D vector of length n.
    - ``cov``  -- the covariance, the *uncertainty*. An n x n matrix; its
      diagonal is the variance per state dimension, its off-diagonals the
      correlations between them.

    Beliefs are immutable values: updating a belief produces a *new* ``Belief``
    rather than mutating an existing one. Inputs are accepted as anything
    array-like (lists, tuples, arrays) and stored as float ``jax.Array``.

    A ``Belief`` is a registered JAX pytree (its leaves are ``mean`` and ``cov``),
    so it passes through ``jit``/``vmap``/``grad`` as data. Direct construction
    checks the shapes, that the mean is finite, and that the covariance is symmetric
    and positive semi-definite. JAX rebuilds a ``Belief`` from its leaves without
    those checks, and under a trace the value checks are skipped because the values
    are abstract.
    """

    mean: Float64[Array, "n"]
    cov: Float64[Array, "n n"]  # covariance

    def __init__(self, mean: ArrayLike, cov: ArrayLike) -> None:
        object.__setattr__(self, "mean", jnp.asarray(mean, dtype=float))
        object.__setattr__(self, "cov", jnp.asarray(cov, dtype=float))
        self._validate()

    def _validate(self) -> None:
        if self.mean.ndim != 1:
            raise ValueError(
                f"belief mean must be a 1-D vector, got shape {self.mean.shape}"
            )
        validate_finite(self.mean, "belief mean")
        validate_covariance(self.cov, "belief covariance")
        n = self.mean.shape[0]
        if self.cov.shape != (n, n):
            raise ValueError(
                f"belief covariance must be {n}x{n} to match a {n}-D mean, "
                f"got shape {self.cov.shape}"
            )

    @property
    def n_states(self) -> int:
        """Dimension of the state (n), the length of the mean vector."""
        return self.mean.shape[0]

    def tree_flatten(
        self,
    ) -> tuple[tuple[Float64[Array, "n"], Float64[Array, "n n"]], None]:
        """Leaves for JAX: ``(mean, cov)``, no static aux data."""
        return (self.mean, self.cov), None

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: None,
        children: tuple[Float64[Array, "n"], Float64[Array, "n n"]],
    ) -> "Belief":
        """Rebuild from leaves without validating — the leaves may be tracers."""
        mean, cov = children
        obj = object.__new__(cls)
        object.__setattr__(obj, "mean", mean)
        object.__setattr__(obj, "cov", cov)
        return obj


def _field_value(
    declared: ArrayLike | None,
    carried: ArrayLike | None,
    *,
    field: str,
    carrier: str,
    carried_as: str,
) -> Float64[Array, "k l"]:
    """What a model field holds, given what was passed and what its object carries.

    With no object the field is what was passed, and must be passed. Beside an object
    the field is the object's value at the prior mean, so a passed value must be that
    value exactly (DECISIONS.md ADR-071). Under a trace the values cannot be compared.
    The check is skipped like every other construction check, and the field still
    holds the object's value.

    Args:
        declared: The value passed for the field, or ``None``.
        carried: The object's value at the prior mean, or ``None`` with no object.
        field: The field's name, for the messages.
        carrier: The object's slot, for the messages.
        carried_as: What the object's value is, for the messages.

    Raises:
        TypeError: If neither supplies a value.
        ValueError: If both do and they differ.
    """
    if carried is None:
        if declared is None:
            raise TypeError(f"{field} is required when no {carrier} supplies it")
        return jnp.asarray(declared, dtype=float)
    carried = jnp.asarray(carried, dtype=float)
    if declared is None:
        return carried
    declared_values, carried_values = concrete(declared), concrete(carried)
    if (
        declared_values is not None
        and carried_values is not None
        and not np.array_equal(declared_values, carried_values)
    ):
        raise ValueError(
            f"{field} is not the {carrier}'s {carried_as}. The field is read from "
            f"the {carrier}, so omit it or pass None."
        )
    return carried


def _refuse_noise_object(value: object, field: str) -> None:
    """Name the slot a noise object belongs in when it is passed for a matrix field."""
    for protocol, kind, slot in (
        (ObservationModel, "an ObservationModel", "observation_model"),
        (DynamicsNoiseModel, "a DynamicsNoiseModel", "dynamics_noise_model"),
    ):
        if isinstance(value, protocol):
            raise TypeError(
                f"{field} takes a matrix, got {type(value).__name__}, {kind}. "
                f"Pass it as {slot}."
            )


# A pytree leaf of a LinearGaussianModel: a matrix, the prior Belief, a child
# observation or dynamics noise model, or None (an absent control_matrix,
# observation_model or dynamics_noise_model).
_ModelLeaf = Array | Belief | ObservationModel | DynamicsNoiseModel | None


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class LinearGaussianModel:
    """A linear-Gaussian state-space model — the agent's generative model.

    The agent's assumed story for how a hidden state evolves and produces
    observations, under linear maps and Gaussian noise::

        next_state  = dynamics_matrix @ state + control_matrix @ action + dynamics noise
        observation = observation_matrix @ state + observation noise

    The noise terms are zero-mean Gaussians with covariances ``dynamics_noise``
    and ``observation_noise``; the initial state is drawn from ``prior``.

    Parameters are *role-named* rather than using the traditional control-theory
    letters, to avoid the letter collision with discrete active inference
    (pymdp), where the same letters mean different things. The "also known as"
    column lists the terms other backgrounds use, so readers can still find the
    right field. Each field has that one name.

    | role name | letter | meaning | shape | also known as |
    | --- | --- | --- | --- | --- |
    | ``dynamics_matrix`` | A | state -> next state | (n,n) | state-transition |
    | ``control_matrix`` | B | action -> state (optional) | (n,p) | input/control |
    | ``observation_matrix`` | C | state -> reading | (m,n) | measurement/emission |
    | ``dynamics_noise`` | Q | dynamics-noise covariance | (n,n) | process noise |
    | ``observation_noise`` | R | reading-noise covariance | (m,m) | measurement noise |
    | ``prior`` | -- | initial belief over state | n-D | Belief / D (pymdp) |

    Dimensions: ``n`` = state, ``m`` = observation, ``p`` = action. A model with
    no ``control_matrix`` is a pure filtering (tracking) model.

    Everything after ``dynamics_matrix`` is keyword-only. Two of the four matrices
    are maps and two are covariances, all of the same rank, and only the covariances are
    content-checked. A transposed pair therefore constructs in silence whenever the maps
    are square and symmetric. Naming them at the call site rules that out.

    Three optional fields (all default ``None`` → the plain fixed-matrix model)
    extend it: ``observation_model`` (an [`ObservationModel`][cpomdp.ObservationModel]
    for state-dependent sensing ``R(x)``), ``dynamics_noise_model`` (a
    [`DynamicsNoiseModel`][cpomdp.DynamicsNoiseModel] for state-dependent dynamics
    noise ``Q(x)``), and ``structure`` (a [`ModelStructure`][cpomdp.ModelStructure]
    declaring the factor / Markov-blanket partition).

    ``observation_matrix`` and ``observation_noise`` are optional when an
    ``observation_model`` is given. Omitted, they are the sensor's ``C`` and its ``R``
    at ``prior.mean``, read once at construction. Passed, they must be those values
    exactly, so a sensor's channel is never described twice and each field means one
    thing (DECISIONS.md ADR-064, ADR-071). ``dynamics_noise`` follows the same rule
    beside a ``dynamics_noise_model``, and is its ``Q`` at ``prior.mean`` (ADR-065).
    A consumer that understands a state-dependent object reads ``R(x)`` or ``Q(x)``
    from it. One that needs a constant refuses the model. To move the prior with
    ``dataclasses.replace``, pass the noises an object supplies as ``None`` so they
    are read again at the new mean.
    """

    dynamics_matrix: Float64[Array, "n n"]
    observation_matrix: Float64[Array, "m n"]
    dynamics_noise: Float64[Array, "n n"]
    observation_noise: Float64[Array, "m m"]
    prior: Belief
    control_matrix: Float64[Array, "n p"] | None
    observation_model: ObservationModel | None
    dynamics_noise_model: DynamicsNoiseModel | None
    structure: ModelStructure | None

    def __init__(
        self,
        dynamics_matrix: ArrayLike,
        *,
        observation_matrix: ArrayLike | None = None,
        dynamics_noise: ArrayLike | None = None,
        observation_noise: ArrayLike | None = None,
        prior: Belief,
        control_matrix: ArrayLike | None = None,
        observation_model: ObservationModel | None = None,
        dynamics_noise_model: DynamicsNoiseModel | None = None,
        structure: ModelStructure | None = None,
    ) -> None:
        for field_name, value in (
            ("dynamics_matrix", dynamics_matrix),
            ("observation_matrix", observation_matrix),
            ("dynamics_noise", dynamics_noise),
            ("observation_noise", observation_noise),
            ("control_matrix", control_matrix),
        ):
            _refuse_noise_object(value, field_name)
        object.__setattr__(
            self, "dynamics_matrix", jnp.asarray(dynamics_matrix, dtype=float)
        )
        object.__setattr__(self, "prior", prior)
        object.__setattr__(
            self,
            "control_matrix",
            None
            if control_matrix is None
            else jnp.asarray(control_matrix, dtype=float),
        )
        object.__setattr__(self, "observation_model", observation_model)
        object.__setattr__(self, "dynamics_noise_model", dynamics_noise_model)
        object.__setattr__(self, "structure", structure)
        self._validate_noise_sources()
        sensor_matrix, sensor_noise = self._sensor_at_prior()
        object.__setattr__(
            self,
            "observation_matrix",
            _field_value(
                observation_matrix,
                sensor_matrix,
                field="observation_matrix",
                carrier="observation_model",
                carried_as="C",
            ),
        )
        object.__setattr__(
            self,
            "observation_noise",
            _field_value(
                observation_noise,
                sensor_noise,
                field="observation_noise",
                carrier="observation_model",
                carried_as="R at prior.mean",
            ),
        )
        object.__setattr__(
            self,
            "dynamics_noise",
            _field_value(
                dynamics_noise,
                self._dynamics_noise_at_prior(),
                field="dynamics_noise",
                carrier="dynamics_noise_model",
                carried_as="Q at prior.mean",
            ),
        )
        self._validate()

    def _validate_noise_sources(self) -> None:
        """Check the kinds of the prior and the two noise objects before any is read.

        The fields beside an object are read from it at the prior mean before anything
        else is validated, so these three are checked first.
        """
        if not isinstance(self.prior, Belief):
            raise TypeError(f"prior must be a Belief, got {type(self.prior).__name__}")
        if self.observation_model is not None and not isinstance(
            self.observation_model, ObservationModel
        ):
            raise TypeError(
                f"observation_model must be an ObservationModel, "
                f"got {type(self.observation_model).__name__}"
            )
        if self.dynamics_noise_model is not None and not isinstance(
            self.dynamics_noise_model, DynamicsNoiseModel
        ):
            raise TypeError(
                f"dynamics_noise_model must be a DynamicsNoiseModel, "
                f"got {type(self.dynamics_noise_model).__name__}"
            )

    def _sensor_at_prior(
        self,
    ) -> tuple[Float64[Array, "m n"] | None, Float64[Array, "m m"] | None]:
        """The sensor's ``(C, R)`` at the prior mean, ``(None, None)`` with no sensor.

        A fixed sensor returns its own pair at any state (DECISIONS.md ADR-064).
        """
        if self.observation_model is None:
            return None, None
        return self.observation_model.linearize(self.prior.mean)

    def _dynamics_noise_at_prior(self) -> Float64[Array, "n n"] | None:
        """The dynamics noise model's ``Q`` at the prior mean, ``None`` with none.

        The rule ADR-064 sets for ``R`` (DECISIONS.md ADR-065). ``_validate`` probes
        ``Q(x)`` at the origin as well, a separate check left where it was.
        """
        if self.dynamics_noise_model is None:
            return None
        return self.dynamics_noise_model.noise_at(self.prior.mean)

    def _validate(self) -> None:
        # dynamics is square and defines the state dimension n.
        if (
            self.dynamics_matrix.ndim != 2
            or self.dynamics_matrix.shape[0] != self.dynamics_matrix.shape[1]
        ):
            raise ValueError(
                f"dynamics_matrix must be a square (n x n) matrix, "
                f"got shape {self.dynamics_matrix.shape}"
            )
        n = self.n_states

        # observation_matrix maps state -> observation: (m, n). Its rows define m.
        if self.observation_matrix.ndim != 2 or self.observation_matrix.shape[1] != n:
            raise ValueError(
                f"observation_matrix must have {n} columns to match the {n}-D state, "
                f"got shape {self.observation_matrix.shape}"
            )
        m = self.n_observations

        # dynamics_noise: covariance of the dynamics noise, (n, n), symmetric.
        validate_covariance(self.dynamics_noise, "dynamics_noise")
        if self.dynamics_noise.shape != (n, n):
            raise ValueError(
                f"dynamics_noise must be {n}x{n} to match the {n}-D state, "
                f"got shape {self.dynamics_noise.shape}"
            )

        # observation_noise: covariance of the observation noise, (m, m), symmetric.
        validate_covariance(
            self.observation_noise,
            "observation_noise",
            require_definite=True,
            reason=NOISELESS_SENSOR,
        )
        if self.observation_noise.shape != (m, m):
            raise ValueError(
                f"observation_noise must be {m}x{m} to match the {m}-D observation, "
                f"got shape {self.observation_noise.shape}"
            )

        # control_matrix (optional) maps action -> state: (n, p). Rows match n.
        if self.control_matrix is not None and (
            self.control_matrix.ndim != 2 or self.control_matrix.shape[0] != n
        ):
            raise ValueError(
                f"control_matrix must have {n} rows to match the {n}-D state, "
                f"got shape {self.control_matrix.shape}"
            )
        # dynamics_noise_model (optional): state-dependent Q(x). CallableDynamicsNoise
        # cannot check its own shape (no n), so probe it here, where n is known.
        # _validate_noise_sources checked its kind.
        if self.dynamics_noise_model is not None:
            q_probe = jnp.asarray(self.dynamics_noise_model.noise_at(jnp.zeros(n)))
            validate_covariance(q_probe, "dynamics_noise_model.noise_at(state)")
            if q_probe.shape != (n, n):
                raise ValueError(
                    f"dynamics_noise_model.noise_at(state) must return an {n}x{n} "
                    f"covariance to match the {n}-D state, got shape {q_probe.shape}"
                )

        # structure (optional): declarative metadata; validated opt-in via
        # structure.validate(model), never here (the constructor stays lean, RFC-001).
        if self.structure is not None and not isinstance(
            self.structure, ModelStructure
        ):
            raise TypeError(
                f"structure must be a ModelStructure, "
                f"got {type(self.structure).__name__}"
            )

        # prior is over the same n-D state. _validate_noise_sources checked its kind.
        if self.prior.n_states != n:
            raise ValueError(
                f"prior must be over the {n}-D state, "
                f"got a {self.prior.n_states}-D belief"
            )

    @property
    def n_states(self) -> int:
        """Dimension of the hidden state (n)."""
        return self.dynamics_matrix.shape[0]

    @property
    def n_observations(self) -> int:
        """Dimension of an observation (m)."""
        return self.observation_matrix.shape[0]

    @property
    def n_controls(self) -> int:
        """Dimension of an action (p); 0 if the model has no control."""
        return 0 if self.control_matrix is None else self.control_matrix.shape[1]

    def observation_noise_at(self, state: ArrayLike) -> Float64[Array, "m m"]:
        """The observation noise R at ``state``, whatever sensor the model carries.

        A model with no ``observation_model``, or a fixed one, returns
        ``observation_noise`` itself and reads no state. A state-dependent sensor is
        linearized at ``state``. A consumer that needs R where a belief sits reads it
        here and does not branch on the sensor.

        Args:
            state: the state to read R at, shape ``(n,)``.

        Returns:
            R at ``state``, shape ``(m, m)``.
        """
        sensor = state_dependent(self.observation_model)
        if sensor is None:
            return self.observation_noise
        return sensor.linearize(state)[1]

    def tree_flatten(self) -> tuple[tuple[_ModelLeaf, ...], ModelStructure | None]:
        """Leaves for JAX: every matrix plus the ``prior`` belief; ``structure`` is aux.

        ``control_matrix``, ``observation_model`` and ``dynamics_noise_model`` are
        included as (possibly ``None``) children; an uncontrolled / fixed-sensor /
        fixed-Q model contributes no leaf there and the ``None`` is restored on
        rebuild. A non-``None`` ``observation_model``/``dynamics_noise_model`` is itself
        a pytree and recurses into its own leaves. ``structure`` (declarative metadata,
        no array leaves) rides in the static aux_data, so two models differing only in
        ``structure`` are different pytrees and a jit keyed on the model re-specialises
        when it changes.
        """
        children = (
            self.dynamics_matrix,
            self.observation_matrix,
            self.dynamics_noise,
            self.observation_noise,
            self.prior,
            self.control_matrix,
            self.observation_model,
            self.dynamics_noise_model,
        )
        return children, self.structure

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: ModelStructure | None,
        children: tuple[_ModelLeaf, ...],
    ) -> "LinearGaussianModel":
        """Rebuild from leaves without validating — the leaves may be tracers.

        ``aux_data`` is the static ``structure`` (or ``None``), restored as-is.
        """
        obj = object.__new__(cls)
        fields = (
            "dynamics_matrix",
            "observation_matrix",
            "dynamics_noise",
            "observation_noise",
            "prior",
            "control_matrix",
            "observation_model",
            "dynamics_noise_model",
        )
        for name, value in zip(fields, children, strict=True):
            object.__setattr__(obj, name, value)
        object.__setattr__(obj, "structure", aux_data)
        return obj


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class _GraphInterfaceModel(LinearGaussianModel):
    """A coupled ``CouplingGraphBackend``'s model: its interface, not its plant.

    The sizes, prior, sensors and control the backend reads, with the couplings left
    out (ADR-090). The type is the mark. ``dataclasses.replace`` builds through
    ``obj.__class__`` and a pytree rebuild through ``tree_unflatten``'s ``cls``, so
    neither loses it, and a flat consumer refuses it with ``refuse_graph_interface``.
    """


def is_graph_interface(model: object) -> bool:
    """Whether ``model`` is a coupled graph backend's interface model."""
    return isinstance(model, _GraphInterfaceModel)


def refuse_graph_interface(model: object, consumer: str) -> None:
    """Refuse a coupled graph backend's model, which leaves out the couplings.

    Raises:
        TypeError: If ``model`` is a coupled ``CouplingGraphBackend``'s ``model``.
    """
    if is_graph_interface(model):
        raise TypeError(
            f"{consumer} was given a CouplingGraphBackend's model, which leaves out "
            "the graph's couplings, so it would compute on a plant without them. "
            "Pass the backend where one is taken, or backend.to_flat_model() for a "
            "flat model that keeps them."
        )
