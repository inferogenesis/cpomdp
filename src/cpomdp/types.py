"""Core data types: the Gaussian ``Belief`` and its ``LinearGaussianModel``."""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float64
from numpy.typing import ArrayLike

from cpomdp._validation import validate_covariance, validate_finite
from cpomdp.dynamics import DynamicsNoise
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
    def ndim(self) -> int:
        """Dimensionality of the state — the length of the mean vector."""
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


def _declared_or(
    declared: ArrayLike | None, derive: Callable[[], ArrayLike]
) -> Float64[Array, "k k"]:
    """``declared`` as a float array, or what ``derive`` returns when it is ``None``."""
    return jnp.asarray(derive() if declared is None else declared, dtype=float)


# A pytree leaf of a LinearGaussianModel: a matrix, the prior Belief, a child
# sensor/process-noise model, or None (an absent control_matrix,
# observation_model or dynamics_noise_model).
_ModelLeaf = Array | Belief | ObservationModel | DynamicsNoise | None


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
    right field. (Letters survive as ``.A``/``.B``/``.C``/``.Q``/``.R`` aliases
    for backend use.)

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
    [`DynamicsNoise`][cpomdp.DynamicsNoise] for state-dependent process noise
    ``Q(x)``), and ``structure`` (a [`ModelStructure`][cpomdp.ModelStructure]
    declaring the factor / Markov-blanket partition).

    ``observation_noise`` is optional when an ``observation_model`` is given. Omitted,
    it is the sensor's ``R`` at ``prior.mean``, read once at construction. Beside a
    state-dependent sensor the field is a nominal value. A consumer that understands
    the sensor reads ``R(x)`` from the sensor. One that needs a single ``R`` either
    refuses the model or documents that it reads this one (DECISIONS.md ADR-064).
    ``dynamics_noise`` follows the same rule beside a ``dynamics_noise_model``, and is
    then its ``Q`` at ``prior.mean`` (ADR-065).
    """

    dynamics_matrix: Float64[Array, "n n"]
    observation_matrix: Float64[Array, "m n"]
    dynamics_noise: Float64[Array, "n n"]
    observation_noise: Float64[Array, "m m"]
    prior: Belief
    control_matrix: Float64[Array, "n p"] | None
    observation_model: ObservationModel | None
    dynamics_noise_model: DynamicsNoise | None
    structure: ModelStructure | None

    def __init__(
        self,
        dynamics_matrix: ArrayLike,
        *,
        observation_matrix: ArrayLike,
        dynamics_noise: ArrayLike | None = None,
        observation_noise: ArrayLike | None = None,
        prior: Belief,
        control_matrix: ArrayLike | None = None,
        observation_model: ObservationModel | None = None,
        dynamics_noise_model: DynamicsNoise | None = None,
        structure: ModelStructure | None = None,
    ) -> None:
        object.__setattr__(
            self, "dynamics_matrix", jnp.asarray(dynamics_matrix, dtype=float)
        )
        object.__setattr__(
            self, "observation_matrix", jnp.asarray(observation_matrix, dtype=float)
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
        object.__setattr__(
            self,
            "dynamics_noise",
            _declared_or(dynamics_noise, self._process_noise_at_prior),
        )
        object.__setattr__(
            self,
            "observation_noise",
            _declared_or(observation_noise, self._sensor_noise_at_prior),
        )
        self._validate()

    def _validate_noise_sources(self) -> None:
        """Check the kinds of the prior and the two noise objects before any is read.

        An omitted noise is read from its object at the prior mean before anything
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
            self.dynamics_noise_model, DynamicsNoise
        ):
            raise TypeError(
                f"dynamics_noise_model must be a DynamicsNoise, "
                f"got {type(self.dynamics_noise_model).__name__}"
            )

    def _process_noise_at_prior(self) -> Float64[Array, "n n"]:
        """The process noise's ``Q`` at the prior mean, for a model given no ``Q``.

        The rule ADR-064 sets for ``R`` (DECISIONS.md ADR-065). ``_validate`` probes
        ``Q(x)`` at the origin as well, a separate check left where it was.
        """
        if self.dynamics_noise_model is None:
            raise TypeError(
                "dynamics_noise is required when no dynamics_noise_model supplies it"
            )
        return self.dynamics_noise_model.noise_at(self.prior.mean)

    def _sensor_noise_at_prior(self) -> Float64[Array, "m m"]:
        """The sensor's ``R`` at the prior mean, for a model given no observation noise.

        The prior mean is where ``_validate`` already probes the sensor, so a fixed
        sensor's own ``R`` comes back here and meets the exact-equality guard by
        construction (DECISIONS.md ADR-064).
        """
        if self.observation_model is None:
            raise TypeError(
                "observation_noise is required when no observation_model supplies it"
            )
        return self.observation_model.linearize(self.prior.mean)[1]

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
            self.observation_noise, "observation_noise", require_definite=True
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
        if self.observation_model is not None:
            # An observation_model restates the observation channel, and the consumers
            # split on which restatement they read: the Kalman filter and the world use
            # the plain fields when the sensor is fixed, the EFE kernel uses the sensor
            # whenever one is present. Probe it here so a disagreement is refused at
            # the one place every consumer passes through.
            sensor_matrix, sensor_noise = self.observation_model.linearize(
                self.prior.mean
            )
            sensor_matrix = jnp.asarray(sensor_matrix)
            sensor_noise = jnp.asarray(sensor_noise)
            if not jnp.array_equal(sensor_matrix, self.observation_matrix):
                raise ValueError(
                    "observation_model.linearize returns an observation matrix that "
                    "is not the model's observation_matrix. One of the two would be "
                    "read depending on the consumer, so they must be the same array."
                )
            if sensor_noise.shape != (m, m):
                raise ValueError(
                    f"observation_model.linearize must return an {m}x{m} observation "
                    f"noise to match the {m}-D observation, got shape "
                    f"{sensor_noise.shape}"
                )
            if self.observation_model.is_fixed and not jnp.array_equal(
                sensor_noise, self.observation_noise
            ):
                raise ValueError(
                    "a fixed observation_model carries an observation noise that is "
                    "not the model's observation_noise. A state-dependent sensor may "
                    "vary; a fixed one restates the field and must restate it exactly."
                )

        # dynamics_noise_model (optional): state-dependent Q(x). CallableProcessNoise
        # check its own shape (no n), so probe it here, where n is known.
        # _validate_noise_sources checked its kind.
        if self.dynamics_noise_model is not None:
            q_probe = jnp.asarray(self.dynamics_noise_model.noise_at(jnp.zeros(n)))
            validate_covariance(q_probe, "dynamics_noise_model.noise_at(x)")
            if q_probe.shape != (n, n):
                raise ValueError(
                    f"dynamics_noise_model.noise_at(x) must return an {n}x{n} "
                    f"covariance to match the {n}-D state, got shape {q_probe.shape}"
                )
            # The same split as the sensor: a fixed object restates dynamics_noise,
            # and the EFE kernel reads the object where the filter reads the field.
            if self.dynamics_noise_model.is_fixed and not jnp.array_equal(
                q_probe, self.dynamics_noise
            ):
                raise ValueError(
                    "a fixed dynamics_noise_model returns a covariance that is not "
                    "the model's dynamics_noise. A state-dependent Q(x) may vary; a "
                    "fixed one restates the field and must restate it exactly."
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
        if self.prior.ndim != n:
            raise ValueError(
                f"prior must be over the {n}-D state, got a {self.prior.ndim}-D belief"
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

    # --- control-theory letter aliases (for backend/maths internals) ---
    @property
    def A(self) -> Float64[Array, "n n"]:
        """A: the state-transition matrix (alias of ``dynamics_matrix``)."""
        return self.dynamics_matrix

    @property
    def B(self) -> Float64[Array, "n p"] | None:
        """B: the control matrix (alias of ``control_matrix``); ``None`` if unset."""
        return self.control_matrix

    @property
    def C(self) -> Float64[Array, "m n"]:
        """C: the observation matrix (alias of ``observation_matrix``)."""
        return self.observation_matrix

    @property
    def Q(self) -> Float64[Array, "n n"]:
        """Q: the process-noise covariance (alias of ``dynamics_noise``)."""
        return self.dynamics_noise

    @property
    def R(self) -> Float64[Array, "m m"]:
        """R: the observation-noise covariance (alias of ``observation_noise``)."""
        return self.observation_noise

    def tree_flatten(self) -> tuple[tuple[_ModelLeaf, ...], ModelStructure | None]:
        """Leaves for JAX: every matrix plus the ``prior`` belief; ``structure`` is aux.

        ``control_matrix``, ``observation_model`` and ``dynamics_noise_model`` are
        included as (possibly
        ``None``) children; an uncontrolled / fixed-sensor / fixed-Q model contributes
        no leaf there and the ``None`` is restored on rebuild. A non-``None``
        ``observation_model``/``dynamics_noise_model`` is itself a pytree and recurses
        into its own
        leaves. ``structure`` (declarative metadata, no array leaves) rides in the
        static aux_data, so two models differing only in ``structure`` are different
        pytrees and a jit keyed on the model re-specialises when it changes.
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
