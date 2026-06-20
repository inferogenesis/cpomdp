"""Core data types: the Gaussian ``Belief`` and its ``LinearGaussianModel``."""

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
    so it passes through ``jit``/``vmap``/``grad`` as data. JAX rebuilds it from
    its leaves without re-running validation; the shape/symmetry checks fire only
    on direct construction, at the trust boundary. Positive-semi-definiteness is
    enforced at the trust boundary too, not here (see DECISIONS.md ADR-002).
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


# A pytree leaf of a LinearGaussianModel: a matrix, the prior Belief, a child
# sensor/process-noise model, or None (an absent control/observation/process_noise).
_ModelLeaf = Array | Belief | ObservationModel | DynamicsNoise | None


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class LinearGaussianModel:
    """A linear-Gaussian state-space model — the agent's generative model.

    The agent's assumed story for how a hidden state evolves and produces
    observations, under linear maps and Gaussian noise::

        next_state  = dynamics @ state + control @ action + dynamics noise
        observation = sensor_model @ state               + sensor noise

    The noise terms are zero-mean Gaussians with covariances ``dynamics_noise``
    and ``sensor_noise``; the initial state is drawn from ``prior``.

    Parameters are *role-named* rather than using the traditional control-theory
    letters, to avoid the letter collision with discrete active inference
    (pymdp), where the same letters mean different things. The "also known as"
    column lists the terms other backgrounds use, so readers can still find the
    right field. (Letters survive as ``.A``/``.B``/``.C``/``.Q``/``.R`` aliases
    for backend use.)

    ================  ======  =========================  =====  ====================
    role name         letter  meaning                    shape  also known as
    ================  ======  =========================  =====  ====================
    ``dynamics``      A       state -> next state        (n,n)  state-transition
    ``control``       B       action -> state (optional) (n,p)  input/control matrix
    ``sensor_model``  C       state -> expected reading  (m,n)  observation/emission
    ``dynamics_noise``  Q     dynamics-noise covariance  (n,n)  process noise
    ``sensor_noise``  R       sensor-noise covariance    (m,m)  observation noise
    ``prior``         --      initial belief over state  n-D    Belief / D (pymdp)
    ================  ======  =========================  =====  ====================

    Dimensions: ``n`` = state, ``m`` = observation, ``p`` = action. A model with
    no ``control`` is a pure filtering (tracking) model.

    Three optional fields (all default ``None`` → the plain fixed-matrix model)
    extend it: ``observation`` (an :class:`~cpomdp.observation.ObservationModel` for
    state-dependent sensing ``R(x)``), ``process_noise`` (a
    :class:`~cpomdp.dynamics.DynamicsNoise` for state-dependent process noise
    ``Q(x)``), and ``structure`` (a :class:`~cpomdp.structure.ModelStructure`
    declaring the factor / Markov-blanket partition).
    """

    dynamics: Float64[Array, "n n"]
    sensor_model: Float64[Array, "m n"]
    dynamics_noise: Float64[Array, "n n"]
    sensor_noise: Float64[Array, "m m"]
    prior: Belief
    control: Float64[Array, "n p"] | None
    observation: ObservationModel | None
    process_noise: DynamicsNoise | None
    structure: ModelStructure | None

    def __init__(
        self,
        dynamics: ArrayLike,
        sensor_model: ArrayLike,
        dynamics_noise: ArrayLike,
        sensor_noise: ArrayLike,
        prior: Belief,
        control: ArrayLike | None = None,
        observation: ObservationModel | None = None,
        process_noise: DynamicsNoise | None = None,
        structure: ModelStructure | None = None,
    ) -> None:
        object.__setattr__(self, "dynamics", jnp.asarray(dynamics, dtype=float))
        object.__setattr__(self, "sensor_model", jnp.asarray(sensor_model, dtype=float))
        object.__setattr__(
            self, "dynamics_noise", jnp.asarray(dynamics_noise, dtype=float)
        )
        object.__setattr__(self, "sensor_noise", jnp.asarray(sensor_noise, dtype=float))
        object.__setattr__(self, "prior", prior)
        object.__setattr__(
            self,
            "control",
            None if control is None else jnp.asarray(control, dtype=float),
        )
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "process_noise", process_noise)
        object.__setattr__(self, "structure", structure)
        self._validate()

    def _validate(self) -> None:
        # dynamics is square and defines the state dimension n.
        if self.dynamics.ndim != 2 or self.dynamics.shape[0] != self.dynamics.shape[1]:
            raise ValueError(
                f"dynamics must be a square (n x n) matrix, "
                f"got shape {self.dynamics.shape}"
            )
        n = self.n_states

        # sensor_model maps state -> observation: (m, n). Its rows define m.
        if self.sensor_model.ndim != 2 or self.sensor_model.shape[1] != n:
            raise ValueError(
                f"sensor_model must have {n} columns to match the {n}-D state, "
                f"got shape {self.sensor_model.shape}"
            )
        m = self.n_observations

        # dynamics_noise: covariance of the dynamics noise, (n, n), symmetric.
        validate_covariance(self.dynamics_noise, "dynamics_noise")
        if self.dynamics_noise.shape != (n, n):
            raise ValueError(
                f"dynamics_noise must be {n}x{n} to match the {n}-D state, "
                f"got shape {self.dynamics_noise.shape}"
            )

        # sensor_noise: covariance of the sensor noise, (m, m), symmetric.
        validate_covariance(self.sensor_noise, "sensor_noise", require_definite=True)
        if self.sensor_noise.shape != (m, m):
            raise ValueError(
                f"sensor_noise must be {m}x{m} to match the {m}-D observation, "
                f"got shape {self.sensor_noise.shape}"
            )

        # control (optional) maps action -> state: (n, p). Rows must match n.
        if self.control is not None and (
            self.control.ndim != 2 or self.control.shape[0] != n
        ):
            raise ValueError(
                f"control must have {n} rows to match the {n}-D state, "
                f"got shape {self.control.shape}"
            )
        if self.observation is not None and not isinstance(
            self.observation, ObservationModel
        ):
            raise TypeError(
                f"observation must be an ObservationModel, "
                f"got {type(self.observation).__name__}"
            )

        # process_noise (optional): state-dependent Q(x). CallableProcessNoise can't
        # check its own shape (no n), so probe it here, where n is known.
        if self.process_noise is not None:
            if not isinstance(self.process_noise, DynamicsNoise):
                raise TypeError(
                    f"process_noise must be a DynamicsNoise, "
                    f"got {type(self.process_noise).__name__}"
                )
            q_probe = jnp.asarray(self.process_noise.noise_at(jnp.zeros(n)))
            validate_covariance(q_probe, "process_noise.noise_at(x)")
            if q_probe.shape != (n, n):
                raise ValueError(
                    f"process_noise.noise_at(x) must return an {n}x{n} covariance "
                    f"to match the {n}-D state, got shape {q_probe.shape}"
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

        # prior is a Belief over the same n-D state.
        if not isinstance(self.prior, Belief):
            raise TypeError(f"prior must be a Belief, got {type(self.prior).__name__}")
        if self.prior.ndim != n:
            raise ValueError(
                f"prior must be over the {n}-D state, got a {self.prior.ndim}-D belief"
            )

    @property
    def n_states(self) -> int:
        """Dimension of the hidden state (n)."""
        return self.dynamics.shape[0]

    @property
    def n_observations(self) -> int:
        """Dimension of an observation (m)."""
        return self.sensor_model.shape[0]

    @property
    def n_controls(self) -> int:
        """Dimension of an action (p); 0 if the model has no control."""
        return 0 if self.control is None else self.control.shape[1]

    # --- control-theory letter aliases (for backend/maths internals) ---
    @property
    def A(self) -> Float64[Array, "n n"]:
        """A: the state-transition matrix (alias of ``dynamics``)."""
        return self.dynamics

    @property
    def B(self) -> Float64[Array, "n p"] | None:
        """B: the control matrix (alias of ``control``); ``None`` if uncontrolled."""
        return self.control

    @property
    def C(self) -> Float64[Array, "m n"]:
        """C: the observation matrix (alias of ``sensor_model``)."""
        return self.sensor_model

    @property
    def Q(self) -> Float64[Array, "n n"]:
        """Q: the process-noise covariance (alias of ``dynamics_noise``)."""
        return self.dynamics_noise

    @property
    def R(self) -> Float64[Array, "m m"]:
        """R: the observation-noise covariance (alias of ``sensor_noise``)."""
        return self.sensor_noise

    def tree_flatten(self) -> tuple[tuple[_ModelLeaf, ...], ModelStructure | None]:
        """Leaves for JAX: every matrix plus the ``prior`` belief; ``structure`` is aux.

        ``control``, ``observation`` and ``process_noise`` are included as (possibly
        ``None``) children; an uncontrolled / fixed-sensor / fixed-Q model contributes
        no leaf there and the ``None`` is restored on rebuild. A non-``None``
        ``observation``/``process_noise`` is itself a pytree and recurses into its own
        leaves. ``structure`` (declarative metadata, no array leaves) rides in the
        static aux_data, so two models differing only in ``structure`` are different
        pytrees and a jit keyed on the model re-specialises when it changes.
        """
        children = (
            self.dynamics,
            self.sensor_model,
            self.dynamics_noise,
            self.sensor_noise,
            self.prior,
            self.control,
            self.observation,
            self.process_noise,
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
            "dynamics",
            "sensor_model",
            "dynamics_noise",
            "sensor_noise",
            "prior",
            "control",
            "observation",
            "process_noise",
        )
        for name, value in zip(fields, children, strict=True):
            object.__setattr__(obj, name, value)
        object.__setattr__(obj, "structure", aux_data)
        return obj
