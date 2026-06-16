"""Core data types: the Gaussian ``Belief`` and its ``LinearGaussianModel``."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float64
from numpy.typing import ArrayLike

__all__ = ["Belief", "LinearGaussianModel"]


def _validate_covariance(cov: Float64[Array, "n n"], name: str) -> None:
    """Square (2-D, n x n) + symmetric check.

    Shared by Belief.cov, dynamics_noise and sensor_noise — all three are
    covariance matrices with the same invariants. Positive-semi-definiteness is
    deliberately NOT checked here: it's enforced at the trust boundary (user
    input), not on every construction. See DECISIONS.md ADR-002.
    """
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"{name} must be a square 2-D matrix, got shape {cov.shape}")
    if not jnp.allclose(cov, cov.T):
        raise ValueError(f"{name} must be symmetric.")


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
        _validate_covariance(self.cov, "belief covariance")
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
    """

    dynamics: Float64[Array, "n n"]
    sensor_model: Float64[Array, "m n"]
    dynamics_noise: Float64[Array, "n n"]
    sensor_noise: Float64[Array, "m m"]
    prior: Belief
    control: Float64[Array, "n p"] | None

    def __init__(
        self,
        dynamics: ArrayLike,
        sensor_model: ArrayLike,
        dynamics_noise: ArrayLike,
        sensor_noise: ArrayLike,
        prior: Belief,
        control: ArrayLike | None = None,
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
        _validate_covariance(self.dynamics_noise, "dynamics_noise")
        if self.dynamics_noise.shape != (n, n):
            raise ValueError(
                f"dynamics_noise must be {n}x{n} to match the {n}-D state, "
                f"got shape {self.dynamics_noise.shape}"
            )

        # sensor_noise: covariance of the sensor noise, (m, m), symmetric.
        _validate_covariance(self.sensor_noise, "sensor_noise")
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

    def tree_flatten(
        self,
    ) -> tuple[tuple[Array | Belief | None, ...], None]:
        """Leaves for JAX: every matrix plus the ``prior`` belief, no static aux.

        ``control`` is included as a (possibly ``None``) leaf; an uncontrolled
        model contributes no control leaf and the ``None`` is restored on rebuild.
        """
        children = (
            self.dynamics,
            self.sensor_model,
            self.dynamics_noise,
            self.sensor_noise,
            self.prior,
            self.control,
        )
        return children, None

    @classmethod
    def tree_unflatten(
        cls, aux_data: None, children: tuple[Array | Belief | None, ...]
    ) -> "LinearGaussianModel":
        """Rebuild from leaves without validating — the leaves may be tracers."""
        obj = object.__new__(cls)
        fields = (
            "dynamics",
            "sensor_model",
            "dynamics_noise",
            "sensor_noise",
            "prior",
            "control",
        )
        for name, value in zip(fields, children, strict=True):
            object.__setattr__(obj, name, value)
        return obj
