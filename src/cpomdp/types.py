from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["Belief", "LinearGaussianModel"]


def _validate_covariance(cov: NDArray[np.float64], name: str) -> None:
    """Square (2-D, n x n) + symmetric check.

    Shared by Belief.cov, dynamics_noise and sensor_noise — all three are
    covariance matrices with the same invariants. Positive-semi-definiteness is
    deliberately NOT checked here: it's enforced at the trust boundary (user
    input), not on every construction. See DECISIONS.md ADR-002.
    """
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"{name} must be a square 2-D matrix, got shape {cov.shape}")
    if not np.allclose(cov, cov.T):
        raise ValueError(f"{name} must be symmetric.")


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
    array-like (lists, tuples, arrays) and stored as float ``ndarray``.

    Construction validates shape and symmetry of ``cov``; positive-semi-
    definiteness is enforced at the trust boundary, not here (see DECISIONS.md
    ADR-002).
    """

    mean: NDArray[np.float64]
    cov: NDArray[np.float64]  # covariance

    def __init__(self, mean: ArrayLike, cov: ArrayLike) -> None:
        object.__setattr__(self, "mean", np.asarray(mean, dtype=float))
        object.__setattr__(self, "cov", np.asarray(cov, dtype=float))
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

    dynamics: NDArray[np.float64]
    sensor_model: NDArray[np.float64]
    dynamics_noise: NDArray[np.float64]
    sensor_noise: NDArray[np.float64]
    prior: Belief
    control: NDArray[np.float64] | None

    def __init__(
        self,
        dynamics: ArrayLike,
        sensor_model: ArrayLike,
        dynamics_noise: ArrayLike,
        sensor_noise: ArrayLike,
        prior: Belief,
        control: ArrayLike | None = None,
    ) -> None:
        object.__setattr__(self, "dynamics", np.asarray(dynamics, dtype=float))
        object.__setattr__(self, "sensor_model", np.asarray(sensor_model, dtype=float))
        object.__setattr__(
            self, "dynamics_noise", np.asarray(dynamics_noise, dtype=float)
        )
        object.__setattr__(self, "sensor_noise", np.asarray(sensor_noise, dtype=float))
        object.__setattr__(self, "prior", prior)
        object.__setattr__(
            self,
            "control",
            None if control is None else np.asarray(control, dtype=float),
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
    def A(self) -> NDArray[np.float64]:
        return self.dynamics

    @property
    def B(self) -> NDArray[np.float64] | None:
        return self.control

    @property
    def C(self) -> NDArray[np.float64]:
        return self.sensor_model

    @property
    def Q(self) -> NDArray[np.float64]:
        return self.dynamics_noise

    @property
    def R(self) -> NDArray[np.float64]:
        return self.sensor_noise
