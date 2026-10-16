"""Tier-1 linear-Gaussian factor nodes: the dynamics (ADR-012).

``GaussianTransition`` is the dynamics ``N(x'; Ax + b, Q)``. Its forward predict builds
the joint over ``[x, x']``, folds in the incoming message, and marginalizes x out (the
Schur complement). It emits a ``CanonicalGaussian`` message assembled from the
canonical-form algebra.

The within-slice coupling ``N(child; W·parent, Q)`` is the graph edge itself,
``Coupling`` in ``cpomdp.ffg.graph`` (ADR-075). A node's observation is the flat model's
sensor (ADR-073). Its message is ``likelihood_message`` in ``cpomdp.ffg._message``,
built from the sensor's ``(C, R)``.

The transition inverts its noise covariance, so it requires it positive-**definite**.
Unlike moment-form Kalman, the canonical transition factor cannot represent a
deterministic (``Q=0``) transition.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float64
from numpy.typing import ArrayLike

from cpomdp._validation import concrete, validate_covariance
from cpomdp.ffg._message import CanonicalGaussian

__all__ = ["GaussianTransition"]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class GaussianTransition:
    """Tier-1 dynamics factor ``N(x'; Ax + b, Q)``: how one node steps in time.

    Holds the node's fixed transition and process noise. A
    [`CouplingGraphBackend`][cpomdp.CouplingGraphBackend] takes one per node.

    - ``dynamics_matrix`` — A, shape ``(n, n)``.
    - ``dynamics_noise`` — Q, shape ``(n, n)``, positive-definite (it is inverted).
    """

    dynamics_matrix: Float64[Array, "n n"]
    dynamics_noise: Float64[Array, "n n"]

    def __init__(
        self, dynamics_matrix: ArrayLike, *, dynamics_noise: ArrayLike
    ) -> None:
        object.__setattr__(
            self, "dynamics_matrix", jnp.asarray(dynamics_matrix, dtype=float)
        )
        object.__setattr__(
            self, "dynamics_noise", jnp.asarray(dynamics_noise, dtype=float)
        )
        self._validate()

    def _validate(self) -> None:
        dynamics_matrix, dynamics_noise = (
            self.dynamics_matrix,
            self.dynamics_noise,
        )  # A, Q
        if (
            dynamics_matrix.ndim != 2
            or dynamics_matrix.shape[0] != dynamics_matrix.shape[1]
        ):
            raise ValueError(
                f"dynamics_matrix must be square (n, n), got shape "
                f"{dynamics_matrix.shape}"
            )
        # Q is inverted in the joint, so it must be positive-definite.
        validate_covariance(dynamics_noise, "dynamics_noise", require_definite=True)
        n = dynamics_matrix.shape[0]
        if dynamics_noise.shape != (n, n):
            raise ValueError(
                f"dynamics_noise must be {n}x{n} to match the {n}-D state, "
                f"got shape {dynamics_noise.shape}"
            )

    @classmethod
    def from_ou(
        cls, tau: float, *, stationary_var: float, dt: float
    ) -> "GaussianTransition":
        """Build a 1-D transition from Ornstein–Uhlenbeck (OU) parameters.

        An Ornstein–Uhlenbeck process is a scalar state that relaxes toward zero on a
        timescale ``tau`` while random noise keeps it wobbling with a stationary
        variance ``stationary_var`` (Σ_stat). Exactly discretising it over a step ``dt``
        gives the linear-Gaussian transition ``x' = A·x + noise(Q)`` (ADR-017):

            A = exp(−dt / tau)             — the fraction of the state surviving a step
            Q = stationary_var · (1 − A²)  — the kick that holds the stationary variance

        Scalar (1-D) only: the vector OU would need a matrix exponential and a Lyapunov
        solve, which no cpomdp node needs.

        Args:
            tau: the relaxation timescale τ (same time unit as ``dt``).
            stationary_var: the steady-state variance Σ_stat the node settles to; A
                (dynamics) and Q (dynamics_noise) are set so it holds this spread.
            dt: the discretisation step.

        Returns:
            A ``GaussianTransition`` with 1×1 ``dynamics_matrix`` (A) and
            ``dynamics_noise`` (Q).

        Raises:
            ValueError: if ``tau``, ``stationary_var`` or ``dt`` is not a finite
                positive number.
        """
        for name, value in (
            ("tau", tau),
            ("stationary_var", stationary_var),
            ("dt", dt),
        ):
            _require_positive(value, name)
        dynamics_matrix = jnp.exp(-dt / tau)  # A = e^(−dt/τ)
        dynamics_noise = stationary_var * (
            1.0 - dynamics_matrix * dynamics_matrix
        )  # Q = Σ_stat (1 − A²)
        return cls(
            jnp.reshape(dynamics_matrix, (1, 1)),
            dynamics_noise=jnp.reshape(dynamics_noise, (1, 1)),
        )

    def _predict(
        self,
        message: CanonicalGaussian,
        control_term: ArrayLike | None = None,
    ) -> CanonicalGaussian:
        """Push an incoming belief on x through the dynamics to a belief on x'.

        The transition is the joint Gaussian over ``z = [x, x']``::

            Λ_J = [[ AᵀQ⁻¹A, −AᵀQ⁻¹ ],     h_J = [ −AᵀQ⁻¹b ,
                   [ −Q⁻¹A,    Q⁻¹   ]]            Q⁻¹b ]

        with ``b`` = ``control_term`` (the Bu shift; ``None`` → zero). The predict:

        1. Folds the incoming message into the x block — its precision into the
           top-left ``n×n`` of ``Λ_J``, its potential into the top ``n`` of ``h_J``
           (a block add during construction, *not* ``__add__``).
        2. Marginalizes x out, leaving the predicted message on x'.

        In moment form this lands exactly on ``cov_pred = AΣAᵀ + Q`` and
        ``mean_pred = Aμ + b``.

        Args:
            message: the incoming belief on x, as a ``CanonicalGaussian`` (n-D).
            control_term: b = Bu, shape ``(n,)``; ``None`` for an uncontrolled step.

        Returns:
            A ``CanonicalGaussian`` over the n-D next state x'.
        """
        dynamics_matrix, dynamics_noise = (
            self.dynamics_matrix,
            self.dynamics_noise,
        )  # A, Q
        n = dynamics_matrix.shape[0]
        # b = Bu, the control shift; None means no shift.
        if control_term is None:
            shift = jnp.zeros(n)
        else:
            shift = jnp.asarray(control_term, dtype=float)

        noise_precision = jnp.linalg.inv(dynamics_noise)  # Q⁻¹
        noise_weighted_dynamics = noise_precision @ dynamics_matrix  # Q⁻¹A
        # Joint precision over [x, x']: [[AᵀQ⁻¹A + Λ, −AᵀQ⁻¹], [−Q⁻¹A, Q⁻¹]], with
        # the incoming message's precision folded into the x (top-left) block.
        state_block = dynamics_matrix.T @ noise_weighted_dynamics + message.precision
        precision = jnp.block(
            [
                [state_block, -noise_weighted_dynamics.T],
                [-noise_weighted_dynamics, noise_precision],
            ]
        )
        # Joint potential [−AᵀQ⁻¹b + h, Q⁻¹b], message's potential folded into x.
        noise_weighted_shift = noise_precision @ shift  # Q⁻¹b
        state_potential = message.potential - dynamics_matrix.T @ noise_weighted_shift
        potential = jnp.concatenate([state_potential, noise_weighted_shift])

        joint = CanonicalGaussian._unchecked(precision, potential)
        return joint.marginalize(over=range(n))  # eliminate x, keep x'

    def tree_flatten(
        self,
    ) -> tuple[tuple[Float64[Array, "n n"], Float64[Array, "n n"]], None]:
        """Leaves for JAX: ``(dynamics, dynamics_noise)``, no static aux data."""
        return (self.dynamics_matrix, self.dynamics_noise), None

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: None,
        children: tuple[Float64[Array, "n n"], Float64[Array, "n n"]],
    ) -> "GaussianTransition":
        """Rebuild from leaves without validating — the leaves may be tracers."""
        dynamics_matrix, dynamics_noise = children
        obj = object.__new__(cls)
        object.__setattr__(obj, "dynamics_matrix", dynamics_matrix)
        object.__setattr__(obj, "dynamics_noise", dynamics_noise)
        return obj


def _require_positive(value: float, name: str) -> None:
    """Refuse a value that is not finite and positive. Skipped under a trace."""
    number = concrete(value)
    if number is not None and not (np.isfinite(number) and number > 0):
        raise ValueError(f"{name} must be a finite positive number, got {value}")
