"""Tier-1 linear-Gaussian factor nodes: the dynamics and the couplings (ADR-012).

Two factor types carry a graph's linear-Gaussian structure, and each one emits a
``CanonicalGaussian`` message assembled from the canonical-form algebra:

- ``GaussianTransition``, the dynamics ``N(x'; Ax + b, Q)``. Its forward predict builds
  the joint over ``[x, x']``, folds in the incoming message, and marginalizes x out
  (the Schur complement).
- ``GaussianCoupling``, the within-slice edge ``N(child; W·parent, Q)``. It sends a
  message up to the parent or down to the child the same way.

A node's observation is the flat model's sensor (ADR-071). Its message is
``likelihood_message`` in ``cpomdp.ffg._message``, built from the sensor's ``(C, R)``.

Both factors invert their noise covariance, so both require it positive-**definite**.
Unlike moment-form Kalman, the canonical transition factor cannot represent a
deterministic (``Q=0``) transition.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float64
from numpy.typing import ArrayLike

from cpomdp._validation import validate_covariance
from cpomdp.ffg._message import CanonicalGaussian

__all__ = ["GaussianCoupling", "GaussianTransition"]


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
        """
        a = jnp.exp(-dt / tau)  # A = e^(−dt/τ)
        q = stationary_var * (1.0 - a * a)  # Q = Σ_stat (1 − A²)
        return cls(jnp.reshape(a, (1, 1)), dynamics_noise=jnp.reshape(q, (1, 1)))

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


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class GaussianCoupling:
    """Tier-1 structural coupling factor ``N(child; W·parent, Q)`` — a graph edge.

    Where [`GaussianTransition`][cpomdp.GaussianTransition] couples a state to its
    *successor in time*, this couples two variables joined by an *edge of the factor
    graph* (e.g. the shared
    ``CheA`` node to a branch latent). The maths is identical — a linear-Gaussian
    coupling — but a coupling carries no time semantics and ``W`` need not be square.

    - ``coupling_matrix`` — W, shape ``(c, p)``: maps the p-D parent's mean to the c-D
      child.
    - ``coupling_noise`` — Q, shape ``(c, c)``, positive-definite (it is inverted).
    """

    coupling_matrix: Float64[Array, "c p"]  # W: child-rows × parent-cols
    coupling_noise: Float64[Array, "c c"]  # Q: child × child, positive-definite

    def __init__(
        self, coupling_matrix: ArrayLike, *, coupling_noise: ArrayLike
    ) -> None:
        object.__setattr__(
            self, "coupling_matrix", jnp.asarray(coupling_matrix, dtype=float)
        )
        object.__setattr__(
            self, "coupling_noise", jnp.asarray(coupling_noise, dtype=float)
        )
        self._validate()

    def _validate(self) -> None:
        coupling_matrix = self.coupling_matrix  # W
        coupling_noise = self.coupling_noise  # Q
        # Unlike GaussianTransition's square dynamics, the parent→child map W need
        # NOT be square — parent and child may differ in dimension.
        if coupling_matrix.ndim != 2:
            raise ValueError(
                f"coupling_matrix must be 2-D (c, p), got shape {coupling_matrix.shape}"
            )
        # Q is inverted in the message, so it must be positive-definite.
        validate_covariance(coupling_noise, "coupling_noise", require_definite=True)
        c = coupling_matrix.shape[0]
        if coupling_noise.shape != (c, c):
            raise ValueError(
                f"coupling_noise must be {c}x{c} to match the {c}-row coupling_matrix, "
                f"got shape {coupling_noise.shape}"
            )

    def _message_to_parent(self, child_message: CanonicalGaussian) -> CanonicalGaussian:
        """Summarise what a child's belief says about the parent: eliminate the child.

        The coupling is the joint Gaussian over ``z = [parent, child]``::

            Λ_J = [[ WᵀQ⁻¹W, −WᵀQ⁻¹ ],     h_J = 0   (a pure coupling has no bias)
                   [ −Q⁻¹W,    Q⁻¹   ]]

        The upward message:

        1. Folds ``child_message`` into the *child* block — its precision into the
           bottom-right ``c×c`` of ``Λ_J``, its potential into the trailing ``c`` of
           ``h_J`` (a block add during construction, *not* ``__add__``).
        2. Marginalizes the child out, leaving the message on the p-D parent.

        This is the mirror of ``GaussianTransition._predict`` (which folds into the
        parent block and eliminates the parent, emitting downward onto the child);
        here we fold into the child block and eliminate the child, emitting upward.

        Args:
            child_message: the incoming belief on the c-D child, as a
                ``CanonicalGaussian``.

        Returns:
            A ``CanonicalGaussian`` over the p-D parent.
        """
        coupling_matrix = self.coupling_matrix  # W
        coupling_noise = self.coupling_noise  # Q
        c, p = coupling_matrix.shape  # W is (child, parent)
        noise_precision = jnp.linalg.inv(coupling_noise)  # Q⁻¹
        noise_weighted_coupling = noise_precision @ coupling_matrix  # Q⁻¹W

        # The incoming message is on the CHILD, so it folds into the child block —
        # the mirror of predict, where the message folds into the parent (state) block.
        parent_block = coupling_matrix.T @ noise_weighted_coupling  # WᵀQ⁻¹W
        child_block = noise_precision + child_message.precision  # Q⁻¹ + message

        precision = jnp.block(
            [
                [parent_block, -noise_weighted_coupling.T],
                [-noise_weighted_coupling, child_block],
            ]
        )
        # No bias and no parent message → the parent potential is zero; the child
        # slot carries the incoming message's potential.
        potential = jnp.concatenate([jnp.zeros(p), child_message.potential])

        joint = CanonicalGaussian._unchecked(precision, potential)
        return joint.marginalize(over=range(p, p + c))  # eliminate child, keep parent

    def _message_to_child(self, parent_message: CanonicalGaussian) -> CanonicalGaussian:
        """Push a parent's belief down the edge onto the child: eliminate the parent.

        The distribute-pass mirror of ``_message_to_parent``. Over the same joint on
        ``z = [parent, child]``, but here the *parent* is known, so its message folds
        into the *parent* block of ``Λ_J`` and the parent is marginalized out, leaving
        the message on the c-D child.

        Structurally this is ``GaussianTransition._predict`` (fold the incoming belief
        into the source block, eliminate the source, emit onto the target) with a
        non-square ``W`` and no control shift — a pure coupling carries no bias. So on
        its own the downward message is a full child belief, landing in moment form on
        ``mean = W·μ_parent`` and ``cov = W·Σ_parent·Wᵀ + Q``.

        Args:
            parent_message: the incoming belief on the p-D parent, as a
                ``CanonicalGaussian``.

        Returns:
            A ``CanonicalGaussian`` over the c-D child.
        """
        coupling_matrix = self.coupling_matrix  # W
        coupling_noise = self.coupling_noise  # Q
        c, p = coupling_matrix.shape  # W is (child, parent)

        noise_precision = jnp.linalg.inv(coupling_noise)  # Q⁻¹
        noise_weighted_coupling = noise_precision @ coupling_matrix  # Q⁻¹W

        parent_block = (
            coupling_matrix.T @ noise_weighted_coupling + parent_message.precision
        )  # WᵀQ⁻¹W + message
        child_block = noise_precision

        precision = jnp.block(
            [
                [parent_block, -noise_weighted_coupling.T],
                [-noise_weighted_coupling, child_block],
            ]
        )

        potential = jnp.concatenate([parent_message.potential, jnp.zeros(c)])

        joint = CanonicalGaussian._unchecked(precision, potential)
        return joint.marginalize(over=range(p))

    def tree_flatten(
        self,
    ) -> tuple[tuple[Float64[Array, "c p"], Float64[Array, "c c"]], None]:
        """Leaves for JAX: ``(coupling_matrix, coupling_noise)``, no static aux data."""
        return (self.coupling_matrix, self.coupling_noise), None

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: None,
        children: tuple[Float64[Array, "c p"], Float64[Array, "c c"]],
    ) -> "GaussianCoupling":
        """Rebuild from leaves without validating — the leaves may be tracers."""
        coupling_matrix, coupling_noise = children
        obj = object.__new__(cls)
        object.__setattr__(obj, "coupling_matrix", coupling_matrix)
        object.__setattr__(obj, "coupling_noise", coupling_noise)
        return obj
