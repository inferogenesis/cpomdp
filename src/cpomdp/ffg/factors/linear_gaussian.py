"""Tier-1 linear-Gaussian factor nodes: the message producers (ADR-012, Phase 2).

Two factor types span a linear-Gaussian chain, and each one's job is to *emit a
``CanonicalGaussian`` message* assembled from the Phase 1 algebra:

- ``GaussianObservation`` — the likelihood ``N(y; Cx, R)``. Its message into x is
  the information form of the reading, ``(CᵀR⁻¹C, CᵀR⁻¹y)``; the measurement
  *update* is then ``belief + message`` (the factor product, ``__add__``).
- ``GaussianTransition`` — the dynamics ``N(x'; Ax + b, Q)``. Its forward
  *predict* builds the joint over ``[x, x']``, folds in the incoming message, and
  marginalizes x out (the Schur complement).

These nodes are thin: the heavy lifting (add, marginalize, readout) already lives
in ``CanonicalGaussian``. A linear chain of them reproduces the Kalman filter —
the Phase 2 keystone gate.

Note (information-form constraint): both factors invert their noise covariance
(``R⁻¹``, ``Q⁻¹``), so both require it positive-**definite**. Unlike moment-form
Kalman, the canonical transition factor cannot represent a deterministic (``Q=0``)
transition — a real divergence to keep in mind, harmless for the PD-noise chain
the keystone uses.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float64
from numpy.typing import ArrayLike

from cpomdp._validation import validate_covariance
from cpomdp.ffg.message import CanonicalGaussian

__all__ = ["GaussianCoupling", "GaussianObservation", "GaussianTransition"]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class GaussianObservation:
    """Tier-1 likelihood factor ``N(y; Cx, R)`` — emits a message into the state.

    Holds the fixed sensor map and noise; ``message(y)`` turns a reading into its
    canonical-form contribution to the belief on x.

    - ``sensor_model`` — C, shape ``(m, n)``.
    - ``sensor_noise`` — R, shape ``(m, m)``, positive-definite (it is inverted).
    """

    sensor_model: Float64[Array, "m, n"]
    sensor_noise: Float64[Array, "m, m"]

    def __init__(self, sensor_model: ArrayLike, sensor_noise: ArrayLike) -> None:
        object.__setattr__(self, "sensor_model", jnp.asarray(sensor_model, dtype=float))
        object.__setattr__(self, "sensor_noise", jnp.asarray(sensor_noise, dtype=float))
        self._validate()

    def _validate(self) -> None:
        sensor_model, sensor_noise = self.sensor_model, self.sensor_noise  # C, R
        if sensor_model.ndim != 2:
            raise ValueError(
                f"sensor_model must be 2-D (m, n), got shape {sensor_model.shape}"
            )
        # R is inverted in the message, so it must be positive-definite.
        validate_covariance(sensor_noise, "sensor_noise", require_definite=True)
        m = sensor_model.shape[0]
        if sensor_noise.shape != (m, m):
            raise ValueError(
                f"sensor_noise must be {m}x{m} to match the {m}-row sensor_model, "
                f"got shape {sensor_noise.shape}"
            )

    def message(self, observation: ArrayLike) -> CanonicalGaussian:
        """The likelihood's message into x: ``Λ = CᵀR⁻¹C``, ``h = CᵀR⁻¹y``.

        The information form of the reading — the evidence the observation injects
        about the state. The measurement update is then ``prior_message + this``
        (``CanonicalGaussian.__add__``). A solve against R avoids forming ``R⁻¹``;
        the result is valid by construction, so it builds via the no-validate seam.

        Args:
            observation: the reading y, shape ``(m,)``.

        Returns:
            A ``CanonicalGaussian`` over the n-D state — precision ``(n, n)``,
            potential ``(n,)``.
        """
        sensor_model, sensor_noise = self.sensor_model, self.sensor_noise  # C, R
        reading = jnp.asarray(observation, dtype=float)  # y
        # Λ = CᵀR⁻¹C, h = CᵀR⁻¹y — solved against R rather than forming R⁻¹.
        noise_weighted_model = jnp.linalg.solve(sensor_noise, sensor_model)  # R⁻¹C
        precision = sensor_model.T @ noise_weighted_model  # CᵀR⁻¹C
        potential = sensor_model.T @ jnp.linalg.solve(sensor_noise, reading)  # CᵀR⁻¹y
        return CanonicalGaussian._unchecked(precision, potential)

    def tree_flatten(
        self,
    ) -> tuple[tuple[Float64[Array, "m, n"], Float64[Array, "m, m"]], None]:
        """Leaves for JAX: ``(sensor_model, sensor_noise)``, no static aux data."""
        return (self.sensor_model, self.sensor_noise), None

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: None,
        children: tuple[Float64[Array, "m, n"], Float64[Array, "m, m"]],
    ) -> "GaussianObservation":
        """Rebuild from leaves without validating — the leaves may be tracers."""
        sensor_model, sensor_noise = children
        obj = object.__new__(cls)
        object.__setattr__(obj, "sensor_model", sensor_model)
        object.__setattr__(obj, "sensor_noise", sensor_noise)
        return obj


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class GaussianTransition:
    """Tier-1 dynamics factor ``N(x'; Ax + b, Q)`` — emits the forward predict.

    Holds the fixed transition and process noise; ``predict(message, b)`` pushes a
    belief on x through the dynamics to a belief on x'.

    - ``dynamics`` — A, shape ``(n, n)``.
    - ``dynamics_noise`` — Q, shape ``(n, n)``, positive-definite (it is inverted).
    """

    dynamics: Float64[Array, "n, n"]
    dynamics_noise: Float64[Array, "n, n"]

    def __init__(self, dynamics: ArrayLike, dynamics_noise: ArrayLike) -> None:
        object.__setattr__(self, "dynamics", jnp.asarray(dynamics, dtype=float))
        object.__setattr__(
            self, "dynamics_noise", jnp.asarray(dynamics_noise, dtype=float)
        )
        self._validate()

    def _validate(self) -> None:
        dynamics, dynamics_noise = self.dynamics, self.dynamics_noise  # A, Q
        if dynamics.ndim != 2 or dynamics.shape[0] != dynamics.shape[1]:
            raise ValueError(
                f"dynamics must be square (n, n), got shape {dynamics.shape}"
            )
        # Q is inverted in the joint, so it must be positive-definite.
        validate_covariance(dynamics_noise, "dynamics_noise", require_definite=True)
        n = dynamics.shape[0]
        if dynamics_noise.shape != (n, n):
            raise ValueError(
                f"dynamics_noise must be {n}x{n} to match the {n}-D state, "
                f"got shape {dynamics_noise.shape}"
            )

    def predict(
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
        dynamics, dynamics_noise = self.dynamics, self.dynamics_noise  # A, Q
        n = dynamics.shape[0]
        # b = Bu, the control shift; None means no shift.
        if control_term is None:
            shift = jnp.zeros(n)
        else:
            shift = jnp.asarray(control_term, dtype=float)

        noise_precision = jnp.linalg.inv(dynamics_noise)  # Q⁻¹
        noise_weighted_dynamics = noise_precision @ dynamics  # Q⁻¹A
        # Joint precision over [x, x']: [[AᵀQ⁻¹A + Λ, −AᵀQ⁻¹], [−Q⁻¹A, Q⁻¹]], with
        # the incoming message's precision folded into the x (top-left) block.
        state_block = dynamics.T @ noise_weighted_dynamics + message.precision
        precision = jnp.block(
            [
                [state_block, -noise_weighted_dynamics.T],
                [-noise_weighted_dynamics, noise_precision],
            ]
        )
        # Joint potential [−AᵀQ⁻¹b + h, Q⁻¹b], message's potential folded into x.
        noise_weighted_shift = noise_precision @ shift  # Q⁻¹b
        state_potential = message.potential - dynamics.T @ noise_weighted_shift
        potential = jnp.concatenate([state_potential, noise_weighted_shift])

        joint = CanonicalGaussian._unchecked(precision, potential)
        return joint.marginalize(over=range(n))  # eliminate x, keep x'

    def tree_flatten(
        self,
    ) -> tuple[tuple[Float64[Array, "n, n"], Float64[Array, "n, n"]], None]:
        """Leaves for JAX: ``(dynamics, dynamics_noise)``, no static aux data."""
        return (self.dynamics, self.dynamics_noise), None

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: None,
        children: tuple[Float64[Array, "n, n"], Float64[Array, "n, n"]],
    ) -> "GaussianTransition":
        """Rebuild from leaves without validating — the leaves may be tracers."""
        dynamics, dynamics_noise = children
        obj = object.__new__(cls)
        object.__setattr__(obj, "dynamics", dynamics)
        object.__setattr__(obj, "dynamics_noise", dynamics_noise)
        return obj


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class GaussianCoupling:
    """Tier-1 structural coupling factor ``N(child; W·parent, Q)`` — a graph edge.

    Where ``GaussianTransition`` couples a state to its *successor in time*, this
    couples two variables joined by an *edge of the factor graph* (e.g. the shared
    ``CheA`` node to a branch latent). The maths is identical — a linear-Gaussian
    coupling — but a coupling carries no time semantics and ``W`` need not be square.

    - ``coupling`` — W, shape ``(c, p)``: maps the p-D parent's mean to the c-D child.
    - ``coupling_noise`` — Q, shape ``(c, c)``, positive-definite (it is inverted).
    """

    coupling: Float64[Array, "c, p"]  # W: child-rows × parent-cols
    coupling_noise: Float64[Array, "c, c"]  # Q: child × child, positive-definite

    def __init__(self, coupling: ArrayLike, coupling_noise: ArrayLike) -> None:
        object.__setattr__(self, "coupling", jnp.asarray(coupling, dtype=float))
        object.__setattr__(
            self, "coupling_noise", jnp.asarray(coupling_noise, dtype=float)
        )
        self._validate()

    def _validate(self) -> None:
        coupling, coupling_noise = self.coupling, self.coupling_noise  # W, Q
        # Unlike GaussianTransition's square dynamics, the parent→child map W need
        # NOT be square — parent and child may differ in dimension.
        if coupling.ndim != 2:
            raise ValueError(f"coupling must be 2-D (c, p), got shape {coupling.shape}")
        # Q is inverted in the message, so it must be positive-definite.
        validate_covariance(coupling_noise, "coupling_noise", require_definite=True)
        c = coupling.shape[0]
        if coupling_noise.shape != (c, c):
            raise ValueError(
                f"coupling_noise must be {c}x{c} to match the {c}-row coupling, "
                f"got shape {coupling_noise.shape}"
            )

    def message_to_parent(self, child_message: CanonicalGaussian) -> CanonicalGaussian:
        """Summarise what a child's belief says about the parent: eliminate the child.

        The coupling is the joint Gaussian over ``z = [parent, child]``::

            Λ_J = [[ WᵀQ⁻¹W, −WᵀQ⁻¹ ],     h_J = 0   (a pure coupling has no bias)
                   [ −Q⁻¹W,    Q⁻¹   ]]

        The upward message:

        1. Folds ``child_message`` into the *child* block — its precision into the
           bottom-right ``c×c`` of ``Λ_J``, its potential into the trailing ``c`` of
           ``h_J`` (a block add during construction, *not* ``__add__``).
        2. Marginalizes the child out, leaving the message on the p-D parent.

        This is the mirror of ``GaussianTransition.predict`` (which folds into the
        parent block and eliminates the parent, emitting downward onto the child);
        here we fold into the child block and eliminate the child, emitting upward.

        Args:
            child_message: the incoming belief on the c-D child, as a
                ``CanonicalGaussian``.

        Returns:
            A ``CanonicalGaussian`` over the p-D parent.
        """
        coupling, coupling_noise = self.coupling, self.coupling_noise  # W, Q
        c, p = coupling.shape  # W is (child, parent)
        noise_precision = jnp.linalg.inv(coupling_noise)  # Q⁻¹
        noise_weighted_coupling = noise_precision @ coupling  # Q⁻¹W

        # The incoming message is on the CHILD, so it folds into the child block —
        # the mirror of predict, where the message folds into the parent (state) block.
        parent_block = coupling.T @ noise_weighted_coupling  # WᵀQ⁻¹W
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

    def tree_flatten(
        self,
    ) -> tuple[tuple[Float64[Array, "c, p"], Float64[Array, "c, c"]], None]:
        """Leaves for JAX: ``(coupling, coupling_noise)``, no static aux data."""
        return (self.coupling, self.coupling_noise), None

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: None,
        children: tuple[Float64[Array, "c, p"], Float64[Array, "c, c"]],
    ) -> "GaussianCoupling":
        """Rebuild from leaves without validating — the leaves may be tracers."""
        coupling, coupling_noise = children
        obj = object.__new__(cls)
        object.__setattr__(obj, "coupling", coupling)
        object.__setattr__(obj, "coupling_noise", coupling_noise)
        return obj
