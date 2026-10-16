"""Transition kernels evaluated pairwise, as ``log p(x' | x, u)`` between nodes.

The dynamics half of the exact filter, and the expensive half. Conditioning touches
each node once; predicting has to know how much mass moves from every node to every
other one, so it asks for a square matrix of log-densities rather than a vector.

That matrix is what makes a grid filter costly, and the cost is quadratic in the node
count rather than linear. It is returned to the caller instead of being built inside
the prediction, so a trajectory that reuses one kernel builds it once and a reader
can see where the work went.

``LinearGaussianKernel`` is the mean-linear, fixed-noise case, factored once at
construction. The protocol asks only for a log-density, so a kernel whose noise
varies with the state satisfies it too. Nothing here is written against one, and none
of the surface claims support for it (issue #56).
"""

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from jaxtyping import Array, Float64
from numpy.typing import ArrayLike

from cpomdp._validation import validate_covariance

__all__ = ["LinearGaussianKernel", "TransitionKernel"]

_LOG_TWO_PI = math.log(2.0 * math.pi)


@runtime_checkable
class TransitionKernel(Protocol):
    """``log p(x' | x, u)`` between two arrays of states, evaluated in one pass.

    The result is indexed ``[destination, origin]``, which is the orientation the
    prediction contracts over: summing across the origin axis is the integral
    ``∫ p(x' | x) p(x) dx``.

    Attributes:
        is_fixed: ``True`` when the noise does not vary with the state. Nothing
            branches on it. It records which regime a result came from.
    """

    is_fixed: bool

    def log_transition(
        self,
        destinations: ArrayLike,
        origins: ArrayLike,
        action: ArrayLike | None = None,
    ) -> Float64[Array, "M N"]:
        """``log p(destination | origin, action)`` for every pair, destination-major."""
        ...


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class LinearGaussianKernel:
    """``log N(x'; A x + B u, Q)``, the mean-linear kernel with constant noise.

    The regime where the Kalman prediction is exact, so it is what the grid
    prediction is checked against before anything else is asked of it.

    The noise is factored once at construction. What a call still pays is the
    ``M x N x n`` array of differences, which is the quadratic cost and is not
    avoidable by front-loading.

    Attributes:
        dynamics_matrix: the state-transition matrix ``A`` (shape ``n x n``).
        dynamics_noise: the process-noise covariance ``Q`` (shape ``n x n``),
            positive-definite because the density inverts it. A deterministic
            transition has no density against Lebesgue measure and is refused here
            rather than approximated by a small ``Q`` chosen for the caller.
        control_matrix: the input matrix ``B`` (shape ``n x p``), or ``None`` for a
            model with no control.
        noise_cholesky: the lower Cholesky factor of ``Q``, built at construction.
        log_det_noise: ``log det Q``, built at construction from that factor.
    """

    dynamics_matrix: Float64[Array, "n n"]  # A
    dynamics_noise: Float64[Array, "n n"]  # Q
    control_matrix: Float64[Array, "n p"] | None  # B
    noise_cholesky: Float64[Array, "n n"]
    log_det_noise: Float64[Array, ""]
    is_fixed = True

    def __init__(
        self,
        dynamics_matrix: ArrayLike,
        *,
        dynamics_noise: ArrayLike,
        control_matrix: ArrayLike | None = None,
    ) -> None:
        """Store ``(A, Q, B)`` and factor ``Q`` for the per-call path.

        Args:
            dynamics_matrix: the state-transition matrix ``A``.
            dynamics_noise: the process-noise covariance ``Q``. Keyword-only, so a
                transposed pair cannot construct in silence.
            control_matrix: the input matrix ``B``, or ``None``.

        Raises:
            ValueError: if ``A`` is not square, if ``Q`` is not a positive-definite
                covariance of ``A``'s size, or if ``B`` does not map into the state.
        """
        object.__setattr__(
            self, "dynamics_matrix", jnp.asarray(dynamics_matrix, dtype=float)
        )
        object.__setattr__(
            self, "dynamics_noise", jnp.asarray(dynamics_noise, dtype=float)
        )
        object.__setattr__(
            self,
            "control_matrix",
            None
            if control_matrix is None
            else jnp.asarray(control_matrix, dtype=float),
        )
        self._validate()
        cholesky = jnp.linalg.cholesky(self.dynamics_noise)
        object.__setattr__(self, "noise_cholesky", cholesky)
        object.__setattr__(
            self, "log_det_noise", 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky)))
        )

    def _validate(self) -> None:
        if (
            self.dynamics_matrix.ndim != 2
            or self.dynamics_matrix.shape[0] != self.dynamics_matrix.shape[1]
        ):
            raise ValueError(
                "dynamics_matrix must be a square n x n matrix, got shape "
                f"{self.dynamics_matrix.shape}"
            )
        validate_covariance(
            self.dynamics_noise, "dynamics_noise", require_definite=True
        )
        n = self.dynamics_matrix.shape[0]
        if self.dynamics_noise.shape != (n, n):
            raise ValueError(
                f"dynamics_noise must be {n}x{n} to match dynamics_matrix, got shape "
                f"{self.dynamics_noise.shape}"
            )
        if self.control_matrix is not None and (
            self.control_matrix.ndim != 2 or self.control_matrix.shape[0] != n
        ):
            raise ValueError(
                f"control_matrix must be an {n} x p matrix mapping an action into "
                f"the state, got shape {self.control_matrix.shape}"
            )

    @property
    def ndim(self) -> int:
        """Dimensionality of the state the kernel moves."""
        return self.dynamics_matrix.shape[0]

    def log_transition(
        self,
        destinations: ArrayLike,
        origins: ArrayLike,
        action: ArrayLike | None = None,
    ) -> Float64[Array, "M N"]:
        """``log N(destination; A·origin + B·action, Q)`` for every pair.

        Args:
            destinations: the ``M x n`` states mass arrives at.
            origins: the ``N x n`` states it leaves from.
            action: the ``p``-vector applied on this step, or ``None`` for a kernel
                with no control matrix.

        Returns:
            The log-densities, shape ``(M, N)``, indexed destination-major.

        Raises:
            ValueError: if either array is not ``· x n``, if the kernel has a control
                matrix and no action was given, or if the action is the wrong length.
        """
        destinations = _as_states(destinations, self.ndim, "destinations")
        origins = _as_states(origins, self.ndim, "origins")
        means = origins @ self.dynamics_matrix.T
        means = means + self._control_offset(action)

        differences = destinations[:, None, :] - means[None, :, :]
        flattened = differences.reshape(-1, self.ndim)
        whitened = solve_triangular(self.noise_cholesky, flattened.T, lower=True).T
        quadratic = jnp.sum(whitened**2, axis=-1).reshape(differences.shape[:2])
        return -0.5 * (self.ndim * _LOG_TWO_PI + self.log_det_noise + quadratic)

    def _control_offset(self, action: ArrayLike | None) -> Float64[Array, "n"]:
        """``B·u`` as a row to add to every mean, or zero for a control-free kernel."""
        if self.control_matrix is None:
            if action is not None:
                raise ValueError(
                    "this kernel has no control_matrix; log_transition takes no action"
                )
            return jnp.zeros(self.ndim)
        if action is None:
            raise ValueError(
                "this kernel has a control_matrix; log_transition requires an action"
            )
        action = jnp.asarray(action, dtype=float)
        p = self.control_matrix.shape[1]
        if action.shape != (p,):
            raise ValueError(
                f"action must be a 1-D vector of length {p} (the action dimension), "
                f"got shape {action.shape}"
            )
        return self.control_matrix @ action

    def tree_flatten(self) -> tuple[tuple[Float64[Array, "..."] | None, ...], bool]:
        """Leaves: the matrices and the two values derived from the noise.

        Whether a control matrix is present is static, since it changes the shape of
        every call rather than its value.
        """
        return (
            self.dynamics_matrix,
            self.dynamics_noise,
            self.control_matrix,
            self.noise_cholesky,
            self.log_det_noise,
        ), self.control_matrix is not None

    @classmethod
    def tree_unflatten(
        cls, aux_data: bool, children: tuple[Float64[Array, "..."] | None, ...]
    ) -> "LinearGaussianKernel":
        """Rebuild from leaves without re-validating or re-factoring."""
        kernel = object.__new__(cls)
        names = (
            "dynamics_matrix",
            "dynamics_noise",
            "control_matrix",
            "noise_cholesky",
            "log_det_noise",
        )
        for name, value in zip(names, children, strict=True):
            object.__setattr__(kernel, name, value)
        return kernel


def _as_states(states: ArrayLike, ndim: int, name: str) -> Float64[Array, "N n"]:
    """Coerce and shape-check one side of the pair, at the trust boundary."""
    states = jnp.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[-1] != ndim:
        raise ValueError(
            f"{name} must be an N x {ndim} array to match the state dimension, "
            f"got shape {states.shape}"
        )
    return states
