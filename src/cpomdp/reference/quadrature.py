"""A uniform quadrature lattice and the log-densities carried on it.

``QuadratureGrid`` is a tensor product of evenly spaced nodes over a box the caller
declares, carrying composite-trapezoid weights computed once at construction.
``GridDensity`` is a vector of log-density values on such a grid, with the moments
and the divergence read off by weighted summation.

The box is an argument, never a rule the grid picks. A half-width of "k standard
deviations" sizes a Gaussian tail, and the predictive densities this substrate exists
to integrate are scale mixtures whose tails are exponential, so no k is safe on
principle (``research.checks.predictive_truncation`` measures where a given one
fails). Sizing belongs to the caller who knows the density, and what falls outside
the box is reported rather than absorbed.

That reporting is why construction does not normalise. A density built on a box too
small integrates to less than one, and dividing that away at construction would erase
the only place the deficit is visible. ``GridDensity.log_mass`` asks what the grid
integral actually is, and ``normalise`` records on its result what it divided out.
"""

import functools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.errors import ConcretizationTypeError, TracerArrayConversionError
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float64
from numpy.typing import ArrayLike

from cpomdp._validation import concrete, validate_covariance

__all__ = ["GridDensity", "QuadratureGrid", "gaussian_on"]


def _validate_log_values(values: Float64[Array, "N"], name: str) -> None:
    """Reject NaN and ``+inf`` in a log-density, allowing ``-inf``.

    A log-density is ``-inf`` wherever the density vanishes, which is ordinary and
    must construct. ``+inf`` and NaN are not: both survive the weighted sum and turn
    every moment into NaN, several steps from wherever they came in.

    Skipped under a trace via the ``np.asarray`` guard, matching the covariance and
    finiteness checks in ``cpomdp._validation``.
    """
    try:
        concrete = np.asarray(values, dtype=float)
    except (TracerArrayConversionError, ConcretizationTypeError):
        return
    if bool(np.isnan(concrete).any() or (concrete == np.inf).any()):
        raise ValueError(
            f"{name} must contain no NaN and no +inf (-inf is allowed, and means "
            "the density vanishes at that node)."
        )


def _contract(
    node_weights: Float64[Array, "N"], values: Float64[Array, "N ..."]
) -> Float64[Array, "..."]:
    """Contract a per-node weight vector against per-node values.

    The one place that knows the convention: the node axis leads, and whatever
    trailing axes ``values`` carries survive into the result. A vector or matrix
    integrand therefore integrates componentwise in a single pass.

    Both integrations in this module route through here and differ only in the
    weights they bring. The grid brings its quadrature weights, and a density brings
    those times its own normalised mass.
    """
    return jnp.tensordot(node_weights, jnp.asarray(values, dtype=float), axes=1)


def _trapezoid_weights(lower: float, upper: float, count: int) -> Float64[Array, "n"]:
    """One axis of composite-trapezoid weights: the spacing, halved at both ends."""
    spacing = (upper - lower) / (count - 1)
    return jnp.full((count,), spacing).at[0].multiply(0.5).at[-1].multiply(0.5)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class QuadratureGrid:
    """Evenly spaced nodes over a declared box, with composite-trapezoid weights.

    The nodes are the tensor product of ``counts[i]`` points spanning
    ``[lower[i], upper[i]]`` inclusive, flattened so the last axis varies fastest.
    The weights are the matching tensor product of the one-dimensional trapezoid
    rule: the spacing everywhere except each axis's two endpoints, where it is half.

    Nodes and weights are built once here, so integrating is a contraction and
    carries no per-call setup.

    Attributes:
        nodes: the lattice points, shape ``(N, d)`` for ``N = prod(counts)``.
        weights: the quadrature weight at each node, shape ``(N,)``. They sum to the
            box's volume, which is what integrating the constant one gives.
        lower: the box's lower corner, one float per axis.
        upper: the box's upper corner, one float per axis.
        counts: nodes per axis. Two or more, since a single point has no spacing and
            the trapezoid rule is undefined on it.

    The rule is fixed rather than selected. A second rule would arrive as a sibling
    type offering the same ``nodes``, ``weights`` and ``integrate``, not as a switch
    here, since the two would not share a constructor.
    """

    nodes: Float64[Array, "N d"]
    weights: Float64[Array, "N"]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    counts: tuple[int, ...]

    def __init__(
        self,
        lower: Sequence[float],
        upper: Sequence[float],
        counts: Sequence[int],
    ) -> None:
        """Build the lattice and its weights over the box the arguments declare.

        Args:
            lower: lower corner, one entry per axis.
            upper: upper corner, one entry per axis, each above its ``lower``.
            counts: nodes per axis, each at least 2.

        Raises:
            ValueError: if the three do not have one entry per axis, if there are no
                axes, if any axis has fewer than two nodes, or if any axis is empty
                or inverted.
        """
        object.__setattr__(self, "lower", tuple(float(value) for value in lower))
        object.__setattr__(self, "upper", tuple(float(value) for value in upper))
        object.__setattr__(self, "counts", tuple(int(value) for value in counts))
        self._validate()

        mesh = jnp.meshgrid(
            *(jnp.linspace(lo, hi, n) for lo, hi, n in self._axes()), indexing="ij"
        )
        object.__setattr__(
            self, "nodes", jnp.stack([axis.ravel() for axis in mesh], axis=-1)
        )
        object.__setattr__(
            self,
            "weights",
            functools.reduce(
                lambda outer, axis: (outer[:, None] * axis[None, :]).ravel(),
                [_trapezoid_weights(lo, hi, n) for lo, hi, n in self._axes()],
            ),
        )

    def _axes(self) -> list[tuple[float, float, int]]:
        """The per-axis ``(lower, upper, count)`` triples, in axis order."""
        return list(zip(self.lower, self.upper, self.counts, strict=True))

    def _validate(self) -> None:
        if not (len(self.lower) == len(self.upper) == len(self.counts)):
            raise ValueError(
                "lower, upper and counts must have one entry per axis, got lengths "
                f"{len(self.lower)}, {len(self.upper)}, {len(self.counts)}"
            )
        if not self.lower:
            raise ValueError("a grid needs at least one axis")
        for axis, (lo, hi, n) in enumerate(self._axes()):
            if n < 2:
                raise ValueError(
                    f"axis {axis} has {n} nodes; the trapezoid rule needs at least 2"
                )
            if hi <= lo:
                raise ValueError(
                    f"axis {axis} spans [{lo}, {hi}]; upper must exceed lower"
                )

    @property
    def ndim(self) -> int:
        """Dimensionality of the latent the grid covers."""
        return len(self.counts)

    @property
    def size(self) -> int:
        """Number of nodes, the product of ``counts``."""
        return math.prod(self.counts)

    @property
    def spacing(self) -> tuple[float, ...]:
        """Node spacing per axis, ``(upper - lower) / (count - 1)``."""
        return tuple((hi - lo) / (n - 1) for lo, hi, n in self._axes())

    @property
    def box(self) -> tuple[tuple[float, float], ...]:
        """The declared box as one ``(lower, upper)`` pair per axis."""
        return tuple(zip(self.lower, self.upper, strict=True))

    def integrate(self, values: Float64[Array, "N ..."]) -> Float64[Array, "..."]:
        """Quadrature of ``values`` over the box: the weights contracted with them.

        This integrates against the box, taking no view on whether the values are a
        density. ``GridDensity.expectation`` is the one that integrates against a
        normalised measure, and the two answer differently for the same input.

        Args:
            values: one value per node along the leading axis. Trailing axes are
                carried through, so a vector or matrix integrand integrates
                componentwise in one pass.

        Returns:
            The integral, shaped like ``values`` with the node axis removed.
        """
        return _contract(self.weights, values)

    def same_lattice_as(self, other: "QuadratureGrid") -> bool:
        """Whether ``other`` declares the same box at the same resolution."""
        return (
            self.lower == other.lower
            and self.upper == other.upper
            and self.counts == other.counts
        )

    def __repr__(self) -> str:
        """Box and resolution, the two things that identify a grid."""
        return (
            f"QuadratureGrid(lower={self.lower}, upper={self.upper}, "
            f"counts={self.counts})"
        )

    def tree_flatten(
        self,
    ) -> tuple[
        tuple[Float64[Array, "N d"], Float64[Array, "N"]],
        tuple[tuple[float, ...], tuple[float, ...], tuple[int, ...]],
    ]:
        """Leaves ``(nodes, weights)``; aux the box and the counts.

        The box is static because it is a declaration rather than a quantity. Nothing
        differentiates with respect to where the grid was placed, and keeping it out
        of the leaves is what lets two grids be compared for identity under a trace.
        """
        return (self.nodes, self.weights), (self.lower, self.upper, self.counts)

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: tuple[tuple[float, ...], tuple[float, ...], tuple[int, ...]],
        children: tuple[Float64[Array, "N d"], Float64[Array, "N"]],
    ) -> "QuadratureGrid":
        """Rebuild from leaves without re-validating or rebuilding the lattice."""
        grid = object.__new__(cls)
        nodes, weights = children
        lower, upper, counts = aux_data
        object.__setattr__(grid, "nodes", nodes)
        object.__setattr__(grid, "weights", weights)
        object.__setattr__(grid, "lower", lower)
        object.__setattr__(grid, "upper", upper)
        object.__setattr__(grid, "counts", counts)
        return grid


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class GridDensity:
    """A log-density sampled on a ``QuadratureGrid``.

    Held in logs because the densities this integrates span the range that makes a
    linear representation lose the tails outright, and the tails are where the
    truncation question lives.

    A density is not normalised on construction and is not required to integrate to
    one. ``log_mass`` reports what it does integrate to, and is how a box too small
    for its density shows up. ``normalise`` returns a density integrating to one that
    carries the discarded mass forward in ``log_normaliser``, so the record survives
    the caller dropping the original.

    Moments and divergences normalise internally, so they are right on an
    unnormalised density without changing it.

    Attributes:
        grid: the lattice the values sit on.
        log_density: natural log of the density at each node, shape ``(N,)``.
            ``-inf`` is allowed and means the density vanishes there.
        log_normaliser: log of the mass already divided out by earlier calls to
            ``normalise``. Zero on a density that has not been normalised.
    """

    grid: QuadratureGrid
    log_density: Float64[Array, "N"]
    log_normaliser: Float64[Array, ""]

    def __init__(
        self,
        grid: QuadratureGrid,
        log_density: ArrayLike,
        log_normaliser: ArrayLike = 0.0,
    ) -> None:
        """Carry ``log_density`` on ``grid`` without normalising it.

        Args:
            grid: the lattice the values were evaluated on.
            log_density: one log-density value per node, in the grid's node order.
            log_normaliser: log of the mass already divided out, for a density that
                came from ``normalise``. Defaults to zero.

        Raises:
            ValueError: if the values are not a 1-D vector of length ``grid.size``,
                or contain NaN or ``+inf``.
        """
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "log_density", jnp.asarray(log_density, dtype=float))
        object.__setattr__(
            self, "log_normaliser", jnp.asarray(log_normaliser, dtype=float)
        )
        self._validate()

    def _validate(self) -> None:
        if self.log_density.ndim != 1:
            raise ValueError(
                "log_density must be a 1-D vector, one value per node, got shape "
                f"{self.log_density.shape}"
            )
        if self.log_density.shape != (self.grid.size,):
            raise ValueError(
                f"log_density must have {self.grid.size} entries to match the grid, "
                f"got {self.log_density.shape[0]}"
            )
        _validate_log_values(self.log_density, "log_density")

    @property
    def log_mass(self) -> Float64[Array, ""]:
        """Log of the quadrature integral over the box, as the values stand.

        Zero to within quadrature error on a normalised density. Below zero on a
        density whose box clipped part of it away, and that shortfall is the
        truncation this substrate exists to keep visible.
        """
        return logsumexp(jnp.log(self.grid.weights) + self.log_density)

    def normalise(self) -> "GridDensity":
        """A density integrating to one, carrying what was divided out.

        Returns:
            A new ``GridDensity`` on the same grid whose ``log_mass`` is zero, with
            ``log_normaliser`` raised by the mass this call removed. Calling it on an
            already normalised density changes nothing beyond quadrature error.
        """
        log_mass = self.log_mass
        return GridDensity(
            self.grid,
            self.log_density - log_mass,
            self.log_normaliser + log_mass,
        )

    def _node_measure(self) -> Float64[Array, "N"]:
        """Probability mass at each node: quadrature weight times normalised density.

        The measure an expectation integrates against. Whoever needs it more than
        once binds it once, since it costs a pass over every node.
        """
        return self.grid.weights * jnp.exp(self.log_density - self.log_mass)

    def expectation(self, values: Float64[Array, "N ..."]) -> Float64[Array, "..."]:
        """Expectation of ``values`` under the density, normalising internally.

        Same calling convention as ``QuadratureGrid.integrate`` and the same
        contraction, differing only in the weights. This one folds in the normalised
        density, which is the division a caller reaching for the grid directly would
        have to remember and would silently get wrong on a clipped box.

        Args:
            values: the integrand at each node, in the grid's node order, along the
                leading axis. Trailing axes are carried through, so a vector or
                matrix integrand comes back with its shape.

        Returns:
            The expectation, shaped like ``values`` with the node axis removed.
        """
        return _contract(self._node_measure(), values)

    @property
    def mean(self) -> Float64[Array, "d"]:
        """First moment under the density, a vector of length ``grid.ndim``."""
        return self.expectation(self.grid.nodes)

    @property
    def cov(self) -> Float64[Array, "d d"]:
        """Central second moment under the density, ``grid.ndim`` square.

        Contracts against the measure twice off one binding of it, rather than going
        back through ``mean`` and paying for the normalisation a second time.
        """
        measure = self._node_measure()
        mean = _contract(measure, self.grid.nodes)
        centred = self.grid.nodes - mean
        return _contract(measure, centred[:, :, None] * centred[:, None, :])

    def kl_to(self, other: "GridDensity") -> Float64[Array, ""]:
        """``D_KL[self ‖ other]`` by quadrature, with both normalised first.

        A KL divergence is an expectation of a log-ratio, and is computed as one.
        Each mass is taken once. Normalising and then calling ``expectation`` would
        take this density's twice, and the sweep pays this per observation node.

        Nodes where this density vanishes are already weighted to zero by the
        measure. The guard is on the subtraction, where ``-inf - -inf`` is NaN and
        would poison the whole contraction. A node where ``other`` vanishes and this
        one does not sends the result to ``+inf``, which is the divergence rather
        than a failure.

        Args:
            other: a density on the same lattice.

        Returns:
            The divergence, a scalar.

        Raises:
            ValueError: if ``other`` sits on a different box or resolution. The two
                integrands would be sampled at different points, and the difference
                of their logs would not be a divergence.
        """
        if not self.grid.same_lattice_as(other.grid):
            raise ValueError(
                "kl_to needs both densities on the same lattice, got "
                f"{self.grid!r} and {other.grid!r}"
            )
        log_p = self.log_density - self.log_mass
        log_q = other.log_density - other.log_mass
        integrand = jnp.where(jnp.isneginf(log_p), 0.0, log_p - log_q)
        return _contract(self.grid.weights * jnp.exp(log_p), integrand)

    def tree_flatten(
        self,
    ) -> tuple[
        tuple[QuadratureGrid, Float64[Array, "N"], Float64[Array, ""]],
        None,
    ]:
        """Leaves ``(grid, log_density, log_normaliser)``, with no static aux data."""
        return (self.grid, self.log_density, self.log_normaliser), None

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: None,
        children: tuple[QuadratureGrid, Float64[Array, "N"], Float64[Array, ""]],
    ) -> "GridDensity":
        """Rebuild from leaves without validating — the leaves may be tracers."""
        density = object.__new__(cls)
        grid, log_density, log_normaliser = children
        object.__setattr__(density, "grid", grid)
        object.__setattr__(density, "log_density", log_density)
        object.__setattr__(density, "log_normaliser", log_normaliser)
        return density


def gaussian_on(grid: QuadratureGrid, mean: ArrayLike, cov: ArrayLike) -> GridDensity:
    """``log N(x; mean, cov)`` at every node of ``grid``.

    The one place a Gaussian is put on a lattice. Every rung of the rule ladder ends
    by rendering its belief this way, and the exact filter's oracles start from it.

    Args:
        grid: the lattice to evaluate on.
        mean: the Gaussian's mean, length ``grid.ndim``. A scalar on a 1-D grid.
        cov: its covariance, ``grid.ndim`` square and positive definite. A scalar
            variance on a 1-D grid.

    Returns:
        The log-density at each node, not normalised by the grid. Its ``log_mass`` is
        zero to within quadrature error when the box holds the Gaussian, and below it
        when the box clips it.

    Raises:
        ValueError: if ``mean`` or ``cov`` does not match the grid's dimension, or
            ``cov`` is not a positive-definite covariance. Any positive variance is
            a Gaussian, however sharp. There is no floor: the floor a sensor noise
            carries is about inverting it, and nothing here inverts.
    """
    mean_vector = jnp.atleast_1d(jnp.asarray(mean, dtype=float))
    covariance = jnp.atleast_2d(jnp.asarray(cov, dtype=float))
    if mean_vector.shape != (grid.ndim,):
        raise ValueError(
            f"mean must have {grid.ndim} entries to match the grid, got shape "
            f"{mean_vector.shape}"
        )
    if covariance.shape != (grid.ndim, grid.ndim):
        raise ValueError(
            f"cov must be {grid.ndim} x {grid.ndim} to match the grid, got shape "
            f"{covariance.shape}"
        )
    validate_covariance(covariance, "cov")
    concrete_cov = concrete(covariance)
    if concrete_cov is not None and float(np.linalg.eigvalsh(concrete_cov).min()) <= 0:
        raise ValueError(
            "cov must be positive-definite: a Gaussian with a zero-variance "
            "direction has no density on the lattice"
        )
    centred = grid.nodes - mean_vector
    _, log_det = jnp.linalg.slogdet(covariance)
    quadratic = jnp.einsum(
        "ni,ni->n", centred, jnp.linalg.solve(covariance, centred.T).T
    )
    return GridDensity(
        grid, -0.5 * (grid.ndim * math.log(2.0 * math.pi) + log_det + quadratic)
    )
