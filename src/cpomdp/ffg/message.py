"""Canonical-form Gaussian messages: the FFG's wire payload.

A message on an edge of the factor graph is a Gaussian potential carried in
canonical (information) form, ``(precision, potential) = (Λ, h)`` with
``Λ = Σ⁻¹`` (precision matrix) and ``h = Σ⁻¹μ`` (potential / information
vector). Two properties make this the storage form (DECISIONS.md ADR-012):

- **Factor product is addition.** Multiplying Gaussian potentials — what a
  factor node does when it combines its incoming messages — is
  ``(Λ1, h1) + (Λ2, h2) = (Λ1+Λ2, h1+h2)``. No inversion, ever, on this path.
- **Moment form (mean, covariance) is a view**, computed only at readout
  (``to_moment``) or when a block of variables must be eliminated
  (``marginalize`` — the one place an inversion is intrinsic to the
  operation: it inverts only the eliminated block, never the whole ``Λ``).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float64
from numpy.typing import ArrayLike

from cpomdp._validation import validate_covariance, validate_finite


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class CanonicalGaussian:
    """A Gaussian message in canonical (information) form, ``(Λ, h)``.

    - ``precision`` -- Λ = Σ⁻¹. An n x n matrix. Symmetric positive-*semi*-
      definite (not necessarily definite: a single, not-yet-combined factor
      message can be rank-deficient — e.g. a factor that doesn't yet
      constrain every direction. Only the *combined* product of all of a
      variable's incoming messages needs to be definite, and only at the
      point something reads its moment form).
    - ``potential`` -- h = Σ⁻¹μ. A 1-D vector of length n.

    Construct directly from canonical parameters. There is no
    ``from_moment``/moment-form constructor in v0.4 (open question, parked in
    the build plan) — moment form is purely a readout view via
    ``to_moment``.
    """

    precision: Float64[Array, "n n"]
    potential: Float64[Array, "n"]

    def __init__(self, precision: ArrayLike, potential: ArrayLike):
        object.__setattr__(self, "precision", jnp.asarray(precision, dtype=float))
        object.__setattr__(self, "potential", jnp.asarray(potential, dtype=float))
        self._validate()

    def _validate(self):
        if self.potential.ndim != 1:
            raise ValueError(
                f"potential must be 1-D vector, got shape {self.potential.shape}"
            )
        validate_finite(self.potential, "potential")
        validate_covariance(self.precision, "precision", require_definite=False)
        n = self.potential.shape[0]
        if self.precision.shape != (n, n):
            raise ValueError(
                f"precision must be {n}x{n} to match a {n}-D potential, "
                f"got shape {self.precision.shape}"
            )

    @property
    def ndim(self) -> int:
        """Dimensionality of the message — the length of the potential vector."""
        return self.precision.shape[0]

    @classmethod
    def _unchecked(cls, precision, potential):
        """Build without validating — for trusted, invariant-preserving inputs.

        Used by the factor product, the marginal, and the pytree rebuild path;
        precision/potential are trusted and may be tracers.
        """
        obj = object.__new__(cls)
        object.__setattr__(obj, "precision", precision)
        object.__setattr__(obj, "potential", potential)
        return obj

    def __add__(self, other: "CanonicalGaussian") -> "CanonicalGaussian":
        """Factor product: combine two messages on the same variable.

        In canonical form the product of two Gaussian potentials is the
        elementwise sum of their parameters::

            (Λ1, h1) + (Λ2, h2) = (Λ1 + Λ2, h1 + h2)

        This is the whole reason messages live in information form: a factor
        node combining its inputs never inverts anything on this path. Both
        operands must share the same dimension; a shape mismatch raises
        ``ValueError`` (the message mentions "shape").
        """
        if self.precision.shape != other.precision.shape:
            raise ValueError(
                f"cannot add messages of shape {self.precision.shape} "
                f"and {other.precision.shape}"
            )

        return CanonicalGaussian._unchecked(
            self.precision + other.precision, potential=self.potential + other.potential
        )

    def to_moment(self) -> tuple[Float64[Array, "n"], Float64[Array, "n n"]]:
        """Read out moment form ``(mean, cov)`` from canonical (information) form.

        Takes no parameters — it reads the message's own two stored fields and
        inverts the precision to recover covariance and mean::

            Σ = Λ⁻¹        μ = Σ h = Λ⁻¹ h

        The two fields, and the names they go by across domains:

        - ``self.precision`` = Λ = Σ⁻¹ — the precision, a.k.a. the information
          matrix.
        - ``self.potential`` = h = Λμ = Σ⁻¹μ — the *potential* (cpomdp's field
          name); the information-filter and message-passing literature call this
          exact vector the *information vector*. One quantity, two names.

        A *view*, computed on demand — not the storage form. The precision must
        be positive-**definite** here (something is actually reading a combined
        belief); a singular Λ has no moment form and must raise ``ValueError``
        rather than return inf/NaN — use ``validate_covariance(...,
        require_definite=True)`` (its message says "positive-definite"). Prefer a
        single linear solve over forming Λ⁻¹ explicitly, and stay jit/grad-clean.

        Returns:
            ``(mean, cov)``, matching the signature
            ``tuple[Float64[Array, "n"], Float64[Array, "n n"]]``:

            - ``mean`` — μ = Λ⁻¹ h, the distribution mean. Shape ``(n,)``
              (``Float64[Array, "n"]``).
            - ``cov`` — Σ = Λ⁻¹, the covariance. Shape ``(n, n)``
              (``Float64[Array, "n n"]``).

        Raises:
            ValueError: If the precision is not positive-definite (singular or
                indefinite) — moment form does not exist.
        """
        validate_covariance(self.precision, "precision", require_definite=True)

        mean = jnp.linalg.solve(self.precision, self.potential)
        cov = jnp.linalg.inv(self.precision)
        return (mean, cov)

    def marginalize(self, over: Sequence[int]) -> "CanonicalGaussian":
        """Eliminate the variables in ``over``, returning the marginal on the rest.

        Partition the indices into the eliminated set ``b = over`` and the kept
        set ``a`` (everything else). With

            Λ = [[Λaa, Λab],      h = [ha,
                 [Λba, Λbb]]            hb]

        the marginal over ``a`` is the Schur complement of the ``b`` block::

            Λ' = Λaa − Λab Λbb⁻¹ Λba
            h' = ha  − Λab Λbb⁻¹ hb

        The kept indices come back in ascending order, whatever order ``over``
        was given in. Only the eliminated block ``Λbb`` is inverted — never the
        whole precision — and it must be positive-definite, else this raises
        ``ValueError`` (message "positive-definite"). This is the one operation
        where an inversion is intrinsic to the algebra.
        """
        # Two notions of "index" meet here; keep them straight:
        #   (1) the FIXED variable-name → index mapping — slot i (0..n-1) always
        #       denotes the same variable: the order variables were stacked into
        #       Λ/h, fixed for the message's whole life.
        #   (2) `over` — derived PER EMISSION (which variables to eliminate this
        #       call), given as integer indices under mapping (1).
        # So over_set / keep / elim below are all positions in mapping (1); this
        # step just splits those fixed slots into stay (keep) vs go (elim).
        over_set = {int(i) for i in over}
        keep = [
            i for i in range(self.ndim) if i not in over_set
        ]  # ascending by construction
        elim = [
            i for i in range(self.ndim) if i in over_set
        ]  # same ordering everywhere

        keep_idx = jnp.asarray(keep)
        elim_idx = jnp.asarray(elim)

        Laa = self.precision[jnp.ix_(keep_idx, keep_idx)]
        Lab = self.precision[jnp.ix_(keep_idx, elim_idx)]
        Lba = self.precision[jnp.ix_(elim_idx, keep_idx)]
        Lbb = self.precision[jnp.ix_(elim_idx, elim_idx)]
        ha = self.potential[keep_idx]
        hb = self.potential[elim_idx]

        validate_covariance(Lbb, "eliminated block", require_definite=True)

        Lprime = Laa - Lab @ jnp.linalg.solve(Lbb, Lba)  # Λaa − Λab Λbb⁻¹ Λba
        hprime = ha - Lab @ jnp.linalg.solve(Lbb, hb)  # ha  − Λab Λbb⁻¹ hb
        return CanonicalGaussian._unchecked(Lprime, hprime)

    def tree_flatten(
        self,
    ) -> tuple[tuple[Float64[Array, "n"], Float64[Array, "n n"]], None]:
        """Leaves for JAX: ``(precision, potential)``, no static aux data."""
        return (self.precision, self.potential), None

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: None,
        children: tuple[Float64[Array, "n"], Float64[Array, "n n"]],
    ) -> "CanonicalGaussian":
        """Rebuild from leaves without validating — the leaves may be tracers."""
        precision, potential = children
        return cls._unchecked(precision, potential)
