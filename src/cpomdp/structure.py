"""Declarable, inspectable model structure — the factor / Markov-blanket seam.

A ``LinearGaussianModel`` is a dense matrix blob: nothing in it says which state
indices form a cause, which are internal vs external, or which observation rows are a
sensory channel. ``ModelStructure`` is optional metadata that declares exactly that —
the "metadata version ships first" seam of RFC-003 section 4.5. It is pure data: the
matrices and every kernel are byte-for-byte unchanged whether a model carries structure
or not.

Why it lands in v0.3 (the deliberate YAGNI break, DECISIONS.md ADR-010): to secure the
API early — a structure vocabulary added now is a pure addition; added after people have
models it churns everyone — and because it has a concrete consumer. The leading reading
of E. coli's internals is a *distributed, multi-variable generative model* (Mattingly),
not a monolith, and I want this toolbox to express and probe that. The right
factorisation is itself open research; a better reading is welcome as a repo Discussion
or Issue.

The load-bearing constraint: a ``ModelStructure`` rides in the model's pytree aux_data,
not its children — it has no traced array leaves. JAX hashes aux_data for ``jit``'s
cache key, so every field is a tuple of tuples (a dict/list would be unhashable and
break ``jit``); the frozen dataclass then hashes itself.

API stability (ADR-010): the data and inspection surface here is stable, promised API.
``validate`` (added with the structure-validation step) is experimental — its partition
checks are durable, but its conditional-independence criterion tightens to the rigorous
precision-based test in v0.4.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

# TODO(revisit): this makes me uneasy. ModelStructure referencing LinearGaussianModel
# even for types is a peer back-reference only validate needs — the reason this import
# hides under TYPE_CHECKING. Not a layering break, so I'm keeping it FOR NOW. Come back
# and decide if model.validate_structure() (model as information-expert) or a Protocol
# here reads cleaner; flip unless this stays well-organised with no real coupling cost.
if TYPE_CHECKING:
    from cpomdp.types import LinearGaussianModel

__all__ = ["ModelStructure"]

# The stored, hashable form: an ordered tuple of (name, index-tuple) pairs.
_Groups = tuple[tuple[str, tuple[int, ...]], ...]
# The accepted input form: an iterable of (name, indices) pairs; indices may be any int
# sequence (a list is fine — it is frozen to a tuple). Dicts enter via ``from_dicts``.
_Pairs = Iterable[tuple[str, Sequence[int]]]


def _freeze(pairs: _Pairs) -> _Groups:
    """Normalise an iterable of (name, indices) pairs into the hashable tuple form.

    Rejects duplicate names within a group: a repeat would make the stable
    ``factor``/``role_of``/``channel`` lookups silently return only the first match.
    """
    frozen = tuple((str(name), tuple(int(i) for i in idx)) for name, idx in pairs)
    names = [name for name, _ in frozen]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(
            f"duplicate group name(s) {dupes} — factor/role/channel names must be "
            f"unique (a repeat would shadow earlier indices in lookups)."
        )
    return frozen


@dataclass(frozen=True, init=False)
class ModelStructure:
    """A static, hashable declaration of a model's factor / blanket / channel structure.

    Three index groupings over a model with state dimension ``n`` and observation
    dimension ``m``:

    - ``factors``  -- name -> state indices: which indices form which cause/block.
    - ``roles``    -- role -> state indices: the Markov-blanket typing of the state. The
      intended (epistemic, not metaphysical — RFC-003 section 7) vocabulary is
      ``"external"`` / ``"internal"`` / ``"active"``; names are free-form labels, only
      partition-checked in v0.3 (the blanket independence test is v0.4).
    - ``channels`` -- name -> observation-row indices: the sensory typing of outputs.

    Construct with the tuple form directly, or :meth:`from_dicts` for the dict form.
    Every field normalises to ``(("name", (idx, ...)), ...)`` so the whole object is
    hashable — it must be, it rides in the model's pytree aux_data. There is no
    construction-time ``_validate``: every real check needs the model, so it lives in
    the opt-in :meth:`validate`.
    """

    factors: _Groups
    roles: _Groups
    channels: _Groups

    def __init__(
        self,
        factors: _Pairs = (),
        roles: _Pairs = (),
        channels: _Pairs = (),
    ) -> None:
        object.__setattr__(self, "factors", _freeze(factors))
        object.__setattr__(self, "roles", _freeze(roles))
        object.__setattr__(self, "channels", _freeze(channels))

    @classmethod
    def from_dicts(
        cls,
        *,
        factors: Mapping[str, Sequence[int]] | None = None,
        roles: Mapping[str, Sequence[int]] | None = None,
        channels: Mapping[str, Sequence[int]] | None = None,
    ) -> "ModelStructure":
        """Build from the natural dict form, e.g. ``factors={"pos": [0, 1]}``."""
        return cls(
            factors=() if factors is None else factors.items(),
            roles=() if roles is None else roles.items(),
            channels=() if channels is None else channels.items(),
        )

    # --- inspection (stable API) ---------------------------------------------------
    @property
    def factor_names(self) -> tuple[str, ...]:
        """The declared factor names, in declaration order."""
        return tuple(name for name, _ in self.factors)

    def factor(self, name: str) -> tuple[int, ...]:
        """State indices of the named factor (raises ``KeyError`` if undeclared)."""
        return self._lookup(self.factors, name, "factor")

    def channel(self, name: str) -> tuple[int, ...]:
        """Observation-row indices of the named channel (raises ``KeyError``)."""
        return self._lookup(self.channels, name, "channel")

    def role_of(self, index: int) -> str | None:
        """The role typing state ``index``, or ``None`` if no role contains it."""
        for role, idx in self.roles:
            if index in idx:
                return role
        return None

    def summary(self) -> str:
        """A readable multi-line dump of the declared structure (does not print)."""

        def block(title: str, groups: _Groups, unit: str) -> list[str]:
            if not groups:
                return [f"  {title}: (none)"]
            rows = [f"    {name} -> {unit} {idx}" for name, idx in groups]
            return [f"  {title}:", *rows]

        return "\n".join(
            [
                "ModelStructure(",
                *block("factors", self.factors, "states"),
                *block("roles", self.roles, "states"),
                *block("channels", self.channels, "rows"),
                ")",
            ]
        )

    def validate(self, model: "LinearGaussianModel", *, atol: float = 1e-9) -> None:
        """Raise if this declaration contradicts ``model`` (opt-in; EXPERIMENTAL).

        Partition well-formedness (pure index arithmetic, a stable contract): declared
        factors and roles each partition the ``n``-state space — every index in
        ``[0, n)``, pairwise disjoint, covering all of it; channels index valid,
        distinct observation rows in ``[0, m)`` but need not cover them. The
        full-coverage requirement for factors/roles is a strict but reversible choice
        (ADR-010), to relax if it proves a faff.

        Conditional independence (EXPERIMENTAL): factors declared independent must have
        ≈0 cross-blocks in the dynamics ``A`` (and the fixed process noise ``Q``) to
        ``atol``, and a sensory channel must read within a single factor. A
        state-dependent ``Q(x)`` has no single matrix to check, so it is skipped. This
        criterion checks one-step blocks now and tightens to the rigorous
        precision-based (``Σ⁻¹`` block-diagonal) test in v0.4.

        Not run at construction to remain lean; opt in via
        ``model.structure.validate(model)``.
        """
        n_states = model.n_states
        n_observations = model.n_observations

        self._validate_partition(self.factors, n_states, "factor", require_cover=True)
        self._validate_partition(self.roles, n_states, "role", require_cover=True)
        self._validate_partition(
            self.channels, n_observations, "channel", require_cover=False
        )
        a_mat = np.asarray(model.dynamics)
        c_mat = np.asarray(model.sensor_model)
        # fixed Q only — a state-dependent Q(x) has no single matrix to check (skip it).
        q_mat = (
            None
            if model.process_noise is not None
            else np.asarray(model.dynamics_noise)
        )
        for name_i, idx_i in self.factors:
            for name_j, idx_j in self.factors:
                if name_i == name_j:
                    continue
                self._assert_zero_block(a_mat, idx_i, idx_j, atol, "A", name_i, name_j)
                if q_mat is not None:
                    self._assert_zero_block(
                        q_mat, idx_i, idx_j, atol, "dynamics_noise", name_i, name_j
                    )
        if self.channels and self.factors:
            for ch_name, rows in self.channels:
                self._assert_channel_clean(c_mat, rows, self.factors, atol, ch_name)

    @staticmethod
    def _assert_zero_block(mat, rows, cols, atol, mat_name, fi, fj):
        block = mat[np.ix_(list(rows), list(cols))]
        if block.size == 0:
            return
        finite = np.isfinite(block)
        if not finite.all():
            a, b = np.unravel_index(int(np.argmax(~finite)), block.shape)
            bad = block[a, b]
            raise ValueError(
                f"factors {fi!r} and {fj!r} are declared conditionally independent, "
                f"but {mat_name}[{fi}, {fj}] has a non-finite entry {bad} at "
                f"({list(rows)[a]}, {list(cols)[b]}) — a NaN/Inf cannot be certified "
                f"zero; the declaration does not match the matrix sparsity."
            )
        a, b = np.unravel_index(int(np.argmax(np.abs(block))), block.shape)
        peak = float(block[a, b])
        if abs(peak) > atol:
            raise ValueError(
                f"factors {fi!r} and {fj!r} are declared conditionally independent, "
                f"but {mat_name}[{fi}, {fj}] has a nonzero entry {peak:.3g} at "
                f"({list(rows)[a]}, {list(cols)[b]}) — {mat_name} couples them; the "
                f"declaration does not match the matrix sparsity."
            )

    @staticmethod
    def _assert_channel_clean(c_mat, rows, factors, atol, ch_name):
        rows = list(rows)
        if not rows:
            return  # empty channel: no rows to read, nothing to cross-contaminate
        sub = c_mat[rows, :]
        if not np.isfinite(sub).all():
            raise ValueError(
                f"channel {ch_name!r} reads a non-finite entry in the sensor matrix "
                f"C — a NaN/Inf cannot be certified within a single factor."
            )
        active = set(np.nonzero(np.max(np.abs(sub), axis=0) > atol)[0].tolist())
        if not active:
            return
        touched = [name for name, idx in factors if active & set(idx)]
        if len(touched) > 1:
            raise ValueError(
                f"channel {ch_name!r} reads state columns {sorted(active)} spanning "
                f"multiple factors {touched} — a sensory channel must read within "
                f"a single factor (cross-contamination; does not match C)."
            )

    @staticmethod
    def _validate_partition(
        groups: _Groups, size: int, kind: str, *, require_cover: bool
    ) -> None:
        if not groups:
            return
        seen: set[int] = set()
        for name, idx in groups:
            for i in idx:
                if not 0 <= i < size:
                    raise ValueError(
                        f"{kind} {name!r} has index {i} out of range [0, {size})"
                    )
                if i in seen:
                    raise ValueError(
                        f"{kind} {name!r} index {i} overlaps another {kind} "
                        f"— {kind}s must be disjoint"
                    )
                seen.add(i)
        if require_cover and seen != set(range(size)):
            missing = sorted(set(range(size)) - seen)
            raise ValueError(
                f"{kind}s must cover all {size} indices; missing {missing}"
            )

    @staticmethod
    def _lookup(groups: _Groups, name: str, kind: str) -> tuple[int, ...]:
        for declared, idx in groups:
            if declared == name:
                return idx
        raise KeyError(f"no {kind} named {name!r}; declared: {[n for n, _ in groups]}")
