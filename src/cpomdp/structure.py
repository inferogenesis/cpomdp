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

__all__ = ["ModelStructure"]

# The stored, hashable form: an ordered tuple of (name, index-tuple) pairs.
_Groups = tuple[tuple[str, tuple[int, ...]], ...]
# The accepted input form: an iterable of (name, indices) pairs; indices may be any int
# sequence (a list is fine — it is frozen to a tuple). Dicts enter via ``from_dicts``.
_Pairs = Iterable[tuple[str, Sequence[int]]]


def _freeze(pairs: _Pairs) -> _Groups:
    """Normalise an iterable of (name, indices) pairs into the hashable tuple form."""
    return tuple((str(name), tuple(int(i) for i in idx)) for name, idx in pairs)


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

    @staticmethod
    def _lookup(groups: _Groups, name: str, kind: str) -> tuple[int, ...]:
        for declared, idx in groups:
            if declared == name:
                return idx
        raise KeyError(f"no {kind} named {name!r}; declared: {[n for n, _ in groups]}")
