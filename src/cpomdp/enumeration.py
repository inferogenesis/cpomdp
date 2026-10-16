"""Exhaustive EFE search over a declared finite action set (ADR-030, ADR-031).

The enumerated counterpart of ``EFESelector``'s continuous grid, and a deliberately
*different object* so the two cannot be confused. ``EFESelector`` samples a continuous
action box and tiles each sample into a **constant-action** policy — a sample of a
continuum, warrant ``CORROBORATED`` (Prover 3 · sample: corroborates, never
decides).
``EnumeratedEfeSearch`` enumerates **every** length-H sequence of a finite, declared,
versioned ``FiniteActionSet`` — all ``|A|^H`` of them, *varying* sequences included — so
"no policy in this set flips" is **decided**, not sampled: warrant ``PROVED``
(Prover 3 · enumeration). A ``CompletenessCertificate`` records ``expected = |A|^H``
against the ``visited`` count and asserts they match (ADR-030), so the decisive warrant
is earned, not assumed.

The set is versioned because it is a modelling commitment: an action added after results
are seen must show up in the diff, not be discovered by a reviewer. The ``PROVED``
warrant is scoped to the *declared* set — whether that set is fine enough is a separate
question a refinement-stability check answers, not this certificate.

Imported from ``cpomdp.enumeration``. None of it is re-exported at the top level.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax.errors import ConcretizationTypeError, TracerArrayConversionError
from jaxtyping import Array, Float64, Int
from numpy.typing import ArrayLike

from cpomdp._validation import concrete
from cpomdp.efe import _checked_info_block, _policy_efe_ffg, policy_efe
from cpomdp.types import Belief, LinearGaussianModel

# `CompletenessCertificate` is an evidence kind of the warrant vocabulary, so it is
# defined in `warrantlib` beside `SymbolicReduction`. It is re-exported below, because
# this is where an enumeration produces one.
from warrantlib import CompletenessCertificate, Warrant

if TYPE_CHECKING:
    # Duck-typed at runtime (read for .goal/.precision via the efe kernels, and the
    # backend's predict/observation methods), so no runtime dependency on selection or
    # the backends package — keeps selection/backends -> enumeration the only edges.
    from cpomdp.backends.base import EfeBackend
    from cpomdp.selection import Preference

__all__ = [
    "ChunkedEfeSearch",
    "CompletenessCertificate",
    "EnumeratedEfeSearch",
    "EnumeratedSearchResult",
    "ExhaustiveEfeSearch",
    "FiniteActionSet",
    "IncompleteEnumerationError",
    "OpenLoopSelector",
    "PolicyBudgetError",
    "RecedingHorizonSelector",
    "SearchResult",
]


class IncompleteEnumerationError(Exception):
    """Fewer than ``|A|^H`` policies enumerated: a sample, not a decision."""


class PolicyBudgetError(ValueError):
    """A front-loaded search asked to hold more policies than its budget allows.

    ``EnumeratedEfeSearch`` holds every policy and every score at once, so its
    footprint grows with ``|A|^H``. The count is exact before anything is allocated,
    and an oversized set is refused there rather than exhausting the host. The message
    gives the count, the step evaluations one plan costs and a floor on the memory. It
    names [`ChunkedEfeSearch`][cpomdp.enumeration.ChunkedEfeSearch], which runs the
    same enumeration at a footprint its block sets and drives the same selectors.
    """


# The front-loaded search's default policy budget (ADR-083). It admits every
# front-loaded run the repository makes, the largest being 9^6 = 531,441, and refuses
# 9^7, whose front-loaded run the warrant ledger records at 5.4 GiB.
_DEFAULT_MAX_POLICIES = 1 << 21


@dataclass(frozen=True, init=False)
class FiniteActionSet:
    """A finite, declared, **versioned** set of actions to enumerate over.

    A distinct type from the continuous grid's ``(lo, hi, n_candidates)`` config, so the
    enumerated (``PROVED``) and sampled (``CORROBORATED``) families cannot be mixed up
    at a call site. Construction-time only (not a pytree).

    Args:
        actions: the declared actions, shape ``(A, p)`` — ``A`` actions of dimension
            ``p``. At least two (a set of one is no choice to search), every entry
            finite. A repeated action counts toward ``A``, so it is scored twice.
        version: a non-empty tag, so an action added later shows up in the diff.

    Raises:
        ValueError: If ``actions`` is not 2-D, holds fewer than two actions or a
            non-finite entry, or if ``version`` is empty.
    """

    actions: Float64[Array, "A p"]
    version: str

    def __init__(self, actions: ArrayLike, *, version: str) -> None:
        object.__setattr__(self, "actions", jnp.asarray(actions, dtype=float))
        object.__setattr__(self, "version", version)
        self._validate()

    def _validate(self) -> None:
        if self.actions.ndim != 2:
            raise ValueError(
                f"actions must be a 2-D (A, p) array, got shape {self.actions.shape}"
            )
        if self.actions.shape[0] < 2:
            raise ValueError(
                f"a FiniteActionSet needs at least 2 actions to search, got "
                f"{self.actions.shape[0]}"
            )
        values = concrete(self.actions)
        if values is None:
            raise TypeError(
                "a FiniteActionSet is built outside jit, vmap and grad. Under a trace "
                "its actions cannot be checked for NaN. Build it once and close over "
                "it."
            )
        finite = np.all(np.isfinite(values), axis=1)
        if not finite.all():
            rows = ", ".join(str(row) for row in np.flatnonzero(~finite))
            raise ValueError(
                f"every action must be finite. Non-finite rows: {rows}. A policy that "
                "takes one scores NaN, and the search would still certify covering it."
            )
        if not isinstance(self.version, str) or not self.version:
            raise ValueError(
                "version must be a non-empty string — the set is declared and versioned"
            )

    @property
    def size(self) -> int:
        """The action count ``A`` (the base of ``|A|^H``)."""
        return int(self.actions.shape[0])

    @property
    def action_dim(self) -> int:
        """The action dimension ``p``."""
        return int(self.actions.shape[1])


def _validate_spec(
    scorer: "_PolicyScorer", action_set: FiniteActionSet, horizon: int
) -> None:
    """The preconditions both enumeration paths share."""
    if action_set.action_dim != scorer.action_dim:
        raise ValueError(
            f"the action set is p={action_set.action_dim} but the control is "
            f"p={scorer.action_dim}; the declared actions must match the action "
            "dimension."
        )
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")


def _check_index_ceiling(n_policies: int) -> None:
    """Refuse an enumeration whose indices the index dtype cannot hold."""
    ceiling = int(jnp.iinfo(jnp.arange(1).dtype).max)
    if n_policies > ceiling:
        raise ValueError(
            f"|A|^H = {n_policies} exceeds the index dtype's ceiling "
            f"{ceiling}. Enable x64, or declare a smaller set: an index that "
            "wraps would enumerate the wrong policies and still certify."
        )


def _check_budget(
    scorer: "_PolicyScorer",
    action_set: FiniteActionSet,
    horizon: int,
    max_policies: int | None,
) -> None:
    """Refuse a front-loaded set past ``max_policies``, from the count alone."""
    if max_policies is None:
        return
    if max_policies < 1:
        raise ValueError(f"max_policies must be >= 1 or None, got {max_policies}")
    n_policies = action_set.size**horizon
    if n_policies <= max_policies:
        return
    # What is resident at once while `evaluate` runs: every policy, every score, and
    # the mean and covariance the batched rollout carries for each policy.
    per_policy = (
        horizon * action_set.action_dim + 1 + scorer.n_states * (1 + scorer.n_states)
    )
    floor_gib = 8 * n_policies * per_policy / 1024**3
    raise PolicyBudgetError(
        f"|A|^H = {action_set.size}^{horizon} = {n_policies:,} policies exceeds "
        f"max_policies = {max_policies:,}. One plan costs "
        f"{n_policies * horizon:,} step evaluations, and holding the set needs at "
        f"least {floor_gib:.2f} GiB for the policies, their scores and one belief per "
        "policy, before XLA's intermediates. ChunkedEfeSearch runs the same "
        "enumeration in blocks at a footprint its block sets, earns the same "
        "certificate and drives RecedingHorizonSelector and OpenLoopSelector. To hold "
        "the set anyway, pass max_policies=None or a larger budget."
    )


def _certify(
    action_set: FiniteActionSet, horizon: int, visited: int
) -> CompletenessCertificate:
    """The certificate for an enumeration that reported ``visited`` policies.

    One guard for both enumeration paths. On the front-loaded path ``visited`` is an
    array's length and cannot disagree; on the chunked path it is a loop-carried count
    of unpadded lanes, where a masking bug shows up here and nowhere else.
    """
    expected = action_set.size**horizon
    if visited != expected:  # ADR-030: the certificate is asserted, not assumed
        raise IncompleteEnumerationError(
            f"enumeration visited {visited} of an expected {expected} = |A|^H "
            f"policies over set {action_set.version!r}; the search would be a sample, "
            "not a decision."
        )
    return CompletenessCertificate(
        expected=expected,
        visited=visited,
        warrant=Warrant.PROVED,
        action_set_size=action_set.size,
        horizon=horizon,
        action_set_version=action_set.version,
    )


def _decode_policies(
    indices: Array, action_set: FiniteActionSet, horizon: int
) -> Float64[Array, "... H p"]:
    """Policy sequences for global policy indices, without materialising the set.

    Index ``i`` is read as a base-``|A|`` numeral, most significant digit first, the
    order ``itertools.product`` emits. Both paths decode with this function, the
    front-loaded one over every index at construction, so they agree on *which* policy
    each index names. Without that a cross-path argmin comparison would be comparing
    labels rather than plans.
    """
    powers = jnp.asarray(
        [action_set.size ** (horizon - 1 - step) for step in range(horizon)]
    )
    digits = (indices[..., None] // powers) % action_set.size
    return action_set.actions[digits]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SearchResult:
    """The outcome of one exhaustive search: the best plan and the evidence for it.

    [`ChunkedEfeSearch`][cpomdp.enumeration.ChunkedEfeSearch] returns one. The
    front-loaded search returns
    [`EnumeratedSearchResult`][cpomdp.enumeration.EnumeratedSearchResult], which adds
    the score vector. Fields are read by name. A result does not unpack or index, so
    no position can name one field on one search and another on the other.

    A JAX pytree whose certificate is static data, so a jitted or vmapped
    ``evaluate`` returns the whole result.

    Attributes:
        best_policy: the argmin sequence, shape ``(H, p)``.
        best_g: the argmin's summed EFE, G.
        best_index: the argmin's position in the enumeration.
        certificate: the completeness certificate (ADR-030).

    When no policy scores finitely there is no argmin. ``best_policy`` and ``best_g``
    are then NaN and ``best_index`` is -1. The certificate still holds, since it
    certifies that every policy was scored.
    """

    best_policy: Float64[Array, "H p"]
    best_g: Float64[Array, ""]
    best_index: Int[Array, ""]
    certificate: CompletenessCertificate = field(metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EnumeratedSearchResult(SearchResult):
    """The outcome of one front-loaded exhaustive search, every policy's score included.

    A [`SearchResult`][cpomdp.enumeration.SearchResult] plus the score vector, which
    only the front-loaded path holds. Code that reads the shared fields runs on either
    search.

    Attributes:
        g: the summed EFE of every enumerated policy, in enumeration order, shape
            ``(|A|^H,)``.
    """

    g: Float64[Array, "N"]


class _PolicyScorer(Protocol):
    """Scores one policy to its summed EFE ``G`` — the seam that swaps flat for FFG.

    The enumerated search owns the enumeration, the certificate, and the argmin; *how* a
    policy is scored is injected here (ADR-034). ``action_dim`` lets the search validate
    the declared action set against the scorer's control dimension, and ``n_states``
    sizes the memory floor a policy budget reports.
    """

    action_dim: int
    n_states: int

    def score(
        self, belief: Belief, policy: Float64[Array, "H p"], preference: "Preference"
    ) -> Float64[Array, ""]: ...


@dataclass(frozen=True)
class _FlatScorer:
    """Scores with ``policy_efe`` on a flat ``LinearGaussianModel``."""

    model: LinearGaussianModel

    @property
    def action_dim(self) -> int:
        return self.model.n_controls

    @property
    def n_states(self) -> int:
        return self.model.n_states

    def score(
        self, belief: Belief, policy: Float64[Array, "H p"], preference: "Preference"
    ) -> Float64[Array, ""]:
        return policy_efe(self.model, belief, policy, preference)[0]


@dataclass(frozen=True)
class _FfgScorer:
    """Scores with ``policy_efe_ffg`` on an FFG backend, aimed at ``info_block``.

    The block is checked once, by ``over_backend``, so ``score`` skips the check.
    """

    backend: "EfeBackend"
    info_block: tuple[int, ...]

    @property
    def action_dim(self) -> int:
        return self.backend.model.n_controls

    @property
    def n_states(self) -> int:
        return self.backend.model.n_states

    def score(
        self, belief: Belief, policy: Float64[Array, "H p"], preference: "Preference"
    ) -> Float64[Array, ""]:
        return _policy_efe_ffg(
            self.backend, belief, policy, preference, self.info_block
        )[0]


class _Scored(NamedTuple):
    """What the compiled front-loaded scorer returns in one call."""

    g: Float64[Array, "N"]
    best_index: Int[Array, ""]
    best_g: Float64[Array, ""]
    best_policy: Float64[Array, "H p"]


def _score_every_policy(
    scorer: _PolicyScorer,
    policies: Float64[Array, "N H p"],
    belief: Belief,
    preference: "Preference",
) -> _Scored:
    """Every policy's summed EFE, then the NaN-safe argmin, its score and its policy."""
    g = jax.vmap(lambda policy: scorer.score(belief, policy, preference))(policies)
    masked = jnp.where(jnp.isnan(g), jnp.inf, g)
    best_index = jnp.argmin(masked)
    best_g, best_index, best_policy = _winner_or_none(
        masked[best_index], g[best_index], best_index, policies[best_index]
    )
    return _Scored(g, best_index, best_g, best_policy)


def _winner_or_none(
    masked_best: Float64[Array, ""],
    best_g: Float64[Array, ""],
    best_index: Int[Array, ""],
    best_policy: Float64[Array, "H p"],
) -> tuple[Float64[Array, ""], Int[Array, ""], Float64[Array, "H p"]]:
    """The argmin, or NaN, -1 and a NaN policy when no policy scored below ``+inf``.

    A NaN score is masked to ``+inf`` before the argmin, so when every score is NaN the
    argmin lands on index 0 by default. Reporting that policy would present the first
    declared sequence as a decision. The NaN policy is jit-safe. A driver refuses it
    on a concrete run and passes it on under a trace.
    """
    found = masked_best < jnp.inf
    return (
        jnp.where(found, best_g, jnp.nan),
        jnp.where(found, best_index, -1),
        jnp.where(found, best_policy, jnp.nan),
    )


class ExhaustiveEfeSearch(ABC):
    """An EFE search that scores every length-H sequence of a declared action set.

    The kind the drivers take.
    [`EnumeratedEfeSearch`][cpomdp.enumeration.EnumeratedEfeSearch] holds every policy
    and every score at once, and
    [`ChunkedEfeSearch`][cpomdp.enumeration.ChunkedEfeSearch] scores the set a chunk
    at a time. Both earn the same ``PROVED`` certificate. Code that reads only what
    this class and [`SearchResult`][cpomdp.enumeration.SearchResult] declare runs on
    either.

    It is public as a type, for annotations and ``isinstance``. Subclassing it outside
    cpomdp is not supported: a subclass records its set and horizon through the private
    ``_enumerate``, and the ``PROVED`` it inherits holds only for a search that does.
    """

    _action_set: FiniteActionSet
    _horizon: int
    _n_policies: int

    def _enumerate(
        self, scorer: _PolicyScorer, action_set: FiniteActionSet, horizon: int
    ) -> None:
        """Validate the specification and record the enumeration's size."""
        _validate_spec(scorer, action_set, horizon)
        self._action_set = action_set
        self._horizon = int(horizon)
        self._n_policies = action_set.size**self._horizon

    @classmethod
    def _flat_scorer(cls, model: LinearGaussianModel) -> _FlatScorer:
        """The flat constructor's scorer, refusing a model no action can move."""
        if model.control_matrix is None:
            raise ValueError(
                f"{cls.__name__} needs a model with a control matrix; an action has no "
                "effect on a control-free (pure-tracking) model."
            )
        return _FlatScorer(model)

    @classmethod
    def _graph_scorer(
        cls, backend: "EfeBackend", info_block: Sequence[int]
    ) -> _FfgScorer:
        """``over_backend``'s scorer, its block checked here rather than per policy."""
        if backend.model.control_matrix is None:
            raise ValueError(
                f"{cls.__name__}.over_backend needs a backend with a control matrix; "
                "an action has no effect without one."
            )
        return _FfgScorer(backend, _checked_info_block(backend, info_block))

    @abstractmethod
    def evaluate(self, belief: Belief, preference: "Preference") -> SearchResult:
        """Score every policy from ``belief``, and return the argmin with its evidence.

        Args:
            belief: the belief to plan from.
            preference: the goal and precision the EFE scores against.
        """

    def policies_at(self, indices: ArrayLike) -> Float64[Array, "... H p"]:
        """The policies ``indices`` name, decoded from the index rather than looked up.

        The way to turn a result's ``best_index`` back into a plan, on either search.
        An index outside ``0..|A|^H - 1`` names no policy. Under ``jit`` or ``vmap`` it
        decodes to a NaN policy, the one a result with no winner carries.

        Raises:
            ValueError: If a concrete index lies outside the enumeration, including
                the -1 a result carries when no policy scored finitely. Decoded, it
                would wrap to a real policy.
        """
        indices = jnp.asarray(indices)
        outside = (indices < 0) | (indices >= self._n_policies)
        flagged = concrete(outside)  # None under a trace, where nothing can raise
        if flagged is not None and flagged.any():
            named = np.asarray(indices)[flagged.astype(bool)].tolist()
            raise ValueError(
                f"indices {named} lie outside 0..{self._n_policies - 1}, the "
                "enumeration's range."
            )
        policies = _decode_policies(indices, self._action_set, self._horizon)
        return jnp.where(outside[..., None, None], jnp.nan, policies)

    @property
    def n_policies(self) -> int:
        """The enumerated policy count ``|A|^H``, attributable work (RFC-001)."""
        return self._n_policies

    @property
    def horizon(self) -> int:
        """The sequence length H."""
        return self._horizon

    @property
    @abstractmethod
    def cost_per_plan(self) -> int:
        """Step evaluations one ``evaluate`` runs (RFC-001)."""

    @property
    def warrant(self) -> Warrant:
        """``PROVED``: a finite set enumerated in full decides its universal."""
        return Warrant.PROVED


class EnumeratedEfeSearch(ExhaustiveEfeSearch):
    """Exhaustive EFE search over ``A^H`` for a declared finite action set (ADR-031).

    Front-loads (ADR-002) the full ``|A|^H`` enumeration of *varying* sequences at
    construction, and checks its completeness there. ``evaluate`` scores them all and
    returns the argmin, its score and index, the certificate and the full ``G`` vector.
    Because the set is finite and fully enumerated the search **decides** its universal,
    warrant ``PROVED``, distinct from ``EFESelector``'s ``CORROBORATED`` grid.

    [`ChunkedEfeSearch`][cpomdp.enumeration.ChunkedEfeSearch] runs the same enumeration
    in blocks. The two take the same model, set and horizon, answer the same
    ``evaluate`` and drive the same selectors. Swapping one for the other changes the
    memory a plan needs and no line that reads the result.

    Scoring is a strategy (ADR-034): the default constructor scores on a flat
    ``LinearGaussianModel`` (``policy_efe``); ``over_backend`` scores on an FFG backend
    (``policy_efe_ffg``, aimed at a node's block) so the same exhaustive search runs the
    coupled-tree crossover model a flat model cannot express. The enumeration,
    the certificate, and the cost are shared — only the per-policy scoring differs.

    Unlike ``EFESelector`` this supports ``p >= 1`` and *varying* sequences: it can
    express a detour-then-exploit policy the constant-action family cannot.

    Memory grows with the set. While ``evaluate`` runs, every policy, its score and a
    belief per policy are resident at once, at least ``8 · |A|^H · (H·p + 1 + n + n²)``
    bytes before XLA's intermediates. A set past ``max_policies`` is refused at
    construction with that floor in the message. ``ChunkedEfeSearch`` holds one block
    at a time.

    Args:
        model: the flat model to score on. It needs a control matrix.
        action_set: the declared actions, enumerated as every length-H sequence.
        horizon: H, the sequence length.
        max_policies: the most policies the search will hold, ``2^21 = 2,097,152``
            unless set. ``None`` lifts the budget.

    Raises:
        PolicyBudgetError: If ``|A|^H`` exceeds ``max_policies``.
        ValueError: If the model has no control matrix, if the action set's dimension
            or the horizon does not fit, or if ``max_policies`` is below 1.
    """

    def __init__(
        self,
        model: LinearGaussianModel,
        action_set: FiniteActionSet,
        *,
        horizon: int,
        max_policies: int | None = _DEFAULT_MAX_POLICIES,
    ) -> None:
        self._setup(self._flat_scorer(model), action_set, horizon, max_policies)

    @classmethod
    def over_backend(
        cls,
        backend: "EfeBackend",
        action_set: FiniteActionSet,
        *,
        info_block: Sequence[int],
        horizon: int,
        max_policies: int | None = _DEFAULT_MAX_POLICIES,
    ) -> "EnumeratedEfeSearch":
        """Exhaustive search that scores on an FFG ``backend`` via ``policy_efe_ffg``.

        The FFG counterpart of the flat constructor: it enumerates the same ``A^H`` set
        with the same completeness certificate, but scores each policy on a coupling
        graph with the epistemic aimed at ``info_block`` — a node's block (via
        ``backend.block``) or the whole state. This runs the coupled-tree model the flat
        constructor cannot express. The policy budget is the flat constructor's.

        Raises:
            InadmissiblePartitionError: If the backend's carry partition severs an
                EFE-load-bearing edge.
            PolicyBudgetError: If ``|A|^H`` exceeds ``max_policies``.
            ValueError: If the backend has no control matrix, if ``info_block`` is
                empty, repeats an index or names one outside the joint state, if the
                action set's dimension or the horizon does not fit, or if
                ``max_policies`` is below 1.
        """
        search = cls.__new__(cls)
        scorer = cls._graph_scorer(backend, info_block)
        search._setup(scorer, action_set, horizon, max_policies)
        return search

    def _setup(
        self,
        scorer: _PolicyScorer,
        action_set: FiniteActionSet,
        horizon: int,
        max_policies: int | None,
    ) -> None:
        """Shared construction: validate, front-load ``A^H``, assert the certificate."""
        self._enumerate(scorer, action_set, horizon)
        _check_budget(scorer, action_set, self._horizon, max_policies)
        _check_index_ceiling(self._n_policies)
        # Front-load the enumeration once: one row per A^H sequence, decoded from its
        # index by the function the chunked path decodes with.
        self._policies = _decode_policies(
            jnp.arange(self._n_policies), action_set, self._horizon
        )  # (N, H, p)

        self._certificate = _certify(
            action_set, self._horizon, int(self._policies.shape[0])
        )
        # Compiled on the first plan and reused by every later one. The belief and the
        # preference are arguments, so a new one is new data rather than a new program.
        self._score_every_policy = jax.jit(partial(_score_every_policy, scorer))

    def evaluate(
        self, belief: Belief, preference: "Preference"
    ) -> EnumeratedSearchResult:
        """Score every enumerated policy, and return the argmin with its evidence.

        One ``vmap`` of the scorer over the front-loaded ``|A|^H`` policies, then a
        NaN-safe ``argmin`` (a NaN-scoring policy must not silently win). The scorer
        holds the flat model or the FFG backend as a closure constant, so the ``vmap``
        maps only over the policies. It is compiled once per search, so a plan after
        the first runs the compiled program.

        Args:
            belief: the belief to plan from.
            preference: the goal and precision the EFE scores against.

        Returns:
            The argmin, its score, its index, the certificate and the score vector.
        """
        scored = self._score_every_policy(self._policies, belief, preference)
        return EnumeratedSearchResult(
            best_policy=scored.best_policy,
            g=scored.g,
            best_g=scored.best_g,
            best_index=scored.best_index,
            certificate=self._certificate,
        )

    @property
    def policies(self) -> Float64[Array, "N H p"]:
        """The enumerated ``A^H`` policies, shape ``(|A|^H, H, p)`` (front-loaded).

        A result's ``best_index`` goes through ``policies_at``, which refuses the -1
        of a result with no winner. Used to index this array, -1 picks the last
        policy.
        """
        return self._policies

    @property
    def cost_per_plan(self) -> int:
        """Step evaluations one ``evaluate`` runs, ``|A|^H * H`` (RFC-001)."""
        return self._n_policies * self._horizon


# Measured on the crossover model at |A|^H = 9^6, sweeping 64 to 65536. Throughput has
# a broad plateau from 512 to 4096 and falls away on both sides: below it per-block
# overhead dominates, above it 65536 runs 3.4x slower than the plateau and carries a
# higher peak. Peak resident memory is flat at 0.42-0.43 GiB from 64 to 8192, so within
# the plateau the block size buys nothing back in residency and the choice is about
# rate alone. 4096 rather than the slightly faster 2048 because its spread across
# repeats is 2% against 20%, and PR-2 declares compute budgets off this rate.
_DEFAULT_CHUNK = 1 << 12


class _ArgminCarry(NamedTuple):
    """The running scalars the block loop carries: the argmin and the visited count."""

    best_g: Float64[Array, ""]
    best_index: Array
    visited: Array


class _BlockReducer(Protocol):
    """An extra fold over each block — the seam for evidence beyond the argmin.

    The search owns the enumeration, the certificate and the argmin (ADR-031). Anything
    else wanted over all ``|A|^H`` policies folds here rather than reading a
    materialised score vector, which is what keeps residency at ``O(chunk)``. Only
    ``_reduce`` takes one, so the seam is internal (ADR-069).

    ``update`` receives scores already masked: NaN mapped to ``+inf``, so a NaN-scoring
    policy cannot silently win, and padded lanes likewise. A reduction phrased as "how
    many below this baseline" or "the best of these" therefore needs no masking of its
    own, and ``valid`` is there for the ones that count rather than compare.
    """

    def init(self) -> Any: ...

    def update(
        self,
        carry: Any,
        g: Float64[Array, "chunk"],
        index: Array,
        valid: Array,
    ) -> Any: ...


class ChunkedEfeSearch(ExhaustiveEfeSearch):
    """Exhaustive EFE search over ``A^H`` in blocks, at ``O(chunk)`` residency.

    The same enumeration as
    [`EnumeratedEfeSearch`][cpomdp.enumeration.EnumeratedEfeSearch] and the same
    warrant, run so that neither the policy set nor the score vector is ever
    materialised. That is what puts sets past the front-loaded path's memory ceiling in
    reach, and the reason to have it is a set declared too large to hold, never a
    result that came out inconvenient.

    Three things make the two paths interchangeable rather than merely similar. Blocks
    are visited in increasing index order and the argmin updates on a strict ``<``, so
    ties resolve to the globally lowest index exactly as ``jnp.argmin`` does over the
    whole vector. Both paths decode an index to its policy with one function. And
    the trailing block is padded, with its padded lanes masked out of the argmin
    and out of ``visited``, so completeness is a count of real work rather than of
    array slots.

    It takes no policy budget. Its footprint is set by ``chunk``, and its time grows
    with ``cost_per_plan``.

    ``evaluate`` composes with ``jit``, ``vmap`` and ``lax.scan``. Inside one, the
    loop's count exists only once the program runs. The certificate then states the
    expected count, and a callback checks the loop's count in the run, so a miscount
    fails the run instead of certifying it.

    Args:
        model: the flat model to score on. It needs a control matrix.
        action_set: the declared actions, enumerated as every length-H sequence.
        horizon: H, the sequence length.
        chunk: the most policies one block scores, ``4096`` unless set. Clamped to
            ``|A|^H``. The answer does not depend on it.

    Raises:
        ValueError: If the model has no control matrix, if the action set's dimension
            or the horizon does not fit, if ``chunk`` is below 1, or if ``|A|^H``
            exceeds the index dtype's ceiling.
    """

    def __init__(
        self,
        model: LinearGaussianModel,
        action_set: FiniteActionSet,
        *,
        horizon: int,
        chunk: int = _DEFAULT_CHUNK,
    ) -> None:
        self._setup(self._flat_scorer(model), action_set, horizon, chunk)

    @classmethod
    def over_backend(
        cls,
        backend: "EfeBackend",
        action_set: FiniteActionSet,
        *,
        info_block: Sequence[int],
        horizon: int,
        chunk: int = _DEFAULT_CHUNK,
    ) -> "ChunkedEfeSearch":
        """Chunked search that scores on an FFG ``backend`` via ``policy_efe_ffg``.

        Raises:
            InadmissiblePartitionError: If the backend's carry partition severs an
                EFE-load-bearing edge.
            ValueError: If the backend has no control matrix, if ``info_block`` is
                empty, repeats an index or names one outside the joint state, if the
                action set's dimension or the horizon does not fit, or if ``chunk`` is
                below 1.
        """
        search = cls.__new__(cls)
        search._setup(
            cls._graph_scorer(backend, info_block), action_set, horizon, chunk
        )
        return search

    def _setup(
        self,
        scorer: _PolicyScorer,
        action_set: FiniteActionSet,
        horizon: int,
        chunk: int,
    ) -> None:
        """Validate, and size the enumeration against the index dtype's ceiling."""
        self._enumerate(scorer, action_set, horizon)
        if chunk < 1:
            raise ValueError(f"chunk must be >= 1, got {chunk}")

        self._scorer = scorer
        # A block wider than the enumeration pads the difference and scores it, so an
        # oversized default would charge real work for lanes it then masks away. At
        # |A|^H = 6561 against a 65536 block that is ten times the arithmetic for the
        # same answer. Clamping costs nothing and keeps the request an upper bound.
        self._chunk = min(int(chunk), self._n_policies)

        _check_index_ceiling(self._n_policies)
        # Compiled on the first plan and reused by every later one, as on the
        # front-loaded path.
        self._scan_compiled = jax.jit(partial(self._scan_blocks, reducer=None))

    def evaluate(self, belief: Belief, preference: "Preference") -> SearchResult:
        """Score every policy in blocks, and return the argmin with its evidence.

        One ``lax.scan`` over ``ceil(|A|^H / chunk)`` blocks. Each block decodes its own
        indices, scores them under one ``vmap``, and folds the result into running
        scalars, so nothing of size ``|A|^H`` is ever alive. The loop is compiled once
        per search, so a plan after the first runs the compiled program.

        Args:
            belief: the belief to plan from.
            preference: the goal and precision the EFE scores against.

        Returns:
            The argmin, its score, its index, and the certificate the loop's own count
            earns.
        """
        argmin, best_policy, _ = self._scan_compiled(belief, preference)
        return self._result(argmin, best_policy)

    def _reduce(
        self,
        belief: Belief,
        preference: "Preference",
        reducer: _BlockReducer | None,
    ) -> tuple[SearchResult, Any]:
        """``evaluate`` with an extra per-block fold, and the fold's final carry.

        The seam for evidence the argmin does not carry. The carry is ``Any`` because it
        is whatever the fold accumulates, and one seam with one consumer does not repay
        making the result and the protocol generic. Each call compiles its own fold.
        """
        scan = jax.jit(partial(self._scan_blocks, reducer=reducer))
        argmin, best_policy, extra = scan(belief, preference)
        return self._result(argmin, best_policy), extra

    def _result(
        self, argmin: _ArgminCarry, best_policy: Float64[Array, "H p"]
    ) -> SearchResult:
        """The loop's answer, certified on the count the loop itself carried."""
        return SearchResult(
            best_policy=best_policy,
            best_g=argmin.best_g,
            best_index=argmin.best_index,
            certificate=self._certify_count(argmin.visited),
        )

    def _certify_count(self, visited: Array) -> CompletenessCertificate:
        """The certificate the loop's count earns, checked now or when the run ends.

        Eagerly the count is read on the host and checked here. Under ``jit``, ``vmap``
        or ``scan`` it exists only once the program runs. The certificate then states
        the expected count, and a callback checks the loop's count against it in the
        run, so a miscount fails the run rather than certifying it.
        """
        try:
            count = int(np.asarray(visited))
        except (TracerArrayConversionError, ConcretizationTypeError):
            jax.debug.callback(self._check_count, visited)
            return _certify(self._action_set, self._horizon, self._n_policies)
        return _certify(self._action_set, self._horizon, count)

    def _check_count(self, visited: Array) -> None:
        """Raise from inside a compiled run if the loop's count is short."""
        for count in np.ravel(np.asarray(visited)):
            _certify(self._action_set, self._horizon, int(count))

    def _scan_blocks(
        self,
        belief: Belief,
        preference: "Preference",
        reducer: _BlockReducer | None,
    ) -> tuple[_ArgminCarry, Float64[Array, "H p"], Any]:
        """The block loop, returning the argmin, its decoded policy and the fold."""
        chunk, total = self._chunk, self._n_policies
        n_blocks = -(-total // chunk)  # ceil
        offsets = jnp.arange(chunk)
        score = lambda pol: self._scorer.score(belief, pol, preference)  # noqa: E731

        def run_block(carry, block_index):
            argmin, extra = carry
            index = block_index * chunk + offsets
            valid = index < total
            # Padded lanes score a real policy (index 0) rather than a made-up one, so
            # the kernel never sees an input the model does not define; the mask below
            # is what keeps them out of the answer.
            policies = _decode_policies(
                jnp.where(valid, index, 0), self._action_set, self._horizon
            )
            g = jax.vmap(score)(policies)
            g = jnp.where(jnp.isnan(g) | ~valid, jnp.inf, g)

            local = jnp.argmin(g)  # ties -> lowest local index
            # Strict `<`, and blocks arrive in increasing order, so the first
            # occurrence of a repeated minimum wins globally. That is the tie-break
            # `jnp.argmin` gives over the whole vector, reproduced block by block.
            better = g[local] < argmin.best_g
            argmin = _ArgminCarry(
                best_g=jnp.where(better, g[local], argmin.best_g),
                best_index=jnp.where(better, index[local], argmin.best_index),
                visited=argmin.visited + jnp.sum(valid),
            )
            if reducer is not None:
                extra = reducer.update(extra, g, index, valid)
            return (argmin, extra), None

        start = (
            _ArgminCarry(
                best_g=jnp.asarray(jnp.inf),
                best_index=jnp.asarray(0),
                visited=jnp.asarray(0),
            ),
            None if reducer is None else reducer.init(),
        )
        (argmin, extra), _ = jax.lax.scan(run_block, start, jnp.arange(n_blocks))
        best_g, best_index, best_policy = _winner_or_none(
            argmin.best_g,
            argmin.best_g,
            argmin.best_index,
            _decode_policies(argmin.best_index, self._action_set, self._horizon),
        )
        argmin = argmin._replace(best_g=best_g, best_index=best_index)
        return argmin, best_policy, extra

    @property
    def chunk(self) -> int:
        """The block size actually used, clamped to ``|A|^H``.

        The answer does not depend on it. The residency does, and so does the wasted
        arithmetic in the padded tail.
        """
        return self._chunk

    @property
    def n_blocks(self) -> int:
        """How many blocks the enumeration takes, the last one padded."""
        return -(-self._n_policies // self._chunk)

    @property
    def cost_per_plan(self) -> int:
        """Step evaluations one ``evaluate`` runs, ``|A|^H * H`` (RFC-001)."""
        return self._n_policies * self._horizon


class _SearchDriver(ABC):
    """What both drivers share: the search, its accounting and the last certificate."""

    def __init__(self, search: ExhaustiveEfeSearch) -> None:
        if not isinstance(search, ExhaustiveEfeSearch):
            raise TypeError(
                f"{type(self).__name__} drives an ExhaustiveEfeSearch, such as "
                f"EnumeratedEfeSearch or ChunkedEfeSearch, got {type(search).__name__}."
            )
        self._search = search
        self._certificate: CompletenessCertificate | None = None

    def _plan(self, belief: Belief, preference: "Preference") -> Float64[Array, "H p"]:
        """Run the search on ``belief`` and keep the certificate it earned.

        Raises:
            ValueError: If a concrete run finds no policy with a finite score.
        """
        result = self._search.evaluate(belief, preference)
        # A trace scores nothing, so only a run that read a real count keeps its
        # certificate. It cannot raise either, and returns the NaN policy instead.
        best_index = concrete(result.best_index)
        if best_index is None:
            return result.best_policy
        if best_index < 0:
            raise ValueError(
                "no policy scored finitely from this belief, so there is no action to "
                "take. Every score was NaN, which points at a noise that is not "
                "positive-definite where the rollouts reach."
            )
        self._certificate = result.certificate
        return result.best_policy

    @property
    def horizon(self) -> int:
        """The planning horizon H."""
        return self._search.horizon

    @property
    def warrant(self) -> Warrant:
        """``PROVED``, inherited from the exhaustive search."""
        return self._search.warrant

    @property
    def certificate(self) -> CompletenessCertificate | None:
        """The last plan's completeness certificate (ADR-030), ``None`` before one.

        The chunked search earns its certificate by running, so a driver has one only
        once it has planned.
        """
        return self._certificate

    @property
    def cost_per_plan(self) -> int:
        """Step evaluations one search costs, ``|A|^H * H`` (RFC-001)."""
        return self._search.cost_per_plan

    @property
    @abstractmethod
    def replan_interval(self) -> int:
        """Agent steps between two plans."""

    @property
    def cost_per_cycle(self) -> int:
        """Step evaluations per agent step, one plan spread over its interval."""
        return self.cost_per_plan // self.replan_interval


class RecedingHorizonSelector(_SearchDriver):
    """Re-plans every step and applies the first action of the best sequence.

    An ``ActionSelector`` over an exhaustive search. Every ``select`` re-runs the
    full ``A^H`` search on the current belief, so the belief drives every choice. It
    costs a whole plan per step, ``cost_per_cycle = |A|^H * H``. A measurement has to
    declare which mode it ran, so this and ``OpenLoopSelector`` stay separate (ADR-031).

    Args:
        search: the search to plan with. Either shipped search gives the same
            actions.

    Raises:
        TypeError: If ``search`` is not an
            [`ExhaustiveEfeSearch`][cpomdp.enumeration.ExhaustiveEfeSearch].
    """

    def select(self, belief: Belief, preference: "Preference") -> Float64[Array, "p"]:
        """Re-plan on ``belief`` and return the first action of the best sequence."""
        return self._plan(belief, preference)[0]

    @property
    def replan_interval(self) -> int:
        """Agent steps between two plans. 1: this driver plans every step."""
        return 1


class OpenLoopSelector(_SearchDriver):
    """Plans once, then applies the whole best sequence without looking at beliefs.

    An ``ActionSelector`` over an exhaustive search. It plans on the belief of the
    first ``select`` and returns the plan's actions in order, ignoring the interim
    beliefs. It re-plans only when the committed sequence is spent, or after ``reset``.
    Ignoring the interim beliefs is what makes it open-loop. One plan covers H steps, so
    ``cost_per_cycle = |A|^H``. A measurement has to declare which mode it ran, so this
    and ``RecedingHorizonSelector`` stay separate (ADR-031).

    Stateful: it holds the committed plan and a step counter. The ``Agent`` drives it
    eagerly, one ``select`` per step, so the mutation is safe.

    Args:
        search: the search to plan with. Either shipped search gives the same
            actions.

    Raises:
        TypeError: If ``search`` is not an
            [`ExhaustiveEfeSearch`][cpomdp.enumeration.ExhaustiveEfeSearch].
    """

    def __init__(self, search: ExhaustiveEfeSearch) -> None:
        super().__init__(search)
        self._committed: Float64[Array, "H p"] | None = None
        self._step = 0

    def select(self, belief: Belief, preference: "Preference") -> Float64[Array, "p"]:
        """Return the next committed action, re-planning on ``belief`` when spent."""
        if self._committed is None or self._step >= self._search.horizon:
            self._committed = self._plan(belief, preference)
            self._step = 0
        action = self._committed[self._step]
        self._step += 1
        return action

    def reset(self) -> None:
        """Drop the committed plan and its certificate, so the next ``select`` plans."""
        self._committed = None
        self._step = 0
        self._certificate = None

    @property
    def replan_interval(self) -> int:
        """Agent steps between two plans, H. It commits to the whole sequence."""
        return self._search.horizon
