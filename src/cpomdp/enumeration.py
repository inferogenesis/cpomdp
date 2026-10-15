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

Internal seam: imported from ``cpomdp.enumeration``, not re-exported at the top level.
"""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float64

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
    "ChunkedSearchResult",
    "CompletenessCertificate",
    "EnumeratedEfeSearch",
    "EnumeratedSearchResult",
    "FiniteActionSet",
    "IncompleteEnumerationError",
    "OpenLoopSelector",
    "RecedingHorizonSelector",
    "SearchWarrant",
]


# The search vocabulary predates the shared one, and named two of its three levels. It
# is now that enum, so `SearchWarrant.PROVED` and `Warrant.PROVED` are one object and
# `EnumeratedEfeSearch.warrant` keeps its return type. A search earns `PROVED` from a
# full enumeration (Prover 3 · enumeration) and `CORROBORATED` from a grid sample.
SearchWarrant = Warrant


class IncompleteEnumerationError(Exception):
    """Fewer than ``|A|^H`` policies enumerated: a sample, not a decision."""


@dataclass(frozen=True, init=False)
class FiniteActionSet:
    """A finite, declared, **versioned** set of actions to enumerate over.

    A distinct type from the continuous grid's ``(lo, hi, n_candidates)`` config, so the
    enumerated (``PROVED``) and sampled (``CORROBORATED``) families cannot be mixed up
    at a call site. Construction-time only (not a pytree).

    Args:
        actions: the declared actions, shape ``(A, p)`` — ``A`` actions of dimension
            ``p``. At least two (a set of one is no choice to search).
        version: a non-empty tag, so an action added later shows up in the diff.
    """

    actions: Float64[Array, "A p"]
    version: str

    def __init__(self, actions, *, version: str) -> None:
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
        warrant=SearchWarrant.PROVED,
        action_set_size=action_set.size,
        horizon=horizon,
        action_set_version=action_set.version,
    )


class EnumeratedSearchResult(NamedTuple):
    """The outcome of one exhaustive search.

    Attributes:
        best_policy: the argmin sequence, shape ``(H, p)``.
        g: the EFE of every enumerated policy, shape ``(|A|^H,)``. The search
            computes it anyway, so it costs nothing extra.
    """

    best_policy: Float64[Array, "H p"]
    g: Float64[Array, "N"]


class _PolicyScorer(Protocol):
    """Scores one policy to its summed EFE ``G`` — the seam that swaps flat for FFG.

    The enumerated search owns the enumeration, the certificate, and the argmin; *how* a
    policy is scored is injected here (ADR-034). ``action_dim`` lets the search validate
    the declared action set against the scorer's control dimension.
    """

    action_dim: int

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

    def score(
        self, belief: Belief, policy: Float64[Array, "H p"], preference: "Preference"
    ) -> Float64[Array, ""]:
        return _policy_efe_ffg(
            self.backend, belief, policy, preference, self.info_block
        )[0]


class EnumeratedEfeSearch:
    """Exhaustive EFE search over ``A^H`` for a declared finite action set (ADR-031).

    Front-loads (ADR-002) the full ``|A|^H`` enumeration of *varying* sequences at
    construction, with its completeness certificate; ``evaluate`` scores them all and
    returns the argmin policy and the full ``G`` vector. Because the set is finite and
    fully enumerated the search **decides** its universal — warrant ``PROVED``, distinct
    from ``EFESelector``'s ``CORROBORATED`` grid.

    Scoring is a strategy (ADR-034): the default constructor scores on a flat
    ``LinearGaussianModel`` (``policy_efe``); ``over_backend`` scores on an FFG backend
    (``policy_efe_ffg``, aimed at a node's block) so the same exhaustive search runs the
    coupled-tree crossover model a flat model cannot express. The enumeration,
    the certificate, and the cost are shared — only the per-policy scoring differs.

    Unlike ``EFESelector`` this supports ``p >= 1`` and *varying* sequences: it can
    express a detour-then-exploit policy the constant-action family cannot.
    """

    def __init__(
        self,
        model: LinearGaussianModel,
        action_set: FiniteActionSet,
        *,
        horizon: int,
    ) -> None:
        if model.control_matrix is None:
            raise ValueError(
                "EnumeratedEfeSearch needs a model with a control matrix; an "
                "action has no effect on a control-free (pure-tracking) model."
            )
        self._setup(_FlatScorer(model), action_set, horizon)

    @classmethod
    def over_backend(
        cls,
        backend: "EfeBackend",
        action_set: FiniteActionSet,
        *,
        info_block: Sequence[int],
        horizon: int,
    ) -> "EnumeratedEfeSearch":
        """Exhaustive search that scores on an FFG ``backend`` via ``policy_efe_ffg``.

        The FFG counterpart of the flat constructor: it enumerates the same ``A^H`` set
        with the same completeness certificate, but scores each policy on a coupling
        graph with the epistemic aimed at ``info_block`` — a node's block (via
        ``backend.block``) or the whole state. This runs the coupled-tree model the flat
        constructor cannot express.

        Raises:
            InadmissiblePartitionError: If the backend's carry partition severs an
                EFE-load-bearing edge.
            ValueError: If the backend has no control matrix, if ``info_block`` is
                empty, repeats an index or names one outside the joint state, or if
                the action set's dimension or the horizon does not fit.
        """
        if backend.model.control_matrix is None:
            raise ValueError(
                "EnumeratedEfeSearch.over_backend needs a backend with a control "
                "matrix; an action has no effect without one."
            )
        search = cls.__new__(cls)
        target = _checked_info_block(backend, info_block)
        search._setup(_FfgScorer(backend, target), action_set, horizon)
        return search

    def _setup(
        self, scorer: _PolicyScorer, action_set: FiniteActionSet, horizon: int
    ) -> None:
        """Shared construction: validate, front-load ``A^H``, assert the certificate."""
        _validate_spec(scorer, action_set, horizon)
        self._scorer = scorer
        self._action_set = action_set
        self._horizon = int(horizon)
        # Front-load the enumeration: one row per A^H sequence, built once at
        # construction (host-side itertools; never in the hot path).
        index_grid = np.array(
            list(itertools.product(range(action_set.size), repeat=self._horizon))
        )  # (N, H)
        self._policies = action_set.actions[jnp.asarray(index_grid)]  # (N, H, p)

        self._certificate = _certify(
            action_set, self._horizon, int(self._policies.shape[0])
        )

    def evaluate(
        self, belief: Belief, preference: "Preference"
    ) -> EnumeratedSearchResult:
        """Score every enumerated policy; return the argmin and the full ``G`` vector.

        One ``vmap`` of the scorer over the front-loaded ``|A|^H`` policies, then a
        NaN-safe ``argmin`` (a NaN-scoring policy must not silently win). The scorer
        holds the flat model or the FFG backend as a closure constant, so the ``vmap``
        maps only over the policies.
        """
        g = jax.vmap(lambda pol: self._scorer.score(belief, pol, preference))(
            self._policies
        )
        best = jnp.argmin(jnp.where(jnp.isnan(g), jnp.inf, g))
        return EnumeratedSearchResult(best_policy=self._policies[best], g=g)

    @property
    def policies(self) -> Float64[Array, "N H p"]:
        """The enumerated ``A^H`` policies, shape ``(|A|^H, H, p)`` (front-loaded)."""
        return self._policies

    @property
    def n_policies(self) -> int:
        """The enumerated policy count ``|A|^H`` — attributable work (RFC-001)."""
        return int(self._policies.shape[0])

    @property
    def horizon(self) -> int:
        """The sequence length H."""
        return self._horizon

    @property
    def cost_per_cycle(self) -> int:
        """Per-cycle step-evals = ``|A|^H * H`` — the honest exponential cost."""
        return self.n_policies * self._horizon

    @property
    def warrant(self) -> SearchWarrant:
        """``PROVED`` — a finite set enumerated in full decides its universal."""
        return SearchWarrant.PROVED

    @property
    def certificate(self) -> CompletenessCertificate:
        """The front-loaded completeness certificate (ADR-030)."""
        return self._certificate


# Measured on the crossover model at |A|^H = 9^6, sweeping 64 to 65536. Throughput has
# a broad plateau from 512 to 4096 and falls away on both sides: below it per-block
# overhead dominates, above it 65536 runs 3.4x slower than the plateau and carries a
# higher peak. Peak resident memory is flat at 0.42-0.43 GiB from 64 to 8192, so within
# the plateau the block size buys nothing back in residency and the choice is about
# rate alone. 4096 rather than the slightly faster 2048 because its spread across
# repeats is 2% against 20%, and PR-2 declares compute budgets off this rate.
_DEFAULT_CHUNK = 1 << 12


def _decode_policies(
    indices: Array, action_set: FiniteActionSet, horizon: int
) -> Float64[Array, "... H p"]:
    """Policy sequences for global policy indices, without materialising the set.

    Index ``i`` is read as a base-``|A|`` numeral, most significant digit first, which
    is exactly the order ``itertools.product`` emits on the front-loaded path. The two
    paths therefore agree on *which* policy each index names. Without that a cross-path
    argmin comparison would be comparing labels rather than plans.
    """
    powers = jnp.asarray(
        [action_set.size ** (horizon - 1 - step) for step in range(horizon)]
    )
    digits = (indices[..., None] // powers) % action_set.size
    return action_set.actions[digits]


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


class ChunkedSearchResult(NamedTuple):
    """The outcome of one chunked exhaustive search.

    No score vector, which is the point: at ``17^7`` one would cost 3.28 GB.

    Attributes:
        best_policy: the argmin sequence, shape ``(H, p)``.
        best_g: its summed EFE — G.
        best_index: its position in the enumeration, for comparison against the
            front-loaded path's argmin.
        visited: how many policies the loop actually scored, counted in the loop
            rather than read off an array's length.
        certificate: the completeness certificate (ADR-030). It arrives on the result
            rather than at construction, because the count it certifies is what the
            loop produces.
    """

    best_policy: Float64[Array, "H p"]
    best_g: Float64[Array, ""]
    best_index: int
    visited: int
    certificate: CompletenessCertificate


class ChunkedEfeSearch:
    """Exhaustive EFE search over ``A^H`` in blocks, at ``O(chunk)`` residency.

    The same enumeration as [`EnumeratedEfeSearch`][cpomdp.enumeration] and the same
    warrant, run so that neither the policy set nor the score vector is ever
    materialised. That is what puts sets past the front-loaded path's memory ceiling in
    reach, and the reason to have it is a set declared too large to hold, never a
    result that came out inconvenient.

    Three things make the two paths interchangeable rather than merely similar. Blocks
    are visited in increasing index order and the argmin updates on a strict ``<``, so
    ties resolve to the globally lowest index exactly as ``jnp.argmin`` does over the
    whole vector. Indices decode to the same policies ``itertools.product`` emits.
    And the trailing block is padded, with its padded lanes masked out of the argmin
    and out of ``visited``, so completeness is a count of real work rather than of
    array slots.
    """

    def __init__(
        self,
        model: LinearGaussianModel,
        action_set: FiniteActionSet,
        *,
        horizon: int,
        chunk: int = _DEFAULT_CHUNK,
    ) -> None:
        if model.control_matrix is None:
            raise ValueError(
                "ChunkedEfeSearch needs a model with a control matrix; an action has "
                "no effect on a control-free (pure-tracking) model."
            )
        self._setup(_FlatScorer(model), action_set, horizon, chunk)

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
        if backend.model.control_matrix is None:
            raise ValueError(
                "ChunkedEfeSearch.over_backend needs a backend with a control "
                "matrix; an action has no effect without one."
            )
        search = cls.__new__(cls)
        target = _checked_info_block(backend, info_block)
        search._setup(_FfgScorer(backend, target), action_set, horizon, chunk)
        return search

    def _setup(
        self,
        scorer: _PolicyScorer,
        action_set: FiniteActionSet,
        horizon: int,
        chunk: int,
    ) -> None:
        """Validate, and size the enumeration against the index dtype's ceiling."""
        _validate_spec(scorer, action_set, horizon)
        if chunk < 1:
            raise ValueError(f"chunk must be >= 1, got {chunk}")

        self._scorer = scorer
        self._action_set = action_set
        self._horizon = int(horizon)
        self._n_policies = action_set.size**self._horizon
        # A block wider than the enumeration pads the difference and scores it, so an
        # oversized default would charge real work for lanes it then masks away. At
        # |A|^H = 6561 against a 65536 block that is ten times the arithmetic for the
        # same answer. Clamping costs nothing and keeps the request an upper bound.
        self._chunk = min(int(chunk), self._n_policies)

        ceiling = int(jnp.iinfo(jnp.arange(1).dtype).max)
        if self._n_policies > ceiling:
            raise ValueError(
                f"|A|^H = {self._n_policies} exceeds the index dtype's ceiling "
                f"{ceiling}. Enable x64, or declare a smaller set: an index that "
                "wraps would enumerate the wrong policies and still certify."
            )

    def reduce(self, belief: Belief, preference: "Preference") -> ChunkedSearchResult:
        """Score every policy in blocks; return the argmin and the loop's own count.

        One ``lax.scan`` over ``ceil(|A|^H / chunk)`` blocks. Each block decodes its own
        indices, scores them under one ``vmap``, and folds the result into running
        scalars, so nothing of size ``|A|^H`` is ever alive.

        Args:
            belief: the belief to plan from.
            preference: the goal and precision the EFE scores against.

        Returns:
            The argmin, its index and score, the visited count and the certificate.
        """
        return self._reduce(belief, preference, None)[0]

    def _reduce(
        self,
        belief: Belief,
        preference: "Preference",
        reducer: _BlockReducer | None,
    ) -> tuple[ChunkedSearchResult, Any]:
        """``reduce`` with an extra per-block fold, and the fold's final carry.

        The seam for evidence the argmin does not carry. The carry is ``Any`` because it
        is whatever the fold accumulates, and one seam with one consumer does not repay
        making the result and the protocol generic.
        """
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

        return ChunkedSearchResult(
            best_policy=_decode_policies(
                argmin.best_index, self._action_set, self._horizon
            ),
            best_g=argmin.best_g,
            best_index=int(argmin.best_index),
            visited=int(argmin.visited),
            certificate=_certify(self._action_set, self._horizon, int(argmin.visited)),
        ), extra

    def policies_at(self, indices: Array) -> Float64[Array, "... H p"]:
        """The policies named by ``indices``, decoded rather than looked up.

        The counterpart of ``best_index``: a chunked run reports where its winner sat
        in the enumeration, and this turns that back into a plan. Also how a caller
        checks that an index means the same policy here as on the front-loaded path.
        """
        return _decode_policies(jnp.asarray(indices), self._action_set, self._horizon)

    @property
    def n_policies(self) -> int:
        """The policy count ``|A|^H`` — attributable work (RFC-001)."""
        return self._n_policies

    @property
    def horizon(self) -> int:
        """The sequence length H."""
        return self._horizon

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
    def cost_per_cycle(self) -> int:
        """Per-cycle step-evals = ``|A|^H * H`` — the honest exponential cost."""
        return self._n_policies * self._horizon

    @property
    def warrant(self) -> SearchWarrant:
        """``PROVED`` — a finite set enumerated in full decides its universal."""
        return SearchWarrant.PROVED


class RecedingHorizonSelector:
    """Receding-horizon driver over ``EnumeratedEfeSearch`` (an ``ActionSelector``).

    Every ``select`` re-runs the full ``A^H`` search on the current belief and returns
    the first action of the best sequence. The belief drives every choice, so the plan
    is recomputed each step. Kept separate from ``OpenLoopSelector`` because a
    matched-horizon comparison depends on which mode is running (ADR-031).
    """

    def __init__(self, search: EnumeratedEfeSearch) -> None:
        self._search = search

    def select(self, belief: Belief, preference: "Preference") -> Float64[Array, "p"]:
        """Re-plan on ``belief`` and return the first action of the best sequence."""
        return self._search.evaluate(belief, preference).best_policy[0]

    @property
    def horizon(self) -> int:
        """The planning horizon H."""
        return self._search.horizon

    @property
    def warrant(self) -> SearchWarrant:
        """``PROVED`` — inherited from the enumerated search."""
        return self._search.warrant

    @property
    def certificate(self) -> CompletenessCertificate:
        """The search's completeness certificate (ADR-030)."""
        return self._search.certificate

    @property
    def cost_per_plan(self) -> int:
        """Step-evals for one search = ``|A|^H * H`` (RFC-001 attributable work)."""
        return self._search.cost_per_cycle

    @property
    def replan_interval(self) -> int:
        """Cycles between re-plans. 1: this driver plans every step."""
        return 1

    @property
    def cost_per_cycle(self) -> int:
        """Per-cycle step-evals = ``|A|^H * H``. It plans every cycle."""
        return self.cost_per_plan // self.replan_interval


class OpenLoopSelector:
    """Open-loop driver over ``EnumeratedEfeSearch`` (an ``ActionSelector``).

    Plans once on the belief of the first ``select``, commits to the whole best
    sequence, and returns its actions in order while ignoring the interim beliefs. It
    re-plans only when the committed sequence is exhausted, or after ``reset``. Ignoring
    the interim beliefs is what makes it open-loop, and the distinction from
    ``RecedingHorizonSelector`` is what a matched-horizon comparison turns on (ADR-031).

    Stateful: it holds the committed plan and a step counter. The ``Agent`` drives it
    eagerly, one ``select`` per step, so the mutation is safe.
    """

    def __init__(self, search: EnumeratedEfeSearch) -> None:
        self._search = search
        self._plan: Float64[Array, "H p"] | None = None
        self._step = 0

    def select(self, belief: Belief, preference: "Preference") -> Float64[Array, "p"]:
        """Return the next committed action, re-planning on ``belief`` when spent."""
        if self._plan is None or self._step >= self._search.horizon:
            self._plan = self._search.evaluate(belief, preference).best_policy
            self._step = 0
        action = self._plan[self._step]
        self._step += 1
        return action

    def reset(self) -> None:
        """Drop the committed plan so the next ``select`` re-plans from scratch."""
        self._plan = None
        self._step = 0

    @property
    def horizon(self) -> int:
        """The planning horizon H, which is also the re-plan interval."""
        return self._search.horizon

    @property
    def warrant(self) -> SearchWarrant:
        """``PROVED`` — inherited from the enumerated search."""
        return self._search.warrant

    @property
    def certificate(self) -> CompletenessCertificate:
        """The search's completeness certificate (ADR-030)."""
        return self._search.certificate

    @property
    def cost_per_plan(self) -> int:
        """Step-evals for one search = ``|A|^H * H`` (RFC-001 attributable work)."""
        return self._search.cost_per_cycle

    @property
    def replan_interval(self) -> int:
        """Cycles between re-plans = H. It commits to the whole sequence."""
        return self._search.horizon

    @property
    def cost_per_cycle(self) -> int:
        """Amortised per-cycle step-evals = ``|A|^H`` (one ``|A|^H * H`` plan per H)."""
        return self.cost_per_plan // self.replan_interval
