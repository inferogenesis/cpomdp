"""One-step Expected Free Energy (EFE) for the linear-Gaussian regime.

This module computes ``G(a)`` — the Expected Free Energy of taking action ``a``
from the current belief — and its decomposition into a *pragmatic* (goal-seeking)
and an *epistemic* (information-seeking) part. Minimising ``G`` over actions is
how an agent chooses what to do.

================================================================================
THE DECISION THIS FILE ENCODES  (see DECISIONS.md ADR-005)
================================================================================
There is *no single agreed formula* for EFE in the active-inference literature —
the pragmatic term in particular has at least three forms in circulation, and
sources disagree on signs and on whether risk is a cross-entropy or a KL. One
route is chosen here deliberately, and the two questions that most affect what
the objective *does* are settled rather than assumed: the cross-entropy and
KL-risk groupings are the same objective (the derivation is below, and both forms
are reproduced by the tests), and the sign convention follows from that. What
stays genuinely open is flagged ``# FRAGILE(lit):`` — the preference domain, and
the fork between the full pragmatic term and a mean-only one.

--------------------------------------------------------------------------------
THE LOCKED DEFINITION  (decomposition (b): cross-entropy − info-gain)
--------------------------------------------------------------------------------
Given belief ``(μ, Σ)``, action ``a``, model ``(A, B, Q)`` with sensor ``(C, R)``,
and an OBSERVATION-space preference ``(g, Λ)`` (goal observation ``g``, precision
``Λ``):

    predict:    μ⁺ = A·μ + B·a            Σ⁺ = A·Σ·Aᵀ + Q
    sense:      (C, R) = observation.linearize(μ⁺)
                o⁺ = C·μ⁺                  S = C·Σ⁺·Cᵀ + R          # predicted-obs cov
    pragmatic:  ½·(o⁺ − g)ᵀ·Λ·(o⁺ − g)  +  ½·tr(Λ·S)
    epistemic:  ½·(ln det S − ln det R)            # = I(state; obs) ≥ 0, info gain
    G = pragmatic − epistemic                      # minimise: low cost, high info

``S`` is computed once and feeds BOTH terms — there is no n×n work and no Σ_post
or Kalman gain in the one-step EFE (those are only needed for belief propagation
in the H-step rollout, ``policy_efe``). The epistemic identity
``½(ln det Σ⁺ − ln det Σ_post) = ½ ln(det S / det R)`` lets us stay in m×m.

--------------------------------------------------------------------------------
THE FRAGILE CHOICES  (grep: ``FRAGILE(lit)``)
--------------------------------------------------------------------------------
1. Preference domain = OBSERVATIONS, not states. Canonical pymdp/Friston puts
   preferences over outcomes, so this is the faithful choice — but it diverges
   from ADR-003's collapse argument, which is written in state space, and it does
   NOT match the state-space ``goal`` the LQR path currently uses. Reconciling the
   two consumers of ``Preference`` (state-space LQR vs obs-space EFE) is an OPEN
   design point; for C = I (fully observed) they coincide.
2. Pragmatic = FULL form (mean + ½tr(ΛS)), i.e. cross-entropy −E_Q[ln P(o)] up to a
   fixed constant. Cross-entropy paired with −info-gain (as here) and KL-risk
   paired with +ambiguity are the SAME objective — not a behavioural fork; the two
   groupings differ by a constant, and each carries one half of the same entropy
   gap. The genuine literature fork is FULL vs *mean-only* (drop the ½tr(ΛS) term →
   an agent blind to predicted-observation variance/ambiguity). The *forbidden mix*
   (KL-risk pragmatic − info-gain) double-counts H[Q(o)]: a bug, not an option.
3. Epistemic = STATE information gain (salience), not parameter information gain
   (novelty). Only the state info-gain I(state; obs) is computed; parameter/novelty
   EFE is out of scope.
4. The sensor is linearized at μ⁺ (the predicted mean). A fixed sensor is its own
   linearization everywhere, so the point is immaterial there — but for a
   state-dependent ``R(x)`` it is the modelling commitment the whole epistemic term
   rests on: the action moves μ⁺, μ⁺ chooses R, and R sets the information. It is a
   first-order rule, and a plug-in one: it drops the correction for R's curvature
   across the belief's spread.
5. Sign convention: G is MINIMISED; ``pragmatic`` is a cost (lower better) and
   ``epistemic`` is a value (higher better), so G = pragmatic − epistemic.

NOT IMPLEMENTED (named seams): the *mean-only* pragmatic (drops ½tr(ΛS); the real
literature alternative — an ambiguity-blind agent); parameter/novelty info gain.
The KL-risk grouping is NOT a separate option: paired correctly (+ambiguity) it is
this same G; paired with −info-gain it double-counts H[Q(o)] (a bug to avoid).

--------------------------------------------------------------------------------
THE DATA FLOW  (top → bottom: what goes in → what comes out)
--------------------------------------------------------------------------------
    IN ── model=(A,control,Q,sensor)  belief=(μ,Σ)  action=a  preference=(g,Λ)
      │
      ▼    GUARD      control is None?  ──►  raise ValueError
      │
      ▼    PREDICT    μ⁺ = A·μ + control·a     (action enters HERE only)
      │               Σ⁺ = A·Σ·Aᵀ + Q          (this step's Σ⁺ does not see the action)
      │
      ▼    SENSE      (C, R) = linearize(μ⁺)   or fixed (model.C, model.R)
      │               o⁺ = C·μ⁺
      │               S  = C·Σ⁺·Cᵀ + R         (predicted-obs cov; computed ONCE)
      │
      ├──► PRAGMATIC  ½·(o⁺−g)ᵀ·Λ·(o⁺−g) + ½·tr(Λ·S)    (cost,  lower better)
      ├──► EPISTEMIC  ½·(ln det S − ln det R)            (value, higher better)
      │
      ▼    G = pragmatic − epistemic
      │
      ▼    OUT        return EfeResult(g=G, pragmatic=…, epistemic=…)

"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float64

from cpomdp.backends.base import InadmissiblePartitionError
from cpomdp.types import Belief, LinearGaussianModel

if TYPE_CHECKING:
    # Type-only imports: the kernel reads preference.goal/.precision and the backend's
    # predict/observation methods by duck-typing, so it does NOT depend on selection or
    # on a backend at runtime. Its one runtime import from the backends package is the
    # exception the graph rollouts raise, and `backends.base` imports nothing that
    # imports this module, so EFESelector / FfgEfeSelector import it without a cycle.
    from cpomdp.backends.base import EfeBackend
    from cpomdp.selection import Preference

__all__ = [
    "EfeResult",
    "PolicyEfeTrace",
    "expected_free_energy",
    "policy_efe",
    "policy_efe_ffg",
    "policy_efe_ffg_trace",
    "policy_efe_trace",
]


class EfeResult(NamedTuple):
    """Expected free energy and the two terms it is made of.

    Every scoring function in this module returns one. ``g`` comes first, so
    ``result[0]`` is ``G``. A ``NamedTuple`` is a JAX pytree with no registration, so
    a ``vmap`` over candidate actions returns one record with every field batched.

    Attributes:
        g: The expected free energy ``G = pragmatic − epistemic``. Lower is preferred.
        pragmatic: The goal term, a cost.
        epistemic: The information the observation carries about the state, a value.
    """

    g: Float64[Array, ""]
    pragmatic: Float64[Array, ""]
    epistemic: Float64[Array, ""]


class PolicyEfeTrace(NamedTuple):
    """One H-step rollout's per-step record, each field stacked along a leading H axis.

    ``policy_efe`` sums each step's ``(g, pragmatic, epistemic)`` and discards the
    moments it propagates between steps; ``policy_efe_trace`` returns this instead, so
    the per-step quantities the ``lax.scan`` already computes can be inspected — the
    covariance trajectory a policy carries, the per-step pragmatic/epistemic split, and
    the matrices whose conditioning the rollout depends on.

    A ``NamedTuple`` because it is a JAX pytree with no registration: ``lax.scan`` emits
    it directly as its ``ys`` (stacking each field along the leading H axis) and it
    survives ``jit`` / ``vmap`` / ``grad``, gaining any outer ``vmap`` axis too. That is
    the difference from ``_EfeStep``, which is a plain dataclass precisely because it
    never crosses a scan/transform boundary.

    Moments use this module's convention: ``⁺`` marks the one-step *prediction* (after
    the dynamics and action, before any observation); ``Σ_post`` is the covariance after
    the predict-only Kalman contraction.

    Attributes:
        g: Per-step EFE ``G = pragmatic − epistemic`` — (H,).
        pragmatic: Per-step pragmatic (goal) term — (H,).
        epistemic: Per-step epistemic (state information gain) term — (H,).
        mu_pred: Predicted mean μ⁺ = ``A·μ + control·a`` — (H, n).
        sigma_pred: Predicted covariance Σ⁺ = ``A·Σ·Aᵀ + Q`` — (H, n, n).
        sigma_post: Post-observation covariance Σ_post, the contraction carried to the
            next step — (H, n, n).
        s: Innovation covariance S = ``C·Σ⁺·Cᵀ + R`` (observation space) — (H, m, m).
    """

    g: Float64[Array, "H"]
    pragmatic: Float64[Array, "H"]
    epistemic: Float64[Array, "H"]
    mu_pred: Float64[Array, "H n"]  # μ⁺ (predicted mean)
    sigma_pred: Float64[Array, "H n n"]  # Σ⁺ (predicted cov)
    sigma_post: Float64[Array, "H n n"]  # Σ_post (Kalman-contracted, post-observation)
    s: Float64[Array, "H m m"]  # S (innovation cov)


@dataclass(frozen=True)
class _EfeStep:
    """One EFE step: the public split plus the moments the rollout propagates.

    ``expected_free_energy`` exposes only ``g`` + ``{pragmatic, epistemic}``; the
    H-step rollout (``policy_efe``, B2) additionally needs ``mu_pred``/``sigma_pred``/
    ``s`` (μ⁺, Σ⁺, S) to propagate the belief — but NOT ``C`` (it fetches its own where
    it propagates). Consumed locally; it never crosses a ``jit``/``vmap``/``scan``
    boundary, so it needs no pytree registration.
    """

    g: Float64[Array, ""]
    pragmatic: Float64[Array, ""]
    epistemic: Float64[Array, ""]
    mu_pred: Float64[Array, "n"]
    sigma_pred: Float64[Array, "n n"]
    s: Float64[Array, "m m"]


def expected_free_energy(
    model: LinearGaussianModel,
    belief: Belief,
    action: Float64[Array, "p"],
    preference: "Preference",
) -> EfeResult:
    """Expected Free Energy of taking ``action`` from ``belief``, and its split.

    Computes ``G = pragmatic − epistemic`` for the locked linear-Gaussian
    definition documented at the top of this module. Pure ``jnp``, so it composes
    under ``jit``/``vmap``/``grad`` — in particular ``vmap``/``grad`` over a batch
    of candidate ``action`` vectors (with ``model``/``belief``/``preference`` held
    fixed), which is how ``EFESelector`` will search.

    Args:
        model: The generative model. Must have a control matrix (an action has no
            meaning without one). Its ``observation_model`` supplies the local
            ``(C, R)``; ``None`` means the fixed sensor
            ``(observation_matrix, observation_noise)``.
        belief: The current belief ``(μ, Σ)``.
        action: The candidate action ``a``, shape ``(p,)``.
        preference: The goal as an OBSERVATION-space ``Preference`` — ``goal`` is a
            preferred observation ``g`` (shape ``(m,)``) and ``precision`` is ``Λ``
            (shape ``(m, m)``). See FRAGILE(lit) #1 in the module docstring.

    Returns:
        The scalar EFE ``g`` and its two non-negative terms, ``pragmatic`` and
        ``epistemic``. Lower ``g`` is preferred.

    Raises:
        ValueError: If the model has no control matrix.
    """
    if model.control_matrix is None:
        raise ValueError(
            "expected_free_energy needs a model with a control matrix; an action "
            "has no effect on a control-free (pure-tracking) model."
        )
    control_matrix = model.control_matrix  # narrowed to Array by the guard above
    action = jnp.asarray(action, dtype=float)
    step = _efe_step(
        model,
        belief.mean,
        belief.cov,
        control_matrix,
        action,
        preference.goal,
        preference.precision,
    )
    return EfeResult(g=step.g, pragmatic=step.pragmatic, epistemic=step.epistemic)


def policy_efe(
    model: LinearGaussianModel,
    belief: Belief,
    policy: Float64[Array, "H p"],
    preference: "Preference",
) -> EfeResult:
    """Summed EFE of a horizon-H ``policy`` from ``belief`` (the rollout seam).

    Internal — ``EFESelector`` searches over it. A ``lax.scan`` over the ``policy``
    rows sums each step's ``G`` while propagating the belief predict-only between
    steps (the mean follows the prediction; the covariance contracts by the Kalman
    update, reusing the moments ``_efe_step`` returns). ``H`` is ``policy.shape[0]``;
    at ``H = 1`` it reduces exactly to ``expected_free_energy``. Composes under
    ``jit`` / ``vmap`` / ``grad``.

    Returns:
        The EFE and its two terms, each summed over the horizon.
    """
    if model.control_matrix is None:
        raise ValueError(
            "policy_efe needs a model with a control matrix; an action has no "
            "effect on a control-free (pure-tracking) model."
        )
    control_matrix = model.control_matrix
    goal, precision = preference.goal, preference.precision
    policy = jnp.asarray(policy, dtype=float)
    body = _rollout_body(model, control_matrix, goal, precision)

    # Lean projection: keep only the three scalars in the scan's ys, so EFESelector's
    # vmap over |A|^H candidate policies never stacks the H×n×n covariances.
    def scalars_only(carry, action):
        new_carry, tr = body(carry, action)
        return new_carry, (tr.g, tr.pragmatic, tr.epistemic)

    _, (gs, prags, epis) = lax.scan(scalars_only, (belief.mean, belief.cov), policy)
    return EfeResult(g=jnp.sum(gs), pragmatic=jnp.sum(prags), epistemic=jnp.sum(epis))


# The rollout's scan carry: the propagated belief moments (mean μ, cov Σ) threaded
# from one step to the next.
_RolloutCarry = tuple[Float64[Array, "n"], Float64[Array, "n n"]]

# One rollout scan step: (carry, action) -> (next carry, that step's trace record).
_RolloutStep = Callable[
    [_RolloutCarry, Float64[Array, "p"]],
    tuple[_RolloutCarry, PolicyEfeTrace],
]


def policy_efe_trace(
    model: LinearGaussianModel,
    belief: Belief,
    policy: Float64[Array, "H p"],
    preference: "Preference",
) -> PolicyEfeTrace:
    """Per-step trace of the H-step rollout — the diagnostic sibling of ``policy_efe``.

    Runs the same ``lax.scan`` as ``policy_efe`` (both drive ``_rollout_body``) but
    keeps the whole per-step record as its ``ys`` instead of summing it away. Because
    the arithmetic is shared, ``jnp.sum`` of each returned scalar column equals the
    matching ``policy_efe`` scalar bit-for-bit. Composes under ``jit`` / ``vmap`` /
    ``grad``.

    Not on the selector hot path: stacking the H×n×n covariances is the memory cost, so
    ``EFESelector`` stays on ``policy_efe`` and this is called only for diagnostics.

    Returns:
        A ``PolicyEfeTrace`` of arrays stacked along a leading H axis.
    """
    if model.control_matrix is None:
        raise ValueError(
            "policy_efe_trace needs a model with a control matrix; an action has no "
            "effect on a control-free (pure-tracking) model."
        )
    control_matrix = model.control_matrix
    goal, precision = preference.goal, preference.precision
    policy = jnp.asarray(policy, dtype=float)
    body = _rollout_body(model, control_matrix, goal, precision)

    _, trace = lax.scan(body, (belief.mean, belief.cov), policy)
    return trace


def _rollout_body(
    model: LinearGaussianModel,
    control_matrix: Float64[Array, "n p"],
    goal: Float64[Array, "m"],
    precision: Float64[Array, "m m"],
) -> _RolloutStep:
    """Build the shared ``lax.scan`` step for the H-step rollout.

    One per-step arithmetic driven by both ``policy_efe`` (which keeps only the three
    scalars of the returned trace) and ``policy_efe_trace`` (which keeps all of it), so
    the two agree by construction rather than by a second implementation. The step
    predicts the belief one step, scores the EFE via ``_efe_step``, then contracts the
    covariance predict-only for the next step, fetching its own ``C`` and reusing the
    ``(Σ⁺, S)`` that ``_efe_step`` returns.
    """

    def step(carry, action):
        mu, sigma = carry
        r = _efe_step(model, mu, sigma, control_matrix, action, goal, precision)
        # predict-only propagation: rollout fetches its OWN C (the one eval the
        # one-step wrapper never pays), reusing the (Σ⁺, S) r already returned.
        c = (
            model.C
            if model.observation_model is None
            else model.observation_model.linearize(r.mu_pred)[0]
        )
        p_xo = r.sigma_pred @ c.T
        sigma_post = r.sigma_pred - p_xo @ jnp.linalg.solve(r.s, p_xo.T)
        sigma_post = 0.5 * (sigma_post + sigma_post.T)
        return (r.mu_pred, sigma_post), PolicyEfeTrace(
            g=r.g,
            pragmatic=r.pragmatic,
            epistemic=r.epistemic,
            mu_pred=r.mu_pred,
            sigma_pred=r.sigma_pred,
            sigma_post=sigma_post,
            s=r.s,
        )

    return step


def policy_efe_ffg(
    backend: "EfeBackend",
    belief: Belief,
    policy: Float64[Array, "H p"],
    preference: "Preference",
    *,
    info_block: Sequence[int],
) -> EfeResult:
    """Summed EFE of a horizon-H ``policy`` over a branching FFG backend.

    The FFG counterpart of ``policy_efe``: instead of ``_efe_step`` on a flat model it
    drives ``_ffg_efe_step`` over the backend's predicted joint each step, aiming the
    epistemic at ``info_block`` — a node's block (via ``backend.block``) or the whole
    state.
    A ``lax.scan`` over the ``policy`` rows sums each step's ``G`` while propagating the
    joint belief predict-only between steps: the mean carries as the coupling-resolved
    ``μ⁺``, the covariance contracts by the Kalman update at ``R(μ⁺)``. At ``H = 1`` it
    reduces exactly to ``_ffg_efe_step``; with no couplings and a whole-state
    ``info_block`` it reproduces ``policy_efe``.

    The ``backend`` is held as a closure constant (it is not a JAX pytree), so this
    composes under ``jit`` / ``vmap`` / ``grad`` over the array arguments — the way
    ``FfgEfeSelector`` already ``vmap``s its per-candidate score.

    Args:
        backend: The FFG inference backend — supplies ``predicted_belief`` (μ⁺, Σ⁺),
            ``observation_model`` (the constant sensor C), ``observation_noise_at``
            (R at μ⁺), and ``block``.
        belief: The starting joint belief ``(μ, Σ)`` over the ``n_total``-D joint state.
        policy: The ``H`` actions to roll out, shape ``(H, p)``.
        preference: The OBSERVATION-space goal — ``goal`` (g) and ``precision`` (Λ).
        info_block: The joint-state indices whose info gain is the per-step epistemic
            value.

    Returns:
        The EFE and its two terms, each summed over the horizon.

    Raises:
        InadmissiblePartitionError: If the backend's carry partition severs an
            EFE-load-bearing edge.
        ValueError: If ``info_block`` is empty, repeats an index, or names an index
            outside the joint state.
    """
    return _policy_efe_ffg(
        backend, belief, policy, preference, _checked_info_block(backend, info_block)
    )


def _policy_efe_ffg(
    backend: "EfeBackend",
    belief: Belief,
    policy: Float64[Array, "H p"],
    preference: "Preference",
    info_block: tuple[int, ...],
) -> EfeResult:
    """``policy_efe_ffg`` on an ``info_block`` already checked, for the searches.

    A search checks its target once at construction and then scores every policy in
    the set, so the per-policy call skips the host-side check.
    """
    goal, precision = preference.goal, preference.precision
    policy = jnp.asarray(policy, dtype=float)
    body = _ffg_rollout_body(backend, goal, precision, info_block)

    # Lean projection: keep only the three scalars in the scan's ys, so a vmap over
    # candidate policies never stacks the H×n×n covariances (RFC-001, the H-step
    # selector).
    def scalars_only(carry, action):
        new_carry, tr = body(carry, action)
        return new_carry, (tr.g, tr.pragmatic, tr.epistemic)

    _, (gs, prags, epis) = lax.scan(scalars_only, belief, policy)
    return EfeResult(g=jnp.sum(gs), pragmatic=jnp.sum(prags), epistemic=jnp.sum(epis))


# One FFG rollout scan step: (joint belief, action) -> (next joint belief, trace).
# The carry is the whole Belief (a registered pytree), unlike the flat rollout's bare
# (mean, cov) tuple, because ``predicted_belief`` consumes and returns a Belief.
_FfgRolloutStep = Callable[
    [Belief, Float64[Array, "p"]],
    tuple[Belief, PolicyEfeTrace],
]


def policy_efe_ffg_trace(
    backend: "EfeBackend",
    belief: Belief,
    policy: Float64[Array, "H p"],
    preference: "Preference",
    *,
    info_block: Sequence[int],
) -> PolicyEfeTrace:
    """Per-step trace of the FFG rollout — the diagnostic sibling of ``policy_efe_ffg``.

    Runs the same ``lax.scan`` as ``policy_efe_ffg`` (both drive ``_ffg_rollout_body``)
    but keeps the whole per-step record as its ``ys`` instead of summing it. Because the
    arithmetic is shared, ``jnp.sum`` of each returned scalar column equals the matching
    ``policy_efe_ffg`` scalar bit-for-bit. The ``epistemic`` column is the
    node-restricted info gain about ``info_block`` — not the whole-state term the flat
    ``policy_efe_trace`` carries — and the moment columns are the coupling-resolved
    ``μ⁺``/``Σ⁺`` and the contracted ``Σ_post``. Composes under ``jit`` / ``vmap`` /
    ``grad`` with the backend held fixed.

    Returns:
        A ``PolicyEfeTrace`` (the same type the flat rollout emits) of arrays stacked
        along a leading H axis.

    Raises:
        InadmissiblePartitionError: If the backend's carry partition severs an
            EFE-load-bearing edge.
        ValueError: If ``info_block`` is empty, repeats an index, or names an index
            outside the joint state.
    """
    goal, precision = preference.goal, preference.precision
    policy = jnp.asarray(policy, dtype=float)
    checked = _checked_info_block(backend, info_block)
    body = _ffg_rollout_body(backend, goal, precision, checked)

    _, trace = lax.scan(body, belief, policy)
    return trace


def _checked_info_block(
    backend: "EfeBackend", info_block: Sequence[int]
) -> tuple[int, ...]:
    """``info_block`` as a tuple, refused unless an FFG EFE can be aimed at it.

    The carry partition must keep every EFE-load-bearing edge whole (ADR-018), or the
    epistemic about the targeted latent degrades while the drift may still look right.
    The block must name distinct joint-state indices. JAX clamps an out-of-range
    gather, so index 99 on a 3-state joint scores as index 2, and an empty or repeated
    block gives a zero or NaN epistemic. None of them raises on its own.
    """
    severed = backend.severed_efe_edges()
    if severed:
        edges = ", ".join(f"{e.parent}->{e.child}" for e in severed)
        raise InadmissiblePartitionError(
            f"the carry partition severs EFE-load-bearing edge(s) [{edges}]; the "
            "epistemic about the targeted latent would degrade while the drift may "
            "still pass. Keep the edge's endpoints in one cluster, or unflag it if "
            "a severed-mass check shows the cut is safe."
        )
    block = tuple(int(i) for i in info_block)
    if not block:
        raise ValueError("info_block is empty; name the joint-state indices to aim at")
    if len(set(block)) != len(block):
        raise ValueError(f"info_block {block} repeats an index")
    out_of_range = [i for i in block if not 0 <= i < backend.n_total]
    if out_of_range:
        raise ValueError(
            f"info_block {block} has indices {out_of_range} outside the "
            f"{backend.n_total}-D joint state (0..{backend.n_total - 1})"
        )
    return block


def _ffg_rollout_body(
    backend: "EfeBackend",
    goal: Float64[Array, "m"],
    precision: Float64[Array, "m m"],
    info_block: Sequence[int],
) -> _FfgRolloutStep:
    """Build the shared ``lax.scan`` step for the FFG H-step rollout.

    One per-step arithmetic driven by both ``policy_efe_ffg`` (which keeps only the
    three scalars) and ``policy_efe_ffg_trace`` (which keeps all of it), so the two
    agree by construction. The step predicts the joint (``backend.predicted_belief`` →
    μ⁺, Σ⁺, the structural couplings folded in), scores the node-targeted EFE
    (``_ffg_efe_step`` at ``R(μ⁺)``), then contracts the covariance predict-only for the
    next step. The sensor ``C`` is constant, so it is fetched once here; ``R(μ⁺)`` is
    re-read per step (it may be state-dependent). The carried mean is ``μ⁺`` — zero
    expected innovation, so the observation moves only the covariance.
    """
    observation_matrix, _ = backend.observation_model  # C (constant, front-loaded)

    def step(belief, action):
        predicted = backend.predicted_belief(belief, action)  # μ⁺, Σ⁺
        observation_noise = backend.observation_noise_at(predicted.mean)  # R(μ⁺)
        scored = _ffg_efe_step(
            predicted.mean,
            predicted.cov,
            observation_matrix,
            observation_noise,
            goal,
            precision,
            info_block,
        )
        # Predict-only contraction for the carry: mean stays μ⁺, cov contracts by the
        # Kalman update at (C, R(μ⁺)). ``_ffg_efe_step`` forms this posterior internally
        # for the epistemic (information form); it is recomputed here in moment form to
        # carry the joint belief forward without widening that step's single-step API.
        s = (
            observation_matrix @ predicted.cov @ observation_matrix.T
            + observation_noise
        )  # S=CΣ⁺Cᵀ+R
        p_xo = predicted.cov @ observation_matrix.T
        sigma_post = predicted.cov - p_xo @ jnp.linalg.solve(s, p_xo.T)
        sigma_post = 0.5 * (sigma_post + sigma_post.T)
        return Belief(mean=predicted.mean, cov=sigma_post), PolicyEfeTrace(
            g=scored.g,
            pragmatic=scored.pragmatic,
            epistemic=scored.epistemic,
            mu_pred=predicted.mean,
            sigma_pred=predicted.cov,
            sigma_post=sigma_post,
            s=s,
        )

    return step


def _logdet_pd(matrix: Float64[Array, "k k"]) -> Float64[Array, ""]:
    """``ln det`` of a covariance, or NaN when it is not positive definite.

    Every determinant the epistemic term takes is of a covariance, so a non-positive
    ``slogdet`` sign means the caller handed over something that is not one — a
    degenerate ``R(x)`` at a reachable state, say. Returning NaN rather than the real
    ``ln|det|`` keeps a meaningless value out of the objective: it propagates to ``G``
    and *loses* the nan-safe argmin at the selection boundary instead of winning it
    with a plausible-looking number.

    The test is a Cholesky factorisation, which succeeds with a strictly positive
    diagonal exactly when the matrix is positive definite. A determinant sign would
    be cheaper but would pass any matrix with an even number of negative eigenvalues
    — ``diag(-1, -2)`` has a positive determinant and is not a covariance.
    """
    chol = jnp.linalg.cholesky(matrix)
    diag = jnp.diagonal(chol)
    definite = jnp.all(diag > 0)
    # The inner guard keeps log() off a non-positive diagonal, so the NaN that comes
    # back is the one this function chose rather than one the logarithm produced.
    safe = jnp.where(definite, diag, 1.0)
    return jnp.where(definite, 2.0 * jnp.sum(jnp.log(safe)), jnp.nan)


def _state_info_gain(
    sigma_pred: Float64[Array, "n n"],
    observation_matrix: Float64[Array, "m n"],
    observation_noise: Float64[Array, "m m"],
    info_block: Sequence[int],
) -> Float64[Array, ""]:
    """Information gain about one block of the state from a single observation.

    The entropy drop of the marginal at ``info_block`` — how much observing sharpens
    belief about *those* state indices (issue #26). Unlike the whole-state
    observation-space shortcut in ``_efe_step`` (``½(ln det S − ln det R)``), this forms
    the posterior covariance explicitly so a sub-block can be read out::

        Σ_post = (Σ⁺⁻¹ + Cᵀ·R⁻¹·C)⁻¹   # prior info + observation info, inverted back
        gain   = ½·(ln det Σ⁺[info_block] − ln det Σ_post[info_block])

    With ``info_block`` the whole state this equals the observation-space epistemic (via
    Sylvester's identity); restricting ``info_block`` to a node's indices gives info
    gain
    about that latent — the factored epistemics ADR-014 finding #3 needs.

    Args:
        sigma_pred: Σ⁺, the predicted joint state covariance (n x n).
        observation_matrix: C, the observation matrix (m x n).
        observation_noise: R, the observation noise covariance (m x m).
        info_block: the state indices whose marginal the info gain is about.

    Returns:
        The scalar information gain (nats), ≥ 0 for nested info_block/observation; NaN
        where ``R`` is not positive definite, matching ``_efe_step``.
    """
    prior_info = jnp.linalg.inv(sigma_pred)  # Σ⁺⁻¹
    obs_info = (
        observation_matrix.T @ jnp.linalg.inv(observation_noise) @ observation_matrix
    )  # Cᵀ R⁻¹ C
    sigma_post = jnp.linalg.inv(prior_info + obs_info)

    idx = jnp.asarray(list(info_block))
    gain = 0.5 * (
        _logdet_pd(sigma_pred[jnp.ix_(idx, idx)])
        - _logdet_pd(sigma_post[jnp.ix_(idx, idx)])
    )
    # A non-positive-definite R still inverts to something finite above, so the two
    # blocks can both come back positive definite and the gain look reasonable. The
    # noise has to be checked on its own for the guard to mean anything.
    return jnp.where(jnp.isnan(_logdet_pd(observation_noise)), jnp.nan, gain)


def _ffg_efe_step(
    mu_plus: Float64[Array, "n"],
    sigma_plus: Float64[Array, "n n"],
    observation_matrix: Float64[Array, "m n"],
    observation_noise: Float64[Array, "m m"],
    goal: Float64[Array, "m"],
    precision: Float64[Array, "m m"],
    info_block: Sequence[int],
) -> EfeResult:
    """One EFE step over the FFG's predicted joint, with a node-targeted epistemic.

    The FFG counterpart of ``_efe_step`` (issue #26). It takes the predicted joint
    moments directly — ``μ⁺``/``Σ⁺`` from ``CouplingGraphBackend.predicted_belief``,
    which already fold in the structural couplings — rather than recomputing
    ``Σ⁺ = A·Σ·Aᵀ + Q`` from a flat model (that route mistakes the couplings for
    observations). The ``pragmatic`` term is identical to ``_efe_step``
    (observation-space cross-entropy); the ``epistemic`` term is the only change — info
    gain about the marginal at ``info_block`` (``_state_info_gain``) instead of the
    whole-state observation-space determinant.

    With no couplings and ``info_block`` the whole state, this reproduces
    ``_efe_step`` /
    ``expected_free_energy`` exactly; a node's block targets the epistemic at that
    latent — the factored analogue of the T-Maze cue (ADR-014 finding #3). Pure
    ``jnp``, so it rides ``jit``/``vmap`` over a grid of candidate actions (which vary
    ``μ⁺``, while ``Σ⁺`` at this one step does not see the action (ADR-003)). What the
    action does reach is ``R`` at ``μ⁺``, and through it the posterior.

    Args:
        mu_plus: μ⁺, the predicted joint mean under the candidate action (n-D).
        sigma_plus: Σ⁺, the predicted joint covariance (n x n) — couplings included.
        observation_matrix: C, the real observation matrix over the joint state (m x n).
        observation_noise: R, the observation noise covariance (m x m).
        goal: g, the preferred observation (m-D).
        precision: Λ, how sharply the goal is preferred (m x m).
        info_block: the state indices whose info gain is the epistemic value (a node's
            block, or the whole state).

    Returns:
        The scalar EFE ``G = pragmatic − epistemic`` (minimised) and its two terms.
    """
    o_pred = observation_matrix @ mu_plus  # o⁺ = C·μ⁺
    s = (
        observation_matrix @ sigma_plus @ observation_matrix.T + observation_noise
    )  # S = C·Σ⁺·Cᵀ + R

    residual = o_pred - goal
    pragmatic = 0.5 * residual @ precision @ residual + 0.5 * jnp.trace(precision @ s)

    epistemic = _state_info_gain(
        sigma_plus, observation_matrix, observation_noise, info_block
    )

    return EfeResult(g=pragmatic - epistemic, pragmatic=pragmatic, epistemic=epistemic)


def _efe_step(
    model: LinearGaussianModel,
    mu: Float64[Array, "n"],
    sigma: Float64[Array, "n n"],
    control_matrix: Float64[Array, "n p"],
    action: Float64[Array, "p"],
    goal: Float64[Array, "m"],
    precision: Float64[Array, "m m"],
) -> _EfeStep:
    # --- predict: push the belief one step through the dynamics under `action` ---
    # Mirrors the covariance predict in kalman._gain_and_posterior_cov (cov_pred);
    # NB within THIS step the action moves only the mean, so Σ⁺ here does not see it —
    # the whole reason the epistemic term collapses under a fixed sensor (ADR-003). The
    # independence is local: a state-dependent R(x) or Q(x) makes the noise, and so the
    # posterior this step hands the next one, depend on where the action put the mean.
    # Over a horizon the covariance does move with the policy; only the one-step
    # predicted covariance is blind to it.
    mu_pred = model.A @ mu + control_matrix @ action
    process_q = (
        model.Q
        if model.dynamics_noise_model is None
        else model.dynamics_noise_model.noise_at(mu_pred)
    )

    sigma_pred = model.A @ sigma @ model.A.T + process_q  # Σ⁺ = AΣAᵀ + process_q

    # --- sense: predicted-observation moments (o⁺, S) + conditional noise R at μ⁺ ---
    # The sensor owns its moment-matching (D1): the kernel never reconstructs o⁺/S.
    # FRAGILE(lit) #4: everything is evaluated at μ⁺. Immaterial for a fixed sensor,
    # which is its own linearization everywhere; for a state-dependent R(x) this point
    # is what the action reaches, and so what the epistemic term ends up measuring.
    if model.observation_model is None:
        # FAST PATH — a bare matvec/matmul, byte-identical to Phase 1A. Kept inline
        # (no method dispatch) so the fixed-sensor hot path stays lean.
        observation_matrix, observation_noise = model.C, model.R
        o_pred = observation_matrix @ mu_pred
        pred_obs_cov = (
            observation_matrix @ sigma_pred @ observation_matrix.T + observation_noise
        )
    else:
        # Linear sensors return exact (C·μ⁺, C·Σ⁺·Cᵀ+R, R); NonlinearSensor (2.5)
        # returns its 2nd-order moments. S feeds the pragmatic term, R the epistemic.
        o_pred, pred_obs_cov, observation_noise = model.observation_model.gaussianize(
            mu_pred, sigma_pred
        )

    # --- pragmatic: expected negative log-preference (cross-entropy form) ---
    # FRAGILE(lit) #1: `preference` is read in OBSERVATION space (g over o, Λ over o).
    # FRAGILE(lit) #2: cross-entropy form = mean term + ½tr(ΛS). The ½tr(ΛS) piece is
    # the variance penalty that distinguishes this from the mean-only form; against the
    # KL-risk form it differs only by −½ln det S, which the epistemic term restores.
    residual = o_pred - goal
    pragmatic_mean = 0.5 * residual @ precision @ residual
    pragmatic_var = 0.5 * jnp.trace(precision @ pred_obs_cov)
    pragmatic = pragmatic_mean + pragmatic_var

    # --- epistemic: state information gain I(state; obs) = ½ ln(det S / det R) ---
    # FRAGILE(lit) #3: this is *salience* (state info gain), not *novelty* (parameter
    # info gain). `_logdet_pd` carries the positive-definiteness guard, so a degenerate
    # R(x) at a reachable state yields NaN rather than a plausible-but-wrong finite
    # value; `_state_info_gain` guards the same way, so both epistemic routes agree.
    epistemic = 0.5 * (_logdet_pd(pred_obs_cov) - _logdet_pd(observation_noise))

    # FRAGILE(lit) #5: G = pragmatic − epistemic (minimise). Pairing cross-entropy
    # with −info-gain is decomposition (b); it is self-consistent (no double-count).
    g = pragmatic - epistemic
    return _EfeStep(
        g=g,
        pragmatic=pragmatic,
        epistemic=epistemic,
        mu_pred=mu_pred,
        sigma_pred=sigma_pred,
        s=pred_obs_cov,
    )
