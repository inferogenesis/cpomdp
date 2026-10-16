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
and an OBSERVATION-space preference ``(o*, Λ)`` (preferred observation ``o*``,
precision ``Λ``):

    predict:    μ⁺ = A·μ + B·a            Σ⁺ = A·Σ·Aᵀ + Q
    sense:      (C, R) = observation_model.linearize(μ⁺)
                o⁺ = C·μ⁺                  S = C·Σ⁺·Cᵀ + R  # predicted observation cov
    pragmatic:  ½·(o⁺ − o*)ᵀ·Λ·(o⁺ − o*)  +  ½·tr(Λ·S)
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
   NOT match the state-space ``target`` the LQR path currently uses. Reconciling the
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
    IN ── model=(A,control,Q,sensor)  belief=(μ,Σ)  action=a  preference=(o*,Λ)
      │
      ▼    GUARD      control is None?  ──►  raise ValueError
      │
      ▼    PREDICT    μ⁺ = A·μ + control·a     (action enters HERE only)
      │               Σ⁺ = A·Σ·Aᵀ + Q          (this step's Σ⁺ does not see the action)
      │
      ▼    SENSE      (C, R) = linearize(μ⁺)   or the model's fixed (C, R)
      │               o⁺ = C·μ⁺
      │               S  = C·Σ⁺·Cᵀ + R         (predicted observation cov, ONCE)
      │
      ├──► PRAGMATIC  ½·(o⁺−o*)ᵀ·Λ·(o⁺−o*) + ½·tr(Λ·S)  (cost,  lower better)
      ├──► EPISTEMIC  ½·(ln det S − ln det R)            (value, higher better)
      │
      ▼    G = pragmatic − epistemic
      │
      ▼    OUT        return EFEResult(g=G, pragmatic=…, epistemic=…)

"""

import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, SupportsIndex

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array, Float64

from cpomdp._validation import refuse_graph_interface
from cpomdp.backends.base import InadmissiblePartitionError
from cpomdp.types import Belief, LinearGaussianModel

if TYPE_CHECKING:
    # Type-only imports. The kernel reads the backend's predict/observation methods by
    # duck-typing, so it does not depend on a backend at runtime. Its one runtime import
    # from the backends package is the exception the graph rollouts raise, and
    # `backends.base` imports nothing that imports this module. selection imports this
    # module, so `_target_and_precision` reads `Preference` at call time.
    from cpomdp.backends.base import GraphBackend
    from cpomdp.selection import Preference

__all__ = [
    "EFEResult",
    "PolicyEFETrace",
    "expected_free_energy",
    "policy_efe",
    "policy_efe_ffg",
    "policy_efe_ffg_trace",
    "policy_efe_trace",
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EFEResult:
    """Expected free energy and the two terms it is made of.

    [`expected_free_energy`][cpomdp.expected_free_energy],
    [`policy_efe`][cpomdp.efe.policy_efe] and
    [`policy_efe_ffg`][cpomdp.efe.policy_efe_ffg] return one.
    [`PolicyEFETrace`][cpomdp.efe.PolicyEFETrace] has the same three fields, one entry
    per step. Fields are read by name. A record does not index or unpack, as the search
    results do not.

    A registered JAX pytree, so a ``vmap`` over candidate actions returns one record
    with every field batched. Rank the batch by its ``g``, as ``jnp.argmin(batch.g)``.

    Differentiate ``g`` too. Three fields are not the ``(value, aux)`` pair that
    ``jax.value_and_grad(..., has_aux=True)`` expects, so a function differentiated
    that way returns ``(result.g, result)``.

    Attributes:
        g: The expected free energy ``G = pragmatic − epistemic``. Lower is preferred.
        pragmatic: The goal term, a cost.
        epistemic: The information the observation carries about the state, or
            about the targeted node where one is set, a value.
    """

    g: Float64[Array, ""]
    pragmatic: Float64[Array, ""]
    epistemic: Float64[Array, ""]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PolicyEFETrace:
    """One H-step rollout's per-step record, each field stacked along a leading H axis.

    The first three fields are the per-step [`EFEResult`][cpomdp.efe.EFEResult], and
    each sums to the matching field of the rollout's result. The rest are the moments
    the rollout carries between steps, kept for inspection: the covariance a policy
    carries, and the matrices whose conditioning the rollout depends on.

    A registered JAX pytree. ``lax.scan`` emits it as its ``ys``, and it survives
    ``jit``, ``vmap`` and ``grad``, gaining any outer ``vmap`` axis too.

    *Predicted* means after the dynamics and the action and before any observation.
    *Posterior* means after the predict-only Kalman update the rollout carries to the
    next step.

    Attributes:
        g: Per-step EFE ``G = pragmatic − epistemic``, shape ``(H,)``.
        pragmatic: Per-step goal term, shape ``(H,)``.
        epistemic: Per-step information gain about the state, shape ``(H,)``.
        predicted_mean: The predicted state mean, μ⁺ = ``A·μ + B·a``, shape ``(H, n)``.
        predicted_cov: The predicted state covariance, Σ⁺ = ``A·Σ·Aᵀ + Q``, shape
            ``(H, n, n)``.
        posterior_cov: The state covariance after the predicted observation, Σ_post,
            shape ``(H, n, n)``.
        predicted_observation_cov: The predicted observation covariance,
            S = ``C·Σ⁺·Cᵀ + R``, shape ``(H, m, m)``.
    """

    g: Float64[Array, "H"]
    pragmatic: Float64[Array, "H"]
    epistemic: Float64[Array, "H"]
    predicted_mean: Float64[Array, "H n"]  # μ⁺
    predicted_cov: Float64[Array, "H n n"]  # Σ⁺
    posterior_cov: Float64[Array, "H n n"]  # Σ_post
    predicted_observation_cov: Float64[Array, "H m m"]  # S


@dataclass(frozen=True)
class _EFEStep:
    """One EFE step: the public split plus the moments the rollout propagates.

    ``expected_free_energy`` exposes only the first three fields. The H-step rollout
    (``policy_efe``) also needs the predicted moments to propagate the belief, named as
    ``PolicyEFETrace`` names them. It fetches its own ``C``. Consumed locally, it never
    crosses a ``jit``/``vmap``/``scan`` boundary, so it needs no pytree registration.
    """

    g: Float64[Array, ""]
    pragmatic: Float64[Array, ""]
    epistemic: Float64[Array, ""]
    predicted_mean: Float64[Array, "n"]  # μ⁺
    predicted_cov: Float64[Array, "n n"]  # Σ⁺
    predicted_observation_cov: Float64[Array, "m m"]  # S


def expected_free_energy(
    model: LinearGaussianModel,
    belief: Belief,
    action: Float64[Array, "p"],
    preference: "Preference",
) -> EFEResult:
    """Expected Free Energy of taking ``action`` from ``belief``, and its split.

    The locked linear-Gaussian definition (ADR-005), with the sensor read at the
    predicted mean::

        predict:    μ⁺ = A·μ + B·a              Σ⁺ = A·Σ·Aᵀ + Q
        sense:      (C, R) at μ⁺,  o⁺ = C·μ⁺,   S = C·Σ⁺·Cᵀ + R
        pragmatic:  ½·(o⁺ − o*)ᵀ·Λ·(o⁺ − o*) + ½·tr(Λ·S)
        epistemic:  ½·(ln det S − ln det R)
        G = pragmatic − epistemic

    μ and Σ are the belief's mean and covariance, and a is the action. A, B and Q are
    the dynamics, control and dynamics-noise matrices. C and R are the sensor's
    observation matrix and noise. μ⁺ and Σ⁺ are the predicted state moments, o⁺ the
    predicted reading and S its covariance. o* is the preferred reading and Λ its
    precision.

    Pure ``jnp``, so it composes under ``jit``, ``vmap`` and ``grad``, in particular
    over a batch of candidate ``action`` vectors with ``model``, ``belief`` and
    ``preference`` held fixed, which is how [`EFESelector`][cpomdp.EFESelector]
    searches. Take ``jax.grad`` of a function that returns ``result.g``, and rank a
    batch by its ``g`` too. The Expected free energy page of the API reference opens
    the same account in plain terms.

    Args:
        model: The generative model. Must have a control matrix (an action has no
            meaning without one). Its ``observation_model`` supplies the local
            ``(C, R)``. ``None`` means the fixed sensor
            ``(observation_matrix, observation_noise)``.
        belief: The current belief, mean μ and covariance Σ.
        action: The candidate action a, shape ``(p,)``.
        preference: The goal as an observation-space
            [`Preference`][cpomdp.Preference]. Its ``target`` is the preferred reading
            o*, shape ``(m,)``, and its ``precision`` is Λ, shape ``(m, m)``. A
            reading coincides with a state only when ``C = I``. An
            [`ObservationGoal`][cpomdp.ObservationGoal] hands over its own as
            ``goal.preference``.

    Returns:
        An [`EFEResult`][cpomdp.efe.EFEResult] holding the scalar EFE G in its ``g``
        field and its two non-negative terms, ``pragmatic`` and ``epistemic``. Lower G
        is preferred.

    Raises:
        TypeError: If ``model`` is a coupled graph backend's model, which leaves
            out the couplings.
        ValueError: If the model has no control matrix.
        TypeError: If ``preference`` is not a [`Preference`][cpomdp.Preference].
    """
    refuse_graph_interface(model, "expected_free_energy")
    if model.control_matrix is None:
        raise ValueError(
            "expected_free_energy needs a model with a control matrix; an action "
            "has no effect on a control-free (pure-tracking) model."
        )
    control_matrix = model.control_matrix  # narrowed to Array by the guard above
    target, precision = _target_and_precision(preference)
    action = jnp.asarray(action, dtype=float)
    step = _efe_step(
        model,
        belief.mean,
        belief.cov,
        control_matrix,
        action,
        target,
        precision,
    )
    return EFEResult(g=step.g, pragmatic=step.pragmatic, epistemic=step.epistemic)


def policy_efe(
    model: LinearGaussianModel,
    belief: Belief,
    policy: Float64[Array, "H p"],
    preference: "Preference",
) -> EFEResult:
    """Summed EFE of a horizon-H ``policy`` from ``belief``.

    [`EFESelector`][cpomdp.EFESelector] above horizon 1 and the flat exhaustive
    searches in ``cpomdp.enumeration`` score with it. A ``lax.scan`` over the
    ``policy`` rows sums each step's G while propagating the belief predict-only
    between steps. The mean follows the prediction. The covariance contracts by the
    Kalman update at the moments that step's score already predicted. ``H`` is
    ``policy.shape[0]``, and at ``H = 1`` it reduces exactly to
    [`expected_free_energy`][cpomdp.expected_free_energy]. Composes under ``jit``,
    ``vmap`` and ``grad``. The Expected free energy page of the API reference opens
    the rollout in plain terms.

    Args:
        model: The generative model. Must have a control matrix.
        belief: The belief the rollout starts from.
        policy: The ``H`` actions to roll out, shape ``(H, p)``.
        preference: The observation-space [`Preference`][cpomdp.Preference], as for
            ``expected_free_energy``.

    Returns:
        An [`EFEResult`][cpomdp.efe.EFEResult] with each field summed over the
        horizon.

    Raises:
        TypeError: If ``model`` is a coupled graph backend's model, which leaves
            out the couplings.
        ValueError: If the model has no control matrix.
        TypeError: If ``preference`` is not a [`Preference`][cpomdp.Preference].
    """
    refuse_graph_interface(model, "policy_efe")
    if model.control_matrix is None:
        raise ValueError(
            "policy_efe needs a model with a control matrix; an action has no "
            "effect on a control-free (pure-tracking) model."
        )
    control_matrix = model.control_matrix
    target, precision = _target_and_precision(preference)
    policy = jnp.asarray(policy, dtype=float)
    body = _rollout_body(model, control_matrix, target, precision)

    # Lean projection: keep only the three scalars in the scan's ys, so EFESelector's
    # vmap over |A|^H candidate policies never stacks the H×n×n covariances.
    def scalars_only(carry, action):
        new_carry, tr = body(carry, action)
        return new_carry, (tr.g, tr.pragmatic, tr.epistemic)

    _, (gs, prags, epis) = lax.scan(scalars_only, (belief.mean, belief.cov), policy)
    return EFEResult(g=jnp.sum(gs), pragmatic=jnp.sum(prags), epistemic=jnp.sum(epis))


# The rollout's scan carry: the propagated belief moments (mean μ, cov Σ) threaded
# from one step to the next.
_RolloutCarry = tuple[Float64[Array, "n"], Float64[Array, "n n"]]

# One rollout scan step: (carry, action) -> (next carry, that step's trace record).
_RolloutStep = Callable[
    [_RolloutCarry, Float64[Array, "p"]],
    tuple[_RolloutCarry, PolicyEFETrace],
]


def policy_efe_trace(
    model: LinearGaussianModel,
    belief: Belief,
    policy: Float64[Array, "H p"],
    preference: "Preference",
) -> PolicyEFETrace:
    """Per-step trace of the H-step rollout, the diagnostic sibling of ``policy_efe``.

    Runs the same scan as [`policy_efe`][cpomdp.efe.policy_efe] and keeps the whole
    per-step record instead of summing it away. The arithmetic is shared, so
    ``jnp.sum`` of each returned scalar column equals the matching ``policy_efe``
    field bit-for-bit. Composes under ``jit``, ``vmap`` and ``grad``.

    Not on the selector hot path. Stacking the H×n×n covariances is the memory cost,
    so [`EFESelector`][cpomdp.EFESelector] stays on ``policy_efe`` and this is called
    only for diagnostics.

    Args:
        model: The generative model. Must have a control matrix.
        belief: The belief the rollout starts from.
        policy: The ``H`` actions to roll out, shape ``(H, p)``.
        preference: The observation-space [`Preference`][cpomdp.Preference], as for
            ``expected_free_energy``.

    Returns:
        A [`PolicyEFETrace`][cpomdp.efe.PolicyEFETrace] of arrays stacked along a
        leading H axis.

    Raises:
        TypeError: If ``model`` is a coupled graph backend's model, which leaves
            out the couplings.
        ValueError: If the model has no control matrix.
        TypeError: If ``preference`` is not a [`Preference`][cpomdp.Preference].
    """
    refuse_graph_interface(model, "policy_efe_trace")
    if model.control_matrix is None:
        raise ValueError(
            "policy_efe_trace needs a model with a control matrix; an action has no "
            "effect on a control-free (pure-tracking) model."
        )
    control_matrix = model.control_matrix
    target, precision = _target_and_precision(preference)
    policy = jnp.asarray(policy, dtype=float)
    body = _rollout_body(model, control_matrix, target, precision)

    _, trace = lax.scan(body, (belief.mean, belief.cov), policy)
    return trace


def _rollout_body(
    model: LinearGaussianModel,
    control_matrix: Float64[Array, "n p"],
    target: Float64[Array, "m"],
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
        r = _efe_step(model, mu, sigma, control_matrix, action, target, precision)
        # predict-only propagation: rollout fetches its OWN C (the one eval the
        # one-step wrapper never pays), reusing the (Σ⁺, S) r already returned.
        c = (
            model.observation_matrix
            if model.observation_model is None
            else model.observation_model.linearize(r.predicted_mean)[0]
        )
        p_xo = r.predicted_cov @ c.T
        sigma_post = r.predicted_cov - p_xo @ jnp.linalg.solve(
            r.predicted_observation_cov, p_xo.T
        )
        sigma_post = 0.5 * (sigma_post + sigma_post.T)
        return (r.predicted_mean, sigma_post), PolicyEFETrace(
            g=r.g,
            pragmatic=r.pragmatic,
            epistemic=r.epistemic,
            predicted_mean=r.predicted_mean,
            predicted_cov=r.predicted_cov,
            posterior_cov=sigma_post,
            predicted_observation_cov=r.predicted_observation_cov,
        )

    return step


def policy_efe_ffg(
    backend: "GraphBackend",
    belief: Belief,
    policy: Float64[Array, "H p"],
    preference: "Preference",
    *,
    info_node: SupportsIndex | Sequence[SupportsIndex] | None = None,
) -> EFEResult:
    """Summed EFE of a horizon-H ``policy`` over a branching FFG backend.

    The graph counterpart of [`policy_efe`][cpomdp.efe.policy_efe]. Each step scores
    the backend's predicted joint, with the epistemic term aimed at ``info_node``'s
    state or the whole joint state. A ``lax.scan`` over
    the ``policy`` rows sums each step's G while propagating the joint belief
    predict-only between steps. The mean carries as the coupling-resolved predicted
    mean. The covariance contracts by the Kalman update at the sensor noise read
    there. At ``H = 1`` it is one step of the graph score. With no couplings and a
    whole-state target it reproduces ``policy_efe``.

    The ``backend`` is held as a closure constant, since it is not a JAX pytree. This
    composes under ``jit``, ``vmap`` and ``grad`` over the array arguments, the way
    [`FFGEFESelector`][cpomdp.FFGEFESelector] ``vmap``s its per-candidate score.

    Args:
        backend: The FFG inference backend. It supplies ``predicted_belief``
            (μ⁺, Σ⁺), ``model.observation_matrix`` (the constant sensor C),
            ``model.observation_noise_at`` (R at μ⁺) and ``block``.
        belief: The starting joint belief ``(μ, Σ)`` over the backend's joint state.
        policy: The ``H`` actions to roll out, shape ``(H, p)``.
        preference: The observation-space [`Preference`][cpomdp.Preference], as for
            ``expected_free_energy``. An
            [`ObservationGoal`][cpomdp.ObservationGoal] hands over its own as
            ``goal.preference``.
        info_node: The node whose state the per-step epistemic value is about.
            ``None``, the default, aims it at the whole joint state.

    Returns:
        An [`EFEResult`][cpomdp.efe.EFEResult] with each field summed over the
        horizon.

    Raises:
        InadmissiblePartitionError: If the backend's carry partition severs an
            EFE-load-bearing edge.
        ValueError: If ``info_node`` is not a node of the graph.
        TypeError: If ``preference`` is not a [`Preference`][cpomdp.Preference].
    """
    return _policy_efe_ffg(
        backend, belief, policy, preference, _info_block_of(backend, info_node)
    )


def _policy_efe_ffg(
    backend: "GraphBackend",
    belief: Belief,
    policy: Float64[Array, "H p"],
    preference: "Preference",
    info_block: tuple[int, ...],
) -> EFEResult:
    """``policy_efe_ffg`` on an ``info_block`` already checked, for the searches.

    A search checks its target once at construction and then scores every policy in
    the set, so the per-policy call skips the host-side check.
    """
    target, precision = _target_and_precision(preference)
    policy = jnp.asarray(policy, dtype=float)
    body = _ffg_rollout_body(backend, target, precision, info_block)

    # Lean projection: keep only the three scalars in the scan's ys, so a vmap over
    # candidate policies never stacks the H×n×n covariances (RFC-001, the H-step
    # selector).
    def scalars_only(carry, action):
        new_carry, tr = body(carry, action)
        return new_carry, (tr.g, tr.pragmatic, tr.epistemic)

    _, (gs, prags, epis) = lax.scan(scalars_only, belief, policy)
    return EFEResult(g=jnp.sum(gs), pragmatic=jnp.sum(prags), epistemic=jnp.sum(epis))


# One FFG rollout scan step: (joint belief, action) -> (next joint belief, trace).
# The carry is the whole Belief (a registered pytree), unlike the flat rollout's bare
# (mean, cov) tuple, because ``predicted_belief`` consumes and returns a Belief.
_FFGRolloutStep = Callable[
    [Belief, Float64[Array, "p"]],
    tuple[Belief, PolicyEFETrace],
]


def policy_efe_ffg_trace(
    backend: "GraphBackend",
    belief: Belief,
    policy: Float64[Array, "H p"],
    preference: "Preference",
    *,
    info_node: SupportsIndex | Sequence[SupportsIndex] | None = None,
) -> PolicyEFETrace:
    """Per-step trace of the FFG rollout, the diagnostic sibling of ``policy_efe_ffg``.

    Runs the same scan as [`policy_efe_ffg`][cpomdp.efe.policy_efe_ffg] and keeps the
    whole per-step record instead of summing it. The arithmetic is shared, so
    ``jnp.sum`` of each returned scalar column equals the matching ``policy_efe_ffg``
    field bit-for-bit. The ``epistemic`` column is the info gain about the
    ``info_node`` targeted. The flat
    [`policy_efe_trace`][cpomdp.efe.policy_efe_trace] carries the whole-state term
    there instead. The moment columns are the
    coupling-resolved predicted moments and the contracted posterior covariance.
    Composes under ``jit``, ``vmap`` and ``grad`` with the backend held fixed.

    Args:
        backend: The FFG inference backend, as for ``policy_efe_ffg``.
        belief: The starting joint belief over the backend's joint state.
        policy: The ``H`` actions to roll out, shape ``(H, p)``.
        preference: The observation-space [`Preference`][cpomdp.Preference], as for
            ``expected_free_energy``.
        info_node: The node whose state the per-step epistemic value is about.
            ``None``, the default, aims it at the whole joint state.

    Returns:
        A [`PolicyEFETrace`][cpomdp.efe.PolicyEFETrace], the type the flat rollout
        emits, of arrays stacked along a leading H axis.

    Raises:
        InadmissiblePartitionError: If the backend's carry partition severs an
            EFE-load-bearing edge.
        ValueError: If ``info_node`` is not a node of the graph.
        TypeError: If ``preference`` is not a [`Preference`][cpomdp.Preference].
    """
    target, precision = _target_and_precision(preference)
    policy = jnp.asarray(policy, dtype=float)
    info_block = _info_block_of(backend, info_node)
    body = _ffg_rollout_body(backend, target, precision, info_block)

    _, trace = lax.scan(body, belief, policy)
    return trace


def _target_and_precision(
    preference: "Preference",
) -> tuple[Float64[Array, "m"], Float64[Array, "m m"]]:
    """The two fields a score reads off ``preference``, or a refusal naming the fix.

    Anything but a ``Preference`` is refused, and a field missing from one surfaces as
    its own error. Under ``jit`` and ``vmap`` the check runs once, at trace time.
    """
    # selection imports this module, so the type is read at call time.
    from cpomdp.selection import Preference

    if not isinstance(preference, Preference):
        raise TypeError(_not_a_preference(preference))
    return preference.target, preference.precision


_SCORED_AGAINST = "Expected free energy is scored against a Preference."


def _not_a_preference(given: object) -> str:
    """The refusal for ``given``, naming the ``Preference`` to pass in its place.

    The selectors and searches reach the same check, so it names no caller.
    """
    from cpomdp.selection import ObservationGoal, StateGoal

    if isinstance(given, ObservationGoal):
        return (
            f"{_SCORED_AGAINST} This is an ObservationGoal. Pass its .preference, "
            "which keeps the target and precision and leaves the search settings "
            "behind."
        )
    if isinstance(given, StateGoal):
        return (
            "Expected free energy is scored against a Preference over readings. This "
            "is a StateGoal, whose target is a state. Pass a Preference over the "
            "reading you want, such as Preference(C @ target) for a fixed sensor with "
            "observation matrix C."
        )
    if isinstance(given, (list, tuple, int, float)) or hasattr(given, "__array__"):
        return (
            f"{_SCORED_AGAINST} This is a bare array or sequence. Wrap the preferred "
            "reading as Preference(reading, precision)."
        )
    return f"{_SCORED_AGAINST} This is a value of type {type(given).__name__}."


def _info_node_index(node: object) -> int:
    """One node index as an int, a NumPy integer included. A bool is refused."""
    if isinstance(node, bool | np.bool_):
        raise TypeError(f"info_node takes node indices, got the bool {node}")
    try:
        return operator.index(node)  # ty: ignore[invalid-argument-type]
    except TypeError:
        raise TypeError(
            f"info_node takes node indices, got {type(node).__name__}"
        ) from None


def _info_nodes(
    info_node: SupportsIndex | Sequence[SupportsIndex],
) -> tuple[int, ...]:
    """``info_node`` as a tuple of node indices: one node, or each of several."""
    if isinstance(info_node, Sequence) and not isinstance(info_node, str):
        return tuple(_info_node_index(node) for node in info_node)
    return (_info_node_index(info_node),)


def _info_block_of(
    backend: "GraphBackend", info_node: SupportsIndex | Sequence[SupportsIndex] | None
) -> tuple[int, ...]:
    """The joint-state indices ``info_node`` names, refused unless an EFE can aim there.

    ``None`` is the whole joint state. A node is its block, ``backend.block``, and a
    sequence of nodes is their blocks together, the joint marginal of those nodes. The
    carry partition must keep every EFE-load-bearing edge whole (ADR-018), or the
    epistemic about the targeted latent degrades while the drift may still look right.
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
    if info_node is None:
        return tuple(range(backend.model.n_states))
    nodes = _info_nodes(info_node)
    if not nodes:
        raise ValueError("info_node names no node. Pass None for the whole state.")
    if len(set(nodes)) != len(nodes):
        raise ValueError(f"info_node {list(nodes)} repeats a node")
    try:
        return tuple(index for node in nodes for index in backend.block(node))
    except ValueError as error:
        raise ValueError(f"info_node {info_node}: {error}") from error


def _ffg_rollout_body(
    backend: "GraphBackend",
    target: Float64[Array, "m"],
    precision: Float64[Array, "m m"],
    info_block: Sequence[int],
) -> _FFGRolloutStep:
    """Build the shared ``lax.scan`` step for the FFG H-step rollout.

    One per-step arithmetic driven by both ``policy_efe_ffg`` (which keeps only the
    three scalars) and ``policy_efe_ffg_trace`` (which keeps all of it), so the two
    agree by construction. The step predicts the joint (``backend.predicted_belief`` →
    μ⁺, Σ⁺, the structural couplings folded in), scores the node-targeted EFE
    (``_ffg_efe_step`` at ``R(μ⁺)``), then contracts the covariance predict-only for the
    next step. The sensor ``C`` is constant, so it is fetched once here; ``R(μ⁺)`` is
    re-read per step (it may be state-dependent). The carried mean is ``μ⁺``. The
    reading is expected to equal its prediction, so it moves only the covariance.
    """
    model = backend.model
    observation_matrix = model.observation_matrix  # C (constant, front-loaded)

    def step(belief, action):
        predicted = backend.predicted_belief(belief, action)  # μ⁺, Σ⁺
        observation_noise = model.observation_noise_at(predicted.mean)  # R(μ⁺)
        scored = _ffg_efe_step(
            predicted.mean,
            predicted.cov,
            observation_matrix,
            observation_noise,
            target,
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
        return Belief(mean=predicted.mean, cov=sigma_post), PolicyEFETrace(
            g=scored.g,
            pragmatic=scored.pragmatic,
            epistemic=scored.epistemic,
            predicted_mean=predicted.mean,
            predicted_cov=predicted.cov,
            posterior_cov=sigma_post,
            predicted_observation_cov=s,
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
    target: Float64[Array, "m"],
    precision: Float64[Array, "m m"],
    info_block: Sequence[int],
) -> EFEResult:
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
        target: o*, the preferred observation (m-D).
        precision: Λ, how sharply the goal is preferred (m x m).
        info_block: the joint-state indices whose info gain is the epistemic value,
            resolved from ``info_node`` by the public functions.

    Returns:
        The scalar EFE ``G = pragmatic − epistemic`` (minimised) and its two terms.
    """
    o_pred = observation_matrix @ mu_plus  # o⁺ = C·μ⁺
    s = (
        observation_matrix @ sigma_plus @ observation_matrix.T + observation_noise
    )  # S = C·Σ⁺·Cᵀ + R

    residual = o_pred - target
    pragmatic = 0.5 * residual @ precision @ residual + 0.5 * jnp.trace(precision @ s)

    epistemic = _state_info_gain(
        sigma_plus, observation_matrix, observation_noise, info_block
    )

    return EFEResult(g=pragmatic - epistemic, pragmatic=pragmatic, epistemic=epistemic)


def _efe_step(
    model: LinearGaussianModel,
    mu: Float64[Array, "n"],
    sigma: Float64[Array, "n n"],
    control_matrix: Float64[Array, "n p"],
    action: Float64[Array, "p"],
    target: Float64[Array, "m"],
    precision: Float64[Array, "m m"],
) -> _EFEStep:
    # --- predict: push the belief one step through the dynamics under `action` ---
    # Mirrors the covariance predict in kalman._gain_and_posterior_cov (cov_pred);
    # NB within THIS step the action moves only the mean, so Σ⁺ here does not see it —
    # the whole reason the epistemic term collapses under a fixed sensor (ADR-003). The
    # independence is local: a state-dependent R(x) or Q(x) makes the noise, and so the
    # posterior this step hands the next one, depend on where the action put the mean.
    # Over a horizon the covariance does move with the policy; only the one-step
    # predicted covariance is blind to it.
    mu_pred = model.dynamics_matrix @ mu + control_matrix @ action
    dynamics_noise = (
        model.dynamics_noise
        if model.dynamics_noise_model is None
        else model.dynamics_noise_model.noise_at(mu_pred)
    )

    sigma_pred = (
        model.dynamics_matrix @ sigma @ model.dynamics_matrix.T + dynamics_noise
    )  # Σ⁺ = AΣAᵀ + dynamics_noise

    # --- sense: predicted-observation moments (o⁺, S) + conditional noise R at μ⁺ ---
    # The sensor owns its moment-matching (D1): the kernel never reconstructs o⁺/S.
    # FRAGILE(lit) #4: everything is evaluated at μ⁺. Immaterial for a fixed sensor,
    # which is its own linearization everywhere; for a state-dependent R(x) this point
    # is what the action reaches, and so what the epistemic term ends up measuring.
    if model.observation_model is None:
        # FAST PATH — a bare matvec/matmul. Kept inline (no method dispatch) so the
        # fixed-sensor hot path stays lean.
        observation_matrix, observation_noise = (
            model.observation_matrix,
            model.observation_noise,
        )
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
    # FRAGILE(lit) #1: `preference` is read in OBSERVATION space (o* and Λ over o).
    # FRAGILE(lit) #2: cross-entropy form = mean term + ½tr(ΛS). The ½tr(ΛS) piece is
    # the variance penalty that distinguishes this from the mean-only form; against the
    # KL-risk form it differs only by −½ln det S, which the epistemic term restores.
    residual = o_pred - target
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
    return _EFEStep(
        g=g,
        pragmatic=pragmatic,
        epistemic=epistemic,
        predicted_mean=mu_pred,
        predicted_cov=sigma_pred,
        predicted_observation_cov=pred_obs_cov,
    )
