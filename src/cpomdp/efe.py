"""One-step Expected Free Energy (EFE) for the linear-Gaussian regime.

This module computes ``G(a)`` — the Expected Free Energy of taking action ``a``
from the current belief — and its decomposition into a *pragmatic* (goal-seeking)
and an *epistemic* (information-seeking) part. Minimising ``G`` over actions is
how a v0.3 agent will choose what to do.

================================================================================
THE DECISION THIS FILE ENCODES  (see DECISIONS.md ADR-005, and rfcs/004)
================================================================================
There is *no single agreed formula* for EFE in the active-inference literature —
the pragmatic term in particular has at least three forms in circulation, and
sources disagree on signs and on whether risk is a cross-entropy or a KL. I have
**chosen one route deliberately**, and it is, frankly, somewhat speculative: it
sits in an area that is well-trodden but inconsistently written down, and well
outside my core expertise. We are committing to it *and* committing to
proving (or honestly bounding) that it is the right one — that proof is the job of
rfcs/004 and the validation tests it will spawn. Until then, treat the choices
flagged ``# FRAGILE(lit):`` below as load-bearing assumptions that may move as the
literature is pinned down. Over-commented on purpose: this file is the shared
reference for re-understanding the reasoning next session. Will trim it once it's
more intuitive.

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
in the H-step rollout, Phase 3). The epistemic identity
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
   fixed constant. CLARIFIED by rfcs/004: cross-entropy paired with −info-gain (as
   here) and KL-risk paired with +ambiguity are the SAME objective — NOT a
   behavioural fork. The genuine literature fork is FULL vs *mean-only* (drop the
   ½tr(ΛS) term → an agent blind to predicted-observation variance/ambiguity). The
   *forbidden mix* (KL-risk pragmatic − info-gain) is a double-counting BUG, not an
   option. rfcs/004 holds the discriminating tests (they need a state-dep sensor).
3. Epistemic = STATE information gain (salience), not parameter information gain
   (novelty). We compute I(state; obs) only; parameter/novelty EFE is out of scope.
4. We linearize the sensor at μ⁺ (the predicted mean). For a fixed sensor this is
   irrelevant; for a nonlinear sensor *where* you linearize matters (Phase 2).
5. Sign convention: G is MINIMISED; ``pragmatic`` is a cost (lower better) and
   ``epistemic`` is a value (higher better), so G = pragmatic − epistemic.

NOT IMPLEMENTED (named seams): the *mean-only* pragmatic (drops ½tr(ΛS); the real
literature alternative — an ambiguity-blind agent); parameter/novelty info gain.
The KL-risk grouping is NOT a separate option: paired correctly (+ambiguity) it is
this same G; paired with −info-gain it double-counts H[Q(o)] (a bug to avoid).
"""

import jax.numpy as jnp
from jaxtyping import Array, Float64

from cpomdp.selection import Preference
from cpomdp.types import Belief, LinearGaussianModel

__all__ = ["expected_free_energy"]


def expected_free_energy(
    model: LinearGaussianModel,
    belief: Belief,
    action: Float64[Array, "p"],
    preference: Preference,
) -> tuple[Float64[Array, ""], dict[str, Float64[Array, ""]]]:
    """Expected Free Energy of taking ``action`` from ``belief``, and its split.

    Computes ``G = pragmatic − epistemic`` for the locked linear-Gaussian
    definition documented at the top of this module. Pure ``jnp``, so it composes
    under ``jit``/``vmap``/``grad`` — in particular ``vmap``/``grad`` over a batch
    of candidate ``action`` vectors (with ``model``/``belief``/``preference`` held
    fixed), which is how ``EFESelector`` will search.

    Args:
        model: The generative model. Must have a control matrix (an action has no
            meaning without one). Its ``observation`` supplies the local ``(C, R)``;
            ``None`` means the fixed sensor ``(sensor_model, sensor_noise)``.
        belief: The current belief ``(μ, Σ)``.
        action: The candidate action ``a``, shape ``(p,)``.
        preference: The goal as an OBSERVATION-space ``Preference`` — ``goal`` is a
            preferred observation ``g`` (shape ``(m,)``) and ``precision`` is ``Λ``
            (shape ``(m, m)``). See FRAGILE(lit) #1 in the module docstring.

    Returns:
        ``(G, {"pragmatic": ..., "epistemic": ...})`` — the scalar EFE and its two
        non-negative components. Lower ``G`` is preferred.

    Raises:
        ValueError: If the model has no control matrix.
    """
    if model.control is None:
        raise ValueError(
            "expected_free_energy needs a model with a control matrix; an action "
            "has no effect on a control-free (pure-tracking) model."
        )
    control = model.control  # narrowed to Array by the guard above

    action = jnp.asarray(action, dtype=float)
    mu, sigma = belief.mean, belief.cov

    # --- predict: push the belief one step through the dynamics under `action` ---
    # Mirrors the covariance predict in kalman._gain_and_posterior_cov (cov_pred);
    # NB the action moves only the mean — Σ⁺ is action-independent, which is the
    # whole reason the epistemic term collapses under a fixed sensor (ADR-003).
    # FRAGILE(lit): nothing fragile here — this is exact Kalman prediction.
    mu_pred = model.A @ mu + control @ action
    sigma_pred = model.A @ sigma @ model.A.T + model.Q

    # --- sense: local linear-Gaussian observation model at the predicted mean ---
    # FRAGILE(lit) #4: linearize at μ⁺. Irrelevant for a fixed sensor; matters for
    # a nonlinear sensor (Phase 2 / CallableSensor).
    if model.observation is None:
        sensor_model, sensor_noise = model.C, model.R
    else:
        sensor_model, sensor_noise = model.observation.linearize(mu_pred)

    o_pred = sensor_model @ mu_pred
    # S: covariance of the predicted observation = obs-noise inflated by the
    # state uncertainty seen through the sensor. Computed ONCE; feeds both terms.
    pred_obs_cov = sensor_model @ sigma_pred @ sensor_model.T + sensor_noise

    # --- pragmatic: expected negative log-preference (cross-entropy form) ---
    # FRAGILE(lit) #1: `preference` is read in OBSERVATION space (g over o, Λ over o).
    # FRAGILE(lit) #2: cross-entropy form = mean term + ½tr(ΛS). The ½tr(ΛS) piece
    # is the variance penalty that distinguishes this from the mean-only form and,
    # via −½ln det S, from the KL-risk form. rfcs/004 must prove this is the right one.
    goal, precision = preference.goal, preference.precision
    residual = o_pred - goal
    pragmatic_mean = 0.5 * residual @ precision @ residual
    pragmatic_var = 0.5 * jnp.trace(precision @ pred_obs_cov)
    pragmatic = pragmatic_mean + pragmatic_var

    # --- epistemic: state information gain I(state; obs) = ½ ln(det S / det R) ---
    # FRAGILE(lit) #3: this is *salience* (state info gain), not *novelty* (parameter
    # info gain). slogdet (not det) for numerical stability; the sign is +1 for the
    # PD covariances here, so we keep only the log-abs-det.
    _, logdet_pred_obs = jnp.linalg.slogdet(pred_obs_cov)
    _, logdet_noise = jnp.linalg.slogdet(sensor_noise)
    epistemic = 0.5 * (logdet_pred_obs - logdet_noise)

    # FRAGILE(lit) #5: G = pragmatic − epistemic (minimise). Pairing cross-entropy
    # with −info-gain is decomposition (b); it is self-consistent (no double-count).
    g = pragmatic - epistemic
    return g, {"pragmatic": pragmatic, "epistemic": epistemic}
