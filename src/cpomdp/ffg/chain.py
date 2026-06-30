"""FFG chain inference backend: ``predict ∘ update ∘ to_moment`` = a Kalman step.

The keystone of v0.4 Phase 2 (ADR-012): wiring the Phase-1/Phase-2 message algebra
into the ``InferenceBackend`` protocol and showing that, on a *chain* topology, it
reproduces the existing Kalman path. Gaussian belief propagation on a linear chain
*is* the Kalman filter — so this backend is interchangeable with ``KalmanBackend``
behind the same seam, and the gate (``tests/test_ffg_chain.py``) holds it to
numerical identity against that path.

One step decomposes into the owned algebra exactly as the factor docstrings promise::

    prior (moment) ──┐
                     ▼  to canonical: Λ₀ = Σ⁻¹, h₀ = Λ₀μ
        GaussianTransition.predict(prior_msg, control_term)   # predict  (x → x')
                     ▼
        predicted_msg + GaussianObservation.message(y)        # update   (+ = product)
                     ▼  to_moment: Σ = Λ⁻¹, μ = Λ⁻¹h
                posterior (moment)

Scope (tier-1). Both factors invert their noise covariance, so a deterministic
(``Q = 0``) transition has no information form and is rejected by the transition
factor regardless of fixedness — the one documented divergence from moment-form
Kalman, harmless for the chain the gate exercises. State-dependent ``R(x)``/``Q(x)``
(the ``observation``/``process_noise`` fields) reach parity with ``KalmanBackend``
in Phase 2.5 (ADR-012 amendment 2026-06-26): the fixed sides keep their front-loaded
factors; a state-dependent side is linearized at the predicted mean ``μ⁻`` and its
factor rebuilt per step (see ``infer_states``).

Energy note (RFC-001). Bridging a moment-form protocol (``Belief`` in, ``Belief``
out) to info-form internals costs two inversions per step that the native Kalman
path does not pay: ``Σ⁻¹`` to lift the incoming prior into canonical form, and
``Λ⁻¹`` to read the posterior back out. The *factors* themselves are front-loaded —
built once at construction from the fixed model matrices, never per step — so the
loop body stays the four cheap algebra ops above. This backend is the correctness
demonstration, not the production hot path; the extra inversions are the price of
the protocol bridge, flagged here rather than hidden.
"""

import jax.numpy as jnp
from numpy.typing import ArrayLike

from cpomdp.backends.base import validate_step_inputs
from cpomdp.ffg.factors.linear_gaussian import GaussianObservation, GaussianTransition
from cpomdp.ffg.message import CanonicalGaussian
from cpomdp.types import Belief, LinearGaussianModel

__all__ = ["ChainBackend"]


class ChainBackend:
    """FFG message-passing inference on a linear-Gaussian *chain*.

    Implements the ``InferenceBackend`` protocol via the canonical-form message
    algebra (``CanonicalGaussian`` + the tier-1 factors), not the moment-form
    Kalman recursion. Constructed from a model, then advances a belief one step at
    a time (prior in, posterior out); see the module docstring for the per-step
    decomposition and the scope/energy notes.

    Args:
        model: The linear-Gaussian generative model to filter under. A
            state-dependent ``observation`` (``R(x)``) or ``process_noise``
            (``Q(x)``) is supported (Phase 2.5) and linearized at the predicted
            mean each step; the fixed sides are front-loaded once here. Whichever
            covariance ends up feeding the transition factor (fixed ``dynamics_noise``
            or a state-dependent ``Q(x)``) must be positive-*definite* (the
            information form inverts it) — a fixed ``Q = 0`` is rejected here, a
            state-dependent ``Q(x)`` evaluating to non-PD is rejected per step.
    """

    def __init__(self, model: LinearGaussianModel) -> None:
        """Front-load the fixed factor nodes; leave state-dependent ones for later.

        Build the ``GaussianObservation`` (from C, R) and ``GaussianTransition``
        (from A, Q) *once* here when they are data-independent — constructing them
        per step would burn compute the fixed regime doesn't need (RFC-001). Test
        fixedness with the ``is_fixed`` flag, not ``is None`` (an ``observation`` can
        be present but fixed, e.g. a ``FixedSensor``), mirroring ``KalmanBackend``::

            sensor_fixed  = model.observation   is None or model.observation.is_fixed
            process_fixed = model.process_noise is None or model.process_noise.is_fixed

        When a side is *not* fixed, the corresponding factor is left ``None`` here
        and built per step in ``infer_states`` from ``observation.linearize(μ⁻)`` /
        ``process_noise.noise_at(μ⁻)`` instead (Phase 2.5, ADR-012 amendment
        2026-06-26). This is not just laziness: ``GaussianTransition`` requires Q
        positive-*definite* (it inverts it), but a model carrying a state-dependent
        ``process_noise`` is only required to give ``model.dynamics_noise`` itself a
        positive-*semi*-definite placeholder (it's unused) — front-loading
        unconditionally would reject that legitimate placeholder.

        Args:
            model: see the class docstring.
        """
        self.model = model
        self._sensor_fixed = model.observation is None or model.observation.is_fixed
        self._process_fixed = (
            model.process_noise is None or model.process_noise.is_fixed
        )
        self._transition: GaussianTransition | None = (
            GaussianTransition(model.dynamics, model.dynamics_noise)  # A, Q
            if self._process_fixed
            else None
        )
        self._observation: GaussianObservation | None = (
            GaussianObservation(model.sensor_model, model.sensor_noise)  # C, R
            if self._sensor_fixed
            else None
        )

    def infer_states(
        self,
        observation: ArrayLike,
        prior: Belief,
        action: ArrayLike | None = None,
    ) -> Belief:
        """Advance the belief by one filter step via the FFG algebra.

        Validate the per-step inputs at the trust boundary with the shared
        ``validate_step_inputs`` (identical checks to ``KalmanBackend`` — same seam,
        same errors), form the control shift ``b = control @ action`` (zero when the
        model is uncontrolled), then run the module-docstring pipeline: lift the
        prior into canonical form, ``predict`` through the transition factor, add the
        observation factor's ``message(y)`` (the measurement update), and
        ``to_moment`` the result back into a ``Belief``.

        On a state-dependent side (Phase 2.5), the transition/observation factor for
        *this* step is built from ``process_noise.noise_at(μ⁻)`` /
        ``observation.linearize(μ⁻)``, where ``μ⁻ = A·prior.mean + b`` is the
        predicted mean — pure mean-propagation, so it needs no Q and can be computed
        before any factor exists. This is exactly ``KalmanBackend``'s linearization
        point (ADR-008), so the two backends see the same noise each step. The fully
        fixed path computes no extra matvec and reuses the front-loaded factors.

        Args:
            observation: The latest sensor reading, shape ``(m,)``.
            prior: The current belief, this step's previous posterior. Never mutated.
            action: The action just taken, shape ``(p,)``. Required iff the model has
                a control matrix; pass ``None`` for pure filtering.

        Returns:
            The posterior belief — a new ``Belief``; the prior is left untouched.

        Raises:
            ValueError: If ``observation`` is not shape ``(m,)``, ``prior`` is not a
                belief over the model's ``n``-D state, the model has a control matrix
                but ``action`` is ``None``, or ``action`` is not shape ``(p,)``.
        """
        observation, action = validate_step_inputs(
            self.model, observation, prior, action
        )
        model = self.model
        control = model.control
        if control is None:
            control_term = jnp.zeros(model.n_states)
        else:
            # validate_step_inputs guarantees a non-None action when control exists
            assert action is not None
            control_term = control @ action

        # μ⁻ is needed only to linearize a state-dependent sensor and/or process
        # noise; the fully-fixed hot path computes no extra matvec (mirrors
        # KalmanBackend, ADR-008).
        mean_pred = (
            model.dynamics @ prior.mean + control_term
            if not (self._sensor_fixed and self._process_fixed)
            else prior.mean  # placeholder, unused on the fixed path
        )

        if self._process_fixed:
            assert self._transition is not None  # built in __init__ on this path
            transition = self._transition
        else:
            assert model.process_noise is not None  # guaranteed by _process_fixed
            transition = GaussianTransition(
                model.dynamics, model.process_noise.noise_at(mean_pred)
            )

        if self._sensor_fixed:
            assert self._observation is not None  # built in __init__ on this path
            observation_factor = self._observation
        else:
            assert model.observation is not None  # guaranteed by _sensor_fixed
            sensor_model, sensor_noise = model.observation.linearize(mean_pred)
            observation_factor = GaussianObservation(sensor_model, sensor_noise)

        prior_precision = jnp.linalg.inv(prior.cov)  # Λ₀ = Σ⁻¹
        prior_msg = CanonicalGaussian._unchecked(
            prior_precision, prior_precision @ prior.mean
        )  # h₀ = Λ₀μ; invariant-preserving lift of a validated Belief — no re-validate

        predicted = transition.predict(prior_msg, control_term)
        posterior_msg = predicted + observation_factor.message(observation)

        mean_post, cov_post = posterior_msg.to_moment()  # Σ = Λ⁻¹, μ = Λ⁻¹h

        return Belief(mean=mean_post, cov=cov_post)
