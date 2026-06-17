# Architecture Decisions

Decisions are append-only. Each records the choice, the evidence, and the date.

---

## ADR-005 — v0.3 EFE decomposition: **observation-space cross-entropy pragmatic − state info-gain epistemic** (provisional / speculative)

**Date:** 2026-06-17
**Status:** Accepted **provisionally** — the *choice of form* is not yet validated; see "Validation obligation" (rfcs/004).
**Phase:** v0.3, Phase 1A (the one-step EFE core, `efe.py`)
**Extends:** ADR-003 (which argued the EFE collapse in *state* space; this commits v0.3 to *observation*-space EFE and records the resulting tension).

### Decision

The one-step Expected Free Energy for the linear-Gaussian regime is computed as
**decomposition (b): cross-entropy pragmatic minus state information-gain
epistemic**, with `G = pragmatic − epistemic` minimised. For belief `(μ, Σ)`,
action `a`, model `(A, B, Q)` with sensor `(C, R)`, and an **observation-space**
preference `(g, Λ)`:

    μ⁺ = Aμ + Ba    Σ⁺ = AΣAᵀ + Q    o⁺ = Cμ⁺    S = CΣ⁺Cᵀ + R
    pragmatic = ½(o⁺ − g)ᵀΛ(o⁺ − g) + ½tr(ΛS)        # cross-entropy = −E_Q[ln P(o)] + const
    epistemic = ½(ln det S − ln det R)               # = I(state; obs), state info gain ≥ 0

`S` is computed once and feeds both terms; `Σ_post`/Kalman gain are not needed for
the one-step value (only for the H-step rollout, Phase 3). The full derivation and
the per-line `FRAGILE(lit)` flags live in `efe.py`'s module docstring.

### Why this is flagged speculative

There is **no single agreed EFE formula** in the active-inference literature: the
pragmatic term has at least three forms in circulation and sources disagree on
signs and on whether risk is a cross-entropy or a KL. This is an area the owner is
candidly **outside their core expertise** on. We are choosing one route and
*committing to prove it* rather than asserting it is canonical.

### The three candidate pragmatic forms (the disagreement axis)

- **mean-only:** `½(o⁺ − g)ᵀΛ(o⁺ − g)` — drops the variance penalty.
- **cross-entropy (CHOSEN):** `mean + ½tr(ΛS)` = `−E_Q[ln P(o)]` up to a constant.
- **KL-risk:** `cross-entropy − H[Q(o)]` = `mean + ½tr(ΛS) − ½ln det S − ½ln det Λ − m/2`.

**No-double-count rule (load-bearing):** cross-entropy pairs with **−info-gain**
(decomposition b); KL-risk pairs with **+ambiguity** `½ln det(2πe R)` (decomposition
a). Both give the *same* `G`. Pairing KL-risk with −info-gain double-counts `H[Q(o)]`.
Our pairing (cross-entropy − info-gain) is internally consistent.

**Validated correction (rfcs/004, multi-agent + numeric proof).** The framing above
of "three candidate forms" is partly misleading: cross-entropy (−info-gain) and
*correctly-paired* KL-risk (+ambiguity) are the **same objective** (differ only by
the constant `c`), so they can never be discriminated. The genuinely distinct trio
is **mean-only / full form / forbidden mix**: their S-dependent parts (scalar, Λ)
are minimised at `S = ∞`-indifferent, `S = 1/Λ`, and `S = 2/Λ` respectively. So the
real literature fork is **mean-only vs the full form** (whether risk includes the
`½tr(ΛS)` predicted-observation-variance penalty); the forbidden mix is a
double-counting *bug*, not a third option. Independent re-derivation verified the
locked algebra to machine precision.

### Open tensions (do not lose these)

1. **Preference domain.** v0.3 EFE reads preferences in **observation** space
   (canonical pymdp/Friston), but the v0.1 LQR path uses a **state**-space goal.
   The single `Preference` type now has two consumers with different domain
   assumptions; they coincide only when `C = I`. Reconciling them (map via `C`, or
   a typed domain) is an **open design item**, deferred to the EFESelector/Agent
   wiring (Phase 4–5).
2. **Salience only.** We compute *state* information gain (salience), not
   *parameter* information gain (novelty). Novelty is out of scope.

### Validation obligation (the reason this ADR is "provisional")

Because the literature disagrees, **a passing implementation is not evidence of a
correct choice.** Critically, the **fixed-sensor collapse test does NOT
discriminate** the three forms — they differ only by terms that are constant in
the action under a fixed sensor, so all three pass it. A genuine discriminating
test must (a) use a state-dependent sensor so the forms choose different actions,
or (b) check the value against an independent oracle, or (c) verify the
decision-theoretic limit reductions (Sajid et al. 2021: flat-preference → Bayesian
optimal design; no-ambiguity → expected utility). **rfcs/004** now records the
discriminating plan (produced by a multi-agent research pass): an analytic
tied-mean / straddled-S argmin flip [*proves*], a murky-goal-corridor behavioural
test [*hints*], and an MC convention-independent cross-check [*proves faithfulness*]
— all requiring the Phase-2 state-dependent sensor. Until one is implemented and
passes, treat the *form choice* (not the implementation) as unproven. Honest
ceiling: no oracle can prove decomposition (b) is *the* correct EFE; the strongest
earnable claim is "self-consistent and double-count-free."

### Validation strategy (implementation correctness, distinct from form choice)

`expected_free_energy` is checked against an **independent NumPy oracle**
(`tests/test_efe.py::_numpy_efe`, a separate code path — no shared helpers), plus
the collapse property and `jit`/`vmap`/`grad` agreement. That confirms the *algebra
of the chosen form* is right; it says nothing about whether the form is the right
one (see above).

---

## ADR-004 — v0.2 array backend: **JAX (`jax.numpy`), pytree-registered types, x64 on import**

**Date:** 2026-06-15
**Status:** Accepted
**Phase:** v0.2 (array-backend migration)
**Amends:** ADR-002 (supersedes its "JAX: not adopted reflexively … core stays NumPy-only" scope guard)

### Decision

The core array backend moves from NumPy to `jax.numpy`. Three concrete choices:

1. **Pytree-registered types, jit-ready hot paths.** `Belief` and
   `LinearGaussianModel` register as JAX pytrees, and the per-step filter and
   action selection are pure functions over `jnp` arrays. This is the actual
   payoff: `vmap` over beliefs, `grad` of a cost/EFE, and `jit`-compiled rollouts
   become available without a second rewrite.
2. **`jax_enable_x64` is set at import** (`cpomdp/__init__.py`). The whole library
   is validated against the RxInfer oracle to 1e-9; JAX defaults to float32, which
   would silently break those matches. The trade-off is a process-global side
   effect — importing `cpomdp` flips x64 on for the user's entire JAX session.
   Accepted because silent float32 degradation in a numerical library is the worse
   failure.
3. **NumPy is kept as a dependency.** JAX pulls it in transitively anyway, and the
   RxInfer backend hands *real* numpy arrays across the juliacall boundary
   (juliacall does not speak `jax.Array`). Core maths is `jnp`; numpy survives at
   the Julia boundary and in test assertions.

### Why now (the ADR-002 trigger has fired)

ADR-002 deferred JAX until "autodiff (EFE gradients, param learning) or vmap/GPU
actually pays." v0.2's roadmap is exactly that work — gradients of preferences and
batched rollouts — so the trigger condition it named has arrived. v0.1 was a
proof of concept; the migration is the first v0.2 increment, done module-by-module
under TDD with the RxInfer oracle held fixed as the cross-check.

### Validation strategy

Unchanged in spirit: the native JAX filter is still checked against the RxInfer
oracle and the per-step Kalman recursion. The oracle path stays NumPy/Julia, so
agreement to tolerance is an independent confirmation the `jnp` algebra is right.

---

## ADR-003 — v0.1 grows an acting agent: stateful `Agent` + front-loaded LQR

**Date:** 2026-06-14
**Status:** Accepted
**Phase:** 2 (abstraction wall) → 3 (agent assembly)
**Amends:** ADR-002 (reverses its "LQR/control side: deferred" scope guard)

### Decision

v0.1 ships an agent that *acts*, not just one that perceives. Two additions:

1. A stateful `Agent` façade that owns the current belief and exposes
   `infer_states(obs, action=None)` and `sample_action()` — the continuous answer
   to pymdp's `Agent`.
2. Action selection via a **front-loaded steady-state LQR** controller: solve the
   control Riccati once at construction for `L∞`, then `u = -L∞·(mean − goal)` in
   the loop.

ADR-002 deferred the whole control side. We're pulling it back because without it
the library is a Kalman filter with a nice type system, not the "continuous
sibling of pymdp" the README promises. pymdp's shape is perceive → evaluate → act;
shipping only the first verb undersells what turns out to be a small amount of
remaining work.

### Why LQR counts as active inference here (the load-bearing argument)

The objection to adding LQR is that we've quietly swapped active inference for
plain optimal control. We haven't, and the reason is specific to the
linear-Gaussian case.

Expected Free Energy has a pragmatic term (reach preferred observations) and an
epistemic term (act to reduce uncertainty). In `LinearGaussianModel` the
covariance recursion is **control-independent** — the same property that lets us
front-load `K∞`. Control shifts the mean only; it never touches the covariance. So
the epistemic value (expected entropy reduction `½·log(det Σ_pred / det Σ_post)`)
is identical for every action and falls out of the argmin. EFE-minimising action
selection *provably* reduces to its pragmatic term, and the pragmatic term under a
Gaussian preference is a quadratic cost whose optimum is LQR.

So LQR isn't a stand-in for EFE here — it's what EFE *is* when sensing doesn't
depend on where you are. The epistemic term only re-enters once the observation
model becomes state- or action-dependent (position-varying sensor precision,
choosing a modality), which is out of scope for a fixed linear-Gaussian sensor. We
record that as the seam, not a gap.

### The symmetry we're buying

Filter and controller become duals, both solved once at construction, neither
dependent on data:

- perception: Kalman/DARE → `K∞`, loop does `mean += K∞·prediction_error`
- action:     control Riccati → `L∞`, loop does `u = -L∞·(mean − goal)`

Together that's LQG. The front-loading thesis (RESEARCH.md) now covers both halves
of the agent, not just perception.

### Interface shape

- `Agent` is stateful: it holds `belief` (the analog of pymdp's `qs`) and updates
  it in place across `infer_states` calls. The backends stay functional/pure
  underneath — façade for ergonomics, engine for testability.
- Preferences live on the `Agent`, not the model. The model is the generative
  story; the goal and the effort trade-off are the agent's. Role-named to avoid
  the Q/R collision (`dynamics_noise`/`sensor_noise` are already "Q"/"R", and
  LQR's cost matrices are conventionally Q/R too): `goal`, `effort_penalty`, etc.
- `sample_action()` reads the current belief mean — one matrix-vector product, no
  inference of its own.

### Scope (v0.1, updated)

- **Added:** stateful `Agent`, steady-state LQR controller (front-loaded `L∞`),
  agent-side preferences, 2D point-mass reaching demo that closes the loop (the
  agent chooses the action).
- **Still deferred:** epistemic/exploratory EFE (named seam above), receding-
  horizon and time-varying control, nonlinear control. `CovarianceRep`, BMR — as
  in ADR-002.

### Validation strategy

Same discipline as the filter. `L∞` is checked against an independent oracle —
scipy's `solve_discrete_are` (control algebraic Riccati) — so a bug in our own
solve can't pass silently. The reaching demo is the end-to-end acceptance test:
the point mass must converge to `goal` under the closed loop.

---

## ADR-002 — v0.1 inference engine: **native fixed-gain fast path; RxInfer as oracle + general fallback**

**Date:** 2026-06-12
**Status:** Accepted
**Phase:** 2 (the abstraction wall)
**Amends:** ADR-001 (does not revoke it — re-roles RxInfer rather than removing it)

### Decision

v0.1's *default* inference is a **native, front-loaded steady-state Kalman
filter** (Option 1 in the build plan), exposed as a backend behind the
`InferenceBackend` Protocol. **RxInfer (via juliacall, per ADR-001) is retained
as a second backend** — serving now as the *correctness oracle* and later as the
*general engine* for the cases the native fast path cannot handle (nonlinear,
non-stationary, intermittent observations, structure learning, hierarchical).

### Why this changes ADR-001's emphasis

ADR-001 made RxInfer "the engine." The front-loading analysis (RESEARCH.md) shows
that for the **LTI-Gaussian** v0.1 scope the inference loop reduces to a fixed-gain
filter so cheap that RxInfer would never run in the hot path — it would be a Julia
dependency carried for nothing. We arrive at the native path *not* because the
bridge failed (it worked, ADR-001 stands as evidence) but because front-loading
removes the only reason the bridge was load-bearing. The Phase-2 abstraction wall
is exactly what lets both coexist as swappable backends instead of a fork.

### The principle being implemented (RESEARCH.md)

**Front-load the *structure* of the computation, never the *values*.** For an LTI
Gaussian model the covariance/gain sequence is data-independent: solve the
discrete algebraic Riccati equation (DARE) **once at agent construction** to get
the steady-state gain `K∞`, then run a fixed-gain update in the loop. No
inversion, no covariance update, no O(n³) op in the hot path.

### Scope guards (resisting the doc's own scope creep)

- **In for v0.1:** `Belief` (plain covariance, scalar), `InferenceBackend`
  Protocol, native fixed-gain backend (DARE → `K∞` + warmup), RxInfer oracle
  backend, 2D point-mass reaching demo validated against a full per-step Kalman.
- **Deferred (named seams only, no impl):** `CovarianceRep` strategy/Protocol
  (YAGNI until a 2nd representation exists — scalar is the trivial 1×1 case of
  all three), BMR outer loop, LQR/control side.
- **JAX:** not adopted reflexively. v0.1 scalar fixed-gain is instant in NumPy;
  JAX is revisited when autodiff (EFE gradients, param learning) or vmap/GPU
  actually pays. Core stays NumPy-only until then.

### Boundaries where the native fast path is INVALID (fall back to RxInfer)

- Nonlinear models — EKF/UKF gains depend on the linearisation point → the
  estimate → the data → gains become data-dependent → not front-loadable.
- Non-stationary `A,Q,R` — `K∞` goes stale; needs drift detection + re-solve.
- **Intermittent / irregularly-sampled / varying-`R` observations** — breaks the
  "regular complete observations" assumption that makes `K` constant.

### Validation strategy

The native filter's posterior is checked against (a) a plain NumPy RTS
smoother / full per-step Kalman (analytic oracle) and (b) the RxInfer backend.
The Phase-0 spike (`spike/`) is re-roled from "shipping engine prototype" to
"oracle harness."

---

## ADR-001 — Backend bridge shape: **Shape A (juliacall, in-process)**

**Date:** 2026-06-12
**Status:** Accepted (emphasis amended by ADR-002)
**Phase:** 0 (verification spike — the gate)

### Decision

cpomdp's v0.1 inference engine is **RxInfer.jl, reached in-process via `juliacall`**
(Shape A). Not the HTTP `RxInferClient.py` → `RxInferServer.jl` route (Shape B).

### Evidence from the spike (`spike/`, throwaway)

A scalar linear-Gaussian state-space model was the test vehicle:

    xₜ = A·xₜ₋₁ + 𝒩(0,Q),   yₜ = B·xₜ + 𝒩(0,R),   x₀ ~ 𝒩(m0,v0)

1. **Julia-only ground truth** (`lgssm_groundtruth.jl`): RxInfer runs, posteriors
   read out cleanly. **Validated correct** against an independent NumPy RTS
   smoother (`rts_oracle.py`) — agreement to **5e-13** (machine precision).
2. **juliacall bridge** (`juliacall_driver.py`): the *same* model driven from
   Python — NumPy array in, array out — reproduced the Julia-only posteriors to
   **5e-13**. The bridge introduces no numerical error.
3. **Shape B not deeply evaluated.** The decision rule in the build plan defaults
   to Shape A unless it proves unworkable. It held on the first real attempt, so
   the default stands. Shape B remains a documented fallback, not a need.

### Consequences / things learned (carry into Phase 1+)

- **Toolchain that worked:** Julia **1.12.6** (via juliaup), **RxInfer v5.4.0**,
  **juliacall 0.9.35**, on **CPython 3.14.5**. The feared Python-3.14
  incompatibility did **not** materialise — 3.14 is fine.
- **juliacall needs PythonCall.jl in the active Julia project.** It's juliacall's
  Julia-side counterpart. The real backend must ensure both PythonCall.jl and
  RxInfer.jl are present — juliacall ships a `juliapkg.json` mechanism for
  declaring Julia deps; cpomdp should ship its own `juliapkg.json` declaring
  RxInfer so `pip install cpomdp[rxinfer]` auto-provisions the Julia side.
- **Startup cost is real but acceptable.** First-ever run paid a one-time
  ~70s (registry update + PythonCall add + precompile). Steady-state startup is
  the `import juliacall` + `using RxInfer` load (tens of seconds, JIT warmup),
  paid once per process — not per inference. Not prohibitive for a library used
  in a session; worth a note in user docs.
- **Inference convention — SMOOTHER, not filter.** Handing RxInfer a whole
  observation sequence at once yields the *smoothed* posterior p(xₜ|y₁..y_T)
  (message passing flows both directions). The Phase-4 correctness oracle must
  therefore be an **RTS smoother** (already written: `rts_oracle.py`), not a bare
  Kalman filter. For an online agent acting in real time we will likely want the
  *filter* instead — drive RxInfer in streaming/one-observation-at-a-time mode.
  Decide this when building `agent.py`.

### The wall (unchanged, restated)

juliacall, PythonCall, RxInfer, and the `@model` DSL all live behind
`backends/base.py`'s Protocol. None of it appears in any public signature, return
type, exception, or docstring. Shape A vs B is an implementation detail the wall
makes swappable.

## On changing the matrices names
To explicitly name the matricices to avoid further confusion and collision within the space.
An example:

LinearGaussianModel(
    dynamics=...,        # A: state → next state
    control=...,         # B: action → state
    observation=...,     # C: state → observation
    process_noise=...,   # Q
    observation_noise=...,# R
)

The letters can survive as aliases/internal attributes and definitely in the docstrings but the primary interface is role-named.
