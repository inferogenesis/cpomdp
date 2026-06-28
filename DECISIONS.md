# Architecture Decisions

Decisions are append-only. Each records the choice, the evidence, and the date.

---

## ADR-006 — v0.3 Phase 2: **state-dependent sensing and internal process noise** (the collapse breaks two ways)

**Date:** 2026-06-18
**Status:** Accepted
**Phase:** v0.3, Phase 2
**Extends:** ADR-003 (its fixed-sensor collapse now breaks), ADR-005 (the EFE kernel these seams feed)

### Decision

The EFE epistemic term re-enters action selection. Three resolutions make that
happen, and the collapse of ADR-003 now breaks from two independent directions.

1. **D0 — noise-first sensor.** `CallableSensor` carries state-dependent observation
   noise `R(x)` with a *constant* `C`. The mean stays linear, so `o⁺ = C·μ⁺` is
   exact and the kernel's mean code is untouched; the action-dependence lives in
   `R(μ⁺)`. The nonlinear-mean case (a curved `g(x)`, needing a 2nd-order moment
   match) is a separate, riskier class — deferred to Phase 2.5 (see below).

2. **D1 — the `gaussianize` seam.** Each `ObservationModel` owns its
   predicted-observation moment match: `gaussianize(x, Σ) → (o⁺, S, R)`. The kernel
   calls that, never reconstructing `o⁺`/`S` itself, so a fixed sensor
   (`observation is None`) stays a bare matvec on the hot path while a future
   nonlinear sensor does 2nd-order work without the kernel being reopened
   (Open-Closed). The return is a *triple*, not `(o⁺, S)`: the epistemic term needs
   `R` for `½ln det R`, and bundling it is one sensor call instead of two (and
   avoids recomputing a Jacobian for the nonlinear case).

3. **D2 — internal process noise at `μ⁺`.** An optional `process_noise: DynamicsNoise`
   on the model supplies `Q(x)`; when present it replaces the fixed `dynamics_noise`
   matrix and is evaluated at `μ⁺`. The honest reason for `μ⁺` (not `μ`): `Q` is the
   diffusion of the *arrived-at* state, so end-of-interval discretization evaluates
   it at the end of the step. Action-dependence falls out of that, it is not the
   motive. This is the internal-noise route to epistemic value (RFC-001 chapter 8):
   the binding precision constraint can live in internal processing, not only the
   sensor — the picture the 2025 *E. coli* work insists on.

So `Σ⁺` and the per-observation information gain depend on the action through either
`R(μ⁺)` (the input route) or `Q(μ⁺)` (the internal route). The epistemic term
`½(ln det S − ln det R) = I(state; obs)` is `İ_silicon`, the bits resolved per
observation — but it is the *perceptual ceiling* (signal→belief), not Mattingly's
signal→action rate (RFC-001 chapter 8).

### Deferred to Phase 2.5 (not built)

`NonlinearSensor` + 2nd-order Gaussianization — the nonlinear-mean case and the Kouw
curvature-avoidance demonstrator. The `gaussianize` seam lands *linear* in Phase 2,
so 2.5 is a pure additive class with no kernel edit. The corrected full-2nd-order
formula (mean **and** covariance correction together — taking one without the other
is a real bug) and its dual-oracle definition-of-done are pinned in the build plan
so they cannot be re-forgotten.

### Validation

- **Form-proof (RFC-004, Phase 2b):** the kernel is the *full* form, not mean-only
  (the `½tr(ΛS)` variance penalty is present), not the forbidden mix (its `G`
  differs by exactly `H[Q(o)]`). A Monte-Carlo cross-entropy estimate confirms the
  pragmatic *formula*, independent of the analytic NumPy oracle that confirms the
  *implementation*.
- **The clean straddled-S flip** (full picks `S=1/Λ`, the forbidden mix picks
  `S=2/Λ`) lives in the internal-`Q` regime (2d), where `R` is held fixed so the
  flip's math is honest — in the `R(x)` regime `S` and `R` co-vary and the clean
  flip does not hold.
- **Hot path:** `observation=None` and `process_noise=None` produce byte-identical
  `Σ⁺`/`G` to Phase 1A (tested), so the fixed-sensor fast path is untouched.
- 9 new tests; full suite 106 green. Figures: `docs/assets/efe_collapse.png` (the
  input route) and `docs/assets/internal_noise.png` (the internal route).

---

## ADR-005 — v0.3 EFE decomposition: **observation-space cross-entropy pragmatic − state info-gain epistemic** (provisional / speculative)

**Date:** 2026-06-17
**Status:** Accepted — the validation obligation is **discharged in v0.3** (the Phase-2 discriminators landed; see "Resolution" at the end). The residual — no oracle can prove decomposition (b) is uniquely *the* EFE — is a permanent epistemic ceiling, not a v0.3 blocker.
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

### Resolution (v0.3 — obligation discharged)

The discriminating tests this ADR demanded have landed and pass (full suite green), so
the *form choice* is now validated to its honest ceiling:

- **Not mean-only:** `test_pragmatic_carries_variance_penalty_not_mean_only` — the
  `½tr(ΛS)` penalty moves `G` where mean-only would tie it.
- **Faithful to the cross-entropy:** `test_pragmatic_matches_monte_carlo_cross_entropy`
  (Phase 2b) — the pragmatic *formula* matches a Monte-Carlo estimate of
  `E_Q[½(o−g)ᵀΛ(o−g)]`, independent of the analytic oracle.
- **Full, not the forbidden mix:** `TestStraddledSFlip` (Phase 2d, internal-`Q` regime,
  `R` held fixed so the flip's math is honest) — the kernel picks `S=1/Λ` while the
  forbidden mix picks `S=2/Λ`; `test_kernel_g_is_full_not_forbidden_mix` shows the gap
  is exactly `H[Q(o)]`.

Open tension #1 (the observation- vs state-space `Preference` domain) was resolved by
**ADR-007** (typed `StateGoal`/`ObservationGoal`). What remains is only the *permanent*
ceiling this ADR already named — no oracle proves decomposition (b) is uniquely *the*
EFE — which is acknowledged, not a blocker. The earnable claim ("self-consistent,
double-count-free, and MC-faithful") is now earned.

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

To explicitly name the matrices to avoid further confusion and collision within the space.
An example:

LinearGaussianModel(
    dynamics=...,        # A: state → next state
    control=...,         # B: action → state
    observation=...,     # C: state → observation
    process_noise=...,   # Q
    observation_noise=...,# R
)

The letters can survive as aliases/internal attributes and definitely in the docstrings but the primary interface is role-named.

## ADR-007 — v0.3 Phase 4–5: typed Agent objectives + greedy EFESelector

**Status:** accepted. Resolves ADR-005 open tension #1 (the `Preference` domain split).

### What acts on the EFE kernel

Phase 1A built `expected_free_energy` but nothing chose actions with it. Phase 4–5
closes that gap: `EFESelector` turns the kernel into action selection, and `Agent`
learns to wire it.

**EFESelector (Phase 4).** Greedy, one-step (H=1): front-load a fixed grid of
candidate actions over the actuator box at construction, then each cycle `vmap` the
kernel over the grid and take the `argmin`. No inner optimiser. Per-cycle cost is
therefore *exactly* `n_candidates` kernel evaluations — a single attributable
number, which is the RFC-001 energy constraint made concrete (an optimiser's cost
is data-dependent and unattributable). Myopic by design; the H-step rollout is the
named Phase 3 seam, and the demos/tests stay honest about it (asserted against a
brute-force *one-step* oracle, never a horizon optimum). Critically, H=1 greedy EFE
is **not** infinite-horizon LQR even under a fixed sensor — it is the one-step
*deadbeat* pragmatic argmin; a test asserts the two differ so no one "simplifies"
the selector into `== LQRController`.

### The Preference-domain reconciliation (the open tension)

ADR-005 left this open: v0.3 EFE reads preferences in **observation** space, but the
v0.1 LQR path uses a **state**-space goal — one `Preference` type, two consumers
with different domain assumptions, coinciding only at `C = I`. The first cut (a
`goal=`/`preference=` kwarg pair on `Agent`) made an **illegal state representable**
— you could pass both, or pass LQR knobs with an obs preference — so it needed
runtime guards to bat away mistakes the API itself invited.

**Decision: a typed objective sum type.** The `Agent` takes one `objective`:

- `StateGoal(target, *, precision, effort)` → the LQR / state-space regime.
- `ObservationGoal(target, action_bounds, *, precision, n_candidates)` → the EFE /
  observation-space regime.
- `None` → a perceive-only tracker.

The objective's *type* is the dispatch key; the **sensor type** decides which regime
is legal (fixed → `StateGoal`/LQR, state-dependent → `ObservationGoal`/EFE), so one
agent never straddles both preference domains. This is the "typed domain" option
ADR-005 named, chosen over "map via `C`".

Why it beats the kwarg pair, concretely:

- **Illegal states are unrepresentable.** One objective slot, so "both given" cannot
  be expressed; the config bundles *into the type* (`effort` on `StateGoal`,
  `action_bounds`/`n_candidates` on `ObservationGoal`), so "LQR knob on an obs goal"
  is a `TypeError` at construction, not a runtime guard. The guards those mistakes
  needed simply evaporate.
- **`Preference` survives as an internal type.** The `Agent` extracts a
  `Preference(target, precision)` from either objective to hand the selector, so
  `select(belief, preference)` is unchanged and the v0.2 LQR path stays
  **byte-identical** (a regression test asserts exact equality, not `allclose`).
- **What's left are genuine objective/model compatibility checks**, not
  self-inflicted ones: `StateGoal` on a state-dependent sensor raises (don't convert
  through `C`); `ObservationGoal` on a control-free model raises; `ObservationGoal`
  on a fixed sensor raises (output regulation — see below) unless an explicit
  `selector=` overrides the dispatch.

`StateGoal`/`ObservationGoal` are top-level exports (the objects a user constructs);
the selectors stay in `cpomdp.selection` (the dispatch picks them; advanced override
via `selector=`).

### The biological reading (why observation-space is primary)

A `StateGoal` is a wish in *world* coordinates ("be at position x") — it assumes a
god's-eye fix on the agent's own configuration; the engineering case, and the
special case where the whole state is observed. An `ObservationGoal` is a wish in
*sensory* coordinates ("sense reading o") — what an organism actually has. *E. coli*
climbing a nutrient gradient has no concept of "move to (x, y)"; its preference is
"taste high concentration," and movement is the emergent side-effect. That is why
observation-space is primary and `StateGoal` is the privileged special case.

### Deferred (named, not built)

- **Output-regulation LQR** — letting the LQR/fixed-sensor path consume an obs-space
  preference (pull `Λ` back through `C`). When it lands, the duality collapses:
  everything is an `ObservationGoal`, `StateGoal` becomes sugar, and even the
  surviving compatibility guards mostly go. The current duality is transitional
  scaffolding pending this, *not* a committed design.
- **R(x) in perception.** The Kalman backend still filters with the model's fixed
  `R`; an `ObservationGoal` agent therefore *acts* on `R(x)` (via the kernel) but
  *perceives* on fixed `R`. Fine for dispatch, but it must be reconciled before the
  end-to-end "EFE drives uncertainty down faster than LQR" payoff (the RFC-001
  comparison), where the agent's online belief has to see `R(x)`.
- Multi-step (H≥2) rollout; `GradientEFESelector`; LQR-seeded / Sobol candidate
  grids; the mixture (disjunctive) `Preference`.

## ADR-008 — R(x) in perception: state-dependent sensor noise in the filter

**Status:** accepted. Closes the "R(x) in perception" deferred seam named in ADR-007.

### The gap

ADR-007 shipped an `ObservationGoal` agent that **acts** on state-dependent sensor
noise `R(x)` — the EFE kernel calls `model.observation.gaussianize(μ⁺, …)` — but
**perceives** on the model's *fixed* `R`: `KalmanBackend` only ever read
`model.sensor_noise`. So the agent would detour toward a high-precision beacon to
"sense better," yet its filter couldn't see the sharper sensing. This blocked the
v0.3 payoff (RFC-001): *EFE drives uncertainty below LQR*.

### Decision: linearize `R` at the predicted mean `μ⁻`, gated to callable sensors

`KalmanBackend.infer_states` now gates on the sensor type:

- **Fixed sensor** (`observation is None or is_fixed`): unchanged — direct reads of
  `model.sensor_model`/`model.sensor_noise`, no `linearize`, no dispatch. The hot
  path stays **byte-identical and lean** (RFC-001); the whole existing
  `test_kalman.py` suite passing unmodified is the regression proof.
- **State-dependent sensor**: compute the predicted mean `μ⁻ = A·μ + B·a`, then
  `(C, R) = observation.linearize(μ⁻)`, and feed that `(C, R)` to the (unchanged)
  jit kernels. One extra `μ⁻` matvec, callable path only.

`μ⁻` is the load-bearing choice: it is **exactly the EFE kernel's linearization
point**, so the agent's filter and its action-evaluation evaluate `R` at the same
state — "the agent perceives what it planned for." This makes the filter a
first-order EKF-style filter, consistent with the documented "mean-exact, R-plug-in"
approximation; the second-order Jensen term `½tr(H_R Σ⁺)` stays deferred to
`NonlinearSensor` (Phase 2.5), dropped consistently by filter *and* kernel.

**Steady-state mode is incompatible** with `R(x)` (no state-independent Riccati
fixed point) and now raises at construction rather than freezing a silently-wrong
gain. A single source of truth for `(C, R)`: both the gain/cov *and* the mean update
read the linearized `C` (a `CallableSensor` keeps `C` constant, so this is
byte-identical today, but it closes the trap for a future varying-`C` sensor).

### The payoff (validated, deterministic)

The end-to-end test compares the EFE agent's belief covariance against an LQR
baseline via a **covariance-only replay**: the Kalman covariance recursion is
observation-*independent*, so "the LQR path's uncertainty under the same `R(x)`" is
fully determined by the LQR agent's `μ⁻` sequence — no noise, no RNG, fully
deterministic. It isolates "the path won, not the model." Result on the
precision-well corridor: the EFE agent ends **~5× more certain** than LQR
(trace(cov) ≈ 0.03 vs ≈ 0.17).

An honest note on behaviour: with a *weak* preference the **epistemic drive
dominates** — the agent seeks and *holds* the beacon rather than returning to the
pragmatic goal (it never reaches observe-0). That is the correct active-inference
regime, not a bug; a stronger preference recovers goal-seeking but forgoes the
uncertainty win. The test documents this and asserts the uncertainty gap, the
detour mechanism, and that the LQR baseline does reach its own state goal.

### The `Q(x)` dual — closed in the same pass

The exact mirror of `R(x)`: `efe.py` evaluates state-dependent process noise at
`μ⁺` (`process_noise.noise_at`), so the filter now evaluates `Q(x)` at `μ⁻` in the
covariance predict, through the same `is None or is_fixed` gate. `μ⁻` is computed
once and *shared* by both seams (lazy — the fully-fixed path still does no extra
matvec), and the steady-state guard rejects a state-dependent `Q(x)` for the same
reason it rejects `R(x)`. Same independent-NumPy-oracle strategy
(`_numpy_qx_filter`: scalar + 2-D + a `μ⁻`-not-prior discriminator + a constant-Q
consistency net). The two seams are now symmetric: `R(x)` on the sensor
(`linearize`), `Q(x)` on the dynamics (`noise_at`), both at the predicted state.

### Named seams (not built here)

- **Gate harmonization.** The filter gates on `is None or is_fixed`; `efe.py`'s
  inline fast path gates on `is None` only (harmless — `FixedSensor.gaussianize`
  returns the constant `R` — but an asymmetry to reconcile later).
- The Jensen / second-order term (Phase 2.5 `NonlinearSensor`).

### Shown end-to-end — the flagship demo

`examples/bacillus_seeking_food.py` is the visual counterpart to the payoff above
(and the README hero). Four bacilli share one `CallableSensor` precision-well `R(x)`
and one `KalmanBackend` perceiving *on* that `R(x)`, differing only in how much they
value information. It renders the regimes this ADR's "honest note" already describes:
pragmatic-dominant (beelines, stays uncertain), balanced (detours to the beacon,
localises, *then* reaches the goal), and epistemic-dominant (seeks and *holds* the
beacon, never reaches the goal). The original v0.2 fixed-sensor LQR demo is kept in
the gallery (`examples/README.md`) as the before-picture.

Two framing notes, so the demo isn't mis-read back into the library:

- **The explore/exploit knob is the preference precision `Λ` — a real, public knob.**
  `expected_free_energy` is fixed at `pragmatic − epistemic` (ADR-005); there is no
  weight in it. But the pragmatic term is *linear* in `Λ` and the epistemic term is
  *independent* of it, so scaling `Λ` by `c` gives `G = c·pragmatic − epistemic`, whose
  argmin equals minimising `pragmatic − (1/c)·epistemic`. So the preference precision IS
  the explore/exploit axis: weak `Λ` ⇒ epistemic-dominant (curious), sharp `Λ` ⇒
  goal-dominant. The bacillus demo varies exactly this — each agent an `ObservationGoal`
  with a different `precision`, scored through the real kernel over its own 2-D grid
  (`EFESelector` is still p=1). (An earlier cut hand-recombined the split as
  `pragmatic − λ·epistemic`; the same knob reparameterised as `λ = 1/c`, but it read as a
  kernel weight users could not reach — so the demo now uses the precision knob directly,
  and `tests/test_efe.py` pins that precision controls the balance.)
- **One-step EFE needs one-step observability.** The demo is a single integrator
  (`μ⁺ = μ + dt·a`; the action moves the observed position *this* step). On a double
  integrator the action moves only velocity, so it does not touch the predicted
  observation for one step and the H=1 kernel goes action-flat in *both* terms — a
  concrete face of the ADR-007 myopia, and another reason the H≥2 rollout stays a
  named seam.

---

## ADR-009 — v0.3 Phase 3: the H-step rollout seam (`policy_efe`, default H=1)

**Date:** 2026-06-20
**Status:** Accepted
**Phase:** v0.3, Phase 3 (Workstream B)
**Extends:** ADR-005 (rolls out its one-step kernel); retires the myopia named in ADR-007 and in ADR-008's "one-step observability" note.

### Decision

Action selection becomes horizon-shaped, with the horizon a public knob defaulting to
1 (so existing behaviour is unchanged).

1. **`_efe_step` (Fowler Extract Function).** The predict→sense→score body of
   `expected_free_energy` moves into a private `_efe_step` returning an `_EfeStep`
   result — the public split (`g`, `pragmatic`, `epistemic`) **plus** the three
   intermediates the rollout consumes: `μ⁺`, `Σ⁺`, `S`. **No `C` is returned** — the
   rollout fetches its own `C` only where it propagates, so the one-step wrapper does
   *zero* extra work (structurally, not by trusting dead-code elimination). The wrapper
   is byte-identical to Phase 1A.

2. **`policy_efe` (the rollout).** A `lax.scan` over the policy rows, carry = the
   propagated belief `(μ, Σ)`, summing each step's `G`. Propagation is **predict-only**:
   the mean carries forward as the prediction `μ⁺` (the innovation has zero expectation
   — there is no real future observation), and the covariance contracts by the Kalman
   update `Σ_post = Σ⁺ − Σ⁺Cᵀ S⁻¹ C Σ⁺`, computed inline from the `(Σ⁺, S)` already
   returned plus the `C` fetched in the scan step (`model.C` fixed, else
   `linearize(μ⁺)[0]`) — *not* via `kalman._gain_and_posterior_cov`, which re-predicts
   and would evaluate `Q`/`R` at the wrong point. `R(x)`/`Q(x)` work for free (each step
   linearises at its own `μ⁺`). At `H=1` the rollout reduces **exactly** to
   `expected_free_energy`. The signature is `horizon`-free — `H` is `policy.shape[0]`,
   so a kwarg would be a redundant second source of truth.

3. **`EFESelector.horizon` (the public knob).** Default 1. At `H>1` the candidate family
   is **constant-action policies** (each grid action held for H steps), scored by
   `policy_efe`; `select` returns the *first* action of the best one (receding-horizon).
   Per-cycle cost stays one attributable number, `cost_per_cycle = n_candidates ·
   horizon` (RFC-001). `horizon` threads through `ObservationGoal` to the Agent-built
   selector; default 1 ⇒ no behaviour change.

### The honest caveat (load-bearing)

`horizon` selects the best *constant* action, **not** the best *sequence*. It makes
delayed consequences visible — retiring the double-integrator action-flatness — but a
genuinely sequential epistemic policy (*move to sense, then exploit*) needs a varying
sequence the constant-action family cannot express. So at `H>1` the selector can still
look myopic-ish on such tasks; it must not be over-trusted as full lookahead.
Varying-sequence / gradient action search is the deferred v0.4 `GradientEFESelector`
seam.

### Scope

`policy_efe` stays **internal** (not exported); the public surface is
`EFESelector(horizon=…)` / `ObservationGoal(horizon=…)`. Time-varying policy families,
gradient search, and energy instrumentation around the rollout are deferred.

### Validation

- **H=1 byte-identical:** `policy_efe` at H=1 equals `expected_free_energy` bit-for-bit
  (`assert_array_equal`) across fixed / `R(x)` / `Q(x)`; the `_efe_step` extraction is
  guarded by a frozen-kernel snapshot (`tests/test_efe_step.py`).
- **Independent oracle:** a plain-NumPy rollout (`tests/test_policy_efe.py`, no
  `lax.scan`, no kernel import) matches `policy_efe` to `1e-9` at H=2,3 under fixed
  sensor, `R(x)`, and `Q(x)`; `jit` / `vmap`-over-policies / `grad`-over-policy survive;
  the propagated covariance stays PSD each step.
- **The demonstration:** on a double integrator (act on velocity, observe position) the
  H=1 `G` is action-flat to machine precision while H=2 picks a sensible action matching
  the brute-force argmin (`tests/test_double_integrator_horizon.py`).
- `test_efe.py` and `test_efe_selector.py` pass **unmodified** — the seam is additive.

---

## ADR-010 — v0.3 Workstream A: declarable model structure (`ModelStructure`) + the multi-model reframing

**Date:** 2026-06-20
**Status:** Accepted
**Phase:** v0.3, Workstream A
**Extends:** RFC-003 §4.5 ("metadata version ships first"); relates to ADR-006 / RFC-001 ch. 8 (the *E. coli* internal-structure motive).

### Decision

A model may carry optional, **static** structure metadata — `ModelStructure` — that
declares its factorisation without the v0.3 engine yet exploiting it.

1. **Structure goes on the model; the Agent stays one-model.** "Multiple models" and
   "declarable dense structure" are the **same problem** — relational structure over
   variables — so v0.3 ships *one* substrate: a `ModelStructure` on the
   `LinearGaussianModel`. The array-of-models convenience and the
   hierarchical-vs-ensemble *semantics* are deferred to a v0.4 composition layer built
   on this. (The literature is genuinely open on the semantics; committing now is the
   opposite of securing the API.)

2. **Declare + inspect + validate.** `ModelStructure` carries three index groupings —
   `factors` (state indices per cause/block), `roles` (Markov-blanket typing:
   external / internal / active), `channels` (observation-row typing) — with inspection
   (`factor` / `role_of` / `channel` / `summary`) and an opt-in `validate(model)`.

3. **Rides in pytree aux_data, tuple-of-tuples.** It has no traced array leaves, so it
   is `tree_flatten` **aux**, not a child; `jit` hashes aux for its cache key, so every
   field is a tuple of tuples (a dict/list would be unhashable and break `jit`). Two
   models differing only in structure have different treedefs and recompile when swapped
   as a traced arg — correct: aux *is* static identity. Arithmetic is byte-identical with
   or without structure.

### The deliberate YAGNI break (recorded on purpose)

Shipping a structure layer the v0.3 engine does not yet exploit looks like the
speculative generality this project otherwise defers (ADR-002). It is broken
deliberately, for two reasons: **(1) secure the API early** — a structure vocabulary
added now is a pure, backward-compatible addition; added after users have models, it
churns everyone; **(2) it has a concrete near-term consumer** — Mattingly's *E. coli*
work points to an internal generative model that is **distributed and multi-variable**,
not a monolith (the same "take the internals seriously" thread as ADR-006 / RFC-001
ch. 8). v0.3 ships the vocabulary so a researcher can *express* that reading.

**Call for input.** The right factorisation of E. coli-style distributed internals is
itself open research; field experts with a better reading are invited to a pinned repo
Discussion (the structure docstring points there too).

### Sub-decisions

- **`validate()` is EXPERIMENTAL** (flagged in its docstring + the API-stability note).
  Its *partition* checks (bounds, disjointness, coverage) are durable; its
  *conditional-independence / sparsity* criterion is provisional — it checks one-step
  `A`/`Q` cross-blocks now and tightens to the rigorous precision-based (`Σ⁻¹`
  block-diagonal) test in v0.4. A model passing `validate()` in 0.3 could validate
  differently once the rigorous test lands; flagging it keeps the annotate-now benefit
  without promising a semantics we intend to tighten.
- **Strict factor/role coverage is provisional and reversible.** `validate()` currently
  requires factors and roles to *partition* the whole state (cover every index) — a
  deliberate, reversible choice, to be relaxed if it proves a faff that turns users off.
  Recorded so the reversal is a known option, not a regret.
- **API tiering.** `ModelStructure`'s data + inspection surface is stable, promised API;
  `validate()` ships experimental. `ModelStructure` is a public export (C1).

### Validation

- Pytree round-trip + `jit` survival + `__hash__` (the aux-hashability proof);
  byte-identical arithmetic with structure vs `None` (`assert_array_equal`);
  `structure=None` leaves an unchanged 8-child / `None`-aux treedef.
- Partition failures (out-of-bounds, overlap, non-coverage); a block-structured model
  honouring its declaration passes, while an off-block `A` or a cross-contaminating `C`
  fails with a message naming the offending factor pair (`tests/test_structure.py`).

## ADR-011 — Runnable doc examples (Rust-style doctests) deferred to v1.0

**Date:** 2026-06-21
**Status:** Accepted — implementation deferred to v1.0 (overkill pre-1.0)
**Phase:** post-v0.3 / v1.0 roadmap
**Relates to:** the D3 docs-accuracy pass (the current, manual guard); ADR-002 (don't
build ahead of a consumer).

### Decision

Defer Rust-style doc tests — executing the docs' fenced ` ```python ` blocks as part of
the suite so examples can't silently rot — until v1.0. Pre-1.0 it is overkill.

### Why not now

- **The public API is still moving.** Pre-1.0 a minor version may break the surface (per
  the README status note), so example code churns with it. Doc tests lock the *example
  contracts*; that only pays once the contracts are stable — i.e. at 1.0. Adding them
  earlier means re-cutting them on every API shift for little protection.
- **The cost isn't free, and pre-1.0 it isn't yet earned.** A correct setup has three
  repo-specific wrinkles: (1) the build-up tutorial's blocks share state across the file
  (needs shared-namespace execution, not block-isolation); (2) at least one block
  *intentionally* raises (the README "no objective" `sample_action()` → `ValueError`) and
  must be marked, not fail the run; (3) it wants its own pytest marker + CI step so the
  cost stays isolable. Worth it at 1.0; premature before the examples settle.
- **The gap is covered for now** by the by-hand D3 docs-accuracy pass against the source —
  adequate at pre-1.0 volume, not a standing guarantee.

### What we'll adopt at v1.0

- A fenced-block runner (`pytest-markdown-docs` for pytest-native per-block collection +
  skip/raises markers, or `mktestdocs` for a minimal `check_md_file(..., memory=True)`
  shared-namespace run). Stdlib `doctest` is a poor fit — it only reads `>>>` REPL
  examples, and the docs use script-style fenced blocks.
- **Gate it behind a pytest marker** (mirror the existing `rxinfer` marker) so it is a
  labeled, deselectable, attributable cost, not buried in the default run.
- Only ` ```python ` blocks execute; output/diagram blocks are already tagged ` ```text `
  (the markdownlint MD040 pass did this), so the runnable-vs-illustrative split is done.
- Resolve the intentional-error block (skip or assert-raises) and the tutorial's
  cross-block state (shared namespace).

## ADR-012 — v0.4: FFG message passing, canonical form, from-scratch JAX

**Date:** 2026-06-24
**Status:** Accepted
**Phase:** v0.4, Phase 0
**Extends:** ADR-004 (the JAX backend this stays inside); does not touch the v0.1-v0.3
Kalman/EFE path, which remains the chain special case (validated against it, not
replaced by it).

### Decision

v0.4 generalises the existing Kalman/EFE machinery to a Forney-style factor graph
(FFG) — variables as wires, factors as nodes — to express the E. coli chemotaxis
network, where the shared `CheA` node has edges into both a fast (CheY-P/motor)
and a slow (CheR/CheB methylation) branch and so cannot be drawn cleanly as a model
hierarchy. Four choices, settled in the build plan and recorded here as the ADR of
record:

1. **From scratch in JAX, not RxInfer.** Message passing is owned code. A Julia
   call in the inference core would break `jax.grad`/`jax.jit`/`jax.vmap` through
   the agent — the franchise property this library exists to deliver (ADR-002,
   ADR-004). Non-negotiable.
2. **RxInfer's role narrows to oracle-only.** It stays the test-time ground truth
   (the existing `rxinfer` pytest marker) plus an optional, minimal tier-4
   fallback held strictly off the differentiable hot path. Never imported by the
   core; `pip install cpomdp` stays Julia-free, continuing ADR-002's wall.
3. **Message representation is canonical/information form.** Messages carry
   `(Λ, h)` with `Λ = Σ⁻¹` (precision) and `h = Σ⁻¹μ` (precision-mean). Factor
   product is addition of `(Λ, h)`; marginalization is a Schur complement. This
   matches the information-filter algebra the Kalman backend already owns and
   avoids inversions in the product step; moment form is a readout view, not the
   storage form.
4. **The schedule is hand-authored, not reactive.** The chemotaxis graph is small
   and fixed, so v0.4 writes its message order by hand rather than building a
   general reactive/automatic-conjugacy scheduler (named out of scope below).

### Why this generalises rather than replaces

Gaussian belief propagation on a linear chain *is* the Kalman filter — the v0.4
Phase 2 keystone gate is therefore byte-identity against the existing Kalman path
on a chain topology, not mere agreement. The FFG is the more general structure;
the chain is its degenerate case, already trusted.

### Out of scope (say no on sight)

General `@model`-style frontend / arbitrary user models; a full tier-2
conjugate-exponential engine for arbitrary exponential families (the seam is
declared and stubbed, deferred to v0.5+); reactive message scheduling /
automatic conjugacy dispatch across arbitrary graphs; constrained Bethe Free
Energy as a general objective (free energy is evaluated on the fixed graph, not
minimised generally); structure *learning* (continuous coupling pruning) — v0.4
ships representation only.

### Hierarchy as a derived view

Fast/slow strata are not a primitive of the graph — they are computed from a
`CouplingGraph.levels()` projection at a τ cutoff. The graph (and its τ labels)
is stored; the hierarchy is a view recomputed from it, never the reverse. This
is what makes the shared-CheA node representable at all: a model hierarchy would
force a choice of which branch CheA "belongs to," but the factor graph just gives
it two edges.

### Validation strategy

Same discipline as the existing backends: a Kalman-path byte-identity gate on the
linear-chain case (Phase 2), an RxInfer oracle check on small graphs (behind the
`rxinfer` marker), and jit/grad/vmap smoke tests treated as gates, not
nice-to-haves, on every new public inference entry point. Full detail, phase
breakdown, and exit gates live in `.claude/cpomdp_v0.4_build_plan.md`.

### Numbering note

The v0.4 build plan originally named this "ADR-004"; that slot was already taken
by the v0.2 JAX-backend decision (above). Renumbered to ADR-012, the next free
slot — a clerical fix, not a reopened decision.

### Amendment (2026-06-26) — keystone tolerance + R(x)/Q(x) parity

Two Phase-2 clarifications, recorded as the work landed:

1. **"Byte-identity" reads as tight *numerical* identity (atol 1e-7).** The keystone
   gate runs the FFG chain in information form against the moment-form Kalman path;
   the two invert/re-invert at different points, so literal bit-for-bit agreement is
   impossible. The decision (chain == Kalman on a chain topology) stands; only the
   wording softens. The validation-strategy line above should be read this way.

2. **The FFG chain path gains R(x)/Q(x) parity before v0.4 ships.** Phase 2 ships
   fixed-matrix only — `ChainBackend` rejects a state-dependent `observation`/
   `process_noise` at construction — to keep the keystone clean. This is *not* a
   capability regression: `KalmanBackend` keeps R(x)/Q(x) on the chain throughout.
   A Phase 2.5 then lifts the restriction via the same *linearize-at-μ⁻ plug-in*
   Kalman already uses (evaluate `C, R(μ⁻)` / `Q(μ⁻)` at the predicted mean each
   step; factors go per-step on that path only, the fixed path stays front-loaded).
   This is the conjugate of the Phase-3 Gaussianization machinery and reuses it.

---

## ADR-013 — v0.4 Phase 3: the beacon's epistemic value moves from agent-state to the food latent

**Date:** 2026-06-28
**Status:** Accepted
**Phase:** v0.4, Phase 3 (build plan)
**Extends:** ADR-008 (the bacillus demo this redesigns); relies on ADR-012/Phase 2.5
(`ChainBackend` R(x)/Q(x) parity, the precondition for a meaningful Kalman-vs-FFG
comparison on this model).

### The critique

`examples/bacillus_seeking_food.py` (ADR-008) has agents detour to a beacon
because visiting it sharpens the agent's *own* position belief — `R(x)` is a
precision well keyed on the agent's own location, and the food's location is a
known, fixed `Preference` target throughout. A domain-expert critique (quoted to
me by the project owner, attributed to Conor Heins, in the spirit of "Epistemic
value and active inference" and the discrete T-Maze task) names this a *trivial*
form of state information gain: the agent gains information about itself for its
own sake, never tied to resolving a genuine *contextual* unknown — unlike the
T-Maze task, where visiting the cue resolves *which arm holds the reward*, a fact
the agent could not otherwise act correctly without. The fix has to make the
beacon's epistemic value about something the agent cannot directly act on and
does not already know — not "visiting precise states because they're precise."

### Decision

Promote the food's position to an explicit latent state. The model's state grows
from `[agent_xy]` (2-D) to `[agent_xy, food_xy]` (4-D); `food_xy` carries a wide
Gaussian prior (loosely known a priori) and a small, strictly-positive process
noise (stationary; `ChainBackend`'s information form rejects exact `Q = 0`, ADR-012
Phase 2). The sensor gains a second channel alongside the existing self-position
read: `o_disp = food_xy − agent_xy`, a relative displacement/bearing vector whose
noise is the **existing, unmodified** beacon-falloff function (`beacon_noise`),
evaluated at the agent's own position — the beacon mechanic itself does not
change, only what it is wired to reveal.

The `Preference` stays a single static object:
`Preference(goal=[*, *, 0, 0], precision=block_diag(0·I₂, Λ·I₂))` — zero weight on
the self-channel, weight `Λ` on "observe zero displacement from food" (i.e.
"stand on the food"). Because the predicted reading is
`E[food_xy]⁺ − agent_xy⁺`, this single static target *algebraically* chases the
agent's current belief about where food is — confirmed by a Jacobian check
(`∂o⁺/∂a = −B_agent`, the correct sign, no degenerate or flipped argmin): the food
block has no actuator, so the residual moves only through the agent's own
predicted position, and minimizing it is gradient ascent on a quadratic potential
peaked at the food. This reads as chemotaxis-shaped *behaviour* — climbing toward
the food — without literally simulating a concentration field (that is Phase 5's
job, and needs a real nonlinear sensor; see below).

This requires **zero changes to `src/cpomdp/`.** `LinearGaussianModel`,
`CallableSensor`, and `expected_free_energy` are already generic over
state/observation block structure: a sensor channel can read one state block
while its noise depends on a different block (`CallableSensor.noise_fn(x,
params)` already receives the *full* predicted state), and the EFE kernel's
pragmatic/epistemic terms are plain `m`-dimensional algebra that does not care how
many channels are stacked or what they're labelled. This is a model-construction
exercise (`examples/bacillus_uncertain_food.py`), not a library feature — verified
by direct reads of `efe.py`, `observation.py`, `selection.py`, `structure.py`, and
`chain.py`, cross-checked by an independent design review before implementation.

### The rejected alternative: per-step `Preference` rebuild

Keep absolute position sensing (no relative channel); rebuild
`Preference(goal=belief.mean[2:4], ...)` fresh every loop iteration from the
current food-belief mean, hand-rolled in the demo script. This is **behaviourally
equivalent** — the epistemic mechanism (a channel reading food's position, R(x)
keyed on the agent's own beacon-proximity) is identical either way, since that
part of the fix is what actually answers the critique, not the pragmatic-term
plumbing. It is arguably *more legible* to a reviewer steeped in the discrete
T-Maze framing: "the preference is fixed, belief about the unknown changes" reads
more directly as the T-Maze shape than a displacement channel's algebra.

Rejected for v0.4 because it does not scale as cleanly to multiple goal items —
each additional item needs its own per-step Python rebuild rather than one more
static `Λ_i` block in a single object — and because the relative-channel version
is *also* the more general posture (it composes with stacking more displacement
channels with no script-side bookkeeping). Recorded here so the choice is visible
and not just "the cleverer one happened to get built."

### Open: the multi-goal beacon topology (not resolved here)

Stacking `N` food blocks (`(2+2N)`-D state, one displacement channel + one `Λ_i`
weight per item) is mechanically just bigger block matrices — confirmed, no new
abstraction needed. But whether **one shared beacon reveals every item's
displacement at once, or each item needs its own (distinct) beacon**, is a real,
undecided *behavioural* design choice, not a capability gap: a shared beacon gives
no genuine "which uncertainty is worth resolving" tradeoff (visiting it resolves
everything), while per-item beacons create the actually T-Maze-flavoured problem
of choosing which cue to visit. Left open for whichever future `N > 1` demo
exercises it; do not resolve silently by whichever is easiest to wire up first.

### Staged second half: the nonlinear sensor (Phase 4/5, not this ADR)

The displacement-vector channel is a *linear* proxy for "moving up a gradient" —
true biological chemotaxis senses a *scalar* concentration via temporal sampling
(E. coli is too small to sense a spatial gradient across its body), which is
genuinely nonlinear in the state and needs `NonlinearSensor` + second-order
Gaussianization — named as a deferred seam since ADR-006 but never built. That is
real `src/cpomdp/` work, tracked separately as Phase 4/5 in `BUILD_PLAN.md`
(spec-and-tests handed over, not authored here, per the session's mentor-mode
split-by-stakes convention) and will get its own ADR once it lands, rather than
being folded into this one.

### Validation strategy

Same discipline as the existing backends: `examples/bacillus_uncertain_food.py`'s
`--scan` mode runs the identical model/seed/loop through both `KalmanBackend` and
`ChainBackend` and checks agreement to `atol=1e-7` — the same bar
`tests/test_ffg_chain.py` already holds, now exercised on a topology neither
backend's existing tests cover (a sensor channel reading one state block with
noise keyed on a different block). A test of that same topology, independent of
the example script, is recommended in `tests/test_ffg_chain.py` near
`TestChainCallableSensorParity`.

---

## ADR-014 — v0.4 scope re-anchored on FFG factorisation; later work deferred

**Date:** 2026-06-28
**Status:** Accepted
**Phase:** v0.4 (scope correction)
**Extends:** ADR-012 (restates its DOD); reclassifies ADR-013's demo (kept, but it is
not the factorisation deliverable — see below).

### The decision

v0.4's definition of done is, exactly and only: **build FFG message passing that
represents an agent with a *factorisable* (branching) model, and a demo that shows
the difference between a normal backend and the factor-graph one.** The motivating
model is ADR-012's E. coli chemotaxis network — shared `CheA` feeding a fast
(CheY-P/motor) and a slow (CheR/CheB methylation) branch — which "cannot be drawn
cleanly as a model hierarchy" and needs the factor graph's native branching.
Everything else is out of scope for v0.4 and moves to GitHub issues, with its
rationale preserved here.

### Status at the time of this ADR (honest)

The FFG **substrate** is built and trusted, but the DOD is **not yet met**:

- Done: `CanonicalGaussian` (Λ, h) messages (Phase 1); Tier-1 factor nodes +
  `ChainBackend` with the chain == Kalman keystone (Phase 2); R(x)/Q(x) parity
  (Phase 2.5).
- Not done: there is **no branching representation** anywhere in `src/cpomdp/` — no
  `CouplingGraph`/`.levels()`, no non-chain backend. A chain is the *degenerate* case
  of an FFG (it *is* the Kalman filter), so the branching structure that justifies
  the whole effort is unbuilt; a factorisable model can currently only be handled by
  flattening it into one joint Gaussian, exactly what the FFG was meant to avoid. The
  "shows the difference" demo does not exist — the only backend comparison
  (`bacillus_uncertain_food.py --scan`) shows Kalman and `ChainBackend` *agreeing* on
  a chain (identity by construction), the opposite of a difference. The RxInfer
  oracle on a small graph is still open.
- Reclassified: ADR-013's `bacillus_uncertain_food.py` is a valuable linear-Gaussian
  *epistemic-value* demo, but it exercises a chain and shows backend *agreement*, so
  it is **not** the factorisation difference demo the DOD requires. It stays as a
  journey/epistemics demo, not the v0.4 capstone.

### Findings preserved (so they are not re-derived or lost)

A session exploring "make epistemics beat LQR" produced results worth keeping even
though the work itself is deferred:

1. **Separation principle / dual control.** For linear-Gaussian systems with
   quadratic cost and *fixed* noise, the optimal controller is certainty-equivalent
   (LQR on the mean) and assigns **zero** value to information — the estimator
   covariance evolves independently of control (Bar-Shalom & Tse 1974). Already
   encoded as ADR-003 ("fixed sensor → epistemic collapses → LQR"). Only a
   state/action-dependent sensor `R(x)` (or `Q(x)`) breaks it — the *dual effect* —
   making information-seeking provably valuable. So "a single agent can only ever do
   LQR" is **false**, and false specifically because real sensing is action-dependent.
2. **One-step EFE under-credits information.** The value of information is temporal.
   The current `expected_free_energy` is greedy/one-step, so the dual-effect advantage
   shows only as a modest *precision* edge (the honest `displacement` demo), not as
   LQR failing. The dramatic T-Maze-style result needs **multi-step policy
   evaluation** (planning as inference), which also dissolves the one-step "myopic
   trap." → deferred (issue).
3. **Why discrete is clean and continuous entangles.** In the Gaussian/continuous
   formulation the pragmatic risk term `½tr(ΛΣ_o)` and the epistemic term
   `½(ln|Σ_o| − ln|R|)` share the *same* observation covariance, so a single channel
   that is both goal and information source couples them. The discrete T-Maze avoids
   this by factorisation (separate cue/reward modalities over separate hidden
   factors). This is itself an argument *for* the FFG factorisation work: native
   factored structure is the principled way to express such separations.
4. **Biology.** Epistemic foraging in a single cell is real and evolved — E. coli
   run-and-tumble is dual control via short temporal integration (methylation memory
   ~1–4 s). A *receding horizon* is biologically defensible as (a) a normative model
   whose optimum evolution compiles into a reactive policy, and (b) at *short*
   horizons, an abstraction of that memory window (cf. infotaxis, Vergassola et al.
   2007). Long deliberative horizons are cognition, not single cells.

### Deferred to post-v0.4 (now GitHub issues)

- Multi-step EFE / planning-as-inference (with a receding-horizon spike as its first
  acceptance step).
- The honest "epistemics genuinely beats LQR" demo (depends on the above).
- `NonlinearSensor` + second-order Gaussianization (was BUILD_PLAN Phase 4 — a sensor
  feature, orthogonal to the factorisation DOD).
- The nonlinear scalar-concentration chemotaxis demo (was Phase 5).

ADR-012's existing "out of scope (say no on sight)" list (general `@model` frontend,
tier-2 conjugate engine, reactive scheduling, Bethe FE, structure learning) stands
unchanged.
