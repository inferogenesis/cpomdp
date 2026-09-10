# Architecture Decisions

Decisions are append-only. Each records the choice, the evidence, and the date.

A path inside an ADR is the path as of that ADR's date. It is not a claim about the
current tree and it is not maintained against the tree. ADR-008 and ADR-013 both name
`examples/bacillus_seeking_food.py`, which v0.4.4 removed. The entries still
name it, and that is the record working rather than rotting. `CHANGELOG.md` is where a
path goes when it moves.

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

**Date:** 2026-06-19 (recovered from commit `c465c01`)
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

**Date:** 2026-06-19 (recovered from commit `a8849d8`)
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

---

## ADR-015 — FFG is a configuration-agnostic toolbox; levels() deferred

**Date:** 2026-06-29
**Status:** Accepted (the objective and the deferral); the `levels()` semantics is open
and likely a future RFC.
**Phase:** v0.4 and forward
**Extends:** ADR-012 (the FFG decision), ADR-014 (v0.4 scope)

### Objective: an agnostic toolbox, not a chemotaxis simulator

The FFG is a general toolbox for modelling *any* coupled-Gaussian structure a user
wants. The E. coli chemotaxis network (a shared node feeding a fast and a slow branch)
is only a worked example — chosen because it is a well-defined target to aim at — not
the design target. The library must support, with full and *tested* scaling:

- arbitrary numbers of parallel branches off a node,
- chains of unbounded depth,
- whichever timescale semantics a user intends (per-edge / "gate-kept"; see below),
- any valid configuration, with no example-domain vocabulary or assumptions baked into
  the core (consistent with integer-index nodes and the domain-agnostic docstring rule).

Realised scope today is linear-Gaussian *trees* (the chain is the degenerate case);
fully general loopy graphs remain out of scope per ADR-012, a separate later question.
The point of record: design and test for the general configuration and treat any one
network as a single instance of it.

### levels() (the fast/slow hierarchy projection) is deferred

ADR-012 names `CouplingGraph.levels()` as the τ-cutoff projection that derives the
fast/slow strata. It is **not built**, and is deferred past v0.4 (it is not on the
ADR-014 definition-of-done). Reasons:

- **The semantics is undecided.** Two readings of "which stratum a node is in":
  - *per-edge* — the node's own incoming edge's τ (the intrinsic coupling rate; local);
  - *path-gated* — the slowest edge between the node and the root (effective response
    latency; a node inherits a slow ancestor's gate).
- **They agree on the only model v0.4 ships.** When every node is one hop from the root
  (parallel branches), the node's incoming edge *is* the slowest edge on its path, so
  the two coincide. They diverge only for series chains two or more edges deep.
- **There the physical meaning is genuinely ambiguous.** A reaction runs as fast as its
  own chemistry regardless of neighbours (favours per-edge); a deep node's *response* to
  the root is bottlenecked by the slowest intervening step (favours path-gated). Which
  one "fast/slow stratum" should mean is not settled.
- **Freezing a guess is the costly mistake.** Changing the *returned partition* later is
  a silent behaviour break for callers — worse than a signature change. Whatever ships
  must be the believed-correct semantics, and there is not yet grounds to pick one.

So: ship the representation (τ stored on edges — done) and add the projection only when
a real multi-level consumer pins the needed semantics and number of strata.

### The crux to research first: temporal / reactive inference

The `levels()` semantics cannot be settled in the abstract because **in v0.4 τ does not
affect inference at all** — inference is exact, computed in one sweep, and τ is pure
metadata on the modelled dynamics. "Fast/slow stratum" only gains an operational meaning
once there is **temporal / reactive inference**: a scheme that schedules updates by
timescale (fast variables updated more often than slow ones). That update policy is what
makes "which stratum a node is in" a decision with consequences.

Action for a future session: **research temporal / reactive inference** (how timescale
separation drives update scheduling in message-passing / active-inference systems)
before finalising `levels()`. The result likely warrants its own **RFC**, since it
shapes a public API and touches scheduling, not just one method. Recorded here so it is
not lost; deliberately not resolved now.

### If/when levels() is built: API-shape caution

To keep the projection from ageing badly after release:

- return a **node → stratum mapping** (or a grouping that admits N bands), never a fixed
  `(fast, slow)` two-tuple — multi-timescale models must not force a breaking change;
- treat the cutoff as possibly plural (a sequence of band boundaries), not one scalar;
- if forced to choose before the research lands, default to **per-edge** — it is the
  more defensible reading of "how fast is this coupling" and matches the chemistry
  argument above.

---

## ADR-016 — v0.4 FFG: the carry-partition backend (one backend, a node partition)

**Date:** 2026-07-01
**Status:** Accepted
**Phase:** v0.4, FFG active-inference loop (issue #25)
**Extends:** ADR-012 (FFG, canonical form), ADR-002 (front-loading)
**Supersedes:** the deferred *scheduling* half of ADR-015 (`levels()` as an update
scheduler); ADR-015's representation objective still stands.

### Decision

The branching FFG becomes an `InferenceBackend` as a **recursive Gaussian filter over the
tree**, parameterised by a **partition of the node set** — a list of clusters over integer
node indices. The partition is the off-diagonal precision (Λ) block-sparsity retained
across the time boundary: within-cluster off-diagonals are kept, between-cluster
off-diagonals are zeroed at the carry.

- `[[all_nodes]]` = joint carry = **exact** (drops nothing).
- singletons `[[i], [j], …]` = per-node carry = the fully-factored approximation.
- the useful chemotaxis config lives in between.

There is no separate "joint backend" and "factored backend": one `CouplingGraphBackend`,
one `partition` argument, default = full joint, so the out-of-the-box path is exact and
keystone-green. Exactness is a *knob*, not a type.

Two orthogonal axes, and this ADR owns only the first:

- **(A) partition** — which correlations survive the carry [here];
- **(B) per-cluster tick rate** — multi-rate scheduling [deferred to v0.5, ADR-017].

The partition sets the granularity at which multi-rate would even be expressible, which is
why it lands first. It is the concrete replacement for ADR-015's deferred
`levels()`-as-scheduler: "which stratum is a node in" becomes the operational "which
cluster's correlations persist across a step" — a decision the user makes explicitly, not
one the library must guess a universal semantics for.

### Purity fixes what the belief is

`InferenceBackend` is pure (prior in, posterior out, no hidden belief state; ADR-002) and
`Agent` feeds the returned `Belief` back as next step's `prior`. So the belief flowing
through `infer_states` is the **joint over all nodes** (block-sparse per the partition),
never one node's marginal — returning a marginal would force the rest of the joint into
hidden mutable state and break the wall. A chosen node (the agent's latent of interest,
not always the root — issue #25) is exposed as a pure *slice* of the joint
(`marginal`/`readout`).

### Acceptance gate (two-tier)

1. **Exact endpoint (non-negotiable, stays green everywhere).** The joint carry (single
   cluster) and the branch-free chain hold atol 1e-7: chain vs `KalmanBackend` (the
   ADR-012 keystone, already green); the branched filter vs an independent joint-precision
   oracle, with the RxInfer tree oracle as the external cross-check.
2. **Approximate partitions (measured, not asserted).** A **severed-mass diagnostic** —
   the norm of the between-cluster Λ blocks dropped at each carry — surfaced as a
   user-visible `partition_error` per step / summarised per run. The concrete v1 pass/fail
   is that the `{fast+CheA}/{slow}` partition reproduces η and drift within the full-joint
   run's values and Mattingly's error bars.

### Consequences

- The default full-joint path is exact and keystone-green out of the box.
- The exact endpoint (Phase 1) is a *dense* joint-precision solve (ADR-017); the
  structure-exploiting cheap solve and the factored carry + diagnostic are Phase 2. The
  distribute-pass tree BP those need is not built in Phase 1 (it is not the exact recursive
  filter — ADR-017).

### Diagnostic surface — a pure surface, not a stored field (Phase 2)

The severed-mass diagnostic is exposed as a pure, traceable surface, never a mutable
per-step attribute: `_carry` returns the severed mass as a jnp scalar next to the factored
precision, `partition_error()` is the eager `float` wrapper, and `rollout` stacks the
scalar inside one `lax.scan` for the per-run profile. I rejected a stored per-step field —
it can't be read inside `scan`/`vmap`, so it can't deliver the per-run summary, and it
would hold stale / last-of-batch values under exactly the transforms this library targets.
So `infer_states` stays a pure query, which reaffirms ADR-012's jit/grad/vmap discipline.
If per-step side-effecting delivery is ever genuinely needed (e.g. live logging from inside
a compiled rollout), the sanctioned mechanism is `jax.debug.callback` / `io_callback`, not
stored state.

### Carry the covariance, and solve factored partitions by tree BP (Phase 2, refined)

Two refinements landed once I wired the cheap solve. First, the carry factors the joint
**covariance**, not the precision. Masking precision blocks and then inverting changes the
diagonal (marginal) blocks too, so it silently perturbed *this slice's* node marginals —
violating ADR-017's "within-slice marginals exact". Masking the covariance leaves the
per-cluster diagonal blocks untouched: every node's marginal stays exact, and only the
cross-cluster correlation *carried forward* is dropped. So `partition_error` is now a
covariance magnitude (correlation severed), not a precision norm, and `[[all]]` is still a
byte-identical no-op.

Second, a fully-factored (singleton) partition carries a block-diagonal belief, which makes
the within-slice problem a tree. That path runs two-pass tree belief propagation
(`CouplingGraph.infer_all`, on `GaussianCoupling.message_to_child` and
`CanonicalGaussian.__sub__`) — per-node seeds up-and-down the tree — instead of forming and
inverting the dense joint: O(tree) rather than O(n³), validated equal to the dense path at
atol 1e-7 and measured ~2× faster on an 81-node star (more as the tree grows). Multi-node
clusters still take the dense within-slice solve; a super-node BP for them is future work.

---

## ADR-017 — v0.4 FFG: temporal-edge composition, driven relaxation, single clock

**Date:** 2026-07-01
**Status:** Accepted
**Phase:** v0.4, FFG active-inference loop (issue #25)
**Extends:** ADR-012 (FFG)
**Depends on:** ADR-016 (the carry-partition backend)

### Decision: how temporal edges compose with the structural couplings

`CouplingGraph` stays **purely spatial** — one time slice. Time lives in the backend's
recursion loop, not in the graph object: this is a recursive *filter*, not an unrolled
two-slice graph (the closed-loop agent forces online filtering). Per global step:

1. **predict** — advance each node by its own dynamics, block-diagonal `F = blkdiag(A_i)`
   on the one `dt`, with control `b = B·action`;
2. **update** — fold the structural couplings and the observations into the predicted
   joint (the within-slice collect);
3. **carry** — apply factorisation only at the temporal boundary: zero between-cluster
   off-diagonal precision blocks (ADR-016). Never inside a slice.

### Driven relaxation: each node has its own dynamics *and* a structural drive

The generative model is *driven relaxation*: a non-root node evolves as
`node_i(t) = A_i·node_i(t−1) + W·parent(t) + noise` — its own temporal memory (relaxation
on timescale τ_i) plus the structural drive from its parent, applied **every slice**. This
is the E. coli physics (RFC-003 section 3.1): methylation is a slow integrator (τ₂≈9.9s)
driven by CheA, CheY-P a fast responder (τ₁≈0.05s). The structural coupling is the
parent→child input *within* the transition, not a one-off prior — so if a partition cuts a
CheA edge, that coupling is dropped at the carry and re-established under a stale prior at
the next collect (exactly the error the ADR-016 diagnostic reports).

### Consequence: the exact endpoint is a dense joint-precision solve

Because every node carries its own temporal edge, the one-step filtering posterior over the
node vector is a **dense** joint (the temporal prediction `Σ⁻ = FΣFᵀ + Q` inherits last
step's correlations; marginalising the past fills in). So the exact `[[all]]` filter *must*
carry the dense joint, and exact inference is the joint-precision assemble+solve:

    predict: μ⁻ = Fμ + b,   Σ⁻ = FΣFᵀ + Q,   Λ⁻ = (Σ⁻)⁻¹
    update:  Λ = Λ⁻ + Λ_struct + Σ_node CᵀR⁻¹C,   h = Λ⁻μ⁻ + Σ_node CᵀR⁻¹y
    read:    Σ = Λ⁻¹,   μ = Σh

`Λ_struct` is the fixed structural-coupling precision (each edge's
`[[WᵀQ⁻¹W, −WᵀQ⁻¹], [−Q⁻¹W, Q⁻¹]]`, front-loaded). This *is* the flattened Kalman with the
structural couplings as within-slice factors, so the exact endpoint matches the
hand-flattened joint oracle by construction. Two-slice tree belief-propagation is **not**
the exact recursive filter here — it represents only tree-adjacent correlations. It is
exact only for the *static* single-slice collect (today's `CouplingGraph.infer`, checked by
the RxInfer tree oracle), and becomes the cheap structure-exploiting solver for the
*factored* regime in Phase 2. Phase 1 therefore ships the dense solve; the distribute-pass
machinery arrives with partitioning.

### Q2: one global clock; multi-rate deferred to v0.5

One global `dt`; every cluster advances together each step. Multi-rate clocks (axis B of
ADR-016) are deferred. Rationale: Mattingly's information rate is the continuous-time
Gaussian-channel spectral integral — clock-agnostic, no native slow sampling — so a single
fine clock reproduces the v1 target. Fine-stepping the stiff slow mode is numerically
benign in exact Gaussian message passing (`A_i ≈ I + A·dt` is well-conditioned in float64;
no stiff-integrator instability). Multi-rate is a long-horizon performance optimisation,
not a correctness requirement.

### Q5: filter vs smoother, query contract (consequences)

The filter is required for the v1 closed loop; a smoother is optional and off the critical
path. Query contract: within any single slice, after the update, all joint marginals over
any node subset are exact regardless of partition — the partition restricts only what
persists across a time boundary. The only unavailable queries span *both* a cluster
boundary and a time boundary. v1 reads within-slice quantities plus the emitted trajectory,
so nothing it needs is obstructed.

---

## ADR-018 — v0.4 FFG: partition admissibility under EFE

**Date:** 2026-07-01
**Status:** Accepted (the constraint); the guard's wiring lands with the EFE-on-FFG work.
**Phase:** v0.4, FFG active-inference loop (issue #26)
**Extends:** ADR-016 (partitions), ADR-005 (EFE decomposition), ADR-010 (`ModelStructure`)

### Decision

Not every partition (ADR-016) is admissible for an *epistemic* agent. The EFE epistemic
term is the covariance-mediated coupling between the sensed state and the latent whose
ambiguity the agent reduces (ADR-005: `½(ln|S| − ln|R|)` with `S = CΣ⁺Cᵀ + R`; the coupling
rides in the predicted `Σ⁺`). A partition that cuts that edge sends the two independent,
zeros the epistemic term, and silently collapses the v0.3 instrumental epistemics — a
drift-speed eyeball would still pass while the epistemic behaviour is dead.

**Rule:** a partition must not sever an EFE-load-bearing edge. Guard it — warn/error when a
partition cuts an edge flagged EFE-relevant, tying to `ModelStructure` metadata (ADR-010's
`factors`/`roles`/`channels`) where declared, or a per-edge flag on `Coupling` where not.

### Chemotaxis specifics (checked, not assumed)

The natural cut is safe on the physics: gradient information rides receptor→CheA→CheY-P
(fast); methylation is adaptation, high-passing the DC and carrying no gradient-direction
information, so cutting `{slow methylation}` off does not touch the epistemics. This must be
verified with the ADR-016 severed-mass diagnostic on the EFE-relevant edges — a partition
chosen for cost that happens to cut receptor↔food-latent would pass a drift-speed eyeball
while quietly killing the epistemic term.

### Status

Recorded now because it shapes the partition API from the start. The concrete guard wiring
lands with the factored-EFE work (issue #26); Phase 1's exact `[[all]]` endpoint severs
nothing, so it is trivially admissible.

## ADR-019 — v0.4 FFG: state-dependent sensing R(x) linearized at the predictive prior μ⁺

**Date:** 2026-07-05
**Status:** Accepted
**Phase:** v0.4, closed-loop active inference on the FFG (issue #27)
**Extends:** ADR-003 (whose fixed-sensor EFE collapse this breaks), ADR-005 (the EFE kernel
that reads R), ADR-016/017 (the carry-partition backend R(x) threads through)

### Decision

A `CouplingGraphBackend` observation node may carry a state-dependent sensor,
`CallableGaussianObservation` with `noise_fn(x, params) -> R(x)` — the FFG twin of the flat
`CallableSensor`. This is the dual effect (ADR-014 finding #1): with R depending on the
predicted state (and so on the action), the epistemic term is no longer action-invariant,
so the chosen action can seek states where the sensor is sharper. Without it the FFG EFE
collapses to LQR (ADR-003).

**Linearization point — the predictive prior μ⁺, not the temporal predict μ⁻.** R(x) is
evaluated at μ⁺ = `predicted_belief().mean` (temporal predict **plus** the structural
couplings), the pre-observation joint the epistemic's Σ⁺ already uses. The structural
couplings are part of the generative prior over the slice, so μ⁺ is the mean of the
predictive q(x) — and R's EFE term is an expectation under exactly that q(x). R and Σ⁺
share one linearization point, the FFG analogue of the flat path taking both at μ⁻.

**Two passes, exact, not iterated.** On the factored (singleton-partition) path the
per-node predicted seeds do not yet carry μ⁺ (couplings only resolve inside BP), so R(x)
takes one extra pass: run `infer_all` on the *R-free* predicted seeds to read each node's
μ⁺ marginal, evaluate R(μ⁺) there, then seed and BP again. Because pass 1 carries no
likelihood, every R(μⱼ⁺) is linearized at the R-free coupling-resolved prior — non-circular
by construction, and *exact in exactly two passes*, not a truncated fixed point. Pass 2's
marginals move (R_i propagates through the couplings to j) but the linearization points stay
frozen at pass-1 μ⁺. This is the EKF "linearize at the prior mean, then update" structure
lifted onto the tree. Iterating would silently switch to *posterior* linearization — a
different estimator — and reintroduce the circularity; do not add an iterate-to-stable loop
unless deliberately changing the estimator. The dense `[[all]]` path gets μ⁺ for free from
the joint predicted precision (one extra `to_moment`), so it needs no second BP.

**Abstraction: linearization-point-dependence, not "R".** The pre-pass fires for any factor
whose linearization point is a coupling-resolved marginal — a state-dependent R(x) today, a
nonlinear sensor C(x) later. The fixed-sensor exemption is the *constant Jacobian* (μ⁻ and
μ⁺ linearizations coincide), not the sensor-ness; `is_fixed` marks that. A future
`NonlinearSensor` slots into the identical pre-pass with no rework.

**Gradients: μ⁺ is differentiated through, not `stop_gradient`'d — consistently in both
paths.** Rationale: the dense and factored paths must agree in value *and* derivative; for
sensor learning (∂/∂params) μ⁺ is independent of params anyway; the planner gradient
(∂/∂action) case belongs to multi-step differentiable planning (issue #20), where the choice
is revisited. A future iterated/posterior scheme should reconsider this deliberately.

**`to_flat_model` on a coupled R(x) is a category error, encoded as a typed exception.** The
flattened-Kalman oracle route encodes couplings as pseudo-observations, so the flat Kalman
linearizes R at μ⁻ (pre-coupling); with a mean-shifting coupling μ⁻ ≠ μ⁺, and Σ⁺ =
(Σ⁻⁻¹ + Λ_struct)⁻¹ is not `A'ΣA'ᵀ + Q'` for any constant A',Q', so no fixed flat model
reproduces the filter. `to_flat_model` raises `IncompatibleLinearizationError(ValueError)`
— a theorem, not a TODO — signposting the real R(x) oracle (the standalone NumPy R(μ⁺)
filter plus the single-node `CallableSensor` cross-check). The guard fires on
*nonlinear-R ∧ mean-shifting-coupling*, not on "has R": a coupling-free R(x) has μ⁻ = μ⁺ and
*would* be a faithful oracle via a `CallableSensor` flat model, so it raises the honest
`NotImplementedError` ("could be done, unbuilt") instead — a different truth, a different
type.

### Validation

- Reduction: a constant `noise_fn` reproduces the fixed `GaussianObservation` message and
  the fixed backend step-for-step (dense and factored) — the affine no-op guarantee, so the
  extra pass is free when R does not vary.
- Single-node R(x) matches the flat `KalmanBackend(CallableSensor)` at atol 1e-7 (couplings
  absent, so μ⁺ = μ⁻ and the trusted flat path is a valid oracle).
- Branching R(x) (dense and factored one-step) matches an independent NumPy R(μ⁺) filter at
  atol 1e-7 — couplings and the μ⁺ linearization together.
- `predicted_belief` is R-independent (R never enters the predict), so it rides `vmap` over
  a candidate-action grid — the Phase-3 selector seam.
- `to_flat_model` raises `IncompatibleLinearizationError` on a coupled R(x) backend.

### Consequences / deferred

- The extra μ⁺ pass (factored) / extra `to_moment` (dense) is the attributable dual-effect
  cost (RFC-001) — paid only on the state-dependent path, the fixed hot path stays
  byte-identical. Caching the shared pass-1 μ⁺ across policy prefixes in a rollout is a
  deferred throughput lever, not built.
- The Koudahl–Kouw–de Vries epistemic-drift-magnitude regression against the RxInfer oracle
  is the Phase-3/4 gate (it needs the EFE term wired to the selector); the filter-level R(μ⁺)
  oracle match above is what proves the linearization fix here.

## ADR-020 — v0.4 FFG: Mattingly's η does not fall out of cpomdp; the certified-regret reframe

**Date:** 2026-07-05
**Status:** Accepted
**Phase:** v0.4, closed-loop active inference on the FFG (issue #27) — scope/honesty
**Extends:** RFC-001 (corrects its §8 "upper bound" framing), ADR-005 (the EFE terms),
ADR-003/ADR-014 (the dual effect), ADR-019 (R(x))

### The question

Does *E. coli* chemotaxis efficiency η = 0.65 ± 0.05 (Mattingly, Kamino, Machta, Emonet,
Nat. Phys. 2021, "E. coli chemotaxis is information limited") fall faithfully out of
cpomdp with honest AIF math? Verified adversarially — four independent analyses plus a
red-team, checked against source and the paper's exact equations.

### Decision (the verdict)

**No — FALSE as a derivation, MISLEADING if the EFE epistemic term is narrated as an
information rate.** η is a *behavioural* efficiency: achieved swimming drift over a bound

    v_d / v_0  ≤  f(θ) · sqrt( ln2 · İ_{s→a} / (12 · D_r) )

built from swimming physics (v_0 = 22.6 µm/s, rotational diffusion D_r, behavioural
function f(θ) = 0.531) and two *directed* trajectory information rates (İ_{s→a}, signal →
kinase activity; İ_{s→m*}, relevant-and-communicated-to-motor; η² = İ_{s→m*}/İ_{s→a} ≈
0.42). cpomdp as built has none of the behavioural quantities and neither rate. The
structural gaps, each verified against source:

- No spatial state / swimming / drift `v_d` — no y-coordinate exists to plot.
- The EFE epistemic term ½(ln det S − ln det R) = I(latent; obs) is a **symmetric,
  per-step, nats** mutual information on the pair (latent, sensor obs). Mattingly's
  İ_{s→a} is a **directed, continuous-time, bits/s** rate on (external signal, kinase
  activity) — a different object, not merely different units. No data-processing chain
  links the two variable pairs and the units differ, so the epistemic term is **not** a
  rigorous upper bound on İ_{s→a} (this corrects RFC-001 §8); a "bound" whose value scales
  with the discretisation `dt` is not a bound.
- No f(θ), D_r, v_0; no length scale, no gradient `g` — the curve's constants and the
  İ ∝ g² dependence are absent; the epistemic magnitude is set by placeholder OU knobs.
- `CouplingGraph` is a rooted **tree**: it structurally cannot represent the CheB →
  receptor demethylation **feedback loop** that Mattingly identifies as the *cause* of
  η < 1. The example model copies node labels and the τ₂ = 9.9 s scalar, not the feedback
  that matters; CheB is a gradient-blind dead-end leaf.
- Observed/hidden roles are inverted vs the experiment (kinase activity is *measured* by
  FRET there; the CheA analogue is *hidden* at the root here).

The single defensible sentence (everything past it overclaims): *cpomdp computes, in
closed form for the linear-Gaussian shallow-gradient regime — which is Mattingly's regime
— the per-step perceptual information gain I(target latent; observation) in nats, including
about a hidden hub inferred through its children, on an OU graph that copies chemotaxis
node labels and one timescale but not its feedback topology.* Drop "upper bound", "bits/s
rate", and "topology matches".

### The reframe (what IS faithful)

The honest quantity is not "İ off the EFE" but a **certified regret** against a provably-
optimal reference computed from the true world p*(which the experimenter owns in a
built world): F_rel = F_achieved − F_optimal ≥ 0, with η the normalised readout. In the
linear-Gaussian regime the Kalman/LQG solution is that certified optimum — the boundary-
theorem framing: a certified regret exists iff a provably-optimal reference exists. The
faithful *curve* — the in-regime analogue of Mattingly's √-law ceiling — is the **LQG
sequential-rate-distortion bound**: control cost lower-bounded by the *directed*
information rate through the sensing channel (Tatikonda–Mitter; Nair–Evans;
Silva–Derpich–Østergaard). A cpomdp agent could be placed under *that* curve at efficiency
η_LG, reproducing Mattingly's *structure* (information-limited, near-optimal) in cpomdp's
native regime with no swimming — but it still needs (a) a directed-information bits/s rate
(not the symmetric per-step nat-gain), (b) the derived bound, (c) an LQ performance metric,
(d) F_optimal from p*.

### The boundary connection

Koudahl–Kouw–de Vries (Entropy 2021): in **fixed-noise** LGSSM the EFE reduces to KL
control plus a constant and the epistemic term does not drive action — the certifiable
regime. The #27 state-dependent R(x) work is the deliberate minimal departure
(Bar-Shalom–Tse dual effect; certainty-equivalence fails): it revives the epistemic term
precisely where the LQG certification gives way. #27 is thus the *coded boundary crossing*,
not a Mattingly reproduction. (KKdV's "under any circumstances" is scoped to their fixed-
covariance model class; R(x) is outside it by construction — worth verifying against KKdV
directly when the paper leans on it.)

### Consequences

- The *E. coli* / Mattingly reproduction is a model built **on top of** cpomdp — a swimming
  simulator (RFC-002) + a directed-information rate estimator + the relevance/policy
  projection + M26's internal-noise `Q(µ⁺)` placement — not a cpomdp-core feature. It is
  **paused, not abandoned**: AIF is universal, so the witness is buildable atop the
  toolbox; it is simply not a v0.4 deliverable and its numbers must never be implied by the
  current example model.
- Any demo on the chemotaxis-shaped FFG is *illustrative of instrumental epistemics*, not a
  biophysical chemotaxis model; it must not display or imply η, β, or the drift bound.
- The certified-regret credibility suite (F_rel on known-p* worlds) is the honest north-star
  flagship direction; deferred beyond the small-work v0.4 close.

## ADR-021 — v0.4 FFG: the public export surface (top-level construction API, internal selector)

**Date:** 2026-07-06
**Status:** Accepted
**Phase:** v0.4, closed-loop active inference on the FFG (issue #27) — API/release
**Extends:** ADR-002 (the construction/loop split), ADR-019 (R(x)), ADR-020 (the boundary
framing); gates the v0.4 release and the dissociation flagship

### The question

The branching-FFG machinery (issues #25–#27) shipped behind deep module paths: a user
building the flagship had to reach into `cpomdp.ffg.graph`, `cpomdp.backends.coupling`, and `cpomdp.ffg.factors.linear_gaussian`.

### Decision

Promote the model-*construction* symbols to the top-level `cpomdp` namespace, and only
those. A user declares and runs a branching model entirely off `from cpomdp import …`:

- `CouplingGraph`, `Coupling` — the graph and its edges.
- `GaussianObservation`, `CallableGaussianObservation`, `GaussianCoupling`,
  `GaussianTransition` — the factors an edge/node carries.
- `CouplingGraphBackend`, `IncompatibleLinearizationError` — the engine and the error it
  raises when a coupled `R(x)` model is asked to flatten.

The **selector family** is also uniform public: `ActionSelector` (the protocol),
`EFESelector`, `FfgEfeSelector`, and `LQRSelector` all sit in `cpomdp.__all__`. I first
kept `FfgEfeSelector` internal because the `Agent` auto-dispatches it, but that split it
from its flat twin `EFESelector`, which was already exported — an odd seam to expose from
one side only. So `FfgEfeSelector` now carries the same introspection surface as
`EFESelector` (`n_candidates`, `horizon`, `cost_per_cycle`) and the same top-level
visibility. Pass `selector=` to `Agent` to override the dispatched one.

The canonical *homes* are unchanged: the symbols still live in `cpomdp.ffg.*` and
`cpomdp.backends.coupling`, and those paths keep working. The top-level names are the
documented, stable entry point; the deep paths are where the code lives, not where callers
are asked to look. `cpomdp.ffg` / `cpomdp.ffg.factors` / `cpomdp.backends` `__init__`
modules stay docstring-only (no re-export layer to keep in sync).

### Consequences

- `cpomdp.__all__` grows by the branching-construction symbols and the selector family; the
  import block sits in the lower, `agent`-free layer, so there is no import cycle (the FFG
  modules never import the top package or `cpomdp.agent`).
- `examples/ffg/epistemic_dissociation_figure.py` builds off a single `from cpomdp import (…)`
  block — the reproducible snippet the release and the docs quote.
- The selectors now read as one family: same `select(belief, preference)` contract, same
  `n_candidates`/`horizon`/`cost_per_cycle` introspection, same top-level visibility.

*Can refine in future.*

## ADR-030 — the enumeration completeness certificate

**Date:** 2026-07-30
**Status:** Accepted
**Phase:** v0.4.4 preliminary (multi-step EFE, horizon > 1) — item A
**Extends:** ADR-002 (construction/loop split); realises the warrant-vocabulary standing
rule for the enumerated search

### The question

A finite search over `|A|^H` policies wants to state "no policy in this set flips the
crossover sign". That is a universal, and a universal is only *decided* (Prover 3b) if
the search was exhaustive. Without evidence of exhaustiveness the same run is a *sample*
(Prover 3a) wearing a decision's clothes — and the crossover result would inherit a 3a
licence it must not have.

### Decision

`EnumeratedEfeSearch` emits a `CompletenessCertificate` at construction: `expected =
|A|^H` computed independently from the action-set size and the horizon, `visited` = the
count of policies actually enumerated, and it raises `IncompleteEnumerationError` if they
differ. The certificate carries the `PROVED` warrant and prints in that vocabulary
(`PROVED (finite set, |A|^H = N, visited N)`), never a bare `PASS`.

This lands now rather than at v0.6. It is about ten lines, and deferring it risks the
crossover being computed under a 3a licence and needing a re-run once the certificate
finally arrives.

### Consequences

- The enumerated family can state `PROVED` honestly; a later change that samples,
  deduplicates, or prunes the enumeration trips the assertion instead of silently
  downgrading the warrant.
- The certificate is scoped to the *declared* set. Whether the set is fine enough is a
  separate question — a refinement-stability check, pre-registered as a falsifier — not
  something this certificate can or should settle.

## ADR-031 — the search-family seam

**Date:** 2026-07-30
**Status:** Accepted
**Phase:** v0.4.4 preliminary (multi-step EFE, horizon > 1) — item A
**Extends:** ADR-030 (the certificate); relates to ADR-021 (the selector family)

### The question

`EFESelector` searches a *continuous* action box by sampling an evenly-spaced grid and
tiling each sample into a constant-action policy — a sample of a continuum, warrant
`CORROBORATED`. The crossover needs a *decisive* search over a finite declared set,
warrant `PROVED`. These are different objects with different warrants. If the API lets
them be confused, a sampled result reads as a decided one and the certificate ADR-030
earns is eroded at exactly the point it matters.

### Decision

Keep them physically and typographically distinct:

- A separate module, `cpomdp.enumeration`, holds the enumerated family — an internal
  seam, not re-exported at the top level (a public surface is scheduled, not assumed).
- `FiniteActionSet` (declared, finite, **versioned**) is a distinct type from the grid's
  `(lo, hi, n_candidates)` config, so the two cannot be passed to the wrong search.
- Two families with two warrants, both self-describing via a `SearchWarrant`-valued
  `.warrant`: `EFESelector` (grid, `CORROBORATED`, constant-action, `p = 1`) and
  `EnumeratedEfeSearch` (finite set, `PROVED`, varying `A^H`, `p >= 1`).
- Two cost vocabularies: `n_candidates * horizon` (grid) versus `|A|^H * H`
  (enumerated) — the honest exponential cost of an exhaustive search.
- The action set is versioned so an action added after results are seen shows up in the
  diff, not in a reviewer's objection.

### Consequences

- The two warrants cannot leak into each other; a continuous-action search
  (`GradientEfeSelector`, still 3a) stays out of the enumerated evidence by construction.
- `EnumeratedEfeSearch` supports `p >= 1` and *varying* sequences, so it expresses a
  detour-then-exploit policy the constant-action grid cannot — the capability the
  crossover model actually needs.
- `selection` gains a one-way import of `SearchWarrant` from `enumeration`; the reverse
  edge stays type-only (`Preference` under `TYPE_CHECKING`), so there is no cycle.

## ADR-032 — the multi-step FFG rollout (node-targeted EFE over a horizon)

**Date:** 2026-07-30
**Status:** Accepted
**Phase:** v0.4.4 preliminary (multi-step EFE, horizon > 1) — the crossover statistic's
prerequisite
**Extends:** ADR-019 (R(x) linearized at μ⁺), issue #26 (`_ffg_efe_step`, node-targeted
epistemic); mirrors the flat rollout `policy_efe` / `policy_efe_trace`

### The question

The crossover the horizon-sweep measures is a fact about the coupled T-maze: its anchors
are the node-restricted epistemic (info gain about the CONTEXT marginal, ~1.72 nats) read
at the coupling-resolved predictive mean μ⁺. That quantity lives only on the FFG path —
`_ffg_efe_step` fed by `CouplingGraphBackend.predicted_belief` — and that path was
single-step. `policy_efe` is a genuine H-step rollout, but on a flat `LinearGaussianModel`
its epistemic is the *whole-state* observation-space term (the same contrast reads ~2.42
nats), and a flat corridor has no context node to sense at all. So the crossover cannot be
measured on the flat rollout; a multi-step FFG rollout is the missing piece.

The fork it opens: how to propagate the joint belief between steps, and how much to reuse
versus reinvent.

### Decision

Add `policy_efe_ffg` (summed) and `policy_efe_ffg_trace` (per-step) to `cpomdp.efe`,
sharing one `lax.scan` step `_ffg_rollout_body`, exactly mirroring the flat trio so the
two rollouts read as parallel.

- **Propagation is predict-then-contract on the coupled joint.** Each step:
  `backend.predicted_belief` gives μ⁺/Σ⁺ with the couplings folded in;
  `backend.observation_noise_at(μ⁺)` gives R(μ⁺); `_ffg_efe_step` scores the
  node-targeted EFE; the carry contracts the *joint* covariance by the Kalman update at
  `(C, R(μ⁺))` with the mean held at μ⁺ (zero expected innovation). This is the flat
  `_rollout_body` with three swaps — flat predict → `predicted_belief`, `_efe_step` →
  `_ffg_efe_step`, flat C/R → the backend's.
- **Reuse `PolicyEfeTrace` unchanged.** Same seven fields; the only semantic shift is that
  the `epistemic` column is node-restricted, not whole-state.
- **The backend is duck-typed.** `EfeBackend` is imported under `TYPE_CHECKING` only
  (`backends` does not import `efe`, so no cycle); the backend is not a JAX pytree, so it
  is held as a closure constant, and the functions compose under `jit`/`vmap`/`grad` over
  the array arguments — the way `FfgEfeSelector` already `vmap`s its per-candidate score.
- **Two oracle reductions gate it.** H=1 is byte-identical to a single `_ffg_efe_step`
  (holds *with* couplings); a single-node backend with the whole-state target agrees with
  `policy_efe` to `allclose` at H=2,3 (numerical, not byte-identical: the FFG predicts
  through the precision form, the flat rollout through `AΣAᵀ + Q`).
- **The hot path stays lean.** `policy_efe_ffg` keeps only the three scalars in the scan's
  `ys`, so a future H-step selector's `vmap` never stacks the H×n×n covariances (RFC-001).
- Internal seam: in `efe.__all__`, not re-exported at the top level.

### Consequences

- The deferred H-step `FfgEfeSelector` (today `horizon = 1`) now has its rollout: the lean
  `policy_efe_ffg` is the object it will search over. Wiring it is a later step, out of
  this ADR's scope.
- Cost is `|A|^H` step-evaluations, and under R(x) the planning covariance is
  policy-dependent (Theorem 1(i)), so per-branch covariance trajectories are mandatory —
  there is no single precomputed trajectory to share. This is the rollout's own labeled
  work, attributable rather than buried in the filter loop; pruning is deferred and fine
  at H ≤ 3 with a small action set.
- The crossover statistic reads this trace's node-restricted `epistemic` column and must
  reduce to the anchors at H=1. The whole-state 2.42 is the flat path's number, not this
  one; the two paths are kept distinct on purpose.
- One deliberate redundancy: `_ffg_efe_step` returns only `(G, parts)`, not its `S` or the
  posterior it forms internally, so the body recomputes `S = CΣ⁺Cᵀ + R` for the
  contraction. Chosen over widening the single-step API that the selector and this scan
  both depend on.

## ADR-033 — the crossover statistic (the horizon aggregation)

**Date:** 2026-07-31
**Status:** Accepted
**Phase:** v0.4.4 preliminary (multi-step EFE, horizon > 1) — item M5
**Extends:** ADR-032 (the FFG rollout it reads); realises the standing-rule
pre-registration for the crossover result

### The question

Paper 1 defines the per-step epistemic value ε_k and, in its own words, leaves the
horizon aggregation to future work. So the H > 1 statistic is genuinely undefined, and the
only constraint on it is that it collapse to the H = 1 anchors (the epistemic pull 1.72
nats, the pragmatic gradient 4.49 nats). A plausible alternative — sum ε across the
horizon, against the change in the pragmatic term from step 0 to H−1 — fails twice: at
H = 1 the pragmatic span is empty, so it reads 0 rather than 4.49, and it is a
within-policy temporal difference, a different object from a between-policy contrast. The
aggregation has to be chosen, and chosen *before* the sweep measures H*, or the headline
crossover inherits a fitted statistic.

### Decision

The statistic is the symmetric between-policy contrast, summed over the horizon:

    Δε(H) = Σ_k [ε_k(walk) − ε_k(reach)]        the accumulated epistemic pull
    Δc(H) = Σ_k [c_k(walk) − c_k(reach)]        the accumulated pragmatic gradient
    ΔG(H) = Δc(H) − Δε(H) = G(walk) − G(reach)
    H*    = min{H : ΔG(H) < 0}

- Both sides contrast the *same* two policies. That symmetry is what makes it collapse:
  at H = 1, `Δε = 1.7232`, `Δc = 4.4910`, `ΔG = +2.7678` (measured, `tests/test_crossover.py`).
- The epistemic is read from the FFG rollout (ADR-032) at the node-restricted CONTEXT
  target, so under R(x) it rides the coupling-resolved planning covariance. The whole-state
  target reads 2.4166 for the same contrast; the two are kept distinct, and the
  node-restricted number is the headline.
- `ΔG` is *defined* as `Δc − Δε` (which equals `G(walk) − G(reach)`), so its sign flip is
  exactly the argmin flip — the horizon at which the planner's chosen policy changes. The
  code asserts this identity at tolerance 0.
- `reach` / `walk` are constant-action policies over declared members of a versioned
  `FiniteActionSet`: `a_myopic = argmin G` (prior-ward), `a_sense = argmax ε` (cue-ward). A
  two-phase walk would beat the constant one but breaks the H = 1 collapse, so it is a
  separate labelled variant, never the registered pair.
- Lives in `cpomdp.crossover` (internal seam, not re-exported). `crossover_horizon`
  *defines* H* but does not *measure* it — the H-sweep harness (M7) does, with the cost
  and conditioning table.

### Consequences

- Pre-registered: the statistic, the sign convention, the anchor magnitudes, and the
  reach/walk pair are fixed in code and in `warrant_numbers.md` before M7 runs. The commit
  that introduces them timestamps the pre-registration; its id is recorded in the ledger.
- "No crossover at any feasible H" is an explicit outcome — `crossover_horizon` returns
  `None`, not a laundered number, so that D3 falsifier stays visible to the harness.
- The tie caveat (two policies can differ per step yet tie on the horizon sum) is a
  property of the registered pair. Asserting `|Δε(H)|` stays bounded from zero across the
  sweep is registered for M7, not asserted here.
- The statistic reads only the summed rollout components, so it is cheap; the cost driver
  is the rollouts themselves (ADR-032, `|A|^H`).

## ADR-034 — the enumerated-search scoring seam (flat vs FFG)

**Date:** 2026-07-31
**Status:** Accepted
**Phase:** v0.4.4 preliminary (multi-step EFE, horizon > 1) — the exhaustive-sweep
prerequisite
**Extends:** ADR-031 (the search-family seam); ADR-032 (the FFG rollout it scores with)

### The question

`EnumeratedEfeSearch` (ADR-031) scores every `A^H` policy with `policy_efe` on a flat
`LinearGaussianModel`. The crossover model is the two-node coupled tree, whose headline is
the node-restricted epistemic at the coupling-resolved μ⁺ — `policy_efe_ffg`, not
`policy_efe`. So the exhaustive sweep that finds H\* (the horizon at which the argmin over
`A^H` becomes a two-phase walk) cannot run on the crossover model without a way to score
enumerated policies on an FFG backend. Three routes: a parallel `FfgEnumeratedEfeSearch`
(duplicates the enumeration and the certificate), a model-vs-backend branch inside the
class (branchy, violates Open-Closed), or inject the scoring as a strategy.

### Decision

Inject scoring as a strategy. `EnumeratedEfeSearch` owns the enumeration, the completeness
certificate, the argmin, and the cost — all model-independent. *How* a policy is scored is
a `_PolicyScorer` (internal):

- `_FlatScorer(model)` scores with `policy_efe`.
- `_FfgScorer(backend, target)` scores with `policy_efe_ffg`, epistemic aimed at `target`.

The default constructor `EnumeratedEfeSearch(model, action_set, *, horizon)` is unchanged —
it builds a `_FlatScorer`, so every M3/M4 test keeps passing untouched. The FFG path is a
classmethod `over_backend(backend, action_set, *, target, horizon)` building a `_FfgScorer`;
both funnel through one `_setup` (validation, `A^H` front-load, certificate). `evaluate`
`vmap`s `scorer.score`, the scorer holding the model or the backend as a closure constant.

The reduce-to-flat oracle gates it: on a coupling-free single-node backend with the
whole-state target, `over_backend` reproduces the flat search — same argmin policy, same
`G` vector at `allclose` — under both a fixed sensor and `R(x)`.

### Consequences

- The exhaustive `A^H` sweep now runs on the coupled-tree crossover model with the same
  `PROVED` completeness certificate — the prerequisite for M7b's H\*.
- Open-Closed: a further scorer (a new backend, a different objective) needs no change to
  the enumeration or the certificate.
- The receding-horizon and open-loop drivers wrap the search and call `.evaluate`, so they
  now work over an FFG-backed search unchanged. That opens M8's deferred receding-horizon
  FFG selector for free — but it stays deferred; M7b drives the enumeration open-loop.
- Cost is `|A|^H` FFG rollouts, each an H-step `policy_efe_ffg`; under R(x) the per-branch
  covariance is mandatory (ADR-032). Same `|A|^H · H` accounting, and the certificate keeps
  it honest.

---

## ADR-029 — three-valued check outcomes: a falsifier that cannot fire is not a pass

**Date:** 2026-08-01
**Status:** Accepted
**Phase:** v0.4.4 preliminary (multi-step EFE, horizon > 1) — the reporting rule M7 consumed
**Extends:** ADR-031 (its two warrant vocabularies, applied one level down at the check)

**On the number and the date.** 029 was reserved for this decision by the preliminary build
plan, which also said it had to exist before M7 ran. It did not. M7 ran, R10 was measured,
and the write-up reached for this vocabulary anyway with no ADR behind it. I have kept the
reserved number and appended the entry here, out of sequence, because this file is
append-only. Arriving late is part of what it records.

### The question

A registered falsifier has three outcomes, not two.

A suite that reports `PASS` and `FAIL` can only say that a check ran and did not fail. R10
registered four D3 falsifiers. One of them, "not reproducible across seeds", is void by
construction: the open-loop crossover object carries no observation draw, so the enumeration
recomputes identically and that falsifier can never fire. Report it as `PASS` next to the
other three and the line reads as four falsifiers surviving a test. Three survived a test.
The fourth was not a test.

That is the same erosion ADR-031 stops for warrants, one level down. There the risk is a
sampled continuum printing as `PASS` beside a decided finite set. Here it is a check that
cannot fail printing as `PASS` beside checks that could have.

### Decision

A registered-falsifier report carries one of three outcomes, in this vocabulary:

- `NOT TRIGGERED` — the check ran, the condition did not obtain, the claim survives it.
- `FIRED` — the condition obtained. The claim is refuted, and the refutation is the result.
- `NOT APPLICABLE` — void by construction. It cannot fire here, so it is evidence for
  nothing, and it is not counted among the surviving falsifiers.

Two rules travel with the vocabulary.

First, a falsifier that a given gate does not run is reported as **not run here**, naming
where it was measured instead. It never disappears into the gate's summary line. A cheap CI
gate skipping a heavy enumeration is fine. Quoting that enumeration's outcome as though the
gate had checked it is not.

Second, outcomes print per falsifier. One trailing `PASS` covering a set of them is exactly
the reading this decision removes.

This binds falsifier reporting, not ordinary assertions. "Does the shipped number match the
NumPy oracle" stays two-valued, because it passes or it raises. Three values are for a check
whose job is to try to refute a registered claim. That is where "did not apply" is a real and
distinct answer.

### Consequences

- `examples/ffg/crossover.py --check` prints the four D3 falsifiers by name with an outcome
  each. Two are `NOT TRIGGERED` on the assertions above them, one is `NOT APPLICABLE`, and
  the refinement-stability falsifier says it was not run in the gate and points at the
  write-up. The blanket `PASS` line is gone.
- The count a reader needs is now legible on the gate's own output. R10 registered four
  falsifiers, three of which were tests, and none fired.
- The vocabulary lives in the harness and the write-up, never in `src/cpomdp`. Which claims
  are registered falsifiers belongs to the research programme, not to the library, and the
  library has no use for the enum. Shipping one would be the scope drift the checklist
  exists to catch.
- Other examples keep their `PASS` lines. They gate results rather than registered
  falsifiers, and two values are the honest report there.
- The cost of the rule is a print. If a later harness wants the outcomes machine-readable,
  that is a small addition to the harness and still not a library type.

## ADR-035 — the warrant vocabulary ships, and a falsifier still does not pass

**Date:** 2026-08-05
**Status:** Accepted
**Phase:** v0.4.5 (certifiable active inference, Paper 2 groundwork)
**Reverses:** ADR-029 on where the vocabulary lives
**Extends:** ADR-029 on what it says; ADR-030 (the completeness certificate); ADR-031
(the two warrant vocabularies)

### The question

Two decisions had drifted apart. ADR-031 gave *searches* a warrant, `PROVED` for an
exhaustive enumeration and `CORROBORATED` for a grid sample, and shipped it as
`SearchWarrant` in `cpomdp.enumeration`. ADR-029 gave *checks* a three-valued outcome and
put it deliberately in the harness, ruling that "the vocabulary lives in the harness and
the write-up, never in `src/cpomdp`" because which claims are registered falsifiers belongs
to the research programme.

Three things then arrived that neither covers. Validated numerics over a compact domain
prove a universal by construction, which is stronger than a sample and weaker than a
decision, and the two-level enum has no word for it. Reference filters would need that word
in `src`. And a check needs to report warrant, outcome and tier together, which means one
type carrying all three, which means deciding where that type lives.

### Decision

**The vocabulary ships.** `cpomdp.warrant` holds `Warrant`, `Outcome`, `Tier`,
`CheckReport` and `check_summary`, all public and documented. This reverses ADR-029's
placement rule. The reason that rule gave has weakened: `EFESelector.warrant` was already a
public property returning a type nobody could import, and the warrant is no longer only a
research-programme concern once a library object reports one. ADR-029 was protecting
against the library acquiring opinions about which claims are worth falsifying. That is
untouched. `cpomdp.warrant` classifies evidence and names no claim.

`Warrant` gains `CERTIFIED` for Prover 3c. `SearchWarrant` becomes an alias of `Warrant`,
so every call site keeps its members and its return type. The promotion is not a rename.

**A falsifier does not pass.** ADR-029's outcome vocabulary was right and is kept. The
`BUILD_PLAN` primer proposed `{PASS, FAIL, NOT_RESOLVED}` for every check, and that was
tried and rejected. Standing rule 6 asks the suite to distinguish a decided claim from a
sampled one *rather than printing both as `PASS`*. Satisfying it by printing `PASS` beside
a prover column that disambiguates it inverts the rule. `PASS` is absent from `Outcome`
entirely, and a test asserts the word is unreachable.

`Outcome` has five members. `NOT_TRIGGERED`, `FIRED` and `NOT_APPLICABLE` are ADR-029's,
unchanged. `NOT_RUN_HERE` promotes ADR-029's second rule, that a falsifier the gate skips
says so and names where it was measured, from prose to a member, so a harness author
cannot forget it. `NOT_RESOLVED` is new and narrow: two quantities' intervals overlap and
the ordering is genuinely undetermined. It is not a synonym for either of the two above,
and collapsing the three loses the survivor accounting.

**Two claims become unrepresentable.** `PROVED` without evidence does not construct. The
evidence is a `CompletenessCertificate` today and a theorem citation when a Prover 1 check
needs one. An outcome that never ran here may not carry a warrant at all: `CORROBORATED`
asserts sampling-grade evidence was obtained, so attributing it to a falsifier void by
construction claims evidence that does not exist. Its prover cell reads `—`.

`CompletenessCertificate` gains the matching precondition: `PROVED` with `visited !=
expected` no longer constructs. It previously did, reading `complete = False`, which left
the contradiction one attribute access from anyone who did not look.

### Consequences

- `check_summary` prints `n registered, m tested here, k fired`, then counts per
  `(warrant × outcome)`. Registering four falsifiers and testing two is a different claim
  from testing four, and one number cannot carry both.
- `examples/ffg/crossover.py` reports its four D3 falsifiers on both axes. Rows 1 and 2 are
  `PROVED` from the enumeration certificate and Tier B from a stated error bound. Rows 3
  and 4 are Tier C with no warrant.
- A near-tie routes to `NOT_RESOLVED` rather than to an `AssertionError`. A tie is a
  finding about the measurement, and raising erases it.
- The `BUILD_PLAN` warrant primer is amended, since its outcome triple no longer matches
  the shipped enum and every later PR self-labels off that primer.
- Ordinary two-valued assertions are unaffected. ADR-029 scoped its vocabulary to
  registered falsifiers and that scoping holds: "does the shipped number match the NumPy
  oracle" still passes or raises, and needs no enum.
- The cost of shipping the vocabulary is a public surface that has to be maintained past
  1.0. Judged worth it because the alternative is every downstream harness inventing its
  own labels, which is the drift ADR-031 exists to stop.

---

## ADR-036 — the chunked enumerator: same decision, `O(chunk)` residency

**Date:** 2026-08-06
**Status:** Accepted
**Phase:** v0.4.5 (certifiable active inference, Paper 2 groundwork)
**Extends:** ADR-030 (the completeness certificate); ADR-031 (the search-family seam);
ADR-034 (the injected scorer)

### The question

`EnumeratedEfeSearch` front-loads the whole `|A|^H` policy array at construction and
`evaluate` returns the whole score vector. Both are `O(|A|^H)` resident. That ceiling is
what makes two declared refinement cells unreachable: `9^8` needs a 14.11 GiB policy
array and `17^7` needs 131 GiB, against a configured 23 GiB box. The score vector alone
at `17^7` is 3.28 GB.

An unreachable cell has to be reported. The honest label depends on why, and "the box is
small" is an infrastructure fact rather than a statement about the model, so a write-up
that prints it as a boundary is claiming something it did not establish.

### The timing, which is the part that matters

This lands **before** the D3 falsifier-4 registration exists and before any cell is
declared void. The registration will pre-commit against adopting a chunked enumerator in
response to a `VOID` outcome, because doing so would be choosing the instrument after
seeing the result. Taking it now is not that, and the record has to be able to show it.
No cell had been declared void on the date above.

Two cells move as a consequence, from `VOID (memory)` to a budget the registration
accepts or declines deliberately: `9^8` at 344,373,768 scored steps and `17^7` at
2,872,370,711, against the ledger's declared 17.6M. Step-0.4 stays void whatever the
enumerator does, because its lattice never lands on the cue, which is the whole reason
those two stops carry different labels.

### The decision

**A second path, not a replacement.** `ChunkedEfeSearch` enumerates in blocks and
reduces to running scalars. `EnumeratedEfeSearch` keeps its contract, including the full
score vector, which Paper 3's G-D dual reporting reads. Two paths behind one validated
seam, in the style of ADR-034's injected scorer.

**Interchangeable, and asserted so.** The load-bearing test is not that the chunked path
is self-consistent. It is that it agrees with the path that produced the published
numbers, under `==` rather than a tolerance, at the largest cell where both exist. Same
device, same dtype, same per-policy arithmetic, and every cross-policy operation the
loop needs is order-invariant in exact arithmetic, so bit-identity is the right bar and
a one-ULP disagreement would mean the policies differ rather than the rounding.

**The combine carries the tie-break, and the assertion records it never mattered.**
`jnp.argmin` resolves ties to the lowest index. Blocks are scanned in increasing index
order and the running best updates on a strict `<`, so the first occurrence of a
repeated minimum wins globally, which is that same rule reproduced block by block. A
non-strict `<=` would silently return the last copy. The guarantee lives in the combine.
A separate test on a fixture with a deliberate 32-way tie records that the two paths
agree there too.

**The certificate gains its set, and both preconditions.** `visited` stops being an
array's length and becomes a loop-carried count of unpadded lanes, so domain and
coverage come apart exactly where a padding bug lives. `CompletenessCertificate` now
carries `action_set_size`, `horizon` and `action_set_version`, and a `PROVED` warrant
requires both `expected == |A|^H` and `visited == expected`. Neither is a printed field.
The set naming also fixes a latent defect that predates chunking: `expected` alone
conflates the base with the exponent, since 81 is `9^2` and `3^4` alike, so rendered
evidence could not tell two enumerations apart. That is standing prohibition 9 applied
to evidence rather than to prose.

**Padded lanes score policy 0 and are masked.** The tail block pads to a real index
rather than a made-up one, so the kernel never sees an input the model does not define,
then masks those lanes to `+inf` and out of `visited`. Tests run at an `N` deliberately
indivisible by the block.

**The block is clamped to `|A|^H`.** A block wider than the enumeration pads the
difference and scores it. At `|A|^H = 6561` against a 65536 block that is ten times the
arithmetic for the same answer, measured. The request is an upper bound.

**The default block is measured, not guessed.** Swept 64 to 65536 at `9^6`. Throughput
has a broad plateau from 512 to 4096 and falls away on both sides, with 65536 running
3.4× slower than the plateau. Peak is flat at 0.42–0.43 GiB from 64 to 8192, so inside
the plateau the block size buys nothing back in residency and the choice is about rate
alone. `DEFAULT_CHUNK = 4096` rather than the slightly faster 2048, because its spread
across repeats is 2% against 20% and PR-2 declares compute budgets off this rate. A
number that occasionally drops a quarter is a poor basis for a declared budget.

### Consequences

- Measured on the crossover model at `9^6`: identical argmin index, `G` equal under
  `==`, and the value is `364.642964185792`, which is the `364.6430` published at
  `r10:190`. Peak resident memory is 0.454 GiB against the front-loaded path's 1.027,
  and the chunked figure is flat across `9^4`, `9^5` and `9^6` because it is the fixed
  XLA baseline rather than the enumeration.
- Throughput improves rather than degrading: 25.3k policies/s against 15.4k at `9^6`.
  The block loop was expected to cost something for the residency it buys, and does not.
- The enumerator's memory model is now chunk-determined, so any budget line derived from
  the old `|A|^H` figure is re-derived before it is declared.
- A cost the chunked path cannot avoid: it has no score vector, so anything a caller
  wants over all `|A|^H` policies has to be phrased as a fold. `_BlockReducer` is that
  seam. A caller wanting the vector back uses the front-loaded path and accepts its
  ceiling.
- `VoidReason.MEMORY` may now describe no live cell. It stays in the type regardless: a
  configured limit and a declared budget are different stops, and a distinction the type
  drops is one the prose has to carry.
- R10 hardening (issue #65) takes **ADR-037**, one past this.

---

## ADR-037 — symbolic evidence, and a result that ran before its registration

**Date:** 2026-08-16
**Status:** Accepted
**Phase:** v0.4.5 (certifiable active inference, GATE-D4 groundwork)
**Extends:** ADR-035 (the warrant vocabulary and the evidence precondition)

### The question

Issue #65's symbolic track needed three things decided. What backs a Prover 2 claim, since
`CheckReport` refuses `PROVED` without evidence and a completeness certificate is the wrong
object for a claim decided by argument. How the expansion is computed, since the obvious
route does not terminate. And what the programme does when a registered-track result is
produced before anything registers it, which is the one that happened rather than the one
that was planned.

### Decision 1 — `SymbolicReduction` is what a Prover 2 claim carries

The warrant ledger puts symbolic computation at theorem-grade *"provided the symbolic
reduction is faithful to the analytic claim, which is a human obligation the CAS does not
discharge."* A CAS establishes that one expression equals another. Whether those are the
expressions the claim is about is not a thing it can be asked.

`SymbolicReduction` records that obligation: the claim in words, where the setup was hand
derived against the problem, and the assumptions the identity is contingent on. `Evidence`
becomes a union of it and `CompletenessCertificate`, one member per decisive prover.

Blank fields do not construct. Without that the type is a formality and the rule it
encodes has nothing enforcing it. That rule: a correspondence nobody can fill honestly
means the check reports `CORROBORATED` and says why in its detail.

### Decision 2 — the expansion is built by truncation, and the predictive is exact

`sympy.series` on the assembled log-ratio does not terminate in fifteen minutes. Every
primitive is therefore an explicit polynomial in `σ`, geometric for the gain, binomial for
the posterior width and the exponential series for `e^{−δ}`, with truncation applied inside
each product rather than at the end. `series` survives only in the checks, on small rational
functions, where it is the independent arm that licenses the swap. The load-bearing check is
that the truncation path reproduces `series` on the assembled `W` term for term through
`σ³`, which is as far as `series` can be afforded.

The two order conventions differ and the difference is real: `series(expr, x, 0, n)` keeps
powers *below* `n`, `truncate(expr, order)` keeps powers *up to and including* `order`.
Reading one as the other silently drops the top term. It was misread once, during this work.

At `σ⁴` the innovation is averaged under the exact `ν = σz₁ + √R̄·e^{δ/2}·z₂`, not under a
Gaussian with a corrected variance. This is not a refinement. The leading-order predictive
gives a `c₄` that is 5.7 times the exact one at the declared operating point and matches no
measurement, so the choice decides the coefficient rather than polishing it.
`predictive_truncation` had already recorded why: `p*` is a scale mixture with exponential
tails, and no Gaussian stands in for it at any variance.

### Decision 3 — a result produced out of order is disclosed, not discarded

The `c₄` derivation was run before any amendment registered what it would produce or what
would count as a failure. Three responses were available.

**Discard and redo after registering.** Rejected. The result cannot be unseen, so a rerun
would be a rerun by someone who knows the answer, and presenting it as fresh would be a
worse misrepresentation than the original slip.

**Argue that no harm was done.** Rejected as insufficient on its own. "We did not tune it"
is not a claim a reviewer can test, and a defence that rests on our own account of our
intentions is exactly the kind the falsification battery exists to refuse.

**Disclose the sequence, separate content from schedule, and repair what is still
repairable.** Adopted. `research/gate_d4_registration.md` section 7 carries the numbered
sequence at its head, before the result. The predictions the derivation did satisfy are
cited by their earlier dates in the same file, so their standing is checkable from the git
history rather than from assertion. The independence argument is made structural rather than
biographical: the symbolic modules contain no floats, no family and no value for `R̄`, so
there is no quantity a fitted number could have been substituted into, and a reader can
confirm that by reading them. The derivation also disagrees with the fit by 1.2%, which is
not what steering toward it produces.

What that does not repair is the ordering, and the section says so in those words. The
repair is applied where it still can be: the three families not yet run are pre-registered
with their predicted coefficients, their pass bar and their VOID condition, before the runs.

**The standing rule this sets.** A registered-track result computed before its registration
is recorded with its sequence, at the head of the section that reports it, and the remaining
untested cases are pre-registered before they are run. The disclosure is not a penalty and
does not weaken the result's content. It changes what may be claimed about the result's
*schedule*, which is a separate axis, and one a reader is entitled to judge separately.

### Consequences

- `SymbolicReduction` is public API and carries a `docs/api` page. A `PROVED` report is now
  reachable by argument as well as by enumeration.
- 70 identities across `series_kernel`, `log_ratio_series` and `gap_series` report `PROVED`
  at Tier A, each carrying a reduction. Mutation probes confirm the suite discriminates
  rather than merely agreeing. The CI job pins all three counts.
- `c₄` is a reverse-KL coefficient. `c₂` is direction-free, checked at `σ²` and asserted no
  further, so every site quoting `c₄` carries the direction with it.
- One of the four out-of-sample families fires. `1.5 + 0.5 tanh(x)` returns `σ^6.302`
  against a ±0.25 bar, on the over-cancellation side of 6, and the leave-one-out
  diagnostic leaves it standing. The fire is carried alongside the three passes rather
  than summarised with them.
- `c₄` moves from a Tier C fit with a 0.36% extraction spread to a closed form. The fit's
  residual error is explained rather than absorbed, and the registration's simple-fraction
  hypothesis is settled instead of left open.
- Section 7 of the registration is the one place in that file where the ordering has to be
  argued rather than read off the commit record. It is marked as such in the "What was known
  when" table so a reader finds it without looking for it.
- The quadratic family's cell is post-hoc in its scheduling. The other four are
  out-of-sample. Any write-up quoting `c₄` carries that split rather than averaging over it.

---

## ADR-038 — the warrant vocabulary drops its letters

**Date:** 2026-08-17
**Status:** Accepted
**Phase:** v0.4.5 (certifiable active inference, Paper 2 groundwork)
**Extends:** ADR-035 (the warrant vocabulary ships)

### The question

`Tier.A`, `Tier.B` and `Tier.C` ranked by position in the alphabet. So did the prover
sub-modes `3a`, `3b` and `3c`. A reader met `tier B · 3b` in a battery row and had to
already know two orderings to read it. Neither letter says what it means, and the two
scales collided badly enough that the ledger carried a standing warning telling readers
which "tier" was meant.

A second scale made it worse. "Register Tier 1/2" classified deferred work by whether it
ships a feature, and had nothing to do with how well a number is known.

### Decision

**The tier members become words.** `Tier` keeps its class name, because the concept is
cited in prose outside this repo. Its members and values do not:

| was | is | value |
| --- | --- | --- |
| `Tier.A` | `Tier.EXACT` | `"exact"` |
| `Tier.B` | `Tier.BOUNDED` | `"bounded"` |
| `Tier.C` | `Tier.COMPUTED` | `"computed"` |

`CheckReport.__str__` keeps the field word, so a row now reads
`q: NOT TRIGGERED (PROVED, tier exact). …`.

**The prover sub-modes become words in prose.** They are prose-only, with no code behind
them, so the rename is free:

| was | is |
| --- | --- |
| `3a` | Prover 3 · sample |
| `3b` | Prover 3 · enumeration |
| `3c` | Prover 3 · validated |

**"Register Tier 1/2" becomes "register class 1/2".** The collision the ledger's
vocabulary warning existed to manage is gone, so the warning goes with it.

**`Warrant` and `Outcome` are untouched.** Their members are already words, already
ordered, and `Warrant` carries the only invariants that matter: `PROVED` requires
evidence, and a check that never ran carries no warrant.

**Warrant renders strongest-first** wherever it is tabulated: `PROVED`, `CERTIFIED`,
`CORROBORATED`. No letter stands in for rank anywhere.

**One canonical table**, at the head of `research/warrant_ledger.md`, carrying the
warrant a claim earns and the evidence it requires, and the tier a number is known to.
The ledger's sections 1 and 2 are merged under a combined `1–2` heading rather than
renumbered, because `research/gate_d4_registration.md` cites ledger sections 5 and 8 and
the battery cites 3 through 6. Renumbering would have broken pointers in a registration
document that must not be amended. Every other document points at the canonical table
instead of restating it.

**Aliases were rejected.** `A = EXACT` would have kept every old call site working and
cost nothing to write. It also would have left two live vocabularies in the repo at once,
and the point of the change is that a document's vocabulary dates it. A reader meeting
`3b` in a file should conclude the file predates this decision, not that the author chose
the older of two current spellings.

### The boundary, for anyone auditing across it

The rename landed on `65-warrant-symbolic-evidence` at
**`9c70c08`** (code and call sites), with the prose
following in `3280d72` and the documents after that. Anything at or before `ffc53f7`
speaks the old vocabulary. The two tables above are the translation, in both directions.

`cpomdp.warrant` was added 2026-08-05, one day after the v0.4.4 release, and has sat
under `## [Unreleased]` ever since. No release carries `Tier.A`, so this is not a
breaking change and the changelog records it as the vocabulary the module ships with.

### What was deliberately not renamed

The dated ADRs above keep the words they were written in. `DECISIONS.md` is append-only,
and a record that silently acquires today's vocabulary stops dating itself, which is the
property this decision exists to buy. Five references to `3a`, `3b` and `3c` survive in
ADR-030 and ADR-035 for that reason, and the map above is what translates them.

`research/gate_d4_registration.md` quotes no report lines and needed no amendment.

### Consequences

- A battery row names its tier as `EXACT` rather than as `A`, so it needs no legend.
- The ledger's vocabulary warning is deleted rather than corrected.
- Any external document quoting `Tier.B` or `3b` translates through the tables above.
- `research/` is covered by neither `ty` nor `pytest`, so the check modules under
  `research/checks/` were smoke-run by hand rather than caught by CI. That gap is real
  and this decision does not close it.

---

## ADR-039 — the warrant vocabulary becomes its own distribution, in a uv workspace

**Date:** 2026-08-18
**Status:** Accepted
**Phase:** v0.4.5 (certifiable active inference, Paper 2 groundwork)
**Extends:** ADR-035 (the warrant vocabulary ships), ADR-038 (it drops its letters)

### The question

The vocabulary has two audiences and one of them does not want cpomdp. A check suite
labelling its findings needs `Warrant`, `Outcome`, `Tier` and `CheckReport`. Installing
those meant installing JAX, jaxtyping and NumPy, because they shipped inside a library
whose reason for existing is continuous active inference.

The coupling ran the other way too, and worse. `CompletenessCertificate` lived in
`cpomdp.enumeration`, which imports `cpomdp.warrant`. So the warrant module could not
import the certificate at module scope, and did it inside a function called on every
`CheckReport` construction that carried evidence. The module docstring described that
back-reference as annotation-only. It was not: the `isinstance` guard behind `PROVED`
ran a real import on a real path. A cycle documented as absent is worse than one
documented as present.

### Decision

**`warrantlib` is a separate distribution, developed in this repository.** It publishes
to PyPI beside cpomdp, at `0.1.0`, from `packages/warrantlib`. Its dependency list is
empty and stays empty. cpomdp depends on it and adds nothing to it.

`warrant` was taken on PyPI. The import name had to differ from it regardless of the
distribution name, because two installed packages answering to `import warrant` produce
a silent wrong-module bug rather than an error.

**`CompletenessCertificate` moves into `warrantlib`.** It is an evidence kind of the
warrant vocabulary, tabulated in the ledger beside `SymbolicReduction`, and a frozen
dataclass over `int, int, Warrant, int, int, str` that touches nothing numerical. It
lived in `cpomdp.enumeration` by history, and that history is what created the cycle.
`cpomdp.enumeration` re-exports it, so every call site is unchanged and the two names
remain one object.

With both evidence kinds in one module the deferred import goes, the `Evidence` union
leaves `if TYPE_CHECKING:` and becomes a real alias, and the guard reads a module-scope
tuple.

**A `@runtime_checkable` Protocol was rejected.** It would have let the certificate stay
where it was. `isinstance` against a data Protocol checks attribute *presence*, not
types and not invariants, so any object with the right field names would pass a guard
whose entire purpose is refusing evidence that does not construct. Weakening the check
to preserve a directory boundary is the wrong trade.

**`cpomdp.warrant` stays, as a re-export.** Eight lines, no `DeprecationWarning`. cpomdp
is `Development Status :: 2 - Pre-Alpha` and a warning on every import of a name the
project itself moved is noise. Revisit at warrantlib 1.0.

**The repository becomes a uv workspace, and does not split.** cpomdp stays the root
package, so `src/cpomdp/` does not move and existing CI keeps working. `warrantlib` and
`cpomdp-research` are members. One `uv.lock`, one `.venv`.

**`research` becomes a member too**, at `research/src/research/`, carrying the scipy and
sympy the check suites need. Those leave cpomdp's `dev` group, where they were listed
under a comment explaining that nothing in the library imports them. The `*.md`
registrations do not move.

**`cpomdp-research` is a `dev` dependency of the root, rather than a member reached
through `uv sync --all-packages`.** `--all-packages` is the documented way and it is a
trap here: a bare `uv run` re-syncs to the root's own environment and takes the member
back out again, so `python -m research.checks.X` fails with a `ModuleNotFoundError`
naming a module that is right there on disk. Declaring the dependency makes the default
environment the correct one, and no workflow file needs a flag it can forget.

**Extraction into its own repository is not this decision.** The module boundary is now
real, so a later `git subtree split` is a topology change rather than a rewrite. Two
things would trigger it: a second consumer of the vocabulary that is not this project,
or a contributor who needs to edit registrations without a JAX and Julia toolchain.

### Consequences

- `pip install warrantlib` gets the vocabulary and nothing else, on Python 3.10 and up.
- The published cpomdp wheel carries `Requires-Dist: warrantlib>=0.1`. The workspace
  source is stripped at build time, so no `file://` reaches the metadata.
- A second PyPI Trusted Publisher must be configured by hand for `warrantlib` before the
  first publish run, along with a `pypi-warrantlib` GitHub environment. Nothing in this
  repository can do that.
- `publish.yml` builds each distribution explicitly. A bare `uv build` in a workspace
  builds the root alone, so the old single command would have silently stopped shipping
  anything new. warrantlib publishes only when its version differs from PyPI's, so a
  cpomdp release does not churn it.
- ADR-038 recorded that `research/` was covered by neither `ty` nor `pytest`. Half of
  that gap closes here: `ty` now checks `research/src`. `pytest` still does not, and the
  `symbolic` CI job with its pinned counts is still what runs the suites.
- The `research.checks.<module>` import path is unchanged, which is why the member uses
  a src layout with the package name repeated. CI, the registrations and the build
  tracker all name that path.
- A permanent test asserts that importing `warrantlib` in a clean interpreter pulls in
  no `cpomdp` module. That is what stops the cycle growing back.

---

## ADR-040 — warrantlib publishes on a manual trigger, and the ordering hazard that leaves

**Date:** 2026-08-20
**Status:** Accepted
**Extends:** ADR-039 (the warrant vocabulary becomes its own distribution)

### The question

cpomdp 0.4.4 is on PyPI without a warrant dependency. Every cpomdp release after ADR-039
carries `Requires-Dist: warrantlib>=0.1`, so that dependency has to exist on PyPI before
the release that names it. `publish.yml` builds both distributions from one `release`
event, and `publish-cpomdp` and `publish-warrantlib` each declare only `needs: build`.
They run in parallel with nothing ordering them. A failed warrantlib upload beside a
successful cpomdp one publishes a cpomdp that `pip` cannot resolve, and the version number
is spent.

### Decision

**warrantlib 0.1.0 publishes on its own, before any cpomdp release needs it.**
`publish.yml` gains `workflow_dispatch`. cpomdp's build, artifact and publish job are all
gated on `github.event_name == 'release'`, so a dispatch touches warrantlib alone and
cpomdp's version still always corresponds to a tag.

Ordering the two publish jobs was the alternative. It needs an `always()` guard, because a
plain `needs: publish-warrantlib` skips cpomdp on every release where warrantlib's version
has not moved, which is most of them. Publishing once by hand retires the hazard for
`>=0.1` with no conditional logic added to the release path.

**A dispatch is restricted to `main` and to release events.** A Trusted Publisher is scoped
to repository, workflow filename and environment. It does not constrain the ref, so a
dispatch from any branch mints a valid upload token, and PyPI never lets a version be
reclaimed. The guard lives in two places on purpose: an `if` on the job, which shows up in
a diff, and a deployment-branch rule on the `pypi-warrantlib` environment, which survives
the workflow file being edited on the branch being dispatched.

### Consequences

- The hazard is retired for `>=0.1` only. A future cpomdp release requiring `>=0.2` meets
  the identical race, because the two publish jobs are still unordered. The options at
  that point are unchanged: dispatch the new warrantlib first, or order the jobs with an
  `always()` guard. This is the entry that says so, since the reasoning otherwise lived
  only in a pull-request description.
- A dispatch whose version already matches PyPI now fails instead of exiting green. Someone
  pressing the button is asking for a publish, and a green run that uploaded nothing reads
  as one that did.
- `publish.yml` carries a `concurrency` group, keyed on the event. The dispatch button is
  a double-click surface that `release: published` never was, and two presses would both
  pass the version guard before either upload landed. Keying on the event is what stops a
  release queueing behind a dispatch that is sitting at an environment gate.
- A dispatch from any other ref fails before the build runs. Left to the publish job's
  own condition it would build, upload an artifact and finish green, with a skipped job
  carrying no annotation as the only sign that nothing shipped.
- The `pypi-warrantlib` environment needs its deployment-branch rule set when it is
  created. A new GitHub environment allows every branch by default, which is the condition
  the job guard above is compensating for.

## ADR-041 — a claim says where it was registered, not only that it was decided

**Date:** 2026-08-20
**Status:** Accepted
**Extends:** ADR-035 (the warrant vocabulary ships), ADR-037 (symbolic evidence)

### The question

A `PROVED` report says a claim was decided and what backs it. It says nothing about when
the bar was set. A threshold chosen once the number is visible decides nothing, and the
report of one is indistinguishable from the report of a genuine pre-registration.

The repository already answers this in prose. `research/gate_d4_registration.md` carries a
table of `first registered` against `measured or proved`, one row per claim, and where the
two cells hold the same hash the surrounding text says the ordering rests on the
document's account rather than on the history. That table is the thing a reviewer wants
and it is reachable only by reading the registration end to end.

### Decision

**`Provenance` carries two refs and a statement.** `registered_at` is where the
prediction, the bar or the derivation was registered. `measured_at` is the ref whose tree
produced the number. `registered` says in one line what a reviewer will find at the first
of them, because a bare ref sends them to a diff and leaves them to work out which part
of it mattered.

**A ref is a git commit SHA, an http(s) URL or a DOI.** A path, a branch, a tag and `HEAD`
are refused. Each satisfies a presence check exactly as well as a commit does, and each
resolves to a different tree every time it is read. A URL is taken to be a permalink. One
tracking a branch has the defect that rules out a path, and the validator cannot tell them
apart.

**`PROVED` requires one, as a tuple.** The same rule the evidence already follows, and the
tuple for the same reason: a claim resting on two registrations rests on both, and
carrying one of them understates what a reviewer has to check.

**Same-ref is allowed and marked.** Registering and measuring in one commit is what
happens whenever a check and the derivation behind it land together. The render says the
ordering is not established by history, so nobody reads it as a fact the graph supports.
An abbreviated ref counts as the same commit, or lengthening one hash would walk away
from the marker while naming the same thing.

### Consequences

- **The type cannot detect the failure it exists to expose.** Equality is checkable in a
  string and ordering is not, so a registration written after the fact renders exactly
  like one written before. `tests/test_provenance_ordering.py` asks git, and it lives
  outside warrantlib because warrantlib depends on the standard library alone.
- **Three of the nine sources fail that test, and are marked `xfail`.**
  `research/c4_hand_derivation.md` was committed 2026-08-17, after `log_ratio_series`
  (`23f0c47`, 2026-08-15) and `gap_series` (`1888ad4`, 2026-08-16) were already measuring
  against it. ADR-037 discloses the same ordering for the result these back. `measured_at`
  names the earliest reliance rather than the latest commit to touch the file, since the
  latest would read as an ordering the history does not support. Repairing this means
  re-measuring against the derivation, not editing the refs.
- **`measured_at` is unknowable while it is being written.** A commit cannot contain its
  own hash. This migration escapes it because no arithmetic changed, so every ref already
  exists. A new check measuring a new number has two honest options: mark it same-ref and
  let the render say so, or land it in two commits. Same-ref is the default, because it
  states the truth without ceremony.
- **The downgrade route is open and documented.** A real proof with no registration
  reports `CORROBORATED` and says why in its detail, the same escape the correspondence
  rule already has. Nobody has to invent a ref, since same-ref-and-marked is always
  available and always honest.
- **cpomdp requires `warrantlib>=0.2`.** `cpomdp.warrant` re-exports `Provenance`, so an
  installed 0.1 fails at import. Per ADR-040, warrantlib 0.2.0 publishes before any cpomdp
  release carrying the new floor.
- The three pinned suite counts did not move. The provenance renders after `detail`, and
  `check_summary` never reads `detail`.

---

## ADR-042 — a check has a key, and a report has a wire form

**Date:** 2026-08-21
**Status:** Accepted
**Extends:** ADR-035 (the warrant vocabulary ships), ADR-038 (it drops its letters),
ADR-039 (it becomes its own distribution)

### The question

Three strings in `.github/workflows/ci.yml` are what stop the symbolic suites going green
by asking less:

    run_suite series_kernel    "23 registered, 23 tested here, none fired"
    run_suite log_ratio_series "18 registered, 18 tested here, none fired"
    run_suite gap_series       "29 registered, 29 tested here, none fired"

A stage dropped from a module's `run_checks()` removes its checks and fails nothing. The
count is the only thing in the way, and it is a hand-maintained mirror of whatever the
suite happens to produce. Editing it reads as housekeeping. It also cannot say which check
left: 23 becoming 22 names nothing.

A second gap sits beside it. Nothing writes a report anywhere but a terminal.
`check_summary` renders, CI greps the header back. So no run can be compared with another,
and every number in `research/*.md` is typed by hand. The figure 70, being 23 plus 18 plus
29, appears in four documents and is derived in none of them.

Both gaps need one thing that did not exist: a name for a check that survives the prose
being reworded.

### Decision 1 — the key is declared, never derived

`CheckReport` carries `check_id`, required, validated at construction. Dot-separated
segments of ASCII letters, digits and underscores. Anything else does not construct.

Deriving it from `name` was the cheap option and is the one to avoid. A slugged key moves
the moment the prose is improved, and a ledger joining two runs then reads one check as one
dropped and one added. That is the exact reading a ledger exists to rule out.

This is not a guess about which way the trade goes. Every tool facing the same problem
landed in the same place. Allure's `@allure.id` exists because the derived `fullName`
breaks under refactor and abandons a test's history. jamb and pytest-testrail carry an
external case id on a marker. pytreqt defines ids in a specification file and validates in
both directions. pytest core declined to supply one, in issue #10460, closed as not
planned, so there is nothing upstream to wait for.

The character class is spelled out rather than written `\w`. `\w` matches unicode by
default, and this programme's prose is full of `c₂` and `R̄`. Two ids differing by a
character a reader cannot tell apart is worse than no id at all.

Required rather than defaulted. A default lets a check ship with no key, and the key is the
whole point of the field.

`__str__` does not render it. The printed line is for a person, who already has the name.
The id is for the record.

**The guard reaches the two prose fields beside it.** `name` and `detail` were the only
text on a report that nothing checked, so a blank one constructed and was written out.
`detail` is documented as the field that stops a report being a bare outcome, and a blank
`name` is a summary row nobody can attribute. Both now go through the same readability
check as the key and the evidence, which is what the shipped schema had already been
asserting on their behalf.

### Decision 2 — the vocabulary and its wire form are separate modules

`warrantlib` splits into `_vocabulary.py` and `_serialise.py`, with `__init__.py` as a
façade owning `__all__`. The records and the form they take on disk are different
responsibilities, and the serialiser cannot live beside the classes it reads without the
package importing itself.

A test parses both modules with `ast` and asserts `__all__` covers every public name they
define. Reflection over the imported module cannot tell a definition from an import, so the
source is what gets read. The test carries its own floor, because a set that comes back
empty satisfies the comparison by asking nothing.

The split has a cost that is not fixed here. mkdocstrings resolves cross-reference tooltips
to the canonical path, so a signature table's hover text reads
`Warrant (warrantlib._vocabulary.Warrant)`. Link text, anchors and `--strict` are all
correct. Rewriting `__module__` in the façade is the usual repair and it cannot reach
`Evidence`, which is a union rather than a class, so it would leave behind the
inconsistency it was applied to remove.

### Decision 3 — reading refuses what it cannot read

`report_from_dict` raises on an unknown schema version, an evidence kind with no class
behind it, an enum value no member carries, and a missing field. Every field the writer
emits is required on the way in, with no field read through a default. A record missing
its `warrant` read back as one carrying none, which is the status change nobody made,
arriving through the reader rather than through the data.

**A list field is read as a list, never coerced into one.** `tuple` accepts any iterable
and a bare string is one, so `assumptions: "formal"` became six one-character assumptions.
Each passed its own blank check, and `SymbolicReduction`'s guard against exactly this
reads a tuple by the time it looks. Coercion is how a wire form fabricates evidence while
every precondition reports satisfied.

Guessing has a measured cost in this repository. ADR-038 renamed `Tier.A` to `Tier.EXACT`.
A reader that mapped an unknown tier onto its nearest neighbour would have read every
record written before that rename as a tier change, and reported status changes nobody
made. Refusing to compare is the weaker claim and the true one.

Reading goes through the constructor rather than around it, so every precondition still
applies on the way in. A record naming `PROVED` with its evidence stripped does not
construct. A wire form able to bypass the guards would make the guards optional.

A JSON Schema ships beside the code at `warrantlib/report.schema.json`, for a consumer
reading a ledger without Python. The suite validates the writer's own output against it,
because a schema nothing checks drifts from the writer silently. `jsonschema` is a
development dependency of the root project. warrantlib's own dependency list stays empty,
as ADR-039 requires.

### Decision 4 — `cpomdp.warrant` does not grow

The shim mirrors the nine names cpomdp exported before ADR-039 moved them, and stops there.
`SCHEMA_VERSION`, `report_to_dict` and `report_from_dict` were never in cpomdp, so no
import path needs preserving, and carrying them would build a second public surface to
maintain past 1.0. The test that pinned set equality now pins a subset plus the exact
nine-name floor, since a subset rule on its own is satisfied by a shim that has lost a name.

### Consequences

- **The pinned CI counts still pass, unchanged.** The three symbolic suites report 23, 18
  and 29 with none fired. `gap_expansion` reports 54 registered, 38 tested, 7 fired.
  `predictive_truncation` reports 88, 49 and 6. `check_summary` reads neither the id nor
  `detail`. Those strings
  are still what guards the suites. This decision gives a guard something to be built on
  and does not build it.
- **Around 86 call sites declare an id**, because the shared constructors in
  `series_kernel` take the name as a parameter and every caller supplies its own. Five
  suites produce 232 reports carrying 232 distinct ids. `tests/test_check_ids.py` asserts
  that, and that every id carries its module's namespace. `gap_series` costs half a minute
  to derive, so its cases carry the `slow` marker and gate on merge rather than on every
  pull request.
- **A cell's id names its `σ` losslessly.** The first slug rounded to three decimals, so
  `--sigmas 0.0301 0.0302` handed two cells one id. Rounding makes a collision reachable
  from the command line, which is a defect in the rule rather than in a particular grid,
  so the slug renders the value at full precision and is one-to-one by construction.
- **`NoiseFamily` carries a `key`**, duplicating its own entry in `FAMILIES`. The printed
  `name` is maths notation and cannot be a key. Nothing asserts the two agree, so they can
  drift.
- **warrantlib is 0.3.0, and the change is breaking.** cpomdp's floor stays at `>=0.2`
  deliberately: nothing under `src/cpomdp` constructs a report, and moving the floor
  re-arms ADR-040's publish-ordering race for no gain. `cpomdp-research` moves to `>=0.3`,
  and never publishes.
- **`py.typed` ships**, which it should have done since ADR-039. The distribution carried
  the `Typing :: Typed` classifier without the marker file, so an installed consumer got
  no annotations from it.
- **The manifest, the reconciliation and the pytest collector are not in this decision.**
  Nothing here replaces a count string. What it removes is the reason one could not be
  replaced.

## ADR-043 — the fourth D3 falsifier, registered on two axes and answered on both

**Date:** 2026-08-21
**Status:** Accepted
**Phase:** v0.4.5 (PR-2, R10 hardening)
**Extends:** ADR-034 (the scorer seam), ADR-036 (the chunked enumerator), ADR-041
(a claim says where it was registered)

### The question

R10 registered `H* = 7` as an upper bound *because the declared action set clips the
reach*. That makes stability under a change of action set load-bearing rather than
precautionary: the release's own qualifier says the number moves if the set moves. Two
prior answers stood in the tree, and neither had a run behind it in this repository. The
extension cell `{−4,…,2}` carried a deduced `H* = 6`. The step-`0.5` refinement carried a
measured-looking "stable under refinement, falsifier 4 not triggered" with no commit
building the nine-action set.

### Decision

**Two axes, registered separately and not interchangeable.** Extension widens the
magnitude range at the same spacing. Refinement subdivides the same range. They are
different operations and a single "action-set stability" claim would blur them.

**Extension takes a directional prediction, refinement a stability test.** `H* ≤ 6` for
`{−4,…,2}`, argued from geometry: at `−3` the prior-ward branch already covers its `−3` in
one step, so magnitude `4` buys it nothing, while it cuts the cue-ward return from two
steps to one. Refinement gets `|ΔH*| ≤ 1` instead, because neither branch's step count
moves when the largest magnitude does not, so no direction is arguable. Registering a
stability test as a stability test, rather than dressing it as a prediction, is the whole
of the difference between the two axes here.

**Budgets are declared in both units, because they disagree.** Policy counts and scored
steps diverge: the step-`0.5` cell at `H = 7` is 4,782,969 policies, far inside the `9^7`
line, and 33.5M scored steps against the ledger's `H_max` budget of 17.6M. Declaring one
unit silently picks the flattering one.

**Registration lands in its own commit, before the measurement.** `86d1f22` carries both
axes' predictions, and the measurements point back at it through `Provenance`. ADR-041
records
why `measured_at` cannot be the commit being written.

### Results

**Extension: `H* = 6`, PASS.** The argmin at `H = 6` is `[+1,−4,0,0,0,0]` — one step to
the cue, then the single `−4` return the argument was built on. The mechanism the
prediction rested on is the one that fired.

Two things the prediction got wrong, recorded as such. "Plausibly 5" did not hold: `H = 5`
stays prior-ward though the walk is feasible there, so feasibility of the shorter return
is not what sets the crossover. And extension **saturates**: `H* = 6` on `{−3,…,2}` and on
`{−4,…,2}` alike, so the one-step return is taken without moving the horizon.

The deduced row it replaces reached the right number by a route the measurement
contradicts, since the winning policy uses `−4` and not `−3`. A row can carry a correct
number for a false reason, and only the run tells them apart.

**Refinement at step-`0.5`: `H* = 7`, PASS, and the published numbers hold.** Measured
`364.642964185792` and `425.163110098734` against the write-up's `364.6430` and
`425.1631`, inside the `5e-5` tolerance registered on 2026-08-21 for exactly this
comparison. Nothing is retracted. What the write-up lacked was a run in this repository,
not accuracy, so the retraction was about provenance and it is now discharged.

### Consequences

- **`9^8` is moot, not deferred.** The registration made the `H = 8` cell contingent on
  `H*` rising under refinement. It did not rise, so the cell answers a question that did
  not arise.
- **The memory wall was a front-loaded artifact.** Peak measured at 0.46 GiB on the
  chunked path, against `enumeration_cost`'s 22.6 GiB for `9^8` and 210 GiB for `17^7`.
  ADR-036's "block-determined and flat in `|A|^H`" holds, and the cells that looked
  budget-bound are bound by wall-clock alone.
- **The measured rows are recorded constants, not a gated re-run.** The cheapest cell is
  531,441 policies and the dearest 410,338,673, so no gate can afford the set. What separates this from the write-up claim it
  replaces is that a commit builds the set, each row carries its completeness certificate,
  the provenance names both refs, and `--refinement` reproduces it. Cheap tests assert the
  recorded values against the published ones, which is the standing rule discharged
  without putting minutes on every merge.
- **Falsifier 4 moved off `NOT_RUN_HERE`.** The demo now reports five falsifiers, four
  tested. Its pinned count moved with the diff that justified it.
- **step-`0.25` is outstanding**, at 410,338,673 policies and a projected 2.7 hours at the
  measured 41,982 policies/s. Deferred on time, not on budget or memory. The registered
  stability test stands for it, so PR-2's merge gate is not yet met.

## ADR-044 — the refinement axis answered at both spacings

**Date:** 2026-08-21
**Status:** Accepted
**Extends:** ADR-043 (the fourth D3 falsifier, registered on two axes)

### What changed

ADR-043 left step-`0.25` outstanding, deferred on wall-clock. It has now run:
410,338,673 policies at `H = 7` in 2.5 hours at 46,124 policies/s, peak 0.47 GiB,
certified `PROVED` with every policy visited.

`H* = 7`, so `|ΔH*| = 0` against the registered bar of 1. Both refinement cells pass, both
axes of the fourth D3 falsifier report an outcome, and PR-2's merge gate is met.

### The finding worth keeping

**The two spacings agree to the digit.** `364.642964185792` at `H = 6` and
`425.163110098734` at `H = 7`, byte-identical between step-`0.5` and step-`0.25`, across a
set with twice the actions and 86 times the policies. No quarter-step action reaches
either argmin, just as no half-step did.

Half of that is expected and worth naming as such: the coarser set is a subset, so where
the argmin lies in it the scores must match. That half is a code-correctness check riding
along. The evidential half is that the argmin does lie in it, at both spacings. Refinement
had two chances to move the optimum toward the cue and took neither.

### Consequences

- **`H* = 7` is robust to how finely the range is sampled, and moves only when the range
  itself widens.** Extension shifted it to 6 and then saturated. Refinement does not shift
  it at all. The release's qualifier, that 7 is an upper bound *because the set clips the
  reach*, is now measured rather than asserted: it is the clipping that matters, not the
  grid.
- **The chunked path is what made this answerable.** `enumeration_cost` read 210 GiB for
  this cell on the front-loaded path against 19 GiB of machine. Measured peak was 0.47
  GiB, so the cell was never budget-bound, only slow. A registration that had treated the
  front-loaded figure as the budget would have declared it `VOID` and recorded a
  non-result.
- **Neither refinement cell re-enumerates on a gate.** Both are recorded constants
  carrying their
  certificates and a provenance, reproducible with `--refinement`. Cheap tests assert the
  recorded values, including that the two cells agree exactly.

---

## ADR-045 — the floor moves to Python 3.11, and a generated file gets comments

**Date:** 2026-08-22
**Status:** Accepted
**Extends:** ADR-039 (the warrant vocabulary becomes its own distribution)

### The question

The check manifest is a generated file whose whole job is to be read in a diff. A stage
dropped from a suite shows up there as a removed line, and the reviewer seeing that line
needs to know two things the line cannot say: that the file is generated, and what command
regenerates it.

JSON cannot carry a comment. TOML can, and `tomllib` reads it from the standard library,
which is the only library `warrantlib` is allowed to want. It arrived in 3.11.

`warrantlib` declared `>=3.10`, and cpomdp depends on `warrantlib`, so the floor is one
number for both distributions whether or not they state it separately.

### Decision

**Both distributions require Python 3.11.** `cpomdp`, `warrantlib` and `cpomdp-research`
move together, the 3.10 classifier goes, the CI matrix drops a rung, and `ruff` and `ty`
target 3.11.

**The manifest is TOML.** A generated file that explains itself is worth more than one
that needs a paragraph elsewhere saying what it is. The alternative was a JSON file with a
`generated_by` field carrying the same sentence as data, which works and reads as a
workaround.

**A dependency was not the way to get there.** `tomli` would have read TOML on 3.10 and
`tomli-w` would write it on any version. Either one ends "the standard library is the only
dependency", which is the property that makes the vocabulary installable next to a check
suite that wants nothing else. Moving the floor spends a supported version once. Taking a
dependency spends the property permanently.

**The writer emits a restricted subset by hand**, since `tomllib` reads and does not
write. The manifest holds table headers, quoted strings and arrays of quoted strings, and
nothing else: no dates, no floats, no nested inline tables. A round-trip test parses what
the writer emitted with `tomllib` and compares, so the subset is checked against the real
parser rather than against the author's reading of the specification.

**3.10's calendar is the reason this is cheap rather than the reason it is right.** It
reaches end of life in October 2026, two months out. Had it been a year out the trade
would have gone the other way, and the manifest would have been JSON.

### Consequences

- **A user on 3.10 cannot install either distribution after this.** cpomdp is
  `Development Status :: 2 - Pre-Alpha` and warrantlib `3 - Alpha`, and neither has a
  pinned-version audience to strand. The published `warrantlib` 0.2.0 keeps its `>=3.10`
  metadata, so an existing 3.10 install resolves as it always did.
- **The CI matrix runs four interpreters rather than five**, which is a saving nobody
  asked for and should not be read as the motivation.
- `ruff` targets `py311` and may now rewrite code into syntax 3.10 rejects. That is the
  point of the target, and it is what stops the floor being nominal.
- The manifest's writer is ours to maintain. A field whose value is not a string or a
  list of strings needs the writer extended, and the round-trip test is what says so.

---

## ADR-046 — a run is reconciled against a declared inventory, not against a count

**Date:** 2026-08-22
**Status:** Accepted
**Extends:** ADR-042 (a check has a key), ADR-039 (the vocabulary is its own distribution)

### The question

Three strings in `.github/workflows/ci.yml` were the only thing standing between a
dropped stage and a green run:

    run_suite series_kernel    "23 registered, 23 tested here, none fired"

ADR-042 gave every check a key and said in its closing consequence that nothing yet
replaced those strings. This is what replaces them.

A count has three defects and they compound. It cannot say which check left, so a failing
build sends a reader to a diff to work it out. It is satisfied by a different check
arriving in place of the one that went, so a rename passes silently and the suite is
measuring something else under the same number. And it fails only when a person notices a
number moved, which is the kind of attention that lapses exactly when a branch is busy.

### Decision

**The inventory is declared in a file, generated from the suites.**
`research/registered_checks.toml` names each suite's entry point and every check id it
reported when the file was last written. `python -m warrantlib.manifest <path>` rewrites
it, and `--check` asks whether it is current, which is what CI runs.

**A declared check is a pytest item, whether or not the suite reported it.** warrantlib's
plugin collects the manifest as a file and yields an item per declared id. The item exists
because the manifest declares it, so a check that stops reporting still has a row, and the
row fails naming the check and the entry point it went missing from. There is no count
anywhere for a shorter run to satisfy.

**Both directions are checked.** A second item per suite fails on any id the run reported
that the manifest does not declare. A renamed check therefore reports as one drop and one
addition, which is what it is. Under the old strings a rename left `18 registered, 18
tested here, none fired` untouched and the gate passed.

**The suite runs once per session, not once per check.** `gap_series` derives `c₂` and
`c₄` symbolically and costs about half a minute. One run per declared check would have
multiplied that by twenty-nine. Whichever item runs first pays, and the rest read the
result, which is the front-loading this repository applies to its hot paths.

**The manifest is compared as text.** The declared ids can agree while the file is stale,
because the writer's own layout is generated too. Comparing parsed content would let a
change to the header never reach the file while `--check` kept passing.

**It stays out of `testpaths`.** The symbolic suites have always run in their own CI job,
and folding thirty seconds into every pull request buys nothing the separate job does not
already give. `pytest research/registered_checks.toml` is what runs them.

**Two suites are not in it.** `gap_expansion` and `predictive_truncation` take required
parameters, so their runners are not entry points a manifest can call. Neither was pinned
by the old strings either, so nothing regressed. Giving them zero-argument runners is the
work that would bring them in.

### Consequences

- **The three count strings are gone**, and with them the only gate that could pass a
  renamed check. The `symbolic` job runs pytest and a manifest freshness check.
- **70 checks are declared**, being the 23, 18 and 29 the strings asserted. That number
  now appears in a generated file rather than in three hand-maintained places.
- **A new check fails the run until it is registered.** That is the intended cost. The
  fix is to rewrite the manifest, which puts the new id in a diff where it is reviewed.
- **A suite that fails to import fails every one of its items**, with the import error on
  each. The alternative was reporting its whole inventory as dropped, which is true and
  much less useful.
- **The manifest's freshness check runs the suites a second time in CI.** Thirty seconds,
  on a job that was already the long one. Reusing the first run's results would mean the
  gate trusting the thing it is checking.
- `conftest.py` exempts manifest items from the slow-marker report. They are not on the
  pull-request path, so the marker has nothing to move them off, and the cost belongs to
  the suite rather than to whichever check happened to run first.

## ADR-047 — the world and the agent, separated and driven from outside

**Date:** 2026-08-22
**Status:** Accepted

### The seam

`World` owns the process that produces observations. `ScoredAgent` owns the model it
filters with. Neither reaches the other's parameters: `World` has no accessor returning
its model or a parameter of it, and `ScoredAgent` has no constructor slot for a `World`.

Without this the misspecification term is unmeasurable rather than small. One model
object serving as both the process and the agent's belief about it makes the two
identical by construction, which is why `research/warrant_ledger.md` pins p and p\*
separately "with a type seam making circularity impossible rather than discouraged".
That separation is what lets a defect found with p = p\* be ruled out as
misspecification.

The test asserts a negative, so the detector is tested first. Five cases plant a
reference to the world in each way one can be held (attribute, container, bound method,
closure cell) and require an object-graph walk to find it. Growing `World` a `model`
accessor fails the exposure tests. Without those probes the seam tests would pass just
as well on a broken walk.

### `ScoredAgent` composes a backend rather than subclassing `Agent`

Subclassing was considered and rejected on two grounds.

`Agent.infer_states` documents that the agent supplies its own last sampled action to
the predict step. A `ScoredAgent` takes the action from outside, so it would override
that method against its stated contract.

The second ground decided it. Inheriting publishes `sample_action`. An agent free to act
faces observations of its own making, the driven sequence stops being common across the
arms, and both divergences are then wrong with nothing raised. Composition makes the
method absent rather than present and merely unreachable.

Drift between the two was the argument for inheriting, and `InferenceBackend` already
answers it. `Agent.infer_states` is three lines of bookkeeping over a backend that holds
the filter arithmetic, so there is little there to drift.

### Degradation is wrapped, never a mismatched model

The inference axis needs a filter reading numbers the model does not declare, and
`ScoredAgent` refuses a backend built from a different model. Rather than relax that,
the degraded backends keep `model` pointing at the model being scored and hold the
substitution internally. `WrongFixedRBackend` builds its own filter over a modified
model and never exposes it. `DiagonalCovarianceBackend` wraps any backend and reports
the wrapped one's model.

Every degradation therefore carries a name rather than arriving as an anonymous
mismatch, and `backend.model` keeps one meaning.

On a fixed-noise model `WrongFixedR` at magnitude zero reproduces exact inference. On a
model carrying `R(x)` it does not, because the substitution replaces varying noise with
a constant. Magnitude zero is already a degradation there, and that is the correct
reading of "wrong *fixed* R".

### Both axes are data, and versioned

`Perturbation` and `InferenceRule` are frozen records rather than callables: a name and
what it changes, with a single magnitude. A declared set holding functions is a set
nobody can diff. Standing rule 7 asks for versioning, and both sets carry a version and
refuse a duplicate name.

Each set also refuses to be built without the cell the others are measured against.
`ConstructorSet` requires an unperturbed member, `InferenceSet` an exact one. A cross
missing either produces cells with nothing to compare against.

### The severed control loop is carried on the result

Driving a common exogenous sequence is what makes the entropy of the true process a
shared constant that cancels, leaving two divergences computable with no entropy
estimate anywhere. It also cuts the control loop. Under `R(x)` an agent choosing its own
actions steers toward low-noise regions and so changes its own inference gap, which
means a gap measured here can misrepresent the closed-loop one.

`DrivenRun.control_loop` holds that as a `ModellingChoice` with no default, so a run
cannot return trajectories without it. One field states what was chosen. A second states
where the choice is contestable.

### `World` refuses state-dependent noise, and that is registered as open

Sampling `R(x)` or `Q(x)` in the world needs an evaluation point. The filter reads both
at the predicted mean. A process that knows its own state exactly has two candidates,
the departed state and the arrived-at one, and they give different trajectories.

`World` raises rather than choosing. The choice moves every number measured under `R(x)`,
and no such number exists yet, so a decision taken now cannot be fitted to an answer. It
is settled in its own change before the scoring harness needs a world under `R(x)`.

### Consequences

- **PR-3's merge gate is met.** The no-read-path test passes, the import test holds, and
  the constructor set round-trips through the model spec.
- **Nothing is exported at the top level.** The import paths are `cpomdp.harness` and
  `cpomdp.constructors`, so no `docs/api/` page is owed until a consumer needs one.
- **The module boundary is a test rather than a convention.**
  `tests/test_module_boundary.py` walks the transitive first-party import closure of the
  seam and refuses `cpomdp.scoring`, the evaluator's module, named there before it
  exists. It refuses `cpomdp.selection` and `cpomdp.enumeration` on the same terms.
- **A world under `R(x)` is the next thing needed and it is small.** It blocks nothing in
  PR-4, whose constructor cross is fixed-`R` throughout.
- **A perturbation the filter would not read is refused rather than built.** A
  state-dependent sensor supplies its own `(C, R)` and a state-dependent process noise
  its own `Q`, so scaling the matrix either stands in for reaches nothing: the cell
  builds what `CORRECT` builds and reports a name saying otherwise. `ModelSpec.build`
  raises on that combination, on the same grounds as `InferenceRule.build` refusing an
  unhandled kind. Scaling a state-dependent model's own parameters is a separate
  question and is not answered here.

## ADR-048 — where a world reads state-dependent noise

**Date:** 2026-08-23
**Status:** Accepted
**Extends:** ADR-047 (the world and the agent, separated and driven from outside)

ADR-047 had `World` refuse a state-dependent `R(x)` or `Q(x)` rather than pick an
evaluation point with no number yet measured under either. This picks them. No number is
measured under `R(x)` yet, which is the condition that lets the choice be made on its
merits rather than fitted to a result.

### The sensor is read at the state it measures

`R(x)` is a property of the sensor at the state producing the reading, and that state is
drawn before the reading is taken. There is no ambiguity to resolve: the world knows the
arrived-at state exactly and reads `R` there.

The filter cannot. It reads `R` at its predicted mean, dropping the `½tr(H_R Σ⁺)` Jensen
term, which `CallableSensor` already documents as a deliberate first-order choice. Having
the world read at the same point would hand the world the filter's approximation and
report a smaller inference gap than the one being measured. The gap between `R(μ⁻)` and
`R` at the truth is the thing under study, so the world has to sit on the other side of
it.

### The process noise is read at the mean the step pushes forward to

`Q(x)` is the diffusion of the arrived-at state, which `dynamics.py` records as the
kernel's convention. A world cannot read it there: the arrived-at state is not known
until the diffusion it parameterises has been drawn.

Two non-circular readings were available.

**The departed state**, `Q(x_t)`, is the Euler–Maruyama discretisation of an Itô
diffusion and the standard choice in that literature. It was rejected because the filter
evaluates at the pushed-forward point, so a world reading the departed state would
disagree with the filter even where the filter's belief sits exactly on the truth. An
inference gap that cannot reach zero under perfect inference is measuring the convention
rather than the inference.

**The pushed-forward mean**, `Q(A·x_t + B·u_t)`, was taken. It is a function of the
current state and action alone, so the transition stays a well-defined Markov kernel. It
is also the exact point-mass limit of the filter's `μ⁻`, so the world is what the filter
converges to as its belief concentrates on the truth, which is the property the
decomposition needs.

The asymmetry between the two is not an inconsistency. `R`'s evaluation point is
determined by what the sensor physically measures. `Q`'s is not determined by anything,
so it is set by the requirement above.

### Consequences

- **A world under `R(x)` exists**, which the reference filter and the gated half of the
  battery both need. Nothing else in the harness changed.
- **The fixed-noise path is bit-identical.** One reading measured on the commit before
  this one, in a worktree of it, reproduces exactly: `1.0105588758439044`. A test pins
  that value rather than a tolerance around it.
- **A fixed `R` or `Q` carried as a model object rather than as a matrix now works too.**
  `World` reads `is_fixed` on both, matching what `KalmanBackend` does, where before it
  refused any non-`None` `dynamics_noise_model`.
- **`linearize` is exact here rather than approximate**, since the observation mean is
  linear in this regime and the Jacobian it returns is the map itself. A sensor with a
  nonlinear mean would need the mean map instead, and issue #21 is where that lands.
- **The evaluation points are pinned by test, not by comment.** Each is checked against
  an independently drawn sample at the point claimed, and against the same draw at the
  point not claimed.

## ADR-049 — a rationalised parameter is registered with its revision condition bounded

**Date:** 2026-08-23
**Status:** Accepted

### The instance

`κ_min = 0.1` is declared as the lower bound of the `κ` sweep, and the rule evaluating
`T` at the `κ` minimising the window width means the floor is the only part of the sweep
that reaches `T`.

The argument is that D2's second leg needs `κ ≲ 0.1` to resolve, that 0.1 sits below the
`c₆` zero at `3/13` so the window's upper edge stays defined, and that it holds the
`μ`-rule inflation at 3×. Every part of that was registered before `c₆` existed and
before `T` could be evaluated.

That is a defensible reading of an existing constraint. It is not a derivation, and no
experiment distinguishes 0.1 from a nearby value.

### The decision

**A parameter that is rationalised rather than derived is registered as such, and its
revision condition is bounded before any result exists.**

**Say the epistemic status.** The registration records `κ_min` as a plausibly weak
decision in those words. Presenting a chosen number as a forced one is the failure this
avoids, and it costs nothing to avoid.

**Bound the revision in advance.** "May need revisiting" is honest as an intention and
useless as a rule: it licenses a change after a disappointing result on the same terms as
a change after a genuine finding. So the licence is narrowed to a measurement bearing on
the argument that produced the value, and a gate outcome is excluded by name.

**Say what a revision does to results already obtained.** Nothing. The outcome under the
declared value stands as obtained, and a re-evaluation is a separate cell carrying both.
This is ADR-037's treatment of the `tanh` fire generalised: a result is not undone by
re-running it with inputs chosen after seeing it.

### The incentive is registered too

`κ_min` moves `T`, and `T` has opposite senses for the two things that read it. Lowering
`κ_min` lowers `T`, which makes GATE-D4 easier to pass and makes D1 and D2 harder to be
tests at all. Recording that is what lets a later reader check whether it was acted on,
and the absence of a direction that helps everything is a structural defence rather than
a rhetorical one.

### Where it is visible

The registration carries the declaration. `Provenance.registered` carries it to every
printed row once the `T` check exists, that field being one line on what a reviewer finds
at the registering ref.

It is **not** carried in `warrantlib`. That package is published on its own and knows
nothing about this programme beyond one historical note. A `κ` belonging to one study has
no place in a general vocabulary, and the mechanism for saying so from cpomdp already
exists. Nor is it carried in `research/registered_checks.toml`, which is generated and
would lose it on the next rewrite.

### Consequences

- **`T` is unblocked on this axis.** What remains is `σ_max` against `c₆`, and the noise
  model that `research/d2_noise_model_exploration.md` puts in question.
- **The pattern applies beyond `κ_min`.** `f`, `D` and `k_min` were declared on stated
  grounds without this treatment, and a later amendment may owe them the same three
  parts.

## ADR-050 — the code that produced a number is committed beside it

**Date:** 2026-08-23
**Status:** Accepted

### The failure this closes, which already happened

The scripts that produced the original 28 `c₄` cases were lost. Only prose survived them,
so the numbers could be read and not checked. `research.checks` exists because of it, and
`research/gate_d4_registration.md` records the loss rather than passing over it.

That fix was scoped to warranted checks. It leaves out everything that decides a *form*
rather than reporting a result: the probe that ruled an option out, the calibration that
fixed a constant, the sweep that showed a parameter did not move. Those reach registrations
and write-ups too, and until now they lived in a scratch directory.

### The decision

**A number that reached a registration, a write-up or a `PROVED` row has its derivation
committed and runnable.**

Two homes, and the difference between them is warrant rather than importance.

**`research/src/research/checks/`** for anything carrying one. It emits `CheckReport`s, is
declared in the manifest, and reconciles against it.

**`research/src/research/explorations/`** for the rest. An exploration reports no warrant
and appears in no manifest, because it has none to report. It is committed, linted, typed,
and run by a test, so it cannot rot into a file nobody executes. `explorations/__init__.py`
states the invariant and a parametrised test enforces it over every module in the package.

### A→B→C, not just C

A reviewer following how a number was reached wants the order: what was tried, what it
ruled out, what replaced it. So an intermediate step is committed too, **including one
that turned out wrong**. A superseded exploration keeps its file and gains a dated note
naming what replaced it, on the terms the protected files already work.

This has bitten twice in the work that prompted it. `research/c6_hand_derivation.md`
predicted `κ₅` contributes at `σ⁶` and it does not, and the amendment recording that is
what a reader needs to see beside the correction. `research/c6_window_exploration.md`
claimed two bias arms were equal where they are a factor of two apart, and the assertion
in the module is what caught it.

### Assert, do not only print

An exploration's claims are asserted inside it, so running it is checking it. A module
printing a table nobody verifies has recorded a number without recording whether it is
true, which is the shape of the failure the standing rule on prose already names.

### Consequences

- **A scratch file is not a record.** Anything under `/tmp` or a session scratchpad is
  gone by the time a reviewer asks, which is when it is wanted.
- **The cost is a test per exploration and a line in the write-up naming the module.**
  Both are cheap beside re-deriving a number whose method was lost.
- **It does not make an exploration authoritative.** No warrant, no manifest entry, no
  citation from a check. `research/d2_noise_model_exploration.md` puts a registered term
  in question and settles nothing, which is what an exploration is for.

## ADR-051 — ADR-048 supersedes ADR-047 on state-dependent noise, and says so

**Date:** 2026-08-23
**Status:** Accepted
**Supersedes:** the **Extends** header on ADR-048, and ADR-047's section "`World` refuses
state-dependent noise, and that is registered as open"

### What was wrong with the labelling

ADR-048 decided that `World` reads a state-dependent `R(x)` and `Q(x)` rather than
refusing them. ADR-047 had decided it refuses them. That is a reversal, and this record
keeps **Supersedes** for a reversal and **Extends** for a decision that adds to one still
standing. ADR-048 carried **Extends**.

The cost is not cosmetic. ADR-047's section reads as current, so the two records state
opposite rules with nothing between them saying which holds. The same sentence reached
`CHANGELOG.md`, where it shipped in the Unreleased notes describing behaviour the same
cycle had already reversed, and is corrected there separately.

### The decision

ADR-048 supersedes ADR-047 on this point and only this point. Everything else in ADR-047
— the seam between world and agent, the exogenous action sequence, the cut control loop —
stands unchanged, which is why this is recorded here rather than by retiring ADR-047.

Both are left in place with their original text, per the rule that a decision record is
appended to and not edited. This entry is the pointer a reader following either one
arrives at.

### Consequences

- **A reader of ADR-047's refusal section needs this entry to know it is dead.** That is
  the cost of append-only, and it is paid here rather than by a silent edit.
- **The convention is now stated where it can be checked**: reversal is *Supersedes*,
  accumulation is *Extends*. ADR-048's own header is left as written and corrected by
  this record.

## ADR-052 — the averaged inference gap is built twice, on purpose, for now

**Date:** 2026-08-24
**Status:** Accepted

### The quantity, and the two places it will live

`E_{y∼p*}[ KL(q ‖ p(x|y)) ]` is the averaged inference gap. R6 is a claim about it, and
PR-7 requires the reference filter to return it directly.

`research.checks.gap_kernel` already computes it, for the scalar GATE-D4 case, by
`scipy.integrate.quad`. It pins three conventions that were nearly lost once already:
reverse direction, `R` frozen at the prior mean, and the average taken under the true
`p*(y)` rather than the agent's plug-in predictive. It is cited from
`research/c4_hand_derivation.md` and twice from `research/gate_d4_registration.md`, and it
stands behind the 28 fitted `c₄` cases.

PR-7's version is general: any grid, any pointwise likelihood, any transition kernel, any
rung of the ladder. It is a different object with the same integral inside it.

### The decision

Build the general one in `cpomdp.reference` and leave `gap_kernel` alone. Cross-check the
two on a scalar case in a test, so a divergence between them fails rather than propagates.

Two implementations of one quantity is how numbers drift apart, and this record exists
because that is a real cost being accepted rather than overlooked. It is accepted because
the alternative is worse right now. Migrating `gap_kernel` onto the new engine edits a
module carrying registered provenance for numbers already declared, inside a PR whose
merge gate is about a filter rather than about `c₄`. A refactor there would put the `c₄`
cases behind a code path that did not exist when they were measured, and no reader could
afterwards tell which engine produced them.

The conventions are the thing that must not fork. They belong to the ladder and the
averaging, not to either implementation, and the cross-check test is what holds them
together until there is one engine.

### When it gets cut

Not on a date. When the `p*` work is finished and it is settled where that code lives and
how it splits between the library and `research/`. That question is open today: the
predictive, its truncation and its error shape are all still being decided, and PR-8's `T`
is parked on the last of them. Deciding the module boundary before those settle would fix
it against a shape that has not stopped moving.

### Consequences

- **A number is reachable by two routes**, and the cross-check is the only thing making
  that safe. If it is ever deleted or relaxed, this decision stops being defensible.
- **`gap_kernel` keeps its provenance intact.** The `c₄` cases stay behind the engine that
  measured them, which is what the derivation-code rule asks for.
- **The new engine carries no warrant on arrival.** It computes a number; nothing declares
  it until R6 registers one. Until then it is `COMPUTED`, and the word is not "certified".
- **The debt is tracked, not remembered.** A GitHub issue points here and is closed by the
  merge, not by the decision.

## ADR-053 — the two gap engines do not overlap where ADR-052 said to compare them

**Date:** 2026-08-24
**Status:** Accepted
**Supersedes:** one sentence of ADR-052, "Cross-check the two on a scalar case in a
test", and nothing else in it

### What ADR-052 assumed

That both engines compute the averaged inference gap over the same domain, so a scalar
fixed-`R` case would exercise each of them and their agreement would be evidence. The
closed form under a fixed `R` was the obvious place to meet, since it is the one
configuration with an answer known independently of either.

That configuration is unreachable for one of the two.

### What is actually true

`research.checks.gap_kernel` freezes `R̂ = R(μ)` by construction. Its agent's plug-in is
read off the declared family at the prior mean, and nothing in its interface lets a
caller supply a different one. Under a constant `R` the plug-in *is* the truth, so the
divergence is zero at every `y` and the averaged gap is zero. Measured at `3.6e-17`,
which is the quadrature's own floor.

So there is no fixed-`R` configuration in which both engines produce a number worth
comparing. One of them produces zero by construction, and a comparison against zero
tests the quadrature and nothing else.

### The decision

The cross-check runs on the state-dependent case, which is where the two genuinely
overlap. `gap_identity.engines_agree` compares them on `R(x) = 1 + x²` at four spreads
and requires agreement within `1e-10`. Measured worst is `3.2e-14`.

Its family is declared locally in the check rather than added to `gap_kernel.FAMILIES`.
That dict is a registered set, and a cross-check has no business extending one to make
itself possible.

The comparison samples a continuum of spreads, so it reports `CORROBORATED` at
`BOUNDED`. Adding spreads does not make it a universal, and a test asserts it never
claims otherwise.

### What this costs, stated rather than smoothed over

**The closed-form identity is not a cross-engine anchor.** ADR-052 implied it could
become one. It anchors `cpomdp.reference` alone, where the fixed-`R` case is reachable
and the answer is known. `gap_kernel` is tied to it only indirectly: through the three
shared conventions, and through agreeing with the other engine on `R(x)`.

That is weaker than two engines each independently matching a known answer. It is what
is available. A reader comparing the two should know that their agreement rests on a
case where neither has an independent oracle, and that the oracle exists only on the
side one of them cannot reach.

### Consequences

- **ADR-052's substance stands.** Two engines, a load-bearing cross-check, and the
  merge deferred until the `p*` work settles the module boundary. Issue #102 tracks it
  and this entry does not move its trigger.
- **A future single engine inherits the harder job.** Whichever survives has to reach
  both regimes, since the fixed-`R` case is the only one with an oracle and the `R(x)`
  case is the only one the results are about.
- **The unreachability is a property of `gap_kernel`'s interface, not a defect.** It was
  built to measure one registered family at one plug-in rule. Widening it to accept an
  arbitrary plug-in would change a module carrying registered `c₄` provenance, which is
  the thing ADR-052 declined to do.

## ADR-054 — the append-only rule is enforced, and its exception is in the log

**Date:** 2026-08-25
**Status:** Accepted
**Extends:** ADR-038 (an ADR keeps the vocabulary it was written in)

### What was unenforced

`DECISIONS.md` and `research/gate_d4_registration.md` are append-only by rule. The rule
lived in a gitignored `CLAUDE.md` and in whoever was reading it at the time. It had
already been broken once, by two in-place edits to the registration on 2026-08-17, and
was caught by review rather than by anything that runs.

ADR-038 records the principle for vocabulary: a record written before a rename keeps the
words it was written in, because that is what dates it. The same argument covers every
line in these files, and nothing acted on it.

### The decision

`protected_files.py` holds the list and reads a zero-context diff over it, failing on
any removed line. Pure additions pass, including an insertion in the middle of a file,
which is what appending within a section looks like to a diff. The `commit-msg` hook
runs it over the index. The `append-only` CI job runs it over the pull request.

**Landed means reached the branch the change will land on, not committed.** A block
added in one commit and reworked in the next has no reader who could have seen the first
version. Blocking that would train everyone to reach for `--no-verify`, which is how a
hook stops being enforcement. So the comparison point is the merge base, read from
`GITHUB_BASE_REF` on a pull request so a stacked branch is judged against the branch
below it.

**The exception is `[in-place]` in a commit subject.** A subject rather than a trailer,
because commit messages here are one line. A marker rather than `--no-verify`, because
it survives into the log where a reviewer meets it. One marked commit clears the branch:
the diff against the merge base is cumulative, so reading only the message being written
would oblige every later commit to carry the marker, and a plain append would have to
call itself an in-place edit to get recorded. CI reads the same subjects.

**A declared path that git is not tracking fails the check.** `git diff` exits zero on a
pathspec matching nothing, so an entry naming a file that moved or never landed would
guard nothing and say nothing about it. A record joins the list in the change that lands
the record, which is why `research/spinello_stilwell_rung.md` is not in it yet.

### Consequences

- **`CONTRIBUTING.md` states the rule**, which `CLAUDE.md` could not: it is gitignored,
  so the only durable statement of a repository-wide rule was one nobody outside this
  machine could read.
- **The exception is visible and countable.** `git log --grep` finds every in-place edit
  ever taken. `--no-verify` leaves nothing behind at all, which is what made the 2026-08-17
  breach invisible until someone read the diff.
- **Rename detection is deliberately not asked for.** The pathspec names the source and
  not the destination, so a protected file moved out from under the rule reads as a
  whole-file deletion. Failing on that is the answer wanted, and `-M` would not change it.
- **The hook is only as real as its installation.** `pre-commit install --hook-type
  commit-msg` is a per-clone step, and this repository had no hooks installed at all when
  the check was written. The CI job is what does not depend on that.

## ADR-055 — the symbolic suites leave the pull-request path

**Date:** 2026-08-25
**Status:** Accepted
**Extends:** ADR-045 (the manifest is TOML and says how to regenerate itself)
**Supersedes:** ADR-045's sentence that `--check` "is what CI runs"

### What CI was paying

The `symbolic suites (c₂ gate)` job ran on every pull request and took between 24 and 32
minutes. Its two steps, measured on the run of 2026-08-24:

    symbolic suites, reconciled against the manifest   15m 59s
    the manifest is current                            15m 42s

The second number is the whole finding. `warrantlib.manifest --check` imports each
suite's entry point and calls it, because a suite's check ids are only readable by
running it. So the job derived `c₂`, `c₄` and `c₆` twice per pull request, and the second
derivation existed to answer questions the first had already answered: the pytest plugin
reconciles both directions as it runs, failing by name on a declared check that stopped
reporting and failing on a reported check nobody declared.

One question was genuinely only the second step's. `--check` compares the file as text,
so a layout the writer no longer emits counts as stale even when every id is right. That
question needs no suite at all. It is answerable from what the file already declares.

### What the job could catch, and when

Nothing under `research/src` imports `cpomdp`. The suites read no data and take no
measurement. Their inputs are the research package itself, `warrantlib`, and sympy, and
the job syncs `--locked`, so a sympy change can only reach it through `uv.lock`.

The derivations are fixed. Re-deriving fixed inputs is not a check, and a pull request
that touched a hook script and a YAML file paid half an hour to re-print `c₆`.

### The decision

Three changes.

`warrantlib.manifest` gains `--layout-only`, which compares the file against what the
writer would emit from the ids **already declared in it**, running nothing. It answers
the layout question in 70ms and says in its own output that it did not ask the other one.

That check moves to the `lint` job, where it runs on every pull request unconditionally.

The `symbolic` job gains a path gate. A new `scope` job diffs against the merge base with
`main` and the symbolic job runs only when `research/`, `packages/warrantlib/`,
`uv.lock`, `conftest.py` or this workflow changed. On a push to `main` it always runs.

The last two are in the filter because both reach the run. Rootdir is the repository
root, since `pyproject.toml` carries the ini table, so the symbolic invocation loads the
root `conftest.py`, which edits `sys.path`, declares `pytest_plugins` and special-cases
`research/registered_checks.toml` nodeids by name. `ci.yml` carries the job's command,
its interpreter pin and its action versions. The diff is taken with `--no-renames`: with
detection on, a file moved out of `research/` prints only its destination, so the move
that changed the suite set would read as a change somewhere else and the job would sit
out.

**`pyproject.toml` is deliberately outside the filter, and that leaves one channel
open.** A dependency edit that does not reach `uv.lock` fails `uv sync --locked` before
anything runs, and the job names the manifest on its own command line rather than
reading the `warrant_manifest` ini setting. `[tool.pytest.ini_options]` is neither of
those. A future `filterwarnings = ["error"]` or an `addopts` edit reaches the symbolic
run with nothing gating it, and is first seen on the push to `main`. Adding `pyproject.toml` to the
filter would re-tax every pull request that touches a ruff ignore, which is the cost
this exists to remove, so the residual is named here rather than closed. Running the job
under `-c research/pytest.ini` would close it, and would cut `conftest.py` out of the
run as well, at the price of diverging from the invocation `pyproject.toml` documents
for local use.

### Consequences

- **A pull request that does not touch the derivation pays nothing**, and the job reports
  as skipped rather than as passed. A green tick for work that did not happen is the
  failure this shape avoids.
- **The filter is exhaustive rather than approximate.** It rests on the suites importing
  no `cpomdp` and on `--locked` pinning sympy. Both are checkable, and either one ceasing
  to hold makes the filter wrong, so a change that makes a suite read library code has to
  come here first.
- **`--layout-only` states its own boundary.** Asked about a stale file it says the
  layout drifted and that whether the declared ids are current is a question it does not
  ask. A flag that quietly answered less than `--check` while printing the same words
  would be worse than the duplicate run it replaces.
- **Id currency is no longer asserted on a pull request that skips the job.** That is the
  trade, and it is safe for the reason above: if `research/` did not change, the ids
  cannot have. A push to `main` runs the full reconciliation, and so do the release and
  merge runs through `tests/test_check_ids.py`.
- **`tests/test_check_ids.py` does not close this gap.** It runs the suites on the merge
  and release paths and reads their ids, and it asserts nothing about their outcomes. The
  pytest plugin is what turns a `FIRED` check into a failure, and only the `symbolic` job
  runs it. A change to `research/checks/` still has to keep that job green.
- **ADR-045's line about `--check` is dead.** It says `--check` is what CI runs. Nothing
  in CI runs the bare form now: `lint` runs `--check --layout-only` and the symbolic job
  reconciles the ids through the pytest plugin. The line keeps its place, per the rule
  this record is itself subject to, and this entry is the pointer a reader of it arrives
  at. Every other statement of the same claim was in a file that is not a record, and
  those were corrected in place.

## ADR-056 — the Spinello–Stilwell rung: five rungs, and the pole is a units question

**Date:** 2026-08-24
**Status:** Accepted
**Extends:** ADR-052 and ADR-053 on the gap the ladder measures

`research/spinello_stilwell_rung.md` holds the open questions this entry rests on, the
routes that would settle them, and the tests each route owes. This is the decided part.

### The ladder declares five rungs, not four

PR-7 scheduled four: plug-in `R(μ⁻)`, Spinello–Stilwell iterated, belief-smoothed
`E[R(x)]`, and the exact reference. Between the first two the ladder changes two things
at once. Rung one's posterior precision is `P⁻⁻¹ + (1/σ)∇ᵀh∇h`. The paper's single-step
filter (36e) adds `(1/2σ²)∇ᵀσ∇σ` on top of that, and the iterated form (35) then also
runs Gauss–Newton to convergence. An improvement measured across that pair cannot be
attributed to either mechanism.

(36) is (35) at a budget of one, so it costs almost nothing to declare it separately.
Five rungs give two adjacent differences that isolate distinct mechanisms: the
derivative-of-covariance terms, and the iteration.

This is decided now because R7 is an ordering over the declared set, and a rung added
after an ordering is seen is what standing rule 7 refuses.

### The warrant for dropping `1/ln σ` is invariance, not convergence

Under a rescaling of the observation, `o → λo`, every term in `s`, `R⁽ᶥ⁾` and `Ū` returns
to itself except the log-determinant term, which becomes `1/(ln σ + 2 ln λ)`. Seven of
eight terms are invariant and that one is not, checked symbolically.

The ladder reports a divergence between distributions over the state, which no choice of
observation units can move. So that term is the sole place an arbitrary unit choice
enters an estimator whose output is unit-free. It is the same objection the standing
prohibition on entropy subtraction rests on, and it is checkable by a rescaling test
rather than argued.

An earlier justification, that `R⁽ᶥ⁾` shapes the Gauss–Newton path and not its fixed
point, is true and much weaker. It licenses changing the term. It does not say the term
should not have been there.

### No guard is written until the units question is answered

The pole at `σ = 1` moves with the unit choice: rescaling puts it at `σ = λ⁻²` while
leaving the fixed point where it was. A `λ` that puts a family's whole reachable `σ` on
one side of the pole removes the hazard by construction and costs nothing.

That choice is a declaration, not a tuned parameter, and it is declared before any
number exists. An undeclared choice that happens to avoid a pole is what standing rule 2
exists to catch.

A guard is written only for a family where no such `λ` exists. Route 2 in the companion
file decides which families those are, and the registered ridge is the case to check
first: `σ(μ*) = 2R₀`, so `R₀ = 0.5` puts the pole on the operating point.

### A non-convergent step is `VOID`, not a number

`Ū` is evaluated at the current iterate, so exhausting the budget returns a wrong
covariance as well as a wrong mean and both halves of the divergence move by an
unmeasured amount. Approaching the pole from above the step goes to zero, so a truncated
run returns the prediction looking like a converged posterior. That failure is silent,
which is why it is routed rather than printed.

### Consequences

- **The paper's `R` is never spelled `R` in cpomdp.** It is a Gauss–Newton Hessian and
  the tree already uses `R` for what the paper calls `Σ`. The companion file carries the
  full three-way notation swap.
- **`p ≤ n` is asserted rather than inherited.** The paper assumes it throughout and
  cpomdp enforces it nowhere.
- **The rung has no external numerical validation.** Every figure in the paper's §IV
  uses the single-step filter, and the iterated scheme is derived but never simulated.
  The
  `∇σ = 0` reduction to the ordinary Kalman update is the only independent check
  available, so the test suite carries more here than it does for the other rungs.
- **Rung one cannot see the mechanism under test.** The term it discards is non-zero
  exactly when the noise varies with the state. Reportable in Paper 2 Part 2, stated
  carefully: the filter's posterior information and expected free energy's epistemic
  term are related and are not the same object.

## ADR-057 — the shipped Spinello–Stilwell rungs run `R_mod`, and no unit choice is declared

**Date:** 2026-09-04
**Status:** Accepted
**Extends:** ADR-056, whose units question this answers
**Supersedes:** two sentences of ADR-056's section "No guard is written until the
units question is answered" — "That choice is a declaration, not a tuned parameter,
and it is declared before any number exists", and "A guard is written only for a
family where no such `λ` exists" — and nothing else in it

`research/spinello_stilwell_hand_derivation.md` is the scan of the notebook typed up.
Its step 6 names the block that fails, step 8 argues the deletion is surgical, and
`research.spinello_stilwell.repair` runs the three tests step 10 names. This entry is
the decision those were written for.

### Decision

Rungs (36) and (35) of the ladder run the paper's scheme with the `r₃` block of (35d)
removed. (35c), (35e) and the objective (18) stay verbatim. The curvature is

    R_mod = (∂r₂)ᵀ(∂r₂) = (1/σ) bᵀb,    b = ∇h + (ζ/2σ)∇σ

and the deleted term is `∇ᵀσ∇σ/(4σ² ln σ)`, the fourth printed term of (35d), whole.

The rung is named as modified wherever it appears: the Spinello–Stilwell iterated
filter with a documented modification, declared at first use in Paper 2. It is not
the paper's equation verbatim and nothing in the tree will say it is.

No observation unit choice is declared. The shipped ladder runs at native units and
nothing in it depends on that, which is a test rather than a promise. `λ` stays a
sweep variable in `research/` and is not a convention anywhere.

### Why the deletion is justified

Step 8 of the derivation, point by point, with what each point rests on.

- **The fixed point is where the gradient vanishes.** The iteration is
  `x ← x − M⁻¹g`. `M` steers the route and `g` picks the destination. Changing `M`
  to any positive-definite matrix leaves the set of fixed points alone.
- **`ln σ` cancels out of the gradient.** Step 4 has `r₃ ∂r₃ = ∇σ/2σ`, real for every
  `σ > 0`, and that is what (35c) prints. `s` stays verbatim, so `g` does.
- **The deleted object is the curvature block alone.** `(∂r₃)ᵀ(∂r₃)` is
  `∇ᵀσ∇σ/(4σ² ln σ)` and nothing in it cancels. It is the only place `ln σ` survives
  into the scheme.
- **`Ū` is built from `s` alone.** No `ln σ` reaches (35e), so the posterior covariance
  `P₊` is the paper's, unchanged.
- **The objective is untouched.** The mode of (18) is where it was.

Net: same `x̂`, same `P₊`, same `g`, same gap in nats. Only the path of iterates
changes. `M = P⁻¹ + R_mod` is positive definite for every state, since `P⁻¹` is and
`R_mod` is a real square, so every step is a descent direction of (18). Step 8 writes
that as "provably descends". Monotone descent without a line search is not claimed
here and does not need to be.

What it costs is the name. The iteration is a modified-metric Newton with the
Gauss–Newton fixed point and it is not literal Gauss–Newton on the residual vector
(20). Where `σ < 1` a repair of some kind was compulsory anyway: the printed `R`
claims to be `∇ᵀr∇r` and every real such matrix is positive-semi-definite, and the
printed one is not.

Three measurements back the argument, all asserted in
`research.spinello_stilwell.repair` and in `tests/test_spinello_stilwell_repair.py`.

- **The deleted block is the whole of the unit dependence.** Under `o → λo` every term
  of `s`, `R_mod` and `Ū` returns to itself and the deleted block does not, checked
  symbolically in route 1. Run to convergence the printed scheme is already unit-free,
  since an additive `ln λ` in the objective cannot move a root. At a budget of one the
  printed estimate spans `2.955e-05` across four unit choices and the modified spans
  `1.11e-16`. A budget of one is rung (36), so the term made a rung the ladder
  **reports** depend on the observation's units. The ladder reports a divergence
  between distributions over the state, which no unit choice can move. This is the
  warrant ADR-056 named, now measured.
- **The reduction to Kalman is exact either way, and only one variant can take it.**
  With `∇σ = 0` both variants reproduce the ordinary Kalman posterior to `0.0` in mean
  and variance at every unit choice tried. At `σ = 1` with `∇σ = 0` the printed block is
  `0/0`, so the printed scheme cannot be evaluated in the one regime that has an
  oracle. The modification can.
- **The change is confined to where it was argued for.** Above the pole the two
  curvatures agree to a relative `7.24e-08` by `σ = 10⁶`, and the difference falls
  monotonically over six decades. Below the pole the printed curvature is negative
  and the modified one is positive.

Two further points from the scan's last page.

- **The paper never cashed the term in.** Every figure in section IV uses the
  single-step filter (36), and its stated constants appear to put `σ < 1` throughout
  figures 2 to 4. That arithmetic is route 7 and is not done. Until it is, this point
  is a reason to expect the deletion costs the paper's results nothing, and not a
  result.
- **Why repair rather than derive fresh.** A published derivation exists, in a known
  lineage back to Bell and Cathey's iterated filter. One declared change to it has a
  cost that code can monitor and tests can pin. A scheme of cpomdp's own would have
  neither the citation nor the single named difference.

### Why `λ` is not declared

ADR-056 held the guard until the units question was answered, and the answer is that
the shipped scheme has no pole to clear. The one term that moved under rescaling is
the term removed. A unit convention declared to keep `σ > 1` would be a parameter with
no observable effect on any number the ladder reports. Standing rule 2 objects to a
choice that is made and not declared. It has nothing to say about a choice that is
absent, provided the absence is checked, which is the invariance test on the shipped
rung.

The printed scheme still runs in `research/`, where the pole is the object of study.
Route 3 probes the two-sided failure near `σ = 1`, route 5 exhausts the budget against
it, and route 7 evaluates the paper's constants in the paper's units. All three need
the pole reachable, so a clearing `λ` there would defeat the measurement. Route 1's
empirical half sweeps `λ`. None of them declares one.

Route 2's table stays as what it is: the measurement that `λ = 4.0195` would have
cleared every declared family but `exp(x)`. It decides nothing now, and it records that
a units-only repair was available for five families and unavailable for the sixth.
`exp(x)` was what made the units answer insufficient on its own. With the block gone
it needs no box and no guard for the pole, and its vanishing noise as `x → −∞` is a
property every rung shares rather than one of this scheme.

### Consequences

- **Q2, Q3 and Q4 of the companion file close for the shipped rung.** Q3's two-sided
  failure remains a true statement about the printed scheme and is measured in
  `research/` by route 3 rather than guarded in the tree.
- **Rung (36) is the single `R_mod`-preconditioned step.** It is unit-free at budget
  one, which the paper's (36) is not. Route 6 compares it against rung one on that
  footing.
- **Route 1's empirical half changes purpose.** It was to decide the modification. It
  now measures what the modification cost against the printed scheme, at a declared
  budget, and it stays in `research/`.
- **The rescaling invariance of the shipped rung is a required test**, on the same
  terms as the fixed-`R` Kalman agreement: a reported gap that moves under `o → λo` is
  a defect, whatever the budget.
- **The iteration count is unit-free too.** Every term the modified iterate touches is
  invariant, so the path of iterates is, and so is its length. The printed scheme's
  count was not (6, 9, 8, 8 across four unit choices). RFC-001's accounting inherits a
  per-decision cost that does not depend on the observation's units. This follows from
  the invariance and has not been measured on its own.
- **Still owed, unchanged.** The iteration budget and the tolerance, which ADR-056
  routes to a measurement. `VOID` for a non-convergent step stands, since a positive
  definite `M` guarantees a descent direction and not convergence within a budget.

## ADR-058 — the Spinello–Stilwell rungs run 64 iterations at a tolerance of `1e-12`

**Date:** 2026-09-04
**Extends:** ADR-056, whose budget and tolerance this supplies, and ADR-057, whose
scheme these numbers were measured on

ADR-056 left the iteration budget and the tolerance as arguments and routed them to a
measurement. `research.spinello_stilwell.budget` is that measurement. This entry is the
declaration it was run for.

### Decision

Rungs (36) and (35) run at a budget of **64 iterations** and a tolerance of **`1e-12`**,
relative to the prior standard deviation. A step smaller than `tolerance * sqrt(P)` ends
the run. A run that spends the budget reports `VOID`, per ADR-056.

Rung (36) is unaffected. It is (35) at a budget of one by definition, and a budget
declared for the iterated rung does not reach it.

The rung obtains `R'` by automatic differentiation and never by a finite difference.
The tolerance is only reachable if it does, which section three below measures.

### Why 64

The counts come from `research.spinello_stilwell.budget`, run with

    uv run --no-sync python -m research.spinello_stilwell.budget

Over the declared spreads, curvatures from `0.1` to `100`, and readings placed in
predictive standard deviations off the prior mean, the modified scheme at `1e-12` needs

| where the reading sits | curvatures `0.1`–`100` | the declared families |
| --- | --- | --- |
| the bulk, out to two predictive spreads | 23 | 16 |
| the box edge, nine predictive spreads out | 65 | 124 |

Nine is `measure_truncation`'s half-width multiplier, in plug-in predictive spreads,
which is where the reference truncations the registration quotes were measured. The rung
is called at every node inside it, so at readings up to thirteen off the prior mean. The
convergence survey's ten iterations are not a comparable number: it reads at two *prior*
spreads, which is at most `0.6` off the mean.

64 is a little under three times the worst count in the bulk, which is where the
predictive mass is. At the box edge it is not enough, and the first consequence below
says which cells that voids.

**The budget is a cap and not a cost.** A convergent run stops at its tolerance, so
raising the cap changes nothing about a cell that already converges inside it. What it
changes is the price of a run that will report `VOID` anyway, and the gap calls the rung
once per observation node, so that price is `64 K` evaluations rather than `500 K`.

**128 would convert both voided nodes and is rejected on the trade.** It doubles the
price of every run that still reports `VOID`, and what it buys is two nodes carrying
`2.6e-18` of the centre's predictive weight, which cannot move a reported gap at any
precision the ladder works to. RFC-001 measures per-cycle compute, and a cap sized by
the least-weighted node in the box is how a baseline gets bloated.

**No cap converts every cell.** One offset past the box edge, at thirteen predictive
spreads, `1.5 + 0.5 sin(x)` at spread `0.30` settles into a stable two-cycle and stays
in it: the iterate alternates by `2.83` for as long as it is run. That is outside the
registered box and outside what this declaration covers. It is the reason the budget is
paired with a routing rather than chosen large enough to be safe.

### Why `1e-12`

At `1e-12` the estimate is within `2.1e-15` of the fully settled one, which contributes
around `1e-29` to a divergence. The smallest gap the ladder has to resolve is the top
rung's, and rung one's is already as small as `1e-14` on the declared grid. A
convergence error has to sit far below the quantity being measured, and this one sits
fifteen decades below the smallest measured so far.

`1e-14` costs four more iterations in the bulk, 27 against 23, and buys nothing that
reaches a reported gap. It also sits at the arithmetic's floor for this iteration, which
is where the derivative condition bites. A central difference at `1e-6` carries about
`1e-11` into the iterate, and five of the thirty declared cells then ran to a budget of
200 without settling: the run was measuring the difference rather than the scheme. At
`1e-12` there are two decades of room above that floor for an exact derivative and none
for a differenced one, which is why the condition is part of the decision rather than
advice attached to it.

`1e-8` would be defensible on today's numbers and is rejected on timing. The gaps it
would have to be small against are the higher rungs', and none of them has been measured.
Choosing the loosest tolerance that looks safe before the quantity exists is what
standing rule 2 objects to.

### Consequences

- **Two nodes at the box edge exhaust the budget, and one of them is a declared cell.**
  At nine predictive spreads `1.5 + 0.5 sin(x)` at spread `0.30` takes 124 iterations
  and `kappa = 100` takes 65, so both report `VOID`. `R` is bounded and periodic on the
  first, so the iterate crosses several periods of it before it settles. That is a
  declared family at a declared spread and not a bracket. No curvature range is
  registered anywhere, so `kappa = 100` is the bracket. Both nodes carry `2.6e-18` of
  the centre's predictive weight, so neither carries anything into the average. What a
  voided node does to a reported gap is a decision PR-7 owes, and this entry does not
  take it: dropping the node, voiding the gap and widening the budget are all still
  open.
- **`test_the_rung_s_open_decisions_stay_arguments` changes meaning.** The research
  probes keep taking a budget and a tolerance as arguments and printing what they ran
  at. What is now settled is the rung's own, and it is declared here rather than
  defaulted anywhere.
- **The derivative condition reaches PR-7's rung contract.** A rung built on
  `StateDependentNoiseLikelihood` differentiates `observation_noise_fn`, and JAX does
  that for nothing. A rung that differences instead cannot hold this tolerance and the
  declaration would have to be reopened.
- **The per-decision cost is bounded and attributable.** 64 iterations per observation
  node is the ceiling an iterating rung can cost, which is what RFC-001 needs to
  separate the rung's compute from the ladder's.

## ADR-059 — the evaluator returns two divergences, and a difference is read at its own error

**Date:** 2026-09-05
**Status:** Accepted
**Extends:** ADR-047, whose seam these terms are measured across

`research/fep_falsification_battery.md` section C is the contract: two divergences
that separate what a model gets wrong from what its filter gets wrong, on a cross the
seam of ADR-047 makes measurable. `research/warrant_ledger.md` section 4 is the
reporting discipline. `research/three_term_decomposition_flow.md` is the record of
how the code arrived at what follows, one section per change. This entry is the
decisions.

### The result has no slot for an entropy

`Decomposition` carries `misspecification` and `inference_gap` and nothing else. The
ledger's rule against reaching a term by subtracting `H(p*)` could have been a
docstring. It is instead a type with no field to put an entropy in, and a test that
pins the field list, so the prohibition is unrepresentable rather than discouraged.
Under a common exogenous action sequence the entropy is a shared floor that cancels
between cells, and the two divergences are computable without it.

The one place it is estimated is `cpomdp.additivity`, a separate module holding a
separate object. `AdditivityCheck` takes its `EntropyEstimator` explicitly, measures
`E[F]` directly from the cell filter's belief rather than through the identity, and
reports a residual bounded by four terms, `δ_F + δ_H + δ₁ + δ₂`. A worked case closes
at `δ_F = 0.01` and fails to close with `δ_F` dropped, nothing else changed, which is
the under-bounding the ledger corrects.

### Nothing is subtracted on the way to a small number

Both terms are Gaussian KL divergences and one primitive serves both. The textbook
form takes `n` from a trace and one log-determinant from another, so a divergence that
is truly zero reads as rounding noise of either sign, and a reading below `1e-12`
then cannot say which of two Gaussians moved. `gaussian_kl` sums squares and
`λ − 1 − ln λ` over the eigenvalues of `cov_b⁻¹ cov_a`, each non-negative on its own,
with the last evaluated by its series below `1e-4` where it too is a cancellation.
Identical Gaussians read exactly `0.0` on a correlated covariance. The inference gap's
average over the true predictive has a closed form and is built the same way.

### The conditioning and the ratio travel with the number

A `CellScore` carries a `RunConditioning`: the condition number of every predictive
and posterior covariance the terms factored, per step, read by one function
`cpomdp.diagnostics.condition_numbers` that the rollout now reads too. A cell with
exactly one axis at the reference carries a `Separation`, which is the pinned term,
the moving term and their ratio. The ratio is the claim. `render` prints no separation
without its ratio and its worst conditioning on the one line, and `score_cross`
refuses to name a separation on a cross with no cell where both terms move, since a
contrast needs somewhere both can.

### The cross is enumerated, and that is the warrant

`build_cross` builds `|model axis| × |inference axis|` cells and carries a
`ProductCompletenessCertificate` naming both axes and both versions, at `PROVED`. This
is the Route 2 purchase from the build plan and it was done first inside the change.
It warrants the enumeration, not the numbers: every cell value the evaluator prints is
`COMPUTED` until PR-5 registers what it is measured against, and the flow record says
so where it reports a reading.

### A difference is read at the error on the difference

Two quantities scored against one reference share its error, and the shared part
cancels in their difference. The ledger asks that the resolution threshold be that
error, propagated, and pre-registered, rather than the sum of the two bars.

`cpomdp.resolution` splits a `Bar` into a *signed* common-mode part, the shift the
reference's error puts on the value, and an own part. A difference's bar carries the
difference of the common-mode parts and the sum of the own parts. The sign is the
decision: sensitivities of one sign cancel and sensitivities of opposite sign do not,
and an unsigned split would under-bound the second case the way a three-term bound
under-bounds the additivity residual. `resolution_threshold` takes two `Bar`s, and a
`Bar` has no value field, so the threshold is pre-registered by construction rather
than by discipline. `resolve` reports `BELOW`, `ABOVE` or `NOT_RESOLVED`, the third
being the measured tie the ledger asks the fidelity ladder for, neither confirmation
nor refutation. Two `EXACT` values that agree read as not resolved, since a difference
of zero exceeds nothing.

The module is a leaf. It imports no first-party module and the boundary test asserts
that, because it is built once for the evaluator and for the things that may not
import the evaluator: the reference filter, and Paper 3's G10. It is its own module
rather than part of `cpomdp.diagnostics` because it is not a diagnostic of a model.

### Not decided here

Which magnitudes the perturbation axis runs at, which seed set a stable reading is
declared across, and what any cell is registered as measuring. Those are PR-5's, and
nothing printed by this change is a result until it declares them.

## ADR-060 — a voided reading leaves the gap by weight, and the report carries the weight

**Date:** 2026-09-10
**Status:** Accepted
**Extends:** ADR-056, which routes an exhausted budget to `VOID`, and ADR-058, which
declared the budget and left open what a voided node does to the reported gap

ADR-058 found two nodes at the observation box's edge that exhaust the declared budget
and named three ways the sweep could treat them: drop the node, void the gap, or widen
the budget. This entry takes the first and says what the report carries so a reader can
tell it happened.

### Decision

A rule may answer `Void` in place of a belief. `ApproximatePosterior` returns
`GridDensity | Void`, and `Void` carries the iterations spent and one line of detail.

`averaged_inference_gap` treats a voided node as unmeasured. Its divergence is NaN in
`divergences`, its index is set in `voided_nodes`, and its predictive weight is summed
into `voided_mass`. `value` is the expectation over the readings the rule answered,
normalised to their own mass. When the rule answers nothing, `value` is NaN.

`voided_mass` is reported beside `predictive_mass` and on the same terms. A check that
wants the strict figure asserts it is zero.

### Why by weight

A clipped observation box already leaves part of `p*` unmeasured, and the report
handles it by normalising to the captured mass and printing what was captured. A
voided node is the same shape of gap in the same average. One convention for both
means one reading of `value`: the conditional expectation over what was measured,
with the unmeasured weight printed next to it. The strict figure is recoverable from
the report, and a caller that needs it says so in one assertion.

The weighing does not need the rule. `log p*(y)` and the edge ratio come off the prior
and the likelihood alone, so a voided node still counts toward `predictive_mass` and
`worst_edge_ratio`. Only the divergence is missing, and that is the only thing the
rule was asked for.

### Why not the other two

**Voiding the gap** would let a node carrying `2.6e-18` of the centre's weight erase
a sweep's reading of the rule where its weight lies. The reported figure would then
depend on where the box edge fell rather than on the rule.

**Widening the budget** was rejected in ADR-058 on the trade. Route 5 also found the
bounded family stops converging at all past the box edge, so no budget closes the
case. The routing has to exist whatever the budget is.

### Consequences

- `InferenceGap` gains `voided_mass` and `voided_nodes`. `value` changes meaning by
  exactly the mass those report, and not at all for a rule that always answers.
- The hot path is unchanged. Weighing a node costs what it did, plus one type check
  per node outside the compiled functions.
- A rung's `VOID` is observable end to end: the rung reports it, the sweep counts it,
  and the report prints its weight. That closes the question PR-7a left for PR-7 and
  is recorded under Q7 of `research/spinello_stilwell_rung.md`.
- `tests/test_reference_gap.py` pins both cases: a single voided node leaves the
  average by its weight, and a rule that declines every reading leaves no value.
