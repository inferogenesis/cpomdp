# cpomdp build plan / progress tracker

Tracked, surviving replacement for the old `.claude/cpomdp_v0.4_build_plan.md`,
which was gitignored and didn't transfer between sittings. Authoritative
decisions still live in `DECISIONS.md` (ADRs); this file is the running
checklist of what's built and what's next.

Conventions: `[x]` done, `[ ]` open, `[~]` partial. Phases follow ADR-012.

---

## v0.4 — FFG message passing (ADR-012)

Generalise the Kalman/EFE machinery to a Forney factor graph so the E. coli
chemotaxis network — with its shared `CheA` node feeding a fast (CheY-P/motor)
and a slow (CheR/CheB methylation) branch — is representable. Canonical
(information) form; from-scratch JAX; RxInfer narrowed to oracle-only;
hand-authored schedule.

### Phase 0 — scaffolding decisions — DONE (commit `7a2713c`)

- [x] The four ADR-012 choices settled: from-scratch-JAX, RxInfer-as-oracle,
      canonical-form messages, hand-authored schedule.

### Phase 1 — `CanonicalGaussian` message algebra — DONE (2026-06-25, pending commit)

The FFG wire payload: `src/cpomdp/ffg/message.py`, spec in
`tests/test_ffg_message.py`. 255 tests green, `ty` clean, `ruff` clean.

- [x] Scaffold — construct/validate `(Λ, h)`, `ndim`, pytree flatten/unflatten.
- [x] `__add__` — factor product as elementwise sum; jit-safe shape guard;
      builds via the no-validate seam (no inversion on this path).
- [x] `to_moment` — `(mean, cov)` readout via solve/inv; positive-definite
      guard; `h` = potential = information-vector naming pinned in the docstring.
- [x] `marginalize` — Schur-complement elimination; kept indices ascending;
      positive-definite guard on the eliminated block; only that block inverted.
- [x] `_unchecked` — shared non-validating constructor (hot-path-lean,
      tracer-clean); `tree_unflatten` de-duplicated onto it.
- [x] Supporting: `_validation.py` symmetry check made trace-safe (latent bug
      that blocked construction under `jit`); associativity oracle relaxed to
      `allclose` (IEEE addition isn't associative); `cspell` dict += `elim`,
      `Schur`.

Parked open question: a `from_moment` / moment-form constructor (none in v0.4;
moment form is readout-only via `to_moment`).

### Phase 2 — factor nodes + chain = Kalman numerical-identity gate — IN PROGRESS (RxInfer oracle open)

Tier-1 linear-Gaussian factor nodes (`src/cpomdp/ffg/factors/linear_gaussian.py`)
plus the chain backend (`src/cpomdp/ffg/chain.py`); specs in
`tests/test_ffg_factors.py` and `tests/test_ffg_chain.py`. Registered as JAX
pytrees. 290 tests green, `ty`/`ruff` clean.

- [x] **Observation factor** — `GaussianObservation.message(y)` = the likelihood's
      information form `(CᵀR⁻¹C, CᵀR⁻¹y)`; the update is `belief + message`. Oracle:
      moment-form measurement update.
- [x] **Transition factor** — `GaussianTransition.predict(message, control_term)`:
      build the joint over `[x, x']`, fold the message into the x block, marginalize
      x out. Oracle: moment-form predict `AΣAᵀ+Q` / `Aμ+b`. PD-Q only — the
      information form inverts Q, so no deterministic (`Q=0`) transition.
- [x] **Chain backend** — `src/cpomdp/ffg/chain.py`: `ChainBackend` wires
      `lift → predict → update → to_moment` into `infer_states` (satisfies the
      `InferenceBackend` protocol). Factors front-loaded in `__init__`; the moment→
      canonical lift builds via `_unchecked`, so the eager loop's only validation is
      the output `Belief` — same per-step cost as `KalmanBackend`. Tier-1 fixed only
      (state-dependent R(x)/Q(x) rejected → Phase 2.5; `Q=0` rejected as the
      info-form divergence).
- [x] **KEYSTONE GATE** — `tests/test_ffg_chain.py` (18 tests): numerical identity
      (atol 1e-7) vs `KalmanBackend` over sequences, dims (1,1)→(4,3), with/without
      control; plus an independent NumPy scalar-filter oracle. (Tolerance note below.)
- [ ] RxInfer oracle check on small graphs (behind the `rxinfer` marker).
- [x] jit/grad/vmap smoke tests as gates on every new public inference entry
      (`TestChainBackendTransforms`).

Tolerance note: the keystone is *numerical* identity (atol 1e-7), not literal
bit-for-bit — info-vs-moment form inverts/re-inverts. ADR-012's "byte-identity"
wording amended accordingly (2026-06-26).

### Phase 2.5 — `ChainBackend` R(x)/Q(x) parity — DONE (2026-06-28)

Before v0.4 ships, the FFG chain path reaches feature parity with `KalmanBackend`
on state-dependent noise (decided 2026-06-26; recorded in ADR-012). Phase 2 ships
fixed-matrix only (rejected at construction) to keep the keystone clean; this phase
lifts that restriction via the same *linearize-at-μ⁻ plug-in* Kalman already uses.

- [x] `μ⁻ = A·prior.mean + b` is computed directly (pure mean-propagation, needs no
      Q) *before* any factor is built; the observation factor comes from
      `observation.linearize(μ⁻)` and the transition's Q from
      `process_noise.noise_at(μ⁻)` that step (per-step factors on this path only —
      the fixed path keeps its construction-time front-loaded factors). `__init__`
      front-loads each side only when fixed — unconditional front-loading would
      reject a model whose `dynamics_noise` placeholder is merely PSD (legitimate
      when `process_noise` is state-dependent), since `GaussianTransition` requires
      PD.
- [x] Dropped the Phase 2 scope rejection; gated directly against `KalmanBackend`'s
      R(x)/Q(x) path in `tests/test_ffg_chain.py`
      (`TestChainCallableSensorParity`/`TestChainCallableProcessNoiseParity`,
      8 new tests — constant-callable-reduces-to-fixed, scalar/2-D sequences, and a
      control-shifted μ⁻ discriminator, mirroring `test_kalman.py`'s ADR-008
      fixtures). The one durable rejection (`Q=0`, no information form) stays.
      296 total green (was 290; net +6 from −2 obsolete scope tests, +8 parity
      tests), `ruff`/`ty` clean.

### Phase 3 — latent-goal epistemic value, linear stage — PLANNED

Fulfils the Extras line below and a domain-expert critique (attributed to Conor
Heins, re: epistemic value and the discrete T-Maze task): the existing
`bacillus_seeking_food.py` beacon collapses uncertainty about the agent's *own*
position, which is a trivial form of state information gain — it isn't tied to
resolving a genuine contextual unknown, unlike visiting the T-Maze cue to learn
which arm holds the reward. This phase makes the beacon's epistemic value about
an explicit latent — the food's position — instead.

Confirmed by direct reads of `efe.py`/`observation.py`/`selection.py`/
`structure.py`/`chain.py` (cross-checked by an independent design review): needs
**zero core-library changes**. `LinearGaussianModel`, `CallableSensor`, and
`expected_free_energy` are already generic over state/observation block
structure — a sensor channel can read one state block while its noise depends on
a *different* block, and the EFE kernel doesn't care how many channels are
stacked.

- [ ] Augment the bacillus state to `[agent_xy, food_xy]` (4-D); food block
      stationary with small strictly-positive Q (`ChainBackend` rejects exact
      `Q=0`).
- [ ] One `CallableSensor`, two stacked channels: fixed-precision `o_self`
      (agent's own position, unchanged) + `o_disp = food_xy − agent_xy` (a
      relative displacement/bearing vector) whose noise is the **existing,
      unmodified** beacon falloff evaluated at the agent's own position block.
      Minimizing squared displacement via EFE's quadratic pragmatic term is
      mathematically gradient ascent on a potential peaked at the food — this
      already reads as chemotaxis-shaped behaviour without a literal
      concentration-field sensor (that's Phase 5).
- [ ] Static `Preference(goal=[*,*,0,0], precision=block_diag(0·I₂, Λ·I₂))` —
      zero weight on self, weight Λ on "observe zero displacement from food."
      Because the predicted reading is `E[food_xy]⁺ − agent_xy⁺`, this single
      static preference algebraically chases the *current belief* of food's
      location — no per-step preference rebuilding.
- [ ] New demo `examples/bacillus_uncertain_food.py` (additive — the existing
      flagship is unchanged, same convention as keeping `bacillus_lqr.py`
      alongside it), `simulate()` parameterized over backend
      (`KalmanBackend`/`ChainBackend`).
- [ ] `--scan` mode: behaviour metrics plus a Kalman-vs-`ChainBackend` agreement
      check (`atol=1e-7`, the bar `tests/test_ffg_chain.py` already holds) — the
      literal "use both backends" deliverable, and new territory (no existing
      test covers a sensor channel reading one state block with noise keyed on a
      different block).
- [ ] Optional, recommended: a parity test near `TestChainCallableSensorParity`
      in `tests/test_ffg_chain.py` locking down that cross-block sensor topology
      independent of the example script.
- [ ] ADR-013 records the decision, the rejected alternative (a per-step
      `Preference` rebuild on absolute sensing — behaviourally equivalent, more
      legible, but doesn't scale as cleanly to multiple goal items), and the open
      multi-goal fork (shared vs. per-item beacons — a behavioural design choice,
      not a capability gap; deferred, not resolved here).

### Phase 4 — `NonlinearSensor`: second-order Gaussianization — PLANNED

Resolves ADR-006's long-deferred item ("Deferred to Phase 2.5: NonlinearSensor +
2nd-order Gaussianization — the nonlinear-mean case... the corrected full-2nd-order
formula (mean **and** covariance correction together) and its dual-oracle
definition-of-done are pinned in the build plan so they cannot be re-forgotten").
Folded in from a detailed, never-migrated design in the gitignored
`.claude/cpomdp_v0.4_build_plan.md` (its own Phase 3) rather than re-derived.
**Real core-library change — high-stakes: spec + failing tests handed over, user
implements, reviewed rather than authored** (mentor-mode split-by-stakes
convention).

- [ ] `NonlinearSensor` implements `ObservationModel`, owning `gaussianize`
      directly (not `linearize` + a generic fallback) — a nonlinear mean has no
      single local `(C, R)` that's exact, which is exactly why `gaussianize` is a
      sensor-owned seam (ADR-006 D1).
- [ ] **Second-order** moment matching (carries the Hessian/curvature term), not
      first-order EKF. Cite Kouw (arXiv 2409.01974): first-order linearization
      drops the state-dependent ambiguity the epistemic term lives on.
- [ ] Validation: a sensor with a closed-form second moment recovers it exactly;
      a first-order (EKF-style) comparison on the same case is shown to
      under-report ambiguity (the regression witness for the Kouw point).
- [ ] jit/grad/vmap smoke tests, same gate discipline as every other public
      inference entry point (ADR-012).

### Phase 5 — latent-goal epistemic value, nonlinear stage — PLANNED

Built on Phase 4. Swaps the linear displacement channel for a literal scalar
concentration sensor `o = c(‖food − agent‖)` (e.g. a Gaussian bump centred on the
food) — the biologically literal chemotaxis mechanism (E. coli senses a scalar
via temporal sampling, not a vectorial bearing). Run through the same
Kalman-vs-`ChainBackend` comparison as Phase 3.

- [ ] Sensor-type toggle on `bacillus_uncertain_food.py`, or a second demo file
      (decide once Phase 4's shape is known).
- [ ] Same `--scan` agreement-check discipline as Phase 3.

### Extras

- [ ] Update contribution section of the docs that explicitly state code blatantly wrote by AI with zero regard for quality cpomdp tries to upkeep will result in PR being closed.

### Out of scope (ADR-012 — say no on sight)

General `@model` frontend; tier-2 conjugate-exponential engine (seam stubbed,
deferred to v0.5+); reactive scheduling / automatic conjugacy; constrained
Bethe Free Energy as a general objective; structure *learning*.
