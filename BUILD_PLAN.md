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

### Phase 2 — factor nodes + chain = Kalman numerical-identity gate — DONE (RxInfer oracle closed in Phase 4/5)

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
- [x] RxInfer oracle check on small graphs (behind the `rxinfer` marker) — deferred to
      and closed in Phase 4/5 on the *branching* tree
      (`test_branching_tree_matches_coupling_graph`), the more demanding non-chain case
      that subsumes the chain check.
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

### Phase 3 — latent-goal epistemic value, linear stage — DONE (2026-06-28)

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

- [x] Augment the bacillus state to `[agent_xy, food_xy]` (4-D); food block
      stationary with small strictly-positive Q (`ChainBackend` rejects exact
      `Q=0`).
- [x] One `CallableSensor`, two stacked channels: fixed-precision `o_self`
      (agent's own position, unchanged) + `o_disp = food_xy − agent_xy` (a
      relative displacement/bearing vector) whose noise is the **existing,
      unmodified** beacon falloff evaluated at the agent's own position block.
      Minimizing squared displacement via EFE's quadratic pragmatic term is
      mathematically gradient ascent on a potential peaked at the food — this
      already reads as chemotaxis-shaped behaviour without a literal
      concentration-field sensor (that's Phase 5).
- [x] Static `Preference(goal=[*,*,0,0], precision=block_diag(0·I₂, Λ·I₂))` —
      zero weight on self, weight Λ on "observe zero displacement from food."
      Because the predicted reading is `E[food_xy]⁺ − agent_xy⁺`, this single
      static preference algebraically chases the *current belief* of food's
      location — no per-step preference rebuilding. Tuned by sweep:
      `Λ_disp=0.015` gives a clean detour-then-exploit trajectory (below ~0.03
      the agent never bothers detouring, just slowly averages the murk).
- [x] New demo `examples/bacillus_uncertain_food.py` (additive — the existing
      flagship is unchanged, same convention as keeping `bacillus_lqr.py`
      alongside it), `simulate()` parameterized over backend
      (`KalmanBackend`/`ChainBackend`).
- [x] `--scan` mode: behaviour metrics plus a Kalman-vs-`ChainBackend` agreement
      check (`atol=1e-7`, the bar `tests/test_ffg_chain.py` already holds) — the
      literal "use both backends" deliverable, and new territory (no existing
      test covers a sensor channel reading one state block with noise keyed on a
      different block). Result: `max|Δmean|≈1e-10`, `max|Δcov|≈1e-10` — both
      backends pick the identical detour-then-exploit trajectory.
- [x] Parity test `TestChainCrossBlockSensorParity` added in
      `tests/test_ffg_chain.py`, locking down that cross-block sensor topology
      independent of the example script (298 total green, was 296; `ruff`/`ty`
      clean).
- [x] ADR-013 records the decision, the rejected alternative (a per-step
      `Preference` rebuild on absolute sensing — behaviourally equivalent, more
      legible, but doesn't scale as cleanly to multiple goal items), and the open
      multi-goal fork (shared vs. per-item beacons — a behavioural design choice,
      not a capability gap; deferred, not resolved here).
- [x] Promoted to the new **flagship** (the v0.3 demo moves into "the journey"
      instead): the demo grew a fourth, genuine **classic LQR** regime alongside
      sharp/balanced/weak Λ, matching the v0.3 flagship's 2x2-grid structure
      exactly. `LQRController` needed no change either — it only ever consumes
      the dynamics/control matrices and cost weights, never the sensor, so it
      runs unmodified on this state-dependent-sensor model; its per-step `goal`
      is `belief.mean[2:4]` (the current food estimate, zero-weighted on its own
      block), not a static target. Sweep-tuned regimes:
      `Λ=0.10` (sharp, never detours), `Λ=0.015` (balanced, detours then
      exploits), `Λ=0.006` (weak, parks at the beacon for 77/90 steps and never
      reaches food).
- [x] Rendering: a 2x2 grid GIF reusing the flagship's palette,
      `_draw_bacillus`, and `_precision_field` helper; both belief
      markers/ellipses (agent + food) drawn per panel. Displayed ellipse
      diameter is capped (`_ellipse(..., max_diameter=...)`, never the
      underlying belief) — the food prior is deliberately wide
      (`FOOD_PRIOR_COV`), and its raw step-0 diameter exceeded the whole plot
      before this cap. Each panel's border AND its own `t=` step counter turn
      green/freeze once that regime first settles near the food and stays
      (`_arrival_step`, suffix-AND over the "within `ARRIVAL_THRESHOLD`"
      boolean array, so a transient close pass doesn't count) — confirmed
      against the actual per-regime arrival steps (sharp 18, LQR 35, balanced
      41, weak never), and what makes those numbers legible straight off the
      GIF rather than only in `--scan`'s printed metrics.
- [x] `examples/README.md`: the flagship section now points at
      `bacillus_uncertain_food.py`; the v0.3 demo moved into "the journey" with
      a note on what it lacks (ADR-013).
- [x] Reframed in literature-accurate terms: epistemic value here is
      **instrumental** (Friston et al. 2015, "Active inference and epistemic
      value" / the T-Maze task) — the resolved uncertainty is decision-relevant,
      changing where the agent subsequently heads — not merely salience for its
      own sake, which is what the v0.3 beacon exhibited.
- [x] Checked, not assumed: "is balanced quicker?" was computed directly
      rather than asserted. It is not — balanced is the *slowest* regime that
      arrives and travels the *farthest* (it deliberately detours), but is far
      more precise once settled (≈7x tighter final food-covariance trace,
      ≈4-5x smaller final position error than sharp/LQR). `examples/README.md`
      states the verified explore/exploit tradeoff, not the unverified "quicker"
      claim.

### Phase 4 — branching factor graph: the `CouplingGraph` — DONE (2026-06-29; levels() deferred)

**The core v0.4 representation** (ADR-012 / ADR-014): a *factorisable* (branching) model
the chain/Kalman path cannot draw cleanly — nodes coupled into a tree with a shared node
of degree > 2 (the chemotaxis network is the worked example, but the toolbox is
configuration-agnostic, ADR-015). Built on the existing `CanonicalGaussian` messages and
Tier-1 factors; domain-agnostic, integer-indexed nodes.

- [x] New message primitive — `GaussianCoupling` in `factors/linear_gaussian.py`: the
      structural edge `N(child; W·parent, Q)` with `message_to_parent`, the upward
      (child→parent) message that mirrors `GaussianTransition.predict`. Non-square `W`
      allowed. Tested vs a moment-form joint oracle incl. non-square cases.
- [x] Representation — `src/cpomdp/ffg/graph.py`: `Coupling` (a directed
      `child = W·parent + noise` edge carrying a `GaussianCoupling` and a `tau`) and
      `CouplingGraph` (integer-indexed nodes, per-node dims, leaf observations).
      Construction validates a well-formed rooted tree with dimension-consistent factors.
- [x] Inference — `CouplingGraph.infer`: collect each branch's upward message to the
      root and combine with the prior. Hand-authored tree-collect schedule (deepest-first,
      so children fold into a node before it is sent up; ADR-012 choice 4, not a reactive
      scheduler). Only the root crosses moment form (lift in / read out); messages stay
      canonical between — the per-root, not per-node, inversion cost.
- [x] Tests (`tests/test_ffg_tree.py`): branching marginal vs an independent moment-form
      full-joint oracle over arbitrary trees — depth-1 (incl. non-square), depth-2 through
      an unobserved internal node, mixed trees, internal / under-determined / root
      observations, empty readings, single node — plus jit/grad/vmap through `infer` and
      the construction-validation cases.
- [~] `CouplingGraph.levels()` (the τ-cutoff fast/slow view) — **deferred past v0.4**
      (ADR-015): the per-edge vs path-gated semantics is undecided, the two agree on every
      depth-1 model v0.4 ships, and the choice needs temporal/reactive-inference research
      (likely a future RFC). `tau` is stored on edges; the projection is not built.

### Phase 5 — the difference demo + RxInfer oracle — DONE (2026-06-30)

**Closes the DOD.** A demo that *shows the difference* between a normal backend and
the factor-graph one — the branching chemotaxis model represented natively as an FFG
vs. what a `KalmanBackend` must flatten by hand. Unlike Phase 3's `--scan` (which
shows the two backends *agreeing* on a chain, i.e. identity by construction), this
exhibits a topology the chain path cannot express cleanly.

- [x] RxInfer oracle check on the small branching graph (also closes Phase 2's open box,
      now on a non-chain topology), behind the `rxinfer` marker —
      `test_branching_tree_matches_coupling_graph` collects the same root marginal up a
      degree-3 tree through RxInfer's machinery and `CouplingGraph.infer`.
- [x] Demo/figure contrasting the native FFG representation with the flattened-joint
      Kalman equivalent — `examples/coupling_graph_figure.py` (asset
      `docs/assets/coupling_graph.png`): `CouplingGraph` (name three edges, one `infer`)
      beside the 4x4 joint precision you otherwise assemble, invert, and marginalise.
- [x] Numerical agreement of the FFG posterior with the flattened-joint oracle — the
      figure asserts the gap (`≈1.7e-16`, well under `1e-7`) live before rendering, and
      `tests/test_ffg_tree.py` gates `CouplingGraph` against an independent moment-form
      full-joint oracle over arbitrary trees. The *difference* shown is representational,
      which is the point.

## v0.4 → FFG active-inference loop (issues #25–#27; ADR-016/017/018)

Phases 0–5 above make a branching model *perceivable* as a static within-slice collect.
This workstream adds the **time axis and control** so the branching FFG is a full
`InferenceBackend` the `Agent`/EFE loop drives — the road to the v1 chemotaxis result
(emergent drift + information efficiency vs Mattingly et al. 2021). Design crux resolved
as a **carry-partition backend** (ADR-016) under **driven-relaxation** composition on a
**single clock** (ADR-017); admissibility of a partition under EFE guarded per ADR-018.

### Phase 1 — temporal recursion, full-joint carry, single clock — IN PROGRESS

`ChainBackend` generalised to an exact recursive filter over the tree. Driven relaxation
(each node its own dynamics + its structural parent drive every slice, ADR-017) makes the
one-step filtering posterior a *dense* joint, so the exact `[[all]]` endpoint is the
joint-precision solve — trivially the hand-flattened Kalman with the couplings as
within-slice factors. The distribute-pass / factored machinery is Phase 2, not here
(decided 2026-07-01: it is not the exact recursive filter).

- [~] `CouplingGraphBackend` (`src/cpomdp/ffg/backend.py`): `infer_states` = lift joint
      prior → predict through block-diagonal `F=blkdiag(A_i)` with control → add the
      front-loaded structural precision `Λ_struct` + per-node observation messages →
      `to_moment`. Carries the joint `Belief`; `marginal`/`readout` slice a chosen node
      (issue #25 — target latent need not be the root). Front-loaded per ADR-002.
- [ ] Keystone (`tests/test_ffg_backend.py`): all-node marginals vs an independent NumPy
      driven-relaxation joint-precision filter over multi-step sequences, with/without
      control, atol 1e-7. Plus `CouplingGraph.to_flat_model` + a `KalmanBackend`
      cross-check, and the static case vs the RxInfer tree oracle (extended to non-root
      nodes) behind the `rxinfer` marker.
- [ ] jit/grad/vmap smoke + `isinstance(backend, InferenceBackend)` (ADR-012 gate).

### Phase 2 — partition parameter + carry factorisation + severed-mass diagnostic

- [ ] `partition` argument (default full joint); at the carry, zero between-cluster Λ
      blocks; surface `partition_error` (dropped-block norm) per step / per run.
- [ ] The structure-exploiting two-pass tree BP as the cheap exact solver for the factored
      regime: `GaussianCoupling.message_to_child`, `CanonicalGaussian.__sub__` (belief
      division), `CouplingGraph.infer_all` (collect + distribute → all-node marginals).
- [ ] Gate: full-joint partition reproduces Phase-1 numbers exactly; a singleton partition
      runs and reports non-zero severed mass (`tests/test_ffg_partition.py`).

### Phase 3 — chemotaxis generative model as an FFG

- [ ] Receptor → CheA (degree-3) → CheY-P (fast) and CheR/CheB methylation (slow), native
      `CouplingGraph` + per-node transitions carrying the Mattingly timescales
      (τ₁≈0.05s kinase, τ₂≈9.9s methylation, λ_tot≈0.86 s⁻¹); discretise on the one `dt`.
      Gate: static inference vs RxInfer + flattened joint at atol 1e-7.

### Phase 4 — close the loop (EFE on the FFG marginals, issue #26)

- [ ] `info_target=<node>` on `ObservationGoal`; EFE pragmatic/epistemic from the FFG node
      marginals, reducing to `expected_free_energy` on a chain (atol 1e-7); ADR-018
      admissibility guard. Sign/decomposition matches ADR-005; rfcs/004 tests pass.

### Phase 5 — v1 validation demo + figure (issue #27)

- [ ] `{fast+CheA}/{slow}` partition reproduces η (=0.66±0.05, **dimensionless — never a
      rate**) and drift within the full-joint run and Mattingly bounds; EFE-admissibility
      check passes on the cut. New `examples/chemotaxis_*.py`, the closed-loop successor to
      `coupling_graph_figure.py`.

### Deferred beyond v0.4 (ADR-014 — tracked as GitHub issues)

Briefly in this plan or explored this session; all outside the v0.4 DOD, now filed as
GitHub issues #20 (multi-step EFE), #21 (`NonlinearSensor`), and #22 (nonlinear
chemotaxis demo) — the "epistemics beats LQR" demo rides on #20's multi-step EFE
(rationale + preserved findings in ADR-014):

- [ ] `NonlinearSensor` + second-order Gaussianization (was Phase 4 here — a sensor
      feature, orthogonal to FFG factorisation). The Kouw (arXiv 2409.01974)
      dual-oracle design lives in the gitignored old plan / `reference_old_v04_build_plan`
      memory so it is not lost.
- [ ] Nonlinear scalar-concentration chemotaxis demo (was Phase 5 here).
- [ ] Multi-step EFE / planning-as-inference — the principled route to epistemics
      genuinely beating LQR (the current one-step EFE under-credits information; see
      ADR-014's separation-principle / dual-control finding).
- [ ] The honest "epistemics beats LQR" demo (depends on multi-step EFE; the cue
      shortcut explored this session was a fudge and was reverted — ADR-014).

### Extras

- [ ] Update contribution section of the docs that explicitly state code blatantly wrote by AI with zero regard for quality cpomdp tries to upkeep will result in PR being closed.

### Out of scope (ADR-012 — say no on sight)

General `@model` frontend; tier-2 conjugate-exponential engine (seam stubbed,
deferred to v0.5+); reactive scheduling / automatic conjugacy; constrained
Bethe Free Energy as a general objective; structure *learning*.
