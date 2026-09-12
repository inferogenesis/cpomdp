# p\* programme — falsification criteria

Reference checklist for the build. The argued version of this document lives outside the
repo. Nothing here restates the case; use this to check scope, not to understand it.

Standing rule: no number reaches prose unless the check suite prints and asserts it.

**Currency.** `FALSIFY` a sign, ordering or count the process theory forbids ·
`SEVERE-TEST` a pre-registered magnitude or exponent, in Mayo's sense · `DEMARCATE`
where testability ends.

**Tier.** `EXACT` closed-form at machine precision · `BOUNDED` a stated bar or certified
bracket · `COMPUTED` no statable bar. The word for a `COMPUTED` number is *computed*,
never *certified*. `research/warrant_ledger.md` carries the canonical table.

**Readiness.** `v0.4.x ✓` shipped · `PR-n · tag` scheduled, per `BUILD_PLAN.md` ·
`further` needs capability not yet scheduled.

---

## The decomposition under test

```
E_{p*(·|u)}[F] = H(p*(·|u)) + D_KL[p*(y|u) ‖ p(y|u)] + E_{p*(·|u)}[ D_KL[q(x) ‖ p(x|y)] ]
                 └── floor ──┘   └── misspecification ──┘   └────── inference gap ──────┘
```

Floor: a property of p\*, untouchable by policy. Misspecification: zero iff p = p\*.
Inference gap: zero iff q = p(x|y). The standard accounting identifies p with p\* and
collapses all three to the floor.

---

## A — The collapse: fixed-R linear-Gaussian under additive control has *exactly zero* action-dependent epistemic value

**A1 · flat epistemic term** · FALSIFY · R-none, P1 eq. (4) · tier `EXACT` · **v0.4.2 ✓**

- Predict: per-step epistemic value constant across policies to machine precision.
  `I[x;o|π] = ½ ln det(CΣ⁻Cᵀ + R) − ½ ln det R`, μ absent.
- Build: `efe_collapse_figure.py`.
- Falsify: any action-dependent variation above the printed numerical floor, fixed-R.
- Does not buy: only that *this* model class collapses. It is the null the rest is
  measured against, not evidence for the FEP.
- Note: R1 is the *decomposition's* calibration zero, which is C1. A1 is P1's
  mutual-information result, a distinct measurement sharing the correct/exact cell.

**A2 · flattening reproduction** · FALSIFY · fixed-R triviality, Rmk 7 · tier `EXACT` · **v0.4.2 ✓**

- Predict: the fixed-gain Kalman schedule reproduces posterior mean and covariance for
  every policy and observation sequence. `to_flat_model` succeeds.
- Falsify: no reproducing schedule in the fixed-R regime. Indicts the implementation,
  not the theory.
- Does not buy: certainty equivalence here. Nothing about R(x).
- **Do not** route through Cor 2. That is a *planning* reduction (Def 2), and Rmk 5
  exhibits one coexisting with no Def-1 schedule (observation-independence fails). The
  biconditional is false in general and holds here only by the fixed-R triviality.

**A3 · structural, not accidental** · FALSIFY · P1 eq. (4) sweep · tier `EXACT` · **v0.4.2 ✓ / trivial v0.5 hardening**

- Predict: survives arbitrary perturbation of A, B, C, Q and of the *value* of fixed R.
  Breaks only when R gains a state argument or control turns multiplicative.
- Falsify: a fixed-R additive-control model with action-dependent epistemic value.
- Does not buy: a structural silence is strong content. Register it as such, against
  Family D's ties, which vanish under perturbation.

---

## B — The reintroduction: R(x) returns epistemic value, as determinant-visibility

**B1 · dual effect (H3)** · FALSIFY · Thm 1(i) · tier `BOUNDED`, ungated · **v0.4.2 ✓**

- Mechanism: R(μ⁻) couples control to the posterior covariance; the Kalman gain
  inherits the dependence.
- Predict: Σ⁺ₖ(π) ≠ Σ⁺ₖ(π′) for some policies and some k, under H3 (R non-constant on
  the reachable set of predicted means).
- Build: `KalmanBackend` over `CallableSensor`.
- Falsify: policy-independent posterior covariance despite verified H3.
- Does not buy: covariance motion is necessary, not sufficient. See B3.
- **Qualifier that travels (Rmk 2)**: the dual effect exhibited is a property of the
  *agent's maintained Gaussian recursion*. Whether the exact non-Gaussian conditional
  covariance carries a dual effect in the strict Bar-Shalom–Tse sense is open in P1.
  Bounds B1 and E1's dual-control bridge.

**B2 · determinant-visibility (H3′)** · FALSIFY · Thm 1(iii), Cor 1 · tier `BOUNDED` · **v0.4.2 ✓ (scalar)**

- Predict: εₖ(π) ≠ εₖ(π′) under H3′. Automatic for scalar observations under H1–H3
  (Cor 1), epistemic value being strictly monotone in the noise variance.
- Build: checkable sufficient condition Rmk 6, Löwner-comparable pinning covariances.
- Falsify: flat epistemic value under verified H3′, or under verified H1–H3 in the
  scalar case.
- Does not buy: the title's claim rests on H3′, not H3. Keep the distinction
  load-bearing.

**B3 · the Rmk-4 knife-edge, dual effect *without* epistemic value** · FALSIFY · Ex 1 · tier `BOUNDED` · **v0.4.2–v0.5, analytic, no new capability**

- Predict: `R(x) = diag(r(x₁), s(x₁))` tracing `(1 + 1/r)(1 + 1/s) = const`. Every
  per-step epistemic value equals `½ ln 3` for every policy while Σ⁺ₖ varies. Parts
  (i)–(ii) fire, (iii) does not.
- Falsify: epistemic value moving on the exact analytic level set. Breaks the H3/H3′
  distinction and the determinant-visibility calibration of the whole result.
- Scope: unproved (deferred item 10) is vector-case *positivity* for Thm 1(iii) in
  general, not this example, which exhibits the level set in closed form.
- Does not buy: corroborates the *visibility* reading specifically. Most severe
  self-imposed test in the battery.

**B4 · behavioural dissociation** · FALSIFY · P1 section 5 · tier `BOUNDED` · **v0.4.2 ✓**

- Predict: Agent A (fixed R, frozen at the prior position) and Agent B (R(x)) set off
  together toward the shared-prior arm. Only B reads the cue, learns the context and
  crosses. Opposite arms, sole difference R vs R(x).
- Build: `epistemic_dissociation_figure.py`, continuous T-maze, two-node coupling tree.
- Falsify: A also crosses, or B fails to, under matched preferences and matched
  pragmatic pull.
- Does not buy: the outbound leg is confounded, both heading to the cue for the
  pragmatic reason. The objective-level extension strips it: B's epistemic value peaks at
  a cue off the pragmatic path, worth **1.72 nats** against a **4.49-nat** pragmatic
  gradient.

**B5 · no-flattening witness** · FALSIFY · Thm 1(ii) · tier `BOUNDED` · **v0.4.2 ✓**

- Mechanism: structural and data-free. A mean-shifting coupling under state-dependent R
  makes the flat linearisation point μ⁺ diverge from the prior mean μ⁻.
- Predict: `to_flat_model` raises `IncompatibleLinearizationError` on the
  R(x)-plus-coupling configuration before any data is processed. The identical topology
  with fixed R flattens without complaint.
- Falsify: the published one-liner, any fixed linear-Gaussian filter whose Kalman
  recursion reproduces Agent B's posteriors on every observation sequence. Drops the
  guard, the check suite and Thm 1 together.
- **Do not** oversell the tripwire as the theorem. The claim is a **negative existential**
  which no execution can establish (ledger section 3). Thm 1(ii) proves it, the code
  witnesses that one named route fails. Refutable by one counterexample, never
  confirmable. Infinite-dimensional representations are left open by theorem and code
  alike. P1 section 5 splits this correctly; the split must survive into Paper 2.

---

## C — The decomposition as an instrument (p\* ≠ p)

Off-diagonal cells carry the claim. The calibration zero validates nothing on its own.

**C1 · calibration zero** · calibration · R1 · toolbox B, F · tier `EXACT` · **PR-5 · v0.4.5**

- Predict: p = p\*, exact filter → misspecification and inference gap both below `1e-12`.
- Falsify: a nonzero floor top-left. Textbook zero (Kalman is the exact Bayesian filter
  for fixed-R LG), so a nonzero reading indicts the instrument. Concede its textbook
  status; the contribution is the cross.
- Does not buy: necessary calibration, not a result.

**C2 · misspecification isolation** · FALSIFY · R2 · toolbox B constructors · tier `EXACT`/`BOUNDED` · **PR-5 · v0.4.5**

- Predict: parameter-perturb the model, keep inference exact → `D_KL[p* ‖ p]` positive
  and stable across a **declared** seed set, inference gap below `1e-12`.
- Falsify: the inference gap moving under pure misspecification. Cross-contamination
  falsifies separability, the core metrological claim.
- Does not buy: this cell, not C1, is what makes the instrument an instrument.

**C3 · inference-gap isolation** · FALSIFY · R3 · toolbox B degraded variants · tier `EXACT`/`BOUNDED` · **PR-5 · v0.4.5**

- Predict: p = p\*, degrade inference (frozen gain / wrong fixed R / diagonal-only
  covariance) → inference gap positive, misspecification below `1e-12`.
- Falsify: misspecification moving under pure inference degradation.
- Does not buy: with C2, the two-term separability the one-term accounting cannot
  represent.

**C4 · additivity** · FALSIFY · R4 · toolbox F · tier `EXACT` · **PR-5 · v0.4.5**

- Predict: `H(p*) + misspecification + inference gap` reconstruct measured `E[F]` within
  the four-term bound.
- Falsify: residual above tolerance. The decomposition is incomplete, a fourth term
  exists, or a term is mis-estimated. Each is a real result about the accounting.
- Does not buy: closure of *this* decomposition on *this* model class.

**C5 · incompleteness demonstration** · DEMARCATE · R2 + R3 at a matched total · tier `EXACT`/`BOUNDED` · **PR-5 · v0.4.5**

- Predict: two agents the collapsed accounting scores *identically* at the same total
  surprise, one misspecified-but-exact and one correct-but-approximate, matched to equal
  `E[F]`, which the instrument separates.
- Build: **solve for the matched pair analytically**, not by root-search over
  perturbation magnitude. A searched match carries the searcher's tolerance and is
  a sample; a constructed match is `EXACT`.
- Falsify: no such matched pair exists. That collapses the instrument to one dimension
  and *vindicates* the standard accounting.
- Does not buy: demarcates incompleteness. It does not falsify the FEP. Frame as
  incompleteness of the received accounting, **never** as refuting the FEP.

**C6 · breaking the calibration zero under R(x)** · FALSIFY · R6 · toolbox C · tier `BOUNDED` · **PR-9 · v0.5**, gated on GATE-D4

- Mechanism: under R(x) the exact posterior is non-Gaussian and every tractable rule is
  a Gaussian surrogate, so the gap is strictly positive by theorem. The correct/exact
  cell is unreachable.
- Predict: correct model, plug-in inference, R(x) → inference gap positive and separated
  from zero by more than the certified numerical error.
- Build: the exact reference filter is needed to *certify the separation*, not merely to
  observe a positive number. GATE-D4's bound is what licenses "separated from zero".
- Falsify: gap at machine precision under R(x) plug-in inference. Contradicts Part 2's
  positivity theorem (v2.1's B6) and indicts either the theorem or the reference filter.
- Does not buy: a positive gap is *predicted*, so measuring it confirms nothing and
  refutes nothing. The falsifiable content is that the gap cannot be driven to zero.

**Reporting requirement, C1–C3** (ledger section 4). "Below 1e-12" licenses only *smaller
than 1e-12 in these units*, never *zero*. Every cell prints the pinned term, the moving
term's magnitude beside it, their **separation ratio**, and `cond(Σ)` / `cond(S)` for the
matrices inverted. A cell reporting only the small number is not evidence: a term
naturally `O(1e-13)` clears the bar trivially.

---

## D — Quantitative severe tests

Protocol: register the diagnosis, not the conclusion. Never discover a threshold by
raising a parameter until the result fires. Require the sign flip at the registered point
*and* one step below.

**D1 · fidelity-ladder ordering** · SEVERE · R7 · toolbox C, D · tier `BOUNDED` · **PR-9 · v0.5**, gated on GATE-D4

- Predict: along plug-in `R(μ⁻)` → Spinello–Stilwell iterated → belief-smoothed
  `E[R(x)]` → exact reference, the inference gap decreases at a certified tolerance, ε
  non-constant throughout.
- Falsify: non-monotone ordering. Registered as a *conjecture with a predicted
  direction*, not a theorem: higher approximation order does not imply monotone KL to the
  exact posterior, and Rmk 3 calls the smoothed rule a refinement only in the
  approximation-order sense. A non-monotone result belongs in the paper, not a drawer.
- **Warrant constraint** (ledger section 5): reportable only when the difference between
  adjacent rungs exceeds the sum of their certified bars. Pre-register the minimum
  separation that counts. Overlapping bars are pre-committed to a **third outcome,
  `NOT_RESOLVED`**, neither confirmation nor refutation. Without `BOUNDED` this leg produces
  four computed numbers in an order, which is not a result in either direction.
- Does not buy: weaker content than D2's exponent. The qualitative half.

### AMENDMENT 2026-09-04: D1's ladder is five rungs

ADR-056 declares the ladder as five rungs, inserting the paper's single-step filter
(36) between plug-in `R(μ⁻)` and the iterated scheme, so that the derivative-of-
covariance terms and the iteration are two adjacent differences rather than one. The
prediction above is therefore over plug-in `R(μ⁻)` → single-step (36) → iterated (35) →
belief-smoothed `E[R(x)]` → exact reference, monotone throughout at the certified
tolerance, and `NOT_RESOLVED` between any adjacent pair whose bars overlap. The two
middle rungs run the modified scheme ADR-057 ships, so the registered line's
"Spinello–Stilwell iterated" reads as modified from here. The prediction's direction and
its falsifier are unchanged, and "four computed numbers" in the warrant constraint reads
five. Registered here before any rung exists, since an ordering
seen before the set is fixed is what standing rule 7 refuses.

### PRE-REGISTRATION 2026-09-12: the ordering's first reading, at `COMPUTED`, before the gate

Written before any rung's gap is read beside another's. The five rungs exist in
`cpomdp.reference.ladder` and GATE-D4 has not been evaluated, so this is not the D1
test. It is the ordering the test will run on, read once at `COMPUTED` with measured
bars, so that PR-8 has the R6 signal it compares against `T` and so that the direction
is on record before a certified bar exists. The D1 test proper runs in PR-9 under PR-8's
certified bars, and nothing here pre-empts it.

**The cells.** `d4-family-v1` at the binding cell of the GATE-D4 registration:
`R(x) = 1 + 0.1·x²`, `μ* = √10`, fifteen prior spreads even in `ln σ` on `[0.005, 0.7]`,
on the lattice `T` is registered under and on the fine lattice beside it. These are
`research.explorations.threshold`'s `SPREADS`, `DECLARED` and `FINE`, read from there
rather than restated, so the ordering is measured where `T` was.

**What each rung reports.** `averaged_inference_gap` on the declared lattice, one value
per rung per spread, with `predictive_mass`, `worst_edge_ratio` and `voided_mass` printed
beside it. The exact rung's value is whatever the engine returns for a rule that is the
reference, and it is reported as measured rather than written as zero.

**The bar, before certification.** A value's bar is its own refinement difference,
`|gap_declared − gap_fine|`, floored at `100 × 2⁻⁵²` of the value so that no bar is
claimed below what the arithmetic resolves. It is carried in `cpomdp.resolution` as
`Bar(common_mode=0, own=…)`: the whole difference as the quantity's own, nothing
cancelled between rungs. The five gaps are read against one exact posterior and a
common-mode part exists. Claiming it needs the error's shape, which is what PR-8
certifies. Until then the conservative reading is the registered one, and the report
prints `threshold` and `sum_of_bars` side by side so a reader can see they coincide.

**The minimum separation that counts.** For an adjacent pair at one spread,
`resolution_threshold(bar_lower, bar_higher)`, and nothing else. A pair whose
`|difference|` does not exceed it reports `NOT_RESOLVED` at that spread.

**The prediction, and the falsifier.** Along `plug-in → modified-single-step →
modified-iterated → belief-smoothed → exact` the gap decreases, so each adjacent pair
reads `ABOVE` wherever it resolves, the lower rung's gap above the higher rung's. One
row per pair, whose outcome over the fifteen spreads is `FIRED` if the pair resolves
`BELOW` at any spread, `NOT_RESOLVED` if it resolves at none, and `NOT_TRIGGERED`
otherwise. The row lists the order at every spread. Every ordering row is `COMPUTED`,
since its bar is measured and not certified.

**`VOID`.** The iterated rung may decline a reading. Predicted: none on this family at
this curvature, since the budget survey found exhaustion only at `κ = 100` and on the
bounded sine family. Where `voided_mass` is above the roundoff floor at a spread, the
two pairs the iterated rung sits in read `NOT_RESOLVED` at that spread rather than on
the conditional value, and the mass is printed.

**The R6 signal.** The plug-in rung's row prints its gap at each spread beside
`T = 5.962e−4` nats, and the spreads at which the gap exceeds `T`. It asserts agreement
with `research.explorations.threshold.measure_gap` at every spread to `1e−12` relative,
since both run one engine on one lattice and a disagreement would be a defect in one of
them. It renders no verdict on the gate. That is PR-8's.

**The completeness certificate.** A `ProductCompletenessCertificate` over one axis,
`rung`, of size five at `LADDER.version`, `visited` being the rungs whose gap was read
at every spread. `PROVED` when the two agree, with a `Provenance` whose `registered_at`
is the commit this entry lands in and whose `measured_at` is the commit the suite lands
in, both filled in the RESULT.

**Where it runs.** `research.checks.ladder`, declared in
`research/registered_checks.toml`:

```text
uv run --no-sync python -m research.checks.ladder --check
```

**What would change this.** A different family, lattice or spread set re-registers this
entry. Route 6's reading of the same numbers is registered under Q5 of
`research/spinello_stilwell_rung.md`, dated with this one.

**D2 · scaling exponent** · SEVERE · R8 · toolbox C, F · tier `BOUNDED` · **PR-9 · v0.5**, gated on GATE-D4

- Predict: `gap ∝ (curvature of R) × (belief spread)²`. Sweep both factors
  independently, fit the exponent, require it inside a pre-registered interval **around
  2** in the regime where the second-order term dominates.
- **Prerequisite** (ledger section 5): demonstrate the fit window is non-empty *before*
  fitting. Squeezed from both ends: higher-order terms contaminate at large spread, and
  relative error against the certified bound diverges at small spread. Print the window's
  bounds on both swept axes and the signal-to-bound ratio across it. An empty window is
  **VOID**, not FAIL: report the leg unmeasurable and drop it rather than fitting anyway.
  The lower edge sits near `√(k·δ_ref/curvature)`, so a tighter bound widens it.
- Falsify: fitted exponent outside the registered interval, within a demonstrated
  non-empty window. Two failure modes pre-registered as *boring* and excluded from the
  agreement criterion in advance: a clean quadratic with an unexplained prefactor, and an
  exponent contaminated by higher-order terms at large spread.
- Does not buy: bears on the surrogate-vs-exact geometry only, not on the FEP's core.

**D3 · crossover horizon H\*** · SEVERE · R10 · toolbox A, E · tier `BOUNDED` · **v0.4.4 ✓ measured, PR-2 + PR-5 · v0.4.5 complete**

- Predict: sweep H; there exists H\* at which the accumulated epistemic pull overtakes
  the pragmatic gradient, so reach becomes walk. Sign flip asserted at H\* **and**
  H\* − 1. Anchored at H = 1 (pull 1.72 nats, gradient 4.49 nats), forcing H\* > 1.
- **Registered value: H\* = 7.** Coupled-tree cue task, exhaustive over the declared
  `{0, ±1, ±2}`, one dimension, completeness certificate at each horizon. H = 1 anchors
  pinned at `Δε = 1.7232`, `Δc = 4.4910`, `ΔG = +2.7678` nats to `1e-4` (ADR-033).
  Registered before R10 drafting begins.
- **Qualifiers that travel with the number**: H\* = 7 is an **upper bound**, because the
  action grid clips the reach. The action mode (receding-horizon or open-loop) is
  declared on the result, never silently read as closed-loop. The mechanism split is
  disclosed as post-selection, because the scored pair was found by the search.
- Build: toolbox A shipped at v0.4.4, `cpomdp.enumeration` supplying exhaustive `|A|^H`
  search under a cardinality certificate, which is what makes the flip decided rather
  than sampled. Toolbox E (control bracket) outstanding, PR-5.
- Falsify: no crossover at any feasible H. Or a flip that is not clean at H\* and
  H\* − 1. Or not reproducible across seeds. **Or H\* not stable under action-set
  change**, which is two falsifiers rather than one: **refinement** at finer spacing over
  the same range, and **extension** at a wider range. The suite emits five rows for that
  reason.
- **Fourth falsifier, registered before it is run.** Two axes, separate and not
  interchangeable: **extension** is a wider magnitude range at the same spacing,
  **refinement** is finer spacing over the same range. Write the predicted direction and
  its argument per axis *before* running; where no direction can be argued, register a
  stability test at `|ΔH*| ≤ 1` rather than dressing a stability check as a directional
  prediction. Pre-declared budget: `5^7 = 78,125`, `7^7 = 823,543`, `9^7 = 4,782,969`,
  `9^8 ≈ 4.3 × 10^7` if H\* rises under refinement. Budget overrun is **VOID**, never
  "stable". `cue_maze.best_reachable_noise` is the void guard: a refined set that cannot
  land on the cue gives a null indistinguishable from "information is never worth the
  detour", which is geometry, not a result.
- Does not buy: a prediction about the planning horizon, not about the R(x) mechanism. A
  null weakens the "reach becomes walk" reading without touching Thm 1.

### PRE-REGISTRATION 2026-08-20: the fourth D3 falsifier, both axes

Written before any cell below is run in this repository. The registered set is `V1 = [−2,−1,0,1,2]` at
`H* = 7`. `V1_EDGE = [−3,−2,−1,0,1,2]` is already measured at `H* = 6`.

Geometry the arguments rest on, from `cue_maze`: the agent starts at `0`, the cue sits at
`+1`, the prior goal at `−3`. The cue-ward branch spends one step reaching the cue and
then covers a displacement of `−4`. The prior-ward branch covers `−3` directly.

**Extension. Directional prediction, `H* ≤ 6`.** The named cell is `{−4,…,2}`, seven
actions, same spacing. The argument is an asymmetry. At `−3` the prior-ward branch already
covers its `−3` in one step, so a magnitude of `4` buys it nothing. The cue-ward branch
still needs two steps for its `−4` return, and `−4` cuts that to one. The extension
therefore helps the cue-ward branch alone, and a branch that gets cheaper cannot win later.
Predicted `H* ≤ 6`, and plausibly `5`.

This is an argument, not a proof: the argmin over a superset may move either way, since a
larger set changes both branches' scores and not only their step counts. A measured rise
is a real refutation rather than a modelling surprise, and it would say the step-count
reading of this task is wrong.

**Refinement. No direction is arguable, so a stability test at `|ΔH*| ≤ 1`.** The cells
are step `0.5` (nine actions) and step `0.25` (seventeen), both over the same `[−2, 2]`.
Neither changes the largest magnitude, so neither branch's minimum step count moves: the
cue-ward return still needs two steps and the prior-ward reach two. Refinement buys finer
positioning only, and the cue at `+1` is already exactly reachable on `V1`, so it buys no
reachability either. With nothing to argue in either direction, this is registered as a
stability test and not dressed as a prediction. `|ΔH*| ≤ 1` is `PASS`, `|ΔH*| ≥ 2` is
`FAIL`.

**Correction, before this entry landed: the step-`0.5` cell is not virgin ground.**
`research/r10_open_loop_crossover.md` already reports it, with numbers: on a step-`0.5`
grid the argmin is byte-identical to the coarse set at `H = 6` (`Gmin = 364.6430`,
prior-ward) and `H = 7` (`Gmin = 425.1631`, the same walk), no intermediate action scoring
lower `G`, concluding that `H*` is stable under refinement and reading falsifier 4 as *not
triggered*. An earlier draft of this registration called that cell unmeasured. It is not,
and a pre-registration that mis-states what is already known is the failure it exists to
prevent.

What remains is narrower and worth stating exactly. **No commit in this repository builds
a nine-action step-`0.5` set**, so the published numbers have no in-repo reproduction, and
the repository disagrees with itself about the outcome: the write-up reads falsifier 4 as
not triggered while `examples/ffg/crossover.py` reports it `NOT_RUN_HERE` with no warrant.
That is the same shape as the extension row this entry already retracted, still standing.
So the step-`0.5` cell is registered as a **re-measurement under a completeness
certificate**, against the published `364.6430` and `425.1631`, and the stability test
above is what it reports against. The `H = 8` and step-`0.25` cells are new ground and
carry no prior claim.

**Void guard, discharged before the run.** `cue_maze.best_reachable_noise` returns exactly
`R_LO = 0.02` on all four sets, so every lattice lands on the cue and no cell is void by
geometry. A null from any of them is about the objective.

**Budget, and the two units disagree.** Declared in both, because a cell can sit inside one
and outside the other. Time is at the measured 39.0k policies/s.

| cell | actions | H | policies | scored steps | time |
| --- | --- | --- | --- | --- | --- |
| `V1`, registered | 5 | 7 | 78,125 | 546,875 | 2s |
| `V1_EDGE`, measured | 6 | 7 | 279,936 | 1,959,552 | 7s |
| extension `{−4,…,2}` | 7 | 7 | 823,543 | 5,764,801 | 21s |
| refinement step `0.5` | 9 | 7 | 4,782,969 | 33,480,783 | 2.0m |
| refinement step `0.5` | 9 | 8 | 43,046,721 | 344,373,768 | 18.4m |
| refinement step `0.25` | 17 | 7 | 410,338,673 | 2,872,370,711 | 2.92h |

The ledger's `H_max = 9` budget is 17.6M *scored steps*. The step-`0.5` cell at `H = 7` is
33.5M, double that, while its policy count sits far inside the `9^7` line. Both numbers are
declared so neither can be quoted alone.

**`9^8` and `17^7`: accepted, and only on the chunked path.** Time is not what gates them.
`enumeration_cost` describes the front-loaded path, and on this machine, with 19 GiB free,
it reads 14.11 GiB for `9^8` and 131.46 GiB for `17^7`. The measured correction is 1.6x,
giving 22.6 GiB and 210.3 GiB. Both exceed the ceiling, and the WSL cap is configured
rather than physical, so a front-loaded attempt takes the session down rather than raising
`MemoryError`. Both cells run under `ChunkedEfeSearch`, whose peak is block-determined and
flat in `|A|^H` (ADR-036). A front-loaded run of either is **VOID (budget)**, not a result.

**Outcome vocabulary.** Each axis reports `PASS`, `FAIL` or `VOID` against the above. Budget
overrun is `VOID`, meaning unmeasured, and never "stable".

### AMENDMENT 2026-08-21: what a `Gmin` disagreement means, registered before the run

The step-`0.5` re-measurement compares against two published numbers, `Gmin = 364.6430` at
`H = 6` and `425.1631` at `H = 7`. The registered falsifier is `|ΔH*| ≤ 1`, which is a
statement about the horizon and says nothing about those scores. Without a rule written
first, a disagreement would be adjudicated after the fact by whoever preferred which
answer.

**The two are reported separately, and neither is collapsed into the other.**

- **`H*` is the falsifier.** `|ΔH*| ≤ 1` reports `NOT_TRIGGERED`; `|ΔH*| ≥ 2` reports
  `FIRED`. That verdict does not depend on the scores agreeing.
- **A `Gmin` disagreement is a separate result.** It says the published numbers are wrong,
  not that refinement moved the optimum. It lands as its own dated `RESULT`, retracts the
  published values and records the measured ones. It does **not** fire falsifier 4 on its
  own.
- **Tolerance.** The published values carry four decimal places, so agreement means equal
  to within `5e-5`, the half-ulp of the last printed digit. A disagreement larger than that
  is a real difference rather than a rounding artefact of the transcription.

Both outcomes get stated. A run where `H*` is stable and the scores disagree is a passing
falsifier beside a retracted number, and reporting only the first would bury the second.

The two new-ground cells, `H = 8` and step-`0.25`, have no published values to compare
against, so this amendment does not apply to them.

### RESULT 2026-08-20: extension axis, `{−4,…,2}`, **PASS**

Registered above at `H* ≤ 6` before the run. Measured `H* = 6`. Certified `PROVED (set
v1-ext, |A|^H = 7^6 = 117649, visited 117649)`, one certificate per horizon, front-loaded
at 0.42 GiB against 19 GiB free.

| H | policies | `Gmin` | argmin | plan |
| --- | --- | --- | --- | --- |
| 5 | 16,807 | 303.6592 | `[−3, 0, 0, 0, 0]` | reach |
| 6 | 117,649 | 363.9394 | `[+1, −4, 0, 0, 0, 0]` | walk |

**The registered mechanism is the one that fires.** The argument for `H* ≤ 6` was that `−4`
cuts the cue-ward return from two steps to one while buying the prior-ward reach nothing.
The winning policy at `H = 6` is exactly that: one step to the cue at `+1`, then a single
`−4` to the goal at `−3`. The prediction did not merely pass, its mechanism is visible in
the argmin.

**Two things the prediction got wrong, recorded as such.** The parenthetical "plausibly 5"
did not hold: `H = 5` is still prior-ward even though the walk `[+1, −4, 0, 0, 0]` is
feasible there. Feasibility of the shorter return is not what sets the crossover. The
epistemic and pragmatic balance does, and the step-count argument says nothing about it. And
extension **saturates**: `H* = 6` on both `{−3,…,2}` and `{−4,…,2}`, so the one-step return
is used without advancing the horizon.

**What this settles in the write-up.** `research/r10_open_loop_crossover.md` retracted a row
reading "`H* = 6`, unchanged (`−3` already optimal)" as deduced rather than measured. The
retraction was right and the number was right. The stated reason was wrong: `−3` is *not*
what the optimal policy uses at `H = 6`, `−4` is. A row can carry a correct number for a
false reason, and only the measurement tells them apart.

### RESULT 2026-08-21: refinement axis, step-`0.5` — **PASS**, and the published numbers hold

Measured on the chunked path at `3619016`, reproducible with
`python examples/ffg/crossover.py --refinement`.

| H | measured `Gmin` | published | argmin | plan |
| --- | --- | --- | --- | --- |
| 6 | 364.642964185792 | 364.6430 | `[−2,−1,0,0,0,0]` | prior-ward |
| 7 | 425.163110098734 | 425.1631 | `[+1,−2,−2,0,0,0,0]` | cue-ward |

`H* = 7` on the refined set against 7 on the coarse one, so `|ΔH*| = 0`, inside the
registered bar of 1. Falsifier 4 reports `NOT_TRIGGERED` with both certificates as
evidence, `PROVED (set v1-refine-0.5, 9^6 = 531441)` and `9^7 = 4782969`, each visited in
full.

**The published numbers are confirmed to the digit.** Both agree within the `5e-5`
tolerance the 2026-08-21 amendment registered, so the disagreement branch does not fire
and nothing is retracted. What the write-up lacked was a run in this repository, not
accuracy. The retraction was about provenance and it is now discharged.

**No half-step action appears in either argmin.** Byte-identity to the coarse set is
expected, since the coarse set is a subset. The evidential half is that subdividing
offered the optimum nothing it took.

**`9^8` is not required.** The registration made the H = 8 cell contingent on `H*` rising
under refinement. It did not rise, so the cell answers a question that did not arise. It
was started and stopped rather than run to completion, and it is not outstanding work.

**Still outstanding: step-`0.25`.** 410,338,673 policies, projected 2.7 hours at the
measured 41,982 policies/s, peak flat at 0.46 GiB. Deferred on time, not on budget or
memory. The registered stability test stands for it.

### RESULT 2026-08-21: refinement axis, step-`0.25` — **PASS**, and both cells agree exactly

Measured at `c37fac3` on the chunked path, 410,338,673 policies at `H = 7` in 2.5 hours at
46,124 policies/s, peak 0.47 GiB.

| H | policies | `Gmin` | argmin | plan |
| --- | --- | --- | --- | --- |
| 6 | 24,137,569 | 364.642964185792 | `[−2,−1,0,0,0,0]` | prior-ward |
| 7 | 410,338,673 | 425.163110098734 | `[+1,−2,−2,0,0,0,0]` | cue-ward |

`H* = 7`, so `|ΔH*| = 0` against the registered bar of 1. Both horizons certified
`PROVED`, every policy visited.

**The two refinement cells agree to the digit.** `364.642964185792` and
`425.163110098734` are byte-identical to the step-`0.5` values, across a set with twice
the actions and 86 times the policies. No quarter-step action reaches either argmin, just
as no half-step did.

Byte-identity is the expected half: the coarser set is a subset, so if the argmin lies in
it the scores must match. The evidential half is that the argmin does lie in it. Twice
now, at two spacings, subdividing offered the optimum nothing it took.

**Both axes now report.** Extension `PASS` at `H* = 6`, refinement `PASS` at `H* = 7` on
both cells. The fourth D3 falsifier is discharged and PR-2's merge gate is met.

**D4 · certified discretisation bound · GATE-D4** · SEVERE · R9 · toolbox C · tier `BOUNDED` · **PR-8 · v0.4.5, hard gate**

- Predict: the reference filter's error is stated as a number, small relative to R6's
  signal by a **pre-agreed factor**. The factor is written down *before* the bound is
  computed.
- Build: a certified bound, not a fine grid with a convergence plot. Interval arithmetic
  or a proved quadrature error bound, licensing *for all x in the domain,
  |p_grid − p_exact| ≤ δ*. Emits `CERTIFIED` from validated numerics, never `PROVED`
  from an enumeration.
- Falsify: the bound is not statable at the pre-agreed factor. `BOUNDED` collapses to
  `COMPUTED`, Part 2's numbers become uncertified, and the response is a different paper.
- Gate mechanics: v0.4.5 is cut at PR-8's merge whatever the outcome, so a re-scoped
  paper cites a release rather than a commit. C6, D1 and D2 do not merge against an
  uncertified reference.
- Does not buy: certification of the instrument, not a claim about agents. Existential
  for Part 2. State it in the abstract, not the appendix.

---

## E — Is the FEP doing independent work?

**E1 · vs dual control** · DEMARCATE · R5 · toolbox E · tier `EXACT` signature / `BOUNDED` leg · **PR-5 · v0.4.5 (signature), v0.5 (R(x) leg)**

- Two regimes, kept apart. (a) Fixed-R control signature, where certainty equivalence
  *holds*: certify `J_CE = J*` exactly by separation, bracket width `= J_LQG − J_LQR` in
  closed form, `η_ctrl = 0` to a stated floor. (b) R(x) control leg at horizon > 1, where
  certainty equivalence is genuinely *suboptimal*: `J_agent < J_CE` and `η_ctrl ≠ 0`, so
  the comparison is real rather than degenerate.
- Build: finite-horizon Riccati, needed twice: the full-information floor `J_lower` and
  the matched-horizon comparison against the EFE planner. **Match the horizon.** A
  receding-horizon planner at horizon H implies the finite-horizon gain with zero
  terminal cost, which converges to but does not equal the steady-state gain. An
  unmatched comparison shrinks with H and looks exactly like a bug.
- Falsify (the *independence* claim, not the FEP): the R(x) bracket and the crossover
  are fully predicted by dual control with no active-inference-specific content. Expected
  outcome; state it as such.
- Does not buy: corroborates the *unification* thesis, which is P1's actual claim. Not
  an independent-prediction thesis. Do not let a reader take the bridge as novel physics.

**E2 · vs reward-plus-information-bonus** · DEMARCATE · rival-agent harness · tier `BOUNDED` · **further**

- Test: search for any `reward + λ·(info gain)` agent reproducing Agent B's
  action-dependent epistemic value **and** the crossover H\* simultaneously.
- Constraint: λ and the precision parameter γ are each declared **fixed or free in the
  model specification**, never at analysis time. A free parameter discovered during
  analysis is an accommodation; one declared in the spec is a hypothesis.
- Falsify (independence): one λ reproduces both signatures across tasks.
- Does not buy: separates "the FEP predicts the trade-off" from "the FEP re-labels a
  trade-off already tuned by hand". Both publishable; only one supports independence.

**E3 · observational equivalence, why the battery is in silico** · DEMARCATE · toolbox B seam · **PR-3 · v0.4.5**

- Build: the `World`/`Agent` seam, where the type system makes it impossible for the
  agent to read the world's parameters (the discipline of `IncompatibleLinearizationError`).
  A test asserts the absence of the path, not that it is unused.
- **The pinned conjunct that costs something** (ledger section 6): driving a **common
  exogenous action sequence** is what makes `H(p*)` cancel. It also severs the control
  loop. Under R(x) an agent choosing its own actions steers toward low-noise regions and
  changes its own inference gap, so a gap measured under an imposed sequence can
  misrepresent the closed-loop gap. Record it on the result object as a declared,
  contestable modelling choice. Never read exogenous-action results as closed-loop.
- Falsify: none. Boundary condition under which A–D are severe at all.

---

## F — The wall

**F1 · strong-FEP identifiability from behaviour** · DEMARCATE · **closed**

- Status: answered in the negative by the good-regulator theorem and inverse-RL
  non-identifiability. Any behaviour is FE-minimising for some (model, preferences).
  Untestable as posed.
- Open lever: *multi-environment* identifiability. A system minimising free energy across
  many environments with a *shared* generative model is more constrained than one observed
  in a single environment. The only route through the wall not already closed. Revisit
  only if it bites harder than it currently looks.

**F2 · scoring a real organism** · DEMARCATE · **further**

- Status: the instrument measures the gap between a *known* p\*. Aimed at an organism,
  p\* must be estimated, which is exactly the non-identifiable step the in-silico design
  dissolves.
- Bridge: certified estimation of p\* with error bars propagating into the two KL terms
  would have to exist first. Name the bridge; do not build it.

**F3 · E. coli as a structural instance** · FALSIFY · **further**

- Claim only that its sensing instantiates Thm 1's *structure*, reachable
  heteroscedasticity in the observation channel. Attribute **no** generative model to the
  organism.
- Predict (structural, deferred item 12, needs nothing new theoretically): chemotactic
  sensing is reachably state-dependent, the kinase readout's reliability depending on a
  controllable operating point, placing E. coli inside the R(x) model class.
- Falsify: the organism's sensing is not reachably heteroscedastic.
- **Do not conflate** with deferred item 13, reproducing Mattingly et al. (2021)'s
  quantitative benchmark (η ≈ 0.65, β ≈ 0.22 bits/s/mm², v₀ ≈ 22.6 µm/s, external and
  deferred). That needs point-process emissions (item 2), a rate-distortion module
  (item 7) and directed-information estimation (item 8), and still does not cross F1 or
  F2. The structural instance attributes nothing and needs nothing.
- Does not buy: hold it to "instance of the structure", never "confirmation of the FEP".

---

## Standing prohibitions

1. **Never obtain a term by subtracting `H(p*)`.** Under a common exogenous action
   sequence it is a shared constant and cancels, leaving two reparameterisation-invariant
   KL divergences computable without estimating an entropy.
2. **Never route A2 through Corollary 2.** Def 2 is a planning reduction; the
   biconditional is false in general.
3. **Never present B5's witness as the theorem.** Negative existential, code witnesses
   one route.
4. **Never claim C5 refutes the FEP.** It demarcates incompleteness of the received
   accounting.
5. **Never report a separation without its ratio and conditioning.**
6. **Never call a budget overrun or an empty window a null.** Both are `VOID`.
7. **Never read an exogenous-action result as closed-loop.**
8. **Never quote H\* = 7 without the upper-bound qualifier and the action mode.**
9. **Never let a declared set go unversioned.** Action sets, rule ladders, constructor
   crosses, seed lists, functional variants each live in the model spec, so an addition
   after results are seen appears in the diff rather than in the prose.
10. **Never print a sample and an enumeration as the same `PASS`.** A grid sample of a
    continuum corroborates. A fully enumerated finite set with a cardinality certificate
    decides.

---

## Readiness roll-up

| Tag | Carries | Gates |
|---|---|---|
| **v0.4.2 ✓** | P1 witness | A1–A3, B1–B5 |
| **v0.4.3 ✓** | `cpomdp.diagnostics`, per-claim theorem suite, FFG PD fixes. **Not** the scoring harness | — |
| **v0.4.4 ✓** | multi-step EFE, exhaustive enumerator + completeness certificate | D3's measurement |
| **v0.4.5** | PR-1 warrant vocabulary · PR-2 R10 hardening · PR-3 World/Agent seam · PR-4 scoring harness · PR-5 control bracket + Part 1 results · PR-7 reference filter + rule family · PR-8 **GATE-D4** | C1–C5, E3, E1 signature, D3 completion, D4 |
| **v0.5** | PR-9 window harness + Part 2 results · PR-10 Paper 3 Part 2 · PR-11 release | C6, D1, D2, E1 R(x) leg. R1–R10 all print and assert |
| **further** | rival harness, parameter estimation, point-process and rate-distortion modules | E2, F2, F3 |
