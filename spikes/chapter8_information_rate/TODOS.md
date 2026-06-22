
# TODOs — what to build to honestly test the Mattingly upper bound

A **linear** task list: work it top to bottom. It is the [SPIKE.md](SPIKE.md) roadmap
broken into implementable units. Each task carries its **done-when** (the oracle/test that
keeps it honest). Where the requirement is not pinned down — usually because it is a
theoretical or literature question — it is flagged inline and collected again in
[Appendix A](#appendix-a--open-theoretical--literature-questions). Numbers still to be read
off the primary source are in [Appendix B](#appendix-b--values-to-confirm).

The ordering is a dependency chain: each phase's claim must be defensible before the next
builds on it. Do not skip ahead — a later phase resting on an unverified earlier one
re-introduces exactly the artefacts this spike exists to prevent.

**Legend.** `[ ]` todo · `[x]` done · `[~]` partially done.
🔬 = open theoretical / literature question (see Appendix A). 📏 = a number to confirm
(see Appendix B). 🚧 = engineering only, no open question.

---

## Standing acceptance gates (always-on; re-check every phase)

These are the don't-fudge guardrails from SPIKE.md section 4, restated as a checklist the
work is measured against at every step. A green phase that breaks one of these is not green.

- [ ] The EFE epistemic output is named `I_acquired_ceiling` (nats) everywhere; the symbol
  `İ` / `İ_{s→a}` is reserved strictly for the behavioural rate. The ceiling-vs-behavioural
  caveat is a **runtime assertion**, not a comment.
- [ ] No rate is ever manufactured by dividing a horizon-summed epistemic by `H`
  (`policy_efe` propagates predict-only belief that contracts between steps). Rates come only
  from a single-step gain × an externally-justified sensing frequency.
- [ ] The binding precision limit is sited in internal processing noise `Q(x)`, not purely in
  sensor `R` — an R-limited agent is arrival-limited, the picture the 2025 follow-up
  overturned (📏 η_internal ≈ 0.014). A guard asserts the swept knob lives in `Q`.
- [ ] Any ceiling-based "below the curve" plot is banner-labelled **non-falsifiable / vacuous**.
  The falsification test plots the deflated `İ_{s→a}`, never the ceiling.
- [ ] Any point **above** the curve halts as a hard units/mapping-bug error — never reported.
- [ ] `v_d` is net up-gradient drift, never instantaneous `v_0`; nats→bits is applied exactly
  once; every dimensional constant traces to a citation.
- [ ] Ceiling and best-available behavioural rate are reported as **separate rows** showing
  `İ_ceiling ≥ İ_{s→a}` (RFC-001's three-row honesty rule); never one efficiency scalar.

---

## Phase 0 — Verify the bit-counting (gate)

*Nothing biological proceeds until this is green.*

- [x] **Independent MI oracle.** Assert the kernel epistemic term equals `I(state;obs)`
  computed a second way (state-side `½ln(det Σ⁺/det Σ_post)` vs the kernel's obs-side
  `½ln(det S/det R)`) to machine precision. — *Done in
  [`demo_information_rate.py`](demo_information_rate.py)`::phase0_oracle` (err 2.2e-16, both
  `C=I` and general `C`).* 🚧
- [ ] **Promote the oracle into `tests/`.** Move it to `tests/test_information_rate.py` as a
  permanent regression so a future kernel edit can't silently break the bit-count.
  *Done-when:* the test runs in the suite and fails if the epistemic sign/scale changes. 🚧
- [ ] **Single nats→bits site.** One helper (or one documented call site) does `/ln2`; assert
  it is applied exactly once on any path that reports bits. *Done-when:* a test exercises a
  ceiling-rate path and checks the bit value against a hand-computed nats/ln2. 🚧

## Phase 1 — Energy / rate instrumentation layer (RFC-001 section 4)

*Turn the live, discarded epistemic term into a labelled, attributable, task-pinned quantity.*

- [ ] **`energy.py` → `BeliefCost`.** `kl_nats(prior, posterior)`, `bits = kl_nats/ln2`,
  `landauer_joules(bits, T)`. Closed-form Gaussian KL from the covariances (no sampling).
  *Done-when:* matches an independent closed-form Gaussian-KL oracle to machine precision. 🚧
- [ ] 🔬 **Pick the bit-count convention.** RFC-001 open question 1: bound by total-KL or by
  entropy-reduction `H(prior) − H(posterior)` — they differ. Choose and justify in the
  `EnergyReport.convention` field. *Gates:* what "bits erased per cycle" means. *Consult:*
  RFC-001 section 4.1; Landauer (1961); Still et al. (2012) thermodynamics of prediction.
- [ ] **Closed-loop epistemic logger.** Record the *realised* per-step epistemic gain along the
  `Agent` loop — fixing the discard at [`selection.py:181`](../../src/cpomdp/selection.py#L181)
  (`sample_action` keeps only `[0]`) and the fact that `Agent` never logs the split.
  *Done-when:* logged realised nats == `expected_free_energy(model, belief, action, pref)[1]
  ["epistemic"]` recomputed at the chosen action, every step. 🚧
- [ ] **`TaskSpec` / `EnergyReport` / `EnergyProbe` scaffolding** (RFC-001 4.2–4.4). Minimal is
  fine; no energy figure emitted without a `TaskSpec`. *Done-when:* a report carries the
  three honesty rows (floor / reference / measured) and refuses to emit a lone efficiency
  scalar. 🚧
- [ ] **Forbid horizon-sum-as-rate in code.** A guard/assert so no caller divides the
  `policy_efe` summed epistemic by `H`. 🚧
- [ ] **Zero hot-path cost when disabled** (RFC-001 design principle). *Done-when:* a
  byte-identical-`G` test with instrumentation off. 🚧

## Phase 2 — Chemotaxis arena (RFC-002)

*The substrate every quantitative number runs on. Still a linear channel here — explicitly
NOT yet a Mattingly comparison.*

- [ ] **Concentration field + world-step + run dynamics.** A gradient `c(x)`, the true-plant
  update, and trajectory recording. 🚧
- [ ] **`v_d` = net up-gradient drift metric** (not instantaneous `v_0`); arm the `v_d > v_0`
  tripwire. *Done-when:* the metric matches an analytic expected drift on a known-gradient toy
  with a fixed policy. 🚧
- [ ] 🔬📏 **Choose the gradient profile and steepness.** Mattingly used shallow, ~cm-scale
  gradients where `İ = β·g²` holds in linear response. *Gates:* whether the agent sits in the
  regime the bound was derived for. *Consult:* Mattingly 2021 methods (gradient steepness `g`
  in mm⁻¹, the exponential ramp); Kalinin 2009 (log-sensing) for the profile shape.
- [ ] **Defensible claim checkpoint:** "the agent climbs a gradient; I can read a ceiling rate
  and a `v_d`" — banner it as **not** an E. coli comparison (linear channel).

## Phase 3 — Nonlinear MWC / log sensor (`NonlinearSensor`, Phase 2.5)

*Hard prerequisite, not a refinement: a linear-Gaussian channel provably cannot represent the
cell's sensing (RFC-003 section 4.4). Without this every İ is measured on a fiction.*

- [ ] 🔬 **Curved-mean sensor with full 2nd-order Gaussianization.** Mean correction **and**
  covariance/curvature together (taking one without the other is a real bug, per ADR-006's
  deferred note), restoring the `½tr(H_R·Σ⁺)` Jensen term that `CallableSensor` drops.
  *Gates:* the whole channel. *Consult:* DECISIONS.md ADR-006 "Deferred to Phase 2.5"; the
  build plan's pinned full-2nd-order formula and its dual-oracle definition-of-done.
- [ ] 🔬📏 **MWC receptor free energy** `F = N·ln[(1+c/Koff)/(1+c/Kon)] − α·m`, with the slow
  methylation `m` as an adaptation state. *Gates:* the sensing nonlinearity and fold-change
  behaviour. *Consult:* Shimizu, Tu & Berg (2010) for `N, Koff, Kon, α`; Lazova (2011) and
  Tu/Shimizu for the `m` adaptation dynamics and timescale.
- [ ] **Dual oracle** (mean + curvature) per the Phase-2.5 spec. *Done-when:* both moments match
  the oracle to tolerance. 🚧
- [ ] **Fold-change-invariance test.** Resolved information is ~scale-invariant in
  log-concentration over decades (Lazova 2011's ~10⁴-fold). *Done-when:* the test passes;
  a linear channel demonstrably fails it (keep that as the contrast). 🚧
- [ ] **Apply the identical channel to every model compared** (flat vs hierarchical, ceiling vs
  behavioural) so no comparison is biased by an unmodelled nonlinearity. 🚧

## Phase 4 — Internal-noise siting + physical parametrisation

*Put the agent in the cell's units, with the limit where the biology puts it.*

- [ ] **Site the binding limit in `Q(x)`** (internal processing noise), asserted in code; reject
  an R-only-limited configuration. 🚧
- [ ] 🔬📏 **Map `A`, `Q(x)`, `R(x)`, `dt`, sensing-rate from cited cell parameters.** This is a
  modelling/calibration step with no single right answer — defend each choice. *Gates:* whether
  `İ` and `v_d` land in Mattingly's units honestly rather than by tuning. *Consult / pin:*
  - 📏 `v_0 = 22.61 µm/s` (confirmed).
  - 📏 `D_r` rotational-diffusion coefficient (sets how perishable each bit is) — Berg lineage /
    Mattingly methods.
  - 📏 adaptation timescale `τ_m` — Shimizu/Tu/Berg, Lazova 2011.
  - 📏 kinase response `K(ω)` and noise `N(ω)` magnitudes — Mattingly 2021 FRET
    (CheY-mRFP / CheZ-mYFP) measurements.
  - 📏 `η_internal ≈ 0.014` (2025 follow-up) as the justification that internal noise, not
    arrivals, binds.
- [ ] **Provenance gate.** *Done-when:* every dimensional constant in the parametrisation has a
  citation attached and a test asserts none is a bare literal. 🚧

## Phase 5 — Behavioural rate: run/tumble channel + directed-information projection

*The load-bearing caveat made real. This is where the ceiling becomes the comparable
`İ_{s→a}`, and it is the hardest, most open phase.*

- [ ] 🔬 **Stochastic-policy (run/tumble) selector.** Replace deterministic argmin with a
  softmax over `G` and a sampled action, modelling the run/tumble decision the cell actually
  makes (on ≪1 bit). *Gates:* whether the agent's action channel is the one Mattingly's rate is
  defined over. *Consult:* Friston et al. on policy posterior + precision `γ`; Baltieri &
  Buckley (2019) PID-as-active-inference; RFC-005 step 1 (run/tumble ↔ pragmatic policy
  sampling).
- [ ] 🔬 **Compute `İ_{s→a}` as a directed-information RATE, mirroring Mattingly's estimator.**
  This is the crux and the deepest open item. Mattingly's `İ` is **not** a single-step mutual
  information — it is a spectral, directed (transfer-entropy) signal→action rate, reduced in
  linear response to `∫ df · log(1 + SNR(ω))` built from three spectra: signal power `S(ω)`,
  kinase frequency response `K(ω)`, kinase noise power `N(ω)`, keeping only the forward
  signal→action term (discarding behaviour→signal feedback). The silicon side must mirror
  *that*, which requires three things the kernel does not do:
  - the **directed/transfer-entropy rate** analogue of the per-step `½ln(det S/det R)` (a rate
    over the closed loop, not a one-shot gain);
  - the **action-relevant-subspace projection** that subtracts the irrelevant bits (only
    uncertainty about "am I going up-gradient, how steeply" counts);
  - the **belief→run/tumble channel loss** (a data-processing-inequality drop with no term in
    the linear-Gaussian kernel).
  *Consult:* Mattingly 2021 SI (the spectral estimator, eqn ~102); Tostevin & ten Wolde
  (2009/2010) information transmission in biochemical networks; Barato/Hartich/Seifert on
  directed/transfer information rates; Tishby information bottleneck (for the relevant-bits
  projection). *Done-when:* `İ_{s→a} ≤ I_ceiling` holds at runtime AND, on a toy where the
  directed rate is analytically reducible (a 1-D gradient-sign decision), the computed rate
  matches the closed form.
- [ ] **`p>1` action search (`GradientEFESelector`).** Lift the hard `p=1` /
  constant-action restriction so 2-D steering and "move-to-sense-then-exploit" are expressible
  and the drift is not artificially crippled vs the cell. *Done-when:* matches a brute-force
  EFE oracle on a 2-D toy; the existing p=1 path stays byte-identical. 🚧 (deferred v0.4 seam)

## Phase 6 — The curve test

- [ ] 📏 **Bound oracle.** Encode both forms: the empirical `v_d = χ·(İ/β)^½`, and the Eqn-1
  ceiling `v_d/v_0 ≤ f(θ)·(ln2·İ_{s→a}/(c·D_r))^½`. *Gates:* the literal comparison. *Confirm
  from the typeset PDF:* the denominator constant `c` (2 vs 12·D_r — two fetches disagreed),
  `χ ≈ 4300 µm²/s`, `β ≈ 0.22 bits/s/mm⁻²`, `f(θ) ≤ 1`, `v_0`.
- [ ] **Pre-register the sqrt exponent + CI before any sweep.** Lock the predicted exponent so it
  can't be tuned post-hoc; treat exponent ≠ 0.5 (outside CI) as a real finding. 🚧
- [ ] **Sweep + plot.** Produce `(İ_{s→a}, v_d)` pairs across the gradient/precision sweep, plot
  against the curve, arm the above-curve tripwire, report ceiling and behavioural rows
  separately. 🚧
- [ ] **Defensible claim checkpoint:** a real, falsifiable "does AIF fall below the curve"
  result — the first point in this roadmap where that sentence is honest.

## Phase 7 — Derivation (parallel research; gates the *strong* claim only)

- [ ] 🔬 **Is the sqrt law derivable from EFE, or only fittable?** Attempt a closed-form
  linear-response derivation of `v_d ∝ √İ` from minimising `G` around the gradient-climbing
  fixed point — or document the finding that AIF yields a different exponent. A sweep that lands
  near 0.5 without this derivation is **consistency, never identity of structure**, and must be
  reported as such. *Consult:* Sajid et al. (2021) EFE → Bayesian optimal experimental design
  (the flat-preference and no-ambiguity limit reductions); the Fisher-information chain
  Mattingly uses (rate → estimate precision → drift). *Done-when:* the analytic exponent is
  derived and compared to the pre-registered swept one — or the divergence is written up as a
  result about active inference.

---

## Appendix A — Open theoretical / literature questions

The items above tagged 🔬, gathered so none is lost. Each says what it gates and where to look.

|#|Question|Gates|Where to resolve it|
|---|---|---|---|
|A1|Bit-count convention: total-KL vs entropy-reduction|what "bits erased/cycle" means (Phase 1)|RFC-001 4.1; Landauer 1961; Still et al. 2012|
|A2|Gradient profile + steepness regime|whether the agent is in the bound's linear-response regime (Phase 2)|Mattingly 2021 methods; Kalinin 2009|
|A3|Correct full-2nd-order Gaussianization (mean **and** curvature)|the entire nonlinear channel (Phase 3)|DECISIONS.md ADR-006 deferred note; build-plan Phase-2.5 spec|
|A4|MWC parameters `N, Koff, Kon, α` + methylation `m` dynamics|the sensing nonlinearity + fold-change (Phase 3)|Shimizu/Tu/Berg 2010; Lazova 2011|
|A5|Physical parametrisation `A/Q(x)/R(x)/dt`, `D_r`, `τ_m`, `K(ω)`, `N(ω)`|İ and v_d in honest units (Phase 4)|Mattingly 2021 (FRET); Berg lineage|
|A6|Run/tumble as an AIF stochastic policy (softmax over G, precision γ)|the action channel İ is defined over (Phase 5)|Friston policy posterior; Baltieri & Buckley 2019; RFC-005 step 1|
|**A7**|**The directed-information `İ_{s→a}` estimator: directed/transfer-entropy rate + action-relevant projection + belief→behaviour loss**|**the behavioural rate — the difference between a real test and an overclaim (Phase 5)**|**Mattingly 2021 SI (spectral estimator); Tostevin & ten Wolde 2009/2010; Barato/Hartich/Seifert; Tishby information bottleneck**|
|A8|Is `v_d ∝ √İ` derivable from argmin G, or only fittable?|the strong "identity of structure" claim (Phase 7)|Sajid et al. 2021; Mattingly's Fisher-info chain|

**A7 is the one to worry about most.** Everything else is buildable engineering or a value to
look up. A7 is where "the bits-to-behaviour bound falls out of the AIF objective" is either
earned or quietly abandoned — and it may turn out that the honest answer is a documented
*difference* from Mattingly rather than a match. That is still a result; pre-registration
(Phase 6) is what keeps it one.

## Appendix B — Values to confirm

All TO-CONFIRM against the **typeset Nature Physics PDF / SI**, not a fetch summary. Until
confirmed, no published number may lean on them.

|Symbol|Value (provisional)|Status|Source|
|---|---|---|---|
|`v_0`|22.61 ± 0.07 µm/s|**confirmed**|Mattingly 2021|
|`η` (near-bound efficiency)|0.65 ± 0.05 (≈ "factor of two")|"factor of two" confirmed; 0.65 TO-CONFIRM|Mattingly 2021 fig caption|
|`İ_{s→a}` magnitude|~10⁻² bits/s at the operating point|order confirmed; exact TO-CONFIRM|Mattingly 2021|
|`β` (in `İ = β·g²`)|0.22 ± 0.03 bits/s/mm⁻²|form confirmed; value TO-CONFIRM|Mattingly 2021 SI|
|`χ` = RFC's `X` (in `v_d = χ·g`)|4300 ± 150 µm²/s|TO-CONFIRM|Mattingly 2021 results|
|bound constant `c`|**2 vs 12 · D_r — unresolved**|TO-CONFIRM (fetches disagreed)|Mattingly 2021 Eqn. 1|
|spectral-estimator prefactor|"makes units bits/s"|TO-CONFIRM|Mattingly 2021 SI eqn ~102|
|`η_internal` (2025)|0.014 ± 0.002 (~1.4% of physical limit)|TO-CONFIRM|Mattingly et al. 2025|

**Primary sources.** Mattingly, Kamino, Machta & Emonet, *E. coli chemotaxis is information
limited*, Nature Physics 17, 1426–1431 (2021), doi:10.1038/s41567-021-01380-3 (PMC8758097;
arXiv:2102.11732). Follow-up: Mattingly et al., *E. coli chemosensing accuracy is not limited by
stochastic molecule arrivals*, Nature Physics (2025), doi:10.1038/s41567-025-03111-4 (preprint
bioRxiv 2024.07.09.602750).
