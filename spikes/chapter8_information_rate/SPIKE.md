# Spike: can cpomdp v0.3 reach the RFC-001 chapter-8 Mattingly information-rate test with pure active inference?

- **Status:** Spike / feasibility study — verdict reached
- **Date:** 2026-06-22
- **Scope:** Whether the `İ`-vs-`v_d` "below the theoretical-maximum curve" test of RFC-001 chapter 8 is reachable on v0.3 as it stands, and if not, exactly what is missing.
- **Companion:** [`demo_information_rate.py`](demo_information_rate.py) — the runnable half of this spike (it executes; this document explains).
- **Method:** code audit (`efe.py`, `observation.py`, `dynamics.py`, `selection.py`, `kalman.py`), RFC/ADR read, web research on the primary paper, an adversarial honesty pass, and four independent refutation passes on the load-bearing information-theory claims. The Phase-0 bit-counting check below is verified to machine precision by the demonstrator.

---

## 1. The objective

Mattingly, Kamino, Machta & Emonet (*Escherichia coli chemotaxis is information-limited*, Nature Physics 17, 1426–1431, 2021; PMC8758097; arXiv:2102.11732) measured that a cell climbing a concentration gradient acquires far less than one bit per decision about whether it is heading up-gradient (order 10⁻² bits/s in shallow, cm-scale gradients), yet converts that information into up-gradient drift at within a factor of two of the theoretical maximum — efficiency η ≈ 0.65 ± 0.05 (the RFC's "~66%"). "The Mattingly curve" is the rate-distortion bound tying the directed **signal→action** information rate `İ_{s→a}` (bits/s) to the maximum up-gradient drift speed `v_d`. It has two forms RFC-001 conflates: (A) the bound proper, `v_d/v_0 ≤ f(θ)·(ln2·İ_{s→a}/(c·D_r))^½` with `f(θ) ≤ 1`; (B) the empirical scaling `v_d = χ·(İ/β)^½`, which follows from drift being linear in gradient steepness and rate quadratic in it. `v_0 = 22.61 ± 0.07 µm/s` (CONFIRMED); RFC-001's symbol `X` is the paper's `χ`. **Reaching the test** means: parametrise an active-inference agent to the cell's sensing physics, measure its `(İ_{s→a}, v_d)` pairs, and *definitively* test whether they fall below this curve — with the result a real finding, not an artefact.

**Provenance flags (do not fudge these).** `v_0 = 22.61 µm/s` confirmed. `η = 0.65 ± 0.05`, `χ ≈ 4300 µm²/s`, `β ≈ 0.22 bits/s/mm⁻²`, the bound constant `c` (`2` vs `12·D_r` — two fetches disagreed, the arXiv glyphs would not extract), and the 2025 follow-up's internal-noise efficiency (`η_internal ≈ 0.014`) are all **TO-CONFIRM against the typeset PDF** before any number leans on them.

## 2. Verdict

**NO — not reachable, and not defensibly, with pure active inference on v0.3 as it stands.**

The sharpest single reason: **v0.3 computes only the perceptual ceiling — the signal→belief mutual information `I(state;obs)` — which is a strict upper bound on Mattingly's behavioural signal→action rate `İ_{s→a}`. Plotting the agent at `(ceiling, v_d)` lands below the curve *by construction*, so it tests nothing.** Because `İ_ceiling ≥ İ_true` and the bound `v_max(İ)` rises with `İ`, evaluating the curve at the inflated rate puts it higher, making the same `v_d` a smaller fraction of it. The bound cannot bite at an over-estimated x-coordinate (section 4 derives this).

It is a **partial**-no in one narrow sense: the *kernel arithmetic* — the bit-counting itself — exists and is exact (Phase-0 oracle passes to 2e-16). Everything that would turn it into a Mattingly-comparable number is unbuilt.

## 3. What v0.3 can honestly measure today

The one fully-built ingredient is the perceptual-ceiling information gain. For the linear-Gaussian observation model the EFE epistemic term

```text
epistemic = ½·(ln det S − ln det R),   S = C·Σ⁺·Cᵀ + R,   Σ⁺ = A·Σ·Aᵀ + Q(x)
```

is **exactly** the per-observation mutual information `I(state;obs)` in **nats** — verified against information theory and against [`efe.py:284-288`](../../src/cpomdp/efe.py#L284-L288), and cross-checked an independent way (state-side `½ln(det Σ⁺/det Σ_post)`) to machine precision by the demonstrator's Phase-0 gate. It surfaces as `expected_free_energy(...)[1]["epistemic"]` per action, and `policy_efe(...)[1]["epistemic"]` sums it over a constant-action rollout.

To read a perceptual-ceiling rate off this today — **outside the library**:

1. Run the `Agent` loop; at each chosen action call `expected_free_energy(model, agent.belief, action, preference)[1]["epistemic"]` to recover the realised nats. The library discards this — `sample_action` takes `[0]` at [`selection.py:181`](../../src/cpomdp/selection.py#L181) and `Agent` never logs it.
2. Convert nats→bits: `bits = nats/ln2`. **There is no `/ln2` anywhere in `src`**; the caller applies it exactly once.
3. Divide by `n_steps·dt`, where `dt` is *your external* mapping of a step to wall-clock sensing time. The library has **no notion of physical time, no `dt`, no sensing-rate field** — RFC-001 section 4's energy layer is unbuilt.

The demonstrator does exactly this and prints `I_acquired_ceiling = 2.5 bits/s, v_drift = 0.74 units/s` on a synthetic precision-well task. That number is real and honestly computed. It is **not** Mattingly-comparable, for the reasons in section 4 — which the demonstrator prints rather than hides.

## 4. Honesty guardrails

**Signal→belief vs signal→action.** The kernel yields `I_acquired_ceiling` (perception-side). Mattingly's `İ_{s→a}` is end-to-end, with irrelevant bits and communication losses already subtracted. Three subtractions the kernel does **not** perform separate them: (i) projection through the action-relevant subspace — only resolved uncertainty bearing on "am I going up-gradient, how steeply" counts; the det-ratio counts *all* state dimensions including nuisance ones; (ii) the encoding loss from belief to the discrete run/tumble channel (a data-processing-inequality loss with no term in the linear-Gaussian kernel); (iii) the rate conversion (per-observation nats → per-second bits needs a defensible, cited sensing frequency, not a tuned knob). So `İ_ceiling ≥ İ_{s→a} = I_acquired − I_irrelevant − I_commloss`, all three subtractions **unbuilt**.

**Exact inequality direction — why a ceiling-based "below the curve" is vacuous.** Derived from scratch and independently confirmed. `v_max(İ) = χ·(İ/β)^½` is monotone increasing in `İ`. Since `İ_ceiling ≥ İ_true`, `v_max(İ_ceiling) ≥ v_max(İ_true)`: the curve at the inflated rate sits **higher**, so a fixed `v_d` clears it more easily and sits more comfortably below. There is a wedge `v_max(İ_true) < v_d ≤ v_max(İ_ceiling)` that **violates the true bound yet passes the ceiling test**. Bluntly: using the ceiling makes "below the curve" trivially easy and a bound violation essentially impossible to observe — the test is vacuous, not honest. The only meaningful "below the curve" uses the deflated behavioural `İ_{s→a}`, where the bound can actually bite. Complementary tripwire: any point **above** the curve is physically impossible and is a **hard units/mapping bug alarm** (`v_0` mistaken for `v_d`, nats/bits confusion, wrong `β`/`χ`) — never a result.

**Channel fidelity — fatal for any E. coli claim on v0.3.** RFC-003 section 4.4 states outright that a strictly linear-Gaussian channel cannot represent the documented sensing: log-concentration (Kalinin 2009), fold-change detection invariant over ~10⁴-fold (Lazova 2011), MWC receptor free energy `F = N·ln[(1+c/Koff)/(1+c/Kon)] − α·m` (Shimizu/Tu/Berg 2010). The infidelity is **mean-side**: the response lives in the transduction mean `F(c)`, nonlinear in `c` and adaptation-coupled. The one live state-dependent seam, `CallableSensor` `R(x)`, varies only the *noise covariance* with a constant linear mean `C·x` — it **cannot** narrow this gap. A linear channel makes resolved information depend on *absolute* concentration, overstating it at low and understating at high, and never showing fold-change invariance — the whole point of the biology. Any "Mattingly ballpark" on a linear channel is an artefact of the unmodelled nonlinearity. The MWC/log `NonlinearSensor` (Phase 2.5, curved mean + 2nd-order gaussianize, restoring the `½tr(H_R·Σ⁺)` Jensen term `CallableSensor` drops) is a **hard prerequisite, not a refinement**, applied identically to every model compared.

**Is the sqrt law derivable from EFE, or only fittable? — the central scientific risk.** It is **not presently derivable** from the EFE objective as implemented; at best it is fittable, and a fit proves nothing. Two claims must not be conflated: (A) **identity of structure** — `v_d ∝ √İ` falls *analytically* out of minimising `G`; (B) **consistency** — sweep a knob, the points happen to lie near a sqrt curve. Only (A) is a result. (B) is curve-fitting that *any* diminishing-returns mechanism passes, because √ is the generic small-signal Fisher-information/CLT scaling (accuracy ∝ √samples; speed monotone in accuracy ⇒ speed ∝ √rate almost regardless of mechanism). Mattingly's exponent comes from a specific chain: rate → gradient-estimate precision (Fisher info, linear in `İ`) → drift via a response *linear in estimated gradient near optimum* → `v_d ∝ √İ`. The cpomdp agent's speed instead emerges from `EFESelector`'s greedy argmin of `pragmatic − epistemic` over a grid (p=1, constant-action), governed by preference precision `Λ`, `n_candidates`, `horizon`, and the arena — none pinned to the epistemic rate by any analytic relation. **There is no proof** that argmin `G` yields `v_d ∝ √(epistemic rate)`; the exponent is emergent and any of those knobs can bend it off 0.5. Honest position: reproducing 0.5 by sweeping is **necessary but far from sufficient**; the strong claim needs a closed-form linear-response derivation around the gradient-climbing fixed point, which is not in v0.3 and **may fail** — AIF may give a different exponent. **Pre-register the exponent and CI before any sweep; treat exponent ≠ 0.5 (outside CI) as a real finding about AIF, not a bug to tune away.**

**Standing guardrails (enforce as code, not footnotes):**

- Name the EFE epistemic output `I_acquired_ceiling` (nats) everywhere; reserve `İ` / `İ_{s→a}` strictly for the behavioural rate. Make the RFC-001 chapter-8 caveat a runtime assertion.
- Forbid dividing the horizon-summed epistemic by `H` to manufacture a rate — `policy_efe` propagates predict-only belief that contracts between steps, so the sum is not a rate. Rates come only from a single-step gain × an externally-justified sensing frequency. (The demonstrator obeys this: it reads the single-step gain at the chosen action, never the horizon sum.)
- Site the limiting knob in **internal processing precision `Q(x)`**, not purely sensor `R` — an R-limited agent is input/arrival-limited, exactly the picture the 2025 follow-up overturned. Assert the swept knob lives in `Q`.
- Banner any ceiling-based "below the curve" plot as **non-falsifiable / vacuous**; the falsification test plots the deflated `İ_{s→a}`.
- Tripwire: any above-curve point is a hard error.
- Pin `v_d` as **net up-gradient drift**, not instantaneous `v_0 ≈ 22.6 µm/s`; pin `β`, `χ`, `v_0` to cited values with provenance; assert nats→bits applied exactly once.
- Report ceiling and best-available behavioural rate as **separate rows** with `İ_ceiling ≥ İ_{s→a}` shown (RFC-001's three-row honesty rule); never collapse to one efficiency scalar.
- Oracle gate: for a fully-observed (`C = I`) linear-Gaussian toy where `I(state;obs)` is analytic, assert the epistemic term matches the closed form to machine precision *before* any biology mapping. (Done — `demo_information_rate.py::phase0_oracle`.)

## 5. Capability-gap ladder

|#|Capability|Status|Why chapter 8 needs it|What it unblocks|
|---|---|---|---|---|
|1|EFE epistemic = `I(state;obs)` in nats (perceptual ceiling)|**built**|The silicon information quantity; everything projects from it|Raw signal→belief gain ([`efe.py:284-288`](../../src/cpomdp/efe.py#L284-L288)); already flagged ceiling-not-`İ` in DECISIONS.md 45-47|
|2|State-dependent `R(x)`/`Q(x)` (non-collapse)|**built**|Under a fixed sensor the epistemic term is action-invariant and EFE collapses to LQR (ADR-003) — no speed/info trade-off|`CallableSensor` `R(x)`, `CallableProcessNoise` `Q(x)`, both perceived-on (ADR-008); `Q(x)` hosts the internal-noise constraint|
|3|nats→bits + sensing-rate multiply (`İ` assembly)|**planned-seam**|Kernel emits nats/obs; Mattingly is bits/s|The unit bridge for every comparison; named RFC-001 4.1 (`BeliefCost.bits = kl_nats/ln2`), exists nowhere in code|
|4|`BeliefCost`/`EnergyReport`/`TaskSpec`/`EnergyProbe` layer|**planned-seam**|The report is a task-pinned rate; "same work" must be well-posed|No `energy.py` exists; engineering only (closed-form Gaussian KL). Turns the live term into a labelled, attributable bits/s|
|5|Chemotaxis arena (RFC-002)|**unbuilt-hard**|The bound is about a cell climbing a gradient at `v_d`|Gradient field + world-step + drift/success metric; the substrate every quantitative number runs on (RFC-002 "pending")|
|6|Nonlinear log/MWC sensor (`NonlinearSensor`, Phase 2.5)|**unbuilt-hard**|Linear channel resolves the wrong quantity (RFC-003 4.4)|Curved mean + 2nd-order gaussianize + Jensen term + dual oracle; without it `İ` is measured on a fiction|
|7|Physical/dimensional parametrisation (µm/s, bits/s, adaptation timescale, `v_0`)|**unbuilt-research**|`İ` and `v_d` must be in Mattingly's units; the toolbox is dimensionless|The "parametrise to the real cell" step; a calibration with no single right answer — defend the `A`/`Q(x)`/`R(x)`/`dt` mapping from literature|
|8|signal→belief → signal→action projection|**unbuilt-research**|The load-bearing caveat; the ceiling over-claims as `İ`|The action-relevant-subspace projection that recovers `İ_{s→a}`; the line between identity-of-structure and overclaim|
|9|Run/tumble stochastic-policy action model|**unbuilt-hard**|Mattingly's rate is the run/tumble decision's; v0.3 `sample_action` is deterministic argmin|Softmax over `G` + sampled action; substrate for the behavioural-rate projection and for a cell deciding on ≪1 bit|
|10|sqrt-law bound oracle (`v_d = χ·(İ/β)^½` + efficiency)|**unbuilt-research**|The literal apples-to-apples curve test|Encodes `χ`, `β`, `v_0`, `c·D_r` from Mattingly; the only thing the agent's `(İ, v_d)` can be tested against|
|11|p>1 / sequential epistemic action search|**planned-seam**|2-D steering + "move to sense then exploit"; `EFESelector` is hard p=1|Deferred v0.4 `GradientEFESelector`; without it drift is artificially crippled vs the cell|

## 6. Sequenced roadmap

Ordered so each phase's claim is defensible before the next builds on it.

**Phase 0 — verify the bit-counting (gate).** Oracle test on a `C = I` (and general-`C`) linear-Gaussian toy asserting `epistemic` matches the analytic `I(state;obs)` to machine precision; nats→bits applied exactly once. *Done in the demonstrator (err 2e-16); promote it into `tests/` when the work starts.* No biology mapping proceeds until green.

**Phase 1 — energy/rate instrumentation.** `energy.py` with `BeliefCost` (`kl_nats`, `bits = kl_nats/ln2`), `EnergyProbe` wrapping the isolated EFE core, `TaskSpec`, `EnergyReport`; a closed-loop logger recording the *realised* per-step gain along the `Agent` loop (fixing the `sample_action` `[0]` discard). Forbid horizon-sum-as-rate in code. Oracle: closed-form Gaussian KL; assert logged realised nats == recomputed `expected_free_energy[...]["epistemic"]` per step. Keep the hot path lean — the probe is labelled, isolable work, not buried in the loop.

**Phase 2 — chemotaxis arena (RFC-002).** Gradient/food field, world-step, trajectory scoring with **`v_d` = net up-gradient drift** (not instantaneous `v_0`). Still a linear channel. Oracle: known-gradient toy with analytic expected drift for a fixed policy; tripwire rejecting `v_d > v_0`. Defensible claim: "the agent climbs a gradient and we can read a ceiling `İ` and a `v_d`" — explicitly **not** a Mattingly comparison.

**Phase 3 — nonlinear MWC/log sensor (`NonlinearSensor`, Phase 2.5).** Curved-mean sensor with full 2nd-order gaussianize (mean correction + curvature, restoring `½tr(H_R·Σ⁺)`); MWC free-energy `F(c)`; identical channel across every compared model. Oracle: the dual oracle from the Phase-2.5 spec; a fold-change-invariance test asserting resolved information is ~scale-invariant in log-concentration. Defensible claim: the channel resolves the *right* quantity.

**Phase 4 — internal-noise siting + physical parametrisation.** Place the binding limit in `Q(x)` (assert it lives in `Q`, not `R`); map `A`/`Q(x)`/`R(x)`/`dt`/sensing-rate from cited cell parameters (`v_0 = 22.61 µm/s`; adaptation timescale, signalling-noise magnitude). Oracle: provenance check — every dimensional constant traced to a citation; guard asserting the swept knob is in `Q`. Defensible claim: the agent is parametrised to the cell's physics with a defended mapping.

**Phase 5 — behavioural rate: run/tumble + action-relevant projection.** Stochastic-policy selector (softmax over `G`, sampled action) for run/tumble; the projection through the action-relevant subspace + belief→behaviour channel computing `İ_{s→a} = I_acquired − I_irrelevant − I_commloss`; `p>1` `GradientEFESelector` so the repertoire is not crippled. Oracle: on a toy where `İ_{s→a}` is analytically reducible (a 1-D gradient-sign decision), assert the computed behavioural rate matches; assert `İ_{s→a} ≤ I_ceiling` always. Defensible claim: a deflated behavioural `İ_{s→a}` exists where the bound can bite.

**Phase 6 — the curve test.** Bound oracle encoding `v_d = χ·(İ/β)^½` (and the `c·D_r` form once the 2-vs-12 constant is confirmed from the typeset PDF) with cited `χ`, `β`, `v_0`; a **pre-registered** sqrt exponent + CI; a sweep producing `(İ_{s→a}, v_d)` pairs plotted against the curve; the above-curve tripwire armed. Oracle: the Mattingly bound itself; the exponent pre-registration is the honesty instrument. Report ceiling and behavioural rows separately. Defensible claim: a real, falsifiable "does AIF fall below the curve" result.

**Phase 7 (parallel research, gates the *strong* claim) — derivation.** A closed-form linear-response derivation of `v_d ∝ √İ` from argmin `G` around the gradient-climbing fixed point — or a documented finding that AIF gives a different exponent. Oracle: the analytic exponent vs the pre-registered swept one. A consistency-only result (sweep near 0.5 without derivation) is reported as consistency, **never** as identity of structure.

## 7. Definition of done — "definitively test whether AIF falls below the curve"

The result is meaningful (not an artefact) iff **all** of:

1. **Bit-counting verified** — epistemic term matches analytic MI to machine precision (Phase 0); nats→bits applied exactly once.
2. **Behavioural rate, not ceiling** — the x-coordinate is `İ_{s→a}` (deflated), via the action-relevant projection + belief→run/tumble channel, with `İ_{s→a} ≤ I_ceiling` asserted at runtime. Plotting the ceiling is banned as vacuous.
3. **Right channel** — the MWC/log `NonlinearSensor` is in use, identical across compared models; fold-change invariance demonstrated. No linear-channel number is called an E. coli result.
4. **Limit in `Q(x)`** — the binding constraint is sited in internal processing noise, asserted in code; an R-only-limited curve is rejected as the wrong organism.
5. **Pre-registered exponent** — the sqrt exponent and CI are fixed before the sweep; no post-hoc tuning of `Λ`, grid, horizon, sensing-rate, or arena to move it toward 0.5. exponent ≠ 0.5 is a finding.
6. **Honest sensing-rate** — the nats→bits/s frequency is a cited physical parameter with provenance, never fit to land in Mattingly's magnitude.
7. **`v_d` pinned correctly** — net up-gradient drift, not instantaneous `v_0`; `β`, `χ`, `v_0` (and the bound constant) cited with provenance.
8. **Tripwires armed** — any above-curve point halts as a units/mapping bug; ceiling and behavioural rates reported as separate rows showing the inequality.
9. **Strong vs weak claim separated** — "identity of structure" is asserted **only** if Phase 7's derivation succeeds; otherwise the result is reported as a falsifiable consistency test, explicitly.

Absent any one of these, the result is an artefact and must not be reported as a Mattingly comparison.

## 8. The minimal honest experiment v0.3 can run today

[`demo_information_rate.py`](demo_information_rate.py) — runs now, on a synthetic linear-Gaussian precision-well task with a live `CallableSensor` `R(x)`. It (a) gates itself with the Phase-0 MI oracle (kernel epistemic == independent state-side MI, err 2e-16), then (b) drives the `Agent` loop, recovers the realised epistemic gain at each chosen action, converts nats→bits, and divides by `n_steps·dt` (a single *declared, arbitrary* `dt`) to print a **perceptual-ceiling** rate: `I_acquired_ceiling = 2.5 bits/s`, `v_drift = 0.74 units/s`.

It then **refuses** to plot that against the bound and prints why (ceiling not behavioural rate; linear not MWC channel; no `β`/`χ`/`v_0` calibration; loss sited in `R` not `Q`). The disclaimer it carries, in intent:

> This is a synthetic-task **perceptual-ceiling** demonstration of the EFE epistemic information gain, `I_acquired_ceiling`, in nats and (caller-converted) bits/s. It is an **upper bound** on a signal→action rate, **not `İ_{s→a}`**. It is **not an E. coli comparison** (a linear channel cannot represent log/fold-change/MWC sensing, RFC-003 4.4) and **not a Mattingly bound test** (plotting a ceiling against the bound is trivially-below-the-curve and vacuous). The sensing-rate `dt` is an undefended placeholder, not a calibrated physical parameter. No efficiency scalar is implied.

That is the honest shape of where v0.3 stands: the bit is real and exactly counted; the curve test is four unbuilt capabilities (channel, behavioural projection, calibration, bound oracle) and one open research question (is the √ law derivable, or only fittable?) away.
