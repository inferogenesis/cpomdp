# R10: the open-loop EFE crossover — H\* = 7

*Status: positive result, measured selection-free and kernel-verified, through three review
passes. The first turned an earlier "no crossover" reading (a sweep capped one horizon
short) into the H\* = 7 finding. The second made the decisive measurement the exhaustive
argmin flip rather than the post-selection ΔG curve, corrected the mechanism to a decaying
pragmatic gradient against a flat epistemic pull, and split H\* = 7 (clipped reach) from
H\* = 6 (one-step reach). The third added the direct counterfactual that the flip is
epistemically driven (chapter 4.1), moved the conditioning onto the registered rollout-
hygiene bars (chapter 5), declared the feasibility bound (chapter 6), and fixed the
falsifier accounting (chapter 7).*

## TL;DR

R10 was scoped as the horizon H\* at which the best plan stops being a direct reach and
becomes a two-phase sense-then-commit walk. Measured on the model that produced Paper 1's
anchors, on the pre-registered `crossover-v1` action set `{−2,−1,0,1,2}`:

- **Decisive, selection-free.** The *exhaustive* argmin over every `A^H` sequence is
  prior-ward (a reach) for H = 1..6 and flips to the two-phase walk `[+1,−2,−2,0,0,0,0]`
  at **H = 7** (cue-ward at H = 7, 8, 9). Each horizon is a complete finite enumeration, so
  no policy is chosen with knowledge of the outcome. **H\* = 7.**
- **The flip is epistemic (counterfactual).** Re-run the enumeration with the epistemic
  term zeroed and the argmin is still prior-ward at H = 7, 8, 9; the pragmatic-only
  crossing is at H ≈ 10. So the flat ~1.7-nat epistemic pull is what brings the flip
  forward to H = 7. It is not a pragmatic phenomenon wearing an epistemic label.
- **Mechanism, as exposition.** Scoring the selected walk back against the coast reach:
  the epistemic pull is flat (`Δε: 1.72 → 1.64`), the pragmatic gradient decays
  (`Δc: 4.49 → 0.86`) and crosses below the pull at H = 7. The registered phrase "the
  accumulated epistemic pull overtakes the pragmatic gradient" is literally wrong: the
  pull overtakes nothing. The gradient decays below a constant pull.
- **Equal billing.** H\* = 7 is an upper bound: `crossover-v1` clips the reach at the grid
  edge `−2`, so it needs two steps to reach the goal. On the six-action set containing the
  one-step reach `−3`, **H\* = 6**. H\* is stable under genuine refinement (a step-0.5 grid over the
  same range gives the identical argmin at H = 6 and H = 7).

The headline margin is small (`ΔG(7) = −0.152` on a 425-nat scale, `3.6e−4` relative), so
chapter 5 shows the conditioning clears the registered hygiene bars and an independent
NumPy kernel reproduces the number.

## 1. The model (the crossover task)

The model is the two-node coupled-tree T-maze from `examples/ffg/epistemic_dissociation_figure.py`
(Paper 1's flagship). Joint state `[c, x, f] = [context, position, perceived-arm]`; node 0 =
CONTEXT (dim 1), node 1 = ARM_NODE (dim 2 = `[position, arm]`).

| piece | value |
|---|---|
| transitions | node 0: `A=[[1]]`, `Q=[[1e-2]]` (near-static context); node 1: `A=I₂`, `Q=diag(1e-4, 1e-2)` |
| coupling 0→1 | `W=[[0],[1]]` (context drives the perceived arm `f`), `Q_s=diag(1e3, 1e-2)`, `efe_relevant=True` |
| control | `B=[[0],[1],[0]]` — the 1-D action drives *position* (joint index 1), additively: `x_{k+1}=x_k+u_k` |
| sensor | `C=[[-1,1],[-1,1]]` on node 1 — two channels, both read displacement `o = f − x` |
| noise `R(x)` | `diag(r_goal, r_info(x))`, `r_goal = 200` (fixed), `r_info(x) = 0.02 + (200−0.02)·(1 − e^{−(x−x₀)²/(2·1.2²)})` — sharp (0.02) at the cue, dull (200) far |
| goal | observe `g = [0, 0]` displacement, precision `Λ = diag(0.6, 1e-4)`, `info_node = CONTEXT` |
| prior belief | `mean = [−3, 0, −3]`, `cov = diag(5, 0.05, 5)` |
| geometry | start `x = 0`; cue at `CUE_DETOUR_X = +1.0`; prior arm `−3`; true reward `+3` |

Two constants are load-bearing:

- **`R_GOAL = 200`** — the commit (goal) channel is deliberately dull and fixed. Its
  ambiguity is action-invariant, which sets the ~60/step floor every policy pays and stops
  the agent being trapped at the cue.
- **`INFO_PRECISION = 1e-4`** — the goal puts ~zero *direct* weight on the info channel.
  This does not decouple sensing from the pragmatic term. Because the sensor reads
  `o = f − x` on *both* channels and `f` is coupled to the context, sharpening the context
  through the info channel also sharpens the *commit* channel's predicted reading.
  `INFO_PRECISION` sets *where* H\* lands, not whether the crossover exists: with a direct
  info weight the payoff is immediate and H\* collapses to 1; with `INFO_PRECISION ≈ 0` it
  arrives only through the commit channel, so it takes seven steps to accumulate.

## 2. The objects being measured

### 2.1 The H-step rollout (`policy_efe_ffg`)

The rollout scores a policy `π = (u₀,…,u_{H−1})` by an open-loop, predict-then-contract
scan (`_ffg_rollout_body`). Per step, given predicted joint moments `(μ⁺, Σ⁺)`, sensor
`(C, R)`, goal `(g, Λ)`:

    o⁺ = C·μ⁺                          S = C·Σ⁺·Cᵀ + R(μ⁺)
    pragmatic  = ½·(o⁺ − g)ᵀ·Λ·(o⁺ − g)  +  ½·tr(Λ·S)        (a cost, lower better)
    epistemic  = ½·(ln det Σ⁺[CONTEXT] − ln det Σ_post[CONTEXT])   (a value, higher better)
    G          = pragmatic − epistemic                          (MINIMISED)
    Σ_post     = Σ⁺ − Σ⁺Cᵀ(CΣ⁺Cᵀ + R)⁻¹CΣ⁺                     # contract for the carry
    carry      = Belief(mean = μ⁺, cov = Σ_post)                # mean predict-only

The epistemic is node-restricted to the CONTEXT block. The pragmatic has a *mean* term
(goal distance) and an *ambiguity* term `½·tr(Λ·S)`. The mean never updates from sensing,
but the covariance contracts and carries forward, so a sharper belief lowers the expected
ambiguity of every future reading. That covariance channel is what pays off over the
horizon.

### 2.2 The exhaustive search (`EnumeratedEFESearch.over_backend`)

For a finite action set `A` and horizon H, this scores every `A^H` sequence and returns
the argmin. Because the quantified set is finite and fully enumerated, "the best plan at
horizon H is prior-ward" is *decided*, not sampled — a complete enumeration at each H.

### 2.3 The crossover statistic (`cpomdp.crossover`)

    Δε(H) = Σ_k [ε_k(walk) − ε_k(reach)]        the accumulated epistemic pull
    Δc(H) = Σ_k [c_k(walk) − c_k(reach)]        the accumulated pragmatic gradient
    ΔG(H) = Δc(H) − Δε(H) = G(walk) − G(reach)

For a *given* pair of policies this reads out the epistemic/pragmatic split. It does not,
on its own, decide the crossover — the pair has to come from somewhere, and here the walk
is one the exhaustive search selected. So the statistic is used for exposition (chapter
3.3), never as the decisive measurement.

## 3. What was measured

### 3.1 The H = 1 anchors (Paper 1, Result 4) — reproduced and asserted

| anchor | value (nats) |
|---|---|
| pull `Δε(1)` | **1.7232** |
| gradient `Δc(1)` | **4.4910** |
| `ΔG(1) = Δc − Δε` | **+2.7678** (reach wins at H = 1, so H\* > 1) |
| whole-state pull | 2.4166 (the alternative reading of the epistemic) |

Asserted in `tests/test_crossover.py` (`ANCHOR_TOL = 1e-4`).

### 3.2 The decisive measurement: the exhaustive argmin flips at H = 7 (selection-free)

`EnumeratedEFESearch.over_backend(backend, {−2,−1,0,1,2}, info_node=CONTEXT, horizon=H)`,
every `A^H` policy enumerated at each horizon. Feasibility is the enumeration cost
`|A|^H · H`, printed so it is not an undeclared budget (chapter 6):

    H    |A|^H   cost |A|^H·H   argmin                     positions x               plan
    1        5            5     [−2]                       [−2]                      reach
    5     3125        15625     [−2,−1, 0, 0, 0]           [−2,−3,−3,−3,−3]          reach
    6    15625        93750     [−2,−1, 0, 0, 0, 0]        [−2,−3,−3,−3,−3,−3]       reach
    7    78125       546875     [+1,−2,−2, 0, 0, 0, 0]     [+1,−1,−3,−3,−3,−3,−3]    walk ← H*
    8   390625      3125000     [+1,−2,−2, 0, 0, 0, 0, 0]  [+1,−1,−3,…]              walk
    9  1953125     17578125     [+1,−2,−2, 0×6]            [+1,−1,−3,…]              walk

No policy here was chosen with knowledge of the outcome: at each horizon the whole `A^H`
set is enumerated and the minimiser reported. The argmin is a direct reach through H = 6
and a two-phase walk (drive to the cue at `x = +1`, sense, reverse, commit at the goal
`x = −3`) from H = 7. It is cue-ward at H = 7, 8, **and 9** (1.95M policies enumerated), so
H\* = 7 is not a one-horizon blip. **This enumeration flip is the result.**

### 3.3 The mechanism split (exposition, and post-selection)

Take the walk the search selected at H = 7, `[+1,−2,−2,0,…]`, and score it back across the
horizon against the coast-to-stop reach `[−2,−1,0,…]`. This pair was *found by* 3.2, so the
crossing below is shown to explain the flip, not to establish it.

    H    Δε (pull)   Δc (gradient)      ΔG      winner
    1       1.7232        4.4910     +2.7678    reach
    2       1.7112        4.9477     +3.2365    reach
    3       1.6994        4.2252     +2.5258    reach
    4       1.6880        3.5203     +1.8323    reach
    5       1.6768        2.8321     +1.1553    reach
    6       1.6658        2.1599     +0.4941    reach
    7       1.6551        1.5031     −0.1520    walk   ← clean sign change at H*/H*−1
    8       1.6447        0.8611     −0.7836    walk

Read the two registered anchors down their columns. **The pull `Δε` is flat** (`1.72 →
1.64`, range 0.07): the epistemic value does not accumulate, exactly as the H = 1 anchor
says. **The gradient `Δc` decays** from 4.49 toward zero, and crosses below the flat pull
at H = 7. So the correct statement of the crossover is *the pragmatic gradient decays below
a constant epistemic pull*, which is the reverse of the registered wording. The result is
cleaner this way: one constant, one decaying curve, and the decay rate is the mechanism
(chapter 4). ΔG changes sign exactly once, at H\* (registered falsifier 2, satisfied).

**The search-found family contains the pre-registered pair at H = 1.** This is the answer
to "you changed the test object." At H = 1 the two-phase walk truncates to `[+1] = a_sense`
and the coast reach to `[−2] = a_myopic` — the exact pre-registered constant reach/walk
pair — and the H = 1 row above reproduces both anchors (`Δε(1) = 1.7232`, `Δc(1) = 4.4910`)
to `ANCHOR_TOL`. The varying-sequence family (`crossover-v2`) is not a disconnected object:
it *extends* the registered pair, agreeing with it at H = 1 and diverging only where a
constant policy cannot express the two-phase walk.

### 3.4 Equal billing: H\* = 7 is a clipped-reach upper bound; H\* = 6 with the optimum

`crossover-v1` clips the reach at the grid edge `−2`, so the direct reach needs two steps
(`0 → −2 → −3`) to reach the goal. Adding `−3`, which reaches the goal from the start in one
step, moves the flip one horizon sooner:

| action set | H\* | argmin at H\* |
|---|---|---|
| `crossover-v1` `{−2,−1,0,1,2}` (registered) | **7** | `[+1,−2,−2,0,0,0,0]` |
| `{−3,−2,−1,0,1,2}` (contains the one-step reach) | **6** | `[+1,−3,−1,0,0,0]` |
| `{−4,…,2}` (contains the one-step return) | **6** | `[+1,−4,0,0,0,0]` |
| wider than `{−4,…,2}` | not measured | see below |

So 7 is the pre-registered number and an upper bound; 6 is the value on the six-action set.
Both get stated, not one as a footnote to the other.

**`{−4,…,2}` has now been measured, and an earlier version of this table answered it by
deduction.** That row read "`H* = 6`, unchanged (`−3` already optimal)", which was reasoned
from `−3` reaching the goal in one step rather than measured. The deduction was not safe:
`−3` is the one-step reach *from the start*, while the walk arrives at the cue at `x = +1`,
from where the goal at `x = −3` is a displacement of `−4`. A set containing `−4` therefore
offers the walk a one-step return that `{−3,…,2}` does not.

**The number was right and the reason was wrong.** `H* = 6` on `{−4,…,2}`, registered at
`H* ≤ 6` before the run and measured under a completeness certificate
(`PROVED (set v1-ext, |A|^H = 7^6 = 117649, visited 117649)`). But the argmin at `H = 6` is
`[+1,−4,0,0,0,0]`, which uses `−4` and not `−3`, so the deduction reached the right value
along a route the measurement contradicts. A row can carry a correct number for a false
reason, and only the run tells them apart.

Extension **saturates** here: `H* = 6` on `{−3,…,2}` and on `{−4,…,2}` alike, so the
one-step return is taken without moving the horizon. Sets wider than `{−4,…,2}` remain
unmeasured. The registration, the argument and the result are in
`research/fep_falsification_battery.md` under `PRE-REGISTRATION 2026-08-20` and
`RESULT 2026-08-20`.

This is distinct from *refinement*. A genuine refinement subdivides the same range: on a
step-0.5 grid over `[−2, 2]` (`|A|^H·H = 9⁷·7 = 33.5M` at H = 7) the argmin is byte-identical
to the coarse set at both H = 6 (`Gmin = 364.6430`, prior-ward) and H = 7 (`Gmin = 425.1631`,
the same walk). The byte-identity is *expected* — the coarse set is a subset, so if the
argmin lies in it the scores must match; that half is a code-correctness check riding along.
The evidential content is the other half: **no intermediate action yields a lower `G`**, so
subdividing does not move the optimum toward the cue.

**Re-measured 2026-08-21, and the numbers hold.** These values had no in-repo
reproduction when they were first written, so the reading was withdrawn and the cell
registered as a re-measurement. It has now run under a completeness certificate:
`364.642964185792` at H = 6 and `425.163110098734` at H = 7, agreeing with the published
figures to every printed digit. The step-0.25 grid, dropped on cost at the time
(`17⁷·7 ≈ 1.6B` scored steps), has also run and returns the identical two numbers across
86 times the policies. `H* = 7` on both, so falsifier 4 is discharged and carries its
warrant. The 7 → 6 shift is a range *extension* supplying the omitted one-step reach, a
different operation from refinement.

## 4. Why — the mechanism (a decaying gradient, from accumulating ambiguity relief)

Per-step decomposition of the H = 8 walk against the reach (`mean` = goal-distance term,
`amb` = `½·tr(Λ·S)`, `eps` = epistemic):

    walk [+1,−2,−2,0,…]                    reach [−2,−1,0,…]
     k   x     mean    amb     eps          k   x     mean    amb     eps
     0  +1    4.801  60.767  1.736          0  −2    0.300  60.777  0.013
     1  −1    1.200  60.016  0.000          1  −3    0.000  60.759  0.012
     2  −3    0.000  60.020  0.000          2  −3    0.000  60.742  0.012
     3+ −3    0.000  60.02  (flat)          3+ −3    0.000  60.73→60.67 (slow drift)

Three facts:

**(1) The mean is open-loop, so the detour cost is one-time.** The walk pays a mean-term
penalty only while off the goal: 4.801 at the cue (`o⁺ = f − x = −3 − 1 = −4`), 1.200 on
the return, then zero once parked. Fixed cost, incurred once per plan.

**(2) The epistemic value is flat, not accumulating.** `ε` is ~1.736 at the sensing step
and ~0 afterward, matching the flat `Δε` column. Walking buys a fixed lump of epistemic
value. The crossover is *not* an epistemic-accumulation story.

**(3) The ambiguity relief is per-step and accumulates.** After sensing, the walk's
context/`f` covariance is contracted, so its commit-channel innovation `(C·Σ⁺·Cᵀ)[0,0]`
collapses from ~2.6 to ~0.03. The commit channel carries the goal weight `Λ_commit = 0.6`,
so the walk's ambiguity sits at ~60.02 while the reach's sits at ~60.7. **The walk pays
~0.67 nats/step less ambiguity at every parked step**, and that saving is enjoyed `H − 2`
times.

That per-step saving is exactly why the `Δc` column decays. The one-time detour cost is
~2.77 nats; the per-step relief is ~0.67; the fixed cost is overtaken when
`0.67·(H − 2) ≳ 2.77`, i.e. at H = 7.

**The corrected reading.** The earlier brief claimed the `INFO_PRECISION = 1e-4` gate kept
sensing out of the pragmatic term. It does not: the gate closes only the *direct*
info-channel route. The coupling opens an *indirect* route through the shared perceived-arm
`f`, and the commit channel's ambiguity rides it. The information has a pragmatic payoff —
ambiguity reduction on the commit channel — and that payoff, carried over the horizon, is
the crossover.

### 4.1 The flip is epistemic — a direct counterfactual

The mechanism above is pragmatic (ambiguity relief), which invites the fair question: in
what sense is this an *epistemic* crossover? Answer it by measurement. Re-run the exhaustive
enumeration scoring on the pragmatic term alone — the epistemic term zeroed — and compare
the argmin to the full-`G` one:

| horizon | argmin(`G`), with epistemic | argmin(pragmatic only), epistemic zeroed |
|---|---|---|
| 7 | cue-ward (walk) | **prior-ward (reach)** |
| 8 | cue-ward (walk) | prior-ward (reach) |
| 9 | cue-ward (walk) | prior-ward (reach) |

With the epistemic term the flip is at H = 7; without it the argmin is still a reach at
H = 7, 8, 9, and the pragmatic-only crossing does not arrive until H ≈ 10 (the pair's `Δc`
crosses zero between H = 9 at `+0.23` and H = 10 at `−0.38`). So the flat ~1.7-nat epistemic
pull is exactly what brings the flip forward from H ≈ 10 to H = 7. In the `ΔG = Δc − Δε`
picture: the pull lifts the crossing threshold from `Δc < 0` (H ≈ 10) to `Δc < 1.7`
(H = 7). The epistemic term is load-bearing at H = 7, 8, 9 — the crossover is epistemic, not
a pragmatic phenomenon wearing an epistemic label. Beyond H ≈ 10 the pragmatic relief alone
suffices and the epistemic is decorative.

## 5. Numerical hygiene (the margin is small, so this is not optional)

`ΔG(7) = −0.152` on a `|G| ≈ 425` scale is a `3.6e−4` relative difference, so the headline
has to be shown well-conditioned rather than a numerical accident. The quantities and bars
here are the shipped ones (`cpomdp.diagnostics.rollout_conditioning`, gated in
`tests/test_rollout_hygiene.py` against `MIN_EIG_FLOOR = 1e-9` on `min_eig(Σ_post)` and
`COND_CEILING = 1e8` on `cond(Σ⁺)`, `cond(S)`, `cond(Σ_post)`).

**Registered conditioning along the H = 7 walk:**

    k   cond(Σ⁺)   cond(S)   cond(Σ_post)   minEig(Σ_post)
    0    1003.0      79.6         42.3        4.21e−03
    1      49.8       1.3         49.8        3.71e−03
    2..6   ~55        1.0         ~55         3.66e−03

`all_positive_definite = True` for every `Σ⁺`, `S`, `Σ_post` over the horizon. The maximum
condition number is 1003 (`Σ⁺` at the sharp-sensing step, `R = diag(200, 0.02)`), five
orders under the `1e8` ceiling. `min_eig(Σ_post)` sits at `3.66e−3`, `3.7e6×` (about 6.6
orders) above the `1e-9` floor and slowly decreasing to a plateau well clear of it. So the
Cholesky guard in `_logdet_pd` never fires and the epistemic never goes NaN.

**The NaN guard, stated precisely.** `jnp.argmin` *selects* NaN, not rejects it, so the
guard is not the NaN value losing on its own. The info-gain path takes `ln det` by Cholesky
(positive diagonal ⟺ positive definite), mapping any non-PD input to NaN; and the
enumeration selects with a NaN-safe argmin, `jnp.argmin(jnp.where(jnp.isnan(g), jnp.inf, g))`
(`enumeration.py`), which sends a NaN score to `+∞` so it cannot win a minimisation.
Together those make a degenerate covariance lose. For this fixture the guard is inert
anyway: **all 78,125 enumerated `G` at H = 7 are finite** (asserted in the harness), so no
policy ever routes through it.

**Independent kernel.** The shipped `over_backend`, a hand enumeration, and `policy_efe_ffg`
all route through `_ffg_efe_step`, so their agreement tests plumbing, not the kernel. A
separate NumPy rollout recomputes the scoring kernel (pragmatic, ambiguity, node-restricted
info gain, contraction) independently, taking `ln det` by `slogdet` where the kernel uses
Cholesky — different routines. It reproduces `G(walk_7)` and `G(reach_7)` with `|Δ| = 0.0`,
inside `atol = 1e-9`. The exact zero is expected rather than suspicious: the two routines
differ only in the epistemic term (~1.7 nats), and that sub-ULP difference is masked when
it is summed into the ~425-nat total. The check isolates the *scoring* kernel by sharing the
belief moments (which have their own oracle tests); a fully independent belief propagation
is a larger reimplementation the couplings do not repay here. The 0.152-nat margin is real.

**Analytic cross-check.** The per-step ambiguity relief is capped by
`½·Λ_commit·(C·Σ⁺·Cᵀ)[0,0] ≈ ½·0.6·2.56 = 0.77` nats/step, against a one-time net detour
cost of ~2.77 nats. So `H\* ≥ 2 + 2.77/0.77 ≈ 5.6`, i.e. **H\* ≥ 6 independent of the
enumeration**, consistent with the measured 7 (clipped) and 6 (one-step reach).

## 6. How I first read this as a null, and why the correction is a departure, not a tuning

**What happened.** M7a scored the *constant* reach/walk pair and found no crossover — a
correct search-family artefact (a constant walk overshoots the cue and never returns). I
then ran the exhaustive search, the right instrument, but capped it at `CROSSOVER_MAX_H =
6`. Every argmin through H = 6 was prior-ward, and I read that as structural absence. A
four-agent adversarial pass endorsed the null.

**Why the null was premature, in the programme's own terms.** Registered falsifier 1 reads
"no crossover at any *feasible* H", and "feasible" was never given a number. The ledger's
scope note put the enumeration cap at H ≤ 3. So H = 6 was already *past* the registered
scope and was an undeclared compute budget, not a pre-committed stopping point. There was
nothing to violate by extending it. The mistake was reporting a null from an undeclared
budget as if it were a scoped result, and reading the ambiguity term per-step (`C·Σ⁺·Cᵀ ≈
2.6` is small next to `R_GOAL = 200`, so sensing "barely moves" the floor) where it needed
reading cumulatively (the ~0.67/step it does move accumulates past the detour cost).

I now define feasibility with a declared *maximum*, not just a costed point. The enumeration
cost is `|A|^H · H`; the declared bound is **H_max = 9** (`5⁹·9 = 17.6M` scored steps,
measured), and any larger H_max is a declared compute-budget increase recorded in the ledger,
not an open question. The crossover sits at H = 7, well inside the bound, and the argmin is
cue-ward at every H = 7, 8, 9. This is the fix for the undeclared-budget failure: the earlier
H = 6 was a compute convenience mistaken for a scope; H_max is now the scope.

**The methodological lesson, for the ledger.** The four-agent audit endorsed the null
while inheriting the same H ≤ 6 budget. An audit that shares the target's assumptions does
not test them. Diversity of method, not just of agent, is what an adversarial pass has to
buy.

**Why this is not a rule-2 violation.** Standing rule 2 forbids discovering a threshold by
tuning until the result fires. Nothing was tuned: the model, `INFO_PRECISION`, `R_GOAL`,
and the `crossover-v1` action set are all the unchanged Result-4 fixture. Only the sweep
horizon was extended past an undeclared budget, and the horizon is the free variable of the
statistic itself (`H\* = min{H : ΔG < 0}`), not a model knob. Extending it measures H\*; it
does not fit it.

## 7. The decision — register H\* = 7

R10 materialises as an open-loop crossover at H\* = 7 on the pre-registered action set,
decided by exhaustive enumeration and kernel-verified. A falsifier *fires* when its
condition obtains and would refute the crossover. None did, so all five survive (consistent
vocabulary: *not triggered* for the four this run tested, *not applicable* for the seed
one):

- **Falsifier 1** (no crossover at any feasible H): *not triggered* — a crossover exists at
  H = 7, well inside the declared H_max = 9. (In the *previous* pass, with the sweep capped
  at 6, this falsifier genuinely fired; extending to the declared bound removed it.) The
  constant pair still returns `None`, the honest search-family null, not the objective's.
- **Falsifier 2** (the flip is not a clean sign change at H\*/H\*−1): *not triggered* — `ΔG`
  changes sign exactly once (`+0.49 → −0.15`), and the argmin flips reach → walk at the same
  step.
- **Falsifier 3** (not reproducible across seeds): *not applicable* — void by construction,
  since the open-loop object carries no observation draw; the enumeration is deterministic
  and recomputes identically.
- **Falsifier 4** (H\* unstable under a declared refinement): *not triggered* — measured
  at two spacings under completeness certificates. `H* = 7` on step-0.5 (`9⁷` policies)
  and on step-0.25 (`17⁷ = 410,338,673`), against 7 on the coarse set, so `|ΔH*| = 0`
  inside the registered bar of 1. Both cells return `Gmin = 364.642964185792` and
  `425.163110098734`, identical to each other and to the figures chapter 4.3 published
  before any commit built the sets. No sub-step action reaches any argmin.
- **Falsifier 5** (H\* unstable under a declared extension): *not triggered* — `H* = 6` on
  `{−4,…,2}`, against a bar of `H* ≤ 6` registered a commit before the run, decided by
  exhaustive enumeration under a completeness certificate. The 7 → 6 shift is a range
  extension supplying the one-step reach, and extension saturates at 6.

The result to register:

- **H\* = 7** on `crossover-v1` (upper bound, clipped reach); **H\* = 6** on `{−3,…,2}` and
  on `{−4,…,2}`, both measured. The pre-registered number is 7.
- **Mechanism: a pragmatic gradient decaying below a constant epistemic pull, driven by
  accumulating commit-channel ambiguity relief (~0.67 nats/step); the flip is epistemically
  driven (chapter 4.1).** Not epistemic accumulation, and not "the pull overtaking the
  gradient".
- **Warrant.** The flip is a proved finite-enumeration result (Prover 3 · enumeration) at
  each horizon;
  the mechanism decomposition and the epistemic counterfactual are corroborated by the
  per-step trace and an independent NumPy kernel. The earlier "structural, all H" claim is
  retired.

The corresponding ledger and number-register edits: warrant-ledger §10 records H\* = 7 (and
the H\* = 6 refinement), the four falsifier outcomes above, the `crossover-v2` two-phase
family with its H = 1 containment of the registered pair, the H_max = 9 feasibility bound,
and the audit-inherited-budget lesson; §7 retires the "H ≤ 3" enumeration cap. The
number-register (`warrant_numbers.md`) gains a crossover section: `ΔG(7) = −0.152` with its
`3.6e−4` relative margin, the conditioning bars (`min_eig(Σ_post) = 3.66e−3` vs `1e-9`, max
`cond = 1003` vs `1e8`), and the pragmatic-only crossing at H ≈ 10.

The harness is `examples/ffg/crossover.py` (gated by `tests/test_example_checks.py`): the
selection-free flip, the mechanism split disclosed as post-selection, the epistemic
counterfactual, the NumPy oracle, the finiteness guard, and the registered conditioning, all
asserted. The scorer seam (`EnumeratedEFESearch.over_backend`, ADR-034), the crossover
statistic (`cpomdp.crossover`, ADR-033), and the constant-null motivation
(`crossover_sweep.py`) are the infrastructure it rides.

## Appendix — key code paths

- Rollout carry (predict-only mean, contracting cov): `src/cpomdp/efe.py::_ffg_rollout_body`.
- One-step EFE, node-restricted epistemic, and the Cholesky `ln det` guard:
  `src/cpomdp/efe.py::_ffg_efe_step`, `_state_info_gain`, `_logdet_pd`.
- Per-step decomposition (`predicted_mean`, `predicted_observation_cov` give the
  mean/ambiguity split):
  `src/cpomdp/efe.py::policy_efe_ffg_trace`.
- Exhaustive FFG search: `src/cpomdp/enumeration.py::EnumeratedEFESearch.over_backend`.
- Crossover statistic: `src/cpomdp/crossover.py`.
- The whole result, reproducible and gated: `examples/ffg/crossover.py` (`--check`).
