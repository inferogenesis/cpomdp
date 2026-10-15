# Warrant numbers

Every declared number in the cpomdp test suite — a margin, a ceiling, a floor, or a
tolerance — is recorded here with its reason: what it is, the magnitude it gates, how much
headroom it clears, and the first-principles basis for its value. The point is that no bar
is fitted to make a fixture pass. A reader should be able to read any number here and see
it was chosen from a stated fact, not tuned.

A growing tracker alongside `DECISIONS.md` (the architecture decisions) and `BUILD_PLAN.md`
(the roadmap), one section per workstream. It opens with the multi-step EFE thresholds and
the crossover statistic's H=1 anchors; more sections land as new declared numbers enter the
suite.

`research/warrant_ledger.md` fixes what the instrument may claim, naming the provers, the
labels, the evidence each one has to carry, and the claim shapes that are out of reach.
This file records the numbers those claims are measured against. Where a bar's *licence*
is in question, read the ledger. Where its *value* is, read on.

## How to read a warrant

Two vocabularies, kept apart on purpose.

**Prover class** (how well a claim is warranted, from Paper 1's taxonomy). All five modes,
the `Warrant` label each earns, and the evidence `CheckReport` requires before it accepts
`PROVED` are tabulated at the head of `research/warrant_ledger.md`. The two that gate most
of the numbers below:

- **Prover 3 · sample** corroborates. A sample of a continuum, or a local optimum. It can
  refute a universal by counterexample but never decide one.
- **Prover 3 · enumeration** decides. An exhaustive enumeration over a finite set. "No
  member does X" is a proof, not a sample.

Two further ways a number is pinned here, neither of which is a prover class:

- **byte-identity** (RFC-001). Two code paths that run the identical IEEE-754 operations
  in the identical order return bit-for-bit equal results. Asserted with
  `assert_array_equal`, not `allclose`.
- **numerical agreement**. Two paths that compute the same value by different rounding (a
  different library, or a different XLA transform) agree to a stated tolerance.

**Certified vs computed.** A number is *certified* only with a stated tolerance beside it.
Without one the honest word is *computed*.

## Terms

- **eps**: machine epsilon for float64, 2.22e-16. The float64 mantissa is 53 bits, about
  15.95 decimal digits.
- **ULP**: unit in the last place, the gap between adjacent float64 values at a given
  scale. A one-ULP disagreement is the smallest a float64 result can move.
- **PD**: positive definite.
- **cond**: the 2-norm condition number. At `cond ~ 1/sqrt(eps) = 6.7e7` a solve has lost
  about half its significant digits. At `cond ~ 1e8` the relative error `cond*eps` is
  roughly 2e-8, so about eight of the sixteen digits are gone.
- **nat**: a unit of information or free energy in natural-log units.

## Multi-step EFE (horizon > 1)

The numbers gating the M1-M6 checks: the rollout trace, the Σ(π) witness, the numerical
hygiene, and the enumerated search.

### Margins

Two one-sided witness thresholds. Each asks whether a quantity moves by a structurally
large amount, not whether it matches a tight target.

**`SEPARATION_MARGIN = 1e-2`** (`tests/test_sigma_policy_dependence.py`). Gates three
lower-bound separations. Under a fixed sensor two distinct policies must carry the same
covariance trajectory while their means differ (measured mean separation 1.4). Under
`R(x)` the same two policies must separate `Σ_post` (measured 0.261). Under `Q(x)` they
must separate `Σ⁺` (measured 0.336). The tightest gated case clears the margin by 26x, the
others by 34x and 140x. On the log scale the real separations sit about 15 orders above
eps while the margin sits about 13.6 orders above eps, so a full 1.4 decades of headroom
separate the margin from the smallest real signal. The only inverted matrix on these paths
is the scalar innovation `S` at condition number 1, so nothing here is numerically fragile.
The fixed-sensor side is not gated by this margin. There the covariance agreement is
asserted exactly (see the byte-identity locks); the margin only gates the mean separation
that proves the two policies genuinely differ.

**`VARYING_WIN_MARGIN = 1.0`** (`tests/test_enumeration.py`). Gates the claim that a
genuinely varying policy beats every constant one. On the beacon fixture the best varying
sequence scores about 27 nats below the best constant sequence (`G` 7.16 versus 34.12). I
compare best-varying against best-constant over the whole enumerated set, so the result
does not depend on `argmin`'s tie-break or the order the actions are listed in. A one-nat
bar is cleared 27x and sits about 15 orders above eps. One nat is a large, unambiguous
free-energy gap, so a win this size is a real objective difference rather than a rounding
artifact. See the corrections section for why the earlier version of this test was wrong.

### Numerical-hygiene bars

Two-sided tripwires on the conditioning of the rollout covariances, read from the trace on
the host in `tests/test_rollout_hygiene.py`.

**`COND_CEILING = 1e8`**. Every per-step `cond(Σ⁺)`, `cond(S)`, and `cond(Σ_post)` must
stay below this. The value is the float64 half-precision knee. At `cond ~ 1e8` a solve
keeps only about half its digits, so a covariance this ill-conditioned is a real
degradation flag. The fixtures peak near `cond 6`, roughly seven orders of magnitude below
the ceiling. The gap is deliberate. The bar flags catastrophic conditioning, it is not a
snug fit to the fixtures. One caveat to carry forward. With `p = 1` and the 1-D `Q(x)`
model, `cond(S)` and all `Q(x)` condition numbers are exactly 1, so `cond(Σ_post)` on the
fixed and `R(x)` fixtures is the only column doing any work today.

**`MIN_EIG_FLOOR = 1e-9`**. The smallest eigenvalue of every `Σ_post` must stay above this.
The healthy fixtures keep it near 0.1, clearing the floor by about eight orders of
magnitude. A planted near-singular step at 1e-15 (the `bad_post` negative control) falls
six orders below the floor, which is what makes the bite-test meaningful. The floor itself
sits about seven orders above eps, so an eigenvalue this small is heading toward singular
while it still carries real signal. A reviewer can note that 1e-9 is a round number rather
than a specific power of eps. They cannot call it tuned when the fixtures pass by a factor
of 1e8.

### Byte-identity locks (exact, tolerance 0)

These use `assert_array_equal`, so the tolerance is exactly zero. Each is an architectural
fact, not a small number. A tolerance would understate it and could mask the very coupling
the check forbids.

| Lock | Where | Measured |
| --- | --- | --- |
| Fixed-sensor `Σ⁺`/`Σ_post`/`S` are policy-independent | `test_sigma_policy_dependence.py` | 0.0 across the two declared H=3 policies; means separate by 1.4 |
| Trace column sums equal `policy_efe`'s scalars | `test_policy_efe_trace.py` | 0 ULP across all 9 model/H combinations (27 summed-column assertions) |
| H=1 trace moments equal `_efe_step`'s fields | `test_policy_efe_trace.py` | 0 ULP on all three branches |
| Enumerated policy set equals the itertools product | `test_enumeration.py` | exact; pure integer index-gather |

The fixed-sensor lock is the clearest. There the covariance recursion never reads the
mean, so two distinct policies feed the same `lax.scan` body the identical operands, and
deterministic float64 arithmetic returns the identical bytes. The action moves only the
mean. That is why the exact assertion is correct and a tolerance would be wrong. One
portability caveat. Byte-identity across two separately compiled XLA graphs is a
CPU-backend fact. A GPU or TPU fused-multiply-add change could cost a ULP. If that happens
the companion NumPy-oracle test at `atol=1e-9` carries the cross-implementation agreement
instead.

### Numerical-agreement tolerances

Where two paths compute the same value by different rounding, I use a generic float64
allowance rather than a fitted number:

- **`atol=1e-9`** for cross-implementation agreement (a JAX path against a plain-NumPy
  loop).
- **`atol=1e-12`** for cross-transform agreement (jit against eager, vmap against a
  per-item loop, a JAX solve against a NumPy solve).

On every fixture the measured disagreement is either exactly 0.0 or a single ULP (about
2.2e-16 for order-one values), because the inputs stay well-conditioned (`cond(S) = 1`,
`cond(Σ⁺) <= 3.3`). The tolerances therefore sit three to seven orders of magnitude above
the real floor. They are loose safety bars against a gross regression such as an
accidental float32 fallback, which would diverge by order 1e-7 and miss them by orders. One
honest wrinkle. `numpy.testing.assert_allclose` keeps its default `rtol=1e-7`. For
order-one values the relative term governs, so the stated `atol` only binds near zero. The
bars are still safe, since a real formula error moves the result by order one and fails
hard.

### Exact counts and structural bounds

Integer identities, so eps never enters. Both operands are Python ints, which stay exact
past the float64 `2^53` ceiling where a float count would drift.

| Number | Value | Gates |
| --- | --- | --- |
| `certificate.expected == visited` | `\|A\|^H` | completeness (ADR-030), warrant `PROVED` |
| `n_policies == \|A\|^H` | e.g. 9, 27 | the enumerated count |
| `cost_per_cycle == \|A\|^H * H` | e.g. 81 | the honest exponential cost (RFC-001) |
| `FiniteActionSet` size `>= 2` | 2 | a set of one is no choice to search |
| `horizon >= 1` | 1 | a policy needs at least one step |

### Negative-control fixtures

Not tolerances. Each is an input chosen to make a guard fire, and each is asserted to
actually fire.

- **`diag(-1, -2)`** for the log-determinant guard. Its determinant is +2, so a
  determinant-sign shortcut returns a finite `log|det|` and passes. The test asserts the
  shortcut is fooled and that `_logdet_pd` (Cholesky) returns NaN. This is what proves the
  guard keeps the sign rather than trusting the determinant.
- **`R = -0.5`** for the oracle epistemic guard. It keeps the innovation `S = 0.5` still
  PD, so the required NaN can only come from the `R` guard, not from `S`.
- **`bad_post = 1e-15`** for the eigenvalue floor. At about 4.5 eps the covariance is
  near-singular yet still strictly PD, so the eigenvalue floor is the thing that bites
  rather than the PD flag.

### Corrections from the adversarial pass

An adversarial review recomputed every number above and attacked each as tuned. Two tests
failed that review and were rewritten. I record them because the fix changed what the test
actually decides.

1. **The varying-wins test was an `argmin` tie-break.** The earlier fixture used a
   velocity-control, position-sensor model at horizon 2. There the last action never
   reaches an observation, so every sequence sharing the first action scores a
   bit-identical `G`. Asserting only "the argmin is non-constant" passed because `argmin`
   returns the first tied index, which the action listing happened to make varying.
   Reordering the same action set flipped the winner to a constant. The rewrite uses a 1-D
   direct-control model and asserts a strict `G` margin between best-varying and
   best-constant, which is independent of the tie-break and the ordering.
2. **The indefinite fixture did not discriminate.** The earlier fixture was `diag(-1, 1)`,
   labelled "det > 0 but not PD". Its determinant is actually -1, so a determinant-sign
   guard returns NaN on it exactly as Cholesky does. It could not tell a sign shortcut from
   the real guard. The rewrite uses `diag(-1, -2)`, whose determinant is +2, and asserts
   the sign shortcut passes while Cholesky does not.

### Reporting rule for downstream cells (F1-F3)

The pytest asserts here evaluate a bare comparison, since a passing assertion prints
nothing. Any report that consumes these numbers is different. A cell must print the moving
term, the pinned term beside it, their ratio, and the condition numbers of any inverted
matrices. It must never print a bare pass, and never the small number on its own. A term
that is naturally near zero clears a small-number bar trivially, so the ratio and the
conditioning are what make the cell evidence. This binds the H-sweep harness when it lands.

## The crossover statistic and its H=1 anchors

The crossover statistic (`cpomdp.crossover`) contrasts a walk policy against a reach policy
over an EFE horizon; `tests/test_crossover.py` pins it. This section records the anchors it
must reduce to, the reach/walk declaration, and the one tolerance the reduction is checked
at. ADR-033 records the aggregation *decision*; this records its *numbers*.

### The anchors

All four are read on the two-node coupled-tree T-maze
(`examples/ffg/epistemic_dissociation_figure.py`, Result 4) with the cue off the prior path
(`CUE_DETOUR_X = +1.0`), on Agent B, the R(x) agent. The boundary scan carries no
observation draw, so these are exact facts rather than sampled ones — recomputing returns
them bit-for-bit, which the suite asserts.

| Anchor | Value (nats) | What it is |
| --- | --- | --- |
| pull `Δε(1)` | 1.7232 | epistemic value of the sense action over the myopic one, node-restricted to the CONTEXT marginal |
| gradient `Δc(1)` | 4.4910 | pragmatic cost of the same contrast |
| crossover `ΔG(1)` | 2.7678 | `gradient − pull`, positive, so the reach wins at H=1 — which is what forces `H* > 1` |
| whole-state pull | 2.4166 | the same epistemic contrast aimed at the whole state, not the CONTEXT node |

The two epistemic numbers are the point of the node-restricted versus whole-state
distinction. 1.72 is the information about the CONTEXT latent that an agent choosing between
arms is actually served by; 2.42 is the whole-state observation-space reading of the same
move. I record both so a reviewer never mistakes one for the other. The pragmatic 4.49 is
identical either way, since it does not depend on the epistemic target.

### The statistic and its sign

**These are post-selection, and the qualifier travels with them.** The walk and reach the
`Δ` numbers contrast at `H > 1` are the pair the exhaustive search *found*, so scoring
them back across the horizon exhibits the mechanism and cannot establish the flip. The
flip is established by the enumeration, which is selection-free. A reader quoting `Δε(H)`
or `Δc(H)` as evidence *for* the crossover has the direction backwards. At `H = 1` the
pair collapses to the registered constant anchors below, which were declared before the
search and are not post-selection.

`Δε(H) = Σ_k [ε_k(walk) − ε_k(reach)]`, `Δc(H) = Σ_k [c_k(walk) − c_k(reach)]`, and
`ΔG(H) = Δc − Δε`. The pragmatic term is a cost (lower better) and the epistemic a value
(higher better), so `ΔG < 0` is the crossover. `ΔG` is defined as `Δc − Δε`, asserted at
tolerance 0, so the sign flip is exactly the planner's argmin flip. At H=1 the pair
collapses to the pull and gradient above.

### The reach/walk declaration

Constant-action policies over declared members of a versioned action set
(`crossover-v1 = {−2, −1, 0, 1, 2}`, a superset of the two anchor actions):

- `a_sense = +1.0` — `argmax ε` over the grid, cue-ward. Its resolved μ⁺ lands on the cue
  (`+0.9999` against `CUE_DETOUR_X = +1.0`). WALK holds it.
- `a_myopic = −2.0` — `argmin G` over the grid, prior-ward. The prior points left, so the
  myopic optimum runs to the grid edge. REACH holds it.

Both are members of the declared set, and `argmax ε` / `argmin G` over the coarse set still
land on them (asserted). So the pair is a property of the model, not of the grid resolution
or of whichever two an action sweep happens to surface.

### The one tolerance

**`ANCHOR_TOL = 1e-4`** (`tests/test_crossover.py`). The H=1 reduction asserts the statistic
matches each anchor to this. It is a numerical-agreement bar, not a fitted margin. The
statistic at H=1 runs through `policy_efe_ffg`, which is byte-identical to the single-step
`_ffg_efe_step` the anchors were measured with (ADR-032), so the real disagreement sits at
the ULP. `1e-4` sits about twelve orders above that floor and four decimals below the
anchors' own precision, so it is a loose guard against a gross regression — an accidental
float32 fallback, or a changed model constant — not a snug fit to the fixtures.

### The multi-step crossover H\* (v0.4.4)

The H=1 anchors force `H* > 1`; the exhaustive varying-sequence search finds where the argmin
actually flips. `examples/ffg/crossover.py` pins these; the horizon is the free variable of
the statistic, not a tuned parameter.

**Every number in this section is open-loop.** The `H*` rows come from
`EnumeratedEfeSearch.over_backend(...).evaluate(...)`, which scores whole length-H action
sequences and never re-plans between steps. `ΔG(7)` comes from `_numpy_score` and the
pragmatic-only crossing from `epistemic_counterfactual`, neither of which goes through
that call, and both of which are open-loop by the same construction. Nothing here drives
`RecedingHorizonSelector` or `OpenLoopSelector`.

The same statistic under a receding-horizon driver is a different quantity, and **it has
not been measured**. The closed-loop leg is open (`research/warrant_ledger.md`), and
ADR-034 records that M7b drives the enumeration open-loop deliberately. The reason to name
the seam is that a sentence like "the exhaustive argmin flips at 7" reads the same under
either mode, so a row without the seam lets a reader take it for the mode they had in
mind. `research/r10_open_loop_crossover.md` is the write-up.

| number | value | what it is |
| --- | --- | --- |
| `H*` (registered set) | 7 | first horizon whose exhaustive argmin over `crossover-v1^H` is cue-ward — a two-phase walk `[+1,−2,−2,0,0,0,0]`. Cue-ward at H = 7, 8, 9 |
| `H*` (with the one-step reach) | 6 | on `{−3,…,2}`, which contains `−3`, the action reaching the goal in one step from the start. So the registered 7 is an upper bound (the grid clips the reach at `−2`). `−3` is not established as optimal: the walk arrives at the cue at `x = +1`, from where the goal is a displacement of `−4`. The seven-action `{−4,…,2}` is measured below and also gives 6; wider sets are not measured |
| `ΔG(7)` | −0.1520 | `G(walk) − G(reach)` at H=7; the flip margin. Relative size 3.6e−4 against `G ≈ 425`, so the margin is small and must be shown well-conditioned |
| pragmatic-only crossing | H ≈ 10 | with the epistemic term zeroed, the argmin is prior-ward through H = 9 and crosses near 10 — so the ~1.7-nat epistemic pull is what advances the flip to 7 |
| `H_max` | 9 | declared feasibility bound; enumeration cost `5⁹·9 = 17,578,125` scored steps, measured. Larger H_max is a declared budget increase |

**The `H = 7` in `examples/crossover_horizon_figure.py` is a different number.** That plane
contrasts two named plans on a different model with a different backend, whole-state
epistemic, and no search at all. Its crossing is an exact statement about those two plans;
this `H*` is an argmin over every policy in a declared set. The integers coincide and mean
nothing by it. Disclaimed once here, beside the number, and once in the README.

#### `H*` stability under action-set change: registered, extension measured

Declared 2026-08-20, before any cell was run in this repository. The full argument is the
`PRE-REGISTRATION 2026-08-20` entry in `research/fep_falsification_battery.md`. The
numbers are here because this is where they are quoted from. The extension row is a
result and says so; the refinement rows are not, with the qualification below.

| axis | cell | registered prediction | outcome |
| --- | --- | --- | --- |
| extension | `{−4,…,2}`, 7 actions, spacing 1 | `H* ≤ 6`. `−4` shortens the cue-ward return from two steps to one and buys the prior-ward reach nothing, since `−3` already covers it in one | **PASS**, measured `H* = 6`, see below |
| refinement | step `0.5`, 9 actions over `[−2,2]` | stability, `\|ΔH*\| ≤ 1`. No direction is arguable: the largest magnitude does not move, so neither branch's step count does | **PASS**, `H* = 7`, `\|ΔH*\| = 0`, see below |
| refinement | step `0.25`, 17 actions over `[−2,2]` | stability, `\|ΔH*\| ≤ 1`, same argument | **PASS**, `H* = 7`, `\|ΔH*\| = 0`, byte-identical to step `0.5` |

Budget, declared in both units because they disagree. Time at the measured 39.0k
policies/s. `VOID (budget)` on overrun, which means unmeasured and never "stable".

| cell | actions | H | policies | scored steps | front-loaded peak, ×1.6 | time |
| --- | --- | --- | --- | --- | --- | --- |
| extension `{−4,…,2}` | 7 | 1–6 | 137,256 | 800,667 | 0.06 GiB | 16s |
| refinement `0.5` | 9 | 7 | 4,782,969 | 33,480,783 | 2.45 GiB | 2.0m |
| refinement `0.5` | 9 | 8 | 43,046,721 | 344,373,768 | 22.58 GiB | 18.4m |
| refinement `0.25` | 17 | 7 | 410,338,673 | 2,872,370,711 | 210.34 GiB | 2.92h |

The extension row budgets the sweep that was actually run. `H*` is the *first* cue-ward
horizon, so the sweep stops at the registered bar of 6 and never enumerates `7^7`. An
earlier version of this table budgeted `7^7 = 823,543` at `H = 7`, a cell no run visits.

Two lines to read carefully. The step-`0.5` cell at `H = 7` costs 33.5M scored steps
against the `H_max = 9` budget of 17.6M, so it is inside one declared unit and double the
other. And the last two rows exceed the 19 GiB free on the reference machine, so both run
under `ChunkedEfeSearch`, whose peak is block-determined and flat in `|A|^H` (ADR-036). The
memory column is `cue_maze.enumeration_cost` times its measured 1.6x correction, and it
describes the front-loaded path only. A front-loaded attempt at either is `VOID (budget)`:
the WSL cap is configured rather than physical, so it takes the session down instead of
raising `MemoryError`.

`cue_maze.best_reachable_noise` returns `R_LO = 0.02` exactly on all four sets, so every
lattice lands on the cue and no cell is void by geometry.

#### Measured 2026-08-21: refinement step-`0.5` → `H* = 7`, **PASS**

| number | value | what it is |
| --- | --- | --- |
| `Gmin` at H=6 | 364.642964185792 | step-`0.5` over `[−2,2]`, argmin `[−2,−1,0,0,0,0]`, prior-ward. Certified `PROVED (set v1-refine-0.5, 9^6 = 531441 visited)` |
| `Gmin` at H=7 | 425.163110098734 | argmin `[+1,−2,−2,0,0,0,0]`, cue-ward. Certified at `9^7 = 4782969` |
| `H*` on step-`0.5` | 7 | unchanged from the coarse set, so `\|ΔH*\| = 0` against a registered bar of 1 |
| chunked rate | 41,982 policies/s | measured, and the basis for the step-`0.25` projection |
| chunked peak | 0.46 GiB | flat in `\|A\|^H` (ADR-036), against 22.6 GiB front-loaded at `9^8` |

Both scores agree with the published `364.6430` and `425.1631` inside the `5e-5` tolerance
registered on 2026-08-21, so nothing is retracted. No half-step action appears in either
argmin. The `9^8` cell was contingent on `H*` rising and is therefore not required.

**step-`0.25` gives the identical numbers.** Measured at `c37fac3`: `364.642964185792` at
`H = 6` over 24,137,569 policies and `425.163110098734` at `H = 7` over 410,338,673, both
certified `PROVED`, 2.5 hours at 46,124 policies/s, peak 0.47 GiB. Not a digit moves
against step-`0.5`, across twice the actions and 86 times the policies, and no
quarter-step action reaches either argmin. `H* = 7` on both cells, so the refinement axis
passes at both spacings.

**The step-`0.5` cell already has published numbers, and no run behind them.**
`research/r10_open_loop_crossover.md` reports the argmin byte-identical to the coarse set
at `H = 6` (`Gmin = 364.6430`) and `H = 7` (`Gmin = 425.1631`), and reads falsifier 4 as
not triggered. When that was written no commit in this repository built a nine-action step-`0.5` set,
so the numbers had no in-repo reproduction and the check suite reported the falsifier
`NOT_RUN_HERE`. Both are now false: `crossover.py` builds the set, the cell was
re-measured under a completeness certificate against `364.6430` and `425.1631`, and the
falsifier carries `PROVED`. The re-measurement is recorded above.

#### Measured 2026-08-20: extension `{−4,…,2}` → `H* = 6`, **PASS**

| number | value | what it is |
| --- | --- | --- |
| `H*` on `{−4,…,2}` | 6 | first cue-ward exhaustive argmin on the seven-action set, `[+1,−4,0,0,0,0]`. Certified `PROVED (set v1-ext, \|A\|^H = 7^6 = 117649, visited 117649)` |
| `Gmin` at H=6 | 363.9394 | on `{−4,…,2}`. At H=5 it is 303.6592 and still prior-ward, `[−3,0,0,0,0]` |

Registered at `H* ≤ 6` before the run, so this passes. The mechanism the argument rested on
is the one that fires: the winning policy steps to the cue at `+1` and then takes a single
`−4` to the goal at `−3`, which is the one-step return `−4` was predicted to buy.

Two qualifications belong beside the number. The parenthetical "plausibly 5" did not hold.
`H = 5` stays prior-ward even though the walk `[+1,−4,0,0,0]` is feasible there, so
feasibility of the shorter return is not what sets the crossover. And extension
**saturates**: `H* = 6` on `{−3,…,2}` and on `{−4,…,2}` alike, so `−4` is used without
moving the horizon. Both facts are compatible with the registered prediction and neither was
predicted by it.

The conditioning of the H=7 walk clears the numerical-hygiene bars (recorded above): every
`Σ⁺`, `S`, `Σ_post` positive definite; `min_eig(Σ_post) = 3.66e−3` against `MIN_EIG_FLOOR =
1e-9` (about 6.6 orders of margin); max `cond = 1003` (`Σ⁺` at the sharp-sensing step)
against `COND_CEILING = 1e8`. An independent NumPy kernel (slogdet `ln det`, where the shipped
kernel uses Cholesky) reproduces `G(walk_7)` and `G(reach_7)` within `atol = 1e-9`.

### The oracle audit anchors (v0.5.0)

That NumPy kernel kept only `slogdet`'s log-magnitude. It therefore accepted a covariance
block with an even number of negative eigenvalues, where the shipped kernel's Cholesky
guard returns NaN. `tests/test_crossover_oracle_audit.py` recorded what the unguarded path
returned at H=7 before the guard, now `diagnostics._host_logdet_pd`, was wired in, so
that the guard could be shown not to move it.

The guard moved nothing. All three anchors below compare equal to the unguarded values
under `==`, not merely within `ORACLE_RTOL`. A positive-definiteness precondition ahead of
an unchanged `slogdet` is not a change of arithmetic. It also never fires on this rollout,
where every block is positive definite by a wide margin, `min_eig(Σ_post) = 3.66e−3`
against a `1e-9` floor. `H*` and `ΔG(7)` keep their quoted values. The oracle route is
audited now rather than assumed.

| number | value | what it is |
| --- | --- | --- |
| `ANCHOR_WALK` | 425.163110098734 | oracle `G(walk_7)`, measured on the unguarded path |
| `ANCHOR_REACH` | 425.3151092512748 | oracle `G(reach_7)`, same |
| `ANCHOR_MARGIN` | −0.15199915254078178 | their difference: `ΔG(7)` above, at full precision |

**`ORACLE_RTOL = 1e-11`** (`tests/test_crossover_oracle_audit.py`). The guard is a
precondition ahead of an unchanged `slogdet`, so on a positive-definite rollout the
movement it should cause is zero and a ULP bar would hold. `1e-11` is `4.3e-9` absolute at
`G ≈ 425`, the scale of the `atol = 1e-9` the demo already trusts for
shipped-against-oracle agreement. It sits there rather than at the ULP so that BLAS
variation across platforms cannot flake it. Any change to the epistemic term large enough
to bear on the flip moves `ΔG` by `1e-3` or more, six orders above the bar.

**`MARGIN_ATOL = 1e-8`**. The margin is a difference of two numbers near 425, so it
inherits about `8.5e-9` of absolute slack from the pair above. Rounded up from that.

An anchor is not a warrant. It records that a number did not move. Whether the number was
right is what the negative-eigenvalue rejection decides.

### The flip separation bar (v0.5.0)

`H* = 7` rested on `G(walk) < G(reach)`, a bare inequality between two computed floats.
Nothing said how far apart they had to be. The registered claim was decided by exhaustive
enumeration and the delivered margin was measured against nothing, which is the mismatch
this closes.

No new bar was invented for it. `COND_CEILING = 1e8` is already declared above and already
gated on every `Σ⁺`, `S` and `Σ_post` by `tests/test_rollout_hygiene.py`. A float64 solve at
condition number `k` carries relative error near `k · eps`, so that ceiling states an error
on each score and, doubled, on their difference:

```text
flip_margin_error(G₁, G₂) = 2 · max(|G₁|, |G₂|) · COND_CEILING · eps
```

At `G ≈ 425` that is `1.89e−5` nats. The measured `|ΔG(7)| = 0.1520` clears it by `8.0e3`,
about 3.9 orders. A margin inside the bound is reported `NOT RESOLVED`, the honest label
for an ordering that is genuinely undetermined. It is not an assertion failure, because a
tie is a finding about the measurement and an exception would erase it.

The bound is loose in the safe direction on purpose. It propagates the *declared* ceiling
rather than the measured conditioning, and the H\* walk measures `max cond = 1003`, five
orders under the ceiling. The true error is nearer `1.9e−10`. Deriving the bar from what
was declared keeps it a bar rather than a description of this fixture.

**Derived after the measurement, and why that is not an accommodation.** The bound was
written once `ΔG(7)` was already known, which is the shape scoping rule 2 warns about. The
contamination it guards against cannot occur here. A stricter guard can only fail a result
that was passing; it has no route to rescue one that was failing. Tightening this bound
moves rows toward `NOT RESOLVED`, never away from it, so choosing it with the answer in
hand could not have manufactured the separation.

**What the derivation is, stated honestly.** `cond · eps` is a forward-error rule of thumb
for a single linear solve. This applies it to a seven-step accumulation of predicts,
contractions and log-determinants, so it is an argument by analogy, not a proved bound.
That makes it a `BOUNDED` *stated error bar*, not the certified bracket validated numerics
supply, and the demo labels it `BOUNDED` rather than reaching for `CERTIFIED`. The five orders of headroom between
the declared ceiling and the measured conditioning is what makes the analogy safe enough to
report, and the gap is stated here rather than left for a reviewer to find.

**Effect on the reported tiers.** Falsifiers 1 and 2 in that demo read `PROVED` on the
prover axis, from the completeness certificate of the `|A|^H = 78125` enumeration, and
`BOUNDED` on the tier axis, from this bound. The two axes answer different questions and neither
substitutes for the other: a claim can be decided with nothing stated behind its margin,
or measured against a tight bar and only sampled. Falsifiers 3 and 4 stay `COMPUTED` and carry
no warrant at all, since neither produced evidence here.

**What this bound does not cover.** It certifies the float arithmetic, which was never the
fragile part of `H* = 7`. The live exposure is falsifier 4: `H*` is an upper bound, because
the declared set clips the reach at `−2` while `−3` reaches the goal in one step, and a set
containing `−3` flips at 6. A `BOUNDED` row reading `8.0e3x` clear invites a reader to take
the number as firmer than it is, so the qualifier travels in both detail strings and not
only here.
