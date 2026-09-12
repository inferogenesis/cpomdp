# cpomdp build plan / progress tracker

Running checklist of what's built and what's next. Authoritative decisions live
in `DECISIONS.md` (ADRs); this file is the roadmap.

Conventions: `[x]` done, `[ ]` open, `[~]` partial. **⛔** marks a PR that cannot merge
until GATE-D4 passes.

Window: v0.4.4 → v0.5. Every remaining change for Papers 2 and 3, as merge-sized pull
requests. Eleven PRs across two tags. Module and symbol names below are proposals. Each
PR's ADR fixes them.

---

## Where this stands — v0.4.4, released 2026-08-04

**Shipped ahead of schedule.** Toolbox **A** (multi-step EFE) was scheduled at v0.5.
Toolbox **G-B** (exhaustive enumerator with a completeness certificate) was Paper 3's,
scheduled at v0.6. Both landed at v0.4.4: `cpomdp.enumeration`, `FiniteActionSet`,
`EnumeratedEfeSearch`, `CompletenessCertificate`, `IncompleteEnumerationError` (ADR-030,
ADR-031). R10 is measured. `H* = 7`, exhaustive over the declared `{0, ±1, ±2}`, H = 1
anchors pinned at `Δε = 1.7232`, `Δc = 4.4910`, `ΔG = +2.7678` nats to `1e-4` (ADR-033).

**Scheduled and not shipped.** v0.4.3 shipped `cpomdp.diagnostics`,
`tests/test_theorem.py` and the FFG positive-definiteness fixes. It did not ship toolbox
**B** (the scoring harness) or **G-A** / **G-D** (the Paper 3 hedge), all three of which
the original Phase 0 assumed it would. Those three, plus **C**, **C′**, **D**, **E**,
**F1–F6** and **G-C**, are the whole of what follows.

**Blast radius, and it shrank.** D3 was always gate-independent in content. With A and
R10 banked at v0.4.4, a gate failure now costs Part 2's numbers and nothing else. R1–R5,
R10, C5 and all of Paper 3 Part 1 are off the reference filter.

---

## Two tags

**v0.4.5 — GATE-D4.** Cut at PR-8's merge, whatever the gate's outcome. This is the
citation and rollback point. If the bound holds, it is where "certified" starts being
true. If it fails, it is the witness the re-scoped `EXACT` papers cite, and without it that
witness is "main at some commit".

**v0.5 — terminal.** Everything Papers 2 and 3 need from cpomdp. Both pin v0.5. v0.6
stops existing as a planned tag. Paper 4 adds no code, because standing rule 4 of its
scoping is zero new numbers, so it appears here as a consumer of v0.5 only.

Two naming decisions, closed:

- [x] **`GATE-D4`**, never "the v0.4.4 gate" and never "the v0.4.5 gate". The bound is
      named independently of the tag that carries it. Cite it as `GATE-D4 (PR-8, v0.4.5)`.
      `research/warrant_ledger.md` is amended.
- [x] **The `P2-n` namespace stands as an alias layer** for the external programme
      paperwork, which is not tracked here. In-repo, only the toolbox letters have a live
      consumer: `research/fep_falsification_battery.md` cites toolbox A–F in its build
      lines. The bare letter `B` is retired, having had three live referents: build
      item B, Workstream B1–B5, battery family B1–B5. PRs are the working unit.

---

## Warrant primer — read before writing any check

Warrant is a property of the check, not of the number.

| Prover | What it does | Label |
| --- | --- | --- |
| **1** pen-and-paper theorem | Universals within stated hypotheses | `PROVED`, numerics as *witness* |
| **2** symbolic computation | Closed-form identities, algebraic non-existence | `PROVED` |
| **3 · enumeration** exhaustive enumeration over a finite domain | *Decides* the universal, since ¬∃ ≡ ∀¬ | `PROVED`, only with a cardinality certificate |
| **3 · validated** validated numerics | Proves universals over a compact domain by construction | `CERTIFIED` |
| **3 · sample** sampling a continuum | Exhibits existence, refutes a universal by counterexample. Never proves a universal, at any sample count | `CORROBORATED` |

`research/warrant_ledger.md` carries the canonical table, with the evidence each warrant
requires. This is the working copy; where they differ, the ledger is right.

Outcome is orthogonal. A registered falsifier emits `warrant ∈ {PROVED, CERTIFIED,
CORROBORATED}` and one of five outcomes. `PASS` is not among them: a falsifier fires or it
does not, and printing `PASS` beside a prover column that disambiguates it inverts the rule
the prover column exists to satisfy (ADR-035).

| Outcome | What it means |
| --- | --- |
| `NOT TRIGGERED` | It ran, the condition did not obtain, the claim survives it |
| `FIRED` | The condition obtained. The claim is refuted, and that is the result |
| `NOT RESOLVED` | It ran and the ordering is genuinely undetermined: the intervals overlap |
| `NOT APPLICABLE` | Void by construction. Evidence for nothing, and not a survivor |
| `NOT RUN HERE` | Measured elsewhere, or not yet. The detail says where |

The last three are not interchangeable. Collapsing them loses the survivor accounting and
burns the word a real tie needs. The last two never ran, so they carry **no warrant** —
`CORROBORATED` asserts sampling-grade evidence was obtained, and they obtained none. That
cell reads `—`, enforced by `CheckReport`.

Ordinary two-valued assertions are outside this. "Does the shipped number match the NumPy
oracle" passes or raises, and needs no vocabulary.

Tiers cut across both. `EXACT` is a closed-form reference at machine precision. `BOUNDED`
is a stated bar or a certified bracket. `COMPUTED` has no statable bar, and the word for
such a number is *computed*, never *certified*. A bar can be derived from one already
declared elsewhere in the suite rather than invented for the claim; `warrant_numbers.md`
records the derivation and whether it is a proved bound or a stated error bar.

**The error that recurs.** An action *sweep* over a continuous range is a finite grid
over an infinite domain, so it samples. A *policy enumeration* over a declared finite set
enumerates.
v0.4.4 already encodes this: `EFESelector` prints `CORROBORATED`, `EnumeratedEfeSearch`
prints `PROVED`. PR-1 extended the vocabulary to checks, added the missing third level,
and shipped it as `warrantlib`, re-exported as `cpomdp.warrant` (ADR-035, ADR-039).
Import it; do not restate it.

---

## PR map

| PR | Title | Serves | Blocked by | Alias | Size | Tag |
| --- | --- | --- | --- | --- | --- | --- |
| **PR-1** | Warrant plumbing and the `slogdet` oracle audit | 2, 3 | — | P2-3, F4, P2-9 | M | v0.4.5 |
| **PR-1b** | Chunked enumerator (`O(chunk)` residency) | 2, 3 | — | — | S | v0.4.5 |
| **PR-2** | R10 hardening | 2, 3 | PR-1, PR-1b | P2-9 | M | v0.4.5 |
| **PR-3** | World/Agent seam, exogenous action, constructors | 2, 3 | PR-1, PR-1b | P2-1a (B) | L | v0.4.5 |
| **PR-4** | Paper 2 scoring: evaluator, cells, error bars | 2, 3 | PR-3 | P2-1b (B), F1–F3, F5 | L | v0.4.5 |
| **PR-5** | Control bracket and Paper 2 Part 1 results | 2 | PR-4 | P2-4 (E) | L | v0.4.5 |
| **PR-6** | Paper 3 toolbox and Part 1 results | 3 | PR-1, PR-3 | G-A, G-C, G-D | L | v0.4.5 |
| **PR-7a** | Printed-scheme failure probes and the declared budget | 2, 3 | — | — | S | v0.4.5 |
| **PR-7** | Exact reference filter and the rule ladder | 2, 3 | PR-3, PR-7a | P2-5 (C) | L | v0.4.5 |
| **PR-7b** | The ladder's ordering, and the R6 gap | 2, 3 | PR-7 | P2-7 (D) | M | v0.4.5 |
| **PR-8** | **Certified discretisation bound · GATE-D4 · tag v0.4.5** | 2, 3 | PR-7b | P2-6 (C′) | L | v0.4.5 |
| **PR-9** ⛔ | Window harness and Paper 2 Part 2 results | 2 | GATE-D4, PR-4 | P2-8 (F6) | L | v0.5 |
| **PR-10** ⛔ | Paper 3 Part 2 results | 3 | GATE-D4, PR-2, PR-6 | — | M | v0.5 |
| **PR-11** | v0.5 release | 2, 3 | all | — | M | v0.5 |

Size key: S under a day, M two to four days, L a week or more. Grouping the original
nineteen items into eleven PRs pushed six of them to L. That is the trade.

**Critical path.** `PR-1 → PR-3 → PR-7 → PR-7b → PR-8 → PR-9 → PR-11`. It runs through the
gate, so nothing shortens it except starting PR-7 early. PR-7a blocks PR-7 without being
on the path: it has no blockers of its own and is a day's work, so it can land any time
before PR-7, which cannot ship a declared budget until it has.

**Parallel tracks.** PR-1 and PR-2 touch nothing else and can go first or alongside
anything. PR-1b is the exception: it edits `enumeration.py`, which PR-3 also opens, so it
goes ahead of both rather than beside them. PR-3 is the fan-out point. After it lands,
three tracks run independently: the Paper 2 scoring track (PR-4 → PR-5), the
reference-filter track (PR-7a → PR-7 → PR-7b → PR-8), and the Paper 3 track (PR-6). Only PR-9 and PR-10
wait on the gate.

**ADR numbers are not pre-allocated.** A PR takes the next free number when it lands, and
the heading carries its decision date. Reserving numbers against unwritten work collides
the moment one PR needs two, and it sorts the file against the commit record, which is
what an audit reads. Landed work keeps the number it took: PR-1 is ADR-035, PR-1b ADR-036,
issue #65's symbolic track ADR-037.

**Work outside the ladder.** Issue #65's continuation runs beside this numbering rather
than in it, since a `PR-N` label would collide with the plan's own. Three branches, in
order, all serving PR-8's registration:

| branch | the question it settles |
| --- | --- |
| `65-warrant-symbolic-evidence` | does the programme accept a Prover 2 evidence type, and what must it carry |
| `65-gap-series-c2-gate` | the symbolic kernel, the warrant wiring, and `c₂` |
| `65-gap-series-c4` | σ⁴ and `c₄` |

Landed on the first: `SymbolicReduction`, carrying the claim, where the setup was hand
derived against the problem, and the assumptions the identity is contingent on.
`Evidence` becomes a union of it and `CompletenessCertificate`, so `PROVED` is reachable
by argument as well as by enumeration. Every item in the evidence tuple is checked for
its kind, so a path naming where the proof lives cannot satisfy the precondition by being
present. Blank means blank to a reader. A field that is not text, one holding only
whitespace, one holding only zero-width characters, and one carrying a line break all
fail to construct, with `assumptions` checked entry by entry. That is what keeps the
correspondence from being a formality.

`Tier` lost its letters on the same branch (ADR-038). `EXACT`, `BOUNDED` and `COMPUTED`
replace `A`, `B` and `C`, and the prover sub-modes read `Prover 3 · sample`,
`Prover 3 · enumeration` and `Prover 3 · validated` in prose. Aliases were rejected. Two
live vocabularies would stop a document's vocabulary from dating it. The canonical table
moved to the head of `research/warrant_ledger.md` and every other document points at it. The README gained a section on what the labels mean, and was
brought up to v0.4 while it was open.

That branch is wider than it was scoped to be, and the widening is deliberate. Landing
the vocabulary docs beside the type they describe keeps them from trailing three PRs
behind it. The review question is unchanged: does the programme accept a Prover 2
evidence type, and what must it carry. Read that first. The rename is mechanical and sits
in its own commits, so it skims.

Landed on the second, as three modules. `series_kernel.py` owns the construction of `W`,
its expansion and the moment operator. `log_ratio_series.py` keeps its structural pins and
now reports them as `CheckReport`. `gap_series.py` derives `c₂` and stops.

- [x] The expansion is built by explicit polynomial truncation rather than by
      `sympy.series`. Every primitive is a geometric, binomial or exponential sum, and
      truncation happens inside each product. The swap is licensed by a check, not by
      inspection: the truncation path equals `series(W)` term for term through `σ⁴`,
      which is where `DERIVATIVE_ORDER` stops the truncation path rather than where
      `series` stops being affordable. `series` costs about an order of magnitude more
      per order (2 s at `σ⁴`, 8.5 s at `σ⁵`, 83.5 s at `σ⁶`), which is the real reason
      the pipeline does not rest on it.
- [x] `c₂ = ℓ₁²/4` symbolically, against the registration's `(R'(μ)/2R(μ))²` derived
      before this series existed. Two independently computed closed forms agreeing is the
      `EXACT` licence, not a tolerance being met.
- [x] The gap is derived twice and the two are checked against each other: once from
      `log E_q[e^W] − E_q[W]` directly, once as `κ₂/2` from the cumulant recursion.
      Reverse and forward KL agree at `σ²`, and the agreement is **not** asserted beyond
      it, since `κ₃` separates them and that separation is why the pinned conventions
      matter at `σ⁴`.
- [x] 53 identities across the three modules, every one `PROVED` at `EXACT` carrying a
      `SymbolicReduction`. Mutation probes confirm the suite discriminates: an off-by-one
      truncation fires four kernel checks, a wrong fourth moment fires its own, and a
      wrong `c₂` claim fires rather than being absorbed.
- [x] Deleting the Kalman shift from `h` fires **five checks**: T4's `σ²` coefficient
      and its Kalman-shift report, and K4 at `σ²`, `σ³` and `σ⁴`. T4 is the only one
      that names the shift. K4 reports only that two paths disagree. `gap_series` stays
      blind to it, because at `σ²` the gap reads only the first-order term of `W`. That
      was measured rather than assumed, and it is what the quartic work inherits.

Landed on the third, in the same three modules. `series_kernel.py` gained the exponential
series and the exact predictive average. `gap_series.py` runs to `σ⁴` and derives `c₄`.
**ADR-037** carries the three decisions: what a Prover 2 claim is backed by, how the
expansion and the predictive are computed, and what happens to a result produced out of
order.

- [x] `c₄ = 7ℓ₁⁴/16 − ℓ₁²ℓ₂/4 + ℓ₂²/8 + ℓ₁ℓ₃/4 − 3ℓ₁²/(4R̄)` under **reverse KL**, on
      free `ℓ₁..ℓ₄` and a symbolic `R̄`. `c₂` is direction-free and `c₄` is not, since
      `κ₃` separates the directions above `σ²`. Every registered prediction holds. The five conjectured fractions are
      the derived ones, the two coefficients reported consistent with zero are identically
      zero, the seven-term basis spans it with no remainder, and the exponential family
      keeps the two terms section 2 of the registration said it could. At the declared
      operating point it is `−3/16`, 1.2% off the `COMPUTED` fit it supersedes.
- [x] The innovation is averaged under the exact predictive `ν = σz₁ + √R̄·e^{δ/2}·z₂`.
      Collapsing it to `N(0, R̄)` leaves `c₂` untouched and gives a `c₄` 5.7 times the
      exact one, so the nesting decides the coefficient rather than refining it. C9
      measures the difference symbolically. `predictive_truncation` had already recorded
      why: `p*` is a scale mixture with exponential tails.
- [x] 70 identities now, up from 53: 23 in `series_kernel`, 18 in `log_ratio_series`,
      29 in `gap_series`. All `PROVED` at `EXACT` on a `SymbolicReduction`. Each is
      declared in `research/registered_checks.toml` and reconciled by id, so a dropped
      one fails by name. The CI job pinned all three counts until ADR-046.
- [x] The derivation ran before anything registered it, and that is disclosed at the head
      of the registration's section 7 rather than repaired. Rerunning after registering
      would be a rerun by someone who knows the answer.
- [x] The three families not yet run were registered before their runs, with the bar and
      the VOID escape declared. `exp(x)` carries `c₄` of the opposite sign to the quadratic
      family, so a formula fitted to the quadratic could not produce it.
- [x] Those runs: `exp(x)` at σ^5.984, `sin` at σ^6.004, `tanh` **fired** at σ^6.302
      against a 0.25 bar. `sin` did not need the escape it was granted. The fire stands as
      fired. It sits on the over-cancellation side of 6 and the registration set
      refutation at 5 or below, so the closed form is neither refuted nor passed by that
      cell.
- [x] The `tanh` fire is candidate-precision sensitive, disclosed rather than re-run. The
      6.302 belongs to a `--c4` of `0.0061107`, the five-figure value the registration
      tabled. At full precision the same six cells read σ^6.148, inside the bar. The cell
      keeps its recorded outcome, because a result is not un-fired by re-running it with
      better inputs after seeing it fire. A candidate-precision precondition is registered
      for the next cell instead.
- [x] Three guards on the instrument, none of which revises a recorded outcome: `--c4`
      takes one family, G4c voids off the declared grid, and the control branch that lived
      in prose is now a report. All four recorded runs reproduce unchanged.
- [x] G4c, leave-one-out exponent stability in `gap_expansion.py`, written against a
      pre-registration that declares its three readings for the `tanh` fire. It cannot
      convert the fired cell into a pass. All four family runs are recorded in the
      registration's RESULT 2026-08-16 on the diagnostic.

Results from the third land in `research/gate_d4_registration.md`, not here.

**Every PR.** `uv run --no-sync pytest -m "not rxinfer and not slow"` green,
`uv run --no-sync ruff check src/cpomdp tests examples mkdocs_hooks.py` clean,
`uv run --no-sync ty check` clean,
`uv run --no-sync python -m warrantlib.manifest --check --layout-only
research/registered_checks.toml` current, `mkdocs build --strict` green whenever a
docstring or doc page moves. A new name in `cpomdp.__all__` needs a `docs/api/` page before anything
can link to it.

---

## PR-1 — Warrant plumbing and the `slogdet` oracle audit

`serves: 2, 3` · `blocked by: —` · `alias: P2-3, F4, part of P2-9` · `size: M` ·
`tag: v0.4.5` · `ADR-035`

Do this first. Everything downstream self-labels off it, and a registered number depends
on the audit half.

- [x] Promote `SearchWarrant` to a shared `Warrant` enum (`cpomdp/warrant.py`) and add
      **`CERTIFIED`** for Prover 3 · validated. Keep `SearchWarrant` as an alias so
      `EFESelector.warrant` and `EnumeratedEfeSearch.warrant` keep their labels
      unchanged. Without `CERTIFIED`, every reference-filter number either borrows
      `PROVED`, which overclaims, or reads `CORROBORATED`, which throws away the bound
      GATE-D4 exists to buy.
- [x] Carry `(warrant, outcome, tier)` on **checks**, not only on selectors.
      `CheckReport` in the same module. Landed as a frozen dataclass, not a NamedTuple:
      `_replace` routes through `_make` and would bypass a custom `__new__`, so the
      precondition two items below would have held at construction and leaked on the
      first edit.
- [x] Generalise ADR-029's three-valued outcome from `crossover.py --check` to the
      suite. Do not rebuild it. Kept its three and added two: `NOT RUN HERE` promoted
      from a prose rule to a member, `NOT RESOLVED` for a genuine tie. The primer's
      `{PASS, FAIL, NOT_RESOLVED}` was tried and rejected (ADR-035).
- [x] `PROVED` under enumeration requires a `CompletenessCertificate`, enforced as a
      **constructor precondition**. `PROVED` without a certificate becomes
      unrepresentable rather than merely wrong. Widened to *evidence*, since `PROVED`
      also covers Provers 1 and 2, which have no enumeration to certify.
- [x] Suite summary prints counts per `(warrant × outcome)`, under an `n registered,
      m tested here, k fired` header. A run that survives everything and decides nothing
      is visibly that.
- [x] **The audit.** v0.4.3 fixed the kernel by routing `_efe_step` and
      `_state_info_gain` through one `_logdet_pd` helper, testing positive definiteness
      by Cholesky rather than by determinant sign. The changelog does not say the NumPy
      oracle path was fixed. `examples/ffg/crossover.py` runs an independent NumPy kernel
      on the headline number at **H = 7**, past the H = 3 regime where the
      range-dependent noise branch makes this bite. Confirm it, do not assume it.
      **Outcome: it was not fixed.** Both epistemic terms now route through
      `diagnostics.logdet_pd`, and the H = 7 anchors are bit-identical across the change,
      so the guard corrected an exposure rather than a number.
- [x] Add a test that a matrix with an even number of negative eigenvalues
      (`diag(-1, -2)` is the canonical one) is rejected on **both** paths. Two paths that
      both discard the sign agree wrongly, which is the worst outcome available and is
      invisible to a two-route agreement check.
- [x] Until this merges, `H* = 7` is quoted at the warrant of an unaudited oracle.

Two items the audit added, neither registered above:

- [x] The flip rested on a bare inequality between two computed floats. It is now
      measured against the error `COND_CEILING` allows on their difference, a bar derived
      from one already declared rather than invented. A margin inside it reports
      `NOT RESOLVED`, not an `AssertionError`.
- [x] `H* = 7` is an upper bound, because the declared set clips the reach at `−2` while
      `−3` reaches the goal in one step. That qualifier now travels in both `BOUNDED` rows,
      not only in the write-up.

**Merge gate:** the negative-eigenvalue rejection test passes on kernel and oracle. The
suite summary renders. No existing check loses its label. **ADR-035.** — met.

## PR-1b — the chunked enumerator

`serves: 2, 3` · `blocked by: —` · `alias: —` · `size: S` · `tag: v0.4.5` · `ADR-036`

Ahead of PR-2, and dated, because PR-2's registration pre-commits against adopting a
chunked enumerator in response to a `VOID` outcome. Taking it before any cell is declared
is a different act, and the record has to be able to show it.

- [x] `ChunkedEfeSearch` beside `EnumeratedEfeSearch`. Blocks decode their own indices as
      base-`|A|` numerals in `itertools.product` order and reduce to running scalars, so
      neither the policy set nor the score vector is ever resident. `evaluate` keeps its
      contract, including the full `G` vector G-D reads.
- [x] The combine defines the tie-break rather than inheriting it: blocks in increasing
      order, strict `<`, so the globally lowest index wins exactly as `jnp.argmin` does.
      A 32-way-tie fixture records that both paths agree there.
- [x] `CompletenessCertificate` carries `action_set_size`, `horizon` and
      `action_set_version`, and `PROVED` requires **both** `expected == |A|^H` (domain)
      and `visited == expected` (coverage). `visited` is loop-carried now, so those come
      apart where padding bugs live. The set naming also fixes a defect older than
      chunking: `expected` alone conflates base with exponent, 81 being `9^2` and `3^4`.
- [x] Cross-path bit-identity at `9^7`, the largest cell both paths run: identical argmin
      index, identical policy, `G` equal under `==` at `425.163110098734`, which is
      `ANCHOR_WALK`. Recorded in `research/warrant_ledger.md`, not only in the suite.
- [x] Measured: peak 5.401 GiB → 0.431 GiB (12.5×), flat across `9^4`–`9^7` because what
      remains is the fixed XLA baseline. Throughput 13.1k → 39.0k policies/s.

**Merge gate:** the chunked path reproduces the front-loaded argmin bit for bit at the
largest shared cell, completeness holds at an `N` indivisible by the block, and residency
tracks the block rather than the enumeration. **ADR-036.** — met.

## PR-2 — R10 hardening

`serves: 2, 3` · `blocked by: PR-1` · `alias: P2-9` · `size: M` · `tag: v0.4.5` ·
`ADR: on landing`

Gate-independent. Paper 3's G9 inherits the qualifier this produces.

- [x] **Declare the action mode on the registered result.** `RecedingHorizonSelector` and
      `OpenLoopSelector` genuinely differ, and v0.4.4 requires a measurement to say which
      it used. The M7b sweep drove `EnumeratedEfeSearch` directly, which is open-loop.
      Confirm that in code, then carry the declaration into the paper. The ledger requires
      R10 reported under its seam, never silently read as closed-loop.
      Confirmed in code: `examples/ffg/crossover.py` calls
      `EnumeratedEfeSearch.over_backend(...).evaluate(...)` and instantiates neither
      selector. The declaration now travels with the number, in both `PROVED` falsifier
      details (rows 1 and 2; row 5 states its own registered bar instead) and at the head
      of `warrant_numbers.md`'s crossover section, which had not
      named the seam at all. The write-up and ADR-034 already carried it. No number moved.
- [x] **Run the fourth D3 falsifier: `H*` stability under action-set refinement.**
      Registered in the battery before it is run, which is what keeps it a test. It is
      load-bearing rather than precautionary, because the release itself calls `H* = 7` an
      upper bound *because the grid clips the reach*.
  - [x] Two axes, registered separately. **Extension** is a wider magnitude range at the
        same spacing. **Refinement** is finer spacing over the same range.
  - [x] Predicted direction and its argument written down **before running**, on each
        axis. Where no direction can be argued in advance, register a stability test at a
        stated tolerance (`|ΔH*| ≤ 1`) rather than dressing a stability check as a
        directional prediction.
  - [x] Pre-declare the compute budget, in **both units**, because they disagree: the
        cell counts here are policies (`5^7 = 78,125`, `7^7 = 823,543`,
        `9^7 = 4,782,969`, `9^8 ≈ 4.3 × 10^7`) while the ledger's `H_max = 9` is 17.6M
        *scored steps*. `9^7` at H = 7 is 33.5M scored steps: inside one budget, double
        the other. Budget exceeded is **VOID**, meaning unmeasured, never "stable".
  - [x] Decide `9^8` and `17^7` deliberately rather than by contingency. PR-1b removed
        the memory wall, so they are 18 minutes and 2.9 hours at the measured 39.0k
        policies/s, not the `VOID (memory)` they were. Accept or decline each in the
        registration, before the run.
  - [x] The extension axis has one named unmeasured cell to answer, `{−4,…,2}`.
        `research/r10_open_loop_crossover.md` carried a row for it reading "`H* = 6`,
        unchanged (`−3` already optimal)", deduced from `−3` reaching the goal in one step
        rather than measured. No commit in this repo builds that set, so the row is
        retracted and the cell reads *not measured*. The deduction is also not safe: the
        walk arrives at the cue at `x = +1`, from where the goal at `x = −3` is a
        displacement of `−4`, so a set containing `−4` offers a one-step return the
        six-action set does not. Measure it under a completeness certificate.
        Measured 2026-08-20: `H* = 6`, `PROVED (set v1-ext, 7^6 = 117649 visited)`, argmin
        `[+1,−4,0,0,0,0]`. The deduced row's number was right and its reason was wrong, since
        the argmin uses `−4` rather than `−3`. Extension saturates at 6.
  - [x] Size the run against `free -g` before launching it. `cue_maze.enumeration_cost`
        describes the **front-loaded** path only. On the chunked path peak is
        block-determined and flat in `|A|^H`, so re-derive any budget line taken from the
        old figure. The WSL memory cap is configured rather than physical, and an
        over-sized enumeration takes the whole session down.
  - [x] Wire `cue_maze.best_reachable_noise` in as the void guard. A refined set that
        cannot land on the cue produces a null indistinguishable from "information is
        never worth the detour", which is pure geometry and not a result.
  - [x] Carry the `slow` marker. This runs on merge-to-main, not on pull requests.
        **Discharged differently, and the difference is the point.** Nothing carries the
        marker, because no *refinement* cell re-enumerates on any gate. The measured rows are
        recorded
        constants with their certificates and a provenance, and `--refinement` is the
        reproduction route, off both the test path and `--check`. Cheap tests assert the
        recorded values against the published ones. Slow-marking the sweep
        would have put two minutes on every merge to re-derive a number that cannot
        drift, since a recorded constant changes only when someone edits it.
- [x] Carry the post-selection disclosure into the paper. The mechanism split in
      `crossover.py` is disclosed as post-selection, because the scored pair was found by
      the search. Now stated in `warrant_numbers.md` beside the `Δ` numbers themselves,
      which is where a paper author quotes them from, alongside the existing disclosures
      in the write-up, the ledger and the demo.
- [x] Keep the `H = 7` coincidence disclaimed once, near the number.
      `examples/crossover_horizon_figure.py` crosses at the same integer on a different
      model, a different backend, whole-state epistemic, no search. Disclaimed beside the
      `H*` table in `warrant_numbers.md`, which had it nowhere, and once in the README.

**Merge gate:** both axes report an outcome, `PASS`, `FAIL` or `VOID`, against their
registered prediction. **ADR on landing.**

**Met 2026-08-21.** Extension `PASS` at `H* = 6` on `{−4,…,2}`; refinement `PASS` at
`H* = 7` on step-`0.5` and step-`0.25` alike, `|ΔH*| = 0` against a bar of 1. ADR-043
records the extension axis and step-`0.5`, ADR-044 step-`0.25`, and the `9^8` cell turned
out moot rather than deferred.

## PR-3 — World/Agent seam, exogenous action, constructors

`serves: 2, 3` · `blocked by: PR-1` · `alias: P2-1a (B)` · `size: L` · `tag: v0.4.5` ·
`ADR-047` · issue #66

The foundation, and the fan-out point. Paper 3 reuses the seam, the exogenous action mode
and the constructors without modification, which is why the module boundary below is a
test rather than a convention.

- [x] `World` owns p\*. `ScoredAgent` owns p. **No code path lets the agent read the
      world's parameters**, enforced at the type level in the spirit of
      `IncompatibleLinearizationError`. A test asserts the absence of the path, not
      merely that it is unused.
      Landed in `cpomdp.harness`. `World` has no accessor returning its model or a
      parameter of it, and `ScoredAgent` has no constructor slot for a `World`. The
      absence is tested by an object-graph walk that follows attributes, containers,
      bound methods and closure cells. Seven tests probe the walk itself first, because
      a negative assertion passes just as well on a broken detector.
      `ScoredAgent` composes a backend rather than subclassing `Agent`: inheriting would
      publish `sample_action`, and an agent free to act breaks the comparison the driven
      sequence exists to support. ADR-047 carries the reasoning.
- [x] `ExogenousActionSequence`: one common control sequence `u_{1:k}` driven into every
      agent under comparison. This is what makes `H(p*)` a shared constant that cancels.
      It also severs the control loop. Record that on the result object as a declared,
      contestable modelling choice, not in a comment.
      `drive` advances the world once per step and folds one reading into every agent, so
      the arms stay comparable. `DrivenRun.control_loop` carries a `ModellingChoice` with
      no default: one field states what was chosen, a second states where it is
      contestable.
- [x] Constructors, declared as a versioned set in the same discipline as
      `FiniteActionSet`. Model axis
      `{correct, perturb_parameters(axis, magnitude) × declared magnitudes}`. Inference
      axis `{exact, FrozenGain, WrongFixedR, DiagonalCovarianceOnly}`.
      Both axes are frozen records rather than callables, so a declared set can be
      diffed. `ModelSpec.build` copies, so two models built from one spec share no array.
      Each set refuses a duplicate name and refuses to be built without the cell the
      others are measured against. The declared magnitudes themselves are a research
      declaration and are not fixed here.
- [x] **Keep the module boundary the nineteen-PR plan bought with a PR split.** Nothing
      in this seam may import PR-4's three-term evaluator, because Paper 3 explicitly
      does not import it. Assert it with an import test. A test outlives a PR boundary.
      `tests/test_module_boundary.py` walks the transitive first-party import closure and
      refuses `cpomdp.scoring`, named before it exists, along with `cpomdp.selection` and
      `cpomdp.enumeration`. Probed at one hop and at two.

**Merge gate:** the no-read-path test passes. The import test holds. The constructor set
round-trips through the model spec and shows up in a diff when extended. **ADR-047.**

**Met 2026-08-22.** All three conditions hold. Nothing is exported at the top level, so
no `docs/api/` page is owed yet.

**Registered as open, and it blocks nothing here.** `World` refuses a state-dependent
`R(x)` or `Q(x)` rather than choosing an evaluation point, since the departed state and
the arrived-at one give different trajectories and the filter's own convention is not
available to a process that knows its state exactly. It is settled in its own change
before PR-4, which is fixed-`R` throughout. Writing it now is what keeps it from being
fitted to a gap figure that does not yet exist.

## PR-4 — Paper 2 scoring: evaluator, separation cells, error bars

`serves: 2, and F5 serves 3` · `blocked by: PR-3` · `alias: P2-1b (B), F1, F2, F3, F5` ·
`size: L` · `tag: v0.4.5` · `ADR: ADR-059`

- [x] `ThreeTermEvaluator` returns `Decomposition(misspecification, inference_gap)`, two
      divergences directly computed. The type carries no entropy field and no entropy
      estimator, which makes the standing prohibition on entropy subtraction structurally
      unrepresentable rather than discouraged.
- [x] `AdditivityCheck` is a **separate object** taking an entropy estimator explicitly.
      It is the only place `H(p*)` is estimated, which is why its residual carries an
      entropy bar and the divergences do not.
- [x] **Reuse the completeness certificate on the constructor cross.** It is
      `warrantlib.ProductCompletenessCertificate`, the two-axis form of the certificate
      `cpomdp.enumeration` reports, so the cross names both axes and both versions.
      Expected cardinality `|model axis| × |inference axis|`, visited, equality
      asserted, `IncompleteEnumerationError` on mismatch. This is the cheapest provability
      purchase in the plan (Route 2), so do it first inside the PR.
- [x] Report the both-positive diagonal cell explicitly. Off-diagonal separations are only
      meaningful against a cell where both terms move.
- [x] **F2.** Condition numbers `cond(Σ)` and `cond(S)` printed in every separation cell.
      An ill-conditioned matrix manufactures or destroys twelve orders. Extend
      `diagnostics.rollout_conditioning` rather than writing a second one.
- [x] **F3.** Separation ratio: the pinned term, the moving term's magnitude beside it,
      and their ratio. "Below 1e-12" is empty if the term is naturally O(1e-13). The claim
      is the ratio, roughly twelve orders, not the small number.
- [x] Absence of catastrophic cancellation checked wherever a term is computed as a
      difference of large quantities.
- [x] **F1.** The additivity residual is `E[F]_measured − (H(p*) + D₁ + D₂)` and the bound
      is `δ_F + δ_H + δ₁ + δ₂`, four terms. Omitting `δ_F` under-bounds the residual and
      lets a real closure failure read as within tolerance.
- [x] **F5.** Common-mode propagation for differences. Quantities scored against one
      reference share its discretisation error, which largely cancels in their
      differences. Pre-register the resolution threshold as the error *on the difference*,
      not the sum of the bars.
- [x] F5 is shared. Paper 3's **G10** is defined as propagation with common-mode
      cancellation against the same reference filter. Build it once.

**Merge gate:** the cross enumerates completely at `PROVED`. No cell asserts a
separation without printing its ratio and conditioning. The four-term bound is asserted.
A difference and a sum-of-bars are shown to differ on a worked case. R2 and R3 printing
at `PROVED` is PR-5's gate and not this one's: the battery schedules C2 and C3 there,
and a cell value has no warrant until PR-5 registers what it is measured against.
**ADR-059.**

**Met 2026-09-05.** All four conditions hold. The cross carries a
`ProductCompletenessCertificate` at `PROVED`, which warrants the enumeration and not
the cell values: those are `COMPUTED` until PR-5 registers what they are measured
against. F5 lives in `cpomdp.resolution`, a leaf the boundary test asserts, so the seam
and the reference filter reach it without reaching the evaluator. Nothing is exported
at the top level, so no `docs/api/` page is owed yet.

**Warrant:** R1 `EXACT` / Prover 1 with a sampled witness. R2 and R3 `EXACT`–`BOUNDED` /
**Prover 3 · enumeration**. R4 `BOUNDED` / Prover 3 · sample.

## PR-5 — Control bracket and Paper 2 Part 1 results

`serves: 2` · `blocked by: PR-4` · `alias: P2-4 (E), R1–R5, C5` · `size: L` ·
`tag: v0.4.5`

Gate-independent. It banks R1–R5 and C5 before the gate is even attempted, which is half
of what survives a failure.

- [x] Finite-horizon backward Riccati recursion, needed twice: the full-information floor
      `J_lower`, and matched-horizon comparison against the EFE planner. Extend
      `cpomdp/control.py`, which already carries `LQRController`.
- [x] **Match the horizon.** A receding-horizon planner at horizon H implies the
      finite-horizon gain with zero terminal cost, which converges to but does not equal
      the steady-state gain. An unmatched comparison produces a mismatch that shrinks with
      H and looks exactly like a bug. About twenty lines, and it must exist before any
      control comparison runs.
- [x] Certainty-equivalent controller for `J_CE`.
- [x] Bracket width as the primary reported object. `η_ctrl` is derived, within-model,
      with a stated resolution floor.
- [ ] **R1** correct + exact: both terms below 1e-12, with the ratio and conditioning.
- [ ] **R2** wrong + exact: misspecification positive and stable across a **declared**
      seed set, and the inference gap below 1e-12.
- [ ] **R3** correct + degraded: misspecification below 1e-12, inference gap positive.
- [ ] **R4** additivity: three terms reconstruct measured `E[F]` within the four-term
      bound.
- [ ] **R5** `EXACT` control signature: `J_CE = J*` in closed form, bracket width
      `= J_LQG − J_LQR`, `η_ctrl = 0` to the stated floor.
- [ ] **C5** matched-pair dissociation. **Solve for the matched pair analytically, not by
      root-search over perturbation magnitude** (Route 1). Frame it as demonstrating
      incompleteness of the received accounting, never as refuting the FEP.

**Merge gate:** `J_CE = J*` to machine precision in closed form. R1–R5 and C5 all print
and assert. No check claims `PROVED` without a certificate.

**Warrant:** R5 `EXACT` / Provers 1–2, the agreement of two independently computed closed
forms. That is the strongest thing the tier table licenses without a bound.

## PR-6 — Paper 3 toolbox and Part 1 results

`serves: 3` · `blocked by: PR-1, PR-3` · `alias: G-A, G-C, G-D, G1–G5` · `size: L` ·
`tag: v0.4.5` · `ADR: on landing`

G-A and G-D were the original Phase 0 hedge, scheduled at v0.4.3 and never shipped. They
are gate-independent and cheap. They are what stands between "a gate failure stalls the
programme" and "a gate failure costs Part 2's numbers while Paper 3 Part 1 proceeds on
`EXACT` results that need no reference filter at all". They belong ahead of PR-7, not
after it.

- [ ] **G-A.** A common interface making each action-selection functional swappable in one
      line, with closed-form evaluation on the `EXACT` model class.
- [ ] The variant list lives in the model spec and is **versioned**, so a variant added
      after results are seen appears in the diff rather than in the prose.
- [ ] **G-D.** Every functional comparison returns both the value difference and the
      argmin comparison, and the suite asserts on both. This puts Paper 3's standing rule
      4 in code rather than prose. Reporting one of the two is the equivocation the paper
      exists to remove.
- [ ] **G-C.** Reward-plus-λ-information-gain rival agents, with λ and the precision
      parameter γ each declared **fixed or free in the model specification** rather than
      at analysis time. A free parameter discovered during analysis is an accommodation.
      One declared in the spec is a hypothesis. The code should not let the two be
      confused, and the declaration is mandatory at the type level.
- [ ] **G1** constant-offset lemma. Prove it, then print the value difference and assert
      it flat across policies to machine precision.
- [ ] **G2** argmin-equivalence partition **with its completeness certificate**. The class
      count is the paper's headline and must not be quoted before this runs.
- [ ] **G3** value separation despite argmin identity.
- [ ] **G4** preference-convention discrimination.
- [ ] **G5** the fixed-R λ-sweep, which is G8's control.

**Merge gate:** G1's constant offset prints and is flat across policies at H = 1. The G2
certificate holds at the stated horizon and action set. An undeclared λ or γ is a
construction error, not a runtime warning. **ADR on landing.**

**Warrant:** G2 is **Prover 3 · enumeration**, decided rather than surveyed. G4 is a
sampled existence claim, settled by one construction.

## PR-7a — the printed scheme's failure probes, and the declared budget

`serves: 2, 3` · `blocked by: —` · `alias: —` · `size: S` ·
`tag: v0.4.5` · `ADR: on landing`

Ahead of the ladder. The ladder cannot ship a rung whose budget is undeclared, and the
budget cannot be declared before the probe that sizes it exists. Nothing here touches
`src/`. Everything runs against `research.spinello_stilwell.scheme`, which carries no
guard and takes its budget as an argument.

- [x] **Route 3, the two-sided failure**, in `research.spinello_stilwell.pole_failure`.
      The pole has no value at all: the printed curvature raises `ZeroDivisionError` at
      `σ = 1`. Above it the step collapses in proportion to the distance, which makes the
      stall conditional on the tolerance rather than absolute: at `1e-8` off the pole any
      relative tolerance above `2.84e-7` stops the run on its first iteration and reports
      the prediction, wrong by `0.049`. Below it the sum crosses zero at
      `x = 0.694474243706`, the step runs to `8.8e6`, and past the crossing it climbs the
      objective.
- [x] **Route 5, budget exhaustion**, in `research.spinello_stilwell.budget`. Both halves
      move on all twelve declared cells, and on the widest prior at the strongest
      curvature the covariance is the worse error, `0.425` against the mean's `0.357`.
      `VOID` is a decision and not a preference.
- [x] **The modified iterate against the printed one.** The mean differs by `3.2e-3` at a
      budget of one, `6.7e-6` by five, and exactly `0` run out. That last number is
      ADR-057's "surgical".
- [x] **A convergence-count survey.** The declared families need 10 iterations at
      `1e-14` two prior spreads out. The gap calls the rung nine predictive spreads out
      as well, at the edge of its observation box: there, at `1e-12`, the curvature axis
      needs 65 and `1.5 + 0.5 sin(x)` at spread `0.30` needs 124.
- [x] **What a wider budget cannot buy.** One offset past the box edge the bounded
      family settles into a stable two-cycle instead of converging, so a cap large
      enough to be safe everywhere does not exist and the routing is what handles it.
- [x] **The ADR**, ADR-058: budget 64, tolerance `1e-12` relative to the prior spread,
      and the rung differentiates `R` by automatic differentiation rather than by a
      difference. A difference floors convergence above the declared tolerance, which
      route 5 measured.
- [x] `ROUTE` entries appended under Q3 and Q7 of `research/spinello_stilwell_rung.md`,
      and Q7 closed on the declaration.

**Merge gate:** every number that reaches prose is asserted in the module that prints
it, and a printed table is asserted on the claim it illustrates rather than cell by cell.
Nothing here acquires a warrant, and no module gains a `run_checks`.

**Left open for PR-7**, from ADR-058's consequences: two nodes at the box edge exhaust
the budget, `1.5 + 0.5 sin(x)` at spread `0.30` and `κ = 100`, each carrying `2.6e-18` of
the centre's predictive weight. Whether a voided node drops out, voids the gap, or widens
the budget was the ladder's decision. ADR-060 took it: the node drops out by weight, and
the report carries the weight.

## PR-7 — Exact reference filter and the rule ladder

`serves: 2, 3` · `blocked by: PR-3, PR-7a` · `alias: P2-5 (C)` · `size: L` ·
`tag: v0.4.5`

The hard item, and it is shared. P2-7 (D) moved to PR-7b with the ordering work.

- [x] Grid or quadrature filter over a low-dimensional latent, accepting an **arbitrary
      pointwise-evaluable likelihood** rather than an R(x)-specific one. The generality is
      nearly free for a grid filter and is what lets later model classes reuse the engine.
      `cpomdp.reference.filtering.condition` over the `ObservationLikelihood` protocol.
- [x] Returns `E_p*[D_KL[q ‖ p(x|y)]]` directly. `averaged_inference_gap`.
- [x] Written against a **general transition kernel**, not a hard-coded linear-Gaussian
      one. If Q(x) falls out of the internal interfaces at no cost, let it. Do not
      document it, write examples against it, or claim it in release notes (issue #56).
      `TransitionKernel` is the protocol and `LinearGaussianKernel` the one instance.
- [x] Rule ladder, common interface, **five rungs** (ADR-056): plug-in `R(μ⁻)`,
      Spinello–Stilwell single-step (36) and iterated (35), both with the documented
      modification of ADR-057, belief-smoothed `E[R(x)]`, exact reference at the top.
      Swappable in one line. (36) is (35) at a budget of one, so declaring it separately
      costs almost nothing and buys two adjacent differences that isolate distinct
      mechanisms. **All five built**, in `cpomdp.reference.ladder`, one `RungKind`
      each. The smoothed rung averages `R` under the prior on its own grid, not under
      the Gaussian with its moments (ADR-061). The two Spinello–Stilwell rungs run the
      paper's scalar-observation scheme over a vector state and refuse a second
      channel (ADR-062, #115). Left open on the smoothed rung: `E[R(x)]` need not exist under H1,
      and the box quadrature is finite whether it does or not. The rung has no test
      that returns `Void` when the average leans on the box edge, and the criterion
      is a declaration owed before PR-9's closure modes read the rung.
- [x] The rule list is **declared and versioned**, like `FiniteActionSet`. A rung added
      after results are seen shows up in the diff. `RuleLadder` is the type, validated
      as `InferenceSet` is. `LADDER` is the declared constant at `v1`, written once the
      fifth rung existed, so no version ever names a ladder with a rung missing.
The completeness certificate and the R6 gap move to PR-7b, which is where the numbers
they rest on get measured.

### The rung contract

The substrate is built. `QuadratureGrid` and `GridDensity` carry `mean`, `cov` and
`kl_to`, `FixedNoiseLikelihood` and `StateDependentNoiseLikelihood` evaluate the channel,
and `averaged_inference_gap` takes the rule under test as `approximate_posterior`. The
ladder is `cpomdp.reference.ladder`, five implementations of that callable behind
`Rung.build`, under the contract below.

- [x] **A rung is a factory.** Model in, `ApproximatePosterior` out. The seam passes only
      `(prior, observation)`, so the matrices, the noise function and the budget are closed
      over at construction. This is the front-loading the energy constraint asks for: the
      fixed-sensor rung stays one matrix-vector product per reading. `Rung.build` takes
      the likelihood and not `LinearGaussianModel`: the reference may reach no first-party
      type beyond the validator (`tests/test_module_boundary.py`), and the likelihood
      already carries the matrix and the noise. The Gaussian rungs read those through
      `GaussianChannel`. Per reading the plug-in rung pays one solve of the innovation
      covariance plus the render onto the lattice, and the render is every rung's.
- [x] **One public way to put a Gaussian on a lattice.** `gaussian_on` in
      `cpomdp.reference.quadrature`. The copies in `tests/test_reference_gap.py` and
      `research.checks.gap_identity` now call it.
- [x] **`VOID` on a non-convergent step**, per ADR-056 and route 5's measurement. A rule
      returns `Void` in place of a belief. `averaged_inference_gap` reports the node
      as NaN, adds its predictive weight to `voided_mass`, and averages over the
      measured readings normalised to their own mass, the same conditional reading a
      clipped box already gets through `predictive_mass`. A check that wants the strict
      figure asserts `voided_mass` is zero. This answers the question PR-7a left open:
      a voided node drops out by weight and the report says how much weight
      (ADR-060).
- [~] **Iteration work is labeled and isolable.** RFC-001 has to attribute the
      per-decision cost of an iterating rung without reading the loop body.
      `iterated_update` is the one function that iterates, and its result carries the
      count and whether the run settled. What the seam does not yet carry is the count
      of a reading that converged: `averaged_inference_gap` sees a density and no
      number, so a sweep's total iteration cost is not on its report. That slot is the
      remaining half of this item.
- [x] **`R'` comes from automatic differentiation**, per ADR-058. The declared `1e-12`
      is only reachable with an exact derivative: a central difference carries about
      `1e-11` into the iterate and floors convergence above the tolerance, so a rung
      that differences cannot hold the declaration. Done: `jax.value_and_grad` over
      the channel's `observation_noise_at`, one reverse-mode pass per iterate that
      gives the noise and its gradient together (ADR-063 corrects ADR-062's word for
      it).

**Per-rung merge gate, all three required:**

- Fixed-`R` agreement with the closed-form Kalman posterior, where the Kalman filter is
  the exact Bayesian filter.
- A reported gap that does not move under `o → λo` (ADR-057).
- No finite difference anywhere in the rung. The iterated rung converges at `1e-12` on
  the declared families inside the declared budget, which a differenced `R'` cannot do
  (ADR-058).

### The two Spinello–Stilwell rungs

Durable record: `research/spinello_stilwell_rung.md` for the questions the paper leaves
open and the routes that settle them, `research/spinello_stilwell_hand_derivation.md`
with the scan beside it for the derivation of (35c) to (35e) and the modification, and
ADR-056 and ADR-057 for what is decided.

Run so far:

- [x] Route 1's symbolic half. Seven of eight terms survive an observation rescaling and
      the log-determinant term does not. The empirical half, a rescaling sweep of the
      reported **gap**, ran on 2026-09-12 once the rung existed, and both halves hold.
- [x] Route 2. Four of the six declared families never dip below `R = 1`. Two of them,
      `1 + x²` and `1.5 + 0.5 sin(x)`, attain it exactly, so a units-only repair would
      have had to move the pole and not merely widen a margin. One `λ` clears every
      family except `exp(x)`.
- [x] Route 4, the `∇σ = 0` reduction against the Kalman oracle. Exact in every unit
      choice, for the printed scheme and the modification alike. This is the rung's only
      external check, since the paper simulates (36) and never (35).
- [x] The derivation's three tests, in `research.spinello_stilwell.repair`. Test 1 came
      out different from the notes' prediction and the correction is recorded under Q1.

Decided (ADR-057):

- [x] The shipped rungs (36) and (35) run `R_mod`, the paper's scheme with the `r₃`
      block of (35d) removed. Named as modified wherever it appears.
- [x] No `λ` is declared. The deleted block was the only term a rescaling moved, so the
      shipped scheme has no pole to clear and a unit convention would change nothing it
      reports. Route 2's table stays as the measurement it was. `exp(x)` needs no box
      and no guard for the pole. Q2, Q3 and Q4 close for the shipped rung.

Still owed, split across three PRs:

- **PR-7a** took routes 3 and 5 and declared the budget in ADR-058. Done.
- **PR-7** builds the five rungs behind the seam, under the rung contract above.
- **PR-7b** measures the ordering and prints the R6 gap. PR-8 waits on that number.

One PR would have to hold all three, and two constraints stop it. The budget's ADR reads
a number off a survey that has to be committed and runnable before the ADR cites it, and
the R6 gap blocks PR-8, which should not queue behind a ladder review. Route 7 is
optional throughout and decides nothing the rung needs.

**Merge gate:** agreement with the closed-form Kalman posterior in the fixed-R case, where
the Kalman filter *is* the exact Bayesian filter, and a reported gap that does not move
under `o → λo` (ADR-057). Each rung evaluates. The ladder enumerates completely.

## PR-7b — the ladder's ordering, and the R6 gap

`serves: 2, 3` · `blocked by: PR-7` · `alias: P2-7 (D)` · `size: M` · `tag: v0.4.5`

Split from PR-7 because PR-8 waits on one number in here and should not also wait on a
ladder review.

- [x] **Route 1's empirical half**, changed in purpose by ADR-057. It measures what the
      modification cost against the printed scheme at the declared budget. Run in
      `research.spinello_stilwell.gap_rescaling`: the modification's reported gap is
      unit-free to `1e−14` at both budgets, the printed scheme's moves by four per cent
      at rung (36) and reaches the same fixed point at the declared budget wherever it
      converges, and the scalar scheme matches the ladder's rungs at unit scale. Q1 of
      the rung record, RESOLVED 2026-09-12.
- [x] **Route 6, rung one against (36).** Two adjacent differences separate the
      derivative-of-covariance terms from the iteration. If they do not separate, five
      rungs was the wrong declaration and the record has to say so. Run: both resolve at
      all fifteen spreads, the terms by two to four decades and the iteration by a part
      in `10³` to `10⁵` of that. Five rungs stands. Q5 of the rung record, RESOLVED
      2026-09-12.
- [x] **The completeness certificate** over the declared five. A finite declared set is
      what lets R7 reach a decided ordering rather than a sampled one. A one-axis
      `ProductCompletenessCertificate` over `LADDER`, `PROVED`, in `ladder.certificate`.
- [x] **The R6 gap printed at `COMPUTED`, before certification.** GATE-D4 compares the
      certified bound against that number by the pre-agreed factor, so the gate cannot be
      evaluated until the uncertified signal exists. This is the item that blocks PR-8.
      `ladder.r6_signal` prints it at each spread beside `T`, bit-identical to the
      threshold exploration's field. It exceeds `T` from `σ = 0.1706` up. PR-8 is
      unblocked.
- [x] **Warrants for anything certified.** A check suite declared in
      `research/registered_checks.toml`, with the registration commit ahead of the
      measuring commit for every `PROVED` row. `research.checks.ladder`, nine ids. The
      one `PROVED` row carries `PROVENANCE`, and `tests/test_provenance_ordering.py`
      asks git that its registration precedes its measurement.
- [ ] Route 7, section IV's constants, rides here or nowhere. It fixes what Paper 2 may
      say about the published runs and decides nothing the rung needs. Parked as #117:
      the arithmetic needs the journal version's section IV, and the only open copy,
      the 2008 technical report, has no simulation section.

**Merge gate:** the ordering is reported with its bars, and an adjacent pair whose bars
overlap reports `NOT_RESOLVED` rather than a direction.

**Read once, 2026-09-12.** Three of the four registered orderings fire at the binding
cell. The belief-smoothed rung sits with the plug-in, two to four decades above both
Spinello–Stilwell rungs, at every spread but the largest. Above the registered window
those two fall behind the plug-in. Only belief-smoothed to exact holds throughout.
Every pair resolved at every spread and nothing voided. The battery's RESULT of
2026-09-12 under D1 carries the table. The D1 test proper is PR-9's, under PR-8's bar.

## PR-8 — Certified discretisation bound · GATE-D4 · tag v0.4.5

`serves: 2, 3` · `blocked by: PR-7b` · `alias: P2-6 (C′)` · `size: L` · `tag: v0.4.5` ·
`ADR: on landing`

**Landed ahead of the PR: `research/gate_d4_registration.md`, opened 2026-08-07.** The gate
asks whether the bound is small relative to R6's signal, and the signal comes from the very
filter the bound certifies. By the time the gate is evaluable both quantities exist, and a
factor fixed then can be sized to the answer. The only window in which neither exists is
the one before PR-7 builds the filter, so the registration opens in it and fixes what it
can. Amendments are appended and dated rather than folded in, and the file's git history is
what makes the timing checkable.

- [x] `d4-family-v1` declared before any coefficient was computed. Scalar chain,
      `R(x) = R₀ + κ·x²`, spread `σ²` and curvature `κ` as the swept axes, `μ = √(R₀/κ)`
      derived rather than swept. The selection reason is that Paper 1's `R(x)` result
      already uses this family, so it was not chosen toward the bar.
- [x] The curvature ceiling is settled and **vacuous** for this family, which closes one of
      the three stop branches analytically.
- [x] `c₂ = (R'(μ)/2R(μ))²` in closed form. It is a perfect square, so `c₂ > 0` is an
      analytic positivity statement at leading order rather than a measurement. The
      registered `c₂ ≤ 0` stop branch collapsed into the `R'(μ) ≠ 0` precondition and was
      retired, because a row that can never fire reads as a check when it is not one.
- [x] `c₄` refit over the declared `σ ∈ [0.06, 0.30]`, 28 cases across four `R` families,
      exact `c₂σ²` subtraction, seven-term dimensional basis. Median relative residual
      0.60% against the 24% the `(a, b)`-only basis left. Extraction spread at the
      operating point 0.36%, against 35% before. Two of the seven coefficients came out at
      0.1% and 0.3% of the largest and are reported as consistent with zero, not deleted.
- [x] The dilute-versus-subtract rule was written down before the refit ran and fired for
      **subtract**: 0.36% by extraction and 1.03% by basis fit, both far inside the
      registered `X = 0.1`. The branch was decided without the answer visible.
- [x] The convergence shortfall at the bottom of the range disclosed before the refit, not
      after it.
- [x] `c₄` in closed form under **reverse KL**, which supersedes that refit:
      `7ℓ₁⁴/16 − ℓ₁²ℓ₂/4 + ℓ₂²/8 + ℓ₁ℓ₃/4 − 3ℓ₁²/(4R̄)`. Derived symbolically on the
      branch above, against predictions the registration carried at earlier dates. The
      fit's residual error is explained rather than absorbed: it was a `COMPUTED`
      extraction
      of a small rational. Section 7 of the registration holds it, the disclosure that the
      run preceded its registration, and the out-of-sample runs on the other families,
      where `tanh` fires.
- [x] **`T` has a registered form and no value, and is parked until PR-7.** `k_min` was
      never outstanding and this line said otherwise: the AMENDMENT of 2026-08-07
      registers `k_min = 10`, `β = 0.05`, D2's interval at `2 ± 0.5` and `X = 0.1`, and
      registers `D` as an expression in `k` evaluated at `k_min`. What remains is `f*`,
      re-derived under the sextic edge, which needs a statistical term that two findings
      put in question. Both turn on the reference filter's error *shape*, which is a
      property of a filter PR-7 builds. The DECISION of 2026-08-23 records what unparks
      it and what does not. No longer parked: the RESULT of 2026-09-11 measures the
      filter's error field at the binding cell, finds it benign, and registers
      `T = 5.962e−4` nats at `D = 0.5`, `f = 0.078157`, under the declared lattice.
- [x] **The sweep's lower bound.** Declared at `κ_min = 0.1`, rationalised rather than
      derived, with its revision condition bounded in advance (ADR-049). The argument is
      D2's second leg, registered before `c₆` existed.
- [x] `c₆` in closed form, so the binding truncation is no longer unmeasured.
      `c₆ = −κ(7κ + 9)(13κ − 3)/(48R₀³)` on the ridge, resolved onto an eighteen-term
      basis with remainder zero, `PROVED` at `EXACT` on two agreeing arms. `c₄` vanishes
      at `κ = 2` where `σ_max` diverges; `c₆` there is `−529/24`, and the two never
      vanish together. The derivation is `research/c6_hand_derivation.md`, registered
      before the suite computed anything.
- [x] **`σ_max` against `c₆`.** The subtract branch had already fired on a rule
      registered before its input was visible, so this discharged a consequence rather
      than taking a choice. The edge is `σ_max⁴ = f·c₂/|c₆|` and `T` goes as `√f` rather
      than `f`. The registered `f* = 0.0488` was optimised against the quartic and does
      not carry across.

**The checks that back it live in `research.checks`.** Standalone modules run with
`--check`, not on the `pytest` path, and each prints under `warrantlib`'s vocabulary.
They exist because the scripts that produced the original 28 `c₄` cases were lost and only
prose survived them, which is a failure mode worth not repeating.

- [x] `gap_kernel.py`. The averaged inference-gap quadrature, defined once, pinning the
      three conventions the lost scripts recorded only in prose: reverse KL, `R` frozen at
      the prior mean, the average taken under the true predictive. Callers own the
      questions, this module owns the integral.
- [x] `predictive_truncation.py`. Measures the `y` grid against the density it actually
      integrates. The half-width comes from the plug-in `R(μ)` while `p*(y)` is a scale
      mixture, so a "`k` standard deviations" rule sizes a Gaussian tail against a
      non-Gaussian one and no `k` is safe on principle. **Six of 49 cells fire**, all on the
      two unbounded-`R` families at `σ = 0.25` and `0.30`: the nominal 9σ grid covers 8.81
      to 8.87 true standard deviations, and the log-density there is linear rather than
      quadratic (slope `−2.604`/unit for `R = 1 + x²`, crossover at `ν* ≈ 13.93` against a
      grid edge at 13.01), which is the exponential tail the rule cannot see. C4 settles
      what it costs: the worst truncation is four or more orders below the 0.36% extraction
      spread already carried on `c₄`, so the fires are real and immaterial to the
      coefficient. The module reports and does not fix. The production grid rule stays where
      it is and the red cells are the output.
- [x] `gap_expansion.py`. Structure of the small-spread expansion **without** its
      coefficients. It certifies the gap on three axes, checks `c₂` against its closed form
      to `5.8e−7` relative or better on all four families, and tests the residual's
      *exponent* after each known term comes off: `σ^3.965` to `σ^4.038` against a predicted
      `σ⁴`. Seven of 38 cells fire, all G3 quadrature certification on the two bounded
      families below `σ = 0.04`, where the `x`-extent and refinement tolerances sit up to
      21× over a `1e-10` bar. It never fits a `c₄`, because a number produced here would
      become the thing the derivation is checked against and the ledger would carry a `COMPUTED`
      fit under a Prover 1 label. Pass a derived candidate in with `--c4` and G4b tries to
      refute it. G4c refits G4b's exponent once per omitted `σ` cell and reports the
      spread, which classifies a fired G4b as grid instability or as a real deviation. It
      is a diagnostic and revises no outcome.
- [x] `series_kernel.py`. The construction of `W`, its truncated expansion, the moment
      operator, and the exact predictive average. 23 pins. Explicit polynomial truncation
      rather than `sympy.series`, which costs about an order of magnitude per order on
      the assembled `W`. `series` survives as the independent arm that licenses the swap
      through `σ⁴`, which is where `DERIVATIVE_ORDER` stops the truncation path.
- [x] `log_ratio_series.py`. Symbolic pins for the log-ratio series `W`, 18 of them, all
      holding. It stops at first order in `σ` and its expectation, which is where the
      structure lives and the arithmetic does not. Adds `sympy` as a dev dependency.
      Nothing under `src/cpomdp` imports it.
- [x] `gap_series.py`. `c₂` and `c₄` from the kernel, 29 identities. No floats and no
      numeric `R̄` in the derivation, so there is no quantity a measured number could be
      substituted into. C7 specialises to the exponential family after the fact, never
      before it. That is what makes the agreement with the earlier fit evidence rather than
      circularity, and it is checkable by reading the module.

- [x] Write down the **pre-agreed factor** before this PR is opened. A factor agreed after
      seeing the bound is not a gate. The registration is where it goes, and `T` is an
      expression there rather than a value, so this closes when `T` does. Closed by the
      RESULT of 2026-09-11: `T = 5.962e−4` nats, so D1 and D2 are tests iff
      `δ_ref ≤ 5.96e−5` nats.
- [ ] A **certified** bound, not a fine grid with a convergence plot. Interval arithmetic
      or a proved quadrature error bound, the device licensing *for all x in the domain,
      |p_grid − p_exact| ≤ δ*.
- [ ] Stated as a number and shown small relative to R6's measured signal by the
      pre-agreed factor.
- [ ] Emits `CERTIFIED`, not `PROVED`. The distinction is validated numerics against
      enumeration, and it is not
      cosmetic.
- [ ] **Cut v0.4.5 at this merge whatever the outcome.** Release notes record the gate
      result as a number against the factor, `PASS` or `FAIL`. Changelog, `DECISIONS.md`
      entry, `CITATION.cff` and `__init__.__version__` on the release commit, matching
      v0.4.4's discipline.

**Merge gate — hard, existential.** See "GATE-D4" below. **ADR on landing.**

## PR-9 ⛔ — Window harness and Paper 2 Part 2 results

`serves: 2` · `blocked by: GATE-D4, PR-4` · `alias: P2-8 (F6), R6–R9` · `size: L` ·
`tag: v0.5`

- [ ] Before any exponent is fitted, demonstrate the D2 fit window is non-empty. Print its
      bounds **on both swept axes** and the signal-to-bound ratio across it.
- [ ] Three independent closure modes, all checked. Higher-order contamination at large
      spread. Relative-error divergence against the bound at small spread. The
      belief-smoothed rung requiring `E[R(x)]` to exist, which H1 puts no ceiling on, so
      at high curvature a rung can drop off the ladder entirely.
- [ ] An empty window is **VOID**, not `FAIL`. Report the leg unmeasurable and route it to
      the adaptive-grid register item. The lower edge sits near `√(k·δ_ref/curvature)`, so
      a tighter bound widens it directly. This is the one failure that code can fix.
- [ ] **R6** gap positive and separated from the certified error by the pre-agreed factor.
- [ ] **R7** ladder ordering, three-valued, with F5 difference propagation. Pre-register
      the minimum separation that counts.
- [ ] **R8** scaling exponent, only against a demonstrated non-empty window, else VOID.
- [ ] **R9** the bound as a number, small relative to R6's signal.

**Merge gate:** the window's bounds print, and R8 does not run until they do. R6–R9 print
and assert at their stated tiers.

## PR-10 ⛔ — Paper 3 Part 2 results

`serves: 3` · `blocked by: GATE-D4, PR-2, PR-6` · `alias: G6–G10` · `size: M` ·
`tag: v0.5`

- [ ] **G6** class splitting under R(x).
- [ ] **G7** argmin divergence separated from the certified error.
- [ ] **G8** the λ-sweep under R(x).
- [ ] **G9** crossover interaction, `H*` per functional, inheriting PR-2's action-set
      qualifier.
- [ ] **G10** certification report with common-mode propagation, reusing PR-4's F5.

**Merge gate:** G6–G10 print and assert. G9 carries PR-2's stability outcome.

## PR-11 — v0.5 release

`serves: 2, 3` · `blocked by: all` · `size: M` · `tag: v0.5`

- [ ] R1–R10 and G1–G10 all print and assert in one run.
- [ ] Changelog, docs, examples gallery, deferred-and-unsupported section with an issue
      per boundary.
- [ ] Zenodo archive. **Papers 2 and 3 both pin v0.5.**
- [ ] Close or re-register every carry-over below. Anything still open ships as
      registered, never as silence.

---

## GATE-D4

With one bound and two tags, the gate is a blocking condition on **PR-8** and an explicit
merge block on **PR-9 and PR-10**.

- [x] Write down the pre-agreed factor before PR-8 is opened.
      `research/gate_d4_registration.md` carries it. The family, the stop branches and the
      gate's form as `gap > T` are all dated 2026-08-07, before any coefficient existed.
      `c₂` and `c₄` have since landed, both in closed form. `T` took its value on
      2026-09-11, `5.962e−4` nats, with the PRE-REGISTRATION committed ahead of the RESULT
      and before PR-7 merges.
- [ ] Mark PR-9 and PR-10 blocked in the tracker, not by convention. A gate honoured by
      memory is not honoured.
- [ ] Tag v0.4.5 at PR-8's merge regardless of outcome.

**On failure: stop.** Do not merge PR-9 or PR-10 against an uncertified reference.

**What survives a failure, and it is a paper.** Paper 2 re-scopes to `EXACT`: Part 1
complete (R1–R5), R10 complete, C5, all of battery families A and B, C1–C5, D3, E1's
`EXACT` leg, and all of F. **Paper 3 Part 1 is untouched**, because it is closed-form and
touches no grid, so PR-6 still stands. What dies: D1, D2, C6, G7, G10, and the certified
half of G6–G9. Two publishable units remain, which is what the hedge was built to
preserve.

---

## Buying the highest level of provability

Three routes, descending value per hour.

### Route 1 — free: restate the claim so its natural prover is 1 or 2

- **R1** is not "two numbers came out below 1e-12". In fixed-R linear-Gaussian with
  p = p\* and exact inference, both divergences are **identically zero by construction**.
  Prove it, then assert numerically as witness. `EXACT`, Prover 1.
- **R5** is agreement of two independently computed closed forms. Do not compute `J_CE`
  and `J*` by simulation and compare.
- **C5's matched pair** should be **solved for analytically**, not found by root-search. A
  searched match is a sample and carries the searcher's tolerance. A constructed match is
  `EXACT`. Same result, two tiers apart.
- **G1** is a lemma with a proof, not a numerical observation that an offset looks flat.

### Route 2 — cheap: make the quantified domain finite and declared, then certify the cardinality

The sample → enumeration conversion, and the largest return per hour on offer, because v0.4.4 already
built the machinery. `CompletenessCertificate` exists. Pointing it at a new domain costs
an afternoon and converts "we found no counterexample" into "there is no counterexample
on this set".

- [ ] **Constructor cross** (PR-4). R2 and R3 decided rather than sampled. **Do this one
      first.**
- [ ] **Evaluation-rule ladder** (PR-7b). Five declared rungs, so once each rung is
      `CERTIFIED`, "rung i < rung i+1 for all adjacent i" is a finite conjunction of
      certified interval comparisons. **Decided, not sampled.** R7 reaches `PROVED`
      conditional on `CERTIFIED` rung values, which is stronger than the original framing
      and free.
- [ ] **Functional variant list** (PR-6). G2's partition is already built on this, and the
      same discipline applied to the rival-agent set extends it to G8.
- [ ] **Seed sets.** A declared seed list with an asserted cardinality is a finite domain.
      "Stable across seeds" over a declared list is decided. Over "some seeds we ran" it
      is not.

**The boundary, stated so it is not crossed.** A finite grid over a *continuous* range is
still a sample. Sweeping perturbation *magnitude* on a grid corroborates. Enumerating a
*declared set of magnitudes* decides a claim about that set and nothing more. The
certificate licenses the enumeration, never the sweep.

### Route 3 — paid: validated numerics

Validated numerics are the only route to a universal over a continuum by execution, and
that is what GATE-D4 buys. R6, R7's rung values, R8, R9, G7 and G10 sit at `COMPUTED`
without it.

**Second-order return, worth pricing in.** A tighter bound does not only certify. It
widens D2's fit window, whose lower edge sits at `√(k·δ_ref/curvature)`. Adaptive
quadrature buys warrant on R6, R7 and R9, and it buys R8 the right to exist.

### What cannot be bought

- **R8's exponent.** A fitted exponent over a continuous sweep is a sample, permanently. Report
  it `CORROBORATED` against a registered interval and stop. Its severity comes from having
  been able to come out wrong, not from its warrant.
- **Any universal over the action continuum.** `H* = 7` is decided over the declared
  `{0, ±1, ±2}`. PR-2 bounds the artefact risk. It does not remove the qualifier.
- **Equivalence over a continuum of models.** G2 is proved over an enumerated policy set
  at a stated horizon and action set. Do not let the abstract widen it.
- **Theorem 1's ceiling.** Finite-dimensional, H3′ for the title-strength claim, Remark 2's
  dual-effect scoping. All three travel with every restatement.

### Warrant ledger, per result

| Result | Best prover attainable | Tier | Bought by |
| --- | --- | --- | --- |
| R1 | **1** plus a sampled witness | `EXACT` | Route 1 |
| R2, R3 | **3 · enumeration** over the declared cross | `EXACT`–`BOUNDED` | Route 2, PR-4 |
| R4 | 3 · sample, **3 · validated** if all four bars certified | `BOUNDED` | PR-4 plus Route 3 |
| R5 | **1–2** plus a machine-precision witness | `EXACT` | Route 1, PR-5 |
| R6 | **3 · validated** | `BOUNDED` | GATE-D4 |
| R7 | **3 · enumeration over rungs, conditional on validated rung values** | `BOUNDED` | Routes 2 and 3 |
| R8 | 3 · sample, the ceiling | `BOUNDED` | Nothing. Say so |
| R9 | **3 · validated** | `BOUNDED` | This *is* Route 3 |
| R10 | **3 · enumeration**, already achieved | `BOUNDED` | Certificate, hardened by PR-2 |
| G1 | **1** plus witness | `EXACT` | Route 1 |
| G2 | **3 · enumeration** with certificate | `EXACT` | Shipped at v0.4.4, applied at PR-6 |
| G4 | **3 · sample existence**, settled by one construction | `EXACT` | Construction |
| G7, G10 | **3 · validated** | `BOUNDED` | GATE-D4 |

---

## Failure routing

| Failure | Routing |
| --- | --- |
| **GATE-D4 fails** | PR-9 and PR-10 do not merge. Paper 2 re-scopes to `EXACT`. Paper 3 becomes an `EXACT` paper on the collapse of the zoo. Two units survive |
| **Ladder non-monotone** | Reported. Registered as a conjecture with a predicted direction, not a theorem |
| **Adjacent rungs overlap** | `NOT_RESOLVED`. Needs PR-4's F5 to have any resolving power |
| **D2 window empty** | VOID, routed to the adaptive-grid register item. PR-9 catches it before the fit |
| **`H*` moves under refinement** | `FAIL` on the fourth falsifier. Report the sensitivity, scope the claim to the declared set. G9 inherits |
| **Refinement exceeds budget** | VOID, unmeasured, never "stable" |
| **Functionals coincide under R(x)** | Pre-registered as reportable. Part 1's proved degeneracy stands alone |
| **C6 gap at machine precision under R(x)** | `FAIL` against a theorem. Bug hunt before result |

---

## Documents to amend

Three tracked programme documents carry the gate. A fourth, the R10 write-up, carries the
result PR-2 hardens.

- [x] `research/gate_d4_registration.md`, new. It holds the gate's registered side:
      `d4-family-v1`, the stop branches, `c₂`, `c₄`, the dilute-versus-subtract rule and
      the `T` expression. PR-8's section lists what is fixed and what is still outstanding.
      Amend it by appending a dated block, never by editing the text a later block
      corrects.
- [x] `research/warrant_ledger.md`. The five "v0.4.4 gate" sites read `GATE-D4`, the
      header names v0.4.5 as the tag that carries it, the scoring-harness and rule-family
      pinning points at PR-3/PR-4 and PR-7, and the opening line names the battery by
      filename.
- [x] `research/warrant_ledger.md`, again. The instrument changed under R10's published
      numbers, so the two paths' agreement is registered in the ledger rather than left in
      the test suite: identical argmin index and `G` equal under `==` at `9⁷`, residency
      12.5× down, throughput 3.0× up, and the superseded budget lines restated at 39.0k
      policies/s.
- [ ] `research/warrant_ledger.md`, still open. Section 2's availability column moves off
      bare version numbers onto PRs.
- [x] `research/r10_open_loop_crossover.md`. "Optimal reach" was the wrong name for `−3`
      and is now "one-step reach" throughout, since `−3` reaches the goal in one step from
      the start and is not optimal from where the walk actually stands. The deduced
      `{−4,…,2}` row is retracted and reads *not measured*, with the reason stated and the
      measurement routed to PR-2's extension axis.
- [x] `research/fep_falsification_battery.md`. Readiness reads `PR-n · tag` throughout,
      `GATE-D4 at v0.4.6` reads `GATE-D4 · PR-8 · v0.4.5`, D3's "E outstanding, scheduled
      v0.4.5" reads PR-5, and D3 now completes at v0.4.5 rather than v0.5, because PR-2
      and PR-5 both land before the gate.
- [ ] `research/fep_falsification_battery.md`, still open. Two items need a decision
      rather than an edit, both listed under "Registered wording" below.
- [ ] This file supersedes the earlier build plans. Anything still citing them is stale.

**Registered wording, two open decisions.**

- [ ] **D3's prediction sentence.** The battery registers "the accumulated epistemic pull
      overtakes the pragmatic gradient". The ledger's section 10 records that as literally
      false against the measurement: `Δε` is flat (1.72 → 1.64) and `Δc` decays
      (4.49 → 0.86) to cross below it. The mechanism is a decaying pragmatic gradient. The
      ledger says D3 should be restated in those terms. Restating a registered prediction
      is not an editing call, so the battery still carries the original wording.
- [ ] **E2's readiness.** The battery reads "further work" for the rival-agent harness.
      PR-6 builds one for Paper 3's G-C. Whether Paper 3's harness serves Paper 2's E2 is
      a scoping call, so the entry is unchanged.

---

## Standing build rules

1. No number reaches prose unless the check suite prints and asserts it.
2. Conditions are reported, not assumed. A `probe_model` per cell.
3. "Certified" requires a stated tolerance. Otherwise the word is "computed".
4. Import Paper 1 by number, never re-derive. Import Paper 2 by result ID in Paper 3.
5. A public surface costs docs, tests, examples and a support commitment. Undocumented
   capability that falls out of internal interfaces stays undocumented until it is
   scheduled.
6. Warrant is a property of the check, not of the number. The suite distinguishes a
   sample from an enumeration in its output vocabulary rather than printing both as
   `PASS`.
7. **A declared set is versioned or it is not declared.** Action sets, rule ladders,
   constructor crosses, seed lists, functional variants. Each lives in the model spec, so
   an addition after results are seen appears in the diff rather than in the prose.
8. **`PROVED` requires a certificate object, enforced at construction.** An enumerator
   asserting its own completeness without checking cardinality proves nothing, and the
   type system should say so before the test runs.
9. **A blocked PR is blocked in the tracker, not by memory.**

---

## Carried forward from the v0.4.4 window

The deferred register, one issue per boundary. Each ships as registered rather than as
silence, per standing rule 5.

- [ ] **#52** closed-loop `FfgEfeSelector` above H = 1. Unsupported, not untested.
      ADR-034 notes the FFG-backed drivers would now make it nearly free. It stays
      deferred, because nothing scoped needs it.
- [ ] **#53** pruning the `|A|^H` enumerated search without losing the completeness
      certificate. Defer rather than half-build. A design that does not start at the
      certificate is not a design.
- [ ] **#54** `GradientEfeSelector`, the continuous-action corroboration track. Detail
      below.
- [ ] **#55** notation unification. Detail below.
- [ ] **#56** the Q(x) surface above H = 1. Works, undocumented, unclaimed. PR-7 must not
      change that.
- [ ] **#58** does `H*` collapse onto a dimensionless group. Design pass done, build
      deferred.

### Notation unification (#55)

The `μ`/`Σ` predict/update superscript convention is inconsistent across the codebase.
Three variants coexist. `efe.py` and its neighbours use `μ⁺`/`Σ⁺` for the predicted
moment with `Σ_post` after observation. `kalman.py`, `ffg/chain.py` and `diagnostics.py`
use standard Kalman `μ⁻`/`Σ⁻` with `post`. `coupling.py` uses both symbols for
pre-coupling against coupling-resolved *predicted* means (ADR-019), a distinction
orthogonal to predict/update.

Variant 3 is the trap. A naive `μ⁺` → `μ⁻` sweep corrupts ADR-019's meaning.

- [ ] Pick one canonical convention. Standard Kalman `Σ⁻`/`Σ⁺` is the textbook default and
      what external readers expect. Record it as an ADR, since it touches ADR-003's
      epistemic-collapse wording and ADR-019.
- [ ] Give `coupling.py`'s pre-coupling against coupling-resolved means a **separate**
      disambiguator, not the predict/update superscript, so ADR-019's distinction survives
      the rename.
- [ ] Sweep comments and docstrings across `src/` to the chosen convention. Verify it is a
      pure doc change: arithmetic byte-identical, whole suite unmodified, `ruff` and `ty`
      clean.

### Continuous-action corroboration track (#54)

A continuous-state agent should also exercise genuinely continuous action spaces, not
only declared finite repertoires. `GradientEfeSelector` is that selector: gradient ascent
on the differentiable `policy_efe` over a continuous action box.

**Warrant: Prover 3 · sample / `CORROBORATED` only.** Gradient ascent finds a *local* optimum of a
non-convex objective. Like the grid it searches a continuum without exhausting it, so it
can never *decide* a universal over the action space. Every result it produces carries the
`CORROBORATED` label, never `PROVED`, never a bare `PASS` (standing rule 6).

**Home: the self-acting regimes, not the p\* scoring harness.** PR-3's harness runs in
exogenous action mode and severs the control loop, so action *selection* is not part of
the decomposition. Continuous action *values* already are. `GradientEfeSelector` belongs
to the self-acting brackets: a corroborating companion to the crossover and, principally,
to PR-5's control bracket, where a self-acting agent under R(x) steers toward low-noise
regions and changes its own gap.

**Wall: strictly separated from the enumerated evidence.** R10's crossover decision rides
on the finite enumeration. A gradient-selected policy may corroborate alongside it. It
must never enter the decisive cells, or the enumeration's certificate is contaminated
back to a sample.

- [ ] `GradientEfeSelector` over a continuous action box (`p >= 1`), returning the
      optimised sequence and its `G`. Labelled `CORROBORATED`.
- [ ] The warrant label travels with every continuous-action result, printed and asserted,
      so a corroboration is never read as a certification.
- [ ] No gradient result enters R10's enumerated evidence. The two families' outputs stay
      separately labelled in any shared harness.

**Not this track — register, do not build.** *Certified* continuous-action coverage,
deciding "no action in the compact box flips", is Prover 3 · validated. It needs a certified
branch-and-bound with Lipschitz or interval bounds on `policy_efe` over the box. That is a
distinct, larger, later workstream. `GradientEfeSelector` does not deliver it and nothing
here should imply it does.
