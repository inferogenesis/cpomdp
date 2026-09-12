# Changelog

Everything worth noting lands here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [semantic versioning](https://semver.org). While we're pre-1.0, treat the minor version as the place breaking changes can show up.

## [Unreleased]

Warrant plumbing and the `slogdet` oracle audit. The two-level `SearchWarrant` becomes a
shared three-level `Warrant` that checks can reach, and a check now reports what it found
alongside how well it was warranted and what it was measured against. The audit half
confirmed what v0.4.3 left behind: the kernel's log-determinant guard was never applied to
the NumPy oracle the headline `H* = 7` is checked against.

The `H = 7` numbers did not move. They are bit-identical across the guard, so it corrected
an exposure rather than a result.

A world and an agent that cannot reach one another. `cpomdp.harness` separates the process
that produces observations from the model an agent filters with, so how wrong that model
is about the process becomes measurable rather than zero by construction. Actions arrive
from a declared sequence. The agents do not choose them, which is what leaves several of
them comparable under one trajectory and what cuts the control loop, and every run carries
that second fact rather than leaving it to a comment.

`cpomdp.constructors` declares the two axes a model can be wrong along, one for what its
parameters get wrong and one for what its filter does. Both are versioned, so a cell added
after results are seen shows up in the diff.

The cross gets its two numbers. `cpomdp.scoring` scores every cell of the declared
constructor cross with two divergences, what the model gets wrong and what the filter
gets wrong, and nothing else: the result has no slot for an entropy, so the one term
the ledger forbids reaching by subtraction cannot be reached that way. Every separation
is a ratio, printed beside the conditioning of every matrix it inverted. The check that
the two divergences and the entropy add up to the measured free energy is a separate
object, the one place an entropy is estimated. And a difference between two numbers
scored against one reference is read at the error on the difference, which the shared
reference makes far smaller than the sum of the two bars.

The control bracket. `cpomdp.control` grows a finite-horizon regulator beside the
steady-state one, and with it the two costs every controller under one plan sits between:
the plan with the state observed exactly, and the best any controller that has to infer
the state from readings can do. Their difference is the price of inference under that
plan, and the width is the object reported, since either end alone is a number in cost
units that nothing calibrates. Both ends are closed forms. The ceiling is reached twice,
by the separation principle and by propagating the joint second moment of state and
estimate, and the two share no arithmetic past the Riccati recursions, so their agreement
to machine precision is the fixed-noise signature: certainty equivalence is optimal, and
no use of the readings beats it.

### Added

- `cpomdp.harness` — a world and an agent held apart, so what an agent's model gets wrong
  about the process is a measurable quantity rather than zero by construction. `World`
  runs the process and hands out observations, exposing no accessor that returns its
  model or a parameter of it. `ScoredAgent` runs its own model and has no constructor
  slot for a `World`. `drive` advances one world per step and folds the same reading into
  every agent, over an `ExogenousActionSequence` declared before the run. Imposing the
  actions is what makes the entropy of the true process cancel between agents, and it
  cuts the control loop: `DrivenRun.control_loop` carries that as a `ModellingChoice`
  with no default, saying what was chosen and where the choice is contestable. A
  state-dependent `R(x)` is read at the state that produced the reading, and `Q(x)` at
  the predicted mean, rather than at the point the filter evaluates them: the filter
  drops a Jensen term the world must not drop too, since the distance between them is
  the inference gap under study (ADR-048, superseding ADR-047's refusal). Both are
  re-checked on every step, because construction probes them at one state only. The two
  fail differently and the check follows that: the sensor draw is cholesky, so a noise
  that loses definiteness returns NaN, while the dynamics draw is svd and returns a
  finite state drawn from `|Q|`. An indefinite `Q(x)` is refused; a noiseless state
  direction stays legal.
- `cpomdp.constructors` — the two axes a model can be wrong along, each declared and
  versioned. `ModelSpec` holds parameters and builds a fresh model per call, sharing no
  array between builds, so two models that agree do so by value rather than by pointing
  at one object. `Perturbation` names a parameter and scales it. `InferenceRule` names
  one of four ways to infer. Both are records rather than functions, so a declared set
  can be read in a diff. A scale the filter would not read is refused rather than built,
  since the cell would otherwise report a label for a model identical to the correct one.
- `cpomdp.backends.degraded` — `WrongFixedRBackend` and `DiagonalCovarianceBackend`, two
  filters that infer worse than the model they are built for and name how. Each keeps
  `model` pointing at the model being scored, so a degradation is never an anonymous
  mismatch between two models.
- `cpomdp.scoring` — the three-term evaluator. `build_cross` enumerates the declared
  model and inference axes in full and carries a `ProductCompletenessCertificate` at
  `PROVED`, refusing a cross that comes up short. `Decomposition` holds a
  misspecification term and an inference gap, both KL divergences in nats and both in
  closed form on a linear-Gaussian model. `gaussian_kl` is built from non-negative parts
  and subtracts nothing, so a reading near zero is a small divergence rather than a
  cancellation. `ThreeTermEvaluator.score` walks a driven run and returns a `CellScore`
  with a `RunConditioning` beside it: the condition number of every predictive and
  posterior covariance the terms factored, per step. `score_cross` scores every cell,
  reports the cell where both terms move, and refuses to name a separation without one.
  A `Separation` is the pinned term, the moving term and their ratio, and `render`
  prints no separation without its ratio and its worst conditioning on the same line.
- `cpomdp.additivity` — whether `E[F] = H(p*) + D₁ + D₂` closes, with both sides
  measured on their own. `AdditivityCheck` takes an `EntropyEstimator` explicitly and is
  the only object that estimates one; `GaussianEntropy` is the closed form and
  `MonteCarloEntropy` samples with a standard-error bar. `variational_free_energy` is
  computed term by term from the cell filter's belief, never through the identity. The
  residual's bound is four terms, `δ_F + δ_H + δ₁ + δ₂`, and a worked case shows it
  failing to close when `δ_F` is dropped.
- `cpomdp.resolution` — the error on a difference. A `Bar` is split into a signed
  common-mode part, what the reference's error puts on the value, and an own part. Two
  values against one reference subtract the first in their difference, so
  `resolution_threshold` reads `|Δ common-mode| + Σ own` rather than the sum of the two
  bars. It reads bars only, which is what makes it pre-registered. `resolve` reports
  `BELOW`, `ABOVE` or `NOT_RESOLVED`, the measured tie that is neither confirmation nor
  refutation. The module imports nothing first-party, asserted in the boundary test, so
  the seam and the reference filter reach it without reaching the evaluator.
- `cpomdp.control` — the finite-horizon schedule and the control bracket.
  `finite_horizon_lqr` runs the control Riccati recursion backward a declared number of
  steps from a zero terminal cost and keeps every gain, where `LQRController` iterates to
  a fixed point and keeps one. `FiniteHorizonLQR.first_gain` is the gain a
  receding-horizon planner at horizon `H` applies at every step; the steady-state gain is
  its limit and differs from it at every finite `H`, by an amount that shrinks with `H`
  and reads as an error when the horizons are not matched. `kalman_schedule` writes the
  exact filter's gains and covariances down before a reading exists, since under fixed
  noise they never see one. `full_information_cost` is `J_lower`, the plan priced with
  the state observed. `CertaintyEquivalentController` acts on the estimate as if it were
  the state, and `closed_loop_cost` prices it, or any data-independent gain sequence, by
  propagating the joint second moment of state and estimate; nothing is sampled.
  `optimal_cost` is `J*` by the separation principle, and the two reach one number by
  different arithmetic. `control_bracket` returns the floor and ceiling as a
  `ControlBracket`, and `control_efficiency` reads `η_ctrl` from it: the share of the
  width an agent recovers, within-model, with a bar that is the agent's own divided by
  the width, the floor below which `η_ctrl` cannot be told from zero. A goal the
  dynamics do not hold, a noise that depends on the state, and a bracket of zero width
  are all refused. The Control page of the API reference opens the schedule and the
  bracket in plain terms.
- `cpomdp.reference` — the exact reference filter's substrate, reached by module path
  and not re-exported at the package top. `QuadratureGrid` and `GridDensity` are a
  declared lattice and log-values on it, and `gaussian_on` is the one place a Gaussian
  is put on that lattice. `averaged_inference_gap` sweeps an observation box and returns
  an `InferenceGap`: the rule's expected divergence from the exact posterior under
  `p*`, beside the predictive mass the box caught and the worst edge ratio. A rule may
  answer `Void` for a reading it could not converge on. The sweep then reports that
  node as NaN, sums its weight into `voided_mass`, and averages the answered readings
  normalised to their own mass, the conditional reading a clipped box already gets
  (ADR-060).
- `cpomdp.reference.ladder` — the rule ladder. A `Rung` is a factory: the likelihood a
  reading is conditioned on goes in, and the `rule(prior, observation)` callable the
  gap measures comes out, with everything the rung reads from the model closed over at
  construction. All five of ADR-056's rungs are built: `PLUG_IN`, the Kalman update
  with the noise read once at the prior mean; `SINGLE_STEP` and `ITERATED`, the
  Spinello–Stilwell scheme under ADR-057's modification at a budget of one and at
  ADR-058's budget and tolerance, with the noise's slope taken by automatic
  differentiation and one observation channel only (ADR-062); `BELIEF_SMOOTHED`, the
  Kalman update with the noise averaged under the prior on its own grid (ADR-061); and
  `EXACT_RUNG`, the reference at the top. `LADDER` is the declared `v1` set of the
  five, in the order the battery's D1 leg is registered over. `iterated_update` runs
  the scheme on one reading at any budget and returns its count, so the cost of an
  iterating rung is read off a result and not a loop body. `RuleLadder` is the
  declared, versioned set the rungs sit in, refused without an exact rung the same
  way an `InferenceSet` is refused without an exact cell. The Gaussian
  rungs read the channel through `GaussianChannel`, a second small protocol beside
  `ObservationLikelihood` that adds the observation matrix and `observation_noise_at`,
  so the exact rung is not asked for either.
- `research.explorations.threshold` — GATE-D4's threshold `T` takes a value. The
  reference engine's error field is measured across the fit window at the binding
  cell, its shift on the fitted exponent replaces the registered `σ_p`, and
  `T = 5.962e−4` nats is registered at `D = 0.5`, `f = 0.078157` under a declared
  lattice (`research/gate_d4_registration.md`, PRE-REGISTRATION and RESULT of
  2026-09-11).
- `research.checks.ladder` — the ladder's first reading. Five rungs on `d4-family-v1`
  at the binding cell, at the fifteen spreads and on the two lattices `T` was registered
  under, each gap with a bar measured by refinement and carried whole as its own. One
  row per adjacent pair resolves the ordering through `cpomdp.resolution` at every
  spread, `NOT_RESOLVED` where the bars overlap, and every ordering row is `COMPUTED`
  since the bar is measured and not certified. Two rows read route 6 off the same
  numbers, one reports the R6 signal beside `T` with no verdict on the gate, and the
  completeness certificate is a one-axis product over the declared ladder. Registered
  under D1 of the battery before it was read. Read once: three of the four registered
  orderings fire at the binding cell. The belief-smoothed rung sits with the plug-in,
  two to four decades above both Spinello–Stilwell rungs, and above the registered
  window those two fall behind the plug-in. Both route 6 mechanisms resolve at every
  spread. The battery's RESULT of 2026-09-12 carries the table. A tenth row reads the
  lattice at every spread on both lattices against the two bars `T` was registered
  under. The suite stays out of the manifest, since it runs the reference engine and
  three of its rows fire by recorded result. The slow test path pins its ids and
  outcomes instead.
- `research.spinello_stilwell.gap_rescaling` — route 1's empirical half, on the
  reported gap. The printed scheme and ADR-057's modification run through the gap at
  four unit choices, at rung (36)'s budget of one and at rung (35)'s declared budget.
  The modification's gap is unit-free to roundoff at both. The printed scheme's moves
  by four per cent at a budget of one and reaches the same fixed point at the declared
  budget wherever it converges. Q1 of the rung record closes on it.

- `Suite.refuted` (warrantlib 0.4.0) — a manifest suite may list the checks whose
  registered result is a refutation. The pytest plugin passes such a check by firing,
  rendered `REFUTED`, and fails it by name as `NOT REFUTED` otherwise, so a job that
  reconciles the manifest holds a recorded refutation instead of going red on it. The
  warrant summary still counts the check as fired and names what was held. Written by
  hand after the result is on record, kept by a rewrite, manifest schema `1.1`
  (ADR-064).
- `warrantlib.CompletenessEvidence` (warrantlib 0.3.0) — the two predicates a `PROVED`
  completeness claim rests on, held once in a base, with a leaf per domain shape under
  it. `CompletenessCertificate` keeps its name, its fields and its rendering, and is now
  the tree leaf, `|A|^H` over one versioned action set. `ProductCompletenessCertificate`
  is the cross, carrying an `AxisDeclaration` per axis so 3 x 4 and 2 x 6 are not both
  reported as 12. A leaf defining its own `__post_init__` is refused at class creation,
  since it would shadow the gate. Evidence now crosses the wire through a tag/class
  registry, and a kind extends under the same `SCHEMA_VERSION` because `kind`
  self-describes and unknown-kind refusal is the designed failure.

- `warrantlib.manifest` (warrantlib 0.3.0) — the checks a suite is registered to report,
  declared in a generated TOML file and reconciled against what a run actually produced.
  A count could say a suite got shorter. It could not say which check left, and it was
  satisfied by a different check arriving in place of the one that went. The manifest
  names both directions: declared and not reported, reported and not declared. Generate
  or refresh one with `python -m warrantlib.manifest <path>`, and ask whether it is
  current with `--check`. The file compares as text rather
  than as parsed content, so a layout the writer no longer produces counts as stale.
  `--layout-only` asks that half on its own, from the ids the file already declares and
  without running a suite. Reading a suite's ids means running it, so on a project whose
  suites are symbolic the plain `--check` costs a full re-derivation to answer a question
  about whitespace.
  `--warrant-detail`, or `-vv`, prints each check's own line the way a suite run on its
  own does.
  The pytest plugin collects the manifest, turning every declared check into an item.
  The item exists because the check was declared, so a check that stops reporting still
  has a row and the row fails naming it.
- A pytest plugin, behind `pip install "warrantlib[pytest]"` (warrantlib 0.3.0). A test
  hands its findings to the run with the `record_check` fixture, and the run reports them
  in the warrant vocabulary: the progress letter separates a void check from one measured
  elsewhere, `-v` prints `NOT TRIGGERED` where it would have printed `PASSED`, and the run
  closes with the registered / tested here / fired accounting rather than a count of dots.
  A fired check fails its test, and so does an unresolved one, because a falsifier that
  ran and could not decide has not left the claim standing. The pytest outcome underneath
  stays one of pytest's own three: `junitxml` branches on that and nothing else, so a
  fourth value would write the row as no row at all. Categories stay standard too, so
  `N passed` still counts what it always did and `assert_outcomes` still sees it. Loaded
  through the `pytest11` entry point, so no configuration is needed; `dependencies` stays
  empty and importing `warrantlib` still pulls in no pytest, which a test asserts in a
  clean interpreter.
- `report_to_dict` / `report_from_dict` / `SCHEMA_VERSION` (warrantlib 0.3.0) — a
  report as a machine record and back. Every value is JSON-ready and the key order is
  fixed, so two runs of one suite produce the same bytes and a diff shows what changed
  rather than what moved. Enums travel as their values. Reading goes through the
  constructor, so a record naming `PROVED` with its evidence stripped still does not
  construct. A record from an unknown schema version, an evidence kind with no class
  behind it, or an enum value that has since been renamed is refused rather than mapped
  to whatever fits: `Tier.A` became `Tier.EXACT` once already, and a reader guessing its
  way through that rename would have reported a status change nobody made. Nothing here
  touches a filesystem. A JSON Schema for the record ships beside the code at
  `warrantlib/report.schema.json`, for a consumer reading a ledger without Python, and
  the suite validates the writer's own output against it.
- `CheckReport.check_id` (warrantlib 0.3.0) — the check as a key, beside the prose name
  it already carried. Dot-separated segments of letters, digits and underscores, refused
  at construction if anything else appears. The name is what a summary line reads and is
  reworded whenever the wording improves; the key is what a manifest declares before a
  run and what joins one run's report to the next. Deriving the key from the name would
  tie the two together, so the first rewording would read as one check dropped and one
  added. Required rather than defaulted: a report with no key is reconcilable with
  nothing, which is every job the field exists for. Every check in `research.checks` and
  in `examples/ffg/crossover.py` now declares one, and `NoiseFamily` carries the `key`
  those ids are built from.
- The fourth D3 falsifier is pre-registered on both axes and answered on both. Extension:
  `{−4,…,2}` gives `H* = 6` against a bar of `H* ≤ 6` registered a commit earlier, and
  extension saturates there. Refinement: `H* = 7` at step-`0.5` and step-`0.25` alike,
  `|ΔH*| = 0` against a bar of 1, the two cells agreeing to the digit across 86 times the
  policies. Each row is a `PROVED` report carrying its completeness certificates and a
  `Provenance` whose ordering git can check. The step-`0.5` cell was a re-measurement: the
  write-up published it before any commit built the set, and the numbers hold.
- `Provenance` (warrantlib 0.2.0) — which ref registered a claim and which one measured
  it, plus one line saying what a reviewer will find at the first. A `PROVED` report
  requires one, on the same terms as it requires evidence, and carries them as a tuple
  for the same reason the evidence is one. A ref is a git commit SHA, an http(s) URL or a
  DOI; a path, a branch, a tag and `HEAD` are refused, because each resolves to a
  different tree every time it is read. Where the two refs name one commit the render
  says the ordering is not established by history, which is the honest reading whenever a
  check and the derivation behind it land together. The type cannot order two refs, only
  compare them, so `tests/test_provenance_ordering.py` asks git. Three of the suite's nine
  sources fail that check today and are marked `xfail`: the hand derivation in
  `research/c4_hand_derivation.md` landed 2026-08-17, after the suites citing it, which
  ADR-037 already discloses (ADR-041).
- `warrantlib` — the vocabulary every check labels itself from, published as its own
  distribution and re-exported as `cpomdp.warrant` so existing import paths keep working.
  It depends on the standard library alone, so a suite can label its findings without a
  numerical stack; cpomdp depends on it and adds nothing to it (ADR-039). `Warrant` carries the
  prover class (`PROVED`, `CERTIFIED`, `CORROBORATED`), `Outcome` what a registered
  falsifier did, `Tier` what it was measured against (`EXACT`, `BOUNDED`, `COMPUTED`), and
  `CheckReport` all four with a reason. `CERTIFIED` is new: validated numerics prove a
  universal over a compact domain, which is stronger than a sample and weaker than a
  decision, and forcing it into either neighbour overclaims or throws away the bound
  (ADR-035). Every member is a word rather than a letter, so a row says what it means
  without a reader knowing the ordering. The prover sub-modes read the same way in prose:
  `Prover 3 · sample`, `Prover 3 · enumeration`, `Prover 3 · validated`. The canonical
  table is at `research/warrant_ledger.md`.
- `cpomdp.SymbolicReduction` — evidence for a `PROVED` claim decided by argument rather
  than by enumeration (Provers 1 and 2). A CAS establishes that one expression equals
  another and has nothing to say about whether those are the expressions the analytic
  claim is about, which the warrant ledger records as a human obligation. The type carries
  the claim, where the setup was hand derived against the problem, and the assumptions the
  identity is contingent on. Blank fields do not construct, so a `correspondence` nobody
  can fill honestly routes a check to `CORROBORATED` instead of passing silently. Blank
  means blank to a reader: whitespace, the zero-width formatting characters, a line break
  into a one-line render, and a field that is not text are all refused, with `assumptions`
  checked entry by entry.
- `check_summary` prints `n registered, m tested here, k fired`, then counts per
  `(warrant × outcome)`. Registering four falsifiers and testing two is a different claim
  from testing four, and one number cannot carry both.
- `cpomdp.diagnostics.logdet_pd` — `ln det` behind a Cholesky positive-definiteness guard,
  NaN otherwise. `slogdet` reports a sign separately, so reading only the magnitude returns
  `ln 2` for `diag(-1, -2)`, a plausible number for a matrix whose log-determinant does not
  exist. `epistemic_value` and the crossover demo's oracle both route through it.
- `examples/ffg/crossover.py` gains `flip_margin_error`, the error the already-declared
  `COND_CEILING` allows on a difference of two scores. `H* = 7` rested on a bare inequality
  between two computed floats with nothing said about how far apart they had to be. The
  bar is derived from a ceiling `tests/test_rollout_hygiene.py` already enforces rather
  than invented for the claim. `warrant_numbers.md` records the derivation, that it was
  written after the measurement, and why a stricter guard cannot rescue a failing result.
- A **References** page rendering the whole bibliography, `docs/references.md`.
  `mkdocs-bibtex` only emits the keys a page cites, so eleven entries in
  `docs/references.bib` were invisible on the site. Four sources join them: the ASME
  paper the applied surge work reads against (GT2024-124905), Greitzer 1976 for the
  compressor dynamics that notes disclaim modelling, and ISO 5167 parts 1 and 2, where
  flow-rate uncertainty for a differential-pressure element is standardised and where
  pulsating flow is explicitly excluded.
- Koudahl, Kouw and de Vries (2021) is cited wherever the fixed-sensor collapse is
  claimed, rather than ADR-003 alone. An ADR records a decision. The proof that expected
  free energy reduces to KL control under a fixed linear-Gaussian sensor is theirs, and
  the README, both example galleries and two API pages asserted it without saying so.
  Friston et al. (2015) gets the same treatment where the T-maze is named.

### Fixed

- `EFESelector` described varying-sequence search as a deferred v0.4 `GradientEFESelector`
  seam, in its rendered class docstring and in the `ValueError` a `p > 1` caller hits. That
  search shipped in v0.4.4 as `EnumeratedEfeSearch`, whose own docstring says it supports
  `p >= 1` and varying sequences (ADR-031). No such class has ever existed under `src/`.
  The name is still a planned continuous-action selector elsewhere, so only these two
  strings were wrong, not the name itself. The error path was the worse of the two: it
  told a caller to reimplement a search the package already had. Both now name
  `EnumeratedEfeSearch` and the `FiniteActionSet` it needs.
- The NumPy oracle in `examples/ffg/crossover.py` discarded `slogdet`'s sign on both halves
  of its per-step epistemic. v0.4.3 fixed the shipped kernel and the changelog did not say
  whether the oracle path went with it. It did not. Both paths now reject a matrix with an
  even number of negative eigenvalues, which a two-route agreement check cannot catch on
  its own: two routes that both discard the sign agree, and are both wrong.
- The crossover falsifiers computed their outcome from `|ΔG| > bound`, a magnitude test on
  a direction question. Nothing could emit `FIRED`, and a reversed argmin would have
  printed as a survivor beside a hardcoded "argmin is cue-ward". Outcome and detail are now
  computed from the exhaustive argmin, and the measurement is injectable so a refuting
  result is something a test can execute.
- `CompletenessCertificate(expected=9, visited=8, warrant=PROVED)` used to construct and
  read `complete = False`, a certificate recording its own shortfall with the contradiction
  one attribute access away. It now raises. The honest label for a partial enumeration is
  `CORROBORATED`.
- Citations rendered as literal `[^key]` on the published site. `mkdocs-bibtex` rewrites
  `[@key]` into footnote syntax, and `footnotes` was missing from `markdown_extensions`,
  so four API pages shipped a raw key where a reference should have been and no page
  carried a bibliography at all. Enabling the extension fixes every citation in the tree
  at once. No prose contained `[^` beforehand, so nothing existing changes meaning.

### Changed

- `cpomdp.diagnostics.condition_numbers` — the batched condition number
  `rollout_conditioning` computed inline, now a function of its own so a rollout and a
  scored cell read the same number for the same matrix.
- **Breaking:** a *fixed* `observation_model` or `dynamics_noise_model` must now restate
  the model's `observation_noise` / `dynamics_noise` exactly, and a sensor's linearized
  observation matrix must equal `observation_matrix`. `LinearGaussianModel` previously
  accepted the two disagreeing and left which one was read to the consumer: the EFE
  kernel reads the object where the filter reads the field, so a model could score
  against one noise and filter against another. A state-dependent object may still vary
  — it is the fixed case that restates the field and must restate it exactly. This is a
  construction-time break:
  `LinearGaussianModel(observation_matrix=[[1., 0.]], observation_noise=[[1.]],
  observation_model=FixedSensor([[1., 0.]], observation_noise=[[0.5]]), ...)` built
  before and now raises. Pass the same value twice, or drop the redundant field.
- The `symbolic` CI job reconciles the three symbolic suites against
  `research/registered_checks.toml` instead of comparing three hand-typed strings. The
  strings said `23 registered, 23 tested here, none fired` and two more like it. They
  could report that a suite got shorter. They could not say which check left, and a check
  renamed or swapped for another left the count untouched, so the gate passed. Every one
  of the 70 checks is now an item that fails by name, in both directions: declared and
  not reported, reported and not declared.
- **Breaking:** cpomdp and warrantlib require Python 3.11, up from 3.10. The check
  manifest is TOML so the generated file can say what it is and how to regenerate it, and
  `tomllib` reads TOML from the standard library only from 3.11. Taking `tomli` instead
  would have ended warrantlib's "the standard library is the only dependency", which is
  the property that makes it installable beside a check suite wanting nothing else.
  Python 3.10 reaches end of life in October 2026, which is what makes the trade cheap
  rather than what makes it right (ADR-045). The published warrantlib 0.2.0 keeps its
  `>=3.10` metadata, so an existing 3.10 install resolves as it always did.
- `cpomdp-research` requires `warrantlib>=0.3`, for `CheckReport.check_id`. cpomdp's own
  floor stays at `>=0.2`: nothing under `src/cpomdp` constructs a report, and moving the
  floor would re-arm ADR-040's publish-ordering race for no gain.
- The crossover falsifiers name their action mode. `H* = 7` is an open-loop number: the
  sweep scores whole length-H sequences with no re-planning, driving neither
  `RecedingHorizonSelector` nor `OpenLoopSelector`. The same statistic under a
  receding-horizon driver is a different quantity and is unmeasured. The declaration now
  travels with the number in the `PROVED` details, in `warrant_numbers.md`, in the ledger
  and in the README, and a test asserts it so the qualifier cannot be dropped silently.
- cpomdp requires `warrantlib>=0.2`, up from `>=0.1`. `cpomdp.warrant` re-exports
  `Provenance`, so an installed 0.1 fails at `import cpomdp`. Breaking for anyone pinning
  warrantlib 0.1.
- **Breaking:** the model's fields are `dynamics_matrix`, `control_matrix`,
  `observation_model` and `dynamics_noise_model` (were `dynamics`, `control`,
  `observation`, `process_noise`). One rule now covers every field: `_matrix` is a linear
  map, `_noise` a fixed covariance, `_model` the state-dependent stand-in for one of them.
  Before this the fixed process noise was `dynamics_noise` and the state-dependent one
  `process_noise`, the same channel under two roots, while A was `dynamics` and C was
  `observation_matrix`. Keyword-only construction is what made the names load-bearing:
  every argument after the first is typed out at each call site, so a member on its own
  scheme is a question the caller has to answer each time. `GaussianTransition.dynamics`
  and the `CouplingGraphBackend(control=...)` argument move with it, as do the demo
  constants (`SENSOR` is `OBSERVATION_MATRIX`, `PROCESS_NOISE` is `DYNAMICS_NOISE`).
  `LinearGaussianModel.observation_model` and `InferenceBackend.observation_model()` now
  share a name on two classes: the field holds the declaration, the method returns the
  `(C, R)` it resolves to. The `.A`/`.B`/`.C`/`.Q`/`.R` aliases are unchanged, and no
  numerical behaviour moved.
- **Breaking:** the epistemic term's aim is named `info_node` and `info_block`, not
  `target`. `target` carried two unrelated types on exported API: a desired value
  (`Float64[Array, "n"]`) on `StateGoal` and `ObservationGoal`, and the joint-state
  indices the epistemic term reads (`Sequence[int]`) on `FfgEfeSelector`,
  `policy_efe_ffg` and both `over_backend` class methods. A reader who learned the first
  from the Goal classes and passed a goal vector to the FFG API got `G = nan` with no
  raise, and `_argmin` maps nan to `+inf`, so a different action was selected in silence.
  The two are one address at two resolutions: `info_node` names a graph node,
  `backend.block(node)` returns the `info_block` of joint-state coordinates it occupies.
  `node` and `block` are the words the rest of the tree already uses for those two kinds.
  `ObservationGoal.info_target` becomes `info_node`. `StateGoal.target` and
  `ObservationGoal.target` keep the name, which now has one meaning.
- **Breaking:** `ObservationGoal` takes `action_bounds` by keyword. `StateGoal` already
  takes everything after its first argument by keyword, so this was the odd one out in a
  public pair a reader meets together. The two positional slots also transposed without
  raising for a 2-D increasing target, since `(lo, hi)` then satisfies the target's shape
  check and the target satisfies `lo < hi`. 24 call sites moved to keywords.
- **Breaking:** `FixedSensor` and `GaussianObservation` take `observation_noise` by
  keyword. The observation pair is (m, n) against (m, m), so the two are distinguishable
  by shape only when m != n, and a 1-D sensor is the case where they coincide. Only the
  noise is content-checked, and a square symmetric observation matrix passes that check,
  so a transposed pair built a sensor whose Jacobian was its noise. `CallableSensor` and
  `CallableGaussianObservation` are left alone: their arguments are a matrix, a callable
  and a pytree, so a swap raises rather than going quiet. 46 call sites moved to keywords.
- **Breaking:** `GaussianTransition` takes `dynamics_noise` by keyword, and
  `GaussianTransition.from_ou` takes `stationary_var` and `dt` by keyword. Both follow
  `LinearGaussianModel` for the same reason, and the transition factor is the worse of
  the two: its `dynamics_matrix` and `dynamics_noise` are both (n, n) at *every*
  dimension, so a transposed pair needs no square-shape coincidence to pass. Only the
  noise is content-checked. `from_ou`'s `tau` and `dt` are both positive scalars in the
  same time unit and `A = exp(-dt/tau)` is finite either way round. 45 call sites moved
  to keywords.
- **Breaking:** every `LinearGaussianModel` argument after `dynamics` is keyword-only.
  The four matrices are two maps and two covariances of the same rank, and only the
  covariances are content-checked, so a transposed pair constructed in silence whenever
  the maps were square and symmetric. `LinearGaussianModel(Q, C, A, R, prior)` built a
  model and returned a posterior of `[0.900, -0.444]` where the answer is
  `[0.896, -0.434]`. Detecting the mistake after the fact needs a rule for which of two
  same-shaped matrices is which. Naming them at the call site needs none. One positional
  construction existed in the tree, `examples/efe_collapse_figure.py`, now converted.
  `ty` reports the arity error statically, so a stale positional call fails the type gate
  rather than running.
- **Breaking:** `sensor_model` is now `observation_matrix`, on the same classes plus
  `_JointObservation` and `epistemic_value`. C is the *observation matrix* in the same
  literature that gives R its name, so the two halves of the measurement equation now
  read from one vocabulary. C is not `observation_model`: that name means the whole
  `(C, R)` channel, both on `InferenceBackend.observation_model()` and on the model field
  the entry above renames. The 2026-06-13 sketch wanted plain `observation` for C, which the state-
  dependent sensor field later claimed. The `.C` alias is unchanged, and no numerical
  behaviour moved.
- **Breaking:** `sensor_noise` is now `observation_noise`, on `LinearGaussianModel`,
  `FixedSensor`, `CallableSensor`, the FFG observation factors and `epistemic_value`.
  R is the *observation noise covariance* in the state-space literature this library
  writes in, and it is the noun in "state-dependent observation noise", the phrase the
  `R(x)` work is published under. The backends already said it. `observation_noise_at`
  has always been the state-dependent accessor, so the model was the one place still
  calling R a sensor quantity. The 2026-06-13 naming sketch in `DECISIONS.md` proposed
  `observation_noise` at the outset. The code drifted. The `.R` alias is unchanged, and
  no numerical behaviour moved.
- The repository is a uv workspace. cpomdp stays the root; `packages/warrantlib` and
  `research` are members, sharing one `uv.lock` and one `.venv`. `warrantlib` publishes to
  PyPI beside cpomdp. `cpomdp-research` never publishes, and holds the check suites along
  with the scipy and sympy they need, which have left cpomdp's `dev` group (ADR-039).
- `CompletenessCertificate` is defined in `warrantlib` beside `SymbolicReduction`, the
  other evidence kind, and re-exported from `cpomdp.enumeration` where an enumeration
  produces one. It used to live in `cpomdp.enumeration`, which made the warrant module
  import it at call time to run the `isinstance` guard behind `PROVED`.
- The check suites moved to `research/src/research/checks/`. The `research.checks.<module>`
  import path is unchanged, and so is every registered count.
- `SearchWarrant` is an alias of `Warrant`. Every call site keeps its members and its
  return type, so `EFESelector.warrant` still reads `CORROBORATED` and
  `EnumeratedEfeSearch.warrant` still reads `PROVED`.
- `PROVED` without evidence does not construct, enforced in `CheckReport`. Two kinds
  qualify, one per decisive prover: a completeness certificate for an exhaustive
  enumeration, a `SymbolicReduction` for a theorem or a symbolic identity (Provers 1 and
  2). Evidence is a tuple, so a claim quantified over several enumerations carries all
  their certificates rather than one of them. Every item is checked for its kind, so a
  path naming where the proof lives cannot satisfy the precondition by being present.
- A check that never ran carries no warrant. `CORROBORATED` asserts sampling-grade evidence
  was obtained, so attributing it to a falsifier void by construction claims evidence that
  does not exist. That cell reads `—`, enforced at construction.
- `examples/ffg/crossover.py --check` reports its falsifiers on two axes. Rows 1 and 2 read
  `PROVED` from the enumeration certificates and `BOUNDED` from the error bound. The `H* = 7`
  upper-bound qualifier now travels in both detail strings, not only in the write-up. The
  declared set clips the reach at `-2` while `-3` reaches the goal in one step. A margin inside its
  bound reports `NOT RESOLVED` rather than raising, since a tie is a finding about the
  measurement and an exception erases it.
- ADR-029 said the check vocabulary must never ship in `cpomdp`. ADR-035 reverses that on
  placement and extends it on content: three outcomes become five, `NOT RUN HERE` promoted
  from a prose rule to a member, `NOT RESOLVED` added for a genuine tie. `PASS` is absent
  from the enum. A falsifier does not pass, and printing `PASS` beside a prover column that
  disambiguates it inverts the rule the prover column exists to satisfy.

## [0.4.4] — 2026-08-04

The multi-step slice of expected free energy. Scoring one policy over a horizon was already
here; this release adds searching over a policy family and certifying that the search was
exhaustive, which is what the crossover measurement needed. The headline is `H* = 7`: the
horizon at which the best plan on the coupled-tree cue task stops being a direct reach and
becomes a two-phase sense-then-commit walk. Every horizon is a complete enumeration, so the
flip is decided rather than sampled.

Nothing on the H = 1 path moved. The existing suite passes unmodified and the one-step
arithmetic stays byte-identical to 0.4.3.

`examples/crossover_horizon_figure.py` puts the same crossover on the flat Kalman/EFE
route, with a frozen-`R` twin as its control. `examples/ffg/cue_maze.py` builds the cue
task for any number of dimensions. About 390 lines duplicated across eight scripts moved
into `examples/gallery.py`, and every committed asset re-renders byte-identical across the
move. Three example checks that could exit zero on a failure are now gated in
`tests/test_example_checks.py`. Two demos were removed.

### Added

- `cpomdp.enumeration` — the exhaustive search family, deliberately a separate object from
  `EFESelector`'s continuous grid so the two warrants cannot be confused. `FiniteActionSet`
  is versioned, so an action added after results are seen shows up in the diff rather than
  in the prose. `EnumeratedEfeSearch` enumerates `A^H` and returns the argmin policy with
  the full `G` vector. `CompletenessCertificate` asserts expected against visited and raises
  `IncompleteEnumerationError` on a mismatch, which is what earns the search a `PROVED`
  warrant where a grid earns only `CORROBORATED` (ADR-030, ADR-031).
- `EnumeratedEfeSearch.over_backend` scores enumerated policies on an FFG backend instead of
  a flat model. Scoring is injected as a strategy, so the enumeration, the certificate and
  the cost accounting are shared rather than duplicated. A reduce-to-flat oracle gates it:
  on a coupling-free backend it reproduces the flat search exactly (ADR-034).
- `RecedingHorizonSelector` and `OpenLoopSelector`, both `ActionSelector`s wrapping the
  enumerated search. They stay separate because the two modes genuinely differ, and a
  measurement has to declare which one it used. Both report `cost_per_plan = |A|^H · H` and
  a `replan_interval`. Neither reports the grid's `n_candidates × horizon`, which would
  under-count the real work by a factor of `|A|^(H−1)`.
- `policy_efe_trace`, returning a `PolicyEfeTrace`: the per-step `g`, pragmatic, epistemic,
  μ⁺, Σ⁺, Σ_post and S that the rollout scan already computed and then discarded. It sits
  beside the hot path rather than in it, so the selector stays allocation-free. Its sums
  equal the scalars `policy_efe` returns under `assert_array_equal`, which is the proof that
  it is the same arithmetic and not a second implementation.
- `policy_efe_ffg` and `policy_efe_ffg_trace`, the H-step rollout on a coupling graph, with
  the epistemic term aimed at a named node at the coupling-resolved μ⁺. Gated from both
  ends: at H = 1 it is byte-identical to the one-step FFG kernel, and with no coupling it
  matches `policy_efe` (ADR-032).
- `cpomdp.crossover` — `crossover_statistic`, `crossover_horizon` and `CrossoverStatistic`.
  The horizon aggregation is the symmetric between-policy contrast `ΔG = Δc − Δε`. At H = 1
  it collapses to the anchors `Δε = 1.7232`, `Δc = 4.4910`, `ΔG = +2.7678` nats, pinned at a
  tolerance of `1e-4` (ADR-033).
- `diagnostics.rollout_conditioning` — per-step `cond(Σ⁺)`, `cond(S)`, `cond(Σ_post)`, the
  minimum eigenvalue of the posterior covariance, and a positive-definiteness flag, computed
  host-side. Three `S` re-inversions under `R(x)` against a contracting Σ is where orders of
  magnitude go missing quietly. The eigenvalue bar is an assertion and not a clamp, because
  a clamp launders the failure it was meant to catch.
- `examples/ffg/crossover.py` — the crossover measurement, gated. It prints the exhaustive
  argmin per horizon, the mechanism split (disclosed as post-selection, because the pair it
  scores was found by the search), an independent NumPy kernel on the headline number, the
  counterfactual showing the flip is epistemically driven, and the conditioning against its
  registered bars.
- `examples/ffg/crossover_sweep.py` — the constant-action null. A constant walk overshoots
  the cue and never exploits, so the constant family cannot express the two-phase plan at
  all. That null is the reason the exhaustive search is necessary, which is why it ships as
  a check rather than as a footnote.
- A `slow` pytest marker for the exhaustive gate. Pull requests run without it; merges to
  main and releases run with it.

- `examples/crossover_horizon_figure.py` — a showcase of the epistemic/pragmatic crossover on
  an open plane, and the first demo of it on the *flat* Kalman/EFE route rather than the factor
  graph. An augmented state `x = (p, g)` carries a position the control moves and a static
  latent goal it does not. `R(p)` makes the channel that reads `g` sharp only near a beacon off
  the direct path. The animation runs the same world once per planning horizon: at short `H`
  the agent settles where its prior said without ever checking, and past a crossing it detours
  to the beacon, finds out it was wrong, and goes to the real goal. Nothing changes but `H`.
  The margin `ΔG(H) = G(detour, H) − G(direct, H)`, in nats, crosses zero exactly once, because
  the epistemic pull is flat while the pragmatic gradient decays under it. A frozen-R twin re-runs
  the whole sweep with `R` pinned and never crosses, so the behaviour is attributable to the
  state-dependent sensor and the horizon together rather than to the beacon-and-goal geometry.
  Under a fixed sensor the epistemic term is the same for both plans at every horizon, so it
  drops out of the margin. The term itself stays nonzero and grows with `H` (ADR-003). Renders
  the animation plus a three-panel companion. `--check` prints the sweep and asserts what the
  figures claim, gated in `tests/test_example_checks.py`. Its crossing lands at `H = 7`,
  which is the same integer as this release's registered `H*` and is unrelated to it: different
  model, different backend, whole-state epistemic rather than node-restricted, and no
  search at all. The demo says so in its own output, and neither number should be quoted
  as the other.
- `examples/ffg/cue_maze.py`. The cue-maze task, built for any `n_dims >= 1`. A hidden
  context decides which of two goals pays, the prior points at the wrong one, and a cue
  elsewhere in the arena reads the context sharply while the agent stands on it.
  `axis_action_set` grows as `2·n·|magnitudes| + 1` rather than as a product grid, so two
  dimensions cost exactly what the corridor did. `enumeration_cost` prices a sweep before
  you commit to it, and `best_reachable_noise` catches the failure that motivated it. A cue
  the action lattice cannot land on is one no policy can read, and the null that produces
  is indistinguishable from "information is never worth the detour". It is pure geometry.
- `examples/gallery.py`. The shared machinery the demos were each copying: three palettes,
  the GIF writer, the figure and still writers, the bacillus glyph as a style object, the
  covariance-ellipse geometry, the EFE sweep and decomposition panel, the action grid, and
  the two-route agreement report. Roughly 390 lines of duplication removed. Eight scripts
  touched. Every committed asset re-renders byte-identical to its pre-refactor baseline.
  `tests/test_horizon_dimensions.py` builds its arena from `cue_maze` rather than inlining
  a copy of it.
- `tests/test_cue_maze_parity.py`. `cue_maze.build_maze(1)` is claimed to reproduce
  `epistemic_dissociation_figure.build_backend(cue_x=CUE_DETOUR_X)` element for element, and
  the registered crossover is measured on the latter. The claim held on two files kept in
  sync by hand. Now it holds on dims, root, A, Q, B, the coupling and its noise, C, and
  `R(x)` sampled the length of the corridor rather than at one point. The frozen twins are
  excluded, because they genuinely differ and `cue_maze.py` says why.
- `mkdocs_hooks.py`. Expands `--8<--` snippet includes and rewrites the relative links
  inside them to site-local targets, so `examples/README.md` can carry links that work both
  on GitHub and on the published site without hard-coding either. `tests/test_docs_hooks.py`
  gates it, dead links included.

### Changed

- `EFESelector` now reports a `warrant` of `CORROBORATED`. It samples a continuum, so it can
  corroborate a universal over the action space but never decide one. The enumerated
  search's `PROVED` prints in different words for exactly that reason.
- `examples/ffg/crossover.py --check` reports the four registered falsifiers one line each,
  in three-valued outcomes, rather than closing on a single `PASS`. A falsifier that is void
  by construction did not pass a test, and one the gate skips now says so instead of
  borrowing the write-up's answer (ADR-029).

- The README's crossover section says what the horizon result is for. It now cites
  arXiv:2607.20306, which establishes that `R(x)` makes the epistemic term non-constant,
  and separates that from the question this repo answers: the horizon at which the term
  starts changing which plan gets picked. The answer given is the one-dimensional `H* = 7`,
  with its warrant stated (exhaustive over the declared `{0, ±1, ±2}`, completeness
  certificate, upper bound because the grid clips the reach), and the plane figure is
  labelled as the readable illustration rather than the proof. `examples/README.md` carries
  the same distinction at length.
- `examples/bacillus_uncertain_food.py` is gated. `check_backend_agreement` — `KalmanBackend`
  against the FFG `ChainBackend` on one scripted input sequence, `atol=1e-7` — now runs in
  `tests/test_example_checks.py`. The README's flagship had no test of any kind before this.
- `gallery.precision_field` samples the model's own `noise_fn` instead of re-deriving the
  well from its parameters, and takes the channel to read as a required argument. It had
  been asserting the sensor was a `CallableSensor` and then ignoring it, so on the flagship
  it drew the right channel by coincidence of how that model's `block_diag` is ordered. A
  demo with a different layout would have got a silently wrong figure that still passed the
  assert. One `vmap` replaces the `res²` Python-level calls, so the field is also cheaper to
  draw. The flagship's field agrees with the old computation to 2e-15, float noise from the
  `vmap` and nothing else, well under the width of a contour band.
- `gallery.print_two_route_agreement` is `check_two_route_agreement` and raises on a
  disagreement. It printed `FAIL` and returned, and neither caller was gated, so a broken
  equivalence would have exited zero. `coupling_graph_figure.check` and
  `chemotaxis_figure.check` are public and run in `tests/test_example_checks.py`.
- `DECISIONS.md` states at the top that a path inside an ADR is the path as of that ADR's
  date. `examples/bacillus_seeking_food.py` and `examples/bacillus_lqr.py` are named by
  ADR-008 and ADR-013 and no longer exist, which the append-only rule requires and which the
  file previously left the reader to work out.
- `mkdocs_hooks.py` skips a fenced block wherever it is indented, and skips inline code
  spans. Both were anchored at column zero, so a fence inside a numbered step, which is
  ordinary markdown, had its links rewritten *and* reported dead, and `mkdocs build --strict`
  aborted the docs deploy on a correct README edit. Reference definitions (`[label]: path`)
  are now repointed and dead-link checked too. They resolve to real anchors and previously
  escaped both, so the repository's layout shipped to the site. A four-space indented code
  block is still not recognised, and the module docstring says so and says why: the same
  pattern matches four-space list continuation, so closing it would trade a loud build
  failure for a silently wrong link. All four included sources rewrite byte-identically
  across the change.
- `cue_maze.enumeration_cost` takes `context_dim`. It hardcoded a joint width of
  `1 + 2·n_dims` while `build_maze` builds `context_dim + 2·n_dims`, and the estimate is
  quadratic in that width, so a two-wide context was priced 1.44x low and a four-wide one
  2.56x low, on top of the 1.6x floor the docstring already warns about, against a number
  whose whole job is to stop a sweep taking the machine down.
  `tests/test_horizon_dimensions.py` recovers the width the estimate charged for and pins
  it to the built backend's own `n_total` across five `(n_dims, context_dim)` pairs.
- The `slow` pytest marker is a wall-clock rule now, not a description of one test.
  Anything past `conftest.SLOW_TEST_SECONDS` (20s) carries it and runs on merge-to-main
  and release rather than on pull requests. `conftest.py` measures each run and prints
  any unmarked test that has drifted over, so the rule does not depend on anyone
  remembering it. One test crosses the line and is newly marked:
  `test_double_integrator_horizon.py::test_h2_choice_matches_brute_force_oracle` (~21s).
  Marks are decided on an isolated run on the owner's development machine, which
  `conftest.py` now records as the reference. Wall clock is machine-relative, and without
  one the same suite marks differently per reviewer. Pull-request coverage is unaffected at
  93.8% against the 80% gate.

### Removed

- The `## The journey` section of `examples/README.md`, and with it the gallery entries for
  `bacillus_seeking_food.py`, `bacillus_lqr.py`, `efe_collapse_figure.py` and
  `internal_noise_figure.py`. `efe_collapse_figure.py` and `internal_noise_figure.py` stay,
  keep their `check()` gates, and keep their figures, which `DECISIONS.md` still cites.
  `docs/assets/bacillus.gif` and `docs/assets/bacillus_lqr.gif` had no other referrer and go
  with the section, 4.2 MB of them.
- `examples/bacillus_lqr.py`. Every claim it demonstrated is asserted elsewhere and more
  strongly: the LQR reduction at exact array equality in
  `tests/test_agent.py::test_state_goal_path_is_byte_identical_to_lqr`, and the epistemic
  collapse three times over in `tests/test_efe.py`, `tests/test_theorem.py` and
  `efe_collapse_figure.py`'s gated `check()`. Nothing imported it.
- `examples/bacillus_seeking_food.py`. It had stopped being a demo and become an undeclared
  dependency: `bacillus_uncertain_food.py` imported its beacon falloff, its palette, its
  glyph and its precision field at module scope, so the README's flagship could not run
  without it and no test would have said so. The shared parts are now `gallery.beacon_noise`,
  `gallery.precision_field` and `gallery.SMALL_BACILLUS`; the beacon geometry moved into the
  flagship, which is the only thing that used it. The flagship's GIF re-renders byte-identical
  to the committed one across the move. Its own claims were already asserted in
  `tests/test_agent.py`, `tests/test_efe_selector.py` and `tests/test_efe.py`, whose
  `TestPrecisionControlsBalance` was written for it. ADR-008 and ADR-013 still name the path;
  `DECISIONS.md` is append-only, so those references stand as the historical record.

### Deferred and unsupported

Named here so each boundary is a statement rather than an omission. Each has an issue.

- A closed-loop `FfgEfeSelector` above H = 1 stays unsupported, not untested. The rollout it
  would need now exists and the FFG-backed drivers would make it nearly free, but nothing in
  scope uses it. (#52)
- No pruning of the `|A|^H` search. A pruner that cannot prove what it discarded would
  destroy the completeness certificate, which is the only reason the search is worth having,
  so it waits for a design that starts at the certificate. (#53)
- No continuous or gradient search over varying sequences. This one is a deliberate wall:
  gradient ascent on a non-convex objective corroborates and cannot decide, so letting it
  near the enumerated evidence would downgrade the certificate that evidence exists to earn.
  It is registered as a separate, separately-labelled track. (#54)
- `Q(x)` above H = 1 remains undocumented and unclaimed. It may work; nothing says so and
  nothing tests it. (#56)

## [0.4.3] — 2026-07-27

Closes the code-side divergences found by auditing the library against the paper written
about it (`PAPER_DIVERGENCE.md`). The paper's own quoted numbers are unchanged and both
demonstrations' `--check` gates still pass; the paper-side items are tracked separately.

### Added

- `cpomdp.diagnostics` — the conditions a state-dependent sensor has to meet, asked over
  the states an action can actually reach. `probe_model(model, belief, actions)` predicts
  under each candidate action and reports a `SensorReport`: whether `C` has full row rank,
  whether `R` stays positive definite, whether `R` moves at all across the reachable
  means, and whether the epistemic value moves with it. `is_positive_definite`,
  `epistemic_value` and `loewner_order` are exposed alongside for use on their own.
  `probe_model` and `SensorReport` are re-exported at the top level. A declared `R(x)`
  that no action can move now has something that says so.
- `tests/test_theorem.py` — one test per published claim: the pinning result and its
  rank-deficient counterexample, the dual effect on both the posterior covariance and the
  gain, the impossibility of a fixed noise schedule, the scalar monotonicity, the
  zero-innovation realisation, the planning-reduction equivalence, the level-set
  construction where the covariance moves but the epistemic value does not, and the
  worked one-step numbers (gains 2/3 and 2/7, ½ln3 and ½ln(7/5)).
- `tests/test_diagnostics.py` covering the new module.
- `examples/efe_collapse_figure.py --check` now also asserts the dual effect directly —
  the posterior variance and the Kalman gain moving across the action grid — rather than
  only the epistemic scalar that rides on it.

### Fixed

- **The FFG epistemic term had no positive-definiteness guard.** `_efe_step` mapped a
  non-positive-definite `R` to NaN so it would lose the selector's argmin;
  `_state_info_gain`, the term the coupling-graph path actually scores, discarded the
  determinant signs and returned a finite, plausible-but-wrong value that could *win*.
  Both now share one `_logdet_pd` helper, which tests positive definiteness by Cholesky
  rather than by determinant sign — the sign passes any matrix with an even number of
  negative eigenvalues, `diag(-1, -2)` among them.
- **`CouplingGraphBackend.model` handed out a frozen `R`.** For a state-dependent graph
  the flat model carried the noise evaluated once at a representative state, so anything
  filtering with it silently used the wrong noise at every step. The model now carries a
  `_JointObservation` that linearizes each node's `R` at that node's slice of the mean,
  the same point the backend's own filter uses.
- **`Agent` could not see a graph's state-dependence.** Because that flat model declared
  no observation model, a coupled `R(x)` backend read as a fixed sensor and a `StateGoal`
  was accepted onto it, silently selecting actions with the certainty-equivalent LQR
  controller. It now raises, as it always did on the flat path.
- **The theorem's own model class could not be flattened.** A single chain with no
  couplings raised `NotImplementedError`; it now emits a faithful state-dependent flat
  model, verified to reproduce the native filter to ~1e-16. It is a state-dependent
  model, not a fixed one — no fixed one exists.
- `examples/ffg/epistemic_dissociation_figure.py` numbered its first two results the
  opposite way round to the write-up, and printed them out of order. Result 4 asserted
  only that the *best* action B rates below the pragmatic-only agent is cue-ward; it now
  asserts the claim actually made, that *every* such action is.

### Changed

- The `IncompatibleLinearizationError` message gains a closing sentence: the guard reads
  the sensor's *declared* state-dependence, and a declared `R(x)` that ignores the state
  is constant in fact and does flatten. The original two sentences are unchanged, so the
  message is now a strict extension of what came before. Whether `R` really varies over
  reachable means is what `cpomdp.diagnostics.probe_model` is for.
- Trimmed prose in `src/` that had gone stale: references to an `rfcs/` directory that
  does not exist, sensors and selectors described as arriving in a release that has
  shipped, and the unqualified claim that the predicted covariance is action-independent
  — true of a single step, false over a horizon under `R(x)` or `Q(x)`.

## [0.4.2] — 2026-07-18

Archival release accompanying the paper. No library code changed — the public API and
numerics are identical to 0.4.1. The changes are in the example check suite, which the
paper's reproducibility claims quote, and are now gated in CI.

### Added

- A single-chain theorem-instance check on the EFE-collapse demo
  (`examples/efe_collapse_figure.py --check`): the model class of the paper's section
  3.1 — one node, no couplings, additive control, a state-dependent `R(x)` sensor — run
  through the flat `KalmanBackend(CallableSensor)` route, asserting Theorem 1 clauses (i)
  and (iii): the one-step epistemic value varies across the action grid, its frozen-`R`
  twin is flat (the ADR-003 collapse), and `R(μ⁻)` traces a curve. Runs with no plotting
  deps.
- `tests/test_example_checks.py` runs both demos' `check()` gates in CI, so their
  paper-facing assertions fire on every test run rather than only when the script is run
  by hand.

### Changed

- Strengthened Result 4 of the epistemic-dissociation demo
  (`examples/ffg/epistemic_dissociation_figure.py --check`): `_boundary_scan` now also
  returns the pragmatic term, and Result 4 prints and asserts the horizon-1 ordering
  `epistemic pull < pragmatic gradient` — the "reaches without walking" claim, previously
  an unasserted diagnostic number, is now a checked fact.

## [0.4.1] — 2026-07-13

Maintenance release cut for the paper-1 Zenodo archive. No library code changed — the
public API and numerics are identical to 0.4.0.

### Changed

- Reworked the epistemic-dissociation flagship figure
  (`examples/ffg/epistemic_dissociation_figure.py`): renamed the candidate action to `u`
  to stop it clashing with "Agent A", moved the `R(mu+u)` label into a clean gap, and
  added paper-facing assertions that pin Result 2 at the geometry where only B's edge
  over A survives. Regenerated `docs/assets/epistemic_dissociation_boundary.png`.

## [0.4.0] — 2026-07-06

Forney factor-graph (FFG) message passing — the continuous-state generalisation of the
Kalman/EFE path to a branching model the chain cannot draw cleanly (ADR-012, ADR-014).
The chain is the degenerate case and stays validated byte-numerically against the Kalman
filter.

### Added

- FFG message passing in canonical/information form, owned from-scratch in JAX, reachable
  under `cpomdp.ffg.*` (the model-construction symbols are also re-exported at the top
  level — see the public-surface entry below, ADR-021):
  - `CanonicalGaussian` (`cpomdp.ffg.message`) — the `(Λ, h)` message payload; factor
    product is addition, marginalisation a Schur complement, moment form a readout view.
  - Tier-1 linear-Gaussian factor nodes (`cpomdp.ffg.factors.linear_gaussian`):
    `GaussianObservation`, `GaussianTransition`, `GaussianCoupling`.
  - `ChainBackend` (`cpomdp.ffg.chain`) — an `InferenceBackend` over a linear chain,
    numerically identical to `KalmanBackend` (atol 1e-7, the keystone gate), including
    state-dependent `R(x)`/`Q(x)` parity.
  - `CouplingGraph` (`cpomdp.ffg.graph`) — the v0.4 core representation: integer-indexed
    nodes coupled into a rooted tree with a shared node of degree > 2, inferred by a
    hand-authored deepest-first tree-collect schedule (only the root crosses moment form,
    so the inversion cost is per-root, not per-node). `levels()` is deferred (ADR-015).
  - `CouplingGraphBackend` (`cpomdp.backends.coupling`) — the branching FFG as a recursive
    Gaussian filter satisfying `InferenceBackend` (issue #25): driven-relaxation
    composition (each node's own dynamics + its structural parent's drive every slice,
    ADR-017), the exact full-joint carry (ADR-016), `marginal`/`readout` for any node, and
    `to_flat_model` (structural couplings as within-slice pseudo-observations). Validated
    at atol 1e-7 against an independent NumPy joint-precision oracle, `KalmanBackend`, and
    `RxInferBackend` on the flattened model.
  - Carry partition on `CouplingGraphBackend` (ADR-016): a `partition` of the node set
    controls which between-cluster correlations survive the time boundary — `[[all]]`
    (the default) keeps the exact full joint, singletons fully factor it. The carry
    factors the joint *covariance* (not the precision), so every node's marginal stays
    exact within a slice (ADR-017); only the cross-cluster correlation carried forward is
    dropped. A pure `partition_error` reports the severed mass (a covariance magnitude,
    not a rate), and `rollout` profiles it over a whole sequence in one traced `lax.scan`
    pass. A fully-factored (singleton) partition runs cheap two-pass tree belief
    propagation (`CouplingGraph.infer_all`, plus `GaussianCoupling.message_to_child` and
    `CanonicalGaussian.__sub__`) in place of the dense joint solve — O(tree) rather than
    O(n³), matching the dense path to atol 1e-7. The exact endpoint stays byte-identical
    under `[[all]]`.
- State-dependent sensing `R(x)` on the branching backend (issue #27, ADR-019): a
  `CallableGaussianObservation` factor and a two-pass linearisation that reads `R` at the
  coupling-resolved predictive mean μ⁺ — the dual effect (`observation_noise_at`,
  `predicted_belief`).
- `FfgEfeSelector` scores the per-candidate epistemic term over the FFG, so an `Agent` with
  an `ObservationGoal(info_target=…)` seeks information through a branch. `severed_efe_edges`
  / `InadmissiblePartitionError` reject a carry partition that would sever an EFE-relevant
  edge (ADR-018).
- `IncompatibleLinearizationError` (`cpomdp.backends.coupling`): `to_flat_model` raises it
  when a coupled `R(x)` model is asked to flatten. A mean-shifting coupling makes μ⁺ ≠ μ⁻, so
  no fixed linear-Gaussian model reproduces `R(μ⁺)` (ADR-019, ADR-020).
- Public surface (ADR-021): the branching construction symbols (`CouplingGraph`, `Coupling`,
  `CouplingGraphBackend`, `IncompatibleLinearizationError`, the four Gaussian factors) and the
  selector family (`ActionSelector`, `EFESelector`, `FfgEfeSelector`, `LQRSelector`) now sit in
  the top-level `cpomdp` namespace. `FfgEfeSelector` gains `EFESelector`'s `n_candidates` /
  `horizon` / `cost_per_cycle`.
- Examples: `epistemic_dissociation_figure.py`, the v0.4 flagship — two agents on one maze,
  differing only in a state-dependent vs. fixed cue sensor; the `R(x)` agent resolves the
  hidden context through a branch and crosses to the reward, the fixed-sensor agent collapses
  to LQR, and the `R(x)` model can't be flattened (`IncompatibleLinearizationError`).
  `bacillus_uncertain_food.py`, the instrumental-epistemics demo — the beacon resolves an
  explicit food *latent* rather than the agent's own position (ADR-013), on both
  `KalmanBackend` and `ChainBackend`. `coupling_graph_figure.py`, the difference demo — a
  branching tree resolved natively vs. the hand-flattened joint precision.
  `chemotaxis_figure.py`, the same on a real branching network (the E. coli chemotaxis
  pathway's shape, native FFG vs flattened Kalman) — illustrative only, no biophysics
  (ADR-020).

### Validation

- RxInfer oracle (behind the `rxinfer` marker) extended to the branching tree, alongside
  the existing chain checks; jit/grad/vmap smoke tests gate every new public inference
  entry point.

## [0.3.0] — 2026-06-22

Epistemic action selection. v0.2 could perceive and pursue a goal; v0.3 lets the agent *seek information* — it minimises Expected Free Energy `G = pragmatic − epistemic`, so it will detour to where its sensor is sharp, localise, then act. Under a fixed sensor this collapses exactly to the v0.2 LQR behaviour (ADR-003), so that path is unchanged.

### Added

- Expected Free Energy action selection: `expected_free_energy` (the one-step kernel) and `EFESelector` (a front-loaded candidate grid, `argmin G`). Observation-space cross-entropy pragmatic minus state info-gain epistemic (ADR-005).
- State-dependent sensing and dynamics: `CallableSensor` for `R(x)` and `CallableProcessNoise` for `Q(x)`, honoured in *both* the planner and the Kalman filter (ADR-006, ADR-008). The fixed-matrix path stays byte-identical.
- Typed objectives: `StateGoal` (state-space → LQR) and `ObservationGoal` (observation-space → EFE). The `Agent` dispatches on the objective's type, so an illegal pairing is unrepresentable rather than a runtime check (ADR-007).
- H-step horizon: `EFESelector(horizon=H)` scores constant-action policies over an H-step rollout (default `H=1`), making delayed consequences visible — e.g. acting on velocity to steer an observed position (ADR-009).
- `ModelStructure`: optional, declarable factor / Markov-blanket / sensory-channel metadata on a model, with inspection and an opt-in, experimental `validate()` that checks the declaration against the matrix sparsity (ADR-010).
- Public building blocks for the above: `ObservationModel`, `FixedSensor`, `DynamicsNoise`.

### Changed

- **Breaking (pre-1.0):** `Agent` takes a typed objective instead of `goal=`/`goal_precision=` keywords. `Agent(model, goal=[1.0, 0.0])` becomes `Agent(model, StateGoal([1.0, 0.0]))`; observation-seeking uses `ObservationGoal(...)` (ADR-007).

## [0.2.0] — 2026-06-16

The array backend moves from NumPy to JAX (ADR-004). v0.1 was a proof of concept; this is the groundwork for the autodiff and batching v0.3 is aiming at.

### Changed

- The core runs on `jax.numpy`. `Belief` and `LinearGaussianModel` now hold `jax.Array`s, and the Kalman filter and LQR are pure `jnp`. If you were reaching past the public API and expecting `numpy.ndarray` off `belief.mean`, you'll get a `jax.Array` now — both still hand off to NumPy, so most code won't notice.
- Importing `cpomdp` switches JAX into float64 mode (`jax_enable_x64`) process-wide. The library is validated to 1e-9 against the RxInfer oracle and JAX defaults to float32, so this keeps the numbers right — but it does change float behaviour for any other JAX code in the same process.

### Added

- `Belief` and `LinearGaussianModel` are registered JAX pytrees, so they flow through `jit`, `vmap`, and `grad` as data.
- The Kalman step is split into pure, `jit`-compiled kernels — one filter step now `vmap`s over a batch of beliefs.

### Dependencies

- Added `jax` and `jaxtyping`. NumPy stays: JAX pulls it in anyway, and the RxInfer backend still hands real NumPy arrays across the Julia bridge.

## [0.1.1] — 2026-06-15

A metadata-only re-release, functionally identical to 0.1.0. The 0.1.0 release has
been removed from PyPI, so use 0.1.1.

### Changed

- Trimmed the author entry in the package metadata.
- README: the DECISIONS.md link is now an absolute URL, so it resolves on PyPI.

## [0.1.0] — 2026-06-15

The first cut. Linear-Gaussian active inference, end to end: perceive with a Kalman filter, act with LQR, all behind a pymdp-style `Agent`.

### Added

- `Agent`, the stateful façade you actually drive. `infer_states` to perceive, `sample_action` to act, the same loop pymdp users know. It remembers the last action it took, so you don't have to thread that back in by hand. Build it without a goal and it's a pure tracker that perceives but won't act.
- `LinearGaussianModel`, the generative model. Matrices are named for their role (`dynamics`, `control`, `sensor_model`, `dynamics_noise`, `sensor_noise`), with the control-theory letters (`A`/`B`/`C`/`Q`/`R`) kept as aliases for when you're reading the maths.
- `Belief`, an immutable Gaussian belief: a mean and a covariance, validated on the way in.
- `KalmanBackend`, exact Kalman filtering. Has an optional steady-state mode that solves the gain once up front and reuses it.
- `LQRController`, steady-state LQR action selection. For a linear-Gaussian sensor this *is* the expected-free-energy-optimal action rather than a stand-in for it (the why is in DECISIONS.md, ADR-003).
- `RxInferBackend`, an optional [RxInfer](https://github.com/ReactiveBayes/RxInfer.jl) (Julia) backend. It re-derives the same filtering results through completely separate machinery and exists as a correctness oracle for the native path. Lives behind the `rxinfer` extra so the core install stays Julia-free.
- `InferenceBackend`, the protocol the backends satisfy, so you can drop in your own engine.

This is pre-alpha. The API works and is tested against the RxInfer oracle, but it can still move before 1.0.
