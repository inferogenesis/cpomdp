# The Spinello–Stilwell rung: what the paper leaves open

Rung two of PR-7's evaluation ladder is the iterated extended Kalman filter for
state-dependent observation noise, Spinello and Stilwell, *IEEE Transactions on
Automatic Control* 55(6), 1358-1366, 2010. This file records what building it requires
deciding that the paper does not decide, and what has to be measured before those
decisions are made. ADR-056 carries the decisions that are settled and points here for
the rest. The derivation of (35c) to (35e) that the questions below cite by equation
number was worked on paper first: `research/spinello_stilwell_hand_derivation.pdf` is
the scan of that notebook, and `research/spinello_stilwell_hand_derivation.md` is the
scan typed up.

Nothing below is a result. These are routes and tests, written before the rung exists,
so that a choice made later can be read against what was known when.

## Notation: a complete three-way swap

The paper's symbols collide with cpomdp's, and not one of the three means the same
thing in both. Transcribing an equation without renaming is the obvious way to produce
a filter that runs and is wrong.

| paper | means | cpomdp |
| --- | --- | --- |
| `Σ(x)` | observation-noise covariance, state-dependent | `observation_noise`, `R(x)` |
| `P[k\|k−1]`, `P[k\|k]` | belief covariance, predicted and posterior | `Belief.cov`, `Σ` |
| `R⁽ᶥ⁾` | the Gauss–Newton approximate Hessian | no equivalent. Name it `gn_hessian` |
| `U`, `Ū` | Fisher information contribution | no equivalent |
| `h` | observation mean function | `observation_matrix @ x` in the linear case |
| `ζ` | innovation `z − h(x)` | the residual |

The paper's `R` must never be spelled `R` in cpomdp source. It is the Hessian of a cost,
not a noise covariance, and the tree already uses `R` for the thing the paper calls `Σ`.

The paper assumes `p ≤ n` throughout, no more observation channels than states. cpomdp
does not enforce this anywhere. The rung should assert it rather than inherit it
silently.

## The scheme, scalar case (§III-D-2, `p = 1`)

```
x̂⁽ᶥ⁺¹⁾ = x̂⁽ᶥ⁾ − [P⁻¹ + R⁽ᶥ⁾]⁻¹ ( P⁻¹(x̂⁽ᶥ⁾ − x̂⁻) + s⁽ᶥ⁾ )        (35a)

s⁽ᶥ⁾ = −(ζ/σ)∇h + (1/2σ)(1 − ζ²/σ)∇σ                            (35c)
R⁽ᶥ⁾ = (1/σ)∇h∇h + (ζ/2σ²)(∇h∇σ + ∇σ∇h)
       + (1/4σ²)(ζ²/σ + 1/ln σ)∇σ∇σ                              (35d)
Ū    = (1/σ)∇h∇h + (1/2σ²)∇σ∇σ                                   (35e)
P⁻¹[k|k] = P⁻¹[k|k−1] + Ū                                        (35b)
```

Two matrices, two jobs. `R⁽ᶥ⁾` shapes the step. `Ū`, the Fisher information, sets the
posterior covariance, evaluated at the converged estimate per (33e). Conflating them
gives a filter with a plausible mean and a wrong covariance, which the ladder cannot
detect by inspection because the ladder is a comparison of posteriors.

## Q1. `1/ln σ` is the only unit-dependent object in the estimator

Rescale the observation, `o → λo`. Then `z → λz`, `h → λh`, `σ → λ²σ`, `ζ → λζ`,
`∇h → λ∇h`, `∇σ → λ²∇σ`. Every term above is invariant under that substitution except
one:

```
1/ln σ   →   1/(ln σ + 2 ln λ)
```

Verified symbolically: seven of the eight terms in `s`, `R⁽ᶥ⁾` and `Ū` return to
themselves, and only the log-determinant term moves.

The quantity the ladder reports is `E_p*[ KL(q ‖ p(x|y)) ]`, a divergence between
distributions over the *state*. It does not depend on the units the observation is
measured in. So this term makes the estimator depend on a choice the reported number
cannot see, which is the same defect the standing prohibition on entropy subtraction
guards against and belongs in the same family of arguments.

That is the warrant for dropping the term, if it is dropped: not that it changes the
path rather than the fixed point, but that it is the sole place an arbitrary unit
choice enters an estimator whose output is unit-free.

**Route.** An exploration that rescales one worked case over several decades of `λ` and
reports which quantities move. **Test.** The converged estimate and the reported gap are
invariant under rescaling once the term is removed, and are not before.

### AMENDMENT 2026-09-04: the reported estimate moves at a finite budget, not at the fixed point

The Route entry of 2026-08-24 measured the converged estimate as unmoved by rescaling.
The test this question asks for was still written as though the printed scheme fails it,
which cannot both be true. `research/src/research/spinello_stilwell/repair.py` settles
which.

Run to convergence, the printed scheme spans `0.0` across `λ ∈ {1, ½, 3, 7}` and so does
the modification. Neither is unit-dependent there. At a budget of one the printed
estimate spans `2.955e-05` over the same four choices and the modification spans
`1.11e-16`.

So the test above is right that a unit-dependent report is the defect and wrong about
where it appears. It appears at a finite budget, and a budget of one is the single-step
filter (36) that ADR-056 declares as its own rung. The consequence is a strengthening
rather than a retraction: the term makes a rung the ladder **reports** unit-dependent,
which is a worse fault than making an iteration path unit-dependent.

What this does not settle. The gap itself has still never been computed through a rung,
so the test as this question words it, on the reported gap rather than on the estimate,
remains unrun. Routes 3, 5, 6 and 7 are untouched.

### RESOLVED 2026-09-12: route 1's empirical half is run, and the test Q1 asks for holds on the gap

`research.spinello_stilwell.gap_rescaling` runs the test this question words, on the
quantity the ladder reports and not on the estimate: `E_p*[KL(q ‖ p(x|y))]` over the
symbolic half's worked case, prior `N(1, 0.04)` under `R(x) = 1 + x²`, at
`λ ∈ {1, ½, 3, 7}`, on the lattice `T` is registered under. The printed scheme and the
modification each run at a budget of one, rung (36), and at the rung's declared 64 at
`1e−12`, rung (35). The exact posterior and the observation box stay in native units,
since the divergence is over the state, and `λ` reaches only the scheme.

| scheme | budget | `λ = 1` | `λ = ½` | `λ = 3` | `λ = 7` |
| --- | --- | --- | --- | --- | --- |
| printed | 1 | 8.565375e−04 | 8.946782e−04 | 8.673353e−04 | 8.688809e−04 |
| printed | 64 | 8.470280e−04 | 8.470280e−04 † | 8.470280e−04 | 8.470280e−04 |
| modified | 1 | 8.716589e−04 | 8.716589e−04 | 8.716589e−04 | 8.716589e−04 |
| modified | 64 | 8.470280e−04 | 8.470280e−04 | 8.470280e−04 | 8.470280e−04 |

† declined `4.6e−12` of the predictive weight. At this unit choice the pole sits below
the operating point, and a few far readings spend the budget against it. The value is
the conditional reading ADR-060 defines.

**The modification is unit-free on the gap.** Relative spread across the four unit
choices `3.7e−15` at budget one and `1.3e−14` at 64. Both at roundoff.

**The printed scheme is not, at rung (36).** Relative spread `4.4e−2`. The reported gap
of the single-step rung moves by four per cent with the observation's units, and it
reads below the modification at `λ = 1` and above it at `λ = ½`. Run to the declared
budget it converges to the modification's fixed point wherever it converges, to
`8e−14` relative. That is the AMENDMENT of 2026-09-04 read on the gap: the term moves a
reported number at a finite budget and nothing at the fixed point.

**The scalar scheme is the rung.** At `λ = 1` the modification through the scalar
transcription agrees with `SINGLE_STEP_RUNG` to `2.5e−16` relative and with
`ITERATED_RUNG` to `0.0`, on the same grids. That is the one check on the probe's
wrapper that does not go through the wrapper.

**What the modification costs at rung (36), in native units.** `−1.51e−05` nats, the
printed scheme reading `1.7%` below the modification at `λ = 1`. The figure is a
property of the unit choice and not of the scheme, since at `λ = ½` the sign reverses.
That reversal is the reason the term left.

Both halves of the test are now run and both hold.

## Q2. Ask whether the pole is a units artefact before writing any guard

`R⁽ᶥ⁾` is singular at `σ = 1`. Under rescaling the pole sits at `σ = λ⁻²`, so the unit
choice alone decides where it falls: `λ = 2` moves it to `σ = 1/4`, `λ = ½` to `σ = 4`.

Since the fixed point of (35a) is where `P⁻¹(x̂ − x̂⁻) + s = 0`, and `s` is invariant,
rescaling moves the pole without moving the answer. Choosing observation units that put
the whole reachable `σ` on one side of 1 therefore removes the hazard by construction
and costs nothing.

That is a declared unit choice, not a tuned parameter. It must still be declared, and
before any number exists, because an undeclared choice that happens to avoid a pole is
exactly what standing rule 2 exists to catch.

A guard and its instrumentation become necessary only where no such choice exists, on
a family whose reachable `σ` straddles every achievable pole.

**Route.** Determine, for each declared `R` family and its swept range, whether a single
`λ` puts the whole reachable set on `σ > 1`. On the registered ridge
`R(x) = R₀ + κx²` at `μ* = √(R₀/κ)`, `σ(μ*) = 2R₀`, so `R₀ = 0.5` puts the pole on the
operating point at `λ = 1`. **Test.** The declared `λ` keeps `σ > 1` across the swept
range, asserted rather than assumed.

### RESOLVED 2026-09-04: no unit choice is declared, because the shipped rung has no pole

ADR-057 ships rungs (36) and (35) with the `r₃` block of (35d) removed, and that block
is the only term a rescaling moves. So the shipped scheme has no pole, and a `λ`
declared to keep `σ > 1` would change nothing the ladder reports. Standing rule 2
catches an undeclared choice. An absent one is checked instead, by a rescaling
invariance test on the shipped rung.

Route 2's table is kept as a measurement: five families had a units-only repair
available and `exp(x)` did not, which is what made the units answer insufficient on
its own. The printed scheme still runs in `research/` at native units, where routes 3,
5 and 7 need the pole reachable. No guard is written.

## Q3. The failure is two-sided, and one side is silent

**From above.** As `σ → 1⁺`, `1/ln σ → +∞`, so `R⁽ᶥ⁾ → +∞` and the step
`[P⁻¹ + R⁽ᶥ⁾]⁻¹(…) → 0`. The iteration freezes. Under a fixed budget it returns
`x̂[k|k−1]`, the prediction, wearing the appearance of a converged posterior.

**From below.** For `σ < 1`, `1/ln σ < 0`. `P⁻¹ + R⁽ᶥ⁾` passes through singular and
becomes indefinite, giving unbounded steps and steps that climb the objective.

On the ridge with `R₀ < 1` a single run crosses from one regime to the other.

The stall is the severity-one case: it produces a number, the number is wrong, and
nothing about it looks wrong. The blow-up at least announces itself.

### RESOLVED 2026-09-04: both sides of the failure leave with the block

Under ADR-057 the shipped curvature is `(1/σ)bᵀb`, a real square, so
`P⁻¹ + R_mod` is positive definite for every `σ > 0`. Neither the freeze from above nor
the indefiniteness from below can occur. Both stay true of the printed scheme and are
route 3's to measure in `research/`, where that scheme still runs.

### ROUTE 2026-09-04: route 3 is run, and the silent side is quieter than stated

`research.spinello_stilwell.pole_failure`, run with

```text
uv run --no-sync python -m research.spinello_stilwell.pole_failure
```

Both sides are now measured on the printed scheme. Every number below came out of a
probe at a budget of 200 and a tolerance of `1e-14`.

**The pole itself has no value.** `gauss_newton_curvature(1.0, …)` raises
`ZeroDivisionError`, so the printed curvature is not evaluable at `σ = 1` rather than
merely large there. The modification returns `2.56` on the same arguments.

**From above, the curvature grows as `0.25/offset`** and the step falls with it. The
block is `R'²/(4R² ln R)` and `ln(1 + offset)` is `offset` to first order, so at the
prediction, where `R' = 1`, the block is `1/(4 offset)`. The first step is the gradient
there, `-1.42`, divided by it: `-5.68 × offset`. Both constants are asserted from one
decade in, and every row from `1e-4` to `1e-8` sits within two percent of them. At
`1e-8` off the pole the curvature is `2.5e7` and the step `-5.68e-8`.

**The stall is conditional on the tolerance, which the original entry did not say.** At
a tolerance of `1e-14` the run does not freeze. It takes 24 iterations against the
modification's 12 and lands on the same estimate to `2e-16`. What the collapse buys is
a *threshold*: at `1e-8` off the pole the first step is `5.68e-8`, so any relative
tolerance above `2.84e-7` stops the run on iteration one and reports `0.500000057`
where the answer is `0.549191167`. Wrong by `0.049`, reported as converged. A hundred
times closer to the pole and the threshold falls a hundredfold, so no declared
tolerance is safe for every case. The original claim that a fixed budget returns the
prediction holds, and it needs the tolerance stated beside it.

**From below, the sum crosses zero at `x = 0.694474243706`** on the registered ridge,
inside the pole at `x = √½`. Approaching it, the step goes `8.81e4`, `8.81e5`,
`8.81e6` for distances `1e-8`, `1e-9`, `1e-10`, and the two sides point opposite ways.
Past the crossing the printed step climbs: from `x = 0.700791` the objective goes
`0.499721 → 0.574875`, where the modification takes it to `0.476722`.

The iteration count is the other cost. The printed scheme needed 24 iterations where
the modification needed 12 for the same answer, which is per-decision compute RFC-001
will account for.

## Q4. Below `σ = 1`, Gauss–Newton is outside its domain

`R⁽ᶥ⁾` is meant to be `∇ᵀr∇r` for the residual vector `r` of (20), and any such matrix
is positive-semi-definite. The negative term is precisely the third component of `r`,
`(ln det Σ)^½`, being imaginary and then squared. The paper's own footnote concedes the
component is imaginary when `det Σ < 1`.

The objective (18) is real and correct throughout. It simply is not a sum of squares
where `σ < 1`, so the method used to minimise it does not apply there. That is the
reason for any guard, and it belongs in the record as a statement about the method's
domain rather than as a numerical footnote.

### AMENDMENT 2026-09-04: the block that fails is named, and the repair is derived

From `research/spinello_stilwell_hand_derivation.md`, which is the scan of the notebook
beside it typed up. The question above says `R⁽ᶥ⁾` is not `∇ᵀr∇r` where `σ < 1` and
leaves it there. The derivation says which block, and what removing it costs.

Steps 5 and 6 of that file split the curvature by Jacobian row:

```text
R = (∂r₂)ᵀ(∂r₂) + (∂r₃)ᵀ(∂r₃)
```

The `r₂` block sandwiches to a real square for every direction, every innovation and
every `σ > 0`, so it is positive-semi-definite everywhere the mode is defined. The `r₃`
block sandwiches to `(σ'w)²/(4σ² ln σ)`, whose sign is decided by `ln σ` alone. The
whole of the defect is that one block, which is the whole of the fourth printed term of
(35d).

**The repair, derived and not yet run.** Drop the `r₃` block from `R`. Keep `s` (35c),
`Ū` (35e) and the objective (18) verbatim. Step 8 of the derivation gives the argument
that this is surgical: `ln σ` cancels out of the gradient, so `s` is untouched, `Ū` is
built from `s` alone and never contains `ln σ`, and the objective is not modified, so
the mode does not move. Only the path of iterates changes, and it then descends
provably. The cost is the name, since the iteration is no longer literal Gauss-Newton
on (20).

That argument is an argument. Nothing has compared modified against unmodified iterates
in code, and until something does, the repair is a route.

**A route the original pass missed.** Section IV's own constants appear to put `σ < 1`
throughout figures 2 to 4 for any `N ≥ 3`, since `κ = 24/(α²N³)` falls as `N⁻³`. If
that holds, the term was negative in every published run of the paper's own filter. It
is arithmetic on stated constants and no module has done it, so it is route 7 below
rather than a statement here.

### RESOLVED 2026-09-04: the repair is adopted for the shipped rungs

ADR-057. The three tests of the derivation's step 10 are run in
`research.spinello_stilwell.repair` and the amendment above records their results
under Q1 and here. The deletion keeps `s`, `Ū` and the objective verbatim, so the
fixed point, the posterior covariance and the gap are the paper's. What changes is the
path of iterates, and each step is now a descent direction because the step matrix is
positive definite. The name changes with it: the rung is the Spinello–Stilwell
iterated filter with a documented modification, and it is never called the paper's
equation verbatim.

The comparison of modified against printed iterates that the amendment above says is
unrun stays unrun, and it stays a route rather than a decision. Route 1's empirical
half now measures what the deletion cost rather than whether to make it.

## Q5. The ladder changes two things at once between rung one and rung two

Rung one, plug-in at the predicted mean, gives

```
P⁺⁻¹ = P⁻⁻¹ + (1/σ)∇ᵀh∇h
```

The paper's single-step filter (36e) gives that **plus** `(1/2σ²)∇ᵀσ∇σ`.

So moving from rung one to the iterated rung changes the inclusion of the
derivative-of-covariance terms *and* the iteration to convergence, together. An
improvement between them cannot be attributed to either.

(36) is (35) at a budget of one, so adding it as its own rung is nearly free. It turns a
four-rung ordering into a five-rung one in which two adjacent differences isolate
distinct mechanisms.

**This has to be decided before any R7 number exists.** A rung added after an ordering
is seen is what standing rule 7 refuses.

### ROUTE 2026-09-12: route 6 is registered, and what "separately visible" means

The five rungs exist. Route 6 asks whether the two mechanisms Q5 separated are visible
as two differences, and this entry fixes how that is read before the numbers exist. The
numbers come from the ordering's first reading, pre-registered under D1 of
`research/fep_falsification_battery.md` on this date: five gaps at fifteen spreads on
`d4-family-v1` at the binding cell, each with a measured refinement bar, adjacent pairs
resolved through `cpomdp.resolution`.

Two pairs answer route 6. `plug-in → modified-single-step` is the
derivative-of-covariance terms alone: both are one update from the prior mean, and the
second adds `(1/2σ²)∇ᵀσ∇σ` to the posterior precision and the `∇σ` terms to the score
and the curvature, and nothing else. `modified-single-step → modified-iterated` is the
iteration alone: both run the same step, and the second runs it to the declared
tolerance.

**Separately visible** means each pair resolves, in either direction, at one or more of
the fifteen spreads. The count of spreads at which each resolves is printed. A pair that
resolves nowhere is a mechanism the ladder cannot see on this family at these bars, and
then five rungs was the wrong declaration for this family. The record says so in a
RESOLVED entry here, and D1's prediction is read over the rungs that remain
distinguishable. Nothing leaves `LADDER`: a rung deleted after the numbers is what
standing rule 7 refuses, in the other direction.

Direction is not route 6's question. Which way each pair falls is D1's, and it is
registered there.

### RESOLVED 2026-09-12: route 6 is run, and both mechanisms are visible

The ordering's first reading, `research.checks.ladder` at `bf34ea0` against the
registration at `ac9fbc5`, resolves both pairs at all fifteen spreads on `d4-family-v1`
at the binding cell. The RESULT under D1 of `research/fep_falsification_battery.md`
carries the table.

`plug-in → modified-single-step` resolves at `1.7e8` times its bar or more. The
derivative-of-covariance terms move the gap by two decades at `σ = 0.7` and by four at
`σ = 0.005`. `modified-single-step → modified-iterated` resolves at `1.4` times its bar
at `σ = 0.005`, where one step is within `1.2e−16` nats of the fixed point, and at
`1e³` times or more from `σ = 0.03` up. The iteration moves the gap by a part in `10³`
to `10⁵` of what the terms moved it.

So the two adjacent differences Q5 asked for are two differences, and five rungs was
the right declaration. Which way each falls is D1's, and the RESULT records that both
reverse above the registered window.

One thing the reading adds to Q6. The belief-smoothed rung sits within a part in `10⁴`
of the plug-in at every spread but the largest, two to four decades above the
single-step rung. Averaging `R` under the prior does not recover what rung one
discards, since `E[R(x)]` carries no `∇σ` either. The blindness Q6 names is the
fourth rung's too.

## Q6. What rung one is blind to

The term rung one discards, `(1/2σ²)∇ᵀσ∇σ`, is non-zero exactly when the noise varies
with the state. That is the same derivative-of-covariance information that makes the
channel epistemically active at all.

So rung one does not merely approximate the `R(x)` posterior badly. It approximates it
in a way that cannot see the mechanism under test.

Worth a sentence in Paper 2 Part 2, stated carefully. The filter's posterior information
and expected free energy's epistemic term are related quantities and not the same
object, and a sentence that runs them together would claim more than this observation
supports.

## Q7. Budget exhaustion corrupts the covariance too

`Ū` is evaluated at `x̂⁽ᶥ⁾` per (33e). A truncated iteration therefore returns a wrong
`P[k|k]` as well as a wrong mean, so both halves of the divergence move.

That argues for a non-convergent step being routed to `VOID` rather than reported as a
number. A gap computed from a posterior whose mean and covariance are both wrong by an
unmeasured amount is not a measurement of anything.

### ROUTE 2026-09-04: route 5 is run, and the covariance is not the smaller error

`research.spinello_stilwell.budget`, run with

```text
uv run --no-sync python -m research.spinello_stilwell.budget
```

**Both halves move, on all twelve declared cells.** The grid sweeps `κ ∈ {0.1, 1, 5,
20}` against prior variance `{0.01, 0.25, 1}`, the two axes GATE-D4's registered family
sweeps. At a budget of one the mean departs by `0.0069` to `0.357` and the variance by
`7.6e-7` to `0.425`.

**On the widest prior at the strongest curvature the variance is the worse of the
two**, `0.425` against the mean's `0.357`. So the covariance is not a bystander whose
error is bounded by the mean's, and a caller who trusted the covariance of a truncated
run would be worse off than one who trusted its mean. That is what makes `VOID` the
right routing rather than reporting the number with a caveat.

Both departures fall as the budget grows, so what is measured is truncation and not the
scheme.

**What a budget has to cover, in the bulk and at the box edge.** Over the five declared
families at the six declared spreads, with the reading two *prior* spreads out, the
modified scheme at tolerance `1e-14` needs 2 to 10 iterations, worst at `1 + x²` and
spread `0.30`, and the fixed family needs 2 everywhere. That reading is inside the bulk.
`averaged_inference_gap` runs its observation box nine *predictive* spreads out on
either side of the prior mean and calls the rung at every node of it. Out there, at
`1e-12`, the same families need up to 124 iterations, worst at `1.5 + 0.5 sin(x)` and
spread `0.30`, and the curvature axis from `0.1` to `100` needs up to 65. Nothing here
declares a budget. Those are the numbers a declaration reads.

**The slopes are declared, not differenced.** `FAMILIES` declares `R` and (35) needs
`R'`. A central difference at `1e-6` carries about `1e-11` of error into the iterate,
which floors convergence well above `1e-14`: five of the thirty cells then ran to the
probe budget without settling, measuring the difference rather than the scheme. The
slopes are written out in `budget.SLOPES` and checked against a difference to `1e-6`.
A tolerance declaration has to sit above whatever error the sensor model's own
derivative carries, and that is not a property of the scheme.

**What the deletion costs at a finite budget.** Same case, same budget, both
curvatures: the mean differs by `3.2e-3` at a budget of one, `6.7e-6` by five, and `0`
run out. ADR-057's "surgical" is that last number. The two curvatures share a fixed
point and differ only on the way to it.

### RESOLVED 2026-09-04: the budget is 64 and the tolerance is `1e-12`

ADR-058 declares both, against the counts route 5 measured. `VOID` for an exhausted
budget stands as ADR-056 left it, and Q7's argument for it is now a number: at a budget
of one the covariance is out by up to `0.425`, and on one declared cell by more than the
mean is.

The declaration carries a condition. The rung differentiates `R` by automatic
differentiation, because a central difference floors convergence above `1e-12` and the
run would then be measuring the difference. That reaches the rung's contract in PR-7
rather than staying here.

Two cells exhaust the budget, both at the observation box's edge. `1.5 + 0.5 sin(x)` at
spread `0.30` takes 124 iterations and `κ = 100` takes 65. The first is a declared family
at a declared spread rather than a bracket: `R` is bounded and periodic there, so the
iterate crosses several periods of it before it settles. Both nodes carry `2.6e-18` of
the centre's predictive weight. What a voided node does to a reported gap is left open,
and ADR-058 names the three ways it could go.

Past the box edge the scheme stops converging at all on the bounded family: at thirteen
predictive spreads `1.5 + 0.5 sin(x)` at spread `0.30` settles into a stable two-cycle,
alternating by `2.83` for as long as it is run. That is outside the registered box, and
it is why the budget comes with a routing instead of being chosen large enough to be
safe.

### RESOLVED 2026-09-10: a voided node leaves the average by weight

ADR-060 takes the decision ADR-058 left open. A rule answers `Void` in place of a
belief, and `averaged_inference_gap` records the node as unmeasured: NaN in
`divergences`, its predictive weight in `voided_mass`, and `value` taken over the
readings the rule answered, normalised to their own mass. That is the conditional
reading a clipped box already gets through `predictive_mass`, reported on the same
terms. Route 5's two edge nodes carry `2.6e-18` each, so under this routing they reach
`voided_mass` and move `value` by nothing at the precision anything here is read at.
A check that wants the strict figure asserts `voided_mass` is zero. Widening the budget
stays rejected on ADR-058's trade, and voiding the whole sweep for a node at that weight
would erase the reading of the rule where its weight lies.

Landed in `src/cpomdp/reference/gap.py`, pinned by two tests in
`tests/test_reference_gap.py`.

## Q8. Rung two has no published numerical validation

Every figure in §IV uses (36), the single-step filter. The iterated scheme (35) is
derived and never simulated.

So the rung arrives with no external check of any kind, and whatever oracle the test
suite provides is the only thing standing behind it. That raises what the fixed-`R`
reduction has to carry: at `∇σ = 0` the scheme must reproduce the ordinary Kalman
update exactly, and that is the one place an independent answer exists.

## The routes, collected

| # | Route | What it decides |
| --- | --- | --- |
| 1 | Rescaling exploration over several decades of `λ` | whether `1/ln σ` is the only term that moves |
| 2 | Reachable-`σ` survey per declared family | whether one `λ` clears the pole for all of them |
| 3 | Two-sided failure probe near `σ = 1` | that the stall returns the prediction, silently |
| 4 | `∇σ = 0` reduction against the Kalman oracle | the only external check the rung has |
| 5 | Budget-exhaustion probe | that both mean and covariance move, so `VOID` is right |
| 6 | Rung-one-versus-(36) separation | whether the two mechanisms are separately visible |
| 7 | Section IV's constants, evaluated | whether the paper's own runs ever had `σ > 1` |

Routes 1, 2 and 4 are prerequisites for the rung. Routes 3, 5 and 6 decide how it
reports and how many rungs the ladder declares. Route 7 was added by the AMENDMENT of
2026-09-04 under Q4 and decides nothing the rung needs, only what the citation may say.

### ROUTE 2026-08-24: routes 1 and 2's symbolic half are run, and Q2's premise is measured

`research/src/research/spinello_stilwell/invariance.py`, run with

```text
uv run --no-sync python -m research.spinello_stilwell.invariance
```

Three things, asserted inside the module rather than printed:

- **Route 1, symbolic half.** Seven of the eight terms in `s`, `R⁽ᶥ⁾` and `Ū` return to
  themselves under `o → λo`. Only `(1/4σ²)(1/ln σ)∇σ∇σ` moves, becoming
  `1/(ln σ + 2 ln λ)`. That is the whole of Q1's claim.
- **Route 2, symbolic half.** The pole therefore sits at `σ = λ⁻²`. On the ridge,
  `σ` at the operating point is `2R₀` whatever `κ` is, so `R₀ = ½` puts the pole exactly
  there at `λ = 1`. Both derived rather than substituted.
- **Q2's premise, which the original pass asserted without checking.** The converged
  estimate and the posterior variance are unmoved by the rescaling, to 1e-12 across
  `λ ∈ {1, ½, 3, 7}`, while the iteration count is not: 6, 9, 8, 8. If rescaling moved
  the answer, choosing units to clear the pole would be choosing an answer, and it would
  be a tuned parameter rather than a declared convention. It does not.

The numeric part uses a reference implementation of (35) written for that check alone.
It is not the rung. It carries no guard at the pole and takes its iteration budget as an
argument rather than declaring one, both deliberately, so that nothing in it can stand
in for a decision the rung still owes.

What is **not** run: the empirical half of route 1 (a rescaling sweep of the reported
gap, which needs the rung), and routes 3 to 6 entirely. The two-sided failure of Q3, the
budget-exhaustion behaviour of Q7 and the rung-one-versus-(36) separation of Q5 remain
routes, and no number here bears on any of them.


### AMENDMENT 2026-09-04: the budget those numbers ran at is stated

The entry above says the reference implementation "takes its iteration budget as an
argument rather than declaring one". It did, and it also gave that argument a default
of 200 with a tolerance of `1e-14`, and `main()` passed neither, so every number in the
table was produced at a budget the record did not state. Review caught it.

Now: `scheme.iterate` and `invariance.iterate` have no default budget or tolerance,
every caller passes both, and the probes' own are named `PROBE_BUDGET = 200` and
`PROBE_TOLERANCE = 1e-14` in `invariance.py` and printed beside every table they
produce. Every run in the table above converged within that budget, asserted, since a
run that exhausts it returns a count equal to the budget and Q7 says such a run's
covariance is wrong as well as its mean. The rung's budget stays undeclared (ADR-056).
The test that polices it asserted the wrong thing, that a default existed, and now
asserts that none does.

### AMENDMENT 2026-09-05: the budget that entry left undeclared is declared

"The rung's budget stays undeclared (ADR-056)" above was true when it was written. It is
not now. ADR-058 declares 64 iterations at a tolerance of `1e-12` relative to the prior
spread, read off the counts route 5 measured, and Q7's section records it. The probes
keep their own `PROBE_BUDGET = 200` at `1e-14`, which is not the same number and is not
a candidate for it: a probe that ran to the declared cap would report the cap rather
than the count the cap has to be judged against.

### RESOLVED 2026-08-24: route 2 is run, and only one declared family resists a unit choice

`research/src/research/spinello_stilwell/reachable_noise.py`, run with

```text
uv run --no-sync python -m research.spinello_stilwell.reachable_noise
```

The infimum of `R` over the whole state line, per declared family:

| family | `inf R` | clears the pole at |
| --- | --- | --- |
| `1 + x²` | 1 | any `λ > 1` |
| `1.5 + 0.5 tanh(x)` | 1 | any `λ > 1` |
| `1.5 + 0.5 sin(x)` | 1, **attained** | any `λ > 1` |
| `2` (fixed) | 2 | already clear at `λ = 1` |
| ridge, `R₀ = ½` at `μ*` | 0.5 | `λ > 1.5492` |
| `exp(x)` | 0 | **no `λ`** |

Four of the six never dip below `R = 1`, so the pole is outside their range for any
`λ > 1`. That was not obvious in advance and it is a property of how they were declared
rather than of any sweep width.

**`1.5 + 0.5 sin(x)` attains `R = 1` exactly**, wherever `sin(x) = −1`. At `λ = 1` the
pole is *on* the family, not near it. That single family is what makes the unit choice
compulsory rather than prudent, and it was already in the registered set before any of
this was noticed.

**One `λ` covers the entire survey.** Over the reachable window, taken at nine prior
standard deviations either side of the mean, `λ = 2.5630` puts every family at every
declared spread a factor 1.2 clear of the pole. A single declared convention suffices,
which is what keeps this a unit choice rather than a per-cell tuning.

**`exp(x)` is the exception, and it is a real one.** Its infimum over the line is zero,
so no `λ` clears every state. It is cleared only over a declared box, and the required
`λ` grows without bound as that box widens: at spread 0.30 the reachable minimum is
0.183 and needs `λ > 2.56`, and a wider box needs more. So for that family the guard
question of Q2 stays open, and it is bound to route 3: a run whose iterate escapes the
declared window can still reach the pole, and Q3 says the approach from above is silent.

What this does **not** settle. The `λ` above is a measurement, not a declaration. What
gets registered is the owner's call and belongs in an ADR, and until then no rung may
read a unit choice off this table. Routes 3, 4, 5 and 6 are untouched, and route 4, the
`∇σ = 0` reduction against the Kalman oracle, remains the rung's only external check.

### AMENDMENT 2026-09-04: two families attain the pole, and the reach was too narrow

Two corrections to the entry above, both from review of the pull request that landed it.

**`1 + x²` attains `R = 1` too**, at the origin, and the table's own `inf R` column
already said so while the marker and the prose withheld it. The claim that `sin` is the
single family making the unit choice compulsory is false. Over the declared grid it is
the quadratic that reaches the pole in more cells: both priors sit at `x = 1`, the
quadratic's minimum is one unit away and the sine's nearest is at `−π/2`, so the
windows reach the first in five cells of six and the second in two. The quadratic is
the family carrying the registered `c₄` provenance, so the family that forced the unit
choice was the registered one and not a bystander.

**The reach cited the wrong axis.** Nine was the gap quadrature's `y` grid, in
predictive standard deviations. The quadrature's *state* window starts at twelve prior
standard deviations (`X_WINDOW_LADDER[0]`), so nine was narrower than the range the
reference already treats as the state's, against the constant's own stated principle.
`REACH_IN_SPREADS` now reads that foot. At twelve, `exp(x)` at spread `0.30` reaches
down to `0.0743` and the single `λ` covering the survey is **`4.0195`**, not `2.5630`.
The margin convention is also now one convention: the four families that never dip
below one clear at `λ > √1.2 = 1.0954` with the declared margin, where the entry above
said "any `λ > 1`" for those and quoted the ridge with the margin included.

Nothing in the entry's conclusion changes: one `λ` still covers every family but
`exp(x)`, and `exp(x)` still has no `λ` over the line. The unit-choice ADR still owed
above may make the number moot for the shipped rung. The correction is recorded because
the entry stated it as a result. `research.spinello_stilwell.reachable_noise` asserts
the corrected claims and `tests/test_spinello_stilwell_invariance.py` pins them.
