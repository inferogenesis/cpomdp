# How a cell gets its two numbers

I'm writing this as the evaluator gets built, one step per commit, so the logic flow
is readable without opening the diffs. The maths contract is
`research/fep_falsification_battery.md` section C. The code is `cpomdp._lab.scoring`.

## The cross comes first

`build_cross` builds every cell of the two declared axes and refuses to hand back fewer
than `|model axis| × |inference axis|`. The certificate is a
`ProductCompletenessCertificate` naming both axes and both versions, so a run over a
different cross cannot be mistaken for this one. This is the Route 2 purchase from the
build plan, and it is why every separation reported later is a comparison between two
cells of one decided set rather than two cells someone happened to run.

## The shape comes next

`Decomposition` holds `misspecification` and `inference_gap`. Nothing else. The floor
`H(p*)` belongs to the process, not to any cell, and prohibition 1 bans reaching a
term by subtracting it. The type has no slot to put an entropy in, and a test pins the
field list, so growing it is a visible decision.

## The divergence is a sum of non-negative parts

Both terms are KL divergences between Gaussians, so one primitive serves both.
`gaussian_kl` does not use the textbook form. That form subtracts `n` from a trace and
one log-determinant from another, and near equality it returns rounding noise of either
sign. A term reading `1e-16` under it says nothing about whether the term is zero
(ledger section 4, "absence of catastrophic cancellation").

The primitive whitens by `cov_b`'s Cholesky factor and reads the eigenvalues `λ` of
`cov_b⁻¹ cov_a` off a singular value decomposition. Each contributes `λ − 1 − ln λ`,
which is non-negative on its own, and the mean term is a sum of squares. Near `λ = 1`
even `λ − 1 − ln λ` is a cancellation, so below `1e-4` it is evaluated by its series
instead. Nothing is subtracted from anything of like size. Identical inputs return `0.0` exactly, and two
Gaussians `1e-13` apart return `~1e-26` with a sign that cannot flip. That is what lets
a later cell say "below `1e-12`" and mean it.

The tests hold it against the scalar closed form, the textbook form where that form is
accurate, and an affine change of coordinates, which is the invariance the ledger
relies on to treat nats as a scale-free unit.

## The misspecification term reads two predictives

`observation_predictive` pushes a belief through the dynamics and then through the
sensor and returns the Gaussian `p(y | u)` for the next reading. `misspecification_step`
is the divergence from the true model's predictive to the cell model's, each predicting
from its own **exact** belief.

That last word is the design decision. If the cell's predictive came from the agent's
own belief, a cell that filtered badly under the correct model would show a nonzero
misspecification, and C3 would fail by construction rather than by finding. So the
term is a function of `(p*, p, y_{1:t−1}, u)` and nothing about the filter. Two
separately built models with equal numbers score `0.0` exactly at every step, since
both sides run the same arithmetic on the same values and identical Gaussians diverge
by exactly zero.

Both functions refuse state-dependent noise. Under `R(x)` the predictive is a scale
mixture with no covariance that describes it, and scoring that is the reference
filter's job at PR-9, not a formula's.

## The inference gap is an average with a closed form

`inference_gap_step` takes two functions of the reading: the cell's filter step at its
current belief, and the exact step under the cell's own model at the exact filter's
belief. Every Gaussian filter step is affine in `y`, with a covariance that never sees
the reading, so the average over `y ∼ p*(y | u)` collapses to

```
E_y[KL] = KL at y = m*  +  ½ · tr(Σ_p⁻¹ · D · S* · Dᵀ)
```

with `D` the difference of the two gains and `S*` the true predictive covariance. The
trace is a Frobenius norm of a whitened `D`, so the whole thing is squares and
`λ − 1 − ln λ` terms, like the divergence it is built on.

The gains are not asked for. A backend exposes `infer_states` and nothing about how it
got there, so the step is probed: `m + 1` calls read the map off unit offsets, and one
more call in the opposite direction checks the map predicts it. A step that is not
affine, or whose covariance moves with the reading, is refused. The closed form then
cannot be applied to a rule it does not describe, which matters more than the probe's
cost (it is an instrument, not the agent's per-cycle path).

Three oracles hold it. The scalar closed form `test_reference_gap` already uses. The
grid engine in `cpomdp._lab.reference`, on the fixed-`R` case that ADR-053 says is the
one both engines can reach, agreeing to `1e-6` with an integrator that knows nothing
about gains. And a two-dimensional case with the gains written out by formula and the
average taken by the textbook trace. The exact rule scores `0.0` exactly, since it and
the exact step are one computation.

## The evaluator walks a run

`ThreeTermEvaluator` holds `p*`. It is the experimenter's object, never an agent's, so
it is the one place the true model and a cell's model are in hand together. `score`
takes a cell, a `DrivenRun` and the sequence that drove it, and refuses a sequence
whose version the run did not record.

Three filters walk the readings side by side: the exact filter under `p*`, the exact
filter under the cell's model, and the cell's own filter. At each step the two
predictives give the misspecification term and the two updates give the gap, both
given the readings so far. Then all three fold the actual reading and move on. The
two terms are sums over steps.

What that buys, on a twelve-step double integrator with the sensor noise perturbed by
half and three degraded rules:

- correct model, exact rule: both terms `0.0`, exactly (R1)
- perturbed model, exact rule: misspecification positive, gap `0.0` exactly (C2)
- correct model, any degraded rule: misspecification `0.0` exactly, gap positive (C3)
- perturbed model, degraded rule: both positive, the diagonal cell the off-diagonal
  separations are read against

The exact zeros are structural. C2's gap is zero because the cell's filter and the
exact step under the cell's model are one computation. C3's misspecification is zero
because the term never reads the cell's filter. Neither is a tolerance being met.

Not yet here: the separation ratio and the conditioning beside every cell (F2, F3),
the additivity check with its four-term bound (F1), and common-mode propagation (F5).

## The conditioning travels with the score

`score` now returns a `CellScore`: the two terms, the step count, and a
`RunConditioning` holding the condition number of every matrix the terms factored or
inverted, per step. Four columns: the true predictive `S*`, the cell model's
predictive `S`, the exact posterior `Σ` under the cell's model, and the cell filter's
own `Σ`. An ill-conditioned matrix can manufacture or destroy twelve orders (ledger
section 4), so the reading is carried beside the number rather than looked up later.

One implementation serves the rollout and the score. `condition_numbers` moved into
`cpomdp.diagnostics` and `rollout_conditioning` calls it, so a rollout and a cell
read the same number for the same matrix.

## A separation is a ratio, read against a cell where both terms move

Each `CrossCell` now knows whether its row is the unperturbed model and whether its
column is the exact rule. A `CellScore` with exactly one axis at the reference has a
`Separation`: which term the axes pinned, what it read, what the other term read, and
the ratio between them. The ratio is the claim (ledger section 4). Where the pinned
term is exactly `0.0` the ratio is `inf`, and the pinned value beside it says why.

`score_cross` scores every cell and hands back a `CrossScore` carrying the cross's
certificate. Its `both_positive` cells are the perturbed model under a degraded rule
with both terms above zero. `separations` refuses to answer when there are none: a
contrast needs a cell that shows both terms can move, and a cross with only the exact
rule cannot show it.

`render` prints one row per cell. A separation row carries its pinned term, its
moving term, the ratio, and the worst `cond(Σ)` and `cond(S)` behind it, on that one
line, so a separation cannot be printed without them (prohibition 5). On the
twelve-step double integrator the frozen gain and the diagonal rule read gaps in the
tens of nats. Both start from a prior with unit covariance that the exact filter
contracts fast and they do not, so the early steps dominate. That is a reading, not a
result: nothing here is registered yet, and the numbers are `COMPUTED` until PR-5
declares what they are measured against.

## The additivity check measures both sides on its own

`cpomdp._lab.additivity` is a separate module, and `AdditivityCheck` is a separate
object, because it is the one place `H(p*)` is estimated. The evaluator's two terms
never need an entropy. The check takes an `EntropyEstimator` explicitly:
`GaussianEntropy` is the closed form with no bar, `MonteCarloEntropy` samples and
carries a standard-error bar, and anything else that fits the protocol can be wired in.

The left side is measured directly. `variational_free_energy` computes
`E_q[ln q − ln p(y|x) − ln p(x)]` term by term from the cell filter's belief and the
model's joint, never through the identity. Readings are drawn from the true
predictive and `F` is averaged over them, with `δ_F` as a stated number of standard
errors. The right side is the estimator's entropy plus the two closed-form
divergences. Nothing on one side is derived from the other.

The residual carries the four-term bound `δ_F + δ_H + δ₁ + δ₂`. The worked case in
the tests shows a residual of `0.008` closing inside `δ_F = 0.01` and failing when
`δ_F` is dropped, with nothing about the measurement changed. On the double
integrator the accounting closes on the calibration cell, both off-diagonal kinds
and the both-positive cell, with the closed-form entropy and with the sampled one. A
closed form nudged by a twentieth of a nat fails to close, which is what says the
check can tell.

## A difference is read at its own error, not the sum of two bars

`cpomdp._lab.resolution` is a leaf. It imports nothing first-party, and the boundary
test asserts that, because it is built once for the evaluator and for the things that
may not import the evaluator: the reference filter, and Paper 3's G10. It went into its
own module rather than `cpomdp.diagnostics` because it is not a diagnostic of a model.
It is the arithmetic of comparing two bounded numbers.

A `Bar` is split by where its error came from. The `common_mode` part is the shift
the reference's error puts on the value, at the reference's bar, and it is *signed*.
The `own` part is the quantity's alone. Two quantities scored against one reference
share the reference's error, so in their difference the common-mode parts subtract:
`difference_bar` carries `a.common_mode − b.common_mode` and `a.own + b.own`. The
conservative rule reads `|a| + |b|` on the common-mode parts instead. When the two
sensitivities share a sign the difference bar is smaller, which is the cancellation
ledger section 4 describes. When they do not, nothing cancels and the two rules
agree. The sign is what keeps the second case honest; an unsigned split would
under-bound it, the way dropping `δ_F` under-bounds the additivity residual.

The threshold is pre-registered by construction. `resolution_threshold` takes two
`Bar`s, and a `Bar` has no value field, so it cannot be read off the numbers it
judges. `resolve` compares two `Bounded` values at that threshold and reports one of
three: `BELOW`, `ABOVE`, or `NOT_RESOLVED`. The third is the measured tie the ledger
asks the fidelity ladder for, and the strict inequality means two `EXACT` values that
agree also read as not resolved, since a difference of zero exceeds nothing. The
`Resolution` carries `sum_of_bars` beside `threshold` so a report can show what the
shared reference bought.

Two worked cases. Two rungs at `0.500` and `0.503`, each carrying `0.010` from the
reference and `1e-4` of its own: the sum of bars is `0.0202` and calls it a tie, the
difference bar is `2e-4` and resolves it. Then two nearby Gaussians scored by
`gaussian_kl` against a third, whose mean is shifted by `1e-2` to stand in for its
discretisation error. Both divergences move by about `0.01` with the same sign and
the difference moves by `1e-4`. The two thresholds differ by a factor of two hundred
on the same numbers.
