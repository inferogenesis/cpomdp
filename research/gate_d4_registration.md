# GATE-D4 pre-registration

*Opened 2026-08-07. GATE-D4 compares a certified discretisation bound `δ_ref` against R6's
inference-gap signal by a pre-agreed factor. `fep_falsification_battery.md`'s D4 entry
states the timing requirement: the factor is written down **before the bound is computed**.
This document fixes what can be fixed while neither quantity exists.*

**Why it opens this early.** D4's predicate is that the bound be small relative to R6's
signal, so evaluating the gate needs the signal, and the signal is produced by the exact
reference filter whose error the bound certifies. By the time the gate is evaluable, both
the signal and the bound exist. A factor fixed then can be sized to what the bound turned
out to be. The only window in which neither quantity exists is the one before the reference
filter is built, which is now.

**Why it opens partially.** Section 1 is dated today and the rest is not, deliberately. The
family is the largest free parameter in the construction: `c₂`, `c₄` and the curvature
ceiling are all functions of it, so choosing the family chooses the bar. Holding the
declaration back until the coefficients were in hand would mean choosing the family with its
consequences in view. Versioning catches a family *changed* after results. Nothing catches
one *chosen toward* them except declaring it first.

Sections marked **OUTSTANDING** carry their method and their trigger, and each is dated when
it lands.

---

## 1. The model family (DECLARED 2026-08-07)

**`d4-family-v1`**

```text
scalar chain, 1-D latent:   A = B = C = 1,  Q = 1
sensor noise:               R(x) = R₀ + κ·x²          (R₀ = 1)
swept axis 1 (spread):      prior variance σ²
swept axis 2 (curvature):   κ,  where R''(x) = 2κ
derived (not swept):        μ = μ*(κ) = √(R₀/κ)      -- see the amendments
```

**Selection reason, prior to and independent of D4.** This is the family Paper 1's `R(x)`
result already uses. `tests/test_theorem.py` declares it as "the scalar chain the worked
example uses: A = B = C = Q = 1, R(x) = 1 + x²", and has done since 2026-06-19, before
GATE-D4 was scoped or named. The only generalisation here is promoting the fixed coefficient
on `x²` to a swept parameter `κ`, which D2 requires because it sweeps the curvature of `R`.
`κ = 1` recovers the worked example exactly.

It also sits inside the reference filter's declared scope, which `warrant_ledger.md`
section 7 limits to roughly one or two latent dimensions.

**What a later change costs.** The version tag is what makes a changed family appear in a
diff rather than in prose (standing prohibition 9). A replacement family requires its own
declaration and its own dated reason, written *before* any measurement on it.

### AMENDMENT 2026-08-07: the prior mean is part of the family, and `R'(μ) ≠ 0` is required

`d4-family-v1` as declared above left the prior mean `μ` unstated. It cannot be left
unstated, because it decides the exponent D2 sets out to fit.

Measured on the quadrature gap of section 2, at `κ = 1`, as `σ → 0`:

| prior mean | local exponent `d log gap / d log σ` | consequence |
| --- | --- | --- |
| `μ = 0` | → 4 | `c₂ = 0`. D2 fits 4, not 2 |
| `μ = 1` | → 2 | `c₂ ≠ 0`. D2 fits 2, as predicted |

The reason is structural. `R(x) = R₀ + κx²` has `R'(0) = 0`, so at `μ = 0` the belief sits
exactly on `R`'s minimum, `R` varies only quadratically across it, and the gap drops two
orders. Section 2's law makes this exact: `c₂ = (R'(μ)/2R(μ))²`, which vanishes precisely
at a stationary point of `R`.

**`d4-family-v1` is amended to fix `μ = 1`**, recovering `R(x) = 1 + x²` evaluated away from
its minimum, with `R'(1) = 2κ ≠ 0`. `tests/test_theorem.py`'s `scalar_chain` defaults to
`mean=0.0`, so Paper 1's worked example sits at the degenerate point. The
generalisation to `μ = 1` is part of this declaration and not inherited from it.

**Discovery order, stated plainly.** `μ` was fixed *after* observing the exponent collapse
at `μ = 0`, not before. Two things bound what that could have bought. The ground is
structural and checkable without any measurement, since `R'(μ) ≠ 0` is a property of the
declared `R` and nothing else. And `δ_ref`, R6's signal and the threshold `T` did not exist
on this date and still do not, so no choice here could be aimed at the gate's outcome. What
it does affect is whether D2 is a test at all, which is the thing a reader should weigh.

### AMENDMENT 2026-08-07: `μ` is *derived*, which E2's dichotomy has no slot for

Once `μ = μ*(κ)`, `μ` stops being an independent swept axis: the spec named two, and one is
now a function of the other. The family block above is corrected to say so.

E2 requires every parameter "declared **fixed or free in the model specification**, never at
analysis time". `μ` is neither. It is **derived**, and that is a third category with its own
contamination profile: it cannot be tuned directly, which is stronger than *free*, but it
inherits whatever freedom sits in the rule that derives it, which is weaker than *fixed*.
The freedom here is the choice to maximise `c₂`, declared above and dated.

### AMENDMENT 2026-08-07: `R'(μ) ≠ 0` is a precondition on any fixture without a mean-moving action

Paper 1's worked example is safe at `mean = 0.0` because its *actions* displace the
predicted mean, so `R(μ⁻)` takes 1 and 5 and `R'(μ⁻) ≠ 0` where it matters. That
generalises: the degeneracy arises exactly where nothing displaces the mean.

**So any fixture without a mean-moving action asserts `R'(μ) ≠ 0` rather than assuming it.**
Checkable from the declared `R` alone, at no cost, and it joins the span-positivity
precondition already recorded from the linear family. Both were found the same way in one
sitting, which suggests the class is worth checking for rather than waiting to trip over.

## 2. The Taylor coefficients `c₂` and `c₄` (OUTSTANDING)

The inference gap as a function of belief spread, expanded about zero:

```text
gap(σ) = c₂·σ² + c₄·σ⁴ + O(σ⁶)
upper window edge (spread axis) = √(c₂/|c₄|)
```

`|c₄|` rather than `c₄`. The edge is the spread at which the quartic term matches the
quadratic, and that crossing exists either sign: a negative `c₄` cancels the quadratic there
rather than overtaking it, and it is the same breakdown scale.

### AMENDMENT 2026-08-07: the registered method was wrong

As first written, this section said to nest `jax.jacfwd` on "the closed-form fixed-`R` path,
where the Kalman filter *is* the exact Bayesian filter". That path returns zero. Under fixed
`R` with `p = p*` and exact inference, both divergences are **identically zero by
construction**, which is R1's content. The gap exists only under `R(x)`, where the exact
posterior is non-Gaussian. The original method computed the wrong quantity, and it is
recorded here rather than overwritten.

**The method that works.** Build the gap by quadrature: the exact posterior
`p(x|y) ∝ N(x; μ, σ²)·N(y; x, R(x))` on a grid, the agent's Gaussian `q` from a Kalman
update with the plug-in `R̂ = R(μ)`, then `E_{y∼p*}[KL(q ‖ p)]` with the reverse KL PR-7
specifies. It converges to twelve digits at a few thousand grid points.

One precondition the grid imposes: `R` must stay positive across the whole quadrature span,
not merely near `μ`. A family that violates it silently produces `log` of a negative number,
and the span is wide (many `σ`) precisely where `σ` is large.

Two traps still stand. `c₂ = f''(0)/2` and `c₄ = f⁗(0)/24`. Using raw derivatives puts a
factor of 12 in the ratio and moves the upper edge by `√12 ≈ 3.5×`. And a free correctness
check remains available wherever the gap is even in belief spread.

**Diagnosis rule, registered in advance (ledger section 8, third standing rule).** A second
independent method cross-checks with a tolerance stated before it runs. Disagreement
*inside* that tolerance is reported as the expected discrepancy. Disagreement *outside* it
is attributed **first to an implementation error in one of the two**, not to precision.
Registering "one method wins" would be a rule for walking past a bug.

### RESULT 2026-08-07: `c₂` in closed form

```text
c₂ = ( R'(μ) / (2·R(μ)) )²
```

Half the log-derivative of `R` at the prior mean, squared. Verified against the quadrature
gap on five structurally different `R` families (quadratic, exponential, linear, cubic, and
the quadratic at nine `(μ, κ)` combinations), agreeing to five significant figures in every
case. It is a general law of the construction, not a fit to one family.

Three consequences.

**`c₂ ≥ 0` always**, being a square. So the stop branch "`c₂ ≤ 0`" collapses to `c₂ = 0`,
which holds exactly when `R'(μ) = 0`. That is a structural condition checkable from the
declared `R` without any measurement, and it is what the section 1 amendment acts on.

**For `d4-family-v1`, `c₂ = κ²μ² / (R₀ + κμ²)²`.** At `μ = 1, κ = 1` that is exactly `0.25`,
and the quadrature returns `0.250009`.

**The battery's stated scaling is wrong on the curvature exponent.** D2 predicts
`gap ∝ (curvature of R) × (belief spread)²`. The spread exponent of 2 is confirmed. The
curvature dependence is not: `c₂` is quadratic in `κ` at small `κ`, not linear, and the
quantity that actually governs the gap is the *relative gradient* `R'/R` rather than the
curvature `R''`. This is a correction to a registered prediction, found analytically before
any measurement, and it bears on how D2's second axis should be swept.

### AMENDMENT 2026-08-07: the upper edge, and why `√(c₂/|c₄|)` is unusable

`c₄ < 0` is the measured case, not a defensive possibility, and it breaks the edge as first
written. With `gap = c₂σ² + c₄σ⁴` and `c₄` negative, `σ = √(c₂/|c₄|)` is where the quartic
*cancels* the quadratic, so the expansion predicts a gap of zero there. The gap turns over
earlier still, at `σ = √(c₂/2|c₄|)`, and decreases beyond it. The old edge therefore sits
`√2 ≈ 1.414×` past the point where the gap stops being monotone in spread, which is fatal
for fitting a power-law exponent.

**Redefined.** The edge is the spread at which the quartic correction reaches a declared
fraction `f` of the quadratic term:

```text
σ_max = √( f · c₂ / |c₄| )
```

`f` is declarable on the same footing as `D`, being a statement about how well exponent-2
must hold across the fit range, and it carries no route to D4's outcome for the same reason.

**`f` must be settled before `D` is frozen.** It moves the edge directly: `f = 0.1` scales it
by `√0.1 = 0.316`, costing exactly half a decade, which comes straight out of `D`. Freezing
`D` first would freeze it against an edge that then moves.

The threshold picks up `f` with it:

```text
T = f · (c₂/|c₄|) · curvature · 10^(−2D)
```

### AMENDMENT 2026-08-07: `μ` by rule rather than by value

The first amendment fixed `μ = 1`. That was a value, and `R'(μ) ≠ 0` rules out `μ = 0`
without picking `1`. Since `c₂` depends on `μ` quantitatively, `μ` moves `T`, which is the
same residual freedom the threshold construction exists to remove.

**The rule: `μ` sits at the argmax of `|R'/2R|`.** For this family,

```text
μ* = √(R₀/κ)        c₂(μ*) = κ / (4·R₀)
```

Both verified exactly against a numerical maximisation at `κ ∈ {0.25, 0.5, 1, 2, 4}`. The
rule is family-general, derivable now, and severity-maximising in `c₂`, since a larger `c₂`
raises `T`. At `κ = 1, R₀ = 1` it returns `μ* = 1`, so the earlier value coincides with the
rule at that point rather than being contradicted by it.

**A consequence for the sweep.** `μ*` depends on `κ`, and `κ` is a swept axis, so `μ` must
co-vary with `κ` along it rather than staying pinned. Holding `μ` fixed would put a single
`κ` at the optimum and the rest off it.

**And it changes the curvature story again.** At the optimum `c₂ = κ/(4R₀)` is *linear* in
`κ`. The battery's `gap ∝ curvature × spread²` is therefore recovered along the optimal
ridge, even though the mechanism it names remains wrong: what governs the gap is the
relative gradient `R'/R`, and linearity in `κ` is a property of the ridge rather than of the
curvature.

**Declared target.** `μ` maximises `c₂`, not `c₂/|c₄|`. `c₄` does not exist yet, so `c₂` is
the only target that can be declared before the reference filter, and declaring the other
later would be choosing the operating point with its consequences in view. `T`'s sensitivity
to `μ` is printed beside its sensitivity to `D`.

### `c₄` (OUTSTANDING)

Not yet in closed form, and harder than `c₂` for a reason worth recording: **`c₄` is not a
function of `R'/R` and `R''/R` alone.** Two exponential cases with identical values of both
ratios, differing only in `R(μ)`, return different `c₄`. So the basis needs `R(μ)` itself,
and a first fit on `{a⁴, a²b, b²}` with `a = R'/R`, `b = R''/R` leaves residuals of 24%.

The numerics are also not yet converged: two quadrature settings gave `c₄ = −0.2615` and
`−0.1941` at `μ = 1, κ = 1`, a 35% spread. `c₂` was stable to five digits under the same
change, so the instability is `c₄`'s own and has to be resolved before any number is
registered from it.

**The basis is fixed by dimensions, so the refit is a linear solve rather than a search.**
With `y = x + v` and `var(v) = R(x)`, `R` carries units of `x²`, so `1/R` has the units of
`ℓ''` and `ℓ'²`, where `ℓ = log R`. Every term in `c₄` must have units `x⁻⁴` *and* vanish
when `R` is constant, which is R1 again. That gives seven terms:

```text
ℓ'⁴,  ℓ'²ℓ'',  ℓ''²,  ℓ'ℓ''',  ℓ'''',  ℓ'²/R(μ),  ℓ''/R(μ)
```

A bare `1/R²` is excluded because it survives constant `R`. A parity check confirms the
enumeration is complete rather than merely sufficient: for symmetric `R`, counting `ℓ'` and
`ℓ'''` as odd, all seven terms are even under `x → −x`, nothing of odd parity can appear,
and nothing of even parity at that dimension is missing. This also explains the
`R(μ)` dependence found empirically rather than leaving it a curiosity, and it recasts `c₂`
as `ℓ'²/4`.

The exponential family isolates two coefficients directly: for `R = A·e^{bx}` every log-
derivative above the first vanishes, so `c₄ = α·b⁴ + ζ·b²/R(μ)`. Same `b` with different
`A` moves only `R(μ)` and gives `ζ`. A second `b` at fixed `A` then gives `α`.

**Extraction, corrected.** The gap converges to twelve digits while `c₄` moved 35%, so the
loss is in extraction rather than quadrature. Two fixes, both available now that `c₂` is
closed-form. Subtract the dominant term exactly and fit `gap(σ) − c₂σ²`. And fit at *large*
spread rather than small: as `σ → 0` the ratio `c₄σ⁴/c₂σ²` vanishes and the quartic sits
below the quadrature floor, which is exactly the setting-dependence observed. Choose `σ`
just under `σ_max`, where the quartic is a few percent of the total. The quartic is only
visible where it is nearly significant.

**Trigger.** If `c₄` is not solved by end of day two, register `T` with `c₂/|c₄|` bracketed
numerically rather than solved. Still a pre-registration, still dated before the reference
filter exists, weaker only in the tightness of the bracket.

## 3. The curvature ceiling (SETTLED 2026-08-07): vacuous for this family

The belief-smoothed rung needs `E[R(x)]` under the Gaussian prediction to exist, and H1 puts
no ceiling on how fast `R` may grow, so a family can lose a rung at high curvature. That is
a second, independent way D1 and D2 can fail to be tests (`warrant_ledger.md` section 5).

For `d4-family-v1` it does not arise. With `x ~ N(μ, σ²)`,

```text
E[R(x)] = R₀ + κ·(μ² + σ²)
```

finite for every finite `κ` and `σ²`. The rung cannot drop off the ladder on this family at
any curvature, so the second stop branch is closed by construction rather than by
observation. A family with `R` growing faster than `exp(x²/2σ²)` would reopen it, which is
one more thing a replacement family would have to re-establish.

## 4. `D`, `k_min` and the threshold `T` (OUTSTANDING, needs section 2)

The gate is stated as an **absolute threshold**, not as a factor:

```text
window edges (spread axis):  lower √(k·δ_ref / c₂),  upper √(f·c₂/|c₄|)
at exactly D decades:        k·δ_ref = f·c₂²/|c₄|·10^(−2D)  ≡  T      (see the audit)

GATE-D4 passes iff:          gap > T
D1/D2 are tests iff:         δ_ref ≤ T / k_min
reported diagnostic:         k = T / δ_ref
```

`T` contains no `δ_ref`. It is a constant of the declared family and `D`, computable before
the reference filter exists, and nothing about it moves when the filter lands. That is what
makes the gate immune to the measurement rather than merely arguably immune.

**`D` is not neutral, and this document will not claim otherwise.** Since `T ∝ 10^(−2D)`, a
larger `D` lowers the threshold monotonically, so more decades makes GATE-D4 easier *and*
D2's fit better. Both pressures push the same way and nothing bounds `D` from above. `D` is
declared on D2 fitting grounds, and it must be frozen and independently defensible for that
reason rather than because it is neutral. This registration prints `T` at `D−1`, `D` and
`D+1`, since one decade moves `T` by 100× and a reader should see that rather than infer it.

**`k_min`** is the floor below which C6/R6 stops being a test, since at `k = 1` the claim is
a bare inequality between two computed numbers. It is not derivable, but it is declarable on
grounds independent of D4.

**`D`, `f` and `k_min` are frozen at declaration**, and `f` is settled first, since it
moves the edge that `D` is counted across.

Stated explicitly, because the move available on seeing a disappointing `δ_ref` is to lower
`D`, or to loosen `f` so that `D` survives.

### AMENDMENT 2026-08-07: `T` rests on three declared parameters, and the span is 10⁴

The opening framing had `T` fixed by the family's analytic properties and `D`. That is no
longer true. `T = f·(c₂/|c₄|)·curvature·10^(−2D)` and `c₂` depends on the `μ`-rule, so three
things are declared: **`D`, `f`, and the `μ`-rule**. Each is individually defensible and all
three precede `δ_ref`, so the construction holds. Presenting one while carrying three would
not.

Joint sensitivity of the known factors `f·c₂·10^(−2D)`, with `|c₄|` still pending, over
defensible ranges `D ∈ [1.5, 3.0]`, `f ∈ [0.02, 0.2]`, and the `μ`-rule against a pinned
`μ = 1`:

| knob | range | factor on `T` |
| --- | --- | --- |
| `D` | 1.5 to 3.0 decades | **1000×** |
| `f` | 0.02 to 0.2 | 10× |
| `μ`-rule vs pinned `μ = 1` | at `κ ∈ {0.25, 1, 4}` | 1.56× at most, and exactly 1× at `κ = 1` |
| **joint** | | **≈ 1.0 × 10⁴** |

The answer to "is the construction robust" is no, and it is worth having before the
reference filter rather than after. Essentially the whole span is `D`: it moves `T` by 100×
per decade while `f` and the `μ`-rule together contribute about 16×. **`D`'s justification
carries the severity almost by itself**, and the fitting argument behind it has to be made
to that standard. `f` and the `μ`-rule are second-order by comparison.

### AMENDMENT 2026-08-07: D2 would pass for the wrong reason, and gets a second leg

Along the ridge `μ = μ*(κ)`, `c₂ = κ/(4R₀)` is linear in `κ`, so the measured behaviour is
`gap ∝ κ·σ²`. That is the battery's registered D2 prediction *verbatim*. D2 would report
`NOT TRIGGERED` on a mechanism already refuted analytically above, which is a severity
failure rather than a pass: a falsifier that cannot fire on a true refutation is not testing
anything.

**Two corrections, both registered now.**

D2's prediction is restated to cover the **σ-exponent only**, with the κ-dependence
explicitly out of scope on the ridge, since linearity there is a property of the ridge
rather than of the mechanism.

And a **second leg is added: a κ-sweep at pinned `μ`**, off the ridge. There
`c₂ = κ²μ²/(R₀+κμ²)²`, which is quadratic in `κ` at small `κ`, so the two mechanisms
separate on an axis already being swept. Measured local exponent `d log c₂ / d log κ` at
`μ = 1`:

| `κ` | 0.05 | 0.10 | 0.20 | 0.40 | 0.80 |
| --- | --- | --- | --- | --- | --- |
| exponent | 1.87 | 1.75 | 1.56 | 1.28 | 0.94 |

Approaching 2 as `κ → 0` off-ridge, against exactly 1 on it. The leg therefore needs small
`κ`, roughly `κ ≤ 0.1`, to resolve cleanly. This converts an analytic refutation into a
measured one and costs one extra sweep on an existing axis.

### AMENDMENT 2026-08-07: corrections to the sensitivity, and what actually sets `D`

**The `μ`-rule factor was understated, and understated exactly where the new leg lives.**
The ratio is `c₂(μ*)/c₂(μ=1) = (R₀+κ)²/(4R₀κ)`, which by AM-GM is `≥ 1` everywhere and
equals 1 only at `κ = R₀`, where `μ* = 1` and the rule coincides with the pin. It diverges
as `κ → 0`.

| `κ` | 0.05 | 0.10 | 0.25 | 1.00 | 4.00 |
| --- | --- | --- | --- | --- | --- |
| factor | 5.51× | 3.03× | 1.56× | 1.00× | 1.56× |

The previous amendment computed 1.56× over `κ ∈ {0.25, 1, 4}`. The D2 leg registered in the
same sitting needs `κ ≲ 0.1`, where the factor is 3.03× to 5.51×. **The joint span is
therefore nearer 5 × 10⁴, not 1 × 10⁴.** The conclusion is unchanged, `D` still carrying
essentially all of it. Worth recording is the mechanism: two amendments written in one
sitting, and one silently invalidated the other's inputs by moving the `κ` range it was
computed over.

**`T` is a single number and `c₂` is not.** Both `c₂` and `c₄` vary along the `κ` sweep, so
`σ_max` varies and the decades between the edges vary with it. `T` is not well-defined until
the `κ` is named. **Registered rule: `T` is evaluated at the `κ` minimising the window
width**, the binding cell, rather than at a reference `κ` or the sweep's midpoint. That
makes `T` the most demanding admissible value and puts the choice beyond argument. It is
derivable once `c₄` lands, so the rule is declared now and evaluated after.

### `D` is a bias argument, not a statistics argument

This is why `D` read as freely choosable. For an OLS log-log fit over `D` decades with `N`
points and relative error `ε`, the standard error on the exponent is about `ε√12/(D√N)`. The
quadrature gap converges to twelve digits, so `ε ~ 10⁻¹⁰` and the random error is nowhere
near binding. **A statistical argument gives no lower bound on `D` at all.**

What binds is systematic. The gap is not a power law: with `a = |c₄|/c₂`,

```text
gap = c₂σ²(1 − aσ²)
local log-log slope = 2 − 2aσ²/(1 − aσ²)
at the top of the window (aσ_max² = f):  2 − 2f/(1−f)
```

At `f = 0.1` the local exponent at the top is 1.778, not 2. The fitted exponent is biased
low, and `D` has to buy that bias down rather than buy down noise.

**The exact bias, from the OLS geometry.** With `v = ln σ` uniform over `L = D·ln10`, the
first-order-in-`f` result is `b = −3f/(D·ln10)²`. That envelope has the right shape and the
wrong size:

| | `D=1` | `D=2` | `D=3` |
| --- | --- | --- | --- |
| `f = 0.02` | 1.71× | 1.27× | 1.16× |
| `f = 0.10` | 1.67× | 1.24× | 1.14× |

overstating the true bias by 14% to 71%, worst at small `D`. **The exact OLS integral is
what gets registered**, not the closed form, which stands only as a scaling guide.

**`D`, `f` and D2's interval are one declaration, not three.** Declare the bias budget `β`,
how far the fitted exponent may sit from 2 through truncation alone. Then `D` follows from
`f` through the bias relation, and `f` follows from maximising `T` subject to it. Since
`T ∝ f·10^(−2D)`, the smallest `D` the budget allows is the severity-maximising choice, which
is the same shape as every other rule here.

Carrying that through gives a closed form, verified numerically at `β ∈ {0.01, 0.02, 0.05}`:

```text
f* = β/3        D* = 1/ln10 = 0.4343 decades        independent of β
```

A window of exactly `e ≈ 2.718` in spread. **This shrinks the 10⁴ span**, which assumed the
three knobs move independently when they cannot.

**An open decision, to be taken before any of the three is declared.** The optimum above
*dilutes* the truncation bias by widening the window. Since `c₄` will be known, the bias can
instead be **subtracted analytically**, which removes it rather than diluting it and
decouples `D` from `f` entirely. The two constructions give different `D`, so choosing
between them after seeing either would be choosing with the consequences in view. There is
also a tension the optimisation does not see: 0.43 decades is statistically ample given
`ε ~ 10⁻¹⁰`, and is a short range to present as a power-law fit. That is a judgement about
what a reader accepts rather than a fact the algebra supplies, and it belongs in the
declaration rather than in a footnote.

**D2's interval inherits this.** Whatever bias survives, the registered interval must
accommodate it, or the leg fires on truncation rather than on the world. Same coupling,
arriving at the registration rather than at the derivation.

### AMENDMENT 2026-08-07: `T` carries `c₂` squared, and "curvature" was the refuted mechanism's name

The lower edge is where the gap clears the error: `gap = c₂σ² = k·δ_ref`, so
`σ_min = √(k·δ_ref/c₂)`. `warrant_ledger.md` section 5 writes it as `√(k·δ_ref/curvature)`,
which is the same thing *only if* "curvature" means `c₂`. Working `T` through both edges:

```text
σ_min² = k·δ_ref/c₂        σ_max² = f·c₂/|c₄|        ratio = 10^(2D)

T  =  k·δ_ref  =  f · c₂² / |c₄| · 10^(−2D)
```

**`c₂` squared**, because it appears in both edges: once setting where the signal clears the
error, once setting where the expansion breaks. Verified numerically, `c₂·σ_min² = T` exactly.

The earlier form in section 4 carried `curvature` as a separate factor beside `c₂/|c₄|`,
inherited from the ledger. That is wrong either way. If "curvature" means `c₂` the formula
was right and misnamed after the very mechanism refuted above. If it means something
`R''`-flavoured, the lower edge inherited the falsified mechanism and is simply wrong. The
ledger's wording is audited before `T` is declared.

**This is the third instance of the tracked pattern** of mechanism wording outliving its
falsified rationale, after the epistemic-pull phrasing and the "optimal reach" row. The
first two were presentational. This one is load-bearing.

**Sensitivity, again revised.** The `μ`-rule now enters `T` squared: `5.51² = 30.4×` at
`κ = 0.05`, `9.2×` at `κ = 0.1`. Joint span `1000 × 10 × 30 ≈ 3 × 10⁵`. The line that `D`
carries the severity almost by itself needs re-checking rather than re-quoting: `D` still
dominates, and `30×` is no longer a rounding error.

### AMENDMENT 2026-08-07: the fitting argument used the wrong `ε`

`ε ~ 10⁻¹⁰` is the *quadrature's* precision. D2 measures against the reference filter, and
the lower edge is *defined* as where the gap is exactly `k` times `δ_ref`. So the relative
error on the gap at the bottom of the window is `1/k` by construction, and 10% at `k = 10`.
It improves as `σ²` across the window, reaching `1/(k·10^(2D))` at the top.

Propagating that through OLS with heteroscedastic weights, at the bias-only optimum
`D = 0.4343`:

| `k` | 5 | 10 | 30 |
| --- | --- | --- | --- |
| `σ_p` | 0.089 | 0.045 | 0.015 |

`σ_p = 0.045` at `k = 10` exceeds every `β` tested (0.01, 0.02, 0.05). **The statistical term
does not merely bind, it dominates the bias it was dismissed in favour of.** `D* = 0.4343`
answered a bias-only optimisation the measurement will not be operating under.

Re-optimising `T` subject to `√(bias² + σ_p²) ≤ β`:

| `β` | `k = 10` | `k = 30` |
| --- | --- | --- |
| 0.02 | `D* = 0.97`, `f* = 0.031` | `D* = 0.47`, `f* = 0.019` |
| 0.05 | `D* = 0.51`, `f* = 0.045` | `D* = 0.26`, `f* = 0.035` |

Both terms fall with `D`, the bias as `1/D²` and the noise as `1/D`, so the corrected optimum
is generally larger, `T` is smaller, and the construction is less severe and more honest. The
presentational worry about a 0.43-decade window largely dissolves at the values that matter,
which is why that argument is not spent here.

**A consequence to accept.** `σ_p` depends on `k`, hence on `δ_ref`, hence on the reference
filter. So `D` is registered as an **expression in `k`, evaluated conservatively at `k_min`**,
since smaller `k` means more noise and demands more decades. That is the
register-expressions-not-constants discipline arriving where a number was hoped for.

### AMENDMENT 2026-08-07: dilute-versus-subtract becomes a rule

The decision was flagged rather than taken. It converts to a criterion.

Subtraction uses `c₄` to correct the fit, so the residual bias after subtracting is `c₄`'s own
relative error times the bias. At `c₄`'s current 35% that buys a factor of about 3, not
removal. Dilution buys `1/D²` with no dependence on `c₄`'s accuracy at all.

**Registered rule: subtract if and only if `c₄`'s relative error is below `X`, dilute
otherwise.** `X` is declared now, before the refit reveals which side the result lands on.
Same move as `f*` and `μ*`: a choice that would otherwise be made with its consequences in
view becomes a rule fixed before they are visible.

### AMENDMENT 2026-08-07: the closed form does not survive the exact bias, and `D` is set by noise

The convergence check proposed for the re-optimisation was that switching the noise off
should recover the bias-only closed form `D* = 1/ln10 = 0.4343`. **It fails.**

| `k` | 10 | 30 | 100 | 1000 | 10⁶ |
| --- | --- | --- | --- | --- | --- |
| `D*` at `β = 0.02` | 0.960 | 0.436 | 0.204 | 0.150 | 0.150 |

As `k → ∞` the optimum runs to the search boundary, not to 0.4343. The agreement at `k = 30`
is a coincidence of where the noise constraint happens to bind.

**The cause.** `D* = 1/ln10` was derived from the *first-order-in-`f`* bias `3f/(D·ln10)²`,
already recorded above as overstating the true bias by 14% to 71%. At small `D` it is far
worse: at `D = 0.15` the exact integral admits `f = 0.0138` where the analytic permits
`0.0008`. Taking the largest feasible `f` at each `D` and forming `T ∝ f·10^(−2D)`:

| `D` | 0.15 | 0.434 | 1.0 | 2.0 | 3.0 |
| --- | --- | --- | --- | --- | --- |
| `T`, exact bias | 6.9e−3 | 3.3e−3 | 6.0e−4 | 1.7e−5 | 3.4e−7 |
| `T`, first-order bias | 4.0e−4 | 9.0e−4 | 3.5e−4 | 1.4e−5 | 3.2e−7 |

The first-order column has an interior peak at 0.4343. The exact column is **monotone
decreasing**, so the bias-only problem has no interior optimum at all: `T` is maximised by
shrinking the window to nothing.

**Two things follow, and the second inverts an earlier conclusion.**

`f* = β/3` and `D* = 1/ln10` are **withdrawn as results**. They stand only as a first-order
illustration, and the exact OLS integral governs.

And the bias-only problem is **ill-posed**, since a zero-width window fits no exponent.
What makes it well-posed is the noise term, which diverges as `D → 0`. So *noise*, not bias,
is what sets `D` from below. The earlier amendment concluded the opposite, and did so
resting on the same first-order approximation this one retires.

### AMENDMENT 2026-08-07: the numbers, committed before the refit

Anything the `c₄` refit could inform is fixed here, as figures rather than intents. A rule
whose deciding quantity is already on screen is not a rule.

| symbol | value | derivation |
| --- | --- | --- |
| `k_min` | **10** | the gap clears the certified error by one order of magnitude. At `k = 1` the claim is a bare inequality between two computed numbers, which is the defect the flip-margin bar was introduced to fix. One order is the smallest round separation that makes "separated from zero" mean anything |
| `β` | **0.05**, and it is a **total** budget | D2's interval must separate an exponent of 2 from the nearest alternative. The degenerate `μ = 0` case gives 4 and a cubic gap would give 3, so the half-width is 0.5. `β` is the instrument's share of it, declared at one tenth. Total, not bias-only, matching the `√(bias² + σ_p²) ≤ β` constraint the re-optimisation actually used |
| D2's interval | **2 ± 0.5** | the separation above. The instrument consumes at most 10% of it, so the leg fires on the world rather than on truncation or noise |
| `X` | **0.1** | subtraction must buy at least an order of magnitude on the bias, since its residual is `X ×` the bias. Below that, dilution's `1/D²` is competitive and carries no dependence on `c₄`'s accuracy |

At `k_min = 10` and `β = 0.05` the constraint is feasible: `D* = 0.520`, `f* = 0.0488`,
`σ_p = 0.0359`. `D` is registered as an expression in `k` evaluated at `k_min`, so this is
the conservative corner rather than a typical one.

### AMENDMENT 2026-08-07: the range the `c₄` refit fits over

"Just under `σ_max`" is circular: `σ_max = √(f·c₂/|c₄|)` needs `c₄`. Registered instead is a
**derivation range chosen purely on conditioning grounds, declared not to be the registered
window**:

```text
σ ∈ [0.06, 0.30]        quadrature converged to ≥ 10 digits throughout,
                        spanning a factor of 5 in σ and 25 in σ²
```

`f` does not enter it. What the quartic's share of the total turns out to be across that
range is reported afterwards as an outcome, never used to choose the range, so `f` cannot
leak into how `c₄` is measured while `c₄` sets the window `f` parameterises.

### DISCLOSURE 2026-08-07, before the refit: the declared convergence is not met at the bottom

The derivation range was registered with "quadrature converged to ≥ 10 digits throughout".
Measured, by refining the `x`-grid from 16001 to 32001 points at span 9σ:

| `σ` | 0.06 | 0.12 | 0.20 | 0.30 |
| --- | --- | --- | --- | --- |
| relative change | 1.8e−9 | 4.6e−10 | 1.0e−10 | 6.8e−11 |

So roughly **8.7 digits at the bottom** of the range against the 10 declared, reaching 10.2
at the top. The `y`-grid is the reverse, 8.6e−15 at `σ = 0.06` and 4.0e−10 at `σ = 0.30`, so
the binding axis swaps across the range. Recorded before the refit runs rather than after,
because a precondition disclosed once the result is visible is not a precondition.

**What it propagates to.** At `σ = 0.06` the quartic residual after subtracting `c₂σ²` is
`2.58e−6`, which is 0.288% of the gap. A relative gap error of 1.8e−9 is `1.62e−12`
absolute, so the error carried onto `c₄` is

```text
1.62e−12 / 2.58e−6  =  6.3e−7  relative
```

Seven orders below the 0.1 the dilute-versus-subtract rule turns on, and five below the 1%
that would matter for `T`. The shortfall is real and inconsequential, and both halves of
that are stated rather than the convenient one.

The declaration is **not** amended to match the measurement. It stands at 10 digits with the
shortfall recorded against it, since rewriting a precondition to fit what was achieved is
the move the document exists to prevent.

### RESULT 2026-08-07: `c₂ > 0` is an analytic positivity statement at leading order

`c₂ = (ℓ'(μ)/2)² ≥ 0`, with equality only at a stationary point of `R`. C6 already claims
positivity "by theorem" and needs the reference filter to *certify the separation* rather
than to observe a sign, so this does not move C6 off the filter. It is more than the
existing claim in one respect: it supplies the leading coefficient analytically, where
before there was a sign and a numerical magnitude.

The caveat is load-bearing and keeps it from reaching further. `c₄ < 0` means the truncated
expansion turns over, so what is established is positivity in a **neighbourhood of zero
spread**, not globally, and C6 measures at finite spread where exactly that validity is in
question. Recorded because a Tier-A-flavoured statement sitting inside an afternoon's
algebra is better found now than in draft.

### RESULT 2026-08-10: the `c₄` refit, and the rule fires for subtraction

Run over the declared range `σ ∈ [0.06, 0.30]`, extracting by exact `c₂σ²` subtraction, on
28 cases across four `R` families (quadratic, exponential, `tanh`, `sin`) with `μ` and the
shape parameters varied. Log-derivatives taken by autodiff rather than by hand.

**The seven-term dimensional basis holds.** Median relative residual **0.60%**, max 5.59%,
full rank, condition 109. The `(a, b)`-only basis it replaced left 24%, so the dimensional
argument is what fixed it.

| term | coefficient |
| --- | --- |
| `ℓ'⁴` | +0.43231 |
| `ℓ'²ℓ''` | −0.25874 |
| `ℓ''²` | +0.12985 |
| `ℓ'ℓ'''` | +0.24935 |
| `ℓ''''` | **−0.00068** |
| `ℓ'²/R` | −0.74537 |
| `ℓ''/R` | **+0.00192** |

**Two coefficients came out at zero.** `ℓ''''` and `ℓ''/R` sit at 0.1% and 0.3% of the
largest, so the operative basis is five terms rather than seven. Dropping them costs little
in the median (0.60% → 0.70%) and more in the worst case (5.59% → 15.48%), so they are
reported as consistent with zero rather than deleted.

**The coefficients look like simple fractions, and that is not established.** Against
`{7/16, −1/4, 1/8, 1/4, −3/4}` the fitted ratios are 0.991, 1.011, 1.056, 1.009, 0.995.
Pinning them improves the median to 0.15% and degrades the max to 32.7%, which is the
signature of a hypothesis that is close but not exact at this precision. `ℓ'ℓ''' = 1/4` and
`ℓ'²/R = −3/4` are the strongest. Settling them needs an analytic derivation rather than a
finer fit, and nothing downstream depends on it.

**At the declared operating point** (`κ = 1`, `μ* = 1`): `c₂ = 0.250000`, `c₄ = −0.18980`,
so `a = |c₄|/c₂ = 0.759`. The gap turns over at `σ = 0.812`, and `σ_max = 0.254` at the
`f* = 0.0488` that `β = 0.05, k_min = 10` implies — inside the derivation range, so the
extraction did not have to extrapolate into the window it parameterises.

**The 35% spread is resolved.** Jackknifing the range gives an extraction spread of 0.36%
at that point, against 35% before. The earlier diagnosis was right: the loss was extraction,
not quadrature, and exact `c₂` subtraction plus a range chosen for conditioning fixed it.

### DECISION 2026-08-07: the rule fires for **subtract**

`c₄`'s relative error at the operating point is 0.36% by extraction and 1.03% by the basis
fit. Both are far below the registered `X = 0.1`, so **the truncation bias is subtracted
analytically rather than diluted**. The rule was declared and committed before the refit
ran, and it decided the branch without a judgement being made with the answer visible.

**Two consequences, one of them an open question the decision creates.**

Subtraction removes the bias term from the `D` optimisation, so `D` is set by the noise
constraint `σ_p(k, D) ≤ β` alone. That is consistent with the finding above that noise is
what makes the problem well-posed, and it simplifies `D` to a single-term condition.

And it moves the upper edge. `σ_max` was defined as where the *quartic* reaches a fraction
`f` of the quadratic. With the quartic subtracted, the window extends to wherever `c₆`
starts to bite, which is **unmeasured**. So `f` no longer parameterises the binding
truncation, and either `σ_max` is redefined against `c₆` or `f` is retained as a
conservative bound that subtraction makes slack. That is registered as open rather than
resolved, since choosing now would be choosing with the refit's outcome in view.

`T` is still pending: the registered rule evaluates it at the `κ` minimising the window
width, which needs `c₄` scanned along the ridge rather than at the single point above.

### DISCLOSURE 2026-08-19: the subtract decision carries the declaration's date

The DECISION entry above is dated three days before the measurement it rests on. It
reports `c₄`'s relative error as 0.36% by extraction, and that jackknifed spread is
established in `### RESULT 2026-08-10: the c₄ refit, and the rule fires for subtraction`.
An entry recording that a rule fired cannot predate the run that fired it.

The history says so directly. Commit `343f812`, dated 2026-08-10, added the RESULT entry
and the DECISION entry in the same commit, and the DECISION was written with a 2026-08-07
heading. Anyone can check that with `git log -S "the rule fires for **subtract**" --
research/gate_d4_registration.md`.

The 1.03% basis-fit figure the decision quotes beside it is not stated in that RESULT
entry either, which reports the basis residual as 0.60% median and 5.59% max. In this
document the figure appears only in the decision itself and in RESULT 2026-08-16, which
takes it as given. Its provenance is not recorded here. The dating argument does not rest
on it: the 0.36% extraction spread alone postdates the entry's heading.

The pre-registration is not in doubt. The rule is registered in `### AMENDMENT 2026-08-07:
dilute-versus-subtract becomes a rule`, added by commit `c35ec2c`, and its threshold
`X = 0.1` in `### AMENDMENT 2026-08-07: the numbers, committed before the refit`, added by
commit `fc8aac4`. Both commits are dated 2026-08-07, genuinely earlier than the 08-10
refit, so the branch was decided against a bar set before the answer was visible. That is
what the entry claims about itself and it holds.

What is wrong is the date on the entry recording the firing, which should be the refit's
rather than the declaration's. The heading and its text stay as written.

## 5. D1's resolution threshold (OUTSTANDING): an expression, not a value

Registered as a **propagated expression**, not a constant. A formula fixed before the
measurement has no knobs left to turn.

The four rungs are all scored against one reference filter, so `δ_ref` is common-mode and
largely cancels in their differences. The error on a difference is therefore far smaller
than `δᵢ + δⱼ`, and the sum-of-bars rule is safe but understates the test
(`warrant_ledger.md` section 4). What gets registered is the residual differential
sensitivity, derived and written down before any rung is evaluated.

## 6. D2's exponent interval (OUTSTANDING): derived, not chosen

"A pre-registered interval around 2" leaves the width unstated, which is the same
choice-versus-rule problem sections 4 and 5 exist to remove. The width follows from the fit
design already being declared: `D` decades, a declared point count and a declared residual
model give a standard error on the fitted exponent, and the interval is a stated multiple of
it. `D` then does double duty.

Also registered here: the two failure modes the battery already calls *boring* and excludes
from the agreement criterion in advance.

## 7. `c₄` in closed form (RESULT 2026-08-16), and the disclosure that it ran first

### DISCLOSURE 2026-08-16: the derivation ran before this entry existed

Recorded first, because it is the part a reader has to be able to audit and because the
sequence cannot be repaired by describing it well.

The symbolic pipeline (`research/checks/series_kernel.py`, `research/checks/gap_series.py`)
was extended to `σ⁴` and run **before** any amendment registered what the run would
produce or what would count as a failure. In order, on 2026-08-16:

1. The exact predictive nesting was added to the kernel.
2. `c₄` was computed symbolically at `σ⁴`, on free `ℓ₁..ℓ₄` and a symbolic `R̄`.
3. It was evaluated at the declared operating point, giving `−3/16`.
4. It was compared against the `−0.18980` of RESULT 2026-08-10.
5. `gap_expansion --check --families quadratic --c4 -0.1875` was run, and did not refute.

Steps 2 to 5 have no prior registration naming them. Everything below in this section that
is labelled a *result* is therefore post-hoc in its scheduling, whatever its content, and
should be read as such.

### What was registered before it, and is therefore predicted rather than fitted

Four statements were in this file with earlier dates, and the derivation was in a position
to contradict every one of them:

- **The seven-term dimensional basis** (section 2, 2026-08-07), with its parity argument
  for completeness. The derivation had to land inside that span or refute it.
- **The two coefficients consistent with zero** (RESULT 2026-08-10): `ℓ''''` at 0.1% and
  `ℓ''/R` at 0.3% of the largest term, reported as consistent with zero rather than
  deleted.
- **The simple-fraction hypothesis** (RESULT 2026-08-10): `{7/16, −1/4, 1/8, 1/4, −3/4}`,
  against which the fitted ratios were 0.991, 1.011, 1.056, 1.009 and 0.995. That entry
  says in terms that the fractions are "close but not exact at this precision" and "not
  established".
- **The exponential-family reduction** (section 2): for `R = A·e^{bx}`, exactly two of the
  seven terms may survive, `c₄ = α·b⁴ + ζ·b²/R(μ)`.

### RESULT 2026-08-16: the closed form

```text
c₄  =  7ℓ₁⁴/16  −  ℓ₁²ℓ₂/4  +  ℓ₂²/8  +  ℓ₁ℓ₃/4  −  3ℓ₁²/(4R̄)
```

Against the seven-term basis and the fit of RESULT 2026-08-10:

| term | fitted 2026-08-10 | derived 2026-08-16 | registered hypothesis |
| --- | --- | --- | --- |
| `ℓ'⁴` | +0.43231 | **7/16** = +0.4375 | 7/16 |
| `ℓ'²ℓ''` | −0.25874 | **−1/4** | −1/4 |
| `ℓ''²` | +0.12985 | **1/8** | 1/8 |
| `ℓ'ℓ'''` | +0.24935 | **1/4** | 1/4 |
| `ℓ''''` | −0.00068 | **0** | consistent with zero |
| `ℓ'²/R` | −0.74537 | **−3/4** | −3/4 |
| `ℓ''/R` | +0.00192 | **0** | consistent with zero |

Every registered prediction holds. The five fractions are the hypothesised ones, the two
zeros are identically zero, the span is the declared basis with nothing left over, and the
exponential family keeps exactly the two terms section 2 said it could, with `α = 7/16` and
`ζ = −3/4`.

**At the declared operating point** (`κ = 1`, `μ* = 1`, so `R̄ = 2`, `ℓ₁ = 1`, `ℓ₂ = 0`,
`ℓ₃ = −1`): `c₄ = −3/16 = −0.1875` exactly, against the fitted `−0.18980`. **They disagree
by 1.2%**, which is 3.4 times the 0.36% jackknifed extraction spread and above the 1.03%
basis-fit error. The derivation supersedes the fit, and the fit's residual error is now
explained rather than absorbed: it was a `COMPUTED` extraction of a quantity whose true
value
is a small rational.

**The exact predictive is what makes it work, and this is the sharpest thing in the
section.** Averaging over a leading-order `N(0, R̄)` predictive instead of the exact
`ν = σz₁ + √R̄·e^{δ/2}·z₂` gives

```text
c₄(leading order)  =  −3ℓ₁⁴/16  −  ℓ₁²ℓ₂/2  +  ℓ₂²/8  +  ℓ₁ℓ₃/4  −  5ℓ₁²/(4R̄)
```

which is `−1.0625` at the operating point, 5.7 times the derived value and of no
resemblance to the fit. `predictive_truncation` had already recorded that `p*` is a scale
mixture with exponential tails and that no Gaussian stands in for it at any variance. This
is that finding biting a coefficient.

**The refutation attempt that has been run.** `gap_expansion`'s G4b subtracts a candidate
and reads the residual's exponent: a candidate wrong by any amount leaves the `σ⁴` term
surviving and the exponent near 4. On the quadratic family it returned **σ^5.982** against
a predicted `σ⁶`, bar 0.25, and did not fire. Without a candidate the same cells return
σ^4.038.

### Why a tuned derivation is not available as an explanation

The claim a reviewer should test is not that we avoided looking at the fit. It is that the
derivation could not have been aimed at it:

- **The symbolic path contains no numbers.** `series_kernel.py` and `gap_series.py` carry
  no floats, no family, and no value for `R̄`. `ℓ₁..ℓ₄` are free symbols throughout. There
  is no quantity in either module that a fitted `−0.18980` could have been substituted
  into. This is checkable by reading the two files, and by the checks that fail if `R̄` is
  ever set to one.
- **It disagrees with the fit.** A derivation steered toward `−0.18980` would arrive at
  `−0.18980`. It arrives at `−3/16`, outside the fit's own tightest declared bar.
- **It was refutable and was put at risk.** The two zeros, the basis span and the
  exponential-family reduction were all registered earlier and could each have come out
  wrong.
- **The falsifier is independent of the derivation.** G4b reads a residual *exponent* from
  quadrature and never sees the symbolic path.

What none of that repairs is the scheduling in the disclosure above. The correct reading is
that the content is strong and the ordering is weak, and the ordering is fixed for the
remaining families below rather than argued away.

### PRE-REGISTRATION 2026-08-16: the three families not yet run

Registered **before** the runs, so what follows is a genuine out-of-sample test. The
derived formula is evaluated on the other declared families of
`research/checks/gap_kernel.py`, all at `μ = 1`. Only the quadratic row has been measured.

| family | `ℓ₁` | `ℓ₂` | `ℓ₃` | `R̄` | predicted `c₂` | predicted `c₄` | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `1 + x²` | 1.000000 | 0.000000 | −1.000000 | 2.00000 | +0.2500000 | **−0.1875000** | run, not refuted |
| `exp(x)` | 1.000000 | 0.000000 | 0.000000 | 2.71828 | +0.2500000 | **+0.1615904** | not run |
| `1.5 + 0.5 tanh(x)` | 0.111648 | −0.182526 | 0.225000 | 1.88080 | +0.0031163 | **+0.0061107** | not run |
| `1.5 + 0.5 sin(x)` | 0.140650 | −0.238832 | −0.042657 | 1.92074 | +0.0049456 | **−0.0007420** | not run |
| `2 (fixed)` | 0 | 0 | 0 | 2.00000 | 0 | **0** | not run |

**The prediction has teeth.** `c₄` changes sign between `1 + x²` and `exp(x)`, so a formula
that merely fitted the quadratic family cannot survive the exponential one. `tanh` and
`sin` sit three orders lower and also differ in sign from each other.

**What counts as a pass, declared now.** G4b's residual exponent within 0.25 of 6, the bar
already in `gap_expansion`. Anything at or below 5 on a family whose quartic is resolvable
refutes the closed form.

**What counts as VOID rather than a pass, also declared now.** For `1.5 + 0.5 sin(x)` the
predicted `c₄` is 0.15% of `c₂`, so on the declared `σ` grid the quartic term sits near the
quadrature floor and `c₆` may dominate the residual before the quartic does. If the
measured residual is at or under the floor, that cell is **VOID on resolving power**, not a
pass and not a refutation. The same escape is *not* available to `1 + x²` or `exp(x)`,
where the quartic is percent-level.

**The fixed-`R` row is a falsifier of the instrument, not of the formula.** The gap is
identically zero there, so a non-zero `c₄` would be a bug in the pipeline.

### RESULT 2026-08-16: the out-of-sample runs, three passing and one firing

Run against the table registered above, with no value changed after it was written.

| family | predicted `c₄` | G4a, no candidate | G4b, candidate subtracted | outcome |
| --- | --- | --- | --- | --- |
| `1 + x²` | −0.1875000 | σ^4.038 | **σ^5.982** | not refuted |
| `exp(x)` | +0.1615904 | σ^3.976 | **σ^5.984** | not refuted |
| `1.5 + 0.5 tanh(x)` | +0.0061107 | σ^4.000 | **σ^6.302** | **FIRED**, 0.302 against a 0.25 bar |
| `1.5 + 0.5 sin(x)` | −0.0007420 | σ^3.965 | **σ^6.004** | not refuted |
| `2 (fixed)` | 0 | — | — | gap identically zero, nothing to test |

**The exponential is the load-bearing pass.** Its `c₄` has the opposite sign to the
quadratic's, so a coefficient reverse-engineered from the quadratic family could not have
produced it. It lands 0.016 from the predicted exponent.

**`sin` did not need the escape it was granted.** The pre-registration allowed it to come
back VOID on resolving power, its quartic being 1.4% of the quadratic at the top of the
grid. It returned σ^6.004, the closest of the four, so the escape went unused.

**`tanh` fires, and the registered outcome is that it fires.** The bar was declared at
±0.25 and the cell returned 0.302. No VOID escape was declared for this family, and
inventing one now is the move this section exists to prevent.

Two things are true about it and neither is a defence. The deviation is in the
**over-cancellation** direction: a wrong `ĉ₄` leaves the quartic surviving and pulls the
exponent toward 4, and 6.302 is on the far side of 6, not the near side. And the
pre-registration defined refutation as an exponent at or below 5, which this is not. So the
closed form is not refuted by this cell, and the cell is also not a pass. It is a fired
falsifier with an unexplained direction.

### PRE-REGISTRATION 2026-08-16: the diagnostic for the `tanh` fire

Declared before the diagnostic is written or run.

**What this cannot do.** It cannot convert the fired cell into a pass. The tanh outcome
above is fixed and stays fired whatever follows. A diagnostic that could revise its own
target would be the window-shopping this whole section exists to refuse. What it can do is
classify the fire as attributable or unexplained, which is a different claim and is recorded
as one.

**Why not the obvious test.** The natural move is to refit on the `σ` cells whose G3
quadrature certification did not fire. On `tanh` that leaves two of six cells, and a
two-point log-log slope has no residual and no power to say anything. Choosing the window
after seeing the result is also the failure mode being guarded against. So the window is not
touched.

**The diagnostic: leave-one-out exponent stability (G4c).** Refit G4b's exponent six times,
each time dropping one `σ` cell from the declared grid `(0.02, 0.025, 0.03, 0.035, 0.04,
0.05)`. Report the spread, `max − min`, across the six refits. The full grid is used every
time except for the single omission, so no window is selected.

**Declared in advance, before the diagnostic exists:**

- **Spread above 0.25 on `tanh`** — the exponent is not stable on this grid for this family,
  and the fire is **attributed to grid instability**. The cell stays fired, and it is
  recorded as a statement about the measurement rather than about the closed form.
- **Spread at or below 0.25 on `tanh`** — the exponent is stable and the deviation is real.
  The fire stands as an **unexplained discrepancy** against the closed form, and it is
  carried as such into anything that quotes `c₄`, until a further registered test resolves
  it.
- **The control.** The same diagnostic runs on `1 + x²`, `exp(x)` and `1.5 + 0.5 sin(x)`. If
  their spreads are also above 0.25, the diagnostic is measuring the grid rather than the
  family and is **uninformative on all four**, including `tanh`. That outcome is registered
  here as a real possibility rather than treated as a failure of the test.

**One weakness in the record, stated rather than hidden.** This pre-registration and the
result below land in the same commit, so a reader cannot confirm the ordering from the git
history the way they can for the entries above. The ordering rests on this document's own
account. Splitting the commit would fix that and was offered.

### RESULT 2026-08-16: the diagnostic does not rescue the `tanh` cell

`G4c` implements the rule above in `research/checks/gap_expansion.py`. It runs only when a
candidate is supplied, so the no-candidate baseline is unchanged.

| family | G4b exponent | G4c leave-one-out spread | range |
| --- | --- | --- | --- |
| `1 + x²` | σ^5.982 | 0.008 | σ^5.979 to σ^5.986 |
| `exp(x)` | σ^5.984 | 0.007 | σ^5.981 to σ^5.988 |
| `1.5 + 0.5 tanh(x)` | **σ^6.302** | **0.018** | σ^6.290 to σ^6.307 |
| `1.5 + 0.5 sin(x)` | σ^6.004 | 0.005 | σ^6.001 to σ^6.007 |

**The control holds, so the diagnostic is informative.** All four spreads sit far below the
0.25 bar, so the exponent is a property of the residual rather than of the grid, and the
registered "uninformative on all four" branch does not fire.

**The registered reading, applied.** `tanh`'s spread is 0.018. That is the branch declared
as *spread at or below 0.25*, so by the rule written before the diagnostic existed:

> the exponent is stable and the deviation is real. The fire stands as an **unexplained
> discrepancy** against the closed form, and it is carried as such into anything that
> quotes `c₄`, until a further registered test resolves it.

Dropping any single `σ` cell moves `tanh`'s exponent by at most 0.018 and never brings it
within 0.25 of 6. The deviation is not an artefact of one cell, and it is not attributable
to the four cells whose G3 certification fires on this family.

**What `c₄` may now be claimed to be.** Derived in closed form, agreeing with the registered
basis and both predicted zeros, and surviving three of four out-of-sample refutation
attempts. On `1.5 + 0.5 tanh(x)` the residual after subtraction does not scale as the
structure predicts, stably, and that is unexplained. Any write-up quoting `c₄` carries the
three passes and this failure together, not a summary of them.

**A hypothesis, registered as a hypothesis and not as an explanation.** An exponent above 6
is over-cancellation, so `c₄` surviving at 4 is not the candidate. The residual after
`c₂σ² + c₄σ⁴` is governed by `c₆`, and a `c₆` anomalously small for this family would let
`c₈` compete inside the grid and lift the local slope. This is testable: `gap_expansion`
already takes `--c6` and shifts the predicted exponent to 8 when given one, so deriving
`c₆` at `σ⁶` and re-running would decide it. That derivation does not exist, this paragraph
is not evidence, and the `tanh` cell stays fired until such a test is registered and run.

### AMENDMENT 2026-08-17: the fired cell was measured with a rounded candidate

The `tanh` row above reads `σ^6.302` with a leave-one-out spread of 0.018. Both numbers
belong to a `--c4` of `0.0061107`, the five-significant-figure value this document tabled,
not to the closed form the derivation produced. Re-measured on the same six cells with the
same code:

| `--c4` supplied for `tanh` | G4b exponent | G4c spread | smallest residual |
| --- | --- | --- | --- |
| `0.0061107` (the value tabled above) | σ^6.302 | 0.018 | 6.4× the quadrature floor |
| `0.00611074` (six significant figures) | σ^6.133 | 0.069 | 7.7× the floor |
| `0.0061107361819873` (the closed form) | σ^6.148 | 0.062 | 7.6× the floor |

**The mechanism, and why it reaches this family alone.** The rounded candidate differs from
the closed form by `3.6e-8`. Multiplied by `σ⁴` at the bottom cell that is `5.8e-15`, against
a residual there of `3.8e-14`, so the rounding accounts for about 15% of the smallest
residual on the grid. That cell carries the leverage in a log-log slope. It bites `tanh`
because only there does the residual sit within an order of magnitude of the quadrature
floor: at full precision the smallest residual is 7.6× the floor.

**The exact candidates, so this cannot recur silently.** All four are the closed form
evaluated at `μ = 1`, to seventeen significant figures:

| family | `--c4` |
| --- | --- |
| `1 + x²` | `-0.18750000000000000` |
| `exp(x)` | `0.16159041912141826` |
| `1.5 + 0.5 tanh(x)` | `0.0061107361819873193` |
| `1.5 + 0.5 sin(x)` | `-0.00074203206428907733` |

One family per invocation, since a single `--c4` applied to a run of several mis-tests all
but one:

```text
uv run --no-sync python -m research.checks.gap_expansion --check \
    --families tanh --c4 0.0061107361819873193
```

**What this does and does not change.** The recorded outcome stands: the `tanh` cell fired,
and it is not un-fired by re-running it with better inputs after seeing that it fired. That
is the move this section exists to prevent, and it applies to a rounding the registrant made
as much as to a threshold. What changes is what may be claimed about *why* it fired. The
0.302 deviation is no longer attributable to the residual alone, because 0.154 of it is
attributable to the candidate's precision. The remaining 0.148 at full precision sits inside
the ±0.25 bar, so a run registered at full precision would not have fired this cell. That
is a statement about a run nobody registered.

Any write-up quoting the fire carries this amendment with it. Quoting `σ^6.302` without the
candidate that produced it is quoting a number that cannot be reproduced from the closed
form.

### PRE-REGISTRATION 2026-08-17: candidate precision, before any exponent cell is read again

Registered before the next exponent measurement of any kind, and before `c₆` exists.

**The precondition.** A `--c4` or `--c6` candidate is supplied at no fewer than fifteen
significant figures, and the digits used are recorded beside the cell they produced. A cell
measured with a rounded candidate is **VOID**, not passed and not fired.

**The rule that makes it checkable.** Before an exponent is read, the run states
`|Δcandidate| · σ⁴` at the smallest `σ` on the grid against the residual there. Above 1% of
that residual the candidate is not precise enough for the grid and the cell is VOID. This
is arithmetic on numbers the run already has, and it is decidable before the exponent is
looked at.

**Why it is registered rather than applied now.** Applying it to the `tanh` cell would
convert a fired falsifier into a VOID one on the strength of a defect found after the fire.
The cell keeps its recorded outcome. The precondition binds what comes next.

### AMENDMENT 2026-08-17: what the leave-one-out diagnostic licenses

The RESULT above reads the diagnostic as showing "the exponent is a property of the residual
rather than of the grid", and as showing the deviation "is not attributable to the four cells
whose G3 certification fires on this family". Both are wider than the diagnostic supports.

**Dropping one cell at a time tests one cell at a time.** Every refit still contains at
least three of the four uncertified cells, so a joint effect of those four cannot appear in
the spread. The pre-registration declined a two-cell refit for want of power, and the
conclusion drawn is the one that refit would have supported.

**A grid-wide effect is invisible to it, by construction.** An effect present in every
cell, whether the grid's range, its spacing, or a bias shared across it, moves every refit
together and reports as a small spread. From here that is indistinguishable from a stable real deviation.

**Narrowed to what it licenses:** no single `σ` cell accounts for the deviation. The joint
effect of the four uncertified cells is untested and stays open, and so does the grid as a
whole. Deciding either needs a different grid or a multi-cell refit, neither of which is
this diagnostic.

### AMENDMENT 2026-08-17: the closed form is a reverse-KL coefficient

The boxed `c₄` in RESULT 2026-08-16 carries no direction. It is the **reverse** direction,
`KL(q ‖ p)`, which is the one section 2 specifies and the one the quadrature implements.

`c₂` is direction-free, and that is checked at `σ²` and asserted no further, because `κ₃`
separates the two directions above it. `c₄` is therefore direction-specific. Under forward
KL the `ℓ₁⁴` term changes sign:

```text
reverse:  c₄  =   7ℓ₁⁴/16  −  ℓ₁²ℓ₂/4  +  ℓ₂²/8  +  ℓ₁ℓ₃/4  −  3ℓ₁²/(4R̄)
forward:  c₄  =  −3ℓ₁⁴/16  +  ℓ₁²ℓ₂/4  +  ℓ₂²/8  +  ℓ₁ℓ₃/4  −  3ℓ₁²/(4R̄)
```

At the declared operating point that is `−3/16` reverse against `−13/16` forward. Every site
stating the closed form carries the word "reverse" inside the box, so the two cannot be
confused by a reader who arrives at the formula without the section around it.

### AMENDMENT 2026-08-17: the anti-circularity claim, restated to what is true

Section 7 says the symbolic modules contain "no floats, no family … `ℓ₁..ℓ₄` are free
symbols throughout". The substance holds and the letter does not. `gap_series` hard-codes
the five conjectured fractions in order to check against them, and `check_exponential_family`
substitutes `ℓ₂ = ℓ₃ = ℓ₄ = 0` to specialise to `R = A·e^{bx}`. Both live in checks that
read the derived coefficient, never in the path that computes it.

**Restated, stronger and checkable:** no fitted number enters the path that computes `c₄`.
That path is `averaged_gap`, `quartic_coefficient` and `basis_coefficients`, and a reader
can confirm it by reading those three functions. Everything downstream of them compares the
result against something. Nothing upstream of them has a number in it.

**And the `R̄` sentence.** `R̄` is never given a **numeric value** anywhere in the symbolic
path. It is reparameterised as `1/inverse` in two places for polynomial manipulation, which
is a change of variable and not a value. That is the precise claim, and it is what makes a
check that would still pass at `R̄ = 1` a check that has lost a variable.

### AMENDMENT 2026-08-17: three guards on the instrument, and one disclosed late

None of these revises a recorded outcome. The runs above used one family on the declared
grid, which is what the guards now require, so all four reproduce unchanged: `σ^5.982`
spread 0.008, `σ^5.984` spread 0.007, `σ^6.302` spread 0.018, `σ^6.004` spread 0.005. The
guards make future misuse visible. They do not re-read anything.

**One family per `--c4`.** `--families` defaults to all five and each has its own `c₄`, so
a single candidate across a run tests one family and mis-tests the rest. The failure is
not quiet in the harmless direction: the wrong candidate leaves the quartic surviving, G4b
reads an exponent near 4 and fires, and G4c reports a spread near zero because every refit
is wrong the same way. That prints as a stable real deviation. `gap_expansion` now refuses
the combination at the parser.

**G4c reports VOID off the declared grid.** The readings registered for this diagnostic
were declared against `EXPANSION_SIGMAS` literally. `_stability_report` never saw that
tuple and would happily interpret a spread from any `--sigmas`. It now returns
`NOT_APPLICABLE` naming both grids, since the registered rule does not interpret a number
from a grid it did not declare.

**The control is now a report rather than a sentence.** The pre-registration's
"uninformative on all four" branch existed only in prose, with no cross-family aggregation
in code to evaluate it. `G4c control` emits once per run and fires when every family that
produced a spread read unstable.

**Disclosed late: the four-cell minimum.** `_stability_report` requires four cells above
the quadrature floor before it will report a spread. That guard was written with the
implementation, not before it, so it is a rule this document did not declare in advance.
It is kept, because a leave-one-out spread over three cells is not a measurement, and it
is recorded here as an undeclared guard rather than presented as a registered one. It did
not bind on any run above: all four families supplied six cells.

### AMENDMENT 2026-08-17: the commit record for section 7, and what a rebase did to it

Section 7's entries assert their own ordering. Where that ordering is checkable from the
git history, this is the map. Identities are as they stand on `65-gap-series-c4` after the
rebase described below.

| claim | first registered | measured or proved |
| --- | --- | --- |
| `c₄` closed form, and the disclosure that it ran first | `bfe98d8` 2026-08-16 | `bfe98d8` 2026-08-16 |
| the three out-of-sample families, bar and VOID escape declared | `bfe98d8` 2026-08-16 | `bfe98d8` 2026-08-16 |
| G4c's rule and its three readings | `8e29a0e` 2026-08-16 | `8e29a0e` 2026-08-16 |
| the rounded-candidate disclosure | `ff1e622` 2026-08-17 | `ff1e622` 2026-08-17 |
| candidate precision as a precondition | `ff1e622` 2026-08-17 | binds the next cell, unmeasured |
| the three instrument guards | this commit | this commit |

**Three rows register and measure in the same commit, which is the weakness already stated
above for G4c.** It applies to the `c₄` result and the out-of-sample table as well. For
those the ordering rests on this document's account rather than on the history, and the
structural argument carries it instead: the symbolic path contains no fitted number, which
a reader checks by reading it rather than by trusting a date.

**A rebase changed the hashes and nothing else.** `65-gap-series-c4` was rebased onto the
warrant-vocabulary work on 2026-08-17. Author dates survived it: `8e29a0e` is authored
2026-08-16 and committed 2026-08-17, and every replayed commit carries the same split.
Commit order survived it too. The ordering above rests on that sequence and those author
dates, so nothing about what was known when is weakened by it. Only the identities moved,
and the pre-rebase ones are reachable from no branch, so a hash taken from an earlier draft
is re-read against the table above.

### AMENDMENT 2026-08-18: where the check modules live on disk

The suites moved to `research/src/research/checks/`. This section cites four of them by
filesystem path: `series_kernel.py` and `gap_series.py` under section 7's symbolic
pipeline, `gap_kernel.py` under the quadrature rows, and `gap_expansion.py` under G4c.
Those paths were written against `research/checks/` and are read through this entry.

The import path is unchanged. Every one of them still runs as
`python -m research.checks.<module> --check`, which is the form CI pins and the form a
reader should use. Nothing about a registered count, a bar, a tolerance or a measured
number is touched by the move: `research/` became a package of its own so the modules are
type-checked and installed rather than resolved off `sys.path` (ADR-039).

## Stop conditions (DECLARED 2026-08-07)

Two branches, disjoint, split by which half of the instrument failed.

**A third branch was retired rather than left standing.** `c₂ ≤ 0` was registered when
`c₂`'s sign was open. `c₂ = (ℓ'/2)²` is a perfect square, so it is non-negative identically
and vanishes only at `ℓ'(μ) = 0`, which the `R'(μ) ≠ 0` precondition now catches upstream at
declaration. The branch did not fail. It collapsed into the precondition, and a row that can
never fire reads as a check when it is not one.

| branch | evaluable | report |
| --- | --- | --- |
| the curvature ceiling binds inside the swept range | analytically | the belief-smoothed rung drops off the ladder, and D1 reports on three rungs or `VOID`. **Closed for `d4-family-v1` by section 3** |
| `δ_ref > T / k_min` | after the reference filter | the filter is not accurate enough for D1 and D2 to be tests at `D` decades. `VOID` on unmeasurability, **not** a gate `FAIL` |

The first two are family-side. The third is filter-side and is the likelier class.

There is no family-only way to empty the window: as `δ_ref → 0` the lower edge goes to zero
and the window has unboundedly many decades, so with `c₂ > 0` and finite curvature the
family alone cannot empty it. Emptiness is always the third row.

**The response to an empty window is not "pick another family"** unless the replacement and
its reason were declared first. Otherwise the stop condition becomes the contamination route
it exists to close.

## What was known when

| date | fixed | not yet known |
| --- | --- | --- |
| 2026-08-07 (opened) | the family and its version, the curvature ceiling being vacuous, the three stop branches, and the gate's form as `gap > T` | `c₂`, `c₄`, `D`, `k_min`, `T`, R6's signal, `δ_ref` |
| 2026-08-07 (amended) | the quadrature method replacing one that computed zero, and `c₂ = (R'(μ)/2R(μ))²` | `c₄`, `D`, `k_min`, `T`, R6's signal, `δ_ref` |
| 2026-08-07 (amended again) | `σ_max = √(f·c₂/\|c₄\|)` replacing an edge that sat past the gap's turnover, `μ` by the argmax rule `μ* = √(R₀/κ)` rather than by value, and `c₄`'s seven-term dimensional basis | `c₄`, `f`, `D`, `k_min`, `T`, R6's signal, `δ_ref` |
| 2026-08-07 (fourth) | `μ` reclassified as *derived*, `R'(μ) ≠ 0` as a fixture precondition, `T`'s three declared parameters, and D2's second leg off the ridge | `c₄`, `f`, `D`, `k_min`, `T`, R6's signal, `δ_ref` |
| 2026-08-07 (fifth) | the `μ`-factor correction, `T` evaluated at the binding `κ`, and `D` as a bias argument | `c₄`, `β`, `f`, `D`, `k_min`, `T`, R6's signal, `δ_ref` |
| 2026-08-07 (sixth) | `T = f·c₂²/\|c₄\|·10^(−2D)` with the "curvature" naming audited, the noise term set by `k` rather than by quadrature, `D` as an expression in `k`, and the dilute-versus-subtract rule | `c₄`, `X`, `β`, `f`, `D`, `k_min`, `T`, R6's signal, `δ_ref` |

| 2026-08-15 | `c₂ = ℓ₁²/4` as a **symbolic identity** rather than a verified closed form, Prover 2 with a `SymbolicReduction` | `c₄`, `X`, `β`, `f`, `D`, `k_min`, `T`, R6's signal, `δ_ref` |
| 2026-08-16 | `c₄` in closed form, all seven basis coefficients, and the quadratic family's refutation attempt. Recorded in section 7 **after** the run, which section 7 discloses | `X`, `β`, `f`, `D`, `k_min`, `T`, R6's signal, `δ_ref`, and the three unrun families |
| 2026-08-16 (later) | the three out-of-sample families: `exp(x)` and `sin` do not refute, **`tanh` fires** at σ^6.302 against a 0.25 bar and the leave-one-out diagnostic shows the deviation is stable rather than a grid artefact | why `tanh` deviates. `c₆` is the registered hypothesis and is underived, so the cell stands unexplained |

Every number GATE-D4 turns on was unknown on the date the family was declared, and `T`,
R6's signal and `δ_ref` are unknown still. That is the claim this document exists to make
checkable, and this file's git history is the evidence.

Amendments are appended and dated rather than folded into the text they correct, so a
reader can see what was believed when. Two stand so far: the prior mean was missing from
the family, and the registered method for `c₂` and `c₄` computed a quantity that is zero by
construction.

**One entry is out of order, and it is marked.** Section 7's `c₄` was derived and tested on
one family before anything registered it. The disclosure sits at the head of that section
rather than in a footnote, the predictions it did satisfy are dated earlier in this file,
and the families it has not been tested on are pre-registered with their expected values
before those runs. A reader who wants to know whether this file was written to fit its
results should start there, since it is the one place where the ordering has to be argued
rather than read off the git history.

### AMENDMENT 2026-08-20: the commit record now runs in code as well as in prose

The table in `### AMENDMENT 2026-08-17` maps four of section 7's claims to the commits
that registered and measured them. That map is now also carried by the suites themselves.
Every `PROVED` report the three symbolic suites emit carries a `Provenance`: the ref where
its derivation was registered, the ref whose tree measured against it, and one line naming
what is at the first. Running a suite prints them.

Nothing in the table above changes, and no number moves. The suites report the same
counts, `23`, `18` and `29`, all `NOT TRIGGERED`.

Three of the nine sources record an ordering that runs backwards, and they say so.
`research/c4_hand_derivation.md` was committed at `99e3c34` on 2026-08-17. The suites
citing its Steps 1-2, Step 3 and Step 4 were measuring against it earlier: `23f0c47` on
2026-08-15 and `1888ad4` on 2026-08-16. `tests/test_provenance_ordering.py` asks git
whether each registration ref is an ancestor of the ref that measured against it, and
those three are marked `xfail` rather than excused. This is the same ordering ADR-037
already discloses for the result they back, now visible from a test run instead of from
reading this file.

Four further sources are self-backing: the Gaussian moment table, the truncation operator,
the cumulant-moment recursion and the innovation moments are each stated and checked in
one commit. Their two refs are identical and the render says the ordering is not
established by history, which is the same reservation the 2026-08-17 table states in prose
for its own same-hash rows.

### RESULT 2026-08-23: `c₆` in closed form, and the edge it bounds where `c₄` does not

Derived symbolically on free `l₁..l₆` and a symbolic `R̄`, by the same pipeline that
produced `c₄` and against the same two arms. `research/c6_hand_derivation.md` registers
the basis and the cumulant reach at `018ccc7`, before the suite computed either.

**The basis holds.** `c₆` resolves onto the eighteen-term dimensional basis with
remainder exactly zero, fifteen coefficients non-zero. The counting rule behind it,
`Σ_{k<n} p(2n − 2k)`, reproduces `c₄`'s registered seven and `c₂`'s two without being
told them. Three coefficients come out zero: `l₆`, `l₄/R̄` and `l₂/R̄²`. The first was
predicted, the other two were not and are reported rather than explained.

**Two arms agree.** The generating-function route and the cumulant recursion, neither
calling the other, agree at `σ²`, `σ⁴` and `σ⁶`. Separately the truncation path agrees
with `sympy.series` term for term through `σ⁶`, two orders further than before. The
`EXACT` licence is that agreement, not a tolerance.

**On the ridge of `d4-family-v1`**, with `μ* = √(R₀/κ)` and `R̄ = 2R₀`:

```text
c₂ =  κ / (4R₀)
c₄ =  3κ(κ − 2) / (16R₀²)
c₆ = −κ(7κ + 9)(13κ − 3) / (48R₀³)
```

`c₅ = 0`, and `σ³` carries nothing either. The gap is even in `σ` through `σ⁶`.

**The κ = 2 problem is real and `c₆` answers it.** `c₄` is identically zero at `κ = 2`,
where `σ_max = √(f·c₂/|c₄|)` diverges and the quartic bounds nothing. `c₆` there is
`−529/24`. Its only positive root is `κ = 3/13`, where `c₄` is not zero, so the two never
vanish together and an edge taken against whichever term binds is defined across the
whole positive axis.

**`c₂²/|c₄|` is monotone increasing on `(0, 2)`**, so the rule evaluating `T` at the `κ`
minimising the window width selects the sweep's lower edge whatever that edge is. No
lower edge is declared. Over `κ ∈ [0.05, 1]` the factor on `T` is 39. That is a
declaration this document still owes, and it is independent of everything above.

### DISCLOSURE 2026-08-23: the `σ_max` choice is now being made with the numbers visible

The 2026-08-07 entry left `σ_max` open between redefining the edge against `c₆` and
keeping `f` as a bound that subtraction makes slack, on the grounds that "choosing now is
choosing with the refit's outcome in view".

That condition has not improved. It has worsened: `c₆` is now in closed form on the
ridge, so both branches can be priced exactly before either is chosen. Whoever takes the
decision takes it with more of the answer in view than the entry was written to avoid,
and no ordering of commits changes that.

Recorded rather than repaired. The alternative was to take the decision quietly inside
the work that produced the coefficient, which is the failure mode the whole registration
exists to make visible.

### DISCLOSURE 2026-08-23: `N` is undeclared, and `D*` and `f*` rest on it

The statistical term reads `σ_p ≈ ε√12/(D√N)`, so it depends on `N`, the number of `σ`
samples the D2 fit uses. `N` appears nowhere else in this document and is not declared.

It is not a presentational gap. The amendment that retired the bias-only closed form
concluded that the bias-only problem is **ill-posed**, `T` being maximised by shrinking
the window to nothing, and that what makes it well-posed is the noise term diverging as
`D → 0`. That term is the one carrying `N`. So `D* = 0.520` and `f* = 0.0488` are the
optimum *at some N*, and this document does not say which.

`N` is also not neutral, on the same terms `D` is not: more samples lower `σ_p`
monotonically, which loosens the constraint and admits a larger `f`, which raises `T`.
Nothing here bounds it from above.

Declared on grounds independent of D4, before the propagation behind `σ_p` is recovered.
Recovering that propagation first and choosing `N` afterwards would be choosing with its
consequence for `T` already computable.

`research/c6_window_exploration.md` records what this was found by. That file is
exploratory, uses a stand-in for `σ_p`, and is cited by nothing.

### DECLARATION 2026-08-23: `κ_min = 0.1`, rationalised rather than derived

The sweep's lower bound, which the rule evaluating `T` at the `κ` minimising the window
width selects whatever else the sweep contains. `c₂²/|c₄| = κ/(3|κ − 2|)` rises across
the range in play, so the floor is the binding cell and nothing above it reaches `T`.

**The argument.** D2's second leg, registered 2026-08-07, sweeps `κ` off the ridge and
needs `κ ≲ 0.1` to resolve: its slope runs to 2 as `κ → 0` against exactly 1 on the
ridge, and at `κ = 0.1` the separation is 0.82. A floor of 0.1 also sits below the `c₆`
zero at `κ = 3/13`, so the window's upper edge is defined across the whole range whichever
term binds, and it holds the `μ`-rule inflation at 3× where this document still treats it
as second order.

**What that argument is worth.** It is a defensible reading of a constraint registered
before `c₆` existed and before `T` was computable. It is not a value the mathematics
forces. No experiment has been run that could distinguish 0.1 from 0.08 or 0.15, and none
is planned. **This is a plausibly weak decision and is recorded as one.**

**Revision is licensed by one condition only:** a measurement bearing on where the second
leg resolves. A gate outcome is not such a measurement, and neither is a `T` that reads
uncomfortably.

**A revision after any gate result does not revise that result.** The outcome obtained
under `κ_min = 0.1` stands as obtained. A re-evaluation is a separate cell carrying both
values, on the same terms the `tanh` fire is not un-fired by better inputs.

**The incentive, stated so a later reader can check whether it was acted on.** `κ_min` is
not neutral and does not push one way. Lowering it lowers `T`, which makes GATE-D4 easier
to pass, since the gate passes on `gap > T`. The same move makes D1 and D2 harder to be
tests at all, since they need `δ_ref ≤ T/k_min`. No value of `κ_min` is good for both.

### AMENDMENT 2026-08-23: `σ_max` against `c₆`, and the shape `T` takes under it

Discharging a consequence rather than taking a choice. The dilute-versus-subtract rule
registered 2026-08-07 fired for **subtract**, before its input was visible. Subtraction
removes the quartic from the fit, so the leading term the fit still carries is the
sextic, and `σ_max` was defined against the quartic. `c₆` is now in closed form.

**The edge.** `σ_max` is where the leading unmodelled term reaches a fraction `f` of the
leading one. Under subtraction that term is `c₆σ⁶`, so `|c₆|σ_max⁶ = f·c₂σ_max²`:

```text
registered   σ_max² = f·c₂/|c₄|      T = f · c₂²/|c₄| · 10^(−2D)
amended      σ_max⁴ = f·c₂/|c₆|      T = c₂^(3/2) · √(f/|c₆|) · 10^(−2D)
```

`T` was linear in `f` and is now a square root of it. `f` is unchanged in meaning, the
fraction the leading unmodelled term reaches at the top edge, and changed in referent,
that term now being the sextic rather than the quartic. **The registered `f* = 0.0488`
does not carry across**, having been optimised against the quartic. It is re-derived or
it is not used.

**The edge is defined wherever the sweep goes.** `c₄` vanishes at `κ = 2` and `c₆` at
`κ = 3/13`. They never coincide, so one of the two always sets an edge.

**The binding cell is still the floor.** `T` is evaluated at the `κ` minimising the
window width, which under the amended edge minimises `c₂^{3/2}/√|c₆|`. That factor rises
from the floor to a pole at `κ = 3/13`, dips after it, and rises again to
`(1/4)^{3/2}/√(91/48) ≈ 0.0908` as `κ` grows. It never returns below its value at
`κ_min = 0.1`, so the declared floor binds whatever ceiling the sweep takes, under both
the registered edge and this one.

`research/src/research/explorations/sigma_max_edge.py` computes it and asserts the two
claims above. It evaluates no `T`: the form is registered here, and the value comes after.

**What `T` still waits on.** `f*` re-derived under the sextic edge, which needs the
statistical term settled, which `research/d2_noise_model_exploration.md` puts in question
on two counts this document has not answered.

### AMENDMENT 2026-08-23: two sextic provenance refs were wrong, and are corrected

The RESULT of 2026-08-23 says `research/c6_hand_derivation.md` "registers the basis and
the cumulant reach at `018ccc7`". Half of that is false and is corrected here rather than
in place.

`018ccc7` carries the basis and its counting rule, so `SEXTIC_BASIS_SOURCE` was right to
name it. It does **not** carry the cumulant reach: the connectivity bound
`m ≤ N/2 + 1` landed at `2f39903`, in the amendment that retired the table `018ccc7`
opened with. A reviewer following the earlier ref would have found the superseded `κ₅`
claim, which the bound contradicts. Five `PROVED` rows carried it.

`SEXTIC_CUMULANT_SOURCE.registered_at` now reads `2f39903`.

Separately, `measured_at` on both sextic sources read `5ce7695`, the commit that raised
`DERIVATIVE_ORDER` and made the expansion computable. Nothing measured `c₆` there. The
checks first ran at `88afd7c`, which is what both now name.

`tests/test_provenance_ordering.py` passed throughout, both wrong refs still preceding
their measurement. Ordering is what that test asks about, and whether a ref carries the
claim it is cited for is a separate question it does not put.

### DECISION 2026-08-23: `T` is parked, and what unparks it

`T` has a registered form and no value. It stays that way until the reference filter
exists, which is PR-7's.

**What is left.** `f*` has to be re-derived under the sextic edge, the registered
`f* = 0.0488` having been optimised against the quartic one. That re-derivation needs the
statistical term, and the statistical term is in question on two counts recorded above
and in `research/d2_noise_model_exploration.md`: this document states the estimator two
ways, and its published `σ_p` values match only one of them; and its stated formula
`ε√12/(D√N)` carries decades where nats are needed.

**Why waiting is not deferral.** Both counts turn on the same unknown. `δ_ref` is a
certified bound on a deterministic error, so what it does to a fitted exponent depends on
the error's *shape* across the window, not on its size at a point. A shape is a property
of a filter. Until one exists there is nothing to measure, and the honest range on the
exponent's error runs from zero to twice the whole budget.

Settling `σ_p` now would mean choosing among readings of a quantity nobody has observed,
and then registering `f*` and `T` on top of it.

**What unparks it.** The reference filter, with its error measured across the window
rather than bounded at a point. `Cov(v, ε)/Var(v)` on the measured field is then the
statistical term, and it carries no `N`.

**What does not.** Nothing about the gate's difficulty, and no reading of `T` that would
follow from one branch rather than another. Parking moves no number: everything already
registered stands, including `κ_min`, the sextic edge and the form `T` takes under it.

### AMENDMENT 2026-08-23: the corrected ref count was itself wrong, and `c₆` now has a committed derivation

Two things, both appended rather than fixed in place.

**The count.** The AMENDMENT above ends "Five `PROVED` rows carried it", of
`SEXTIC_CUMULANT_SOURCE`. Six do: `gap_series.sextic_l6_absent`, the two from
`check_sextic_cumulant_reach` (`κ₅`, `κ₆`) and the three from `check_sextic_arms_agree`
(`σ²`, `σ⁴`, `σ⁶`). The number was not asserted anywhere, which is why correcting one
count left the next one wrong.

**The ridge closed forms.** `c₆ = −κ(7κ + 9)(13κ − 3)/(48R₀³)` reached this document,
`BUILD_PLAN.md` and `sigma_max_edge.py` as a hand transcription. No committed code
substituted `R(x) = R₀ + κx²` at `μ* = √(R₀/κ)` into the derived coefficient to produce
it, which is the practice ADR-050 registers in this same change. `gap_series.ridge_c2`,
`ridge_c4` and `ridge_c6` now do.

The transcription was **correct**. All three coefficients reproduce, at general `R₀`:

```text
c₂ = κ/(4R₀)      c₄ = 3κ(κ − 2)/(16R₀²)      c₆ = −κ(7κ + 9)(13κ − 3)/(48R₀³)
```

so nothing registered on top of `c₆` moves — not `σ_max⁴ = f·c₂/|c₆|`, not the `−529/24`
at `κ = 2`, not the `3/13` root, not the `(1/4)^{3/2}/√(91/48)` tail limit, and not the
conclusion that the declared floor binds whatever ceiling the sweep takes. What changes
is that a slip in any of them would now fail a check rather than survive the suite.

`c₂` and `c₄` were in closed form on the ridge before this ran, in
`research/c6_hand_derivation.md` at `018ccc7`, so those two rows are predictions and
carry that ref. `c₆` was not registered anywhere ahead of it, so its row registers and
measures at one commit and says so rather than borrowing `018ccc7`, which is the error
the AMENDMENT above corrects.

### PRE-REGISTRATION 2026-09-11: `T` unparks, and what is fixed before the field is looked at

The DECISION of 2026-08-23 parks `T` until the reference filter exists and names what
unparks it: the filter's error measured across the window rather than bounded at a
point. The filter exists, in `cpomdp.reference`, and its gap evaluator is the engine
PR-8 certifies. This entry fixes everything that has to be fixed before that field is
measured, so that no choice is made with its consequence for `T` in view. The
measurement and the value follow in a RESULT entry, in a later commit.

**What is measured.** The relative error field of the tree's reference engine on
`d4-family-v1` at the binding cell, `κ_min = 0.1`, `R₀ = 1`, `μ* = √10`, evaluating the
plug-in gap that `c₂`, `c₄` and `c₆` expand:

```text
ε(σ) = ( gap_declared(σ) − gap_fine(σ) ) / gap_fine(σ)
```

over fifteen values of `σ` spaced evenly in `ln σ` on `[0.005, 0.7]`, which covers every
window the optimisation below can reach. `gap_declared` runs on the declared lattice
and `gap_fine` on a lattice with twice the nodes and boxes half again as wide, so the
difference is the declared lattice's error. At three of the fifteen, the two ends and
the middle, a third lattice with twice the nodes and boxes wider again is run, and the
estimate is read as converged only where replacing fine by finer moves it by under 10%.
Where `|ε|` sits under `100 × 2⁻⁵²`, the field is read as zero: that is roundoff, not a
discretisation error, and its shape is not a property of the filter.

**The declared lattice**, which is what PR-8's bound certifies and what `T` is
registered against. A change to it re-registers `T`, and an outcome obtained under this
one stands as obtained.

| grid | box | nodes |
| --- | --- | --- |
| state | `μ* ± 12σ` | 1601 |
| observation | `μ* ± 9·√(σ² + R(μ*))` | 401 |

The observation half-width is the truncation study's, so its measured truncations
apply. The state half-width is in prior spreads, since the plug-in posterior's spread
is under the prior's and the exact posterior's mode moves by at most a few prior
spreads across the observation box on this family.

**The estimator.** Unweighted ordinary least squares of `ln(gap − c₄σ⁴)` on `ln σ`,
uniform in `ln σ` across the window. The published `σ_p` figures were shown on
2026-08-23 to come from this estimator and not from the weighted one the prose
described, so this corrects the prose to the numbers rather than changing the fit.
`N` is not declared, because under a deterministic error field it does not enter: the
shift in the fitted exponent is `Cov(v, ε)/Var(v)` with `v = ln σ`, a functional of the
field and not of a sample count.

**How the two errors combine.** The truncation bias and the filter's shift are both
biases, so they add in magnitude:

```text
|b_trunc(f, D)| + |b_ref(f, D)| ≤ β        β = 0.05, registered 2026-08-07
```

This supersedes the quadrature sum `√(bias² + σ_p²) ≤ β` for the case the noise-model
exploration found operative, a deterministic error, and it is the conservative
reading: adding in quadrature would let two biases partly hide each other. `b_ref` is
evaluated on the measured field over the window `[σ_max/10^D, σ_max]`, so it depends
on `f` and `D` through where the window sits, and the optimisation below sees that.

**The sign of the residual.** Under subtraction the term left in the fit is the sextic,
and `c₆ > 0` at `κ_min` (`c₆ = −κ(7κ + 9)(13κ − 3)/48`, positive below `κ = 3/13`). The
correction is `+f·e^{4u}`, bending the curve above a power law, so the truncation bias
is positive and the integral is taken with that sign rather than the negative one the
window exploration used. The two differ at second order in `f`.

**A floor on `D`, declared blind.** The AMENDMENT of 2026-08-07 that retired the
bias-only closed form found the bias-only problem ill-posed: `T` is maximised by
shrinking the window to nothing, and only the noise term bounded `D` from below. If the
field turns out benign, that term is gone and nothing in the algebra sets `D`. So:

```text
D ≥ 0.5 decades
```

Half a decade in `σ` is one decade in `σ²`, the variable the fitted quantity is
quadratic in, so a reader sees the exponent read across a full order of magnitude of
the ordinate. It is the smallest round width that gives that, and it sits within 4% of
the registered `D* = 0.520`, which was the noise-set optimum. The incentive is stated:
`T ∝ 10^(−2D)`, so a lower floor raises `T` and makes the gate harder, and this floor is
the severity-maximising choice among round ones, on the same footing as every other
rule here. Nothing measured can move it; only a reader's judgement about what a fit
over half a decade is worth, and that judgement is being made now, with no number in
view.

**The optimisation, stated so it can be run and not steered.** For each `D ≥ 0.5`, `f*(D)`
is the largest `f` satisfying the combined budget, and

```text
T(D) = c₂^(3/2) · √(f*(D) / |c₆|) · 10^(−2D)        at κ_min, per the AMENDMENT of 2026-08-23
D*   = argmax T(D)
```

Under the benign reading `T(D)` is decreasing and `D* = 0.5`. `T` is printed at `D*`,
`D* + 0.5` and `D* + 1`. Section 4 promised `D − 1`, `D` and `D + 1`; `D* − 1` is below
the floor and is not a window, so the half-decade step replaces it.

**Predictions, written before the run.**

- The field is benign: `|b_ref| < 0.01·β` on every window the optimisation visits.
  The lattice integrates smooth, decaying densities by the trapezoid rule, whose error
  is exponentially small in the node spacing, and the state lattice scales with `σ`
  while the observation lattice does not, so the error that remains is set by the
  observation box's truncation and is far below the gap on this family.
- `D* = 0.5`, `f*` within a factor of two of the window exploration's stand-in
  `0.034`, and `T` of order `10⁻⁴` nats.

If `|b_ref|` exceeds `0.1·β` anywhere, the registered `σ_p` is superseded rather than
recalibrated, `b_ref` is carried as the statistical term in its own right, and the
RESULT says which window it bit on. Between `0.01·β` and `0.1·β` the term is carried in
the constraint and the prediction is recorded as missed.

**Status of the numbers.** The field is `COMPUTED`: a refinement difference on a
declared lattice, with no warrant claimed. Certifying the bound is PR-8's. `T` is a
declaration derived from registered constants and that field, and the exploration that
prints it reports no warrant, on the terms `research.explorations` states. What makes it
a registration rather than a fit is this entry's date against the RESULT's.

**The code.** `research.explorations.threshold`, run with
`uv run --no-sync python -m research.explorations.threshold`. It prints every number the
RESULT quotes, asserts the convergence read and the roundoff floor, checks the shift
integral against a fit that shares no code with it, and evaluates no `T` until the field
has been read.
