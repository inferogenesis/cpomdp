# The discretisation bound: the hand derivation

`research/discretisation_bound_hand_derivation_part1.pdf` came first. It is a scan of the
notebook Part 1 was worked in, dated 18/09/2026 to 11/10/2026. It is the original. This
is a typed up version of that scan by AI-Claude Opus: the same steps, in the same order,
under the notes' own numbering, so the two read side by side.

The steps transcribe the scan and stop where it stops. Anything dated at the foot of
this file is not in the scan. It is added after code has checked the steps, and says
what in the scan it corrects. Where a step and the scan disagree, the scan is what
happened and the step has the transcription error.

GATE-D4 compares a certified bound `δ_ref` on the reference engine's discretisation
error against the threshold `T`. D1 and D2 are tests only if `δ_ref ≤ T/k_min`. Section
4 of `research/gate_d4_registration.md` sets that up and its RESULT of 2026-09-11 gives
`T`. This file derives the bound. Part 1 sets up the cell and the engine, splits the error
into four sources, and runs Route B on the state axis for the likelihood integrand and
the four prior-side sums. The notes' step 2 takes the observation axis and the ratios the
engine forms from those sums. The tails, the bound on `ŝ²` that sum (d) leans on, and
the propagation into the gap come after it.

Nothing in the steps is a measured result except the threshold RESULT the notes quote.

Line citations in the notes are against `main` at `2836ec5`, which was still the tip of
`main` on 2026-09-18.

## Prelims

Based on `main` as of 2026-09-18.

### The cell `d4-family-v1`

```text
R(x) = R₀ + κx²           R₀ = 1, κ = 0.1         threshold.py:72, operating_point.py:45
μ*   = √(R₀/κ) = √10                               threshold.py:78
R̄    = R(μ*) = 2R₀ = 2
s_p² = σ² + R̄                                      threshold.py:169
π(x) = N(x; μ*, σ²)                                threshold.py:176
σ    ∈ geomspace(0.005, 0.7, 15)                   threshold.py:81
```

### The declared lattice

`threshold.py:121`.

```text
x_i = μ* − 12σ + i·h_x     i = 0 … 1600     h_x = 24σ/1600  = 0.015σ
y_k = μ* − 9s_p + k·h_y    k = 0 … 400      h_y = 18s_p/400 = 0.045s_p
```

Nodes come from `linspace`, with the endpoints exact (`quadrature.py:139`). The weights
are `w = h`, halved at both ends (`quadrature.py:76-79`).

```text
Σ_x f := Σ_i w_i f(x_i)          Σ_y g := Σ_k w_k g(y_k)
```

### Per observation node `y = y_k`

1. The prior on the grid, normalised. `gap.py:209`, `quadrature.py:335`.

   ```text
   π_i = π(x_i)      m_π = Σ_x π_i      π̃_i = π_i / m_π
   ```

2. Its moments, and the noise at its mean. `ladder.py:131`, `quadrature.py:391-394`,
   `ladder.py:132`.

   ```text
   μ̂ = Σ_x π̃_i x_i      ŝ² = Σ_x π̃_i (x_i − μ̂)²      R̂ = R(μ̂)
   ```

3. The joint, the grid predictive and the grid posterior. `gap.py:125`,
   `likelihood.py:38`, `quadrature.py:335`.

   ```text
   J_i(y) = π̃_i N(y; x_i, R(x_i))
   m_J(y) = Σ_x J_i(y) =: p*_grid(y)
   p̂_i    = J_i / m_J
   ```

4. The plug-in rung. `ladder.py:101-106`, `:140`, `quadrature.py:335`.

   ```text
   K   = ŝ² / (ŝ² + R̂)
   m_q = μ̂ + K(y − μ̂)      v_q = (1 − K)ŝ²
   q_i = N(x_i; m_q, v_q)   m_Q = Σ_x q_i      q̃_i = q_i / m_Q
   ```

5. The divergence at this node.

   ```text
   KL_grid(y) = Σ_x q̃_i (log q̃_i − log p̂_i)
              = (1/m_Q) Σ_x q_i (log q_i − log J_i − log m_Q + log m_J)
   ```

   The second line is exact algebra on the first, since `Σ_x q̃_i = 1`.

6. The reported value. `gap.py:242-250`, `quadrature.py:358`.

   ```text
   value = Σ_y m_J(y_k) KL_grid(y_k) / Σ_y m_J(y_k)
   ```

7. The edge ratio. `gap.py:127`.

   ```text
   edge = max_{i ∈ {0, 1600}} J_i(y) / max_i J_i(y)
   ```

Every mass goes through `logsumexp`. The direction is `KL(q̃ ‖ p̂)`.

### The exact quantity

```text
p*(y)  = ∫ π(x) N(y; x, R(x)) dx
q_y    = the Kalman Gaussian from (μ*, σ², R̄), not from the grid moments
KL(y)  = ∫ q_y (log q_y − log π − log N(y; ·, R)) dx + log p*(y)
E      = ∫ p*(y) KL(y) dy
```

All integrals are over `ℝ`.

### The target

```text
δ_ref ≥ |value − E|        at each of the 15 spreads
bar:    δ_ref ≤ T/k_min = 5.96e−5 nats,     T = 5.962e−4,  k_min = 10
```

### Measured

From the RESULT of 2026-09-11, declared lattice minus fine. Two lattices, neither exact.

- `1e−16` to `5e−16` nats below `σ ≈ 0.3`, with no sign pattern.
- `1.8e−12` nats at `σ = 0.7`, with a steady sign.
- Box mass at least `1 − 1.1e−11`. Edge ratio below `1e−16`.
- `ROUNDOFF_FLOOR = 100 · 2⁻⁵² ≈ 2.2e−14`, relative. `threshold.py:87`.

## Step 1: the split

For any integrand `f` on the state axis,

```text
Σ_x f − ∫_ℝ f  =  [ Σ_box f − ∫_box f ]  +  [ −∫_{|x−μ*| > 12σ} f ]
                          S1                          S3
```

and on the observation axis,

```text
Σ_y g − ∫_ℝ g  =  [ Σ_box g − ∫_box g ]  +  [ −∫_{|y−μ*| > 9s_p} g ]
                          S2                          S4
```

- **S1** acts on six sums: `m_π`, `μ̂`, `ŝ²`, `m_J(y)`, `m_Q`, and the cross sum in step 5.
- **S2** acts on two: the numerator and the denominator of step 6.
- **S3** is the same six integrands, their tails outside the state box, at fixed `y`.
- **S4** is `p*`'s tail outside the observation box. Step 6 then estimates
  `E[KL | y ∈ box]`.

### Decision 1: the route for S1 and S2

**Route A** is Euler–Maclaurin on the finite box. The error goes as
`h^{2m} · sup|f^{(2m)}|`. It needs `2m` bounded derivatives.

**Route B** is the exponentially convergent trapezoid rule for analytic integrands
(Trefethen and Weideman, SIAM Review 56(3), 2014). The error is at most
`2M/(e^{2πa/h} − 1)`. It needs analyticity in a strip of half-width `a`, and an
integral bound `M` inside it.

The state axis (S1) takes Route B. The observation axis (S2) is deferred to step 2, since
`log p*(y)` may have no strip.

Route B on `x` because `h_x/σ = 0.015`, so `e^{−2πa/h_x} = e^{−419a/σ}`. Route A at
order `2m` buys `(0.015)^{2m}` and needs the `2m`-th derivative of `J` by hand.

### Route B

Let `f` be analytic in the strip `S_a = {z : |Im z| < a}`, `a > 0`.

- **(H1)** `f` is analytic on `S_a` and `f(z) → 0` uniformly as `|Re z| → ∞` in `S_a`.
- **(H2)** `∫_ℝ |f(u + iv)| du ≤ M` for every `|v| < a`.

Then for every `h > 0` and every shift `x₀`,

```text
| h Σ_{j∈ℤ} f(x₀ + jh) − ∫_ℝ f |  ≤  2M / (e^{2πa/h} − 1)
```

What it bounds is the infinite trapezoid sum. The engine's sum is finite:

```text
Σ_x f = h Σ_{j∈ℤ} f(x₀ + jh) − h Σ_{nodes outside box} f(x_j) + (end-weight correction)
```

S3's tail argument bounds the middle term. The end-weight correction is
`h·f(x₀)/2 + h·f(x₁₆₀₀)/2`, each a value of `f` at the box edge, `12σ` out. So

```text
S1 = [theorem bound] + [outside nodes] + [two edge values]
```

### The likelihood integrand

At fixed `y`, as a function of `x`,

```text
f_y(x) = π(x) · N(y; x, R(x))
       = (2πσ²)^{−½} e^{−(x−μ*)²/2σ²} · (2πR(x))^{−½} e^{−(y−x)²/2R(x)}
```

`J_i = π̃_i N(·)` carries the constant `1/m_π`. I bound `f_y` and divide by `m_π` at the
end.

#### Check H1 on `f_y`

The Gaussian factor is entire. `R(z) = R₀ + κz²` vanishes at `z = ±i√(R₀/κ) = ±i√10`.
For `z = u + iv` on `S_a`:

```text
R(z)    = (R₀ + κu² − κv²) + i · 2κuv
Re R(z) = R₀ + κ(u² − v²) > 0        whenever v² < R₀/κ = 10
```

So on `S_a` with `a < √10`, `R(z)` lies in the right half-plane. The principal branch of
`R(z)^{−½}` is analytic there, and so is `1/R(z)`. The exponential of an analytic
function is analytic, so `f_y` is analytic on `S_a` for `a < √10`.

Decay: `|f_y(u + iv)|` is at most the H2 bound below, which goes to 0 as `|u| → ∞`.

#### Check H2 on `f_y`

Bound `|f_y(u + iv)|` factor by factor. Notation:

```text
z = u + iv,    |v| < a < √10,    d := y − u
R(z) = A + iB,    A = R₀ + κ(u² − v²),    B = 2κuv
A ≥ R₀ − κv² > 0 on S_a,    |R|² = A² + B²
```

**Factor 1, the prior.**

```text
|e^{−(z−μ*)²/2σ²}| = e^{−(u−μ*)²/2σ²} · e^{v²/2σ²}
```

since `(z − μ*)² = (u − μ*)² − v² + 2i(u − μ*)v` and only the real part survives the
modulus.

**Factor 2, the normaliser.** Using `|R| ≥ Re R = A`,

```text
|R(z)|^{−½} ≤ A^{−½} ≤ (R₀ − κv²)^{−½} ≤ (R₀ − κa²)^{−½}
```

**Factor 3, the likelihood exponent.** This needs a lower bound on
`Q := Re[(y − z)²/2R(z)]`.

```text
(y − z)² = (d − iv)² = (d² − v²) − 2i·dv =: p + iq,     p = d² − v²,  q = −2dv
(y − z)²/R = (p + iq)(A − iB) / |R|²
Re[(p + iq)(A − iB)] = pA + qB = (d² − v²)A − 2dvB = (d² − v²)A − 4κuv²d
```

So

```text
Q = [Ad² − 4κuv²d − v²A] / (2|R|²)
```

Complete the square in `d`:

```text
Ad² − 4κuv²d − v²A = A(d − 2κuv²/A)² − 4κ²u²v⁴/A − v²A
                   ≥ −(v²/A)(4κ²u²v² + A²)
                   = −(v²/A)(B² + A²)
                   = −(v²/A)|R|²
```

Hence `Q ≥ −v²/2A ≥ −v²/2(R₀ − κv²)`, uniformly in `u` and `y`. So

```text
|e^{−(y−z)²/2R(z)}| = e^{−Q} ≤ e^{v²/2(R₀ − κv²)} ≤ e^{a²/2(R₀ − κa²)}
```

#### Assemble

```text
C(a) := (2π(R₀ − κa²))^{−½} e^{a²/2(R₀ − κa²)}
```

Then for every `|v| < a`, every `u` and every `y`,

```text
|f_y(u + iv)| ≤ C(a) · e^{v²/2σ²} · (2πσ²)^{−½} e^{−(u−μ*)²/2σ²}
```

which also closes the decay left open in H1: Gaussian in `u`, uniformly in `v`.

```text
M(v) = ∫_ℝ |f_y(u + iv)| du ≤ C(a) · e^{v²/2σ²}
M    = sup_{|v|<a} M(v)     ≤ C(a) · e^{a²/2σ²}
```

The cap comes from `R₀ − κa² > 0`, so `a < √(R₀/κ) = √10`, and `C(a) → ∞` as
`a → √10`. The cost comes from `e^{a²/2σ²}`, the prior Gaussian growing off the real
axis.

#### Numbers at the cell (`R₀ = 1`, `κ = 0.1`)

```text
a = 1.58:                   R₀ − κa² = 0.750,   C(a) = 0.461 · e^{1.664} = 2.43
a = 10σ ≤ 1.58 (σ ≤ 0.158): C(a) ≤ 2.43,  e^{a²/2σ²} = e^{50},   M ≤ 1.3e22
a = 1.58, σ = 0.7:          e^{a²/2σ²} = e^{2.55} = 12.8,          M ≤ 31
```

The Route B bound on the infinite sum, `2M/(e^{2πa/h_x} − 1)`:

```text
σ ≤ 0.158:  2πa/h_x = 2π·10/0.015      = 4189,   bound ≤ 2.6e22 · e^{−4189}
σ = 0.7:    2πa/h_x = 2π·1.58/0.0105   = 945,    bound ≤ 62 · e^{−945}
```

#### Strip width, as a rule of `σ`

```text
a(σ) = min(10σ, a_cap),     a_cap = √10/2 = 1.58
```

The reason for the cap: any `a_cap < √10` keeps H1, and `√10/2` keeps
`R₀ − κa² = 3/4`, so `C(a)` stays `O(1)`. The reason for `c = 10`: it gains `e^{4189}`
per spread against a cost of `e^{50}`. The cap binds above `σ = 0.158`.

## Step 1.4: the trapezoid error on the four prior-side sums

The scan's page 8, where this step starts, is dated 11/10/2026.

The engine computes four sums over the state grid that depend only on the prior or on the
plug-in rung's Gaussian. Each sum stands in for an integral. This step shows that the
trapezoid rule's error on each is too small to matter. It uses the same theorem as the
likelihood section.

Route B is Trefethen and Weideman, SIAM Review 56(3), 2014, Theorem 5.1. Take a function
`f` of a real variable `x`. Extend it to complex inputs `z = u + iv`, where `u` is the
real part and `v` the imaginary. Fix a width `a > 0` and call the band `|v| < a` the
strip. The theorem needs two conditions.

1. **Smoothness.** `f` is analytic (complex differentiable) at every point of the strip,
   and `f(z) → 0` as `|u| → ∞` anywhere in it.
2. **Size.** Integrating `|f|` along any horizontal line inside the strip gives at most
   some number `M`.

If both hold, the trapezoid sum with spacing `h` over the whole real line differs from
`∫f` by at most `2M/(e^{2πa/h} − 1)`.

As I read it, a function that stays smooth a distance `a` off the real axis is
integrated almost exactly, with the error falling as `e^{2πa/h} − 1` grows. `M` measures
how much the function grows as you move off the axis. Widening the strip shrinks the
exponential factor and grows `M`. The strip `a = min(10σ, 1.58)` chosen earlier balances
the two. These are H1 and H2 for short.

Two choices:

1. **Unnormalised sums.** The engine divides by `m_π` to get `μ̂` and `ŝ²`. A ratio's
   error depends on both its numerator and its denominator, so all ratios are handled
   together in step 2. Each sum here is bounded on its own.
2. **Order.** The sums are bounded in the order (a), (b), (c), (d). Sum (d) needs a lower
   bound on `ŝ²`, which comes from (a) to (c), so the argument is not circular.

The fact underneath, stated plainly: how a Gaussian grows off the real axis.

```text
(z − μ*)² = (u − μ*)² − v² + 2i(u − μ*)v
|e^w| = e^{Re w}            for any complex w
⇒ |π(u + iv)| = π(u) · e^{v²/2σ²}
```

Moving a Gaussian of variance `σ²` a distance `v` off the axis multiplies its size by
`e^{v²/2σ²}`. This works for every Gaussian.

The condition for (a) to (d), made explicit. Each integrand is a Gaussian times a
polynomial. No division, square root, log or non-integer power appears, so each is
analytic at every complex `z`. The size bounds below are a Gaussian in `u` times a
polynomial, so each integrand tends to 0 as `|u| → ∞` throughout the strip.

### (a) The prior's total mass, `m_π = Σ_x π_i`

The integrand is `f(z) = π(z)`.

```text
∫ |π(u + iv)| du = e^{v²/2σ²} · ∫ π(u) du = e^{v²/2σ²}
M ≤ e^{a²/2σ²}                                   largest over |v| < a
```

### (b) The first-moment sum, `Σ_x x_i π_i`

The integrand is `f(z) = z π(z)`, the numerator of `μ̂`.

```text
|z| ≤ |u| + |v| < |u| + a                         triangle inequality
U ~ N(μ*, σ²):  ∫ |u| π(u) du = E|U| ≤ μ* + E|U − μ*| = μ* + σ√(2/π)
M ≤ (μ* + σ√(2/π) + a) · e^{a²/2σ²}
```

`E|U − μ*| = σ√(2/π)` is the mean absolute deviation of a Gaussian.

### (c) The second-moment sum, `Σ_x (x_i − μ̂)² π_i`

The integrand is `f(z) = (z − μ̂)² π(z)`, with `μ̂` held fixed as a real number. It is
the numerator of `ŝ²`.

```text
|(z − μ̂)²| = (u − μ̂)² + v² ≤ (u − μ̂)² + a²
∫ (u − μ̂)² π(u) du = σ² + (μ* − μ̂)²             variance plus squared offset
|μ* − μ̂| ≤ 12σ
M ≤ (145σ² + a²) · e^{a²/2σ²}
```

Why `12σ`. `μ̂` is a weighted average of the grid nodes. The weights `w_i π_i / m_π` are
non-negative and sum to 1, so `μ̂` lies inside the grid box, which runs from `μ* − 12σ`
to `μ* + 12σ`. The true offset is far smaller. Using it would mean assuming `μ̂ ≈ μ*`,
which is part of what has to be proved. The crude bound costs a factor of about 145 in
`M`, which changes nothing against `e^{−945}`.

### (d) The plug-in posterior's mass, `m_Q = Σ_x q_i`

The integrand is `f(z) = q(z)`, a Gaussian with real mean `m_q` and variance `v_q`. The
growth fact applies with `v_q` in place of `σ²`.

```text
M ≤ e^{a²/2v_q}
v_q = (1 − K)ŝ² = ŝ²R̂ / (ŝ² + R̂)
⇒ 1/v_q = 1/(ŝ² + 1/R̂)
R̂  ≥ R₀ = 1
ŝ² ≥ σ²/2                                        uses (a) to (c)
M ≤ e^{a²/σ²} · e^{a²/2R₀}
```

Why `R̂ ≥ R₀`: the noise `R(x) = R₀ + κx²` is at least `R₀` for every real `x`, `μ̂`
included. Why `ŝ² ≥ σ²/2`: once (a) to (c) and their tail terms are bounded (S3, the part
of each integral outside the state box), `ŝ²` differs from `σ²` by far less than `σ²/2`.
This is the only place a later bound leans on an earlier one.

## AMENDMENT 2026-10-11: the strip cap is `√10/2` exactly

Page 7 declares `a_cap = √10/2` and gives the reason, `R₀ − κa² = 3/4`. Its numbers, and
the table drafted from them, used `1.58` as a decimal, which gives `R₀ − κa² = 0.75036`.
Either is an admissible strip, since any `a_cap < √10` keeps H1. I take `√10/2` exactly,
so the stated reason holds and the constants come out in closed form.

The crossover moves from `0.158` to `σ_c = √10/20 ≈ 0.15811`. No declared spread sits on
it. Ten of the fifteen are below it, the largest of them `0.1198`.

Below the crossover nothing moves, since `2πa/h_x = 4000π/3` there whatever the cap. At
`σ = 0.7`:

| | as written, `a = 1.58` | `a_cap = √10/2` |
| --- | --- | --- |
| `C(a_cap)` | 2.43 | 2.439, which is `(3π/2)^{−½} e^{5/3}` |
| `M` for `f_y` | ≤ 31 | 31.27 |
| `2πa/h_x` | 945 | 946.15, which is `2000√10π/21` |
| `M` for (b) | 67.7 | 67.97 |
| `M` for (c) | 939 | 942.9 |
| `M` for (d) | 569 | 573.7 |

## RESULT 2026-10-11: Part 1 checked

`research/src/research/discretisation_bound/`, run with

```text
uv run --no-sync python -m research.discretisation_bound.prelims
uv run --no-sync python -m research.discretisation_bound.likelihood_strip
uv run --no-sync python -m research.discretisation_bound.prior_sums
```

and asserted from `tests/test_discretisation_bound_{prelims,likelihood_strip,prior_sums}.py`.
Each module asserts every claim it prints. Nothing reports a warrant.

**Every identity in the steps holds in SymPy**, with `R₀` and `κ` left symbolic where the
notes are general. The inequalities are checked as identities with a remainder that is
non-negative by inspection. `Q + v²/2A` is a square over `2|R|²`. (d)'s exponent gap is
a sum of two terms that are non-negative once `ŝ² ≥ σ²/2` and `R̂ ≥ R₀`. Two of the
integrals are computed rather than restated: `∫π = 1` and `E|U − μ*| = σ√(2/π)`.

**`C(a)` is attained.** At `u = y = 0` and `v = a` both
inequalities in factors 2 and 3 are equalities, so `N(0; ia, R(ia)) = C(a)`. On a dense
grid over the strip the largest `|N(y; z, R(z))|/C(a)` is 1 to fifteen digits.

**Every log-error grows with `σ`**, on both sides of the crossover. Below it only `ln M`
moves. Above it, `ln M − 2πa/h_x` has derivative at least `(a/σ²)(400π/3 − 10k)` with
`k = 1` or `2`. So `σ = 0.7` bounds all fifteen spreads, and the declared spreads agree.

**(d)'s hypothesis is only corroborated.** The engine's `ŝ²/σ²` is 1 to within
`1e−10` at all fifteen spreads. S3 still has to prove `ŝ² ≥ σ²/2`.

**Against exact errors.** For a Gaussian, Route B at its best strip `a = 2πs/h` gives
Poisson's exponent `−2π²s/h²` exactly, so the theorem's constants are the sharp ones. On
the declared lattice the bounds on (a) and (d) sit above the Poisson error by at least
34293 nats. At steps of `σ` to `1.5σ` at `σ = 0.7`, where the error is measurable, every
bound sits above the measured error, `f_y` and (a) to (d) alike. The float lattice sum
those measurements use agrees with the Poisson series to better than `1e−14`.

**The engine's lattice.** Weights are `h`, halved at both ends, exactly. Endpoints are
exact. Interior nodes sit within `1e−9` of `h` of `x₀ + ih` at every spread.

**The table**, with `a_cap = √10/2`. The `σ ≤ σ_c` column is the supremum over that
regime, attained at `σ_c`.

| Sum | `M`, `σ ≤ σ_c` | `M`, `σ = 0.7` | ln error, `σ ≤ σ_c` | ln error, `σ = 0.7` |
| --- | --- | --- | --- | --- |
| (a) `m_π` | `e^50` | 12.82 | −4138.097 | −942.907 |
| (b) first moment | `4.870 · e^50` | 67.97 | −4136.514 | −941.239 |
| (c) second moment | `6.125 · e^50` | 942.9 | −4136.285 | −938.609 |
| (d) `m_Q` | `e^101.25` | 573.7 | −4086.847 | −939.106 |
| `2πa/h_x` | 4188.79 | 946.15 | | |

The Route B error on each sum is `2M/(e^{2πa/h_x} − 1)`. Below the crossover the
largest is (d), under `e^{−4086}`, about `10^{−1775}`. Over the ten declared spreads
there it is under `e^{−4087.38}`. At `σ = 0.7` the largest is (c), under `e^{−938}`,
about `10^{−407.6}`, and that is the worst over all fifteen spreads. For `f_y` the log
bound is `−4137.205` below the crossover and `−942.016` at `σ = 0.7`.

None of this is `δ_ref`. It bounds one piece of S1, the trapezoid error on the infinite
lattice. The nodes outside the box, the two edge values, S2 to S4, and the propagation
through the normalisations and the log are still to come.

## CLARIFICATION 2026-10-11: what the run corrects in the scan

1. **"Below `e^{−4087}`" fails at the crossover.** The table drafted from the scan said
   the largest error for `σ ≤ 0.158` is below `e^{−4087}`. (d) gives
   `ln 2 + 101.25 − 4000π/3 = −4086.85`. The figure came from rounding
   `2πa/h_x = 4188.79` up to 4189, and it fails under either cap. Page 7's
   `bound ≤ 2.6e22 · e^{−4189}` for `f_y` fails the same way: the bound is
   `2.5e22 · e^{−4188.79}`, which is `3.1e22 · e^{−4189}`. At `σ = 0.7` both statements
   hold.
2. **Page 7's `M ≤ 31` at `σ = 0.7` fails by rounding.** It is 31.05 at `a = 1.58` and
   31.27 at `√10/2`. `M ≤ 1.3e22` and `bound ≤ 62 · e^{−945}` hold.
3. **"The exponents do not change with `σ`" below 0.158.** True of `2πa/h_x` and
   `a²/2σ²`. False of (d)'s `a²/2R₀ = 50σ²`, and the prefactors of (b) and (c) grow too.
   That column is a supremum attained at `σ_c`, which is not a declared spread. That two
   columns cover all fifteen spreads rests on monotonicity instead. Every log-error is
   nondecreasing in `σ` on `(0, 0.7]`, so `σ = 0.7` alone bounds them.
4. **Where the drafted `σ = 0.7` column came from.** (b) 67.7, (c) 939 and
   `2πa/h_x = 945` reproduce at `a = 1.58` (67.71, 939.4, and 945.47 truncated). (d) 569
   matches neither cap: it is 568.4 at `1.58` and 573.7 at `√10/2`.
5. **Step 1.4 (d).** `1/v_q = 1/(ŝ² + 1/R̂)` should read `1/v_q = 1/ŝ² + 1/R̂`. The
   bound on the next lines uses the right form.
6. **Step 1.4 (c), the cost of the crude offset.** 145 is the coefficient on `σ²`. The
   factor on `M` is smaller. Against the offset-free `σ² + a²`, the bound inflates `M` by
   `(145σ² + a²)/(σ² + a²)`. That is 2.4 where `a = 10σ` and 24.6 at `σ = 0.7`. Nothing
   downstream changes.
7. **Step 5, "since `Σ_x q̃_i = 1`".** The second line is the first with `q̃ = q/m_Q`
   and `p̂ = J/m_J` substituted, and needs nothing else. `Σ q̃ = 1` is what would let
   the constants come outside the sum, which the second line does not do.
8. **Step 1, the sign of the end-weight correction.** The weights are `h`, halved at both
   ends, so

   ```text
   Σ_x f = h Σ_{j∈ℤ} f(x₀ + jh) − h Σ_{outside} f(x_j) − (h/2)(f(x₀) + f(x₁₆₀₀))
   ```

   The scan writes the correction with a plus and defines it as the positive
   `h·f(x₀)/2 + h·f(x₁₆₀₀)/2`, so the identity as written is off by
   `h(f(x₀) + f(x₁₆₀₀))`. The split of S1 into the theorem bound, the outside nodes and
   the two edge values stands as a bound on magnitudes. Later parts carry the signed
   split, so the sign matters there.
9. **Two line citations miss at `2836ec5`.**
   - `threshold.py:72` holds `R₀` only. `R(x)` is at `:131-134` and `κ` at `:75`.
   - `likelihood.py:38` is the constant `_LOG_TWO_PI`. The `N(y; x_i, R(x_i))` that
     step 3 builds `J_i` from is `StateDependentNoiseLikelihood.log_likelihood`, at
     `:328-345`.
10. **The nodes are not `x₀ + ih` bit for bit.** `linspace` computes
    `x₀(1 − t) + x_N t`. The endpoints are exact, as the notes say. Route B assumes
    exact spacing, so the departure belongs with the roundoff disclosure.
11. **Not an error.** The (a) to (d) integrands are entire, so `a_cap` is inherited from
    `f_y` there. Nothing forces it. For (a) the best strip would give the Poisson exponent,
    about `e^{−87730}`, against the inherited strip's `e^{−4138}`. Neither matters.
