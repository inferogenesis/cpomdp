# The DP flow meter gives you `R(x)` for free

Supporting note for [`examples/surge_margin.py`](https://github.com/inferogenesis/cpomdp/blob/main/examples/surge_margin.py).
It derives the sensor model that demo runs on, fixes its one calibration constant, and
records what the demo does not model.

Nothing here is a citable result. The example gallery's numbers are showcases, and this
note inherits that status.

The application comes from Sathish, Khader, Sayma and Shah,
[*Centrifugal Compressor Design and Surge Simulation for Active Inference Based
Control*](https://doi.org/10.1115/GT2024-124905) (ASME Turbo Expo 2024,
GT2024-124905), which proposes active inference for anti-surge control. The sensor model
and every choice below are mine. Nothing here is a claim about what that paper did.

## The meter

Anti-surge control on a centrifugal compressor needs flow, and flow is almost always
measured with a differential-pressure element: an orifice plate, a venturi, a flow
nozzle. The element gives a pressure drop, and the flow follows the square-root law

```text
Q = k·√ΔP
```

The transmitter's error is approximately flat across its span. Call it `σ_ΔP ≈ const`.
That is a statement about the instrument, not about the process. Propagate it through
the square root:

```text
σ_Q = |dQ/dΔP|·σ_ΔP = (k / 2√ΔP)·σ_ΔP = k²σ_ΔP / (2Q)
```

So the variance of the flow measurement is

```text
R(Q) = σ_Q² = τ / Q²,        τ = k⁴σ_ΔP² / 4
```

The noise on the flow reading depends on the flow being read. Nobody chose that. It
follows from the square root and a flat transmitter error, and it is why a DP meter has
a turndown limit at all.

## The element type sets the sign

That derivation is about a DP element. It is not a fact about flow metering, and the
sign it produces is what the whole demo turns on:

| Element | Error behaviour | `R(Q)` | `ℓ' = R'/R` | Epistemic term pushes |
| --- | --- | --- | --- | --- |
| Orifice, venturi, nozzle | flat in ΔP span | `∝ 1/Q²` | `−2/Q` | away from surge |
| Coriolis, ultrasonic | roughly constant % of rate | `∝ Q²` | `+2/Q` | toward surge |
| Some thermal mass | constant absolute | constant | `0` | nowhere, it collapses |

Read the middle row. A meter whose relative accuracy is flat has `σ_Q = e·Q`, so
`R = e²Q²` and `ℓ' = +2/Q`. The sign flips, the information gain is now largest at low
flow, and the agent is pulled *toward* surge instead of held off it. Same machinery,
opposite conclusion, and nothing about the inference changed.

The bottom row is the one that matters for whether any of this is worth doing. A
constant-`R` meter puts `ℓ' = 0`, and then the epistemic term is constant across
actions and the agent reduces to a controller.

There is a further wrinkle for anti-surge specifically. Conventional anti-surge control
often works in reduced ΔP coordinates and never takes the square root at all, because
the surge line is approximately a straight line through the origin there. The meter
carries the same information about flow in either coordinate. What changes is where a
model puts the square root. Here it sits in the sensor, as `R(Q)`. In ΔP coordinates
the sensor is linear with flat noise, and the square root moves into the dynamics
because the valve acts on flow. A linear-Gaussian model with additive control
linearises that away. Its `R` is then constant, and it collapses exactly like the
fixed-`R` twin below. Whether an agent has an epistemic term at all depends on the
coordinates its model is written in.

## Calibration

One constant, and I want it in units an engineer recognises. Reparameterise by the
relative error at design flow `Q_ref`:

```text
σ_Q(Q) = e_ref·Q_ref² / Q          R(Q) = e_ref²·Q_ref⁴ / Q²
```

`e_ref` is the relative flow uncertainty at `Q_ref`. I take `e_ref = 0.002`, which is
0.2% at design flow.

**This is the only free constant in the build, and it is an assumption rather than a
derived quantity.** [ISO 5167](https://www.iso.org/standard/79179.html) is where the
computation of flow-rate uncertainty for a DP element is standardised, with orifice
plates in [its second part](https://www.iso.org/standard/79180.html). A real value
belongs to a specific installation rather than to a note like this one. The reasoning
behind the value I use: instrumentation practice quotes orifice-plate flow uncertainty
at roughly ±20% once you are down at 10% of maximum flow. Set `Q/Q_ref = 0.1` in the
expression above and the relative error is
`e_ref·(Q_ref/Q)² = e_ref·100`. So `e_ref = 0.002` reproduces the ±20% figure exactly.

That agreement is not independent evidence. The ±20%-at-10%-flow rule of thumb *is* the
square-root law restated, so the two agreeing confirms the algebra and nothing more.
Treat `e_ref` as a plausible calibration, and re-anchor it against a real datasheet
before any number here is used for anything.

## Coordinates

The demo's latent is the surge margin, not the flow:

```text
m = Q − Q_s           Q = Q_s + m
R(m) = e_ref²·Q_ref⁴ / (Q_s + m)²
```

`Q_s` is the surge-line flow at the operating speed line. Positive `m` is safe, negative
`m` is surge.

`R` is strictly decreasing in `m` on the whole operating range, so `R'(m) ≠ 0`
everywhere the demo evaluates it. That matters beyond tidiness.
`research/gate_d4_registration.md` names `R'(μ) ≠ 0` as a precondition on any fixture
without a mean-moving action, in the amendments dated 2026-08-07. A fixture sitting at a
stationary point of `R` looks like a state-dependent sensor and behaves like a fixed one.
This one is nowhere near that degeneracy.

Near `Q = 0` the expression blows up. The demo clamps `Q` at a floor well below `Q_s`, so
`R` stays finite and positive-definite across the whole action grid. A non-positive-
definite `R` does not raise. It surfaces as a NaN at action selection, which is a worse
failure than an exception.

## What the fixed-`R` twin does, and what it does not say

The twin's epistemic term is constant across actions, so its `argmin G` is its
`argmin pragmatic` exactly. Koudahl, Kouw and de Vries proved this for fixed-noise
linear Gaussian state space models, where EFE minimisation reduces to KL control and
the exploratory drive does no work
([Entropy 23(12):1565](https://doi.org/10.3390/e23121565)). `DECISIONS.md` records it
as ADR-003, and the demo's first two rows measure it.

The scope condition is easy to overstate, so here it is precisely. The collapse is a
property of **the agent's generative model**, not of the plant. It needs a
linear-Gaussian model with additive control and a fixed `R`: control then enters the
mean only, the covariance recursion `P⁻ = APAᵀ + q`, `S = CP⁻Cᵀ + R`, `P⁺` is
action-independent, and the epistemic term `½(ln det S − ln det R)` cannot vary. `q` is
the dynamics noise, written small to keep it apart from the flow `Q`. The plant the
agent is pointed at may be arbitrarily nonlinear without changing any of that.

What it does not cover: an EKF linearised on a nonlinear plant has Jacobians `A(x)` and
`C(x)` that depend on the linearisation point, which depends on the predicted mean,
which depends on the action. Its covariance recursion *is* action-dependent, so its
epistemic term is not flat, with no `R(x)` involved. Whether that variation is enough
to move the argmin is an empirical question and this demo does not answer it.

## Where this sits relative to `c₂`

The log-noise and its derivative are the quantities the programme's gap work is written
in:

```text
ℓ(m) = log R(m) = log(e_ref²Q_ref⁴) − 2·log(Q_s + m)
ℓ'(m) = −2 / (Q_s + m) = −2/Q
```

`research/gate_d4_registration.md` (RESULT 2026-08-07) and
`research/c4_hand_derivation.md` give the leading coefficient of the inference gap as
`c₂ = (ℓ'(μ)/2)²`. For this sensor that is

```text
c₂ = 1/Q²
```

A clean closed form, and the demo asserts `ℓ'` against it.

**The demo does not measure an inference gap.** Two different quantities travel under
similar names, and conflating them is the easy mistake here:

- The **EFE epistemic term**, `½(ln det S − ln det R)`, is what `expected_free_energy`
  returns. It is what varies across the action grid under `R(x)` and goes flat under a
  frozen `R`. This is the demo's content.
- **`c₂`** is the leading σ² coefficient of `KL(q ‖ p(·|y))`, the gap between the
  plug-in-`R` Kalman posterior and the exact non-Gaussian posterior. Measuring it needs
  the quadrature in `research/checks/gap_kernel.py` over the exact predictive. The demo
  runs none of that.

Both are driven by `ℓ' = R'/R` and both vanish when `R' = 0`. That shared root is why
they look like the same object. They are not.

So the claim here is narrow: this compressor sensor lands in the same shape as the
family the gate work is registered on, with `ℓ'` in closed form and nowhere zero. That
is a statement about the sensor.

Two limits on reading `c₂` as the whole gap. It is the same for forward and reverse KL
only at order σ², and the third cumulant separates them above that. It is positive only
locally, because `c₄ < 0` turns the series over at larger spread. Both are recorded in
`research/gate_d4_registration.md`.

## What this does not model

The demo and anything written about it quote the same list:

- No Greitzer dynamics. Surge is a dynamic instability of the compressor-plenum system
  ([Greitzer 1976](https://doi.org/10.1115/1.3446138)), and none of that is here.
- No cubic compressor characteristic.
- No speed line, and no travel along one.
- A local linearisation about a single operating point.
- Horizon-1 EFE only. No multi-step planning.
- The surge point `Q_s` is known and fixed. Treating it as an unknown latent is the
  interesting version of this problem and is not what the demo does.
- A DP element is assumed. The sign table above is what that assumption buys, and a
  different element buys a different sign or none.
- **The meter model is derived from a standard that excludes the regime of interest.**
  [ISO 5167](https://www.iso.org/standard/79179.html) does not apply to pulsating
  flow, and surge is a pulsating flow instability. So `R(Q)` describes the meter on the
  approach to surge and not in it. Anything the demo says about behaviour at the surge
  line inherits that.
- Reduced ΔP coordinates are not modelled. A controller working in them is reading a
  different quantity.
- No gas thermodynamics. No real-gas properties, no composition, no suction conditions.
- **Not a controller benchmark.** The comparison is against a fixed-`R` twin of itself,
  not against PID, MPC or a conventional anti-surge controller. Nothing here says the
  `R(x)` agent controls a compressor better than anything.
- The operating point is illustrative. `Q_ref`, `Q_s` and the design margin describe no
  machine and were not read off a performance map.
- Where the standoff sits is set by the economic weight `Λ`, which is a policy choice.
  Sweeping `Λ` over 10:1 moves the standoff further than the whole meter ladder does.
  The demo's claim is the *direction* of the offset, not its size.
- The standoff is one EFE step from a prior with σ = 2% of design flow. It is not where
  a closed-loop agent settles, and the prior's width scales it as much as `Λ` does.

The demo is a statement about what a state-dependent sensor does to action selection. It
is not a compressor model.
