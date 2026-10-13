# Anti-surge control

An active-inference agent holds a centrifugal compressor's surge margin, and it reads
flow through a meter with a real error model. The meter decides where it stands. A
differential-pressure meter keeps it on the safe side of its setpoint. A Coriolis meter
of the same accuracy settles it past the surge line.

The numbers come from a showcase,
[`examples/surge_margin.py`](https://github.com/inferogenesis/cpomdp/blob/main/examples/surge_margin.py),
and are not citable facts. The application is my reading of Sathish, Khader, Sayma and
Shah, [*Centrifugal Compressor Design and Surge Simulation for Active Inference Based
Control*](https://doi.org/10.1115/GT2024-124905) (ASME Turbo Expo 2024, GT2024-124905).
Every modelling choice is mine. Nothing here is a claim about what that paper did.

## Results

![Four panels. Top left, the fixed-R agent: its epistemic curve is a flat horizontal line, so the minimum of G sits exactly on the minimum of the pragmatic term. Top right, the DP meter: its epistemic curve rises away from the surge line and pulls the minimum of G off the economic optimum. Bottom left, a Coriolis meter: its epistemic curve rises toward the surge line and the minimum of G lands past it. Bottom right, the standoff against meter uncertainty: a dashed curve where the DP agent's first step aims, a solid curve below it where the loop settles with a star where the library's own Agent settled, a flat line for the fixed-R twin at the preference, and a single diamond for the Coriolis meter just below the surge line.](../assets/surge_margin.png)

The DP and Coriolis meters are both 0.2% accurate at design flow. The fixed-`R` twin is
the DP meter frozen at the operating point. All three agents share one plant and one
economic preference. Only the error's shape differs.

| Meter | Flow variance `R(Q)` | Where the agent settles | Rows |
| --- | --- | --- | --- |
| Fixed `R`, frozen at the operating point | constant | on the preference | 2, 8 |
| Differential pressure: orifice, venturi, nozzle | `∝ 1/Q²` | above the preference, on the safe side | 7, 8, 9 |
| Coriolis | `∝ Q²` | below the preference, past the surge line | 10, 11 |

The rows are the falsifiers `--check` runs. Each one fires if its claim fails, and none
does. The settled margins come from a closed-form law and from the library's own
`Agent` run in closed loop. The two agree to within a grid step.

A worse DP meter moves the agent closer to surge, on its first step and once settled.
Rows 3 and 9 assert that direction.

## Why the meter decides it

The epistemic term in expected free energy values a reading by how much it would teach.
Quiet readings teach most. So the term pulls the agent toward the flows its own meter
reads best.

A DP element reads flow through a square root, `Q = k·√ΔP`, and its transmitter error is
roughly flat. Propagate one through the other and `R ∝ 1/Q²`. The meter is quiet at
high flow and noisy near surge, so the pull is away from surge. A Coriolis meter's error
is a constant percentage of rate. That gives `R ∝ Q²`, quietest at low flow, and the
pull runs toward surge. Constant absolute error gives no pull. The agent then collapses
onto its economic optimum, as Koudahl, Kouw and de Vries proved for fixed-noise linear
Gaussian models ([Entropy 23(12):1565](https://doi.org/10.3390/e23121565)).

[The derivation](surge_margin_derivation.md) makes this exact. It gives a closed-form
law for where the agent stands, and the sign of its offset from the preference is the
sign of `−d log R / dm`.

## What it means for anti-surge

The ambiguity term is not a safety factor. It is an information term. With a DP meter
it happens to hold the agent off surge, and that is easy to read as caution. With a
Coriolis meter the same term pulls the agent in. Real surge margin prices the cost of
being wrong, which is a pragmatic quantity. It belongs in the preference or in a hard
limit.

The coordinates matter too. Conventional anti-surge control often works in reduced ΔP
coordinates and never takes the square root. Write a linear-Gaussian model with
additive control in those coordinates and its `R` is constant. The epistemic term then
vanishes and the agent reduces to a controller.

So an active-inference controller on a compressor needs its meter's error model. Above
all it needs the sign.

## Limits

The demo leaves a lot out.
[The derivation](surge_margin_derivation.md#what-this-does-not-model) lists all of it.
Four omissions matter most here.

- **No surge dynamics.** There is no Greitzer model, no compressor characteristic, no
  speed line and no gas thermodynamics. The surge line is a fixed number.
- **The DP meter model stops short of surge.** It comes from ISO 5167, which excludes
  pulsating flow. It describes the meter on the approach to surge and not in it.
- **The Coriolis meter has no zero-stability term.** A real one adds a fixed error near
  zero flow. That stops `R` falling and weakens the pull toward low flow.
- **Not a controller benchmark.** The comparison is with a fixed-`R` twin of the same
  agent. Nothing here says it controls a compressor better than a conventional
  anti-surge controller.

Where the agent stands is set by the economic weight `Λ`, which is a policy choice. The
demo's claim is the direction of the offset. Its size is not a result.

## Run it

```bash
uv run python examples/surge_margin.py --check                  # the eleven falsifiers
uv run --extra examples python examples/surge_margin.py         # the figure above
```

`--check` needs no plotting libraries. It prints the numbers behind every row.
