# Examples gallery

Runnable scripts that render the figures in the docs and README. They are **not**
part of the installed package (only `src/cpomdp` ships in the wheel) — they import
plotting libraries the core does not depend on.

Get the plotting deps with the `examples` extra:

```bash
pip install "cpomdp[examples]"        # then: python examples/<script>.py
```

…or, from a source checkout, with uv (no install needed):

```bash
uv run --extra examples python examples/<script>.py
```

Each script writes its asset into [`../docs/assets/`](../docs/assets) and takes an
optional output path as `argv[1]`.

---

## Flagship — four bacilli, one knob (the epistemic weight λ)

[`bacillus_seeking_food.py`](bacillus_seeking_food.py) · v0.3

Four bacilli in one world, differing in a single number: the weight **λ** on the
*information-seeking* term of the Expected Free Energy each minimises
(`G = pragmatic − λ·epistemic`). A **beacon** marks where the sensor is sharp, so
visiting it collapses the agent's uncertainty. Classic LQR beelines to the food;
too little λ barely deflects; the right λ detours to the beacon to localise *then*
heads to the food; too much λ and it never leaves. The simulation is real — every
agent shares one Kalman filter over a `CallableSensor` whose `R(x)` dips at the
beacon, and the EFE agents call the library's own `expected_free_energy` kernel.

![Four bacilli navigating to food under different epistemic weights λ](../docs/assets/bacillus.gif)

`bacillus_seeking_food.py --scan` prints the λ-sweep tuning metrics without rendering.

---

## The journey

### Bacillus seeking food — the original (pure LQR)

[`bacillus_lqr.py`](bacillus_lqr.py) · v0.2

Where it started: a *single* bacillus with a fixed sensor. Here the epistemic term
collapses to nothing (ADR-003) and active inference reduces to LQR — it simply
perceives, acts, and arrives. The flagship above is its v0.3 successor, switching
the epistemic term back on with a state-dependent sensor.

![A bacillus navigating to food via continuous active inference](../docs/assets/bacillus_lqr.gif)

### EFE epistemic collapse, and how a state-dependent sensor breaks it

[`efe_collapse_figure.py`](efe_collapse_figure.py)

Sweeps a one-step action and plots `G = pragmatic − epistemic` for a fixed sensor
(epistemic dead-flat → EFE collapses to LQR) versus a state-dependent sensor (a
precision well makes the epistemic term curve, pulling the argmin off the goal
toward the information). The "why v0.3 exists" figure.

![EFE epistemic collapse and the state-dependent-sensor detour](../docs/assets/efe_collapse.png)

### Internal process noise breaks the collapse from the inside

[`internal_noise_figure.py`](internal_noise_figure.py)

The companion: here the sensor noise `R` is held fixed and the action-dependence of
the epistemic term comes entirely from state-dependent **process** noise `Q(x)` —
the internal-precision route of RFC-001 §8.

![Internal process noise breaking the epistemic collapse](../docs/assets/internal_noise.png)
