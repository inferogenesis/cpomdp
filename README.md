# cpomdp

[![PyPI](https://img.shields.io/pypi/v/cpomdp.svg)](https://pypi.org/project/cpomdp/)
[![Python](https://img.shields.io/pypi/pyversions/cpomdp.svg)](https://pypi.org/project/cpomdp/)
[![CI](https://github.com/inferogenesis/cpomdp/actions/workflows/ci.yml/badge.svg)](https://github.com/inferogenesis/cpomdp/actions/workflows/ci.yml)
[![coverage](https://cpomdp.inferogenesis.com/assets/coverage.svg)](https://github.com/inferogenesis/cpomdp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/inferogenesis/cpomdp/blob/main/LICENSE)

Continuous active inference for Python. The continuous-state sibling of [pymdp](https://github.com/infer-actively/pymdp).

pymdp is great, but it speaks in discrete states. A lot of the world isn't discrete. Positions, velocities, temperatures, the kinds of things you'd actually want an agent to track and steer, don't come in neat little categories. cpomdp fills that gap. You hand it a linear-Gaussian model of how the world moves and what you can see of it, and you get back an agent that perceives and acts in the same `infer_states` / `sample_action` loop pymdp users already know.

That's the whole idea: keep the pymdp muscle memory, swap the discrete machinery underneath for continuous.

Full documentation — API reference and guides — lives at [cpomdp.inferogenesis.com](https://cpomdp.inferogenesis.com/).

## Example

Four bacilli seeking food in the same world — the continuous-state answer to pymdp's mouse-seeking-cheese, now with the **epistemic** term v0.3 adds. The twist: the food's position is **hidden**, and a **beacon** marks where the agent can *see* it. Visiting the beacon doesn't sharpen where the agent thinks *it* is — it sharpens where it thinks the *food* is, which it can't act on directly. That makes the information genuinely **instrumental**: resolving it changes where the agent then heads. Each body sits at its **true** hidden state; the blue `+` is where it believes it is, the diamond is where it believes the food is (both with their uncertainty ellipses), and the star is the food's true, hidden location. The four differ in **one number only** — the **goal precision Λ** each is built with. They all minimise the same Expected Free Energy `G = pragmatic − epistemic`; because the pragmatic (goal) term scales with Λ while the epistemic (information) term doesn't, Λ alone tips the balance: **classic LQR** and a **sharp Λ** beeline to the agent's current food guess and never detour; a **balanced Λ** detours to the beacon, learns where the food really is, *then* heads there with confidence; a **weak Λ** is so over-curious it parks at the beacon and never eats. One real knob — the precision you'd actually pass — four behaviours.

![Four bacilli learning where the food is, under different goal precisions Λ, via continuous active inference](https://raw.githubusercontent.com/inferogenesis/cpomdp/main/docs/assets/bacillus_uncertain_food.gif)

Reproduce it with [`examples/bacillus_uncertain_food.py`](https://github.com/inferogenesis/cpomdp/blob/main/examples/bacillus_uncertain_food.py) (`pip install "cpomdp[examples]"`). More — including the v0.3 demo where the beacon reveals the agent's *own* position, and the original v0.2 single-bacillus demo — in the [examples gallery](https://github.com/inferogenesis/cpomdp/blob/main/examples/README.md).

## Install

```bash
pip install cpomdp
```

Or the latest from source:

```bash
pip install git+https://github.com/inferogenesis/cpomdp
```

That's all you need for normal use. There's also an optional RxInfer (Julia) backend that the test suite leans on as a correctness oracle. You almost certainly don't need it, but if you want it:

```bash
pip install "cpomdp[rxinfer]"
```

It pulls in a Julia bridge and bootstraps itself the first time you use it.

## Quickstart

Here's an agent steering a point mass to a target. It can push the mass and it can see where the mass is, but it never sees the velocity. The filter has to work that out from how the position moves.

```python
import jax.numpy as jnp
from cpomdp import Agent, Belief, LinearGaussianModel, StateGoal

# State is [position, velocity]. A push changes velocity, velocity carries
# position along, and we only ever observe position (through a noisy sensor).
dt = 0.1
model = LinearGaussianModel(
    dynamics=[[1, dt], [0, 1]],          # velocity carries position along
    control=[[0], [dt]],                 # a push nudges velocity
    sensor_model=[[1, 0]],               # we observe position only
    dynamics_noise=jnp.eye(2) * 1e-6,
    sensor_noise=[[1e-2]],
    prior=Belief(mean=[0, 0], cov=jnp.eye(2)),
)

# Tell it where to go: sit still at position 1.
agent = Agent(model, StateGoal([1.0, 0.0]))

true_state = jnp.array([0.0, 0.0])
for _ in range(100):
    obs = model.sensor_model @ true_state            # what the agent gets to see
    agent.infer_states(obs)                           # perceive
    action = agent.sample_action()                    # act
    true_state = model.dynamics @ true_state + model.control @ action

print(jnp.round(agent.belief.mean, 3))   # ≈ [1, 0]
```

Run that and the belief lands on `[1, 0]`. The agent worked out it was at position 1 and sitting still, which is exactly where we asked it to go, and it did it without ever seeing the velocity it had to control.

## The pymdp parallel

If you've used pymdp, the loop is the same and most of the names are too. Four carry over verbatim:

> `Agent` · `qs` · `infer_states` · `sample_action`

(`qs` is a read-only alias for `belief`, cpomdp's canonical name — so `agent.qs` and `agent.belief` are the same posterior. Use whichever your fingers reach for.)

Only two things are spelled differently:

| pymdp | cpomdp                          | what it is                           |
| ----- | ------------------------------- | ------------------------------------ |
| `C`   | `StateGoal` / `ObservationGoal` | the goal you pursue, and how sharply |
| `D`   | `model.prior`                   | belief before you've seen anything   |

One honest difference in behaviour. `sample_action` here is deterministic, not a sample from a policy posterior. For a linear-Gaussian sensor the action that minimises expected free energy turns out to be exactly the LQR optimum, so there's a single best action and that's what comes back. Same loop, exact answer. The reasoning is in [DECISIONS.md](https://github.com/inferogenesis/cpomdp/blob/main/DECISIONS.md) (ADR-003) if you want it.

## Just want to track, not act?

A model with no `control` matrix is a pure tracker. Drop the goal and `infer_states` still folds in observations and sharpens the belief, while `sample_action` stops you — there's nothing to steer toward, and nothing to steer with.

```python
tracker = LinearGaussianModel(        # no control matrix -> pure tracking
    dynamics=[[1, dt], [0, 1]],
    sensor_model=[[1, 0]],
    dynamics_noise=jnp.eye(2) * 1e-6,
    sensor_noise=[[1e-2]],
    prior=Belief(mean=[0, 0], cov=jnp.eye(2)),
)
agent = Agent(tracker)                # no objective
agent.infer_states([0.5])             # perceiving is fine
agent.sample_action()                 # ValueError: this Agent has no objective ...
```

## What it handles

cpomdp handles linear-Gaussian models end to end — and, as of v0.3, a little past the "linear-Gaussian" label. The mean dynamics and observations are linear and the noise is Gaussian, so perception is **exact Kalman filtering**, no approximation. For action you get both steady-state **LQR** (reach a target state) and **Expected Free Energy** selection (seek information) — the epistemic, information-seeking behaviour that arrived in v0.3.

What v0.3 added beyond the fixed linear-Gaussian model is state-dependence in the *noise*. The mean stays linear, but:

- **state-dependent sensing `R(x)`** — the observation noise can vary with the state, so some places see more sharply than others (the beacon in the example above);
- **state-dependent process noise `Q(x)`** — the dynamics can diffuse more in some states than others;
- an **H-step planning horizon** for EFE selection;
- an optional **declarable model-structure layer**.

It's the state-dependent noise that gives the agent a reason to seek information: when sensing is sharper somewhere, going there is worth something. The mean is still linear, though — genuinely **nonlinear sensors** (a curved `g(x)` that needs a second-order moment match) are the next step, not here yet.

## Swappable backends

You can swap the inference engine if you want to. `KalmanBackend` is the default and does the real work; `RxInferBackend` re-derives the same answers through Julia and exists mainly so the fast path has something independent to check itself against. Both sit behind the `InferenceBackend` protocol, so you can write your own.

## Status

Still pre-1.0: v0.3 aims to secure the public API, however if you have a request or suggestion to make this front-facing API more usable please open a GitHub issue, I'm happy to listen. Until 1.0 a minor version is where breaking changes can land.

## Development

I designed and built cpomdp — the architecture, the conditionally-linear-Gaussian formulation, the API, and every decision in [DECISIONS.md](https://github.com/inferogenesis/cpomdp/blob/main/DECISIONS.md) are mine. The design draws on my day-to-day work as a full-time software engineer and on hands-on expertise integrating and developing large machine-learning models at scale using event-driven microservice architecture.

I used an AI coding assistant (Claude Opus-4.8) as a tool under close review: to draft docstrings, probe for edge cases and candidate bugs, and expand the test suite, including adversarial ones. Everything it produced I read, checked, and approved before it landed. None of it is taken on trust — the numbers are validated independently against the RxInfer (Julia) and analytic NumPy oracles described above. Correctness rests on those checks, not on the tool that helped write the code.

## Contributions

If you would like to contribute either your dev time or help steer the direction of the toolbox, please add a GitHub issue or discussion thread. I am monitoring this repository closely and would love to collaborate.

If you notice a better method in something I've already done or are just curious and want to chat I am more than happy to talk through my decision processes. I intend to blog my construction of cpomdp provided it doesn't interfere with developing it.

## Acknowledgements

Thanks to **Kevin Backhouse** (Postgraduate Researcher in Cognitive Neuroscience, Durham University) for guidance on the active-inference formulation, collaboration on related discrete generative-model projects, and for being a consistent sounding board throughout the design of this work.
