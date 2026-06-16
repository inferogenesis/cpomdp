# cpomdp

Continuous active inference for Python. The continuous-state sibling of [pymdp](https://github.com/infer-actively/pymdp).

pymdp is great, but it speaks in discrete states. A lot of the world isn't discrete. Positions, velocities, temperatures, the kinds of things you'd actually want an agent to track and steer, don't come in neat little categories. cpomdp fills that gap. You hand it a linear-Gaussian model of how the world moves and what you can see of it, and you get back an agent that perceives and acts in the same `infer_states` / `sample_action` loop pymdp users already know.

That's the whole idea: keep the pymdp muscle memory, swap the discrete machinery underneath for continuous.

## Install

```bash
pip install cpomdp
```

Or the latest from source:

```bash
pip install git+https://github.com/DanBoringName/cpomdp
```

There's also an optional RxInfer (Julia) backend the test suite leans on as a correctness oracle. You almost certainly don't need it, but if you want it:

```bash
pip install "cpomdp[rxinfer]"
```

## Quickstart

Here's an agent steering a point mass to a target. It can push the mass and it can see its position, but it never sees velocity. The filter has to work that out from how the position moves.

```python
import jax.numpy as jnp
from cpomdp import Agent, Belief, LinearGaussianModel

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
agent = Agent(model, goal=[1.0, 0.0])

true_state = jnp.array([0.0, 0.0])
for _ in range(100):
    obs = model.sensor_model @ true_state            # what the agent gets to see
    agent.infer_states(obs)                           # perceive
    action = agent.sample_action()                    # act
    true_state = model.dynamics @ true_state + model.control @ action

print(jnp.round(agent.belief.mean, 3))   # ≈ [1, 0]
```

The belief lands on `[1, 0]`: the agent worked out it was at position 1 and sitting still, which is exactly where we asked it to go, and it did it without ever seeing the velocity it had to control.

## The pymdp parallel

If you've used pymdp, this table is basically the whole API:

| pymdp (discrete) | cpomdp (continuous)       | what it is                          |
| ---------------- | ------------------------- | ----------------------------------- |
| `Agent`          | `Agent`                   | the stateful thing you drive        |
| `qs`             | `belief`                  | the posterior over the state        |
| `infer_states`   | `infer_states`            | fold in an observation              |
| `sample_action`  | `sample_action`           | pick an action                      |
| `C`              | `goal` + `goal_precision` | the state you prefer, how sharply   |
| `D`              | `model.prior`             | belief before you've seen anything  |

One honest difference. `sample_action` here is deterministic, not a sample from a policy posterior. For a linear-Gaussian sensor the action that minimises expected free energy turns out to be exactly the LQR optimum, so there's a single best action and that's what comes back. Same loop, exact answer.

## Where next

- [Agent](api/agent.md) — the façade you drive
- [Model & belief](api/model.md) — `LinearGaussianModel` and `Belief`
- [Backends](api/backends.md) — Kalman (default) and the RxInfer oracle
- [Control](api/control.md) — the steady-state LQR controller

!!! note "Status"
    This is pre-alpha. v0.1 handles linear-Gaussian models: Kalman filtering for perception, steady-state LQR for action. The API works and the maths is tested against an independent oracle, but expect things to move before 1.0.
