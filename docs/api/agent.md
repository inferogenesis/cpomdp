# Agent

The stateful façade you drive: `infer_states` to perceive, `sample_action` to act.

The `Agent` is eager. Its methods update the belief it holds, so `jax.jit`, `vmap` and
`grad` do not apply to them. A jitted `infer_states` returns the posterior it was traced
with on every call. To compile a loop, drive a backend directly under `jax.lax.scan`.

```python
import jax
import jax.numpy as jnp

from cpomdp import Belief, KalmanBackend, LinearGaussianModel

model = LinearGaussianModel(
    [[1.0, 0.1], [0.0, 1.0]],
    control_matrix=[[0.0], [0.1]],
    observation_matrix=[[1.0, 0.0]],
    dynamics_noise=jnp.eye(2) * 1e-6,
    observation_noise=[[1e-2]],
    prior=Belief(mean=[0.0, 0.0], cov=jnp.eye(2)),
)
backend = KalmanBackend(model)


def step(belief, reading):
    observation, action = reading
    belief = backend.infer_states(observation, belief, action)
    return belief, belief.mean


observations = jnp.ones((50, 1))
actions = jnp.zeros((50, 1))
final, means = jax.jit(lambda xs: jax.lax.scan(step, model.prior, xs))(
    (observations, actions)
)
```

A selector's `select(belief, preference)` is pure as well. Call it inside `step` to
choose each action within the compiled loop.

::: cpomdp.Agent
