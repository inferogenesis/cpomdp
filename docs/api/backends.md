# Backends

The inference engine is swappable behind the `InferenceBackend` protocol. `KalmanBackend` is the default. The `Agent` runs it per step, and the gain and covariance are recomputed from each belief. `KalmanBackend(model, steady_state=True)` freezes them at construction instead, for a fixed-noise model, and reaches an agent through `Agent.from_backend`. On a fixed-noise model the per-step filter is the exact Kalman filter. Under a state-dependent `R(x)` or `Q(x)` it reads the noise at the predicted mean, a first-order plug-in. `RxInferBackend` is imported from `cpomdp.backends.rxinfer` and needs the optional `rxinfer` extra. It re-derives the fixed-noise answers through Julia as an independent correctness oracle [@bagaev2023rxinfer], and it refuses a model with `R(x)` or `Q(x)`.

::: cpomdp.InferenceBackend

::: cpomdp.KalmanBackend

::: cpomdp.backends.rxinfer.RxInferBackend

`CouplingGraphBackend` is the branching peer: message passing on a [`CouplingGraph`][cpomdp.CouplingGraph] rather than a chain. A state-dependent `R(x)` on a coupled node cannot be flattened to a fixed linear-Gaussian model, and asking it to raises `IncompatibleLinearizationError`.

::: cpomdp.CouplingGraphBackend

::: cpomdp.IncompatibleLinearizationError

A carry partition that severs an edge the model flags as EFE-load-bearing filters fine. The EFE selectors, rollouts and searches refuse it with `InadmissiblePartitionError`.

::: cpomdp.InadmissiblePartitionError

## Which names trace

The JAX backends, the EFE functions and the selectors' `select` are pure over pytrees.
They compose under `jax.jit`, `vmap`, `grad` and `lax.scan`. The `Agent` is an eager
convenience over them, and its page shows the same loop written under `lax.scan`.

Construction is host-side work done once. Build outside the transform and close over the
result. `Belief`, `LinearGaussianModel` and `Preference` are pytrees, so they pass as
arguments, and under a trace their construction checks are skipped because the values
are abstract. Goals, selectors and backends are not pytrees. They enter a transform as
closures.

| Name | Under `jit`, `vmap`, `grad` and `lax.scan` |
| --- | --- |
| `KalmanBackend.infer_states` | Traces, per-step or steady-state, under fixed noise, `R(x)` or `Q(x)` |
| `CouplingGraphBackend.infer_states` | Traces |
| `expected_free_energy`, `cpomdp.efe.policy_efe` | Traces |
| `LQRSelector.select` | Traces, with a gradient through the gain |
| `EFESelector.select`, `FfgEfeSelector.select`, `cpomdp.enumeration.RecedingHorizonSelector.select` | Traces. The action is an argmin over fixed candidates, so its gradient is zero |
| `Agent.infer_states`, `Agent.sample_action`, `cpomdp.enumeration.OpenLoopSelector.select` | Does not. Each keeps state between calls. A traced call restarts from the state it was traced with and leaves a tracer behind |
| `KalmanBackend(model, steady_state=True)`, `LQRController(...)`, `Agent(model, goal)` | Does not. Each solves once on the host at construction |
| `RxInferBackend.infer_states` | Does not. It calls Julia |
| `probe_model` | Does not. It runs in NumPy on the host |

[`tests/test_transform_table.py`](https://github.com/inferogenesis/cpomdp/blob/main/tests/test_transform_table.py) runs every row.
