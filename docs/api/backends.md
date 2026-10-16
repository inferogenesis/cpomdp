# Backends

The inference engine is swappable behind the `InferenceBackend` protocol. `KalmanBackend` is the default. The `Agent` runs it per step, and the gain and covariance are recomputed from each belief. `KalmanBackend(model, steady_state=True)` freezes them at construction instead, for a fixed-noise, time-invariant model read every step, and reaches an agent through `Agent.from_backend`. On a fixed-noise model the per-step filter is the exact Kalman filter. Under a state-dependent `R(x)` or `Q(x)` it reads the noise at the predicted mean, a first-order plug-in. `RxInferBackend` is imported from `cpomdp.backends.rxinfer` and needs the optional `rxinfer` extra. It re-derives the fixed-noise answers through Julia as an independent correctness oracle [@bagaev2023rxinfer], and it refuses a model with `R(x)` or `Q(x)`.

::: cpomdp.InferenceBackend

::: cpomdp.KalmanBackend

::: cpomdp.backends.rxinfer.RxInferBackend

`CouplingGraphBackend` filters a [`CouplingGraph`][cpomdp.CouplingGraph] whose nodes each carry their own dynamics. It reads like a flat backend. Its `model` holds the stacked sensor as `observation_matrix` and the joint width as `n_states`. `stack_observations` turns readings keyed by node into the vector `infer_states` takes, and `marginal` reads one node back out of the joint belief. A state-dependent `R(x)` on a coupled node cannot be flattened to a fixed linear-Gaussian model, and asking it to raises `IncompatibleLinearizationError`.

??? note "In plain terms"
    Some systems are a web of parts that run at their own pace and push on one another.
    A cell's signalling chain is one. `CouplingGraphBackend` keeps a single belief over
    every part at once, and each step updates all of it from whichever parts were
    measured. A part nobody measures still grows more certain when a part it pushes on
    is measured.

    One convention to know. The backend takes each step's readings as one vector, the
    measured parts in ascending node order. Hand `stack_observations` a dictionary from
    node to reading and it builds that vector, so the order never has to be typed by
    hand.

    `partition` trades accuracy for speed. Grouping the parts tells the filter which
    links between groups it may forget from one step to the next. The default keeps one
    group and forgets nothing. `severed_mass` reports how much a grouping forgets.

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
| `EFESelector.select`, `FFGEFESelector.select`, `cpomdp.enumeration.RecedingHorizonSelector.select` | Traces. The action is an argmin over fixed candidates, so its gradient is zero |
| `Agent.infer_states`, `cpomdp.enumeration.OpenLoopSelector.select` | Does not. Each keeps state between calls. A traced call restarts from the state it was traced with and leaves a tracer behind |
| `Agent.sample_action` | Does not. It reads the action on the host to refuse a NaN, so a traced call raises before it keeps anything |
| `KalmanBackend(model, steady_state=True)`, `LQRController(...)`, `Agent(model, goal)` | Does not. Each solves once on the host at construction |
| `RxInferBackend.infer_states` | Does not. It calls Julia |
| `probe_model` | Does not. It runs in NumPy on the host |

[`tests/test_transform_table.py`](https://github.com/inferogenesis/cpomdp/blob/main/tests/test_transform_table.py) runs every row except `RxInferBackend`'s, which needs Julia. A row that traces runs under `jit`, `vmap` and `grad`, and a row that does not is shown failing under `jit`. The `lax.scan` loop is on the Agent page.
