# Backends

The inference engine is swappable behind the `InferenceBackend` protocol. `KalmanBackend` is the default. On a fixed-noise model it is the exact Kalman filter. Under a state-dependent `R(x)` or `Q(x)` it reads the noise at the predicted mean, a first-order plug-in. `RxInferBackend` is imported from `cpomdp.backends.rxinfer` and needs the optional `rxinfer` extra. It re-derives the fixed-noise answers through Julia as an independent correctness oracle [@bagaev2023rxinfer], and it refuses a model with `R(x)` or `Q(x)`.

::: cpomdp.InferenceBackend

::: cpomdp.KalmanBackend

::: cpomdp.backends.rxinfer.RxInferBackend

`CouplingGraphBackend` is the branching peer: message passing on a [`CouplingGraph`][cpomdp.CouplingGraph] rather than a chain. A state-dependent `R(x)` on a coupled node cannot be flattened to a fixed linear-Gaussian model, and asking it to raises `IncompatibleLinearizationError`.

::: cpomdp.CouplingGraphBackend

::: cpomdp.IncompatibleLinearizationError

A carry partition that severs an edge the model flags as EFE-load-bearing filters fine. The EFE selectors, rollouts and searches refuse it with `InadmissiblePartitionError`.

::: cpomdp.InadmissiblePartitionError
