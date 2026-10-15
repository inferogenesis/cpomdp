# Backends

The inference engine is swappable behind the `InferenceBackend` protocol. `KalmanBackend` is the default fast path; `RxInferBackend` — imported from `cpomdp.backends.rxinfer` and gated behind the optional `rxinfer` extra — re-derives the same answers through Julia and exists as an independent correctness oracle [@bagaev2023rxinfer].

::: cpomdp.InferenceBackend

::: cpomdp.KalmanBackend

::: cpomdp.backends.rxinfer.RxInferBackend

`CouplingGraphBackend` is the branching peer: message passing on a [`CouplingGraph`][cpomdp.CouplingGraph] rather than a chain. A state-dependent `R(x)` on a coupled node cannot be flattened to a fixed linear-Gaussian model, and asking it to raises `IncompatibleLinearizationError`.

::: cpomdp.CouplingGraphBackend

::: cpomdp.IncompatibleLinearizationError

A carry partition that severs an edge the model flags as EFE-load-bearing filters fine. The EFE selectors, rollouts and searches refuse it with `InadmissiblePartitionError`.

::: cpomdp.InadmissiblePartitionError
