# Backends

The inference engine is swappable behind the `InferenceBackend` protocol. `KalmanBackend` is the default fast path; `RxInferBackend` — imported from `cpomdp.backends.rxinfer` and gated behind the optional `rxinfer` extra — re-derives the same answers through Julia and exists as an independent correctness oracle [@bagaev2023rxinfer].

::: cpomdp.InferenceBackend

::: cpomdp.KalmanBackend

::: cpomdp.backends.rxinfer.RxInferBackend
