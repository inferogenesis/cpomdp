# Backends

The inference engine is swappable behind the `InferenceBackend` protocol. `KalmanBackend` is the default fast path; `RxInferBackend` re-derives the same answers through Julia and exists as an independent correctness oracle.

::: cpomdp.InferenceBackend

::: cpomdp.KalmanBackend

::: cpomdp.backends.rxinfer.RxInferBackend
