# Backends

The inference engine is swappable behind the `InferenceBackend` protocol. `KalmanBackend` is the default fast path; `RxInferBackend` — imported from `cpomdp.backends.rxinfer` and gated behind the optional `rxinfer` extra — re-derives the same answers through Julia and exists as an independent correctness oracle [@bagaev2023rxinfer].

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
