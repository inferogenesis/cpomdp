# Diagnostics

Whether a state-dependent `R(x)` earns its keep is a question about the states an action can actually reach. A noise that varies only where no policy goes is, for every purpose the filter and the objective have, a constant. `probe_model` samples the reachable set and reports back a `SensorReport`: whether `R` is positive definite at every sample, whether it moves at all, and whether the epistemic value moves with it. The set is sampled, not exhausted, so a negative is evidence rather than proof.

::: cpomdp.probe_model

::: cpomdp.SensorReport

A flat `LinearGaussianModel` and a graph backend reach their predicted means by different routes, so `probe_model` takes either. `ProbeBackend` is the three members it needs from the second.

::: cpomdp.diagnostics.ProbeBackend
