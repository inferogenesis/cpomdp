# Action selection & goals

What the agent wants, and how it picks an action. `StateGoal` and `ObservationGoal` are the goals you hand the `Agent`, the continuous-state answer to pymdp's `C`. Each one also describes the selector the `Agent` builds from it. A `Preference` is the target and precision alone, the part every selector reads. Hand the `Agent` a `Preference` together with a selector of your own when no goal describes the search you want. `EFESelector` is the grid search the `Agent` builds for an `ObservationGoal`.

::: cpomdp.StateGoal

::: cpomdp.ObservationGoal

::: cpomdp.Preference

::: cpomdp.EFESelector

`ActionSelector` is the protocol every selector satisfies. `LQRSelector` is the fixed-noise path. With a fixed sensor and fixed process noise, expected free energy provably reduces to LQR [@koudahl2021epistemics] (ADR-003). `FfgEfeSelector` is `EFESelector`'s peer for a branching backend.

::: cpomdp.ActionSelector

::: cpomdp.LQRSelector

::: cpomdp.FfgEfeSelector
