# Observation models

How a hidden state produces a sensor reading. `FixedSensor` is a constant linear sensor; `CallableSensor` lets the observation noise `R(x)` vary with the state — the reason an agent has anything to gain from seeking information. Both satisfy the `ObservationModel` protocol.

::: cpomdp.ObservationModel

::: cpomdp.FixedSensor

::: cpomdp.CallableSensor
