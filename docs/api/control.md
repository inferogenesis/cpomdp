# Control

Steady-state LQR action selection: the action-side dual of the Kalman filter. The `Agent` builds one of these for you when you give it a goal; you rarely touch it directly.

!!! note "Internal — not part of the public API"
    `LQRController` is not exported from `cpomdp` and carries no stability promise; the `Agent` constructs it for you. It's documented here for the architecture it illustrates — LQR as the fixed-sensor reduction of active inference (ADR-003). Build agents with `StateGoal`, not this directly.

::: cpomdp.control.LQRController
