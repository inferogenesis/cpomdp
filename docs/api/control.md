# Control

Steady-state LQR action selection: the action-side dual of the Kalman filter. The `Agent` builds one of these for you when you give it a goal; you rarely touch it directly.

!!! note "Internal — not part of the public API"
    `LQRController` is not exported from `cpomdp` and carries no stability promise; the `Agent` constructs it for you. It's documented here for the architecture it illustrates — LQR as the fixed-sensor reduction of active inference [@koudahl2021epistemics] (ADR-003). Build agents with `StateGoal`, not this directly.

::: cpomdp.control.LQRController

## The finite-horizon schedule

??? note "In plain terms"
    A controller steers a system toward a goal. `LQRController` builds one by solving a
    puzzle that asks how hard to push right now, given where the state is and given
    forever to reach the goal. It answers by repeating one calculation until the answer
    stops changing. That final answer is the steady-state gain, and the controller keeps
    only it.

    The intermediate answers were never junk. After one repetition the answer is how hard
    to push with one step left. After five, with five steps left. `finite_horizon_lqr`
    runs the same calculation a chosen number of times, `H`, and keeps every answer. The
    result is a schedule: the right push with `H` steps left, then with `H − 1`, down to
    the last one.

    The expected-free-energy planner looks a fixed number of steps ahead, applies the
    first action and plans again. A planner like that is using the "`H` steps left" rule
    at every step. The "forever" rule is close to it when `H` is large and visibly
    different when `H` is small. Held against the forever rule, the planner shows a gap
    that fades as `H` grows and looks exactly like a bug. `first_gain` is the matching
    rule, so no such gap is manufactured.

    One convention to know. The cost is charged on the state each action arrives at, and
    nothing is charged after the last action. The planner's pragmatic term does the same
    accounting, so the two line up without adjustment.

::: cpomdp.control.finite_horizon_lqr

::: cpomdp.control.FiniteHorizonLQR

::: cpomdp.control.kalman_schedule

::: cpomdp.control.KalmanSchedule
