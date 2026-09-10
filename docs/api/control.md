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

## The control bracket

??? note "In plain terms"
    Two costs bound every controller that follows a plan. The lower one is what the
    plan costs when the state is known exactly at every step. The upper one is the
    best any controller can do when it has to work the state out from readings. The
    gap between them is the price of not knowing: what the sensor fails to deliver.
    That gap is the number to report. Either end on its own is a cost in units nothing
    calibrates.

    Under fixed noise the upper end is settled. Take the gains the full-information
    plan would use and apply them to the filtered estimate instead of the state. That
    is the certainty-equivalent controller, and the separation principle says nothing
    that infers the state does better. Its cost is written down two ways that share
    nothing but the two Riccati recursions: once by propagating how the state and the
    estimate spread together, once as the floor plus what the estimate's error costs at
    each step. The two agreeing to machine precision is the signature of the
    fixed-noise regime, and it is what makes the upper end a closed form rather than a
    measurement.

    An agent's cost is then read as a position inside the bracket. Zero means it did
    exactly as well as certainty equivalence. One means it did as well as knowing the
    state. An agent that can move its own sensor to where the readings are sharper can
    sit above zero, and that is the effect the bracket exists to measure. The bar on
    the agent's cost, divided by the width, is the floor below which its position
    cannot be told from zero.

::: cpomdp.control.full_information_cost

::: cpomdp.control.kalman_schedule

::: cpomdp.control.KalmanSchedule

::: cpomdp.control.closed_loop_cost

::: cpomdp.control.CertaintyEquivalentController

::: cpomdp.control.optimal_cost

::: cpomdp.control.ControlBracket

::: cpomdp.control.control_bracket

::: cpomdp.control.control_efficiency
