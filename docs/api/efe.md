# Expected free energy

The scalar each candidate action is scored by, `G = pragmatic − epistemic`. The pragmatic term pulls toward preferred outcomes. The epistemic term rewards actions that sharpen the belief. Lower `G` is better.

??? note "In plain terms"
    An agent choosing an action asks two questions about it. Where will it leave me
    relative to what I want to see? How much will the next reading tell me about where I
    am? `expected_free_energy` answers both and returns them as one number per action.
    Lower is better.

    The first answer is the pragmatic term. It is a cost. It grows with the distance
    between the reading the agent expects and the reading it wants, and with how blurred
    that expected reading is. The second answer is the epistemic term. It is a value, and
    it measures how much the reading will sharpen the agent's belief about its state. The
    score is the cost minus the value, so an action can win by landing near the goal or
    by being informative.

    The goal is a reading. The agent says what it wants to observe, and that is a state
    only when the sensor reads the state directly. Pass that reading as a `Preference`.
    An `ObservationGoal` you already hold hands over its own as `goal.preference`.

    Fix the sensor's noise and the dynamics' noise, and every action sharpens the belief
    by the same amount. The epistemic term is then a constant. The best action is the
    one a classic regulator picks, and that regulator is worked out once, with no
    candidates to score. The information term matters only when the noise depends on
    where the action takes the state.

::: cpomdp.expected_free_energy

::: cpomdp.efe.EfeResult

## Scoring a plan over several steps

??? note "In plain terms"
    `expected_free_energy` looks one step ahead. Some actions pay off later. A push
    changes a velocity first and a reading only on the step after, so a one-step look
    sees no difference between pushes. `policy_efe` scores a whole sequence of actions
    instead. It steps the belief forward through the sequence and adds up each step's
    score.

    Between steps it carries the belief the agent expects to hold. The mean follows the
    action. The uncertainty shrinks as if each expected reading had arrived. Under a
    Gaussian model that shrinkage does not depend on what the reading says, so it can be
    worked out before any reading exists.

    The selectors and the searches call it once per candidate sequence. That is the work
    each decision costs. `policy_efe_trace` runs the same steps and keeps every step's
    numbers, for inspecting one sequence rather than ranking many.

::: cpomdp.efe.policy_efe

::: cpomdp.efe.policy_efe_trace

::: cpomdp.efe.PolicyEfeTrace

## On a factor graph

The same two rollouts over a `CouplingGraphBackend`, with the epistemic term aimed at one node's block of the joint state.

::: cpomdp.efe.policy_efe_ffg

::: cpomdp.efe.policy_efe_ffg_trace
