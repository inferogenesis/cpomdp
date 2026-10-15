# Agent

The stateful façade you drive: `infer_states` to perceive, `sample_action` to act.

??? note "Variations of Agent"
    `Agent(model, goal)` filters with the per-step Kalman filter.
    `Agent.from_backend(backend, goal)` takes any other engine, a graph model included,
    and plans under `backend.model`. Both lead into the same choices below.

    A curious agent is one whose information term changes with the action. It comes out
    of an `ObservationGoal` on a sensor whose noise depends on the state, and nowhere
    else. Green marks it. Amber is EFE whose information term is the same for every
    action, blue is LQR, grey a tracker, a red outline a refusal at construction, and a
    dashed red outline a combination that runs and does not do what its name suggests.

    ```mermaid
    flowchart LR
        none["no goal<br/>Agent(model)"]
        state["StateGoal(target)"]
        obs["ObservationGoal(target, …)"]

        none -- "model without control" --> track["Tracker: perceives, never acts"]:::track
        none -- "model with control" --> stuck["Tracker whose infer_states raises:<br/>it has no action to predict with"]:::gap
        state -- "fixed sensor,<br/>target held still" --> lqr["LQR: no information term"]:::lqr
        state -- "fixed sensor,<br/>target drifts" --> r1["Refused: LQR would settle short"]:::refused
        state -- "R(x) sensor" --> r2["Refused: a state goal on varying sensing"]:::refused
        state -- "graph model" --> uncoupled["LQR that ignores the graph's couplings"]:::gap
        obs -- "R(x) sensor" --> c1["Curious EFE: horizon read off the model"]:::curious
        obs -- "R(x) sensor,<br/>horizon set too short" --> r3["Refused: the target does not move the scores"]:::refused
        obs -- "fixed sensor" --> r4["Refused: same information everywhere"]:::refused
        obs -- "Q(x) only" --> rq["Refused, though Q(x) alone<br/>can make an agent curious"]:::gap
        obs -- "graph model,<br/>R(x) node sensors" --> c2["Curious EFE: graph selector, horizon 1"]:::curious
        obs -- "graph model,<br/>fixed sensors" --> flat["EFE: information term flat"]:::flat

        classDef curious fill:#26a69a33,stroke:#26a69a,stroke-width:2px
        classDef flat fill:#f9a82533,stroke:#f9a825,stroke-width:2px
        classDef lqr fill:#5c8fd633,stroke:#5c8fd6,stroke-width:2px
        classDef track fill:#90a4ae26,stroke:#90a4ae,stroke-width:2px
        classDef refused fill:#e5737314,stroke:#e57373,stroke-width:2px
        classDef gap fill:#e5737326,stroke:#e57373,stroke-width:2px,stroke-dasharray:5 4
    ```

    A selector of your own replaces the one the Agent would choose.

    | You pass | Fixed sensor | `R(x)` sensor |
    | --- | --- | --- |
    | `ObservationGoal` + `selector=EFESelector(…)` | EFE, the same information for every action | Curious EFE at the horizon given, which is not checked |
    | `ObservationGoal` + `selector=RecedingHorizonSelector(…)` or `OpenLoopSelector(…)` | Search, information flat | Curious search over varying action sequences |
    | `StateGoal` + `selector=LQRSelector(…)` | LQR with your controller | Refused |
    | `StateGoal` + an EFE selector, or `ObservationGoal` + `LQRSelector` | Refused | Refused |
    | Any selector with no goal | Ignored, and the agent tracks | Ignored, and the agent tracks |

    An `ObservationGoal` with no horizon set gets the shortest horizon at which the
    target moves the scores. `agent.selector.horizon` reads it back,
    `agent.selector.cost_per_cycle` gives what each `sample_action` costs, and printing
    the agent shows both. A longer horizon looks further ahead for more work per step,
    and can change what the agent does. To compare, build one agent per horizon and run
    each. The sensor below is sharpest at the goal, position 1, and the push reaches the
    reading a step after it changes the velocity, so the shortest horizon is 2.

    ```python
    import jax.numpy as jnp

    from cpomdp import Agent, Belief, CallableSensor, LinearGaussianModel, ObservationGoal


    def sharper_near_goal(x, params):
        return jnp.array([[0.01 + (x[0] - 1.0) ** 2]])


    model = LinearGaussianModel(
        [[1.0, 0.1], [0.0, 1.0]],
        control_matrix=[[0.0], [0.1]],
        observation_matrix=[[1.0, 0.0]],
        dynamics_noise=jnp.eye(2) * 1e-6,
        observation_model=CallableSensor([[1.0, 0.0]], sharper_near_goal, ()),
        prior=Belief(mean=[0.0, 0.0], cov=jnp.eye(2)),
    )

    for horizon in (None, 4, 8):
        goal = ObservationGoal([1.0], action_bounds=(-2.0, 2.0), horizon=horizon)
        agent = Agent(model, goal)
        state = jnp.zeros(2)
        for _ in range(40):
            agent.infer_states(model.observation_matrix @ state)
            action = agent.sample_action()
            state = model.dynamics_matrix @ state + model.control_matrix @ action
        print(agent.selector.horizon, agent.selector.cost_per_cycle, jnp.round(state, 2))
    # 2 42 [1.08 0.4 ]
    # 4 84 [1. 0.]
    # 8 168 [0.99 0.  ]
    ```

    At the shortest horizon the agent overshoots and is still moving after 40 steps. At
    4 it settles on the goal, for twice the work per step.

The `Agent` is eager. Its methods update the belief it holds, so `jax.jit`, `vmap` and
`grad` do not apply to them. A jitted `infer_states` returns the posterior it was traced
with on every call. To compile a loop, drive a backend directly under `jax.lax.scan`.

??? example "A backend driven under `jax.lax.scan`"
    ```python
    import jax
    import jax.numpy as jnp

    from cpomdp import Belief, KalmanBackend, LinearGaussianModel

    model = LinearGaussianModel(
        [[1.0, 0.1], [0.0, 1.0]],
        control_matrix=[[0.0], [0.1]],
        observation_matrix=[[1.0, 0.0]],
        dynamics_noise=jnp.eye(2) * 1e-6,
        observation_noise=[[1e-2]],
        prior=Belief(mean=[0.0, 0.0], cov=jnp.eye(2)),
    )
    backend = KalmanBackend(model)


    def step(belief, reading):
        observation, action = reading
        belief = backend.infer_states(observation, belief, action)
        return belief, belief.mean


    observations = jnp.ones((50, 1))
    actions = jnp.zeros((50, 1))
    final, means = jax.jit(lambda xs: jax.lax.scan(step, model.prior, xs))(
        (observations, actions)
    )
    ```

    A selector's `select(belief, preference)` is pure as well. Call it inside `step` to
    choose each action within the compiled loop.

::: cpomdp.Agent
