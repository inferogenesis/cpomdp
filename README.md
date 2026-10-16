# cpomdp — Continuous Active Inference for Python

[![PyPI](https://img.shields.io/pypi/v/cpomdp.svg)](https://pypi.org/project/cpomdp/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21334562-blue.svg)](https://doi.org/10.5281/zenodo.21334562)
[![Python](https://img.shields.io/pypi/pyversions/cpomdp.svg)](https://pypi.org/project/cpomdp/)
[![CI](https://github.com/inferogenesis/cpomdp/actions/workflows/ci.yml/badge.svg)](https://github.com/inferogenesis/cpomdp/actions/workflows/ci.yml)
[![coverage](https://cpomdp.inferogenesis.com/assets/coverage.svg)](https://github.com/inferogenesis/cpomdp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/inferogenesis/cpomdp/blob/main/LICENSE)

Continuous active inference for Python. The continuous-state sibling of [pymdp](https://github.com/infer-actively/pymdp).

**cpomdp** is a JAX-native active inference toolbox for continuous state spaces. You give it a linear-Gaussian generative model and you get an agent that tracks hidden states and selects actions by minimising expected free energy.

Continuous means continuous: positions, velocities, concentrations, temperatures. State-dependent observation noise keeps the epistemic term alive, so information-seeking behaviour is available in a regime where it is [**provably**](#how-provable-is-your-result-experimental) absent. Inference runs as message passing over a Forney factor graph. The JAX backends, the EFE functions and the action selectors are pure over pytrees, so they compose under `jit`, `vmap`, `grad` and `lax.scan`. The `Agent` that drives them is an eager convenience, and the [backends page](https://cpomdp.inferogenesis.com/api/backends/#which-names-trace) lists which public names trace.

The loop is `infer_states` / `sample_action`, which pymdp users will recognise. cpomdp is its continuous-state sibling. The machinery underneath is its own.

Full documentation, including API reference and guides, lives at [cpomdp.inferogenesis.com](https://cpomdp.inferogenesis.com/).

## What cpomdp does

**Exact continuous perception under fixed noise.** Linear mean dynamics and fixed Gaussian noise make the filter an exact Kalman filter, closed form, nothing sampled. There is no variational gap. Under a state-dependent `R(x)` or `Q(x)` the filter reads the noise at the predicted mean. That is a first-order plug-in, and its posterior is approximate. The same code path runs whether you are tracking or acting.

**Action that reaches the posterior covariance.** cpomdp lets the observation noise depend on the state, `R(x)` (Corva 2026), or the process noise, `Q(x)`. The mean stays linear. Because the covariance now responds to what the agent does, the epistemic term of expected free energy differs across policies and information-seeking behaviour becomes available. Under a *fixed* linear-Gaussian sensor and fixed process noise that term is identical for every policy (Koudahl, Kouw & de Vries 2021). State-dependent noise is the smallest deviation from that model which escapes the result.

**One knob between exploiting and exploring.** `StateGoal` and `ObservationGoal` carry a precision Λ. The pragmatic term scales with Λ and the epistemic term does not, so Λ alone decides whether an agent beelines for the goal or detours to sharpen its belief first. The four bacilli above differ in that number and nothing else.

**Multi-step EFE, searched and certified.** `cpomdp.efe.policy_efe` scores an H-step horizon. `cpomdp.enumeration.EnumeratedEfeSearch` sweeps the full action space `A^H` and returns a `CompletenessCertificate`, so "this is the best plan" is decided over the declared action set rather than sampled from it, and the object that says so is checkable. Receding-horizon and open-loop selectors wrap the search. The search reports `cost_per_plan`, the `|A|^H · H` step evaluations one plan costs. Each selector reports `cost_per_cycle`, the cost of one agent step. That is a whole plan for the receding-horizon selector, which re-plans every step, and `|A|^H` for the open-loop one, which plans once every H steps. Past `2^21` policies the search refuses at construction and names `ChunkedEfeSearch`, which runs the same enumeration in blocks at a flat footprint and drives both selectors. Drop the epistemic term and steady-state LQR is what remains.

**Inference as message passing.** Beliefs propagate over a Forney factor graph through the `CouplingGraph` backend, with a temporal-edge partition controlling which off-diagonal precision blocks cross the time boundary. Where a model cannot be flattened, cpomdp raises `IncompatibleLinearizationError` rather than returning a quietly wrong number: state-dependent `R(x)` combined with a mean-shifting coupling makes flattening structurally impossible, and the two-pass linearisation is the supported route.

**What is not here.** The mean stays linear. Genuinely nonlinear sensors, a curved `g(x)` needing a second-order moment match, are the next step rather than a current feature.

Horizon length decides whether the epistemic machinery pays off. With `R(x)` alive a short-horizon agent still walks past the information. Stretch the horizon and it detours to collect it. Curiosity needs the term and the lookahead together.

## Example

Four bacilli seeking food in the same world. The continuous-state answer to pymdp's mouse-seeking-cheese, now with an **epistemic** term. The twist: the food's position is **hidden**, and a **beacon** marks where the agent can *see* it. Visiting the beacon doesn't sharpen where the agent thinks *it* is. It sharpens where it thinks the *food* is, which it can't act on directly. That makes the information genuinely **instrumental**: resolving it changes where the agent then heads.

Each body sits at its **true** hidden state. The blue `+` is where it believes it is, the diamond is where it believes the food is (both with their uncertainty ellipses), and the star is the food's true, hidden location.

The four differ in **one number only**, the **goal precision Λ** each is built with. They all minimise the same Expected Free Energy `G = pragmatic − epistemic`. The pragmatic (goal) term scales with Λ and the epistemic (information) term does not, so Λ alone tips the balance. **Classic LQR** and a **sharp Λ** beeline to the agent's current food guess and never detour. A **balanced Λ** detours to the beacon, learns where the food really is, *then* heads there with confidence. A **weak Λ** is so over-curious it parks at the beacon and never eats. One real knob, the precision you'd actually pass. Four behaviours.

![Four bacilli learning where the food is, under different goal precisions Λ, via continuous active inference](https://raw.githubusercontent.com/inferogenesis/cpomdp/348b9eb50417d43bc0e9d4e1520c1c2eb934d36b/docs/assets/bacillus_uncertain_food.gif)

Reproduce it with [`examples/bacillus_uncertain_food.py`](https://github.com/inferogenesis/cpomdp/blob/main/examples/bacillus_uncertain_food.py) (`pip install "cpomdp[examples]"`).

### How far ahead before information is worth a detour?

An agent on an open plane wants a goal it cannot locate. Its prior points the wrong way. A beacon well off that line is the only thing that can say where the goal really is. Walking there costs ground.

Run the same world once per planning horizon. At `H = 2` the agent walks straight to the spot it already believed in and settles there. It never checks. At `H = 14` it walks *away* from that spot, reads the beacon, finds out it was wrong, and then goes to the real goal. Only `H` changed.

![One agent run once per planning horizon: at short horizons it settles where its prior said, and past a crossing it detours to the beacon, learns where the goal really is, and goes there](https://raw.githubusercontent.com/inferogenesis/cpomdp/011d8b9d42536353f73e2cfbade6ce88d7dfc679/docs/assets/crossover_horizon.gif)

The margin between the two plans, `ΔG(H) = G(detour) − G(direct)` in nats, crosses zero exactly once. The epistemic pull is flat, because sensing once is worth what sensing once is worth. The pragmatic gradient decays under it. Freeze `R` at a constant and the sweep never crosses at any horizon, so the behaviour belongs to the state-dependent sensor and the horizon together.

Reproduce it with [`examples/crossover_horizon_figure.py`](https://github.com/inferogenesis/cpomdp/blob/main/examples/crossover_horizon_figure.py).

**Why the horizon is the question.** [State-dependent observation noise reintroduces epistemic value in linear-Gaussian active inference](https://arxiv.org/abs/2607.20306) (Corva 2026) establishes that `R(x)` makes the epistemic term non-constant, so a linear-Gaussian agent can be curious at all. Non-constant is not the same as decision-changing. The horizon at which curiosity starts changing which plan an agent picks is a separate question, and it has a measured answer.

**The answer is one-dimensional, and it is proven.** On the corridor cue task, `H* = 7`: the first horizon whose *open-loop* argmin is a two-phase sense-then-commit walk rather than a direct reach. Open-loop means whole action sequences are scored with no re-planning between steps. The same statistic under a receding-horizon driver is a different quantity and is unmeasured. That comes from enumerating every sequence in the declared action set `{0, ±1, ±2}` to depth `H`, so it decides the flip rather than sampling for it, and the search returns a `CompletenessCertificate` saying how many policies it was obliged to visit and how many it did. Zero the epistemic term and the crossing moves out to `H ≈ 10`, which is what makes the pull load-bearing rather than incidental. The numbers are registered in [`warrant_numbers.md`](https://github.com/inferogenesis/cpomdp/blob/main/warrant_numbers.md), the model is [`examples/ffg/crossover.py`](https://github.com/inferogenesis/cpomdp/blob/main/examples/ffg/crossover.py), and `tests/test_example_checks.py::test_crossover_check` asserts it on every merge and release. The `7` is an upper bound, because the declared set clips the reach at `−2` while reaching the goal from the start takes `−3`. On the wider `{−3…2}` the flip is at 6. Both wider sets are now certified: `{−3,…,2}` and `{−4,…,2}` each flip at 6, each under a completeness certificate, so extension saturates there. Refinement leaves `H* = 7` unmoved at two spacings, `0.5` and `0.25`, the two agreeing to the digit.

**The plane above is the readable version, not the proof.** It contrasts two named plans instead of searching, so its crossing is an exact statement about those two plans and says nothing about the argmin over all plans. Its `H = 7` and the corridor's `H* = 7` are different quantities on different models that happen to coincide.

More in the [examples gallery](https://github.com/inferogenesis/cpomdp/blob/main/examples/README.md), including the [FFG examples](https://github.com/inferogenesis/cpomdp/blob/main/examples/ffg/README.md), where a branch-coupled state-dependent sensor resolves a hidden context and can't be flattened to a Kalman filter.

## Install

```bash
pip install cpomdp
```

Or the latest from source:

```bash
pip install git+https://github.com/inferogenesis/cpomdp
```

Importing cpomdp switches JAX to 64-bit floats for the whole process (`jax_enable_x64`). The filter is checked against its oracles to 1e-9, which float32 cannot hold. Every other JAX array in the process then defaults to float64 too, at twice the memory. Turning the flag off again after the import makes beliefs and posteriors come back float32, with no warning.

That's all you need for normal use. There's also an optional RxInfer (Julia) backend that the test suite leans on as a correctness oracle. You almost certainly don't need it, but if you want it:

```bash
pip install "cpomdp[rxinfer]"
```

It pulls in a Julia bridge and bootstraps itself the first time you use it.

## Quickstart

Here's an agent steering a point mass to a target. It can push the mass and it can see where the mass is, but it never sees the velocity. The filter has to work that out from how the position moves.

```python
import jax.numpy as jnp
from cpomdp import Agent, Belief, LinearGaussianModel, StateGoal

# State is [position, velocity]. A push changes velocity, velocity carries
# position along, and we only ever observe position (through a noisy sensor).
dt = 0.1
model = LinearGaussianModel(
    dynamics_matrix=[[1, dt], [0, 1]],   # velocity carries position along
    control_matrix=[[0], [dt]],          # a push nudges velocity
    observation_matrix=[[1, 0]],               # we observe position only
    dynamics_noise=jnp.eye(2) * 1e-6,
    observation_noise=[[1e-2]],
    prior=Belief(mean=[0, 0], cov=jnp.eye(2)),
)

# Tell it where to go: sit still at position 1.
agent = Agent(model, StateGoal([1.0, 0.0]))

true_state = jnp.array([0.0, 0.0])
for _ in range(100):
    obs = model.observation_matrix @ true_state            # what the agent gets to see
    agent.infer_states(obs)                           # perceive
    action = agent.sample_action()                    # act
    true_state = model.dynamics_matrix @ true_state + model.control_matrix @ action

print(jnp.round(agent.belief.mean, 3))   # ≈ [1, 0]
```

Run that and the belief lands on `[1, 0]`. The agent worked out it was at position 1 and sitting still, which is exactly where we asked it to go, and it did it without ever seeing the velocity it had to control.

## The pymdp parallel

If you've used pymdp, the loop is the same. Four names carry over verbatim:

> `Agent` · `qs` · `infer_states` · `sample_action`

(`qs` is a read-only alias for `belief`, cpomdp's canonical name, so `agent.qs` and `agent.belief` are the same posterior. Use whichever your fingers reach for.)

The model's parts and the policy step are spelled differently:

| pymdp            | cpomdp                                                | what it is                                      |
| ---------------- | ----------------------------------------------------- | ----------------------------------------------- |
| `A`              | `observation_matrix`, `observation_noise`             | how a state produces a reading                  |
| `B`              | `dynamics_matrix`, `control_matrix`, `dynamics_noise` | how the state moves, and how an action moves it |
| `C`              | `StateGoal` / `ObservationGoal`                       | the goal you pursue, and how sharply            |
| `D`              | `model.prior`                                         | belief before you've seen anything              |
| `infer_policies` | folded into `sample_action`                           | scoring the actions on offer                    |

The model also answers to `.A`, `.B`, `.C`, `.Q` and `.R`. Those are control-theory letters. `model.A` is the dynamics matrix, pymdp's `B`, and `model.C` is the observation matrix, pymdp's `A`.

One honest difference in behaviour. `sample_action` here is deterministic, not a sample from a policy posterior. For a linear-Gaussian model with fixed noise the action that minimises expected free energy turns out to be exactly the LQR optimum ([Koudahl, Kouw and de Vries 2021](https://doi.org/10.3390/e23121565)), so there's a single best action and that's what comes back. Same loop, exact answer. The reasoning is in [DECISIONS.md](https://github.com/inferogenesis/cpomdp/blob/main/DECISIONS.md) (ADR-003) if you want it.

## Just want to track, not act?

A model with no `control_matrix` is a pure tracker. Drop the goal and `infer_states` still folds in observations and sharpens the belief, while `sample_action` stops you. There's nothing to steer toward, and nothing to steer with.

```python
tracker = LinearGaussianModel(        # no control matrix -> pure tracking
    dynamics_matrix=[[1, dt], [0, 1]],
    observation_matrix=[[1, 0]],
    dynamics_noise=jnp.eye(2) * 1e-6,
    observation_noise=[[1e-2]],
    prior=Belief(mean=[0, 0], cov=jnp.eye(2)),
)
agent = Agent(tracker)                # no objective
agent.infer_states([0.5])             # perceiving is fine
agent.sample_action()                 # ValueError: this Agent has no objective ...
```

## What's in the box

| you want to | reach for |
| ----------------------------------------- | ---------------------------------------------------------------------------- |
| perceive, exactly under fixed noise | `Agent.infer_states`, `KalmanBackend` |
| reach a target state | `StateGoal` with `LQRSelector` |
| act on expected free energy, where `R(x)` or `Q(x)` varies | `ObservationGoal` (the `Agent` builds an `EFESelector` and reads its horizon off the model), `expected_free_energy` |
| ...over an H-step horizon | `cpomdp.efe.policy_efe`, `policy_efe_trace` |
| ...searched exhaustively, with a certificate | `cpomdp.enumeration`, `EnumeratedEfeSearch`, `RecedingHorizonSelector`, `OpenLoopSelector` |
| sense more sharply in some places than others | `CallableSensor`, state-dependent `R(x)` |
| diffuse more in some states than others | `CallableDynamicsNoise`, state-dependent `Q(x)` |
| declare a branching model | `CouplingGraph`, `Coupling`, `CouplingGraphBackend` |
| ask whether a state-dependent sensor earns its keep | `probe_model` → `ProbeReport` |
| check a rollout stayed well conditioned | `cpomdp.diagnostics.rollout_conditioning` |
| perceive but never act | a model with no `control_matrix`, `Agent(model)` alone |
| re-plan every step, closed loop | `cpomdp.enumeration.RecedingHorizonSelector` (`replan_interval = 1`) |
| commit to one plan and execute it | `cpomdp.enumeration.OpenLoopSelector` (`replan_interval = H`) |
| swap the inference engine | the `InferenceBackend` protocol, or `cpomdp.backends.rxinfer.RxInferBackend` |

The state-dependence is in the *noise*. The mean stays linear. Genuinely **nonlinear sensors**, a curved `g(x)` needing a second-order moment match, are the next step and are not here yet.

## How provable is your result? (experimental)

Most toolboxes let you build an agent and stop there. cpomdp also labels how well each of your results is established, so you do not have to invent a warrant scheme of your own before you can report one honestly.

Every check the suite runs carries three labels. A **warrant** says what established the claim. `PROVED` covers a theorem, a symbolic identity, or a finite domain exhausted under a completeness certificate. `CERTIFIED` covers validated numerics over a compact domain. `CORROBORATED` covers a sample of a continuum, which settles existence and refutes a universal by counterexample and decides no universal at any sample count. A **tier** says how well the number itself is known: `EXACT`, `BOUNDED`, or `COMPUTED`. An **outcome** says what the falsifier did, and a falsifier does not pass. It fires or it does not. `PROVED` with nothing behind it does not construct.

Those labels reach a test run. cpomdp installs warrantlib, and warrantlib registers a pytest plugin that every pytest run in the environment loads. A test hands its findings over with the `record_check` fixture. A fired check fails, so does an unresolved one, and the two that never ran here skip with the progress letters keeping them apart: `v` for void by construction, `e` for measured elsewhere. Under `-v` a row reads `NOT TRIGGERED` where it would have read `PASSED`. The run closes with the accounting a reader needs first.

```text
70 registered, 70 tested here, none fired
   PROVED        NOT TRIGGERED   70
```

The plugin needs pytest 8 or newer. A project on an older pytest, or one that does not want the plugin, switches it off with `-p no:warrant`, on the command line or in `addopts`.

Registered and tested here are separate counts because they answer separate questions. Registering seventy falsifiers and testing sixty of them is a different claim from testing all seventy, and one number cannot carry both.

A count is still the weaker guard, though, and this repository used to lean on three of them. [research/registered_checks.toml](https://github.com/inferogenesis/cpomdp/blob/main/research/registered_checks.toml) declares every check each suite is registered to report. Each declared id becomes a pytest item because the manifest declares it, not because a suite reported it, so a check that stops reporting still has a row and the row fails naming it. A second item per suite catches the other direction. A renamed check therefore reports as one drop and one addition, where a count of them would have stayed at seventy and passed.

```bash
uv run pytest research/registered_checks.toml --warrant-detail
uv run python -m warrantlib.manifest --check research/registered_checks.toml
```

[research/warrant_ledger.md](https://github.com/inferogenesis/cpomdp/blob/main/research/warrant_ledger.md) is the canonical table, and every other document points at it. [warrant_numbers.md](https://github.com/inferogenesis/cpomdp/blob/main/warrant_numbers.md) records the declared numbers those claims are measured against. The vocabulary itself is [warrantlib](https://pypi.org/project/warrantlib/), published separately, with `Warrant` re-exported at cpomdp's top level, and documented at [cpomdp.inferogenesis.com/api/warrant](https://cpomdp.inferogenesis.com/api/warrant/).

Treat this as experimental. It may move out of cpomdp into a standalone inferogenesis tool before the 1.0 release, which is undecided.

## Swappable backends

You can swap the inference engine if you want to. `KalmanBackend` is the default and does the real work. `RxInferBackend` re-derives its answers on fixed-noise models through Julia, so the Kalman path has something independent to check itself against. It refuses a model with `R(x)` or `Q(x)`. Both sit behind the `InferenceBackend` protocol, so you can write your own.

## Status

Still pre-1.0. v0.5 renames the model's arguments to match its fields, `observation_matrix` and `observation_noise` among them, and makes every argument after `dynamics_matrix` keyword-only. Python 3.11 is the new floor. The [changelog](https://github.com/inferogenesis/cpomdp/blob/main/CHANGELOG.md) lists each break. v0.5 also adds the vocabulary a check uses to say what it is entitled to claim, and the anti-surge application. If you have a request or a suggestion that would make the front-facing API more usable, open a GitHub issue, I am happy to listen. Until 1.0, a minor version is where breaking changes can land.

The public API is every name in `cpomdp.__all__`, plus the names in the `__all__` of `cpomdp.efe`, `cpomdp.enumeration`, `cpomdp.diagnostics`, `cpomdp.control` and `cpomdp.backends.rxinfer`. Those are what the changelog's breaking-change rule covers. A path with a leading underscore on it, such as `cpomdp._lab`, is private and can change in any release. Modules like `cpomdp.types` stay importable, and the top-level name is still the one to use.

## Development

I designed and built cpomdp. The architecture, the conditionally-linear-Gaussian formulation, the API, and every decision in [DECISIONS.md](https://github.com/inferogenesis/cpomdp/blob/main/DECISIONS.md) are mine. The design draws on my day-to-day work as a full-time software engineer and on hands-on expertise integrating and developing large machine-learning models at scale using event-driven microservice architecture.

I do not classify myself as pro/against AI usage in code, but I do appreciate their usefulness when handled with rigour. Up to v0.4 AI coding assistants were used as tools under close review. They were Claude Opus 4.8, then Opus 5. They drafted docstrings and probed for edge cases and candidate bugs. They also expanded the test suite, adversarial tests included and wrote very little src.

Since August 2026 with the release of Opus 5.5 & Fable, I feel more relaxed about AI models writing src for me to review. Since then I have been playing around with optimal AI workflows whilst keeping the rigour built into cpomdp. These workflows will adapt and change as the models and climate shift. Typical workflow consists of AI staging changes in incremental review appropriate chunks for me to commit/change or reject. The implementing agent follows TDD red-greening, against an exact oracle where one exists. For optimisation  separate agent then asks whether there is a clearer design, presents it to me to review. Every pull request then clears the CI gate: lint, type checks, spelling and markdown, the strict docs build, the append-only check on the protected records, the fast tests on Python 3.11 to 3.14 under an 80% coverage floor, the RxInfer oracle, and the symbolic suites when a change touches what they depend on. The slow tests run on merge and release. Every pull request is checked via Claude-Review. I review and merge each pull request.

Correctness specifically rests on checks that do not rely on an agents assessment. The numbers are validated against two independent oracles, RxInfer in Julia and hand-written NumPy references in the tests. A result reported as proved has to name the commit that registered its bar and the commit that measured against it. A test checks git which came first. Where both are one commit, the report says history does not order them. The sources that fail the test today stay in as expected failures, with the reason recorded. By rule, a number reaches a research record only once the suite asserts it. The records of what was claimed before a result existed are append-only. CI refuses an in-place edit to a landed line unless a commit subject carries the typo marker, so taking that exception is itself on the record.

## Contributions

If you would like to contribute either your dev time or help steer the direction of the toolbox, please add a GitHub issue or discussion thread. I am monitoring this repository closely and would love to collaborate. [CONTRIBUTING.md](https://github.com/inferogenesis/cpomdp/blob/main/CONTRIBUTING.md) covers the setup, the hooks, and the bar a PR has to clear.

If you notice a better method in something I've already done or are just curious and want to chat I am more than happy to talk through my decision processes and improve on my work. I intend to blog my construction of cpomdp provided it doesn't interfere with developing it.

## Acknowledgements

Thanks to **Kevin Backhouse** (Postgraduate Researcher in Cognitive Neuroscience, Durham University) for guidance on the active-inference formulation, collaboration on related discrete generative-model projects, and for being a consistent sounding board throughout the design of this work.
