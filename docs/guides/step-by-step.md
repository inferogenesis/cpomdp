# cpomdp - A Step-by-Step Guide

By *Dan Elliott*

A micro-organism floating in a petri dish never sees the whole dish. It uses surface-level sensors to record a faint gradient, exercises something called [run and tumble](https://en.wikipedia.org/wiki/Run-and-tumble_motion), senses again, repeats. That loop can be represented in active inference. By the end you will have built it yourself: an agent that steers to a target it cannot see, in about fifteen lines of Python.

cpomdp fills a real gap in the experimental toolkit of the Active Inference Framework (AIF). The leading package of its kind is pymdp which only operates in **discrete** space. That's where cpomdp comes in. Under the framework of AIF, most biological systems would possess continuous generative models, meaning anyone who wanted to explore this gap in the literature would have to build a custom model every time in MATLAB or Julia. Being from a software engineering background there was no hope in hell I was going to pay for MATLAB and I'd barely heard of Julia -  a sentiment I suspect is shared by anyone whose university isn't quietly footing the licence bill. Although Python and I have our disagreements, a language like Rust (my preference) isn't suitable for experimental toolboxes.

This guide is aimed at complete beginners to the field of AIF and computational neuroscience. I myself have no formal background in these areas, but have found them extremely interesting frontiers. For an overview of Active Inference and a guide on deriving Variational Free Energy (for the linear-gaussian case with fixed parameters) check out the blog section of my portfolio website [www.dj-elliott.com/blog](https://www.dj-elliott.com/blog). There you will find part 1 and 2 of my active inference talk slides and the derivation I mentioned above if you fancy dipping your toes into the math.

## Setup

>Note: cpomdp requires Python 3.11+

```bash
pip install cpomdp
```

Inside some Python file or Jupyter Notebook:

```python
import jax.numpy as jnp
from cpomdp import Belief, LinearGaussianModel
```

## The World on a Line

The plain English explanation of active inference is:

>The agent holds a belief about the world; sensing nudges that belief toward what it sees (perceiving/inferring); acting nudges the world toward what the agent wants (acting); round and round.

The examples I give that show off cpomdp are in 2 dimensions but for the purposes of getting someone to a position they feel familiar with the toolbox, the math is much more intuitive and less convoluted in 1 dimension. So the scenario we're going to define and use is a bug sliding along a line. A line leading directly to food. To define its state in the world we need two numbers: its **position** - how far down the line it is; and **velocity** - how fast it's moving down the line.

First we define what it is like for something to experience being on that line. In our everyday lives this is our interpretation of physics. You know that if you are standing still, you will remain still, unless you act. Or if you're sliding on ice, you will remain sliding (forget friction, says the physics grad).

### Dynamics

We start by defining a time-step `dt` and turning our examples from the last paragraph into equations - these equations are Newtonian kinematics.

```text
new position = position + dt × velocity     (you move by your speed)
new velocity = velocity                      (you coast — no friction)
```

In matrix form this is:

```text
new position = 1·position + dt·velocity   →   row 1:  [1, dt]
new velocity = 0·position +  1·velocity   →   row 2:  [0,  1]
```

We call this `dynamics_matrix`. Written in Python it looks like this:

```python
dt = 0.1
dynamics_matrix = [[1, dt],
                   [0, 1]]
```

`dynamics_matrix` is literally Newtonian kinematics in matrix form. *Add these python blocks as we go*.

### Sensor

This can be a little convoluted but stay with me. We now need to define what of those properties (position & velocity) of the world the agent can *sense*. For example, you don't have some magical number in your head that tells you your velocity, you **infer** your velocity based on how your position changes. It's the same case with our bug on the line. Its sensors observe where it is, not how fast it's moving.

We can define this sensor experience with:

```text
observation = 1 x position + 0 x velocity
```

It's one reading this time, so it's a single row, looking across both state variables position and velocity.

```python
observation_matrix = [[1, 0]]
```

Notice the 0. The agent can never measure its velocity, it can only *infer* it based on its position over time.

### Noise

The last piece we need is the noise of the world. In this case we have two noises. Dynamic noise, and observation noise. Let's use our bacteria in a petri-dish example again.

- **Dynamic noise** can be thought of as the random pelting you would take from neighbouring particles in the jelly and vibrations in the dish. This makes your idea of where you are much harder to read.
- **Observation noise** is the uncertainty of what you are sensing with your tiny microbial sensor. The blurriness of what you "see" if you will.

Now represent these terms as scalar values. The `dynamics_noise` (the wobble) affects position and velocity. Say we give it the value of **1e-6**. Dynamics has two properties that get pelted so `dynamics_noise` is a 2x2 matrix.

```text
dynamics_noise = [[1e-6,   0  ],     # position wobble | no shared wobble
                  [  0,  1e-6 ]]      # no shared wobble | velocity wobble
```

In python this is written as:

```python
dynamics_noise = jnp.eye(2) * 1e-6
```

Broken down this is saying "multiply a 2x2 identity matrix by 1e-6" which comes out exactly as written above in full matrix form.

The sensor only gives one reading, so `observation_noise` is a 1x1 matrix.

```python
observation_noise   = [[1e-2]]            # one reading, one wobble
```

>Note: `observation_noise` looks the same written down as it does in Python, so there's no separate "in Python this is written as…" step.

### The Prior

So...everything up until now has been the agent's own interpretation of the world. What the agent thinks the physics of the world is, hence why there is noise.

Now we need to define what the agent thinks of itself **before it has made any observations**. It's *vanity* if you will (love using that phrase). For this the agent needs two things.

- **mean** - the agent's best guess at where it is.
- **covariance (cov for short)** - the agent's uncertainty about that guess.

I'm going to start by giving the python, then talking about it this time.

```python
prior = Belief(mean=[0,0],
               cov=jnp.eye(2))
```

- mean = [0, 0] - "I think I'm at position 0 (1st term), sitting still (velocity 0; second term)".
- cov = jnp.eye(2) - "I have uncertainty in both my position and my velocity". Written out, that identity matrix is:

```text
[[1, 0],
 [0, 1]]
```

Notice that the uncertainty here in `cov` (1) is much larger than our `dynamics_noise` value of 1e-6. That is intentional. The agent starts **vague**. The next chapter will discuss *perceiving*, whose job it is to shrink that uncertainty: every observation pulls the guess tighter.

### The whole model

Five pieces, one object. `LinearGaussianModel` takes each by name. Since you built every piece as a named variable, the assembly reads almost like a list of what you've made: `dynamics_matrix=dynamics_matrix`, `observation_matrix=observation_matrix`, and so on. This is the moment the world (how it moves, what's seen, how fuzzy it all is) and the agent's starting belief (the prior) fuse into a single thing you can hand to an agent and run. One piece is deliberately missing, a way to **act**. A thing that only perceives doesn't need it yet. We'll add it the moment we start steering.

The full code picture so far should look like this:

```python
import jax.numpy as jnp
from cpomdp import Belief, LinearGaussianModel

# --- the world's mechanics ---
dt = 0.1

dynamics_matrix = [[1, dt],
                   [0, 1]]   # how the state drifts on its own (Newtonian kinematics)

observation_matrix = [[1, 0]]     # what the agent senses: position only (the 0 hides velocity)

dynamics_noise = jnp.eye(2) * 1e-6   # the world's own wobble
observation_noise   = [[1e-2]]            # the sensor's wobble

# --- the agent's starting belief ---
prior = Belief(mean=[0, 0],
               cov=jnp.eye(2))

# --- snap the world and the belief into one object ---
model = LinearGaussianModel(
    dynamics_matrix=dynamics_matrix,
    observation_matrix=observation_matrix,
    dynamics_noise=dynamics_noise,
    observation_noise=observation_noise,
    prior=prior,
)
# the world is built — nothing has happened yet; that starts when we perceive and act
```

If you run this nothing will happen and that is expected. You have `built a world and a belief`, but the agent hasn't sensed anything or moved yet. That is the next two chapters.

<details markdown="1">
<summary>Why linear-Gaussian, and why first?</summary>

Honest answer: it's the *easy* one. But easy here is a feature, not a cop-out. Linear dynamics + Gaussian noise is the single case where the maths closes cleanly — the belief stays a tidy bell curve forever, and perceiving collapses to a few matrix multiplications (the Kalman filter) with **no approximation at all**. That buys three things: it's **exact** (you can actually prove the code is right against known answers), it's **cheap** (no iterating, no sampling — just matrices), and it's the **foundation** — curved dynamics and nastier noise are almost always handled by bending them *back* toward this one.

</details>

## Perceiving

Time to let the agent actually sense something. It's carrying a belief, a guess plus an uncertainty, and `infer_states` folds a single reading into it and hands back a sharper one. That's the whole verb.

Every call to `infer_states` does two things:

1) **Predict** - before looking, roll the belief forward through the physics (that we built last time). In English this is: "given what I believed and how the world drifts, where should I be now?". This step increases uncertainty via dynamic noise. *Think of taking a step whilst wearing a blindfold*.

2) **Update** - Now look (sense). Compare what you're sensing to that prediction you hold and nudge the belief toward it. Uncertainty shrinks now, you're learning something.

How *hard* we update is exactly what we set up in the sensor. If it's fuzzy -> we can't trust it too well -> budge our belief slightly. If it's a sharp sensor -> trust heavily -> lean into the update. The trust ratio is called the **Kalman gain**.

Our `model` is already a pure observer — we never gave it a way to act — so we can hand it straight to an `Agent`. First, add `Agent` to your imports:

```python
import jax.numpy as jnp
from cpomdp import Agent, Belief, LinearGaussianModel   # <- Agent is new
```

Then hand it to an `Agent` with **no goal** — a pure observer just watches, it doesn't steer — and push a few readings through it:

```python
agent = Agent(model)
for y in [0.1, 0.2, 0.3, 0.4, 0.5]:
    obs = [y]
    agent.infer_states(obs)
    print("saw", y, "->", jnp.round(agent.belief.mean, 2))
```

Run this and you'll see the belief's best guess after each reading — `[position, velocity]`:

```text
saw 0.1 -> [0.1  0.01]
saw 0.2 -> [0.17 0.34]
saw 0.3 -> [0.27 0.67]
saw 0.4 -> [0.38 0.84]
saw 0.5 -> [0.48 0.91]
```

Few things to point out here:

- **Position pins down fast** — the first number locks onto the reading almost immediately.
- **Velocity is never measured, yet it's inferred** — the second number climbs `0.01 → 0.91`, homing in on the true `1.0` it was never told, worked out purely from how its position changes over time (like clocking a runner between two markers).
- **It doesn't just echo you** — reading `0.2` lands the position at `0.17`, not `0.2`. That's the Kalman gain from a moment ago: the agent blends your reading with its own prediction, trusting neither blindly. A number that *doesn't* snap to your input is proof the filter is actually filtering.

<details markdown="1">
<summary>Want the uncertainty too, as a tidy table? (click to expand)</summary>

It's the same thing with extra formatting, don't panic.

Swap the print loop for this — it also shows each variable's *spread* (`var`), which you'll watch collapse as the agent grows confident:

```python
print(f"{'reading':>9}   {'pos':>5} {'vel':>5}    {'var(pos)':>8} {'var(vel)':>8}")
b = agent.belief
print(f"{'(start)':>9}   {b.mean[0]:5.2f} {b.mean[1]:5.2f}    {b.cov[0,0]:8.2f} {b.cov[1,1]:8.2f}")
for y in [0.1, 0.2, 0.3, 0.4, 0.5]:
    obs = [y]
    agent.infer_states(obs)
    b = agent.belief
    label = f"see {y}"
    print(f"{label:>9}   {b.mean[0]:5.2f} {b.mean[1]:5.2f}    {b.cov[0,0]:8.2f} {b.cov[1,1]:8.2f}")
```

```text
  reading     pos   vel    var(pos) var(vel)
  (start)    0.00  0.00        1.00     1.00
  see 0.1    0.10  0.01        0.01     0.99
  see 0.2    0.17  0.34        0.01     0.66
  see 0.3    0.27  0.67        0.01     0.33
  see 0.4    0.38  0.84        0.01     0.16
  see 0.5    0.48  0.91        0.01     0.09
```

The `var` columns falling from `1.00` toward `0.01` / `0.09` is the uncertainty shrinking — the agent going from "I could be anywhere" to "I know where I am."

</details>

## Acting

Our bug can see now, but it's going nowhere. To actually chase the food it needs two things it hasn't got yet: a way to **move**, and somewhere to move **to**.

### The lever: control

This is the piece we held back. Picture yourself as that bacterium again: wiggle your little flagellum and you propel yourself forward.

```text
velocity = last-velocity + dt × wiggle
```

The wiggle never touches position directly — it only changes your **velocity**, which then carries position along (the dynamics do that part). Same coefficient trick as the others, and because the wiggle drives velocity, not position, the top row is `0`:

```python
control_matrix = [[0],
                  [dt]]
```

Now re-run the model definition with `control_matrix` added — the one new line that turns our observer into something that can move:

```python
model = LinearGaussianModel(
    dynamics_matrix=dynamics_matrix,
    control_matrix=control_matrix,   # <-- the new piece
    observation_matrix=observation_matrix,
    dynamics_noise=dynamics_noise,
    observation_noise=observation_noise,
    prior=prior,
)
```

### A destination: the goal

A lever is useless without somewhere to aim it. We tell the agent where the food sits with a `StateGoal` — a target state. Ours: get to position `1`, and settle there (velocity `0`).

```python
from cpomdp import Agent, Belief, LinearGaussianModel, StateGoal   # StateGoal is new

agent = Agent(model, StateGoal([1.0, 0.0]))   # reach position 1, come to rest
```

### Choosing a wiggle: `sample_action`

Now the agent can answer the question it simply couldn't before: *given where I think I am and where I want to be, which way should I wiggle?* That's `sample_action`:

```python
action = agent.sample_action()
print(jnp.round(action, 2))     # [0.92] — an array, one number per lever
```

From its starting belief (position `0`) with the food at position `1`, it picks a firm forward wiggle, about **+0.92**. And here's the satisfying part: as it closes in it eases off — the wiggles get gentler the nearer it gets — so it settles *onto* the food instead of barrelling past it.

One wiggle isn't a journey, though. In the next chapter we finally let **perceiving and acting run together**, round and round, and watch the bug actually arrive.

## The whole loop

Perceiving and acting have only happened *once* each so far. Real behaviour is the two of them **together, on repeat**: see a little, update the belief, wiggle the flagellum, let the world move, see again. That loop *is* the agent.

To run it we play two parts — the **world** (moving the real bug) and the **agent** (perceiving and acting). The agent never touches the real position; all it ever gets is a reading.

```python
real = jnp.array([0.0, 0.0])      # the bug's REAL position & velocity — the agent never sees this directly

for _ in range(100):
    obs    = model.observation_matrix @ real                    # the world shows the agent a reading
                                                          # '@' is matrix multiplication in python notation
    agent.infer_states(obs)                               # PERCEIVE: fold it in, sharpen the belief
    action = agent.sample_action()                        # ACT: how hard to wiggle the flagellum toward the food
    real   = model.dynamics_matrix @ real + model.control_matrix @ action   # the world moves on

print(agent.belief.mean)     # ≈ [1, 0] — it arrived
```

Peek at the belief every so often and you'll watch it close the gap:

```text
step   0:  true_pos=0.000   belief=[0.00, 0.00]
step   4:  true_pos=0.077   belief=[0.05, 0.28]
step  19:  true_pos=0.631   belief=[0.60, 0.31]
step  49:  true_pos=0.996   belief=[0.99, 0.02]
step  99:  true_pos=1.000   belief=[1.00, 0.00]
```

It arrived. The bug sits on the food at position `1`, at rest — and it got there steering by a **velocity it was never once shown**, working its speed out from how its position changed and wiggling its flagellum accordingly. That's the promise from the top of the page, delivered.

### The whole thing

Everything, start to finish — the ~15 lines we promised:

```python
import jax.numpy as jnp
from cpomdp import Agent, Belief, LinearGaussianModel, StateGoal

dt = 0.1
model = LinearGaussianModel(
    dynamics_matrix=[[1, dt], [0, 1]],
    control_matrix=[[0], [dt]],
    observation_matrix=[[1, 0]],
    dynamics_noise=jnp.eye(2) * 1e-6,
    observation_noise=[[1e-2]],
    prior=Belief(mean=[0, 0], cov=jnp.eye(2)),
)
agent = Agent(model, StateGoal([1.0, 0.0]))

true = jnp.array([0.0, 0.0])
for _ in range(100):
    obs = model.observation_matrix @ true     # what the agent gets to see
    agent.infer_states(obs)             # perceive
    action = agent.sample_action()      # act
    true = model.dynamics_matrix @ true + model.control_matrix @ action

print(agent.belief.mean)                # ≈ [1, 0]
```

### Where to go next

- **The maths, properly.** My [blog](https://www.dj-elliott.com/blog) derives the Variational Free Energy all of this rests on — start there if you want to see *why* the belief updates the way it does.
- **Two dimensions, and seeking information.** The [examples gallery](https://github.com/inferogenesis/cpomdp/blob/main/examples/README.md) takes this same loop into 2-D and adds the *epistemic* drive — an agent that detours to look before it leaps.
- **The rest of the library.** Everything `Agent`, `LinearGaussianModel` and friends can do is in the [API reference](../api/agent.md).
- Brows the docs at [cpomdp.inferogenesis.com](https://cpomdp.inferogenesis.com/).
- Contribute! I'd love to work with you, regardless of background or experience. Fill out issues, discussions. Whatever you need.
