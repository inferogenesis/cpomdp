# cpomdp - A Step-by-Step Guide

### Wrote by - Dan C

A micro-organism floating in a petri dish never sees the whole dish. It uses surface-level sensors to record a faint gradient, exercises something called [run and tumble](https://en.wikipedia.org/wiki/Run-and-tumble_motion), senses again, repeats. That loop can be represented in active inference. By the end you will have built it yourself: an agent that steers to a target it cannot see, in about fifteen lines of Python.

Cpomdp fills the gap in experimental literature of Active Inference (AIF). The leading package of its kind is pymdp which only operates in **discrete** space. That's where cpomdp comes in. Under the framework of AIF, most biological systems would possess continuous generative models, meaning anyone who wanted to explore this gap in the literature would have to build a custom model every time in MATLAB or Julia. Being from a software engineering background there was no hope in hell I was going to use MATLAB and I'd barely heard of Julia - a sentiment I think is shared outside of the academic community. Although Python and I have our disagreements, a language like Rust (my preference) isn't suitable for experimental toolboxes.

This guide is aimed at complete beginners to the field of AIF and computational neuroscience. I myself have no formal background in these areas, but have found them extremely interesting frontiers. For an overview of Active Inference and a guide on deriving Variational Free Energy (for the linear-gaussian case with fixed parameters) check out the blog section of my portfolio website [www.dj-elliott.com/blog](https://www.dj-elliott.com/blog). There you will find part 1 and 2 of my active inference talk slides and the derivation I mentioned above if you fancy dipping your toes into the math.

## Setup

>Note: cpomdp requires Python 3.10+

```bash
pip install cpomdp
```

Inside some Python file or Jupyter Notebook:
```python
import jax.numpy as jnp
from cpomdp import Belief, LinearGaussianModel
```

>If you're using Jax already in a notebook, you will need to run this import cell first or restart the kernel so cpomdp's 64-bit switch takes effect.

## The 2D World

Conceptually, it's most intuitive (for me at least, I would imagine for most humans this is the case) if we keep to a physical world simplified to 2-Dimensions.

First we define what it is like for something to experience being in that petri-dish. In our everyday lives this is our interpretation of physics. You know that if you are standing still, you will remain still, unless you act. Or if you're sliding on ice, you will remain sliding (forget friction, says the physics grad).


### Dynamics

We start by defining a time-step `dt` and turning our examples form the last paragraph into equations - these equations are Newtonian kinematics.

```
new position = position + dt × velocity     (you move by your speed)
new velocity = velocity                      (you coast — no friction)
```

In matrix form this is:

```
new position = 1·position + dt·velocity   →   row 1:  [1, dt]
new velocity = 0·position +  1·velocity   →   row 2:  [0,  1]
```

We call this `dynamics`. Wrote in python it looks like this:

```python
dt = 0.1
dynamics = [[1, dt],
            [0, 1]]
```

`dynamics` is literally Newtonian kinematics in matrix form.

### Control

Next we need to define how we **act** in the world. Going back to the petri-dish, imagine you are that bacteria, wiggle your little flagellum and propel yourself forward.

```
velocity = last-velocity + dt x shove
```

Position isn't touched, so the wiggle of your flagellum can only effect velocity, which in turn effects position.
That "wiggle" of the flagellum is how the agent **acts** in the world, remember that for later. This means in matrix for our top row must be 0. In python we represent the matrix form as:

```python
control = [[0],
           [dt]]
```

### Sensor

This can be a little convoluted but stay with me. We now need to define what of those properties of the world the agent can _sense_. For example, you don't have some magical number in your head that tells you your velocity, you **infer** your velocity based on how your position changes. It's the same case with our agent in a petri-dish. It's sensors observe where it is, not how fast its moving.

We can define this sensor experience with:

```
observation = 1 x position + 0 x velocity
```

It's one reading this time, so it's a single row, looking across both state variables position and velocity.

```python
sensor_model = [[1, 0]]
```

Notice the 0. The agent can never measure its velocity, it can only _infer_ it based on its position over time.

### Noise

The last piece we need is the noise of the world. In this case we have two noises. Dynamic noise, and sensor noise. Let's use our bacteria in a petri-dish example again.

- **Dynamic noise** can be thought of as the random pelting you would take from neighbouring particles in the jelly and vibrations in the dish. This makes your idea of where you are much harder to read.
- **Sensor noise** is the uncertainty of what you are sensing with your tiny microbial sensor. The blurriness of what you "see" if you will.

Now represent these terms as scalar values. The dynamics_noise (the wobble) effects position and velocity. Say we give it the value of **1e-6**. Dynamics has two properties that get pelted so `dynamics_noise` is a 2x2 matrix.

```
dynamics_noise = [[1e-6,   0  ],     # position wobble | no shared wobble
                  [  0,  1e-6 ]]      # no shared wobble | velocity wobble
```

In python this is written as:

```python
dynamics_noise = jnp.eye(2) * 1e-6
```

Broken down this is saying "multiply a 2x2 identity matrix by 1e-6" which comes out exactly as wrote above in full matrix form.

The sensor only gives one reading, so `sensor_noise` is a 1x1 matrix.

```python
sensor_noise   = [[1e-2]]            # one reading, one wobble
```

>Note: Sensor noise looks the same written down as it does in python hence the no "In python this is written as...

### The Prior

So...everything up until now has been the agents own interpretation of the world. What the agent thinks the physics of the world is, hence why there is noise.

Now we need to define what the agent thinks of itself **before it has made any observations**. It's _vanity_ if you will (love using that phrase). For this the agent needs two things.

- **mean** - the agents best guess at where it is.
- **covariance (cov for short)** - the agents uncertainty about that guess.

I'm going to start by giving the python, then talking about it this time.

```python
prior = Belief(mean=[0,0],
               cov=jnp.eye(2))
```

- mean = [0, 0] - "I think I'm at position 0 (1st term), sitting still (velocity 0; second term)".
- cov = jnp.eye(2) = [[1, 0], - "I have uncertainty in my position and my velocity".
                       0, 1]

Notice that the uncertainty here in cov 1 is much larger than our dynamics_noice value of 1e-6. That is intential. The agent starts **vague**. The next chapter will discuss _perceiving_, whose job it is to shrink that uncertainty: every observation pulls the guess tighter.

### The whole model

Six pieces, one object. `LinearGaussianModel` takes each by name — and since you built every piece as a named variable, the assembly reads almost like a list of what you've made: dynamics=dynamics, control=control, and so on. This is the moment the world (how it moves, how it's pushed, what's seen, how fuzzy it all is) and the agent's starting belief (the prior) fuse into a single thing you can hand to an agent and run.

The full code picture should look like this:

```python
import jax.numpy as jnp
from cpomdp import Belief, LinearGaussianModel

# --- the world's mechanics ---
dt = 0.1

dynamics = [[1, dt],
            [0, 1]]          # how the state drifts on its own (Newtonian kinematics)

control = [[0],
           [dt]]            # how a push enters the state — into velocity, not position

sensor_model = [[1, 0]]     # what the agent senses: position only (the 0 hides velocity)

dynamics_noise = jnp.eye(2) * 1e-6   # the world's own wobble
sensor_noise   = [[1e-2]]            # the sensor's wobble

# --- the agent's starting belief ---
prior = Belief(mean=[0, 0],
               cov=jnp.eye(2))

# --- snap the world and the belief into one object ---
model = LinearGaussianModel(
    dynamics=dynamics,
    control=control,
    sensor_model=sensor_model,
    dynamics_noise=dynamics_noise,
    sensor_noise=sensor_noise,
    prior=prior,
)
# the world is built — nothing has happened yet; that starts when we perceive and act
```

If you run this nothing will happen and that is expected. You have `built a world and a belief`, but the agent hasn't sensed anything or moved yet. That is the next two chapters.

!!! note "Why linear-Gaussian, and why first?"
    Honest answer: it's the *easy* one. But easy here is a feature, not a
    cop-out. Linear dynamics + Gaussian noise is the single case where the
    maths closes cleanly — the belief stays a tidy bell curve forever, and
    perceiving collapses to a few matrix multiplications (the Kalman filter)
    with **no approximation at all**. That buys three things: it's **exact**
    (you can actually prove the code is right against known answers), it's
    **cheap** (no iterating, no sampling — just matrices), and it's the
    **foundation** — curved dynamics and nastier noise are almost always
    handled by bending them *back* toward this one. Every field has its
    hydrogen atom; this is ours.

## Perceiving

Time to let the agent actually sense something. It's carrying a belief, a guess plus an uncertainty, and `infer_states` folds a single reading into it and hands back a sharper one. That's the whole verb.

> One housekeeping note: a pure observer doesn't push on anything, so for this chapter we use the perceive-only model — same six pieces, minus the control matrix. We'll snap control back the moment it starts acting.

Every call to `infer_states` does two things:

1) **Predict** - before looking, roll the belief forward through the physics (that we built last time). In english this is: "given what I believed and how the world drifts, where should I be now?". This step increases uncertainty via dynamic noise. _Think of taking a step whilst wearing a blindfold_.

2) **Update** - Now look (sense). Compare what you're sensing to that prediction you hold and nudge the belief toward it. Uncertainty shrinks now, you learning something.

How _hard_ we update is exactly what we setup in the sensor. If it's fuzzy -> we can't trust it too well -> budge our belief slightly. If it's a sharp sensor -> trust heavily -> lean into the update. The trust ration is called the **Kalman gain**.
