"""Render an animated GIF of a bacillus seeking food via continuous active inference.

The continuous-state answer to pymdp's mouse-seeking-cheese demo. A single
rod-shaped agent (a "bacillus") lives in a 2-D continuous plane. Its true
position is the hidden state; it only ever sees a noisy reading of where it is,
so it has to *infer* its own location while *acting* to reach a stationary food
particle (the goal the generative model prefers).

What each visual element maps onto in the model:

- **bacillus body** -- the true hidden state (position), rendered as a capsule
  with a wiggling flagellum so it reads as a swimming microbe rather than a dot.
- **belief marker (+)** -- the posterior mean ``agent.belief.mean``, where the
  agent *thinks* it is.
- **uncertainty ellipse** -- the positional posterior covariance
  ``agent.belief.cov``, a 2-sigma contour that shrinks as the filter grows
  confident.
- **food particle** -- the goal / prior preference the LQR controller steers
  toward.

Run it::

    python examples/bacillus_seeking_food.py            # -> docs/assets/bacillus.gif
    python examples/bacillus_seeking_food.py out.gif    # custom path

Needs ``matplotlib`` and ``pillow`` on top of cpomdp (``pip install matplotlib
pillow``); neither is a runtime dependency of the library itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")  # headless: render to a buffer, never open a window

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyBboxPatch
from matplotlib.transforms import Affine2D
from PIL import Image

from cpomdp import Agent, Belief, LinearGaussianModel

# --- Okabe-Ito colourblind-safe palette --------------------------------------
BG = "#FAFAFA"
INK = "#2B2B2B"
GRID = "#E4E4E4"
AGENT = "#009E73"  # bluish-green -- the true bacillus
BELIEF = "#E69F00"  # orange       -- the belief mean (mu)
SIGMA = "#56B4E9"  # sky blue     -- the uncertainty ellipse (Sigma)
FOOD = "#D55E00"  # vermillion   -- the food / goal

VERSION_TAG = "cpomdp v0.2.0"


def build_model(dt: float) -> tuple[LinearGaussianModel, np.ndarray, np.ndarray]:
    """A 2-D point-swimmer: state ``[x, y, vx, vy]``, action pushes velocity.

    Velocity is lightly damped so that ``[fx, fy, 0, 0]`` is a genuine
    zero-action equilibrium (what the LQR controller needs in a goal). The agent
    senses position only -- never velocity -- so the filter has to recover the
    velocity from how the (noisy) position moves, exactly the quickstart story
    lifted into two dimensions.

    Returns the model plus the true start state and the prior belief mean, which
    are deliberately offset from each other so the belief is seen converging onto
    the truth.
    """
    damp = 0.92  # velocity decay per step -> v=0 is the only equilibrium
    dynamics = [
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, damp, 0],
        [0, 0, 0, damp],
    ]
    control = [
        [0, 0],
        [0, 0],
        [dt, 0],
        [0, dt],
    ]
    sensor_model = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ]

    true_start = np.array([-3.6, -2.2, 0.0, 0.0])
    # The belief starts off-target and *uncertain*. The offset is deliberately
    # *lateral* to the food direction: the agent first commits toward where it
    # wrongly believes it is, then the filter drags the belief onto the truth, so
    # the trajectory curves as perception corrects action. Velocity is unknown
    # and the positional covariance is wide.
    belief_mean = np.array([-4.3, -1.0, 0.0, 0.0])
    prior_cov = np.diag([2.0, 2.0, 0.8, 0.8])

    model = LinearGaussianModel(
        dynamics=dynamics,
        control=control,
        sensor_model=sensor_model,
        dynamics_noise=np.diag([1e-5, 1e-5, 1e-4, 1e-4]),
        # Moderately noisy position sensor: the belief takes several steps to
        # lock on, so the uncertainty ellipse is seen *shrinking*, not snapping.
        sensor_noise=np.diag([0.22, 0.22]),
        prior=Belief(mean=belief_mean, cov=prior_cov),
    )
    return model, true_start, belief_mean


def simulate(n_steps: int, dt: float, seed: int = 7):
    """Run the perceive -> act loop, recording everything needed to draw it.

    Returns parallel lists: true states, belief means, positional 2x2
    covariances, and the food/goal position.
    """
    rng = np.random.default_rng(seed)
    model, true_state, _ = build_model(dt)
    food = np.array([2.6, 1.8])
    goal = np.array([food[0], food[1], 0.0, 0.0])

    # Softer effort penalty than the identity -> a swimmer that commits to the
    # food rather than creeping, giving a trajectory with visible curvature.
    agent = Agent(model, goal=goal, effort_penalty=np.eye(2) * 3.0)

    # Frame 0 is the prior, before any observation: the wide opening ellipse.
    true_states = [true_state.copy()]
    means = [agent.belief.mean.copy()]
    covs = [agent.belief.cov[:2, :2].copy()]
    sensor_chol = np.linalg.cholesky(model.sensor_noise)

    for _ in range(n_steps):
        # The agent sees a noisy reading of its true position, then perceives.
        obs = model.sensor_model @ true_state + sensor_chol @ rng.standard_normal(2)
        agent.infer_states(obs)
        action = agent.sample_action()

        true_states.append(true_state.copy())
        means.append(agent.belief.mean.copy())
        covs.append(agent.belief.cov[:2, :2].copy())  # positional block only

        # Advance the true plant with the action the agent actually applied.
        true_state = model.dynamics @ true_state + model.control @ action

    return true_states, means, covs, food


def _draw_bacillus(ax, pos, heading, phase, *, length=0.62, width=0.30):
    """A capsule body with a wiggling flagellum, oriented along ``heading``."""
    angle = np.degrees(np.arctan2(heading[1], heading[0]))

    # Rounded-rectangle capsule, drawn centred at the origin then rotated/moved.
    body = FancyBboxPatch(
        (-length / 2, -width / 2),
        length,
        width,
        boxstyle="round,pad=0,rounding_size=" + str(width / 2),
        linewidth=1.6,
        edgecolor=INK,
        facecolor=AGENT,
        joinstyle="round",
        zorder=6,
    )
    body.set_transform(
        Affine2D().rotate_deg(angle).translate(pos[0], pos[1]) + ax.transData
    )
    ax.add_patch(body)

    # Flagellum: a damped sine trailing from the rear, swimming as ``phase`` runs.
    n = 24
    t = np.linspace(0, 1, n)
    fx = -length / 2 - t * length * 1.5
    fy = 0.16 * np.sin(2.5 * np.pi * t + phase) * t
    local = np.vstack([fx, fy])
    rot = np.array(
        [
            [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
            [np.sin(np.radians(angle)), np.cos(np.radians(angle))],
        ]
    )
    world = rot @ local + pos[:, None]
    ax.plot(world[0], world[1], color=AGENT, lw=1.4, alpha=0.85, zorder=5)

    # A pair of eyespots so the front end reads as the front end.
    for off in (-0.07, 0.07):
        eye_local = np.array([length * 0.22, off])
        ex, ey = rot @ eye_local + pos[:2]
        ax.plot(ex, ey, "o", color=INK, ms=2.4, zorder=7)


def render(true_states, means, covs, food, out_path: Path, dt: float, fps: int = 20):
    """Draw every frame and write the looping GIF."""
    xs = [s[0] for s in true_states] + [food[0]]
    ys = [s[1] for s in true_states] + [food[1]]
    pad = 1.4
    xlim = (min(xs) - pad, max(xs) + pad)
    ylim = (min(ys) - pad, max(ys) + pad)

    frames: list[Image.Image] = []
    n = len(true_states)

    for i in range(n):
        fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=100)
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.tick_params(colors="#B0B0B0", labelsize=7)

        true_pos = np.array(true_states[i][:2])
        mean_pos = np.array(means[i][:2])
        cov = covs[i]

        # --- food / goal -----------------------------------------------------
        ax.plot(
            food[0],
            food[1],
            "*",
            color=FOOD,
            ms=26,
            markeredgecolor=INK,
            markeredgewidth=0.8,
            zorder=4,
        )

        # --- true trajectory so far -----------------------------------------
        if i > 0:
            tr = np.array(true_states[: i + 1])
            ax.plot(tr[:, 0], tr[:, 1], color=AGENT, lw=1.3, alpha=0.35, zorder=2)

        # --- uncertainty ellipse (2-sigma of the positional covariance) ------
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 1e-9, None)
        ell_angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        w, h = 2 * 2.0 * np.sqrt(vals)  # 2-sigma diameters
        ax.add_patch(
            Ellipse(
                mean_pos,
                w,
                h,
                angle=ell_angle,
                facecolor=SIGMA,
                edgecolor=SIGMA,
                alpha=0.18,
                lw=1.2,
                zorder=3,
            )
        )
        ax.add_patch(
            Ellipse(
                mean_pos,
                w,
                h,
                angle=ell_angle,
                facecolor="none",
                edgecolor=SIGMA,
                alpha=0.55,
                lw=1.2,
                zorder=3,
            )
        )

        # --- belief mean (mu) ------------------------------------------------
        ax.plot(
            mean_pos[0],
            mean_pos[1],
            "+",
            color=BELIEF,
            ms=13,
            mew=2.4,
            zorder=8,
        )

        # --- the bacillus at the TRUE state ----------------------------------
        vel = np.array(true_states[i][2:])
        heading = food - true_pos if np.linalg.norm(vel) < 0.001 else vel
        _draw_bacillus(ax, true_pos, heading, phase=i * 0.9)

        # --- legend ----------------------------------------------------------
        handles = [
            plt.Line2D(
                [],
                [],
                marker="o",
                color=AGENT,
                ls="none",
                ms=9,
                mec=INK,
                label="agent  (true state)",
            ),
            plt.Line2D(
                [],
                [],
                marker="+",
                color=BELIEF,
                ls="none",
                ms=11,
                mew=2.4,
                label="belief  (μ)",
            ),
            plt.Line2D(
                [],
                [],
                marker="o",
                color=SIGMA,
                ls="none",
                ms=9,
                alpha=0.5,
                label="uncertainty  (Σ)",
            ),
            plt.Line2D(
                [],
                [],
                marker="*",
                color=FOOD,
                ls="none",
                ms=13,
                mec=INK,
                label="food  (goal)",
            ),
        ]
        ax.legend(
            handles=handles,
            loc="upper left",
            framealpha=0.9,
            edgecolor=GRID,
            fontsize=8.5,
            labelcolor=INK,
        )

        ax.set_title(
            "bacillus seeking food  ·  continuous active inference",
            color=INK,
            fontsize=11,
            pad=10,
        )
        # error between belief and truth, shrinking as the filter locks on
        err = np.linalg.norm(mean_pos - true_pos)
        ax.text(
            0.985,
            0.02,
            f"step {i:>2d}/{n - 1}   |μ−x| = {err:4.2f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color="#8A8A8A",
            fontsize=8,
            family="monospace",
        )
        ax.text(
            0.015,
            0.02,
            VERSION_TAG,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color="#8A8A8A",
            fontsize=8,
            family="monospace",
        )

        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(buf).convert("RGB"))
        plt.close(fig)

    # Hold the final frame a beat, then loop cleanly.
    hold = max(1, int(fps * 1.2))
    frames.extend(frames[-1:] * hold)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
        disposal=2,
    )
    return out_path


def main() -> None:
    """Main body for demo."""
    dt = 0.12
    n_steps = 60
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/assets/bacillus.gif")

    true_states, means, covs, food = simulate(n_steps, dt)
    path = render(true_states, means, covs, food, out, dt)

    final_err = np.linalg.norm(np.array(means[-1][:2]) - np.array(true_states[-1][:2]))
    reached = np.linalg.norm(np.array(true_states[-1][:2]) - food)
    print(f"wrote {path}  ({len(true_states)} steps)")
    print(f"  final belief error |μ−x| = {final_err:.3f}")
    print(f"  final distance to food   = {reached:.3f}")


if __name__ == "__main__":
    main()
