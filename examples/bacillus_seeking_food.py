"""Render an animated GIF of four bacilli weighing *information* against *food*.

The continuous-state answer to pymdp's mouse-seeking-cheese demo — now with the
**epistemic** term that v0.3 adds. Four rod-shaped agents (the "bacilli") live in
the same 2-D plane with the same two landmarks:

- a **food** particle (the goal the agent prefers to observe), and
- a **beacon** — a region where the sensor is sharp (low ``R(x)``), so visiting it
  *collapses the agent's uncertainty*. Away from the beacon the world is murky and
  the agent can barely tell where it is.

Each agent senses only a noisy reading of its own position, so it must *infer*
where it is while *acting*. The four differ in **one number only** — the weight λ
on the epistemic (information-gain) term of the Expected Free Energy it minimises::

    G(a) = pragmatic(a)  −  λ · epistemic(a)
           └ goal cost ┘     └ info gain ┘

- **classic LQR** (top-left) — no epistemic term at all. It beelines straight to
  the food; it never exploits the beacon, so it only ever localises *slowly*, by
  averaging the murk as it goes.
- **low λ** (top-right) — the beacon tugs at it, so it bulges off the straight
  line, but the food pull wins before it ever reaches the beacon.
- **right λ** (bottom-left) — the hero. Information is worth enough to *detour* to
  the beacon; the uncertainty ellipse collapses the moment it arrives there; then,
  with nothing left to learn, the food pull takes over and it heads on, **confident**.
- **λ too strong** (bottom-right) — over-curious. The beacon is worth so much it
  parks there and never leaves for the food.

What each visual element maps onto in the model:

- **bacillus body** — the true hidden state (position).
- **belief marker (+)** — the posterior mean ``belief.mean`` (where it *thinks* it
  is).
- **uncertainty ellipse** — the 2-σ contour of ``belief.cov``; it collapses fastest
  where the sensor is sharp (at the beacon).
- **beacon field** — faint contour bands of sensor sharpness (−ln R); brightest
  where ``R(x)`` is low, marking the legible region around the beacon.
- **food star** — the preferred observation (the goal).

The simulation is real: every agent shares one ``KalmanBackend`` filter over a
``CallableSensor`` whose ``R(x)`` dips at the beacon, and the three EFE agents pick
each action by minimising the library's own ``expected_free_energy`` kernel over a
grid of candidate moves, re-weighting its epistemic split by λ. The classic agent
uses the ``LQRController`` — which is what EFE collapses to when λ = 0 (ADR-003).

Run it (``RUN`` = ``uv run --with matplotlib --with pillow python``)::

    RUN examples/bacillus_seeking_food.py            # -> docs/assets/bacillus.gif
    RUN examples/bacillus_seeking_food.py out.gif    # custom path
    RUN examples/bacillus_seeking_food.py --scan     # λ-sweep tuning metrics, no GIF

Needs ``matplotlib`` and ``pillow`` on top of cpomdp; neither is a runtime
dependency of the library itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cpomdp.control import LQRController
from cpomdp.efe import expected_free_energy
from cpomdp.observation import CallableSensor
from cpomdp.selection import Preference
from cpomdp.types import Belief, LinearGaussianModel

# --- Okabe-Ito colourblind-safe palette --------------------------------------
BG = "#FAFAFA"
INK = "#2B2B2B"
GRID = "#E4E4E4"
BODY = "#009E73"  # bluish-green -- the bacillus body (the organism itself)
BELIEF = "#E69F00"  # orange       -- the belief mean (mu)
BEACON = "#0072B2"  # blue         -- the beacon / good-sensing region
FOOD = "#D55E00"  # vermillion   -- the food / goal

VERSION_TAG = "cpomdp v0.3"

# --- the four regimes: same world, one knob (the epistemic weight λ) ----------
# `kind` is "lqr" (no epistemic term) or "efe" (minimise pragmatic − λ·epistemic).
REGIMES = [
    {
        "key": "lqr",
        "kind": "lqr",
        "lam": 0.0,
        "title": "classic LQR  ·  no epistemic term",
        "note": "beelines to the food — never detours to localise",
        "accent": "#7A7A7A",
    },
    {
        "key": "low",
        "kind": "efe",
        "lam": 8.5,
        "title": "low λ  ·  information barely counts",
        "note": "tugged toward the beacon, but the food wins",
        "accent": "#56B4E9",
    },
    {
        "key": "right",
        "kind": "efe",
        "lam": 14.0,
        "title": "right λ  ·  detour, then dinner",
        "note": "detours to the beacon, localises, then to the food",
        "accent": "#009E73",
    },
    {
        "key": "strong",
        "kind": "efe",
        "lam": 50.0,
        "title": "λ too strong  ·  over-curious",
        "note": "so over-curious it parks at the beacon — never eats",
        "accent": "#CC79A7",
    },
]

# --- world geometry -----------------------------------------------------------
DT = 0.16
START = np.array([-3.4, -2.6])
FOOD_PT = np.array([3.4, -2.6])
BEACON_PT = np.array([0.0, 2.7])  # up and central — clearly off the start→food line

# sensor precision well R(x): a smooth, flat-bottomed Gaussian dip (see
# `beacon_noise`) — `r_lo` at the beacon, saturating to `r_hi` in the murk, over a
# `width`. Flat-bottomed (not a cone) so a localised agent isn't trapped at the floor.
R_LO = 0.02  # sensor noise floor at the beacon (sharp sensing)
R_HI = 1.30  # sensor noise far from the beacon (murky world)
R_WIDTH = 2.3  # width of the precision well — how far the beacon's pull reaches
GOAL_PRECISION = 0.22  # Λ scalar (obs-space): gentle enough that a detour can win
# near-zero process noise: at the beacon Σ collapses (slowly, ∝1/t), so the epistemic
# pull fades and a moderate-λ agent rolls on to the food. The fade is gradual, so
# *dwell time grows with λ*: a very large λ over-values the slowly-fading residual and
# is still dithering at the beacon at the end.
PROCESS_Q = 2e-5
PRIOR_COV = 2.6  # wide initial uncertainty → strong early epistemic drive (easy detour)

# action search (EFE) / clip (LQR): a per-step velocity command, bounded to a box.
ACTION_LO, ACTION_HI = -2.4, 2.4
GRID_N = 25  # candidates per axis → GRID_N² one-step moves scored each cycle

N_STEPS = 72


def beacon_noise(x, params):
    """R(x) for a 'precision well': an isotropic 2×2 noise that floors at the beacon.

    Module-level (so it can ride in ``CallableSensor``'s static aux); all tunables
    live in ``params``. ``R = (r_lo + (r_hi−r_lo)·(1−exp(−d²/2w²)))·I`` — a smooth,
    flat-bottomed well: ``r_lo`` at the beacon, saturating to ``r_hi`` far away. The
    flat bottom matters: its spatial gradient vanishes *at the beacon*, so a
    localised agent feels no trapping pull there and can leave for the food once it
    has nothing left to learn (the cone shape, with its blow-up gradient at the
    floor, traps every agent that reaches it — the myopic local-minimum problem).
    """
    d2 = (x[0] - params["bx"]) ** 2 + (x[1] - params["by"]) ** 2
    falloff = 1.0 - jnp.exp(-d2 / (2.0 * params["width"] ** 2))
    r = params["r_lo"] + (params["r_hi"] - params["r_lo"]) * falloff
    return r * jnp.eye(2)


def build_model():
    """A 2-D single-integrator swimmer with a state-dependent (beacon) sensor.

    State is position ``[x, y]``; an action is a bounded one-step velocity command
    (``μ⁺ = μ + dt·a``), so a *single* greedy EFE step can already move the observed
    position — the regime where the epistemic term is live (a double integrator
    would hide the action one step downstream and the one-step EFE would collapse).
    The sensor reads position with noise ``R(x)`` that dips at the beacon.
    """
    sensor = CallableSensor(
        sensor_model=[[1.0, 0.0], [0.0, 1.0]],
        noise_fn=beacon_noise,
        noise_params={
            "bx": float(BEACON_PT[0]),
            "by": float(BEACON_PT[1]),
            "r_lo": R_LO,
            "r_hi": R_HI,
            "width": R_WIDTH,
        },
    )
    return LinearGaussianModel(
        dynamics=[[1.0, 0.0], [0.0, 1.0]],
        control=[[DT, 0.0], [0.0, DT]],
        sensor_model=[[1.0, 0.0], [0.0, 1.0]],
        dynamics_noise=np.eye(2) * PROCESS_Q,
        sensor_noise=np.eye(2) * R_LO,  # nominal; the live R comes from `sensor`
        prior=Belief(mean=START, cov=np.eye(2) * PRIOR_COV),
        observation=sensor,
    )


def _candidate_grid():
    """The front-loaded GRID_N² grid of one-step action candidates, shape (k², 2)."""
    axis = jnp.linspace(ACTION_LO, ACTION_HI, GRID_N)
    ax, ay = jnp.meshgrid(axis, axis)
    return jnp.stack([ax.ravel(), ay.ravel()], axis=1)


@jax.jit
def _efe_split_grid(model, belief, preference, candidates):
    """Pragmatic and epistemic of every candidate action, via the library kernel.

    One ``vmap`` of ``expected_free_energy`` across the grid; the λ re-weighting and
    the ``argmin`` happen in plain NumPy outside, so a single scored grid serves
    every λ that shares this belief.
    """

    def one(a):
        _, parts = expected_free_energy(model, belief, a, preference)
        return parts["pragmatic"], parts["epistemic"]

    return jax.vmap(one)(candidates)


def simulate(regime, *, seed=7):
    """Run one agent's perceive → act loop, recording truth, belief mean, belief cov.

    Perception is identical across regimes (a per-step Kalman filter over the
    beacon sensor); only the *action* differs — LQR for the classic agent, a
    λ-weighted one-step EFE argmin for the others.
    """
    from cpomdp.backends.kalman import KalmanBackend

    rng = np.random.default_rng(seed)
    model = build_model()
    backend = KalmanBackend(model)
    preference = Preference(goal=FOOD_PT, precision=np.eye(2) * GOAL_PRECISION)
    candidates = _candidate_grid()
    candidates_np = np.asarray(candidates)

    a_mat = np.eye(2)
    b_mat = np.eye(2) * DT

    controller = None
    if regime["kind"] == "lqr":
        controller = LQRController(
            model,
            goal_precision=np.eye(2) * GOAL_PRECISION,
            effort_penalty=np.eye(2) * 0.6,
        )

    def choose(belief):
        if regime["kind"] == "lqr":
            act = np.asarray(controller.action(belief.mean, FOOD_PT))
            return np.clip(act, ACTION_LO, ACTION_HI)
        prag, epi = _efe_split_grid(model, belief, preference, candidates)
        cost = np.asarray(prag) - regime["lam"] * np.asarray(epi)
        return candidates_np[int(np.argmin(cost))]

    belief = model.prior
    true = START.astype(float).copy()
    last_action = jnp.zeros(2)

    true_states = [true.copy()]
    means = [np.asarray(belief.mean)]
    covs = [np.asarray(belief.cov)]

    for _ in range(N_STEPS):
        # sense the current true position with the local noise R(true)
        r_here = np.asarray(
            beacon_noise(jnp.asarray(true), model.observation.noise_params)
        )
        obs = true + np.linalg.cholesky(r_here) @ rng.standard_normal(2)
        belief = backend.infer_states(obs, belief, last_action)
        action = np.asarray(choose(belief), dtype=float)
        last_action = jnp.asarray(action)

        true_states.append(true.copy())
        means.append(np.asarray(belief.mean))
        covs.append(np.asarray(belief.cov))

        true = a_mat @ true + b_mat @ action  # advance the true plant

    return (
        np.array(true_states),
        np.array(means),
        np.array(covs),
    )


def _metrics(true_states):
    """Behaviour summary for tuning: closest beacon approach, dwell, final food dist."""
    d_to_beacon = np.linalg.norm(true_states - BEACON_PT, axis=1)
    d_beacon = d_to_beacon.min()
    step_min = int(d_to_beacon.argmin())
    d_food = np.linalg.norm(true_states[-1] - FOOD_PT)
    # dwell: how many steps it stayed within 0.6 of the beacon
    dwell = int((d_to_beacon < 0.6).sum())
    return d_beacon, step_min, dwell, d_food


def scan():
    """Print behaviour metrics over a λ sweep — the tuning harness (no rendering)."""
    print(f"world: start={START}  food={FOOD_PT}  beacon={BEACON_PT}")
    print(
        f"  R(x): r_lo={R_LO} r_hi={R_HI} w={R_WIDTH}  Λ={GOAL_PRECISION}  "
        f"Q={PROCESS_Q}  dt={DT}"
    )
    print(
        f"  ‖start−beacon‖={np.linalg.norm(START - BEACON_PT):.2f}  "
        f"‖start−food‖={np.linalg.norm(START - FOOD_PT):.2f}  "
        f"‖beacon−food‖={np.linalg.norm(BEACON_PT - FOOD_PT):.2f}"
    )
    print(f"  N_STEPS={N_STEPS}\n")
    print("  LQR (classic):")
    ts, _, covs = simulate({"key": "lqr", "kind": "lqr", "lam": 0.0})
    db, sm, dw, df = _metrics(ts)
    print(
        f"    minBeacon={db:5.2f}@{sm:2d}  dwell={dw:2d}  finalFood={df:5.2f}  "
        f"finalΣtr={np.trace(covs[-1]):.3f}"
    )
    print("\n  EFE λ sweep   (B=reached beacon, F=reached food):")
    for lam in [2, 4, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20, 25, 35, 50]:
        ts, _, covs = simulate({"key": "efe", "kind": "efe", "lam": float(lam)})
        db, sm, dw, df = _metrics(ts)
        flags = ("B" if db < 0.6 else " ") + ("F" if df < 0.5 else " ")
        print(
            f"    λ={lam:5.1f}  minBeacon={db:5.2f}@{sm:2d}  dwell={dw:2d}  "
            f"finalFood={df:5.2f}  finalΣtr={np.trace(covs[-1]):5.3f}  [{flags}]"
        )


def _draw_bacillus(ax, pos, heading, phase, *, length=0.52, width=0.26):
    """A capsule body with a wiggling flagellum, oriented along ``heading``."""
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.transforms import Affine2D

    n = float(np.hypot(*heading))
    heading = heading / n if n > 1e-6 else np.array([1.0, 0.0])
    angle = float(np.degrees(np.arctan2(heading[1], heading[0])))

    body = FancyBboxPatch(
        (-length / 2, -width / 2),
        length,
        width,
        boxstyle="round,pad=0,rounding_size=" + str(width / 2),
        linewidth=1.4,
        edgecolor=INK,
        facecolor=BODY,
        joinstyle="round",
        zorder=7,
    )
    body.set_transform(
        Affine2D().rotate_deg(angle).translate(pos[0], pos[1]) + ax.transData
    )
    ax.add_patch(body)

    # Flagellum: a damped sine trailing from the rear, swimming as `phase` runs.
    t = np.linspace(0, 1, 22)
    fx = -length / 2 - t * length * 1.5
    fy = 0.14 * np.sin(2.5 * np.pi * t + phase) * t
    rad = np.radians(angle)
    rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    world = rot @ np.vstack([fx, fy]) + pos[:, None]
    ax.plot(world[0], world[1], color=BODY, lw=1.2, alpha=0.85, zorder=6)

    # Eyespots so the front end reads as the front end.
    for off in (-0.06, 0.06):
        ex, ey = rot @ np.array([length * 0.22, off]) + pos[:2]
        ax.plot(ex, ey, "o", color=INK, ms=2.0, zorder=8)


def _precision_field(xlim, ylim, model, res=130):
    """Sensor *sharpness* (−ln R) sampled over the arena, as ``(xs, ys, field)``.

    Drawn as a few discrete contour bands — a faint 'signal strength' map showing,
    at a glance, *where the world is legible*: bright at the beacon, dark in the
    murk. Computed once from the same ``R(x)`` the agents actually filter under.
    """
    xs = np.linspace(*xlim, res)
    ys = np.linspace(*ylim, res)
    params = model.observation.noise_params
    field = np.empty((res, res))
    for j, yy in enumerate(ys):
        for i, xx in enumerate(xs):
            r = float(beacon_noise(jnp.array([xx, yy]), params)[0, 0])
            field[j, i] = -np.log(r)  # higher = sharper sensing
    return xs, ys, field


def render(regimes, runs, beacon, food, out_path, *, fps=20):
    """Draw the 2×2 grid, one panel per regime, and write the looping GIF."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from PIL import Image

    # shared arena limits across all panels
    allpts = np.concatenate(
        [runs[r["key"]][0] for r in regimes] + [beacon[None], food[None]]
    )
    pad = 1.1
    xlim = (allpts[:, 0].min() - pad, allpts[:, 0].max() + pad)
    ylim = (allpts[:, 1].min() - pad, allpts[:, 1].max() + pad)

    field_xs, field_ys, field = _precision_field(xlim, ylim, build_model())
    field_levels = np.linspace(field.min(), field.max(), 9)
    n_frames = len(runs[regimes[0]["key"]][0])

    frames = []
    for i in range(n_frames):
        fig, axes = plt.subplots(2, 2, figsize=(8.2, 8.5), dpi=100)
        fig.patch.set_facecolor(BG)

        for ax, reg in zip(axes.ravel(), regimes, strict=True):
            true_states, means, covs = runs[reg["key"]]
            accent = reg["accent"]
            ax.set_facecolor(BG)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(GRID)

            # sensor-sharpness field (where the world is legible), as faint bands
            ax.contourf(
                field_xs,
                field_ys,
                field,
                levels=field_levels,
                cmap="Blues",
                alpha=0.32,
                zorder=0,
            )

            # beacon + food landmarks
            ax.plot(
                beacon[0],
                beacon[1],
                "o",
                color=BEACON,
                ms=9,
                markeredgecolor="white",
                markeredgewidth=1.2,
                zorder=4,
            )
            ax.plot(
                food[0],
                food[1],
                "*",
                color=FOOD,
                ms=20,
                markeredgecolor=INK,
                markeredgewidth=0.7,
                zorder=4,
            )

            # true trajectory so far
            if i > 0:
                tr = true_states[: i + 1]
                ax.plot(tr[:, 0], tr[:, 1], color=accent, lw=1.8, alpha=0.6, zorder=2)

            # uncertainty ellipse (2σ of the positional covariance)
            cov = covs[i]
            vals, vecs = np.linalg.eigh(cov)
            vals = np.clip(vals, 1e-9, None)
            ell_angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            w, h = 2 * 2.0 * np.sqrt(vals)
            for fa, ea, lw in ((0.16, "none", 0), (0.0, accent, 1.2)):
                ax.add_patch(
                    Ellipse(
                        means[i],
                        w,
                        h,
                        angle=ell_angle,
                        facecolor=accent if fa else "none",
                        edgecolor=ea,
                        alpha=(fa or 0.6),
                        lw=lw,
                        zorder=3,
                    )
                )

            # belief mean (μ)
            ax.plot(
                means[i][0], means[i][1], "+", color=BELIEF, ms=10, mew=2.0, zorder=8
            )

            # the bacillus at the TRUE state, headed along its recent motion
            pos = true_states[i].astype(float)
            j = max(1, i)
            heading = true_states[j] - true_states[j - 1]
            _draw_bacillus(ax, pos, np.asarray(heading), phase=i * 0.9)

            ax.set_title(
                reg["title"], color=INK, fontsize=10.5, fontweight="bold", pad=5
            )
            ax.text(
                0.5,
                -0.045,
                reg["note"],
                transform=ax.transAxes,
                ha="center",
                va="top",
                color="#555555",
                fontsize=7.8,
            )
            label = "LQR" if reg["kind"] == "lqr" else f"λ = {reg['lam']:g}"
            ax.text(
                0.028,
                0.962,
                label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                color=accent,
                fontsize=10,
                fontweight="bold",
                family="monospace",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "fc": "white",
                    "ec": GRID,
                    "alpha": 0.9,
                },
            )

        fig.suptitle(
            "four bacilli, one knob — the epistemic weight λ  ·  "
            "continuous active inference",
            color=INK,
            fontsize=12.5,
            fontweight="bold",
            y=0.972,
        )
        # shared legend + footer along the bottom
        handles = [
            plt.Line2D(
                [],
                [],
                marker="o",
                color=BODY,
                ls="none",
                ms=8,
                mec=INK,
                label="bacillus (truth)",
            ),
            plt.Line2D(
                [],
                [],
                marker="+",
                color=BELIEF,
                ls="none",
                ms=10,
                mew=2.0,
                label="belief μ",
            ),
            plt.Line2D(
                [],
                [],
                marker="o",
                color="#9A9A9A",
                ls="none",
                ms=8,
                alpha=0.6,
                label="uncertainty Σ (2σ)",
            ),
            plt.Line2D(
                [],
                [],
                marker="o",
                color=BEACON,
                ls="none",
                ms=8,
                mec="white",
                label="beacon",
            ),
            plt.Line2D(
                [],
                [],
                marker="*",
                color=FOOD,
                ls="none",
                ms=12,
                mec=INK,
                label="food (goal)",
            ),
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=5,
            fontsize=8.5,
            framealpha=0.9,
            edgecolor=GRID,
            labelcolor=INK,
            columnspacing=1.4,
            handletextpad=0.4,
            bbox_to_anchor=(0.5, 0.022),
        )
        fig.text(
            0.98,
            0.008,
            f"step {i:>2d}/{n_frames - 1}",
            ha="right",
            va="bottom",
            color="#9A9A9A",
            fontsize=8,
            family="monospace",
        )
        fig.text(
            0.02,
            0.008,
            VERSION_TAG,
            ha="left",
            va="bottom",
            color="#9A9A9A",
            fontsize=8,
            family="monospace",
        )

        fig.subplots_adjust(
            left=0.035, right=0.965, top=0.905, bottom=0.10, wspace=0.10, hspace=0.30
        )
        fig.canvas.draw()
        frames.append(
            Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
        )
        plt.close(fig)

    hold = max(1, int(fps * 1.1))
    frames.extend(frames[-1:] * hold)

    # Quantise every frame to ONE shared ≤128-colour palette built from the final
    # frame (it carries the murk heatmap, all four trajectories, and every marker
    # colour). A smooth per-frame palette would balloon the GIF; a shared indexed
    # palette keeps it small and avoids inter-frame colour flicker.
    palette = frames[-1].quantize(colors=128, method=Image.MEDIANCUT)
    qframes = [f.quantize(palette=palette, dither=Image.NONE) for f in frames]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    qframes[0].save(
        out_path,
        save_all=True,
        append_images=qframes[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
        disposal=2,
    )
    return out_path


def main():
    """Entry point: ``--scan`` to tune, otherwise render the 2×2 GIF."""
    if "--scan" in sys.argv:
        scan()
        return
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/assets/bacillus.gif")
    runs = {r["key"]: simulate(r) for r in REGIMES}
    path = render(REGIMES, runs, BEACON_PT, FOOD_PT, out)
    for r in REGIMES:
        ts, _, _ = runs[r["key"]]
        db, _sm, dw, df = _metrics(ts)
        tag = "LQR" if r["kind"] == "lqr" else f"λ={r['lam']:g}"
        print(
            f"  {tag:>7}: min‖·−beacon‖={db:4.2f}  dwell={dw:2d}  "
            f"final‖·−food‖={df:4.2f}"
        )
    print(f"wrote {path}  ({len(runs[REGIMES[0]['key']][0])} steps × 4 panels)")


if __name__ == "__main__":
    main()
