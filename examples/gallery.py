"""Shared machinery for the example gallery.

Every demo in `examples/` tells its own story. They were all reaching for the same
handful of tools to tell it: the same colour-blind-safe palette, the same headless
matplotlib setup, the same figure-to-GIF loop, the same covariance ellipse, the same
expected-free-energy sweep. That was copied from script to script, so a fix in one never
reached the others. It lives here instead.

Nothing in this module is part of cpomdp's public API. It is presentation and plumbing.
It sits beside the demos rather than inside the library so that matplotlib and Pillow
stay out of the package. `conftest.py` puts `examples/` on the path, so the demos import
it by bare name.

Raid this file. The drawing helpers take an axis and leave the composition to you. The
EFE helpers are thin wrappers over the library kernel rather than reimplementations of
it.

Matplotlib and Pillow are imported lazily. The imports sit inside the functions that
need them, so a demo whose `check()` never renders anything still runs without the
`examples` extra.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from cpomdp.efe import expected_free_energy
from cpomdp.observation import CallableSensor

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL.Image import Image


# --- palette ------------------------------------------------------------------------
# Accents come from the Okabe-Ito qualitative set, distinguishable under every common
# form of colour blindness. Demos import these rather than pasting hexes. A colour then
# means the same thing across the whole gallery.
GREEN = "#009E73"  # the organism itself, or a live epistemic term
ORANGE = "#E69F00"  # the belief mean
BLUE = "#0072B2"  # a cue, beacon, or observed leaf
VERMILLION = "#D55E00"  # the goal: food or reward
SKY = "#56B4E9"  # an uncertainty ellipse
PINK = "#CC79A7"  # a hidden node being inferred


@dataclass(frozen=True)
class Palette:
    """The neutrals a figure is built on. Accents are the module constants above."""

    bg: str  # page background
    ink: str  # near-black text and outlines
    grid: str  # hairlines, axes, walls
    panel: str  # panel fill, where a figure has panels
    faint: str  # secondary text


# The three looks the gallery uses. FIELD is the bacillus simulations, which read as a
# petri dish. PAPER is the figure demos, which read as a printed page. DIAGRAM is the
# graph demos: the field neutrals with a slightly darker grid, because they draw rules
# and boxes rather than a dish.
FIELD = Palette(
    bg="#FAFAFA", ink="#2B2B2B", grid="#E4E4E4", panel="#FFFFFF", faint="#9A9A9A"
)
PAPER = Palette(
    bg="#F6F6F3", ink="#22262B", grid="#E2E2DE", panel="#FFFFFF", faint="#8B9095"
)
DIAGRAM = Palette(
    bg="#FAFAFA", ink="#2B2B2B", grid="#D8D8D8", panel="#FFFFFF", faint="#9A9A9A"
)


# --- figure output ------------------------------------------------------------------
def use_headless_backend() -> None:
    """Select the Agg backend: render to a buffer, never open a window."""
    import matplotlib as mpl

    mpl.use("Agg")


def figure_frame(fig) -> Image:
    """Draw a figure and take its pixels as one RGB frame."""
    from PIL import Image as PilImage

    fig.canvas.draw()
    return PilImage.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")


def write_gif(
    frames: list[Image],
    out_path: Path,
    *,
    fps: int,
    hold_seconds: float = 0.0,
    quantize_colors: int | None = None,
) -> Path:
    """Save frames as one looping GIF.

    Args:
        frames: the rendered frames, in order.
        out_path: where to write. Parent directories are created.
        fps: playback rate. Slow rates keep a narrated demo readable.
        hold_seconds: repeat the final frame for this long, so a loop lands on the
            result rather than snapping straight back to the start.
        quantize_colors: if given, collapse every frame onto one shared indexed palette
            of this many colours, built from the final frame. A per-frame palette
            balloons the file and flickers between frames. One shared palette does not.
            Pick the final frame deliberately, since it has to carry every colour the
            animation uses.

    Returns:
        `out_path`, for chaining into a print.
    """
    from PIL import Image as PilImage

    if hold_seconds:
        frames = [*frames, *frames[-1:] * max(1, int(fps * hold_seconds))]
    if quantize_colors is not None:
        shared = frames[-1].quantize(
            colors=quantize_colors, method=PilImage.Quantize.MEDIANCUT
        )
        frames = [
            f.quantize(palette=shared, dither=PilImage.Dither.NONE) for f in frames
        ]

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


def save_figure(
    fig,
    out_path: Path,
    *,
    dpi: int | None = None,
    facecolor: str | None = None,
    tight: bool = False,
) -> Path:
    """Write a figure to a still image. Parent directories are created.

    Only the arguments actually given are forwarded, so an omitted `dpi` or `facecolor`
    still falls through to matplotlib's own defaults rather than being overridden
    with `None`.
    """
    options: dict[str, Any] = {}
    if dpi is not None:
        options["dpi"] = dpi
    if facecolor is not None:
        options["facecolor"] = facecolor
    if tight:
        options["bbox_inches"] = "tight"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, **options)
    return out_path


# --- easing -------------------------------------------------------------------------
def ease(t: float) -> float:
    """Smoothstep: eases both ends, so motion starts and stops gently."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation from `a` to `b`."""
    return a + (b - a) * t


# --- drawing ------------------------------------------------------------------------
def xy(point) -> tuple[float, float]:
    """A plane point as the plain pair matplotlib's patch constructors take.

    The demos carry positions as arrays. `Circle`, `Ellipse` and the rest declare a
    two-tuple. They accept an array at runtime, so this is for the readers of the code
    and for the type checkers, not for matplotlib.
    """
    return float(point[0]), float(point[1])


@dataclass(frozen=True)
class BacillusStyle:
    """How to draw one bacillus.

    The two simulations that draw one want slightly different sizes and stacking. The
    shape is shared. What varies between demos is visible in one place instead of by
    diffing two files.
    """

    body: str = GREEN  # capsule fill and flagellum
    ink: str = FIELD.ink  # outline and eyespots
    length: float = 0.62
    width: float = 0.30
    edge_width: float = 1.6
    flagellum_points: int = 24
    flagellum_amplitude: float = 0.16
    flagellum_width: float = 1.4
    eye_offset: float = 0.07
    eye_size: float = 2.4
    zorder: int = 6  # the body, with the flagellum one below and the eyespots one above


SMALL_BACILLUS = BacillusStyle(
    body=GREEN,
    ink=FIELD.ink,
    length=0.52,
    width=0.26,
    edge_width=1.4,
    flagellum_points=22,
    flagellum_amplitude=0.14,
    flagellum_width=1.2,
    eye_offset=0.06,
    eye_size=2.0,
    zorder=7,  # one layer above the default, to ride over a shaded precision field
)


def draw_bacillus(ax, pos, heading, phase: float, style: BacillusStyle) -> None:
    """A capsule body with a wiggling flagellum, oriented along `heading`.

    Args:
        ax: the axis to draw on.
        pos: 2-vector, the body centre in data coordinates.
        heading: 2-vector; only its direction is used, and a zero heading points right.
        phase: advances the flagellum's sine, so the organism swims across frames.
        style: sizes, colours and stacking.
    """
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.transforms import Affine2D

    pos = np.asarray(pos, dtype=float)
    heading = np.asarray(heading, dtype=float)
    norm = float(np.hypot(*heading))
    heading = heading / norm if norm > 1e-6 else np.array([1.0, 0.0])
    angle = float(np.degrees(np.arctan2(heading[1], heading[0])))

    body = FancyBboxPatch(
        (-style.length / 2, -style.width / 2),
        style.length,
        style.width,
        boxstyle="round,pad=0,rounding_size=" + str(style.width / 2),
        linewidth=style.edge_width,
        edgecolor=style.ink,
        facecolor=style.body,
        joinstyle="round",
        zorder=style.zorder,
    )
    body.set_transform(
        Affine2D().rotate_deg(angle).translate(pos[0], pos[1]) + ax.transData
    )
    ax.add_patch(body)

    # Flagellum: a damped sine trailing from the rear. `phase` advances the swim.
    t = np.linspace(0, 1, style.flagellum_points)
    trail_x = -style.length / 2 - t * style.length * 1.5
    trail_y = style.flagellum_amplitude * np.sin(2.5 * np.pi * t + phase) * t
    radians = np.radians(angle)
    rotation = np.array(
        [
            [np.cos(radians), -np.sin(radians)],
            [np.sin(radians), np.cos(radians)],
        ]
    )
    world = rotation @ np.vstack([trail_x, trail_y]) + pos[:, None]
    ax.plot(
        world[0],
        world[1],
        color=style.body,
        lw=style.flagellum_width,
        alpha=0.85,
        zorder=style.zorder - 1,
    )

    # Eyespots, so the front end reads as the front end.
    for offset in (-style.eye_offset, style.eye_offset):
        eye_x, eye_y = rotation @ np.array([style.length * 0.22, offset]) + pos[:2]
        ax.plot(
            eye_x,
            eye_y,
            "o",
            color=style.ink,
            ms=style.eye_size,
            zorder=style.zorder + 1,
        )


def covariance_ellipse(
    cov, *, sigmas: float = 2.0, max_diameter: float | None = None
) -> tuple[float, float, float]:
    """The `(width, height, angle_degrees)` of a covariance ellipse.

    The geometry only. Demos draw the patch themselves, because how an uncertainty
    ellipse should be shaded is a per-figure decision and forcing one styling on all of
    them would be a worse abstraction than the shared eigendecomposition is a good one.

    `max_diameter` caps the *returned* size, never the underlying belief. A prior that
    starts deliberately wide can have a 2-sigma diameter larger than the whole plot, and
    drawing that literally floods the panel and makes early frames look broken. Capping
    keeps the frame readable. It still reads as uncertain.
    """
    values, vectors = np.linalg.eigh(cov)
    values = np.clip(values, 1e-9, None)
    angle = float(np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0])))
    width, height = 2 * sigmas * np.sqrt(values)
    if max_diameter is not None:
        width, height = min(width, max_diameter), min(height, max_diameter)
    return float(width), float(height), angle


def draw_covariance_ellipse(
    ax,
    mean,
    cov,
    color: str,
    *,
    sigmas: float = 2.0,
    alpha_fill: float = 0.16,
    max_diameter: float | None = None,
    zorder: int = 3,
) -> None:
    """A filled patch under a solid outline, both at `sigmas` standard deviations."""
    from matplotlib.patches import Ellipse

    width, height, angle = covariance_ellipse(
        cov, sigmas=sigmas, max_diameter=max_diameter
    )
    shared = {"angle": angle, "zorder": zorder}
    ax.add_patch(
        Ellipse(mean, width, height, facecolor=color, alpha=alpha_fill, **shared)
    )
    ax.add_patch(
        Ellipse(
            mean, width, height, facecolor="none", edgecolor=color, lw=1.2, **shared
        )
    )


# The EFE decomposition's own colours. Distinct from the Okabe-Ito accents on purpose:
# these label terms of an objective. The accents label objects in a world.
PRAGMATIC_C = "#e8833a"
EPISTEMIC_C = "#2ca6a4"
G_C = "#5b3a8a"


def draw_efe_panel(
    ax,
    actions,
    pragmatic,
    epistemic,
    g,
    *,
    title: str,
    note: str,
    xlabel: str,
) -> None:
    """The three EFE terms over a one-dimensional action sweep.

    Marks the pragmatic argmin and the `G` argmin separately. The panel turns on
    whether they coincide.
    """
    actions = np.asarray(actions)
    ax.plot(
        actions, pragmatic, color=PRAGMATIC_C, lw=2.2, label="pragmatic (goal cost)"
    )
    ax.plot(
        actions, epistemic, color=EPISTEMIC_C, lw=2.2, label="epistemic (info gain)"
    )
    ax.plot(actions, g, color=G_C, lw=3.0, label="G = pragmatic − epistemic")

    best_g = actions[int(np.argmin(g))]
    best_pragmatic = actions[int(np.argmin(pragmatic))]
    ax.axvline(best_pragmatic, color=PRAGMATIC_C, ls=":", lw=1.6)
    ax.axvline(best_g, color=G_C, ls="--", lw=1.6)
    ax.scatter([best_g], [g.min()], color=G_C, zorder=5, s=45)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.annotate(
        note,
        xy=(0.5, -0.30),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#333333",
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)


# --- the precision well the plane demos sense through --------------------------------
def beacon_noise(x, params):
    """`R(x)` for a precision well: an isotropic 2x2 noise that floors at the beacon.

    Module-level so it can ride in a `CallableSensor`'s static aux, with every tunable
    in `params` (`bx`, `by`, `width`, `r_lo`, `r_hi`).
    `R = (r_lo + (r_hi − r_lo)·(1 − exp(−d²/2w²)))·I`, a smooth flat-bottomed well:
    `r_lo` on the beacon, saturating to `r_hi` far away.

    The flat bottom is the design. Its spatial gradient vanishes *at* the beacon, so a
    localised agent feels no trapping pull there and can leave for the goal once it has
    nothing left to learn. A cone, whose gradient blows up at the floor, traps every
    agent that reaches it, which is the myopic local-minimum problem.
    """
    d2 = (x[0] - params["bx"]) ** 2 + (x[1] - params["by"]) ** 2
    falloff = 1.0 - jnp.exp(-d2 / (2.0 * params["width"] ** 2))
    r = params["r_lo"] + (params["r_hi"] - params["r_lo"]) * falloff
    return r * jnp.eye(2)


def precision_field(
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    model,
    *,
    channel: int,
    res: int = 130,
):
    """Sensor sharpness `−ln R[channel, channel]` over the arena, as `(xs, ys, field)`.

    Drawn as a few discrete contour bands, a faint signal-strength map showing where
    the world is legible: bright at the beacon, dark in the murk. The sample comes from
    the model's own `noise_fn`, so this is the `R(x)` the agents filter under rather
    than a redrawing of it.

    `channel` is required rather than defaulted because a block-diagonal `R` carries
    more than one well and only one of them is the information channel. A model whose
    first rows are fixed proprioception would plot a flat field under a default of 0,
    and the figure would still render.

    The sweep varies the first two state entries, the arena position both plane demos
    put there. Every other block is held at the model's prior mean. A `noise_fn` that
    keyed on one of those blocks would draw a field that is not a function of position,
    which is not a thing a contour map can show.

    Args:
        xlim: the arena's x extent, as `(low, high)`.
        ylim: its y extent.
        model: the model whose `observation` is the `CallableSensor` to sample.
        channel: which row of `R(x)` to read the sharpness off.
        res: how many samples per axis.

    Returns:
        The x samples, the y samples, and the `res × res` field, indexed `[y, x]`.
    """
    sensor = model.observation_model
    if not isinstance(sensor, CallableSensor):
        raise TypeError(
            "precision_field draws a state-dependent R(x). This model's observation is "
            f"{type(sensor).__name__}, which has no field to draw"
        )
    xs = np.linspace(*xlim, res)
    ys = np.linspace(*ylim, res)
    grid_x, grid_y = np.meshgrid(xs, ys)  # both (res, res), indexed [y, x]
    states = np.tile(np.asarray(model.prior.mean, dtype=float), (res * res, 1))
    states[:, 0] = grid_x.ravel()
    states[:, 1] = grid_y.ravel()
    # One vmapped sweep rather than res² Python-level calls into jax.
    noise = jax.vmap(sensor.noise_fn, in_axes=(0, None))(
        jnp.asarray(states), sensor.noise_params
    )
    sharpness = -np.log(np.asarray(noise[:, channel, channel]))  # higher = sharper
    return xs, ys, sharpness.reshape(res, res)


# --- expected free energy over a candidate set ---------------------------------------
def action_grid(low: float, high: float, n: int) -> jnp.ndarray:
    """The front-loaded n² grid of two-dimensional action candidates, shape (n², 2)."""
    axis = jnp.linspace(low, high, n)
    grid_x, grid_y = jnp.meshgrid(axis, axis)
    return jnp.stack([grid_x.ravel(), grid_y.ravel()], axis=1)


@jax.jit
def efe_over_grid(model, belief, preference, candidates):
    """The EFE `G` of every candidate action, via the library kernel.

    One `vmap` of `expected_free_energy` across the grid. The `argmin` (the chosen
    action) happens outside. The explore/exploit balance is set entirely by
    `preference.precision` (Λ), so this is the real public path rather than a
    hand-weighted recombination of the pragmatic and epistemic parts.
    """
    return jax.vmap(lambda a: expected_free_energy(model, belief, a, preference).g)(
        candidates
    )


def efe_sweep(model, belief, goal, actions):
    """Pragmatic, epistemic and `G` over a one-dimensional action sweep.

    Eager rather than jitted: these sweeps are tens of actions wide and feed a figure,
    so the compile would cost more than it saves.

    Returns:
        Three arrays, `(pragmatic, epistemic, g)`, aligned with `actions`.
    """
    pragmatic, epistemic, g = [], [], []
    for action in actions:
        efe = expected_free_energy(model, belief, jnp.array([action]), goal)
        g.append(float(efe.g))
        pragmatic.append(float(efe.pragmatic))
        epistemic.append(float(efe.epistemic))
    return np.array(pragmatic), np.array(epistemic), np.array(g)


# --- check reporting ------------------------------------------------------------------
def check_two_route_agreement(
    headline: str,
    native,
    flattened,
    gap: float,
    tolerance: float,
    *,
    mean_symbol: str = "μ",
) -> None:
    """Print two inference routes' posteriors side by side, then assert they agree.

    The demos that flatten a graph by hand to check the native route all report the
    same shape: both posteriors, the largest difference, and a verdict against a
    declared tolerance. The tolerance is printed with the gap rather than assumed,
    so a passing line still shows what it passed against.

    It raises rather than printing `FAIL` and returning. A verdict that only prints
    exits zero on a disagreement, which reads as a gate and is not one.

    Raises:
        AssertionError: if the two routes differ by `tolerance` or more.
    """
    agrees = gap < tolerance
    print(headline)
    print(
        f"  CouplingGraph.infer_root: {mean_symbol}={float(native.mean[0]):.6f}  "
        f"var={float(native.cov[0, 0]):.6f}"
    )
    print(
        f"  flattened KalmanBackend : {mean_symbol}={float(flattened.mean[0]):.6f}  "
        f"var={float(flattened.cov[0, 0]):.6f}"
    )
    print(f"  max |difference|        : {gap:.2e}  ->  {'PASS' if agrees else 'FAIL'}")
    if not agrees:
        raise AssertionError(
            f"{headline} routes differ by {gap:.2e}, tolerance {tolerance:.0e}"
        )


def figure_main(
    render: Callable[[Path], Path],
    default_out: str,
    *,
    check: Callable[[], None],
) -> None:
    """The gallery script entry point: `--check` runs the gate, otherwise render.

    A bare path argument overrides the default output. Flags are filtered out first, so
    `--check` never gets mistaken for a filename.
    """
    if "--check" in sys.argv:
        check()
        return
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out_path = Path(args[0]) if args else Path(default_out)
    print(f"rendering -> {render(out_path)}")
