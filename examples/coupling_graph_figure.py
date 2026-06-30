"""Declare a branching model, or assemble its joint by hand — same answer, less work.

A linear-Gaussian tree's posterior is nothing a normal (Kalman) backend couldn't also
reach: flatten the tree into one joint over every variable and the numbers come out
identical. What `CouplingGraph` changes is not the answer but the work to get it. You
name the edges of the tree and call `infer`; the by-hand route assembles the joint
precision, inverts it, and marginalises the nuisance variables back out — then redoes
all of it whenever the wiring changes.

This figure sets the two routes side by side on the smallest tree that genuinely
branches. A hidden root ``r`` feeds a hidden hub ``h``, which fans out to two observed
leaves ``a`` and ``b``, so ``h`` has three neighbours — a degree no path (and so no
chain backend built on one) can hold. The hub is never measured; the root is inferred
only through it.

The left panel is the whole declaration: the tree, and the handful of lines that build
it and call ``infer``. The right panel is the 4x4 joint precision the by-hand route
assembles instead — in information form, so the tree shows up directly as the matrix's
sparsity (off the diagonal, only the three edges are non-zero). Both routes land on the
same belief over ``r``, asserted before the figure is ever drawn.

Needs the ``examples`` extra (matplotlib, not a runtime dependency of the library)::

    uv run --extra examples python examples/coupling_graph_figure.py

Pass ``--check`` to print the two routes' posteriors and their agreement, skipping the
render (no plotting deps on that path).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from cpomdp.backends.kalman import KalmanBackend
from cpomdp.ffg.factors.linear_gaussian import GaussianCoupling, GaussianObservation
from cpomdp.ffg.graph import Coupling, CouplingGraph
from cpomdp.types import Belief, LinearGaussianModel

# --- the tree --------------------------------------------------------------------
# Integer node indices; the r/h/a/b labels are ours (the library is name-agnostic).
ROOT, HUB, LEAF_A, LEAF_B = 0, 1, 2, 3
N_NODES = 4
DIMS = (1, 1, 1, 1)  # all scalar — keeps the joint a readable 4x4

# Prior on the hidden root: wide, so the leaf readings have something to sharpen.
PRIOR_MEAN = 0.0
PRIOR_VAR = 4.0

# Edges `child = W * parent + noise(Q)`. The hub HUB has three neighbours (ROOT, A, B),
# the branching a path cannot express.
# (parent, child, W, Q)
EDGES = [
    (ROOT, HUB, 0.9, 0.05),
    (HUB, LEAF_A, 1.0, 0.03),
    (HUB, LEAF_B, 1.0, 0.03),
]

# The two observed leaves: scalar readouts, C=1 with noise R.
OBS_NODES = [LEAF_A, LEAF_B]
OBS_R = 0.10
READINGS = {LEAF_A: 1.2, LEAF_B: 1.1}  # consistent with r a touch above 1

# How close the two routes must land for the figure's claim to hold: float noise only.
EQUIV_TOL = 1e-7


def build_graph() -> tuple[CouplingGraph, Belief, dict[int, np.ndarray]]:
    """The tree as a `CouplingGraph`, with its root prior and the leaf readings."""
    couplings = tuple(
        Coupling(parent, child, GaussianCoupling([[w]], [[q]]), tau=0.0)
        for parent, child, w, q in EDGES
    )
    observations = {node: GaussianObservation([[1.0]], [[OBS_R]]) for node in OBS_NODES}
    graph = CouplingGraph(
        root=ROOT, dims=DIMS, couplings=couplings, observations=observations
    )
    prior = Belief(mean=[PRIOR_MEAN], cov=[[PRIOR_VAR]])
    readings = {node: np.array([READINGS[node]]) for node in OBS_NODES}
    return graph, prior, readings


# --- the by-hand route: flatten the tree for the normal (Kalman) backend ---------


def _joint_prior() -> tuple[np.ndarray, np.ndarray]:
    """The 4-D joint prior implied by the root prior + couplings (moment form).

    A root-outward forward pass: each child's mean is ``W * parent``, its variance
    ``W^2 * var_parent + Q``, and its cross-covariance to every node already placed is
    ``W * cov(parent, that node)``. This is the assembly the graph spares you.
    """
    mean = np.zeros(N_NODES)
    cov = np.zeros((N_NODES, N_NODES))
    mean[ROOT] = PRIOR_MEAN
    cov[ROOT, ROOT] = PRIOR_VAR

    placed = [ROOT]
    # EDGES is listed parent-before-child, so one pass suffices.
    for parent, child, w, q in EDGES:
        mean[child] = w * mean[parent]
        cov[child, child] = w * cov[parent, parent] * w + q
        for node in placed:
            cross = w * cov[parent, node]
            cov[child, node] = cross
            cov[node, child] = cross
        placed.append(child)
    return mean, cov


def _stacked_observation() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack the per-leaf readouts into one tall ``(C, R, y)`` over the 4-D joint."""
    c = np.zeros((len(OBS_NODES), N_NODES))
    for row, node in enumerate(OBS_NODES):
        c[row, node] = 1.0
    r = OBS_R * np.eye(len(OBS_NODES))
    y = np.array([READINGS[node] for node in OBS_NODES])
    return c, r, y


def infer_flattened() -> Belief:
    """Root posterior via the normal backend on the hand-flattened joint model.

    Identity dynamics with zero process noise make the predict a no-op, so the single
    `KalmanBackend` update just fuses the stacked observations into the joint prior;
    marginalising the posterior to the root block recovers the belief over ``r``.
    """
    joint_mean, joint_cov = _joint_prior()
    c, r, y = _stacked_observation()
    model = LinearGaussianModel(
        dynamics=np.eye(N_NODES),
        sensor_model=c,
        dynamics_noise=np.zeros((N_NODES, N_NODES)),
        sensor_noise=r,
        prior=Belief(mean=joint_mean, cov=joint_cov),
    )
    posterior = KalmanBackend(model).infer_states(y, Belief(joint_mean, joint_cov))
    mean = np.asarray(posterior.mean)
    cov = np.asarray(posterior.cov)
    return Belief(mean=mean[[ROOT]], cov=cov[np.ix_([ROOT], [ROOT])])


def assemble_joint_precision() -> np.ndarray:
    """The full joint precision the by-hand route builds (prior + couplings + reads).

    Information form, so the tree shows up directly as sparsity: each edge contributes
    ``W^2/Q`` to the parent diagonal, ``1/Q`` to the child diagonal, and ``-W/Q`` off
    the diagonal; each reading adds ``1/R`` to its node; the prior seeds the root. This
    matrix is only for the figure; the numeric belief comes from `infer_flattened`.
    """
    lam = np.zeros((N_NODES, N_NODES))
    lam[ROOT, ROOT] += 1.0 / PRIOR_VAR
    for parent, child, w, q in EDGES:
        lam[child, child] += 1.0 / q
        lam[parent, parent] += w * w / q
        lam[parent, child] += -w / q
        lam[child, parent] += -w / q
    for node in OBS_NODES:
        lam[node, node] += 1.0 / OBS_R
    return lam


def _posteriors() -> tuple[Belief, Belief, float]:
    """Both routes' root beliefs and the max absolute gap between them."""
    graph, prior, readings = build_graph()
    native = graph.infer(prior, readings)
    flat = infer_flattened()
    gap = max(
        float(np.max(np.abs(np.asarray(native.mean) - np.asarray(flat.mean)))),
        float(np.max(np.abs(np.asarray(native.cov) - np.asarray(flat.cov)))),
    )
    return native, flat, gap


# --- rendering -------------------------------------------------------------------

BG, INK, GRID, GRAY = "#FAFAFA", "#2B2B2B", "#D8D8D8", "#9A9A9A"
TARGET, HUB_C, LEAF_C = "#CC79A7", "#56707F", "#0072B2"  # root / hub / observed leaf
GLOW = "#E69F00"
PANEL = "#F0F0F0"  # the code box background

NODE_META = {
    ROOT: ("r", (0.50, 0.86), TARGET),
    HUB: ("h", (0.50, 0.50), HUB_C),
    LEAF_A: ("a", (0.27, 0.15), LEAF_C),
    LEAF_B: ("b", (0.73, 0.15), LEAF_C),
}

CODE = """\
couplings = (
    Coupling(r, h, GaussianCoupling([[0.9]], [[0.05]]), tau=0.0),
    Coupling(h, a, GaussianCoupling([[1.0]], [[0.03]]), tau=0.0),
    Coupling(h, b, GaussianCoupling([[1.0]], [[0.03]]), tau=0.0),
)
graph  = CouplingGraph(root=r, dims=dims,
                       couplings=couplings, observations=obs)
belief = graph.infer(prior, {a: y_a, b: y_b})"""


def _draw_tree(ax) -> None:
    from matplotlib.patches import Circle

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("what you declare:  a branching tree", color=INK, fontsize=11, pad=4)

    for parent, child, w, _q in EDGES:
        (px, py), (cx, cy) = NODE_META[parent][1], NODE_META[child][1]
        ax.plot(
            [px, cx],
            [py, cy],
            color=GRAY,
            lw=2.2,
            zorder=1,
            solid_capstyle="round",
        )
        ax.text(
            (px + cx) / 2 + 0.035,
            (py + cy) / 2,
            f"W={w}",
            color=INK,
            fontsize=8,
            zorder=2,
        )

    for node, (label, (x, y), face) in NODE_META.items():
        observed = node in OBS_NODES
        ax.add_patch(
            Circle((x, y), 0.085, facecolor=face, edgecolor=INK, lw=1.4, zorder=4)
        )
        ax.text(
            x,
            y,
            label,
            color="white",
            fontsize=12,
            ha="center",
            va="center",
            zorder=5,
            fontweight="bold",
        )
        if observed:  # a solid ring marks a measured leaf, with its reading
            ax.add_patch(
                Circle(
                    (x, y),
                    0.105,
                    facecolor="none",
                    edgecolor=INK,
                    lw=1.0,
                    ls=":",
                    zorder=3,
                )
            )
            ax.text(
                x,
                y - 0.165,
                f"observed\ny={READINGS[node]}",
                color=INK,
                fontsize=7.5,
                ha="center",
                va="top",
                zorder=4,
            )
    rx, ry = NODE_META[ROOT][1]
    ax.text(
        rx + 0.13,
        ry,
        "← the unknown\n   we want",
        color=TARGET,
        fontsize=8,
        ha="left",
        va="center",
        zorder=4,
    )
    hx, hy = NODE_META[HUB][1]
    ax.text(
        hx + 0.13,
        hy,
        "hidden hub\n(degree 3)",
        color=INK,
        fontsize=7.5,
        ha="left",
        va="center",
        zorder=4,
    )


def _draw_code(ax) -> None:
    from matplotlib.patches import FancyBboxPatch

    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.01, 0.16),
            0.98,
            0.72,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            transform=ax.transAxes,
            facecolor=PANEL,
            edgecolor=GRID,
            lw=1.0,
            zorder=1,
        )
    )
    ax.text(
        0.045,
        0.80,
        CODE,
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=8.0,
        color=INK,
        zorder=2,
    )
    ax.text(
        0.5,
        0.07,
        "three edges, one infer call — the same code runs any tree",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color=INK,
        fontsize=8.5,
        style="italic",
    )


def _draw_matrix(ax, lam: np.ndarray) -> None:
    from matplotlib import colormaps

    labels = [NODE_META[i][0] for i in range(N_NODES)]
    cmap = colormaps["RdBu_r"]
    vmax = float(np.max(np.abs(lam)))
    ax.imshow(lam, cmap=cmap, vmin=-vmax, vmax=vmax, zorder=1)
    ax.set_title(
        "what you assemble by hand:  the joint precision",
        color=INK,
        fontsize=11,
        pad=4,
    )
    ax.set_xticks(range(N_NODES))
    ax.set_yticks(range(N_NODES))
    ax.set_xticklabels(labels, fontsize=9, color=INK)
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.tick_params(length=0)
    for i in range(N_NODES):
        for j in range(N_NODES):
            val = lam[i, j]
            if abs(val) < 1e-9:
                ax.text(
                    j,
                    i,
                    "0",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=GRAY,
                    zorder=2,
                )
                continue
            text_color = "white" if abs(val) > 0.6 * vmax else INK
            ax.text(
                j,
                i,
                f"{val:.1f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color=text_color,
                zorder=2,
            )
    ax.text(
        0.5,
        -0.07,
        "invert, marginalise out h, a, b — and re-derive it on every rewiring",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color=INK,
        fontsize=8.5,
        style="italic",
    )


def render(out_path: Path) -> Path:
    """Build the two-panel difference figure and write it to ``out_path``."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    native, _flat, gap = _posteriors()
    assert gap < EQUIV_TOL, f"routes disagree by {gap:.2e}; equivalence broken"
    mu, var = float(native.mean[0]), float(native.cov[0, 0])
    lam = assemble_joint_precision()

    fig = plt.figure(figsize=(12.0, 6.0))
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.08, 1.0],
        height_ratios=[1.4, 1.05],
        left=0.04,
        right=0.97,
        top=0.86,
        bottom=0.18,
        wspace=0.16,
        hspace=0.14,
    )
    ax_tree = fig.add_subplot(gs[0, 0])
    ax_code = fig.add_subplot(gs[1, 0])
    ax_mat = fig.add_subplot(gs[:, 1])
    for ax in (ax_tree, ax_code, ax_mat):
        ax.set_facecolor(BG)

    _draw_tree(ax_tree)
    _draw_code(ax_code)
    _draw_matrix(ax_mat, lam)

    fig.suptitle(
        "same posterior, two routes — the difference is the work, not the answer",
        color=INK,
        fontsize=13,
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.075,
        f"both routes  ->  belief over r:   mu = {mu:.3f},   sigma^2 = {var:.3f}"
        f"      identical to {gap:.0e}",
        ha="center",
        va="bottom",
        color=INK,
        fontsize=10.5,
    )
    fig.text(
        0.02,
        0.025,
        "cpomdp v0.4  ·  CouplingGraph.infer vs flattened KalmanBackend",
        ha="left",
        va="bottom",
        color=GRAY,
        fontsize=7.5,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=BG)
    plt.close(fig)
    return out_path


def _print_check() -> None:
    native, flat, gap = _posteriors()
    nm, nv = float(native.mean[0]), float(native.cov[0, 0])
    fm, fv = float(flat.mean[0]), float(flat.cov[0, 0])
    verdict = "PASS" if gap < EQUIV_TOL else "FAIL"
    print("belief over r — same tree, two routes:")
    print(f"  CouplingGraph.infer    : mu={nm:.6f}  var={nv:.6f}")
    print(f"  flattened KalmanBackend: mu={fm:.6f}  var={fv:.6f}")
    print(f"  max |difference|       : {gap:.2e}  ->  {verdict}")


def main() -> None:
    """``--check`` prints the two routes' posteriors; otherwise render the figure."""
    if "--check" in sys.argv:
        _print_check()
        return
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out_path = Path(args[0]) if args else Path("docs/assets/coupling_graph.png")
    print(f"rendering -> {render(out_path)}")


if __name__ == "__main__":
    main()
