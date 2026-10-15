"""A chemotaxis network as a branching factor graph -- the shape, not the biophysics.

The E. coli chemotaxis pathway is a real branching network: a receptor-driven CheA
kinase hub feeds a fast CheY -> motor branch and a slow CheB methylation branch. That
is a tree with a degree-3 node -- a shape no chain (and so no plain Kalman filter laid
out as one) can hold. cpomdp declares it as a ``CouplingGraph`` and infers the hidden
CheA hub from the downstream readouts (CheB and the two motors), exact to a flattened
Kalman.

This is the shape, not a chemotaxis model. It leaves out the CheB -> receptor feedback
that makes real chemotaxis adaptive -- feedback is a loop, a ``CouplingGraph`` is a
tree, so it can't hold it -- and there is no swimming or drift. A faithful E. coli model
is a build-on-top, not a v0.4 feature (RFC-002, ADR-020). Read it as "a realistic
branching network runs natively as an FFG", nothing more.

Reuses ``chemotaxis_model.py``. Needs the ``examples`` extra (matplotlib)::

    uv run --extra examples python examples/ffg/chemotaxis_figure.py --check
    uv run --extra examples python examples/ffg/chemotaxis_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # examples/, for `gallery`

import gallery
import numpy as np
from chemotaxis_model import CHEA, CHEB, CHEY, MOTOR_A, MOTOR_B, chemotaxis_ffg

from cpomdp.backends.kalman import KalmanBackend
from cpomdp.types import Belief, LinearGaussianModel

N_NODES = 5
DT = 0.1  # only sets the OU transitions, which static structural inference never uses

# A wide prior on the hidden CheA hub, readings on the three observed leaves; the FFG
# resolves the hub through them. Values are illustrative -- no biological units.
PRIOR_MEAN, PRIOR_VAR = 0.0, 4.0
READINGS = {CHEB: 0.9, MOTOR_A: 1.1, MOTOR_B: 1.0}
EQUIV_TOL = 1e-7


def _graph():
    """The chemotaxis ``CouplingGraph``; the returned transitions go unused here."""
    graph, _transitions = chemotaxis_ffg(DT)
    return graph


def _topology(graph) -> tuple[dict[int, tuple[int, float, float]], dict[int, float]]:
    """Read scalar edges ``child -> (parent, W, Q)`` and readout noise off the graph."""
    edges = {
        c.child: (
            c.parent,
            float(np.asarray(c.factor.coupling_matrix)[0, 0]),
            float(np.asarray(c.factor.coupling_noise)[0, 0]),
        )
        for c in graph.couplings
    }
    obs_r = {
        node: float(np.asarray(o.observation_noise)[0, 0])
        for node, o in graph.observations.items()
    }
    return edges, obs_r


def _joint_prior(edges) -> tuple[np.ndarray, np.ndarray]:
    """The 5-D joint prior from the CheA prior + couplings, placed root-outward."""
    mean = np.zeros(N_NODES)
    cov = np.zeros((N_NODES, N_NODES))
    mean[CHEA], cov[CHEA, CHEA] = PRIOR_MEAN, PRIOR_VAR
    placed, remaining = [CHEA], dict(edges)
    while remaining:  # place a child once its parent is down; drains in a few rounds
        for child, (parent, w, q) in list(remaining.items()):
            if parent not in placed:
                continue
            mean[child] = w * mean[parent]
            cov[child, child] = w * cov[parent, parent] * w + q
            for node in placed:
                cross = w * cov[parent, node]
                cov[child, node] = cov[node, child] = cross
            placed.append(child)
            del remaining[child]
    return mean, cov


def _infer_flattened(edges, obs_r) -> Belief:
    """CheA posterior via a plain KalmanBackend on the hand-flattened joint."""
    mean, cov = _joint_prior(edges)
    nodes = sorted(obs_r)
    c = np.zeros((len(nodes), N_NODES))
    for row, node in enumerate(nodes):
        c[row, node] = 1.0
    r = np.diag([obs_r[node] for node in nodes])
    y = np.array([READINGS[node] for node in nodes])
    model = LinearGaussianModel(
        dynamics_matrix=np.eye(N_NODES),
        observation_matrix=c,
        dynamics_noise=np.zeros((N_NODES, N_NODES)),
        observation_noise=r,
        prior=Belief(mean=mean, cov=cov),
    )
    posterior = KalmanBackend(model).infer_states(y, Belief(mean, cov))
    pm, pc = np.asarray(posterior.mean), np.asarray(posterior.cov)
    return Belief(mean=pm[[CHEA]], cov=pc[np.ix_([CHEA], [CHEA])])


def _posteriors() -> tuple[Belief, Belief, float]:
    """Both routes' CheA belief and the max gap between them."""
    graph = _graph()
    edges, obs_r = _topology(graph)
    readings = {node: np.array([READINGS[node]]) for node in obs_r}
    native = graph.infer(Belief(mean=[PRIOR_MEAN], cov=[[PRIOR_VAR]]), readings)
    flat = _infer_flattened(edges, obs_r)
    gap = max(
        float(np.max(np.abs(np.asarray(native.mean) - np.asarray(flat.mean)))),
        float(np.max(np.abs(np.asarray(native.cov) - np.asarray(flat.cov)))),
    )
    return native, flat, gap


# --- rendering ------------------------------------------------------------------
BG, INK = gallery.DIAGRAM.bg, gallery.DIAGRAM.ink
GRID, GRAY = gallery.DIAGRAM.grid, gallery.DIAGRAM.faint
HUB_C = gallery.PINK  # the hidden CheA hub we infer
HIDDEN_C = "#56707F"  # other hidden node (CheY)
OBS_C = gallery.BLUE  # observed leaf

# label, (x, y) in axes fraction, colour, observed?
_NODES = {
    CHEA: ("CheA", (0.50, 0.86), HUB_C, False),
    CHEY: ("CheY", (0.34, 0.52), HIDDEN_C, False),
    CHEB: ("CheB", (0.70, 0.52), OBS_C, True),
    MOTOR_A: ("motor", (0.22, 0.16), OBS_C, True),
    MOTOR_B: ("motor", (0.48, 0.16), OBS_C, True),
}
_LINKS = [(CHEA, CHEY), (CHEA, CHEB), (CHEY, MOTOR_A), (CHEY, MOTOR_B)]


def _draw_network(ax) -> None:
    from matplotlib.patches import Circle

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for parent, child in _LINKS:  # generative direction parent -> child
        (px, py), (cx, cy) = _NODES[parent][1], _NODES[child][1]
        ax.annotate(
            "",
            xy=(cx, cy),
            xytext=(px, py),
            zorder=1,
            arrowprops={
                "arrowstyle": "-|>",
                "color": GRAY,
                "lw": 2.0,
                "shrinkA": 15,
                "shrinkB": 15,
            },
        )

    for node, (label, (x, y), face, observed) in _NODES.items():
        ax.add_patch(
            Circle((x, y), 0.058, facecolor=face, edgecolor=INK, lw=1.4, zorder=4)
        )
        ax.text(
            x,
            y,
            label,
            color="white",
            fontsize=8.5,
            ha="center",
            va="center",
            zorder=5,
            fontweight="bold",
        )
        if observed:
            ax.add_patch(
                Circle(
                    (x, y),
                    0.072,
                    facecolor="none",
                    edgecolor=INK,
                    lw=1.0,
                    ls=":",
                    zorder=3,
                )
            )
            ax.text(
                x,
                y - 0.10,
                f"observed  y={READINGS[node]}",
                color=INK,
                fontsize=7,
                ha="center",
                va="top",
                zorder=4,
            )

    ax.text(
        0.50,
        0.98,
        "receptor input",
        color=GRAY,
        fontsize=7.5,
        ha="center",
        va="top",
        style="italic",
    )
    ax.annotate(
        "",
        xy=_NODES[CHEA][1],
        xytext=(0.50, 0.955),
        arrowprops={"arrowstyle": "-|>", "color": GRID, "lw": 1.6},
    )
    ax.text(
        _NODES[CHEA][1][0] + 0.09,
        _NODES[CHEA][1][1],
        "hidden kinase hub\ncpomdp infers this",
        color=HUB_C,
        fontsize=7.5,
        ha="left",
        va="center",
        zorder=6,
    )
    ax.text(
        _NODES[CHEY][1][0] - 0.09,
        _NODES[CHEY][1][1],
        "degree 3 —\na chain can't\nhold it",
        color=INK,
        fontsize=7,
        ha="right",
        va="center",
        zorder=6,
    )


def render(out_path: Path) -> Path:
    """Draw the chemotaxis-network figure and write it to ``out_path``."""
    gallery.use_headless_backend()
    import matplotlib.pyplot as plt

    native, _flat, gap = _posteriors()
    assert gap < EQUIV_TOL, f"routes disagree by {gap:.2e}; equivalence broken"
    mu, var = float(native.mean[0]), float(native.cov[0, 0])

    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    _draw_network(ax)
    fig.suptitle(
        "a chemotaxis network as a branching factor graph",
        color=INK,
        fontsize=13,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.155,
        f"hidden CheA, inferred through the readouts:  μ = {mu:.3f},  "
        f"σ² = {var:.3f}    exact to {gap:.0e} vs a flattened Kalman",
        ha="center",
        va="center",
        color=INK,
        fontsize=9.5,
    )
    fig.text(
        0.5,
        0.105,
        "the certifiable side of the boundary: here a provably-optimal reference "
        "exists — the Kalman floor — and the exact FFG sits on it",
        ha="center",
        va="center",
        color=INK,
        fontsize=8,
        style="italic",
    )
    fig.text(
        0.5,
        0.05,
        "the shape, not the biophysics — no CheB→receptor feedback (a loop, "
        "not a tree), no swimming or efficiency;\na faithful E. coli model is a "
        "build-on-top (RFC-002, ADR-020)",
        ha="center",
        va="center",
        color=GRAY,
        fontsize=7.6,
    )
    fig.subplots_adjust(left=0.03, right=0.97, top=0.9, bottom=0.2)
    gallery.save_figure(fig, out_path, dpi=150, facecolor=BG)
    plt.close(fig)
    return out_path


def check() -> None:
    """Both routes' belief over the hidden CheA hub, and the assertion that they agree.

    Plotting-free, so `tests/test_example_checks.py` can call it in the base
    environment.
    """
    native, flat, gap = _posteriors()
    gallery.check_two_route_agreement(
        "hidden CheA hub — same network, two routes:", native, flat, gap, EQUIV_TOL
    )


def main() -> None:
    """``--check`` asserts both routes agree on CheA, otherwise render the figure."""
    gallery.figure_main(render, "docs/assets/chemotaxis.png", check=check)


if __name__ == "__main__":
    main()
