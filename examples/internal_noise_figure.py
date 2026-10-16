"""Figure: the epistemic collapse breaks from the *inside* — through process noise.

The companion to examples/efe_collapse_figure.py. There, a state-dependent sensor
R(x) made the epistemic term action-dependent. Here the observation noise R is held
FIXED and the action-dependence comes entirely from state-dependent internal
process noise Q(x) — the route RFC-001 chapter 8 cares about, where the binding
precision constraint lives in internal processing rather than the sensor.

- LEFT (fixed Q): epistemic is flat across actions — the ADR-003 collapse, same as
  a fixed sensor. R and Q are both constant, so Σ⁺, S, and the info gain do not
  move with the action.
- RIGHT (state-dependent Q(x)): epistemic curves. Q is evaluated at μ⁺ (the
  arrived-at state), so it depends on the action; Σ⁺ = AΣAᵀ + Q(μ⁺) and the info
  gain move with it. R is the SAME constant as on the left — the whole effect is
  internal.

Run: `uv run --extra examples python examples/internal_noise_figure.py`
Output: docs/assets/internal_noise.png
"""

from pathlib import Path

import gallery
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from cpomdp.dynamics import CallableProcessNoise
from cpomdp.selection import Preference
from cpomdp.types import Belief, LinearGaussianModel

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "internal_noise.png"

# 1-D single integrator: μ⁺ = μ + a, so the action moves the state directly and a
# one-step sweep is clean. R (observation noise) is fixed throughout — the only thing
# that varies between the panels is Q.
DYNAMICS_MATRIX = [[1.0]]
CONTROL_MATRIX = [[1.0]]
OBSERVATION_MATRIX = [[1.0]]
OBSERVATION_NOISE = [[0.3]]  # the SAME on both panels
DYNAMICS_NOISE = [[0.2]]
BELIEF = Belief(mean=[0.0], cov=[[0.4]])
GOAL = Preference(goal=[0.0], precision=[[0.15]])
ACTIONS = jnp.linspace(-2.5, 3.5, 400)
BUMP = 1.2  # Q(x) is noisiest here — internal processing is least precise


def _bump_q(x, params):
    """Q(x): a bump of high internal noise at params['bump']. Module-level fn."""
    gap = x[0] - params["bump"]
    return jnp.array(
        [
            [
                params["base"]
                + params["amp"] * jnp.exp(-(gap**2) / (2.0 * params["width"] ** 2))
            ]
        ]
    )


def _model(dynamics_noise_model=None):
    return LinearGaussianModel(
        dynamics_matrix=DYNAMICS_MATRIX,
        observation_matrix=OBSERVATION_MATRIX,
        dynamics_noise=DYNAMICS_NOISE if dynamics_noise_model is None else None,
        observation_noise=OBSERVATION_NOISE,
        prior=BELIEF,
        control_matrix=CONTROL_MATRIX,
        dynamics_noise_model=dynamics_noise_model,
    )


def _sweep(model):
    """Pragmatic, epistemic, and G over this demo's action sweep."""
    return gallery.efe_sweep(model, BELIEF, GOAL, ACTIONS)


def _panel(ax, prag, epi, g, title, note):
    """This demo's EFE panel: the shared one, labelled for a latent state."""
    gallery.draw_efe_panel(
        ax,
        ACTIONS,
        prag,
        epi,
        g,
        title=title,
        note=note,
        xlabel="action  (one-step move of state)",
    )


def main():
    """Render the two-panel internal-noise figure to docs/assets/."""
    prag_f, epi_f, g_f = _sweep(_model())  # fixed Q (process_noise=None)
    q_well = CallableProcessNoise(
        _bump_q,
        {
            "bump": BUMP,
            "base": jnp.array(0.05),
            "amp": jnp.array(1.6),
            "width": jnp.array(0.6),
        },
    )
    prag_s, epi_s, g_s = _sweep(_model(dynamics_noise_model=q_well))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=True)
    fig.suptitle(
        "Internal process noise breaks the collapse from the inside (R held fixed)",
        fontsize=13.5,
        fontweight="bold",
    )

    a_g_f = np.asarray(ACTIONS)[int(np.argmin(g_f))]
    _panel(
        ax_l,
        prag_f,
        epi_f,
        g_f,
        "Fixed Q — collapse (ADR-003)",
        f"epistemic is flat → argmin G ({a_g_f:.2f}) tracks the goal.\n"
        "Σ⁺ and S do not move with the action.",
    )
    ax_l.set_ylabel("nats")

    a_g_s = np.asarray(ACTIONS)[int(np.argmin(g_s))]
    ax_r.axvline(BUMP, color="#999999", ls="-", lw=1.0, alpha=0.7)
    _panel(
        ax_r,
        prag_s,
        epi_s,
        g_s,
        "State-dependent Q(x) — the internal route",
        f"epistemic curves through Q(μ⁺) (grey = noisiest x) → argmin G ({a_g_s:.2f}) "
        "shifts.\nR is the same constant as on the left; the effect is internal.",
    )

    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    print(f"wrote {gallery.save_figure(fig, OUT, dpi=150, tight=True)}")


if __name__ == "__main__":
    main()
