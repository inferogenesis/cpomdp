"""The single-chain EFE epistemic collapse, and how a state-dependent sensor breaks it.

This is the model class of the paper's section 3.1 made executable: one node, no
couplings, additive control, and a sensor whose noise `R(x)` depends on the state — run
through the flat `KalmanBackend(CallableSensor)` route (the coupling-free `R(x)` case
the `CouplingGraphBackend.to_flat_model` docstring points at: build the Kalman model
with a `CallableSensor` directly rather than flattening a graph). The T-maze demo
illustrates the same mechanism on a richer branching graph; here it is the theorem's
own chain.

`--check` asserts Theorem 1's clauses (i) and (iii) on this class, plus the pinning
Lemma 1 rests on, with no plotting deps:

- (i)   the posterior covariance and the Kalman gain *move with the action* — the dual
        effect, the control reaching a covariance rather than only a mean;
- (iii) the one-step epistemic value *varies* across a candidate-action grid;
- twin  freezing `R` at the predicted mean `μ⁻` makes that same term dead-flat —
        ADR-003's collapse: a fixed linear-Gaussian sensor with fixed process noise
        reduces EFE to LQR;
- pin   `R(μ⁻(a))` — the noise the sensor presents at the predicted mean — traces a
        curve across the grid. No horizontal line follows a curve, which is the
        pinning argument drawn: a fixed noise schedule is named in advance and cannot
        track it.

The bare command renders the two-panel figure that plots the same sweep:

- LEFT (fixed sensor): the epistemic term is dead-flat across actions, so `G`'s minimum
  is driven entirely by the pragmatic term — `argmin G == argmin pragmatic`.
- RIGHT (state-dependent sensor): a "precision well" makes the sensor sharp near a
  beacon away from the goal. The epistemic term *curves* (peaks at the beacon) and
  `G`'s minimum is pulled off the goal toward information — the detour.

The right-hand sensor is a real `CallableSensor` supplying a position-dependent
`R(x)` to `expected_free_energy` through the `gaussianize` seam.

Run:
    uv run --no-sync python examples/efe_collapse_figure.py --check
    uv run --extra examples python examples/efe_collapse_figure.py
Output (bare command): docs/assets/efe_collapse.png
"""

import sys
from pathlib import Path

import gallery
import jax.numpy as jnp
import numpy as np

from cpomdp.backends.kalman import KalmanBackend, _gain_and_posterior_cov
from cpomdp.efe import expected_free_energy
from cpomdp.observation import CallableSensor
from cpomdp.selection import Preference
from cpomdp.types import Belief, LinearGaussianModel

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "efe_collapse.png"

# A 1-D single integrator: action moves the observed position directly, so a
# one-step sweep tells a clean story (μ⁺ = μ + a, observation = position).
DYNAMICS_MATRIX = [[1.0]]
CONTROL_MATRIX = [[1.0]]
OBSERVATION_MATRIX = [[1.0]]
DYNAMICS_NOISE = [[0.01]]
OBSERVATION_NOISE = [[0.3]]
BELIEF = Belief(mean=[0.0], cov=[[2.0]])
GOAL = Preference(goal=[0.0], precision=[[0.4]])  # prefer to observe position 0
ACTIONS = jnp.linspace(-2.0, 4.0, 400)
BEACON = 1.5  # the sensor is sharpest here — away from the goal at 0
# The R(x) well the figure plots and --check asserts on — one source, so the gate and
# the picture can never disagree about the sensor.
WELL_PARAMS = {"beacon": BEACON, "width": 0.6, "r_lo": 0.02, "r_hi": 0.8}


def _precision_well_noise(x, params):
    """R(x) for a 'precision well': dips to ``r_lo`` at the beacon, rises to ``r_hi``.

    Module-level (jit-safe) so it can ride in ``CallableSensor``'s static aux; all
    tunables live in ``params``.
    """
    pos = x[0]
    falloff = 1.0 - jnp.exp(
        -((pos - params["beacon"]) ** 2) / (2.0 * params["width"] ** 2)
    )
    r = params["r_lo"] + (params["r_hi"] - params["r_lo"]) * falloff
    return jnp.array([[r]])


def _model(observation_model=None):
    fixed = observation_model is None  # a sensor carries C and R itself
    return LinearGaussianModel(
        dynamics_matrix=DYNAMICS_MATRIX,
        observation_matrix=OBSERVATION_MATRIX if fixed else None,
        dynamics_noise=DYNAMICS_NOISE,
        observation_noise=OBSERVATION_NOISE if fixed else None,
        prior=BELIEF,
        control_matrix=CONTROL_MATRIX,
        observation_model=observation_model,
    )


def _well() -> CallableSensor:
    """The state-dependent sensor: constant ``C``, ``R(x)`` dipping at the beacon."""
    return CallableSensor(OBSERVATION_MATRIX, _precision_well_noise, WELL_PARAMS)


def _sweep(model):
    """Pragmatic, epistemic, and G over this demo's action sweep."""
    return gallery.efe_sweep(model, BELIEF, GOAL, ACTIONS)


def _panel(ax, prag, epi, g, title, note):
    """This demo's EFE panel: the shared one, labelled for a directly observed state."""
    gallery.draw_efe_panel(
        ax,
        ACTIONS,
        prag,
        epi,
        g,
        title=title,
        note=note,
        xlabel="action  (one-step move of observed position)",
    )


def check() -> None:
    """Assert Theorem 1 on section 3.1's own model class — the single R(x) chain.

    One node, no couplings, additive control, and a state-dependent ``R(x)`` sensor,
    run through the flat ``KalmanBackend(CallableSensor)`` route. The sweep is the same
    one the figure plots; here the clauses are asserted, not drawn.
    """
    live = _model(observation_model=_well())

    # The single chain actually filters. KalmanBackend(CallableSensor) is the
    # coupling-free R(x) route the CouplingGraphBackend.to_flat_model docstring points
    # at (build the Kalman model with a CallableSensor directly). One update at the
    # beacon — where R dips — sharpens the belief: the section 3.1 model class, run.
    prior = Belief(mean=[BEACON], cov=[[2.0]])
    post = KalmanBackend(live).infer_states(
        jnp.array([BEACON]), prior, action=jnp.array([0.0])
    )
    prior_var, post_var = float(prior.cov[0, 0]), float(post.cov[0, 0])
    assert post_var < prior_var
    print("The single-chain model filters (KalmanBackend + CallableSensor):")
    print(f"  belief var at the beacon: {prior_var:.2f} → {post_var:.3f} (sharpened)\n")

    # The frozen-R twin: the same sensor, but R pinned at what a flat Kalman would
    # linearize once at the prior mean μ⁻ and freeze. Any fixed R collapses the
    # epistemic term to a constant (ADR-003), whatever the action.
    r_frozen = float(_precision_well_noise(jnp.asarray(BELIEF.mean), WELL_PARAMS)[0, 0])
    frozen = LinearGaussianModel(
        DYNAMICS_MATRIX,
        observation_matrix=OBSERVATION_MATRIX,
        dynamics_noise=DYNAMICS_NOISE,
        observation_noise=[[r_frozen]],
        prior=BELIEF,
        control_matrix=CONTROL_MATRIX,
    )

    a_mat, b_mat = np.asarray(DYNAMICS_MATRIX), np.asarray(CONTROL_MATRIX)
    mu = np.asarray(BELIEF.mean)
    epi_live, epi_frozen, r_curve, post_vars, gains = [], [], [], [], []
    for a in np.asarray(ACTIONS):
        live_efe = expected_free_energy(live, BELIEF, jnp.array([a]), GOAL)
        frozen_efe = expected_free_energy(frozen, BELIEF, jnp.array([a]), GOAL)
        epi_live.append(float(live_efe.epistemic))
        epi_frozen.append(float(frozen_efe.epistemic))
        mu_minus = a_mat @ mu + b_mat @ np.array([a])  # μ⁻(a) = A·μ + B·a
        r_here = _precision_well_noise(jnp.asarray(mu_minus), WELL_PARAMS)
        r_curve.append(float(r_here[0, 0]))
        # The covariance half of the same step: the action chose where R was read, so
        # it also chose this posterior and this gain.
        gain, cov_post = _gain_and_posterior_cov(
            jnp.asarray(DYNAMICS_MATRIX),
            jnp.asarray(OBSERVATION_MATRIX),
            jnp.asarray(DYNAMICS_NOISE),
            r_here,
            jnp.asarray(BELIEF.cov),
        )
        post_vars.append(float(cov_post[0, 0]))
        gains.append(float(gain[0, 0]))
    epi_live, epi_frozen, r_curve, post_vars, gains = map(
        np.array, (epi_live, epi_frozen, r_curve, post_vars, gains)
    )

    live_swing = float(np.ptp(epi_live))
    frozen_swing = float(np.ptp(epi_frozen))
    r_swing = float(np.ptp(r_curve))
    post_swing = float(np.ptp(post_vars))
    gain_swing = float(np.ptp(gains))

    # (i) the dual effect: the action reaches the posterior covariance, and the gain
    # inherits it. Under a fixed sensor both would be constants named in advance.
    assert post_swing > 1e-6
    assert gain_swing > 1e-6
    # (iii) the epistemic value varies across the candidate-action grid.
    assert live_swing > 1e-3
    # twin: freeze R and that same term is dead-flat (the ADR-003 collapse).
    assert frozen_swing < 1e-9
    # the pinning: R(μ⁻(a)) traces a curve, and no constant follows a curve.
    assert r_swing > 1e-3

    print("Theorem 1 on the single-chain R(x) model class (section 3.1):")
    print(
        f"  (i)   posterior variance over the grid: {post_vars.min():.4f} → "
        f"{post_vars.max():.4f} (swing {post_swing:.4f}); gain swing {gain_swing:.4f} "
        f"— the DUAL EFFECT"
    )
    print(
        f"  (iii) epistemic value over the action grid: swing = {live_swing:.2f} nats "
        f"(peak {epi_live.max():.2f} at μ⁻ = beacon) — VARIES"
    )
    print(
        f"  twin  frozen-R epistemic swing = {frozen_swing:.0e} nats — flat "
        f"(ADR-003 collapse)"
    )
    print(
        f"  pin   R(μ⁻) over the grid: range = {r_swing:.2f} "
        f"({r_curve.min():.2f} → {r_curve.max():.2f}) — a curve no constant follows"
    )
    print("\nAll clauses PASS.")


def main():
    """Render the two-panel collapse/detour figure to docs/assets/."""
    import matplotlib.pyplot as plt

    prag_f, epi_f, g_f = _sweep(_model())  # fixed sensor (observation=None)
    prag_s, epi_s, g_s = _sweep(_model(observation_model=_well()))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=True)
    fig.suptitle(
        "Expected Free Energy: epistemic collapse, and how a state-dependent "
        "sensor breaks it",
        fontsize=13.5,
        fontweight="bold",
    )

    a_prag_f = np.asarray(ACTIONS)[int(np.argmin(prag_f))]
    a_g_f = np.asarray(ACTIONS)[int(np.argmin(g_f))]
    _panel(
        ax_l,
        prag_f,
        epi_f,
        g_f,
        "Fixed sensor — collapse (ADR-003)",
        f"epistemic is flat → argmin G ({a_g_f:.2f}) == argmin pragmatic "
        f"({a_prag_f:.2f}).\nEFE reduces to LQR; no information-seeking.",
    )
    ax_l.set_ylabel("nats")

    a_prag_s = np.asarray(ACTIONS)[int(np.argmin(prag_s))]
    a_g_s = np.asarray(ACTIONS)[int(np.argmin(g_s))]
    ax_r.axvline(BEACON, color="#999999", ls="-", lw=1.0, alpha=0.7)
    _panel(
        ax_r,
        prag_s,
        epi_s,
        g_s,
        "State-dependent sensor — the detour",
        f"epistemic peaks at the beacon (grey) → argmin G ({a_g_s:.2f}) detours "
        f"off the goal\n(argmin pragmatic {a_prag_s:.2f}) toward information.",
    )

    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    print(f"wrote {gallery.save_figure(fig, OUT, dpi=150, tight=True)}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        main()
