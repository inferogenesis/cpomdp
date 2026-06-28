"""Instrumental epistemics: the beacon resolves food location, not self-location.

Active inference's Expected Free Energy decomposes into an *epistemic*
(information-seeking) value and a *pragmatic*/*instrumental* (goal-seeking) value.
Epistemic value is genuinely *instrumental* — not merely curious — when the
uncertainty it resolves is decision-relevant: the discrete T-Maze task (Friston
et al. 2015, "Active inference and epistemic value") is the canonical case, where
visiting a cue resolves which arm holds the reward, changing the *subsequent*
action.

`bacillus_seeking_food.py` (the v0.3 flagship, now kept in "the journey") has the
beacon collapse uncertainty about the agent's *own* position instead — salience
without an instrumental payoff: knowing your own position more precisely doesn't
change which action is later correct. This demo promotes the food's position to
an explicit latent the agent does not know a priori, and wires the beacon's
existing precision-well mechanic to reveal *that* instead — now the resolved
uncertainty is decision-relevant (it changes where the agent then heads), the
property the original demo's epistemic value lacked.

The whole change, before -> after, in one picture (ADR-013)::

    # v0.3: the channel reads the agent's OWN position, and the noise it carries
    # is keyed on that SAME position -- self-revealing.
    sensor_model = I                       # C: o = agent_xy
    noise_fn(x, p) = beacon_noise(x, p)    # R(x): keyed on the channel's own block

    # v0.4 (here): the channel reads a DIFFERENT block (food - agent), but the
    # noise is keyed on the SAME agent-position block as before -- the beacon
    # mechanic itself is UNCHANGED, only what it is wired to reveal.
    sensor_model = [-I, I]                  # C: o = food_xy - agent_xy
    noise_fn(x, p) = beacon_noise(x[:2], p)  # R(x): still keyed on agent_xy only

State is now 4-D: ``[agent_xy, food_xy]``. The sensor still has the agent read its
own position (fixed precision — plain proprioception), but gains a second,
*relative* channel: ``o_disp = food_xy - agent_xy``, whose noise is the existing,
**unmodified** ``beacon_noise`` falloff from the flagship demo, evaluated at the
agent's own position. Visiting the beacon does not sharpen "where am I" anymore —
it sharpens "where is the food," which the agent cannot directly act on.

Same four-regime structure as the flagship, same single real knob (the goal
precision Λ), plus a genuine **classic LQR** regime (no epistemic term at all):
``LQRController`` only ever consumes the dynamics/control matrices and the cost
weights, never the sensor, so it works unmodified on this state-dependent-sensor
model — it just steers the agent block toward the *current* food-belief mean each
step (zero weight on the food block of its own cost), recomputed from the belief,
never from a static goal. The EFE agents instead use a single static
``Preference``, weighted only on the displacement channel with target ``[0, 0]``
("observe zero distance from food"); because the predicted reading is
``E[food_xy]⁺ - agent_xy⁺``, that one static target algebraically chases the
*current belief* of food's location too — no per-step preference rebuilding
(ADR-013 spells out why this beats the rebuild alternative for the EFE regimes,
and names the open multi-goal question it leaves unresolved).

This needs zero core-library changes (ADR-013): the model is just a bigger
``LinearGaussianModel``, the sensor is one ``CallableSensor`` with a 4x4
block-diagonal ``R(x)``, and ``expected_free_energy``/``LQRController`` don't care
how the observation/state vectors are built up. The mechanism is checked, not
just rendered: ``--scan`` runs the identical model through both
``KalmanBackend`` and ``ChainBackend`` (ADR-012 Phase 2.5) and checks they agree
to ``atol=1e-7``.

Needs the ``examples`` extra (matplotlib + pillow, neither a runtime dependency of
the library itself)::

    uv run --extra examples python examples/bacillus_uncertain_food.py        # GIF
    uv run --extra examples python examples/bacillus_uncertain_food.py --scan # metrics

(or ``pip install "cpomdp[examples]"`` from a regular install — see
``examples/README.md``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from bacillus_seeking_food import BEACON_PT, beacon_noise

from cpomdp.backends.kalman import KalmanBackend
from cpomdp.control import LQRController
from cpomdp.efe import expected_free_energy
from cpomdp.ffg.chain import ChainBackend
from cpomdp.observation import CallableSensor
from cpomdp.selection import Preference
from cpomdp.types import Belief, LinearGaussianModel

# --- world geometry -------------------------------------------------------------
DT = 0.16
START = np.array([-3.4, -2.6])
FOOD_TRUE = np.array([3.4, -2.6])  # the food's TRUE position — unknown to the agent
FOOD_PRIOR_MEAN = np.array([1.0, 0.0])  # the agent's a-priori guess: vague, wrong
FOOD_PRIOR_COV = 6.0  # wide: "loosely known," not "unknown" — has to detour to learn

# the beacon mechanic itself is UNCHANGED from the flagship demo (`beacon_noise`,
# imported above) — only the channel it's attached to differs. Reuse its tuning.
R_LO, R_HI, R_WIDTH = 0.02, 1.30, 2.3
R_SELF = 0.05  # fixed proprioceptive noise: the agent always senses ITSELF clearly
Q_AGENT = 2e-5  # near-zero process noise on the agent block (existing idiom)
Q_FOOD = 1e-6  # strictly positive (ChainBackend rejects Q=0) — food is stationary
PRIOR_COV_AGENT = 2.6
LQR_GOAL_PRECISION = 0.22  # the LQR agent's cost weight on (agent - food-belief)
LQR_EFFORT = 0.6

ACTION_LO, ACTION_HI = -2.4, 2.4
GRID_N = 25
N_STEPS = 90
ARRIVAL_THRESHOLD = 0.5  # "settled near the food" radius for the render's border cue

# --- the four regimes: same world, one real knob — the goal precision Λ --------
# `kind` is "lqr" (no epistemic term, steers toward the CURRENT food belief each
# step) or "efe" (minimise G = pragmatic - epistemic via the displacement
# channel). Tuned by sweep: above ~0.03 the agent never detours (averages the
# murk instead); ~0.015 is the clean detour-then-exploit case; below ~0.008 it
# parks at the beacon and never leaves.
REGIMES = [
    {
        "key": "lqr",
        "kind": "lqr",
        "precision": 0.0,
        "title": "classic LQR · no epistemic term",
        "note": "beelines toward its current food estimate — never detours",
        "accent": "#7A7A7A",
    },
    {
        "key": "sharp",
        "kind": "efe",
        "precision": 0.10,
        "title": "sharp Λ · the goal dominates",
        "note": "never detours — averages the murk on the way",
        "accent": "#56B4E9",
    },
    {
        "key": "balanced",
        "kind": "efe",
        "precision": 0.015,
        "title": "balanced Λ · detour, then dinner",
        "note": "detours to the beacon, learns where food is, then heads there",
        "accent": "#009E73",
    },
    {
        "key": "weak",
        "kind": "efe",
        "precision": 0.006,
        "title": "weak Λ · over-curious",
        "note": "so over-curious it parks at the beacon — never eats",
        "accent": "#CC79A7",
    },
]


def _beacon_params() -> dict[str, float]:
    """The beacon falloff's tunables — shared by ``build_model`` and ``simulate``.

    Single source of truth so the model's filtering noise and the simulator's
    truth-process noise can never drift apart (``model.observation`` is typed as
    the ``ObservationModel`` Protocol).
    """
    return {
        "bx": float(BEACON_PT[0]),
        "by": float(BEACON_PT[1]),
        "r_lo": R_LO,
        "r_hi": R_HI,
        "width": R_WIDTH,
    }


def build_model() -> LinearGaussianModel:
    """The 4-D ``[agent_xy, food_xy]`` model: two sensor channels, one beacon.

    ``sensor_model`` (C) is 4x4: rows 0-1 read the agent block directly (``o_self``);
    rows 2-3 read ``food_xy - agent_xy`` (``o_disp``). ``noise_fn`` returns a 4x4
    block-diagonal R(x): a FIXED ``R_SELF`` block (proprioception never sharpens or
    dulls) and the existing ``beacon_noise`` block, evaluated on ``x[:2]`` — the
    agent's own position component of the full predicted state — exactly the
    quantity the flagship demo's beacon already keys off.
    """
    c_self = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    c_disp = jnp.array([[-1.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 1.0]])
    sensor_model = jnp.concatenate([c_self, c_disp], axis=0)  # (4, 4)

    beacon_params = _beacon_params()

    def noise_fn(x, params):
        r_disp = beacon_noise(x[:2], params)  # unchanged flagship falloff
        r_self = R_SELF * jnp.eye(2)
        return jax.scipy.linalg.block_diag(r_self, r_disp)

    sensor = CallableSensor(
        sensor_model=sensor_model, noise_fn=noise_fn, noise_params=beacon_params
    )

    dynamics = jnp.eye(4)  # agent: single integrator; food: stationary
    control = jnp.array(
        [[DT, 0.0], [0.0, DT], [0.0, 0.0], [0.0, 0.0]]
    )  # control only drives the agent block
    dynamics_noise = jnp.diag(jnp.array([Q_AGENT, Q_AGENT, Q_FOOD, Q_FOOD]))

    prior_mean = jnp.concatenate([jnp.asarray(START), jnp.asarray(FOOD_PRIOR_MEAN)])
    prior_cov = jnp.diag(
        jnp.array([PRIOR_COV_AGENT, PRIOR_COV_AGENT, FOOD_PRIOR_COV, FOOD_PRIOR_COV])
    )

    return LinearGaussianModel(
        dynamics=dynamics,
        control=control,
        sensor_model=sensor_model,
        dynamics_noise=dynamics_noise,
        sensor_noise=R_SELF * jnp.eye(4),  # nominal; the live R comes from `sensor`
        prior=Belief(mean=prior_mean, cov=prior_cov),
        observation=sensor,
    )


def build_preference(precision: float) -> Preference:
    """A static obs-space preference: zero weight on self, weight Λ on "find food".

    Target is ``[0, 0, 0, 0]`` — irrelevant for the self block (its precision
    weight is 0) and "observe zero displacement from food" for the disp block.
    Because the predicted disp reading is ``E[food]⁺ - agent⁺``, this single
    static object chases whatever the agent currently believes about food's
    location — no per-step rebuild (ADR-013).
    """
    block = jax.scipy.linalg.block_diag(jnp.zeros((2, 2)), precision * jnp.eye(2))
    return Preference(goal=jnp.zeros(4), precision=block)


def _candidate_grid() -> jnp.ndarray:
    """The front-loaded GRID_N² grid of one-step action candidates, shape (k², 2)."""
    axis = jnp.linspace(ACTION_LO, ACTION_HI, GRID_N)
    ax, ay = jnp.meshgrid(axis, axis)
    return jnp.stack([ax.ravel(), ay.ravel()], axis=1)


@jax.jit
def _efe_grid(model, belief, preference, candidates):
    """The EFE ``G`` of every candidate action — one ``vmap`` of the library kernel."""
    return jax.vmap(lambda a: expected_free_energy(model, belief, a, preference)[0])(
        candidates
    )


def simulate(regime, backend_cls=KalmanBackend, *, seed=7):
    """Run one regime's closed perceive -> act loop under the given backend class.

    The food never moves (``FOOD_TRUE`` is the simulator's ground truth); the
    agent does, under whichever action the regime picks each step. The simulated
    reading is sampled from the SAME noise the model uses to filter — fixed
    ``R_SELF`` for the self channel, the unmodified ``beacon_noise`` falloff
    (evaluated at the agent's TRUE position) for the disp channel.

    Args:
        regime: one of the ``REGIMES`` dicts (``kind`` "lqr" or "efe", plus
            ``precision``).
        backend_cls: ``KalmanBackend`` or ``ChainBackend`` — same model, same
            perceive/act shape, swappable behind the ``InferenceBackend`` protocol.
        seed: RNG seed for the observation noise draws.

    Returns:
        ``(true_states, means, covs)``, each length ``N_STEPS + 1``: the agent's
        true 2-D trajectory, the full 4-D belief means, and the full 4x4 belief
        covariances.
    """
    rng = np.random.default_rng(seed)
    model = build_model()
    backend = backend_cls(model)
    beacon_params = _beacon_params()

    controller = preference = candidates = candidates_np = None
    if regime["kind"] == "lqr":
        goal_precision = jnp.diag(jnp.array([LQR_GOAL_PRECISION] * 2 + [0.0, 0.0]))
        controller = LQRController(
            model, goal_precision=goal_precision, effort_penalty=jnp.eye(2) * LQR_EFFORT
        )
    else:
        preference = build_preference(regime["precision"])
        candidates = _candidate_grid()
        candidates_np = np.asarray(candidates)

    def choose(belief):
        if regime["kind"] == "lqr":
            assert controller is not None
            goal = jnp.concatenate([belief.mean[2:4], jnp.zeros(2)])
            act = np.asarray(controller.action(belief.mean, goal))
            return np.clip(act, ACTION_LO, ACTION_HI)
        assert preference is not None  # built above on the "efe" path
        assert candidates is not None  # built above on the "efe" path
        g = _efe_grid(model, belief, preference, candidates)
        return candidates_np[int(np.argmin(np.asarray(g)))]

    a_mat = np.eye(2)
    b_mat = np.eye(2) * DT

    belief = model.prior
    true_agent = START.astype(float).copy()
    last_action = jnp.zeros(2)

    true_states = [true_agent.copy()]
    means = [np.asarray(belief.mean)]
    covs = [np.asarray(belief.cov)]

    for _ in range(N_STEPS):
        r_self = R_SELF * np.eye(2)
        r_disp = np.asarray(beacon_noise(jnp.asarray(true_agent), beacon_params))
        r_full = np.block([[r_self, np.zeros((2, 2))], [np.zeros((2, 2)), r_disp]])
        obs_mean = np.concatenate([true_agent, FOOD_TRUE - true_agent])
        obs = obs_mean + np.linalg.cholesky(r_full) @ rng.standard_normal(4)

        belief = backend.infer_states(obs, belief, last_action)
        action = np.asarray(choose(belief), dtype=float)
        last_action = jnp.asarray(action)

        true_states.append(true_agent.copy())
        means.append(np.asarray(belief.mean))
        covs.append(np.asarray(belief.cov))

        true_agent = a_mat @ true_agent + b_mat @ action

    return np.array(true_states), np.array(means), np.array(covs)


def _metrics(true_states, means, covs):
    """Behaviour summary: closest beacon approach, dwell, food-belief/reach error."""
    d_to_beacon = np.linalg.norm(true_states - BEACON_PT, axis=1)
    d_beacon = d_to_beacon.min()
    step_min = int(d_to_beacon.argmin())
    dwell = int((d_to_beacon < 0.6).sum())
    food_belief_err = np.linalg.norm(means[-1][2:4] - FOOD_TRUE)
    agent_to_food = np.linalg.norm(true_states[-1] - FOOD_TRUE)
    food_cov_trace = float(np.trace(covs[-1][2:, 2:]))
    return d_beacon, step_min, dwell, food_belief_err, agent_to_food, food_cov_trace


def _arrival_step(true_states, *, threshold=ARRIVAL_THRESHOLD):
    """First step the agent is within ``threshold`` of the food AND stays there.

    A transient close pass that then wanders off doesn't count — ``settled[i]``
    is true only if EVERY later step also stays under threshold, computed via a
    suffix-AND (a reversed cumulative product of the boolean "is close" array).

    Returns:
        The first settled step index, or ``None`` if it never settles.
    """
    close = np.linalg.norm(true_states - FOOD_TRUE, axis=1) < threshold
    settled = np.cumprod(close[::-1])[::-1].astype(bool)
    return int(np.argmax(settled)) if settled.any() else None


def check_backend_agreement(*, seed=11, n_steps=30):
    """``KalmanBackend`` vs ``ChainBackend`` on an IDENTICAL scripted input sequence.

    Deliberately not two independent closed loops: each regime's argmin (or LQR's
    belief-dependent goal) could in principle land on a different choice from a
    near-tied score under tiny numerical differences, and the resulting
    trajectories would then diverge for a reason that has nothing to do with the
    backends actually disagreeing. Feeding both backends the same scripted
    ``(observation, action)`` pairs isolates exactly the claim ADR-013/
    BUILD_PLAN.md Phase 3 cares about: do the two backends compute the same
    belief from the same inputs, on a sensor topology (a channel reading one
    state block, noise keyed on a different block) neither backend's existing
    test suite exercises. Mirrors the methodology in ``tests/test_ffg_chain.py``.

    Returns:
        ``(max_mean_diff, max_cov_diff)`` — the largest elementwise absolute
        difference between the two backends' beliefs, over every step.
    """
    rng = np.random.default_rng(seed)
    model = build_model()
    kalman, chain = KalmanBackend(model), ChainBackend(model)
    k_belief = c_belief = model.prior
    max_mean_diff = max_cov_diff = 0.0
    for _ in range(n_steps):
        obs = rng.standard_normal(4)
        action = rng.standard_normal(2)
        k_belief = kalman.infer_states(obs, k_belief, action)
        c_belief = chain.infer_states(obs, c_belief, action)
        max_mean_diff = max(
            max_mean_diff, float(np.abs(k_belief.mean - c_belief.mean).max())
        )
        max_cov_diff = max(
            max_cov_diff, float(np.abs(k_belief.cov - c_belief.cov).max())
        )
    return max_mean_diff, max_cov_diff


def scan():
    """Print each regime's behaviour metrics, then the backend agreement check."""
    print(
        f"world: start={START}  food_true={FOOD_TRUE}  "
        f"food_prior_mean={FOOD_PRIOR_MEAN}  beacon={BEACON_PT}\n"
    )
    for regime in REGIMES:
        ts, means, covs = simulate(regime)
        d_beacon, step_min, dwell, food_err, agent_to_food, food_cov_tr = _metrics(
            ts, means, covs
        )
        tag = "LQR" if regime["kind"] == "lqr" else f"Λ={regime['precision']:g}"
        print(
            f"  {tag:>9}: minBeacon={d_beacon:5.2f}@{step_min:2d}  dwell={dwell:2d}  "
            f"foodBeliefErr={food_err:5.2f}  finalAgentToFood={agent_to_food:5.2f}  "
            f"finalFoodCovTr={food_cov_tr:.4f}"
        )

    print("\n  Kalman vs ChainBackend agreement (scripted inputs, atol=1e-7):")
    max_mean_diff, max_cov_diff = check_backend_agreement()
    ok = max_mean_diff < 1e-7 and max_cov_diff < 1e-7
    print(
        f"    max |Δmean|={max_mean_diff:.2e}  max |Δcov|={max_cov_diff:.2e}  "
        f"{'OK' if ok else 'MISMATCH'}"
    )
    if not ok:
        raise SystemExit("Kalman/ChainBackend disagree beyond atol=1e-7")


def _ellipse(ax, mean, cov, color, *, alpha_fill=0.16, max_diameter=None):
    """A 2-sigma covariance ellipse, with its on-screen size capped.

    The food prior covariance starts wide (``FOOD_PRIOR_COV``, by design — the
    agent is meant to have to detour to learn it), so its raw 2-sigma diameter at
    step 0 can exceed the whole plot. Drawing that literally floods the panel
    with solid fill and makes early frames look broken; capping the DISPLAYED
    diameter (never the underlying belief, only this patch's size) keeps every
    frame readable while still reading as "wide and uncertain."
    """
    from matplotlib.patches import Ellipse

    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-9, None)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * 2.0 * np.sqrt(vals)
    if max_diameter is not None:
        w, h = min(w, max_diameter), min(h, max_diameter)
    ax.add_patch(
        Ellipse(mean, w, h, angle=angle, facecolor=color, alpha=alpha_fill, zorder=3)
    )
    ax.add_patch(
        Ellipse(
            mean, w, h, angle=angle, facecolor="none", edgecolor=color, lw=1.2, zorder=3
        )
    )


def render(regimes, runs, out_path, *, fps=20):
    """Draw the 2x2 grid, one panel per regime, and write the looping GIF.

    Reuses the flagship demo's palette, ``_draw_bacillus``, and
    ``_precision_field`` rather than re-deriving them — the beacon mechanic is
    visually identical, only what it reveals has changed. Each panel shows BOTH
    belief markers/ellipses (agent + food), where the flagship only ever needed
    one, and its border turns green and bold once that regime first settles near
    the food (see ``ARRIVAL_THRESHOLD``) — a visible "did it get there" signal,
    not just the printed metrics. Each panel also gets its own ``t=`` step
    counter (top-right) that FREEZES the instant that regime arrives, rather
    than continuing to climb with the shared global frame index — so the
    fastest-arriving regime's final on-screen count is directly comparable to
    the others' as a per-regime "how long did this take" readout.
    """
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from bacillus_seeking_food import BEACON as BEACON_COLOR
    from bacillus_seeking_food import BELIEF, BG, BODY, FOOD, GRID, INK, _draw_bacillus
    from bacillus_seeking_food import _precision_field as flagship_precision_field
    from PIL import Image

    allpts = np.concatenate(
        [runs[r["key"]][0] for r in regimes]
        + [runs[r["key"]][1][:, 2:4] for r in regimes]
        + [BEACON_PT[None], FOOD_TRUE[None]]
    )
    pad = 1.3
    xlim = (allpts[:, 0].min() - pad, allpts[:, 0].max() + pad)
    ylim = (allpts[:, 1].min() - pad, allpts[:, 1].max() + pad)

    field_xs, field_ys, field = flagship_precision_field(xlim, ylim, build_model())
    field_levels = np.linspace(field.min(), field.max(), 9)
    n_frames = len(runs[regimes[0]["key"]][0])

    # Cap displayed ellipse size to a fraction of the panel span: the food prior
    # is deliberately wide (ADR-013), so its raw step-0 diameter can exceed the
    # whole plot and flood it with solid fill. Capped at display time only — the
    # underlying belief/covariance used everywhere else is untouched.
    max_ellipse_diameter = 0.6 * min(xlim[1] - xlim[0], ylim[1] - ylim[0])
    arrived_steps = {r["key"]: _arrival_step(runs[r["key"]][0]) for r in regimes}
    arrived_color = "#1B9E50"

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
            step_arrived = arrived_steps[reg["key"]]
            arrived = step_arrived is not None and i >= step_arrived
            for spine in ax.spines.values():
                spine.set_color(arrived_color if arrived else GRID)
                spine.set_linewidth(2.6 if arrived else 0.8)

            ax.contourf(
                field_xs,
                field_ys,
                field,
                levels=field_levels,
                cmap="Blues",
                alpha=0.32,
                zorder=0,
            )
            ax.plot(
                BEACON_PT[0],
                BEACON_PT[1],
                "o",
                color=BEACON_COLOR,
                ms=9,
                markeredgecolor="white",
                markeredgewidth=1.2,
                zorder=4,
            )
            ax.plot(
                FOOD_TRUE[0],
                FOOD_TRUE[1],
                "*",
                color=FOOD,
                ms=20,
                markeredgecolor=INK,
                markeredgewidth=0.7,
                zorder=4,
            )

            if i > 0:
                tr = true_states[: i + 1]
                ax.plot(tr[:, 0], tr[:, 1], color=accent, lw=1.8, alpha=0.6, zorder=2)

            _ellipse(
                ax,
                means[i, :2],
                covs[i, :2, :2],
                BELIEF,
                max_diameter=max_ellipse_diameter,
            )  # agent belief
            _ellipse(
                ax,
                means[i, 2:4],
                covs[i, 2:, 2:],
                FOOD,
                max_diameter=max_ellipse_diameter,
            )  # food belief

            ax.plot(*means[i, :2], "+", color=BELIEF, ms=10, mew=2.0, zorder=8)
            ax.plot(*means[i, 2:4], "D", color=FOOD, ms=6, mec=INK, mew=0.6, zorder=8)

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
            label = "LQR" if reg["kind"] == "lqr" else f"Λ = {reg['precision']:g}"
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
            panel_step = step_arrived if arrived else i
            ax.text(
                0.972,
                0.962,
                f"t={panel_step:>2d}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                color=arrived_color if arrived else "#7A7A7A",
                fontsize=10,
                fontweight="bold" if arrived else "normal",
                family="monospace",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "fc": "white",
                    "ec": arrived_color if arrived else GRID,
                    "alpha": 0.9,
                },
            )

        fig.suptitle(
            "instrumental epistemics: the beacon resolves food location",
            color=INK,
            fontsize=12.5,
            fontweight="bold",
            y=0.972,
        )
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
                label="agent belief",
            ),
            plt.Line2D(
                [],
                [],
                marker="D",
                color=FOOD,
                ls="none",
                ms=7,
                mec=INK,
                label="food belief",
            ),
            plt.Line2D(
                [],
                [],
                marker="o",
                color=BEACON_COLOR,
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
                label="food (truth, hidden)",
            ),
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=5,
            fontsize=8.2,
            framealpha=0.9,
            edgecolor=GRID,
            labelcolor=INK,
            columnspacing=1.3,
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
            "cpomdp v0.4",
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
    """Entry point: ``--scan`` for metrics, otherwise render the 2x2 GIF."""
    if "--scan" in sys.argv:
        scan()
        return
    out = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("docs/assets/bacillus_uncertain_food.gif")
    )
    runs = {r["key"]: simulate(r) for r in REGIMES}
    path = render(REGIMES, runs, out)
    for r in REGIMES:
        ts, means, covs = runs[r["key"]]
        d_beacon, _step_min, dwell, food_err, agent_to_food, _tr = _metrics(
            ts, means, covs
        )
        tag = "LQR" if r["kind"] == "lqr" else f"Λ={r['precision']:g}"
        print(
            f"  {tag:>9}: minBeacon={d_beacon:4.2f}  dwell={dwell:2d}  "
            f"foodErr={food_err:4.2f}  agentToFood={agent_to_food:4.2f}"
        )
    print(f"wrote {path}  ({len(runs[REGIMES[0]['key']][0])} steps × 4 panels)")


if __name__ == "__main__":
    main()
