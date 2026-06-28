"""The beacon reveals the FOOD's position, not the agent's own (ADR-013).

`bacillus_seeking_food.py` has the beacon collapse uncertainty about the agent's
*own* position — a trivial form of state information gain (visiting a precise
state for its own sake), not tied to any genuine unknown. This demo promotes the
food's position to an explicit latent the agent does not know a priori, and wires
the beacon's existing precision-well mechanic to reveal *that* instead — closer to
the discrete T-Maze task's shape (visiting a cue resolves a real contextual
unknown) than the original demo's beacon was.

State is now 4-D: ``[agent_xy, food_xy]``. The sensor still has the agent read its
own position (fixed precision — plain proprioception), but gains a second,
*relative* channel: ``o_disp = food_xy - agent_xy``, whose noise is the existing,
**unmodified** ``beacon_noise`` falloff from the flagship demo, evaluated at the
agent's own position. Visiting the beacon does not sharpen "where am I" anymore —
it sharpens "where is the food," which the agent cannot directly act on.

The ``Preference`` stays a single static object, weighted only on the displacement
channel with target ``[0, 0]`` ("observe zero distance from food"). Because the
predicted reading is ``E[food_xy]⁺ - agent_xy⁺``, this one static target
algebraically chases the *current belief* of food's location — no per-step
preference rebuilding (ADR-013 spells out why this beats that alternative, and
names the open multi-goal question it leaves unresolved).

This needs zero core-library changes (ADR-013): the model is just a bigger
``LinearGaussianModel``, the sensor is one ``CallableSensor`` with a 4x4
block-diagonal ``R(x)``, and ``expected_free_energy`` doesn't care how the
observation vector is built up. The mechanism is checked, not just rendered:
``--scan`` runs the identical model through both ``KalmanBackend`` and
``ChainBackend`` (ADR-012 Phase 2.5) and checks they agree to ``atol=1e-7``.

Needs the ``examples`` extra (matplotlib + pillow, neither a runtime dependency of
the library itself)::

    uv run --extra examples python examples/bacillus_uncertain_food.py --scan

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

LAMBDA_DISP = 0.015  # pragmatic weight on the displacement channel ("go to food")
# tuned by sweep: below ~0.03 the agent never detours, slowly averaging the murk
# instead (the "classic LQR" regime); 0.015 is the balanced detour-then-exploit case.

ACTION_LO, ACTION_HI = -2.4, 2.4
GRID_N = 25
N_STEPS = 90


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


def build_preference() -> Preference:
    """A static obs-space preference: zero weight on self, weight on "find food".

    Target is ``[0, 0, 0, 0]`` — irrelevant for the self block (its precision
    weight is 0) and "observe zero displacement from food" for the disp block.
    Because the predicted disp reading is ``E[food]⁺ - agent⁺``, this single
    static object chases whatever the agent currently believes about food's
    location — no per-step rebuild (ADR-013).
    """
    precision = jax.scipy.linalg.block_diag(jnp.zeros((2, 2)), LAMBDA_DISP * jnp.eye(2))
    return Preference(goal=jnp.zeros(4), precision=precision)


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


def simulate(backend_cls, *, seed=7):
    """Run one closed perceive -> act loop under the given backend class.

    The food never moves (``FOOD_TRUE`` is the simulator's ground truth); the
    agent does, under whichever action the EFE grid argmin picks each step. The
    simulated reading is sampled from the SAME noise the model uses to filter —
    fixed ``R_SELF`` for the self channel, the unmodified ``beacon_noise`` falloff
    (evaluated at the agent's TRUE position) for the disp channel.

    Args:
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
    preference = build_preference()
    candidates = _candidate_grid()
    candidates_np = np.asarray(candidates)
    beacon_params = _beacon_params()

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
        g = _efe_grid(model, belief, preference, candidates)
        action = candidates_np[int(np.argmin(np.asarray(g)))]
        last_action = jnp.asarray(action)

        true_states.append(true_agent.copy())
        means.append(np.asarray(belief.mean))
        covs.append(np.asarray(belief.cov))

        true_agent = a_mat @ true_agent + b_mat @ action

    return np.array(true_states), np.array(means), np.array(covs)


def _metrics(true_states, means, covs):
    """Behaviour summary: closest beacon approach, final food-belief error, etc."""
    d_to_beacon = np.linalg.norm(true_states - BEACON_PT, axis=1)
    d_beacon = d_to_beacon.min()
    step_min = int(d_to_beacon.argmin())
    food_belief_err = np.linalg.norm(means[-1][2:4] - FOOD_TRUE)
    agent_to_food = np.linalg.norm(true_states[-1] - FOOD_TRUE)
    food_cov_trace = float(np.trace(covs[-1][2:, 2:]))
    return d_beacon, step_min, food_belief_err, agent_to_food, food_cov_trace


def check_backend_agreement(*, seed=11, n_steps=30):
    """``KalmanBackend`` vs ``ChainBackend`` on an IDENTICAL scripted input sequence.

    Deliberately not two independent closed EFE loops: each loop's argmin could
    in principle land on a different grid cell from a near-tied score under tiny
    numerical differences, and the resulting trajectories would then diverge for
    a reason that has nothing to do with the backends actually disagreeing.
    Feeding both backends the same scripted ``(observation, action)`` pairs
    isolates exactly the claim ADR-013/BUILD_PLAN.md Phase 3 cares about: do the
    two backends compute the same belief from the same inputs, on a sensor
    topology (a channel reading one state block, noise keyed on a different
    block) neither backend's existing test suite exercises. Mirrors the
    methodology in ``tests/test_ffg_chain.py``.

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
    """Print behaviour metrics for both backends, then the agreement check."""
    print(
        f"world: start={START}  food_true={FOOD_TRUE}  "
        f"food_prior_mean={FOOD_PRIOR_MEAN}  beacon={BEACON_PT}"
    )
    print(f"  Λ_disp={LAMBDA_DISP}  N_STEPS={N_STEPS}\n")

    backends = (("KalmanBackend", KalmanBackend), ("ChainBackend", ChainBackend))
    for name, backend_cls in backends:
        ts, means, covs = simulate(backend_cls)
        d_beacon, step_min, food_err, agent_to_food, food_cov_tr = _metrics(
            ts, means, covs
        )
        print(
            f"  {name:>13}: minBeacon={d_beacon:5.2f}@{step_min:2d}  "
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


def render(true_states, means, covs, out_path, *, fps=20):
    """Render one bacillus's run: true path, both belief means, both 2σ ellipses.

    Reuses the flagship demo's palette, ``_draw_bacillus``, and
    ``_precision_field`` rather than re-deriving them — the beacon mechanic is
    visually identical, only what it reveals has changed.
    """
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from bacillus_seeking_food import BELIEF, BODY, FOOD, GRID, INK, _draw_bacillus
    from bacillus_seeking_food import _precision_field as flagship_precision_field
    from matplotlib.patches import Ellipse
    from PIL import Image

    pad = 1.4
    allpts = np.concatenate([true_states, means[:, :2], means[:, 2:4]])
    xlim = (allpts[:, 0].min() - pad, allpts[:, 0].max() + pad)
    ylim = (allpts[:, 1].min() - pad, allpts[:, 1].max() + pad)

    field_xs, field_ys, field = flagship_precision_field(xlim, ylim, build_model())
    field_levels = np.linspace(field.min(), field.max(), 9)

    def _ellipse(ax, mean, cov, color):
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 1e-9, None)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        w, h = 2 * 2.0 * np.sqrt(vals)
        ax.add_patch(
            Ellipse(mean, w, h, angle=angle, facecolor=color, alpha=0.16, zorder=3)
        )
        ax.add_patch(
            Ellipse(
                mean,
                w,
                h,
                angle=angle,
                facecolor="none",
                edgecolor=color,
                lw=1.2,
                zorder=3,
            )
        )

    frames = []
    for i in range(len(true_states)):
        fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=110)
        fig.patch.set_facecolor("#FAFAFA")
        ax.set_facecolor("#FAFAFA")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRID)

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
            color="#0072B2",
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
            ax.plot(tr[:, 0], tr[:, 1], color=BODY, lw=1.8, alpha=0.6, zorder=2)

        _ellipse(ax, means[i, :2], covs[i, :2, :2], BELIEF)  # agent belief
        _ellipse(ax, means[i, 2:4], covs[i, 2:, 2:], FOOD)  # food belief

        ax.plot(*means[i, :2], "+", color=BELIEF, ms=10, mew=2.0, zorder=8)
        ax.plot(*means[i, 2:4], "D", color=FOOD, ms=7, mec=INK, mew=0.6, zorder=8)

        pos = true_states[i].astype(float)
        j = max(1, i)
        heading = true_states[j] - true_states[j - 1]
        _draw_bacillus(ax, pos, np.asarray(heading), phase=i * 0.9)

        ax.set_title(
            "the beacon reveals the FOOD's position (ADR-013)",
            color=INK,
            fontsize=10.5,
            fontweight="bold",
            pad=8,
        )
        fig.text(
            0.98,
            0.012,
            f"step {i:>2d}/{len(true_states) - 1}",
            ha="right",
            va="bottom",
            color="#9A9A9A",
            fontsize=8,
            family="monospace",
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
    """Entry point: ``--scan`` for metrics, otherwise render the GIF."""
    if "--scan" in sys.argv:
        scan()
        return
    out = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("docs/assets/bacillus_uncertain_food.gif")
    )
    true_states, means, covs = simulate(KalmanBackend)
    path = render(true_states, means, covs, out)
    d_beacon, step_min, food_err, agent_to_food, food_cov_tr = _metrics(
        true_states, means, covs
    )
    print(
        f"minBeacon={d_beacon:.2f}@{step_min}  foodBeliefErr={food_err:.2f}  "
        f"finalAgentToFood={agent_to_food:.2f}  finalFoodCovTr={food_cov_tr:.4f}"
    )
    print(f"wrote {path}  ({len(true_states)} steps)")


if __name__ == "__main__":
    main()
