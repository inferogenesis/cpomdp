"""Spike demonstrator: read the EFE information rate off cpomdp v0.3.

This is the *minimal honest experiment* the RFC-001 chapter-8 ("Mattingly curve")
roadmap says v0.3 can run today. It does ONE real thing and refuses to do the rest:

  REAL: drive a state-dependent-sensing agent toward a target observation, harvest
        the EFE epistemic split (state information gain I(state;obs), in nats) off
        the *actual* kernel each cycle, convert to bits, and divide by the cycle
        time to get a bits/second rate -- the same units Mattingly reports.

  REFUSED (printed, not faked): placing this point against Mattingly's
        v_d <= X*(Idot/beta)^(1/2) bound. That needs four things v0.3 does NOT have
        (log/MWC channel, the behavioural signal->action rate, the physical beta/X
        constants, and an arena calibrated in um/s). See SPIKE.md.

The number this prints is the PERCEPTUAL CEILING Idot (signal->belief): an UPPER
BOUND on Mattingly's signal->action rate, on a linearised cartoon of the cell's
sensing, in arbitrary length units. It is deliberately NOT Mattingly-comparable.
The whole point of the spike is to show exactly where the honest wall is.

Run:  .venv/bin/python spikes/chapter8_information_rate/demo_information_rate.py
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from cpomdp.agent import Agent
from cpomdp.efe import expected_free_energy
from cpomdp.observation import CallableSensor
from cpomdp.selection import ObservationGoal, Preference
from cpomdp.types import Belief, LinearGaussianModel

# --- arena ("chemotaxis cartoon", arbitrary length units) ----------------------
# 1-D position x along a gradient. The source is at x = 0. A single integrator
# (mu+ = mu + dt*a) so the action moves the *observed* position this step -- the
# one-step observability the H=1 EFE kernel needs (DECISIONS.md ADR-008).
DT_SECONDS = 0.1  # cycle time; sensing rate = 1/DT observations per second
N_STEPS = 80
SOURCE = 0.0
START = 6.0  # start position (units), agent must climb to the source
ACTION_BOUND = 4.0  # |run speed| cap (units/s)
N_CANDIDATES = 81  # EFE action grid resolution (attributable per-cycle cost)
PROCESS_STD = 0.05  # true-plant diffusion per step (units)

# State-dependent sensing R(x): a *precision well* at the source. Far away the
# sensor is noisy (variance R_FAR); near the source it sharpens to R_NEAR. This is
# the v0.3 seam (CallableSensor, constant C, linear mean) that keeps the epistemic
# term alive and action-dependent -- without it I(state;obs) is constant and EFE
# collapses to LQR (ADR-003), so there is no information rate to read.
R_FAR = 1.5
R_NEAR = 0.02
WELL_WIDTH = 2.5


def well_noise(x, params):
    """R(x) = R_near + (R_far - R_near) * (1 - exp(-(x-c)^2 / 2w^2)); always PD."""
    c, w, r_far, r_near = params["c"], params["w"], params["r_far"], params["r_near"]
    gap = r_far - r_near
    var = r_near + gap * (1.0 - jnp.exp(-((x[0] - c) ** 2) / (2.0 * w**2)))
    return jnp.reshape(var, (1, 1))


def build_agent():
    """The chemotaxis-cartoon agent: single integrator + precision-well R(x)."""
    A = jnp.array([[1.0]])
    B = jnp.array([[DT_SECONDS]])  # single integrator
    Q = jnp.array([[PROCESS_STD**2]])  # model process noise (fixed)
    C = jnp.array([[1.0]])  # observe position directly (LINEAR proxy)
    sensor = CallableSensor(
        C,
        well_noise,
        {"c": SOURCE, "w": WELL_WIDTH, "r_far": R_FAR, "r_near": R_NEAR},
    )
    prior = Belief(jnp.array([START]), jnp.array([[1.0]]))
    model = LinearGaussianModel(
        A, C, Q, jnp.array([[R_FAR]]), prior, control=B, observation=sensor
    )
    # Observation goal: prefer to *observe* the source. precision = explore/exploit
    # knob (ADR-008): modest here so the epistemic term genuinely participates.
    goal = ObservationGoal(
        target=jnp.array([SOURCE]),
        action_bounds=(-ACTION_BOUND, ACTION_BOUND),
        precision=jnp.array([[1.0]]),
        n_candidates=N_CANDIDATES,
        horizon=1,
    )
    return Agent(model, goal), model, goal


def epistemic_bits_at(model, belief, action, goal) -> float:
    """The EFE epistemic split at the *chosen* action, converted nats -> bits.

    epistemic = (1/2)(ln det S - ln det R) = I(state;obs) for the linear-Gaussian
    case (efe.py:288). It is computed in NATS (natural log via slogdet); bits = /ln2.
    """
    pref = Preference(goal.target, goal.precision)
    _, split = expected_free_energy(model, belief, action, pref)
    return float(split["epistemic"]) / float(jnp.log(2.0))


def phase0_oracle() -> None:
    """Gate the bit-counting before trusting any rate.

    The kernel's epistemic term must equal the mutual information I(x;o), computed an
    INDEPENDENT way, to machine precision -- before any biology mapping. The kernel uses
    the observation-side form  I = (1/2) ln(det S / det R). I cross-check it against the
    state-side form  I = (1/2) ln(det Sigma+ / det Sigma_post), where
    Sigma_post = Sigma+ - Sigma+ C^T S^-1 C Sigma+ is the Kalman-updated covariance.
    Both are the same MI; agreement confirms the bit-counting, not the biology.
    """
    rng = np.random.default_rng(7)
    for trial, m_dim, n_dim in [("C=I", 2, 2), ("general C", 1, 3)]:
        A = jnp.asarray(rng.normal(size=(n_dim, n_dim)) * 0.3 + jnp.eye(n_dim))
        sig = jnp.asarray(rng.normal(size=(n_dim, n_dim)))
        sigma = sig @ sig.T + jnp.eye(n_dim)  # PD prior cov
        Q = 0.1 * jnp.eye(n_dim)
        C = (
            jnp.eye(n_dim)
            if trial == "C=I"
            else jnp.asarray(rng.normal(size=(m_dim, n_dim)))
        )
        Rn = rng.normal(size=(m_dim, m_dim))
        R = jnp.asarray(Rn @ Rn.T + 0.5 * jnp.eye(m_dim))  # PD noise
        B = jnp.ones((n_dim, 1))
        prior = Belief(jnp.zeros(n_dim), sigma)
        model = LinearGaussianModel(A, C, Q, R, prior, control=B)
        action = jnp.array([0.4])
        pref = Preference(jnp.zeros(C.shape[0]), jnp.eye(C.shape[0]))
        _, split = expected_free_energy(model, prior, action, pref)
        kernel = float(split["epistemic"])
        # independent state-side MI
        sigma_pred = A @ sigma @ A.T + Q
        s = C @ sigma_pred @ C.T + R
        p_xo = sigma_pred @ C.T
        sigma_post = sigma_pred - p_xo @ jnp.linalg.solve(s, p_xo.T)
        oracle = 0.5 * (
            jnp.linalg.slogdet(sigma_pred)[1] - jnp.linalg.slogdet(sigma_post)[1]
        )
        err = abs(kernel - float(oracle))
        assert err < 1e-9, f"Phase-0 MI oracle FAILED ({trial}): err={err}"
        print(
            f"  [phase-0 oracle] {trial:>9}: kernel I = {kernel:.6f} nats "
            f"== state-side MI (err {err:.1e}) OK"
        )


def run():
    """Gate on the MI oracle, then run the loop and print the ceiling rate + caveats."""
    print("=" * 74)
    print("Phase-0 gate: verify the bit-counting before trusting any rate")
    print("=" * 74)
    phase0_oracle()
    print()

    agent, model, goal = build_agent()
    rng = np.random.default_rng(0)

    true_x = float(START)
    bits_per_obs: list[float] = []
    true_path = [true_x]

    for _ in range(N_STEPS):
        # perceive: real sensor reading from the true state, noise = R(true_x)
        r_true = float(
            well_noise(
                jnp.array([true_x]),
                {"c": SOURCE, "w": WELL_WIDTH, "r_far": R_FAR, "r_near": R_NEAR},
            )[0, 0]
        )
        obs = true_x + rng.normal() * np.sqrt(r_true)
        belief = agent.infer_states(jnp.array([obs]))

        # act: EFE-minimising action over the front-loaded grid (the real selector)
        action = agent.sample_action()

        # harvest the information gain at the action actually taken
        bits_per_obs.append(epistemic_bits_at(model, belief, action, goal))

        # advance the true plant under that action + diffusion
        true_x = true_x + DT_SECONDS * float(action[0]) + rng.normal() * PROCESS_STD
        true_path.append(true_x)

    # --- the two silicon numbers, in Mattingly's units ------------------------
    sensing_rate_hz = 1.0 / DT_SECONDS
    mean_bits = float(np.mean(bits_per_obs))
    idot_ceiling = mean_bits * sensing_rate_hz  # bits / second
    net_climb = abs(true_path[0] - SOURCE) - abs(true_path[-1] - SOURCE)
    total_time = N_STEPS * DT_SECONDS
    v_drift = net_climb / total_time  # units / second

    print("=" * 74)
    print("cpomdp v0.3 -- EFE information rate (RFC-001 ch.8 minimal honest spike)")
    print("=" * 74)
    print(f"  cycles               : {N_STEPS} @ dt={DT_SECONDS}s -> {total_time:.1f}s")
    print(f"  sensing rate          : {sensing_rate_hz:.1f} obs/s")
    print(f"  mean info gain        : {mean_bits:.4f} bits / observation")
    print(f"  net climb to source   : {net_climb:.3f} units (start {true_path[0]:.2f})")
    print()
    print("  SILICON POINT (read off the real EFE kernel):")
    print(f"    I_acquired_ceiling  = {idot_ceiling:.3f} bits/s  <-- CEILING, not Idot")
    print(f"    v_drift             = {v_drift:.3f} units/s")
    print()
    print("-" * 74)
    print("  HONEST CAVEATS (why this is NOT yet a point on Mattingly's curve):")
    print("  1. CEILING, not behavioural rate. This is I(state;obs), signal->belief.")
    print("     Mattingly's Idot is signal->action -- strictly smaller. This number")
    print("     is an UPPER BOUND on the comparable rate (over-counts the bits).")
    print("  2. LINEAR channel. The agent observes position directly. E. coli senses")
    print("     log-concentration / fold-change (MWC). v0.3 has no nonlinear sensor,")
    print("     so this is a linearised cartoon, not the cell's sensing physics.")
    print("  3. No beta / X / v0 calibration, arena in arbitrary units -- I refuse to")
    print("     draw the v_d <= X*(Idot/beta)^(1/2) bound from fabricated constants.")
    print("  4. Internal-noise route (Q(x)) not exercised; loss lives in R here,")
    print("     which is the input-limited picture the 2025 follow-up overturned.")
    print("-" * 74)
    print("  => v0.3 can MEASURE a bits/s rate honestly; it cannot yet place it on")
    print("     the curve. See SPIKE.md for the capability ladder that closes 1-4.")
    print("=" * 74)


if __name__ == "__main__":
    run()
