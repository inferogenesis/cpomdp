"""Builders more than one test module shares.

A helper moves here once a second module needs it verbatim. A helper only one module
uses stays in that module.
"""

import jax.numpy as jnp
import numpy as np

from cpomdp import Belief, CouplingGraph
from cpomdp.control import lqr_schedule
from cpomdp.types import LinearGaussianModel


def unit_prior(graph: CouplingGraph) -> Belief:
    """N(0, I) over a graph's joint state."""
    n = sum(graph.dims)
    return Belief(jnp.zeros(n), jnp.eye(n))


# --- the double-integrator point mass the control and bracket tests share ---------
#
# state = [position, velocity], a force moves the velocity, velocity moves the
# position. dt small enough to be well-conditioned.
# This is both a controllable system (so a steady-state gain exists) and exactly
# the plant the 2-D reaching demo will use, so the oracle here guards the demo too.
DT = 0.1
DYNAMICS = [[1.0, DT], [0.0, 1.0]]
CONTROL = [[0.0], [DT]]
GOAL_PRECISION = [[1.0, 0.0], [0.0, 1.0]]
EFFORT_PENALTY = [[0.1]]
PROCESS_NOISE = np.array([[1e-4, 0.0], [0.0, 1e-4]])
OBSERVATION = np.array([[1.0, 0.0]])
OBSERVATION_NOISE = np.array([[1e-2]])

# The oracles below keep the terse letters a/b/c of the textbook recursions.


def point_mass_model():
    # The noise/sensor fields are required to build a model but don't enter the
    # LQR solve at all — control selection reads only dynamics + control.
    return LinearGaussianModel(
        dynamics_matrix=DYNAMICS,
        observation_matrix=[[1.0, 0.0]],  # observe position
        dynamics_noise=[[1e-4, 0.0], [0.0, 1e-4]],
        observation_noise=[[1e-2]],
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        control_matrix=CONTROL,
    )


def point_mass_schedule(horizon):
    return lqr_schedule(
        point_mass_model(),
        horizon=horizon,
        precision=GOAL_PRECISION,
        effort=EFFORT_PENALTY,
    )


def brute_force_plan(dynamics, control, goal_precision, effort_penalty, horizon, x0):
    """The H-step open-loop optimum as a stacked least-squares problem.

    Stacks x_1..x_H = Φ·x0 + Γ·U with a cost on every arrived-at state and every
    action, and solves the normal equations for U. No Riccati recursion anywhere,
    so agreement with the schedule is evidence about the schedule.
    """
    a = np.asarray(dynamics, dtype=float)
    b = np.asarray(control, dtype=float)
    n, p = b.shape
    powers = [np.linalg.matrix_power(a, k) for k in range(horizon + 1)]
    phi = np.vstack([powers[k] for k in range(1, horizon + 1)])
    gamma = np.zeros((horizon * n, horizon * p))
    for i in range(horizon):
        for j in range(i + 1):
            gamma[i * n : (i + 1) * n, j * p : (j + 1) * p] = powers[i - j] @ b
    stage = np.kron(np.eye(horizon), np.asarray(goal_precision, dtype=float))
    effort = np.kron(np.eye(horizon), np.asarray(effort_penalty, dtype=float))
    hessian = gamma.T @ stage @ gamma + effort
    linear = gamma.T @ stage @ phi @ x0
    actions = -np.linalg.solve(hessian, linear)
    cost = x0 @ phi.T @ stage @ phi @ x0 - linear @ np.linalg.solve(hessian, linear)
    return actions[:p], cost


def numpy_kalman_covariances(prior_cov, horizon):
    """The textbook covariance recursion, kept apart from the library's kernel."""
    a, c = np.asarray(DYNAMICS), OBSERVATION
    gains, covs = [], [np.asarray(prior_cov, dtype=float)]
    for _ in range(horizon):
        pred = a @ covs[-1] @ a.T + PROCESS_NOISE
        innovation = c @ pred @ c.T + OBSERVATION_NOISE
        gain = pred @ c.T @ np.linalg.inv(innovation)
        gains.append(gain)
        covs.append((np.eye(2) - gain @ c) @ pred)
    return np.stack(gains), np.stack(covs)
