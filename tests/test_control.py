import numpy as np
import pytest
import scipy.linalg

from cpomdp.control import LQRController
from cpomdp.types import Belief, LinearGaussianModel

# Inside this module the terse letters a/b/qc/rc are local scalars for the
# Riccati/gain hand-math below, NOT the role-named public API
# (dynamics/control/state_cost/control_cost). The library deliberately spells
# those out to avoid the Q/R collision (ADR-003); here, where we're transcribing
# the textbook DARE formula to check it line-for-line, the letters keep the
# matrix algebra readable and carry no API meaning.

# A double-integrator point mass: state = [position, velocity], a force moves the
# velocity, velocity moves the position. dt small enough to be well-conditioned.
# This is both a controllable system (so a steady-state gain exists) and exactly
# the plant the 2-D reaching demo will use, so the oracle here guards the demo too.
DT = 0.1
DYNAMICS = [[1.0, DT], [0.0, 1.0]]
CONTROL = [[0.0], [DT]]
STATE_COST = [[1.0, 0.0], [0.0, 1.0]]
CONTROL_COST = [[0.1]]


def _point_mass_model():
    # The noise/sensor fields are required to build a model but don't enter the
    # LQR solve at all — control selection reads only dynamics + control.
    return LinearGaussianModel(
        dynamics=DYNAMICS,
        sensor_model=[[1.0, 0.0]],  # observe position
        dynamics_noise=[[1e-4, 0.0], [0.0, 1e-4]],
        sensor_noise=[[1e-2]],
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        control=CONTROL,
    )


def _scipy_gain(dynamics, control, state_cost, control_cost):
    """L∞ via scipy's Schur-based DARE solver — the independent oracle.

    scipy returns the cost-to-go P, not the gain, so we derive the gain with the
    same closing formula LQRController uses: L∞ = (Rc + BᵀPB)⁻¹(BᵀPA). Because
    scipy reaches P by Schur decomposition rather than value iteration, a
    transpose or orientation bug in our loop can't survive in both.
    """
    a = np.asarray(dynamics, dtype=float)
    b = np.asarray(control, dtype=float)
    qc = np.asarray(state_cost, dtype=float)
    rc = np.asarray(control_cost, dtype=float)
    p = scipy.linalg.solve_discrete_are(a, b, qc, rc)
    return np.linalg.solve(rc + b.T @ p @ b, b.T @ p @ a)


class TestLQRGain:
    def test_gain_matches_scipy_dare(self):
        # The core oracle: our hand-rolled fixed-point iteration must agree with
        # scipy's independent Schur solve. Disagreement = the bug is ours.
        controller = LQRController(
            _point_mass_model(), state_cost=STATE_COST, control_cost=CONTROL_COST
        )
        expected = _scipy_gain(DYNAMICS, CONTROL, STATE_COST, CONTROL_COST)
        np.testing.assert_allclose(controller.gain, expected, atol=1e-8)

    def test_gain_has_shape_p_by_n(self):
        controller = LQRController(
            _point_mass_model(), state_cost=STATE_COST, control_cost=CONTROL_COST
        )
        # one action, two states
        assert controller.gain.shape == (1, 2)

    def test_closed_loop_is_stable(self):
        # The real test that the gain is right in sign AND magnitude: the
        # closed-loop dynamics (A - B·L∞) must be stable, i.e. every eigenvalue
        # strictly inside the unit circle. A sign-flipped gain would push the
        # eigenvalues out and this would fail loudly.
        controller = LQRController(
            _point_mass_model(), state_cost=STATE_COST, control_cost=CONTROL_COST
        )
        a = np.asarray(DYNAMICS)
        b = np.asarray(CONTROL)
        closed_loop = a - b @ controller.gain
        eigvals = np.linalg.eigvals(closed_loop)
        assert np.all(np.abs(eigvals) < 1.0)


class TestLQRAction:
    def test_action_pushes_toward_goal(self):
        # From the origin with a target at position +1, the force must be
        # positive (accelerate toward the goal). A dropped minus sign flips this.
        controller = LQRController(
            _point_mass_model(), state_cost=STATE_COST, control_cost=CONTROL_COST
        )
        action = controller.action(mean=np.array([0.0, 0.0]), goal=np.array([1.0, 0.0]))
        assert action[0] > 0

    def test_zero_error_gives_zero_action(self):
        # Sitting exactly on an equilibrium goal, the controller asks for nothing.
        controller = LQRController(
            _point_mass_model(), state_cost=STATE_COST, control_cost=CONTROL_COST
        )
        action = controller.action(mean=np.array([1.0, 0.0]), goal=np.array([1.0, 0.0]))
        np.testing.assert_allclose(action, [0.0], atol=1e-12)

    def test_rejects_wrong_shape_goal(self):
        controller = LQRController(
            _point_mass_model(), state_cost=STATE_COST, control_cost=CONTROL_COST
        )
        with pytest.raises(ValueError, match="goal"):
            controller.action(mean=np.array([0.0, 0.0]), goal=np.array([1.0, 0.0, 0.0]))


class TestLQRValidation:
    def test_rejects_model_without_control(self):
        model = LinearGaussianModel(
            dynamics=DYNAMICS,
            sensor_model=[[1.0, 0.0]],
            dynamics_noise=[[1e-4, 0.0], [0.0, 1e-4]],
            sensor_noise=[[1e-2]],
            prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        )  # no control matrix
        with pytest.raises(ValueError, match="control matrix"):
            LQRController(model, state_cost=STATE_COST, control_cost=CONTROL_COST)

    def test_rejects_wrong_state_cost_shape(self):
        with pytest.raises(ValueError, match="state_cost"):
            LQRController(
                _point_mass_model(), state_cost=[[1.0]], control_cost=CONTROL_COST
            )

    def test_rejects_wrong_control_cost_shape(self):
        with pytest.raises(ValueError, match="control_cost"):
            LQRController(
                _point_mass_model(),
                state_cost=STATE_COST,
                control_cost=[[1.0, 0.0], [0.0, 1.0]],
            )

    def test_rejects_asymmetric_state_cost(self):
        # An off-diagonal typo: symmetric on shape, asymmetric in value. Without
        # the check this silently yields a non-symmetric cost-to-go and a wrong
        # gain — the hardest failure to trace in a control loop.
        with pytest.raises(ValueError, match="symmetric"):
            LQRController(
                _point_mass_model(),
                state_cost=[[1.0, 0.5], [-0.5, 1.0]],
                control_cost=CONTROL_COST,
            )

    def test_rejects_indefinite_control_cost(self):
        # control_cost is inverted against in the gain solve; a zero (singular)
        # cost must fail loudly, not blow up mid-recursion.
        with pytest.raises(ValueError, match="positive-definite"):
            LQRController(
                _point_mass_model(), state_cost=STATE_COST, control_cost=[[0.0]]
            )

    def test_rejects_negative_semidefinite_state_cost(self):
        with pytest.raises(ValueError, match="positive-semi-definite"):
            LQRController(
                _point_mass_model(),
                state_cost=[[-1.0, 0.0], [0.0, 1.0]],
                control_cost=CONTROL_COST,
            )

    def test_raises_when_not_converged(self):
        # max_iter too small to reach the fixed point -> a loud failure, not a
        # silently-wrong frozen gain. Mirrors the Kalman steady-state guard.
        with pytest.raises(RuntimeError, match="converge"):
            LQRController(
                _point_mass_model(),
                state_cost=STATE_COST,
                control_cost=CONTROL_COST,
                max_iter=1,
            )
