"""The rollout's covariance trajectory depends on the policy only under state-dependent
noise — a witness read from `policy_efe_trace`.

Under a fixed sensor the open-loop covariance recursion — predict `Σ⁺ = AΣAᵀ + Q`,
innovation `S = CΣ⁺Cᵀ + R`, contract `Σ_post = Σ⁺ − Σ⁺Cᵀ S⁻¹ C Σ⁺` — reads none of the
mean, so the action moves only the mean and two different policies carry a **byte-
identical** `Σ` trajectory. Make `R` or `Q` state-dependent and the noise, hence the
whole covariance trajectory, follows where the action puts the mean: the two policies
now separate.

The value is an exhibit. It forecloses "your planner just carries one covariance
trajectory like everyone else's" — under `R(x)`/`Q(x)` the planner manipulates the
open-loop planning covariances themselves, at horizon > 1, checkable with no grid. It
also prices the cost driver in the same run: the `Σ` columns compared here are the
`H × n × n` arrays whose stacking is what a naive rollout would pay for.

The locks:

- `TestFixedSensorSigmaPolicyIndependent`: two distinct policies give byte-identical
  (`assert_array_equal`) `Σ⁺`, `Σ_post` and `S` trajectories, while their means
  genuinely differ (so the covariance agreement is not because the policies coincide).
- `TestStateDependentNoiseSigmaSeparates`: the same two policies separate the
  covariance trajectory by more than a declared margin — `R(x)` through `Σ_post`,
  `Q(x)` through `Σ⁺`.
"""

import jax.numpy as jnp
import numpy as np

from cpomdp.dynamics import CallableProcessNoise
from cpomdp.efe import policy_efe_trace
from cpomdp.observation import CallableSensor
from cpomdp.selection import Preference
from cpomdp.types import Belief, LinearGaussianModel

# The R(x)/Q(x) covariance trajectories separate by ~0.06–0.34 across these two
# policies; the fixed-sensor separation is exactly 0. This margin sits well below the
# real separation and ~13 orders above float64 noise. See warrant_numbers.md.
SEPARATION_MARGIN = 1e-2


# --- fixtures: one model per branch the rollout exercises (mirror test_policy_efe) ---
def _model(observation_model=None):
    return LinearGaussianModel(
        dynamics_matrix=[[1.0, 0.1], [0.0, 1.0]],
        observation_matrix=[[1.0, 0.0]],
        dynamics_noise=[[0.1, 0.0], [0.0, 0.1]],
        observation_noise=[[0.5]],
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        control_matrix=[[0.0], [1.0]],
        observation_model=observation_model,
    )


def _belief():
    return Belief(mean=[0.3, -0.2], cov=[[0.7, 0.1], [0.1, 0.4]])


def _obs_preference():
    return Preference(goal=[1.0], precision=[[2.0]])


def _state_noise(x, params):
    return jnp.array([[params["base"] + params["slope"] * x[1] ** 2]])


def _callable_model():
    sensor = CallableSensor(
        observation_matrix=[[1.0, 0.0]],
        noise_fn=_state_noise,
        noise_params={"base": jnp.array(0.2), "slope": jnp.array(0.5)},
    )
    return _model(observation_model=sensor)


def _q_well(x, params):
    return jnp.array([[params["base"] + params["slope"] * x[0] ** 2]])


def _internal_q_model():
    pn = CallableProcessNoise(
        q_fn=_q_well, q_params={"base": jnp.array(0.05), "slope": jnp.array(0.4)}
    )
    return LinearGaussianModel(
        dynamics_matrix=[[1.0]],
        observation_matrix=[[1.0]],
        dynamics_noise=[[0.1]],
        observation_noise=[[0.3]],
        prior=Belief(mean=[0.0], cov=[[0.2]]),
        control_matrix=[[1.0]],
        dynamics_noise_model=pn,
    )


def _policy(values):
    return jnp.array([[v] for v in values])


# Two genuinely different H=3 policies — the witness compares their Σ trajectories.
_POLICY_A = _policy([0.4, -0.2, 0.1])
_POLICY_B = _policy([-1.0, 0.5, -0.3])


def _max_abs_diff(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


class TestFixedSensorSigmaPolicyIndependent:
    """Fixed sensor: the Σ trajectory is byte-identical across two distinct policies."""

    def test_sigma_trajectories_byte_identical(self):
        model, belief, pref = _model(), _belief(), _obs_preference()
        a = policy_efe_trace(model, belief, _POLICY_A, pref)
        b = policy_efe_trace(model, belief, _POLICY_B, pref)
        np.testing.assert_array_equal(a.predicted_cov, b.predicted_cov, err_msg="Σ⁺")
        np.testing.assert_array_equal(
            a.posterior_cov, b.posterior_cov, err_msg="Σ_post"
        )
        np.testing.assert_array_equal(
            a.predicted_observation_cov, b.predicted_observation_cov, err_msg="S"
        )

    def test_means_genuinely_differ(self):
        # The Σ agreement above is policy-independence, not two identical policies:
        # the action still moves the mean.
        model, belief, pref = _model(), _belief(), _obs_preference()
        a = policy_efe_trace(model, belief, _POLICY_A, pref)
        b = policy_efe_trace(model, belief, _POLICY_B, pref)
        assert _max_abs_diff(a.predicted_mean, b.predicted_mean) > SEPARATION_MARGIN


class TestStateDependentNoiseSigmaSeparates:
    """State-dependent noise: the same two policies separate the Σ trajectory."""

    def test_rx_sigma_post_separates(self):
        model, belief, pref = _callable_model(), _belief(), _obs_preference()
        a = policy_efe_trace(model, belief, _POLICY_A, pref)
        b = policy_efe_trace(model, belief, _POLICY_B, pref)
        assert _max_abs_diff(a.posterior_cov, b.posterior_cov) > SEPARATION_MARGIN

    def test_qx_sigma_pred_separates(self):
        model = _internal_q_model()
        belief = Belief(mean=[0.0], cov=[[0.2]])
        pref = Preference(goal=[0.0], precision=[[1.0]])
        a = policy_efe_trace(model, belief, _POLICY_A, pref)
        b = policy_efe_trace(model, belief, _POLICY_B, pref)
        # Q(x) makes the predicted covariance itself policy-dependent.
        assert _max_abs_diff(a.predicted_cov, b.predicted_cov) > SEPARATION_MARGIN
