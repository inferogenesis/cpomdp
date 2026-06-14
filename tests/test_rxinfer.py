"""RxInfer oracle backend tests.

These boot the Julia runtime, so they are slow and gated behind the ``rxinfer``
marker (deselect with ``-m "not rxinfer"``). The point of the backend is to agree
with the native KalmanBackend through entirely separate machinery, so most tests
here assert exactly that agreement rather than re-deriving numbers by hand.
"""

import numpy as np
import pytest

from cpomdp.backends.base import InferenceBackend
from cpomdp.backends.kalman import KalmanBackend
from cpomdp.backends.rxinfer import RxInferBackend
from cpomdp.types import Belief, LinearGaussianModel

pytestmark = pytest.mark.rxinfer

OBSERVATIONS = [1.2, 0.8, 1.5, 2.1, 1.9, 2.4, 2.0, 1.7]


def _scalar_model():
    return LinearGaussianModel(
        dynamics=[[0.9]],
        sensor_model=[[1.0]],
        dynamics_noise=[[0.5]],
        sensor_noise=[[1.0]],
        prior=Belief(mean=[0.0], cov=[[10.0]]),
    )


def _pos_vel_model():
    return LinearGaussianModel(
        dynamics=[[1.0, 1.0], [0.0, 1.0]],
        sensor_model=[[1.0, 0.0]],
        dynamics_noise=[[0.01, 0.0], [0.0, 0.01]],
        sensor_noise=[[1.0]],
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
    )


def _control_model():
    return LinearGaussianModel(
        dynamics=[[1.0]],
        sensor_model=[[1.0]],
        dynamics_noise=[[0.5]],
        sensor_noise=[[1.0]],
        prior=Belief(mean=[0.0], cov=[[10.0]]),
        control=[[1.0]],
    )


def test_satisfies_protocol():
    assert isinstance(RxInferBackend(_scalar_model()), InferenceBackend)


def test_scalar_matches_rxinfer_published_oracle():
    # The exact values the Phase-0 spike read straight out of RxInfer.jl, so this
    # also pins that we drive RxInfer the same way the spike did.
    model = _scalar_model()
    belief = model.prior
    backend = RxInferBackend(model)
    for y in OBSERVATIONS:
        belief = backend.infer_states(np.array([y]), belief)
    np.testing.assert_allclose(belief.mean, [1.679270599888], rtol=1e-9)
    np.testing.assert_allclose(belief.cov, [[0.467784044120]], rtol=1e-9)


def test_scalar_agrees_with_kalman_every_step():
    model = _scalar_model()
    kalman, rxinfer = KalmanBackend(model), RxInferBackend(model)
    kb, rb = model.prior, model.prior
    for y in OBSERVATIONS:
        obs = np.array([y])
        kb = kalman.infer_states(obs, kb)
        rb = rxinfer.infer_states(obs, rb)
        np.testing.assert_allclose(rb.mean, kb.mean, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(rb.cov, kb.cov, rtol=1e-9, atol=1e-12)


def test_multivariate_agrees_with_kalman():
    # The off-diagonal / transpose coverage scalars can't give: observing position
    # must sharpen velocity through the cross-covariance, identically in both.
    model = _pos_vel_model()
    kalman, rxinfer = KalmanBackend(model), RxInferBackend(model)
    obs = np.array([1.0])
    kb = kalman.infer_states(obs, model.prior)
    rb = rxinfer.infer_states(obs, model.prior)
    np.testing.assert_allclose(rb.mean, kb.mean, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(rb.cov, kb.cov, rtol=1e-9, atol=1e-12)


def test_control_agrees_with_kalman():
    model = _control_model()
    kalman, rxinfer = KalmanBackend(model), RxInferBackend(model)
    obs, action = np.array([1.0]), np.array([5.0])
    kb = kalman.infer_states(obs, model.prior, action=action)
    rb = rxinfer.infer_states(obs, model.prior, action=action)
    np.testing.assert_allclose(rb.mean, kb.mean, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(rb.cov, kb.cov, rtol=1e-9, atol=1e-12)


def test_requires_action_when_model_has_control():
    backend = RxInferBackend(_control_model())
    with pytest.raises(ValueError, match="action"):
        backend.infer_states(np.array([1.0]), _control_model().prior)


def test_does_not_mutate_prior():
    model = _scalar_model()
    prior = model.prior
    mean_before, cov_before = prior.mean.copy(), prior.cov.copy()
    RxInferBackend(model).infer_states(np.array([1.2]), prior)
    np.testing.assert_array_equal(prior.mean, mean_before)
    np.testing.assert_array_equal(prior.cov, cov_before)
