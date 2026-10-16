"""Tests for the abstraction wall (backends/base.py).

The point of these is not any one backend's behaviour but *swappability*: two
unrelated classes satisfy InferenceBackend purely by shape, with no shared base.
If both pass isinstance, the wall holds and engines are interchangeable.
"""

import numpy as np

from cpomdp.backends.base import InferenceBackend
from cpomdp.backends.kalman import KalmanBackend
from cpomdp.types import Belief, LinearGaussianModel


class _PassthroughBackend:
    """A trivial fake backend: returns the prior unchanged. It inherits nothing
    and knows nothing about InferenceBackend — it conforms by structure alone
    (a matching ``infer_states`` and the ``model`` it was built from)."""

    def __init__(self, model: LinearGaussianModel) -> None:
        self.model = model

    def infer_states(
        self,
        observation: np.ndarray,
        prior: Belief,
        action: np.ndarray | None = None,
    ) -> Belief:
        return prior


def _scalar_model():
    return LinearGaussianModel(
        dynamics_matrix=[[0.9]],
        observation_matrix=[[1.0]],
        dynamics_noise=[[0.5]],
        observation_noise=[[1.0]],
        prior=Belief(mean=[0.0], cov=[[10.0]]),
    )


def test_fake_backend_satisfies_protocol():
    assert isinstance(_PassthroughBackend(_scalar_model()), InferenceBackend)


def test_kalman_backend_satisfies_protocol():
    assert isinstance(KalmanBackend(_scalar_model()), InferenceBackend)
