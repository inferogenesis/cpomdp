import numpy as np
import pytest

from cpomdp.types import Belief


def test_valid_belief_stores_what_you_passed():
    b = Belief(mean=[1.0, 2.0], cov=[[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_array_equal(b.mean, [1.0, 2.0])
    np.testing.assert_array_equal(b.cov, [[1.0, 0.0], [0.0, 1.0]])

def test_coerce_lists_to_float_arrays():
    b = Belief(mean=[0, 1], cov=[[1, 0], [0, 1]])
    assert isinstance(b.mean, np.ndarray)
    assert b.mean.dtype == np.float64

def test_rejects_mean_not_1D():
    with pytest.raises(ValueError, match="1-D"):
        Belief(mean=[[0.0]], cov=[[1.0]])

def test_rejects_cov_not_2D():
    with pytest.raises(ValueError, match="2-D"):
        Belief(mean=[0.0], cov=[1.0])

def test_reject_shape_mismatch():
    with pytest.raises(ValueError, match="match"):
        Belief(mean=[0.0, 0.0], cov=[[1.0]])

def test_rejects_asymmetric_cov():
    with pytest.raises(ValueError, match="symmetric"):
        Belief(mean=[0.0, 0.0], cov=[[1.0, 0.2], [0.9, 1.0]])

def test_ndim_reports_state_dimension():
    assert Belief(mean=[0.0], cov=[[1.0]]).ndim == 1
    assert Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]).ndim == 2