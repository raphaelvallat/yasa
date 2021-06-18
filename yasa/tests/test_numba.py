"""Test the functions in the yasa/numba.py file."""
import unittest
import numpy as np
from scipy.signal import detrend
from yasa.numba import _corr, _covar, _rms, _slope_lstsq, _detrend


class TestNumba(unittest.TestCase):

    def test_numba(self):
        """Test numba functions
        """
        x = np.asarray([4, 5, 7, 8, 5, 6], dtype=np.float64)
        y = np.asarray([1, 5, 4, 6, 8, 5], dtype=np.float64)

        np.testing.assert_almost_equal(_corr(x, y), np.corrcoef(x, y)[0][1])
        assert _covar(x, y) == np.cov(x, y)[0][1]
        assert _rms(x) == np.sqrt(np.mean(np.square(x)))

        # Least square slope and detrending
        y = np.arange(30) + 3 * np.random.random(30)
        times = np.arange(y.size, dtype=np.float64)
        slope = _slope_lstsq(times, y)
        np.testing.assert_array_almost_equal(_detrend(times, y),
                                             detrend(y, type='linear'))
        X = times[..., np.newaxis]
        X = np.column_stack((np.ones(X.shape[0]), X))
        slope_np = np.linalg.lstsq(X, y, rcond=None)[0][1]
        np.round(slope, 5) == np.round(slope_np, 5)
