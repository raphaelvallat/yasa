"""
This file contains Numba-accelerated functions used in the main detections.
"""
import numpy as np
from numba import jit

__all__ = []


#############################################################################
# NUMBA JIT UTILITY FUNCTIONS
#############################################################################


@jit('float64(float64[:], float64[:])', nopython=True)
def _corr(x, y):
    """Fast Pearson correlation."""
    n = x.size
    mx, my = x.mean(), y.mean()
    xm2s, ym2s, r_num = 0, 0, 0
    for i in range(n):
        xm = x[i] - mx
        ym = y[i] - my
        r_num += (xm * ym)
        xm2s += xm**2
        ym2s += ym**2
    r_d1 = np.sqrt(xm2s)
    r_d2 = np.sqrt(ym2s)
    r_den = r_d1 * r_d2
    return r_num / r_den


@jit('float64(float64[:], float64[:])', nopython=True)
def _covar(x, y):
    """Fast Covariance."""
    n = x.size
    mx, my = x.mean(), y.mean()
    cov = 0
    for i in range(n):
        xm = x[i] - mx
        ym = y[i] - my
        cov += (xm * ym)
    return cov / (n - 1)


@jit('float64(float64[:])', nopython=True)
def _rms(x):
    """Fast root mean square."""
    n = x.size
    ms = 0
    for i in range(n):
        ms += x[i]**2
    ms /= n
    return np.sqrt(ms)


@jit('float64(float64[:], float64[:])', nopython=True)
def _slope_lstsq(x, y):
    """Slope of a 1D least-squares regression.
    """
    n_times = x.shape[0]
    sx2 = 0
    sx = 0
    sy = 0
    sxy = 0
    for j in range(n_times):
        sx2 += x[j] ** 2
        sx += x[j]
        sxy += x[j] * y[j]
        sy += y[j]
    den = n_times * sx2 - (sx ** 2)
    num = n_times * sxy - sx * sy
    return num / den


@jit('float64[:](float64[:], float64[:])', nopython=True)
def _detrend(x, y):
    """Fast linear detrending.
    """
    slope = _slope_lstsq(x, y)
    intercept = y.mean() - x.mean() * slope
    return y - (x * slope + intercept)
