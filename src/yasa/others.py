"""
This file contains several helper functions to manipulate 1D and 2D EEG data.
"""

import logging

import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.interpolate import interp1d
from scipy.special import erfinv

logger = logging.getLogger("yasa")

__all__ = ["moving_transform", "trimbothstd", "sliding_window", "get_centered_indices"]


def _merge_close(index, min_distance_ms, sf):
    """Merge events that are too close in time.

    Parameters
    ----------
    index : array_like
        Indices of supra-threshold events.
    min_distance_ms : int
        Minimum distance (ms) between two events to consider them as two
        distinct events
    sf : float
        Sampling frequency of the data (Hz)

    Returns
    -------
    f_index : array_like
        Filled (corrected) Indices of supra-threshold events

    Notes
    -----
    Original code imported from the Visbrain package.
    """
    # Convert min_distance_ms
    min_distance = min_distance_ms / 1000.0 * sf
    idx_diff = np.diff(index)
    condition = idx_diff > 1
    idx_distance = np.where(condition)[0]
    distance = idx_diff[condition]
    bad = idx_distance[np.where(distance < min_distance)[0]]
    # Fill gap between events separated with less than min_distance_ms
    if len(bad) > 0:
        fill = np.hstack([np.arange(index[j] + 1, index[j + 1]) for i, j in enumerate(bad)])
        f_index = np.sort(np.append(index, fill))
        return f_index
    else:
        return index


def _index_to_events(x):
    """Convert a 2D (start, end) array into a continuous one.

    Parameters
    ----------
    x : array_like
        2D array of indices.

    Returns
    -------
    index : array_like
        Continuous array of indices.

    Notes
    -----
    Original code imported from the Visbrain package.
    """
    x_copy = np.copy(x)
    x_copy[:, 1] += 1
    split_idx = x_copy.reshape(-1).astype(int)
    full_idx = np.arange(split_idx.max())
    index = np.split(full_idx, split_idx)[1::2]
    index = np.concatenate(index)
    return index


def moving_transform(x, y=None, sf=100, window=0.3, step=0.1, method="corr", interp=False):
    """Moving transformation of one or two time-series.

    Parameters
    ----------
    x : array_like
        Single-channel data
    y : array_like, optional
        Second single-channel data (only used if method in ['corr', 'covar']).
    sf : float
        Sampling frequency.
    window : int
        Window size in seconds.
    step : int
        Step in seconds.
        A step of 0.1 second (100 ms) is usually a good default.
        If step == 0, overlap at every sample (slowest)
        If step == nperseg, no overlap (fastest)
        Higher values = higher precision = slower computation.
    method : str
        Transformation to use.
        Available methods are::

            'mean' : arithmetic mean of x
            'min' : minimum value of x
            'max' : maximum value of x
            'ptp' : peak-to-peak amplitude of x
            'prop_above_zero' : proportion of values of x that are above zero
            'rms' : root mean square of x
            'slope' : slope of the least-square regression of x (in a.u / sec)
            'corr' : Correlation between x and y
            'covar' : Covariance between x and y
    interp : boolean
        If True, a linear interpolation is performed to ensure that the output
        has the same size as the input.

    Returns
    -------
    t : np.array
        Time vector, in seconds, corresponding to the MIDDLE of each epoch.
    out : np.array
        Transformed signal

    Notes
    -----
    This function was inspired by the `transform_signal` function of the
    Wonambi package (https://github.com/wonambi-python/wonambi).
    """
    # Safety checks
    assert method in [
        "mean",
        "min",
        "max",
        "ptp",
        "rms",
        "prop_above_zero",
        "slope",
        "covar",
        "corr",
    ]
    x = np.asarray(x, dtype=np.float64)
    if method in ("corr", "covar") and y is None:
        raise ValueError(f"y must be provided when method='{method}'")
    if y is not None:
        y = np.asarray(y, dtype=np.float64)
        assert x.size == y.size

    if step == 0:
        step = 1 / sf

    halfdur = window / 2
    n = x.size
    total_dur = n / sf
    last = n - 1
    idx = np.arange(0, total_dur, step)
    out = np.zeros(idx.size)

    # Define beginning, end and time (centered) vector.
    # end is an exclusive index (consistent with Python slice semantics and the
    # prefix-sum formula cx[end] - cx[beg]).  Clip to [0, n] so edge windows
    # can include the last sample x[n-1] without going out of bounds.
    beg = ((idx - halfdur) * sf).astype(int)
    end = ((idx + halfdur) * sf).astype(int)
    beg[beg < 0] = 0
    end[end > last] = n
    # Alternatively, to cut off incomplete windows (comment the 2 lines above)
    # mask = ~((beg < 0) | (end > n))
    # beg, end = beg[mask], end[mask]
    t = np.column_stack((beg, end)).mean(1) / sf

    # Window sizes in samples (may be smaller at the edges due to clipping)
    win_sz = (end - beg).astype(np.float64)

    if method == "mean":
        cx = np.zeros(n + 1)
        cx[1:] = np.cumsum(x)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(win_sz > 0, (cx[end] - cx[beg]) / win_sz, np.nan)

    elif method == "rms":
        cx2 = np.zeros(n + 1)
        cx2[1:] = np.cumsum(x**2)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(win_sz > 0, np.sqrt((cx2[end] - cx2[beg]) / win_sz), np.nan)

    elif method in ("corr", "covar"):
        # Prefix cumsums let each window's sum be read in O(1): sum(x[a:b]) =
        # cx[b] - cx[a].  Expanding the sample covariance / correlation by
        # substituting mean = sx/n yields the "computational formula":
        #
        #   cov(x,y)  = (Σxy  -  Σx·Σy / n) / (n - 1)
        #   var(x)    = (Σx²  -  (Σx)² / n) / (n - 1)
        #   corr(x,y) = cov(x,y) / sqrt(var(x) · var(y))
        #
        # where every Σ is evaluated over the window with a single subtraction
        # on the precomputed prefix arrays.  This avoids iterating over each
        # window individually.  Numerical precision is adequate here because
        # the EEG signals are bandpass-filtered (zero-mean), so the
        # cancellation term sx·sy/n is negligible.
        cx = np.zeros(n + 1)
        cx[1:] = np.cumsum(x)
        cy = np.zeros(n + 1)
        cy[1:] = np.cumsum(y)
        cxy = np.zeros(n + 1)
        cxy[1:] = np.cumsum(x * y)
        sx = cx[end] - cx[beg]
        sy = cy[end] - cy[beg]
        sxy = cxy[end] - cxy[beg]
        valid = win_sz >= 2
        if method == "covar":
            with np.errstate(invalid="ignore", divide="ignore"):
                out = np.where(valid, (sxy - sx * sy / win_sz) / (win_sz - 1), np.nan)
        else:  # corr
            cx2 = np.zeros(n + 1)
            cx2[1:] = np.cumsum(x**2)
            cy2 = np.zeros(n + 1)
            cy2[1:] = np.cumsum(y**2)
            sx2 = cx2[end] - cx2[beg]
            sy2 = cy2[end] - cy2[beg]
            with np.errstate(invalid="ignore", divide="ignore"):
                num = sxy - sx * sy / win_sz
                # Clip variance terms to 0 to guard against small negative values from floating-point error
                vx = np.clip(sx2 - sx**2 / win_sz, 0, None)
                vy = np.clip(sy2 - sy**2 / win_sz, 0, None)
                den = np.sqrt(vx * vy)
                out = np.where(valid & (den != 0.0), num / den, np.nan)

    elif method in ("min", "max", "ptp", "prop_above_zero"):
        # Group windows by their actual sample count (end - beg).  When
        # window * sf is not an integer, neighbouring windows alternate
        # between floor and ceil sizes, giving at most ~2 distinct values for
        # interior windows plus a handful of smaller edge windows.  For each
        # group, build a zero-copy (n_wins, wsz) view with as_strided, then
        # fancy-index to pick the correct rows and reduce along axis=1.
        # This is exact (no ±1 sample approximation) and avoids any Python loop.
        win_sz_int = (end - beg).astype(int)
        out[:] = np.nan  # default; overwritten for wsz >= 1 below
        for wsz in np.unique(win_sz_int):
            if wsz == 0:
                continue  # leave out[mask] = np.nan
            mask = win_sz_int == wsz
            # as_strided view: row i is x[i : i+wsz], shape (n-wsz+1, wsz).
            # beg[mask] + wsz == end[mask] <= n-1, so all row indices are valid.
            all_wins = as_strided(
                x,
                shape=(n - wsz + 1, wsz),
                strides=(x.strides[0], x.strides[0]),
            )
            wins = all_wins[beg[mask]]  # copy of selected rows, shape (k, wsz)
            if method == "min":
                out[mask] = wins.min(axis=1)
            elif method == "max":
                out[mask] = wins.max(axis=1)
            elif method == "ptp":
                out[mask] = wins.max(axis=1) - wins.min(axis=1)
            else:  # prop_above_zero
                out[mask] = (wins >= 0).sum(axis=1) / wsz

    elif method == "slope":
        for i in range(idx.size):
            seg = x[beg[i] : end[i]]
            m = seg.size
            if m < 2:
                out[i] = np.nan
                continue
            times = np.arange(m, dtype=np.float64) / sf
            sx = times.sum()
            sy = seg.sum()
            sxy = np.dot(times, seg)
            sx2 = np.dot(times, times)
            den = m * sx2 - sx * sx
            out[i] = (m * sxy - sx * sy) / den if den != 0 else np.nan

    # Finally interpolate
    if interp and step != 1 / sf:
        f = interp1d(t, out, kind="linear", bounds_error=False, fill_value=0, assume_sorted=True)
        t = np.arange(n) / sf
        out = f(t)

    return t, out


def _zerocrossings(x):
    """Find indices of zero-crossings in a 1D array.

    Parameters
    ----------
    x : np.array
        One dimensional data vector.

    Returns
    -------
    idx_zc : np.array
        Indices of zero-crossings

    Examples
    --------
    >>> import numpy as np
    >>> from yasa.main import _zerocrossings
    >>> a = np.array([4, 2, -1, -3, 1, 2, 3, -2, -5])
    >>> _zerocrossings(a)
        array([1, 3, 6], dtype=int64)
    """
    pos = x > 0
    npos = ~pos
    return ((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0]


def trimbothstd(x, cut=0.10):
    """
    Slices off a proportion of items from both ends of an array and then
    compute the sample standard deviation.

    Slices off the passed proportion of items from both ends of the passed
    array (i.e., with ``cut`` = 0.1, slices leftmost 10% **and**
    rightmost 10% of scores). The trimmed values are the lowest and
    highest ones.
    Slices off less if proportion results in a non-integer slice index.

    Parameters
    ----------
    x : np.array
        Input array.
    cut : float
        Proportion (in range 0-1) of total data to trim of each end.
        Default is 0.10, i.e. 10% lowest and 10% highest values are removed.

    Returns
    -------
    trimmed_std : float
        Sample standard deviation of the trimmed array, calculated on the last
        axis.
    """
    x = np.asarray(x)
    n = x.shape[-1]
    lowercut = int(cut * n)
    uppercut = n - lowercut
    atmp = np.partition(x, (lowercut, uppercut - 1), axis=-1)
    sl = slice(lowercut, uppercut)
    return np.nanstd(atmp[..., sl], ddof=1, axis=-1)


def sliding_window(data, sf, window, step=None, axis=-1):
    """Calculate a sliding window of a 1D or 2D EEG signal.

    .. versionadded:: 0.1.7

    Parameters
    ----------
    data : numpy array
        The 1D or 2D EEG data.
    sf : float
        The sampling frequency of ``data``.
    window : int
        The sliding window length, in seconds.
    step : int
        The sliding window step length, in seconds.
        If None (default), ``step`` is set to ``window``,
        which results in no overlap between the sliding windows.
    axis : int
        The axis to slide over. Defaults to the last axis.

    Returns
    -------
    times : numpy array
        Time vector, in seconds, corresponding to the START of each sliding
        epoch in ``strided``.
    strided : numpy array
        A matrix where row in last dimension consists of one instance
        of the sliding window, shape (n_epochs, ..., n_samples).

    Notes
    -----
    This is a wrapper around the
    :py:func:`numpy.lib.stride_tricks.as_strided` function.

    Examples
    --------
    With a 1-D array

    >>> import numpy as np
    >>> from yasa import sliding_window
    >>> data = np.arange(20)
    >>> times, epochs = sliding_window(data, sf=1, window=5)
    >>> times
    array([ 0.,  5., 10., 15.])

    >>> epochs
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19]])

    >>> sliding_window(data, sf=1, window=5, step=1)[1]
    array([[ 0,  1,  2,  3,  4],
           [ 2,  3,  4,  5,  6],
           [ 4,  5,  6,  7,  8],
           [ 6,  7,  8,  9, 10],
           [ 8,  9, 10, 11, 12],
           [10, 11, 12, 13, 14],
           [12, 13, 14, 15, 16],
           [14, 15, 16, 17, 18]])

    >>> sliding_window(data, sf=1, window=11)[1]
    array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10]])

    With a N-D array

    >>> np.random.seed(42)
    >>> # 4 channels x 20 samples
    >>> data = np.random.randint(-100, 100, size=(4, 20))
    >>> epochs = sliding_window(data, sf=1, window=10)[1]
    >>> epochs.shape  # shape (n_epochs, n_channels, n_samples)
    (2, 4, 10)

    >>> epochs
    array([[[  2,  79,  -8, -86,   6, -29,  88, -80,   2,  21],
            [-13,  57, -63,  29,  91,  87, -80,  60, -43, -79],
            [-50,   7, -46, -37,  30, -50,  34, -80, -28,  66],
            [ -9,  10,  87,  98,  71, -93,  74, -66, -20,  63]],
           [[-26, -13,  16,  -1,   3,  51,  30,  49, -48, -99],
            [-12, -52, -42,  69,  87, -86,  89,  89,  74,  89],
            [-83,  31, -12, -41, -87, -92, -11, -48,  29, -17],
            [-51,   3,  31, -99,  33, -47,   5, -97, -47,  90]]])
    """
    assert axis <= data.ndim, "Axis value out of range."
    assert isinstance(sf, (int, float)), "sf must be int or float"
    assert isinstance(window, (int, float)), "window must be int or float"
    assert isinstance(step, (int, float, type(None))), "step must be int, float or None."
    if isinstance(sf, float):
        assert sf.is_integer(), "sf must be a whole number."
        sf = int(sf)
    assert isinstance(axis, int), "axis must be int."

    # window and step in samples instead of points
    window *= sf
    step = window if step is None else step * sf

    if isinstance(window, float):
        assert window.is_integer(), "window * sf must be a whole number."
        window = int(window)

    if isinstance(step, float):
        assert step.is_integer(), "step * sf must be a whole number."
        step = int(step)

    assert step >= 1, "Stepsize may not be zero or negative."
    assert window < data.shape[axis], "Sliding window size may not exceed size of selected axis"

    # Define output shape
    shape = list(data.shape)
    shape[axis] = np.floor(data.shape[axis] / step - window / step + 1).astype(int)
    shape.append(window)

    # Calculate strides and time vector
    strides = list(data.strides)
    strides[axis] *= step
    strides.append(data.strides[axis])
    strided = as_strided(data, shape=shape, strides=strides)
    t = np.arange(strided.shape[-2]) * (step / sf)

    # Swap axis: n_epochs, ..., n_samples
    if strided.ndim > 2:
        strided = np.rollaxis(strided, -2, 0)
    return t, strided


def get_centered_indices(data, idx, npts_before, npts_after):
    """Get a 2D array of indices in data centered around specific time points,
    automatically excluding indices that are outside the bounds of data.

    Parameters
    ----------
    data : 1-D array_like
        Input data.
    idx : 1-D array_like
        Indices of events in data (e.g. peaks)
    npts_before : int
        Number of data points to include before ``idx``
    npts_after : int
        Number of data points to include after ``idx``

    Returns
    -------
    idx_ep : 2-D array
        Array of indices of shape (len(idx_nomask), npts_before +
        npts_after + 1). Indices outside the bounds of data are removed.
    idx_nomask : 1-D array
        Indices of ``idx`` that are not masked (= valid).

    Examples
    --------
    >>> import numpy as np
    >>> from yasa import get_centered_indices
    >>> np.random.seed(123)
    >>> data = np.random.normal(size=100).round(2)
    >>> idx = [1.0, 10.0, 20.0, 30.0, 50.0, 102]
    >>> before, after = 3, 2
    >>> idx_ep, idx_nomask = get_centered_indices(data, idx, before, after)
    >>> idx_ep
    array([[ 7,  8,  9, 10, 11, 12],
           [17, 18, 19, 20, 21, 22],
           [27, 28, 29, 30, 31, 32],
           [47, 48, 49, 50, 51, 52]])

    >>> data[idx_ep]
    array([[-0.43,  1.27, -0.87, -0.68, -0.09,  1.49],
           [ 2.19,  1.  ,  0.39,  0.74,  1.49, -0.94],
           [-1.43, -0.14, -0.86, -0.26, -2.8 , -1.77],
           [ 0.41,  0.98,  2.24, -1.29, -1.04,  1.74]])

    >>> idx_nomask
    array([1, 2, 3, 4], dtype=int64)
    """
    # Safety check
    assert isinstance(npts_before, (int, float))
    assert isinstance(npts_after, (int, float))
    assert float(npts_before).is_integer()
    assert float(npts_after).is_integer()
    npts_before = int(npts_before)
    npts_after = int(npts_after)
    data = np.asarray(data)
    idx = np.asarray(idx, dtype="int")
    assert idx.ndim == 1, "idx must be 1D."
    assert data.ndim == 1, "data must be 1D."

    def rng(x):
        """Create a range before and after a given value."""
        return np.arange(x[0] - npts_before, x[0] + npts_after + 1, dtype="int")

    idx_ep = np.apply_along_axis(rng, 1, idx[..., np.newaxis])
    # We drop the events for which the indices exceed data
    idx_ep = np.ma.mask_rows(np.ma.masked_outside(idx_ep, 0, data.shape[0]))
    # Indices of non-masked (valid) epochs in idx
    idx_ep_nomask = np.unique(idx_ep.nonzero()[0])
    idx_ep = np.ma.compress_rows(idx_ep)
    return idx_ep, idx_ep_nomask


def _norm_direct_pac(pha, amp, p=0.05):
    """Normalized direct PAC (ndPAC).

    Re-implementation of tensorpac's ``norm_direct_pac`` (Ozkurt et al. 2012).

    Parameters
    ----------
    pha : array_like
        Phase array of shape (n_pha, ..., n_times).
    amp : array_like
        Amplitude array of shape (n_amp, ..., n_times).
    p : float | .05
        P-value threshold. Sub-threshold PAC values are set to 0.
        Use ``p=1`` or ``p=None`` to disable thresholding.

    Returns
    -------
    pac : array_like
        Phase-amplitude coupling array of shape (n_amp, n_pha, ...).
    """
    n_times = amp.shape[-1]
    amp = np.subtract(amp, np.mean(amp, axis=-1, keepdims=True))
    amp = np.divide(amp, np.std(amp, ddof=1, axis=-1, keepdims=True))
    pac = np.abs(np.einsum("i...j, k...j->ik...", amp, np.exp(1j * pha)))
    if p == 1.0 or p is None:
        return pac / n_times
    s = pac**2
    pac /= n_times
    xlim = n_times * erfinv(1 - p) ** 2
    pac[s <= 2 * xlim] = 0.0
    return pac
