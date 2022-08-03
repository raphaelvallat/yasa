"""
ECG-based functions.

Author: Dr Raphael Vallat <raphaelvallat@berkeley.edu>, UC Berkeley.
Date: May 2022
"""
import logging
import numpy as np
import pandas as pd

from .hypno import hypno_find_periods
from .detection import _check_data_hypno
from .io import set_log_level, is_sleepecg_installed

logger = logging.getLogger("yasa")

__all__ = ["hrv_stage"]


def hrv_stage(
    data,
    sf,
    *,
    hypno=None,
    include=(2, 3, 4),
    threshold="2min",
    equal_length=False,
    rr_limit=(400, 2000),
    verbose=False,
):
    """Calculate heart rate and heart rate variability (HRV) features from an ECG.

    By default, the cardiac features are calculated for each period of N2, N3 or REM sleep that
    are longer than 2 minutes.

    .. versionadded:: 0.6.2

    Parameters
    ----------
    data : :py:class:`numpy.ndarray`
        Single-channel ECG data. Must be a 1D NumPy array.
    sf : float
        The sampling frequency of the data.
    hypno : array_like
        Sleep stage (hypnogram). The heart rate calculation will be applied for each sleep stage
        defined in ``include`` (default = N2, N3 and REM sleep separately).

        The hypnogram must have the same number of samples as ``data``.
        To upsample your hypnogram, please refer to
        :py:func:`yasa.hypno_upsample_to_data`.

        .. note::
            The default hypnogram format in YASA is a 1D integer
            vector where:

            - -2 = Unscored
            - -1 = Artefact / Movement
            - 0 = Wake
            - 1 = N1 sleep
            - 2 = N2 sleep
            - 3 = N3 sleep
            - 4 = REM sleep
    include : tuple, list or int
        Values in ``hypno`` that will be included in the mask. The default is
        (2, 3, 4), meaning that the detection is applied on N2, N3 and REM
        sleep separately.
    threshold : str
        Only periods of a given stage that exceed the duration defined in ``threshold`` will be
        kept in subsequent analysis. The default is 2 minutes ('2min'). Other possible values
        include: '5min', '15min', '30sec', '1hour', etc. To disable thresholding, use '0min'.
    equal_length : bool
        If True, the periods will all have the exact duration defined in ``threshold``.
        That is, periods that are longer than the duration threshold will be divided into
        sub-periods of exactly the length of threshold.
    rr_limit : tuple
        Lower and upper limit for the RR interval. Default is 400 to 2000 ms, corresponding to a
        heart rate of 30 to 150 bpm. RR intervals outside this range will be set to NaN and
        filled with linear interpolation. Use ``rr_limit=(0, np.inf)`` to disable RR correction.
    verbose : bool or str
        Verbose level. Default (False) will only print warning and error
        messages. The logging levels are 'debug', 'info', 'warning', 'error',
        and 'critical'. For most users the choice is between 'info'
        (or ``verbose=True``) and warning (``verbose=False``). Set this to True if you are getting
        invalid results and want to better understand what is happening.

    Returns
    -------
    epochs : :py:class:`pandas.DataFrame`
        Output dataframe with values (= the sleep stages defined in ``include``) and
        epoch number as index. The columns are

        * ``start`` : The start of the epoch, in seconds from the beginning of the recording.
        * ``duration`` : The duration of the epoch, in seconds.
        * ``hr_mean``: The mean heart rate (HR) across the epoch, in beats per minute (bpm).
        * ``hr_std``: The standard deviation of the HR across the epoch, in bpm
        * ``hrv_rmssd``: Heart rate variability across the epoch (RMSSD), in milliseconds.
    rpeaks : dict
        A dictionary with the detected heartbeats (R-peaks) indices for each epoch of each stage.
        Indices are expressed as samples from the beginning of the epoch. This can be used to
        manually recalculate the RR intervals, apply a custom preprocessing on the RR intervals,
        and/or calculate more advanced HRV metrics.

    Notes
    -----
    This function returns three cardiac features for each epoch: the mean and standard deviation of
    the heart rate, and the root mean square of successive differences between normal heartbeats
    (RMSSD). The RMSSD reflects the beat-to-beat variance in HR and is the primary time-domain
    measure used to estimate the vagally mediated changes reflected in heart rate variability.

    Heartbeat detection is performed with the SleepECG library: https://github.com/cbrnr/sleepecg

    For an example of this function, please see the `Jupyter notebook
    <https://github.com/raphaelvallat/yasa/blob/master/notebooks/16_EEG-HRV_coupling.ipynb>`_

    References
    ----------
    * Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and
      norms. Frontiers in public health, 258.
    """
    set_log_level(verbose)
    is_sleepecg_installed()
    from sleepecg import detect_heartbeats

    if isinstance(hypno, type(None)):
        logger.warning(
            "No hypnogram was passed. The entire recording will be used, i.e. "
            "hypno will be set to np.zeros(data.size) and include will be set to 0."
        )
        data = np.asarray(data, dtype=np.float64)
        hypno = np.zeros(max(data.shape), dtype=int)
        include = 0

    # Safety check
    (data, sf, _, hypno, include, _, n_chan, n_samples, _) = _check_data_hypno(
        data, sf, None, hypno, include, check_amp=False
    )
    assert n_chan == 1, "data must be a 1D ECG array."
    data = np.squeeze(data)

    # Find periods of equal duration
    epochs = hypno_find_periods(hypno, sf, threshold=threshold, equal_length=equal_length)
    assert epochs.shape[0] > 0, f"No epochs longer than {threshold} found in hypnogram."
    epochs = epochs[epochs["values"].isin(include)].reset_index(drop=True)
    # Sort by stage and add epoch number
    epochs = epochs.sort_values(by=["values", "start"])
    epochs["epoch"] = epochs.groupby("values")["start"].transform(lambda x: range(len(x)))
    epochs = epochs.set_index(["values", "epoch"])

    # Loop over epochs
    rpeaks = {}
    for idx in epochs.index:
        start = epochs.loc[idx, "start"]
        duration = epochs.loc[idx, "length"]
        end = int(epochs.loc[idx, "start"] + duration)
        # Detect R-peaks
        try:
            pks = detect_heartbeats(data[start:end], fs=sf)
        except Exception as e:
            logger.info(f"Heartbeat detection failed for epoch {idx[1]} of stage {idx[0]}: {e}")
            continue

        # Save rpeaks to dict
        rpeaks[idx] = pks

        # If not enough R-peaks were detected, skip epochs and return NaN
        # Here, we assume a minimal HR of 30 bpm
        constant_hr = 60 * (pks.size / (duration / sf))
        if constant_hr < 30:
            logger.info(f"Too few detected heartbeats in epoch {idx[1]} of stage {idx[0]}.")
            continue

        # Find and correct RR intervals. Default is 400 ms (150 bpm) to 2000 ms (30 bpm)
        rri = 1000 * np.diff(pks) / sf
        rri = np.ma.masked_outside(rri, rr_limit[0], rr_limit[1]).filled(np.nan)
        # Interpolate NaN values, but no more than 10 consecutive values
        if np.isnan(rri).any():
            rri = pd.Series(rri).interpolate(limit_direction="both", limit=10).to_numpy()
        if np.isnan(rri).any():
            # If there are still NaN present, skip current epoch
            logger.info(f"Invalid RR intervals in epoch {idx[1]} of stage {idx[0]}.")
            continue

        # Heart rate
        hr = 60000 / rri
        epochs.loc[idx, "hr_mean"] = np.mean(hr)
        epochs.loc[idx, "hr_std"] = np.std(hr, ddof=1)
        epochs.loc[idx, "hrv_rmssd"] = np.sqrt(np.mean(np.diff(rri) ** 2))

    # Convert start and duration to seconds
    epochs["start"] /= sf
    epochs["length"] /= sf
    epochs = epochs.rename(columns={"length": "duration"})

    return epochs, rpeaks
