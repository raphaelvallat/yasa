"""
YASA (Yet Another Spindle Algorithm): fast and robust detection of spindles,
slow-waves, and rapid eye movements from sleep EEG recordings.

- Author: Raphael Vallat (www.raphaelvallat.com)
- GitHub: https://github.com/raphaelvallat/yasa
- License: BSD 3-Clause License
"""

import logging
from itertools import product

import mne
import numpy as np
import pandas as pd
from mne.filter import filter_data
from scipy import signal
from scipy.fftpack import next_fast_len
from scipy.special import erfinv
from scipy.stats import circmean
from sklearn.ensemble import IsolationForest

from ._validation import _check_data_hypno
from .io import _restore_log_level, is_pyriemann_installed
from .others import (
    _index_to_events,
    _merge_close,
    _zerocrossings,
    get_centered_indices,
    moving_transform,
    sliding_window,
    trimbothstd,
)
from .spectral import stft_power

logger = logging.getLogger("yasa")

__all__ = [
    "art_detect",
    "spindles_detect",
    "SpindlesResults",
    "sw_detect",
    "SWResults",
    "rem_detect",
    "REMResults",
    "compare_detection",
]


#############################################################################
# DATA PREPROCESSING
#############################################################################


def _detrend_linear(x):
    """Remove the least-squares linear trend of a 1D array.

    Same output as ``scipy.signal.detrend(x, type="linear")``, but much faster on short arrays.
    """
    t = np.arange(x.size) - (x.size - 1) / 2  # Centered time, so that sum(t) = 0
    return x - x.mean() - t * (t @ x) / (t @ t)


def _remove_outliers(df, features, ch_name=None):
    """Remove outlier events using an Isolation Forest on the event features.

    At least 50 events are required, otherwise ``df`` is returned unchanged.
    """
    if df.shape[0] < 50:
        return df
    ilf = IsolationForest(contamination="auto", max_samples="auto", verbose=0, random_state=42)
    is_inlier = ilf.fit_predict(df[list(features)]) == 1
    where = "" if ch_name is None else f" in channel {ch_name}"
    logger.info("%i outliers were removed%s.", (~is_inlier).sum(), where)
    return df[is_inlier]


#############################################################################
# BASE DETECTION RESULTS CLASS
#############################################################################


class _DetectionResults(object):
    """Main class for detection results."""

    # Event features, used as the averaged columns of summary() and for outlier removal
    _features = ()
    # Title of plot_average()
    _title = ""
    # Channel label used when the events are not detected on a single channel (e.g. REMs)
    _channel_label = None

    def __init__(self, events, data, sf, ch_names, hypno, data_filt):
        self._events = events
        self._data = data
        self._sf = sf
        self._hypno = hypno
        self._ch_names = ch_names
        self._data_filt = data_filt

    def _check_mask(self, mask):
        assert isinstance(mask, (pd.Series, np.ndarray, list, type(None)))
        n_events = self._events.shape[0]
        if mask is None:
            mask = np.ones(n_events, dtype="bool")  # All set to True
        else:
            mask = np.asarray(mask)
            assert mask.dtype.kind == "b", "Mask must be a boolean array."
            assert mask.ndim == 1, "Mask must be one-dimensional"
            assert mask.size == n_events, "Mask.size must be the number of detected events."
        return mask

    def _iter_channels(self, events):
        """Yield (index of channel in data, events of this channel)."""
        for i, ev_chan in events.groupby("IdxChannel"):
            yield i, ev_chan

    def _summary_with_channel(self):
        """Return summary() with a Channel column, as needed by compare_detection."""
        df = self.summary()
        if self._channel_label is not None:
            df["Channel"] = self._channel_label
        return df

    def _get_aggdict(self, aggfunc):
        return {"Start": "count", **dict.fromkeys(self._features, aggfunc)}

    def summary(self, grp_chan=False, grp_stage=False, aggfunc="mean", sort=True, mask=None):
        """Summary"""
        # Check masking
        mask = self._check_mask(mask)

        # Define grouping
        grouper = []
        if grp_stage is True and "Stage" in self._events:
            grouper.append("Stage")
        if grp_chan is True and "Channel" in self._events:
            grouper.append("Channel")
        if not len(grouper):
            # Return a copy of self._events after masking, without grouping
            return self._events.loc[mask, :].copy()

        # Apply grouping, after masking
        df_grp = (
            self._events.loc[mask, :]
            .groupby(grouper, sort=sort, as_index=False)
            .agg(self._get_aggdict(aggfunc))
        )
        df_grp = df_grp.rename(columns={"Start": "Count"})

        # Calculate density (= number per min of each stage)
        if self._hypno is not None and grp_stage is True:
            # Duration in minutes of each stage
            dur = pd.Series(self._hypno).value_counts() / (60 * self._sf)
            # Insert new density column in grouped dataframe after count
            df_grp.insert(
                loc=df_grp.columns.get_loc("Count") + 1,
                column="Density",
                value=df_grp["Count"] / df_grp["Stage"].map(dur),
            )

        return df_grp.set_index(grouper)

    def get_mask(self):
        """get_mask"""
        mask = np.zeros(self._data.shape, dtype=int)
        for i, ev_chan in self._iter_channels(self._events):
            # Round (not truncate) to recover the sample indices, e.g. 0.29 * 100 = 28.999...
            idx_ev = _index_to_events(np.round(ev_chan[["Start", "End"]].to_numpy() * self._sf))
            mask[i, idx_ev] = 1
        return np.squeeze(mask)

    def _filter_data(self, filt):
        """Return the data, optionally bandpass-filtered with filt=(l_freq, h_freq)."""
        if not any(filt):
            return self._data
        return mne.filter.filter_data(
            self._data, self._sf, l_freq=filt[0], h_freq=filt[1], method="fir", verbose=False
        )

    def get_sync_events(
        self, center, time_before, time_after, filt=(None, None), mask=None, as_dataframe=True
    ):
        """Get_sync_events"""
        assert time_before >= 0
        assert time_after >= 0
        bef = int(self._sf * time_before)
        aft = int(self._sf * time_after)
        # TODO: Step size is determined by sf: 0.01 sec at 100 Hz, 0.002 sec at
        # 500 Hz, 0.00390625 sec at 256 Hz. Should we add resample=100 (Hz) or step_size=0.01?
        time = np.arange(-bef, aft + 1, dtype="int") / self._sf
        n_times = time.size
        data = self._filter_data(filt)

        # Apply mask
        mask = self._check_mask(mask)
        masked_events = self._events.loc[mask, :]

        output = []
        for i, ev_chan in self._iter_channels(masked_events):
            peaks = np.round(ev_chan[center].to_numpy() * self._sf).astype(int)
            # Get centered indices. Events too close to the data edges are dropped.
            idx, idx_valid = get_centered_indices(data[i, :], peaks, bef, aft)
            # If no good epochs are returned raise a warning
            if len(idx_valid) == 0:
                logger.error(
                    "Time before and/or time after exceed data bounds, please "
                    "lower the temporal window around center. Skipping channel."
                )
                continue

            # Get data at indices, shape (n_events, n_times)
            amps = data[i, idx]

            if not as_dataframe:
                # Output is a list (n_channels) of numpy arrays (n_events, n_times)
                output.append(amps)
                continue

            # Convert to long-format dataframe
            n_events = amps.shape[0]
            df_chan = pd.DataFrame(
                {
                    "Time": np.tile(time, n_events),
                    "Event": np.repeat(np.arange(n_events), n_times),
                    "Amplitude": amps.ravel(),
                }
            )
            if "Stage" in masked_events:
                # idx_valid maps each epoch back to its event, which gives the correct stage
                # even when some events were dropped because they were too close to the edges
                df_chan["Stage"] = np.repeat(ev_chan["Stage"].to_numpy()[idx_valid], n_times)
            df_chan["Channel"] = self._ch_names[i]
            df_chan["IdxChannel"] = i
            output.append(df_chan)

        if as_dataframe:
            output = pd.concat(output, ignore_index=True) if output else pd.DataFrame()

        return output

    def get_coincidence_matrix(self, scaled=True):
        """get_coincidence_matrix"""
        if len(self._ch_names) < 2:
            raise ValueError("At least 2 channels are required to calculate coincidence.")
        mask = self.get_mask().astype(np.float64)
        # Number of samples that are marked as an event in both channels, for each pair of channels
        coinc = mask @ mask.T
        n_ev = mask.sum(axis=1)
        if scaled:
            with np.errstate(divide="ignore", invalid="ignore"):
                coinc = coinc / np.outer(n_ev, n_ev)
            coinc[~np.isfinite(coinc)] = np.nan
            np.fill_diagonal(coinc, 1)
        else:
            coinc = coinc.astype(int)
        coinc_mat = pd.DataFrame(coinc, index=self._ch_names, columns=self._ch_names)
        coinc_mat.index.name = "Channel"
        coinc_mat.columns.name = "Channel"
        return coinc_mat

    @staticmethod
    def _starts_by_channel(df):
        """Return a dict with, for each channel, the start of the events in deciseconds.

        Start times are rounded to the nearest decisecond (100 ms). This is needed for three
        reasons: 1) speed up yasa.compare_detection, 2) avoid memory errors and 3) make sure that
        max_distance works even when the two detections have different sampling frequencies.
        """
        starts = (df["Start"] * 10).round().astype(int)
        return {ch: grp.to_numpy() for ch, grp in starts.groupby(df["Channel"], sort=False)}

    def compare_channels(self, score="f1", max_distance_sec=0):
        """
        Compare detected events across channels.
        See full documentation in the methods of SpindlesResults and SWResults.
        """
        assert score in ["f1", "precision", "recall"], f"Invalid scoring metric: {score}"
        # TODO: Only the Start of the event is currently supported. Add more flexibility?
        starts = self._starts_by_channel(self._summary_with_channel())
        chan = list(starts)
        max_distance = int(10 * max_distance_sec)

        scores = pd.DataFrame(index=chan, columns=chan, dtype=float)
        scores.index.name = "Channel"
        scores.columns.name = "Channel"
        for c_index, c_col in product(chan, repeat=2):
            # DANGER: Note how we invert c_col and c_index here. This is because
            # c_index (the index of the dataframe) should be the ground-truth.
            res = compare_detection(starts[c_col], starts[c_index], max_distance)
            scores.loc[c_index, c_col] = res[score]
        return scores

    def compare_detection(self, other, max_distance_sec=0, other_is_groundtruth=True):
        """
        Compare detected events between two detection methods, or against a ground-truth scoring.
        See full documentation in the methods of SpindlesResults and SWResults.
        """
        if isinstance(other, _DetectionResults):
            groundtruth = other._summary_with_channel()
        elif isinstance(other, pd.DataFrame):
            assert "Start" in other.columns
            groundtruth = other.copy()
            if "Channel" not in groundtruth and self._channel_label is not None:
                groundtruth["Channel"] = self._channel_label
            assert "Channel" in groundtruth.columns
        else:
            raise ValueError(
                f"Invalid argument other: {other}. It must be a YASA detection output or a Pandas "
                f"DataFrame with the columns Start and Channels"
            )

        detected = self._starts_by_channel(self._summary_with_channel())
        groundtruth = self._starts_by_channel(groundtruth)
        max_distance = int(10 * max_distance_sec)

        # Find channels that are present in both self and other
        chan_both = np.intersect1d(list(detected), list(groundtruth))  # Sort
        if not len(chan_both):
            raise ValueError(
                f"No intersecting channel between self and other:\n"
                f"{list(detected)}\n{list(groundtruth)}"
            )

        # The output is a pandas.DataFrame (n_chan, n_metrics).
        rows = []
        for chan in chan_both:
            idx_detected, idx_groundtruth = detected[chan], groundtruth[chan]
            if other_is_groundtruth:
                res = compare_detection(idx_detected, idx_groundtruth, max_distance)
            else:
                res = compare_detection(idx_groundtruth, idx_detected, max_distance)
            rows.append(
                {
                    "Channel": chan,
                    "precision": float(res["precision"]),
                    "recall": float(res["recall"]),
                    "f1": float(res["f1"]),
                    "n_self": len(idx_detected),
                    "n_other": len(idx_groundtruth),
                }
            )
        return pd.DataFrame(rows).set_index("Channel")

    def plot_average(
        self,
        center="Peak",
        hue="Channel",
        time_before=1,
        time_after=1,
        filt=(None, None),
        mask=None,
        figsize=(6, 4.5),
        **kwargs,
    ):
        """Plot the average event"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        df_sync = self.get_sync_events(
            center=center, time_before=time_before, time_after=time_after, filt=filt, mask=mask
        )
        assert not df_sync.empty, "Could not calculate event-locked data."
        assert hue in ["Stage", "Channel"], "hue must be 'Channel' or 'Stage'"
        assert hue in df_sync.columns, "%s is not present in data." % hue

        # Translate deprecated seaborn ci= kwarg to errorbar= (seaborn >= 0.12)
        if "ci" in kwargs:
            ci_val = kwargs.pop("ci")
            if "errorbar" not in kwargs:
                kwargs["errorbar"] = None if ci_val is None else ("ci", ci_val)
        _, ax = plt.subplots(1, 1, figsize=figsize)
        sns.lineplot(data=df_sync, x="Time", y="Amplitude", hue=hue, ax=ax, **kwargs)
        ax.set_xlim(df_sync["Time"].min(), df_sync["Time"].max())
        ax.set_title(self._title)
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Amplitude (uV)")
        return ax

    def plot_detection(self):
        """Plot an overlay of the detected events on the signal."""
        import ipywidgets as ipy
        import matplotlib.pyplot as plt

        # Define mask
        sf = self._sf
        win_size = 10
        mask = self.get_mask()
        highlight = self._data * mask
        highlight = np.where(highlight == 0, np.nan, highlight)
        highlight_filt = self._data_filt * mask
        highlight_filt = np.where(highlight_filt == 0, np.nan, highlight_filt)

        n_epochs = int((self._data.shape[-1] / sf) / win_size)
        times = np.arange(self._data.shape[-1]) / sf

        # Define xlim and xrange
        xlim = [0, win_size]
        xrng = np.arange(xlim[0] * sf, (xlim[1] * sf + 1), dtype=int)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 4))
        plt.plot(times[xrng], self._data[0, xrng], "k", lw=1)
        plt.plot(times[xrng], highlight[0, xrng], "indianred")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude (uV)")
        fig.canvas.header_visible = False
        fig.tight_layout()

        # WIDGETS
        layout = ipy.Layout(width="50%", justify_content="center", align_items="center")

        sl_ep = ipy.IntSlider(
            min=0,
            max=n_epochs,
            step=1,
            value=0,
            layout=layout,
            description="Epoch:",
        )

        sl_amp = ipy.IntSlider(
            min=25,
            max=500,
            step=25,
            value=150,
            layout=layout,
            orientation="horizontal",
            description="Amplitude:",
        )

        dd_ch = ipy.Dropdown(
            options=self._ch_names, value=self._ch_names[0], description="Channel:"
        )

        dd_win = ipy.Dropdown(
            options=[1, 5, 10, 30, 60],
            value=win_size,
            description="Window size:",
        )

        dd_check = ipy.Checkbox(
            value=False,
            description="Filtered",
        )

        def update(epoch, amplitude, channel, win_size, filt):
            """Update plot."""
            n_epochs = int((self._data.shape[-1] / sf) / win_size)
            sl_ep.max = n_epochs
            xlim = [epoch * win_size, (epoch + 1) * win_size]
            xrng = np.arange(xlim[0] * sf, (xlim[1] * sf), dtype=int)
            # Check if filtered
            data = self._data if not filt else self._data_filt
            overlay = highlight if not filt else highlight_filt
            try:
                ax.lines[0].set_data(times[xrng], data[dd_ch.index, xrng])
                ax.lines[1].set_data(times[xrng], overlay[dd_ch.index, xrng])
                ax.set_xlim(xlim)
            except IndexError:
                pass
            ax.set_ylim([-amplitude, amplitude])

        return ipy.interact(
            update, epoch=sl_ep, amplitude=sl_amp, channel=dd_ch, win_size=dd_win, filt=dd_check
        )


#############################################################################
# SPINDLES DETECTION
#############################################################################


@_restore_log_level
def spindles_detect(
    data,
    sf=None,
    ch_names=None,
    hypno=None,
    include=(1, 2, 3),
    freq_sp=(12, 15),
    freq_broad=(1, 30),
    duration=(0.5, 2),
    min_distance=500,
    thresh={"rel_pow": 0.2, "corr": 0.65, "rms": 1.5},
    multi_only=False,
    remove_outliers=False,
    verbose=False,
):
    """Spindles detection.

    Parameters
    ----------
    data : array_like or :py:class:`mne.io.BaseRaw`
        Single or multi-channel data. If ``data`` is *array_like*, unit must be uV and of
        shape (n_samples) or (n_chan, n_samples). If ``data`` is a :py:class:`~mne.io.BaseRaw`
        instance, ``data``, ``sf``, and ``ch_names`` will be automatically extracted, and
        ``data`` will be automatically converted from Volts (MNE) to micro-Volts (YASA).
    sf : float
        Sampling frequency of the data in Hz when ``data`` is *array_like*.
        Should be omitted if ``data`` is a :py:class:`~mne.io.BaseRaw` instance.

        .. tip:: If the detection is taking too long, make sure to downsample
            your data to 100 Hz (or 128 Hz). For more details, please refer to
            :py:func:`mne.filter.resample`.
    ch_names : list of str
        Channel names if ``data`` is *array_like*.
        Should be omitted if ``data`` is a :py:class:`~mne.io.BaseRaw` instance.
    hypno : array_like or :py:class:`yasa.Hypnogram`
        Sleep stage (hypnogram). If the hypnogram is loaded, the
        detection will only be applied to the value defined in
        ``include`` (default = N1 + N2 + N3 sleep).

        Can be an upsampled integer array (same number of samples as ``data``)
        or a :py:class:`yasa.Hypnogram` instance (automatically upsampled).
        To manually upsample an integer array, use
        :py:meth:`yasa.Hypnogram.upsample_to_data`.

        .. note::
            When passing an integer array, hypnogram values follow this mapping:

            - -2 = Unscored
            - -1 = Artefact / Movement
            - 0 = Wake
            - 1 = N1 sleep
            - 2 = N2 sleep
            - 3 = N3 sleep
            - 4 = REM sleep
    include : tuple, list or int or str
        Values in ``hypno`` that will be included in the mask. The default is
        (1, 2, 3), meaning that the detection is applied on N1, N2 and N3
        sleep. This has no effect when ``hypno`` is None.

        When ``hypno`` is a :py:class:`yasa.Hypnogram`, string labels can be
        used instead of integers (e.g. ``["N1", "N2", "N3"]``).
    freq_sp : tuple or list
        Spindles frequency range. Default is 12 to 15 Hz. Please note that YASA
        uses a FIR filter (implemented in MNE) with a 1.5Hz transition band,
        which means that for `freq_sp = (12, 15 Hz)`, the -6 dB points are
        located at 11.25 and 15.75 Hz.
    freq_broad : tuple or list
        Broad band frequency range. Default is 1 to 30 Hz.
    duration : tuple or list
        The minimum and maximum duration of the spindles.
        Default is 0.5 to 2 seconds.
    min_distance : int
        If two spindles are closer than ``min_distance`` (in ms), they are
        merged into a single spindles. Default is 500 ms.
    thresh : dict
        Detection thresholds:

        * ``'rel_pow'``: Relative power (= power ratio freq_sp / freq_broad).
        * ``'corr'``: Moving correlation between original signal and
          sigma-filtered signal.
        * ``'rms'``: Number of standard deviations above the mean of a moving
          root mean square of sigma-filtered signal.

        You can disable one or more threshold by putting ``None`` instead:

        .. code-block:: python

            thresh = {'rel_pow': None, 'corr': 0.65, 'rms': 1.5}
            thresh = {'rel_pow': None, 'corr': None, 'rms': 3}
    multi_only : boolean
        Define the behavior of the multi-channel detection. If True, only
        spindles that are present on at least two channels are kept. If False,
        no selection is applied and the output is just a concatenation of the
        single-channel detection dataframe. Default is False.
    remove_outliers : boolean
        If True, YASA will automatically detect and remove outliers spindles
        using :py:class:`sklearn.ensemble.IsolationForest`.
        The outliers detection is performed on all the spindles
        parameters with the exception of the ``Start``, ``Peak``, ``End``,
        ``Stage``, and ``SOPhase`` columns.
        YASA uses a random seed (42) to ensure reproducible results.
        Note that this step will only be applied if there are more than 50
        detected spindles in the first place. Default to False.
    verbose : bool or str
        Verbose level. Default (False) will only print warning and error
        messages. The logging levels are 'debug', 'info', 'warning', 'error',
        and 'critical'. For most users the choice is between 'info'
        (or ``verbose=True``) and warning (``verbose=False``).

        .. versionadded:: 0.2.0

    Returns
    -------
    sp : :py:class:`yasa.SpindlesResults`
        To get the full detection dataframe, use:

        >>> sp = spindles_detect(...)  # doctest: +SKIP
        >>> sp.summary()  # doctest: +SKIP

        This will give a :py:class:`pandas.DataFrame` where each row is a
        detected spindle and each column is a parameter (= feature or property)
        of this spindle. To get the average spindles parameters per channel and
        sleep stage:

        >>> sp.summary(grp_chan=True, grp_stage=True)  # doctest: +SKIP

    Notes
    -----
    The parameters that are calculated for each spindle are:

    * ``'Start'``: Start time of the spindle, in seconds from the beginning of
      data.
    * ``'Peak'``: Time at the most prominent spindle peak (in seconds).
    * ``'End'`` : End time (in seconds).
    * ``'Duration'``: Duration (in seconds)
    * ``'Amplitude'``: Peak-to-peak amplitude of the (detrended) spindle in
      the broadband-filtered data (in µV).
    * ``'AmpFiltered'``: Peak-to-peak amplitude of the spindle in the
      sigma-band filtered data (in µV).
    * ``'RMS'``: Root-mean-square (in µV)
    * ``'AbsPower'``: Median absolute power (in log10 µV^2),
      calculated from the Hilbert-transform of the ``freq_sp`` filtered signal.
    * ``'RelPower'``: Median relative power of the ``freq_sp`` band in spindle
      calculated from a short-term fourier transform and expressed as a
      proportion of the total power in ``freq_broad``.
    * ``'Frequency'``: Median instantaneous frequency of spindle (in Hz),
      derived from an Hilbert transform of the ``freq_sp`` filtered signal.
    * ``'Oscillations'``: Number of oscillations (= number of positive peaks
      in spindle.)
    * ``'Symmetry'``: Location of the most prominent peak of spindle,
      normalized from 0 (start) to 1 (end). Ideally this value should be close
      to 0.5, indicating that the most prominent peak is halfway through the
      spindle.
    * ``'Stage'`` : Sleep stage during which spindle occured, if ``hypno``
      was provided.

      All parameters are calculated from the broadband-filtered EEG
      (frequency range defined in ``freq_broad``).

    For better results, apply this detection only on artefact-free NREM sleep.

    .. warning::
        A critical bug was fixed in YASA 0.6.1, in which the number of detected spindles could
        vary drastically depending on the sampling frequency of the data. Please make sure to check
        any results obtained with this function prior to the 0.6.1 release.


    References
    ----------
    The sleep spindles detection algorithm is based on:

    * Lacourse, K., Delfrate, J., Beaudry, J., Peppard, P., & Warby, S. C.
      (2018). `A sleep spindle detection algorithm that emulates human expert
      spindle scoring. <https://doi.org/10.1016/j.jneumeth.2018.08.014>`_
      Journal of Neuroscience Methods.

    Examples
    --------
    1. Detect spindles on an MNE Raw object with an upsampled integer hypnogram (legacy):

    .. code-block:: python

        >>> import yasa
        >>> sp = yasa.spindles_detect(raw, hypno=hypno_up, include=(1, 2, 3))  # doctest: +SKIP

    2. Pass a :py:class:`~yasa.Hypnogram` directly — upsampling and stage filtering are
       handled automatically. String stage labels can be used for ``include``:

    .. code-block:: python

        >>> hyp = yasa.Hypnogram(["W", "N1", "N2", "N2", "N3", "REM"], freq="30s")
        >>> sp = yasa.spindles_detect(raw, hypno=hyp, include=["N1", "N2", "N3"])  # doctest: +SKIP

    For a full walkthrough, please refer to the following Jupyter notebooks:

    https://github.com/raphaelvallat/yasa/blob/master/notebooks/01_spindles_detection.ipynb

    https://github.com/raphaelvallat/yasa/blob/master/notebooks/02_spindles_detection_multi.ipynb

    https://github.com/raphaelvallat/yasa/blob/master/notebooks/03_spindles_detection_NREM_only.ipynb

    https://github.com/raphaelvallat/yasa/blob/master/notebooks/04_spindles_slow_fast.ipynb
    """

    (data, sf, ch_names, hypno, include, mask, n_chan, n_samples, bad_chan) = _check_data_hypno(
        data, sf, ch_names, hypno, include, verbose=verbose
    )

    # If all channels are bad
    if sum(bad_chan) == n_chan:
        logger.warning("All channels have bad amplitude. Returning None.")
        return None

    # Check detection thresholds. Missing keys are set to their default value.
    thresh = {"rel_pow": 0.20, "corr": 0.65, "rms": 1.5, **thresh}
    do_rel_pow = thresh["rel_pow"] not in [None, "none", "None"]
    do_corr = thresh["corr"] not in [None, "none", "None"]
    do_rms = thresh["rms"] not in [None, "none", "None"]
    n_thresh = sum([do_rel_pow, do_corr, do_rms])
    assert n_thresh >= 1, "At least one threshold must be defined."

    # Filtering
    nfast = next_fast_len(n_samples)
    # 1) Broadband bandpass filter (optional -- careful of lower freq for PAC)
    data_broad = filter_data(data, sf, freq_broad[0], freq_broad[1], method="fir", verbose=False)
    # 2) Sigma bandpass filter
    # The width of the transition band is set to 1.5 Hz on each side,
    # meaning that for freq_sp = (12, 15 Hz), the -6 dB points are located at
    # 11.25 and 15.75 Hz.
    data_sigma = filter_data(
        data,
        sf,
        freq_sp[0],
        freq_sp[1],
        l_trans_bandwidth=1.5,
        h_trans_bandwidth=1.5,
        method="fir",
        verbose=False,
    )

    # Number of oscillations (number of peaks separated by at least 60 ms)
    # --> 60 ms because 1000 ms / 16 Hz = 62.5 m, in other words, at 16 Hz,
    # peaks are separated by 62.5 ms. At 11 Hz peaks are separated by 90 ms
    distance = 60 * sf / 1000

    # Collect per-channel DataFrames, then concat once at the end
    dfs = []

    for i in range(n_chan):
        # ####################################################################
        # START SINGLE CHANNEL DETECTION
        # ####################################################################

        # First, skip channels with bad data amplitude
        if bad_chan[i]:
            continue

        # Hilbert power (to define the instantaneous frequency / power). This is done one channel
        # at a time to limit memory usage.
        analytic = signal.hilbert(data_sigma[i, :], N=nfast)[:n_samples]
        inst_pow = analytic.real**2 + analytic.imag**2
        inst_freq = sf / (2 * np.pi) * np.diff(np.angle(analytic))

        # Compute the relative sigma power using the STFT (step=200 ms, no interp).
        # rel_pow_coarse holds one value per STFT frame and is used directly for
        # per-spindle RelPow extraction, avoiding a full-resolution interpolation.
        f, t_stft, Sxx = stft_power(
            data_broad[i, :], sf, window=2, step=0.2, band=freq_broad, interp=False, norm=False
        )
        idx_sigma = np.logical_and(f >= freq_sp[0], f <= freq_sp[1])
        rel_pow_coarse = Sxx[idx_sigma].sum(0) / Sxx.sum(0)

        # Full-resolution interpolation is only needed when rel_pow is used as a
        # detection threshold. Linear interpolation is fast and sufficient here.
        if do_rel_pow:
            rel_pow = np.interp(np.arange(n_samples) / sf, t_stft, rel_pow_coarse, left=0, right=0)

        if do_corr:
            _, mcorr = moving_transform(
                x=data_sigma[i, :],
                y=data_broad[i, :],
                sf=sf,
                window=0.3,
                step=0.1,
                method="corr",
                interp=True,
            )
        if do_rms:
            _, mrms = moving_transform(
                x=data_sigma[i, :], sf=sf, window=0.3, step=0.1, method="rms", interp=True
            )
            # Let's define the thresholds (mask is all True if there is no hypnogram)
            thresh_rms = mrms[mask].mean() + thresh["rms"] * trimbothstd(mrms[mask], cut=0.10)
            # Avoid too high threshold caused by Artefacts / Motion during Wake
            thresh_rms = min(thresh_rms, 10)
            logger.info("Moving RMS threshold = %.3f", thresh_rms)

        # Number of supra-threshold detection methods at each sample
        idx_sum = np.zeros(n_samples)
        if do_rel_pow:
            idx_rel_pow = rel_pow >= thresh["rel_pow"]
            idx_sum += idx_rel_pow
            logger.info("N supra-theshold relative power = %i", idx_rel_pow.sum())
        if do_corr:
            idx_mcorr = mcorr >= thresh["corr"]
            idx_sum += idx_mcorr
            logger.info("N supra-theshold moving corr = %i", idx_mcorr.sum())
        if do_rms:
            idx_mrms = mrms >= thresh_rms
            idx_sum += idx_mrms
            logger.info("N supra-theshold moving RMS = %i", idx_mrms.sum())

        # Make sure that we do not detect spindles outside mask
        idx_sum[~mask] = 0

        # The detection using the three thresholds tends to underestimate the
        # real duration of the spindle. To overcome this, we compute a soft
        # threshold by smoothing the idx_sum vector with a ~100 ms window.
        # Sampling frequency = 100 Hz --> w = 10 samples
        # Sampling frequecy = 256 Hz --> w = 25 samples = 97 ms
        w = int(0.1 * sf)
        # Critical bugfix March 2022, see https://github.com/raphaelvallat/yasa/pull/55
        idx_sum = np.convolve(idx_sum, np.ones(w), mode="same") / w
        # And we then find indices that are strictly greater than 2, i.e. we
        # find the 'true' beginning and 'true' end of the events by finding
        # where at least two out of the three treshold were crossed.
        where_sp = np.where(idx_sum > (n_thresh - 1))[0]

        # If no events are found, skip to next channel
        if not len(where_sp):
            logger.warning("No spindle were found in channel %s.", ch_names[i])
            continue

        # Merge events that are too close
        if min_distance is not None and min_distance > 0:
            where_sp = _merge_close(where_sp, min_distance, sf)

        # Extract start, end, and duration of each spindle
        sp = np.split(where_sp, np.where(np.diff(where_sp) != 1)[0] + 1)
        idx_start_end = np.array([[k[0], k[-1]] for k in sp]) / sf
        sp_start, sp_end = idx_start_end.T
        sp_dur = sp_end - sp_start

        # Keep only the events with a good duration
        good_dur = np.logical_and(sp_dur > duration[0], sp_dur < duration[1])
        if not good_dur.any():
            logger.warning("No spindle were found in channel %s.", ch_names[i])
            continue
        sp = [sp[j] for j in np.flatnonzero(good_dur)]
        sp_start, sp_end, sp_dur = sp_start[good_dur], sp_end[good_dur], sp_dur[good_dur]

        # Initialize empty variables
        n_sp = len(sp)
        sp_amp = np.zeros(n_sp)
        sp_amp_filt = np.zeros(n_sp)
        sp_freq = np.zeros(n_sp)
        sp_rms = np.zeros(n_sp)
        sp_osc = np.zeros(n_sp)
        sp_sym = np.zeros(n_sp)
        sp_abs = np.zeros(n_sp)
        sp_rel = np.zeros(n_sp)
        sp_pro = np.zeros(n_sp)

        for j, idx_sp in enumerate(sp):
            # Important: detrend the signal to avoid wrong PTP amplitude
            sp_det = _detrend_linear(data_broad[i, idx_sp])
            sp_amp[j] = np.ptp(sp_det)  # Peak-to-peak amplitude
            sp_amp_filt[j] = np.ptp(data_sigma[i, idx_sp])  # Amplitude on sigma-filtered signal
            sp_rms[j] = np.sqrt(np.mean(sp_det**2))  # Root mean square
            # Median relative power from the coarse STFT grid (avoids
            # indexing a full-resolution interpolated array).
            idx_start = np.searchsorted(t_stft, sp_start[j], side="left")
            idx_end = np.searchsorted(t_stft, sp_end[j], side="right")
            if idx_start < idx_end:
                # At least one STFT frame falls within the spindle
                sp_rel[j] = np.median(rel_pow_coarse[idx_start:idx_end])
            else:  # pragma: no cover
                # Spindle shorter than one STFT step (only with a custom duration): use the
                # nearest frame
                sp_mid = 0.5 * (sp_start[j] + sp_end[j])
                sp_rel[j] = rel_pow_coarse[np.abs(t_stft - sp_mid).argmin()]

            # Hilbert-based instantaneous properties
            sp_inst_freq = inst_freq[idx_sp]
            sp_inst_pow = inst_pow[idx_sp]
            sp_abs[j] = np.median(np.log10(sp_inst_pow[sp_inst_pow > 0]))
            sp_freq[j] = np.median(sp_inst_freq[sp_inst_freq > 0])

            # Number of oscillations
            peaks, peaks_params = signal.find_peaks(
                sp_det, distance=distance, prominence=(None, None)
            )
            sp_osc[j] = len(peaks)

            # Peak location & symmetry index
            # pk is expressed in sample since the beginning of the spindle
            pk = peaks[peaks_params["prominences"].argmax()]
            sp_pro[j] = sp_start[j] + pk / sf
            sp_sym[j] = pk / sp_det.size

        # Create a dataframe
        sp_params = {
            "Start": sp_start,
            "Peak": sp_pro,
            "End": sp_end,
            "Duration": sp_dur,
            "Amplitude": sp_amp,
            "AmpFiltered": sp_amp_filt,
            "RMS": sp_rms,
            "AbsPower": sp_abs,
            "RelPower": sp_rel,
            "Frequency": sp_freq,
            "Oscillations": sp_osc,
            "Symmetry": sp_sym,
        }
        if hypno is not None:
            # Sleep stage at the start of the spindle
            sp_params["Stage"] = hypno[[k[0] for k in sp]]
        df_chan = pd.DataFrame(sp_params)

        if remove_outliers:
            df_chan = _remove_outliers(df_chan, SpindlesResults._features, ch_names[i])

        # ####################################################################
        # END SINGLE CHANNEL DETECTION
        # ####################################################################
        df_chan["Channel"] = ch_names[i]
        df_chan["IdxChannel"] = i
        dfs.append(df_chan)

    # If no spindles were detected, return None
    if not dfs:
        logger.warning("No spindles were found in data. Returning None.")
        return None
    df = pd.concat(dfs, axis=0, ignore_index=True)

    # Find spindles that are present on at least two channels
    if multi_only and df["Channel"].nunique() > 1:
        # We round to the nearest second
        idx_good = np.logical_or(
            df["Start"].round(0).duplicated(keep=False), df["End"].round(0).duplicated(keep=False)
        )
        df = df[idx_good].reset_index(drop=True)

    return SpindlesResults(
        events=df, data=data, sf=sf, ch_names=ch_names, hypno=hypno, data_filt=data_sigma
    )


class SpindlesResults(_DetectionResults):
    """Output class for spindles detection.

    Attributes
    ----------
    _events : :py:class:`pandas.DataFrame`
        Output detection dataframe
    _data : array_like
        Original EEG data of shape *(n_chan, n_samples)*.
    _data_filt : array_like
        Sigma-filtered EEG data of shape *(n_chan, n_samples)*.
    _sf : float
        Sampling frequency of data.
    _ch_names : list
        Channel names.
    _hypno : array_like or None
        Sleep staging vector.
    """

    _features = (
        "Duration",
        "Amplitude",
        "AmpFiltered",
        "RMS",
        "AbsPower",
        "RelPower",
        "Frequency",
        "Oscillations",
        "Symmetry",
    )
    _title = "Average spindle"

    def summary(self, grp_chan=False, grp_stage=False, mask=None, aggfunc="mean", sort=True):
        """Return a summary of the spindles detection, optionally grouped
        across channels and/or stage.

        Parameters
        ----------
        grp_chan : bool
            If True, group by channel (for multi-channels detection only).
        grp_stage : bool
            If True, group by sleep stage (provided that an hypnogram was
            used).
        mask : array_like or None
            Custom boolean mask. Only the detected events for which mask is True will be
            included in the summary dataframe. Default is None, i.e. no masking
            (all events are included).
        aggfunc : str or function
            Averaging function (e.g. ``'mean'`` or ``'median'``).
        sort : bool
            If True, sort group keys when grouping.
        """
        return super().summary(
            grp_chan=grp_chan,
            grp_stage=grp_stage,
            aggfunc=aggfunc,
            sort=sort,
            mask=mask,
        )

    def get_coincidence_matrix(self, scaled=True):
        """Return the (scaled) coincidence matrix.

        Parameters
        ----------
        scaled : bool
            If True (default), the coincidence matrix is scaled (see Notes).

        Returns
        -------
        coincidence : pd.DataFrame
            A symmetric matrix with the (scaled) coincidence values.

        Notes
        -----
        Do spindles occur at the same time? One way to measure this is to
        calculate the coincidence matrix, which gives, for each pair of
        channel, the number of samples that were marked as a spindle in both
        channels. The output is a symmetric matrix, in which the diagonal is
        simply the number of data points that were marked as a spindle in the
        channel.

        The coincidence matrix can be scaled (default) by dividing the output
        by the product of the sum of each individual binary mask, as shown in
        the example below. It can then be used to define functional
        networks or quickly find outlier channels.

        Examples
        --------
        Calculate the coincidence of two binary mask:

        >>> import numpy as np
        >>> x = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1])
        >>> y = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1])
        >>> x * y
        array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1])

        >>> int((x * y).sum())  # Unscaled coincidence
        3

        >>> float((x * y).sum() / (x.sum() * y.sum()))  # Scaled coincidence
        0.12

        References
        ----------
        - https://github.com/Mark-Kramer/Sleep-Networks-2021
        """
        return super().get_coincidence_matrix(scaled=scaled)

    def compare_channels(self, score="f1", max_distance_sec=0):
        """
        Compare detected spindles across channels.

        This is a wrapper around the :py:func:`yasa.compare_detection` function. Please
        refer to the documentation of this function for more details.

        Parameters
        ----------
        score : str
            The performance metric to compute. Accepted values are "precision", "recall"
            (aka sensitivity) and "f1" (default). The F1-score is the harmonic mean of precision
            and recall, and is usually the preferred metric to evaluate the agreement between
            two channels. All three metrics are bounded by 0 and 1, where 1 indicates perfect
            agreement.
        max_distance_sec : float
            The maximum distance between spindles, in seconds, to consider as the same event.

            .. warning:: To reduce computation cost, YASA rounds the start time of each spindle to
                the nearest decisecond (= 100 ms). This means that the lowest possible resolution
                is 100 ms, regardless of the sampling frequency of the data. Two spindles starting
                at 500 ms and 540 ms on their respective channels will therefore always be
                considered the same event, even when max_distance_sec=0.

        Returns
        -------
        scores : :py:class:`pandas.DataFrame`
            A Pandas DataFrame with the output scores, of shape (n_chan, n_chan).

        Notes
        -----
        Some use cases of this function:

        1. What proportion of spindles detected in one channel are also detected on
           another channel (if using ``score="recall"``).
        2. What is the overall agreement in the detected events between channels?
        3. Is the agreement better in channels that are close to one another?
        """
        return super().compare_channels(score, max_distance_sec)

    def compare_detection(self, other, max_distance_sec=0, other_is_groundtruth=True):
        """
        Compare the detected spindles against either another YASA detection or against custom
        annotations (e.g. ground-truth human scoring).

        This function is a wrapper around the :py:func:`yasa.compare_detection` function. Please
        refer to the documentation of this function for more details.

        Parameters
        ----------
        other : dataframe or detection results
            This can be either a) the output of another YASA detection, for example if you want to
            test the impact of tweaking some parameters on the detected events or b) a pandas
            DataFrame with custom annotations, obtained by another detection method outside
            of YASA, or with manual labelling. If b), the dataframe must contain the "Start" and
            "Channel" columns, with the start of each event in seconds from the beginning
            of the recording and the channel name, respectively. The channel names should match
            the output of the summary() method.
        max_distance_sec : float
            The maximum distance between spindles, in seconds, to consider as the same event.

            .. warning:: To reduce computation cost, YASA rounds the start time of each spindle to
                the nearest decisecond (= 100 ms). This means that the lowest possible resolution
                is 100 ms, regardless of the sampling frequency of the data.
        other_is_groundtruth : bool
            If True (default), ``other`` will be considered as the ground-truth scoring. If False,
            the current detection will be considered as the ground-truth, and the precision and
            recall scores will be inverted. This parameter has no effect on the F1-score.

            .. note:: when ``other`` is the ground-truth (default), the recall score is the
                fraction of events in other that were succesfully detected by the current
                detection, and the precision score is the proportion of detected events by the
                current detection that are also present in other.

        Returns
        -------
        scores : :py:class:`pandas.DataFrame`
            A Pandas DataFrame with the channel names as index, and the following columns

            * ``precision``: Precision score, aka positive predictive value
            * ``recall``: Recall score, aka sensitivity
            * ``f1``: F1-score
            * ``n_self``: Number of detected events in ``self`` (current method).
            * ``n_other``: Number of detected events in ``other``.

        Notes
        -----
        Some use cases of this function:

        1. How well does YASA events detection perform against ground-truth human annotations?
        2. If I change the threshold(s) of the events detection, do the detected events match
           those obtained with the default parameters?
        3. Which detection thresholds give the highest agreement with the ground-truth scoring?
        """
        return super().compare_detection(other, max_distance_sec, other_is_groundtruth)

    def get_mask(self):
        """
        Return a boolean array indicating for each sample in data if this
        sample is part of a detected event (True) or not (False).
        """
        return super().get_mask()

    def get_sync_events(
        self,
        center="Peak",
        time_before=1,
        time_after=1,
        filt=(None, None),
        mask=None,
        as_dataframe=True,
    ):
        """
        Return the raw or filtered data of each detected event after
        centering to a specific timepoint.

        Parameters
        ----------
        center : str
            Landmark of the event to synchronize the timing on.
            Default is to use the center peak of the spindles.
        time_before : float
            Time (in seconds) before ``center``.
        time_after : float
            Time (in seconds) after ``center``.
        filt : tuple
            Optional filtering to apply to data. For instance, ``filt=(1, 30)``
            will apply a 1 to 30 Hz bandpass filter, and ``filt=(None, 40)``
            will apply a 40 Hz lowpass filter. Filtering is done using default
            parameters in the :py:func:`mne.filter.filter_data` function.
        mask : array_like or None
            Custom boolean mask. Only the detected events for which mask is True will be
            included. Default is None, i.e. no masking (all events are included).
        as_dataframe : boolean
            If True (default), returns a long-format pandas dataframe. If False, returns a list of
            numpy arrays. Each element of the list a unique channel, and the shape of the numpy
            arrays within the list is (n_events, n_times).

        Returns
        -------
        df_sync : :py:class:`pandas.DataFrame`
            Ouput long-format dataframe (if ``as_dataframe=True``)::

            'Event' : Event number
            'Time' : Timing of the events (in seconds)
            'Amplitude' : Raw or filtered data for event
            'Channel' : Channel
            'IdxChannel' : Index of channel in data
            'Stage': Sleep stage in which the events occured (if available)
        """
        return super().get_sync_events(
            center=center,
            time_before=time_before,
            time_after=time_after,
            filt=filt,
            mask=mask,
            as_dataframe=as_dataframe,
        )

    def plot_average(
        self,
        center="Peak",
        hue="Channel",
        time_before=1,
        time_after=1,
        filt=(None, None),
        mask=None,
        figsize=(6, 4.5),
        **kwargs,
    ):
        """
        Plot the average spindle.

        Parameters
        ----------
        center : str
            Landmark of the event to synchronize the timing on.
            Default is to use the most prominent peak of the spindle.
        hue : str
            Grouping variable that will produce lines with different colors.
            Can be either 'Channel' or 'Stage'.
        time_before : float
            Time (in seconds) before ``center``.
        time_after : float
            Time (in seconds) after ``center``.
        filt : tuple
            Optional filtering to apply to data. For instance, ``filt=(12, 16)``
            will apply a 12 to 16 Hz bandpass filter, and ``filt=(None, 40)``
            will apply a 40 Hz lowpass filter. Filtering is done using the default
            parameters in the :py:func:`mne.filter.filter_data` function.
        mask : array_like or None
            Custom boolean mask. Only the detected events for which mask is True will be
            plotted. Default is None, i.e. no masking (all events are included).
        figsize : tuple
            Figure size in inches.
        **kwargs : dict
            Optional argument that are passed to :py:func:`seaborn.lineplot`.
        """
        return super().plot_average(
            center=center,
            hue=hue,
            time_before=time_before,
            time_after=time_after,
            filt=filt,
            mask=mask,
            figsize=figsize,
            **kwargs,
        )

    def plot_detection(self):
        """Plot an overlay of the detected spindles on the EEG signal.

        This only works in Jupyter and it requires the ipywidgets
        (https://ipywidgets.readthedocs.io/en/latest/) package.

        To activate the interactive mode, make sure to run:

        >>> %matplotlib widget  # doctest: +SKIP

        .. versionadded:: 0.4.0
        """
        return super().plot_detection()


#############################################################################
# SLOW-WAVES DETECTION
#############################################################################


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


@_restore_log_level
def sw_detect(
    data,
    sf=None,
    ch_names=None,
    hypno=None,
    include=(2, 3),
    freq_sw=(0.3, 1.5),
    dur_neg=(0.3, 1.5),
    dur_pos=(0.1, 1),
    amp_neg=(40, 200),
    amp_pos=(10, 150),
    amp_ptp=(75, 350),
    coupling=False,
    coupling_params={"freq_sp": (12, 16), "time": 1, "p": 0.05},
    remove_outliers=False,
    verbose=False,
):
    """Slow-waves detection.

    Parameters
    ----------
    data : array_like or :py:class:`mne.io.BaseRaw`
        Single or multi-channel data. If ``data`` is *array_like*, unit must be uV and of
        shape (n_samples) or (n_chan, n_samples). If ``data`` is a :py:class:`~mne.io.BaseRaw`
        instance, ``data``, ``sf``, and ``ch_names`` will be automatically extracted, and
        ``data`` will be automatically converted from Volts (MNE) to micro-Volts (YASA).
    sf : float
        Sampling frequency of the data in Hz if ``data`` is *array_like*.
        Should be omitted if ``data`` is a :py:class:`~mne.io.BaseRaw` instance.

        .. tip:: If the detection is taking too long, make sure to downsample
            your data to 100 Hz (or 128 Hz). For more details, please refer to
            :py:func:`mne.filter.resample`.
    ch_names : list of str
        Channel names if ``data`` is *array_like*.
        Should be omitted if ``data`` is a :py:class:`~mne.io.BaseRaw` instance.
    hypno : array_like or :py:class:`yasa.Hypnogram`
        Sleep stage (hypnogram). If the hypnogram is loaded, the
        detection will only be applied to the value defined in
        ``include`` (default = N2 + N3 sleep).

        Can be an upsampled integer array (same number of samples as ``data``)
        or a :py:class:`yasa.Hypnogram` instance (automatically upsampled).
        To manually upsample an integer array, use
        :py:meth:`yasa.Hypnogram.upsample_to_data`.

        .. note::
            When passing an integer array, hypnogram values follow this mapping:

            - -2 = Unscored
            - -1 = Artefact / Movement
            - 0 = Wake
            - 1 = N1 sleep
            - 2 = N2 sleep
            - 3 = N3 sleep
            - 4 = REM sleep
    include : tuple, list or int or str
        Values in ``hypno`` that will be included in the mask. The default is
        (2, 3), meaning that the detection is applied on N2 and N3
        sleep. This has no effect when ``hypno`` is None.

        When ``hypno`` is a :py:class:`yasa.Hypnogram`, string labels can be
        used instead of integers (e.g. ``["N2", "N3"]``).
    freq_sw : tuple or list
        Slow wave frequency range. Default is 0.3 to 1.5 Hz. Please note that
        YASA uses a FIR filter (implemented in MNE) with a 0.2 Hz transition
        band, which means that the -6 dB points are located at 0.2 and 1.6 Hz.
    dur_neg : tuple or list
        The minimum and maximum duration of the negative deflection of the
        slow wave. Default is 0.3 to 1.5 second.
    dur_pos : tuple or list
        The minimum and maximum duration of the positive deflection of the
        slow wave. Default is 0.1 to 1 second.
    amp_neg : tuple or list
        Absolute minimum and maximum negative trough amplitude of the
        slow-wave. Default is 40 uV to 200 uV. Can also be in unit of standard
        deviations if the data has been previously z-scored. If you do not want
        to specify any negative amplitude thresholds,
        use ``amp_neg=(None, None)``.
    amp_pos : tuple or list
        Absolute minimum and maximum positive peak amplitude of the
        slow-wave. Default is 10 uV to 150 uV. Can also be in unit of standard
        deviations if the data has been previously z-scored.
        If you do not want to specify any positive amplitude thresholds,
        use ``amp_pos=(None, None)``.
    amp_ptp : tuple or list
        Minimum and maximum peak-to-peak amplitude of the slow-wave.
        Default is 75 uV to 350 uV. Can also be in unit of standard
        deviations if the data has been previously z-scored.
        Use ``np.inf`` to set no upper amplitude threshold
        (e.g. ``amp_ptp=(75, np.inf)``).
    coupling : boolean
        If True, YASA will also calculate the phase-amplitude coupling between
        the slow-waves phase and the spindles-related sigma band
        amplitude. Specifically, the following columns will be added to the
        output dataframe:

        1. ``'SigmaPeak'``: The location (in seconds) of the maximum sigma peak amplitude within a
           2-seconds epoch centered around the negative peak (through) of the current slow-wave.

        2. ``PhaseAtSigmaPeak``: the phase of the bandpas-filtered slow-wave signal (in radians)
           at ``'SigmaPeak'``.

           Importantly, since ``PhaseAtSigmaPeak`` is expressed in radians, one should use circular
           statistics to calculate the mean direction and vector length:

           .. code-block:: python

               import pingouin as pg

               mean_direction = pg.circ_mean(sw["PhaseAtSigmaPeak"])
               vector_length = pg.circ_r(sw["PhaseAtSigmaPeak"])

        3. ``ndPAC``: the normalized Mean Vector Length (also called the normalized direct PAC,
           or ndPAC) within a 2-sec epoch centered around the negative peak of the slow-wave.

        The lower and upper frequencies for the slow-waves and spindles-related sigma signals are
        defined in ``freq_sw`` and ``coupling_params['freq_sp']``, respectively.
        For more details, please refer to the `Jupyter notebook
        <https://github.com/raphaelvallat/yasa/blob/master/notebooks/12_SO-sigma_coupling.ipynb>`_

        Note that setting ``coupling=True`` may increase computation time.

        .. versionadded:: 0.2.0

    coupling_params : dict
        Parameters for the phase-amplitude coupling.

        * ``freq_sp`` is a tuple or list that defines the spindles-related frequency of interest.
          The default is 12 to 16 Hz, with a wide transition bandwidth of 1.5 Hz.

        * ``time`` is an int or a float that defines the time around the negative peak of each
          detected slow-waves, in seconds. For example, a value of 1 means that the coupling will
          be calculated for each slow-waves using a 2-seconds epoch centered around the negative
          peak of the slow-waves (i.e. 1 second on each side).

        * ``p`` is the p-value used for thresholding of unreliable coupling values (ndPAC).
          Sub-threshold PAC values will be set to 0. To disable this behavior (no masking),
          use ``p=1`` or ``p=None``.

        .. versionadded:: 0.6.0

    remove_outliers : boolean
        If True, YASA will automatically detect and remove outliers slow-waves
        using :py:class:`sklearn.ensemble.IsolationForest`.
        The outliers detection is performed on the frequency, amplitude and
        duration parameters of the detected slow-waves. YASA uses a random seed
        (42) to ensure reproducible results. Note that this step will only be
        applied if there are more than 50 detected slow-waves in the first
        place. Default to False.
    verbose : bool or str
        Verbose level. Default (False) will only print warning and error
        messages. The logging levels are 'debug', 'info', 'warning', 'error',
        and 'critical'. For most users the choice is between 'info'
        (or ``verbose=True``) and warning (``verbose=False``).

        .. versionadded:: 0.2.0

    Returns
    -------
    sw : :py:class:`yasa.SWResults`
        To get the full detection dataframe, use:

        >>> sw = sw_detect(...)  # doctest: +SKIP
        >>> sw.summary()  # doctest: +SKIP

        This will give a :py:class:`pandas.DataFrame` where each row is a
        detected slow-wave and each column is a parameter (= property).
        To get the average SW parameters per channel and sleep stage:

        >>> sw.summary(grp_chan=True, grp_stage=True)  # doctest: +SKIP

    Notes
    -----
    The parameters that are calculated for each slow-wave are:

    * ``'Start'``: Start time of each detected slow-wave, in seconds from the beginning of data.
    * ``'NegPeak'``: Location of the negative peak (in seconds)
    * ``'MidCrossing'``: Location of the negative-to-positive zero-crossing (in seconds)
    * ``'Pospeak'``: Location of the positive peak (in seconds)
    * ``'End'``: End time(in seconds)
    * ``'Duration'``: Duration (in seconds)
    * ``'ValNegPeak'``: Amplitude of the negative peak (in uV, calculated on the ``freq_sw``
      bandpass-filtered signal)
    * ``'ValPosPeak'``: Amplitude of the positive peak (in uV, calculated on the ``freq_sw``
      bandpass-filtered signal)
    * ``'PTP'``: Peak-to-peak amplitude (= ``ValPosPeak`` - ``ValNegPeak``, calculated on the
      ``freq_sw`` bandpass-filtered signal)
    * ``'Slope'``: Slope between ``NegPeak`` and ``MidCrossing`` (in uV/sec, calculated on the
      ``freq_sw`` bandpass-filtered signal)
    * ``'Frequency'``: Frequency of the slow-wave (= 1 / ``Duration``)
    * ``'SigmaPeak'``: Location of the sigma peak amplitude within a 2-sec epoch centered around
      the negative peak of the slow-wave. This is only calculated when ``coupling=True``.
    * ``'PhaseAtSigmaPeak'``: SW phase at max sigma amplitude within a 2-sec epoch centered around
      the negative peak of the slow-wave. This is only calculated when ``coupling=True``
    * ``'ndPAC'``: Normalized direct PAC within a 2-sec epoch centered around the negative peak
      of the slow-wave. This is only calculated when ``coupling=True``
    * ``'Stage'``: Sleep stage (only if hypno was provided)

    .. image:: https://raw.githubusercontent.com/raphaelvallat/yasa/refs/tags/v0.6.5/docs/pictures/slow_waves.png  # noqa
      :width: 500px
      :align: center
      :alt: slow-wave

    For better results, apply this detection only on artefact-free NREM sleep.

    References
    ----------
    The slow-waves detection algorithm is based on:

    * Massimini, M., Huber, R., Ferrarelli, F., Hill, S., & Tononi, G. (2004). `The sleep slow
      oscillation as a traveling wave. <https://doi.org/10.1523/JNEUROSCI.1318-04.2004>`_. The
      Journal of Neuroscience, 24(31), 6862–6870.

    * Carrier, J., Viens, I., Poirier, G., Robillard, R., Lafortune, M., Vandewalle, G., Martin,
      N., Barakat, M., Paquet, J., & Filipini, D. (2011). `Sleep slow wave changes during the
      middle years of life. <https://doi.org/10.1111/j.1460-9568.2010.07543.x>`_. The European
      Journal of Neuroscience, 33(4), 758–766.

    Examples
    --------
    1. Detect slow-waves on an MNE Raw object with an upsampled integer hypnogram (legacy):

    .. code-block:: python

        >>> import yasa
        >>> sw = yasa.sw_detect(raw, hypno=hypno_up, include=(2, 3))  # doctest: +SKIP

    2. Pass a :py:class:`~yasa.Hypnogram` directly — upsampling and stage filtering are
       handled automatically. String stage labels can be used for ``include``:

    .. code-block:: python

        >>> hyp = yasa.Hypnogram(["W", "N1", "N2", "N2", "N3", "REM"], freq="30s")
        >>> sw = yasa.sw_detect(raw, hypno=hyp, include=["N2", "N3"])  # doctest: +SKIP

    For a full walkthrough, please refer to the tutorial:
    https://github.com/raphaelvallat/yasa/blob/master/notebooks/05_sw_detection.ipynb
    """

    (data, sf, ch_names, hypno, include, mask, n_chan, n_samples, bad_chan) = _check_data_hypno(
        data, sf, ch_names, hypno, include, verbose=verbose
    )

    # If all channels are bad
    if sum(bad_chan) == n_chan:
        logger.warning("All channels have bad amplitude. Returning None.")
        return None

    # Bandpass filter
    data_filt = filter_data(
        data,
        sf,
        freq_sw[0],
        freq_sw[1],
        method="fir",
        verbose=False,
        l_trans_bandwidth=0.2,
        h_trans_bandwidth=0.2,
    )

    # Extract the spindles-related sigma signal for coupling
    if coupling:
        # Missing keys are set to their default value.
        assert isinstance(coupling_params, dict)
        coupling_params = {"freq_sp": (12, 16), "time": 1, "p": 0.05, **coupling_params}
        # The width of the transition band is set to 1.5 Hz on each side,
        # meaning that for freq_sp = (12, 15 Hz), the -6 dB points are located
        # at 11.25 and 15.75 Hz. The frequency band for the amplitude signal
        # must be large enough to fit the sidebands caused by the assumed
        # modulating lower frequency band (Aru et al. 2015).
        # https://doi.org/10.1016/j.conb.2014.08.002
        freq_sp = coupling_params["freq_sp"]
        data_sp = filter_data(
            data,
            sf,
            freq_sp[0],
            freq_sp[1],
            method="fir",
            l_trans_bandwidth=1.5,
            h_trans_bandwidth=1.5,
            verbose=False,
        )
        nfast = next_fast_len(n_samples)
        # Epoch around the negative peak of each slow-wave
        time_before = time_after = coupling_params["time"]
        assert float(sf * time_before).is_integer(), (
            "Invalid time parameter for coupling. Must be a whole number of samples."
        )
        bef = int(sf * time_before)
        aft = int(sf * time_after)

    # Collect per-channel DataFrames, then concat once at the end
    dfs = []

    for i in range(n_chan):
        # ####################################################################
        # START SINGLE CHANNEL DETECTION
        # ####################################################################
        # First, skip channels with bad data amplitude
        if bad_chan[i]:
            continue

        # Find peaks in data
        # Negative peaks with value comprised between -40 to -300 uV
        idx_neg_peaks, _ = signal.find_peaks(-1 * data_filt[i, :], height=amp_neg)
        # Positive peaks with values comprised between 10 to 200 uV
        idx_pos_peaks, _ = signal.find_peaks(data_filt[i, :], height=amp_pos)
        # Keep only the peaks that are in the sleep stages defined in include
        idx_neg_peaks = idx_neg_peaks[mask[idx_neg_peaks]]
        idx_pos_peaks = idx_pos_peaks[mask[idx_pos_peaks]]

        # If no peaks are detected, return None
        if len(idx_neg_peaks) == 0 or len(idx_pos_peaks) == 0:
            logger.warning("No SW were found in channel %s.", ch_names[i])
            continue

        # Make sure that the last detected peak is a positive one
        if idx_pos_peaks[-1] < idx_neg_peaks[-1]:
            # If not, append a fake positive peak one sample after the last neg
            idx_pos_peaks = np.append(idx_pos_peaks, idx_neg_peaks[-1] + 1)

        # For each negative peak, we find the closest following positive peak
        pk_sorted = np.searchsorted(idx_pos_peaks, idx_neg_peaks)
        closest_pos_peaks = idx_pos_peaks[pk_sorted] - idx_neg_peaks
        closest_pos_peaks = closest_pos_peaks[np.nonzero(closest_pos_peaks)]
        idx_pos_peaks = idx_neg_peaks + closest_pos_peaks

        # Now we compute the PTP amplitude and keep only the good peaks
        sw_pt = np.abs(data_filt[i, idx_neg_peaks])
        sw_ptp = sw_pt + data_filt[i, idx_pos_peaks]
        good_ptp = np.logical_and(sw_ptp > amp_ptp[0], sw_ptp < amp_ptp[1])

        # If good_ptp is all False
        if not good_ptp.any():
            logger.warning("No SW were found in channel %s.", ch_names[i])
            continue

        sw_ptp = sw_ptp[good_ptp]
        sw_pt = sw_pt[good_ptp]
        idx_neg_peaks = idx_neg_peaks[good_ptp]
        idx_pos_peaks = idx_pos_peaks[good_ptp]

        # Now we need to check the negative and positive phase duration
        # For that we need to compute the zero crossings of the filtered signal
        zero_crossings = _zerocrossings(data_filt[i, :])
        # Make sure that there is a zero-crossing after the last detected peak
        if zero_crossings[-1] < max(idx_pos_peaks[-1], idx_neg_peaks[-1]):
            # If not, append the index of the last peak
            zero_crossings = np.append(zero_crossings, max(idx_pos_peaks[-1], idx_neg_peaks[-1]))

        # Find distance to previous and following zc
        neg_sorted = np.searchsorted(zero_crossings, idx_neg_peaks)
        previous_neg_zc = zero_crossings[neg_sorted - 1] - idx_neg_peaks
        following_neg_zc = zero_crossings[neg_sorted] - idx_neg_peaks

        # Distance between the positive peaks and the previous and
        # following zero-crossings
        pos_sorted = np.searchsorted(zero_crossings, idx_pos_peaks)
        previous_pos_zc = zero_crossings[pos_sorted - 1] - idx_pos_peaks
        following_pos_zc = zero_crossings[pos_sorted] - idx_pos_peaks

        # Duration of the negative and positive phases, in seconds
        neg_phase_dur = (np.abs(previous_neg_zc) + following_neg_zc) / sf
        pos_phase_dur = (np.abs(previous_pos_zc) + following_pos_zc) / sf

        # We now compute a set of metrics
        sw_start = (idx_neg_peaks + previous_neg_zc) / sf
        sw_end = (idx_pos_peaks + following_pos_zc) / sf
        # This should be the same as `sw_dur = pos_phase_dur + neg_phase_dur`
        # We round to avoid floating point errr (e.g. 1.9000000002)
        sw_dur = (sw_end - sw_start).round(4)
        sw_dur_both_phase = (pos_phase_dur + neg_phase_dur).round(4)
        sw_midcrossing = (idx_neg_peaks + following_neg_zc) / sf
        sw_idx_neg = idx_neg_peaks / sf  # Location of negative peak
        # Slope between peak trough and midcrossing.
        sw_slope = sw_pt / (sw_midcrossing - sw_idx_neg)

        # And we apply a set of thresholds to remove bad slow waves
        good_sw = np.logical_and.reduce(
            (
                # Data edges
                previous_neg_zc != 0,
                following_neg_zc != 0,
                previous_pos_zc != 0,
                following_pos_zc != 0,
                # Duration criteria
                sw_dur == sw_dur_both_phase,  # dur = negative + positive
                sw_dur <= dur_neg[1] + dur_pos[1],  # dur < max(neg) + max(pos)
                sw_dur >= dur_neg[0] + dur_pos[0],  # dur > min(neg) + min(pos)
                neg_phase_dur > dur_neg[0],
                neg_phase_dur < dur_neg[1],
                pos_phase_dur > dur_pos[0],
                pos_phase_dur < dur_pos[1],
                # Sanity checks
                sw_midcrossing > sw_start,
                sw_midcrossing < sw_end,
                np.isfinite(sw_slope),
                sw_slope > 0,
            )
        )

        if not good_sw.any():
            logger.warning("No SW were found in channel %s.", ch_names[i])
            continue

        # Create a dataframe, keeping only good events
        sw_params = {
            "Start": sw_start,
            "NegPeak": sw_idx_neg,
            "MidCrossing": sw_midcrossing,
            "PosPeak": idx_pos_peaks / sf,
            "End": sw_end,
            "Duration": sw_dur,
            "ValNegPeak": data_filt[i, idx_neg_peaks],
            "ValPosPeak": data_filt[i, idx_pos_peaks],
            "PTP": sw_ptp,
            "Slope": sw_slope,
            "Frequency": 1 / sw_dur,
        }
        df_chan = pd.DataFrame(sw_params)[good_sw].reset_index(drop=True)
        idx_neg_peaks = idx_neg_peaks[good_sw]

        # Add phase (in radians) of slow-oscillation signal at maximum
        # spindles-related sigma amplitude within a XX-seconds centered epochs.
        if coupling:
            # Instantaneous phase/amplitude using Hilbert transform. This is done one channel at
            # a time to limit memory usage.
            sw_pha = np.angle(signal.hilbert(data_filt[i, :], N=nfast)[:n_samples])
            sp_amp = np.abs(signal.hilbert(data_sp[i, :], N=nfast)[:n_samples])
            # Center of each epoch is defined as the negative peak of the SW
            n_peaks = idx_neg_peaks.shape[0]
            # idx.shape = (len(idx_valid), bef + aft + 1)
            idx, idx_valid = get_centered_indices(data[i, :], idx_neg_peaks, bef, aft)
            sw_pha_ev = sw_pha[idx]
            sp_amp_ev = sp_amp[idx]
            # Values are set back into the original shape, since some epochs may be out of bounds
            sigma_peak = np.full(n_peaks, np.nan)
            phase_at_sigma_peak = np.full(n_peaks, np.nan)
            ndpac = np.full(n_peaks, np.nan)
            # 1) Find location of max sigma amplitude in epoch
            idx_max_amp = sp_amp_ev.argmax(axis=1)
            # Timestamp at sigma peak, expressed in seconds from negative peak
            # e.g. -0.39, 0.5, 1, 2 -- limits are [time_before, time_after]
            # and converted to absolute time from beginning of the recording
            sigma_peak[idx_valid] = (
                df_chan["NegPeak"].to_numpy()[idx_valid] + (idx_max_amp - bef) / sf
            )
            # 2) PhaseAtSigmaPeak: SW phase at max sigma amplitude in epoch
            phase_at_sigma_peak[idx_valid] = np.take_along_axis(
                sw_pha_ev, idx_max_amp[..., None], axis=1
            )[:, 0]
            # 3) Normalized Direct PAC, with thresholding
            # Unreliable values are set to 0
            ndpac[idx_valid] = np.squeeze(
                _norm_direct_pac(sw_pha_ev[None, ...], sp_amp_ev[None, ...], p=coupling_params["p"])
            )
            df_chan["SigmaPeak"] = sigma_peak
            df_chan["PhaseAtSigmaPeak"] = phase_at_sigma_peak
            df_chan["ndPAC"] = ndpac

        if hypno is not None:
            df_chan["Stage"] = hypno[idx_neg_peaks]

        # Remove all duplicates
        df_chan = df_chan.drop_duplicates(subset=["Start"], keep=False)
        df_chan = df_chan.drop_duplicates(subset=["End"], keep=False)

        if remove_outliers:
            df_chan = _remove_outliers(df_chan, SWResults._features, ch_names[i])

        # ####################################################################
        # END SINGLE CHANNEL DETECTION
        # ####################################################################

        df_chan["Channel"] = ch_names[i]
        df_chan["IdxChannel"] = i
        dfs.append(df_chan)

    # If no SW were detected, return None
    if not dfs:
        logger.warning("No SW were found in data. Returning None.")
        return None
    df = pd.concat(dfs, axis=0, ignore_index=True)

    return SWResults(
        events=df, data=data, sf=sf, ch_names=ch_names, hypno=hypno, data_filt=data_filt
    )


class SWResults(_DetectionResults):
    """Output class for slow-waves detection.

    Attributes
    ----------
    _events : :py:class:`pandas.DataFrame`
        Output detection dataframe
    _data : array_like
        EEG data of shape *(n_chan, n_samples)*.
    _data_filt : array_like
        Slow-wave filtered EEG data of shape *(n_chan, n_samples)*.
    _sf : float
        Sampling frequency of data.
    _ch_names : list
        Channel names.
    _hypno : array_like or None
        Sleep staging vector.
    """

    _features = ("Duration", "ValNegPeak", "ValPosPeak", "PTP", "Slope", "Frequency")
    _title = "Average SW"

    def _get_aggdict(self, aggfunc):
        aggdict = super()._get_aggdict(aggfunc)
        if "PhaseAtSigmaPeak" in self._events:
            aggdict["PhaseAtSigmaPeak"] = lambda x: circmean(x, low=-np.pi, high=np.pi)
            aggdict["ndPAC"] = aggfunc
        if "CooccurringSpindle" in self._events:
            # We do not average "CooccurringSpindlePeak"
            aggdict["CooccurringSpindle"] = aggfunc
            aggdict["DistanceSpindleToSW"] = aggfunc
        return aggdict

    def summary(self, grp_chan=False, grp_stage=False, mask=None, aggfunc="mean", sort=True):
        """Return a summary of the SW detection, optionally grouped across
        channels and/or stage.

        Parameters
        ----------
        grp_chan : bool
            If True, group by channel (for multi-channels detection only).
        grp_stage : bool
            If True, group by sleep stage (provided that an hypnogram was used).
        mask : array_like or None
            Custom boolean mask. Only the detected events for which mask is True will be
            included in the summary. Default is None, i.e. no masking (all events are included).
        aggfunc : str or function
            Averaging function (e.g. ``'mean'`` or ``'median'``).
        sort : bool
            If True, sort group keys when grouping.
        """
        return super().summary(
            grp_chan=grp_chan,
            grp_stage=grp_stage,
            aggfunc=aggfunc,
            sort=sort,
            mask=mask,
        )

    def find_cooccurring_spindles(self, spindles, lookaround=1.2):
        """Given a spindles detection summary dataframe, find slow-waves that co-occur with
        sleep spindles.

        .. versionadded:: 0.6.0

        Parameters
        ----------
        spindles : :py:class:`pandas.DataFrame`
            Output dataframe of :py:meth:`yasa.SpindlesResults.summary`.
        lookaround : float
            Lookaround window, in seconds. The default is +/- 1.2 seconds around the
            negative peak of the slow-wave, as in [1]_. This means that YASA will look for a
            spindle in a 2.4 seconds window centered around the downstate of the slow-wave.

        Returns
        -------
        _events : :py:class:`pandas.DataFrame`
            The slow-wave detection is modified IN-PLACE (see Notes). To see the updated dataframe,
            call the :py:meth:`yasa.SWResults.summary` method.

        Notes
        -----
        From [1]_:

            "SO–spindle co-occurrence was first determined by the number of spindle centers
            occurring within a ±1.2-sec window around the downstate peak of a SO, expressed as
            the ratio of all detected SO events in an individual channel."

        This function adds three columns to the output detection dataframe:

        * `CooccurringSpindle`: a boolean column (True / False) that indicates whether the given
          slow-wave co-occur with a sleep spindle.

        * `CooccurringSpindlePeak`: the timestamp of the peak of the co-occurring,
          in seconds from beginning of recording. Values are set to np.nan when no co-occurring
          spindles were found.

        * `DistanceSpindleToSW`: The distance in seconds from the center peak of the spindles and
          the negative peak of the slow-waves. Negative values indicate that the spindles occured
          before the negative peak of the slow-waves. Values are set to np.nan when no co-occurring
          spindles were found.

        References
        ----------
        .. [1] Kurz, E. M., Conzelmann, A., Barth, G. M., Renner, T. J., Zinke, K., & Born, J.
               (2021). How do children with autism spectrum disorder form gist memory during sleep?
               A study of slow oscillation–spindle coupling. Sleep, 44(6), zsaa290.
        """
        assert isinstance(spindles, pd.DataFrame), "spindles must be a detection dataframe."
        # Intersect the unique channel names (np.isin is very slow on long arrays of strings)
        common_ch = np.intersect1d(self._events["Channel"].unique(), spindles["Channel"].unique())
        assert len(common_ch), "No common channel(s) were found."
        sw_channels = self._events["Channel"].to_numpy()
        sw_peaks = self._events["NegPeak"].to_numpy()
        cooccurring_spindle_peaks = np.full(sw_peaks.size, np.nan)

        for chan in np.unique(sw_channels):
            is_chan = sw_channels == chan
            sw_chan_peaks = sw_peaks[is_chan]
            sp_chan_peaks = np.sort(spindles.loc[spindles["Channel"] == chan, "Peak"].to_numpy())
            # Last spindle peak strictly before the end of the lookaround window
            idx_last = np.searchsorted(sp_chan_peaks, sw_chan_peaks + lookaround, side="left") - 1
            sp_peak = sp_chan_peaks[idx_last.clip(min=0)] if sp_chan_peaks.size else sw_chan_peaks
            # ... which must also be strictly after the start of the window
            is_cooccurring = (idx_last >= 0) & (sp_peak > sw_chan_peaks - lookaround)
            cooccurring_spindle_peaks[is_chan] = np.where(is_cooccurring, sp_peak, np.nan)

        # Add columns to self._events: IN-PLACE MODIFICATION!
        self._events["CooccurringSpindle"] = ~np.isnan(cooccurring_spindle_peaks)
        self._events["CooccurringSpindlePeak"] = cooccurring_spindle_peaks
        self._events["DistanceSpindleToSW"] = cooccurring_spindle_peaks - sw_peaks

    def compare_channels(self, score="f1", max_distance_sec=0):
        """
        Compare detected slow-waves across channels.

        This is a wrapper around the :py:func:`yasa.compare_detection` function. Please
        refer to the documentation of this function for more details.

        Parameters
        ----------
        score : str
            The performance metric to compute. Accepted values are "precision", "recall"
            (aka sensitivity) and "f1" (default). The F1-score is the harmonic mean of precision
            and recall, and is usually the preferred metric to evaluate the agreement between
            two channels. All three metrics are bounded by 0 and 1, where 1 indicates perfect
            agreement.
        max_distance_sec : float
            The maximum distance between slow-waves, in seconds, to consider as the same event.

            .. warning:: To reduce computation cost, YASA rounds the start time of each spindle to
                the nearest decisecond (= 100 ms). This means that the lowest possible resolution
                is 100 ms, regardless of the sampling frequency of the data. Two slow-waves
                starting at 500 ms and 540 ms on their respective channels will therefore always be
                considered the same event, even when max_distance_sec=0.

        Returns
        -------
        scores : :py:class:`pandas.DataFrame`
            A Pandas DataFrame with the output scores, of shape (n_chan, n_chan).

        Notes
        -----
        Some use cases of this function:

        1. What proportion of slow-waves detected in one channel are also detected on
           another channel (if using ``score="recall"``).
        2. What is the overall agreement in the detected events between channels?
        3. Is the agreement better in channels that are close to one another?
        """
        return super().compare_channels(score, max_distance_sec)

    def compare_detection(self, other, max_distance_sec=0, other_is_groundtruth=True):
        """
        Compare the detected slow-waves against either another YASA detection or against custom
        annotations (e.g. ground-truth human scoring).

        This function is a wrapper around the :py:func:`yasa.compare_detection` function. Please
        refer to the documentation of this function for more details.

        Parameters
        ----------
        other : dataframe or detection results
            This can be either a) the output of another YASA detection, for example if you want to
            test the impact of tweaking some parameters on the detected events or b) a pandas
            DataFrame with custom annotations, obtained by another detection method outside
            of YASA, or with manual labelling. If b), the dataframe must contain the "Start" and
            "Channel" columns, with the start of each event in seconds from the beginning
            of the recording and the channel name, respectively. The channel names should match
            the output of the summary() method.
        max_distance_sec : float
            The maximum distance between slow-waves, in seconds, to consider as the same event.

            .. warning:: To reduce computation cost, YASA rounds the start time of each slow-wave
                to the nearest decisecond (= 100 ms). This means that the lowest possible
                resolution is 100 ms, regardless of the sampling frequency of the data.
        other_is_groundtruth : bool
            If True (default), ``other`` will be considered as the ground-truth scoring. If False,
            the current detection will be considered as the ground-truth, and the precision and
            recall scores will be inverted. This parameter has no effect on the F1-score.

            .. note:: when ``other`` is the ground-truth (default), the recall score is the
                fraction of events in other that were succesfully detected by the current
                detection, and the precision score is the proportion of detected events by the
                current detection that are also present in other.

        Returns
        -------
        scores : :py:class:`pandas.DataFrame`
            A Pandas DataFrame with the channel names as index, and the following columns

            * ``precision``: Precision score, aka positive predictive value
            * ``recall``: Recall score, aka sensitivity
            * ``f1``: F1-score
            * ``n_self``: Number of detected events in ``self`` (current method).
            * ``n_other``: Number of detected events in ``other``.

        Notes
        -----
        Some use cases of this function:

        1. How well does YASA events detection perform against ground-truth human annotations?
        2. If I change the threshold(s) of the events detection, do the detected events match
           those obtained with the default parameters?
        3. Which detection thresholds give the highest agreement with the ground-truth scoring?
        """
        return super().compare_detection(other, max_distance_sec, other_is_groundtruth)

    def get_coincidence_matrix(self, scaled=True):
        """Return the (scaled) coincidence matrix.

        Parameters
        ----------
        scaled : bool
            If True (default), the coincidence matrix is scaled (see Notes).

        Returns
        -------
        coincidence : pd.DataFrame
            A symmetric matrix with the (scaled) coincidence values.

        Notes
        -----
        Do slow-waves occur at the same time? One way to measure this is to
        calculate the coincidence matrix, which gives, for each pair of
        channel, the number of samples that were marked as a slow-waves in both
        channels. The output is a symmetric matrix, in which the diagonal is
        simply the number of data points that were marked as a slow-waves in
        the channel.

        The coincidence matrix can be scaled (default) by dividing the output
        by the product of the sum of each individual binary mask, as shown in
        the example below. It can then be used to define functional
        networks or quickly find outlier channels.

        Examples
        --------
        Calculate the coincidence of two binary mask:

        >>> import numpy as np
        >>> x = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1])
        >>> y = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1])
        >>> x * y
        array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1])

        >>> int((x * y).sum())  # Coincidence
        3

        >>> float((x * y).sum() / (x.sum() * y.sum()))  # Scaled coincidence
        0.12

        References
        ----------
        - https://github.com/Mark-Kramer/Sleep-Networks-2021
        """
        return super().get_coincidence_matrix(scaled=scaled)

    def get_mask(self):
        """Return a boolean array indicating for each sample in data if this
        sample is part of a detected event (True) or not (False).
        """
        return super().get_mask()

    def get_sync_events(
        self,
        center="NegPeak",
        time_before=0.4,
        time_after=0.8,
        filt=(None, None),
        mask=None,
        as_dataframe=True,
    ):
        """
        Return the raw data of each detected event after centering to a specific timepoint.

        Parameters
        ----------
        center : str
            Landmark of the event to synchronize the timing on.
            Default is to use the negative peak of the slow-wave.
        time_before : float
            Time (in seconds) before ``center``.
        time_after : float
            Time (in seconds) after ``center``.
        filt : tuple
            Optional filtering to apply to data. For instance, ``filt=(1, 30)``
            will apply a 1 to 30 Hz bandpass filter, and ``filt=(None, 40)``
            will apply a 40 Hz lowpass filter. Filtering is done using default
            parameters in the :py:func:`mne.filter.filter_data` function.
        mask : array_like or None
            Custom boolean mask. Only the detected events for which mask is True will be
            included. Default is None, i.e. no masking (all events are included).
        as_dataframe : boolean
            If True (default), returns a long-format pandas dataframe. If False, returns a list of
            numpy arrays. Each element of the list a unique channel, and the shape of the numpy
            arrays within the list is (n_events, n_times).

        Returns
        -------
        df_sync : :py:class:`pandas.DataFrame` or list
            Ouput long-format dataframe (if ``as_dataframe=True``)::

            'Event' : Event number
            'Time' : Timing of the events (in seconds)
            'Amplitude' : Raw or filtered data for event
            'Channel' : Channel
            'IdxChannel' : Index of channel in data
            'Stage': Sleep stage in which the events occured (if available)
        """
        return super().get_sync_events(
            center=center,
            time_before=time_before,
            time_after=time_after,
            filt=filt,
            mask=mask,
            as_dataframe=as_dataframe,
        )

    def plot_average(
        self,
        center="NegPeak",
        hue="Channel",
        time_before=0.4,
        time_after=0.8,
        filt=(None, None),
        mask=None,
        figsize=(6, 4.5),
        **kwargs,
    ):
        """
        Plot the average slow-wave.

        Parameters
        ----------
        center : str
            Landmark of the event to synchronize the timing on. The default is to use the negative
            peak of the slow-wave.
        hue : str
            Grouping variable that will produce lines with different colors.
            Can be either 'Channel' or 'Stage'.
        time_before : float
            Time (in seconds) before ``center``.
        time_after : float
            Time (in seconds) after ``center``.
        filt : tuple
            Optional filtering to apply to data. For instance, ``filt=(1, 30)``
            will apply a 1 to 30 Hz bandpass filter, and ``filt=(None, 40)``
            will apply a 40 Hz lowpass filter. Filtering is done using default
            parameters in the :py:func:`mne.filter.filter_data` function.
        mask : array_like or None
            Custom boolean mask. Only the detected events for which mask is True will be
            plotted. Default is None, i.e. no masking (all events are included).
        figsize : tuple
            Figure size in inches.
        **kwargs : dict
            Optional argument that are passed to :py:func:`seaborn.lineplot`.
        """
        return super().plot_average(
            center=center,
            hue=hue,
            time_before=time_before,
            time_after=time_after,
            filt=filt,
            mask=mask,
            figsize=figsize,
            **kwargs,
        )

    def plot_detection(self):
        """Plot an overlay of the detected slow-waves on the EEG signal.

        This only works in Jupyter and it requires the ipywidgets
        (https://ipywidgets.readthedocs.io/en/latest/) package.

        To activate the interactive mode, make sure to run:

        >>> %matplotlib widget  # doctest: +SKIP

        .. versionadded:: 0.4.0
        """
        return super().plot_detection()


#############################################################################
# REMs DETECTION
#############################################################################


@_restore_log_level
def rem_detect(
    loc,
    roc,
    sf,
    hypno=None,
    include=4,
    amplitude=(50, 325),
    duration=(0.3, 1.2),
    relative_prominence=0.8,
    freq_rem=(0.5, 5),
    remove_outliers=False,
    verbose=False,
):
    """Rapid eye movements (REMs) detection.

    This detection requires both the left EOG (LOC) and right EOG (LOC).
    The units of the data must be uV. The algorithm is based on an amplitude
    thresholding of the negative product of the LOC and ROC
    filtered signal.

    .. versionadded:: 0.1.5

    Parameters
    ----------
    loc, roc : array_like
        Continuous EOG data (Left and Right Ocular Canthi, LOC / ROC) channels.
        Unit must be uV.

        .. warning::
            The default unit of :py:class:`mne.io.BaseRaw` is Volts.
            Therefore, if passing data from a :py:class:`mne.io.BaseRaw`,
            make sure to use units="uV" to get the data in micro-Volts, e.g.:

            >>> data = raw.get_data(units="uV")  # doctest: +SKIP

    sf : float
        Sampling frequency of the data, in Hz.
    hypno : array_like or :py:class:`yasa.Hypnogram`
        Sleep stage (hypnogram). If the hypnogram is loaded, the
        detection will only be applied to the value defined in
        ``include`` (default = REM sleep).

        Can be an upsampled integer array (same number of samples as ``data``)
        or a :py:class:`yasa.Hypnogram` instance (automatically upsampled).
        To manually upsample an integer array, use
        :py:meth:`yasa.Hypnogram.upsample_to_data`.

        .. note::
            When passing an integer array, hypnogram values follow this mapping:

            - -2 = Unscored
            - -1 = Artefact / Movement
            - 0 = Wake
            - 1 = N1 sleep
            - 2 = N2 sleep
            - 3 = N3 sleep
            - 4 = REM sleep
    include : tuple, list or int or str
        Values in ``hypno`` that will be included in the mask. The default is
        (4), meaning that the detection is applied on REM sleep.
        This has no effect when ``hypno`` is None.

        When ``hypno`` is a :py:class:`yasa.Hypnogram`, string labels can be
        used instead of integers (e.g. ``"REM"``).
    amplitude : tuple or list
        Minimum and maximum amplitude of the peak of the REM.
        Default is 50 uV to 325 uV.
    duration : tuple or list
        The minimum and maximum duration of the REMs.
        Default is 0.3 to 1.2 seconds.
    relative_prominence : float
        Relative prominence used to detect the peaks. The actual prominence is computed
        by multiplying relative prominence by the minimal amplitude. Default is 0.8.
    freq_rem : tuple or list
        Frequency range of REMs. Default is 0.5 to 5 Hz.
    remove_outliers : boolean
        If True, YASA will automatically detect and remove outliers REMs
        using :py:class:`sklearn.ensemble.IsolationForest`.
        YASA uses a random seed (42) to ensure reproducible results.
        Note that this step will only be applied if there are more than
        50 detected REMs in the first place. Default to False.
    verbose : bool or str
        Verbose level. Default (False) will only print warning and error
        messages. The logging levels are 'debug', 'info', 'warning', 'error',
        and 'critical'. For most users the choice is between 'info'
        (or ``verbose=True``) and warning (``verbose=False``).

        .. versionadded:: 0.2.0

    Returns
    -------
    rem : :py:class:`yasa.REMResults`
        To get the full detection dataframe, use:

        >>> rem = rem_detect(...)  # doctest: +SKIP
        >>> rem.summary()  # doctest: +SKIP

        This will give a :py:class:`~pandas.DataFrame` where each row is a
        detected REM and each column is a parameter (= property).
        To get the average parameters for each sleep stage:

        >>> rem.summary(grp_stage=True)  # doctest: +SKIP

        This will give a :py:class:`~pandas.DataFrame` where each row corresponds with a
        sleep stage where >0 REMs were detected. Additional columns ``Count`` (number
        of REMs detected) and ``Density`` (number of REMs detected per minute) will be included.

    Notes
    -----
    The parameters that are calculated for each REM are:

    * ``'Start'``: Start of each detected REM, in seconds from the
      beginning of data.
    * ``'Peak'``: Location of the peak (in seconds of data)
    * ``'End'``: End time (in seconds)
    * ``'Duration'``: Duration (in seconds)
    * ``'LOCAbsValPeak'``: LOC absolute amplitude at REM peak (in uV)
    * ``'ROCAbsValPeak'``: ROC absolute amplitude at REM peak (in uV)
    * ``'LOCAbsRiseSlope'``: LOC absolute rise slope (in uV/s)
    * ``'ROCAbsRiseSlope'``: ROC absolute rise slope (in uV/s)
    * ``'LOCAbsFallSlope'``: LOC absolute fall slope (in uV/s)
    * ``'ROCAbsFallSlope'``: ROC absolute fall slope (in uV/s)
    * ``'Stage'``: Sleep stage (only if hypno was provided)

    Note that all the output parameters are computed on the filtered LOC and
    ROC signals.

    For better results, apply this detection only on artefact-free REM sleep.

    References
    ----------
    The rapid eye movements detection algorithm is based on:

    * Agarwal, R., Takeuchi, T., Laroche, S., & Gotman, J. (2005).
      `Detection of rapid-eye movements in sleep studies.
      <https://doi.org/10.1109/TBME.2005.851512>`_
      IEEE Transactions on Bio-Medical Engineering, 52(8), 1390–1396.

    * Yetton, B. D., Niknazar, M., Duggan, K. A., McDevitt, E. A., Whitehurst,
      L. N., Sattari, N., & Mednick, S. C. (2016). `Automatic detection of
      rapid eye movements (REMs): A machine learning approach.
      <https://doi.org/10.1016/j.jneumeth.2015.11.015>`_
      Journal of Neuroscience Methods, 259, 72–82.

    Examples
    --------
    1. Detect REMs with an upsampled integer hypnogram (legacy):

    .. code-block:: python

        >>> import yasa
        >>> rem = yasa.rem_detect(loc, roc, sf, hypno=hypno_up, include=4)  # doctest: +SKIP

    2. Pass a :py:class:`~yasa.Hypnogram` directly — upsampling and stage filtering are
       handled automatically. String stage labels can be used for ``include``:

    .. code-block:: python

        >>> hyp = yasa.Hypnogram(["W", "N1", "N2", "N3", "REM", "REM"], freq="30s")
        >>> rem = yasa.rem_detect(loc, roc, sf, hypno=hyp, include="REM")  # doctest: +SKIP

    For a full walkthrough, please refer to:
    https://github.com/raphaelvallat/yasa/blob/master/notebooks/07_REMs_detection.ipynb
    """
    # Safety checks
    loc = np.squeeze(np.asarray(loc))
    roc = np.squeeze(np.asarray(roc))
    assert loc.ndim == 1, "LOC must be 1D."
    assert roc.ndim == 1, "ROC must be 1D."
    assert loc.size == roc.size, "LOC and ROC must have the same size."
    data = np.vstack((loc, roc))  # Converted to float64 in _check_data_hypno

    (data, sf, ch_names, hypno, include, mask, n_chan, n_samples, bad_chan) = _check_data_hypno(
        data, sf, ["LOC", "ROC"], hypno, include, verbose=verbose
    )

    # If all channels are bad
    if any(bad_chan):
        logger.warning("At least one channel has bad amplitude. Returning None.")
        return None

    # Bandpass filter
    data_filt = filter_data(data, sf, freq_rem[0], freq_rem[1], verbose=False)

    # Calculate the negative product of LOC and ROC, maximal during REM.
    negp = -data_filt[0, :] * data_filt[1, :]

    # Find peaks in data
    # - height: required height of peaks (min and max.)
    # - distance: required distance in samples between neighboring peaks.
    # - prominence: required prominence of peaks.
    # - wlen: limit search for bases to a specific window.
    hmin, hmax = amplitude[0] ** 2, amplitude[1] ** 2
    pks, pks_params = signal.find_peaks(
        negp,
        height=(hmin, hmax),
        distance=(duration[0] * sf),
        prominence=(relative_prominence * hmin),
        wlen=(duration[1] * sf),
    )

    # Keep only the peaks that are in the sleep stages defined in include
    # We do that before calculating the features in order to gain some time
    is_in_mask = mask[pks]
    pks = pks[is_in_mask]
    pks_params = {k: v[is_in_mask] for k, v in pks_params.items()}

    # If no peaks are detected, return None
    if len(pks) == 0:
        logger.warning("No REMs were found in data. Returning None.")
        return None

    left, right = pks_params["left_bases"], pks_params["right_bases"]
    loc_filt, roc_filt = data_filt

    # Calculate time features
    rem_params = {
        "Start": left / sf,
        "Peak": pks / sf,
        "End": right / sf,
    }
    rem_params["Duration"] = rem_params["End"] - rem_params["Start"]
    # Absolute LOC / ROC value at peak (filtered)
    rem_params["LOCAbsValPeak"] = np.abs(loc_filt[pks])
    rem_params["ROCAbsValPeak"] = np.abs(roc_filt[pks])
    # Absolute rising and falling slope
    dist_pk_left = (pks - left) / sf
    dist_pk_right = (right - pks) / sf
    rem_params["LOCAbsRiseSlope"] = np.abs((loc_filt[pks] - loc_filt[left]) / dist_pk_left)
    rem_params["ROCAbsRiseSlope"] = np.abs((roc_filt[pks] - roc_filt[left]) / dist_pk_left)
    rem_params["LOCAbsFallSlope"] = np.abs((loc_filt[right] - loc_filt[pks]) / dist_pk_right)
    rem_params["ROCAbsFallSlope"] = np.abs((roc_filt[right] - roc_filt[pks]) / dist_pk_right)
    if hypno is not None:
        # The sleep stage at the beginning of the REM is considered.
        rem_params["Stage"] = hypno[left]

    # Keep only the REMs with opposite sign of LOC and ROC, and good duration
    tmin, tmax = duration
    is_good = np.logical_and.reduce(
        (
            np.sign(roc_filt[pks]) != np.sign(loc_filt[pks]),
            rem_params["Duration"] >= tmin,
            rem_params["Duration"] < tmax,
        )
    )
    df = pd.DataFrame(rem_params)[is_good]

    if remove_outliers:
        df = _remove_outliers(df, REMResults._features)

    logger.info("%i REMs were found in data.", df.shape[0])
    df = df.reset_index(drop=True)
    return REMResults(
        events=df, data=data, sf=sf, ch_names=ch_names, hypno=hypno, data_filt=data_filt
    )


class REMResults(_DetectionResults):
    """Output class for REMs detection.

    Attributes
    ----------
    _events : :py:class:`pandas.DataFrame`
        Output detection dataframe
    _data : array_like
        EOG data of shape *(n_chan, n_samples)*, where the two channels are
        LOC and ROC.
    _data_filt : array_like
        Filtered EOG data of shape *(n_chan, n_samples)*, where the two
        channels are LOC and ROC.
    _sf : float
        Sampling frequency of data.
    _ch_names : list
        Channel names (= ``['LOC', 'ROC']``)
    _hypno : array_like or None
        Sleep staging vector.
    """

    _features = (
        "Duration",
        "LOCAbsValPeak",
        "ROCAbsValPeak",
        "LOCAbsRiseSlope",
        "ROCAbsRiseSlope",
        "LOCAbsFallSlope",
        "ROCAbsFallSlope",
    )
    _title = "Average REM"
    # REMs are detected on the product of LOC and ROC, so there is a single "channel"
    _channel_label = "LOC-ROC"

    def _iter_channels(self, events):
        """Each REM is present on both the LOC and ROC channels."""
        for i in range(len(self._ch_names)):
            yield i, events

    def summary(self, grp_stage=False, mask=None, aggfunc="mean", sort=True):
        """Return a summary of the REM detection, optionally grouped across stage.

        Parameters
        ----------
        grp_stage : bool
            If True, group by sleep stage (provided that an hypnogram was
            used).
        mask : array_like or None
            Custom boolean mask. Only the detected events for which mask is True will be
            included in the summary. Default is None, i.e. no masking (all events are included).
        aggfunc : str or function
            Averaging function (e.g. ``'mean'`` or ``'median'``).
        sort : bool
            If True, sort group keys when grouping.
        """
        # ``grp_chan`` is always False for REM detection because the
        # REMs are always detected on a combination of LOC and ROC.
        return super().summary(
            grp_chan=False,
            grp_stage=grp_stage,
            aggfunc=aggfunc,
            sort=sort,
            mask=mask,
        )

    def get_mask(self):
        """Return a boolean array indicating for each sample in data if this
        sample is part of a detected event (True) or not (False).
        """
        return super().get_mask()

    def get_sync_events(
        self,
        center="Peak",
        time_before=0.4,
        time_after=0.4,
        filt=(None, None),
        mask=None,
        as_dataframe=True,
    ):
        """
        Return the raw or filtered data of each detected event after centering to a specific
        timepoint.

        Parameters
        ----------
        center : str
            Landmark of the event to synchronize the timing on.
            Default is to use the peak of the REM.
        time_before : float
            Time (in seconds) before ``center``.
        time_after : float
            Time (in seconds) after ``center``.
        filt : tuple
            Optional filtering to apply to data. For instance, ``filt=(1, 30)``
            will apply a 1 to 30 Hz bandpass filter, and ``filt=(None, 40)``
            will apply a 40 Hz lowpass filter. Filtering is done using default
            parameters in the :py:func:`mne.filter.filter_data` function.
        mask : array_like or None
            Custom boolean mask. Only the detected events for which mask is True will be
            included. Default is None, i.e. no masking (all events are included).
        as_dataframe : boolean
            If True (default), returns a long-format pandas dataframe. If False, returns a list of
            two numpy arrays (LOC and ROC) of shape (n_events, n_times).

        Returns
        -------
        df_sync : :py:class:`pandas.DataFrame` or list
            Ouput long-format dataframe (if ``as_dataframe=True``)::

            'Event' : Event number
            'Time' : Timing of the events (in seconds)
            'Amplitude' : Raw or filtered data for event
            'Channel' : Channel
            'IdxChannel' : Index of channel in data
            'Stage': Sleep stage in which the events occured (if available)
        """
        return super().get_sync_events(
            center=center,
            time_before=time_before,
            time_after=time_after,
            filt=filt,
            mask=mask,
            as_dataframe=as_dataframe,
        )

    def plot_average(
        self,
        center="Peak",
        time_before=0.4,
        time_after=0.4,
        filt=(None, None),
        mask=None,
        figsize=(6, 4.5),
        **kwargs,
    ):
        """
        Plot the average REM.

        Parameters
        ----------
        center : str
            Landmark of the event to synchronize the timing on.
            Default is to use the peak of the REM.
        time_before : float
            Time (in seconds) before ``center``.
        time_after : float
            Time (in seconds) after ``center``.
        filt : tuple
            Optional filtering to apply to data. For instance, ``filt=(1, 30)``
            will apply a 1 to 30 Hz bandpass filter, and ``filt=(None, 40)``
            will apply a 40 Hz lowpass filter. Filtering is done using default
            parameters in the :py:func:`mne.filter.filter_data` function.
        mask : array_like or None
            Custom boolean mask. Only the detected events for which mask is True will be
            included. Default is None, i.e. no masking (all events are included).
        figsize : tuple
            Figure size in inches.
        **kwargs : dict
            Optional argument that are passed to :py:func:`seaborn.lineplot`.
        """
        return super().plot_average(
            center=center,
            hue="Channel",
            time_before=time_before,
            time_after=time_after,
            filt=filt,
            mask=mask,
            figsize=figsize,
            **kwargs,
        )


#############################################################################
# ARTEFACT DETECTION
#############################################################################


@_restore_log_level
def art_detect(
    data,
    sf=None,
    window=5,
    hypno=None,
    include=(1, 2, 3, 4),
    method="covar",
    threshold=3,
    n_chan_reject=1,
    verbose=False,
):
    r"""
    Automatic artifact rejection.

    .. versionadded:: 0.2.0

    Parameters
    ----------
    data : array_like or :py:class:`mne.io.BaseRaw`
        Single or multi-channel EEG data. If ``data`` is array_like, unit must be uV and of
        shape *(n_chan, n_samples)*. If ``data`` is a :py:class:`~mne.io.BaseRaw`
        instance, ``data`` and ``sf`` will be automatically extracted, and
        ``data`` will be automatically converted from Volts (MNE) to micro-Volts (YASA).

        .. warning::
            ``data`` must only contains EEG channels. Please make sure to
            exclude any EOG, EKG or EMG channels.
    sf : float
        Sampling frequency of the data in Hz.
        Can be omitted if ``data`` is a :py:class:`~mne.io.BaseRaw` instance.
    window : float
        The window length (= resolution) for artifact rejection, in seconds.
        Default to 5 seconds. Shorter windows (e.g. 1 or 2-seconds) will
        drastically increase computation time when ``method='covar'``.
    hypno : array_like or :py:class:`yasa.Hypnogram`
        Sleep stage (hypnogram). If the hypnogram is passed, the
        detection will be applied separately for each of the stages defined in
        ``include``.

        Can be an upsampled integer array (same number of samples as ``data``)
        or a :py:class:`yasa.Hypnogram` instance (automatically upsampled).
        To manually upsample an integer array, use
        :py:meth:`yasa.Hypnogram.upsample_to_data`.

        .. note::
            When passing an integer array, hypnogram values follow this mapping:

            - -2 = Unscored
            - -1 = Artefact / Movement
            - 0 = Wake
            - 1 = N1 sleep
            - 2 = N2 sleep
            - 3 = N3 sleep
            - 4 = REM sleep
    include : tuple, list or int or str
        Sleep stages in ``hypno`` on which to perform the artifact rejection.
        The default is ``hypno=(1, 2, 3, 4)``, meaning that the artifact
        rejection is applied separately for all sleep stages, excluding wake.
        This parameter has no effect when ``hypno`` is None.

        When ``hypno`` is a :py:class:`yasa.Hypnogram`, string labels can be
        used instead of integers (e.g. ``["N1", "N2", "N3", "REM"]``).
    method : str
        Artifact detection method (see Notes):

        * ``'covar'`` : Covariance-based, default for 4+ channels data
        * ``'std'`` : Standard-deviation-based, default for single-channel data
    threshold : float
        The number of standard deviations above or below which an
        epoch is considered an artifact. Higher values will result in a more
        conservative detection, i.e. less rejected epochs.
    n_chan_reject : int
        The number of channels that must be below or above ``threshold`` on any
        given epochs to consider this epoch as an artefact when
        ``method='std'``. The default is 1, which means that the epoch will
        be marked as artifact as soon as one channel is above or below the
        threshold. This may be too conservative when working with a large
        number of channels (e.g.hdEEG) in which case users can increase
        ``n_chan_reject``. Note that this parameter only has an effect
        when ``method='std'``.
    verbose : bool or str
        Verbose level. Default (False) will only print warning and error
        messages. The logging levels are 'debug', 'info', 'warning', 'error',
        and 'critical'. For most users the choice is between 'info'
        (or ``verbose=True``) and warning (``verbose=False``).

        .. versionadded:: 0.2.0

    Returns
    -------
    art_epochs : array_like
        1-D array of shape *(n_epochs)* where 1 = Artefact and 0 = Good.
    zscores : array_like
        Array of z-scores, shape is *(n_epochs)* if ``method='covar'`` and
        *(n_epochs, n_chan)* if ``method='std'``.

    Notes
    -----
    .. caution::
        This function will only detect major body artefacts present on the EEG
        channel. It will not detect EKG contamination or eye blinks. For more
        artifact rejection tools, please refer to the `MNE Python package
        <https://mne.tools/stable/auto_tutorials/preprocessing/10_preprocessing_overview.html>`_.

    .. tip::
        For best performance, apply this function on pre-staged data and make
        sure to pass the hypnogram.
        Sleep stages have very different EEG signatures
        and the artifect rejection will be much more accurate when applied
        separately on each sleep stage.

    We provide below a short description of the different methods. For
    multi-channel data, and if computation time is not an issue, we recommend
    using ``method='covar'`` which uses a clustering approach on
    variance-covariance matrices, and therefore takes into account
    not only the variance in each channel and each epoch, but also the
    inter-relationship (covariance) between channel.

    ``method='covar'`` is however not supported for single-channel EEG or when
    less than 4 channels are present in ``data``. In these cases, one can
    use the much faster ``method='std'`` which is simply based on a z-scoring
    of the log-transformed standard deviation of each channel and each epoch.

    **1/ Covariance-based multi-channel artefact rejection**

    ``method='covar'`` is essentially a wrapper around the
    :py:class:`pyriemann.artifact_detection.Potato` class implemented in the
    `pyRiemann package
    <https://pyriemann.readthedocs.io/en/latest/index.html>`_.

    The main idea of this approach is to estimate a reference covariance
    matrix :math:`\bar{C}` (for each sleep stage separately if ``hypno`` is
    present) and reject every epoch which is too far from this reference
    matrix.
    The distance of the covariance matrix of the current epoch :math:`C`
    from the reference matrix is calculated using Riemannian
    geometry, which is more adapted than Euclidean geometry for
    symmetric positive definite covariance matrices:

    .. math::  d = {\left( \sum_i \log(\lambda_i)^2 \right)}^{-1/2}

    where :math:`\lambda_i` are the joint eigenvalues of :math:`C` and
    :math:`\bar{C}`. The epoch with covariance matric :math:`C`
    will be marked as an artifact if the distance :math:`d`
    is greater than a threshold :math:`T`
    (typically 2 or 3 standard deviations).
    :math:`\bar{C}` is iteratively estimated using a clustering approach.

    **2/ Standard-deviation-based single and multi-channel artefact rejection**

    ``method='std'`` is a much faster and straightforward approach which
    is simply based on the distribution of the standard deviations of each
    epoch. Specifically, one first calculate the standard
    deviations of each epoch and each channel. Then, the resulting array of
    standard deviations is log-transformed and z-scored (for each sleep
    stage separately if ``hypno`` is present). Any epoch with one or more
    channel exceeding the threshold will be marked as artifact.

    Note that this approach is more sensitive to noise and/or the influence of
    one bad channel (e.g. electrode fell off at some point during the night).
    We therefore recommend that you visually inspect and remove any bad
    channels prior to using this function.

    References
    ----------
    * Barachant, A., Andreev, A., & Congedo, M. (2013). `The Riemannian
      Potato: an automatic and adaptive artifact detection method for online
      experiments using Riemannian geometry.
      <https://hal.science/hal-00781701/>`_ TOBI
      Workshop lV, 19–20.

    * Barthélemy, Q., Mayaud, L., Ojeda, D., & Congedo, M. (2019).
      `The Riemannian Potato Field: A Tool for Online Signal Quality Index of
      EEG. <https://doi.org/10.1109/TNSRE.2019.2893113>`_
      IEEE Transactions on Neural Systems and Rehabilitation Engineering:
      A Publication of the IEEE Engineering in Medicine and Biology Society,
      27(2), 244–255.

    * https://pyriemann.readthedocs.io/en/latest/index.html

    Examples
    --------
    1. Detect artefacts per sleep stage with an upsampled integer hypnogram (legacy):

    .. code-block:: python

        >>> import yasa
        >>> art = yasa.art_detect(data, sf, hypno=hypno_up, include=(1, 2, 3, 4))  # doctest: +SKIP

    2. Pass a :py:class:`~yasa.Hypnogram` directly — upsampling and stage filtering are
       handled automatically. String stage labels can be used for ``include``:

    .. code-block:: python

        >>> hyp = yasa.Hypnogram(["W", "N1", "N2", "N3", "REM"], freq="30s")
        >>> art = yasa.art_detect(data, sf, hypno=hyp, include=["N1", "N2", "N3", "REM"])  # doctest: +SKIP

    For a full walkthrough, please refer to:
    https://github.com/raphaelvallat/yasa/blob/master/notebooks/13_artifact_rejection.ipynb
    """
    ###########################################################################
    # PREPROCESSING
    ###########################################################################

    (data, sf, _, hypno, include, _, n_chan, n_samples, _) = _check_data_hypno(
        data, sf, ch_names=None, hypno=hypno, include=include, check_amp=False, verbose=verbose
    )

    assert isinstance(n_chan_reject, int), "n_chan_reject must be int."
    assert n_chan_reject >= 1, "n_chan_reject must be >= 1."
    assert n_chan_reject <= n_chan, "n_chan_reject must be <= n_chan."

    # Safety check: sampling frequency and window
    assert isinstance(window, (int, float)), "window must be int or float"
    if isinstance(sf, float):
        assert sf.is_integer(), "sf must be a whole number."
        sf = int(sf)
    win_sec = window
    window = win_sec * sf  # Convert window to samples
    if isinstance(window, float):
        assert window.is_integer(), "window * sf must be a whole number."
        window = int(window)

    # Safety checks: methods
    assert isinstance(method, str), "method must be a string."
    method = method.lower()
    if method in ["cov", "covar", "covariance", "riemann", "potato"]:
        method = "covar"
        is_pyriemann_installed()
        # Potato moved from pyriemann.clustering to pyriemann.artifact_detection in pyriemann
        # 0.12. The back-compat alias kept in pyriemann.clustering is itself broken (it imports
        # from a misspelled `artifactdetection` module), so try the new location first and only
        # fall back to the old one for pyriemann <= 0.11.
        try:
            from pyriemann.artifact_detection import Potato
        except ImportError:
            from pyriemann.clustering import Potato
        from pyriemann.estimation import Covariances, Shrinkage

        # Must have at least 4 channels to use method='covar'
        if n_chan < 4:
            logger.warning(
                "Must have at least 4 channels for method='covar'. "
                "Automatically switching to method='std'."
            )
            method = "std"
    elif method in ["std", "sd"]:
        method = "std"
    else:
        raise ValueError(f"Invalid method '{method}'. Must be 'covar' or 'std'.")

    ###########################################################################
    # START THE REJECTION
    ###########################################################################
    # Remove flat channels
    isflat = np.nanstd(data, axis=-1) == 0
    if isflat.any():
        logger.warning("Flat channel(s) were found and removed in data.")
        data = data[~isflat]
        n_chan = data.shape[0]

    # Epoch the data (n_epochs, n_chan, n_samples)
    _, epochs = sliding_window(data, sf, window=win_sec)
    n_epochs = epochs.shape[0]

    # We first need to identify epochs with flat data (n_epochs, n_chan)
    isflat = (epochs == epochs[:, :, 1][..., None]).all(axis=-1)
    # 1 when all channels are flat, 0 when none ar flat (n_epochs)
    prop_chan_flat = isflat.sum(axis=-1) / n_chan
    # If >= 50% of channels are flat, automatically mark as artefact
    epoch_is_flat = prop_chan_flat >= 0.5
    where_flat_epochs = np.nonzero(epoch_is_flat)[0]
    n_flat_epochs = where_flat_epochs.size

    # Now let's make sure that we have an hypnogram and an include variable
    if hypno is not None:
        # One value per complete epoch. The copy ensures that the user's hypnogram is not
        # modified when flagging the flat epochs below.
        hypno_win = hypno[::window][:n_epochs].copy()
    else:
        # [-2, -2, -2, -2, ...], where -2 stands for unscored
        hypno_win = np.full(n_epochs, -2)
        include = np.array([-2])

    # We want to make sure that hypno-win and n_epochs have EXACTLY same shape
    assert n_epochs == hypno_win.shape[-1], "Hypno and epochs do not match."

    # Finally, we make sure not to include any flat epochs in calculation
    # just using a random number that is unlikely to be picked by users
    if n_flat_epochs > 0:
        hypno_win[where_flat_epochs] = -111991

    # Add logger info (number of samples, sf and duration are logged in _check_data_hypno)
    logger.info("Number of channels in data = %i", n_chan)
    logger.info("Number of epochs = %i" % n_epochs)
    logger.info("Artifact window = %.2f seconds" % win_sec)
    logger.info("Method = %s" % method)
    logger.info("Threshold = %.2f standard deviations" % threshold)

    if method == "covar":
        # Calculate the covariance matrices,
        # shape (n_epochs, n_chan, n_chan)
        covmats = Covariances().fit_transform(epochs)
        # Shrink the covariance matrix (ensure positive semi-definite)
        covmats = Shrinkage().fit_transform(covmats)
        # Define Potato instance: 0 = clean, 1 = art
        # To increase speed we set the max number of iterations from 10 to 100
        potato = Potato(
            metric="riemann", threshold=threshold, pos_label=0, neg_label=1, n_iter_max=10
        )
        # Empty z-scores output (n_epochs)
        zscores = np.full(n_epochs, np.nan)

        def _reject(where_stage):
            """Return the z-scores and artefact labels of the epochs of one stage."""
            zs = potato.fit_transform(covmats[where_stage])
            art = potato.predict(covmats[where_stage]).astype(int)
            return zs, art

    else:
        # Calculate log-transformed standard dev in each epoch
        # We add 1 to avoid log warning id std is zero (e.g. flat line)
        # (n_epochs, n_chan)
        std_epochs = np.log(np.nanstd(epochs, axis=-1) + 1)
        # Empty zscores output (n_epochs, n_chan)
        zscores = np.full((n_epochs, n_chan), np.nan)

        def _reject(where_stage):
            """Return the z-scores and artefact labels of the epochs of one stage."""
            # Calculate z-scores of STD for each channel x stage
            c_mean = np.nanmean(std_epochs[where_stage], axis=0, keepdims=True)
            c_std = np.nanstd(std_epochs[where_stage], axis=0, keepdims=True)
            zs = (std_epochs[where_stage] - c_mean) / c_std
            # Any epoch with at least X channel above or below threshold
            n_chan_supra = (np.abs(zs) > threshold).sum(axis=1)  # >
            art = (n_chan_supra >= n_chan_reject).astype(int)  # >= !
            return zs, art

    # Create empty `hypno_art` vector (1 sample = 1 epoch)
    epoch_is_art = np.zeros(n_epochs, dtype="int")

    for stage in include:
        where_stage = np.where(hypno_win == stage)[0]
        # At least 30 epochs are required to calculate z-scores
        # which amounts to 2.5 minutes when using 5-seconds window
        if where_stage.size < 30:
            if hypno is not None:
                # Only show warnig if user actually pass an hypnogram
                logger.warning(
                    f"At least 30 epochs are required to calculate z-score. Skipping stage {stage}"
                )
            continue
        zs, art = _reject(where_stage)
        if hypno is not None:
            # Only shows if user actually pass an hypnogram
            perc_reject = 100 * (art.sum() / art.size)
            logger.info(
                f"Stage {stage}: {art.sum()} / {art.size} epochs rejected ({perc_reject:.2f}%)"
            )
        # Append to global vector
        epoch_is_art[where_stage] = art
        zscores[where_stage] = zs

    # Mark flat epochs as artefacts
    if n_flat_epochs > 0:
        logger.info(
            f"Rejecting {n_flat_epochs} epochs with >=50% of channels "
            f"that are flat. Z-scores set to np.nan for these epochs."
        )
        epoch_is_art[where_flat_epochs] = 1

    # Log total percentage of epochs rejected
    perc_reject = 100 * (epoch_is_art.sum() / n_epochs)
    text = f"TOTAL: {epoch_is_art.sum()} / {n_epochs} epochs rejected ({perc_reject:.2f}%)"
    logger.info(text)

    # Convert epoch_is_art to boolean [0, 0, 1] -- > [False, False, True]
    epoch_is_art = epoch_is_art.astype(bool)
    return epoch_is_art, zscores


#############################################################################
# COMPARE DETECTION
#############################################################################


def compare_detection(indices_detection, indices_groundtruth, max_distance=0):
    """
    Determine correctness of detected events against ground-truth events.

    Parameters
    ----------
    indices_detection : array_like
        Indices of the detected events. For example, this could be the indices of the
        start of the spindles, or the negative peak of the slow-waves. The indices must be in
        samples, and not in seconds.
    indices_groundtruth : array_like
        Indices of the ground-truth events, in samples.
    max_distance : int, optional
        Maximum distance between indices, in samples, to consider as the same event (default = 0).
        For example, if the sampling frequency of the data is 100 Hz, using `max_distance=100` will
        search for a matching event 1 second before or after the current event.

    Returns
    -------
    results : dict
        A dictionary with the comparison results:

        * ``tp``: True positives, i.e. actual events detected as events.
        * ``fp``: False positives, i.e. non-events detected as events.
        * ``fn``: False negatives, i.e. actual events not detected as events.
        * ``precision``: Precision score, aka positive predictive value (see Notes)
        * ``recall``: Recall score, aka sensitivity (see Notes)
        * ``f1``: F1-score (see Notes)

    Notes
    -----
    * The precision score is calculated as TP / (TP + FP), i.e. the proportion of detected events
      that match a ground-truth event.
    * The recall score is calculated as (N - FN) / N, where N is the number of ground-truth events,
      i.e. the proportion of ground-truth events that match a detected event. This is the same as
      TP / (TP + FN) when ``max_distance=0``.
    * The F1-score is the harmonic mean of precision and recall.

    This function is inspired by the `sleepecg.compare_heartbeats
    <https://sleepecg.readthedocs.io/en/stable/generated/sleepecg.compare_heartbeats.html>`_
    function.

    Examples
    --------
    A simple example. Here, `detected` refers to the indices (in the data) of the detected events.
    These could be for example the index of the onset of each detected spindle. `grndtrth` refers
    to the ground-truth (e.g. human-annotated) events.

    >>> from yasa import compare_detection
    >>> detected = [5, 12, 20, 34, 41, 57, 63]
    >>> grndtrth = [5, 12, 18, 26, 34, 41, 55, 63, 68]
    >>> compare_detection(detected, grndtrth)
    {'tp': array([ 5, 12, 34, 41, 63]),
     'fp': array([20, 57]),
     'fn': array([18, 26, 55, 68]),
     'precision': 0.7142857142857143,
     'recall': 0.5555555555555556,
     'f1': 0.6250000000000001}

    There are 5 true positives, 2 false positives and 4 false negatives. This gives a precision
    score of 0.71 (= 5 / (5 + 2)), a recall score of 0.56 (= 5 / (5 + 4)) and a F1-score of 0.625.
    The F1-score is the harmonic average of precision and recall, and should be the preferred
    metric when comparing the performance of a detection against a ground-truth.

    Order matters! If we set `detected` as the ground-truth, FP and FN are inverted, and same for
    precision and recall. The TP and F1-score remain the same though. Therefore, when comparing two
    detections (and not a detection against a ground-truth), the F1-score is the preferred metric
    because it is independent of the order.

    >>> compare_detection(grndtrth, detected)
    {'tp': array([ 5, 12, 34, 41, 63]),
     'fp': array([18, 26, 55, 68]),
     'fn': array([20, 57]),
     'precision': 0.5555555555555556,
     'recall': 0.7142857142857143,
     'f1': 0.6250000000000001}

    There might be some events that are very close to each other, and we would like to count them
    as true positive even though they do not occur exactly at the same index. This is possible
    with the `max_distance` argument, which defines the lookaround window (in samples) for
    each event.

    >>> compare_detection(detected, grndtrth, max_distance=2)
    {'tp': array([ 5, 12, 20, 34, 41, 57, 63]),
     'fp': array([], dtype=int64),
     'fn': array([26, 68]),
     'precision': 1.0,
     'recall': 0.7777777777777778,
     'f1': 0.8750000000000001}

    Finally, if detected is empty, all performance metrics will be set to zero, and a copy of
    the groundtruth array will be returned as false negatives.

    >>> compare_detection([], grndtrth)
    {'tp': array([], dtype=int64),
     'fp': array([], dtype=int64),
     'fn': array([ 5, 12, 18, 26, 34, 41, 55, 63, 68]),
     'precision': 0,
     'recall': 0,
     'f1': 0}
    """
    # Safety check
    indices_detection = np.asarray(indices_detection, dtype=float)
    indices_groundtruth = np.asarray(indices_groundtruth, dtype=float)
    assert indices_detection.ndim == 1, "detection indices must be a 1D list or array."
    assert indices_groundtruth.ndim == 1, "groundtruth indices must be a 1D list or array."
    assert np.all(np.mod(indices_detection, 1) == 0), "detection indices must be integers."
    assert np.all(np.mod(indices_groundtruth, 1) == 0), "groundtruth indices must be integers."
    assert isinstance(max_distance, int), "max_distance must be 0 or a positive integer."
    assert max_distance >= 0, "max_distance must be 0 or a positive integer."
    # Sorted unique indices
    indices_detection = np.unique(indices_detection.astype(int))
    indices_groundtruth = np.unique(indices_groundtruth.astype(int))

    # Handle cases where indices_detection or indices_groundtruth is empty
    if indices_detection.size == 0 or indices_groundtruth.size == 0:
        return dict(
            tp=np.array([], dtype=int),
            fp=indices_detection,
            fn=indices_groundtruth,
            precision=0,
            recall=0,
            f1=0,
        )

    def _has_match(x, y):
        """For each element of x, whether there is an element of sorted y within max_distance."""
        idx = np.searchsorted(y, x - max_distance, side="left")
        is_valid = idx < y.size
        has_match = np.zeros(x.size, dtype=bool)
        has_match[is_valid] = y[idx[is_valid]] <= x[is_valid] + max_distance
        return has_match

    # Confusion matrix. A detected event is a true positive if there is a ground-truth event
    # within max_distance, and a ground-truth event is a false negative otherwise.
    is_detected_tp = _has_match(indices_detection, indices_groundtruth)
    is_groundtruth_tp = _has_match(indices_groundtruth, indices_detection)
    results = {}
    results["tp"] = indices_detection[is_detected_tp]
    results["fp"] = indices_detection[~is_detected_tp]
    results["fn"] = indices_groundtruth[~is_groundtruth_tp]

    # Performance metrics. With max_distance > 0, one ground-truth event can match several
    # detected events (and vice versa), so precision and recall are each computed on their side.
    precision = float(is_detected_tp.mean())
    recall = float(is_groundtruth_tp.mean())
    results["precision"] = precision
    results["recall"] = recall
    results["f1"] = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return results
