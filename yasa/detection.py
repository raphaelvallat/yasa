"""
YASA (Yet Another Spindle Algorithm): fast and robust detection of spindles,
slow-waves, and rapid eye movements from sleep EEG recordings.

- Author: Raphael Vallat (www.raphaelvallat.com)
- GitHub: https://github.com/raphaelvallat/yasa
- License: BSD 3-Clause License
"""
import mne
import logging
import numpy as np
import pandas as pd
from scipy import signal
from mne.filter import filter_data
from collections import OrderedDict
from scipy.interpolate import interp1d
from scipy.fftpack import next_fast_len
from sklearn.ensemble import IsolationForest

from .spectral import stft_power
from .numba import _detrend, _rms
from .io import set_log_level, is_tensorpac_installed, is_pyriemann_installed
from .others import (moving_transform, trimbothstd, get_centered_indices,
                     sliding_window, _merge_close, _zerocrossings)


logger = logging.getLogger('yasa')

__all__ = ['art_detect', 'spindles_detect', 'SpindlesResults', 'sw_detect', 'SWResults',
           'rem_detect', 'REMResults']


#############################################################################
# DATA PREPROCESSING
#############################################################################

def _check_data_hypno(data, sf=None, ch_names=None, hypno=None, include=None, check_amp=True):
    """Helper functions for preprocessing of data and hypnogram."""
    # 1) Extract data as a 2D NumPy array
    if isinstance(data, mne.io.BaseRaw):
        sf = data.info['sfreq']  # Extract sampling frequency
        ch_names = data.ch_names  # Extract channel names
        data = data.get_data() * 1e6  # Convert from V to uV
    else:
        assert sf is not None, 'sf must be specified if not using MNE Raw.'
    data = np.asarray(data, dtype=np.float64)
    assert data.ndim in [1, 2], 'data must be 1D (times) or 2D (chan, times).'
    if data.ndim == 1:
        # Force to 2D array: (n_chan, n_samples)
        data = data[None, ...]
    n_chan, n_samples = data.shape

    # 2) Check channel names
    if ch_names is None:
        ch_names = ['CHAN' + str(i).zfill(3) for i in range(n_chan)]
    else:
        assert len(ch_names) == n_chan

    # 3) Check hypnogram
    if hypno is not None:
        hypno = np.asarray(hypno, dtype=int)
        assert hypno.ndim == 1, 'Hypno must be one dimensional.'
        assert hypno.size == n_samples, 'Hypno must have same size as data.'
        unique_hypno = np.unique(hypno)
        logger.info('Number of unique values in hypno = %i', unique_hypno.size)
        assert include is not None, 'include cannot be None if hypno is given'
        include = np.atleast_1d(np.asarray(include))
        assert include.size >= 1, '`include` must have at least one element.'
        assert hypno.dtype.kind == include.dtype.kind, ('hypno and include must have same dtype')
        assert np.in1d(hypno, include).any(), ('None of the stages specified '
                                               'in `include` are present in '
                                               'hypno.')

    # 4) Check data amplitude
    logger.info('Number of samples in data = %i', n_samples)
    logger.info('Sampling frequency = %.2f Hz', sf)
    logger.info('Data duration = %.2f seconds', n_samples / sf)
    all_ptp = np.ptp(data, axis=-1)
    all_trimstd = trimbothstd(data, cut=0.05)
    bad_chan = np.zeros(n_chan, dtype=bool)
    for i in range(n_chan):
        logger.info('Trimmed standard deviation of %s = %.4f uV' % (ch_names[i], all_trimstd[i]))
        logger.info('Peak-to-peak amplitude of %s = %.4f uV' % (ch_names[i], all_ptp[i]))
        if check_amp and not(0.1 < all_trimstd[i] < 1e3):
            logger.error('Wrong data amplitude for %s '
                         '(trimmed STD = %.3f). Unit of data MUST be uV! '
                         'Channel will be skipped.'
                         % (ch_names[i], all_trimstd[i]))
            bad_chan[i] = True

    # 5) Create sleep stage vector mask
    if hypno is not None:
        mask = np.in1d(hypno, include)
    else:
        mask = np.ones(n_samples, dtype=bool)

    return (data, sf, ch_names, hypno, include, mask, n_chan, n_samples, bad_chan)


#############################################################################
# BASE DETECTION RESULTS CLASS
#############################################################################


class _DetectionResults(object):
    """Main class for detection results."""

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

    def summary(self, event_type, grp_chan=False, grp_stage=False, aggfunc='mean', sort=True,
                mask=None):
        """Summary"""
        # Check masking
        mask = self._check_mask(mask)

        # Define grouping
        grouper = []
        if grp_stage is True and 'Stage' in self._events:
            grouper.append('Stage')
        if grp_chan is True and 'Channel' in self._events:
            grouper.append('Channel')
        if not len(grouper):
            # Return a copy of self._events after masking, without grouping
            return self._events.loc[mask, :].copy()

        if event_type == 'spindles':
            aggdict = {'Start': 'count',
                       'Duration': aggfunc,
                       'Amplitude': aggfunc,
                       'RMS': aggfunc,
                       'AbsPower': aggfunc,
                       'RelPower': aggfunc,
                       'Frequency': aggfunc,
                       'Oscillations': aggfunc,
                       'Symmetry': aggfunc}

            # if 'SOPhase' in self._events:
            #     from scipy.stats import circmean
            #     aggdict['SOPhase'] = lambda x: circmean(x, low=-np.pi, high=np.pi)

        elif event_type == 'sw':
            aggdict = {'Start': 'count',
                       'Duration': aggfunc,
                       'ValNegPeak': aggfunc,
                       'ValPosPeak': aggfunc,
                       'PTP': aggfunc,
                       'Slope': aggfunc,
                       'Frequency': aggfunc}

            if 'PhaseAtSigmaPeak' in self._events:
                from scipy.stats import circmean
                aggdict['PhaseAtSigmaPeak'] = lambda x: circmean(x, low=-np.pi, high=np.pi)
                aggdict['ndPAC'] = aggfunc

            if "CooccurringSpindle" in self._events:
                # We do not average "CooccurringSpindlePeak"
                aggdict["CooccurringSpindle"] = aggfunc
                aggdict["DistanceSpindleToSW"] = aggfunc

        else:  # REM
            aggdict = {'Start': 'count',
                       'Duration': aggfunc,
                       'LOCAbsValPeak': aggfunc,
                       'ROCAbsValPeak': aggfunc,
                       'LOCAbsRiseSlope': aggfunc,
                       'ROCAbsRiseSlope': aggfunc,
                       'LOCAbsFallSlope': aggfunc,
                       'ROCAbsFallSlope': aggfunc}

        # Apply grouping, after masking
        df_grp = self._events.loc[mask, :].groupby(grouper, sort=sort, as_index=False).agg(aggdict)
        df_grp = df_grp.rename(columns={'Start': 'Count'})

        # Calculate density (= number per min of each stage)
        if self._hypno is not None and grp_stage is True:
            stages = np.unique(self._events['Stage'])
            dur = {}
            for st in stages:
                # Get duration in minutes of each stage present in dataframe
                dur[st] = self._hypno[self._hypno == st].size / (60 * self._sf)

            # Insert new density column in grouped dataframe after count
            df_grp.insert(
                loc=df_grp.columns.get_loc('Count') + 1, column='Density',
                value=df_grp.apply(lambda rw: rw['Count'] / dur[rw['Stage']], axis=1))

        return df_grp.set_index(grouper)

    def get_mask(self):
        """get_mask"""
        from yasa.others import _index_to_events
        mask = np.zeros(self._data.shape, dtype=int)
        for i in self._events['IdxChannel'].unique():
            ev_chan = self._events[self._events['IdxChannel'] == i]
            idx_ev = _index_to_events(
                ev_chan[['Start', 'End']].to_numpy() * self._sf)
            mask[i, idx_ev] = 1
        return np.squeeze(mask)

    def get_sync_events(self, center, time_before, time_after, filt=(None, None), mask=None,
                        as_dataframe=True):
        """Get_sync_events (not for REM, spindles & SW only)"""
        from yasa.others import get_centered_indices
        assert time_before >= 0
        assert time_after >= 0
        bef = int(self._sf * time_before)
        aft = int(self._sf * time_after)
        # TODO: Step size is determined by sf: 0.01 sec at 100 Hz, 0.002 sec at
        # 500 Hz, 0.00390625 sec at 256 Hz. Should we add resample=100 (Hz) or step_size=0.01?
        time = np.arange(-bef, aft + 1, dtype='int') / self._sf

        if any(filt):
            data = mne.filter.filter_data(
                self._data, self._sf, l_freq=filt[0], h_freq=filt[1], method='fir', verbose=False)
        else:
            data = self._data

        # Apply mask
        mask = self._check_mask(mask)
        masked_events = self._events.loc[mask, :]

        output = []

        for i in masked_events['IdxChannel'].unique():
            # Copy is required to merge with the stage later on
            ev_chan = masked_events[masked_events['IdxChannel'] == i].copy()
            ev_chan['Event'] = np.arange(ev_chan.shape[0])
            peaks = (ev_chan[center] * self._sf).astype(int).to_numpy()
            # Get centered indices
            idx, idx_valid = get_centered_indices(data[i, :], peaks, bef, aft)
            # If no good epochs are returned raise a warning
            if len(idx_valid) == 0:
                logger.error(
                    'Time before and/or time after exceed data bounds, please '
                    'lower the temporal window around center. Skipping channel.')
                continue

            # Get data at indices and time vector
            amps = data[i, idx]

            if not as_dataframe:
                # Output is a list (n_channels) of numpy arrays (n_events, n_times)
                output.append(amps)
                continue

            # Convert to long-format dataframe
            df_chan = pd.DataFrame(amps.T)
            df_chan['Time'] = time
            # Convert to long-format
            df_chan = df_chan.melt(id_vars='Time', var_name='Event', value_name='Amplitude')
            # Append stage
            if 'Stage' in masked_events:
                df_chan = df_chan.merge(ev_chan[['Event', 'Stage']].iloc[idx_valid])
            # Append channel name
            df_chan['Channel'] = ev_chan['Channel'].iloc[0]
            df_chan['IdxChannel'] = i
            # Append to master dataframe
            output.append(df_chan)

        if as_dataframe:
            output = pd.concat(output, ignore_index=True)

        return output

    def get_coincidence_matrix(self, scaled=True):
        """get_coincidence_matrix"""
        if len(self._ch_names) < 2:
            raise ValueError("At least 2 channels are required to calculate coincidence.")
        mask = self.get_mask()
        mask = pd.DataFrame(mask.T, columns=self._ch_names)
        mask.columns.name = "Channel"

        def _coincidence(x, y):
            """Calculate the (scaled) coincidence."""
            coincidence = (x * y).sum()
            if scaled:
                # Handle division by zero error
                denom = (x.sum() * y.sum())
                if denom == 0:
                    coincidence = np.nan
                else:
                    coincidence /= denom
            return coincidence

        coinc_mat = mask.corr(method=_coincidence)

        if not scaled:
            # Otherwise diagonal values are set to 1
            np.fill_diagonal(coinc_mat.values, mask.sum())
            coinc_mat = coinc_mat.astype(int)

        return coinc_mat

    def plot_average(self, event_type, center='Peak', hue='Channel', time_before=1,
                     time_after=1, filt=(None, None), mask=None, figsize=(6, 4.5), **kwargs):
        """Plot the average event (not for REM, spindles & SW only)"""
        import seaborn as sns
        import matplotlib.pyplot as plt

        df_sync = self.get_sync_events(center=center, time_before=time_before,
                                       time_after=time_after, filt=filt, mask=mask)
        assert not df_sync.empty, "Could not calculate event-locked data."
        assert hue in ['Stage', 'Channel'], "hue must be 'Channel' or 'Stage'"
        assert hue in df_sync.columns, "%s is not present in data." % hue

        if event_type == 'spindles':
            title = "Average spindle"
        else:  # "sw":
            title = "Average SW"

        # Start figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        sns.lineplot(data=df_sync, x='Time', y='Amplitude', hue=hue, ax=ax, **kwargs)
        # ax.legend(frameon=False, loc='lower right')
        ax.set_xlim(df_sync['Time'].min(), df_sync['Time'].max())
        ax.set_title(title)
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Amplitude (uV)')
        return ax

    def plot_detection(self):
        """Plot an overlay of the detected events on the signal."""
        import matplotlib.pyplot as plt
        import ipywidgets as ipy

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
        plt.plot(times[xrng], self._data[0, xrng], 'k', lw=1)
        plt.plot(times[xrng], highlight[0, xrng], 'indianred')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude (uV)')
        fig.canvas.header_visible = False
        fig.tight_layout()

        # WIDGETS
        layout = ipy.Layout(
            width="50%",
            justify_content='center',
            align_items='center'
        )

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
            orientation='horizontal',
            description="Amplitude:"
        )

        dd_ch = ipy.Dropdown(
            options=self._ch_names, value=self._ch_names[0],
            description='Channel:'
        )

        dd_win = ipy.Dropdown(
            options=[1, 5, 10, 30, 60],
            value=win_size,
            description='Window size:',
        )

        dd_check = ipy.Checkbox(
            value=False,
            description='Filtered',
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

        return ipy.interact(update, epoch=sl_ep, amplitude=sl_amp,
                            channel=dd_ch, win_size=dd_win, filt=dd_check)


#############################################################################
# SPINDLES DETECTION
#############################################################################


def spindles_detect(data, sf=None, ch_names=None, hypno=None,
                    include=(1, 2, 3), freq_sp=(12, 15), freq_broad=(1, 30),
                    duration=(0.5, 2), min_distance=500,
                    thresh={'rel_pow': 0.2, 'corr': 0.65, 'rms': 1.5},
                    multi_only=False, remove_outliers=False, verbose=False):
    """Spindles detection.

    Parameters
    ----------
    data : array_like
        Single or multi-channel data. Unit must be uV and shape (n_samples) or
        (n_chan, n_samples). Can also be a :py:class:`mne.io.BaseRaw`,
        in which case ``data``, ``sf``, and ``ch_names`` will be automatically
        extracted, and ``data`` will also be automatically converted from
        Volts (MNE) to micro-Volts (YASA).
    sf : float
        Sampling frequency of the data in Hz.
        Can be omitted if ``data`` is a :py:class:`mne.io.BaseRaw`.

        .. tip:: If the detection is taking too long, make sure to downsample
            your data to 100 Hz (or 128 Hz). For more details, please refer to
            :py:func:`mne.filter.resample`.
    ch_names : list of str
        Channel names. Can be omitted if ``data`` is a
        :py:class:`mne.io.BaseRaw`.
    hypno : array_like
        Sleep stage (hypnogram). If the hypnogram is loaded, the
        detection will only be applied to the value defined in
        ``include`` (default = N1 + N2 + N3 sleep).

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
        (1, 2, 3), meaning that the detection is applied on N1, N2 and N3
        sleep. This has no effect when ``hypno`` is None.
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

        >>> sp = spindles_detect(...)
        >>> sp.summary()

        This will give a :py:class:`pandas.DataFrame` where each row is a
        detected spindle and each column is a parameter (= feature or property)
        of this spindle. To get the average spindles parameters per channel and
        sleep stage:

        >>> sp.summary(grp_chan=True, grp_stage=True)

    Notes
    -----
    The parameters that are calculated for each spindle are:

    * ``'Start'``: Start time of the spindle, in seconds from the beginning of
      data.
    * ``'Peak'``: Time at the most prominent spindle peak (in seconds).
    * ``'End'`` : End time (in seconds).
    * ``'Duration'``: Duration (in seconds)
    * ``'Amplitude'``: Peak-to-peak amplitude of the (detrended) spindle in
      the raw data (in µV).
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

    References
    ----------
    The sleep spindles detection algorithm is based on:

    * Lacourse, K., Delfrate, J., Beaudry, J., Peppard, P., & Warby, S. C.
      (2018). `A sleep spindle detection algorithm that emulates human expert
      spindle scoring. <https://doi.org/10.1016/j.jneumeth.2018.08.014>`_
      Journal of Neuroscience Methods.

    Examples
    --------
    For a walkthrough of the spindles detection, please refer to the following
    Jupyter notebooks:

    https://github.com/raphaelvallat/yasa/blob/master/notebooks/01_spindles_detection.ipynb

    https://github.com/raphaelvallat/yasa/blob/master/notebooks/02_spindles_detection_multi.ipynb

    https://github.com/raphaelvallat/yasa/blob/master/notebooks/03_spindles_detection_NREM_only.ipynb

    https://github.com/raphaelvallat/yasa/blob/master/notebooks/04_spindles_slow_fast.ipynb
    """
    set_log_level(verbose)

    (data, sf, ch_names, hypno, include, mask, n_chan, n_samples, bad_chan
     ) = _check_data_hypno(data, sf, ch_names, hypno, include)

    # If all channels are bad
    if sum(bad_chan) == n_chan:
        logger.warning('All channels have bad amplitude. Returning None.')
        return None

    # Check detection thresholds
    if 'rel_pow' not in thresh.keys():
        thresh['rel_pow'] = 0.20
    if 'corr' not in thresh.keys():
        thresh['corr'] = 0.65
    if 'rms' not in thresh.keys():
        thresh['rms'] = 1.5
    do_rel_pow = thresh['rel_pow'] not in [None, "none", "None"]
    do_corr = thresh['corr'] not in [None, "none", "None"]
    do_rms = thresh['rms'] not in [None, "none", "None"]
    n_thresh = sum([do_rel_pow, do_corr, do_rms])
    assert n_thresh >= 1, 'At least one threshold must be defined.'

    # Filtering
    nfast = next_fast_len(n_samples)
    # 1) Broadband bandpass filter (optional -- careful of lower freq for PAC)
    data_broad = filter_data(data, sf, freq_broad[0], freq_broad[1], method='fir', verbose=0)
    # 2) Sigma bandpass filter
    # The width of the transition band is set to 1.5 Hz on each side,
    # meaning that for freq_sp = (12, 15 Hz), the -6 dB points are located at
    # 11.25 and 15.75 Hz.
    data_sigma = filter_data(
        data, sf, freq_sp[0], freq_sp[1], l_trans_bandwidth=1.5, h_trans_bandwidth=1.5,
        method='fir', verbose=0)

    # Hilbert power (to define the instantaneous frequency / power)
    analytic = signal.hilbert(data_sigma, N=nfast)[:, :n_samples]
    inst_phase = np.angle(analytic)
    inst_pow = np.square(np.abs(analytic))
    inst_freq = (sf / (2 * np.pi) * np.diff(inst_phase, axis=-1))

    # Extract the SO signal for coupling
    # if coupling:
    #     # We need to use the original (non-filtered data)
    #     data_so = filter_data(data, sf, freq_so[0], freq_so[1], method='fir',
    #                           l_trans_bandwidth=0.1, h_trans_bandwidth=0.1,
    #                           verbose=0)
    #     # Now extract the instantaneous phase using Hilbert transform
    #     so_phase = np.angle(signal.hilbert(data_so, N=nfast)[:, :n_samples])

    # Initialize empty output dataframe
    df = pd.DataFrame()

    for i in range(n_chan):

        # ####################################################################
        # START SINGLE CHANNEL DETECTION
        # ####################################################################

        # First, skip channels with bad data amplitude
        if bad_chan[i]:
            continue

        # Compute the pointwise relative power using interpolated STFT
        # Here we use a step of 200 ms to speed up the computation.
        # Note that even if the threshold is None we still need to calculate it
        # for the individual spindles parameter (RelPow).
        f, t, Sxx = stft_power(
            data_broad[i, :], sf, window=2, step=.2, band=freq_broad, interp=False, norm=True)
        idx_sigma = np.logical_and(f >= freq_sp[0], f <= freq_sp[1])
        rel_pow = Sxx[idx_sigma].sum(0)

        # Let's interpolate `rel_pow` to get one value per sample
        # Note that we could also have use the `interp=True` in the
        # `stft_power` function, however 2D interpolation is much slower than
        # 1D interpolation.
        func = interp1d(t, rel_pow, kind='cubic', bounds_error=False, fill_value=0)
        t = np.arange(n_samples) / sf
        rel_pow = func(t)

        if do_corr:
            _, mcorr = moving_transform(x=data_sigma[i, :], y=data_broad[i, :], sf=sf, window=.3,
                                        step=.1, method='corr', interp=True)
        if do_rms:
            _, mrms = moving_transform(x=data_sigma[i, :], sf=sf, window=.3, step=.1, method='rms',
                                       interp=True)
            # Let's define the thresholds
            if hypno is None:
                thresh_rms = mrms.mean() + thresh['rms'] * trimbothstd(mrms, cut=0.10)
            else:
                thresh_rms = mrms[mask].mean() + thresh['rms'] * trimbothstd(mrms[mask], cut=0.10)
            # Avoid too high threshold caused by Artefacts / Motion during Wake
            thresh_rms = min(thresh_rms, 10)
            logger.info('Moving RMS threshold = %.3f', thresh_rms)

        # Boolean vector of supra-threshold indices
        idx_sum = np.zeros(n_samples)
        if do_rel_pow:
            idx_rel_pow = (rel_pow >= thresh['rel_pow']).astype(int)
            idx_sum += idx_rel_pow
            logger.info('N supra-theshold relative power = %i', idx_rel_pow.sum())
        if do_corr:
            idx_mcorr = (mcorr >= thresh['corr']).astype(int)
            idx_sum += idx_mcorr
            logger.info('N supra-theshold moving corr = %i', idx_mcorr.sum())
        if do_rms:
            idx_mrms = (mrms >= thresh_rms).astype(int)
            idx_sum += idx_mrms
            logger.info('N supra-theshold moving RMS = %i', idx_mrms.sum())

        # Make sure that we do not detect spindles outside mask
        if hypno is not None:
            idx_sum[~mask] = 0

        # The detection using the three thresholds tends to underestimate the
        # real duration of the spindle. To overcome this, we compute a soft
        # threshold by smoothing the idx_sum vector with a 100 ms window.
        w = int(0.1 * sf)
        idx_sum = np.convolve(idx_sum, np.ones(w) / w, mode='same')
        # And we then find indices that are strictly greater than 2, i.e. we
        # find the 'true' beginning and 'true' end of the events by finding
        # where at least two out of the three treshold were crossed.
        where_sp = np.where(idx_sum > (n_thresh - 1))[0]

        # If no events are found, skip to next channel
        if not len(where_sp):
            logger.warning('No spindle were found in channel %s.', ch_names[i])
            continue

        # Merge events that are too close
        if min_distance is not None and min_distance > 0:
            where_sp = _merge_close(where_sp, min_distance, sf)

        # Extract start, end, and duration of each spindle
        sp = np.split(where_sp, np.where(np.diff(where_sp) != 1)[0] + 1)
        idx_start_end = np.array([[k[0], k[-1]] for k in sp]) / sf
        sp_start, sp_end = idx_start_end.T
        sp_dur = sp_end - sp_start

        # Find events with bad duration
        good_dur = np.logical_and(sp_dur > duration[0], sp_dur < duration[1])

        # If no events of good duration are found, skip to next channel
        if all(~good_dur):
            logger.warning('No spindle were found in channel %s.', ch_names[i])
            continue

        # Initialize empty variables
        sp_amp = np.zeros(len(sp))
        sp_freq = np.zeros(len(sp))
        sp_rms = np.zeros(len(sp))
        sp_osc = np.zeros(len(sp))
        sp_sym = np.zeros(len(sp))
        sp_abs = np.zeros(len(sp))
        sp_rel = np.zeros(len(sp))
        sp_sta = np.zeros(len(sp))
        sp_pro = np.zeros(len(sp))
        # sp_cou = np.zeros(len(sp))

        # Number of oscillations (number of peaks separated by at least 60 ms)
        # --> 60 ms because 1000 ms / 16 Hz = 62.5 m, in other words, at 16 Hz,
        # peaks are separated by 62.5 ms. At 11 Hz peaks are separated by 90 ms
        distance = 60 * sf / 1000

        for j in np.arange(len(sp))[good_dur]:
            # Important: detrend the signal to avoid wrong PTP amplitude
            sp_x = np.arange(data_broad[i, sp[j]].size, dtype=np.float64)
            sp_det = _detrend(sp_x, data_broad[i, sp[j]])
            # sp_det = signal.detrend(data_broad[i, sp[i]], type='linear')
            sp_amp[j] = np.ptp(sp_det)  # Peak-to-peak amplitude
            sp_rms[j] = _rms(sp_det)  # Root mean square
            sp_rel[j] = np.median(rel_pow[sp[j]])  # Median relative power

            # Hilbert-based instantaneous properties
            sp_inst_freq = inst_freq[i, sp[j]]
            sp_inst_pow = inst_pow[i, sp[j]]
            sp_abs[j] = np.median(np.log10(sp_inst_pow[sp_inst_pow > 0]))
            sp_freq[j] = np.median(sp_inst_freq[sp_inst_freq > 0])

            # Number of oscillations
            peaks, peaks_params = signal.find_peaks(
                sp_det, distance=distance, prominence=(None, None))
            sp_osc[j] = len(peaks)

            # For frequency and amplitude, we can also optionally use these
            # faster alternatives. If we use them, we do not need to compute
            # the Hilbert transform of the filtered signal.
            # sp_freq[j] = sf / np.mean(np.diff(peaks))
            # sp_amp[j] = peaks_params['prominences'].max()

            # Peak location & symmetry index
            # pk is expressed in sample since the beginning of the spindle
            pk = peaks[peaks_params['prominences'].argmax()]
            sp_pro[j] = sp_start[j] + pk / sf
            sp_sym[j] = pk / sp_det.size

            # SO-spindles coupling
            # if coupling:
            #     sp_cou[j] = so_phase[i, sp[j]][pk]

            # Sleep stage
            if hypno is not None:
                sp_sta[j] = hypno[sp[j]][0]

        # Create a dataframe
        sp_params = {'Start': sp_start,
                     'Peak': sp_pro,
                     'End': sp_end,
                     'Duration': sp_dur,
                     'Amplitude': sp_amp,
                     'RMS': sp_rms,
                     'AbsPower': sp_abs,
                     'RelPower': sp_rel,
                     'Frequency': sp_freq,
                     'Oscillations': sp_osc,
                     'Symmetry': sp_sym,
                     # 'SOPhase': sp_cou,
                     'Stage': sp_sta}

        df_chan = pd.DataFrame(sp_params)[good_dur]

        # We need at least 50 detected spindles to apply the Isolation Forest.
        if remove_outliers and df_chan.shape[0] >= 50:
            col_keep = ['Duration', 'Amplitude', 'RMS', 'AbsPower', 'RelPower',
                        'Frequency', 'Oscillations', 'Symmetry']
            ilf = IsolationForest(
                contamination='auto', max_samples='auto', verbose=0, random_state=42)
            good = ilf.fit_predict(df_chan[col_keep])
            good[good == -1] = 0
            logger.info('%i outliers were removed in channel %s.'
                        % ((good == 0).sum(), ch_names[i]))
            # Remove outliers from DataFrame
            df_chan = df_chan[good.astype(bool)]
            logger.info('%i spindles were found in channel %s.'
                        % (df_chan.shape[0], ch_names[i]))

        # ####################################################################
        # END SINGLE CHANNEL DETECTION
        # ####################################################################
        df_chan['Channel'] = ch_names[i]
        df_chan['IdxChannel'] = i
        df = df.append(df_chan, ignore_index=True)

    # If no spindles were detected, return None
    if df.empty:
        logger.warning('No spindles were found in data. Returning None.')
        return None

    # Remove useless columns
    to_drop = []
    if hypno is None:
        to_drop.append('Stage')
    else:
        df['Stage'] = df['Stage'].astype(int)
    # if not coupling:
    #     to_drop.append('SOPhase')
    if len(to_drop):
        df = df.drop(columns=to_drop)

    # Find spindles that are present on at least two channels
    if multi_only and df['Channel'].nunique() > 1:
        # We round to the nearest second
        idx_good = np.logical_or(
            df['Start'].round(0).duplicated(keep=False),
            df['End'].round(0).duplicated(keep=False)).to_list()
        df = df[idx_good].reset_index(drop=True)

    return SpindlesResults(events=df, data=data, sf=sf, ch_names=ch_names,
                           hypno=hypno, data_filt=data_sigma)


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

    def __init__(self, events, data, sf, ch_names, hypno, data_filt):
        super().__init__(events, data, sf, ch_names, hypno, data_filt)

    def summary(self, grp_chan=False, grp_stage=False, mask=None, aggfunc='mean', sort=True):
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
        return super().summary(event_type='spindles', grp_chan=grp_chan, grp_stage=grp_stage,
                               aggfunc=aggfunc, sort=sort, mask=mask)

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

        >>> (x * y).sum()  # Unscaled coincidence
        3

        >>> (x * y).sum() / (x.sum() * y.sum())  # Scaled coincidence
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

    def get_sync_events(self, center='Peak', time_before=1, time_after=1, filt=(None, None),
                        mask=None, as_dataframe=True):
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
        return super().get_sync_events(center=center, time_before=time_before,
                                       time_after=time_after, filt=filt, mask=mask,
                                       as_dataframe=as_dataframe)

    def plot_average(self, center='Peak', hue='Channel', time_before=1,
                     time_after=1, filt=(None, None), mask=None, figsize=(6, 4.5), **kwargs):
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
        return super().plot_average(event_type='spindles', center=center,
                                    hue=hue, time_before=time_before,
                                    time_after=time_after, filt=filt, mask=mask,
                                    figsize=figsize, **kwargs)

    def plot_detection(self):
        """Plot an overlay of the detected spindles on the EEG signal.

        This only works in Jupyter and it requires the ipywidgets
        (https://ipywidgets.readthedocs.io/en/latest/) package.

        To activate the interactive mode, make sure to run:

        >>> %matplotlib widget

        .. versionadded:: 0.4.0
        """
        return super().plot_detection()


#############################################################################
# SLOW-WAVES DETECTION
#############################################################################


def sw_detect(data, sf=None, ch_names=None, hypno=None, include=(2, 3), freq_sw=(0.3, 1.5),
              dur_neg=(0.3, 1.5), dur_pos=(0.1, 1), amp_neg=(40, 200), amp_pos=(10, 150),
              amp_ptp=(75, 350), coupling=False,
              coupling_params={"freq_sp": (12, 16), "time": 1, "p": 0.05},
              remove_outliers=False, verbose=False):
    """Slow-waves detection.

    Parameters
    ----------
    data : array_like
        Single or multi-channel data. Unit must be uV and shape (n_samples) or
        (n_chan, n_samples). Can also be a :py:class:`mne.io.BaseRaw`,
        in which case ``data``, ``sf``, and ``ch_names`` will be automatically
        extracted, and ``data`` will also be automatically converted from
        Volts (MNE) to micro-Volts (YASA).
    sf : float
        Sampling frequency of the data in Hz.
        Can be omitted if ``data`` is a :py:class:`mne.io.BaseRaw`.

        .. tip:: If the detection is taking too long, make sure to downsample
            your data to 100 Hz (or 128 Hz). For more details, please refer to
            :py:func:`mne.filter.resample`.
    ch_names : list of str
        Channel names. Can be omitted if ``data`` is a
        :py:class:`mne.io.BaseRaw`.
    hypno : array_like
        Sleep stage (hypnogram). If the hypnogram is loaded, the
        detection will only be applied to the value defined in
        ``include`` (default = N2 + N3 sleep).

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
        (2, 3), meaning that the detection is applied on N2 and N3
        sleep. This has no effect when ``hypno`` is None.
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
               mean_direction = pg.circ_mean(sw['PhaseAtSigmaPeak'])
               vector_length = pg.circ_r(sw['PhaseAtSigmaPeak'])

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

        * ``p`` is a parameter passed to the :py:func:`tensorpac.methods.norm_direct_pac``
          function. It represents the p-value to use for thresholding of unreliable coupling
          values. Sub-threshold PAC values will be set to 0. To disable this behavior (no masking),
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

        >>> sw = sw_detect(...)
        >>> sw.summary()

        This will give a :py:class:`pandas.DataFrame` where each row is a
        detected slow-wave and each column is a parameter (= property).
        To get the average SW parameters per channel and sleep stage:

        >>> sw.summary(grp_chan=True, grp_stage=True)

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

    .. image:: https://raw.githubusercontent.com/raphaelvallat/yasa/master/docs/pictures/slow_waves.png  # noqa
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
    For an example of how to run the detection, please refer to the tutorial:
    https://github.com/raphaelvallat/yasa/blob/master/notebooks/05_sw_detection.ipynb
    """
    set_log_level(verbose)

    (data, sf, ch_names, hypno, include, mask, n_chan, n_samples, bad_chan
     ) = _check_data_hypno(data, sf, ch_names, hypno, include)

    # If all channels are bad
    if sum(bad_chan) == n_chan:
        logger.warning('All channels have bad amplitude. Returning None.')
        return None

    # Define time vector
    times = np.arange(data.size) / sf
    idx_mask = np.where(mask)[0]

    # Bandpass filter
    nfast = next_fast_len(n_samples)
    data_filt = filter_data(
        data, sf, freq_sw[0], freq_sw[1], method='fir', verbose=0, l_trans_bandwidth=0.2,
        h_trans_bandwidth=0.2)

    # Extract the spindles-related sigma signal for coupling
    if coupling:
        is_tensorpac_installed()
        import tensorpac.methods as tpm
        # The width of the transition band is set to 1.5 Hz on each side,
        # meaning that for freq_sp = (12, 15 Hz), the -6 dB points are located
        # at 11.25 and 15.75 Hz. The frequency band for the amplitude signal
        # must be large enough to fit the sidebands caused by the assumed
        # modulating lower frequency band (Aru et al. 2015).
        # https://doi.org/10.1016/j.conb.2014.08.002
        assert isinstance(coupling_params, dict)
        assert "freq_sp" in coupling_params.keys()
        assert "time" in coupling_params.keys()
        assert "p" in coupling_params.keys()
        freq_sp = coupling_params['freq_sp']
        data_sp = filter_data(
            data, sf, freq_sp[0], freq_sp[1], method='fir', l_trans_bandwidth=1.5,
            h_trans_bandwidth=1.5, verbose=0)
        # Now extract the instantaneous phase/amplitude using Hilbert transform
        sw_pha = np.angle(signal.hilbert(data_filt, N=nfast)[:, :n_samples])
        sp_amp = np.abs(signal.hilbert(data_sp, N=nfast)[:, :n_samples])

    # Initialize empty output dataframe
    df = pd.DataFrame()

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
        # Intersect with sleep stage vector
        idx_neg_peaks = np.intersect1d(idx_neg_peaks, idx_mask, assume_unique=True)
        idx_pos_peaks = np.intersect1d(idx_pos_peaks, idx_mask, assume_unique=True)

        # If no peaks are detected, return None
        if len(idx_neg_peaks) == 0 or len(idx_pos_peaks) == 0:
            logger.warning('No SW were found in channel %s.', ch_names[i])
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
        sw_ptp = (np.abs(data_filt[i, idx_neg_peaks]) + data_filt[i, idx_pos_peaks])
        good_ptp = np.logical_and(sw_ptp > amp_ptp[0], sw_ptp < amp_ptp[1])

        # If good_ptp is all False
        if all(~good_ptp):
            logger.warning('No SW were found in channel %s.', ch_names[i])
            continue

        sw_ptp = sw_ptp[good_ptp]
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
        sw_start = times[idx_neg_peaks + previous_neg_zc]
        sw_end = times[idx_pos_peaks + following_pos_zc]
        # This should be the same as `sw_dur = pos_phase_dur + neg_phase_dur`
        # We round to avoid floating point errr (e.g. 1.9000000002)
        sw_dur = (sw_end - sw_start).round(4)
        sw_dur_both_phase = (pos_phase_dur + neg_phase_dur).round(4)
        sw_midcrossing = times[idx_neg_peaks + following_neg_zc]
        sw_idx_neg = times[idx_neg_peaks]  # Location of negative peak
        sw_idx_pos = times[idx_pos_peaks]  # Location of positive peak
        # Slope between peak trough and midcrossing
        sw_slope = sw_ptp / (sw_midcrossing - sw_idx_neg)
        # Hypnogram
        if hypno is not None:
            sw_sta = hypno[idx_neg_peaks]
        else:
            sw_sta = np.zeros(sw_dur.shape)

        # And we apply a set of thresholds to remove bad slow waves
        good_sw = np.logical_and.reduce((
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
            sw_slope > 0,
        ))

        if all(~good_sw):
            logger.warning('No SW were found in channel %s.', ch_names[i])
            continue

        # Filter good events
        idx_neg_peaks = idx_neg_peaks[good_sw]
        idx_pos_peaks = idx_pos_peaks[good_sw]
        sw_start = sw_start[good_sw]
        sw_idx_neg = sw_idx_neg[good_sw]
        sw_midcrossing = sw_midcrossing[good_sw]
        sw_idx_pos = sw_idx_pos[good_sw]
        sw_end = sw_end[good_sw]
        sw_dur = sw_dur[good_sw]
        sw_ptp = sw_ptp[good_sw]
        sw_slope = sw_slope[good_sw]
        sw_sta = sw_sta[good_sw]

        # Create a dictionnary
        sw_params = OrderedDict({
            'Start': sw_start,
            'NegPeak': sw_idx_neg,
            'MidCrossing': sw_midcrossing,
            'PosPeak': sw_idx_pos,
            'End': sw_end,
            'Duration': sw_dur,
            'ValNegPeak': data_filt[i, idx_neg_peaks],
            'ValPosPeak': data_filt[i, idx_pos_peaks],
            'PTP': sw_ptp,
            'Slope': sw_slope,
            'Frequency': 1 / sw_dur,
            'Stage': sw_sta,
        })

        # Add phase (in radians) of slow-oscillation signal at maximum
        # spindles-related sigma amplitude within a XX-seconds centered epochs.
        if coupling:
            # Get phase and amplitude for each centered epoch
            time_before = time_after = coupling_params['time']
            assert float(sf * time_before).is_integer(), (
                "Invalid time parameter for coupling. Must be a whole number of samples.")
            bef = int(sf * time_before)
            aft = int(sf * time_after)
            # Center of each epoch is defined as the negative peak of the SW
            n_peaks = idx_neg_peaks.shape[0]
            # idx.shape = (len(idx_valid), bef + aft + 1)
            idx, idx_valid = get_centered_indices(data[i, :], idx_neg_peaks, bef, aft)
            sw_pha_ev = sw_pha[i, idx]
            sp_amp_ev = sp_amp[i, idx]
            # 1) Find location of max sigma amplitude in epoch
            idx_max_amp = sp_amp_ev.argmax(axis=1)
            # Now we need to append it back to the original unmasked shape
            # to avoid error when idx.shape[0] != idx_valid.shape, i.e.
            # some epochs were out of data bounds.
            sw_params['SigmaPeak'] = np.ones(n_peaks) * np.nan
            # Timestamp at sigma peak, expressed in seconds from negative peak
            # e.g. -0.39, 0.5, 1, 2 -- limits are [time_before, time_after]
            time_sigpk = (idx_max_amp - bef) / sf
            # convert to absolute time from beginning of the recording
            # time_sigpk only includes valid epoch
            time_sigpk_abs = sw_idx_neg[idx_valid] + time_sigpk
            sw_params['SigmaPeak'][idx_valid] = time_sigpk_abs
            # 2) PhaseAtSigmaPeak
            # Find SW phase at max sigma amplitude in epoch
            pha_at_max = np.squeeze(np.take_along_axis(sw_pha_ev, idx_max_amp[..., None], axis=1))
            sw_params['PhaseAtSigmaPeak'] = np.ones(n_peaks) * np.nan
            sw_params['PhaseAtSigmaPeak'][idx_valid] = pha_at_max
            # 3) Normalized Direct PAC, with thresholding
            # Unreliable values are set to 0
            ndp = np.squeeze(tpm.norm_direct_pac(
                sw_pha_ev[None, ...], sp_amp_ev[None, ...], p=coupling_params['p']))
            sw_params['ndPAC'] = np.ones(n_peaks) * np.nan
            sw_params['ndPAC'][idx_valid] = ndp
            # Make sure that Stage is the last column of the dataframe
            sw_params.move_to_end('Stage')

        # Convert to dataframe, keeping only good events
        df_chan = pd.DataFrame(sw_params)

        # Remove all duplicates
        df_chan = df_chan.drop_duplicates(subset=['Start'], keep=False)
        df_chan = df_chan.drop_duplicates(subset=['End'], keep=False)

        # We need at least 50 detected slow waves to apply the Isolation Forest
        if remove_outliers and df_chan.shape[0] >= 50:
            col_keep = ['Duration', 'ValNegPeak', 'ValPosPeak', 'PTP', 'Slope', 'Frequency']
            ilf = IsolationForest(contamination='auto', max_samples='auto',
                                  verbose=0, random_state=42)
            good = ilf.fit_predict(df_chan[col_keep])
            good[good == -1] = 0
            logger.info('%i outliers were removed in channel %s.'
                        % ((good == 0).sum(), ch_names[i]))
            # Remove outliers from DataFrame
            df_chan = df_chan[good.astype(bool)]
            logger.info('%i slow-waves were found in channel %s.'
                        % (df_chan.shape[0], ch_names[i]))

        # ####################################################################
        # END SINGLE CHANNEL DETECTION
        # ####################################################################

        df_chan['Channel'] = ch_names[i]
        df_chan['IdxChannel'] = i
        df = df.append(df_chan, ignore_index=True)

    # If no SW were detected, return None
    if df.empty:
        logger.warning('No SW were found in data. Returning None.')
        return None

    if hypno is None:
        df = df.drop(columns=['Stage'])
    else:
        df['Stage'] = df['Stage'].astype(int)

    return SWResults(events=df, data=data, sf=sf, ch_names=ch_names,
                     hypno=hypno, data_filt=data_filt)


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

    def __init__(self, events, data, sf, ch_names, hypno, data_filt):
        super().__init__(events, data, sf, ch_names, hypno, data_filt)

    def summary(self, grp_chan=False, grp_stage=False, mask=None, aggfunc='mean', sort=True):
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
        return super().summary(event_type='sw', grp_chan=grp_chan, grp_stage=grp_stage,
                               aggfunc=aggfunc, sort=sort, mask=mask)

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
        distance_sp_to_sw_peak = []
        cooccurring_spindle_peaks = []

        # Find intersecting channels
        common_ch = np.intersect1d(self._events['Channel'].unique(), spindles['Channel'].unique())
        assert len(common_ch), "No common channel(s) were found."

        # Loop across channels
        for chan in self._events['Channel'].unique():
            sw_chan_peaks = self._events[self._events["Channel"] == chan]["NegPeak"].to_numpy()
            sp_chan_peaks = spindles[spindles["Channel"] == chan]['Peak'].to_numpy()
            # Loop across individual slow-waves
            for sw_negpeak in sw_chan_peaks:
                start = sw_negpeak - lookaround
                end = sw_negpeak + lookaround
                mask = np.logical_and(start < sp_chan_peaks, sp_chan_peaks < end)
                if any(mask):
                    # If multiple spindles are present, take the last one
                    sp_peak = sp_chan_peaks[mask][-1]
                    cooccurring_spindle_peaks.append(sp_peak)
                    distance_sp_to_sw_peak.append(sp_peak - sw_negpeak)
                else:
                    cooccurring_spindle_peaks.append(np.nan)
                    distance_sp_to_sw_peak.append(np.nan)

        # Add columns to self._events: IN-PLACE MODIFICATION!
        self._events["CooccurringSpindle"] = ~np.isnan(distance_sp_to_sw_peak)
        self._events["CooccurringSpindlePeak"] = cooccurring_spindle_peaks
        self._events['DistanceSpindleToSW'] = distance_sp_to_sw_peak

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

        >>> (x * y).sum()  # Coincidence
        3

        >>> (x * y).sum() / (x.sum() * y.sum())  # Scaled coincidence
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

    def get_sync_events(self, center='NegPeak', time_before=0.4, time_after=0.8, filt=(None, None),
                        mask=None, as_dataframe=True):
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
            center=center, time_before=time_before, time_after=time_after, filt=filt, mask=mask,
            as_dataframe=as_dataframe)

    def plot_average(self, center='NegPeak', hue='Channel', time_before=0.4, time_after=0.8,
                     filt=(None, None), mask=None, figsize=(6, 4.5), **kwargs):
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
            event_type='sw', center=center, hue=hue, time_before=time_before,
            time_after=time_after, filt=filt, mask=mask, figsize=figsize, **kwargs)

    def plot_detection(self):
        """Plot an overlay of the detected slow-waves on the EEG signal.

        This only works in Jupyter and it requires the ipywidgets
        (https://ipywidgets.readthedocs.io/en/latest/) package.

        To activate the interactive mode, make sure to run:

        >>> %matplotlib widget

        .. versionadded:: 0.4.0
        """
        return super().plot_detection()


#############################################################################
# REMs DETECTION
#############################################################################


def rem_detect(loc, roc, sf, hypno=None, include=4, amplitude=(50, 325), duration=(0.3, 1.2),
               freq_rem=(0.5, 5), remove_outliers=False, verbose=False):
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
            you need to multiply the data by 1e6 to convert to micro-Volts
            (1 V = 1,000,000 uV), e.g.:

            >>> data = raw.get_data() * 1e6  # Make sure that data is in uV
    sf : float
        Sampling frequency of the data, in Hz.
    hypno : array_like
        Sleep stage (hypnogram). If the hypnogram is loaded, the
        detection will only be applied to the value defined in
        ``include`` (default = REM sleep).

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
        (4), meaning that the detection is applied on REM sleep.
        This has no effect when ``hypno`` is None.
    amplitude : tuple or list
        Minimum and maximum amplitude of the peak of the REM.
        Default is 50 uV to 325 uV.
    duration : tuple or list
        The minimum and maximum duration of the REMs.
        Default is 0.3 to 1.2 seconds.
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

        >>> rem = rem_detect(...)
        >>> rem.summary()

        This will give a :py:class:`pandas.DataFrame` where each row is a
        detected REM and each column is a parameter (= property).
        To get the average parameters sleep stage:

        >>> rem.summary(grp_stage=True)

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
    For an example of how to run the detection, please refer to
    https://github.com/raphaelvallat/yasa/blob/master/notebooks/07_REMs_detection.ipynb
    """
    set_log_level(verbose)
    # Safety checks
    loc = np.squeeze(np.asarray(loc, dtype=np.float64))
    roc = np.squeeze(np.asarray(roc, dtype=np.float64))
    assert loc.ndim == 1, 'LOC must be 1D.'
    assert roc.ndim == 1, 'ROC must be 1D.'
    assert loc.size == roc.size, 'LOC and ROC must have the same size.'
    data = np.vstack((loc, roc))

    (data, sf, ch_names, hypno, include, mask, n_chan, n_samples, bad_chan
     ) = _check_data_hypno(data, sf, ['LOC', 'ROC'], hypno, include)

    # If all channels are bad
    if any(bad_chan):
        logger.warning('At least one channel has bad amplitude. '
                       'Returning None.')
        return None

    # Bandpass filter
    data_filt = filter_data(data, sf, freq_rem[0], freq_rem[1], verbose=0)

    # Calculate the negative product of LOC and ROC, maximal during REM.
    negp = -data_filt[0, :] * data_filt[1, :]

    # Find peaks in data
    # - height: required height of peaks (min and max.)
    # - distance: required distance in samples between neighboring peaks.
    # - prominence: required prominence of peaks.
    # - wlen: limit search for bases to a specific window.
    hmin, hmax = amplitude[0]**2, amplitude[1]**2
    pks, pks_params = signal.find_peaks(negp, height=(hmin, hmax), distance=(duration[0] * sf),
                                        prominence=(0.8 * hmin), wlen=(duration[1] * sf))

    # Intersect with sleep stage vector
    # We do that before calculating the features in order to gain some time
    idx_mask = np.where(mask)[0]
    pks, idx_good, _ = np.intersect1d(pks, idx_mask, True, True)
    for k in pks_params.keys():
        pks_params[k] = pks_params[k][idx_good]

    # If no peaks are detected, return None
    if len(pks) == 0:
        logger.warning('No REMs were found in data. Returning None.')
        return None

    # Hypnogram
    if hypno is not None:
        # The sleep stage at the beginning of the REM is considered.
        rem_sta = hypno[pks_params['left_bases']]
    else:
        rem_sta = np.zeros(pks.shape)

    # Calculate time features
    pks_params['Start'] = pks_params['left_bases'] / sf
    pks_params['Peak'] = pks / sf
    pks_params['End'] = pks_params['right_bases'] / sf
    pks_params['Duration'] = pks_params['End'] - pks_params['Start']
    # Time points in minutes (HH:MM:SS)
    # pks_params['StartMin'] = pd.to_timedelta(pks_params['Start'], unit='s').dt.round('s')  # noqa
    # pks_params['PeakMin'] = pd.to_timedelta(pks_params['Peak'], unit='s').dt.round('s')  # noqa
    # pks_params['EndMin'] = pd.to_timedelta(pks_params['End'], unit='s').dt.round('s')  # noqa
    # Absolute LOC / ROC value at peak (filtered)
    pks_params['LOCAbsValPeak'] = abs(data_filt[0, pks])
    pks_params['ROCAbsValPeak'] = abs(data_filt[1, pks])
    # Absolute rising and falling slope
    dist_pk_left = (pks - pks_params['left_bases']) / sf
    dist_pk_right = (pks_params['right_bases'] - pks) / sf
    locrs = (data_filt[0, pks] - data_filt[0, pks_params['left_bases']]) / dist_pk_left
    rocrs = (data_filt[1, pks] - data_filt[1, pks_params['left_bases']]) / dist_pk_left
    locfs = (data_filt[0, pks_params['right_bases']] - data_filt[0, pks]) / dist_pk_right
    rocfs = (data_filt[1, pks_params['right_bases']] - data_filt[1, pks]) / dist_pk_right
    pks_params['LOCAbsRiseSlope'] = abs(locrs)
    pks_params['ROCAbsRiseSlope'] = abs(rocrs)
    pks_params['LOCAbsFallSlope'] = abs(locfs)
    pks_params['ROCAbsFallSlope'] = abs(rocfs)
    pks_params['Stage'] = rem_sta  # Sleep stage

    # Convert to Pandas DataFrame
    df = pd.DataFrame(pks_params)

    # Make sure that the sign of ROC and LOC is opposite
    df['IsOppositeSign'] = (np.sign(data_filt[1, pks]) != np.sign(data_filt[0, pks]))
    df = df[np.sign(data_filt[1, pks]) != np.sign(data_filt[0, pks])]

    # Remove bad duration
    tmin, tmax = duration
    good_dur = np.logical_and(pks_params['Duration'] >= tmin, pks_params['Duration'] < tmax)
    df = df[good_dur]

    # Keep only useful channels
    df = df[['Start', 'Peak', 'End', 'Duration', 'LOCAbsValPeak', 'ROCAbsValPeak',
             'LOCAbsRiseSlope', 'ROCAbsRiseSlope', 'LOCAbsFallSlope', 'ROCAbsFallSlope', 'Stage']]

    if hypno is None:
        df = df.drop(columns=['Stage'])
    else:
        df['Stage'] = df['Stage'].astype(int)

    # We need at least 50 detected REMs to apply the Isolation Forest.
    if remove_outliers and df.shape[0] >= 50:
        col_keep = ['Duration', 'LOCAbsValPeak', 'ROCAbsValPeak', 'LOCAbsRiseSlope',
                    'ROCAbsRiseSlope', 'LOCAbsFallSlope', 'ROCAbsFallSlope']
        ilf = IsolationForest(contamination='auto', max_samples='auto',
                              verbose=0, random_state=42)
        good = ilf.fit_predict(df[col_keep])
        good[good == -1] = 0
        logger.info('%i outliers were removed.', (good == 0).sum())
        # Remove outliers from DataFrame
        df = df[good.astype(bool)]

    logger.info('%i REMs were found in data.', df.shape[0])
    df = df.reset_index(drop=True)
    return REMResults(events=df, data=data, sf=sf, ch_names=ch_names,
                      hypno=hypno, data_filt=data_filt)


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

    def __init__(self, events, data, sf, ch_names, hypno, data_filt):
        super().__init__(events, data, sf, ch_names, hypno, data_filt)

    def summary(self, grp_stage=False, mask=None, aggfunc='mean', sort=True):
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
        return super().summary(event_type='rem', grp_chan=False, grp_stage=grp_stage,
                               aggfunc=aggfunc, sort=sort, mask=mask)

    def get_mask(self):
        """Return a boolean array indicating for each sample in data if this
        sample is part of a detected event (True) or not (False).
        """
        # We cannot use super() because "Channel" is not present in _events.
        from yasa.others import _index_to_events
        mask = np.zeros(self._data.shape, dtype=int)
        idx_ev = _index_to_events(
            self._events[['Start', 'End']].to_numpy() * self._sf)
        mask[:, idx_ev] = 1
        return mask

    def get_sync_events(self, center='Peak', time_before=0.4, time_after=0.4,
                        filt=(None, None), mask=None):
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

        Returns
        -------
        df_sync : :py:class:`pandas.DataFrame`
            Ouput long-format dataframe::

            'Event' : Event number
            'Time' : Timing of the events (in seconds)
            'Amplitude' : Raw or filtered data for event
            'Channel' : Channel
            'IdxChannel' : Index of channel in data
        """
        from yasa.others import get_centered_indices
        assert time_before >= 0
        assert time_after >= 0
        bef = int(self._sf * time_before)
        aft = int(self._sf * time_after)

        if any(filt):
            data = mne.filter.filter_data(
                self._data, self._sf, l_freq=filt[0], h_freq=filt[1], method='fir', verbose=False)
        else:
            data = self._data

        # Apply mask
        mask = self._check_mask(mask)
        masked_events = self._events.loc[mask, :]

        time = np.arange(-bef, aft + 1, dtype='int') / self._sf
        # Get location of peaks in data
        peaks = (masked_events[center] * self._sf).astype(int).to_numpy()
        # Get centered indices (here we could use second channel as well).
        idx, idx_valid = get_centered_indices(data[0, :], peaks, bef, aft)
        # If no good epochs are returned raise a warning
        assert len(idx_valid), (
            'Time before and/or time after exceed data bounds, please '
            'lower the temporal window around center.')

        # Initialize empty dataframe
        df_sync = pd.DataFrame()

        # Loop across both EOGs (LOC and ROC)
        for i, ch in enumerate(self._ch_names):
            amps = data[i, idx]
            df_chan = pd.DataFrame(amps.T)
            df_chan['Time'] = time
            df_chan = df_chan.melt(id_vars='Time', var_name='Event', value_name='Amplitude')
            df_chan['Channel'] = ch
            df_chan['IdxChannel'] = i
            df_sync = df_sync.append(df_chan, ignore_index=True)

        return df_sync

    def plot_average(self, center='Peak', time_before=0.4, time_after=0.4, filt=(None, None),
                     mask=None, figsize=(6, 4.5), **kwargs):
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
        import seaborn as sns
        import matplotlib.pyplot as plt

        df_sync = self.get_sync_events(center=center, time_before=time_before,
                                       time_after=time_after, filt=filt, mask=mask)

        # Start figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        sns.lineplot(data=df_sync, x='Time', y='Amplitude', hue='Channel', ax=ax, **kwargs)
        # ax.legend(frameon=False, loc='lower right')
        ax.set_xlim(df_sync['Time'].min(), df_sync['Time'].max())
        ax.set_title("Average REM")
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Amplitude (uV)')
        return ax


#############################################################################
# ARTEFACT DETECTION
#############################################################################


def art_detect(data, sf=None, window=5, hypno=None, include=(1, 2, 3, 4),
               method='covar', threshold=3, n_chan_reject=1, verbose=False):
    r"""
    Automatic artifact rejection.

    .. versionadded:: 0.2.0

    Parameters
    ----------
    data : array_like
        Single or multi-channel EEG data.
        Unit must be uV and shape *(n_chan, n_samples)*.
        Can also be a :py:class:`mne.io.BaseRaw`, in which case ``data``
        and ``sf`` will be automatically extracted,
        and ``data`` will also be automatically converted from Volts (MNE)
        to micro-Volts (YASA).

        .. warning::
            ``data`` must only contains EEG channels. Please make sure to
            exclude any EOG, EKG or EMG channels.
    sf : float
        Sampling frequency of the data in Hz.
        Can be omitted if ``data`` is a :py:class:`mne.io.BaseRaw` object.
    window : float
        The window length (= resolution) for artifact rejection, in seconds.
        Default to 5 seconds. Shorter windows (e.g. 1 or 2-seconds) will
        drastically increase computation time when ``method='covar'``.
    hypno : array_like
        Sleep stage (hypnogram). If the hypnogram is passed, the
        detection will be applied separately for each of the stages defined in
        ``include``.

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
        Sleep stages in ``hypno`` on which to perform the artifact rejection.
        The default is ``hypno=(1, 2, 3, 4)``, meaning that the artifact
        rejection is applied separately for all sleep stages, excluding wake.
        This parameter has no effect when ``hypno`` is None.
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
        <https://mne.tools/stable/auto_tutorials/preprocessing/plot_10_preprocessing_overview.html>`_.

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
    :py:class:`pyriemann.clustering.Potato` class implemented in the
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
      <https://hal.archives-ouvertes.fr/hal-00781701/>`_ TOBI
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
    For an example of how to run the detection, please refer to
    https://github.com/raphaelvallat/yasa/blob/master/notebooks/13_artifact_rejection.ipynb
    """
    ###########################################################################
    # PREPROCESSING
    ###########################################################################
    set_log_level(verbose)

    (data, sf, _, hypno, include, _, n_chan, n_samples, _
     ) = _check_data_hypno(data, sf, ch_names=None, hypno=hypno, include=include, check_amp=False)

    assert isinstance(n_chan_reject, int), 'n_chan_reject must be int.'
    assert n_chan_reject >= 1, 'n_chan_reject must be >= 1.'
    assert n_chan_reject <= n_chan, 'n_chan_reject must be <= n_chan.'

    # Safety check: sampling frequency and window
    assert isinstance(sf, (int, float)), 'sf must be int or float'
    assert isinstance(window, (int, float)), 'window must be int or float'
    if isinstance(sf, float):
        assert sf.is_integer(), 'sf must be a whole number.'
        sf = int(sf)
    win_sec = window
    window = win_sec * sf  # Convert window to samples
    if isinstance(window, float):
        assert window.is_integer(), 'window * sf must be a whole number.'
        window = int(window)

    # Safety check: hypnogram
    if hypno is not None:
        # Extract hypnogram with only complete epochs
        idx_max_full_epoch = int(np.floor(n_samples / window))
        hypno_win = hypno[::window][:idx_max_full_epoch]

    # Safety checks: methods
    assert isinstance(method, str), "method must be a string."
    method = method.lower()
    if method in ['cov', 'covar', 'covariance', 'riemann', 'potato']:
        method = 'covar'
        is_pyriemann_installed()
        from pyriemann.estimation import Covariances, Shrinkage
        from pyriemann.clustering import Potato
        # Must have at least 4 channels to use method='covar'
        if n_chan <= 4:
            logger.warning("Must have at least 4 channels for method='covar'. "
                           "Automatically switching to method='std'.")
            method = 'std'

    ###########################################################################
    # START THE REJECTION
    ###########################################################################
    # Remove flat channels
    isflat = (np.nanstd(data, axis=-1) == 0)
    if isflat.any():
        logger.warning('Flat channel(s) were found and removed in data.')
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
    if 'hypno_win' not in locals():
        # [-2, -2, -2, -2, ...], where -2 stands for unscored
        hypno_win = -2 * np.ones(n_epochs, dtype='float')
        include = np.array([-2], dtype='float')

    # We want to make sure that hypno-win and n_epochs have EXACTLY same shape
    assert n_epochs == hypno_win.shape[-1], 'Hypno and epochs do not match.'

    # Finally, we make sure not to include any flat epochs in calculation
    # just using a random number that is unlikely to be picked by users
    if n_flat_epochs > 0:
        hypno_win[where_flat_epochs] = -111991

    # Add logger info
    logger.info('Number of channels in data = %i', n_chan)
    logger.info('Number of samples in data = %i', n_samples)
    logger.info('Sampling frequency = %.2f Hz', sf)
    logger.info('Data duration = %.2f seconds', n_samples / sf)
    logger.info('Number of epochs = %i' % n_epochs)
    logger.info('Artifact window = %.2f seconds' % win_sec)
    logger.info('Method = %s' % method)
    logger.info('Threshold = %.2f standard deviations' % threshold)

    # Create empty `hypno_art` vector (1 sample = 1 epoch)
    epoch_is_art = np.zeros(n_epochs, dtype='int')

    if method == 'covar':
        # Calculate the covariance matrices,
        # shape (n_epochs, n_chan, n_chan)
        covmats = Covariances().fit_transform(epochs)
        # Shrink the covariance matrix (ensure positive semi-definite)
        covmats = Shrinkage().fit_transform(covmats)
        # Define Potato instance: 0 = clean, 1 = art
        # To increase speed we set the max number of iterations from 10 to 100
        potato = Potato(metric='riemann', threshold=threshold, pos_label=0,
                        neg_label=1, n_iter_max=10)
        # Create empty z-scores output (n_epochs)
        zscores = np.zeros(n_epochs, dtype='float') * np.nan

        for stage in include:
            where_stage = np.where(hypno_win == stage)[0]
            # At least 30 epochs are required to calculate z-scores
            # which amounts to 2.5 minutes when using 5-seconds window
            if where_stage.size < 30:
                if hypno is not None:
                    # Only show warnig if user actually pass an hypnogram
                    logger.warning(f"At least 30 epochs are required to "
                                   f"calculate z-score. Skipping "
                                   f"stage {stage}")
                continue
            # Apply Potato algorithm, extract z-scores and labels
            zs = potato.fit_transform(covmats[where_stage])
            art = potato.predict(covmats[where_stage]).astype(int)
            if hypno is not None:
                # Only shows if user actually pass an hypnogram
                perc_reject = 100 * (art.sum() / art.size)
                text = (f"Stage {stage}: {art.sum()} / {art.size} "
                        f"epochs rejected ({perc_reject:.2f}%)")
                logger.info(text)
            # Append to global vector
            epoch_is_art[where_stage] = art
            zscores[where_stage] = zs

    elif method in ['std', 'sd']:
        # Calculate log-transformed standard dev in each epoch
        # We add 1 to avoid log warning id std is zero (e.g. flat line)
        # (n_epochs, n_chan)
        std_epochs = np.log(np.nanstd(epochs, axis=-1) + 1)
        # Create empty zscores output (n_epochs, n_chan)
        zscores = np.zeros((n_epochs, n_chan), dtype='float') * np.nan
        for stage in include:
            where_stage = np.where(hypno_win == stage)[0]
            # At least 30 epochs are required to calculate z-scores
            # which amounts to 2.5 minutes when using 5-seconds window
            if where_stage.size < 30:
                if hypno is not None:
                    # Only show warnig if user actually pass an hypnogram
                    logger.warning(f"At least 30 epochs are required to "
                                   f"calculate z-score. Skipping "
                                   f"stage {stage}")
                continue
            # Calculate z-scores of STD for each channel x stage
            c_mean = np.nanmean(std_epochs[where_stage], axis=0, keepdims=True)
            c_std = np.nanstd(std_epochs[where_stage], axis=0, keepdims=True)
            zs = (std_epochs[where_stage] - c_mean) / c_std
            # Any epoch with at least X channel above or below threshold
            n_chan_supra = (np.abs(zs) > threshold).sum(axis=1)  # >
            art = (n_chan_supra >= n_chan_reject).astype(int)  # >= !
            if hypno is not None:
                # Only shows if user actually pass an hypnogram
                perc_reject = 100 * (art.sum() / art.size)
                text = (f"Stage {stage}: {art.sum()} / {art.size} "
                        f"epochs rejected ({perc_reject:.2f}%)")
                logger.info(text)
            # Append to global vector
            epoch_is_art[where_stage] = art
            zscores[where_stage, :] = zs

    # Mark flat epochs as artefacts
    if n_flat_epochs > 0:
        logger.info(f"Rejecting {n_flat_epochs} epochs with >=50% of channels "
                    f"that are flat. Z-scores set to np.nan for these epochs.")
        epoch_is_art[where_flat_epochs] = 1

    # Log total percentage of epochs rejected
    perc_reject = 100 * (epoch_is_art.sum() / n_epochs)
    text = (f"TOTAL: {epoch_is_art.sum()} / {n_epochs} epochs rejected ({perc_reject:.2f}%)")
    logger.info(text)

    # Convert epoch_is_art to boolean [0, 0, 1] -- > [False, False, True]
    epoch_is_art = epoch_is_art.astype(bool)
    return epoch_is_art, zscores
