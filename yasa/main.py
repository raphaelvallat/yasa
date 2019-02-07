"""
YASA (Yet Another Spindle Algorithm) is a fast and data-agnostic sleep
spindles detection algorithm written in Python 3.

The algorithm behind YASA is largely inspired by the method described in:

Lacourse, K., Delfrate, J., Beaudry, J., Peppard, P., Warby, S.C., 2018.
A sleep spindle detection algorithm that emulates human expert spindle scoring.
J. Neurosci. Methods. https://doi.org/10.1016/j.jneumeth.2018.08.014

- Author: Raphael Vallat (www.raphaelvallat.com)
- Creation date: December 2018
- GitHub: https://github.com/raphaelvallat/yasa
- License: BSD 3-Clause License
"""
import logging
import numpy as np
import pandas as pd
from numba import jit
from scipy import signal
from scipy.fftpack import next_fast_len
from mne.filter import filter_data, resample
from scipy.interpolate import interp1d, RectBivariateSpline

logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

logger = logging.getLogger('yasa')

__all__ = ['spindles_detect', 'spindles_detect_multi', 'stft_power',
           'moving_transform', 'get_bool_vector']


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

#############################################################################
# HELPER FUNCTIONS
#############################################################################


def moving_transform(x, y=None, sf=100, window=.3, step=.1, method='corr',
                     interp=False):
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

            'rms' : root mean square of x
            'corr' : Correlation between x and y
            'covar' : Covariance between x and y
    interp : boolean
        If True, a cubic interpolation is performed to ensure that the output
        is the same size as the input (= pointwise power).

    Returns
    -------
    t : np.array
        Time vector
    out : np.array
        Transformed signal

    Notes
    -----
    This function was inspired by the `transform_signal` function of the
    Wonambi package (https://github.com/wonambi-python/wonambi).
    """
    # Safety checks
    assert method in ['covar', 'corr', 'rms']
    x = np.asarray(x, dtype=np.float64)
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

    # Define beginning, end and time (centered) vector
    beg = ((idx - halfdur) * sf).astype(int)
    beg[beg < 0] = 0
    end = ((idx + halfdur) * sf).astype(int)
    end[end > last] = last
    t = np.column_stack((beg, end)).mean(1) / sf

    if method == 'covar':
        def func(x, y):
            return _covar(x, y)

    elif method == 'corr':
        def func(x, y):
            return _corr(x, y)

    else:
        def func(x):
            return _rms(x)

    # Now loop over successive epochs
    if method in ['covar', 'corr']:
        for i in range(idx.size):
            out[i] = func(x[beg[i]:end[i]], y[beg[i]:end[i]])
    else:
        for i in range(idx.size):
            out[i] = func(x[beg[i]:end[i]])

    # Finally interpolate
    if interp and step != 1 / sf:
        f = interp1d(t, out, kind='cubic',
                     bounds_error=False,
                     fill_value=0)
        t = np.arange(n) / sf
        out = f(t)

    return t, out


def stft_power(data, sf, window=2, step=.2, band=(1, 30), interp=True,
               norm=False):
    """Compute the pointwise power via STFT and interpolation.

    Parameters
    ----------
    data : array_like
        Single-channel data.
    sf : float
        Sampling frequency of the data.
    window : int
        Window size in seconds for STFT.
        2 or 4 seconds are usually a good default.
        Higher values = higher frequency resolution = lower time resolution.
    step : int
        Step in seconds for the STFT.
        A step of 0.2 second (200 ms) is usually a good default.
        If step == 0, overlap at every sample (slowest)
        If step == nperseg, no overlap (fastest)
        Higher values = higher precision = slower computation.
    band : tuple or None
        Broad band frequency range.
        Default is 1 to 30 Hz.
    interp : boolean
        If True, a cubic interpolation is performed to ensure that the output
        is the same size as the input (= pointwise power).
    norm : bool
        If True, return bandwise normalized band power, i.e. for each time
        point, the sum of power in all the frequency bins equals 1.

    Returns
    -------
    f : ndarray
        Frequency vector
    t : ndarray
        Time vector
    Sxx : ndarray
        Power in the specified frequency bins of shape (f, t)

    Notes
    -----
    2D Interpolation is done using `scipy.interpolate.RectBivariateSpline`
    which is much faster than `scipy.interpolate.interp2d` for a rectangular
    grid. The default is to use a bivariate spline with 3 degrees.
    """
    # Safety check
    data = np.asarray(data)
    assert step <= window

    step = 1 / sf if step == 0 else step

    # Define STFT parameters
    nperseg = int(window * sf)
    noverlap = int(nperseg - (step * sf))

    # Compute STFT and remove the last epoch
    f, t, Sxx = signal.stft(data, sf, nperseg=nperseg, noverlap=noverlap,
                            detrend=False, padded=True)

    # Let's keep only the frequency of interest
    if band is not None:
        idx_band = np.logical_and(f >= band[0], f <= band[1])
        f = f[idx_band]
        Sxx = Sxx[idx_band, :]

    # Compute power
    Sxx = np.square(np.abs(Sxx))

    # Interpolate
    if interp:
        func = RectBivariateSpline(f, t, Sxx)
        t = np.arange(data.size) / sf
        Sxx = func(f, t)

    if norm:
        sum_pow = Sxx.sum(0).reshape(1, -1)
        np.divide(Sxx, sum_pow, out=Sxx)
    return f, t, Sxx


def trimbothstd(x, cut=0.10):
    """
    Slices off a proportion of items from both ends of an array and then
    compute the sample standard deviation.

    Slices off the passed proportion of items from both ends of the passed
    array (i.e., with `cut` = 0.1, slices leftmost 10% **and**
    rightmost 10% of scores). The trimmed values are the lowest and
    highest ones.
    Slices off less if proportion results in a non-integer slice index (i.e.,
    conservatively slices off`proportiontocut`).

    Parameters
    ----------
    x : 1D np.array
        Input array.
    cut : float
        Proportion (in range 0-1) of total data to trim of each end.
        Default is 0.10, i.e. 10% lowest and 10% highest values are removed.

    Returns
    -------
    trimmed_std : float
        Sample standard deviation of the trimmed array.
    """
    x = np.asarray(x)
    n = x.size
    lowercut = int(cut * n)
    uppercut = n - lowercut
    atmp = np.partition(x, (lowercut, uppercut - 1))
    sl = slice(lowercut, uppercut)
    return atmp[sl].std(ddof=1)


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
    min_distance = min_distance_ms / 1000. * sf
    idx_diff = np.diff(index)
    condition = idx_diff > 1
    idx_distance = np.where(condition)[0]
    distance = idx_diff[condition]
    bad = idx_distance[np.where(distance < min_distance)[0]]
    # Fill gap between events separated with less than min_distance_ms
    if len(bad) > 0:
        fill = np.hstack([np.arange(index[j] + 1, index[j + 1])
                          for i, j in enumerate(bad)])
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
    index = np.array([])
    for k in range(x.shape[0]):
        index = np.append(index, np.arange(x[k, 0], x[k, 1] + 1))
    return index.astype(int)


def get_bool_vector(data, sf, sp):
    """Return a Boolean vector given the original data and sf and
    a YASA's detection dataframe.

    Parameters
    ----------
    data : array_like
        Single-channel EEG data.
    sf : float
        Sampling frequency of the data.
    sp : pandas DataFrame
        YASA's detection dataframe returned by the spindles_detect function.

    Returns
    -------
    bool_vector : array
        Array of bool indicating for each sample in data if this sample is
        part of a spindle (True) or not (False).
    """
    data = np.asarray(data)
    assert isinstance(sp, pd.DataFrame)
    assert 'Start' in sp.keys()
    assert 'End' in sp.keys()
    bool_spindles = np.zeros(data.shape, dtype=int)

    # For multi-channel detection
    multi = False
    if 'Channel' in sp.keys():
        chan = sp['Channel'].unique()
        n_chan = chan.size
        if n_chan > 1:
            multi = True

    if multi:
        for c in chan:
            sp_chan = sp[sp['Channel'] == c]
            idx_sp = _index_to_events(sp_chan[['Start', 'End']].values * sf)
            bool_spindles[sp_chan['IdxChannel'].iloc[0], idx_sp] = 1
    else:
        idx_sp = _index_to_events(sp[['Start', 'End']].values * sf)
        bool_spindles[idx_sp] = 1
    return bool_spindles

#############################################################################
# MAIN FUNCTION
#############################################################################


def spindles_detect(data, sf, freq_sp=(11, 16), duration=(0.4, 2),
                    freq_broad=(1, 30), min_distance=500,
                    thresh={'rel_pow': 0.2, 'corr': 0.65, 'rms': 1.5},
                    remove_outliers=False):
    """Spindles detection.

    Parameters
    ----------
    data : array_like
        Single-channel data. Unit must be uV.
    sf : float
        Sampling frequency of the data in Hz.
    freq_sp : tuple or list
        Spindles frequency range. Default is 11 to 16 Hz.
    duration : tuple or list
        The minimum and maximum duration of the spindles.
        Default is 0.4 to 2 seconds.
    freq_broad : tuple or list
        Broad band frequency of interest.
        Default is 1 to 30 Hz.
    min_distance : int
        If two spindles are closer than `min_distance` (in ms), they are
        merged into a single spindles. Default is 500 ms.
    thresh : dict
        Detection thresholds::

            'rel_pow' : Relative power (= power ratio freq_sp / freq_broad).
            'corr' : Pearson correlation coefficient.
            'rms' : Mean(RMS) + 1.5 * STD(RMS).
    remove_outliers : boolean
        If True, YASA will automatically detect and remove outliers spindles
        using an Isolation Forest (implemented in the scikit-learn package).
        The outliers detection is performed on all the spindles
        parameters with the exception of the 'Start' and 'End' columns.
        YASA uses a random seed (42) to ensure reproducible results.
        Note that this step will only be applied if there are more than 50
        detected spindles in the first place. Default to False.

    Returns
    -------
    sp_params : pd.DataFrame
        Pandas DataFrame::

            'Start' : Start time of each detected spindles (in seconds)
            'End' : End time (in seconds)
            'Duration' : Duration (in seconds)
            'Amplitude' : Amplitude (in uV)
            'RMS' : Root-mean-square (in uV)
            'AbsPower' : Median absolute power (in log10 uV^2)
            'RelPower' : Median relative power (ranging from 0 to 1, in % uV^2)
            'Frequency' : Median frequency (in Hz)
            'Oscillations' : Number of oscillations (peaks)
            'Symmetry' : Symmetry index, ranging from 0 to 1

    Notes
    -----
    For better results, apply this detection only on artefact-free NREM sleep.
    """
    # Safety check
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 2:
        data = np.squeeze(data)
    assert data.ndim == 1, 'Wrong data dimension. Please pass 1D data.'
    assert freq_sp[0] < freq_sp[1]
    assert freq_broad[0] < freq_broad[1]

    # Check data amplitude
    data_trimstd = trimbothstd(data, cut=0.10)
    data_ptp = np.ptp(data)
    logger.info('Number of samples in data = %i', data.size)
    logger.info('Sampling frequency = %.2f Hz', sf)
    logger.info('Data duration = %.2f seconds', data.size / sf)
    logger.info('Trimmed standard deviation of data = %.4f uV', data_trimstd)
    logger.info('Peak-to-peak amplitude of data = %.4f uV', data_ptp)
    if not(1 < data_trimstd < 1e3 or 1 < data_ptp < 1e6):
        logger.error('Wrong data amplitude. Unit must be uV. Returning None.')
        return None

    if 'rel_pow' not in thresh.keys():
        thresh['rel_pow'] = 0.20
    if 'corr' not in thresh.keys():
        thresh['corr'] = 0.65
    if 'rms' not in thresh.keys():
        thresh['rms'] = 1.5

    # Downsample to 100 Hz
    if sf > 100:
        fac = 100 / sf
        logger.info('Downsampling by a factor of %.2f', 1 / fac)
        data = resample(data, up=fac, down=1.0, npad='auto', axis=-1,
                        window='boxcar', n_jobs=1, pad='reflect_limited',
                        verbose=False)
        sf = 100

    # Bandpass filter
    data = filter_data(data, sf, freq_broad[0], freq_broad[1], method='fir',
                       verbose=0)

    # If freq_sp is not too narrow, we add and remove 1 Hz to adjust the
    # FIR filter frequency response. The width of the transition band is set
    # to 1.5 Hz on each side, meaning that for freq_sp = (11, 16 Hz), the -6 dB
    # point is located at 11.25 and 15.75 Hz.
    trans = 1 if freq_sp[1] - freq_sp[0] > 3 else 0
    data_sigma = filter_data(data, sf, freq_sp[0] + trans, freq_sp[1] - trans,
                             l_trans_bandwidth=1.5, h_trans_bandwidth=1.5,
                             method='fir', verbose=0)

    # Compute the pointwise relative power using interpolated STFT
    # Here we use a step of 200 ms to speed up the computation.
    f, t, Sxx = stft_power(data, sf, window=2, step=.2, band=freq_broad,
                           interp=False, norm=True)
    idx_sigma = np.logical_and(f >= freq_sp[0], f <= freq_sp[1])
    rel_pow = Sxx[idx_sigma].sum(0)

    # Let's interpolate `rel_pow` to get one value per sample
    # Note that we could also have use the `interp=True` in the `stft_power`
    # function, however 2D interpolation is much slower than
    # 1D interpolation.
    func = interp1d(t, rel_pow, kind='cubic', bounds_error=False,
                    fill_value=0)
    t = np.arange(data.size) / sf
    rel_pow = func(t)

    # Now we apply moving RMS and correlation on the sigma-filtered signal
    _, mcorr = moving_transform(data_sigma, data, sf, window=.3, step=.1,
                                method='corr', interp=True)
    _, mrms = moving_transform(data_sigma, data, sf, window=.3, step=.1,
                               method='rms', interp=True)

    # Hilbert power (to define the instantaneous frequency / power)
    n = data_sigma.size
    nfast = next_fast_len(n)
    analytic = signal.hilbert(data_sigma, N=nfast)[:n]
    inst_phase = np.angle(analytic)
    inst_pow = np.square(np.abs(analytic))
    # inst_freq = sf / 2pi * 1st-derivative of the phase of the analytic signal
    inst_freq = (sf / (2 * np.pi) * np.ediff1d(inst_phase))

    # Let's define the thresholds
    thresh_rms = mrms.mean() + thresh['rms'] * trimbothstd(mrms, cut=0.10)
    # Avoid too high threshold caused by Artefacts / Motion during Wake.
    thresh_rms = min(thresh_rms, 10)
    idx_rel_pow = (rel_pow >= thresh['rel_pow']).astype(int)
    idx_mcorr = (mcorr >= thresh['corr']).astype(int)
    idx_mrms = (mrms >= thresh_rms).astype(int)
    idx_sum = (idx_rel_pow + idx_mcorr + idx_mrms).astype(int)

    # For debugging
    logger.info('Moving RMS threshold = %.3f', thresh_rms)
    logger.info('Number of supra-theshold samples for relative power = %i',
                idx_rel_pow.sum())
    logger.info('Number of supra-theshold samples for moving correlation = %i',
                idx_mcorr.sum())
    logger.info('Number of supra-theshold samples for moving RMS = %i',
                idx_mrms.sum())

    # The detection using the three thresholds tends to underestimate the
    # real duration of the spindle. To overcome this, we compute a soft
    # threshold by smoothing the idx_sum vector with a 100 ms window.
    w = int(0.1 * sf)
    idx_sum = np.convolve(idx_sum, np.ones(w) / w, mode='same')
    # And we then find indices that are strictly greater than 2, i.e. we find
    # the 'true' beginning and 'true' end of the events by finding where at
    # least two out of the three treshold were crossed.
    where_sp = np.where(idx_sum > 2)[0]

    # If no events are found, return an empty dataframe
    if not len(where_sp):
        logger.warning('No spindles were found in data. Returning None.')
        return None

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

    # If no events of good duration are found, return an empty dataframe
    if all(~good_dur):
        logger.warning('No spindles were found in data. Returning None.')
        return None

    # Initialize empty variables
    n_sp = len(sp)
    sp_amp = np.zeros(n_sp)
    sp_freq = np.zeros(n_sp)
    sp_rms = np.zeros(n_sp)
    sp_osc = np.zeros(n_sp)
    sp_sym = np.zeros(n_sp)
    sp_abs = np.zeros(n_sp)
    sp_rel = np.zeros(n_sp)

    # Number of oscillations (= number of peaks separated by at least 60 ms)
    # --> 60 ms because 1000 ms / 16 Hz = 62.5 ms, in other words, at 16 Hz,
    # peaks are separated by 62.5 ms. At 11 Hz, peaks are separated by 90 ms.
    distance = 60 * sf / 1000

    for i in np.arange(len(sp))[good_dur]:
        # Important: detrend the signal to avoid wrong peak-to-peak amplitude
        sp_det = signal.detrend(data[sp[i]], type='linear')
        sp_amp[i] = np.ptp(sp_det)  # Peak-to-peak amplitude
        sp_rms[i] = _rms(sp_det)  # Root mean square
        sp_rel[i] = np.median(rel_pow[sp[i]])  # Mean relative power

        # Hilbert-based instantaneous properties
        sp_inst_freq = inst_freq[sp[i]]
        sp_inst_pow = inst_pow[sp[i]]
        sp_abs[i] = np.median(np.log10(sp_inst_pow[sp_inst_pow > 0]))
        sp_freq[i] = np.median(sp_inst_freq[sp_inst_freq > 0])

        # Number of oscillations
        peaks, peaks_params = signal.find_peaks(sp_det,
                                                distance=distance,
                                                prominence=(None, None))
        sp_osc[i] = len(peaks)

        # For frequency and amplitude, we can also optionally use these
        # faster alternatives. If we use them, we do not need to compute the
        # Hilbert transform of the filtered signal.
        # sp_freq[i] = sf / np.mean(np.diff(peaks))
        # sp_amp[i] = peaks_params['prominences'].max()

        # Symmetry index
        sp_sym[i] = peaks[peaks_params['prominences'].argmax()] / sp_det.size

    # Create a dictionnary
    sp_params = {'Start': sp_start,
                 'End': sp_end,
                 'Duration': sp_dur,
                 'Amplitude': sp_amp,
                 'RMS': sp_rms,
                 'AbsPower': sp_abs,
                 'RelPower': sp_rel,
                 'Frequency': sp_freq,
                 'Oscillations': sp_osc,
                 'Symmetry': sp_sym}

    df_sp = pd.DataFrame.from_dict(sp_params)[good_dur].reset_index(drop=True)

    # We need at least 50 detected spindles to apply the Isolation Forest.
    if remove_outliers and df_sp.shape[0] >= 50:
        from sklearn.ensemble import IsolationForest
        ilf = IsolationForest(behaviour='new', contamination='auto',
                              max_samples='auto', verbose=0, random_state=42)

        good = ilf.fit_predict(df_sp[df_sp.columns.difference(['Start',
                                                               'End'])])
        good[good == -1] = 0
        logger.info('%i outliers were removed.', (good == 0).sum())
        # Remove outliers from DataFrame
        df_sp = df_sp[good.astype(bool)].reset_index(drop=True)

    logger.info('%i spindles were found in data.', df_sp.shape[0])
    return df_sp


def spindles_detect_multi(data, sf, ch_names, multi_only=False, **kwargs):
    """Multi-channel spindles detection.

    Parameters
    ----------
    data : array_like
        Multi-channel data. Unit must be uV and shape (n_chan, n_samples).
        If you used MNE to load the data, you should pass `raw._data * 1e6`.
    sf : float
        Sampling frequency of the data in Hz.
        If you used MNE to load the data, you should pass `raw.info['sfreq']`.
    ch_names : list of str
        Channel names.
        If you used MNE to load the data, you should pass `raw.ch_names`.
    multi_only : boolean
        Define the behavior of the multi-channel detection. If True, only
        spindles that are present on at least two channels are kept. If False,
        no selection is applied and the output is just a concatenation of the
        single-channel detection dataframe. Default is False.
    **kwargs
        Keywords arguments that are passed to the `spindles_detect` function.

    Returns
    -------
    sp_params : pd.DataFrame
        Pandas DataFrame::

            'Start' : Start time of each detected spindles (in seconds)
            'End' : End time (in seconds)
            'Duration' : Duration (in seconds)
            'Amplitude' : Amplitude (in uV)
            'RMS' : Root-mean-square (in uV)
            'AbsPower' : Median absolute power (in log10 uV^2)
            'RelPower' : Median relative power (ranging from 0 to 1, in % uV^2)
            'Frequency' : Median frequency (in Hz)
            'Oscillations' : Number of oscillations (peaks)
            'Symmetry' : Symmetry index, ranging from 0 to 1
            'Channel' : Channel name
            'IdxChannel' : Integer index of channel in data
    """
    # Safety check
    data = np.asarray(data, dtype=np.float64)
    assert data.ndim == 2
    assert data.shape[0] < data.shape[1]
    n_chan = data.shape[0]
    assert isinstance(ch_names, (list, np.ndarray))
    if len(ch_names) != n_chan:
        raise AssertionError('ch_names must have same length as data.shape[0]')

    # Single channel detection
    df = pd.DataFrame()
    for i in range(n_chan):
        df_chan = spindles_detect(data[i, :], sf, **kwargs)
        if df_chan is not None:
            df_chan['Channel'] = ch_names[i]
            df_chan['IdxChannel'] = i
            df = df.append(df_chan, ignore_index=True)
        else:
            logger.warning('No spindles were found in channel %s.',
                           ch_names[i])

    # If no spindles were detected, return None
    if df.empty:
        logger.warning('No spindles were found in data. Returning None.')
        return None

    # Find spindles that are present on at least two channels
    if multi_only and df['Channel'].unique().size > 1:
        # We round to the nearest second
        idx_good = np.logical_or(df['Start'].round(0).duplicated(keep=False),
                                 df['End'].round(0).duplicated(keep=False)
                                 ).to_list()
        return df[idx_good].reset_index(drop=True)
    else:
        return df
