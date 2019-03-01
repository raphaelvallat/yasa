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
from mne.filter import filter_data
from scipy.fftpack import next_fast_len
from scipy.interpolate import interp1d, RectBivariateSpline

logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

logger = logging.getLogger('yasa')

__all__ = ['spindles_detect', 'spindles_detect_multi', 'stft_power',
           'moving_transform', 'get_bool_vector', 'sw_detect']


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
# MAIN FUNCTIONS
#############################################################################


def spindles_detect(data, sf, hypno=None, freq_sp=(12, 15), duration=(0.5, 2),
                    freq_broad=(1, 30), min_distance=500, downsample=True,
                    thresh={'rel_pow': 0.2, 'corr': 0.65, 'rms': 1.5},
                    remove_outliers=False):
    """Spindles detection.

    Parameters
    ----------
    data : array_like
        Single-channel continuous EEG data. Unit must be uV.
    sf : float
        Sampling frequency of the data in Hz.
    hypno : array_like
        Sleep stage vector (hypnogram). If the hypnogram is loaded, the
        detection will only be applied to NREM sleep epochs (stage 1, 2 and 3),
        therefore slightly improving the accuracy.
        hypno MUST be a 1D array of integers with the same size as data and
        where -1 = Artefact, 0 = Wake, 1 = N1, 2 = N2, 3 = N3, 4 = REM.
        If you need help loading your hypnogram vector, please read the
        Visbrain documentation at http://visbrain.org/sleep.
    freq_sp : tuple or list
        Spindles frequency range. Default is 12 to 15 Hz. Please note that YASA
        uses a FIR filter (implemented in MNE) with a 1.5Hz transition band,
        which means that for `freq_sp = (12, 15 Hz)`, the -6 dB points are
        located at 11.25 and 15.75 Hz.
    duration : tuple or list
        The minimum and maximum duration of the spindles.
        Default is 0.5 to 2 seconds.
    freq_broad : tuple or list
        Broad band frequency of interest.
        Default is 1 to 30 Hz.
    min_distance : int
        If two spindles are closer than `min_distance` (in ms), they are
        merged into a single spindles. Default is 500 ms.
    downsample : boolean
        If True, the data will be downsampled to 100 Hz or 128 Hz (depending
        on whether the original sampling frequency is a multiple of 100 or 128,
        respectively).
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
            'Stage' : Sleep stage (only if hypno was provided)

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
    assert isinstance(downsample, bool), 'Downsample must be True or False.'

    # Hypno processing
    if hypno is not None:
        hypno = np.asarray(hypno, dtype=int)
        assert hypno.ndim == 1, 'Hypno must be one dimensional.'
        assert hypno.size == data.size, 'Hypno must have same size as data.'
        unique_hypno = np.unique(hypno)
        logger.info('Number of unique values in hypno = %i', unique_hypno.size)
        if not any(np.in1d(unique_hypno, [1, 2, 3])):
            logger.error('No NREM sleep in hypno. Switching to hypno = None')
            hypno = None
        else:
            idx_nrem = np.logical_and(hypno >= 1, hypno < 4)

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

    # Check if we can downsample to 100 or 128 Hz
    if downsample is True and sf > 128:
        if sf % 100 == 0 or sf % 128 == 0:
            new_sf = 100 if sf % 100 == 0 else 128
            fac = int(sf / new_sf)
            sf = new_sf
            data = data[::fac]
            logger.info('Downsampled data by a factor of %i', fac)
            if hypno is not None:
                hypno = hypno[::fac]
                assert hypno.size == data.size
                idx_nrem = np.logical_and(hypno >= 1, hypno < 4)
                logger.info('Seconds of NREM sleep = %.2f',
                            idx_nrem.sum() / sf)
        else:
            logger.warning("Cannot downsample if sf is not a mutiple of 100 "
                           "or 128. Skipping downsampling.")

    # Bandpass filter
    data = filter_data(data, sf, freq_broad[0], freq_broad[1], method='fir',
                       verbose=0)

    # The width of the transition band is set to 1.5 Hz on each side,
    # meaning that for freq_sp = (12, 15 Hz), the -6 dB points are located at
    # 11.25 and 15.75 Hz.
    data_sigma = filter_data(data, sf, freq_sp[0], freq_sp[1],
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
    if hypno is None:
        thresh_rms = mrms.mean() + thresh['rms'] * trimbothstd(mrms, cut=0.10)
    else:
        thresh_rms = mrms[idx_nrem].mean() + thresh['rms'] * \
            trimbothstd(mrms[idx_nrem], cut=0.10)
    # Avoid too high threshold caused by Artefacts / Motion during Wake.
    thresh_rms = min(thresh_rms, 10)
    idx_rel_pow = (rel_pow >= thresh['rel_pow']).astype(int)
    idx_mcorr = (mcorr >= thresh['corr']).astype(int)
    idx_mrms = (mrms >= thresh_rms).astype(int)
    idx_sum = (idx_rel_pow + idx_mcorr + idx_mrms).astype(int)

    # Make sure that we do not detect spindles in REM or Wake if hypno != None
    if hypno is not None:
        idx_sum[~idx_nrem] = 0

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
    sp_sta = np.zeros(n_sp)

    # Number of oscillations (= number of peaks separated by at least 60 ms)
    # --> 60 ms because 1000 ms / 16 Hz = 62.5 ms, in other words, at 16 Hz,
    # peaks are separated by 62.5 ms. At 11 Hz, peaks are separated by 90 ms.
    distance = 60 * sf / 1000

    for i in np.arange(len(sp))[good_dur]:
        # Important: detrend the signal to avoid wrong peak-to-peak amplitude
        sp_det = signal.detrend(data[sp[i]], type='linear')
        sp_amp[i] = np.ptp(sp_det)  # Peak-to-peak amplitude
        sp_rms[i] = _rms(sp_det)  # Root mean square
        sp_rel[i] = np.median(rel_pow[sp[i]])  # Median relative power

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

        # Sleep stage
        if hypno is not None:
            sp_sta[i] = hypno[sp[i]][0]

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
                 'Symmetry': sp_sym,
                 'Stage': sp_sta}

    df_sp = pd.DataFrame.from_dict(sp_params)[good_dur].reset_index(drop=True)

    if hypno is None:
        df_sp = df_sp.drop(columns=['Stage'])
    else:
        df_sp['Stage'] = df_sp['Stage'].astype(int).astype('category')

    # We need at least 50 detected spindles to apply the Isolation Forest.
    if remove_outliers and df_sp.shape[0] >= 50:
        from sklearn.ensemble import IsolationForest
        df_sp_dummies = pd.get_dummies(df_sp)
        col_keep = df_sp_dummies.columns.difference(['Start', 'End'])
        ilf = IsolationForest(behaviour='new', contamination='auto',
                              max_samples='auto', verbose=0, random_state=42)

        good = ilf.fit_predict(df_sp_dummies[col_keep])
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
            'Stage' : Sleep stage (only if hypno was provided)
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


def sw_detect(data, sf, hypno=None, freq_sw=(0.3, 3.5), dur_neg=(0.3, 1.5),
              dur_pos=(0.1, 1), amp_neg=(40, 300), amp_pos=(10, 150),
              amp_ptp=(75, 400), downsample=True, remove_outliers=False):
    """Slow-waves detection.

    Parameters
    ----------
    data : array_like
        Single-channel continuous EEG data. Unit must be uV.
    sf : float
        Sampling frequency of the data in Hz.
    hypno : array_like
        Sleep stage vector (hypnogram). If the hypnogram is loaded, the
        detection will only be applied to N2 and N3 sleep epochs.
        ``hypno`` MUST be a 1D array of integers with the same size as data
        and where -1 = Artefact, 0 = Wake, 1 = N1, 2 = N2, 3 = N3, 4 = REM.
        If you need help loading your hypnogram vector, please read the
        Visbrain documentation at http://visbrain.org/sleep.
    freq_sw : tuple or list
        Slow wave frequency range. Default is 0.3 to 3.5 Hz. Please note that
        YASA uses a FIR filter (implemented in MNE) with a 0.2Hz transition
        band, which means that for `freq_sw = (.3, 3.5 Hz)`, the -6 dB points
        are located at 0.2 and 3.6 Hz.
    dur_neg : tuple or list
        The minimum and maximum duration of the negative deflection of the
        slow wave. Default is 0.3 to 1.5 second.
    dur_pos : tuple or list
        The minimum and maximum duration of the positive deflection of the
        slow wave. Default is 0.1 to 1 second.
    amp_neg : tuple or list
        Absolute minimum and maximum negative trough amplitude of the
        slow-wave. Default is 40 uV to 300 uV.
    amp_pos : tuple or list
        Absolute minimum and maximum positive peak amplitude of the
        slow-wave. Default is 10 uV to 150 uV.
    amp_ptp : tuple or list
        Minimum and maximum peak-to-peak amplitude of the slow-wave.
        Default is 75 uV to 400 uV.
    downsample : boolean
        If True, the data will be downsampled to 100 Hz or 128 Hz (depending
        on whether the original sampling frequency is a multiple of 100 or 128,
        respectively).
    remove_outliers : boolean
        If True, YASA will automatically detect and remove outliers slow-waves
        using an Isolation Forest (implemented in the scikit-learn package).
        The outliers detection is performed on the frequency, amplitude and
        duration parameters of the detected slow-waves. YASA uses a random seed
        (42) to ensure reproducible results. Note that this step will only be
        applied if there are more than 100 detected slow-waves in the first
        place. Default to False.

    Returns
    -------
    sw_params : pd.DataFrame
        Pandas DataFrame::

            'Start' : Start of each detected slow-wave (in seconds of data)
            'NegPeak' : Location of the negative peak (in seconds of data)
            'MidCrossing' : Location of the negative-to-positive zero-crossing
            'Pospeak' : Location of the positive peak
            'End' : End time (in seconds)
            'Duration' : Duration (in seconds)
            'ValNegPeak' : Amplitude of the negative peak (in uV - filtered)
            'ValPosPeak' : Amplitude of the positive peak (in uV - filtered)
            'PTP' : Peak to peak amplitude (ValPosPeak - ValNegPeak)
            'Slope' : Slope between ``NegPeak`` and ``MidCrossing`` (in uV/sec)
            'Frequency' : Frequency of the slow-wave (1 / ``Duration``)
            'Stage' : Sleep stage (only if hypno was provided)

    Notes
    -----
    For better results, apply this detection only on artefact-free NREM sleep.

    Note that the ``PTP``, ``Slope``, ``ValNegPeak`` and ``ValPosPeak`` are
    computed on the filtered signal.
    """
    # Safety check
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 2:
        data = np.squeeze(data)
    assert data.ndim == 1, 'Wrong data dimension. Please pass 1D data.'
    assert freq_sw[0] < freq_sw[1]
    assert amp_ptp[0] < amp_ptp[1]
    assert isinstance(downsample, bool), 'Downsample must be True or False.'

    # Hypno processing
    if hypno is not None:
        hypno = np.asarray(hypno, dtype=int)
        assert hypno.ndim == 1, 'Hypno must be one dimensional.'
        assert hypno.size == data.size, 'Hypno must have same size as data.'
        unique_hypno = np.unique(hypno)
        logger.info('Number of unique values in hypno = %i', unique_hypno.size)
        if not any(np.in1d(unique_hypno, [2, 3])):
            logger.error('No N2/3 sleep in hypno. Switching to hypno = None')
            hypno = None
        else:
            idx_nrem = np.logical_and(hypno >= 2, hypno < 4)

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

    # Check if we can downsample to 100 or 128 Hz
    if downsample is True and sf > 128:
        if sf % 100 == 0 or sf % 128 == 0:
            new_sf = 100 if sf % 100 == 0 else 128
            fac = int(sf / new_sf)
            sf = new_sf
            data = data[::fac]
            logger.info('Downsampled data by a factor of %i', fac)
            if hypno is not None:
                hypno = hypno[::fac]
                assert hypno.size == data.size
                idx_nrem = np.logical_and(hypno >= 2, hypno < 4)
                logger.info('Seconds of NREM sleep = %.2f',
                            idx_nrem.sum() / sf)
        else:
            logger.warning("Cannot downsample if sf is not a mutiple of 100 "
                           "or 128. Skipping downsampling.")

    # Define time vector
    times = np.arange(data.size) / sf

    # Bandpass filter
    data_filt = filter_data(data, sf, freq_sw[0], freq_sw[1], method='fir',
                            verbose=0, l_trans_bandwidth=0.2,
                            h_trans_bandwidth=0.2)

    # Find peaks in data
    # Negative peaks with value comprised between -40 to -300 uV
    idx_neg_peaks, _ = signal.find_peaks(-1 * data_filt, height=amp_neg)

    # Positive peaks with values comprised between 10 to 150 uV
    idx_pos_peaks, _ = signal.find_peaks(data_filt, height=amp_pos)

    # If no peaks are detected, return None
    if len(idx_neg_peaks) == 0 or len(idx_pos_peaks) == 0:
        logger.warning('No peaks were found in data. Returning None.')
        return None

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
    sw_ptp = np.abs(data_filt[idx_neg_peaks]) + data_filt[idx_pos_peaks]
    good_ptp = np.logical_and(sw_ptp > amp_ptp[0], sw_ptp < amp_ptp[1])

    # If good_ptp is all False
    if all(~good_ptp):
        logger.warning('No slow-wave with good amplitude. Returning None.')
        return None

    sw_ptp = sw_ptp[good_ptp]
    idx_neg_peaks = idx_neg_peaks[good_ptp]
    idx_pos_peaks = idx_pos_peaks[good_ptp]

    # Now we need to check the negative and positive phase duration
    # For that we need to compute the zero crossings of the filtered signal
    zero_crossings = _zerocrossings(data_filt)
    # Make sure that there is a zero-crossing after the last detected peak
    if zero_crossings[-1] < max(idx_pos_peaks[-1], idx_neg_peaks[-1]):
        # If not, append the index of the last peak
        zero_crossings = np.append(zero_crossings,
                                   max(idx_pos_peaks[-1], idx_neg_peaks[-1]))

    # Find distance to previous and following zc
    neg_sorted = np.searchsorted(zero_crossings, idx_neg_peaks)
    previous_neg_zc = zero_crossings[neg_sorted - 1] - idx_neg_peaks
    following_neg_zc = zero_crossings[neg_sorted] - idx_neg_peaks
    neg_phase_dur = (np.abs(previous_neg_zc) + following_neg_zc) / sf

    # Distance (in samples) between the positive peaks and the previous and
    # following zero-crossings
    pos_sorted = np.searchsorted(zero_crossings, idx_pos_peaks)
    previous_pos_zc = zero_crossings[pos_sorted - 1] - idx_pos_peaks
    following_pos_zc = zero_crossings[pos_sorted] - idx_pos_peaks
    pos_phase_dur = (np.abs(previous_pos_zc) + following_pos_zc) / sf

    # We now compute a set of metrics
    sw_start = times[idx_neg_peaks + previous_neg_zc]  # Start in time vector
    sw_end = times[idx_pos_peaks + following_pos_zc]  # End in time vector
    sw_dur = sw_end - sw_start  # Same as pos_phase_dur + neg_phase_dur
    sw_midcrossing = times[idx_neg_peaks + following_neg_zc]  # Neg-to-pos zc
    sw_idx_neg = times[idx_neg_peaks]  # Location of negative peak
    sw_idx_pos = times[idx_pos_peaks]  # Location of positive peak
    # Slope between peak trough and midcrossing
    sw_slope = sw_ptp / (sw_midcrossing - sw_idx_neg)
    # Hypnogram
    if hypno is not None:
        sw_sta = hypno[idx_neg_peaks + previous_neg_zc]
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
        logger.warning('No slow-wave satisfying all criteria. Returning None.')
        return None

    # Create a dictionnary and then a dataframe (much faster)
    sw_params = {'Start': sw_start,
                 'NegPeak': sw_idx_neg,
                 'MidCrossing': sw_midcrossing,
                 'PosPeak': sw_idx_pos,
                 'End': sw_end,
                 'Duration': sw_dur,
                 'ValNegPeak': data_filt[idx_neg_peaks],
                 'ValPosPeak': data_filt[idx_pos_peaks],
                 'PTP': sw_ptp,
                 'Slope': sw_slope,
                 'Frequency': 1 / sw_dur,
                 'Stage': sw_sta,
                 }

    df_sw = pd.DataFrame.from_dict(sw_params)[good_sw]

    # Remove all duplicates
    df_sw = df_sw.drop_duplicates(subset=['Start'], keep=False)
    df_sw = df_sw.drop_duplicates(subset=['End'], keep=False)

    if hypno is None:
        df_sw = df_sw.drop(columns=['Stage'])
    else:
        df_sw['Stage'] = df_sw['Stage'].astype(int).astype('category')
        # Keep only N2 and N3
        df_sw = df_sw[df_sw['Stage'].isin([2, 3])]

    # We need at least 100 detected slow waves to apply the Isolation Forest.
    if remove_outliers and df_sw.shape[0] >= 100:
        from sklearn.ensemble import IsolationForest
        col_keep = ['Duration', 'ValNegPeak', 'ValPosPeak', 'PTP', 'Slope',
                    'Frequency']
        ilf = IsolationForest(behaviour='new', contamination='auto',
                              max_samples='auto', verbose=0, random_state=42)

        good = ilf.fit_predict(df_sw[col_keep])
        good[good == -1] = 0
        logger.info('%i outliers were removed.', (good == 0).sum())
        # Remove outliers from DataFrame
        df_sw = df_sw[good.astype(bool)]

    logger.info('%i spindles were found in data.', df_sw.shape[0])
    return df_sw.reset_index(drop=True)
