"""
This file contains several helper functions to calculate spectral power from
1D and 2D EEG data.
"""
import mne
import numpy as np
import pandas as pd
from scipy import signal
from scipy.integrate import simps
from scipy.interpolate import RectBivariateSpline


__all__ = ['bandpower', 'bandpower_from_psd', 'stft_power']


def bandpower(data, sf=None, ch_names=None, hypno=None, include=(2, 3),
              win_sec=4, relative=True,
              bands=[(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
                     (12, 30, 'Beta'), (30, 40, 'Gamma')],
              kwargs_welch=dict(average='median', window='hamming')):
    """
    Calculate the Welch bandpower for each channel and, if specified,
    for each sleep stage.

    .. versionadded:: 0.1.6

    Parameters
    ----------
    data : np.array_like or mne.io.Raw
        1D or 2D EEG data. Can also be a MNE Raw object, in which case data,
        sf, and ch_names will be automatically extracted. Data will also
        be internally converted from Volts (MNE default) to micro-Volts (YASA).
    sf : float
        The sampling frequency of data AND the hypnogram.
        Can be omitted if ``data`` is a MNE Raw object.
    ch_names : list
        List of channel names, e.g. ['Cz', 'F3', 'F4', ...]. If None,
        channels will be labelled ['CHAN001', 'CHAN002', ...].
        Can be omitted if ``data`` is a MNE Raw object.
    hypno : array_like
        The sleep staging (hypnogram) 1D array. Must have the same
        sampling frequency and number of samples as data.
    relative : boolean
        If True, bandpower is divided by the total power between the min and
        max frequencies defined in ``band``.
    bands : list of tuples
        List of frequency bands of interests. Each tuple must contain the
        lower and upper frequencies, as well as the band name
        (e.g. (0.5, 4, 'Delta')).
    kwargs_welch : dict
        Optional keywords arguments that are passed to the
        ``scipy.signal.welch`` function.

    Returns
    -------
    bandpowers : pandas.DataFrame
        Bandpower dataframe, in which each row is a channel and each column
        a spectral band.
    """
    # Check if input data is a MNE Raw object
    if isinstance(data, mne.io.BaseRaw):
        sf = data.info['sfreq']  # Extract sampling frequency
        ch_names = data.ch_names  # Extract channel names
        data = data.get_data() * 1e6  # Convert from V to uV
        _, npts = data.shape
    else:
        # Safety checks
        assert isinstance(data, np.ndarray), 'Data must be a numpy array.'
        data = np.atleast_2d(data)
        assert data.ndim == 2, 'Data must be of shape (nchan, n_samples).'
        nchan, npts = data.shape
        assert nchan < npts, 'Data must be of shape (nchan, n_samples).'
        assert sf is not None, 'sf must be specified if passing a numpy array.'
        assert isinstance(sf, (int, float))
        if ch_names is None:
            ch_names = ['CHAN' + str(i + 1).zfill(3) for i in range(nchan)]
        else:
            ch_names = np.atleast_1d(np.asarray(ch_names, dtype=str))
            assert ch_names.ndim == 1, 'ch_names must be 1D.'
            assert len(ch_names) == nchan, 'ch_names must match data.shape[0].'

    win = int(win_sec * sf)  # nperseg

    if hypno is None:
        # Calculate the PSD over the whole data
        freqs, psd = signal.welch(data, sf, nperseg=win, **kwargs_welch)
        return bandpower_from_psd(psd, freqs, ch_names,
                                  bands=bands,
                                  relative=relative).set_index('Chan')
    else:
        # Per each sleep stage defined in ``include``.
        hypno = np.asarray(hypno)
        assert include is not None, 'include cannot be None if hypno is given'
        include = np.atleast_1d(np.asarray(include))
        assert hypno.ndim == 1, 'Hypno must be a 1D array.'
        assert hypno.size == npts, 'Hypno must have same size as data.shape[1]'
        assert include.size >= 1, '`include` must have at least one element.'
        assert hypno.dtype.kind == include.dtype.kind, ('hypno and include '
                                                        'must have same dtype')
        assert np.in1d(hypno, include).any(), ('None of the stages '
                                               'specified in `include` '
                                               'are present in hypno.')
        # Initialize empty dataframe and loop over stages
        df_bp = pd.DataFrame([])
        for stage in include:
            if stage not in hypno:
                continue
            data_stage = data[:, hypno == stage]
            freqs, psd = signal.welch(data_stage, sf, nperseg=win,
                                      **kwargs_welch)
            bp_stage = bandpower_from_psd(psd, freqs, ch_names, bands=bands,
                                          relative=relative)
            bp_stage['Stage'] = stage
            df_bp = df_bp.append(bp_stage)
        return df_bp.set_index(['Stage', 'Chan'])


def bandpower_from_psd(psd, freqs, ch_names=None, bands=[(0.5, 4, 'Delta'),
                       (4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 30, 'Beta'),
                       (30, 40, 'Gamma')], relative=True):
    """Compute the average power of the EEG in specified frequency band(s)
    given a pre-computed PSD.

    .. versionadded:: 0.1.5

    Parameters
    ----------
    psd : array_like
        Power spectral density of data, in uV^2/Hz.
        Must be of shape (n_channels, n_freqs).
        See ``scipy.signal.welch`` for more details.
    freqs : array_like
        Array of sample frequencies.
    ch_names : list
        List of channel names, e.g. ['Cz', 'F3', 'F4', ...]. If None,
        channels will be labelled ['CHAN001', 'CHAN002', ...].
    bands : list of tuples
        List of frequency bands of interests. Each tuple must contain the
        lower and upper frequencies, as well as the band name
        (e.g. (0.5, 4, 'Delta')).
    relative : boolean
        If True, bandpower is divided by the total power between the min and
        max frequencies defined in ``band`` (default 0.5 to 40 Hz).

    Returns
    -------
    bandpowers : pandas.DataFrame
        Bandpower dataframe, in which each row is a channel and each column
        a spectral band.
    """
    # Safety checks
    freqs = np.asarray(freqs)
    assert freqs.ndim == 1
    psd = np.atleast_2d(psd)
    assert psd.ndim == 2, 'PSD must be of shape (n_channels, n_freqs).'
    all_freqs = np.hstack([[b[0], b[1]] for b in bands])
    fmin, fmax = min(all_freqs), max(all_freqs)
    idx_good_freq = np.logical_and(freqs >= fmin, freqs <= fmax)
    freqs = freqs[idx_good_freq]
    res = freqs[1] - freqs[0]
    nchan = psd.shape[0]
    assert nchan < psd.shape[1], 'PSD must be of shape (n_channels, n_freqs).'
    if ch_names is not None:
        ch_names = np.atleast_1d(np.asarray(ch_names, dtype=str))
        assert ch_names.ndim == 1, 'ch_names must be 1D.'
        assert len(ch_names) == nchan, 'ch_names must match psd.shape[0].'
    else:
        ch_names = ['CHAN' + str(i + 1).zfill(3) for i in range(nchan)]
    bp = np.zeros((nchan, len(bands)), dtype=np.float)
    psd = psd[:, idx_good_freq]
    total_power = simps(psd, dx=res)
    total_power = total_power[..., np.newaxis]

    # Enumerate over the frequency bands
    labels = []
    for i, band in enumerate(bands):
        b0, b1, la = band
        labels.append(la)
        idx_band = np.logical_and(freqs >= b0, freqs <= b1)
        if psd.ndim == 2:
            bp[:, i] = simps(psd[:, idx_band], dx=res)
        else:
            bp[i] = simps(psd[idx_band], dx=res)

    if relative:
        bp /= total_power

    # Convert to DataFrame
    bp = pd.DataFrame(bp, columns=labels)
    bp['FreqRes'] = res
    # bp['WindowSec'] = 1 / res
    bp['Relative'] = relative
    bp['Chan'] = ch_names
    bp = bp.set_index('Chan').reset_index()
    # Add hidden attributes
    bp.bands_ = str(bands)
    return bp


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
