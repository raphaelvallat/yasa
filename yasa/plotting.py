"""
This file contains several helper functions to calculate spectral power from
1D and 2D EEG data.
"""
import numpy as np
import pandas as pd
from lspopt import spectrogram_lspopt
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Set default font size to 12
plt.rcParams.update({'font.size': 12})


__all__ = ['plot_spectrogram']


def plot_spectrogram(data, sf, hypno=None, include=(2, 3), win_sec=30,
                     fmin=0.5, fmax=25, trimperc=2.5, cmap='RdBu_r'):
    """
    Plot a full-night multi-taper spectrogram,
    optionally with the hypnogram on top.

    .. versionadded:: 0.1.8

    Parameters
    ----------
    data : :py:class:`numpy.ndarray`
        1D EEG data.
    sf : float
        The sampling frequency of data AND the hypnogram.
    hypno : array_like
        Sleep stage vector (hypnogram).

        The hypnogram must have the exact same number of samples as ``data``.
        To upsample your hypnogram, please refer to
        :py:func:`yasa.hypno_upsample_to_data`.

        .. note::
            The default hypnogram format in YASA is a 1D integer
            vector where:

            - 0 = Wake
            - 1 = N1 sleep
            - 2 = N2 sleep
            - 3 = N3 sleep
            - 4 = REM
    win_sec : int or float
        The length of the sliding window, in seconds, used for multitaper PSD
        calculation. Default is 30 seconds.
    fmin, fmax : int or float
        The lower and upper frequency of the spectrogram. Default 0.5 to 25 Hz.
    trimperc : int or float
        The amount of data to trim on both ends of the distribution when
        normalizing the colormap. This parameter directly impacts the
        contrast of the spectrogram plot (higher values = higher contrast).
        Default is 2.5, meaning that the min and max of the colormap
        are defined as the 2.5 and 97.5 percentiles of the spectrogram.
    cmap : str
        Colormap. Default to 'RdBu_r'.


    Returns
    -------
    fig : :py:class:`matplotlib.pyplot.Figure`
        Matplotlib Figure
    """
    # Safety checks
    assert isinstance(data, np.ndarray), 'Data must be a 1D NumPy array.'
    assert isinstance(sf, (int, float)), 'sf must be int or float.'
    assert data.ndim == 1, 'Data must be a 1D (single-channel) NumPy array.'
    assert data.size > (60 * sf), 'At least 60 seconds of data is required.'
    assert isinstance(win_sec, (int, float)), 'win_sec must be int or float.'
    assert isinstance(fmin, (int, float)), 'fmin must be int or float.'
    assert isinstance(fmax, (int, float)), 'fmax must be int or float.'
    assert fmin < fmax, 'fmin must be strictly inferior to fmax.'
    assert fmax < sf / 2, 'fmax must be less than Nyquist (sf / 2).'

    # Calculate multi-taper spectrogram
    nperseg = int(win_sec * sf)
    f, t, Sxx = spectrogram_lspopt(data, sf, nperseg=nperseg, noverlap=0)
    Sxx = 10 * np.log10(Sxx)  # Convert to dB

    # Select only relevant frequencies (up to 30 Hz)
    good_freqs = np.logical_and(f >= fmin, f <= fmax)
    Sxx = Sxx[good_freqs, :]
    f = f[good_freqs]
    t /= 3600  # Convert t to hours

    # Normalization
    vmin, vmax = np.percentile(Sxx, [0 + trimperc, 100 - trimperc])
    norm = Normalize(vmin=vmin, vmax=vmax)

    if hypno is None:
        fig, ax = plt.subplots(nrows=1, figsize=(12, 4))
        im = ax.pcolormesh(t, f, Sxx, norm=norm, cmap=cmap, antialiased=True)
        ax.set_xlim(0, t.max())
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [sec]')

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.95, pad=0.05, fraction=0.1,
                            aspect=25)
        cbar.ax.set_ylabel('Log Power (dB / Hz)', rotation=270, labelpad=20)
        return fig
    else:
        hypno = np.asarray(hypno).astype(int)
        assert hypno.ndim == 1, 'Hypno must be 1D.'
        assert hypno.size == data.size, 'Hypno must have the same sf as data.'
        t_hyp = np.arange(hypno.size) / (sf * 3600)
        hypno = pd.Series(hypno).map({-1: -1, 0: 0, 1: 2,
                                      2: 3, 3: 4, 4: 1}).values
        hypno_rem = np.ma.masked_not_equal(hypno, 1)

        fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(12, 6),
                                       gridspec_kw={'height_ratios': [1, 2]})
        plt.subplots_adjust(hspace=0.1)

        # Hypnogram (top axis)
        ax0.step(t_hyp, -1 * hypno, color='k')
        ax0.step(t_hyp, -1 * hypno_rem, color='r')
        if -1 in hypno:
            ax0.set_yticks([1, 0, -1, -2, -3, -4])
            ax0.set_yticklabels(['Art', 'W', 'R', 'N1', 'N2', 'N3'])
            ax0.set_ylim(-4.5, 1.5)
        else:
            ax0.set_yticks([0, -1, -2, -3, -4])
            ax0.set_yticklabels(['W', 'R', 'N1', 'N2', 'N3'])
            ax0.set_ylim(-4.5, 0.5)
        ax0.set_xlim(0, t_hyp.max())
        ax0.set_ylabel('Stage')
        ax0.xaxis.set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.spines['top'].set_visible(False)

        # Spectrogram (bottom axis)
        im = ax1.pcolormesh(t, f, Sxx, norm=norm, cmap=cmap, antialiased=True)
        ax1.set_xlim(0, t.max())
        ax1.set_ylabel('Frequency [Hz]')
        ax1.set_xlabel('Time [hrs]')
        return fig
