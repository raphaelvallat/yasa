"""
Plotting functions of YASA.
"""
import mne
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lspopt import spectrogram_lspopt
from matplotlib.colors import Normalize, ListedColormap

__all__ = ['plot_hypnogram', 'plot_spectrogram', 'topoplot']


def plot_hypnogram(hypno, sf_hypno=1/30, lw=1.5, figsize=(9, 3)):
    """
    Plot a hypnogram.

    .. versionadded:: 0.6.0

    Parameters
    ----------
    hypno : array_like
        Sleep stage (hypnogram).

        .. note::
            The default hypnogram format in YASA is a 1D integer vector where:

            * -2 = Unscored
            * -1 = Artefact / Movement
            * 0 = Wake
            * 1 = N1 sleep
            * 2 = N2 sleep
            * 3 = N3 sleep
            * 4 = REM sleep
    sf_hypno : float
        The current sampling frequency of the hypnogram, in Hz, e.g.

        * 1/30 = 1 value per each 30 seconds of EEG data,
        * 1 = 1 value per second of EEG data
    lw : float
        Linewidth.
    figsize : tuple
       Width, height in inches.

    Returns
    -------
    ax : :py:class:`matplotlib.axes.Axes`
        Matplotlib Axes

    Examples
    --------
    .. plot::

        >>> import yasa
        >>> import numpy as np
        >>> hypno = np.loadtxt("https://github.com/raphaelvallat/yasa/raw/master/notebooks/data_full_6hrs_100Hz_hypno_30s.txt")
        >>> ax = yasa.plot_hypnogram(hypno)
    """
    # Increase font size while preserving original
    old_fontsize = plt.rcParams['font.size']
    plt.rcParams.update({'font.size': 18})

    # Safety checks
    assert isinstance(hypno, (np.ndarray, pd.Series, list)), 'hypno must be an array.'
    hypno = np.asarray(hypno).astype(int)
    assert (hypno >= -2).all() and (hypno <= 4).all(), "hypno values must be between -2 to 4."
    assert hypno.ndim == 1, 'hypno must be a 1D array.'
    assert isinstance(sf_hypno, (int, float)), 'sf must be int or float.'

    t_hyp = np.arange(hypno.size) / (sf_hypno * 3600)
    # Make sure that REM is displayed after Wake
    hypno = pd.Series(hypno).map({-2: -2, -1: -1, 0: 0, 1: 2, 2: 3, 3: 4, 4: 1}).values
    hypno_rem = np.ma.masked_not_equal(hypno, 1)
    hypno_art_uns = np.ma.masked_greater(hypno, -1)

    fig, ax0 = plt.subplots(nrows=1, figsize=figsize)

    # Hypnogram (top axis)
    ax0.step(t_hyp, -1 * hypno, color='k', lw=lw)
    ax0.step(t_hyp, -1 * hypno_rem, color='red', lw=lw)
    ax0.step(t_hyp, -1 * hypno_art_uns, color='grey', lw=lw)
    if -2 in hypno and -1 in hypno:
        # Both Unscored and Artefacts are present
        ax0.set_yticks([2, 1, 0, -1, -2, -3, -4])
        ax0.set_yticklabels(['Uns', 'Art', 'W', 'R', 'N1', 'N2', 'N3'])
        ax0.set_ylim(-4.5, 2.5)
    elif -2 in hypno and -1 not in hypno:
        # Only Unscored are present
        ax0.set_yticks([2, 0, -1, -2, -3, -4])
        ax0.set_yticklabels(['Uns', 'W', 'R', 'N1', 'N2', 'N3'])
        ax0.set_ylim(-4.5, 2.5)
    elif -2 not in hypno and -1 in hypno:
        # Only Artefacts are present
        ax0.set_yticks([1, 0, -1, -2, -3, -4])
        ax0.set_yticklabels(['Art', 'W', 'R', 'N1', 'N2', 'N3'])
        ax0.set_ylim(-4.5, 1.5)
    else:
        # No artefacts or Unscored
        ax0.set_yticks([0, -1, -2, -3, -4])
        ax0.set_yticklabels(['W', 'R', 'N1', 'N2', 'N3'])
        ax0.set_ylim(-4.5, 0.5)
    ax0.set_xlim(0, t_hyp.max())
    ax0.set_ylabel('Stage')
    ax0.set_xlabel('Time [hrs]')
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    # Revert font-size
    plt.rcParams.update({'font.size': old_fontsize})
    return ax0

def plot_spectrogram(data, sf, hypno=None, win_sec=30, fmin=0.5, fmax=25,
                     trimperc=2.5, cmap='RdBu_r'):
    """
    Plot a full-night multi-taper spectrogram, optionally with the hypnogram on top.

    For more details, please refer to the `Jupyter notebook
    <https://github.com/raphaelvallat/yasa/blob/master/notebooks/10_spectrogram.ipynb>`_

    .. versionadded:: 0.1.8

    Parameters
    ----------
    data : :py:class:`numpy.ndarray`
        Single-channel EEG data. Must be a 1D NumPy array.
    sf : float
        The sampling frequency of data AND the hypnogram.
    hypno : array_like
        Sleep stage (hypnogram), optional.

        The hypnogram must have the exact same number of samples as ``data``.
        To upsample your hypnogram, please refer to :py:func:`yasa.hypno_upsample_to_data`.

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
    win_sec : int or float
        The length of the sliding window, in seconds, used for multitaper PSD
        calculation. Default is 30 seconds. Note that ``data`` must be at least
        twice longer than ``win_sec`` (e.g. 60 seconds).
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
    fig : :py:class:`matplotlib.figure.Figure`
        Matplotlib Figure

    Examples
    --------
    1. Full-night multitaper spectrogram on Cz, no hypnogram

    .. plot::

        >>> import yasa
        >>> import numpy as np
        >>> # In the next 5 lines, we're loading the data from GitHub.
        >>> import requests
        >>> from io import BytesIO
        >>> r = requests.get('https://github.com/raphaelvallat/yasa/raw/master/notebooks/data_full_6hrs_100Hz_Cz%2BFz%2BPz.npz', stream=True)
        >>> npz = np.load(BytesIO(r.raw.read()))
        >>> data = npz.get('data')[0, :]
        >>> sf = 100
        >>> fig = yasa.plot_spectrogram(data, sf)

    2. Full-night multitaper spectrogram on Cz with the hypnogram on top

    .. plot::

        >>> import yasa
        >>> import numpy as np
        >>> # In the next lines, we're loading the data from GitHub.
        >>> import requests
        >>> from io import BytesIO
        >>> r = requests.get('https://github.com/raphaelvallat/yasa/raw/master/notebooks/data_full_6hrs_100Hz_Cz%2BFz%2BPz.npz', stream=True)
        >>> npz = np.load(BytesIO(r.raw.read()))
        >>> data = npz.get('data')[0, :]
        >>> sf = 100
        >>> # Load the 30-sec hypnogram and upsample to data
        >>> hypno = np.loadtxt('https://raw.githubusercontent.com/raphaelvallat/yasa/master/notebooks/data_full_6hrs_100Hz_hypno_30s.txt')
        >>> hypno = yasa.hypno_upsample_to_data(hypno, 1/30, data, sf)
        >>> fig = yasa.plot_spectrogram(data, sf, hypno, cmap='Spectral_r')
    """
    # Increase font size while preserving original
    old_fontsize = plt.rcParams['font.size']
    plt.rcParams.update({'font.size': 18})

    # Safety checks
    assert isinstance(data, np.ndarray), 'Data must be a 1D NumPy array.'
    assert isinstance(sf, (int, float)), 'sf must be int or float.'
    assert data.ndim == 1, 'Data must be a 1D (single-channel) NumPy array.'
    assert isinstance(win_sec, (int, float)), 'win_sec must be int or float.'
    assert isinstance(fmin, (int, float)), 'fmin must be int or float.'
    assert isinstance(fmax, (int, float)), 'fmax must be int or float.'
    assert fmin < fmax, 'fmin must be strictly inferior to fmax.'
    assert fmax < sf / 2, 'fmax must be less than Nyquist (sf / 2).'

    # Calculate multi-taper spectrogram
    nperseg = int(win_sec * sf)
    assert data.size > 2 * nperseg, 'Data length must be at least 2 * win_sec.'
    f, t, Sxx = spectrogram_lspopt(data, sf, nperseg=nperseg, noverlap=0)
    Sxx = 10 * np.log10(Sxx)  # Convert uV^2 / Hz --> dB / Hz

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
        im = ax.pcolormesh(t, f, Sxx, norm=norm, cmap=cmap, antialiased=True, shading="auto")
        ax.set_xlim(0, t.max())
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [hrs]')

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.95, fraction=0.1, aspect=25)
        cbar.ax.set_ylabel('Log Power (dB / Hz)', rotation=270, labelpad=20)
        return fig
    else:
        hypno = np.asarray(hypno).astype(int)
        assert hypno.ndim == 1, 'Hypno must be 1D.'
        assert hypno.size == data.size, 'Hypno must have the same sf as data.'
        t_hyp = np.arange(hypno.size) / (sf * 3600)
        # Make sure that REM is displayed after Wake
        hypno = pd.Series(hypno).map({-2: -2, -1: -1, 0: 0, 1: 2, 2: 3, 3: 4, 4: 1}).values
        hypno_rem = np.ma.masked_not_equal(hypno, 1)

        fig, (ax0, ax1) = plt.subplots(
            nrows=2, figsize=(12, 6), gridspec_kw={'height_ratios': [1, 2]})
        plt.subplots_adjust(hspace=0.1)

        # Hypnogram (top axis)
        ax0.step(t_hyp, -1 * hypno, color='k')
        ax0.step(t_hyp, -1 * hypno_rem, color='r')
        if -2 in hypno and -1 in hypno:
            # Both Unscored and Artefacts are present
            ax0.set_yticks([2, 1, 0, -1, -2, -3, -4])
            ax0.set_yticklabels(['Uns', 'Art', 'W', 'R', 'N1', 'N2', 'N3'])
            ax0.set_ylim(-4.5, 2.5)
        elif -2 in hypno and -1 not in hypno:
            # Only Unscored are present
            ax0.set_yticks([2, 0, -1, -2, -3, -4])
            ax0.set_yticklabels(['Uns', 'W', 'R', 'N1', 'N2', 'N3'])
            ax0.set_ylim(-4.5, 2.5)

        elif -2 not in hypno and -1 in hypno:
            # Only Artefacts are present
            ax0.set_yticks([1, 0, -1, -2, -3, -4])
            ax0.set_yticklabels(['Art', 'W', 'R', 'N1', 'N2', 'N3'])
            ax0.set_ylim(-4.5, 1.5)
        else:
            # No artefacts or Unscored
            ax0.set_yticks([0, -1, -2, -3, -4])
            ax0.set_yticklabels(['W', 'R', 'N1', 'N2', 'N3'])
            ax0.set_ylim(-4.5, 0.5)
        ax0.set_xlim(0, t_hyp.max())
        ax0.set_ylabel('Stage')
        ax0.xaxis.set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.spines['top'].set_visible(False)

        # Spectrogram (bottom axis)
        im = ax1.pcolormesh(t, f, Sxx, norm=norm, cmap=cmap, antialiased=True, shading="auto")
        ax1.set_xlim(0, t.max())
        ax1.set_ylabel('Frequency [Hz]')
        ax1.set_xlabel('Time [hrs]')

        # Revert font-size
        plt.rcParams.update({'font.size': old_fontsize})
        return fig


def topoplot(data, montage="standard_1020", vmin=None, vmax=None, mask=None, title=None,
             cmap=None, n_colors=100, cbar_title=None, cbar_ticks=None, figsize=(4, 4), dpi=80,
             fontsize=14, **kwargs):
    """
    Topoplot.

    This is a wrapper around :py:func:`mne.viz.plot_topomap`.

    For more details, please refer to this `example notebook
    <https://github.com/raphaelvallat/yasa/blob/master/notebooks/15_topoplot.ipynb>`_.

    .. versionadded:: 0.4.1

    Parameters
    ----------
    data : :py:class:`pandas.Series`
        A pandas Series with the values to plot. The index MUST be the channel
        names (e.g. ['C4', 'F4'] or ['C4-M1', 'C3-M2']).
    montage : str
        The name of the montage to use. Valid montages can be found at
        :py:func:`mne.channels.make_standard_montage`.
    vmin, vmax : float
        The minimum and maximum values of the colormap. If None, these will be
        defined based on the min / max values of ``data``.
    mask : :py:class:`pandas.Series`
        A pandas Series indicating the significant electrodes. The index MUST
        be the channel names (e.g. ['C4', 'F4'] or ['C4-M1', 'C3-M2']).
    title : str
        The plot title.
    cmap : str
        A matplotlib color palette. A list of color palette can be found at:
        https://seaborn.pydata.org/tutorial/color_palettes.html
    n_colors : int
        The number of colors to discretize the color palette.
    cbar_title : str
        The title of the colorbar.
    cbar_ticks : list
        The ticks of the colorbar.
    figsize : tuple
       Width, height in inches.
    dpi : int
        The resolution of the plot.
    fontsize : int
        Global font size of all the elements of the plot.
    **kwargs : dict
        Other arguments that are passed to :py:func:`mne.viz.plot_topomap`.

    Returns
    -------
    fig : :py:class:`matplotlib.figure.Figure`
        Matplotlib Figure

    Examples
    --------

    1. Plot all-positive values

    .. plot::

        >>> import yasa
        >>> import pandas as pd
        >>> data = pd.Series([4, 8, 7, 1, 2, 3, 5],
        ...                  index=['F4', 'F3', 'C4', 'C3', 'P3', 'P4', 'Oz'],
        ...                  name='Values')
        >>> fig = yasa.topoplot(data, title='My first topoplot')

    2. Plot correlation coefficients (values ranging from -1 to 1)

    .. plot::

        >>> import yasa
        >>> import pandas as pd
        >>> data = pd.Series([-0.5, -0.7, -0.3, 0.1, 0.15, 0.3, 0.55],
        ...                  index=['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'Pz'])
        >>> fig = yasa.topoplot(data, vmin=-1, vmax=1, n_colors=8,
        ...                     cbar_title="Pearson correlation")
    """
    # Increase font size while preserving original
    old_fontsize = plt.rcParams['font.size']
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams.update({'savefig.bbox': 'tight'})
    plt.rcParams.update({'savefig.transparent': 'True'})

    # Make sure we don't do any in-place modification
    assert isinstance(data, pd.Series), 'Data must be a Pandas Series'
    data = data.copy()

    # Add mask, if present
    if mask is not None:
        assert isinstance(mask, pd.Series), 'mask must be a Pandas Series'
        assert mask.dtype.kind in 'bi', "mask must be True/False or 0/1."
    else:
        mask = pd.Series(1, index=data.index, name="mask")

    # Convert to a dataframe (col1 = values, col2 = mask)
    data = data.to_frame().join(mask, how="left")

    # Preprocess channel names: C4-M1 --> C4
    data.index = data.index.str.split('-').str.get(0)

    # Define electrodes coordinates
    Info = mne.create_info(data.index.tolist(), sfreq=100, ch_types='eeg')
    Info.set_montage(montage, match_case=False, on_missing='ignore')
    chan = Info.ch_names

    # Define vmin and vmax
    if vmin is None:
        vmin = data.iloc[:, 0].min()
    if vmax is None:
        vmax = data.iloc[:, 0].max()

    # Choose and discretize colormap
    if cmap is None:
        if vmin < 0 and vmax <= 0:
            cmap = 'mako'
        elif vmin < 0 and vmax > 0:
            cmap = 'Spectral_r'
        elif vmin >= 0 and vmax > 0:
            cmap = 'rocket_r'

    cmap = ListedColormap(sns.color_palette(cmap, n_colors).as_hex())

    if 'sensors' not in kwargs:
        kwargs['sensors'] = False
    if 'res' not in kwargs:
        kwargs['res'] = 256
    if 'names' not in kwargs:
        kwargs['names'] = chan
    if 'show_names' not in kwargs:
        kwargs['show_names'] = True
    if 'mask_params' not in kwargs:
        kwargs['mask_params'] = dict(marker=None)

    # Hidden feature: if names='values', show the actual values.
    if kwargs['names'] == 'values':
        kwargs['names'] = data.iloc[:, 0][chan].round(2).to_numpy()

    # Start the plot
    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        im, _ = mne.viz.plot_topomap(
            data=data.iloc[:, 0][chan], pos=Info, vmin=vmin, vmax=vmax,
            mask=data.iloc[:, 1][chan], cmap=cmap, show=False, axes=ax,
            **kwargs)

        if title is not None:
            ax.set_title(title)

        # Add colorbar
        if cbar_title is None:
            cbar_title = data.iloc[:, 0].name

        cax = fig.add_axes([0.95, 0.3, 0.02, 0.5])
        cbar = fig.colorbar(im, cax=cax, ticks=cbar_ticks, fraction=0.5)
        cbar.set_label(cbar_title)

        # Revert font-size
        plt.rcParams.update({'font.size': old_fontsize})
    return fig
