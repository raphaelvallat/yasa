"""
Plotting functions of YASA.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from lspopt import spectrogram_lspopt
from matplotlib.colors import ListedColormap, Normalize
from scipy import stats

__all__ = ["plot_hypnogram", "plot_spectrogram", "topoplot"]


def plot_hypnogram(hyp, sf_hypno=1 / 30, highlight="REM", fill_color=None, ax=None, **kwargs):
    """
    Plot a hypnogram.

    .. versionadded:: 0.6.0

    Parameters
    ----------
    hyp : :py:class:`yasa.Hypnogram` or array_like
        A YASA hypnogram instance, or a 1D integer array where:

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

        This has no impact if `hyp` is a :py:class:`yasa.Hypnogram`.
    highlight : str or None
        Optional stage to highlight with alternate color.
    fill_color : str or None
        Optional color to fill space above hypnogram line.
    ax : :py:class:`matplotlib.axes.Axes`
        Axis on which to draw the plot, optional.
    **kwargs : dict
        Keyword arguments controlling hypnogram line display (e.g., ``lw``, ``linestyle``).
        Passed to :py:func:`matplotlib.pyplot.stairs` and py:func:`matplotlib.pyplot.hlines`.

    Returns
    -------
    ax : :py:class:`matplotlib.axes.Axes`
        Matplotlib Axes

    Examples
    --------
    .. plot::

        >>> from yasa import simulate_hypnogram
        >>> import matplotlib.pyplot as plt
        >>> hyp = simulate_hypnogram(tib=300, seed=11)
        >>> ax = hyp.plot_hypnogram()
        >>> plt.tight_layout()

    .. plot::

        >>> from yasa import Hypnogram
        >>> values = 4 * ["W", "N1", "N2", "N3", "REM"] + ["ART", "N2", "REM", "W", "UNS"]
        >>> hyp = Hypnogram(values, freq="24min").upsample("30s")
        >>> ax = hyp.plot_hypnogram(lw=2, fill_color="thistle")
        >>> plt.tight_layout()

    .. plot::

        >>> from yasa import simulate_hypnogram
        >>> import matplotlib.pyplot as plt
        >>> fig, axes = plt.subplots(nrows=2, figsize=(6, 4), constrained_layout=True)
        >>> hyp_a = simulate_hypnogram(n_stages=3, seed=99)
        >>> hyp_b = simulate_hypnogram(n_stages=3, seed=99, start="2022-01-31 23:30:00")
        >>> hyp_a.plot_hypnogram(lw=1, fill_color="whitesmoke", highlight=None, ax=axes[0])
        >>> hyp_b.plot_hypnogram(lw=1, fill_color="whitesmoke", highlight=None, ax=axes[1])
    """
    from .hypno import Hypnogram, hypno_int_to_str  # Avoiding circular imports

    if not isinstance(hyp, Hypnogram):
        # Convert sampling frequency to pandas timefrequency string (e.g., "30s")
        freq_str = pd.tseries.frequencies.to_offset(pd.Timedelta(1 / sf_hypno, "s")).freqstr
        # Prepend "1" if freqstr has no numeric prefix (e.g. "s" -> "1s" when sf_hypno=1)
        if not freq_str[0].isdigit():
            freq_str = "1" + freq_str
        # Create Hypnogram instance for plotting
        hyp = Hypnogram(hypno_int_to_str(hyp), freq=freq_str)

    # Work with a copy of the Hypnogram to not alter the original
    hyp = hyp.copy()

    # Increase font size while preserving original
    old_fontsize = plt.rcParams["font.size"]
    plt.rcParams.update({"font.size": 18})

    # Open the figure
    if ax is None:
        ax = plt.gca()

    ## Remap stages to be in desired y-axis order ##
    # Start with default of all allowed labels
    stage_order = hyp.labels.copy()
    stages_present = hyp.hypno.unique().tolist()
    # Reverse order so WAKE is highest, and exclude ART/UNS which are always last
    stage_order = stage_order[:-2][::-1]
    # Add ART/UNS back above WAKE if they're present in the current hypnogram or existing axis
    gca_ylabels = [x.get_text() for x in ax.get_yticklabels()]
    if "ART" in stages_present or "ART" in gca_ylabels:
        stage_order += ["ART"]
    if "UNS" in stages_present or "UNS" in gca_ylabels:
        stage_order += ["UNS"]
    # Put REM after WAKE if all 5 standard stages are allowed
    if hyp.n_stages == 5:
        stage_order.insert(stage_order.index("WAKE") - 1, stage_order.pop(stage_order.index("REM")))
    # Reset the Hypnogram mapping so any future returns have this order
    hyp.mapping = {stage: i for i, stage in enumerate(stage_order)}

    ## Extract values to plot ##
    hypno = hyp.as_int()
    # Reduce to breakpoints (where stages change) to avoid drawing individual lines for every epoch
    hypno = hypno[hypno.shift().ne(hypno)]
    # Extract x-values (bins) and y-values to plot
    yvalues = hypno.to_numpy()
    if hyp.start is not None:
        final_bin_edge = pd.Timestamp(hyp.start) + pd.Timedelta(hyp.duration, unit="min")
        bins = np.append(hypno.index.to_list(), final_bin_edge)
        bins = [mdates.date2num(b) for b in bins]
        xlabel = "Time"
    else:
        final_bin_edge = hyp.duration * 60
        bins = np.append(hyp.timedelta[hypno.index].total_seconds(), final_bin_edge)
        bins /= 60 if hyp.duration <= 90 else 3600
        xlabel = "Time [mins]" if hyp.duration <= 90 else "Time [hrs]"

    # Make mask to draw the highlighted stage
    yvals_highlight = np.ma.masked_not_equal(yvalues, hyp.mapping.get(highlight))

    # Draw background filling
    if fill_color is not None:
        bline = hyp.mapping["WAKE"]
        ax.stairs(yvalues.clip(max=bline), bins, baseline=bline, color=fill_color, fill=True, lw=0)
    # Draw main hypnogram line and highlighted stage line
    line_kwargs = {"color": "black", "linewidth": 1.5, "label": hyp.scorer}
    line_kwargs.update(kwargs)
    ax.stairs(yvalues, bins, baseline=None, **line_kwargs)
    if not yvals_highlight.mask.all():
        line_kwargs.update({"color": "red", "label": None})
        ax.hlines(yvals_highlight, xmin=bins[:-1], xmax=bins[1:], **line_kwargs)

    # Aesthetics
    ax.use_sticky_edges = False
    ax.margins(x=0, y=1 / len(stage_order) / 2)  # 1/n_epochs/2 gives half-unit margins
    ax.set_yticks(range(len(stage_order)))
    ax.set_yticklabels(stage_order)
    ax.set_ylabel("Stage")
    ax.set_xlabel(xlabel)
    ax.spines[["right", "top"]].set_visible(False)
    if hyp.start is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    # Revert font-size
    plt.rcParams.update({"font.size": old_fontsize})
    return ax


def plot_spectrogram(
    data,
    sf,
    hypno=None,
    win_sec=30,
    fmin=0.5,
    fmax=25,
    trimperc=2.5,
    cmap="RdBu_r",
    vmin=None,
    vmax=None,
    **kwargs,
):
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
    hypno : array_like or :py:class:`yasa.Hypnogram`
        Sleep stage (hypnogram), optional.

        Can be an upsampled integer array (same number of samples as ``data``) or a
        :py:class:`yasa.Hypnogram` instance (automatically upsampled). When a
        :py:class:`yasa.Hypnogram` is passed, the hypnogram is used directly for plotting.

        To manually upsample an integer array, use :py:meth:`yasa.Hypnogram.upsample_to_data` or
        :py:func:`yasa.hypno_upsample_to_data`.

        .. note::
            When passing an integer array, hypnogram values follow this mapping:

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
    vmin : int or float
        The lower range of color scale. Overwrites ``trimperc``
    vmax : int or float
        The upper range of color scale. Overwrites ``trimperc``
    **kwargs : dict
        Other arguments that are passed to :py:meth:`yasa.Hypnogram.plot_hypnogram`.

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
        >>> fpath = yasa.fetch_sample("full_6hrs_100Hz_Cz+Fz+Pz.npz")
        >>> npz = np.load(fpath)
        >>> data = npz["data"][0, :]
        >>> sf = 100
        >>> fig = yasa.plot_spectrogram(data, sf)

    2. Full-night multitaper spectrogram on Cz with the hypnogram on top (legacy integer array)

    .. plot::

        >>> import yasa
        >>> import numpy as np
        >>> fpath = yasa.fetch_sample("full_6hrs_100Hz_Cz+Fz+Pz.npz")
        >>> npz = np.load(fpath)
        >>> data = npz["data"][0, :]
        >>> sf = 100
        >>> hypno = np.loadtxt(yasa.fetch_sample("full_6hrs_100Hz_hypno_30s.txt"))
        >>> hypno = yasa.hypno_upsample_to_data(hypno, 1 / 30, data, sf)
        >>> fig = yasa.plot_spectrogram(data, sf, hypno, cmap="Spectral_r")

    3. Same plot using a :py:class:`~yasa.Hypnogram` directly — no upsampling needed:

    .. plot::

        >>> import yasa
        >>> import numpy as np
        >>> fpath = yasa.fetch_sample("full_6hrs_100Hz_Cz+Fz+Pz.npz")
        >>> npz = np.load(fpath)
        >>> data = npz["data"][0, :]
        >>> sf = 100
        >>> hypno_30s = yasa.hypno_int_to_str(
        ...     np.loadtxt(yasa.fetch_sample("full_6hrs_100Hz_hypno_30s.txt")).astype(int)
        ... )
        >>> hyp = yasa.Hypnogram(hypno_30s, freq="30s")
        >>> fig = yasa.plot_spectrogram(data, sf, hyp, cmap="Spectral_r")
    """
    from .hypno import Hypnogram, hypno_int_to_str  # Avoiding circular imports

    # Increase font size while preserving original
    old_fontsize = plt.rcParams["font.size"]
    plt.rcParams.update({"font.size": 18})

    # If hypno is a Hypnogram instance, upsample it and keep the original for plotting
    hyp_obj = None
    if isinstance(hypno, Hypnogram):
        hyp_obj = hypno  # Use directly for plotting
        hypno = hypno.upsample_to_data(data, sf=sf)

    # Safety checks
    assert isinstance(data, np.ndarray), "`data` must be a 1D NumPy array."
    assert isinstance(sf, (int, float)), "`sf` must be int or float."
    assert data.ndim == 1, "`data` must be a 1D (single-channel) NumPy array."
    assert isinstance(win_sec, (int, float)), "`win_sec` must be int or float."
    assert isinstance(fmin, (int, float)), "`fmin` must be int or float."
    assert isinstance(fmax, (int, float)), "`fmax` must be int or float."
    assert fmin < fmax, "`fmin` must be strictly inferior to `fmax`."
    assert fmax < sf / 2, "`fmax` must be less than Nyquist (sf / 2)."
    assert isinstance(vmin, (int, float, type(None))), "`vmin` must be int, float, or None."
    assert isinstance(vmax, (int, float, type(None))), "`vmax` must be int, float, or None."
    if vmin is not None:
        assert isinstance(vmax, (int, float)), "`vmax` must be int or float if `vmin` is provided."
    if vmax is not None:
        assert isinstance(vmin, (int, float)), "`vmin` must be int or float if `vmax` is provided."
    if hypno is not None:
        assert hypno.size == data.size, "`hypno` must have the same number of samples as `data`."

    # Calculate multi-taper spectrogram
    nperseg = int(win_sec * sf)
    assert data.size > 2 * nperseg, "`data` length must be at least 2 * `win_sec`."
    f, t, Sxx = spectrogram_lspopt(data, sf, nperseg=nperseg, noverlap=0)
    Sxx = 10 * np.log10(Sxx)  # Convert uV^2 / Hz --> dB / Hz

    # Select only relevant frequencies (up to 30 Hz)
    good_freqs = np.logical_and(f >= fmin, f <= fmax)
    Sxx = Sxx[good_freqs, :]
    f = f[good_freqs]
    t /= 3600  # Convert t to hours

    # Normalization
    if vmin is None:
        vmin, vmax = np.percentile(Sxx, [0 + trimperc, 100 - trimperc])
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Open figure
    if hypno is None:
        fig, ax1 = plt.subplots(nrows=1, figsize=(12, 4))
    else:
        fig, (ax0, ax1) = plt.subplots(
            nrows=2,
            figsize=(12, 6),
            gridspec_kw={"height_ratios": [1, 2], "hspace": 0.1},
        )

    # Draw Spectrogram
    im = ax1.pcolormesh(t, f, Sxx, norm=norm, cmap=cmap, antialiased=True, shading="auto")
    ax1.set_xlim(0, t.max())
    ax1.set_ylabel("Frequency [Hz]")
    ax1.set_xlabel("Time [hrs]")

    if hypno is not None:
        if hyp_obj is None:
            # Convert sampling frequency to pandas timefrequency string (e.g., "30s")
            freq_str = pd.tseries.frequencies.to_offset(pd.Timedelta(1 / sf, "s")).freqstr
            # Prepend "1" if freqstr has no numeric prefix (e.g. "s" -> "1s" when sf=1)
            if not freq_str[0].isdigit():
                freq_str = "1" + freq_str
            # Create Hypnogram instance for plotting
            hyp_obj = Hypnogram(hypno_int_to_str(hypno), freq=freq_str)
        hypnoplot_kwargs = dict(lw=1.5, fill_color=None)
        hypnoplot_kwargs.update(kwargs)
        # Draw hypnogram
        ax0 = hyp_obj.plot_hypnogram(ax=ax0, **hypnoplot_kwargs)
        ax0.xaxis.set_visible(False)
    else:
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax1, shrink=0.95, fraction=0.1, aspect=25)
        cbar.ax.set_ylabel("Log Power (dB / Hz)", rotation=270, labelpad=20)

    # Revert font-size
    plt.rcParams.update({"font.size": old_fontsize})
    return fig


def topoplot(
    data,
    montage="standard_1020",
    vmin=None,
    vmax=None,
    mask=None,
    title=None,
    cmap=None,
    n_colors=100,
    cbar_title=None,
    cbar_ticks=None,
    figsize=(4, 4),
    dpi=80,
    fontsize=14,
    ax=None,
    **kwargs,
):
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
    ax : :py:class:`matplotlib.axes.Axes` or None
        Axes on which to draw the topoplot. If None (default), a new figure
        and axes are created. When ``ax`` is provided, the ``figsize`` and
        ``dpi`` arguments are ignored.
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
        >>> data = pd.Series(
        ...     [4, 8, 7, 1, 2, 3, 5],
        ...     index=["F4", "F3", "C4", "C3", "P3", "P4", "Oz"],
        ...     name="Values",
        ... )
        >>> fig = yasa.topoplot(data, title="My first topoplot")

    2. Plot correlation coefficients (values ranging from -1 to 1)

    .. plot::

        >>> import yasa
        >>> import pandas as pd
        >>> data = pd.Series(
        ...     [-0.5, -0.7, -0.3, 0.1, 0.15, 0.3, 0.55],
        ...     index=["F3", "Fz", "F4", "C3", "Cz", "C4", "Pz"],
        ... )
        >>> fig = yasa.topoplot(data, vmin=-1, vmax=1, n_colors=8, cbar_title="Pearson correlation")

    3. Plot two topoplots side-by-side using the ``ax`` parameter

    .. plot::

        >>> import yasa
        >>> import pandas as pd
        >>> import matplotlib.pyplot as plt
        >>> data1 = pd.Series(
        ...     [4, 8, 7, 1, 2, 3, 5],
        ...     index=["F4", "F3", "C4", "C3", "P3", "P4", "Oz"],
        ...     name="NREM",
        ... )
        >>> data2 = pd.Series(
        ...     [2, 5, 4, 3, 6, 7, 4],
        ...     index=["F4", "F3", "C4", "C3", "P3", "P4", "Oz"],
        ...     name="REM",
        ... )
        >>> fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        >>> yasa.topoplot(data1, ax=axes[0])
        >>> yasa.topoplot(data2, ax=axes[1])
    """
    # Increase font size while preserving original
    old_fontsize = plt.rcParams["font.size"]
    plt.rcParams.update({"font.size": fontsize})
    plt.rcParams.update({"savefig.bbox": "tight"})
    plt.rcParams.update({"savefig.transparent": "True"})

    # Make sure we don't do any in-place modification
    assert isinstance(data, pd.Series), "`data` must be a Pandas Series"
    data = data.copy()

    # Add mask, if present
    if mask is not None:
        assert isinstance(mask, pd.Series), "`mask` must be a Pandas Series"
        assert mask.dtype.kind in "bi", "`mask` must be True/False or 0/1."
    else:
        mask = pd.Series(1, index=data.index, name="mask")

    # Convert to a dataframe (col1 = values, col2 = mask)
    data = data.to_frame().join(mask, how="left")

    # Preprocess channel names: C4-M1 --> C4
    data.index = data.index.str.split("-").str.get(0)

    # Define electrodes coordinates
    Info = mne.create_info(data.index.tolist(), sfreq=100, ch_types="eeg")
    Info.set_montage(montage, match_case=False, on_missing="ignore")
    chan = Info.ch_names

    # Define vmin and vmax
    if vmin is None:
        vmin = data.iloc[:, 0].min()
    if vmax is None:
        vmax = data.iloc[:, 0].max()

    # Choose and discretize colormap
    if cmap is None:
        if vmin < 0 and vmax <= 0:
            cmap = "mako"
        elif vmin < 0 and vmax > 0:
            cmap = "Spectral_r"
        elif vmin >= 0 and vmax > 0:
            cmap = "rocket_r"

    cmap = ListedColormap(sns.color_palette(cmap, n_colors).as_hex())

    if "sensors" not in kwargs:
        kwargs["sensors"] = False
    if "res" not in kwargs:
        kwargs["res"] = 256
    if "names" not in kwargs:
        kwargs["names"] = chan
    if "mask_params" not in kwargs:
        kwargs["mask_params"] = dict(marker=None)

    # Hidden feature: if names='values', show the actual values.
    if kwargs["names"] == "values":
        kwargs["names"] = data.iloc[:, 0][chan].round(2).to_numpy()

    # Start the plot
    with sns.axes_style("white"):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            _ax_provided = False
        else:
            fig = ax.get_figure()
            _ax_provided = True

        if "show_names" in kwargs:
            kwargs.pop("show_names")
        im, _ = mne.viz.plot_topomap(
            data=data.iloc[:, 0][chan],
            pos=Info,
            vlim=(vmin, vmax),
            mask=data.iloc[:, 1][chan],
            cmap=cmap,
            show=False,
            axes=ax,
            **kwargs,
        )

        if title is not None:
            ax.set_title(title)

        # Add colorbar
        if cbar_title is None:
            cbar_title = data.iloc[:, 0].name

        if _ax_provided:
            cbar = fig.colorbar(im, ax=ax, ticks=cbar_ticks, fraction=0.046, pad=0.04)
        else:
            cax = fig.add_axes([0.95, 0.3, 0.02, 0.5])
            cbar = fig.colorbar(im, cax=cax, ticks=cbar_ticks, fraction=0.5)
        cbar.set_label(cbar_title)

        # Revert font-size
        plt.rcParams.update({"font.size": old_fontsize})
    return fig


def blandaltman(
    x, y, agreement=1.96, xaxis="mean", confidence=0.95, annotate=True, ax=None, **kwargs
):
    """
    Generate a Bland-Altman plot to compare two sets of measurements.

    Parameters
    ----------
    x, y : pd.Series, np.array, or list
        First and second measurements.
    agreement : float
        Multiple of the standard deviation to plot agreement limits.
        The defaults is 1.96, which corresponds to 95% confidence interval if
        the differences are normally distributed.
    xaxis : str
        Define which measurements should be used as the reference (x-axis).
        Default is to use the average of x and y ("mean"). Accepted values are
        "mean", "x" or "y".
    confidence : float
        If not None, plot the specified percentage confidence interval of
        the mean and limits of agreement. The CIs of the mean difference and
        agreement limits describe a possible error in the
        estimate due to a sampling error. The greater the sample size,
        the narrower the CIs will be.
    annotate : bool
        If True (default), annotate the values for the mean difference
        and agreement limits.
    ax : matplotlib axes
        Axis on which to draw the plot.
    **kwargs : optional
        Optional argument(s) passed to :py:func:`matplotlib.pyplot.scatter`.

    Returns
    -------
    ax : Matplotlib Axes instance
        Returns the Axes object with the plot for further tweaking.

    Notes
    -----
    Bland-Altman plots [1]_ are extensively used to evaluate the agreement
    among two different instruments or two measurements techniques.
    They allow identification of any systematic difference between the
    measurements (i.e., fixed bias) or possible outliers.

    The mean difference (= x - y) is the estimated bias, and the SD of the
    differences measures the random fluctuations around this mean.
    If the mean value of the difference differs significantly from 0 on the
    basis of a 1-sample t-test, this indicates the presence of fixed bias.
    If there is a consistent bias, it can be adjusted for by subtracting the
    mean difference from the new method.

    It is common to compute 95% limits of agreement for each comparison
    (average difference ± 1.96 standard deviation of the difference), which
    tells us how far apart measurements by 2 methods were more likely to be
    for most individuals. If the differences within mean ± 1.96 SD are not
    clinically important, the two methods may be used interchangeably.
    The 95% limits of agreement can be unreliable estimates of the population
    parameters especially for small sample sizes so, when comparing methods
    or assessing repeatability, it is important to calculate confidence
    intervals for the 95% limits of agreement.

    The code is an adaptation of the
    `PyCompare <https://github.com/jaketmp/pyCompare>`_ package. The present
    implementation is a simplified version; please refer to the original
    package for more advanced functionalities.

    References
    ----------
    .. [1] Bland, J. M., & Altman, D. (1986). Statistical methods for assessing
           agreement between two methods of clinical measurement. The lancet,
           327(8476), 307-310.
    """
    # Safety check
    assert xaxis in ["mean", "x", "y"]
    # Get names before converting to NumPy array
    xname = x.name if isinstance(x, pd.Series) else "x"
    yname = y.name if isinstance(y, pd.Series) else "y"
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.ndim == 1 and y.ndim == 1
    assert x.size == y.size
    assert not np.isnan(x).any(), "Missing values in x or y are not supported."
    assert not np.isnan(y).any(), "Missing values in x or y are not supported."

    # Update default kwargs with specified inputs
    _scatter_kwargs = {"color": "tab:blue", "alpha": 0.8}
    _scatter_kwargs.update(kwargs)

    # Calculate mean, STD and SEM of x - y
    n = x.size
    dof = n - 1
    diff = x - y
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    mean_diff_se = np.sqrt(std_diff**2 / n)
    # Limits of agreements
    high = mean_diff + agreement * std_diff
    low = mean_diff - agreement * std_diff
    high_low_se = np.sqrt(3 * std_diff**2 / n)

    # Define x-axis
    if xaxis == "mean":
        xval = np.vstack((x, y)).mean(0)
        xlabel = f"Mean of {xname} and {yname}"
    elif xaxis == "x":
        xval = x
        xlabel = xname
    else:
        xval = y
        xlabel = yname

    # Start the plot
    if ax is None:
        ax = plt.gca()

    # Plot the mean diff, limits of agreement and scatter
    ax.scatter(xval, diff, **_scatter_kwargs)
    ax.axhline(mean_diff, color="k", linestyle="-", lw=2)
    ax.axhline(high, color="k", linestyle=":", lw=1.5)
    ax.axhline(low, color="k", linestyle=":", lw=1.5)

    # Annotate values
    if annotate:
        loa_range = high - low
        offset = (loa_range / 100.0) * 1.5
        trans = plt.matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
        xloc = 0.98
        ax.text(xloc, mean_diff + offset, "Mean", ha="right", va="bottom", transform=trans)
        ax.text(xloc, mean_diff - offset, "%.2f" % mean_diff, ha="right", va="top", transform=trans)
        ax.text(
            xloc, high + offset, "+%.2f SD" % agreement, ha="right", va="bottom", transform=trans
        )
        ax.text(xloc, high - offset, "%.2f" % high, ha="right", va="top", transform=trans)
        ax.text(xloc, low - offset, "-%.2f SD" % agreement, ha="right", va="top", transform=trans)
        ax.text(xloc, low + offset, "%.2f" % low, ha="right", va="bottom", transform=trans)

    # Add 95% confidence intervals for mean bias and limits of agreement
    if confidence is not None:
        assert 0 < confidence < 1
        ci = dict()
        ci["mean"] = stats.t.interval(confidence, dof, loc=mean_diff, scale=mean_diff_se)
        ci["high"] = stats.t.interval(confidence, dof, loc=high, scale=high_low_se)
        ci["low"] = stats.t.interval(confidence, dof, loc=low, scale=high_low_se)
        ax.axhspan(ci["mean"][0], ci["mean"][1], facecolor="tab:grey", alpha=0.2)
        ax.axhspan(ci["high"][0], ci["high"][1], facecolor=_scatter_kwargs["color"], alpha=0.2)
        ax.axhspan(ci["low"][0], ci["low"][1], facecolor=_scatter_kwargs["color"], alpha=0.2)

    # Labels
    ax.set_ylabel(f"{xname} - {yname}")
    ax.set_xlabel(xlabel)
    return ax
