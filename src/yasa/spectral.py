"""
This file contains several helper functions to calculate spectral power from
1D and 2D EEG data.
"""

import fractions
import logging
import warnings

import mne
import numpy as np
import pandas as pd
from scipy import signal
from scipy.integrate import simpson
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import curve_fit

from ._validation import _check_data, _check_hypno_include
from .io import _restore_log_level

logger = logging.getLogger("yasa")

__all__ = ["bandpower", "bandpower_from_psd", "bandpower_from_psd_ndarray", "irasa", "stft_power"]

# Default frequency bands (lower frequency, upper frequency, name). A tuple, so that it can
# safely be used as a default argument.
DEFAULT_BANDS = (
    (0.5, 4, "Delta"),
    (4, 8, "Theta"),
    (8, 12, "Alpha"),
    (12, 16, "Sigma"),
    (16, 30, "Beta"),
    (30, 40, "Gamma"),
)

# Default resampling factors of irasa: 1.1 to 1.9 with an increment of 0.05
_DEFAULT_HSET = (
    1.1,
    1.15,
    1.2,
    1.25,
    1.3,
    1.35,
    1.4,
    1.45,
    1.5,
    1.55,
    1.6,
    1.65,
    1.7,
    1.75,
    1.8,
    1.85,
    1.9,
)

# Default keyword arguments of scipy.signal.welch in bandpower and irasa
_DEFAULT_WELCH_KWARGS = {"average": "median", "window": "hamming"}


def _check_welch_kwargs(welch_kwargs, kwargs_welch, stacklevel):
    """Return the keyword arguments of :py:func:`scipy.signal.welch`.

    ``kwargs_welch`` is the name used before v0.8.0. ``stacklevel`` points the FutureWarning to
    the caller of the public function.
    """
    if kwargs_welch is not None:
        warnings.warn(
            "The `kwargs_welch` argument is deprecated and will be removed in v0.9. "
            "Please use `welch_kwargs` instead.",
            FutureWarning,
            stacklevel=stacklevel,
        )
        if welch_kwargs is not None:
            raise TypeError("Use either `welch_kwargs` or the deprecated `kwargs_welch`, not both.")
        welch_kwargs = kwargs_welch
    return dict(_DEFAULT_WELCH_KWARGS) if welch_kwargs is None else welch_kwargs


def _bands_range(bands):
    """Return the minimum and maximum frequencies of a list of bands."""
    return min(b[0] for b in bands), max(b[1] for b in bands)


def _bandpower_psd(psd, freqs, bands):
    """Integrate a N-D PSD (..., n_freqs) in each frequency band, using Simpson's rule.

    Returns
    -------
    bp : np.ndarray
        Absolute power in each band, of shape (n_bands, ...).
    total_power : np.ndarray
        Absolute power between the minimum and maximum frequencies of the bands, of shape (...).
    res : float
        Frequency resolution of the PSD.
    """
    fmin, fmax = _bands_range(bands)
    idx_good_freq = np.logical_and(freqs >= fmin, freqs <= fmax)
    freqs = freqs[idx_good_freq]
    if freqs.size < 2:
        raise ValueError(f"At least 2 frequency bins are required between {fmin} and {fmax} Hz.")
    res = freqs[1] - freqs[0]
    psd = psd[..., idx_good_freq]

    # Check if there are negative values in PSD
    if (psd < 0).any():
        logger.warning(
            "There are negative values in PSD. This will result in incorrect "
            "bandpower values. We highly recommend working with an "
            "all-positive PSD. For more details, please refer to: "
            "https://github.com/raphaelvallat/yasa/issues/29"
        )

    total_power = simpson(psd, dx=res, axis=-1)
    bp = np.zeros((len(bands), *psd.shape[:-1]), dtype=np.float64)
    for i, (b0, b1, name) in enumerate(bands):
        idx_band = np.logical_and(freqs >= b0, freqs <= b1)
        if idx_band.sum() < 2:
            raise ValueError(
                f"Band {name} ({b0}-{b1} Hz) contains fewer than 2 frequency bins at a frequency "
                f"resolution of {res} Hz. Use a wider band or a longer window."
            )
        bp[i] = simpson(psd[..., idx_band], dx=res, axis=-1)
    return bp, total_power, res


def bandpower(
    data,
    sf=None,
    ch_names=None,
    hypno=None,
    include=(2, 3),
    win_sec=4,
    relative=True,
    bandpass=False,
    bands=DEFAULT_BANDS,
    welch_kwargs=None,
    *,
    kwargs_welch=None,
):
    """
    Calculate the Welch bandpower for each channel and, if specified, for each sleep stage.

    .. versionadded:: 0.1.6

    Parameters
    ----------
    data : array_like or :py:class:`mne.io.BaseRaw`
        1D or 2D EEG data. If ``data`` is *array_like*, unit must be uV.
        If ``data`` is a :py:class:`~mne.io.BaseRaw` instance, ``data``, ``sf``, and
        ``ch_names`` will be automatically extracted, and ``data`` will be automatically
        converted from Volts (MNE) to micro-Volts (YASA).
    sf : float
        The sampling frequency of data AND the hypnogram if ``data`` is *array_like*.
        Should be omitted if ``data`` is a :py:class:`~mne.io.BaseRaw`.
    ch_names : list
        List of channel names, e.g. ['Cz', 'F3', 'F4', ...], if ``data`` is *array_like*.
        If None, channels will be labelled ['CHAN000', 'CHAN001', ...].
        Should be omitted if ``data`` is a :py:class:`~mne.io.BaseRaw`.
    hypno : array_like or :py:class:`yasa.Hypnogram`
        Sleep stage (hypnogram). If the hypnogram is loaded, the bandpower will be extracted for
        each sleep stage defined in ``include``.

        Can be an upsampled integer array (same number of samples as ``data``) or a
        :py:class:`yasa.Hypnogram` instance (automatically upsampled). To manually upsample an
        integer array, use :py:meth:`yasa.Hypnogram.upsample_to_data`.

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
        Values in ``hypno`` that will be included in the mask. The default is (2, 3), meaning that
        the bandpower are sequentially calculated for N2 and N3 sleep. This has no effect when
        ``hypno`` is None.

        When ``hypno`` is a :py:class:`yasa.Hypnogram`, string labels can be used instead of
        integers (e.g. ``["N2", "N3"]``).
    win_sec : int or float
        The length of the sliding window, in seconds, used for the Welch PSD calculation.
        Ideally, this should be at least two times the inverse of the lower frequency of
        interest (e.g. for a lower frequency of interest of 0.5 Hz, the window length should
        be at least 2 * 1 / 0.5 = 4 seconds).
    relative : boolean
        If True, bandpower is divided by the total power between the min and max frequencies
        defined in ``band``.
    bandpass : boolean
        If True, apply a standard FIR bandpass filter using the minimum and maximum frequencies
        in ``bands``. Fore more details, refer to :py:func:`mne.filter.filter_data`.
    bands : list of tuples
        List of frequency bands of interests. Each tuple must contain the lower and upper
        frequencies, as well as the band name (e.g. (0.5, 4, 'Delta')).
    welch_kwargs : dict or None
        Optional keywords arguments that are passed to the :py:func:`scipy.signal.welch` function.
        Default is ``{"average": "median", "window": "hamming"}``.

        .. versionadded:: 0.8.0
            Replaces ``kwargs_welch``.
    kwargs_welch : dict or None
        .. deprecated:: 0.8.0
            Use ``welch_kwargs`` instead. This argument will be removed in v0.9.

    Returns
    -------
    bandpowers : :py:class:`pandas.DataFrame`
        Bandpower dataframe, in which each row is a channel and each column a spectral band.

    Examples
    --------
    1. Bandpower per sleep stage using an upsampled integer hypnogram (legacy):

    .. code-block:: python

        >>> import yasa
        >>> bp = yasa.bandpower(raw, hypno=hypno_up, include=(2, 3, 4))  # doctest: +SKIP

    2. Pass a :py:class:`~yasa.Hypnogram` directly — upsampling is handled automatically.
       String stage labels can be used for ``include``:

    .. code-block:: python

        >>> hyp = yasa.Hypnogram.from_integers(hypno_30s, freq="30s")  # doctest: +SKIP
        >>> bp = yasa.bandpower(raw, hypno=hyp, include=["N2", "N3", "REM"])  # doctest: +SKIP

    For a full walkthrough, please refer to:
    https://github.com/raphaelvallat/yasa/blob/master/notebooks/08_bandpower.ipynb
    """
    # Type checks
    welch_kwargs = _check_welch_kwargs(welch_kwargs, kwargs_welch, stacklevel=3)
    assert isinstance(bands, (list, tuple)), "bands must be a list of tuple(s)"
    assert isinstance(relative, bool), "relative must be a boolean"
    assert isinstance(bandpass, bool), "bandpass must be a boolean"

    data, sf, ch_names, raw = _check_data(data, sf, ch_names)

    if bandpass:
        # Apply FIR bandpass filter
        fmin, fmax = _bands_range(bands)
        data = mne.filter.filter_data(data, sf, fmin, fmax, verbose=False)

    win = int(win_sec * sf)  # nperseg

    if hypno is None:
        # Calculate the PSD over the whole data
        freqs, psd = signal.welch(data, sf, nperseg=win, **welch_kwargs)
        bp = bandpower_from_psd(psd, freqs, ch_names, bands=bands, relative=relative)
        return bp.set_index("Chan")

    # Per each sleep stage defined in ``include``. When ``include`` contains string labels of a
    # Hypnogram, int_to_str is used to label the output index with the original string labels,
    # e.g. so that ``bp.xs("N3")`` works as documented. The original Raw is passed so that a
    # Hypnogram with a start time is aligned with the recording using absolute timestamps.
    hypno, include, int_to_str = _check_hypno_include(
        hypno, include, raw if raw is not None else data, sf, verbose=None
    )
    bp_stages = []
    for stage in include:
        is_stage = hypno == stage
        if not is_stage.any():
            continue
        if is_stage.sum() < win:
            # Welch would silently shorten the window, giving a coarse and unreliable PSD
            logger.warning(
                f"Stage {int_to_str.get(stage, stage)} is shorter than the Welch window "
                f"({win_sec} seconds). Skipping stage."
            )
            continue
        freqs, psd = signal.welch(data[:, is_stage], sf, nperseg=win, **welch_kwargs)
        bp_stage = bandpower_from_psd(psd, freqs, ch_names, bands=bands, relative=relative)
        bp_stage["Stage"] = int_to_str.get(stage, stage)
        bp_stages.append(bp_stage)
    if not bp_stages:
        raise ValueError(
            f"All the stages in `include` are shorter than the Welch window ({win_sec} seconds)."
        )
    return pd.concat(bp_stages, axis=0).set_index(["Stage", "Chan"])


def bandpower_from_psd(
    psd,
    freqs,
    ch_names=None,
    bands=DEFAULT_BANDS,
    relative=True,
):
    """Compute the average power of the EEG in specified frequency band(s)
    given a pre-computed PSD.

    .. versionadded:: 0.1.5

    Parameters
    ----------
    psd : array_like
        Power spectral density of data, in uV^2/Hz. Must be of shape (n_channels, n_freqs).
        See :py:func:`scipy.signal.welch` for more details.
    freqs : array_like
        Array of frequencies.
    ch_names : list
        List of channel names, e.g. ['Cz', 'F3', 'F4', ...]. If None, channels will be labelled
        ['CHAN000', 'CHAN001', ...].
    bands : list of tuples
        List of frequency bands of interests. Each tuple must contain the lower and upper
        frequencies, as well as the band name (e.g. (0.5, 4, 'Delta')).
    relative : boolean
        If True, bandpower is divided by the total power between the min and
        max frequencies defined in ``band`` (default 0.5 to 40 Hz).

    Returns
    -------
    bandpowers : :py:class:`pandas.DataFrame`
        Bandpower dataframe, in which each row is a channel and each column a spectral band.
    """
    # Type checks
    assert isinstance(bands, (list, tuple)), "bands must be a list of tuple(s)"
    assert isinstance(relative, bool), "relative must be a boolean"

    # Safety checks
    freqs = np.asarray(freqs)
    assert freqs.ndim == 1, "freqs must be a 1-D array of shape (n_freqs,)"
    psd = np.atleast_2d(psd)
    assert psd.ndim == 2, "PSD must be of shape (n_channels, n_freqs)."
    assert psd.shape[1] == freqs.size, "PSD must be of shape (n_channels, n_freqs)."
    nchan = psd.shape[0]
    if ch_names is not None:
        ch_names = np.atleast_1d(np.asarray(ch_names, dtype=str))
        assert ch_names.ndim == 1, "ch_names must be 1D."
        assert len(ch_names) == nchan, "ch_names must match psd.shape[0]."
    else:
        ch_names = ["CHAN" + str(i).zfill(3) for i in range(nchan)]

    bp, total_power, res = _bandpower_psd(psd, freqs, bands)
    if relative:
        bp /= total_power

    # Convert to DataFrame
    bp = pd.DataFrame(bp.T, columns=[b[2] for b in bands])
    bp["TotalAbsPow"] = total_power
    bp["FreqRes"] = res
    bp["Relative"] = relative
    bp.insert(0, "Chan", ch_names)
    # Add hidden attributes
    bp.bands_ = str(list(bands))
    return bp


def bandpower_from_psd_ndarray(
    psd,
    freqs,
    bands=DEFAULT_BANDS,
    relative=True,
):
    """Compute bandpowers in N-dimensional PSD.

    This is a NumPy-only implementation of the :py:func:`yasa.bandpower_from_psd` function,
    which supports 1-D arrays of shape (n_freqs), or N-dimensional arays (e.g. 2-D (n_chan,
    n_freqs) or 3-D (n_chan, n_epochs, n_freqs))

    .. versionadded:: 0.2.0

    Parameters
    ----------
    psd : :py:class:`numpy.ndarray`
        Power spectral density of data, in uV^2/Hz. Must be a N-D array of shape (..., n_freqs).
        See :py:func:`scipy.signal.welch` for more details.
    freqs : :py:class:`numpy.ndarray`
        Array of frequencies. Must be a 1-D array of shape (n_freqs,)
    bands : list of tuples
        List of frequency bands of interests. Each tuple must contain the lower and upper
        frequencies, as well as the band name (e.g. (0.5, 4, 'Delta')).
    relative : boolean
        If True, bandpower is divided by the total power between the min and
        max frequencies defined in ``band`` (default 0.5 to 40 Hz).

    Returns
    -------
    bandpowers : :py:class:`numpy.ndarray`
        Bandpower array of shape *(n_bands, ...)*.
    """
    # Type checks
    assert isinstance(bands, (list, tuple)), "bands must be a list of tuple(s)"
    assert isinstance(relative, bool), "relative must be a boolean"

    # Safety checks
    freqs = np.asarray(freqs)
    psd = np.asarray(psd)
    assert freqs.ndim == 1, "freqs must be a 1-D array of shape (n_freqs,)"
    assert psd.shape[-1] == freqs.shape[-1], "n_freqs must be last axis of psd"

    bp, total_power, _ = _bandpower_psd(psd, freqs, bands)
    if relative:
        bp /= total_power
    return bp


@_restore_log_level
def irasa(
    data,
    *,
    sf=None,
    ch_names=None,
    hypno=None,
    include=(2, 3),
    band=(1, 30),
    hset=_DEFAULT_HSET,
    return_fit=True,
    win_sec=4,
    welch_kwargs=None,
    verbose=False,
    kwargs_welch=None,
):
    r"""
    Separate the aperiodic (= fractal, or 1/f) and oscillatory component
    of the power spectra of EEG data using the IRASA method.

    .. versionadded:: 0.1.7

    .. versionchanged:: 0.8.0
        All the parameters except ``data`` are now keyword-only, e.g. ``irasa(data, sf=sf)``.

    Parameters
    ----------
    data : :py:class:`numpy.ndarray` or :py:class:`mne.io.BaseRaw`
        1D or 2D EEG data. If ``data`` is *array_like*, unit must be uV.
        If ``data`` is a :py:class:`~mne.io.BaseRaw` instance, ``data``, ``sf``, and
        ``ch_names`` will be automatically extracted, and ``data`` will be automatically
        converted from Volts (MNE) to micro-Volts (YASA).
    sf : float
        The sampling frequency of data AND the hypnogram if ``data`` is *array_like*.
        Should be omitted if ``data`` is a :py:class:~`mne.io.BaseRaw`.
    ch_names : list
        List of channel names, e.g. ['Cz', 'F3', 'F4', ...], if ``data`` is *array_like*.
        If None, channels will be labelled ['CHAN000', 'CHAN001', ...].
        Should be omitted if ``data`` is a :py:class:`~mne.io.BaseRaw`.
    hypno : array_like or :py:class:`yasa.Hypnogram`
        Sleep stage (hypnogram). If the hypnogram is loaded, IRASA is applied separately to the
        data of each sleep stage defined in ``include``.

        Can be an upsampled integer array (same number of samples as ``data``) or a
        :py:class:`yasa.Hypnogram` instance (automatically upsampled). To manually upsample an
        integer array, use :py:meth:`yasa.Hypnogram.upsample_to_data`.

        .. note::
            When passing an integer array, hypnogram values follow this mapping:

            - -2 = Unscored
            - -1 = Artefact / Movement
            - 0 = Wake
            - 1 = N1 sleep
            - 2 = N2 sleep
            - 3 = N3 sleep
            - 4 = REM sleep

        .. versionadded:: 0.8.0
    include : tuple, list or int or str
        Values in ``hypno`` that will be included in the mask. The default is (2, 3), meaning that
        IRASA is sequentially applied to N2 and N3 sleep. This has no effect when ``hypno`` is
        None. Stages that are too short for the Welch window and resampling factors are skipped
        with a warning.

        When ``hypno`` is a :py:class:`yasa.Hypnogram`, string labels can be used instead of
        integers (e.g. ``["N2", "N3"]``).

        .. versionadded:: 0.8.0
    band : tuple or None
        Broad band frequency range.
        Default is 1 to 30 Hz.
    hset : tuple, list or :py:class:`numpy.ndarray`
        Resampling factors used in IRASA calculation. Default is to use a range
        of values from 1.1 to 1.9 with an increment of 0.05.
    return_fit : boolean
        If True (default), fit an exponential function to the aperiodic PSD
        and return the fit parameters (intercept, slope) and :math:`R^2` of
        the fit.

        The aperiodic signal, :math:`L`, is modeled using an exponential
        function in semilog-power space (linear frequencies and log PSD) as:

        .. math:: L = a + \text{log}(F^b)

        where :math:`a` is the intercept, :math:`b` is the slope, and
        :math:`F` the vector of input frequencies.
    win_sec : int or float
        The length of the sliding window, in seconds, used for the Welch PSD
        calculation. Ideally, this should be at least two times the inverse of
        the lower frequency of interest (e.g. for a lower frequency of interest
        of 0.5 Hz, the window length should be at least 2 * 1 / 0.5 =
        4 seconds).
    welch_kwargs : dict or None
        Optional keywords arguments that are passed to the :py:func:`scipy.signal.welch` function.
        Default is ``{"average": "median", "window": "hamming"}``.

        .. versionadded:: 0.8.0
            Replaces ``kwargs_welch``.
    verbose : bool or str
        Verbose level. Default (False) will only print warning and error
        messages. The logging levels are 'debug', 'info', 'warning', 'error',
        and 'critical'. For most users the choice is between 'info'
        (or ``verbose=True``) and warning (``verbose=False``).

        .. versionchanged:: 0.8.0
            The default is now False, as documented. Previously, the default was True, but the
            info messages were not shown because they were sent to the root logger.
    kwargs_welch : dict or None
        .. deprecated:: 0.8.0
            Use ``welch_kwargs`` instead. This argument will be removed in v0.9.

    Returns
    -------
    freqs : :py:class:`numpy.ndarray`
        Frequency vector.
    psd_aperiodic : :py:class:`numpy.ndarray` or dict
        The fractal (= aperiodic) component of the PSD. If ``hypno`` is specified, this is a
        dictionary with one array per sleep stage, e.g. ``psd_aperiodic["N2"]``.
    psd_oscillatory : :py:class:`numpy.ndarray` or dict
        The oscillatory (= periodic) component of the PSD. If ``hypno`` is specified, this is a
        dictionary with one array per sleep stage.
    fit_params : :py:class:`pandas.DataFrame` (optional)
        Dataframe of fit parameters. Only if ``return_fit=True``. If ``hypno`` is specified,
        the dataframe has one row per sleep stage and channel, and an additional ``Stage``
        column.

    Notes
    -----
    The Irregular-Resampling Auto-Spectral Analysis (IRASA) method is
    described in Wen & Liu (2016). In a nutshell, the goal is to separate the
    fractal and oscillatory components in the power spectrum of EEG signals.

    The steps are:

    1. Compute the original power spectral density (PSD) using Welch's method.
    2. Resample the EEG data by multiple non-integer factors and their
       reciprocals (:math:`h` and :math:`1/h`).
    3. For every pair of resampled signals, calculate the PSD and take the
       geometric mean of both. In the resulting PSD, the power associated with
       the oscillatory component is redistributed away from its original
       (fundamental and harmonic) frequencies by a frequency offset that varies
       with the resampling factor, whereas the power solely attributed to the
       fractal component remains the same power-law statistical distribution
       independent of the resampling factor.
    4. It follows that taking the median of the PSD of the variously
       resampled signals can extract the power spectrum of the fractal
       component, and the difference between the original power spectrum and
       the extracted fractal spectrum offers an approximate estimate of the
       power spectrum of the oscillatory component.

    Note that an estimate of the original PSD can be calculated by simply
    adding ``psd = psd_aperiodic + psd_oscillatory``.

    If ``hypno`` is specified, the samples of each sleep stage are concatenated before applying
    IRASA, in the same way as in :py:func:`yasa.bandpower`.

    For an example of how to use this function, please refer to
    https://github.com/raphaelvallat/yasa/blob/master/notebooks/09_IRASA.ipynb

    For an article discussing the challenges of using IRASA (or fooof) see [5].

    References
    ----------
    [1] Wen, H., & Liu, Z. (2016). Separating Fractal and Oscillatory
        Components in the Power Spectrum of Neurophysiological Signal.
        Brain Topography, 29(1), 13–26.
        https://doi.org/10.1007/s10548-015-0448-0

    [2] https://github.com/fieldtrip/fieldtrip/blob/master/specest/

    [3] https://github.com/fooof-tools/fooof

    [4] https://www.biorxiv.org/content/10.1101/299859v1

    [5] https://doi.org/10.1101/2021.10.15.464483

    Examples
    --------
    IRASA applied separately to each sleep stage:

    .. code-block:: python

        >>> import yasa
        >>> hyp = yasa.Hypnogram.from_integers(hypno_30s, freq="30s")  # doctest: +SKIP
        >>> freqs, psd_ap, psd_osc, fit = yasa.irasa(  # doctest: +SKIP
        ...     raw, hypno=hyp, include=["N2", "N3", "REM"]
        ... )
        >>> psd_ap["N3"]  # Aperiodic PSD in N3 sleep, shape (n_chan, n_freqs)  # doctest: +SKIP
    """
    data, sf, ch_names, raw = _check_data(data, sf, ch_names)
    nchan, npts = data.shape
    assert nchan < npts, "Data must be of shape (nchan, n_samples)."
    if raw is not None:
        hp = raw.info["highpass"]  # Extract highpass filter
        lp = raw.info["lowpass"]  # Extract lowpass filter
    else:
        hp = 0  # Highpass filter unknown -> set to 0 Hz
        lp = sf / 2  # Lowpass filter unknown -> set to Nyquist

    # Check the other arguments
    welch_kwargs = _check_welch_kwargs(welch_kwargs, kwargs_welch, stacklevel=4)
    hset = np.asarray(hset)
    assert hset.ndim == 1, "hset must be 1D."
    assert hset.size > 1, "2 or more resampling fators are required."
    hset = np.round(hset, 4)  # avoid float precision error with np.arange.
    band = sorted(band)
    assert band[0] > 0, "first element of band must be > 0."
    assert band[1] < (sf / 2), "second element of band must be < (sf / 2)."
    win = int(win_sec * sf)  # nperseg

    # Inform about maximum resampled fitting range
    h_max = np.max(hset)
    band_evaluated = (band[0] / h_max, band[1] * h_max)
    freq_Nyq = sf / 2  # Nyquist frequency
    freq_Nyq_res = freq_Nyq / h_max  # minimum resampled Nyquist frequency
    logger.info(f"Fitting range: {band[0]:.2f}Hz-{band[1]:.2f}Hz")
    logger.info(f"Evaluated frequency range: {band_evaluated[0]:.2f}Hz-{band_evaluated[1]:.2f}Hz")
    if band_evaluated[0] < hp:
        logger.warning(
            "The evaluated frequency range starts below the "
            f"highpass filter ({hp:.2f}Hz). Increase the lower band"
            f" ({band[0]:.2f}Hz) or decrease the maximum value of "
            f"the hset ({h_max:.2f})."
        )
    if band_evaluated[1] > lp and lp < freq_Nyq_res:
        logger.warning(
            "The evaluated frequency range ends after the "
            f"lowpass filter ({lp:.2f}Hz). Decrease the upper band"
            f" ({band[1]:.2f}Hz) or decrease the maximum value of "
            f"the hset ({h_max:.2f})."
        )
    if band_evaluated[1] > freq_Nyq_res:
        logger.warning(
            "The evaluated frequency range ends after the "
            "resampled Nyquist frequency "
            f"({freq_Nyq_res:.2f}Hz). Decrease the upper band "
            f"({band[1]:.2f}Hz) or decrease the maximum value "
            f"of the hset ({h_max:.2f})."
        )

    # The downsampled signal must be at least as long as the Welch window. Otherwise, Welch
    # silently shortens the window and the PSDs of the resampled signals have different shapes.
    # resample_poly(data, down, up) returns ceil(n_samples * down / up) samples.
    rat_max = fractions.Fraction(str(h_max))

    def is_too_short(n_samples):
        return -(-n_samples * rat_max.denominator // rat_max.numerator) < win

    if hypno is None:
        if is_too_short(npts):
            raise ValueError(
                f"Data is too short for IRASA: at least win_sec * max(hset) = "
                f"{win_sec * h_max:.2f} seconds are required. Use a shorter win_sec or a lower "
                f"max(hset)."
            )
        freqs, psd_aperiodic, psd_osc = _irasa(data, sf, hset, win, band, welch_kwargs)
        if not return_fit:
            return freqs, psd_aperiodic, psd_osc
        return freqs, psd_aperiodic, psd_osc, _irasa_fit(freqs, psd_aperiodic, psd_osc, ch_names)

    # Per each sleep stage defined in ``include``. As in bandpower, the original Raw is passed so
    # that a Hypnogram with a start time is aligned with the recording using absolute timestamps.
    hypno, include, int_to_str = _check_hypno_include(
        hypno, include, raw if raw is not None else data, sf, verbose=verbose
    )
    psd_aperiodic, psd_osc, fit_params = {}, {}, []
    for stage in include:
        is_stage = hypno == stage
        if not is_stage.any():
            continue
        label = int_to_str.get(stage, stage)
        if is_too_short(is_stage.sum()):
            logger.warning(
                f"Stage {label} is shorter than win_sec * max(hset) = {win_sec * h_max:.2f} "
                "seconds. Skipping stage."
            )
            continue
        freqs, psd_aperiodic[label], psd_osc[label] = _irasa(
            data[:, is_stage], sf, hset, win, band, welch_kwargs
        )
        if return_fit:
            fit_stage = _irasa_fit(freqs, psd_aperiodic[label], psd_osc[label], ch_names)
            fit_stage.insert(0, "Stage", label)
            fit_params.append(fit_stage)
    if not psd_aperiodic:
        raise ValueError(
            "All the stages in `include` are shorter than win_sec * max(hset) = "
            f"{win_sec * h_max:.2f} seconds."
        )
    if not return_fit:
        return freqs, psd_aperiodic, psd_osc
    return freqs, psd_aperiodic, psd_osc, pd.concat(fit_params, ignore_index=True)


def _irasa(data, sf, hset, win, band, welch_kwargs):
    """Apply IRASA to a 2D array of shape (n_chan, n_samples) and crop the PSDs to ``band``."""
    # Calculate the original PSD over the whole data
    freqs, psd = signal.welch(data, sf, nperseg=win, **welch_kwargs)

    # Start the IRASA procedure
    psds = np.zeros((len(hset), *psd.shape))

    for i, h in enumerate(hset):
        # Get the upsampling/downsampling (h, 1/h) factors as integer
        rat = fractions.Fraction(str(h))
        up, down = rat.numerator, rat.denominator
        # Much faster than FFT-based resampling
        data_up = signal.resample_poly(data, up, down, axis=-1)
        data_down = signal.resample_poly(data, down, up, axis=-1)
        # Calculate the PSD using same params as original
        _, psd_up = signal.welch(data_up, h * sf, nperseg=win, **welch_kwargs)
        _, psd_dw = signal.welch(data_down, sf / h, nperseg=win, **welch_kwargs)
        # Geometric mean of h and 1/h
        psds[i] = np.sqrt(psd_up * psd_dw)

    # Now we take the median PSD of all the resampling factors, which gives
    # a good estimate of the aperiodic component of the PSD.
    psd_aperiodic = np.median(psds, axis=0)

    # We can now calculate the oscillations (= periodic) component.
    psd_osc = psd - psd_aperiodic

    # Let's crop to the frequencies defined in band
    in_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    return freqs[in_band], psd_aperiodic[..., in_band], psd_osc[..., in_band]


def _irasa_fit(freqs, psd_aperiodic, psd_osc, ch_names):
    """Fit an exponential function to the aperiodic PSD of each channel, in semilog space."""
    intercepts, slopes, r_squared = [], [], []

    def func(t, a, b):
        # a + log(t^b). See https://github.com/fooof-tools/fooof
        return a + b * np.log(t)

    for y in np.atleast_2d(psd_aperiodic):
        y_log = np.log(y)
        # Note that here we define bounds for the slope but not for the
        # intercept.
        popt, _ = curve_fit(func, freqs, y_log, p0=(2, -1), bounds=((-np.inf, -10), (np.inf, 2)))
        intercepts.append(popt[0])
        slopes.append(popt[1])
        # Calculate R^2: https://stackoverflow.com/q/19189362/10581531
        residuals = y_log - func(freqs, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_log - np.mean(y_log)) ** 2)
        r_squared.append(1 - (ss_res / ss_tot))

    # Create fit parameters dataframe
    fit_params = {
        "Chan": ch_names,
        "Intercept": intercepts,
        "Slope": slopes,
        "R^2": r_squared,
        "std(osc)": np.std(psd_osc, axis=-1, ddof=1),
    }
    return pd.DataFrame(fit_params)


def stft_power(data, sf, window=2, step=0.2, band=(1, 30), interp=True, norm=False):
    """Compute the pointwise power via STFT and interpolation.

    Parameters
    ----------
    data : array_like
        Single-channel data.
    sf : float
        Sampling frequency of the data.
    window : int
        Window size in seconds for STFT. 2 or 4 seconds are usually a good default.
        Higher values = higher frequency resolution = lower time resolution.
    step : int
        Step in seconds for the STFT.
        A step of 0.2 second (200 ms) is usually a good default.

        * If ``step`` == 0, overlap at every sample (slowest)
        * If ``step`` == window, no overlap (fastest)

        Lower values = higher time resolution = slower computation.
    band : tuple or None
        Broad band frequency range. Default is 1 to 30 Hz.
    interp : boolean
        If True, a cubic interpolation is performed to ensure that the output is the same size as
        the input (= pointwise power).
    norm : bool
        If True, return bandwise normalized band power, i.e. for each time point, the sum of power
        in all the frequency bins equals 1.

    Returns
    -------
    f : :py:class:`numpy.ndarray`
        Frequency vector
    t : :py:class:`numpy.ndarray`
        Time vector
    Sxx : :py:class:`numpy.ndarray`
        Power in the specified frequency bins of shape (f, t)

    Notes
    -----
    2D Interpolation is done using :py:class:`scipy.interpolate.RectBivariateSpline`
    which is much faster than :py:class:`scipy.interpolate.interp2d` for a rectangular grid.
    The default is to use a bivariate spline with 3 degrees.
    """
    # Safety check
    data = np.asarray(data)
    assert step <= window
    step = 1 / sf if step == 0 else step

    # Define STFT parameters
    nperseg = int(window * sf)
    noverlap = int(nperseg - (step * sf))

    # Compute STFT
    f, t, Sxx = signal.stft(
        data, sf, nperseg=nperseg, noverlap=noverlap, detrend=False, padded=True
    )

    # Let's keep only the frequency of interest
    if band is not None:
        idx_band = np.logical_and(f >= band[0], f <= band[1])
        f = f[idx_band]
        Sxx = Sxx[idx_band, :]

    # Compute power (= squared magnitude) and interpolate
    Sxx = Sxx.real**2 + Sxx.imag**2
    if interp:
        assert f.size >= 2, "At least 2 frequency bins are required for the interpolation."
        # The spline degree must be lower than the number of frequency bins
        func = RectBivariateSpline(f, t, Sxx, kx=min(3, f.size - 1))
        t = np.arange(data.size) / sf
        Sxx = func(f, t)

    # Normalize
    if norm:
        Sxx /= Sxx.sum(0, keepdims=True)
    return f, t, Sxx
