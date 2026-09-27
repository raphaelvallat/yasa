"""
This file contains several helper functions to calculate sleep statistics from
a one-dimensional sleep staging vector (hypnogram).
"""

import warnings

import numpy as np

from .hypno import Hypnogram, _transition_matrix

__all__ = ["transition_matrix", "sleep_statistics"]


#############################################################################
# TRANSITION MATRIX
#############################################################################


def transition_matrix(hypno):
    """Create a state-transition matrix from an hypnogram.

    .. deprecated:: 0.8.0
        Use :py:meth:`yasa.Hypnogram.transition_matrix` instead. This function will be removed
        in v0.9.

    .. versionadded:: 0.1.9

    Parameters
    ----------
    hypno : array_like or :py:class:`yasa.Hypnogram`
        Hypnogram. Can be either:

        * An integer array (e.g. ``[0, 2, 2, 1, 1, 1, ...]``). The sampling
          frequency must be the original one, i.e. 1 value per 30 seconds if
          the staging was done in 30-second epochs. Using an upsampled
          hypnogram will result in an incorrect transition matrix. The output
          DataFrames will use integer index and column labels.
        * A :py:class:`yasa.Hypnogram` instance. The output DataFrames will
          use string stage labels (e.g. ``"WAKE"``, ``"N1"``). This is
          equivalent to calling :py:meth:`yasa.Hypnogram.transition_matrix`
          directly.

        For best results, the hypnogram should be cropped to either the time
        in bed (TIB) or the sleep period time (SPT), without any artefact or
        unscored epochs.

    Returns
    -------
    counts : :py:class:`pandas.DataFrame`
        Counts transition matrix (number of transitions from stage A to
        stage B). The pre-transition states are the rows and the
        post-transition states are the columns.
    probs : :py:class:`pandas.DataFrame`
        Conditional probability transition matrix, i.e.
        given that current state is A, what is the probability that
        the next state is B.
        ``probs`` is a `right stochastic matrix
        <https://en.wikipedia.org/wiki/Stochastic_matrix>`_,
        i.e. each row sums to 1.

    Examples
    --------
    Integer array input — output uses integer labels:

    >>> import numpy as np
    >>> from yasa import transition_matrix
    >>> a = [0, 0, 0, 1, 1, 0, 1, 2, 2, 3, 3, 2, 3, 3, 0, 2, 2, 1, 2, 2, 3, 3]
    >>> counts, probs = transition_matrix(a)
    >>> counts
    To Stage  0  1  2  3
    From Stage
    0         2  2  1  0
    1         1  1  2  0
    2         0  1  3  3
    3         1  0  1  3

    >>> probs.round(2)
    To Stage     0     1     2     3
    From Stage
    0         0.40  0.40  0.20  0.00
    1         0.25  0.25  0.50  0.00
    2         0.00  0.14  0.43  0.43
    3         0.20  0.00  0.20  0.60

    Several metrics of sleep fragmentation can be calculated from the
    probability matrix. For example, the stability of sleep stages can be
    calculated by taking the average of the diagonal values (excluding Wake
    and N1 sleep):

    >>> float(np.diag(probs.loc[2:, 2:]).mean().round(3))
    0.514

    :py:class:`yasa.Hypnogram` input — output uses string stage labels:

    >>> from yasa import Hypnogram, transition_matrix
    >>> hyp = Hypnogram(["W", "N1", "N2", "N3", "N2", "REM", "W"])
    >>> counts, probs = transition_matrix(hyp)
    >>> counts
    To Stage    WAKE  N1  N2  N3  REM
    From Stage
    WAKE           0   1   0   0    0
    N1             0   0   1   0    0
    N2             0   0   0   1    1
    N3             0   0   1   0    0
    REM            1   0   0   0    0

    Finally, we can plot the transition matrix using :py:func:`seaborn.heatmap`

    .. plot::

        >>> import numpy as np
        >>> import seaborn as sns
        >>> import matplotlib.pyplot as plt
        >>> from yasa import transition_matrix
        >>> # Calculate probability matrix
        >>> a = [1, 1, 1, 0, 0, 2, 2, 0, 2, 0, 1, 1, 0, 0]
        >>> _, probs = transition_matrix(a)
        >>> # Start the plot
        >>> grid_kws = {"height_ratios": (0.9, 0.05), "hspace": 0.1}
        >>> f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws, figsize=(5, 5))
        >>> ax = sns.heatmap(
        ...     probs,
        ...     ax=ax,
        ...     square=False,
        ...     vmin=0,
        ...     vmax=1,
        ...     cbar=True,
        ...     cbar_ax=cbar_ax,
        ...     cmap="YlOrRd",
        ...     annot=True,
        ...     fmt=".2f",
        ...     cbar_kws={
        ...         "orientation": "horizontal",
        ...         "fraction": 0.1,
        ...         "label": "Transition probability",
        ...     },
        ... )
        >>> ax.xaxis.tick_top()
        >>> ax.xaxis.set_label_position("top")
        >>> _ = ax.set(xlabel="To sleep stage", ylabel="From sleep stage")
    """
    warnings.warn(
        "The `yasa.transition_matrix` function is deprecated and will be removed in v0.9. "
        "Please use the `yasa.Hypnogram.transition_matrix` method instead.",
        FutureWarning,
        stacklevel=2,
    )
    if isinstance(hypno, Hypnogram):
        return hypno.transition_matrix()
    return _transition_matrix(hypno)


#############################################################################
# SLEEP STATISTICS
#############################################################################


def sleep_statistics(hypno, sf_hyp):
    """Compute standard sleep statistics from an hypnogram.

    .. deprecated:: 0.8.0
        Use :py:meth:`yasa.Hypnogram.sleep_statistics` instead. This function will be removed in
        v0.9.

    .. versionadded:: 0.1.9

    Parameters
    ----------
    hypno : array_like
        Hypnogram, assumed to be already cropped to time in bed (TIB,
        also referred to as Total Recording Time,
        i.e. "lights out" to "lights on").

        .. note::
            Hypnogram values are integers with the following mapping:

            - -2 = Unscored
            - -1 = Artefact / Movement
            - 0 = Wake
            - 1 = N1 sleep
            - 2 = N2 sleep
            - 3 = N3 sleep
            - 4 = REM sleep
    sf_hyp : float
        The sampling frequency of the hypnogram. Should be 1/30 if there is one
        value per 30-seconds, 1/20 if there is one value per 20-seconds,
        1 if there is one value per second, and so on.

    Returns
    -------
    stats : dict
        Sleep statistics (expressed in minutes)

    Notes
    -----
    All values except SE, SME and percentages of each stage are expressed in
    minutes. YASA follows the AASM guidelines to calculate these parameters:

    * Time in Bed (TIB): total duration of the hypnogram.
    * Sleep Period Time (SPT): duration from first to last period of sleep.
    * Wake After Sleep Onset (WASO): duration of wake periods within SPT.
    * Total Sleep Time (TST): total duration of N1 + N2 + N3 + REM sleep in SPT.
    * Sleep Efficiency (SE): TST / TIB * 100 (%).
    * Sleep Maintenance Efficiency (SME): TST / SPT * 100 (%).
    * W, N1, N2, N3 and REM: sleep stages duration. NREM = N1 + N2 + N3.
    * % (W, ... REM): sleep stages duration expressed in percentages of TST.
    * Latencies: latencies of sleep stages from the beginning of the record.
    * Sleep Onset Latency (SOL): Latency to first epoch of any sleep.

    .. warning::
        Since YASA 0.5.0, Artefact and Unscored epochs are now excluded from the calculation of the
        total sleep time (TST). Previously, YASA calculated TST as SPT - WASO, thus including
        Art and Uns. TST is now calculated as the sum of all REM and NREM sleep in SPT.

    .. warning::
        The definition of REM latency in the AASM scoring manual differs from the REM latency
        reported here. The former uses the time from first epoch of sleep, while YASA uses the
        time from the beginning of the recording. The AASM definition of the REM latency can be
        found with `Lat_REM - SOL`.

    References
    ----------
    * Iber, C. (2007). The AASM manual for the scoring of sleep and
      associated events: rules, terminology and technical specifications.
      American Academy of Sleep Medicine.

    * Silber, M. H., Ancoli-Israel, S., Bonnet, M. H., Chokroverty, S.,
      Grigg-Damberger, M. M., Hirshkowitz, M., Kapen, S., Keenan, S. A.,
      Kryger, M. H., Penzel, T., Pressman, M. R., & Iber, C. (2007).
      `The visual scoring of sleep in adults
      <https://www.ncbi.nlm.nih.gov/pubmed/17557422>`_. Journal of Clinical
      Sleep Medicine: JCSM: Official Publication of the American Academy of
      Sleep Medicine, 3(2), 121–131.

    Examples
    --------
    >>> from yasa import sleep_statistics
    >>> hypno = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 3, 3, 4, 4, 4, 4, 0, 0]
    >>> # Assuming that we have one-value per 30-second.
    >>> sleep_statistics(hypno, sf_hyp=1 / 30)
    {'TIB': 10.0,
     'SPT': 8.0,
     'WASO': 0.0,
     'TST': 8.0,
     'N1': 1.5,
     'N2': 2.0,
     'N3': 2.5,
     'REM': 2.0,
     'NREM': 6.0,
     'SOL': 1.0,
     'Lat_N1': 1.0,
     'Lat_N2': 2.5,
     'Lat_N3': 4.0,
     'Lat_REM': 7.0,
     '%N1': 18.75,
     '%N2': 25.0,
     '%N3': 31.25,
     '%REM': 25.0,
     '%NREM': 75.0,
     'SE': 80.0,
     'SME': 100.0}
    """
    warnings.warn(
        "The `yasa.sleep_statistics` function is deprecated and will be removed in v0.9. "
        "Please use the `yasa.Hypnogram.sleep_statistics` method instead.",
        FutureWarning,
        stacklevel=2,
    )
    stats = {}
    hypno = np.asarray(hypno)
    assert hypno.ndim == 1, "hypno must have only one dimension."
    assert hypno.size > 1, "hypno must have at least two elements."
    stages = {"N1": 1, "N2": 2, "N3": 3, "REM": 4}

    # TIB, first and last sleep
    stats["TIB"] = len(hypno)
    idx_sleep = np.flatnonzero(hypno > 0)
    has_sleep = idx_sleep.size > 0

    # Crop to SPT. Without any sleep, SPT and TST are 0 and WASO and SOL are undefined.
    hypno_s = hypno[idx_sleep[0] : (idx_sleep[-1] + 1)] if has_sleep else hypno[:0]
    stats["SPT"] = hypno_s.size
    stats["WASO"] = np.count_nonzero(hypno_s == 0) if has_sleep else np.nan
    # Before YASA v0.5.0, TST was calculated as SPT - WASO, meaning that Art
    # and Unscored epochs were included. TST is now restrained to sleep stages.
    stats["TST"] = np.count_nonzero(hypno_s > 0)

    # Duration of each sleep stages
    for st, val in stages.items():
        stats[st] = np.count_nonzero(hypno == val)
    stats["NREM"] = stats["N1"] + stats["N2"] + stats["N3"]

    # Sleep stage latencies -- only relevant if hypno is cropped to TIB
    stats["SOL"] = idx_sleep[0] if has_sleep else np.nan
    for st, val in stages.items():
        idx_st = np.flatnonzero(hypno == val)
        stats[f"Lat_{st}"] = idx_st[0] if idx_st.size else np.nan

    # Convert to minutes
    for key, value in stats.items():
        stats[key] = value / (60 * sf_hyp)

    # Percentage
    for st in [*stages, "NREM"]:
        stats[f"%{st}"] = 100 * stats[st] / stats["TST"] if has_sleep else np.nan
    stats["SE"] = 100 * stats["TST"] / stats["TIB"]
    stats["SME"] = 100 * stats["TST"] / stats["SPT"] if has_sleep else np.nan
    # Return built-in floats rather than NumPy scalars
    return {key: float(value) for key, value in stats.items()}
