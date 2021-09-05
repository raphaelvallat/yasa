"""
This file contains several helper functions to calculate sleep statistics from
a one-dimensional sleep staging vector (hypnogram).
"""
import numpy as np
import pandas as pd

__all__ = ['transition_matrix', 'sleep_statistics']


#############################################################################
# TRANSITION MATRIX
#############################################################################

def transition_matrix(hypno):
    """Create a state-transition matrix from an hypnogram.

    .. versionadded:: 0.1.9

    Parameters
    ----------
    hypno : array_like
        Hypnogram. The dtype of ``hypno`` must be integer
        (e.g. [0, 2, 2, 1, 1, 1, ...]). The sampling frequency must be the
        original one, i.e. 1 value per 30 seconds if the staging was done in
        30 seconds epochs. Using an upsampled hypnogram will result in an
        incorrect transition matrix.
        For best results, we recommend using an hypnogram cropped to
        either the time in bed (TIB) or the sleep period time (SPT), without
        any artefact / unscored epochs.

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
    >>> import numpy as np
    >>> from yasa import transition_matrix
    >>> a = [0, 0, 0, 1, 1, 0, 1, 2, 2, 3, 3, 2, 3, 3, 0, 2, 2, 1, 2, 2, 3, 3]
    >>> counts, probs = transition_matrix(a)
    >>> counts
           0  1  2  3
    Stage
    0      2  2  1  0
    1      1  1  2  0
    2      0  1  3  3
    3      1  0  1  3

    >>> probs.round(2)
              0     1     2     3
    Stage
    0      0.40  0.40  0.20  0.00
    1      0.25  0.25  0.50  0.00
    2      0.00  0.14  0.43  0.43
    3      0.20  0.00  0.20  0.60

    Several metrics of sleep fragmentation can be calculated from the
    probability matrix. For example, the stability of sleep stages can be
    calculated by taking the average of the diagonal values (excluding Wake
    and N1 sleep):

    >>> np.diag(probs.loc[2:, 2:]).mean().round(3)
    0.514

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
        >>> grid_kws = {"height_ratios": (.9, .05), "hspace": .1}
        >>> f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws,
        ...                                 figsize=(5, 5))
        >>> sns.heatmap(probs, ax=ax, square=False, vmin=0, vmax=1, cbar=True,
        ...             cbar_ax=cbar_ax, cmap='YlOrRd', annot=True, fmt='.2f',
        ...             cbar_kws={"orientation": "horizontal", "fraction": 0.1,
        ...                       "label": "Transition probability"})
        >>> ax.set_xlabel("To sleep stage")
        >>> ax.xaxis.tick_top()
        >>> ax.set_ylabel("From sleep stage")
        >>> ax.xaxis.set_label_position('top')
    """
    x = np.asarray(hypno, dtype=int)
    unique, inverse = np.unique(x, return_inverse=True)  # unique is sorted
    n = unique.size
    # Integer transition counts
    counts = np.zeros((n, n), dtype=int)
    np.add.at(counts, (inverse[:-1], inverse[1:]), 1)
    # Conditional probabilities
    probs = counts / counts.sum(axis=-1, keepdims=True)
    # Convert to a Pandas DataFrame
    counts = pd.DataFrame(counts, index=unique, columns=unique)
    probs = pd.DataFrame(probs, index=unique, columns=unique)
    counts.index.name = 'From Stage'
    probs.index.name = 'From Stage'
    counts.columns.name = 'To Stage'
    probs.columns.name = 'To Stage'
    return counts, probs

#############################################################################
# SLEEP STATISTICS
#############################################################################


def sleep_statistics(hypno, sf_hyp):
    """Compute standard sleep statistics from an hypnogram.

    .. versionadded:: 0.1.9

    Parameters
    ----------
    hypno : array_like
        Hypnogram, assumed to be already cropped to time in bed (TIB,
        also referred to as Total Recording Time,
        i.e. "lights out" to "lights on").

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
    >>> sleep_statistics(hypno, sf_hyp=1/30)
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
    stats = {}
    hypno = np.asarray(hypno)
    assert hypno.ndim == 1, 'hypno must have only one dimension.'
    assert hypno.size > 1, 'hypno must have at least two elements.'

    # TIB, first and last sleep
    stats['TIB'] = len(hypno)
    first_sleep = np.where(hypno > 0)[0][0]
    last_sleep = np.where(hypno > 0)[0][-1]

    # Crop to SPT
    hypno_s = hypno[first_sleep:(last_sleep + 1)]
    stats['SPT'] = hypno_s.size
    stats['WASO'] = hypno_s[hypno_s == 0].size
    # Before YASA v0.5.0, TST was calculated as SPT - WASO, meaning that Art
    # and Unscored epochs were included. TST is now restrained to sleep stages.
    stats['TST'] = hypno_s[hypno_s > 0].size

    # Duration of each sleep stages
    stats['N1'] = hypno[hypno == 1].size
    stats['N2'] = hypno[hypno == 2].size
    stats['N3'] = hypno[hypno == 3].size
    stats['REM'] = hypno[hypno == 4].size
    stats['NREM'] = stats['N1'] + stats['N2'] + stats['N3']

    # Sleep stage latencies -- only relevant if hypno is cropped to TIB
    stats['SOL'] = first_sleep
    stats['Lat_N1'] = np.where(hypno == 1)[0].min() if 1 in hypno else np.nan
    stats['Lat_N2'] = np.where(hypno == 2)[0].min() if 2 in hypno else np.nan
    stats['Lat_N3'] = np.where(hypno == 3)[0].min() if 3 in hypno else np.nan
    stats['Lat_REM'] = np.where(hypno == 4)[0].min() if 4 in hypno else np.nan

    # Convert to minutes
    for key, value in stats.items():
        stats[key] = value / (60 * sf_hyp)

    # Percentage
    stats['%N1'] = 100 * stats['N1'] / stats['TST']
    stats['%N2'] = 100 * stats['N2'] / stats['TST']
    stats['%N3'] = 100 * stats['N3'] / stats['TST']
    stats['%REM'] = 100 * stats['REM'] / stats['TST']
    stats['%NREM'] = 100 * stats['NREM'] / stats['TST']
    stats['SE'] = 100 * stats['TST'] / stats['TIB']
    stats['SME'] = 100 * stats['TST'] / stats['SPT']
    return stats
