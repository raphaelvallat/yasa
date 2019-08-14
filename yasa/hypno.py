"""
This file contains several helper functions to manipulate sleep staging
(hypnogram) data. The default hypnogram format in YASA is a one dimensional
integer array where:

* -1  = Artefact / Undefined / Movement
* 0   = Wake
* 1   = N1 sleep
* 2   = N2 sleep
* 3   = N3 sleep
* 4   = REM sleep

For more details, please refer to the following references:

- Iber, C. (2007). The AASM manual for the scoring of sleep and
associated events: rules, terminology and technical specifications.
American Academy of Sleep Medicine.

- Silber, M. H., Ancoli-Israel, S., Bonnet, M. H., Chokroverty, S.,
Grigg-Damberger, M. M., Hirshkowitz, M., … Iber, C. (2007). The visual scoring
of sleep in adults. Journal of Clinical Sleep Medicine: JCSM: Official
Publication of the American Academy of Sleep Medicine, 3(2), 121–131.

- Combrisson, E., Vallat, R., Eichenlaub, J.-B., O’Reilly, C., Lajnef, T.,
Guillot, A., … Jerbi, K. (2017). Sleep: An Open-Source Python Software for
Visualization, Analysis, and Staging of Sleep Data. Frontiers in
Neuroinformatics, 11, 60. https://doi.org/10.3389/fninf.2017.00060
"""
import mne
import warnings
import numpy as np
import pandas as pd

__all__ = ['hypno_str_to_int', 'hypno_int_to_str', 'hypno_upsample_to_sf',
           'hypno_upsample_to_data']


#############################################################################
# STR <--> INT CONVERSION
#############################################################################

def hypno_str_to_int(hypno, mapping_dict={'w': 0, 'wake': 0, 'n1': 1, 's1': 1,
                                          'n2': 2, 's2': 2, 'n3': 3, 's3': 3,
                                          'r': 4, 'rem': 4, 'art': -1}):
    """Convert a string hypnogram array to integer.

    ['W', 'N2', 'N2', 'N3', 'R'] ==> [0, 2, 2, 3, 4]

    Parameters
    ----------
    hypno : array_like
        The sleep staging (hypnogram) 1D array.
    mapping_dict : dict
        The mapping dictionnary, in lowercase. Note that this function is
        essentially a wrapper around `pandas.Series.map`.

    Returns
    --------
    hypno : array_like
        The corresponding integer hypnogram.
    """
    assert isinstance(hypno, (list, np.ndarray, pd.Series)), 'Not an array.'
    hypno = pd.Series(np.asarray(hypno, dtype=str))
    assert not hypno.str.isnumeric().any(), 'Hypno contains numeric values.'
    return hypno.str.lower().map(mapping_dict).values


def hypno_int_to_str(hypno, mapping_dict={0: 'W', 1: 'N1', 2: 'N2', 3: 'N3',
                                          4: 'R', -1: 'Art'}):
    """Convert an integer hypnogram array to a string array.

    [0, 2, 2, 3, 4] ==> ['W', 'N2', 'N2', 'N3', 'R']

    Parameters
    ----------
    hypno : array_like
      The sleep staging (hypnogram) 1D array.
    mapping_dict : dict
      The mapping dictionnary. Note that this function is
      essentially a wrapper around `pandas.Series.map`.

    Returns
    --------
    hypno : array_like
      The corresponding integer hypnogram.
    """
    assert isinstance(hypno, (list, np.ndarray, pd.Series)), 'Not an array.'
    hypno = pd.Series(np.asarray(hypno, dtype=int))
    return hypno.map(mapping_dict).values

#############################################################################
# UPSAMPLING
#############################################################################


def hypno_upsample_to_sf(hypno, sf_hypno, sf_data):
    """Upsample the hypnogram to a given sampling frequency.

    Parameters
    ----------
    hypno : array_like
        The sleep staging (hypnogram) 1D array.
    sf_hypno : float
        The current sampling frequency of the hypnogram, in Hz, e.g.
        1/30 = 1 value per each 30 seconds of EEG data,
        1 = 1 value per second of EEG data
    sf_data : float
        The desired sampling frequency of the hypnogram, in Hz
        (e.g. 100 Hz, 256 Hz, ...)

    Returns
    -------
    hypno : array_like
        The hypnogram, upsampled to ``sf_data``.
    """
    repeats = sf_data / sf_hypno
    assert sf_hypno <= sf_data, 'sf_hypno must be less than sf_data.'
    assert repeats.is_integer(), 'sf_hypno / sf_data must be a whole number.'
    assert isinstance(hypno, (list, np.ndarray, pd.Series))
    return np.repeat(np.asarray(hypno), repeats)


def hypno_fit_to_data(hypno, data, sf=None):
    """Crop or pad the hypnogram to fit the length of data.

    Hypnogram and data MUST have the SAME sampling frequency.

    This is an internal function.

    Parameters
    ----------
    hypno : array_like
        The sleep staging (hypnogram) 1D array.
    data : np.array_like or mne.io.Raw
        1D or 2D EEG data. Can also be a MNE Raw object, in which case data and
        sf will be automatically extracted.
    sf : float, optional
        The sampling frequency of data AND the hypnogram.

    Returns
    -------
    hypno : array_like
        Hypnogram, with the same number of samples as data.
    """
    # Check if data is an MNE raw object
    if isinstance(data, mne.io.BaseRaw):
        sf = data.info['sfreq']
        data = data.times  # 1D array and does not require to preload data
    data = np.asarray(data)
    hypno = np.asarray(hypno)
    assert hypno.ndim == 1, 'Hypno must be 1D.'
    npts_hyp = hypno.size
    npts_data = max(data.shape)  # Support for 2D data
    if npts_hyp < npts_data:
        # Hypnogram is shorter than data
        npts_diff = npts_data - npts_hyp
        if sf is not None:
            dur_diff = npts_diff / sf
            warnings.warn('Hypnogram is SHORTER then data by %.2f seconds. '
                          'Padding hypnogram (with last value) to match '
                          'data.size.' % dur_diff)
        else:
            warnings.warn('Hypnogram is SHORTER then data by %i samples. '
                          'Padding hypnogram (with last value) to match '
                          'data.size.' % npts_diff)
        hypno = np.pad(hypno, (0, npts_diff), mode='edge')
    elif npts_hyp > npts_data:
        # Hypnogram is longer than data
        npts_diff = npts_hyp - npts_data
        if sf is not None:
            dur_diff = npts_diff / sf
            warnings.warn('Hypnogram is LONGER then data by %.2f seconds. '
                          'Cropping hypnogram to match data.size.' % dur_diff)
        else:
            warnings.warn('Hypnogram is LONGER then data by %i samples. '
                          'Cropping hypnogram to match data.size.' % npts_diff)
        hypno = hypno[0:npts_data]
    return hypno


def hypno_upsample_to_data(hypno, sf_hypno, data, sf_data=None):
    """Upsample an hypnogram to a given sampling frequency and fit the
    resulting hypnogram to corresponding EEG data, such that the hypnogram
    and EEG data have the exact same number of samples.

    Parameters
    ----------
    hypno : array_like
        The sleep staging (hypnogram) 1D array.
    sf_hypno : float
        The current sampling frequency of the hypnogram, in Hz, e.g.
        1/30 = 1 value per each 30 seconds of EEG data,
        1 = 1 value per second of EEG data
    data : array_like or mne.io.Raw
        1D or 2D EEG data. Can also be a MNE Raw object, in which case
        the EEG ``data`` and ``sf`` will be automatically extracted.
    sf_data : float
        The sampling frequency of ``data``, in Hz (e.g. 100 Hz, 256 Hz, ...).
        Can be omitted if ``data`` is a mne.io.Raw object.

    Returns
    -------
    hypno : array_like
        The hypnogram, upsampled to ``sf_data`` and cropped/padded to
        ``data.size``.
    """
    if isinstance(data, mne.io.BaseRaw):
        sf_data = data.info['sfreq']
        data = data.times
    hypno_up = hypno_upsample_to_sf(hypno=hypno, sf_hypno=sf_hypno,
                                    sf_data=sf_data)
    return hypno_fit_to_data(hypno=hypno_up, data=data, sf=sf_data)
