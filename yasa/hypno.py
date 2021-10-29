"""
This file contains several helper functions to manipulate sleep staging
(hypnogram) data. The default hypnogram format in YASA is a one dimensional
integer array where:

* -2  = Unscored
* -1  = Artefact / Movement
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
import logging
import numpy as np
import pandas as pd
from .io import set_log_level

__all__ = ['hypno_str_to_int', 'hypno_int_to_str', 'hypno_upsample_to_sf',
           'hypno_upsample_to_data', 'load_profusion_hypno']


logger = logging.getLogger('yasa')


#############################################################################
# STR <--> INT CONVERSION
#############################################################################

def hypno_str_to_int(hypno, mapping_dict={'w': 0, 'wake': 0, 'n1': 1, 's1': 1,
                                          'n2': 2, 's2': 2, 'n3': 3, 's3': 3,
                                          's4': 3, 'r': 4, 'rem': 4, 'art': -1,
                                          'mt': -1, 'uns': -2, 'nd': -2}):
    """Convert a string hypnogram array to integer.

    ['W', 'N2', 'N2', 'N3', 'R'] ==> [0, 2, 2, 3, 4]

    .. versionadded:: 0.1.5

    Parameters
    ----------
    hypno : array_like
        The sleep staging (hypnogram) 1D array.
    mapping_dict : dict
        The mapping dictionnary, in lowercase. Note that this function is essentially a wrapper
        around :py:meth:`pandas.Series.map`.

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
                                          4: 'R', -1: 'Art', -2: 'Uns'}):
    """Convert an integer hypnogram array to a string array.

    [0, 2, 2, 3, 4] ==> ['W', 'N2', 'N2', 'N3', 'R']

    .. versionadded:: 0.1.5

    Parameters
    ----------
    hypno : array_like
        The sleep staging (hypnogram) 1D array.
    mapping_dict : dict
        The mapping dictionnary. Note that this function is essentially a wrapper around
        :py:meth:`pandas.Series.map`.

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

    .. versionadded:: 0.1.5

    Parameters
    ----------
    hypno : array_like
        The sleep staging (hypnogram) 1D array.
    sf_hypno : float
        The current sampling frequency of the hypnogram, in Hz, e.g.

        * 1/30 = 1 value per each 30 seconds of EEG data,
        * 1 = 1 value per second of EEG data
    sf_data : float
        The desired sampling frequency of the hypnogram, in Hz (e.g. 100 Hz, 256 Hz, ...)

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
        1D or 2D EEG data. Can also be a MNE Raw object, in which case data and sf will be
        automatically extracted.
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
            logger.warning('Hypnogram is SHORTER than data by %.2f seconds. '
                           'Padding hypnogram with last value to match data.size.' % dur_diff)
        else:
            logger.warning('Hypnogram is SHORTER than data by %i samples. '
                           'Padding hypnogram with last value to match data.size.' % npts_diff)
        hypno = np.pad(hypno, (0, npts_diff), mode='edge')
    elif npts_hyp > npts_data:
        # Hypnogram is longer than data
        npts_diff = npts_hyp - npts_data
        if sf is not None:
            dur_diff = npts_diff / sf
            logger.warning('Hypnogram is LONGER than data by %.2f seconds. '
                           'Cropping hypnogram to match data.size.' % dur_diff)
        else:
            logger.warning('Hypnogram is LONGER than data by %i samples. '
                           'Cropping hypnogram to match data.size.' % npts_diff)
        hypno = hypno[0:npts_data]
    return hypno


def hypno_upsample_to_data(hypno, sf_hypno, data, sf_data=None, verbose=True):
    """Upsample an hypnogram to a given sampling frequency and fit the
    resulting hypnogram to corresponding EEG data, such that the hypnogram
    and EEG data have the exact same number of samples.

    .. versionadded:: 0.1.5

    Parameters
    ----------
    hypno : array_like
        The sleep staging (hypnogram) 1D array.
    sf_hypno : float
        The current sampling frequency of the hypnogram, in Hz, e.g.

        * 1/30 = 1 value per each 30 seconds of EEG data,
        * 1 = 1 value per second of EEG data
    data : array_like or :py:class:`mne.io.BaseRaw`
        1D or 2D EEG data. Can also be a :py:class:`mne.io.BaseRaw`, in which
        case ``data`` and ``sf_data`` will be automatically extracted.
    sf_data : float
        The sampling frequency of ``data``, in Hz (e.g. 100 Hz, 256 Hz, ...).
        Can be omitted if ``data`` is a :py:class:`mne.io.BaseRaw`.
    verbose : bool or str
        Verbose level. Default (False) will only print warning and error
        messages. The logging levels are 'debug', 'info', 'warning', 'error',
        and 'critical'. For most users the choice is between 'info'
        (or ``verbose=True``) and warning (``verbose=False``).

    Returns
    -------
    hypno : array_like
        The hypnogram, upsampled to ``sf_data`` and cropped/padded to ``max(data.shape)``.

    Warns
    -----
    UserWarning
        If the upsampled ``hypno`` is shorter / longer than ``max(data.shape)``
        and therefore needs to be padded/cropped respectively. This output can be disabled by
        passing ``verbose='ERROR'``.
    """
    set_log_level(verbose)
    if isinstance(data, mne.io.BaseRaw):
        sf_data = data.info['sfreq']
        data = data.times
    hypno_up = hypno_upsample_to_sf(hypno=hypno, sf_hypno=sf_hypno, sf_data=sf_data)
    return hypno_fit_to_data(hypno=hypno_up, data=data, sf=sf_data)


#############################################################################
# HYPNO LOADING
#############################################################################

def load_profusion_hypno(fname, replace=True):  # pragma: no cover
    """
    Load a Compumedics Profusion hypnogram (.xml).

    The Compumedics Profusion hypnogram format is one of the two hypnogram
    formats found in the `National Sleep Research Resource (NSRR)
    <https://sleepdata.org/>`_ website. For more details on the format,
    please refer to
    https://github.com/nsrr/edf-editor-translator/wiki/Compumedics-Annotation-Format

    Parameters
    ----------
    fname : str
        Filename with full path.
    replace : bool
        If True, the integer values will be mapped to YASA default, i.e.
        0 for Wake, 1 for N1, 2 for N2, 3 for N3 / S4 and 4 for REM.
        Note that the native profusion format is identical except for REM
        sleep which is marked as 5.

    Returns
    -------
    hypno : 1D array (n_epochs, )
        Hypnogram, with one value per 30 second epochs.
    sf_hyp : float
        Sampling frequency of the hypnogram. Default is 1 / 30 Hz.
    """
    # Note that an alternative is to use the `xmltodict` library:
    # >>> with open(fname) as in_file:
    # >>>   xml = in_file.read()
    # >>> epoch_length = xml['EpochLength']
    # >>> hypno = np.array(xml['SleepStages']['SleepStage'], dtype='int')
    # >>> xml = xmltodict.parse(xml, process_namespaces=True)['CMPStudyConfig']
    # >>> annotations = pd.DataFrame(xml['ScoredEvents']['ScoredEvent'])
    # >>> annotations["Start"] = annotations["Start"].astype(float)
    # >>> annotations["Duration"] = annotations["Duration"].astype(float)
    import xml.etree.ElementTree as ET
    tree = ET.parse(fname)
    root = tree.getroot()
    epoch_length = float(root[0].text)
    sf_hyp = 1 / epoch_length
    hypno = []
    for s in root[4]:
        hypno.append(s.text)
    hypno = np.array(hypno).astype(int)
    if replace:
        # Stage 4 --> 3 and REM --> 4
        hypno = pd.Series(hypno).replace({4: 3, 5: 4}).to_numpy()
    return hypno, sf_hyp
