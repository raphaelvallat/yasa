"""
Hypnogram-related functions.
"""
import mne
import logging
import numpy as np
import pandas as pd
from .io import set_log_level
from yasa.sleepstats import transition_matrix

__all__ = [
    "hypno_str_to_int",
    "hypno_int_to_str",
    "hypno_consolidate_stages",
    "hypno_upsample_to_sf",
    "hypno_upsample_to_data",
    "hypno_find_periods",
    "load_profusion_hypno",
    "simulate_hypno",
]


logger = logging.getLogger("yasa")


class Hypnogram:
    """Main class for manipulating sleep hypnogram in YASA."""

    def __init__(self, values, n_stages=5, *, freq="30s", start=None):
        assert isinstance(values, (list, np.ndarray, pd.Series))
        assert isinstance(n_stages, int)
        assert n_stages in [2, 3, 4, 5]
        assert isinstance(freq, str)
        assert isinstance(start, (type(None), str, pd.Timestamp))
        if n_stages == 2:
            accepted = ["S", "W", "SLEEP", "WAKE", "ART", "UNS"]
            mapping = {"WAKE": 0, "SLEEP": 1, "ART": -1, "UNS": -2}
        elif n_stages == 3:
            accepted = ["WAKE", "W", "NREM", "REM", "R", "ART", "UNS"]
            mapping = {"WAKE": 0, "NREM": 2, "REM": 4, "ART": -1, "UNS": -2}
        elif n_stages == 4:
            accepted = ["WAKE", "W", "LIGHT", "DEEP", "REM", "R", "ART", "UNS"]
            mapping = {"WAKE": 0, "LIGHT": 2, "DEEP": 3, "REM": 4, "ART": -1, "UNS": -2}
        else:
            accepted = ["WAKE", "W", "N1", "N2", "N3", "REM", "R", "ART", "UNS"]
            mapping = {"WAKE": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "ART": -1, "UNS": -2}
        assert all([val.upper() in accepted for val in values]), (
            f"{np.unique(values)} do not match the accepted values for a {n_stages} stages "
            f"hypnogram: {accepted}"
        )
        if isinstance(values, pd.Series):
            # Make sure to remove index if the input is a pandas.Series
            values = values.to_numpy(copy=True)
        hypno = pd.Series(values, name="Stage").str.upper()
        # Combine accepted values
        map_accepted = {"S": "SLEEP", "W": "WAKE", "R": "REM"}
        hypno = hypno.replace(map_accepted)
        labels = pd.Series(accepted).replace(map_accepted).unique().tolist()
        if start is not None:
            hypno.index = pd.date_range(start=start, freq=freq, periods=hypno.size)
            timedelta = hypno.index - hypno.index[0]
        else:
            fake_dt = pd.date_range(start="2022-12-03 00:00:00", freq=freq, periods=hypno.shape[0])
            timedelta = fake_dt - fake_dt[0]
        hypno.index.name = "Epoch"
        self._hypno = hypno
        self._n_epochs = hypno.shape[0]
        self._freq = freq
        self._sampling_frequency = 1 / pd.Timedelta(freq).total_seconds()
        self._timedelta = timedelta
        self._start = start
        self._n_stages = n_stages
        self._labels = labels
        self._mapping = mapping

    def __repr__(self):
        return f"{self.hypno}"

    def __str__(self):
        return f"{self.hypno}"

    @property
    def hypno(self):
        # Q: Should this be called `hyp.stage`?
        return self._hypno

    @property
    def n_epochs(self):
        return self._n_epochs

    @property
    def freq(self):
        return self._freq

    @property
    def sampling_frequency(self):
        return self._sampling_frequency

    @property
    def start(self):
        return self._start

    @property
    def timedelta(self):
        return self._timedelta

    @property
    def n_stages(self):
        return self._n_stages

    @property
    def labels(self):
        return self._labels

    @property
    def mapping(self):
        return self._mapping

    @mapping.setter
    def mapping(self, map_dict):
        assert isinstance(map_dict, dict), "`mapping` must be a dictionary, e.g. {'WAKE': 0, ...}"
        assert all([val in map_dict.keys() for val in self.hypno.unique()]), (
            f"Some values in `hypno` ({self.hypno.unique()}) are not in `map_dict` "
            f"({map_dict.keys()})"
        )
        if "ART" not in map_dict.keys():
            map_dict["ART"] = -1
        if "UNS" not in map_dict.keys():
            map_dict["UNS"] = -2
        self._mapping = map_dict

    @property
    def mapping_int(self):
        return {v: k for k, v in self.mapping.items()}

    def as_int(self):
        """Return hypnogram as integer.

        The default mapping is:

        * 2 stages: {"WAKE": 0, "SLEEP": 1, "ART": -1, "UNS": -2}
        * 3 stages: {"WAKE": 0, "NREM": 2, "REM": 4, "ART": -1, "UNS": -2}
        * 4 stages: {"WAKE": 0, "LIGHT": 2, "DEEP": 3, "REM": 4, "ART": -1, "UNS": -2}
        * 5 stages: {"WAKE": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "ART": -1, "UNS": -2}

        Users can define a custom mapping:

        >>> hyp.mapping = {"WAKE": 0, "NREM": 1, "REM": 2}
        """
        return self.hypno.replace(self.mapping).astype(int)

    def transition_matrix(self):
        counts, probs = transition_matrix(self.as_int())
        counts.index = counts.index.map(self.mapping_int)
        counts.columns = counts.columns.map(self.mapping_int)
        probs.index = probs.index.map(self.mapping_int)
        probs.columns = probs.columns.map(self.mapping_int)
        return counts, probs

    def sleep_statistics(self):
        """
        Compute standard sleep statistics from an hypnogram.

        Parameters
        ----------
        self : yasa.Hypnogram
            Hypnogram, assumed to be already cropped to time in bed (TIB, also referred to as
            Total Recording Time, i.e. "lights out" to "lights on").

        Returns
        -------
        stats : dict
            Summary sleep statistics.

        Notes
        -----
        All values except SE, SME, SFI and the percentage of each stage are expressed in minutes.
        YASA follows the AASM guidelines to calculate these parameters:

        * Time in Bed (TIB): total duration of the hypnogram.
        * Sleep Period Time (SPT): duration from first to last period of sleep.
        * Wake After Sleep Onset (WASO): duration of wake periods within SPT.
        * Total Sleep Time (TST): total sleep duration in SPT.
        * Sleep Onset Latency (SOL): Latency to first epoch of any sleep.
        * SOL 5min: Latency to 5 minutes of persistent sleep (any stage).
        * REM latency: latency to first REM sleep.
        * Sleep Efficiency (SE): TST / TIB * 100 (%).
        * Sleep Maintenance Efficiency (SME): TST / SPT * 100 (%).
        * Sleep Fragmentation Index: number of transitions from sleep to wake / hours of TST
        * Sleep stages amount and proportion of TST

        .. warning::
            Artefact and Unscored epochs are excluded from the calculation of the
            total sleep time (TST). TST is calculated as the sum of all REM and NREM sleep in SPT.

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
        Sleep statistics for a 2-stage hypnogram

        >>> from yasa import Hypnogram
        >>> # Generate a fake hypnogram, where "S" = Sleep, "W" = Wake
        >>> values = 10 * ["W"] + 40 * ["S"] + 5 * ["W"] + 40 * ["S"] + 10 * ["W"]
        >>> hyp = Hypnogram(values, n_stages=2)
        >>> hyp.sleep_statistics()
        {'TIB': 52.5,
        'SPT': 42.5,
        'WASO': 2.5,
        'TST': 40.0,
        'SE': 76.1905,
        'SME': 94.1176,
        'SFI': 1.5,
        'SOL': 5.0,
        'SOL_5min': 5.0,
        'WAKE': 12.5}

        Sleep statistics for a 5-stages hypnogram, where each epoch is one minute

        >>> values = (10 * ["W"] + 4 * ["N1"] + 1 * ["W"] + 30 * ["N2"] + 30 * ["N3"] + 5 * ["W"]
        ...           + 25 * ["REM"] + 15 * ["N2"] + 10 * ["W"])
        >>> hyp = Hypnogram(values, freq="1min", n_stages=5)
        >>> hyp.sleep_statistics()
        {'TIB': 130.0,
        'SPT': 110.0,
        'WASO': 6.0,
        'TST': 104.0,
        'SE': 80.0,
        'SME': 94.5455,
        'SFI': 1.7308,
        'SOL': 10.0,
        'SOL_5min': 15.0,
        'Lat_REM': 80.0,
        'WAKE': 26.0,
        'N1': 4.0,
        'N2': 45.0,
        'N3': 30.0,
        'REM': 25.0,
        '%N1': 3.8462,
        '%N2': 43.2692,
        '%N3': 28.8462,
        '%REM': 24.0385}
        """
        hypno = self.hypno.to_numpy()
        assert self.n_epochs > 0, "Hypnogram is empty!"
        all_sleep = ["SLEEP", "N1", "N2", "N3", "NREM", "REM", "LIGHT", "DEEP"]
        all_non_sleep = ["WAKE", "ART", "UNS"]
        stats = {}

        # TIB, first and last sleep
        stats["TIB"] = self.n_epochs
        idx_sleep = np.where(~np.isin(hypno, all_non_sleep))[0]
        if not len(idx_sleep):
            first_sleep, last_sleep = 0, self.n_epochs
        else:
            first_sleep = idx_sleep[0]
            last_sleep = idx_sleep[-1]
        # Crop to SPT
        hypno_s = hypno[first_sleep : (last_sleep + 1)]
        stats["SPT"] = hypno_s.size if len(idx_sleep) else 0
        stats["WASO"] = hypno_s[hypno_s == "WAKE"].size if len(idx_sleep) else np.nan
        # Before YASA v0.5.0, TST was calculated as SPT - WASO, meaning that Art
        # and Unscored epochs were included. TST is now restrained to sleep stages.
        stats["TST"] = hypno_s[np.isin(hypno_s, all_sleep)].shape[0]

        # Sleep efficiency and fragmentation
        stats["SE"] = 100 * stats["TST"] / stats["TIB"]
        if stats["SPT"] == 0:
            stats["SME"] = np.nan
            stats["SFI"] = np.nan
        else:
            # Sleep maintenance efficiency
            stats["SME"] = 100 * stats["TST"] / stats["SPT"]
            # SFI is the ratio of the number of transitions from sleep into Wake to TST (hours)
            # The original definition included transitions into Wake or N1.
            counts, _ = self.transition_matrix()
            n_trans_to_wake = np.sum(
                counts.loc[
                    np.intersect1d(counts.index, all_sleep), np.intersect1d(counts.index, ["WAKE"])
                ].to_numpy()
            )
            stats["SFI"] = n_trans_to_wake / (stats["TST"] / (3600 * self.sampling_frequency))

        # Sleep stage latencies -- only relevant if hypno is cropped to TIB
        stats["SOL"] = first_sleep if stats["TST"] > 0 else np.nan
        sleep_periods = hypno_find_periods(
            np.isin(hypno, all_sleep), self.sampling_frequency, threshold="5min"
        ).query("values == True")
        if sleep_periods.shape[0]:
            stats["SOL_5min"] = sleep_periods["start"].iloc[0]
        else:
            stats["SOL_5min"] = np.nan

        if "REM" in self.labels:
            # Question: should we add latencies for other stage too?
            stats["Lat_REM"] = np.where(hypno == "REM")[0].min() if "REM" in hypno else np.nan

        # Duration of each stage
        for st in self.labels:
            if st == "SLEEP":
                # SLEEP == TST
                continue
            stats[st] = hypno[hypno == st].size

        # Remove ART and UNS if they are empty
        if stats["ART"] == 0:
            stats.pop("ART")
        if stats["UNS"] == 0:
            stats.pop("UNS")

        # Convert to minutes
        for key, value in stats.items():
            if key in ["SE", "SME"]:
                continue
            stats[key] = value / (60 * self.sampling_frequency)

        # Proportion of each sleep stages
        for st in all_sleep:
            if st in stats.keys():
                if stats["TST"] == 0:
                    stats[f"%{st}"] = np.nan
                else:
                    stats[f"%{st}"] = 100 * stats[st] / stats["TST"]

        # Round to 4 decimals
        stats = {key: np.round(val, 4) for key, val in stats.items()}
        return stats

    def plot_hypnogram(self):
        # This function requires mapping to be defined
        raise NotImplementedError

    def upsample(self, new_freq, **kwargs):
        """Upsample hypnogram to a higher frequency.

        Frequency here is defined with a pandas Offset, e.g. "10s" or "1min".

        Returns a copy: the original hypnogram is not modified in place.
        """
        assert pd.Timedelta(new_freq) < pd.Timedelta(self.freq), (
            f"The upsampling `new_freq` ({new_freq}) must be higher than the current frequency of "
            f"hypnogram {self.freq}"
        )
        if isinstance(self.hypno.index, pd.DatetimeIndex):
            new_hyp = self.hypno.resample(new_freq, origin="start", **kwargs).ffill()
        else:
            new_hyp = self.hypno.copy()
            new_hyp.index = self.timedelta
            new_hyp = new_hyp.resample(new_freq, **kwargs).ffill().reset_index(drop=True)
            new_hyp.index.name = "Epoch"
        return Hypnogram(values=new_hyp, n_stages=self.n_stages, freq=new_freq, start=self.start)


#############################################################################
# STR <--> INT CONVERSION
#############################################################################


def hypno_str_to_int(
    hypno,
    mapping_dict={
        "w": 0,
        "wake": 0,
        "n1": 1,
        "s1": 1,
        "n2": 2,
        "s2": 2,
        "n3": 3,
        "s3": 3,
        "s4": 3,
        "r": 4,
        "rem": 4,
        "art": -1,
        "mt": -1,
        "uns": -2,
        "nd": -2,
    },
):
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
    assert isinstance(hypno, (list, np.ndarray, pd.Series)), "Not an array."
    hypno = pd.Series(np.asarray(hypno, dtype=str))
    assert not hypno.str.isnumeric().any(), "Hypno contains numeric values."
    return hypno.str.lower().map(mapping_dict).values


def hypno_int_to_str(
    hypno, mapping_dict={0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R", -1: "Art", -2: "Uns"}
):
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
    assert isinstance(hypno, (list, np.ndarray, pd.Series)), "Not an array."
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
    assert sf_hypno <= sf_data, "sf_hypno must be less than sf_data."
    assert repeats.is_integer(), "sf_hypno / sf_data must be a whole number."
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
        sf = data.info["sfreq"]
        data = data.times  # 1D array and does not require to preload data
    data = np.asarray(data)
    hypno = np.asarray(hypno)
    assert hypno.ndim == 1, "Hypno must be 1D."
    npts_hyp = hypno.size
    npts_data = max(data.shape)  # Support for 2D data
    if npts_hyp < npts_data:
        # Hypnogram is shorter than data
        npts_diff = npts_data - npts_hyp
        if sf is not None:
            dur_diff = npts_diff / sf
            logger.warning(
                "Hypnogram is SHORTER than data by %.2f seconds. "
                "Padding hypnogram with last value to match data.size." % dur_diff
            )
        else:
            logger.warning(
                "Hypnogram is SHORTER than data by %i samples. "
                "Padding hypnogram with last value to match data.size." % npts_diff
            )
        hypno = np.pad(hypno, (0, npts_diff), mode="edge")
    elif npts_hyp > npts_data:
        # Hypnogram is longer than data
        npts_diff = npts_hyp - npts_data
        if sf is not None:
            dur_diff = npts_diff / sf
            logger.warning(
                "Hypnogram is LONGER than data by %.2f seconds. "
                "Cropping hypnogram to match data.size." % dur_diff
            )
        else:
            logger.warning(
                "Hypnogram is LONGER than data by %i samples. "
                "Cropping hypnogram to match data.size." % npts_diff
            )
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
        sf_data = data.info["sfreq"]
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


#############################################################################
# PERIODS & CYCLES
#############################################################################


def hypno_find_periods(hypno, sf_hypno, threshold="5min", equal_length=False):
    """Find sequences of consecutive values exceeding a certain duration in hypnogram.

    .. versionadded:: 0.6.2

    Parameters
    ----------
    hypno : array_like
        A 1D array with the sleep stages (= hypnogram). The dtype can be anything (int, bool, str).
        More generally, this can be any vector for which you wish to find runs of
        consecutive items.
    sf_hypno : float
        The current sampling frequency of ``hypno``, in Hz, e.g. 1/30 = 1 value per each 30 seconds
        of EEG data, 1 = 1 value per second of EEG data.
    threshold : str
        This function will only keep periods that exceed a certain duration (default '5min'), e.g.
        '5min', '15min', '30sec', '1hour'. To disable thresholding, use '0sec'.
    equal_length : bool
        If True, the periods will all have the exact duration defined
        in threshold. That is, periods that are longer than the duration threshold will be divided
        into sub-periods of exactly the length of ``threshold``.

    Returns
    -------
    periods : :py:class:`pandas.DataFrame`
        Output dataframe

        * ``values`` : The value in hypno of the current period
        * ``start`` : The index of the start of the period in hypno
        * ``length`` : The duration of the period, in number of samples

    Examples
    --------
    Let's assume that we have an hypnogram where sleep = 1 and wake = 0. There is one value per
    minute, and therefore the sampling frequency of the hypnogram is 1 / 60 sec (~0.016 Hz).

    >>> import yasa
    >>> hypno = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    >>> yasa.hypno_find_periods(hypno, sf_hypno=1/60, threshold="0min")
       values  start  length
    0       0      0      11
    1       1     11       3
    2       0     14       2
    3       1     16       9
    4       0     25       2

    This gives us the start and duration of each sequence of consecutive values in the hypnogram.
    For example, the first row tells us that there is a sequence of 11 consecutive 0 starting at
    the first index of hypno.

    Now, we may want to keep only periods that are longer than a specific threshold,
    for example 5 minutes:

    >>> yasa.hypno_find_periods(hypno, sf_hypno=1/60, threshold="5min")
       values  start  length
    0       0      0      11
    1       1     16       9

    Only the two sequences that are longer than 5 minutes (11 minutes and 9 minutes respectively)
    are kept. Feel free to play around with different values of threshold!

    This function is not limited to binary arrays, e.g.

    >>> hypno = [0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 0, 1]
    >>> yasa.hypno_find_periods(hypno, sf_hypno=1/60, threshold="2min")
       values  start  length
    0       0      0       4
    1       2      5       6
    2       0     11       3

    Lastly, using ``equal_length=True`` will further divide the periods into segments of the
    same duration, i.e. the duration defined in ``threshold``:

    >>> hypno = [0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 0, 1]
    >>> yasa.hypno_find_periods(hypno, sf_hypno=1/60, threshold="2min", equal_length=True)
       values  start  length
    0       0      0       2
    1       0      2       2
    2       2      5       2
    3       2      7       2
    4       2      9       2
    5       0     11       2

    Here, the first period of 4 minutes of consecutive 0 is further divided into 2 periods of
    exactly 2 minutes. Next, the sequence of 6 consecutive 2 is further divided into 3 periods of
    2 minutes. Lastly, the last value in the sequence of 3 consecutive 0 at the end of the array is
    removed to keep only a segment of 2 exactly minutes. In other words, the remainder of the
    division of a given segment by the desired duration is discarded.
    """
    # Convert the threshold to number of samples
    assert isinstance(threshold, str), "Threshold must be a string, e.g. '5min', '30sec', '15min'"
    thr_sec = pd.Timedelta(threshold).total_seconds()
    thr_samp = sf_hypno * thr_sec
    if float(thr_samp).is_integer():
        thr_samp = int(thr_samp)
    else:
        raise ValueError(
            f"The selected threshold does not result in an whole number of samples ("
            f"{thr_sec:.3f} seconds * {sf_hypno:.3f} Hz = {thr_samp:.3f} samples)"
        )

    # Find run starts
    # https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
    assert isinstance(hypno, (list, np.ndarray, pd.Series)), "hypno must be an array."
    x = np.asarray(hypno)
    n = x.shape[0]
    loc_run_start = np.empty(n, dtype=bool)
    loc_run_start[0] = True
    loc_run_start[1:] = x[:-1] != x[1:]
    run_starts = np.nonzero(loc_run_start)[0]
    # Find run values
    run_values = x[loc_run_start]
    # Find run lengths
    run_lengths = np.diff(np.append(run_starts, n))
    seq = pd.DataFrame({"values": run_values, "start": run_starts, "length": run_lengths})

    # Remove runs that are shorter than threshold
    seq = seq[seq["length"] >= thr_samp].reset_index(drop=True)

    if not equal_length:
        return seq

    # Divide into epochs of equal length
    assert thr_samp > 0, "Threshold must be non-zero if using equal_length=True."
    new_seq = {"values": [], "start": [], "length": []}

    for i, row in seq.iterrows():
        quotient, remainder = np.divmod(row["length"], thr_samp)
        new_start = row["start"]
        if quotient > 0:
            while quotient != 0:
                new_seq["values"].append(row["values"])
                new_seq["start"].append(new_start)
                new_seq["length"].append(thr_samp)
                new_start += thr_samp
                quotient -= 1
        else:
            new_seq["values"].append(row["values"])
            new_seq["start"].append(row["start"])
            new_seq["length"].append(row["length"])

    new_seq = pd.DataFrame(new_seq)
    return new_seq


#############################################################################
# STAGE CONVERSION
#############################################################################


def hypno_consolidate_stages(hypno, n_stages_in, n_stages_out):
    """Reduce the number of stages in a hypnogram to match actigraphy or wearables.

    For example, a standard 5-stage hypnogram (W, N1, N2, N3, REM) could be consolidated
    to a hypnogram more common with actigraphy (W, Light, Deep, REM).

    Parameters
    ----------
    hypno : array_like
        The sleep staging (hypnogram) 1D array.
    n_stages_in : int
        Number of possible stages of ``hypno``, where:

        - 5 stages - 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM
        - 4 stages - 0=Wake, 2=Light, 3=Deep, 4=REM
        - 3 stages - 0=Wake, 2=NREM, 4=REM
        - 2 stages - 0=Wake, 1=Sleep

        .. note:: The default YASA values for Unscored (-2) and Artefact (-1) are always allowed.
    n_stages_out : int
        Similar to ``n_stages_in`` but for output. Must be higher than ``n_stages_out``.

    Returns
    -------
    hypno : array_like
        The hypnogram, with stages converted to ``n_stages_out`` staging scheme.
    """
    assert isinstance(hypno, (list, np.ndarray, pd.Series)), "hypno must be array_like"
    hypno = np.asarray(hypno, dtype=int).copy()
    assert n_stages_in in [3, 4, 5], "n_stages_in must be 3, 4, or 5"
    assert n_stages_out in [2, 3, 4], "n_stages_in must be 2, 3, or 4"
    assert n_stages_out < n_stages_in, "n_stages_out must be lower than n_stages_in"

    # Change sleep codes where applicable.
    if n_stages_out == 2:
        # Consolidate all Sleep
        mapping = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, -1: -1, -2: -2}
    elif n_stages_out == 3:
        # Consolidate N1/N2/N3 or Light/Deep into NREM
        mapping = {0: 0, 1: 2, 2: 2, 3: 2, 4: 4, -1: -1, -2: -2}
    elif n_stages_out == 4:
        # Consolidate N1/N2 into Light
        mapping = {0: 0, 1: 2, 2: 2, 3: 3, 4: 4, -1: -1, -2: -2}
    hypno = pd.Series(hypno).map(mapping).to_numpy()

    return hypno


#############################################################################
# SIMULATION
#############################################################################


def simulate_hypno(tib=90, sf=1 / 30, n_stages=5, trans_probas=None, init_probas=None, seed=None):
    """Simulate a hypnogram based on transition probabilities.

    Current implentation is a naive Markov model. The initial stage of a hypnogram
    is generated using probabilites from ``init_probas`` and then subsequent stages
    are generated from a Markov sequence based on ``trans_probas``.

    .. important:: The Markov simulation model is not meant to accurately portray sleep
        macroarchitecture and should only be used for testing or other unique purposes.

    .. versionadded:: 0.6.3

    Parameters
    ----------
    tib : int
        Total duration of the hypnogram (i.e., time in bed), expressed in minutes.
        Returned hypnogram will be slightly shorter if ``tib`` in seconds is not
        evenly divisible by ``sf``.

        .. seealso:: :py:func:`yasa.sleep_statistics`
    sf : float
        Sampling frequency.
    n_stages : int
        Staging scheme of returned hypnogram. Input should follow 5-stage scheme but can
        be converted to lower scheme if desired.

        .. seealso:: :py:func:`yasa.hypno_consolidate_stages`
    trans_probas : :py:class:`pandas.DataFrame` or None
        Transition probability matrix where each cell is a transition probability
        between sleep stages of consecutive *epochs*.

        ``trans_probas`` is a `right stochastic matrix
        <https://en.wikipedia.org/wiki/Stochastic_matrix>`_, i.e. each row sums to 1.

        If None (default), use transition probabilites from Metzner et al., 2021 [Metzner2021]_.
        If :py:class:`pandas.DataFrame`, must have "from"-stages as indices and
        "to"-stages as columns. Indices and columns must follow YASA integer
        hypnogram convention (W = 0, N = 1, ...). Unscored/Artefact stages are not allowed.

        .. note:: Transition probability matrices should indicate the transition
            probability between *epochs* (i.e., probability of the next epoch) and
            not simply stage (i.e., probability of non-similar stage).

        .. seealso:: Return value from :py:func:`yasa.transition_matrix`
    init_probas : :py:class:`pandas.Series` or None
        Probabilites of each stage to initialize random walk.
        If None (default), initialize with "From"-Wake row of ``trans_probas``.
        If :py:class:`pandas.Series`, indices must be stages following YASA integer
        hypnogram convention (see ``trans_probas``).
    seed : int or None
        Random seed for generating Markov sequence.
        If an integer number is provided, the random hypnogram will be predictable.
        This argument is required if reproducible results are desired.

    Returns
    -------
    hypno : np.ndarray
        Hypnogram containing simulated sleep stages.

    Notes
    -----
    Default transition probabilities can be found in the ``traMat_Epoch.npy`` file of
    Supplementary Information for Metzner et al., 2021 [Metzner2021]_ (rounded values
    are viewable in Figure 5b). Please cite this work if these probabilites are used
    for publication.

    References
    ----------
    .. [Metzner2021] Metzner, C., Schilling, A., Traxdorf, M., Schulze, H., & Krausse, P.
                     (2021). Sleep as a random walk: a super-statistical analysis of EEG
                     data across sleep stages. Communications Biology, 4.
                     https://doi.org/10.1038/s42003-021-02912-6

    Examples
    --------
    >>> from yasa import simulate_hypno
    >>> hypno = simulate_hypno(tib=10, seed=0)
    >>> print(hypno)
    [0 0 0 0 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2]

    >>> hypno = simulate_hypno(tib=10, n_stages=2, seed=0)
    >>> print(hypno)
    [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]

    Base the data off a real subject's transition matrix.

    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> import yasa
        >>> url = (
        >>>     "https://github.com/raphaelvallat/yasa/raw/master/"
        >>>     "notebooks/data_full_6hrs_100Hz_hypno_30s.txt"
        >>> )
        >>> hypno = np.loadtxt(url)
        >>> _, probas = yasa.transition_matrix(hypno)
        >>> hypno_sim = yasa.simulate_hypno(tib=360, trans_probas=probas, seed=9)
        >>> fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))
        >>> yasa.plot_hypnogram(hypno, ax=ax1)
        >>> yasa.plot_hypnogram(hypno_sim, ax=ax2)
        >>> ax1.set_title("True hypnogram")
        >>> ax2.set_title("Simulated hypnogram")
        >>> plt.tight_layout()
    """
    # Validate input
    assert isinstance(tib, (int, float)), "tib must be a number"
    assert isinstance(sf, (int, float)), "sf must be a number"
    assert isinstance(n_stages, int), "n_stages must be an integer"
    assert 2 <= n_stages <= 5, "n_stages must be 2, 3, 4, or 5"
    if seed is not None:
        assert isinstance(seed, int) and seed >= 0, "seed must be an integer >= 0"
    if trans_probas is not None:
        assert isinstance(trans_probas, pd.DataFrame), "trans_probas must be a pandas DataFrame"
    if init_probas is not None:
        assert isinstance(init_probas, pd.Series), "init_probas must be a pandas Series"

    # Initialize random number generator
    rng = np.random.default_rng(seed)

    def _markov_sequence(p_init, p_transition, sequence_length):
        """Generate a Markov sequence based on p_init and p_transition.
        https://ericmjl.github.io/essays-on-data-science/machine-learning/markov-models
        """
        initial_state = list(rng.multinomial(1, p_init)).index(1)
        states = [initial_state]
        while len(states) < sequence_length:
            p_tr = p_transition[states[-1]]
            new_state = list(rng.multinomial(1, p_tr)).index(1)
            states.append(new_state)
        return np.asarray(states)

    if trans_probas is None:
        # Generate transition probability DataFrame (here, ordered W R 1 2 3)
        trans_freqs = np.array(
            [
                [11737, 2, 571, 84, 2],
                [57, 10071, 189, 84, 2],
                [281, 59, 6697, 1661, 11],
                [253, 272, 1070, 26259, 505],
                [49, 12, 176, 279, 9630],
            ]
        )
        trans_probas = trans_freqs / trans_freqs.sum(axis=1, keepdims=True)
        trans_probas = pd.DataFrame(
            trans_probas,
            index=[0, 4, 1, 2, 3],
            columns=[0, 4, 1, 2, 3],
        )

    if init_probas is None:
        # Extract Wake row of initial probabilities as a Series
        init_probas = trans_probas.loc[0, :]

    # Ensure trans_probas DataFrame and init_probas Series are in row/column order W N1 N2 N3 R
    trans_probas = trans_probas.reindex([0, 1, 2, 3, 4], axis=0)
    trans_probas = trans_probas.reindex([0, 1, 2, 3, 4], axis=1)
    init_probas = init_probas.reindex([0, 1, 2, 3, 4])
    assert trans_probas.notna().values.all(), "trans_proba indices must be YASA integer codes"
    assert init_probas.notna().all(), "init_probas index must be YASA integer codes"

    # Extract probabilities as arrays
    trans_arr = trans_probas.to_numpy()
    init_arr = init_probas.to_numpy()

    # Make sure all rows sum to 1
    assert np.allclose(trans_arr.sum(axis=1), 1)
    assert np.isclose(init_arr.sum(), 1)

    # Find number of *complete* epochs within TIB timeframe
    n_epochs = np.floor(tib * 60 * sf).astype(int)
    # Generate hypnogram
    hypno = _markov_sequence(init_arr, trans_arr, n_epochs)

    if n_stages < 5:
        hypno = hypno_consolidate_stages(hypno, n_stages_in=5, n_stages_out=n_stages)

    return hypno
