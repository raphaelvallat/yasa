"""
Hypnogram-related functions.
"""
import mne
import logging
import numpy as np
import pandas as pd
from yasa.io import set_log_level
from yasa.sleepstats import transition_matrix

__all__ = [
    "Hypnogram",
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
    """
    Main class for manipulating hypnogram in YASA.

    Starting with YASA v0.7, YASA takes a more object-oriented approach to hypnograms. That is,
    hypnograms are now stored as a class (aka object), which comes with its own attributes and
    functions. Furthermore, YASA does not allow integer values to define the stages anymore.
    Instead, users must pass an array of strings with the actual stage names
    (e.g. ["WAKE", "WAKE", "N1", ..., "REM", "REM"]).

    .. seealso:: :py:func:`yasa.simulate_hypno`

    .. versionadded:: 0.7.0

    Parameters
    ----------
    values : array_like
        A vector of stage values, represented as strings. See some examples below:

        * 2-stages hypnogram (Wake/Sleep): ``["W", "S", "S", "W", "S"]``
        * 3-stages (Wake/NREM/REM): ``pd.Series(["WAKE", "NREM", "NREM", "REM", "REM"])``
        * 4-stages (Wake/Light/Deep/REM): ``np.array(["Wake", "Light", "Deep", "Deep"])``
        * 5-stages (default): ``["N1", "N1", "N2", "N3", "N2", "REM", "W"]``

        Artefacts ("Art") and unscored ("Uns") epochs are always allowed regardless of the
        number of stages in the hypnogram.

        .. note:: Abbreviated or full spellings for the stages are allowed, as well as
            lower/upper/mixed case. Internally, YASA will convert the stages to to full spelling
            and uppercase (e.g. "w" -> "WAKE").
    n_stages : int
        Whether ``values`` comes from a 2, 3, 4 or 5-stages hypnogram. Default is 5 stages, meaning
        that the following sleep stages are allowed: N1, N2, N3, REM, WAKE.
    freq : str
        A pandas frequency string indicating the frequency resolution of the hypnogram. Default is
        "30s" meaning that each value in the hypnogram represents a 30-seconds epoch.
        Examples: "1min", "10s", "15min". A full list of accepted values can be found at
        https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases

        ``freq`` will be passed to the :py:func:`pandas.date_range` function to create the time
        index of the hypnogram.
    start : str or datetime
        An optional string indicating the starting datetime of the hypnogram
        (e.g. "2022-12-15 22:30:00"). If ``start`` is specified and valid, the index of the
        hypnogram will be a :py:class:`pandas.DatetimeIndex`. Otherwise it will be a
        :py:class:`pandas.RangeIndex`, indicating the epoch number.
    scorer : str
        An optional string indicating the scorer name. If specified, this will be set as the name
        of the :py:class:`pandas.Series`, otherwise the name will be set to "Stage".

    Examples
    --------
    Create a 2-stages hypnogram

    >>> from yasa import Hypnogram
    >>> values = ["W", "W", "W", "S", "S", "S", "S", "S", "W", "S", "S", "S"]
    >>> hyp = Hypnogram(values, n_stages=2)
    >>> hyp.hypno
    Epoch
    0      WAKE
    1      WAKE
    2      WAKE
    3     SLEEP
    4     SLEEP
    5     SLEEP
    6     SLEEP
    7     SLEEP
    8      WAKE
    9     SLEEP
    10    SLEEP
    11    SLEEP
    Name: Stage, dtype: object

    >>> hyp.n_epochs
    12

    >>> hyp.mapping
    {'WAKE': 0, 'SLEEP': 1, 'ART': -1, 'UNS': -2}

    >>> hyp.as_int()
    Epoch
    0     0
    1     0
    2     0
    3     1
    4     1
    5     1
    6     1
    7     1
    8     0
    9     1
    10    1
    11    1
    Name: Stage, dtype: int64

    >>> hyp.sleep_statistics()
    {'TIB': 6.0,
     'SPT': 4.5,
     'WASO': 0.5,
     'TST': 4.0,
     'SE': 66.6667,
     'SME': 88.8889,
     'SFI': 7.5,
     'SOL': 1.5,
     'SOL_5min': nan,
     'WAKE': 2.0}

    >>> counts, probs = hyp.transition_matrix()
    >>> counts
    To Stage    WAKE  SLEEP
    From Stage
    WAKE           2      2
    SLEEP          1      6

    Simulate a 5-stages hypnogram

    >>> from yasa import simulate_hypno
    >>> hyp = simulate_hypno(tib=500, n_stages=5, start="2022-12-15 22:30:00", scorer="S1", seed=42)
    >>> hyp
    Time
    2022-12-15 22:30:00    WAKE
    2022-12-15 22:30:30    WAKE
    2022-12-15 22:31:00    WAKE
    2022-12-15 22:31:30    WAKE
    2022-12-15 22:32:00    WAKE
                        ...
    2022-12-16 06:47:30      N2
    2022-12-16 06:48:00      N2
    2022-12-16 06:48:30      N2
    2022-12-16 06:49:00      N2
    2022-12-16 06:49:30      N2
    Freq: 30S, Name: S1, Length: 1000, dtype: object

    >>> hyp.sleep_statistics()
    {'TIB': 500.0,
     'SPT': 497.5,
     'WASO': 79.5,
     'TST': 418.0,
     'SE': 83.6,
     'SME': 84.0201,
     'SFI': 0.7177,
     'SOL': 2.5,
     'SOL_5min': 2.5,
     'Lat_REM': 67.0,
     'WAKE': 82.0,
     'N1': 69.0,
     'N2': 247.0,
     'N3': 64.5,
     'REM': 37.5,
     '%N1': 16.5072,
     '%N2': 59.0909,
     '%N3': 15.4306,
     '%REM': 8.9713}
    """

    def __init__(self, values, n_stages=5, *, freq="30s", start=None, scorer=None):
        assert isinstance(
            values, (list, np.ndarray, pd.Series)
        ), "`values` must be a list, numpy.array or pandas.Series"
        assert isinstance(n_stages, int), "`n_stages` must be an integer between 2 and 5."
        assert n_stages in [2, 3, 4, 5], "`n_stages` must be an integer between 2 and 5."
        assert isinstance(freq, str), "`freq` must be a pandas frequency string."
        assert isinstance(
            start, (type(None), str, pd.Timestamp)
        ), "`start` must be either None, a string or a pandas.Timestamp."
        assert isinstance(
            scorer, (type(None), str, int)
        ), "`scorer` must be either None, or a string or an integer."
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
        hypno = pd.Series(values).str.upper()
        if scorer is None:
            hypno.name = "Stage"
        else:
            hypno.name = scorer
        # Combine accepted values
        map_accepted = {"S": "SLEEP", "W": "WAKE", "R": "REM"}
        hypno = hypno.replace(map_accepted)
        labels = pd.Series(accepted).replace(map_accepted).unique().tolist()
        if start is not None:
            hypno.index = pd.date_range(start=start, freq=freq, periods=hypno.shape[0])
            hypno.index.name = "Time"
            timedelta = hypno.index - hypno.index[0]
        else:
            fake_dt = pd.date_range(start="2022-12-03 00:00:00", freq=freq, periods=hypno.shape[0])
            hypno.index.name = "Epoch"
            timedelta = fake_dt - fake_dt[0]
        self._hypno = hypno
        self._n_epochs = hypno.shape[0]
        self._freq = freq
        self._sampling_frequency = 1 / pd.Timedelta(freq).total_seconds()
        self._start = start
        self._timedelta = timedelta
        self._tib = self._n_epochs / (60 * self._sampling_frequency)
        # self._tib = self._n_epochs * pd.Timedelta(freq).total_seconds() / 60
        self._n_stages = n_stages
        self._labels = labels
        self._mapping = mapping
        self._scorer = scorer

    def __repr__(self):
        return f"{self.hypno}"

    def __str__(self):
        return f"{self.hypno}"

    @property
    def hypno(self):
        """The hypnogram values, stored in a :py:class:`pandas.Series`."""
        return self._hypno

    @property
    def n_epochs(self):
        """The number of epochs in the hypnogram."""
        return self._n_epochs

    @property
    def freq(self):
        """The frequency resolution of the hypnogram. Default is '30s'"""
        return self._freq

    @property
    def sampling_frequency(self):
        """The sampling frequency (Hz) of the hypnogram."""
        return self._sampling_frequency

    @property
    def start(self):
        """The start date/time of the hypnogram. Default is None."""
        return self._start

    @property
    def timedelta(self):
        """
        A :py:class:`pandas.TimedeltaIndex` vector with the accumulated time difference of each
        epoch compared to the first epoch.
        """
        return self._timedelta

    @property
    def tib(self):
        """Time in bed, or total duration of the hypnogram, expressed in minutes."""
        return self._tib
        ## Q: Should this be called "duration" instead?

    @property
    def n_stages(self):
        """
        The number of allowed stages in the hypnogram. This is not the number of unique stages
        in the current hypnogram.
        """
        return self._n_stages

    @property
    def labels(self):
        """The allowed stage labels."""
        return self._labels

    @property
    def mapping(self):
        """A dictionary with the mapping from string to integer values."""
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
        """A dictionary with the mapping from integer to string values."""
        return {v: k for k, v in self.mapping.items()}

    @property
    def scorer(self):
        """The scorer name."""
        return self._scorer

    # CLASS METHODS BELOW

    def as_annotations(self):
        """
        Return a pandas DataFrame summarizing epoch-level information.

        Column order and names are compliant with BIDS
        `events files
        <https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/05-task-events.html>`_
        and MNE `events/annotations dataframes
        <https://mne.tools/stable/glossary.html#term-annotations>`_.

        Returns
        -------
        annotations : :py:class:`pandas.DataFrame`
            A dataframe containing epoch onset, duration, stage, etc.

        Examples
        --------
        >>> from yasa import Hypnogram
        >>> hyp = Hypnogram(["W", "W", "LIGHT", "LIGHT", "DEEP", "REM", "WAKE"], n_stages=4)
        >>> hyp.as_annotations()
               onset  duration  value description
        epoch
        0        0.0      30.0      0        WAKE
        1       30.0      30.0      0        WAKE
        2       60.0      30.0      2       LIGHT
        3       90.0      30.0      2       LIGHT
        4      120.0      30.0      3        DEEP
        5      150.0      30.0      4         REM
        6      180.0      30.0      0        WAKE
        """
        data = {
            "onset": self.timedelta.total_seconds(),
            "duration": 1 / self.sampling_frequency,
            "value": self.as_int().to_numpy(),
            "description": self.hypno.to_numpy(),
            "epoch": np.arange(self.n_epochs),
        }
        if self.scorer is not None:
            data["scorer"] = self.scorer
        return pd.DataFrame(data).set_index("epoch")

    def as_int(self):
        """Return hypnogram as integers.

        The default mapping from string to integer is:

        * 2 stages: {"WAKE": 0, "SLEEP": 1, "ART": -1, "UNS": -2}
        * 3 stages: {"WAKE": 0, "NREM": 2, "REM": 4, "ART": -1, "UNS": -2}
        * 4 stages: {"WAKE": 0, "LIGHT": 2, "DEEP": 3, "REM": 4, "ART": -1, "UNS": -2}
        * 5 stages: {"WAKE": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "ART": -1, "UNS": -2}

        Users can define a custom mapping:

        >>> hyp.mapping = {"WAKE": 0, "NREM": 1, "REM": 2}

        Examples
        --------
        Convert a 2-stages hypnogram to integers

        >>> from yasa import Hypnogram
        >>> hyp = Hypnogram(["W", "W", "S", "S", "W", "S"], n_stages=2)
        >>> hyp.as_int()
        Epoch
        0    0
        1    0
        2    1
        3    1
        4    0
        5    1
        Name: Stage, dtype: int64

        Same with a 4-stages hypnogram

        >>> from yasa import Hypnogram
        >>> hyp = Hypnogram(["W", "W", "LIGHT", "LIGHT", "DEEP", "REM", "WAKE"], n_stages=4)
        >>> hyp.as_int()
        Epoch
        0    0
        1    0
        2    2
        3    2
        4    3
        5    4
        6    0
        Name: Stage, dtype: int64
        """
        return self.hypno.replace(self.mapping).astype(int)

    def consolidate_stages(self, new_n_stages):
        """Reduce the number of stages in a hypnogram to match actigraphy or wearables.

        For example, a standard 5-stage hypnogram (W, N1, N2, N3, REM) could be consolidated
        to a hypnogram more common with actigraphy (e.g. 2-stages: [Wake, Sleep] or
        4-stages: [W, Light, Deep, REM]).

        Parameters
        ----------
        self : yasa.Hypnogram
            Hypnogram, assumed to be already cropped to time in bed (TIB, also referred to as
            Total Recording Time, i.e. "lights out" to "lights on").
        new_n_stages : int
            Desired number of sleep stages. Must be lower than the current number of stages.

            - 5 stages - Wake, N1, N2, N3, REM
            - 4 stages - Wake, Light, Deep, REM
            - 3 stages - Wake, NREM, REM
            - 2 stages - Wake, Sleep

            .. note:: Unscored and Artefact are always allowed.

        Returns
        -------
        hyp : yasa.Hypnogram
            The consolidated Hypnogram object. This function returns a copy, i.e. the original
            hypnogram is not modified in place.

        Examples
        --------
        >>> from yasa import Hypnogram
        >>> hyp = Hypnogram(["W", "W", "N1", "N2", "N2", "N2", "N2", "W"], n_stages=5)
        >>> hyp_2s = hyp.consolidate_stages(2)
        >>> print(hyp_2s)
        Epoch
        0     WAKE
        1     WAKE
        2    SLEEP
        3    SLEEP
        4    SLEEP
        5    SLEEP
        6    SLEEP
        7     WAKE
        Name: Stage, dtype: object
        """
        assert self.n_stages in [3, 4, 5], "`self.n_stages` must be 3, 4, or 5"
        assert new_n_stages in [2, 3, 4], "`new_n_stages` must be 2, 3, or 4"
        assert new_n_stages < self.n_stages, "`new_n_stages` must be lower than `self.n_stages`"

        # Change sleep codes where applicable.
        if new_n_stages == 2:
            # Consolidate all Sleep
            mapping = {
                "N1": "S",
                "N2": "S",
                "N3": "S",
                "REM": "S",
                "LIGHT": "S",
                "DEEP": "S",
                "NREM": "S",
            }
        elif new_n_stages == 3:
            # Consolidate N1/N2/N3 or Light/Deep into NREM
            mapping = {"N1": "NREM", "N2": "NREM", "N3": "NREM", "LIGHT": "NREM", "DEEP": "NREM"}
        elif new_n_stages == 4:
            # Consolidate N1/N2 into Light
            mapping = {"N1": "LIGHT", "N2": "LIGHT", "N3": "DEEP"}
        new_hyp = self.hypno.replace(mapping).to_numpy()

        return Hypnogram(
            values=new_hyp,
            n_stages=new_n_stages,
            freq=self.freq,
            start=self.start,
            scorer=self.scorer,
        )

    def find_periods(self, threshold="5min", equal_length=False):
        """Find sequences of consecutive values exceeding a certain duration in hypnogram.

        Parameters
        ----------
        self : yasa.Hypnogram
            Hypnogram, assumed to be already cropped to time in bed (TIB, also referred to as
            Total Recording Time, i.e. "lights out" to "lights on").
        threshold : str
            This function will only keep periods that exceed a certain duration (default '5min'),
            e.g. '5min', '15min', '30sec', '1hour'. To disable thresholding, use '0sec'.
        equal_length : bool
            If True, the periods will all have the exact duration defined
            in threshold. That is, periods that are longer than the duration threshold will be
            divided into sub-periods of exactly the length of ``threshold``.

        Returns
        -------
        periods : :py:class:`pandas.DataFrame`
            Output dataframe

            * ``values`` : The value in hypno of the current period
            * ``start`` : The index of the start of the period in hypno
            * ``length`` : The duration of the period, in number of samples

        Examples
        --------
        Let's assume that we have an hypnogram where sleep = 1 and wake = 0, with one value
        per minute.

        >>> from yasa import Hypnogram
        >>> val = 11 * ["W"] + 3 * ["S"] + 2 * ["W"] + 9 * ["S"] + ["W", "W"]
        >>> hyp = Hypnogram(val, n_stages=2, freq="1min")
        >>> hyp.find_periods(threshold="0min")
          values  start  length
        0   WAKE      0      11
        1  SLEEP     11       3
        2   WAKE     14       2
        3  SLEEP     16       9
        4   WAKE     25       2

        This gives us the start and duration of each sequence of consecutive values in the
        hypnogram. For example, the first row tells us that there is a sequence of 11 consecutive
        WAKE starting at the first index of hypno.

        Now, we may want to keep only periods that are longer than a specific threshold,
        for example 5 minutes:

        >>> hyp.find_periods(threshold="5min")
          values  start  length
        0   WAKE      0      11
        1  SLEEP     16       9

        Only the two sequences that are longer than 5 minutes (11 minutes and 9 minutes
        respectively) are kept. Feel free to play around with different values of threshold!

        This function is not limited to binary arrays, e.g. a 5-stages hypnogram at 30-sec
        resolution:

        >>> val = ["W", "W", "W", "W", "N1"] + 6 * ["N2"] + ["W", "W", "W", "N1", "W", "N1"]
        >>> hyp = Hypnogram(val)
        >>> hyp.find_periods(threshold="2min")
          values  start  length
        0   WAKE      0       4
        1     N2      5       6

        Lastly, using ``equal_length=True`` will further divide the periods into segments of the
        same duration, i.e. the duration defined in ``threshold``:

        >>> hyp.find_periods(threshold="1min", equal_length=True)
          values  start  length
        0   WAKE      0       2
        1   WAKE      2       2
        2     N2      5       2
        3     N2      7       2
        4     N2      9       2
        5   WAKE     11       2

        Here, the first 2 minutes of consecutive WAKE are divided into 2 periods of exactly 1
        minute each. Next, the sequence of 6 consecutive 30-sec N2 epochs is further divided
        into 3 periods of 1 minute each. Lastly, the last value in the sequence of 3
        consecutive WAKE at the end of the hypnogram is removed to keep only a segment of exactly
        1 minute. In other words, the remainder of the division of a given segment by the desired
        duration is discarded.
        """
        return hypno_find_periods(
            self.hypno, self.sampling_frequency, threshold=threshold, equal_length=equal_length
        )

    def plot_hypnogram(self):
        """Plot the hypnogram."""
        # TODO: Add support for 2, 3 and 4-stages hypnogram
        raise NotImplementedError

    def simulate_similar(self, **kwargs):
        """Simulate a new hypnogram based on properties of the current hypnogram.

        .. seealso:: :py:func:`yasa.simulate_hypno`

        Parameters
        ----------
        self : :py:class:`yasa.Hypnogram`
            Hypnogram, assumed to be already cropped to time in bed (TIB).
        **kwargs : dict
            Optional keyword arguments passed to :py:func:`yasa.simulate_hypno`.

        Returns
        -------
        hyp : :py:class:`yasa.Hypnogram`
            A simulated hypnogram.

        Examples
        --------
        >>> import pandas as pd
        >>> from yasa import Hypnogram
        >>> hyp = Hypnogram(["W", "S", "W"], n_stages=2, freq="2min", scorer="Human").upsample("30s")
        >>> shyp = hyp.simulate_similar(scorer="Simulated", seed=6)
        >>> df = pd.concat([hyp.hypno, shyp.hypno], axis=1)
        >>> print(df)
               Human Simulated
        Epoch
        0       WAKE      WAKE
        1       WAKE      WAKE
        2       WAKE      WAKE
        3       WAKE      WAKE
        4      SLEEP     SLEEP
        5      SLEEP     SLEEP
        6      SLEEP     SLEEP
        7      SLEEP     SLEEP
        8       WAKE     SLEEP
        9       WAKE     SLEEP
        10      WAKE     SLEEP
        11      WAKE      WAKE
        """
        kwargs_ = {
            "tib": self.tib,
            "n_stages": self.n_stages,
            "freq": self.freq,
            "trans_probas": self.transition_matrix()[1],
            "start": self.start,
            "scorer": self.scorer,
        }
        if (n_stages := kwargs.pop("n_stages", None)) is not None:
            assert n_stages <= self.n_stages, "new n_stages must be <= original n_stages"
        kwargs_.update(kwargs)
        hyp = simulate_hypno(**kwargs_)
        if n_stages is not None and n_stages < self.n_stages:
            hyp = hyp.consolidate_stages(n_stages)
        return hyp

    def sleep_statistics(self):
        """
        Compute standard sleep statistics from an hypnogram.

        This function supports a 2, 3, 4 or 5-stages hypnogram.

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
        * Iber (2007). The AASM manual for the scoring of sleep and associated events: rules,
          terminology and technical specifications. American Academy of Sleep Medicine.

        * Silber et al. (2007). `The visual scoring of sleep in adults
          <https://www.ncbi.nlm.nih.gov/pubmed/17557422>`_. Journal of Clinical
          Sleep Medicine, 3(2), 121-131.

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

    def transition_matrix(self):
        """Create a state-transition matrix from an hypnogram.

        Parameters
        ----------
        self : yasa.Hypnogram
            Hypnogram, assumed to be already cropped to time in bed (TIB, also referred to as
            Total Recording Time, i.e. "lights out" to "lights on"). For best results, the
            hypnogram should not contain any artefact or unscored epochs.

        Returns
        -------
        counts : :py:class:`pandas.DataFrame`
            Counts transition matrix (number of transitions from stage A to stage B). The
            pre-transition states are the rows and the post-transition states are the columns.
        probs : :py:class:`pandas.DataFrame`
            Conditional probability transition matrix, i.e. given that current state is A, what is
            the probability that the next state is B. ``probs`` is a `right stochastic matrix
            <https://en.wikipedia.org/wiki/Stochastic_matrix>`_, i.e. each row sums to 1.

        Examples
        --------
        >>> from yasa import Hypnogram
        >>> values = (10 * ["W"] + 4 * ["N1"] + 1 * ["W"] + 30 * ["N2"] + 30 * ["N3"] + 5 * ["W"]
        ...           + 25 * ["REM"] + 15 * ["N2"] + 10 * ["W"])
        >>> hyp = Hypnogram(values, n_stages=5)
        >>> counts, probs = hyp.transition_matrix()
        >>> counts
        To Stage    WAKE  N1  N2  N3  REM
        From Stage
        WAKE          22   1   1   0    1
        N1             1   3   0   0    0
        N2             1   0  43   1    0
        N3             1   0   0  29    0
        REM            0   0   1   0   24

        >>> probs.round(3)
        To Stage     WAKE    N1     N2     N3   REM
        From Stage
        WAKE        0.880  0.04  0.040  0.000  0.04
        N1          0.250  0.75  0.000  0.000  0.00
        N2          0.022  0.00  0.956  0.022  0.00
        N3          0.033  0.00  0.000  0.967  0.00
        REM         0.000  0.00  0.040  0.000  0.96
        """
        counts, probs = transition_matrix(self.as_int())
        counts.index = counts.index.map(self.mapping_int)
        counts.columns = counts.columns.map(self.mapping_int)
        probs.index = probs.index.map(self.mapping_int)
        probs.columns = probs.columns.map(self.mapping_int)
        return counts, probs

    def upsample(self, new_freq, **kwargs):
        """Upsample hypnogram to a higher frequency.

        Parameters
        ----------
        self : yasa.Hypnogram
            Hypnogram, assumed to be already cropped to time in bed (TIB, also referred to as
            Total Recording Time, i.e. "lights out" to "lights on"). For best results, the
            hypnogram should not contain any artefact or unscored epochs.
        new_freq : str
            Frequency is defined with a pandas frequency string, e.g. "10s" or "1min".

        Returns
        -------
        hyp : yasa.Hypnogram
            The upsampled Hypnogram object. This function returns a copy, i.e. the original
            hypnogram is not modified in place.

        Examples
        --------
        Create a 30-sec hypnogram

        >>> from yasa import Hypnogram
        >>> hyp = Hypnogram(["W", "W", "S", "S", "W"], n_stages=2, start="2022-12-23 23:00")
        >>> hyp.hypno
        Time
        2022-12-23 23:00:00     WAKE
        2022-12-23 23:00:30     WAKE
        2022-12-23 23:01:00    SLEEP
        2022-12-23 23:01:30    SLEEP
        2022-12-23 23:02:00     WAKE
        Freq: 30S, Name: Stage, dtype: object

        Upsample to a 15-seconds resolution

        >>> hyp_up = hyp.upsample("15s")
        >>> hyp_up.hypno
        Time
        2022-12-23 23:00:00     WAKE
        2022-12-23 23:00:15     WAKE
        2022-12-23 23:00:30     WAKE
        2022-12-23 23:00:45     WAKE
        2022-12-23 23:01:00    SLEEP
        2022-12-23 23:01:15    SLEEP
        2022-12-23 23:01:30    SLEEP
        2022-12-23 23:01:45    SLEEP
        2022-12-23 23:02:00     WAKE
        2022-12-23 23:02:15     WAKE
        Freq: 15S, Name: Stage, dtype: object
        """
        assert pd.Timedelta(new_freq) < pd.Timedelta(self.freq), (
            f"The upsampling `new_freq` ({new_freq}) must be higher than the current frequency of "
            f"hypnogram {self.freq}"
        )
        if isinstance(self.hypno.index, pd.DatetimeIndex):
            # Upsampling should extend the last epoch, e.g.
            # - 30-sec: last epoch at 07:20:30
            # - 10-sec: last epoch should be 07:20:50 and not 07:20:30 otherwise we're losing 20 sec
            hyp_extend = self.hypno.copy()
            hyp_extend = hyp_extend.reindex(
                hyp_extend.index.union([hyp_extend.index[-1] + pd.Timedelta(self.freq)])
            ).ffill()
            new_hyp = hyp_extend.resample(new_freq, origin="start").ffill().iloc[:-1]
        else:
            hyp_extend = self.hypno.copy()
            hyp_extend.index = self.timedelta
            hyp_extend = hyp_extend.reindex(
                hyp_extend.index.union([hyp_extend.index[-1] + pd.Timedelta(self.freq)])
            ).ffill()
            new_hyp = (
                hyp_extend.resample(new_freq, origin="start")
                .ffill()
                .reset_index(drop=True)
                .iloc[:-1]
            )
            new_hyp.index.name = "Epoch"
        return Hypnogram(
            values=new_hyp,
            n_stages=self.n_stages,
            freq=new_freq,
            start=self.start,
            scorer=self.scorer,
        )

    def upsample_to_data(self, data, sf=None, verbose=True):
        """
        Upsample an hypnogram to a given sampling frequency and fit the resulting hypnogram to
        corresponding EEG data, such that the hypnogram and EEG data have the exact same number of
        samples.

        Parameters
        ----------
        self : yasa.Hypnogram
            Hypnogram, assumed to be already cropped to time in bed (TIB, also referred to as
            Total Recording Time, i.e. "lights out" to "lights on"). For best results, the
            hypnogram should not contain any artefact or unscored epochs.
        data : array_like or :py:class:`mne.io.BaseRaw`
            1D or 2D EEG data. Can also be a :py:class:`mne.io.BaseRaw`, in which case ``data``
            and ``sf_data`` will be automatically extracted.
        sf_data : float
            The sampling frequency of ``data``, in Hz (e.g. 100 Hz, 256 Hz, ...).
            Can be omitted if ``data`` is a :py:class:`mne.io.BaseRaw`.
        verbose : bool or str
            Verbose level. Default (False) will only print warning and error messages. The logging
            levels are 'debug', 'info', 'warning', 'error', and 'critical'. For most users the
            choice is between 'info' (or ``verbose=True``) and warning (``verbose=False``).

        Returns
        -------
        hypno : array_like
            The hypnogram values, upsampled to ``sf_data`` and cropped/padded to
            ``max(data.shape)``. For compatibility with most YASA functions, the returned hypnogram
            is an array with integer values, and not a :py:class:`yasa.Hypnogram` object.

        Warns
        -----
        UserWarning
            If the upsampled ``hypno`` is shorter / longer than ``max(data.shape)``
            and therefore needs to be padded/cropped respectively. This output can be disabled by
            passing ``verbose='ERROR'``.
        """
        hypno_up = hypno_upsample_to_data(
            self.as_int(), self.sampling_frequency, data=data, sf_data=sf, verbose=verbose
        )
        return hypno_up


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


def simulate_hypno(
    tib=90,
    n_stages=5,
    freq="30s",
    trans_probas=None,
    init_probas=None,
    seed=None,
    **kwargs,
):
    """Simulate a hypnogram based on transition probabilities.

    Current implentation is a naive Markov model. The initial stage of a hypnogram
    is generated using probabilites from ``init_probas`` and then subsequent stages
    are generated from a Markov sequence based on ``trans_probas``.

    .. important:: The Markov simulation model is not meant to accurately portray sleep
        macroarchitecture and should only be used for testing or other unique purposes.

    .. seealso:: :py:meth:`yasa.Hypnogram.simulate_similar`

    .. versionadded:: 0.6.3

    Parameters
    ----------
    tib : int, float
        Total duration of the hypnogram (i.e., time in bed), expressed in minutes.
        Returned hypnogram will be slightly shorter if ``tib`` is not evenly divisible by ``freq``.

        .. seealso:: :py:func:`yasa.sleep_statistics`
    n_stages : int
        Staging scheme of returned hypnogram. Input should follow 5-stage scheme but can
        be converted to lower scheme if desired.

        .. seealso:: :py:func:`yasa.hypno_consolidate_stages`
    freq : str
        A pandas frequency string indicating the frequency resolution of the hypnogram.
        See :py:class:`yasa.Hypnogram` for details.
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
    **kwargs : dict
        Other arguments that are passed to :py:class:`yasa.Hypnogram`.

    Returns
    -------
    hyp : :py:class:`yasa.Hypnogram`
        Hypnogram containing simulated sleep stages.

    Notes
    -----
    Default transition probabilities are based on 30-second epochs and can be found in the
    ``traMat_Epoch.npy`` file of Supplementary Information for Metzner et al., 2021 [Metzner2021]_
    (rounded values are viewable in Figure 5b). Please cite this work if these probabilites are used
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
    >>> hyp = simulate_hypno(tib=5, seed=1)
    >>> print(hyp)
    Epoch
    0    WAKE
    1      N1
    2      N1
    3      N2
    4      N2
    5      N2
    6      N2
    7      N2
    8      N2
    9      N2
    Name: Stage, dtype: object

    >>> hyp = simulate_hypno(tib=5, n_stages=2, seed=1)
    >>> print(hyp)
    Epoch
    0     WAKE
    1    SLEEP
    2    SLEEP
    3    SLEEP
    4    SLEEP
    5    SLEEP
    6    SLEEP
    7    SLEEP
    8    SLEEP
    9    SLEEP
    Name: Stage, dtype: object

    Add some Unscored epochs.
    >>> hyp = simulate_hypno(tib=5, n_stages=2, seed=1)
    >>> hyp.hypno.iloc[-2:] = "UNS"
    >>> print(hyp)
    Epoch
    0     WAKE
    1    SLEEP
    2    SLEEP
    3    SLEEP
    4    SLEEP
    5    SLEEP
    6    SLEEP
    7    SLEEP
    8      UNS
    9      UNS
    Name: Stage, dtype: object

    Base the data off a real subject's transition matrix.

    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import yasa
        >>> url = (
        >>>     "https://github.com/raphaelvallat/yasa/raw/master/"
        >>>     "notebooks/data_full_6hrs_100Hz_hypno_30s.txt"
        >>> )
        >>> values_int = np.loadtxt(url)
        >>> values_str = yasa.hypno_int_to_str(values_int)
        >>> hyp = yasa.Hypnogram(values_str)
        >>> shyp = hyp.simulate_similar(seed=2)
        >>> fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))
        >>> yasa.plot_hypnogram(hyp.as_int(), ax=ax1)
        >>> yasa.plot_hypnogram(shyp.as_int(), ax=ax2)
        >>> ax1.set_title("True hypnogram")
        >>> ax2.set_title("Simulated hypnogram")
        >>> plt.tight_layout()
    """
    # Validate input
    assert isinstance(tib, (int, float)) and tib > 0, "tib must be a number > 0"
    assert isinstance(n_stages, int) and (2 <= n_stages <= 5), "n_stages must be 2, 3, 4, or 5"
    assert isinstance(freq, str) and pd.Timedelta(freq) <= pd.Timedelta("30s"), (
        "freq must be a pandas frequency string and <= 30 seconds"
    )
    if seed is not None:
        assert isinstance(seed, int) and seed >= 0, "seed must be an integer >= 0"
    if trans_probas is not None:
        assert isinstance(trans_probas, pd.DataFrame), "trans_probas must be a pandas DataFrame"
        assert np.all(np.less_equal(trans_probas.shape, n_stages)), "too many trans_probas stages"
    if init_probas is not None:
        assert isinstance(init_probas, pd.Series), "init_probas must be a pandas Series"
        assert init_probas.size <= n_stages, "too many stages in init_probas"

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
        # Generate transition probability DataFrame
        trans_freqs = np.array(
            [
                [11737, 571, 84, 2, 2],
                [281, 6697, 1661, 11, 59],
                [253, 1070, 26259, 505, 272],
                [49, 176, 279, 9630, 12],
                [57, 189, 84, 2, 10071],
            ]
        )
        trans_probas = trans_freqs / trans_freqs.sum(axis=1, keepdims=True)
        trans_probas = pd.DataFrame(
            trans_probas,
            index=["WAKE", "N1", "N2", "N3", "REM"],
            columns=["WAKE", "N1", "N2", "N3", "REM"],
        )
        trans_probas.attrs = {"default": True}

    if init_probas is None:
        # Extract Wake row of initial probabilities as a Series
        for w in ["w", "W", "wake", "WAKE"]:
            if w in trans_probas.index:
                init_probas = trans_probas.loc[w, :].copy()
        assert init_probas is not None, "trans_probas must include 'WAKE' in the index"

    stage_order = init_probas.index.tolist()
    assert stage_order == trans_probas.index.tolist() == trans_probas.columns.tolist(), (
        "init_probas and trans_probas must all have matching indices"
    )

    # Extract probabilities as arrays
    trans_arr = trans_probas.to_numpy()
    init_arr = init_probas.to_numpy()

    # Make sure all rows sum to 1
    assert np.allclose(trans_arr.sum(axis=1), 1)
    assert np.isclose(init_arr.sum(), 1)

    # Find number of *complete* epochs within TIB duration
    freq_sec = 30 if trans_probas.attrs.get("default") else pd.Timedelta(freq).total_seconds()
    n_epochs = np.floor(tib * 60 / freq_sec).astype(int)

    # Generate hypnogram integer values
    values_int = _markov_sequence(init_arr, trans_arr, n_epochs)
    # Convert to hypnogram string values (based on indices)
    values_str = [ stage_order[x] for x in values_int ]

    # Create YASA hypnogram instance
    if trans_probas.attrs.get("default"):
        # If using default trans_probas, hyp must be initialized with 5 stages and 30s epochs
        hyp = Hypnogram(values_str, n_stages=5, freq="30s", **kwargs)
        if pd.Timedelta(freq) != pd.Timedelta("30s"):
            hyp = hyp.upsample(freq)
        if n_stages < 5:
            hyp = hyp.consolidate_stages(n_stages)
    else:
        hyp = Hypnogram(values_str, n_stages=n_stages, freq=freq, **kwargs)

    return hyp
