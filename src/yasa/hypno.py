"""
Hypnogram-related functions and class.
"""

import datetime
import logging
import warnings

import mne
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from yasa.evaluation import EpochByEpochAgreement
from yasa.io import set_log_level
from yasa.plotting import plot_hypnogram
from yasa.sleepstats import transition_matrix

__all__ = [
    "Hypnogram",
    "hypno_str_to_int",
    "hypno_int_to_str",
    "hypno_upsample_to_sf",
    "hypno_upsample_to_data",
    "hypno_find_periods",
    "load_profusion_hypno",
    "simulate_hypnogram",
]


logger = logging.getLogger("yasa")


class Hypnogram:
    """
    A class for manipulating hypnograms in YASA.

    Starting with v0.7, YASA takes a more object-oriented approach to hypnograms. That is,
    hypnograms are stored as a class (aka object), which comes with its own attributes and
    functions. Furthermore, YASA does not allow integer values to define the stages anymore.
    Instead, users must pass an array of strings with the actual stage names
    (e.g. ["WAKE", "WAKE", "N1", ..., "REM", "REM"]). If your hypnogram is encoded as integers, use
    :py:meth:`Hypnogram.from_integers` instead.

    .. versionadded:: 0.7.0

    Parameters
    ----------
    values : array_like
        A vector of stage values, represented as strings. See some examples below:

        * 2-stage hypnogram (Wake/Sleep): ``["W", "S", "S", "W", "S"]``
        * 3-stage (Wake/NREM/REM): ``pd.Series(["WAKE", "NREM", "NREM", "REM", "REM"])``
        * 4-stage (Wake/Light/Deep/REM): ``np.array(["Wake", "Light", "Deep", "Deep"])``
        * 5-stage (default): ``["N1", "N1", "N2", "N3", "N2", "REM", "W"]``

        Artefacts ("Art") and unscored ("Uns") epochs are always allowed regardless of the
        number of stages in the hypnogram.

        .. note:: Abbreviated or full spellings for the stages are allowed, as well as
            lower/upper/mixed case. Internally, YASA will convert the stages to full spelling
            and uppercase (e.g. "w" -> "WAKE").
    n_stages : int
        Whether ``values`` comes from a 2, 3, 4 or 5-stage hypnogram. Default is 5-stage, meaning
        that the following sleep stages are allowed: N1, N2, N3, REM, WAKE.
    freq : str
        A pandas frequency string indicating the frequency resolution of the hypnogram. Default is
        "30s" meaning that each value in the hypnogram represents a 30-seconds epoch.
        Examples: "1min", "10s", "15min". A full list of accepted values can be found in the
        `pandas offset aliases documentation
        <https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases>`_.

        ``freq`` will be passed to the :py:func:`pandas.date_range` function to create the time
        index of the hypnogram.
    start : str, :py:class:`datetime.datetime`, or :py:class:`pandas.Timestamp`, optional
        Start datetime of the hypnogram (e.g. ``"2022-12-15 22:30:00"``). When provided, the
        hypnogram index becomes a :py:class:`pandas.DatetimeIndex`, otherwise it is a
        :py:class:`pandas.RangeIndex` of epoch numbers. Accepts timezone-naive strings /
        datetimes as well as tz-aware :py:class:`~datetime.datetime` or
        :py:class:`~pandas.Timestamp` objects. When the start time comes from a scoring tool
        and is in local time (timezone-naive), pair it with ``tz`` so that
        :py:meth:`upsample_to_data` can align the hypnogram to MNE's UTC ``meas_date``.
    tz : str or :py:class:`datetime.timezone`, optional
        Timezone of the hypnogram ``start`` time, e.g. ``"Europe/Paris"`` or
        ``"America/New_York"``. Only used when ``start`` is timezone-naive.
        A full list of valid timezone strings is available via
        ``import zoneinfo; zoneinfo.available_timezones()``.
    scorer : str
        An optional string indicating the scorer name. If specified, this will be set as the name
        of the :py:class:`pandas.Series`, otherwise the name will be set to "Stage".
    proba : :py:class:`pandas.DataFrame`
        An optional dataframe with the probability of each sleep stage for each epoch in hypnogram.
        Each row must sum to 1. This is automatically included if the hypnogram is created with
        :py:class:`yasa.SleepStaging`.

    Attributes
    ----------
    hypno : :py:class:`pandas.Series`
        The hypnogram values as a categorical :py:class:`~pandas.Series`.
    n_epochs : int
        Number of epochs in the hypnogram.
    freq : str
        Frequency resolution of the hypnogram (e.g. ``'30s'``).
    sampling_frequency : float
        Sampling frequency of the hypnogram in Hz (e.g. ``1/30`` for 30-second epochs).
    start : :py:class:`pandas.Timestamp` or None
        Start datetime of the hypnogram. Tz-aware in local time when ``tz`` was provided or a
        tz-aware datetime was passed as ``start``, timezone-naive otherwise.
    timedelta : :py:class:`pandas.TimedeltaIndex`
        Elapsed time of each epoch relative to the first epoch.
    duration : float
        Total duration of the hypnogram in minutes (i.e., Time in Bed).
    n_stages : int
        Number of allowed sleep stages (2, 3, 4, or 5). Does not include ART and UNS.
    labels : list
        List of allowed stage label strings for this hypnogram.
    mapping : dict
        Mapping from stage string labels to integer values. Can be overridden by assignment.
    mapping_int : dict
        Reverse mapping from integer values to stage string labels.
    scorer : str or None
        Name of the scorer, if provided.
    proba : :py:class:`pandas.DataFrame` or None
        Per-epoch stage probabilities, if provided.

    Examples
    --------
    Create a 2-stage hypnogram

    >>> from yasa import Hypnogram
    >>> values = ["W", "W", "W", "S", "S", "S", "S", "S", "W", "S", "S", "S"]
    >>> hyp = Hypnogram(values, n_stages=2)
    >>> hyp
    <Hypnogram | 12 epochs x 30s (6.00 minutes), 2 unique stages>
     - Use `.hypno` to get the string values as a pandas.Series
     - Use `.as_int()` to get the integer values as a pandas.Series
     - Use `.plot_hypnogram()` to plot the hypnogram
    See the online documentation for more details.

    We can access the actual values, which are stored as a :py:class:`pandas.Series`, with:

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
    Name: Stage, dtype: category
    Categories (4, str): ['WAKE', 'SLEEP', 'ART', 'UNS']

    >>> # Number of epochs in the hypnogram
    >>> hyp.n_epochs
    12

    >>> # Total duration of the hypnogram, in minutes (12 epochs * 30 seconds = 6 minutes)
    >>> hyp.duration
    6.0

    >>> # Default mapping from strings to integers. Can be changed with `hyp.mapping = {}`
    >>> hyp.mapping
    {'WAKE': 0, 'SLEEP': 1, 'ART': -1, 'UNS': -2}

    >>> # Get the hypnogram Series integer values
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
    Name: Stage, dtype: int16

    >>> # Calculate the summary sleep statistics
    >>> import pandas as pd
    >>> pd.Series(hyp.sleep_statistics())
    TIB         6.0000
    SPT         4.5000
    WASO        0.5000
    TST         4.0000
    SE         66.6667
    SME        88.8889
    SFI         7.5000
    SOL         1.5000
    SOL_5min       NaN
    WAKE        2.0000
    dtype: float64

    >>> # Get the state-transition matrix
    >>> counts, probs = hyp.transition_matrix()
    >>> counts
    To Stage    WAKE  SLEEP
    From Stage
    WAKE           2      2
    SLEEP          1      6

    All these methods and properties are also valid with a 5-stage hypnogram. In the example below,
    we use the :py:func:`yasa.simulate_hypnogram` to generate a plausible 5-stage hypnogram with a
    30-seconds resolution. A random seed is specified to ensure that we get reproducible results.
    Lastly, we set an actual start time to the hypnogram. As a result, the index of the resulting
    hypnogram is a :py:class:`pandas.DatetimeIndex`.

    >>> from yasa import simulate_hypnogram
    >>> hyp = simulate_hypnogram(
    ...     tib=500, n_stages=5, start="2022-12-15 22:30:00", scorer="S1", seed=42
    ... )
    >>> hyp
    <Hypnogram | 1000 epochs x 30s (500.00 minutes), 5 unique stages, scored by S1>
     - Use `.hypno` to get the string values as a pandas.Series
     - Use `.as_int()` to get the integer values as a pandas.Series
     - Use `.plot_hypnogram()` to plot the hypnogram
    See the online documentation for more details.

    >>> hyp.hypno
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
    Freq: 30s, Name: S1, Length: 1000, dtype: category
    Categories (7, str): ['WAKE', 'N1', 'N2', 'N3', 'REM', 'ART', 'UNS']

    The summary sleep statistics will include more items with a 5-stage hypnogram than a 2-stage
    hypnogram, i.e. the amount and percentage of each sleep stage, the REM latency, etc.

    >>> pd.Series(hyp.sleep_statistics())
    TIB        500.0000
    SPT        497.5000
    WASO        79.5000
    TST        418.0000
    SE          83.6000
    SME         84.0201
    SFI          0.7177
    SOL          2.5000
    SOL_5min     2.5000
    Lat_REM     67.0000
    WAKE        82.0000
    N1          69.0000
    N2         247.0000
    N3          64.5000
    REM         37.5000
    %N1         16.5072
    %N2         59.0909
    %N3         15.4306
    %REM         8.9713
    dtype: float64
    """

    def __init__(
        self, values, n_stages=5, *, freq="30s", start=None, tz=None, scorer=None, proba=None
    ):
        assert isinstance(
            values, (list, np.ndarray, pd.Series, pd.api.extensions.ExtensionArray)
        ), "`values` must be a list, numpy.array or pandas.Series"
        assert all(isinstance(val, str) for val in values), (
            "Since v0.7, YASA expects strings to represent sleep stages, e.g. ['WAKE', 'N1', ...]. "
            "Please refer to the documentation for more details."
        )
        assert isinstance(n_stages, int), "`n_stages` must be an integer between 2 and 5."
        assert n_stages in [2, 3, 4, 5], "`n_stages` must be an integer between 2 and 5."
        assert isinstance(freq, str), "`freq` must be a pandas frequency string."
        assert isinstance(start, (type(None), str, pd.Timestamp, datetime.datetime)), (
            "`start` must be None, a string, a pandas.Timestamp, or a datetime.datetime."
        )
        assert isinstance(scorer, (type(None), str, int)), (
            "`scorer` must be either None, a string or an integer."
        )
        assert isinstance(proba, (pd.DataFrame, type(None))), (
            "`proba` must be either None or a pandas.DataFrame"
        )
        if n_stages == 2:
            accepted = ["W", "WAKE", "S", "SLEEP", "ART", "UNS"]
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
        n_unique_values = len(np.unique(values))
        if not all([val.upper() in accepted for val in values]):
            msg = (
                f"{np.unique(values)} do not match the accepted values for a {n_stages}-stage "
                f"hypnogram: {accepted}."
            )
            if n_unique_values < n_stages:
                msg += (
                    f"\nIf your hypnogram only has {n_unique_values} possible stages, make sure to "
                    f"specify `Hypnogram(values, n_stages={n_unique_values})`."
                )
            raise ValueError(msg)

        if isinstance(values, pd.api.extensions.ExtensionArray):
            # pandas 3.0 may return ArrowStringArray from .to_numpy() on Categorical series
            values = np.asarray(values, dtype=object)
        elif isinstance(values, pd.Series):
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
        # Change dtype of series to "categorical" (reduces memory)
        cat_dtype = CategoricalDtype(labels, ordered=False)
        hypno = hypno.astype(cat_dtype)
        # Normalize start to a pd.Timestamp and apply tz if provided
        if start is not None:
            start = pd.Timestamp(start)
            if tz is not None:
                if start.tzinfo is not None:
                    raise ValueError(
                        "`start` is already timezone-aware. Do not pass `tz` when `start` already "
                        "contains timezone information."
                    )
                start = start.tz_localize(tz)
        # Create Index
        if start is not None:
            hypno.index = pd.date_range(start=start, freq=freq, periods=hypno.shape[0])
            hypno.index.name = "Time"
            timedelta = hypno.index - hypno.index[0]
        else:
            fake_dt = pd.date_range(start="2022-12-03 00:00:00", freq=freq, periods=hypno.shape[0])
            hypno.index.name = "Epoch"
            timedelta = fake_dt - fake_dt[0]
        # Validate proba
        if proba is not None:
            assert proba.shape[1] > 0, "`proba` must have at least one column."
            assert proba.shape[0] == hypno.shape[0], "`proba` must have the same length as `values`"
            assert np.allclose(proba.sum(axis=1), 1), "Each row of `proba` must sum to 1."
            in_proba_but_not_labels = np.setdiff1d(proba.columns, labels)
            # in_labels_but_not_proba = np.setdiff1d(labels, proba.columns)
            assert not len(in_proba_but_not_labels), (
                f"Invalid stages in `proba`: {in_proba_but_not_labels}. The accepted stages are: "
                f"{labels}."
            )
            # Ensure same order as `labels`
            proba = proba.reindex(columns=labels).dropna(how="all", axis=1)
        # Set attributes
        self._hypno = hypno
        self._n_epochs = hypno.shape[0]
        self._freq = freq
        self._sampling_frequency = 1 / pd.Timedelta(freq).total_seconds()
        self._start = start
        self._timedelta = timedelta
        self._duration = self._n_epochs / (60 * self._sampling_frequency)
        self._n_stages = n_stages
        self._labels = labels
        self._mapping = mapping
        self._scorer = scorer
        self._proba = proba

    def __repr__(self):
        # TODO v0.8: Keep only the text between < and >
        text_scorer = f", scored by {self.scorer}" if self.scorer is not None else ""
        return (
            f"<Hypnogram | {self.n_epochs} epochs x {self.freq} ({self.duration:.2f} minutes), "
            f"{self.n_stages} unique stages{text_scorer}>\n"
            " - Use `.hypno` to get the string values as a pandas.Series\n"
            " - Use `.as_int()` to get the integer values as a pandas.Series\n"
            " - Use `.plot_hypnogram()` to plot the hypnogram\n"
            "See the online documentation for more details."
        )

    def __str__(self):
        return self.__repr__()

    @classmethod
    def from_integers(
        cls,
        values,
        mapping={0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R", -1: "Art", -2: "Uns"},
        n_stages=5,
        *,
        freq="30s",
        start=None,
        tz=None,
        scorer=None,
        proba=None,
    ):
        """Create a :py:class:`Hypnogram` from an integer-encoded hypnogram array.

        This is a convenience constructor for users migrating from the legacy integer-based API
        (e.g. ``[0, 1, 2, 3, 4]`` for Wake, N1, N2, N3, REM). Internally, it calls
        :py:func:`yasa.hypno_int_to_str` to convert integers to strings before creating the
        :py:class:`Hypnogram`.

        .. versionadded:: 0.7.0

        Parameters
        ----------
        values : array_like
            A 1D array of integer sleep stage values, e.g. ``np.array([0, 1, 1, 2, 3, 4])``.
        mapping : dict
            A dictionary mapping integer values to stage string labels. The default mapping is:

            .. code-block:: python

                {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R", -1: "Art", -2: "Uns"}

            Override this to support non-standard integer encodings.
        n_stages : int
            Whether ``values`` comes from a 2, 3, 4 or 5-stage hypnogram. Default is 5.
        freq : str
            Frequency resolution of the hypnogram. Default is ``"30s"``.
        start : str or datetime, optional
            Optional start datetime of the hypnogram (e.g. ``"2022-12-15 22:30:00"``).
        scorer : str, optional
            Optional scorer name.
        proba : :py:class:`pandas.DataFrame`, optional
            Optional dataframe of per-epoch stage probabilities.

        Returns
        -------
        hyp : :py:class:`Hypnogram`
            A :py:class:`Hypnogram` instance with string-encoded stages.

        Examples
        --------
        Convert a legacy integer hypnogram to a :py:class:`Hypnogram` object:

        >>> import numpy as np
        >>> from yasa import Hypnogram
        >>> int_hypno = np.array([0, 0, 1, 2, 3, 2, 4, 4, 0])
        >>> hyp = Hypnogram.from_integers(int_hypno)
        >>> hyp
        <Hypnogram | 9 epochs x 30s (4.50 minutes), 5 unique stages>
         - Use `.hypno` to get the string values as a pandas.Series
         - Use `.as_int()` to get the integer values as a pandas.Series
         - Use `.plot_hypnogram()` to plot the hypnogram
        See the online documentation for more details.

        >>> hyp.hypno
        Epoch
        0    WAKE
        1    WAKE
        2      N1
        3      N2
        4      N3
        5      N2
        6     REM
        7     REM
        8    WAKE
        Name: Stage, dtype: category
        Categories (7, str): ['WAKE', 'N1', 'N2', 'N3', 'REM', 'ART', 'UNS']

        Typical usage when loading a hypnogram from a plain-text file:

        >>> import numpy as np
        >>> from yasa import Hypnogram
        >>> int_hypno = np.loadtxt("path/to/hypnogram.txt").astype(int)  # doctest: +SKIP
        >>> hyp = Hypnogram.from_integers(
        ...     int_hypno, freq="30s", start="2022-12-15 22:30:00"
        ... )  # doctest: +SKIP

        Use a custom mapping to handle non-standard integer encodings:

        >>> # Hypnogram encoded as: 1=WAKE, 2=REM, 3=N1, 4=N2, 5=N3
        >>> custom_mapping = {1: "W", 2: "R", 3: "N1", 4: "N2", 5: "N3"}
        >>> hyp = Hypnogram.from_integers([1, 3, 4, 5, 2], mapping=custom_mapping)
        """
        str_hypno = hypno_int_to_str(values, mapping_dict=mapping)
        return cls(
            str_hypno, n_stages=n_stages, freq=freq, start=start, tz=tz, scorer=scorer, proba=proba
        )

    @property
    def hypno(self):
        """
        The hypnogram values, stored in a :py:class:`pandas.Series`. To reduce memory usage, the
        stages are stored as categories (:py:class:`pandas.Categorical`). ``hypno``
        inherits all the methods of a standard :py:class:`pandas.Series`, e.g. ``.describe()``,
        ``.unique()``, ``.to_csv()``, and more.

        .. note:: ``print(Hypnogram)`` is a shortcut to ``print(Hypnogram.hypno)``.
        """
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
    def duration(self):
        """Total duration of the hypnogram, expressed in minutes. AKA Time in Bed."""
        return self._duration

    @property
    def n_stages(self):
        """
        The number of allowed sleep stages in the hypnogram (2, 3, 4, or 5). This reflects the
        hypnogram type set at construction time, not the number of unique stages actually present
        in the data. For example, a hypnogram with only N2 and REM epochs will still report
        ``n_stages=5`` if it was created as a 5-stage hypnogram. Artefact (ART) and Unscored (UNS)
        are always allowed and are not counted.
        """
        return self._n_stages

    @property
    def labels(self):
        """The allowed stage labels."""
        return self._labels

    @property
    def mapping(self):
        """A dictionary with the mapping from string to integer values.

        Can be overridden by direct assignment, e.g. ``hyp.mapping = {"WAKE": 0, "SLEEP": 1}``.
        When setting a custom mapping, ``ART`` and ``UNS`` are automatically added with values
        ``-1`` and ``-2`` respectively, if they are not already present.
        """
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

    @property
    def proba(self):
        """
        If specified, a :py:class:`pandas.DataFrame` with the probability of each sleep stage
        for each epoch in hypnogram.
        """
        return self._proba

    # CLASS METHODS BELOW

    def as_events(self):
        """
        Return a pandas DataFrame summarizing epoch-level information.

        The ``onset`` and ``duration`` columns (in seconds) follow the
        `BIDS events file specification
        <https://bids-specification.readthedocs.io/en/stable/modality-agnostic-files/events.html>`_.
        The ``description`` column is compatible with
        `MNE Annotations <https://mne.tools/stable/generated/mne.Annotations.html>`_,
        making it straightforward to attach the hypnogram to an MNE
        :py:class:`~mne.io.Raw` object (see Examples below).

        Returns
        -------
        events : :py:class:`pandas.DataFrame`
            A dataframe containing epoch onset, duration, stage, etc.

        Examples
        --------
        >>> from yasa import Hypnogram
        >>> hyp = Hypnogram(["W", "W", "LIGHT", "LIGHT", "DEEP", "REM", "WAKE"], n_stages=4)
        >>> hyp.as_events()
               onset  duration  value description
        epoch
        0        0.0      30.0      0        WAKE
        1       30.0      30.0      0        WAKE
        2       60.0      30.0      2       LIGHT
        3       90.0      30.0      2       LIGHT
        4      120.0      30.0      3        DEEP
        5      150.0      30.0      4         REM
        6      180.0      30.0      0        WAKE

        To attach the hypnogram to an MNE :py:class:`~mne.io.Raw` object as
        annotations, pass the ``onset``, ``duration``, and ``description``
        columns to :py:class:`mne.Annotations` and call
        :py:meth:`~mne.io.Raw.set_annotations`:

        .. code-block:: python

            import mne

            events = hyp.as_events()
            annotations = mne.Annotations(
                onset=events["onset"],
                duration=events["duration"],
                description=events["description"],
            )
            raw.set_annotations(annotations)  # raw is an mne.io.Raw object
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
        """Return hypnogram values as integers.

        The default mapping from string to integer is:

        * 2-stage: {"WAKE": 0, "SLEEP": 1, "ART": -1, "UNS": -2}
        * 3-stage: {"WAKE": 0, "NREM": 2, "REM": 4, "ART": -1, "UNS": -2}
        * 4-stage: {"WAKE": 0, "LIGHT": 2, "DEEP": 3, "REM": 4, "ART": -1, "UNS": -2}
        * 5-stage: {"WAKE": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "ART": -1, "UNS": -2}

        Users can define a custom mapping:

        >>> hyp.mapping = {"WAKE": 0, "NREM": 1, "REM": 2}  # doctest: +SKIP

        Examples
        --------
        Convert a 2-stage hypnogram to a pandas.Series of integers

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
        Name: Stage, dtype: int16

        Same with a 4-stage hypnogram

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
        Name: Stage, dtype: int16
        """
        # Return as int16 (-32768 to 32767) to reduce memory usage
        return self.hypno.cat.rename_categories(self.mapping).astype(np.int16)

    def consolidate_stages(self, new_n_stages):
        """Reduce the number of stages in a hypnogram to match actigraphy or wearables.

        For example, a standard 5-stage hypnogram (W, N1, N2, N3, REM) could be consolidated
        to a hypnogram more common with actigraphy (e.g. 2-stage: [Wake, Sleep] or
        4-stage: [W, Light, Deep, REM]).

        Parameters
        ----------
        new_n_stages : int
            Desired number of sleep stages. Must be lower than the current number of stages.
            Valid target values and their stage sets are:

            - 4-stage (Wake, Light, Deep, REM)
            - 3-stage (Wake, NREM, REM)
            - 2-stage (Wake, Sleep)

            .. note:: Unscored and Artefact are always allowed.

        Returns
        -------
        hyp : :py:class:`yasa.Hypnogram`
            The consolidated Hypnogram object. This function returns a copy, i.e. the original
            hypnogram is not modified in place.

        Examples
        --------
        >>> from yasa import Hypnogram
        >>> hyp = Hypnogram(["W", "W", "N1", "N2", "N2", "N2", "N2", "W"], n_stages=5)
        >>> hyp_2s = hyp.consolidate_stages(2)
        >>> print(hyp_2s.hypno)
        Epoch
        0     WAKE
        1     WAKE
        2    SLEEP
        3    SLEEP
        4    SLEEP
        5    SLEEP
        6    SLEEP
        7     WAKE
        Name: Stage, dtype: category
        Categories (4, str): ['WAKE', 'SLEEP', 'ART', 'UNS']
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
        new_hyp = self.hypno.astype(object).replace(mapping).to_numpy()

        return Hypnogram(
            values=new_hyp,
            n_stages=new_n_stages,
            freq=self.freq,
            start=self.start,
            scorer=self.scorer,
            proba=None,  # TODO: Combine stages probability?
        )

    def copy(self):
        """Return a new copy of the current Hypnogram."""
        return type(self)(
            values=np.asarray(self.hypno, dtype=object),
            n_stages=self.n_stages,
            freq=self.freq,
            start=self.start,
            scorer=self.scorer,
            proba=self.proba,
        )

    def evaluate(self, obs_hyp):
        """Evaluate agreement between two hypnograms of the same sleep session.

        For example, the reference hypnogram (i.e., ``self``) might be a manually-scored hypnogram
        and the observed hypnogram (i.e., ``obs_hyp``) might be a hypnogram from actigraphy, a
        wearable device, or an automated scorer (e.g., :py:meth:`yasa.SleepStaging.predict`).

        Parameters
        ----------
        obs_hyp : :py:class:`yasa.Hypnogram`
            The observed or to-be-evaluated hypnogram.

        Returns
        -------
        ebe : :py:class:`yasa.EpochByEpochAgreement`
            See :py:class:`~yasa.EpochByEpochAgreement` documentation for more detail.

        Examples
        --------
        >>> from yasa import simulate_hypnogram
        >>> hyp_a = simulate_hypnogram(tib=90, scorer="AASM", seed=8)
        >>> hyp_b = hyp_a.simulate_similar(scorer="YASA", seed=9)
        >>> ebe = hyp_a.evaluate(hyp_b)
        >>> ebe.get_agreement().round(3)
        accuracy        0.550
        balanced_acc    0.355
        kappa           0.227
        mcc             0.231
        precision       0.515
        recall          0.550
        f1              0.524
        Name: agreement, dtype: float64
        """
        return EpochByEpochAgreement([self], [obs_hyp])

    def find_periods(self, threshold="5min", equal_length=False):
        """Find sequences of consecutive values exceeding a certain duration in hypnogram.

        Parameters
        ----------
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
            Output dataframe with one row per period and the following columns:

            * ``values`` (str): The stage label of the current period.
            * ``start`` (int): The index of the first epoch of the period in the hypnogram.
            * ``length`` (int): The duration of the period in number of epochs.

        Examples
        --------
        Let's assume that we have a hypnogram where sleep = 1 and wake = 0, with one value
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

        This function is not limited to binary arrays, e.g. a 5-stage hypnogram at 30-sec
        resolution:

        >>> from yasa import simulate_hypnogram
        >>> hyp = simulate_hypnogram(tib=30, seed=42)
        >>> hyp.find_periods(threshold="2min")
          values  start  length
        0   WAKE      0       5
        1     N1      5       6
        2     N2     11      49

        Lastly, using ``equal_length=True`` will further divide the periods into segments of the
        same duration, i.e. the duration defined in ``threshold``:

        >>> hyp.find_periods(threshold="5min", equal_length=True)
          values  start  length
        0     N2     11      10
        1     N2     21      10
        2     N2     31      10
        3     N2     41      10

        Here, the 24.5 minutes of consecutive N2 sleep (= 49 epochs) are divided into 4 periods of
        exactly 5 minute each. The remaining 4.5 minutes at the end of the hypnogram are removed
        because it is less than 5 minutes. In other words, the remainder of the division of a given
        segment by the desired duration is discarded.
        """
        return hypno_find_periods(
            self.hypno, self.sampling_frequency, threshold=threshold, equal_length=equal_length
        )

    def plot_hypnogram(self, **kwargs):
        """Plot the hypnogram.

        .. seealso:: :py:func:`yasa.plot_hypnogram`

        Parameters
        ----------
        **kwargs : dict
            Optional keyword arguments passed to :py:func:`yasa.plot_hypnogram`.

        Returns
        -------
        ax : :py:class:`matplotlib.axes.Axes`
            Matplotlib Axes

        Examples
        --------
        .. plot::

            >>> from yasa import simulate_hypnogram
            >>> ax = simulate_hypnogram(tib=480, seed=88).plot_hypnogram(highlight="REM")
        """
        return plot_hypnogram(self, **kwargs)

    def simulate_similar(self, **kwargs):
        """Simulate a new hypnogram based on properties of the current hypnogram.

        .. seealso:: :py:func:`yasa.simulate_hypnogram`

        Parameters
        ----------
        **kwargs : dict
            Optional keyword arguments passed to :py:func:`yasa.simulate_hypnogram`.

        Returns
        -------
        hyp : :py:class:`yasa.Hypnogram`
            A simulated hypnogram.

        Examples
        --------
        >>> import pandas as pd
        >>> from yasa import Hypnogram
        >>> hyp = Hypnogram(["W", "S", "W"], n_stages=2, freq="2min", scorer="Human").upsample(
        ...     "30s"
        ... )
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
        assert "n_stages" not in kwargs and "freq" not in kwargs, (
            "`n_stages` and `freq` cannot be included as additional `**kwargs` "
            "because they must match properties of the current Hypnogram."
        )
        simulate_hypnogram_kwargs = {
            "tib": self.duration,
            "n_stages": self.n_stages,
            "freq": self.freq,
            "trans_probas": self.transition_matrix()[1],
            "start": self.start,
            "scorer": self.scorer,
        }
        simulate_hypnogram_kwargs.update(kwargs)
        return simulate_hypnogram(**simulate_hypnogram_kwargs)

    def sleep_statistics(self):
        """
        Compute standard sleep statistics from a hypnogram.

        This function supports a 2, 3, 4 or 5-stage hypnogram.

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

        .. warning::
            The definition of REM latency in the AASM scoring manual differs from the REM latency
            reported here. The former uses the time from first epoch of sleep, while YASA uses the
            time from the beginning of the recording. The AASM definition of the REM latency can be
            found with `Lat_REM - SOL`.

        References
        ----------
        .. [Iber2007] Iber (2007). The AASM manual for the scoring of sleep and associated events:
                      rules, terminology and technical specifications. American Academy of Sleep
                      Medicine.

        .. [Silber2007] Silber et al. (2007). `The visual scoring of sleep in adults
                        <https://www.ncbi.nlm.nih.gov/pubmed/17557422>`_. Journal of Clinical
                        Sleep Medicine, 3(2), 121-131.

        Examples
        --------
        Sleep statistics for a 2-stage hypnogram with a resolution of 15-seconds

        >>> import pandas as pd
        >>> from yasa import Hypnogram
        >>> # Generate a fake hypnogram, where "S" = Sleep, "W" = Wake
        >>> values = 10 * ["W"] + 40 * ["S"] + 5 * ["W"] + 40 * ["S"] + 9 * ["W"]
        >>> hyp = Hypnogram(values, freq="15s", n_stages=2)
        >>> pd.Series(hyp.sleep_statistics())
        TIB         26.0000
        SPT         21.2500
        WASO         1.2500
        TST         20.0000
        SE          76.9231
        SME         94.1176
        SFI          1.5000
        SOL          2.5000
        SOL_5min     2.5000
        WAKE         6.0000
        dtype: float64

        Sleep statistics for a 5-stage hypnogram

        >>> from yasa import simulate_hypnogram
        >>> # Generate a 8 hr (= 480 minutes) 5-stage hypnogram with a 30-seconds resolution
        >>> hyp = simulate_hypnogram(tib=480, seed=42)
        >>> pd.Series(hyp.sleep_statistics())
        TIB        480.0000
        SPT        477.5000
        WASO        79.5000
        TST        398.0000
        SE          82.9167
        SME         83.3508
        SFI          0.7538
        SOL          2.5000
        SOL_5min     2.5000
        Lat_REM     67.0000
        WAKE        82.0000
        N1          67.0000
        N2         240.5000
        N3          53.0000
        REM         37.5000
        %N1         16.8342
        %N2         60.4271
        %N3         13.3166
        %REM         9.4221
        dtype: float64
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
        """Create a state-transition matrix from a hypnogram.

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
        >>> from yasa import Hypnogram, simulate_hypnogram
        >>> # Generate a 8 hr (= 480 minutes) 5-stage hypnogram with a 30-seconds resolution
        >>> hyp = simulate_hypnogram(tib=480, seed=42)
        >>> counts, probs = hyp.transition_matrix()
        >>> counts
        To Stage    WAKE  N1   N2  N3  REM
        From Stage
        WAKE         153  11    0   0    0
        N1             6  99   29   0    0
        N2             3  16  447  10    5
        N3             1   3    4  96    1
        REM            0   5    1   0   69

        >>> probs.round(3)
        To Stage     WAKE     N1     N2     N3   REM
        From Stage
        WAKE        0.933  0.067  0.000  0.000  0.00
        N1          0.045  0.739  0.216  0.000  0.00
        N2          0.006  0.033  0.929  0.021  0.01
        N3          0.010  0.029  0.038  0.914  0.01
        REM         0.000  0.067  0.013  0.000  0.92
        """
        counts, probs = transition_matrix(self.as_int())
        counts.index = counts.index.map(self.mapping_int)
        counts.columns = counts.columns.map(self.mapping_int)
        probs.index = probs.index.map(self.mapping_int)
        probs.columns = probs.columns.map(self.mapping_int)
        return counts, probs

    def upsample(self, new_freq):
        """Upsample hypnogram to a higher frequency.

        Parameters
        ----------
        new_freq : str
            Target frequency as a pandas frequency string (e.g. ``"10s"`` or ``"1min"``). Must
            represent a higher sampling rate than the current hypnogram frequency, i.e. a shorter
            epoch duration (e.g. ``"10s"`` when the current frequency is ``"30s"``).

        Returns
        -------
        hyp : :py:class:`yasa.Hypnogram`
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
        Freq: 30s, Name: Stage, dtype: category
        Categories (4, str): ['WAKE', 'SLEEP', 'ART', 'UNS']

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
        Freq: 15s, Name: Stage, dtype: category
        Categories (4, str): ['WAKE', 'SLEEP', 'ART', 'UNS']
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
            proba=None,  # NOTE: Do not upsample probability
        )

    def upsample_to_data(self, data, sf=None, verbose=True):
        """
        Upsample a hypnogram to a given sampling frequency and fit the resulting hypnogram to
        corresponding EEG data, such that the hypnogram and EEG data have the exact same number of
        samples.

        When the hypnogram has a ``start`` attribute set **and** ``data`` is a
        :py:class:`mne.io.BaseRaw` with a valid ``meas_date``, the alignment is based on the
        actual recording timestamps rather than just the number of samples. This means the correct
        hypnogram epochs are selected even when the hypnogram and the recording do not share the
        same start time (e.g. the recording is a segment extracted from a full-night file).

        For timestamp-aware alignment to work, ``self.start`` must be timezone-aware. Pass ``tz``
        when creating the :py:class:`Hypnogram` (e.g. ``Hypnogram(..., tz='Europe/Paris')``) or
        pass a tz-aware datetime directly as ``start``.

        Parameters
        ----------
        data : array_like or :py:class:`mne.io.BaseRaw`
            1D or 2D EEG data. Can also be a :py:class:`mne.io.BaseRaw`, in which case ``data``
            and ``sf`` will be automatically extracted.
        sf : float
            The sampling frequency of ``data``, in Hz (e.g. 100 Hz, 256 Hz, ...).
            Can be omitted if ``data`` is a :py:class:`mne.io.BaseRaw`.
        verbose : bool or str
            Verbose level. Default (False) will only print warning and error messages. The logging
            levels are 'debug', 'info', 'warning', 'error', and 'critical'. For most users the
            choice is between 'info' (or ``verbose=True``) and warning (``verbose=False``).

        Returns
        -------
        hypno : :py:class:`numpy.ndarray`
            The hypnogram values as a 1D integer array, upsampled to ``sf`` Hz and
            cropped/padded to ``max(data.shape)`` samples. For compatibility with most YASA
            functions, integer values are returned rather than a :py:class:`yasa.Hypnogram` object.

        Raises
        ------
        ValueError
            If timestamp-aware alignment is triggered but ``self.start`` is timezone-naive while
            ``raw.meas_date`` is timezone-aware (UTC). Fix by passing ``tz`` at construction:
            ``Hypnogram(..., tz='Europe/Paris')``.

        Warns
        -----
        UserWarning
            If the upsampled ``hypno`` is shorter or longer than ``max(data.shape)``
            and therefore needs to be padded or cropped respectively. This warning can be silenced
            by passing ``verbose='error'``.

        Examples
        --------
        Upsample a 30-second hypnogram to match 100 Hz EEG data:

        >>> import numpy as np
        >>> from yasa import Hypnogram
        >>> hyp = Hypnogram(["W", "W", "N1", "N2", "N2", "REM"], freq="30s")
        >>> # Simulate 3 minutes of EEG data at 100 Hz (= 18000 samples)
        >>> data = np.zeros((1, 18000))
        >>> hypno = hyp.upsample_to_data(data, sf=100)
        >>> hypno.shape
        (18000,)
        >>> np.unique(hypno)
        array([0, 1, 2, 4], dtype=int16)

        Timestamp-aware upsampling for a cropped MNE Raw recording. Here the hypnogram covers a
        full 5-minute night starting at 23:00 local time (Paris, CET = UTC+1), but the EEG
        recording only captures the last 3 minutes (starting 2 minutes in). Passing ``tz`` at
        construction means no timezone handling is needed at upsampling time:

        >>> import mne
        >>> import datetime
        >>> stages = ["W", "W", "N1", "N2", "N2", "N3", "N3", "REM", "REM", "W"]
        >>> hyp_ts = Hypnogram(stages, freq="30s", start="2024-01-15 23:00:00", tz="Europe/Paris")
        >>> hyp_ts.start  # stored as tz-aware local time
        Timestamp('2024-01-15 23:00:00+0100', tz='Europe/Paris')
        >>> info = mne.create_info(["EEG"], sfreq=100, ch_types=["eeg"], verbose=False)
        >>> raw = mne.io.RawArray(np.zeros((1, 18000)), info, verbose=False)
        >>> # meas_date is UTC: 22:02 UTC = 23:02 CET, i.e. 2 min (4 epochs) into the hypnogram
        >>> meas_date = datetime.datetime(2024, 1, 15, 22, 2, 0, tzinfo=datetime.timezone.utc)
        >>> _ = raw.set_meas_date(meas_date)
        >>> hypno_ts = hyp_ts.upsample_to_data(raw)
        >>> hypno_ts.shape
        (18000,)
        >>> np.unique(hypno_ts)  # epochs 4–9: N2, N3, N3, REM, REM, W
        array([0, 2, 3, 4], dtype=int16)
        """
        if (
            self.start is not None
            and isinstance(data, mne.io.BaseRaw)
            and data.info["meas_date"] is not None
        ):
            return self._upsample_to_raw_timestamps(data, verbose=verbose)
        hypno_up = hypno_upsample_to_data(
            self.as_int(), self.sampling_frequency, data=data, sf_data=sf, verbose=verbose
        )
        return hypno_up

    def _upsample_to_raw_timestamps(self, raw, verbose=True):
        """Timestamp-aware upsampling for MNE Raw objects with a valid meas_date.

        Internal method called by :py:meth:`upsample_to_data` when both ``self.start`` and
        ``raw.meas_date`` are available. Aligns the hypnogram to the recording based on wall-clock
        time rather than sample count.
        """
        set_log_level(verbose)
        sf_data = raw.info["sfreq"]
        epoch_dur = 1.0 / self.sampling_frequency  # seconds per epoch, e.g. 30.0

        # --- resolve and align timestamps ---
        # self.start is a pd.Timestamp (tz-aware when tz was passed at construction, else naive)
        hyp_start = self.start
        raw_start = pd.Timestamp(raw.info["meas_date"])  # always UTC-aware

        if hyp_start.tzinfo is None and raw_start.tzinfo is not None:
            # Hypnogram start is naive — tz was not specified at construction
            raise ValueError(
                "The hypnogram `start` time is timezone-naive but `raw.meas_date` is "
                "timezone-aware (UTC). YASA cannot safely align the two without knowing the "
                "timezone of the hypnogram. Pass `tz` when creating the Hypnogram, e.g.:\n"
                "    Hypnogram(..., tz='Europe/Paris')\n"
                "Or pass a tz-aware datetime as `start` directly. Available timezone strings: "
                "import zoneinfo; zoneinfo.available_timezones()"
            )
        elif hyp_start.tzinfo is not None and raw_start.tzinfo is None:
            # Unusual: hypnogram aware, meas_date naive — strip tz for arithmetic
            hyp_start = hyp_start.tz_convert("UTC").tz_localize(None)
        # else: both naive or both aware — subtraction works directly

        # --- compute epoch offset ---
        offset_sec = (raw_start - hyp_start).total_seconds()
        epoch_offset_float = offset_sec / epoch_dur
        epoch_offset = int(round(epoch_offset_float))

        if abs(epoch_offset_float - epoch_offset) > 1e-6:
            logger.warning(
                "The offset between hypnogram start and recording start (%.6f s) is not a "
                "whole number of %g-second epochs. Rounding to the nearest epoch (%d)."
                % (offset_sec, epoch_dur, epoch_offset)
            )

        logger.info(
            "Timestamp-aware upsampling: recording starts %.3f s after hypnogram start "
            "(%d epoch(s) of %g s)." % (offset_sec, epoch_offset, epoch_dur)
        )

        # --- build the aligned integer hypnogram ---
        hypno_int = self.as_int().to_numpy()
        uns_val = np.int16(self.mapping.get("UNS", -2))

        if epoch_offset >= 0:
            # Recording starts at or after hypnogram start: drop leading epochs
            hypno_sliced = hypno_int[epoch_offset:]
        else:
            # Recording starts before hypnogram start: prepend Unscored epochs
            n_prepend = -epoch_offset
            logger.warning(
                "The recording starts %.3f s before the hypnogram start. "
                "Prepending %d Unscored (UNS) epoch(s)." % (-offset_sec, n_prepend)
            )
            prepend = np.full(n_prepend, uns_val, dtype=np.int16)
            hypno_sliced = np.concatenate([prepend, hypno_int])

        # --- upsample and fit to exact sample count ---
        hypno_up = hypno_upsample_to_sf(
            hypno=hypno_sliced, sf_hypno=self.sampling_frequency, sf_data=sf_data
        )
        return hypno_fit_to_data(hypno=hypno_up, data=raw, sf=sf_data)


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

    .. deprecated:: 0.7.0
        Use :py:class:`yasa.Hypnogram` and its :py:meth:`~yasa.Hypnogram.as_int` method instead.
        This function will be removed in v0.8.

    .. versionadded:: 0.1.5

    Parameters
    ----------
    hypno : array_like
        The sleep staging (hypnogram) 1D array.
    mapping_dict : dict
        The mapping dictionnary, in lowercase. Note that this function is essentially a wrapper
        around :py:meth:`pandas.Series.map`.

    Returns
    -------
    hypno : array_like
        The corresponding integer hypnogram.
    """
    warnings.warn(
        "The `yasa.hypno_str_to_int` function is deprecated and will be removed in v0.8. "
        "Please use the `yasa.Hypnogram` class and its `.as_int()` method instead.",
        FutureWarning,
        stacklevel=2,
    )
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
    -------
    hypno : array_like
        The corresponding string hypnogram.

    See Also
    --------
    :py:meth:`yasa.Hypnogram.from_integers` : Convenience constructor that combines this
        conversion with :py:class:`yasa.Hypnogram` creation in a single step.
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
    # warnings.warn(
    #     "The `yasa.hypno_upsample_to_sf` function is deprecated and will be removed in v0.8. "
    #     "Please use the `yasa.Hypnogram.upsample` method instead.",
    #     FutureWarning,
    # )
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
    # NOTE: FutureWarning not added here otherwise it would also be shown when calling
    # yasa.Hypnogram.upsample_to_data.
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
    # TODO: This should return a yasa.Hypnogram object
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
    >>> yasa.hypno_find_periods(hypno, sf_hypno=1 / 60, threshold="0min")
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

    >>> yasa.hypno_find_periods(hypno, sf_hypno=1 / 60, threshold="5min")
       values  start  length
    0       0      0      11
    1       1     16       9

    Only the two sequences that are longer than 5 minutes (11 minutes and 9 minutes respectively)
    are kept. Feel free to play around with different values of threshold!

    This function is not limited to binary arrays, e.g.

    >>> hypno = [0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 0, 1]
    >>> yasa.hypno_find_periods(hypno, sf_hypno=1 / 60, threshold="2min")
       values  start  length
    0       0      0       4
    1       2      5       6
    2       0     11       3

    Lastly, using ``equal_length=True`` will further divide the periods into segments of the
    same duration, i.e. the duration defined in ``threshold``:

    >>> hypno = [0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 0, 1]
    >>> yasa.hypno_find_periods(hypno, sf_hypno=1 / 60, threshold="2min", equal_length=True)
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
    # NOTE: FutureWarning not added here otherwise it would also be shown when calling
    # yasa.Hypnogram.find_periods
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
# SIMULATION
#############################################################################


def simulate_hypnogram(
    tib=480,
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
        Default is 480 minutes (= 8 hours).

        .. seealso:: :py:func:`yasa.sleep_statistics`
    trans_probas : :py:class:`pandas.DataFrame` or None
        Transition probability matrix where each cell is a transition probability
        between sleep stages of consecutive *epochs*.

        ``trans_probas`` is a `right stochastic matrix
        <https://en.wikipedia.org/wiki/Stochastic_matrix>`_, i.e. each row sums to 1.

        If None (default), transition probabilities come from Metzner et al., 2021 [Metzner2021]_.
        If :py:class:`pandas.DataFrame`, must have "from"-stages as indices and
        "to"-stages as columns. Indices and columns must follow YASA string
        hypnogram convention (e.g., WAKE, N1). Unscored/Artefact stages are not allowed.

        .. note:: Transition probability matrices should indicate the transition
            probability between *epochs* (i.e., probability of the next epoch) and
            not simply stage (i.e., probability of non-similar stage).

        .. seealso:: Return value from :py:func:`yasa.transition_matrix`
    init_probas : :py:class:`pandas.Series` or None
        Probabilites of each stage to initialize random walk.
        If None (default), initialize with "from"-WAKE row of ``trans_probas``.
        If :py:class:`pandas.Series`, indices must be stages following YASA string
        hypnogram convention and identical to those of ``trans_probas``.
    seed : int or None
        Random seed for generating Markov sequence.
        If an integer number is provided, the random hypnogram will be predictable.
        This argument is required if reproducible results are desired.
    **kwargs : dict
        Other arguments that are passed to :py:class:`yasa.Hypnogram`.

        .. note:: `n_stages` and `freq` must be consistent with a user-specificied `trans_probas`.

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
    >>> from yasa import simulate_hypnogram
    >>> hyp = simulate_hypnogram(tib=5, seed=1)
    >>> hyp
    <Hypnogram | 10 epochs x 30s (5.00 minutes), 5 unique stages>
     - Use `.hypno` to get the string values as a pandas.Series
     - Use `.as_int()` to get the integer values as a pandas.Series
     - Use `.plot_hypnogram()` to plot the hypnogram
    See the online documentation for more details.

    >>> hyp.hypno
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
    Name: Stage, dtype: category
    Categories (7, str): ['WAKE', 'N1', 'N2', 'N3', 'REM', 'ART', 'UNS']

    >>> hyp = simulate_hypnogram(tib=5, n_stages=2, seed=1)
    >>> hyp.hypno
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
    Name: Stage, dtype: category
    Categories (4, str): ['WAKE', 'SLEEP', 'ART', 'UNS']

    Add some Unscored epochs.

    >>> hyp = simulate_hypnogram(tib=5, n_stages=2, seed=1)
    >>> hyp.hypno.iloc[-2:] = "UNS"
    >>> hyp.hypno
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
    Name: Stage, dtype: category
    Categories (4, str): ['WAKE', 'SLEEP', 'ART', 'UNS']

    Base the data off a real subject's transition matrix.

    .. plot::

        >>> import numpy as np
        >>> import yasa
        >>> import matplotlib.pyplot as plt
        >>> from yasa import Hypnogram, hypno_int_to_str
        >>> values_str = hypno_int_to_str(
        ...     np.loadtxt(yasa.fetch_sample("full_6hrs_100Hz_hypno_30s.txt"))
        ... )
        >>> real_hyp = Hypnogram(values_str)
        >>> fake_hyp = real_hyp.simulate_similar(seed=2)
        >>> fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(7, 5))
        >>> real_hyp.plot_hypnogram(ax=ax1).set_title("Real hypnogram")  # doctest: +SKIP
        >>> fake_hyp.plot_hypnogram(ax=ax2).set_title("Fake hypnogram")  # doctest: +SKIP
        >>> plt.tight_layout()
    """
    # Extract yasa.Hypnogram defaults, which will be assumed later but need throughout
    if "n_stages" not in kwargs:
        kwargs["n_stages"] = 5
    if "freq" not in kwargs:
        kwargs["freq"] = "30s"
    # Validate input
    assert isinstance(tib, (int, float)) and tib > 0, "`tib` must be a number > 0"
    if trans_probas is not None:
        assert isinstance(trans_probas, pd.DataFrame), "`trans_probas` must be a pandas DataFrame"
        assert np.all(np.less_equal(trans_probas.shape, kwargs["n_stages"])), (
            "user-specified `trans_probas` must not include more stages than `n_stages`"
        )
    if init_probas is not None:
        assert isinstance(init_probas, pd.Series), "`init_probas` must be a pandas Series"
    if seed is not None:
        assert isinstance(seed, int) and seed >= 0, "`seed` must be an integer >= 0"
    if trans_probas is None:
        # Check this here, rather than letting hyp.upsample catch it, to be clear about reason
        assert pd.Timedelta(kwargs["freq"]) <= pd.Timedelta("30s"), (
            "`freq` must be <= 30s when using default `trans_probas`"
        )

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
        trans_probas.attrs = {"is_default": True}
    else:
        trans_probas.attrs = {"is_default": False}

    if init_probas is None:
        # Extract Wake row of initial probabilities as a Series
        for w in ["w", "W", "wake", "WAKE"]:
            if w in trans_probas.index:
                init_probas = trans_probas.loc[w, :].copy()
        assert init_probas is not None, "`trans_probas` must include 'WAKE' in the index"

    stage_order = init_probas.index.tolist()
    assert stage_order == trans_probas.index.tolist() == trans_probas.columns.tolist(), (
        "`init_probas` and `trans_probas` must have all matching indices"
    )

    # Extract probabilities as arrays
    trans_arr = trans_probas.to_numpy()
    init_arr = init_probas.to_numpy()

    # Make sure all rows sum to 1
    assert np.allclose(trans_arr.sum(axis=1), 1), "All rows of `trans_probas` must sum to 1"
    assert np.isclose(init_arr.sum(), 1), "`init_probas` must sum to 1"

    # Find number of *complete* epochs within TIB duration to simulate
    if trans_probas.attrs.get("is_default"):
        freq_sec = 30
    else:
        freq_sec = pd.Timedelta(kwargs["freq"]).total_seconds()
    n_epochs = np.floor(tib * 60 / freq_sec).astype(int)

    # Generate hypnogram integer values
    values_int = _markov_sequence(init_arr, trans_arr, n_epochs)
    # Convert to hypnogram string values (based on indices)
    values_str = [stage_order[x] for x in values_int]

    # Create YASA hypnogram instance
    if trans_probas.attrs.get("is_default"):
        # If using default trans_probas, hyp *must* be initialized with 5 stages and 30s epochs
        n_stages = kwargs.pop("n_stages")
        freq = kwargs.pop("freq")
        hyp = Hypnogram(values_str, n_stages=5, freq="30s", **kwargs)
        if pd.Timedelta(freq) < pd.Timedelta("30s"):
            hyp = hyp.upsample(freq)
        if n_stages != 5:
            hyp = hyp.consolidate_stages(n_stages)
    else:
        hyp = Hypnogram(values_str, **kwargs)

    return hyp
