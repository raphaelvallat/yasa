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

from .evaluation import EpochByEpochAgreement
from .io import set_log_level
from .plotting import _plot_hypnogram

__all__ = [
    "Hypnogram",
    "hypno_str_to_int",
    "hypno_int_to_str",
    "hypno_upsample_to_sf",
    "hypno_upsample_to_data",
    "hypno_find_periods",
    "simulate_hypnogram",
]


logger = logging.getLogger("yasa")

# Single source of truth for the stage vocabulary. For each n_stages, the canonical labels (in
# display order) and their default integer encoding. ART and UNS are always allowed.
_STAGE_MAPPINGS = {
    2: {"WAKE": 0, "SLEEP": 1, "ART": -1, "UNS": -2},
    3: {"WAKE": 0, "NREM": 2, "REM": 4, "ART": -1, "UNS": -2},
    4: {"WAKE": 0, "LIGHT": 2, "DEEP": 3, "REM": 4, "ART": -1, "UNS": -2},
    5: {"WAKE": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "ART": -1, "UNS": -2},
}
# Accepted abbreviations, converted to the full canonical label
_STAGE_ALIASES = {"W": "WAKE", "S": "SLEEP", "R": "REM"}
# Stages that count as sleep / non-sleep, regardless of n_stages
_SLEEP_STAGES = ["SLEEP", "N1", "N2", "N3", "NREM", "REM", "LIGHT", "DEEP"]
_NON_SLEEP_STAGES = ["WAKE", "ART", "UNS"]
# Mapping from finer to coarser stages, used by Hypnogram.consolidate_stages
_CONSOLIDATION_MAPPINGS = {
    2: {s: "SLEEP" for s in ["N1", "N2", "N3", "REM", "LIGHT", "DEEP", "NREM"]},
    3: {"N1": "NREM", "N2": "NREM", "N3": "NREM", "LIGHT": "NREM", "DEEP": "NREM"},
    4: {"N1": "LIGHT", "N2": "LIGHT", "N3": "DEEP"},
}
# Default integer to string mapping of legacy integer hypnograms
_DEFAULT_INT_TO_STR = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R", -1: "Art", -2: "Uns"}
# Compumedics Profusion integer stages to YASA: S4 -> N3, REM (5) -> 4, Active (9) -> WAKE
_PROFUSION_TO_YASA = {4: 3, 5: 4, 9: 0}
_TIME_TYPES = (str, pd.Timestamp, datetime.datetime)


def _canonical_stage(stage):
    """Convert a stage label to its canonical uppercase full spelling (e.g. "w" -> "WAKE")."""
    stage = str(stage).upper()
    return _STAGE_ALIASES.get(stage, stage)


class Hypnogram:
    """
    Standard class for representing and analyzing a sleep hypnogram.

    A ``Hypnogram`` is a sequence of sleep stage labels sampled at a fixed epoch duration (default
    30 seconds). Three assumptions underpin every method in the class:

    1. **Uniform epoch duration.** Every epoch has the same length, set once via ``freq``.
       Variable-length epochs are not supported.
    2. **Contiguous recording.** Epochs are assumed to be consecutive with no temporal gaps.
    3. **Closed stage vocabulary.** Valid stage labels are fixed by ``n_stages`` at construction
       and cannot be customised. Supported sets are: 2-stage (Wake/Sleep), 3-stage
       (Wake/NREM/REM), 4-stage (Wake/Light/Deep/REM), and 5-stage (Wake/N1/N2/N3/REM).
       Artefact (ART) and Unscored (UNS) are always part of the vocabulary regardless of
       ``n_stages``.

    Stages are stored as strings (``"WAKE"``, ``"N2"``, ``"REM"``, ...) rather than integers,
    reducing the risk of misinterpretation. The object also carries its own metadata: epoch
    duration, an optional start datetime with timezone, and an optional scorer name.

    To create a ``Hypnogram`` from a legacy integer array, use :py:meth:`from_integers`.

    To save a ``Hypnogram`` to disk and reload it with all metadata intact, use
    :py:meth:`to_json` and :py:meth:`from_json`.

    For a step-by-step introduction covering all features, see the
    :ref:`tutorial_hypnogram` tutorial.

    .. versionadded:: 0.7.0

    .. rubric:: Main methods

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Method
         - Description
       * - :py:meth:`as_int`
         - Return hypnogram values as a :py:class:`~pandas.Series` of integers.
       * - :py:meth:`as_events`
         - Return a BIDS-compatible events :py:class:`~pandas.DataFrame` (onset, duration, stage).
       * - :py:meth:`get_mask`
         - Return a boolean array marking epochs that match one or more stage labels.
       * - :py:meth:`to_dict` / :py:meth:`to_json`
         - Serialize the hypnogram and all metadata to a dictionary or JSON file.
       * - :py:meth:`crop`
         - Slice the hypnogram by epoch index or absolute timestamp.
       * - :py:meth:`pad`
         - Extend the hypnogram before and/or after with a chosen fill stage.
       * - :py:meth:`upsample`
         - Resample the hypnogram to a finer epoch resolution.
       * - :py:meth:`consolidate_stages`
         - Merge stages to a coarser hypnogram (e.g. 5-stage to 2-stage).
       * - :py:meth:`upsample_to_data`
         - Align and upsample the hypnogram to match an EEG recording sample-by-sample.
       * - :py:meth:`sleep_statistics`
         - Compute standard AASM sleep statistics (TIB, TST, SE, WASO, stage durations, ...).
       * - :py:meth:`transition_matrix`
         - Compute the stage-transition count matrix and probability matrix.
       * - :py:meth:`find_periods`
         - Detect consecutive runs of a single stage exceeding a minimum duration.
       * - :py:meth:`evaluate`
         - Compare two hypnograms epoch-by-epoch (kappa, F1, MCC, ...).
       * - :py:meth:`plot_hypnogram`
         - Plot the hypnogram as a standard hypnogram figure.
       * - :py:meth:`plot_hypnodensity`
         - Plot per-epoch stage probabilities as a stacked area chart (requires ``proba``).
       * - :py:meth:`simulate_similar`
         - Simulate a new hypnogram with the same transition probabilities as this one.

    The full list of methods and attributes is available at the bottom of this page.

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
        :py:class:`~pandas.Timestamp` objects.
    tz : str or :py:class:`datetime.timezone`, optional
        Timezone of the hypnogram ``start`` time, e.g. ``"Europe/Paris"`` or
        ``"America/New_York"``. Only used when ``start`` is timezone-naive. A full list of valid
        timezone strings is available via ``import zoneinfo; zoneinfo.available_timezones()``.
    scorer : str
        An optional string indicating the scorer name. If specified, this will be set as the name
        of the :py:class:`pandas.Series`, otherwise the name will be set to "Stage".
    proba : :py:class:`pandas.DataFrame`
        An optional dataframe with the probability of each sleep stage for each epoch in hypnogram.
        Each row must sum to 1. This is automatically included if the hypnogram is created with
        :py:class:`yasa.SleepStaging`.

    Examples
    --------
    Create a 2-stage hypnogram and inspect its contents:

    >>> from yasa import Hypnogram
    >>> values = ["W", "W", "W", "S", "S", "S", "S", "S", "W", "S", "S", "S"]
    >>> hyp = Hypnogram(values, n_stages=2)
    >>> hyp
    <Hypnogram | 12 epochs x 30s (6.00 minutes), 2 unique stages>
     - Use `.hypno` to get the string values as a pandas.Series
     - Use `.as_int()` to get the integer values as a pandas.Series
     - Use `.plot_hypnogram()` to plot the hypnogram
    See the online documentation for more details.

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

    >>> hyp.n_epochs
    12

    >>> import pandas as pd
    >>> pd.Series(hyp.sleep_statistics())
    TIB         6.0000
    SPT         4.5000
    WASO        0.5000
    TST         4.0000
    SE         66.6667
    SME        88.8889
    SFI        15.0000
    SOL         1.5000
    SOL_5min       NaN
    WAKE        2.0000
    dtype: float64

    >>> counts, probs = hyp.transition_matrix()
    >>> counts
    To Stage    WAKE  SLEEP
    From Stage
    WAKE           2      2
    SLEEP          1      6

    For more examples covering 4-stage and 5-stage hypnograms, visualization, slicing,
    EEG alignment, saving, and scorer comparison, see the :ref:`tutorial_hypnogram` tutorial.
    """

    def __init__(
        self, values, n_stages=5, *, freq="30s", start=None, tz=None, scorer=None, proba=None
    ):
        assert isinstance(
            values, (list, np.ndarray, pd.Series, pd.api.extensions.ExtensionArray)
        ), "`values` must be a list, numpy.array or pandas.Series"
        # Normalize all inputs (list, ndarray, Series, Categorical, ArrowStringArray...) to a plain
        # object array once, so that nothing downstream depends on the input container. This also
        # drops the index of a pandas.Series.
        values = np.asarray(values, dtype=object)
        unique_values = pd.unique(values)
        assert all(isinstance(val, str) for val in unique_values), (
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
        mapping = _STAGE_MAPPINGS[n_stages].copy()
        labels = list(mapping)
        accepted = labels + [alias for alias, st in _STAGE_ALIASES.items() if st in labels]
        # Validate on the unique values only, which is much faster than a Python loop over epochs
        if not all(val.upper() in accepted for val in unique_values):
            n_unique_values = len(unique_values)
            msg = (
                f"{np.sort(unique_values)} do not match the accepted values for a {n_stages}-stage "
                f"hypnogram: {accepted}."
            )
            if n_unique_values < n_stages:
                msg += (
                    f"\nIf your hypnogram only has {n_unique_values} possible stages, make sure to "
                    f"specify `Hypnogram(values, n_stages={n_unique_values})`."
                )
            raise ValueError(msg)

        # Convert to canonical labels and a categorical dtype (reduces memory)
        hypno = pd.Series(values, name="Stage" if scorer is None else scorer)
        hypno = hypno.map({val: _canonical_stage(val) for val in unique_values})
        hypno = hypno.astype(CategoricalDtype(labels, ordered=False))
        # Normalize start to a pd.Timestamp, apply tz if provided, and create the index
        if start is not None:
            start = pd.Timestamp(start)
            if tz is not None:
                if start.tzinfo is not None:
                    raise ValueError(
                        "`start` is already timezone-aware. Do not pass `tz` when `start` already "
                        "contains timezone information."
                    )
                start = start.tz_localize(tz)
            hypno.index = pd.date_range(start=start, freq=freq, periods=hypno.shape[0], name="Time")
        else:
            hypno.index.name = "Epoch"
        # Validate proba
        if proba is not None:
            assert proba.shape[1] > 0, "`proba` must have at least one column."
            assert proba.shape[0] == hypno.shape[0], "`proba` must have the same length as `values`"
            assert np.allclose(proba.sum(axis=1), 1), "Each row of `proba` must sum to 1."
            # Accept the same abbreviations / case as `values`, e.g. "W" or "wake" -> "WAKE"
            proba = proba.rename(columns=_canonical_stage)
            in_proba_but_not_labels = np.setdiff1d(proba.columns, labels)
            assert not len(in_proba_but_not_labels), (
                f"Invalid stages in `proba`: {in_proba_but_not_labels}. The accepted stages are: "
                f"{labels}."
            )
            # Ensure same order as `labels`, and a positional index aligned with the epochs
            proba = proba.reindex(columns=labels).dropna(how="all", axis=1)
            proba.index = pd.RangeIndex(hypno.shape[0], name="Epoch")
        # Set attributes
        self._hypno = hypno
        self._freq = freq
        self._sampling_frequency = 1 / pd.Timedelta(freq).total_seconds()
        self._start = start
        self._n_stages = n_stages
        self._labels = labels
        self._mapping = mapping
        self._scorer = scorer
        self._proba = proba

    def _replace(self, values, **kwargs):
        """Return a new Hypnogram of the same class and metadata, with new ``values``.

        Any constructor argument (``n_stages``, ``freq``, ``start``, ``scorer``, ``proba``) can be
        overridden via ``kwargs``. ``proba`` is dropped unless explicitly passed. A custom
        :py:attr:`mapping` is preserved as long as ``n_stages`` is unchanged.
        """
        params = {
            "n_stages": self._n_stages,
            "freq": self._freq,
            "start": self._start,
            "scorer": self._scorer,
            "proba": None,
        }
        params.update(kwargs)
        new = type(self)(values, **params)
        if new.n_stages == self._n_stages:
            new._mapping = self._mapping.copy()
        return new

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

    def __len__(self):
        """Return the number of epochs. Allows ``len(hyp)``."""
        return self.n_epochs

    def __eq__(self, other):
        """Element-wise equality comparison with another :py:class:`Hypnogram`.

        Returns a boolean NumPy array of length ``n_epochs``. Both hypnograms must have the same
        number of epochs. To test full equality, use ``(hyp1 == hyp2).all()``.

        Examples
        --------
        >>> from yasa import Hypnogram
        >>> hyp1 = Hypnogram(["W", "N1", "N2", "REM"], freq="30s")
        >>> hyp2 = Hypnogram(["W", "N1", "N3", "REM"], freq="30s")
        >>> hyp1 == hyp2
        array([ True,  True, False,  True])
        >>> (hyp1 == hyp2).all()
        False
        """
        if not isinstance(other, Hypnogram):
            return NotImplemented
        if self.n_epochs != other.n_epochs:
            raise ValueError(
                f"Cannot compare Hypnograms with different numbers of epochs "
                f"({self.n_epochs} vs {other.n_epochs})."
            )
        return self._hypno.to_numpy() == other._hypno.to_numpy()

    def __getitem__(self, key):
        """Slice the hypnogram by epoch position, returning a new :py:class:`Hypnogram`.

        Supports integer indexing (including negative) and slices. For time-based slicing,
        use :py:meth:`crop` instead.

        Parameters
        ----------
        key : int or slice
            Epoch index or slice. Negative integers are supported (e.g. ``hyp[-1]``).

        Returns
        -------
        hyp : :py:class:`Hypnogram`
            A new :py:class:`Hypnogram` covering the selected epochs.

        Examples
        --------
        >>> from yasa import Hypnogram
        >>> hyp = Hypnogram(["W", "W", "N1", "N2", "N3", "REM"], freq="30s")
        >>> hyp[0]
        <Hypnogram | 1 epochs x 30s (0.50 minutes), 5 unique stages>
         - Use `.hypno` to get the string values as a pandas.Series
         - Use `.as_int()` to get the integer values as a pandas.Series
         - Use `.plot_hypnogram()` to plot the hypnogram
        See the online documentation for more details.

        >>> hyp[-1].hypno.iloc[0]
        'REM'

        >>> hyp[1:4].hypno.to_list()
        ['WAKE', 'N1', 'N2']
        """
        if isinstance(key, (int, np.integer)):
            if not -self.n_epochs <= key < self.n_epochs:
                raise IndexError(
                    f"Epoch index {key} is out of range for a Hypnogram of {self.n_epochs} epochs."
                )
            idx = int(key) % self.n_epochs  # supports negative indexing
            return self.crop(idx, idx)
        if isinstance(key, slice):
            if key.step is not None:
                raise ValueError(
                    "Step slicing is not supported for Hypnogram. Use crop() for range selection."
                )
            epoch_range = range(*key.indices(self.n_epochs))
            if not epoch_range:
                raise IndexError("Slice results in an empty Hypnogram.")
            return self.crop(epoch_range[0], epoch_range[-1])
        raise TypeError(
            f"Unsupported key type '{type(key).__name__}'. Use int or slice, "
            "or use crop() for time-based selection."
        )

    @classmethod
    def from_integers(
        cls,
        values,
        mapping=_DEFAULT_INT_TO_STR,
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
        (e.g. ``[0, 1, 2, 3, 4]`` for Wake, N1, N2, N3, REM). Integers are converted to string
        labels using ``mapping`` before creating the :py:class:`Hypnogram`.

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
        str_hypno = _hypno_int_to_str(values, mapping_dict=mapping)
        return cls(
            str_hypno, n_stages=n_stages, freq=freq, start=start, tz=tz, scorer=scorer, proba=proba
        )

    @classmethod
    def from_profusion(cls, fname, *, start=None, tz=None, scorer=None):  # pragma: no cover
        """Create a :py:class:`Hypnogram` from a Compumedics Profusion hypnogram (.xml).

        The Compumedics Profusion hypnogram format is one of the two hypnogram formats found on
        the `National Sleep Research Resource (NSRR) <https://sleepdata.org/>`_ website. For
        details on the format, see
        https://github.com/nsrr/edf-editor-translator/wiki/Compumedics-Annotation-Format.

        The epoch length and sleep stage integers are read directly from the XML file. Profusion
        integer stages are mapped to YASA conventions: stage 4 (S4/N3) → 3, stage 5 (REM) → 4,
        stage 9 (Active) → 0 (WAKE).

        .. versionadded:: 0.7.0

        Parameters
        ----------
        fname : str or path-like
            Path to the Profusion XML file.
        start : str, datetime, or pd.Timestamp, optional
            Start datetime of the recording (e.g. ``"2022-12-15 22:30:00"``). Can be combined
            with ``tz`` to localize a naive string to a specific timezone.
        tz : str, optional
            Timezone string to localize a naive ``start`` (e.g. ``"Europe/Paris"``). See
            :py:class:`Hypnogram` for details.
        scorer : str, optional
            Name of the scorer.

        Returns
        -------
        hyp : :py:class:`Hypnogram`
            A :py:class:`Hypnogram` instance with string-encoded stages.

        Examples
        --------
        >>> from yasa import Hypnogram
        >>> hyp = Hypnogram.from_profusion("path/to/hypnogram.xml")  # doctest: +SKIP
        >>> hyp = Hypnogram.from_profusion(
        ...     "path/to/hypnogram.xml",
        ...     start="2022-12-15 22:30:00",
        ...     tz="Europe/Paris",
        ... )  # doctest: +SKIP
        """
        hypno_int, epoch_length = _read_profusion(fname)
        hypno_int = pd.Series(hypno_int).replace(_PROFUSION_TO_YASA).to_numpy()
        freq = f"{int(epoch_length)}s"
        return cls.from_integers(hypno_int, freq=freq, start=start, tz=tz, scorer=scorer)

    @classmethod
    def from_dict(cls, d):
        """Reconstruct a :py:class:`Hypnogram` from a dictionary produced by :py:meth:`to_dict`.

        All metadata is restored: epoch duration, start datetime (including timezone), scorer
        name, and stage probabilities (if present).

        .. versionadded:: 0.7.0

        Parameters
        ----------
        d : dict
            A dictionary with keys ``"values"``, ``"n_stages"``, ``"freq"``, ``"start"``,
            ``"scorer"``, and ``"proba"``, as returned by :py:meth:`to_dict`. An optional ``"tz"``
            key restores the named timezone of ``"start"``.

        Returns
        -------
        hyp : :py:class:`Hypnogram`
            The reconstructed Hypnogram.

        See Also
        --------
        to_dict : Return the Hypnogram as a JSON-serializable dictionary.
        from_json : Load a :py:class:`Hypnogram` from a JSON file on disk.
        """
        start = pd.Timestamp(d["start"]) if d["start"] is not None else None
        if start is not None and d.get("tz") is not None:
            # Restore the named timezone (the ISO string only contains a fixed UTC offset)
            start = start.tz_convert(d["tz"])
        proba = pd.DataFrame(d["proba"]) if d["proba"] is not None else None
        return cls(
            values=d["values"],
            n_stages=d["n_stages"],
            freq=d["freq"],
            start=start,
            scorer=d["scorer"],
            proba=proba,
        )

    @classmethod
    def from_json(cls, fname):
        """Load a :py:class:`Hypnogram` from a JSON file saved with :py:meth:`to_json`.

        All metadata is restored: epoch duration, start datetime (including timezone), scorer
        name, and stage probabilities (if present).

        This method delegates to :py:meth:`from_dict` for deserialization.

        .. versionadded:: 0.7.0

        Parameters
        ----------
        fname : str or path-like
            Path to the JSON file.

        Returns
        -------
        hyp : :py:class:`Hypnogram`
            The loaded Hypnogram.

        See Also
        --------
        from_dict : Reconstruct a :py:class:`Hypnogram` from an in-memory dictionary.
        to_json : Save the Hypnogram to a JSON file on disk.
        """
        import json

        with open(fname) as f:
            data = json.load(f)
        return cls.from_dict(data)

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
        return self._hypno.shape[0]

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
    def end(self):
        """End date/time of the hypnogram (exclusive: start of the epoch after the last one).

        Returns ``None`` if :py:attr:`start` is not set. When set, ``end - start`` equals the
        total recording duration.

        Examples
        --------
        >>> from yasa import simulate_hypnogram
        >>> hyp = simulate_hypnogram(tib=60, start="2022-12-15 22:30:00", seed=0)
        >>> hyp.start
        Timestamp('2022-12-15 22:30:00')
        >>> hyp.end
        Timestamp('2022-12-15 23:30:00')
        >>> hyp.end - hyp.start
        Timedelta('0 days 01:00:00')
        """
        if self._start is None:
            return None
        return self._start + pd.Timedelta(self._freq) * self.n_epochs

    @property
    def timedelta(self):
        """
        A :py:class:`pandas.TimedeltaIndex` vector with the accumulated time difference of each
        epoch compared to the first epoch.
        """
        return pd.timedelta_range(start=0, freq=self._freq, periods=self.n_epochs)

    @property
    def duration(self):
        """Total duration of the hypnogram, expressed in minutes."""
        return self.n_epochs / (60 * self._sampling_frequency)

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
        # Add the default ART and UNS values without modifying the user's dictionary
        map_dict = {"ART": -1, "UNS": -2, **map_dict}
        assert all(val in map_dict for val in self.hypno.unique()), (
            f"Some values in `hypno` ({self.hypno.unique()}) are not in `map_dict` "
            f"({map_dict.keys()})"
        )
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

    #######################################################################
    # CONVERSION
    #######################################################################

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
        # Map through the categorical codes with a lookup table, which (unlike renaming the
        # categories) supports custom mappings that are partial or many-to-one. Categories that are
        # absent from the mapping are, by construction of the mapping setter, not in the data.
        codes = self.hypno.cat.codes.to_numpy()
        if (codes < 0).any():
            raise ValueError("Hypnogram contains missing (NaN) values.")
        lut = np.array([self.mapping.get(st, 0) for st in self.hypno.cat.categories], np.int16)
        # Return as int16 (-32768 to 32767) to reduce memory usage
        return pd.Series(lut[codes], index=self.hypno.index, name=self.hypno.name)

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

    def get_mask(self, stages):
        """Return a boolean NumPy array marking epochs that match the given stages.

        Parameters
        ----------
        stages : str or list of str
            One or more stage labels, e.g. ``"N2"`` or ``["N2", "N3"]``. Must be valid
            labels for this hypnogram (see :attr:`labels`).

        Returns
        -------
        mask : :py:class:`numpy.ndarray` of bool
            Boolean array of length :attr:`n_epochs`, ``True`` where the
            hypnogram matches any of the requested stages.

        Examples
        --------
        >>> from yasa import Hypnogram
        >>> hyp = Hypnogram(["W", "N1", "N2", "N2", "N3", "REM"], n_stages=5)
        >>> hyp.get_mask(["N2", "N3"])
        array([False, False,  True,  True,  True, False])
        >>> hyp.get_mask("REM")
        array([False, False, False, False, False,  True])
        """
        stages = np.atleast_1d(stages)
        invalid = [s for s in stages if s not in self.labels]
        if invalid:
            raise ValueError(f"Invalid stage(s): {invalid}. Valid stages are: {self.labels}")
        return np.isin(self.hypno.to_numpy(), stages)

    #######################################################################
    # SERIALIZATION
    #######################################################################

    def copy(self):
        """Return a new copy of the current Hypnogram."""
        return self._replace(self._hypno, proba=self._proba)

    def to_dict(self):
        """Return the Hypnogram as a JSON-serializable dictionary.

        All metadata is preserved: epoch duration, start datetime (including timezone), scorer
        name, and stage probabilities. Stage probabilities (``proba``) are rounded to 6 decimal
        places.

        The dictionary has the following keys: ``"values"``, ``"n_stages"``, ``"freq"``,
        ``"start"``, ``"scorer"``, ``"proba"``, plus ``"tz"`` when ``start`` has a named timezone
        (e.g. ``"Europe/Paris"``). It can be passed to :py:meth:`from_dict` to reconstruct the
        :py:class:`Hypnogram`.

        .. versionadded:: 0.7.0

        Returns
        -------
        d : dict
            A JSON-serializable dictionary representing the Hypnogram and all its metadata.

        See Also
        --------
        from_dict : Reconstruct a :py:class:`Hypnogram` from a dictionary.
        to_json : Save the Hypnogram to a JSON file on disk.
        """
        d = {
            "values": self.hypno.to_numpy().tolist(),
            "n_stages": self.n_stages,
            "freq": self.freq,
            "start": self.start.isoformat() if self.start is not None else None,
            "scorer": self.scorer,
            "proba": self.proba.round(6).to_dict(orient="list") if self.proba is not None else None,
        }
        # The ISO string only stores a fixed UTC offset. Also save the name of the timezone (e.g.
        # "Europe/Paris"), if any, so that it survives a round-trip (e.g. for DST transitions).
        tz_name = getattr(self.start, "tz", None)
        tz_name = getattr(tz_name, "key", None) or getattr(tz_name, "zone", None)
        if tz_name is not None:
            d["tz"] = tz_name
        return d

    def to_json(self, fname):
        """Save the Hypnogram to a JSON file.

        The file can be reloaded with :py:meth:`from_json`. All metadata is preserved: epoch
        duration, start datetime (including timezone), scorer name, and stage probabilities.
        Stage probabilities (``proba``) are rounded to 6 decimal places.

        This method delegates to :py:meth:`to_dict` for serialization.

        .. versionadded:: 0.7.0

        Parameters
        ----------
        fname : str or path-like
            Output file path. By convention, use a ``.json`` extension.

        See Also
        --------
        to_dict : Return the same representation as an in-memory dictionary.
        from_json : Reload the Hypnogram from a JSON file.
        """
        import json

        with open(fname, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    #######################################################################
    # TRANSFORMATION
    #######################################################################

    def crop(self, start=None, end=None):
        """Crop the hypnogram to a range of epochs or absolute timestamps.

        Both ``start`` and ``end`` are **inclusive**. Pass integers for epoch-based cropping,
        or strings / :py:class:`~pandas.Timestamp` objects for time-based cropping (requires
        :py:attr:`start` to be set on the hypnogram). Negative integers count from the end, e.g.
        ``crop(start=-10)`` keeps the last 10 epochs.

        Parameters
        ----------
        start : int, str, or :py:class:`pandas.Timestamp`, optional
            First epoch to include. Defaults to the first epoch.
        end : int, str, or :py:class:`pandas.Timestamp`, optional
            Last epoch to include (inclusive). Defaults to the last epoch.

        Returns
        -------
        hyp : :py:class:`Hypnogram`
            A new :py:class:`Hypnogram` covering the selected range.

        Examples
        --------
        Epoch-based crop (first 10 epochs):

        >>> from yasa import simulate_hypnogram
        >>> hyp = simulate_hypnogram(tib=60, seed=0)
        >>> hyp.crop(end=9).n_epochs
        10

        Time-based crop (requires ``start`` to be set):

        >>> hyp = simulate_hypnogram(tib=60, start="2022-12-15 22:30:00", seed=0)
        >>> cropped = hyp.crop(start="2022-12-15 23:00:00", end="2022-12-15 23:30:00")
        >>> cropped.n_epochs
        60
        >>> cropped.start
        Timestamp('2022-12-15 23:00:00')
        """
        n = self.n_epochs
        if isinstance(start, _TIME_TYPES) or isinstance(end, _TIME_TYPES):
            if self._start is None:
                raise ValueError(
                    "Time-based crop requires the Hypnogram to have a `start` datetime set."
                )
            start_key = pd.Timestamp(start) if start is not None else None
            end_key = pd.Timestamp(end) if end is not None else None
            # Convert the (inclusive) time window to epoch positions
            start_idx, end_idx, _ = self._hypno.index.slice_indexer(start_key, end_key).indices(n)
        else:
            # Integer positions, negative values count from the end (e.g. -1 = last epoch)
            start_idx = 0 if start is None else (start + n if start < 0 else start)
            # Convert inclusive end to exclusive
            end_idx = n if end is None else (end + n if end < 0 else end) + 1
            start_idx, end_idx = max(start_idx, 0), min(end_idx, n)

        if end_idx <= start_idx:
            raise ValueError("Crop window is empty. Check your start/end parameters.")

        # Slice by position so that `proba` (positional index) and `hypno` stay aligned
        new_start = None
        if self._start is not None:
            new_start = self._start + pd.Timedelta(self._freq) * start_idx
        return self._replace(
            self._hypno.iloc[start_idx:end_idx],
            start=new_start,
            proba=self._proba.iloc[start_idx:end_idx] if self._proba is not None else None,
        )

    def pad(self, before=None, after=None, fill_value="UNS"):
        """Extend the hypnogram by padding epochs before and/or after.

        Parameters
        ----------
        before : int, str, or :py:class:`pandas.Timestamp`, optional
            Number of epochs to prepend (int ≥ 0), or a timestamp for the new start (must be
            strictly before :py:attr:`start`). Requires :py:attr:`start` to be set when a
            timestamp is given.
        after : int, str, or :py:class:`pandas.Timestamp`, optional
            Number of epochs to append (int ≥ 0), or a timestamp for the new end (exclusive;
            must be strictly after :py:attr:`end`). Requires :py:attr:`start` to be set when a
            timestamp is given.
        fill_value : str or tuple of str, optional
            Stage label(s) for the added epochs. Default is ``"UNS"`` (Unscored).

            * A single string applies the same fill to both ends. Use ``"edge"`` to repeat
              the first epoch for ``before`` and the last epoch for ``after``, analogous to
              :func:`numpy.pad` with ``mode="edge"``. Any valid stage label (see
              :attr:`labels`) is also accepted.
            * A 2-tuple ``(fill_before, fill_after)`` sets different values for each end,
              e.g. ``("UNS", "WAKE")`` pads the start with Unscored epochs and the end with
              Wake epochs. Each element follows the same rules as the scalar form.

        Returns
        -------
        hyp : :py:class:`Hypnogram`
            A new :py:class:`Hypnogram` with the requested padding. ``proba`` is not propagated.

        Warns
        -----
        UserWarning
            When a timestamp-based duration is not a perfect multiple of :py:attr:`freq`,
            the padding is floored to the nearest complete epoch count.

        Examples
        --------
        Epoch-based padding with the default fill value (UNS):

        >>> from yasa import Hypnogram
        >>> hyp = Hypnogram(["N2", "N2", "REM"], freq="30s")
        >>> hyp.pad(before=2, after=1).hypno.to_list()
        ['UNS', 'UNS', 'N2', 'N2', 'REM', 'UNS']

        Edge padding (repeat first/last epoch):

        >>> hyp.pad(before=2, after=1, fill_value="edge").hypno.to_list()
        ['N2', 'N2', 'N2', 'N2', 'REM', 'REM']

        Different fill values for each end — pad start with UNS and end with WAKE:

        >>> hyp.pad(before=1, after=2, fill_value=("UNS", "WAKE")).hypno.to_list()
        ['UNS', 'N2', 'N2', 'REM', 'WAKE', 'WAKE']

        Timestamp-based padding — extend to a fixed recording window. Here the hypnogram
        starts at 22:01:00 and ends at 22:02:30, so two 30-s UNS epochs are prepended to
        align it to 22:00:00, and one is appended to reach 22:03:00:

        >>> hyp_ts = Hypnogram(["N2", "N2", "REM"], freq="30s", start="2023-01-01 22:01:00")
        >>> padded = hyp_ts.pad(before="2023-01-01 22:00:00", after="2023-01-01 22:03:00")
        >>> padded.n_epochs
        6
        >>> padded.start
        Timestamp('2023-01-01 22:00:00')
        >>> padded.end
        Timestamp('2023-01-01 22:03:00')
        >>> padded.hypno.to_list()
        ['UNS', 'UNS', 'N2', 'N2', 'REM', 'UNS']
        """
        # -- Normalise and validate fill_value --------------------------------
        if isinstance(fill_value, (list, tuple)):
            if len(fill_value) != 2:
                raise ValueError(
                    "`fill_value` tuple must have exactly 2 elements: (fill_before, fill_after)."
                )
            fill_before_val, fill_after_val = fill_value
        else:
            fill_before_val = fill_after_val = fill_value

        for fv in (fill_before_val, fill_after_val):
            if fv != "edge" and fv not in self.labels:
                raise ValueError(
                    f"`fill_value` must be 'edge' or a valid stage label. "
                    f"Valid labels are: {self.labels}"
                )

        n_before = self._n_pad_epochs(before, "before")
        n_after = self._n_pad_epochs(after, "after")

        # -- Build padded values --------------------------------------------
        fill_before = str(self._hypno.iloc[0]) if fill_before_val == "edge" else fill_before_val
        fill_after = str(self._hypno.iloc[-1]) if fill_after_val == "edge" else fill_after_val

        original = np.asarray(self._hypno, dtype=object)
        new_values = np.concatenate(
            [
                np.full(n_before, fill_before, dtype=object),
                original,
                np.full(n_after, fill_after, dtype=object),
            ]
        )

        new_start = (
            self._start - pd.Timedelta(self._freq) * n_before if self._start is not None else None
        )
        return self._replace(new_values, start=new_start)

    def _n_pad_epochs(self, value, side):
        """Convert the ``before`` / ``after`` argument of :py:meth:`pad` to a number of epochs."""
        if value is None:
            return 0
        if isinstance(value, (int, np.integer)):
            if value < 0:
                raise ValueError(f"`{side}` must be a non-negative integer.")
            return int(value)
        if not isinstance(value, _TIME_TYPES):
            raise TypeError(f"`{side}` must be an int or a timestamp, got {type(value).__name__}.")
        if self._start is None:
            raise ValueError(
                "Timestamp-based padding requires the Hypnogram to have a `start` datetime set."
            )
        ts = pd.Timestamp(value)
        # `before` is compared to the start, `after` to the (exclusive) end of the hypnogram
        ref_name, ref = ("start", self._start) if side == "before" else ("end", self.end)
        if (ref.tzinfo is not None) != (ts.tzinfo is not None):
            raise ValueError(
                f"`{side}` and the Hypnogram {ref_name} must have matching timezone "
                f"awareness ({ref_name}: {ref}, {side}: {ts})."
            )
        delta = ref - ts if side == "before" else ts - ref
        if delta <= pd.Timedelta(0):
            raise ValueError(
                f"`{side}` ({ts}) must be strictly {side} the Hypnogram {ref_name} ({ref})."
            )
        freq_td = pd.Timedelta(self._freq)
        n_exact = delta / freq_td
        n_epochs = int(np.floor(n_exact))
        if (delta - freq_td * n_epochs).total_seconds() > 1e-6:
            warnings.warn(
                f"`{side}` padding duration ({delta}) is not a perfect multiple of the epoch "
                f"duration ({self._freq}). Padding with {n_epochs} complete epoch(s) "
                f"(flooring {n_exact:.6g}).",
                UserWarning,
                stacklevel=3,
            )
        return n_epochs

    def upsample(self, new_freq):
        """Upsample hypnogram to a higher frequency.

        Parameters
        ----------
        new_freq : str
            Target frequency as a pandas frequency string (e.g. ``"10s"`` or ``"1min"``). Must
            represent a higher sampling rate than the current hypnogram frequency, i.e. a shorter
            epoch duration (e.g. ``"10s"`` when the current frequency is ``"30s"``). The current
            epoch duration must be a whole multiple of ``new_freq`` (e.g. ``"20s"`` is not
            allowed when the current frequency is ``"30s"``).

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
        ratio = pd.Timedelta(self.freq) / pd.Timedelta(new_freq)
        assert float(ratio).is_integer(), (
            f"The current frequency of the hypnogram ({self.freq}) must be a whole multiple of "
            f"`new_freq` ({new_freq}), otherwise the stage boundaries would be shifted."
        )
        # Each epoch is repeated, so that the last epoch is fully preserved (e.g. a 30-sec epoch
        # at 07:20:30 becomes three 10-sec epochs at 07:20:30, 07:20:40 and 07:20:50).
        # NOTE: Do not upsample probability
        new_values = np.repeat(np.asarray(self._hypno, dtype=object), int(ratio))
        return self._replace(new_values, freq=new_freq)

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

        # Change sleep codes where applicable, e.g. N1/N2 -> LIGHT for a 4-stage hypnogram
        mapping = _CONSOLIDATION_MAPPINGS[new_n_stages]
        new_values = self.hypno.astype(object).replace(mapping).to_numpy()
        # TODO: Combine stages probability?
        return self._replace(new_values, n_stages=new_n_stages)

    #######################################################################
    # ALIGNMENT TO DATA
    #######################################################################

    def upsample_to_data(self, data, sf=None, meas_date_is_local=True, verbose=True):
        """
        Upsample a hypnogram to a given sampling frequency and fit the resulting hypnogram to
        corresponding EEG data, such that the hypnogram and EEG data have the exact same number of
        samples.

        When ``self.start`` is set **and** ``data`` is a :py:class:`mne.io.BaseRaw` with a
        valid ``meas_date``, alignment uses absolute timestamps rather than sample
        count. See the :ref:`tutorial_hypnogram` tutorial for a full description of all alignment
        scenarios and when to use ``start`` / ``tz``.

        Parameters
        ----------
        data : array_like or :py:class:`mne.io.BaseRaw`
            1D or 2D EEG data. Can also be a :py:class:`mne.io.BaseRaw`, in which case ``data``
            and ``sf`` will be automatically extracted.
        sf : float
            The sampling frequency of ``data``, in Hz (e.g. 100 Hz, 256 Hz, ...).
            Can be omitted if ``data`` is a :py:class:`mne.io.BaseRaw`.
        meas_date_is_local : bool
            If ``True`` (default), ``meas_date`` is treated as a local absolute timestamp,
            consistent with the EDF+ standard, which explicitly defines ``starttime`` as local
            time at the patient's location. Set to ``False`` only if your EDF files genuinely store
            UTC in ``meas_date``, in which case pass ``tz`` when constructing the
            :py:class:`~yasa.Hypnogram` so the two timestamps share a common reference frame.
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
            Only when ``meas_date_is_local=False``: raised if ``self.start`` is timezone-naive
            while ``raw.meas_date`` is timezone-aware (UTC). Fix by passing ``tz`` at
            construction: ``Hypnogram(..., tz='Europe/Paris')``. This error cannot occur with
            the default ``meas_date_is_local=True``.

        Warns
        -----
        UserWarning
            If the hypnogram is shorter or longer than the data and needs to be padded or
            cropped. Silenced by passing ``verbose='error'``.

        Examples
        --------
        >>> import numpy as np
        >>> from yasa import Hypnogram
        >>> hyp = Hypnogram(["W", "W", "N1", "N2", "N2", "REM"], freq="30s")
        >>> data = np.zeros((1, 18000))
        >>> hypno = hyp.upsample_to_data(data, sf=100)
        >>> hypno.shape
        (18000,)
        >>> np.unique(hypno)
        array([0, 1, 2, 4], dtype=int16)
        """
        if (
            self.start is not None
            and isinstance(data, mne.io.BaseRaw)
            and data.info["meas_date"] is not None
        ):
            return self._upsample_to_raw_timestamps(
                data, meas_date_is_local=meas_date_is_local, verbose=verbose
            )
        hypno_up = _hypno_upsample_to_data(
            self.as_int(), self.sampling_frequency, data=data, sf_data=sf, verbose=verbose
        )
        return hypno_up

    #######################################################################
    # ANALYSIS
    #######################################################################

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
        SFI          6.0000
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
        SFI          1.5075
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
        is_sleep = np.isin(hypno, _SLEEP_STAGES)
        # Every duration is converted from epochs to minutes at assignment. SE, SME (percentages)
        # and SFI (a rate) are then derived from the minute values, so they never go through a
        # unit conversion and do not depend on the epoch length of the hypnogram.
        epochs_per_min = 60 * self.sampling_frequency
        stats = {}

        # TIB, first and last sleep
        stats["TIB"] = self.n_epochs / epochs_per_min
        idx_sleep = np.flatnonzero(~np.isin(hypno, _NON_SLEEP_STAGES))
        has_sleep = idx_sleep.size > 0
        first_sleep, last_sleep = (idx_sleep[0], idx_sleep[-1]) if has_sleep else (0, self.n_epochs)
        # Crop to SPT
        spt = slice(first_sleep, last_sleep + 1)
        stats["SPT"] = (last_sleep + 1 - first_sleep) / epochs_per_min if has_sleep else 0
        stats["WASO"] = np.sum(hypno[spt] == "WAKE") / epochs_per_min if has_sleep else np.nan
        # Before YASA v0.5.0, TST was calculated as SPT - WASO, meaning that Art
        # and Unscored epochs were included. TST is now restrained to sleep stages.
        stats["TST"] = np.sum(is_sleep[spt]) / epochs_per_min

        # Sleep efficiency and fragmentation
        stats["SE"] = 100 * stats["TST"] / stats["TIB"]
        if stats["SPT"] == 0:
            stats["SME"] = np.nan
            stats["SFI"] = np.nan
        else:
            # Sleep maintenance efficiency
            stats["SME"] = 100 * stats["TST"] / stats["SPT"]
            # SFI is a rate: number of transitions from sleep into Wake per hour of TST.
            # The original definition included transitions into Wake or N1.
            n_trans_to_wake = np.count_nonzero(is_sleep[:-1] & (hypno[1:] == "WAKE"))
            stats["SFI"] = n_trans_to_wake / (stats["TST"] / 60)

        # Sleep stage latencies -- only relevant if hypno is cropped to TIB
        stats["SOL"] = first_sleep / epochs_per_min if stats["TST"] > 0 else np.nan
        # Latency to the first run of at least 5 minutes of consecutive sleep. The threshold is
        # rounded up to a whole number of epochs, e.g. 3 epochs for a 2-min hypnogram.
        min_epochs = int(np.ceil(5 * epochs_per_min - 1e-9))
        run_values, run_starts, run_lengths = _find_runs(is_sleep)
        idx_sol_5min = np.flatnonzero(run_values & (run_lengths >= min_epochs))
        stats["SOL_5min"] = (
            run_starts[idx_sol_5min[0]] / epochs_per_min if idx_sol_5min.size else np.nan
        )

        if "REM" in self.labels:
            # Question: should we add latencies for other stage too?
            idx_rem = np.flatnonzero(hypno == "REM")
            stats["Lat_REM"] = idx_rem[0] / epochs_per_min if idx_rem.size else np.nan

        # Duration of each stage (SLEEP is skipped because it is equal to TST). ART and UNS are
        # only reported if present in the hypnogram.
        counts = self.hypno.value_counts(sort=False)
        for st in self.labels:
            if st == "SLEEP" or (st in ["ART", "UNS"] and counts[st] == 0):
                continue
            stats[st] = counts[st] / epochs_per_min

        # Proportion of each sleep stages
        for st in _SLEEP_STAGES:
            if st in stats:
                stats[f"%{st}"] = 100 * stats[st] / stats["TST"] if stats["TST"] > 0 else np.nan

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
            <https://en.wikipedia.org/wiki/Stochastic_matrix>`_, i.e. each row sums to 1. The
            only exception is a stage with no outgoing transition (i.e. a stage that only
            occurs at the very last epoch), for which the probabilities are undefined (NaN).

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
        # Build the matrix from the stage labels rather than from ``as_int()``, so that stages
        # sharing the same integer in a custom mapping keep their own row and column. Stages are
        # sorted by integer value, then by category order (the sort is stable).
        categories = self.hypno.cat.categories
        codes = self.hypno.cat.codes.to_numpy()
        if (codes < 0).any():
            raise ValueError("Hypnogram contains missing (NaN) values.")
        stages = sorted(categories[np.unique(codes)], key=self.mapping.get)
        rank = np.zeros(categories.size, dtype=int)
        rank[categories.get_indexer(stages)] = np.arange(len(stages))
        counts, probs = _transition_matrix(rank[codes])
        labels = dict(enumerate(stages))
        counts = counts.rename(index=labels, columns=labels)
        probs = probs.rename(index=labels, columns=labels)
        return counts, probs

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
        return _hypno_find_periods(
            self.hypno, self.sampling_frequency, threshold=threshold, equal_length=equal_length
        )

    def evaluate(self, obs_hyp):
        """Evaluate agreement between two hypnograms of the same sleep session.

        For example, the reference hypnogram (i.e., ``self``) might be a manually-scored hypnogram
        and the observed hypnogram (i.e., ``obs_hyp``) might be a hypnogram from actigraphy, a
        wearable device, or an automated scorer (e.g., :py:meth:`yasa.SleepStaging.predict`).

        .. warning:: **Experimental** — this method returns a :py:class:`yasa.EpochByEpochAgreement`
            object whose API may change before the full release planned for v0.8.0.

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
        accuracy        55.000
        balanced_acc    35.497
        kappa            0.227
        mcc              0.231
        precision       51.484
        f1              52.380
        Name: agreement, dtype: float64
        """
        return EpochByEpochAgreement([self], [obs_hyp])

    #######################################################################
    # VISUALIZATION
    #######################################################################

    def plot_hypnogram(self, highlight="REM", fill_color=None, ax=None, **kwargs):
        """Plot the hypnogram.

        Parameters
        ----------
        highlight : str or None
            Optional stage to highlight with alternate color.
        fill_color : str or None
            Optional color to fill space above hypnogram line.
        ax : :py:class:`matplotlib.axes.Axes`
            Axis on which to draw the plot, optional.
        **kwargs : dict
            Keyword arguments controlling hypnogram line display (e.g., ``lw``, ``linestyle``).
            Passed to :py:func:`matplotlib.pyplot.stairs` and
            :py:func:`matplotlib.pyplot.hlines`.

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
        return _plot_hypnogram(self, highlight=highlight, fill_color=fill_color, ax=ax, **kwargs)

    def plot_hypnodensity(self, palette=None, ax=None):
        """Plot the hypnodensity: per-epoch stage probabilities as a stacked area chart.

        Requires that the :py:attr:`proba` attribute is set (i.e. the hypnogram was created by
        :py:meth:`yasa.SleepStaging.predict`).

        Parameters
        ----------
        palette : dict or None
            A dictionary mapping stage names to matplotlib colors, e.g.
            ``{"WAKE": "#99d7f1", "REM": "xkcd:sunflower"}``. When ``None`` (default), a
            built-in palette is used. Missing stage keys fall back to ``"gray"``.
        ax : :py:class:`matplotlib.axes.Axes` or None
            Axis on which to draw the plot. If ``None`` (default), the current axis is used.

        Returns
        -------
        ax : :py:class:`matplotlib.axes.Axes`
            Matplotlib Axes

        Raises
        ------
        ValueError
            If :py:attr:`proba` is ``None``.

        Examples
        --------
        5-stage hypnogram:

        .. plot::

            >>> import numpy as np
            >>> import pandas as pd
            >>> from yasa import Hypnogram, simulate_hypnogram
            >>> import matplotlib.pyplot as plt
            >>> hyp = simulate_hypnogram(tib=300, n_stages=5, seed=42)
            >>> stages = ["WAKE", "N1", "N2", "N3", "REM"]
            >>> rng = np.random.default_rng(42)
            >>> one_hot = (
            ...     pd.get_dummies(hyp.hypno)
            ...     .reindex(columns=stages, fill_value=0)
            ...     .to_numpy(dtype=float)
            ... )
            >>> noise = rng.dirichlet(np.ones(5) * 0.5, size=hyp.n_epochs)
            >>> raw = 0.75 * one_hot + 0.25 * noise
            >>> proba = pd.DataFrame(raw / raw.sum(axis=1, keepdims=True), columns=stages)
            >>> ax = Hypnogram(hyp.hypno, n_stages=5, proba=proba).plot_hypnodensity()
            >>> plt.tight_layout()

        4-stage hypnogram:

        .. plot::

            >>> import numpy as np
            >>> import pandas as pd
            >>> from yasa import Hypnogram, simulate_hypnogram
            >>> import matplotlib.pyplot as plt
            >>> hyp = simulate_hypnogram(tib=300, n_stages=4, seed=42)
            >>> stages = ["WAKE", "LIGHT", "DEEP", "REM"]
            >>> rng = np.random.default_rng(42)
            >>> one_hot = (
            ...     pd.get_dummies(hyp.hypno)
            ...     .reindex(columns=stages, fill_value=0)
            ...     .to_numpy(dtype=float)
            ... )
            >>> noise = rng.dirichlet(np.ones(4) * 0.5, size=hyp.n_epochs)
            >>> raw = 0.75 * one_hot + 0.25 * noise
            >>> proba = pd.DataFrame(raw / raw.sum(axis=1, keepdims=True), columns=stages)
            >>> ax = Hypnogram(hyp.hypno, n_stages=4, proba=proba).plot_hypnodensity()
            >>> plt.tight_layout()

        2-stage hypnogram:

        .. plot::

            >>> import numpy as np
            >>> import pandas as pd
            >>> from yasa import Hypnogram, simulate_hypnogram
            >>> import matplotlib.pyplot as plt
            >>> hyp = simulate_hypnogram(tib=300, n_stages=2, seed=42)
            >>> stages = ["WAKE", "SLEEP"]
            >>> rng = np.random.default_rng(42)
            >>> one_hot = (
            ...     pd.get_dummies(hyp.hypno)
            ...     .reindex(columns=stages, fill_value=0)
            ...     .to_numpy(dtype=float)
            ... )
            >>> noise = rng.dirichlet(np.ones(2) * 0.5, size=hyp.n_epochs)
            >>> raw = 0.75 * one_hot + 0.25 * noise
            >>> proba = pd.DataFrame(raw / raw.sum(axis=1, keepdims=True), columns=stages)
            >>> ax = Hypnogram(hyp.hypno, n_stages=2, proba=proba).plot_hypnodensity()
            >>> plt.tight_layout()
        """
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        if self._proba is None:
            raise ValueError(
                "No probability data found. `proba` is only available when the Hypnogram "
                "was created by `yasa.SleepStaging.predict()`."
            )

        # Default color palette covering all possible stage names.
        # Base 5-stage colors: WAKE=#99d7f1, N1=#009ddc, N2=#0a437a, N3=#720058, REM=#ffc512
        # Derived colors: LIGHT=avg(N1,N2), NREM=avg(N1,N2,N3), DEEP=N3, SLEEP=dark navy
        _default_palette = {
            "WAKE": "#99d7f1",
            "N1": "#009ddc",
            "N2": "#0a437a",
            "N3": "#720058",
            "REM": "#ffc512",
            "LIGHT": "#0570ab",  # avg(N1, N2)
            "DEEP": "#720058",  # = N3
            "NREM": "#294b8f",  # avg(N1, N2, N3)
            "SLEEP": "#003366",  # dark navy, pairs with light-blue WAKE
            "ART": "#999999",
            "UNS": "#cccccc",
        }
        if palette is None:
            palette = _default_palette

        stages = self._proba.columns.tolist()
        colors = [palette.get(s, "gray") for s in stages]

        # Build x-axis values
        if self._start is not None:
            x = mdates.date2num(self._hypno.index)
            xlabel = "Time"
        else:
            x = self.timedelta.total_seconds() / 60  # minutes
            xlabel = "Time [mins]" if self.duration <= 90 else "Time [hrs]"
            if self.duration > 90:
                x = x / 60  # convert to hours

        # Increase font size, restoring the original even if plotting fails
        with plt.rc_context({"font.size": 18}):
            if ax is None:
                _, ax = plt.subplots(figsize=(12, 4))
            ax.stackplot(x, self._proba.to_numpy().T, labels=stages, colors=colors, alpha=0.85)
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability")
            ax.set_xlabel(xlabel)
            ax.legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper left")
            ax.spines[["right", "top"]].set_visible(False)
            if self._start is not None:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        return ax

    #######################################################################
    # SIMULATION
    #######################################################################

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
        trans_probas = self.transition_matrix()[1]
        # A stage that only occurs at the very last epoch has no outgoing transition, and
        # therefore undefined (NaN) transition probabilities. Like in the original hypnogram, we
        # assume that the simulated hypnogram stays in that stage.
        for st in trans_probas.index[trans_probas.isna().all(axis=1)]:
            trans_probas.loc[st] = (trans_probas.columns == st).astype(float)
        simulate_hypnogram_kwargs = {
            "tib": self.duration,
            "n_stages": self.n_stages,
            "freq": self.freq,
            "trans_probas": trans_probas,
            "start": self.start,
            "scorer": self.scorer,
        }
        simulate_hypnogram_kwargs.update(kwargs)
        # By default, the simulation starts from the WAKE row of `trans_probas`. If there is no
        # WAKE, start from the first stage of the current hypnogram instead. This is done after
        # merging `kwargs`, so that it uses the index of a user-defined `trans_probas`.
        trans_probas = simulate_hypnogram_kwargs["trans_probas"]
        first_stage = self.hypno.iloc[0]
        if (
            "init_probas" not in kwargs
            and trans_probas is not None
            and "WAKE" not in trans_probas.index
            and first_stage in trans_probas.index
        ):
            simulate_hypnogram_kwargs["init_probas"] = pd.Series(
                (trans_probas.index == first_stage).astype(float), index=trans_probas.index
            )
        return simulate_hypnogram(**simulate_hypnogram_kwargs)

    #######################################################################
    # PRIVATE METHODS
    #######################################################################

    def _upsample_to_raw_timestamps(self, raw, meas_date_is_local=True, verbose=True):
        """Timestamp-aware upsampling for MNE Raw objects with a valid meas_date.

        Internal method called by :py:meth:`upsample_to_data` when both ``self.start`` and
        ``raw.meas_date`` are available. Aligns the hypnogram to the recording based on absolute
        timestamps rather than sample count.
        """
        set_log_level(verbose)
        epoch_dur = 1.0 / self.sampling_frequency  # seconds per epoch, e.g. 30.0

        # --- resolve and align timestamps ---
        # self.start is a pd.Timestamp (tz-aware when tz was passed at construction, else naive)
        hyp_start = self.start
        # The EDF+ standard defines starttime as local time at the patient's location; MNE
        # reads this value and tags it as UTC. When meas_date_is_local=True (the default),
        # we strip MNE's UTC label so both timestamps are compared as local absolute values.
        # `meas_date` is the time of sample 0 of the original recording, and it is not updated
        # by `raw.crop()`. The first sample of `raw` is `raw.first_time` seconds after it.
        raw_start = pd.Timestamp(raw.info["meas_date"]) + pd.Timedelta(seconds=raw.first_time)
        if meas_date_is_local and raw_start.tzinfo is not None:
            raw_start = raw_start.replace(tzinfo=None)

        if hyp_start.tzinfo is None and raw_start.tzinfo is not None:
            # Only reachable when meas_date_is_local=False and meas_date is UTC-aware.
            # The hypnogram start is naive so YASA cannot safely convert it to UTC.
            raise ValueError(
                "The hypnogram `start` time is timezone-naive but `raw.meas_date` is "
                "timezone-aware (UTC) and `meas_date_is_local=False`. YASA cannot safely "
                "align the two without knowing the timezone of the hypnogram. Either:\n"
                "  • Pass `tz` when creating the Hypnogram, e.g. "
                "Hypnogram(..., tz='Europe/Paris')\n"
                "  • Pass a tz-aware datetime as `start` directly.\n"
                "  • Use `meas_date_is_local=True` (the default) for EDF files, where "
                "starttime is defined as local time at the patient's location (EDF+ standard).\n"
                "Available timezone strings: "
                "import zoneinfo; zoneinfo.available_timezones()"
            )
        elif hyp_start.tzinfo is not None and raw_start.tzinfo is None:
            if meas_date_is_local:
                # Both timestamps are local absolute timestamps — strip tz label without
                # UTC conversion so the stored values can be compared directly.
                hyp_start = hyp_start.replace(tzinfo=None)
            else:
                # Unusual: hypnogram aware, meas_date naive — convert to UTC for arithmetic
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
        return _hypno_upsample_to_data(
            hypno_sliced, self.sampling_frequency, data=raw, verbose=verbose
        )


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
        This function will be removed in v0.9.

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
        "The `yasa.hypno_str_to_int` function is deprecated and will be removed in v0.9. "
        "Please use the `yasa.Hypnogram` class and its `.as_int()` method instead.",
        FutureWarning,
        stacklevel=2,
    )
    assert isinstance(hypno, (list, np.ndarray, pd.Series)), "Not an array."
    hypno = pd.Series(np.asarray(hypno, dtype=str))
    assert not hypno.str.isnumeric().any(), "Hypno contains numeric values."
    return hypno.str.lower().map(mapping_dict).values


def hypno_int_to_str(hypno, mapping_dict=_DEFAULT_INT_TO_STR):
    """Convert an integer hypnogram array to a string array.

    [0, 2, 2, 3, 4] ==> ['W', 'N2', 'N2', 'N3', 'R']

    .. deprecated:: 0.8.0
        Use :py:meth:`yasa.Hypnogram.from_integers` instead. This function will be removed in
        v0.9.

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
    warnings.warn(
        "The `yasa.hypno_int_to_str` function is deprecated and will be removed in v0.9. "
        "Please use `yasa.Hypnogram.from_integers` instead.",
        FutureWarning,
        stacklevel=2,
    )
    return _hypno_int_to_str(hypno, mapping_dict=mapping_dict)


def _hypno_int_to_str(hypno, mapping_dict=_DEFAULT_INT_TO_STR):
    """Convert an integer hypnogram array to a string array. See :py:func:`hypno_int_to_str`."""
    assert isinstance(hypno, (list, np.ndarray, pd.Series)), "Not an array."
    hypno = pd.Series(np.asarray(hypno, dtype=int))
    return hypno.map(mapping_dict).values


#############################################################################
# UPSAMPLING
#############################################################################


def hypno_upsample_to_sf(hypno, sf_hypno, sf_data):
    """Upsample the hypnogram to a given sampling frequency.

    .. deprecated:: 0.8.0
        Use :py:meth:`yasa.Hypnogram.upsample` or :py:meth:`yasa.Hypnogram.upsample_to_data`
        instead. This function will be removed in v0.9.

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
    warnings.warn(
        "The `yasa.hypno_upsample_to_sf` function is deprecated and will be removed in v0.9. "
        "Please use the `yasa.Hypnogram.upsample` or `yasa.Hypnogram.upsample_to_data` "
        "methods instead.",
        FutureWarning,
        stacklevel=2,
    )
    return _hypno_upsample_to_sf(hypno, sf_hypno, sf_data)


def _hypno_upsample_to_sf(hypno, sf_hypno, sf_data):
    """Upsample the hypnogram to a given sampling frequency. See :py:func:`hypno_upsample_to_sf`."""
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
    if npts_hyp == npts_data:
        return hypno
    npts_diff = abs(npts_data - npts_hyp)
    diff = f"{npts_diff / sf:.2f} seconds" if sf is not None else f"{npts_diff} samples"
    if npts_hyp < npts_data:
        # Hypnogram is shorter than data: trailing samples are Unscored (UNS = -2)
        logger.warning(
            f"Hypnogram is SHORTER than data by {diff}. "
            "Padding hypnogram with Unscored (UNS) to match data.size."
        )
        return np.pad(hypno, (0, npts_diff), mode="constant", constant_values=np.int16(-2))
    # Hypnogram is longer than data
    logger.warning(
        f"Hypnogram is LONGER than data by {diff}. Cropping hypnogram to match data.size."
    )
    return hypno[:npts_data]


def hypno_upsample_to_data(hypno, sf_hypno, data, sf_data=None, verbose=True):
    """Upsample an hypnogram to a given sampling frequency and fit the
    resulting hypnogram to corresponding EEG data, such that the hypnogram
    and EEG data have the exact same number of samples.

    .. deprecated:: 0.8.0
        Use :py:meth:`yasa.Hypnogram.upsample_to_data` instead. This function will be removed in
        v0.9.

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
    warnings.warn(
        "The `yasa.hypno_upsample_to_data` function is deprecated and will be removed in v0.9. "
        "Please use the `yasa.Hypnogram.upsample_to_data` method instead.",
        FutureWarning,
        stacklevel=2,
    )
    return _hypno_upsample_to_data(hypno, sf_hypno, data, sf_data=sf_data, verbose=verbose)


def _hypno_upsample_to_data(hypno, sf_hypno, data, sf_data=None, verbose=True):
    """Upsample an hypnogram and fit it to data. See :py:func:`hypno_upsample_to_data`."""
    set_log_level(verbose)
    if isinstance(data, mne.io.BaseRaw):
        sf_data = data.info["sfreq"]
        data = data.times
    hypno_up = _hypno_upsample_to_sf(hypno=hypno, sf_hypno=sf_hypno, sf_data=sf_data)
    return hypno_fit_to_data(hypno=hypno_up, data=data, sf=sf_data)


#############################################################################
# HYPNO LOADING
#############################################################################


def load_profusion_hypno(fname, replace=True):  # pragma: no cover
    """Load a Compumedics Profusion hypnogram (.xml).

    .. deprecated:: 0.7.0
        Use :py:meth:`yasa.Hypnogram.from_profusion` instead, which returns a
        :py:class:`~yasa.Hypnogram` object directly.

    Parameters
    ----------
    fname : str
        Filename with full path.
    replace : bool
        If True (default), integer values are mapped to YASA convention:
        0=Wake, 1=N1, 2=N2, 3=N3/S4, 4=REM. The native Profusion format is
        identical except S4 is encoded as 4, REM as 5 and Active (mapped to Wake) as 9.

    Returns
    -------
    hypno : 1D array
        Hypnogram with one value per epoch.
    sf_hyp : float
        Sampling frequency of the hypnogram (e.g. 1/30 Hz).
    """
    warnings.warn(
        "The `yasa.load_profusion_hypno` function is deprecated and will be removed in v0.9. "
        "Please use the `yasa.Hypnogram.from_profusion` method instead.",
        FutureWarning,
        stacklevel=2,
    )
    hypno, epoch_length = _read_profusion(fname)
    if replace:
        hypno = pd.Series(hypno).replace(_PROFUSION_TO_YASA).to_numpy()
    return hypno, 1 / epoch_length


def _read_profusion(fname):  # pragma: no cover
    """Read the raw integer stages and epoch length (in seconds) of a Profusion XML file."""
    import xml.etree.ElementTree as ET

    root = ET.parse(fname).getroot()
    epoch_length = float(root[0].text)
    hypno_int = np.array([int(s.text) for s in root[4]])
    return hypno_int, epoch_length


#############################################################################
# TRANSITION MATRIX
#############################################################################


def _transition_matrix(hypno):
    """Create a state-transition matrix from an integer array.

    See :py:meth:`yasa.Hypnogram.transition_matrix`. Stages without any outgoing transition (i.e.
    only present in the last epoch) have undefined (NaN) conditional probabilities.
    """
    x = np.asarray(hypno, dtype=int)
    unique, inverse = np.unique(x, return_inverse=True)  # unique is sorted
    n = unique.size
    # Integer transition counts
    counts = np.zeros((n, n), dtype=int)
    np.add.at(counts, (inverse[:-1], inverse[1:]), 1)
    # Conditional probabilities (0 / 0 = NaN for stages without outgoing transitions)
    with np.errstate(invalid="ignore"):
        probs = counts / counts.sum(axis=-1, keepdims=True)
    # Convert to a Pandas DataFrame
    index = pd.Index(unique, name="From Stage")
    columns = pd.Index(unique, name="To Stage")
    return pd.DataFrame(counts, index, columns), pd.DataFrame(probs, index, columns)


#############################################################################
# PERIODS & CYCLES
#############################################################################


def hypno_find_periods(hypno, sf_hypno, threshold="5min", equal_length=False):
    """Find sequences of consecutive values exceeding a certain duration in hypnogram.

    .. deprecated:: 0.8.0
        Use :py:meth:`yasa.Hypnogram.find_periods` instead. This function will be removed in
        v0.9.

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
    warnings.warn(
        "The `yasa.hypno_find_periods` function is deprecated and will be removed in v0.9. "
        "Please use the `yasa.Hypnogram.find_periods` method instead.",
        FutureWarning,
        stacklevel=2,
    )
    return _hypno_find_periods(hypno, sf_hypno, threshold=threshold, equal_length=equal_length)


def _hypno_find_periods(hypno, sf_hypno, threshold="5min", equal_length=False):
    """Find runs of consecutive values in an array. See :py:func:`hypno_find_periods`."""
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

    assert isinstance(hypno, (list, np.ndarray, pd.Series)), "hypno must be an array."
    run_values, run_starts, run_lengths = _find_runs(hypno)
    # Remove runs that are shorter than threshold
    keep = run_lengths >= thr_samp
    run_values, run_starts, run_lengths = run_values[keep], run_starts[keep], run_lengths[keep]

    if equal_length:
        # Divide each run into periods of exactly `thr_samp` samples, discarding the remainder.
        # Since the runs were thresholded above, each run has at least one period.
        assert thr_samp > 0, "Threshold must be non-zero if using equal_length=True."
        n_periods = run_lengths // thr_samp
        # Position of each period within its run, e.g. [0, 1, 2, 0, 1] for runs of 3 and 2 periods
        offset = np.arange(n_periods.sum()) - np.repeat(np.cumsum(n_periods) - n_periods, n_periods)
        run_values = np.repeat(run_values, n_periods)
        run_starts = np.repeat(run_starts, n_periods) + offset * thr_samp
        run_lengths = np.full(run_values.size, thr_samp)

    return pd.DataFrame({"values": run_values, "start": run_starts, "length": run_lengths})


def _find_runs(x):
    """Find runs of consecutive identical values in a 1D array.

    Returns the value, start index and length of each run.
    See https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
    """
    x = np.asarray(x)
    n = x.shape[0]
    loc_run_start = np.empty(n, dtype=bool)
    loc_run_start[:1] = True
    loc_run_start[1:] = x[:-1] != x[1:]
    run_starts = np.flatnonzero(loc_run_start)
    run_lengths = np.diff(np.append(run_starts, n))
    return x[loc_run_start], run_starts, run_lengths


#############################################################################
# SIMULATION
#############################################################################


def simulate_hypnogram(
    tib=300,
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

        .. seealso:: :py:meth:`yasa.Hypnogram.sleep_statistics`
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

        .. seealso:: Return value from :py:meth:`yasa.Hypnogram.transition_matrix`
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
        >>> from yasa import Hypnogram
        >>> values_int = np.loadtxt(yasa.fetch_sample("full_6hrs_100Hz_hypno_30s.txt"))
        >>> real_hyp = Hypnogram.from_integers(values_int)
        >>> fake_hyp = real_hyp.simulate_similar(seed=2)
        >>> fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(7, 5))
        >>> real_hyp.plot_hypnogram(ax=ax1).set_title("Real hypnogram")  # doctest: +SKIP
        >>> fake_hyp.plot_hypnogram(ax=ax2).set_title("Fake hypnogram")  # doctest: +SKIP
        >>> plt.tight_layout()
    """
    # Extract yasa.Hypnogram defaults, which will be assumed later but need throughout
    kwargs.setdefault("n_stages", 5)
    kwargs.setdefault("freq", "30s")
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
        ratio = pd.Timedelta("30s") / pd.Timedelta(kwargs["freq"])
        assert ratio >= 1 and float(ratio).is_integer(), (
            "`freq` must be <= 30s, and 30s must be a whole multiple of `freq`, when using "
            "default `trans_probas`"
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

    use_default = trans_probas is None
    if use_default:
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
    if use_default:
        freq_sec = 30
    else:
        freq_sec = pd.Timedelta(kwargs["freq"]).total_seconds()
    n_epochs = np.floor(tib * 60 / freq_sec).astype(int)

    # Generate hypnogram integer values
    values_int = _markov_sequence(init_arr, trans_arr, n_epochs)
    # Convert to hypnogram string values (based on indices)
    values_str = [stage_order[x] for x in values_int]

    # Create YASA hypnogram instance
    if use_default:
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
