.. _tutorial_hypnogram:

Working with Hypnograms
#######################

.. currentmodule:: yasa

This tutorial introduces the :py:class:`Hypnogram` class, which is the standard way to represent
and work with sleep hypnograms in YASA (since version 0.7).

.. note::

    A hypnogram is a time-series of sleep stage labels, one per epoch (usually 30 seconds). In
    YASA, stages are stored as strings such as ``"WAKE"``, ``"N1"``, ``"N2"``, ``"N3"``, and
    ``"REM"`` (not integers). This makes the data easier to read and less error-prone.

.. contents:: Contents
    :local:
    :depth: 2

--------

Creating a Hypnogram
--------------------

From string labels
~~~~~~~~~~~~~~~~~~

The simplest way to create a :py:class:`Hypnogram` is to pass a list (or array) of stage strings.
YASA supports **2**, **3**, **4**, and **5-stage** hypnograms. Set ``n_stages`` accordingly (the
default is 5). Abbreviated spellings (``"W"``, ``"R"``) and mixed case (``"wake"``, ``"Rem"``)
are accepted. YASA normalizes them automatically.

.. tab-set::

    .. tab-item:: 5-stage (default)

        The 5-stage vocabulary is: ``WAKE``, ``N1``, ``N2``, ``N3``, ``REM``.

        .. code-block:: python

            >>> import yasa
            >>> hyp = yasa.Hypnogram(["WAKE", "WAKE", "N1", "N2", "N2", "N3", "N2", "REM", "WAKE"])
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
            4      N2
            5      N3
            6      N2
            7     REM
            8    WAKE
            Name: Stage, dtype: category
            Categories (7, str): ['WAKE', 'N1', 'N2', 'N3', 'REM', 'ART', 'UNS']

    .. tab-item:: 4-stage

        The 4-stage vocabulary is: ``WAKE``, ``LIGHT``, ``DEEP``, ``REM``.

        .. code-block:: python

            >>> import yasa
            >>> hyp = yasa.Hypnogram(
            ...     ["WAKE", "WAKE", "LIGHT", "LIGHT", "DEEP", "DEEP", "REM", "WAKE"],
            ...     n_stages=4,
            ... )
            >>> hyp
            <Hypnogram | 8 epochs x 30s (4.00 minutes), 4 unique stages>
             - Use `.hypno` to get the string values as a pandas.Series
             - Use `.as_int()` to get the integer values as a pandas.Series
             - Use `.plot_hypnogram()` to plot the hypnogram
            See the online documentation for more details.

            >>> hyp.hypno
            Epoch
            0     WAKE
            1     WAKE
            2    LIGHT
            3    LIGHT
            4     DEEP
            5     DEEP
            6      REM
            7     WAKE
            Name: Stage, dtype: category
            Categories (6, str): ['WAKE', 'LIGHT', 'DEEP', 'REM', 'ART', 'UNS']

    .. tab-item:: 2-stage

        The 2-stage vocabulary is: ``WAKE``, ``SLEEP``.

        .. code-block:: python

            >>> import yasa
            >>> hyp = yasa.Hypnogram(
            ...     ["W", "W", "S", "S", "S", "W", "S"],
            ...     n_stages=2,
            ... )
            >>> hyp
            <Hypnogram | 7 epochs x 30s (3.50 minutes), 2 unique stages>
             - Use `.hypno` to get the string values as a pandas.Series
             - Use `.as_int()` to get the integer values as a pandas.Series
             - Use `.plot_hypnogram()` to plot the hypnogram
            See the online documentation for more details.

            >>> hyp.hypno
            Epoch
            0     WAKE
            1     WAKE
            2    SLEEP
            3    SLEEP
            4    SLEEP
            5     WAKE
            6    SLEEP
            Name: Stage, dtype: category
            Categories (4, str): ['WAKE', 'SLEEP', 'ART', 'UNS']

.. note::

    ``ART`` (Artefact) and ``UNS`` (Unscored) are always part of the vocabulary regardless of
    ``n_stages``, but they are never required.

From integer arrays (legacy format)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many older pipelines store hypnograms as integer arrays. Use the
:py:meth:`Hypnogram.from_integers` class method to convert them.

.. tab-set::

    .. tab-item:: 5-stage (default)

        The default mapping is: 0 = Wake, 1 = N1, 2 = N2, 3 = N3, 4 = REM.

        .. code-block:: python

            >>> import numpy as np
            >>> import yasa
            >>> hyp = yasa.Hypnogram.from_integers(np.array([0, 0, 1, 2, 3, 2, 4, 4, 0]))
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

        To load from a file:

        .. code-block:: python

            >>> import pandas as pd
            >>> int_hypno = pd.read_csv("hypnogram.csv").squeeze().to_numpy()  # doctest: +SKIP
            >>> hyp = yasa.Hypnogram.from_integers(int_hypno, freq="30s", scorer="Expert")  # doctest: +SKIP

    .. tab-item:: 4-stage

        Pass a custom ``mapping`` dictionary when your integer encoding differs from the default.

        .. code-block:: python

            >>> import yasa
            >>> hyp = yasa.Hypnogram.from_integers(
            ...     [0, 0, 2, 2, 3, 3, 4, 0],
            ...     mapping={0: "WAKE", 2: "LIGHT", 3: "DEEP", 4: "REM"},
            ...     n_stages=4,
            ... )
            >>> hyp.hypno
            Epoch
            0     WAKE
            1     WAKE
            2    LIGHT
            3    LIGHT
            4     DEEP
            5     DEEP
            6      REM
            7     WAKE
            Name: Stage, dtype: category
            Categories (6, str): ['WAKE', 'LIGHT', 'DEEP', 'REM', 'ART', 'UNS']

    .. tab-item:: 2-stage

        Pass a custom ``mapping`` dictionary and set ``n_stages=2``.

        .. code-block:: python

            >>> import yasa
            >>> hyp = yasa.Hypnogram.from_integers(
            ...     [0, 0, 1, 1, 1, 0, 1],
            ...     mapping={0: "W", 1: "S"},
            ...     n_stages=2,
            ... )
            >>> hyp.hypno
            Epoch
            0     WAKE
            1     WAKE
            2    SLEEP
            3    SLEEP
            4    SLEEP
            5     WAKE
            6    SLEEP
            Name: Stage, dtype: category
            Categories (4, str): ['WAKE', 'SLEEP', 'ART', 'UNS']

From a Compumedics Profusion XML file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hypnograms in the `NSRR <https://sleepdata.org/>`_ format can be loaded directly:

.. code-block:: python

    >>> hyp = yasa.Hypnogram.from_profusion("path/to/hypnogram.xml")  # doctest: +SKIP

Simulated hypnograms
~~~~~~~~~~~~~~~~~~~~

For testing and demonstration, :py:func:`simulate_hypnogram` generates a realistic 5-stage
hypnogram with physiologically plausible stage transitions:

.. code-block:: python

    >>> hyp = yasa.simulate_hypnogram(tib=480, n_stages=5, seed=42)
    >>> hyp
    <Hypnogram | 960 epochs x 30s (480.00 minutes), 5 unique stages>
     - Use `.hypno` to get the string values as a pandas.Series
     - Use `.as_int()` to get the integer values as a pandas.Series
     - Use `.plot_hypnogram()` to plot the hypnogram
    See the online documentation for more details.

Adding a start time
~~~~~~~~~~~~~~~~~~~

Attaching a recording start time turns the epoch index into a
:py:class:`pandas.DatetimeIndex`:

.. code-block:: python

    >>> hyp = yasa.simulate_hypnogram(
    ...     tib=480, n_stages=5, start="2024-01-15 23:00:00", seed=42
    ... )
    >>> hyp.start
    Timestamp('2024-01-15 23:00:00')
    >>> hyp.end
    Timestamp('2024-01-16 07:00:00')

The timezone can be set with the ``tz`` parameter:

.. code-block:: python

    >>> hyp = yasa.Hypnogram(
    ...     ["WAKE", "N1", "N2", "N3", "REM"],
    ...     start="2024-01-15 23:00:00",
    ...     tz="America/New_York",
    ... )

--------

Exploring the data
------------------

The stage labels are stored as a :py:class:`pandas.Series` and can be accessed with the
:py:attr:`~Hypnogram.hypno` property, which inherits all standard pandas methods
(``.describe()``, ``.value_counts()``, ``.to_csv()``, ...):

.. code-block:: python

    >>> hyp = yasa.simulate_hypnogram(tib=480, n_stages=5, seed=42)
    >>> hyp.hypno.value_counts()
    Stage
    N2      481
    WAKE    164
    N1      134
    N3      106
    REM      75
    ART       0
    UNS       0
    Name: count, dtype: int64

A few useful properties:

.. code-block:: python

    >>> hyp.n_epochs     # number of 30-s epochs
    960
    >>> hyp.duration     # total recording duration in minutes
    480.0
    >>> hyp.freq         # epoch length as a pandas offset string
    '30s'
    >>> hyp.n_stages     # number of sleep stages (2 / 3 / 4 / 5)
    5

To get integer-encoded stages (compatible with legacy YASA functions), use :py:meth:`~Hypnogram.as_int`:

.. code-block:: python

    >>> hyp.as_int().head()
    Epoch
    0    0
    1    0
    2    0
    3    0
    4    0
    Name: Stage, dtype: int16

To get a `BIDS <https://bids-specification.readthedocs.io>`_-compatible events table (onset, duration, stage) use :py:meth:`~Hypnogram.as_events`:

.. code-block:: python

    >>> hyp.as_events().head()
           onset  duration  value description
    epoch
    0        0.0      30.0      0        WAKE
    1       30.0      30.0      0        WAKE
    2       60.0      30.0      0        WAKE
    3       90.0      30.0      0        WAKE
    4      120.0      30.0      0        WAKE

Boolean masks
~~~~~~~~~~~~~

:py:meth:`~Hypnogram.get_mask` returns a boolean NumPy array marking which epochs belong to the
specified stages. This is useful for indexing EEG data arrays or computing stage-specific metrics:

.. code-block:: python

    >>> nrem_mask = hyp.get_mask(["N2", "N3"])
    >>> nrem_mask[:5]
    array([False, False, False, False, False])

    >>> # How many NREM epochs?
    >>> nrem_mask.sum()
    587

Slicing and cropping
~~~~~~~~~~~~~~~~~~~~

You can slice a :py:class:`Hypnogram` with Python's standard indexing syntax. The result is
always a new :py:class:`Hypnogram`:

.. code-block:: python

    >>> hyp = yasa.simulate_hypnogram(tib=480, n_stages=5, seed=42)

    >>> # First epoch
    >>> hyp[0].hypno.iloc[0]
    'WAKE'

    >>> # Epochs 100 to 199
    >>> hyp[100:200]
    <Hypnogram | 100 epochs x 30s (50.00 minutes), 5 unique stages>
     - Use `.hypno` to get the string values as a pandas.Series
     - Use `.as_int()` to get the integer values as a pandas.Series
     - Use `.plot_hypnogram()` to plot the hypnogram
    See the online documentation for more details.

For time-based slicing when a start time is set, use :py:meth:`~Hypnogram.crop`:

.. code-block:: python

    >>> hyp = yasa.simulate_hypnogram(tib=480, start="2024-01-15 23:00:00", seed=42)
    >>> hyp_night = hyp.crop("2024-01-15 23:30:00", "2024-01-16 06:00:00")  # doctest: +SKIP

Padding
~~~~~~~

:py:meth:`~Hypnogram.pad` lets you extend the hypnogram before or after with a fill stage,
which is handy when aligning hypnograms recorded at different times:

.. code-block:: python

    >>> hyp = yasa.Hypnogram(["N2", "N2", "REM"])
    >>> hyp.pad(before=2, after=1, fill_value="WAKE")
    <Hypnogram | 6 epochs x 30s (3.00 minutes), 5 unique stages>
     - Use `.hypno` to get the string values as a pandas.Series
     - Use `.as_int()` to get the integer values as a pandas.Series
     - Use `.plot_hypnogram()` to plot the hypnogram
    See the online documentation for more details.

--------

Sleep statistics
----------------

:py:meth:`~Hypnogram.sleep_statistics` returns a dictionary of standard AASM metrics: Total
Sleep Time (TST), Sleep Efficiency (SE), Wake After Sleep Onset (WASO), stage durations, and more:

.. code-block:: python

    >>> import pandas as pd
    >>> pd.Series(hyp.sleep_statistics())
    TIB         480.0000
    SPT         477.5000
    WASO         79.5000
    TST         398.0000
    SE           82.9167
    SME          83.3508
    SFI           0.7538
    SOL           2.5000
    SOL_5min      2.5000
    Lat_REM      67.0000
    WAKE         82.0000
    N1           67.0000
    N2          240.5000
    N3           53.0000
    REM          37.5000
    %N1          16.8342
    %N2          60.4271
    %N3          13.3166
    %REM          9.4221
    dtype: float64

Stage-transition matrix
~~~~~~~~~~~~~~~~~~~~~~~

:py:meth:`~Hypnogram.transition_matrix` returns two DataFrames: the raw transition counts and
the row-normalized probability matrix. The probability matrix answers the question: *"Given that
the current epoch is stage A, how likely is the next epoch to be stage B?"*

.. code-block:: python

    >>> counts, probs = hyp.transition_matrix()
    >>> probs.round(3)
    To Stage     WAKE     N1     N2     N3   REM
    From Stage
    WAKE        0.933  0.067  0.000  0.000  0.00
    N1          0.045  0.739  0.216  0.000  0.00
    N2          0.006  0.033  0.929  0.021  0.01
    N3          0.010  0.029  0.038  0.914  0.01
    REM         0.000  0.067  0.013  0.000  0.92

Sleep periods
~~~~~~~~~~~~~

:py:meth:`~Hypnogram.find_periods` detects consecutive runs of a given stage that exceed a minimum
duration. This is useful for identifying stable sleep bouts, wake periods, or REM episodes:

.. code-block:: python

    >>> # Find all REM periods lasting at least 5 minutes
    >>> hyp.find_periods(threshold="5min").query("values == 'REM'")

Merging stages
~~~~~~~~~~~~~~

:py:meth:`~Hypnogram.consolidate_stages` collapses a fine-grained hypnogram into a coarser one.
This is useful when a downstream analysis does not require the full 5-stage resolution:

.. code-block:: python

    >>> hyp = yasa.simulate_hypnogram(tib=480, n_stages=5, seed=42)

    >>> # 5-stage to 3-stage (Wake / NREM / REM)
    >>> hyp3 = hyp.consolidate_stages(3)
    >>> hyp3.hypno.value_counts()
    Stage
    NREM    721
    WAKE    164
    REM      75
    ART       0
    UNS       0
    Name: count, dtype: int64

    >>> # 5-stage to 2-stage (Wake / Sleep)
    >>> hyp2 = hyp.consolidate_stages(2)
    >>> hyp2.hypno.value_counts()
    Stage
    SLEEP    796
    WAKE     164
    ART        0
    UNS        0
    Name: count, dtype: int64

--------

Visualization
-------------

Hypnogram plot
~~~~~~~~~~~~~~

:py:meth:`~Hypnogram.plot_hypnogram` draws the classic staircase hypnogram:

.. code-block:: python

    >>> hyp.plot_hypnogram()

.. plot::

    import matplotlib.pyplot as plt
    import yasa
    hyp = yasa.simulate_hypnogram(tib=480, n_stages=5, seed=42)
    fig, ax = plt.subplots(figsize=(12, 4))
    hyp.plot_hypnogram(ax=ax)
    fig.tight_layout()

Hypnodensity plot
~~~~~~~~~~~~~~~~~

When stage probabilities are available (e.g. from :py:class:`SleepStaging`),
:py:meth:`~Hypnogram.plot_hypnodensity` shows the per-epoch probability of each stage as a
stacked area chart, giving a more nuanced view of sleep dynamics:

.. code-block:: python

    >>> sls = yasa.SleepStaging(raw, eeg_name="C3-A2")  # doctest: +SKIP
    >>> hyp = sls.predict()                              # doctest: +SKIP
    >>> hyp.plot_hypnodensity()                          # doctest: +SKIP

The example below uses a simulated hypnogram with synthetic stage probabilities to illustrate
the output:

.. plot::

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import yasa
    hyp = yasa.simulate_hypnogram(tib=480, n_stages=5, seed=42)
    stages = ["WAKE", "N1", "N2", "N3", "REM"]
    rng = np.random.default_rng(42)
    one_hot = (
        pd.get_dummies(hyp.hypno)
        .reindex(columns=stages, fill_value=0)
        .to_numpy(dtype=float)
    )
    noise = rng.dirichlet(np.ones(5) * 0.5, size=hyp.n_epochs)
    raw_proba = 0.75 * one_hot + 0.25 * noise
    proba = pd.DataFrame(raw_proba / raw_proba.sum(axis=1, keepdims=True), columns=stages)
    hyp_with_proba = yasa.Hypnogram(hyp.hypno, n_stages=5, proba=proba)
    fig, ax = plt.subplots(figsize=(12, 4))
    hyp_with_proba.plot_hypnodensity(ax=ax)
    fig.tight_layout()

--------

Aligning with EEG data
-----------------------

To use a hypnogram alongside raw EEG data, YASA needs a sample-level label for every EEG sample,
not just one per 30-second epoch. :py:meth:`~Hypnogram.upsample_to_data` handles this
automatically:

.. code-block:: python

    >>> import mne
    >>> raw = mne.io.read_raw_edf("recording.edf", preload=True)  # doctest: +SKIP
    >>> hyp = yasa.Hypnogram.from_integers(int_hypno, freq="30s") # doctest: +SKIP
    >>> hypno_up = hyp.upsample_to_data(raw)                      # doctest: +SKIP

.. tip::

    As of YASA 0.7, most detection functions accept a :py:class:`Hypnogram` object directly.
    No manual upsampling is needed. Just pass ``hypno=hyp``:

    .. code-block:: python

        >>> sp = yasa.spindles_detect(raw, hypno=hyp, include=["N2", "N3"])  # doctest: +SKIP

Alignment modes
~~~~~~~~~~~~~~~

The behavior of :py:meth:`~Hypnogram.upsample_to_data` depends on whether timestamp information
is available.

**Length-based alignment (default)**

YASA assumes the hypnogram and the recording start at the same time. Any length mismatch is
resolved by padding or cropping at the end. This mode is always used when ``data`` is a NumPy
array. It is also used when ``data`` is an :py:class:`mne.io.BaseRaw` but either
``Hypnogram.start`` is not set or ``raw.meas_date`` is ``None``.

.. code-block:: python

    >>> hyp = yasa.Hypnogram(stages, freq="30s")  # doctest: +SKIP
    >>> hypno = hyp.upsample_to_data(raw)          # doctest: +SKIP

**Timestamp-aware alignment**

Triggered automatically when *both* ``Hypnogram.start`` and ``raw.meas_date`` are set. YASA
computes the absolute offset between the two timestamps and selects the correct hypnogram epochs.

.. code-block:: python

    >>> # EDF recorded at 22:11:37 local time
    >>> hyp = yasa.Hypnogram(stages, freq="30s", start="2024-11-08 22:11:37")  # doctest: +SKIP
    >>> hypno = hyp.upsample_to_data(raw)  # doctest: +SKIP

Common scenarios
~~~~~~~~~~~~~~~~

**Hypnogram and recording cover the same window**

The hypnogram is upsampled and fits the data exactly. Both alignment modes give the same result.

**Hypnogram is shorter than the recording**

This happens when the hypnogram covers only the Lights Off to Lights On period while the PSG spans
a longer window.

* *Length-based*: the hypnogram is padded with Unscored (``UNS``) at the end.
* *Timestamp-aware*: the correct number of ``UNS`` epochs is prepended before Lights Off, and any
  remaining tail is also padded.

.. code-block:: python

    >>> hyp = yasa.Hypnogram(stages, freq="30s", start="2024-01-15 23:00:00")  # doctest: +SKIP
    >>> hypno = hyp.upsample_to_data(raw)  # doctest: +SKIP
    >>> # Epochs before Lights Off and after Lights On become UNS

**Hypnogram is longer than the recording**

This happens when working with a cropped segment of a full-night recording.

* *Length-based*: the hypnogram is cropped from the end. This is only correct if the segment
  starts at the very beginning of the recording.
* *Timestamp-aware*: YASA skips the correct leading epochs based on the timestamp offset and
  selects only the epochs that fall within the recording window.

.. code-block:: python

    >>> # Full-night hypnogram, but only the second half of the night is loaded
    >>> hyp = yasa.Hypnogram(stages, freq="30s", start="2024-01-15 23:00:00")  # doctest: +SKIP
    >>> hypno = hyp.upsample_to_data(raw_cropped)  # correct epochs selected automatically  # doctest: +SKIP

**Automatic staging with** :py:class:`~yasa.SleepStaging`

When using :py:class:`~yasa.SleepStaging`, the ``start`` attribute is populated automatically
from ``raw.meas_date`` when available, so timestamp-aware alignment works out of the box:

.. code-block:: python

    >>> sls = yasa.SleepStaging(raw, eeg_name="C4-M1")  # doctest: +SKIP
    >>> hyp = sls.predict()  # hyp.start set automatically from raw.meas_date  # doctest: +SKIP
    >>> hypno = hyp.upsample_to_data(raw_cropped)  # doctest: +SKIP

--------

Saving and loading
------------------

A :py:class:`Hypnogram` including all metadata (epoch length, start time, scorer, stage
probabilities) can be saved to a JSON file and reloaded later:

.. code-block:: python

    >>> hyp = yasa.simulate_hypnogram(tib=480, n_stages=5, scorer="Expert", seed=42)

    >>> # Save to disk
    >>> hyp.to_json("my_hypnogram.json")  # doctest: +SKIP

    >>> # Reload it later, all metadata is preserved
    >>> hyp2 = yasa.Hypnogram.from_json("my_hypnogram.json")  # doctest: +SKIP

You can also use :py:meth:`~Hypnogram.to_dict` /
:py:meth:`~Hypnogram.from_dict`, which produce a plain JSON-serializable Python dictionary in
the same format as :py:meth:`~Hypnogram.to_json`.

--------

Comparing two hypnograms
------------------------

:py:meth:`~Hypnogram.evaluate` computes epoch-by-epoch agreement metrics between a reference and
an observer hypnogram, including Cohen's kappa, Matthews correlation coefficient, and per-stage
F1 scores:

.. code-block:: python

    >>> ref = yasa.simulate_hypnogram(tib=480, n_stages=5, scorer="Expert", seed=42)
    >>> obs = yasa.simulate_hypnogram(tib=480, n_stages=5, scorer="Auto",   seed=99)
    >>> agreement = ref.evaluate(obs)  # doctest: +SKIP

.. note::

    :py:meth:`~Hypnogram.evaluate` is experimental and the output format may change in future
    releases.

--------

Next steps
----------

* :ref:`quickstart`: end-to-end walkthrough with real PSG data.
* :ref:`api_ref`: full API reference for :py:class:`Hypnogram` and all other YASA classes.
* `Jupyter notebooks <https://github.com/raphaelvallat/yasa/tree/master/notebooks>`_: hands-on
  notebooks with downloadable example datasets.
