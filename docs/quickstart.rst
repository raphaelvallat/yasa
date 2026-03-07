.. _quickstart:

Quickstart
##########

.. .. sectnum::
..   :depth: 2
..   :suffix: )

Prerequisites
-------------

This is an introduction to `YASA <https://github.com/raphaelvallat/yasa>`_, geared mainly for new users. However, you'll need to know a bit of Python, and especially scientific libraries such as `NumPy <https://numpy.org/doc/stable/user/quickstart.html>`_ and `Pandas <https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html>`_. This tutorial also assumes that you are familiar with basic sleep research and methods for analyzing `polysomnography <https://en.wikipedia.org/wiki/Polysomnography>`_ (PSG) data.

Make sure to install the latest version of YASA by opening a terminal and entering: ``uv pip install yasa`` (or ``pip install --upgrade yasa``).

The sample files used in this tutorial can be downloaded automatically using :py:func:`yasa.fetch_sample`:

.. code-block:: python

    >>> import yasa
    >>> edf_path = yasa.fetch_sample("night_young.edf")
    >>> hypno_path = yasa.fetch_sample("night_young_hypno.csv")

The PSG recording is a full-night polysomnography from a 22-year-old healthy female (19 EEGs, 2 EOGs, 1 EMG, 1 EKG), spanning 11:59 PM to 08:01 AM. The hypnogram is in 30-second epochs encoded as: 0 = Wake, 1 = N1, 2 = N2, 3 = N3, 4 = REM.

********

Data loading and preprocessing
------------------------------

Polysomnography data
~~~~~~~~~~~~~~~~~~~~

We will use the `MNE package <https://mne.tools/stable/index.html>`_ to load and preprocess the data in Python:

.. code-block:: python

    >>> import mne
    >>> raw = mne.io.read_raw_edf(edf_path, preload=True)
    >>> raw
    <RawEDF | night_young.edf, 23 x 5784000 (28920.0 s), ~1015.0 MB, data loaded>

.. note::

    YASA does not allow to visualize or scroll through the PSG data. However, this can be done using the free `EDFBrowser <https://www.teuniz.net/edfbrowser/>`_ software.

**Selecting channels**

The channel names can be shown with:

.. code-block:: python

    >>> print(raw.ch_names)
    ['ROC-A1', 'LOC-A2', 'C3-A2', 'O2-A1', 'C4-A1', 'O1-A2', 'EMG1-EMG2', 'Fp1-A2', 'Fp2-A1', 'F7-A2',
    'F3-A2', 'FZ-A2', 'F4-A1', 'F8-A1', 'T3-A2', 'CZ-A2', 'T4-A1', 'T5-A2', 'P3-A2', 'PZ-A2', 'P4-A1',
    'T6-A1', 'EKG-R-EKG-L']

Let's remove the EOG, EMG and EKG channels:

.. code-block:: python

    >>> raw.drop_channels(["ROC-A1", "LOC-A2", "EMG1-EMG2", "EKG-R-EKG-L"])
    >>> chan = raw.ch_names
    >>> print(chan)
    ['C3-A2', 'O2-A1', 'C4-A1', 'O1-A2', 'Fp1-A2', 'Fp2-A1', 'F7-A2', 'F3-A2', 'FZ-A2',
    'F4-A1', 'F8-A1', 'T3-A2', 'CZ-A2', 'T4-A1', 'T5-A2', 'P3-A2', 'PZ-A2', 'P4-A1', 'T6-A1']

**Downsampling and filtering**

The sampling frequency of the data in Hertz (Hz) is given by:

.. code-block:: python

    >>> print(raw.info["sfreq"])
    200.00

The current sampling frequency of the data is therefore 200 Hz. To speed up computation, let's downsample the data to 100 Hz:

.. code-block:: python

    >>> raw.resample(100)
    >>> sf = raw.info["sfreq"]
    >>> sf
    100.0

Optionally, we can apply a 0.3-45 Hz bandpass-filter:

.. code-block:: python

    >>> raw.filter(0.3, 45)

Finally, the underlying data can be accessed with:

.. code-block:: python

    >>> data = raw.get_data(units="uV")
    >>> print(data.shape)
    (19, 2892000)

In this example, ``data`` is a two-dimensional NumPy array where the rows represent the channels (19 EEG channels) and the columns represent the data samples (~3 million samples per channel).

Hypnogram
~~~~~~~~~

Sleep staging (aka hypnogram) for this example night was performed by a trained technician following the standard rules of the American Academy of Sleep Medicine (AASM).
The output is saved in a .csv file, where each row represents 30 seconds of data. The stages are mapped to integers such that 0 = Wake, 1 = N1 sleep, 2 = N2 sleep, 3 = N3 sleep and 4 = REM sleep.
We can load this file using :py:func:`pandas.read_csv` and then convert the integer stages to a :py:class:`~yasa.Hypnogram` object using :py:meth:`~yasa.Hypnogram.from_integers`:

.. code-block:: python

    >>> import pandas as pd
    >>> import yasa
    >>> hypno = pd.read_csv(hypno_path).squeeze().to_numpy()
    >>> hyp = yasa.Hypnogram.from_integers(hypno, freq="30s", scorer="Expert")
    >>> hyp
    <Hypnogram | 964 epochs x 30s (482.00 minutes), 5 unique stages, scored by Expert>
     - Use `.hypno` to get the string values as a pandas.Series
     - Use `.as_int()` to get the integer values as a pandas.Series
     - Use `.plot_hypnogram()` to plot the hypnogram
    See the online documentation for more details.

.. note::

    If you do not have sleep staging, you can use YASA to automatically detect the sleep stages for you. We'll come back to this later on in this tutorial.

.. tip::

    If available, we recommend passing the hypnogram's start time via the ``start`` argument when
    creating the :py:class:`~yasa.Hypnogram`:

    .. code-block:: python

        >>> hyp = yasa.Hypnogram.from_integers(
        ...     hypno, freq="30s", scorer="Expert",
        ...     start="2024-01-15 23:59:00", tz="Europe/Paris"
        ... )

    The ``tz`` argument localizes the naive start string to your local timezone. Alternatively,
    pass a tz-aware :py:class:`datetime.datetime` directly as ``start`` and omit ``tz``.
    When you later call :py:meth:`~yasa.Hypnogram.upsample_to_data`, YASA will automatically
    compute the offset between the hypnogram start and the recording start, and select
    the correct epochs:

    .. code-block:: python

        >>> hypno_up = hyp.upsample_to_data(raw_cropped)

    If ``start`` is not set, or if ``data`` is a NumPy array, the hypnogram is upsampled to
    match the number of samples in ``data``. Any samples beyond the hypnogram end are set to
    **Unscored** (``UNS``, integer value ``-2``).

It's easy to plot the hypnogram:

.. code-block:: python

    >>> hyp.plot_hypnogram();

.. figure:: https://raw.githubusercontent.com/raphaelvallat/yasa/refs/tags/v0.6.5/docs/pictures/quickstart/hypnogram.png
    :align: center

****************

Sleep statistics and stage-transition matrix
--------------------------------------------

Using the hypnogram, we can calculate standard sleep statistics using the :py:meth:`~yasa.Hypnogram.sleep_statistics` method:

.. code-block:: python

    >>> hyp.sleep_statistics()
    {'TIB': 482.0,
    'SPT': 468.5,
    'WASO': 9.0,
    'TST': 459.5,
    'SE': 95.332,
    'SME': 98.079,
    'SFI': 0.6529,
    'SOL': 13.0,
    'SOL_5min': 14.5,
    'Lat_REM': 77.0,
    'WAKE': 22.5,
    'N1': 17.5,
    'N2': 214.0,
    'N3': 85.5,
    'REM': 142.5,
    '%N1': 3.8085,
    '%N2': 46.5724,
    '%N3': 18.6072,
    '%REM': 31.012}

Furthermore, we can also calculate the sleep stages transition matrix using the :py:meth:`~yasa.Hypnogram.transition_matrix` method:

.. code-block:: python

    >>> counts, probs = hyp.transition_matrix()
    >>> probs.round(3)

============  ======  =====  =====  =====  =====
From Stage      WAKE     N1     N2     N3    REM
============  ======  =====  =====  =====  =====
WAKE           0.773  0.205  0.023  0      0
N1             0.086  0.629  0.257  0      0.029
N2             0.009  0.002  0.876  0.103  0.009
N3             0.006  0.006  0.246  0.743  0
REM            0.007  0.007  0.004  0      0.982
============  ======  =====  =====  =====  =====

``probs`` is the probability transition matrix, i.e. given that the current sleep stage is A, what is the probability that the next sleep stage is B.

Several metrics of sleep fragmentation can be calculated from ``probs``. For example, the *stability of sleep* can be calculated by taking the average of the diagonal values of N2, N3 and REM sleep:

.. code-block:: python

    >>> import numpy as np
    >>> np.diag(probs.loc["N2":, "N2":]).mean().round(3)

********

Spectral analyses
-----------------

Full-night spectrogram plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

    As of YASA 0.7.0, you can pass the :py:class:`~yasa.Hypnogram` object directly to most YASA functions —
    upsampling to the data sampling frequency is handled automatically.

We can plot the hypnogram on top of a multitaper `spectrogram <https://en.wikipedia.org/wiki/Spectrogram>`_ using the :py:func:`~yasa.plot_spectrogram` function, which shows the time-frequency representation of a single EEG channel across the entire night. The x-axis of the spectrogram is time in hours, and the y-axis is the frequency range (from 0 to 25 Hz).
Warmer colors indicate higher spectral power in this specific frequency band at this specific time for this channel. This kind of plot is very useful to quickly identify periods of NREM sleep (high power in frequencies below 5 Hz and spindle-related activity around ~14 Hz) and REM sleep (almost no power in frequencies below 5 Hz).

.. code-block:: python

    # We select only the C4-A1 EEG channel.
    >>> yasa.plot_spectrogram(data[chan.index("C4-A1")], sf, hyp);

.. image:: https://raw.githubusercontent.com/raphaelvallat/yasa/refs/tags/v0.6.5/docs/pictures/quickstart/spectrogram.png
    :align: center

.. note::

    Whenever you start a new analysis in YASA, we always recommend that you use the :py:func:`~yasa.plot_spectrogram` function to check your data. This can help you easily identify artefact in the data or misalignement between the PSG data and hypnogram.

EEG power in specific frequency bands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Spectral analysis quantifies the power (or amplitude) of the EEG signal in different frequency bands. In neuroscience, the most common frequency bands are **delta** (0.5–4 Hz), **theta** (4–8 Hz), **alpha** (8–12 Hz), **beta** (12–30 Hz), and **gamma** (30–~100 Hz). There are numerous studies that have reported significant relationship between the EEG power spectrum and human behavior, cognitive state, or mental illnesses, and EEG spectral analysis is now one of the principal analysis methods in the field of neuroscience and sleep research.
It is especially relevant for sleep analysis, as it is well-known that the different stages of sleep vary drastically in their spectral content. For example, deep slow-wave sleep (N3) is associated with increased power in the low frequencies, especially the delta band (0.5-4Hz), and decreased power in the beta and gamma bands.

Calculating the average spectral power in different frequency bands is straightforward with the :py:func:`~yasa.bandpower` function:

.. code-block:: python

    >>> yasa.bandpower(raw)

.. image:: https://raw.githubusercontent.com/raphaelvallat/yasa/refs/tags/v0.6.5/docs/pictures/quickstart/bandpower.png
    :align: center

This calculates, for each channel separately, the average power in the main frequency bands across the entire recording. Importantly, the values are **relative** power, i.e. they are expressed as a proportion of the total power between the lowest frequency (default 0.5 Hz) and the highest frequency (default 40 Hz). We can disable this behavior and get the **absolute** spectral power values in :math:`μV^2 / Hz` by using the ``relative=False`` argument. Similarly, we can define custom frequency bands with the ``bands`` parameter. In the example below, we calculate the absolute power in the 1-9 Hz frequency range (named "Slow") and the 9-30 Hz range (named "Fast"):

.. code-block:: python

    >>> yasa.bandpower(raw, relative=False, bands=[(1, 9, "Slow"), (9, 30, "Fast")])

.. image:: https://raw.githubusercontent.com/raphaelvallat/yasa/refs/tags/v0.6.5/docs/pictures/quickstart/bandpower2.png
    :align: center

We can also pass a hypnogram to calculate the spectral powers separately for each sleep stage. In the example below, we pass the :py:class:`~yasa.Hypnogram` object directly and use string stage labels for ``include``. We save the results in a new variable named ``bandpower``.

.. code-block:: python

    >>> bandpower = yasa.bandpower(raw, hypno=hyp, include=["N2", "N3", "REM"])

If desired, we can then export the ``bandpower`` dataframe to a CSV file using :py:meth:`pandas.DataFrame.to_csv`:

.. code-block:: python

    >>> bandpower.to_csv("bandpower.csv")

Finally, we can use the :py:func:`~yasa.topoplot` function to visualize the spectral powers across all electrodes. In the example below, we only plot the spectral values of stage N3, using the :py:meth:`pandas.DataFrame.xs` function. As expected, the relative delta power is higher in frontal channels.

.. code-block:: python

    >>> fig = yasa.topoplot(bandpower.xs("N3")["Delta"])

.. image:: https://raw.githubusercontent.com/raphaelvallat/yasa/refs/tags/v0.6.5/docs/pictures/quickstart/topoplot.png
    :align: center

********

Events detection
----------------

Spindles
~~~~~~~~

Automatic spindles detection can be performed with the :py:func:`yasa.spindles_detect` function. The detection is based on the algorithm described in `Lacourse et al 2018 <https://pubmed.ncbi.nlm.nih.gov/30107208/>`_, and a step-by-step explanation is provided in `this notebook <https://github.com/raphaelvallat/yasa/blob/master/notebooks/01_spindles_detection.ipynb>`_. For the sake of this tutorial, we'll use the default detection thresholds, but these can (and should) be adjusted based on your own data. In the example below, we pass the :py:class:`~yasa.Hypnogram` object directly and limit the detection to N2 and N3 sleep using string stage labels.

.. code-block:: python

    >>> sp = yasa.spindles_detect(raw, hypno=hyp, include=["N2", "N3"])

Here, the ``sp`` variable is a :py:class:`~yasa.SpindlesResults`, which is simply a bundle of functions (called methods) and data (attributes). For example, we can see a dataframe with all the detected events with:

.. code-block:: python

    >>> sp.summary()

.. image:: https://raw.githubusercontent.com/raphaelvallat/yasa/refs/tags/v0.6.5/docs/pictures/quickstart/spindles_summary.png
    :align: center

The documentation of the :py:func:`~yasa.spindles_detect` function explains what each of these columns represent and how they're calculated. Furthermore, by specifying the ``grp_chan`` and ``grp_stage`` parameters, we tell YASA to first average across channels and sleep stages, respectively:

.. code-block:: python

    >>> sp.summary(grp_chan=True, grp_stage=True)

Finally, we can plot the average spindle, calculated for each channel separately and time-synced to the most prominent peak of the spindles:

.. code-block:: python

    >>> # Because of the large number of channels, we disable the 95%CI and legend
    >>> sp.plot_average(ci=None, legend=False, palette="Blues");

.. image:: https://raw.githubusercontent.com/raphaelvallat/yasa/refs/tags/v0.6.5/docs/pictures/quickstart/avg_spindles.png
    :align: center

Slow-waves
~~~~~~~~~~

The exact same steps can be applied with the :py:func:`~yasa.sw_detect` function to automatically detect slow-waves:

.. code-block:: python

    >>> sw = yasa.sw_detect(raw, hypno=hyp, include=["N2", "N3"])
    >>> sw.summary()

.. image:: https://raw.githubusercontent.com/raphaelvallat/yasa/refs/tags/v0.6.5/docs/pictures/quickstart/sw_summary.png
  :align: center

.. code-block:: python

    >>> sw.plot_average(ci=None, legend=False, palette="Blues");

.. image:: https://raw.githubusercontent.com/raphaelvallat/yasa/refs/tags/v0.6.5/docs/pictures/quickstart/avg_sw.png
  :align: center

For more details on the output of the slow-waves detection, be sure to read the `documentation <https://yasa-sleep.org/generated/yasa.sw_detect.html>`__ and try the `Jupyter notebooks <https://github.com/raphaelvallat/yasa/tree/master/notebooks>`__.

********

Automatic sleep staging
-----------------------

In this final section, we'll see how to perform automatic sleep staging in YASA. As shown below, this takes no more than a few lines of code! Here, we'll use a single EEG channel to predict a full-night hypnogram. For more details on the algorithm, check out the `eLife publication <https://elifesciences.org/articles/70092>`_ or the `documentation <https://yasa-sleep.org/generated/yasa.SleepStaging.html#yasa.SleepStaging>`__ of the function.

.. code-block:: python

    >>> sls = yasa.SleepStaging(raw, eeg_name="C3-A2")
    >>> hypno_pred = sls.predict()  # Returns a yasa.Hypnogram
    >>> yasa.plot_hypnogram(hypno_pred);  # Plot

.. figure:: https://raw.githubusercontent.com/raphaelvallat/yasa/refs/tags/v0.6.5/docs/pictures/quickstart/hypno_pred.png
    :align: center
