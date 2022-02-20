.. _quickstart:

Quickstart
##########

.. .. sectnum::
..   :depth: 2
..   :suffix: )

.. contents::
    :depth: 3

Prerequisites
-------------

This is an introduction to `YASA <https://github.com/raphaelvallat/yasa>`_, geared mainly for new users. However, you’ll need to know a bit of Python, and especially scientific libraries such as `NumPy <https://numpy.org/doc/stable/user/quickstart.html>`_ and `Pandas <https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html>`_. This tutorial also assumes that you are familiar with basic sleep research and methods for analyzing `polysomnography <https://en.wikipedia.org/wiki/Polysomnography>`_ (PSG) data.

Make sure to install the latest version of YASA, by opening a terminal or Anaconda command prompt and entering: ``pip install --upgrade yasa``

To follow this tutorial, you'll need to download the two following files on your computer. The files should be saved in the same folder as this notebook. The PSG data is saved in the `European Data Format <https://www.edfplus.info/>`_ (.edf).

* Full-night :download:`polysomnography data (.edf format, 266MB) <https://drive.google.com/uc?id=13YFQrvAGvcz77Qm8kFXu1nyRKhTti3VB>` from a 22-years-old healthy female. The PSG recording includes 19 EEGs, 2 EOGs, 1 EMG and 1 EKG. The recording spans an entire night of sleep from 11:59 PM to 08:01 AM.

* :download:`Sleep stages (aka hypnogram, .csv format, 3KB) <https://drive.google.com/uc?id=1s-XXBIXt0YKsbihMa1imBo4oftpb1Ds_>`, in 30-seconds epoch. The sleep stages are encoded as follow: 0 = Wake, 1 = N1, 2 = N2, 3 = N3, and 4 = REM.

********

Data loading and preprocessing
------------------------------

Polysomnography data
~~~~~~~~~~~~~~~~~~~~

We will use the `MNE package <https://mne.tools/stable/index.html>`_ to load and preprocess the data in Python:

.. code-block:: python

  >>> import mne
  >>> raw = mne.io.read_raw_edf('yasa_example_night_young.edf', preload=True)
  >>> raw
  <RawEDF | yasa_example_night_young.edf, 23 x 5784000 (28920.0 s), ~1015.0 MB, data loaded>

.. note::

  YASA does not allow to visualize or scroll through the PSG data. However, this can be done using the free and excellent `EDFBrowser <https://www.teuniz.net/edfbrowser/>`_ software.

**Selecting channels**

The channel names can be shown with:

.. code-block:: python

  >>> print(raw.ch_names)
  ['ROC-A1', 'LOC-A2', 'C3-A2', 'O2-A1', 'C4-A1', 'O1-A2', 'EMG1-EMG2', 'Fp1-A2', 'Fp2-A1', 'F7-A2',
   'F3-A2', 'FZ-A2', 'F4-A1', 'F8-A1', 'T3-A2', 'CZ-A2', 'T4-A1', 'T5-A2', 'P3-A2', 'PZ-A2', 'P4-A1',
   'T6-A1', 'EKG-R-EKG-L']

Let's remove the EOG, EMG and EKG channels:

.. code-block:: python

  >>> raw.drop_channels(['ROC-A1', 'LOC-A2', 'EMG1-EMG2', 'EKG-R-EKG-L'])
  >>> chan = raw.ch_names
  >>> print(chan)
  ['C3-A2', 'O2-A1', 'C4-A1', 'O1-A2', 'Fp1-A2', 'Fp2-A1', 'F7-A2', 'F3-A2', 'FZ-A2',
   'F4-A1', 'F8-A1', 'T3-A2', 'CZ-A2', 'T4-A1', 'T5-A2', 'P3-A2', 'PZ-A2', 'P4-A1', 'T6-A1']

**Downsampling and filtering**

The sampling frequency of the data in Hertz (Hz) is given by:

>>> print(raw.info['sfreq'])
200.00

The current sampling frequency of the data is therefore 200 Hz. To speed up computation, let's downsample the data to 100 Hz:

.. code-block:: python

  >>> raw.resample(100)
  >>> sf = raw.info['sfreq']
  >>> sf
  100.0

Optionally, we can apply a 0.3-45 Hz bandpass-filter:

.. code-block:: python

  # We use "verbose" and ";" to disable the text output
  >>> raw.filter(0.3, 45)

Finally, the underlying data can be accessed with:

.. code-block:: python

  >>> data = raw.get_data() * 1e6
  >>> print(data.shape)
  (19, 2892000)

In this example, ``data`` is a two-dimensional NumPy array where the rows represent the channels (19 EEG channels) and the columns represent the data samples (~3 million samples per channel).

Note that we have also multiplied the data by a million (=1e6). This is because MNE converts the data from microVolts (the standard EEG unit) to Volts when loading the EDF file. We therefore revert this operation, i.e. convert from Volts to microVolts.

Hypnogram
~~~~~~~~~

Sleep staging (aka hypnogram) for this example night was performed by a trained technician following the standard rules of the American Academy of Sleep Medicine (AASM).
The output is saved in a .csv file, where each row represents 30 seconds of data. The stages are mapped to integers such that 0 = Wake, 1 = N1 sleep, 2 = N2 sleep, 3 = N3 sleep and 4 = REM sleep.
We can load this file using the :py:func:`pandas.read_csv` function:

.. code-block:: python

  >>> import pandas as pd
  >>> hypno = pd.read_csv("yasa_example_night_young_hypno.csv", squeeze=True)
  >>> hypno
  0      0
  1      0
  2      0
  3      0
  4      0
        ..
  959    2
  960    2
  961    2
  962    2
  963    0
  Name: Stage, Length: 964, dtype: int64

.. note::

  If you do not have sleep staging, you can use YASA to automatically detect the sleep stages for you. We'll come back to this later on in this tutorial.

Using the :py:func:`yasa.plot_hypnogram` function, we can plot the hypnogram:

.. code-block:: python

  >>> import yasa
  >>> yasa.plot_hypnogram(hypno);

.. figure::  /pictures/quickstart/hypnogram.png
  :align: center

****************

Sleep statistics and stage-transition matrix
--------------------------------------------

Using the hypnogram, we can calculate standard sleep statistics using the :py:func:`yasa.sleep_statistics` function.
Importantly, this function has an ``sf_hyp`` argument, which is the sampling frequency of the hypnogram. Since we have one value every 30-seconds, the sampling frequency is 0.3333 Hz, or 1 / 30 Hz.

.. code-block:: python

  >>> yasa.sleep_statistics(hypno, sf_hyp=1/30)
  {'TIB': 482.0,
   'SPT': 468.5,
   'WASO': 9.0,
   'TST': 459.5,
   'N1': 17.5,
   'N2': 214.0,
   'N3': 85.5,
   'REM': 142.5,
   'NREM': 317.0,
   'SOL': 13.0,
   'Lat_N1': 13.0,
   'Lat_N2': 16.5,
   'Lat_N3': 31.5,
   'Lat_REM': 77.0,
   '%N1': 3.808487486398259,
   '%N2': 46.572361262241564,
   '%N3': 18.607181719260065,
   '%REM': 31.01196953210011,
   '%NREM': 68.98803046789989,
   'SE': 95.33195020746888,
   'SME': 98.07897545357524}

Furthermore, we can also calculate the sleep stages transition matrix using the :py:func:`yasa.transition_matrix` function:

.. code-block:: python

  >>> counts, probs = yasa.transition_matrix(hypno)
  >>> probs.round(3)

.. image::  /pictures/quickstart/transition_matrix.png
  :align: center
  :scale: 75%

``probs`` is the probability transition matrix, i.e. given that the current sleep stage is A, what is the probability that the next sleep stage is B.

Several metrics of sleep fragmentation can be calculated from ``probs``. For example, the *stability of sleep* can be calculated by taking the average of the diagonal values of N2, N3 and REM sleep:

.. code-block:: python

  >>> import numpy as np
  >>> np.diag(probs.loc[2:, 2:]).mean().round(3)

********

Spectral analyses
-----------------

Full-night spectrogram plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current sampling frequency of the hypnogram is one value every 30-seconds, i.e. ~0.3333 Hz. However, most YASA functions requires the sampling frequency of the hypnogram to be the same as the sampling frequency of the PSG data. In this example, we therefore need to upsample our hypnogram from 0.333 Hz to 100 Hz.
This can be done with the :py:func:`yasa.hypno_upsample_to_data` function:

.. code-block:: python

  >>> hypno_up = yasa.hypno_upsample_to_data(hypno, sf_hypno=1/30, data=raw)
  >>> print(len(hypno_up))

Now that the hypnogram and data have the same shape, we can plot our hypnogram on top of a multitaper `spectrogram <https://en.wikipedia.org/wiki/Spectrogram>`_ using the :py:func:`yasa.plot_spectrogram` function, which shows the time-frequency representation of a single EEG channel across the entire night. The x-axis of the spectrogram is time in hours, and the y-axis is the frequency range (from 0 to 25 Hz).
Warmer colors indicate higher spectral power in this specific frequency band at this specific time for this channel. This kind of plot is very useful to quickly identify periods of NREM sleep (high power in frequencies below 5 Hz and spindle-related activity around ~14 Hz) and REM sleep (almost no power in frequencies below 5 Hz).

.. code-block:: python

  # We select only the C4-A1 EEG channel.
  >>> yasa.plot_spectrogram(data[chan.index("C4-A1")], sf, hypno_up);

.. image::  /pictures/quickstart/spectrogram.png
  :align: center

.. note::

  Whenever you start a new analysis in YASA, we always recommend that you the :py:func:`yasa.plot_spectrogram` function to check your data. This can help you easily identify artefact in the data or misalignement between the PSG data and hypnogram.

EEG power in specific frequency bands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a primer on EEG spectral bandpower please refer to https://raphaelvallat.com/bandpower.html.

Spectral analysis quantifies the power (or amplitude) of the EEG signal in different frequency bands. In neuroscience, the most common frequency bands are **delta** (0.5–4 Hz), **theta** (4–8 Hz), **alpha** (8–12 Hz), **beta** (12–30 Hz), and **gamma** (30–~100 Hz). There are numerous studies that have reported significant relationship between the EEG power spectrum and human behavior, cognitive state, or mental illnesses, and EEG spectral analysis is now one of the principal analysis methods in the field of neuroscience and sleep research.
It is especially relevant for sleep analysis, as it is well-known that the different stages of sleep `vary drastically in their spectral content <https://raphaelvallat.com/bandpower.html>`_. For example, deep slow-wave sleep (N3) is associated with increased power in the low frequencies, especially the delta band (0.5-4Hz), and decreased power in the beta and gamma bands.

Calculating the average spectral power in different frequency bands is straightforward with the :py:func:`yasa.bandpower` function:

>>> yasa.bandpower(raw)

.. image::  /pictures/quickstart/bandpower.png
  :align: center

This calculates, for each channel separately, the average power in the main frequency bands across the entire recording. Importantly, the values are **relative** power, i.e. they are expressed as a proportion of the total power between the lowest frequency (default 0.5 Hz) and the highest frequency (default 40 Hz). We can disable this behavior and get the **absolute** spectral power values in :math:`μV^2 / Hz` by using the ``relative=False`` argument. Similarly, we can define custom frequency bands with the ``bands`` parameter. In the example below, we calculate the absolute power in the 1-9 Hz frequency range (named "Slow") and the 9-30 Hz range (named "Fast"):

>>> yasa.bandpower(raw, relative=False, bands=[(1, 9, "Slow"), (9, 30, "Fast")])

.. image::  /pictures/quickstart/bandpower2.png
  :align: center

We can also pass an hypnogram to calculate the spectral powers separately for each sleep stage. In the example below, we use the upsampled hypnogram to calculate the spectral power separately for N2, N3 and REM. We save the results in a new variable named ``bandpower``.

>>> bandpower = yasa.bandpower(raw, hypno=hypno_up, include=(2, 3, 4))

If desired, we can then export the ``bandpower`` dataframe to a CSV file using :py:meth:`pandas.DataFrame.to_csv`:

>>> bandpower.to_csv("bandpower.csv")

Finally, we can use the :py:func:`yasa.topoplot` function to visualize the spectral powers across all electrodes. In the example below, we only plot the spectral values of stage N3, using the :py:meth:`pandas.DataFrame.xs` function. As expected, the relative delta power is higher in frontal channels.

>>> fig = yasa.topoplot(bandpower.xs(3)['Delta'])

.. image::  /pictures/quickstart/topoplot.png
  :align: center
  :scale: 60%

********

Events detection
----------------

Spindles
~~~~~~~~

Automatic spindles detection can be performed with the :py:func:`yasa.spindles_detect` function. The detection is based on the algorithm described in `Lacourse et al 2018 <https://pubmed.ncbi.nlm.nih.gov/30107208/>`_, and a step-by-step explanation is provided in `this notebook <https://github.com/raphaelvallat/yasa/blob/master/notebooks/01_spindles_detection.ipynb>`_. For the sake of this tutorial, we'll use the default detection thresholds, but these can (and should) be adjusted based on your own data. In the example below, we'll specify the hypnogram and limit the detection to stage N2 and 3 (``include=(2, 3)``).

>>> sp = yasa.spindles_detect(raw, hypno=hypno_up, include=(2, 3))

Here, the ``sp`` variable is a `class <https://raphaelvallat.com/yasa/build/html/generated/yasa.SpindlesResults.html#yasa.SpindlesResults>`_, which is simply a bundle of functions (called methods) and data (attributes). For example, we can see a dataframe with all the detected events with:

>>> sp.summary()

.. image::  /pictures/quickstart/spindles_summary.png
  :align: center

The `documentation of the function <https://raphaelvallat.com/yasa/build/html/generated/yasa.spindles_detect.html>`_ explains what each of these columns represent and how they're calculated. Furthermore, by specifying the ``grp_chan`` and ``grp_stage`` parameters, we tell YASA to first average across channels and slep stages, respectively:

>>> sp.summary(grp_chan=True, grp_stage=True)

Finally, we can plot the average spindle, calculated for each channel separately and time-synced to the most prominent peak of the spindles:

.. code-block:: python

  >>> # Because of the large number of channels, we disable the 95%CI and legend
  >>> sp.plot_average(ci=None, legend=False, palette="Blues");

.. image::  /pictures/quickstart/avg_spindles.png
  :align: center
  :scale: 50%

Slow-waves
~~~~~~~~~~

The exact same steps can be applied with the :py:func:`yasa.sw_detect` function to automatically detect slow-waves:

.. code-block:: python

  >>> sw = yasa.sw_detect(raw, hypno=hypno_up, include=(2, 3))
  >>> sw.summary()

.. image::  /pictures/quickstart/sw_summary.png
  :align: center

>>> sw.plot_average(ci=None, legend=False, palette="Blues");

.. image::  /pictures/quickstart/avg_sw.png
  :align: center
  :scale: 50%

For more details on the output of the slow-waves detection, be sure to read the `documentation <https://raphaelvallat.com/yasa/build/html/generated/yasa.sw_detect.html>`_ and try the `Jupyter notebooks <https://github.com/raphaelvallat/yasa/tree/master/notebooks>`_.

********

Automatic sleep staging
-----------------------

In this final section, we'll see how to perform automatic sleep staging in YASA. As shown below, this takes no more than a few lines of code! Here, we'll use a single EEG channel to predict a full-night hypnogram. For more details on the algorithm, check out the `eLife publication <https://elifesciences.org/articles/70092>`_ or the `documentation <https://raphaelvallat.com/yasa/build/html/generated/yasa.SleepStaging.html#yasa.SleepStaging>`_ of the function.

.. code-block:: python

  >>> sls = yasa.SleepStaging(raw, eeg_name='C3-A2')
  >>> hypno_pred = sls.predict()  # Predict the sleep stages
  >>> hypno_pred = yasa.hypno_str_to_int(hypno_pred)  # Convert "W" to 0, "N1" to 1, etc
  >>> yasa.plot_hypnogram(hypno_pred);  # Plot

.. figure::  /pictures/quickstart/hypnogram.png
  :align: center

Let's calculate the agreement against the ground-truth expert scoring:

.. code-block:: python

  >>> from sklearn.metrics import accuracy_score
  >>> print(f"The accuracy is {100 * accuracy_score(hypno, hypno_pred):.3f}%")
  The accuracy is 82.676%

|