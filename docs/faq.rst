.. _faq:

FAQ
===

Loading and visualizing polysomnography data
--------------------------------------------

.. ----------------------------- LOAD EDF -----------------------------
.. dropdown:: How can I load an EDF file in Python?
    :animate: fade-in-slide-down
    :icon: question
    :name: load_edf

    If you have polysomnography data in European Data Format (.edf), you can use the `MNE package <https://mne.tools/stable/index.html>`_ to load and preprocess your data in Python. MNE also supports several other standard formats (e.g. BrainVision, BDF, EEGLab). A simple preprocessing pipeline using MNE is shown below.

    .. code-block:: python

        import mne
        # Load the EDF file
        raw = mne.io.read_raw_edf("MYEDFFILE.edf", preload=True)
        # Downsample the data to 100 Hz
        raw.resample(100)
        # Apply a bandpass filter from 0.1 to 40 Hz
        raw.filter(0.1, 40)
        # Select a subset of EEG channels
        raw.pick(["C4-A1", "C3-A2"])

.. ----------------------------- LOAD EDF -----------------------------
.. dropdown:: Should my data be in Volts or micro-Volts?
    :animate: fade-in-slide-down
    :icon: question
    :name: what_voltage

    Where EEG data is concerned, YASA's algorithms are designed to work with data in units of micro-Volts.
    When ``data`` is passed to a YASA function as a :py:class:`~numpy.ndarray`, the unit must be micro-Volts (uV).
    For most functions, YASA allows the ``data`` parameter to be either a :py:class:`numpy.ndarray` or a :py:class:`mne.io.BaseRaw`.
    Instances of :py:class:`~mne.io.BaseRaw` have unit information included in them, so in these cases, YASA will handle any
    necessary conversions by extracting the data internally in the required units (using :py:meth:`~mne.io.BaseRaw.get_data`).

.. ----------------------------- VISUALIZE -----------------------------
.. dropdown:: Can I visualize my polysomnography data in YASA?
    :animate: fade-in-slide-down
    :icon: question
    :name: visualize

    YASA is a command-line software and does not support data visualization. To scroll through your data, we recommend the free software EDFBrowser (https://www.teuniz.net/edfbrowser/):

    .. figure:: https://raw.githubusercontent.com/raphaelvallat/yasa/refs/tags/v0.6.5/docs/pictures/edfbrowser_with_hypnogram.png
        :align: center

.. ----------------------------- HYPNOGRAM -----------------------------
.. .. dropdown:: How can I read an hypnogram file in YASA?
..     :animate: fade-in-slide-down
..     :icon: question


.. ############################################################################
.. ############################################################################
..                                  HYPNOGRAM
.. ############################################################################
.. ############################################################################

Hypnogram
---------

.. ----------------------------- ALIGNMENT -----------------------------
.. dropdown:: How do I align a hypnogram with my EEG recording?
    :animate: fade-in-slide-down
    :icon: question
    :name: hypno_alignment

    The :py:meth:`~yasa.Hypnogram.upsample_to_data` method upsamples the hypnogram to the
    sampling frequency of the EEG data and ensures both have the exact same number of samples.
    Its behaviour depends on the relative lengths of the two and on whether timestamp
    information is available.

    **Two alignment modes**

    * **Length-based** (default): YASA assumes the hypnogram and the recording start at the
      same time. Any length mismatch is resolved by padding or cropping at the *end*.
      This mode is always used when ``data`` is a NumPy array. It is also used when
      ``data`` is an :py:class:`mne.io.BaseRaw` but either ``Hypnogram.start`` is not set or
      ``raw.meas_date`` is ``None``.

    * **Timestamp-aware**: triggered automatically when **both** ``Hypnogram.start`` is set and
      ``raw.meas_date`` is set on the :py:class:`~mne.io.BaseRaw`. YASA computes the absolute
      offset between the two timestamps and uses it to select the correct hypnogram epochs.

    **EDF files and timezones**

    The EDF+ standard explicitly defines ``starttime`` as local time at the patient's
    location. :py:func:`mne.io.read_raw_edf` reads this value and tags it as UTC (since it
    has no other choice). As a result, ``raw.info["meas_date"]`` is labelled UTC but
    actually contains local time.

    :py:meth:`~yasa.Hypnogram.upsample_to_data` handles this transparently:
    ``meas_date_is_local=True`` is the default, so both ``meas_date`` and
    ``Hypnogram.start`` are compared as local absolute timestamp values.

    .. code-block:: python

        # EDF recorded at 22:11:37 local time
        hyp = yasa.Hypnogram(stages, freq="30s", start="2024-11-08 22:11:37")
        hypno = hyp.upsample_to_data(raw)

    If your EDF files genuinely store UTC in ``meas_date``, pass
    ``meas_date_is_local=False`` and ``Hypnogram.tz`` so YASA can compare both timestamps in the
    same reference frame:

    .. code-block:: python

        hyp = yasa.Hypnogram(stages, freq="30s", start="2024-11-08 22:11:37",
                             tz="Europe/Paris")
        hypno = hyp.upsample_to_data(raw, meas_date_is_local=False)

    The three cases below describe what happens in each alignment mode.

    **Case 1 — Hypnogram and data cover the same window**

    The hypnogram covers exactly the same time span as the EEG (same start, same end,
    but different sampling frequency).

    In both modes the result is identical: the hypnogram is upsampled and fits the data
    exactly.

    .. code-block:: python

        hyp = yasa.Hypnogram(stages, freq="30s")
        hypno = hyp.upsample_to_data(raw)

    **Case 2 — Hypnogram is shorter than the data**

    This can occur for example if the Hypnogram is cropped to the Lights Off to Lights On period,
    while the PSG spans the entire recording period.

    *Length-based*: YASA assumes both start at the same time, so the hypnogram is padded
    with **Unscored** (``UNS``) at the end. A :py:exc:`UserWarning` is emitted.

    *Timestamp-aware*: YASA computes the offset. Because the recording typically starts
    *before* the hypnogram, the appropriate number of **Unscored** (``UNS``) epochs is
    prepended. Any remaining mismatch at the end is also padded with **Unscored** (``UNS``).

    .. code-block:: python

        hyp = yasa.Hypnogram(stages, freq="30s", start="2024-01-15 23:00:00")
        hypno = hyp.upsample_to_data(raw)
        # Epochs before Lights Off → UNS; epochs after Lights On → UNS

    **Case 3 — Hypnogram is longer than the data**

    This happens when working with a segment of a longer recording (e.g. a file cropped with
    :py:meth:`mne.io.Raw.crop`). The full-night hypnogram has more epochs than the data.

    *Length-based*: YASA assumes both start at the same time and **crops** the hypnogram
    from the end. A :py:exc:`UserWarning` is emitted. This is only correct if the segment
    starts at the very beginning of the hypnogram.

    *Timestamp-aware*: YASA skips the correct number of leading epochs based on the
    timestamp differences and selects only the epochs that fall within the recording window.
    No warning is emitted when the match is exact.

    .. code-block:: python

        # Full-night hypnogram, recording is only the second half of the night
        hyp = yasa.Hypnogram(stages, freq="30s", start="2024-01-15 23:00:00")
        hypno = hyp.upsample_to_data(raw_cropped)  # correct epochs selected automatically

    **Automatic staging with** :py:class:`~yasa.SleepStaging`

    When using :py:class:`~yasa.SleepStaging`, the ``start`` attribute is populated
    automatically from ``raw.meas_date`` (when available), so timestamp-aware alignment works
    out of the box with no extra configuration:

    .. code-block:: python

        sls = yasa.SleepStaging(raw, eeg_name="C4-M1")
        hyp = sls.predict()  # hyp.start set automatically from raw.meas_date
        hypno = hyp.upsample_to_data(raw_cropped)

.. ############################################################################
.. ############################################################################
..                                  DETECTION
.. ############################################################################
.. ############################################################################

Event detection
---------------

.. ----------------------------- ALGO -----------------------------
.. dropdown:: How do the spindle detection and slow-waves detection algorithms work?
    :animate: fade-in-slide-down
    :icon: question
    :name: detection_algo

    The **spindles** detection is a custom adaptation of the `Lacourse et al 2018 <https://doi.org/10.1016/j.jneumeth.2018.08.014>`_ method. A step-by-step description of the algorithm can be found in `this notebook <https://github.com/raphaelvallat/yasa/blob/master/notebooks/01_spindles_detection.ipynb>`_.

    The **slow-waves detection** combines the methods proposed in `Massimini et al 2004 <https://www.jneurosci.org/content/24/31/6862>`_ and `Carrier et al 2011 <https://doi.org/10.1111/j.1460-9568.2010.07543.x>`_. A step-by-step description of the algorithm can be found `here <https://github.com/raphaelvallat/yasa/blob/master/notebooks/05_sw_detection.ipynb>`_.

    .. important::
        Both algorithms have parameters that can (and should) be fine-tuned to your data, as explained in the next question.

.. ----------------------------- PARAMETERS -----------------------------
.. dropdown:: How do I find the optimal parameters for my data?
    :animate: fade-in-slide-down
    :icon: question
    :name: best_params

    There are several parameters that can be adjusted in the spindles / slow-waves / artefact detection. While the default parameters should work reasonably well on most data, they might not be adequate for your data, especially if you're working with specific populations (e.g. older adults, kids, patients with certain disorders, etc).

    For the sake of example, let's say that you have 100 recordings and you want to apply YASA to automatically detect the spindles. However, you'd like to fine-tune the parameters to your data. **We recommend the following approach:**

    1. Grab a few representative recordings (e.g. 5 or 10 out of 100) and manually annotate the sleep spindles. You can use `EDFBrowser <https://www.teuniz.net/edfbrowser/>`_ to manually score the sleep spindles. Ideally, the manual scoring should be high-quality, so you may also ask a few other trained individuals to score the same data until you reach a consensus.
    2. Apply YASA on the same recordings, first with the default parameters and then by slightly varying each parameter. For example, you may want to use a different detection threshold each time you run the algorithm, or a different frequency band for the filtering. In other words, you loop across several possible combinations of parameters. Save the resulting detection dataframe.
    3. Finally, find the combination of parameters that give you the results that are the most similar to your own scoring. For example, you can use the combination of parameters that maximize the `F1-score <https://en.wikipedia.org/wiki/F-score>`_ of the detected spindles against your own visual detection.
    4. Use the "winning" combination to score the remaining recordings in your database.

.. ----------------------------- MANUAL EDITING -----------------------------
.. dropdown:: Can I manually add or remove detected events?
    :animate: fade-in-slide-down
    :icon: question
    :name: edit_detection

    YASA does not currently support visual editing of the detected events. However, you can import the events as annotations in `EDFBrowser <https://www.teuniz.net/edfbrowser/>`_ and edit the events from there. If you simply want to visualize the detected events (no editing), you can also use the `plot_detection <https://yasa-sleep.org/generated/yasa.SpindlesResults.html#yasa.SpindlesResults.plot_detection>`_ method.


.. ############################################################################
.. ############################################################################
..                                  SLEEP STAGING
.. ############################################################################
.. ############################################################################

Sleep staging
-------------

.. ----------------------------- ACCURACY -----------------------------
.. dropdown:: How accurate is YASA for automatic sleep staging?
    :animate: fade-in-slide-down
    :icon: question
    :name: accuracy_yasa

    YASA was trained and evaluated on a large and heterogeneous database of thousands of polysomnography recordings, including healthy individuals and patients with sleep disorders. Overall, the results show that **YASA matches human inter-rater agreement, with an accuracy of ~85% against expert consensus scoring**. The full validation of YASA was published in `eLife <https://elifesciences.org/articles/70092>`_:

    * Vallat, Raphael, and Matthew P. Walker. "An open-source, high-performance tool for automated sleep staging." Elife 10 (2021). doi: https://doi.org/10.7554/eLife.70092

    However, our recommendation is that **YASA should not replace human scoring, but rather serve as a starting point to speed up sleep staging**. If possible, you should always have a trained sleep scorer visually check the predictions of YASA, with a particular emphasis on low-confidence epochs and/or N1 sleep epochs, as these are the epochs most often misclassified by the algorithm.
    Finally, users can also leverage the :py:func:`yasa.plot_spectrogram` function to plot the predicted hypnogram on top of the full-night spectrogram. Such plots are very useful to quickly identify blatant errors in the hypnogram.

    .. figure:: https://raw.githubusercontent.com/raphaelvallat/yasa/refs/tags/v0.6.5/docs/pictures/spectrogram.png
        :align: center

.. ----------------------------- EDITING -----------------------------
.. dropdown:: How do I edit the predicted hypnogram?
    :animate: fade-in-slide-down
    :icon: question
    :name: yasa_editing

    YASA does not come with a graphical user interface (GUI) and therefore editing the predicted hypnogram is not currently possible. The simplest way is therefore to export the hypnogram in CSV format and then open the file — together with the corresponding polysomnography data — in an external GUI, as shown below.

    **EDFBrowser**

    `EDFBrowser <https://www.teuniz.net/edfbrowser/>`_ is a free software for visualizing polysomnography data in European Data Format (.edf), which also provides a module for visualizing and editing hypnograms.

    The code below show hows to export the hypnogram in an EDFBrowser-compatible format. It assumes that you have already run the algorithm and stored the predicted hypnogram in an array named ``hypno``.

    .. code-block:: python

        # Export to a CSV file compatible with EDFBrowser
        import numpy as np
        import pandas as pd
        hypno_export = pd.DataFrame({
            "onset": np.arange(len(hypno)) * 30,
            "label": hypno,
            "duration": 30})
        hypno_export.to_csv("my_hypno_EDFBrowser.csv", index=False)

    You can then import the hypnogram in EDFBrowser by clicking on the "Import annotations/events" in the "Tools" menu. Then, select the "ASCII/CSV" tab and change the parameters as follow:

    .. figure:: https://raw.githubusercontent.com/raphaelvallat/yasa/refs/tags/v0.6.5/docs/pictures/edfbrowser_import_annotations.png
        :align: center

    Click "Import". Once it's done, the hypnogram can be enabled via the "Window" menu. A dialog will appear where you can setup the labels for the different sleep stages and the mapping to the annotations in the file. The default parameters should work.
    When using the Annotation editor, the hypnogram will be updated realtime when adding, moving or deleting annotations. Once you're done editing, you can export the edited hypnogram with "Export anotations/events" in the "Tools" menu.

    .. figure:: https://raw.githubusercontent.com/raphaelvallat/yasa/refs/tags/v0.6.5/docs/pictures/edfbrowser_with_hypnogram.png
        :align: center

.. ----------------------------- ANIMAL DATA -----------------------------
.. dropdown:: Can I use YASA to score animal data and/or human intracranial data?
    :animate: fade-in-slide-down
    :icon: question
    :name: animal_data

    YASA was only designed for human scalp data and as such will not work with animal data or intracranial data. Adding support for such data would require the two following steps:

    1. Modifying (some of) the features. For example, rodent sleep does not have the same temporal dynamics as human sleep, and therefore one could modify the length of the smoothing window to better capture these dynamics.
    2. Re-training the classifier using a large database of previously-scored data.

    Despite these required changes, one advantage of YASA is that it provides a useful framework for implementing such sleep staging algorithms. For example, one can save a huge amount of time by simply re-using and adapting the built-in :py:class:`yasa.SleepStaging` class.
    In addition, all the code used to train YASA is freely available at https://github.com/raphaelvallat/yasa_classifier and can be re-used to re-train the classifier on non-human data.


.. ############################################################################
.. ############################################################################
..                                  OTHERS
.. ############################################################################
.. ############################################################################

Others
------

.. ----------------------------- NEW RELEASES -----------------------------
.. dropdown:: How can I be notified of new releases?
    :animate: fade-in-slide-down
    :icon: question
    :name: collapse_release

    You can click "Watch" on the `YASA GitHub repository <https://github.com/raphaelvallat/yasa>`_ and
    select "Releases only" to receive email notifications whenever a new version is published.

    To upgrade to the latest version, run the following in a terminal:

    .. tab-set::

        .. tab-item:: pip

            .. code-block:: shell

                pip install --upgrade yasa

        .. tab-item:: uv

            .. code-block:: shell

                uv pip install --upgrade yasa


.. ----------------------------- DEVELOPMENT -----------------------------
.. dropdown:: Is there a graphical user interface (GUI) for YASA?
    :animate: fade-in-slide-down
    :icon: question
    :name: collapse_development

    `YASA Flaskified <https://github.com/bartromb/YASAFlaskified>`_ is a web-based application for
    analyzing EEG data using YASA. It requires deployment on a physical or virtual server, which can
    be done using the scripts provided in its repository.

.. ----------------------------- CITING YASA -----------------------------
.. dropdown:: How can I cite YASA?
    :animate: fade-in-slide-down
    :icon: question
    :name: collapse_cite

    To cite YASA, please use the `eLife publication <https://elifesciences.org/articles/70092>`_:

    * Vallat, Raphael, and Matthew P. Walker. "An open-source, high-performance tool for automated sleep staging." Elife 10 (2021). doi: https://doi.org/10.7554/eLife.70092

    BibTeX:

    .. code-block:: latex

        @article {vallat2021open,
          title={An open-source, high-performance tool for automated sleep staging},
          author={Vallat, Raphael and Walker, Matthew P},
          journal={Elife},
          volume={10},
          year={2021},
          doi = {https://doi.org/10.7554/eLife.70092},
          URL = {https://elifesciences.org/articles/70092},
          publisher={eLife Sciences Publications, Ltd}
        }

.. ----------------------------- END -----------------------------
