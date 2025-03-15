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

    The **spindles** detection is a custom adaptation of the `Lacourse et al 2018 <https://doi.org/10.1016/j.jneumeth.2018.08.014>`_ method. A step-by-step description of the algorithm can be found in `this notebook <https://github.com/raphaelvallat/yasa/blob/develop/notebooks/01_spindles_detection.ipynb>`_.

    The **slow-waves detection** combines the methods proposed in `Massimini et al 2004 <https://www.jneurosci.org/content/24/31/6862>`_ and `Carrier et al 2011 <https://doi.org/10.1111/j.1460-9568.2010.07543.x>`_. A step-by-step description of the algorithm can be found `here <https://github.com/raphaelvallat/yasa/blob/develop/notebooks/05_sw_detection.ipynb>`_.

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

    **SpiSOP**

    `SpiSOP <https://www.spisop.org/>`_ is an open-source Matlab toolbox for the analysis and visualization of polysomnography sleep data. It comes with a sleep scoring GUI.
    As explained in `the documentation <https://www.spisop.org/faq/#What_is_needed_to_run_SpiSOP_and_in_what_format>`_, the hypnogram should be a tab-separated text file with two columns (no headers). The first column has the sleep stages (0: Wake, 1: N1, 2: N2, 3: N3, 5: REM) and the second column indicates whether the current epoch should be marked as artefact (1) or valid (0).

    .. code-block:: python

        hypno_int = pd.Series(hypno).map({"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 5}).to_numpy()
        hypno_export = pd.DataFrame({"label": hypno_int, "artefact": 0})
        hypno_export.to_csv("my_hypno_SpiSOP.txt", sep="\t", header=False, index=False)

    **Visbrain**

    `Visbrain <https://visbrain.org/sleep.html>`_ is an open-source Python toolbox that includes a module for visualizing polysomnography sleep data and scoring sleep (see screenshot below).

    .. figure:: https://raw.githubusercontent.com/raphaelvallat/yasa/refs/tags/v0.6.5/docs/pictures/visbrain.PNG
        :align: center

    Visbrain accepts several `formats for the hypnogram <https://visbrain.org/sleep.html#hypnogram>`_. The code below show how to export the hypnogram in the `Elan software format <https://pubmed.ncbi.nlm.nih.gov/21687568/>`_ (i.e. a text file with the *.hyp* extension):

    .. code-block:: python

        hypno_int = pd.Series(hypno).map({"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 5}).to_numpy()
        header = "time_base 30\nsampling_period 1/30\nepoch_nb %i\nepoch_list" % len(hypno_int)
        np.savetxt("my_hypno_Visbrain.txt", hypno_int, fmt="%s", delimiter=",", newline="\n",
                   header=header, comments="", encoding="utf-8")

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

    You can click "Watch" on the `YASA GitHub repository <https://github.com/raphaelvallat/yasa>`_.
    Whenever a new release is out there, you can upgrade your version by typing the following line in a terminal window:

    .. code-block:: shell

        pip install --upgrade yasa
        pip install --upgrade yasa


.. ----------------------------- DEVELOPMENT -----------------------------
.. raw:: html

    <div class="panel panel-default">
      <div class="panel-heading">
        <h5 class="panel-title">
          <a data-toggle="collapse" href="#development">Development</a>
        </h5>
      </div>
      <div id="development" class="panel-collapse collapse">
        <div class="panel-body">

**YASA Flaskified**

YASA Flaskified is a web-based application for analyzing EEG data using YASA. It requires deployment on a physical or virtual server, which can be done using the scripts provided in its repository. For more details, visit the **[YASA Flaskified GitHub repository](https://github.com/bartromb/YASAFlaskified)**.

        </div>
      </div>
    </div>

.. ----------------------------- DONATION -----------------------------
.. dropdown:: I am not a programmer, how can I contribute to YASA?
    :animate: fade-in-slide-down
    :icon: question
    :name: collapse_donate

    There are many ways to contribute to YASA, even if you are not a programmer, for example, reporting bugs or results that are inconsistent with other softwares, improving the documentation and examples, or, even `buying the developers a coffee <https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=K2FZVJGCKYPAG&currency_code=USD&source=url>`_!

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
