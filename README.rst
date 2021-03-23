.. -*- mode: rst -*-

|

.. image:: https://badge.fury.io/py/yasa.svg
    :target: https://badge.fury.io/py/yasa

.. image:: https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue
    :target: https://www.python.org/downloads/

.. image:: https://img.shields.io/github/license/raphaelvallat/yasa.svg
    :target: https://github.com/raphaelvallat/yasa/blob/master/LICENSE

.. image:: https://travis-ci.org/raphaelvallat/yasa.svg?branch=master
    :target: https://travis-ci.org/raphaelvallat/yasa

.. image:: https://codecov.io/gh/raphaelvallat/yasa/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/raphaelvallat/yasa

.. image:: https://pepy.tech/badge/yasa
    :target: https://pepy.tech/badge/yasa

.. .. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2370600.svg
..    :target: https://doi.org/10.5281/zenodo.2370600

----------------

.. figure::  /docs/pictures/yasa_logo.png
   :align:   center

**YASA** (*Yet Another Spindle Algorithm*) is a sleep analysis toolbox in Python. YASA includes several fast and convenient command-line functions to:

* Perform automatic sleep staging.
* Detect sleep spindles, slow-waves, and rapid eye movements on single and multi-channel EEG data.
* Reject major artifacts on single or multi-channel EEG data.
* Perform advanced spectral analyses: spectral bandpower, phase-amplitude coupling, event-locked analyses, 1/f, and more!
* Manipulate hypnogram and calculate sleep statistics.

For more details, check out the `API documentation <https://raphaelvallat.com/yasa/build/html/index.html>`_ or try the
`tutorial (Jupyter notebooks) <https://github.com/raphaelvallat/yasa/tree/master/notebooks>`_.

----------------

Installation
~~~~~~~~~~~~

To install YASA, simply open a terminal or Anaconda command prompt and enter:

.. code-block:: shell

  pip install --upgrade yasa

**What are the prerequisites for using YASA?**

To use YASA, all you need is:

- Some basic knowledge of Python, especially the `NumPy <https://docs.scipy.org/doc/numpy/user/quickstart.html>`_, `Pandas <https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html>`_ and `MNE <https://martinos.org/mne/stable/index.html>`_ packages.
- A Python editor: YASA works best with `Jupyter Lab <https://jupyterlab.readthedocs.io/en/stable/index.html>`_, a web-based interactive user interface.
- Some sleep EEG data and optionally a sleep staging file (hypnogram) to perform calculations on specific sleep stages. To facilitate masking and indexing operations, the data and hypnogram must have the same sampling frequency and number of samples. YASA provide some convenient functions to load and upsample hypnogram data to the desired shape.

**I have sleep EEG data in European Data Format (.edf), how do I load the data in Python?**

If you have sleep EEG data in standard formats (e.g. EDF or BrainVision), you can use the `MNE package <https://mne.tools/stable/index.html>`_ to load and preprocess your data in Python. A simple preprocessing pipeline using MNE is shown below:

.. code-block:: python

  import mne
  # Load the EDF file, excluding the EOGs and EKG channels
  raw = mne.io.read_raw_edf('MYEDFFILE.edf', preload=True, exclude=['EOG1', 'EOG2', 'EKG'])
  raw.resample(100)                      # Downsample the data to 100 Hz
  raw.filter(0.1, 40)                    # Apply a bandpass filter from 0.1 to 40 Hz
  raw.pick_channels(['C4-A1', 'C3-A2'])  # Select a subset of EEG channels

How do I get started with YASA?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to dive right in, you can simply go to the `documentation <https://raphaelvallat.com/yasa/build/html/api.html>`_ and try to apply YASA's functions on your own EEG data. However, for most users, we strongly recommend that you first try running the examples Jupyter notebooks to get a sense of how YASA works and what it can do! The advantage is that the notebooks also come with example datasets so they should work right out of the box as long as you've installed YASA first. The notebooks and datasets can be found on `GitHub <https://github.com/raphaelvallat/yasa/tree/master/notebooks>`_ (make sure that you download the whole *notebooks/* folder). A short description of all notebooks is provided below:

**Spindles detection**

* `01_spindles_detection <notebooks/01_spindles_detection.ipynb>`_: single-channel spindles detection and step-by-step description of the spindles detection algorithm.
* `02_spindles_detection_multi <notebooks/02_spindles_detection_multi.ipynb>`_: multi-channel spindles detection.
* `03_spindles_detection_NREM_only <notebooks/03_spindles_detection_NREM_only.ipynb>`_: how to limit the spindles detection on specific sleep stages using an hypnogram.
* `04_spindles_slow_fast <notebooks/04_spindles_slow_fast.ipynb>`_: slow versus fast spindles.
* `run_visbrain.py <notebooks/run_visbrain.py>`_: interactive display of the detected spindles using the Visbrain visualization software in Python.

**Slow-waves detection**

* `05_sw_detection <notebooks/05_sw_detection.ipynb>`_: single-channel slow-waves detection and step-by-step description of the slow-waves detection algorithm.
* `06_sw_detection_multi <notebooks/06_sw_detection_multi.ipynb>`_: multi-channel slow-waves detection.

**Rapid Eye Movements (REMs) detection**

* `07_REMs_detection <notebooks/07_REMs_detection.ipynb>`_: REMs detection.

**Spectral analysis**

* `08_bandpower <notebooks/08_bandpower.ipynb>`_: calculate spectral band power, optionally averaged across channels and sleep stages.
* `09_IRASA <notebooks/09_IRASA.ipynb>`_: separate the aperiodic (= fractal = 1/f) components of the EEG power spectrum using the IRASA method.
* `10_spectrogram <notebooks/10_spectrogram.ipynb>`_: plot a multi-taper full-night spectrogram on single-channel EEG data with the hypnogram on top.
* `11_nonlinear_features <notebooks/11_nonlinear_features.ipynb>`_: calculate non-linear EEG features on 30-seconds epochs and perform sleep stage classification.
* `12_spindles-SO_coupling <notebooks/12_spindles-SO_coupling.ipynb>`_: slow-oscillations/spindles phase-amplitude coupling and data-driven comodulogram.
* `15_topoplot <notebooks/15_topoplot.ipynb>`_: topoplot.

**Artifact rejection**

* `13_artifact_rejection <notebooks/13_artifact_rejection.ipynb>`_: automatic artifact rejection on single and multi-channel EEG data.

**Automatic sleep staging**

* `14_automatic_sleep_staging <notebooks/14_automatic_sleep_staging.ipynb>`_: automatic sleep staging of polysomnography data.


.. Typical use: spindles detection
.. -------------------------------

.. .. code-block:: python

..   import yasa

..   # 1) Single-channel spindles detection, in its simplest form.
..   # There are many optional arguments that you can change to customize the detection.
..   sp = yasa.spindles_detect(data, sf)
..   # The output of the the detection (`sp`) is a class that has several attributes and methods.
..   # For instance, to get the full detection dataframe, one can simply use:
..   sp.summary()
..   # To plot an average template of all the detected spindles,
..   # centered around the most prominent peak (+/- 1 second)
..   sp.plot_average(center='Peak', time_before=1, time_after=1)
..   # To interactively inspect the detected spindles
..   sp.plot_detection()

..   # 2) Multi-channels spindles detection limited to N2/N3 sleep, with automatic outlier rejection
..   sp = yasa.spindles_detect(data, sf, ch_names, hypno=hypno, include=(2, 3), remove_outliers=True)
..   # Return spindles count / density and parameters averaged across channels and sleep stages
..   sp.summary(grp_stage=True, grp_chan=True)

.. The output of ``sp.summary()`` is a `pandas DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_ where each row is a  detected spindle and each column a parameter of this event, including the start and end timestamps (in seconds from the beginning of the data), duration, amplitude, etc.

.. .. table::
..   :widths: auto

..   =======  =====  ==========  ===========  =====  ==========  ==========  ===========  ==============  ==========
..     Start    End    Duration    Amplitude    RMS    AbsPower    RelPower    Frequency    Oscillations    Symmetry
..   =======  =====  ==========  ===========  =====  ==========  ==========  ===========  ==============  ==========
..      3.32   4.06        0.74        81.80  19.65        2.72        0.49        12.85              10        0.67
..     13.26  13.85        0.59        99.30  24.49        2.82        0.24        12.15               7        0.25
..   =======  =====  ==========  ===========  =====  ==========  ==========  ===========  ==============  ==========

Gallery
~~~~~~~

Below some plots demonstrating the functionalities of YASA. To reproduce these, check out the `tutorial (Jupyter notebooks) <https://github.com/raphaelvallat/yasa/tree/master/notebooks>`_.

.. figure::  /docs/pictures/gallery.png
  :align:   center

  *The top plot show an overlay of the detected spindles on real EEG data. The middle left panel shows a time-frequency representation of the whole-night recording (spectrogram), plotted with the hypnogram (sleep stages) on top. The middle right panel shows the sleep stage probability transition matrix, calculated across the entire night. The bottom row shows, from left to right: a topographic plot, the average template of all detected slow-waves across the entire night stratified by channels, and a phase-amplitude coupling comodulogram.*

Development
~~~~~~~~~~~

YASA was created and is maintained by `Raphael Vallat <https://raphaelvallat.com>`_. Contributions are more than welcome so feel free to contact me, open an issue or submit a pull request!

To see the code or report a bug, please visit the `GitHub repository <https://github.com/raphaelvallat/yasa>`_.

Note that this program is provided with NO WARRANTY OF ANY KIND.

Citation
~~~~~~~~

To cite YASA, please use the Zenodo DOI:

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2370600.svg
   :target: https://doi.org/10.5281/zenodo.2370600
