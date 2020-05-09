.. -*- mode: rst -*-

|

.. image:: https://badge.fury.io/py/yasa.svg
    :target: https://badge.fury.io/py/yasa

.. image:: https://img.shields.io/badge/python-3.6%20%7C%203.7-blue.svg
    :target: https://www.python.org/downloads/

.. image:: https://img.shields.io/github/license/raphaelvallat/yasa.svg
    :target: https://github.com/raphaelvallat/yasa/blob/master/LICENSE

.. image:: https://travis-ci.org/raphaelvallat/yasa.svg?branch=master
    :target: https://travis-ci.org/raphaelvallat/yasa

.. .. image:: https://ci.appveyor.com/api/projects/status/4ua0pwy62jhpd9mx?svg=true
..     :target: https://ci.appveyor.com/project/raphaelvallat/yasa

.. image:: https://codecov.io/gh/raphaelvallat/yasa/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/raphaelvallat/yasa

.. image:: https://pepy.tech/badge/yasa
    :target: https://pepy.tech/badge/yasa

.. .. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2370600.svg
..    :target: https://doi.org/10.5281/zenodo.2370600

----------------

.. figure::  /pictures/yasa_logo.png
  :align:   center

**YASA** (*Yet Another Spindle Algorithm*) is a sleep analysis toolbox in Python. YASA includes several fast and convenient command-line functions to:

* Automatically detect sleep spindles, slow-waves, and rapid eye movements on single and multi-channel EEG data
* Automatically reject major artifacts on single or multi-channel EEG data
* Perform advanced spectral analyses: spectral bandpower, phase-amplitude coupling, event-locked analyses, 1/f, and more!
* Manipulate hypnogram and calculate sleep statistics

For more details, check out the `API documentation <https://raphaelvallat.com/yasa/build/html/index.html>`_ or try the
`tutorial (Jupyter notebooks) <https://github.com/raphaelvallat/yasa/tree/master/notebooks>`_.

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

If you have sleep EEG data in standard formats (e.g. EDF or BrainVision), you can use the excellent `MNE package <https://mne.tools/stable/index.html>`_ to load and preprocess your data in Python. A simple preprocessing pipeline using MNE is shown below:

.. code-block:: python

  import mne
  # Load the EDF file, excluding the EOGs and EKG channels
  raw = mne.io.read_raw_edf('MYEDFFILE.edf', preload=True, exclude=['EOG1', 'EOG2', 'EKG'])
  raw.resample(100)                      # Downsample the data to 100 Hz
  raw.filter(0.1, 40)                    # Apply a bandpass filter from 0.1 to 40 Hz
  raw.pick_channels(['C4-A1', 'C3-A2'])  # Select a subset of EEG channels


How do I get started with YASA?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to dive right in, you can simply go to the main documentation (:ref:`api_ref`) and try to apply YASA's functions on your own EEG data. However, for most users, we strongly recommend that you first try running the examples Jupyter notebooks to get a sense of how YASA works and what it can do! The advantage is that the notebooks also come with example datasets so they should work right out of the box as long as you've installed YASA first. The notebooks and datasets can be found on `GitHub <https://github.com/raphaelvallat/yasa/tree/master/notebooks>`_ (make sure that you download the whole *notebooks/* folder). A short description of all notebooks is provided below:

**Spindles**

* `01_spindles_detection <https://github.com/raphaelvallat/yasa/blob/master/notebooks/01_spindles_detection.ipynb>`_: single-channel spindles detection and step-by-step description of the spindles detection algorithm.
* `02_spindles_detection_multi <https://github.com/raphaelvallat/yasa/blob/master/notebooks/02_spindles_detection_multi.ipynb>`_: multi-channel spindles detection.
* `03_spindles_detection_NREM_only <https://github.com/raphaelvallat/yasa/blob/master/notebooks/03_spindles_detection_NREM_only.ipynb>`_: how to limit the spindles detection on specific sleep stages using an hypnogram.
* `04_spindles_slow_fast <https://github.com/raphaelvallat/yasa/blob/master/notebooks/04_spindles_slow_fast.ipynb>`_: slow versus fast spindles.
* `run_visbrain <https://github.com/raphaelvallat/yasa/blob/master/notebooks/run_visbrain.py>`_: interactive display of the detected spindles using the Visbrain visualization software in Python.

**Slow-waves**

* `05_sw_detection <https://github.com/raphaelvallat/yasa/blob/master/notebooks/05_sw_detection.ipynb>`_: single-channel slow-waves detection and step-by-step description of the slow-waves detection algorithm.
* `06_sw_detection_multi <https://github.com/raphaelvallat/yasa/blob/master/notebooks/06_sw_detection_multi.ipynb>`_: multi-channel slow-waves detection.

**Rapid Eye Movements (REMs)**

* `07_REMs_detection <https://github.com/raphaelvallat/yasa/blob/master/notebooks/07_REMs_detection.ipynb>`_: REMs detection.

**Spectral analysis**

* `08_bandpower <https://github.com/raphaelvallat/yasa/blob/master/notebooks/08_bandpower.ipynb>`_: calculate spectral band power, optionally averaged across channels and sleep stages.
* `09_IRASA <https://github.com/raphaelvallat/yasa/blob/master/notebooks/09_IRASA.ipynb>`_: separate the aperiodic (= fractal = 1/f) components of the EEG power spectrum using the IRASA method.
* `10_spectrogram <https://github.com/raphaelvallat/yasa/blob/master/notebooks/10_spectrogram.ipynb>`_: plot a multi-taper full-night spectrogram on single-channel EEG data with the hypnogram on top.
* `11_nonlinear_features <https://github.com/raphaelvallat/yasa/blob/master/notebooks/11_nonlinear_features.ipynb>`_: calculate non-linear EEG features on 30-seconds epochs and perform sleep stage classification.
* `12_spindles-SO_coupling <https://github.com/raphaelvallat/yasa/blob/master/notebooks/12_spindles-SO_coupling.ipynb>`_: slow-oscillations/spindles phase-amplitude coupling and data-driven comodulogram.

**Artifact rejection**

* `13_artifact_rejection <https://github.com/raphaelvallat/yasa/blob/master/notebooks/13_artifact_rejection.ipynb>`_: automatic artifact rejection on single and multi-channel EEG data.

Typical use: spindles detection
-------------------------------

.. code-block:: python

  import yasa

  # 1) Single-channel spindles detection, in its simplest form.
  # There are many optional arguments that you can change to customize the detection.
  sp = yasa.spindles_detect(data, sf)
  # The output of the the detection (`sp`) is a class that has several attributes and methods.
  # For instance, to get the full detection dataframe, one can simply use:
  sp.summary()
  # To plot an average template of all the detected spindles,
  # centered around the most prominent peak (+/- 1 second)
  sp.plot_average(center='Peak', time_before=1, time_after=1)

  # 2) Multi-channels spindles detection limited to N2/N3 sleep, with automatic outlier rejection
  sp = yasa.spindles_detect(data, sf, ch_names, hypno=hypno, include=(2, 3), remove_outliers=True)
  # Return spindles count / density and parameters averaged across channels and sleep stages
  sp.summary(grp_stage=True, grp_chan=True)

The output of ``sp.summary()`` is a `pandas DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_ where each row is a  detected spindle and each column a parameter of this event, including the start and end timestamps (in seconds from the beginning of the data), duration, amplitude, etc.

.. table::
   :widths: auto

   =======  =====  ==========  ===========  =====  ==========  ==========  ===========  ==============  ==========
     Start    End    Duration    Amplitude    RMS    AbsPower    RelPower    Frequency    Oscillations    Symmetry
   =======  =====  ==========  ===========  =====  ==========  ==========  ===========  ==============  ==========
      3.32   4.06        0.74        81.80  19.65        2.72        0.49        12.85              10        0.67
     13.26  13.85        0.59        99.30  24.49        2.82        0.24        12.15               7        0.25
   =======  =====  ==========  ===========  =====  ==========  ==========  ===========  ==============  ==========

In turn, the detection dataframe can be used to plot the events.

.. figure::  /pictures/detection.png
   :align:   center

Interactive visualization with Visbrain
---------------------------------------

YASA can also be used in combination with the `Sleep <http://visbrain.org/sleep.html>`_ module of the `Visbrain visualization package <http://visbrain.org/index.html>`_. The result of the detection can then easily be displayed and checked in an interactive graphical user interface. To do so, load Visbrain using the following python file (make sure to update *'PATH/TO/EEGFILE'*).

.. code-block:: python

  from visbrain.gui import Sleep
  from yasa import spindles_detect

  sl = Sleep(data='PATH/TO/EEGFILE')

  def fcn_spindle(data, sf, time, hypno):
      """Replace Visbrain built-in spindles detection by YASA algorithm.
      See http://visbrain.org/sleep.html#use-your-own-detections-in-sleep
      """
      # Apply on the full recording...
      # sp = spindles_detect(data, sf).summary()
      # ...or on NREM sleep only
      sp = spindles_detect(data, sf, hypno=hypno).summary()
      return (sp[['Start', 'End']].values * sf).astype(int)

  sl.replace_detections('spindle', fcn_spindle)
  sl.show()

Then navigate to the *Detection* tab and click on *Apply* to run the YASA algorithm on the specified channel.

.. figure::  /pictures/visbrain.PNG
   :align:   center


Outlier rejection
-----------------

YASA incorporates an optional post-processing step to identify and remove pseudo (outlier) events.
The method is based on a machine-learning algorithm (the `Isolation Forest <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html>`_, implemented in the `scikit-learn <https://scikit-learn.org/stable/index.html>`_ package),
which uses the events parameters (e.g. amplitude, duration, frequency, etc) as input features to identify *aberrant* spindles / slow-waves / REMs.

To activate this post-processing step, simply use:

.. code-block:: python

  import yasa
  yasa.spindles_detect(data, sf, remove_outliers=True)  # Spindles
  yasa.sw_detect(data, sf, remove_outliers=True)        # Slow-waves
  yasa.rem_detect(loc, roc, sf, remove_outliers=True)   # REMs


Gallery
~~~~~~~

Below some plots demonstrating the functionalities of YASA. To reproduce these, check out the `tutorial (Jupyter notebooks) <https://github.com/raphaelvallat/yasa/tree/master/notebooks>`_.

.. figure::  /pictures/gallery.png
  :align:   center

  *The two top plots show an overlay of the detected spindles (blue) and slow-waves (red) on real EEG data. The middle right panel shows a time-frequency representation of the whole-night recording (spectrogram), plotted with the hypnogram (sleep stages) on top. The middle right panel shows the sleep stage probability transition matrix, calculated across the entire night. The left and right plots of the bottom row show the average template of all detected slow-waves and spindles across the entire night, stratified by channels. The middle bottom plot shows a phase-amplitude coupling comodulogram between slower (0.2-4Hz) and faster (7.5-25Hz) frequency ranges.*

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

|
