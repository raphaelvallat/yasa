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
* Easily manipulate sleep staging vector (hypnogram)

For more details, check out the `API documentation <https://raphaelvallat.com/yasa/build/html/index.html>`_ or try the
`tutorial <https://github.com/raphaelvallat/yasa/tree/master/notebooks>`_.

Installation
~~~~~~~~~~~~

.. code-block:: shell

  pip install --upgrade yasa

**Dependencies**

- python>=3.6
- numpy
- scipy
- pandas
- matplotlib
- seaborn
- mne
- numba
- scikit-learn
- `tensorpac <https://etiennecmb.github.io/tensorpac/>`_ (for phase-amplitude coupling)
- `pyriemann <https://pyriemann.readthedocs.io/en/latest/api.html>`_ (for atifact rejection based on covariance matrices)
- `lspopt <https://github.com/hbldh/lspopt>`_ (for multitaper spectrogram estimation)

Several functions of YASA are written using `Numba <http://numba.pydata.org/>`_, a just-in-time compiler for Python. This allows to greatly speed up the computation time of the microstructure detection (typically a few seconds for a full night recording).

**What are the prerequisites for using YASA?**

YASA works best when used in combination with the `MNE library <https://mne.tools/stable/index.html>`_. For example, to read an EDF file,
one can use the `mne.io.read_raw_edf <https://mne.tools/stable/generated/mne.io.read_raw_edf.html?highlight=read_raw_edf#mne.io.read_raw_edf>`_ function.

In order to use YASA, you need:

- Some basic knowledge of Python and especially the `NumPy <https://docs.scipy.org/doc/numpy/user/quickstart.html>`_, `Pandas <https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html>`_ and `MNE <https://martinos.org/mne/stable/index.html>`_ libraries.
- A Python editor: YASA works best with `Jupyter Lab <https://jupyterlab.readthedocs.io/en/stable/index.html>`_, a web-based interactive user interface.
- Some sleep EEG data, either as a NumPy array, or as a `mne.Raw <https://mne.tools/stable/generated/mne.io.Raw.html>`_ object.
- Optionally, a sleep staging vector (a.k.a hypnogram) to perform calculations on specific sleep stages. To facilitate masking and indexing operations, the data and hypnogram MUST have the same sampling frequency and number of samples. YASA provide some convenient functions to load and upsample hypnogram data to the desired shape.

.. note::
      The default hypnogram format in YASA is a one dimensional integer vector where:
        - -2 = Unscored
        - -1 = Artefact / Movement
        - 0 = Wake
        - 1 = N1 sleep
        - 2 = N2 sleep
        - 3 = N3 sleep
        - 4 = REM

Examples
~~~~~~~~

Check out the :ref:`api_ref` for more details on YASA's functions.

Tutorial
--------

The examples Jupyter notebooks are really what make YASA great! In addition to showing how to use the main functions of YASA, they also provide an extensive step-by-step description of the algorithms, as well as several useful code snippets to analyze and plot your data.

**Spindles**

* `01_spindles_detection.ipynb <https://github.com/raphaelvallat/yasa/blob/master/notebooks/01_spindles_detection.ipynb>`_: single-channel spindles detection and step-by-step description of the algorithm.
* `02_spindles_detection_multi.ipynb <https://github.com/raphaelvallat/yasa/blob/master/notebooks/02_spindles_detection_multi.ipynb>`_: multi-channel spindles detection using MNE data.
* `03_spindles_detection_NREM_only.ipynb <https://github.com/raphaelvallat/yasa/blob/master/notebooks/03_spindles_detection_NREM_only.ipynb>`_: spindles detection on NREM sleep only.
* `04_spindles_slow_fast.ipynb <https://github.com/raphaelvallat/yasa/blob/master/notebooks/04_spindles_slow_fast.ipynb>`_: slow and fast spindles analysis.
* `run_visbrain.py <https://github.com/raphaelvallat/yasa/blob/master/notebooks/run_visbrain.py>`_: interactive display with the Visbrain graphical user interface.

**Slow-waves**

* `05_sw_detection.ipynb <https://github.com/raphaelvallat/yasa/blob/master/notebooks/05_sw_detection.ipynb>`_: single-channel slow-waves detection and step-by-step description of the algorithm.
* `06_sw_detection_multi.ipynb <https://github.com/raphaelvallat/yasa/blob/master/notebooks/06_sw_detection_multi.ipynb>`_: multi-channel slow-waves detection using MNE data.

**Rapid Eye Movements (REMs)**

* `07_REMs_detection.ipynb <https://github.com/raphaelvallat/yasa/blob/master/notebooks/07_REMs_detection.ipynb>`_: REMs detection.

**Spectral analysis**

* `08_bandpower.ipynb <https://github.com/raphaelvallat/yasa/blob/master/notebooks/08_bandpower.ipynb>`_: bandpower per channel and per sleep stage.
* `09_IRASA.ipynb <https://github.com/raphaelvallat/yasa/blob/master/notebooks/09_IRASA.ipynb>`_: separate the aperiodic (= fractal = 1/f) components of the EEG power spectra using the IRASA method.
* `10_spectrogram.ipynb <https://github.com/raphaelvallat/yasa/blob/master/notebooks/10_spectrogram.ipynb>`_: plot a multi-taper full-night spectrogram on single-channel EEG data with the hypnogram on top.
* `11_nonlinear_features.ipynb <https://github.com/raphaelvallat/yasa/blob/master/notebooks/11_nonlinear_features.ipynb>`_: extract epoch-based non-linear features of sleep EEG.
* `12_spindles-SO_coupling.ipynb <https://github.com/raphaelvallat/yasa/blob/master/notebooks/12_spindles-SO_coupling.ipynb>`_: performs event-locked spindles-SO coupling, as well as data-driven Phase-Amplitude Coupling.

**Artifact rejection**

* `13_artifact_rejection.ipynb <https://github.com/raphaelvallat/yasa/blob/master/notebooks/13_artifact_rejection.ipynb>`_: automatic artifact rejection on single and multi-channel EEG data.

Typical uses
------------

.. code-block:: python

  import yasa

  # Single-channel spindles detection (shows all the default implicit parameters)
  sp = yasa.spindles_detect(data, sf=None, ch_names=None, hypno=None,
                            include=(1, 2, 3), freq_sp=(12, 15), freq_broad=(1, 30),
                            duration=(0.5, 2),  min_distance=500,
                            thresh={'rel_pow': 0.2, 'corr': 0.65, 'rms': 1.5},
                            multi_only=False, remove_outliers=False, verbose=False)

  # Return a Pandas DataFrame with all detected spindles
  sp.summary()

  # Multi-channels detection on N2 sleep only with automatic outlier rejection
  sp = yasa.spindles_detect(data, sf, ch_names, hypno=hypno, include=(2), remove_outliers=True)

  # Return spindles count / density and properties averaged across channels and sleep stages
  sp.summary(grp_stage=True, grp_chan=True)

The output of the detection is a `pandas DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_ where each row is a unique detected event (e.g. spindle, slow-waves, REMs) and each column a parameter of this event, including, the start and end timestamps, duration, amplitude, etc.

.. table:: Output dataframe
   :widths: auto

   =======  =====  ==========  ===========  =====  ==========  ==========  ===========  ==============  ==========
     Start    End    Duration    Amplitude    RMS    AbsPower    RelPower    Frequency    Oscillations    Symmetry
   =======  =====  ==========  ===========  =====  ==========  ==========  ===========  ==============  ==========
      3.32   4.06        0.74        81.80  19.65        2.72        0.49        12.85              10        0.67
     13.26  13.85        0.59        99.30  24.49        2.82        0.24        12.15               7        0.25
   =======  =====  ==========  ===========  =====  ==========  ==========  ===========  ==============  ==========

In turn, the detection dataframe can be easily used to plot the events.

.. figure::  https://raw.githubusercontent.com/raphaelvallat/yasa/master/notebooks/detection.png
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

.. figure::  https://raw.githubusercontent.com/raphaelvallat/yasa/master/images/visbrain.PNG
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
