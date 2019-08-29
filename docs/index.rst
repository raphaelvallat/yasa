.. -*- mode: rst -*-

|

.. image:: https://badge.fury.io/py/yasa.svg
    :target: https://badge.fury.io/py/yasa

.. image:: https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7-blue.svg
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

**YASA** (*Yet Another Spindle Algorithm*) is a fast and robust Python 3 toolbox to detect sleep microstructure events from EEG recordings.

The **sleep spindles** algorithm of YASA is largely inspired by the method described in:

- Lacourse, ..., & Warby. (2018). `A sleep spindle detection algorithm that emulates human expert spindle scoring <https://doi.org/10.1016/j.jneumeth.2018.08.014>`_. *J. Neurosci. Methods*.

The **slow-waves** detection algorithm is adapted from:

- Massimini, ..., & Tononi. (2004). `The sleep slow oscillation as a traveling wave <https://doi.org/10.1523/JNEUROSCI.1318-04.2004>`_. *J. Neurosci*.

- Carrier, ..., & Filipini. (2011). `Sleep slow wave changes during the middle years of life <https://doi.org/10.1111/j.1460-9568.2010.07543.x>`_. *Eur. J. Neurosci*.

The **rapid eye movements (REMs)** detection algorithm is a custom adaptation inspired by:

- Agarwal, ..., & Gotman. (2005). `Detection of rapid-eye movements in sleep studies <https://ieeexplore.ieee.org/abstract/document/1463327/>`_. *IEEE Transactions on biomedical engineering*.

- Yetton, ..., & Mednick. (2016). `Automatic detection of rapid eye movements (REMs): A machine learning approach <https://www.sciencedirect.com/science/article/pii/S0165027015004173>`_. *Journal of neuroscience methods*.

In addition, YASA also provides some convenient functions to **load and manipulate hypnogram** (sleep stages vector), and to perform **spectral analysis**.

Installation
~~~~~~~~~~~~

.. code-block:: shell

  pip install --upgrade yasa

**Dependencies**

- python>=3.5
- numpy>=1.14
- scipy>=1.1.0
- pandas>=0.23,
- mne>=0.17.0
- numba>=0.39.0
- scikit-learn>=0.20

Several functions of YASA are written using `Numba <http://numba.pydata.org/>`_, a just-in-time compiler for Python. This allows to greatly speed up the computation time (typically a few seconds for a full night recording).

**What are the prerequisites for using YASA?**

In order to use YASA, you need:

- Some basic knowledge of Python and especially the `NumPy <https://docs.scipy.org/doc/numpy/user/quickstart.html>`_, `Pandas <https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html>`_ and `MNE <https://martinos.org/mne/stable/index.html>`_ libraries.
- A Python editor: YASA works best with `Jupyter Lab <https://jupyterlab.readthedocs.io/en/stable/index.html>`_, a web-based interactive user interface.
- Some sleep EEG data, either as a NumPy array, or as a raw MNE object (for instance, using the `mne.io.read_raw_edf <https://mne-tools.github.io/stable/generated/mne.io.read_raw_edf.html>`_ function for EDF file). The units of the data MUST be uV.
- Optionally, a sleep staging vector (a.k.a hypnogram) to run the detections on specific sleep stages. To facilitate masking and indexing operations, the data and hypnogram MUST have the same sampling frequency and number of samples. Fortunately, YASA provide some convenient functions to load and upsample hypnogram data to the desired shape.

.. note::
      The default hypnogram format in YASA is a one dimensional integer vector where:
        - -1 = Artefact / Movement
        - 0 = Wake
        - 1 = N1 sleep
        - 2 = N2 sleep
        - 3 = N3 sleep
        - 4 = REM

Examples
~~~~~~~~

Check out the :ref:`api_ref` for more details on YASA's functions.

Notebooks
---------

The examples Jupyter notebooks are really what make YASA great! In addition to showing how to use the main functions of YASA, they also provide an extensive step-by-step description of the detection algorithms, as well as several useful code snippets to analyze and plot your data.

**Spindles**

* `01_spindles_detection.ipynb <https://github.com/raphaelvallat/yasa/blob/master/notebooks/01_spindles_detection.ipynb>`_: single-channel spindles detection and step-by-step description of the algorithm.
* `02_spindles_detection_multi.ipynb <https://github.com/raphaelvallat/yasa/blob/master/notebooks/02_spindles_detection_multi.ipynb>`_: multi-channel spindles detection using MNE data.
* `03_spindles_detection_NREM_only.ipynb <https://github.com/raphaelvallat/yasa/blob/master/notebooks/03_spindles_detection_NREM_only.ipynb>`_: spindles detection on NREM sleep only.
* `04_spindles_slow_fast.ipynb <https://github.com/raphaelvallat/yasa/blob/master/notebooks/04_spindles_slow_fast.ipynb>`_: slow and fast spindles analysis.
* `05_run_visbrain.py <https://github.com/raphaelvallat/yasa/blob/master/notebooks/05_run_visbrain.py>`_: interactive display with the Visbrain graphical user interface.

**Slow-waves**

* `06_sw_detection.ipynb <https://github.com/raphaelvallat/yasa/blob/master/notebooks/06_sw_detection.ipynb>`_: single-channel slow-waves detection and step-by-step description of the algorithm.
* `07_sw_detection_multi.ipynb <https://github.com/raphaelvallat/yasa/blob/master/notebooks/07_sw_detection_multi.ipynb>`_: multi-channel slow-waves detection using MNE data.
* `08_sw_average.ipynb <https://github.com/raphaelvallat/yasa/blob/master/notebooks/08_sw_average.ipynb>`_: plot the average template of the detected slow-waves, per channel.

**Rapid Eye Movements (REMs)**

* `09_REMs_detection.ipynb <https://github.com/raphaelvallat/yasa/blob/master/notebooks/09_REMs_detection.ipynb>`_: REMs detection.

**Spectral analysis**

* `10_bandpower.ipynb <https://github.com/raphaelvallat/yasa/blob/master/notebooks/10_bandpower.ipynb>`_: bandpower per channel and per sleep stage.
* `11_IRASA.ipynb <https://github.com/raphaelvallat/yasa/blob/master/notebooks/11_IRASA.ipynb>`_: separate the aperiodic (= fractal = 1/f) components of the EEG power spectra using the IRASA method.

Typical uses
------------

.. code-block:: python

  import yasa

  # SLEEP SPINDLES
  # ==============
  # Single-channel spindles detection
  yasa.spindles_detect(data, sf)

  # Single-channel full command (shows all the default implicit parameters)
  yasa.spindles_detect(data, sf, hypno=None, include=(1, 2, 3),
                       freq_sp=(12, 15), duration=(0.5, 2), freq_broad=(1, 30),
                       min_distance=500, downsample=True,
                       thresh={'rel_pow': 0.2, 'corr': 0.65, 'rms': 1.5},
                       remove_outliers=False)

  # Multi-channels detection on NREM sleep only (requires an hypnogram)
  yasa.spindles_detect_multi(data, sf, ch_names, hypno=hypno)

  # Multi-channels detection on N2 sleep only with automatic outlier rejection
  yasa.spindles_detect_multi(data, sf, ch_names, hypno=hypno, include=(2), remove_outliers=True)

  # SLOW-WAVES
  # ==========
  # Single-channel slow-wave detection
  yasa.sw_detect(data, sf)

  # Single-channel full command (shows all the default implicit parameters)
  yasa.sw_detect(data, sf, hypno=hypno, include=(2, 3), freq_sw=(0.3, 3.5),
                 dur_neg=(0.3, 1.5), dur_pos=(0.1, 1), amp_neg=(40, 300),
                 amp_pos=(10, 150), amp_ptp=(75, 400), downsample=True,
                 remove_outliers=False)

  # Multi-channel slow-waves detection on N2 + N3 sleep only (requires an hypnogram)
  yasa.sw_detect_multi(data, sf, ch_names, hypno=hypno)

  # RAPID EYE MOVEMENTS
  # ===================
  # Default detection (requires both LOC and ROC EOG channels)
  yasa.rem_detect(loc, roc, sf)

  # On REM sleep only + all implicit parameters
  yasa.rem_detect(loc, roc, sf, hypno=hypno, include=4, amplitude=(50, 325),
                  duration=(0.3, 1.5), freq_rem=(0.5, 5), downsample=True,
                  remove_outliers=False)

The result of the detection is a `pandas DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_ where each row is a unique detected event (e.g. spindle, slow-waves, REMs) and each column a parameter of this event, including, the start and end timestamps, duration, amplitude, etc.

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
      # sp = spindles_detect(data, sf)
      # ...or on NREM sleep only
      sp = spindles_detect(data, sf, hypno=hypno)
      return (sp[['Start', 'End']].values * sf).astype(int)

  sl.replace_detections('spindle', fcn_spindle)
  sl.show()

Then navigate to the *Detection* tab and click on *Apply* to run the YASA algorithm on the specified channel.

.. figure::  https://raw.githubusercontent.com/raphaelvallat/yasa/master/images/visbrain.PNG
   :align:   center


Outlier rejection
-----------------

YASA incorporates an optional post-processing step to identify and remove pseudo (fake) events.
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
