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

.. image:: https://ci.appveyor.com/api/projects/status/4ua0pwy62jhpd9mx?svg=true
    :target: https://ci.appveyor.com/project/raphaelvallat/yasa

.. .. image:: https://codecov.io/gh/raphaelvallat/yasa/branch/master/graph/badge.svg
..     :target: https://codecov.io/gh/raphaelvallat/yasa

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2370600.svg
   :target: https://doi.org/10.5281/zenodo.2370600

----------------

YASA
====

YASA (*Yet Another Spindle Algorithm*) is a fast, robust, and data-agnostic sleep spindles & slow-waves detection algorithm written in Python 3.

The **sleep spindles** algorithm of YASA is largely inspired by the method described in:

- Lacourse, K., Delfrate, J., Beaudry, J., Peppard, P. & Warby, S. C. (2018). A sleep spindle detection algorithm that emulates human expert spindle scoring. *J. Neurosci. Methods*.

The **slow-waves** detection algorithm is adapted from:

- Massimini, M., Huber, R., Ferrarelli, F., Hill, S. & Tononi, G. (2004). The sleep slow oscillation as a traveling wave. *J. Neurosci.*.

- Carrier, J. et al. (2011). Sleep slow wave changes during the middle years of life. *Eur. J. Neurosci*.

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

**Note**

Several functions of YASA are written using `Numba <http://numba.pydata.org/>`_, a just-in-time compiler for Python. This allows to greatly speed up the computation time (typically a few seconds for a full night recording).

Examples
~~~~~~~~

Notebooks
---------

**Spindles**

1. `notebooks/01_spindles_detection.ipynb <notebooks/00_spindles_detection.ipynb>`_: single-channel spindles detection and step-by-step description of the algorithm.
2. `notebooks/02_spindles_detection_multi.ipynb <notebooks/01_spindles_detection_multi.ipynb>`_: multi-channel spindles detection using MNE Raw objects.
3. `notebooks/03_slow_fast_spindles.ipynb <notebooks/02_slow_fast_spindles.ipynb>`_: slow and fast spindles differentiation.
4. `notebooks/04_detection_NREM_only.ipynb <notebooks/03_detection_NREM_only.ipynb>`_: detection on NREM sleep only.
5. `notebooks/05_run_visbrain.py <notebooks/04_run_visbrain.py>`_: interactive display with the Visbrain graphical user interface.

**Slow-waves**

6. `notebooks/06_sw_detect.ipynb <notebooks/05_sw_detect.ipynb>`_: single-channel slow-waves detection and step-by-step description of the algorithm.

Typical uses
------------

.. code-block:: python

  import yasa

  # SLEEP SPINDLES
  # ==============
  # 1 - Single-channel spindles detection
  yasa.spindles_detect(data, sf)

  # 2 - Single-channel full command (shows all the default parameters)
  yasa.spindles_detect(data, sf, hypno=None, freq_sp=(12, 15), duration=(0.5, 2),
                       freq_broad=(1, 30), min_distance=500, downsample=True,
                       thresh={'rel_pow': 0.2, 'corr': 0.65, 'rms': 1.5},
                       remove_outliers=False)

  # 3 - Multi-channels detection on NREM sleep only (requires an hypnogram)
  yasa.spindles_detect_multi(data, sf, ch_names, hypno=hypno)

  # 4 - Multi-channels detection with automatic outlier rejection
  yasa.spindles_detect_multi(data, sf, ch_names, hypno=hypno, remove_outliers=True)

  # SLOW-WAVES
  # ==========
  # 1 - Single-channel slow-wave detection
  yasa.sw_detect(data, sf)

  # 2 - Single-channel full command
  # Long version (with all the optional arguments)
  sw = sw_detect(data, sf, hypno=hypno, freq_sw=(0.3, 3.5), dur_neg=(0.3, 1.5),
                 dur_pos=(0.1, 1), amp_neg=(40, 300), amp_pos=(10, 150),
                 amp_ptp=(75, 400), downsample=True, remove_outliers=False)

The result of the detection is a `pandas DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_...

.. table:: Output
   :widths: auto

=======  =====  ==========  ===========  =====  ==========  ==========  ===========  ==============  ==========
  Start    End    Duration    Amplitude    RMS    AbsPower    RelPower    Frequency    Oscillations    Symmetry
=======  =====  ==========  ===========  =====  ==========  ==========  ===========  ==============  ==========
   3.32   4.06        0.74        81.80  19.65        2.72        0.49        12.85              10        0.67
  13.26  13.85        0.59        99.30  24.49        2.82        0.24        12.15               7        0.25
=======  =====  ==========  ===========  =====  ==========  ==========  ===========  ==============  ==========

...that can be easily used to plot the detected spindles / slow-waves.

.. figure::  notebooks/detection.png
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
      sp = spindles_detect(data, sf)
      # Alternatively if you want to apply the detection only on NREM sleep
      # sp = spindles_detect(data, sf, hypno=hypno)
      return (sp[['Start', 'End']].values * sf).astype(int)

  sl.replace_detections('spindle', fcn_spindle)
  sl.show()

Then navigate to the *Detection* tab and click on *Apply* to run the YASA algorithm on the specified channel.

.. figure::  images/visbrain.PNG
   :align:   center


Outlier rejection
-----------------

YASA incorporates an optional post-processing step to identify and remove pseudo (fake) spindles / slow-waves.
The method is based on a machine-learning algorithm (the `Isolation Forest <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html>`_, implemented in the `scikit-learn <https://scikit-learn.org/stable/index.html>`_ package),
which uses the spindles parameters (e.g. amplitude, duration, frequency, etc) as input features to identify *aberrant* spindles / slow-waves.

To activate this post-processing step, simply use:

.. code-block:: python

  import yasa
  yasa.spindles_detect(data, sf, remove_outliers=True)  # For spindles
  yasa.sw_detect(data, sf, remove_outliers=True)  # For slow-waves


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
