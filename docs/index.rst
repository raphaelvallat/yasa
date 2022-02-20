.. -*- mode: rst -*-

|

.. image:: https://badge.fury.io/py/yasa.svg
    :target: https://badge.fury.io/py/yasa

.. image:: https://img.shields.io/github/license/raphaelvallat/yasa.svg
    :target: https://github.com/raphaelvallat/yasa/blob/master/LICENSE

.. image:: https://codecov.io/gh/raphaelvallat/yasa/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/raphaelvallat/yasa

.. image:: https://pepy.tech/badge/yasa
    :target: https://pepy.tech/badge/yasa

----------------

.. figure::  /pictures/yasa_logo.png
  :align:   center

**YASA** (*Yet Another Spindle Algorithm*) is a command-line sleep analysis toolbox in Python. The main functions of YASA are:

* Automatic sleep staging of polysomnography data (see `preprint article <https://doi.org/10.1101/2021.05.28.446165>`_).
* Event detection: sleep spindles, slow-waves and rapid eye movements, on single or multi-channel EEG data.
* Artefact rejection, on single or multi-channel EEG data.
* Spectral analyses: bandpower, phase-amplitude coupling, 1/f slope, and more!
* Hypnogram analysis: sleep statistics and stage tranisitions.

For more details, check out the `API documentation <https://raphaelvallat.com/yasa/build/html/index.html>`_, try the
`tutorial (Jupyter notebooks) <https://github.com/raphaelvallat/yasa/tree/master/notebooks>`_ or read the `FAQ <https://raphaelvallat.com/yasa/build/html/faq.html>`_.

**********

Installation
~~~~~~~~~~~~

To install YASA, simply open a terminal or Anaconda command prompt and enter:

.. code-block:: shell

  pip install --upgrade yasa

**What are the prerequisites for using YASA?**

To use YASA, all you need is:

- Some basic knowledge of Python, especially the `NumPy <https://docs.scipy.org/doc/numpy/user/quickstart.html>`_, `Pandas <https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html>`_ and `MNE <https://martinos.org/mne/stable/index.html>`_ packages.
- A Python editor: YASA works best with `Jupyter Lab <https://jupyterlab.readthedocs.io/en/stable/index.html>`_, a web-based interactive user interface.
- Some sleep EEG data and optionally a sleep staging file (hypnogram).

**I have sleep EEG data in European Data Format (.edf), how do I load the data in Python?**

If you have sleep EEG data in standard formats (e.g. EDF or BrainVision), you can use the `MNE package <https://mne.tools/stable/index.html>`_ to load and preprocess your data in Python. A simple preprocessing pipeline using MNE is shown below:

.. code-block:: python

  import mne
  # Load the EDF file
  raw = mne.io.read_raw_edf('MYEDFFILE.edf', preload=True)
  # Downsample the data to 100 Hz
  raw.resample(100)
  # Apply a bandpass filter from 0.1 to 40 Hz
  raw.filter(0.1, 40)
  # Select a subset of EEG channels
  raw.pick_channels(['C4-A1', 'C3-A2'])

**********

How do I get started with YASA?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to dive right in, you can simply go to the main documentation (:ref:`api_ref`) and try to apply YASA's functions on your own EEG data.
However, for most users, we strongly recommend that you first try running the examples Jupyter notebooks to get a sense of how YASA works and what it can do!
The notebooks also come with example datasets so they should work right out of the box as long as you've installed YASA first.
The notebooks and datasets can be found on `GitHub <https://github.com/raphaelvallat/yasa/tree/master/notebooks>`_ (make sure that you download the whole *notebooks/* folder). A short description of all notebooks is provided below:

**Automatic sleep staging**

* `automatic_staging <https://github.com/raphaelvallat/yasa/blob/master/notebooks/14_automatic_sleep_staging.ipynb>`_: Automatic sleep staging of polysomnography data.

**Event detection**

* `spindles_detection <https://github.com/raphaelvallat/yasa/blob/master/notebooks/01_spindles_detection.ipynb>`_: single-channel spindles detection and step-by-step description of the spindles detection algorithm.
* `spindles_detection_multi <https://github.com/raphaelvallat/yasa/blob/master/notebooks/02_spindles_detection_multi.ipynb>`_: multi-channel spindles detection.
* `spindles_detection_NREM_only <https://github.com/raphaelvallat/yasa/blob/master/notebooks/03_spindles_detection_NREM_only.ipynb>`_: how to limit the spindles detection on specific sleep stages using an hypnogram.
* `spindles_slow_fast <https://github.com/raphaelvallat/yasa/blob/master/notebooks/04_spindles_slow_fast.ipynb>`_: slow versus fast spindles.
* `sw_detection <https://github.com/raphaelvallat/yasa/blob/master/notebooks/05_sw_detection.ipynb>`_: single-channel slow-waves detection and step-by-step description of the slow-waves detection algorithm.
* `sw_detection_multi <https://github.com/raphaelvallat/yasa/blob/master/notebooks/06_sw_detection_multi.ipynb>`_: multi-channel slow-waves detection.
* `artifact_rejection <https://github.com/raphaelvallat/yasa/blob/master/notebooks/13_artifact_rejection.ipynb>`_: automatic artifact rejection on single and multi-channel EEG data.
* `REMs_detection <https://github.com/raphaelvallat/yasa/blob/master/notebooks/07_REMs_detection.ipynb>`_: REMs detection.
* `run_visbrain <https://github.com/raphaelvallat/yasa/blob/master/notebooks/run_visbrain.py>`_: interactive display of the detected spindles using the Visbrain visualization software in Python.

**Spectral analysis**

* `bandpower <https://github.com/raphaelvallat/yasa/blob/master/notebooks/08_bandpower.ipynb>`_: calculate spectral band power, optionally averaged across channels and sleep stages.
* `IRASA <https://github.com/raphaelvallat/yasa/blob/master/notebooks/09_IRASA.ipynb>`_: separate the aperiodic (= fractal = 1/f) components of the EEG power spectrum using the IRASA method.
* `spectrogram <https://github.com/raphaelvallat/yasa/blob/master/notebooks/10_spectrogram.ipynb>`_: plot a multi-taper full-night spectrogram on single-channel EEG data with the hypnogram on top.
* `nonlinear_features <https://github.com/raphaelvallat/yasa/blob/master/notebooks/11_nonlinear_features.ipynb>`_: calculate non-linear EEG features on 30-seconds epochs and perform a naive sleep stage classification.
* `SO-sigma_coupling <https://github.com/raphaelvallat/yasa/blob/master/notebooks/12_SO-sigma_coupling.ipynb>`_: slow-oscillations/spindles phase-amplitude coupling and data-driven comodulogram.
* `topoplot <https://github.com/raphaelvallat/yasa/blob/master/notebooks/15_topoplot.ipynb>`_: topoplot.

**********

Gallery
~~~~~~~

Below some plots demonstrating the functionalities of YASA. To reproduce these, check out the `tutorial (Jupyter notebooks) <https://github.com/raphaelvallat/yasa/tree/master/notebooks>`_.

.. figure::  /pictures/gallery.png
  :align:   center

  *The top plot show an overlay of the detected spindles on real EEG data. The middle left panel shows a time-frequency representation of the whole-night recording (spectrogram), plotted with the hypnogram (sleep stages) on top. The middle right panel shows the sleep stage probability transition matrix, calculated across the entire night. The bottom row shows, from left to right: a topographic plot, the average template of all detected slow-waves across the entire night stratified by channels, and a phase-amplitude coupling comodulogram.*

**********

Development
~~~~~~~~~~~

YASA was created and is maintained by `Raphael Vallat <https://raphaelvallat.com>`_, a postdoctoral researcher in `Matthew Walker's lab <https://www.humansleepscience.com/>`_ at UC Berkeley. Contributions are more than welcome so feel free to contact me, open an issue or submit a pull request!

To see the code or report a bug, please visit the `GitHub repository <https://github.com/raphaelvallat/yasa>`_.

Note that this program is provided with NO WARRANTY OF ANY KIND.

**********

Citation
~~~~~~~~

To cite YASA, please use the `eLife publication <https://elifesciences.org/articles/70092>`_:

* Vallat, Raphael, and Matthew P. Walker. "An open-source, high-performance tool for automated sleep staging." Elife 10 (2021). doi: https://doi.org/10.7554/eLife.70092

|
