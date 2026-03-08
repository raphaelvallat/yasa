.. -*- mode: rst -*-

|

.. image:: https://badge.fury.io/py/yasa.svg
    :target: https://badge.fury.io/py/yasa

.. image:: https://img.shields.io/github/license/raphaelvallat/yasa.svg
    :target: https://github.com/raphaelvallat/yasa/blob/master/LICENSE

.. image:: https://codecov.io/gh/raphaelvallat/yasa/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/raphaelvallat/yasa

.. image:: https://static.pepy.tech/badge/yasa
    :target: https://pepy.tech/projects/yasa

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
  :target: https://github.com/astral-sh/ruff
  :alt: Ruff

----------------

.. figure:: https://raw.githubusercontent.com/raphaelvallat/yasa/refs/tags/v0.6.5/docs/pictures/yasa_logo.png
   :align: center

**YASA** (*Yet Another Spindle Algorithm*) is a command-line sleep analysis toolbox in Python. The main functions of YASA are:

* Automatic sleep staging of polysomnography data (see `eLife article <https://elifesciences.org/articles/70092>`_).
* Event detection: sleep spindles, slow-waves and rapid eye movements, on single or multi-channel EEG data.
* Artefact rejection, on single or multi-channel EEG data.
* Spectral analyses: bandpower, phase-amplitude coupling, 1/f slope, and more!
* Hypnogram analysis: sleep statistics and stage transitions.
* Scorer agreement evaluation: epoch-by-epoch and sleep-statistics Bland–Altman agreement between two scorers.

For more details, try the `quickstart <https://yasa-sleep.org/quickstart.html>`_ or read the `FAQ <https://yasa-sleep.org/faq.html>`_.

----------------

Installation
~~~~~~~~~~~~

User installation
-----------------

YASA can be easily installed using pip, conda, or uv:

.. code-block:: shell

    uv pip install yasa
    pip install --upgrade yasa
    conda install -c conda-forge yasa

Some features require optional dependencies. Install them with extras:

.. code-block:: shell

    pip install "yasa[full]"    # all optional dependencies

Development
-----------

To build and install from source, clone this repository and install in editable mode with `uv <https://docs.astral.sh/uv/>`_

.. code-block:: shell

  git clone https://github.com/raphaelvallat/yasa.git
  cd yasa
  uv pip install --group=test --editable .

  # test the package
  pytest --verbose

For common questions about prerequisites, data formats, and how to load EEG data, see the `FAQ <https://yasa-sleep.org/faq.html>`_.

How do I get started with YASA?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to dive right in, you can simply go to the main `documentation <https://yasa-sleep.org/quickstart.html>`_ and try to apply YASA's functions on your own EEG data.
However, for most users, we strongly recommend that you first try running the examples Jupyter notebooks to get a sense of how YASA works and what it can do!
The notebooks also come with example datasets so they should work right out of the box as long as you've installed YASA first.
The notebooks and datasets can be found on `GitHub <https://github.com/raphaelvallat/yasa/tree/master/notebooks>`_ (make sure that you download the whole *notebooks/* folder). A short description of all notebooks is provided below:

**Automatic sleep staging**

* `automatic_staging <notebooks/14_automatic_sleep_staging.ipynb>`_: Automatic sleep staging of polysomnography data.

**Event detection**

* `spindles_detection <notebooks/01_spindles_detection.ipynb>`_: single-channel spindles detection and step-by-step description of the spindles detection algorithm.
* `spindles_detection_multi <notebooks/02_spindles_detection_multi.ipynb>`_: multi-channel spindles detection.
* `spindles_detection_NREM_only <notebooks/03_spindles_detection_NREM_only.ipynb>`_: how to limit the spindles detection on specific sleep stages using an hypnogram.
* `spindles_slow_fast <notebooks/04_spindles_slow_fast.ipynb>`_: slow versus fast spindles.
* `sw_detection <notebooks/05_sw_detection.ipynb>`_: single-channel slow-waves detection and step-by-step description of the slow-waves detection algorithm.
* `sw_detection_multi <notebooks/06_sw_detection_multi.ipynb>`_: multi-channel slow-waves detection.
* `artifact_rejection <notebooks/13_artifact_rejection.ipynb>`_: automatic artifact rejection on single and multi-channel EEG data.
* `REMs_detection <notebooks/07_REMs_detection.ipynb>`_: REMs detection.
* `run_visbrain <notebooks/run_visbrain.py>`_: interactive display of the detected spindles using the Visbrain visualization software in Python.

**Spectral analysis**

* `bandpower <notebooks/08_bandpower.ipynb>`_: calculate spectral band power, optionally averaged across channels and sleep stages.
* `IRASA <notebooks/09_IRASA.ipynb>`_: separate the aperiodic (= fractal = 1/f) components of the EEG power spectrum using the IRASA method.
* `spectrogram <notebooks/10_spectrogram.ipynb>`_: plot a multi-taper full-night spectrogram on single-channel EEG data with the hypnogram on top.
* `nonlinear_features <notebooks/11_nonlinear_features.ipynb>`_: calculate non-linear EEG features on 30-seconds epochs and perform a naive sleep stage classification.
* `SO-sigma_coupling <notebooks/12_SO-sigma_coupling.ipynb>`_: slow-oscillations/spindles phase-amplitude coupling and data-driven comodulogram.
* `EEG-HRV coupling <notebooks/16_EEG-HRV_coupling.ipynb>`_: overnight coupling between EEG bandpower and heart rate variability.
* `topoplot <notebooks/15_topoplot.ipynb>`_: topoplot.

Gallery
~~~~~~~

Below some plots demonstrating the functionalities of YASA. To reproduce these, check out the `tutorial (Jupyter notebooks) <https://github.com/raphaelvallat/yasa/tree/master/notebooks>`_.

.. figure:: https://raw.githubusercontent.com/raphaelvallat/yasa/refs/tags/v0.6.5/docs/pictures/gallery.png
  :align: center

  *The top plot show an overlay of the detected spindles on real EEG data. The middle left panel shows a time-frequency representation of the whole-night recording (spectrogram), plotted with the hypnogram (sleep stages) on top. The middle right panel shows the sleep stage probability transition matrix, calculated across the entire night. The bottom row shows, from left to right: a topographic plot, the average template of all detected slow-waves across the entire night stratified by channels, and a phase-amplitude coupling comodulogram.*

Development
~~~~~~~~~~~

YASA was created and is maintained by `Raphael Vallat <https://raphaelvallat.com>`_, a former postdoctoral researcher in `Matthew Walker's lab <https://www.humansleepscience.com/>`_ at UC Berkeley. Contributions are more than welcome so feel free to contact me, open an issue or submit a pull request!

To see the code or report a bug, please visit the `GitHub repository <https://github.com/raphaelvallat/yasa>`_.

Note that this program is provided with NO WARRANTY OF ANY KIND.

Citation
~~~~~~~~

To cite YASA, please use the `eLife publication <https://elifesciences.org/articles/70092>`_:

* Vallat, Raphael, and Matthew P. Walker. "An open-source, high-performance tool for automated sleep staging." Elife 10 (2021). doi: https://doi.org/10.7554/eLife.70092


