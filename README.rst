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
* Hypnogram analysis: sleep statistics, stage transitions, visualization, and manipulation.

For more details, try the `tutorials <https://yasa-sleep.org/tutorials/index.html>`_ or read the `FAQ <https://yasa-sleep.org/faq.html>`_.

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

The best starting point is the `tutorials <https://yasa-sleep.org/tutorials/index.html>`_ section
of the documentation, which includes a quickstart guide and step-by-step walkthroughs of the most
common workflows.

Additional worked examples are available as `Jupyter notebooks on GitHub
<https://github.com/raphaelvallat/yasa/tree/master/notebooks>`_. Note that some notebooks may
not reflect the latest API.

Gallery
~~~~~~~

Below some plots demonstrating the functionalities of YASA. For step-by-step examples, see the
`tutorials <https://yasa-sleep.org/tutorials/index.html>`_ or the `Jupyter notebooks
<https://github.com/raphaelvallat/yasa/tree/master/notebooks>`_.

.. figure:: https://raw.githubusercontent.com/raphaelvallat/yasa/refs/tags/v0.6.5/docs/pictures/gallery.png
  :align: center

  *The top plot show an overlay of the detected spindles on real EEG data. The middle left panel shows a time-frequency representation of the whole-night recording (spectrogram), plotted with the hypnogram (sleep stages) on top. The middle right panel shows the sleep stage probability transition matrix, calculated across the entire night. The bottom row shows, from left to right: a topographic plot, the average template of all detected slow-waves across the entire night stratified by channels, and a phase-amplitude coupling comodulogram.*

Development
~~~~~~~~~~~

YASA was created and is maintained by `Raphael Vallat <https://raphaelvallat.com>`_. Contributions are more than welcome! See the `contributing guide <https://yasa-sleep.org/contributing.html>`_ for guidelines.

To see the code or report a bug, please visit the `GitHub repository <https://github.com/raphaelvallat/yasa>`_.

Note that this program is provided with NO WARRANTY OF ANY KIND.

Citation
~~~~~~~~

To cite YASA, please use the `eLife publication <https://elifesciences.org/articles/70092>`_:

* Vallat, Raphael, and Matthew P. Walker. "An open-source, high-performance tool for automated sleep staging." Elife 10 (2021). doi: https://doi.org/10.7554/eLife.70092


