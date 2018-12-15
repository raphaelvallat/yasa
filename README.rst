.. -*- mode: rst -*-

.. .. image:: https://img.shields.io/github/license/raphaelvallat/yasa.svg
..   :target: https://github.com/raphaelvallat/yasa/blob/master/LICENSE
..
.. .. image:: https://travis-ci.org/raphaelvallat/yasa.svg?branch=master
..     :target: https://travis-ci.org/raphaelvallat/yasa

.. ----------------

YASA
====

YASA (*Yet Another Spindle Algorithm*) is a fast and data-agnostic sleep spindles detection algorithm written in Python 3.

The algorithm behind YASA is largely inspired by the method described in:

Lacourse, K., Delfrate, J., Beaudry, J., Peppard, P., Warby, S.C., 2018. A sleep spindle detection algorithm that emulates human expert spindle scoring. *J. Neurosci. Methods*. https://doi.org/10.1016/j.jneumeth.2018.08.014

Installation
~~~~~~~~~~~~

.. code-block:: shell

  git clone https://github.com/raphaelvallat/yasa.git yasa/
  cd yasa/
  pip install -r requirements.txt
  python setup.py develop

**Dependencies**

- numpy>=1.14
- scipy>=1.1.0
- pandas>=0.23,
- mne>=0.17.0
- numba>=0.39.0

**Note**

Several functions of YASA are written using `Numba <http://numba.pydata.org/>`_, a just-in-time compiler for Python. This allows to greatly speed up the computation time (typically a few seconds for a full night recording).

Examples
========

Please refer to `notebooks/spindles_detection.ipynb <notebooks/spindles_detection.ipynb>`_ for an example on how to use YASA as well as a step-by-step description of the algorithm.

**Typical use**

.. code-block:: python

  import yasa
  yasa.spindles_detect(data, sf)

.. table:: Output
   :widths: auto

=======  =====  ==========  ===========  =====  ==========  ==========  ===========  ==============  ==========  ============
  Start    End    Duration    Amplitude    RMS    AbsPower    RelPower    Frequency    Oscillations    Symmetry  Confidence
=======  =====  ==========  ===========  =====  ==========  ==========  ===========  ==============  ==========  ============
   3.37   4.00        0.63        82.14  20.56        2.43        0.53        12.72               8        0.69  high
  13.17  13.82        0.65       102.21  25.40        2.58        0.21        12.13               7        0.36  medium
=======  =====  ==========  ===========  =====  ==========  ==========  ===========  ==============  ==========  ============

.. figure::  notebooks/detection.png
   :align:   center

Development
===========

YASA was created and is maintained by `Raphael Vallat <https://raphaelvallat.com>`_. Contributions are more than welcome so feel free to contact me, open an issue or submit a pull request!

To see the code or report a bug, please visit the `GitHub repository <https://github.com/raphaelvallat/yasa>`_.

Note that this program is provided with NO WARRANTY OF ANY KIND.
