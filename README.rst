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

=======  =====  ==========  ===========  =====  ==========  ==========  ===========  ==============  ==========
  Start    End    Duration    Amplitude    RMS    AbsPower    RelPower    Frequency    Oscillations    Symmetry
=======  =====  ==========  ===========  =====  ==========  ==========  ===========  ==============  ==========
   3.35   4.03        0.68        81.49  20.17        2.74        0.54        12.82               8        0.67
  13.16  13.86        0.70        99.32  24.19        2.84        0.31        12.23               8        0.35
=======  =====  ==========  ===========  =====  ==========  ==========  ===========  ==============  ==========

.. figure::  notebooks/detection.png
   :align:   center

Development
===========

YASA was created and is maintained by `Raphael Vallat <https://raphaelvallat.com>`_. Contributions are more than welcome so feel free to contact me, open an issue or submit a pull request!

To see the code or report a bug, please visit the `GitHub repository <https://github.com/raphaelvallat/yasa>`_.

Note that this program is provided with NO WARRANTY OF ANY KIND.
