.. _tutorial_migrate:

Migrating to the Hypnogram class
##################################

.. currentmodule:: yasa

YASA 0.7 introduced the :py:class:`Hypnogram` class as the new standard for working with sleep
hypnograms. Previous versions used plain integer or string NumPy arrays together with a set of
standalone functions. This page shows the most common old patterns alongside their new
equivalents, so you can update your code quickly.

.. note::

    The old helper functions (``hypno_upsample_to_data``, ``hypno_str_to_int``, etc.) still work
    in v0.7 but will be removed in v0.8. Migrating now will make your code cleaner and compatible
    with future releases.

.. contents:: Contents
    :local:
    :depth: 1

--------

Loading a hypnogram from a file
---------------------------------

Most pipelines start by loading an integer-encoded hypnogram from a plain-text or CSV file.

.. tab-set::

    .. tab-item:: Before (v0.6)

        .. code-block:: python

            import numpy as np
            import yasa

            # Load as integer array (0=W, 1=N1, 2=N2, 3=N3, 4=REM)
            hypno = np.loadtxt("hypnogram.txt").astype(int)

    .. tab-item:: After (v0.7+)

        .. code-block:: python

            import numpy as np
            import pandas as pd
            import yasa

            # From a .txt file
            hypno = np.loadtxt("hypnogram.txt").astype(int)
            hyp = yasa.Hypnogram.from_integers(hypno, freq="30s")

            # From a .csv file
            hypno = pd.read_csv("hypnogram.csv").squeeze().to_numpy()
            hyp = yasa.Hypnogram.from_integers(hypno, freq="30s")

--------

Loading a Profusion XML file
-----------------------------

.. tab-set::

    .. tab-item:: Before (v0.6)

        .. code-block:: python

            # Returns (integer array, sampling frequency)
            hypno, sf_hyp = yasa.load_profusion_hypno("hypnogram.xml")

    .. tab-item:: After (v0.7+)

        .. code-block:: python

            # Returns a Hypnogram object with freq set automatically from the XML
            hyp = yasa.Hypnogram.from_profusion("hypnogram.xml")

--------

Upsampling to match EEG data
------------------------------

Upsampling the hypnogram to the EEG sampling frequency was a required manual step before passing
it to detection or analysis functions.

.. tab-set::

    .. tab-item:: Before (v0.6)

        .. code-block:: python

            import mne
            import yasa

            raw = mne.io.read_raw_edf("recording.edf", preload=True)
            sf = raw.info["sfreq"]
            data = raw.get_data(units="uV")

            # Upsample the 30-second hypnogram to the EEG sampling frequency
            hypno_up = yasa.hypno_upsample_to_data(
                hypno=hypno, sf_hypno=1/30, data=data, sf_data=sf
            )

            # Then pass the upsampled array to detection functions
            sp = yasa.spindles_detect(data, sf=sf, hypno=hypno_up, include=(2, 3))

    .. tab-item:: After (v0.7+)

        .. code-block:: python

            import mne
            import yasa

            raw = mne.io.read_raw_edf("recording.edf", preload=True)
            hyp = yasa.Hypnogram.from_integers(hypno, freq="30s")

            # Pass the Hypnogram directly — upsampling is automatic
            sp = yasa.spindles_detect(raw, hypno=hyp, include=["N2", "N3"])

        If you still need the upsampled integer array (e.g. for a custom analysis):

        .. code-block:: python

            hypno_up = hyp.upsample_to_data(raw)

.. note::

    Detection functions now accept string stage labels in ``include`` (e.g.
    ``include=["N2", "N3"]``) when a :py:class:`Hypnogram` is passed. Integer labels
    (e.g. ``include=(2, 3)``) still work when a plain array is passed.

--------

Sleep statistics
-----------------

.. tab-set::

    .. tab-item:: Before (v0.6)

        .. code-block:: python

            # Standalone function — takes an integer array and sf_hyp
            stats = yasa.sleep_statistics(hypno, sf_hyp=1/30)

    .. tab-item:: After (v0.7+)

        .. code-block:: python

            hyp = yasa.Hypnogram.from_integers(hypno, freq="30s")
            stats = hyp.sleep_statistics()

--------

Stage-transition matrix
------------------------

.. tab-set::

    .. tab-item:: Before (v0.6)

        .. code-block:: python

            # Standalone function — integer input, integer labels in output
            counts, probs = yasa.transition_matrix(hypno)

    .. tab-item:: After (v0.7+)

        .. code-block:: python

            hyp = yasa.Hypnogram.from_integers(hypno, freq="30s")
            # Method — string labels in output (e.g. "WAKE", "N2")
            counts, probs = hyp.transition_matrix()

--------

Finding sleep periods
----------------------

.. tab-set::

    .. tab-item:: Before (v0.6)

        .. code-block:: python

            # Standalone function — takes sf_hypno as a float
            periods = yasa.hypno_find_periods(
                hypno, sf_hypno=1/30, threshold="5min"
            )

    .. tab-item:: After (v0.7+)

        .. code-block:: python

            hyp = yasa.Hypnogram.from_integers(hypno, freq="30s")
            periods = hyp.find_periods(threshold="5min")

--------

Plotting the hypnogram
-----------------------

.. tab-set::

    .. tab-item:: Before (v0.6)

        .. code-block:: python

            # plot_hypnogram accepted a plain integer or string array
            yasa.plot_hypnogram(hypno)

    .. tab-item:: After (v0.7+)

        .. code-block:: python

            hyp = yasa.Hypnogram.from_integers(hypno, freq="30s")
            hyp.plot_hypnogram()

--------

Converting between strings and integers
-----------------------------------------

The old ``hypno_str_to_int`` and ``hypno_int_to_str`` utility functions are replaced by
:py:class:`Hypnogram` methods.

.. tab-set::

    .. tab-item:: Before (v0.6)

        .. code-block:: python

            # Integer array to string array
            hypno_str = yasa.hypno_int_to_str(hypno_int)

            # String array to integer array
            hypno_int = yasa.hypno_str_to_int(hypno_str)

    .. tab-item:: After (v0.7+)

        .. code-block:: python

            # Integer array to Hypnogram (string storage)
            hyp = yasa.Hypnogram.from_integers(hypno_int, freq="30s")
            hypno_str = hyp.hypno.to_numpy()   # string NumPy array if needed

            # String array to Hypnogram
            hyp = yasa.Hypnogram(hypno_str, freq="30s")
            hypno_int = hyp.as_int().to_numpy()  # integer NumPy array if needed

--------

Automatic sleep staging
------------------------

:py:meth:`SleepStaging.predict` now returns a :py:class:`Hypnogram` instead of a NumPy array.

.. tab-set::

    .. tab-item:: Before (v0.6)

        .. code-block:: python

            sls = yasa.SleepStaging(raw, eeg_name="C3-A2")
            hypno_pred = sls.predict()   # returned a string NumPy array
            hypno_pred_int = yasa.hypno_str_to_int(hypno_pred)  # convert if needed

            stats = yasa.sleep_statistics(hypno_pred_int, sf_hyp=1/30)

    .. tab-item:: After (v0.7+)

        .. code-block:: python

            sls = yasa.SleepStaging(raw, eeg_name="C3-A2")
            hyp = sls.predict()   # returns a Hypnogram

            # Use methods directly on the object
            stats = hyp.sleep_statistics()
            hyp.plot_hypnogram()
            hyp.plot_hypnodensity()  # stage probabilities are included automatically

            # Recover arrays if needed for custom code
            hypno_str = hyp.hypno.to_numpy()
            hypno_int = hyp.as_int().to_numpy()

--------

Reference table
---------------

.. list-table::
    :widths: 45 55
    :header-rows: 1

    * - Old (v0.6)
      - New (v0.7+)
    * - ``np.loadtxt(...).astype(int)`` passed directly
      - :py:meth:`Hypnogram.from_integers`
    * - ``yasa.load_profusion_hypno(fname)``
      - :py:meth:`Hypnogram.from_profusion`
    * - ``yasa.hypno_upsample_to_data(hypno, sf_hypno, data, sf_data)``
      - :py:meth:`Hypnogram.upsample_to_data`
    * - ``yasa.sleep_statistics(hypno, sf_hyp)``
      - :py:meth:`Hypnogram.sleep_statistics`
    * - ``yasa.transition_matrix(hypno)``
      - :py:meth:`Hypnogram.transition_matrix`
    * - ``yasa.hypno_find_periods(hypno, sf_hypno, threshold)``
      - :py:meth:`Hypnogram.find_periods`
    * - ``yasa.plot_hypnogram(hypno_array)``
      - ``hyp.plot_hypnogram()`` or ``yasa.plot_hypnogram(hyp)``
    * - ``yasa.hypno_int_to_str(hypno_int)``
      - ``hyp.hypno.to_numpy()`` (after :py:meth:`Hypnogram.from_integers`)
    * - ``yasa.hypno_str_to_int(hypno_str)``
      - :py:meth:`Hypnogram.as_int`
    * - ``SleepStaging.predict()`` returned a string array
      - ``SleepStaging.predict()`` returns a :py:class:`Hypnogram`
    * - ``include=(2, 3)`` in detection functions
      - ``include=["N2", "N3"]`` when passing a :py:class:`Hypnogram`

--------

Next steps
----------

* :ref:`tutorial_hypnogram`: full tutorial on all :py:class:`Hypnogram` features.
* :ref:`api_ref`: complete API reference.
