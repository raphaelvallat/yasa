.. _changelog:

What's new
##########

.. contents:: Table of Contents
    :depth: 2

----------------------------------------------------------------------------------------

v0.6.0 (February 2022)
----------------------

This is a MAJOR release with several API-breaking changes, new functions, bugfixes and a new section in the documentation.

**Documentation**

* Added a :ref:`quickstart` section to illustrate the main functions of YASA. Make sure to check it out!

**Plotting**

* Added the :py:func:`yasa.plot_hypnogram` function to plot an hypnogram.

**Slow oscillations—sigma coupling**

IMPORTANT — The default behavior of ``coupling=True`` in :py:func:`yasa.sw_detect` has been changed:

* YASA now uses a ± 1 second window around the negative peak of the slow-waves (2 sec total) to calculate the coupling, instead of a ± 2 sec window. Overall, this tends to increase the ndPAC values because of the higher temporal specificity. To keep a 4-sec window, use ``coupling_params['time'] = 2``.

* We've enabled the statistical thresholding in the ndPAC calculation. Practically, this means that events with a weak/unreliable coupling are assigned an ndPAC value of zero. Statistical thresholding can be disabled with ``coupling_params['p'] = None``.

.. warning:: Because of these changes, the coupling values are therefore not comparable with previous versions of YASA. Please make sure to re-run your analyses with the new default parameters.

**Events detection**

* The :py:func:`yasa.sw_detect` function now uses more conservative amplitude thresholds: the max PTP amplitude has been reduced from 500 to 350 uV, the max negative amplitude has been reduced from 300 to 200 uV, and the max positive amplitude has been reduced from 200 to 150 uV.

* Added :py:meth:`yasa.SWResults.find_cooccurring_spindles` to detect whether each slow-wave co-occurr with a sleep spindle.

* Added the ``as_dataframe`` parameter in :py:meth:`yasa.SWResults.get_sync_events` and :py:meth:`yasa.SpindlesResults.get_sync_events`. If set to False, YASA will return the peak-locked data as a list (n_channels) of numpy arrays (n_events, n_times). This facilitates any analyses that requires access to event-locked data (e.g. time-frequency plot, or comodulogram).

* Added the ``mask`` parameter in :py:meth:`yasa.SWResults.summary`, :py:meth:`yasa.SWResults.get_sync_events`, and :py:meth:`yasa.SWResults.plot_average`. This allows users to only include selected events in the summary or plots (e.g. the slow-waves with the largest peak-to-peak amplitude, or strongest coupling).

* Added the ``mask`` parameter in :py:meth:`yasa.SpindlesResults.summary`, :py:meth:`yasa.SpindlesResults.get_sync_events`, and :py:meth:`yasa.SpindlesResults.plot_average`. This allows users to only include selected events in the summary or plots (e.g. the spindles with the largest amplitude).

* Added the ``mask`` parameter in :py:meth:`yasa.REMResults.summary`, :py:meth:`yasa.REMResults.get_sync_events`, and :py:meth:`yasa.REMResults.plot_average`.

**Others**

* :py:func:`yasa.irasa` now informs about the maximum resampled fitting range, and raises a warning if parameters/frequencies are ill-specified. See `PR42 <https://github.com/raphaelvallat/yasa/pull/42>`_ and associated paper: https://doi.org/10.1101/2021.10.15.464483

* Added a ``verbose`` parameter to :py:func:`yasa.hypno_upsample_to_data` and :py:func:`yasa.irasa`.

* Remove Travis CI

* Remove CI testing for Python 3.6

----------------------------------------------------------------------------------------

v0.5.1 (August 2021)
--------------------

This is a bugfix release. The latest pre-trained classifiers for :py:class:`yasa.SleepStaging` were accidentally missing from the previous release. They have now been included in this release.

v0.5.0 (August 2021)
--------------------

This is a major release with an important bugfix for the slow-waves detection as well as API-breaking changes in the automatic sleep staging module. We recommend all users to upgrade to this version with `pip install --upgrade yasa`.

**Slow-waves detection**

We have fixed a critical bug in :py:func:`yasa.sw_detect` in which the detection could keep slow-waves with invalid duration (e.g. several tens of seconds). We have now added extra safety checks to make sure that the total duration of the slow-waves does not exceed the maximum duration allowed by the ``dur_neg`` and ``dur_pos`` parameters (default = 2.5 seconds).

.. warning::
  Please make sure to double-check any results obtained with :py:func:`yasa.sw_detect`.

**Sleep staging**

Recently, we have published a `preprint article <https://www.biorxiv.org/content/10.1101/2021.05.28.446165v1>`_ describing YASA's sleep staging algorithm and its validation across hundreds of polysomnography recordings. In July 2021, we have received comments from three reviewers, which have led us to implement several changes to the sleep staging algorithm.
The most significant change is that the time lengths of the rolling windows have been updated from 5.5 minutes centered / 5 minutes past to 7.5 minutes centered / 2 min past, leading to slight improvements in accuracy. Furthermore, we have also updated the training database and the parameters of the LightGBM classifier.
Unfortunately, these changes mean that the new version of the algorithm is no longer compatible with the previous version (0.4.0 or 0.4.1). Therefore, if you're running a longitudinal study with YASA's sleep staging, we either recommend to keep the previous version of YASA, or to update to the new version and reprocess all your nights with the new algorithm for consistency.

**Sleep statistics**

Artefact and Unscored epochs are now excluded from the calculation of the total sleep time (TST) in :py:func:`yasa.sleep_statistics`. Previously, YASA calculated TST as SPT - WASO, thus including Art and Uns. TST is now calculated as the sum of all REM and NREM sleep in SPT.

**New FAQ**

The online documentation now has a brand new FAQ section! Make sure to check it out at https://raphaelvallat.com/yasa/build/html/faq.html

**New function: coincidence matrix**

We have added the :py:meth:`yasa.SpindlesResults.get_coincidence_matrix` and :py:meth:`yasa.SWResults.get_coincidence_matrix` methods to calculate the (scaled) coincidence matrix.
The coincidence matrix gives, for each pair of channel, the number of samples that were marked as an event (spindles or slow-waves) in both channels. In other words, it gives an indication of whether events (spindles or slow-waves) are co-occuring for any pair of channel.
The scaled version of the coincidence matrix can then be used to define functional networks or quickly find outlier channels.

**Minor enhancements**

a. Minor speed improvements in :py:class:`yasa.SleepStaging`.
b. Updated dependency to pyRiemann>=0.2.7, which solves the version conflict with scikit-learn (see `issue 33 <https://github.com/raphaelvallat/yasa/issues/33>`_).
c. flake8 requirements for max line length has been changed from 80 to 100 characters.

----------------------------------------------------------------------------------------

v0.4.1 (March 2021)
-------------------

**New functions**

a. Added :py:func:`yasa.topoplot`, a wrapper around :py:func:`mne.viz.plot_topomap`. See `15_topoplot.ipynb <https://github.com/raphaelvallat/yasa/blob/master/notebooks/15_topoplot.ipynb>`_

**Enhancements**

a. The default frequency range for slow-waves in :py:func:`yasa.sw_detect` is now 0.3-1.5 Hz instead of 0.3-2 Hz. Indeed, most slow-waves have a frequency below 1Hz. This may result in slightly different coupling values when ``coupling=True`` so make sure to homogenize your slow-waves detection pipeline across all nights in your dataset.
b. :py:func:`yasa.trimbothstd` now handles missing values in input array.
c. :py:func:`yasa.bandpower_from_psd` and :py:func:`yasa.bandpower_from_psd_ndarray` now print a warning if the PSD contains negative values. See `issue 29 <https://github.com/raphaelvallat/yasa/issues/29>`_.
d. Upon loading, YASA will now use the `outdated <https://github.com/alexmojaki/outdated>`_ package to check and warn the user if a newer stable version is available.
e. YASA now uses the `antropy <https://github.com/raphaelvallat/antropy>`_ package to calculate non-linear features in the automatic sleep staging module. Previously, YASA was using `EntroPy <https://github.com/raphaelvallat/entropy>`_, which could not be installed using pip.

----------------------------------------------------------------------------------------

v0.4.0 (November 2020)
----------------------

This is a major release with several new functions, the biggest of which is the addition of an **automatic sleep staging module** (:py:class:`yasa.SleepStaging`). This means that YASA can now automatically score the sleep stages of your raw EEG data. The classifier was trained and validated on more than 3000 nights from the `National Sleep Research Resource (NSRR) <https://sleepdata.org/>`_ website.

Briefly, the algorithm works by calculating a set of features for each 30-sec epochs from a central EEG channel (required), as well as an EOG channel (optional) and an EMG channel (optional). For best performance, users can also specify the age and the sex of the participants. Pre-trained classifiers are already included in YASA. The automatic sleep staging algorithm requires the `LightGBM <https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html>`_ and `antropy <https://github.com/raphaelvallat/antropy>`_ package.

**Other changes**

a. :py:meth:`yasa.SpindlesResults` and :py:meth:`yasa.SWResults` now have a ``plot_detection`` method which allows to interactively display the raw data with an overlay of the detected spindles. For now, this only works with Jupyter and it requires the `ipywidgets <https://ipywidgets.readthedocs.io/en/latest/user_install.html>`_ package.
b. Added ``hue`` input parameter to :py:meth:`yasa.SpindlesResults.plot_average`, :py:meth:`yasa.SWResults.plot_average` to allow plotting by stage.
c. The ``get_sync_events()`` method now also returns the sleep stage when available.
d. The :py:func:`yasa.sw_detect` now also returns the timestamp of the sigma peak in the SW-through-locked 4-seconds epochs. The timestamp is expressed in seconds from the beginning of the recording and can be found in the ``SigmaPeak`` column.

**Dependencies**

a. Switch to latest version of `TensorPAC <https://etiennecmb.github.io/tensorpac/index.html>`_.
b. Added `ipywidgets <https://ipywidgets.readthedocs.io/en/latest/user_install.html>`_, `LightGBM <https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html>`_ and `entropy <https://github.com/raphaelvallat/entropy>`_ to dependencies.

----------------------------------------------------------------------------------------

v0.3.0 (May 2020)
-----------------

This is a major release with several API-breaking changes in the spindles, slow-waves and REMs detection.

First, the :py:func:`yasa.spindles_detect_multi` and :py:func:`yasa.sw_detect_multi` have been removed. Instead, the :py:func:`yasa.spindles_detect` and :py:func:`yasa.sw_detect` functions can now handle both single and multi-channel data.

Second, I was getting some feedback that it was difficult to get summary statistics from the detection dataframe. For instance, how can you get the average duration of the detected spindles, per channel and/or per stage? Similarly, how can you get the slow-waves count and density per stage and channel? To address these issues, I've now modified the output of the :py:func:`yasa.spindles_detect`, :py:func:`yasa.sw_detect` and :py:func:`yasa.rem_detect` functions, which is now a class (= object) and not a simple Pandas DataFrame. The advantage is that the new output allows you to quickly get the raw data or summary statistics grouped by channel and/or sleep stage using the ``.summary()`` method.

>>> sp = yasa.spindles_detect(...)
>>> sp.summary()  # Returns the full detection dataframe
>>> sp.summary(grp_chan=True, grp_stage=True, aggfunc='mean')

Similarly, the :py:func:`yasa.get_bool_vector` and :py:func:`yasa.get_sync_events` functions are now directly implemented into the output, i.e.

>>> sw = yasa.sw_detect(...)
>>> sw.summary()
>>> sw.get_mask()
>>> sw.get_sync_events(center='NegPeak', time_before=0.4, time_after=0.8)

One can also quickly plot an average "template" of all the detected events:

>>> sw.plot_average(center="NegPeak", time_before=0.4, time_after=0.8)

For more details, please refer to the documentation of :py:meth:`yasa.SpindlesResults`, :py:meth:`yasa.SWResults` and :py:meth:`yasa.REMResults`.

.. important::
  This is an experimental feature, and it's likely that these functions will be modified, renamed, or even deprecated in future releases based on feedbacks from users. Please make sure to let me know what you think about the new output of the detection functions!

**Other changes**

a. The ``coupling`` argument has been removed from the :py:func:`yasa.spindles_detect` function. Instead, slow-oscillations / sigma coupling can only be calculated from the slow-waves detection, which is 1) the most standard way, 2) better because PAC assumptions require a strong oscillatory component in the lower frequency range (slow-oscillations). This also avoids unecessary confusion between spindles-derived coupling and slow-waves-derived coupling. For more details, refer to the Jupyter notebooks.
b. Downsampling of data in detection functions has been removed. In other words, YASA will no longer downsample the data to 100 / 128 Hz before applying the events detection. If the detection is too slow, we recommend that you manually downsample your data before applying the detection. See for example :py:func:`mne.filter.resample`.
c. :py:func:`yasa.trimbothstd` can now work with multi-dimensional arrays. The trimmed standard deviation will always be calculated on the last axis of the array.
d. Filtering and Hilbert transform are now applied at once on all channels (instead of looping across individual channels) in the :py:func:`yasa.spindles_detect` and :py:func:`yasa.sw_detect` functions. This should lead to some improvements in computation time.

----------------------------------------------------------------------------------------

v0.2.0 (April 2020)
-------------------

This is a major release with several new functions, bugfixes and miscellaneous enhancements in existing functions.

**Bugfixes**

a. Sleep efficiency in the :py:func:`yasa.sleep_statistics` is now calculated using time in bed (TIB) as the denominator instead of sleep period time (SPT), in agreement with the AASM guidelines. The old way of computing the efficiency (TST / SPT) has now been renamed Sleep Maintenance Efficiency (SME).
b. The :py:func:`yasa.sliding_window` now always return an array of shape (n_epochs, ..., n_samples), i.e. the epochs are now always the first dimension of the epoched array. This is consistent with MNE default shape of :py:class:`mne.Epochs` objects.

**New functions**

a. Added :py:func:`yasa.art_detect` to automatically detect artefacts on single or multi-channel EEG data.
b. Added :py:func:`yasa.bandpower_from_psd_ndarray` to calculate band power from a multi-dimensional PSD. This is a NumPy-only implementation and this function will return a np.array and not a pandas DataFrame. This function is useful if you need to calculate the bandpower from a 3-D PSD array, e.g. of shape *(n_epochs, n_chan, n_freqs)*.
c. Added :py:func:`yasa.get_centered_indices` to extract indices in data centered around specific events or peaks.
d. Added :py:func:`yasa.load_profusion_hypno` to load a Compumedics Profusion hypnogram (.xml), as found on the `National Sleep Research Resource (NSRR) <https://sleepdata.org/>`_ website.

**Enhancements**

a. :py:func:`yasa.sleep_statistics` now also returns the sleep onset latency, i.e. the latency to the first epoch of any sleep.
b. Added the `bandpass` argument to :py:func:`yasa.bandpower` to apply a FIR bandpass filter using the lowest and highest frequencies defined in `bands`. This is useful if you work with absolute power and want to remove contributions from frequency bands of non-interests.
c. The :py:func:`yasa.bandpower_from_psd` now always return the total absolute physical power (`TotalAbsPow`) of the signal, in units of uV^2 / Hz. This allows to quickly calculate the absolute bandpower from the relative bandpower.
d. Added sigma (12-16Hz) to the default frequency bands (`bands`) in :py:func:`yasa.bandpower` and :py:func:`yasa.bandpower_from_psd`.
e. Added the ``coupling`` and ``freq_sp`` keyword-arguments to the :py:func:`yasa.sw_detect` function. If ``coupling=True``, the function will return the phase of the slow-waves (in radians) at the most prominent peak of sigma-filtered band (``PhaseAtSigmaPeak``), as well as the normalized mean vector length (``ndPAC``).
f. Added an section in the `06_sw_detection.ipynb <https://github.com/raphaelvallat/yasa/blob/master/notebooks/06_sw_detection.ipynb>`_ notebooks on how to use relative amplitude thresholds (e.g. z-scores or percentiles) instead of absolute thresholds in slow-waves detection.
g. The upper frequency band for :py:func:`yasa.sw_detect` has been changed from ``freq_sw=(0.3, 3.5)`` to ``freq_sw=(0.3, 2)`` Hz to comply with AASM guidelines.
h. ``Stage`` is no longer taken into account when finding outliers with :py:class:`sklearn.ensemble.IsolationForest` in :py:func:`yasa.spindles_detect`.
i. To be consistent with :py:func:`yasa.spindles_detect`, automatic outlier removal now requires at least 50 (instead of 100) detected events in :py:func:`yasa.sw_detect` and :py:func:`yasa.rem_detect`.
j. Added the ``verbose`` parameter to all detection functions.
k. Added -2 to the default hypnogram format to denote unscored data.

**Dependencies**

a. Removed deprecated ``behavior`` argument to avoid warning when calling :py:class:`sklearn.ensemble.IsolationForest`.
b. Added `TensorPAC <https://etiennecmb.github.io/tensorpac/index.html>`_ and `pyRiemann <https://pyriemann.readthedocs.io/en/latest/api.html>`_ to dependencies.
c. Updated dependencies version for MNE and scikit-learn.

----------------------------------------------------------------------------------------

v0.1.9 (February 2020)
----------------------

**New functions**

a. Added :py:func:`yasa.transition_matrix` to calculate the state-transition matrix of an hypnogram.
b. Added :py:func:`yasa.sleep_statistics` to extract the sleep statistics from an hypnogram.
c. Added the ``coupling`` and ``freq_so`` keyword-arguments to the :py:func:`yasa.spindles_detect` function. If ``coupling=True``, the function will also returns the phase of the slow-waves (in radians) at the most prominent peak of the spindles. This can be used to perform spindles-SO coupling, as explained in the new Jupyter notebooks on PAC and spindles-SO coupling.

**Enhancements**

a. It is now possible to disable one or two out of the three thresholds in the :py:func:`yasa.spindles_detect`. This allows the users to run a simpler detection (for example focusing exclusively on the moving root mean square signal).
b. The :py:func:`yasa.spindles_detect` now returns the timing (in seconds) of the most prominent peak of each spindles (``Peak``).
c. The yasa.get_sync_sw has been renamed to :py:func:`yasa.get_sync_events` and is now compatible with spindles detection. This can be used for instance to plot the peak-locked grand averaged spindle.

**Code testing**

a. Removed Travis and AppVeyor testing for Python 3.5.

----------------------------------------------------------------------------------------

v0.1.8 (October 2019)
---------------------

a. Added :py:func:`yasa.plot_spectrogram` function.
b. Added `lspopt <https://github.com/hbldh/lspopt>`_ in the dependencies.
c. YASA now requires `MNE <https://mne.tools/stable/index.html>`_>0.19.
d. Added a notebook on non-linear features.

----------------------------------------------------------------------------------------

v0.1.7 (August 2019)
--------------------

a. Added :py:func:`yasa.sliding_window` function.
b. Added :py:func:`yasa.irasa` function.
c. Reorganized code into several sub-files for readability (internal changes with no effect on user experience).

----------------------------------------------------------------------------------------

v0.1.6 (August 2019)
--------------------

a. Added bandpower function
b. One can now directly pass a raw MNE object in several multi-channel functions of YASA, instead of manually passing data, sf, and ch_names. YASA will automatically convert MNE data from Volts to uV, and extract the sampling frequency and channel names. Examples of this can be found in the Jupyter notebooks examples.

----------------------------------------------------------------------------------------

v0.1.5 (August 2019)
--------------------

a. Added REM detection (rem_detect) on LOC and ROC EOG channels + example notebook
b. Added yasa/hypno.py file, with several functions to load and upsample sleep stage vector (hypnogram).
c. Added yasa/spectral.py file, which includes the bandpower_from_psd function to calculate the single or multi-channel spectral power in specified bands from a pre-computed PSD (see example notebook at notebooks/10_bandpower.ipynb)

----------------------------------------------------------------------------------------

v0.1.4 (May 2019)
-----------------

a. Added get_sync_sw function to get the synchronized timings of landmarks timepoints in slow-wave sleep. This can be used in combination with seaborn.lineplot to plot an average template of the detected slow-wave, per channel.

----------------------------------------------------------------------------------------

v0.1.3 (March 2019)
-------------------

a. Added slow-waves detection for single and multi channel
b. Added include argument to select which values of hypno should be used as a mask.
c. New examples notebooks + changes in README
d. Minor improvements in performance (e.g. faster detrending)
e. Added html API (/html)
f. Travis and AppVeyor test for Python 3.5, 3.6 and 3.7

----------------------------------------------------------------------------------------

v0.1.2 (February 2019)
----------------------

a. Added support for multi-channel detection via spindles_detect_multi function.
b. Added support for hypnogram mask
c. Added several notebook examples
d. Changed some default parameters to optimize behavior

----------------------------------------------------------------------------------------

v0.1.1 (January 2019)
----------------------

a. Added post-processing Isolation Forest
b. Updated Readme and added support with Visbrain
c. Added Cz full night in notebooks/

----------------------------------------------------------------------------------------

v0.1 (December 2018)
--------------------

Initial release of YASA: basic spindles detection.
