.. _changelog:

What's new
##########

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

v0.1.8 (October 2019)
---------------------

a. Added :py:func:`yasa.plot_spectrogram` function.
b. Added `lspopt <https://github.com/hbldh/lspopt>`_ in the dependencies.
c. YASA now requires `MNE <https://mne.tools/stable/index.html>`_>0.19.
d. Added a notebook on non-linear features.

v0.1.7 (August 2019)
--------------------

a. Added :py:func:`yasa.sliding_window` function.
b. Added :py:func:`yasa.irasa` function.
c. Reorganized code into several sub-files for readability (internal changes with no effect on user experience).

v0.1.6 (August 2019)
--------------------

a. Added bandpower function
b. One can now directly pass a raw MNE object in several multi-channel functions of YASA, instead of manually passing data, sf, and ch_names. YASA will automatically convert MNE data from Volts to uV, and extract the sampling frequency and channel names. Examples of this can be found in the Jupyter notebooks examples.

v0.1.5 (August 2019)
--------------------

a. Added REM detection (rem_detect) on LOC and ROC EOG channels + example notebook
b. Added yasa/hypno.py file, with several functions to load and upsample sleep stage vector (hypnogram).
c. Added yasa/spectral.py file, which includes the bandpower_from_psd function to calculate the single or multi-channel spectral power in specified bands from a pre-computed PSD (see example notebook at notebooks/10_bandpower.ipynb)

v0.1.4 (May 2019)
-----------------

a. Added get_sync_sw function to get the synchronized timings of landmarks timepoints in slow-wave sleep. This can be used in combination with seaborn.lineplot to plot an average template of the detected slow-wave, per channel.

v0.1.3 (March 2019)
-------------------

a. Added slow-waves detection for single and multi channel
b. Added include argument to select which values of hypno should be used as a mask.
c. New examples notebooks + changes in README
d. Minor improvements in performance (e.g. faster detrending)
e. Added html API (/html)
f. Travis and AppVeyor test for Python 3.5, 3.6 and 3.7


v0.1.2 (February 2019)
----------------------

a. Added support for multi-channel detection via spindles_detect_multi function.
b. Added support for hypnogram mask
c. Added several notebook examples
d. Changed some default parameters to optimize behavior

v0.1.1 (January 2019)
----------------------

a. Added post-processing Isolation Forest
b. Updated Readme and added support with Visbrain
c. Added Cz full night in notebooks/

v0.1 (December 2018)
--------------------

Initial release of YASA: basic spindles detection.
