.. _changelog:

What's new
##########

v0.2.0 (dev)
------------

**Bugfixes**

a. Sleep efficiency in the :py:func:`yasa.sleep_statistics` is now calculated using time in bed (TIB) as the denominator instead of sleep period time (SPT), in agreement with the AASM guidelines. The old way of computing the efficiency (TST / SPT) has now been renamed Sleep Maintenance Efficiency (SME).

**New functions**

a. Added :py:func:`bandpower_from_psd_ndarray` to calculate band power from a multi-dimensional PSD. This is a Numpy-only implementation and this function will return a np.array and not a pandas DataFrame.
This function is useful if you need to calculate the bandpower from a 3-D PSD array, e.g. of shape (n_chan, n_epochs, n_freqs).

**Enhancements**

a. :py:func:`yasa.sleep_statistics` now also returns the sleep onset latency, i.e. the latency to the first epoch of any sleep.
b. Added the `bandpass` argument to :py:func:`yasa.bandpower` to apply a FIR bandpass filter using the lowest and highest frequencies defined in `bands`. This is useful if you work with absolute power and want to remove contributions from frequency bands of non-interests.
c. The :py:func:`yasa.bandpower_from_psd` now always return the total absolute physical power (`TotalAbsPow`) of the signal, in units of uV^2 / Hz. This allows to quickly calculate the absolute bandpower from the relative bandpower.
d. Added sigma (12-16Hz) to the default frequency bands (`bands`) in :py:func:`yasa.bandpower` and :py:func:`yasa.bandpower_from_psd`.
e. Added an section in the `06_sw_detection.ipynb <https://github.com/raphaelvallat/yasa/blob/master/notebooks/06_sw_detection.ipynb>`_ notebooks on how to use relative amplitude thresholds (e.g. z-scores) instead of absolute thresholds in physical units.

**Dependencies**

a. Removed deprecated ``behavior`` argument to avoid warning when calling :py:class:`sklearn.ensemble.IsolationForest`.

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
