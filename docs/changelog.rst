.. _changelog:

What's new
##########

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
