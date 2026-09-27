"""Automatic sleep staging of polysomnography data."""

import datetime
import glob
import logging
import os
import re

import antropy as ant
import joblib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy.signal as sp_sig
from mne.filter import filter_data
from scipy.integrate import trapezoid
from sklearn.preprocessing import robust_scale

from .hypno import Hypnogram
from .others import sliding_window
from .spectral import bandpower_from_psd_ndarray

logger = logging.getLogger("yasa")


class SleepStaging:
    """
    Automatic sleep staging of polysomnography data.

    The automatic sleep staging requires the
    `LightGBM <https://lightgbm.readthedocs.io/>`_ and
    `antropy <https://github.com/raphaelvallat/antropy>`_ packages.

    .. versionadded:: 0.4.0

    Parameters
    ----------
    raw : :py:class:`mne.io.BaseRaw`
        An MNE Raw instance.
    eeg_name : str
        The name of the EEG channel in ``raw``. Preferentially a central
        electrode referenced either to the mastoids (C4-M1, C3-M2) or to the
        Fpz electrode (C4-Fpz).
    eog_name : str or None
        The name of the EOG channel in ``raw``. Preferentially,
        the left LOC channel referenced either to the mastoid (e.g. E1-M2)
        or Fpz. Can also be None.
    emg_name : str or None
        The name of the EMG channel in ``raw``. Preferentially a chin
        electrode. Can also be None.
    metadata : dict or None
        A dictionary of metadata (optional). Currently supported keys are:

        * ``'age'``: age of the participant, in years.
        * ``'male'``: sex of the participant (1 or True = male, 0 or
          False = female)

    Notes
    -----

    If you use the SleepStaging module in a publication, please cite the following publication:

    * Vallat, R., & Walker, M. P. (2021). An open-source, high-performance tool for automated
      sleep staging. Elife, 10. doi: https://doi.org/10.7554/eLife.70092

    We provide below some key points on the algorithm and its validation. For more details,
    we refer the reader to the peer-reviewed publication. If you have any questions,
    make sure to first check the
    `FAQ section <https://yasa-sleep.org/faq.html>`_ of the documentation.
    If you did not find the answer to your question, please feel free to open an issue on GitHub.

    **1. Features extraction**

    For each 30-seconds epoch and each channel, the following features are calculated:

    * Standard deviation
    * Interquartile range
    * Skewness and kurtosis
    * Number of zero crossings
    * Hjorth mobility and complexity
    * Absolute total power in the 0.4-30 Hz band.
    * Relative power in the main frequency bands (for EEG and EOG only)
    * Power ratios (e.g. delta / beta)
    * Permutation entropy
    * Higuchi and Petrosian fractal dimension

    In addition, the algorithm also calculates a smoothed and normalized version of these features.
    Specifically, a 7.5 min centered triangular-weighted rolling average and a 2 min past rolling
    average are applied. The resulting smoothed features are then normalized using a robust
    z-score. The algorithm assumes data are in micro-Volts.

    .. important:: Do NOT transform (e.g. z-score) or filter the signal before running
        the sleep staging algorithm.

    The data are automatically downsampled to 100 Hz for faster computation.

    The data are automatically converted to micro-Volts if necessary.

    **2. Sleep stages prediction**

    YASA comes with a default set of pre-trained classifiers, which were trained and validated
    on ~3000 nights from the `National Sleep Research Resource <https://sleepdata.org/>`_.
    These nights involved participants from a wide age range, of different ethnicities, gender,
    and health status. The default classifiers should therefore works reasonably well on most data.

    The code that was used to train the classifiers can be found on GitHub at:
    https://github.com/raphaelvallat/yasa_classifier

    In addition with the predicted sleep stages, YASA can also return the predicted probabilities
    of each sleep stage at each epoch. This can be used to derive a confidence score at each epoch.

    .. important:: The predictions should ALWAYS be double-check by a trained visual scorer,
        especially for epochs with low confidence. A full inspection should be performed in the
        following cases:

        * Nap data, because the classifiers were exclusively trained on full-night recordings.
        * Participants with sleep disorders.
        * Sub-optimal PSG system and/or referencing

    .. warning:: N1 sleep is the sleep stage with the lowest detection accuracy. This is expected
        because N1 is also the stage with the lowest human inter-rater agreement. Be very
        careful for potential misclassification of N1 sleep (e.g. scored as Wake or N2) when
        inspecting the predicted sleep stages.

    References
    ----------
    If you use YASA's default classifiers, these are the main references for
    the `National Sleep Research Resource <https://sleepdata.org/>`_:

    * Dean, Dennis A., et al. "Scaling up scientific discovery in sleep medicine: the National
      Sleep Research Resource." Sleep 39.5 (2016): 1151-1164.

    * Zhang, Guo-Qiang, et al. "The National Sleep Research Resource: towards a sleep data
      commons." Journal of the American Medical Informatics Association 25.10 (2018): 1351-1358.

    Examples
    --------
    For a concrete example, please refer to the example Jupyter notebook:
    https://github.com/raphaelvallat/yasa/blob/master/notebooks/14_automatic_sleep_staging.ipynb

    >>> import mne
    >>> import yasa
    >>> # Load an EDF file using MNE
    >>> raw = mne.io.read_raw_edf("myfile.edf", preload=True)
    >>> # Initialize the sleep staging instance
    >>> sls = yasa.SleepStaging(
    ...     raw,
    ...     eeg_name="C4-M1",
    ...     eog_name="LOC-M2",
    ...     emg_name="EMG1-EMG2",
    ...     metadata=dict(age=29, male=True),
    ... )
    >>> # Print some basic info
    >>> sls
    >>> # Get the predicted sleep stages
    >>> hyp = sls.predict()
    >>> hyp.hypno
    >>> # Get the predicted probabilities
    >>> hyp.proba
    >>> # Get the confidence
    >>> confidence = hyp.proba.max(axis=1)
    >>> # Plot the predicted probabilities
    >>> sls.plot_predict_proba()

    The sleep scores can then be manually edited in an external graphical user interface
    (e.g. EDFBrowser), as described in the
    `FAQ <https://yasa-sleep.org/faq.html>`_.
    """

    def __init__(self, raw, eeg_name, *, eog_name=None, emg_name=None, metadata=None):
        # Type check
        assert isinstance(eeg_name, str), "`eeg_name` must be a string."
        assert isinstance(eog_name, (str, type(None))), "`eog_name` must be a string or None."
        assert isinstance(emg_name, (str, type(None))), "`emg_name` must be a string or None."
        assert isinstance(metadata, (dict, type(None))), "`metadata` must be a dict or None."

        # Validate metadata. An empty dict is equivalent to no metadata. We work on a copy so that
        # the caller's dictionary is not modified.
        metadata = dict(metadata) if metadata else None
        if metadata is not None:
            if "age" in metadata:
                assert 0 < metadata["age"] < 120, "age must be between 0 and 120."
            if "male" in metadata:
                metadata["male"] = int(metadata["male"])
                assert metadata["male"] in [0, 1], "male must be 0 or 1."

        # Validate Raw instance and load data
        assert isinstance(raw, mne.io.BaseRaw), "`raw` must be a MNE Raw object."
        sf = raw.info["sfreq"]
        assert sf > 80, "Sampling frequency must be at least 80 Hz."
        ch_names, ch_types = [], []
        for c, t in zip([eeg_name, eog_name, emg_name], ["eeg", "eog", "emg"]):
            if c is not None:
                assert c in raw.ch_names, "%s does not exist" % c
                ch_names.append(c)
                ch_types.append(t)
        # Keep only the selected channels, in that order. Building a new Raw from only these
        # channels avoids copying every channel of `raw` (e.g. a full 64-channel PSG).
        picks = mne.pick_channels(raw.ch_names, ch_names, ordered=True)
        raw_pick = mne.io.RawArray(
            raw.get_data(picks=picks), mne.pick_info(raw.info, picks), verbose=False
        )

        # Downsample if sf != 100
        if sf != 100:
            raw_pick.resample(100, npad="auto")
            sf = raw_pick.info["sfreq"]

        # Get data and convert to microVolts
        data = raw_pick.get_data(units=dict(eeg="uV", emg="uV", eog="uV", ecg="uV"))

        # Extract duration of recording in minutes
        duration_minutes = data.shape[1] / sf / 60
        if data.shape[1] < 30 * sf:
            raise ValueError("Insufficient data. At least one 30-seconds epoch is required.")
        if duration_minutes < 5:
            msg = (
                "Insufficient data. A minimum of 5 minutes of data is recommended "
                "otherwise results may be unreliable."
            )
            logger.warning(msg)

        # Add to self
        self.sf = sf
        self.ch_names = ch_names
        self.ch_types = ch_types
        self.data = data
        self.metadata = metadata
        # Start time of the first sample of `raw`, as a UTC-aware datetime or None. `meas_date` is
        # the time of the first sample of the original recording and is not updated by
        # `raw.crop()`, so we need to add the time of the first sample (`raw.first_time`).
        self._meas_date = raw.info["meas_date"]
        if self._meas_date is not None:
            self._meas_date += datetime.timedelta(seconds=raw.first_time)

    def __repr__(self):
        n_samples = self.data.shape[-1]
        duration = (n_samples / self.sf) / 60
        return (
            f"<SleepStaging | {len(self.ch_names)} x {n_samples} samples ({duration:.1f} minutes), "
            f"{self.sf} Hz>"
        )

    def fit(self):
        """Extract features from data.

        Returns
        -------
        self : returns an instance of self.
        """
        #######################################################################
        # MAIN PARAMETERS
        #######################################################################

        # Bandpass filter
        freq_broad = (0.4, 30)
        # FFT & bandpower parameters
        win_sec = 5  # = 2 / freq_broad[0]
        sf = self.sf
        win = int(win_sec * sf)
        kwargs_welch = dict(window="hamming", nperseg=win, average="median")
        bands = [
            (0.4, 1, "sdelta"),
            (1, 4, "fdelta"),
            (4, 8, "theta"),
            (8, 12, "alpha"),
            (12, 16, "sigma"),
            (16, 30, "beta"),
        ]

        #######################################################################
        # CALCULATE FEATURES
        #######################################################################

        features = []

        # Filter all channels at once — coefficients are computed once, ~2x faster
        # than filtering each channel separately.
        data_filt = filter_data(
            self.data, sf, l_freq=freq_broad[0], h_freq=freq_broad[1], verbose=False
        )

        for i, c in enumerate(self.ch_types):
            # - Extract epochs. Data is now of shape (n_epochs, n_samples).
            times, epochs = sliding_window(data_filt[i], sf=sf, window=30)

            # Calculate standard descriptive statistics
            hmob, hcomp = ant.hjorth_params(epochs, axis=1)

            q = np.quantile(epochs, [0.25, 0.75], axis=1)
            # Skewness and (Fisher) kurtosis from the central moments, which is equivalent to
            # scipy.stats.skew / kurtosis (with bias=True) but computes the moments only once.
            dev = epochs - epochs.mean(axis=1, keepdims=True)
            dev2 = dev**2
            m2 = dev2.mean(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):  # NaN for flat epochs
                skew = (dev2 * dev).mean(axis=1) / m2**1.5
                kurt = (dev2**2).mean(axis=1) / m2**2 - 3
            feat = {
                "std": np.std(epochs, ddof=1, axis=1),
                "iqr": q[1] - q[0],
                "skew": skew,
                "kurt": kurt,
                "nzc": ant.num_zerocross(epochs, axis=1),
                "hmob": hmob,
                "hcomp": hcomp,
            }

            # Calculate spectral power features (for EEG + EOG)
            freqs, psd = sp_sig.welch(epochs, sf, **kwargs_welch)
            if c != "emg":
                bp = bandpower_from_psd_ndarray(psd, freqs, bands=bands)
                for j, (_, _, b) in enumerate(bands):
                    feat[b] = bp[j]

            # Add power ratios for EEG
            if c == "eeg":
                delta = feat["sdelta"] + feat["fdelta"]
                feat["dt"] = delta / feat["theta"]
                feat["ds"] = delta / feat["sigma"]
                feat["db"] = delta / feat["beta"]
                feat["at"] = feat["alpha"] / feat["theta"]

            # Add total power
            idx_broad = np.logical_and(freqs >= freq_broad[0], freqs <= freq_broad[1])
            dx = freqs[1] - freqs[0]
            feat["abspow"] = trapezoid(psd[:, idx_broad], dx=dx)

            # Calculate entropy and fractal dimension features
            feat["perm"] = np.apply_along_axis(ant.perm_entropy, axis=1, arr=epochs, normalize=True)
            feat["higuchi"] = np.apply_along_axis(ant.higuchi_fd, axis=1, arr=epochs)
            feat["petrosian"] = ant.petrosian_fd(epochs, axis=1)

            # Convert to dataframe
            feat = pd.DataFrame(feat).add_prefix(c + "_")
            features.append(feat)

        #######################################################################
        # SMOOTHING & NORMALIZATION
        #######################################################################

        # Save features to dataframe
        features = pd.concat(features, axis=1)
        features.index.name = "epoch"

        # Apply centered rolling average (15 epochs = 7 min 30)
        # Triang: [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.,
        #          0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]
        rollc = features.rolling(window=15, center=True, min_periods=1, win_type="triang").mean()
        rollc[rollc.columns] = robust_scale(rollc, quantile_range=(5, 95))
        rollc = rollc.add_suffix("_c7min_norm")

        # Now look at the past 2 minutes
        rollp = features.rolling(window=4, min_periods=1).mean()
        rollp[rollp.columns] = robust_scale(rollp, quantile_range=(5, 95))
        rollp = rollp.add_suffix("_p2min_norm")

        # Add to current set of features
        features = features.join(rollc).join(rollp)

        #######################################################################
        # TEMPORAL + METADATA FEATURES AND EXPORT
        #######################################################################

        # Add temporal features and metadata using concat to avoid fragmentation
        # time_norm goes from 0 to 1 (it is 0 when there is only one epoch)
        time_norm = times / times[-1] if times[-1] > 0 else np.zeros_like(times)
        extra = {"time_hour": times / 3600, "time_norm": time_norm}
        if self.metadata is not None:
            extra.update(self.metadata)
        features = pd.concat([features, pd.DataFrame(extra, index=features.index)], axis=1)

        # Downcast float64 to float32 (to reduce size of training datasets)
        cols_float = features.select_dtypes(np.float64).columns.tolist()
        features[cols_float] = features[cols_float].astype(np.float32)
        # Make sure that age and sex are encoded as int
        for c in ["age", "male"]:
            if c in features.columns:
                features[c] = features[c].astype(int)

        # Sort the column names here (same behavior as lightGBM)
        features.sort_index(axis=1, inplace=True)

        # Add to self
        self._features = features
        self.feature_name_ = self._features.columns.tolist()

    def get_features(self):
        """Extract features from data and return a copy of the dataframe.

        Returns
        -------
        features : :py:class:`pandas.DataFrame`
            Feature dataframe.
        """
        if not hasattr(self, "_features"):
            self.fit()
        return self._features.copy()

    def _validate_predict(self, clf):
        """Validate classifier."""
        # Check that we're using exactly the same features
        # Note that clf.feature_name_ is only available in lightgbm>=3.0
        for a, b, where in [
            (
                clf.feature_name_,
                self.feature_name_,
                "classifier but not in the current feature set",
            ),
            (
                self.feature_name_,
                clf.feature_name_,
                "current feature set but not in the classifier",
            ),
        ]:
            f_diff = np.setdiff1d(a, b)
            if len(f_diff):
                raise ValueError(f"The following features are present in the {where}: {f_diff}")

    def _load_model(self, path_to_model):
        """Load the relevant trained classifier."""
        if path_to_model == "auto":
            clf_dir = os.path.join(os.path.dirname(__file__), "classifiers")
            name = "clf_eeg"
            name = name + "+eog" if "eog" in self.ch_types else name
            name = name + "+emg" if "emg" in self.ch_types else name
            name = name + "+demo" if self.metadata is not None else name
            # e.g. clf_eeg+eog+emg+demo_lgb_0.4.0.joblib. The "_lgb_" suffix prevents matching
            # other combinations of channels (e.g. "clf_eeg" would otherwise match "clf_eeg+eog").
            all_matching_files = glob.glob(
                os.path.join(clf_dir, glob.escape(name) + "_lgb_*.joblib")
            )
            assert len(all_matching_files), f"No pre-trained classifier found for {name}."

            # Find the latest version, comparing version numbers and not strings (0.10 > 0.9)
            def _version(fname):
                return tuple(
                    int(v) for v in re.findall(r"_lgb_([\d.]+)\.joblib$", fname)[0].split(".")
                )

            path_to_model = max(all_matching_files, key=_version)
        # Check that file exists
        assert os.path.isfile(path_to_model), "File does not exist."
        logger.info("Using pre-trained classifier: %s" % path_to_model)
        # Load using Joblib
        clf = joblib.load(path_to_model)
        # Validate features
        self._validate_predict(clf)
        return clf

    def predict(self, path_to_model="auto"):
        """
        Return the predicted sleep stage for each 30-sec epoch of data.

        Currently, only classifiers that were trained using a
        `LGBMClassifier <https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html>`_
        are supported.

        Parameters
        ----------
        path_to_model : str or "auto"
            Full path to a trained LGBMClassifier, exported as a joblib file. Can be "auto" to
            use YASA's default classifier.

        Returns
        -------
        pred : :py:class:`yasa.Hypnogram`
            The predicted sleep stages. Since YASA v0.7, the predicted sleep stages are now
            returned as a :py:class:`yasa.Hypnogram` instance, which also includes the
            probability of each sleep stage for each epoch.
        """
        if not hasattr(self, "_features"):
            self.fit()
        # Load and validate pre-trained classifier
        clf = self._load_model(path_to_model)
        # Now we make sure that the features are aligned
        X = self._features[clf.feature_name_]
        # Predict the sleep stages and probabilities. The classifier uses "W" and "R" for Wake
        # and REM, which yasa.Hypnogram converts to "WAKE" and "REM" in both values and proba.
        proba = pd.DataFrame(clf.predict_proba(X), columns=clf.classes_)
        # Convert to a `yasa.Hypnogram` instance (including `proba`)
        # If meas_date was set on the original Raw, pass it as start so that the returned
        # Hypnogram is timestamp-aware and upsample_to_data aligns correctly on cropped data.
        start = pd.Timestamp(self._meas_date) if self._meas_date is not None else None
        hyp = Hypnogram(
            values=clf.predict(X),
            freq="30s",
            n_stages=5,
            scorer="YASA",
            proba=proba,
            start=start,
        )
        self._proba = hyp.proba
        return hyp

    def plot_predict_proba(
        self,
        proba=None,
        majority_only=False,
        palette=("#99d7f1", "#009DDC", "xkcd:twilight blue", "xkcd:rich purple", "xkcd:sunflower"),
    ):
        """
        Plot the predicted probability for each sleep stage for each 30-sec epoch of data.

        Parameters
        ----------
        proba : self or DataFrame
            A dataframe with the probability of each sleep stage for each 30-sec epoch of data.
        majority_only : boolean
            If True, probabilities of the non-majority classes will be set to 0.
        palette : list or tuple
            The color of each column of ``proba``. The default colors are for Wake, N1, N2, N3 and
            REM, in that order.
        """
        if proba is None and not hasattr(self, "_proba"):
            raise ValueError("Must call `.predict` before this function")
        if proba is None:
            proba = self._proba
        else:
            assert isinstance(proba, pd.DataFrame), "`proba` must be a pandas.DataFrame"
        if majority_only:
            proba = proba.where(proba.eq(proba.max(axis=1), axis=0), other=0)
        ax = proba.plot(
            kind="area", color=list(palette), figsize=(10, 5), alpha=0.8, stacked=True, lw=0
        )
        ax.set_xlim(0, proba.shape[0])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_xlabel("Time (30-sec epoch)")
        plt.legend(frameon=False, bbox_to_anchor=(1, 1))
        return ax
