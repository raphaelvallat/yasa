"""Automatic sleep staging of polysomnography data."""
import os
import mne
import glob
import joblib
import logging
import numpy as np
import pandas as pd
import antropy as ant
import scipy.signal as sp_sig
import scipy.stats as sp_stats
import matplotlib.pyplot as plt
from mne.filter import filter_data
from sklearn.preprocessing import robust_scale

from .others import sliding_window
from .spectral import bandpower_from_psd_ndarray

logger = logging.getLogger('yasa')


class SleepStaging:
    """
    Automatic sleep staging of polysomnography data.

    To run the automatic sleep staging, you must install the
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
        Fpz electrode (C4-Fpz). Data are assumed to be in Volts (MNE default)
        and will be converted to uV.
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
    `FAQ section <https://raphaelvallat.com/yasa/build/html/faq.html>`_ of the documentation.
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
    z-score.

    .. important:: The PSG data should be in micro-Volts. Do NOT transform (e.g. z-score) or filter
        the signal before running the sleep staging algorithm.

    The data are automatically downsampled to 100 Hz for faster computation.

    **2. Sleep stages prediction**

    YASA comes with a default set of pre-trained classifiers, which were trained and validated
    on ~3000 nights from the `National Sleep Research Resource <https://sleepdata.org/>`_.
    These nights involved participants from a wide age range, of different ethnicities, gender,
    and health status. The default classifiers should therefore works reasonably well on most data.

    The code that was used to train the classifiers can be found on GitHub at:
    https://github.com/raphaelvallat/yasa_classifier

    In addition with the predicted sleep stages, YASA can also return the predicted probabilities
    of each sleep stage at each epoch. This can be used to derive a confidence score at each epoch.

    .. important:: The predictions should ALWAYS be double-check by a trained
        visual scorer, especially for epochs with low confidence. A full
        inspection should be performed in the following cases:

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

    * Dean, Dennis A., et al. "Scaling up scientific discovery in sleep
      medicine: the National Sleep Research Resource." Sleep 39.5 (2016):
      1151-1164.

    * Zhang, Guo-Qiang, et al. "The National Sleep Research Resource: towards
      a sleep data commons." Journal of the American Medical Informatics
      Association 25.10 (2018): 1351-1358.

    Examples
    --------
    For a concrete example, please refer to the example Jupyter notebook:
    https://github.com/raphaelvallat/yasa/blob/master/notebooks/14_automatic_sleep_staging.ipynb

    >>> import mne
    >>> import yasa
    >>> # Load an EDF file using MNE
    >>> raw = mne.io.read_raw_edf("myfile.edf", preload=True)
    >>> # Initialize the sleep staging instance
    >>> sls = yasa.SleepStaging(raw, eeg_name="C4-M1", eog_name="LOC-M2",
    ...                         emg_name="EMG1-EMG2",
    ...                         metadata=dict(age=29, male=True))
    >>> # Get the predicted sleep stages
    >>> hypno = sls.predict()
    >>> # Get the predicted probabilities
    >>> proba = sls.predict_proba()
    >>> # Get the confidence
    >>> confidence = proba.max(axis=1)
    >>> # Plot the predicted probabilities
    >>> sls.plot_predict_proba()

    The sleep scores can then be manually edited in an external graphical user interface
    (e.g. EDFBrowser), as described in the
    `FAQ <https://raphaelvallat.com/yasa/build/html/faq.html>`_.
    """

    def __init__(self, raw, eeg_name, *, eog_name=None, emg_name=None, metadata=None):
        # Type check
        assert isinstance(eeg_name, str)
        assert isinstance(eog_name, (str, type(None)))
        assert isinstance(emg_name, (str, type(None)))
        assert isinstance(metadata, (dict, type(None)))

        # Validate metadata
        if isinstance(metadata, dict):
            if 'age' in metadata.keys():
                assert 0 < metadata['age'] < 120, 'age must be between 0 and 120.'
            if 'male' in metadata.keys():
                metadata['male'] = int(metadata['male'])
                assert metadata['male'] in [0, 1], 'male must be 0 or 1.'

        # Validate Raw instance and load data
        assert isinstance(raw, mne.io.BaseRaw), 'raw must be a MNE Raw object.'
        sf = raw.info['sfreq']
        ch_names = np.array([eeg_name, eog_name, emg_name])
        ch_types = np.array(['eeg', 'eog', 'emg'])
        keep_chan = []
        for c in ch_names:
            if c is not None:
                assert c in raw.ch_names, '%s does not exist' % c
                keep_chan.append(True)
            else:
                keep_chan.append(False)
        # Subset
        ch_names = ch_names[keep_chan].tolist()
        ch_types = ch_types[keep_chan].tolist()
        # Keep only selected channels (creating a copy of Raw)
        raw_pick = raw.copy().pick_channels(ch_names, ordered=True)

        # Downsample if sf != 100
        assert sf > 80, 'Sampling frequency must be at least 80 Hz.'
        if sf != 100:
            raw_pick.resample(100, npad="auto")
            sf = raw_pick.info['sfreq']

        # Get data and convert to microVolts
        data = raw_pick.get_data() * 1e6

        # Extract duration of recording in minutes
        duration_minutes = data.shape[1] / sf / 60
        assert duration_minutes >= 5, 'At least 5 minutes of data is required.'

        # Add to self
        self.sf = sf
        self.ch_names = ch_names
        self.ch_types = ch_types
        self.data = data
        self.metadata = metadata

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
        kwargs_welch = dict(window='hamming', nperseg=win, average='median')
        bands = [
            (0.4, 1, 'sdelta'), (1, 4, 'fdelta'), (4, 8, 'theta'),
            (8, 12, 'alpha'), (12, 16, 'sigma'), (16, 30, 'beta')
        ]

        #######################################################################
        # CALCULATE FEATURES
        #######################################################################

        features = []

        for i, c in enumerate(self.ch_types):
            # Preprocessing
            # - Filter the data
            dt_filt = filter_data(
                self.data[i, :], sf, l_freq=freq_broad[0], h_freq=freq_broad[1], verbose=False)
            # - Extract epochs. Data is now of shape (n_epochs, n_samples).
            times, epochs = sliding_window(dt_filt, sf=sf, window=30)

            # Calculate standard descriptive statistics
            hmob, hcomp = ant.hjorth_params(epochs, axis=1)

            feat = {
                'std': np.std(epochs, ddof=1, axis=1),
                'iqr': sp_stats.iqr(epochs, rng=(25, 75), axis=1),
                'skew': sp_stats.skew(epochs, axis=1),
                'kurt': sp_stats.kurtosis(epochs, axis=1),
                'nzc': ant.num_zerocross(epochs, axis=1),
                'hmob': hmob,
                'hcomp': hcomp
            }

            # Calculate spectral power features (for EEG + EOG)
            freqs, psd = sp_sig.welch(epochs, sf, **kwargs_welch)
            if c != 'emg':
                bp = bandpower_from_psd_ndarray(psd, freqs, bands=bands)
                for j, (_, _, b) in enumerate(bands):
                    feat[b] = bp[j]

            # Add power ratios for EEG
            if c == 'eeg':
                delta = feat['sdelta'] + feat['fdelta']
                feat['dt'] = delta / feat['theta']
                feat['ds'] = delta / feat['sigma']
                feat['db'] = delta / feat['beta']
                feat['at'] = feat['alpha'] / feat['theta']

            # Add total power
            idx_broad = np.logical_and(freqs >= freq_broad[0], freqs <= freq_broad[1])
            dx = freqs[1] - freqs[0]
            feat['abspow'] = np.trapz(psd[:, idx_broad], dx=dx)

            # Calculate entropy and fractal dimension features
            feat['perm'] = np.apply_along_axis(
                ant.perm_entropy, axis=1, arr=epochs, normalize=True)
            feat['higuchi'] = np.apply_along_axis(
                ant.higuchi_fd, axis=1, arr=epochs)
            feat['petrosian'] = ant.petrosian_fd(epochs, axis=1)

            # Convert to dataframe
            feat = pd.DataFrame(feat).add_prefix(c + '_')
            features.append(feat)

        #######################################################################
        # SMOOTHING & NORMALIZATION
        #######################################################################

        # Save features to dataframe
        features = pd.concat(features, axis=1)
        features.index.name = 'epoch'

        # Apply centered rolling average (15 epochs = 7 min 30)
        # Triang: [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.,
        #          0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]
        rollc = features.rolling(
            window=15, center=True, min_periods=1, win_type='triang').mean()
        rollc[rollc.columns] = robust_scale(rollc, quantile_range=(5, 95))
        rollc = rollc.add_suffix('_c7min_norm')

        # Now look at the past 2 minutes
        rollp = features.rolling(window=4, min_periods=1).mean()
        rollp[rollp.columns] = robust_scale(rollp, quantile_range=(5, 95))
        rollp = rollp.add_suffix('_p2min_norm')

        # Add to current set of features
        features = features.join(rollc).join(rollp)

        #######################################################################
        # TEMPORAL + METADATA FEATURES AND EXPORT
        #######################################################################

        # Add temporal features
        features['time_hour'] = times / 3600
        features['time_norm'] = times / times[-1]

        # Add metadata if present
        if self.metadata is not None:
            for c in self.metadata.keys():
                features[c] = self.metadata[c]

        # Downcast float64 to float32 (to reduce size of training datasets)
        cols_float = features.select_dtypes(np.float64).columns.tolist()
        features[cols_float] = features[cols_float].astype(np.float32)
        # Make sure that age and sex are encoded as int
        if 'age' in features.columns:
            features['age'] = features['age'].astype(int)
        if 'male' in features.columns:
            features['male'] = features['male'].astype(int)

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
        if not hasattr(self, '_features'):
            self.fit()
        return self._features.copy()

    def _validate_predict(self, clf):
        """Validate classifier."""
        # Check that we're using exactly the same features
        # Note that clf.feature_name_ is only available in lightgbm>=3.0
        f_diff = np.setdiff1d(clf.feature_name_, self.feature_name_)
        if len(f_diff):
            raise ValueError("The following features are present in the "
                             "classifier but not in the current features set:", f_diff)
        f_diff = np.setdiff1d(self.feature_name_, clf.feature_name_, )
        if len(f_diff):
            raise ValueError("The following features are present in the "
                             "current feature set but not in the classifier:", f_diff)

    def _load_model(self, path_to_model):
        """Load the relevant trained classifier."""
        if path_to_model == "auto":
            from pathlib import Path
            clf_dir = os.path.join(str(Path(__file__).parent), 'classifiers/')
            name = 'clf_eeg'
            name = name + '+eog' if 'eog' in self.ch_types else name
            name = name + '+emg' if 'emg' in self.ch_types else name
            name = name + '+demo' if self.metadata is not None else name
            # e.g. clf_eeg+eog+emg+demo_lgb_0.4.0.joblib
            all_matching_files = glob.glob(clf_dir + name + "*.joblib")
            # Find the latest file
            path_to_model = np.sort(all_matching_files)[-1]
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
        pred : :py:class:`numpy.ndarray`
            The predicted sleep stages.
        """
        if not hasattr(self, '_features'):
            self.fit()
        # Load and validate pre-trained classifier
        clf = self._load_model(path_to_model)
        # Now we make sure that the features are aligned
        X = self._features.copy()[clf.feature_name_]
        # Predict the sleep stages and probabilities
        self._predicted = clf.predict(X)
        proba = pd.DataFrame(clf.predict_proba(X), columns=clf.classes_)
        proba.index.name = 'epoch'
        self._proba = proba
        return self._predicted.copy()

    def predict_proba(self, path_to_model="auto"):
        """
        Return the predicted probability for each sleep stage for each 30-sec epoch of data.

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
        proba : :py:class:`pandas.DataFrame`
            The predicted probability for each sleep stage for each 30-sec epoch of data.
        """
        if not hasattr(self, '_proba'):
            self.predict(path_to_model)
        return self._proba.copy()

    def plot_predict_proba(self, proba=None, majority_only=False,
                           palette=['#99d7f1', '#009DDC', 'xkcd:twilight blue',
                                    'xkcd:rich purple', 'xkcd:sunflower']):
        """
        Plot the predicted probability for each sleep stage for each 30-sec epoch of data.

        Parameters
        ----------
        proba : self or DataFrame
            A dataframe with the probability of each sleep stage for each 30-sec epoch of data.
        majority_only : boolean
            If True, probabilities of the non-majority classes will be set to 0.
        """
        if proba is None and not hasattr(self, '_features'):
            raise ValueError("Must call .predict_proba before this function")
        if proba is None:
            proba = self._proba.copy()
        else:
            assert isinstance(proba, pd.DataFrame), 'proba must be a dataframe'
        if majority_only:
            cond = proba.apply(lambda x: x == x.max(), axis=1)
            proba = proba.where(cond, other=0)
        ax = proba.plot(kind='area', color=palette, figsize=(10, 5), alpha=.8, stacked=True, lw=0)
        # Add confidence
        # confidence = proba.max(1)
        # ax.plot(confidence, lw=1, color='k', ls='-', alpha=0.5,
        #         label='Confidence')
        ax.set_xlim(0, proba.shape[0])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_xlabel("Time (30-sec epoch)")
        plt.legend(frameon=False, bbox_to_anchor=(1, 1))
        return ax
