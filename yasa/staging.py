"""Automatic sleep staging of polysomnography data."""
import os
import mne
import joblib
import logging
import numpy as np
import pandas as pd
import entropy as ent
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

    To run the automatic sleep staging, you must install the lightGBM
    (https://lightgbm.readthedocs.io/) and entropy
    (https://github.com/raphaelvallat/entropy) packages.

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
    For each 30-seconds epoch and each channel, the following features are
    calculated:

    * Standard deviation
    * Interquartile range
    * 10 and 90 percentiles
    * Skewness and kurtosis
    * Number of zero crossings
    * Hjorth mobility and complexity
    * Absolute total power in the 0.4-30 Hz band.
    * Relative power in the main frequency bands (for EEG and EOG only)
    * Power ratios (e.g. delta / beta)
    * Permutation entropy and singular value decomposition entropy
    * Higuchi and Petrosian fractal dimension

    In addition with the raw estimates, the algorithm also calculates a
    smoothed and normalized version of these features. Specifically, a 5-min
    centered weighted rolling average and a 10 min past rolling average are
    applied. The resulting smoothed features are then normalized using a robust
    z-score.

    Note that the data are automatically downsampled to 100 Hz for faster
    computation.

    Examples
    --------
    >>> import mne
    >>> import yasa
    >>> raw = mne.io.read_raw_edf("myfile.edf", preload=True)
    >>> # Initialize the sleep staging instance
    >>> sls = yasa.SleepStaging(raw, eeg_name="C4-M1", eog_name="LOC-M2",
    ...                         emg_name="EMG1-EMG2",
    ...                         metadata=dict(age=29, male=True))
    >>> # Get the predicted sleep stages
    >>> sls.predict("mytrainedclassifier.joblib")
    """

    def __init__(self, raw, *, eeg_name, eog_name=None, emg_name=None,
                 metadata=None):
        # Type check
        assert isinstance(eeg_name, str)
        assert isinstance(eog_name, (str, type(None)))
        assert isinstance(emg_name, (str, type(None)))
        assert isinstance(metadata, (dict, type(None)))

        # Validate metadata
        if isinstance(metadata, dict):
            if 'age' in metadata.keys():
                assert 0 < metadata['age'] < 120, ('age must be between 0 and '
                                                   '120.')
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
            sf = 100

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
        # FUNCTIONS
        #######################################################################

        def nzc(x):
            """Calculate the number of zero-crossings along the last axis."""
            return ((x[..., :-1] * x[..., 1:]) < 0).sum(axis=1)

        def mobility(x):
            """Calculate Hjorth mobility on the last axis."""
            return np.sqrt(np.diff(x, axis=1).var(axis=1) / x.var(axis=1))

        def petrosian(x):
            """Calculate the Petrosian fractal dimension on the last axis."""
            n = x.shape[1]
            ln10 = np.log10(n)
            diff = np.diff(x, axis=1)
            return ln10 / (ln10 + np.log10(n / (n + 0.4 * nzc(diff))))

        #######################################################################
        # CALCULATE FEATURES
        #######################################################################

        features = []

        for i, c in enumerate(self.ch_types):
            # Preprocessing
            # - Filter the data
            dt_filt = filter_data(
                self.data[i, :],
                sf, l_freq=freq_broad[0], h_freq=freq_broad[1], verbose=False)
            # - Extract epochs. Data is now of shape (n_epochs, n_samples).
            times, epochs = sliding_window(dt_filt, sf=sf, window=30)

            # Calculate standard descriptive statistics
            hmob = mobility(epochs)

            feat = {
                'std': np.std(epochs, ddof=1, axis=1),
                'iqr': sp_stats.iqr(epochs, rng=(25, 75), axis=1),
                'skew': sp_stats.skew(epochs, axis=1),
                'kurt': sp_stats.kurtosis(epochs, axis=1),
                'nzc': nzc(epochs),
                'hmob': hmob,
                'hcomp': mobility(np.diff(epochs, axis=1)) / hmob
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
            idx_broad = np.logical_and(
                freqs >= freq_broad[0], freqs <= freq_broad[1])
            dx = freqs[1] - freqs[0]
            feat['abspow'] = np.trapz(psd[:, idx_broad], dx=dx)

            # Calculate entropy and fractal dimension features
            feat['perm'] = np.apply_along_axis(
                ent.perm_entropy, axis=1, arr=epochs, normalize=True)
            feat['higuchi'] = np.apply_along_axis(
                ent.higuchi_fd, axis=1, arr=epochs)
            feat['petrosian'] = petrosian(epochs)

            # Convert to dataframe
            feat = pd.DataFrame(feat).add_prefix(c + '_')
            features.append(feat)

        #######################################################################
        # SMOOTHING & NORMALIZATION
        #######################################################################

        # Save features to dataframe
        features = pd.concat(features, axis=1)
        features.index.name = 'epoch'

        # Apply centered rolling average (11 epochs = 5 min 30)
        # Triang: [1/6, 2/6, 3/6, 4/6, 5/6, 6/6 (X), 5/6, 4/6, 3/6, 2/6, 1/6]
        rollc = features.rolling(
            window=11, center=True, min_periods=1, win_type='triang').mean()
        rollc[rollc.columns] = robust_scale(rollc, quantile_range=(5, 95))
        rollc = rollc.add_suffix('_c5min_norm')

        # Now look at the past 5 minutes
        rollp = features.rolling(window=10, min_periods=1).mean()
        rollp[rollp.columns] = robust_scale(rollp, quantile_range=(5, 95))
        rollp = rollp.add_suffix('_p5min_norm')

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
                if c in ['age', 'male']:
                    features[c] = self.metadata[c]

        # Add to self
        # Note that we sort the column names here (same behavior as lightGBM)
        features.sort_index(axis=1, inplace=True)
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
                             "classifier but not in the current features set:",
                             f_diff)
        f_diff = np.setdiff1d(self.feature_name_, clf.feature_name_, )
        if len(f_diff):
            raise ValueError("The following features are present in the "
                             "current feature set but not in the classifier:",
                             f_diff)

    def predict(self, path_to_model, smooth=False):
        """
        Return the predicted sleep stage for each 30-sec epoch of data.

        Currently, only classifiers that were trained using a LGBMClassifier
        (https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)
        are supported.

        Parameters
        ----------
        path_to_model : str
            Full path to a trained LightGBMClassifier, exported as a
            joblib file.
        smooth : boolean
            If True, smooth the probability using a 2 min 30 centered rolling
            average.

        Returns
        -------
        pred : :py:class:`numpy.ndarray`
            The predicted sleep stages.
        """
        if not hasattr(self, '_features'):
            self.fit()
        # Load and validate pre-trained classifier
        assert os.path.isfile(path_to_model), "File does not exist."
        clf = joblib.load(path_to_model)
        self._validate_predict(clf)
        # Now we make sure that the features are aligned
        X = self._features.copy()[clf.feature_name_]
        if not smooth:
            # Case 1: raw predictions
            self._predicted = clf.predict(X)
        else:
            # Case 2: smoothed predictions
            proba = self.predict_proba(path_to_model, smooth=True)
            self._predicted = proba.idxmax(axis=1).to_numpy()
        return self._predicted.copy()

    def predict_proba(self, path_to_model, smooth=False):
        """
        Return the predicted probability for each sleep stage for each 30-sec
        epoch of data.

        Currently, only classifiers that were trained using a LGBMClassifier
        (https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)
        are supported.

        Parameters
        ----------
        path_to_model : str
            Full path to a trained LightGBMClassifier, exported as a
            joblib file.
        smooth : boolean
            If True, smooth the probability using a 2 min 30 centered rolling
            average.

        Returns
        -------
        proba : :py:class:`pandas.DataFrame`
            The predicted probability for each sleep stage for each 30-sec
            epoch of data.
        """
        if not hasattr(self, '_features'):
            self.fit()
        # Load and validate pre-trained classifier
        assert os.path.isfile(path_to_model), "File does not exist."
        clf = joblib.load(path_to_model)
        self._validate_predict(clf)
        # Now we make sure that the features are aligned
        X = self._features.copy()[clf.feature_name_]
        # Finally, we return the predicted sleep stages
        proba = pd.DataFrame(clf.predict_proba(X), columns=clf.classes_)
        proba.index.name = 'epoch'
        # Optional: smooth the predictions
        if smooth:
            # 5 * 30-sec epochs = 2 minutes 30
            # w = [1/3, 2/3, 1, 2/3, 1/3]
            proba = proba.rolling(
                window=5, center=True, min_periods=1, win_type="triang"
            ).mean()
        self._proba = proba
        return proba.copy()

    def plot_predict_proba(self, proba=None, majority_only=False,
                           palette=['#99d7f1', '#009DDC', 'xkcd:twilight blue',
                                    'xkcd:rich purple', 'xkcd:sunflower']):
        """
        Plot the predicted probability for each sleep stage for each 30-sec
        epoch of data.

        Parameters
        ----------
        proba : self or DataFrame
            A dataframe with the probability of each sleep stage for each
            30-sec epoch of data.
        majority_only : boolean
            If True, probabilities of the non-majority classes will be set
            to 0.
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
        ax = proba.plot(kind='area', color=palette, figsize=(10, 5), alpha=.8,
                        stacked=True, lw=0)
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
