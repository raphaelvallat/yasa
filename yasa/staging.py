"""
Automatic sleep staging of polysomnography data.
"""
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
import tensorpac.methods as tpm
from mne.filter import filter_data
from scipy.fftpack import next_fast_len
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
    loc_name : str or None
        The name of the LOC (Right EOG) channel in ``raw``.
    roc_name : str or None
        The name of the ROC (Right EOG) channel in ``raw``.
    emg_name : str or None
        The name of the EMG channel in ``raw``.
    metadata : dict or None
        A dictionary of metadata. Currently supported keys are:

        * ``'age'``: Age of the participant, in years.
        * ``'male'``: Sex of the participant (1 or True = male, 0 or
          False = female)
    """

    def __init__(self, raw, *, eeg_name, loc_name=None, roc_name=None,
                 emg_name=None, metadata=None):
        # Validate metadata
        assert isinstance(metadata, (dict, type(None))
                          ), "metadata must be a dict or None"
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
        ch_names = np.array([eeg_name, loc_name, roc_name, emg_name])
        ch_types = np.array(['eeg', 'loc', 'roc', 'emg'])
        keep_chan = []
        for c in ch_names:
            if c is not None:
                assert c in raw.ch_names, '%s is not a channel of Raw' % c
                keep_chan.append(True)
            else:
                keep_chan.append(False)
        # Subset
        ch_names = ch_names[keep_chan].tolist()
        ch_types = ch_types[keep_chan].tolist()
        # Keep only selected channels
        raw = raw.pick_channels(ch_names, ordered=True)
        # Get data and convert to microVolts
        data = raw.get_data() * 1e6
        # Extract duration of recording in minutes
        duration_minutes = data.shape[1] / sf / 60
        assert duration_minutes >= 5, 'At least 5 minutes of data is required.'

        # Extract channels
        eeg = data[0, :]
        loc = data[ch_types.index('loc')] if 'loc' in ch_types else None
        roc = data[ch_types.index('roc')] if 'roc' in ch_types else None
        emg = data[ch_types.index('emg')] if 'emg' in ch_types else None

        # Validate sampling frequency
        assert sf > 80, 'Sampling frequency must be at least 80 Hz.'
        if sf >= 1000:
            logger.warning(
                'Very high sampling frequency (sf >= 1000 Hz) can '
                'significantly reduce computation time. For faster execution, '
                'please downsample your data to the 100-500Hz range.'
            )

        # Add to self
        self.sf = sf
        self.ch_names = ch_names
        self.ch_types = ch_types
        self.eeg = eeg
        self.loc = loc
        self.roc = roc
        self.emg = emg
        self.metadata = metadata

    def fit(self):
        """Extract features from data.

        Returns
        -------
        self : returns an instance of self.

        Examples
        --------
        Using an EDF file

        >>> import mne
        >>> import yasa
        >>> raw = mne.io.read_raw_edf("myfile.edf", preload=True)
        """
        #######################################################################
        # MAIN PARAMETERS
        #######################################################################
        # Bandpass filter
        freq_broad = (0.4, 30)
        # FFT & bandpower parameters
        win_sec = 5  # = 2 / freq_broad[0]
        bands = [
            (0.4, 1, 'sdelta'), (1, 4, 'fdelta'), (4, 8, 'theta'),
            (8, 12, 'alpha'), (12, 16, 'sigma'), (16, 30, 'beta')
        ]

        #######################################################################
        # EEG FEATURES
        #######################################################################
        # 1) Preprocessing
        # - Filter the data
        sf = self.sf
        eeg_filt = filter_data(
            self.eeg, sf, l_freq=freq_broad[0], h_freq=freq_broad[1],
            verbose=False)
        # - Extract 30 sec epochs. Data is now of shape (n_epochs, n_samples).
        times, eeg_ep = sliding_window(eeg_filt, sf=sf, window=30)

        # 2) Calculate standard descriptive statistics
        perc = np.percentile(eeg_ep, q=[10, 90], axis=1)

        def nzc(x):
            """Calculate the number of zero-crossings along the last axis."""
            return ((x[..., :-1] * x[..., 1:]) < 0).sum(axis=-1)

        features = {
            'time_hour': times / 3600,
            'time_norm': times / times[-1],
            'eeg_absmean': np.abs(eeg_ep).mean(axis=1),
            'eeg_std': eeg_ep.std(ddof=1, axis=1),
            'eeg_10p': perc[0],
            'eeg_90p': perc[1],
            'eeg_iqr': sp_stats.iqr(eeg_ep, axis=1),
            'eeg_skew': sp_stats.skew(eeg_ep, axis=1),
            'eeg_kurt': sp_stats.kurtosis(eeg_ep, axis=1),
            'eeg_nzc': nzc(eeg_ep),
        }

        # 3) Calculate spectral power features
        win = int(win_sec * sf)
        freqs, psd = sp_sig.welch(
            eeg_ep, sf, window='hamming', nperseg=win, average='median')

        bp = bandpower_from_psd_ndarray(psd, freqs, bands=bands)
        for i, (_, _, b) in enumerate(bands):
            features['eeg_' + b] = bp[i]

        # Add power ratios
        delta = features['eeg_sdelta'] + features['eeg_fdelta']
        features['eeg_dt'] = delta / features['eeg_theta']
        features['eeg_ds'] = delta / features['eeg_sigma']
        features['eeg_db'] = delta / features['eeg_beta']
        features['eeg_at'] = features['eeg_alpha'] / features['eeg_theta']

        # Add total power
        idx_broad = np.logical_and(
            freqs >= freq_broad[0], freqs <= freq_broad[1])
        dx = freqs[1] - freqs[0]
        features['eeg_abspow'] = np.trapz(psd[:, idx_broad], dx=dx)

        # 4) Calculate entropy features
        features['eeg_perm'] = np.apply_along_axis(
            ent.perm_entropy, axis=1, arr=eeg_ep, normalize=True)
        features['eeg_higuchi'] = np.apply_along_axis(
            ent.higuchi_fd, axis=1, arr=eeg_ep)

        # 5) Save features to dataframe
        features = pd.DataFrame(features)
        features.index.name = 'epoch'
        cols_eeg = features.filter(like="eeg_").columns.tolist()

        # 6) Apply centered rolling average (5 min 30)
        roll = features[cols_eeg].rolling(
            window=11, center=True, min_periods=1)
        feat_rollmean = roll.mean().add_suffix('_rollavg_c5min_norm')
        features = features.join(feat_rollmean)

        # 7) Apply in-place normalization on all "*_norm" columns
        features = features.join(features[cols_eeg].add_suffix("_norm"))
        cols_norm = features.filter(like="_norm").columns.tolist()
        cols_norm.remove('time_norm')  # make sure we remove 'time_norm'
        features[cols_norm] = robust_scale(
            features[cols_norm], quantile_range=(5, 95))

        #######################################################################
        # EOG FEATURES
        #######################################################################

        if self.eog is not None:
            # 1) Preprocessing
            # - Filter the data
            eog_filt = filter_data(
                self.eog, sf, l_freq=freq_broad[0], h_freq=freq_broad[1],
                verbose=False)
            # - Extract 30 sec epochs.
            times, eog_ep = sliding_window(eog_filt, sf=sf, window=30)

            # 2) Calculate standard descriptive statistics
            feat_eog = {
                'eog_absmean': np.abs(eog_ep).mean(axis=1),
                'eog_std': eog_ep.std(ddof=1, axis=1),
                'eog_iqr': sp_stats.iqr(eog_ep, axis=1),
                'eog_skew': sp_stats.skew(eog_ep, axis=1),
                'eog_kurt': sp_stats.kurtosis(eog_ep, axis=1),
                'eog_nzc': nzc(eog_ep),
            }

            # 3) Calculate spectral power features
            freqs, psd = sp_sig.welch(
                eog_ep, sf, window='hamming', nperseg=win,
                average='median')

            bp = bandpower_from_psd_ndarray(psd, freqs, bands=bands)
            for i, (_, _, b) in enumerate(bands):
                features['eog_' + b] = bp[i]

            # Add total power
            idx_broad = np.logical_and(
                freqs >= freq_broad[0], freqs <= freq_broad[1])
            dx = freqs[1] - freqs[0]
            features['eog_abspow'] = np.trapz(psd[:, idx_broad], dx=dx)

            # 4) Calculate entropy features
            features['eog_perm'] = np.apply_along_axis(
                ent.perm_entropy, axis=1, arr=eog_ep, normalize=True)
            features['eog_higuchi'] = np.apply_along_axis(
                ent.higuchi_fd, axis=1, arr=eog_ep)

            # 5) Save features to dataframe
            feat_eog = pd.DataFrame(feat_eog)
            feat_eog.index.name = 'epoch'
            cols_eog = feat_eog.filter(like="eog_").columns.tolist()

            # 6) Apply centered rolling average (5 min 30 window)
            roll = feat_eog[cols_eog].rolling(
                window=11, center=True, min_periods=1)
            feat_rollmean = roll.mean().add_suffix('_rollavg_c5min_norm')
            feat_eog = feat_eog.join(feat_rollmean)

            # 7) Apply in-place normalization on all "*_norm" columns
            feat_eog = feat_eog.join(
                feat_eog[cols_eog].add_suffix("_norm"))
            cols_norm = feat_eog.filter(like="_norm").columns.tolist()
            feat_eog[cols_norm] = robust_scale(
                feat_eog[cols_norm], quantile_range=(5, 95))

            # 7) Merge with EEG dataframe
            features = features.join(feat_eog)

        #######################################################################
        # METADATA AND EXPORT
        #######################################################################

        # 8) Add metadata if present
        if self.metadata is not None:
            for c in self.metadata.keys():
                if c in ['age', 'male']:
                    features[c] = self.metadata[c]

        # 9) Add to self
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

        Examples
        --------
        Using an EDF file

        >>> import mne
        >>> import yasa
        >>> raw = mne.io.read_raw_edf("myfile.edf", preload=True)
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

        Examples
        --------
        Using an EDF file

        >>> import mne
        >>> import yasa
        >>> raw = mne.io.read_raw_edf("myfile.edf", preload=True)
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

        Examples
        --------
        Using an EDF file

        >>> import mne
        >>> import yasa
        >>> raw = mne.io.read_raw_edf("myfile.edf", preload=True)

        Calculate confidence (in %) from the probabilities

        >>> confidence = predicted_proba.max(axis=1)
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
