"""
Automatic sleep staging of polysomnography data.
"""
import os
import joblib
import logging
import numpy as np
import pandas as pd
import entropy as ent
import scipy.signal as sp_sig
import scipy.stats as sp_stats
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
    eeg : array_like
        Single-channel EEG data. Preferentially the C4-M1 or C3-M2 derivation.
        The units of the data must be uV.
    sf : float
        Sampling frequency of the data in Hz.
    eog : array_like or None
        Single-channel EOG data. Preferentially the E1 (left)-M2 derivation.
        The units of the data must be uV, and the sampling frequency must be
        the same as the EEG.
    metadata : dict or None
        A dictionary of metadata. Currently supported keys are:

        * ``'age'``: Age of the participant, in years.
        * ``'male'``: Sex of the participant (1 or True = male, 0 or
          False = female)
    """

    def __init__(self, eeg, sf, *, eog=None, metadata=None):
        # Type checks
        assert isinstance(sf, (int, float)), "sf must be an int or a float."
        assert isinstance(metadata, (dict, type(None))
                          ), "metadata must be a dict or None"

        # Validate metadata
        if isinstance(metadata, dict):
            if 'age' in metadata.keys():
                assert 0 < metadata['age'] < 120, ('age must be between 0 and '
                                                   '120.')
            if 'male' in metadata.keys():
                metadata['male'] = int(metadata['male'])
                assert metadata['male'] in [0, 1], 'male must be 0 or 1.'

        # Validate EEG data
        eeg = np.squeeze(np.asarray(eeg, dtype=np.float64))
        assert eeg.ndim == 1, 'Only single-channel EEG data is supported.'
        duration_minutes = eeg.size / sf / 60
        assert duration_minutes >= 5, 'At least 5 minutes of data is required.'

        # Validate EOG data
        if eog is not None:
            eog = np.squeeze(np.asarray(eog, dtype=np.float64))
            assert eog.ndim == 1, 'Only single-channel EOG data is supported.'
            assert eog.size == eeg.size, 'EOG must have the same size as EEG.'

        # Validate sampling frequency
        assert sf > 80, 'Sampling frequency must be at least 80 Hz.'
        if sf >= 1000:
            logger.warning(
                'Very high sampling frequency (sf >= 1000 Hz) can '
                'significantly reduce computation time. For faster execution, '
                'please downsample your data to the 100-500Hz range.'
            )

        # Add to self
        self.eeg = eeg
        self.sf = sf
        self.eog = eog
        self.metadata = metadata

    def fit(self, freq_broad=(0.5, 40), win_sec=4):
        """Extract features from data.

        Parameters
        ----------
        freq_broad : tuple or list
            Broadband bandpass filter. Default is 0.5 to 40 Hz.
        win_sec : int or float
            The length of the sliding window, in seconds, used for the Welch
            PSD calculation. Ideally, this should be at least two times the
            inverse of the lower frequency of interest (e.g. for a lower
            frequency of interest of 0.5 Hz, the window length should be at
            least 2 * 1 / 0.5 = 4 seconds).

        Returns
        -------
        self : returns an instance of self.
        """
        #######################################################################
        # EEG FEATURES
        #######################################################################
        # 1) Preprocessing
        # - Filter the data
        eeg_filt = filter_data(
            self.eeg, self.sf, l_freq=freq_broad[0], h_freq=freq_broad[1],
            verbose=False)
        # - Extract 30 sec epochs. Data is now of shape (n_epochs, n_samples).
        times, eeg_ep = sliding_window(eeg_filt, sf=self.sf, window=30)

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
        win = int(win_sec * self.sf)
        freqs, psd = sp_sig.welch(
            eeg_ep, self.sf, window='hamming', nperseg=win, average='median')

        bp = bandpower_from_psd_ndarray(psd, freqs)
        bands = ['delta', 'theta', 'alpha', 'sigma', 'beta', 'gamma']
        for i, b in enumerate(bands):
            features['eeg_' + b] = bp[i]

        # Add power ratios
        features['eeg_dt'] = features['eeg_delta'] / features['eeg_theta']
        features['eeg_ds'] = features['eeg_delta'] / features['eeg_sigma']
        features['eeg_db'] = features['eeg_delta'] / features['eeg_beta']
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

        # 6) Apply centered rolling average / std (5 min 30)
        roll = features[cols_eeg].rolling(
            window=11, center=True, min_periods=1)
        feat_rollmean = roll.mean().add_suffix('_rollavg_c5min_norm')
        features = features.join(feat_rollmean)
        # feat_rollstd = roll.std(ddof=0).add_suffix('_rollstd_c5min_norm')
        # features = features.join(feat_rollstd)

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
                self.eog, self.sf, l_freq=freq_broad[0], h_freq=freq_broad[1],
                verbose=False)
            # - Extract 30 sec epochs.
            times, eog_ep = sliding_window(eog_filt, sf=self.sf, window=30)

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
                eog_ep, self.sf, window='hamming', nperseg=win,
                average='median')
            bp = bandpower_from_psd_ndarray(psd, freqs)
            bands = ['delta', 'theta', 'alpha', 'sigma', 'beta', 'gamma']
            for i, b in enumerate(bands):
                feat_eog['eog_' + b] = bp[i]

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

    def get_features(self, **kwargs):
        """Extract features from data and return a copy of the dataframe.

        Parameters
        ----------
        kwargs : key, value mappings
            Keyword arguments are passed through to the fit method.

        Returns
        -------
        features : :py:class:`pandas.DataFrame`
            Feature dataframe.
        """
        if not hasattr(self, '_features'):
            self.fit(**kwargs)
        return self._features.copy()

    def _validate_predict(self, clf):
        """Validate classifier."""
        # Check that we're using exactly the same features
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

    def predict(self, path_to_model):
        """
        Return the predicted sleep stage for each 30-sec epoch of data.

        Currently, only classifiers that were trained using a LGBMClassifier
        (https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)
        are supported.
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
        return clf.predict(X)

    def predict_proba(self, path_to_model):
        """
        Return the predicted probability for each sleep stage for each 30-sec
        epoch of data.

        Currently, only classifiers that were trained using a LGBMClassifier
        (https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)
        are supported.
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
        return clf.predict_proba(X)
