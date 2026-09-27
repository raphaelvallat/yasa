"""Test the functions in yasa/staging.py."""

import unittest
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy.stats as sp_stats
from mne.filter import filter_data

from yasa.fetchers import fetch_sample
from yasa.hypno import Hypnogram
from yasa.others import sliding_window
from yasa.staging import SleepStaging

##############################################################################
# DATA LOADING
##############################################################################

# MNE Raw
raw_fp = fetch_sample("sub-02_mne_raw.fif")
y_true_fp = fetch_sample("sub-02_hypno_30s.txt")
raw = mne.io.read_raw_fif(raw_fp, preload=True, verbose=0)
y_true = Hypnogram(np.loadtxt(y_true_fp, dtype=str))


class TestStaging(unittest.TestCase):
    """Test SleepStaging."""

    def test_sleep_staging(self):
        """Test sleep staging"""
        sls = SleepStaging(
            raw, eeg_name="C4", eog_name="EOG1", emg_name="EMG1", metadata=dict(age=21, male=False)
        )
        assert str(sls) == repr(sls)
        assert "(49.0 minutes)" in str(sls)
        sls.get_features()
        y_pred = sls.predict()
        assert isinstance(y_pred, Hypnogram)
        assert y_pred.proba is not None
        # Classifier labels "W" and "R" are converted to canonical stage names
        assert y_pred.proba.columns.tolist() == ["WAKE", "N1", "N2", "N3", "REM"]
        proba = y_pred.proba
        assert y_pred.hypno.size == y_true.hypno.size
        assert y_true.duration == y_pred.duration
        assert y_true.n_stages == y_pred.n_stages
        # Check that the accuracy is at least 80%
        # Compare values directly (indexes differ: y_true has integer Epoch, y_pred has Time)
        accuracy = (y_true.hypno.to_numpy() == y_pred.hypno.to_numpy()).mean()
        assert accuracy > 0.80

        # Plot
        sls.plot_predict_proba()
        sls.plot_predict_proba(proba, majority_only=True)
        plt.close("all")

        # Same with different combinations of predictors
        # .. without metadata
        SleepStaging(raw, eeg_name="C4", eog_name="EOG1", emg_name="EMG1").fit()
        # .. without EMG
        SleepStaging(raw, eeg_name="C4", eog_name="EOG1").fit()
        # .. just the EEG
        SleepStaging(raw, eeg_name="C4").fit()

    def test_short_data_warning(self):
        """Test that a warning is raised for recordings shorter than 5 minutes."""
        raw_short = raw.copy().crop(tmax=200)
        with self.assertLogs("yasa", level="WARNING"):
            SleepStaging(raw_short, eeg_name="C4")

    def test_validate_predict_errors(self):
        """Test _validate_predict raises ValueError for mismatched features."""
        sls = SleepStaging(raw, eeg_name="C4")
        sls.fit()

        # Features in clf not present in current feature set
        clf_mock = MagicMock()
        clf_mock.feature_name_ = ["nonexistent_feature"]
        with self.assertRaises(ValueError):
            sls._validate_predict(clf_mock)

        # Features in current set not present in clf
        clf_mock.feature_name_ = sls.feature_name_[:-1]
        with self.assertRaises(ValueError):
            sls._validate_predict(clf_mock)

    def test_plot_predict_proba_no_predict(self):
        """Test that plot_predict_proba raises ValueError before predict is called."""
        sls = SleepStaging(raw, eeg_name="C4")
        with self.assertRaises(ValueError):
            sls.plot_predict_proba()
        sls.fit()
        with self.assertRaises(ValueError):
            sls.plot_predict_proba()

    def test_metadata_not_modified(self):
        """Test that the metadata dict of the caller is not modified, and that {} = None."""
        metadata = dict(age=21, male=True)
        SleepStaging(raw, eeg_name="C4", metadata=metadata)
        assert metadata["male"] is True
        assert SleepStaging(raw, eeg_name="C4", metadata={}).metadata is None

    def test_very_short_data(self):
        """Test that one epoch of data works, and that less than one epoch raises an error."""
        features = SleepStaging(raw.copy().crop(tmax=45), eeg_name="C4").get_features()
        assert features.shape[0] == 1
        assert features["time_norm"].iloc[0] == 0
        with self.assertRaises(ValueError):
            SleepStaging(raw.copy().crop(tmax=20), eeg_name="C4")

    def test_features_skew_kurt(self):
        """Test that skew and kurtosis match scipy."""
        sls = SleepStaging(raw, eeg_name="C4")
        features = sls.get_features()
        data_filt = filter_data(sls.data, sls.sf, l_freq=0.4, h_freq=30, verbose=False)
        _, epochs = sliding_window(data_filt[0], sf=sls.sf, window=30)
        np.testing.assert_allclose(
            features["eeg_skew"], sp_stats.skew(epochs, axis=1).astype(np.float32), rtol=1e-5
        )
        np.testing.assert_allclose(
            features["eeg_kurt"], sp_stats.kurtosis(epochs, axis=1).astype(np.float32), rtol=1e-5
        )

    def test_cropped_raw_start(self):
        """Test that the predicted hypnogram start accounts for cropping of the Raw."""
        raw_dated = raw.copy().set_meas_date(pd.Timestamp("2022-01-01 23:00:00", tz="UTC"))
        raw_cropped = raw_dated.copy().crop(tmin=150)
        hyp = SleepStaging(raw_cropped, eeg_name="C4").predict()
        assert hyp.start == pd.Timestamp("2022-01-01 23:02:30", tz="UTC")
