"""Test the functions in yasa/staging.py."""

import unittest
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from yasa.fetchers import fetch_sample
from yasa.hypno import Hypnogram
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
        print(sls)
        print(str(sls))
        assert repr(sls)
        sls.get_features()
        y_pred = sls.predict()
        assert isinstance(y_pred, Hypnogram)
        assert y_pred.proba is not None
        proba = sls.predict_proba()
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

    def test_predict_proba_without_prior_predict(self):
        """Test that predict_proba internally calls predict when _proba is not set."""
        sls = SleepStaging(raw, eeg_name="C4")
        sls.fit()
        with self.assertWarns(FutureWarning):
            proba = sls.predict_proba()
        assert isinstance(proba, pd.DataFrame)
        plt.close("all")
