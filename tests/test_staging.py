"""Test the functions in yasa/staging.py."""

import unittest

import matplotlib.pyplot as plt
import mne
import numpy as np

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
        sls.get_features()
        y_pred = sls.predict()
        assert isinstance(y_pred, Hypnogram)
        assert y_pred.proba is not None
        proba = sls.predict_proba()
        assert y_pred.hypno.size == y_true.hypno.size
        assert y_true.duration == y_pred.duration
        assert y_true.n_stages == y_pred.n_stages
        # Check that the accuracy is at least 80%
        accuracy = (y_true.hypno == y_pred.hypno).mean()
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
