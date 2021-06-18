"""Test the functions in yasa/staging.py."""
import mne
import unittest
import numpy as np
import matplotlib.pyplot as plt
from yasa.staging import SleepStaging

##############################################################################
# DATA LOADING
##############################################################################

# MNE Raw
raw = mne.io.read_raw_fif('notebooks/sub-02_mne_raw.fif', preload=True,
                          verbose=0)
hypno = np.loadtxt('notebooks/sub-02_hypno_30s.txt', dtype=str)


class TestStaging(unittest.TestCase):
    """Test SleepStaging."""

    def test_sleep_staging(self):
        """Test sleep staging"""
        sls = SleepStaging(raw, eeg_name="C4", eog_name="EOG1",
                           emg_name="EMG1", metadata=dict(age=21, male=False))
        sls.get_features()
        y_pred = sls.predict()
        proba = sls.predict_proba()
        assert y_pred.size == hypno.size
        # Check that the accuracy is at least 80%
        accuracy = (hypno == y_pred).sum() / y_pred.size
        assert accuracy > 0.80

        # Plot
        sls.plot_predict_proba()
        sls.plot_predict_proba(proba, majority_only=True)
        plt.close('all')

        # Same with different combinations of predictors
        # .. without metadata
        SleepStaging(raw, eeg_name="C4", eog_name="EOG1",
                     emg_name="EMG1").fit()
        # .. without EMG
        SleepStaging(raw, eeg_name="C4", eog_name="EOG1").fit()
        # .. just the EEG
        SleepStaging(raw, eeg_name="C4").fit()
