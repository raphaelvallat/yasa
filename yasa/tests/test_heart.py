"""Test the functions in the yasa/heart.py file."""

import unittest
import numpy as np
from yasa.heart import hrv_stage

# Load data
ecg_file = np.load("notebooks/data_ECG_8hrs_200Hz.npz")
data = ecg_file["data"]
sf = int(ecg_file["sf"])
hypno = ecg_file["hypno"]


class TestHeart(unittest.TestCase):
    def test_hrv_stage(self):
        """Test function hrv_stage"""
        epochs, rpeaks = hrv_stage(data, sf, hypno=hypno)
        assert epochs.shape[0] == len(rpeaks)
        assert epochs["duration"].min() == 120  # 2 minutes
        assert np.array_equal(
            epochs.columns, ["start", "duration", "hr_mean", "hr_std", "hrv_rmssd"]
        )

        # Only N2
        epochs_N2, _ = hrv_stage(data, sf, hypno=hypno, include=2)
        assert epochs.xs(2, drop_level=False).equals(epochs_N2)

        # Disabling RR correction
        epochs_norr, _ = hrv_stage(data, sf, hypno=hypno, rr_limit=(0, np.inf))
        assert not epochs.equals(epochs_norr)

        # Disabling the duration threshold
        epochs_nothresh, _ = hrv_stage(data, sf, hypno=hypno, threshold="0min")
        assert epochs_nothresh.shape[0] > epochs.shape[0]
        assert epochs_nothresh["duration"].min() == 30  # 1 epoch

        # Equal length
        epochs_eq, _ = hrv_stage(data, sf, hypno=hypno, threshold="5min", equal_length=True)
        assert epochs_eq["duration"].nunique() == 1
        assert epochs_eq["duration"].unique()[0] == 300

        # No hypno (= full recording)
        # The heartbeat detection is applied on the entire recording!
        epochs_nohypno, _ = hrv_stage(data, sf)
        assert epochs_nohypno.shape[0] == 1
        assert epochs_nohypno.loc[(0, 0), "duration"] == data.size / sf

        # No hypno (= full recording) with equal_length
        # Equivalent to a sliding window approach
        epochs_nohypno, _ = hrv_stage(data, sf, equal_length=True)
        assert epochs_nohypno["start"].is_monotonic_increasing
        assert epochs_nohypno["duration"].nunique() == 1
        assert epochs_nohypno.shape[0] == data.size / (2 * 60 * sf)  # 2 minutes
