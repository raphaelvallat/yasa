"""Test the functions in the yasa/heart.py file."""

import numpy as np
import pytest

from yasa.heart import hrv_stage


@pytest.fixture(scope="module")
def hrv_default(ecg_8hrs):
    """HRV per stage with the default parameters."""
    return hrv_stage(ecg_8hrs.data, ecg_8hrs.sf, hypno=ecg_8hrs.hypno)


def test_hrv_stage_default(hrv_default):
    """Test the output of hrv_stage with the default parameters."""
    epochs, rpeaks = hrv_default
    assert epochs.shape[0] == len(rpeaks)
    assert epochs["duration"].min() == 120  # 2 minutes
    assert np.array_equal(epochs.columns, ["start", "duration", "hr_mean", "hr_std", "hrv_rmssd"])


def test_hrv_stage_include(ecg_8hrs, hrv_default):
    """Only N2."""
    epochs_N2, _ = hrv_stage(ecg_8hrs.data, ecg_8hrs.sf, hypno=ecg_8hrs.hypno, include=2)
    assert hrv_default[0].xs(2, drop_level=False).equals(epochs_N2)


def test_hrv_stage_no_rr_correction(ecg_8hrs, hrv_default):
    """Disabling RR correction."""
    epochs_norr, _ = hrv_stage(
        ecg_8hrs.data, ecg_8hrs.sf, hypno=ecg_8hrs.hypno, rr_limit=(0, np.inf)
    )
    assert not hrv_default[0].equals(epochs_norr)


def test_hrv_stage_no_threshold(ecg_8hrs, hrv_default):
    """Disabling the duration threshold."""
    epochs_nothresh, _ = hrv_stage(
        ecg_8hrs.data, ecg_8hrs.sf, hypno=ecg_8hrs.hypno, threshold="0min"
    )
    assert epochs_nothresh.shape[0] > hrv_default[0].shape[0]
    assert epochs_nothresh["duration"].min() == 30  # 1 epoch


def test_hrv_stage_equal_length(ecg_8hrs):
    """Equal length periods."""
    epochs_eq, _ = hrv_stage(
        ecg_8hrs.data, ecg_8hrs.sf, hypno=ecg_8hrs.hypno, threshold="5min", equal_length=True
    )
    assert epochs_eq["duration"].nunique() == 1
    assert epochs_eq["duration"].unique()[0] == 300


def test_hrv_stage_no_hypno(ecg_8hrs):
    """No hypno (= full recording). The heartbeat detection is applied on the entire recording."""
    data, sf = ecg_8hrs.data, ecg_8hrs.sf
    epochs_nohypno, _ = hrv_stage(data, sf)
    assert epochs_nohypno.shape[0] == 1
    assert epochs_nohypno.loc[(0, 0), "duration"] == data.size / sf


def test_hrv_stage_no_hypno_equal_length(ecg_8hrs):
    """No hypno with equal_length: equivalent to a sliding window approach."""
    data, sf = ecg_8hrs.data, ecg_8hrs.sf
    epochs_nohypno, _ = hrv_stage(data, sf, equal_length=True)
    assert epochs_nohypno["start"].is_monotonic_increasing
    assert epochs_nohypno["duration"].nunique() == 1
    assert epochs_nohypno.shape[0] == data.size / (2 * 60 * sf)  # 2 minutes


def test_hrv_stage_too_few_heartbeats(ecg_8hrs, caplog):
    """Epochs with fewer than 30 detected heartbeats per minute are skipped (NaN)."""
    sf = ecg_8hrs.sf
    # 2 minutes of ECG, then 10 seconds of ECG followed by noise (9 detected heartbeats)
    data = ecg_8hrs.data[: 4 * 60 * sf].copy()
    data[130 * sf :] = np.random.default_rng(0).normal(size=110 * sf)
    epochs, rpeaks = hrv_stage(data, sf, equal_length=True, verbose="INFO")
    assert "Too few detected heartbeats in epoch 1 of stage 0" in caplog.text
    assert len(rpeaks) == 2
    assert epochs.loc[(0, 0), ["hr_mean", "hr_std", "hrv_rmssd"]].notna().all()
    assert epochs.loc[(0, 1), ["hr_mean", "hr_std", "hrv_rmssd"]].isna().all()


def test_hrv_stage_invalid_rr(ecg_8hrs, caplog):
    """Epochs where the RR intervals cannot be interpolated are skipped."""
    sf = ecg_8hrs.sf
    # All the RR intervals are outside of rr_limit
    epochs, rpeaks = hrv_stage(ecg_8hrs.data[: 2 * 60 * sf], sf, rr_limit=(0, 1), verbose="INFO")
    assert "Invalid RR intervals in epoch 0 of stage 0" in caplog.text
    assert len(rpeaks) == 1
    assert epochs.columns.tolist() == ["start", "duration"]
