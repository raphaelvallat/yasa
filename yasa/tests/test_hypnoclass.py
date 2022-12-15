"""Test the class Hypnogram."""
import mne
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from yasa.hypno import simulate_hypno, Hypnogram


def create_raw(npts, ch_names=["F4-M1", "F3-M2"], sf=100):
    """Utility function for test fit to data."""
    nchan = len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=sf, ch_types=["eeg"] * nchan, verbose=0)
    data = np.random.rand(nchan, npts)
    raw = mne.io.RawArray(data, info, verbose=0)
    return raw


class TestHypnoClass(unittest.TestCase):
    """Test class Hypnogram"""

    def test_2stages_hypno(self):
        """Test 2-stages Hypnogram class"""
        values_int = simulate_hypno(tib=120, n_stages=2, seed=42)
        values = pd.Series(values_int).map({0: "W", 1: "S"}).to_numpy()
        hyp = Hypnogram(values, n_stages=2)

        # Check properties
        np.testing.assert_array_equal(hyp.hypno.str.get(0).to_numpy(), values)
        assert isinstance(hyp.hypno.index, pd.RangeIndex)
        assert hyp.sampling_frequency == 1 / 30
        assert hyp.freq == "30s"
        assert hyp.n_epochs == len(values)
        assert hyp.n_stages == 2
        assert hyp.labels == ["SLEEP", "WAKE", "ART", "UNS"]
        assert hyp.mapping == {"WAKE": 0, "SLEEP": 1, "ART": -1, "UNS": -2}
        assert hyp.start is None
        assert hyp.scorer is None
        assert hyp.timedelta[0] == pd.Timedelta("0 days 00:00:00")
        assert hyp.timedelta[-1] == pd.Timedelta("0 days 01:59:30")

        # Adding start time
        hyp = Hypnogram(values, n_stages=2, start="2022-11-10 13:30:10", freq="15s", scorer="Test")
        assert isinstance(hyp.hypno.index, pd.DatetimeIndex)
        assert hyp.hypno.name == "Test"
        assert hyp.scorer == "Test"
        assert hyp.sampling_frequency == 1 / 15
        assert hyp.freq == "15s"
        assert hyp.start == "2022-11-10 13:30:10"
        assert hyp.n_epochs == len(values)
        assert hyp.timedelta[0] == pd.Timedelta("0 days 00:00:00")
        assert hyp.timedelta[-1] == pd.Timedelta("0 days 00:59:45")
        assert hyp.hypno.index[-1] == hyp.hypno.index[0] + pd.Timedelta(
            seconds=(len(values) * 15) - 15
        )

        # Test class methods
        np.testing.assert_array_equal(hyp.as_int(), values_int)
        hyp.transition_matrix()
        hyp.find_periods()
        sstats = hyp.sleep_statistics()
        truth = {
            "TIB": 60.0,
            "SPT": 58.75,
            "WASO": 9.25,
            "TST": 49.5,
            "SE": 82.5,
            "SME": 84.2553,
            "SFI": 0.303,
            "SOL": 1.25,
            "SOL_5min": 1.25,
            "WAKE": 10.5,
        }
        assert sstats == truth

        # Invert the mapping
        hyp.mapping = {"SLEEP": 0, "WAKE": 1}
        hyp.mapping_int == {0: "SLEEP", 1: "WAKE", -1: "ART", -2: "UNS"}
        np.testing.assert_array_equal(hyp.as_int(), (values_int == 0).astype(int))
        sstats = hyp.sleep_statistics()
        assert sstats == truth

        # yasa.Hypnogram.upsample
        hyp_up = hyp.upsample("5s")
        assert hyp_up.hypno.index[0] == hyp.hypno.index[0]
        assert hyp_up.hypno.index[-1] != hyp.hypno.index[-1]
        assert hyp_up.n_epochs == 3 * hyp.n_epochs  # 15-sec to 5-sec

        # yasa.Hypnogram.upsample_to_data
        npts = (3600 * 100) + 10 * 100  # 60 min + 10 seconds (at 100 Hz)
        raw = create_raw(npts=npts, sf=100)
        hyp_up = hyp.upsample_to_data(raw)
        assert isinstance(hyp_up, np.ndarray)
        assert hyp_up.size == npts
        assert hyp_up.dtype == int
        hyp_up = hyp.upsample_to_data(raw.get_data(), sf=100)
