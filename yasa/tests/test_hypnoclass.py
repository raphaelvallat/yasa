"""Test the class Hypnogram."""
import mne
import unittest
import numpy as np
import pandas as pd
from yasa.hypno import simulate_hypno, Hypnogram, hypno_str_to_int


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
        hyp = simulate_hypno(tib=120, n_stages=2, seed=42)
        print(hyp)
        print(str(hyp))

        # Check properties
        np.testing.assert_array_equal(hyp.hypno.str.get(0)[:10], np.repeat(["W", "S"], 5))
        assert isinstance(hyp.hypno.index, pd.RangeIndex)
        assert hyp.hypno.index.name == "Epoch"
        assert hyp.sampling_frequency == 1 / 30
        assert hyp.freq == "30s"
        assert hyp.n_epochs == 240
        assert hyp.tib == 120
        assert hyp.n_stages == 2
        assert hyp.labels == ["SLEEP", "WAKE", "ART", "UNS"]
        assert hyp.mapping == {"WAKE": 0, "SLEEP": 1, "ART": -1, "UNS": -2}
        assert hyp.start is None
        assert hyp.scorer is None
        assert hyp.timedelta[0] == pd.Timedelta("0 days 00:00:00")
        assert hyp.timedelta[-1] == pd.Timedelta("0 days 01:59:30")

        # Adding start time
        values = hyp.hypno.to_numpy()
        hyp = Hypnogram(values, n_stages=2, start="2022-11-10 13:30:10", freq="15s", scorer="Test")
        assert isinstance(hyp.hypno.index, pd.DatetimeIndex)
        assert hyp.hypno.index.name == "Time"
        assert hyp.hypno.name == "Test"
        assert hyp.scorer == "Test"
        assert hyp.sampling_frequency == 1 / 15
        assert hyp.freq == "15s"
        assert hyp.start == "2022-11-10 13:30:10"
        assert hyp.n_epochs == len(values)
        assert hyp.tib == 60
        assert hyp.timedelta[0] == pd.Timedelta("0 days 00:00:00")
        assert hyp.timedelta[-1] == pd.Timedelta("0 days 00:59:45")
        assert hyp.hypno.index[-1] == hyp.hypno.index[0] + pd.Timedelta(
            seconds=(len(values) * 15) - 15
        )

        # Test class methods
        values_int = hypno_str_to_int(hyp.hypno.tolist(), mapping_dict={"wake": 0, "sleep": 1})
        np.testing.assert_array_equal(hyp.as_int(), values_int)
        hyp.transition_matrix()
        hyp.find_periods()
        hyp.as_annotations()
        hyp.simulate_similar(tib=10, scorer="YASA")
        assert hyp.simulate_similar().n_epochs == hyp.n_epochs
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

    def test_3stages_hypno(self):
        """Test 3-stages Hypnogram class"""
        hyp = simulate_hypno(tib=120, n_stages=3, freq="1s", seed=42)
        assert hyp.sampling_frequency == 1
        assert hyp.freq == "1s"
        assert hyp.n_stages == 3
        assert hyp.labels == ["WAKE", "NREM", "REM", "ART", "UNS"]
        assert hyp.mapping == {"WAKE": 0, "NREM": 2, "REM": 4, "ART": -1, "UNS": -2}
        sstats = hyp.sleep_statistics()
        assert sstats["TIB"] == 120
        assert "%REM" in sstats.keys()
        assert "Lat_REM" in sstats.keys()

    def test_4stages_hypno(self):
        """Test 4-stages Hypnogram class"""
        hyp = simulate_hypno(tib=400, n_stages=4, freq="30s", seed=42)
        assert hyp.n_stages == 4
        assert hyp.labels == ["WAKE", "LIGHT", "DEEP", "REM", "ART", "UNS"]
        assert hyp.mapping == {"WAKE": 0, "LIGHT": 2, "DEEP": 3, "REM": 4, "ART": -1, "UNS": -2}
        sstats = hyp.sleep_statistics()
        assert sstats["TIB"] == 400
        assert "%DEEP" in sstats.keys()
        assert "Lat_REM" in sstats.keys()
        assert isinstance(hyp.as_annotations(), pd.DataFrame)

    def test_5stages_hypno(self):
        """Test 5-stages Hypnogram class"""
        hyp = simulate_hypno(tib=600, n_stages=5, freq="30s", seed=42)
        assert hyp.n_stages == 5
        assert hyp.labels == ["WAKE", "N1", "N2", "N3", "REM", "ART", "UNS"]
        assert hyp.mapping == {
            "WAKE": 0,
            "N1": 1,
            "N2": 2,
            "N3": 3,
            "REM": 4,
            "ART": -1,
            "UNS": -2,
        }

        # Test upsampling (without a pd.DatetimeIndex)
        hyp_up = hyp.upsample("10s")
        assert hyp_up.n_epochs == 3 * hyp.n_epochs
        sstats = hyp.sleep_statistics()
        sstats_up = hyp_up.sleep_statistics()
        assert sstats["TIB"] == sstats_up["TIB"] == 600

        # Hypno is all WAKE (with Art and Uns)
        hyp = Hypnogram(100 * ["W"] + 10 * ["Art"] + 30 * ["Uns"], n_stages=5)
        sstats = hyp.sleep_statistics()
        assert "ART" in sstats.keys()
        assert "UNS" in sstats.keys()
        assert np.isnan(sstats["Lat_REM"])
        assert sstats["SPT"] == 0
        assert sstats["N3"] == 0

        # Consolidate stages
        assert hyp.consolidate_stages(new_n_stages=4).n_stages == 4
        assert hyp.consolidate_stages(new_n_stages=3).n_stages == 3
        assert hyp.consolidate_stages(new_n_stages=2).n_stages == 2
