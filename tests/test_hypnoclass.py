"""Test the class Hypnogram."""

import unittest

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pytest

from yasa.hypno import Hypnogram, hypno_int_to_str, hypno_str_to_int, simulate_hypnogram


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
        hyp = simulate_hypnogram(tib=120, n_stages=2, seed=42)
        print(hyp)
        print(str(hyp))

        # Check properties
        np.testing.assert_array_equal(hyp.hypno.str.get(0)[:10], np.repeat(["W", "S"], 5))
        assert isinstance(hyp.hypno.index, pd.RangeIndex)
        assert hyp.hypno.dtype == "category"
        assert hyp.hypno.index.name == "Epoch"
        assert hyp.sampling_frequency == 1 / 30
        assert hyp.freq == "30s"
        assert hyp.n_epochs == 240
        assert hyp.duration == 120
        assert hyp.n_stages == 2
        assert hyp.labels == ["WAKE", "SLEEP", "ART", "UNS"]
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
        assert hyp.start == pd.Timestamp("2022-11-10 13:30:10")
        assert hyp.n_epochs == len(values)
        assert hyp.duration == 60
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
        hyp.as_events()
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
        assert sstats["TIB"] == hyp.duration
        hyp_cp = hyp.copy()
        np.testing.assert_array_equal(hyp_cp.as_int(), hyp.as_int())
        assert hyp_cp.sleep_statistics() == truth
        assert hyp_cp.scorer == hyp.scorer

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
        assert hyp_up.dtype == np.int16
        hyp_up = hyp.upsample_to_data(raw.get_data(), sf=100)

        # yasa.Hypnogram.simulate_similar
        shyp = hyp.simulate_similar()
        assert shyp.freq == hyp.freq
        assert shyp.start == hyp.start
        assert shyp.scorer == hyp.scorer
        assert shyp.labels == hyp.labels
        assert shyp.duration == hyp.duration
        assert shyp.n_epochs == hyp.n_epochs
        assert shyp.n_stages == hyp.n_stages
        assert shyp.hypno.index.name == hyp.hypno.index.name
        assert shyp.sampling_frequency == hyp.sampling_frequency
        assert hyp.simulate_similar(tib=2, scorer="YASA").scorer == "YASA"
        assert hyp.simulate_similar(tib=2, start="2022-11-10").start == pd.Timestamp("2022-11-10")
        np.testing.assert_array_equal(
            simulate_hypnogram(seed=1).simulate_similar(tib=5, seed=6).as_int(),
            [0, 0, 0, 0, 1, 1, 1, 2, 2, 2],
        )

        # yasa.Hypnogram.plot_hypnogram
        assert isinstance(hyp.plot_hypnogram(), plt.Axes)
        hyp.plot_hypnogram(fill_color="cornflowerblue", highlight="N3", lw=0.5)
        plt.close("all")
        # Make sure mapping stays intact after plotting
        assert hyp.mapping == {"SLEEP": 0, "WAKE": 1, "ART": -1, "UNS": -2}

    def test_3stages_hypno(self):
        """Test 3-stages Hypnogram class"""
        hyp = simulate_hypnogram(tib=120, n_stages=3, freq="1s", seed=42)
        assert hyp.sampling_frequency == 1
        assert hyp.freq == "1s"
        assert hyp.n_stages == 3
        assert hyp.labels == ["WAKE", "NREM", "REM", "ART", "UNS"]
        assert hyp.mapping == {"WAKE": 0, "NREM": 2, "REM": 4, "ART": -1, "UNS": -2}
        sstats = hyp.sleep_statistics()
        assert sstats["TIB"] == 120
        assert "%REM" in sstats.keys()
        assert "Lat_REM" in sstats.keys()

        # Try to set a value that is not a valid category
        with pytest.raises((TypeError, ValueError)):  # TypeError in newer versions of Pandas
            hyp.hypno.loc[0] = "Dream sleep"

    def test_4stages_hypno(self):
        """Test 4-stages Hypnogram class"""
        hyp = simulate_hypnogram(tib=400, n_stages=4, freq="30s", seed=42)
        assert hyp.n_stages == 4
        assert hyp.labels == ["WAKE", "LIGHT", "DEEP", "REM", "ART", "UNS"]
        assert hyp.mapping == {"WAKE": 0, "LIGHT": 2, "DEEP": 3, "REM": 4, "ART": -1, "UNS": -2}
        sstats = hyp.sleep_statistics()
        assert sstats["TIB"] == 400
        assert "%DEEP" in sstats.keys()
        assert "Lat_REM" in sstats.keys()
        assert isinstance(hyp.as_events(), pd.DataFrame)

    def test_5stages_hypno(self):
        """Test 5-stages Hypnogram class"""
        hyp = simulate_hypnogram(tib=600, n_stages=5, freq="30s", seed=42)
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

    def test_from_integers(self):
        """Test Hypnogram.from_integers classmethod."""
        # --- default mapping, 5-stage ---
        int_hypno = np.array([0, 0, 1, 2, 3, 2, 4, 4, 0])
        hyp = Hypnogram.from_integers(int_hypno)
        assert isinstance(hyp, Hypnogram)
        assert hyp.n_stages == 5
        assert hyp.freq == "30s"
        assert hyp.n_epochs == len(int_hypno)
        assert hyp.start is None
        assert hyp.scorer is None
        expected_str = np.array(["WAKE", "WAKE", "N1", "N2", "N3", "N2", "REM", "REM", "WAKE"])
        np.testing.assert_array_equal(hyp.hypno.to_numpy(), expected_str)

        # round-trip: from_integers -> as_int should recover the original array
        np.testing.assert_array_equal(hyp.as_int().to_numpy(), int_hypno)

        # --- list input ---
        hyp_list = Hypnogram.from_integers([0, 1, 2, 3, 4])
        assert hyp_list.n_epochs == 5
        np.testing.assert_array_equal(hyp_list.hypno.to_numpy(), ["WAKE", "N1", "N2", "N3", "REM"])

        # --- pd.Series input ---
        hyp_series = Hypnogram.from_integers(pd.Series([0, 2, 4]))
        np.testing.assert_array_equal(hyp_series.hypno.to_numpy(), ["WAKE", "N2", "REM"])

        # --- ART / UNS epochs (-1, -2) ---
        hyp_art = Hypnogram.from_integers([-1, -2, 0, 2])
        np.testing.assert_array_equal(hyp_art.hypno.to_numpy(), ["ART", "UNS", "WAKE", "N2"])

        # --- optional kwargs forwarded correctly ---
        hyp_kw = Hypnogram.from_integers(
            int_hypno, freq="30s", start="2023-01-01 22:00:00", scorer="S1"
        )
        assert isinstance(hyp_kw.hypno.index, pd.DatetimeIndex)
        assert hyp_kw.hypno.index.name == "Time"
        assert hyp_kw.scorer == "S1"
        assert hyp_kw.hypno.name == "S1"
        assert hyp_kw.start == pd.Timestamp("2023-01-01 22:00:00")

        # --- custom mapping ---
        custom = {1: "W", 2: "R", 3: "N1", 4: "N2", 5: "N3"}
        hyp_custom = Hypnogram.from_integers([1, 3, 4, 5, 2], mapping=custom)
        np.testing.assert_array_equal(
            hyp_custom.hypno.to_numpy(), ["WAKE", "N1", "N2", "N3", "REM"]
        )

        # --- consistency with hypno_int_to_str ---
        int_arr = np.array([0, 1, 2, 3, 4, -1, -2])
        str_arr = hypno_int_to_str(int_arr)
        hyp_via_fn = Hypnogram(str_arr)
        hyp_via_cls = Hypnogram.from_integers(int_arr)
        np.testing.assert_array_equal(hyp_via_fn.hypno.to_numpy(), hyp_via_cls.hypno.to_numpy())

        # --- invalid integer (not in mapping) raises ---
        with pytest.raises(Exception):
            Hypnogram.from_integers([0, 99])
