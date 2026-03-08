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

    def test_json_roundtrip(self):
        """Test that to_json / from_json preserves all metadata."""
        import json
        import os
        import tempfile

        stages = ["W", "W", "N1", "N2", "N3", "REM", "W"]

        # Basic round-trip: no start, no scorer, no proba
        hyp = Hypnogram(stages, freq="30s")
        fd, fname = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            hyp.to_json(fname)
            hyp2 = Hypnogram.from_json(fname)
            assert hyp2.freq == hyp.freq
            assert hyp2.n_stages == hyp.n_stages
            assert hyp2.start is None
            assert hyp2.scorer is None
            assert hyp2.proba is None
            np.testing.assert_array_equal(hyp2.hypno.to_numpy(), hyp.hypno.to_numpy())
        finally:
            os.unlink(fname)

        # With tz-aware start and scorer
        hyp_ts = Hypnogram(
            stages, freq="30s", start="2024-01-15 23:00:00", tz="UTC", scorer="Expert"
        )
        fd, fname = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            hyp_ts.to_json(fname)
            hyp_ts2 = Hypnogram.from_json(fname)
            assert hyp_ts2.start == hyp_ts.start
            assert hyp_ts2.scorer == "Expert"
            assert hyp_ts2.start.tzinfo is not None  # tz preserved
        finally:
            os.unlink(fname)

        # File is valid JSON
        fd, fname = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            hyp.to_json(fname)
            with open(fname) as f:
                parsed = json.load(f)
            assert set(parsed.keys()) == {"values", "n_stages", "freq", "start", "scorer", "proba"}
        finally:
            os.unlink(fname)

    def test_dict_roundtrip(self):
        """Test that to_dict / from_dict preserves all metadata."""
        stages = ["W", "W", "N1", "N2", "N3", "REM", "W"]

        # Basic round-trip: no start, no scorer, no proba
        hyp = Hypnogram(stages, freq="30s")
        d = hyp.to_dict()
        assert isinstance(d, dict)
        assert set(d.keys()) == {"values", "n_stages", "freq", "start", "scorer", "proba"}
        assert d["start"] is None
        assert d["scorer"] is None
        assert d["proba"] is None
        assert d["values"] == list(hyp.hypno.to_numpy())
        hyp2 = Hypnogram.from_dict(d)
        assert hyp2.freq == hyp.freq
        assert hyp2.n_stages == hyp.n_stages
        assert hyp2.start is None
        assert hyp2.scorer is None
        assert hyp2.proba is None
        np.testing.assert_array_equal(hyp2.hypno.to_numpy(), hyp.hypno.to_numpy())

        # With tz-aware start and scorer
        hyp_ts = Hypnogram(
            stages, freq="30s", start="2024-01-15 23:00:00", tz="UTC", scorer="Expert"
        )
        d_ts = hyp_ts.to_dict()
        assert d_ts["scorer"] == "Expert"
        assert d_ts["start"] == "2024-01-15T23:00:00+00:00"  # isoformat with tz
        hyp_ts2 = Hypnogram.from_dict(d_ts)
        assert hyp_ts2.start == hyp_ts.start
        assert hyp_ts2.scorer == "Expert"
        assert hyp_ts2.start.tzinfo is not None

        # to_dict and to_json produce identical serializable content
        import json
        import os
        import tempfile

        fd, fname = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            hyp_ts.to_json(fname)
            with open(fname) as f:
                from_file = json.load(f)
            assert from_file == hyp_ts.to_dict()
        finally:
            os.unlink(fname)

        # proba round-trip and 6-decimal rounding
        proba = pd.DataFrame(
            {
                "WAKE": [0.8, 0.1],
                "N1": [0.1, 0.2],
                "N2": [0.05, 0.4],
                "N3": [0.03, 0.2],
                "REM": [0.02, 0.1],
            },
        )
        hyp_p = Hypnogram(["W", "N2"], freq="30s", proba=proba)
        d_p = hyp_p.to_dict()
        assert d_p["proba"] is not None
        # All values rounded to ≤ 6 decimal places
        for col_vals in d_p["proba"].values():
            for v in col_vals:
                assert v == round(v, 6)
        hyp_p2 = Hypnogram.from_dict(d_p)
        pd.testing.assert_frame_equal(hyp_p2.proba, hyp_p.proba.round(6), check_like=True)


# ---------------------------------------------------------------------------
# __init__ — invalid stage values
# ---------------------------------------------------------------------------


def test_invalid_stage_raises():
    with pytest.raises(ValueError, match="do not match"):
        Hypnogram(["W", "N1", "DREAM"])


def test_invalid_stage_wrong_n_stages_hint():
    # "S" is only accepted for n_stages=2; using it with n_stages=5 triggers
    # the "specify n_stages=..." hint in the error message.
    with pytest.raises(ValueError, match="n_stages"):
        Hypnogram(["W", "S", "S"], n_stages=5)


# ---------------------------------------------------------------------------
# __len__, __eq__, __getitem__
# ---------------------------------------------------------------------------

_STAGES = ["W", "W", "N1", "N2", "N3", "REM", "W"]  # 7 epochs


def test_len():
    assert len(Hypnogram(_STAGES)) == len(_STAGES)


def test_eq_non_hypnogram_returns_not_implemented():
    assert Hypnogram(_STAGES).__eq__("not a Hypnogram") is NotImplemented


def test_eq_different_lengths_raises():
    with pytest.raises(ValueError, match="different numbers"):
        Hypnogram(_STAGES) == Hypnogram(_STAGES[:4])


def test_eq_returns_boolean_array():
    hyp1 = Hypnogram(["W", "N2", "REM"])
    hyp2 = Hypnogram(["W", "N3", "REM"])
    np.testing.assert_array_equal(hyp1 == hyp2, [True, False, True])


def test_getitem_negative_index():
    assert Hypnogram(_STAGES)[-1].hypno.iloc[0] == "WAKE"


def test_getitem_advances_start():
    hyp = Hypnogram(_STAGES, start="2024-01-01 23:00:00")
    assert hyp[2].start == pd.Timestamp("2024-01-01 23:01:00")  # 2 × 30 s


def test_getitem_step_raises():
    with pytest.raises(ValueError, match="Step"):
        Hypnogram(_STAGES)[::2]


def test_getitem_empty_slice_raises():
    with pytest.raises(IndexError, match="empty"):
        Hypnogram(_STAGES)[5:3]


def test_getitem_bad_type_raises():
    with pytest.raises(TypeError):
        Hypnogram(_STAGES)["bad"]


def test_getitem_preserves_proba():
    proba = pd.DataFrame(
        {
            "WAKE": [1.0, 0.0, 0.0],
            "N1": [0.0, 1.0, 0.0],
            "N2": [0.0, 0.0, 1.0],
            "N3": [0.0, 0.0, 0.0],
            "REM": [0.0, 0.0, 0.0],
        }
    )
    hyp = Hypnogram(["W", "N1", "N2"], proba=proba)
    sliced = hyp[0:2]
    assert sliced.proba is not None
    assert len(sliced.proba) == 2


# ---------------------------------------------------------------------------
# end property
# ---------------------------------------------------------------------------


def test_end_none_when_no_start():
    assert Hypnogram(_STAGES).end is None


def test_end_computed_when_start_set():
    hyp = Hypnogram(_STAGES, start="2024-01-01 23:00:00")  # 7 × 30 s = 3.5 min
    assert hyp.end == pd.Timestamp("2024-01-01 23:03:30")


# ---------------------------------------------------------------------------
# mapping.setter — auto-fills ART / UNS; preserves custom values when present
# ---------------------------------------------------------------------------


def test_mapping_setter_fills_art_uns():
    hyp = Hypnogram(["W", "N1", "N2", "N3", "REM"])
    hyp.mapping = {"WAKE": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
    assert hyp.mapping["ART"] == -1
    assert hyp.mapping["UNS"] == -2


def test_mapping_setter_keeps_existing_art_uns():
    hyp = Hypnogram(["W", "N1", "N2", "N3", "REM"])
    hyp.mapping = {"WAKE": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "ART": -9, "UNS": -8}
    assert hyp.mapping["ART"] == -9
    assert hyp.mapping["UNS"] == -8


# ---------------------------------------------------------------------------
# consolidate_stages — 5-stage → 4-stage path (N1/N2 → LIGHT, N3 → DEEP)
# ---------------------------------------------------------------------------


def test_consolidate_5_to_4():
    hyp = simulate_hypnogram(tib=60, n_stages=5, seed=0)
    hyp4 = hyp.consolidate_stages(4)
    assert hyp4.n_stages == 4
    assert "LIGHT" in hyp4.labels
    assert "N1" not in hyp4.labels


# ---------------------------------------------------------------------------
# crop
# ---------------------------------------------------------------------------


def test_crop_by_index():
    hyp = Hypnogram(_STAGES)
    cropped = hyp.crop(start=1, end=4)
    assert cropped.n_epochs == 4
    assert cropped.hypno.iloc[0] == "WAKE"


def test_crop_by_timestamp():
    # Epochs: 23:00:00 23:00:30 23:01:00 23:01:30 23:02:00 23:02:30 23:03:00
    hyp = Hypnogram(_STAGES, start="2024-01-01 23:00:00")
    cropped = hyp.crop(start="2024-01-01 23:01:00", end="2024-01-01 23:02:00")
    assert cropped.start == pd.Timestamp("2024-01-01 23:01:00")
    assert cropped.n_epochs == 3  # loc is inclusive on both ends


def test_crop_timestamp_requires_start():
    with pytest.raises(ValueError, match="start"):
        Hypnogram(_STAGES).crop(start="2024-01-01 23:00:00")


def test_crop_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        Hypnogram(_STAGES).crop(start=5, end=3)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_returns_epoch_by_epoch_agreement():
    from yasa.evaluation import EpochByEpochAgreement

    hyp_ref = Hypnogram(_STAGES, scorer="Expert")
    hyp_obs = Hypnogram(_STAGES, scorer="YASA")
    assert isinstance(hyp_ref.evaluate(hyp_obs), EpochByEpochAgreement)


# ---------------------------------------------------------------------------
# find_periods — non-integer threshold raises
# ---------------------------------------------------------------------------


def test_find_periods_non_integer_threshold_raises():
    hyp = Hypnogram(["W"] * 20, freq="30s")
    # 45 s × (1/30 Hz) = 1.5 samples → non-integer → ValueError
    with pytest.raises(ValueError, match="whole number"):
        hyp.find_periods(threshold="45s")
