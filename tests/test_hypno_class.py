"""Test the class Hypnogram (except upsampling, see test_hypno_resample.py)."""

import datetime
import json
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from yasa.evaluation import EpochByEpochAgreement
from yasa.hypno import Hypnogram, simulate_hypnogram

_STAGES = ["W", "W", "N1", "N2", "N3", "REM", "W"]  # 7 epochs
_PAD_STAGES = ["N2", "N2", "REM"]  # 3-epoch base hypnogram
_START = "2022-11-10 13:30:10"

###############################################################################
# Fixtures
###############################################################################


@pytest.fixture
def hyp2():
    """2-stage hypnogram of 2 hours with 30-s epochs, no start and no scorer."""
    return simulate_hypnogram(tib=120, n_stages=2, seed=42)


@pytest.fixture
def hyp2_15s(hyp2):
    """Same values as ``hyp2``, but with 15-s epochs (1 hour), a start and a scorer."""
    return Hypnogram(hyp2.hypno.to_numpy(), n_stages=2, start=_START, freq="15s", scorer="Test")


# Sleep statistics of hyp2_15s
_TRUTH_2STAGES = {
    "TIB": 60.0,
    "SPT": 58.75,
    "WASO": 9.25,
    "TST": 49.5,
    "SE": 82.5,
    "SME": 84.2553,
    "SFI": 1.2121,
    "SOL": 1.25,
    "SOL_5min": 1.25,
    "WAKE": 10.5,
}

###############################################################################
# Construction and properties
###############################################################################


@pytest.mark.parametrize(
    "n_stages, labels, mapping",
    [
        (2, ["WAKE", "SLEEP"], {"WAKE": 0, "SLEEP": 1}),
        (3, ["WAKE", "NREM", "REM"], {"WAKE": 0, "NREM": 2, "REM": 4}),
        (4, ["WAKE", "LIGHT", "DEEP", "REM"], {"WAKE": 0, "LIGHT": 2, "DEEP": 3, "REM": 4}),
        (
            5,
            ["WAKE", "N1", "N2", "N3", "REM"],
            {"WAKE": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4},
        ),
    ],
)
def test_n_stages_labels_and_mapping(n_stages, labels, mapping):
    hyp = simulate_hypnogram(tib=60, n_stages=n_stages, seed=42)
    assert hyp.n_stages == n_stages
    assert hyp.labels == labels + ["ART", "UNS"]
    assert hyp.mapping == {**mapping, "ART": -1, "UNS": -2}


def test_properties_no_start(hyp2):
    assert "<Hypnogram | 240 epochs x 30s (120.00 minutes), 2 unique stages>" in repr(hyp2)
    assert str(hyp2) == repr(hyp2)
    np.testing.assert_array_equal(hyp2.hypno.str.get(0)[:10], np.repeat(["W", "S"], 5))
    assert isinstance(hyp2.hypno.index, pd.RangeIndex)
    assert hyp2.hypno.dtype == "category"
    assert hyp2.hypno.index.name == "Epoch"
    assert hyp2.sampling_frequency == 1 / 30
    assert hyp2.freq == "30s"
    assert hyp2.n_epochs == 240
    assert hyp2.duration == 120
    assert hyp2.start is None
    assert hyp2.scorer is None
    assert hyp2.timedelta[0] == pd.Timedelta("0 days 00:00:00")
    assert hyp2.timedelta[-1] == pd.Timedelta("0 days 01:59:30")


def test_properties_with_start(hyp2_15s):
    hyp = hyp2_15s
    assert "scored by Test" in repr(hyp)
    assert isinstance(hyp.hypno.index, pd.DatetimeIndex)
    assert hyp.hypno.index.name == "Time"
    assert hyp.hypno.name == "Test"
    assert hyp.scorer == "Test"
    assert hyp.sampling_frequency == 1 / 15
    assert hyp.freq == "15s"
    assert hyp.start == pd.Timestamp(_START)
    assert hyp.n_epochs == 240
    assert hyp.duration == 60
    assert hyp.timedelta[0] == pd.Timedelta("0 days 00:00:00")
    assert hyp.timedelta[-1] == pd.Timedelta("0 days 00:59:45")
    assert hyp.hypno.index[-1] == hyp.hypno.index[0] + pd.Timedelta(seconds=(240 * 15) - 15)


def test_freq_1s():
    hyp = simulate_hypnogram(tib=120, n_stages=3, freq="1s", seed=42)
    assert hyp.sampling_frequency == 1
    assert hyp.freq == "1s"


def test_set_invalid_category_raises():
    hyp = simulate_hypnogram(tib=10, n_stages=3, seed=42)
    with pytest.raises((TypeError, ValueError)):  # TypeError in newer versions of Pandas
        hyp.hypno.loc[0] = "Dream sleep"


def test_invalid_stage_raises():
    # Fewer unique values than n_stages → hint to specify n_stages
    with pytest.raises(ValueError, match="do not match") as e:
        Hypnogram(["W", "N1", "DREAM"])
    assert "n_stages=3" in str(e.value)
    # As many unique values as n_stages → no hint
    with pytest.raises(ValueError, match="do not match") as e:
        Hypnogram(["W", "N1", "N2", "N3", "DREAM"])
    assert "n_stages=" not in str(e.value)


def test_invalid_stage_wrong_n_stages_hint():
    # "S" is only accepted for n_stages=2; using it with n_stages=5 triggers
    # the "specify n_stages=..." hint in the error message.
    with pytest.raises(ValueError, match="n_stages"):
        Hypnogram(["W", "S", "S"], n_stages=5)


def test_input_types_equivalent():
    """list, ndarray, Series (with custom index), Categorical and string arrays are equivalent."""
    values = ["W", "N1", "n2", "Rem", "art"]
    ref = Hypnogram(values).hypno
    assert ref.tolist() == ["WAKE", "N1", "N2", "REM", "ART"]
    for v in [
        np.array(values),
        pd.Series(values, index=[10, 11, 12, 13, 14]),
        pd.Categorical(values),
        pd.array(values, dtype="string"),
    ]:
        pd.testing.assert_series_equal(Hypnogram(v).hypno, ref)


def test_proba_aliases_and_index():
    """proba columns accept the same aliases as values, and get a positional index."""
    proba = pd.DataFrame(
        {"W": [0.9, 0.1, 0.0], "N2": [0.1, 0.8, 0.5], "R": [0.0, 0.1, 0.5]}, index=[10, 11, 12]
    )
    hyp = Hypnogram(["W", "N2", "R"], proba=proba)
    assert hyp.proba.columns.tolist() == ["WAKE", "N2", "REM"]
    assert hyp.proba.index.tolist() == [0, 1, 2]
    # The user's dataframe is not modified
    assert proba.columns.tolist() == ["W", "N2", "R"]
    assert proba.index.tolist() == [10, 11, 12]


def test_start_tz_string():
    # naive string + tz → stored as tz-aware Timestamp in local time
    hyp = Hypnogram(_STAGES, start="2024-01-15 23:00:00", tz="Europe/Paris")
    assert hyp.start == pd.Timestamp("2024-01-15 23:00:00", tz="Europe/Paris")


def test_start_tz_aware_datetime():
    # passing a tz-aware datetime directly → tz= not needed
    aware_dt = datetime.datetime(2024, 1, 15, 23, 0, tzinfo=datetime.timezone.utc)
    hyp = Hypnogram(_STAGES, start=aware_dt)
    assert hyp.start == pd.Timestamp("2024-01-15 23:00:00", tz="UTC")


def test_start_tz_conflict_raises():
    # tz-aware datetime + tz= → ValueError
    aware_dt = datetime.datetime(2024, 1, 15, 23, 0, tzinfo=datetime.timezone.utc)
    with pytest.raises(ValueError, match="already timezone-aware"):
        Hypnogram(_STAGES, start=aware_dt, tz="UTC")


def test_end_none_when_no_start():
    assert Hypnogram(_STAGES).end is None


def test_end_computed_when_start_set():
    hyp = Hypnogram(_STAGES, start="2024-01-01 23:00:00")  # 7 × 30 s = 3.5 min
    assert hyp.end == pd.Timestamp("2024-01-01 23:03:30")


###############################################################################
# from_integers
###############################################################################


def test_from_integers_default():
    int_hypno = np.array([0, 0, 1, 2, 3, 2, 4, 4, 0])
    hyp = Hypnogram.from_integers(int_hypno)
    assert isinstance(hyp, Hypnogram)
    assert hyp.n_stages == 5
    assert hyp.freq == "30s"
    assert hyp.n_epochs == len(int_hypno)
    assert hyp.start is None
    assert hyp.scorer is None
    expected_str = ["WAKE", "WAKE", "N1", "N2", "N3", "N2", "REM", "REM", "WAKE"]
    np.testing.assert_array_equal(hyp.hypno.to_numpy(), expected_str)
    # round-trip: from_integers -> as_int should recover the original array
    np.testing.assert_array_equal(hyp.as_int().to_numpy(), int_hypno)


@pytest.mark.parametrize(
    "values, expected",
    [
        ([0, 1, 2, 3, 4], ["WAKE", "N1", "N2", "N3", "REM"]),  # list
        (pd.Series([0, 2, 4]), ["WAKE", "N2", "REM"]),  # pd.Series
        ([-1, -2, 0, 2], ["ART", "UNS", "WAKE", "N2"]),  # ART / UNS
    ],
)
def test_from_integers_input_types(values, expected):
    hyp = Hypnogram.from_integers(values)
    assert hyp.n_epochs == len(expected)
    np.testing.assert_array_equal(hyp.hypno.to_numpy(), expected)


def test_from_integers_kwargs():
    hyp = Hypnogram.from_integers(
        [0, 0, 1, 2], freq="30s", start="2023-01-01 22:00:00", scorer="S1"
    )
    assert isinstance(hyp.hypno.index, pd.DatetimeIndex)
    assert hyp.hypno.index.name == "Time"
    assert hyp.scorer == "S1"
    assert hyp.hypno.name == "S1"
    assert hyp.start == pd.Timestamp("2023-01-01 22:00:00")


def test_from_integers_custom_mapping():
    custom = {1: "W", 2: "R", 3: "N1", 4: "N2", 5: "N3"}
    hyp = Hypnogram.from_integers([1, 3, 4, 5, 2], mapping=custom)
    np.testing.assert_array_equal(hyp.hypno.to_numpy(), ["WAKE", "N1", "N2", "N3", "REM"])


def test_from_integers_invalid_raises():
    with pytest.raises(ValueError, match=r"\[7, 99\] are not in the mapping"):
        Hypnogram.from_integers([0, 99, 7, 99])


###############################################################################
# __len__, __eq__, __getitem__
###############################################################################


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


def test_getitem_out_of_range_raises():
    """Out-of-range integers raise IndexError instead of wrapping around."""
    hyp = Hypnogram(["W", "W", "N1", "N2", "N3", "REM"])
    assert hyp[-6].hypno.iloc[0] == "WAKE"
    for key in [6, 10, -7]:
        with pytest.raises(IndexError):
            hyp[key]


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


###############################################################################
# mapping, as_int, get_mask, copy
###############################################################################


def test_as_int(hyp2_15s):
    values_int = hyp2_15s.hypno.map({"WAKE": 0, "SLEEP": 1}).to_numpy()
    np.testing.assert_array_equal(hyp2_15s.as_int(), values_int)


def test_mapping_inverted(hyp2_15s):
    """Inverting the mapping changes the integers, but not the sleep statistics."""
    values_int = hyp2_15s.as_int().to_numpy()
    hyp2_15s.mapping = {"SLEEP": 0, "WAKE": 1}
    assert hyp2_15s.mapping_int == {0: "SLEEP", 1: "WAKE", -1: "ART", -2: "UNS"}
    np.testing.assert_array_equal(hyp2_15s.as_int(), (values_int == 0).astype(int))
    assert hyp2_15s.sleep_statistics() == _TRUTH_2STAGES


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


def test_mapping_custom_partial_and_many_to_one():
    """Custom mappings can be partial or many-to-one, and do not modify the input dict."""
    hyp = Hypnogram(["W", "N1", "N2", "N3", "ART"])
    mapping = {"WAKE": 0, "N1": 1, "N2": 1, "N3": 1}
    hyp.mapping = mapping
    assert mapping == {"WAKE": 0, "N1": 1, "N2": 1, "N3": 1}
    assert hyp.mapping == {"ART": -1, "UNS": -2, **mapping}
    assert hyp.as_int().tolist() == [0, 1, 1, 1, -1]
    assert hyp.as_int().dtype == np.int16
    # The mapping is kept when slicing / copying
    assert hyp.copy().mapping == hyp.mapping
    assert hyp[1:3].as_int().tolist() == [1, 1]
    # Missing stages still raise, including stages added by pad
    with pytest.raises(AssertionError):
        hyp.mapping = {"WAKE": 0, "N1": 1}
    with pytest.raises(AssertionError):
        hyp.pad(after=1, fill_value="REM")
    assert hyp.pad(after=1, fill_value="N2").as_int().tolist() == [0, 1, 1, 1, -1, 1]


def test_as_int_nan_raises():
    hyp = Hypnogram(["W", "N1", "N2"])
    hyp.hypno.iloc[0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        hyp.as_int()
    with pytest.raises(ValueError, match="missing"):
        hyp.transition_matrix()


def test_get_mask():
    hyp = Hypnogram(["W", "N1", "N2", "N2", "N3", "REM"])
    np.testing.assert_array_equal(hyp.get_mask(["N2", "N3"]), [0, 0, 1, 1, 1, 0])
    np.testing.assert_array_equal(hyp.get_mask("REM"), [0, 0, 0, 0, 0, 1])
    with pytest.raises(ValueError, match="Invalid stage"):
        hyp.get_mask(["N2", "DREAM"])


def test_copy(hyp2_15s):
    hyp_cp = hyp2_15s.copy()
    np.testing.assert_array_equal(hyp_cp.as_int(), hyp2_15s.as_int())
    assert hyp_cp.sleep_statistics() == _TRUTH_2STAGES
    assert hyp_cp.scorer == hyp2_15s.scorer


###############################################################################
# Analysis methods: sleep_statistics, transition_matrix, find_periods, as_events
###############################################################################


def test_sleep_statistics_2stages(hyp2_15s):
    hyp2_15s.transition_matrix()
    hyp2_15s.find_periods()
    hyp2_15s.as_events()
    sstats = hyp2_15s.sleep_statistics()
    assert sstats == _TRUTH_2STAGES
    assert sstats["TIB"] == hyp2_15s.duration


@pytest.mark.parametrize(
    "n_stages, tib, keys",
    [(3, 120, ["%REM", "Lat_REM"]), (4, 400, ["%DEEP", "Lat_REM"]), (5, 600, ["%N3", "Lat_REM"])],
)
def test_sleep_statistics_n_stages(n_stages, tib, keys):
    hyp = simulate_hypnogram(tib=tib, n_stages=n_stages, seed=42)
    sstats = hyp.sleep_statistics()
    assert sstats["TIB"] == tib
    for key in keys:
        assert key in sstats.keys()


def test_sleep_statistics_no_sleep():
    """Hypno is all WAKE (with ART and UNS)."""
    hyp = Hypnogram(100 * ["W"] + 10 * ["Art"] + 30 * ["Uns"], n_stages=5)
    sstats = hyp.sleep_statistics()
    assert "ART" in sstats.keys()
    assert "UNS" in sstats.keys()
    assert np.isnan(sstats["Lat_REM"])
    assert sstats["SPT"] == 0
    assert sstats["N3"] == 0


def test_sol_5min_any_epoch_length():
    """SOL_5min is defined for epoch lengths that do not evenly divide 5 minutes."""
    # 2-min epochs: 5 minutes of persistent sleep = 3 epochs
    hyp = Hypnogram(["W", "N2", "N2", "W", "N2", "N2", "N2", "W"], freq="2min")
    stats = hyp.sleep_statistics()
    assert stats["SOL"] == 2
    assert stats["SOL_5min"] == 8


def test_sleep_statistics_sfi():
    """SFI counts transitions from any sleep stage into WAKE."""
    hyp = Hypnogram(["W", "N2", "N2", "W", "REM", "W", "N1", "N1"], freq="1min")
    # 2 sleep -> wake transitions, TST = 5 minutes
    assert hyp.sleep_statistics()["SFI"] == np.round(2 / (5 / 60), 4)


def test_transition_matrix_many_to_one_mapping():
    """Stages that share the same integer in a custom mapping keep their own label."""
    hyp = Hypnogram(["W", "N1", "N2", "N2", "N1"])
    hyp.mapping = {"WAKE": 0, "N1": 1, "N2": 1, "N3": 1, "REM": 1}
    counts, probs = hyp.transition_matrix()
    assert counts.index.tolist() == ["WAKE", "N1", "N2"]
    assert counts.columns.tolist() == ["WAKE", "N1", "N2"]
    assert counts.loc["N2", "N1"] == 1
    assert counts.loc["N2", "N2"] == 1
    assert np.allclose(probs.sum(axis=1), 1)


def test_find_periods_non_integer_threshold_raises():
    hyp = Hypnogram(["W"] * 20, freq="30s")
    # 45 s × (1/30 Hz) = 1.5 samples → non-integer → ValueError
    with pytest.raises(ValueError, match="whole number"):
        hyp.find_periods(threshold="45s")


def test_as_events_4stages():
    hyp = simulate_hypnogram(tib=400, n_stages=4, seed=42)
    assert isinstance(hyp.as_events(), pd.DataFrame)


def test_evaluate_returns_epoch_by_epoch_agreement():
    hyp_ref = Hypnogram(_STAGES, scorer="Expert")
    hyp_obs = Hypnogram(_STAGES, scorer="YASA")
    assert isinstance(hyp_ref.evaluate(hyp_obs), EpochByEpochAgreement)


###############################################################################
# consolidate_stages
###############################################################################


@pytest.mark.parametrize("new_n_stages", [4, 3, 2])
def test_consolidate_stages(new_n_stages):
    hyp = Hypnogram(100 * ["W"] + 10 * ["Art"] + 30 * ["Uns"], n_stages=5)
    assert hyp.consolidate_stages(new_n_stages=new_n_stages).n_stages == new_n_stages


def test_consolidate_5_to_4():
    """N1/N2 → LIGHT, N3 → DEEP."""
    hyp = simulate_hypnogram(tib=60, n_stages=5, seed=0)
    hyp4 = hyp.consolidate_stages(4)
    assert hyp4.n_stages == 4
    assert "LIGHT" in hyp4.labels
    assert "N1" not in hyp4.labels


###############################################################################
# crop
###############################################################################


def test_crop_by_index():
    hyp = Hypnogram(_STAGES)
    cropped = hyp.crop(start=1, end=4)
    assert cropped.n_epochs == 4
    assert cropped.hypno.iloc[0] == "WAKE"


def test_crop_by_timestamp():
    # Epochs: 23:00:00 23:00:30 23:01:00 23:01:30 23:02:00 23:02:30 23:03:00
    proba = pd.get_dummies(Hypnogram(_STAGES).hypno).astype(float).reset_index(drop=True)
    hyp = Hypnogram(_STAGES, start="2024-01-01 23:00:00", proba=proba)
    cropped = hyp.crop(start="2024-01-01 23:01:00", end="2024-01-01 23:02:00")
    assert cropped.start == pd.Timestamp("2024-01-01 23:01:00")
    assert cropped.n_epochs == 3  # inclusive on both ends
    # proba is cropped to the same epochs
    np.testing.assert_array_equal(cropped.proba.to_numpy(), hyp.proba.iloc[2:5].to_numpy())
    # Only one bound
    assert hyp.crop(start="2024-01-01 23:02:30").n_epochs == 2
    assert hyp.crop(end="2024-01-01 23:00:30").n_epochs == 2


def test_crop_negative_indices():
    """Negative integers in crop count from the end, and update the start accordingly."""
    hyp = Hypnogram(["W", "W", "N1", "N2", "N3", "REM"], start="2022-01-01 23:00:00")
    cropped = hyp.crop(start=-2)
    assert cropped.hypno.tolist() == ["N3", "REM"]
    assert cropped.start == pd.Timestamp("2022-01-01 23:02:00")
    assert hyp.crop(end=-1).n_epochs == 6
    assert hyp.crop(end=-2).hypno.tolist() == ["WAKE", "WAKE", "N1", "N2", "N3"]
    assert hyp.crop(start=-3, end=-2).hypno.tolist() == ["N2", "N3"]


def test_crop_timestamp_requires_start():
    with pytest.raises(ValueError, match="start"):
        Hypnogram(_STAGES).crop(start="2024-01-01 23:00:00")


def test_crop_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        Hypnogram(_STAGES).crop(start=5, end=3)


###############################################################################
# pad
###############################################################################


@pytest.mark.parametrize(
    "before, after, fill_value, expected",
    [
        # Default fill (None = not passed) pads both ends with UNS
        (2, 1, None, ["UNS", "UNS", "N2", "N2", "REM", "UNS"]),
        # Scalar fill_value applies the same stage to both ends
        (1, 2, "WAKE", ["WAKE", "N2", "N2", "REM", "WAKE", "WAKE"]),
        # 'edge' repeats first/last epoch on the respective end
        (2, 1, "edge", ["N2", "N2", "N2", "N2", "REM", "REM"]),
        # Tuple sets different fill stages for before and after
        (1, 2, ("UNS", "WAKE"), ["UNS", "N2", "N2", "REM", "WAKE", "WAKE"]),
        # Tuple can mix 'edge' with a concrete stage label
        (2, 2, ("edge", "UNS"), ["N2", "N2", "N2", "N2", "REM", "UNS", "UNS"]),
        # Padding with zero epochs returns identical values
        (0, 0, None, _PAD_STAGES),
        # Only one side
        (3, None, None, ["UNS", "UNS", "UNS", "N2", "N2", "REM"]),
        (None, 2, None, ["N2", "N2", "REM", "UNS", "UNS"]),
    ],
)
def test_pad(before, after, fill_value, expected):
    kwargs = {} if fill_value is None else dict(fill_value=fill_value)
    padded = Hypnogram(_PAD_STAGES).pad(before=before, after=after, **kwargs)
    assert padded.n_epochs == len(expected)
    assert padded.hypno.to_list() == expected


def test_pad_preserves_metadata():
    """n_stages, freq, scorer are preserved; proba is dropped."""
    proba = pd.DataFrame(
        {
            "WAKE": [0.1, 0.1, 0.0],
            "N1": [0.1, 0.1, 0.0],
            "N2": [0.7, 0.7, 0.1],
            "N3": [0.0, 0.0, 0.0],
            "REM": [0.1, 0.1, 0.9],
        }
    )
    hyp = Hypnogram(_PAD_STAGES, freq="30s", scorer="Expert", proba=proba)
    padded = hyp.pad(before=1, after=1)
    assert padded.freq == "30s"
    assert padded.n_stages == hyp.n_stages
    assert padded.scorer == "Expert"
    assert padded.proba is None


def test_pad_updates_start():
    """Prepending n epochs shifts start back by n * freq."""
    hyp = Hypnogram(_PAD_STAGES, start="2023-01-01 22:00:00")
    padded = hyp.pad(before=2)
    assert padded.start == pd.Timestamp("2023-01-01 21:59:00")  # 2 × 30 s earlier
    assert padded.end == hyp.end  # end unchanged


def test_pad_start_none_stays_none():
    """When Hypnogram has no start, padded Hypnogram also has no start."""
    assert Hypnogram(_PAD_STAGES).pad(before=2, after=2).start is None


def test_pad_timestamp_before_partial_warning():
    """Fractional epoch count for 'before' triggers a UserWarning."""
    hyp = Hypnogram(_PAD_STAGES, start="2023-01-01 22:00:00")
    # 75 s = 2.5 epochs → floor to 2
    with pytest.warns(UserWarning, match="flooring"):
        padded = hyp.pad(before="2023-01-01 21:58:45")
    assert padded.n_epochs == hyp.n_epochs + 2


def test_pad_timestamp_before_exact():
    """Timestamp-based before: exact multiple, no warning."""
    hyp = Hypnogram(_PAD_STAGES, start="2023-01-01 22:00:00")
    # 21:59:30 to 22:00:00 = 30 s = 1 epoch
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        padded = hyp.pad(before="2023-01-01 21:59:30")
    assert padded.n_epochs == hyp.n_epochs + 1
    assert padded.start == pd.Timestamp("2023-01-01 21:59:30")


def test_pad_timestamp_after_exact():
    """Timestamp-based after: exact multiple, no warning."""
    hyp = Hypnogram(_PAD_STAGES, start="2023-01-01 22:00:00")
    # end = 22:01:30; after = 22:02:00 → 30 s = 1 epoch (exact)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        padded = hyp.pad(after="2023-01-01 22:02:00")
    assert padded.n_epochs == hyp.n_epochs + 1
    assert padded.end == pd.Timestamp("2023-01-01 22:02:00")


def test_pad_timestamp_after_partial_warning():
    """Fractional epoch count for 'after' triggers a UserWarning."""
    hyp = Hypnogram(_PAD_STAGES, start="2023-01-01 22:00:00")
    # end = 22:01:30; after = 22:02:15 → 45 s = 1.5 epochs → floor to 1
    with pytest.warns(UserWarning, match="flooring"):
        padded = hyp.pad(after="2023-01-01 22:02:15")
    assert padded.n_epochs == hyp.n_epochs + 1


@pytest.mark.parametrize(
    "start, kwargs, error, match",
    [
        (None, dict(before="2023-01-01 21:59:00"), ValueError, "start"),
        ("2023-01-01 22:00:00", dict(before="2023-01-01 22:01:00"), ValueError, "strictly before"),
        ("2023-01-01 22:00:00", dict(after="2023-01-01 22:00:00"), ValueError, "strictly after"),
        (None, dict(before=1, fill_value="DREAM"), ValueError, "fill_value"),
        (None, dict(before=1, fill_value=("UNS", "DREAM")), ValueError, "fill_value"),
        (None, dict(before=1, fill_value=("UNS", "WAKE", "N1")), ValueError, "2 elements"),
        (None, dict(before=-1), ValueError, "non-negative"),
        (None, dict(after=-2), ValueError, "non-negative"),
        (None, dict(before=3.5), TypeError, None),
    ],
)
def test_pad_invalid_raises(start, kwargs, error, match):
    hyp = Hypnogram(_PAD_STAGES, start=start)
    with pytest.raises(error, match=match):
        hyp.pad(**kwargs)


def test_pad_tz_mismatch_raises():
    hyp = Hypnogram(_PAD_STAGES, start="2023-01-01 22:00:00", tz="UTC")
    with pytest.raises(ValueError, match="timezone"):
        hyp.pad(before="2023-01-01 21:59:00")  # tz-naive


###############################################################################
# Serialization: to_json / from_json, to_dict / from_dict
###############################################################################

_KEYS = {"values", "n_stages", "freq", "start", "scorer", "proba"}


def test_json_roundtrip_basic(tmp_path):
    """No start, no scorer, no proba."""
    hyp = Hypnogram(_STAGES, freq="30s")
    fname = tmp_path / "hyp.json"
    hyp.to_json(fname)
    hyp2 = Hypnogram.from_json(fname)
    assert hyp2.freq == hyp.freq
    assert hyp2.n_stages == hyp.n_stages
    assert hyp2.start is None
    assert hyp2.scorer is None
    assert hyp2.proba is None
    np.testing.assert_array_equal(hyp2.hypno.to_numpy(), hyp.hypno.to_numpy())
    # The file is valid JSON
    with open(fname) as f:
        assert set(json.load(f).keys()) == _KEYS


def test_json_roundtrip_start_scorer(tmp_path):
    """With a tz-aware start and a scorer. The JSON file has the same content as to_dict."""
    hyp = Hypnogram(_STAGES, freq="30s", start="2024-01-15 23:00:00", tz="UTC", scorer="Expert")
    fname = tmp_path / "hyp.json"
    hyp.to_json(fname)
    hyp2 = Hypnogram.from_json(fname)
    assert hyp2.start == hyp.start
    assert hyp2.scorer == "Expert"
    assert hyp2.start.tzinfo is not None  # tz preserved
    with open(fname) as f:
        assert json.load(f) == hyp.to_dict()


def test_dict_roundtrip_basic():
    """No start, no scorer, no proba."""
    hyp = Hypnogram(_STAGES, freq="30s")
    d = hyp.to_dict()
    assert isinstance(d, dict)
    assert set(d.keys()) == _KEYS
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


def test_dict_roundtrip_start_scorer():
    hyp = Hypnogram(_STAGES, freq="30s", start="2024-01-15 23:00:00", tz="UTC", scorer="Expert")
    d = hyp.to_dict()
    assert d["scorer"] == "Expert"
    assert d["start"] == "2024-01-15T23:00:00+00:00"  # isoformat with tz
    hyp2 = Hypnogram.from_dict(d)
    assert hyp2.start == hyp.start
    assert hyp2.scorer == "Expert"
    assert hyp2.start.tzinfo is not None


def test_dict_roundtrip_proba():
    """proba round-trip and 6-decimal rounding."""
    proba = pd.DataFrame(
        {
            "WAKE": [0.8, 0.1],
            "N1": [0.1, 0.2],
            "N2": [0.05, 0.4],
            "N3": [0.03, 0.2],
            "REM": [0.02, 0.1],
        },
    )
    hyp = Hypnogram(["W", "N2"], freq="30s", proba=proba)
    d = hyp.to_dict()
    assert d["proba"] is not None
    # All values rounded to ≤ 6 decimal places
    for col_vals in d["proba"].values():
        for v in col_vals:
            assert v == round(v, 6)
    hyp2 = Hypnogram.from_dict(d)
    pd.testing.assert_frame_equal(hyp2.proba, hyp.proba.round(6), check_like=True)


def test_dict_roundtrip_named_timezone():
    """The named timezone of start survives a to_dict / from_dict roundtrip."""
    hyp = Hypnogram(["W", "N1", "N2"], start="2022-03-26 23:00:00", tz="Europe/Paris")
    d = hyp.to_dict()
    assert d["tz"] == "Europe/Paris"
    hyp2 = Hypnogram.from_dict(json.loads(json.dumps(d)))
    assert str(hyp2.start.tz) == "Europe/Paris"
    assert hyp2.start == hyp.start
    # No tz key when start is naive or None
    assert "tz" not in Hypnogram(["W"], start="2022-01-01").to_dict()
    assert "tz" not in Hypnogram(["W"]).to_dict()


###############################################################################
# simulate_similar
###############################################################################


def test_simulate_similar(hyp2_15s):
    hyp = hyp2_15s
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


def test_simulate_similar_kwargs(hyp2_15s):
    assert hyp2_15s.simulate_similar(tib=2, scorer="YASA").scorer == "YASA"
    assert hyp2_15s.simulate_similar(tib=2, start="2022-11-10").start == pd.Timestamp("2022-11-10")


def test_simulate_similar_seed():
    np.testing.assert_array_equal(
        simulate_hypnogram(seed=1).simulate_similar(tib=5, seed=6).as_int(),
        [0, 0, 0, 0, 1, 1, 1, 2, 2, 2],
    )


def test_simulate_similar_edge_cases():
    """simulate_similar works without WAKE, or with a stage only present at the last epoch."""
    hyp = Hypnogram(["W", "N1", "N2", "N2", "N3", "REM"])
    assert hyp.simulate_similar(seed=0).n_epochs == hyp.n_epochs
    hyp = Hypnogram(["N2", "N2", "N3", "N2", "REM", "REM"])
    sim = hyp.simulate_similar(seed=0)
    assert sim.hypno.iloc[0] == "N2"
    assert "WAKE" not in sim.hypno.tolist()
    # A user-defined trans_probas (with WAKE) overrides the default one without error
    default = simulate_hypnogram(tib=480, seed=42).transition_matrix()[1]
    sim = hyp.simulate_similar(trans_probas=default, seed=0)
    assert sim.n_epochs == hyp.n_epochs


###############################################################################
# plot_hypnogram
###############################################################################


def test_plot_hypnogram(hyp2_15s):
    hyp2_15s.mapping = {"SLEEP": 0, "WAKE": 1}
    assert isinstance(hyp2_15s.plot_hypnogram(), plt.Axes)
    hyp2_15s.plot_hypnogram(fill_color="cornflowerblue", highlight="N3", lw=0.5)
    plt.close("all")
    # Make sure mapping stays intact after plotting
    assert hyp2_15s.mapping == {"SLEEP": 0, "WAKE": 1, "ART": -1, "UNS": -2}


###############################################################################
# plot_hypnodensity
###############################################################################

_HD_5STAGES = ["WAKE"] * 20 + ["N1"] * 10 + ["N2"] * 40 + ["N3"] * 20 + ["REM"] * 10


def _make_proba(stages, n=100):
    """Return a valid probability DataFrame for the given stage list."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.dirichlet(np.ones(len(stages)), size=n), columns=stages)


@pytest.fixture
def hyp_hd():
    """5-stage hypnogram of 100 epochs (50 minutes) with probabilities."""
    return Hypnogram(_HD_5STAGES, proba=_make_proba(["WAKE", "N1", "N2", "N3", "REM"]))


@pytest.mark.parametrize(
    "n_stages, values",
    [
        (5, _HD_5STAGES),
        (4, ["WAKE"] * 25 + ["LIGHT"] * 25 + ["DEEP"] * 25 + ["REM"] * 25),
        (3, ["WAKE"] * 34 + ["NREM"] * 33 + ["REM"] * 33),
        (2, ["WAKE"] * 50 + ["SLEEP"] * 50),
    ],
)
def test_plot_hypnodensity_returns_axes(n_stages, values):
    stages = list(dict.fromkeys(values))  # Unique, in order
    hyp = Hypnogram(values, n_stages=n_stages, proba=_make_proba(stages))
    assert isinstance(hyp.plot_hypnodensity(), plt.Axes)
    plt.close("all")


def test_plot_hypnodensity_no_proba_raises():
    hyp = Hypnogram(["WAKE"] * 10 + ["N2"] * 90)
    with pytest.raises(ValueError, match="proba"):
        hyp.plot_hypnodensity()


def test_plot_hypnodensity_with_start_uses_datetime_axis():
    hyp = Hypnogram(
        _HD_5STAGES,
        proba=_make_proba(["WAKE", "N1", "N2", "N3", "REM"]),
        start="2022-12-15 22:30:00",
    )
    ax = hyp.plot_hypnodensity()
    assert isinstance(ax, plt.Axes)
    # x-axis should use a DateFormatter when start is set
    assert isinstance(ax.xaxis.get_major_formatter(), mdates.DateFormatter)
    plt.close("all")


@pytest.mark.parametrize("n_epochs, xlabel", [(100, "Time [mins]"), (200, "Time [hrs]")])
def test_plot_hypnodensity_xlabel(n_epochs, xlabel):
    """Without start, the x-axis is in hours when the hypnogram is longer than 90 minutes."""
    hyp = Hypnogram(["WAKE"] * n_epochs, proba=_make_proba(["WAKE", "N2"], n=n_epochs))
    ax = hyp.plot_hypnodensity()
    assert ax.get_xlabel() == xlabel
    assert ax.get_xlim()[1] == pytest.approx((n_epochs - 1) / (2 if n_epochs <= 180 else 120))
    plt.close("all")


def test_plot_hypnodensity_accepts_ax_argument(hyp_hd):
    _, ax = plt.subplots()
    assert hyp_hd.plot_hypnodensity(ax=ax) is ax
    plt.close("all")


def test_plot_hypnodensity_custom_palette(hyp_hd):
    custom = {"WAKE": "red", "N1": "green", "N2": "blue", "N3": "purple", "REM": "orange"}
    assert isinstance(hyp_hd.plot_hypnodensity(palette=custom), plt.Axes)
    plt.close("all")


def test_plot_hypnodensity_ylim_and_legend(hyp_hd):
    ax = hyp_hd.plot_hypnodensity()
    assert ax.get_ylim() == (0, 1)
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert set(legend_labels) == {"WAKE", "N1", "N2", "N3", "REM"}
    plt.close("all")


def test_plot_hypnodensity_restores_font_size():
    font_size = plt.rcParams["font.size"]
    hyp = Hypnogram(["WAKE"] * 50 + ["N2"] * 50, proba=_make_proba(["WAKE", "N2"]))
    hyp.plot_hypnodensity()
    assert plt.rcParams["font.size"] == font_size
    plt.close("all")


###############################################################################
# Subclassing
###############################################################################


def test_subclass_preserved():
    """Methods that return a new hypnogram preserve the subclass."""

    class MyHypnogram(Hypnogram):
        pass

    hyp = MyHypnogram(["W", "N1", "N2", "N2", "N3", "REM"], start="2022-01-01 23:00:00")
    for new in [
        hyp.copy(),
        hyp.crop(start=1),
        hyp[1:3],
        hyp.pad(before=1),
        hyp.upsample("10s"),
        hyp.consolidate_stages(2),
        hyp.simulate_similar(seed=0),
    ]:
        assert type(new) is MyHypnogram
