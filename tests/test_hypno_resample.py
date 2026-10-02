"""Test hypnogram upsampling: Hypnogram.upsample, Hypnogram.upsample_to_data and the helpers.

Hypnogram.upsample_to_data is tested on all combinations of:
  - data type : NumPy array | MNE Raw without meas_date | MNE Raw with meas_date
  - Hypnogram : no start | naive start | tz-aware start (via tz=) | tz-aware datetime

Default behaviour (meas_date_is_local=True): meas_date is treated as a local absolute
timestamp, consistent with the EDF+ standard, which defines starttime as local time at
the patient's location. MNE reads this and tags it as UTC; YASA corrects this by default.
Both meas_date and Hypnogram.start are compared as absolute values, so no tz is required.

Set meas_date_is_local=False only for EDF files that genuinely store UTC in meas_date.
"""

import datetime
import logging

import mne
import numpy as np
import pytest

import yasa
from yasa.hypno import Hypnogram, _hypno_fit_to_data, simulate_hypnogram

###############################################################################
# Constants and helpers
###############################################################################

SF = 100  # EEG sampling frequency (Hz)
SPE = SF * 30  # samples per 30-second epoch = 3000

# 9-epoch integer array used by the _hypno_fit_to_data tests
HYPNO_INT = np.array([0, 0, 0, 1, 2, 2, 3, 3, 4])

# 10-epoch string hypnogram used by Hypnogram class tests
#   stages : W  W  N1  N2  N2  N3  N3  REM  REM  W
#   ints   : 0  0   1   2   2   3   3    4    4  0
STAGES = ["W", "W", "N1", "N2", "N2", "N3", "N3", "REM", "REM", "W"]
N = len(STAGES)  # 10

# Reference hypnogram start used across timestamp tests
HYP_START = "2024-01-15 23:00:00"  # naive string, represents local time


def make_raw(n_epochs, meas_date=None):
    """Single-channel MNE RawArray with an optional meas_date."""
    info = mne.create_info(["EEG"], sfreq=SF, ch_types=["eeg"], verbose=False)
    raw = mne.io.RawArray(np.zeros((1, n_epochs * SPE)), info, verbose=False)
    if meas_date is not None:
        raw.set_meas_date(meas_date)
    return raw


def utc(h, m, s=0):
    """UTC datetime on 2024-01-15."""
    return datetime.datetime(2024, 1, 15, h, m, s, tzinfo=datetime.timezone.utc)


###############################################################################
# Hypnogram.upsample
###############################################################################


def test_upsample():
    """Each epoch is repeated, and new_freq must evenly divide the current epoch duration."""
    hyp = Hypnogram(["W", "N1", "N2"], start="2022-01-01 23:00")
    up = hyp.upsample("10s")
    assert up.hypno.tolist() == 3 * ["WAKE"] + 3 * ["N1"] + 3 * ["N2"]
    assert up.n_epochs == 3 * hyp.n_epochs
    assert up.hypno.index[0] == hyp.hypno.index[0]
    assert up.hypno.index[-1] != hyp.hypno.index[-1]
    assert up.end == hyp.end
    with pytest.raises(AssertionError):
        hyp.upsample("20s")


def test_upsample_sleep_statistics():
    """Sleep statistics must not depend on the epoch length of the hypnogram (no start)."""
    hyp = simulate_hypnogram(tib=600, n_stages=5, freq="30s", seed=42)
    hyp_up = hyp.upsample("10s")
    assert hyp_up.n_epochs == 3 * hyp.n_epochs
    sstats = hyp.sleep_statistics()
    sstats_up = hyp_up.sleep_statistics()
    assert sstats["TIB"] == sstats_up["TIB"] == 600
    # Values are rounded to 4 decimals, so upsampling can shift the last digit
    assert sstats.keys() == sstats_up.keys()
    for key in sstats.keys():
        np.testing.assert_allclose(
            sstats[key], sstats_up[key], atol=1e-3, err_msg=f"sleep_stat={key}"
        )


###############################################################################
# _hypno_fit_to_data
###############################################################################


@pytest.fixture
def hypno100():
    return np.repeat(HYPNO_INT, SPE)


def _make_data(n_epochs, as_raw):
    return make_raw(n_epochs) if as_raw else np.zeros(n_epochs * SPE)


@pytest.mark.parametrize("as_raw", [False, True], ids=["array", "raw"])
def test_fit_exact(hypno100, as_raw):
    assert np.array_equal(
        _hypno_fit_to_data(hypno100, _make_data(HYPNO_INT.size, as_raw)), hypno100
    )


@pytest.mark.parametrize("as_raw", [False, True], ids=["array", "raw"])
def test_fit_pads_when_shorter(hypno100, as_raw):
    # hypno shorter than data → last value repeated at the end
    n = HYPNO_INT.size + 1
    assert _hypno_fit_to_data(hypno100, _make_data(n, as_raw)).size == n * SPE


@pytest.mark.parametrize("as_raw", [False, True], ids=["array", "raw"])
def test_fit_crops_when_longer(hypno100, as_raw):
    # hypno longer than data → trailing epochs removed
    n = HYPNO_INT.size - 1
    assert _hypno_fit_to_data(hypno100, _make_data(n, as_raw)).size == n * SPE


###############################################################################
# Deprecated hypno_upsample_to_sf and hypno_upsample_to_data
###############################################################################


def test_deprecated_hypno_upsample_to_sf():
    with pytest.warns(FutureWarning, match="deprecated and will be removed in v0.9"):
        out = yasa.hypno_upsample_to_sf(HYPNO_INT, 1 / 30, 1)
    np.testing.assert_array_equal(out, np.repeat(HYPNO_INT, 30))


def test_deprecated_hypno_upsample_to_data():
    with pytest.warns(FutureWarning, match="deprecated and will be removed in v0.9"):
        out = yasa.hypno_upsample_to_data(HYPNO_INT, 1 / 30, np.zeros(270), 1)
    np.testing.assert_array_equal(out, np.repeat(HYPNO_INT, 30))


###############################################################################
# Hypnogram.upsample_to_data — length-based path
#
# Triggered when EITHER:
#   - data is a NumPy array, OR
#   - data is MNE Raw without meas_date, OR
#   - data is MNE Raw with meas_date but Hypnogram has no start
#
# In all three cases the behaviour is identical: align at t=0, crop/pad at end.
###############################################################################


@pytest.fixture(params=["array", "raw_no_meas", "raw_with_meas"])
def length_data(request):
    """Factory(n_epochs) → data object that always triggers length-based alignment."""
    if request.param == "array":
        return lambda n: np.zeros(n * SPE)
    elif request.param == "raw_no_meas":
        return lambda n: make_raw(n)
    else:
        # meas_date is set, but Hypnogram will have no start → still length-based
        return lambda n: make_raw(n, meas_date=utc(23, 0))


def test_length_based_exact(length_data):
    hyp = Hypnogram(STAGES, freq="30s")
    result = hyp.upsample_to_data(length_data(N), sf=SF)
    assert result.size == N * SPE
    assert np.all(result[:SPE] == 0)  # first epoch: W
    assert np.all(result[2 * SPE : 3 * SPE] == 1)  # third epoch: N1


def test_length_based_pads_when_shorter(length_data):
    # Hypnogram covers N epochs, data covers N+2 → trailing samples set to UNS
    hyp = Hypnogram(STAGES, freq="30s")
    result = hyp.upsample_to_data(length_data(N + 2), sf=SF)
    assert result.size == (N + 2) * SPE
    assert np.all(result[-2 * SPE :] == -2)  # UNS after hypnogram end


def test_length_based_crops_when_longer(length_data):
    # Hypnogram covers N epochs, data covers N-3 → trailing epochs removed
    hyp = Hypnogram(STAGES, freq="30s")
    result = hyp.upsample_to_data(length_data(N - 3), sf=SF)
    assert result.size == (N - 3) * SPE
    assert np.all(result[-SPE:] == 3)  # 7th epoch (index 6): N3


def test_length_based_custom_mapping_15s():
    """2-stage hypnogram with 15-s epochs and an inverted mapping, on a longer recording."""
    values = simulate_hypnogram(tib=120, n_stages=2, seed=42).hypno.to_numpy()
    hyp = Hypnogram(values, n_stages=2, start="2022-11-10 13:30:10", freq="15s", scorer="Test")
    hyp.mapping = {"SLEEP": 0, "WAKE": 1}
    npts = (3600 * 100) + 10 * 100  # 60 min + 10 seconds (at 100 Hz)
    raw = mne.io.RawArray(
        np.zeros((2, npts)),
        mne.create_info(["F4-M1", "F3-M2"], sfreq=100, ch_types="eeg", verbose=False),
        verbose=False,
    )
    hyp_up = hyp.upsample_to_data(raw)
    assert isinstance(hyp_up, np.ndarray)
    assert hyp_up.size == npts
    assert hyp_up.dtype == np.int16
    np.testing.assert_array_equal(hyp_up[: 15 * 100 * 240 : 1500], hyp.as_int())
    assert np.all(hyp_up[-10 * 100 :] == -2)  # UNS after hypnogram end
    np.testing.assert_array_equal(hyp.upsample_to_data(raw.get_data(), sf=100), hyp_up)


###############################################################################
# Hypnogram.upsample_to_data — timestamp-aware path
#
# Triggered when BOTH self.start is set AND raw.meas_date is set.
# The hypnogram epochs are selected by absolute timestamp offset, not sample count.
#
# Hypnogram : 10 epochs starting at 23:00 (local time)
#   index:  0   1   2    3    4    5    6     7     8   9
#   stage:  W   W   N1   N2   N2   N3   N3   REM   REM  W
#   int:    0   0    1    2    2    3    3     4     4   0
###############################################################################


@pytest.fixture
def hyp_utc():
    """10-epoch Hypnogram with start=23:00, tz="UTC".

    The UTC label is stripped under the default meas_date_is_local=True, so all
    arithmetic is performed on the stored value 23:00. Tests that exercise the true-UTC
    path (meas_date_is_local=False) must pass that flag explicitly.
    """
    return Hypnogram(STAGES, freq="30s", start=HYP_START, tz="UTC")


def test_ts_naive_start_local_default():
    # Default (meas_date_is_local=True): naive start works fine — both sides treated
    # as local absolute timestamps, which is the common EDF case.
    hyp_naive = Hypnogram(STAGES, freq="30s", start=HYP_START)
    raw = make_raw(N, meas_date=utc(23, 0))  # "UTC" label, actually local time
    result = hyp_naive.upsample_to_data(raw)  # meas_date_is_local=True (default)
    assert result.size == N * SPE
    assert np.all(result[:SPE] == 0)  # epoch 0: W
    assert np.all(result[2 * SPE : 3 * SPE] == 1)  # epoch 2: N1


def test_ts_naive_start_raises_when_true_utc():
    # meas_date_is_local=False (true UTC path): naive start + UTC-aware meas_date
    # → ValueError because YASA cannot convert naive time to UTC.
    hyp_naive = Hypnogram(STAGES, freq="30s", start=HYP_START)
    with pytest.raises(ValueError, match="timezone"):
        hyp_naive.upsample_to_data(make_raw(N, meas_date=utc(23, 0)), meas_date_is_local=False)


def test_ts_zero_offset(hyp_utc):
    # meas_date == hyp.start → perfect alignment, first epoch is W, third is N1
    raw = make_raw(N, meas_date=utc(23, 0))
    result = hyp_utc.upsample_to_data(raw)
    assert result.size == N * SPE
    assert np.all(result[:SPE] == 0)  # epoch 0: W
    assert np.all(result[2 * SPE : 3 * SPE] == 1)  # epoch 2: N1


def test_ts_positive_offset(hyp_utc):
    # Recording starts 2 min (4 epochs) after hypnogram → epochs 0-3 skipped
    # Remaining epochs 4-9: N2 N3 N3 REM REM W → ints 2 3 3 4 4 0
    raw = make_raw(6, meas_date=utc(23, 2))
    result = hyp_utc.upsample_to_data(raw)
    assert result.size == 6 * SPE
    assert np.all(result[:SPE] == 2)  # epoch 4: N2
    assert np.all(result[SPE : 2 * SPE] == 3)  # epoch 5: N3
    assert np.all(result[-SPE:] == 0)  # epoch 9: W


def test_ts_negative_offset(hyp_utc):
    # Recording starts 30 s before hypnogram → 1 UNS epoch prepended
    raw = make_raw(5, meas_date=utc(22, 59, 30))
    result = hyp_utc.upsample_to_data(raw)
    assert result.size == 5 * SPE
    assert np.all(result[:SPE] == -2)  # prepended UNS
    assert np.all(result[SPE : 2 * SPE] == 0)  # epoch 0: W


def test_ts_local_timezone():
    # meas_date_is_local=False (true UTC): start = "23:00 CET" = "22:00 UTC",
    # meas_date = 22:00 UTC → YASA converts hyp_start to UTC → zero offset.
    hyp = Hypnogram(STAGES, freq="30s", start=HYP_START, tz="Europe/Paris")
    raw = make_raw(N, meas_date=utc(22, 0))  # genuinely 22:00 UTC = 23:00 CET
    result = hyp.upsample_to_data(raw, meas_date_is_local=False)
    assert result.size == N * SPE
    assert np.all(result[:SPE] == 0)  # epoch 0: W
    assert np.all(result[2 * SPE : 3 * SPE] == 1)  # epoch 2: N1


def test_ts_aware_start_naive_meas_date_true_utc():
    # Unusual: tz-aware start and a naive meas_date with meas_date_is_local=False. MNE only
    # accepts UTC-aware dates in set_meas_date, so the naive date is set through the private
    # Info._unlock. The naive meas_date is taken as UTC: 23:00 CET = 22:00 UTC → zero offset.
    hyp = Hypnogram(STAGES, freq="30s", start=HYP_START, tz="Europe/Paris")
    raw = make_raw(N)
    with raw.info._unlock():
        raw.info["meas_date"] = datetime.datetime(2024, 1, 15, 22, 0)
    result = hyp.upsample_to_data(raw, meas_date_is_local=False)
    assert result.size == N * SPE
    assert np.array_equal(result[::SPE], hyp.as_int())


def test_ts_hypno_shorter_than_data(hyp_utc):
    # Recording starts 1 epoch early (UNS prepended) and extends 1 epoch past hypnogram end
    # → 1 UNS + 10 real epochs + 1 padded = 12 epochs window
    raw = make_raw(12, meas_date=utc(22, 59, 30))
    result = hyp_utc.upsample_to_data(raw)
    assert result.size == 12 * SPE
    assert np.all(result[:SPE] == -2)  # UNS (before hypnogram start)
    assert np.all(result[SPE : 2 * SPE] == 0)  # epoch 0: W
    assert np.all(result[-SPE:] == -2)  # UNS after hypnogram end


def test_ts_hypno_longer_than_data(hyp_utc):
    # Recording starts 4 epochs into the hypnogram and is only 4 epochs long
    # → epochs 4-7: N2 N3 N3 REM
    raw = make_raw(4, meas_date=utc(23, 2))
    result = hyp_utc.upsample_to_data(raw)
    assert result.size == 4 * SPE
    assert np.all(result[:SPE] == 2)  # epoch 4: N2
    assert np.all(result[-SPE:] == 4)  # epoch 7: REM


def test_non_whole_epoch_offset_warns(caplog):
    # 45 s offset → 1.5 epochs at 30 s/epoch → non-whole → warning emitted
    hyp = Hypnogram(["W"] * 10, start="2024-01-01 23:00:00")
    raw = make_raw(
        10, meas_date=datetime.datetime(2024, 1, 1, 23, 0, 45, tzinfo=datetime.timezone.utc)
    )
    with caplog.at_level(logging.WARNING, logger="yasa"):
        hyp.upsample_to_data(raw)
    assert "not a whole number" in caplog.text


def test_ts_cropped_raw():
    # raw.crop() does not update meas_date, so the offset of the first sample (raw.first_time)
    # must be taken into account. Cropping 4 epochs → the data starts at epoch 4: N2
    hyp = Hypnogram(STAGES, start=HYP_START)
    raw = make_raw(N, meas_date=utc(23, 0)).crop(tmin=4 * 30)
    assert raw.info["meas_date"] == utc(23, 0)
    result = hyp.upsample_to_data(raw)
    assert result.size == 6 * SPE
    assert np.array_equal(result[::SPE], [2, 3, 3, 4, 4, 0])
