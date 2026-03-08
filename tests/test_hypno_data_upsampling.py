"""Tests for hypnogram upsampling: legacy helpers and Hypnogram.upsample_to_data.

Covers all combinations of:
  - data type : NumPy array | MNE Raw without meas_date | MNE Raw with meas_date
  - Hypnogram : no start | naive start | tz-aware start (via tz=) | tz-aware datetime

Default behaviour (meas_date_is_local=True): meas_date is treated as a local absolute
timestamp, consistent with the EDF+ standard, which defines starttime as local time at
the patient's location. MNE reads this and tags it as UTC; YASA corrects this by default.
Both meas_date and Hypnogram.start are compared as absolute values, so no tz is required.

Set meas_date_is_local=False only for EDF files that genuinely store UTC in meas_date.
"""

import datetime

import mne
import numpy as np
import pandas as pd
import pytest

from yasa.hypno import (
    Hypnogram,
    hypno_fit_to_data,
    hypno_upsample_to_data,
    hypno_upsample_to_sf,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SF = 100  # EEG sampling frequency (Hz)
SPE = SF * 30  # samples per 30-second epoch = 3000

# 9-epoch integer array used by the legacy helper tests (same as original test suite)
HYPNO_INT = np.array([0, 0, 0, 1, 2, 2, 3, 3, 4])

# 10-epoch string hypnogram used by Hypnogram class tests
#   stages : W  W  N1  N2  N2  N3  N3  REM  REM  W
#   ints   : 0  0   1   2   2   3   3    4    4  0
STAGES = ["W", "W", "N1", "N2", "N2", "N3", "N3", "REM", "REM", "W"]
N = len(STAGES)  # 10

# Reference hypnogram start used across timestamp tests
HYP_START = "2024-01-15 23:00:00"  # naive string, represents local time

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Legacy helper: hypno_upsample_to_sf
# ---------------------------------------------------------------------------


def test_upsample_to_sf_basic():
    up = hypno_upsample_to_sf(HYPNO_INT, sf_hypno=1 / 30, sf_data=SF)
    assert up.size == HYPNO_INT.size * SPE
    # each epoch is repeated exactly SPE times
    assert up[up == 2].size == (HYPNO_INT == 2).sum() * SPE


def test_upsample_to_sf_passthrough():
    # sf_hypno == sf_data → output identical to input
    assert np.array_equal(hypno_upsample_to_sf(HYPNO_INT, sf_hypno=1, sf_data=1), HYPNO_INT)


# ---------------------------------------------------------------------------
# Legacy helper: hypno_fit_to_data
# ---------------------------------------------------------------------------


@pytest.fixture
def hypno100():
    return hypno_upsample_to_sf(HYPNO_INT, sf_hypno=1 / 30, sf_data=SF)


@pytest.mark.parametrize(
    "data",
    [
        np.zeros(HYPNO_INT.size * SPE),  # numpy array
        pytest.param("raw_exact", id="raw"),  # MNE Raw (resolved in test body)
    ],
)
def test_fit_exact(hypno100, data):
    if isinstance(data, str) and data == "raw_exact":
        data = make_raw(HYPNO_INT.size)
    assert np.array_equal(hypno_fit_to_data(hypno100, data), hypno100)


def test_fit_pads_when_shorter(hypno100):
    # hypno shorter than data → last value repeated at the end
    assert (
        hypno_fit_to_data(hypno100, make_raw(HYPNO_INT.size + 1)).size == (HYPNO_INT.size + 1) * SPE
    )
    assert (
        hypno_fit_to_data(hypno100, np.zeros((HYPNO_INT.size + 1) * SPE)).size
        == (HYPNO_INT.size + 1) * SPE
    )


def test_fit_crops_when_longer(hypno100):
    # hypno longer than data → trailing epochs removed
    assert (
        hypno_fit_to_data(hypno100, make_raw(HYPNO_INT.size - 1)).size == (HYPNO_INT.size - 1) * SPE
    )
    assert (
        hypno_fit_to_data(hypno100, np.zeros((HYPNO_INT.size - 1) * SPE)).size
        == (HYPNO_INT.size - 1) * SPE
    )


# ---------------------------------------------------------------------------
# Legacy helper: hypno_upsample_to_data (upsample + fit combined)
# ---------------------------------------------------------------------------


def test_upsample_to_data_with_raw():
    assert (
        hypno_upsample_to_data(HYPNO_INT, sf_hypno=1 / 30, data=make_raw(HYPNO_INT.size - 1)).size
        == (HYPNO_INT.size - 1) * SPE
    )


def test_upsample_to_data_with_array():
    assert (
        hypno_upsample_to_data(HYPNO_INT, sf_hypno=1 / 30, data=np.zeros(27250), sf_data=100).size
        == 27250
    )


def test_upsample_to_data_different_sf():
    # double SF → double samples, same padding logic
    n_expected = 2 * (HYPNO_INT.size * SPE + 250)
    assert (
        hypno_upsample_to_data(
            HYPNO_INT, sf_hypno=1 / 30, data=np.zeros(n_expected), sf_data=200
        ).size
        == n_expected
    )


# ---------------------------------------------------------------------------
# Hypnogram.upsample_to_data — length-based path
#
# Triggered when EITHER:
#   - data is a NumPy array, OR
#   - data is MNE Raw without meas_date, OR
#   - data is MNE Raw with meas_date but Hypnogram has no start
#
# In all three cases the behaviour is identical: align at t=0, crop/pad at end.
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=["array", "raw_no_meas", "raw_with_meas"],
    ids=["array", "raw_no_meas", "raw_with_meas"],
)
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


# ---------------------------------------------------------------------------
# Hypnogram start / tz construction
# ---------------------------------------------------------------------------


def test_start_tz_string():
    # naive string + tz → stored as tz-aware Timestamp in local time
    hyp = Hypnogram(STAGES, freq="30s", start=HYP_START, tz="Europe/Paris")
    assert hyp.start == pd.Timestamp(HYP_START, tz="Europe/Paris")


def test_start_tz_aware_datetime():
    # passing a tz-aware datetime directly → tz= not needed
    aware_dt = datetime.datetime(2024, 1, 15, 23, 0, tzinfo=datetime.timezone.utc)
    hyp = Hypnogram(STAGES, freq="30s", start=aware_dt)
    assert hyp.start == pd.Timestamp("2024-01-15 23:00:00", tz="UTC")


def test_start_tz_conflict_raises():
    # tz-aware datetime + tz= → ValueError
    aware_dt = datetime.datetime(2024, 1, 15, 23, 0, tzinfo=datetime.timezone.utc)
    with pytest.raises(ValueError, match="already timezone-aware"):
        Hypnogram(STAGES, freq="30s", start=aware_dt, tz="UTC")


# ---------------------------------------------------------------------------
# Hypnogram.upsample_to_data — timestamp-aware path
#
# Triggered when BOTH self.start is set AND raw.meas_date is set.
# The hypnogram epochs are selected by absolute timestamp offset, not sample count.
#
# Hypnogram : 10 epochs starting at 23:00 (local time)
#   index:  0   1   2    3    4    5    6     7     8   9
#   stage:  W   W   N1   N2   N2   N3   N3   REM   REM  W
#   int:    0   0    1    2    2    3    3     4     4   0
# ---------------------------------------------------------------------------


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
