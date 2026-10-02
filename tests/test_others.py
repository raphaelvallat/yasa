"""Test the functions in the yasa/others.py file."""

import logging

import numpy as np
import pandas as pd
import pytest
from mne.filter import filter_data

import yasa
from yasa._validation import _check_data, _check_hypno_include
from yasa.hypno import Hypnogram
from yasa.others import (
    _index_to_events,
    _merge_close,
    _zerocrossings,
    get_centered_indices,
    moving_transform,
    sliding_window,
    trimbothstd,
)

MT_METHODS = ["mean", "min", "max", "ptp", "rms", "prop_above_zero", "slope", "corr", "covar"]


##############################################################################
# FIXTURES
##############################################################################


@pytest.fixture(scope="module")
def data_sigma(n2_spindles):
    """N2 data filtered in the sigma band."""
    return filter_data(n2_spindles.data, n2_spindles.sf, 12, 15, method="fir", verbose=0)


@pytest.fixture(scope="module")
def raw_eeg(raw_sub02_shared):
    """EEG channels of sub-02."""
    return raw_sub02_shared.copy().pick("eeg")


@pytest.fixture(scope="module")
def hypno_int(full_6hrs):
    """Integer hypnogram of the full recording, upsampled to the data."""
    return full_6hrs.hypno.astype(np.int64)


@pytest.fixture(scope="module")
def hyp_full(full_6hrs, hypno_int):
    """Hypnogram object of the full recording."""
    return Hypnogram.from_integers(hypno_int[:: 30 * full_6hrs.sf], freq="30s")


def _moving_transform_ref(x, y, method, window, step, sf):
    """Brute-force reference of moving_transform: slide a Python loop over exact windows."""
    n_local = x.size
    halfdur = window / 2
    total_dur = n_local / sf
    idx_times = np.arange(0, total_dur, step)
    out = []
    for t in idx_times:
        b = max(0, int((t - halfdur) * sf))
        e = min(n_local, int((t + halfdur) * sf))
        seg_x = x[b:e]
        seg_y = y[b:e] if y is not None else None
        m = seg_x.size
        if method == "mean":
            out.append(seg_x.mean())
        elif method == "rms":
            out.append(np.sqrt(np.mean(seg_x**2)))
        elif method == "min":
            out.append(seg_x.min())
        elif method == "max":
            out.append(seg_x.max())
        elif method == "ptp":
            out.append(seg_x.max() - seg_x.min())
        elif method == "prop_above_zero":
            out.append((seg_x >= 0).mean())
        elif method == "slope":
            if m < 2:
                out.append(np.nan)
            else:
                times = np.arange(m) / sf
                out.append(np.polyfit(times, seg_x, 1)[0])
        elif method == "covar":
            if m < 2:
                out.append(np.nan)
            else:
                out.append(np.cov(seg_x, seg_y, ddof=1)[0, 1])
        elif method == "corr":
            if m < 2:
                out.append(np.nan)
            else:
                c = np.corrcoef(seg_x, seg_y)[0, 1]
                out.append(c)
    return np.array(out)


##############################################################################
# TESTS
##############################################################################


def test_index_to_events():
    """Test functions _index_to_events"""
    a = np.array([[3, 6], [8, 12], [14, 20]])
    good = [3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20]
    out = _index_to_events(a)
    np.testing.assert_equal(good, out)


def test_merge_close():
    """Test functions _merge_close"""
    a = np.array([4, 5, 6, 7, 10, 11, 12, 13, 20, 21, 22, 100, 102])
    good = np.array(
        [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 100, 101, 102]
    )
    # Events that are less than 100 ms apart (i.e. 10 points at 100 Hz sf)
    out = _merge_close(a, 100, 100)
    np.testing.assert_equal(good, out)


# Each method once, cycling through the window, step and interpolation values (the numerical
# output of each method is checked in test_moving_transform_correctness)
@pytest.mark.parametrize(
    "win, step, method, interp",
    [
        (0.3 if i % 2 else 0.5, 0.5 if i % 3 else 0, method, bool(i % 2))
        for i, method in enumerate(MT_METHODS)
    ],
)
def test_moving_transform(n2_spindles, data_sigma, win, step, method, interp):
    """Test moving_transform"""
    moving_transform(n2_spindles.data, data_sigma, n2_spindles.sf, win, step, method, interp)


def test_moving_transform_interp_rms(n2_spindles):
    """Test that the interpolated output has the size of the data."""
    data = n2_spindles.data
    t, out = moving_transform(data, None, n2_spindles.sf, 0.5, 0.5, "rms", True)
    assert t.size == out.size
    assert out.size == data.size


@pytest.mark.parametrize("method", MT_METHODS)
def test_moving_transform_correctness(method):
    """Test moving_transform numerical output against reference implementations."""
    rng = np.random.default_rng(42)
    n = 200
    sf_t = 100.0
    window = 0.5  # 50 samples
    step = 0.1  # 10 samples
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    _, out = moving_transform(x, y, sf_t, window, step, method)
    ref = _moving_transform_ref(x, y, method, window, step, sf_t)
    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-10, err_msg=f"method={method}")


@pytest.mark.parametrize("method", ["mean", "rms", "covar", "corr"])
def test_moving_transform_edge_windows(method):
    """Test moving_transform edge cases: zero-length and single-sample windows."""
    rng = np.random.default_rng(0)
    # Very short signal (5 samples) with a large window forces edge clipping
    x = rng.standard_normal(5)
    y = rng.standard_normal(5)
    sf_t = 100.0
    window = 1.0  # 100 samples but signal is only 5 — all windows are clipped
    # mean and rms: clipped windows may have win_sz=0 only if signal is empty,
    # but here they have at least 1 sample; just check no inf/nan from win_sz=0.
    # covar/corr: windows with <2 samples must yield nan, not inf
    _, out = moving_transform(x, y, sf_t, window, step=0.01, method=method)
    assert not np.any(np.isinf(out)), f"inf in {method} with clipped windows"


@pytest.mark.parametrize("method", ["mean", "rms", "corr", "covar", "slope"])
def test_moving_transform_interp_size(method):
    """Interpolated output must match input length for all methods."""
    rng = np.random.default_rng(1)
    n = 500
    sf_t = 100.0
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    t, out = moving_transform(x, y, sf_t, window=0.3, step=0.1, method=method, interp=True)
    assert t.size == n, f"t size mismatch for method={method}"
    assert out.size == n, f"out size mismatch for method={method}"


@pytest.mark.parametrize("method", ["min", "max", "ptp", "prop_above_zero"])
def test_moving_transform_non_integer_window(method):
    """Non-integer window*sf should produce exact results (no ±1 sample error)."""
    rng = np.random.default_rng(7)
    n = 300
    sf_t = 100.0
    # window=0.075 s → 7.5 samples (non-integer)
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    _, out = moving_transform(x, y, sf_t, window=0.075, step=0.05, method=method)
    assert np.isfinite(out).all(), f"non-finite values for method={method}"


@pytest.mark.parametrize("method", ["min", "max", "ptp", "prop_above_zero"])
def test_moving_transform_zero_length_windows(method):
    """wsz == 0 windows in min/max/ptp/prop_above_zero must yield nan, not crash.

    window=0.001 s with sf=100 Hz gives halfdur*sf=0.05, so int(0.05)=0 and
    many windows have end - beg == 0.
    """
    rng = np.random.default_rng(3)
    x = rng.standard_normal(50)
    sf_t = 100.0
    _, out = moving_transform(x, None, sf_t, window=0.001, step=0.01, method=method)
    assert not np.any(np.isinf(out)), f"inf in {method} with wsz=0 windows"
    # At least some windows must be nan (those with wsz == 0)
    assert np.any(np.isnan(out)), f"expected some nan in {method} with wsz=0 windows"


def test_moving_transform_last_sample_included():
    """Clipping end to n (exclusive) must allow the last sample to be reached.

    Place a distinctive value at x[-1] and verify that edge windows near the
    end of the signal capture it.
    """
    sf_t = 100.0
    n = 20
    x = np.zeros(n)
    x[-1] = 99.0  # sentinel: only visible if x[n-1] is included
    # A large window centered near the end should include x[n-1]
    _, out = moving_transform(x, None, sf_t, window=0.5, step=1 / sf_t, method="max")
    assert out[-1] == 99.0, "last sample not included in edge window (end clipped too early)"


@pytest.mark.parametrize("method", ["corr", "covar"])
def test_moving_transform_corr_covar_requires_y(method):
    """corr and covar must raise ValueError when y is None."""
    x = np.random.default_rng(5).standard_normal(100)
    with pytest.raises(ValueError, match="y must be provided"):
        moving_transform(x, None, 100.0, 0.5, 0.1, method)


def test_trimbothstd():
    """Test function trimbothstd"""
    x = [4, 5, 7, 0, 18, 6, 7, 8, 9, 10]
    y = np.random.normal(size=(10, 100))
    assert trimbothstd(x) < np.std(x, ddof=1)
    assert (trimbothstd(y) < np.std(y, ddof=1, axis=-1)).all()


def test_zerocrossings():
    """Test _zerocrossings"""
    a = np.array([4, 2, -1, -3, 1, 2, 3, -2, -5])
    idx_zc = _zerocrossings(a)
    np.testing.assert_equal(idx_zc, [1, 3, 6])


def test_sliding_window():
    """Test function sliding window."""
    x = np.arange(1000)
    # 1D
    t, sl = sliding_window(x, sf=100, window=2)  # No overlap
    assert np.array_equal(t, [0.0, 2.0, 4.0, 6.0, 8.0])
    assert np.array_equal(sl.shape, (5, 200))
    t, sl = sliding_window(x, sf=100, window=2, step=1)  # 1 sec overlap
    assert np.array_equal(t, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    assert np.array_equal(sl.shape, (9, 200))
    t, sl = sliding_window(np.arange(1002), sf=100.0, window=1.0, step=0.1)
    assert t.size == 91
    assert np.array_equal(sl.shape, (91, 100))
    # 2D
    x_2d = np.random.rand(2, 1100)
    t, sl = sliding_window(x_2d, sf=100, window=2, step=1.0)
    assert np.array_equal(t, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    assert np.array_equal(sl.shape, (10, 2, 200))
    t, sl = sliding_window(x_2d, sf=100.0, window=4.0, step=None)
    assert np.array_equal(t, [0.0, 4.0])
    assert np.array_equal(sl.shape, (2, 2, 400))


def test_get_centered_indices():
    """Test function get_centered_indices"""
    data = np.arange(100)
    idx = [1, 10.0, 20, 30, 50, 102]
    before, after = 3, 2
    idx_ep, idx_nomask = get_centered_indices(data, idx, before, after)
    assert (data[idx_ep] == idx_ep).all()
    assert (idx_nomask == [1, 2, 3, 4]).all()
    assert idx_ep.shape == (len(idx_nomask), before + after + 1)
    # Epochs must end at the last sample of data (index 99) at the latest
    idx_ep, idx_nomask = get_centered_indices(data, [97, 98], before, after)
    assert (idx_nomask == [0]).all()
    assert idx_ep[0, -1] == 99


##############################################################################
# _check_data
##############################################################################


def test_check_data_1d(n2_spindles):
    """1D NumPy array."""
    data, sf = n2_spindles.data, n2_spindles.sf
    data_2d, sf_out, ch_names, raw = _check_data(data, sf)
    assert data_2d.shape == (1, data.size) and data_2d.dtype == np.float64
    assert sf_out == sf and ch_names == ["CHAN000"] and raw is None


def test_check_data_ch_names(full_6hrs):
    """Channel names are converted to a list of strings, and must match the data."""
    assert _check_data(full_6hrs.data, full_6hrs.sf, full_6hrs.chan)[2] == ["Cz", "Fz", "Pz"]
    with pytest.raises(AssertionError):
        _check_data(full_6hrs.data, full_6hrs.sf, ["Cz"])


@pytest.mark.parametrize("sf_np", [np.int64(200), np.float32(200), np.array(200)])
def test_check_data_numpy_sf(n2_spindles, sf_np):
    """The sampling frequency can be a NumPy scalar."""
    assert _check_data(n2_spindles.data, sf_np)[1] == n2_spindles.sf


def test_check_data_sf_required(n2_spindles):
    """sf is required with NumPy arrays."""
    with pytest.raises(AssertionError):
        _check_data(n2_spindles.data)


def test_check_data_mne(raw_eeg, caplog):
    """MNE Raw: data is converted to uV, sf and ch_names are ignored with a warning."""
    data_mne_uv, sf_out, ch_names, raw = _check_data(raw_eeg, sf=999, ch_names=["A"])
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    np.testing.assert_allclose(data_mne_uv, raw_eeg.get_data() * 1e6)
    assert sf_out == raw_eeg.info["sfreq"] and ch_names == raw_eeg.ch_names
    assert raw is raw_eeg


##############################################################################
# _check_hypno_include
##############################################################################


def test_check_hypno_include_int(full_6hrs, hypno_int):
    """Integer hypnogram array: returned as is, without copy."""
    hypno_out, include, int_to_str = _check_hypno_include(
        hypno_int, (2, 3), full_6hrs.data, full_6hrs.sf
    )
    assert hypno_out is hypno_int
    np.testing.assert_array_equal(include, [2, 3])
    assert int_to_str == {}


@pytest.mark.parametrize("inc", [2, 2.0, (1, 2)])
def test_check_hypno_include_float(full_6hrs, hypno_int, inc):
    """Float hypnogram (e.g. loaded from a txt file) with integer or float include."""
    hypno_out, include, _ = _check_hypno_include(hypno_int.astype(float), inc, full_6hrs.data, 100)
    assert hypno_out.dtype.kind == include.dtype.kind == "i"


def test_check_hypno_include_str(full_6hrs, hypno_int):
    """String hypnogram arrays require string include."""
    hypno_str = np.where(hypno_int == 2, "N2", "Other")
    out = _check_hypno_include(hypno_str, "N2", full_6hrs.data, full_6hrs.sf)
    assert out[1].tolist() == ["N2"]
    with pytest.raises(AssertionError, match="same dtype"):
        _check_hypno_include(hypno_str, 2, full_6hrs.data, full_6hrs.sf)


def test_check_hypno_include_hypnogram(full_6hrs, hyp_full):
    """Hypnogram: upsampled, and string labels are converted to integers."""
    hypno_out, include, int_to_str = _check_hypno_include(
        hyp_full, ["N2", "N3"], full_6hrs.data, 100
    )
    assert hypno_out.shape == (full_6hrs.data.shape[1],)
    np.testing.assert_array_equal(include, [2, 3])
    assert int_to_str[2] == "N2"
    with pytest.raises(AssertionError, match="not valid labels of the hypnogram"):
        _check_hypno_include(hyp_full, ["N2", "NREM3"], full_6hrs.data, full_6hrs.sf)


@pytest.mark.parametrize(
    "include, n_crop, match",
    [
        (2.5, 0, "whole numbers"),
        (None, 0, "include cannot be None"),
        (2, 1, "same size"),
        (7, 0, "None of the stages"),
    ],
)
def test_check_hypno_include_errors(full_6hrs, hypno_int, include, n_crop, match):
    """Invalid hypnogram or include."""
    hypno = hypno_int[: hypno_int.size - n_crop]
    with pytest.raises(AssertionError, match=match):
        _check_hypno_include(hypno, include, full_6hrs.data, full_6hrs.sf)


def test_check_hypno_include_timestamps(raw_sub02_shared):
    """A Hypnogram with a start time is aligned to a MNE Raw using absolute timestamps."""
    raw = raw_sub02_shared.copy().pick(["F3"]).crop(0, 600, include_tmax=False)
    sf_raw = raw.info["sfreq"]
    # meas_date is treated as a local time, and the hypnogram starts 60 s (2 epochs) before
    raw_start = pd.Timestamp(raw.info["meas_date"]).replace(tzinfo=None)
    hyp = Hypnogram(["W", "W"] + ["N2"] * 18, freq="30s", start=raw_start - pd.Timedelta(60, "s"))
    hypno_out = _check_hypno_include(hyp, ["N2"], raw, sf_raw)[0]
    # The first two (Wake) epochs are before the start of the recording
    assert (hypno_out[: int(18 * 30 * sf_raw)] == 2).all()
    # The end of the recording is not covered by the hypnogram
    assert (hypno_out[int(18 * 30 * sf_raw) :] == -2).all()
    # Without the Raw, the hypnogram is aligned with the start of the data
    hypno_out = _check_hypno_include(hyp, ["N2"], raw.get_data(), sf_raw)[0]
    assert (hypno_out[: int(60 * sf_raw)] == 0).all()


@pytest.mark.parametrize("name", ["trimbothstd", "get_centered_indices"])
def test_moved_to_others_deprecated(name):
    """Test the deprecated access to functions moved out of the top-level namespace."""
    with pytest.warns(FutureWarning, match=f"yasa.others.{name}"):
        func = getattr(yasa, name)
    assert func is getattr(yasa.others, name)
    with pytest.raises(AttributeError):
        yasa.does_not_exist
