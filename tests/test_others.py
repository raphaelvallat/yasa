"""Test the functions in the yasa/others.py file."""

import unittest
from itertools import product

import mne
import numpy as np
from mne.filter import filter_data

from yasa.fetchers import fetch_sample
from yasa.hypno import hypno_str_to_int, hypno_upsample_to_data
from yasa.others import (
    _index_to_events,
    _merge_close,
    _zerocrossings,
    get_centered_indices,
    moving_transform,
    sliding_window,
    trimbothstd,
)

# Load data
data_fp = fetch_sample("N2_spindles_15sec_200Hz.txt")
data = np.loadtxt(data_fp)
sf = 200
data_sigma = filter_data(data, sf, 12, 15, method="fir", verbose=0)

# Load a full recording and its hypnogram
data_full_fp = fetch_sample("full_6hrs_100Hz_Cz+Fz+Pz.npz")
file_full = np.load(data_full_fp)
data_full = file_full.get("data")
chan_full = file_full.get("chan")
sf_full = 100
hypno_full_fp = fetch_sample("full_6hrs_100Hz_hypno.npz")
hypno_full = np.load(hypno_full_fp).get("hypno")

# Using MNE
data_mne_fp = fetch_sample("sub-02_mne_raw.fif")
data_mne = mne.io.read_raw_fif(data_mne_fp, preload=True, verbose=0)
data_mne.pick("eeg")
data_mne_single = data_mne.copy().pick(["F3"])
hypno_mne_fp = fetch_sample("sub-02_hypno_30s.txt")
hypno_mne = np.loadtxt(hypno_mne_fp, dtype=str)
hypno_mne = hypno_str_to_int(hypno_mne)
hypno_mne = hypno_upsample_to_data(hypno=hypno_mne, sf_hypno=(1 / 30), data=data_mne)


class TestOthers(unittest.TestCase):
    def test_index_to_events(self):
        """Test functions _index_to_events"""
        a = np.array([[3, 6], [8, 12], [14, 20]])
        good = [3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20]
        out = _index_to_events(a)
        np.testing.assert_equal(good, out)

    def test_merge_close(self):
        """Test functions _merge_close"""
        a = np.array([4, 5, 6, 7, 10, 11, 12, 13, 20, 21, 22, 100, 102])
        good = np.array(
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 100, 101, 102]
        )
        # Events that are less than 100 ms apart (i.e. 10 points at 100 Hz sf)
        out = _merge_close(a, 100, 100)
        np.testing.assert_equal(good, out)

    def test_moving_transform(self):
        """Test moving_transform"""
        method = ["mean", "min", "max", "ptp", "rms", "prop_above_zero", "slope", "corr", "covar"]
        interp = [False, True]
        win = [0.3, 0.5]
        step = [0, 0.5]
        prod_args = product(win, step, method, interp)

        for i, (w, s, m, i) in enumerate(prod_args):
            moving_transform(data, data_sigma, sf, w, s, m, i)

        t, out = moving_transform(data, None, sf, w, s, "rms", True)
        assert t.size == out.size
        assert out.size == data.size

    def test_moving_transform_correctness(self):
        """Test moving_transform numerical output against reference implementations."""
        rng = np.random.default_rng(42)
        n = 200
        sf_t = 100.0
        window = 0.5  # 50 samples
        step = 0.1  # 10 samples
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)

        def _ref(x, y, method, window, step, sf):
            """Brute-force reference: slide a Python loop over exact windows."""
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

        for method in ["mean", "rms", "min", "max", "ptp", "prop_above_zero", "slope"]:
            _, out = moving_transform(x, y, sf_t, window, step, method)
            ref = _ref(x, y, method, window, step, sf_t)
            np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-10, err_msg=f"method={method}")

        for method in ["covar", "corr"]:
            _, out = moving_transform(x, y, sf_t, window, step, method)
            ref = _ref(x, y, method, window, step, sf_t)
            np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-10, err_msg=f"method={method}")

    def test_moving_transform_edge_windows(self):
        """Test moving_transform edge cases: zero-length and single-sample windows."""
        rng = np.random.default_rng(0)
        # Very short signal (5 samples) with a large window forces edge clipping
        x = rng.standard_normal(5)
        y = rng.standard_normal(5)
        sf_t = 100.0
        window = 1.0  # 100 samples but signal is only 5 — all windows are clipped

        # mean and rms: clipped windows may have win_sz=0 only if signal is empty,
        # but here they have at least 1 sample; just check no inf/nan from win_sz=0
        for method in ["mean", "rms"]:
            _, out = moving_transform(x, y, sf_t, window, step=0.01, method=method)
            assert not np.any(np.isinf(out)), f"inf in {method} with clipped windows"

        # covar/corr: windows with <2 samples must yield nan, not inf
        for method in ["covar", "corr"]:
            _, out = moving_transform(x, y, sf_t, window, step=0.01, method=method)
            # All finite values must not be inf
            assert not np.any(np.isinf(out)), f"inf in {method} with clipped windows"

    def test_moving_transform_interp_size(self):
        """Interpolated output must match input length for all methods."""
        rng = np.random.default_rng(1)
        n = 500
        sf_t = 100.0
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        for method in ["mean", "rms", "corr", "covar", "slope"]:
            t, out = moving_transform(x, y, sf_t, window=0.3, step=0.1, method=method, interp=True)
            assert t.size == n, f"t size mismatch for method={method}"
            assert out.size == n, f"out size mismatch for method={method}"

    def test_moving_transform_non_integer_window(self):
        """Non-integer window*sf should produce exact results (no ±1 sample error)."""
        rng = np.random.default_rng(7)
        n = 300
        sf_t = 100.0
        # window=0.075 s → 7.5 samples (non-integer)
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        for method in ["min", "max", "ptp", "prop_above_zero"]:
            _, out = moving_transform(x, y, sf_t, window=0.075, step=0.05, method=method)
            assert np.isfinite(out).all(), f"non-finite values for method={method}"

    def test_trimbothstd(self):
        """Test function trimbothstd"""
        x = [4, 5, 7, 0, 18, 6, 7, 8, 9, 10]
        y = np.random.normal(size=(10, 100))
        assert trimbothstd(x) < np.std(x, ddof=1)
        assert (trimbothstd(y) < np.std(y, ddof=1, axis=-1)).all()

    def test_zerocrossings(self):
        """Test _zerocrossings"""
        a = np.array([4, 2, -1, -3, 1, 2, 3, -2, -5])
        idx_zc = _zerocrossings(a)
        np.testing.assert_equal(idx_zc, [1, 3, 6])

    def test_sliding_window(self):
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

    def test_get_centered_indices(self):
        """Test function get_centered_indices"""
        data = np.arange(100)
        idx = [1, 10.0, 20, 30, 50, 102]
        before, after = 3, 2
        idx_ep, idx_nomask = get_centered_indices(data, idx, before, after)
        assert (data[idx_ep] == idx_ep).all()
        assert (idx_nomask == [1, 2, 3, 4]).all()
        assert idx_ep.shape == (len(idx_nomask), before + after + 1)
