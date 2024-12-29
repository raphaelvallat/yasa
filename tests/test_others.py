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
