"""Test the functions in yasa/spectral.py."""

import unittest
from itertools import product

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pytest
from mne.filter import filter_data

from yasa.detection import (
    _check_data_hypno,
    art_detect,
    compare_detection,
    rem_detect,
    spindles_detect,
    sw_detect,
)
from yasa.fetchers import fetch_sample
from yasa.hypno import Hypnogram, hypno_int_to_str, hypno_str_to_int, hypno_upsample_to_data

##############################################################################
# DATA LOADING
##############################################################################

# For all data, the sampling frequency is always 100 Hz
sf = 100

# 1) Single channel, we take one every other point to keep a sf of 100 Hz
data_fp = fetch_sample("N2_spindles_15sec_200Hz.txt")
data = np.loadtxt(data_fp)[::2]
data_sigma = filter_data(data, sf, 12, 15, method="fir", verbose=0)

# Load an extract of N3 sleep without any spindle
data_n3_fp = fetch_sample("N3_no-spindles_30sec_100Hz.txt")
data_n3 = np.loadtxt(data_n3_fp)

# 2) Multi-channel
# Load a full recording and its hypnogram
data_full_fp = fetch_sample("full_6hrs_100Hz_Cz+Fz+Pz.npz")
hypno_full_fp = fetch_sample("full_6hrs_100Hz_hypno.npz")
_data_full_raw = np.load(data_full_fp).get("data")
chan_full = np.load(data_full_fp).get("chan")
hypno_full = np.load(hypno_full_fp).get("hypno")

# Keep only Fz and during a N3 sleep period with (huge) slow-waves
# (extracted before truncation since the N3 segment is at ~1h51m)
data_sw = _data_full_raw[1, 666000:672000].astype(np.float64)
hypno_sw = hypno_full[666000:672000]

# Truncate to 3 hours for faster multi-channel tests (~2x speedup).
# The N3 segment used for data_sw at 666k–672k samples is still included.
_N = 1_080_000  # 3 hours at 100 Hz
data_full = _data_full_raw[:, :_N]
hypno_full = hypno_full[:_N]

# Let's add a channel with bad data amplitude
chan_full = np.append(chan_full, "Bad")  # ['Cz', 'Fz', 'Pz', 'Bad']
data_full = np.vstack((data_full, data_full[-1, :] * 1e8))

# MNE Raw
data_mne_fp = fetch_sample("sub-02_mne_raw.fif")
data_mne = mne.io.read_raw_fif(data_mne_fp, preload=True, verbose=0)
data_mne.pick("eeg")
data_mne_single = data_mne.copy().pick(["F3"])
hypno_mne_fp = fetch_sample("sub-02_hypno_30s.txt")
hypno_mne = np.loadtxt(hypno_mne_fp, dtype=str)
hypno_mne = hypno_str_to_int(hypno_mne)
hypno_mne = hypno_upsample_to_data(hypno=hypno_mne, sf_hypno=(1 / 30), data=data_mne)

# Hypnogram objects for testing Hypnogram-based hypno support
# Full-night 5-stage Hypnogram (30s epochs, 100 Hz data)
hypno_full_30s = hypno_full[:: int(sf * 30)]  # downsample to 1 value per 30s epoch
hyp_full = Hypnogram(hypno_int_to_str(hypno_full_30s), freq="30s")
# sub-02 Hypnogram (30s epochs, string stages already)
hypno_mne_str = np.loadtxt(fetch_sample("sub-02_hypno_30s.txt"), dtype=str)
hyp_mne = Hypnogram(hypno_mne_str, freq="30s")


class TestDetection(unittest.TestCase):
    """Unit tests for detection.py"""

    def test_check_data_hypno(self):
        """Test preprocessing of data and hypno."""
        with self.assertLogs("yasa", level="WARNING"):
            _check_data_hypno(data_mne, sf=999)  # sf is ignored
        with self.assertLogs("yasa", level="WARNING"):
            _check_data_hypno(data_mne, ch_names=["CH999"])  # ch_names is ignored

        # Test with Hypnogram instance + integer include (default behavior preserved)
        _, _, _, hypno_out, include_out, mask, _, n_samples, _ = _check_data_hypno(
            data_full[1, :], sf, hypno=hyp_full, include=(2, 3)
        )
        assert hypno_out.shape == (n_samples,)
        assert set(include_out) == {2, 3}

        # Test with Hypnogram instance + string include
        _, _, _, hypno_out2, include_out2, _, _, _, _ = _check_data_hypno(
            data_full[1, :], sf, hypno=hyp_full, include=["N2", "N3"]
        )
        np.testing.assert_array_equal(include_out, include_out2)

        # Test with Hypnogram instance + single string include
        _, _, _, _, include_out3, _, _, _, _ = _check_data_hypno(
            data_full[1, :], sf, hypno=hyp_full, include="REM"
        )
        np.testing.assert_array_equal(include_out3, [4])

        # Test with MNE raw + Hypnogram
        _, _, _, hypno_out_mne, _, _, _, n_mne, _ = _check_data_hypno(
            data_mne, hypno=hyp_mne, include=["N2", "N3"]
        )
        assert hypno_out_mne.shape == (n_mne,)

    def test_spindles_detect(self):
        """Test spindles_detect"""
        #######################################################################
        # SINGLE CHANNEL
        #######################################################################
        # Representative parameter combinations (covers all values, avoids full
        # Cartesian product of 24 iterations).
        param_combos_sp = [
            ((11, 16), (0.5, 30), (0.3, 2.5), None),
            ([12, 14], (0.5, 30), (0.3, 2.5), 0),
            ((11, 16), [1, 25], (0.3, 2.5), 500),
            ([12, 14], [1, 25], [0.5, 3], None),
            ((11, 16), (0.5, 30), [0.5, 3], 0),
            ([12, 14], [1, 25], [0.5, 3], 500),
        ]
        for s, b, d, m in param_combos_sp:
            spindles_detect(data, sf, freq_sp=s, duration=d, freq_broad=b, min_distance=m)

        sp = spindles_detect(data, sf, verbose=True)
        assert sp.summary().shape[0] == 2
        sp.get_mask()
        sp.get_sync_events()
        sp.get_sync_events(time_before=10)  # Invalid time window
        sp.plot_average(errorbar=None, filt=(None, 30))  # Skip bootstrapping
        np.testing.assert_array_equal(np.squeeze(sp._data), data)
        # Compare channels return dataframe with single cell
        assert sp.compare_channels().shape == (1, 1)
        assert sp._sf == sf
        sp.summary(grp_chan=True, grp_stage=True, aggfunc="median", sort=False)

        # Test with custom thresholds
        spindles_detect(data, sf, thresh={"rel_pow": 0.25})
        spindles_detect(data, sf, thresh={"rms": 1.25})
        spindles_detect(data, sf, thresh={"rel_pow": 0.25, "corr": 0.60})

        # Test with disabled thresholds
        spindles_detect(data, sf, thresh={"rel_pow": None})
        spindles_detect(data, sf, thresh={"corr": None}, verbose="debug")
        spindles_detect(data, sf, thresh={"rms": None})
        spindles_detect(data, sf, thresh={"rms": None, "corr": None})
        spindles_detect(data, sf, thresh={"rms": None, "rel_pow": None})
        spindles_detect(data, sf, thresh={"corr": None, "rel_pow": None})

        # Test with hypnogram
        spindles_detect(data, sf, hypno=np.ones(data.size))

        # Test with 1-sec of flat data -- we should still have 2 detected spindles
        data_flat = data.copy()
        data_flat[100:200] = 1
        sp = spindles_detect(data_flat, sf).summary()
        assert sp.shape[0] == 2

        # Full night single channel with Isolation Forest + hypnogram
        sp = spindles_detect(data_full[1, :], sf, hypno=hypno_full)
        sp_no_out = spindles_detect(data_full[1, :], sf, hypno=hypno_full, remove_outliers=True)
        assert sp.compare_detection(sp_no_out).shape[0] == 1

        # Calculate the coincidence matrix with only one channel
        with pytest.raises(ValueError):
            sp.get_coincidence_matrix()

        # compare_detection with invalid other
        with pytest.raises(ValueError):
            sp.compare_detection(other="WRONG")

        with self.assertLogs("yasa", level="WARNING"):
            spindles_detect(data_n3, sf)
        # assert sp is None --> Fails?

        # Ensure that the two warnings are tested
        with self.assertLogs("yasa", level="WARNING"):
            sp = spindles_detect(data_n3, sf, thresh={"corr": 0.95})
        assert sp is None

        # Test with wrong data amplitude (1)
        with self.assertLogs("yasa", level="ERROR"):
            sp = spindles_detect(data_n3 / 1e6, sf)
        assert sp is None

        # Test with wrong data amplitude (2)
        with self.assertLogs("yasa", level="ERROR"):
            sp = spindles_detect(data_n3 * 1e6, sf)
        assert sp is None

        # Test with a random array
        # with self.assertLogs("yasa", level="ERROR"):
        #     np.random.seed(123)
        #     sp = spindles_detect(np.random.random(size=1000), sf)
        # assert sp is None

        # No values in hypno intersect with include
        with pytest.raises(AssertionError):
            sp = spindles_detect(data, sf, include=2, hypno=np.zeros(data.size, dtype=int))

        #######################################################################
        # MULTI CHANNEL
        #######################################################################

        sp = spindles_detect(data_full, sf, chan_full)
        sp.get_mask()
        sp.get_sync_events(filt=(12, 15))
        sp.summary()
        sp.summary(grp_chan=True)
        sp.plot_average(errorbar=None)
        sp.get_coincidence_matrix()
        sp.get_coincidence_matrix(scaled=False)
        sp.plot_detection()
        assert sp._data.shape == sp._data_filt.shape
        np.testing.assert_array_equal(sp._data, data_full)
        assert sp._sf == sf
        sp_no_out = spindles_detect(data_full, sf, chan_full, remove_outliers=True)
        sp_multi = spindles_detect(data_full, sf, chan_full, multi_only=True)
        assert sp_multi.summary().shape[0] < sp.summary().shape[0]
        assert sp_no_out.summary().shape[0] < sp.summary().shape[0]

        # Test compare_detection
        assert (sp.compare_detection(sp)["f1"] == 1).all()  # self vs self, f1-score is 1
        sp_vs_multi = sp.compare_detection(sp_multi)
        # When comparing against sp_multi as the reference, we expect a perfect recall
        assert (sp_vs_multi["n_self"] > sp_vs_multi["n_other"]).all()
        assert (sp_vs_multi["recall"] == 1).all()
        # Setting `other_is_groundtruth=False`` == other.compare(self)
        sp_vs_multi_revert = sp.compare_detection(sp_multi, other_is_groundtruth=False)
        multi_vs_sp = sp_multi.compare_detection(sp)
        assert (sp_vs_multi_revert["recall"] == multi_vs_sp["recall"]).all()
        # With a look around and using the summary
        sp_vs_nout_1s = sp.compare_detection(sp_no_out.summary(), max_distance_sec=1)
        sp_vs_nout_2s = sp.compare_detection(sp_no_out.summary(), max_distance_sec=2)
        assert (sp_vs_nout_2s["f1"] > sp_vs_nout_1s["f1"]).all()
        assert (sp_vs_nout_2s["recall"] == sp_vs_nout_1s["recall"]).all()
        assert (sp_vs_nout_2s["n_self"] == sp_vs_nout_1s["n_self"]).all()

        # Test with hypnogram
        sp = spindles_detect(data_full, sf, hypno=hypno_full, include=2)
        sp.summary(grp_chan=False, grp_stage=False)
        sp.summary(grp_chan=False, grp_stage=True, aggfunc="median")
        sp.summary(grp_chan=True, grp_stage=False)
        sp.summary(grp_chan=True, grp_stage=True, sort=False)
        sp.plot_average(errorbar=None)
        sp.plot_average(hue="Stage", errorbar=None)
        sp.plot_detection()

        # Test compare_channels function
        # .. F1-score -- symmetric matrix
        mat = sp.compare_channels()
        assert mat.equals(mat.T)  # mat is a symmetric matrix
        assert (np.diag(mat) == 1).all()  # diagonal is all 1
        idx_triu = np.triu_indices_from(mat, k=1)
        mat_2s = sp.compare_channels(max_distance_sec=2)
        # Make sure that the overall scores are higher when using a lookaround
        assert mat.to_numpy()[idx_triu].mean() < mat_2s.to_numpy()[idx_triu].mean()
        # Precision / recall -- not symmetric
        mat_prec = sp.compare_channels(score="precision")
        mat_rec = sp.compare_channels(score="recall")
        assert mat_prec.T.equals(mat_rec)  # tril precision == triu recall

        # Using a MNE raw object (and disabling one threshold)
        spindles_detect(data_mne, thresh={"corr": None, "rms": 3})
        spindles_detect(data_mne, hypno=hypno_mne, include=2, verbose=True)

        # Test with Hypnogram instance (integer include, default)
        sp_hyp = spindles_detect(data_full[1, :], sf, hypno=hyp_full, include=(1, 2, 3))
        assert sp_hyp is not None
        # Test with Hypnogram instance + string include
        sp_hyp_str = spindles_detect(
            data_full[1, :], sf, hypno=hyp_full, include=["N1", "N2", "N3"]
        )
        assert sp_hyp_str is not None
        # Both should detect the same spindles
        assert sp_hyp.summary().shape[0] == sp_hyp_str.summary().shape[0]
        # Test with MNE raw + Hypnogram
        spindles_detect(data_mne, hypno=hyp_mne, include=["N2"], verbose=True)
        plt.close("all")

    def test_sw_detect(self):
        """Test function slow-wave detect"""
        # Representative parameter combinations (covers all values and None thresholds,
        # avoids full Cartesian product of 64 iterations).
        param_combos_sw = [
            ((0.3, 3.5), (0.3, 1.5), (0.3, 1.5), (40, 300), (10, 150), (75, 400)),
            ((0.5, 4), [0.1, 2], [0, 1], [40, None], (0, None), [80, 300]),
            ((0.3, 3.5), [0.1, 2], (0.3, 1.5), [40, None], (10, 150), [80, 300]),
            ((0.5, 4), (0.3, 1.5), [0, 1], (40, 300), (0, None), (75, 400)),
        ]
        for f, dn, dp, an, ap, aptp in param_combos_sw:
            sw_detect(
                data_sw, sf, freq_sw=f, dur_neg=dn, dur_pos=dp, amp_neg=an, amp_pos=ap, amp_ptp=aptp
            )

        # With N3 hypnogram
        sw = sw_detect(data_sw, sf, hypno=hypno_sw, coupling=True)
        sw.summary()
        sw.get_mask()
        sw.get_sync_events()
        sw.plot_average(errorbar=None)
        sw.plot_detection()
        np.testing.assert_array_equal(np.squeeze(sw._data), data_sw)
        np.testing.assert_array_equal(sw._hypno, hypno_sw)
        assert sw._sf == sf

        # Test with wrong data amplitude
        with self.assertLogs("yasa", level="ERROR"):
            sw = sw_detect(data_sw * 0, sf)  # All channels are flat
        assert sw is None

        # With 2D data
        sw_detect(data_sw[np.newaxis, ...], sf, verbose="INFO")

        # No values in hypno intersect with include
        with pytest.raises(AssertionError):
            sw = sw_detect(data_sw, sf, include=3, hypno=np.ones(data_sw.shape, dtype=int))

        #######################################################################
        # MULTI CHANNEL
        #######################################################################

        sw = sw_detect(data_full, sf, chan_full)
        sw.get_mask()
        sw.get_sync_events()
        sw.plot_average(errorbar=None)
        sw.plot_detection()
        sw.get_coincidence_matrix()
        sw.get_coincidence_matrix(scaled=False)
        # Test with outlier removal. There should be fewer events.
        sw_no_out = sw_detect(data_full, sf, chan_full, remove_outliers=True)
        assert sw_no_out._events.shape[0] < sw._events.shape[0]

        # Test compare_detection
        assert (sw.compare_detection(sw_no_out)["recall"] == 1).all()

        # Test with hypnogram
        sw = sw_detect(data_full, sf, chan_full, hypno=hypno_full, coupling=True)
        sw.summary(grp_chan=False, grp_stage=False)
        sw.summary(grp_chan=False, grp_stage=True, aggfunc="median")
        sw.summary(grp_chan=True, grp_stage=False)
        sw.summary(grp_chan=True, grp_stage=True, sort=False)
        sw.plot_average(errorbar=None)
        sw.plot_average(hue="Stage", errorbar=None)
        sw.plot_detection()
        # Check coupling
        sw_sum = sw.summary()
        assert "ndPAC" in sw_sum.columns
        # There should be some zero in the ndPAC (full dataframe)
        assert sw._events[sw._events["ndPAC"] == 0].shape[0] > 0
        # Coinciding spindles and masking
        sp = spindles_detect(data_full, sf, chan_full, hypno=hypno_full)
        sw.find_cooccurring_spindles(sp.summary())
        sw_sum = sw.summary()
        assert "CooccurringSpindle" in sw_sum.columns
        assert "DistanceSpindleToSW" in sw_sum.columns
        sw_sum_masked = sw.summary(
            grp_chan=True, grp_stage=False, mask=sw._events["CooccurringSpindle"]
        )
        assert sw_sum_masked.shape[0] < sw_sum.shape[0]

        # Test with different coupling params
        sw_detect(
            data_full,
            sf,
            chan_full,
            hypno=hypno_full,
            coupling=True,
            coupling_params={"freq_sp": (12, 16), "time": 2, "p": None},
        )

        # Using a MNE raw object
        sw_detect(data_mne)
        sw_detect(data_mne, hypno=hypno_mne, include=3)

        # Test with Hypnogram instance + integer include
        sw_hyp = sw_detect(data_full[1, :], sf, hypno=hyp_full, include=(2, 3))
        assert sw_hyp is not None
        # Test with Hypnogram instance + string include
        sw_hyp_str = sw_detect(data_full[1, :], sf, hypno=hyp_full, include=["N2", "N3"])
        assert sw_hyp_str is not None
        assert sw_hyp.summary().shape[0] == sw_hyp_str.summary().shape[0]
        # Test with MNE raw + Hypnogram
        sw_detect(data_mne, hypno=hyp_mne, include=["N3"])
        plt.close("all")

    def test_rem_detect(self):
        """Test function REM detect"""
        file_rem_fp = fetch_sample("EOGs_REM_256Hz.npz")
        file_rem = np.load(file_rem_fp)
        data_rem = file_rem["data"]
        loc, roc = data_rem[0, :], data_rem[1, :]
        sf_rem = file_rem["sf"]
        hypno_rem = 4 * np.ones_like(loc)

        # Parameters product testing
        freq_rem = [(0.5, 5), (0.3, 8)]
        duration = [(0.3, 1.5), [0.5, 1]]
        amplitude = [(50, 200), [60, 300]]
        hypno = [hypno_rem, None]
        prod_args = product(freq_rem, duration, amplitude, hypno)

        for i, (f, dr, am, h) in enumerate(prod_args):
            rem_detect(loc, roc, sf_rem, hypno=h, freq_rem=f, duration=dr, amplitude=am)

        # With isolation forest
        rem = rem_detect(loc, roc, sf, verbose="info")
        rem2 = rem_detect(loc, roc, sf, remove_outliers=True)
        assert rem.summary().shape[0] > rem2.summary().shape[0]
        rem.summary()
        rem.get_mask()
        rem.get_sync_events()
        rem.plot_average(errorbar=None)
        rem.plot_average(filt=(0.5, 5), errorbar=None)

        # With REM hypnogram
        hypno_rem = 4 * np.ones_like(loc)
        rem = rem_detect(loc, roc, sf, hypno=hypno_rem)
        hypno_rem = np.r_[np.ones(int(loc.size / 2)), 4 * np.ones(int(loc.size / 2))]
        rem2 = rem_detect(loc, roc, sf, hypno=hypno_rem)
        assert rem.summary().shape[0] > rem2.summary().shape[0]
        rem2.summary(grp_stage=True, aggfunc="median")

        # Test with wrong data amplitude on ROC
        with self.assertLogs("yasa", level="ERROR"):
            rem = rem_detect(loc * 1e-8, roc, sf)
        assert rem is None

        # Test with wrong data amplitude on LOC
        with self.assertLogs("yasa", level="ERROR"):
            rem = rem_detect(loc, roc * 1e8, sf)
        assert rem is None

        # No values in hypno intersect with include
        with pytest.raises(AssertionError):
            rem_detect(loc, roc, sf, hypno=hypno_rem, include=5)

        # Test with Hypnogram instance + string include
        hypno_rem_str = np.array(["REM"] * loc.size)
        hyp_rem = Hypnogram(hypno_rem_str, freq="1s")
        rem_hyp = rem_detect(loc, roc, sf_rem, hypno=hyp_rem, include="REM")
        assert rem_hyp is not None
        # Integer and string include should give the same result
        rem_int = rem_detect(loc, roc, sf_rem, hypno=hyp_rem, include=4)
        assert rem_hyp.summary().shape[0] == rem_int.summary().shape[0]

    def test_art_detect(self):
        """Test function art_detect"""
        file_9_fp = fetch_sample("full_6hrs_100Hz_9channels.npz")
        file_9 = np.load(file_9_fp)
        data_9 = file_9.get("data")
        hypno_9_fp = fetch_sample("full_6hrs_100Hz_hypno.npz")
        hypno_9 = np.load(hypno_9_fp).get("hypno")  # noqa
        # For the sake of the example, let's add some flat data at the end
        data_9 = np.concatenate((data_9, np.zeros((data_9.shape[0], 20000))), axis=1)
        hypno_9 = np.concatenate((hypno_9, np.zeros(20000)))

        # Start different combinations
        art_detect(data_9, sf=100, window=10, method="covar", threshold=3)
        art_detect(
            data_9, sf=100, window=6, hypno=hypno_9, include=(2, 3), method="covar", threshold=3
        )
        art_detect(data_9, sf=100, window=5, method="std", threshold=2)
        art_detect(
            data_9,
            sf=100,
            window=5,
            hypno=hypno_9,
            method="std",
            include=(0, 1, 2, 3, 4, 5, 6),
            threshold=2,
        )
        art_detect(
            data_9,
            sf=100,
            window=5.0,
            hypno=hypno_9,
            method="std",
            include=(0, 1, 2, 3, 4, 5, 6),
            threshold=10,
        )
        # Single channel
        art_detect(data_9[0], 100, window=10, method="covar")
        art_detect(data_9[0], 100, window=5, method="std", verbose=True)

        # Not enough epochs for stage
        hypno_9[:100] = 6
        art_detect(
            data_9,
            sf,
            window=5.0,
            hypno=hypno_9,
            include=6,
            method="std",
            threshold=3,
            n_chan_reject=5,
        )

        # With a flat channel
        data_with_flat = np.vstack((data_9, np.zeros(data_9.shape[-1])))
        art_detect(data_with_flat, sf, method="std", n_chan_reject=5)

        # Using a MNE raw object
        art_detect(data_mne, window=10.0, hypno=hypno_mne, method="covar", verbose="INFO")

        with pytest.raises(AssertionError):
            # None of include in hypno
            art_detect(data_mne, window=10.0, hypno=hypno_mne, include=[7, 8])

        # Test with Hypnogram instance + string include
        art_detect(data_9, sf=100, window=6, hypno=hyp_full, include=["N2", "N3"], method="covar")

    def test_compare_detect(self):
        """Test compare_detect function."""
        from scipy.stats import hmean

        # Default
        detected = [5, 12, 20, 34, 41, 57, 63]
        grndtrth = [5, 12, 18, 26, 34, 41, 55, 63, 68]
        res = compare_detection(detected, grndtrth)
        assert all(res["tp"] == [5, 12, 34, 41, 63])
        assert all(res["fp"] == [20, 57])
        assert all(res["fn"] == [18, 26, 55, 68])
        assert np.isclose(res["precision"], 5 / 7)
        assert np.isclose(res["recall"], 5 / 9)
        assert np.isclose(res["f1"], hmean([5 / 7, 5 / 9]))

        # Changing the order: FN <--> FP, precision <--> recall No change in F1-score.
        res = compare_detection(grndtrth, detected)
        assert all(res["tp"] == [5, 12, 34, 41, 63])
        assert all(res["fn"] == [20, 57])
        assert all(res["fp"] == [18, 26, 55, 68])
        assert np.isclose(res["precision"], 5 / 9)
        assert np.isclose(res["recall"], 5 / 7)
        assert np.isclose(res["f1"], hmean([5 / 7, 5 / 9]))

        # With max_distance
        res = compare_detection(detected, grndtrth, max_distance=2)
        assert all(res["tp"] == [5, 12, 20, 34, 41, 57, 63])
        assert len(res["fp"]) == 0
        assert all(res["fn"] == [26, 68])
        assert np.isclose(res["precision"], 1)
        assert np.isclose(res["recall"], 7 / 9)
        assert np.isclose(res["f1"], hmean([1, 7 / 9]))

        # Special cases
        # ..detected is empty
        res = compare_detection([], grndtrth)
        assert len(res["tp"]) == 0
        assert len(res["fp"]) == 0
        assert all(res["fn"] == grndtrth)

        # ..ground-truth is empty
        res = compare_detection(detected, [])
        assert len(res["tp"]) == 0
        assert all(res["fp"] == detected)
        assert len(res["fn"]) == 0

        # ..detected is not sorted
        np.random.seed(42)
        np.random.shuffle(detected)
        res = compare_detection(detected, grndtrth)
        assert np.isclose(res["f1"], hmean([5 / 7, 5 / 9]))  # Same as first example

        # ..detected has duplicate values
        detected = [5, 12, 12, 20, 34, 41, 41, 57, 63]
        res = compare_detection(detected, grndtrth)
        assert np.isclose(res["f1"], hmean([5 / 7, 5 / 9]))  # Same as first example

        # Handle dtypes
        detected = np.array([5, 12, 20, 34, 41, 57, 63], dtype=float)
        grndtrth = np.array([5, 12, 18, 26, 34, 41, 55, 63, 68], dtype=int)
        res = compare_detection(detected, grndtrth)
        assert np.isclose(res["f1"], hmean([5 / 7, 5 / 9]))
        detected = [5.0, 12, 20.0, 34, 41.0, 57.0, 63]
        grndtrth = pd.Series([5, 12, 18, 26, 34, 41, 55, 63, 68])
        res = compare_detection(detected, grndtrth)
        assert np.isclose(res["f1"], hmean([5 / 7, 5 / 9]))

        # Errors
        with pytest.raises(AssertionError):
            # Arrays contain non-integer floats
            compare_detection([5.4, 12.2, 20], [5, 12.3, 18])

        with pytest.raises(ValueError):
            # Arrays contain non-integer floats
            compare_detection(detected, grndtrth, max_distance=100)
