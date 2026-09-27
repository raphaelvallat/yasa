"""Test the functions in yasa/detection.py."""

import unittest
from functools import cache
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
from yasa.hypno import Hypnogram

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
_file_full = np.load(fetch_sample("full_6hrs_100Hz_Cz+Fz+Pz.npz"))
_data_full_raw = _file_full.get("data")
chan_full = _file_full.get("chan")
_hypno_full_raw = np.load(fetch_sample("full_6hrs_100Hz_hypno.npz")).get("hypno")

# Keep only Fz and during a N3 sleep period with (huge) slow-waves
data_sw = _data_full_raw[1, 666000:672000].astype(np.float64)
hypno_sw = _hypno_full_raw[666000:672000]

# Keep one hour of data for faster multi-channel tests. The third hour of the recording
# includes all the sleep stages (W, N1, N2, N3 and REM).
_start, _end = 720_000, 1_080_000
data_full = _data_full_raw[:, _start:_end]
hypno_full = _hypno_full_raw[_start:_end]

# Let's add a channel with bad data amplitude
chan_full = np.append(chan_full, "Bad")  # ['Cz', 'Fz', 'Pz', 'Bad']
data_full = np.vstack((data_full, data_full[-1, :] * 1e8))

# Hypnogram object (30s epochs, 100 Hz data)
hyp_full = Hypnogram.from_integers(hypno_full[:: int(sf * 30)], freq="30s")

# MNE Raw: 10 minutes of N2 and N3 sleep (epochs 60 to 80 of the hypnogram)
data_mne_fp = fetch_sample("sub-02_mne_raw.fif")
data_mne = mne.io.read_raw_fif(data_mne_fp, preload=True, verbose=0)
data_mne.pick("eeg").crop(1800, 2400, include_tmax=False)
data_mne_single = data_mne.copy().pick(["F3"])
# sub-02 Hypnogram (30s epochs, string stages already)
hypno_mne_str = np.loadtxt(fetch_sample("sub-02_hypno_30s.txt"), dtype=str)[60:80]
hyp_mne = Hypnogram(hypno_mne_str, freq="30s")
hypno_mne = hyp_mne.upsample_to_data(data_mne)

# EOG data for the REM detection
_file_rem = np.load(fetch_sample("EOGs_REM_256Hz.npz"))
loc, roc = _file_rem["data"]
sf_rem = float(_file_rem["sf"])


# Detection results that are used in several tests
@cache
def _sp_full():
    return spindles_detect(data_full, sf, chan_full)


@cache
def _sp_full_hypno():
    return spindles_detect(data_full, sf, chan_full, hypno=hypno_full)


@cache
def _sw_full():
    return sw_detect(data_full, sf, chan_full)


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
        np.testing.assert_array_equal(mask, np.isin(hypno_out, [2, 3]))

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

        # Invalid string labels give an informative error
        with pytest.raises(AssertionError, match="not valid labels of the hypnogram"):
            _check_data_hypno(data_full[1, :], sf, hypno=hyp_full, include=["NREM2"])

        # Test with MNE raw + Hypnogram
        _, _, _, hypno_out_mne, _, _, _, n_mne, _ = _check_data_hypno(
            data_mne, hypno=hyp_mne, include=["N2", "N3"]
        )
        assert hypno_out_mne.shape == (n_mne,)

        # The sampling frequency can be a NumPy scalar
        for sf_np in [np.int64(sf), np.float32(sf), np.array(sf)]:
            assert _check_data_hypno(data, sf_np)[1] == sf

        # A float hypnogram (e.g. loaded from a txt file) works with integer and float include
        hypno_float = np.full(data.size, 2.0)
        for inc in [2, 2.0, (1, 2)]:
            hypno_out = _check_data_hypno(data, sf, hypno=hypno_float, include=inc)[3]
            assert hypno_out.dtype.kind == "i"

    def test_check_data_hypno_timestamps(self):
        """A Hypnogram with a start time is aligned to a MNE Raw using absolute timestamps."""
        raw = data_mne_single.copy()
        # meas_date is treated as a local time, and the hypnogram starts 60 s (2 epochs) before
        raw_start = pd.Timestamp(raw.info["meas_date"]).replace(tzinfo=None)
        values = ["W", "W"] + ["N2"] * 18
        hyp = Hypnogram(values, freq="30s", start=raw_start - pd.Timedelta(seconds=60))
        hypno_out = _check_data_hypno(raw, hypno=hyp, include=["N2"])[3]
        # The first two (Wake) epochs are before the start of the recording
        assert (hypno_out[: int(18 * 30 * sf)] == 2).all()
        # The end of the recording is not covered by the hypnogram
        assert (hypno_out[int(18 * 30 * sf) :] == -2).all()

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
            sp = spindles_detect(data, sf, freq_sp=s, duration=d, freq_broad=b, min_distance=m)
            assert sp is not None
            assert sp.summary()["Duration"].between(*d).all()

        sp = sp_default = spindles_detect(data, sf, verbose=True)
        assert sp.summary().shape[0] == 2
        assert sp.get_mask().shape == data.shape
        df_sync = sp.get_sync_events()
        assert set(df_sync["Event"]) == {0, 1}
        # Invalid time window: the events are skipped and an error is logged
        with self.assertLogs("yasa", level="ERROR"):
            assert sp.get_sync_events(time_before=20).empty
        sp.plot_average(errorbar=None, filt=(None, 30))  # Skip bootstrapping
        np.testing.assert_array_equal(np.squeeze(sp._data), data)
        # Compare channels return dataframe with single cell
        assert sp.compare_channels().shape == (1, 1)
        assert sp._sf == sf
        sp.summary(grp_chan=True, grp_stage=True, aggfunc="median", sort=False)

        # Test with custom thresholds. The thresh dictionary of the user is not modified.
        thresh = {"rms": 1.25}
        spindles_detect(data, sf, thresh=thresh)
        assert thresh == {"rms": 1.25}
        n_default = sp.summary().shape[0]
        for thresh in [
            {"rel_pow": 0.25},
            {"rel_pow": 0.25, "corr": 0.60},
            {"rel_pow": None},
            {"rms": None},
            {"rms": None, "corr": None},
            {"rms": None, "rel_pow": None},
            {"corr": None, "rel_pow": None},
        ]:
            assert spindles_detect(data, sf, thresh=thresh).summary().shape[0] >= 1
        # Disabling one threshold detects more (or as many) spindles
        assert spindles_detect(data, sf, thresh={"corr": None}).summary().shape[0] >= n_default
        with pytest.raises(AssertionError, match="At least one threshold"):
            spindles_detect(data, sf, thresh={"rms": None, "corr": None, "rel_pow": None})

        # Test with hypnogram
        sp = spindles_detect(data, sf, hypno=np.ones(data.size))
        assert (sp.summary()["Stage"] == 1).all()

        # Outliers are only removed with at least 50 spindles
        pd.testing.assert_frame_equal(
            spindles_detect(data, sf, remove_outliers=True).summary(), sp_default.summary()
        )

        # Test with 1-sec of flat data -- we should still have 2 detected spindles
        data_flat = data.copy()
        data_flat[100:200] = 1
        sp = spindles_detect(data_flat, sf).summary()
        assert sp.shape[0] == 2

        # Full night single channel with Isolation Forest + hypnogram
        sp = spindles_detect(data_full[1, :], sf, hypno=hypno_full)
        sp_no_out = spindles_detect(data_full[1, :], sf, hypno=hypno_full, remove_outliers=True)
        assert sp_no_out.summary().shape[0] < sp.summary().shape[0]
        assert sp.compare_detection(sp_no_out).shape[0] == 1
        # Spindles are only detected in the stages defined in include (N1, N2, N3)
        assert sp.summary()["Stage"].isin([1, 2, 3]).all()

        # Spindles shorter than the 200 ms step of the STFT use the nearest STFT frame
        sp_short = spindles_detect(data_full[1, :], sf, duration=(0.01, 0.2), min_distance=None)
        assert (sp_short.summary()["Duration"] < 0.2).all()
        assert sp_short.summary()["RelPower"].between(0, 1).all()

        # Calculate the coincidence matrix with only one channel
        with pytest.raises(ValueError):
            sp.get_coincidence_matrix()

        # compare_detection with invalid other
        with pytest.raises(ValueError):
            sp.compare_detection(other="WRONG")

        with self.assertLogs("yasa", level="WARNING"):
            spindles_detect(data_n3, sf)

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

        # No values in hypno intersect with include
        with pytest.raises(AssertionError):
            sp = spindles_detect(data, sf, include=2, hypno=np.zeros(data.size, dtype=int))

        #######################################################################
        # MULTI CHANNEL
        #######################################################################

        sp = _sp_full()
        assert sp.get_mask().shape == data_full.shape
        df_sync = sp.get_sync_events(filt=(12, 15))
        # The channel with bad amplitude is skipped
        assert "Bad" not in df_sync["Channel"].unique()
        sp_grp = sp.summary(grp_chan=True)
        assert sp_grp.index.tolist() == ["Cz", "Fz", "Pz"]
        assert (sp_grp["Count"] == sp.summary()["Channel"].value_counts()[sp_grp.index]).all()
        sp.plot_average(errorbar=None)
        coinc = sp.get_coincidence_matrix()
        assert coinc.shape == (4, 4)
        assert (np.diag(coinc) == 1).all()
        coinc_unscaled = sp.get_coincidence_matrix(scaled=False)
        # The diagonal is the number of samples marked as spindle in each channel
        np.testing.assert_array_equal(np.diag(coinc_unscaled), sp.get_mask().sum(axis=1))
        assert coinc_unscaled.equals(coinc_unscaled.T)
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
        assert (sp_vs_nout_2s["f1"] >= sp_vs_nout_1s["f1"]).all()
        assert (sp_vs_nout_2s["recall"] == sp_vs_nout_1s["recall"]).all()
        assert (sp_vs_nout_2s["n_self"] == sp_vs_nout_1s["n_self"]).all()

        # Test with hypnogram
        sp = spindles_detect(data_full, sf, hypno=hypno_full, include=2)
        assert (sp.summary()["Stage"] == 2).all()
        assert sp.summary(grp_chan=False, grp_stage=False).shape == sp.summary().shape
        sp_stage = sp.summary(grp_chan=False, grp_stage=True, aggfunc="median")
        # Density = number of spindles per minute of N2 sleep
        n2_min = (hypno_full == 2).sum() / (60 * sf)
        np.testing.assert_allclose(sp_stage["Density"], sp_stage["Count"] / n2_min)
        assert sp.summary(grp_chan=True, grp_stage=False).shape[0] == 3
        assert sp.summary(grp_chan=True, grp_stage=True, sort=False).shape[0] == 3
        sp.plot_average(errorbar=None)
        sp.plot_average(hue="Stage", errorbar=None)
        sp.plot_detection()

        # Test compare_channels function
        # .. F1-score -- symmetric matrix
        mat = sp.compare_channels()
        assert mat.index.tolist() == ["CHAN000", "CHAN001", "CHAN002"]  # Order of channels
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
        sp = spindles_detect(data_mne, thresh={"corr": None, "rms": 3})
        assert set(sp.summary()["Channel"]) <= set(data_mne.ch_names)
        sp = spindles_detect(data_mne, hypno=hypno_mne, include=2, verbose=True)
        assert (sp.summary()["Stage"] == 2).all()

        # Test with Hypnogram instance (integer include, default)
        sp_hyp = spindles_detect(data_full[1, :], sf, hypno=hyp_full, include=(1, 2, 3))
        assert sp_hyp is not None
        # Test with Hypnogram instance + string include
        sp_hyp_str = spindles_detect(
            data_full[1, :], sf, hypno=hyp_full, include=["N1", "N2", "N3"]
        )
        assert sp_hyp_str is not None
        # Both should detect the same spindles
        pd.testing.assert_frame_equal(sp_hyp.summary(), sp_hyp_str.summary())
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
            sw = sw_detect(
                data_sw, sf, freq_sw=f, dur_neg=dn, dur_pos=dp, amp_neg=an, amp_pos=ap, amp_ptp=aptp
            )
            assert sw.summary()["PTP"].between(*aptp).all()

        # With N3 hypnogram
        sw = sw_detect(data_sw, sf, hypno=hypno_sw, coupling=True)
        sw_sum = sw.summary()
        assert {"SigmaPeak", "PhaseAtSigmaPeak", "ndPAC", "Stage"} <= set(sw_sum.columns)
        assert sw_sum.columns[-3:].tolist() == ["Stage", "Channel", "IdxChannel"]
        assert sw.get_mask().shape == data_sw.shape
        assert not sw.get_sync_events().empty
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
        sw_2d = sw_detect(data_sw[np.newaxis, ...], sf, verbose="INFO")
        pd.testing.assert_frame_equal(sw_2d.summary(), sw_detect(data_sw, sf).summary())

        # No values in hypno intersect with include
        with pytest.raises(AssertionError):
            sw = sw_detect(data_sw, sf, include=3, hypno=np.ones(data_sw.shape, dtype=int))

        #######################################################################
        # MULTI CHANNEL
        #######################################################################

        sw = _sw_full()
        assert sw.get_mask().shape == data_full.shape
        # Without coupling, there are no coupling columns in the grouped summary
        sw_grp = sw.summary(grp_chan=True)
        assert sw_grp.index.tolist() == ["Cz", "Fz", "Pz"]
        assert "ndPAC" not in sw_grp.columns
        df_sync = sw.get_sync_events()
        assert df_sync["Channel"].unique().tolist() == ["Cz", "Fz", "Pz"]
        sw.plot_average(errorbar=None)
        sw.plot_detection()
        assert (np.diag(sw.get_coincidence_matrix()) == 1).all()
        assert sw.get_coincidence_matrix(scaled=False).to_numpy().dtype.kind == "i"
        # Test with outlier removal. There should be fewer events.
        sw_no_out = sw_detect(data_full, sf, chan_full, remove_outliers=True)
        assert sw_no_out._events.shape[0] < sw._events.shape[0]

        # Test compare_detection
        assert (sw.compare_detection(sw_no_out)["recall"] == 1).all()

        # Test with hypnogram
        sw = sw_detect(data_full, sf, chan_full, hypno=hypno_full, coupling=True)
        assert sw.summary()["Stage"].isin([2, 3]).all()
        assert sw.summary(grp_chan=False, grp_stage=False).shape == sw.summary().shape
        assert sw.summary(grp_chan=False, grp_stage=True, aggfunc="median").shape[0] == 2
        assert sw.summary(grp_chan=True, grp_stage=False).shape[0] == 3
        assert sw.summary(grp_chan=True, grp_stage=True, sort=False).shape[0] == 6
        sw.plot_average(hue="Stage", errorbar=None)
        # Check coupling
        sw_sum = sw.summary()
        assert "ndPAC" in sw_sum.columns
        assert sw_sum["PhaseAtSigmaPeak"].between(-np.pi, np.pi).all()
        # There should be some zero in the ndPAC (full dataframe)
        assert sw._events[sw._events["ndPAC"] == 0].shape[0] > 0
        # Coinciding spindles and masking
        sp_sum = _sp_full_hypno().summary()
        sw.find_cooccurring_spindles(sp_sum)
        sw_sum = sw.summary()
        assert "CooccurringSpindle" in sw_sum.columns
        assert "DistanceSpindleToSW" in sw_sum.columns
        cooc = sw_sum[sw_sum["CooccurringSpindle"]]
        assert cooc["DistanceSpindleToSW"].abs().max() < 1.2
        assert cooc["CooccurringSpindlePeak"].isin(sp_sum["Peak"]).all()
        assert sw_sum.loc[~sw_sum["CooccurringSpindle"], "DistanceSpindleToSW"].isna().all()
        sw_sum_masked = sw.summary(
            grp_chan=True, grp_stage=False, mask=sw._events["CooccurringSpindle"]
        )
        assert (sw_sum_masked["Count"] < sw.summary(grp_chan=True)["Count"]).all()

        # Test with different coupling params. Missing keys are set to their default value.
        sw = sw_detect(
            data_full,
            sf,
            chan_full,
            hypno=hypno_full,
            coupling=True,
            coupling_params={"freq_sp": (12, 16), "time": 2, "p": None},
        )
        assert (sw.summary()["ndPAC"] > 0).all()  # No thresholding
        sw_detect(data_sw, sf, coupling=True, coupling_params={"p": None})

        # Using a MNE raw object
        assert sw_detect(data_mne) is not None
        assert (sw_detect(data_mne, hypno=hypno_mne, include=3).summary()["Stage"] == 3).all()

        # Test with Hypnogram instance + integer include
        sw_hyp = sw_detect(data_full[1, :], sf, hypno=hyp_full, include=(2, 3))
        assert sw_hyp is not None
        # Test with Hypnogram instance + string include
        sw_hyp_str = sw_detect(data_full[1, :], sf, hypno=hyp_full, include=["N2", "N3"])
        pd.testing.assert_frame_equal(sw_hyp.summary(), sw_hyp_str.summary())
        # Test with MNE raw + Hypnogram
        sw_detect(data_mne, hypno=hyp_mne, include=["N3"])
        plt.close("all")

    def test_get_sync_events_edges(self):
        """Events too close to the data edges are dropped, the other keep their stage."""
        sw = sw_detect(data_sw, sf, hypno=hypno_sw)
        events = sw._events.iloc[:3].copy()
        # First event close to the start of the data, the other two with a different stage
        events["NegPeak"] = [0.1, 20.0, 40.0]
        events["Stage"] = [1, 2, 3]
        sw._events = events.reset_index(drop=True)
        df_sync = sw.get_sync_events(time_before=0.5, time_after=0.5)
        assert df_sync["Event"].nunique() == 2
        assert df_sync.groupby("Event")["Stage"].first().tolist() == [2, 3]
        # The amplitudes are those of the 20 s and 40 s events
        amps = sw.get_sync_events(time_before=0.5, time_after=0.5, as_dataframe=False)[0]
        np.testing.assert_array_equal(amps[:, 50], data_sw[[2000, 4000]])
        np.testing.assert_array_equal(
            df_sync.loc[df_sync["Time"] == 0, "Amplitude"], data_sw[[2000, 4000]]
        )
        # Events at the very end of the data (n_samples = 6000)
        events["NegPeak"] = [59.49, 20.0, 40.0]  # Last sample of the epoch = 5999
        sw._events = events.reset_index(drop=True)
        assert sw.get_sync_events(time_before=0.5, time_after=0.5)["Event"].nunique() == 3
        events["NegPeak"] = [59.5, 20.0, 40.0]  # Last sample of the epoch = 6000, out of bounds
        sw._events = events.reset_index(drop=True)
        assert sw.get_sync_events(time_before=0.5, time_after=0.5)["Event"].nunique() == 2

    def test_get_mask_rounding(self):
        """get_mask includes the first and last sample of each event."""
        sp = spindles_detect(data, sf)
        events = sp._events.iloc[:1].copy()
        events[["Start", "End"]] = [0.29, 0.57]  # 0.29 * 100 = 28.999999999999996
        sp._events = events
        np.testing.assert_array_equal(np.flatnonzero(sp.get_mask()), np.arange(29, 58))

    def test_rem_detect(self):
        """Test function REM detect"""
        hypno_rem = 4 * np.ones_like(loc)

        # Parameters product testing
        freq_rem = [(0.5, 5), (0.3, 8)]
        duration = [(0.3, 1.5), [0.5, 1]]
        amplitude = [(50, 200), [60, 300]]
        hypno = [hypno_rem, None]
        prod_args = product(freq_rem, duration, amplitude, hypno)

        for f, dr, am, h in prod_args:
            rem = rem_detect(loc, roc, sf_rem, hypno=h, freq_rem=f, duration=dr, amplitude=am)
            rem_sum = rem.summary()
            assert rem_sum["Duration"].between(dr[0], dr[1], inclusive="left").all()
            assert ("Stage" in rem_sum) == (h is not None)

        # With isolation forest
        rem = rem_detect(loc, roc, sf_rem, verbose="info")
        rem2 = rem_detect(loc, roc, sf_rem, remove_outliers=True)
        assert rem.summary().shape[0] > rem2.summary().shape[0]
        assert rem.get_mask().shape == (2, loc.size)
        df_sync = rem.get_sync_events()
        assert df_sync["Channel"].unique().tolist() == ["LOC", "ROC"]
        assert df_sync["Event"].nunique() == rem.summary().shape[0]
        rem.plot_average(errorbar=None)
        rem.plot_average(filt=(0.5, 5), errorbar=None)
        plt.close("all")

        # compare_detection works on the combined LOC and ROC channels
        res = rem.compare_detection(rem2)
        assert res.index.tolist() == ["LOC-ROC"]
        assert res.loc["LOC-ROC", "recall"] == 1
        assert res.loc["LOC-ROC", "n_other"] == rem2.summary().shape[0]
        # ... also with a dataframe of annotations without Channel column
        res_df = rem.compare_detection(rem2.summary()[["Start"]])
        pd.testing.assert_frame_equal(res, res_df)

        # With REM hypnogram
        rem = rem_detect(loc, roc, sf_rem, hypno=hypno_rem)
        hypno_rem_half = np.r_[np.ones(int(loc.size / 2)), 4 * np.ones(int(loc.size / 2))]
        rem2 = rem_detect(loc, roc, sf_rem, hypno=hypno_rem_half)
        assert rem.summary().shape[0] > rem2.summary().shape[0]
        assert (rem2.summary()["Start"] >= loc.size / 2 / sf_rem).all()
        rem_stage = rem2.summary(grp_stage=True, aggfunc="median")
        assert rem_stage.index.tolist() == [4]

        # Test with wrong data amplitude on ROC
        with self.assertLogs("yasa", level="ERROR"):
            rem = rem_detect(loc * 1e-8, roc, sf_rem)
        assert rem is None

        # Test with wrong data amplitude on LOC
        with self.assertLogs("yasa", level="ERROR"):
            rem = rem_detect(loc, roc * 1e8, sf_rem)
        assert rem is None

        # No values in hypno intersect with include
        with pytest.raises(AssertionError):
            rem_detect(loc, roc, sf_rem, hypno=hypno_rem_half, include=5)

        # Test with Hypnogram instance + string include
        n_epochs = int(np.ceil(loc.size / (30 * sf_rem)))
        hyp_rem = Hypnogram(["REM"] * n_epochs, freq="30s")
        rem_hyp = rem_detect(loc, roc, sf_rem, hypno=hyp_rem, include="REM")
        assert rem_hyp is not None
        # Integer and string include should give the same result
        rem_int = rem_detect(loc, roc, sf_rem, hypno=hyp_rem, include=4)
        pd.testing.assert_frame_equal(rem_hyp.summary(), rem_int.summary())

    def test_art_detect(self):
        """Test function art_detect"""
        file_9_fp = fetch_sample("full_6hrs_100Hz_9channels.npz")
        # One hour of data
        data_9 = np.load(file_9_fp).get("data")[:, :360_000]
        hypno_9 = _hypno_full_raw[:360_000].astype(np.int64)
        # For the sake of the example, let's add some flat data at the end
        data_9 = np.concatenate((data_9, np.zeros((data_9.shape[0], 20000))), axis=1)
        hypno_9 = np.concatenate((hypno_9, np.zeros(20000, dtype=np.int64)))
        hypno_9_orig = hypno_9.copy()
        n_flat = 20000 // (5 * sf)

        # Start different combinations
        art, zscores = art_detect(data_9, sf=100, window=10, method="covar", threshold=3)
        assert art.shape == zscores.shape == (data_9.shape[1] // (10 * sf),)
        assert art.dtype == bool
        art, zscores = art_detect(
            data_9, sf=100, window=6, hypno=hypno_9, include=(2, 3), method="covar", threshold=3
        )
        assert art.shape == (data_9.shape[1] // (6 * sf),)
        art, zscores = art_detect(data_9, sf=100, window=5, method="std", threshold=2)
        assert zscores.shape == (art.size, data_9.shape[0])
        # The flat epochs at the end are always marked as artefacts, with z-scores set to NaN
        assert art[-n_flat:].all()
        assert np.isnan(zscores[-n_flat:]).all()
        assert 0 < art[:-n_flat].mean() < 0.2
        art, _ = art_detect(
            data_9,
            sf=100,
            window=5,
            hypno=hypno_9,
            method="std",
            include=(0, 1, 2, 3, 4, 5, 6),
            threshold=2,
        )
        assert art[-n_flat:].all()
        # The hypnogram of the user is not modified by the flagging of flat epochs
        np.testing.assert_array_equal(hypno_9, hypno_9_orig)
        # A higher threshold rejects fewer epochs
        art_10, _ = art_detect(
            data_9,
            sf=100,
            window=5.0,
            hypno=hypno_9,
            method="std",
            include=(0, 1, 2, 3, 4, 5, 6),
            threshold=10,
        )
        assert art_10.sum() < art.sum()
        # Single channel
        art, _ = art_detect(data_9[0], 100, window=10, method="covar")  # Switches to std
        assert art.shape == (data_9.shape[1] // (10 * sf),)
        art_detect(data_9[0], 100, window=5, method="std", verbose=True)

        # Exactly 4 channels is enough for method="covar"
        with self.assertNoLogs("yasa", level="WARNING"):
            art_detect(data_9[:4, :360_000], sf, window=10, method="covar")
        with self.assertLogs("yasa", level="WARNING"):
            art_detect(data_9[:3, :360_000], sf, window=10, method="covar")
        with pytest.raises(ValueError, match="Invalid method"):
            art_detect(data_9, sf, method="wrong")

        # Not enough epochs for stage
        hypno_9[:100] = 6
        with self.assertLogs("yasa", level="WARNING"):
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
        with self.assertLogs("yasa", level="WARNING"):
            art, zscores = art_detect(data_with_flat, sf, method="std", n_chan_reject=5)
        assert zscores.shape[1] == data_9.shape[0]  # The flat channel is removed

        # Using a MNE raw object
        art_detect(data_mne, window=10.0, hypno=hypno_mne, method="covar", verbose="INFO")

        with pytest.raises(AssertionError):
            # None of include in hypno
            art_detect(data_mne, window=10.0, hypno=hypno_mne, include=[7, 8])

        # Test with Hypnogram instance + string include
        hyp_9 = Hypnogram.from_integers(hypno_9_orig[:: 30 * sf], freq="30s")
        art_hyp, _ = art_detect(
            data_9, sf=100, window=6, hypno=hyp_9, include=["N2", "N3"], method="covar"
        )
        hypno_9_up = hyp_9.upsample_to_data(data_9, sf=sf, verbose="error")
        art_int, _ = art_detect(
            data_9, sf=100, window=6, hypno=hypno_9_up, include=(2, 3), method="covar"
        )
        np.testing.assert_array_equal(art_hyp, art_int)

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

        # Several detected events matching the same ground-truth event do not inflate the recall
        res = compare_detection([9, 10, 11], [10, 50], max_distance=1)
        assert np.isclose(res["precision"], 1)
        assert np.isclose(res["recall"], 0.5)
        assert all(res["fn"] == [50])

        # A large max_distance is valid, even for small indices
        res = compare_detection([5], [5], max_distance=3)
        assert res["f1"] == 1
        res = compare_detection(detected, grndtrth, max_distance=100)
        assert res["precision"] == res["recall"] == 1

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

        with pytest.raises(AssertionError):
            # max_distance must be a positive integer
            compare_detection(detected, grndtrth, max_distance=-1)
