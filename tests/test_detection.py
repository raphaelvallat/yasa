"""Test the functions in yasa/detection.py."""

import copy
import logging
from contextlib import contextmanager
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.stats import hmean

from yasa._validation import _check_data_hypno
from yasa.detection import (
    art_detect,
    compare_detection,
    rem_detect,
    spindles_detect,
    sw_detect,
)
from yasa.hypno import Hypnogram

# For all data (except the EOG), the sampling frequency is always 100 Hz
sf = 100


@contextmanager
def assert_logs(level="WARNING", logged=True):
    """Check that the yasa logger emits (or not) a message of at least ``level``.

    Same as ``unittest.TestCase.assertLogs`` / ``assertNoLogs``. The yasa logger does not
    propagate to the root logger, so the ``caplog`` fixture of pytest cannot be used directly.
    """
    logger = logging.getLogger("yasa")
    level = logging.getLevelName(level)
    records = []
    handler = logging.Handler(level)
    handler.emit = records.append
    old_handlers, old_level = logger.handlers[:], logger.level
    logger.handlers = [handler]
    logger.setLevel(level)
    try:
        yield records
    finally:
        logger.handlers = old_handlers
        logger.setLevel(old_level)
    if logged:
        assert records, f"No log of level {logging.getLevelName(level)} or higher"
    else:
        assert not records, f"Unexpected logs: {[r.getMessage() for r in records]}"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


##############################################################################
# DATA
##############################################################################


@pytest.fixture(scope="module")
def data(n2_spindles):
    """15 seconds of N2 sleep, single channel. One every other point to keep a sf of 100 Hz."""
    return n2_spindles.data[::2].copy()


@pytest.fixture(scope="module")
def data_n3(n3_no_spindles):
    """30 seconds of N3 sleep without any spindle, 100 Hz."""
    return n3_no_spindles.data


@pytest.fixture(scope="module")
def sw_n3(full_6hrs):
    """Fz during a N3 sleep period with (huge) slow-waves, with its hypnogram."""
    data_sw = full_6hrs.data[1, 666000:672000].astype(np.float64)
    hypno_sw = full_6hrs.hypno[666000:672000]
    return SimpleNamespace(data=data_sw, hypno=hypno_sw)


@pytest.fixture(scope="module")
def full(full_6hrs):
    """One hour of multi-channel data (Cz, Fz, Pz) plus a channel with bad amplitude.

    The third hour of the recording includes all the sleep stages (W, N1, N2, N3 and REM).
    """
    start, end = 720_000, 1_080_000
    data_full = full_6hrs.data[:, start:end]
    hypno_full = full_6hrs.hypno[start:end]
    # Let's add a channel with bad data amplitude
    chan_full = np.append(full_6hrs.chan, "Bad")  # ['Cz', 'Fz', 'Pz', 'Bad']
    data_full = np.vstack((data_full, data_full[-1, :] * 1e8))
    # Hypnogram object (30s epochs, 100 Hz data)
    hyp_full = Hypnogram.from_integers(hypno_full[:: int(sf * 30)], freq="30s")
    return SimpleNamespace(data=data_full, chan=chan_full, hypno=hypno_full, hyp=hyp_full)


@pytest.fixture(scope="module")
def mne_n2n3(_raw_sub02, hypno_sub02):
    """MNE Raw: 10 minutes of N2 and N3 sleep (epochs 60 to 80 of the hypnogram).

    The detection functions do not modify the Raw, so it is shared by all the tests.
    """
    raw = _raw_sub02.copy().pick("eeg").crop(1800, 2400, include_tmax=False)
    hyp = Hypnogram(hypno_sub02[60:80], freq="30s")
    return SimpleNamespace(raw=raw, hyp=hyp, hypno=hyp.upsample_to_data(raw))


@pytest.fixture(scope="module")
def eog(eog_rem):
    """LOC and ROC during REM sleep, with an all-REM integer hypnogram."""
    return SimpleNamespace(
        loc=eog_rem.loc, roc=eog_rem.roc, sf=eog_rem.sf, hypno=4 * np.ones_like(eog_rem.loc)
    )


# Detection results that are used in several tests
@pytest.fixture(scope="module")
def sp_full(full):
    return spindles_detect(full.data, sf, full.chan)


@pytest.fixture(scope="module")
def sp_full_hypno(full):
    return spindles_detect(full.data, sf, full.chan, hypno=full.hypno)


@pytest.fixture(scope="module")
def sp_multi(full):
    return spindles_detect(full.data, sf, full.chan, multi_only=True)


@pytest.fixture(scope="module")
def sp_no_out(full):
    return spindles_detect(full.data, sf, full.chan, remove_outliers=True)


@pytest.fixture(scope="module")
def sp_n2(full):
    """Multi-channel spindles in N2 only (without channel names)."""
    return spindles_detect(full.data, sf, hypno=full.hypno, include=2)


@pytest.fixture(scope="module")
def sw_full(full):
    return sw_detect(full.data, sf, full.chan)


@pytest.fixture(scope="module")
def sw_coupling(full):
    """Multi-channel slow-waves with hypnogram and phase-amplitude coupling."""
    return sw_detect(full.data, sf, full.chan, hypno=full.hypno, coupling=True)


##############################################################################
# _check_data_hypno
##############################################################################


@pytest.mark.parametrize("kwargs", [{"sf": 999}, {"ch_names": ["CH999"]}])
def test_check_data_hypno_mne_ignored_args(mne_n2n3, kwargs):
    """With a MNE Raw, sf and ch_names are ignored with a warning."""
    with assert_logs("WARNING"):
        _check_data_hypno(mne_n2n3.raw, **kwargs)


def test_check_data_hypno_outputs(full, data):
    """Outputs specific to the detection functions.

    The data and hypnogram checks are tested in test_others.py.
    """
    data_out, _, _, hypno_out, include_out, mask, n_chan, n_samples, bad_chan = _check_data_hypno(
        full.data, sf, hypno=full.hyp, include=["N2", "N3"]
    )
    assert data_out.shape == (n_chan, n_samples) == (4, full.hypno.size)
    # The hypnogram is always returned as int64, as needed for the Stage column
    assert hypno_out.dtype == np.int64
    np.testing.assert_array_equal(include_out, [2, 3])
    np.testing.assert_array_equal(mask, np.isin(hypno_out, [2, 3]))
    np.testing.assert_array_equal(bad_chan, [False, False, False, True])
    # Without hypnogram, the mask is all True
    assert _check_data_hypno(data, sf)[5].all()


def test_check_data_hypno_str_array(data):
    """String hypnogram arrays are not supported."""
    with pytest.raises(AssertionError, match="integer array"):
        _check_data_hypno(data, sf, hypno=np.full(data.size, "N2"), include="N2")


##############################################################################
# SPINDLES -- SINGLE CHANNEL
##############################################################################


# Representative parameter combinations (covers all values, avoids full Cartesian product of 24
# iterations): freq_sp, freq_broad, duration, min_distance
@pytest.mark.parametrize(
    "freq_sp, freq_broad, duration, min_distance",
    [
        ((11, 16), (0.5, 30), (0.3, 2.5), None),
        ([12, 14], (0.5, 30), (0.3, 2.5), 0),
        ((11, 16), [1, 25], (0.3, 2.5), 500),
        ([12, 14], [1, 25], [0.5, 3], None),
        ((11, 16), (0.5, 30), [0.5, 3], 0),
        ([12, 14], [1, 25], [0.5, 3], 500),
    ],
)
def test_spindles_params(data, freq_sp, freq_broad, duration, min_distance):
    """The duration of the detected spindles is within the requested range."""
    sp = spindles_detect(
        data,
        sf,
        freq_sp=freq_sp,
        duration=duration,
        freq_broad=freq_broad,
        min_distance=min_distance,
    )
    assert sp is not None
    assert sp.summary()["Duration"].between(*duration).all()


@pytest.fixture(scope="module")
def sp_default(data):
    return spindles_detect(data, sf, verbose=True)


def test_spindles_single_channel(sp_default, data):
    """Default single-channel detection and methods of the SpindlesResults."""
    sp = sp_default
    assert sp.summary().shape[0] == 2
    assert sp.get_mask().shape == data.shape
    df_sync = sp.get_sync_events()
    assert set(df_sync["Event"]) == {0, 1}
    # Invalid time window: the events are skipped and an error is logged
    with assert_logs("ERROR"):
        assert sp.get_sync_events(time_before=20).empty
    sp.plot_average(errorbar=None, filt=(None, 30))  # Skip bootstrapping
    np.testing.assert_array_equal(np.squeeze(sp._data), data)
    # Compare channels return dataframe with single cell
    assert sp.compare_channels().shape == (1, 1)
    assert sp._sf == sf


def test_spindles_single_channel_errors(sp_default):
    """Coincidence matrix and invalid compare_detection input with a single channel."""
    with pytest.raises(ValueError):
        sp_default.get_coincidence_matrix()
    with pytest.raises(ValueError):
        sp_default.compare_detection(other="WRONG")


def test_spindles_thresh_not_modified(data):
    """The thresh dictionary of the user is not modified."""
    thresh = {"rms": 1.25}
    spindles_detect(data, sf, thresh=thresh)
    assert thresh == {"rms": 1.25}


@pytest.mark.parametrize(
    "thresh",
    [
        {"rel_pow": 0.25},
        {"rel_pow": 0.25, "corr": 0.60},
        {"rel_pow": None},
        {"rms": None},
        {"rms": None, "corr": None},
        {"rms": None, "rel_pow": None},
        {"corr": None, "rel_pow": None},
    ],
)
def test_spindles_custom_thresh(data, thresh):
    """Custom (or disabled) thresholds still detect spindles."""
    assert spindles_detect(data, sf, thresh=thresh).summary().shape[0] >= 1


def test_spindles_disable_threshold(data, sp_default):
    """Disabling one threshold detects more (or as many) spindles, but not all of them."""
    n_default = sp_default.summary().shape[0]
    assert spindles_detect(data, sf, thresh={"corr": None}).summary().shape[0] >= n_default
    with pytest.raises(AssertionError, match="At least one threshold"):
        spindles_detect(data, sf, thresh={"rms": None, "corr": None, "rel_pow": None})


def test_spindles_hypno_single_channel(data):
    """The Stage column is taken from the hypnogram."""
    sp = spindles_detect(data, sf, hypno=np.ones(data.size))
    assert (sp.summary()["Stage"] == 1).all()


def test_spindles_remove_outliers_few_events(data, sp_default):
    """Outliers are only removed with at least 50 spindles."""
    pd.testing.assert_frame_equal(
        spindles_detect(data, sf, remove_outliers=True).summary(), sp_default.summary()
    )


def test_spindles_flat_segment(data):
    """With 1-sec of flat data, we should still have 2 detected spindles."""
    data_flat = data.copy()
    data_flat[100:200] = 1
    sp = spindles_detect(data_flat, sf).summary()
    assert sp.shape[0] == 2


def test_spindles_no_spindles(data_n3):
    """N3 sleep without spindles: warnings, and None if no spindle is found at all."""
    with assert_logs("WARNING"):
        spindles_detect(data_n3, sf)
    # Ensure that the two warnings are tested
    with assert_logs("WARNING"):
        sp = spindles_detect(data_n3, sf, thresh={"corr": 0.95})
    assert sp is None


@pytest.mark.parametrize("factor", [1e-6, 1e6])
def test_spindles_wrong_amplitude(data_n3, factor):
    """Data with a wrong amplitude logs an error and returns None."""
    with assert_logs("ERROR"):
        sp = spindles_detect(data_n3 * factor, sf)
    assert sp is None


def test_spindles_include_not_in_hypno(data):
    """No values in hypno intersect with include."""
    with pytest.raises(AssertionError):
        spindles_detect(data, sf, include=2, hypno=np.zeros(data.size, dtype=int))


##############################################################################
# SPINDLES -- MULTI CHANNEL
##############################################################################


def test_spindles_multi_summary(sp_full, full):
    """Mask, sync events and grouped summary. The channel with bad amplitude is skipped."""
    sp = sp_full
    assert sp.get_mask().shape == full.data.shape
    df_sync = sp.get_sync_events(filt=(12, 15))
    assert "Bad" not in df_sync["Channel"].unique()
    sp_grp = sp.summary(grp_chan=True)
    assert sp_grp.index.tolist() == ["Cz", "Fz", "Pz"]
    assert (sp_grp["Count"] == sp.summary()["Channel"].value_counts()[sp_grp.index]).all()


def test_spindles_multi_coincidence(sp_full):
    """Scaled and unscaled coincidence matrices."""
    coinc = sp_full.get_coincidence_matrix()
    assert coinc.shape == (4, 4)
    assert (np.diag(coinc) == 1).all()
    coinc_unscaled = sp_full.get_coincidence_matrix(scaled=False)
    # The diagonal is the number of samples marked as spindle in each channel
    np.testing.assert_array_equal(np.diag(coinc_unscaled), sp_full.get_mask().sum(axis=1))
    assert coinc_unscaled.equals(coinc_unscaled.T)


def test_spindles_multi_data(sp_full, full):
    """Plot of the detection, and the data stored in the results."""
    sp_full.plot_detection()
    assert sp_full._data.shape == sp_full._data_filt.shape
    np.testing.assert_array_equal(sp_full._data, full.data)


def test_spindles_multi_only_and_outliers(sp_full, sp_multi, sp_no_out):
    """multi_only and remove_outliers both reduce the number of spindles."""
    assert sp_multi.summary().shape[0] < sp_full.summary().shape[0]
    assert sp_no_out.summary().shape[0] < sp_full.summary().shape[0]


def test_spindles_compare_detection(sp_full, sp_multi, sp_no_out):
    """SpindlesResults.compare_detection against other results or a summary dataframe."""
    sp = sp_full
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


def test_spindles_multi_hypno(sp_n2, full):
    """Detection in N2 only, density and grouped summaries."""
    sp = sp_n2
    assert (sp.summary()["Stage"] == 2).all()
    sp_stage = sp.summary(grp_chan=False, grp_stage=True, aggfunc="median")
    # Density = number of spindles per minute of N2 sleep
    n2_min = (full.hypno == 2).sum() / (60 * sf)
    np.testing.assert_allclose(sp_stage["Density"], sp_stage["Count"] / n2_min)
    assert sp.summary(grp_chan=True, grp_stage=False).shape[0] == 3
    assert sp.summary(grp_chan=True, grp_stage=True, sort=False).shape[0] == 3
    sp.plot_average(hue="Stage", errorbar=None)


def test_spindles_compare_channels(sp_n2):
    """compare_channels: F1-score (symmetric matrix), precision and recall."""
    mat = sp_n2.compare_channels()
    assert mat.index.tolist() == ["CHAN000", "CHAN001", "CHAN002"]  # Order of channels
    assert mat.equals(mat.T)  # mat is a symmetric matrix
    assert (np.diag(mat) == 1).all()  # diagonal is all 1
    idx_triu = np.triu_indices_from(mat, k=1)
    mat_2s = sp_n2.compare_channels(max_distance_sec=2)
    # Make sure that the overall scores are higher when using a lookaround
    assert mat.to_numpy()[idx_triu].mean() < mat_2s.to_numpy()[idx_triu].mean()
    # Precision / recall -- not symmetric
    mat_prec = sp_n2.compare_channels(score="precision")
    mat_rec = sp_n2.compare_channels(score="recall")
    assert mat_prec.T.equals(mat_rec)  # tril precision == triu recall


def test_spindles_mne(mne_n2n3):
    """Using a MNE Raw object, with an integer hypnogram or a Hypnogram + string include."""
    raw = mne_n2n3.raw
    # Disabling one threshold
    sp = spindles_detect(raw, thresh={"corr": None, "rms": 3})
    assert set(sp.summary()["Channel"]) <= set(raw.ch_names)
    sp = spindles_detect(raw, hypno=mne_n2n3.hypno, include=2, verbose=True)
    assert (sp.summary()["Stage"] == 2).all()
    # MNE raw + Hypnogram with string include: same as the upsampled integer hypnogram
    sp_hyp = spindles_detect(raw, hypno=mne_n2n3.hyp, include=["N2"])
    pd.testing.assert_frame_equal(sp_hyp.summary(), sp.summary())


##############################################################################
# SLOW-WAVES -- SINGLE CHANNEL
##############################################################################


# Representative parameter combinations (covers all values and None thresholds, avoids full
# Cartesian product of 64 iterations).
@pytest.mark.parametrize(
    "freq_sw, dur_neg, dur_pos, amp_neg, amp_pos, amp_ptp",
    [
        ((0.3, 3.5), (0.3, 1.5), (0.3, 1.5), (40, 300), (10, 150), (75, 400)),
        ((0.5, 4), [0.1, 2], [0, 1], [40, None], (0, None), [80, 300]),
        ((0.3, 3.5), [0.1, 2], (0.3, 1.5), [40, None], (10, 150), [80, 300]),
        ((0.5, 4), (0.3, 1.5), [0, 1], (40, 300), (0, None), (75, 400)),
    ],
)
def test_sw_params(sw_n3, freq_sw, dur_neg, dur_pos, amp_neg, amp_pos, amp_ptp):
    """The peak-to-peak amplitude of the detected slow-waves is within the requested range."""
    sw = sw_detect(
        sw_n3.data,
        sf,
        freq_sw=freq_sw,
        dur_neg=dur_neg,
        dur_pos=dur_pos,
        amp_neg=amp_neg,
        amp_pos=amp_pos,
        amp_ptp=amp_ptp,
    )
    assert sw.summary()["PTP"].between(*amp_ptp).all()


def test_sw_single_channel(sw_n3):
    """Single channel with N3 hypnogram and coupling, and methods of the SWResults."""
    sw = sw_detect(sw_n3.data, sf, hypno=sw_n3.hypno, coupling=True)
    sw_sum = sw.summary()
    assert {"SigmaPeak", "PhaseAtSigmaPeak", "ndPAC", "Stage"} <= set(sw_sum.columns)
    assert sw_sum.columns[-3:].tolist() == ["Stage", "Channel", "IdxChannel"]
    assert sw.get_mask().shape == sw_n3.data.shape
    assert not sw.get_sync_events().empty
    sw.plot_average(errorbar=None)
    sw.plot_detection()
    np.testing.assert_array_equal(np.squeeze(sw._data), sw_n3.data)
    np.testing.assert_array_equal(sw._hypno, sw_n3.hypno)
    assert sw._sf == sf


def test_sw_flat(sw_n3):
    """All channels are flat: an error is logged and None is returned."""
    with assert_logs("ERROR"):
        sw = sw_detect(sw_n3.data * 0, sf)
    assert sw is None


def test_sw_2d(sw_n3):
    """2D data with a single channel gives the same results as 1D data."""
    sw_2d = sw_detect(sw_n3.data[np.newaxis, ...], sf, verbose="INFO")
    pd.testing.assert_frame_equal(sw_2d.summary(), sw_detect(sw_n3.data, sf).summary())


def test_sw_include_not_in_hypno(sw_n3):
    """No values in hypno intersect with include."""
    with pytest.raises(AssertionError):
        sw_detect(sw_n3.data, sf, include=3, hypno=np.ones(sw_n3.data.shape, dtype=int))


##############################################################################
# SLOW-WAVES -- MULTI CHANNEL
##############################################################################


def test_sw_multi(sw_full, full):
    """Mask, grouped summary without coupling, sync events and coincidence matrices."""
    sw = sw_full
    assert sw.get_mask().shape == full.data.shape
    # Without coupling, there are no coupling columns in the grouped summary
    sw_grp = sw.summary(grp_chan=True)
    assert sw_grp.index.tolist() == ["Cz", "Fz", "Pz"]
    assert "ndPAC" not in sw_grp.columns
    df_sync = sw.get_sync_events()
    assert df_sync["Channel"].unique().tolist() == ["Cz", "Fz", "Pz"]
    assert (np.diag(sw.get_coincidence_matrix()) == 1).all()
    assert sw.get_coincidence_matrix(scaled=False).to_numpy().dtype.kind == "i"


def test_sw_remove_outliers(sw_full, full):
    """Outlier removal gives fewer events, all of them found in the full detection."""
    sw_no_out = sw_detect(full.data, sf, full.chan, remove_outliers=True)
    assert sw_no_out._events.shape[0] < sw_full._events.shape[0]
    assert (sw_full.compare_detection(sw_no_out)["recall"] == 1).all()


def test_sw_multi_hypno(sw_coupling):
    """Detection in N2 and N3 with grouped summaries."""
    sw = sw_coupling
    assert sw.summary()["Stage"].isin([2, 3]).all()
    assert sw.summary(grp_chan=False, grp_stage=True, aggfunc="median").shape[0] == 2
    assert sw.summary(grp_chan=True, grp_stage=False).shape[0] == 3
    assert sw.summary(grp_chan=True, grp_stage=True, sort=False).shape[0] == 6
    sw.plot_average(hue="Stage", errorbar=None)


def test_sw_coupling(sw_coupling):
    """Phase-amplitude coupling columns."""
    sw_sum = sw_coupling.summary()
    assert "ndPAC" in sw_sum.columns
    assert sw_sum["PhaseAtSigmaPeak"].between(-np.pi, np.pi).all()
    # There should be some zero in the ndPAC (full dataframe)
    assert sw_coupling._events[sw_coupling._events["ndPAC"] == 0].shape[0] > 0


def test_sw_cooccurring_spindles(sw_coupling, sp_full_hypno):
    """Coinciding spindles and masking of the summary."""
    # find_cooccurring_spindles modifies the events in place: work on a copy of the fixture
    sw = copy.deepcopy(sw_coupling)
    sp_sum = sp_full_hypno.summary()
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


def test_sw_coupling_params(full):
    """Different coupling params. Missing keys (freq_sp) are set to their default."""
    sw = sw_detect(full.data, sf, full.chan, coupling=True, coupling_params={"time": 2, "p": None})
    assert (sw.summary()["ndPAC"] > 0).all()  # No thresholding


def test_sw_mne(mne_n2n3):
    """Using a MNE Raw object, with an integer hypnogram or a Hypnogram + string include."""
    raw = mne_n2n3.raw
    assert sw_detect(raw) is not None
    sw = sw_detect(raw, hypno=mne_n2n3.hypno, include=3)
    assert (sw.summary()["Stage"] == 3).all()
    # MNE raw + Hypnogram with string include: same as the upsampled integer hypnogram
    sw_hyp = sw_detect(raw, hypno=mne_n2n3.hyp, include=["N3"])
    pd.testing.assert_frame_equal(sw_hyp.summary(), sw.summary())


##############################################################################
# _DetectionResults methods
##############################################################################


def test_get_sync_events_edges(sw_n3):
    """Events too close to the data edges are dropped, the other keep their stage."""
    sw = sw_detect(sw_n3.data, sf, hypno=sw_n3.hypno)
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
    np.testing.assert_array_equal(amps[:, 50], sw_n3.data[[2000, 4000]])
    np.testing.assert_array_equal(
        df_sync.loc[df_sync["Time"] == 0, "Amplitude"], sw_n3.data[[2000, 4000]]
    )


def test_get_mask_rounding(data):
    """get_mask includes the first and last sample of each event."""
    sp = spindles_detect(data, sf)
    events = sp._events.iloc[:1].copy()
    events[["Start", "End"]] = [0.29, 0.57]  # 0.29 * 100 = 28.999999999999996
    sp._events = events
    np.testing.assert_array_equal(np.flatnonzero(sp.get_mask()), np.arange(29, 58))


##############################################################################
# REMs
##############################################################################


# Representative parameter combinations (covers all values, avoids full Cartesian product of 16
# iterations)
@pytest.mark.parametrize(
    "freq_rem, duration, amplitude, with_hypno",
    [
        ((0.5, 5), (0.3, 1.5), (50, 200), True),
        ((0.3, 8), [0.5, 1], (50, 200), False),
        ((0.5, 5), [0.5, 1], [60, 300], False),
        ((0.3, 8), (0.3, 1.5), [60, 300], True),
    ],
)
def test_rem_params(eog, freq_rem, duration, amplitude, with_hypno):
    """The duration of the REMs is within the requested range, Stage only with a hypnogram."""
    hypno = eog.hypno if with_hypno else None
    rem = rem_detect(
        eog.loc,
        eog.roc,
        eog.sf,
        hypno=hypno,
        freq_rem=freq_rem,
        duration=duration,
        amplitude=amplitude,
    )
    rem_sum = rem.summary()
    assert rem_sum["Duration"].between(duration[0], duration[1], inclusive="left").all()
    assert ("Stage" in rem_sum) == (hypno is not None)


@pytest.fixture(scope="module")
def rem_default(eog):
    return rem_detect(eog.loc, eog.roc, eog.sf, verbose="info")


@pytest.fixture(scope="module")
def rem_no_out(eog):
    """With isolation forest."""
    return rem_detect(eog.loc, eog.roc, eog.sf, remove_outliers=True)


def test_rem_default(rem_default, rem_no_out, eog):
    """Outlier removal, mask, sync events and plot."""
    rem = rem_default
    assert rem.summary().shape[0] > rem_no_out.summary().shape[0]
    assert rem.get_mask().shape == (2, eog.loc.size)
    rem.plot_detection()
    df_sync = rem.get_sync_events()
    assert df_sync["Channel"].unique().tolist() == ["LOC", "ROC"]
    assert df_sync["Event"].nunique() == rem.summary().shape[0]
    rem.plot_average(filt=(0.5, 5), errorbar=None)


def test_rem_compare_detection(rem_default, rem_no_out):
    """compare_detection works on the combined LOC and ROC channels."""
    res = rem_default.compare_detection(rem_no_out)
    assert res.index.tolist() == ["LOC-ROC"]
    assert res.loc["LOC-ROC", "recall"] == 1
    assert res.loc["LOC-ROC", "n_other"] == rem_no_out.summary().shape[0]
    # ... also with a dataframe of annotations without Channel column
    res_df = rem_default.compare_detection(rem_no_out.summary()[["Start"]])
    pd.testing.assert_frame_equal(res, res_df)


@pytest.mark.parametrize("method", ["compare_channels", "get_coincidence_matrix"])
def test_rem_single_combined_channel(rem_default, method):
    """There is a single (combined) channel, so channels cannot be compared."""
    with pytest.raises(NotImplementedError):
        getattr(rem_default, method)()


def test_rem_hypno(eog):
    """With a REM hypnogram, REMs are only detected in REM sleep."""
    half = int(eog.loc.size / 2)
    rem = rem_detect(eog.loc, eog.roc, eog.sf, hypno=eog.hypno)
    hypno_rem_half = np.r_[np.ones(half), 4 * np.ones(half)]
    rem2 = rem_detect(eog.loc, eog.roc, eog.sf, hypno=hypno_rem_half)
    assert rem.summary().shape[0] > rem2.summary().shape[0]
    assert (rem2.summary()["Start"] >= eog.loc.size / 2 / eog.sf).all()
    rem_stage = rem2.summary(grp_stage=True, aggfunc="median")
    assert rem_stage.index.tolist() == [4]
    # No values in hypno intersect with include
    with pytest.raises(AssertionError):
        rem_detect(eog.loc, eog.roc, eog.sf, hypno=hypno_rem_half, include=5)


@pytest.mark.parametrize("loc_factor, roc_factor", [(1e-8, 1), (1, 1e8)])
def test_rem_wrong_amplitude(eog, loc_factor, roc_factor):
    """Data with a wrong amplitude on LOC or ROC logs an error and returns None."""
    with assert_logs("ERROR"):
        rem = rem_detect(eog.loc * loc_factor, eog.roc * roc_factor, eog.sf)
    assert rem is None


def test_rem_hypnogram_str_include(eog):
    """Hypnogram instance + string include: same as the integer include."""
    n_epochs = int(np.ceil(eog.loc.size / (30 * eog.sf)))
    hyp_rem = Hypnogram(["REM"] * n_epochs, freq="30s")
    rem_hyp = rem_detect(eog.loc, eog.roc, eog.sf, hypno=hyp_rem, include="REM")
    assert rem_hyp is not None
    rem_int = rem_detect(eog.loc, eog.roc, eog.sf, hypno=hyp_rem, include=4)
    pd.testing.assert_frame_equal(rem_hyp.summary(), rem_int.summary())


##############################################################################
# ARTEFACTS
##############################################################################

N_FLAT = 20000  # Number of flat samples added at the end of the 9-channel data
ALL_STAGES = (0, 1, 2, 3, 4, 5, 6)


@pytest.fixture(scope="module")
def art_data(full_6hrs_9ch, full_6hrs):
    """One hour of 9-channel data, followed by some flat data, and its hypnogram (read-only)."""
    data_9 = full_6hrs_9ch.data[:, :360_000]
    hypno_9 = full_6hrs.hypno[:360_000].astype(np.int64)
    # For the sake of the example, let's add some flat data at the end
    data_9 = np.concatenate((data_9, np.zeros((data_9.shape[0], N_FLAT))), axis=1)
    hypno_9 = np.concatenate((hypno_9, np.zeros(N_FLAT, dtype=np.int64)))
    data_9.setflags(write=False)
    hypno_9.setflags(write=False)
    return SimpleNamespace(data=data_9, hypno=hypno_9)


def test_art_covar(art_data):
    """Covariance-based method: one boolean and z-score per epoch."""
    art, zscores = art_detect(art_data.data, sf=100, window=10, method="covar", threshold=3)
    assert art.shape == zscores.shape == (art_data.data.shape[1] // (10 * sf),)
    assert art.dtype == bool


def test_art_std(art_data):
    """Std-based method. The flat epochs at the end are artefacts, with z-scores set to NaN."""
    n_flat = N_FLAT // (5 * sf)
    art, zscores = art_detect(art_data.data, sf=100, window=5, method="std", threshold=2)
    assert zscores.shape == (art.size, art_data.data.shape[0])
    assert art[-n_flat:].all()
    assert np.isnan(zscores[-n_flat:]).all()
    assert 0 < art[:-n_flat].mean() < 0.2


def test_art_std_hypno(art_data):
    """Std-based method with hypnogram, and effect of the threshold."""
    n_flat = N_FLAT // (5 * sf)
    # Writable copy: check that the hypnogram is not modified by the flagging of flat epochs
    hypno_9 = art_data.hypno.copy()
    kwargs = dict(sf=100, hypno=hypno_9, method="std", include=ALL_STAGES)
    art, _ = art_detect(art_data.data, window=5, threshold=2, **kwargs)
    assert art[-n_flat:].all()
    np.testing.assert_array_equal(hypno_9, art_data.hypno)
    # A higher threshold rejects fewer epochs
    art_10, _ = art_detect(art_data.data, window=5.0, threshold=10, **kwargs)
    assert art_10.sum() < art.sum()


def test_art_single_channel(art_data):
    """With a single channel, method="covar" switches to "std"."""
    art, _ = art_detect(art_data.data[0], 100, window=10, method="covar")
    assert art.shape == (art_data.data.shape[1] // (10 * sf),)


@pytest.mark.parametrize("n_chan, logged", [(4, False), (3, True)])
def test_art_covar_n_chan(art_data, n_chan, logged):
    """Exactly 4 channels is enough for method="covar", otherwise a warning is logged."""
    with assert_logs("WARNING", logged=logged):
        art_detect(art_data.data[:n_chan, :360_000], sf, window=10, method="covar")


def test_art_invalid_method(art_data):
    with pytest.raises(ValueError, match="Invalid method"):
        art_detect(art_data.data, sf, method="wrong")


def test_art_not_enough_epochs(art_data):
    """Not enough epochs for stage: a warning is logged."""
    hypno_9 = art_data.hypno.copy()
    hypno_9[:100] = 6
    with assert_logs("WARNING"):
        art_detect(
            art_data.data,
            sf,
            window=5.0,
            hypno=hypno_9,
            include=6,
            method="std",
            threshold=3,
            n_chan_reject=5,
        )


def test_art_flat_channel(art_data):
    """A flat channel is removed with a warning."""
    data_with_flat = np.vstack((art_data.data, np.zeros(art_data.data.shape[-1])))
    with assert_logs("WARNING"):
        _, zscores = art_detect(data_with_flat, sf, method="std", n_chan_reject=5)
    assert zscores.shape[1] == art_data.data.shape[0]


def test_art_mne(mne_n2n3):
    """Using a MNE Raw object with an integer hypnogram."""
    raw = mne_n2n3.raw
    art, _ = art_detect(raw, window=10.0, hypno=mne_n2n3.hypno, method="covar")
    assert art.shape == (raw.n_times // (10 * sf),)
    with pytest.raises(AssertionError):
        # None of include in hypno
        art_detect(raw, window=10.0, hypno=mne_n2n3.hypno, include=[7, 8])


def test_art_hypnogram_str_include(art_data):
    """Hypnogram instance + string include: same as the upsampled integer hypnogram."""
    hyp_9 = Hypnogram.from_integers(art_data.hypno[:: 30 * sf], freq="30s")
    art_hyp, _ = art_detect(
        art_data.data, sf=100, window=6, hypno=hyp_9, include=["N2", "N3"], method="covar"
    )
    assert art_hyp.shape == (art_data.data.shape[1] // (6 * sf),)
    hypno_9_up = hyp_9.upsample_to_data(art_data.data, sf=sf, verbose="error")
    art_int, _ = art_detect(
        art_data.data, sf=100, window=6, hypno=hypno_9_up, include=(2, 3), method="covar"
    )
    np.testing.assert_array_equal(art_hyp, art_int)


##############################################################################
# compare_detection
##############################################################################

DETECTED = [5, 12, 20, 34, 41, 57, 63]
GRNDTRTH = [5, 12, 18, 26, 34, 41, 55, 63, 68]
F1_DEFAULT = hmean([5 / 7, 5 / 9])


def test_compare_detection_default():
    res = compare_detection(DETECTED, GRNDTRTH)
    assert all(res["tp"] == [5, 12, 34, 41, 63])
    assert all(res["fp"] == [20, 57])
    assert all(res["fn"] == [18, 26, 55, 68])
    assert np.isclose(res["precision"], 5 / 7)
    assert np.isclose(res["recall"], 5 / 9)
    assert np.isclose(res["f1"], F1_DEFAULT)


def test_compare_detection_swapped():
    """Changing the order: FN <--> FP, precision <--> recall. No change in F1-score."""
    res = compare_detection(GRNDTRTH, DETECTED)
    assert all(res["tp"] == [5, 12, 34, 41, 63])
    assert all(res["fn"] == [20, 57])
    assert all(res["fp"] == [18, 26, 55, 68])
    assert np.isclose(res["precision"], 5 / 9)
    assert np.isclose(res["recall"], 5 / 7)
    assert np.isclose(res["f1"], F1_DEFAULT)


def test_compare_detection_max_distance():
    res = compare_detection(DETECTED, GRNDTRTH, max_distance=2)
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
    res = compare_detection(DETECTED, GRNDTRTH, max_distance=100)
    assert res["precision"] == res["recall"] == 1


def test_compare_detection_empty():
    """Detected or ground-truth is empty."""
    res = compare_detection([], GRNDTRTH)
    assert len(res["tp"]) == 0
    assert len(res["fp"]) == 0
    assert all(res["fn"] == GRNDTRTH)

    res = compare_detection(DETECTED, [])
    assert len(res["tp"]) == 0
    assert all(res["fp"] == DETECTED)
    assert len(res["fn"]) == 0


@pytest.mark.parametrize(
    "detected, grndtrth",
    [
        # Detected is not sorted
        (np.random.default_rng(42).permutation(DETECTED), GRNDTRTH),
        # Detected has duplicate values
        ([5, 12, 12, 20, 34, 41, 41, 57, 63], GRNDTRTH),
        # Different dtypes
        (np.array(DETECTED, dtype=float), np.array(GRNDTRTH, dtype=int)),
        ([5.0, 12, 20.0, 34, 41.0, 57.0, 63], pd.Series(GRNDTRTH)),
    ],
    ids=["unsorted", "duplicates", "float-int", "list-series"],
)
def test_compare_detection_inputs(detected, grndtrth):
    """Same F1-score as the default example."""
    res = compare_detection(detected, grndtrth)
    assert np.isclose(res["f1"], F1_DEFAULT)


def test_compare_detection_errors():
    with pytest.raises(AssertionError):
        # Arrays contain non-integer floats
        compare_detection([5.4, 12.2, 20], [5, 12.3, 18])

    with pytest.raises(AssertionError):
        # max_distance must be a positive integer
        compare_detection(DETECTED, GRNDTRTH, max_distance=-1)
