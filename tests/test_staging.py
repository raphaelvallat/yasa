"""Test the functions in yasa/staging.py."""

import logging
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.stats as sp_stats
from mne.filter import filter_data

from yasa.hypno import Hypnogram
from yasa.others import sliding_window
from yasa.staging import SleepStaging

##############################################################################
# FIXTURES
##############################################################################


@pytest.fixture
def yasa_caplog(caplog, monkeypatch):
    """caplog that also captures the messages of the YASA logger (which does not propagate)."""
    monkeypatch.setattr(logging.getLogger("yasa"), "propagate", True)
    return caplog


@pytest.fixture(scope="module")
def y_true(hypno_sub02):
    """Human-scored hypnogram of sub-02."""
    return Hypnogram(hypno_sub02)


@pytest.fixture(scope="module")
def sls_full(_raw_sub02):
    """SleepStaging with EEG, EOG, EMG and metadata, after predict()."""
    # SleepStaging does not modify the Raw in place
    sls = SleepStaging(
        _raw_sub02,
        eeg_name="C4",
        eog_name="EOG1",
        emg_name="EMG1",
        metadata=dict(age=21, male=False),
    )
    sls.get_features()
    y_pred = sls.predict()
    return sls, y_pred


@pytest.fixture(scope="module")
def sls_eeg(_raw_sub02):
    """SleepStaging with only the EEG, after fit()."""
    sls = SleepStaging(_raw_sub02, eeg_name="C4")
    sls.fit()
    return sls


##############################################################################
# TESTS
##############################################################################


def test_sleep_staging_repr(sls_full):
    """Test the string representation of SleepStaging."""
    sls, _ = sls_full
    assert str(sls) == repr(sls)
    assert "(49.0 minutes)" in str(sls)


def test_sleep_staging_predict(sls_full, y_true):
    """Test the predicted hypnogram."""
    _, y_pred = sls_full
    assert isinstance(y_pred, Hypnogram)
    assert y_pred.proba is not None
    # Classifier labels "W" and "R" are converted to canonical stage names
    assert y_pred.proba.columns.tolist() == ["WAKE", "N1", "N2", "N3", "REM"]
    assert y_pred.hypno.size == y_true.hypno.size
    assert y_true.duration == y_pred.duration
    assert y_true.n_stages == y_pred.n_stages
    # Check that the accuracy is at least 80%
    # Compare values directly (indexes differ: y_true has integer Epoch, y_pred has Time)
    accuracy = (y_true.hypno.to_numpy() == y_pred.hypno.to_numpy()).mean()
    assert accuracy > 0.80


def test_plot_predict_proba(sls_full):
    """Test plot_predict_proba."""
    sls, y_pred = sls_full
    sls.plot_predict_proba()
    sls.plot_predict_proba(y_pred.proba, majority_only=True)
    plt.close("all")


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(eog_name="EOG1", emg_name="EMG1"),  # without metadata
        dict(eog_name="EOG1"),  # without EMG
        dict(),  # just the EEG
    ],
    ids=["no_metadata", "no_emg", "eeg_only"],
)
def test_sleep_staging_predictors(_raw_sub02, kwargs):
    """Same with different combinations of predictors."""
    SleepStaging(_raw_sub02, eeg_name="C4", **kwargs).fit()


def test_short_data_warning(raw_sub02, yasa_caplog):
    """Test that a warning is raised for recordings shorter than 5 minutes."""
    SleepStaging(raw_sub02.crop(tmax=200), eeg_name="C4")
    assert any(r.levelno == logging.WARNING for r in yasa_caplog.records)


def test_validate_predict_errors(sls_eeg):
    """Test _validate_predict raises ValueError for mismatched features."""
    # Features in clf not present in current feature set
    clf_mock = MagicMock()
    clf_mock.feature_name_ = ["nonexistent_feature"]
    with pytest.raises(ValueError):
        sls_eeg._validate_predict(clf_mock)

    # Features in current set not present in clf
    clf_mock.feature_name_ = sls_eeg.feature_name_[:-1]
    with pytest.raises(ValueError):
        sls_eeg._validate_predict(clf_mock)


def test_plot_predict_proba_no_predict(_raw_sub02):
    """Test that plot_predict_proba raises ValueError before predict is called."""
    sls = SleepStaging(_raw_sub02, eeg_name="C4")
    with pytest.raises(ValueError):
        sls.plot_predict_proba()
    sls.fit()
    with pytest.raises(ValueError):
        sls.plot_predict_proba()


def test_metadata_not_modified(_raw_sub02):
    """Test that the metadata dict of the caller is not modified, and that {} = None."""
    metadata = dict(age=21, male=True)
    SleepStaging(_raw_sub02, eeg_name="C4", metadata=metadata)
    assert metadata["male"] is True
    assert SleepStaging(_raw_sub02, eeg_name="C4", metadata={}).metadata is None


@pytest.mark.parametrize(
    "metadata, expected", [(dict(age=21), {"age": 21}), (dict(male=True), {"male": 1})]
)
def test_partial_metadata(_raw_sub02, metadata, expected):
    """Partial metadata."""
    assert SleepStaging(_raw_sub02, eeg_name="C4", metadata=metadata).metadata == expected


def test_very_short_data(raw_sub02):
    """Test that one epoch of data works."""
    features = SleepStaging(raw_sub02.crop(tmax=45), eeg_name="C4").get_features()
    assert features.shape[0] == 1
    assert features["time_norm"].iloc[0] == 0


def test_less_than_one_epoch(raw_sub02):
    """Test that less than one epoch raises an error."""
    with pytest.raises(ValueError):
        SleepStaging(raw_sub02.crop(tmax=20), eeg_name="C4")


def test_features_skew_kurt(sls_eeg):
    """Test that skew and kurtosis match scipy."""
    features = sls_eeg.get_features()
    data_filt = filter_data(sls_eeg.data, sls_eeg.sf, l_freq=0.4, h_freq=30, verbose=False)
    _, epochs = sliding_window(data_filt[0], sf=sls_eeg.sf, window=30)
    np.testing.assert_allclose(
        features["eeg_skew"], sp_stats.skew(epochs, axis=1).astype(np.float32), rtol=1e-5
    )
    np.testing.assert_allclose(
        features["eeg_kurt"], sp_stats.kurtosis(epochs, axis=1).astype(np.float32), rtol=1e-5
    )


def test_cropped_raw_start(raw_sub02):
    """Test that the predicted hypnogram start accounts for cropping of the Raw."""
    raw_cropped = raw_sub02.set_meas_date(pd.Timestamp("2022-01-01 23:00:00", tz="UTC"))
    raw_cropped.crop(tmin=150)
    hyp = SleepStaging(raw_cropped, eeg_name="C4").predict()
    assert hyp.start == pd.Timestamp("2022-01-01 23:02:30", tz="UTC")


def test_cropped_raw_no_meas_date(raw_sub02):
    """No measurement date."""
    raw_undated = raw_sub02.set_meas_date(None).crop(tmin=150)
    assert SleepStaging(raw_undated, eeg_name="C4").predict().start is None
