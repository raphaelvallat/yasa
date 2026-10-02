"""Shared pytest configuration and fixtures.

The sample data is downloaded from Zenodo (and cached) by :py:func:`yasa.fetch_sample`. It is
loaded in session-scoped fixtures, so that test collection works offline and each file is only
read once per session. The arrays are read-only so that a test cannot silently modify the data
seen by the other tests: use ``.copy()`` before modifying them in place.
"""

import logging
from types import SimpleNamespace

import matplotlib
import mne
import numpy as np
import pytest

from yasa.fetchers import fetch_sample

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _restore_yasa_log_level():
    """Restore the level of the YASA logger after each test, e.g. after ``set_log_level``."""
    logger = logging.getLogger("yasa")
    level = logger.level
    yield
    logger.setLevel(level)


def _readonly(*arrays):
    for arr in arrays:
        arr.setflags(write=False)


@pytest.fixture(scope="session")
def n2_spindles():
    """15 seconds of N2 sleep with spindles, single channel, 200 Hz."""
    data = np.loadtxt(fetch_sample("N2_spindles_15sec_200Hz.txt"))
    _readonly(data)
    return SimpleNamespace(data=data, sf=200)


@pytest.fixture(scope="session")
def n3_no_spindles():
    """30 seconds of N3 sleep without spindles, single channel, 100 Hz."""
    data = np.loadtxt(fetch_sample("N3_no-spindles_30sec_100Hz.txt"))
    _readonly(data)
    return SimpleNamespace(data=data, sf=100)


@pytest.fixture(scope="session")
def full_6hrs():
    """6 hours of sleep EEG (Cz, Fz, Pz) at 100 Hz, with the hypnogram upsampled to the data."""
    file = np.load(fetch_sample("full_6hrs_100Hz_Cz+Fz+Pz.npz"))
    data, chan = file["data"], file["chan"]
    hypno = np.load(fetch_sample("full_6hrs_100Hz_hypno.npz"))["hypno"]
    _readonly(data, chan, hypno)
    return SimpleNamespace(data=data, chan=chan, hypno=hypno, sf=100)


@pytest.fixture(scope="session")
def full_6hrs_9ch():
    """6 hours of sleep EEG (9 channels) at 100 Hz. Use the hypnogram of ``full_6hrs``."""
    file = np.load(fetch_sample("full_6hrs_100Hz_9channels.npz"))
    data, chan = file["data"], file["ch_names"]
    _readonly(data, chan)
    return SimpleNamespace(data=data, chan=chan, sf=100)


@pytest.fixture(scope="session")
def raw_sub02_shared():
    """Polysomnography of sub-02 as a preloaded MNE Raw, shared by all tests: never modify it.

    Use it in module-scoped fixtures, or call ``.copy()`` before cropping or picking channels.
    """
    return mne.io.read_raw_fif(fetch_sample("sub-02_mne_raw.fif"), preload=True, verbose=False)


@pytest.fixture
def raw_sub02(raw_sub02_shared):
    """Polysomnography of sub-02 as a preloaded MNE Raw (a new copy for each test)."""
    return raw_sub02_shared.copy()


@pytest.fixture(scope="session")
def hypno_sub02():
    """Hypnogram of sub-02 (one string stage per 30-second epoch)."""
    hypno = np.loadtxt(fetch_sample("sub-02_hypno_30s.txt"), dtype=str)
    _readonly(hypno)
    return hypno


@pytest.fixture(scope="session")
def eog_rem():
    """LOC and ROC channels during REM sleep."""
    file = np.load(fetch_sample("EOGs_REM_256Hz.npz"))
    loc, roc = file["data"]
    _readonly(loc, roc)
    return SimpleNamespace(loc=loc, roc=roc, sf=float(file["sf"]))


@pytest.fixture(scope="session")
def ecg_8hrs():
    """8 hours of ECG at 200 Hz, with the hypnogram upsampled to the data."""
    file = np.load(fetch_sample("ECG_8hrs_200Hz.npz"))
    data, hypno = file["data"], file["hypno"]
    _readonly(data, hypno)
    return SimpleNamespace(data=data, sf=int(file["sf"]), hypno=hypno)


@pytest.fixture(scope="session")
def raw_resting_eo_shared():
    """Resting-state EEG with eyes open as an MNE Raw, shared by all tests: never modify it."""
    return mne.io.read_raw_fif(fetch_sample("resting_EO_200Hz_raw.fif"), verbose=False)


@pytest.fixture
def raw_resting_eo(raw_resting_eo_shared):
    """Resting-state EEG with eyes open as a (not preloaded) MNE Raw (a new copy for each test)."""
    return raw_resting_eo_shared.copy()
