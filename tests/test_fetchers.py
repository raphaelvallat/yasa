"""Test the fetchers module."""

from pathlib import Path

import pytest

from yasa import fetchers

SMALL_SAMPLE_FILE = "sub-02_hypno_30s.txt"

ALL_SAMPLE_V1_FILES = [
    "ECG_8hrs_200Hz.npz",
    "EOGs_REM_256Hz.npz",
    "N2_spindles_15sec_200Hz.txt",
    "N3_no-spindles_30sec_100Hz.txt",
    "full_6hrs_100Hz_9channels.npz",
    "full_6hrs_100Hz_Cz+Fz+Pz.npz",
    "full_6hrs_100Hz_hypno.npz",
    "full_6hrs_100Hz_hypno_30s.txt",
    "night_young.edf",
    "night_young_hypno.csv",
    "resting_EO_200Hz_raw.fif",
    "sub-02_hypno_30s.txt",
    "sub-02_mne_raw.fif",
]


def test_repository_initialization():
    """Test that the DOI repo initializer works"""
    pup = fetchers._init_repository("sample", version="v1")
    assert sorted(pup.registry_files) == sorted(ALL_SAMPLE_V1_FILES)


def test_repository_data_dir(tmp_path, monkeypatch):
    """Test that the YASA_DATA_DIR environment variable sets the cache directory"""
    monkeypatch.setenv("YASA_DATA_DIR", str(tmp_path))
    pup = fetchers._init_repository("sample", version="v1")
    assert Path(pup.abspath) == tmp_path


def test_version_picker():
    """Test the version parameter in sample fetcher"""
    with pytest.raises(AssertionError, match="`version` must be one of"):
        fetchers.fetch_sample(SMALL_SAMPLE_FILE, version="999")


@pytest.mark.network
def test_file_download(tmp_path, monkeypatch):
    """Test the download of a single arbitrary file from the samples repo"""
    monkeypatch.setenv("YASA_DATA_DIR", str(tmp_path))
    fp = fetchers.fetch_sample(SMALL_SAMPLE_FILE)
    assert fp.exists()
    assert fp.is_file()
    assert fp.parent == tmp_path


@pytest.mark.network
def test_fetch_kwargs(tmp_path, monkeypatch):
    """Test passing of kwargs to Pooch.fetch by printing progress bar"""
    monkeypatch.setenv("YASA_DATA_DIR", str(tmp_path))
    fetchers.fetch_sample(SMALL_SAMPLE_FILE, progressbar=True)
