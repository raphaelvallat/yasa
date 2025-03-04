"""Test the fetchers module."""

import os
import unittest
from tempfile import TemporaryDirectory

from yasa import fetchers

SAMPLES_V1_REPOSITORY_DOI = "10.5281/zenodo.14564285"

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


class TestFetchers(unittest.TestCase):
    """Test fetchers functions"""

    def test_repository_initialization(self):
        """Test that the DOI repo initializer works"""
        doi = SAMPLES_V1_REPOSITORY_DOI
        pup = fetchers._init_doi_repository(doi, populate_registry=False)
        assert len(pup.registry) == 0
        pup = fetchers._init_doi_repository(doi, populate_registry=True)
        repository_files = sorted(pup.registry_files)
        assert repository_files == sorted(ALL_SAMPLE_V1_FILES)

    def test_file_download(self):
        """Test the download of a single arbitrary file from the samples repo"""
        with TemporaryDirectory() as tempdir:
            os.environ["YASA_DATA_DIR"] = tempdir
            fp = fetchers.fetch_sample(SMALL_SAMPLE_FILE)
            assert fp.exists()
            assert fp.is_file()
            assert os.path.dirname(fp) == tempdir

    def test_version_picker(self):
        """Test the version parameter in sample fetcher"""
        with TemporaryDirectory() as tempdir:
            os.environ["YASA_DATA_DIR"] = tempdir
            fp = fetchers.fetch_sample(SMALL_SAMPLE_FILE, version="latest")
            assert fp.exists() and fp.is_file()

    def test_fetch_kwargs(self):
        """Test passing of kwargs to Pooch.fetch by printing progress bar"""
        with TemporaryDirectory() as tempdir:
            os.environ["YASA_DATA_DIR"] = tempdir
            fetchers.fetch_sample(SMALL_SAMPLE_FILE, progressbar=True)
