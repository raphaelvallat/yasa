"""Test the fetchers module."""

import os
import unittest
from tempfile import TemporaryDirectory

from yasa import fetchers

EXAMPLES_REPOSITORY_DOIS = {
    "latest": "10.5281/zenodo.14564284",
    "v1": "10.5281/zenodo.14564285",
}

REQUIRED_SAMPLE_FILES = [
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

SMALL_SAMPLE_FILE = "sub-02_hypno_30s.txt"


class TestFetchers(unittest.TestCase):
    """Test fetchers functions"""

    def test_examples_repo(self):
        """Test that the examples repo has all necessary files"""
        doi = EXAMPLES_REPOSITORY_DOIS["v1"]
        pup = fetchers._init_doi_repository(doi, populate_registry=True)
        repository_files = sorted(pup.registry_files)
        assert repository_files == sorted(REQUIRED_SAMPLE_FILES)

    def test_examples_download(self):
        """Test the download of a single arbitrary file from the examples repo"""
        with TemporaryDirectory() as tempdir:
            os.environ["YASA_DATA_DIR"] = tempdir
            fp = fetchers.fetch_example(SMALL_SAMPLE_FILE)
            assert fp.exists()
            assert fp.is_file()
