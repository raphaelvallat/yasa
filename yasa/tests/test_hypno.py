"""Test the functions in the yasa/hypno.py file."""
import mne
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from yasa.hypno import (
    hypno_str_to_int,
    hypno_int_to_str,
    hypno_upsample_to_sf,
    hypno_fit_to_data,
    hypno_upsample_to_data,
)

from yasa.hypno import hypno_find_periods as hfp

hypno = np.array([0, 0, 0, 1, 2, 2, 3, 3, 4])
hypno_txt = np.array(["W", "W", "W", "N1", "N2", "N2", "N3", "N3", "R"])


def create_raw(npts, ch_names=["F4-M1", "F3-M2"], sf=100):
    """Utility function for test fit to data."""
    nchan = len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=sf, ch_types=["eeg"] * nchan, verbose=0)
    data = np.random.rand(nchan, npts)
    raw = mne.io.RawArray(data, info, verbose=0)
    return raw


class TestHypno(unittest.TestCase):
    """Test functions in the hypno.py file."""

    def test_conversion(self):
        """Test str <--> int conversion."""
        assert np.array_equal(hypno_str_to_int(hypno_txt), hypno)
        assert np.array_equal(hypno_int_to_str(hypno), hypno_txt)

    def test_upsampling(self):
        """Test hypnogram upsampling."""
        hypno100 = hypno_upsample_to_sf(hypno=hypno, sf_hypno=(1 / 30), sf_data=100)
        nhyp100 = hypno100.size
        assert nhyp100 / hypno.size == 3000
        assert hypno100[hypno100 == 2].size / hypno[hypno == 2].size == 3000
        # Test pass trough
        assert np.array_equal(hypno_upsample_to_sf(hypno, 1, 1), hypno)

        # Now test fit to data
        # .. Using MNE Raw
        assert np.array_equal(hypno_fit_to_data(hypno100, create_raw(nhyp100)), hypno100)
        assert hypno_fit_to_data(hypno100, create_raw(27250)).size == 27250
        assert hypno_fit_to_data(hypno100, create_raw(26750)).size == 26750
        # .. Using Numpy + SF
        from numpy.random import rand

        assert np.array_equal(hypno_fit_to_data(hypno100, rand(nhyp100), 100), hypno100)
        assert hypno_fit_to_data(hypno100, rand(27250), 100).size == 27250
        assert hypno_fit_to_data(hypno100, rand(26750), 100).size == 26750
        # .. No SF
        assert np.array_equal(hypno_fit_to_data(hypno100, rand(nhyp100)), hypno100)
        assert hypno_fit_to_data(hypno100, rand(27250)).size == 27250
        assert hypno_fit_to_data(hypno100, rand(26750)).size == 26750

        # Two steps combined
        assert hypno_upsample_to_data(hypno, sf_hypno=1 / 30, data=create_raw(26750)).size == 26750
        assert (
            hypno_upsample_to_data(hypno, sf_hypno=1 / 30, data=rand(27250), sf_data=100).size
            == 27250
        )
        assert (
            hypno_upsample_to_data(
                hypno, sf_hypno=1 / 30, data=rand(2 * (hypno100.size + 250)), sf_data=200
            ).size
            == 2 * 27250
        )

    def test_periods(self):
        """Test periods detection."""
        # TEST 1: BINARY VECTOR
        x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]

        # 1a. No thresholding
        expected = pd.DataFrame(
            {"values": [0, 1, 0, 1, 0], "start": [0, 11, 14, 16, 25], "length": [11, 3, 2, 9, 2]}
        )

        kwargs = dict(
            check_dtype=False,
            check_index_type=False,
            check_column_type=False,
            check_frame_type=False,
        )
        assert_frame_equal(hfp(x, sf_hypno=1 / 60, threshold="0min"), expected, **kwargs)
        assert_frame_equal(hfp(x, sf_hypno=1, threshold="0min"), expected, **kwargs)

        # 1b. With thresholding
        expected = pd.DataFrame({"values": [0, 1], "start": [0, 16], "length": [11, 9]})
        assert_frame_equal(hfp(x, sf_hypno=1 / 60, threshold="5min"), expected, **kwargs)
        assert hfp(x, sf_hypno=1, threshold="5min").size == 0

        # 1c. Equal length
        expected = pd.DataFrame(
            {
                "values": [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0],
                "start": [0, 2, 4, 6, 8, 11, 14, 16, 18, 20, 22, 25],
                "length": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            }
        )
        assert_frame_equal(
            hfp(x, sf_hypno=1 / 60, threshold="2min", equal_length=True), expected, **kwargs
        )

        # TEST 2: MULTI-CLASS VECTOR
        x = [0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 0, 1]

        expected = pd.DataFrame(
            {
                "values": [0, 1, 2, 0, 1, 0, 1],
                "start": [0, 4, 5, 11, 14, 15, 16],
                "length": [4, 1, 6, 3, 1, 1, 1],
            }
        )
        assert_frame_equal(hfp(x, sf_hypno=1 / 60, threshold="0min"), expected, **kwargs)

        # With a string dtype
        expected["values"] = expected["values"].astype(str)
        assert_frame_equal(
            hfp(np.array(x).astype(str), sf_hypno=1 / 60, threshold="0min"), expected, **kwargs
        )
