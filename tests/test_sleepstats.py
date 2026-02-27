"""Test the functions in the yasa/sleepstats.py file."""

import unittest

import numpy as np
import pandas as pd
import pytest

from yasa.hypno import Hypnogram, simulate_hypnogram
from yasa.sleepstats import sleep_statistics, transition_matrix

hypno = np.array([0, 0, 0, 1, 2, 2, 3, 3, 2, 2, 2, 0, 0, 0, 2, 2, 4, 4, 0, 0])


class TestSleepStats(unittest.TestCase):
    def test_transition(self):
        """Test transition_matrix"""
        a = [1, 1, 1, 0, 0, 2, 2, 0, 2, 0, 1, 1, 0, 0]
        counts, probs = transition_matrix(a)
        c = np.array([[2, 1, 2], [2, 3, 0], [2, 0, 1]])
        p = np.array([[0.4, 0.2, 0.4], [0.4, 0.6, 0], [2 / 3, 0, 1 / 3]])
        assert pd.DataFrame(c).equals(counts)
        assert pd.DataFrame(p).equals(probs)
        assert (probs.sum(axis=1) == 1).all()
        # Second example, with only Wake, N2 and REM
        x = np.asarray([0, 2, 2, 0, 0, 2, 0, 4, 4, 0, 0])
        counts, probs = transition_matrix(x)
        c = np.array([[2, 2, 1], [2, 1, 0], [1, 0, 1]])
        p = np.array([[0.4, 0.4, 0.2], [2 / 3, 1 / 3, 0], [0.5, 0, 0.5]])
        assert pd.DataFrame(c, index=[0, 2, 4], columns=[0, 2, 4]).equals(counts)
        assert pd.DataFrame(p, index=[0, 2, 4], columns=[0, 2, 4]).equals(probs)
        assert (probs.sum(axis=1) == 1).all()

    def test_transition_hypnogram(self):
        """Test that transition_matrix accepts a Hypnogram and returns string-labelled output."""
        # --- 5-stage: standalone function == instance method ---
        hyp = simulate_hypnogram(tib=480, seed=42)
        counts_fn, probs_fn = transition_matrix(hyp)
        counts_method, probs_method = hyp.transition_matrix()
        pd.testing.assert_frame_equal(counts_fn, counts_method)
        pd.testing.assert_frame_equal(probs_fn, probs_method)

        # Output labels are strings, not integers (works with both object and StringDtype)
        assert all(isinstance(label, str) for label in counts_fn.index)
        assert all(isinstance(label, str) for label in counts_fn.columns)

        # Probabilities are a right-stochastic matrix (each row sums to 1)
        np.testing.assert_allclose(probs_fn.sum(axis=1), 1.0)

        # --- 2-stage ---
        hyp2 = simulate_hypnogram(tib=120, n_stages=2, seed=1)
        counts2, probs2 = transition_matrix(hyp2)
        assert set(counts2.index).issubset({"WAKE", "SLEEP", "ART", "UNS"})
        np.testing.assert_allclose(probs2.sum(axis=1), 1.0)

        # --- 3-stage ---
        hyp3 = simulate_hypnogram(tib=240, n_stages=3, seed=2)
        counts3, probs3 = transition_matrix(hyp3)
        assert set(counts3.index).issubset({"WAKE", "NREM", "REM", "ART", "UNS"})
        np.testing.assert_allclose(probs3.sum(axis=1), 1.0)

        # --- Small known example: verify counts exactly ---
        hyp_known = Hypnogram(["W", "N1", "N2", "N3", "N2", "REM", "W"])
        counts_k, probs_k = transition_matrix(hyp_known)
        # Transitions: W→N1, N1→N2, N2→N3, N3→N2, N2→REM, REM→W
        assert counts_k.loc["WAKE", "N1"] == 1
        assert counts_k.loc["N1", "N2"] == 1
        assert counts_k.loc["N2", "N3"] == 1
        assert counts_k.loc["N2", "REM"] == 1
        assert counts_k.loc["N3", "N2"] == 1
        assert counts_k.loc["REM", "WAKE"] == 1
        assert counts_k.loc["WAKE", "WAKE"] == 0
        # Row sums equal total transitions out of each stage
        assert counts_k.loc["N2"].sum() == 2  # N2→N3 and N2→REM
        np.testing.assert_allclose(probs_k.loc["N2", "N3"], 0.5)
        np.testing.assert_allclose(probs_k.loc["N2", "REM"], 0.5)

        # --- Consistency: integer array input still uses integer labels ---
        int_arr = [0, 1, 2, 3, 2, 4, 0]
        counts_int, _ = transition_matrix(int_arr)
        assert counts_int.index.dtype != object  # integer dtype, not strings

    def test_sleepstatistics(self):
        """Test sleep statistics."""
        a = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 3, 3, 4, 4, 4, 4, 0, 0]
        validation = {
            "TIB": 10.0,
            "SPT": 8.0,
            "WASO": 0.0,
            "TST": 8.0,
            "N1": 1.5,
            "N2": 2.0,
            "N3": 2.5,
            "REM": 2.0,
            "NREM": 6.0,
            "SOL": 1.0,
            "Lat_N1": 1.0,
            "Lat_N2": 2.5,
            "Lat_N3": 4.0,
            "Lat_REM": 7.0,
            "%N1": 18.75,
            "%N2": 25.0,
            "%N3": 31.25,
            "%REM": 25.0,
            "%NREM": 75.0,
            "SE": 80.0,
            "SME": 100.0,
        }

        s = sleep_statistics(a, sf_hyp=1 / 30)
        # Compare with different sampling frequencies
        s2 = sleep_statistics(np.repeat(a, 30), sf_hyp=1)
        s3 = sleep_statistics(np.repeat(a, 30 * 100), sf_hyp=100)
        assert s == s2 == s3 == validation

        # Now with a second example
        a = [0, 0, 1, 1, 0, 0, 0, 0, 2, 2, 2, 0, 1, 1, 0, 0, 0]
        s = sleep_statistics(a, sf_hyp=1 / 60)
        # We cannot compare with NaN
        assert s["%REM"] == 0
