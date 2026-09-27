"""Test the functions in the yasa/sleepstats.py file.

The public ``transition_matrix`` and ``sleep_statistics`` functions are deprecated (see
``test_hypno.py::test_deprecated_functions``). These tests check the values of the shared
implementation used by :py:meth:`yasa.Hypnogram.transition_matrix`.
"""

import unittest
import warnings

import numpy as np
import pandas as pd
import pytest

from yasa.hypno import Hypnogram, _transition_matrix, simulate_hypnogram
from yasa.sleepstats import sleep_statistics


class TestSleepStats(unittest.TestCase):
    def test_transition(self):
        """Test _transition_matrix on integer arrays."""
        a = [1, 1, 1, 0, 0, 2, 2, 0, 2, 0, 1, 1, 0, 0]
        counts, probs = _transition_matrix(a)
        c = np.array([[2, 1, 2], [2, 3, 0], [2, 0, 1]])
        p = np.array([[0.4, 0.2, 0.4], [0.4, 0.6, 0], [2 / 3, 0, 1 / 3]])
        assert pd.DataFrame(c).equals(counts)
        assert pd.DataFrame(p).equals(probs)
        assert (probs.sum(axis=1) == 1).all()
        # Second example, with only Wake, N2 and REM
        x = np.asarray([0, 2, 2, 0, 0, 2, 0, 4, 4, 0, 0])
        counts, probs = _transition_matrix(x)
        c = np.array([[2, 2, 1], [2, 1, 0], [1, 0, 1]])
        p = np.array([[0.4, 0.4, 0.2], [2 / 3, 1 / 3, 0], [0.5, 0, 0.5]])
        assert pd.DataFrame(c, index=[0, 2, 4], columns=[0, 2, 4]).equals(counts)
        assert pd.DataFrame(p, index=[0, 2, 4], columns=[0, 2, 4]).equals(probs)
        assert (probs.sum(axis=1) == 1).all()

    def test_transition_hypnogram(self):
        """Test that Hypnogram.transition_matrix returns string-labelled output."""
        # --- 5-stage ---
        hyp = simulate_hypnogram(tib=480, seed=42)
        counts_fn, probs_fn = hyp.transition_matrix()

        # Output labels are strings, not integers (works with both object and StringDtype)
        assert all(isinstance(label, str) for label in counts_fn.index)
        assert all(isinstance(label, str) for label in counts_fn.columns)

        # Probabilities are a right-stochastic matrix (each row sums to 1)
        np.testing.assert_allclose(probs_fn.sum(axis=1), 1.0)

        # --- 2-stage ---
        hyp2 = simulate_hypnogram(tib=120, n_stages=2, seed=1)
        counts2, probs2 = hyp2.transition_matrix()
        assert set(counts2.index).issubset({"WAKE", "SLEEP", "ART", "UNS"})
        np.testing.assert_allclose(probs2.sum(axis=1), 1.0)

        # --- 3-stage ---
        hyp3 = simulate_hypnogram(tib=240, n_stages=3, seed=2)
        counts3, probs3 = hyp3.transition_matrix()
        assert set(counts3.index).issubset({"WAKE", "NREM", "REM", "ART", "UNS"})
        np.testing.assert_allclose(probs3.sum(axis=1), 1.0)

        # --- Small known example: verify counts exactly ---
        hyp_known = Hypnogram(["W", "N1", "N2", "N3", "N2", "REM", "W"])
        counts_k, probs_k = hyp_known.transition_matrix()
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

    def test_transition_no_outgoing(self):
        """A stage only present in the last epoch has an undefined (NaN) probability row."""
        hyp = Hypnogram(["W", "N1", "N2", "N3", "REM"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # No RuntimeWarning for 0 / 0
            counts, probs = hyp.transition_matrix()
        assert counts.loc["REM"].sum() == 0
        assert probs.loc["REM"].isna().all()
        np.testing.assert_allclose(probs.drop(index="REM").sum(axis=1), 1.0)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_sleep_statistics_no_sleep():
    """The deprecated sleep_statistics returns NaN instead of crashing without sleep."""
    for hypno in [[0, 0, 0], [0, -1, -1, 0]]:
        stats = sleep_statistics(hypno, sf_hyp=1 / 30)
        assert stats["TST"] == 0 and stats["SPT"] == 0 and stats["SE"] == 0
        for key in ["WASO", "SOL", "SME", "%N2", "Lat_REM"]:
            assert np.isnan(stats[key])


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_sleep_statistics_values():
    """The deprecated sleep_statistics returns the documented values."""
    hypno = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 3, 3, 4, 4, 4, 4, 0, 0]
    expected = {
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
    assert sleep_statistics(hypno, sf_hyp=1 / 30) == expected
