"""Test the functions in the yasa/sleepstats.py file.

The public ``transition_matrix`` and ``sleep_statistics`` functions are deprecated (see
``test_hypno.py::test_deprecated_functions``). These tests check the values of the shared
implementation used by :py:meth:`yasa.Hypnogram.transition_matrix`.
"""

import unittest

import numpy as np
import pandas as pd

from yasa.hypno import Hypnogram, simulate_hypnogram
from yasa.sleepstats import _transition_matrix


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
