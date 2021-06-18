"""Test the functions in the yasa/sleepstats.py file."""
import unittest
import numpy as np
import pandas as pd
from yasa.sleepstats import transition_matrix, sleep_statistics

hypno = np.array([0, 0, 0, 1, 2, 2, 3, 3, 2, 2, 2, 0, 0, 0, 2, 2, 4, 4, 0, 0])


class TestSleepStats(unittest.TestCase):

    def test_transition(self):
        """Test transition_matrix
        """
        a = [1, 1, 1, 0, 0, 2, 2, 0, 2, 0, 1, 1, 0, 0]
        counts, probs = transition_matrix(a)
        c = np.array([[2, 1, 2], [2, 3, 0], [2, 0, 1]])
        p = np.array([[0.4, 0.2, 0.4], [0.4, 0.6, 0], [2 / 3, 0, 1 / 3]])
        assert pd.DataFrame(c).equals(counts)
        assert pd.DataFrame(p).equals(probs)
        assert (probs.sum(1) == 1).all()
        # Second example, with only Wake, N2 and REM
        x = np.asarray([0, 2, 2, 0, 0, 2, 0, 4, 4, 0, 0])
        counts, probs = transition_matrix(x)
        c = np.array([[2, 2, 1], [2, 1, 0], [1, 0, 1]])
        p = np.array([[0.4, 0.4, 0.2], [2 / 3, 1 / 3, 0], [0.5, 0, 0.5]])
        assert pd.DataFrame(c, index=[0, 2, 4],
                            columns=[0, 2, 4]).equals(counts)
        assert pd.DataFrame(p, index=[0, 2, 4],
                            columns=[0, 2, 4]).equals(probs)
        assert (probs.sum(1) == 1).all()

    def test_sleepstatistics(self):
        """Test sleep statistics.
        """
        a = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 3, 3, 4, 4, 4, 4, 0, 0]
        validation = {'TIB': 10.0, 'SPT': 8.0, 'WASO': 0.0, 'TST': 8.0,
                      'N1': 1.5, 'N2': 2.0, 'N3': 2.5, 'REM': 2.0,
                      'NREM': 6.0, 'SOL': 1.0, 'Lat_N1': 1.0, 'Lat_N2': 2.5,
                      'Lat_N3': 4.0, 'Lat_REM': 7.0,
                      '%N1': 18.75, '%N2': 25.0, '%N3': 31.25, '%REM': 25.0,
                      '%NREM': 75.0, 'SE': 80.0, 'SME': 100.0}

        s = sleep_statistics(a, sf_hyp=1 / 30)
        # Compare with different sampling frequencies
        s2 = sleep_statistics(np.repeat(a, 30), sf_hyp=1)
        s3 = sleep_statistics(np.repeat(a, 30 * 100), sf_hyp=100)
        assert s == s2 == s3 == validation

        # Now with a second example
        a = [0, 0, 1, 1, 0, 0, 0, 0, 2, 2, 2, 0, 1, 1, 0, 0, 0]
        s = sleep_statistics(a, sf_hyp=1 / 60)
        # We cannot compare with NaN
        assert s['%REM'] == 0
