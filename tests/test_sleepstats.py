"""Test the functions in the yasa/sleepstats.py file.

The public ``transition_matrix`` and ``sleep_statistics`` functions are deprecated (see
``test_hypno_utils.py::test_deprecated_functions``). These tests check the values of the shared
implementation used by :py:meth:`yasa.Hypnogram.transition_matrix`.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from yasa.hypno import Hypnogram, _transition_matrix, simulate_hypnogram
from yasa.sleepstats import sleep_statistics


@pytest.mark.parametrize(
    "hypno, labels, counts, probs",
    [
        (
            [1, 1, 1, 0, 0, 2, 2, 0, 2, 0, 1, 1, 0, 0],
            [0, 1, 2],
            [[2, 1, 2], [2, 3, 0], [2, 0, 1]],
            [[0.4, 0.2, 0.4], [0.4, 0.6, 0], [2 / 3, 0, 1 / 3]],
        ),
        # Only Wake, N2 and REM
        (
            np.asarray([0, 2, 2, 0, 0, 2, 0, 4, 4, 0, 0]),
            [0, 2, 4],
            [[2, 2, 1], [2, 1, 0], [1, 0, 1]],
            [[0.4, 0.4, 0.2], [2 / 3, 1 / 3, 0], [0.5, 0, 0.5]],
        ),
    ],
)
def test_transition_matrix_int(hypno, labels, counts, probs):
    """Test _transition_matrix on integer arrays."""
    out_counts, out_probs = _transition_matrix(hypno)
    assert pd.DataFrame(np.array(counts), index=labels, columns=labels).equals(out_counts)
    assert pd.DataFrame(np.array(probs), index=labels, columns=labels).equals(out_probs)
    assert (out_probs.sum(axis=1) == 1).all()


def test_transition_hypnogram_5stages():
    """Hypnogram.transition_matrix returns string-labelled output."""
    counts, probs = simulate_hypnogram(tib=480, seed=42).transition_matrix()
    # Output labels are strings, not integers (works with both object and StringDtype)
    assert all(isinstance(label, str) for label in counts.index)
    assert all(isinstance(label, str) for label in counts.columns)
    # Probabilities are a right-stochastic matrix (each row sums to 1)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "n_stages, tib, seed, labels",
    [
        (2, 120, 1, {"WAKE", "SLEEP", "ART", "UNS"}),
        (3, 240, 2, {"WAKE", "NREM", "REM", "ART", "UNS"}),
    ],
)
def test_transition_hypnogram_n_stages(n_stages, tib, seed, labels):
    hyp = simulate_hypnogram(tib=tib, n_stages=n_stages, seed=seed)
    counts, probs = hyp.transition_matrix()
    assert set(counts.index).issubset(labels)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)


def test_transition_hypnogram_known_counts():
    """Small known example: verify counts exactly."""
    counts, probs = Hypnogram(["W", "N1", "N2", "N3", "N2", "REM", "W"]).transition_matrix()
    # Transitions: W→N1, N1→N2, N2→N3, N3→N2, N2→REM, REM→W
    assert counts.loc["WAKE", "N1"] == 1
    assert counts.loc["N1", "N2"] == 1
    assert counts.loc["N2", "N3"] == 1
    assert counts.loc["N2", "REM"] == 1
    assert counts.loc["N3", "N2"] == 1
    assert counts.loc["REM", "WAKE"] == 1
    assert counts.loc["WAKE", "WAKE"] == 0
    # Row sums equal total transitions out of each stage
    assert counts.loc["N2"].sum() == 2  # N2→N3 and N2→REM
    np.testing.assert_allclose(probs.loc["N2", "N3"], 0.5)
    np.testing.assert_allclose(probs.loc["N2", "REM"], 0.5)


def test_transition_no_outgoing():
    """A stage only present in the last epoch has an undefined (NaN) probability row."""
    hyp = Hypnogram(["W", "N1", "N2", "N3", "REM"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # No RuntimeWarning for 0 / 0
        counts, probs = hyp.transition_matrix()
    assert counts.loc["REM"].sum() == 0
    assert probs.loc["REM"].isna().all()
    np.testing.assert_allclose(probs.drop(index="REM").sum(axis=1), 1.0)


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("hypno", [[0, 0, 0], [0, -1, -1, 0]])
def test_sleep_statistics_no_sleep(hypno):
    """The deprecated sleep_statistics returns NaN instead of crashing without sleep."""
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


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_sleep_statistics_hypnogram():
    """The deprecated sleep_statistics dispatches a Hypnogram to its method."""
    hyp = Hypnogram.from_integers([0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 3, 3, 4, 4, 4, 4, 0, 0])
    assert sleep_statistics(hyp) == hyp.sleep_statistics()
    with pytest.raises(AssertionError, match="sf_hyp is required"):
        sleep_statistics(hyp.as_int().to_numpy())
