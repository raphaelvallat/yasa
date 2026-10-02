"""Test the module-level functions of yasa/hypno.py (not the methods of the Hypnogram class)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import yasa
from yasa.hypno import Hypnogram, simulate_hypnogram
from yasa.hypno import _hypno_find_periods as hfp

HYPNO = np.array([0, 0, 0, 1, 2, 2, 3, 3, 4])
HYPNO_TXT = np.array(["W", "W", "W", "N1", "N2", "N2", "N3", "N3", "R"])
STAGES = ["WAKE", "N1", "N2", "N3", "REM"]

# The dtypes of the output of _hypno_find_periods depend on the input
FRAME_KWARGS = dict(
    check_dtype=False, check_index_type=False, check_column_type=False, check_frame_type=False
)

###############################################################################
# _hypno_find_periods
###############################################################################

# Binary vector: 11 x 0, 3 x 1, 2 x 0, 9 x 1, 2 x 0
X_BINARY = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
X_MULTI = [0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 0, 1]


@pytest.mark.parametrize("sf_hypno", [1 / 60, 1])
def test_find_periods_binary_no_threshold(sf_hypno):
    """Without a threshold, every run is returned regardless of the sampling frequency."""
    expected = pd.DataFrame(
        {"values": [0, 1, 0, 1, 0], "start": [0, 11, 14, 16, 25], "length": [11, 3, 2, 9, 2]}
    )
    assert_frame_equal(hfp(X_BINARY, sf_hypno=sf_hypno, threshold="0min"), expected, **FRAME_KWARGS)


def test_find_periods_binary_threshold():
    """Only the runs that are at least as long as the threshold are kept."""
    expected = pd.DataFrame({"values": [0, 1], "start": [0, 16], "length": [11, 9]})
    assert_frame_equal(hfp(X_BINARY, sf_hypno=1 / 60, threshold="5min"), expected, **FRAME_KWARGS)
    # At 1 Hz, no run is longer than 5 minutes
    assert hfp(X_BINARY, sf_hypno=1, threshold="5min").size == 0


def test_find_periods_binary_equal_length():
    """With equal_length=True, runs are split into consecutive periods of threshold length."""
    expected = pd.DataFrame(
        {
            "values": [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0],
            "start": [0, 2, 4, 6, 8, 11, 14, 16, 18, 20, 22, 25],
            "length": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        }
    )
    assert_frame_equal(
        hfp(X_BINARY, sf_hypno=1 / 60, threshold="2min", equal_length=True),
        expected,
        **FRAME_KWARGS,
    )


@pytest.mark.parametrize("as_str", [False, True], ids=["int", "str"])
def test_find_periods_multiclass(as_str):
    """Multi-class vectors, with an integer or a string dtype."""
    expected = pd.DataFrame(
        {
            "values": [0, 1, 2, 0, 1, 0, 1],
            "start": [0, 4, 5, 11, 14, 15, 16],
            "length": [4, 1, 6, 3, 1, 1, 1],
        }
    )
    x = X_MULTI
    if as_str:
        x = np.array(X_MULTI).astype(str)
        expected["values"] = expected["values"].astype(str)
    assert_frame_equal(hfp(x, sf_hypno=1 / 60, threshold="0min"), expected, **FRAME_KWARGS)


###############################################################################
# simulate_hypnogram
###############################################################################


def test_simulate_seed():
    """A given seed always returns the same hypnogram."""
    hyp = simulate_hypnogram(tib=4, seed=1)
    assert hyp.n_epochs == 8
    np.testing.assert_array_equal(hyp.as_int(), [0, 1, 1, 2, 2, 2, 2, 2])


def test_simulate_n_stages():
    assert simulate_hypnogram(tib=1000, n_stages=2).hypno.nunique() == 2
    np.testing.assert_array_equal(
        simulate_hypnogram(tib=4, seed=1, n_stages=3).as_int(), [0, 2, 2, 2, 2, 2, 2, 2]
    )


def test_simulate_freq():
    """The simulation is done at 30 s and then upsampled to the requested frequency."""
    hyp = simulate_hypnogram(tib=4, seed=1)
    np.testing.assert_array_equal(hyp.hypno, simulate_hypnogram(tib=4, freq="0.5min", seed=1).hypno)
    np.testing.assert_array_equal(
        hyp.upsample("5s").hypno, simulate_hypnogram(tib=4, freq="5s", seed=1).hypno
    )
    assert simulate_hypnogram(tib=4, freq="15s").n_epochs == 16
    assert simulate_hypnogram(tib=4, freq="15s").duration == 4
    assert simulate_hypnogram(tib=4, freq="30s").duration == 4


@pytest.mark.parametrize("freq", ["60s", "20s"])  # 30 s is not a multiple of 20 s
def test_simulate_invalid_freq_raises(freq):
    with pytest.raises(AssertionError):
        simulate_hypnogram(freq=freq)


def test_simulate_custom_probas():
    """User-defined transition and initial probabilities, which are not modified in place."""
    trans_probas = pd.DataFrame(data=np.full((5, 5), 0.2), index=STAGES, columns=STAGES)
    simulate_hypnogram(tib=2, trans_probas=trans_probas)
    assert trans_probas.attrs == {}  # The user's dataframe is not modified
    simulate_hypnogram(tib=2, init_probas=trans_probas.loc["WAKE"])
    simulate_hypnogram(tib=2, trans_probas=trans_probas, init_probas=trans_probas.loc["WAKE"])


def test_simulate_identity_probas():
    """With no transition between stages, the hypnogram stays in the initial stage (WAKE)."""
    trans_probas = pd.DataFrame(np.eye(5, 5), index=STAGES, columns=STAGES)
    assert not simulate_hypnogram(trans_probas=trans_probas).as_int().any()
    # trans_probas has more stages than allowed by n_stages
    with pytest.raises(AssertionError):
        simulate_hypnogram(n_stages=4, trans_probas=trans_probas)


def test_simulate_subset_probas():
    """trans_probas can include only a subset of the stages allowed by n_stages."""
    stages = STAGES[:-1]  # No REM
    trans_probas = pd.DataFrame(np.full((4, 4), 0.25), index=stages, columns=stages)
    simulate_hypnogram(trans_probas=trans_probas)


def test_simulate_kwargs():
    """**kwargs are passed through to yasa.Hypnogram."""
    shyp = simulate_hypnogram(tib=5, scorer="RV", start="2022-12-15 22:30:00")
    assert shyp.scorer == shyp.hypno.name == "RV"
    assert shyp.start == pd.Timestamp("2022-12-15 22:30:00")


###############################################################################
# Compumedics Profusion (load_profusion_hypno and Hypnogram.from_profusion)
###############################################################################

# Native Profusion stages: 0=Wake, 1=N1, 2=N2, 3=N3, 4=S4 (N3), 5=REM, 9=Active (Wake)
PROFUSION_STAGES = [0, 1, 2, 3, 4, 5, 9, 2]
PROFUSION_TO_YASA = [0, 1, 2, 3, 3, 4, 0, 2]


@pytest.fixture
def profusion_xml(tmp_path):
    """Write a minimal Compumedics Profusion XML file and return its path.

    The parser reads the epoch length from the first child of the root and the sleep stages
    from the fifth child, as in the NSRR files.
    """
    stages = "".join(f"<SleepStage>{s}</SleepStage>" for s in PROFUSION_STAGES)
    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        "<CMPStudyConfig>"
        "<EpochLength>20</EpochLength>"
        "<StepChannels/>"
        "<ScoredEvents/>"
        "<Montage/>"
        f"<SleepStages>{stages}</SleepStages>"
        "</CMPStudyConfig>"
    )
    fname = tmp_path / "hypnogram.xml"
    fname.write_text(xml)
    return fname


@pytest.mark.parametrize(
    "replace, expected", [(True, PROFUSION_TO_YASA), (False, PROFUSION_STAGES)]
)
def test_load_profusion_hypno(profusion_xml, replace, expected):
    with pytest.warns(FutureWarning, match="Hypnogram.from_profusion"):
        hypno, sf_hyp = yasa.load_profusion_hypno(profusion_xml, replace=replace)
    np.testing.assert_array_equal(hypno, expected)
    assert sf_hyp == 1 / 20


def test_from_profusion(profusion_xml):
    hyp = Hypnogram.from_profusion(profusion_xml)
    assert hyp.freq == "20s"
    assert hyp.n_stages == 5
    assert hyp.start is None
    assert hyp.scorer is None
    assert hyp.hypno.tolist() == ["WAKE", "N1", "N2", "N3", "N3", "REM", "WAKE", "N2"]
    np.testing.assert_array_equal(hyp.as_int(), PROFUSION_TO_YASA)


def test_from_profusion_kwargs(profusion_xml):
    hyp = Hypnogram.from_profusion(
        str(profusion_xml), start="2022-12-15 22:30:00", tz="Europe/Paris", scorer="Expert"
    )
    assert hyp.start == pd.Timestamp("2022-12-15 22:30:00", tz="Europe/Paris")
    assert hyp.end == pd.Timestamp("2022-12-15 22:32:40", tz="Europe/Paris")  # 8 x 20 s
    assert hyp.scorer == hyp.hypno.name == "Expert"


###############################################################################
# Deprecated functions
###############################################################################


@pytest.mark.parametrize(
    "func, args, expected",
    [
        ("hypno_str_to_int", (HYPNO_TXT,), HYPNO),
        ("hypno_int_to_str", (HYPNO,), ["W", "W", "W", "N1", "N2", "N2", "N3", "N3", "R"]),
    ],
)
def test_deprecated_str_int_conversion(func, args, expected):
    with pytest.warns(FutureWarning, match="deprecated"):
        out = getattr(yasa, func)(*args)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize(
    "name, args",
    [
        ("hypno_str_to_int", (HYPNO_TXT,)),
        ("hypno_int_to_str", (HYPNO,)),
        ("hypno_find_periods", (HYPNO, 1 / 30, "1min")),
        ("sleep_statistics", (HYPNO, 1 / 30)),
        ("transition_matrix", (HYPNO,)),
        ("transition_matrix", (Hypnogram.from_integers(HYPNO),)),
        ("plot_hypnogram", (HYPNO,)),
    ],
)
def test_deprecated_functions(name, args):
    """Standalone hypnogram functions emit a FutureWarning pointing to the Hypnogram API.

    The deprecated upsampling functions are tested in test_hypno_resample.py.
    """
    with pytest.warns(FutureWarning, match="deprecated and will be removed in v0.9"):
        getattr(yasa, name)(*args)
    plt.close("all")
