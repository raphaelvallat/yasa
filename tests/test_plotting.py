"""Test the functions in the yasa/plotting.py file."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from yasa.hypno import simulate_hypnogram
from yasa.plotting import plot_hypnogram, topoplot

DATA_POS = pd.Series(
    [4, 8, 7, 1, 2, 3, 5], index=["F4", "F3", "C4", "C3", "P3", "P4", "Oz"], name="Values"
)
DATA_NEG = pd.Series(
    [-4, -8, -7, -1, -2, -3], index=["F4-M1", "F3-M1", "C4-M1", "C3-M1", "P3-M1", "P4-M1"]
)
DATA_MIXED = pd.Series(
    [-0.5, -0.7, -0.3, 0.1, 0.15, 0.3, 0.55], index=["F3", "Fz", "F4", "C3", "Cz", "C4", "Pz"]
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize(
    "data, kwargs",
    [
        (DATA_POS, dict(title="My first topoplot")),
        (DATA_POS, dict(vmin=0, vmax=8, cbar_title="Hello")),
        (DATA_POS, dict(n_colors=10, vmin=0, cmap="Blues")),
        (DATA_POS, dict(sensors="ko", res=64, names="values", show_names=True)),
        (DATA_POS, dict(mask_params=dict(marker="o"))),
        (DATA_NEG, dict()),
        (DATA_NEG, dict(vmin=0, vmax=8, cbar_title="Hello")),
        (DATA_NEG, dict(n_colors=10, vmin=0, cmap="Blues")),
        (DATA_NEG, dict(show_names=False)),
        (DATA_MIXED, dict(vmin=-1, vmax=1, n_colors=8)),
    ],
)
def test_topoplot(data, kwargs):
    """Test topoplot"""
    fig = topoplot(data, **kwargs)
    assert isinstance(fig, plt.Figure)


def test_topoplot_montage_no_warning():
    """The default montage does not trigger the MNE >= 1.13 deprecation of 'standard_1020'"""
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        topoplot(DATA_POS)


def test_topoplot_mask():
    """Test topoplot with a mask of significant channels"""
    mask = pd.Series([True, False, True, False, False, True, False], index=DATA_POS.index)
    topoplot(DATA_POS, mask=mask)
    topoplot(DATA_POS, mask=mask.astype(int))
    with pytest.raises(AssertionError, match="must be a Pandas Series"):
        topoplot(DATA_POS, mask=mask.to_numpy())
    with pytest.raises(AssertionError, match="must be True/False or 0/1"):
        topoplot(DATA_POS, mask=mask.astype(float))


def test_topoplot_existing_ax():
    """Test topoplot on an existing axis"""
    _, ax = plt.subplots()
    fig = topoplot(DATA_POS, ax=ax)
    assert fig is ax.get_figure()


def test_topoplot_input_not_modified():
    """Test that topoplot does not modify its input or the global rcParams"""
    data = DATA_NEG.copy()
    rc = {key: plt.rcParams[key] for key in ["font.size", "savefig.bbox", "savefig.transparent"]}
    topoplot(data, fontsize=20)
    pd.testing.assert_series_equal(data, DATA_NEG)
    assert {key: plt.rcParams[key] for key in rc} == rc
    with pytest.raises(AssertionError, match="must be a Pandas Series"):
        topoplot(data.to_numpy())


def test_plot_hypnogram():
    """Test Hypnogram.plot_hypnogram method."""
    # Default parameters
    hyp5 = simulate_hypnogram(n_stages=5)
    hyp2 = simulate_hypnogram(n_stages=2)
    ax = hyp5.plot_hypnogram()
    assert isinstance(ax, plt.Axes)
    # Aesthetic parameters
    _ = hyp5.plot_hypnogram(fill_color="gainsboro")
    _ = hyp5.plot_hypnogram(fill_color="gainsboro", highlight="REM")
    _ = hyp5.plot_hypnogram(highlight=None)
    _ = hyp2.plot_hypnogram(fill_color="gainsboro", highlight="SLEEP")
    _ = hyp2.plot_hypnogram(fill_color="gainsboro", highlight="SLEEP", lw=3)
    # Draw on an existing axis.
    ax = plt.subplot()
    assert hyp5.plot_hypnogram(ax=ax) is ax
    # With datetime axis
    hyp3 = simulate_hypnogram(n_stages=3, tib=800, start="2020-01-01 20:00:00")
    hyp3.plot_hypnogram()
    # With Artefacts and Unscored
    hyp3.hypno.iloc[-100:] = "UNS"
    hyp3.hypno.loc["2020-01-01 22:10:00":"2020-01-01 22:15:00"] = "ART"
    hyp3.hypno.loc["2020-01-01 23:30:00":"2020-01-02 01:00:00"] = "ART"
    hyp3.plot_hypnogram(fill_color="peachpuff")


def test_plot_hypnogram_restores_fontsize():
    """The larger font size of plot_hypnogram is restored, even when the plot fails."""
    fontsize = plt.rcParams["font.size"]
    hyp = simulate_hypnogram(n_stages=5)
    hyp.plot_hypnogram()
    assert plt.rcParams["font.size"] == fontsize
    with pytest.raises(AttributeError):
        hyp.plot_hypnogram(ax="not an axis")
    assert plt.rcParams["font.size"] == fontsize


@pytest.mark.parametrize("sf_hypno", [1 / 30, 1])
def test_plot_hypnogram_deprecated(sf_hypno):
    """Test the deprecated yasa.plot_hypnogram function, with an integer hypnogram."""
    hypno = np.array([0, 0, 1, 2, 2, 3, 3, 2, 4, 4, 0])
    with pytest.warns(FutureWarning, match="deprecated"):
        ax = plot_hypnogram(hypno, sf_hypno=sf_hypno)
    assert isinstance(ax, plt.Axes)
    # A Hypnogram is passed through unchanged
    with pytest.warns(FutureWarning, match="deprecated"):
        plot_hypnogram(simulate_hypnogram(n_stages=5))
