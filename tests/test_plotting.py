"""Test the functions in the yasa/plotting.py file."""

import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from yasa.fetchers import fetch_sample
from yasa.hypno import simulate_hypnogram
from yasa.plotting import plot_hypnogram, topoplot


class TestPlotting(unittest.TestCase):
    def test_topoplot(self):
        """Test topoplot"""
        data = pd.Series(
            [4, 8, 7, 1, 2, 3, 5], index=["F4", "F3", "C4", "C3", "P3", "P4", "Oz"], name="Values"
        )
        _ = topoplot(data, title="My first topoplot")
        _ = topoplot(data, vmin=0, vmax=8, cbar_title="Hello")
        _ = topoplot(data, n_colors=10, vmin=0, cmap="Blues")
        _ = topoplot(data, sensors="ko", res=64, names="values", show_names=True)

        data = pd.Series(
            [-4, -8, -7, -1, -2, -3], index=["F4-M1", "F3-M1", "C4-M1", "C3-M1", "P3-M1", "P4-M1"]
        )
        _ = topoplot(data)
        _ = topoplot(data, vmin=0, vmax=8, cbar_title="Hello")
        _ = topoplot(data, n_colors=10, vmin=0, cmap="Blues")
        _ = topoplot(data, show_names=False)

        data = pd.Series(
            [-0.5, -0.7, -0.3, 0.1, 0.15, 0.3, 0.55],
            index=["F3", "Fz", "F4", "C3", "Cz", "C4", "Pz"],
        )
        _ = topoplot(data, vmin=-1, vmax=1, n_colors=8)

        plt.close("all")

    def test_plot_hypnogram(self):
        """Test plot_hypnogram function."""
        # Old format: array of integer
        hypno_fp = fetch_sample("full_6hrs_100Hz_hypno_30s.txt")
        hypno = np.loadtxt(hypno_fp)
        _ = plot_hypnogram(hypno)
        # Error because of input is not a yasa.Hypnogram
        # with pytest.raises(AssertionError):
        #     _ = plot_hypnogram(np.repeat([0, 1, 2, 3, 4, -2, -1, -3], 120))
        # Default parameters
        hyp5 = simulate_hypnogram(n_stages=5)
        hyp2 = simulate_hypnogram(n_stages=2)
        ax = hyp5.plot_hypnogram()
        assert isinstance(ax, plt.Axes)
        # Aesthetic parameters
        _ = hyp5.plot_hypnogram(fill_color="gainsboro")
        _ = hyp5.plot_hypnogram(fill_color="gainsboro", highlight="REM")
        _ = hyp2.plot_hypnogram(fill_color="gainsboro", highlight="SLEEP")
        _ = hyp2.plot_hypnogram(fill_color="gainsboro", highlight="SLEEP", lw=3)
        # Draw on an existing axis.
        ax = plt.subplot()
        _ = hyp5.plot_hypnogram(ax=ax)
        # With datetime axis
        hyp3 = simulate_hypnogram(n_stages=3, tib=800, start="2020-01-01 20:00:00")
        hyp3.plot_hypnogram()
        # With Artefacts and Unscored
        hyp3.hypno.iloc[-100:] = "UNS"
        hyp3.hypno.loc["2020-01-01 22:10:00":"2020-01-01 22:15:00"] = "ART"
        hyp3.hypno.loc["2020-01-01 23:30:00":"2020-01-02 01:00:00"] = "ART"
        hyp3.plot_hypnogram(fill_color="peachpuff")
        plt.close("all")
