"""Test the functions in the yasa/plotting.py file."""
import unittest
import pandas as pd
import matplotlib.pyplot as plt
from yasa.plotting import topoplot


class TestPlotting(unittest.TestCase):

    def test_topoplot(self):
        """Test topoplot"""
        data = pd.Series(
            [4, 8, 7, 1, 2, 3, 5],
            index=['F4', 'F3', 'C4', 'C3', 'P3', 'P4', 'Oz'],
            name='Values')
        _ = topoplot(data, title='My first topoplot')
        _ = topoplot(data, vmin=0, vmax=8, cbar_title='Hello')
        _ = topoplot(data, n_colors=10, vmin=0, cmap="Blues")
        _ = topoplot(data, sensors='ko', res=64, names='values',
                     show_names=True)

        data = pd.Series(
            [-4, -8, -7, -1, -2, -3],
            index=['F4-M1', 'F3-M1', 'C4-M1', 'C3-M1', 'P3-M1', 'P4-M1'])
        _ = topoplot(data)
        _ = topoplot(data, vmin=0, vmax=8, cbar_title='Hello')
        _ = topoplot(data, n_colors=10, vmin=0, cmap="Blues")
        _ = topoplot(data, show_names=False)

        data = pd.Series([-0.5, -0.7, -0.3, 0.1, 0.15, 0.3, 0.55],
                         index=['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'Pz'])
        _ = topoplot(data, vmin=-1, vmax=1, n_colors=8)

        plt.close('all')
