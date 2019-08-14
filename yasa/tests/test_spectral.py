"""
Test the functions in the yasa/main.py file.
"""
import unittest
import numpy as np
from itertools import product
from scipy.signal import welch
from yasa.spectral import (stft_power, bandpower_from_psd)

# Load 1D data
data = np.loadtxt('notebooks/data_N2_spindles_15sec_200Hz.txt')
sf = 200

# Load a full recording and its hypnogram
file_full = np.load('notebooks/data_full_6hrs_100Hz_Cz+Fz+Pz.npz')
data_full = file_full.get('data')
chan_full = file_full.get('chan')
sf_full = 100
hypno_full = np.load('notebooks/data_full_6hrs_100Hz_hypno.npz').get('hypno')


class TestStringMethods(unittest.TestCase):

    def test_bandpower(self):
        """Test function bandpower
        """
        # 1-D EEG data
        win = int(2 * sf)
        freqs, psd = welch(data, sf, nperseg=win)
        bp = bandpower_from_psd(psd, freqs, ch_names=['F4'])
        assert bp.shape[0] == 1
        assert bp.at[0, 'Chan'] == 'F4'
        assert bp.at[0, 'FreqRes'] == 1 / (win / sf)
        assert np.isclose(bp.loc[0, ['Delta', 'Theta', 'Alpha',
                                     'Beta', 'Gamma']].sum(), 1, atol=1e-2)
        assert (bp.bands_ == "[(0.5, 4, 'Delta'), (4, 8, 'Theta'), "
                             "(8, 12, 'Alpha'), (12, 30, 'Beta'), "
                             "(30, 40, 'Gamma')]")

        # 2-D EEG data
        win = int(4 * sf)
        freqs, psd = welch(data_full, sf_full, nperseg=win)
        bp = bandpower_from_psd(psd, freqs, ch_names=chan_full)
        assert bp.shape[0] == len(chan_full)
        assert bp.at[0, 'Chan'].upper() == 'CZ'
        assert bp.at[1, 'FreqRes'] == 1 / (win / sf_full)
        # Unlabelled
        bp = bandpower_from_psd(psd, freqs, ch_names=None)
        assert np.array_equal(bp.loc[:, 'Chan'],
                              ['CHAN001', 'CHAN002', 'CHAN003'])

    def test_stft_power(self):
        """Test function stft_power
        """
        window = [2, 4]
        step = [0, .1, 1]
        band = [(0.5, 20), (1, 30), [5, 12], None]
        norm = [True, False]
        interp = [True, False]

        prod_args = product(window, step, band, interp, norm)

        for i, (w, s, b, i, n) in enumerate(prod_args):
            stft_power(data, sf, window=w, step=s, band=b, interp=i,
                       norm=n)

        f, t, _ = stft_power(data, sf, window=4, step=.1, band=(11, 16),
                             interp=True, norm=False)

        assert f[1] - f[0] == 0.25
        assert t.size == data.size
        assert max(f) == 16
        assert min(f) == 11
