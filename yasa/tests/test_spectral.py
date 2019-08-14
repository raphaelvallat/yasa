"""
Test the functions in the yasa/main.py file.
"""
import mne
import unittest
import numpy as np
from itertools import product
from scipy.signal import welch
from yasa import hypno_str_to_int, hypno_upsample_to_data
from yasa.spectral import (bandpower, bandpower_from_psd, stft_power)

# Load 1D data
data = np.loadtxt('notebooks/data_N2_spindles_15sec_200Hz.txt')
sf = 200

# Load a full recording and its hypnogram
file_full = np.load('notebooks/data_full_6hrs_100Hz_Cz+Fz+Pz.npz')
data_full = file_full.get('data')
chan_full = file_full.get('chan')
sf_full = 100
hypno_full = np.load('notebooks/data_full_6hrs_100Hz_hypno.npz').get('hypno')

# Using MNE
data_mne = mne.io.read_raw_fif('notebooks/sub-02_mne_raw.fif', preload=True,
                               verbose=0)
data_mne.pick_types(eeg=True)
hypno_mne = np.loadtxt('notebooks/sub-02_hypno_30s.txt', dtype=str)
hypno_mne = hypno_str_to_int(hypno_mne)
hypno_mne = hypno_upsample_to_data(hypno=hypno_mne, sf_hypno=(1 / 30),
                                   data=data_mne)


class TestStringMethods(unittest.TestCase):

    def test_bandpower(self):
        """Test function bandpower
        """
        # BANDPOWER
        bandpower(data_mne)  # Raw MNE multi-channel
        bandpower(data, sf=sf)  # Single channel Numpy
        bandpower(data, sf=sf, ch_names='F4')  # Single channel Numpy labelled
        bandpower(data_full, sf=sf_full, ch_names=chan_full, hypno=hypno_full,
                  include=(2, 3))  # Multi channel numpy
        bandpower(data_full, sf=sf_full, hypno=hypno_full,
                  include=(3, 4))  # Multi channel numpy
        bandpower(data_mne, hypno=hypno_mne, include=2)  # Raw MNE with hypno

        # BANDPOWER_FROM_PSD
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
