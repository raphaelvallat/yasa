"""Test the functions in the yasa/spectral.py file."""
import mne
import unittest
import numpy as np
from itertools import product
from scipy.signal import welch
import matplotlib.pyplot as plt

from yasa.plotting import plot_spectrogram
from yasa.hypno import hypno_str_to_int, hypno_upsample_to_data
from yasa.spectral import (bandpower, bandpower_from_psd,
                           bandpower_from_psd_ndarray, irasa, stft_power)

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

# Eyes-open 6 minutes resting-state, 2 channels, 200 Hz
raw_eo = mne.io.read_raw_fif('notebooks/data_resting_EO_200Hz_raw.fif',
                             verbose=0)
data_eo = raw_eo.get_data() * 1e6
sf_eo = raw_eo.info['sfreq']
chan_eo = raw_eo.ch_names


class TestSpectral(unittest.TestCase):

    def test_bandpower(self):
        """Test function bandpower
        """
        # BANDPOWER
        bandpower(data_mne)  # Raw MNE multi-channel
        bandpower(data, sf=sf, bandpass=True)  # Single channel Numpy
        bandpower(data, sf=sf, ch_names='F4')  # Single channel Numpy labelled
        bandpower(data_full, sf=sf_full, ch_names=chan_full, hypno=hypno_full,
                  include=(2, 3))  # Multi channel numpy
        bandpower(data_full, sf=sf_full, hypno=hypno_full,
                  include=(3, 4, 5), bandpass=True)  # Multi channel numpy
        bandpower(data_mne, hypno=hypno_mne, include=2)  # Raw MNE with hypno

        # BANDPOWER_FROM_PSD
        # 1-D EEG data
        win = int(2 * sf)
        freqs, psd = welch(data, sf, nperseg=win)
        bp_abs_true = bandpower_from_psd(psd, freqs, relative=False)
        bp = bandpower_from_psd(psd, freqs, ch_names=['F4'])
        bands = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta', 'Gamma']
        assert bp.shape[0] == 1
        assert bp.at[0, 'Chan'] == 'F4'
        assert bp.at[0, 'FreqRes'] == 1 / (win / sf)
        assert np.isclose(bp.loc[0, bands].sum(), 1, atol=1e-2)
        assert (bp.bands_ == "[(0.5, 4, 'Delta'), (4, 8, 'Theta'), "
                             "(8, 12, 'Alpha'), (12, 16, 'Sigma'), "
                             "(16, 30, 'Beta'), (30, 40, 'Gamma')]")

        # Check that we can recover the physical power using TotalAbsPow
        bands = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta', 'Gamma']
        bp_abs = (bp[bands] * bp['TotalAbsPow'].values[..., None])
        np.testing.assert_array_almost_equal(bp_abs[bands].values,
                                             bp_abs_true[bands].values)

        # 2-D EEG data
        win = int(4 * sf)
        freqs, psd = welch(data_full, sf_full, nperseg=win)
        bp = bandpower_from_psd(psd, freqs, ch_names=chan_full)
        assert bp.shape[0] == len(chan_full)
        assert bp.at[0, 'Chan'].upper() == 'CZ'
        assert bp.at[1, 'FreqRes'] == 1 / (win / sf_full)
        # Unlabelled
        bp = bandpower_from_psd(psd, freqs, ch_names=None, relative=False)
        assert np.array_equal(bp.loc[:, 'Chan'],
                              ['CHAN000', 'CHAN001', 'CHAN002'])

        # Bandpower from PSD with NDarray
        n_chan = 4
        n_epochs = 400
        n_times = 3000
        data_1d = np.random.rand(n_times)
        data_2d = np.random.rand(n_chan, n_times)
        data_3d = np.random.rand(n_chan, n_epochs, n_times)
        freqs, psd_1d = welch(data_1d, sf, nperseg=int(4 * sf), axis=-1)
        freqs, psd_2d = welch(data_2d, sf, nperseg=int(4 * sf), axis=-1)
        freqs, psd_3d = welch(data_3d, sf, nperseg=int(4 * sf), axis=-1)
        bandpower_from_psd_ndarray(psd_1d, freqs, relative=True)
        bandpower_from_psd_ndarray(psd_2d, freqs, relative=False)
        assert (bandpower_from_psd_ndarray(psd_3d, freqs,
                                           bands=[(0.5, 4, 'Delta')],
                                           relative=True) == 1).all()

        # With negative values: we should get a logger warning
        freqs = np.arange(0, 50.5, 0.5)
        psd = np.random.normal(size=(6, freqs.size))
        with self.assertLogs('yasa', level='WARNING'):
            bandpower_from_psd(psd, freqs)
        with self.assertLogs('yasa', level='WARNING'):
            bandpower_from_psd_ndarray(psd, freqs)

    def test_irasa(self):
        """Test function IRASA.
        """
        # 1D Numpy
        freqs, psd_aperiodic, psd_osc, fit_params = irasa(data=data, sf=sf)
        assert np.isin(freqs, np.arange(1, 30.25, 0.25), True).all()
        assert np.median(psd_aperiodic) > np.median(psd_osc)

        # 2D Numpy
        irasa(data=data_eo, sf=sf_eo, ch_names=chan_eo)
        irasa(data=data_eo, sf=sf_eo, ch_names=None)

        # 2D MNE
        assert len(irasa(data_mne, return_fit=False)) == 3
        assert len(irasa(data_mne, band=(2, 24), win_sec=2)) == 4

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

    def test_plot_spectrogram(self):
        """Test function plot_spectrogram
        """
        plot_spectrogram(data_full[0, :], sf_full, fmin=0.5, fmax=30)
        plot_spectrogram(data_full[0, :], sf_full, hypno_full, trimperc=5)
        hypno_full_art = np.copy(hypno_full)
        hypno_full_art[hypno_full_art == 3.] = -1
        # Replace N3 by Artefact
        plot_spectrogram(data_full[0, :], sf_full, hypno_full_art, trimperc=5)
        # Now replace REM by Unscored
        hypno_full_art[hypno_full_art == 4.] = -2
        plot_spectrogram(data_full[0, :], sf_full, hypno_full_art)
        plt.close('all')
