"""Test the functions in the yasa/spectral.py file."""

import unittest
from itertools import product

import matplotlib.pyplot as plt
import mne
import numpy as np
import pytest
from scipy.signal import welch

from yasa.fetchers import fetch_sample
from yasa.hypno import Hypnogram
from yasa.plotting import plot_spectrogram
from yasa.spectral import (
    bandpower,
    bandpower_from_psd,
    bandpower_from_psd_ndarray,
    irasa,
    stft_power,
)

# Load 1D data
data_fp = fetch_sample("N2_spindles_15sec_200Hz.txt")
data = np.loadtxt(data_fp)
sf = 200

# Load one hour of a full recording and its hypnogram. The third hour of the recording
# includes all the sleep stages (W, N1, N2, N3 and REM).
file_full_fp = fetch_sample("full_6hrs_100Hz_Cz+Fz+Pz.npz")
file_full = np.load(file_full_fp)
data_full = file_full.get("data")[:, 720_000:1_080_000]
chan_full = file_full.get("chan")
sf_full = 100
hypno_full_fp = fetch_sample("full_6hrs_100Hz_hypno.npz")
hypno_full = np.load(hypno_full_fp).get("hypno")[720_000:1_080_000]

# Using MNE
data_mne_fp = fetch_sample("sub-02_mne_raw.fif")
data_mne = mne.io.read_raw_fif(data_mne_fp, preload=True, verbose=0)
data_mne.pick("eeg")
hypno_mne_fp = fetch_sample("sub-02_hypno_30s.txt")
hypno_mne = Hypnogram(np.loadtxt(hypno_mne_fp, dtype=str), freq="30s").upsample_to_data(data_mne)

# Hypnogram objects for testing Hypnogram-based hypno support
hypno_full_30s = hypno_full[:: int(sf_full * 30)]  # 1 value per 30s epoch
hyp_full = Hypnogram.from_integers(hypno_full_30s, freq="30s")

# Eyes-open 6 minutes resting-state, 2 channels, 200 Hz
raw_eo_fp = fetch_sample("resting_EO_200Hz_raw.fif")
raw_eo = mne.io.read_raw_fif(raw_eo_fp, verbose=0)
data_eo = raw_eo.get_data(units=dict(eeg="uV", emg="uV", eog="uV", ecg="uV"))
sf_eo = raw_eo.info["sfreq"]
chan_eo = raw_eo.ch_names

bands = ["Delta", "Theta", "Alpha", "Sigma", "Beta", "Gamma"]


class TestSpectral(unittest.TestCase):
    def test_bandpower(self):
        """Test function bandpower"""
        # BANDPOWER
        bp = bandpower(data_mne)  # Raw MNE multi-channel
        assert bp.index.tolist() == data_mne.ch_names
        np.testing.assert_allclose(bp[bands].sum(axis=1), 1, atol=1e-2)
        bp = bandpower(data, sf=sf, bandpass=True)  # Single channel Numpy
        assert bp.index.tolist() == ["CHAN000"]
        bp = bandpower(data, sf=sf, ch_names="F4")  # Single channel Numpy labelled
        assert bp.index.tolist() == ["F4"]
        bp = bandpower(
            data_full, sf=sf_full, hypno=hypno_full, include=(3, 4, 5), bandpass=True
        )  # Multi channel numpy. There is no stage 5 in the hypnogram.
        assert bp.index.get_level_values("Stage").unique().tolist() == [3, 4]
        bp = bandpower(data_mne, hypno=hypno_mne, include=2)  # Raw MNE with hypno
        assert bp.shape[0] == len(data_mne.ch_names)
        bp_int = bandpower(data_full, sf=sf_full, hypno=hypno_full, include=2)
        # Stages shorter than the Welch window are skipped with a warning
        hypno_short = np.full(data_full.shape[1], 2)
        hypno_short[:200] = 1  # 2 seconds of N1, shorter than the 4-seconds window
        with self.assertLogs("yasa", level="WARNING"):
            bp = bandpower(data_full, sf=sf_full, hypno=hypno_short, include=(1, 2))
        assert bp.index.get_level_values("Stage").unique().tolist() == [2]
        with pytest.raises(ValueError, match="shorter than the Welch window"):
            bandpower(data_full, sf=sf_full, hypno=hypno_short, include=1)
        # String hypnogram arrays (with string include) are labelled with the string stages
        hypno_str = np.where(hypno_full == 2, "N2", "Other")
        bp_str = bandpower(data_full, sf=sf_full, hypno=hypno_str, include="N2")
        assert bp_str.index.get_level_values("Stage").unique().tolist() == ["N2"]
        np.testing.assert_allclose(bp_str.to_numpy(float), bp_int.to_numpy(float))

        # Hypnogram instance: the output is labelled with the stages as given in include
        bp_hyp_int = bandpower(
            data_full, sf=sf_full, ch_names=chan_full, hypno=hyp_full, include=(2, 3)
        )
        bp_hyp_str = bandpower(
            data_full, sf=sf_full, ch_names=chan_full, hypno=hyp_full, include=["N2", "N3"]
        )
        assert bp_hyp_int.index.names == ["Stage", "Chan"]
        assert bp_hyp_int.index.get_level_values("Stage").unique().tolist() == [2, 3]
        assert bp_hyp_str.index.get_level_values("Stage").unique().tolist() == ["N2", "N3"]
        np.testing.assert_array_equal(bp_hyp_int.to_numpy(), bp_hyp_str.to_numpy())

        # BANDPOWER_FROM_PSD
        # 1-D EEG data
        win = int(2 * sf)
        freqs, psd = welch(data, sf, nperseg=win)
        bp_abs_true = bandpower_from_psd(psd, freqs, relative=False)
        bp = bandpower_from_psd(psd, freqs, ch_names=["F4"])
        assert bp.shape[0] == 1
        assert bp.columns.tolist() == ["Chan", *bands, "TotalAbsPow", "FreqRes", "Relative"]
        assert bp.at[0, "Chan"] == "F4"
        assert bp.at[0, "FreqRes"] == 1 / (win / sf)
        assert np.isclose(bp.loc[0, bands].sum(), 1, atol=1e-2)
        assert (
            bp.bands_ == "[(0.5, 4, 'Delta'), (4, 8, 'Theta'), "
            "(8, 12, 'Alpha'), (12, 16, 'Sigma'), "
            "(16, 30, 'Beta'), (30, 40, 'Gamma')]"
        )

        # Check that we can recover the physical power using TotalAbsPow
        bp_abs = bp[bands] * bp["TotalAbsPow"].values[..., None]
        np.testing.assert_array_almost_equal(bp_abs[bands].values, bp_abs_true[bands].values)

        # 2-D EEG data
        win = int(4 * sf)
        freqs, psd = welch(data_full, sf_full, nperseg=win)
        bp = bandpower_from_psd(psd, freqs, ch_names=chan_full)
        assert bp.shape[0] == len(chan_full)
        assert bp.at[0, "Chan"].upper() == "CZ"
        assert bp.at[1, "FreqRes"] == 1 / (win / sf_full)
        # Unlabelled
        bp = bandpower_from_psd(psd, freqs, ch_names=None, relative=False)
        assert np.array_equal(bp.loc[:, "Chan"], ["CHAN000", "CHAN001", "CHAN002"])
        # The DataFrame and NumPy implementations give the same output
        np.testing.assert_allclose(
            bp[bands].to_numpy().T, bandpower_from_psd_ndarray(psd, freqs, relative=False)
        )

        # More channels than frequency bins (e.g. high-density EEG)
        freqs, psd = welch(np.random.rand(128, 2000), sf_full, nperseg=2 * sf_full)
        assert psd.shape[1] < 128
        assert bandpower_from_psd(psd, freqs).shape[0] == 128
        # ... but the PSD must be (n_channels, n_freqs)
        with pytest.raises(AssertionError):
            bandpower_from_psd(psd.T, freqs)

        # Bands with fewer than 2 frequency bins raise an informative error
        freqs, psd = welch(data, sf, nperseg=int(4 * sf))  # 0.25 Hz resolution
        with pytest.raises(ValueError, match="fewer than 2 frequency bins"):
            bandpower_from_psd(psd, freqs, bands=[(1, 4, "Delta"), (10.1, 10.2, "Narrow")])
        with pytest.raises(ValueError, match="At least 2 frequency bins"):
            bandpower_from_psd_ndarray(psd, freqs, bands=[(10, 10, "A")])

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
        bp_1d = bandpower_from_psd_ndarray(psd_1d, freqs, relative=True)
        assert bp_1d.shape == (len(bands),)
        assert np.isclose(bp_1d.sum(), 1, atol=1e-2)
        assert bandpower_from_psd_ndarray(psd_2d, freqs, relative=False).shape == (6, n_chan)
        assert (
            bandpower_from_psd_ndarray(psd_3d, freqs, bands=[(0.5, 4, "Delta")], relative=True) == 1
        ).all()

        # With negative values: we should get a logger warning
        freqs = np.arange(0, 50.5, 0.5)
        psd = np.random.normal(size=(6, freqs.size))
        with self.assertLogs("yasa", level="WARNING"):
            bandpower_from_psd(psd, freqs)
        with self.assertLogs("yasa", level="WARNING"):
            bandpower_from_psd_ndarray(psd, freqs)

    def test_irasa(self):
        """Test function IRASA."""
        # 1D Numpy
        freqs, psd_aperiodic, psd_osc, fit_params = irasa(data=data, sf=sf)
        assert np.isin(freqs, np.arange(1, 30.25, 0.25), True).all()
        assert np.median(psd_aperiodic) > np.median(psd_osc)
        assert fit_params.shape[0] == 1
        assert fit_params.at[0, "Slope"] < 0
        assert 0 < fit_params.at[0, "R^2"] <= 1

        # 2D Numpy
        _, psd_aperiodic, _, fit_params = irasa(data=data_eo, sf=sf_eo, ch_names=chan_eo)
        assert psd_aperiodic.shape[0] == len(chan_eo)
        assert fit_params["Chan"].tolist() == chan_eo
        _, _, _, fit_params = irasa(data=data_eo, sf=sf_eo, ch_names=None)
        assert fit_params["Chan"].tolist() == ["CHAN000", "CHAN001"]

        # 2D MNE (5 minutes of data)
        raw = data_mne.copy().crop(0, 300)
        assert len(irasa(raw, return_fit=False)) == 3
        freqs, psd_aperiodic, psd_osc, fit_params = irasa(raw, band=(2, 24), win_sec=2)
        assert freqs.min() == 2 and freqs.max() == 24
        assert psd_aperiodic.shape == psd_osc.shape == (len(raw.ch_names), freqs.size)
        assert fit_params["Chan"].tolist() == raw.ch_names

        # Messages are sent to the yasa logger
        with self.assertLogs("yasa", level="INFO") as logs:
            irasa(data=data, sf=sf, verbose=True)
        assert any("Fitting range" in msg for msg in logs.output)
        with self.assertLogs("yasa", level="WARNING"):
            irasa(data=data, sf=sf, band=(1, 80))  # Beyond the resampled Nyquist frequency
        with self.assertNoLogs("yasa", level="WARNING"):
            irasa(data=data, sf=sf, band=(1, 20))  # Within the resampled Nyquist frequency

        # Warnings when the evaluated frequency range exceeds the filters of the MNE Raw
        raw_filt = raw.copy().filter(1, 20, verbose=0)
        with self.assertLogs("yasa", level="WARNING") as logs:
            irasa(raw_filt, band=(1, 30))
        assert any("highpass" in msg for msg in logs.output)
        assert any("lowpass" in msg for msg in logs.output)

        # Data is too short for the resampling factors
        with pytest.raises(ValueError, match="too short for IRASA"):
            irasa(data=data[: int(5 * sf)], sf=sf, win_sec=4)

        # Per sleep stage, with an integer hypnogram. There is no stage 5 in the hypnogram.
        freqs, psd_ap, psd_osc, fit_params = irasa(
            data_full, sf=sf_full, ch_names=chan_full, hypno=hypno_full, include=(2, 3, 5)
        )
        assert list(psd_ap) == list(psd_osc) == [2, 3]
        assert psd_ap[2].shape == psd_osc[3].shape == (len(chan_full), freqs.size)
        assert fit_params.columns.tolist() == [
            "Stage", "Chan", "Intercept", "Slope", "R^2", "std(osc)"
        ]  # fmt: skip
        assert fit_params["Stage"].tolist() == [2, 2, 2, 3, 3, 3]
        assert fit_params["Chan"].tolist() == [*chan_full, *chan_full]
        # Same result as applying IRASA to the samples of the stage
        _, psd_ap_n2, _, fit_n2 = irasa(data_full[:, hypno_full == 2], sf=sf_full)
        np.testing.assert_allclose(psd_ap[2], psd_ap_n2)
        np.testing.assert_allclose(fit_params.iloc[:3, 2:], fit_n2.iloc[:, 1:])
        # With a Hypnogram and string labels
        freqs_hyp, psd_ap_hyp, _, fit_hyp = irasa(
            data_full, sf=sf_full, hypno=hyp_full, include=["N2", "N3"]
        )
        assert list(psd_ap_hyp) == ["N2", "N3"]
        assert fit_hyp["Stage"].unique().tolist() == ["N2", "N3"]
        np.testing.assert_allclose(freqs_hyp, freqs)
        # ... gives the same output as a Hypnogram with integer stages
        _, psd_ap_hyp_int, _, fit_hyp_int = irasa(
            data_full, sf=sf_full, hypno=hyp_full, include=(2, 3)
        )
        assert list(psd_ap_hyp_int) == [2, 3]
        np.testing.assert_array_equal(psd_ap_hyp["N3"], psd_ap_hyp_int[3])
        np.testing.assert_array_equal(fit_hyp.iloc[:, 1:], fit_hyp_int.iloc[:, 1:])
        assert len(irasa(data_full, sf=sf_full, hypno=hyp_full, return_fit=False)) == 3
        # Stages that are too short are skipped with a warning
        hypno_short = np.full(data_full.shape[1], 2)
        hypno_short[:500] = 1  # 5 seconds of N1, shorter than win_sec * max(hset)
        with self.assertLogs("yasa", level="WARNING") as logs:
            _, psd_ap, _, fit_params = irasa(
                data_full, sf=sf_full, hypno=hypno_short, include=(1, 2)
            )
        assert any("Stage 1 is shorter" in msg for msg in logs.output)
        assert list(psd_ap) == [2]
        assert fit_params["Stage"].unique().tolist() == [2]
        with pytest.raises(ValueError, match="All the stages"):
            irasa(data_full, sf=sf_full, hypno=hypno_short, include=1)

    def test_stft_power(self):
        """Test function stft_power"""
        window = [2, 4]
        step = [0, 0.1, 1]
        band = [(0.5, 20), (1, 30), [5, 12], None]
        norm = [True, False]
        interp = [True, False]

        for w, s, b, it, n in product(window, step, band, interp, norm):
            f, t, Sxx = stft_power(data, sf, window=w, step=s, band=b, interp=it, norm=n)
            assert Sxx.shape == (f.size, t.size)
            if it:
                assert t.size == data.size
            if n:
                np.testing.assert_allclose(Sxx.sum(0), 1)

        f, t, _ = stft_power(data, sf, window=4, step=0.1, band=(11, 16), interp=True, norm=False)

        assert f[1] - f[0] == 0.25
        assert t.size == data.size
        assert max(f) == 16
        assert min(f) == 11

        # Interpolation with fewer than 4 frequency bins
        f, t, Sxx = stft_power(data, sf, window=2, step=0.2, band=(12, 13), interp=True)
        assert f.size == 3
        assert Sxx.shape == (3, data.size)

    def test_plot_spectrogram(self):
        """Test function plot_spectrogram"""
        # Use 30 minutes of data to keep the test fast (spectrogram is O(n))
        n = int(0.5 * 3600 * sf_full)
        data_s = data_full[0, :n]
        hypno_s = hypno_full[:n]
        hypno_s_art = np.copy(hypno_s)
        hypno_s_art[hypno_s_art == 3.0] = -1  # Replace N3 by Artefact
        hypno_s_art[hypno_s_art == 4.0] = -2  # Replace REM by Unscored
        hyp_s = Hypnogram.from_integers(hypno_s[:: int(sf_full * 30)], freq="30s")
        # No hypnogram, with fmin/fmax and vmin/vmax
        fig = plot_spectrogram(data_s, sf_full, fmin=0.5, fmax=30, vmin=-50, vmax=100)
        assert isinstance(fig, plt.Figure)
        # Integer hypnogram array with trimperc
        plot_spectrogram(data_s, sf_full, hypno_s, trimperc=5)
        # With artefact (-1) and unscored (-2) stages
        plot_spectrogram(data_s, sf_full, hypno_s_art, trimperc=5)
        # Hypnogram object (auto-upsampled) with kwargs
        plot_spectrogram(data_s, sf_full, hyp_s, lw=1, fill_color="whitesmoke")
        plt.close("all")
        # Errors: vmin and vmax must be both provided or neither
        with pytest.raises(AssertionError):
            plot_spectrogram(data_s, sf_full, fmin=0.5, fmax=30, vmin=-50)
        with pytest.raises(AssertionError):
            plot_spectrogram(data_s, sf_full, fmin=0.5, fmax=30, vmax=100)
