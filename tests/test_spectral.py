"""Test the functions in the yasa/spectral.py file."""

import logging
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.signal import welch

from yasa.hypno import Hypnogram
from yasa.plotting import plot_spectrogram
from yasa.spectral import (
    bandpower,
    bandpower_from_psd,
    bandpower_from_psd_ndarray,
    irasa,
    stft_power,
)

BANDS = ["Delta", "Theta", "Alpha", "Sigma", "Beta", "Gamma"]


##############################################################################
# FIXTURES
##############################################################################


@pytest.fixture
def yasa_caplog(caplog, monkeypatch):
    """caplog that also captures the messages of the YASA logger (which does not propagate)."""
    monkeypatch.setattr(logging.getLogger("yasa"), "propagate", True)
    return caplog


def _has_warning(caplog):
    return any(r.levelno == logging.WARNING for r in caplog.records)


@pytest.fixture(scope="module")
def full_1h(full_6hrs):
    """One hour of the full recording and its hypnogram.

    The third hour of the recording includes all the sleep stages (W, N1, N2, N3 and REM).
    """
    sl = slice(720_000, 1_080_000)
    hypno = full_6hrs.hypno[sl]
    # Hypnogram object for testing Hypnogram-based hypno support (1 value per 30-s epoch)
    hyp = Hypnogram.from_integers(hypno[:: int(full_6hrs.sf * 30)], freq="30s")
    return SimpleNamespace(
        data=full_6hrs.data[:, sl], hypno=hypno, hyp=hyp, chan=full_6hrs.chan, sf=full_6hrs.sf
    )


@pytest.fixture(scope="module")
def raw_eeg(_raw_sub02):
    """EEG channels of sub-02, as a MNE Raw."""
    return _raw_sub02.copy().pick("eeg")


@pytest.fixture(scope="module")
def raw_eeg_5min(raw_eeg):
    """5 minutes of the EEG channels of sub-02."""
    return raw_eeg.copy().crop(0, 300)


@pytest.fixture(scope="module")
def eo(_raw_resting_eo):
    """Eyes-open 6 minutes resting-state, 2 channels, 200 Hz."""
    raw = _raw_resting_eo
    data = raw.get_data(units=dict(eeg="uV", emg="uV", eog="uV", ecg="uV"))
    return SimpleNamespace(data=data, sf=raw.info["sfreq"], chan=raw.ch_names)


@pytest.fixture(scope="module")
def bp_n2_int(full_1h):
    """Bandpower in N2 with an integer hypnogram array."""
    return bandpower(full_1h.data, sf=full_1h.sf, hypno=full_1h.hypno, include=2)


@pytest.fixture(scope="module")
def hypno_short_n1(full_1h):
    """Hypnogram with only 2 seconds of N1, shorter than the 4-seconds Welch window."""
    hypno = np.full(full_1h.data.shape[1], 2)
    hypno[:200] = 1
    return hypno


##############################################################################
# BANDPOWER
##############################################################################


def test_bandpower_raw(raw_eeg):
    """Raw MNE multi-channel."""
    bp = bandpower(raw_eeg)
    assert bp.index.tolist() == raw_eeg.ch_names
    np.testing.assert_allclose(bp[BANDS].sum(axis=1), 1, atol=1e-2)


@pytest.mark.parametrize(
    "kwargs, ch_names",
    [(dict(bandpass=True), ["CHAN000"]), (dict(ch_names="F4"), ["F4"])],
    ids=["unlabelled", "labelled"],
)
def test_bandpower_single_channel(n2_spindles, kwargs, ch_names):
    """Single channel NumPy array, unlabelled or labelled."""
    bp = bandpower(n2_spindles.data, sf=n2_spindles.sf, **kwargs)
    assert bp.index.tolist() == ch_names


def test_bandpower_hypno_missing_stage(full_1h):
    """Multi channel NumPy with hypnogram. There is no stage 5 in the hypnogram."""
    bp = bandpower(
        full_1h.data, sf=full_1h.sf, hypno=full_1h.hypno, include=(3, 4, 5), bandpass=True
    )
    assert bp.index.get_level_values("Stage").unique().tolist() == [3, 4]


def test_bandpower_raw_hypno(raw_eeg, hypno_sub02):
    """Raw MNE with hypnogram."""
    hypno = Hypnogram(hypno_sub02, freq="30s").upsample_to_data(raw_eeg)
    bp = bandpower(raw_eeg, hypno=hypno, include=2)
    assert bp.shape[0] == len(raw_eeg.ch_names)


def test_bandpower_short_stage_skipped(full_1h, hypno_short_n1, yasa_caplog):
    """Stages shorter than the Welch window are skipped with a warning."""
    bp = bandpower(full_1h.data, sf=full_1h.sf, hypno=hypno_short_n1, include=(1, 2))
    assert _has_warning(yasa_caplog)
    assert bp.index.get_level_values("Stage").unique().tolist() == [2]


def test_bandpower_short_stage_error(full_1h, hypno_short_n1):
    """An error is raised if all the stages are shorter than the Welch window."""
    with pytest.raises(ValueError, match="shorter than the Welch window"):
        bandpower(full_1h.data, sf=full_1h.sf, hypno=hypno_short_n1, include=1)


def test_bandpower_str_hypno(full_1h, bp_n2_int):
    """String hypnogram arrays (with string include) are labelled with the string stages."""
    hypno_str = np.where(full_1h.hypno == 2, "N2", "Other")
    bp_str = bandpower(full_1h.data, sf=full_1h.sf, hypno=hypno_str, include="N2")
    assert bp_str.index.get_level_values("Stage").unique().tolist() == ["N2"]
    np.testing.assert_allclose(bp_str.to_numpy(float), bp_n2_int.to_numpy(float))


def test_bandpower_hypnogram(full_1h):
    """Hypnogram instance: the output is labelled with the stages as given in include."""
    kwargs = dict(sf=full_1h.sf, ch_names=full_1h.chan, hypno=full_1h.hyp)
    bp_hyp_int = bandpower(full_1h.data, include=(2, 3), **kwargs)
    bp_hyp_str = bandpower(full_1h.data, include=["N2", "N3"], **kwargs)
    assert bp_hyp_int.index.names == ["Stage", "Chan"]
    assert bp_hyp_int.index.get_level_values("Stage").unique().tolist() == [2, 3]
    assert bp_hyp_str.index.get_level_values("Stage").unique().tolist() == ["N2", "N3"]
    np.testing.assert_array_equal(bp_hyp_int.to_numpy(), bp_hyp_str.to_numpy())


##############################################################################
# BANDPOWER_FROM_PSD
##############################################################################


def test_bandpower_from_psd_1d(n2_spindles):
    """1-D EEG data."""
    data, sf = n2_spindles.data, n2_spindles.sf
    win = int(2 * sf)
    freqs, psd = welch(data, sf, nperseg=win)
    bp_abs_true = bandpower_from_psd(psd, freqs, relative=False)
    bp = bandpower_from_psd(psd, freqs, ch_names=["F4"])
    assert bp.shape[0] == 1
    assert bp.columns.tolist() == ["Chan", *BANDS, "TotalAbsPow", "FreqRes", "Relative"]
    assert bp.at[0, "Chan"] == "F4"
    assert bp.at[0, "FreqRes"] == 1 / (win / sf)
    assert np.isclose(bp.loc[0, BANDS].sum(), 1, atol=1e-2)
    assert (
        bp.bands_ == "[(0.5, 4, 'Delta'), (4, 8, 'Theta'), "
        "(8, 12, 'Alpha'), (12, 16, 'Sigma'), "
        "(16, 30, 'Beta'), (30, 40, 'Gamma')]"
    )
    # Check that we can recover the physical power using TotalAbsPow
    bp_abs = bp[BANDS] * bp["TotalAbsPow"].values[..., None]
    np.testing.assert_array_almost_equal(bp_abs[BANDS].values, bp_abs_true[BANDS].values)


def test_bandpower_from_psd_2d(full_1h):
    """2-D EEG data, labelled and unlabelled."""
    win = int(4 * full_1h.sf)
    freqs, psd = welch(full_1h.data, full_1h.sf, nperseg=win)
    bp = bandpower_from_psd(psd, freqs, ch_names=full_1h.chan)
    assert bp.shape[0] == len(full_1h.chan)
    assert bp.at[0, "Chan"].upper() == "CZ"
    assert bp.at[1, "FreqRes"] == 1 / (win / full_1h.sf)
    # Unlabelled
    bp = bandpower_from_psd(psd, freqs, ch_names=None, relative=False)
    assert np.array_equal(bp.loc[:, "Chan"], ["CHAN000", "CHAN001", "CHAN002"])
    # The DataFrame and NumPy implementations give the same output
    np.testing.assert_allclose(
        bp[BANDS].to_numpy().T, bandpower_from_psd_ndarray(psd, freqs, relative=False)
    )


def test_bandpower_from_psd_many_channels():
    """More channels than frequency bins (e.g. high-density EEG)."""
    freqs, psd = welch(np.random.rand(128, 2000), 100, nperseg=200)
    assert psd.shape[1] < 128
    assert bandpower_from_psd(psd, freqs).shape[0] == 128
    # ... but the PSD must be (n_channels, n_freqs)
    with pytest.raises(AssertionError):
        bandpower_from_psd(psd.T, freqs)


def test_bandpower_from_psd_too_few_bins(n2_spindles):
    """Bands with fewer than 2 frequency bins raise an informative error."""
    sf = n2_spindles.sf
    freqs, psd = welch(n2_spindles.data, sf, nperseg=int(4 * sf))  # 0.25 Hz resolution
    with pytest.raises(ValueError, match="fewer than 2 frequency bins"):
        bandpower_from_psd(psd, freqs, bands=[(1, 4, "Delta"), (10.1, 10.2, "Narrow")])
    with pytest.raises(ValueError, match="At least 2 frequency bins"):
        bandpower_from_psd_ndarray(psd, freqs, bands=[(10, 10, "A")])


def test_bandpower_from_psd_ndarray():
    """Bandpower from PSD with 1D, 2D and 3D arrays."""
    sf = 200
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
    assert bp_1d.shape == (len(BANDS),)
    assert np.isclose(bp_1d.sum(), 1, atol=1e-2)
    assert bandpower_from_psd_ndarray(psd_2d, freqs, relative=False).shape == (6, n_chan)
    assert (
        bandpower_from_psd_ndarray(psd_3d, freqs, bands=[(0.5, 4, "Delta")], relative=True) == 1
    ).all()


@pytest.mark.parametrize("func", [bandpower_from_psd, bandpower_from_psd_ndarray])
def test_bandpower_from_psd_negative(func, yasa_caplog):
    """With negative values: we should get a logger warning."""
    freqs = np.arange(0, 50.5, 0.5)
    psd = np.random.normal(size=(6, freqs.size))
    func(psd, freqs)
    assert _has_warning(yasa_caplog)


##############################################################################
# IRASA
##############################################################################


def test_irasa_1d(n2_spindles):
    """1D NumPy."""
    freqs, psd_aperiodic, psd_osc, fit_params = irasa(data=n2_spindles.data, sf=n2_spindles.sf)
    assert np.isin(freqs, np.arange(1, 30.25, 0.25), True).all()
    assert np.median(psd_aperiodic) > np.median(psd_osc)
    assert fit_params.shape[0] == 1
    assert fit_params.at[0, "Slope"] < 0
    assert 0 < fit_params.at[0, "R^2"] <= 1


def test_irasa_2d(eo):
    """2D NumPy, labelled."""
    _, psd_aperiodic, _, fit_params = irasa(data=eo.data, sf=eo.sf, ch_names=eo.chan)
    assert psd_aperiodic.shape[0] == len(eo.chan)
    assert fit_params["Chan"].tolist() == eo.chan


def test_irasa_2d_unlabelled(eo):
    """2D NumPy, unlabelled."""
    _, _, _, fit_params = irasa(data=eo.data, sf=eo.sf, ch_names=None)
    assert fit_params["Chan"].tolist() == ["CHAN000", "CHAN001"]


def test_irasa_raw(raw_eeg_5min):
    """2D MNE (5 minutes of data)."""
    raw = raw_eeg_5min
    assert len(irasa(raw, return_fit=False)) == 3
    freqs, psd_aperiodic, psd_osc, fit_params = irasa(raw, band=(2, 24), win_sec=2)
    assert freqs.min() == 2 and freqs.max() == 24
    assert psd_aperiodic.shape == psd_osc.shape == (len(raw.ch_names), freqs.size)
    assert fit_params["Chan"].tolist() == raw.ch_names


def test_irasa_verbose(n2_spindles, yasa_caplog):
    """Messages are sent to the yasa logger."""
    irasa(data=n2_spindles.data, sf=n2_spindles.sf, verbose=True)
    assert "Fitting range" in yasa_caplog.text


def test_irasa_band_beyond_nyquist(n2_spindles, yasa_caplog):
    """Warning when the band is beyond the resampled Nyquist frequency."""
    irasa(data=n2_spindles.data, sf=n2_spindles.sf, band=(1, 80))
    assert _has_warning(yasa_caplog)


def test_irasa_band_within_nyquist(n2_spindles, yasa_caplog):
    """No warning when the band is within the resampled Nyquist frequency."""
    irasa(data=n2_spindles.data, sf=n2_spindles.sf, band=(1, 20))
    assert not _has_warning(yasa_caplog)


def test_irasa_raw_filter_warnings(raw_eeg_5min, yasa_caplog):
    """Warnings when the evaluated frequency range exceeds the filters of the MNE Raw."""
    raw_filt = raw_eeg_5min.copy().filter(1, 20, verbose=0)
    irasa(raw_filt, band=(1, 30))
    warnings = [r.getMessage() for r in yasa_caplog.records if r.levelno == logging.WARNING]
    assert any("highpass" in msg for msg in warnings)
    assert any("lowpass" in msg for msg in warnings)


def test_irasa_too_short(n2_spindles):
    """Data is too short for the resampling factors."""
    sf = n2_spindles.sf
    with pytest.raises(ValueError, match="too short for IRASA"):
        irasa(data=n2_spindles.data[: int(5 * sf)], sf=sf, win_sec=4)


@pytest.fixture(scope="module")
def irasa_stages(full_1h):
    """IRASA per sleep stage with an integer hypnogram. There is no stage 5 in the hypnogram."""
    return irasa(
        full_1h.data, sf=full_1h.sf, ch_names=full_1h.chan, hypno=full_1h.hypno, include=(2, 3, 5)
    )


def test_irasa_hypno(full_1h, irasa_stages):
    """Per sleep stage, with an integer hypnogram."""
    freqs, psd_ap, psd_osc, fit_params = irasa_stages
    chan = full_1h.chan
    assert list(psd_ap) == list(psd_osc) == [2, 3]
    assert psd_ap[2].shape == psd_osc[3].shape == (len(chan), freqs.size)
    assert fit_params.columns.tolist() == [
        "Stage", "Chan", "Intercept", "Slope", "R^2", "std(osc)"
    ]  # fmt: skip
    assert fit_params["Stage"].tolist() == [2, 2, 2, 3, 3, 3]
    assert fit_params["Chan"].tolist() == [*chan, *chan]


def test_irasa_hypno_same_as_stage_samples(full_1h, irasa_stages):
    """Same result as applying IRASA to the samples of the stage."""
    _, psd_ap, _, fit_params = irasa_stages
    _, psd_ap_n2, _, fit_n2 = irasa(full_1h.data[:, full_1h.hypno == 2], sf=full_1h.sf)
    np.testing.assert_allclose(psd_ap[2], psd_ap_n2)
    np.testing.assert_allclose(fit_params.iloc[:3, 2:], fit_n2.iloc[:, 1:])


def test_irasa_hypnogram(full_1h, irasa_stages):
    """With a Hypnogram, string labels give the same output as integer stages."""
    data, sf, hyp = full_1h.data, full_1h.sf, full_1h.hyp
    freqs_hyp, psd_ap_hyp, _, fit_hyp = irasa(data, sf=sf, hypno=hyp, include=["N2", "N3"])
    assert list(psd_ap_hyp) == ["N2", "N3"]
    assert fit_hyp["Stage"].unique().tolist() == ["N2", "N3"]
    np.testing.assert_allclose(freqs_hyp, irasa_stages[0])
    _, psd_ap_hyp_int, _, fit_hyp_int = irasa(data, sf=sf, hypno=hyp, include=(2, 3))
    assert list(psd_ap_hyp_int) == [2, 3]
    np.testing.assert_array_equal(psd_ap_hyp["N3"], psd_ap_hyp_int[3])
    np.testing.assert_array_equal(fit_hyp.iloc[:, 1:], fit_hyp_int.iloc[:, 1:])
    assert len(irasa(data, sf=sf, hypno=hyp, return_fit=False)) == 3


def test_irasa_hypno_short_stage(full_1h, yasa_caplog):
    """Stages that are too short are skipped with a warning."""
    hypno_short = np.full(full_1h.data.shape[1], 2)
    hypno_short[:500] = 1  # 5 seconds of N1, shorter than win_sec * max(hset)
    _, psd_ap, _, fit_params = irasa(full_1h.data, sf=full_1h.sf, hypno=hypno_short, include=(1, 2))
    assert "Stage 1 is shorter" in yasa_caplog.text
    assert list(psd_ap) == [2]
    assert fit_params["Stage"].unique().tolist() == [2]
    with pytest.raises(ValueError, match="All the stages"):
        irasa(full_1h.data, sf=full_1h.sf, hypno=hypno_short, include=1)


##############################################################################
# STFT_POWER
##############################################################################


# Representative parameter combinations (covers all values, avoids full Cartesian product of 96
# iterations)
@pytest.mark.parametrize(
    "window, step, band, interp, norm",
    [
        (2, 0, (0.5, 20), True, True),
        (4, 0.1, (1, 30), False, True),
        (2, 1, [5, 12], True, False),
        (4, 0, None, False, False),
        (4, 1, None, True, True),
        (2, 0.1, (1, 30), True, False),
    ],
)
def test_stft_power(n2_spindles, window, step, band, interp, norm):
    """Test the shape and normalization of stft_power."""
    data, sf = n2_spindles.data, n2_spindles.sf
    f, t, Sxx = stft_power(data, sf, window=window, step=step, band=band, interp=interp, norm=norm)
    assert Sxx.shape == (f.size, t.size)
    if interp:
        assert t.size == data.size
    if norm:
        np.testing.assert_allclose(Sxx.sum(0), 1)


def test_stft_power_band(n2_spindles):
    """Test the frequency resolution and range of stft_power."""
    data, sf = n2_spindles.data, n2_spindles.sf
    f, t, _ = stft_power(data, sf, window=4, step=0.1, band=(11, 16), interp=True, norm=False)
    assert f[1] - f[0] == 0.25
    assert t.size == data.size
    assert max(f) == 16
    assert min(f) == 11


def test_stft_power_interp_few_bins(n2_spindles):
    """Interpolation with fewer than 4 frequency bins."""
    data, sf = n2_spindles.data, n2_spindles.sf
    f, t, Sxx = stft_power(data, sf, window=2, step=0.2, band=(12, 13), interp=True)
    assert f.size == 3
    assert Sxx.shape == (3, data.size)


##############################################################################
# PLOT_SPECTROGRAM
##############################################################################


@pytest.fixture(scope="module")
def spectro_30min(full_1h):
    """30 minutes of data to keep the tests fast (spectrogram is O(n))."""
    sf = full_1h.sf
    n = int(0.5 * 3600 * sf)
    hypno = full_1h.hypno[:n]
    hypno_art = np.copy(hypno)
    hypno_art[hypno_art == 3.0] = -1  # Replace N3 by Artefact
    hypno_art[hypno_art == 4.0] = -2  # Replace REM by Unscored
    hyp = Hypnogram.from_integers(hypno[:: int(sf * 30)], freq="30s")
    return SimpleNamespace(
        data=full_1h.data[0, :n], sf=sf, hypno=hypno, hypno_art=hypno_art, hyp=hyp
    )


@pytest.mark.parametrize(
    "hypno, kwargs",
    [
        (None, dict(fmin=0.5, fmax=30, vmin=-50, vmax=100)),  # No hypno, with fmin/fmax/vmin/vmax
        ("hypno", dict(trimperc=5)),  # Integer hypnogram array with trimperc
        ("hypno_art", dict(trimperc=5)),  # With artefact (-1) and unscored (-2) stages
        ("hyp", dict(lw=1, fill_color="whitesmoke")),  # Hypnogram object (auto-upsampled)
    ],
    ids=["no_hypno", "int_hypno", "artefact_unscored", "hypnogram"],
)
def test_plot_spectrogram(spectro_30min, hypno, kwargs):
    """Test function plot_spectrogram"""
    hyp = getattr(spectro_30min, hypno) if hypno is not None else None
    fig = plot_spectrogram(spectro_30min.data, spectro_30min.sf, hyp, **kwargs)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


@pytest.mark.parametrize("vlim", [dict(vmin=-50), dict(vmax=100)])
def test_plot_spectrogram_vmin_vmax(spectro_30min, vlim):
    """vmin and vmax must be both provided or neither."""
    with pytest.raises(AssertionError):
        plot_spectrogram(spectro_30min.data, spectro_30min.sf, fmin=0.5, fmax=30, **vlim)
