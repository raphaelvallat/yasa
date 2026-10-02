"""Private helpers to validate the data and hypnogram passed to YASA functions."""

import logging

import mne
import numpy as np

from .hypno import Hypnogram
from .others import trimbothstd

logger = logging.getLogger("yasa")

# Units used when extracting data from a MNE Raw object (MNE stores data in Volts)
UNITS_UV = dict(eeg="uV", emg="uV", eog="uV", ecg="uV")


def _check_data(data, sf=None, ch_names=None):
    """Validate EEG data and extract data, sf and ch_names from a MNE Raw object if needed.

    Returns
    -------
    data : np.ndarray
        2D float64 array of shape (n_chan, n_samples), in uV.
    sf : int or float
        Sampling frequency.
    ch_names : list of str
        Channel names, of length n_chan.
    raw : :py:class:`mne.io.BaseRaw` or None
        The original MNE Raw object, if ``data`` was one. It is needed to align a
        :py:class:`yasa.Hypnogram` with the recording using absolute timestamps.
    """
    raw = None
    if isinstance(data, mne.io.BaseRaw):
        if sf is not None:
            logger.warning("sf parameter will be ignored, sf from MNE Raw will be used")
        if ch_names is not None:
            logger.warning("ch_names parameter will be ignored, ch_names from MNE Raw will be used")
        raw = data
        sf = raw.info["sfreq"]
        ch_names = raw.ch_names
        data = raw.get_data(units=UNITS_UV)
    else:
        assert sf is not None, "sf must be specified if not using MNE Raw."
        if isinstance(sf, (np.ndarray, np.generic)):  # e.g. array(100.) or np.int64(100)
            sf = sf.item()
        assert isinstance(sf, (int, float)), "sf must be int or float."
    data = np.asarray(data, dtype=np.float64)
    assert data.ndim in [1, 2], "data must be 1D (times) or 2D (chan, times)."
    data = np.atleast_2d(data)  # (n_chan, n_samples)
    n_chan = data.shape[0]
    if ch_names is None:
        ch_names = ["CHAN" + str(i).zfill(3) for i in range(n_chan)]
    else:
        ch_names = [str(c) for c in np.atleast_1d(ch_names)]
        assert len(ch_names) == n_chan, "ch_names must match data.shape[0]."
    return data, sf, ch_names, raw


def _check_hypno_include(hypno, include, data, sf, verbose=False):
    """Validate a hypnogram and the stages to include.

    Parameters
    ----------
    hypno : array_like or :py:class:`yasa.Hypnogram`
        Upsampled hypnogram, or a Hypnogram instance that is upsampled to ``data``.
    include : int, str or list
        Stages to include. String labels are only allowed with a Hypnogram instance.
    data : np.ndarray or :py:class:`mne.io.BaseRaw`
        The data. Passing the original MNE Raw object (instead of the extracted array) enables
        the timestamp-based alignment of :py:meth:`yasa.Hypnogram.upsample_to_data`.
    sf : float
        Sampling frequency of ``data``.

    Returns
    -------
    hypno : np.ndarray
        1D hypnogram with one value per sample. Numeric hypnograms are returned as a signed
        integer array, which may be the user's array itself: callers must not modify it in-place.
    include : np.ndarray
        1D array of the stages to include, with the same dtype kind as ``hypno``.
    int_to_str : dict
        Mapping from the integer stages to the string labels of ``include``. Empty unless
        ``include`` contained string labels.
    """
    assert include is not None, "include cannot be None if hypno is given"
    include = np.atleast_1d(np.asarray(include))
    assert include.size >= 1, "`include` must have at least one element."
    n_samples = data.n_times if isinstance(data, mne.io.BaseRaw) else data.shape[-1]
    int_to_str = {}
    if isinstance(hypno, Hypnogram):
        if include.dtype.kind in ("U", "S", "O"):
            unknown = [str(s) for s in include if s not in hypno.mapping]
            assert not unknown, (
                f"The following stages in `include` are not valid labels of the "
                f"hypnogram: {unknown}. Valid labels are {sorted(hypno.mapping)}."
            )
            include = np.array([hypno.mapping[s] for s in include], dtype=int)
            int_to_str = hypno.mapping_int
        hypno = hypno.upsample_to_data(data, sf=sf, verbose=verbose)
    hypno = np.asarray(hypno)
    assert hypno.ndim == 1, "Hypno must be one dimensional."
    assert hypno.size == n_samples, "Hypno must have same size as data."
    if hypno.dtype.kind in "iuf" and include.dtype.kind in "iuf":
        # Numeric stages: compare as integers, e.g. a float hypnogram loaded from a txt file
        assert np.array_equal(include, np.round(include)), "include must contain whole numbers."
        include = include.astype(int)
        if hypno.dtype.kind != "i":
            hypno = hypno.astype(int)
    assert hypno.dtype.kind == include.dtype.kind, "hypno and include must have same dtype"
    assert np.isin(hypno, include).any(), (
        "None of the stages specified in `include` are present in hypno."
    )
    return hypno, include, int_to_str


def _check_data_hypno(
    data, sf=None, ch_names=None, hypno=None, include=None, check_amp=True, verbose=False
):
    """Helper functions for preprocessing of data and hypnogram.

    Accepts an upsampled integer hypnogram (array_like) or a :py:class:`yasa.Hypnogram` instance.
    When a :py:class:`yasa.Hypnogram` is passed, it is automatically upsampled to match ``data``
    and ``include`` may be specified as string stage labels (e.g. ``["N2", "REM"]``).
    """
    # 1) Extract data as a 2D NumPy array, and check channel names
    data, sf, ch_names, raw = _check_data(data, sf, ch_names)
    n_chan, n_samples = data.shape

    # 2) Check hypnogram. The original Raw is passed so that a Hypnogram with a start time is
    # aligned with the recording using absolute timestamps.
    if hypno is not None:
        hypno, include, _ = _check_hypno_include(
            hypno, include, raw if raw is not None else data, sf, verbose=verbose
        )
        assert hypno.dtype.kind == "i", (
            "hypno must be an integer array. Use a yasa.Hypnogram to work with string labels."
        )
        hypno = hypno.astype(int, copy=False)
        logger.info("Number of unique values in hypno = %i", np.unique(hypno).size)

    # 3) Check data amplitude
    logger.info("Number of samples in data = %i", n_samples)
    logger.info("Sampling frequency = %.2f Hz", sf)
    logger.info("Data duration = %.2f seconds", n_samples / sf)
    all_ptp = np.ptp(data, axis=-1)
    all_trimstd = trimbothstd(data, cut=0.05)
    bad_chan = np.zeros(n_chan, dtype=bool)
    for i in range(n_chan):
        logger.info("Trimmed standard deviation of %s = %.4f uV" % (ch_names[i], all_trimstd[i]))
        logger.info("Peak-to-peak amplitude of %s = %.4f uV" % (ch_names[i], all_ptp[i]))
        if check_amp and not (0.1 < all_trimstd[i] < 1e3):
            logger.error(
                "Wrong data amplitude for %s "
                "(trimmed STD = %.3f). Unit of data MUST be uV! "
                "Channel will be skipped." % (ch_names[i], all_trimstd[i])
            )
            bad_chan[i] = True

    # 4) Create sleep stage vector mask
    if hypno is not None:
        mask = np.isin(hypno, include)
    else:
        mask = np.ones(n_samples, dtype=bool)

    return (data, sf, ch_names, hypno, include, mask, n_chan, n_samples, bad_chan)
