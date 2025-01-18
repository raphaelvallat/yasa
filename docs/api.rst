.. _api_ref:

.. currentmodule:: yasa

API reference
=============

Automatic sleep staging & events detection
------------------------------------------

.. _staging:

.. autosummary::
    :toctree: generated/
    :nosignatures:

    SleepStaging
    art_detect
    rem_detect
    REMResults
    spindles_detect
    SpindlesResults
    sw_detect
    SWResults
    compare_detection

Hypnogram & sleep statistics
----------------------------

.. _hgram:

.. autosummary::
    :toctree: generated/
    :nosignatures:

    hypno_upsample_to_data
    hypno_upsample_to_sf
    hypno_str_to_int
    hypno_int_to_str
    hypno_find_periods
    load_profusion_hypno
    plot_hypnogram
    plot_spectrogram
    transition_matrix
    sleep_statistics
    Hypnogram
    simulate_hypnogram
    EpochByEpochAgreement
    SleepStatsAgreement

Spectral analyses
-----------------

.. _spectral:

.. autosummary::
    :toctree: generated/
    :nosignatures:

    bandpower
    bandpower_from_psd
    bandpower_from_psd_ndarray
    irasa
    moving_transform
    plot_spectrogram
    sliding_window
    stft_power
    topoplot

Heart rate analysis
-------------------

.. _others:

.. autosummary::
    :toctree: generated/
    :nosignatures:

    hrv_stage
