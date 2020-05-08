.. _api_ref:

.. currentmodule:: yasa

API reference
=============

Detection
---------

.. _detect:

.. autosummary::
   :toctree: generated/

    art_detect
    rem_detect
    REMResults
    spindles_detect
    SpindlesResults
    sw_detect
    SWResults

Hypnogram
---------

.. _hgram:

.. autosummary::
   :toctree: generated/

    hypno_upsample_to_data
    hypno_upsample_to_sf
    hypno_str_to_int
    hypno_int_to_str
    load_profusion_hypno
    transition_matrix
    sleep_statistics

Signal processing
-----------------

.. _others:

.. autosummary::
   :toctree: generated/

    moving_transform
    sliding_window
    trimbothstd
    get_centered_indices

Spectral analyses
-----------------

.. _bpower:

.. autosummary::
   :toctree: generated/

    bandpower
    bandpower_from_psd
    bandpower_from_psd_ndarray
    irasa
    plot_spectrogram
    stft_power
