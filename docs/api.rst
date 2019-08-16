.. _api_ref:

.. currentmodule:: yasa

API reference
=============

Detection
---------

.. _detect:

.. autosummary::
   :toctree: generated/

    rem_detect
    spindles_detect
    spindles_detect_multi
    sw_detect
    sw_detect_multi
    get_bool_vector
    get_sync_sw

Bandpower
---------

.. _bpower:

.. autosummary::
   :toctree: generated/

    bandpower
    bandpower_from_psd
    stft_power

Hypnogram
---------

.. _hgram:

.. autosummary::
   :toctree: generated/

    hypno_upsample_to_data
    hypno_upsample_to_sf
    hypno_str_to_int
    hypno_int_to_str

Others
------

.. _others:

.. autosummary::
   :toctree: generated/

    moving_transform
    sliding_window
    trimbothstd
