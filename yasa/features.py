"""
Sleep features.

This file calculates a set of features from the PSG sleep data.

These include:

- Spectral power (with and without adjustement for 1/f)
- Spindles and slow-waves detection
- Slow-waves / spindles phase-amplitude coupling
- Entropy and fractal dimension

Author: Dr Raphael Vallat <raphaelvallat@berkeley.edu>, UC Berkeley.
Date: March 2021

DANGER: This function has not been extensively debugged and validated.
Use at your own risk.
"""
import mne
import yasa
import logging
import numpy as np
import pandas as pd
import antropy as ant
import scipy.signal as sp_sig
import scipy.stats as sp_stats


logger = logging.getLogger('yasa')

__all__ = ['compute_features_stage']


def compute_features_stage(raw, hypno, max_freq=35, spindles_params=dict(),
                           sw_params=dict(), do_1f=True):
    """Calculate a set of features for each sleep stage from PSG data.

    Features are calculated for N2, N3, NREM (= N2 + N3) and REM sleep.

    Parameters
    ----------
    raw : :py:class:`mne.io.BaseRaw`
        An MNE Raw instance.
    hypno : array_like
        Sleep stage (hypnogram). The hypnogram must have the exact same
        number of samples as ``data``. To upsample your hypnogram,
        please refer to :py:func:`yasa.hypno_upsample_to_data`.

        .. note::
            The default hypnogram format in YASA is a 1D integer
            vector where:

            - -2 = Unscored
            - -1 = Artefact / Movement
            - 0 = Wake
            - 1 = N1 sleep
            - 2 = N2 sleep
            - 3 = N3 sleep
            - 4 = REM sleep

    max_freq : int
        Maximum frequency. This will be used to bandpass-filter the data and
        to calculate 1 Hz bins bandpower.
    kwargs_sp : dict
        Optional keywords arguments that are passed to the
        :py:func:`yasa.spindles_detect` function. We strongly recommend
        adapting the thresholds to your population (e.g. more liberal for
        older adults).
    kwargs_sw : dict
        Optional keywords arguments that are passed to the
        :py:func:`yasa.sw_detect` function. We strongly recommend
        adapting the thresholds to your population (e.g. more liberal for
        older adults).

    Returns
    -------
    feature : pd.DataFrame
        A long-format dataframe with stage and channel as index and
        all the calculated metrics as columns.
    """
    # #########################################################################
    # 1) PREPROCESSING
    # #########################################################################

    # Safety checks
    assert isinstance(max_freq, int), "`max_freq` must be int."
    assert isinstance(raw, mne.io.BaseRaw), "`raw` must be a MNE Raw object."
    assert isinstance(spindles_params, dict)
    assert isinstance(sw_params, dict)

    # Define 1 Hz bins frequency bands for bandpower
    # Similar to [(0.5, 1, "0.5-1"), (1, 2, "1-2"), ..., (34, 35, "34-35")]
    bands = []
    freqs = [0.5] + list(range(1, max_freq + 1))
    for i, b in enumerate(freqs[:-1]):
        bands.append(tuple((b, freqs[i + 1], "%s-%s" % (b, freqs[i + 1]))))
    # Append traditional bands
    bands_classic = [
        (0.5, 1, 'slowdelta'), (1, 4, 'fastdelta'), (0.5, 4, 'delta'),
        (4, 8, 'theta'), (8, 12, 'alpha'), (12, 16, 'sigma'), (16, 30, 'beta'),
        (30, max_freq, 'gamma')]
    bands = bands_classic + bands

    # Find min and maximum frequencies. These will be used for bandpass-filter
    # and 1/f adjustement of bandpower. l_freq = 0.5 / h_freq = 35 Hz.
    all_freqs_sorted = np.sort(np.unique(
        [b[0] for b in bands] + [b[1] for b in bands]))
    l_freq = all_freqs_sorted[0]
    h_freq = all_freqs_sorted[-1]

    # Mapping dictionnary integer to string for sleep stages (2 --> N2)
    stage_mapping = {
        -2: 'Unscored',
        -1: 'Artefact',
        0: 'Wake',
        1: 'N1',
        2: 'N2',
        3: 'N3',
        4: 'REM',
        6: 'NREM',
        7: 'WN'  # Whole night = N2 + N3 + REM
    }

    # Hypnogram check + calculate NREM hypnogram
    hypno = np.asarray(hypno, dtype=int)
    assert hypno.ndim == 1, 'Hypno must be one dimensional.'
    unique_hypno = np.unique(hypno)
    logger.info('Number of unique values in hypno = %i', unique_hypno.size)

    # IMPORTANT: NREM is defined as N2 + N3, excluding N1 sleep.
    hypno_NREM = pd.Series(hypno).replace({2: 6, 3: 6}).to_numpy()
    minutes_of_NREM = (hypno_NREM == 6).sum() / (60 * raw.info['sfreq'])

    # WN = Whole night = N2 + N3 + REM (excluding N1)
    hypno_WN = pd.Series(hypno).replace({2: 7, 3: 7, 4: 7}).to_numpy()
    # minutes_of_WN = (hypno_WN == 7).sum() / (60 * raw.info['sfreq'])

    # Keep only EEG channels and copy to avoid in-place modification
    raw_eeg = raw.copy().pick_types(eeg=True)

    # Remove flat channels
    bool_flat = raw_eeg.get_data().std(axis=1) == 0
    chan_flat = np.array(raw_eeg.ch_names)[bool_flat].tolist()
    if len(chan_flat):
        logger.warning("Removing flat channel(s): %s" % chan_flat)
    raw_eeg.drop_channels(chan_flat)

    # Remove suffix from channels: C4-M1 --> C4
    chan_nosuffix = [c.split('-')[0] for c in raw_eeg.ch_names]
    raw_eeg.rename_channels(dict(zip(raw_eeg.ch_names, chan_nosuffix)))
    # Rename P7/T5 --> P7
    chan_noslash = [c.split('/')[0] for c in raw_eeg.ch_names]
    raw_eeg.rename_channels(dict(zip(raw_eeg.ch_names, chan_noslash)))
    chan = raw_eeg.ch_names

    # Resample to 100 Hz and bandpass-filter
    raw_eeg.resample(100, verbose=False)
    raw_eeg.filter(l_freq, h_freq, verbose=False)

    # Extract data and sf
    data = raw_eeg.get_data() * 1e6  # Scale from Volts (MNE default) to uV
    sf = raw_eeg.info['sfreq']
    assert data.ndim == 2, 'data must be 2D (chan, times).'
    assert hypno.size == data.shape[1], 'Hypno must have same size as data.'

    # #########################################################################
    # 2) SPECTRAL POWER
    # #########################################################################

    print("  ..calculating spectral powers")

    # 2.1) 1Hz bins, N2 / N3 / REM
    # win_sec = 4 sec = 0.25 Hz freq resolution
    df_bp = yasa.bandpower(raw_eeg, hypno=hypno, bands=bands, win_sec=4,
                           include=(2, 3, 4))
    # Same for NREM / WN
    df_bp_NREM = yasa.bandpower(raw_eeg, hypno=hypno_NREM, bands=bands,
                                include=6)
    df_bp_WN = yasa.bandpower(raw_eeg, hypno=hypno_WN, bands=bands,
                              include=7)
    df_bp = df_bp.append(df_bp_NREM).append(df_bp_WN)
    df_bp.drop(columns=['TotalAbsPow', 'FreqRes', 'Relative'], inplace=True)
    df_bp = df_bp.add_prefix('bp_').reset_index()
    # Replace 2 --> N2
    df_bp['Stage'] = df_bp['Stage'].map(stage_mapping)
    # Assert that there are no negative values (see below issue on 1/f)
    assert not (df_bp._get_numeric_data() < 0).any().any()
    df_bp.columns = df_bp.columns.str.lower()

    # 2.2) Same but after adjusting for 1/F (VERY SLOW!)
    # This is based on the IRASA method described in Wen & Liu 2016.
    if do_1f:
        df_bp_1f = []

        for stage in [2, 3, 4, 6, 7]:
            if stage == 6:
                # Use hypno_NREM
                data_stage = data[:, hypno_NREM == stage]
            elif stage == 7:
                # Use hypno_WN
                data_stage = data[:, hypno_WN == stage]
            else:
                data_stage = data[:, hypno == stage]
            # Skip if stage is not present in data
            if data_stage.shape[-1] == 0:
                continue
            # Calculate aperiodic / oscillatory PSD + slope
            freqs, _, psd_osc, fit_params = yasa.irasa(
                data_stage, sf, ch_names=chan, band=(l_freq, h_freq),
                win_sec=4)
            # Make sure that we don't have any negative values in PSD
            # See https://github.com/raphaelvallat/yasa/issues/29
            psd_osc = psd_osc - psd_osc.min(axis=-1, keepdims=True)
            # Calculate bandpower
            bp = yasa.bandpower_from_psd(psd_osc, freqs, ch_names=chan,
                                         bands=bands)
            # Add 1/f slope to dataframe and sleep stage
            bp['1f_slope'] = np.abs(fit_params['Slope'].to_numpy())
            bp.insert(loc=0, column="Stage", value=stage_mapping[stage])
            df_bp_1f.append(bp)

        # Convert to a dataframe
        df_bp_1f = pd.concat(df_bp_1f)
        # Remove the TotalAbsPower column, incorrect because of negative values
        df_bp_1f.drop(columns=['TotalAbsPow', 'FreqRes', 'Relative'],
                      inplace=True)
        df_bp_1f.columns = [c if c in ['Stage', 'Chan', '1f_slope']
                            else 'bp_adj_' + c for c in df_bp_1f.columns]
        assert not (df_bp_1f._get_numeric_data() < 0).any().any()
        df_bp_1f.columns = df_bp_1f.columns.str.lower()

        # Merge with the main bandpower dataframe
        df_bp = df_bp.merge(df_bp_1f, how="outer")

    # #########################################################################
    # 3) SPINDLES DETECTION
    # #########################################################################

    print("  ..detecting sleep spindles")

    spindles_params.update(include=(2, 3))

    # Detect spindles in N2 and N3
    # Thresholds have to be tuned with visual scoring of a subset of data
    # https://raphaelvallat.com/yasa/build/html/generated/yasa.spindles_detect.html
    sp = yasa.spindles_detect(raw_eeg, hypno=hypno, **spindles_params)

    df_sp = sp.summary(grp_chan=True, grp_stage=True).reset_index()
    df_sp['Stage'] = df_sp['Stage'].map(stage_mapping)

    # Aggregate using the mean (adding NREM = N2 + N3)
    df_sp = sp.summary(grp_chan=True, grp_stage=True)
    df_sp_NREM = sp.summary(grp_chan=True).reset_index()
    df_sp_NREM['Stage'] = 6
    df_sp_NREM.set_index(['Stage', 'Channel'], inplace=True)
    density_NREM = df_sp_NREM['Count'] / minutes_of_NREM
    df_sp_NREM.insert(loc=1, column='Density',
                      value=density_NREM.to_numpy())

    df_sp = df_sp.append(df_sp_NREM)
    df_sp.columns = ['sp_' + c if c in ['Count', 'Density'] else
                     'sp_mean_' + c for c in df_sp.columns]

    # Prepare to export
    df_sp.reset_index(inplace=True)
    df_sp['Stage'] = df_sp['Stage'].map(stage_mapping)
    df_sp.columns = df_sp.columns.str.lower()
    df_sp.rename(columns={'channel': 'chan'}, inplace=True)

    # #########################################################################
    # 4) SLOW-WAVES DETECTION & SW-Sigma COUPLING
    # #########################################################################

    print("  ..detecting slow-waves")

    # Make sure we calculate coupling
    sw_params.update(coupling=True)

    # Detect slow-waves
    # Option 1: Using absolute thresholds
    # IMPORTANT: THRESHOLDS MUST BE ADJUSTED ACCORDING TO AGE!
    sw = yasa.sw_detect(raw_eeg, hypno=hypno, **sw_params)

    # Aggregate using the mean per channel x stage
    df_sw = sw.summary(grp_chan=True, grp_stage=True)
    # Add NREM
    df_sw_NREM = sw.summary(grp_chan=True).reset_index()
    df_sw_NREM['Stage'] = 6
    df_sw_NREM.set_index(['Stage', 'Channel'], inplace=True)
    density_NREM = df_sw_NREM['Count'] / minutes_of_NREM
    df_sw_NREM.insert(loc=1, column='Density',
                      value=density_NREM.to_numpy())
    df_sw = df_sw.append(df_sw_NREM)[['Count', 'Density', 'Duration',
                                      'PTP', 'Frequency', 'ndPAC']]
    df_sw.columns = ['sw_' + c if c in ['Count', 'Density'] else
                     'sw_mean_' + c for c in df_sw.columns]

    # Aggregate using the coefficient of variation
    # The CV is a normalized (unitless) standard deviation. Lower
    # values mean that slow-waves are more similar to each other.
    # We keep only spefific columns of interest. Not duration because it
    # is highly correlated with frequency (r=0.99).
    df_sw_cv = sw.summary(
        grp_chan=True, grp_stage=True, aggfunc=sp_stats.variation
    )[['PTP', 'Frequency', 'ndPAC']]

    # Add NREM
    df_sw_cv_NREM = sw.summary(
        grp_chan=True, grp_stage=False, aggfunc=sp_stats.variation
    )[['PTP', 'Frequency', 'ndPAC']].reset_index()
    df_sw_cv_NREM['Stage'] = 6
    df_sw_cv_NREM.set_index(['Stage', 'Channel'], inplace=True)
    df_sw_cv = df_sw_cv.append(df_sw_cv_NREM)
    df_sw_cv.columns = ['sw_cv_' + c for c in df_sw_cv.columns]

    # Combine the mean and CV into a single dataframe
    df_sw = df_sw.join(df_sw_cv).reset_index()
    df_sw['Stage'] = df_sw['Stage'].map(stage_mapping)
    df_sw.columns = df_sw.columns.str.lower()
    df_sw.rename(columns={'channel': 'chan'}, inplace=True)

    # #########################################################################
    # 5) ENTROPY & FRACTAL DIMENSION
    # #########################################################################

    print("  ..calculating entropy measures")

    # Filter data in the delta band and calculate envelope for CVE
    data_delta = mne.filter.filter_data(
        data, sfreq=sf, l_freq=0.5, h_freq=4, l_trans_bandwidth=0.2,
        h_trans_bandwidth=0.2, verbose=False)
    env_delta = np.abs(sp_sig.hilbert(data_delta))

    # Initialize dataframe
    idx_ent = pd.MultiIndex.from_product(
        [[2, 3, 4, 6, 7], chan], names=['stage', 'chan'])
    df_ent = pd.DataFrame(index=idx_ent)

    for stage in [2, 3, 4, 6, 7]:
        if stage == 6:
            # Use hypno_NREM
            data_stage = data[:, hypno_NREM == stage]
            data_stage_delta = data_delta[:, hypno_NREM == stage]
            env_stage_delta = env_delta[:, hypno_NREM == stage]
        elif stage == 7:
            # Use hypno_WN
            data_stage = data[:, hypno_WN == stage]
            data_stage_delta = data_delta[:, hypno_WN == stage]
            env_stage_delta = env_delta[:, hypno_WN == stage]
        else:
            data_stage = data[:, hypno == stage]
            data_stage_delta = data_delta[:, hypno == stage]
            env_stage_delta = env_delta[:, hypno == stage]
        # Skip if stage is not present in data
        if data_stage.shape[-1] == 0:
            continue

        # Entropy and fractal dimension (FD)
        # See review here: https://pubmed.ncbi.nlm.nih.gov/33286013/
        # These are calculated on the broadband signal.
        # - DFA not implemented because it is dependent on data length.
        # - Sample / app entropy not implemented because it is too slow to
        #   calculate.
        from numpy import apply_along_axis as aal
        df_ent.loc[stage, 'ent_svd'] = aal(
            ant.svd_entropy, axis=1, arr=data_stage, normalize=True)
        df_ent.loc[stage, 'ent_perm'] = aal(
            ant.perm_entropy, axis=1, arr=data_stage, normalize=True)
        df_ent.loc[stage, 'ent_spec'] = ant.spectral_entropy(
            data_stage, sf, method="welch", nperseg=(5 * int(sf)),
            normalize=True, axis=1)
        df_ent.loc[stage, 'ent_higuchi'] = aal(
            ant.higuchi_fd, axis=1, arr=data_stage)

        # We also add the coefficient of variation of the delta envelope
        # (CVE), a measure of "slow-wave stability".
        # See Diaz et al 2018, NeuroImage / Park et al 2021, Sci. Rep.
        # Lower values = more stable slow-waves (= more sinusoidal)
        denom = np.sqrt(4 / np.pi - 1)  # approx 0.5227
        cve = sp_stats.variation(env_stage_delta, axis=1) / denom
        df_ent.loc[stage, 'ent_cve_delta'] = cve

        # Other metrics of slow-wave (= delta) stability
        df_ent.loc[stage, 'ent_higuchi_delta'] = aal(
            ant.higuchi_fd, axis=1, arr=data_stage_delta)

    df_ent = df_ent.dropna(how="all").reset_index()
    df_ent['stage'] = df_ent['stage'].map(stage_mapping)

    # #########################################################################
    # 5) MERGE ALL DATAFRAMES
    # #########################################################################

    df = (df_bp
          .merge(df_sp, how='outer')
          .merge(df_sw, how='outer')
          .merge(df_ent, how='outer'))

    return df.set_index(['stage', 'chan'])
