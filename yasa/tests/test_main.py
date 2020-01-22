"""
Test the functions in the yasa/main.py file.
"""
import mne
import pytest
import unittest
import numpy as np
from itertools import product
from mne.filter import filter_data, resample
from yasa.hypno import hypno_str_to_int, hypno_upsample_to_data
from yasa.main import (get_sync_events, _index_to_events, get_bool_vector,
                       _merge_close, spindles_detect, spindles_detect_multi,
                       sw_detect, sw_detect_multi, rem_detect)

# Load data
data = np.loadtxt('notebooks/data_N2_spindles_15sec_200Hz.txt')
sf = 200
data_sigma = filter_data(data, sf, 12, 15, method='fir', verbose=0)

# Resample the data to 128 Hz
fac = 128 / sf
data_128 = resample(data, up=fac, down=1.0, npad='auto', axis=-1,
                    window='boxcar', n_jobs=1, pad='reflect_limited',
                    verbose=False)
sf_128 = 128

# Resample the data to 150 Hz
fac = 150 / sf
data_150 = resample(data, up=fac, down=1.0, npad='auto', axis=-1,
                    window='boxcar', n_jobs=1, pad='reflect_limited',
                    verbose=False)
sf_150 = 150

# Load an extract of N3 sleep without any spindle
data_n3 = np.loadtxt('notebooks/data_N3_no-spindles_30sec_100Hz.txt')
sf_n3 = 100

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
data_mne_single = data_mne.copy().pick_channels(['F3'])
hypno_mne = np.loadtxt('notebooks/sub-02_hypno_30s.txt', dtype=str)
hypno_mne = hypno_str_to_int(hypno_mne)
hypno_mne = hypno_upsample_to_data(hypno=hypno_mne, sf_hypno=(1 / 30),
                                   data=data_mne)


class TestStringMethods(unittest.TestCase):

    def test_index_to_events(self):
        """Test functions _index_to_events"""
        a = np.array([[3, 6], [8, 12], [14, 20]])
        good = [3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20]
        out = _index_to_events(a)
        np.testing.assert_equal(good, out)

    def test_get_bool_vector(self):
        """Test functions get_bool_vector"""
        # Numpy
        sp_params = spindles_detect(data, sf)
        out = get_bool_vector(data, sf, sp_params)
        assert out.size == data.size
        assert np.unique(out).size == 2
        # MNE
        # Single channel
        sp_params = spindles_detect_multi(data_mne_single)
        out = get_bool_vector(data_mne_single, detection=sp_params)
        assert out.size == max(data_mne_single.get_data().shape)
        assert np.unique(out).size == 2
        # Multi-channels
        sp_params = spindles_detect_multi(data_mne)
        out = get_bool_vector(data_mne, detection=sp_params)
        assert out.size == data_mne.get_data().size
        assert np.unique(out).size == 2

    def test_get_sync_events(self):
        """Test functions get_sync_events"""
        sw = sw_detect_multi(data_full, sf_full, chan_full)
        # Multi-channel with negative slow-wave peak
        df_sync = get_sync_events(data_full, sf, sw)
        assert df_sync['Channel'].nunique() == 3
        # Single-channel with positive slow-wave peak
        sw_c = sw[sw['Channel'] == sw.at[0, 'Channel']].iloc[:, :-2]
        df_sync = get_sync_events(data_full[0, :], sf_full, sw_c,
                                  center='PosPeak', time_before=0,
                                  time_after=2)
        assert df_sync.shape[1] == 3
        # MNE
        # Single channel
        sw = sw_detect_multi(data_mne_single, amp_neg=(20, 300),
                             amp_ptp=(60, 500))
        df_sync = get_sync_events(data_mne_single, detection=sw)
        assert df_sync['Channel'].nunique() == 1
        # Multi channel
        sw = sw_detect_multi(data_mne, amp_neg=(20, 300), amp_ptp=(60, 500))
        df_sync = get_sync_events(data_mne, detection=sw)
        assert df_sync['Channel'].nunique() == 6

    def test_merge_close(self):
        """Test functions _merge_close"""
        a = np.array([4, 5, 6, 7, 10, 11, 12, 13, 20, 21, 22, 100, 102])
        good = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                         17, 18, 19, 20, 21, 22, 100, 101, 102])
        # Events that are less than 100 ms apart (i.e. 10 points at 100 Hz sf)
        out = _merge_close(a, 100, 100)
        np.testing.assert_equal(good, out)

    def test_spindles_detect(self):
        """Test spindles_detect"""
        freq_sp = [(11, 16), [12, 14]]
        freq_broad = [(0.5, 30), [1, 25]]
        duration = [(0.3, 2.5), [0.5, 3]]
        min_distance = [None, 0, 500]

        prod_args = product(freq_sp, freq_broad, duration, min_distance)

        for i, (s, b, d, m) in enumerate(prod_args):
            spindles_detect(data, sf, freq_sp=s, duration=d,
                            freq_broad=b, min_distance=m)

        sp_params = spindles_detect(data, sf)
        print(sp_params.round(2))
        assert spindles_detect(data, sf).shape[0] == 2

        # Test with custom thresholds
        spindles_detect(data, sf, thresh={'rel_pow': 0.25})
        spindles_detect(data, sf, thresh={'rms': 1.25})
        spindles_detect(data, sf, thresh={'rel_pow': 0.25, 'corr': .60})

        # Test with disabled thresholds
        spindles_detect(data, sf, thresh={'rel_pow': None}, coupling=True)
        spindles_detect(data, sf, thresh={'corr': None})
        spindles_detect(data, sf, thresh={'rms': None})
        spindles_detect(data, sf, thresh={'rms': None, 'corr': None})
        spindles_detect(data, sf, thresh={'rms': None, 'rel_pow': None})
        spindles_detect(data, sf, thresh={'corr': None, 'rel_pow': None})

        # Test with downsampling is False
        spindles_detect(data, sf, downsample=False)

        # Power of 2 resampling
        spindles_detect(data_128, sf_128, coupling=True, freq_so=(0.5, 2))

        # Test with hypnogram
        spindles_detect(data_full[0, :], sf_full, hypno=hypno_full)

        # Hypnogram with sf = 200 Hz (require downsampling)
        spindles_detect(data, sf, hypno=np.ones(data.size))

        # Hypnogram with sf = 150
        with self.assertLogs('yasa', level='WARNING'):
            spindles_detect(data_150, sf_150, hypno=np.ones(data_150.size))

        # Now load other data
        with self.assertLogs('yasa', level='WARNING'):
            spindles_detect(data_n3, sf_n3)

        # Ensure that the two warnings are tested
        with self.assertLogs('yasa', level='WARNING'):
            sp = spindles_detect(data_n3, sf_n3, thresh={'corr': .95})
        assert sp is None

        # Test with wrong data amplitude (1)
        with self.assertLogs('yasa', level='ERROR'):
            sp = spindles_detect(data_n3 / 1e6, sf_n3)
        assert sp is None

        # Test with wrong data amplitude (2)
        with self.assertLogs('yasa', level='ERROR'):
            sp = spindles_detect(data_n3 * 1e6, sf_n3)
        assert sp is None

        # Test with a random array
        with self.assertLogs('yasa', level='ERROR'):
            sp = spindles_detect(np.random.random(size=1000), sf_n3)
        assert sp is None

        # Now we try with the isolation forest on the full recording
        with self.assertLogs('yasa', level='INFO'):
            sp = spindles_detect(data_full[1, :], sf_full,
                                 remove_outliers=True)
        assert sp.shape[0] > 100

        with pytest.raises(AssertionError):
            sp = spindles_detect(data_full, sf_full)

        # No values in hypno intersect with include
        with self.assertLogs('yasa', level='ERROR'):
            sp = spindles_detect(data, sf, include=2,
                                 hypno=np.zeros(data.size, dtype=int))
            assert sp is None

    def test_spindles_detect_multi(self):
        """Test spindles_detect_multi"""
        sp = spindles_detect_multi(data_full, sf_full, chan_full)
        sp_no_out = spindles_detect_multi(data_full, sf_full, chan_full,
                                          remove_outliers=True)
        sp_multi = spindles_detect_multi(data_full, sf_full, chan_full,
                                         multi_only=True)
        assert sp_multi.shape[0] < sp.shape[0]
        assert sp_no_out.shape[0] < sp.shape[0]
        bv = get_bool_vector(data_full, sf_full, sp)
        assert bv.shape[0] == len(chan_full)

        # Test with hypnogram
        spindles_detect_multi(data_full, sf_full, chan_full, hypno=hypno_full,
                              include=2, coupling=True)

        # Using a MNE raw object (and disabling one threshold)
        spindles_detect_multi(data_mne, thresh={'corr': None, 'rms': 3})
        spindles_detect_multi(data_mne, hypno=hypno_mne, include=2)

        # Now we replace one channel with no spindle / bad data
        data_full[1, :] = np.random.random(data_full.shape[1])

        # Test where data.shape[0] != len(chan)
        with pytest.raises(AssertionError):
            spindles_detect_multi(data_full, sf_full, chan_full[:-1])

        # Test with only bad channels
        data_full[0, :] = np.random.random(data_full.shape[1])
        data_full[2, :] = np.random.random(data_full.shape[1])
        with self.assertLogs('yasa', level='WARNING'):
            sp = spindles_detect_multi(data_full, sf_full, chan_full)
            assert sp is None

    def test_sw_detect(self):
        """Test function slow-wave detect
        """
        # Keep only Fz and during a N3 sleep period with (huge) slow-waves
        data_sw = data_full[1, 666000:672000].astype(np.float64)
        hypno_sw = hypno_full[666000:672000]

        # Parameters product testing
        freq_sw = [(0.3, 3.5), (0.5, 4)]
        dur_neg = [(0.3, 1.5), [0.1, 2]]
        dur_pos = [(0.3, 1.5), [0, 1]]
        amp_neg = [(40, 300), [40, None]]
        amp_pos = [(10, 150), (0, None)]
        amp_ptp = [(75, 400), [80, 300]]

        prod_args = product(freq_sw, dur_neg, dur_pos, amp_neg, amp_pos,
                            amp_ptp)

        for i, (f, dn, dp, an, ap, aptp) in enumerate(prod_args):
            print((f, dn, dp, an, ap, aptp))
            sw_detect(data_sw, sf_full, freq_sw=f, dur_neg=dn,
                      dur_pos=dp, amp_neg=an, amp_pos=ap,
                      amp_ptp=aptp)

        # With N3 hypnogram
        sw_detect(data_sw, sf_full, hypno=hypno_sw)

        # With 2D data
        sw_detect(data_sw[np.newaxis, ...], sf_full)

        # Downsampling with hypnogram
        data_sw_200 = resample(data_sw, up=2)
        sw_detect(data_sw_200, 200,
                  hypno=2 * np.ones(data_sw_200.shape, dtype=int),
                  include=2)

        # Downsampling without hypnogram
        data_sw_200 = resample(data_sw, up=2)
        sw_detect(data_sw_200, 200)

        # Non-integer sampling frequency
        data_sw_250 = resample(data_sw, up=2.5)
        sw_detect(data_sw_250, 250)

        # No values in hypno intersect with include
        with self.assertLogs('yasa', level='ERROR'):
            sw = sw_detect(data_sw, sf_full, include=3,
                           hypno=np.ones(data_sw.shape, dtype=int))
            assert sw is None

    def test_sw_detect_multi(self):
        """Test sw_detect_multi"""
        data_full = file_full.get('data')
        sw = sw_detect_multi(data_full, sf_full, chan_full)
        sw_no_out = sw_detect_multi(data_full, sf_full, chan_full,
                                    remove_outliers=True)
        assert sw_no_out.shape[0] < sw.shape[0]
        bv = get_bool_vector(data_full, sf_full, sw)
        assert bv.shape[0] == len(chan_full)

        # Test with hypnogram
        sw_detect_multi(data_full, sf_full, chan_full,
                        hypno=hypno_full, include=3)

        # Using a MNE raw object
        sw_detect_multi(data_mne)
        sw_detect_multi(data_mne, hypno=hypno_mne, include=(2, 3))

        # Now we replace one channel with no slow-wave / bad data
        data_full[1, :] = np.random.random(data_full.shape[1])

        # Test where data.shape[0] != len(chan)
        with pytest.raises(AssertionError):
            sw_detect_multi(data_full, sf_full, chan_full[:-1])

        # Test with only bad channels
        data_full[0, :] = np.random.random(data_full.shape[1])
        data_full[2, :] = np.random.random(data_full.shape[1])
        with self.assertLogs('yasa', level='WARNING'):
            sp = sw_detect_multi(data_full, sf_full, chan_full)
            assert sp is None

    def test_rem_detect(self):
        """Test function REM detect
        """
        file_rem = np.load('notebooks/data_EOGs_REM_256Hz.npz')
        data_rem = file_rem['data']
        loc, roc = data_rem[0, :], data_rem[1, :]
        sf_rem = file_rem['sf']
        # chan_rem = file_rem['chan']
        hypno_rem = 4 * np.ones_like(loc)

        # Parameters product testing
        freq_rem = [(0.5, 5), (0.3, 8)]
        duration = [(0.3, 1.5), [0.5, 1]]
        amplitude = [(50, 200), [60, 300]]
        downsample = [True, False]
        hypno = [hypno_rem, None]
        prod_args = product(freq_rem, duration, amplitude, downsample, hypno)

        for i, (f, dr, am, ds, h) in enumerate(prod_args):
            rem_detect(loc, roc, sf_rem, hypno=h, freq_rem=f, duration=dr,
                       amplitude=am, downsample=ds)

        # With isolation forest
        df_rem = rem_detect(loc, roc, sf)
        df_rem2 = rem_detect(loc, roc, sf, remove_outliers=True)
        assert df_rem.shape[0] > df_rem2.shape[0]
        assert get_bool_vector(loc, sf, df_rem).size == loc.size

        # With REM hypnogram
        hypno_rem = 4 * np.ones_like(loc)
        df_rem = rem_detect(loc, roc, sf_full, hypno=hypno_rem)
        hypno_rem = np.r_[np.ones(int(loc.size / 2)),
                          4 * np.ones(int(loc.size / 2))]
        df_rem2 = rem_detect(loc, roc, sf_full, hypno=hypno_rem)
        assert df_rem.shape[0] > df_rem2.shape[0]

        # Test with wrong data amplitude on ROC
        with self.assertLogs('yasa', level='ERROR'):
            rd = rem_detect(loc * 1e-8, roc, sf)
        assert rd is None

        # Test with wrong data amplitude on LOC
        with self.assertLogs('yasa', level='ERROR'):
            rd = rem_detect(loc, roc * 1e8, sf)
        assert rd is None

        # Hypnogram with sf = 150
        with self.assertLogs('yasa', level='WARNING'):
            rem_detect(loc, roc * 1e8, sf=150)  # Fake sampling frequency

        # No values in hypno intersect with include
        with self.assertLogs('yasa', level='ERROR'):
            sp = rem_detect(loc, roc, sf_full, hypno=hypno_rem, include=5)
            assert sp is None
