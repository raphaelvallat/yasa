"""
Test the functions in the yasa/main.py file.
"""
import pytest
import unittest
import numpy as np
from itertools import product
from mne.filter import filter_data, resample
from yasa.main import (_corr, _covar, _rms, moving_transform, stft_power,
                       _index_to_events, get_bool_vector, trimbothstd,
                       _merge_close, spindles_detect, spindles_detect_multi)

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


data_n3 = np.loadtxt('notebooks/data_N3_no-spindles_30sec_100Hz.txt')
sf_n3 = 100

file_full = np.load('notebooks/data_full_6hrs_100Hz_Cz+Fz+Pz.npz')
data_full = file_full.get('data')
chan_full = file_full.get('chan')
sf_full = 100

# Load the hypnogram
hypno_full = np.load('notebooks/data_full_6hrs_100Hz_hypno.npz').get('hypno')


class TestStringMethods(unittest.TestCase):

    def test_numba(self):
        """Test numba functions
        """
        x = np.asarray([4, 5, 7, 8, 5, 6], dtype=np.float64)
        y = np.asarray([1, 5, 4, 6, 8, 5], dtype=np.float64)

        np.testing.assert_almost_equal(_corr(x, y), np.corrcoef(x, y)[0][1])
        assert _covar(x, y) == np.cov(x, y)[0][1]
        assert _rms(x) == np.sqrt(np.mean(np.square(x)))

    def test_index_to_events(self):
        """Test functions _index_to_events"""
        a = np.array([[3, 6], [8, 12], [14, 20]])
        good = [3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20]
        out = _index_to_events(a)
        np.testing.assert_equal(good, out)

    def test_get_bool_vector(self):
        """Test functions get_bool_vector"""
        sp_params = spindles_detect(data, sf)
        out = get_bool_vector(data, sf, sp_params)
        assert out.size == data.size
        assert np.unique(out).size == 2

    def test_merge_close(self):
        """Test functions _merge_close"""
        a = np.array([4, 5, 6, 7, 10, 11, 12, 13, 20, 21, 22, 100, 102])
        good = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                         17, 18, 19, 20, 21, 22, 100, 101, 102])
        # Events that are less than 100 ms apart (i.e. 10 points at 100 Hz sf)
        out = _merge_close(a, 100, 100)
        np.testing.assert_equal(good, out)

    def test_moving_transform(self):
        """Test moving_transform"""
        method = ['corr', 'covar', 'rms']
        interp = [False, True]
        win = [.3, .5]
        step = [0, .1, .3, .5]

        prod_args = product(win, step, method, interp)

        for i, (w, s, m, i) in enumerate(prod_args):
            moving_transform(data, data_sigma, sf, w, s, m, i)

        t, out = moving_transform(data, None, sf, w, s, 'rms', True)
        assert t.size == out.size
        assert out.size == data.size

    def test_spindles_detect(self):
        """Test spindles_detect"""
        freq_sp = [(11, 16), [12, 14], (11, 15)]
        freq_broad = [(0.5, 30), [1, 25]]
        duration = [(0.3, 2.5), [0.5, 3]]
        min_distance = [None, 0, 300, 500]

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

        # Test with downsampling is False
        spindles_detect(data, sf, downsample=False)

        # Test with hypnogram
        spindles_detect(data_full[0, :], sf_full, hypno=hypno_full)

        # Hypnogram with sf = 200 Hz (require downsampling)
        spindles_detect(data, sf, hypno=np.ones(data.size))

        # Hypnogram with sf = 128 (downsampling is not possible)
        with self.assertLogs('yasa', level='ERROR'):
            spindles_detect(data_128, sf_128, hypno=np.ones(data_128.size))

        # Hypnogram with only one unique value
        with self.assertLogs('yasa', level='ERROR'):
            sp = spindles_detect(data, sf, hypno=np.zeros(data.size))
        assert 'Stage' not in sp.keys()

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
        spindles_detect_multi(data_full, sf_full, chan_full, hypno=hypno_full)

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

    def test_trimbothstd(self):
        """Test function trimbothstd
        """
        x = [4, 5, 7, 0, 18, 6, 7, 8, 9, 10]
        assert trimbothstd(x) < np.std(x, ddof=1)
