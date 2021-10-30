"""Test the functions in yasa/spectral.py."""
import mne
import pytest
import unittest
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from mne.filter import filter_data
from yasa.hypno import hypno_str_to_int, hypno_upsample_to_data
from yasa.detection import spindles_detect, sw_detect, rem_detect, art_detect

##############################################################################
# DATA LOADING
##############################################################################

# For all data, the sampling frequency is always 100 Hz
sf = 100

# 1) Single channel, we take one every other point to keep a sf of 100 Hz
data = np.loadtxt('notebooks/data_N2_spindles_15sec_200Hz.txt')[::2]
data_sigma = filter_data(data, sf, 12, 15, method='fir', verbose=0)

# Load an extract of N3 sleep without any spindle
data_n3 = np.loadtxt('notebooks/data_N3_no-spindles_30sec_100Hz.txt')

# 2) Multi-channel
# Load a full recording and its hypnogram
data_full = np.load('notebooks/data_full_6hrs_100Hz_Cz+Fz+Pz.npz').get('data')
chan_full = np.load('notebooks/data_full_6hrs_100Hz_Cz+Fz+Pz.npz').get('chan')
hypno_full = np.load('notebooks/data_full_6hrs_100Hz_hypno.npz').get('hypno')

# Let's add a channel with bad data amplitude
chan_full = np.append(chan_full, 'Bad')  # ['Cz', 'Fz', 'Pz', 'Bad']
data_full = np.vstack((data_full, data_full[-1, :] * 1e8))

# Keep only Fz and during a N3 sleep period with (huge) slow-waves
data_sw = data_full[1, 666000:672000].astype(np.float64)
hypno_sw = hypno_full[666000:672000]

# MNE Raw
data_mne = mne.io.read_raw_fif('notebooks/sub-02_mne_raw.fif', preload=True, verbose=0)
data_mne.pick_types(eeg=True)
data_mne_single = data_mne.copy().pick_channels(['F3'])
hypno_mne = np.loadtxt('notebooks/sub-02_hypno_30s.txt', dtype=str)
hypno_mne = hypno_str_to_int(hypno_mne)
hypno_mne = hypno_upsample_to_data(hypno=hypno_mne, sf_hypno=(1 / 30), data=data_mne)


class TestDetection(unittest.TestCase):

    def test_check_data_hypno(self):
        """Test preprocessing of data and hypno."""
        pass

    def test_spindles_detect(self):
        """Test spindles_detect"""
        #######################################################################
        # SINGLE CHANNEL
        #######################################################################
        freq_sp = [(11, 16), [12, 14]]
        freq_broad = [(0.5, 30), [1, 25]]
        duration = [(0.3, 2.5), [0.5, 3]]
        min_distance = [None, 0, 500]
        prod_args = product(freq_sp, freq_broad, duration, min_distance)

        for i, (s, b, d, m) in enumerate(prod_args):
            spindles_detect(data, sf, freq_sp=s, duration=d, freq_broad=b, min_distance=m)

        sp = spindles_detect(data, sf, verbose=True)
        assert sp.summary().shape[0] == 2
        sp.get_mask()
        sp.get_sync_events()
        sp.get_sync_events(time_before=10)  # Invalid time window
        sp.plot_average(ci=None, filt=(None, 30))  # Skip bootstrapping
        np.testing.assert_array_equal(np.squeeze(sp._data), data)
        assert sp._sf == sf
        sp.summary(grp_chan=True, grp_stage=True, aggfunc='median', sort=False)

        # Test with custom thresholds
        spindles_detect(data, sf, thresh={'rel_pow': 0.25})
        spindles_detect(data, sf, thresh={'rms': 1.25})
        spindles_detect(data, sf, thresh={'rel_pow': 0.25, 'corr': .60})

        # Test with disabled thresholds
        spindles_detect(data, sf, thresh={'rel_pow': None})
        spindles_detect(data, sf, thresh={'corr': None}, verbose='debug')
        spindles_detect(data, sf, thresh={'rms': None})
        spindles_detect(data, sf, thresh={'rms': None, 'corr': None})
        spindles_detect(data, sf, thresh={'rms': None, 'rel_pow': None})
        spindles_detect(data, sf, thresh={'corr': None, 'rel_pow': None})

        # Test with hypnogram
        spindles_detect(data, sf, hypno=np.ones(data.size))

        # Single channel with Isolation Forest + hypnogram
        sp = spindles_detect(data_full[1, :], sf, hypno=hypno_full, remove_outliers=True)

        # Calculate the coincidence matrix with only one channel
        with pytest.raises(ValueError):
            sp.get_coincidence_matrix()

        with self.assertLogs('yasa', level='WARNING'):
            spindles_detect(data_n3, sf)
        # assert sp is None --> Fails?

        # Ensure that the two warnings are tested
        with self.assertLogs('yasa', level='WARNING'):
            sp = spindles_detect(data_n3, sf, thresh={'corr': .95})
        assert sp is None

        # Test with wrong data amplitude (1)
        with self.assertLogs('yasa', level='ERROR'):
            sp = spindles_detect(data_n3 / 1e6, sf)
        assert sp is None

        # Test with wrong data amplitude (2)
        with self.assertLogs('yasa', level='ERROR'):
            sp = spindles_detect(data_n3 * 1e6, sf)
        assert sp is None

        # Test with a random array
        with self.assertLogs('yasa', level='ERROR'):
            np.random.seed(123)
            sp = spindles_detect(np.random.random(size=1000), sf)
        assert sp is None

        # No values in hypno intersect with include
        with pytest.raises(AssertionError):
            sp = spindles_detect(data, sf, include=2, hypno=np.zeros(data.size, dtype=int))

        #######################################################################
        # MULTI CHANNEL
        #######################################################################

        sp = spindles_detect(data_full, sf, chan_full)
        sp.get_mask()
        sp.get_sync_events(filt=(12, 15))
        sp.summary()
        sp.summary(grp_chan=True)
        sp.plot_average(ci=None)
        sp.get_coincidence_matrix()
        sp.get_coincidence_matrix(scaled=False)
        sp.plot_detection()
        assert sp._data.shape == sp._data_filt.shape
        np.testing.assert_array_equal(sp._data, data_full)
        assert sp._sf == sf
        sp_no_out = spindles_detect(data_full, sf, chan_full,
                                    remove_outliers=True)
        sp_multi = spindles_detect(data_full, sf, chan_full,
                                   multi_only=True)
        assert sp_multi.summary().shape[0] < sp.summary().shape[0]
        assert sp_no_out.summary().shape[0] < sp.summary().shape[0]

        # Test with hypnogram
        sp = spindles_detect(data_full, sf, hypno=hypno_full, include=2)
        sp.summary(grp_chan=False, grp_stage=False)
        sp.summary(grp_chan=False, grp_stage=True, aggfunc='median')
        sp.summary(grp_chan=True, grp_stage=False)
        sp.summary(grp_chan=True, grp_stage=True, sort=False)
        sp.plot_average(ci=None)
        sp.plot_average(hue="Stage", ci=None)
        sp.plot_detection()

        # Using a MNE raw object (and disabling one threshold)
        spindles_detect(data_mne, thresh={'corr': None, 'rms': 3})
        spindles_detect(data_mne, hypno=hypno_mne, include=2, verbose=True)
        plt.close('all')

    def test_sw_detect(self):
        """Test function slow-wave detect
        """
        # Parameters product testing
        freq_sw = [(0.3, 3.5), (0.5, 4)]
        dur_neg = [(0.3, 1.5), [0.1, 2]]
        dur_pos = [(0.3, 1.5), [0, 1]]
        amp_neg = [(40, 300), [40, None]]
        amp_pos = [(10, 150), (0, None)]
        amp_ptp = [(75, 400), [80, 300]]
        prod_args = product(freq_sw, dur_neg, dur_pos, amp_neg, amp_pos, amp_ptp)

        for i, (f, dn, dp, an, ap, aptp) in enumerate(prod_args):
            # print((f, dn, dp, an, ap, aptp))
            sw_detect(
                data_sw, sf, freq_sw=f, dur_neg=dn, dur_pos=dp, amp_neg=an, amp_pos=ap,
                amp_ptp=aptp)

        # With N3 hypnogram
        sw = sw_detect(data_sw, sf, hypno=hypno_sw, coupling=True)
        sw.summary()
        sw.get_mask()
        sw.get_sync_events()
        sw.plot_average(ci=None)
        sw.plot_detection()
        np.testing.assert_array_equal(np.squeeze(sw._data), data_sw)
        np.testing.assert_array_equal(sw._hypno, hypno_sw)
        assert sw._sf == sf

        # Test with wrong data amplitude
        with self.assertLogs('yasa', level='ERROR'):
            sw = sw_detect(data_sw * 0, sf)  # All channels are flat
        assert sw is None

        # With 2D data
        sw_detect(data_sw[np.newaxis, ...], sf, verbose='INFO')

        # No values in hypno intersect with include
        with pytest.raises(AssertionError):
            sw = sw_detect(data_sw, sf, include=3, hypno=np.ones(data_sw.shape, dtype=int))

        #######################################################################
        # MULTI CHANNEL
        #######################################################################

        sw = sw_detect(data_full, sf, chan_full)
        sw.get_mask()
        sw.get_sync_events()
        sw.plot_average(ci=None)
        sw.plot_detection()
        sw.get_coincidence_matrix()
        sw.get_coincidence_matrix(scaled=False)
        # Test with outlier removal. There should be fewer events.
        sw_no_out = sw_detect(data_full, sf, chan_full, remove_outliers=True)
        assert sw_no_out._events.shape[0] < sw._events.shape[0]

        # Test with hypnogram
        sw = sw_detect(data_full, sf, chan_full, hypno=hypno_full, coupling=True)
        sw.summary(grp_chan=False, grp_stage=False)
        sw.summary(grp_chan=False, grp_stage=True, aggfunc='median')
        sw.summary(grp_chan=True, grp_stage=False)
        sw.summary(grp_chan=True, grp_stage=True, sort=False)
        sw.plot_average(ci=None)
        sw.plot_average(hue="Stage", ci=None)
        sw.plot_detection()
        # Check coupling
        sw_sum = sw.summary()
        assert "ndPAC" in sw_sum.columns
        # There should be some zero in the ndPAC (full dataframe)
        assert sw._events[sw._events['ndPAC'] == 0].shape[0] > 0
        # Coinciding spindles and masking
        sp = spindles_detect(data_full, sf, chan_full, hypno=hypno_full)
        sw.find_cooccurring_spindles(sp.summary())
        sw_sum = sw.summary()
        assert "CooccurringSpindle" in sw_sum.columns
        assert "DistanceSpindleToSW" in sw_sum.columns
        sw_sum_masked = sw.summary(grp_chan=True, grp_stage=False,
                                   mask=sw._events['CooccurringSpindle'])
        assert sw_sum_masked.shape[0] < sw_sum.shape[0]

        # Test with different coupling params
        sw_detect(data_full, sf, chan_full, hypno=hypno_full, coupling=True,
                  coupling_params={"freq_sp": (12, 16), "time": 2, "p": None})

        # Using a MNE raw object
        sw_detect(data_mne)
        sw_detect(data_mne, hypno=hypno_mne, include=3)
        plt.close('all')

    def test_rem_detect(self):
        """Test function REM detect
        """
        file_rem = np.load('notebooks/data_EOGs_REM_256Hz.npz')
        data_rem = file_rem['data']
        loc, roc = data_rem[0, :], data_rem[1, :]
        sf_rem = file_rem['sf']
        hypno_rem = 4 * np.ones_like(loc)

        # Parameters product testing
        freq_rem = [(0.5, 5), (0.3, 8)]
        duration = [(0.3, 1.5), [0.5, 1]]
        amplitude = [(50, 200), [60, 300]]
        hypno = [hypno_rem, None]
        prod_args = product(freq_rem, duration, amplitude, hypno)

        for i, (f, dr, am, h) in enumerate(prod_args):
            rem_detect(loc, roc, sf_rem, hypno=h, freq_rem=f, duration=dr, amplitude=am)

        # With isolation forest
        rem = rem_detect(loc, roc, sf, verbose='info')
        rem2 = rem_detect(loc, roc, sf, remove_outliers=True)
        assert rem.summary().shape[0] > rem2.summary().shape[0]
        rem.summary()
        rem.get_mask()
        rem.get_sync_events()
        rem.plot_average(ci=None)
        rem.plot_average(filt=(0.5, 5), ci=None)

        # With REM hypnogram
        hypno_rem = 4 * np.ones_like(loc)
        rem = rem_detect(loc, roc, sf, hypno=hypno_rem)
        hypno_rem = np.r_[np.ones(int(loc.size / 2)), 4 * np.ones(int(loc.size / 2))]
        rem2 = rem_detect(loc, roc, sf, hypno=hypno_rem)
        assert rem.summary().shape[0] > rem2.summary().shape[0]
        rem2.summary(grp_stage=True, aggfunc='median')

        # Test with wrong data amplitude on ROC
        with self.assertLogs('yasa', level='ERROR'):
            rem = rem_detect(loc * 1e-8, roc, sf)
        assert rem is None

        # Test with wrong data amplitude on LOC
        with self.assertLogs('yasa', level='ERROR'):
            rem = rem_detect(loc, roc * 1e8, sf)
        assert rem is None

        # No values in hypno intersect with include
        with pytest.raises(AssertionError):
            rem_detect(loc, roc, sf, hypno=hypno_rem, include=5)

    def test_art_detect(self):
        """Test function art_detect
        """
        file_9 = np.load('notebooks/data_full_6hrs_100Hz_9channels.npz')
        data_9 = file_9.get('data')
        hypno_9 = np.load('notebooks/data_full_6hrs_100Hz_hypno.npz').get('hypno')  # noqa
        # For the sake of the example, let's add some flat data at the end
        data_9 = np.concatenate((data_9, np.zeros((data_9.shape[0], 20000))), axis=1)
        hypno_9 = np.concatenate((hypno_9, np.zeros(20000)))

        # Start different combinations
        art_detect(data_9, sf=100, window=10, method='covar', threshold=3)
        art_detect(data_9, sf=100, window=6, hypno=hypno_9, include=(2, 3),
                   method='covar', threshold=3)
        art_detect(data_9, sf=100, window=5, method='std', threshold=2)
        art_detect(data_9, sf=100, window=5, hypno=hypno_9, method='std',
                   include=(0, 1, 2, 3, 4, 5, 6), threshold=2)
        art_detect(data_9, sf=100, window=5., hypno=hypno_9, method='std',
                   include=(0, 1, 2, 3, 4, 5, 6), threshold=10)
        # Single channel
        art_detect(data_9[0], 100, window=10, method='covar')
        art_detect(data_9[0], 100, window=5, method='std', verbose=True)

        # Not enough epochs for stage
        hypno_9[:100] = 6
        art_detect(data_9, sf, window=5., hypno=hypno_9, include=6,
                   method='std', threshold=3, n_chan_reject=5)

        # With a flat channel
        data_with_flat = np.vstack((data_9, np.zeros(data_9.shape[-1])))
        art_detect(data_with_flat, sf, method='std', n_chan_reject=5)

        # Using a MNE raw object
        art_detect(data_mne, window=10., hypno=hypno_mne, method='covar', verbose='INFO')

        with pytest.raises(AssertionError):
            # None of include in hypno
            art_detect(data_mne, window=10., hypno=hypno_mne, include=[7, 8])
