"""Test the functions in the yasa/fetchers.py file."""

import unittest

from matplotlib.pyplot import Axes
from pandas import Series
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.metrics import accuracy_score

from yasa import EpochByEpochAgreement, Hypnogram

# For EpochByEpochAgreement tests
hyp1 = Hypnogram(["W", "W", "N1", "N1", "N2"], scorer="Scorer1")
hyp2 = Hypnogram(["W", "W", "N1", "N1", "N1"], scorer="Scorer2")
hyp3 = Hypnogram(["W", "W", "N1", "N2", "N1", "N3"], scorer="Scorer3")

ebe1 = EpochByEpochAgreement([hyp1], [hyp2])
ebe2 = EpochByEpochAgreement([hyp1, hyp1], [hyp2, hyp2])

class TestEpochByEpochAgreement(unittest.TestCase):
    """Test the EpochByEpochAgreement class."""

    def test_init(self):
        """Test initialization or creation of an instance."""
        # Test that data attribute is compiled correctly
        assert len(ebe1.data) == hyp1.n_epochs == hyp2.n_epochs
        assert ebe1.data.shape[1] == 2
        assert ebe1.data.columns.tolist() == [hyp1.scorer, hyp2.scorer]
        # Test scorer attributes
        assert ebe1.ref_scorer == hyp1.scorer
        assert ebe1.obs_scorer == hyp2.scorer
        # Test handling of incorrect arguments
        with self.assertRaisesRegex(AssertionError, "must be an iterable"):
            EpochByEpochAgreement(hyp1, hyp2)
        with self.assertRaisesRegex(AssertionError, "must have unique scorers"):
            EpochByEpochAgreement([hyp1], [hyp1])
        with self.assertRaisesRegex(AssertionError, "must have the same number of hypnograms"):
            EpochByEpochAgreement([hyp1], [hyp2, hyp2])
        with self.assertRaisesRegex(AssertionError, "must only contain YASA hypnograms"):
            EpochByEpochAgreement([hyp1.hypno], [hyp2.hypno])
        with self.assertRaisesRegex(AssertionError, "must have the same n_epochs"):
            EpochByEpochAgreement([hyp1], [hyp3])

    def test_confusion_matrix(self):
        """Test generation of confusion matrix."""
        # Test `sleep_id` parameter
        with self.assertRaisesRegex(AssertionError, "must be None or a valid sleep ID"):
            ebe2.get_confusion_matrix(sleep_id="FOO")
        assert_frame_equal(ebe2.get_confusion_matrix(), ebe2.get_confusion_matrix(sleep_id=None))
        assert ebe2.get_confusion_matrix().size > ebe2.get_confusion_matrix(sleep_id=1).size
        assert ebe2.get_confusion_matrix(sleep_id=1).sum().sum() == hyp1.n_epochs
        assert ebe2.get_confusion_matrix(sleep_id=2).sum().sum() == hyp2.n_epochs
        assert ebe2.get_confusion_matrix().sum().sum() == hyp1.n_epochs + hyp2.n_epochs
        # Test `agg_func` parameter
        with self.assertRaisesRegex(AssertionError, "must be None or a str"):
            ebe2.get_confusion_matrix(agg_func=999)
        assert_frame_equal(ebe2.get_confusion_matrix(), ebe2.get_confusion_matrix(agg_func=None))
        assert ebe2.get_confusion_matrix(agg_func="sum").to_numpy().dtype == "int64"
        assert ebe2.get_confusion_matrix(agg_func="mean").to_numpy().dtype == "float64"
        # Test **kwargs for sklearn.metrics.confusion_matrix
        assert ebe2.get_confusion_matrix(sleep_id=1, normalize="all").sum().sum() == 1

    def test_agreement(self):
        """Test get_agreement() method."""
        agr = ebe1.get_agreement()
        # Test default arguments
        assert_series_equal(agr, ebe1.get_agreement(sample_weight=None, metrics=None))
        assert agr.at["accuracy"] == 0.8
        assert agr.at["recall"] == 0.8
        # Test `sample_weight` argument
        weights = Series([0.25, 0.25, 0.25, 0.25, 1], index=ebe1.data.index)
        assert ebe1.get_agreement(sample_weight=weights).at["accuracy"] == 0.5
        # Test `metrics` argument
        metrics = {
            "acc1": lambda a, b, c: accuracy_score(a, b, sample_weight=c),
            "acc2": lambda a, b, c: accuracy_score(a, b, sample_weight=c),
        }
        assert ebe1.get_agreement(metrics=metrics).eq(0.8).all()
        assert ebe1.get_agreement(metrics=metrics).index.tolist() == list(metrics.keys())

    def test_agreement_bystage(self):
        """Test getting agreement metrics for each stage."""
        b = ebe1.get_agreement_bystage()
        b0 = ebe1.get_agreement_bystage(beta=0)
        b1 = ebe1.get_agreement_bystage(beta=1)
        assert b.at["N1", "fbeta"] == 0.8
        assert b0["fbeta"].equals(b1["precision"])
        assert not b0["fbeta"].equals(b1["recall"])
        assert not b0.equals(b1)
        assert_frame_equal(b, b1)
        assert b.columns.tolist() == ["fbeta", "precision", "recall", "support"]

    def test_sleep_stats(self):
        """Test get_sleep_stats() method."""
        ss = ebe1.get_sleep_stats()
        assert ss.index.name == "scorer"
        assert ss.index.tolist() == [hyp1.scorer, hyp2.scorer]
        assert ss.index.tolist() == [ebe1.ref_scorer, ebe1.obs_scorer]
        s1 = ss.loc[hyp1.scorer]
        s2 = Series(hyp1.sleep_statistics(), name=hyp1.scorer)
        assert_series_equal(s1, s2)

    def test_hypnogram_overlay(self):
        """Test plot_hypnograms() method."""
        ax = ebe1.plot_hypnograms(sleep_id=1)
        assert isinstance(ax, Axes)
        kwargs1 = {"highlight": None, "color": "green"}
        kwargs2 = {"alpha": 0.8, "ls": "dashed", "label": "CUSTOM"}
        ebe1.plot_hypnograms(legend=False, ax=ax, ref_kwargs=kwargs1, obs_kwargs=kwargs2)
