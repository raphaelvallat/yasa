"""Tests for yasa/evaluation.py — EpochByEpochAgreement and SleepStatsAgreement."""

import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.stats as sps
import sklearn.metrics as skm
from matplotlib.colors import to_rgb
from sklearn.exceptions import UndefinedMetricWarning

from yasa.evaluation import EpochByEpochAgreement, SleepStatsAgreement
from yasa.hypno import Hypnogram, simulate_hypnogram

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SESSIONS = 5
REF_SCORER = "Human"
OBS_SCORER = "YASA"
# Default `confidence` of SleepStatsAgreement, as printed in the report column names
PCT = 95

_FMT_CI = r"\d+\.\d \(\d+\.\d\) \[\d+\.\d, \d+\.\d\]"
_FMT_NOCI = r"\d+\.\d \(\d+\.\d\)"

BOOT_METHODS = ["BCa", "basic", "percentile"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simulate_pairs(n, tib=90, seed_offset=0):
    """Simulate `n` pairs of (reference, observed) hypnograms."""
    ref = [simulate_hypnogram(tib=tib, scorer=REF_SCORER, seed=i + seed_offset) for i in range(n)]
    obs = [h.simulate_similar(scorer=OBS_SCORER, seed=i + seed_offset) for i, h in enumerate(ref)]
    return ref, obs


def _summary_quiet(obj, **kwargs):
    """Call EpochByEpochAgreement.summary() with the BCa small-n warning silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return obj.summary(**kwargs)


def _log_slope(obj):
    """Euser LoA slope per statistic (NaN for statistics that are not log-transformed)."""
    return obj.summary(ci_method=None)["loa_log_slope"]["center"]


def _label(rpt, stat):
    """Report index label ("STAT (unit)") for a sleep statistic."""
    return rpt.index[rpt.index.str.startswith(f"{stat} (")][0]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def hyps():
    """Five simulated (reference, observed) hypnogram pairs."""
    return _simulate_pairs(N_SESSIONS)


@pytest.fixture(scope="module")
def ref_hyps(hyps):
    return hyps[0]


@pytest.fixture(scope="module")
def obs_hyps(hyps):
    return hyps[1]


@pytest.fixture(scope="module")
def ebe(ref_hyps, obs_hyps):
    return EpochByEpochAgreement(ref_hyps, obs_hyps)


@pytest.fixture
def fresh_ebe(ref_hyps, obs_hyps):
    """A new EpochByEpochAgreement, for tests that overwrite or need no cached scores."""
    return EpochByEpochAgreement(ref_hyps, obs_hyps)


@pytest.fixture(scope="module")
def ebe_single(ref_hyps, obs_hyps):
    """Single-night variant (via Hypnogram.evaluate)."""
    return ref_hyps[0].evaluate(obs_hyps[0])


@pytest.fixture
def ebe_missing(ref_hyps, obs_hyps):
    """Two sessions where session 1 has no N1 and no N3 in the reference hypnogram.

    The observed scorer assigns a few epochs of each -> recall (and fbeta) undefined for those
    stages. Function-scoped because get_agreement_bystage() caches its scores on the object, so
    that a test calling it with `zero_division=0` does not affect the others.
    """
    ref = Hypnogram(["WAKE"] * 10 + ["N2"] * 20 + ["REM"] * 10, scorer=REF_SCORER)
    obs = Hypnogram(
        ["WAKE"] * 8 + ["N1"] * 2 + ["N2"] * 18 + ["N3"] * 2 + ["REM"] * 10, scorer=OBS_SCORER
    )
    return EpochByEpochAgreement([ref, ref_hyps[0]], [obs, obs_hyps[0]])


@pytest.fixture(scope="module")
def ebe_large():
    """A 20-session object, the threshold at which the BCa small-n warning stops being emitted.

    `tib` must be long enough for every stage to occur, otherwise `simulate_similar` cannot build a
    transition matrix.
    """
    return EpochByEpochAgreement(*_simulate_pairs(20, tib=480))


@pytest.fixture(scope="module")
def sstats(ebe):
    return ebe.get_sleep_stats()


@pytest.fixture(scope="module")
def ref_stats(sstats):
    return sstats.loc[REF_SCORER]


@pytest.fixture(scope="module")
def obs_stats(sstats):
    return sstats.loc[OBS_SCORER]


@pytest.fixture(scope="module")
def ssa(ref_stats, obs_stats):
    return SleepStatsAgreement(ref_stats, obs_stats, ref_scorer=REF_SCORER, obs_scorer=OBS_SCORER)


@pytest.fixture(scope="module")
def ssa_log(ref_stats, obs_stats):
    return SleepStatsAgreement(
        ref_stats, obs_stats, ref_scorer=REF_SCORER, obs_scorer=OBS_SCORER, log_transform=True
    )


@pytest.fixture(scope="module")
def log_stats(ssa_log):
    """Stats that can be log-transformed (no zero value in either scorer)."""
    return _log_slope(ssa_log).dropna().index.tolist()


@pytest.fixture(scope="module")
def stats_large():
    """Sleep stats of 15 sessions, to avoid degenerate all-zero stats that make BCa fail with N=5.

    Used for ci_method="auto" and ci_method="boot" tests.
    """
    sstats = EpochByEpochAgreement(*_simulate_pairs(15, seed_offset=100)).get_sleep_stats()
    return sstats.loc[REF_SCORER], sstats.loc[OBS_SCORER]


@pytest.fixture(scope="module")
def ssa_log_large(stats_large):
    return SleepStatsAgreement(
        *stats_large,
        ref_scorer=REF_SCORER,
        obs_scorer=OBS_SCORER,
        log_transform=True,
        bootstrap_kwargs={"n_resamples": 200},
    )


@pytest.fixture(scope="module")
def valid_arrays(ref_stats, obs_stats):
    def _valid_arrays(stat):
        """Reference and difference arrays for `stat`, dropping sessions with a NaN value.

        Sessions with a NaN value are dropped by the SleepStatsAgreement constructor (pivot_table).
        """
        valid = ref_stats[stat].notna() & obs_stats[stat].notna()
        ref = ref_stats.loc[valid, stat].to_numpy()
        diff = (obs_stats.loc[valid, stat] - ref_stats.loc[valid, stat]).to_numpy()
        return ref, diff

    return _valid_arrays


# ---------------------------------------------------------------------------
# EpochByEpochAgreement
# ---------------------------------------------------------------------------


class TestEpochByEpochAgreementInit:
    def test_attributes(self, ebe):
        assert REF_SCORER in repr(ebe) and OBS_SCORER in repr(ebe)
        assert str(ebe) == repr(ebe)
        assert (ebe.ref_scorer, ebe.obs_scorer) == (REF_SCORER, OBS_SCORER)
        assert ebe.n_sessions == N_SESSIONS
        assert ebe.data.shape[1] == 2

    def test_dict_input(self, ref_hyps, obs_hyps):
        ref_dict = {f"night{i}": h for i, h in enumerate(ref_hyps)}
        obs_dict = {f"night{i}": h for i, h in enumerate(obs_hyps)}
        assert EpochByEpochAgreement(ref_dict, obs_dict).n_sessions == N_SESSIONS
        # Hypnograms are paired by key, regardless of the order of the dictionaries
        obs_reversed = dict(reversed(obs_dict.items()))
        pd.testing.assert_frame_equal(
            EpochByEpochAgreement(ref_dict, obs_reversed).data,
            EpochByEpochAgreement(ref_dict, obs_dict).data,
        )

    def test_single_night_via_evaluate(self, ebe_single):
        assert ebe_single.n_sessions == 1
        assert (ebe_single.ref_scorer, ebe_single.obs_scorer) == (REF_SCORER, OBS_SCORER)

    def test_invalid_inputs_raise(self, ref_hyps, obs_hyps):
        with pytest.raises(AssertionError):
            EpochByEpochAgreement(ref_hyps, obs_hyps[:-1])
        same = [h.simulate_similar(scorer=REF_SCORER, seed=i) for i, h in enumerate(ref_hyps)]
        with pytest.raises(AssertionError):
            EpochByEpochAgreement(ref_hyps, same)
        no_scorer = [simulate_hypnogram(tib=90, seed=i) for i in range(N_SESSIONS)]
        with pytest.raises(AssertionError):
            EpochByEpochAgreement(ref_hyps, no_scorer)

    def test_art_uns_excluded(self, ref_hyps, obs_hyps):
        # Epochs scored as ART or UNS by either scorer are excluded from the epoch-by-epoch
        # analyses, i.e. the results are the same as when these epochs are removed beforehand
        ref = ref_hyps[0].hypno.to_numpy().copy()
        obs = obs_hyps[0].hypno.to_numpy().copy()
        ref[:5], obs[10:15], ref[20], obs[20] = "ART", "UNS", "UNS", "ART"
        keep = np.ones(ref.size, dtype=bool)
        keep[[*range(5), *range(10, 15), 20]] = False
        ebe_art = EpochByEpochAgreement(
            [Hypnogram(list(ref), scorer=REF_SCORER), ref_hyps[1]],
            [Hypnogram(list(obs), scorer=OBS_SCORER), obs_hyps[1]],
        )
        ebe_rm = EpochByEpochAgreement(
            [Hypnogram(list(ref[keep]), scorer=REF_SCORER), ref_hyps[1]],
            [Hypnogram(list(obs[keep]), scorer=OBS_SCORER), obs_hyps[1]],
        )
        assert ebe_art.data.shape[0] == ebe_rm.data.shape[0]
        np.testing.assert_array_equal(ebe_art.data, ebe_rm.data)
        pd.testing.assert_frame_equal(ebe_art.get_agreement(), ebe_rm.get_agreement())
        pd.testing.assert_frame_equal(
            ebe_art.get_agreement_bystage(), ebe_rm.get_agreement_bystage()
        )
        cm = ebe_art.get_confusion_matrix(agg_func="sum")
        assert "ART" not in cm.index and "UNS" not in cm.columns
        pd.testing.assert_frame_equal(cm, ebe_rm.get_confusion_matrix(agg_func="sum"))
        # ... but they are kept in the hypnograms, e.g. for the sleep statistics
        assert ebe_art.get_sleep_stats().loc[REF_SCORER, "TIB"].iloc[0] == ref.size / 2

    def test_all_art_uns_raises(self, ref_hyps, obs_hyps):
        ref = Hypnogram(["ART"] * 5 + ["UNS"] * 5, scorer=REF_SCORER)
        obs = Hypnogram(["WAKE"] * 10, scorer=OBS_SCORER)
        with pytest.raises(ValueError, match="All epochs are scored as ART or UNS"):
            EpochByEpochAgreement([ref, ref_hyps[0]], [obs, obs_hyps[0]])


class TestMultiScorer:
    """EpochByEpochAgreement.multi_scorer is a staticmethod returning a dict of scores."""

    _SCORERS = {
        "accuracy": lambda t, p, w: skm.accuracy_score(t, p, sample_weight=w),
        "n": lambda t, p, w: len(t),
        "weights": lambda t, p, w: w,
    }

    @pytest.fixture
    def df(self):
        return pd.DataFrame({"ref": [0, 1, 2, 2, 1], "obs": [0, 1, 1, 2, 2]})

    def test_two_columns(self, df):
        scores = EpochByEpochAgreement.multi_scorer(df, scorers=self._SCORERS)
        assert isinstance(scores, dict) and list(scores) == list(self._SCORERS)
        assert scores["accuracy"] == skm.accuracy_score(df["ref"], df["obs"])
        assert scores["n"] == len(df)
        assert scores["weights"] is None

    def test_third_column_is_sample_weight(self, df):
        weights = [1.0, 1.0, 0.0, 1.0, 0.0]
        scores = EpochByEpochAgreement.multi_scorer(df.assign(w=weights), scorers=self._SCORERS)
        # The two disagreeing epochs have a weight of zero
        assert scores["accuracy"] == 1.0
        assert scores["accuracy"] == skm.accuracy_score(df["ref"], df["obs"], sample_weight=weights)
        assert list(scores["weights"]) == weights

    def test_matches_get_agreement(self, ebe):
        # get_agreement(pooled=True) is multi_scorer applied to all epochs at once
        scores = EpochByEpochAgreement.multi_scorer(
            ebe.data, scorers={"acc": lambda t, p, w: 100 * skm.accuracy_score(t, p)}
        )
        np.testing.assert_allclose(scores["acc"], ebe.get_agreement(pooled=True)["accuracy"])

    @pytest.mark.parametrize(
        "df, scorers",
        [
            pytest.param([[0, 1], [1, 1]], _SCORERS, id="not_a_dataframe"),
            pytest.param(pd.DataFrame({"a": [0, 1]}), _SCORERS, id="one_column"),
            pytest.param(pd.DataFrame(np.zeros((2, 4))), _SCORERS, id="four_columns"),
            pytest.param(pd.DataFrame(np.zeros((2, 2))), [skm.accuracy_score], id="not_a_dict"),
            pytest.param(pd.DataFrame(np.zeros((2, 2))), {1: lambda t, p, w: 0}, id="key_not_str"),
            pytest.param(pd.DataFrame(np.zeros((2, 2))), {"a": 0}, id="not_callable"),
        ],
    )
    def test_invalid_inputs_raise(self, df, scorers):
        with pytest.raises(AssertionError):
            EpochByEpochAgreement.multi_scorer(df, scorers=scorers)


class TestGetAgreement:
    def test_shape_and_columns(self, ebe):
        agr = ebe.get_agreement()
        assert isinstance(agr, pd.DataFrame) and agr.shape[0] == N_SESSIONS
        expected_cols = {"accuracy", "balanced_acc", "kappa", "mcc", "precision", "f1"}
        assert set(agr.columns) == expected_cols

    def test_single_night_returns_series(self, ebe_single):
        assert isinstance(ebe_single.get_agreement(), pd.Series)

    def test_pooled_does_not_affect_summary(self, ref_hyps, obs_hyps):
        ebe_pooled = EpochByEpochAgreement(ref_hyps, obs_hyps)
        ebe_pooled.get_agreement(pooled=True)
        expected = EpochByEpochAgreement(ref_hyps, obs_hyps).summary()
        pd.testing.assert_frame_equal(ebe_pooled.summary(), expected)

    def test_scorers_list(self, fresh_ebe):
        default = fresh_ebe.get_agreement()
        # Metric names map to sklearn.metrics.<name>_score (returned as is, i.e. not in percent)
        agr = fresh_ebe.get_agreement(scorers=["accuracy", "cohen_kappa"])
        np.testing.assert_allclose(100 * agr["accuracy"], default["accuracy"])
        np.testing.assert_allclose(agr["cohen_kappa"], default["kappa"])

    def test_scorers_dict(self, fresh_ebe):
        default = fresh_ebe.get_agreement()
        agr = fresh_ebe.get_agreement(
            scorers={"acc": lambda t, p, w: skm.accuracy_score(t, p, sample_weight=w)}
        )
        assert agr.columns.tolist() == ["acc"]
        np.testing.assert_allclose(100 * agr["acc"], default["accuracy"])

    def test_sample_weight(self, ebe):
        default = ebe.get_agreement()
        # Uniform sample weights leave the scores unchanged
        weighted = ebe.get_agreement(sample_weight=pd.Series(2.0, index=ebe.data.index))
        pd.testing.assert_frame_equal(weighted, default)

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(sample_weight=[1, 2]),
            dict(sample_weight=pd.Series([1.0, 2.0])),
            dict(pooled=1),
            dict(scorers="accuracy"),
            dict(scorers=["accuracy", 1]),
            dict(scorers={"acc": "accuracy"}),
        ],
    )
    def test_invalid_args_raise(self, ebe, kwargs):
        with pytest.raises(AssertionError):
            ebe.get_agreement(**kwargs)


class TestGetAgreementByStage:
    def test_structure(self, ebe):
        agr = ebe.get_agreement_bystage()
        assert isinstance(agr, pd.DataFrame)
        assert set(agr.columns) == {"fbeta", "npv", "precision", "recall", "specificity", "support"}
        assert agr.index.names == ["stage", "sleep_id"]

    def test_single_night_no_sleep_id_level(self, ebe_single):
        assert ebe_single.get_agreement_bystage().index.name == "stage"


class TestGetAgreementByStageZeroDivision:
    """Test the zero_division parameter of get_agreement_bystage."""

    def test_default_is_nan_for_absent_reference_stage(self, ebe_missing):
        agr = ebe_missing.get_agreement_bystage()
        for stage in ["N1", "N3"]:
            assert agr.at[(stage, 1), "support"] == 0
            assert np.isnan(agr.at[(stage, 1), "recall"])
        # recall is NaN exactly where the stage is absent from the reference (support == 0), and
        # precision is NaN exactly where the observed scorer never assigned the stage.
        cm = ebe_missing.get_confusion_matrix()
        for (stage, sid), row in agr.iterrows():
            assert np.isnan(row["recall"]) == (row["support"] == 0)
            assert np.isnan(row["precision"]) == (cm.loc[sid][stage].sum() == 0)

    def test_zero_division_zero_restores_old_behavior(self, ebe_missing):
        agr = ebe_missing.get_agreement_bystage(zero_division=0)
        assert not agr.isna().any().any()
        assert agr.at[("N1", 1), "recall"] == 0
        assert agr.at[("N3", 1), "recall"] == 0

    def test_zero_division_one(self, ebe_missing):
        assert ebe_missing.get_agreement_bystage(zero_division=1).at[("N1", 1), "recall"] == 100

    def test_zero_division_warn(self, ebe_missing):
        with pytest.warns(UndefinedMetricWarning):
            agr = ebe_missing.get_agreement_bystage(zero_division="warn")
        assert agr.at[("N1", 1), "recall"] == 0

    @pytest.mark.parametrize("bad", [0.5, "nan"])
    def test_invalid_zero_division_raises(self, ebe, bad):
        with pytest.raises(AssertionError):
            ebe.get_agreement_bystage(zero_division=bad)

    def test_nan_excluded_from_summary(self, ebe_missing):
        # Group mean recall for N1 must equal the recall of the only session with N1 in the
        # reference (session 2), not be dragged down by a spurious 0 from session 1.
        agr = ebe_missing.get_agreement_bystage()
        summ = ebe_missing.summary(by_stage=True, func=["count", "mean"])
        assert summ.at[("N1", "recall"), "count"] == 1
        assert summ.at[("N1", "recall"), "mean"] == agr.at[("N1", 2), "recall"]
        assert summ.at[("N1", "support"), "count"] == 2

    def test_specificity_zero_division(self):
        # If the reference is entirely one stage, specificity (tn / (tn + fp)) is undefined for
        # that stage since there are no negative epochs.
        ref = Hypnogram(["N2"] * 20, scorer=REF_SCORER)
        obs = Hypnogram(["N2"] * 15 + ["WAKE"] * 5, scorer=OBS_SCORER)
        assert np.isnan(ref.evaluate(obs).get_agreement_bystage().at["N2", "specificity"])
        agr0 = ref.evaluate(obs).get_agreement_bystage(zero_division=0)
        assert agr0.at["N2", "specificity"] == 0
        # Like sklearn, zero_division="warn" warns and returns 0 for the hand-computed metrics
        # (sklearn also warns about the recall of WAKE, which is absent from the reference)
        with pytest.warns(UndefinedMetricWarning) as rec:
            agr_warn = ref.evaluate(obs).get_agreement_bystage(zero_division="warn")
        assert any("Specificity is ill-defined" in str(w.message) for w in rec)
        assert agr_warn.at["N2", "specificity"] == 0


class TestSummary:
    @pytest.mark.parametrize("by_stage", [False, True])
    def test_computes_scores_if_needed(self, fresh_ebe, by_stage):
        # summary() calls get_agreement() or get_agreement_bystage() if not done before
        attr = "_agreement_bystage" if by_stage else "_agreement"
        assert not hasattr(fresh_ebe, attr)
        summ = fresh_ebe.summary(by_stage=by_stage)
        assert hasattr(fresh_ebe, attr)
        pd.testing.assert_frame_equal(summ, fresh_ebe.summary(by_stage=by_stage))


class TestSummaryBootCI:
    """Bootstrap confidence intervals for EpochByEpochAgreement.summary()."""

    def test_no_ci_by_default(self, ebe):
        # `ci_method=None` is the default, so the historical output is unchanged
        default = ebe.summary()
        assert not {"ci_lower", "ci_upper"} & set(default.columns)
        pd.testing.assert_frame_equal(default, ebe.summary(ci_method=None))
        pd.testing.assert_frame_equal(
            ebe.summary(by_stage=True), ebe.summary(by_stage=True, ci_method=None)
        )

    @pytest.mark.parametrize("by_stage", [False, True])
    @pytest.mark.parametrize("method", BOOT_METHODS)
    def test_ci_brackets_mean(self, ebe, by_stage, method):
        kwargs = {"n_resamples": 200, "rng": 0, "method": method}
        out = _summary_quiet(ebe, by_stage=by_stage, ci_method="boot", bootstrap_kwargs=kwargs)
        assert out.columns[-2:].tolist() == ["ci_lower", "ci_upper"]
        # Metrics defined in at least one session must have a finite CI around the mean
        valid = out.dropna(subset=["ci_lower", "ci_upper"])
        assert not valid.empty
        assert (valid["ci_lower"] <= valid["mean"] + 1e-9).all()
        assert (valid["ci_upper"] >= valid["mean"] - 1e-9).all()

    def test_bca_is_the_default_method(self, ebe):
        default = _summary_quiet(
            ebe, ci_method="boot", bootstrap_kwargs={"n_resamples": 200, "rng": 0}
        )
        bca = _summary_quiet(
            ebe, ci_method="boot", bootstrap_kwargs={"n_resamples": 200, "rng": 0, "method": "BCa"}
        )
        pd.testing.assert_frame_equal(default, bca)

    def test_reproducible_with_rng(self, ebe):
        kwargs = {"n_resamples": 200, "rng": 3}
        out = _summary_quiet(ebe, ci_method="boot", bootstrap_kwargs=kwargs)
        pd.testing.assert_frame_equal(
            out, _summary_quiet(ebe, ci_method="boot", bootstrap_kwargs=kwargs)
        )
        other = _summary_quiet(
            ebe, ci_method="boot", bootstrap_kwargs={"n_resamples": 200, "rng": 4}
        )
        assert not out["ci_lower"].equals(other["ci_lower"])

    def test_matches_manual_session_bootstrap(self, ebe):
        # Re-implement the basic participant bootstrap by hand with the same rng. Also checks that
        # the CI is NOT clipped, unlike get_confusion_matrix_proportional.
        n_resamples = 100
        out = _summary_quiet(
            ebe,
            ci_method="boot",
            bootstrap_kwargs={"n_resamples": n_resamples, "rng": 42, "method": "basic"},
        )
        arr = ebe.get_agreement().to_numpy(dtype=float)  # (n_sessions, n_metrics)
        rng = np.random.default_rng(42)
        idx = rng.integers(0, N_SESSIONS, size=(n_resamples, N_SESSIONS))
        boot = np.nanmean(arr[idx], axis=1)
        mean = np.nanmean(arr, axis=0)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
        np.testing.assert_allclose(out["ci_lower"].to_numpy(), 2 * mean - hi)
        np.testing.assert_allclose(out["ci_upper"].to_numpy(), 2 * mean - lo)

    def test_undefined_metrics_and_support(self, ebe_missing):
        # Session 1 of `ebe_missing` has no N1 and no N3 in the reference hypnogram
        out = _summary_quiet(
            ebe_missing,
            by_stage=True,
            ci_method="boot",
            bootstrap_kwargs={"n_resamples": 200, "rng": 0},
            func=["count", "mean"],
        )
        # N3 recall is undefined in both sessions -> no mean and no CI
        assert out.at[("N3", "recall"), "count"] == 0
        assert out.loc[("N3", "recall"), ["ci_lower", "ci_upper"]].isna().all()
        # N1 recall is defined in a single session -> the CI collapses onto that value
        assert out.at[("N1", "recall"), "count"] == 1
        np.testing.assert_allclose(
            out.loc[("N1", "recall"), ["ci_lower", "ci_upper"]].to_numpy(dtype=float),
            out.at[("N1", "recall"), "mean"],
        )
        # Metrics defined in both sessions get a real interval
        assert out.loc[("WAKE", "recall"), ["ci_lower", "ci_upper"]].notna().all()
        # `support` is an epoch count, not an agreement score
        assert out.xs("support", level="metric")[["ci_lower", "ci_upper"]].isna().all().all()

    def test_small_n_bca_warning(self, ebe):
        assert ebe.n_sessions < 20
        with pytest.warns(RuntimeWarning, match="BCa"):
            ebe.summary(ci_method="boot", bootstrap_kwargs={"n_resamples": 50})

    @pytest.mark.parametrize(
        "obj, kwargs",
        [
            ("ebe", {"n_resamples": 50, "method": "percentile"}),
            ("ebe", {"n_resamples": 50, "method": "basic"}),
            ("ebe_large", {"n_resamples": 50}),
            ("ebe", None),  # No bootstrap at all
        ],
    )
    def test_no_bca_warning(self, request, obj, kwargs):
        # Only BCa is affected, and only below the threshold
        obj = request.getfixturevalue(obj)
        ci_method = None if kwargs is None else "boot"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            obj.summary(ci_method=ci_method, bootstrap_kwargs=kwargs)
        assert not [w for w in rec if "BCa" in str(w.message)]

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(ci_method="param"),
            dict(ci_method=True),
            dict(confidence=95),
            dict(confidence=0),
            dict(bootstrap_kwargs={"confidence_level": 0.9}),
            dict(bootstrap_kwargs={"method": "bca"}),
            dict(bootstrap_kwargs={"n_resamples": 0}),
            dict(bootstrap_kwargs=[]),
        ],
    )
    def test_invalid_args_raise(self, ebe, kwargs):
        with pytest.raises(AssertionError):
            ebe.summary(**kwargs)

    def test_single_night_raises(self, ebe_single):
        with pytest.raises(AssertionError):
            ebe_single.summary(ci_method="boot")


class TestGetConfusionMatrixProportional:
    def test_structure(self, ebe):
        out = ebe.get_confusion_matrix_proportional(ci_method="param")
        assert out.index.names == [REF_SCORER, OBS_SCORER]
        assert list(out.columns) == ["mean", "std", "ci_lower", "ci_upper", "n_sessions"]
        n_stages = ebe.get_confusion_matrix(sleep_id=1).shape[0]
        assert len(out) == n_stages**2
        out = ebe.get_confusion_matrix_proportional(ci_method=None)
        assert list(out.columns) == ["mean", "std", "n_sessions"]

    def test_matches_manual_computation(self, ebe):
        out = ebe.get_confusion_matrix_proportional(ci_method="param")
        cms = ebe.get_confusion_matrix()
        props = 100 * cms.div(cms.sum(axis=1), axis=0)
        manual_mean = props.groupby(level=REF_SCORER, sort=False).mean()
        manual_std = props.groupby(level=REF_SCORER, sort=False).std(ddof=1)
        for (r, c), row in out.iterrows():
            np.testing.assert_allclose(row["mean"], manual_mean.at[r, c])
            np.testing.assert_allclose(row["std"], manual_std.at[r, c])
            assert row["n_sessions"] == props.xs(r, level=REF_SCORER)[c].notna().sum()
        np.testing.assert_allclose(out["mean"].unstack().sum(axis=1), 100.0)
        # Parametric CI is centered on the mean and clipped to [0, 100]
        assert (out["ci_lower"] <= out["mean"]).all() and (out["ci_upper"] >= out["mean"]).all()
        assert (out["ci_lower"] >= 0).all() and (out["ci_upper"] <= 100).all()

    @pytest.mark.parametrize("method", BOOT_METHODS)
    def test_boot_ci(self, ebe, method):
        kwargs = {"n_resamples": 200, "rng": 0, "method": method}
        out = ebe.get_confusion_matrix_proportional(ci_method="boot", bootstrap_kwargs=kwargs)
        # Rows with at least one contributing session must have a finite CI around the mean
        valid = out[out["n_sessions"] > 0]
        assert valid[["ci_lower", "ci_upper"]].notna().all().all()
        assert (valid["ci_lower"] <= valid["mean"] + 1e-9).all()
        assert (valid["ci_upper"] >= valid["mean"] - 1e-9).all()
        assert (valid["ci_lower"] >= 0).all() and (valid["ci_upper"] <= 100).all()
        # Reproducible with a fixed rng
        out2 = ebe.get_confusion_matrix_proportional(ci_method="boot", bootstrap_kwargs=kwargs)
        pd.testing.assert_frame_equal(out, out2)

    def test_bca_is_the_default_method(self, ebe):
        default = ebe.get_confusion_matrix_proportional(
            bootstrap_kwargs={"n_resamples": 200, "rng": 0}
        )
        bca = ebe.get_confusion_matrix_proportional(
            bootstrap_kwargs={"n_resamples": 200, "rng": 0, "method": "BCa"}
        )
        pd.testing.assert_frame_equal(default, bca)

    def test_bca_degenerate_cells(self, ref_hyps, obs_hyps):
        # Two identical session pairs -> every cell is constant across resamples, so the BCa
        # interval must collapse onto the mean (percentile fallback) rather than being NaN.
        ebe_const = EpochByEpochAgreement([ref_hyps[0], ref_hyps[0]], [obs_hyps[0], obs_hyps[0]])
        out = ebe_const.get_confusion_matrix_proportional(bootstrap_kwargs={"n_resamples": 50})
        valid = out[out["n_sessions"] > 0]
        np.testing.assert_allclose(valid["ci_lower"], valid["mean"])
        np.testing.assert_allclose(valid["ci_upper"], valid["mean"])

    def test_boot_ci_matches_manual_basic_bootstrap(self, ebe):
        # Re-implement the basic participant bootstrap by hand with the same rng
        n_resamples = 100
        out = ebe.get_confusion_matrix_proportional(
            ci_method="boot",
            bootstrap_kwargs={"n_resamples": n_resamples, "rng": 42, "method": "basic"},
        )
        cms = ebe.get_confusion_matrix()
        stages = cms.columns.tolist()
        props = 100 * cms.div(cms.sum(axis=1), axis=0)
        arr = np.stack(
            [
                g.droplevel("sleep_id").loc[stages, stages].to_numpy()
                for _, g in props.groupby(level=0)
            ]
        )
        rng = np.random.default_rng(42)
        idx = rng.integers(0, N_SESSIONS, size=(n_resamples, N_SESSIONS))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            boot = np.nanmean(arr[idx], axis=1)  # (n_resamples, n_ref, n_obs)
            mean = np.nanmean(arr, axis=0)
            lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
        exp_lower = np.clip(2 * mean - hi, 0, 100).ravel()
        exp_upper = np.clip(2 * mean - lo, 0, 100).ravel()
        np.testing.assert_allclose(out["ci_lower"].to_numpy(), exp_lower, equal_nan=True)
        np.testing.assert_allclose(out["ci_upper"].to_numpy(), exp_upper, equal_nan=True)

    def test_formatted(self, ebe):
        fmt = ebe.get_confusion_matrix_proportional(ci_method="param", formatted=True)
        assert fmt.index.name == REF_SCORER and fmt.columns.name == OBS_SCORER
        assert fmt.shape[0] == fmt.shape[1]
        assert fmt.map(lambda s: bool(re.fullmatch(_FMT_CI, s))).all().all()
        fmt_noci = ebe.get_confusion_matrix_proportional(ci_method=None, formatted=True)
        assert fmt_noci.map(lambda s: bool(re.fullmatch(_FMT_NOCI, s))).all().all()

    def test_absent_stage_is_nan_not_zero(self, ebe_missing):
        out = ebe_missing.get_confusion_matrix_proportional(ci_method="param")
        cm = ebe_missing.get_confusion_matrix()
        # Number of sessions in which each reference stage is present
        n_present = cm.sum(axis=1).gt(0).groupby(level=REF_SCORER).sum()
        # Session 1 has no N1 and no N3 in the reference, so at most one session contributes
        assert n_present["N1"] <= 1 and n_present["N3"] <= 1
        for stage in cm.columns:
            sub = out.xs(stage, level=REF_SCORER)
            assert (sub["n_sessions"] == n_present[stage]).all()
            if n_present[stage] == 0:
                # Stage never present in the reference: everything is undefined
                assert sub[["mean", "std", "ci_lower", "ci_upper"]].isna().all().all()
            elif n_present[stage] == 1:
                # Single contributing session: mean is defined, SD and CI are not
                np.testing.assert_allclose(sub["mean"].sum(), 100.0)
                assert sub[["std", "ci_lower", "ci_upper"]].isna().all().all()
            else:
                assert sub[["mean", "std", "ci_lower", "ci_upper"]].notna().all().all()
        # WAKE is present in both sessions
        assert (out.xs("WAKE", level=REF_SCORER)["n_sessions"] == 2).all()

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(bootstrap_kwargs={"confidence_level": 0.9}),
            dict(bootstrap_kwargs={"method": "bca"}),
            dict(bootstrap_kwargs={"n_resamples": 0}),
            dict(ci_method="invalid"),
            dict(confidence=95),
            dict(decimals=-1),
        ],
    )
    def test_invalid_args_raise(self, ebe, kwargs):
        with pytest.raises(AssertionError):
            ebe.get_confusion_matrix_proportional(**kwargs)

    def test_single_night_raises(self, ebe_single):
        with pytest.raises(AssertionError):
            ebe_single.get_confusion_matrix_proportional()


class TestGetConfusionMatrix:
    def test_single_session(self, ebe, ref_hyps):
        cm = ebe.get_confusion_matrix(sleep_id=1)
        assert cm.index.name == REF_SCORER and cm.columns.name == OBS_SCORER
        assert cm.values.sum() == ref_hyps[0].n_epochs
        with pytest.raises(AssertionError):
            ebe.get_confusion_matrix(sleep_id=999)

    def test_all_sessions_and_agg_sum(self, ebe, ref_hyps):
        cm = ebe.get_confusion_matrix()
        assert cm.index.names == ["sleep_id", REF_SCORER]
        cm_sum = ebe.get_confusion_matrix(agg_func="sum")
        assert cm_sum.values.sum() == sum(h.n_epochs for h in ref_hyps)

    def test_row_labels_correct_for_noncontiguous_codes(self):
        # Regression test: row labels were corrupted when YASA's internal integer codes
        # are non-contiguous.  A 4-stage mapping {0:"W",1:"Light",2:"Deep",3:"R"} maps
        # the input integers to YASA codes [0, 2, 3, 4] (skipping 1 = N1).  The old code
        # passed those codes directly to _skm2yasa_map, which expected positional indices
        # [0, 1, 2, 3], producing ['WAKE', 'DEEP', 'REM', 'REM'] instead of
        # ['WAKE', 'LIGHT', 'DEEP', 'REM'].
        rng = np.random.default_rng(0)
        mapping = {0: "W", 1: "Light", 2: "Deep", 3: "R"}
        n = 360  # 3-hour recording at 30-s epochs
        h_ref = Hypnogram.from_integers(
            rng.integers(0, 4, n), mapping=mapping, n_stages=4, scorer="Ref"
        )
        h_obs = Hypnogram.from_integers(
            rng.integers(0, 4, n), mapping=mapping, n_stages=4, scorer="Obs"
        )
        cm = EpochByEpochAgreement({"night1": h_ref}, {"night1": h_obs}).get_confusion_matrix()
        assert sorted(cm.index.tolist()) == sorted(["WAKE", "LIGHT", "DEEP", "REM"])


class TestGetSleepStats:
    def test_structure(self, ebe):
        sstats = ebe.get_sleep_stats()
        assert sstats.index.names == ["scorer", "sleep_id"]
        assert set(sstats.index.get_level_values("scorer")) == {REF_SCORER, OBS_SCORER}
        assert len(sstats) == 2 * N_SESSIONS

    def test_single_night(self, ebe_single):
        assert set(ebe_single.get_sleep_stats().index) == {REF_SCORER, OBS_SCORER}


class TestPlotHypnograms:
    """Each hypnogram is drawn as one StepPatch in ``ax.patches`` (``highlight=None``)."""

    def test_single_session_defaults(self, ebe_single):
        ax = ebe_single.plot_hypnograms()
        assert isinstance(ax, plt.Axes)
        ref_line, obs_line = ax.patches
        assert ref_line.get_label() == REF_SCORER and obs_line.get_label() == OBS_SCORER
        # Solid black reference, dashed green observed, and no REM highlight (hlines)
        np.testing.assert_allclose(ref_line.get_edgecolor()[:3], to_rgb("black"))
        np.testing.assert_allclose(obs_line.get_edgecolor()[:3], to_rgb("green"))
        assert ref_line.get_linestyle() == "solid" and obs_line.get_linestyle() == "dashed"
        assert len(ax.collections) == 0
        assert [t.get_text() for t in ax.get_legend().get_texts()] == [REF_SCORER, OBS_SCORER]

    def test_multi_session_requires_sleep_id(self, ebe):
        with pytest.raises(AssertionError, match="Multi-session"):
            ebe.plot_hypnograms()
        _, ax = plt.subplots()
        assert ebe.plot_hypnograms(sleep_id=2, ax=ax) is ax
        assert len(ax.patches) == 2

    def test_legend(self, ebe_single):
        assert ebe_single.plot_hypnograms(legend=False).get_legend() is None
        _, ax = plt.subplots()
        ebe_single.plot_hypnograms(legend={"title": "Scorers"}, ax=ax)
        assert ax.get_legend().get_title().get_text() == "Scorers"

    def test_kwargs_passthrough(self, ebe_single):
        ax = ebe_single.plot_hypnograms(
            ref_kwargs={"color": "blue", "label": "Ref"},
            obs_kwargs={"highlight": "REM", "lw": 3},
        )
        ref_line, obs_line = ax.patches
        assert ref_line.get_label() == "Ref"
        np.testing.assert_allclose(ref_line.get_edgecolor()[:3], to_rgb("blue"))
        assert obs_line.get_linewidth() == 3
        # The REM highlight of the observed hypnogram is drawn with hlines
        assert len(ax.collections) == 1

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(sleep_id=999),
            dict(legend="yes"),
            dict(ref_kwargs=["color", "red"]),
            dict(obs_kwargs="red"),
            dict(ref_kwargs={"ax": None}),
            dict(obs_kwargs={"ax": None}),
        ],
    )
    def test_invalid_args_raise(self, ebe_single, kwargs):
        with pytest.raises(AssertionError):
            ebe_single.plot_hypnograms(**kwargs)


# ---------------------------------------------------------------------------
# SleepStatsAgreement
# ---------------------------------------------------------------------------


class TestSleepStatsAgreementInit:
    def test_attributes(self, ssa):
        assert REF_SCORER in repr(ssa) and OBS_SCORER in repr(ssa)
        assert str(ssa) == repr(ssa)
        assert (ssa.ref_scorer, ssa.obs_scorer) == (REF_SCORER, OBS_SCORER)
        assert ssa.n_sessions == N_SESSIONS
        assert isinstance(ssa.sleep_statistics, list) and len(ssa.sleep_statistics) > 0
        assert ssa.data.shape[1] == 2
        assert int(ssa._confidence * 100) == PCT

    def test_default_scorer_names(self, ref_stats, obs_stats):
        ssa_default = SleepStatsAgreement(ref_stats, obs_stats)
        assert (ssa_default.ref_scorer, ssa_default.obs_scorer) == ("Reference", "Observed")

    def test_from_get_sleep_stats(self, ssa, sstats):
        # Built directly from EpochByEpochAgreement.get_sleep_stats(): scorers taken from the index
        ssa_multi = SleepStatsAgreement(sstats)
        assert (ssa_multi.ref_scorer, ssa_multi.obs_scorer) == (REF_SCORER, OBS_SCORER)
        pd.testing.assert_frame_equal(
            ssa_multi.summary(ci_method=None), ssa.summary(ci_method=None)
        )

    # Each lambda returns the arguments that override the valid (ref_stats, obs_stats) inputs
    @pytest.mark.parametrize(
        "make_kwargs",
        [
            pytest.param(lambda r, o: dict(ref_data=r.to_numpy()), id="ref_ndarray"),
            pytest.param(lambda r, o: dict(obs_data=o.to_numpy()), id="obs_ndarray"),
            pytest.param(lambda r, o: dict(obs_data=o.set_axis(o.index + 100)), id="index"),
            pytest.param(lambda r, o: dict(obs_data=o.rename(columns={"TST": "X"})), id="columns"),
            pytest.param(lambda r, o: dict(ref_scorer="X", obs_scorer="X"), id="same_scorer"),
            pytest.param(lambda r, o: dict(log_transform="TST"), id="log_transform"),
            pytest.param(lambda r, o: dict(alpha=1.5), id="alpha"),
            pytest.param(lambda r, o: dict(effect_size_gates={"bad_key": 1}), id="gates_key"),
            pytest.param(lambda r, o: dict(effect_size_gates={"r2": -1}), id="gates_value"),
            pytest.param(lambda r, o: dict(effect_size_gates=0.1), id="gates_type"),
            pytest.param(lambda r, o: dict(confidence=95), id="confidence"),
        ],
    )
    def test_invalid_inputs_raise(self, ref_stats, obs_stats, make_kwargs):
        kwargs = {"ref_data": ref_stats, "obs_data": obs_stats}
        kwargs.update(make_kwargs(ref_stats, obs_stats))
        with pytest.raises(AssertionError):
            SleepStatsAgreement(**kwargs)

    def test_constant_reference(self, ref_stats, obs_stats):
        # Constant reference values (e.g. no N3 in any night): the stat is kept with a flat
        # bias regression, so the parametric bias and LoA are used
        ref = ref_stats.assign(N3=0.0)
        ssa_const = SleepStatsAgreement(
            ref, obs_stats, bootstrap_kwargs={"n_resamples": 100, "method": "basic"}
        )
        assert "N3" in ssa_const.sleep_statistics
        assumptions = ssa_const.assumptions.loc["N3"]
        assert assumptions["constant_bias", "passed"] and assumptions["homoscedastic", "passed"]
        assert ssa_const.summary(ci_method="boot").loc["N3"].notna().any()


class TestSleepStatsAgreementAssumptions:
    def test_structure(self, ssa):
        asmp = ssa.assumptions
        assert asmp.columns.names == ["assumption", "metric"]
        assert set(asmp.index) == set(ssa.sleep_statistics)
        expected = {
            "unbiased": {"t", "pvalue", "cohen_d", "passed"},
            "normal": {"W", "pvalue", "skew", "kurtosis", "passed", "method"},
            "constant_bias": {"slope", "pvalue", "r2", "passed", "method"},
            "homoscedastic": {"slope", "pvalue", "r2", "sd_ratio", "passed", "method"},
        }
        for assumption, metrics in expected.items():
            assert set(asmp[assumption].columns) == metrics

    def test_values(self, ssa, ref_stats, obs_stats):
        asmp = ssa.assumptions
        diff = obs_stats["TST"] - ref_stats["TST"]
        ttest = sps.ttest_1samp(diff, 0)
        assert np.isclose(asmp.at["TST", ("unbiased", "t")], ttest.statistic)
        assert np.isclose(asmp.at["TST", ("unbiased", "pvalue")], ttest.pvalue)
        assert np.isclose(asmp.at["TST", ("unbiased", "cohen_d")], diff.mean() / diff.std(ddof=1))
        assert np.isclose(asmp.at["TST", ("normal", "skew")], diff.skew())
        regr = sps.linregress(ref_stats["TST"], diff)
        assert np.isclose(asmp.at["TST", ("constant_bias", "slope")], regr.slope)
        assert np.isclose(asmp.at["TST", ("constant_bias", "r2")], regr.rvalue**2)
        loa_regr = sps.linregress(
            ref_stats["TST"], np.abs(diff - regr.intercept - regr.slope * ref_stats["TST"])
        )
        fitted = loa_regr.intercept + loa_regr.slope * np.array(
            [ref_stats["TST"].min(), ref_stats["TST"].max()]
        )
        assert np.isclose(asmp.at["TST", ("homoscedastic", "sd_ratio")], fitted[1] / fitted[0])

    def test_flags_and_methods(self, ssa, ssa_log):
        asmp = ssa.assumptions
        # Flags: pvalue >= alpha (0.05), or an immaterial effect size (see test_dual_criterion)
        for name in ["unbiased", "normal", "constant_bias", "homoscedastic"]:
            assert asmp[(name, "passed")][asmp[(name, "pvalue")].ge(0.05)].all()
        # Methods applied for "auto"
        mapping = {"normal": "boot", "constant_bias": "regr", "homoscedastic": "regr"}
        for name, failed in mapping.items():
            expected = asmp[(name, "passed")].map({True: "param", False: failed})
            assert (asmp[(name, "method")] == expected).all()
        # With log_transform=True, the LoA method is "log" for every log-transformed stat
        is_log = _log_slope(ssa_log).notna()
        loa = ssa_log.assumptions[("homoscedastic", "method")]
        assert (loa[is_log] == "log").all() and loa[~is_log].isin(["param", "regr"]).all()

    def test_alpha(self, ssa, ref_stats, obs_stats):
        # alpha=1 fails every test with p < 1 once the effect-size gates are disabled;
        # the statistics themselves are unchanged.
        ssa_strict = SleepStatsAgreement(
            ref_stats,
            obs_stats,
            ref_scorer=REF_SCORER,
            obs_scorer=OBS_SCORER,
            alpha=1.0,
            effect_size_gates=dict.fromkeys(SleepStatsAgreement._default_gates),
        )
        strict = ssa_strict.assumptions
        passed = strict.xs("passed", level="metric", axis=1)
        pvalues = strict.xs("pvalue", level="metric", axis=1)
        assert (passed == pvalues.ge(1.0)).all().all()
        pd.testing.assert_frame_equal(pvalues, ssa.assumptions.xs("pvalue", level="metric", axis=1))

    def test_unreachable_effect_size_gates(self, ref_stats, obs_stats):
        # Unreachable thresholds -> the effect size is never material -> nothing is ever
        # flagged as violated, even with alpha=1 (every test significant).
        ssa_lax = SleepStatsAgreement(
            ref_stats,
            obs_stats,
            ref_scorer=REF_SCORER,
            obs_scorer=OBS_SCORER,
            alpha=1.0,
            effect_size_gates={"skew": 1e9, "kurtosis": 1e9, "r2": 1.0, "sd_ratio": 1e9},
        )
        lax = ssa_lax.assumptions.xs("passed", level="metric", axis=1)
        assert lax[["normal", "constant_bias", "homoscedastic"]].all().all()
        # `unbiased` is not gated on an effect size, so it still fails
        assert not lax["unbiased"].any()

    def test_resolved_effect_size_gates(self, ssa, ref_stats, obs_stats):
        # The resolved thresholds are exposed, with the unspecified ones left at their default
        ssa_partial = SleepStatsAgreement(
            ref_stats,
            obs_stats,
            ref_scorer=REF_SCORER,
            obs_scorer=OBS_SCORER,
            effect_size_gates={"r2": 0.5},
        )
        assert ssa_partial.effect_size_gates == {
            "skew": 1.0,
            "kurtosis": 2.0,
            "r2": 0.5,
            "sd_ratio": 1.5,
        }
        assert ssa.effect_size_gates == SleepStatsAgreement._default_gates

    @pytest.mark.parametrize("name, gate", [("constant_bias", "r2"), ("homoscedastic", "sd_ratio")])
    def test_dual_criterion(self, ssa, name, gate):
        # A statistic fails only if it is both significant and material
        asmp = ssa.assumptions
        effect = asmp[(name, gate)]
        if gate == "sd_ratio":
            effect = np.maximum(effect, 1 / effect)
        failed = ~asmp[(name, "passed")]
        # Use ~ge rather than lt to match the implementation when the p-value is NaN, e.g. a
        # regression on two sessions (scipy >= 1.18 returns NaN instead of 0).
        significant = ~asmp[(name, "pvalue")].ge(0.05)
        assert (failed == (significant & effect.gt(ssa.effect_size_gates[gate]))).all()


class TestSleepStatsAgreementSummary:
    def test_matches_manual_computation(self, ssa, valid_arrays):
        s = ssa.summary(ci_method="param")
        assert "loa_log_slope" not in s.columns.get_level_values("variable")
        for stat in ssa.sleep_statistics:
            ref, diff = valid_arrays(stat)
            c = s.xs("center", level="interval", axis=1).loc[stat]
            sd = np.std(diff, ddof=1)
            assert np.isclose(c["bias_mean"], diff.mean())
            assert np.isclose(c["loa_lower"], diff.mean() - 1.96 * sd)
            assert np.isclose(c["loa_upper"], diff.mean() + 1.96 * sd)
            bias_regr = sps.linregress(ref, diff)
            assert np.isclose(c["bias_slope"], bias_regr.slope)
            assert np.isclose(c["bias_intercept"], bias_regr.intercept)
            resid = diff - (bias_regr.intercept + bias_regr.slope * ref)
            loa_regr = sps.linregress(ref, np.abs(resid))
            assert np.isclose(c["loa_slope"], loa_regr.slope)
            assert np.isclose(c["loa_intercept"], loa_regr.intercept)
            # Menghini et al. (2021) eq. 2 half-width: agreement × SD of the bias residuals
            assert np.isclose(c["loa_halfwidth"], 1.96 * np.std(resid, ddof=1))

    def test_param_ci_brackets_center(self, ssa):
        # Parametric CIs bracket the point estimates (NaN for stats with too few sessions)
        s = ssa.summary(ci_method="param")
        lower, center, upper = (
            s.xs(i, level="interval", axis=1) for i in ("lower", "center", "upper")
        )
        assert ((lower <= center + 1e-12) | lower.isna()).all().all()
        assert ((center <= upper + 1e-12) | upper.isna()).all().all()

    def test_missing_values_dropped_per_stat(self, ref_stats, obs_stats):
        # A NaN in one session of one stat only affects that stat, and inputs are not modified
        ref, obs = ref_stats.rename_axis(None), obs_stats.rename_axis(None)
        ref.loc[ref.index[0], "TST"] = np.nan
        ssa_nan = SleepStatsAgreement(ref, obs, ref_scorer=REF_SCORER, obs_scorer=OBS_SCORER)
        assert ref.index.name is None and obs.index.name is None
        assert len(ssa_nan.data.loc["TST"]) == N_SESSIONS - 1
        assert len(ssa_nan.data.loc["WASO"]) == N_SESSIONS
        assert np.isfinite(ssa_nan.summary(ci_method="param").loc["TST"]).all()

    def test_ci_method_none_returns_center_only(self, ssa):
        s = ssa.summary(ci_method=None)
        assert set(s.columns.get_level_values("interval")) == {"center"}
        full = ssa.summary(ci_method="param")
        pd.testing.assert_frame_equal(
            s, full.xs("center", axis=1, level="interval", drop_level=False)
        )

    def test_sleep_stats_subset_and_order(self, ssa):
        subset = ["WASO", "TST", "SE"]
        s = ssa.summary(ci_method="param", sleep_stats=subset)
        assert s.index.tolist() == subset
        pd.testing.assert_frame_equal(s, ssa.summary(ci_method="param").loc[subset])

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(ci_method="invalid"),
            dict(ci_method="param", sleep_stats=["NOT_A_STAT"]),
            dict(ci_method="param", sleep_stats="TST"),
            dict(ci_method="param", sleep_stats=["TST", "TST"]),
        ],
    )
    def test_invalid_args_raise(self, ssa, kwargs):
        with pytest.raises(AssertionError):
            ssa.summary(**kwargs)


class TestSleepStatsAgreementCalibrate:
    """calibrate() requires all columns to be in ssa.sleep_statistics — stats with identical values
    across scorers (e.g. TIB) are removed during construction, so obs_stats is subset first."""

    @pytest.fixture(scope="class")
    def bias_stats(self, ssa):
        """Statistics with a constant ("param") and with a proportional ("regr") bias."""
        methods = ssa.assumptions[("constant_bias", "method")]
        return methods.index[methods == "param"].tolist(), methods.index[methods == "regr"].tolist()

    def test_fixture_includes_both_methods(self, bias_stats):
        param_stats, regr_stats = bias_stats
        assert len(param_stats) >= 2 and len(regr_stats) >= 1

    def test_calibrated_values(self, ssa, obs_stats):
        obs = obs_stats[ssa.sleep_statistics]
        vals = ssa.summary(ci_method=None).xs("center", level="interval", axis=1)
        param = ssa.calibrate(obs, bias_method="param")
        assert isinstance(param, pd.DataFrame) and param.shape == obs.shape
        pd.testing.assert_frame_equal(param, obs - vals["bias_mean"], check_names=False)

    def test_missing_values_and_column_order_kept(self, ssa, obs_stats):
        obs = obs_stats[ssa.sleep_statistics]
        obs_nan = obs.copy()
        obs_nan.iloc[0, 0] = np.nan
        param_nan = ssa.calibrate(obs_nan, bias_method="param")
        assert param_nan.columns.tolist() == obs.columns.tolist()
        assert np.isnan(param_nan.iloc[0, 0])
        assert param_nan.notna().sum().sum() == obs.notna().sum().sum() - 1

    def test_auto_constant_bias(self, ssa, obs_stats, bias_stats):
        # With constant-bias statistics only, "auto" subtracts the mean difference
        obs = obs_stats[bias_stats[0]]
        auto = ssa.calibrate(obs, bias_method="auto")
        pd.testing.assert_frame_equal(auto, ssa.calibrate(obs, bias_method="param"))

    def test_proportional_bias_not_implemented(self, ssa, obs_stats, bias_stats):
        param_stats, regr_stats = bias_stats
        with pytest.raises(NotImplementedError):
            ssa.calibrate(obs_stats[param_stats], bias_method="regr")
        # "auto" raises as soon as one statistic has a proportional bias, and names it
        with pytest.raises(NotImplementedError, match=re.escape(regr_stats[0])):
            ssa.calibrate(obs_stats[[param_stats[0], regr_stats[0]]], bias_method="auto")

    def test_subset_of_columns(self, ssa, obs_stats, bias_stats):
        # Calibrating a subset of statistics returns only these columns, in the same order
        param_stats = bias_stats[0]
        cols = [param_stats[1], param_stats[0]]
        full = ssa.calibrate(obs_stats[ssa.sleep_statistics], bias_method="param")
        subset = ssa.calibrate(obs_stats[cols], bias_method="auto")
        assert subset.columns.tolist() == cols
        pd.testing.assert_frame_equal(subset, full[cols])

    def test_invalid_column_raises(self, ssa, obs_stats):
        bad = obs_stats[ssa.sleep_statistics].rename(columns={"TST": "NOT_A_STAT"})
        with pytest.raises(AssertionError):
            ssa.calibrate(bad)


class TestSleepStatsAgreementReport:
    """Use ci_method="param" to avoid the bootstrap path with small samples (N_SESSIONS=5)."""

    def test_columns(self, ssa):
        rpt = ssa.report(ci_method="param")
        expected = [
            f"{REF_SCORER} mean (SD)",
            f"{OBS_SCORER} mean (SD)",
            f"Bias [{PCT}% CI]",
            f"LoA [{PCT}% CI]",
            "Assumptions",
        ]
        assert rpt.columns.tolist() == expected
        # The unbiased flag is a finding, not a modeling assumption, and is not reported
        assert (
            rpt["Assumptions"]
            .str.fullmatch(r"[✓✗] normal  [✓✗] constant bias  [✓✗] homoscedastic")
            .all()
        )

    @pytest.mark.parametrize("scorer", [REF_SCORER, OBS_SCORER])
    def test_mean_sd_columns(self, ssa, sstats, scorer):
        rpt = ssa.report(ci_method="param", decimals=2)
        data = sstats.loc[scorer]
        col = rpt[f"{scorer} mean (SD)"]
        assert col.str.fullmatch(r"-?\d+\.\d{2} \(\d+\.\d{2}\)").all()
        assert col["TST (min)"] == f"{data['TST'].mean():.2f} ({data['TST'].std(ddof=1):.2f})"

    def test_ci_columns_contain_brackets(self, ssa):
        rpt = ssa.report(bias_method="param", loa_method="param", ci_method="param")
        num = r"-?\d+\.\d+"
        assert rpt[f"Bias [{PCT}% CI]"].str.fullmatch(rf"{num} \[{num}, {num}\]").all()
        assert (
            rpt[f"LoA [{PCT}% CI]"]
            .str.fullmatch(rf"{num} to {num} \[{num}, {num}; {num}, {num}\]")
            .all()
        )

    def test_no_ci(self, ssa):
        rpt = ssa.report(bias_method="param", loa_method="param", ci_method=None)
        assert "Bias" in rpt.columns and "LoA" in rpt.columns
        assert not any("CI" in c for c in rpt.columns)
        assert rpt["LoA"].str.fullmatch(r"-?\d+\.\d+ to -?\d+\.\d+").all()
        center = ssa.summary(ci_method=None)
        for stat in ssa.sleep_statistics:
            bias = center.at[stat, ("bias_mean", "center")]
            assert rpt.at[_label(rpt, stat), "Bias"] == f"{bias:.2f}"

    def test_regr_format(self, ssa):
        rpt = ssa.report(bias_method="regr", loa_method="regr", ci_method=None)
        assert rpt["Bias"].str.fullmatch(r"-?\d+\.\d+ \+ -?\d+\.\d+x").all()
        assert rpt["LoA"].str.fullmatch(r"±\d+\.\d+ \(-?\d+\.\d+ \+ -?\d+\.\d+x\)").all()

    def test_regr_bias_param_loa_uses_residual_halfwidth(self, ssa):
        # Menghini et al. (2021) eq. 2: LoA parallel to the regression bias line
        rpt = ssa.report(bias_method="regr", loa_method="param", ci_method="param", decimals=2)
        s = ssa.summary(ci_method="param")
        for stat in ssa.sleep_statistics:
            hw = s.loc[stat, "loa_halfwidth"]
            expected = f"bias ± {hw['center']:.2f} [{hw['lower']:.2f}, {hw['upper']:.2f}]"
            assert rpt.at[_label(rpt, stat), f"LoA [{PCT}% CI]"] == expected

    def test_sleep_stats_subset_and_order(self, ssa):
        rpt = ssa.report(ci_method="param", sleep_stats=["WASO", "TST", "SE"])
        assert rpt.index.tolist() == ["WASO (min)", "TST (min)", "SE (%)"]
        pd.testing.assert_frame_equal(rpt, ssa.report(ci_method="param").loc[rpt.index])

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(ci_method="param", sleep_stats=["NOT_A_STAT"]),
            dict(decimals=-1),
            dict(bias_method="invalid"),
            dict(ci_method="invalid"),
        ],
    )
    def test_invalid_args_raise(self, ssa, kwargs):
        with pytest.raises(AssertionError):
            ssa.report(**kwargs)


class TestSleepStatsAgreementPlotBlandAltman:
    """Use ci_method="param" to avoid the bootstrap path with small samples (N_SESSIONS=5).

    On each axis, ``ax.lines`` holds the y=0 reference line, then the bias line, then the
    upper and lower LoA lines.
    """

    @pytest.fixture(scope="class")
    def center(self, ssa):
        return ssa.summary(ci_method=None).xs("center", level="interval", axis=1)

    def test_returns_facetgrid_with_one_axis_per_stat(self, ssa):
        import seaborn as sns

        g = ssa.plot_blandaltman(ci_method="param")
        assert isinstance(g, sns.FacetGrid)
        assert len(g.axes.flat) == len(ssa.sleep_statistics)
        subset = ssa.sleep_statistics[:3]
        assert len(ssa.plot_blandaltman(sleep_stats=subset, ci_method="param").axes.flat) == 3

    def test_param_bias_param_loa_lines(self, ssa, center):
        g = ssa.plot_blandaltman(bias_method="param", loa_method="param", ci_method="param")
        for stat, ax in zip(ssa.sleep_statistics, g.axes.flat, strict=True):
            assert len(ax.lines) == 4
            bias, lower, upper = (line.get_ydata()[0] for line in ax.lines[1:4])
            assert np.isclose(bias, center.at[stat, "bias_mean"])
            assert np.isclose(lower, center.at[stat, "loa_lower"])
            assert np.isclose(upper, center.at[stat, "loa_upper"])

    def test_regr_bias_regr_loa_lines(self, ssa, center):
        g = ssa.plot_blandaltman(bias_method="regr", loa_method="regr", ci_method="param")
        for stat, ax in zip(ssa.sleep_statistics, g.axes.flat, strict=True):
            assert len(ax.lines) == 4
            c = center.loc[stat]
            x = ax.lines[1].get_xdata()
            bias, upper, lower = (line.get_ydata() for line in ax.lines[1:4])
            np.testing.assert_allclose(bias, c["bias_intercept"] + c["bias_slope"] * x)
            spread = (
                1.96 * np.sqrt(np.pi / 2) * np.maximum(0, c["loa_intercept"] + c["loa_slope"] * x)
            )
            np.testing.assert_allclose(upper - bias, spread)
            np.testing.assert_allclose(bias - lower, spread)

    def test_regr_bias_param_loa_parallel_to_bias(self, ssa, center):
        # Menghini et al. (2021) eq. 2: constant LoA drawn parallel to the regression bias line
        g = ssa.plot_blandaltman(bias_method="regr", loa_method="param", ci_method="param")
        for stat, ax in zip(ssa.sleep_statistics, g.axes.flat, strict=True):
            bias, upper, lower = (line.get_ydata() for line in ax.lines[1:4])
            hw = center.at[stat, "loa_halfwidth"]
            assert np.allclose(upper - bias, hw) and np.allclose(bias - lower, hw)
            assert len(ax.collections) == 3  # scatter + two CI bands

    def test_no_ci_bands(self, ssa):
        g = ssa.plot_blandaltman(ci_method=None)
        for ax in g.axes.flat:
            assert len(ax.patches) == 0 and len(ax.collections) == 1  # scatter only

    def test_param_ci_bands(self, ssa):
        g = ssa.plot_blandaltman(bias_method="param", loa_method="param", ci_method="param")
        for ax in g.axes.flat:
            assert len(ax.patches) == 3  # axhspan for bias and both LoA

    def test_axis_labels(self, ssa):
        g = ssa.plot_blandaltman(ci_method="param")
        assert g.axes.flat[-1].get_xlabel() == REF_SCORER
        assert g.axes.flat[0].get_ylabel() == f"{OBS_SCORER} - {REF_SCORER}"

    def test_kwargs_passthrough(self, ssa):
        from matplotlib.colors import to_rgba

        g = ssa.plot_blandaltman(ci_method="param", scatter_kwargs={"edgecolor": "red"}, col_wrap=1)
        assert g._col_wrap == 1
        scatter = g.axes.flat[0].collections[0]
        np.testing.assert_allclose(scatter.get_edgecolor()[0][:3], to_rgba("red")[:3])

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(bias_method="invalid"),
            dict(loa_method="invalid"),
            dict(ci_method="invalid"),
        ],
    )
    def test_invalid_args_raise(self, ssa, kwargs):
        with pytest.raises(AssertionError):
            ssa.plot_blandaltman(**kwargs)


class TestSleepStatsAgreementLogTransform:
    """Tests for the log_transform=True path (Euser et al. 2008)."""

    def test_negative_values_raise(self, ref_stats, obs_stats):
        bad_ref = ref_stats.copy()
        bad_ref.iloc[0, 0] = -1.0
        with pytest.raises(ValueError, match="non-negative"):
            SleepStatsAgreement(bad_ref, obs_stats, log_transform=True)

    def test_loa_log_slope_values(self, ssa_log, ref_stats, obs_stats):
        slope = _log_slope(ssa_log)
        assert set(slope.index) == set(ssa_log.sleep_statistics)
        for stat in ssa_log.sleep_statistics:
            valid = ref_stats[stat].notna() & obs_stats[stat].notna()
            ref, obs = ref_stats.loc[valid, stat], obs_stats.loc[valid, stat]
            if (ref == 0).any() or (obs == 0).any():
                # Stats with zeros are not log-transformed
                assert np.isnan(slope[stat])
                continue
            z = 1.96 * np.std(np.log(obs) - np.log(ref), ddof=1)
            expected = 2 * (np.exp(z) - 1) / (np.exp(z) + 1)
            assert np.isclose(slope[stat], expected) and slope[stat] >= 0
        assert SleepStatsAgreement._euser_slope_scalar(0.0, 1.96) == 0.0

    def test_loa_log_slope_param_ci(self, ssa_log, log_stats):
        # The SE of a limit, sqrt(3 * SD^2 / n), is added on the z = 1.96 * SD scale, as in the
        # reference pipeline of Menghini et al. (2021). The lower bound is clamped at 0.
        ci = ssa_log._loa_log_ci
        for stat in log_stats:
            d = np.log(ssa_log.data.loc[stat, OBS_SCORER] / ssa_log.data.loc[stat, REF_SCORER])
            sd, n = d.std(ddof=1), d.size
            t = sps.t.ppf((1 + ssa_log._confidence) / 2, n - 1)
            z = 1.96 * sd + np.array([-1, 1]) * t * np.sqrt(3 * sd**2 / n)
            expected = 2 * (np.exp(z.clip(0)) - 1) / (np.exp(z.clip(0)) + 1)
            np.testing.assert_allclose(ci.loc[stat, ["param_lower", "param_upper"]], expected)

    def test_summary_log_slope_column(self, ssa_log):
        s = ssa_log.summary(ci_method="param")["loa_log_slope"].dropna()
        assert list(s.columns) == ["center", "lower", "upper"]
        assert (s["lower"] < s["center"]).all() and (s["center"] < s["upper"]).all()
        assert list(ssa_log.summary(ci_method=None)["loa_log_slope"].columns) == ["center"]

    @pytest.mark.parametrize("loa_method", ["log", "auto"])
    def test_report_log_format(self, ssa_log, log_stats, loa_method):
        s = ssa_log.summary(ci_method="param")["loa_log_slope"]
        rpt = ssa_log.report(
            loa_method=loa_method, ci_method="param", decimals=2, sleep_stats=log_stats
        )
        for stat in log_stats:
            hw = s.loc[stat]
            expected = f"bias ± {hw['center']:.2f} × ref [{hw['lower']:.2f}, {hw['upper']:.2f}]"
            assert rpt.at[_label(rpt, stat), f"LoA [{PCT}% CI]"] == expected

    def test_report_log_format_no_ci(self, ssa_log, log_stats):
        rpt = ssa_log.report(loa_method="log", ci_method=None, sleep_stats=log_stats)
        assert rpt["LoA"].str.fullmatch(r"bias ± \d+\.\d+ × ref").all()

    def test_stats_with_zeros_are_not_log_transformed(self, ssa_log, log_stats):
        zero_stats = [s for s in ssa_log.sleep_statistics if s not in log_stats]
        assert zero_stats, "the fixture needs at least one stat with a zero value"
        # Regular LoA are used and reported for these stats, and loa_method="log" refuses them
        rpt = ssa_log.report(ci_method="param", sleep_stats=zero_stats)
        assert not rpt[f"LoA [{PCT}% CI]"].str.contains("×").any()
        with pytest.raises(ValueError, match="zero values"):
            ssa_log.report(loa_method="log", ci_method="param", sleep_stats=zero_stats[:1])

    @pytest.mark.parametrize("loa_method", ["param", "regr"])
    def test_loa_method_override_with_log_transform(self, ssa_log, loa_method):
        # loa_method="param"/"regr" force constant/regression LoA even when log_transform=True
        rpt = ssa_log.report(loa_method=loa_method, ci_method="param")
        assert not rpt[f"LoA [{PCT}% CI]"].str.contains("×").any()

    def test_loa_log_without_log_transform_raises(self, ssa):
        with pytest.raises(ValueError):
            ssa.report(loa_method="log")
        with pytest.raises(ValueError):
            ssa.plot_blandaltman(loa_method="log")

    def test_plot_log_lines(self, ssa_log, log_stats):
        g = ssa_log.plot_blandaltman(loa_method="log", ci_method="param", sleep_stats=log_stats)
        for stat, ax in zip(log_stats, g.axes.flat, strict=True):
            assert len(ax.lines) == 4
            x = np.asarray(ax.lines[2].get_xdata())
            bias, upper, lower = (np.asarray(line.get_ydata()) for line in ax.lines[1:4])
            spread = _log_slope(ssa_log)[stat] * x
            np.testing.assert_allclose(upper - bias, spread)
            np.testing.assert_allclose(bias - lower, spread)
            assert len(ax.collections) == 3  # scatter + two CI bands

    def test_plot_log_no_ci(self, ssa_log, log_stats):
        g = ssa_log.plot_blandaltman(loa_method="log", ci_method=None, sleep_stats=log_stats)
        for ax in g.axes.flat:
            assert len(ax.patches) == 0 and len(ax.collections) == 1

    def test_ci_auto(self, ssa_log_large):
        # ci_method="auto" picks "param" when normality holds, "boot" otherwise. Use ssa_log_large
        # (15 sessions, BCa) to avoid degenerate all-zero stats with N=5.
        stats = _log_slope(ssa_log_large).dropna().index.tolist()
        rpt = ssa_log_large.report(loa_method="log", ci_method="auto", sleep_stats=stats)
        assert rpt[f"LoA [{PCT}% CI]"].str.contains("×").all()
        g = ssa_log_large.plot_blandaltman(loa_method="log", ci_method="auto", sleep_stats=stats)
        assert len(g.axes.flat) == len(stats)

    def test_ci_boot(self, stats_large):
        # Exercise the _generate_bootstrap_ci Euser path with method="basic" to avoid BCa
        # degenerate-data warnings/NaNs when a stat has identical values across all sessions.
        fresh = SleepStatsAgreement(
            *stats_large,
            ref_scorer=REF_SCORER,
            obs_scorer=OBS_SCORER,
            log_transform=True,
            bootstrap_kwargs={"n_resamples": 200, "method": "basic"},
        )
        stats = _log_slope(fresh).dropna().index.tolist()
        rpt = fresh.report(loa_method="log", ci_method="boot", sleep_stats=stats)
        assert rpt[f"LoA [{PCT}% CI]"].str.contains("×").all()
        s = fresh.summary(ci_method="boot")["loa_log_slope"]
        assert s.loc[stats, ["lower", "upper"]].notna().all().all()
        g = fresh.plot_blandaltman(loa_method="log", ci_method="boot", sleep_stats=stats)
        assert all(len(ax.collections) == 3 for ax in g.axes.flat)
