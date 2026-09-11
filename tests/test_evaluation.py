"""Tests for yasa/evaluation.py — EpochByEpochAgreement and SleepStatsAgreement."""

import re
import unittest
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.stats as sps

from yasa.evaluation import EpochByEpochAgreement, SleepStatsAgreement
from yasa.hypno import Hypnogram, simulate_hypnogram

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_SESSIONS = 5
REF_SCORER = "Human"
OBS_SCORER = "YASA"

ref_hyps = [simulate_hypnogram(tib=90, scorer=REF_SCORER, seed=i) for i in range(N_SESSIONS)]
obs_hyps = [h.simulate_similar(scorer=OBS_SCORER, seed=i) for i, h in enumerate(ref_hyps)]
ebe = EpochByEpochAgreement(ref_hyps, obs_hyps)

# Single-night variant (via Hypnogram.evaluate)
ebe_single = ref_hyps[0].evaluate(obs_hyps[0])

# Two-session fixture where session 1 has no N1 and no N3 in the reference hypnogram, but the
# observed scorer assigns a few epochs of each -> recall (and fbeta) undefined for those stages.
_ref_missing = Hypnogram(["WAKE"] * 10 + ["N2"] * 20 + ["REM"] * 10, scorer=REF_SCORER)
_obs_missing = Hypnogram(
    ["WAKE"] * 8 + ["N1"] * 2 + ["N2"] * 18 + ["N3"] * 2 + ["REM"] * 10, scorer=OBS_SCORER
)
ebe_missing = EpochByEpochAgreement([_ref_missing, ref_hyps[0]], [_obs_missing, obs_hyps[0]])

_FMT_CI = r"\d+\.\d \(\d+\.\d\) \[\d+\.\d, \d+\.\d\]"
_FMT_NOCI = r"\d+\.\d \(\d+\.\d\)"


class TestEpochByEpochAgreementInit(unittest.TestCase):
    def test_attributes_and_dict_input(self):
        assert REF_SCORER in repr(ebe) and OBS_SCORER in repr(ebe)
        assert (ebe.ref_scorer, ebe.obs_scorer) == (REF_SCORER, OBS_SCORER)
        assert ebe.n_sessions == N_SESSIONS
        assert ebe.data.shape[1] == 2
        ref_dict = {f"night{i}": h for i, h in enumerate(ref_hyps)}
        obs_dict = {f"night{i}": h for i, h in enumerate(obs_hyps)}
        assert EpochByEpochAgreement(ref_dict, obs_dict).n_sessions == N_SESSIONS

    def test_single_night_via_evaluate(self):
        assert ebe_single.n_sessions == 1
        assert (ebe_single.ref_scorer, ebe_single.obs_scorer) == (REF_SCORER, OBS_SCORER)

    def test_invalid_inputs_raise(self):
        with pytest.raises(AssertionError):
            EpochByEpochAgreement(ref_hyps, obs_hyps[:-1])
        same = [h.simulate_similar(scorer=REF_SCORER, seed=i) for i, h in enumerate(ref_hyps)]
        with pytest.raises(AssertionError):
            EpochByEpochAgreement(ref_hyps, same)
        no_scorer = [simulate_hypnogram(tib=90, seed=i) for i in range(N_SESSIONS)]
        with pytest.raises(AssertionError):
            EpochByEpochAgreement(ref_hyps, no_scorer)


class TestGetAgreement(unittest.TestCase):
    def test_shape_and_columns(self):
        agr = ebe.get_agreement()
        assert isinstance(agr, pd.DataFrame) and agr.shape[0] == N_SESSIONS
        expected_cols = {"accuracy", "balanced_acc", "kappa", "mcc", "precision", "f1"}
        assert set(agr.columns) == expected_cols

    def test_single_night_returns_series(self):
        assert isinstance(ebe_single.get_agreement(), pd.Series)

    def test_scorers_list_and_sample_weight(self):
        default = ebe.get_agreement()
        # Metric names map to sklearn.metrics.<name>_score (returned as is, i.e. not in percent)
        agr = ebe.get_agreement(scorers=["accuracy", "cohen_kappa"])
        np.testing.assert_allclose(100 * agr["accuracy"], default["accuracy"])
        np.testing.assert_allclose(agr["cohen_kappa"], default["kappa"])
        # Uniform sample weights leave the scores unchanged
        weighted = ebe.get_agreement(sample_weight=pd.Series(2.0, index=ebe.data.index))
        pd.testing.assert_frame_equal(weighted, default)


class TestGetAgreementByStage(unittest.TestCase):
    def test_structure(self):
        agr = ebe.get_agreement_bystage()
        assert isinstance(agr, pd.DataFrame)
        assert set(agr.columns) == {"fbeta", "npv", "precision", "recall", "specificity", "support"}
        assert agr.index.names == ["stage", "sleep_id"]

    def test_single_night_no_sleep_id_level(self):
        assert ebe_single.get_agreement_bystage().index.name == "stage"


class TestGetAgreementByStageZeroDivision(unittest.TestCase):
    """Test the zero_division parameter of get_agreement_bystage."""

    def test_default_is_nan_for_absent_reference_stage(self):
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

    def test_zero_division_zero_restores_old_behavior(self):
        agr = ebe_missing.get_agreement_bystage(zero_division=0)
        assert not agr.isna().any().any()
        assert agr.at[("N1", 1), "recall"] == 0
        assert agr.at[("N3", 1), "recall"] == 0

    def test_other_zero_division_values(self):
        from sklearn.exceptions import UndefinedMetricWarning

        assert ebe_missing.get_agreement_bystage(zero_division=1).at[("N1", 1), "recall"] == 100
        with pytest.warns(UndefinedMetricWarning):
            agr = ebe_missing.get_agreement_bystage(zero_division="warn")
        assert agr.at[("N1", 1), "recall"] == 0
        for bad in [0.5, "nan"]:
            with pytest.raises(AssertionError):
                ebe.get_agreement_bystage(zero_division=bad)

    def test_nan_excluded_from_summary(self):
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


class TestSummaryBootCI(unittest.TestCase):
    """Bootstrap confidence intervals for EpochByEpochAgreement.summary()."""

    # A 20-session object, the threshold at which the BCa small-n warning stops being emitted.
    # `tib` must be long enough for every stage to occur, otherwise `simulate_similar` cannot
    # build a transition matrix.
    _ref_large = [simulate_hypnogram(tib=480, scorer=REF_SCORER, seed=i) for i in range(20)]
    _obs_large = [h.simulate_similar(scorer=OBS_SCORER, seed=i) for i, h in enumerate(_ref_large)]
    ebe_large = EpochByEpochAgreement(_ref_large, _obs_large)

    @staticmethod
    def _summary(obj, **kwargs):
        """Call summary() with the BCa small-n warning silenced."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return obj.summary(**kwargs)

    def test_no_ci_by_default(self):
        # `ci_method=None` is the default, so the historical output is unchanged
        default = ebe.summary()
        assert not {"ci_lower", "ci_upper"} & set(default.columns)
        pd.testing.assert_frame_equal(default, ebe.summary(ci_method=None))
        pd.testing.assert_frame_equal(
            ebe.summary(by_stage=True), ebe.summary(by_stage=True, ci_method=None)
        )

    def test_ci_brackets_mean(self):
        for by_stage in [False, True]:
            for method in ["BCa", "basic", "percentile"]:
                kwargs = {"n_resamples": 200, "rng": 0, "method": method}
                out = self._summary(
                    ebe, by_stage=by_stage, ci_method="boot", bootstrap_kwargs=kwargs
                )
                msg = f"by_stage={by_stage}, method={method}"
                assert out.columns[-2:].tolist() == ["ci_lower", "ci_upper"], msg
                # Metrics defined in at least one session must have a finite CI around the mean
                valid = out.dropna(subset=["ci_lower", "ci_upper"])
                assert not valid.empty, msg
                assert (valid["ci_lower"] <= valid["mean"] + 1e-9).all(), msg
                assert (valid["ci_upper"] >= valid["mean"] - 1e-9).all(), msg

    def test_bca_is_the_default_method(self):
        default = self._summary(
            ebe, ci_method="boot", bootstrap_kwargs={"n_resamples": 200, "rng": 0}
        )
        bca = self._summary(
            ebe, ci_method="boot", bootstrap_kwargs={"n_resamples": 200, "rng": 0, "method": "BCa"}
        )
        pd.testing.assert_frame_equal(default, bca)

    def test_reproducible_with_rng(self):
        kwargs = {"n_resamples": 200, "rng": 3}
        out = self._summary(ebe, ci_method="boot", bootstrap_kwargs=kwargs)
        pd.testing.assert_frame_equal(
            out, self._summary(ebe, ci_method="boot", bootstrap_kwargs=kwargs)
        )
        other = self._summary(
            ebe, ci_method="boot", bootstrap_kwargs={"n_resamples": 200, "rng": 4}
        )
        assert not out["ci_lower"].equals(other["ci_lower"])

    def test_matches_manual_session_bootstrap(self):
        # Re-implement the basic participant bootstrap by hand with the same rng. Also checks that
        # the CI is NOT clipped, unlike get_confusion_matrix_proportional.
        n_resamples = 100
        out = self._summary(
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

    def test_undefined_metrics_and_support(self):
        # Session 1 of `ebe_missing` has no N1 and no N3 in the reference hypnogram. Other tests
        # share this fixture and cache `zero_division=0` scores on it, so re-compute the default
        # (NaN) scores rather than depending on test order.
        ebe_missing.get_agreement_bystage()
        out = self._summary(
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

    def test_small_n_bca_warning(self):
        assert ebe.n_sessions < 20
        with pytest.warns(RuntimeWarning, match="BCa"):
            ebe.summary(ci_method="boot", bootstrap_kwargs={"n_resamples": 50})
        # Only BCa is affected, and only below the threshold
        for obj, kwargs in [
            (ebe, {"n_resamples": 50, "method": "percentile"}),
            (ebe, {"n_resamples": 50, "method": "basic"}),
            (self.ebe_large, {"n_resamples": 50}),
        ]:
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                obj.summary(ci_method="boot", bootstrap_kwargs=kwargs)
            assert not [w for w in rec if "BCa" in str(w.message)], kwargs
        # No bootstrap at all means no warning
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            ebe.summary()
        assert not [w for w in rec if "BCa" in str(w.message)]

    def test_invalid_args_raise(self):
        bad_kwargs = [
            dict(ci_method="param"),
            dict(ci_method=True),
            dict(confidence=95),
            dict(confidence=0),
            dict(bootstrap_kwargs={"confidence_level": 0.9}),
            dict(bootstrap_kwargs={"method": "bca"}),
            dict(bootstrap_kwargs={"n_resamples": 0}),
            dict(bootstrap_kwargs=[]),
        ]
        for kwargs in bad_kwargs:
            with pytest.raises(AssertionError):
                ebe.summary(**kwargs)
        with pytest.raises(AssertionError):
            ebe_single.summary(ci_method="boot")


class TestGetConfusionMatrixProportional(unittest.TestCase):
    def test_structure(self):
        out = ebe.get_confusion_matrix_proportional(ci_method="param")
        assert out.index.names == [REF_SCORER, OBS_SCORER]
        assert list(out.columns) == ["mean", "std", "ci_lower", "ci_upper", "n_sessions"]
        n_stages = ebe.get_confusion_matrix(sleep_id=1).shape[0]
        assert len(out) == n_stages**2
        out = ebe.get_confusion_matrix_proportional(ci_method=None)
        assert list(out.columns) == ["mean", "std", "n_sessions"]

    def test_matches_manual_computation(self):
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

    def test_boot_ci(self):
        for method in ["BCa", "basic", "percentile"]:
            kwargs = {"n_resamples": 200, "rng": 0, "method": method}
            out = ebe.get_confusion_matrix_proportional(ci_method="boot", bootstrap_kwargs=kwargs)
            # Rows with at least one contributing session must have a finite CI around the mean
            valid = out[out["n_sessions"] > 0]
            assert valid[["ci_lower", "ci_upper"]].notna().all().all(), method
            assert (valid["ci_lower"] <= valid["mean"] + 1e-9).all(), method
            assert (valid["ci_upper"] >= valid["mean"] - 1e-9).all(), method
            assert (valid["ci_lower"] >= 0).all() and (valid["ci_upper"] <= 100).all(), method
            # Reproducible with a fixed rng
            out2 = ebe.get_confusion_matrix_proportional(ci_method="boot", bootstrap_kwargs=kwargs)
            pd.testing.assert_frame_equal(out, out2)
        # BCa is the default method
        default = ebe.get_confusion_matrix_proportional(
            bootstrap_kwargs={"n_resamples": 200, "rng": 0}
        )
        bca = ebe.get_confusion_matrix_proportional(
            bootstrap_kwargs={"n_resamples": 200, "rng": 0, "method": "BCa"}
        )
        pd.testing.assert_frame_equal(default, bca)

    def test_bca_degenerate_cells(self):
        # Two identical session pairs -> every cell is constant across resamples, so the BCa
        # interval must collapse onto the mean (percentile fallback) rather than being NaN.
        ebe_const = EpochByEpochAgreement([ref_hyps[0], ref_hyps[0]], [obs_hyps[0], obs_hyps[0]])
        out = ebe_const.get_confusion_matrix_proportional(bootstrap_kwargs={"n_resamples": 50})
        valid = out[out["n_sessions"] > 0]
        np.testing.assert_allclose(valid["ci_lower"], valid["mean"])
        np.testing.assert_allclose(valid["ci_upper"], valid["mean"])

    def test_boot_ci_matches_manual_basic_bootstrap(self):
        # Re-implement the basic participant bootstrap by hand with the same rng
        import warnings

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

    def test_formatted(self):
        fmt = ebe.get_confusion_matrix_proportional(ci_method="param", formatted=True)
        assert fmt.index.name == REF_SCORER and fmt.columns.name == OBS_SCORER
        assert fmt.shape[0] == fmt.shape[1]
        assert fmt.map(lambda s: bool(re.fullmatch(_FMT_CI, s))).all().all()
        fmt_noci = ebe.get_confusion_matrix_proportional(ci_method=None, formatted=True)
        assert fmt_noci.map(lambda s: bool(re.fullmatch(_FMT_NOCI, s))).all().all()

    def test_absent_stage_is_nan_not_zero(self):
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

    def test_invalid_args_raise(self):
        bad_kwargs = [
            dict(bootstrap_kwargs={"confidence_level": 0.9}),
            dict(bootstrap_kwargs={"method": "bca"}),
            dict(bootstrap_kwargs={"n_resamples": 0}),
            dict(ci_method="invalid"),
            dict(confidence=95),
            dict(decimals=-1),
        ]
        for kwargs in bad_kwargs:
            with pytest.raises(AssertionError):
                ebe.get_confusion_matrix_proportional(**kwargs)
        with pytest.raises(AssertionError):
            ebe_single.get_confusion_matrix_proportional()


class TestGetConfusionMatrix(unittest.TestCase):
    def test_single_session(self):
        cm = ebe.get_confusion_matrix(sleep_id=1)
        assert cm.index.name == REF_SCORER and cm.columns.name == OBS_SCORER
        assert cm.values.sum() == ref_hyps[0].n_epochs
        with pytest.raises(AssertionError):
            ebe.get_confusion_matrix(sleep_id=999)

    def test_all_sessions_and_agg_sum(self):
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


class TestGetSleepStats(unittest.TestCase):
    def test_structure(self):
        sstats = ebe.get_sleep_stats()
        assert sstats.index.names == ["scorer", "sleep_id"]
        assert set(sstats.index.get_level_values("scorer")) == {REF_SCORER, OBS_SCORER}
        assert len(sstats) == 2 * N_SESSIONS

    def test_single_night(self):
        assert set(ebe_single.get_sleep_stats().index) == {REF_SCORER, OBS_SCORER}


# ---------------------------------------------------------------------------
# SleepStatsAgreement shared fixtures
# ---------------------------------------------------------------------------

_sstats = ebe.get_sleep_stats()
_ref_stats = _sstats.loc[REF_SCORER]
_obs_stats = _sstats.loc[OBS_SCORER]
ssa = SleepStatsAgreement(_ref_stats, _obs_stats, ref_scorer=REF_SCORER, obs_scorer=OBS_SCORER)
ssa_log = SleepStatsAgreement(
    _ref_stats, _obs_stats, ref_scorer=REF_SCORER, obs_scorer=OBS_SCORER, log_transform=True
)

# Larger fixture (15 sessions) to avoid degenerate all-zero stats that make BCa fail with N=5.
# Used for ci_method="auto" and ci_method="boot" tests.
_N_LARGE = 15
_ref_hyps_large = [
    simulate_hypnogram(tib=90, scorer=REF_SCORER, seed=i + 100) for i in range(_N_LARGE)
]
_obs_hyps_large = [
    h.simulate_similar(scorer=OBS_SCORER, seed=i + 100) for i, h in enumerate(_ref_hyps_large)
]
_sstats_large = EpochByEpochAgreement(_ref_hyps_large, _obs_hyps_large).get_sleep_stats()
_ref_stats_large = _sstats_large.loc[REF_SCORER]
_obs_stats_large = _sstats_large.loc[OBS_SCORER]
ssa_log_large = SleepStatsAgreement(
    _ref_stats_large,
    _obs_stats_large,
    ref_scorer=REF_SCORER,
    obs_scorer=OBS_SCORER,
    log_transform=True,
    bootstrap_kwargs={"n_resamples": 200},
)

PCT = int(ssa._confidence * 100)
# Stats that can be log-transformed (no zero value in either scorer)


def _log_slope(obj):
    """Euser LoA slope per statistic (NaN for statistics that are not log-transformed)."""
    return obj.summary(ci_method=None)["loa_log_slope"]["center"]


LOG_STATS = _log_slope(ssa_log).dropna().index.tolist()


def _valid_arrays(stat):
    """Reference and difference arrays for `stat`, dropping sessions with a NaN value.

    Sessions with a NaN value are dropped by the SleepStatsAgreement constructor (pivot_table).
    """
    valid = _ref_stats[stat].notna() & _obs_stats[stat].notna()
    ref = _ref_stats.loc[valid, stat].to_numpy()
    diff = (_obs_stats.loc[valid, stat] - _ref_stats.loc[valid, stat]).to_numpy()
    return ref, diff


def _label(rpt, stat):
    """Report index label ("STAT (unit)") for a sleep statistic."""
    return rpt.index[rpt.index.str.startswith(f"{stat} (")][0]


class TestSleepStatsAgreementInit(unittest.TestCase):
    def test_attributes(self):
        assert REF_SCORER in repr(ssa) and OBS_SCORER in repr(ssa)
        assert (ssa.ref_scorer, ssa.obs_scorer) == (REF_SCORER, OBS_SCORER)
        assert ssa.n_sessions == N_SESSIONS
        assert isinstance(ssa.sleep_statistics, list) and len(ssa.sleep_statistics) > 0
        assert ssa.data.shape[1] == 2
        ssa_default = SleepStatsAgreement(_ref_stats, _obs_stats)
        assert (ssa_default.ref_scorer, ssa_default.obs_scorer) == ("Reference", "Observed")
        # Built directly from EpochByEpochAgreement.get_sleep_stats(): scorers taken from the index
        ssa_multi = SleepStatsAgreement(_sstats)
        assert (ssa_multi.ref_scorer, ssa_multi.obs_scorer) == (REF_SCORER, OBS_SCORER)
        pd.testing.assert_frame_equal(
            ssa_multi.summary(ci_method=None), ssa.summary(ci_method=None)
        )

    def test_invalid_inputs_raise(self):
        bad_index = _obs_stats.copy()
        bad_index.index = bad_index.index + 100
        bad_columns = _obs_stats.rename(columns={"TST": "TOTAL_SLEEP_TIME"})
        bad_args = [
            dict(ref_data=_ref_stats.to_numpy(), obs_data=_obs_stats),
            dict(ref_data=_ref_stats, obs_data=_obs_stats.to_numpy()),
            dict(ref_data=_ref_stats, obs_data=bad_index),
            dict(ref_data=_ref_stats, obs_data=bad_columns),
            dict(ref_data=_ref_stats, obs_data=_obs_stats, ref_scorer="X", obs_scorer="X"),
            dict(ref_data=_ref_stats, obs_data=_obs_stats, log_transform="TST"),
            dict(ref_data=_ref_stats, obs_data=_obs_stats, alpha=1.5),
            dict(ref_data=_ref_stats, obs_data=_obs_stats, effect_size_gates={"bad_key": 1}),
            dict(ref_data=_ref_stats, obs_data=_obs_stats, effect_size_gates={"r2": -1}),
            dict(ref_data=_ref_stats, obs_data=_obs_stats, effect_size_gates=0.1),
            dict(ref_data=_ref_stats, obs_data=_obs_stats, confidence=95),
        ]
        for kwargs in bad_args:
            with pytest.raises(AssertionError):
                SleepStatsAgreement(**kwargs)


class TestSleepStatsAgreementAssumptions(unittest.TestCase):
    def test_structure(self):
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

    def test_values_flags_and_methods(self):
        asmp = ssa.assumptions
        diff = _obs_stats["TST"] - _ref_stats["TST"]
        ttest = sps.ttest_1samp(diff, 0)
        assert np.isclose(asmp.at["TST", ("unbiased", "t")], ttest.statistic)
        assert np.isclose(asmp.at["TST", ("unbiased", "pvalue")], ttest.pvalue)
        assert np.isclose(asmp.at["TST", ("unbiased", "cohen_d")], diff.mean() / diff.std(ddof=1))
        assert np.isclose(asmp.at["TST", ("normal", "skew")], diff.skew())
        regr = sps.linregress(_ref_stats["TST"], diff)
        assert np.isclose(asmp.at["TST", ("constant_bias", "slope")], regr.slope)
        assert np.isclose(asmp.at["TST", ("constant_bias", "r2")], regr.rvalue**2)
        loa_regr = sps.linregress(
            _ref_stats["TST"], np.abs(diff - regr.intercept - regr.slope * _ref_stats["TST"])
        )
        fitted = loa_regr.intercept + loa_regr.slope * np.array(
            [_ref_stats["TST"].min(), _ref_stats["TST"].max()]
        )
        assert np.isclose(asmp.at["TST", ("homoscedastic", "sd_ratio")], fitted[1] / fitted[0])
        # Flags: pvalue >= alpha (0.05), or an immaterial effect size (see test_effect_size_gates)
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

    def test_alpha(self):
        # alpha=1 fails every test with p < 1 once the effect-size gates are disabled;
        # the statistics themselves are unchanged.
        ssa_strict = SleepStatsAgreement(
            _ref_stats,
            _obs_stats,
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

    def test_effect_size_gates(self):
        # Unreachable thresholds -> the effect size is never material -> nothing is ever
        # flagged as violated, even with alpha=1 (every test significant).
        ssa_lax = SleepStatsAgreement(
            _ref_stats,
            _obs_stats,
            ref_scorer=REF_SCORER,
            obs_scorer=OBS_SCORER,
            alpha=1.0,
            effect_size_gates={"skew": 1e9, "kurtosis": 1e9, "r2": 1.0, "sd_ratio": 1e9},
        )
        lax = ssa_lax.assumptions.xs("passed", level="metric", axis=1)
        assert lax[["normal", "constant_bias", "homoscedastic"]].all().all()
        # `unbiased` is not gated on an effect size, so it still fails
        assert not lax["unbiased"].any()
        # The resolved thresholds are exposed, with the unspecified ones left at their default
        ssa_partial = SleepStatsAgreement(
            _ref_stats,
            _obs_stats,
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
        # Dual criterion: a statistic fails only if it is both significant and material
        asmp = ssa.assumptions
        for name, gate in [("constant_bias", "r2"), ("homoscedastic", "sd_ratio")]:
            effect = asmp[(name, gate)]
            if gate == "sd_ratio":
                effect = np.maximum(effect, 1 / effect)
            failed = ~asmp[(name, "passed")]
            assert (
                failed == (asmp[(name, "pvalue")].lt(0.05) & effect.gt(ssa.effect_size_gates[gate]))
            ).all()


class TestSleepStatsAgreementSummary(unittest.TestCase):
    def test_matches_manual_computation(self):
        s = ssa.summary(ci_method="param")
        assert "loa_log_slope" not in s.columns.get_level_values("variable")
        for stat in ssa.sleep_statistics:
            ref, diff = _valid_arrays(stat)
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
        # Parametric CIs bracket the point estimates (NaN for stats with too few sessions)
        lower, center, upper = (
            s.xs(i, level="interval", axis=1) for i in ("lower", "center", "upper")
        )
        assert ((lower <= center + 1e-12) | lower.isna()).all().all()
        assert ((center <= upper + 1e-12) | upper.isna()).all().all()

    def test_missing_values_dropped_per_stat(self):
        # A NaN in one session of one stat only affects that stat, and inputs are not modified
        ref, obs = _ref_stats.rename_axis(None), _obs_stats.rename_axis(None)
        ref.loc[ref.index[0], "TST"] = np.nan
        ssa_nan = SleepStatsAgreement(ref, obs, ref_scorer=REF_SCORER, obs_scorer=OBS_SCORER)
        assert ref.index.name is None and obs.index.name is None
        assert len(ssa_nan.data.loc["TST"]) == N_SESSIONS - 1
        assert len(ssa_nan.data.loc["WASO"]) == N_SESSIONS
        assert np.isfinite(ssa_nan.summary(ci_method="param").loc["TST"]).all()

    def test_ci_method_none_returns_center_only(self):
        s = ssa.summary(ci_method=None)
        assert set(s.columns.get_level_values("interval")) == {"center"}
        full = ssa.summary(ci_method="param")
        pd.testing.assert_frame_equal(
            s, full.xs("center", axis=1, level="interval", drop_level=False)
        )

    def test_sleep_stats_subset_and_order(self):
        subset = ["WASO", "TST", "SE"]
        s = ssa.summary(ci_method="param", sleep_stats=subset)
        assert s.index.tolist() == subset
        pd.testing.assert_frame_equal(s, ssa.summary(ci_method="param").loc[subset])

    def test_invalid_args_raise(self):
        for kwargs in [
            dict(ci_method="invalid"),
            dict(ci_method="param", sleep_stats=["NOT_A_STAT"]),
            dict(ci_method="param", sleep_stats="TST"),
            dict(ci_method="param", sleep_stats=["TST", "TST"]),
        ]:
            with pytest.raises(AssertionError):
                ssa.summary(**kwargs)


class TestSleepStatsAgreementCalibrate(unittest.TestCase):
    """calibrate() requires all columns to be in ssa.sleep_statistics — stats with identical values
    across scorers (e.g. TIB) are removed during construction, so _obs_stats is subset first."""

    def test_calibrated_values(self):
        obs = _obs_stats[ssa.sleep_statistics]
        vals = ssa.summary(ci_method=None).xs("center", level="interval", axis=1)
        param = ssa.calibrate(obs, bias_method="param")
        assert isinstance(param, pd.DataFrame) and param.shape == obs.shape
        pd.testing.assert_frame_equal(param, obs - vals["bias_mean"], check_names=False)
        regr = ssa.calibrate(obs, bias_method="regr")
        expected = (obs - vals["bias_intercept"]) / (1 + vals["bias_slope"])
        pd.testing.assert_frame_equal(regr, expected, check_names=False)
        # "auto" keeps the column order and missing values of the input
        obs_nan = obs.copy()
        obs_nan.iloc[0, 0] = np.nan
        auto = ssa.calibrate(obs_nan, bias_method="auto")
        assert auto.columns.tolist() == obs.columns.tolist()
        assert np.isnan(auto.iloc[0, 0]) and auto.notna().sum().sum() == obs.notna().sum().sum() - 1

    def test_invalid_column_raises(self):
        bad = _obs_stats[ssa.sleep_statistics].rename(columns={"TST": "NOT_A_STAT"})
        with pytest.raises(AssertionError):
            ssa.calibrate(bad)


class TestSleepStatsAgreementReport(unittest.TestCase):
    """Use ci_method="param" to avoid the bootstrap path with small samples (N_SESSIONS=5)."""

    def test_columns(self):
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

    def test_mean_sd_columns(self):
        rpt = ssa.report(ci_method="param", decimals=2)
        pattern = r"-?\d+\.\d{2} \(\d+\.\d{2}\)"
        for scorer, data in [(REF_SCORER, _ref_stats), (OBS_SCORER, _obs_stats)]:
            col = rpt[f"{scorer} mean (SD)"]
            assert col.str.fullmatch(pattern).all()
            assert col["TST (min)"] == f"{data['TST'].mean():.2f} ({data['TST'].std(ddof=1):.2f})"

    def test_ci_columns_contain_brackets(self):
        rpt = ssa.report(bias_method="param", loa_method="param", ci_method="param")
        num = r"-?\d+\.\d+"
        assert rpt[f"Bias [{PCT}% CI]"].str.fullmatch(rf"{num} \[{num}, {num}\]").all()
        assert (
            rpt[f"LoA [{PCT}% CI]"]
            .str.fullmatch(rf"{num} to {num} \[{num}, {num}; {num}, {num}\]")
            .all()
        )

    def test_no_ci(self):
        rpt = ssa.report(bias_method="param", loa_method="param", ci_method=None)
        assert "Bias" in rpt.columns and "LoA" in rpt.columns
        assert not any("CI" in c for c in rpt.columns)
        assert rpt["LoA"].str.fullmatch(r"-?\d+\.\d+ to -?\d+\.\d+").all()
        center = ssa.summary(ci_method=None)
        for stat in ssa.sleep_statistics:
            bias = center.at[stat, ("bias_mean", "center")]
            assert rpt.at[_label(rpt, stat), "Bias"] == f"{bias:.2f}"

    def test_regr_format(self):
        rpt = ssa.report(bias_method="regr", loa_method="regr", ci_method=None)
        assert rpt["Bias"].str.fullmatch(r"-?\d+\.\d+ \+ -?\d+\.\d+x").all()
        assert rpt["LoA"].str.fullmatch(r"±\d+\.\d+ \(-?\d+\.\d+ \+ -?\d+\.\d+x\)").all()

    def test_regr_bias_param_loa_uses_residual_halfwidth(self):
        # Menghini et al. (2021) eq. 2: LoA parallel to the regression bias line
        rpt = ssa.report(bias_method="regr", loa_method="param", ci_method="param", decimals=2)
        s = ssa.summary(ci_method="param")
        for stat in ssa.sleep_statistics:
            hw = s.loc[stat, "loa_halfwidth"]
            expected = f"bias ± {hw['center']:.2f} [{hw['lower']:.2f}, {hw['upper']:.2f}]"
            assert rpt.at[_label(rpt, stat), f"LoA [{PCT}% CI]"] == expected

    def test_sleep_stats_subset_and_order(self):
        rpt = ssa.report(ci_method="param", sleep_stats=["WASO", "TST", "SE"])
        assert rpt.index.tolist() == ["WASO (min)", "TST (min)", "SE (%)"]
        pd.testing.assert_frame_equal(rpt, ssa.report(ci_method="param").loc[rpt.index])

    def test_invalid_args_raise(self):
        for kwargs in [
            dict(ci_method="param", sleep_stats=["NOT_A_STAT"]),
            dict(decimals=-1),
            dict(bias_method="invalid"),
            dict(ci_method="invalid"),
        ]:
            with pytest.raises(AssertionError):
                ssa.report(**kwargs)


class TestSleepStatsAgreementPlotBlandAltman(unittest.TestCase):
    """Use ci_method="param" to avoid the bootstrap path with small samples (N_SESSIONS=5).

    On each axis, ``ax.lines`` holds the y=0 reference line, then the bias line, then the
    upper and lower LoA lines.
    """

    @classmethod
    def setUpClass(cls):
        import matplotlib

        matplotlib.use("Agg")
        cls.center = ssa.summary(ci_method=None).xs("center", level="interval", axis=1)

    def test_returns_facetgrid_with_one_axis_per_stat(self):
        import seaborn as sns

        g = ssa.plot_blandaltman(ci_method="param")
        assert isinstance(g, sns.FacetGrid)
        assert len(g.axes.flat) == len(ssa.sleep_statistics)
        subset = ssa.sleep_statistics[:3]
        assert len(ssa.plot_blandaltman(sleep_stats=subset, ci_method="param").axes.flat) == 3

    def test_param_bias_param_loa_lines(self):
        g = ssa.plot_blandaltman(bias_method="param", loa_method="param", ci_method="param")
        for stat, ax in zip(ssa.sleep_statistics, g.axes.flat, strict=True):
            assert len(ax.lines) == 4
            bias, lower, upper = (line.get_ydata()[0] for line in ax.lines[1:4])
            assert np.isclose(bias, self.center.at[stat, "bias_mean"])
            assert np.isclose(lower, self.center.at[stat, "loa_lower"])
            assert np.isclose(upper, self.center.at[stat, "loa_upper"])

    def test_regr_bias_regr_loa_lines(self):
        g = ssa.plot_blandaltman(bias_method="regr", loa_method="regr", ci_method="param")
        for stat, ax in zip(ssa.sleep_statistics, g.axes.flat, strict=True):
            assert len(ax.lines) == 4
            c = self.center.loc[stat]
            x = ax.lines[1].get_xdata()
            bias, upper, lower = (line.get_ydata() for line in ax.lines[1:4])
            np.testing.assert_allclose(bias, c["bias_intercept"] + c["bias_slope"] * x)
            spread = (
                1.96 * np.sqrt(np.pi / 2) * np.maximum(0, c["loa_intercept"] + c["loa_slope"] * x)
            )
            np.testing.assert_allclose(upper - bias, spread)
            np.testing.assert_allclose(bias - lower, spread)

    def test_regr_bias_param_loa_parallel_to_bias(self):
        # Menghini et al. (2021) eq. 2: constant LoA drawn parallel to the regression bias line
        g = ssa.plot_blandaltman(bias_method="regr", loa_method="param", ci_method="param")
        for stat, ax in zip(ssa.sleep_statistics, g.axes.flat, strict=True):
            bias, upper, lower = (line.get_ydata() for line in ax.lines[1:4])
            hw = self.center.at[stat, "loa_halfwidth"]
            assert np.allclose(upper - bias, hw) and np.allclose(bias - lower, hw)
            assert len(ax.collections) == 3  # scatter + two CI bands

    def test_ci_bands(self):
        g = ssa.plot_blandaltman(ci_method=None)
        for ax in g.axes.flat:
            assert len(ax.patches) == 0 and len(ax.collections) == 1  # scatter only
        g = ssa.plot_blandaltman(bias_method="param", loa_method="param", ci_method="param")
        for ax in g.axes.flat:
            assert len(ax.patches) == 3  # axhspan for bias and both LoA

    def test_axis_labels(self):
        g = ssa.plot_blandaltman(ci_method="param")
        assert g.axes.flat[-1].get_xlabel() == REF_SCORER
        assert g.axes.flat[0].get_ylabel() == f"{OBS_SCORER} - {REF_SCORER}"

    def test_kwargs_passthrough(self):
        from matplotlib.colors import to_rgba

        g = ssa.plot_blandaltman(ci_method="param", scatter_kwargs={"edgecolor": "red"}, col_wrap=1)
        assert g._col_wrap == 1
        scatter = g.axes.flat[0].collections[0]
        np.testing.assert_allclose(scatter.get_edgecolor()[0][:3], to_rgba("red")[:3])

    def test_invalid_args_raise(self):
        for kwargs in [
            dict(bias_method="invalid"),
            dict(loa_method="invalid"),
            dict(ci_method="invalid"),
        ]:
            with pytest.raises(AssertionError):
                ssa.plot_blandaltman(**kwargs)


class TestSleepStatsAgreementLogTransform(unittest.TestCase):
    """Tests for the log_transform=True path (Euser et al. 2008)."""

    @classmethod
    def setUpClass(cls):
        import matplotlib

        matplotlib.use("Agg")

    def test_negative_values_raise(self):
        bad_ref = _ref_stats.copy()
        bad_ref.iloc[0, 0] = -1.0
        with pytest.raises(ValueError, match="non-negative"):
            SleepStatsAgreement(bad_ref, _obs_stats, log_transform=True)

    def test_loa_log_slope_values(self):
        slope = _log_slope(ssa_log)
        assert set(slope.index) == set(ssa_log.sleep_statistics)
        for stat in ssa_log.sleep_statistics:
            valid = _ref_stats[stat].notna() & _obs_stats[stat].notna()
            ref, obs = _ref_stats.loc[valid, stat], _obs_stats.loc[valid, stat]
            if (ref == 0).any() or (obs == 0).any():
                # Stats with zeros are not log-transformed
                assert np.isnan(slope[stat])
                continue
            z = 1.96 * np.std(np.log(obs) - np.log(ref), ddof=1)
            expected = 2 * (np.exp(z) - 1) / (np.exp(z) + 1)
            assert np.isclose(slope[stat], expected) and slope[stat] >= 0
        assert SleepStatsAgreement._euser_slope_scalar(0.0, 1.96) == 0.0

    def test_summary_log_slope_column(self):
        s = ssa_log.summary(ci_method="param")["loa_log_slope"].dropna()
        assert list(s.columns) == ["center", "lower", "upper"]
        assert (s["lower"] < s["center"]).all() and (s["center"] < s["upper"]).all()
        assert list(ssa_log.summary(ci_method=None)["loa_log_slope"].columns) == ["center"]

    def test_report_log_format(self):
        s = ssa_log.summary(ci_method="param")["loa_log_slope"]
        for loa_method in ["log", "auto"]:
            rpt = ssa_log.report(
                loa_method=loa_method, ci_method="param", decimals=2, sleep_stats=LOG_STATS
            )
            for stat in LOG_STATS:
                hw = s.loc[stat]
                expected = f"bias ± {hw['center']:.2f} × ref [{hw['lower']:.2f}, {hw['upper']:.2f}]"
                assert rpt.at[_label(rpt, stat), f"LoA [{PCT}% CI]"] == expected
        rpt = ssa_log.report(loa_method="log", ci_method=None, sleep_stats=LOG_STATS)
        assert rpt["LoA"].str.fullmatch(r"bias ± \d+\.\d+ × ref").all()

    def test_stats_with_zeros_are_not_log_transformed(self):
        zero_stats = [s for s in ssa_log.sleep_statistics if s not in LOG_STATS]
        assert zero_stats, "the fixture needs at least one stat with a zero value"
        # Regular LoA are used and reported for these stats, and loa_method="log" refuses them
        rpt = ssa_log.report(ci_method="param", sleep_stats=zero_stats)
        assert not rpt[f"LoA [{PCT}% CI]"].str.contains("×").any()
        with pytest.raises(ValueError, match="zero values"):
            ssa_log.report(loa_method="log", ci_method="param", sleep_stats=zero_stats[:1])

    def test_loa_method_override_with_log_transform(self):
        # loa_method="param"/"regr" force constant/regression LoA even when log_transform=True
        for loa_method in ["param", "regr"]:
            rpt = ssa_log.report(loa_method=loa_method, ci_method="param")
            assert not rpt[f"LoA [{PCT}% CI]"].str.contains("×").any()

    def test_loa_log_without_log_transform_raises(self):
        with pytest.raises(ValueError):
            ssa.report(loa_method="log")
        with pytest.raises(ValueError):
            ssa.plot_blandaltman(loa_method="log")

    def test_plot_log_lines(self):
        g = ssa_log.plot_blandaltman(loa_method="log", ci_method="param", sleep_stats=LOG_STATS)
        for stat, ax in zip(LOG_STATS, g.axes.flat, strict=True):
            assert len(ax.lines) == 4
            x = np.asarray(ax.lines[2].get_xdata())
            bias, upper, lower = (np.asarray(line.get_ydata()) for line in ax.lines[1:4])
            spread = _log_slope(ssa_log)[stat] * x
            np.testing.assert_allclose(upper - bias, spread)
            np.testing.assert_allclose(bias - lower, spread)
            assert len(ax.collections) == 3  # scatter + two CI bands
        g = ssa_log.plot_blandaltman(loa_method="log", ci_method=None, sleep_stats=LOG_STATS)
        for ax in g.axes.flat:
            assert len(ax.patches) == 0 and len(ax.collections) == 1

    def test_ci_auto(self):
        # ci_method="auto" picks "param" when normality holds, "boot" otherwise. Use ssa_log_large
        # (15 sessions, BCa) to avoid degenerate all-zero stats with N=5.
        stats = _log_slope(ssa_log_large).dropna().index.tolist()
        rpt = ssa_log_large.report(loa_method="log", ci_method="auto", sleep_stats=stats)
        assert rpt[f"LoA [{PCT}% CI]"].str.contains("×").all()
        g = ssa_log_large.plot_blandaltman(loa_method="log", ci_method="auto", sleep_stats=stats)
        assert len(g.axes.flat) == len(stats)

    def test_ci_boot(self):
        # Exercise the _generate_bootstrap_ci Euser path with method="basic" to avoid BCa
        # degenerate-data warnings/NaNs when a stat has identical values across all sessions.
        fresh = SleepStatsAgreement(
            _ref_stats_large,
            _obs_stats_large,
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
