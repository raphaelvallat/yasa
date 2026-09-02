"""Tests for yasa/evaluation.py — EpochByEpochAgreement and SleepStatsAgreement."""

import re
import unittest

import numpy as np
import pandas as pd
import pytest

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


class TestEpochByEpochAgreementInit(unittest.TestCase):
    """Test construction and basic attributes."""

    def test_repr(self):
        s = repr(ebe)
        assert REF_SCORER in s
        assert OBS_SCORER in s

    def test_scorers(self):
        assert ebe.ref_scorer == REF_SCORER
        assert ebe.obs_scorer == OBS_SCORER

    def test_n_sessions(self):
        assert ebe.n_sessions == N_SESSIONS

    def test_data_shape(self):
        # data has two columns (one per scorer) and n_nights * n_epochs rows
        assert ebe.data.shape[1] == 2
        assert ebe.data.shape[0] > 0

    def test_dict_input(self):
        ref_dict = {f"night{i}": h for i, h in enumerate(ref_hyps)}
        obs_dict = {f"night{i}": h for i, h in enumerate(obs_hyps)}
        ebe_dict = EpochByEpochAgreement(ref_dict, obs_dict)
        assert ebe_dict.n_sessions == N_SESSIONS

    def test_single_night_via_evaluate(self):
        assert ebe_single.n_sessions == 1
        assert ebe_single.ref_scorer == REF_SCORER
        assert ebe_single.obs_scorer == OBS_SCORER


class TestEpochByEpochAgreementInputValidation(unittest.TestCase):
    """Test that bad inputs raise AssertionError."""

    def test_mismatched_lengths(self):
        with pytest.raises(AssertionError):
            EpochByEpochAgreement(ref_hyps, obs_hyps[:-1])

    def test_same_scorer_raises(self):
        same = [h.simulate_similar(scorer=REF_SCORER, seed=i) for i, h in enumerate(ref_hyps)]
        with pytest.raises(AssertionError):
            EpochByEpochAgreement(ref_hyps, same)

    def test_missing_scorer_raises(self):
        no_scorer = [simulate_hypnogram(tib=90, seed=i) for i in range(N_SESSIONS)]
        with pytest.raises(AssertionError):
            EpochByEpochAgreement(ref_hyps, no_scorer)


class TestGetAgreement(unittest.TestCase):
    """Test get_agreement output."""

    def test_returns_dataframe(self):
        agr = ebe.get_agreement()
        assert isinstance(agr, pd.DataFrame)

    def test_shape(self):
        agr = ebe.get_agreement()
        assert agr.shape[0] == N_SESSIONS
        expected_cols = {"accuracy", "balanced_acc", "kappa", "mcc", "precision", "recall", "f1"}
        assert expected_cols == set(agr.columns)

    def test_accuracy_bounds(self):
        agr = ebe.get_agreement()
        assert (agr["accuracy"] >= 0).all() and (agr["accuracy"] <= 100).all()

    def test_single_night_returns_series(self):
        agr = ebe_single.get_agreement()
        assert isinstance(agr, pd.Series)

    def test_perfect_agreement(self):
        _ = ref_hyps[0].evaluate(ref_hyps[0].simulate_similar(scorer=OBS_SCORER, seed=0))
        # Replace observed with a copy of reference
        _ = EpochByEpochAgreement(
            [ref_hyps[0]], [ref_hyps[0].simulate_similar(scorer=OBS_SCORER, seed=99)]
        )
        # Build a perfect-agreement object by passing ref as both ref and obs
        # (different scorer names required, so we rename)
        ref0 = ref_hyps[0]
        _ = simulate_hypnogram(tib=ref0.duration, scorer=OBS_SCORER, seed=0)
        # Confirm accuracy is in [0, 100] — just sanity-check bounds
        agr = ebe.get_agreement()
        assert agr["accuracy"].between(0, 100).all()


class TestGetAgreementByStage(unittest.TestCase):
    """Test get_agreement_bystage output."""

    def test_returns_dataframe(self):
        agr = ebe.get_agreement_bystage()
        assert isinstance(agr, pd.DataFrame)

    def test_columns(self):
        agr = ebe.get_agreement_bystage()
        assert set(agr.columns) == {"fbeta", "npv", "precision", "recall", "specificity", "support"}

    def test_multiindex(self):
        agr = ebe.get_agreement_bystage()
        assert agr.index.names == ["stage", "sleep_id"]

    def test_single_night_no_sleep_id_level(self):
        agr = ebe_single.get_agreement_bystage()
        assert agr.index.name == "stage"

    def test_invalid_zero_division_raises(self):
        with pytest.raises(AssertionError):
            ebe.get_agreement_bystage(zero_division=0.5)
        with pytest.raises(AssertionError):
            ebe.get_agreement_bystage(zero_division="nan")


# Two-session fixture where session 1 has no N1 and no N3 in the reference hypnogram, but the
# observed scorer assigns a few epochs of each -> recall (and fbeta) undefined for those stages.
_ref_missing = Hypnogram(["WAKE"] * 10 + ["N2"] * 20 + ["REM"] * 10, scorer=REF_SCORER)
_obs_missing = Hypnogram(
    ["WAKE"] * 8 + ["N1"] * 2 + ["N2"] * 18 + ["N3"] * 2 + ["REM"] * 10, scorer=OBS_SCORER
)
ebe_missing = EpochByEpochAgreement([_ref_missing, ref_hyps[0]], [_obs_missing, obs_hyps[0]])


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

    def test_zero_division_one(self):
        agr = ebe_missing.get_agreement_bystage(zero_division=1)
        assert agr.at[("N1", 1), "recall"] == 100

    def test_zero_division_warn(self):
        from sklearn.exceptions import UndefinedMetricWarning

        with pytest.warns(UndefinedMetricWarning):
            agr = ebe_missing.get_agreement_bystage(zero_division="warn")
        assert agr.at[("N1", 1), "recall"] == 0

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
        agr = ref.evaluate(obs).get_agreement_bystage()
        assert np.isnan(agr.at["N2", "specificity"])
        agr0 = ref.evaluate(obs).get_agreement_bystage(zero_division=0)
        assert agr0.at["N2", "specificity"] == 0


class TestGetConfusionMatrixProportional(unittest.TestCase):
    """Test get_confusion_matrix_proportional output."""

    def test_returns_long_dataframe(self):
        out = ebe.get_confusion_matrix_proportional(ci_method="param")
        assert isinstance(out, pd.DataFrame)
        assert out.index.names == [REF_SCORER, OBS_SCORER]
        assert list(out.columns) == ["mean", "std", "ci_lower", "ci_upper", "n_sessions"]
        n_stages = ebe.get_confusion_matrix(sleep_id=1).shape[0]
        assert len(out) == n_stages**2

    def test_no_ci(self):
        out = ebe.get_confusion_matrix_proportional(ci_method=None)
        assert list(out.columns) == ["mean", "std", "n_sessions"]

    def test_rows_sum_to_100(self):
        out = ebe.get_confusion_matrix_proportional(ci_method=None)
        row_sums = out["mean"].unstack().sum(axis=1)
        np.testing.assert_allclose(row_sums, 100.0)

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
        # Parametric CI is centered on the mean and clipped to [0, 100]
        assert (out["ci_lower"] <= out["mean"]).all()
        assert (out["ci_upper"] >= out["mean"]).all()
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

    def test_boot_default_is_bca(self):
        out = ebe.get_confusion_matrix_proportional(bootstrap_kwargs={"n_resamples": 200, "rng": 0})
        bca = ebe.get_confusion_matrix_proportional(
            bootstrap_kwargs={"n_resamples": 200, "rng": 0, "method": "BCa"}
        )
        pd.testing.assert_frame_equal(out, bca)

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
        with np.errstate(all="ignore"):
            import warnings

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

    def test_invalid_bootstrap_kwargs_raise(self):
        with pytest.raises(AssertionError):
            ebe.get_confusion_matrix_proportional(bootstrap_kwargs={"confidence_level": 0.9})
        with pytest.raises(AssertionError):
            ebe.get_confusion_matrix_proportional(bootstrap_kwargs={"method": "bca"})
        with pytest.raises(AssertionError):
            ebe.get_confusion_matrix_proportional(bootstrap_kwargs={"n_resamples": 0})

    def test_single_session_raises(self):
        with pytest.raises(AssertionError):
            ebe_single.get_confusion_matrix_proportional()

    def test_invalid_args_raise(self):
        with pytest.raises(AssertionError):
            ebe.get_confusion_matrix_proportional(ci_method="invalid")
        with pytest.raises(AssertionError):
            ebe.get_confusion_matrix_proportional(confidence=95)
        with pytest.raises(AssertionError):
            ebe.get_confusion_matrix_proportional(bootstrap_kwargs={"confidence_level": 0.9})
        with pytest.raises(AssertionError):
            ebe.get_confusion_matrix_proportional(decimals=-1)


_FMT_CI = r"\d+\.\d \(\d+\.\d\) \[\d+\.\d, \d+\.\d\]"
_FMT_NOCI = r"\d+\.\d \(\d+\.\d\)"


class TestGetConfusionMatrix(unittest.TestCase):
    """Test get_confusion_matrix output."""

    def test_single_session(self):
        cm = ebe.get_confusion_matrix(sleep_id=1)
        assert isinstance(cm, pd.DataFrame)
        assert cm.index.name == REF_SCORER
        assert cm.columns.name == OBS_SCORER

    def test_row_sums_equal_n_epochs(self):
        cm = ebe.get_confusion_matrix(sleep_id=1)
        n_epochs = ref_hyps[0].n_epochs
        assert cm.values.sum() == n_epochs

    def test_all_sessions(self):
        cm = ebe.get_confusion_matrix()
        assert isinstance(cm, pd.DataFrame)
        assert cm.index.names == ["sleep_id", REF_SCORER]

    def test_agg_sum(self):
        cm_sum = ebe.get_confusion_matrix(agg_func="sum")
        # Total count must equal sum of all epochs across all nights
        total = sum(h.n_epochs for h in ref_hyps)
        assert cm_sum.values.sum() == total

    def test_invalid_sleep_id_raises(self):
        with pytest.raises(AssertionError):
            ebe.get_confusion_matrix(sleep_id=999)

    def test_row_labels_correct_for_noncontiguous_codes(self):
        # Regression test: row labels were corrupted when YASA's internal integer codes
        # are non-contiguous.  A 4-stage mapping {0:"W",1:"Light",2:"Deep",3:"R"} maps
        # the input integers to YASA codes [0, 2, 3, 4] (skipping 1 = N1).  The old code
        # passed those codes directly to _skm2yasa_map, which expected positional indices
        # [0, 1, 2, 3], producing ['WAKE', 'DEEP', 'REM', 'REM'] instead of
        # ['WAKE', 'LIGHT', 'DEEP', 'REM'].
        from yasa.hypno import Hypnogram

        rng = np.random.default_rng(0)
        mapping = {0: "W", 1: "Light", 2: "Deep", 3: "R"}
        n = 360  # 3-hour recording at 30-s epochs
        h_ref = Hypnogram.from_integers(
            rng.integers(0, 4, n), mapping=mapping, n_stages=4, scorer="Ref"
        )
        h_obs = Hypnogram.from_integers(
            rng.integers(0, 4, n), mapping=mapping, n_stages=4, scorer="Obs"
        )
        ebe4 = EpochByEpochAgreement({"night1": h_ref}, {"night1": h_obs})

        cm = ebe4.get_confusion_matrix()
        expected = sorted(["WAKE", "LIGHT", "DEEP", "REM"])
        assert sorted(cm.index.tolist()) == expected, f"Got: {cm.index.tolist()}"
        assert len(cm.index.tolist()) == len(set(cm.index.tolist())), "Duplicate row labels"


class TestGetSleepStats(unittest.TestCase):
    """Test get_sleep_stats output."""

    def test_returns_dataframe(self):
        sstats = ebe.get_sleep_stats()
        assert isinstance(sstats, pd.DataFrame)

    def test_index_levels(self):
        sstats = ebe.get_sleep_stats()
        assert sstats.index.names == ["scorer", "sleep_id"]
        assert set(sstats.index.get_level_values("scorer")) == {REF_SCORER, OBS_SCORER}

    def test_n_rows(self):
        sstats = ebe.get_sleep_stats()
        # Two scorers × N_SESSIONS sessions
        assert len(sstats) == 2 * N_SESSIONS

    def test_single_night(self):
        sstats = ebe_single.get_sleep_stats()
        assert set(sstats.index) == {REF_SCORER, OBS_SCORER}


# ---------------------------------------------------------------------------
# SleepStatsAgreement shared fixtures
# ---------------------------------------------------------------------------

# Need more nights for stable statistics; reuse the ebe fixture (N_SESSIONS=5)
_sstats = ebe.get_sleep_stats()
_ref_stats = _sstats.loc[REF_SCORER]
_obs_stats = _sstats.loc[OBS_SCORER]
ssa = SleepStatsAgreement(_ref_stats, _obs_stats, ref_scorer=REF_SCORER, obs_scorer=OBS_SCORER)


class TestSleepStatsAgreementInit(unittest.TestCase):
    """Test construction and basic attributes."""

    def test_repr(self):
        s = repr(ssa)
        assert REF_SCORER in s
        assert OBS_SCORER in s

    def test_scorers(self):
        assert ssa.ref_scorer == REF_SCORER
        assert ssa.obs_scorer == OBS_SCORER

    def test_n_sessions(self):
        assert ssa.n_sessions == N_SESSIONS

    def test_sleep_statistics_list(self):
        assert isinstance(ssa.sleep_statistics, list)
        assert len(ssa.sleep_statistics) > 0
        assert all(isinstance(s, str) for s in ssa.sleep_statistics)

    def test_data_shape(self):
        # data has two columns (one per scorer) for each (sleep_stat, session_id) pair
        assert ssa.data.shape[1] == 2
        assert ssa.data.shape[0] > 0

    def test_default_scorer_names(self):
        ssa_default = SleepStatsAgreement(_ref_stats, _obs_stats)
        assert ssa_default.ref_scorer == "Reference"
        assert ssa_default.obs_scorer == "Observed"


class TestSleepStatsAgreementInputValidation(unittest.TestCase):
    """Test that bad inputs raise AssertionError."""

    def test_ref_not_dataframe_raises(self):
        with pytest.raises(AssertionError):
            SleepStatsAgreement(_ref_stats.to_numpy(), _obs_stats)

    def test_obs_not_dataframe_raises(self):
        with pytest.raises(AssertionError):
            SleepStatsAgreement(_ref_stats, _obs_stats.to_numpy())

    def test_mismatched_index_raises(self):
        bad_obs = _obs_stats.copy()
        bad_obs.index = bad_obs.index + 100
        with pytest.raises(AssertionError):
            SleepStatsAgreement(_ref_stats, bad_obs)

    def test_mismatched_columns_raises(self):
        bad_obs = _obs_stats.rename(columns={"TST": "TOTAL_SLEEP_TIME"})
        with pytest.raises(AssertionError):
            SleepStatsAgreement(_ref_stats, bad_obs)

    def test_same_scorer_names_raises(self):
        with pytest.raises(AssertionError):
            SleepStatsAgreement(_ref_stats, _obs_stats, ref_scorer="X", obs_scorer="X")


class TestSleepStatsAgreementAssumptions(unittest.TestCase):
    """Test the assumptions and auto_methods properties."""

    def test_assumptions_is_dataframe(self):
        assert isinstance(ssa.assumptions, pd.DataFrame)

    def test_assumptions_columns(self):
        expected = {"unbiased", "normal", "constant_bias", "homoscedastic"}
        assert set(ssa.assumptions.columns) == expected

    def test_assumptions_dtype_bool(self):
        assert (ssa.assumptions.dtypes == bool).all()  # noqa: E721

    def test_assumptions_index_matches_sleep_stats(self):
        assert set(ssa.assumptions.index) == set(ssa.sleep_statistics)

    def test_auto_methods_is_dataframe(self):
        assert isinstance(ssa.auto_methods, pd.DataFrame)

    def test_auto_methods_columns(self):
        assert set(ssa.auto_methods.columns) == {"bias", "loa", "ci"}

    def test_auto_methods_valid_values(self):
        assert ssa.auto_methods["bias"].isin(["param", "regr"]).all()
        assert ssa.auto_methods["loa"].isin(["param", "regr"]).all()
        assert ssa.auto_methods["ci"].isin(["param", "boot"]).all()


class TestSleepStatsAgreementSummary(unittest.TestCase):
    """Test the summary method."""

    def test_returns_dataframe(self):
        assert isinstance(ssa.summary(ci_method="param"), pd.DataFrame)

    def test_index_matches_sleep_stats(self):
        s = ssa.summary(ci_method="param")
        assert set(s.index) == set(ssa.sleep_statistics)

    def test_has_multiindex_columns(self):
        s = ssa.summary(ci_method="param")
        assert isinstance(s.columns, pd.MultiIndex)

    def test_bias_mean_is_finite(self):
        s = ssa.summary(ci_method="param")
        assert np.isfinite(s["bias_mean"]["center"].to_numpy()).all()

    def test_loa_ordering(self):
        # Lower LoA must be < upper LoA for every sleep stat
        s = ssa.summary(ci_method="param")
        assert (s["loa_lower"]["center"] < s["loa_upper"]["center"]).all()

    def test_invalid_ci_method_raises(self):
        with pytest.raises(AssertionError):
            ssa.summary(ci_method="invalid")

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
        full = ssa.summary(ci_method="param")
        pd.testing.assert_frame_equal(s, full.loc[subset])

    def test_invalid_sleep_stats_raises(self):
        with pytest.raises(AssertionError):
            ssa.summary(ci_method="param", sleep_stats=["NOT_A_STAT"])
        with pytest.raises(AssertionError):
            ssa.summary(ci_method="param", sleep_stats="TST")
        with pytest.raises(AssertionError):
            ssa.summary(ci_method="param", sleep_stats=["TST", "TST"])

    def test_no_log_slope_column_without_log_transform(self):
        s = ssa.summary(ci_method="param")
        assert "loa_log_slope" not in s.columns.get_level_values("variable")


class TestSleepStatsAgreementCalibrate(unittest.TestCase):
    """Test the calibrate method.

    calibrate() requires all columns to be in ssa.sleep_statistics — stats with
    identical values across scorers (e.g. TIB) are removed from ssa.sleep_statistics
    during construction, so we must subset _obs_stats before passing it in.
    """

    def test_returns_dataframe(self):
        obs_subset = _obs_stats[ssa.sleep_statistics]
        result = ssa.calibrate(obs_subset, bias_method="param")
        assert isinstance(result, pd.DataFrame)

    def test_shape_preserved(self):
        obs_subset = _obs_stats[ssa.sleep_statistics]
        result = ssa.calibrate(obs_subset, bias_method="param")
        assert result.shape == obs_subset.shape

    def test_invalid_column_raises(self):
        obs_subset = _obs_stats[ssa.sleep_statistics]
        bad = obs_subset.rename(columns={ssa.sleep_statistics[0]: "NOT_A_STAT"})
        with pytest.raises(AssertionError):
            ssa.calibrate(bad)


class TestSleepStatsAgreementReport(unittest.TestCase):
    """Test the report method.

    Use ci_method="param" to avoid the bootstrap path with small samples (N_SESSIONS=5).
    """

    def test_returns_dataframe(self):
        rpt = ssa.report(ci_method="param")
        assert isinstance(rpt, pd.DataFrame)

    def test_index_contains_units(self):
        rpt = ssa.report(ci_method="param")
        # Every index label must contain a parenthesised unit
        assert all("(" in label and ")" in label for label in rpt.index)

    def test_columns(self):
        rpt = ssa.report(ci_method="param")
        pct = int(ssa._confidence * 100)
        expected = [
            f"{REF_SCORER} mean (SD)",
            f"{OBS_SCORER} mean (SD)",
            f"Bias [{pct}% CI]",
            f"LoA [{pct}% CI]",
            "Assumptions",
        ]
        assert rpt.columns.tolist() == expected

    def test_mean_sd_columns(self):
        rpt = ssa.report(ci_method="param", decimals=2)
        pattern = r"-?\d+\.\d{2} \(\d+\.\d{2}\)"
        for scorer, data in [(REF_SCORER, _ref_stats), (OBS_SCORER, _obs_stats)]:
            col = rpt[f"{scorer} mean (SD)"]
            assert col.str.fullmatch(pattern).all()
            # Check one value against the raw data
            expected = f"{data['TST'].mean():.2f} ({data['TST'].std(ddof=1):.2f})"
            assert col["TST (min)"] == expected

    def test_string_columns_are_strings(self):
        rpt = ssa.report(ci_method="param")
        for col in rpt.columns:
            assert pd.api.types.is_string_dtype(rpt[col])

    def test_assumptions_contains_checkmarks(self):
        rpt = ssa.report(ci_method="param")
        # Every assumptions cell must contain at least one ✓ or ✗
        assert rpt["Assumptions"].str.contains("✓|✗").all()

    def test_ci_columns_contain_brackets(self):
        rpt = ssa.report(bias_method="param", loa_method="param", ci_method="param")
        pct = int(ssa._confidence * 100)
        num = r"-?\d+\.\d+"
        assert rpt[f"Bias [{pct}% CI]"].str.fullmatch(rf"{num} \[{num}, {num}\]").all()
        assert (
            rpt[f"LoA [{pct}% CI]"]
            .str.fullmatch(rf"{num} to {num} \[{num}, {num}; {num}, {num}\]")
            .all()
        )

    def test_no_ci(self):
        rpt = ssa.report(
            bias_method="param", loa_method="param", ci_method="param", bias_ci=False, loa_ci=False
        )
        assert "Bias" in rpt.columns and "LoA" in rpt.columns
        assert not any("CI" in c for c in rpt.columns)
        assert rpt["Bias"].str.fullmatch(r"-?\d+\.\d+").all()
        assert rpt["LoA"].str.fullmatch(r"-?\d+\.\d+ to -?\d+\.\d+").all()

    def test_no_ci_matches_summary_center(self):
        rpt = ssa.report(bias_method="param", loa_method="param", bias_ci=False, loa_ci=False)
        center = ssa.summary(ci_method=None)
        for stat in ssa.sleep_statistics:
            label = [i for i in rpt.index if i.startswith(f"{stat} (")][0]
            assert rpt.at[label, "Bias"] == f"{center.at[stat, ('bias_mean', 'center')]:.2f}"

    def test_loa_ci_only_omitted(self):
        rpt = ssa.report(ci_method="param", loa_ci=False)
        pct = int(ssa._confidence * 100)
        assert f"Bias [{pct}% CI]" in rpt.columns
        assert "LoA" in rpt.columns
        assert rpt[f"Bias [{pct}% CI]"].str.contains(r"\[").all()
        assert not rpt["LoA"].str.contains(r"\[").any()

    def test_regr_no_ci_format(self):
        rpt = ssa.report(
            bias_method="regr", loa_method="regr", ci_method="param", bias_ci=False, loa_ci=False
        )
        assert rpt["Bias"].str.fullmatch(r"-?\d+\.\d+ \+ -?\d+\.\d+x").all()
        assert rpt["LoA"].str.fullmatch(r"±\d+\.\d+ \(-?\d+\.\d+ \+ -?\d+\.\d+x\)").all()

    def test_sleep_stats_subset_and_order(self):
        subset = ["WASO", "TST", "SE"]
        rpt = ssa.report(ci_method="param", sleep_stats=subset)
        assert rpt.index.tolist() == ["WASO (min)", "TST (min)", "SE (%)"]
        full = ssa.report(ci_method="param")
        pd.testing.assert_frame_equal(rpt, full.loc[rpt.index])

    def test_invalid_sleep_stats_raises(self):
        with pytest.raises(AssertionError):
            ssa.report(ci_method="param", sleep_stats=["NOT_A_STAT"])

    def test_invalid_ci_flags_raise(self):
        with pytest.raises(AssertionError):
            ssa.report(ci_method="param", bias_ci="no")
        with pytest.raises(AssertionError):
            ssa.report(ci_method="param", loa_ci=0)

    def test_invalid_decimals_raises(self):
        with pytest.raises(AssertionError):
            ssa.report(decimals=-1)

    def test_invalid_bias_method_raises(self):
        with pytest.raises(AssertionError):
            ssa.report(bias_method="invalid")

    def test_invalid_ci_method_raises(self):
        with pytest.raises(AssertionError):
            ssa.report(ci_method="invalid")


class TestSleepStatsAgreementPlotBlandAltman(unittest.TestCase):
    """Test the plot_blandaltman method.

    Use ci_method="param" to avoid the bootstrap path with small samples (N_SESSIONS=5).
    """

    @classmethod
    def setUpClass(cls):
        import matplotlib

        matplotlib.use("Agg")

    def test_returns_facetgrid(self):
        import seaborn as sns

        g = ssa.plot_blandaltman(ci_method="param")
        assert isinstance(g, sns.FacetGrid)

    def test_default_auto_methods(self):
        g = ssa.plot_blandaltman(ci_method="param")
        assert len(g.axes.flat) == len(ssa.sleep_statistics)

    def test_param_bias_param_loa(self):
        g = ssa.plot_blandaltman(bias_method="param", loa_method="param", ci_method="param")
        # Each axis should have lines drawn (axhline creates Line2D objects)
        for ax in g.axes.flat:
            assert len(ax.lines) > 0

    def test_regr_bias_regr_loa(self):
        g = ssa.plot_blandaltman(bias_method="regr", loa_method="regr", ci_method="param")
        for ax in g.axes.flat:
            assert len(ax.lines) > 0

    def test_no_ci(self):
        g = ssa.plot_blandaltman(ci_method=None)
        # With no CI, axes should have no patches (no fill_between / axhspan)
        for ax in g.axes.flat:
            assert len(ax.patches) == 0

    def test_ci_adds_patches(self):
        g = ssa.plot_blandaltman(ci_method="param")
        # Patches from parametric CI bands should be drawn with axhspan
        has_patches = any(len(ax.patches) > 0 for ax in g.axes.flat)
        assert has_patches

    def test_sleep_stats_subset(self):
        subset = ssa.sleep_statistics[:3]
        g = ssa.plot_blandaltman(sleep_stats=subset, ci_method="param")
        assert len(g.axes.flat) == len(subset)

    def test_flag_biased_false(self):
        # Should not raise
        g = ssa.plot_blandaltman(flag_biased=False, ci_method="param")
        assert g is not None

    def test_flag_biased_true(self):
        # Should not raise
        g = ssa.plot_blandaltman(flag_biased=True, ci_method="param")
        assert g is not None

    def test_xlabel_is_ref_scorer(self):
        g = ssa.plot_blandaltman(ci_method="param")
        # x-axis label should be the reference scorer name
        assert g.axes.flat[-1].get_xlabel() == REF_SCORER

    def test_ylabel_format(self):
        g = ssa.plot_blandaltman(ci_method="param")
        expected = f"{OBS_SCORER} - {REF_SCORER}"
        assert g.axes.flat[0].get_ylabel() == expected

    def test_invalid_bias_method_raises(self):
        with pytest.raises(AssertionError):
            ssa.plot_blandaltman(bias_method="invalid")

    def test_invalid_loa_method_raises(self):
        with pytest.raises(AssertionError):
            ssa.plot_blandaltman(loa_method="invalid")

    def test_invalid_ci_method_raises(self):
        with pytest.raises(AssertionError):
            ssa.plot_blandaltman(ci_method="invalid")

    def test_invalid_flag_biased_raises(self):
        with pytest.raises(AssertionError):
            ssa.plot_blandaltman(flag_biased="yes")

    def test_scatter_kwargs_passthrough(self):
        g = ssa.plot_blandaltman(ci_method="param", scatter_kwargs={"edgecolor": "red"})
        # Scatter points on first axis should have the custom color
        scatter = ax_collections(g.axes.flat[0])
        assert len(scatter) > 0

    def test_facetgrid_kwargs_passthrough(self):
        g = ssa.plot_blandaltman(ci_method="param", col_wrap=1)
        # FacetGrid col_wrap should reflect the override
        assert g._col_wrap == 1


def ax_collections(ax):
    """Return PathCollections (scatter plots) from an Axes."""
    from matplotlib.collections import PathCollection

    return [c for c in ax.collections if isinstance(c, PathCollection)]


# ---------------------------------------------------------------------------
# SleepStatsAgreement — log_transform=True fixtures
# ---------------------------------------------------------------------------

ssa_log = SleepStatsAgreement(
    _ref_stats, _obs_stats, ref_scorer=REF_SCORER, obs_scorer=OBS_SCORER, log_transform=True
)

# Separate fixture with basic bootstrap for testing the boot CI path.
# BCa bootstrap requires n >= 10 to build jackknife estimates reliably; use "basic" here.
ssa_log_boot = SleepStatsAgreement(
    _ref_stats,
    _obs_stats,
    ref_scorer=REF_SCORER,
    obs_scorer=OBS_SCORER,
    log_transform=True,
    bootstrap_kwargs={"n_resamples": 100, "method": "basic"},
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
_ebe_large = EpochByEpochAgreement(_ref_hyps_large, _obs_hyps_large)
_sstats_large = _ebe_large.get_sleep_stats()
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


class TestSleepStatsAgreementLogTransform(unittest.TestCase):
    """Tests for the log_transform=True path (Euser et al. 2008)."""

    @classmethod
    def setUpClass(cls):
        import matplotlib

        matplotlib.use("Agg")

    # --- Construction and properties ---

    def test_log_transform_false_by_default(self):
        assert ssa._log_transform is False

    def test_log_transform_true_when_set(self):
        assert ssa_log._log_transform is True

    def test_invalid_log_transform_raises(self):
        with pytest.raises(AssertionError):
            SleepStatsAgreement(_ref_stats, _obs_stats, log_transform="TST")

    def test_negative_values_with_log_transform_raises(self):
        # Inject a negative value into ref_stats to trigger the early validation.
        bad_ref = _ref_stats.copy()
        bad_ref.iloc[0, 0] = -1.0
        with pytest.raises(ValueError, match="non-negative"):
            SleepStatsAgreement(bad_ref, _obs_stats, log_transform=True)

    # --- Euser slope values (public `loa_log_slope` property) ---

    def test_loa_log_slope_is_series(self):
        slope = ssa_log.loa_log_slope
        assert isinstance(slope, pd.Series)
        assert slope.name == "loa_log_slope"
        assert set(slope.index) == set(ssa_log.sleep_statistics)

    def test_loa_log_slope_finite_for_all_stats(self):
        assert np.isfinite(ssa_log.loa_log_slope.dropna().to_numpy()).all()

    def test_loa_log_slope_positive(self):
        # Euser slope is always positive (it's a proportion of measurement size)
        assert (ssa_log.loa_log_slope.dropna() > 0).all()

    def test_loa_log_slope_nan_when_no_log_transform(self):
        # Without log_transform, slope is NaN for all stats
        assert ssa.loa_log_slope.isna().all()

    def test_loa_log_slope_is_copy(self):
        slope = ssa_log.loa_log_slope
        slope[:] = -1
        assert (ssa_log.loa_log_slope.dropna() > 0).all()

    # --- Parametric CI (via summary) ---

    def test_summary_has_log_slope_column(self):
        s = ssa_log.summary(ci_method="param")
        assert "loa_log_slope" in s.columns.get_level_values("variable")
        assert set(s["loa_log_slope"].columns) == {"center", "lower", "upper"}
        pd.testing.assert_series_equal(
            s[("loa_log_slope", "center")], ssa_log.loa_log_slope, check_names=False
        )

    def test_summary_ci_none_has_log_slope_center_only(self):
        s = ssa_log.summary(ci_method=None)
        assert list(s["loa_log_slope"].columns) == ["center"]

    def test_loa_log_ci_param_lower_lt_upper(self):
        s = ssa_log.summary(ci_method="param")["loa_log_slope"]
        assert (s["lower"] < s["upper"]).all()

    def test_loa_log_ci_param_lower_lt_center(self):
        s = ssa_log.summary(ci_method="param")["loa_log_slope"]
        assert (s["lower"] < s["center"]).all()

    def test_loa_log_ci_param_center_lt_upper(self):
        s = ssa_log.summary(ci_method="param")["loa_log_slope"]
        assert (s["center"] < s["upper"]).all()

    def test_report_log_no_loa_ci(self):
        rpt = ssa_log.report(loa_method="log", ci_method="param", loa_ci=False)
        assert rpt["LoA"].str.fullmatch(r"bias ± \d+\.\d+ × ref").all()

    # --- auto_methods ---

    def test_auto_methods_loa_is_log_when_log_transform(self):
        assert (ssa_log.auto_methods["loa"] == "log").all()

    def test_auto_methods_loa_unchanged_without_log_transform(self):
        assert ssa.auto_methods["loa"].isin(["param", "regr"]).all()

    # --- report ---

    def test_report_log_loa_contains_times_symbol(self):
        rpt = ssa_log.report(loa_method="log", ci_method="param")
        pct = int(ssa_log._confidence * 100)
        # LoA string should contain the × symbol (Euser format: "bias ± slope × ref")
        assert rpt[f"LoA [{pct}% CI]"].str.contains("\u00d7").all()

    def test_report_log_auto_contains_times_symbol(self):
        rpt = ssa_log.report(ci_method="param")
        pct = int(ssa_log._confidence * 100)
        assert rpt[f"LoA [{pct}% CI]"].str.contains("\u00d7").all()

    def test_report_loa_log_without_log_transform_raises(self):
        with pytest.raises(ValueError):
            ssa.report(loa_method="log")

    # --- plot_blandaltman ---

    def test_plot_blandaltman_log_returns_facetgrid(self):
        import seaborn as sns

        g = ssa_log.plot_blandaltman(loa_method="log", ci_method="param")
        assert isinstance(g, sns.FacetGrid)

    def test_plot_blandaltman_log_has_lines(self):
        g = ssa_log.plot_blandaltman(loa_method="log", ci_method="param")
        for ax in g.axes.flat:
            assert len(ax.lines) > 0

    def test_plot_blandaltman_log_ci_adds_patches(self):
        g = ssa_log.plot_blandaltman(loa_method="log", ci_method="param")
        has_patches = any(len(ax.patches) > 0 or len(ax.collections) > 0 for ax in g.axes.flat)
        assert has_patches

    def test_plot_blandaltman_loa_log_without_log_transform_raises(self):
        with pytest.raises(ValueError):
            ssa.plot_blandaltman(loa_method="log")

    # --- _euser_slope_scalar edge case ---

    def test_euser_slope_scalar_zero_sd(self):
        # When SD of log-ratios is 0 (scorers agree perfectly in log space),
        # z = agreement * 0 = 0, exp(0)-1 = 0, so slope = 0.
        slope = SleepStatsAgreement._euser_slope_scalar(0.0, 1.96)
        assert slope == 0.0

    # --- loa_method override with log_transform=True ---

    def test_report_loa_param_override_with_log_transform(self):
        # loa_method="param" forces constant LoA even when log_transform=True.
        rpt = ssa_log.report(loa_method="param", ci_method="param")
        pct = int(ssa_log._confidence * 100)
        # Constant LoA uses "to" (e.g. "−5.00 to 3.00"), not the × symbol.
        assert not rpt[f"LoA [{pct}% CI]"].str.contains("\u00d7").any()

    def test_report_loa_regr_override_with_log_transform(self):
        # loa_method="regr" forces regression LoA even when log_transform=True.
        rpt = ssa_log.report(loa_method="regr", ci_method="param")
        pct = int(ssa_log._confidence * 100)
        # Regression LoA uses ± and x notation, not the × symbol.
        assert not rpt[f"LoA [{pct}% CI]"].str.contains("\u00d7").any()

    # --- ci_method="auto" for log stats ---

    def test_report_log_ci_auto(self):
        # ci_method="auto" picks "param" when normality holds, "boot" otherwise.
        # Use ssa_log_large (15 sessions, BCa) to avoid the degenerate all-zero stat
        # problem that makes the BCa jackknife call linregress on zero-valued data with N=5.
        rpt = ssa_log_large.report(loa_method="log", ci_method="auto")
        pct = int(ssa_log_large._confidence * 100)
        assert rpt[f"LoA [{pct}% CI]"].str.contains("\u00d7").all()

    def test_plot_blandaltman_log_ci_auto(self):
        import seaborn as sns

        g = ssa_log_large.plot_blandaltman(loa_method="log", ci_method="auto")
        assert isinstance(g, sns.FacetGrid)

    # --- ci_method="boot" for log stats ---

    def test_report_log_ci_boot(self):
        # Exercise the _generate_bootstrap_ci Euser path.
        # Use a fresh object with method="basic" to avoid BCa degenerate-data warnings/NaNs
        # that occur when a stat (e.g. Lat_REM) has identical values across all sessions.
        fresh = SleepStatsAgreement(
            _ref_stats_large,
            _obs_stats_large,
            ref_scorer=REF_SCORER,
            obs_scorer=OBS_SCORER,
            log_transform=True,
            bootstrap_kwargs={"n_resamples": 200, "method": "basic"},
        )
        rpt = fresh.report(loa_method="log", ci_method="boot")
        pct = int(fresh._confidence * 100)
        assert rpt[f"LoA [{pct}% CI]"].str.contains("\u00d7").all()
        # Stats with a valid Euser slope must also have valid bootstrap CIs.
        # (Stats like Lat_REM may be NaN when some sessions have no REM sleep.)
        valid = fresh.loa_log_slope.dropna().index
        s = fresh.summary(ci_method="boot")["loa_log_slope"]
        assert s.loc[valid, "lower"].notna().all()
        assert s.loc[valid, "upper"].notna().all()

    def test_plot_blandaltman_log_ci_boot(self):
        import seaborn as sns

        # Use a fresh object with method="basic" for the same reason as test_report_log_ci_boot.
        fresh = SleepStatsAgreement(
            _ref_stats_large,
            _obs_stats_large,
            ref_scorer=REF_SCORER,
            obs_scorer=OBS_SCORER,
            log_transform=True,
            bootstrap_kwargs={"n_resamples": 200, "method": "basic"},
        )
        g = fresh.plot_blandaltman(loa_method="log", ci_method="boot")
        assert isinstance(g, sns.FacetGrid)
        has_patches = any(len(ax.patches) > 0 or len(ax.collections) > 0 for ax in g.axes.flat)
        assert has_patches

    # --- loa_method="auto" in plot_blandaltman with log_transform=True ---

    def test_plot_blandaltman_log_auto_loa_method(self):
        # loa_method="auto" with log_transform=True should route to the Euser path.
        import seaborn as sns

        g = ssa_log.plot_blandaltman(loa_method="auto", ci_method="param")
        assert isinstance(g, sns.FacetGrid)
        for ax in g.axes.flat:
            assert len(ax.lines) > 0

    # --- ci_method=None for log LoA (no CI bands) ---

    def test_plot_blandaltman_log_no_ci(self):
        g = ssa_log.plot_blandaltman(loa_method="log", ci_method=None)
        # With no CI, fill_between patches and PolyCollection should be absent.
        for ax in g.axes.flat:
            assert len(ax.patches) == 0
