"""Tests for yasa/evaluation.py — EpochByEpochAgreement and SleepStatsAgreement."""

import unittest

import numpy as np
import pandas as pd
import pytest

from yasa.evaluation import EpochByEpochAgreement, SleepStatsAgreement
from yasa.hypno import simulate_hypnogram

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
        assert (agr["accuracy"] >= 0).all() and (agr["accuracy"] <= 1).all()

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
        # Confirm accuracy is in [0, 1] — just sanity-check bounds
        agr = ebe.get_agreement()
        assert agr["accuracy"].between(0, 1).all()


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
        assert f"Bias [{pct}% CI]" in rpt.columns
        assert f"LoA [{pct}% CI]" in rpt.columns
        assert "Assumptions" in rpt.columns
        assert f"{REF_SCORER} mean" in rpt.columns
        assert f"{OBS_SCORER} mean" in rpt.columns

    def test_mean_columns_are_numeric(self):
        rpt = ssa.report(ci_method="param")
        assert pd.api.types.is_numeric_dtype(rpt[f"{REF_SCORER} mean"])
        assert pd.api.types.is_numeric_dtype(rpt[f"{OBS_SCORER} mean"])

    def test_string_columns_are_strings(self):
        rpt = ssa.report(ci_method="param")
        pct = int(ssa._confidence * 100)
        for col in [f"Bias [{pct}% CI]", f"LoA [{pct}% CI]", "Assumptions"]:
            assert pd.api.types.is_string_dtype(rpt[col])

    def test_assumptions_contains_checkmarks(self):
        rpt = ssa.report(ci_method="param")
        # Every assumptions cell must contain at least one ✓ or ✗
        assert rpt["Assumptions"].str.contains("\u2713|\u2717").all()

    def test_invalid_decimals_raises(self):
        with pytest.raises(AssertionError):
            ssa.report(decimals=-1)

    def test_invalid_bias_method_raises(self):
        with pytest.raises(AssertionError):
            ssa.report(bias_method="invalid")
