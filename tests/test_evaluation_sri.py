"""Regression tests for EpochByEpochAgreement against the SRI Analytical Pipeline.

Reference values are taken from AnalyticalPipeline_v1.0.0.html (v1.0.0) published at:
  https://github.com/SRI-human-sleep/sleep-trackers-performance
  https://doi.org/10.1093/sleep/zsaa170

Ground-truth values live in evaluation_sri_full.json.

Dataset notes
-------------
- sample_data_sri.csv contains all 14 subjects (sbj01-sbj14) with complete epoch data.
- Integer stage encoding:  0 = Wake,  1 = Light (N1+N2),  2 = Deep (N3),  3 = REM
- For binary SLEEP/WAKE analyses 1/2/3 are collapsed to "S" (Sleep).

What is tested
--------------
- Per-subject recall (= sensitivity) and specificity for all 14 subjects × 4 stages,
  and their arithmetic means at group level (Block 12 / Block 17 of the R pipeline).
- Group-level mean PPV and NPV per stage (Block 17 advanced metrics).
- Per-subject accuracy, sensitivity, and specificity for binary SLEEP/WAKE classification,
  and their group means (Section 3.2 / 3.3 of the R pipeline).
- Per-subject sleep architecture measures for both the reference and device scorers:
  TST, SE, SOL, stage durations (Light/Deep/REM) and stage percentages
  (%Light/%Deep/%REM) (Section 2.1 of the R pipeline).
- Per-subject device − reference differences for TST, SE, SOL, WASO, stage durations,
  and stage percentages, via SleepStatsAgreement (Section 2.2 of the R pipeline).
- WASO per subject for both scorers.  The R pipeline counts all wake epochs from the
  first sleep epoch to the end of the recording (``ebe2sleep.R`` lines 47–50), which
  includes post-sleep wake after the final sleep epoch.  YASA's built-in ``WASO`` counts
  only wake within the Sleep Period Time (first to last sleep epoch).  The two definitions
  are equivalent to ``TIB − SOL − TST`` and ``SPT − TST`` respectively; the tests use
  ``TIB − SOL − TST``, which matches the R pipeline for all 14 subjects.
- Pooled ("sum") group recall and specificity per stage: epochs from all 14 subjects are
  concatenated into a single session and metrics are computed on the full pool
  (``group_ebe_staging["basic_sum"]``).
- Confusion matrix absolute epoch counts pooled across all 14 subjects
  (``error_matrices["_condition_staging"]["absolute_sum"]``).

What is NOT tested
------------------
- Per-subject accuracy (4-stage): YASA's ``get_agreement()["accuracy"]`` is the fraction
  of correctly labelled epochs across all stages, whereas the R pipeline reports a binary
  (one-vs-rest) accuracy per stage.  These are numerically different quantities.
  (Binary accuracy is comparable and is tested above.)
- Per-subject PPV and NPV: the R pipeline reports these only at group level (Block 17),
  not per subject, so there is no per-subject reference to compare against.
- Cohen's kappa, PABAK, and prevalence index: the R pipeline computes these as
  one-vs-rest per stage; YASA's ``get_agreement()["kappa"]`` is an overall multiclass
  kappa.  The two definitions are not directly comparable.
- Bland-Altman group bias, limits of agreement, and confidence intervals: the R pipeline
  applies conditional regression-modelling depending on assumption tests, making the
  expected outputs data-dependent and complex to pin to fixed reference values.
"""

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from yasa import EpochByEpochAgreement, Hypnogram, SleepStatsAgreement

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
CSV_PATH = DATA_DIR / "sample_data_sri.csv.xz"
REF_PATH = DATA_DIR / "evaluation_sri_full.json"

# Integer → YASA stage-label mapping for 4-stage and binary datasets
_INT_TO_STR = {0: "W", 1: "Light", 2: "Deep", 3: "R"}
_INT_TO_STR_SW = {0: "W", 1: "S", 2: "S", 3: "S"}

# All subjects present in the CSV with complete epoch data
_COMPLETE_SUBJECTS = [f"sbj{i:02d}" for i in range(1, 15)]  # sbj01-sbj14

# The HTML report prints values rounded to 2 decimal places, so any reference value
# can differ from the true value by up to ±0.005 pp.  0.1 is comfortably larger:
#   YASA computes 80.247 → HTML shows 80.25 → difference 0.003 pp  ✓
_ATOL_SUBJECT = 0.1

# For group means the per-subject rounding errors can accumulate, but even in the
# worst case (all 14 values biased in the same direction) the error in the mean is
# 14 × 0.005 / 14 = 0.005 pp.  0.5 also absorbs any difference between the R
# pipeline's internal mean and a straight average of the rounded per-subject values.
_ATOL_GROUP = 0.5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_reference():
    with open(REF_PATH) as fh:
        return json.load(fh)


def _build_ebe(mapping, n_stages):
    """Load the SRI sample CSV and return an EpochByEpochAgreement instance."""
    df = pd.read_csv(CSV_PATH)
    ref_hyps, obs_hyps = {}, {}
    for subj, grp in df.groupby("subject"):
        ref_hyps[subj] = Hypnogram.from_integers(
            grp["reference"].values, mapping=mapping, n_stages=n_stages, scorer="Reference"
        )
        obs_hyps[subj] = Hypnogram.from_integers(
            grp["device"].values, mapping=mapping, n_stages=n_stages, scorer="Device"
        )
    return EpochByEpochAgreement(ref_hyps, obs_hyps)


def _build_pooled_ebe():
    """Build a single-session EBE from epochs of all 14 subjects concatenated in subject order.

    This reproduces the R pipeline's "sum" (pooled) condition, where all epochs are
    treated as one recording.  Subjects are sorted alphabetically (sbj01 … sbj14) to
    match the R pipeline's ordering.
    """
    df = pd.read_csv(CSV_PATH)
    subjects_sorted = sorted(df["subject"].unique())
    ref_all = np.concatenate([df[df["subject"] == s]["reference"].values for s in subjects_sorted])
    obs_all = np.concatenate([df[df["subject"] == s]["device"].values for s in subjects_sorted])
    h_ref = Hypnogram.from_integers(ref_all, mapping=_INT_TO_STR, n_stages=4, scorer="Reference")
    h_obs = Hypnogram.from_integers(obs_all, mapping=_INT_TO_STR, n_stages=4, scorer="Device")
    return EpochByEpochAgreement({"all": h_ref}, {"all": h_obs})


# ---------------------------------------------------------------------------
# Module-level fixtures (built once for the whole test module)
# ---------------------------------------------------------------------------

_REF = _load_reference()

# 4-stage (WAKE / LIGHT / DEEP / REM)
_EBE = _build_ebe(_INT_TO_STR, n_stages=4)
_BYSTAGE = _EBE.get_agreement_bystage()  # MultiIndex: (stage, sleep_id)
_AGREEMENT = _EBE.get_agreement()  # Index: sleep_id
_SLEEP_STATS = _EBE.get_sleep_stats()  # MultiIndex: (scorer, sleep_id)

# Binary (SLEEP / WAKE)
_EBE_SW = _build_ebe(_INT_TO_STR_SW, n_stages=2)
_BYSTAGE_SW = _EBE_SW.get_agreement_bystage()
_AGREEMENT_SW = _EBE_SW.get_agreement()

# Pooled single-session EBE (all 14 subjects concatenated → reproduces "sum" condition)
_EBE_POOLED = _build_pooled_ebe()
_BYSTAGE_POOLED = _EBE_POOLED.get_agreement_bystage()  # MultiIndex: (stage, "all")

# Discrepancy analysis (Device − Reference differences per subject)
_REF_SS = _SLEEP_STATS.xs("Reference", level="scorer")
_OBS_SS = _SLEEP_STATS.xs("Device", level="scorer")
_SSA = SleepStatsAgreement(_REF_SS, _OBS_SS, ref_scorer="Reference", obs_scorer="Device")
# Precompute per-subject differences: rows=sleep_id, cols=sleep_stat
_DIFFS = (_SSA.data["Device"] - _SSA.data["Reference"]).unstack("sleep_stat")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSRIPerSubject(unittest.TestCase):
    """Per-subject per-stage recall and specificity must match the R pipeline (Block 12).

    recall (YASA) == sensitivity (R) == TP / (TP + FN)
    specificity (YASA) == specificity (R) == TN / (TN + FP)  (one-vs-rest)
    All sbj01-sbj14 × 4 stages are checked via subTest.
    Tolerance: 0.1 percentage points to cover rounding in the HTML source.
    """

    _STAGES = ("WAKE", "LIGHT", "DEEP", "REM")

    def test_recall(self):
        for subj in _COMPLETE_SUBJECTS:
            for stage in self._STAGES:
                with self.subTest(subj=subj, stage=stage):
                    expected = _REF["per_subject"][subj][stage]["sensitivity"]
                    actual = _BYSTAGE.loc[(stage, subj), "recall"] * 100
                    np.testing.assert_allclose(
                        actual,
                        expected,
                        atol=_ATOL_SUBJECT,
                        err_msg=f"{subj}/{stage}: recall {actual:.2f} ≠ {expected:.2f} pp",
                    )

    def test_specificity(self):
        for subj in _COMPLETE_SUBJECTS:
            for stage in self._STAGES:
                with self.subTest(subj=subj, stage=stage):
                    expected = _REF["per_subject"][subj][stage]["specificity"]
                    actual = _BYSTAGE.loc[(stage, subj), "specificity"] * 100
                    np.testing.assert_allclose(
                        actual,
                        expected,
                        atol=_ATOL_SUBJECT,
                        err_msg=f"{subj}/{stage}: specificity {actual:.2f} ≠ {expected:.2f} pp",
                    )


class TestSRIPerSubjectSleepWake(unittest.TestCase):
    """Per-subject binary SLEEP/WAKE accuracy, sensitivity, and specificity (Section 3.2).

    EpochByEpochAgreement is built with n_stages=2 (stages 1/2/3 collapsed to SLEEP).
    For the SLEEP stage:
      recall      == sensitivity (proportion of sleep epochs correctly classified)
      specificity == wake sensitivity (proportion of wake epochs correctly classified)
    Accuracy from get_agreement() is directly comparable in the binary case.
    Tolerance: 0.1 percentage points.
    """

    def test_accuracy(self):
        for subj in _COMPLETE_SUBJECTS:
            with self.subTest(subj=subj):
                expected = _REF["per_subject_sleep_wake"][subj]["accuracy"]
                actual = _AGREEMENT_SW.loc[subj, "accuracy"] * 100
                np.testing.assert_allclose(
                    actual,
                    expected,
                    atol=_ATOL_SUBJECT,
                    err_msg=f"{subj}: accuracy {actual:.2f} ≠ {expected:.2f} pp",
                )

    def test_sensitivity(self):
        for subj in _COMPLETE_SUBJECTS:
            with self.subTest(subj=subj):
                expected = _REF["per_subject_sleep_wake"][subj]["sensitivity"]
                actual = _BYSTAGE_SW.loc[("SLEEP", subj), "recall"] * 100
                np.testing.assert_allclose(
                    actual,
                    expected,
                    atol=_ATOL_SUBJECT,
                    err_msg=f"{subj}: sleep sensitivity {actual:.2f} ≠ {expected:.2f} pp",
                )

    def test_specificity(self):
        for subj in _COMPLETE_SUBJECTS:
            with self.subTest(subj=subj):
                expected = _REF["per_subject_sleep_wake"][subj]["specificity"]
                actual = _BYSTAGE_SW.loc[("SLEEP", subj), "specificity"] * 100
                np.testing.assert_allclose(
                    actual,
                    expected,
                    atol=_ATOL_SUBJECT,
                    err_msg=f"{subj}: wake sensitivity {actual:.2f} ≠ {expected:.2f} pp",
                )


class TestSRIGroupMeans(unittest.TestCase):
    """Group-level mean metrics must match 14-subject means from Block 17 (basic_avg / advanced_avg).

    All sbj01-sbj14 are included.
    Tolerance: 0.5 pp to cover accumulated rounding across 14 subjects.
    """

    _STAGES = ("WAKE", "LIGHT", "DEEP", "REM")

    def _group_mean(self, metric):
        """Mean of ``metric`` across all 14 subjects, grouped by stage."""
        return (
            _BYSTAGE.loc[:, metric]
            .loc[(slice(None), _COMPLETE_SUBJECTS)]
            .groupby(level="stage")
            .mean()
            * 100
        )

    def test_mean_recall_by_stage(self):
        mean_recall = self._group_mean("recall")
        for stage in self._STAGES:
            expected = _REF["group_ebe_staging"]["basic_avg"][stage]["sensitivity"]["mean"]
            np.testing.assert_allclose(
                mean_recall[stage],
                expected,
                atol=_ATOL_GROUP,
                err_msg=f"Mean recall {stage}: {mean_recall[stage]:.2f} pp ≠ expected {expected:.2f} pp",
            )

    def test_mean_specificity_by_stage(self):
        mean_spec = self._group_mean("specificity")
        for stage in self._STAGES:
            expected = _REF["group_ebe_staging"]["basic_avg"][stage]["specificity"]["mean"]
            np.testing.assert_allclose(
                mean_spec[stage],
                expected,
                atol=_ATOL_GROUP,
                err_msg=f"Mean specificity {stage}: {mean_spec[stage]:.2f} pp ≠ expected {expected:.2f} pp",
            )

    def test_mean_ppv_by_stage(self):
        """Mean precision (PPV) per stage matches Block 17 advanced_avg."""
        mean_ppv = self._group_mean("precision")
        for stage in self._STAGES:
            expected = _REF["group_ebe_staging"]["advanced_avg"][stage]["ppv"]["mean"]
            np.testing.assert_allclose(
                mean_ppv[stage],
                expected,
                atol=_ATOL_GROUP,
                err_msg=f"Mean PPV {stage}: {mean_ppv[stage]:.2f} pp ≠ expected {expected:.2f} pp",
            )

    def test_mean_npv_by_stage(self):
        """Mean NPV per stage matches Block 17 advanced_avg."""
        mean_npv = self._group_mean("npv")
        for stage in self._STAGES:
            expected = _REF["group_ebe_staging"]["advanced_avg"][stage]["npv"]["mean"]
            np.testing.assert_allclose(
                mean_npv[stage],
                expected,
                atol=_ATOL_GROUP,
                err_msg=f"Mean NPV {stage}: {mean_npv[stage]:.2f} pp ≠ expected {expected:.2f} pp",
            )


class TestSRIGroupMeansSleepWake(unittest.TestCase):
    """Group-level binary SLEEP/WAKE mean metrics (Section 3.3 basic_avg).

    Tolerance: 0.5 pp.
    """

    def _group_mean_sw(self, metric):
        return _BYSTAGE_SW.loc[:, metric].loc[("SLEEP", _COMPLETE_SUBJECTS)].mean() * 100

    def test_mean_sensitivity(self):
        expected = _REF["group_ebe_sleep_wake"]["basic_avg"]["SLEEP"]["sensitivity"]["mean"]
        actual = self._group_mean_sw("recall")
        np.testing.assert_allclose(
            actual,
            expected,
            atol=_ATOL_GROUP,
            err_msg=f"Mean sleep sensitivity: {actual:.2f} pp ≠ expected {expected:.2f} pp",
        )

    def test_mean_specificity(self):
        expected = _REF["group_ebe_sleep_wake"]["basic_avg"]["SLEEP"]["specificity"]["mean"]
        actual = self._group_mean_sw("specificity")
        np.testing.assert_allclose(
            actual,
            expected,
            atol=_ATOL_GROUP,
            err_msg=f"Mean wake sensitivity: {actual:.2f} pp ≠ expected {expected:.2f} pp",
        )


class TestSRISleepStats(unittest.TestCase):
    """Per-subject sleep architecture measures must match the R pipeline (Section 2.1).

    Compares YASA's get_sleep_stats() output against per_subject_sleep_measures in
    the reference JSON for both the Reference and Device scorers.
    Tolerance: 0.1 (minutes or percentage points) to cover rounding in the HTML source.
    """

    # Maps YASA column name → (json_ref_key, json_device_key).
    # WASO is tested separately below using TIB − SOL − TST (see test_waso docstring).
    _COL_MAP = {
        "TIB": ("TIB", "TIB"),
        "TST": ("TST_ref", "TST_device"),
        "SE": ("SE_ref", "SE_device"),
        "SOL": ("SOL_ref", "SOL_device"),
        "LIGHT": ("Light_ref", "Light_device"),
        "DEEP": ("Deep_ref", "Deep_device"),
        "REM": ("REM_ref", "REM_device"),
        "%LIGHT": ("LightPerc_ref", "LightPerc_device"),
        "%DEEP": ("DeepPerc_ref", "DeepPerc_device"),
        "%REM": ("REMPerc_ref", "REMPerc_device"),
    }

    def test_reference_scorer(self):
        ss = _SLEEP_STATS.xs("Reference", level="scorer")
        for subj in _COMPLETE_SUBJECTS:
            for yasa_col, (ref_key, _) in self._COL_MAP.items():
                with self.subTest(subj=subj, col=yasa_col):
                    expected = _REF["per_subject_sleep_measures"][subj][ref_key]
                    actual = ss.loc[subj, yasa_col]
                    np.testing.assert_allclose(
                        actual,
                        expected,
                        atol=_ATOL_SUBJECT,
                        err_msg=f"{subj} Reference {yasa_col}: {actual:.2f} ≠ {expected:.2f}",
                    )

    def test_device_scorer(self):
        ss = _SLEEP_STATS.xs("Device", level="scorer")
        for subj in _COMPLETE_SUBJECTS:
            for yasa_col, (_, dev_key) in self._COL_MAP.items():
                with self.subTest(subj=subj, col=yasa_col):
                    expected = _REF["per_subject_sleep_measures"][subj][dev_key]
                    actual = ss.loc[subj, yasa_col]
                    np.testing.assert_allclose(
                        actual,
                        expected,
                        atol=_ATOL_SUBJECT,
                        err_msg=f"{subj} Device {yasa_col}: {actual:.2f} ≠ {expected:.2f}",
                    )

    def test_waso(self):
        """WASO computed as TIB − SOL − TST matches the R pipeline definition.

        The R pipeline (ebe2sleep.R lines 47–50) counts wake epochs from the first sleep
        epoch to the END of the recording:
            WASO = nrow(sleepID[(SOL_epochs+1):end] where stage==0) × epochLength / 60
        This includes post-sleep wake after the final sleep epoch and is algebraically
        equal to TIB − SOL − TST.

        YASA's built-in WASO counts only wake within the Sleep Period Time (first to last
        sleep epoch), which is equivalent to SPT − TST.  For subjects with post-sleep wake
        (sbj09, sbj11) the two values differ; TIB − SOL − TST agrees with the R pipeline
        for all 14 subjects and is used here.
        """
        for scorer, json_key in (("Reference", "WASO_ref"), ("Device", "WASO_device")):
            ss = _SLEEP_STATS.xs(scorer, level="scorer")
            for subj in _COMPLETE_SUBJECTS:
                with self.subTest(scorer=scorer, subj=subj):
                    expected = _REF["per_subject_sleep_measures"][subj][json_key]
                    actual = ss.loc[subj, "TIB"] - ss.loc[subj, "SOL"] - ss.loc[subj, "TST"]
                    np.testing.assert_allclose(
                        actual,
                        expected,
                        atol=_ATOL_SUBJECT,
                        err_msg=f"{subj} {scorer} WASO: {actual:.2f} ≠ {expected:.2f}",
                    )


class TestSRIDiscrepancies(unittest.TestCase):
    """Per-subject device − reference differences must match the R pipeline (Section 2.2).

    Uses SleepStatsAgreement.data to compute per-subject Device − Reference differences
    and compares against per_subject_discrepancies_staging in the reference JSON.
    WASO differences are tested separately in test_waso_differences (see that method
    for the definition used).
    Tolerance: 0.1 (minutes or percentage points).
    """

    # Maps YASA sleep_stat name → json key for the difference
    _DIFF_MAP = {
        "TST": "TST_diff",
        "SE": "SE_diff",
        "SOL": "SOL_diff",
        "LIGHT": "Light_diff",
        "DEEP": "Deep_diff",
        "REM": "REM_diff",
        "%LIGHT": "LightPerc_diff",
        "%DEEP": "DeepPerc_diff",
        "%REM": "REMPerc_diff",
    }

    def test_per_subject_differences(self):
        for subj in _COMPLETE_SUBJECTS:
            for yasa_stat, json_key in self._DIFF_MAP.items():
                with self.subTest(subj=subj, stat=yasa_stat):
                    expected = _REF["per_subject_discrepancies_staging"][subj][json_key]
                    actual = _DIFFS.loc[subj, yasa_stat]
                    np.testing.assert_allclose(
                        actual,
                        expected,
                        atol=_ATOL_SUBJECT,
                        err_msg=f"{subj} {yasa_stat} diff: {actual:.2f} ≠ {expected:.2f}",
                    )

    def test_waso_differences(self):
        """WASO Device − Reference differences using TIB − SOL − TST match the R pipeline."""
        ss_ref = _SLEEP_STATS.xs("Reference", level="scorer")
        ss_dev = _SLEEP_STATS.xs("Device", level="scorer")
        for subj in _COMPLETE_SUBJECTS:
            with self.subTest(subj=subj):
                expected = _REF["per_subject_discrepancies_staging"][subj]["WASO_diff"]
                waso_ref = (
                    ss_ref.loc[subj, "TIB"] - ss_ref.loc[subj, "SOL"] - ss_ref.loc[subj, "TST"]
                )
                waso_dev = (
                    ss_dev.loc[subj, "TIB"] - ss_dev.loc[subj, "SOL"] - ss_dev.loc[subj, "TST"]
                )
                actual = waso_dev - waso_ref
                np.testing.assert_allclose(
                    actual,
                    expected,
                    atol=_ATOL_SUBJECT,
                    err_msg=f"{subj} WASO diff: {actual:.2f} ≠ {expected:.2f}",
                )


class TestSRIPooledMetrics(unittest.TestCase):
    """Pooled recall and specificity must match group_ebe_staging["basic_sum"].

    All 14 subjects' epochs are concatenated into a single session (sbj01…sbj14 order)
    to reproduce the R pipeline's "sum" condition.  Tolerance: 0.1 pp (single session,
    no inter-subject rounding accumulation).
    """

    _STAGES = ("WAKE", "LIGHT", "DEEP", "REM")

    def test_pooled_recall(self):
        for stage in self._STAGES:
            with self.subTest(stage=stage):
                expected = _REF["group_ebe_staging"]["basic_sum"][stage]["sensitivity"]
                # Single-session EBE: _BYSTAGE_POOLED has only the "stage" index level.
                actual = _BYSTAGE_POOLED.loc[stage, "recall"] * 100
                np.testing.assert_allclose(
                    actual,
                    expected,
                    atol=_ATOL_SUBJECT,
                    err_msg=f"Pooled recall {stage}: {actual:.2f} pp ≠ expected {expected:.2f} pp",
                )

    def test_pooled_specificity(self):
        for stage in self._STAGES:
            with self.subTest(stage=stage):
                expected = _REF["group_ebe_staging"]["basic_sum"][stage]["specificity"]
                actual = _BYSTAGE_POOLED.loc[stage, "specificity"] * 100
                np.testing.assert_allclose(
                    actual,
                    expected,
                    atol=_ATOL_SUBJECT,
                    err_msg=f"Pooled specificity {stage}: {actual:.2f} pp ≠ expected {expected:.2f} pp",
                )


class TestSRIConfusionMatrixValues(unittest.TestCase):
    """Pooled absolute confusion matrix must match error_matrices absolute_sum.

    The pooled single-session EBE (all 14 subjects concatenated) is used to
    reproduce the R pipeline's aggregate confusion matrix.  Cells are accessed
    by label (row = reference stage, column = device stage).
    """

    # Maps (reference_label, device_label) → JSON keys in absolute_sum
    _CELLS = {
        ("WAKE", "WAKE"): ("wake_ref", "device_wake"),
        ("WAKE", "LIGHT"): ("wake_ref", "device_light"),
        ("WAKE", "DEEP"): ("wake_ref", "device_deep"),
        ("WAKE", "REM"): ("wake_ref", "device_REM"),
        ("LIGHT", "WAKE"): ("light_ref", "device_wake"),
        ("LIGHT", "LIGHT"): ("light_ref", "device_light"),
        ("LIGHT", "DEEP"): ("light_ref", "device_deep"),
        ("LIGHT", "REM"): ("light_ref", "device_REM"),
        ("DEEP", "WAKE"): ("deep_ref", "device_wake"),
        ("DEEP", "LIGHT"): ("deep_ref", "device_light"),
        ("DEEP", "DEEP"): ("deep_ref", "device_deep"),
        ("DEEP", "REM"): ("deep_ref", "device_REM"),
        ("REM", "WAKE"): ("REM_ref", "device_wake"),
        ("REM", "LIGHT"): ("REM_ref", "device_light"),
        ("REM", "DEEP"): ("REM_ref", "device_deep"),
        ("REM", "REM"): ("REM_ref", "device_REM"),
    }

    def test_confusion_matrix_cells(self):
        cm = _EBE_POOLED.get_confusion_matrix()
        em = _REF["error_matrices"]["_condition_staging"]["absolute_sum"]
        for (ref_label, dev_label), (row_key, col_key) in self._CELLS.items():
            with self.subTest(ref=ref_label, device=dev_label):
                expected = em[row_key][col_key]
                actual = cm.loc[ref_label, dev_label]
                self.assertEqual(
                    actual,
                    expected,
                    msg=f"CM[{ref_label}, {dev_label}]: {actual} ≠ {expected}",
                )


class TestSRISanity(unittest.TestCase):
    """Sanity checks on dataset size and output shapes."""

    def test_n_subjects(self):
        self.assertEqual(_EBE.n_sessions, 14)

    def test_bystage_index_names(self):
        self.assertEqual(_BYSTAGE.index.names, ["stage", "sleep_id"])

    def test_bystage_stages(self):
        stages = sorted(_BYSTAGE.index.get_level_values("stage").unique())
        self.assertEqual(stages, ["DEEP", "LIGHT", "REM", "WAKE"])

    def test_bystage_sw_stages(self):
        stages = sorted(_BYSTAGE_SW.index.get_level_values("stage").unique())
        self.assertEqual(stages, ["SLEEP", "WAKE"])

    def test_bystage_columns(self):
        expected = {"fbeta", "npv", "precision", "recall", "specificity", "support"}
        self.assertEqual(set(_BYSTAGE.columns), expected)

    def test_agreement_index_size(self):
        self.assertEqual(len(_AGREEMENT), 14)

    def test_precision_in_bounds(self):
        self.assertTrue((_BYSTAGE["precision"] >= 0).all())
        self.assertTrue((_BYSTAGE["precision"] <= 1).all())

    def test_recall_in_bounds(self):
        self.assertTrue((_BYSTAGE["recall"] >= 0).all())
        self.assertTrue((_BYSTAGE["recall"] <= 1).all())

    def test_specificity_in_bounds(self):
        self.assertTrue((_BYSTAGE["specificity"] >= 0).all())
        self.assertTrue((_BYSTAGE["specificity"] <= 1).all())

    def test_npv_in_bounds(self):
        self.assertTrue((_BYSTAGE["npv"] >= 0).all())
        self.assertTrue((_BYSTAGE["npv"] <= 1).all())

    def test_kappa_in_bounds(self):
        self.assertTrue((_AGREEMENT["kappa"] >= -1).all())
        self.assertTrue((_AGREEMENT["kappa"] <= 1).all())

    def test_confusion_matrix_shape(self):
        cm = _EBE.get_confusion_matrix(sleep_id="sbj01")
        self.assertEqual(cm.shape, (4, 4))

    def test_confusion_matrix_pooled_shape(self):
        # The single-session pooled EBE always produces a 4×4 matrix.
        cm = _EBE_POOLED.get_confusion_matrix()
        self.assertEqual(cm.shape, (4, 4))


if __name__ == "__main__":
    unittest.main()
