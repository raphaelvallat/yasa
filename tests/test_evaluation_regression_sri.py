"""Regression tests for EpochByEpochAgreement and SleepStatsAgreement against the SRI Analytical
Pipeline.

Reference values are taken from AnalyticalPipeline_v1.0.0.html (v1.0.0) published at:
  https://github.com/SRI-human-sleep/sleep-trackers-performance
  https://doi.org/10.1093/sleep/zsaa170

Ground-truth values live in evaluation_sri_full.json.

Dataset notes
-------------
- sample_data_sri.csv.xz contains all 14 subjects (sbj01-sbj14) with complete epoch data.
- Integer stage encoding:  0 = Wake,  1 = Light (N1+N2),  2 = Deep (N3),  3 = REM
- For binary SLEEP/WAKE analyses 1/2/3 are collapsed to "S" (Sleep).

What is tested
--------------
- Per-subject recall (= sensitivity) and specificity for all 14 subjects × 4 stages
  (Block 12 of the R pipeline). Group means are arithmetic means of these values and are
  therefore not tested separately.
- Group-level mean PPV and NPV per stage (Block 17 advanced metrics), which have no
  per-subject reference.
- Per-subject accuracy, sensitivity, and specificity for binary SLEEP/WAKE classification
  (Section 3.2 of the R pipeline).
- Per-subject sleep architecture measures for both the reference and device scorers:
  TST, SE, SOL, stage durations (Light/Deep/REM) and stage percentages
  (%Light/%Deep/%REM) (Section 2.1 of the R pipeline).
- Per-subject device − reference differences for TST, SE, SOL, stage durations,
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
- Proportional error matrix (Section 3.1, ``error_matrices["_condition_staging"]["proportional_avg"]``):
  mean and SD across subjects of each row-normalized cell, and the 95% CIs from R's "basic"
  bootstrap, compared against YASA's ``method="basic"`` with a looser tolerance.

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
  expected outputs data-dependent and complex to pin to fixed reference values. These outputs
  are unit-tested against direct ``scipy.stats`` computations in ``test_evaluation.py``.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
_STAGES = ("WAKE", "LIGHT", "DEEP", "REM")

# The HTML report prints values rounded to 2 decimal places, so any reference value
# can differ from the true value by up to ±0.005 pp.  0.1 is comfortably larger:
#   YASA computes 80.247 → HTML shows 80.25 → difference 0.003 pp  ✓
_ATOL_SUBJECT = 0.1

# For group means the per-subject rounding errors can accumulate, but even in the
# worst case (all 14 values biased in the same direction) the error in the mean is
# 14 × 0.005 / 14 = 0.005 pp.  0.5 also absorbs any difference between the R
# pipeline's internal mean and a straight average of the rounded per-subject values.
_ATOL_GROUP = 0.5

# Maps YASA sleep statistic → (json_ref_key, json_device_key) in per_subject_sleep_measures.
# WASO is tested separately using TIB − SOL − TST (see test_waso docstring).
_SLEEP_MEASURES = {
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

# Maps YASA sleep statistic → json key in per_subject_discrepancies_staging
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

# Maps YASA stage label → row / column keys of the reference error matrices
_CM_ROWS = {"WAKE": "wake_ref", "LIGHT": "light_ref", "DEEP": "deep_ref", "REM": "REM_ref"}
_CM_COLS = {
    "WAKE": "device_wake",
    "LIGHT": "device_light",
    "DEEP": "device_deep",
    "REM": "device_REM",
}
# All (reference_label, device_label) cells of the 4 x 4 confusion matrix
_CM_CELLS = [(r, c) for r in _CM_ROWS for c in _CM_COLS]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_close(actual, expected, atol, what):
    """Compare all values at once, listing every mismatched entry in the error message."""
    actual = pd.Series(actual, dtype=float)
    expected = pd.Series(expected, dtype=float).reindex(actual.index)
    bad = ~np.isclose(actual, expected, rtol=1e-7, atol=atol)
    table = pd.DataFrame({"actual": actual, "expected": expected})[bad]
    assert not bad.any(), f"{what} differs from the R pipeline:\n{table.to_string()}"


def _build_ebe(df, mapping, n_stages):
    """Return an EpochByEpochAgreement instance from the SRI sample dataframe."""
    ref_hyps, obs_hyps = {}, {}
    for subj, grp in df.groupby("subject"):
        ref_hyps[subj] = Hypnogram.from_integers(
            grp["reference"].values, mapping=mapping, n_stages=n_stages, scorer="Reference"
        )
        obs_hyps[subj] = Hypnogram.from_integers(
            grp["device"].values, mapping=mapping, n_stages=n_stages, scorer="Device"
        )
    return EpochByEpochAgreement(ref_hyps, obs_hyps)


# ---------------------------------------------------------------------------
# Fixtures (built once for the whole test module)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sri_ref():
    """Ground-truth values of the R pipeline."""
    with open(REF_PATH) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def sri_df():
    return pd.read_csv(CSV_PATH)


@pytest.fixture(scope="module")
def sri_ebe(sri_df):
    """4-stage (WAKE / LIGHT / DEEP / REM)."""
    return _build_ebe(sri_df, _INT_TO_STR, n_stages=4)


@pytest.fixture(scope="module")
def sri_bystage(sri_ebe):
    return sri_ebe.get_agreement_bystage()  # MultiIndex: (stage, sleep_id)


@pytest.fixture(scope="module")
def sri_sleep_stats(sri_ebe):
    return sri_ebe.get_sleep_stats()  # MultiIndex: (scorer, sleep_id)


@pytest.fixture(scope="module")
def sri_ebe_sw(sri_df):
    """Binary (SLEEP / WAKE)."""
    return _build_ebe(sri_df, _INT_TO_STR_SW, n_stages=2)


@pytest.fixture(scope="module")
def sri_bystage_sw(sri_ebe_sw):
    return sri_ebe_sw.get_agreement_bystage()


@pytest.fixture(scope="module")
def sri_agreement_sw(sri_ebe_sw):
    return sri_ebe_sw.get_agreement()


@pytest.fixture(scope="module")
def sri_ebe_pooled(sri_df):
    """Build a single-session EBE from epochs of all 14 subjects concatenated in subject order.

    This reproduces the R pipeline's "sum" (pooled) condition, where all epochs are
    treated as one recording.  Subjects are sorted alphabetically (sbj01 … sbj14) to
    match the R pipeline's ordering.
    """
    df = sri_df
    subjects_sorted = sorted(df["subject"].unique())
    ref_all = np.concatenate([df[df["subject"] == s]["reference"].values for s in subjects_sorted])
    obs_all = np.concatenate([df[df["subject"] == s]["device"].values for s in subjects_sorted])
    h_ref = Hypnogram.from_integers(ref_all, mapping=_INT_TO_STR, n_stages=4, scorer="Reference")
    h_obs = Hypnogram.from_integers(obs_all, mapping=_INT_TO_STR, n_stages=4, scorer="Device")
    return EpochByEpochAgreement({"all": h_ref}, {"all": h_obs})


@pytest.fixture(scope="module")
def sri_bystage_pooled(sri_ebe_pooled):
    # Single-session EBE: only the "stage" index level
    return sri_ebe_pooled.get_agreement_bystage()


@pytest.fixture(scope="module")
def sri_diffs(sri_sleep_stats):
    """Per-subject Device − Reference differences: rows=sleep_id, cols=sleep_stat."""
    ssa = SleepStatsAgreement(sri_sleep_stats)  # scorers taken from the index (Reference, Device)
    return (ssa.data["Device"] - ssa.data["Reference"]).unstack("sleep_stat")


@pytest.fixture(scope="module")
def sri_cm_basic(sri_ebe):
    """Proportional confusion matrix with R's "basic" bootstrap CIs."""
    return sri_ebe.get_confusion_matrix_proportional(
        ci_method="boot", bootstrap_kwargs={"n_resamples": 5000, "method": "basic", "rng": 0}
    )


@pytest.fixture(scope="module")
def sri_cm_expected(sri_ref):
    """Reference proportional error matrix: (mean, sd, ci_lower, ci_upper) per cell, in percent."""
    pattern = re.compile(r"([\d.]+) \(([\d.]+)\) \[([\d.]+), ([\d.]+)\]")
    ref = sri_ref["error_matrices"]["_condition_staging"]["proportional_avg"]
    expected = {}
    for ref_stage, dev_stage in _CM_CELLS:
        m = pattern.fullmatch(ref[_CM_ROWS[ref_stage]][_CM_COLS[dev_stage]])
        expected[(ref_stage, dev_stage)] = tuple(100 * float(x) for x in m.groups())
    return expected


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSRIPerSubject:
    """Per-subject per-stage recall and specificity must match the R pipeline (Block 12).

    recall (YASA) == sensitivity (R) == TP / (TP + FN)
    specificity (YASA) == specificity (R) == TN / (TN + FP)  (one-vs-rest)
    All sbj01-sbj14 × 4 stages are checked.
    Tolerance: 0.1 percentage points to cover rounding in the HTML source.
    """

    @pytest.mark.parametrize(
        "metric, ref_metric", [("recall", "sensitivity"), ("specificity", "specificity")]
    )
    def test_metric(self, sri_ref, sri_bystage, metric, ref_metric):
        cells = [(stage, subj) for stage in _STAGES for subj in _COMPLETE_SUBJECTS]
        expected = {(st, sb): sri_ref["per_subject"][sb][st][ref_metric] for st, sb in cells}
        actual = sri_bystage.loc[cells, metric]
        _assert_close(actual, expected, _ATOL_SUBJECT, metric)


class TestSRIPerSubjectSleepWake:
    """Per-subject binary SLEEP/WAKE accuracy, sensitivity, and specificity (Section 3.2).

    EpochByEpochAgreement is built with n_stages=2 (stages 1/2/3 collapsed to SLEEP).
    For the SLEEP stage:
      recall      == sensitivity (proportion of sleep epochs correctly classified)
      specificity == wake sensitivity (proportion of wake epochs correctly classified)
    Accuracy from get_agreement() is directly comparable in the binary case.
    Tolerance: 0.1 percentage points.
    """

    def test_accuracy(self, sri_ref, sri_agreement_sw):
        ref = sri_ref["per_subject_sleep_wake"]
        expected = {subj: ref[subj]["accuracy"] for subj in _COMPLETE_SUBJECTS}
        actual = sri_agreement_sw.loc[_COMPLETE_SUBJECTS, "accuracy"]
        _assert_close(actual, expected, _ATOL_SUBJECT, "accuracy")

    @pytest.mark.parametrize(
        "metric, ref_metric, label",
        [
            ("recall", "sensitivity", "sleep sensitivity"),
            ("specificity", "specificity", "wake sensitivity"),
        ],
    )
    def test_sleep_metric(self, sri_ref, sri_bystage_sw, metric, ref_metric, label):
        ref = sri_ref["per_subject_sleep_wake"]
        expected = {subj: ref[subj][ref_metric] for subj in _COMPLETE_SUBJECTS}
        actual = sri_bystage_sw.loc["SLEEP", metric].loc[_COMPLETE_SUBJECTS]
        _assert_close(actual, expected, _ATOL_SUBJECT, label)


class TestSRIGroupMeans:
    """Group-level mean metrics must match 14-subject means from Block 17 (advanced_avg).

    Mean precision (PPV) and NPV per stage. All sbj01-sbj14 are included.
    Tolerance: 0.5 pp to cover accumulated rounding across 14 subjects.
    """

    @pytest.mark.parametrize("metric, ref_metric", [("precision", "ppv"), ("npv", "npv")])
    def test_mean_by_stage(self, sri_ref, sri_bystage, metric, ref_metric):
        # Mean of `metric` across all 14 subjects, grouped by stage
        group_mean = (
            sri_bystage.loc[:, metric]
            .loc[(slice(None), _COMPLETE_SUBJECTS)]
            .groupby(level="stage")
            .mean()
        )
        ref = sri_ref["group_ebe_staging"]["advanced_avg"]
        expected = {stage: ref[stage][ref_metric]["mean"] for stage in _STAGES}
        _assert_close(group_mean.loc[list(_STAGES)], expected, _ATOL_GROUP, f"Mean {ref_metric}")


class TestSRISleepStats:
    """Per-subject sleep architecture measures must match the R pipeline (Section 2.1).

    Compares YASA's get_sleep_stats() output against per_subject_sleep_measures in
    the reference JSON for both the Reference and Device scorers.
    Tolerance: 0.1 (minutes or percentage points) to cover rounding in the HTML source.
    """

    @pytest.mark.parametrize("scorer", ["Reference", "Device"])
    def test_sleep_measures(self, sri_ref, sri_sleep_stats, scorer):
        ss = sri_sleep_stats.xs(scorer, level="scorer")
        ref = sri_ref["per_subject_sleep_measures"]
        cells = [(subj, col) for subj in _COMPLETE_SUBJECTS for col in _SLEEP_MEASURES]
        idx = 0 if scorer == "Reference" else 1
        expected = {(sb, col): ref[sb][_SLEEP_MEASURES[col][idx]] for sb, col in cells}
        actual = pd.Series({(sb, col): ss.loc[sb, col] for sb, col in cells})
        _assert_close(actual, expected, _ATOL_SUBJECT, f"{scorer} sleep measures")

    @pytest.mark.parametrize(
        "scorer, json_key", [("Reference", "WASO_ref"), ("Device", "WASO_device")]
    )
    def test_waso(self, sri_ref, sri_sleep_stats, scorer, json_key):
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
        ss = sri_sleep_stats.xs(scorer, level="scorer").loc[_COMPLETE_SUBJECTS]
        ref = sri_ref["per_subject_sleep_measures"]
        expected = {subj: ref[subj][json_key] for subj in _COMPLETE_SUBJECTS}
        actual = ss["TIB"] - ss["SOL"] - ss["TST"]
        _assert_close(actual, expected, _ATOL_SUBJECT, f"{scorer} WASO")


class TestSRIDiscrepancies:
    """Per-subject device − reference differences must match the R pipeline (Section 2.2).

    Uses SleepStatsAgreement.data to compute per-subject Device − Reference differences
    and compares against per_subject_discrepancies_staging in the reference JSON.
    Tolerance: 0.1 (minutes or percentage points).
    """

    def test_per_subject_differences(self, sri_ref, sri_diffs):
        ref = sri_ref["per_subject_discrepancies_staging"]
        cells = [(subj, stat) for subj in _COMPLETE_SUBJECTS for stat in _DIFF_MAP]
        expected = {(sb, stat): ref[sb][_DIFF_MAP[stat]] for sb, stat in cells}
        actual = pd.Series({(sb, stat): sri_diffs.loc[sb, stat] for sb, stat in cells})
        _assert_close(actual, expected, _ATOL_SUBJECT, "Device - Reference differences")


class TestSRIPooledMetrics:
    """Pooled recall and specificity must match group_ebe_staging["basic_sum"].

    All 14 subjects' epochs are concatenated into a single session (sbj01…sbj14 order)
    to reproduce the R pipeline's "sum" condition.  Tolerance: 0.1 pp (single session,
    no inter-subject rounding accumulation).
    """

    @pytest.mark.parametrize(
        "metric, ref_metric", [("recall", "sensitivity"), ("specificity", "specificity")]
    )
    def test_pooled_metric(self, sri_ref, sri_bystage_pooled, metric, ref_metric):
        ref = sri_ref["group_ebe_staging"]["basic_sum"]
        expected = {stage: ref[stage][ref_metric] for stage in _STAGES}
        actual = sri_bystage_pooled.loc[list(_STAGES), metric]
        _assert_close(actual, expected, _ATOL_SUBJECT, f"Pooled {metric}")


class TestSRIConfusionMatrixValues:
    """Pooled absolute confusion matrix must match error_matrices absolute_sum.

    The pooled single-session EBE (all 14 subjects concatenated) is used to
    reproduce the R pipeline's aggregate confusion matrix.  Cells are accessed
    by label (row = reference stage, column = device stage).
    """

    def test_confusion_matrix(self, sri_ref, sri_ebe_pooled):
        cm = sri_ebe_pooled.get_confusion_matrix()
        em = sri_ref["error_matrices"]["_condition_staging"]["absolute_sum"]
        expected = {(r, c): em[_CM_ROWS[r]][_CM_COLS[c]] for r, c in _CM_CELLS}
        actual = pd.Series({(r, c): cm.loc[r, c] for r, c in _CM_CELLS})
        _assert_close(actual, expected, 0, "Confusion matrix")


class TestSRIProportionalConfusionMatrix:
    """Group-level proportional error matrix (Section 3.1, ``error_matrices[...]["proportional_avg"]``).

    The reference reports, for each cell, the mean (SD) [95% bootstrap CI] across the 14 subjects
    of the proportion of reference-stage epochs classified into each device stage (YASA reports
    percentages). Means and SDs are deterministic (tolerance 1 pp = 2 x rounding of the reference
    proportions). The reference CIs come from R's "basic" bootstrap with its own random draws, so
    they are compared against YASA's ``method="basic"`` with a looser tolerance. YASA's default
    BCa method is a deliberate deviation from the R pipeline and has no published reference.
    """

    def test_all_subjects_contribute(self, sri_cm_basic):
        assert (sri_cm_basic["n_sessions"] == 14).all()

    @pytest.mark.parametrize(
        "column, pos, atol",
        [("mean", 0, 1.0), ("std", 1, 1.0), ("ci_lower", 2, 2.0), ("ci_upper", 3, 2.0)],
    )
    def test_cells(self, sri_cm_basic, sri_cm_expected, column, pos, atol):
        expected = {cell: values[pos] for cell, values in sri_cm_expected.items()}
        actual = pd.Series({cell: sri_cm_basic.at[cell, column] for cell in _CM_CELLS})
        _assert_close(actual, expected, atol, f"Proportional confusion matrix {column}")


def test_dataset_and_shapes(sri_ebe, sri_bystage, sri_bystage_sw, sri_ebe_pooled):
    assert sri_ebe.n_sessions == 14
    assert len(sri_ebe.get_agreement()) == 14
    stages = sorted(sri_bystage.index.get_level_values("stage").unique())
    assert stages == ["DEEP", "LIGHT", "REM", "WAKE"]
    stages_sw = sorted(sri_bystage_sw.index.get_level_values("stage").unique())
    assert stages_sw == ["SLEEP", "WAKE"]
    assert sri_ebe.get_confusion_matrix(sleep_id="sbj01").shape == (4, 4)
    assert sri_ebe_pooled.get_confusion_matrix().shape == (4, 4)
