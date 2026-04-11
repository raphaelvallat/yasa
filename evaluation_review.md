# YASA `evaluation.py` — Review Against Menghini et al. 2021 Pipeline

Reviewed against: `sleep-trackers-performance-master/AnalyticalPipeline_v1.0.0.Rmd` and its
companion R functions (`ebe2sleep.R`, `errorMatrix.R`, `indEBE.R`, `groupEBE.R`, `indDiscr.R`,
`groupDiscr.R`, `BAplot.R`).

## Status legend
- ✅ Implemented / Fixed
- ⚠️  Partial / In progress
- ❌ Missing / Not implemented
- 🐛 Bug

See https://github.com/raphaelvallat/yasa/pull/228

---

## EpochByEpochAgreement

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| 1 | **Specificity (TNR) missing** from `get_agreement_bystage()` | ✅ Implemented | Per-stage one-vs-rest confusion matrix in `scorer()` |
| 2 | **PABAK** not computed | ❌ | R: `epi.kappa()$pabak`. Not trivially added via sklearn |
| 3 | **NPV** missing from `get_agreement_bystage()` | ✅ Implemented | Per-stage one-vs-rest confusion matrix in `scorer()` |
| 4 | **Prevalence index / bias index** not computed | ❌ | R: `epi.kappa()$pindex`, `$bindex` |
| 5 | **ROC curves** missing | ❌ | R: `groupEBE(doROC=TRUE)` via `ROCR`. Could add with sklearn |
| 10 | **`pooled` mode** for `get_agreement()` not surfaced | ✅ Implemented | Added `pooled=False` parameter; `True` = R's `metricsType="sum"` |

### Notes
- Accuracy is reported 0–1 in YASA vs 0–100% in R (convention only, not a bug).
- R's `metricsType="avg"` (per-subject average) = YASA's default `get_agreement()` behavior.
- R's `metricsType="sum"` (pooled epochs) = YASA's new `get_agreement(pooled=True)`.
- `get_agreement_bystage()` now returns 6 metrics: `fbeta, npv, precision, recall, specificity, support`.

---

## SleepStatsAgreement

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| 6 | **Log transformation** missing | ✅ Implemented | `log_transform=True` param + Euser et al. (2008) back-transform; `"log"` loa_method. |
| 7 | **Individual discrepancy heatmap** (`indDiscr.R`) | ❌ | No equivalent; data available via `get_sleep_stats()` |
| 8 | **Calibration direction bug** | ✅ Fixed | See detailed investigation below (Bugs A, B, C) |

---

## Features where YASA goes beyond the R pipeline

| Feature | YASA | R pipeline |
|---------|------|------------|
| Calibration of new data | ✅ `calibrate()`, `get_calibration_func()` | ❌ Not present |
| MCC (Matthews Correlation Coefficient) | ✅ | ❌ |
| Balanced accuracy | ✅ | ❌ |
| BCa bootstrap (more robust) | ✅ | ❌ (R uses "basic" default) |
| Overlaid hypnogram plots | ✅ `plot_hypnograms()` | ❌ |
| MAD / median in group summary | ✅ `summary()` | ❌ |
| Human-readable report table | ✅ `report()` | ❌ |
| Bland-Altman plot | ✅ `plot_blandaltman()` | ✅ `BAplot.R` |

---

## Bland-Altman Plot (`BAplot.R` vs YASA)

✅ **`plot_blandaltman()` implemented** in `SleepStatsAgreement` (`evaluation.py:1604`).

### What `BAplot.R` produces vs YASA

| Element | R (`BAplot.R`) | YASA (`plot_blandaltman`) |
|---------|---------------|--------------------------|
| Scatter points | One dot per session | ✅ One dot per session |
| Bias line | Horizontal `mean(diff)` or regression `b0 + b1*ref` | ✅ Same; auto-selected from `assumptions` |
| Bias CI | t-CI or bootstrap band | ✅ Shaded band; method via `ci_method` |
| LoA lines | Constant `bias ± 1.96 SD` or `bias ± 2.46*(c0 + c1*ref)` | ✅ Same; auto-selected from `assumptions` |
| LoA CI | Dashed CI bands | ✅ Shaded bands |
| Flag biased | Red bias line if significant | ✅ `flag_biased=True` |
| Euser LoA | `bias ± ref × euser_slope` | ✅ `log_transform=True` |
| Marginal density | `ggExtra::ggMarginal` | ❌ Not implemented |
| `xaxis="mean"` | (obs+ref)/2 on x-axis | ❌ Only reference on x-axis |

### `BAplot.R` parameters not yet in YASA

| Parameter | Purpose | Status |
|-----------|---------|--------|
| `logTransf = TRUE/FALSE` | Use Euser back-transform for LoA | ✅ `log_transform=True` |
| `xaxis = "mean"` | X-axis: (obs+ref)/2 instead of ref | ❌ Not planned |
| `xlim`, `ylim` | Axis limits | ❌ Can be set via matplotlib post-call |

---

## Log transformation (Euser et al. 2008)

**Reference:** `groupDiscr.R` lines 208–264, `BAplot.R` log section, Euser et al. (2008) and Bland & Altman (1999).

### Design principles (simplified vs R)

- `log_transform` is **bool only** — no per-stat list. Mixed cases use two `SleepStatsAgreement` objects.
- Euser slope is **internal** — not exposed in `summary()`, only rendered in `plot_blandaltman` and `report`.
- `"log"` is a first-class **`loa_method` value**, alongside `"param"` and `"regr"`.
- `auto_methods["loa"]` returns `"log"` for all stats when `log_transform=True`.
- No `log_normal` in `assumptions` — normality of original diffs drives CI selection as before.
- Bootstrap CI for Euser slope handled inside existing `_generate_bootstrap_ci`, no new methods.

---

### Background and motivation

When differences between devices and reference scale proportionally with the measurement size (heteroscedasticity), a log transformation stabilises the variance. The Euser et al. (2008) method computes limits of agreement in log space and back-transforms them to a slope that multiplies the reference value:

```
LoA_upper = bias + slope × ref
LoA_lower = bias − slope × ref
slope = 2 * (exp(agreement × SD(log_diffs)) − 1) / (exp(agreement × SD(log_diffs)) + 1)
```

where `log_diffs = log(obs + ε) − log(ref + ε)` and `ε = 1e-4` to avoid `log(0)`.

This is the **only** LoA representation that applies for log-transformed stats. The standard `loa_lower`/`loa_upper` (constant) and `loa_intercept`/`loa_slope` (regression) representations do not apply to these stats.

---

## Regression tests against the Menghini et al. (2021) pipeline

`tests/test_evaluation_sri.py` loads all 14 subjects (10 766 epochs, 30-s epochs) from
`tests/data/sample_data_sri.csv` and compares YASA's output to reference values extracted
from the published HTML report (`AnalyticalPipeline_v1.0.0.html`), stored in
`tests/data/evaluation_sri_full.json`. 32 test methods are run via `unittest`.

### What is tested (with tolerance)

| Test class | What is checked | Tolerance |
|---|---|:-:|
| `TestSRIPerSubject` | Per-subject recall and specificity — 14 subjects × 4 stages (112 checks) | 0.1 pp |
| `TestSRIPerSubjectSleepWake` | Per-subject binary SLEEP/WAKE accuracy, sensitivity, specificity — 14 subjects (42 checks) | 0.1 pp |
| `TestSRIGroupMeans` | Group mean recall, specificity, PPV, NPV per stage — 4 metrics × 4 stages | 0.5 pp |
| `TestSRIGroupMeansSleepWake` | Group mean binary sensitivity and specificity | 0.5 pp |
| `TestSRISleepStats` | Per-subject TIB, TST, SE, SOL, stage durations, stage % for both scorers — 14 × 10 × 2 (280 checks) | 0.1 |
| `TestSRISleepStats` | Per-subject WASO as TIB − SOL − TST (R pipeline definition, including post-sleep wake) — 14 × 2 | 0.1 |
| `TestSRIDiscrepancies` | Per-subject Device − Reference differences for TST, SE, SOL, WASO, stage durations, stage % — 14 × 10 (140 checks) | 0.1 |
| `TestSRIPooledMetrics` | Pooled (all-epoch) recall and specificity per stage matching R's `metricsType="sum"` | 0.1 pp |
| `TestSRIConfusionMatrixValues` | All 16 cells of the pooled absolute confusion matrix, accessed by label | exact |
| `TestSRISanity` | Dataset size, output shapes, index names, stage labels, metrics in [0, 1] | — |

**Tolerance note:** 0.1 pp covers single rounding in the HTML source (±0.005 pp). 0.5 pp at group level covers accumulated rounding across 14 subjects.

**WASO note:** The R pipeline (`ebe2sleep.R` lines 47–50) counts wake epochs from the first sleep epoch to the **end of the recording**, including any post-sleep wake after the final sleep epoch. This is algebraically equivalent to `TIB − SOL − TST`. YASA's built-in `WASO` counts only wake within the Sleep Period Time (first to last sleep epoch), equivalent to `SPT − TST`. For subjects with post-sleep wake (sbj09, sbj11) the two definitions disagree. The tests use `TIB − SOL − TST`, which matches the R pipeline exactly for all 14 subjects.

### What cannot be tested (yet)

| Item | Reason |
|---|---|
| Per-subject accuracy (4-stage) | YASA accuracy = fraction correct across all stages; R pipeline reports binary one-vs-rest accuracy per stage. Numerically different. (Binary accuracy is tested.) |
| Per-subject PPV / NPV | R pipeline reports these at group level only; no per-subject reference. |
| Per-stage Cohen's κ, PABAK, prevalence index | R pipeline computes one-vs-rest; YASA `kappa` is multiclass. PABAK is not yet in YASA (item 2 above). |
| Bland-Altman bias, LoA, and CIs | R pipeline uses conditional regression depending on assumption tests, making expected outputs data-dependent and impractical to pin as fixed reference values. |
