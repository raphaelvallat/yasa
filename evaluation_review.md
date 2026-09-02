# YASA `evaluation.py` — Review Against Menghini et al. 2021 Pipeline

Reviewed against: `sleep-trackers-performance-master/AnalyticalPipeline_v1.0.0.Rmd` and its
companion R functions (`ebe2sleep.R`, `errorMatrix.R`, `indEBE.R`, `groupEBE.R`, `indDiscr.R`,
`groupDiscr.R`, `BAplot.R`), and the published paper (Menghini et al., 2021, SLEEP 44(2), zsaa170).

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
| 11 | **Undefined metrics counted as 0** (`zero_division=0` hard-coded in `get_agreement_bystage()`) | ✅ Fixed | New `zero_division` param, default `np.nan` (sklearn ≥ 1.3). Nights whose reference lacks a stage no longer drag down group means; `support == 0` flags them. Also applied to specificity/NPV. `0`, `1`, `"warn"` accepted. |
| 12 | **Proportional error matrix** (`errorMatrix.R`, `prop` condition: mean (SD) [95% CI] of row-normalized per-subject matrices) | ✅ Implemented | `get_confusion_matrix_proportional()`. Subject-then-group averaging, absent stages kept as NaN (not zero-filled as sklearn's `normalize="true"` does), `n_sessions` column. Participant bootstrap CI (same resampled sessions for all cells), `formatted=True` gives the `"mean (SD) [lo, hi]"` table. |
| 13 | **Metrics on 0–1 scale** vs R's 0–100 % | ✅ Fixed | All proportion-based EBE metrics (accuracy, balanced_acc, precision, recall, f1/fbeta, specificity, npv, proportional matrix) are now percentages. `kappa`, `mcc` unchanged (−1 to 1). |

### Notes
- All proportion-based EBE metrics are reported in percent (0–100), matching R. `kappa` and `mcc` stay on their natural −1 to 1 scale.
- R's `metricsType="avg"` (per-subject average) = YASA's default `get_agreement()` behavior.
- R's `metricsType="sum"` (pooled epochs) = YASA's new `get_agreement(pooled=True)`.
- `get_agreement_bystage()` now returns 6 metrics: `fbeta, npv, precision, recall, specificity, support`.
- Group summaries (`summary(by_stage=True)`) skip NaN (undefined) sessions; the per-metric `count` differs from `n_sessions` where a stage is absent.

### Justified deviations from the R pipeline
- **Bootstrap CI method.** `errorMatrix.R` uses the "basic" (reverse-percentile) bootstrap. YASA defaults to **BCa** (bias-corrected and accelerated, Efron 1987) for the proportional error matrix, as for `SleepStatsAgreement`, because it corrects for skew and bias of the bootstrap distribution of bounded proportions. `"basic"` and `"percentile"` remain available via `bootstrap_kwargs={"method": ...}`; the regression test validates `"basic"` against the published CIs; BCa has no published reference and is not compared.
- **Undefined cells.** The R pipeline has no subjects with missing stages in the sample data, so its behavior there is untested. YASA keeps 0/0 cells as NaN, excludes them from mean/SD/CI, drops bootstrap replicates in which a stage is absent from every resampled subject, and falls back to plain percentiles for degenerate (constant) cells where BCa is undefined.

---

## SleepStatsAgreement

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| 6 | **Log transformation** missing | ✅ Implemented | `log_transform=True` param + Euser et al. (2008) back-transform; `"log"` loa_method. |
| 7 | **Individual discrepancy heatmap** (`indDiscr.R`) | ❌ | No equivalent; data available via `get_sleep_stats()` |
| 8 | **Calibration direction bug** | ✅ Fixed | See detailed investigation below (Bugs A, B, C) |
| 9 | **Report table** (`groupDiscr.R` output: reference mean (SD), device mean (SD), bias [CI], LoA) | ✅ Implemented | `report()` now shows `mean (SD)` strings for both scorers, takes a `sleep_stats` subset (in the requested order), and `bias_ci` / `loa_ci` flags to omit CIs (no bootstrap is run when both are off). |
| 14 | **Euser LoA slope only reachable via private attribute** | ✅ Fixed | Public `loa_log_slope` property; `summary()` includes a `loa_log_slope` variable (center/lower/upper) when `log_transform=True`. `report()` and `plot_blandaltman()` read it through `summary()`. |
| 15 | `summary()` always bootstrapped all stats | ✅ Improved | `summary(ci_method=None)` returns point estimates only; `sleep_stats` restricts (and orders) rows and limits bootstrapping to the requested stats. |
| 16 | 🐛 **Paper eq. 2 not implemented** (proportional bias + homoscedastic → LoA = `bias_i ± 1.96 SD(residuals)`, parallel to the regression bias line). YASA drew horizontal LoA at `mean ± 1.96 SD(differences)`, i.e. mis-centred and too wide by `1/sqrt(1 − R²)`. | ✅ Fixed | New `loa_halfwidth` variable in `summary()` (`agreement × SD` of the bias-regression residuals, ddof=1 as in R's `sd(resid(lm))`), with parametric CI (`SE(SD) ≈ SD/sqrt(2n)`, Bland & Altman 1999) and bootstrap CI. `report()` shows `"bias ± hw [lo, hi]"` and `plot_blandaltman()` draws LoA parallel to the bias line whenever bias is `regr` and LoA is `param`. |
| 17 | **Minimal detectable change** (half the LoA width; Menghini 2021, Haghayegh 2020) not reported | ✅ Implemented | `"MDC"` column in `report()`. Constant LoA only (`(upper − lower)/2` for mean bias, `loa_halfwidth` for regression bias); `"n/a"` for regression/Euser LoA, where MDC varies with the reference value. |
| 18 | **Assumption tests are booleans only**; paper asks for test results "accompanied by visual inspection" and reports coefficients with CIs | ✅ Implemented | New `diagnostics` property (MultiIndex columns `assumption` × `metric`): `unbiased` (t, pvalue, cohen_d), `normal` (W, pvalue, skew, kurtosis), `constant_bias` and `homoscedastic` (slope, pvalue, r2). `assumptions` is now derived from it as `pvalue >= alpha`. |
| 19 | **Shapiro-Wilk over-rejects** at alpha 0.05 (e.g. n = 57, skew 0.07, p = 0.008 from a single heavy-tail point) | ✅ Mitigated | New `alpha_normal` constructor parameter, default 0.01, applied only to the `normal` flag; `alpha` (0.05) still applies to the other three tests. A stricter alpha is justified here because a normality violation only switches CIs to bootstrap (the R pipeline uses alpha 0.05 for all tests). This shifts, but does not remove, the sample-size dependence. |

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

✅ **`plot_blandaltman()` implemented** in `SleepStatsAgreement` (`evaluation.py:2317`).

### What `BAplot.R` produces vs YASA

| Element | R (`BAplot.R`) | YASA (`plot_blandaltman`) |
|---------|---------------|--------------------------|
| Scatter points | One dot per session | ✅ One dot per session |
| Bias line | Horizontal `mean(diff)` or regression `b0 + b1*ref` | ✅ Same; auto-selected from `assumptions` |
| Bias CI | t-CI or bootstrap band | ✅ Shaded band for the mean bias. No band for a regression bias: combining the intercept and slope CIs is not a valid confidence region for the fitted line (R uses a pointwise `predict(interval="confidence")` band; could be added). |
| LoA lines | Constant `bias ± 1.96 SD`, parallel to a regression bias at `± 1.96 SD(resid)` (eq. 2), or `bias ± 2.46*(c0 + c1*ref)` | ✅ Same; auto-selected from `assumptions` |
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

## Assumption tests: sample-size dependence

All four assumption flags are null-hypothesis tests with no effect-size threshold, mirroring
the R pipeline (designed for n ≈ 10–40): `alpha` (default 0.05) for `unbiased`, `constant_bias`
and `homoscedastic`, and `alpha_normal` (default 0.01) for `normal`. Simulations
(500 replicates per cell) show the practical consequences:

- **Ideal null** (no bias, normal, constant, homoscedastic): each test passes ~95% of the time at
  every n, so ~19% of statistics fail at least one gate by chance (3–4 of YASA's 18 statistics).
- **Negligible real deviations** (bias 3 min, slope 0.02, mildly heavy tails, SD ≈ 20 min): the
  joint pass rate drops from 0.70 at n = 20 to 0.01 at n = 500. Minimum detectable bias for
  SD = 20 min: 9.4 min at n = 20, 1.2 min at n = 1000.
- **Reference measurement error**: regressing `obs − ref` on `ref` has a negative slope artifact
  equal to `−var(e_ref)/var(ref)` even with no true proportional bias (Bland & Altman 1995,
  Lancet). With equal error in both scorers the `constant_bias` flag fails 39% of the time at
  n = 100 and 97% at n = 500; regressing on the mean of the two scorers removes it.

Consequences: `normal` only switches CIs to bootstrap (benign); `unbiased` only colours the bias
line; `constant_bias` and `homoscedastic` change the reported bias/LoA model and are the ones
affected. The new `diagnostics` table exposes the effect sizes needed to judge materiality.

### Open items (not implemented)
| Item | Notes |
|---|---|
| Dual-criterion gates (significance **and** magnitude) | e.g. flag `constant_bias` only if `p < alpha` and the bias swing across the reference IQR exceeds ~0.5 SD of the differences (≈ R² > 0.1); `homoscedastic` only if the modelled SD ratio between the 90th and 10th reference percentiles exceeds ~1.5. Thresholds as constructor parameters. |
| `x_axis="mean"` option | Removes the reference-error artifact; changes the plot x-axis and calibration functions, so a design decision. |
| Log-first remedy order | Paper: log-transform when heteroscedastic, fall back to regression LoA only if heteroscedasticity persists. YASA's `log_transform` is a global switch. |
| Per-stat method override | Accept a user-supplied `auto_methods`-like table instead of hard-coding `bias_method`/`loa_method` globally. |
| `unbiased` is a result, not an assumption | Equivalent to the bias CI excluding zero; belongs in the report rather than the assumptions table. |

---

## Log transformation (Euser et al. 2008)

**Reference:** `groupDiscr.R` lines 208–264, `BAplot.R` log section, Euser et al. (2008) and Bland & Altman (1999).

### Design principles (simplified vs R)

- `log_transform` is **bool only** — no per-stat list. Mixed cases use two `SleepStatsAgreement` objects.
- Euser slope is **public**: `loa_log_slope` property, and a `loa_log_slope` variable (with CI) in `summary()` when `log_transform=True`. `plot_blandaltman` and `report` consume it via `summary()`.
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

where `log_diffs = log(obs) − log(ref)`. Statistics with a zero value in either scorer are not
log-transformed (any offset would dominate the slope); they are excluded with a warning and keep
the regular LoA, and `loa_method="log"` raises for them.

This is the **only** LoA representation that applies for log-transformed stats. The standard `loa_lower`/`loa_upper` (constant) and `loa_intercept`/`loa_slope` (regression) representations do not apply to these stats.

---

## Regression tests against the Menghini et al. (2021) pipeline

`tests/test_evaluation_sri.py` loads all 14 subjects (10 766 epochs, 30-s epochs) from
`tests/data/sample_data_sri.csv` and compares YASA's output to reference values extracted
from the published HTML report (`AnalyticalPipeline_v1.0.0.html`), stored in
`tests/data/evaluation_sri_full.json`. 37 test methods are run via `unittest`.

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
| `TestSRIProportionalConfusionMatrix` | Proportional error matrix: mean and SD of all 16 cells (`proportional_avg`), basic-bootstrap CIs vs the published CIs | 1 pp (mean/SD), 2 pp (basic CI) |
| `TestSRISanity` | Dataset size, output shapes, index names, stage labels, metrics in [0, 100] | — |

**Tolerance note:** 0.1 pp covers single rounding in the HTML source (±0.005 pp). 0.5 pp at group level covers accumulated rounding across 14 subjects.

**WASO note:** The R pipeline (`ebe2sleep.R` lines 47–50) counts wake epochs from the first sleep epoch to the **end of the recording**, including any post-sleep wake after the final sleep epoch. This is algebraically equivalent to `TIB − SOL − TST`. YASA's built-in `WASO` counts only wake within the Sleep Period Time (first to last sleep epoch), equivalent to `SPT − TST`. For subjects with post-sleep wake (sbj09, sbj11) the two definitions disagree. The tests use `TIB − SOL − TST`, which matches the R pipeline exactly for all 14 subjects.

### What cannot be tested (yet)

| Item | Reason |
|---|---|
| Per-subject accuracy (4-stage) | YASA accuracy = fraction correct across all stages; R pipeline reports binary one-vs-rest accuracy per stage. Numerically different. (Binary accuracy is tested.) |
| Per-subject PPV / NPV | R pipeline reports these at group level only; no per-subject reference. |
| Per-stage Cohen's κ, PABAK, prevalence index | R pipeline computes one-vs-rest; YASA `kappa` is multiclass. PABAK is not yet in YASA (item 2 above). |
| Bland-Altman bias, LoA, and CIs | R pipeline uses conditional regression depending on assumption tests, making expected outputs data-dependent and impractical to pin as fixed reference values. Eq. 2 (`loa_halfwidth`), MDC and `diagnostics` are unit-tested against direct `scipy.stats` computations in `tests/test_evaluation.py` instead. |
