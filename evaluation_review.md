# YASA `evaluation.py` — Review Against the Menghini et al. (2021) Pipeline

Reviewed against the published paper (Menghini et al., 2021, *SLEEP* 44(2), zsaa170), the
`AnalyticalPipeline_v1.0.0.Rmd` report, and its companion R functions (`ebe2sleep.R`,
`errorMatrix.R`, `indEBE.R`, `groupEBE.R`, `indDiscr.R`, `groupDiscr.R`, `BAplot.R`).
Original module: https://github.com/raphaelvallat/yasa/pull/228.

Status legend: ✅ implemented / fixed · ❌ missing · 🐛 was a bug

---

## EpochByEpochAgreement

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | Specificity (TNR) and NPV in `get_agreement_bystage()` | ✅ | Per-stage one-vs-rest confusion matrix. |
| 2 | 🐛 Undefined metrics counted as 0 (`zero_division=0` hard-coded) | ✅ | `zero_division` parameter, default `np.nan` (sklearn ≥ 1.3). Nights whose reference lacks a stage no longer drag down group means; `support == 0` flags them. `0`, `1`, `"warn"` accepted; also applied to specificity/NPV. |
| 3 | Pooled epochs (R `metricsType="sum"`) | ✅ | `get_agreement(pooled=True)`. Default (`False`) is the per-session average (R `metricsType="avg"`). |
| 4 | Proportional error matrix (`errorMatrix.R`, `prop`: mean (SD) [95% CI] of row-normalized per-subject matrices) | ✅ | `get_confusion_matrix_proportional()`. Subject-then-group averaging; absent stages kept as NaN (not zero-filled like sklearn's `normalize="true"`); `n_sessions` column; participant bootstrap CI (same resampled sessions for all cells); `formatted=True` gives the `"mean (SD) [lo, hi]"` table. |
| 5 | Metrics on 0–1 scale vs R's 0–100 % | ✅ | All proportion-based metrics are percentages. `kappa` and `mcc` stay on their −1 to 1 scale. |
| 6 | 🐛 Weighted `recall` in `get_agreement()` identical to `accuracy` | ✅ | Removed from the default scorers. Per-stage (one-vs-rest) recall remains in `get_agreement_bystage()`. |
| 7 | 🐛 `get_agreement(scorers=[...])` and `get_agreement(sample_weight=...)` crashed | ✅ | Names map to `sklearn.metrics.<name>_score`; index compared with `.equals()`. `summary()` no longer requires a prior `get_agreement()` call. |
| 8 | PABAK, prevalence index, bias index (`epi.kappa()`) | ❌ | Not trivially available in sklearn. |
| 9 | ROC curves (`groupEBE(doROC=TRUE)`) | ❌ | Could be added with sklearn. |

**Justified deviations from R.** The proportional error matrix CI defaults to the BCa bootstrap
(Efron 1987), which corrects the skew and bias of bounded proportions; R uses the "basic"
(reverse-percentile) bootstrap, available via `bootstrap_kwargs={"method": "basic"}` and validated
against the published CIs. R's sample data has no subject with a missing stage, so its behavior
there is untested; YASA keeps 0/0 cells as NaN, excludes them from mean/SD/CI, drops bootstrap
replicates in which a stage is absent from every resampled subject, and falls back to plain
percentiles for degenerate (constant) cells where BCa is undefined.

---

## SleepStatsAgreement

| # | Item | Status | Notes |
|---|------|--------|-------|
| 10 | Report table (`groupDiscr.R`: scorer means (SD), bias [CI], LoA) | ✅ | `report()` with `mean (SD)` per scorer, `sleep_stats` subset/order, `bias_ci` / `loa_ci` flags (no bootstrap when both are off). |
| 11 | Log transformation (Euser et al. 2008) | ✅ | `log_transform=True`; `"log"` is a first-class `loa_method`; public `loa_log_slope` property and `summary()` variable with CI. See design notes below. |
| 12 | 🐛 Paper eq. 2 not implemented (proportional bias + homoscedastic differences) | ✅ | LoA were horizontal at `mean ± 1.96 SD(differences)`: mis-centred and too wide by `1/sqrt(1 − R²)`. Now `bias_i ± 1.96 SD(residuals)` parallel to the regression bias line, exposed as `loa_halfwidth` in `summary()` (ddof=1 as R's `sd(resid(lm))`; parametric CI from `SE(SD) ≈ SD/sqrt(2n)`, Bland & Altman 1999; bootstrap CI). |
| 13 | Assumption tests are booleans only; paper asks for results "accompanied by visual inspection" | ✅ | `diagnostics` property (MultiIndex `assumption` × `metric`): `unbiased` (t, pvalue, cohen_d), `normal` (W, pvalue, skew, kurtosis), `constant_bias` and `homoscedastic` (slope, pvalue, r2). `assumptions` is derived from it. The `Assumptions` column of `report()` shows only the three modeling assumptions (`unbiased` is a finding). |
| 14 | Shapiro-Wilk over-rejects at alpha 0.05 (e.g. n = 57, skew 0.07, p = 0.008 from one heavy-tail point) | ✅ | `alpha_normal` (default 0.01) for the `normal` flag only; `alpha` (0.05) for the other tests. Justified because a normality violation only switches CIs to bootstrap. R uses 0.05 throughout. |
| 15 | 🐛 Missing values (e.g. `Lat_REM` on a night without REM) produced NaN regressions and `nan + nanx` in the report | ✅ | Sessions with a missing value are dropped per statistic (with a warning); all t critical values and SEs use the per-statistic n. Inputs are no longer modified in place. |
| 16 | 🐛 `log_transform=True` with zeros: the 1e-4 offset dominated the Euser slope | ✅ | Statistics with a zero in either scorer are excluded from the log transform (warning) and keep regular LoA; `loa_method="log"` raises for them. |
| 17 | 🐛 Calibration: direction of the correction, and `bias_method="auto"` reordering columns / dropping NaN columns | ✅ | `calibrate()` subtracts the mean bias or inverts the bias regression `(x − b0) / (1 + b1)`; auto mode uses `DataFrame.where`, preserving columns and NaNs. R has no calibration. |
| 18 | `summary()` always bootstrapped all statistics | ✅ | `summary(ci_method=None)` returns point estimates; `sleep_stats` restricts (and orders) rows and limits bootstrapping. |
| 19 | Individual discrepancy heatmap (`indDiscr.R`) | ❌ | Data available via `get_sleep_stats()`. |

### Log transformation design

`LoA = bias ± slope × ref` with `slope = 2 (e^z − 1) / (e^z + 1)`, `z = agreement × SD(log(obs) − log(ref))`.
This is the only LoA representation for log-transformed statistics.

- `log_transform` is a bool applied to all statistics, except those with a zero value in either
  scorer (item 16). Mixed cases use two `SleepStatsAgreement` objects.
- `auto_methods["loa"]` is `"log"` for log-transformed statistics; `loa_method="param"` / `"regr"`
  still override it.
- Normality of the raw differences drives CI selection as before (no `log_normal` flag).
- The bootstrap CI of the Euser slope is computed inside `_generate_bootstrap_ci`.

### Bland-Altman plot (`BAplot.R` vs `plot_blandaltman()`)

| Element | R (`BAplot.R`) | YASA |
|---------|----------------|------|
| Scatter | One dot per session | ✅ |
| Bias line | `mean(diff)` or `b0 + b1 × ref` | ✅ Auto-selected from `assumptions`; solid line. |
| Bias CI | Band | ✅ For the mean bias. None for a regression bias: combining the intercept and slope CIs is not a valid confidence region (R uses a pointwise `predict(interval="confidence")` band; could be added). |
| LoA lines | Constant, parallel to a regression bias (eq. 2), regression `bias ± 2.46 (c0 + c1 × ref)`, or Euser | ✅ Auto-selected; dashed lines with shaded CI bands. |
| Flag biased | Red bias line if significant | ✅ `flag_biased=True` |
| x-axis | `xaxis="reference"` (default) or `"mean"` | Reference only. The paper recommends the reference (PSG) as the size of measurement, so this matches the default. Regressing `obs − ref` on `ref` is biased toward a negative slope when the reference has its own error (Bland & Altman 1995), so a `"mean"` option may be worth adding; it would also change the calibration functions. |
| Marginal density (`ggMarginal`) | Yes | ❌ |
| `xlim`, `ylim` | Parameters | Set on the returned axes. |
| Layout | One plot per call | Grid of at most 4 columns, balanced rows. |

### Assumption tests

All flags are null-hypothesis tests at `alpha` / `alpha_normal` with no effect-size threshold,
mirroring R (designed for n ≈ 10–40). Power scales with n: trivial deviations fail in large
samples and real ones pass in small ones, and under a perfect null about 19% of statistics fail
at least one of the four gates by chance. `normal` and `unbiased` only affect the CI method and
the bias-line color; `constant_bias` and `homoscedastic` change the reported bias/LoA model.
`diagnostics` exposes the effect sizes needed to judge whether a violation matters.

Open items:

| Item | Notes |
|---|---|
| Dual-criterion gates (significance and magnitude) | e.g. `constant_bias` only if `p < alpha` and R² > ~0.1; `homoscedastic` only if the modelled SD ratio across the reference range exceeds ~1.5. Thresholds as constructor parameters. |
| Per-statistic method override | Accept an `auto_methods`-like table instead of global `bias_method` / `loa_method`. |
| Log-first remedy order | Paper: log-transform when heteroscedastic, regression LoA only if heteroscedasticity persists. YASA's `log_transform` is a global switch. |
| `unbiased` is a result, not an assumption | Equivalent to the bias CI excluding zero. No longer shown in the `Assumptions` column of `report()`; still in `assumptions` / `diagnostics` because `flag_biased` and `calibrate()` use it. |

---

## Features beyond the R pipeline

Calibration of new data (`calibrate()`, `get_calibration_func()`), MCC, balanced accuracy, BCa
bootstrap, overlaid hypnogram plots (`plot_hypnograms()`), MAD/median in group summaries, the
human-readable `report()` table, and the `diagnostics` table. The minimal detectable change
(half the LoA width) was added and then removed; it is directly available as
`loa_halfwidth` or `(loa_upper − loa_lower) / 2` in `summary()`.

---

## Regression tests against the published pipeline

`tests/test_evaluation_sri.py` loads the 14 subjects (10 766 30-s epochs) of
`tests/data/sample_data_sri.csv.xz` and compares YASA to the values of the published HTML report
(`AnalyticalPipeline_v1.0.0.html`) stored in `tests/data/evaluation_sri_full.json`.

| Test class | Checked | Tolerance |
|---|---|:-:|
| `TestSRIPerSubject` | Recall and specificity, 14 subjects × 4 stages | 0.1 pp |
| `TestSRIPerSubjectSleepWake` | Binary SLEEP/WAKE accuracy, sensitivity, specificity per subject | 0.1 pp |
| `TestSRIGroupMeans` | Group mean PPV and NPV per stage (recall/specificity means are implied by the per-subject checks) | 0.5 pp |
| `TestSRISleepStats` | TIB, TST, SE, SOL, stage durations and % for both scorers; WASO as `TIB − SOL − TST` | 0.1 |
| `TestSRIDiscrepancies` | Device − Reference differences via `SleepStatsAgreement.data` | 0.1 |
| `TestSRIPooledMetrics` | Pooled recall and specificity (R `metricsType="sum"`) | 0.1 pp |
| `TestSRIConfusionMatrixValues` | All 16 cells of the pooled confusion matrix | exact |
| `TestSRIProportionalConfusionMatrix` | Mean and SD of all 16 cells; basic-bootstrap CIs vs published | 1 pp / 2 pp |
| `TestSRISanity` | Dataset size, shapes, stage labels | — |

0.1 pp covers rounding in the HTML source (±0.005); 0.5 pp covers accumulated rounding across
14 subjects. **WASO:** the R pipeline (`ebe2sleep.R`) counts wake from the first sleep epoch to
the end of the recording (`TIB − SOL − TST`); YASA's `WASO` counts wake within the sleep period
(`SPT − TST`). They differ for subjects with post-sleep wake (sbj09, sbj11); the tests use the R
definition.

Not testable against the report: per-subject 4-stage accuracy (R reports one-vs-rest per stage),
per-subject PPV/NPV (group level only in R), per-stage kappa/PABAK (YASA's kappa is multiclass),
and Bland-Altman bias/LoA/CIs (conditional on data-dependent assumption tests). Eq. 2,
`diagnostics`, `summary`, `calibrate`, and the Euser slope are unit-tested against direct
`numpy`/`scipy.stats` computations in `tests/test_evaluation.py` instead.
