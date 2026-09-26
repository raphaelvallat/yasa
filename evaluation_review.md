# YASA `evaluation.py` vs the Menghini et al. (2021) pipeline

Reviewed against the paper (Menghini et al., 2021, *SLEEP* 44(2), zsaa170), its
`AnalyticalPipeline_v1.0.0.Rmd` report and companion R functions (`ebe2sleep.R`, `errorMatrix.R`,
`indEBE.R`, `groupEBE.R`, `indDiscr.R`, `groupDiscr.R`, `BAplot.R`).
Original module: https://github.com/raphaelvallat/yasa/pull/228.

Legend: ✅ implemented · ❌ not implemented · 🐛 fixed bug

## EpochByEpochAgreement

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | Specificity and NPV per stage | ✅ | One-vs-rest confusion matrix in `get_agreement_bystage()`. |
| 2 | 🐛 Undefined per-session metrics counted as 0 | ✅ | `zero_division` parameter, default `np.nan` (sklearn ≥ 1.3). Nights whose reference lacks a stage no longer drag down group means. |
| 3 | Pooled epochs (R `metricsType="sum"`) | ✅ | `get_agreement(pooled=True)`; default is the per-session average (`"avg"`). |
| 4 | Proportional error matrix (`errorMatrix.R`) | ✅ | `get_confusion_matrix_proportional()`: mean, SD and CI across sessions of row-normalized per-session matrices; absent stages kept as NaN; participant bootstrap. |
| 5 | Percent scale | ✅ | Proportion-based metrics are 0–100. `kappa` and `mcc` unchanged. |
| 6 | 🐛 Weighted `recall` equal to `accuracy` | ✅ | Removed from the default scorers. |
| 7 | 🐛 `get_agreement(scorers=[...])` and `sample_weight` crashed | ✅ | Names map to `sklearn.metrics.<name>_score`. `summary()` no longer requires a prior `get_agreement()` call. |
| 8 | PABAK, prevalence and bias index | ❌ | Not in sklearn. |
| 9 | ROC curves | ❌ | Could be added with sklearn. |
| 19 | CI on group-level EBE metrics | ✅ | `summary(ci_method="boot")`: participant bootstrap of the mean across sessions, for both `by_stage=False` and `True`. Off by default. `support` gets no CI. |

**Deviation from R.** The proportional error matrix CI defaults to the BCa bootstrap, which corrects
the skew of bounded proportions. R's "basic" bootstrap is available via
`bootstrap_kwargs={"method": "basic"}` and validated against the published CIs. Cells with 0/0 are
NaN and excluded from mean, SD and CI (R's sample data never hits this case).

**Bootstrap CIs are session-level.** Both `get_confusion_matrix_proportional()` and `summary()`
resample *sessions*, so the interval is for the mean across sessions and generalizes to a
population of nights. Epochs are never resampled: sleep stages occur in long bouts, so epochs
within a night are strongly autocorrelated and an i.i.d. epoch bootstrap would badly understate
the uncertainty (a block bootstrap would be required). Both methods share `_bootstrap_ci_mean()`,
which uses NaN-aware percentiles and falls back to plain percentiles for constant columns —
neither of which `scipy.stats.bootstrap` does, hence the hand-rolled implementation.

**BCa needs enough sessions.** The BCa bias and acceleration corrections are estimated from the
sessions themselves. At n = 5 the adjusted percentiles land in the extreme tail of a coarse
bootstrap distribution (only `comb(2n-1, n) = 126` distinct resample means), so the bounds are
driven by individual sessions and keep drifting as `n_resamples` grows (accuracy upper bound
39.1 → 41.1 → 41.3 for 1e3 → 1e4 → 1e5 resamples, versus a stable 38.5 for `"percentile"`). The
three methods agree to ~0.2 units by n = 30. `summary()` warns below 20 sessions; the proportional
error matrix does not, for backwards compatibility with the published CIs.

## SleepStatsAgreement

| # | Item | Status | Notes |
|---|------|--------|-------|
| 10 | Report table (`groupDiscr.R`) | ✅ | `report()`: `mean (SD)` per scorer, bias and LoA with CIs, assumption flags; `sleep_stats` subset; `ci_method=None` omits CIs and skips the bootstrap. |
| 11 | Log transformation (Euser et al. 2008) | ✅ | `log_transform=True` gives `LoA = bias ± slope × ref`, `slope = 2(e^z − 1)/(e^z + 1)`, `z = 1.96 SD(log obs − log ref)`. `"log"` is a `loa_method`; the slope and its CI are `loa_log_slope` in `summary()`. Normality of the raw differences still selects the CI method. |
| 12 | 🐛 Paper eq. 2 missing (proportional bias, homoscedastic differences) | ✅ | LoA were horizontal at `mean ± 1.96 SD(diff)`: mis-centred and too wide by `1/sqrt(1 − R²)`. Now `bias_i ± 1.96 SD(residuals)`, parallel to the bias line; half-width and CI (`SE(SD) ≈ SD/sqrt(2n)`) are `loa_halfwidth` in `summary()`. |
| 13 | Assumption results beyond booleans | ✅ | `assumptions` table (`assumption` × `metric`): statistic, `pvalue`, effect size, `passed`, and the `method` used for `"auto"`. See below. |
| 14 | 🐛 Missing values (e.g. `Lat_REM` without REM) gave NaN regressions | ✅ | Sessions with a missing value are dropped per statistic (warning); CIs use the per-statistic n. Inputs are not modified in place. |
| 15 | 🐛 Zeros under `log_transform=True` | ✅ | The former 1e-4 offset dominated the Euser slope. Statistics with a zero are now excluded from the transform (warning) and keep regular LoA; `loa_method="log"` raises for them. Mixed cases need two objects. |
| 16 | 🐛 Calibration direction and `bias_method="auto"` reordering or dropping columns | ✅ | `calibrate()` subtracts the mean bias or inverts the bias regression `(x − b0)/(1 + b1)`, preserving columns and NaNs. R has no calibration. |
| 17 | API | ✅ | Constructor takes the `get_sleep_stats()` output directly (scorers read from the index); the LoA multiplier is fixed at 1.96 as in R. |
| 18 | Individual discrepancy heatmap (`indDiscr.R`) | ❌ | Data available via `get_sleep_stats()`. |

### Bland-Altman plot (`BAplot.R` vs `plot_blandaltman()`)

| Element | R | YASA |
|---------|---|------|
| Bias line | `mean(diff)` or `b0 + b1 × ref` | ✅ Auto-selected from `assumptions`; solid line. |
| Bias CI band | Yes | ✅ Mean bias only. For a regression bias, combining the intercept and slope CIs is not a valid region; R's pointwise `predict(interval="confidence")` band could be added. |
| LoA | Constant, eq. 2, regression `bias ± 2.46 (c0 + c1 × ref)`, or Euser | ✅ Auto-selected; dashed lines with CI bands. |
| Red bias line if significant | Yes | ❌ The bias CI band already shows whether zero is excluded. |
| x-axis | `"reference"` (default) or `"mean"` | Reference only, matching the paper's recommendation. Regressing `obs − ref` on `ref` is biased toward a negative slope when the reference has error (Bland & Altman 1995), so a `"mean"` option may be worth adding. |
| Marginal densities | Yes | ❌ |
| Layout | One plot per call | Grid of at most 4 columns. |

## Assumption tests

Four null-hypothesis tests run per statistic at `alpha` (default 0.05), as in R. Three of them
select the method applied when `bias_method`, `loa_method` or `ci_method` is `"auto"`:

| Flag | Test | Fail → | Effect size in `assumptions` |
|---|---|---|---|
| `unbiased` | t-test of differences vs 0 | Nothing: a finding, reported but not used | `cohen_d` |
| `normal` | Shapiro-Wilk on differences | Bootstrap CIs instead of parametric | `skew`, `kurtosis` |
| `constant_bias` | Slope of diff ~ ref | Regression bias `b0 + b1·ref`; constant LoA follow it (eq. 2) | `r2` |
| `homoscedastic` | Slope of \|resid\| ~ ref | Regression LoA `bias ± 2.46 (c0 + c1·ref)`; overridden by `log_transform=True` | `r2` |

**Sample-size dependence.** A p-value mixes effect size and n. R was designed for n ≈ 10–40; with
hundreds of nights, trivial deviations fail every test (a slope explaining 2% of the variance fails
`constant_bias`; one heavy-tailed point fails `normal`), while in small samples real violations
pass. Under a perfect null about 19% of statistics fail at least one of the four gates by chance.
Regressing on the reference also biases `constant_bias` toward failure whenever the reference has
its own error. A stricter alpha for Shapiro-Wilk was tried and reverted: it only shifts the
dependence on n.

**Toward more robust gates.** The paper asks for the tests to be "accompanied by visual
inspection"; the effect sizes in `assumptions` are the numerical form of that inspection and could
gate the `auto` methods together with the p-value (fail only if `p < alpha` **and** the effect is
material):

| Flag | Magnitude criterion | Rationale |
|---|---|---|
| `normal` | \|`skew`\| > 1 or excess `kurtosis` > 2 | Below this the t-based CI is accurate for n ≥ 30 (CLT). The cost of a false rejection is only a bootstrap CI, so this gate matters least. |
| `constant_bias` | `r2` > 0.1 | The reference explains at least 10% of the variance of the differences. Below this the fitted line is nearly flat and a constant bias is simpler and more stable. |
| `homoscedastic` | SD ratio > 1.5 | Fitted \|resid\| at the top vs bottom of the observed reference range, `(c0 + c1·max)/(c0 + c1·min)`. Below this the LoA width changes by less than 50% across the range and constant LoA are adequate. Needs the intercept of the \|resid\| ~ ref regression, which `assumptions` does not yet expose. |

Thresholds would be constructor parameters with these defaults. Other open items:

- Per-statistic overrides: accept a `method` table (same shape as the `method` columns of
  `assumptions`) instead of global `bias_method` / `loa_method`.
- Remedy order: the paper log-transforms first and uses regression LoA only if heteroscedasticity
  persists; YASA's `log_transform` is a global switch.

## Regression tests against the published pipeline

`tests/test_evaluation_sri.py` runs the 14 subjects (10 766 epochs) of
`tests/data/sample_data_sri.csv.xz` against the published HTML report values stored in
`tests/data/evaluation_sri_full.json`.

| Test class | Checked | Tolerance |
|---|---|:-:|
| `TestSRIPerSubject` | Recall and specificity, 14 subjects × 4 stages | 0.1 pp |
| `TestSRIPerSubjectSleepWake` | Binary sleep/wake accuracy, sensitivity, specificity | 0.1 pp |
| `TestSRIGroupMeans` | Group mean PPV and NPV per stage | 0.5 pp |
| `TestSRISleepStats` | TIB, TST, SE, SOL, stage durations and % for both scorers; WASO as `TIB − SOL − TST` | 0.1 |
| `TestSRIDiscrepancies` | Device − reference differences | 0.1 |
| `TestSRIPooledMetrics` | Pooled recall and specificity | 0.1 pp |
| `TestSRIConfusionMatrixValues` | 16 cells of the pooled confusion matrix | exact |
| `TestSRIProportionalConfusionMatrix` | Mean and SD of all cells; basic-bootstrap CIs | 1 pp / 2 pp |

Tolerances cover rounding in the HTML source. **WASO:** R counts wake from sleep onset to the end
of the recording (`TIB − SOL − TST`); YASA counts wake within the sleep period (`SPT − TST`). The
tests use the R definition.

Not testable against the report: per-subject 4-stage accuracy, per-subject PPV/NPV, per-stage
kappa/PABAK, and the Bland-Altman outputs (conditional on data-dependent assumption tests).
Eq. 2, `assumptions`, `summary()`, `calibrate()` and the Euser slope are unit-tested against direct
`scipy.stats` computations in `tests/test_evaluation.py`.
