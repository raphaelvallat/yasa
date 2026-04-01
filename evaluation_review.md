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
| 6 | **Log transformation** missing | ⚠️ Deferred | Planned: `log_transform` param + Euser et al. (2008) back-transform. Separate PR. |
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

---

## Future PR: Bland-Altman Plot (`BAplot.R` vs YASA)

YASA has **no Bland-Altman plot** for `SleepStatsAgreement`. This is a major missing piece.

### What `BAplot.R` produces

One scatter plot per sleep statistic (obs − ref differences vs reference), with:

| Element | Description |
|---------|-------------|
| Scatter points | One dot per session |
| Marginal density | Y-axis marginal distribution via `ggExtra::ggMarginal` |
| Bias line (red) | Horizontal constant `mean(diff)` — or regression line `b0 + b1*ref` if proportional bias |
| Bias CI (red dashed) | Classic t-CI or bootstrap CI band around bias line |
| Upper LoA (gray) | `bias + 1.96*SD` — or `bias + 2.46*(c0 + c1*ref)` if heteroscedastic |
| Lower LoA (gray) | `bias − 1.96*SD` — or `bias − 2.46*(c0 + c1*ref)` if heteroscedastic |
| LoA CI (gray dashed) | CI bands around each LoA |
| Log-transform mode | LoA drawn as `bias ± ref × euser_slope` (Euser 2008) |

### `BAplot.R` parameters

| Parameter | Purpose |
|-----------|---------|
| `xaxis = "reference"` or `"mean"` | X-axis: reference value or (obs+ref)/2 |
| `logTransf = TRUE/FALSE` | Use Euser back-transform for LoA |
| `CI.type = "classic"/"boot"` | Parametric t-CI or bootstrap CI |
| `xlim`, `ylim` | Axis limits |

### YASA current state

- `SleepStatsAgreement` has **no `plot_blandaltman()` method** (confirmed by grep)
- All statistical values needed for the plot are already stored: `_vals`, `_ci`, `_regr`, `_data`
- YASA already detects proportional bias and heteroscedasticity and chooses parm/regr methods accordingly — these would drive which line style to draw
- Log-transform Euser slopes are deferred (item 6, separate PR)

### What a `plot_blandaltman()` method would need to implement

1. Scatter of `difference` vs `ref` per sleep stat (one subplot per stat, or single stat via argument)
2. Bias line: constant if `constant_bias`, regression line `b0 + b1*x` otherwise
3. Bias CI: shaded band or dashed lines; use boot CI if `normal == False`
4. LoA lines: constant if `homoscedastic`, regression `bias ± 2.46*(c0 + c1*x)` otherwise
5. Euser LoA: `bias ± euser_slope * x` for log-transformed stats
6. LoA CI bands (dashed)
7. Optional: marginal distribution on y-axis (via `seaborn` or `matplotlib` inset)
8. Optional: `xaxis="reference"` vs `xaxis="mean"` toggle

---

## Future PR: Log transformation (Euser et al. 2008)

**Planned parameter:** `log_transform=False` (bool or list of str)

**Design notes:**
- `log_transform=True` applies to all stats; a list applies to named stats only (e.g., `["SOL", "WASO"]`)
- Log-ratio differences: `d_i = log(obs_i + 1e-4) − log(ref_i + 1e-4)`
- Euser slope: `2*(exp(agreement * SD(d_i)) − 1) / (exp(agreement * SD(d_i)) + 1)`
- Parametric CI: `slope_ci = euser_fn(SD ± t * sqrt(SD² * 3 / n))` (Bland & Altman 1999 SE formula)
- Store in `_vals["loa_log_slope"]` and `_ci["param"]["loa_log_slope"]`
- `get_table()` to show `"Bias ± slope × ref"` for log-transformed stats (fstring key `"loa_log"`)
- New property: `log_transform_stats`

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
