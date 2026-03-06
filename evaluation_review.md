# YASA `evaluation.py` — Review Against Menghini et al. 2021 Pipeline

Reviewed against: `sleep-trackers-performance-master/AnalyticalPipeline_v1.0.0.Rmd` and its
companion R functions (`ebe2sleep.R`, `errorMatrix.R`, `indEBE.R`, `groupEBE.R`, `indDiscr.R`,
`groupDiscr.R`, `BAplot.R`).

## Status legend
- ✅ Implemented / Fixed
- ⚠️  Partial / In progress
- ❌ Missing / Not implemented
- 🐛 Bug

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
| 8 | **Bug in standalone `groupDiscr.R:117`** (reference code, not YASA) | — | `logTransf=FALSE` should be `TRUE`. Bug in the R file, not YASA |
| 9 | **Calibration direction bug** | ✅ Fixed | See detailed investigation below (Bugs A, B, C) |

---

## Bug Investigations

### Bug A — Residuals computation (discovered during investigation of #9)

**File:** `evaluation.py`, line ~1023
**Severity:** High — affects heteroscedasticity test and LoA regression

**Old code:**
```python
data["residuals"] = data[obs_scorer].to_numpy() - predicted_values
```

**Problem:** `predicted_values = b1 * ref + b0` is the *predicted difference* (from regression
`difference ~ ref`). Subtracting it from `obs` gives `obs − predicted_difference`, which is NOT
the regression residual. The true residual is `difference − predicted_difference`. The spurious
`+ref` term in the wrong formula causes `|residuals| ~ ref` to be artificially highly correlated,
making the heteroscedasticity test almost always significant.

**Fix:**
```python
data["residuals"] = data["difference"].to_numpy() - predicted_values
```

**Status:** ✅ Fixed (line ~1070)

---

### Bug B — Calibration `parm` formula (item 9)

**File:** `evaluation.py`, lines ~1471 and ~1534
**Severity:** High — calibrated values go in the wrong direction

**Analysis:**
- `bias_parm = mean(obs − ref)` — positive means obs overestimates reference
- To calibrate new obs to reference scale: `calibrated = obs − bias_parm`
- YASA used `obs + bias_parm`, which amplifies the bias instead of correcting it

**Evidence from doctest:** `calibrate_rem([50,40,30,20], bias_test=False)` returned
`[42.825, 32.825, ...]`. With `bias_parm = −7.175` (device underestimates REM by 7.175 min),
YASA computed `50 + (−7.175) = 42.825` (further below ref), when it should be
`50 − (−7.175) = 57.175` (above, correcting upward to match ref).

**Fix:**
```python
# OLD: parm_adjusted = data + self._vals["bias_parm"]
parm_adjusted = data - self._vals["bias_parm"]
# OLD: return x + parm
return x - parm
```

**Status:** ✅ Fixed (lines ~1568, ~1631)

---

### Bug C — Calibration `regr` formula (item 9)

**File:** `evaluation.py`, lines ~1472 and ~1537
**Severity:** High — produces nonsensical values (negative sleep times)

**Analysis:**
- Bias regression model: `(obs − ref) = b0 + b1 * ref`
  → `obs = ref * (1 + b1) + b0`
  → inverse: `ref_estimated = (obs − b0) / (1 + b1)`
- YASA used `obs * b1 + b0`, which is neither the forward model nor its inverse.

**Evidence from doctest:** `calibrate_rem([50,40,30,20], method="regr")` returned
`[−9.34, −9.87, −10.40, −10.93]` — negative REM minutes, clearly wrong.

**Fix:**
```python
# OLD: regr_adjusted = data * self._vals["bias_slope"] + self._vals["bias_intercept"]
regr_adjusted = (data - self._vals["bias_intercept"]) / (1 + self._vals["bias_slope"])
# OLD: return x * slope + intercept
return (x - intercept) / (1 + slope)
```

**Status:** ✅ Fixed (lines ~1569, ~1633)

---

### Item 6 — Log transformation (Euser et al. 2008) — deferred to separate PR

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

## Features where YASA goes beyond the R pipeline

| Feature | YASA | R pipeline |
|---------|------|------------|
| Calibration of new data | ✅ `calibrate()`, `get_calibration_func()` | ❌ Not present |
| MCC (Matthews Correlation Coefficient) | ✅ | ❌ |
| Balanced accuracy | ✅ | ❌ |
| BCa bootstrap (more robust) | ✅ | ❌ (R uses "basic" default) |
| Overlaid hypnogram plots | ✅ `plot_hypnograms()` | ❌ |
| MAD / median in group summary | ✅ `summary()` | ❌ |

---

## Bland-Altman Plot (`BAplot.R` vs YASA)

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
- Log-transform Euser slopes are now stored in `_vals["loa_log_slope"]` (after item 6 fix)

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

## Naming Convention Changes (implemented)

| Location | Old | New | Rationale |
|----------|-----|-----|-----------|
| Method option strings | `"parm"` | `"param"` | Standard abbreviation of "parametric" |
| Column in `_vals` / `summary()` | `bias_parm` | `bias_mean` | It IS the mean difference — more descriptive |
| Column in `_vals` / `summary()` | `lloa_parm` | `loa_lower` | Drop redundant method suffix; symmetric naming |
| Column in `_vals` / `summary()` | `uloa_parm` | `loa_upper` | Symmetric with `loa_lower` |
| CI MultiIndex key | `"parm"` | `"param"` | Consistent with option strings |
| fstring key in `get_table()` | `"bias_parm"` | `"bias_mean"` | Matches column rename |
| fstring key in `get_table()` | `"loa_parm"` | `"loa_const"` | "const" = constant/homoscedastic LoA (±1.96 SD) |
| fstring key in `get_table()` | `"bias_parm_ci"` | `"bias_mean_ci"` | Matches |
| fstring key in `get_table()` | `"loa_parm_ci"` | `"loa_const_ci"` | Matches |
| Local vars | `parm_vals`, `parm_ci`, `t_parm`, `parm_adjusted`, `parm_idx` | `param_vals`, `param_ci`, `t_param`, `param_adjusted`, `param_idx` | Follow from option rename |
| `_generate_bootstrap_ci` local vars | `bias_parm`, `lloa_parm`, `uloa_parm` | `bias_mean`, `loa_lower`, `loa_upper` | Match column renames |
| `_generate_bootstrap_ci` variable_order | `"bias_parm"`, `"lloa_parm"`, `"uloa_parm"` | `"bias_mean"`, `"loa_lower"`, `"loa_upper"` | Must match `_vals` column names |
| `get_calibration_func()` local var | `parm` | `bias_mean` | Descriptive and matches column |

Unchanged: `"regr"`, `bias_slope`, `bias_intercept`, `loa_slope`, `loa_intercept`, `loa_log_slope`.

---

## R-only features still missing in YASA

- **Bland-Altman plot** (`BAplot.R`) — no `plot_blandaltman()` on `SleepStatsAgreement`
- PABAK (Prevalence-Adjusted Bias-Adjusted Kappa)
- Prevalence index and bias index (from `epiR::epi.kappa`)
- ROC curves for EBE analysis
- Per-subject discrepancy heatmap (`indDiscr.R` style visualization)

