# Hypnogram Class Roadmap

Goal: make `yasa.Hypnogram` the industry-standard Python object for handling sleep hypnograms.

---

## Implemented (v0.7)

### Core class
- `Hypnogram(values, n_stages, freq, start, tz, scorer, proba)` — string-based, categorical storage
- `from_integers(values, mapping, ...)` — migrate from legacy integer arrays
- `from_profusion(fname, ...)` — load Compumedics Profusion XML (NSRR format)
- `start` / `end` properties — absolute timestamp anchoring with timezone support
- `upsample_to_data(data, sf)` — align to EEG data by sample count or absolute timestamp timestamps (MNE `meas_date`)

### Analysis
- `sleep_statistics()` — full AASM summary (TIB, TST, SE, SOL, WASO, stage durations, ...)
- `transition_matrix()` — stage transition counts and probabilities
- `find_periods(threshold)` — detect consecutive runs of a stage
- `consolidate_stages(new_n_stages)` — merge to 2/3/4-stage hypnogram
- `evaluate(obs_hyp)` — epoch-by-epoch agreement metrics (kappa, MCC, F1, ...)

### Python container protocol
- `__len__` — `len(hyp)` returns number of epochs
- `__eq__` — `hyp1 == hyp2` returns a boolean NumPy array (element-wise)
- `__getitem__` — `hyp[0]`, `hyp[-1]`, `hyp[10:50]` return a new `Hypnogram`
- `crop(start, end)` — slice by epoch index (inclusive) or absolute timestamp time

### Visualization & export
- `plot_hypnogram()` — standard hypnogram figure
- `as_int()` — integer-encoded `pandas.Series`
- `as_events()` — BIDS-compatible events `DataFrame` (onset, duration, stage)
- `upsample(new_freq)` — change epoch resolution
- `to_json(fname)` / `from_json(fname)` — save and load to disk, preserving all metadata
- `to_dict()` / `from_dict(d)` — JSON-serializable in-memory representation (same format as `to_json`)

---

## Planned (future PRs)

### I/O
- **`to_csv()` / `from_csv()`** — round-trip to disk preserving metadata (freq, start, scorer, proba).
- **`from_edf_annotations(raw)`** — load hypnogram from EDF+ annotations.


### Analysis
- **`plot_hypnodensity()`** — when `proba` is available, plot the per-epoch stage probability as a color-map (signature visualization of modern auto-staging papers).

### Convenience
- **`get_mask(*stages)`** — return a boolean NumPy array for one or more stages (e.g., `hyp.get_mask("N2", "N3")`). Reduces boilerplate when passing stage masks to detection functions.
