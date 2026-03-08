# Multi-Scorer Hypnogram Support — Design Plan

## Motivation

Real studies frequently produce multiple scorings of the same night: one or more human raters plus one or more automated scorers. These scorings may not share identical start/end times. YASA should provide first-class support for:

1. Storing multiple `Hypnogram` objects together as a unit.
2. Aligning them to a common epoch grid despite different time windows.
3. Computing pairwise and all-vs-reference agreement metrics.
4. Generating a consensus hypnogram via majority vote, with a reference scorer as tiebreaker.

---

## Recommended architecture: new `HypnogramSet` class

Introduce `yasa.HypnogramSet` in a new file `src/yasa/hypnoset.py`. This is a thin container over a `dict[str, Hypnogram]` with domain-specific methods. It does **not** modify `Hypnogram` itself (beyond minor additions noted below).

**Rationale for a separate class rather than extending `Hypnogram`:**
- `Hypnogram` represents one scorer's view of one night — a well-defined, single-responsibility object.
- Multi-scorer operations (alignment, consensus, pairwise comparison) are naturally container-level concepts.
- Keeps the existing `Hypnogram` API stable.
- `EpochByEpochAgreement` can be reused internally for comparison without code duplication.

---

## `HypnogramSet` — proposed API

```python
class HypnogramSet:
    """A collection of Hypnograms for the same sleep session from multiple scorers."""

    def __init__(self, hypnograms, ref=None):
        """
        Parameters
        ----------
        hypnograms : list[Hypnogram] or dict[str, Hypnogram]
            Each Hypnogram must have a unique `scorer` attribute set.
            If a dict is passed, keys override `scorer` attributes.
        ref : str, optional
            Name of the reference scorer. Used as tiebreaker in consensus()
            and as the "ground truth" column in compare(). Defaults to the
            first scorer in insertion order.
        """

    # ---- Container protocol ------------------------------------------------

    def __len__(self):
        """Number of scorers."""

    def __getitem__(self, scorer: str) -> Hypnogram:
        """Access a scorer's Hypnogram by name."""

    def __contains__(self, scorer: str) -> bool:
        """Check whether a scorer is present."""

    def __repr__(self): ...

    # ---- Properties --------------------------------------------------------

    @property
    def scorers(self) -> list[str]:
        """Ordered list of scorer names."""

    @property
    def ref(self) -> str | None:
        """Reference scorer name."""

    @property
    def n_scorers(self) -> int:
        """Number of scorers."""

    @property
    def common_window(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """Intersection of all scorer [start, end) intervals.
        Returns None if any hypnogram lacks a start datetime."""

    # ---- Alignment ---------------------------------------------------------

    def align(
        self,
        start=None,
        end=None,
        freq: str | None = None,
        fill: str = "UNS",
    ) -> "HypnogramSet":
        """Align all hypnograms to a common epoch grid.

        Steps:
        1. Determine target [start, end): defaults to common_window (intersection).
           Raises ValueError if scorers have no temporal overlap.
        2. Crop each hypnogram to [start, end) using Hypnogram.crop().
        3. Resample to `freq` using Hypnogram.upsample() if needed (defaults
           to the finest freq present in the set).
        4. Pad epochs missing at the edges with `fill` (default "UNS").

        Returns a new HypnogramSet whose members share an identical DatetimeIndex.

        Requires all hypnograms to have `start` set; raises ValueError otherwise.
        """

    # ---- Comparison --------------------------------------------------------

    def compare(self, ref: str | None = None) -> "EpochByEpochAgreement":
        """Compute pairwise agreement between all scorers, or each scorer vs ref.

        Internally delegates to the existing EpochByEpochAgreement class.
        Automatically aligns hypnograms before comparison.

        Parameters
        ----------
        ref : str, optional
            If provided, compare every other scorer against this reference.
            If None, compute all N*(N-1)/2 pairwise comparisons.

        Returns
        -------
        EpochByEpochAgreement
        """

    # ---- Consensus ---------------------------------------------------------

    def consensus(
        self,
        ref: str | None = None,
        method: str = "majority",
        scorer: str = "consensus",
        fill: str = "UNS",
    ) -> Hypnogram:
        """Generate a single consensus Hypnogram from all scorers.

        Parameters
        ----------
        ref : str, optional
            Reference scorer used to break ties. Defaults to self.ref.
            If a tie cannot be broken (i.e. ref is also tied among leaders),
            the epoch is assigned `fill`.
        method : {"majority", "unanimous"}
            "majority"  : each epoch is assigned the plurality stage.
            "unanimous" : only epochs where all scorers agree are kept;
                          disagreements are assigned `fill`.
        scorer : str
            Name for the returned consensus Hypnogram. Default "consensus".
        fill : str
            Stage to assign to unresolvable epochs. Default "UNS".

        Returns
        -------
        Hypnogram

        Algorithm (majority)
        --------------------
        1. align() → stack into DataFrame, rows=epochs, cols=scorers
        2. For each row:
            a. Count votes per stage.
            b. If one stage has strictly more votes → winner.
            c. Else if ref scorer's vote is among the tied leaders → use ref.
            d. Else → fill.
        3. Return Hypnogram(consensus_stages, scorer=scorer, ...)
        """

    # ---- Visualization -----------------------------------------------------

    def plot(self, **kwargs):
        """Stack plot of all scorer hypnograms (one row per scorer)."""

    # ---- I/O ---------------------------------------------------------------

    def to_dict(self) -> dict:
        """JSON-serializable dict. Stores each Hypnogram via to_dict()
        plus top-level keys 'ref' and 'scorers' (ordered list)."""

    @classmethod
    def from_dict(cls, d: dict) -> "HypnogramSet": ...

    def to_json(self, fname): ...

    @classmethod
    def from_json(cls, fname) -> "HypnogramSet": ...
```

---

## Minor additions to `Hypnogram` (non-breaking)

- **`scorer` must be set** when adding to a `HypnogramSet`. A `ValueError` is raised at `HypnogramSet.__init__` time if any member has `scorer=None` and no key was supplied via dict.
- Optionally add **`Hypnogram.pad(start, end, fill="UNS")`** as a helper for `HypnogramSet.align()`. Pads epochs before/after the hypnogram to extend its window using `fill` stage values, returning a new `Hypnogram`.

---

## Integration with `EpochByEpochAgreement`

`HypnogramSet.compare()` delegates directly to the existing class:

```python
# Example internal implementation sketch
def compare(self, ref=None):
    aligned = self.align()
    if ref is None:
        # all pairwise: (Rater1, Rater2), (Rater1, YASA), (Rater2, YASA), ...
        from itertools import combinations
        pairs = list(combinations(aligned.scorers, 2))
        refs = [aligned[a] for a, _ in pairs]
        obs  = [aligned[b] for _, b in pairs]
    else:
        refs = [aligned[ref]] * (aligned.n_scorers - 1)
        obs  = [aligned[s] for s in aligned.scorers if s != ref]
    return EpochByEpochAgreement(refs, obs)
```

---

## Usage examples (target API)

```python
import yasa

# Three hypnograms from the same night, potentially different windows
hyp_r1   = yasa.Hypnogram(..., start="2024-01-01 22:00", scorer="Rater1")
hyp_r2   = yasa.Hypnogram(..., start="2024-01-01 22:05", scorer="Rater2")  # starts 5 min later
hyp_auto = yasa.SleepStaging(raw, eeg_name="C4-M1").predict(...)  # scorer="YASA"

# Build a HypnogramSet — ref scorer is used for tiebreaking
hs = yasa.HypnogramSet([hyp_r1, hyp_r2, hyp_auto], ref="Rater1")

# Inspect
hs.scorers          # ["Rater1", "Rater2", "YASA"]
hs.n_scorers        # 3
hs.common_window    # (Timestamp("2024-01-01 22:05"), Timestamp("2024-01-02 06:30"))
hs["YASA"]          # returns the YASA Hypnogram

# Pairwise agreement: each scorer vs Rater1
ebe = hs.compare(ref="Rater1")
ebe.get_agreement()  # DataFrame, one row per (ref, observer) pair

# All pairwise comparisons (N*(N-1)/2 pairs)
ebe_all = hs.compare()

# Majority-vote consensus, Rater1 breaks ties
consensus_hyp = hs.consensus(ref="Rater1")
# → Hypnogram(scorer="consensus"), same window as common_window

# Unanimous consensus (only epochs where all three scorers agree)
strict = hs.consensus(method="unanimous", fill="UNS")

# Visual stack plot
hs.plot()

# Round-trip to disk
hs.to_json("night1_all_scorers.json")
hs2 = yasa.HypnogramSet.from_json("night1_all_scorers.json")
```

---

## File layout

| File | Change |
|------|--------|
| `src/yasa/hypnoset.py` | New file — `HypnogramSet` class |
| `src/yasa/__init__.py` | Add `HypnogramSet` to exports |
| `src/yasa/hypno.py` | Optionally add `Hypnogram.pad()` helper |
| `tests/test_hypnoset.py` | New test file |

---

## Open questions to resolve before implementation

1. **Epoch-position fallback**: Should `HypnogramSet` fall back to positional alignment (assume all scorers start at epoch 0) when no hypnogram has a `start` datetime? Recommendation: yes, but only when **all** members lack a `start` and all share the same length. Otherwise raise.

2. **Mixed `freq`**: Should `align()` always resample to the finest epoch length (shortest), or require the caller to specify `freq`? Recommendation: default to the finest freq present, with an explicit `freq` override.

3. **`proba` in consensus**: When `proba` is available on all scorers, the consensus `proba` could be the average probability across scorers — a useful future extension.

4. **`n_stages` homogeneity**: Should `HypnogramSet` enforce that all members share the same `n_stages`? Recommendation: yes — raise at construction time if they differ.

5. **Naming**: `HypnogramSet` vs `ScoringSession` vs `MultiHypnogram`. `HypnogramSet` is the most Pythonic and mirrors set-container conventions already used in Python data libraries.
