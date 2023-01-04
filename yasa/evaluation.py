"""
YASA code for evaluating the agreement between two scorers.

There are two levels of evaluating staging performance:
- Comparing two hypnograms (e.g., human vs automated scorer)
- Comparing summary sleep statistics between two scorers (e.g., PSG vs actigraphy)

Analyses are modeled after the standardized framework proposed in Menghini et al., 2021, SLEEP.
See the following resources:
- https://doi.org/10.1093/sleep/zsaa170
- https://sri-human-sleep.github.io/sleep-trackers-performance
- https://github.com/SRI-human-sleep/sleep-trackers-performance
"""
import logging

import numpy as np
import pandas as pd
import pingouin as pg
import sklearn.metrics as skm

import seaborn as sns
import matplotlib.pyplot as plt

from yasa.plotting import plot_hypnogram


logger = logging.getLogger("yasa")

__all__ = [
    "EpochByEpochEvaluation",
    "SleepStatsEvaluation",
]


#############################################################################
# EPOCH BY EPOCH
#############################################################################


class EpochByEpochEvaluation:
    """Evaluate agreement between two collections of hypnograms.

    For example, evaluate the agreement between manually-scored hypnograms and automatically-scored
    hypnograms, or hypnograms derived from actigraphy.

    Many steps here are modeled after guidelines proposed in Menghini et al., 2021 [Menghini2021]_.
    See https://sri-human-sleep.github.io/sleep-trackers-performance/AnalyticalPipeline_v1.0.0.html

    Parameters
    ----------
    refr_hyps : iterable of :py:class:`yasa.Hypnogram`
        A collection of reference (i.e., ground-truth) hypnograms.

        Each :py:class:`yasa.Hypnogram` in ``refr_hyps`` must have the same
        :py:attr:`~yasa.Hypnogram.scorer`.

        If a ``dict``, key values are use to generate unique sleep session IDs. If any other
        iterable (e.g., ``list`` or ``tuple``), then unique sleep session IDs are automatically
        generated.
    test_hyps : iterable of :py:class:`yasa.Hypnogram`
        A collection of test (i.e., to-be-evaluated) hypnograms.

        Each :py:class:`yasa.Hypnogram` in ``test_hyps`` must have the same
        :py:attr:`~yasa.Hypnogram.scorer`, and this scorer must be different than the scorer of
        hypnograms in ``refr_hyps``.

        If a ``dict``, key values must match those of ``refr_hyps``.

    .. important::
        It is assumed that the order of hypnograms are the same in ``refr_hyps`` and ``test_hyps``.
        For example, the third hypnogram in ``refr_hyps`` and ``test_hyps`` come from the same sleep
        session, and only differ in that they have different scorers.

    .. seealso:: For comparing just two hypnograms, use :py:meth:`yasa.Hynogram.evaluate`.

    References
    ----------
    .. [Menghini2021] Menghini, L., Cellini, N., Goldstone, A., Baker, F. C., & de Zambotti, M.
                      (2021). A standardized framework for testing the performance of sleep-tracking
                       technology: step-by-step guidelines and open-source code. Sleep, 44(2),
                       zsaa170. https://doi.org/10.1093/sleep/zsaa170

    Examples
    --------
    >>> import yasa
    >>> hyps_a = [yasa.simulate_hypnogram(tib=600, scorer="RaterA", seed=i) for i in range(20)]
    >>> hyps_b = [h.simulate_similar(scorer="RaterB", seed=i) for i, h in enumerate(refr_hyps)]
    >>> ebe = yasa.EpochByEpochEvaluation(hyps_a, hyps_b)

    >>> ebe.get_agreement().round(3)
    metric
    accuracy              0.209
    kappa                -0.051
    weighted_jaccard      0.130
    weighted_precision    0.247
    weighted_recall       0.209
    weighted_f1           0.223
    Name: agreement, dtype: float64

    >>> ebe.get_agreement_by_stage().round(3)
    stage         WAKE       N1       N2       N3   REM  ART  UNS
    metric
    precision    0.188    0.016    0.315    0.429   0.0  0.0  0.0
    recall       0.179    0.018    0.317    0.235   0.0  0.0  0.0
    fscore       0.183    0.017    0.316    0.303   0.0  0.0  0.0
    support    290.000  110.000  331.000  179.000  50.0  0.0  0.0

    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
        >>> ebe.plot_hypnograms()

    .. plot::

        >>> fig, ax = plt.subplots(figsize=(6, 3))
        >>> ebe.plot_hypnograms(ax=ax, kwargs_test={"color": "black", "lw": 2, "ls": "dotted"})
        >>> plt.tight_layout()

    .. plot::

        >>> fig, ax = plt.subplots(figsize=(6.5, 2.5), constrained_layout=True)
        >>> style_a = dict(alpha=1, lw=2.5, ls="solid", color="gainsboro", label="Michel")
        >>> style_b = dict(alpha=1, lw=2.5, ls="solid", color="cornflowerblue", label="Jouvet")
        >>> legend_style = dict(
        >>>     title="Scorer", frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.9)
        >>> )
        >>> ax = ebe.plot_hypnograms(
        >>>     kwargs_ref=style_a, kwargs_test=style_b, legend=legend_style, ax=ax
        >>> )
        >>>
        >>> acc = ebe.get_agreement().multiply(100).round(0).at["accuracy"]
        >>> ax.text(0.01, 1, f"Accuracy = {acc}%", ha="left", va="bottom", transform=ax.transAxes)

    When comparing only 2 hypnograms, use the :py:meth:`yasa.Hynogram.evaluate` method:

    >>> hypno_a = yasa.simulate_hypnogram(tib=90, scorer="RaterA", seed=8)
    >>> hypno_b = hypno_a.simulate_similar(scorer="RaterB", seed=9)
    >>> ebe = hypno_a.evaluate(hypno_b)

    >>> ebe.get_confusion_matrix()
    RaterB  WAKE   N1   N2  N3  REM  ART  UNS  Total
    RaterA
    WAKE      52   38  126  23   51    0    0    290
    N1        59    2   27   8   14    0    0    110
    N2       117   50  105  15   44    0    0    331
    N3        34   26   62  42   15    0    0    179
    REM       15   12   13  10    0    0    0     50
    ART        0    0    0   0    0    0    0      0
    UNS        0    0    0   0    0    0    0      0
    Total    277  128  333  98  124    0    0    960
    """

    def __init__(self, refr_hyps, test_hyps):
        from yasa.hypno import Hypnogram  # Avoiding circular import

        assert hasattr(refr_hyps, "__iter__"), "`refr_hyps` must be a an iterable"
        assert hasattr(test_hyps, "__iter__"), "`test_hyps` must be a an iterable"
        assert type(refr_hyps) == type(test_hyps), "`refr_hyps` and `test_hyps` must be same type"
        assert len(refr_hyps) == len(
            test_hyps
        ), "`refr_hyps` and `test_hyps` must have the same number of hypnograms"

        if isinstance(refr_hyps, dict):
            # If user provides dictionaries, split into sleep IDs and hypnograms
            assert (
                refr_hyps.keys() == test_hyps.keys()
            ), "hypnograms in `refr_hyps` and `test_hyps` must have identical sleep IDs"
            sleep_ids, refr_hyps = zip(*refr_hyps.items())
            test_hyps = tuple(test_hyps.values())
        else:
            # Create hypnogram_ids
            sleep_ids = tuple(range(1, 1 + len(refr_hyps)))

        assert all(
            isinstance(hyp, Hypnogram) for hyp in refr_hyps + test_hyps
        ), "`refr_hyps` and `test_hyps` must only include YASA hypnograms"
        assert all(
            h.scorer is not None for h in refr_hyps + test_hyps
        ), "all hypnograms must have a scorer name"
        for h1, h2 in zip((refr_hyps + test_hyps)[:-1], (refr_hyps + test_hyps)[1:]):
            assert h1.freq == h2.freq, "all hypnograms must have the same freq"
            assert h1.labels == h2.labels, "all hypnograms must have the same labels"
            assert h1.mapping == h2.mapping, "all hypnograms must have the same mapping"
            assert h1.n_stages == h2.n_stages, "all hypnograms must have the same n_stages"
        assert all(
            h1.scorer == h2.scorer for h1, h2 in zip(refr_hyps[:-1], refr_hyps[1:])
        ), "all `refr_hyps` must have the same scorer"
        assert all(
            h1.scorer == h2.scorer for h1, h2 in zip(test_hyps[:-1], test_hyps[1:])
        ), "all `test_hyps` must have the same scorer"
        assert all(
            h1.scorer != h2.scorer for h1, h2 in zip(refr_hyps, test_hyps)
        ), "each `refr_hyps` and `test_hyps` pair must have unique scorers"
        assert all(
            h1.n_epochs == h2.n_epochs for h1, h2 in zip(refr_hyps, test_hyps)
        ), "each `refr_hyps` and `test_hyps` pair must have the same n_epochs"
        ## Q: Could use set() for those above.
        ##    Or set scorer as the first available and check all equal.

        # Convert to dictionaries with sleep_ids and hypnograms
        refr_hyps = {s: h for s, h in zip(sleep_ids, refr_hyps)}
        test_hyps = {s: h for s, h in zip(sleep_ids, test_hyps)}

        # Merge all hypnograms into a single MultiIndexed dataframe
        refr = pd.concat(
            pd.concat({s: h.as_int()}, names=["sleep_id"]) for s, h in refr_hyps.items()
        )
        test = pd.concat(
            pd.concat({s: h.as_int()}, names=["sleep_id"]) for s, h in test_hyps.items()
        )
        data = pd.concat([refr, test], axis=1)

        ########################################################################
        # INDIVIDUAL-LEVEL AGREEMENT
        ########################################################################

        # Get individual-level averaged/weighted agreement scores
        indiv_agree_avg = data.groupby(level=0).apply(self.multi_scorer).apply(pd.Series)
        ## Q: Check speed against pd.DataFrame({s: multscore(hyps[s], hyps[s]) for s in subjects})

        # Get individual-level one-vs-rest/un-weighted agreement scores
        # Labels ensures the order of returned scores is known
        # It also can be used to remove unused labels, but that will be taken care of later anyways
        # skm_labels = [l for l in refr_hyps[sleep_ids[0]].hypno.cat.categories if l in data.values]
        # skm will return an array of results, so mapping must be linear without skips
        ## Q: Another option is to get Series.cat.codes for ints and use cat.categories for mapping
        skm_labels = np.unique(data).tolist()
        skm_mapping = {i: l for i, l in enumerate(skm_labels)}  # skm integers to YASA integers
        mapping_int = refr_hyps[sleep_ids[0]].mapping_int.copy()  # YASA integers to YASA strings
        # labels = refr_hyps[sleep_ids[0]].labels.copy()  # To preserve YASA ordering
        # labels = [v for k, v in mapping_int.items() if k in skm_labels]  # To preserve YASA ordering
        prfs_wrapper = lambda df: skm.precision_recall_fscore_support(
            *df.values.T, beta=1, labels=skm_labels, average=None, zero_division=0
        )
        indiv_agree_ovr = (
            data
            # Get precision, recall, f1, and support for each individual sleep session
            .groupby(level=0)
            .apply(prfs_wrapper)
            # Unpack arrays
            .explode()
            .apply(pd.Series)
            # Add metric labels and prepend to index, creating MultiIndex
            .assign(metric=["precision", "recall", "fbeta", "support"] * len(refr_hyps))
            .set_index("metric", append=True)
            # Convert stage column names to string labels
            .rename_axis(columns="stage")
            .rename(columns=skm_mapping)
            .rename(columns=mapping_int)
            # Remove all-zero rows (i.e., stages that were not present in the hypnogram)
            .pipe(lambda df: df.loc[:, df.any()])
            # Reshape so metrics are columns
            .stack()
            .unstack("metric")
            .rename_axis(columns=None)
            # Swap MultiIndex levels and sort so stages in standard YASA order
            .swaplevel()
            .sort_index(
                level="stage", key=lambda x: x.map(lambda y: list(mapping_int.values()).index(y))
            )
        )

        # Set attributes
        self._data = data
        self._sleep_ids = sleep_ids
        self._n_sleeps = len(sleep_ids)
        self._refr_hyps = refr_hyps
        self._test_hyps = test_hyps
        self._refr_scorer = refr_hyps[sleep_ids[0]].scorer
        self._test_scorer = test_hyps[sleep_ids[0]].scorer
        self._skm_labels = skm_labels
        self._skm_mapping = skm_mapping
        self._mapping_int = mapping_int
        self._indiv_agree_avg = indiv_agree_avg
        self._indiv_agree_ovr = indiv_agree_ovr
        ## Q: Merge these to one individual agreement dataframe?

    def __repr__(self):
        # TODO v0.8: Keep only the text between < and >
        s = "s" if self._n_sleeps > 1 else ""
        return (
            f"<EpochByEpochEvaluation | Test hypnogram{s} scored by {self.test_scorer} evaluated "
            f"against reference hypnogram{s} scored by {self.refr_scorer}, {self._n_sleeps} sleep "
            f"session{s}>\n"
            " - Use `.get_agreement()` to get agreement measures as a pandas.Series\n"
            " - Use `.plot_hypnograms()` to plot the two hypnograms overlaid\n"
            "See the online documentation for more details."
        )

    def __str__(self):
        return self.__repr__()

    @property
    def data(self):
        """A :py:class:`pandas.DataFrame` including all hypnograms."""
        return self._data

    @property
    def refr_hyps(self):
        """A dictionary of all reference YASA hypnograms with sleep IDs as keys."""
        return self._refr_hyps

    @property
    def test_hyps(self):
        """A dictionary of all test YASA hypnograms with sleep IDs as keys."""
        return self._test_hyps

    @property
    def sleep_ids(self):
        """A tuple of all sleep IDs."""
        return self._sleep_ids

    @property
    def n_sleeps(self):
        """The number of unique sleep sessions."""
        return self._n_sleeps

    @property
    def refr_scorer(self):
        """The name of the reference scorer."""
        return self._refr_scorer

    @property
    def test_scorer(self):
        """The name of the test scorer."""
        return self._test_scorer

    @property
    def indiv_agree_avg(self):
        """
        A :py:class:`pandas.DataFrame` of ``refr_hyp``/``test_hyp`` average-based agreement scores
        for each individual sleep session.

        .. seealso:: :py:attr:`yasa.EpochByEvaluation.indiv_agree_ovr`
        """
        return self._indiv_agree_avg

    @property
    def indiv_agree_ovr(self):
        """
        A :py:class:`pandas.DataFrame` of ``refr_hyp``/``test_hyp`` one-vs-rest agreement scores
        for each individual sleep session. Agreement scores are provided for each sleep stage.

        .. seealso:: :py:attr:`yasa.EpochByEvaluation.indiv_agree_avg`
        """
        return self._indiv_agree_ovr

    @staticmethod
    def multi_scorer(df, weights=None):
        """Compute multiple agreement scores from a 2-column dataframe.

        This function offers convenience when calculating multiple agreement scores using
        :py:meth:`pandas.DataFrame.groupby.apply`. Scikit-learn doesn't include a function that
        return multiple scores, and the GroupBy implementation of ``apply`` in pandas does not
        accept multiple functions.

        Parameters
        ----------
        df : :py:class:`pandas.DataFrame`
            A :py:class:`pandas.DataFrame` with exactly 2 columns and length of *n_samples*.
            The first column contains true values and second column contains predicted values.

        weights : None or :py:class:`pandas.Series`
            Sample weights passed to underlying :py:mod:`sklearn.metrics` functions when possible.
            If a :py:class:`pandas.Series`, the index must match exactly that of
            :py:attr:`~yasa.Hypnogram.data`.

        Returns
        -------
        scores : dict
            A dictionary with scorer names (``str``) as keys and scores (``float``) as values.
        """
        assert isinstance(weights, type(None)) or weights in df, "`weights` must be None or a column in `df`"
        if weights is not None:
            raise NotImplementedError("Custom `weights` not currently supported")
        t, p = zip(*df.values)  # Same as (df["col1"], df["col2"]) but teensy bit faster
        # t = df["col1"].to_numpy()
        # p = df["col2"].to_numpy()
        w = df["col3"].to_numpy() if weights is not None else weights
        ## Q: The dictionary below be compiled more concisely if we were comfortable accessing
        ##    "private" attributes. I understand that's a no-no but I'm not exactly sure why.
        ##     For example:
        ##     >>> scorers = ["accuracy", "recall"]
        ##     >>> funcs = { s: skm.__getattribute__(f"{s}_scorer") for s in scorers }
        ##     >>> scores = { s: f(true, pred) for s, f in funcs.items() }
        ##     Keywords could be applied as needed by checking f.__kwdefaults__
        ##     This would offer an easy way for users to add their own scorers with an arg as well.
        return {
            "accuracy": skm.accuracy_score(t, p, normalize=True, sample_weight=w),
            "balanced_acc": skm.balanced_accuracy_score(t, p, adjusted=False, sample_weight=w),
            "kappa": skm.cohen_kappa_score(t, p, labels=None, weights=None, sample_weight=w),
            "mcc": skm.matthews_corrcoef(t, p, sample_weight=w),
            "precision": skm.precision_score(
                t, p, average="weighted", sample_weight=w, zero_division=0
            ),
            "recall": skm.recall_score(t, p, average="weighted", sample_weight=w, zero_division=0),
            "fbeta": skm.fbeta_score(
                t, p, beta=1, average="weighted", sample_weight=w, zero_division=0
            ),
        }

    def summary(self, by_stage=False, **kwargs):
        """Return group-level agreement scores.

        Parameters
        ----------
        self : :py:class:`yasa.EpochByEvaluation`
            A :py:class:`yasa.EpochByEvaluation` instance.
        by_stage : bool
            If True, returned ``summary`` :py:class:`pandas.DataFrame` will include agreement scores
            for each sleep stage, derived from one-vs-rest metrics. If False (default), ``summary``
            will include agreement scores derived from average-based metrics.
        **kwargs : key, value pairs
            Additional keyword arguments are passed to :py:meth:`pandas.DataFrame.groupby.agg`.

        Returns
        -------
        summary : :py:class:`pandas.DataFrame`
            A :py:class:`pandas.DataFrame` summarizing agreement scores across the entire dataset
            with descriptive statistics.

            >>> ebe = yasa.EpochByEpochEvaluation(...)
            >>> ebe.summary()

            This will give a :py:class:`pandas.DataFrame` where each row is an agreement metric and
            each column is a descriptive statistic (e.g., mean, standard deviation).
            To control the descriptive statistics included as columns:

            >>> ebe.summary(func=["count", "mean", "sem"])
        """
        assert isinstance(by_stage, bool), "`by_stage` must be True or False"
        mad = lambda df: (df - df.mean()).abs().mean()
        mad.__name__ = "mad"  # Pandas uses this to name the aggregated column
        agg_kwargs = {"func": [mad, "mean", "std", "min", "median", "max"]} | kwargs
        if by_stage:
            summary = (
                self.indiv_agree_ovr.groupby("stage")
                .agg(**agg_kwargs)
                .stack(0)
                .rename_axis(["stage", "metric"])
            )
        else:
            summary = self.indiv_agree_avg.agg(**agg_kwargs).T.rename_axis("metric")
            ## Q: Should we include a column that calculates agreement treating all hypnograms as
            ##    coming from one individual? Others sometimes report it, though I find it mostly
            ##    meaningless because of possible n_epochs imbalances between subjects. I vote no.
            # summary.insert(0, "all", self.multi_scorer(self.data))
        ## Q: Alternatively, we could remove the `by_stage` parameter and stack these into
        ##    one merged DataFrame where the results that are *not* by-stage are included
        ##    with an "all" stage label:
        ## >>> summary = summary.assign(stage="all").set_index("stage", append=True).swaplevel()
        ## >>> summary = pd.concat([summary, summary_ovr]).sort_index()
        return summary

    def get_sleep_stats(self):
        """
        Return a :py:class:`pandas.DataFrame` of sleep statistics for each individual derived from
        both reference and test scorers.

        .. seealso:: :py:meth:`yasa.Hypnogram.sleep_statistics`

        .. seealso:: :py:class:`yasa.SleepStatsEvaluation`

        Parameters
        ----------
        self : :py:class:`yasa.EpochByEvaluation`
            A :py:class:`yasa.EpochByEvaluation` instance.

        Returns
        -------
        sstats : :py:class:`pandas.DataFrame`
            A :py:class:`pandas.DataFrame` with sleep statistics as columns and two rows for each
            individual (one from reference scorer and another from test scorer).
        """
        # Get all sleep statistics
        refr_sstats = pd.DataFrame({s: h.sleep_statistics() for s, h in self.refr_hyps.items()})
        test_sstats = pd.DataFrame({s: h.sleep_statistics() for s, h in self.test_hyps.items()})
        # Reshape and name axis
        refr_sstats = refr_sstats.T.rename_axis("sleep_id")
        test_sstats = test_sstats.T.rename_axis("sleep_id")
        # Convert to MultiIndex with new scorer level
        refr_sstats = pd.concat({self.refr_scorer: refr_sstats}, names=["scorer"])
        test_sstats = pd.concat({self.test_scorer: test_sstats}, names=["scorer"])
        return pd.concat([refr_sstats, test_sstats])

    def get_confusion_matrix(self, sleep_id=None, agg_func=None, **kwargs):
        """
        Return a ``refr_hyp``/``test_hyp``confusion matrix from either a single session or all
        sessions concatenated together.

        Parameters
        ----------
        self : :py:class:`yasa.EpochByEvaluation`
            A :py:class:`yasa.EpochByEvaluation` instance.
        sleep_id : None or a valid sleep ID
            If None (default), cross-tabulation is derived from the entire group dataset.
            If a valid sleep ID, cross-tabulation is derived using only the reference and test
            scored hypnograms from that sleep session.
        ## Q: This keyword (agg_func) is too complicated, but I wanted your opinion on the best
        ##    approach. And I wanted you to see the returned value when agg_func=None because it
        ##    might be best to generate during __init__ to set and access as an attribute.
        agg_func : str, list, or None
            If None (default), group results returns a :py:class:`~pandas.DataFrame` complete with
            all individual sleep session results. If not None, group results returns a
            :py:class:`~pandas.DataFrame` aggregated across individual sleep sessions where
            ``agg_func`` is passed as ``func`` parameter in :py:meth:`pandas.DataFrame.groupby.agg`.
            Ignored if ``sleep_id`` is not None.
        **kwargs : key, value pairs
            Additional keyword arguments are passed to :py:func:`sklearn.metrics.confusion_matrix`.

        Returns
        -------
        conf_matr : :py:class:`pandas.DataFrame`
            A confusion matrix with stages from the reference scorer as indices and stages from the
            test scorer as columns.

        Examples
        --------
        >>> ebe = yasa.EpochByEpochEvaluation(...)
        >>> ebe.get_confusion_matrix()  # Return results from all individual subjects
        >>> ebe.get_confusion_matrix(agg_func=["mean", "std"])  # Return summary results
        >>> ebe.get_confusion_matrix(sleep_id="sub-002")  # Return results from one subject
        """
        assert (
            sleep_id is None or sleep_id in self.sleep_ids
        ), "`sleep_id` must be None or a valid sleep ID"
        kwargs = {"labels": self._skm_labels} | kwargs
        # Get confusion matrix for each individual sleep session
        ## Q: Should this be done during __init__ and accessible via attribute?
        conf_mats = (self.data
            # Get confusion matrix for each individual sleep session
            .groupby(level=0).apply(lambda df: skm.confusion_matrix(*df.values.T, **kwargs))
            # Expand results matrix out from single cell
            .explode().apply(pd.Series)
            # Convert to MultiIndex with reference scorer as new level
            .assign(**{self.refr_scorer: self._skm_labels * self.n_sleeps})
            .set_index(self.refr_scorer, append=True).rename_axis(columns=self.test_scorer)
            # Convert sleep stage columns and indices to strings
            .rename(columns=self._skm_mapping).rename(columns=self._mapping_int)
            .rename(index=self._skm_mapping, level=self.refr_scorer)
            .rename(index=self._mapping_int, level=self.refr_scorer)
        )
        if sleep_id is None:
            if agg_func is None:
                mat = conf_mats
            else:
                mat = conf_mats.groupby(self.refr_scorer).agg(agg_func)
                mat.columns = mat.columns.map("_".join).set_names(self.test_scorer)
        else:
            mat = conf_mats.loc[sleep_id]
        return mat

    def plot_hypnograms(self, sleep_id=None, legend=True, ax=None, refr_kwargs={}, test_kwargs={}):
        """Plot the two hypnograms, where the reference hypnogram is overlaid on the test hypnogram.

        .. seealso:: :py:func:`yasa.plot_hypnogram`

        Parameters
        ----------
        sleep_id : None or a valid sleep ID
            If a valid sleep ID, plot the reference and test hypnograms from on sleep session.
        legend : bool or dict
            If True (default) or a dictionary, a legend is added. If a dictionary, all key/value
            pairs are passed as keyword arguments to the :py:func:`matplotlib.pyplot.legend` call.
        ax : :py:class:`matplotlib.axes.Axes` or None
            Axis on which to draw the plot, optional.
        refr_kwargs : dict
            Keyword arguments passed to :py:func:`yasa.plot_hypnogram` when plotting the reference
            hypnogram.
        test_kwargs : dict
            Keyword arguments passed to :py:func:`yasa.plot_hypnogram` when plotting the test
            hypnogram.

        Returns
        -------
        ax : :py:class:`matplotlib.axes.Axes`
            Matplotlib Axes

        Examples
        --------
        .. plot::

            >>> from yasa import simulate_hypnogram
            >>> hyp = simulate_hypnogram(seed=7)
            >>> ax = hyp.evaluate(hyp.simulate_similar()).plot_hypnograms()
        """
        assert (
            sleep_id is None or sleep_id in self.sleep_ids
        ), "`sleep_id` must be None or a valid sleep ID"
        assert isinstance(legend, (bool, dict)), "`legend` must be True, False, or a dictionary"
        assert isinstance(refr_kwargs, dict), "`refr_kwargs` must be a dictionary"
        assert isinstance(test_kwargs, dict), "`test_kwargs` must be a dictionary"
        assert (
            not "ax" in refr_kwargs | test_kwargs
        ), "ax can't be supplied to `kwargs_ref` or `test_kwargs`, use the `ax` keyword instead"
        if sleep_id is None:
            if self.n_sleeps == 1:
                refr_hyp = self.refr_hyps[self.sleep_ids[0]]
                test_hyp = self.test_hyps[self.sleep_ids[0]]
            else:
                raise NotImplementedError("Multi-session plotting is not currently supported")
        else:
            refr_hyp = self.refr_hyps[sleep_id]
            test_hyp = self.test_hyps[sleep_id]
        plot_refr_kwargs = {"highlight": None, "alpha": 0.8}
        plot_test_kwargs = {"highlight": None, "alpha": 0.8, "color": "darkcyan", "ls": "dashed"}
        plot_refr_kwargs.update(refr_kwargs)
        plot_test_kwargs.update(test_kwargs)
        if ax is None:
            ax = plt.gca()
        refr_hyp.plot_hypnogram(ax=ax, **plot_refr_kwargs)
        test_hyp.plot_hypnogram(ax=ax, **plot_test_kwargs)
        if legend and "label" in plot_refr_kwargs | plot_test_kwargs:
            if isinstance(legend, dict):
                ax.legend(**legend)
            else:
                ax.legend()
        return ax

    def plot_roc(self, sleep_id=None, palette=None, ax=None, **kwargs):
        """Plot ROC curves for each stage.

        Parameters
        ----------
        palette : dict or None
            If a dictionary, keys are stages and values are corresponding colors.
        ax : :py:class:`matplotlib.axes.Axes`
            Axis on which to draw the plot, optional.
        **kwargs : key, value pairs
            Additional keyword arguments are passed to the :py:func:`matplotlib.pyplot.plot` call.

        Returns
        -------
        ax : :py:class:`matplotlib.axes.Axes`
            Matplotlib Axes
        """
        assert (
            sleep_id is None or sleep_id in self.sleep_ids
        ), "`sleep_id` must be None or a valid sleep ID"
        raise NotImplementedError("ROC plots will be implemented once YASA hypnograms have probas.")


#############################################################################
# SLEEP STATISTICS
#############################################################################


class SleepStatsEvaluation:
    """
    Evaluate agreement between two scorers (e.g., two different manual scorers or one manual scorer
    and YASA's automatic staging) by comparing their summary sleep statistics derived from multiple
    subjects or sessions.

    Parameters
    ----------
    refr_data : :py:class:`pandas.DataFrame`
        A :py:class:`pandas.DataFrame` with sleep statistics from the reference scorer.
        Rows are individual sleep sessions and columns are individual sleep statistics.
    test_data : :py:class:`pandas.DataFrame`
        A :py:class:`pandas.DataFrame` with sleep statistics from the test scorer.
        Shape, indices, and columns must be identical to ``refr_data``.
    refr_scorer : str
        Name of the reference scorer, used for labeling.
    test_scorer : str
        Name of the test scorer, used for labeling.
    alpha : float
        Alpha cutoff used for all three tests.
    kwargs_normality : dict
        Keywords arguments passed to the :py:func:`pingouin.normality` call.
    kwargs_regression : dict
        Keywords arguments passed to the :py:func:`pingouin.linear_regression` call.
    kwargs_homoscedasticity : dict
        Keywords arguments passed to the :py:func:`pingouin.homoscedasticity` call.

    Notes
    -----
    Many steps here are modeled after guidelines proposed in Menghini et al., 2021 [Menghini2021]_.
    See https://sri-human-sleep.github.io/sleep-trackers-performance/AnalyticalPipeline_v1.0.0.html

    References
    ----------
    .. [Menghini2021] Menghini, L., Cellini, N., Goldstone, A., Baker, F. C., & de Zambotti, M.
                      (2021). A standardized framework for testing the performance of sleep-tracking
                       technology: step-by-step guidelines and open-source code. Sleep, 44(2),
                       zsaa170. https://doi.org/10.1093/sleep/zsaa170

    Examples
    --------
    >>> import pandas as pd
    >>> import yasa
    >>>
    >>> # For this example, generate two fake datasets of sleep statistics
    >>> hypsA = [yasa.simulate_hypnogram(tib=600, seed=i) for i in range(20)]
    >>> hypsB = [h.simulate_similar(tib=600, seed=i) for i, h in enumerate(hypsA)]
    >>> sstatsA = pd.Series(hypsA).map(lambda h: h.sleep_statistics()).apply(pd.Series)
    >>> sstatsB = pd.Series(hypsB).map(lambda h: h.sleep_statistics()).apply(pd.Series)
    >>> sstatsA.index = sstatsB.index = sstatsA.index.map(lambda x: f"sub-{x+1:03d}")
    >>>
    >>> sse = yasa.SleepStatsEvaluation(sstatsA, sstatsB)
    >>>
    >>> sse.summary(descriptives=False)
           normal  unbiased  homoscedastic
    sstat
    %N1      True      True           True
    %N2      True      True           True
    %N3      True      True           True
    %REM    False      True           True
    SE       True      True           True
    SOL     False     False           True
    TST      True      True           True

    Access more detailed statistical output of each test.

    >>> sse.normality
                  W      pval  normal
    sstat
    %N1    0.973407  0.824551    True
    %N2    0.960684  0.557595    True
    %N3    0.958591  0.516092    True
    %REM   0.901733  0.044447   False
    SE     0.926732  0.133580    True
    SOL    0.774786  0.000372   False
    TST    0.926733  0.133584    True
    WASO   0.924288  0.119843    True

    >>> sse.homoscedasticity.head(2)
                  W      pval  equal_var
    sstat
    %N1    0.684833  0.508274       True
    %N2    0.080359  0.922890       True

    >>> sse.proportional_bias.round(3).head(2)
            coef     se      T   pval     r2  adj_r2  CI[2.5%]  CI[97.5%]  unbiased
    sstat
    %N1   -0.487  0.314 -1.551  0.138  0.118   0.069    -1.146      0.172      True
    %N2   -0.107  0.262 -0.409  0.688  0.009  -0.046    -0.658      0.444      True

    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> ax = sse.plot_discrepancies_heatmap()
        >>> ax.set_title("Sleep statistic discrepancies")
        >>> plt.tight_layout()

    .. plot::

        >>> sse.plot_blandaltman()
    """

    def __init__(
        self,
        refr_data,
        test_data,
        *,
        refr_scorer="Reference",
        test_scorer="Test",
        kwargs_normality={"alpha": 0.05},
        kwargs_regression={"alpha": 0.05},
        kwargs_homoscedasticity={"alpha": 0.05},
    ):
        assert isinstance(refr_data, pd.DataFrame), "`refr_data` must be a pandas DataFrame"
        assert isinstance(test_data, pd.DataFrame), "`test_data` must be a pandas DataFrame"
        assert np.array_equal(
            refr_data.index, test_data.index
        ), "`refr_data` and `test_data` index values must be identical"
        assert (
            refr_data.index.name == test_data.index.name
        ), "`refr_data` and `test_data` index names must be identical"
        assert np.array_equal(
            refr_data.columns, test_data.columns
        ), "`refr_data` and `test_data` column values must be identical"
        assert isinstance(refr_scorer, str), "`refr_scorer` must be a string"
        assert isinstance(test_scorer, str), "`test_scorer` must be a string"
        assert refr_scorer != test_scorer, "`refr_scorer` and `test_scorer` must be unique"
        assert isinstance(kwargs_normality, dict), "`kwargs_normality` must be a dictionary"
        assert isinstance(kwargs_regression, dict), "`kwargs_regression` must be a dictionary"
        assert isinstance(kwargs_homoscedasticity, dict), "`kwargs_homoscedasticity` must be a dict"
        assert "alpha" in kwargs_normality, "`kwargs_normality` must include 'alpha'"
        assert "alpha" in kwargs_regression, "`kwargs_regression` must include 'alpha'"
        assert "alpha" in kwargs_homoscedasticity, "`kwargs_homoscedasticity` must include 'alpha'"

        # If refr_data and test_data indices are unnamed, name them
        sleep_id_str = "sleep_id" if refr_data.index.name is None else refr_data.index.name
        refr_data.index.name = sleep_id_str
        test_data.index.name = sleep_id_str

        # Get scorer differences
        diff_data = test_data.sub(refr_data)

        # Convert to MultiIndex with new scorer level
        diff_data = pd.concat({"difference": diff_data}, names=["scorer"])
        refr_data = pd.concat({refr_scorer: refr_data}, names=["scorer"])
        test_data = pd.concat({test_scorer: test_data}, names=["scorer"])

        # Merge dataframes and reshape to long format
        data = pd.concat([refr_data, test_data, diff_data])
        data = (
            data.melt(var_name="sstat", ignore_index=False)
            .reset_index()
            .pivot(columns="scorer", index=[sleep_id_str, "sstat"], values="value")
            .reset_index()
            .rename_axis(columns=None)
        )

        # Remove sleep statistics that have no differences between scorers
        stats_nodiff = data.groupby("sstat")["difference"].any().loc[lambda x: ~x].index.tolist()
        data = data.query(f"~sstat.isin({stats_nodiff})")
        for s in stats_nodiff:
            logger.warning(f"All {s} differences are zero, removing from evaluation.")

        ## NORMALITY ##
        # Test reference data for normality at each sleep statistic
        normality = (
            data.groupby("sstat")[refr_scorer].apply(pg.normality, **kwargs_normality).droplevel(-1)
        )

        ## PROPORTIONAL BIAS ##
        # Test each sleep statistic for proportional bias
        prop_bias_results = []
        residuals_results = []
        for ss_name, ss_df in data.groupby("sstat"):
            # Regress the difference scores on the reference scores
            model = pg.linear_regression(
                ss_df[refr_scorer], ss_df["difference"], **kwargs_regression
            )
            model.insert(0, "sstat", ss_name)
            # Extract sleep-level residuals for later homoscedasticity tests
            resid_dict = {
                sleep_id_str: ss_df[sleep_id_str],
                "sstat": ss_name,
                "pbias_residual": model.residuals_,
            }
            resid = pd.DataFrame(resid_dict)
            prop_bias_results.append(model)
            residuals_results.append(resid)
        # Add residuals to raw dataframe, used later when testing homoscedasticity
        data = data.merge(pd.concat(residuals_results), on=[sleep_id_str, "sstat"])
        # Handle proportional bias results
        prop_bias = pd.concat(prop_bias_results)
        # Save all the proportional bias models before removing intercept, for optional user access
        prop_bias_full = prop_bias.reset_index(drop=True)
        # Now remove intercept rows
        prop_bias = prop_bias.query("names != 'Intercept'").drop(columns="names").set_index("sstat")
        # Add True/False passing column for easy access
        prop_bias["unbiased"] = prop_bias["pval"].ge(kwargs_regression["alpha"])

        ## Test each statistic for homoscedasticity ##
        columns = [refr_scorer, "difference", "pbias_residual"]
        homoscedasticity_f = lambda df: pg.homoscedasticity(df[columns], **kwargs_homoscedasticity)
        homoscedasticity = data.groupby("sstat").apply(homoscedasticity_f).droplevel(-1)

        # Set attributes
        self._data = data
        self._normality = normality
        self._proportional_bias = prop_bias
        self._proportional_bias_full = prop_bias_full  ## Q: Is this worth saving??
        self._homoscedasticity = homoscedasticity
        self._refr_scorer = refr_scorer
        self._test_scorer = test_scorer
        self._sleep_id_str = sleep_id_str
        self._n_sleeps = data[sleep_id_str].nunique()
        self._diff_data = diff_data.drop(columns=stats_nodiff)
        # self._diff_data = data.pivot(index=sleep_id_str, columns="sstat", values="difference")

    @property
    def data(self):
        """A :py:class:`pandas.DataFrame` containing all sleep statistics from ``refr_data`` and
        ``test_data`` as well as their difference scores (``test_data`` minus ``refr_data``).
        """
        return self._data

    @property
    def diff_data(self):
        """A :py:class:`pandas.DataFrame` of ``test_data`` minus ``refr_data``."""
        # # Pivot for session-rows and statistic-columns
        return self._diff_data

    @property
    def refr_scorer(self):
        """The name of the reference scorer."""
        return self._refr_scorer

    @property
    def test_scorer(self):
        """The name of the test scorer."""
        return self._test_scorer

    @property
    def sleep_id_str(self):
        """The name of the unique sleep session identifier."""
        return self._sleep_id_str

    @property
    def n_sleeps(self):
        """The number of sleep sessions."""
        return self._n_sleeps

    @property
    def normality(self):
        """A :py:class:`pandas.DataFrame` of normality results for all sleep statistics."""
        return self._normality

    @property
    def homoscedasticity(self):
        """A :py:class:`pandas.DataFrame` of homoscedasticity results for all sleep statistics."""
        return self._homoscedasticity

    @property
    def proportional_bias(self):
        """
        A :py:class:`pandas.DataFrame` of proportional bias results for all sleep statistics, with
        intercept terms removed.
        """
        return self._proportional_bias

    @property
    def proportional_bias_full(self):
        """A :py:class:`pandas.DataFrame` of proportional bias results for all sleep statistics."""
        return self._proportional_bias_full

    def __repr__(self):
        # TODO v0.8: Keep only the text between < and >
        return (
            f"<SleepStatsEvaluation | Test scorer {self.test_scorer} evaluated against reference "
            f"scorer {self.refr_scorer}, {self.n_sleeps} sleep sessions>\n"
            " - Use `.summary()` to get pass/fail values from various checks\n"
            " - Use `.plot_blandaltman()` to get a Bland-Altman-plot grid for sleep statistics\n"
            "See the online documentation for more details."
        )

    def __str__(self):
        return __repr__()

    def summary(self, **kwargs):
        """Return a summary dataframe highlighting whether tests passed for each sleep statistic.

        Parameters
        ----------
        self : :py:class:`SleepStatsEvaluation`
            A :py:class:`SleepStatsEvaluation` instance.
        **kwargs : key, value pairs
            Additional keyword arguments are passed to :py:meth:`pandas.DataFrame.groupby.agg`.

            >>> ebe.summary(func=["mean", "sem", "min", "max"])

        Returns
        -------
        summary : :py:class:`pandas.DataFrame`
            A :py:class:`pandas.DataFrame` with boolean values indicating the pass/fail status for
            normality, proportional bias, and homoscedasticity tests (for each sleep statistic).
        """
        series_list = [
            self.normality["normal"],
            self.proportional_bias["unbiased"],
            self.homoscedasticity["equal_var"].rename("homoscedastic"),
        ]
        summary = pd.concat(series_list, axis=1)
        mad = lambda df: (df - df.mean()).abs().mean()
        mad.__name__ = "mad"  # Pandas uses this to name the aggregated column
        agg_kwargs = {"func": [mad, "mean", "std"]} | kwargs
        desc = self.data.drop(columns=self.sleep_id_str).groupby("sstat").agg(**agg_kwargs)
        desc.columns = desc.columns.map("_".join)
        return summary.join(desc)

    def plot_discrepancies_heatmap(self, sleep_stats=None, **kwargs):
        """Visualize session-level discrepancies, generally for outlier inspection.

        Parameters
        ----------
        sleep_stats : list or None
            List of sleep statistics to plot. Default (None) is to plot all sleep statistics.
        **kwargs : key, value pairs
            Additional keyword arguments are passed to the :py:func:`seaborn.heatmap` call.

        Returns
        -------
        ax : :py:class:`matplotlib.axes.Axes`
            Matplotlib Axes
        """
        assert isinstance(sleep_stats, (list, type(None))), "`sleep_stats` must be a list or None"
        if sleep_stats is None:
            sleep_stats = self.data["sstat"].unique()  # All available sleep statistics
        heatmap_kwargs = {"cmap": "binary", "annot": True, "fmt": ".1f", "square": False}
        heatmap_kwargs["cbar_kws"] = dict(label="Normalized discrepancy %")
        if "cbar_kws" in kwargs:
            heatmap_kwargs["cbar_kws"].update(kwargs["cbar_kws"])
        heatmap_kwargs.update(kwargs)
        table = self.diff_data[sleep_stats]
        # Normalize statistics (i.e., columns) between zero and one then convert to percentage
        table_norm = table.sub(table.min(), axis=1).div(table.apply(np.ptp)).multiply(100)
        if heatmap_kwargs["annot"]:
            # Use raw values for writing
            heatmap_kwargs["annot"] = table.to_numpy()
        return sns.heatmap(table_norm, **heatmap_kwargs)

    def plot_discrepancies_dotplot(self, kwargs_pairgrid={"palette": "winter"}, **kwargs):
        """Visualize session-level discrepancies, generally for outlier inspection.

        Parameters
        ----------
        kwargs_pairgrid : dict
            Keywords arguments passed to the :py:class:`seaborn.PairGrid` call.
        **kwargs : key, value pairs
            Additional keyword arguments are passed to the :py:func:`seaborn.stripplot` call.

        Returns
        -------
        g : :py:class:`seaborn.PairGrid`
            A :py:class:`seaborn.FacetGrid` with sleep statistics dotplots on each axis.

        Examples
        --------
        To plot a limited subset of sleep statistics, use the ``x_vars`` keyword argument of
        :py:class:`seaborn.PairGrid`.

        .. plot::
            ## TODO: Example using x_vars
        """
        assert isinstance(kwargs_pairgrid, dict), "`kwargs_pairgrid` must be a dict"
        stripplot_kwargs = {"size": 10, "linewidth": 1, "edgecolor": "white"}
        stripplot_kwargs.update(kwargs)
        # Initialize the PairGrid
        height = 0.3 * len(self.diff_data)
        aspect = 0.6
        pairgrid_kwargs = dict(hue=self.sleep_id_str, height=height, aspect=aspect)
        pairgrid_kwargs.update(kwargs_pairgrid)
        g = sns.PairGrid(
            self.diff_data.reset_index(), y_vars=[self.sleep_id_str], **pairgrid_kwargs
        )
        # Draw the dots
        g.map(sns.stripplot, orient="h", jitter=False, **stripplot_kwargs)
        # Adjust aesthetics
        for ax in g.axes.flat:
            ax.set(title=ax.get_xlabel())
            ax.margins(x=0.3)
            ax.yaxis.grid(True)
            ax.tick_params(left=False)
        g.set(xlabel="", ylabel="")
        sns.despine(left=True, bottom=True)
        return g

    def plot_blandaltman(self, kwargs_facetgrid={}, **kwargs):
        """

        **Use col_order=sstats_order for plotting a subset.

        Parameters
        ----------
        kwargs_facetgrid : dict
            Keyword arguments passed to the :py:class:`seaborn.FacetGrid` call.
        **kwargs : key, value pairs
            Additional keyword arguments are passed to :py:func:`pingouin.plot_blandaltman`.

        Returns
        -------
        g : :py:class:`seaborn.FacetGrid`
            A :py:class:`seaborn.FacetGrid` with sleep statistics Bland-Altman plots on each axis.
        """
        facetgrid_kwargs = dict(col_wrap=4, height=2, aspect=1, sharex=False, sharey=False)
        facetgrid_kwargs.update(kwargs_facetgrid)
        blandaltman_kwargs = dict(xaxis="y", annotate=False, edgecolor="black", facecolor="none")
        blandaltman_kwargs.update(kwargs)
        # Initialize a grid of plots with an Axes for each sleep statistic
        g = sns.FacetGrid(self.data, col="sstat", **facetgrid_kwargs)
        # Draw Bland-Altman plot on each axis
        g.map(pg.plot_blandaltman, self.test_scorer, self.refr_scorer, **blandaltman_kwargs)
        # Adjust aesthetics
        for ax in g.axes.flat:
            # Tidy-up axis limits with symmetric y-axis and minimal ticks
            bound = max(map(abs, ax.get_ylim()))
            ax.set_ylim(-bound, bound)
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=2, integer=True, symmetric=True))
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=1, integer=True))
        ylabel = " - ".join((self.test_scorer, self.refr_scorer))
        g.set_ylabels(ylabel)
        g.set_titles(col_template="{col_name}")
        g.tight_layout(w_pad=1, h_pad=2)
        return g
