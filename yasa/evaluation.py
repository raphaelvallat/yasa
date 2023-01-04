"""
YASA code for evaluating the agreement between two sleep-measurement systems.

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
    """
    See :py:meth:`yasa.Hypnogram.evaluate`

    Parameters
    ----------
    refr_hyp : :py:class:`yasa.Hypnogram`
        The reference or ground-truth hypnogram, or sequence of hypnograms.
    test_hyp : :py:class:`yasa.Hypnogram`
        The test or to-be-evaluated hypnogram, or sequence of hypnograms.

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
    >>> import yasa
    >>> hypno_a = yasa.simulate_hypnogram(tib=90, seed=8, scorer="RaterA")
    >>> hypno_b = yasa.simulate_hypnogram(tib=90, seed=9, scorer="RaterB")
    >>> ebe = yasa.EpochByEpochEvaluation(hypno_a, hypno_b)  # or hypno_a.evaluate(hypno_b)
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
    """
    def __init__(self, refr_hyps, test_hyps):
        from yasa.hypno import Hypnogram  # Loading here to avoid circular import

        assert isinstance(refr_hyps, Hypnogram) or hasattr(refr_hyps, "__iter__"), (
            "`refr_hyps` must be a YASA hypnogram or iterable containing multiple YASA hypnograms"
        )
        assert isinstance(test_hyps, Hypnogram) or hasattr(test_hyps, "__iter__"), (
            "`test_hyps` must be a YASA hypnogram or iterable containing multiple YASA hypnograms"
        )
        assert type(refr_hyps) == type(test_hyps), "`refr_hyps` and `test_hyps` must be same type"

        # Convert solo hypnograms to len==1 tuples
        if isinstance(refr_hyps, Hypnogram):  # As below, picking refr_hyps for checks arbitrarily
            refr_hyps = [refr_hyps]
            test_hyps = [test_hyps]
        assert len(refr_hyps) == len(test_hyps), "must have same number of subjects"

        if isinstance(refr_hyps, dict):
            assert refr_hyps.keys() == test_hyps.keys(), "must have same subject identifiers and in same order"
            subjects, refr_hyps = zip(*refr_hyps.items())
            # assert all(isinstance(s, str) for s in subjects)
            test_hyps = tuple(test_hyps.values())
        else:
            subjects = 1 + np.arange(len(refr_hyps))

        all_hyps = refr_hyps + test_hyps
        assert all(isinstance(hyp, Hypnogram) for hyp in all_hyps), "`refr_hyps` and `test_hyps` must only include hypnograms"
        assert all(h.scorer is not None for h in all_hyps), "all hypnograms must have a scorer"
        for h1, h2 in zip(all_hyps[:-1], all_hyps[1:]):
            assert h1.n_stages == h2.n_stages, "all hypnograms must have the same n_stages"
            assert h1.labels == h2.labels, "all hypnograms must have the same labels"
            assert h1.mapping == h2.mapping, "all hypnograms must have the same mapping"
        assert all(h1.scorer == h2.scorer for h1, h2 in zip(refr_hyps[:-1], refr_hyps[1:])), "all `refr_hyps` must have the same scorer"
        assert all(h1.scorer == h2.scorer for h1, h2 in zip(test_hyps[:-1], test_hyps[1:])), "all `test_hyps` must have the same scorer"
        assert all(h1.scorer != h2.scorer for h1, h2 in zip(refr_hyps, test_hyps)), "each `refr_hyps` and `test_hyps` pair must have unique scorers"
        assert all(h1.n_epochs == h2.n_epochs for h1, h2 in zip(refr_hyps, test_hyps)), "each `refr_hyps` and `test_hyps` pair must have the same n_epochs"
        ## Could use set() for those above
        ## Or set scorer as the first available and check all equal

        # Convert to dictionaries with subjects and hypnograms
        refr_hyps = { s: h for s, h in zip(subjects, refr_hyps) }
        test_hyps = { s: h for s, h in zip(subjects, test_hyps) }

        # Merge all hypnograms into a single multiindexed dataframe
        refr = pd.concat(pd.concat({s: h.hypno}, names=["subject"]) for s, h in refr_hyps.items())
        test = pd.concat(pd.concat({s: h.hypno}, names=["subject"]) for s, h in test_hyps.items())
        data = pd.concat([refr, test], axis=1)

        ########################################################################
        # INDIVIDUAL-LEVEL AGREEMENT
        ########################################################################

        # Get individual-level averaged/weighted agreement scores
        # indiv_agree_avg = pd.DataFrame({s: multi_scorer_avg(refr_hyps[s].as_int(), test_hyps[s].as_int()) for s in subjects })
        indiv_agree_avg = data.groupby(level=0).apply(self.multi_scorer_avg).apply(pd.Series)

        # Get individual-level one-vs-rest/un-weighted agreement scores
        # Only include stages that appear in the data
        # labels = data[refr_scorer].cat.remove_unused_categories().cat.categories
        labels = [l for l in refr_hyps[subjects[0]].hypno.cat.categories if l in data.values]
        ############ OPTION 1 (uses staticmethod, slower by 500ms)
        indiv_agree_ovr = (data
            # Get multiple metrics for each individual
            .groupby(level=0).apply(self.multi_scorer_ovr, labels=labels)
            # Unpack metrics results and reshape
            .apply(pd.Series).stack().apply(pd.Series)
            # Convert stages to string labels
            .rename_axis(columns="stage").rename(columns={i: l for i, l in enumerate(labels)})
            # Reshape so metrics are columns
            .stack().unstack(level=1)
            # Swap MultiIndex levels and sort so stages drive the view
            .swaplevel().sort_index(level="stage", key=lambda x: x.map(lambda y: labels.index(y)))
        )
        # ############ OPTION 2 (does NOT use staticmethod, faster by 500ms)
        # prfs_func = lambda df: skm.precision_recall_fscore_support(
        #     *df.values.T, labels=labels, average=None, zero_division=0
        # )
        # indiv_agree_ovr = (data
        #     .groupby(level=0).apply(prfs_func)
        #     .explode().apply(pd.Series)
        #     .assign(metric=["precision", "recall", "f1", "support"] * len(refr_hyps)).set_index("metric", append=True)
        #     .rename_axis(columns="stage").rename(columns={i: l for i, l in enumerate(labels)})
        #     .stack().unstack("metric").rename_axis(columns=None)
        # )
        ## Q: Currently both options will leave some all-zero rows, for when a stage is present
        ##    in some subjects but not others. Prefer to remove?
        # agr = agr.loc[agr.any(axis=1)]  # or .pipe
        # And then could drop the label restriction, just passing all labels to preserve order

        # ac_f = lambda s: skm.accuracy_score(*s.values.T)
        # ka_f = lambda s: skm.cohen_kappa_score(*s.values.T)
        # ja_f = lambda s: skm.jaccard_score(*s.values.T, average=average)
        # pr_f = lambda s: skm.precision_score(*s.values.T, average=average, zero_division=zd)
        # re_f = lambda s: skm.recall_score(*s.values.T, average=average, zero_division=zd)
        # f1_f = lambda s: skm.f1_score(*s.values.T, average=average, zero_division=zd)
        # ac = data.groupby(level=0).apply(ac_f)
        # ka = data.groupby(level=0).apply(ka_f)
        # ja = data.groupby(level=0).apply(ja_f)
        # pr = data.groupby(level=0).apply(pr_f)
        # re = data.groupby(level=0).apply(re_f)
        # f1 = data.groupby(level=0).apply(f1_f)
        # agreement_individuals = pd.DataFrame(
        #     {"accuracy": ac, "kappa": ka, "jaccard": ja, "precision": pr, "recall": re, "f1": f1}
        # )

        # Set attributes
        self._data = data
        self._subjects = subjects
        self._n_subjects = len(subjects)
        self._refr_hyps = refr_hyps
        self._test_hyps = test_hyps
        self._refr_scorer = refr_hyps[subjects[0]].scorer
        self._test_scorer = test_hyps[subjects[0]].scorer
        self._labels = refr_hyps[subjects[0]].labels

        self._indiv_agree_avg = indiv_agree_avg
        self._indiv_agree_ovr = indiv_agree_ovr

    def __repr__(self):
        # TODO v0.8: Keep only the text between < and >
        text_subjects = f", {self.n_subjects} subject" + ("s" if self.n_subjects > 1 else "")
        return (
            f"<EpochByEpochEvaluation | Test Hypnogram scored by {self.refr_scorer} evaluated "
            f"against reference Hypnogram scored by {self.test_scorer}{text_subjects}>\n"
            " - Use `.get_agreement()` to get agreement measures as a pandas.Series\n"
            " - Use `.plot_hypnograms()` to plot the two hypnograms overlaid\n"
            "See the online documentation for more details."
        )

    def __str__(self):
        text_subjects = f", {self.n_subjects} subject" + ("s" if self.n_subjects > 1 else "")
        return (
            f"<EpochByEpochEvaluation | Test Hypnogram scored by {self.refr_scorer} evaluated "
            f"against reference Hypnogram scored by {self.test_scorer}{text_subjects}>\n"
            " - Use `.get_agreement()` to get agreement measures as a pandas.Series\n"
            " - Use `.plot_hypnograms()` to plot the two hypnograms overlaid\n"
            "See the online documentation for more details."
        )

    @property
    def data(self):
        """A :py:class:`pandas.DataFrame` including all hypnograms."""
        return self._data

    @property
    def refr_hyps(self):
        """The reference YASA hypnograms."""
        return self._refr_hyps

    @property
    def test_hyps(self):
        """The test YASA hypnograms."""
        return self._test_hyps

    @property
    def subjects(self):
        return self._subjects

    @property
    def n_subjects(self):
        return self._n_subjects

    @property
    def refr_scorer(self):
        """The name of the reference scorer."""
        return self._refr_scorer

    @property
    def test_scorer(self):
        """The name of the test scorer."""
        return self._test_scorer

    @property
    def labels(self):
        """All available sleep stage labels."""
        return self._labels

    @property
    def indiv_agree_avg(self):
        """
        A :py:class:`pandas.DataFrame` of individual-level ``refr_hyp``/``test_hyp`` agreement
        metrics that average over individual sleep stages.

        .. seealso:: :py:attr:`yasa.EpochByEvaluation.individual_agreement_ovr`
        
        .. seealso:: :py:attr:`yasa.EpochByEvaluation.group_agreement_avg`
        """
        return self._indiv_agree_avg

    @property
    def indiv_agree_ovr(self):
        """
        A :py:class:`pandas.DataFrame` of individual-level ``refr_hyp``/``test_hyp`` "one-vs-rest"
        agreement metrics. Metrics for each sleep stage are provided.

        .. seealso:: :py:attr:`yasa.EpochByEvaluation.individual_agreement_ovr`
        
        .. seealso:: :py:attr:`yasa.EpochByEvaluation.group_agreement_avg`
        """
        return self._indiv_agree_ovr

    @staticmethod
    def multi_scorer_avg(df):
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

        Returns
        -------
        scores : dict
            A dictionary with scorer names (``str``) as keys and scores (``float``) as values.
        """
        true, pred = zip(*df.values)  # Same as (df["col1"], df["col2"]) but teensy bit faster
        ## Q: The dictionary below be compiled more concisely if we were comfortable accessing
        ##    "private" attributes. I understand that's a no-no but I'm not exactly sure why.
        ##     For example:
        ##     >>> scorers = ["accuracy", "recall"]
        ##     >>> funcs = { s: skm.__getattribute__(f"{s}_scorer") for s in scorers }
        ##     >>> scores = { s: f(true, pred) for s, f in funcs.items() }
        ##     Keywords could be applied as needed by checking f.__kwdefaults__
        ##     This would offer an easy way for users to add their own scorers with an arg as well.
        return {
            "accuracy": skm.accuracy_score(true, pred),
            "kappa": skm.cohen_kappa_score(true, pred),
            "micro_jaccard": skm.jaccard_score(true, pred, average="micro"),
            "macro_jaccard": skm.jaccard_score(true, pred, average="macro"),
            "weighted_jaccard": skm.jaccard_score(true, pred, average="weighted"),
            "micro_precision": skm.precision_score(true, pred, average="micro", zero_division=0),
            "macro_precision": skm.precision_score(true, pred, average="macro", zero_division=0),
            "weighted_precision": skm.precision_score(true, pred, average="weighted", zero_division=0),
            "micro_recall": skm.recall_score(true, pred, average="micro", zero_division=0),
            "macro_recall": skm.recall_score(true, pred, average="macro", zero_division=0),
            "weighted_recall": skm.recall_score(true, pred, average="weighted", zero_division=0),
            "micro_f1": skm.f1_score(true, pred, average="micro", zero_division=0),
            "macro_f1": skm.f1_score(true, pred, average="macro", zero_division=0),
            "weighted_f1": skm.f1_score(true, pred, average="weighted", zero_division=0),
        }

    @staticmethod
    def multi_scorer_ovr(df, labels):
        """Compute multiple one-vs-rest agreement scores from a 2-column dataframe.

        Parameters
        ----------
        df : :py:class:`pandas.DataFrame`
            A :py:class:`pandas.DataFrame` with exactly 2 columns and length of *n_samples*.
            The first column contains true values and second column contains predicted values.
        labels : array-like
            The labels to include in scoring and control the order of returned scores.

        Returns
        -------
        scores : dict
            A dictionary with scorer names (``str``) as keys and scores (``np.ndarray``) as values.
        """
        true, pred = zip(*df.values)
        return {
            "precision": skm.precision_score(true, pred, labels=labels, average=None, zero_division=0),
            "recall": skm.recall_score(true, pred, labels=labels, average=None, zero_division=0),
            "f1": skm.f1_score(true, pred, labels=labels, average=None, zero_division=0),
            "support": pd.Series(true).value_counts().reindex(labels, fill_value=0).to_numpy(),
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
        agg_kwargs = {"func": ["mean", "std", "min", "median", "max"]} | kwargs
        if by_stage:
            summary = (self.indiv_agree_ovr
                .groupby("stage").agg(**agg_kwargs)
                .stack(0).rename_axis(["stage", "metric"])
            )
        else:
            summary = self.indiv_agree_avg.agg(**agg_kwargs).T.rename_axis("metric")
            ## Q: Should we include a column that calculates agreement treating all hypnograms as
            ##    coming from one individual? Others sometimes report it, though I find it mostly
            ##    meaningless because of possible n_epochs imbalances between subjects. I vote no.
            # summary.insert(0, "all", self.multi_scorer_avg(self.data))
        ## Q: Alternatively, we could remove the `by_stage` parameter and stack these into
        ##    one merged DataFrame where the results that are *not* by-stage are included
        ##    with an "all" stage label:
        # summary = summary.assign(stage="all").set_index("stage", append=True).swaplevel()
        # summary = pd.concat([summary, summary_ovr]).sort_index()
        return summary

    def get_sleep_stats(self):
        """Return a :py:class:`pandas.DataFrame` of sleep statistics for each individual derived
        from both Reference and Test measurement systems.

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
            individual (one from Reference measurement and another from Test measurement).
        """
        # refr_sstats = pd.Series(self.refr_hyps).map(lambda h: h.sleep_statistics()).apply(pd.Series)
        # test_sstats = pd.Series(self.test_hyps).map(lambda h: h.sleep_statistics()).apply(pd.Series)
        refr_sstats = pd.DataFrame({s: h.sleep_statistics() for s, h in self.refr_hyps.items()})
        test_sstats = pd.DataFrame({s: h.sleep_statistics() for s, h in self.test_hyps.items()})
        refr_sstats = pd.concat({self.refr_scorer: refr_sstats.T.rename_axis("subject")}, names=["measurement"])
        test_sstats = pd.concat({self.test_scorer: test_sstats.T.rename_axis("subject")}, names=["measurement"])
        sstats = pd.concat([refr_sstats, test_sstats])
        return sstats

    def get_confusion_matrix(self, subject=None):
        """Return a ``refr_hyp``/``test_hyp``confusion matrix.

        Parameters
        ----------
        self : :py:class:`yasa.EpochByEvaluation`
            A :py:class:`yasa.EpochByEvaluation` instance.
        subject : None or a valid individual identifier
            If None (default), cross-tabulation is derived from the entire group dataset.
            If a valid individual identifier, cross-tabulation is derived using only hypnograms
            from that individual.

        Returns
        -------
        matrix : :py:class:`pandas.DataFrame`
            A confusion matrix with ``refr_hyp`` stages as indices and ``test_hyp`` stages as columns.
        """
        assert subject is None or subject in self.subjects, "`subject` must be None or a valid subject ID"
        true = self.data[self.refr_scorer]
        pred = self.data[self.test_scorer]
        if subject is not None:
            true = true.loc[subject]
            pred = pred.loc[subject]
        # Generate confusion matrix.
        matrix = pd.crosstab(true, pred, margins=True, margins_name="Total")
        # Reorder indices in sensible order and to include all stages
        index_col_labels = self.labels + ["Total"]
        matrix = matrix.reindex(index=index_col_labels, columns=index_col_labels, fill_value=0)
        return matrix.astype(int)

    def plot_hypnograms(self, subject=None, legend=True, ax=None, refr_kwargs={}, test_kwargs={}):
        """Plot the two hypnograms, where ``refr_hyp`` is overlaid on ``refr_hyp``.

        .. seealso:: :py:func:`yasa.plot_hypnogram`

        Parameters
        ----------
        legend : bool or dict
            If True (default) or a dictionary, a legend is added. If a dictionary, all key/value
            pairs are passed as keyword arguments to the :py:func:`matplotlib.pyplot.legend` call.
        ax : :py:class:`matplotlib.axes.Axes` or None
            Axis on which to draw the plot, optional.
        refr_kwargs : dict
            Keyword arguments passed to :py:func:`yasa.plot_hypnogram` when plotting ``refr_hyp``.
        test_kwargs : dict
            Keyword arguments passed to :py:func:`yasa.plot_hypnogram` when plotting ``test_hyp``.

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
        assert subject is None or subject in self.subjects, "`subject` must be None or a valid subject ID"
        assert isinstance(legend, (bool, dict)), "`legend` must be True, False, or a dictionary"
        assert isinstance(refr_kwargs, dict), "`refr_kwargs` must be a dictionary"
        assert isinstance(test_kwargs, dict), "`test_kwargs` must be a dictionary"
        assert not "ax" in refr_kwargs | test_kwargs, (
            "ax can't be supplied to `kwargs_ref` or `test_kwargs`, use the `ax` keyword instead"
        )
        if subject is None:
            if self.n_subjects == 1:
                refr_hyp = self.refr_hyps[self.subjects[0]]
                test_hyp = self.test_hyps[self.subjects[0]]
            else:
                raise NotImplementedError("Plotting is currently allowed for only one subject")
        else:
            refr_hyp = self.refr_hyps[subject]
            test_hyp = self.test_hyps[subject]
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

    def plot_roc(self, subject=None, palette=None, ax=None, **kwargs):
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
        assert subject is None or subject in self.subjects, "`subject` must be None or a valid subject ID"
        # assert self.test_hyp.probas is not None
        raise NotImplementedError("Requires probability/confidence values.")


#############################################################################
# SLEEP STATISTICS
#############################################################################


class SleepStatsEvaluation:
    """
    Evaluate agreement between two measurement systems (e.g., two different manual scorers or one
    one manual scorer againt YASA's automatic staging) by comparing their summary sleep statistics
    derived from multiple subjects or sessions.

    Parameters
    ----------
    refr_data : :py:class:`pandas.DataFrame`
        A :py:class:`pandas.DataFrame` with sleep statistics from the reference measurement system.
        Rows are individual subjects and columns are individual sleep statistics.
    test_data : :py:class:`pandas.DataFrame`
        A :py:class:`pandas.DataFrame` with sleep statistics from the test measurement system.
        Shape, indices, and columns must be identical to ``refr_data``.
    refr_scorer : str
        Name of the reference measurement scorer, used for labeling.
    test_scorer : str
        Name of the test measurement scorer, used for labeling.
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
        assert np.array_equal(refr_data.index, test_data.index), "`refr_data` and `test_data` indices must be identical"
        assert np.array_equal(refr_data.columns, test_data.columns), "`refr_data` and `test_data` columns must be identical"
        assert refr_data.index.name == test_data.index.name, "`refr_data` and `test_data` index names must be identical"
        assert isinstance(refr_scorer, str)
        assert isinstance(test_scorer, str)
        assert refr_scorer != test_scorer
        assert isinstance(kwargs_normality, dict)
        assert isinstance(kwargs_regression, dict)
        assert isinstance(kwargs_homoscedasticity, dict)
        assert "alpha" in kwargs_normality
        assert "alpha" in kwargs_regression
        assert "alpha" in kwargs_homoscedasticity

        # Merge dataframes, get differences, and reshape wide-to-long format
        subj_name = "subject" if refr_data.index.name is None else refr_data.index.name
        refr_data.index.name = subj_name
        test_data.index.name = subj_name
        diff_data = pd.concat({"difference": test_data.sub(refr_data)}, names=["measurement"])
        refr_data = pd.concat({refr_scorer: refr_data}, names=["measurement"])
        test_data = pd.concat({test_scorer: test_data}, names=["measurement"])
        data = (pd.concat([refr_data, test_data, diff_data])
            .melt(var_name="sstat", ignore_index=False).reset_index()
            .pivot(columns="measurement", index=[subj_name, "sstat"], values="value")
            .reset_index().rename_axis(columns=None)
        )

        # Remove sleep statistics that have no differences between measurement systems
        ## TODO: restructure?
        stats_nodiff = data.groupby("sstat")["difference"].any().loc[lambda x: ~x].index.tolist()
        data = data.query(f"~sstat.isin({stats_nodiff})")
        for s in stats_nodiff:
            logger.warning(f"All {s} differences are zero, removing from evaluation.")
            ## Q: Should this be logged as just info?

        ## NORMALITY ## Test reference data for normality at each sleep statistic
        normality = data.groupby("sstat")[refr_scorer].apply(pg.normality, **kwargs_normality).droplevel(-1)

        ## PROPORTIONAL BIAS ## Test each sleep statistic for proportional bias
        # Subject-level residuals for each statistic are added to data.
        prop_bias_results = []
        residuals_results = []
        # proportional bias and residuals that will be used for the later  tests.
        for ss_name, ss_df in data.groupby("sstat"):
            # Regress the difference scores on the reference scores
            model = pg.linear_regression(ss_df[refr_scorer], ss_df["difference"], **kwargs_regression)
            model.insert(0, "sstat", ss_name)
            # Extract subject-level residuals for later homoscedasticity tests
            resid_dict = {subj_name: ss_df[subj_name], "sstat": ss_name, "pbias_residual": model.residuals_}
            resid = pd.DataFrame(resid_dict)
            prop_bias_results.append(model)
            residuals_results.append(resid)
        # Add residuals to raw dataframe, used later when testing homoscedasticity
        data = data.merge(pd.concat(residuals_results), on=[subj_name, "sstat"])
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
        homoscedasticity_func = lambda df: pg.homoscedasticity(df[columns], **kwargs_homoscedasticity)
        homoscedasticity = data.groupby("sstat").apply(homoscedasticity_func).droplevel(-1)

        # Set attributes
        self._data = data
        self._normality = normality
        self._proportional_bias = prop_bias
        self._proportional_bias_full = prop_bias_full  # Q: Is this worth saving??
        self._homoscedasticity = homoscedasticity
        # These will not be set as properties, as they are only needed internally
        self._refr_scorer = refr_scorer
        self._test_scorer = test_scorer
        self._subj_name = subj_name
        self._n_subjects = data[subj_name].nunique()
        # Pivot new to not include removed sstats
        self._diff_data = data.pivot(index=self.subj_name, columns="sstat", values="difference")

    @property
    def data(self):
        """
        ``refr_data`` and ``test_data`` combined in a long-format :py:class:`pandas.DataFrame`.
        Also includes difference scores (``test_data`` minus ``refr_data``).
        """
        return self._data

    @property
    def diff_data(self):
        """A :py:class:`pandas.DataFrame` of ``test_data`` minus ``refr_data``."""
        # # Pivot for subject-rows and statistic-columns
        return self._diff_data

    @property
    def refr_scorer(self):
        """The name of the reference measurement scorer."""
        return self._refr_scorer

    @property
    def test_scorer(self):
        """The name of the test measurement scorer."""
        return self._test_scorer

    @property
    def subj_name(self):
        """The name of the subject identifier."""
        return self._subj_name

    @property
    def n_subjects(self):
        """The number of subjects."""
        return self._n_subjects

    @property
    def normality(self):
        """A :py:class:`pandas.DataFrame` of normality test results for all sleep statistics."""
        return self._normality

    @property
    def homoscedasticity(self):
        """A :py:class:`pandas.DataFrame` of homoscedasticity test results for all sleep statistics."""
        return self._homoscedasticity

    @property
    def proportional_bias(self):
        """A :py:class:`pandas.DataFrame` of proportional bias test results for all sleep statistics."""
        return self._proportional_bias

    @property
    def proportional_bias_full(self):
        """A :py:class:`pandas.DataFrame` of proportional bias test results for all sleep statistics."""
        return self._proportional_bias_full

    def __repr__(self):
        # TODO v0.8: Keep only the text between < and >
        return (
            f"<SleepStatsEvaluation | Test measurement '{self.test_scorer}' evaluated against "
            f"reference measurement '{self.refr_scorer}', {self.n_subjects} subjects>\n"
            " - Use `.summary()` to get pass/fail values from various checks\n"
            " - Use `.plot_blandaltman()` to get a Bland-Altman-plot grid for sleep statistics\n"
            "See the online documentation for more details."
        )

    def __str__(self):
        return (
            f"<SleepStatsEvaluation | Test measurement '{self.test_scorer}' evaluated against "
            f"reference measurement '{self.refr_scorer}', {self.n_subjects} subjects>\n"
            " - Use `.summary()` to get pass/fail values from various checks\n"
            " - Use `.plot_blandaltman()` to get a Bland-Altman-plot grid for sleep statistics\n"
            "See the online documentation for more details."
        )

    def summary(self, descriptives=True):
        """Return a summary dataframe highlighting what statistics pass checks.

        Parameters
        ----------
        self : :py:class:`SleepStatsEvaluation`
            A :py:class:`SleepStatsEvaluation` instance.
        descriptives : bool or dict
            If True (default) or a dictionary, also include descriptive statistics for reference and
            test measurements. If a dictionary, all key/value pairs are passed as keyword arguments
            to the :py:meth:`pandas.DataFrame.agg` call.

        Returns
        -------
        summary : :py:class:`pandas.DataFrame`
            A :py:class:`pandas.DataFrame` with boolean values indicating the pass/fail status for
            normality, proportional bias, and homoscedasticity tests (for each sleep statistic).
        """
        assert isinstance(descriptives, (bool, dict)), "`descriptives` must be True, False, or dict"
        series_list = [
            self.normality["normal"],
            self.proportional_bias["unbiased"],
            self.homoscedasticity["equal_var"].rename("homoscedastic"),
        ]
        summary = pd.concat(series_list, axis=1)
        if descriptives:
            agg_kwargs = {"func": ["mean", "std"]}
            if isinstance(descriptives, dict):
                agg_kwargs.update(descriptives)
            desc = self.data.drop(columns=self.subj_name).groupby("sstat").agg(**agg_kwargs)
            desc.columns = desc.columns.map("_".join)
            summary = summary.join(desc)
        return summary

    def plot_discrepancies_heatmap(self, sleep_stats=None, **kwargs):
        """Visualize subject-level discrepancies, generally for outlier inspection.

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
        """Visualize subject-level discrepancies, generally for outlier inspection.

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
        pairgrid_kwargs = dict(hue=self.subj_name, height=height, aspect=aspect)
        pairgrid_kwargs.update(kwargs_pairgrid)
        g = sns.PairGrid(self.diff_data.reset_index(), y_vars=[self.subj_name], **pairgrid_kwargs)
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
