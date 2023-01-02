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
from sklearn import metrics

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
        The reference or ground-truth hypnogram.
    test_hyp : :py:class:`yasa.Hypnogram`
        The test or to-be-evaluated hypnogram.

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
    def __init__(self, refr_hyp, test_hyp):
        from yasa.hypno import Hypnogram  # Loading here to avoid circular import
        assert isinstance(refr_hyp, Hypnogram), "`refr_hyp` must be a YASA Hypnogram"
        assert isinstance(test_hyp, Hypnogram), "`test_hyp` must be a YASA Hypnogram"
        assert refr_hyp.scorer is not None, "`refr_hyp` must have a scorer label"
        assert test_hyp.scorer is not None, "`test_hyp` must have a scorer label"
        assert refr_hyp.scorer != test_hyp.scorer, (
            "scorer must be unique for `refr_hyp` and `test_hyp`"
        )
        assert refr_hyp.n_stages == test_hyp.n_stages, (
            "`refr_hyp` and `test_hyp` must have the same `n_stages`"
        )
        assert refr_hyp.labels == test_hyp.labels
        assert refr_hyp.mapping == test_hyp.mapping
        if (n_ref := refr_hyp.n_epochs) != (n_test := test_hyp.n_epochs):
            ## NOTE: would be nice to have a Hypnogram.trim() method for moments like this.
            if n_ref > n_test:
                refr_hyp = Hypnogram(refr_hyp.hypno[:n_test], n_stages=refr_hyp.n_stages)
                n_trimmed = n_ref - n_test
                warn_msg = f"`refr_hyp` longer than `test_hyp`, trimmed to {n_test} epochs"
            else:
                test_hyp = Hypnogram(test_hyp.hypno[:n_ref], n_stages=test_hyp.n_stages)
                n_trimmed = n_test - n_ref
                warn_msg = f"`test_hyp` longer than `refr_hyp`, {n_trimmed} epochs trimmed"
            ## Q: Should be downplayed as INFO?
            logger.warning(warn_msg)
        
        # Set attributes
        self._refr_hyp = refr_hyp.copy()
        self._test_hyp = test_hyp.copy()

    def __repr__(self):
        # TODO v0.8: Keep only the text between < and >
        return (
            f"<EpochByEpochEvaluation | Test Hypnogram scored by {self.refr_hyp.scorer} evaluated "
            f"against reference Hypnogram scored by {self.test_hyp.scorer}>\n"
            " - Use `.get_agreement()` to get agreement measures as a pandas.Series\n"
            " - Use `.plot_hypnograms()` to plot the two hypnograms overlaid\n"
            "See the online documentation for more details."
        )

    def __str__(self):
        return (
            f"<EpochByEpochEvaluation | Test Hypnogram scored by {self.refr_hyp.scorer} evaluated "
            f"against reference Hypnogram scored by {self.test_hyp.scorer}>\n"
            " - Use `.get_agreement()` to get agreement measures as a pandas.Series\n"
            " - Use `.plot_hypnograms()` to plot the two hypnograms overlaid\n"
            "See the online documentation for more details."
        )

    @property
    def refr_hyp(self):
        """The reference Hypnogram."""
        ## Q: Starting to think there should be a clear convention on what we mean
        ##    when we say "hypnogram". Should hypnogram mean the Series and Hypnogram
        ##    mean the YASA object? Similarly for hypno/hyp.
        return self._refr_hyp

    @property
    def test_hyp(self):
        """The test Hypnogram."""
        return self._test_hyp

    def get_agreement(self):
        """
        Return a dataframe of ``refr_hyp``/``test_hyp`` performance across all stages as measured by
        common classifier agreement methods.

        .. seealso:: :py:meth:`yasa.EpochByEpochResults.get_agreement_by_stage`
        ## Q: Are there better names to differentiate get_agreement vs get_agreement_by_stage?
        ##    Maybe should be binary vs multiclass?

        Returns
        -------
        agreement : :py:class:`pandas.Series`
            A :py:class:`pandas.Series` with agreement metrics as indices.
        """
        true = self.refr_hyp.hypno.to_numpy()
        pred = self.test_hyp.hypno.to_numpy()
        accuracy = metrics.accuracy_score(true, pred)
        kappa = metrics.cohen_kappa_score(true, pred)
        jaccard = metrics.jaccard_score(true, pred, average="weighted")
        precision = metrics.precision_score(true, pred, average="weighted", zero_division=0)
        recall = metrics.recall_score(true, pred, average="weighted", zero_division=0)
        f1 = metrics.f1_score(true, pred, average="weighted", zero_division=0)
        scores = {
            "accuracy": accuracy,
            "kappa": kappa,
            "weighted_jaccard": jaccard,
            "weighted_precision": precision,
            "weighted_recall": recall,
            "weighted_f1": f1,
        }
        agreement = pd.Series(scores, name="agreement").rename_axis("metric")
        return agreement

    def get_agreement_by_stage(self):
        """
        Return a dataframe of ``refr_hyp``/``test_hyp`` performance for each stage as measured by
        common classifier agreement methods.

        .. seealso:: :py:meth:`yasa.EpochByEpochResults.get_agreement`

        Returns
        -------
        agreement : :py:class:`pandas.DataFrame`
            A DataFrame with agreement metrics as indices and stages as columns.
        """
        true = self.refr_hyp.hypno.to_numpy()
        pred = self.test_hyp.hypno.to_numpy()
        labels = self.test_hyp.labels  # Same as refr_hyp.labels
        scores = metrics.precision_recall_fscore_support(
            true, pred, labels=labels, average=None, zero_division=0
        )
        agreement = pd.DataFrame(scores)
        agreement.index = pd.Index(["precision", "recall", "fscore", "support"], name="metric")
        agreement.columns = pd.Index(labels, name="stage")
        return agreement

    def get_confusion_matrix(self):
        """Return a ``refr_hyp``/``test_hyp``confusion matrix.

        Returns
        -------
        matrix : :py:class:`pandas.DataFrame`
            A confusion matrix with ``refr_hyp`` stages as indices and ``test_hyp`` stages as columns.
        """
        # Generate confusion matrix.
        matrix = pd.crosstab(
            self.refr_hyp.hypno, self.test_hyp.hypno, margins=True, margins_name="Total"
        )
        # Reorder indices in sensible order and to include all stages
        matrix = matrix.reindex(labels=self.refr_hyp.labels + ["Total"], fill_value=0)
        matrix = matrix.reindex(columns=self.test_hyp.labels + ["Total"], fill_value=0)
        return matrix.astype(int)

    def plot_hypnograms(self, legend=True, ax=None, refr_kwargs={}, test_kwargs={}):
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
        assert isinstance(legend, (bool, dict)), "`legend` must be True, False, or a dictionary"
        assert isinstance(refr_kwargs, dict), "`refr_kwargs` must be a dictionary"
        assert isinstance(test_kwargs, dict), "`test_kwargs` must be a dictionary"
        assert not "ax" in refr_kwargs | test_kwargs, (
            "ax can't be supplied to `kwargs_ref` or `test_kwargs`, use the `ax` keyword instead"
        )
        plot_refr_kwargs = {"highlight": None, "alpha": 0.8}
        plot_test_kwargs = {"highlight": None, "alpha": 0.8, "color": "darkcyan", "ls": "dashed"}
        plot_refr_kwargs.update(refr_kwargs)
        plot_test_kwargs.update(test_kwargs)
        if ax is None:
            ax = plt.gca()
        self.refr_hyp.plot_hypnogram(ax=ax, **plot_refr_kwargs)
        self.test_hyp.plot_hypnogram(ax=ax, **plot_test_kwargs)
        if legend and "label" in plot_refr_kwargs | plot_test_kwargs:
            if isinstance(legend, dict):
                ax.legend(**legend)
            else:
                ax.legend()
        return ax

    def plot_roc(self, palette=None, ax=None, **kwargs):
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
    data : :py:class:`pandas.DataFrame`
        A :py:class:`pandas.DataFrame` with sleep statistics from two different measurement systems.
        Each row contains the two different measurements of a single subject and sleep statistic.
        Of shape (n_subjects x n_sleep_statistics, 4).
    reference : str
        Name of column containing the reference device sleep statistics.
    test : str
        Name of column containing the test device sleep statistics.
    subject : str
        Name of column containing the subject ID.
    statistic : str
        Name of column containing the name of the sleep statistic.

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
    >>> # For this example, generate a fake dataset of sleep statistics from two different raters
    >>> data = []
    >>> for i in range(1, 21):
    >>>     hypA = yasa.simulate_hypnogram(tib=600, seed=i)
    >>>     hypB = hypA.simulate_similar(seed=i)
    >>>     data.append({"subject": f"sub-{i:03d}", "rater": "RaterA"} | hypA.sleep_statistics())
    >>>     data.append({"subject": f"sub-{i:03d}", "rater": "RaterB"} | hypB.sleep_statistics())
    >>> df = (pd.json_normalize(data)
    >>>     .melt(id_vars=["subject", "rater"], var_name="sstat", value_name="score")
    >>>     .pivot(index=["subject", "sstat"], columns="rater", values="score")
    >>>     .reset_index().rename_axis(None, axis=1)
    >>>     .query("sstat.isin(['SE', 'TST', 'SOL', 'WASO', '%N1', '%N2', '%N3', '%REM'])")
    >>> )
    >>>
    >>> sse = yasa.SleepStatsEvaluation(
    >>>     data=df, reference="RaterA", test="RaterB", subject="subject", statistic="sstat"
    >>> )
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
    def __init__(self, data, *, reference, test, subject, statistic):
        assert isinstance(data, pd.DataFrame), "`data` must be a pandas DataFrame"
        for col in [reference, test, subject, statistic]:
            assert isinstance(col, str) and col in data, (
                f"`{col}` must be a string and a column in `data`"
            )
        assert data[subject].nunique() > 1, "`data` must include more than one subject"
        assert not data.groupby("subject")["sstat"].count().diff().any(), "same number of sstats for all subjects"
        assert not data.groupby("subject")["sstat"].nunique().is_unique, "no repeated sstats per subject"
        
        # Don't update this, rename to something else like table.
        data = data.copy()

        # Get measurement difference between reference and test devices
        data["difference"] = data[test].sub(data[reference])

        # Remove sleep statistics that have no differences between measurement systems.
        ## TODO: simplify once not manipulating _data
        stats_nodiff = data.groupby(statistic)["difference"].any().loc[lambda x: ~x].index
        for s in stats_nodiff:
            data = data.query(f"{statistic} != '{s}'")
            logger.warning(f"All {s} differences are zero, removing from evaluation.")
            ## Q: Should this be logged as just info?

        # Get list of all statistics to be evaluated
        self.all_sleepstats = data[statistic].unique()

        # Set attributes
        self._data = data
        self._reference = reference
        self._test = test
        self._subject = subject
        self._statistic = statistic

        # Run tests
        self.test_normality(method="shapiro", alpha=0.05)
        self.test_proportional_bias(alpha=0.05)
        self.test_homoscedasticity(method="levene", alpha=0.05)

    @property
    def data(self):
        """The summary dataframe of sleep statistics."""
        return self._data

    @property
    def reference(self):
        """The name of the column containing the reference measurement sleep statistics."""
        return self._reference

    @property
    def test(self):
        """The name of the column containing the test measurement sleep statistics."""
        return self._test

    @property
    def subject(self):
        """The name of the column containing the subject identifiers."""
        return self._subject

    @property
    def statistic(self):
        """The name of the column containing the sleep statistic name."""
        return self._statistic

    def __repr__(self):
        # TODO v0.8: Keep only the text between < and >
        return (
            f"<SleepStatsEvaluation | Test measurement '{self.test}' evaluated against reference "
            f"measurement '{self.reference}'>\n"
            " - Use `.summary()` to get pass/fail values from various checks\n"
            " - Use `.plot_blandaltman()` to get a Bland-Altman-plot grid for sleep statistics\n"
            "See the online documentation for more details."
        )

    def __str__(self):
        return (
            f"<SleepStatsEvaluation | Test measurement '{self.test}' evaluated against reference "
            f"measurement '{self.reference}'>\n"
            " - Use `.summary()` to get pass/fail values from various checks\n"
            " - Use `.plot_blandaltman()` to get a Bland-Altman-plot grid for sleep statistics\n"
            "See the online documentation for more details."
        )

    def test_normality(self, **kwargs):
        """Test reference data for normality at each sleep statistic.

        Parameters
        ----------
        **kwargs : key, value pairs
            Additional keyword arguments are passed to the :py:func:`pingouin.normality` call.
        """
        normality = self.data.groupby(self.statistic)[self.reference].apply(pg.normality, **kwargs)
        self.normality = normality.droplevel(-1)

    def test_proportional_bias(self, **kwargs):
        """Test each sleep statistic for proportional bias.
        
        For each statistic, regress the device difference score on the reference device score to get
        proportional bias and residuals that will be used for the later homoscedasticity
        calculation. Subject-level residuals for each statistic are added to ``data``.

        Parameters
        ----------
        **kwargs : key, value pairs
            Additional keyword arguments are passed to :py:func:`pingouin.linear_regression`.
        """
        if "alpha" not in kwargs:
            kwargs["alpha"] = 0.05
        prop_bias_results = []
        residuals_results = []
        for ss, ss_df in self.data.groupby(self.statistic):
            # Regress the difference score on the reference device
            model = pg.linear_regression(ss_df[self.reference], ss_df["difference"], **kwargs)
            model.insert(0, self.statistic, ss)
            # Extract the subject-level residuals
            resid = pd.DataFrame(
                {
                    self.subject: ss_df[self.subject],
                    self.statistic: ss,
                    "pbias_residual": model.residuals_
                }
            )
            prop_bias_results.append(model)
            residuals_results.append(resid)
        # Add residuals to raw dataframe, used later when testing homoscedasticity
        residuals = pd.concat(residuals_results)
        self.data = self.data.merge(residuals, on=[self.subject, self.statistic])
        # Handle proportional bias results
        prop_bias = pd.concat(prop_bias_results)
        # Save all the proportional bias models before removing intercept, for optional user access
        self.proportional_bias_models_ = prop_bias.reset_index(drop=True)
        # Remove intercept rows
        prop_bias = prop_bias.query("names != 'Intercept'").drop(columns="names")
        # Add True/False passing column for easy access
        prop_bias["unbiased"] = prop_bias["pval"].ge(kwargs["alpha"])
        self.proportional_bias = prop_bias.set_index(self.statistic)

    def test_homoscedasticity(self, **kwargs):
        """Test each statistic for homoscedasticity.

        Parameters
        ----------
        **kwargs : key, value pairs
            Additional keyword arguments are passed to :py:func:`pingouin.homoscedasticity`.

        ..note:: :py:meth:`yasa.SleepStatsEvaluation.test_proportional_bias` must be called first.
        """
        group = self.data.groupby(self.statistic)
        columns = [self.reference, "difference", "pbias_residual"]
        homoscedasticity = group.apply(lambda df: pg.homoscedasticity(df[columns], **kwargs))
        self.homoscedasticity = homoscedasticity.droplevel(-1)

    def summary(self, descriptives=True):
        """Return a summary dataframe highlighting what statistics pass checks."""
        assert isinstance(descriptives, bool), "`descriptives` must be True or False"
        series_list = [
            self.normality["normal"],
            self.proportional_bias["unbiased"],
            self.homoscedasticity["equal_var"].rename("homoscedastic"),
        ]
        summary = pd.concat(series_list, axis=1)
        if descriptives:
            group = self.data.drop(columns=self.subject).groupby(self.statistic)
            desc = group.agg(["mean", "std"])
            desc.columns = desc.columns.map("_".join)
            summary = summary.join(desc)
        return summary

    def plot_discrepancies_heatmap(self, sstats_order=None, **kwargs):
        """Visualize subject-level discrepancies, generally for outlier inspection.

        Parameters
        ----------
        sstats_order : list
            List of sleep statistics to plot. Default (None) is to plot all sleep statistics.
        **kwargs : key, value pairs
            Additional keyword arguments are passed to the :py:func:`seaborn.heatmap` call.

        Returns
        -------
        ax : :py:class:`matplotlib.axes.Axes`
            Matplotlib Axes
        """
        assert isinstance(sstats_order, (list, type(None))), "`sstats_order` must be a list or None"
        if sstats_order is None:
            sstats_order = self.all_sleepstats

        # Merge default heatmap arguments with optional input
        heatmap_kwargs = {"cmap": "binary", "annot": True, "fmt": ".1f", "square": False}
        heatmap_kwargs["cbar_kws"] = dict(label="Normalized discrepancy %")
        if "cbar_kws" in kwargs:
            heatmap_kwargs["cbar_kws"].update(kwargs["cbar_kws"])
        heatmap_kwargs.update(kwargs)
        # Pivot for subject-rows and statistic-columns
        table = self.data.pivot(index=self.subject, columns=self.statistic, values="difference")
        # Normalize statistics (i.e., columns) between zero and one then convert to percentage
        table_norm = table.sub(table.min(), axis=1).div(table.apply(np.ptp)).multiply(100)
        # If annotating, replace with raw values for writing.
        if heatmap_kwargs["annot"]:
            heatmap_kwargs["annot"] = table[sstats_order].to_numpy()
        # Draw heatmap
        ax = sns.heatmap(table_norm[sstats_order], **heatmap_kwargs)
        return ax

    def plot_discrepancies_dotplot(self, sstats_order=None, palette="winter", **kwargs):
        """Visualize subject-level discrepancies, generally for outlier inspection.

        Parameters
        ----------
        sstats_order : list
            List of sleep statistics to plot. Default (None) is to plot all sleep statistics.
        palette : string, list, dict, or :py:class:`matplotlib.colors.Colormap`
            Color palette passed to :py:class:`seaborn.PairGrid`
        **kwargs : key, value pairs
            Additional keyword arguments are passed to the :py:func:`seaborn.stripplot` call.

        Returns
        -------
        g : :py:class:`seaborn.PairGrid`
            Seaborn PairGrid
        """
        assert isinstance(sstats_order, (list, type(None))), "`sstats_order` must be a list or None"
        if sstats_order is None:
            sstats_order = self.all_sleepstats

        # Merge default stripplot arguments with optional input
        stripplot_kwargs = {"size": 10, "linewidth": 1, "edgecolor": "white"}
        stripplot_kwargs.update(kwargs)

        # Pivot data to get subject-rows and statistic-columns
        table = self.data.pivot(index=self.subject, columns=self.statistic, values="difference")

        # Initialize the PairGrid
        height = 0.3 * len(table)
        aspect = 0.6
        g = sns.PairGrid(
            table.reset_index(),
            x_vars=sstats_order,
            y_vars=[self.subject],
            hue=self.subject,
            palette=palette,
            height=height,
            aspect=aspect,
        )
        # Draw the dots
        g.map(sns.stripplot, orient="h", jitter=False, **stripplot_kwargs)

        # Adjust aesthetics
        g.set(xlabel="", ylabel="")
        for ax, title in zip(g.axes.flat, sstats_order):
            ax.set(title=title)
            ax.margins(x=0.3)
            ax.yaxis.grid(True)
            ax.tick_params(left=False)
        sns.despine(left=True, bottom=True)
        return g

    def plot_blandaltman(self, sstats_order=None, facet_kwargs={}, **kwargs):
        """
        Parameters
        ----------
        sstats_order : list or None
            List of sleep statistics to plot. Default (None) is to plot all sleep statistics.
        facet_kwargs : dict
            Keyword arguments passed to :py:class:`seaborn.FacetGrid`.
        **kwargs : key, value pairs
            Additional keyword arguments are passed to :py:func:`pingouin.plot_blandaltman`.

        Returns
        -------
        g : :py:class:`seaborn.FacetGrid`
            Seaborn FacetGrid
        """
        assert isinstance(sstats_order, (list, type(None))), "`sstats_order` must be a list or None"
        if sstats_order is None:
            sstats_order = self.all_sleepstats

        # Select scatterplot arguments (passed to blandaltman) and update with optional input
        blandaltman_kwargs = dict(xaxis="y", annotate=False, edgecolor="black", facecolor="none")
        blandaltman_kwargs.update(kwargs)
        # Select FacetGrid arguments and update with optional input
        col_wrap = 4 if len(sstats_order) > 4 else None
        facetgrid_kwargs = dict(col_wrap=col_wrap, height=2, aspect=1, sharex=False, sharey=False)
        facetgrid_kwargs.update(facet_kwargs)

        # Initialize a grid of plots with an Axes for each sleep statistic
        g = sns.FacetGrid(self.data, col=self.statistic, col_order=sstats_order, **facetgrid_kwargs)
        # Draw Bland-Altman on each axis
        g.map(pg.plot_blandaltman, self.test, self.reference, **blandaltman_kwargs)

        # Tidy-up axis limits with symmetric y-axis and minimal ticks
        for ax in g.axes.flat:
            bound = max(map(abs, ax.get_ylim()))
            ax.set_ylim(-bound, bound)
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=2, integer=True, symmetric=True))
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=1, integer=True))
        # More aesthetics
        ylabel = " - ".join((self.test, self.reference))
        g.set_ylabels(ylabel)
        g.set_titles(col_template="{col_name}")
        g.tight_layout(w_pad=1, h_pad=2)
        return g
