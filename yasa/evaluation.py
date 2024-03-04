"""
YASA code for evaluating the agreement between two scorers (e.g., human vs YASA), either at the
epoch-by-epoch level or at the level of summary sleep statistics.

Analyses are influenced by the standardized framework proposed in Menghini et al., 2021, SLEEP.
See the following resources:
- https://doi.org/10.1093/sleep/zsaa170
- https://sri-human-sleep.github.io/sleep-trackers-performance
- https://github.com/SRI-human-sleep/sleep-trackers-performance
"""

import logging

import numpy as np
import pandas as pd
import scipy.stats as sps
import sklearn.metrics as skm


logger = logging.getLogger("yasa")

__all__ = [
    "EpochByEpochAgreement",
    "SleepStatsAgreement",
]


################################################################################
# EPOCH BY EPOCH
################################################################################


class EpochByEpochAgreement:
    """Evaluate agreement between two hypnograms or two collections of hypnograms.

    Evaluation includes averaged agreement scores, one-vs-rest agreement scores, agreement scores
    summarized across all sleep and summarized by sleep stage, and various plotting options to
    visualize the two hypnograms simultaneously. See examples for more detail.

    .. versionadded:: 0.7.0

    Parameters
    ----------
    ref_hyps : iterable of :py:class:`yasa.Hypnogram`
        A collection of reference hypnograms (i.e., those considered ground-truth).

        Each :py:class:`yasa.Hypnogram` in ``ref_hyps`` must have the same
        :py:attr:`~yasa.Hypnogram.scorer`.

        If a ``dict``, key values are use to generate unique sleep session IDs. If any other
        iterable (e.g., ``list`` or ``tuple``), then unique sleep session IDs are automatically
        generated.
    obs_hyps : iterable of :py:class:`yasa.Hypnogram`
        A collection of observed hypnograms (i.e., those to be evaluated).

        Each :py:class:`yasa.Hypnogram` in ``obs_hyps`` must have the same
        :py:attr:`~yasa.Hypnogram.scorer`, and this scorer must be different than the scorer of
        hypnograms in ``ref_hyps``.

        If a ``dict``, key values must match those of ``ref_hyps``.

    .. important::
        It is assumed that the order of hypnograms are the same in ``ref_hyps`` and ``obs_hyps``.
        For example, the third hypnogram in ``ref_hyps`` and ``obs_hyps`` must come from the same
        sleep session, and they must only differ in that they have different scorers.

    .. seealso:: For comparing just two hypnograms, use :py:meth:`yasa.Hynogram.evaluate`.

    Notes
    -----
    Many steps here are influenced by guidelines proposed in Menghini et al., 2021 [Menghini2021]_.
    See https://sri-human-sleep.github.io/sleep-trackers-performance/AnalyticalPipeline_v1.0.0.html

    References
    ----------
    .. [Menghini2021] Menghini, L., Cellini, N., Goldstone, A., Baker, F. C., & de Zambotti, M.
                      (2021). A standardized framework for testing the performance of sleep-tracking
                      technology: step-by-step guidelines and open-source code. SLEEP, 44(2),
                      zsaa170. https://doi.org/10.1093/sleep/zsaa170

    Examples
    --------
    >>> import yasa
    >>> ref_hyps = [yasa.simulate_hypnogram(tib=600, scorer="Human", seed=i) for i in range(10)]
    >>> obs_hyps = [h.simulate_similar(scorer="YASA", seed=i) for i, h in enumerate(ref_hyps)]
    >>> ebe = yasa.EpochByEpochAgreement(ref_hyps, obs_hyps)
    >>> agr = ebe.get_agreement()
    >>> agr.head(5).round(2)
              accuracy  balanced_acc  kappa   mcc  precision  recall     f1
    sleep_id
    1             0.31          0.26   0.07  0.07       0.31    0.31   0.31
    2             0.33          0.33   0.14  0.14       0.35    0.33   0.34
    3             0.35          0.24   0.06  0.06       0.35    0.35   0.35
    4             0.22          0.21   0.01  0.01       0.21    0.22   0.21
    5             0.21          0.17  -0.06 -0.06       0.20    0.21   0.21

    >>> ebe.get_agreement_bystage().head(12).round(3)
                    fbeta  precision  recall  support
    stage sleep_id
    WAKE  1         0.391      0.371   0.413    189.0
          2         0.299      0.276   0.326    184.0
          3         0.234      0.204   0.275    255.0
          4         0.268      0.285   0.252    321.0
          5         0.228      0.230   0.227    181.0
          6         0.407      0.384   0.433    284.0
          7         0.362      0.296   0.467    287.0
          8         0.298      0.519   0.209    263.0
          9         0.210      0.191   0.233    313.0
          10        0.369      0.420   0.329    362.0
    N1    1         0.185      0.185   0.185    124.0
          2         0.121      0.131   0.112    160.0

    >>> ebe.get_confusion_matrix(sleep_id=1)
    YASA   WAKE  N1   N2  N3  REM
    Human
    WAKE     78  24   50   3   34
    N1       23  23   43  15   20
    N2       60  58  183  43  139
    N3       30  10   50   5   32
    REM      19   9  121  50   78

    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
        >>> ebe.plot_hypnograms(sleep_id=10)

    .. plot::

        >>> fig, ax = plt.subplots(figsize=(6, 3))
        >>> ebe.plot_hypnograms(
        >>>     sleep_id=8, ax=ax, obs_kwargs={"color": "red", "lw": 2, "ls": "dotted"}
        >>> )
        >>> plt.tight_layout()

    .. plot::

        >>> session = 8
        >>> fig, ax = plt.subplots(figsize=(6.5, 2.5), constrained_layout=True)
        >>> style_a = dict(alpha=1, lw=2.5, ls="solid", color="gainsboro", label="Michel")
        >>> style_b = dict(alpha=1, lw=2.5, ls="solid", color="cornflowerblue", label="Jouvet")
        >>> legend_style = dict(
        >>>     title="Scorer", frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.9)
        >>> )
        >>> ax = ebe.plot_hypnograms(
        >>>     sleep_id=session, ref_kwargs=style_a, obs_kwargs=style_b, legend=legend_style, ax=ax
        >>> )
        >>> acc = ebe.get_agreement().multiply(100).at[session, "accuracy"]
        >>> ax.text(
        >>>     0.01, 1, f"Accuracy = {acc:.0f}%", ha="left", va="bottom", transform=ax.transAxes
        >>> )

    When comparing only 2 hypnograms, use the :py:meth:`~yasa.Hynogram.evaluate` method:

    >>> hypno_a = yasa.simulate_hypnogram(tib=90, scorer="RaterA", seed=8)
    >>> hypno_b = hypno_a.simulate_similar(scorer="RaterB", seed=9)
    >>> ebe = hypno_a.evaluate(hypno_b)
    >>> ebe.get_confusion_matrix()
    RaterB  WAKE  N1  N2  N3
    RaterA
    WAKE      71   2  20   8
    N1         1   0   9   0
    N2        12   4  25   0
    N3        24   0   1   3
    """

    def __init__(self, ref_hyps, obs_hyps):
        from yasa.hypno import Hypnogram  # Avoiding circular import, bc hypno imports this class

        assert hasattr(ref_hyps, "__iter__"), "`ref_hyps` must be a an iterable"
        assert hasattr(obs_hyps, "__iter__"), "`obs_hyps` must be a an iterable"
        assert type(ref_hyps) is type(obs_hyps), "`ref_hyps` and `obs_hyps` must be the same type"
        assert len(ref_hyps) == len(
            obs_hyps
        ), "`ref_hyps` and `obs_hyps` must have the same number of hypnograms"

        if isinstance(ref_hyps, dict):
            # If user provides dictionaries, split into sleep IDs and hypnograms
            assert (
                ref_hyps.keys() == obs_hyps.keys()
            ), "keys in `ref_hyps` must be the same as keys in `obs_hyps`"
            sleep_ids, ref_hyps = zip(*ref_hyps.items())
            obs_hyps = tuple(obs_hyps.values())
        else:
            # Create hypnogram_ids
            sleep_ids = tuple(range(1, 1 + len(ref_hyps)))

        assert all(
            isinstance(hyp, Hypnogram) for hyp in ref_hyps + obs_hyps
        ), "`ref_hyps` and `obs_hyps` must only contain YASA hypnograms"
        assert all(
            h.scorer is not None for h in ref_hyps + obs_hyps
        ), "all hypnograms in `ref_hyps` and `obs_hyps` must have a scorer name"
        for h1, h2 in zip((ref_hyps + obs_hyps)[:-1], (ref_hyps + obs_hyps)[1:]):
            assert h1.freq == h2.freq, "all hypnograms must have the same freq"
            assert h1.labels == h2.labels, "all hypnograms must have the same labels"
            assert h1.mapping == h2.mapping, "all hypnograms must have the same mapping"
            assert h1.n_stages == h2.n_stages, "all hypnograms must have the same n_stages"
        assert all(
            h1.scorer == h2.scorer for h1, h2 in zip(ref_hyps[:-1], ref_hyps[1:])
        ), "all `ref_hyps` must have the same scorer"
        assert all(
            h1.scorer == h2.scorer for h1, h2 in zip(obs_hyps[:-1], obs_hyps[1:])
        ), "all `obs_hyps` must have the same scorer"
        assert all(
            h1.scorer != h2.scorer for h1, h2 in zip(ref_hyps, obs_hyps)
        ), "each `ref_hyps` and `obs_hyps` pair must have unique scorers"
        assert all(
            h1.n_epochs == h2.n_epochs for h1, h2 in zip(ref_hyps, obs_hyps)
        ), "each `ref_hyps` and `obs_hyps` pair must have the same n_epochs"
        # Convert ref_hyps and obs_hyps to dictionaries with sleep_id keys and hypnogram values
        ref_hyps = {s: h for s, h in zip(sleep_ids, ref_hyps)}
        obs_hyps = {s: h for s, h in zip(sleep_ids, obs_hyps)}

        # Merge all hypnograms into a single MultiIndexed dataframe
        ref = pd.concat(pd.concat({s: h.as_int()}, names=["sleep_id"]) for s, h in ref_hyps.items())
        obs = pd.concat(pd.concat({s: h.as_int()}, names=["sleep_id"]) for s, h in obs_hyps.items())
        data = pd.concat([ref, obs], axis=1)

        # Generate some mapping dictionaries to be used later in class methods
        skm_labels = np.unique(data).tolist()  # all unique YASA integer codes in this hypno
        skm2yasa_map = {i: l for i, l in enumerate(skm_labels)}  # skm order to YASA integers
        yasa2yasa_map = ref_hyps[sleep_ids[0]].mapping_int.copy()  # YASA integer to YASA string

        # Set attributes
        self._data = data
        self._sleep_ids = sleep_ids
        self._ref_hyps = ref_hyps
        self._obs_hyps = obs_hyps
        self._ref_scorer = ref_hyps[sleep_ids[0]].scorer
        self._obs_scorer = obs_hyps[sleep_ids[0]].scorer
        self._skm_labels = skm_labels
        self._skm2yasa_map = skm2yasa_map
        self._yasa2yasa_map = yasa2yasa_map

    def __repr__(self):
        # TODO v0.8: Keep only the text between < and >
        s = "s" if self.n_sleeps > 1 else ""
        return (
            f"<EpochByEpochAgreement | Observed hypnogram{s} scored by {self.obs_scorer} "
            f"evaluated against reference hypnogram{s} scored by {self.ref_scorer}, "
            f"{self.n_sleeps} sleep session{s}>\n"
            " - Use `.get_agreement()` to get agreement measures as a pandas DataFrame or Series\n"
            " - Use `.plot_hypnograms()` to plot two overlaid hypnograms\n"
            "See the online documentation for more details."
        )

    def __str__(self):
        return self.__repr__()

    @property
    def data(self):
        """A :py:class:`pandas.DataFrame` including all hypnograms."""
        return self._data

    @property
    def n_sleeps(self):
        """The number of unique sleep sessions."""
        return len(self._sleep_ids)

    @property
    def ref_scorer(self):
        """The name of the reference scorer."""
        return self._ref_scorer

    @property
    def obs_scorer(self):
        """The name of the observed scorer."""
        return self._obs_scorer

    @staticmethod
    def multi_scorer(df, scorers):
        """
        Compute multiple agreement scores from a 2-column dataframe (an optional 3rd column may
        contain sample weights).

        This function offers convenience when calculating multiple agreement scores using
        :py:meth:`pandas.DataFrame.groupby.apply`. Scikit-learn doesn't include a function that
        returns multiple scores, and the GroupBy implementation of ``apply`` in pandas does not
        accept multiple functions.

        Parameters
        ----------
        df : :py:class:`pandas.DataFrame`
            A :py:class:`~pandas.DataFrame` with 2 columns and length of *n_samples*.
            The first column contains reference values and second column contains observed values.
            If a third column, it must contain sample weights to be passed to underlying
            :py:mod:`sklearn.metrics` functions as ``sample_weight`` where applicable.
        scorers : dictionary
            The scorers to be used for evaluating agreement. A dictionary with scorer names (str) as
            keys and functions as values.

        Returns
        -------
        scores : dict
            A dictionary with scorer names (``str``) as keys and scores (``float``) as values.
        """
        assert isinstance(df, pd.DataFrame), "`df` must be a pandas DataFrame"
        assert df.shape[1] in [2, 3], "`df` must have either 2 or 3 columns"
        assert isinstance(scorers, dict), "`scorers` must be a dictionary"
        assert all(
            isinstance(k, str) and callable(v) for k, v in scorers.items()
        ), "Each key of `scorers` must be a string, and each value must be a callable function"
        if df.shape[1] == 3:
            true, pred, weights = zip(*df.values)
        elif df.shape[1] == 2:
            true, pred = zip(*df.values)  # Same as (df["col1"], df["col2"]) but teensy bit faster
            weights = None
        scores = {s: f(true, pred, weights) for s, f in scorers.items()}
        return scores

    def get_agreement(self, sample_weight=None, scorers=None):
        """
        Return a :py:class:`pandas.DataFrame` of weighted (i.e., averaged) agreement scores.

        Parameters
        ----------
        self : :py:class:`~yasa.evaluation.EpochByEvaluation`
            A :py:class:`~yasa.evaluation.EpochByEvaluation` instance.
        sample_weight : None or :py:class:`pandas.Series`
            Sample weights passed to underlying :py:mod:`sklearn.metrics` functions where possible.
            If a :py:class:`pandas.Series`, the index must match exactly that of
            :py:attr:`~yasa.Hypnogram.data`.
        scorers : None, list, or dictionary
            The scorers to be used for evaluating agreement. If None (default), default scorers are
            used. If a list, the list must contain strings that represent metrics from the sklearn
            metrics module (e.g., ``accuracy``, ``precision``). If more customization is desired, a
            dictionary can be passed with scorer names (str) as keys and custom functions as values.
            The custom functions should take 3 positional arguments (true values, predicted values,
            and sample weights).

        Returns
        -------
        agreement : :py:class:`pandas.DataFrame`
            A :py:class:`~pandas.DataFrame` with agreement metrics as columns and sessions as rows.
        """
        assert isinstance(
            sample_weight, (type(None), pd.Series)
        ), "`sample_weight` must be None or pandas Series"
        assert isinstance(scorers, (type(None), list, dict))
        if isinstance(scorers, list):
            assert all(isinstance(x, str) for x in scorers)
        elif isinstance(scorers, dict):
            assert all(isinstance(k, str) and callable(v) for k, v in scorers.items())
        if scorers is None:
            # Create dictionary of default scorer functions
            scorers = {
                "accuracy": lambda t, p, w: skm.accuracy_score(
                    t, p, normalize=True, sample_weight=w
                ),
                "balanced_acc": lambda t, p, w: skm.balanced_accuracy_score(
                    t, p, adjusted=False, sample_weight=w
                ),
                "kappa": lambda t, p, w: skm.cohen_kappa_score(
                    t, p, labels=None, weights=None, sample_weight=w
                ),
                "mcc": lambda t, p, w: skm.matthews_corrcoef(t, p, sample_weight=w),
                "precision": lambda t, p, w: skm.precision_score(
                    t, p, average="weighted", sample_weight=w, zero_division=0
                ),
                "recall": lambda t, p, w: skm.recall_score(
                    t, p, average="weighted", sample_weight=w, zero_division=0
                ),
                "f1": lambda t, p, w: skm.f1_score(
                    t, p, average="weighted", sample_weight=w, zero_division=0
                ),
            }
        elif isinstance(scorers, list):
            # Convert the list to a dictionary of sklearn scorers
            scorers = {s: skm.__getattribute__(f"{s}_scorer") for s in scorers}
        # Make a copy of data since weights series might be added to it
        df = self.data.copy()
        if sample_weight is not None:
            assert sample_weight.index == self.data.index, (
                "If not `None`, `sample_weight` Series must be a pandas Series with the same index "
                "as `self.data`"
            )
            # Add weights as a third column for multi_scorer to use
            df["weights"] = sample_weight
        # Get individual-level averaged/weighted agreement scores
        agreement = df.groupby(level=0).apply(self.multi_scorer, scorers=scorers).apply(pd.Series)
        # Set attribute for later access
        self._agreement = agreement
        # Convert to Series if just one session being evaluated
        if self.n_sleeps == 1:
            agreement = agreement.squeeze().rename("agreement")
        return agreement

    def get_agreement_bystage(self, beta=1.0):
        """
        Return a :py:class:`pandas.DataFrame` of unweighted (i.e., one-vs-rest) agreement scores.

        Parameters
        ----------
        self : :py:class:`~yasa.evaluation.EpochByEvaluation`
            A :py:class:`~yasa.evaluation.EpochByEvaluation` instance.
        beta : float
            See :py:func:`sklearn.metrics.precision_recall_fscore_support`.

        Returns
        -------
        agreement : :py:class:`pandas.DataFrame`
            A :py:class:`~pandas.DataFrame` with agreement metrics as columns and a
            :py:class:`~pandas.MultiIndex` with session and sleep stage as rows.
        """

        def scorer(df):
            return skm.precision_recall_fscore_support(
                *df.values.T, beta=beta, labels=self._skm_labels, average=None, zero_division=0
            )

        agreement = (
            self.data
            # Get precision, recall, f1, and support for each individual sleep session
            .groupby(level=0)
            .apply(scorer)
            # Unpack arrays
            .explode()
            .apply(pd.Series)
            # Add metric labels column and prepend it to index, creating MultiIndex
            .assign(metric=["precision", "recall", "fbeta", "support"] * self.n_sleeps)
            .set_index("metric", append=True)
            # Convert stage column names to string labels
            .rename_axis(columns="stage")
            .rename(columns=self._skm2yasa_map)
            .rename(columns=self._yasa2yasa_map)
            # Remove all-zero columns (i.e., stages that were not present in the hypnogram)
            .pipe(lambda df: df.loc[:, df.any()])
            # Reshape so metrics are columns
            .stack()
            .unstack("metric")
            .rename_axis(columns=None)
            # Swap MultiIndex levels and sort so stages are in standard YASA order
            .swaplevel()
            .sort_index(
                level="stage",
                key=lambda x: x.map(lambda y: list(self._yasa2yasa_map.values()).index(y)),
            )
        )
        # Set attribute for later access
        self._agreement_bystage = agreement
        # Remove the MultiIndex if just one session being evaluated
        if self.n_sleeps == 1:
            agreement = agreement.reset_index(level=1, drop=True)
        return agreement

    def get_confusion_matrix(self, sleep_id=None, agg_func=None, **kwargs):
        """
        Return a ``ref_hyp``/``obs_hyp``confusion matrix from either a single session or all
        sessions concatenated together.

        Parameters
        ----------
        self : :py:class:`yasa.EpochByEpochAgreement`
            A :py:class:`yasa.EpochByEpochAgreement` instance.
        sleep_id : None or a valid sleep ID
            If None (default), cross-tabulation is derived from the entire group dataset.
            If a valid sleep ID, cross-tabulation is derived using only the reference and observed
            scored hypnograms from that sleep session.
        agg_func : None or str
            If None (default), group results returns a :py:class:`~pandas.DataFrame` complete with
            all individual session results. If not None, group results returns a
            :py:class:`~pandas.DataFrame` aggregated across sessions where ``agg_func`` is passed as
            ``func`` parameter in :py:meth:`pandas.DataFrame.groupby.agg`. For example, set
            ``agg_func="sum"`` to get a single confusion matrix across all epochs that does not take
            session into account.
        **kwargs : key, value pairs
            Additional keyword arguments are passed to :py:func:`sklearn.metrics.confusion_matrix`.

        Returns
        -------
        conf_matr : :py:class:`pandas.DataFrame`
            A confusion matrix with stages from the reference scorer as indices and stages from the
            test scorer as columns.

        Examples
        --------
        >>> import yasa
        >>> ref_hyps = [yasa.simulate_hypnogram(tib=90, scorer="Rater1", seed=i) for i in range(3)]
        >>> obs_hyps = [h.simulate_similar(scorer="Rater2", seed=i) for i, h in enumerate(ref_hyps)]
        >>> ebe = yasa.EpochByEpochAgreement(ref_hyps, obs_hyps)
        >>> ebe.get_confusion_matrix(sleep_id=2)
        Rater2  WAKE  N1  N2  N3  REM
        Rater1
        WAKE       1   2  23   0    0
        N1         0   9  13   0    0
        N2         0   6  71   0    0
        N3         0  13  42   0    0
        REM        0   0   0   0    0

        >>> ebe.get_confusion_matrix()
        Rater2           WAKE  N1  N2  N3  REM
        sleep_id Rater1
        1        WAKE      30   0   3   0   35
                 N1         3   2   7   0    0
                 N2        21  12   7   0    4
                 N3         0   0   0   0    0
                 REM        2   8  29   0   17
        2        WAKE       1   2  23   0    0
                 N1         0   9  13   0    0
                 N2         0   6  71   0    0
                 N3         0  13  42   0    0
                 REM        0   0   0   0    0
        3        WAKE      16   0   7  19   19
                 N1         0   7   2   0    5
                 N2         0  10  12   7    5
                 N3         0   0  16  11    0
                 REM        0  15  11  18    0

        >>> ebe.get_confusion_matrix(agg_func="sum")
        Rater2  WAKE  N1  N2  N3  REM
        Rater1
        WAKE      47   2  33  19   54
        N1         3  18  22   0    5
        N2        21  28  90   7    9
        N3         0  13  58  11    0
        REM        2  23  40  18   17
        """
        assert (
            sleep_id is None or sleep_id in self._sleep_ids
        ), "`sleep_id` must be None or a valid sleep ID"
        assert isinstance(agg_func, (type(None), str)), "`agg_func` must be None or a str"
        assert not (
            (self.n_sleeps == 1 or sleep_id is not None) and agg_func is not None
        ), "`agg_func` must be None if plotting a single session."
        kwargs = {"labels": self._skm_labels} | kwargs
        # Generate a DataFrame with a confusion matrix for each session
        #   Seems easier to just generate this whole thing and then either
        #   extract a single one or aggregate across them all, depending on user request
        confusion_matrices = (
            self.data
            # Get confusion matrix for each individual sleep session
            .groupby(level=0)
            .apply(lambda df: skm.confusion_matrix(*df.values.T, **kwargs))
            # Expand results matrix out from single cell
            .explode()
            .apply(pd.Series)
            # Convert to MultiIndex with reference scorer as new level
            .assign(**{self.ref_scorer: self._skm_labels * self.n_sleeps})
            .set_index(self.ref_scorer, append=True)
            .rename_axis(columns=self.obs_scorer)
            # Convert sleep stage columns and indices to strings
            .rename(columns=self._skm2yasa_map)
            .rename(columns=self._yasa2yasa_map)
            .rename(index=self._skm2yasa_map, level=self.ref_scorer)
            .rename(index=self._yasa2yasa_map, level=self.ref_scorer)
        )
        if self.n_sleeps == 1:
            # If just one session, use the only session ID as the key, for simplified returned df
            sleep_id = self._sleep_ids[0]
        if sleep_id is None:
            if agg_func is None:
                mat = confusion_matrices
            else:
                mat = confusion_matrices.groupby(self.ref_scorer, sort=False).agg(agg_func)
        else:
            mat = confusion_matrices.loc[sleep_id]
        return mat

    def get_sleep_stats(self):
        """
        Return a :py:class:`pandas.DataFrame` of sleep statistics for each hypnogram derived from
        both reference and observed scorers.

        .. seealso:: :py:meth:`yasa.Hypnogram.sleep_statistics`

        .. seealso:: :py:class:`yasa.SleepStatsAgreement`

        Parameters
        ----------
        self : :py:class:`yasa.EpochByEpochAgreement`
            A :py:class:`yasa.EpochByEpochAgreement` instance.

        Returns
        -------
        sstats : :py:class:`pandas.DataFrame`
            A :py:class:`~pandas.DataFrame` with sleep statistics as columns and two rows for each
            individual (one for reference scorer and another for test scorer).
        """
        # Get all sleep statistics
        ref_sstats = pd.DataFrame({s: h.sleep_statistics() for s, h in self._ref_hyps.items()})
        obs_sstats = pd.DataFrame({s: h.sleep_statistics() for s, h in self._obs_hyps.items()})
        # Reshape and name axis
        ref_sstats = ref_sstats.T.rename_axis("sleep_id")
        obs_sstats = obs_sstats.T.rename_axis("sleep_id")
        # Convert to MultiIndex with new scorer level
        ref_sstats = pd.concat({self.ref_scorer: ref_sstats}, names=["scorer"])
        obs_sstats = pd.concat({self.obs_scorer: obs_sstats}, names=["scorer"])
        # Concatenate into one DataFrame
        sstats = pd.concat([ref_sstats, obs_sstats])
        # Remove the MultiIndex if just one session being evaluated
        if self.n_sleeps == 1:
            sstats = sstats.reset_index(level=1, drop=True)
        return sstats

    def plot_hypnograms(self, sleep_id=None, legend=True, ax=None, ref_kwargs={}, obs_kwargs={}):
        """Plot the two hypnograms of one session overlapping on the same axis.

        .. seealso:: :py:func:`yasa.plot_hypnogram`

        Parameters
        ----------
        self : :py:class:`yasa.EpochByEpochAgreement`
            A :py:class:`yasa.EpochByEpochAgreement` instance.
        sleep_id : a valid sleep ID or None
            The sleep session to plot. If multiple sessions are included in the
            :py:class:`~yasa.EpochByEpochAgreement` instance, a ``sleep_id`` must be provided. If
            only one session is present, ``None`` (default) will plot the two hypnograms of the
            only session.
        legend : bool or dict
            If True (default) or a dictionary, a legend is added. If a dictionary, all key/value
            pairs are passed as keyword arguments to the :py:func:`matplotlib.pyplot.legend` call.
        ax : :py:class:`matplotlib.axes.Axes` or None
            Axis on which to draw the plot, optional.
        ref_kwargs : dict
            Keyword arguments passed to :py:func:`yasa.plot_hypnogram` when plotting the reference
            hypnogram.
        obs_kwargs : dict
            Keyword arguments passed to :py:func:`yasa.plot_hypnogram` when plotting the observed
            hypnogram.

        Returns
        -------
        ax : :py:class:`matplotlib.axes.Axes`
            Matplotlib Axes

        Examples
        --------
        .. plot::

            >>> from yasa import simulate_hypnogram
            >>> hyp = simulate_hypnogram(scorer="Anthony", seed=19)
            >>> ax = hyp.evaluate(hyp.simulate_similar(scorer="Alan", seed=68)).plot_hypnograms()
        """
        assert (
            sleep_id is None or sleep_id in self._sleep_ids
        ), "`sleep_id` must be None or a valid sleep ID"
        assert isinstance(legend, (bool, dict)), "`legend` must be True, False, or a dictionary"
        assert isinstance(ref_kwargs, dict), "`ref_kwargs` must be a dictionary"
        assert isinstance(obs_kwargs, dict), "`obs_kwargs` must be a dictionary"
        assert (
            "ax" not in ref_kwargs | obs_kwargs
        ), "'ax' can't be supplied to `ref_kwargs` or `obs_kwargs`, use the `ax` keyword instead"
        assert not (sleep_id is None and self.n_sleeps > 1), (
            "Multi-session plotting is not currently supported. `sleep_id` must not be None when "
            "multiple sessions are present"
        )
        # Select the session hypnograms to plot
        if sleep_id is None and self.n_sleeps == 1:
            ref_hyp = self._ref_hyps[self._sleep_ids[0]]
            obs_hyp = self._obs_hyps[self._sleep_ids[0]]
        else:
            ref_hyp = self._ref_hyps[sleep_id]
            obs_hyp = self._obs_hyps[sleep_id]
        # Set default plotting kwargs and merge with user kwargs
        plot_ref_kwargs = {
            "label": self.ref_scorer,
            "highlight": None,
            "color": "black",
            "alpha": 0.8,
        }
        plot_obs_kwargs = {
            "label": self.obs_scorer,
            "highlight": None,
            "color": "green",
            "alpha": 0.8,
            "ls": "dashed",
        }
        plot_ref_kwargs.update(ref_kwargs)
        plot_obs_kwargs.update(obs_kwargs)
        # Draw the hypnograms
        ax = ref_hyp.plot_hypnogram(ax=ax, **plot_ref_kwargs)
        ax = obs_hyp.plot_hypnogram(ax=ax, **plot_obs_kwargs)
        # Add legend if desired
        if legend:
            if isinstance(legend, dict):
                ax.legend(**legend)
            else:
                ax.legend()
        return ax

    def summary(self, by_stage=False, **kwargs):
        """Return group-level agreement scores.

        Default aggregated measures are

        Parameters
        ----------
        self : :py:class:`~yasa.evaluation.EpochByEpochAgreement`
            A :py:class:`~yasa.evaluation.EpochByEpochAgreement` instance.
        by_stage : bool
            If ``False`` (default), ``summary`` will include agreement scores derived from
            average-based metrics. If ``True``, returned ``summary`` :py:class:`~pandas.DataFrame`
            will include agreement scores for each sleep stage, derived from one-vs-rest metrics.
        **kwargs : key, value pairs
            Additional keyword arguments are passed to :py:meth:`pandas.DataFrame.groupby.agg`.
            This can be used to customize the descriptive statistics returned.

        Returns
        -------
        summary : :py:class:`pandas.DataFrame`
            A :py:class:`pandas.DataFrame` summarizing agreement scores across the entire dataset
            with descriptive statistics.

            >>> ebe = yasa.EpochByEpochAgreement(...)
            >>> agreement = ebe.get_agreement()
            >>> ebe.summary()

            This will give a :py:class:`~pandas.DataFrame` where each row is an agreement metric and
            each column is a descriptive statistic (e.g., mean, standard deviation).
            To control the descriptive statistics included as columns:

            >>> ebe.summary(func=["count", "mean", "sem"])
        """
        assert self.n_sleeps > 1, "Summary scores can not be computed with only one hypnogram pair."
        assert isinstance(by_stage, bool), "`by_stage` must be True or False"
        if by_stage:
            assert hasattr(
                self, "_agreement_bystage"
            ), "Must run `self.get_agreement_bystage` before obtaining by_stage summary results."
        else:
            assert hasattr(
                self, "_agreement"
            ), "Must run `self.get_agreement` before obtaining summary results."

        # Create a function for getting mean absolute deviation
        def mad(df):
            return (df - df.mean()).abs().mean()

        # Merge default and user kwargs
        agg_kwargs = {"func": [mad, "mean", "std", "min", "median", "max"]} | kwargs
        if by_stage:
            summary = (
                self.agreement_bystage.groupby("stage")
                .agg(**agg_kwargs)
                .stack(level=0)
                .rename_axis(["stage", "metric"])
            )
        else:
            summary = self._agreement.agg(**agg_kwargs).T.rename_axis("metric")
        return summary


################################################################################
# SLEEP STATISTICS
################################################################################


class SleepStatsAgreement:
    """
    Evaluate agreement between sleep statistics reported by two different scorers.
    Evaluation includes bias and limits of agreement (as well as both their confidence intervals),
    various plotting options, and calibration functions for correcting biased values from the
    observed scorer.

    Features include:
    * Get summary calculations of bias, limits of agreement, and their confidence intervals.
    * Test statistical assumptions of bias, limits of agreement, and their confidence intervals,
    and apply corrective procedures when the assumptions are not met.
    * Get bias and limits of agreement in a string-formatted table.
    * Calibrate new data to correct for biases in observed data.
    * Return individual calibration functions.
    * Visualize discrepancies for outlier inspection.
    * Visualize Bland-Altman plots.

    .. seealso:: :py:meth:`yasa.Hypnogram.sleep_statistics`

    .. versionadded:: 0.7.0

    Parameters
    ----------
    ref_data : :py:class:`pandas.DataFrame`
        A :py:class:`pandas.DataFrame` with sleep statistics from the reference scorer.
        Rows are unique observations and columns are unique sleep statistics.
    obs_data : :py:class:`pandas.DataFrame`
        A :py:class:`pandas.DataFrame` with sleep statistics from the observed scorer.
        Rows are unique observations and columns are unique sleep statistics.
        Shape, index, and columns must be identical to ``ref_data``.
    ref_scorer : str
        Name of the reference scorer.
    obs_scorer : str
        Name of the observed scorer.
    agreement : float
        Multiple of the standard deviation to plot agreement limits. The default is 1.96, which
        corresponds to a 95% confidence interval if the differences are normally distributed.

        .. note:: ``agreement`` gets adjusted for regression-modeled limits of agreement.
    confidence : float
        The percentage confidence interval for the confidence intervals that are applied to bias and
        limits of agreement. The same confidence interval percentage is applied to both standard and
        bootstrapped confidence intervals.
    alpha : float
        Alpha cutoff used for all assumption tests.
    verbose : bool or str
        Verbose level. Default (False) will only print warning and error messages. The logging
        levels are 'debug', 'info', 'warning', 'error', and 'critical'. For most users the choice is
        between 'info' (or ``verbose=True``) and warning (``verbose=False``).

    Notes
    -----
    Sleep statistics that are identical between scorers are removed from analysis.

    Many steps here are influenced by guidelines proposed in Menghini et al., 2021 [Menghini2021]_.
    See https://sri-human-sleep.github.io/sleep-trackers-performance/AnalyticalPipeline_v1.0.0.html

    References
    ----------
    .. [Menghini2021] Menghini, L., Cellini, N., Goldstone, A., Baker, F. C., & de Zambotti, M.
                      (2021). A standardized framework for testing the performance of sleep-tracking
                      technology: step-by-step guidelines and open-source code. SLEEP, 44(2),
                      zsaa170. https://doi.org/10.1093/sleep/zsaa170

    Examples
    --------
    >>> import pandas as pd
    >>> import yasa
    >>>
    >>> # Generate fake reference and observed datasets with similar sleep statistics
    >>> ref_scorer = "Henri"
    >>> obs_scorer = "Piéron"
    >>> ref_hyps = [yasa.simulate_hypnogram(tib=600, scorer=ref_scorer, seed=i) for i in range(20)]
    >>> obs_hyps = [h.simulate_similar(scorer=obs_scorer, seed=i) for i, h in enumerate(ref_hyps)]
    >>> # Generate sleep statistics from hypnograms using EpochByEpochAgreement
    >>> eea = yasa.EpochByEpochAgreement(ref_hyps, obs_hyps)
    >>> sstats = eea.get_sleep_stats()
    >>> ref_sstats = sstats.loc[ref_scorer]
    >>> obs_sstats = sstats.loc[obs_scorer]
    >>> # Create SleepStatsAgreement instance
    >>> ssa = yasa.SleepStatsAgreement(ref_sstats, obs_sstats)
    >>> ssa.summary().round(1).head(3)
    variable   bias_intercept             ...   uloa_parm
    interval           center lower upper ...      center lower upper
    sleep_stat                            ...
    %N1                  -5.4 -13.9   3.2 ...         6.1   3.7   8.5
    %N2                 -27.3 -49.1  -5.6 ...        12.4   7.2  17.6
    %N3                  -9.1 -23.8   5.5 ...        20.4  12.6  28.3

    >>> ssa.get_table().head(3)[["bias", "loa"]]
                          bias                            loa
    sleep_stat
    %N1                   0.25  Bias ± 2.46 * (-0.00 + 1.00x)
    %N2         -27.34 + 0.55x   Bias ± 2.46 * (0.00 + 1.00x)
    %N3                   1.38   Bias ± 2.46 * (0.00 + 1.00x)

    >>> ssa.assumptions.head(3)
                unbiased  normal  constant_bias  homoscedastic
    sleep_stat
    %N1             True    True           True          False
    %N2             True    True          False          False
    %N3             True    True           True          False

    >>> ssa.auto_methods.head(3)
                bias   loa    ci
    sleep_stat
    %N1         parm  regr  parm
    %N2         regr  regr  parm
    %N3         parm  regr  parm

    >>> ssa.get_table(bias_method="parm", loa_method="parm").head(3)[["bias", "loa"]]
                 bias            loa
    sleep_stat
    %N1          0.25    -5.55, 6.06
    %N2         -0.23  -12.87, 12.40
    %N3          1.38  -17.67, 20.44

    >>> new_hyps = [h.simulate_similar(scorer="Kelly", seed=i) for i, h in enumerate(obs_hyps)]
    >>> new_sstats = pd.Series(new_hyps).map(lambda h: h.sleep_statistics()).apply(pd.Series)
    >>> new_sstats = new_sstats[["N1", "TST", "WASO"]]
    >>> new_sstats.round(1).head(5)
         N1    TST   WASO
    0  42.5  439.5  147.5
    1  84.0  550.0   38.5
    2  53.5  489.0  103.0
    3  57.0  469.5  120.0
    4  71.0  531.0   69.0

    >>> new_stats_calibrated = ssa.calibrate_stats(new_sstats, bias_method="auto")
    >>> new_stats_calibrated.round(1).head(5)
         N1    TST   WASO
    0  42.9  433.8  150.0
    1  84.4  544.2   41.0
    2  53.9  483.2  105.5
    3  57.4  463.8  122.5
    4  71.4  525.2   71.5

    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> ax = ssa.plot_discrepancies_heatmap()
        >>> ax.set_title("Sleep statistic discrepancies")
        >>> plt.tight_layout()

    .. plot::

        >>> ssa.plot_blandaltman()
    """

    def __init__(
        self,
        ref_data,
        obs_data,
        *,
        ref_scorer="Reference",
        obs_scorer="Observed",
        agreement=1.96,
        confidence=0.95,
        alpha=0.05,
        verbose=True,
        bootstrap_kwargs={},
    ):

        restricted_bootstrap_kwargs = ["confidence_level", "vectorized", "paired"]

        assert isinstance(ref_data, pd.DataFrame), "`ref_data` must be a pandas DataFrame"
        assert isinstance(obs_data, pd.DataFrame), "`obs_data` must be a pandas DataFrame"
        assert np.array_equal(
            ref_data.index, obs_data.index
        ), "`ref_data` and `obs_data` index values must be identical"
        assert (
            ref_data.index.name == obs_data.index.name
        ), "`ref_data` and `obs_data` index names must be identical"
        assert np.array_equal(
            ref_data.columns, obs_data.columns
        ), "`ref_data` and `obs_data` column values must be identical"
        assert isinstance(ref_scorer, str), "`ref_scorer` must be a string"
        assert isinstance(obs_scorer, str), "`obs_scorer` must be a string"
        assert ref_scorer != obs_scorer, "`ref_scorer` and `obs_scorer` must be unique"
        assert (
            isinstance(agreement, (float, int)) and agreement > 0
        ), "`agreement` must be a number greater than 0"
        assert (
            isinstance(confidence, (float, int)) and 0 < alpha < 1
        ), "`confidence` must be a number between 0 and 1"
        assert (
            isinstance(alpha, (float, int)) and 0 <= alpha <= 1
        ), "`alpha` must be a number between 0 and 1 inclusive"
        assert isinstance(bootstrap_kwargs, dict), "`bootstrap_kwargs` must be a dictionary"
        assert all(
            k not in restricted_bootstrap_kwargs for k in bootstrap_kwargs
        ), f"None of {restricted_bootstrap_kwargs} can be set by the user"

        # If `ref_data` and `obs_data` indices are unnamed, name them
        session_key = "session_id" if ref_data.index.name is None else ref_data.index.name
        ref_data.index.name = obs_data.index.name = session_key

        # Reshape to long format DataFrame with 2 columns (observed, reference) and MultiIndex
        data = (
            pd.concat([obs_data, ref_data], keys=[obs_scorer, ref_scorer], names=["scorer"])
            .melt(var_name="sleep_stat", ignore_index=False)
            .pivot_table(index=["sleep_stat", session_key], columns="scorer", values="value")
            .rename_axis(columns=None)
            .sort_index()
        )

        # Get scorer differences (i.e., observed minus reference)
        data["difference"] = data[obs_scorer] - data[ref_scorer]

        # Remove sleep statistics that have no differences between scorers
        stats_rm = data.groupby("sleep_stat")["difference"].any().loc[lambda x: ~x].index.tolist()
        data = data.drop(labels=stats_rm)
        for s in stats_rm:
            logger.warning(f"Removed {s} from evaluation because all scorings were identical.")

        # Create grouper and n_sessions variables for convenience
        grouper = data.groupby("sleep_stat")
        n_sessions = data.index.get_level_values(session_key).nunique()

        ########################################################################
        # Generate parametric Bias and LoA for all sleep stats
        ########################################################################
        # Parametric Bias
        parm_vals = grouper["difference"].mean().to_frame("bias_parm")
        # Parametric LoA
        parm_vals["lloa_parm"], parm_vals["uloa_parm"] = zip(
            *grouper["difference"].apply(self._arr_to_loa, agreement=agreement)
        )

        ########################################################################
        # Generate standard CIs for parametric Bias and LoA for all sleep stats
        ########################################################################
        # Get critical t and standard error used to calculate parametric CIs for parametric Bias/LoA
        t_parm = sps.t.ppf((1 + confidence) / 2, n_sessions - 1)
        sem = grouper["difference"].sem(ddof=1)
        # Parametric CIs for parametric Bias and LoA
        parm_ci = pd.DataFrame(
            {
                "bias_parm-lower": parm_vals["bias_parm"] - sem * t_parm,
                "bias_parm-upper": parm_vals["bias_parm"] + sem * t_parm,
                "lloa_parm-lower": parm_vals["lloa_parm"] - sem * t_parm * np.sqrt(3),
                "lloa_parm-upper": parm_vals["lloa_parm"] + sem * t_parm * np.sqrt(3),
                "uloa_parm-lower": parm_vals["uloa_parm"] - sem * t_parm * np.sqrt(3),
                "uloa_parm-upper": parm_vals["uloa_parm"] + sem * t_parm * np.sqrt(3),
            }
        )

        ########################################################################
        # Generate regression/modeled (slope and intercept) Bias and LoA for all sleep stats
        ########################################################################
        # Run regression used to (a) model bias and (b) test for proportional/constant bias
        bias_regr = grouper[[ref_scorer, "difference"]].apply(self._linregr_dict).apply(pd.Series)
        # Get absolute residuals from this regression bc they are used in the next regression
        idx = data.index.get_level_values("sleep_stat")
        slopes = bias_regr.loc[idx, "slope"].to_numpy()
        intercepts = bias_regr.loc[idx, "intercept"].to_numpy()
        predicted_values = data[ref_scorer].to_numpy() * slopes + intercepts
        data["residuals"] = data[obs_scorer].to_numpy() - predicted_values
        data["residuals_abs"] = data["residuals"].abs()
        # Run regression used to (a) model LoA and (b) test for heteroscedasticity/homoscedasticity
        loa_regr = grouper[[ref_scorer, "residuals_abs"]].apply(self._linregr_dict).apply(pd.Series)
        # Stack the two regression dataframes together
        regr = pd.concat({"bias": bias_regr, "loa": loa_regr}, axis=0)

        ########################################################################
        # Generate parametric CIs for regression/modeled Bias and LoA for all sleep stats
        ########################################################################
        # Get critical t used used to calculate parametric CIs for regression Bias/LoA
        t_regr = sps.t.ppf((1 + confidence) / 2, n_sessions - 2)  # dof=n-2 for regression
        # Parametric CIs for modeled Bias and LoA
        regr_ci = pd.DataFrame(
            {
                "intercept-lower": regr["intercept"] - regr["intercept_stderr"] * t_regr,
                "intercept-upper": regr["intercept"] + regr["intercept_stderr"] * t_regr,
                "slope-lower": regr["slope"] - regr["stderr"] * t_regr,
                "slope-upper": regr["slope"] + regr["stderr"] * t_regr,
            }
        )

        ########################################################################
        # Test all statistical assumptions
        ########################################################################
        assumptions = pd.DataFrame(
            {
                "unbiased": (
                    grouper["difference"].apply(lambda a: sps.ttest_1samp(a, 0).pvalue).ge(alpha)
                ),
                "normal": grouper["difference"].apply(lambda a: sps.shapiro(a).pvalue).ge(alpha),
                "constant_bias": bias_regr["pvalue"].ge(alpha),
                "homoscedastic": loa_regr["pvalue"].ge(alpha),
            }
        )

        ########################################################################
        # Setting attributes
        ########################################################################

        # Merge the parametric and regression values for Bias and LoA
        regr_vals = regr.unstack(0)[["slope", "intercept"]]
        regr_vals.columns = regr_vals.columns.swaplevel().map("_".join)
        vals = parm_vals.join(regr_vals).rename_axis("variable", axis=1)

        # Merge the two CI dataframes for easier access
        regr_ci = regr_ci.unstack(0)
        regr_ci.columns = regr_ci.columns.swaplevel().map("_".join)
        ci = parm_ci.join(regr_ci)
        ci.columns = pd.MultiIndex.from_tuples(
            tuples=ci.columns.str.split("-", expand=True),
            names=["variable", "interval"],
        )
        empty_df = pd.DataFrame().reindex_like(ci)
        ci = pd.concat({"parm": ci, "boot": empty_df}, names=["ci_method"], axis=1)
        ci = ci.sort_index(axis=1)  # Sort MultiIndex columns for cleanliness

        # Set attributes
        self._agreement = agreement
        self._confidence = confidence
        self._bootstrap_kwargs = bootstrap_kwargs
        self._n_sessions = n_sessions
        self._ref_scorer = ref_scorer
        self._obs_scorer = obs_scorer
        self._data = data
        self._assumptions = assumptions
        self._regr = regr
        self._vals = vals
        self._ci = ci
        self._bias_method_opts = ["parm", "regr", "auto"]
        self._loa_method_opts = ["parm", "regr", "auto"]
        self._ci_method_opts = ["parm", "boot", "auto"]

    @property
    def ref_scorer(self):
        """The name of the reference scorer."""
        return self._ref_scorer

    @property
    def obs_scorer(self):
        """The name of the observed scorer."""
        return self._obs_scorer

    @property
    def n_sessions(self):
        """The number of sessions."""
        return self._n_sessions

    @property
    def data(self):
        """A long-format :py:class:`pandas.DataFrame` containing all raw sleep statistics from
        ``ref_data`` and ``obs_data``.
        """
        return self._data.drop(columns=["difference", "residuals", "residuals_abs"])

    @property
    def sleep_statistics(self):
        """Return a list of all sleep statistics included in the agreement analyses."""
        return self.data.index.get_level_values("sleep_stat").unique().to_list()

    @property
    def assumptions(self):
        """A :py:class:`pandas.DataFrame` containing boolean values indicating the pass/fail status
        of all statistical tests performed to test assumptions.
        """
        return self._assumptions

    @property
    def auto_methods(self):
        """
        A :py:class:`pandas.DataFrame` containing the methods applied when ``'auto'`` is selected.
        """
        return pd.concat(
            [
                self.assumptions["constant_bias"].map({True: "parm", False: "regr"}).rename("bias"),
                self.assumptions["homoscedastic"].map({True: "parm", False: "regr"}).rename("loa"),
                self.assumptions["normal"].map({True: "parm", False: "boot"}).rename("ci"),
            ],
            axis=1,
        )

    def __repr__(self):
        # TODO v0.8: Keep only the text between < and >
        return (
            f"<SleepStatsAgreement | Observed scorer ('{self.obs_scorer}') evaluated against "
            f"reference scorer ('{self.ref_scorer}'), {self.n_sessions} sleep sessions>\n"
            " - Use `.summary()` to get a dataframe of bias and limits of agreement for each sleep "
            "statistic\n"
            " - Use `.plot_blandaltman()` to get a Bland-Altman-plot grid for sleep statistics\n"
            "See the online documentation for more details."
        )

    def __str__(self):
        return self.__repr__()

    ############################################################################
    # Define some utility functions, mostly to aid with the use of df.apply and stats.bootstrap
    ############################################################################

    @staticmethod
    def _arr_to_loa(x, agreement):
        """Return a tuple with lower and upper limits of agreement."""
        mean = np.mean(x)
        bound = agreement * np.std(x, ddof=1)
        return mean - bound, mean + bound

    @staticmethod
    def _linregr_dict(*args, **kwargs):
        """
        A wrapper around :py:func:`scipy.stats.linregress` that returns a dictionary instead of a
        named tuple. In the normally returned object, ``intercept_stderr`` is an extra field that is
        not included when converting the named tuple, so this allows it to be included when using
        something like groupby.
        """
        regr = sps.linregress(*args, **kwargs)
        return {
            "slope": regr.slope,
            "intercept": regr.intercept,
            "rvalue": regr.rvalue,
            "pvalue": regr.pvalue,
            "stderr": regr.stderr,
            "intercept_stderr": regr.intercept_stderr,
        }

    def _generate_bootstrap_ci(self, sleep_stats):
        """
        Internal method to generate bootstrapped confidence intervals for bias and LoA.
        This operates in-place by concatenating bootstrapped CIs to existing parametric CIs.
        Note that parametric CIs are generated by default during init (bc they are quicker).

        Parameters
        ----------
        sleep_stats : list
            A list of sleep statistics to bootstrap confidence intervals for.
        """
        assert isinstance(sleep_stats, list), "`sleep_stats` must be a list"
        assert len(sleep_stats) == len(set(sleep_stats)), "elements of `sleep_stats` must be unique"
        assert all(
            isinstance(ss, str) for ss in sleep_stats
        ), "all elements of `sleep_stats` must be strings"
        assert all(
            ss in self.sleep_statistics for ss in sleep_stats
        ), f"all elements of `sleep_stats` must be one of {self.sleep_statistics}"
        # Update bootstrap keyword arguments with defaults
        bs_kwargs = {
            "n_resamples": 1000,
            "method": "BCa",
            "confidence_level": self._confidence,  # should not change from parametric level
            "vectorized": False,  # should stay False, bc of how the custom get_vars function works
            "paired": True,  # should stay True, especially if method is BCa
        } | self._bootstrap_kwargs

        def get_vars(ref_arr, diff_arr, rabs_arr):
            """A function to get all variables at once and avoid redundant stats.bootstrap calls."""
            bias_parm = np.mean(diff_arr)
            lloa_parm, uloa_parm = self._arr_to_loa(diff_arr, self._agreement)
            bias_slope, bias_inter = sps.linregress(ref_arr, diff_arr)[:2]
            # Note this is NOT recalculating residuals each time for the next regression
            loa_slope, loa_inter = sps.linregress(ref_arr, rabs_arr)[:2]
            return bias_parm, lloa_parm, uloa_parm, bias_inter, bias_slope, loa_inter, loa_slope

        # !! Column order MUST match the order of arrays boot_stats expects as INPUT
        # !! Variable order MUST match the order of floats boot_stats returns as OUTPUT
        interval_order = ["lower", "upper"]
        column_order = ["Reference", "difference", "residuals_abs"]
        variable_order = [
            "bias_parm",
            "lloa_parm",
            "uloa_parm",
            "bias_intercept",
            "bias_slope",
            "loa_intercept",
            "loa_slope",
        ]
        boot_ci = (
            self._data.loc[
                sleep_stats, column_order
            ]  # Extract the relevant sleep stats and columns
            .groupby("sleep_stat")  # Group so the bootstrapping is applied once to each sleep stat
            # Apply the bootstrap function, where tuple(df.to_numpy().T) convert the 3 columns
            # of the passed dataframe to a tuple of 3 1D arrays
            .apply(lambda df: sps.bootstrap(tuple(df.to_numpy().T), get_vars, **bs_kwargs))
            .map(lambda res: res.confidence_interval)  # Pull high/low CIs out of the results object
            .explode()  # Break high and low CIs into separate rows
            .to_frame("value")  # Convert to dataframe and name column
            .assign(interval=interval_order * len(sleep_stats))  # Add a column indicating interval
            .explode("value")  # Break low CI variables and high CI variables out of arrays
            .assign(variable=variable_order * len(sleep_stats) * 2)  # Add column indicating variabl
            .pivot(columns=["variable", "interval"], values="value")  # Go long to wide format
            .sort_index(axis=1)  # Sort MultiIndex columns for cleanliness
        )
        # Merge with existing CI dataframe
        self._ci["boot"] = self._ci["boot"].fillna(boot_ci)

    def get_table(self, bias_method="auto", loa_method="auto", ci_method="auto", fstrings={}):
        """
        Return a :py:class:`~pandas.DataFrame` with bias, loa, bias_ci, loa_ci as string equations.

        Parameters
        ----------
        bias_method : str
            If ``'parm'`` (i.e., parametric), bias is always represented as the mean difference
            (observed minus reference).
            If ``'regr'`` (i.e., regression), bias is always represented as a regression equation.
            If ``'auto'`` (default), bias is represented as a regression equation for sleep
            statistics where the score differences are proportionally biased and as the mean
            difference otherwise.
        loa_method : str
            If ``'parm'`` (i.e., parametric), limits of agreement are always represented as
            bias +/- 1.96 standard deviations (where 1.96 can be adjusted through the ``agreement``
            parameter).
            If ``'regr'`` (i.e., regression), limits of agreement are always represented as a
            regression equation.
            If ``'auto'`` (default), limits of agreement are represented as a regression equation
            for sleep statistics where the score differences are proportionally biased and as
            bias +/- 1.96 standard deviation otherwise.
        ci_method : str
            If ``'parm'`` (i.e., parametric), confidence intervals are always represented using a
            standard t-distribution.
            If ``'boot'`` (i.e., bootstrap), confidence intervals are always represented using a
            bootstrap resampling procedure.
            If  ``'auto'`` (default), confidence intervals are represented using a bootstrap
            resampling procedure for sleep statistics where the distribution of score differences is
            non-normal and using a standard t-distribution otherwise.
        fstrings : dict
            Optional custom strings for formatting cells.

        Returns
        -------
        table : :py:class:`pandas.DataFrame`
            A :py:class:`~pandas.DataFrame` of string representations for bias, limits of agreement,
            and their confidence intervals for all sleep statistics.
        """
        assert isinstance(bias_method, str), "`bias_method` must be a string"
        assert (
            bias_method in self._bias_method_opts
        ), f"`bias_method` must be one of {self._bias_method_opts}"
        assert isinstance(loa_method, str), "`loa_method` must be a string"
        assert (
            loa_method in self._loa_method_opts
        ), f"`loa_method` must be one of {self._loa_method_opts}"
        assert isinstance(fstrings, dict), "`fstrings` must be a dictionary"
        # Agreement gets adjusted when LoA is modeled
        loa_regr_agreement = self._agreement * np.sqrt(np.pi / 2)
        if not fstrings:
            fstrings = {
                "bias_parm": "{bias_parm_center:.2f}",
                "bias_regr": "{bias_intercept_center:.2f} + {bias_slope_center:.2f}x",
                "loa_parm": "{lloa_parm_center:.2f}, {uloa_parm_center:.2f}",
                "loa_regr": (
                    "Bias \u00B1 {loa_regr_agreement:.2f} "
                    "* ({loa_intercept_center:.2f} + {loa_slope_center:.2f}x)"
                ),
                "bias_parm_ci": ("[{bias_parm_lower:.2f}, {bias_parm_upper:.2f}]"),
                "bias_regr_ci": (
                    "[{bias_intercept_lower:.2f}, {bias_intercept_upper:.2f}], "
                    "[{bias_slope_lower:.2f}, {bias_slope_upper:.2f}]"
                ),
                "loa_parm_ci": (
                    "[{lloa_parm_lower:.2f}, {lloa_parm_upper:.2f}], "
                    "[{uloa_parm_lower:.2f}, {uloa_parm_upper:.2f}]"
                ),
                "loa_regr_ci": (
                    "[{loa_intercept_lower:.2f}, {loa_intercept_upper:.2f}], "
                    "[{loa_slope_lower:.2f}, {loa_slope_upper:.2f}]"
                ),
            }
        values = self.summary(ci_method=ci_method)
        values.columns = values.columns.map("_".join)  # Convert MultiIndex columns to Index
        # Add a column of regr agreement so it can be used as variable
        values["loa_regr_agreement"] = loa_regr_agreement

        def format_all_str(row, fstrings_dict):
            return {var: fstr.format(**row) for var, fstr in fstrings_dict.items()}

        all_strings = values.apply(format_all_str, fstrings_dict=fstrings, axis=1).apply(pd.Series)
        if bias_method == "auto":
            bias_parm_idx = self.auto_methods.query("bias == 'parm'").index.tolist()
        elif bias_method == "parm":
            bias_parm_idx = self.sleep_statistics
        elif bias_method == "regr":
            bias_parm_idx = []
        if loa_method == "auto":
            loa_parm_idx = self.auto_methods.query("loa == 'parm'").index.tolist()
        elif loa_method == "parm":
            loa_parm_idx = self.sleep_statistics
        elif loa_method == "regr":
            loa_parm_idx = []
        bias_regr_idx = [ss for ss in self.sleep_statistics if ss not in bias_parm_idx]
        loa_regr_idx = [ss for ss in self.sleep_statistics if ss not in loa_parm_idx]
        bias_parm = all_strings.loc[bias_parm_idx, ["bias_parm", "bias_parm_ci"]]
        bias_regr = all_strings.loc[bias_regr_idx, ["bias_regr", "bias_regr_ci"]]
        bias_parm.columns = bias_parm.columns.str.replace("_parm", "")
        bias_regr.columns = bias_parm.columns.str.replace("_regr", "")
        bias = pd.concat([bias_parm, bias_regr])
        loa_parm = all_strings.loc[loa_parm_idx, ["loa_parm", "loa_parm_ci"]]
        loa_regr = all_strings.loc[loa_regr_idx, ["loa_regr", "loa_regr_ci"]]
        loa_parm.columns = loa_parm.columns.str.replace("_parm", "")
        loa_regr.columns = loa_regr.columns.str.replace("_regr", "")
        loa = pd.concat([loa_parm, loa_regr])
        table = bias.join(loa, validate="1:1").sort_index(axis=0)
        return table

    def summary(self, ci_method="auto"):
        """
        Return a :py:class:`~pandas.DataFrame` that includes all calculated metrics:
        * Parametric bias
        * Parametric lower and upper limits of agreement
        * Regression intercept and slope for modeled bias
        * Regression intercept and slope for modeled limits of agreement
        * Lower and upper confidence intervals for all metrics

        Parameters
        ----------
        ci_method : str
            If ``'parm'`` (i.e., parametric), confidence intervals are always represented using a
            standard t-distribution.
            If ``'boot'`` (i.e., bootstrap), confidence intervals are always represented using a
            bootstrap resampling procedure.
            If  ``'auto'`` (default), confidence intervals are represented using a bootstrap
            resampling procedure for sleep statistics where the distribution of score differences is
            non-normal and using a standard t-distribution otherwise.

        Returns
        -------
        summary : :py:class:`pandas.DataFrame`
            A :py:class:`~pandas.DataFrame` of string representations for bias, limits of agreement,
            and their confidence intervals for all sleep statistics.
        """
        assert isinstance(ci_method, str), "`ci_method` must be a string"
        assert ci_method in self._ci_method_opts, f"`ci_method` must be in {self._ci_method_opts}"
        # Make sure relevant sleep statistics have bootstrapped CIs, and generate them if not
        if ci_method in ["boot", "auto"]:
            if ci_method == "boot":
                sleep_stats_to_boot = self.sleep_statistics
            elif ci_method == "auto":
                sleep_stats_to_boot = self.auto_methods.query("ci == 'boot'").index.tolist()
            # Remove any sleep stats already bootstrapped CIs (eg if "boot" is callaed after "auto")
            sleep_stats_booted = self._ci["boot"].dropna().index
            sleep_stats_to_boot = [s for s in sleep_stats_to_boot if s not in sleep_stats_booted]
            if sleep_stats_to_boot:
                self._generate_bootstrap_ci(sleep_stats=sleep_stats_to_boot)
        if ci_method == "auto":
            parm_idx = self.auto_methods.query("ci == 'parm'").index.to_list()
            boot_idx = [ss for ss in self.sleep_statistics if ss not in parm_idx]
            parm_vals = self._ci.loc[parm_idx, "parm"]
            boot_vals = self._ci.loc[boot_idx, "boot"]
            ci_vals = pd.concat([parm_vals, boot_vals])
        else:
            ci_vals = self._ci[ci_method]
        # Add an extra level to values columns, indicating they are the center interval
        centr_vals = pd.concat({"center": self._vals}, names=["interval"], axis=1).swaplevel(axis=1)
        summary = centr_vals.join(ci_vals, how="left", validate="1:1").astype(float)
        return summary.sort_index(axis=1)

    def calibrate(self, data, bias_method="auto", adjust_all=False):
        """
        Calibrate a :py:class:`~pandas.DataFrame` of sleep statistics from a new scorer based on
        observed biases in ``obs_data``/``obs_scorer``.

        Parameters
        ----------
        data : :py:class:`pandas.DataFrame`
            A :py:class:`pandas.DataFrame` with sleep statistics from an observed scorer.
            Rows are unique observations and columns are unique sleep statistics.
        bias_method : str
            If ``'parm'``, sleep statistics are always adjusted based on parametric bias.
            If ``'regr'``, sleep statistics are always adjusted based on regression-modeled bias.
            If ``'auto'`` (default), bias sleep statistics are adjusted by either ``'parm'`` or
            ``'regr'``, depending on assumption violations.

            .. seealso:: :py:meth:`~yasa.SleepStatsAgreement.summary`

        adjust_all: bool
            If False (default), only adjust values for sleep statistics that showed a statistically
            significant bias in the ``obs_data``. If True, adjust values for all sleep statistics.

        Returns
        -------
        calibrated_data : :py:class:`pandas.DataFrame`
            A :py:class:`~pandas.DataFrame` with calibrated sleep statistics.

        .. seealso:: :py:meth:`~yasa.SleepStatsAgreement.calibrate`
        """
        assert isinstance(data, pd.DataFrame), "`data` must be a pandas DataFrame"
        assert all(
            col in self.sleep_statistics for col in data
        ), f"all columns of `data` must be valid sleep statistics: {self.sleep_statistics}"
        assert isinstance(bias_method, str), "`bias_method` must be a string"
        assert (
            bias_method in self._bias_method_opts
        ), f"`bias_method` must be one of {self._bias_method_opts}"
        assert isinstance(adjust_all, bool), "`adjust_all` must be True or False"
        parm_adjusted = data + self._vals["bias_parm"]
        regr_adjusted = data * self._vals["bias_slope"] + self._vals["bias_intercept"]
        if bias_method == "parm":
            calibrated_data = parm_adjusted
        elif bias_method == "regr":
            calibrated_data = regr_adjusted
        elif bias_method == "auto":
            parm_idx = self.auto_methods.query("bias == 'parm'").index.to_list()
            regr_idx = [ss for ss in self.sleep_statistics if ss not in parm_idx]
            calibrated_data = parm_adjusted[parm_idx].join(regr_adjusted[regr_idx]).dropna(axis=1)
        if not adjust_all:
            # Put the raw values back for sleep stats that don't show statistical bias
            unbiased_sstats = self.assumptions.query("unbiased == True").index.to_list()
            calibrated_data[unbiased_sstats] = data[unbiased_sstats]
        return calibrated_data

    def get_calibration_func(self, sleep_stat):
        """
        Return a function for calibrating a specific sleep statistic, based on observed biases in
        ``obs_data``/``obs_scorer``.

        .. seealso:: :py:meth:`~yasa.SleepStatsAgreement.calibrate`

        Examples
        --------
        >>> ssa = yasa.SleepStatsAgreement(...)
        >>> calibrate_rem = ssa.get_calibration_func("REM")
        >>> new_obs_rem_vals = np.array([50, 40, 30, 20])
        >>> calibrate_rem(new_obs_rem_vals)
        >>> calibrate_rem(new_obs_rem_vals)
        array([50, 40, 30, 20])
        >>> calibrate_rem(new_obs_rem_vals, bias_test=False)
        array([42.825, 32.825, 22.825, 12.825])
        >>> calibrate_rem(new_obs_rem_vals, bias_test=False, method="regr")
        array([ -9.33878878,  -9.86815607, -10.39752335, -10.92689064])
        """
        assert isinstance(sleep_stat, str), "`sleep_stat` must be a string"
        assert sleep_stat in self.sleep_statistics, "`sleep_stat` must be a valid sleep statistic"
        columns = ["bias_parm", "bias_slope", "bias_intercept"]
        parm, slope, intercept = self._vals.loc[sleep_stat, columns]
        auto_method = self.auto_methods.at[sleep_stat, "bias"]
        not_biased = self.assumptions.at[sleep_stat, "unbiased"]

        def calibration_func(x, method="auto", adjust_all=False):
            """Calibrate values for sleep statistic.

            Parameters
            ----------
            x : array
                Values to be calibrated
            method: str
                Method of bias calculation for calibration (``'parm'``, ``'regr'``, or ``'auto'``).
            adjust_all : bool
                If False, only adjust sleep stat if observed bias was statistically significant.

            Returns
            -------
            x_calibrated : :py:class:`numpy.array`
                An array of calibrated x values.
            """
            x = np.asarray(x)
            method = auto_method if method == "auto" else method
            if not_biased and not adjust_all:  # Return input if sleep stat is not statstclly biased
                return x
            elif method == "parm":
                return x + parm
            elif method == "regr":
                return x * slope + intercept

        return calibration_func
