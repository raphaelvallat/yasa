"""
YASA code for evaluating the agreement between two scorers (e.g., human vs YASA), either at the
epoch-by-epoch level or at the level of summary sleep statistics.

Analyses are modeled after the standardized framework proposed in Menghini et al., 2021, SLEEP.
See the following resources:
- https://doi.org/10.1093/sleep/zsaa170
- https://sri-human-sleep.github.io/sleep-trackers-performance
- https://github.com/SRI-human-sleep/sleep-trackers-performance
"""
import logging

import numpy as np
import pandas as pd
import sklearn.metrics as skm
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt

from yasa.plotting import plot_hypnogram


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
    Many steps here are modeled after guidelines proposed in Menghini et al., 2021 [Menghini2021]_.
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
              accuracy  balanced_acc  kappa   mcc  precision  recall  fbeta
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
        from yasa.hypno import Hypnogram  # Avoiding circular import

        assert hasattr(ref_hyps, "__iter__"), "`ref_hyps` must be a an iterable"
        assert hasattr(obs_hyps, "__iter__"), "`obs_hyps` must be a an iterable"
        assert type(ref_hyps) == type(obs_hyps), "`ref_hyps` and `obs_hyps` must be the same type"
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
        ), "all hypnograms must have a scorer name"
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
        ref = pd.concat(
            pd.concat({s: h.as_int()}, names=["sleep_id"]) for s, h in ref_hyps.items()
        )
        obs = pd.concat(
            pd.concat({s: h.as_int()}, names=["sleep_id"]) for s, h in obs_hyps.items()
        )
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
        """Compute multiple agreement scores from a 2-column dataframe.

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
        assert isinstance(scorers, dict)
        assert all(isinstance(k, str) and callable(v) for k, v in scorers.items())
        if df.shape[1] == 3:
            true, pred, weights = zip(*df.values)
        else:
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
        assert (
            isinstance(sample_weight, (type(None), pd.Series))
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
                "fbeta": lambda t, p, w: skm.fbeta_score(
                    t, p, beta=1, average="weighted", sample_weight=w, zero_division=0
                ),
            }
        elif isinstance(scorers, list):
            # Convert the list to a dictionary of sklearn scorers
            scorers = {s: skm.__getattribute__(f"{s}_scorer") for s in scorers}
        # Make a copy of data since weights series might be added to it
        df = self.data.copy()
        if sample_weight is not None:
            assert sample_weight.index == self.data.index, (
                "If not ``None``, ``sample_weight`` Series must be a pandas Series with same index as `self.data`"
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
        scorer = lambda df: skm.precision_recall_fscore_support(
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
                key=lambda x: x.map(lambda y: list(self._yasa2yasa_map.values()).index(y))
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
        assert not ((self.n_sleeps == 1 or sleep_id is not None) and agg_func is not None), (
            "`agg_func` must be None if plotting a single session."
        )
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
        assert not "ax" in ref_kwargs | obs_kwargs, (
            "'ax' can't be supplied to `ref_kwargs` or `obs_kwargs`, use the `ax` keyword instead"
        )
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
        assert self.n_sleeps > 1, (
            "Summary scores can not be computed with only one hypnogram pair."
        )
        assert isinstance(by_stage, bool), "`by_stage` must be True or False"
        if by_stage:
            assert hasattr(self, "_agreement_bystage"), (
                "Must run `self.get_agreement_bystage` before obtaining by_stage summary results."
            )
        else:
            assert hasattr(self, "_agreement"), (
                "Must run `self.get_agreement` before obtaining summary results."
            )
        # Create a function for getting mean absolute deviation
        mad = lambda df: (df - df.mean()).abs().mean()
        mad.__name__ = "mad"  # Pandas uses this lambda attribute to name the aggregated column
        # Merge default and user kwargs
        agg_kwargs = {"func": [mad, "mean", "std", "min", "median", "max"]} | kwargs
        if by_stage:
            summary = (
                self
                .agreement_bystage.groupby("stage")
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
    Evaluate agreement between sleep statistics reported by two different scorers or scoring
    methods.

    Bias and limits-of-agreement (and their confidence intervals) are calcualted for each sleep
    statistic. How these are calculated depends on the sleep statistic's underlying error
    distribution. See [Menghini2021]_ for details, but in brief:

    * Bias: The difference between the two scorers (observed minus reference).
        If sleep-statistic differences (observed minus reference) show proportional bias,
        bias is represented as a regression equation that takes into account changes in bias as
        a function of measurement value. Otherwise, bias is represented as the standard mean
        difference.
    * Limits-of-agreement: If sleep statistic differences show proportional bias, ...
    * Confidence intervals: If sleep statistic differences follow a normal distribution,
        confidence intervals are calculated using standard parametric methods. Otherwise,
        bootstrapped confidence intervals are generated (see also ``bootstrap_cis``).

    Observed sleep statistics can be corrected (i.e., ``calibrated``) to bring them into alignment
    with the sleep statistics from the reference scorer.

    Bias values are calculated as...
    LOA ...
    CI ...


    .. important::
        Bias, limits-of-agreement, and confidence intervals are all calculated differently depending
        on assumption violations. See Menghini et al., 2021 [Menghini2021]_ for details.

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
    alpha : float
        Alpha cutoff used for all assumption tests.

        .. note:: set ``alpha=1`` to ignore all corrections.
    bootstrap_all_cis : bool
        If ``True``, generate all 95% confidence intervals using a bootstrap resampling procedure.
        Otherwise (``False``, default) use the resampling procedure only when discrepancy values
        break normality assumptions.
    verbose : bool or str
        Verbose level. Default (False) will only print warning and error messages. The logging
        levels are 'debug', 'info', 'warning', 'error', and 'critical'. For most users the choice is
        between 'info' (or ``verbose=True``) and warning (``verbose=False``).

    Notes
    -----
    Many steps here are modeled after guidelines proposed in Menghini et al., 2021 [Menghini2021]_.
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
    >>> # For this example, generate two fake datasets of sleep statistics
    >>> hypsA = [yasa.simulate_hypnogram(tib=600, scorer="Ref", seed=i) for i in range(20)]
    >>> hypsB = [h.simulate_similar(tib=600, scorer="Test", seed=i) for i, h in enumerate(hypsA)]
    >>> # sstatsA = pd.Series(hypsA).map(lambda h: h.sleep_statistics()).apply(pd.Series)
    >>> # sstatsB = pd.Series(hypsB).map(lambda h: h.sleep_statistics()).apply(pd.Series)
    >>> # sstatsA.index = sstatsB.index = sstatsA.index.map(lambda x: f"sub-{x+1:03d}")
    >>> ebe = yasa.EpochByEpochEvaluation(hypsA, hypsB)
    >>> sstats = ebe.get_sleepstats()
    >>> sstatsA = sstats.loc["Ref"]
    >>> sstatsB = sstats.loc["Test"]
    >>>
    >>> sse = yasa.SleepStatsAgreement(sstatsA, sstatsB)
    >>>
    >>> sse.summary()
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
        ref_data,
        obs_data,
        *,
        ref_scorer="Reference",
        obs_scorer="Observed",
        alpha=0.05,
        bootstrap_all_cis=False,
        verbose=True,
    ):

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
        assert isinstance(alpha, float) and 0 <= alpha <= 1, "`alpha` must be a number between 0 and 1, inclusive"
        assert isinstance(bootstrap_all_cis, bool), "`bootstrap_all_cis` must be True or False"

        # If `ref_data` and `obs_data` indices are unnamed, name them
        session_key = "session_id" if ref_data.index.name is None else ref_data.index.name
        ref_data.index.name = session_key
        obs_data.index.name = session_key

        # Get scorer differences (i.e., observed minus reference)
        diff_data = obs_data.sub(ref_data)

        # Prepend a "scorer" level to index of each individual dataframe, making MultiIndex
        obs_data = pd.concat({obs_scorer: obs_data}, names=["scorer"])
        ref_data = pd.concat({ref_scorer: ref_data}, names=["scorer"])
        diff_data = pd.concat({"difference": diff_data}, names=["scorer"])
        # Merge observed data, reference data, and differences
        data = pd.concat([obs_data, ref_data, diff_data])
        # Reshape to long-format with 3 columns (observed, reference, difference)
        data = (
            data.melt(var_name="sleep_stat", ignore_index=False)
            .reset_index()
            .pivot(columns="scorer", index=["sleep_stat", session_key], values="value")
            .rename_axis(columns=None)
            .sort_index()
        )

        # Remove sleep statistics that have no differences between scorers
        stats_with_nodiff = diff_data.any().loc[lambda x: ~x].index.tolist()
        data = data.query(f"~sleep_stat.isin({stats_with_nodiff})")
        for s in stats_with_nodiff:
            logger.warning(f"Removed {s} from evaluation because all scorings were identical.")

        ########################################################################
        # TEST ASSUMPTION VIOLATIONS
        ########################################################################

        grouper = data.groupby("sleep_stat")  # For convenience

        # Test SYSTEMATIC BIAS between the two scorers for each sleep statistic (do means differ?).
        # This test is used to determine whether corrections are applied during calibration only.
        systematic_bias = grouper["difference"].apply(pg.ttest, y=0).droplevel(-1)

        # Test NORMALITY of difference values at each sleep statistic.
        # This test is used to determine how confidence intervals for Bias and LoA are calculated.
        normality = grouper["difference"].apply(pg.normality, alpha=alpha).droplevel(-1)

        # Test PROPORTIONAL BIAS at each sleep statistic (do scorer diffs vary as with ref measure?)
        # This test is used to determine how Bias and LoA are calculated.
        regr_f = lambda df: pg.linear_regression(df[ref_scorer], df[obs_scorer], alpha=alpha)
        resid_f = lambda df: pd.Series(regr_f(df).residuals_, index=df.index.get_level_values(1))
        proportional_bias = grouper.apply(regr_f).droplevel(-1).set_index("names", append=True)
        proportional_bias = proportional_bias.swaplevel().sort_index()
        residuals = grouper.apply(resid_f).stack().rename("residual")

        # Test HETEROSCEDASTICITY at each sleep statistic.
        # This test is used to determine how LoAs are calculated.
        data = data.join(residuals)
        homosc_columns = [ref_scorer, "difference", "residual"]
        homosc_f = lambda df: pg.homoscedasticity(df[homosc_columns], alpha=alpha)
        heteroscedasticity = data.groupby("sleep_stat").apply(homosc_f).droplevel(-1)
        # Add same test for log-transformed values, also used for determining LoA calculation method
        log_transform = lambda x: np.log(x + 1e-6)
        backlog_transform = lambda x: np.exp(x) - 1e-6
        logdata = data[[ref_scorer, obs_scorer]].applymap(log_transform)
        logdata["difference"] = logdata[obs_scorer].sub(logdata[ref_scorer])
        logdata["residual"] = logdata.groupby("sleep_stat").apply(resid_f).stack()#.rename("residual")
        heteroscedasticity_log = logdata.groupby("sleep_stat").apply(homosc_f).droplevel(-1)
        # data_exp = logdata[[ref_scorer, obs_scorer, "difference"]].applymap(backlog_transform)
        # data_exp = logdata["difference"].map(backlog_transformer)

        # Aggregate test results into a dataframe of True/False for later convenience.
        violations = (
            systematic_bias["p-val"].lt(alpha).to_frame("is_systematically_biased")
            .join(~normality["normal"].rename("is_nonnormal"))
            .join(proportional_bias.loc[ref_scorer, "pval"].lt(alpha).rename("is_proportionally_biased"))
            .join(~heteroscedasticity["equal_var"].rename("is_heteroscedastic"))
            .join(~heteroscedasticity_log["equal_var"].rename("is_log_heteroscedastic"))
        )

        # Get name of method for each calculation.
        # CI - standard or bootstrap
        # Bias - standard or modeled
        # LoA - standard, log_standard, modeled, or residuals
        get_ci_method = lambda row: "bootstrap" if row.is_nonnormal else "standard"
        get_bias_method = lambda row: "modeled" if row.is_proportionally_biased else "standard"
        get_loa_method = lambda row: (
            "modeled" if row.is_log_heteroscedastic else "log_standard"
        ) if row.is_heteroscedastic else (
            "residuals" if row.is_proportionally_biased else "standard"
        )
        methods = {
            "loa": violations.apply(get_loa_method, axis=1),
            "bias": violations.apply(get_bias_method, axis=1),
            "ci": violations.apply(get_ci_method, axis=1),
        }
        methods = pd.DataFrame(methods)
        if bootstrap_all_cis:
            methods["ci"] = ["standard"] * len(violations)

        ########################################################################
        # ATTRIBUTES
        ########################################################################

        self._ref_scorer = ref_scorer
        self._obs_scorer = obs_scorer
        self._n_sessions = data.index.get_level_values(session_key).nunique()
        self._data = data
        self._diff_data = diff_data.droplevel(0).drop(columns=stats_with_nodiff)
        self._systematic_bias = systematic_bias
        self._normality = normality
        self._proportional_bias = proportional_bias
        self._heteroscedasticity = heteroscedasticity
        self._violations = violations
        self._methods = methods
        # self._bias = bias
        # self._bias_vars = bias_vars
        # self._loas = loas
        # self._loas_vars = loas_vars


    @property
    def data(self):
        """A :py:class:`pandas.DataFrame` containing all sleep statistics from ``ref_data`` and
        ``obs_data`` as well as their difference scores (``obs_data`` minus ``ref_data``).
        """
        return self._data

    @property
    def methods(self):
        return self._methods

    @property
    def biased(self):
        return self._biased

    @property
    def discrepancies(self):
        """A :py:class:`pandas.DataFrame` of ``obs_data`` minus ``ref_data``."""
        # # Pivot for session-rows and statistic-columns
        return self._discrepancies

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
        """The number of sleep sessions."""
        return self._n_sessions

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
        """A :py:class:`pandas.DataFrame` of proportional bias results for all sleep statistics."""
        return self._proportional_bias

    def __repr__(self):
        # TODO v0.8: Keep only the text between < and >
        return (
            f"<SleepStatsAgreement | Observed scorer ('{self.obs_scorer}') evaluated against "
            f"reference scorer ('{self.ref_scorer}'), {self.n_sessions} sleep sessions>\n"
            " - Use `.summary()` to get pass/fail values from various checks\n"
            " - Use `.plot_blandaltman()` to get a Bland-Altman-plot grid for sleep statistics\n"
            "See the online documentation for more details."
        )

    def __str__(self):
        return __repr__()

    @staticmethod
    def _get_standard_bias(x):
        """Wrapper around `np.mean`, for organizational purposes. For internal use."""
        return x.mean()

    @staticmethod
    def _get_standard_loas(x, agreement=1.96, std=None):
        """Return standard lower and upper limits of agreement. For internal use only.

        Parameters
        ----------
        x : array_like
        agreement : float, int
        std : float, int

        Returns
        -------
        loas : py:class:`numpy.ndarray`
            A numpy array of shape (2,) where lower LoA is first and upper LoA is second.
        """
        if std is None:
            std = x.std()
        return x.mean() + np.array([-agreement, agreement]) * std

    @staticmethod
    def _get_regression_coefficient(x, y, index):
        """Run linear regression and return a single coefficient.
        
        A wrapper to aid in computing CIs (with pg.compute_bootci). For internal use only.

        Parameters
        ----------
        x : array_like
            Predictor values
        y : array_like
            Outcome values
        index: int
            0 to get coefficient of intercept, N to get coefficient of Nth predictor

        Returns
        -------
        coef: float
            Regression coefficient of the effect of `b`.
        """
        ## Q: Jump straight to np.lstsq for speed?
        return pg.linear_regression(x, y, add_intercept=True).at[index, "coef"]

    @staticmethod
    def _get_standard_bias_ci(x, confidence=0.95):
        """Return standard confidence intervals for bias."""
        n = x.size
        dof = x.size - 1
        avg = x.mean()
        std = x.std()
        sem = np.sqrt(std**2 / n)
        low, high = stats.t.interval(confidence, dof, loc=avg, scale=sem)
        return low, high

    @staticmethod
    def _get_standard_loas_cis(x, agreement=1.96, std=None, confidence=0.95):
        """Return standard confidence intervals for both lower LoA and upper LoA.

        Parameters
        ----------
        x : array_like
        agreement : float, int
        std : float, int
        confidence : float

        Returns
        -------
        cis : dict
            A dictionary of length 2, with keys "lower" and "upper" LoA, and values of tuples
            containing "lower" and "upper" confidence intervals for each.
        """
        n = x.size
        dof = x.size - 1
        if std is None:
            std = x.std()
        lower, upper = DiscrepancyEvaluation._get_standard_loas(x, agreement)
        sem = np.sqrt(3 * std**2 / n)
        lower_lo, lower_hi = stats.t.interval(confidence, dof, loc=lower, scale=sem)
        upper_lo, upper_hi = stats.t.interval(confidence, dof, loc=upper, scale=sem)
        return {"lower": (lower_lo, lower_hi), "upper": (upper_lo, upper_hi)}

    def get_bias(self, alpha=0.05, **bootci_kwargs):
        results = []
        for sstat, row in self.methods.iterrows():
            # Extract difference values once for convenience.
            diffs = self.data.loc[sstat, "difference"].to_numpy()

            # Identify the method that will be used.
            if self._violations.at[sstat, "is_proportionally_biased"]:
                bias_method = "modeled"
            else:
                bias_method = "standard"

            if self._violations.at[sstat, "is_nonnormal"]:
                ci_method = "bootstrap"
            else:
                ci_method = "standard"

            # Initialize dictionary to hold row information.
            metadata = {"sleep_stat": sstat, "method": bias_method}

            # Calculate necessary variables to get bias (either bias or b0 and b1).
            if bias_method == "modeled":
                # Systematic bias and constant bias present, model based on constant bias regression.
                # x, y = self.data.loc[sstat, [self.ref_scorer, "difference"]].T.to_numpy()
                ref = self.data.loc[sstat, self.ref_scorer].to_numpy()
                b0 = self._get_regression_coefficient(ref, diffs, index=0)
                b1 = self._get_regression_coefficient(ref, diffs, index=1)
                # Confidence intervals for b0 and b1
                if ci_method == "bootstrap":
                    b0_lo, b0_hi = pg.compute_bootci(
                        ref,
                        diffs,
                        func=lambda x, y: self._get_regression_coefficient(x, y, index=0),
                        **bootci_kwargs,
                    )
                    b1_lo, b1_hi = pg.compute_bootci(
                        ref,
                        diffs,
                        func=lambda x, y: self._get_regression_coefficient(x, y, index=1),
                        **bootci_kwargs,
                    )
                elif ci_method == "standard":
                    col1 = "CI[{:.1f}%]".format((1 - alpha / 2) * 100) 
                    col2 = "CI[{:.1f}%]".format(alpha / 2 * 100) 
                    b0_lo, b0_hi, b1_lo, b1_hi = pg.linear_regression(
                        ref, diffs, alpha=alpha
                    ).loc[[0, 1], [col1, col2]].to_numpy().flatten()

            elif bias_method == "standard":
                b0 = self._get_standard_bias(diffs)
                if ci_method == "bootstrap":
                    b0_lo, b0_hi = pg.compute_bootci(
                        diffs, func=self._get_standard_bias, **bootci_kwargs
                    )
                elif ci_method == "standard":
                    b0_lo, b0_hi = self._get_standard_bias_ci(diffs)
            else:
                raise ValueError(f"Unexpected bias method {bias_method}.")

            results.append(dict(variable="b0", mean=b0, ci_lower=b0_lo, ci_upper=b0_hi, **metadata))
            if bias_method == "modeled":
                results.append(dict(variable="b1", mean=b1, ci_lower=b1_lo, ci_upper=b1_hi, **metadata))

        df = pd.json_normalize(results).set_index(["method", "sleep_stat", "variable"]).sort_index()
        self._bias_values = df

    def get_loa(self, alpha=0.05, **bootci_kwargs):
        results = []
        for sstat, row in self.methods.iterrows():
            # Extract difference values once for convenience.
            diffs = self.data.loc[sstat, "difference"].to_numpy()

            # Identify the method that will be used.
            if self._violations.at[sstat, "is_heteroscedastic"]:
                if self._violations.at[sstat, "is_log_heteroscedastic"]:
                    loa_method = "modeled"
                else:
                    loa_method = "log_standard"
            else:
                if self._violations.at[sstat, "is_proportionally_biased"]:
                    loa_method = "residuals"
                else:
                    loa_method = "standard"

            if self._violations.at[sstat, "is_nonnormal"]:
                ci_method = "bootstrap"
            else:
                ci_method = "standard"

            metadata = {"sleep_stat": sstat, "method": loa_method}
            if loa_method in ["standard", "residuals"]:
                # Get standard deviation of calibrated (i.e., bias-adjusted) observed values
                # calibration_func = lambda x: x - (b0 + b1 * x)  # b0 and b1 were generated this iteration above
                # Get standard deviation of residuals?
                if loa_method == "residuals":
                    std = self.data.loc[sstat, "residual"].std()
                else:
                    std = diffs.std()  # dof=1
                lower, upper = self._get_standard_loas(diffs, std=std)
                if ci_method == "bootstrap":
                    lower_lo, lower_hi = pg.compute_bootci(diffs, func=lambda x: self._get_standard_loas(x, std=std)[0], **bootci_kwargs)
                    upper_lo, upper_hi = pg.compute_bootci(diffs, func=lambda x: self._get_standard_loas(x, std=std)[1], **bootci_kwargs)
                elif ci_method == "standard":
                    cis = self._get_standard_loas_cis(diffs, std=std)
                    lower_lo, lower_hi = cis["lower"]
                    upper_lo, upper_hi = cis["upper"]

                results.append(dict(variable="lower", mean=lower, ci_lower=lower_lo, ci_upper=lower_hi, **metadata))
                results.append(dict(variable="upper", mean=upper, ci_lower=upper_lo, ci_upper=upper_hi, **metadata))
            elif loa_method == "modeled":
                x, y = self.data.loc[sstat, [obs_scorer, "residual"]].T.values
                c0 = self._get_regression_coefficient(x, y, index=0)
                c1 = self._get_regression_coefficient(x, y, index=1)
                if ci_method == "bootstrap":
                    c0_lo, c0_hi = pg.compute_bootci(x, y, func=lambda x, y: self._get_regression_coefficient(x, y, index=0), **ci_kwargs)
                    c1_lo, c1_hi = pg.compute_bootci(x, y, func=lambda x, y: self._get_regression_coefficient(x, y, index=1), **ci_kwargs)
                elif ci_method == "standard":
                    col1 = "CI[{:.1f}%]".format((1 - alpha / 2) * 100) 
                    col2 = "CI[{:.1f}%]".format(alpha / 2 * 100) 
                    c0_lo, c0_hi, c1_lo, c1_hi = pg.linear_regression(
                        x, y, alpha=alpha
                    ).loc[[0, 1], [col1, col2]].to_numpy().flatten()
                else:
                    raise ValueError(f"Unknown CI method {ci_method}.")
                results.append(dict(variable="c0", mean=lower, ci_lower=lower_lo, ci_upper=lower_hi, **metadata))
                results.append(dict(variable="c1", mean=upper, ci_lower=upper_lo, ci_upper=upper_hi, **metadata))
            else:
                raise ValueError(f"Unexpected LoA method {loa_method}.")
        df = pd.json_normalize(results).set_index(["method", "sleep_stat", "variable"]).sort_index()
        self._loa_values = df

    def get_text_summary(self, fmt_dict=None):
        """
        """
        results = {}
        # Bias
        for (meth, sstat), df in self._bias_values.groupby(["method", "sleep_stat"]):
            if meth == "standard":
                fstr = "{mean:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]"
                bias = df.droplevel([0,1]).apply(lambda r: fstr.format(**r), axis=1).loc["b0"]
            elif meth == "modeled":
                fstr = "{b0_mean:.2f} [{b0_ci_lower:.2f}, {b0_ci_upper:.2f}] + {b1_mean:.2f} [{b1_ci_lower:.2f}, {b1_ci_upper:.2f}] x ref"
                temp = df.unstack("variable").swaplevel(axis=1)
                temp.columns = temp.columns.map("_".join)
                bias = temp.apply(lambda r: fstr.format(**r), axis=1)[0]
            results[sstat] = dict(bias=bias)
        # LoA
        for (meth, sstat), df in self._loa_values.groupby(["method", "sleep_stat"]):
            if meth in ["standard", "residuals"]:
                fstr = "{mean:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]"
                lower, upper = df.droplevel([0,1]).apply(lambda r: fstr.format(**r), axis=1).loc[["lower", "upper"]]
            else:
                fstr = "{c0_mean:.2f} [{c0_ci_lower:.2f}, {c0_ci_upper:.2f}] + {c1_mean:.2f} [{c1_ci_lower:.2f}, {c1_ci_upper:.2f}] x ref"
                temp = df.unstack("variable").swaplevel(axis=1)
                temp.columns = temp.columns.map("_".join)
                lower = temp.apply(lambda r: fstr.format(**r), axis=1)[0]
                upper = lower.copy()
            results[sstat].update({"lower": lower, "upper": upper})

        df = pd.DataFrame(results).T.rename_axis("sleep_stat")
        return df

    def summary(self, **kwargs):
        """Return a summary dataframe highlighting whether tests passed for each sleep statistic.

        Parameters
        ----------
        self : :py:class:`yasa.SleepStatsAgreement`
            A :py:class:`yasa.SleepStatsAgreement` instance.
        **kwargs : key, value pairs
            Additional keyword arguments are passed to :py:meth:`pandas.DataFrame.groupby.agg`.

            >>> ssa.summary(func=["mean", "sem", "min", "max"])

        Returns
        -------
        summary : :py:class:`pandas.DataFrame`
            A :py:class:`pandas.DataFrame` with boolean values indicating the pass/fail status for
            normality, proportional bias, and homoscedasticity tests (for each sleep statistic).
        """
        series_list = [
            self.bias["biased"],
            self.normality["normal"],
            self.proportional_bias["bias_constant"],
            self.homoscedasticity["equal_var"].rename("homoscedastic"),
        ]
        summary = pd.concat(series_list, axis=1)
        mad = lambda df: (df - df.mean()).abs().mean()
        mad.__name__ = "mad"  # Pandas uses this to name the aggregated column
        agg_kwargs = {"func": [mad, "mean", "std"]} | kwargs
        desc = self.data.groupby("sleep_stat").agg(**agg_kwargs)
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
            sleep_stats = self.data.index.get_level_values("sleep_stat").unique()
        heatmap_kwargs = {"cmap": "binary", "annot": True, "fmt": ".1f", "square": False}
        heatmap_kwargs["cbar_kws"] = dict(label="Normalized discrepancy %")
        if "cbar_kws" in kwargs:
            heatmap_kwargs["cbar_kws"].update(kwargs["cbar_kws"])
        heatmap_kwargs.update(kwargs)
        table = self._diff_data[sleep_stats]
        # Normalize statistics (i.e., columns) between zero and one then convert to percentage
        table_norm = table.sub(table.min(), axis=1).div(table.apply(np.ptp)).multiply(100)
        if heatmap_kwargs["annot"]:
            # Use raw values for writing
            heatmap_kwargs["annot"] = table.to_numpy()
        return sns.heatmap(table_norm, **heatmap_kwargs)

    def plot_discrepancies_dotplot(self, pairgrid_kwargs={"palette": "winter"}, **kwargs):
        """Visualize session-level discrepancies, generally for outlier inspection.

        Parameters
        ----------
        pairgrid_kwargs : dict
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
        assert isinstance(pairgrid_kwargs, dict), "`pairgrid_kwargs` must be a dict"
        kwargs_stripplot = {"size": 10, "linewidth": 1, "edgecolor": "white"}
        kwargs_stripplot.update(kwargs)
        # Initialize the PairGrid
        height = 0.3 * len(self._diff_data)
        aspect = 0.6
        kwargs_pairgrid = dict(hue=self.sleep_id_str, height=height, aspect=aspect)
        kwargs_pairgrid.update(pairgrid_kwargs)
        g = sns.PairGrid(
            self._diff_data.reset_index(), y_vars=[self.sleep_id_str], **kwargs_pairgrid
        )
        # Draw the dots
        g.map(sns.stripplot, orient="h", jitter=False, **kwargs_stripplot)
        # Adjust aesthetics
        for ax in g.axes.flat:
            ax.set(title=ax.get_xlabel())
            ax.margins(x=0.3)
            ax.yaxis.grid(True)
            ax.tick_params(left=False)
        g.set(xlabel="", ylabel="")
        sns.despine(left=True, bottom=True)
        return g

    def plot_blandaltman(self, facetgrid_kwargs={}, **kwargs):
        """

        **Use col_order=sstats_order for plotting a subset.

        Parameters
        ----------
        facetgrid_kwargs : dict
            Keyword arguments passed to the :py:class:`seaborn.FacetGrid` call.
        **kwargs : key, value pairs
            Additional keyword arguments are passed to :py:func:`pingouin.plot_blandaltman`.

        Returns
        -------
        g : :py:class:`seaborn.FacetGrid`
            A :py:class:`seaborn.FacetGrid` with sleep statistics Bland-Altman plots on each axis.
        """
        kwargs_facetgrid = dict(col_wrap=4, height=2, aspect=1, sharex=False, sharey=False)
        kwargs_facetgrid.update(facetgrid_kwargs)
        kwargs_blandaltman = dict(xaxis="y", annotate=False, edgecolor="black", facecolor="none")
        kwargs_blandaltman.update(kwargs)
        # Initialize a grid of plots with an Axes for each sleep statistic
        g = sns.FacetGrid(self.data.reset_index(), col="sleep_stat", **kwargs_facetgrid)
        # Draw Bland-Altman plot on each axis
        g.map(pg.plot_blandaltman, self.obs_scorer, self.ref_scorer, **kwargs_blandaltman)
        # Adjust aesthetics
        for ax in g.axes.flat:
            # Tidy-up axis limits with symmetric y-axis and minimal ticks
            bound = max(map(abs, ax.get_ylim()))
            ax.set_ylim(-bound, bound)
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=2, integer=True, symmetric=True))
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=1, integer=True))
        ylabel = " - ".join((self.obs_scorer, self.ref_scorer))
        g.set_ylabels(ylabel)
        g.set_titles(col_template="{col_name}")
        g.tight_layout(w_pad=1, h_pad=2)
        return g
