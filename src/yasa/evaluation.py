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
import warnings

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

    .. warning:: **Experimental** — the API of this class may change before the full release
        planned for v0.8.0. Use with caution.

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
            It is assumed that the order of hypnograms are the same in ``ref_hyps`` and
            ``obs_hyps``. For example, the third hypnogram in ``ref_hyps`` and ``obs_hyps`` must
            come from the same sleep session, and they must only differ in that they have different
            scorers.

        .. seealso:: For comparing just two hypnograms, use :py:meth:`yasa.Hypnogram.evaluate`.

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
    .. [Efron1987] Efron, B. (1987). Better bootstrap confidence intervals. Journal of the
                   American Statistical Association, 82(397), 171-185.
                   https://doi.org/10.1080/01621459.1987.10478410

    Examples
    --------
    >>> import yasa
    >>> ref_hyps = [yasa.simulate_hypnogram(tib=600, scorer="Human", seed=i) for i in range(10)]
    >>> obs_hyps = [h.simulate_similar(scorer="YASA", seed=i) for i, h in enumerate(ref_hyps)]
    >>> ebe = yasa.EpochByEpochAgreement(ref_hyps, obs_hyps)
    >>> agr = ebe.get_agreement()
    >>> agr.head(5).round(1)
              accuracy  balanced_acc  kappa  mcc  precision    f1
    sleep_id
    1             30.6          26.0    0.1  0.1       30.6  30.6
    2             33.3          32.7    0.1  0.1       34.9  33.5
    3             35.1          23.9    0.1  0.1       34.8  34.6
    4             22.5          21.4    0.0  0.0       20.6  20.5
    5             21.4          16.9   -0.1 -0.1       20.3  20.7

    >>> ebe.get_agreement_bystage().head(12).round(1)  # doctest: +SKIP
                    fbeta    npv  precision  recall  specificity  support
    stage sleep_id
    WAKE  1            39.1    ...       37.1    41.3          ...    189.0
          2            29.9    ...       27.6    32.6          ...    184.0
          ...
    N1    1            18.5    ...       18.5    18.5          ...    124.0
          2            12.1    ...       13.1    11.2          ...    160.0

    >>> ebe.get_confusion_matrix(sleep_id=1)
    YASA   WAKE  N1   N2  N3  REM
    Human
    WAKE     78  24   50   3   34
    N1       23  23   43  15   20
    N2       60  58  183  43  139
    N3       30  10   50   5   32
    REM      19   9  121  50   78

    .. plot::

        >>> import yasa
        >>> import matplotlib.pyplot as plt
        >>> ref_hyps = [yasa.simulate_hypnogram(tib=600, scorer="Human", seed=i) for i in range(10)]
        >>> obs_hyps = [h.simulate_similar(scorer="YASA", seed=i) for i, h in enumerate(ref_hyps)]
        >>> ebe = yasa.EpochByEpochAgreement(ref_hyps, obs_hyps)
        >>> fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
        >>> _ = ebe.plot_hypnograms(sleep_id=10)

    .. plot::

        >>> import yasa
        >>> import matplotlib.pyplot as plt
        >>> ref_hyps = [yasa.simulate_hypnogram(tib=600, scorer="Human", seed=i) for i in range(10)]
        >>> obs_hyps = [h.simulate_similar(scorer="YASA", seed=i) for i, h in enumerate(ref_hyps)]
        >>> ebe = yasa.EpochByEpochAgreement(ref_hyps, obs_hyps)
        >>> fig, ax = plt.subplots(figsize=(6, 3))
        >>> _ = ebe.plot_hypnograms(
        ...     sleep_id=8, ax=ax, obs_kwargs={"color": "red", "lw": 2, "ls": "dotted"}
        ... )
        >>> plt.tight_layout()

    .. plot::

        >>> import yasa
        >>> import matplotlib.pyplot as plt
        >>> ref_hyps = [yasa.simulate_hypnogram(tib=600, scorer="Human", seed=i) for i in range(10)]
        >>> obs_hyps = [h.simulate_similar(scorer="YASA", seed=i) for i, h in enumerate(ref_hyps)]
        >>> ebe = yasa.EpochByEpochAgreement(ref_hyps, obs_hyps)
        >>> session = 8
        >>> fig, ax = plt.subplots(figsize=(6.5, 2.5), constrained_layout=True)
        >>> style_a = dict(alpha=1, lw=2.5, ls="solid", color="gainsboro", label="Michel")
        >>> style_b = dict(alpha=1, lw=2.5, ls="solid", color="cornflowerblue", label="Jouvet")
        >>> legend_style = dict(
        ...     title="Scorer", frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.9)
        ... )
        >>> ax = ebe.plot_hypnograms(
        ...     sleep_id=session, ref_kwargs=style_a, obs_kwargs=style_b, legend=legend_style, ax=ax
        ... )
        >>> acc = ebe.get_agreement().at[session, "accuracy"]
        >>> _ = ax.text(
        ...     0.01, 1, f"Accuracy = {acc:.0f}%", ha="left", va="bottom", transform=ax.transAxes
        ... )

    When comparing only 2 hypnograms, use the :py:meth:`~yasa.Hypnogram.evaluate` method:

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
        from .hypno import Hypnogram  # Avoiding circular import, bc hypno imports this class

        assert hasattr(ref_hyps, "__iter__"), "`ref_hyps` must be a an iterable"
        assert hasattr(obs_hyps, "__iter__"), "`obs_hyps` must be a an iterable"
        assert type(ref_hyps) is type(obs_hyps), "`ref_hyps` and `obs_hyps` must be the same type"
        assert len(ref_hyps) == len(obs_hyps), (
            "`ref_hyps` and `obs_hyps` must have the same number of hypnograms"
        )

        if isinstance(ref_hyps, dict):
            # If user provides dictionaries, split into sleep IDs and hypnograms
            assert ref_hyps.keys() == obs_hyps.keys(), (
                "keys in `ref_hyps` must be the same as keys in `obs_hyps`"
            )
            sleep_ids, ref_hyps = zip(*ref_hyps.items())
            obs_hyps = tuple(obs_hyps.values())
        else:
            # Create hypnogram_ids
            sleep_ids = tuple(range(1, 1 + len(ref_hyps)))

        assert all(isinstance(hyp, Hypnogram) for hyp in ref_hyps + obs_hyps), (
            "`ref_hyps` and `obs_hyps` must only contain YASA hypnograms"
        )
        assert all(h.scorer is not None for h in ref_hyps + obs_hyps), (
            "all hypnograms in `ref_hyps` and `obs_hyps` must have a scorer name"
        )
        for h1, h2 in zip((ref_hyps + obs_hyps)[:-1], (ref_hyps + obs_hyps)[1:]):
            assert h1.freq == h2.freq, "all hypnograms must have the same freq"
            assert h1.labels == h2.labels, "all hypnograms must have the same labels"
            assert h1.mapping == h2.mapping, "all hypnograms must have the same mapping"
            assert h1.n_stages == h2.n_stages, "all hypnograms must have the same n_stages"
        assert all(h1.scorer == h2.scorer for h1, h2 in zip(ref_hyps[:-1], ref_hyps[1:])), (
            "all `ref_hyps` must have the same scorer"
        )
        assert all(h1.scorer == h2.scorer for h1, h2 in zip(obs_hyps[:-1], obs_hyps[1:])), (
            "all `obs_hyps` must have the same scorer"
        )
        assert all(h1.scorer != h2.scorer for h1, h2 in zip(ref_hyps, obs_hyps)), (
            "each `ref_hyps` and `obs_hyps` pair must have unique scorers"
        )
        assert all(h1.n_epochs == h2.n_epochs for h1, h2 in zip(ref_hyps, obs_hyps)), (
            "each `ref_hyps` and `obs_hyps` pair must have the same n_epochs"
        )
        # Convert ref_hyps and obs_hyps to dictionaries with sleep_id keys and hypnogram values
        ref_hyps = {s: h for s, h in zip(sleep_ids, ref_hyps)}
        obs_hyps = {s: h for s, h in zip(sleep_ids, obs_hyps)}

        # Merge all hypnograms into a single MultiIndexed dataframe
        ref = pd.concat(pd.concat({s: h.as_int()}, names=["sleep_id"]) for s, h in ref_hyps.items())
        obs = pd.concat(pd.concat({s: h.as_int()}, names=["sleep_id"]) for s, h in obs_hyps.items())
        data = pd.concat([ref, obs], axis=1)

        # Generate some mapping dictionaries to be used later in class methods
        skm_labels = np.unique(data).tolist()  # all unique YASA integer codes in this hypno
        skm2yasa_map = {i: lab for i, lab in enumerate(skm_labels)}  # skm order to YASA integers
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
        s = "s" if self.n_sessions > 1 else ""
        return (
            f"<EpochByEpochAgreement | Observed hypnogram{s} scored by {self.obs_scorer} "
            f"evaluated against reference hypnogram{s} scored by {self.ref_scorer}, "
            f"{self.n_sessions} sleep session{s}>\n"
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
    def n_sessions(self):
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
        assert all(isinstance(k, str) and callable(v) for k, v in scorers.items()), (
            "Each key of `scorers` must be a string, and each value must be a callable function"
        )
        if df.shape[1] == 3:
            true, pred, weights = zip(*df.values)
        elif df.shape[1] == 2:
            true, pred = zip(*df.values)  # Same as (df["col1"], df["col2"]) but teensy bit faster
            weights = None
        scores = {s: f(true, pred, weights) for s, f in scorers.items()}
        return scores

    def get_agreement(self, sample_weight=None, scorers=None, pooled=False):
        """
        Return a :py:class:`pandas.DataFrame` of weighted (i.e., averaged) agreement scores.

        Parameters
        ----------
        sample_weight : None or :py:class:`pandas.Series`
            Sample weights passed to underlying :py:mod:`sklearn.metrics` functions where possible.
            If a :py:class:`pandas.Series`, the index must match exactly that of
            :py:attr:`~yasa.EpochByEpochAgreement.data`.
        scorers : None, list, or dictionary
            The scorers to be used for evaluating agreement. If None (default), default scorers are
            used. If a list of strings, each ``name`` is mapped to the
            ``sklearn.metrics.<name>_score`` function (e.g. ``"accuracy"``, ``"cohen_kappa"``),
            called with ``sample_weight``; metrics that require extra arguments (e.g. ``average``
            for ``precision``) must be passed as a dictionary instead. If a dictionary, keys are
            scorer names (str) and values are functions taking 3 positional arguments (true values,
            predicted values, and sample weights).
        pooled : bool
            If False (default), agreement scores are computed per session and returned as a
            :py:class:`~pandas.DataFrame` with one row per session. If True, all epochs across all
            sessions are pooled before computing a single set of agreement scores, returned as a
            :py:class:`~pandas.Series`.

        Returns
        -------
        agreement : :py:class:`pandas.DataFrame` or :py:class:`pandas.Series`
            If ``pooled=False``, a :py:class:`~pandas.DataFrame` with agreement metrics as columns
            and sessions as rows. If ``pooled=True``, a :py:class:`~pandas.Series` with agreement
            metrics as index.

            With the default scorers, the proportion-based metrics (``accuracy``,
            ``balanced_acc``, ``precision``, ``f1``) are expressed as percentages (0-100).
            ``kappa`` and ``mcc`` are correlation-like coefficients ranging from -1 to 1.
            Custom ``scorers`` are returned as is.
        """
        assert isinstance(sample_weight, (type(None), pd.Series)), (
            "`sample_weight` must be None or pandas Series"
        )
        assert isinstance(pooled, bool), "`pooled` must be True or False"
        assert isinstance(scorers, (type(None), list, dict))
        if isinstance(scorers, list):
            assert all(isinstance(x, str) for x in scorers)
        elif isinstance(scorers, dict):
            assert all(isinstance(k, str) and callable(v) for k, v in scorers.items())
        if scorers is None:
            # Create dictionary of default scorer functions
            # Proportion-based metrics are expressed as percentages; kappa and mcc are not.
            scorers = {
                "accuracy": lambda t, p, w: (
                    100 * skm.accuracy_score(t, p, normalize=True, sample_weight=w)
                ),
                "balanced_acc": lambda t, p, w: (
                    100 * skm.balanced_accuracy_score(t, p, adjusted=False, sample_weight=w)
                ),
                "kappa": lambda t, p, w: skm.cohen_kappa_score(
                    t, p, labels=None, weights=None, sample_weight=w
                ),
                "mcc": lambda t, p, w: skm.matthews_corrcoef(t, p, sample_weight=w),
                "precision": lambda t, p, w: (
                    100
                    * skm.precision_score(
                        t, p, average="weighted", sample_weight=w, zero_division=0
                    )
                ),
                "f1": lambda t, p, w: (
                    100 * skm.f1_score(t, p, average="weighted", sample_weight=w, zero_division=0)
                ),
            }
        elif isinstance(scorers, list):
            # Map metric names to the corresponding sklearn.metrics.<name>_score functions
            funcs = {name: getattr(skm, f"{name}_score", None) for name in scorers}
            assert all(funcs.values()), (
                f"`scorers` must be names of `sklearn.metrics.<name>_score` functions, got {scorers}"
            )
            scorers = {
                name: (lambda t, p, w, f=f: f(t, p, sample_weight=w)) for name, f in funcs.items()
            }
        # Make a copy of data since weights series might be added to it
        df = self.data.copy()
        if sample_weight is not None:
            assert sample_weight.index.equals(self.data.index), (
                "If not `None`, `sample_weight` Series must be a pandas Series with the same index "
                "as `self.data`"
            )
            # Add weights as a third column for multi_scorer to use
            df["weights"] = sample_weight
        if pooled:
            # Pool all epochs across sessions and compute a single set of scores
            agreement = pd.Series(self.multi_scorer(df, scorers=scorers), name="agreement")
        else:
            # Get per-session averaged/weighted agreement scores
            agreement = (
                df.groupby(level=0).apply(self.multi_scorer, scorers=scorers).apply(pd.Series)
            )
            # Convert to Series if just one session being evaluated
            if self.n_sessions == 1:
                agreement = agreement.squeeze().rename("agreement")
        # Set attribute for later access
        self._agreement = agreement
        return agreement

    def get_agreement_bystage(self, beta=1.0, zero_division=np.nan):
        """
        Return a :py:class:`pandas.DataFrame` of unweighted (i.e., one-vs-rest) agreement scores.

        Parameters
        ----------
        beta : float
            Weight of recall relative to precision in the F-score. Default is 1.0 (i.e., F1).
            See :py:func:`sklearn.metrics.precision_recall_fscore_support`.
        zero_division : {np.nan, 0, 1, "warn"}
            Value assigned to a metric whose denominator is zero, e.g. ``recall`` (and ``fbeta``)
            for a stage that is absent from the reference hypnogram of a session, or ``precision``
            for a stage that the observed scorer never assigned in a session. Default is
            ``np.nan``, meaning that undefined scores are left missing and therefore ignored when
            averaging across sessions (e.g. in :py:meth:`summary`). Set to ``0`` to count such
            sessions as a score of 0% instead, or to ``1``
            for a score of 100%. ``"warn"`` behaves like ``0`` but also emits a
            :py:class:`sklearn.exceptions.UndefinedMetricWarning`. Applied to all metrics,
            including ``specificity`` and ``npv``.

            .. versionadded:: 0.8.0

        Returns
        -------
        agreement : :py:class:`pandas.DataFrame`
            A :py:class:`~pandas.DataFrame` with agreement metrics as columns
            (``fbeta``, ``npv``, ``precision``, ``recall``, ``specificity``, ``support``) and a
            :py:class:`~pandas.MultiIndex` with session and sleep stage as rows.

            ``specificity`` (True Negative Rate) and ``npv`` (Negative Predictive Value) are
            computed using a one-vs-rest confusion matrix per stage.

            All metrics are expressed as percentages (0-100), except ``support``, which is the
            number of epochs of each stage in the reference hypnogram. A ``support`` of 0 flags
            the sessions where ``recall`` and ``fbeta`` are undefined.
        """
        assert isinstance(beta, (int, float)) and beta > 0, "`beta` must be a positive number"
        zd_is_warn = isinstance(zero_division, str) and zero_division == "warn"
        assert zd_is_warn or (
            isinstance(zero_division, (int, float, np.integer, np.floating))
            and not isinstance(zero_division, bool)
            and (np.isnan(zero_division) or zero_division in (0, 1))
        ), "`zero_division` must be one of np.nan, 0, 1, or 'warn'"
        # Value used for the hand-computed metrics (specificity, npv). sklearn returns 0 (and
        # warns) when zero_division="warn", so mirror that here.
        zd_value = 0.0 if zd_is_warn else float(zero_division)

        def _safe_ratio(num, den, name):
            """Return num / den, or the zero-division value (and optional warning) if den == 0."""
            if den > 0:
                return num / den
            if zd_is_warn:
                from sklearn.exceptions import UndefinedMetricWarning

                warnings.warn(
                    f"{name} is ill-defined and being set to 0.0 in labels with no negative "
                    "samples. Use `zero_division` parameter to control this behavior.",
                    UndefinedMetricWarning,
                    stacklevel=4,
                )
            return zd_value

        def scorer(df):
            true, pred = df.to_numpy().T
            prfs = skm.precision_recall_fscore_support(
                true,
                pred,
                beta=beta,
                labels=self._skm_labels,
                average=None,
                zero_division=zero_division,
            )
            cm = skm.confusion_matrix(true, pred, labels=self._skm_labels)
            n = cm.sum()
            n_stages = len(self._skm_labels)
            specificity = np.zeros(n_stages)
            npv = np.zeros(n_stages)
            for i in range(n_stages):
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                tn = n - tp - fp - fn
                specificity[i] = _safe_ratio(tn, tn + fp, "Specificity")
                npv[i] = _safe_ratio(tn, tn + fn, "NPV")
            # Express all metrics (except support) as percentages
            precision, recall, fbeta, support = prfs
            return (
                100 * precision,
                100 * recall,
                100 * fbeta,
                support,
                100 * specificity,
                100 * npv,
            )

        agreement = (
            self.data
            # Get precision, recall, fbeta, support, specificity, and NPV for each session
            .groupby(level=0)
            .apply(scorer)
            # Unpack arrays
            .explode()
            .apply(pd.Series)
            # Add metric labels column and prepend it to index, creating MultiIndex
            .assign(
                metric=["precision", "recall", "fbeta", "support", "specificity", "npv"]
                * self.n_sessions
            )
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
        if self.n_sessions == 1:
            agreement = agreement.reset_index(level=1, drop=True)
        return agreement

    def get_confusion_matrix(self, sleep_id=None, agg_func=None, **kwargs):
        """
        Return a ``ref_hyp`` / ``obs_hyp`` confusion matrix from either a single session or all
        sessions concatenated together.

        Parameters
        ----------
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
            observed scorer as columns.

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
        assert sleep_id is None or sleep_id in self._sleep_ids, (
            "`sleep_id` must be None or a valid sleep ID"
        )
        assert isinstance(agg_func, (type(None), str)), "`agg_func` must be None or a str"
        assert not ((self.n_sessions == 1 or sleep_id is not None) and agg_func is not None), (
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
            # Use positional indices [0, 1, ..., n_stages-1] here, not the actual YASA
            # integer codes in _skm_labels.  _skm2yasa_map expects positional keys
            # (e.g. {0:0, 1:2, 2:3, 3:4}); passing the YASA codes directly would
            # cause e.g. code 2 to be looked up as position 2 (→3) instead of LIGHT.
            .assign(**{self.ref_scorer: list(range(len(self._skm_labels))) * self.n_sessions})
            .set_index(self.ref_scorer, append=True)
            .rename_axis(columns=self.obs_scorer)
            # Convert sleep stage columns and indices to strings
            .rename(columns=self._skm2yasa_map)
            .rename(columns=self._yasa2yasa_map)
            .rename(index=self._skm2yasa_map, level=self.ref_scorer)
            .rename(index=self._yasa2yasa_map, level=self.ref_scorer)
        )
        if self.n_sessions == 1:
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

    @staticmethod
    def _resolve_bootstrap_kwargs(bootstrap_kwargs, n_resamples_default):
        """
        Validate a user-supplied ``bootstrap_kwargs`` mapping and return the resolved settings.

        Shared by :py:meth:`get_confusion_matrix_proportional` and :py:meth:`summary`, which
        expose the same three keys and differ only in their default number of resamples.

        Parameters
        ----------
        bootstrap_kwargs : dict or None
            User-supplied settings. Valid keys are ``'n_resamples'``, ``'method'`` and ``'rng'``.
        n_resamples_default : int
            Default number of resamples for the calling method.

        Returns
        -------
        n_resamples : int
        method : {"BCa", "basic", "percentile"}
        rng : :py:class:`numpy.random.Generator`
        """
        allowed_bootstrap_kwargs = ["n_resamples", "method", "rng"]
        assert isinstance(bootstrap_kwargs, (dict, type(None))), (
            "`bootstrap_kwargs` must be a dictionary or None"
        )
        bootstrap_kwargs = {} if bootstrap_kwargs is None else bootstrap_kwargs
        assert all(k in allowed_bootstrap_kwargs for k in bootstrap_kwargs), (
            f"keys of `bootstrap_kwargs` must be in {allowed_bootstrap_kwargs}"
        )
        bs_kwargs = {
            "n_resamples": n_resamples_default,
            "method": "BCa",
            "rng": None,
        } | bootstrap_kwargs
        n_resamples, method = bs_kwargs["n_resamples"], bs_kwargs["method"]
        assert isinstance(n_resamples, (int, np.integer)) and n_resamples > 0, (
            "`n_resamples` must be a positive integer"
        )
        assert method in ("BCa", "basic", "percentile"), (
            "`method` must be one of 'BCa', 'basic', or 'percentile'"
        )
        return n_resamples, method, np.random.default_rng(bs_kwargs["rng"])

    @staticmethod
    def _bootstrap_ci_mean(arr2d, mean, n_valid, confidence, n_resamples, method, rng):
        """
        Participant-level bootstrap CI of the column-wise (nan)mean of ``arr2d``.

        Rows of ``arr2d`` are sessions and columns are the quantities to summarize (here, the
        cells of the row-normalized confusion matrix). Sessions are resampled with replacement,
        using the same resampled sessions for every column, so that dependencies between columns
        (e.g. rows of a confusion matrix summing to 100) are preserved in each replicate. NaN
        entries (undefined cells) are ignored, and replicates in which a column has no defined
        value are excluded from its percentiles.

        Parameters
        ----------
        arr2d : ndarray, shape (n_sessions, n_columns)
        mean : ndarray, shape (n_columns,)
            Point estimate (nanmean over sessions) of each column.
        n_valid : ndarray, shape (n_columns,)
            Number of non-NaN sessions for each column.
        confidence : float
            Confidence level of the interval.
        n_resamples : int
            Number of bootstrap resamples.
        method : {"BCa", "basic", "percentile"}
            Bootstrap interval method. ``"BCa"`` is the bias-corrected and accelerated method of
            Efron (1987); columns whose bootstrap distribution is degenerate (constant) fall back
            to plain percentiles, which then equal the constant value.
        rng : :py:class:`numpy.random.Generator`

        Returns
        -------
        ci_lower, ci_upper : ndarray, shape (n_columns,)
        """
        n_sessions, n_columns = arr2d.shape
        # Resample in batches to bound memory (n_resamples x n_sessions x n_columns floats).
        boot_means = np.empty((n_resamples, n_columns))
        batch_size = 500
        for start in range(0, n_resamples, batch_size):
            stop = min(start + batch_size, n_resamples)
            idx = rng.integers(0, n_sessions, size=(stop - start, n_sessions))
            boot_means[start:stop] = np.nanmean(arr2d[idx], axis=1)
        alpha = (1 - confidence) / 2
        if method == "BCa":
            n_boot_valid = np.sum(~np.isnan(boot_means), axis=0)
            # Bias-correction z0: normal quantile of the fraction of replicates below the estimate
            z0 = sps.norm.ppf(np.sum(boot_means < mean, axis=0) / n_boot_valid)
            # Acceleration a: skewness of the leave-one-session-out (jackknife) estimates, computed
            # over the sessions where the column is defined
            jack = (np.nansum(arr2d, axis=0) - arr2d) / (n_valid - 1)
            jack_dev = np.nanmean(jack, axis=0) - jack
            num = np.nansum(jack_dev**3, axis=0)
            den = 6 * np.nansum(jack_dev**2, axis=0) ** 1.5
            a = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
            # Adjusted percentiles (Efron 1987, eq. 2.5)
            z_lo, z_hi = sps.norm.ppf(alpha), sps.norm.ppf(1 - alpha)
            q_lo = sps.norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
            q_hi = sps.norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))
            # Degenerate columns (constant bootstrap distribution -> z0 infinite) fall back to
            # plain percentiles, which are then equal to the constant value.
            degenerate = ~np.isfinite(z0) | ~np.isfinite(q_lo) | ~np.isfinite(q_hi)
            q_lo = np.where(degenerate, alpha, q_lo)
            q_hi = np.where(degenerate, 1 - alpha, q_hi)
            ci_lower = np.full(n_columns, np.nan)
            ci_upper = np.full(n_columns, np.nan)
            for j in np.flatnonzero(n_boot_valid > 0):
                ci_lower[j] = np.nanpercentile(boot_means[:, j], 100 * q_lo[j])
                ci_upper[j] = np.nanpercentile(boot_means[:, j], 100 * q_hi[j])
        else:
            pct_lo, pct_hi = np.nanpercentile(boot_means, [100 * alpha, 100 * (1 - alpha)], axis=0)
            if method == "percentile":
                ci_lower, ci_upper = pct_lo, pct_hi
            else:
                # Basic (reverse percentile) bootstrap: 2 * estimate - percentiles
                ci_lower, ci_upper = 2 * mean - pct_hi, 2 * mean - pct_lo
        return ci_lower, ci_upper

    def get_confusion_matrix_proportional(
        self,
        ci_method="boot",
        confidence=0.95,
        bootstrap_kwargs=None,
        formatted=False,
        decimals=1,
    ):
        """
        Return the group-level *proportional* confusion (error) matrix, i.e. the mean, standard
        deviation, and confidence interval across sessions of the row-normalized per-session
        confusion matrices, as reported in Menghini et al., 2021 [Menghini2021]_.

        For each session, the confusion matrix is first normalized by row, so that each cell is
        the percentage of reference-scorer epochs of a given stage that the observed scorer
        assigned to each stage (each row sums to 100). These per-session percentages are then
        averaged across sessions (subject-then-group averaging), so that every session
        contributes equally regardless of its duration. This differs from
        ``get_confusion_matrix(agg_func="sum")``, which pools all epochs together.

        Rows of the reference stage that are absent from a session (e.g. no N3 sleep in a given
        night) are undefined (0 / 0) and are excluded from the mean, SD, and CI of that row rather
        than being counted as zeros. The number of contributing sessions is returned in the
        ``n_sessions`` column.

        .. versionadded:: 0.8.0

        Parameters
        ----------
        ci_method : str or None
            Method used to compute the confidence interval of the group mean of each cell.

            * ``'boot'`` (default) — non-parametric bootstrap across sessions (i.e., sessions are
              resampled with replacement, a "participant bootstrap"). The same resampled sessions
              are used for all cells, so that each bootstrapped matrix remains row-normalized.
              Replicates in which a reference stage is absent from every resampled session are
              undefined and are excluded from the percentiles of that row.
            * ``'param'`` — parametric interval ``mean ± t × SD / sqrt(n)`` based on a Student
              t-distribution with ``n - 1`` degrees of freedom, clipped to [0, 100].
            * ``None`` — no confidence interval is computed.
        confidence : float
            Confidence level (between 0 and 1) of the confidence interval. Default is 0.95.
        bootstrap_kwargs : dict or None
            Optional settings of the bootstrap procedure when ``ci_method='boot'``. Valid keys are:

            * ``'n_resamples'`` — number of bootstrap resamples (int, default 1000).
            * ``'method'`` — ``'BCa'`` (default) for bias-corrected and accelerated percentiles
              [Efron1987]_, which corrects for the skewness and bias of the bootstrap
              distribution and is more accurate than the simpler alternatives; ``'percentile'``
              for plain percentiles; or ``'basic'`` for the reverse-percentile method used by the
              reference pipeline of Menghini et al. (2021). Cells that are constant across all
              resamples (e.g. a stage that is never confused) fall back to plain percentiles.
            * ``'rng'`` — an integer seed or :py:class:`numpy.random.Generator` for reproducible
              intervals (default None).
        formatted : bool
            If ``False`` (default), return the numeric statistics in long format. If ``True``,
            return a square, human-readable matrix of strings formatted as
            ``"mean (SD) [lower, upper]"`` (or ``"mean (SD)"`` when ``ci_method=None``).
        decimals : int
            Number of decimal places used when ``formatted=True``. Default is 1.

        Returns
        -------
        conf_matr : :py:class:`pandas.DataFrame`
            If ``formatted=False``, a long-format :py:class:`~pandas.DataFrame` with a
            :py:class:`~pandas.MultiIndex` of (reference stage, observed stage) pairs as rows and
            columns ``mean``, ``std``, ``ci_lower``, ``ci_upper`` (the latter two only when
            ``ci_method`` is not None), and ``n_sessions``. All values except ``n_sessions`` are
            percentages (0-100). Use e.g. ``conf_matr["mean"].unstack()`` to get the mean matrix
            in square format.

            If ``formatted=True``, a square :py:class:`~pandas.DataFrame` with stages from the
            reference scorer as index and stages from the observed scorer as columns, where each
            cell is a formatted string.

        Examples
        --------
        >>> import yasa
        >>> ref_hyps = [yasa.simulate_hypnogram(tib=600, scorer="Human", seed=i) for i in range(10)]
        >>> obs_hyps = [h.simulate_similar(scorer="YASA", seed=i) for i, h in enumerate(ref_hyps)]
        >>> ebe = yasa.EpochByEpochAgreement(ref_hyps, obs_hyps)
        >>> ebe.get_confusion_matrix_proportional(ci_method="param").head(5).round(1)
                    mean   std  ci_lower  ci_upper  n_sessions
        Human YASA
        WAKE  WAKE  31.6   9.3      25.0      38.3          10
              N1    12.2   4.8       8.8      15.7          10
              N2    35.1  11.0      27.2      43.0          10
              N3     9.9   8.0       4.2      15.7          10
              REM   11.1  10.0       3.9      18.2          10

        >>> ebe.get_confusion_matrix_proportional(ci_method=None, formatted=True)
        YASA          WAKE          N1           N2           N3          REM
        Human
        WAKE    31.6 (9.3)  12.2 (4.8)  35.1 (11.0)    9.9 (8.0)  11.1 (10.0)
        N1      21.3 (9.5)  16.2 (8.1)   35.7 (9.5)  14.1 (11.6)   12.7 (9.8)
        N2     22.0 (10.5)  11.7 (3.1)  38.5 (11.8)  15.1 (10.9)  12.7 (11.8)
        N3     22.9 (13.1)   9.7 (5.4)  37.2 (12.4)  21.9 (18.2)   8.4 (11.8)
        REM    20.9 (20.4)   9.3 (5.4)  40.0 (13.0)  14.7 (16.0)  15.0 (17.9)

        With the default bootstrap CI (``ci_method="boot"``), pass ``rng`` for reproducibility:

        >>> ebe.get_confusion_matrix_proportional(formatted=True, bootstrap_kwargs={"rng": 1}).loc[
        ...     "WAKE", "WAKE"
        ... ]  # doctest: +SKIP
        '31.6 (9.3) [26.6, 38.0]'
        """
        assert self.n_sessions > 1, (
            "A proportional confusion matrix can not be computed with only one hypnogram pair."
        )
        assert ci_method is None or ci_method in ("boot", "param"), (
            "`ci_method` must be one of 'boot', 'param', or None"
        )
        assert isinstance(confidence, (int, float)) and 0 < confidence < 1, (
            "`confidence` must be a number between 0 and 1"
        )
        n_resamples, boot_method, rng = self._resolve_bootstrap_kwargs(bootstrap_kwargs, 1000)
        assert isinstance(formatted, bool), "`formatted` must be True or False"
        assert isinstance(decimals, int) and decimals >= 0, (
            "`decimals` must be a non-negative integer"
        )

        # Per-session confusion matrices: MultiIndex (sleep_id, ref stage) x obs stage
        cms = self.get_confusion_matrix()
        stages = cms.columns.tolist()
        # Row-normalize each session (in percent). Rows where the reference stage is absent are
        # 0 / 0 = NaN, and are deliberately kept as NaN (unlike sklearn's normalize="true", which
        # zero-fills).
        props = 100 * cms.div(cms.sum(axis=1), axis=0)
        # Stack into a 3D array of shape (n_sessions, n_stages, n_stages)
        arr = np.stack(
            [
                grp.droplevel("sleep_id").loc[stages, stages].to_numpy(dtype=float)
                for _, grp in props.groupby(level="sleep_id", sort=False)
            ]
        )
        n_stages = len(stages)
        n_cells = n_stages * n_stages
        # Flatten stage dimensions so that each column is one cell of the matrix
        arr2d = arr.reshape(self.n_sessions, n_cells)
        n_valid = np.sum(~np.isnan(arr2d), axis=0)

        with warnings.catch_warnings():
            # Cells whose reference stage is absent from all sessions are all-NaN and legitimately
            # produce NaN statistics; silence the "Mean of empty slice" family of warnings.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(arr2d, axis=0)
            std = np.nanstd(arr2d, axis=0, ddof=1)
            if ci_method == "param":
                t_crit = sps.t.ppf((1 + confidence) / 2, n_valid - 1)
                half_width = t_crit * std / np.sqrt(n_valid)
                ci_lower = np.clip(mean - half_width, 0, 100)
                ci_upper = np.clip(mean + half_width, 0, 100)
            elif ci_method == "boot":
                ci_lower, ci_upper = self._bootstrap_ci_mean(
                    arr2d, mean, n_valid, confidence, n_resamples, boot_method, rng
                )
                # Percentages are bounded; the basic method can slightly overshoot [0, 100].
                ci_lower = np.clip(ci_lower, 0, 100)
                ci_upper = np.clip(ci_upper, 0, 100)

        index = pd.MultiIndex.from_product(
            [stages, stages], names=[self.ref_scorer, self.obs_scorer]
        )
        out = pd.DataFrame({"mean": mean, "std": std}, index=index)
        if ci_method is not None:
            out["ci_lower"] = ci_lower
            out["ci_upper"] = ci_upper
        out["n_sessions"] = n_valid
        if not formatted:
            return out

        # Human-readable square matrix of "mean (SD) [lower, upper]" strings
        d = decimals

        def _fmt(row):
            s = f"{row['mean']:.{d}f} ({row['std']:.{d}f})"
            if ci_method is not None:
                s += f" [{row['ci_lower']:.{d}f}, {row['ci_upper']:.{d}f}]"
            return s

        return out.apply(_fmt, axis=1).unstack(self.obs_scorer).loc[stages, stages]

    def get_sleep_stats(self):
        """
        Return a :py:class:`pandas.DataFrame` of sleep statistics for each hypnogram derived from
        both reference and observed scorers.

        .. seealso:: :py:meth:`yasa.Hypnogram.sleep_statistics`

        .. seealso:: :py:class:`yasa.SleepStatsAgreement`

        Returns
        -------
        sstats : :py:class:`pandas.DataFrame`
            A :py:class:`~pandas.DataFrame` with sleep statistics as columns and two rows for each
            individual (one for reference scorer and another for observed scorer).
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
        if self.n_sessions == 1:
            sstats = sstats.reset_index(level=1, drop=True)
        return sstats

    def plot_hypnograms(
        self, sleep_id=None, legend=True, ax=None, ref_kwargs=None, obs_kwargs=None
    ):
        """Plot the two hypnograms of one session overlapping on the same axis.

        .. seealso:: :py:func:`yasa.plot_hypnogram`

        Parameters
        ----------
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
        assert sleep_id is None or sleep_id in self._sleep_ids, (
            "`sleep_id` must be None or a valid sleep ID"
        )
        assert isinstance(legend, (bool, dict)), "`legend` must be True, False, or a dictionary"
        ref_kwargs = {} if ref_kwargs is None else ref_kwargs
        obs_kwargs = {} if obs_kwargs is None else obs_kwargs
        assert isinstance(ref_kwargs, dict), "`ref_kwargs` must be a dictionary"
        assert isinstance(obs_kwargs, dict), "`obs_kwargs` must be a dictionary"
        assert "ax" not in ref_kwargs | obs_kwargs, (
            "'ax' can't be supplied to `ref_kwargs` or `obs_kwargs`, use the `ax` keyword instead"
        )
        assert not (sleep_id is None and self.n_sessions > 1), (
            "Multi-session plotting is not currently supported. `sleep_id` must not be None when "
            "multiple sessions are present"
        )
        # Select the session hypnograms to plot
        if sleep_id is None and self.n_sessions == 1:
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

    def summary(
        self, by_stage=False, ci_method=None, confidence=0.95, bootstrap_kwargs=None, **kwargs
    ):
        """Return group-level agreement scores.

        Parameters
        ----------
        by_stage : bool
            If ``False`` (default), ``summary`` will include agreement scores derived from
            average-based metrics. If ``True``, returned ``summary`` :py:class:`~pandas.DataFrame`
            will include agreement scores for each sleep stage, derived from one-vs-rest metrics.
        ci_method : str or None
            Method used to compute a confidence interval of the mean of each metric across
            sessions.

            * ``None`` (default) — no confidence interval is computed.
            * ``'boot'`` — non-parametric bootstrap across sessions. The same resampled sessions
              are used for every metric (and stage), so each replicate remains internally
              consistent. Sessions in which a metric is undefined (see the ``zero_division``
              parameter of :py:meth:`get_agreement_bystage`) are ignored, and replicates in which a
              metric has no defined value are excluded from its percentiles.

            .. versionadded:: 0.8.0
        confidence : float
            Confidence level (between 0 and 1) of the confidence interval. Default is 0.95.

            .. versionadded:: 0.8.0
        bootstrap_kwargs : dict or None
            Optional settings of the bootstrap procedure when ``ci_method='boot'``. Valid keys are:

            * ``'n_resamples'`` — number of bootstrap resamples (int, default 10000).
            * ``'method'`` — ``'BCa'`` (default) for bias-corrected and accelerated percentiles
              [Efron1987]_, ``'percentile'`` for plain percentiles, or ``'basic'`` for the
              reverse-percentile method. Metrics that are constant across all resamples fall back
              to plain percentiles.
            * ``'rng'`` — an integer seed or :py:class:`numpy.random.Generator` for reproducible
              intervals (default None).

            .. versionadded:: 0.8.0
        **kwargs : key, value pairs
            Additional keyword arguments are passed to :py:meth:`pandas.DataFrame.groupby.agg`.
            This can be used to customize the descriptive statistics returned.

        Returns
        -------
        summary : :py:class:`pandas.DataFrame`
            A :py:class:`pandas.DataFrame` summarizing agreement scores across the entire dataset
            with descriptive statistics. Each row is an agreement metric and each column is a
            descriptive statistic (e.g., mean, standard deviation).

            When ``ci_method`` is not ``None``, two extra columns ``ci_lower`` and ``ci_upper``
            are appended.

        Examples
        --------
        The scores of the last :py:meth:`get_agreement` (or :py:meth:`get_agreement_bystage` if
        ``by_stage=True``) call are summarized; if none was made, default scores are computed:

        >>> import yasa
        >>> ref_hyps = [yasa.simulate_hypnogram(tib=600, scorer="Human", seed=i) for i in range(5)]
        >>> obs_hyps = [h.simulate_similar(scorer="YASA", seed=i) for i, h in enumerate(ref_hyps)]
        >>> ebe = yasa.EpochByEpochAgreement(ref_hyps, obs_hyps)
        >>> _ = ebe.get_agreement()
        >>> ebe.summary().round(2)
                       mad   mean   std    min  median    max
        metric
        accuracy      5.30  28.58  6.27  21.42   30.58  35.08
        balanced_acc  4.13  24.16  5.85  16.85   23.87  32.68
        kappa         0.05   0.04  0.07  -0.06    0.06   0.14
        mcc           0.05   0.05  0.07  -0.06    0.06   0.14
        precision     6.24  28.26  7.33  20.29   30.64  34.93
        f1            5.89  27.97  6.88  20.50   30.56  34.55

        To control the descriptive statistics included as columns:

        >>> ebe.summary(func=["count", "mean", "sem"]).round(2)
                      count   mean   sem
        metric
        accuracy        5.0  28.58  2.80
        balanced_acc    5.0  24.16  2.62
        kappa           5.0   0.04  0.03
        mcc             5.0   0.05  0.03
        precision       5.0  28.26  3.28
        f1              5.0  27.97  3.07

        To add a bootstrap confidence interval of the mean across sessions, pass ``ci_method``.

        >>> ebe.summary(
        ...     ci_method="boot",
        ...     bootstrap_kwargs={"method": "percentile", "n_resamples": 1000, "rng": 0},
        ...     func=["mean"],
        ... ).round(2)
                       mean  ci_lower  ci_upper
        metric
        accuracy      28.58     23.68     33.28
        balanced_acc  24.16     20.08     29.09
        kappa          0.04     -0.01      0.10
        mcc            0.05     -0.01      0.10
        precision     28.26     22.50     34.03
        f1            27.97     22.60     33.13
        """
        assert self.n_sessions > 1, (
            "Summary scores can not be computed with only one hypnogram pair."
        )
        assert isinstance(by_stage, bool), "`by_stage` must be True or False"
        assert ci_method is None or ci_method == "boot", "`ci_method` must be 'boot' or None"
        assert isinstance(confidence, (int, float)) and 0 < confidence < 1, (
            "`confidence` must be a number between 0 and 1"
        )
        n_resamples, boot_method, rng = self._resolve_bootstrap_kwargs(bootstrap_kwargs, 10000)
        if by_stage and not hasattr(self, "_agreement_bystage"):
            self.get_agreement_bystage()
        elif not by_stage and not hasattr(self, "_agreement"):
            self.get_agreement()

        # Create a function for getting mean absolute deviation
        def mad(df):
            return (df - df.mean()).abs().mean()

        # Merge default and user kwargs
        agg_kwargs = {"func": [mad, "mean", "std", "min", "median", "max"]} | kwargs
        if by_stage:
            summary = (
                self._agreement_bystage.groupby("stage")
                .agg(**agg_kwargs)
                .stack(level=0)
                .rename_axis(["stage", "metric"])
            )
        else:
            summary = self._agreement.agg(**agg_kwargs).T.rename_axis("metric")
        if ci_method is None:
            return summary

        if boot_method == "BCa" and self.n_sessions < 20:
            warnings.warn(
                "The BCa bootstrap interval can be unstable with few sessions "
                f"(n_sessions={self.n_sessions}). Consider "
                "bootstrap_kwargs={'method': 'percentile'} or the parametric "
                "summary(func=['mean', 'sem']) instead.",
                RuntimeWarning,
                stacklevel=2,
            )
        # Reshape so that rows are sessions and columns are the quantities to summarize, which is
        # what `_bootstrap_ci_mean` expects: sessions are the resampling unit and the same
        # resampled sessions are used for every column.
        if by_stage:
            wide = self._agreement_bystage.unstack("stage")
            # Columns are (metric, stage); reorder to match the (stage, metric) summary index
            ci_index = wide.columns.swaplevel().rename(["stage", "metric"])
        else:
            wide = self._agreement
            ci_index = pd.Index(wide.columns, name="metric")
        arr2d = wide.to_numpy(dtype=float)
        with warnings.catch_warnings():
            # Metrics that are undefined in every session are all-NaN and legitimately produce
            # NaN statistics; silence the "Mean of empty slice" family of warnings.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(arr2d, axis=0)
            n_valid = np.sum(~np.isnan(arr2d), axis=0)
            ci_lower, ci_upper = self._bootstrap_ci_mean(
                arr2d, mean, n_valid, confidence, n_resamples, boot_method, rng
            )
        summary["ci_lower"] = pd.Series(ci_lower, index=ci_index).reindex(summary.index)
        summary["ci_upper"] = pd.Series(ci_upper, index=ci_index).reindex(summary.index)
        if by_stage:
            # `support` is the number of epochs of each stage, not an agreement score
            is_support = summary.index.get_level_values("metric") == "support"
            summary.loc[is_support, ["ci_lower", "ci_upper"]] = np.nan
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
    * Test statistical assumptions of bias, limits of agreement, and their confidence intervals, and apply corrective procedures when the assumptions are not met.
    * Get bias and limits of agreement in a string-formatted table.
    * Calibrate new data to correct for biases in observed data.
    * Visualize Bland-Altman plots.

    .. warning:: **Experimental** — the API of this class may change before the full release
        planned for v0.8.0. Use with caution.

    .. seealso:: :py:meth:`yasa.Hypnogram.sleep_statistics`

    .. versionadded:: 0.7.0

    Parameters
    ----------
    ref_data : :py:class:`pandas.DataFrame`
        A :py:class:`pandas.DataFrame` with sleep statistics from the reference scorer.
        Rows are unique observations and columns are unique sleep statistics.

        Alternatively, the output of :py:meth:`yasa.EpochByEpochAgreement.get_sleep_stats`, i.e.
        a single :py:class:`pandas.DataFrame` whose first index level is the scorer. In that
        case ``obs_data`` must be ``None`` and the two scorers are taken from the index in order of
        appearance (reference first).
    obs_data : :py:class:`pandas.DataFrame` or None
        A :py:class:`pandas.DataFrame` with sleep statistics from the observed scorer.
        Rows are unique observations and columns are unique sleep statistics.
        Shape, index, and columns must be identical to ``ref_data``.
    ref_scorer : str
        Name of the reference scorer. Ignored when ``obs_data`` is ``None``.
    obs_scorer : str
        Name of the observed scorer. Ignored when ``obs_data`` is ``None``.
    confidence : float
        Confidence level (between 0 and 1) for the confidence intervals applied to bias and limits
        of agreement. Default is 0.95 (i.e., 95%). The same level is used for both parametric and
        bootstrapped confidence intervals.
    alpha : float
        Alpha cutoff used for all assumption tests. Default is 0.05.
    bootstrap_kwargs : dict
        Optional keyword arguments passed to :py:func:`scipy.stats.bootstrap`. Defaults use
        ``n_resamples=1000`` and ``method='BCa'``. The keys ``'confidence_level'``,
        ``'vectorized'``, and ``'paired'`` cannot be overridden.
    log_transform : bool
        If ``True``, apply the Euser et al. (2008) log-transformation method to all sleep
        statistics. Limits of agreement are then expressed as ``bias ± slope × ref``, where
        ``slope`` is derived from the standard deviation of log-ratio differences. This is
        appropriate when measurement variability is proportional to the measurement magnitude
        (heteroscedasticity), which is common for duration statistics such as TST, SOL, and WASO.
        When ``True``, ``loa_method='auto'`` in :py:meth:`report` and
        :py:meth:`plot_blandaltman` will automatically select the Euser method for all
        statistics, bypassing the homoscedasticity assumption test. Statistics with a value of
        exactly zero in either scorer (e.g. SOL for a subject who fell asleep in the first epoch)
        can not be log-transformed; they are excluded with a warning and keep the regular LoA
        methods. Passing ``loa_method='log'`` for such statistics, or when
        ``log_transform=False``, raises a ``ValueError``. Default is ``False``.

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
    >>> # Create SleepStatsAgreement instance
    >>> ssa = yasa.SleepStatsAgreement(sstats)
    >>> ssa.summary(ci_method="param").round(1).head(3)  # doctest: +NORMALIZE_WHITESPACE
    variable   bias_intercept             bias_mean  ... loa_slope loa_upper
    interval           center lower upper    center  ...     upper    center lower upper
    sleep_stat                                       ...
    %N1                  -5.4 -13.9   3.2       0.3  ...       0.4       6.1   3.7   8.5
    %N2                 -27.3 -49.1  -5.6      -0.2  ...       0.2      12.4   7.2  17.6
    %N3                  -9.1 -23.8   5.5       1.4  ...       0.6      20.4  12.6  28.3
    <BLANKLINE>
    [3 rows x 24 columns]

    >>> ssa.report(ci_method="param").head(3)[["Bias [95% CI]", "LoA [95% CI]"]]  # doctest: +SKIP

    >>> ssa.assumptions["constant_bias"].head(3).round(3)
    metric      slope  pvalue     r2  passed method
    sleep_stat
    %N1         0.370   0.181  0.097    True  param
    %N2         0.553   0.017  0.279   False   regr
    %N3         0.613   0.131  0.122    True  param

    >>> ssa.assumptions.xs("method", level="metric", axis=1).head(3)
    assumption normal constant_bias homoscedastic
    sleep_stat
    %N1         param         param         param
    %N2         param          regr         param
    %N3         param         param         param

    >>> new_hyps = [h.simulate_similar(scorer="Kelly", seed=i) for i, h in enumerate(obs_hyps)]
    >>> new_sstats = pd.Series(new_hyps).map(lambda h: h.sleep_statistics()).apply(pd.Series)
    >>> new_sstats[["N1", "TST", "WASO"]].round(1).head(5)
         N1    TST   WASO
    0  42.5  439.5  147.5
    1  84.0  550.0   38.5
    2  53.5  489.0  103.0
    3  57.0  469.5  120.0
    4  71.0  531.0   69.0

    >>> new_stats_calibrated = ssa.calibrate(new_sstats[ssa.sleep_statistics], bias_method="auto")
    >>> new_stats_calibrated[["N1", "TST", "WASO"]].round(1).head(5)
         N1    TST   WASO
    0  53.0  448.4  143.0
    1  79.8  549.7   40.2
    2  60.1  493.7  101.0
    3  62.4  475.9  117.1
    4  71.4  532.2   69.0

    """

    _bias_method_opts = ("param", "regr", "auto")
    _loa_method_opts = ("param", "regr", "log", "auto")
    _ci_method_opts = ("param", "boot", "auto")

    # Multiple of the SD of the differences defining the limits of agreement (95% coverage)
    _agreement = 1.96

    def __init__(
        self,
        ref_data,
        obs_data=None,
        *,
        ref_scorer="Reference",
        obs_scorer="Observed",
        confidence=0.95,
        alpha=0.05,
        bootstrap_kwargs=None,
        log_transform=False,
    ):
        restricted_bootstrap_kwargs = ["confidence_level", "vectorized", "paired"]
        bootstrap_kwargs = {} if bootstrap_kwargs is None else bootstrap_kwargs
        agreement = self._agreement

        assert isinstance(ref_data, pd.DataFrame), "`ref_data` must be a pandas DataFrame"
        if obs_data is None:
            # Output of EpochByEpochAgreement.get_sleep_stats(): (scorer, session) MultiIndex
            assert ref_data.index.nlevels == 2, (
                "`ref_data` must have a (scorer, session) MultiIndex when `obs_data` is None"
            )
            scorers = ref_data.index.get_level_values(0).unique().tolist()
            assert len(scorers) == 2, f"`ref_data` must contain exactly two scorers, got {scorers}"
            ref_scorer, obs_scorer = scorers
            ref_data, obs_data = ref_data.loc[ref_scorer], ref_data.loc[obs_scorer]
        assert isinstance(obs_data, pd.DataFrame), "`obs_data` must be a pandas DataFrame"
        assert np.array_equal(ref_data.index, obs_data.index), (
            "`ref_data` and `obs_data` index values must be identical"
        )
        assert ref_data.index.name == obs_data.index.name, (
            "`ref_data` and `obs_data` index names must be identical"
        )
        assert np.array_equal(ref_data.columns, obs_data.columns), (
            "`ref_data` and `obs_data` column values must be identical"
        )
        assert isinstance(ref_scorer, str), "`ref_scorer` must be a string"
        assert isinstance(obs_scorer, str), "`obs_scorer` must be a string"
        assert ref_scorer != obs_scorer, "`ref_scorer` and `obs_scorer` must be unique"
        assert isinstance(confidence, (float, int)) and 0 < confidence < 1, (
            "`confidence` must be a number between 0 and 1"
        )
        assert isinstance(alpha, (float, int)) and 0 <= alpha <= 1, (
            "`alpha` must be a number between 0 and 1 inclusive"
        )
        assert isinstance(bootstrap_kwargs, dict), "`bootstrap_kwargs` must be a dictionary"
        assert all(k not in restricted_bootstrap_kwargs for k in bootstrap_kwargs), (
            f"None of {restricted_bootstrap_kwargs} can be set by the user"
        )
        assert isinstance(log_transform, bool), "`log_transform` must be a bool"
        # If `ref_data` and `obs_data` indices are unnamed, name them (on copies, not in place)
        session_key = "session_id" if ref_data.index.name is None else ref_data.index.name
        ref_data, obs_data = ref_data.rename_axis(session_key), obs_data.rename_axis(session_key)

        # Reshape to long format DataFrame with 2 columns (observed, reference) and MultiIndex
        data = (
            pd.concat([obs_data, ref_data], keys=[obs_scorer, ref_scorer], names=["scorer"])
            .melt(var_name="sleep_stat", ignore_index=False)
            .pivot_table(index=["sleep_stat", session_key], columns="scorer", values="value")
            .rename_axis(columns=None)
            .sort_index()
        )

        # Remove sessions with a missing value in either scorer, separately for each sleep stat
        # (e.g. Lat_REM is NaN for a night without REM sleep)
        n_missing = data.isna().any(axis=1).groupby("sleep_stat").sum()
        for stat, n_miss in n_missing[n_missing > 0].items():
            logger.warning(f"Removed {n_miss} session(s) with missing values from {stat}.")
        data = data.dropna()

        # Get scorer differences (i.e., observed minus reference)
        data["difference"] = data[obs_scorer] - data[ref_scorer]

        # Remove sleep statistics that have no differences between scorers
        stats_rm = data.groupby("sleep_stat")["difference"].any().loc[lambda x: ~x].index.tolist()
        data = data.drop(labels=stats_rm)
        for s in stats_rm:
            logger.warning(f"Removed {s} from evaluation because all scorings were identical.")

        # Create grouper and per-stat number of sessions (n) for convenience
        grouper = data.groupby("sleep_stat")
        n_sessions = data.index.get_level_values(session_key).nunique()
        n = grouper.size()

        ########################################################################
        # Generate parametric Bias and LoA for all sleep stats
        ########################################################################
        # Parametric Bias
        param_vals = grouper["difference"].mean().to_frame("bias_mean")
        # Parametric LoA
        param_vals["loa_lower"], param_vals["loa_upper"] = zip(
            *grouper["difference"].apply(self._arr_to_loa, agreement=agreement), strict=True
        )

        ########################################################################
        # Generate standard CIs for parametric Bias and LoA for all sleep stats
        ########################################################################
        # Get critical t and standard error used to calculate parametric CIs for parametric Bias/LoA
        t_param = pd.Series(sps.t.ppf((1 + confidence) / 2, n - 1), index=n.index)
        sem = grouper["difference"].sem(ddof=1)
        # Parametric CIs for parametric Bias and LoA
        param_ci = pd.DataFrame(
            {
                "bias_mean-lower": param_vals["bias_mean"] - sem * t_param,
                "bias_mean-upper": param_vals["bias_mean"] + sem * t_param,
                "loa_lower-lower": param_vals["loa_lower"] - sem * t_param * np.sqrt(3),
                "loa_lower-upper": param_vals["loa_lower"] + sem * t_param * np.sqrt(3),
                "loa_upper-lower": param_vals["loa_upper"] - sem * t_param * np.sqrt(3),
                "loa_upper-upper": param_vals["loa_upper"] + sem * t_param * np.sqrt(3),
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
        data["residuals"] = data["difference"].to_numpy() - predicted_values
        data["residuals_abs"] = data["residuals"].abs()
        # Run regression used to (a) model LoA and (b) test for heteroscedasticity/homoscedasticity
        loa_regr = grouper[[ref_scorer, "residuals_abs"]].apply(self._linregr_dict).apply(pd.Series)
        # Stack the two regression dataframes together
        regr = pd.concat({"bias": bias_regr, "loa": loa_regr}, axis=0)

        ########################################################################
        # Generate parametric CIs for regression/modeled Bias and LoA for all sleep stats
        ########################################################################
        # Get critical t used used to calculate parametric CIs for regression Bias/LoA
        # dof=n-2 for regression; align the per-stat t values with the stacked (bias/loa) index
        t_regr = pd.Series(sps.t.ppf((1 + confidence) / 2, n - 2), index=n.index)
        t_stacked = t_regr.reindex(regr.index.get_level_values("sleep_stat")).to_numpy()
        # Parametric CIs for modeled Bias and LoA
        regr_ci = pd.DataFrame(
            {
                "intercept-lower": regr["intercept"] - regr["intercept_stderr"] * t_stacked,
                "intercept-upper": regr["intercept"] + regr["intercept_stderr"] * t_stacked,
                "slope-lower": regr["slope"] - regr["stderr"] * t_stacked,
                "slope-upper": regr["slope"] + regr["stderr"] * t_stacked,
            }
        )
        # Constant LoA around the regression bias line: bias_i ± agreement × SD of the residuals
        # of the bias regression (Menghini et al. 2021, eq. 2). Used when the bias is proportional
        # but the differences are homoscedastic.
        param_vals["loa_halfwidth"] = agreement * grouper["residuals"].std(ddof=1)
        # Parametric CI: the SE of a standard deviation is ~ SD / sqrt(2n) (Bland & Altman 1999)
        halfwidth_se = param_vals["loa_halfwidth"] / np.sqrt(2 * n)
        param_ci["loa_halfwidth-lower"] = param_vals["loa_halfwidth"] - halfwidth_se * t_regr
        param_ci["loa_halfwidth-upper"] = param_vals["loa_halfwidth"] + halfwidth_se * t_regr

        ########################################################################
        # Log-transform analysis (Euser et al. 2008)
        ########################################################################
        # Pre-allocate containers. They remain NaN/empty when log_transform=False
        # so that the attribute-setting block below is unconditional.
        data["log_difference"] = np.nan
        log_transform_stats = []
        loa_log_slope = pd.Series(np.nan, index=param_vals.index, name="loa_log_slope")
        loa_log_ci = pd.DataFrame(
            np.nan,
            index=param_vals.index,
            columns=["param_lower", "param_upper", "boot_lower", "boot_upper"],
        )
        if log_transform:
            # Validate that all values are non-negative. Negative sleep statistics (e.g. TST = -5)
            # are physically impossible and would silently produce NaN log-differences.
            neg_mask = (data[[ref_scorer, obs_scorer]] < 0).any(axis=1)
            if neg_mask.any():
                bad = data.index.get_level_values("sleep_stat")[neg_mask].unique().tolist()
                raise ValueError(
                    f"`log_transform=True` requires all sleep-statistic values to be "
                    f"non-negative, but negative values were found for: {bad}. "
                    "Pass `log_transform=False` or remove these statistics."
                )
            # Statistics with a zero in either scorer can not be log-transformed (log(0) is
            # undefined and any offset would dominate the result); they keep the regular LoA.
            has_zero = (data[[ref_scorer, obs_scorer]] == 0).any(axis=1).groupby("sleep_stat").any()
            if has_zero.any():
                logger.warning(
                    f"Not log-transforming {has_zero[has_zero].index.tolist()} because of zero "
                    "values; regular LoA are used instead."
                )
            log_transform_stats = has_zero[~has_zero].index.tolist()
            # log_difference = log(obs) - log(ref) is the per-session log-ratio.
            # Its SD quantifies proportional variability between the two scorers.
            is_log = data.index.get_level_values("sleep_stat").isin(log_transform_stats)
            data.loc[is_log, "log_difference"] = np.log(data.loc[is_log, obs_scorer]) - np.log(
                data.loc[is_log, ref_scorer]
            )
            for stat in log_transform_stats:
                log_d = data.loc[stat, "log_difference"].to_numpy()
                sd = np.std(log_d, ddof=1)
                # t critical value for the parametric slope CI (Bland & Altman 1999).
                t_log = sps.t.ppf((1 + confidence) / 2, n[stat] - 1)
                # SE of the SD of log-ratios: sqrt(SD^2 * 3 / n)  (Bland & Altman 1999).
                # Used to propagate uncertainty in SD into the slope CI.
                se = np.sqrt(sd**2 * 3 / n[stat])
                # Point estimate: back-transform SD of log-ratios to a proportional slope.
                loa_log_slope[stat] = self._euser_slope_scalar(sd, agreement)
                # Parametric CI: apply _euser_slope_scalar to the CI bounds of SD.
                # Clamp the lower SD bound at 0 so the slope stays non-negative.
                loa_log_ci.at[stat, "param_lower"] = self._euser_slope_scalar(
                    max(sd - t_log * se, 0.0), agreement
                )
                loa_log_ci.at[stat, "param_upper"] = self._euser_slope_scalar(
                    sd + t_log * se, agreement
                )

        ########################################################################
        # Test all statistical assumptions
        ########################################################################
        # For each assumption: test statistic, p-value, effect size, pass/fail flag (p >= alpha)
        # and the method selected when "auto" is requested. The effect sizes let users judge the
        # materiality of a violation, since p-values scale with n.
        def _test_series(res):
            return pd.Series({"statistic": res.statistic, "pvalue": res.pvalue})

        ttest = grouper["difference"].apply(lambda a: _test_series(sps.ttest_1samp(a, 0))).unstack()
        shapiro = grouper["difference"].apply(
            lambda a: (
                _test_series(sps.shapiro(a))
                if len(a) >= 3
                else pd.Series({"statistic": np.nan, "pvalue": 1.0})
            )
        )
        shapiro = shapiro.unstack()
        unbiased = ttest["pvalue"].ge(alpha)
        normal = shapiro["pvalue"].ge(alpha)
        constant_bias = bias_regr["pvalue"].ge(alpha)
        homoscedastic = loa_regr["pvalue"].ge(alpha)
        loa_method = homoscedastic.map({True: "param", False: "regr"})
        loa_method[log_transform_stats] = "log"
        assumptions = pd.DataFrame(
            {
                ("unbiased", "t"): ttest["statistic"],
                ("unbiased", "pvalue"): ttest["pvalue"],
                ("unbiased", "cohen_d"): param_vals["bias_mean"]
                / grouper["difference"].std(ddof=1),
                ("unbiased", "passed"): unbiased,
                ("normal", "W"): shapiro["statistic"],
                ("normal", "pvalue"): shapiro["pvalue"],
                ("normal", "skew"): grouper["difference"].skew(),
                ("normal", "kurtosis"): grouper["difference"].apply(pd.Series.kurt),
                ("normal", "passed"): normal,
                ("normal", "method"): normal.map({True: "param", False: "boot"}),
                ("constant_bias", "slope"): bias_regr["slope"],
                ("constant_bias", "pvalue"): bias_regr["pvalue"],
                ("constant_bias", "r2"): bias_regr["rvalue"] ** 2,
                ("constant_bias", "passed"): constant_bias,
                ("constant_bias", "method"): constant_bias.map({True: "param", False: "regr"}),
                ("homoscedastic", "slope"): loa_regr["slope"],
                ("homoscedastic", "pvalue"): loa_regr["pvalue"],
                ("homoscedastic", "r2"): loa_regr["rvalue"] ** 2,
                ("homoscedastic", "passed"): homoscedastic,
                ("homoscedastic", "method"): loa_method,
            }
        ).rename_axis(columns=["assumption", "metric"])

        ########################################################################
        # Setting attributes
        ########################################################################

        # Merge the parametric and regression values for Bias and LoA
        regr_vals = regr.unstack(0)[["slope", "intercept"]]
        regr_vals.columns = regr_vals.columns.swaplevel().map("_".join)
        vals = param_vals.join(regr_vals).rename_axis("variable", axis=1)

        # Merge the two CI dataframes for easier access
        regr_ci = regr_ci.unstack(0)
        regr_ci.columns = regr_ci.columns.swaplevel().map("_".join)
        ci = param_ci.join(regr_ci)
        ci.columns = pd.MultiIndex.from_tuples(
            tuples=ci.columns.str.split("-", expand=True),
            names=["variable", "interval"],
        )
        empty_df = pd.DataFrame().reindex_like(ci)
        ci = pd.concat({"param": ci, "boot": empty_df}, names=["ci_method"], axis=1)
        ci = ci.sort_index(axis=1)  # Sort MultiIndex columns for cleanliness

        # Set attributes
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
        self._log_transform = log_transform
        self._log_transform_stats = log_transform_stats
        self._loa_log_slope = loa_log_slope
        self._loa_log_ci = loa_log_ci

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
        ``ref_data`` and ``obs_data``, with a :py:class:`~pandas.MultiIndex` with levels
        ``sleep_stat`` and ``session_id`` (or the original index name from the input data).
        Columns are the reference and observed scorer names.
        """
        return self._data.drop(
            columns=["difference", "residuals", "residuals_abs", "log_difference"]
        )

    @property
    def sleep_statistics(self):
        """Return a list of all sleep statistics included in the agreement analyses."""
        return self.data.index.get_level_values("sleep_stat").unique().to_list()

    @property
    def assumptions(self):
        """A :py:class:`pandas.DataFrame` with the results of the statistical assumption tests
        for each sleep statistic. Columns form a MultiIndex with levels ``assumption`` and
        ``metric``:

        * ``unbiased`` — one-sample t-test of the differences against zero: ``t``, ``pvalue``,
          ``cohen_d`` (mean difference divided by its SD) and ``passed``.
        * ``normal`` — Shapiro-Wilk test of the differences: ``W``, ``pvalue``, sample ``skew``,
          excess ``kurtosis``, ``passed`` and the confidence-interval ``method`` (``'param'`` if
          passed, ``'boot'`` otherwise).
        * ``constant_bias`` — regression of the differences on the reference values: ``slope``,
          ``pvalue``, ``r2``, ``passed`` and the bias ``method`` (``'param'`` if passed,
          ``'regr'`` otherwise).
        * ``homoscedastic`` — regression of the absolute residuals of the bias regression on the
          reference values: ``slope``, ``pvalue``, ``r2``, ``passed`` and the limits-of-agreement
          ``method`` (``'param'`` if passed, ``'regr'`` otherwise, ``'log'`` for log-transformed
          statistics when ``log_transform=True``).

        ``passed`` is ``True`` when ``pvalue >= alpha``, and
        ``method`` is what :py:meth:`report`, :py:meth:`summary`, :py:meth:`calibrate` and
        :py:meth:`plot_blandaltman` apply when ``'auto'`` is requested. Because the power of
        these tests grows with the number of sessions, small and practically irrelevant deviations
        become "significant" in large samples. Use the effect sizes (``cohen_d``, ``skew``, ``r2``)
        and the Bland-Altman plots to judge whether a violation matters.

        .. versionchanged:: 0.8.0
            Includes the test statistics, effect sizes and selected methods (previously only the
            pass/fail flags).
        """
        return self._assumptions

    def __repr__(self):
        # TODO v0.8: Keep only the text between < and >
        return (
            f"<SleepStatsAgreement | Observed scorer ('{self.obs_scorer}') evaluated against "
            f"reference scorer ('{self.ref_scorer}'), {self.n_sessions} sleep sessions>\n"
            " - Use `.report()` to get a human-readable summary table\n"
            " - Use `.summary()` to get a numeric dataframe of bias and limits of agreement\n"
            " - Use `.plot_blandaltman()` to get a grid of Bland-Altman plots\n"
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
    def _euser_slope_scalar(sd, agreement):
        """Euser et al. (2008) antilog LoA slope: 2*(exp(z)-1)/(exp(z)+1), z = agreement * sd.

        Converts the SD of log-ratio differences back to a proportional LoA slope in the original
        scale. The limits of agreement are then ``bias ± slope × ref``, where ``slope`` grows with
        the variability of the log-ratios. When SD is 0, ``z = 0`` and the slope is 0 (no spread).
        """
        z = agreement * sd
        return 2.0 * (np.exp(z) - 1.0) / (np.exp(z) + 1.0)

    @staticmethod
    def _linregr_dict(df):
        """
        A wrapper around :py:func:`scipy.stats.linregress` that returns a dictionary instead of a
        named tuple. In the normally returned object, ``intercept_stderr`` is an extra field that is
        not included when converting the named tuple, so this allows it to be included when using
        something like groupby.
        """
        x, y = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
        regr = sps.linregress(x, y)
        return {
            "slope": regr.slope,
            "intercept": regr.intercept,
            "rvalue": regr.rvalue,
            "pvalue": regr.pvalue,
            "stderr": regr.stderr,
            "intercept_stderr": regr.intercept_stderr,
        }

    def _check_sleep_stats(self, sleep_stats):
        """Validate a user-supplied ``sleep_stats`` list and return it (all stats if None)."""
        if sleep_stats is None:
            return self.sleep_statistics
        assert isinstance(sleep_stats, list), "`sleep_stats` must be a list or None"
        assert len(sleep_stats) > 0, "`sleep_stats` must be a non-empty list"
        assert all(isinstance(stat, str) for stat in sleep_stats), (
            "`sleep_stats` must be a list of strings"
        )
        assert len(sleep_stats) == len(set(sleep_stats)), (
            "`sleep_stats` must not contain duplicate entries"
        )
        valid_stats = set(self.sleep_statistics)
        invalid_stats = [stat for stat in sleep_stats if stat not in valid_stats]
        assert not invalid_stats, (
            "`sleep_stats` contains invalid statistics: "
            f"{sorted(invalid_stats)}; valid options are {sorted(valid_stats)}"
        )
        return list(sleep_stats)

    def _resolve_methods(self, sleep_stats, bias_method, loa_method):
        """Validate method arguments and return the stats using parametric bias, parametric LoA
        and log (Euser) LoA, respectively, for the requested ``sleep_stats``."""
        assert bias_method in self._bias_method_opts, (
            f"`bias_method` must be one of {self._bias_method_opts}"
        )
        assert loa_method in self._loa_method_opts, (
            f"`loa_method` must be one of {self._loa_method_opts}"
        )
        methods = self._assumptions.xs("method", level="metric", axis=1).loc[sleep_stats]
        if bias_method == "auto":
            bias_param_idx = methods.index[methods["constant_bias"] == "param"].tolist()
        else:
            bias_param_idx = sleep_stats if bias_method == "param" else []
        if loa_method == "auto":
            loa_param_idx = methods.index[methods["homoscedastic"] == "param"].tolist()
            loa_log_idx = methods.index[methods["homoscedastic"] == "log"].tolist()
        elif loa_method == "log":
            not_log = [s for s in sleep_stats if s not in self._log_transform_stats]
            if not_log:
                raise ValueError(
                    "`loa_method='log'` requires `log_transform=True` and no zero values, which "
                    f"is not the case for {not_log}"
                )
            loa_param_idx, loa_log_idx = [], sleep_stats
        else:
            loa_param_idx = sleep_stats if loa_method == "param" else []
            loa_log_idx = []
        return bias_param_idx, loa_param_idx, loa_log_idx

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
        assert all(isinstance(ss, str) for ss in sleep_stats), (
            "all elements of `sleep_stats` must be strings"
        )
        assert all(ss in self.sleep_statistics for ss in sleep_stats), (
            f"all elements of `sleep_stats` must be one of {self.sleep_statistics}"
        )
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
            bias_mean = np.mean(diff_arr)
            loa_lower, loa_upper = self._arr_to_loa(diff_arr, self._agreement)
            bias_slope, bias_inter = sps.linregress(ref_arr, diff_arr)[:2]
            resid = diff_arr - (bias_inter + bias_slope * ref_arr)
            loa_halfwidth = self._agreement * np.std(resid, ddof=1)
            # Note this is NOT recalculating residuals each time for the next regression
            loa_slope, loa_inter = sps.linregress(ref_arr, rabs_arr)[:2]
            return (
                bias_mean,
                loa_lower,
                loa_upper,
                bias_inter,
                bias_slope,
                loa_inter,
                loa_slope,
                loa_halfwidth,
            )

        # !! Column order MUST match the order of arrays boot_stats expects as INPUT
        # !! Variable order MUST match the order of floats boot_stats returns as OUTPUT
        interval_order = ["lower", "upper"]
        column_order = [self.ref_scorer, "difference", "residuals_abs"]
        variable_order = [
            "bias_mean",
            "loa_lower",
            "loa_upper",
            "bias_intercept",
            "bias_slope",
            "loa_intercept",
            "loa_slope",
            "loa_halfwidth",
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

        # Bootstrap CI for Euser LoA slope (log-transformed stats only).
        # Only compute for stats not already covered (boot_lower is NaN on first call).
        log_stats_to_boot = [
            s
            for s in sleep_stats
            if s in self._log_transform_stats and pd.isna(self._loa_log_ci.at[s, "boot_lower"])
        ]
        if log_stats_to_boot:
            agreement = self._agreement

            # Resample function: apply _euser_slope_scalar to each bootstrap replicate's SD.
            def _euser_resample(d):
                return self._euser_slope_scalar(np.std(d, ddof=1), agreement)

            for stat in log_stats_to_boot:
                log_d = self._data.loc[
                    self._data.index.get_level_values("sleep_stat") == stat, "log_difference"
                ].to_numpy()
                result = sps.bootstrap((log_d,), _euser_resample, **bs_kwargs)
                self._loa_log_ci.at[stat, "boot_lower"] = result.confidence_interval.low
                self._loa_log_ci.at[stat, "boot_upper"] = result.confidence_interval.high

    def report(
        self,
        bias_method="auto",
        loa_method="auto",
        ci_method="auto",
        decimals=2,
        sleep_stats=None,
    ):
        """
        Return a human-readable :py:class:`~pandas.DataFrame` for reporting bias, limits of
        agreement, and statistical assumption results, following the reporting format proposed by
        Menghini et al. (2021) [Menghini2021]_.

        Each row corresponds to one sleep statistic, labelled with its unit (e.g.
        ``"TST (min)"``). Reference and observed scorer means (SD) are shown first, followed by
        bias and LoA, optionally merged with their confidence intervals (e.g.
        ``"2.34 [1.10, 3.58]"``). An ``"Assumptions"`` column shows whether each statistical
        assumption was met (``"✓"``) or violated (``"✗"``), which drives the automatic
        method selection.

        Parameters
        ----------
        bias_method : str
            If ``'param'`` (parametric), bias is always the mean difference. If ``'regr'``
            (regression), bias is always a regression equation. If ``'auto'`` (default), the method
            is chosen per statistic based on the proportional-bias assumption test.
        loa_method : str
            Method used to compute limits of agreement. Options:

            * ``'param'`` — constant LoA: ``bias ± 1.96 SD``. Always uses this form regardless
              of assumptions or ``log_transform``. When the bias is a regression line, the LoA
              run parallel to it at ``± 1.96 SD`` of its residuals (Menghini et al. 2021, eq. 2)
              and are reported as ``"bias ± halfwidth"``.
            * ``'regr'`` — regression LoA: ``b0 + b1 × ref``. Always uses this form regardless
              of assumptions or ``log_transform``.
            * ``'log'`` — Euser LoA: ``bias ± slope × ref``. Requires ``log_transform=True``;
              raises ``ValueError`` otherwise.
            * ``'auto'`` (default) — if ``log_transform=True``, always uses ``'log'``. Otherwise,
              uses ``'param'`` when the homoscedasticity assumption passes and ``'regr'`` when it
              fails.
        ci_method : str or None
            If ``'param'``, parametric t-distribution CIs are used. If ``'boot'``, BCa bootstrap
            CIs are used. If ``'auto'`` (default), the method is chosen per statistic based on
            the normality assumption test. If ``None``, no confidence intervals are computed or
            shown (the columns are then named ``"Bias"`` and ``"LoA"``).
        decimals : int
            Number of decimal places. Default is 2.
        sleep_stats : list or None
            List of sleep statistics to include, in the desired row order. Default (None) is to
            include all sleep statistics.

            .. versionadded:: 0.8.0

        Returns
        -------
        report : :py:class:`pandas.DataFrame`
            A DataFrame indexed by ``"sleep_stat (unit)"`` with columns:

            * ``f"{ref_scorer} mean (SD)"`` — mean (SD) of the reference scorer values.
            * ``f"{obs_scorer} mean (SD)"`` — mean (SD) of the observed scorer values.
            * ``f"Bias [{pct}% CI]"`` (or ``"Bias"``) — mean bias or regression equation.
            * ``f"LoA [{pct}% CI]"`` (or ``"LoA"``) — lower–upper LoA, regression equation, or
              Euser proportional LoA.
            * ``"Assumptions"`` — pass/fail for the normal, constant bias and homoscedastic
              assumptions that drive the automatic method selection.

        Examples
        --------
        >>> import yasa
        >>> ref_hyps = [yasa.simulate_hypnogram(tib=480, scorer="PSG", seed=i) for i in range(20)]
        >>> obs_hyps = [h.simulate_similar(scorer="Device", seed=i) for i, h in enumerate(ref_hyps)]
        >>> sstats = yasa.EpochByEpochAgreement(ref_hyps, obs_hyps).get_sleep_stats()
        >>> ssa = yasa.SleepStatsAgreement(sstats)
        >>> ssa.report(
        ...     sleep_stats=["TST", "WASO", "SE"],
        ...     bias_method="param",
        ...     loa_method="param",
        ...     ci_method=None,
        ...     decimals=1,
        ... ).drop(columns="Assumptions")  # doctest: +SKIP
        """
        assert ci_method is None or ci_method in self._ci_method_opts, (
            f"`ci_method` must be one of {self._ci_method_opts} or None"
        )
        assert isinstance(decimals, int) and decimals >= 0, (
            "`decimals` must be a non-negative integer"
        )
        sleep_stats = self._check_sleep_stats(sleep_stats)
        bias_param_idx, loa_param_idx, loa_log_idx = self._resolve_methods(
            sleep_stats, bias_method, loa_method
        )
        pct = int(self._confidence * 100)
        loa_regr_agreement = self._agreement * np.sqrt(np.pi / 2)
        d = decimals

        # Unit lookup: covers all sleep statistics that may come from sleep_statistics().
        # %-based stats include percentage-prefixed names and efficiency measures (SE, SME);
        # SFI is in events/hour; all remaining stats (time-based) are in minutes.
        _pct_stats = {"SE", "SME"}

        def _unit(stat):
            if stat.startswith("%") or stat in _pct_stats:
                return "%"
            if stat == "SFI":
                return "events/h"
            return "min"

        # No CI computation (incl. bootstrap) when ci_method is None
        show_ci = ci_method is not None
        values = self.summary(ci_method=ci_method, sleep_stats=sleep_stats)
        values.columns = values.columns.map("_".join)

        # Reference and observed mean (SD) per sleep stat
        grouper = self._data.groupby("sleep_stat")
        ref_desc = grouper[self.ref_scorer].agg(["mean", "std"])
        obs_desc = grouper[self.obs_scorer].agg(["mean", "std"])
        passed = self._assumptions.xs("passed", level="metric", axis=1)

        def _check(b):
            return "✓" if b else "✗"  # ✓ or ✗

        def _ci(*bounds, prefixes=None):
            """Format one or more (lower, upper) CI pairs, e.g. ' [b0: 1.0, 2.0; b1: 0.1, 0.2]'."""
            pairs = [f"{lo:.{d}f}, {hi:.{d}f}" for lo, hi in bounds]
            if prefixes is not None:
                pairs = [f"{p}: {pair}" for p, pair in zip(prefixes, pairs, strict=True)]
            return " [" + "; ".join(pairs) + "]"

        rows = {}
        for stat in sleep_stats:
            v = values.loc[stat]
            unit = _unit(stat)
            label = f"{stat} ({unit})"

            if stat in bias_param_idx:
                bias_str = f"{v['bias_mean_center']:.{d}f}"
                if show_ci:
                    bias_str += _ci((v["bias_mean_lower"], v["bias_mean_upper"]))
            else:
                bias_str = f"{v['bias_intercept_center']:.{d}f} + {v['bias_slope_center']:.{d}f}x"
                if show_ci:
                    bias_str += _ci(
                        (v["bias_intercept_lower"], v["bias_intercept_upper"]),
                        (v["bias_slope_lower"], v["bias_slope_upper"]),
                        prefixes=("b0", "b1"),
                    )

            if stat in loa_log_idx:
                loa_str = f"bias ± {v['loa_log_slope_center']:.{d}f} × ref"
                if show_ci:
                    loa_str += _ci((v["loa_log_slope_lower"], v["loa_log_slope_upper"]))
            elif stat in loa_param_idx and stat in bias_param_idx:
                loa_str = f"{v['loa_lower_center']:.{d}f} to {v['loa_upper_center']:.{d}f}"
                if show_ci:
                    loa_str += _ci(
                        (v["loa_lower_lower"], v["loa_lower_upper"]),
                        (v["loa_upper_lower"], v["loa_upper_upper"]),
                    )
            elif stat in loa_param_idx:
                # Constant LoA parallel to the regression bias line (eq. 2)
                loa_str = f"bias ± {v['loa_halfwidth_center']:.{d}f}"
                if show_ci:
                    loa_str += _ci((v["loa_halfwidth_lower"], v["loa_halfwidth_upper"]))
            else:
                loa_str = (
                    f"±{loa_regr_agreement:.{d}f} "
                    f"({v['loa_intercept_center']:.{d}f} + {v['loa_slope_center']:.{d}f}x)"
                )
                if show_ci:
                    loa_str += _ci(
                        (v["loa_intercept_lower"], v["loa_intercept_upper"]),
                        (v["loa_slope_lower"], v["loa_slope_upper"]),
                        prefixes=("c0", "c1"),
                    )

            asmp = passed.loc[stat]
            assumptions_str = (
                f"{_check(asmp['normal'])} normal  "
                f"{_check(asmp['constant_bias'])} constant bias  "
                f"{_check(asmp['homoscedastic'])} homoscedastic"
            )

            rows[label] = {
                f"{self.ref_scorer} mean (SD)": (
                    f"{ref_desc.at[stat, 'mean']:.{d}f} ({ref_desc.at[stat, 'std']:.{d}f})"
                ),
                f"{self.obs_scorer} mean (SD)": (
                    f"{obs_desc.at[stat, 'mean']:.{d}f} ({obs_desc.at[stat, 'std']:.{d}f})"
                ),
                f"Bias [{pct}% CI]" if show_ci else "Bias": bias_str,
                f"LoA [{pct}% CI]" if show_ci else "LoA": loa_str,
                "Assumptions": assumptions_str,
            }

        result = pd.DataFrame.from_dict(rows, orient="index")
        result.index.name = "sleep_stat"
        return result

    def summary(self, ci_method="auto", sleep_stats=None):
        """
        Return a :py:class:`~pandas.DataFrame` that includes all calculated metrics:

        * Parametric bias
        * Parametric lower and upper limits of agreement
        * Half-width of the constant limits of agreement around the regression bias line, i.e.
          ``agreement × SD`` of the bias-regression residuals (Menghini et al. 2021, eq. 2)
        * Regression intercept and slope for modeled bias
        * Regression intercept and slope for modeled limits of agreement
        * Euser et al. (2008) slope for log-transformed limits of agreement (only when
          ``log_transform=True``)
        * Lower and upper confidence intervals for all metrics

        Parameters
        ----------
        ci_method : str or None
            If ``'param'`` (i.e., parametric), confidence intervals are always represented using a
            standard t-distribution.
            If ``'boot'`` (i.e., bootstrap), confidence intervals are always represented using a
            bootstrap resampling procedure.
            If  ``'auto'`` (default), confidence intervals are represented using a bootstrap
            resampling procedure for sleep statistics where the distribution of score differences is
            non-normal and using a standard t-distribution otherwise.
            If ``None``, no confidence intervals are computed and only the ``'center'`` interval
            is returned. This avoids the (potentially slow) bootstrap procedure entirely.
        sleep_stats : list or None
            List of sleep statistics to include, in the desired row order. Default (None) is to
            include all sleep statistics. Bootstrapped confidence intervals are only computed for
            the requested statistics.

            .. versionadded:: 0.8.0

        Returns
        -------
        summary : :py:class:`pandas.DataFrame`
            A :py:class:`~pandas.DataFrame` of numeric bias, limits of agreement, and their
            confidence intervals for all sleep statistics. Columns form a MultiIndex with levels
            ``variable`` and ``interval`` (``'center'``, ``'lower'``, ``'upper'``).

            When ``log_transform=True``, an additional ``loa_log_slope`` variable holds the Euser
            (2008) LoA slope (``LoA = bias ± slope × ref``) and its confidence interval.
        """
        assert ci_method is None or (
            isinstance(ci_method, str) and ci_method in self._ci_method_opts
        ), f"`ci_method` must be one of {self._ci_method_opts} or None"
        sleep_stats = self._check_sleep_stats(sleep_stats)
        ci_methods = self._assumptions[("normal", "method")]  # per-stat CI method for "auto"
        # Make sure relevant sleep statistics have bootstrapped CIs, and generate them if not
        if ci_method in ["boot", "auto"]:
            if ci_method == "boot":
                sleep_stats_to_boot = sleep_stats
            elif ci_method == "auto":
                boot_idx_all = ci_methods.index[ci_methods == "boot"]
                sleep_stats_to_boot = [s for s in sleep_stats if s in boot_idx_all]
            # Remove any sleep stats already bootstrapped CIs (eg if "boot" is callaed after "auto")
            sleep_stats_booted = self._ci["boot"].dropna().index
            sleep_stats_to_boot = [s for s in sleep_stats_to_boot if s not in sleep_stats_booted]
            if sleep_stats_to_boot:
                self._generate_bootstrap_ci(sleep_stats=sleep_stats_to_boot)
        # Add an extra level to values columns, indicating they are the center interval
        summary = pd.concat({"center": self._vals}, names=["interval"], axis=1).swaplevel(axis=1)
        if ci_method is not None:
            if ci_method == "auto":
                param_idx = ci_methods.index[ci_methods == "param"].to_list()
                boot_idx = [ss for ss in self.sleep_statistics if ss not in param_idx]
                ci_param = self._ci.loc[param_idx, "param"]
                ci_boot = self._ci.loc[boot_idx, "boot"]
                ci_vals = pd.concat([ci_param, ci_boot])
            else:
                ci_vals = self._ci[ci_method]
            summary = summary.join(ci_vals, how="left", validate="1:1")
        # Add the Euser LoA slope (and CI) when the log-transform analysis was run
        if self._log_transform:
            log_slope = {("loa_log_slope", "center"): self._loa_log_slope}
            if ci_method is not None:
                # Per-stat CI method: "param" or "boot" (chosen by the normality test for "auto")
                if ci_method == "auto":
                    methods = ci_methods
                else:
                    methods = pd.Series(ci_method, index=self._loa_log_ci.index)
                for interval in ["lower", "upper"]:
                    log_slope[("loa_log_slope", interval)] = pd.Series(
                        {
                            s: self._loa_log_ci.at[s, f"{methods[s]}_{interval}"]
                            for s in self._loa_log_ci.index
                        },
                        name=interval,
                    )
            log_slope = pd.DataFrame(log_slope)
            log_slope.columns = pd.MultiIndex.from_tuples(
                log_slope.columns, names=["variable", "interval"]
            )
            summary = summary.join(log_slope, how="left", validate="1:1")
        summary = summary.astype(float).sort_index(axis=1)
        return summary.loc[sleep_stats]

    def calibrate(self, data, bias_method="auto"):
        """
        Calibrate a :py:class:`~pandas.DataFrame` of sleep statistics from a new scorer based on
        observed biases in ``obs_data``/``obs_scorer``.

        Parameters
        ----------
        data : :py:class:`pandas.DataFrame`
            A :py:class:`pandas.DataFrame` with sleep statistics from an observed scorer.
            Rows are unique observations and columns are unique sleep statistics.
        bias_method : str
            If ``'param'``, sleep statistics are always adjusted based on parametric bias.
            If ``'regr'``, sleep statistics are always adjusted based on regression-modeled bias.
            If ``'auto'`` (default), bias sleep statistics are adjusted by either ``'param'`` or
            ``'regr'``, depending on assumption violations.

            .. seealso:: :py:meth:`~yasa.SleepStatsAgreement.summary`

        Returns
        -------
        calibrated_data : :py:class:`pandas.DataFrame`
            A :py:class:`~pandas.DataFrame` with calibrated sleep statistics: ``x - bias_mean``
            for the parametric bias and ``(x - b0) / (1 + b1)`` for the regression bias.
        """
        assert isinstance(data, pd.DataFrame), "`data` must be a pandas DataFrame"
        assert all(col in self.sleep_statistics for col in data), (
            f"all columns of `data` must be valid sleep statistics: {self.sleep_statistics}"
        )
        assert isinstance(bias_method, str), "`bias_method` must be a string"
        assert bias_method in self._bias_method_opts, (
            f"`bias_method` must be one of {self._bias_method_opts}"
        )
        param_adjusted = data - self._vals["bias_mean"]
        regr_adjusted = (data - self._vals["bias_intercept"]) / (1 + self._vals["bias_slope"])
        if bias_method == "param":
            calibrated_data = param_adjusted
        elif bias_method == "regr":
            calibrated_data = regr_adjusted
        elif bias_method == "auto":
            use_param = self._assumptions.loc[data.columns, ("constant_bias", "method")].eq("param")
            calibrated_data = param_adjusted.where(use_param, regr_adjusted, axis=1)
        return calibrated_data

    def plot_blandaltman(
        self,
        sleep_stats=None,
        bias_method="auto",
        loa_method="auto",
        ci_method="auto",
        scatter_kwargs=None,
        **kwargs,
    ):
        """Plot Bland-Altman agreement plots for one or more sleep statistics.

        Each panel shows observed-minus-reference differences (y-axis) against reference values
        (x-axis) for one sleep statistic. Bias and limits of agreement are drawn as lines, with
        optional confidence-interval bands. Methods (parametric, regression, or bootstrap) are
        chosen automatically per statistic based on the assumption tests stored in
        :py:attr:`~yasa.SleepStatsAgreement.assumptions`, or can be set explicitly.

        .. seealso:: :py:meth:`~yasa.SleepStatsAgreement.report`,
            :py:meth:`~yasa.SleepStatsAgreement.summary`

        Parameters
        ----------
        sleep_stats : list or None
            List of sleep statistics to plot. Default (None) is to plot all sleep statistics.
        bias_method : str
            If ``'param'``, bias is always the mean difference (horizontal line with an optional
            CI band). If ``'regr'``, bias is always a regression line (no CI band). If ``'auto'``
            (default), the method is chosen per statistic based on the proportional-bias
            assumption test.
        loa_method : str
            Method used to draw limits of agreement. Options:

            * ``'param'`` — constant LoA: horizontal lines at ``bias ± 1.96 SD``. Always uses
              this form regardless of assumptions or ``log_transform``. When the bias is a
              regression line, the LoA run parallel to it at ``± 1.96 SD`` of its residuals
              (Menghini et al. 2021, eq. 2).
            * ``'regr'`` — regression LoA: lines following ``b0 + b1 × ref``. Always uses this
              form regardless of assumptions or ``log_transform``.
            * ``'log'`` — Euser LoA: lines following ``bias ± slope × ref``. Requires
              ``log_transform=True``; raises ``ValueError`` otherwise.
            * ``'auto'`` (default) — if ``log_transform=True``, always uses ``'log'``. Otherwise,
              uses ``'param'`` when the homoscedasticity assumption passes and ``'regr'`` when it
              fails.
        ci_method : str or None
            If ``'param'``, parametric CIs are drawn. If ``'boot'``, bootstrap CIs are drawn. If
            ``'auto'`` (default), chosen per statistic based on the normality assumption test.
            If ``None``, no confidence intervals are drawn.
        scatter_kwargs : dict
            Other keyword arguments are passed through to :py:func:`matplotlib.pyplot.scatter`.
        **kwargs : dict
            Other keyword arguments are passed through to :py:class:`seaborn.FacetGrid`.

        Returns
        -------
        g : :py:class:`seaborn.FacetGrid`
            Seaborn FacetGrid

        Examples
        --------
        .. plot::

            >>> import yasa
            >>> n = 20
            >>> ref_hyps = [yasa.simulate_hypnogram(scorer="PSG", seed=i) for i in range(n)]
            >>> obs_hyps = [ref_hyps[i].simulate_similar(scorer="Device", seed=i) for i in range(n)]
            >>> eea = yasa.EpochByEpochAgreement(ref_hyps, obs_hyps)
            >>> sstats = eea.get_sleep_stats()
            >>> ssa = yasa.SleepStatsAgreement(sstats)
            >>> stats = ["TST", "WASO", "N1", "REM"]
            >>> g = ssa.plot_blandaltman(sleep_stats=stats, ci_method="param")
        """
        import seaborn as sns  # noqa
        import matplotlib.pyplot as plt

        assert ci_method is None or (
            isinstance(ci_method, str) and ci_method in self._ci_method_opts
        ), f"`ci_method` must be one of {self._ci_method_opts} or None"
        assert isinstance(scatter_kwargs, (dict, type(None))), (
            "`scatter_kwargs` must be a dict or None"
        )
        if scatter_kwargs is None:
            scatter_kwargs = {}
        sleep_stats = self._check_sleep_stats(sleep_stats)
        bias_param_idx, loa_param_idx, loa_log_idx = self._resolve_methods(
            sleep_stats, bias_method, loa_method
        )

        # Retrieve values and CIs (only the "center" interval when ci_method is None)
        vals = self.summary(ci_method=ci_method, sleep_stats=sleep_stats)

        agreement_adj = self._agreement * np.sqrt(np.pi / 2)

        # Select scatterplot arguments and update with optional input
        default_scatter_kwargs = dict(s=12, facecolor="none", edgecolor="black", alpha=0.8)
        scatter_kwargs = default_scatter_kwargs | scatter_kwargs
        # Choose a balanced grid layout with at most 4 columns: use as few rows as possible, then
        # as few columns as needed to fill those rows (e.g. 6 -> 2x3, 8 -> 2x4, 9 -> 3x3).
        n_stats = len(sleep_stats)
        if n_stats > 4:
            n_rows = int(np.ceil(n_stats / 4))
            col_wrap = int(np.ceil(n_stats / n_rows))
        else:
            col_wrap = None
        # Select FacetGrid arguments and update with optional input
        default_facetgrid_kwargs = dict(
            data=self._data.reset_index("sleep_stat"),
            col="sleep_stat",
            col_order=sleep_stats,
            col_wrap=col_wrap,
            height=4,
            aspect=1,
            sharex=False,
            sharey=False,
        )
        facetgrid_kwargs = default_facetgrid_kwargs | kwargs
        # Choose display levels with zorder
        data_zorder = 30
        bias_zorder = 20
        loa_zorder = 10
        refline_zorder = 0
        # Initialize a grid of plots with an Axes for each sleep statistic
        g = sns.FacetGrid(**facetgrid_kwargs)
        # Draw scatterplot on each axis
        g.map(plt.scatter, self.ref_scorer, "difference", zorder=data_zorder, **scatter_kwargs)
        # Draw a horizontal line at y=0 on each axis
        g.refline(
            y=0, color="black", linewidth=0.75, linestyle=":", alpha=0.6, zorder=refline_zorder
        )
        # Choose arguments for all calls to axhspan and fill_between for bias and LoA CI bands
        band_kwargs = dict(edgecolor="none", alpha=0.15)
        # Choose arguments for all calls to axhline and plot for bias and LoA lines
        line_kwargs = dict(linewidth=1, linestyle="dashed", alpha=0.9)
        bias_line_kwargs = line_kwargs | dict(linestyle="solid")
        loa_color = "tab:blue"
        bias_color = "tab:gray"
        # Draw bias lines, LoA lines, and CI bands on each axis
        for stat, ax in zip(sleep_stats, g.axes.flat, strict=True):
            x_min, x_max = ax.get_xlim()
            x_line = np.array([x_min, x_max])
            v = vals.loc[stat]
            has_ci = ci_method is not None

            # --- Bias line ---
            if stat in bias_param_idx:
                y_bias = v[("bias_mean", "center")]
                ax.axhline(y_bias, color=bias_color, zorder=bias_zorder, **bias_line_kwargs)
                if has_ci:
                    ax.axhspan(
                        v[("bias_mean", "lower")],
                        v[("bias_mean", "upper")],
                        facecolor=bias_color,
                        zorder=bias_zorder - 1,
                        **band_kwargs,
                    )
                y_bias_arr = np.full_like(x_line, y_bias, dtype=float)
            else:
                # Regression bias: no CI band. The intercept and slope CIs are reported separately
                # in `summary()` and `report()`; combining them into a single band would not be a
                # valid confidence region for the fitted line.
                intercept = v[("bias_intercept", "center")]
                slope = v[("bias_slope", "center")]
                y_bias_arr = intercept + slope * x_line
                ax.plot(
                    x_line, y_bias_arr, color=bias_color, zorder=bias_zorder, **bias_line_kwargs
                )

            # --- LoA lines ---
            if stat in loa_log_idx:
                # Euser LoA: proportional lines at bias ± slope * ref.
                # Unlike constant LoA (axhline), these fan out with the reference value.
                slope_c = v[("loa_log_slope", "center")]
                ax.plot(
                    x_line,
                    y_bias_arr + slope_c * x_line,
                    color=loa_color,
                    zorder=loa_zorder,
                    **line_kwargs,
                )
                ax.plot(
                    x_line,
                    y_bias_arr - slope_c * x_line,
                    color=loa_color,
                    zorder=loa_zorder,
                    **line_kwargs,
                )
                if has_ci:
                    slope_lo = v[("loa_log_slope", "lower")]
                    slope_hi = v[("loa_log_slope", "upper")]
                    # Upper LoA CI band: between bias + slope_lo*ref and bias + slope_hi*ref.
                    ax.fill_between(
                        x_line,
                        y_bias_arr + slope_lo * x_line,
                        y_bias_arr + slope_hi * x_line,
                        facecolor=loa_color,
                        zorder=loa_zorder - 1,
                        **band_kwargs,
                    )
                    # Lower LoA CI band: mirror of the upper band (slope signs flipped).
                    ax.fill_between(
                        x_line,
                        y_bias_arr - slope_hi * x_line,
                        y_bias_arr - slope_lo * x_line,
                        facecolor=loa_color,
                        zorder=loa_zorder - 1,
                        **band_kwargs,
                    )
            elif stat in loa_param_idx and stat in bias_param_idx:
                for loa_var in ("loa_lower", "loa_upper"):
                    y_loa = v[(loa_var, "center")]
                    ax.axhline(y_loa, color=loa_color, zorder=loa_zorder, **line_kwargs)
                    if has_ci:
                        ax.axhspan(
                            v[(loa_var, "lower")],
                            v[(loa_var, "upper")],
                            facecolor=loa_color,
                            zorder=loa_zorder - 1,
                            **band_kwargs,
                        )
            elif stat in loa_param_idx:
                # Constant LoA parallel to the regression bias line (eq. 2)
                halfwidth = v[("loa_halfwidth", "center")]
                for sign in (1, -1):
                    ax.plot(
                        x_line,
                        y_bias_arr + sign * halfwidth,
                        color=loa_color,
                        zorder=loa_zorder,
                        **line_kwargs,
                    )
                    if has_ci:
                        ax.fill_between(
                            x_line,
                            y_bias_arr + sign * v[("loa_halfwidth", "lower")],
                            y_bias_arr + sign * v[("loa_halfwidth", "upper")],
                            facecolor=loa_color,
                            zorder=loa_zorder - 1,
                            **band_kwargs,
                        )
            else:
                loa_int = v[("loa_intercept", "center")]
                loa_slp = v[("loa_slope", "center")]
                y_spread = agreement_adj * np.maximum(0.0, loa_int + loa_slp * x_line)
                ax.plot(
                    x_line, y_bias_arr + y_spread, color=loa_color, zorder=loa_zorder, **line_kwargs
                )
                ax.plot(
                    x_line, y_bias_arr - y_spread, color=loa_color, zorder=loa_zorder, **line_kwargs
                )
                if has_ci:
                    lint_lo = v[("loa_intercept", "lower")]
                    lint_hi = v[("loa_intercept", "upper")]
                    lslp_lo = v[("loa_slope", "lower")]
                    lslp_hi = v[("loa_slope", "upper")]
                    spread_a = agreement_adj * (lint_lo + lslp_lo * x_line)
                    spread_b = agreement_adj * (lint_hi + lslp_hi * x_line)
                    spread_lo = np.minimum(spread_a, spread_b)
                    spread_hi = np.maximum(spread_a, spread_b)
                    ax.fill_between(
                        x_line,
                        y_bias_arr + spread_lo,
                        y_bias_arr + spread_hi,
                        facecolor=loa_color,
                        zorder=loa_zorder - 1,
                        **band_kwargs,
                    )
                    ax.fill_between(
                        x_line,
                        y_bias_arr - spread_hi,
                        y_bias_arr - spread_lo,
                        facecolor=loa_color,
                        zorder=loa_zorder - 1,
                        **band_kwargs,
                    )

        # Tidy-up axis limits with symmetric y-axis and minimal ticks
        for ax in g.axes.flat:
            bound = max(map(abs, ax.get_ylim()))
            ax.set_ylim(-bound, bound)
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4, integer=True, symmetric=True))
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=3, integer=True))
        # More aesthetics
        ylabel = " - ".join((self.obs_scorer, self.ref_scorer))
        g.set_ylabels(ylabel)
        g.set_xlabels(self.ref_scorer)
        g.set_titles(col_template="{col_name}")
        if hasattr(g.fig, "align_titles"):  # introduced in matplotlib v3.9.0
            g.fig.align_titles()
        g.fig.align_labels()
        g.tight_layout(w_pad=1, h_pad=2)
        return g
