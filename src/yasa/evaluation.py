"""
YASA code for evaluating the agreement between two scorers (e.g., human vs YASA), either at the
epoch-by-epoch level or at the level of summary sleep statistics.

Analyses are influenced by the standardized framework proposed in Menghini et al., 2021, SLEEP.
See the following resources:
- https://doi.org/10.1093/sleep/zsaa170
- https://sri-human-sleep.github.io/sleep-trackers-performance
- https://github.com/SRI-human-sleep/sleep-trackers-performance
"""

import functools
import logging
import warnings

import numpy as np
import pandas as pd
import scipy.stats as sps
import sklearn.metrics as skm

from .hypno import Hypnogram
from .io import is_statsmodels_installed

logger = logging.getLogger("yasa")

__all__ = [
    "EpochByEpochAgreement",
    "SleepStatsAgreement",
]


def _subjects_to_series(subjects, index):
    """Convert a user-supplied ``subjects`` argument to a Series of subject IDs indexed by session.

    Parameters
    ----------
    subjects : None, dict, pandas.Series, or array-like
        A dict (or Series) keyed by the session IDs in ``index``, or a 1D array-like with one
        subject ID per session, in the order of ``index``.
    index : array-like
        The session IDs.

    Returns
    -------
    subjects : pandas.Series or None
        Subject IDs indexed by session ID, named ``"subject_id"``. None if ``subjects`` is None.
    """
    if subjects is None:
        return None
    index = pd.Index(index)
    if isinstance(subjects, (dict, pd.Series)):
        subjects = pd.Series(subjects)
        assert subjects.index.is_unique and set(subjects.index) == set(index), (
            "the keys of `subjects` must be the session IDs"
        )
        subjects = subjects.reindex(index)
    else:
        # Object dtype, so that a missing value among strings stays NaN (instead of "nan")
        subjects = np.asarray(subjects, dtype=object)
        assert subjects.ndim == 1 and subjects.size == index.size, (
            "`subjects` must have exactly one subject ID per session"
        )
        subjects = pd.Series(subjects, index=index).infer_objects()
    assert subjects.notna().all(), "`subjects` must not contain missing values"
    return subjects.rename("subject_id").rename_axis(index.name)


################################################################################
# EPOCH BY EPOCH
################################################################################


class EpochByEpochAgreement:
    """Evaluate agreement between two hypnograms or two collections of hypnograms.

    Evaluation includes averaged agreement scores, one-vs-rest agreement scores, agreement scores
    summarized across all sleep and summarized by sleep stage, and various plotting options to
    visualize the two hypnograms simultaneously. See examples for more detail.

    For a complete walkthrough of the evaluation pipeline of Menghini et al. (2021), see the
    :ref:`tutorial_evaluation` tutorial.

    .. versionadded:: 0.7.0

    Parameters
    ----------
    ref_hyps : iterable of :py:class:`yasa.Hypnogram`
        A collection of reference hypnograms (i.e., those considered ground-truth).

        Each :py:class:`yasa.Hypnogram` in ``ref_hyps`` must have the same
        :py:attr:`~yasa.Hypnogram.scorer`.

        If a ``dict``, key values are used to generate unique sleep session IDs. If any other
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
    subjects : None, dict, or array-like
        The subject of each sleep session, for datasets with several nights per subject (repeated
        measures). Either a ``dict`` (or :py:class:`pandas.Series`) that maps each session ID (the
        keys of ``ref_hyps``, or ``1, 2, ...`` if ``ref_hyps`` is a list) to a subject ID, or an
        array-like with one subject ID per hypnogram, in the same order as ``ref_hyps``. If
        ``None`` (default), each session is treated as coming from a different subject.

        When ``subjects`` is given, :py:meth:`summary` and
        :py:meth:`get_confusion_matrix_proportional` average the nights of each subject first, and
        then average across subjects, so that each subject counts equally regardless of its number
        of nights. Their bootstrap resamples subjects instead of nights. The subject IDs are also
        added to the output of :py:meth:`get_sleep_stats`, which passes them on to
        :py:class:`SleepStatsAgreement`. See :ref:`tutorial_evaluation_repeated`.

        .. versionadded:: 0.8.1

    Notes
    -----
    Epochs scored as artefact (``ART``) or unscored (``UNS``) by either scorer are excluded from all
    epoch-by-epoch analyses (agreement scores and confusion matrices), since they are not a sleep
    stage. They are kept in the hypnograms themselves, e.g. in :py:meth:`get_sleep_stats` and
    :py:meth:`plot_hypnograms`.

    .. versionchanged:: 0.8.0
        ``ART`` and ``UNS`` epochs were previously counted as regular stages.

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

    def __init__(self, ref_hyps, obs_hyps, subjects=None):
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
            # Pair by key, not by position: the keys comparison above ignores order
            obs_hyps = tuple(obs_hyps[k] for k in sleep_ids)
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
        # Subject of each session (repeated measures), as a Series indexed by sleep_id, or None
        subjects = _subjects_to_series(subjects, pd.Index(sleep_ids, name="sleep_id"))
        if subjects is not None and subjects.nunique() == len(sleep_ids):
            logger.warning("Each subject has a single session: `subjects` has no effect.")

        # Merge all hypnograms into a single MultiIndexed dataframe
        ref = pd.concat(pd.concat({s: h.as_int()}, names=["sleep_id"]) for s, h in ref_hyps.items())
        obs = pd.concat(pd.concat({s: h.as_int()}, names=["sleep_id"]) for s, h in obs_hyps.items())
        data = pd.concat([ref, obs], axis=1)

        # Exclude the epochs scored as artefact or unscored by either scorer: they are not a sleep
        # stage, so they should neither count as agreement nor as disagreement.
        mapping = ref_hyps[sleep_ids[0]].mapping
        excluded_codes = [mapping[lab] for lab in ("ART", "UNS") if lab in mapping]
        is_excluded = data.isin(excluded_codes).any(axis=1)
        if is_excluded.any():
            n_left = (~is_excluded).groupby(level="sleep_id", sort=False).sum()
            if (n_left == 0).any():
                raise ValueError(
                    "All epochs are scored as ART or UNS in session(s) "
                    f"{n_left[n_left == 0].index.tolist()}, which can therefore not be evaluated."
                )
            logger.info(
                f"Excluding {is_excluded.sum()} epoch(s) scored as ART or UNS by either scorer."
            )
            data = data[~is_excluded]

        # Generate some mapping dictionaries to be used later in class methods
        skm_labels = np.unique(data).tolist()  # all unique YASA integer codes in this hypno
        skm2yasa_map = {i: lab for i, lab in enumerate(skm_labels)}  # skm order to YASA integers
        yasa2yasa_map = ref_hyps[sleep_ids[0]].mapping_int.copy()  # YASA integer to YASA string

        # Set attributes
        self._data = data
        self._sleep_ids = sleep_ids
        self._subjects = subjects
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
        subjects = f" from {self.n_subjects} subjects" if self._subjects is not None else ""
        return (
            f"<EpochByEpochAgreement | Observed hypnogram{s} scored by {self.obs_scorer} "
            f"evaluated against reference hypnogram{s} scored by {self.ref_scorer}, "
            f"{self.n_sessions} sleep session{s}{subjects}>\n"
            " - Use `.get_agreement()` to get agreement measures as a pandas DataFrame or Series\n"
            " - Use `.plot_hypnograms()` to plot two overlaid hypnograms\n"
            "See the online documentation for more details."
        )

    def __str__(self):
        return self.__repr__()

    @property
    def data(self):
        """A :py:class:`pandas.DataFrame` including all hypnograms, without the epochs scored as
        ``ART`` or ``UNS`` by either scorer."""
        return self._data

    @property
    def n_sessions(self):
        """The number of unique sleep sessions."""
        return len(self._sleep_ids)

    @property
    def n_subjects(self):
        """The number of unique subjects. Equal to :py:attr:`n_sessions` when ``subjects`` was not
        given.

        .. versionadded:: 0.8.1
        """
        return self.n_sessions if self._subjects is None else self._subjects.nunique()

    @property
    def ref_scorer(self):
        """The name of the reference scorer."""
        return self._ref_scorer

    @property
    def obs_scorer(self):
        """The name of the observed scorer."""
        return self._obs_scorer

    def _mean_by_subject(self, wide):
        """Average the rows (sessions) of ``wide`` within each subject, if ``subjects`` was given.

        The rows of the result (sessions, or subjects) are the units of the descriptive statistics
        and of the bootstrap in :py:meth:`summary` and :py:meth:`get_confusion_matrix_proportional`.
        """
        if self._subjects is None:
            return wide
        return wide.groupby(self._subjects.reindex(wide.index)).mean()

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
            :py:attr:`~yasa.EpochByEpochAgreement.data`, which excludes ``ART`` and ``UNS``
            epochs.
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
            # Pool all epochs across sessions and compute a single set of scores. Not stored for
            # summary(), which needs per-session scores.
            return pd.Series(self.multi_scorer(df, scorers=scorers), name="agreement")
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
            :py:class:`~pandas.MultiIndex` with sleep stage and session as rows (only sleep stage
            if a single session is evaluated).

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
        confusion matrices, as reported in Menghini et al. (2021).

        For each session, the confusion matrix is first normalized by row, so that each cell is
        the percentage of reference-scorer epochs of a given stage that the observed scorer
        assigned to each stage (each row sums to 100). These per-session percentages are then
        averaged across sessions, so that every session contributes equally regardless of its
        duration. This differs from
        ``get_confusion_matrix(agg_func="sum")``, which pools all epochs together.

        Rows of the reference stage that are absent from a session (e.g. no N3 sleep in a given
        night) are undefined (0 / 0) and are excluded from the mean, SD, and CI of that row rather
        than being counted as zeros. The number of contributing sessions is returned in the
        ``n_sessions`` column.

        With several nights per subject (``subjects``), the mean, SD and CI are computed across
        subjects rather than sessions, and the number of contributing subjects is returned in an
        extra ``n_subjects`` column.

        .. versionadded:: 0.8.0

        .. versionchanged:: 0.8.1
            Support for several nights per subject (``subjects``).

        Parameters
        ----------
        ci_method : str or None
            Method used to compute the confidence interval of the group mean of each cell.

            * ``'boot'`` (default) — non-parametric bootstrap across sessions (or subjects), i.e.
              sessions are resampled with replacement (a "participant bootstrap"). The same
              resamples are used for all cells, so that each bootstrapped matrix remains
              row-normalized. Replicates in which a reference stage is absent from every resampled
              session are undefined and are excluded from the percentiles of that row.
            * ``'param'`` — parametric interval ``mean ± t × SD / sqrt(n)`` based on a Student
              t-distribution with ``n - 1`` degrees of freedom, clipped to [0, 100], ``n`` being
              the number of sessions (or subjects).
            * ``None`` — no confidence interval is computed.
        confidence : float
            Confidence level (between 0 and 1) of the confidence interval. Default is 0.95.
        bootstrap_kwargs : dict or None
            Optional settings of the bootstrap procedure when ``ci_method='boot'``. Valid keys are:

            * ``'n_resamples'`` — number of bootstrap resamples (int, default 1000).
            * ``'method'`` — ``'BCa'`` (default) for bias-corrected and accelerated percentiles
              (Efron, 1987), which corrects for the skewness and bias of the bootstrap
              distribution but can be unstable with few sessions (e.g. fewer than 20);
              ``'percentile'`` for plain percentiles; or ``'basic'`` for the reverse-percentile
              method used by the reference pipeline of Menghini et al. (2021). Cells that are
              constant across all resamples (e.g. a stage that is never confused) fall back to
              plain percentiles.
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
            ``ci_method`` is not None), ``n_sessions`` and, when ``subjects`` was given,
            ``n_subjects``. All values except the counts are percentages (0-100). Use e.g.
            ``conf_matr["mean"].unstack()`` to get the mean matrix in square format.

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
        assert self.n_subjects > 1, (
            "A proportional confusion matrix can not be computed with only one hypnogram pair "
            "or one subject."
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
        # Flatten the stage dimensions into one row per session, each column being one cell of
        # the matrix (shape (n_sessions, n_stages * n_stages))
        wide = pd.DataFrame.from_dict(
            {
                sid: grp.droplevel("sleep_id").loc[stages, stages].to_numpy(dtype=float).ravel()
                for sid, grp in props.groupby(level="sleep_id", sort=False)
            },
            orient="index",
        )
        n_sessions_valid = wide.notna().sum().to_numpy()
        # One row per subject with several nights per subject (see `_mean_by_subject`)
        arr2d = self._mean_by_subject(wide).to_numpy(dtype=float)
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
        out["n_sessions"] = n_sessions_valid
        if self._subjects is not None:
            out["n_subjects"] = n_valid
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
            session (one for reference scorer and another for observed scorer), indexed by
            ``(scorer, sleep_id)``. When ``subjects`` was given, a ``subject_id`` level is inserted
            in between, i.e. ``(scorer, subject_id, sleep_id)``. :py:class:`SleepStatsAgreement`
            reads the subjects from this level to account for the repeated measures.
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
        elif self._subjects is not None:
            # Insert the subject level: (scorer, subject_id, sleep_id)
            subj = sstats.index.get_level_values("sleep_id").map(self._subjects)
            sstats = sstats.set_index(pd.Index(subj, name="subject_id"), append=True)
            sstats = sstats.reorder_levels(["scorer", "subject_id", "sleep_id"])
        return sstats

    def plot_hypnograms(
        self, sleep_id=None, legend=True, ax=None, ref_kwargs=None, obs_kwargs=None
    ):
        """Plot the two hypnograms of one session overlapping on the same axis.

        .. seealso:: :py:meth:`yasa.Hypnogram.plot_hypnogram`

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
            Keyword arguments passed to :py:meth:`yasa.Hypnogram.plot_hypnogram` when plotting the
            reference hypnogram.
        obs_kwargs : dict
            Keyword arguments passed to :py:meth:`yasa.Hypnogram.plot_hypnogram` when plotting the
            observed hypnogram.

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
            sessions (or subjects).

            * ``None`` (default) — no confidence interval is computed.
            * ``'boot'`` — non-parametric bootstrap across sessions (or subjects). The
              same resamples are used for every metric (and stage), so each replicate remains
              internally consistent. Sessions in which a metric is undefined (see the
              ``zero_division`` parameter of :py:meth:`get_agreement_bystage`) are ignored, and
              replicates in which a metric has no defined value are excluded from its percentiles.

            .. versionadded:: 0.8.0
        confidence : float
            Confidence level (between 0 and 1) of the confidence interval. Default is 0.95.

            .. versionadded:: 0.8.0
        bootstrap_kwargs : dict or None
            Optional settings of the bootstrap procedure when ``ci_method='boot'``. Valid keys are:

            * ``'n_resamples'`` — number of bootstrap resamples (int, default 10000).
            * ``'method'`` — ``'BCa'`` (default) for bias-corrected and accelerated percentiles
              (Efron, 1987), ``'percentile'`` for plain percentiles, or ``'basic'`` for the
              reverse-percentile method. Metrics that are constant across all resamples fall back
              to plain percentiles. BCa can be unstable with few sessions (or subjects), and a
              warning is emitted below 20.
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
            with descriptive statistics. Each row is an agreement metric (or a (stage, metric)
            pair if ``by_stage=True``) and each column is a descriptive statistic (e.g., mean,
            standard deviation).

            When ``ci_method`` is not ``None``, two extra columns ``ci_lower`` and ``ci_upper``
            are appended. They are left missing for ``support``, which is not an agreement
            score.

            With several nights per subject (``subjects``), the descriptive statistics and the
            bootstrap are computed across subjects rather than sessions, and ``support`` is the
            average number of epochs per night of each subject.

            .. versionchanged:: 0.8.1
                Support for several nights per subject (``subjects``).

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
        assert self.n_subjects > 1, (
            "Summary scores can not be computed with only one hypnogram pair or one subject."
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
        # Reshape so that rows are sessions (or subjects, see `_mean_by_subject`) and columns are
        # the quantities to summarize. The rows are the resampling unit of `_bootstrap_ci_mean`,
        # and the same resampled rows are used for every column.
        if by_stage:
            wide = self._agreement_bystage.unstack("stage")  # columns are (metric, stage)
            # Reorder to match the (stage, metric) summary index
            ci_index = wide.columns.swaplevel().rename(["stage", "metric"])
        else:
            wide = self._agreement
            ci_index = pd.Index(wide.columns, name="metric")
        wide = self._mean_by_subject(wide)
        if by_stage:
            summary = (
                wide.stack("stage", future_stack=True)
                .groupby("stage")
                .agg(**agg_kwargs)
                .stack(level=0, future_stack=True)
                .rename_axis(["stage", "metric"])
            )
        else:
            summary = wide.agg(**agg_kwargs).T.rename_axis("metric")
        if ci_method is None:
            return summary

        if boot_method == "BCa" and self.n_subjects < 20:
            unit = "sessions" if self._subjects is None else "subjects"
            warnings.warn(
                f"The BCa bootstrap interval can be unstable with few {unit} "
                f"(n_{unit}={self.n_subjects}). Consider "
                "bootstrap_kwargs={'method': 'percentile'} or the parametric "
                "summary(func=['mean', 'sem']) instead.",
                RuntimeWarning,
                stacklevel=2,
            )
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
    various plotting options, and a calibration function for correcting a constant bias of the
    observed scorer.

    Features include:

    * Get summary calculations of bias, limits of agreement, and their confidence intervals.
    * Test statistical assumptions of bias, limits of agreement, and their confidence intervals, and apply corrective procedures when the assumptions are not met.
    * Get bias and limits of agreement in a string-formatted table.
    * Calibrate new data to correct for a constant bias in observed data.
    * Visualize Bland-Altman plots.

    For a complete walkthrough of the evaluation pipeline of Menghini et al. (2021), see the
    :ref:`tutorial_evaluation` tutorial.

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
    subjects : None, dict, or array-like
        The subject of each session, for datasets with several nights per subject (repeated
        measures). Either a ``dict`` (or :py:class:`pandas.Series`) that maps each session ID (the
        index of ``ref_data``) to a subject ID, or an array-like with one subject ID per row of
        ``ref_data``, in the same order. If ``None`` (default), each session is treated as coming
        from a different subject. Not needed when ``ref_data`` is the output of
        :py:meth:`EpochByEpochAgreement.get_sleep_stats` with a ``subject_id`` level, since the
        subjects are then read from the index. See the Notes and
        :ref:`tutorial_evaluation_repeated`.

        .. versionadded:: 0.8.1
    confidence : float
        Confidence level (between 0 and 1) for the confidence intervals applied to bias and limits
        of agreement. Default is 0.95 (i.e., 95%). The same level is used for both parametric and
        bootstrapped confidence intervals.
    alpha : float
        Alpha cutoff used for all assumption tests. Default is 0.05.
    effect_size_gates : dict or None
        Effect-size thresholds that must be exceeded, *in addition* to ``pvalue < alpha``, for
        the normality, constant-bias and homoscedasticity assumptions to be considered not
        met. Because the power of these tests grows with the number of sessions, the
        p-value alone would flag small and practically irrelevant deviations in large samples:
        the p-value assesses the statistical evidence for an association, the effect size its
        practical magnitude. Pass a dict to override one or more of the defaults, or ``None``
        (the default) to use all of them. Setting a threshold to ``None`` disables that
        criterion. Keys:

        * ``'skew'`` (default 1.0) and ``'kurtosis'`` (default 2.0) — normality is only considered
          not met if the differences are markedly asymmetric (``|skew| >`` threshold) or
          heavy-tailed (excess ``kurtosis >`` threshold).
        * ``'r2'`` (default 0.1) — proportional bias is only present if at least 10% of the
          variability of the differences is explained by the reference value.
        * ``'sd_ratio'`` (default 1.5) — heteroscedasticity is only present if the fitted
          absolute residual changes by more than 50% between the lower and the upper end of the
          observed reference range, i.e. if the typical error varies materially across the range.
          The criterion is applied symmetrically (a ratio above 1.5 or below 1/1.5), because the
          variability can either grow or shrink with the reference value.

        .. versionadded:: 0.8.0
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
        As in the reference pipeline of Menghini et al. (2021), ``bias`` is the mean difference on
        the original scale and the slope is multiplied by the reference value. This is an
        approximation: Euser et al. (2008) derived the slope for differences relative to the mean
        of both scorers, and the exact back-transformed limits relative to the reference value
        are slightly asymmetric. When ``True``, ``loa_method='auto'`` in :py:meth:`report` and
        :py:meth:`plot_blandaltman` will automatically select the Euser method for all
        statistics, bypassing the homoscedasticity assumption test. Statistics with a value of
        exactly zero in either scorer (e.g. SOL for a subject who fell asleep in the first epoch)
        can not be log-transformed; they are excluded with a warning and keep the regular LoA
        methods. Passing ``loa_method='log'`` for such statistics, or when
        ``log_transform=False``, raises a ``ValueError``. Default is ``False``.

    Notes
    -----
    Sleep statistics that are identical between scorers are removed from analysis. Sessions with a
    missing value in either scorer (e.g. ``Lat_REM`` for a night without REM sleep) are removed
    separately for each sleep statistic, with a warning, so the number of sessions can differ
    between statistics.

    **Several nights per subject.** When ``subjects`` is given and at least one subject has
    several nights, the nights are not independent and the following methods are used instead
    (Bland & Altman, 2007; Myles & Cui, 2007; Parker et al., 2016):

    * The SD of the differences, which defines the limits of agreement, is the total SD of the
      one-way random-effects ANOVA of Bland & Altman (2007), i.e. the SD of a new night of a new
      subject (see :py:attr:`variance_components`). The bias is the unweighted mean of all
      differences, so that subjects with more nights weigh more, as in Bland & Altman (2007).
      This differs from the intercept of a mixed model, which weighs subjects by precision.
    * The parametric confidence interval of the bias uses the model-based standard error of the
      bias, with ``n_subjects - 1`` degrees of freedom. The confidence intervals of the limits of
      agreement, LoA half-width and Euser LoA slope use the MOVER method of Zou (2013).
    * The regressions behind the proportional bias and heteroscedasticity tests, and behind the
      regression-based bias and LoA, are linear mixed models with a random intercept per subject,
      fitted with `statsmodels <https://www.statsmodels.org/>`_ (``pip install yasa[stats]``).
      Their confidence intervals are always the Wald intervals of the mixed model, even with
      ``ci_method="boot"`` (see :py:meth:`summary`).
    * The bootstrap confidence intervals resample subjects instead of nights.
    * The normality test (Shapiro-Wilk) is applied to all differences, as without repeated
      measures. Its p-value is only approximate when the nights are clustered within subjects, but
      the skewness and kurtosis gates (see ``effect_size_gates``) limit the consequences.

    References
    ----------
    .. [Menghini2021] Menghini, L., Cellini, N., Goldstone, A., Baker, F. C., & de Zambotti, M.
                      (2021). A standardized framework for testing the performance of sleep-tracking
                      technology: step-by-step guidelines and open-source code. SLEEP, 44(2),
                      zsaa170. https://doi.org/10.1093/sleep/zsaa170
    .. [BlandAltman2007] Bland, J. M., & Altman, D. G. (2007). Agreement between methods of
                         measurement with multiple observations per individual. Journal of
                         Biopharmaceutical Statistics, 17(4), 571-582.
                         https://doi.org/10.1080/10543400701329422
    .. [Zou2013] Zou, G. Y. (2013). Confidence interval estimation for the Bland-Altman limits of
                 agreement with multiple observations per individual. Statistical Methods in
                 Medical Research, 22(6), 630-642. https://doi.org/10.1177/0962280211402548
    .. [Parker2016] Parker, R. A., Weir, C. J., Rubio, N., et al. (2016). Application of mixed
                    effects limits of agreement in the presence of multiple sources of variability:
                    exemplar from the comparison of several devices to measure respiratory rate in
                    COPD patients. PLoS ONE, 11(12), e0168321.
                    https://doi.org/10.1371/journal.pone.0168321
    .. [MylesCui2007] Myles, P. S., & Cui, J. (2007). Using the Bland-Altman method to measure
                      agreement with repeated measures. British Journal of Anaesthesia, 99(3),
                      309-311. https://doi.org/10.1093/bja/aem214

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
    >>> summary = ssa.summary(ci_method="param").round(1)  # 24 columns, 3 per variable
    >>> summary[["bias_mean", "loa_lower", "loa_upper"]].head(3)  # doctest: +NORMALIZE_WHITESPACE
    variable   bias_mean             loa_lower             loa_upper
    interval      center lower upper    center lower upper    center lower upper
    sleep_stat
    %N1              0.3  -1.1   1.6      -5.5  -7.9  -3.1       6.1   3.7   8.5
    %N2             -0.2  -3.2   2.8     -12.9 -18.1  -7.6      12.4   7.2  17.6
    %N3              1.4  -3.2   5.9     -17.7 -25.6  -9.8      20.4  12.6  28.3

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

    >>> new_stats_calibrated = ssa.calibrate(new_sstats[["N1", "TST", "WASO"]], bias_method="auto")
    >>> new_stats_calibrated.round(1).head(5)
         N1    TST   WASO
    0  42.1  445.2  145.0
    1  83.6  555.8   36.0
    2  53.1  494.8  100.5
    3  56.6  475.2  117.5
    4  70.6  536.8   66.5

    With several nights per subject, pass the subject of each night to
    :py:class:`EpochByEpochAgreement`, or directly to this class with ``subjects``. Here, we
    simulate 4 subjects with 3 nights each:

    >>> ref_hyps = [yasa.simulate_hypnogram(tib=480, scorer="PSG", seed=i) for i in range(12)]
    >>> obs_hyps = [h.simulate_similar(scorer="Device", seed=i) for i, h in enumerate(ref_hyps)]
    >>> subjects = [f"S{i // 3}" for i in range(12)]
    >>> ebe = yasa.EpochByEpochAgreement(ref_hyps, obs_hyps, subjects=subjects)
    >>> ssa = yasa.SleepStatsAgreement(ebe.get_sleep_stats())  # doctest: +SKIP
    >>> ssa.variance_components.loc[["TST", "WASO", "N3"]].round(1)  # doctest: +SKIP
                sd_between  sd_within  sd_total  n_subjects  n_sessions
    sleep_stat
    TST                0.0       55.0      55.0           4          12
    WASO               0.0       56.8      56.8           4          12
    N3                 6.8       41.5      42.0           4          12
    """

    _bias_method_opts = ("param", "regr", "auto")
    _loa_method_opts = ("param", "regr", "log", "auto")
    _ci_method_opts = ("param", "boot", "auto")

    # Multiple of the SD of the differences defining the limits of agreement (95% coverage)
    _agreement = 1.96

    # Default effect-size thresholds gating the assumption tests (see `effect_size_gates`)
    _default_gates = {"skew": 1.0, "kurtosis": 2.0, "r2": 0.1, "sd_ratio": 1.5}

    def __init__(
        self,
        ref_data,
        obs_data=None,
        *,
        ref_scorer="Reference",
        obs_scorer="Observed",
        subjects=None,
        confidence=0.95,
        alpha=0.05,
        effect_size_gates=None,
        bootstrap_kwargs=None,
        log_transform=False,
    ):
        restricted_bootstrap_kwargs = ["confidence_level", "vectorized", "paired"]
        bootstrap_kwargs = {} if bootstrap_kwargs is None else bootstrap_kwargs
        agreement = self._agreement

        assert isinstance(ref_data, pd.DataFrame), "`ref_data` must be a pandas DataFrame"
        if obs_data is None:
            # Output of EpochByEpochAgreement.get_sleep_stats(): (scorer, session) MultiIndex, or
            # (scorer, subject, session) when it was given the subject of each session
            assert ref_data.index.nlevels in (2, 3), (
                "`ref_data` must have a (scorer, session) or (scorer, subject, session) MultiIndex "
                "when `obs_data` is None"
            )
            scorers = ref_data.index.get_level_values(0).unique().tolist()
            assert len(scorers) == 2, f"`ref_data` must contain exactly two scorers, got {scorers}"
            ref_scorer, obs_scorer = scorers
            ref_data, obs_data = ref_data.loc[ref_scorer], ref_data.loc[obs_scorer]
            if ref_data.index.nlevels == 2:
                assert subjects is None, (
                    "`subjects` must be None when `ref_data` already has a subject level"
                )
                assert ref_data.index.equals(obs_data.index), (
                    "the (subject, session) index must be identical for the two scorers"
                )
                # One subject per row, validated below by `_subjects_to_series`
                subjects = ref_data.index.get_level_values(0)
                ref_data, obs_data = ref_data.droplevel(0), obs_data.droplevel(0)
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
        effect_size_gates = {} if effect_size_gates is None else effect_size_gates
        assert isinstance(effect_size_gates, dict), "`effect_size_gates` must be a dict or None"
        assert all(k in self._default_gates for k in effect_size_gates), (
            f"`effect_size_gates` keys must be a subset of {list(self._default_gates)}"
        )
        assert all(
            v is None or (isinstance(v, (float, int)) and not isinstance(v, bool) and v >= 0)
            for v in effect_size_gates.values()
        ), "`effect_size_gates` values must be non-negative numbers or None"
        gates = self._default_gates | effect_size_gates
        assert isinstance(bootstrap_kwargs, dict), "`bootstrap_kwargs` must be a dictionary"
        assert all(k not in restricted_bootstrap_kwargs for k in bootstrap_kwargs), (
            f"None of {restricted_bootstrap_kwargs} can be set by the user"
        )
        assert isinstance(log_transform, bool), "`log_transform` must be a bool"
        # If `ref_data` and `obs_data` indices are unnamed, name them (on copies, not in place)
        session_key = "session_id" if ref_data.index.name is None else ref_data.index.name
        ref_data, obs_data = ref_data.rename_axis(session_key), obs_data.rename_axis(session_key)

        # Subject of each session, as a Series indexed by session, or None. The repeated-measures
        # methods are only used when at least one subject has several sessions.
        subjects = _subjects_to_series(subjects, ref_data.index)
        repeated = subjects is not None and subjects.nunique() < subjects.size
        if subjects is not None and not repeated:
            logger.warning("Each subject has a single session: `subjects` has no effect.")
        if repeated:
            is_statsmodels_installed()

        # Reshape to long format DataFrame with 2 columns (observed, reference) and MultiIndex
        data = (
            pd.concat([obs_data, ref_data], keys=[obs_scorer, ref_scorer], names=["scorer"])
            .melt(var_name="sleep_stat", ignore_index=False)
            .pivot_table(index=["sleep_stat", session_key], columns="scorer", values="value")
            .rename_axis(columns=None)
            .sort_index()
        )
        if subjects is not None:
            data["subject_id"] = data.index.get_level_values(session_key).map(subjects).to_numpy()

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

        def _per_stat(func):
            """Apply ``func(df)`` to each sleep stat and stack the returned dicts in a DataFrame."""
            frame = pd.DataFrame.from_dict({s: func(df) for s, df in grouper}, orient="index")
            return frame.rename_axis("sleep_stat")

        def _by_subject(df):
            return df["subject_id"].to_numpy() if repeated else None

        def _vc_frame(col):
            """Variance components of column ``col`` per sleep stat (see `_variance_components`)."""
            return _per_stat(
                lambda df: self._variance_components(df[col].to_numpy(), _by_subject(df))
            )

        def _regr_frame(col):
            """Regression of column ``col`` on the reference values, per sleep stat (`_regress`)."""
            return _per_stat(
                lambda df: self._regress(
                    df[ref_scorer].to_numpy(), df[col].to_numpy(), _by_subject(df)
                )
            )

        ########################################################################
        # Generate parametric Bias and LoA for all sleep stats
        ########################################################################
        # Parametric Bias: mean difference
        param_vals = grouper["difference"].mean().to_frame("bias_mean")
        # SD of the differences: the sample SD, or the ANOVA total SD with repeated measures
        vc_diff = _vc_frame("difference")
        sd_diff = np.sqrt(vc_diff["var_total"])
        # Parametric LoA
        param_vals["loa_lower"] = param_vals["bias_mean"] - agreement * sd_diff
        param_vals["loa_upper"] = param_vals["bias_mean"] + agreement * sd_diff

        ########################################################################
        # Generate standard CIs for parametric Bias and LoA for all sleep stats
        ########################################################################
        # SE of the bias: SD / sqrt(n) (Bland & Altman 1999), or the variance of the mean of all
        # differences under the one-way random-effects model, sqrt(var_between * sum(m_i^2) / N^2
        # + var_within / N), with m_i the number of sessions of subject i and N the total number
        # of sessions (Zou 2013 instead uses the mean of the subject means). The critical t has
        # n_subjects - 1 degrees of freedom in both cases. A stat without repeated nights after
        # NaN removal has NaN components: fall back to SD / sqrt(n).
        se_bias = sd_diff / np.sqrt(n)
        if repeated:
            se_rm = np.sqrt(
                vc_diff["var_between"] * vc_diff["sum_m2"] / n**2 + vc_diff["var_within"] / n
            )
            se_bias = se_rm.fillna(se_bias)
        n_subj = vc_diff["n_subjects"]
        t_param = pd.Series(sps.t.ppf((1 + confidence) / 2, n_subj - 1), index=n.index)
        param_ci = pd.DataFrame(
            {
                "bias_mean-lower": param_vals["bias_mean"] - se_bias * t_param,
                "bias_mean-upper": param_vals["bias_mean"] + se_bias * t_param,
            }
        )
        if repeated:
            # MOVER CI of the LoA (Zou 2013): combines the CI of the bias with the chi-square CI
            # of the total variance, itself a MOVER combination of the two mean squares.
            var_lower, var_upper = self._variance_ci(vc_diff, confidence)
            (
                param_ci["loa_lower-lower"],
                param_ci["loa_lower-upper"],
                param_ci["loa_upper-lower"],
                param_ci["loa_upper-upper"],
            ) = self._mover_loa_ci(
                param_vals["bias_mean"],
                param_ci["bias_mean-lower"],
                param_ci["bias_mean-upper"],
                sd_diff,
                np.sqrt(var_lower),
                np.sqrt(var_upper),
                agreement,
            )
        else:
            # SE of a LoA = sqrt(3) * SD / sqrt(n) (Bland & Altman 1999)
            se_loa = se_bias * np.sqrt(3)
            param_ci["loa_lower-lower"] = param_vals["loa_lower"] - se_loa * t_param
            param_ci["loa_lower-upper"] = param_vals["loa_lower"] + se_loa * t_param
            param_ci["loa_upper-lower"] = param_vals["loa_upper"] - se_loa * t_param
            param_ci["loa_upper-upper"] = param_vals["loa_upper"] + se_loa * t_param

        ########################################################################
        # Generate regression/modeled (slope and intercept) Bias and LoA for all sleep stats
        ########################################################################
        # Run regression used to (a) model bias and (b) test for proportional/constant bias
        bias_regr = _regr_frame("difference")
        # Get absolute residuals from this regression bc they are used in the next regression.
        # With a mixed model these are computed from the fixed effects only (marginal residuals).
        idx = data.index.get_level_values("sleep_stat")
        slopes = bias_regr.loc[idx, "slope"].to_numpy()
        intercepts = bias_regr.loc[idx, "intercept"].to_numpy()
        predicted_values = data[ref_scorer].to_numpy() * slopes + intercepts
        data["residuals"] = data["difference"].to_numpy() - predicted_values
        data["residuals_abs"] = data["residuals"].abs()
        # Run regression used to (a) model LoA and (b) test for heteroscedasticity/homoscedasticity
        loa_regr = _regr_frame("residuals_abs")
        # Stack the two regression dataframes together
        regr = pd.concat({"bias": bias_regr, "loa": loa_regr}, axis=0)

        ########################################################################
        # Generate parametric CIs for regression/modeled Bias and LoA for all sleep stats
        ########################################################################
        # Critical t for the regression CIs, with the degrees of freedom returned by `_regress`
        # (n - 2 for OLS, n_subjects - 2 for the mixed model)
        t_regr = pd.Series(sps.t.ppf((1 + confidence) / 2, n - 2), index=n.index)
        t_stacked = sps.t.ppf((1 + confidence) / 2, regr["dof"].to_numpy(dtype=float))
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
        vc_resid = _vc_frame("residuals")
        param_vals["loa_halfwidth"] = agreement * np.sqrt(vc_resid["var_total"])
        if repeated:
            var_lower, var_upper = self._variance_ci(vc_resid, confidence)
            param_ci["loa_halfwidth-lower"] = agreement * np.sqrt(var_lower)
            param_ci["loa_halfwidth-upper"] = agreement * np.sqrt(var_upper)
        else:
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
            # SD of the log-ratios (total SD with repeated measures), per log-transformed stat
            vc_log = _vc_frame("log_difference").loc[log_transform_stats]
            sd = np.sqrt(vc_log["var_total"])
            if repeated:
                var_lower, var_upper = self._variance_ci(vc_log, confidence)
                sd_lower, sd_upper = np.sqrt(var_lower), np.sqrt(var_upper)
            else:
                # SE of a limit of agreement (Bland & Altman 1999): sqrt(3 * SD^2 / n). It is
                # already on the scale of z = agreement * SD, so it is added to z directly, as in
                # the reference pipeline of Menghini et al. (2021), with the t(n - 1) critical
                # value. Clamp the lower bound at 0 so the slope stays non-negative.
                t_log = t_param[log_transform_stats]
                se_z = np.sqrt(sd**2 * 3 / n[log_transform_stats])
                sd_lower = (sd - t_log * se_z / agreement).clip(lower=0.0)
                sd_upper = sd + t_log * se_z / agreement
            # Point estimate: back-transform SD of log-ratios to a proportional slope, and
            # back-transform the CI bounds of the SD the same way.
            loa_log_slope[log_transform_stats] = self._euser_slope_scalar(sd, agreement)
            loa_log_ci.loc[log_transform_stats, "param_lower"] = self._euser_slope_scalar(
                sd_lower, agreement
            )
            loa_log_ci.loc[log_transform_stats, "param_upper"] = self._euser_slope_scalar(
                sd_upper, agreement
            )

        ########################################################################
        # Test all statistical assumptions
        ########################################################################
        # For each assumption: test statistic, p-value, effect size, pass/fail flag and the
        # method selected when "auto" is requested. Because the power of these tests grows with
        # the number of sessions, the normality, proportional-bias and heteroscedasticity
        # assumptions use a dual criterion: they are only flagged as not met when the test is
        # significant (p < alpha) *and* the effect size is material (see `effect_size_gates`).
        def _test_series(res):
            return pd.Series({"statistic": res.statistic, "pvalue": res.pvalue})

        # Test of the bias against zero: t = bias / SE(bias) with n_subjects - 1 degrees of
        # freedom (the one-sample t-test without repeated measures)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_stat = param_vals["bias_mean"] / se_bias
        ttest = pd.DataFrame(
            {"statistic": t_stat, "pvalue": 2 * sps.t.sf(np.abs(t_stat), n_subj - 1)}
        )
        shapiro = grouper["difference"].apply(
            lambda a: (
                _test_series(sps.shapiro(a))
                if len(a) >= 3
                else pd.Series({"statistic": np.nan, "pvalue": 1.0})
            )
        )
        shapiro = shapiro.unstack()
        skew = grouper["difference"].skew()
        kurtosis = grouper["difference"].apply(pd.Series.kurt)
        # Ratio of the fitted absolute residuals at the upper and lower ends of the observed
        # reference range. Because the mean absolute residual is proportional to the residual SD,
        # this is the factor by which the typical error -- and hence the LoA half-width -- changes
        # across the range. Fitted values are clipped to a tiny positive number so that a
        # regression line crossing zero yields a large (rather than a negative) ratio.
        ref_range = grouper[ref_scorer].agg(["min", "max"])
        fitted_min, fitted_max = (
            (loa_regr["intercept"] + loa_regr["slope"] * ref_range[bound]).clip(lower=1e-9)
            for bound in ("min", "max")
        )
        sd_ratio = fitted_max / fitted_min

        def _material(*criteria):
            """Combine the effect-size criteria of one assumption, ignoring disabled (None) ones.

            Criteria are OR-ed together: a single material effect size is enough. When every
            criterion is disabled, the result is all-True and the p-value alone decides.
            """
            criteria = [c for c in criteria if c is not None]
            if not criteria:
                return pd.Series(True, index=n.index)
            return pd.concat(criteria, axis=1).any(axis=1)

        # Normality: heavy tails (excess kurtosis) or marked asymmetry (|skew|)
        normal_material = _material(
            None if gates["skew"] is None else skew.abs().gt(gates["skew"]),
            None if gates["kurtosis"] is None else kurtosis.gt(gates["kurtosis"]),
        )
        # Proportional bias: fraction of the variability in the differences explained by the
        # reference value. Prevents a small but detectable slope from being treated as important
        # solely because of the sample size.
        bias_material = _material(
            None if gates["r2"] is None else (bias_regr["rvalue"] ** 2).gt(gates["r2"])
        )
        # Heteroscedasticity: applied symmetrically, because the variability of the differences
        # can either grow or shrink with the reference value.
        hetero_material = _material(
            None
            if gates["sd_ratio"] is None
            else np.maximum(sd_ratio, 1 / sd_ratio).gt(gates["sd_ratio"])
        )

        unbiased = ttest["pvalue"].ge(alpha)
        normal = shapiro["pvalue"].ge(alpha) | ~normal_material
        constant_bias = bias_regr["pvalue"].ge(alpha) | ~bias_material
        homoscedastic = loa_regr["pvalue"].ge(alpha) | ~hetero_material
        loa_method = homoscedastic.map({True: "param", False: "regr"})
        loa_method[log_transform_stats] = "log"
        assumptions = pd.DataFrame(
            {
                ("unbiased", "t"): ttest["statistic"],
                ("unbiased", "pvalue"): ttest["pvalue"],
                ("unbiased", "cohen_d"): param_vals["bias_mean"] / sd_diff,
                ("unbiased", "passed"): unbiased,
                ("normal", "W"): shapiro["statistic"],
                ("normal", "pvalue"): shapiro["pvalue"],
                ("normal", "skew"): skew,
                ("normal", "kurtosis"): kurtosis,
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
                ("homoscedastic", "sd_ratio"): sd_ratio,
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

        # Between-, within-subject and total SD of the differences, per sleep statistic
        variance_components = pd.DataFrame(
            {
                "sd_between": np.sqrt(vc_diff["var_between"]),
                "sd_within": np.sqrt(vc_diff["var_within"]),
                "sd_total": sd_diff,
                "n_subjects": vc_diff["n_subjects"].astype(int),
                "n_sessions": n,
            }
        )

        # Set attributes
        self._confidence = confidence
        self._effect_size_gates = gates
        self._bootstrap_kwargs = bootstrap_kwargs
        self._n_sessions = n_sessions
        self._subjects = subjects
        self._repeated = repeated
        self._ref_scorer = ref_scorer
        self._obs_scorer = obs_scorer
        self._data = data
        self._assumptions = assumptions
        self._regr = regr
        self._vals = vals
        self._ci = ci
        self._vc = variance_components
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
    def n_subjects(self):
        """The number of unique subjects. Equal to :py:attr:`n_sessions` when ``subjects`` was not
        given.

        .. versionadded:: 0.8.1
        """
        return self._n_sessions if self._subjects is None else self._subjects.nunique()

    @property
    def variance_components(self):
        """Between- and within-subject components of the variability of the differences.

        A :py:class:`pandas.DataFrame` with one row per sleep statistic and the following columns:

        * ``sd_between`` — SD of the mean difference of each subject, i.e. the stable offsets of
          the observed scorer for a given subject. NaN when no subject has several valid nights
          for that statistic (e.g. without ``subjects``).
        * ``sd_within`` — SD of the differences across the nights of a subject. NaN as above.
        * ``sd_total`` — ``sqrt(sd_between² + sd_within²)``, the SD of a new night of a new
          subject, which defines the limits of agreement (``bias ± 1.96 × sd_total``). Without
          repeated measures, this is the sample SD of the differences.
        * ``n_subjects`` and ``n_sessions`` — number of subjects and sessions with a valid value.

        The components are estimated with the one-way random-effects ANOVA of Bland & Altman
        (2007), which handles unequal numbers of nights per subject. Together, they tell whether
        the disagreement between scorers comes mostly from stable subject-level offsets or from
        night-to-night variability (Parker et al., 2016).

        .. versionadded:: 0.8.1
        """
        return self._vc.copy()

    @property
    def effect_size_gates(self):
        """The effect-size thresholds used together with ``alpha`` in the assumption tests.

        .. versionadded:: 0.8.0
        """
        return self._effect_size_gates.copy()

    @property
    def data(self):
        """A long-format :py:class:`pandas.DataFrame` containing all raw sleep statistics from
        ``ref_data`` and ``obs_data``, with a :py:class:`~pandas.MultiIndex` with levels
        ``sleep_stat`` and ``session_id`` (or the original index name from the input data).
        Columns are the reference and observed scorer names, plus ``subject_id`` when
        ``subjects`` was given.
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
          reference values: ``slope``, ``pvalue``, ``r2``, ``sd_ratio`` (ratio of the fitted
          absolute residuals at the upper and lower ends of the observed reference range),
          ``passed`` and the limits-of-agreement ``method`` (``'param'`` if passed, ``'regr'``
          otherwise, ``'log'`` for log-transformed statistics when ``log_transform=True``).

        ``method`` is what :py:meth:`report`, :py:meth:`summary`, :py:meth:`calibrate` and
        :py:meth:`plot_blandaltman` apply when ``'auto'`` is requested.

        ``passed`` is ``True`` when ``pvalue >= alpha`` for the ``unbiased`` assumption. The
        three assumptions that drive the automatic method selection use a dual criterion instead:
        they only fail when ``pvalue < alpha`` **and** the effect size exceeds the corresponding
        threshold in ``effect_size_gates`` (``skew``/``kurtosis`` for ``normal``, ``r2`` for
        ``constant_bias``, ``sd_ratio`` for ``homoscedastic``). This prevents a statistically
        detectable but practically negligible deviation from switching methods in large samples.

        .. versionchanged:: 0.8.0
            Includes the test statistics, effect sizes and selected methods (previously only the
            pass/fail flags), and gates ``passed`` on the effect sizes.
        """
        return self._assumptions

    def __repr__(self):
        # TODO v0.8: Keep only the text between < and >
        subjects = f" from {self.n_subjects} subjects" if self._subjects is not None else ""
        return (
            f"<SleepStatsAgreement | Observed scorer ('{self.obs_scorer}') evaluated against "
            f"reference scorer ('{self.ref_scorer}'), {self.n_sessions} sleep sessions{subjects}>\n"
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
    def _euser_slope_scalar(sd, agreement):
        """Euser et al. (2008) antilog LoA slope: 2*(exp(z)-1)/(exp(z)+1), z = agreement * sd.

        Converts the SD of log-ratio differences back to a proportional LoA slope in the original
        scale. The limits of agreement are then ``bias ± slope × ref``, where ``slope`` grows with
        the variability of the log-ratios. When SD is 0, ``z = 0`` and the slope is 0 (no spread).
        Using ``ref`` (rather than the mean of both scorers) follows Menghini et al. (2021).
        """
        z = agreement * sd
        return 2.0 * (np.exp(z) - 1.0) / (np.exp(z) + 1.0)

    @staticmethod
    def _variance_components(values, subjects=None):
        """Decompose the variance of ``values`` into between- and within-subject components.

        One-way random-effects ANOVA with subject as the grouping factor (Bland & Altman 2007).
        With ``m_i`` values for subject ``i``, ``n`` subjects and ``N = sum(m_i)`` values in
        total::

            MSW = sum_i sum_j (x_ij - mean_i)^2 / (N - n)      # within-subject mean square
            MSB = sum_i m_i (mean_i - mean)^2 / (n - 1)        # between-subject mean square
            m0 = (N - sum_i m_i^2 / N) / (n - 1)               # = m_i for balanced data
            var_within = MSW
            var_between = max(0, (MSB - MSW) / m0)
            var_total = var_between + var_within               # variance of a single new value
                      = max(MSB / m0 + (1 - 1 / m0) * MSW, MSW)

        ``var_total`` is also returned as ``a * MSB + b * MSW``, for its confidence interval in
        :py:meth:`_variance_ci`. When ``subjects`` is None, or when no subject has more than one
        value, ``var_total`` is the sample variance (``a = 1``, ``b = 0``, ``N - 1`` degrees of
        freedom) and the between/within components are NaN.

        Returns
        -------
        vc : dict
            Keys ``var_total``, ``var_between``, ``var_within``, ``msb``, ``msw``, ``a``, ``b``,
            ``dof_b``, ``dof_w``, ``n_subjects`` and ``sum_m2`` (``sum_i m_i^2``).
        """
        values = np.asarray(values, dtype=float)
        n_pairs = values.size
        if subjects is not None:
            _, inverse, counts = np.unique(
                np.asarray(subjects), return_inverse=True, return_counts=True
            )
            n_subjects = counts.size
        if subjects is None or n_subjects < 2 or n_subjects == n_pairs:
            var = np.var(values, ddof=1) if n_pairs > 1 else np.nan
            return {
                "var_total": var,
                "var_between": np.nan,
                "var_within": np.nan,
                "msb": var,
                "msw": np.nan,
                "a": 1.0,
                "b": 0.0,
                "dof_b": n_pairs - 1,
                "dof_w": 0,
                "n_subjects": n_pairs if subjects is None else n_subjects,
                "sum_m2": n_pairs,
            }
        grand_mean = values.mean()
        means = np.bincount(inverse, weights=values) / counts
        msw = np.sum((values - means[inverse]) ** 2) / (n_pairs - n_subjects)
        msb = np.sum(counts * (means - grand_mean) ** 2) / (n_subjects - 1)
        m0 = (n_pairs - np.sum(counts**2) / n_pairs) / (n_subjects - 1)
        a, b = 1 / m0, 1 - 1 / m0
        # A negative between-subject estimate (MSB < MSW) is set to zero, as is usual for ANOVA
        # estimators, so that the total variance is never smaller than the within-subject variance
        return {
            "var_total": max(a * msb + b * msw, msw),
            "var_between": max(0.0, (msb - msw) / m0),
            "var_within": msw,
            "msb": msb,
            "msw": msw,
            "a": a,
            "b": b,
            "dof_b": n_subjects - 1,
            "dof_w": n_pairs - n_subjects,
            "n_subjects": n_subjects,
            "sum_m2": np.sum(counts**2),
        }

    @staticmethod
    def _total_sd(values, subjects=None):
        """Square root of the ``var_total`` of :py:meth:`_variance_components`."""
        return np.sqrt(SleepStatsAgreement._variance_components(values, subjects)["var_total"])

    @staticmethod
    def _variance_ci(vc, confidence):
        """Confidence interval of the total variance ``a * MSB + b * MSW`` of ``vc``.

        The exact chi-square confidence intervals of the two mean squares are combined with the
        MOVER method (method of variance estimates recovery; Zou 2013, eq. 7), which reduces to
        the chi-square interval when ``b = 0`` (no repeated measures):

            L = V - sqrt((a * MSB - a * l_B)^2 + (b * MSW - b * l_W)^2)
            U = V + sqrt((a * u_B - a * MSB)^2 + (b * u_W - b * MSW)^2)

        ``vc`` is a dict of scalars (one statistic) or a DataFrame with one row per statistic
        (see :py:meth:`_variance_components`). Returns the lower and upper bounds, as arrays.
        """
        alpha = 1 - confidence
        msb, msw = np.asarray(vc["msb"], dtype=float), np.asarray(vc["msw"], dtype=float)
        a, b = np.asarray(vc["a"], dtype=float), np.asarray(vc["b"], dtype=float)
        dof_b, dof_w = np.asarray(vc["dof_b"], dtype=float), np.asarray(vc["dof_w"], dtype=float)
        var = np.asarray(vc["var_total"], dtype=float)  # a * MSB + b * MSW, clipped at MSW

        def _chi2_ci(ms, dof):
            with np.errstate(divide="ignore", invalid="ignore"):
                lower = ms * dof / sps.chi2.ppf(1 - alpha / 2, dof)
                upper = ms * dof / sps.chi2.ppf(alpha / 2, dof)
            return lower, upper

        l_b, u_b = _chi2_ci(msb, dof_b)
        l_w, u_w = _chi2_ci(msw, dof_w)
        # The within term vanishes when b = 0 (no repeated measures): avoid 0 * NaN
        has_w = b > 0
        l_w, u_w = np.where(has_w, l_w, 0.0), np.where(has_w, u_w, 0.0)
        b_msw = np.where(has_w, b * msw, 0.0)
        # Half-widths of the MOVER interval around the unclipped combination, applied to the
        # (possibly clipped) point estimate so that the interval always contains it
        lower = var - np.sqrt((a * msb - a * l_b) ** 2 + (b_msw - b * l_w) ** 2)
        upper = var + np.sqrt((a * u_b - a * msb) ** 2 + (b * u_w - b_msw) ** 2)
        return np.maximum(lower, 0.0), upper

    @staticmethod
    def _mover_loa_ci(bias, bias_lower, bias_upper, sd, sd_lower, sd_upper, agreement):
        """MOVER confidence intervals of the limits of agreement ``bias ± agreement * sd``.

        Combines the confidence interval of the bias with that of the SD of a single difference
        (Zou 2013, eq. 10-11). For the upper limit ``U = bias + agreement * sd``::

            U_lower = U - sqrt((bias - bias_lower) ^ 2 + agreement ^ 2(sd - sd_lower) ^ 2)
            U_upper = U + sqrt((bias_upper - bias) ^ 2 + agreement ^ 2(sd_upper - sd) ^ 2)

        and symmetrically for the lower limit, where a larger SD pushes the limit further down.

        Returns
        -------
        lower_lower, lower_upper, upper_lower, upper_upper
            Bounds of the confidence intervals of the lower and upper limits of agreement.
        """
        bias_err_lo, bias_err_hi = bias - bias_lower, bias_upper - bias
        sd_err_lo, sd_err_hi = agreement * (sd - sd_lower), agreement * (sd_upper - sd)
        upper, lower = bias + agreement * sd, bias - agreement * sd
        upper_lower = upper - np.sqrt(bias_err_lo**2 + sd_err_lo**2)
        upper_upper = upper + np.sqrt(bias_err_hi**2 + sd_err_hi**2)
        lower_lower = lower - np.sqrt(bias_err_lo**2 + sd_err_hi**2)
        lower_upper = lower + np.sqrt(bias_err_hi**2 + sd_err_lo**2)
        return lower_lower, lower_upper, upper_lower, upper_upper

    @staticmethod
    def _linregress(x, y):
        """Run :py:func:`scipy.stats.linregress` and return a dictionary of the results.

        ``dof`` is the degrees of freedom of the t critical value of the coefficient CIs.
        """
        if np.ptp(x) == 0:
            # Constant reference values (e.g. N3 = 0 in every night): the slope is not
            # identifiable. Use a flat line through the mean of y (no proportional effect), so that
            # the assumption tests pass and the parametric methods are used.
            return {
                "slope": 0.0,
                "intercept": np.mean(y),
                "rvalue": 0.0,
                "pvalue": 1.0,
                "stderr": np.nan,
                "intercept_stderr": np.nan,
                "dof": np.nan,
                "method": "constant",
            }
        regr = sps.linregress(x, y)
        return {
            "slope": regr.slope,
            "intercept": regr.intercept,
            "rvalue": regr.rvalue,
            "pvalue": regr.pvalue,
            "stderr": regr.stderr,
            "intercept_stderr": regr.intercept_stderr,
            "dof": x.size - 2,
            "method": "ols",
        }

    @staticmethod
    def _regress(x, y, subjects=None):
        """Regress ``y`` on ``x`` and return a dictionary of the results.

        Without ``subjects`` (or with fewer than 3 subjects, or when no subject has more than one
        observation), this is the ordinary least-squares regression of :py:meth:`_linregress`.
        With repeated measures, a linear mixed model with a random intercept per subject is fitted
        instead (:py:class:`statsmodels.regression.mixed_linear_model.MixedLM`, REML). The
        ``slope`` and ``intercept`` are then the fixed effects, ``pvalue`` the Wald t-test p-value
        of the slope with ``dof = n_subjects - 2``, and ``rvalue`` the Pearson correlation of ``x``
        and ``y`` across all observations (a descriptive effect size). If the mixed model can not
        be fitted or has no between-subject variance, the OLS results are returned with a warning.
        """
        n_subjects = 0 if subjects is None else np.unique(subjects).size
        if np.ptp(x) == 0 or n_subjects < 3 or n_subjects == np.size(subjects):
            return SleepStatsAgreement._linregress(x, y)
        from statsmodels.regression.mixed_linear_model import MixedLM

        exog = np.column_stack([np.ones_like(x, dtype=float), x])
        try:
            with warnings.catch_warnings():
                # Convergence and singular-matrix warnings are common when the between-subject
                # variance is close to zero (the model then reduces to OLS) and are not
                # informative here. The result attributes are lazy, so read them in this block.
                warnings.simplefilter("ignore")
                res = MixedLM(np.asarray(y, dtype=float), exog, groups=np.asarray(subjects))
                res = res.fit(reml=True)
                intercept, slope = np.asarray(res.fe_params, dtype=float)
                intercept_stderr, stderr = np.asarray(res.bse_fe, dtype=float)
        except (np.linalg.LinAlgError, ValueError) as e:  # pragma: no cover
            logger.warning(f"Mixed model failed ({e}); using ordinary least squares instead.")
            return SleepStatsAgreement._linregress(x, y)
        # t-test instead of the z-test of statsmodels, consistent with the t(n_subjects - 2) CIs
        with np.errstate(divide="ignore", invalid="ignore"):
            pvalue = 2 * sps.t.sf(np.abs(slope / stderr), n_subjects - 2)
        if not np.isfinite([slope, intercept, stderr, intercept_stderr, pvalue]).all():
            # Boundary solution (zero between-subject variance): the mixed model is equivalent to
            # OLS, whose standard errors are well defined
            logger.warning(
                "Mixed model at the boundary (no between-subject variance); using ordinary least "
                "squares instead, with N - 2 degrees of freedom."
            )
            return SleepStatsAgreement._linregress(x, y)
        return {
            "slope": slope,
            "intercept": intercept,
            "rvalue": np.corrcoef(x, y)[0, 1],
            "pvalue": pvalue,
            "stderr": stderr,
            "intercept_stderr": intercept_stderr,
            "dof": n_subjects - 2,
            "method": "mixed",
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

        With repeated measures, subjects are resampled (see :py:meth:`_cluster_bootstrap`), and
        the regression coefficients are not bootstrapped: their "boot" CIs are the Wald CIs of the
        mixed model (see :py:meth:`summary`).

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

        agreement = self._agreement

        def get_vars(ref_arr, diff_arr, rabs_arr, clusters=None, bias_line=None):
            """A function to get all variables at once and avoid redundant stats.bootstrap calls.

            With repeated measures, ``clusters`` is the subject of each row of the resample, and
            ``bias_line`` the ``(intercept, slope)`` of the full-sample mixed-model bias line, so
            that the half-width is the spread of the resampled nights around the reported line.
            Only the bias, LoA and half-width are then returned. Otherwise the bias line is the
            OLS fit of the resample.
            """
            bias_mean = np.mean(diff_arr)
            sd = self._total_sd(diff_arr, clusters)
            loa_lower, loa_upper = bias_mean - agreement * sd, bias_mean + agreement * sd
            if bias_line is None:
                # A resample can have constant reference values, see _linregress
                regr = self._linregress(ref_arr, diff_arr)
                bias_line = regr["intercept"], regr["slope"]
            bias_inter, bias_slope = bias_line
            resid = diff_arr - (bias_inter + bias_slope * ref_arr)
            loa_halfwidth = agreement * self._total_sd(resid, clusters)
            if self._repeated:
                return bias_mean, loa_lower, loa_upper, loa_halfwidth
            # Note this is NOT recalculating residuals each time for the next regression
            regr = self._linregress(ref_arr, rabs_arr)
            loa_slope, loa_inter = regr["slope"], regr["intercept"]
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
        regr_variables = ["bias_intercept", "bias_slope", "loa_intercept", "loa_slope"]
        if self._repeated:
            variable_order = [v for v in variable_order if v not in regr_variables]

        def _bootstrap(df, statistic, columns):
            """Bootstrap ``statistic`` over the sessions (or subjects) of one sleep statistic."""
            arrays = tuple(df[columns].to_numpy().T)
            if not self._repeated:
                return sps.bootstrap(arrays, statistic, **bs_kwargs)
            return self._cluster_bootstrap(
                df["subject_id"].to_numpy(), arrays, statistic, **bs_kwargs
            )

        def _bootstrap_vars(df):
            """Bootstrap of the variables of one sleep statistic (``df.name``)."""
            if self._repeated:
                line = tuple(self._vals.loc[df.name, ["bias_intercept", "bias_slope"]])
                return _bootstrap(df, functools.partial(get_vars, bias_line=line), column_order)
            return _bootstrap(df, get_vars, column_order)

        boot_ci = (
            self._data.loc[sleep_stats]  # Extract the relevant sleep stats
            .groupby("sleep_stat")  # Group so the bootstrapping is applied once to each sleep stat
            # Apply the bootstrap function to the 3 columns, passed as a tuple of 3 1D arrays
            .apply(_bootstrap_vars)
            .map(lambda res: res.confidence_interval)  # Pull high/low CIs out of the results object
            .explode()  # Break high and low CIs into separate rows
            .to_frame("value")  # Convert to dataframe and name column
            .assign(interval=interval_order * len(sleep_stats))  # Add a column indicating interval
            .explode("value")  # Break low CI variables and high CI variables out of arrays
            .assign(variable=variable_order * len(sleep_stats) * 2)  # Add column indicating variabl
            .pivot(columns=["variable", "interval"], values="value")  # Go long to wide format
            .sort_index(axis=1)  # Sort MultiIndex columns for cleanliness
        )
        if self._repeated:
            # Regression coefficients are not bootstrapped: use the Wald CIs
            wald_ci = self._ci.loc[sleep_stats, "param"][regr_variables]
            boot_ci = pd.concat([boot_ci, wald_ci], axis=1)
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
            # Resample function: apply _euser_slope_scalar to each bootstrap replicate's SD.
            def _euser_resample(d, clusters=None):
                return self._euser_slope_scalar(self._total_sd(d, clusters), agreement)

            for stat in log_stats_to_boot:
                df = self._data.loc[[stat]]
                result = _bootstrap(df, _euser_resample, ["log_difference"])
                self._loa_log_ci.at[stat, "boot_lower"] = result.confidence_interval.low
                self._loa_log_ci.at[stat, "boot_upper"] = result.confidence_interval.high

    @staticmethod
    def _cluster_bootstrap(subjects, arrays, statistic, **bs_kwargs):
        """Cluster (subject-level) bootstrap with :py:func:`scipy.stats.bootstrap`.

        Each replicate draws subjects with replacement and concatenates all their rows, so that
        the within-subject correlation of the nights is preserved. A subject drawn several times
        counts as that many distinct clusters, and the BCa jackknife leaves one subject out at a
        time. ``statistic`` is called as ``statistic(*resampled_arrays, clusters=...)``, where
        ``clusters`` labels the cluster of each resampled row.
        """
        _, inverse = np.unique(np.asarray(subjects), return_inverse=True)
        rows_by_subject = [np.flatnonzero(inverse == k) for k in range(inverse.max() + 1)]
        sizes = np.array([rows.size for rows in rows_by_subject])

        def _statistic(subject_idx):
            subject_idx = np.asarray(subject_idx, dtype=int)
            rows = np.concatenate([rows_by_subject[k] for k in subject_idx])
            clusters = np.repeat(np.arange(subject_idx.size), sizes[subject_idx])
            return statistic(*(arr[rows] for arr in arrays), clusters=clusters)

        return sps.bootstrap((np.arange(sizes.size),), _statistic, **bs_kwargs)

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
        Menghini et al. (2021).

        Each row corresponds to one sleep statistic, labelled with its unit (e.g.
        ``"TST (min)"``). Reference and observed scorer means (SD) are shown first, followed by
        bias and LoA, optionally merged with their confidence intervals (e.g.
        ``"2.34 [1.10, 3.58]"``). An ``"Assumptions"`` column shows whether each statistical
        assumption was met (``"✓"``) or not met (``"✗"``), which drives the automatic
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
            * ``'regr'`` — regression LoA: ``bias ± 2.46 (c0 + c1 × ref)``, where ``c0 + c1 ×
              ref`` models the absolute residuals of the bias. Always uses this form regardless
              of assumptions or ``log_transform``.
            * ``'log'`` — Euser LoA: ``bias ± slope × ref``. Requires ``log_transform=True`` and
              no zero values; raises ``ValueError`` otherwise.
            * ``'auto'`` (default) — uses ``'log'`` for the log-transformed statistics (see
              ``log_transform``). Otherwise, uses ``'param'`` when the homoscedasticity assumption
              passes and ``'regr'`` when it fails.
        ci_method : str or None
            If ``'param'``, parametric t-distribution CIs are used. If ``'boot'``, bootstrap CIs
            are used (BCa by default, see ``bootstrap_kwargs``). If ``'auto'`` (default), the
            method is chosen per statistic based on the normality assumption test. If ``None``, no
            confidence intervals are computed or shown (the columns are then named ``"Bias"`` and
            ``"LoA"``). See :py:meth:`summary` for the case of several nights per subject.
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
          ``1.96 × SD`` of the bias-regression residuals (Menghini et al. 2021, eq. 2)
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

            With several nights per subject (``subjects``), the bootstrap resamples subjects
            instead of nights. The regression coefficients (``bias_intercept``, ``bias_slope``,
            ``loa_intercept``, ``loa_slope``) always use the Wald CI of the mixed model with a
            ``t(n_subjects - 2)`` quantile, even with ``ci_method="boot"``.
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
            If ``'param'``, sleep statistics are always adjusted based on parametric bias, i.e.
            the mean difference is subtracted from ``data``.
            If ``'auto'`` (default), the parametric bias is used for all sleep statistics with a
            constant bias, and a :py:class:`NotImplementedError` is raised if ``data`` contains a
            sleep statistic with a proportional bias (see
            :py:attr:`~yasa.SleepStatsAgreement.assumptions`).
            ``'regr'`` (regression-modeled bias) is not implemented yet and raises a
            :py:class:`NotImplementedError`.

            .. seealso:: :py:meth:`~yasa.SleepStatsAgreement.summary`

        Returns
        -------
        calibrated_data : :py:class:`pandas.DataFrame`
            A :py:class:`~pandas.DataFrame` with calibrated sleep statistics: ``x - bias_mean``.

        Raises
        ------
        NotImplementedError
            If ``bias_method='regr'``, or if ``bias_method='auto'`` and ``data`` contains a sleep
            statistic with a proportional bias. To calibrate only the statistics with a constant
            bias, subset the columns of ``data``, or use ``bias_method='param'`` to subtract the
            mean difference from all statistics.
        """
        assert isinstance(data, pd.DataFrame), "`data` must be a pandas DataFrame"
        assert all(col in self.sleep_statistics for col in data), (
            f"all columns of `data` must be valid sleep statistics: {self.sleep_statistics}"
        )
        assert isinstance(bias_method, str), "`bias_method` must be a string"
        assert bias_method in self._bias_method_opts, (
            f"`bias_method` must be one of {self._bias_method_opts}"
        )
        # TODO: implement the calibration of sleep statistics with a proportional bias, i.e. with
        # a bias modeled as `obs - ref = b0 + b1 * ref`. Three candidate methods:
        #   1. Invert the bias model: `(x - b0) / (1 + b1)`. Exact without random error, but the
        #      random error of the device is multiplied by 1 / (1 + b1), which explodes when the
        #      slope is close to -1. This was the previous implementation of `bias_method="regr"`.
        #   2. Use the device value as the size of the measurement, as suggested by Menghini et
        #      al. (2021): `x - (b0 + b1 * x)`. Less extreme, but it also amplifies the error.
        #   3. Regress the reference on the device, `ref = a0 + a1 * obs`, and predict the
        #      reference from new device values. This minimizes the prediction error, but shrinks
        #      the values toward the mean of the calibration sample, so it only transfers to
        #      similar populations, and it is no longer derived from the Bland-Altman bias.
        # On the 14 nights of the SRI sample dataset (leave-one-night-out), only method 3 reduced
        # the error of the device for deep and REM sleep, and method 1 was worse than no
        # calibration for all statistics with a proportional bias.
        if bias_method == "regr":
            raise NotImplementedError(
                "Calibration based on a regression-modeled (proportional) bias is not implemented "
                "yet. Use `bias_method='param'` to subtract the mean difference instead."
            )
        if bias_method == "auto":
            is_regr = self._assumptions.loc[data.columns, ("constant_bias", "method")].eq("regr")
            if is_regr.any():
                raise NotImplementedError(
                    "Calibration of sleep statistics with a proportional bias is not implemented "
                    f"yet: {is_regr.index[is_regr].tolist()}. Remove these columns from `data`, or "
                    "use `bias_method='param'` to subtract the mean difference instead."
                )
        # Restrict to the columns of `data`, otherwise the subtraction would align on all sleep
        # statistics and add columns of NaN for those not in `data`
        return data - self._vals.loc[data.columns, "bias_mean"]

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
            * ``'regr'`` — regression LoA: lines following ``bias ± 2.46 (c0 + c1 × ref)``,
              where ``c0 + c1 × ref`` models the absolute residuals of the bias. Always uses this
              form regardless of assumptions or ``log_transform``.
            * ``'log'`` — Euser LoA: lines following ``bias ± slope × ref``. Requires
              ``log_transform=True`` and no zero values; raises ``ValueError`` otherwise.
            * ``'auto'`` (default) — uses ``'log'`` for the log-transformed statistics (see
              ``log_transform``). Otherwise, uses ``'param'`` when the homoscedasticity assumption
              passes and ``'regr'`` when it fails.
        ci_method : str or None
            If ``'param'``, parametric CIs are drawn. If ``'boot'``, bootstrap CIs are drawn. If
            ``'auto'`` (default), chosen per statistic based on the normality assumption test.
            If ``None``, no confidence intervals are drawn.
        scatter_kwargs : dict or None
            Keyword arguments passed to :py:func:`matplotlib.pyplot.scatter`.
        **kwargs : dict
            Other keyword arguments are passed through to :py:class:`seaborn.FacetGrid`. With
            several nights per subject, use ``hue="subject_id"`` to give the nights of each subject
            the same color (a legend is added).

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

        # Select scatterplot arguments and update with optional input. With a `hue` (e.g. one color
        # per subject), the default colors are dropped so that the colors of the hue apply.
        default_scatter_kwargs = dict(s=12, facecolor="none", edgecolor="black", alpha=0.8)
        if "hue" in kwargs:
            default_scatter_kwargs = dict(s=12, alpha=0.8)
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
        if "hue" in kwargs:
            g.add_legend()
        if hasattr(g.fig, "align_titles"):  # introduced in matplotlib v3.9.0
            g.fig.align_titles()
        g.fig.align_labels()
        g.tight_layout(w_pad=1, h_pad=2)
        return g
