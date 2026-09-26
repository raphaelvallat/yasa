.. _tutorial_evaluation:

Evaluating a Sleep Tracker
##########################

.. currentmodule:: yasa

How good is a wearable, or a sleep staging algorithm, at scoring sleep? This tutorial answers
that question by comparing a device against polysomnography (PSG), following the standardized
framework of `Menghini et al. (2021) <https://doi.org/10.1093/sleep/zsaa170>`_. We use the same
data and the same steps as the authors'
`R analytical pipeline <https://sri-human-sleep.github.io/sleep-trackers-performance/AnalyticalPipeline_v1.0.0.html>`_,
in a few lines of Python.

The evaluation has two parts, each handled by one class:

1. **Epoch-by-epoch agreement** (:py:class:`EpochByEpochAgreement`): does the device assign the
   right stage to each 30-second epoch?
2. **Sleep statistics agreement** (:py:class:`SleepStatsAgreement`): does the device get the
   night-level summaries right, such as total sleep time or minutes of deep sleep?

.. warning::

    The evaluation module is **experimental**. Its API may change before the full release planned
    for v0.8.0.

.. contents:: Contents
    :local:
    :depth: 2

--------

The data
--------

The sample dataset of the R pipeline contains 14 nights (10,766 epochs of 30 seconds), each
scored simultaneously by PSG and by a consumer device. Stages are coded as integers:
0 = Wake, 1 = Light (N1 + N2), 2 = Deep (N3), 3 = REM.

.. code-block:: python

    >>> import pandas as pd
    >>> import yasa
    >>> url = "https://github.com/raphaelvallat/yasa/raw/master/tests/data/sample_data_sri.csv.xz"
    >>> df = pd.read_csv(url)
    >>> df.head(3)
      subject  epoch  reference  device
    0   sbj01      1          0       0
    1   sbj01      2          0       0
    2   sbj01      3          0       0

We turn each night into a pair of 4-stage :py:class:`Hypnogram` objects, one per scorer. The
``scorer`` name is how YASA tells the reference from the device in all outputs:

.. code-block:: python

    >>> mapping = {0: "WAKE", 1: "LIGHT", 2: "DEEP", 3: "REM"}
    >>> ref_hyps, obs_hyps = {}, {}
    >>> for sub, d in df.groupby("subject"):
    ...     ref_hyps[sub] = yasa.Hypnogram.from_integers(
    ...         d["reference"], mapping=mapping, n_stages=4, scorer="PSG"
    ...     )
    ...     obs_hyps[sub] = yasa.Hypnogram.from_integers(
    ...         d["device"], mapping=mapping, n_stages=4, scorer="Device"
    ...     )
    >>> ref_hyps["sbj01"]
    <Hypnogram | 882 epochs x 30s (441.00 minutes), 4 unique stages, scored by PSG>

Passing dictionaries makes the keys (here, the subject IDs) the session IDs used throughout.

.. note::

    To compare just two hypnograms, skip the dictionaries and use
    ``ref_hyp.evaluate(obs_hyp)`` (see :py:meth:`Hypnogram.evaluate`).

--------

Part 1: Epoch-by-epoch agreement
--------------------------------

.. code-block:: python

    >>> ebe = yasa.EpochByEpochAgreement(ref_hyps, obs_hyps)
    >>> ebe
    <EpochByEpochAgreement | Observed hypnograms scored by Device evaluated against reference
    hypnograms scored by PSG, 14 sleep sessions>

Start by looking at one night. The device tracks the overall architecture, but misses a few
periods of deep sleep, scoring them as light sleep instead:

.. code-block:: python

    >>> ebe.plot_hypnograms(sleep_id="sbj03")

.. plot::
    :context: reset
    :include-source: False

    import matplotlib.pyplot as plt
    import pandas as pd
    import yasa

    df = pd.read_csv("../../tests/data/sample_data_sri.csv.xz")
    mapping = {0: "WAKE", 1: "LIGHT", 2: "DEEP", 3: "REM"}
    ref_hyps, obs_hyps = {}, {}
    for sub, d in df.groupby("subject"):
        ref_hyps[sub] = yasa.Hypnogram.from_integers(
            d["reference"], mapping=mapping, n_stages=4, scorer="PSG"
        )
        obs_hyps[sub] = yasa.Hypnogram.from_integers(
            d["device"], mapping=mapping, n_stages=4, scorer="Device"
        )
    ebe = yasa.EpochByEpochAgreement(ref_hyps, obs_hyps)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ebe.plot_hypnograms(sleep_id="sbj03", ax=ax)
    fig.tight_layout()

Error matrix
~~~~~~~~~~~~

The confusion (or error) matrix shows where the device goes wrong. Rows are PSG stages, columns
are device stages. Summing the epochs of all nights gives:

.. code-block:: python

    >>> ebe.get_confusion_matrix(agg_func="sum")
    Device  WAKE  LIGHT  DEEP  REM
    PSG
    WAKE     871    483    29   71
    LIGHT    303   4381   398  521
    DEEP      34   1142   925   16
    REM       57    564    49  922

Raw counts are dominated by the most common stage and by the longest nights. Menghini et al.
recommend the **proportional** error matrix instead: each night is normalized by row, then
averaged across nights, so that every night weighs the same. The confidence interval comes from
a bootstrap across nights:

.. code-block:: python

    >>> cm = ebe.get_confusion_matrix_proportional(bootstrap_kwargs={"rng": 42})
    >>> cm.round(1).head(4)
                 mean   std  ci_lower  ci_upper  n_sessions
    PSG  Device
    WAKE WAKE    61.9  16.2      52.9      69.2          14
         LIGHT   30.6  17.6      23.0      40.5          14
         DEEP     1.9   2.5       0.9       3.6          14
         REM      5.5  10.4       2.1      15.1          14

Use ``formatted=True`` to get a publication-ready ``mean (SD) [CI]`` table, or plot the means as
a heatmap:

.. code-block:: python

    >>> import seaborn as sns
    >>> stages = ["WAKE", "LIGHT", "DEEP", "REM"]
    >>> mat = cm["mean"].unstack().loc[stages, stages]
    >>> sns.heatmap(mat, annot=True, fmt=".1f", cmap="Blues", vmin=0, vmax=100, square=True)

.. plot::
    :context: close-figs
    :include-source: False

    import seaborn as sns

    cm = ebe.get_confusion_matrix_proportional(ci_method=None)
    stages = ["WAKE", "LIGHT", "DEEP", "REM"]
    mat = cm["mean"].unstack().loc[stages, stages]
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(
        mat, annot=True, fmt=".1f", cmap="Blues", vmin=0, vmax=100, square=True,
        cbar_kws={"label": "% of PSG epochs"}, ax=ax,
    )
    fig.tight_layout()

The diagonal is the percentage of each PSG stage correctly detected. The story is clear: the
device recognizes light sleep well (79%), but classifies more than half of deep sleep, and about
30% of wake and REM, as light sleep.

Overall agreement
~~~~~~~~~~~~~~~~~

:py:meth:`~EpochByEpochAgreement.get_agreement` returns one row of agreement scores per night,
and :py:meth:`~EpochByEpochAgreement.summary` aggregates them across nights. Add
``ci_method="boot"`` for a bootstrap confidence interval of the mean:

.. code-block:: python

    >>> ebe.get_agreement().round(2).head(3)
              accuracy  balanced_acc  kappa   mcc  precision     f1
    sleep_id
    sbj01        61.34         51.86   0.31  0.32      59.21  57.09
    sbj02        59.06         50.52   0.31  0.37      62.22  52.08
    sbj03        76.16         76.53   0.62  0.63      77.25  76.37

    >>> ebe.summary(
    ...     ci_method="boot", bootstrap_kwargs={"method": "percentile", "rng": 42}
    ... )[["mean", "std", "ci_lower", "ci_upper"]].round(2)
                   mean   std  ci_lower  ci_upper
    metric
    accuracy      66.18  6.60     62.83     69.51
    balanced_acc  63.18  8.73     58.87     67.53
    kappa          0.45  0.12      0.39      0.51
    mcc            0.47  0.12      0.42      0.53
    precision     69.53  8.28     65.26     73.54
    f1            64.53  8.19     60.37     68.64

On average, the device agrees with PSG on two out of three epochs (Cohen's kappa = 0.45).

.. tip::

    The default bootstrap method is BCa, which is more accurate with larger samples. With only 14
    nights, YASA warns that BCa may be unstable, so we use plain percentiles here.

Agreement by stage
~~~~~~~~~~~~~~~~~~

Overall scores hide stage-specific errors. :py:meth:`~EpochByEpochAgreement.get_agreement_bystage`
computes one-vs-rest metrics for each stage and each night, and
``summary(by_stage=True)`` averages them. YASA uses the scikit-learn names, which map to the
Menghini et al. metrics as follows:

.. list-table::
    :header-rows: 1
    :widths: 15 30 55

    * - YASA
      - Menghini et al.
      - Question answered
    * - ``recall``
      - Sensitivity
      - % of PSG epochs of this stage that the device detects
    * - ``specificity``
      - Specificity
      - % of PSG epochs of other stages that the device correctly rejects
    * - ``precision``
      - Positive predictive value (PPV)
      - When the device says this stage, how often is it right?
    * - ``npv``
      - Negative predictive value (NPV)
      - When the device says another stage, how often is it right?

.. code-block:: python

    >>> summ = ebe.summary(by_stage=True)
    >>> summ["mean"].unstack("stage").round(1)
    stage         DEEP  LIGHT    REM   WAKE
    metric
    fbeta         50.2   71.9   59.6   62.7
    npv           87.2   73.0   93.4   93.7
    precision     66.1   67.7   62.2   68.1
    recall        44.2   79.3   67.3   61.9
    specificity   94.9   58.4   93.6   95.9
    support      151.2  400.2  113.7  103.9

The recall row is the diagonal of the proportional error matrix. The low specificity of light
sleep (58%) is the flip side of the previous finding: the device over-scores light sleep, at the
expense of the other stages. ``support`` is the average number of PSG epochs of each stage.

.. note::

    If a stage is absent from the PSG of a night (e.g. no deep sleep), its recall is undefined for
    that night. YASA leaves it as ``NaN`` so that it does not pull the group mean down. Use
    ``get_agreement_bystage(zero_division=0)`` to count it as 0% instead.

Sleep vs wake
~~~~~~~~~~~~~

Many devices are first judged on sleep/wake detection. Collapsing Light, Deep and REM into a
single ``SLEEP`` stage only requires a different mapping and ``n_stages=2``:

.. code-block:: python

    >>> sw = {0: "WAKE", 1: "SLEEP", 2: "SLEEP", 3: "SLEEP"}
    >>> ref_sw, obs_sw = {}, {}
    >>> for sub, d in df.groupby("subject"):
    ...     ref_sw[sub] = yasa.Hypnogram.from_integers(
    ...         d["reference"], mapping=sw, n_stages=2, scorer="PSG"
    ...     )
    ...     obs_sw[sub] = yasa.Hypnogram.from_integers(
    ...         d["device"], mapping=sw, n_stages=2, scorer="Device"
    ...     )
    >>> ebe_sw = yasa.EpochByEpochAgreement(ref_sw, obs_sw)
    >>> ebe_sw.summary().loc[["accuracy", "kappa"], ["mean", "std"]].round(2)
               mean   std
    metric
    accuracy  90.82  3.90
    kappa      0.58  0.14

    >>> sleep = ebe_sw.summary(by_stage=True).loc["SLEEP"]
    >>> sleep.loc[["recall", "specificity"], ["mean", "std"]].round(1)
                 mean   std
    metric
    recall       95.9   2.3
    specificity  61.9  16.2

Accuracy jumps to 91%, but that is mostly because sleep makes up most of the night. The device
detects 96% of sleep epochs but only 62% of wake epochs (the specificity of ``SLEEP``). This is
the classic profile of wearables, which tend to mistake quiet wakefulness for sleep.

--------

Part 2: Sleep statistics agreement
----------------------------------

Next, we compare night-level sleep statistics. :py:meth:`~EpochByEpochAgreement.get_sleep_stats`
computes them for every night and both scorers:

.. code-block:: python

    >>> sstats = ebe.get_sleep_stats()
    >>> sstats.loc["Device", ["TST", "SE", "SOL", "WASO", "LIGHT", "DEEP", "REM"]].head(3).round(1)
                TST    SE   SOL  WASO  LIGHT  DEEP   REM
    sleep_id
    sbj01     378.0  85.7  22.0  41.0  315.0  39.5  23.5
    sbj02     354.5  89.9   7.5  32.5  312.0  26.5  16.0
    sbj03     262.5  78.7   8.5  62.5  170.5  38.5  53.5

Durations are in minutes. :py:class:`SleepStatsAgreement` then takes this table as is and runs a
Bland-Altman analysis on every statistic (statistics identical between scorers, such as the time
in bed, are dropped):

.. code-block:: python

    >>> ssa = yasa.SleepStatsAgreement(sstats, bootstrap_kwargs={"rng": 42})
    >>> ssa
    <SleepStatsAgreement | Observed scorer ('Device') evaluated against reference scorer ('PSG'),
    14 sleep sessions>

.. note::

    YASA computes WASO within the sleep period, i.e. from the first to the last sleep epoch. The R
    pipeline also counts wake after the final awakening, so WASO differs for the nights that end
    with a period of wake.

The report table
~~~~~~~~~~~~~~~~

:py:meth:`~SleepStatsAgreement.report` returns the reporting table recommended by Menghini et al.:
the mean (SD) of each scorer, the bias (device − PSG) and the 95% limits of agreement (LoA), each
with their 95% confidence interval:

.. code-block:: python

    >>> stats = ["TST", "SE", "SOL", "WASO", "LIGHT", "DEEP", "REM"]
    >>> report = ssa.report(sleep_stats=stats)
    >>> report[["PSG mean (SD)", "Device mean (SD)", "Bias [95% CI]"]]
                  PSG mean (SD) Device mean (SD)                                         Bias [95% CI]
    sleep_stat
    TST (min)    332.57 (67.25)   339.32 (60.61)                                   6.75 [-6.19, 19.69]
    SE (%)         86.08 (6.90)     88.07 (4.53)   58.10 + -0.65x [b0: 14.71, 85.02; b1: -0.96, -0.17]
    SOL (min)     13.75 (11.60)    11.36 (14.81)                                  -2.39 [-10.93, 5.42]
    WASO (min)    36.18 (21.19)    33.29 (12.57)   25.55 + -0.79x [b0: 10.20, 40.28; b1: -1.32, -0.24]
    LIGHT (min)  200.11 (48.52)   234.64 (56.85)                                   34.54 [4.01, 65.06]
    DEEP (min)    75.61 (23.71)    50.04 (22.58)  23.70 + -0.65x [b0: -20.32, 67.72; b1: -1.21, -0.09]
    REM (min)     56.86 (23.11)    54.64 (27.39)  56.14 + -1.03x [b0: 10.63, 101.66; b1: -1.77, -0.28]

    >>> report[["LoA [95% CI]", "Assumptions"]]
                                                                LoA [95% CI]                                 Assumptions
    sleep_stat
    TST (min)                 -37.16 to 50.66 [-59.57, -14.76; 28.26, 73.07]  ✓ normal  ✓ constant bias  ✓ homoscedastic
    SE (%)       ±2.46 (24.90 + -0.26x) [c0: 15.19, 39.63; c1: -0.40, -0.15]  ✗ normal  ✗ constant bias  ✗ homoscedastic
    SOL (min)                 -34.69 to 29.90 [-54.47, -19.55; 16.45, 54.06]  ✗ normal  ✓ constant bias  ✓ homoscedastic
    WASO (min)                                   bias ± 22.98 [15.74, 36.30]  ✗ normal  ✗ constant bias  ✓ homoscedastic
    LIGHT (min)            -69.08 to 138.15 [-121.94, -16.21; 85.28, 191.01]  ✓ normal  ✓ constant bias  ✓ homoscedastic
    DEEP (min)                                   bias ± 41.18 [24.23, 58.14]  ✓ normal  ✗ constant bias  ✓ homoscedastic
    REM (min)                                    bias ± 53.68 [31.57, 75.78]  ✓ normal  ✗ constant bias  ✓ homoscedastic

A few highlights: total sleep time is well estimated (+7 min on average), but light sleep is
overestimated by 35 min. Individual errors are large, though: for a given night, the device's TST
can be anywhere from 37 min below to 51 min above PSG.

When the differences are not normal (here SE, SOL and WASO), the confidence intervals come from
a bootstrap. Passing a seed with ``bootstrap_kwargs={"rng": 42}`` makes them reproducible.

Assumptions drive the method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Why do some rows show a single number and others an equation? As in the R pipeline, each statistic
is first tested for three assumptions, and the method is adapted when one is violated:

==================  ========================================  =====================================
Assumption          Test                                      If violated
==================  ========================================  =====================================
``normal``          Shapiro-Wilk on the differences           Bootstrap CIs instead of parametric
``constant_bias``   Slope of the differences vs. PSG          Bias is a regression line
                                                              ``b0 + b1 × PSG``
``homoscedastic``   Slope of the absolute residuals vs. PSG   LoA widen or narrow with the PSG value
                                                              ``±2.46 (c0 + c1 × PSG)``
==================  ========================================  =====================================

For deep sleep, for example, the bias depends on how much deep sleep the night actually had: with
a slope of −0.65, the device underestimates deep sleep more on nights with more deep sleep. The
full test results are in :py:attr:`~SleepStatsAgreement.assumptions`:

.. code-block:: python

    >>> ssa.assumptions["constant_bias"].loc[stats].round(3)
    metric      slope  pvalue     r2  passed method
    sleep_stat
    TST        -0.149   0.108  0.201    True  param
    SE         -0.652   0.002  0.579   False   regr
    SOL        -0.694   0.076  0.239    True  param
    WASO       -0.786   0.000  0.669   False   regr
    LIGHT      -0.407   0.188  0.140    True  param
    DEEP       -0.652   0.026  0.351   False   regr
    REM        -1.026   0.011  0.429   False   regr

.. tip::

    With many nights, statistical tests flag even trivial deviations. YASA therefore requires the
    effect size to be material too (e.g. :math:`R^2 > 0.1` for a proportional bias). The thresholds
    can be changed with the ``effect_size_gates`` argument. You can also override the automatic
    choice with ``bias_method``, ``loa_method`` and ``ci_method``.

Bland-Altman plots
~~~~~~~~~~~~~~~~~~

Finally, :py:meth:`~SleepStatsAgreement.plot_blandaltman` visualizes all of the above: the
differences against the PSG values, the bias (solid line) and the LoA (dashed lines), with their
confidence bands. The methods are selected automatically, just as in the report:

.. code-block:: python

    >>> ssa.plot_blandaltman(sleep_stats=["TST", "WASO", "DEEP", "REM"])

.. plot::
    :context: close-figs
    :include-source: False

    ssa = yasa.SleepStatsAgreement(ebe.get_sleep_stats(), bootstrap_kwargs={"rng": 42})
    ssa.plot_blandaltman(sleep_stats=["TST", "WASO", "DEEP", "REM"])

The downward slopes of WASO, DEEP and REM are clear: the device compresses these statistics
toward the group average, overestimating them on nights with little and underestimating them on
nights with a lot.

--------

Going further
-------------

* **Pooled metrics**: ``ebe.get_agreement(pooled=True)`` computes the metrics on all epochs
  together, instead of averaging per night (``"sum"`` vs ``"avg"`` in the R pipeline).
* **Log transformation**: for statistics whose error grows with their magnitude, use
  ``SleepStatsAgreement(sstats, log_transform=True)`` to get proportional LoA (Euser et al.,
  2008).
* **Calibration**: :py:meth:`SleepStatsAgreement.calibrate` corrects the systematic bias of the
  device in new data.
* **Numeric output**: :py:meth:`SleepStatsAgreement.summary` returns all biases, LoA and CIs as
  numbers instead of formatted strings.

See the :ref:`API reference <api_ref>` for all options.

Reference
---------

Menghini, L., Cellini, N., Goldstone, A., Baker, F. C., & de Zambotti, M. (2021). A standardized
framework for testing the performance of sleep-tracking technology: step-by-step guidelines and
open-source code. *SLEEP*, 44(2), zsaa170. https://doi.org/10.1093/sleep/zsaa170
