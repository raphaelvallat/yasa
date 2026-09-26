"""Reproduce all outputs and figures of the "Evaluating a Sleep Tracker" tutorial (evaluation.rst).

Usage (from the repository root, requires scipy >= 1.15 for the seeded bootstrap):

    python docs/tutorials/evaluation.py [--local] [--outdir DIR]

By default, the SRI sample data is downloaded from GitHub, as in the tutorial. Use ``--local`` to
read ``tests/data/sample_data_sri.csv.xz`` instead. Figures are saved to ``--outdir`` (default:
current directory).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

import yasa  # noqa: E402

URL = "https://github.com/raphaelvallat/yasa/raw/master/tests/data/sample_data_sri.csv.xz"
LOCAL = Path(__file__).parents[2] / "tests" / "data" / "sample_data_sri.csv.xz"
STATS = ["TST", "SE", "SOL", "WASO", "LIGHT", "DEEP", "REM"]
STAGES = ["WAKE", "LIGHT", "DEEP", "REM"]
BOOT = {"method": "percentile", "rng": 42}
DPI = 180


def section(title):
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def show(label, obj):
    print(f"\n>>> {label}\n{obj}")


def build_hypnograms(df):
    mapping = {0: "WAKE", 1: "LIGHT", 2: "DEEP", 3: "REM"}
    ref_hyps, obs_hyps = {}, {}
    for sub, d in df.groupby("subject"):
        ref_hyps[sub] = yasa.Hypnogram.from_integers(
            d["reference"], mapping=mapping, n_stages=4, scorer="PSG"
        )
        obs_hyps[sub] = yasa.Hypnogram.from_integers(
            d["device"], mapping=mapping, n_stages=4, scorer="Device"
        )
    return ref_hyps, obs_hyps


def main(local, outdir):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 80)
    sns.set_context("notebook", font_scale=1.15)
    outdir.mkdir(parents=True, exist_ok=True)

    section("The data")
    df = pd.read_csv(LOCAL if local else URL)
    show("df.head(3)", df.head(3))
    ref_hyps, obs_hyps = build_hypnograms(df)
    show('ref_hyps["sbj01"]', ref_hyps["sbj01"])

    section("Part 1: Epoch-by-epoch agreement")
    ebe = yasa.EpochByEpochAgreement(ref_hyps, obs_hyps)
    show("ebe", ebe)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ebe.plot_hypnograms(sleep_id="sbj03", ax=ax)
    fig.tight_layout()
    fig.savefig(outdir / "evaluation_hypnograms.png", dpi=DPI)
    plt.close(fig)

    # Error matrix
    show('ebe.get_confusion_matrix(agg_func="sum")', ebe.get_confusion_matrix(agg_func="sum"))
    cm = ebe.get_confusion_matrix_proportional(bootstrap_kwargs=BOOT)
    show("cm.round(1).head(4)", cm.round(1).head(4))
    show(
        "ebe.get_confusion_matrix_proportional(formatted=True, ...)",
        ebe.get_confusion_matrix_proportional(formatted=True, bootstrap_kwargs=BOOT),
    )

    mat = cm["mean"].unstack().loc[STAGES, STAGES]
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(
        mat, annot=True, fmt=".1f", cmap="Blues", vmin=0, vmax=100, square=True,
        cbar_kws={"label": "% of PSG epochs"}, ax=ax,
    )  # fmt: skip
    fig.tight_layout()
    fig.savefig(outdir / "evaluation_error_matrix.png", dpi=DPI)
    plt.close(fig)

    # Overall agreement
    show("ebe.get_agreement().round(2).head(3)", ebe.get_agreement().round(2).head(3))
    summ = ebe.summary(ci_method="boot", bootstrap_kwargs=BOOT)
    show(
        'ebe.summary(ci_method="boot", ...)', summ[["mean", "std", "ci_lower", "ci_upper"]].round(2)
    )

    # Agreement by stage
    summ = ebe.summary(by_stage=True)
    metrics = ["recall", "specificity", "precision", "npv"]
    show(
        'summ["mean"].unstack("stage").loc[metrics, stages].round(1)',
        summ["mean"].unstack("stage").loc[metrics, STAGES].round(1),
    )

    # Sleep vs wake
    ref_sw = {k: h.consolidate_stages(2) for k, h in ref_hyps.items()}
    obs_sw = {k: h.consolidate_stages(2) for k, h in obs_hyps.items()}
    ebe_sw = yasa.EpochByEpochAgreement(ref_sw, obs_sw)
    show(
        'ebe_sw.summary().loc[["accuracy", "kappa"], ["mean", "std"]].round(2)',
        ebe_sw.summary().loc[["accuracy", "kappa"], ["mean", "std"]].round(2),
    )
    always_sleep = 100 * (df["reference"] != 0).groupby(df["subject"]).mean().mean()
    show("Accuracy of a device that always scores sleep (%)", round(always_sleep, 1))
    sleep = ebe_sw.summary(by_stage=True).loc["SLEEP"]
    show(
        'sleep.loc[["recall", "specificity"], ["mean", "std"]].round(1)',
        sleep.loc[["recall", "specificity"], ["mean", "std"]].round(1),
    )

    section("Part 2: Sleep statistics agreement")
    sstats = ebe.get_sleep_stats()
    show(
        'sstats.loc["Device", STATS].head(3).round(1)', sstats.loc["Device", STATS].head(3).round(1)
    )
    ssa = yasa.SleepStatsAgreement(sstats, bootstrap_kwargs=BOOT)
    show("ssa", ssa)

    # Bland-Altman plots
    g = ssa.plot_blandaltman(sleep_stats=["TST", "WASO", "DEEP", "REM"], col_wrap=2)
    g.savefig(outdir / "evaluation_blandaltman.png", dpi=DPI)
    plt.close("all")

    # Assumptions
    unbiased_stats = ["TST", "LIGHT", "DEEP", "REM"]
    show(
        'ssa.assumptions["unbiased"].loc[["TST", "LIGHT", "DEEP", "REM"]].round(3)',
        ssa.assumptions["unbiased"].loc[unbiased_stats].round(3),
    )

    # Report table
    report = ssa.report(sleep_stats=STATS)
    show(
        'report[["PSG mean (SD)", "Device mean (SD)", "Bias [95% CI]"]]',
        report[["PSG mean (SD)", "Device mean (SD)", "Bias [95% CI]"]].to_string(),
    )
    show(
        'report[["LoA [95% CI]", "Assumptions"]]',
        report[["LoA [95% CI]", "Assumptions"]].to_string(),
    )

    print(f"\nFigures saved to {outdir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--local", action="store_true", help="read the data from tests/data")
    parser.add_argument("--outdir", type=Path, default=Path("."), help="folder for the figures")
    args = parser.parse_args()
    main(args.local, args.outdir)
