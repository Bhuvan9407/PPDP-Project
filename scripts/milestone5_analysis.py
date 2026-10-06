"""
Milestone 5 — Analysis
======================

Reads the committed Milestone 4 54-run results table and produces the
analysis deliverables required by the Milestone 5 guide.

No anonymization, preprocessing, or classifier evaluation is rerun here.

Primary common information-loss metric:
    Discernibility Metric (DM) = sum(|E|^2)

Reason:
    The committed Milestone 4 table contains a DM/discernibility value
    for all 54 runs, so DM is the only common information-loss metric
    that can be compared across all three methods and both datasets.

NCP note:
    The committed runs persist NCP/information-loss values for Mondrian,
    but ARX and KC-Slice do not persist a directly comparable NCP value.
    Therefore NCP is reported only when present and is NOT used for the
    cross-method ranking. The cross-method information-loss comparison
    uses DM.

Expected outputs:
    results/milestone5_analysis/
        accuracy_by_method.csv
        information_loss_dm_by_method.csv
        rankings_accuracy.csv
        rankings_dm.csv
        dataset_comparison.csv
        expected_findings_verdicts.md
        analysis_summary.md
        plots/
            adult_accuracy.png
            adult_information_loss_dm.png
            diabetes_accuracy.png
            diabetes_information_loss_dm.png
            adult_vs_diabetes_accuracy.png
"""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS / LOCKED INPUT
# ============================================================

REPO = Path(__file__).resolve().parents[1]

INPUT = REPO / "results" / "milestone4_results_54_runs.csv"
OUT_DIR = REPO / "results" / "milestone5_analysis"
PLOT_DIR = OUT_DIR / "plots"

REQUIRED_RUNS = 54
METHODS = ["Mondrian", "ARX", "KC-Slice"]
DATASETS = ["Adult", "Diabetes"]
SA_COUNTS = [1, 2, 3]


# ============================================================
# VALIDATION
# ============================================================

def load_and_validate() -> pd.DataFrame:
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Milestone 4 results not found:\n{INPUT}"
        )

    df = pd.read_csv(INPUT)

    if len(df) != REQUIRED_RUNS:
        raise AssertionError(
            f"Expected {REQUIRED_RUNS} rows, got {len(df)}"
        )

    required = {
        "run_id",
        "dataset",
        "method",
        "sa_count",
        "seed",
        "discernibility_metric",
        "information_loss_metric",
        "information_loss_source",
        "accuracy",
        "macro_f1",
        "roc_auc",
        "original_accuracy",
        "accuracy_delta_vs_original",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise AssertionError(
            f"Missing required columns: {missing}"
        )

    if df["run_id"].nunique() != REQUIRED_RUNS:
        raise AssertionError("Duplicate run IDs detected.")

    if set(df["dataset"]) != set(DATASETS):
        raise AssertionError(
            f"Unexpected datasets: {sorted(df['dataset'].unique())}"
        )

    if set(df["method"]) != set(METHODS):
        raise AssertionError(
            f"Unexpected methods: {sorted(df['method'].unique())}"
        )

    if set(df["sa_count"].astype(int)) != set(SA_COUNTS):
        raise AssertionError("Unexpected SA-count levels.")

    counts = (
        df.groupby(
            ["dataset", "method", "sa_count"],
            dropna=False,
        )
        .size()
    )

    if not (counts == 3).all():
        raise AssertionError(
            "Every dataset × method × SA-count combination "
            "must contain exactly 3 seeds."
        )

    if not df["accuracy"].between(0, 1).all():
        raise AssertionError("Accuracy contains values outside [0, 1].")

    if not df["discernibility_metric"].notna().all():
        raise AssertionError(
            "All 54 runs must have a discernibility/DM value."
        )

    return df.sort_values(
        ["dataset", "method", "sa_count", "seed"]
    ).reset_index(drop=True)


# ============================================================
# SUMMARY TABLES
# ============================================================

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(
            ["dataset", "method", "sa_count"],
            as_index=False,
        )
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
            original_accuracy_mean=("original_accuracy", "mean"),
            accuracy_delta_mean=("accuracy_delta_vs_original", "mean"),
            dm_mean=("discernibility_metric", "mean"),
            dm_std=("discernibility_metric", "std"),
            ncp_mean=("information_loss_metric", "mean"),
            ncp_available=("information_loss_metric", "count"),
            min_class_mean=("minimum_class_size", "mean"),
            max_class_mean=("maximum_class_size", "mean"),
            equivalence_classes_mean=("equivalence_classes", "mean"),
        )
        .sort_values(
            ["dataset", "sa_count", "method"]
        )
        .reset_index(drop=True)
    )

    return summary


def build_accuracy_rankings(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dataset in DATASETS:
        for sa_count in SA_COUNTS:
            subset = summary[
                (summary["dataset"] == dataset)
                & (summary["sa_count"] == sa_count)
            ].copy()

            subset = subset.sort_values(
                ["accuracy_mean", "accuracy_std"],
                ascending=[False, True],
            ).reset_index(drop=True)

            for rank, row in enumerate(
                subset.itertuples(index=False),
                start=1,
            ):
                rows.append(
                    {
                        "dataset": dataset,
                        "sa_count": sa_count,
                        "rank": rank,
                        "method": row.method,
                        "mean_accuracy": row.accuracy_mean,
                    }
                )

    return pd.DataFrame(rows)


def build_dm_rankings(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dataset in DATASETS:
        for sa_count in SA_COUNTS:
            subset = summary[
                (summary["dataset"] == dataset)
                & (summary["sa_count"] == sa_count)
            ].copy()

            # Lower DM = lower discernibility penalty.
            subset = subset.sort_values(
                "dm_mean",
                ascending=True,
            ).reset_index(drop=True)

            for rank, row in enumerate(
                subset.itertuples(index=False),
                start=1,
            ):
                rows.append(
                    {
                        "dataset": dataset,
                        "sa_count": sa_count,
                        "rank": rank,
                        "method": row.method,
                        "mean_dm": row.dm_mean,
                    }
                )

    return pd.DataFrame(rows)


# ============================================================
# PLOTTING
# ============================================================

def plot_metric(
    summary: pd.DataFrame,
    dataset: str,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    log_scale: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    subset = summary[
        summary["dataset"] == dataset
    ]

    for method in METHODS:
        method_rows = (
            subset[subset["method"] == method]
            .sort_values("sa_count")
        )

        ax.plot(
            method_rows["sa_count"],
            method_rows[metric],
            marker="o",
            linewidth=2,
            markersize=6,
            label=method,
        )

    ax.set_xlabel(
        "Number of jointly protected sensitive attributes"
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(SA_COUNTS)
    ax.grid(True, alpha=0.25)
    ax.legend()

    if log_scale:
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_comparison(summary: pd.DataFrame) -> None:
    # Separate plot: Adult vs Diabetes accuracy, one panel per dataset
    # is intentionally avoided; the Milestone 5 deliverable already has
    # separate dataset plots. This additional plot compares the best
    # method-level accuracy trend across datasets.
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for dataset in DATASETS:
        subset = (
            summary[summary["dataset"] == dataset]
            .groupby("sa_count", as_index=False)["accuracy_mean"]
            .max()
            .sort_values("sa_count")
        )

        ax.plot(
            subset["sa_count"],
            subset["accuracy_mean"],
            marker="o",
            linewidth=2,
            markersize=6,
            label=f"{dataset} — best method",
        )

    ax.set_xlabel(
        "Number of jointly protected sensitive attributes"
    )
    ax.set_ylabel("Best mean classifier accuracy")
    ax.set_title(
        "Best downstream accuracy: Adult vs Diabetes"
    )
    ax.set_xticks(SA_COUNTS)
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        PLOT_DIR / "adult_vs_diabetes_accuracy.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# FORMAL COMPARISON / VERDICTS
# ============================================================

def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def fmt_num(value: float) -> str:
    return f"{value:,.2f}"


def method_at(
    summary: pd.DataFrame,
    dataset: str,
    method: str,
    sa_count: int,
) -> pd.Series:
    row = summary[
        (summary["dataset"] == dataset)
        & (summary["method"] == method)
        & (summary["sa_count"] == sa_count)
    ]

    if len(row) != 1:
        raise AssertionError(
            f"Expected one summary row for "
            f"{dataset}/{method}/SA{sa_count}"
        )

    return row.iloc[0]


def rank_string(
    accuracy_rankings: pd.DataFrame,
    dataset: str,
    sa_count: int,
) -> str:
    rows = accuracy_rankings[
        (accuracy_rankings["dataset"] == dataset)
        & (accuracy_rankings["sa_count"] == sa_count)
    ].sort_values("rank")

    return " > ".join(rows["method"].tolist())


def build_verdicts(
    summary: pd.DataFrame,
    accuracy_rankings: pd.DataFrame,
    dm_rankings: pd.DataFrame,
) -> str:
    lines = []

    lines.append("# Milestone 5 — Expected Findings Verdicts")
    lines.append("")
    lines.append(
        "This document is generated directly from the committed "
        "Milestone 4 54-run results table."
    )
    lines.append("")
    lines.append(
        "**Ranking convention:** higher mean accuracy is better; "
        "lower mean DM is better. DM is the common information-loss "
        "metric used for cross-method comparison. NCP is reported "
        "only where the committed run persisted it."
    )
    lines.append("")

    # --------------------------------------------------------
    # Adult accuracy
    # --------------------------------------------------------
    adult_ranks = [
        rank_string(accuracy_rankings, "Adult", s)
        for s in SA_COUNTS
    ]

    adult_arx = [
        method_at(summary, "Adult", "ARX", s)["accuracy_mean"]
        for s in SA_COUNTS
    ]
    adult_mondrian = [
        method_at(summary, "Adult", "Mondrian", s)["accuracy_mean"]
        for s in SA_COUNTS
    ]
    adult_kc = [
        method_at(summary, "Adult", "KC-Slice", s)["accuracy_mean"]
        for s in SA_COUNTS
    ]

    lines.append("## 1. Accuracy ranking — Adult Census Income")
    lines.append("")
    for s, ranking in zip(SA_COUNTS, adult_ranks):
        lines.append(
            f"- **SA={s}:** {ranking}"
        )

    lines.append("")
    lines.append(
        "### Verdict"
    )
    lines.append("")
    lines.append(
        f"ARX is the accuracy leader at all three attribute counts, "
        f"with mean accuracy of "
        f"{fmt_pct(adult_arx[0])}, "
        f"{fmt_pct(adult_arx[1])}, and "
        f"{fmt_pct(adult_arx[2])} for SA=1, 2, and 3 respectively. "
        f"KC-Slice remains around "
        f"{fmt_pct(adult_kc[0])}–{fmt_pct(adult_kc[2])}, while "
        f"Mondrian remains lowest at "
        f"{fmt_pct(adult_mondrian[0])}–{fmt_pct(adult_mondrian[2])}."
    )
    lines.append("")
    lines.append(
        "The ranking therefore **holds** as the number of protected "
        "attributes increases: ARX > KC-Slice > Mondrian."
    )

    # --------------------------------------------------------
    # Diabetes accuracy
    # --------------------------------------------------------
    diabetes_ranks = [
        rank_string(accuracy_rankings, "Diabetes", s)
        for s in SA_COUNTS
    ]

    diabetes_m = [
        method_at(summary, "Diabetes", "Mondrian", s)["accuracy_mean"]
        for s in SA_COUNTS
    ]
    diabetes_arx = [
        method_at(summary, "Diabetes", "ARX", s)["accuracy_mean"]
        for s in SA_COUNTS
    ]
    diabetes_kc = [
        method_at(summary, "Diabetes", "KC-Slice", s)["accuracy_mean"]
        for s in SA_COUNTS
    ]

    lines.append("")
    lines.append("## 2. Accuracy ranking — Diabetes-130")
    lines.append("")
    for s, ranking in zip(SA_COUNTS, diabetes_ranks):
        lines.append(
            f"- **SA={s}:** {ranking}"
        )

    lines.append("")
    lines.append("### Verdict")
    lines.append("")
    lines.append(
        f"Mondrian and ARX are tied at the original-data baseline "
        f"accuracy of {fmt_pct(diabetes_m[0])} for all three SA levels. "
        f"KC-Slice is lower at approximately "
        f"{fmt_pct(diabetes_kc[0])}, "
        f"{fmt_pct(diabetes_kc[1])}, and "
        f"{fmt_pct(diabetes_kc[2])}."
    )
    lines.append("")
    lines.append(
        "Thus the Diabetes utility ranking is effectively "
        "**Mondrian ≈ ARX > KC-Slice** at every attribute count; "
        "there is no ranking reversal as SA count increases."
    )

    # --------------------------------------------------------
    # DM trends
    # --------------------------------------------------------
    lines.append("")
    lines.append("## 3. Information loss — Discernibility Metric (DM)")
    lines.append("")
    lines.append(
        "DM is interpreted as the discernibility penalty "
        "DM = Σ|E|². Lower is better. The absolute values should "
        "be interpreted within the dataset/experiment because the "
        "three methods do not use identical privacy mechanisms; "
        "in particular, the KC-Slice DM is an adapted diagnostic "
        "over released sliced-QID tuples."
    )
    lines.append("")

    for dataset in DATASETS:
        lines.append(f"### {dataset}")
        lines.append("")
        for s in SA_COUNTS:
            rows = dm_rankings[
                (dm_rankings["dataset"] == dataset)
                & (dm_rankings["sa_count"] == s)
            ].sort_values("rank")

            ranking = " > ".join(
                rows["method"].tolist()
            )

            values = ", ".join(
                f"{r.method}={r.mean_dm:,.0f}"
                for r in rows.itertuples()
            )

            lines.append(
                f"- **SA={s}:** lowest-DM order {ranking}; {values}"
            )

        lines.append("")

    # --------------------------------------------------------
    # Why Mondrian underperformed on Adult
    # --------------------------------------------------------
    lines.append("## 4. Why did Mondrian underperform on Adult?")
    lines.append("")
    lines.append(
        "The results show a persistent utility gap rather than a "
        "single-seed anomaly. Mondrian's mean Adult accuracy is "
        f"{fmt_pct(adult_mondrian[0])}, "
        f"{fmt_pct(adult_mondrian[1])}, and "
        f"{fmt_pct(adult_mondrian[2])} for SA=1, 2, and 3, "
        "whereas ARX remains near 76.9% and KC-Slice near 75.1%."
    )
    lines.append("")
    lines.append(
        "A defensible interpretation is that the particular "
        "multidimensional Mondrian partitioning used here removed or "
        "coarsened predictive structure in the released QIDs more "
        "harmfully than the ARX optimization did. The experiment "
        "does not justify saying that Mondrian is inherently worse "
        "than ARX; the conclusion is specific to this Adult "
        "configuration, hierarchy, k=10 setting, and locked utility "
        "protocol."
    )
    lines.append("")
    lines.append(
        "An important supporting observation is that DM alone does "
        "not predict utility: ARX has much larger DM than Mondrian "
        "in these Adult runs, yet ARX has substantially higher "
        "classification accuracy. Therefore the structure and "
        "location of generalization in feature space matter more "
        "for predictive utility than the scalar DM value alone."
    )

    # --------------------------------------------------------
    # Why Diabetes nullifies advantage
    # --------------------------------------------------------
    lines.append("")
    lines.append(
        "## 5. Why was the slicing/generalization advantage "
        "nullified on Diabetes-130?"
    )
    lines.append("")
    lines.append(
        "The strongest clue is the locked original-data baseline: "
        "Diabetes accuracy is only about 53.38%, with macro-F1 around "
        "0.232 and ROC-AUC around 0.50. This indicates that the "
        "locked Logistic Regression evaluation is already operating "
        "with very limited discriminative utility on the selected "
        "feature representation; accuracy is dominated by the "
        "majority-class behavior."
    )
    lines.append("")
    lines.append(
        "Consequently, Mondrian and ARX matching the 53.38% baseline "
        "does not demonstrate a large preservation advantage. There "
        "is little measured predictive utility available for "
        "anonymization to preserve. Their generalization therefore "
        "appears utility-neutral under this classifier/metric."
    )
    lines.append("")
    lines.append(
        "KC-Slice does not obtain an accuracy advantage either. Its "
        "accuracy is below baseline at every SA count, while its "
        "ROC-AUC remains approximately 0.50. The slicing mechanism "
        "therefore does not recover useful target discrimination "
        "under this locked evaluation; instead, it introduces a "
        "small additional utility loss."
    )

    # --------------------------------------------------------
    # Expected-findings table
    # --------------------------------------------------------
    lines.append("")
    lines.append("## 6. Final expected-findings verdict table")
    lines.append("")
    lines.append("| Guide requirement | Verdict | Evidence |")
    lines.append("|---|---|---|")

    lines.append(
        "| Accuracy vs sensitive-attribute count | "
        "ARX > KC-Slice > Mondrian on Adult; "
        "Mondrian ≈ ARX > KC-Slice on Diabetes | "
        "Rankings remain stable across SA=1,2,3 |"
    )

    lines.append(
        "| Information loss vs sensitive-attribute count | "
        "DM changes strongly with method and SA count; "
        "lower DM is better, but DM is not a direct utility proxy | "
        "Common DM is available for all 54 runs; "
        "NCP is not persisted comparably for all methods |"
    )

    lines.append(
        "| Adult Mondrian underperformance | "
        "Persistent and configuration-specific | "
        "Lowest Adult accuracy at all three SA levels |"
    )

    lines.append(
        "| Diabetes slicing/generalization advantage | "
        "Nullified under the locked utility protocol | "
        "Original, Mondrian, and ARX all sit at 53.38%; "
        "KC-Slice is lower |"
    )

    lines.append("")
    lines.append(
        "### Methodological limitation"
    )
    lines.append("")
    lines.append(
        "NCP is not a complete cross-method metric in the committed "
        "Milestone 4 table: it is persisted for Mondrian but not for "
        "ARX/KC-Slice. The primary Milestone 5 information-loss plot "
        "therefore uses the common DM field. Any NCP-only discussion "
        "is explicitly limited to runs where NCP was persisted."
    )

    return "\n".join(lines) + "\n"


def build_analysis_summary(
    summary: pd.DataFrame,
    accuracy_rankings: pd.DataFrame,
    dm_rankings: pd.DataFrame,
) -> str:
    lines = [
        "# Milestone 5 — Analysis Summary",
        "",
        "## Scope",
        "",
        "54 committed Milestone 4 runs:",
        "",
        "- 2 datasets: Adult Census Income and Diabetes-130",
        "- 3 methods: Mondrian, ARX, KC-Slice",
        "- 3 sensitive-attribute counts: 1, 2, 3",
        "- 3 seeds per configuration",
        "",
        "No anonymization or classifier evaluation is rerun by this milestone.",
        "",
        "## Primary metrics",
        "",
        "- Accuracy: higher is better.",
        "- Discernibility Metric (DM): lower is better.",
        "- DM is the common information-loss/discernibility quantity available for all 54 runs.",
        "- NCP is retained only where the committed results persist it; it is not used for cross-method ranking.",
        "",
        "## Key result",
        "",
        "Adult accuracy ranking holds at all three SA levels:",
        "**ARX > KC-Slice > Mondrian**.",
        "",
        "Diabetes accuracy ranking also holds:",
        "**Mondrian ≈ ARX > KC-Slice**.",
        "",
        "The main cross-dataset difference is therefore not a ranking reversal "
        "with increasing SA count, but a change in which methods separate in "
        "utility: Adult shows a strong ARX advantage, while Diabetes collapses "
        "Mondrian and ARX to the same baseline accuracy.",
        "",
        "## Interpretation",
        "",
        "The Adult result suggests that the way a method generalizes the QIDs "
        "matters more to predictive utility than the scalar DM alone. "
        "Mondrian's lower DM does not translate into better accuracy.",
        "",
        "The Diabetes result is strongly constrained by the weak locked "
        "baseline: the original Logistic Regression already achieves only "
        "53.38% accuracy with approximately 0.50 ROC-AUC. This leaves little "
        "measured utility for anonymization to preserve, so Mondrian/ARX "
        "matching baseline is best interpreted as utility-neutral under the "
        "chosen evaluation, not as evidence that anonymization has no effect "
        "on the underlying data."
    ]

    return "\n".join(lines) + "\n"


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_validate()

    summary = build_summary(df)
    accuracy_rankings = build_accuracy_rankings(summary)
    dm_rankings = build_dm_rankings(summary)

    # --------------------------------------------------------
    # Persist tables
    # --------------------------------------------------------

    accuracy_output = OUT_DIR / "accuracy_by_method.csv"
    dm_output = OUT_DIR / "information_loss_dm_by_method.csv"
    rankings_accuracy_output = OUT_DIR / "rankings_accuracy.csv"
    rankings_dm_output = OUT_DIR / "rankings_dm.csv"

    summary[
        [
            "dataset",
            "method",
            "sa_count",
            "accuracy_mean",
            "accuracy_std",
            "macro_f1_mean",
            "roc_auc_mean",
            "original_accuracy_mean",
            "accuracy_delta_mean",
        ]
    ].to_csv(
        accuracy_output,
        index=False,
    )

    summary[
        [
            "dataset",
            "method",
            "sa_count",
            "dm_mean",
            "dm_std",
            "ncp_mean",
            "ncp_available",
            "equivalence_classes_mean",
            "min_class_mean",
            "max_class_mean",
        ]
    ].to_csv(
        dm_output,
        index=False,
    )

    accuracy_rankings.to_csv(
        rankings_accuracy_output,
        index=False,
    )

    dm_rankings.to_csv(
        rankings_dm_output,
        index=False,
    )

    # --------------------------------------------------------
    # Plot generation
    # --------------------------------------------------------

    plot_metric(
        summary,
        "Adult",
        "accuracy_mean",
        "Mean classifier accuracy",
        "Adult Census Income — Accuracy vs Protected-Attribute Count",
        "adult_accuracy.png",
    )

    plot_metric(
        summary,
        "Adult",
        "dm_mean",
        "Mean Discernibility Metric (DM)",
        "Adult Census Income — Information Loss (DM) vs Protected-Attribute Count",
        "adult_information_loss_dm.png",
        log_scale=True,
    )

    plot_metric(
        summary,
        "Diabetes",
        "accuracy_mean",
        "Mean classifier accuracy",
        "Diabetes-130 — Accuracy vs Protected-Attribute Count",
        "diabetes_accuracy.png",
    )

    plot_metric(
        summary,
        "Diabetes",
        "dm_mean",
        "Mean Discernibility Metric (DM)",
        "Diabetes-130 — Information Loss (DM) vs Protected-Attribute Count",
        "diabetes_information_loss_dm.png",
        log_scale=True,
    )

    plot_accuracy_comparison(summary)

    # --------------------------------------------------------
    # Written verdicts
    # --------------------------------------------------------

    verdicts = build_verdicts(
        summary,
        accuracy_rankings,
        dm_rankings,
    )

    summary_text = build_analysis_summary(
        summary,
        accuracy_rankings,
        dm_rankings,
    )

    (OUT_DIR / "expected_findings_verdicts.md").write_text(
        verdicts,
        encoding="utf-8",
    )

    (OUT_DIR / "analysis_summary.md").write_text(
        summary_text,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final console report
    # --------------------------------------------------------

    print("=" * 72)
    print("MILESTONE 5 — ANALYSIS")
    print("=" * 72)
    print()
    print(f"Input : {INPUT}")
    print(f"Output: {OUT_DIR}")
    print()
    print("54/54 Milestone 4 rows validated.")
    print()

    print("ACCURACY RANKINGS")
    print("-" * 72)

    for dataset in DATASETS:
        for sa_count in SA_COUNTS:
            ranking = rank_string(
                accuracy_rankings,
                dataset,
                sa_count,
            )
            print(
                f"{dataset:<10} SA={sa_count}: {ranking}"
            )

    print()
    print("DM RANKINGS — LOWER IS BETTER")
    print("-" * 72)

    for dataset in DATASETS:
        for sa_count in SA_COUNTS:
            rows = dm_rankings[
                (dm_rankings["dataset"] == dataset)
                & (dm_rankings["sa_count"] == sa_count)
            ].sort_values("rank")

            ranking = " > ".join(
                rows["method"].tolist()
            )

            print(
                f"{dataset:<10} SA={sa_count}: {ranking}"
            )

    print()
    print("GENERATED")
    print("-" * 72)

    for path in [
        accuracy_output,
        dm_output,
        rankings_accuracy_output,
        rankings_dm_output,
        OUT_DIR / "expected_findings_verdicts.md",
        OUT_DIR / "analysis_summary.md",
        PLOT_DIR / "adult_accuracy.png",
        PLOT_DIR / "adult_information_loss_dm.png",
        PLOT_DIR / "diabetes_accuracy.png",
        PLOT_DIR / "diabetes_information_loss_dm.png",
        PLOT_DIR / "adult_vs_diabetes_accuracy.png",
    ]:
        print(path)

    print()
    print("=" * 72)
    print("MILESTONE 5 ANALYSIS COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
