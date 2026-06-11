"""
Plot generator for the TKDE submission.

Per-dataset only: one figure per (dataset, variant) pair.  Datasets are
never merged into the same figure.

For each dataset that has a `results/<dataset>/summary.csv`, two PDFs
are written under `results/<dataset>/figures/`:

  fig_cr_tradeoff_<dataset>_all.pdf   — every algorithm
  fig_cr_tradeoff_<dataset>_core.pdf  — only ST, PCST, WPCST

Each figure has up to four scenario panels (user-centric, user-group,
item-centric, item-group); panels for which the dataset has no data are
shown empty.  LFM-1M, for example, has only the two user-side panels.

The aggregate `summary.csv` columns this script consumes:
    dataset, scenario, baseline, algorithm,
    comprehensibility_mean, relevance_mean
"""

from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from config import DATASETS, RESULTS_ROOT


SHAPE = {
    "st":          "o",   # circle
    "pcst":        "^",   # triangle
    "wpcst":       "s",   # square
    "naive_union": "D",   # diamond
    "pappas2017":  "h",   # hexagon
    "mst":         "p",   # pentagon
    "faces":       "*",   # star
    "supernode":   "X",   # x-cross
}
# Okabe-Ito palette: color-blind-safe.
COLORS = {
    "pgpr": "#0072B2",
    "cafe": "#D55E00",
    "plm":  "#009E73",
    "plmr": "#CC79A7",
}

CORE_ALGS = ["st", "pcst", "wpcst"]
PANELS = ["user_centric", "user_group", "item_centric", "item_group"]


def _cr_panel(ax, df: pd.DataFrame, scenario: str) -> None:
    """Render one scenario panel.  Empty dataframe -> blank panel with title."""
    sub = df[df.scenario == scenario]
    if sub.empty:
        ax.set_title(f"{scenario.replace('_', '-')} (no data)")
        ax.set_axis_off()
        return
    for _, row in sub.iterrows():
        x = row["comprehensibility_mean"]
        y = row["relevance_mean"]
        marker = SHAPE.get(row["algorithm"], "o")
        color = COLORS.get(row["baseline"], "#888")
        ax.scatter(x, y, marker=marker, c=color, s=70, alpha=0.85,
                   edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Comprehensibility")
    ax.set_ylabel("Relevance (log scale)")
    ax.set_title(scenario.replace("_", "-"))
    ax.set_yscale("symlog", linthresh=1.0)  # symlog handles 0 and small values
    ax.grid(True, alpha=0.3, which="both")


def _legend_handles(algorithms: list[str]) -> list:
    return [
        plt.Line2D([], [], marker=SHAPE[a], linestyle="", color="black",
                   markerfacecolor="white", markeredgecolor="black",
                   markersize=8, label=a)
        for a in algorithms
    ]


def _recommender_handles(baselines: list[str]) -> list:
    return [
        plt.Line2D([], [], marker="s", linestyle="", color=COLORS[b],
                   markersize=10, label=b)
        for b in baselines if b in COLORS
    ]


def _make_figure(df: pd.DataFrame, out_path: Path, *,
                 title: str, alg_filter: list[str] | None) -> None:
    if alg_filter is not None:
        df = df[df.algorithm.isin(alg_filter)]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=120)
    fig.suptitle(title, fontsize=12, y=1.00)

    for ax, sc in zip(axes.flatten(), PANELS):
        _cr_panel(ax, df, sc)

    algs_present = sorted(set(df["algorithm"]).intersection(SHAPE),
                          key=list(SHAPE).index)
    baselines_present = sorted(set(df["baseline"]).intersection(COLORS),
                               key=list(COLORS).index)
    fig.legend(handles=_legend_handles(algs_present),
               loc="lower left", ncol=min(len(algs_present), 5),
               title="Algorithm", bbox_to_anchor=(0.02, -0.02), frameon=False)
    fig.legend(handles=_recommender_handles(baselines_present),
               loc="lower right", ncol=min(len(baselines_present), 4),
               title="Recommender", bbox_to_anchor=(0.98, -0.02), frameon=False)

    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] wrote {out_path}")


def make_for_dataset(dataset: str, summary_csv: Path, out_dir: Path) -> None:
    if not summary_csv.exists():
        print(f"[plots] {summary_csv} not found - skipping {dataset}")
        return
    df = pd.read_csv(summary_csv)
    if df.empty:
        print(f"[plots] {summary_csv} is empty - skipping {dataset}")
        return
    # Defensive: enforce single-dataset content so a misconfigured CSV
    # cannot bleed another dataset into this figure.
    df = df[df["dataset"] == dataset]
    if df.empty:
        print(f"[plots] no rows with dataset={dataset!r} in {summary_csv}; skip")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    _make_figure(
        df, out_dir / f"fig_cr_tradeoff_{dataset}_all.pdf",
        title=f"{dataset.upper()}: comprehensibility-relevance trade-off (all algorithms)",
        alg_filter=None,
    )
    _make_figure(
        df, out_dir / f"fig_cr_tradeoff_{dataset}_core.pdf",
        title=f"{dataset.upper()}: comprehensibility-relevance trade-off (ST, PCST, WPCST)",
        alg_filter=CORE_ALGS,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset", default=None,
        choices=list(DATASETS.keys()) + [None],
        help="Render only this dataset (default: every dataset that has a summary.csv).",
    )
    args = p.parse_args()

    targets = [args.dataset] if args.dataset else list(DATASETS.keys())
    for ds in targets:
        summary = RESULTS_ROOT / ds / "summary.csv"
        out_dir = RESULTS_ROOT / ds / "figures"
        make_for_dataset(ds, summary, out_dir)


if __name__ == "__main__":
    main()
