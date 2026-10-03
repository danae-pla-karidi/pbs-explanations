from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import DATASETS, RESULTS_ROOT

METRIC_LABEL = {"comprehensibility": "Comprehensibility", "relevance": "Relevance",
                "actionability": "Actionability", "diversity": "Diversity",
                "faithfulness": "Faithfulness"}
ALG_COLOR = {"st": "#1f77b4", "pcst": "#2ca02c", "wpcst": "#d62728",
             "naive_union": "#7f7f7f"}
METRIC_STYLE = {"comprehensibility": "-", "relevance": "--",
                "actionability": ":", "diversity": "-.", "faithfulness": (0, (3, 1, 1, 1))}
# spoke order: pure-rating first, then increasing recency
SHARE_ORDER = [0.0, 0.25, 0.5, 0.75, 1.0]


def _s0_from_summary(summary_csv, dataset, scenario, baseline, alg, centrality):
    """Return (C, R) for the committed graph (beta1=1, beta2=0) from summary.csv."""
    if not summary_csv.exists():
        return None
    df = pd.read_csv(summary_csv)
    m = ((df["dataset"] == dataset) & (df["scenario"] == scenario) &
         (df["baseline"] == baseline) & (df["algorithm"] == alg))
    sub = df[m]
    if alg == "wpcst":
        sub = sub[sub["centrality"] == centrality]
    else:
        sub = sub[sub["centrality"].isna() | (sub["centrality"].astype(str) == "")]
    if sub.empty:
        return None
    r = sub.iloc[0]
    return float(r["comprehensibility_mean"]), float(r["relevance_mean"])


def _series(long_df, summary_csv, dataset, scenario, baseline, alg, metric, centrality):
    """Assemble the 5 spoke values for one (alg, metric): 4 from the ablation
    CSV plus the s=0 point from summary.csv.  Returns dict {share: value}."""
    m = ((long_df["dataset"] == dataset) & (long_df["scenario"] == scenario) &
         (long_df["baseline"] == baseline) & (long_df["algorithm"] == alg))
    sub = long_df[m]
    vals = {float(r["recency_share"]): float(r[metric]) for _, r in sub.iterrows()
            if pd.notna(r[metric])}
    if 0.0 not in vals:
        s0 = _s0_from_summary(summary_csv, dataset, scenario, baseline, alg, centrality)
        if s0 is not None:
            vals[0.0] = s0[0] if metric == "comprehensibility" else s0[1]
    return vals


def _radar(ax, series, shares):
    """series: list of (label, color, linestyle, {share: value}). Each series is
    min-max normalized to its own range; the range is appended to its label."""
    n = len(shares)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(1)
    ax.set_xticks(angles[:-1])
    labels = [r"$(\beta_1{=}%g,\beta_2{=}%g)$" % (1 - s, s) for s in shares]
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.08)

    for color, ls, vals in series:
        present = [vals.get(s, np.nan) for s in shares]
        arr = np.array(present, dtype=float)
        if np.all(np.isnan(arr)):
            continue
        lo, hi = np.nanmin(arr), np.nanmax(arr)
        norm = np.full_like(arr, 0.5) if hi - lo < 1e-12 else (arr - lo) / (hi - lo)
        norm = np.nan_to_num(norm, nan=0.0)
        closed = norm.tolist() + norm[:1].tolist()
        ax.plot(angles, closed, color=color, linestyle=ls, linewidth=1.5)
        ax.fill(angles, closed, color=color, alpha=0.06)
    ax.grid(True, alpha=0.3)


def _panel_series(long_df, summary_csv, dataset, scenario, baseline,
                  algorithms, metrics, centrality):
    series = []
    for alg in algorithms:
        for metric in metrics:
            vals = _series(long_df, summary_csv, dataset, scenario,
                           baseline, alg, metric, centrality)
            if vals:
                series.append((ALG_COLOR.get(alg, "#333333"),
                               METRIC_STYLE.get(metric, "-"), vals))
    return series


def _shared_legend(fig, algorithms, metrics):
    """One compact legend: color = algorithm, line style = metric."""
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=ALG_COLOR.get(a, "#333"), lw=2,
                      label=a.upper()) for a in algorithms]
    handles += [Line2D([0], [0], color="#444", lw=2,
                       linestyle=METRIC_STYLE.get(m, "-"),
                       label=METRIC_LABEL.get(m, m)) for m in metrics]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.0))


def make(dataset, *, baselines, scenarios, algorithms, metrics,
         long_csv, summary_csv, out_dir, centrality, layout="grid"):
    if not long_csv.exists():
        print(f"[radar] {long_csv} not found; run runners.run_beta_ablation first.")
        return
    long_df = pd.read_csv(long_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    # panels in reading order: rows = scenarios, cols = recommenders
    panels = [(sc, bl) for sc in scenarios for bl in baselines]

    if layout == "grid":
        ncol = len(baselines)
        nrow = len(scenarios)
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 3.8 * nrow),
                                 subplot_kw={"projection": "polar"})
        axes = np.atleast_1d(axes).ravel()
        for ax, (scenario, baseline) in zip(axes, panels):
            s = _panel_series(long_df, summary_csv, dataset, scenario,
                              baseline, algorithms, metrics, centrality)
            if not s:
                ax.set_axis_off()
                continue
            _radar(ax, s, SHARE_ORDER)
            ax.set_title(f"{scenario.replace('_', '-')}, {baseline.upper()}",
                         fontsize=11, pad=14)
        _shared_legend(fig, algorithms, metrics)
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        out = out_dir / f"radar_beta_{dataset}_grid.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[radar] wrote {out}")
        return

    # separate: one file per panel, no titles, shared legend each
    for scenario, baseline in panels:
        s = _panel_series(long_df, summary_csv, dataset, scenario,
                          baseline, algorithms, metrics, centrality)
        if not s:
            print(f"[radar] no data for {baseline}/{scenario}; skipping.")
            continue
        fig = plt.figure(figsize=(3.6, 3.6))
        ax = fig.add_subplot(111, projection="polar")
        _radar(ax, s, SHARE_ORDER)
        _shared_legend(fig, algorithms, metrics)
        fig.tight_layout(rect=[0, 0.08, 1, 1])
        out = out_dir / f"radar_beta_{dataset}_{scenario}_{baseline}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[radar] wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    p.add_argument("--baselines", default="pgpr,cafe")
    p.add_argument("--scenarios", default="user_centric,user_group")
    p.add_argument("--algorithms", default="st,pcst,wpcst")
    p.add_argument("--metrics", default="comprehensibility,relevance")
    p.add_argument("--centrality", default="degree",
                   help="WPCST centrality variant to read from both CSVs")
    p.add_argument("--long-csv", default=None)
    p.add_argument("--summary-csv", default="summary.csv",
                   help="committed-graph results for the s=0 spoke")
    p.add_argument("--layout", default="grid", choices=["grid", "separate"],
                   help="grid: one combined figure; separate: one file per panel")
    args = p.parse_args()

    long_csv = Path(args.long_csv) if args.long_csv else \
        (RESULTS_ROOT / "_beta_ablation" / args.dataset / "beta_ablation_long.csv")
    out_dir = RESULTS_ROOT / args.dataset / "figures"
    make(args.dataset,
         baselines=[b.strip() for b in args.baselines.split(",")],
         scenarios=[s.strip() for s in args.scenarios.split(",")],
         algorithms=[a.strip() for a in args.algorithms.split(",")],
         metrics=[m.strip() for m in args.metrics.split(",")],
         long_csv=long_csv, summary_csv=Path(args.summary_csv),
         out_dir=out_dir, centrality=args.centrality, layout=args.layout)


if __name__ == "__main__":
    main()
