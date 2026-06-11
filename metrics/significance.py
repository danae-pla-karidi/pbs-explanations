"""
Statistical significance testing for the TKDE submission.

This is the script the reviewer asked for. It reads the per-anchor parquet
files produced by `metrics.compute_metrics`, performs paired Wilcoxon
signed-rank tests for the WPCST-vs-baseline comparisons, applies Holm
multiple-comparisons correction within each (scenario, recommender) cell, and
writes a tidy CSV plus a publication-ready LaTeX table.

Two key contrasts are tested for every (dataset, scenario, recommender):
    1. WPCST vs ST     - did weighting help compared to plain Steiner?
    2. WPCST vs PCST   - did weighting help compared to uniform-prize PCST?

Pairing is by anchor_id: for user-centric, every user contributes one
ST score, one PCST score, and one WPCST score; we pair them and test on
the differences.

Why Wilcoxon and not paired t-test:
    - Most metrics are bounded in [0,1] and skewed (e.g. comprehensibility
      under WPCST piles up near 1 because num_edges is often 1).
    - Runtime is heavy-tailed.
    - The non-parametric test makes no normality assumption.

Holm correction is applied across the metric family within each
(dataset, scenario, recommender) so the family-wise error rate stays
controlled across the ~9 metrics tested.

Outputs (per-dataset only; datasets are not merged into a single table):
    results/<dataset>/significance.csv          - long-form per-cell rows
                                                  for that dataset.
    results/<dataset>/significance_summary.csv  - paper-facing wide form
                                                  with star markers.
    results/<dataset>/significance_table.tex    - LaTeX table for that
                                                  dataset, ready to paste
                                                  into the paper.
    results/significance/wilcoxon.csv           - all datasets together,
                                                  long-form, kept only as
                                                  a debug backstop.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from config import RESULTS_ROOT, METRICS, SIGNIFICANCE_ALPHA


METRIC_FAMILY = [m for m in METRICS if m not in ("memory_usage_mb", "runtime_sec")]
# `memory_usage_mb` is excluded from the test family because RSS deltas are
# noisy on a shared workstation. `runtime_sec` (wall-clock) is also excluded
# because it is affected by worker-pool contention; we test `cpu_time_sec`
# instead, which is the contention-free per-anchor cost. Both timings remain
# in the summary CSV for reference.

CONTRASTS = [("wpcst", "st"), ("wpcst", "pcst")]


# ---------------------------------------------------------------------------
# Per-cell test
# ---------------------------------------------------------------------------

def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return (statistic, p_value). a and b are aligned per-anchor scores.
    NaNs are dropped pairwise; if fewer than 5 pairs remain, returns NaN."""
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 5 or np.allclose(a, b):
        return float("nan"), float("nan")
    try:
        # zero_method='wilcox' (default) discards zero-differences; use
        # 'pratt' to keep them but adjust ranks. We pick 'pratt' to retain
        # statistical power in the (frequent) case where two algorithms
        # produce identical scores for many anchors.
        stat, p = wilcoxon(a, b, zero_method="pratt", alternative="two-sided")
        return float(stat), float(p)
    except ValueError:
        return float("nan"), float("nan")


def holm_correct(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down. Returns adjusted p-values in input order."""
    n = len(p_values)
    order = sorted(range(n), key=lambda i: (np.isnan(p_values[i]), p_values[i]))
    adj = [float("nan")] * n
    cummax = 0.0
    for rank, idx in enumerate(order):
        p = p_values[idx]
        if np.isnan(p):
            adj[idx] = float("nan")
            continue
        adj_p = (n - rank) * p
        cummax = max(cummax, adj_p)
        adj[idx] = min(1.0, cummax)
    return adj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def gather(per_anchor_dir: Path) -> pd.DataFrame:
    parts = []
    for f in sorted(per_anchor_dir.glob("*.parquet")):
        parts.append(pd.read_parquet(f))
    if not parts:
        raise RuntimeError(f"No per-anchor parquet files found in {per_anchor_dir}.")
    return pd.concat(parts, ignore_index=True)


def run_tests(big: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cells = big[["dataset", "scenario", "baseline"]].drop_duplicates().itertuples(index=False)
    for ds, sc, bl in cells:
        cell = big[(big.dataset == ds) & (big.scenario == sc) & (big.baseline == bl)]
        # Pivot: anchors x algorithms for each metric
        for metric in METRIC_FAMILY:
            if metric not in cell.columns:
                continue
            pivot = cell.pivot_table(
                index="anchor_id", columns="algorithm", values=metric,
                aggfunc="first",
            )
            for new_alg, ref_alg in CONTRASTS:
                if new_alg not in pivot.columns or ref_alg not in pivot.columns:
                    rows.append({
                        "dataset": ds, "scenario": sc, "baseline": bl,
                        "metric": metric, "contrast": f"{new_alg}_vs_{ref_alg}",
                        "n_pairs": 0, "stat": float("nan"), "p_value": float("nan"),
                        "median_diff": float("nan"),
                    })
                    continue
                a = pivot[new_alg].to_numpy(dtype=float)
                b = pivot[ref_alg].to_numpy(dtype=float)
                stat, p = paired_wilcoxon(a, b)
                rows.append({
                    "dataset": ds, "scenario": sc, "baseline": bl,
                    "metric": metric, "contrast": f"{new_alg}_vs_{ref_alg}",
                    "n_pairs": int(np.sum(~(np.isnan(a) | np.isnan(b)))),
                    "stat": stat, "p_value": p,
                    "median_diff": float(np.nanmedian(a - b)),
                })

    df = pd.DataFrame(rows)
    # Holm correction within each (dataset, scenario, baseline, contrast)
    df["p_adj_holm"] = float("nan")
    for keys, sub in df.groupby(["dataset", "scenario", "baseline", "contrast"]):
        idx = sub.index.tolist()
        adj = holm_correct(sub["p_value"].tolist())
        for i, a in zip(idx, adj):
            df.at[i, "p_adj_holm"] = a
    df["significant"] = df["p_adj_holm"] < SIGNIFICANCE_ALPHA
    return df


def to_latex_per_dataset(df: pd.DataFrame, out_root: Path) -> None:
    """Write one LaTeX significance table per dataset, under
    `results/<dataset>/significance_table.tex`.

    Each table has one row per (scenario, metric) showing the worst-case
    Holm-adjusted p-value across recommenders for the WPCST-vs-ST and
    WPCST-vs-PCST contrasts, with star markers.  No cross-dataset table
    is produced (the user asked for per-dataset deliverables only).
    """
    def stars(p):
        if np.isnan(p):
            return "--"
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "n.s."

    if df.empty:
        return
    for ds, sub in df.groupby("dataset"):
        grouped = (
            sub.groupby(["scenario", "metric", "contrast"])["p_adj_holm"]
               .max().reset_index()
        )
        pivot = grouped.pivot_table(
            index=["scenario", "metric"], columns="contrast",
            values="p_adj_holm", aggfunc="max",
        ).reset_index()

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{%s: significance of WPCST improvements (Wilcoxon "
            r"signed-rank, Holm-corrected). Reported value is the "
            r"worst-case adjusted $p$ across recommenders within each "
            r"cell. $^{*} p<0.05$, $^{**} p<0.01$, $^{***} p<0.001$.}"
            % ds.upper(),
            rf"\label{{tab:significance-{ds}}}",
            r"\begin{tabular}{llll}",
            r"\toprule",
            r"Scenario & Metric & WPCST vs ST & WPCST vs PCST \\",
            r"\midrule",
        ]
        for _, r in pivot.iterrows():
            lines.append(
                f"{r['scenario'].replace('_', '-')} & {r['metric']} & "
                f"{stars(r.get('wpcst_vs_st', float('nan')))} & "
                f"{stars(r.get('wpcst_vs_pcst', float('nan')))} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        out_path = out_root / ds / "significance_table.tex"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines))
        print(f"[significance] wrote {out_path}")


def _write_per_dataset_csvs(df: pd.DataFrame) -> None:
    """Write two per-dataset CSV files for each dataset that appears in `df`:

      results/<dataset>/significance.csv         - exact mirror of the long-form
                                                   wilcoxon.csv, filtered to that
                                                   dataset.  Sits next to summary.csv
                                                   so per-dataset deliverables are
                                                   in one folder.
      results/<dataset>/significance_summary.csv - paper-ready: one row per
                                                   (scenario, baseline, metric)
                                                   with WPCST-vs-ST and
                                                   WPCST-vs-PCST p-values plus
                                                   star markers, suitable for
                                                   pasting into Excel or a
                                                   table.
    """
    if df.empty:
        return
    for ds, sub in df.groupby("dataset"):
        ds_dir = RESULTS_ROOT / ds
        ds_dir.mkdir(parents=True, exist_ok=True)

        # Mirror
        mirror = ds_dir / "significance.csv"
        sub.to_csv(mirror, index=False)
        print(f"[significance] wrote {mirror}")

        # Paper-ready wide form
        def stars(p: float) -> str:
            if pd.isna(p):
                return "--"
            if p < 0.001:
                return "***"
            if p < 0.01:
                return "**"
            if p < 0.05:
                return "*"
            return "n.s."

        wide = sub.pivot_table(
            index=["scenario", "baseline", "metric"],
            columns="contrast",
            values=["p_adj_holm", "median_diff", "n_pairs"],
            aggfunc="first",
        )
        # Flatten column index (e.g. ("p_adj_holm", "wpcst_vs_st") -> "wpcst_vs_st_p_adj_holm")
        wide.columns = [f"{contrast}_{stat}" for stat, contrast in wide.columns]
        wide = wide.reset_index()
        # Add star columns for the two contrasts the paper reports
        for contrast in ("wpcst_vs_st", "wpcst_vs_pcst"):
            col = f"{contrast}_p_adj_holm"
            if col in wide.columns:
                wide[f"{contrast}_stars"] = wide[col].apply(stars)
        # Order columns: identifiers, then per-contrast blocks
        id_cols = ["scenario", "baseline", "metric"]
        contrast_cols = []
        for contrast in ("wpcst_vs_st", "wpcst_vs_pcst"):
            for stat in ("p_adj_holm", "stars", "median_diff", "n_pairs"):
                col = f"{contrast}_{stat}"
                if col in wide.columns:
                    contrast_cols.append(col)
        wide = wide[id_cols + contrast_cols]

        paper_csv = ds_dir / "significance_summary.csv"
        wide.to_csv(paper_csv, index=False)
        print(f"[significance] wrote {paper_csv}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-anchor-dir", default=str(RESULTS_ROOT / "per_anchor"))
    p.add_argument(
        "--long-csv",
        default=str(RESULTS_ROOT / "significance" / "wilcoxon.csv"),
        help="Long-form, all-datasets CSV used as a debug backstop. "
             "The paper-facing outputs are the per-dataset CSVs and "
             "LaTeX tables under results/<dataset>/.",
    )
    args = p.parse_args()

    big = gather(Path(args.per_anchor_dir))
    df = run_tests(big)

    # Long-form, all-datasets CSV (handy for ad-hoc inspection; not a
    # paper deliverable).
    long_csv = Path(args.long_csv)
    long_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(long_csv, index=False)
    print(f"[significance] wrote {long_csv}")

    # Per-dataset deliverables: one CSV pair + one LaTeX table per dataset.
    _write_per_dataset_csvs(df)
    to_latex_per_dataset(df, RESULTS_ROOT)

    sig = df[df["significant"] == True]  # noqa: E712
    print(f"[significance] {len(sig)}/{len(df)} contrasts significant at alpha={SIGNIFICANCE_ALPHA}")


if __name__ == "__main__":
    main()
