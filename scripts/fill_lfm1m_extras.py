"""
Fill the two placeholder columns of the revised LFM1M tables.

1) Evidence density  Rbar = mean over anchors of relevance/|E_S|,
   then mean over PGPR and CAFE.  Computed from the per-anchor parquets
   written by metrics.compute_metrics (results/per_anchor/lfm1m_*.parquet).

2) Comprehensibility gap (CG), read from the regenerated summary.csv
   (requires the gender-overlay patch to metrics/compute_metrics.py and
   one re-run of `python3 -m metrics.compute_metrics --dataset lfm1m`;
   the sweep itself does NOT need to be re-run).

Usage (from the repo root):
    python3 fill_lfm1m_extras.py
Prints one line per (scenario, algorithm) with Rbar and CG, in table
row order, ready to paste into the \\darev{--} cells.
"""

from __future__ import annotations
import glob
import numpy as np
import pandas as pd

ROW_ORDER = ["st", "pcst", "wpcst(degree)", "wpcst(pagerank)",
             "naive_union", "pappas2017", "mst", "faces", "supernode"]


def _alg(row) -> str:
    a, c = str(row["algorithm"]), str(row.get("centrality", ""))
    return f"wpcst({c})" if a == "wpcst" else a


def main():
    # ---- Rbar from per-anchor parquets --------------------------------
    files = glob.glob("results/per_anchor/lfm1m_*.parquet")
    if not files:
        raise SystemExit("No results/per_anchor/lfm1m_*.parquet found; "
                         "run metrics.compute_metrics --dataset lfm1m first.")
    frames = []
    for f in files:
        d = pd.read_parquet(f)
        need = {"scenario", "baseline", "algorithm", "relevance", "num_edges"}
        if not need <= set(d.columns):
            continue
        frames.append(d)
    big = pd.concat(frames, ignore_index=True)
    big["alg"] = big.apply(_alg, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        big["rbar"] = big["relevance"] / big["num_edges"].replace(0, np.nan)
    # per-(scenario, baseline, alg) mean over anchors, then mean over baselines
    cell = (big.groupby(["scenario", "baseline", "alg"], as_index=False)
              .agg(rbar=("rbar", "mean")))
    rbar = (cell.groupby(["scenario", "alg"], as_index=False)
                .agg(rbar=("rbar", "mean")))

    # ---- CG from the regenerated summary.csv --------------------------
    cg = None
    try:
        s = pd.read_csv("results/lfm1m/summary.csv")
        s["alg"] = s.apply(_alg, axis=1)
        cg = (s.groupby(["scenario", "alg"], as_index=False)
                .agg(cg=("comprehensibility_gap", "mean")))
    except FileNotFoundError:
        print("[warn] results/lfm1m/summary.csv not found; CG skipped.")

    for scen in ["user_centric", "user_group"]:
        print(f"\n===== {scen} (mean over PGPR, CAFE) =====")
        print(f"{'algorithm':<18} {'Rbar':>8} {'CG':>9}")
        for alg in ROW_ORDER:
            r = rbar[(rbar.scenario == scen) & (rbar.alg == alg)]
            rv = f"{float(r['rbar'].iloc[0]):.2f}" if len(r) else "n/a"
            cv = "n/a"
            if cg is not None:
                c = cg[(cg.scenario == scen) & (cg.alg == alg)]
                if len(c) and not np.isnan(c["cg"].iloc[0]):
                    cv = f"{float(c['cg'].iloc[0]):.4f}"
            print(f"{alg:<18} {rv:>8} {cv:>9}")
    print("\nPaste Rbar into the $\\bar R$ column and CG into the CG column "
          "(user-centric only) of the revised tables.")


if __name__ == "__main__":
    main()
