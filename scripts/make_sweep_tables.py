"""
Build the two revision tables from the sweep outputs.

  Table A (budget sensitivity): per baseline and rho in {0.5, 1, 2}, per
      scenario: C, R, Rbar, mean |V_S|, and the saturation rate, i.e. the
      fraction of anchors whose candidate pool (optional path nodes) is not
      larger than the budget, so that the budget admits the whole pool.
  Table B (tuning): per baseline configuration at rho = 1, per scenario:
      C, R, Rbar (and mean cluster count for FACES).

The rho = 1 rows and the main-table configuration come from the main
per-anchor files (results/per_anchor/), so they equal the main tables.
Aggregation follows the main tables: mean over anchors per cell, then mean
over recommenders per scenario; Rbar = R / (1/C - 1) on the scenario means.

Usage:
    python -m scripts.make_sweep_tables --dataset ml1m
    python -m scripts.make_sweep_tables --dataset lfm1m
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd

from config import DATASETS, RESULTS_ROOT

NAMES = {"pappas2017": "CentPrune", "mst": "MST", "faces": "FACES"}
CENT = {"betweenness_approx": "betw.", "pagerank": "PageRank", "degree": "degree"}
KEYS = {"kg_type": "KG type", "path_type": "path type", "relation": "relation"}
SCEN_ORDER = ["user_centric", "item_centric", "user_group", "item_group"]
SCEN_NAME = {"user_centric": "User-centric", "item_centric": "Item-centric",
             "user_group": "User-group", "item_group": "Item-group"}


def load_main(dataset: str) -> pd.DataFrame:
    frames = []
    for alg in NAMES:
        for f in (RESULTS_ROOT / "per_anchor").glob(f"{dataset}_*_{alg}_cent_betweenness_approx.parquet"):
            df = pd.read_parquet(f)
            df["tag"], df["sweep"], df["rho"], df["faces_key"] = "main", "main", 1.0, "kg_type"
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def meta_col(df: pd.DataFrame, key: str) -> pd.Series:
    return df["metadata"].apply(lambda m: json.loads(m).get(key))


RBAR_MODE = "paper"   # "paper": R / (1/C - 1) on the means; "anchor": mean of R/|E_S|


def scenario_means(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["rbar_anchor"] = df["relevance"] / df["num_edges"].clip(lower=1)
    group_cols = ["dataset"] + group_cols
    cell = df.groupby(group_cols + ["scenario", "baseline"]).agg(
        C=("comprehensibility", "mean"), R=("relevance", "mean"),
        V=("num_nodes", "mean"), sat=("saturated", "mean"),
        ncl=("n_clusters", "mean"), Rbar_a=("rbar_anchor", "mean")).reset_index()
    sc = cell.groupby(group_cols + ["scenario"]).agg(
        C=("C", "mean"), R=("R", "mean"), V=("V", "mean"),
        sat=("sat", "mean"), ncl=("ncl", "mean"), Rbar_a=("Rbar_a", "mean"),
        n_rec=("baseline", "nunique")).reset_index()
    sc["Rbar"] = sc["Rbar_a"] if RBAR_MODE == "anchor" else sc["R"] / (1.0 / sc["C"] - 1.0)
    return sc


def fmt(x, nd=2):
    return "--" if pd.isna(x) else f"{x:.{nd}f}"


COMPACT = False   # True: omit the |V_S| and Sat. columns of Table A


def table_a(sc: pd.DataFrame, scenarios: list[str], dataset: str) -> str:
    datasets = [dataset] if dataset != "all" else [d for d in ("ml1m", "lfm1m") if (sc.dataset == d).any()]
    where = "" if dataset == "all" else f" on {dataset.upper()}"
    k = 3 if COMPACT else 5
    cols = "l l r " + " ".join(["r" * k] * len(scenarios))
    lines = [r"\begin{table*}[t]", r"\centering",
             rf"\caption{{Budget sensitivity of the structural baselines{where}: "
             r"comprehensibility ($C$), relevance ($R$), evidence density ($\bar R$), mean summary "
             r"size ($|V_S|$), and saturation rate (Sat., fraction of anchors whose candidate pool "
             r"fits within the budget), for budgets of $0.5\times$, $1\times$, and $2\times$ the "
             r"WPCST(degree) size. The $1\times$ rows are the main-table setting.}",
             rf"\label{{tab:budget_sweep_{dataset}}}",
             r"\begin{adjustbox}{max width=\textwidth}",
             rf"\begin{{tabular}}{{@{{}}{cols}@{{}}}}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{{k}}}{{c}}{{\emph{{{SCEN_NAME[s]}}}}}" for s in scenarios) + r" \\"]
    cm = " ".join(rf"\cmidrule(lr){{{4+k*i}-{3+k+k*i}}}" for i in range(len(scenarios)))
    lines.append(cm)
    hdr = r"$C\uparrow$ & $R\uparrow$ & $\bar R\uparrow$" + ("" if COMPACT else r" & $|V_S|$ & Sat.")
    lines.append("Dataset & Algorithm & $\\rho$ & " + " & ".join(hdr for _ in scenarios) + r" \\")
    lines.append(r"\midrule")
    for di, ds in enumerate(datasets):
        if di:
            lines.append(r"\midrule")
        for ai, alg in enumerate(NAMES):
            for rho in (0.5, 1.0, 2.0):
                row = [ds.upper() if (ai == 0 and rho == 0.5) else "",
                       NAMES[alg] if rho == 0.5 else "", f"{rho:g}"]
                for s in scenarios:
                    r = sc[(sc.dataset == ds) & (sc.algorithm == alg) & (sc.rho == rho) & (sc.scenario == s)]
                    if r.empty:
                        row += ["--"] * k
                    else:
                        r = r.iloc[0]
                        row += [fmt(r.C, 3), fmt(r.R), fmt(r.Rbar)]
                        if not COMPACT:
                            row += [fmt(r.V, 1), fmt(r.sat, 2)]
                lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table*}"]
    return "\n".join(lines)


def table_b(sc: pd.DataFrame, scenarios: list[str], dataset: str) -> str:
    datasets = [dataset] if dataset != "all" else [d for d in ("ml1m", "lfm1m") if (sc.dataset == d).any()]
    where = "" if dataset == "all" else f" on {dataset.upper()}"
    cols = "l l l " + " ".join(["rrr"] * len(scenarios))
    lines = [r"\begin{table*}[t]", r"\centering",
             rf"\caption{{Tuning sweep of the structural baselines{where} at the "
             r"main-table budget: comprehensibility ($C$), relevance ($R$), and evidence density "
             r"($\bar R$) per configuration. The configuration of the main tables is marked "
             r"with $\dagger$. For FACES the clustering key sets the cluster count, reported as "
             r"the mean number of clusters per anchor in parentheses.}",
             rf"\label{{tab:tuning_sweep_{dataset}}}",
             r"\begin{adjustbox}{max width=\textwidth}",
             rf"\begin{{tabular}}{{@{{}}{cols}@{{}}}}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{\emph{{{SCEN_NAME[s]}}}}}" for s in scenarios) + r" \\"]
    lines.append(" ".join(rf"\cmidrule(lr){{{4+3*i}-{6+3*i}}}" for i in range(len(scenarios))))
    lines.append("Dataset & Algorithm & Configuration & " + " & ".join(
        r"$C\uparrow$ & $R\uparrow$ & $\bar R\uparrow$" for _ in scenarios) + r" \\")
    lines.append(r"\midrule")
    configs = ([("pappas2017", c, "kg_type") for c in CENT] +
               [("mst", c, "kg_type") for c in CENT] +
               [("faces", c, k) for k in KEYS for c in CENT])
    for di, ds in enumerate(datasets):
        if di:
            lines.append(r"\midrule")
        prev = None
        for ci, (alg, cent, key) in enumerate(configs):
            if prev and prev != alg:
                lines.append(r"\cmidrule(lr){2-3}")
            main = cent == "betweenness_approx" and key == "kg_type"
            conf = CENT[cent] if alg != "faces" else f"{KEYS[key]}, {CENT[cent]}"
            if main:
                conf += r"$^\dagger$"
            row = [ds.upper() if ci == 0 else "", NAMES[alg] if prev != alg else "", conf]
            prev = alg
            for s in scenarios:
                r = sc[(sc.dataset == ds) & (sc.algorithm == alg) & (sc.centrality == cent)
                       & (sc.faces_key == key) & (sc.scenario == s)]
                if r.empty:
                    row += ["--"] * 3
                else:
                    r = r.iloc[0]
                    c = fmt(r.C, 3)
                    if alg == "faces" and not pd.isna(r.ncl):
                        c += f" ({r.ncl:.1f})"
                    row += [c, fmt(r.R), fmt(r.Rbar)]
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table*}"]
    return "\n".join(lines)


def main():
    global RBAR_MODE, COMPACT
    p = argparse.ArgumentParser()
    p.add_argument("--compact", action="store_true",
                   help="omit the |V_S| and Sat. columns of the budget table")
    p.add_argument("--dataset", required=True, choices=list(DATASETS.keys()) + ["all"],
                   help="one dataset, or 'all' for merged tables with one row block per dataset")
    p.add_argument("--rbar", choices=["paper", "anchor"], default="paper",
                   help="evidence density: 'paper' = R/(1/C-1) on the means (as in the "
                        "submitted tables); 'anchor' = mean over anchors of R(S)/|E_S| "
                        "(the definition in the text, bounded by the maximum edge weight)")
    args = p.parse_args()
    RBAR_MODE = args.rbar
    COMPACT = args.compact
    ds = args.dataset
    ds_list = list(DATASETS.keys()) if ds == "all" else [ds]
    scenarios = [s for s in SCEN_ORDER if any(s in DATASETS[d]["scenarios"] for d in ds_list)]

    frames = []
    for d in ds_list:
        sweep = pd.read_csv(RESULTS_ROOT / "sweeps" / f"{d}_sweep_per_anchor.csv")
        frames += [load_main(d), sweep]
    df = pd.concat(frames, ignore_index=True)
    df["anchor_id"] = df["anchor_id"].astype(str)
    df["budget"] = meta_col(df, "budget")
    df["n_pool"] = meta_col(df, "n_pool_optional")
    df["n_clusters"] = meta_col(df, "n_clusters")

    # Pool size is a property of the anchor, so fill it into rows written
    # before the logging existed (main files) from any sweep row.
    pool = (df.dropna(subset=["n_pool"])
              .groupby(["scenario", "baseline", "anchor_id"])["n_pool"].first())
    df["n_pool"] = df.set_index(["scenario", "baseline", "anchor_id"]).index.map(pool.to_dict())
    df["saturated"] = (df["n_pool"] <= df["budget"]).astype(float)
    df.loc[df["n_pool"].isna(), "saturated"] = float("nan")

    # Table A: rho = 1 rows come from the main files.  Every rho covers
    # exactly the cells of the main run: sweep rows for cells the main run
    # skipped (budget 0, e.g. ML1M item-group CAFE) are dropped, and re-run
    # 1x rows are ignored.
    main_cells = set(map(tuple, df[df.sweep == "main"][["scenario", "baseline", "algorithm"]]
                         .drop_duplicates().values))
    in_main = df[["scenario", "baseline", "algorithm"]].apply(tuple, axis=1).isin(main_cells)
    dropped = df[(df.sweep == "budget") & ~in_main][["scenario", "baseline"]].drop_duplicates()
    if len(dropped):
        print("[tables] cells absent from the main run, excluded from Table A:",
              sorted(map(tuple, dropped.values)))
    a_src = pd.concat([df[df.sweep == "main"],
                       df[(df.sweep == "budget") & (df.rho != 1.0) & in_main]],
                      ignore_index=True)
    ta = scenario_means(a_src, ["algorithm", "rho"])
    tb = scenario_means(df[df.sweep.isin(["tuning", "main"])], ["algorithm", "centrality", "faces_key"])

    out_dir = RESULTS_ROOT / "sweeps"
    suffix = f"{ds}_rbar-{RBAR_MODE}"
    (out_dir / f"{suffix}_table_budget.tex").write_text(table_a(ta, scenarios, ds))
    (out_dir / f"{suffix}_table_tuning.tex").write_text(table_b(tb, scenarios, ds))
    ta.to_csv(out_dir / f"{suffix}_table_budget.csv", index=False)
    tb.to_csv(out_dir / f"{suffix}_table_tuning.csv", index=False)
    print(table_a(ta, scenarios, ds))
    print()
    print(table_b(tb, scenarios, ds))


if __name__ == "__main__":
    main()
