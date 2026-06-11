"""
Path-aware weight (lambda) ablation for ST and WPCST.

Sweeps the reweighting strength lambda of Eq. (2),

    w~(e) = w(e) * (1 + lambda * freq_P(e) / |P|),

for the ST and WPCST algorithms, holding K, gamma, and the centrality fixed.
This reproduces -- and extends to WPCST -- the lambda-ablation table in the
paper (Table II, which was ST-only and reported relevance on a different
scale than the headline tables).

Why a dedicated runner
----------------------
The per-scenario runners (run_user_centric.py, ...) deliberately hardcode
`lam = config.LAMBDA` and are not meant to sweep it.  This runner imports
their request builders unchanged, but dispatches each cell with a per-cell
lambda and then scores the outputs with the *same* metric functions the
main pipeline uses (`metrics.compute_metrics`).  Consequently:

  * Comprehensibility is C(S) = 1 / (|E_S| + 1)               (identical to main).
  * Relevance is the un-normalised total R(S) = sum_e w_M(e)  (identical to main;
    see `metrics.compute_metrics.m_relevance`).  The numbers are therefore on
    the same absolute scale as Tables III-VI, not the ~0.1 scale of the old
    Table II.  Evidence Density Rbar(S) = R(S) / |E_S| is also reported.

Cost reuse
----------
The graph, the PCST index, the undirected projection, and the centrality map
are built once and reused across every (scenario x baseline x algorithm x
lambda) cell.  Request lists depend only on (scenario, baseline, K) -- not on
lambda or the algorithm -- so they are built once per (scenario, baseline)
and reused across the lambda x algorithm grid.

Outputs (under results/_lambda_ablation/<dataset>/, intentionally *outside*
results/<dataset>/ so that `compute_metrics --dataset` does not glob the
sweep files into the main summary):

  * <ds>_<scenario>_<baseline>_<alg>_lam<tag>.jsonl   per-cell per-anchor records
  * <ds>_<scenario>_<baseline>_<alg>_lam<tag>.parquet per-cell per-anchor metrics
  * lambda_ablation_long.csv     tidy (scenario, baseline, alg, lam) -> means
  * lambda_ablation_table.csv    paper Table-II layout, mean over baselines
  * lambda_ablation_table.tex    LaTeX for the above (drop-in replacement for Table II)

Usage
-----
    python -m runners.run_lambda_ablation --dataset ml1m
    python -m runners.run_lambda_ablation --dataset ml1m \
        --lambdas 0.01,1,100 --algorithms st,wpcst --baselines pgpr,cafe
    # quick wiring check on a few anchors:
    python -m runners.run_lambda_ablation --dataset ml1m --limit 5 --workers 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from config import (
    DATASETS, TOP_K, GAMMA, RESULTS_ROOT, DEFAULT_CENTRALITY, NUM_WORKERS,
)
from algorithms import AnchorRequest, PcstIndex
from centralities import get_or_compute
from sampling.make_samples import load_users, load_items
from runners._core import (
    load_graph, load_user_recs, load_item_paths,
    make_dispatcher, execute, write_results,
)
from runners.run_user_centric import build_requests as build_user_centric
from runners.run_item_centric import build_requests as build_item_centric
from runners.run_user_group import build_group_requests as build_user_group
from runners.run_item_group import build_group_requests as build_item_group
from metrics.compute_metrics import compute_per_anchor


# ---------------------------------------------------------------------------
# Scenario specification
# ---------------------------------------------------------------------------
# Each scenario differs only in: the anchor kind, whether the solver is rooted
# at the anchor, whether it needs the item-paths file, and whether it is small
# enough to force serial execution (the group scenarios have two anchors, so
# spawning a pool is pure overhead).

_SCENARIOS: dict[str, dict] = {
    "user_centric": {"anchor_kind": "user",       "use_root": True,
                     "needs_item_paths": False,    "force_serial": False},
    "item_centric": {"anchor_kind": "item",       "use_root": True,
                     "needs_item_paths": True,     "force_serial": False},
    "user_group":   {"anchor_kind": "user_group", "use_root": False,
                     "needs_item_paths": False,    "force_serial": True},
    "item_group":   {"anchor_kind": "item_group", "use_root": False,
                     "needs_item_paths": True,     "force_serial": True},
}

# Display order for the paper-style table.
_SCENARIO_ORDER = ["user_centric", "item_centric", "user_group", "item_group"]
_ALG_ORDER = ["st", "wpcst"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lam_tag(lam: float) -> str:
    """Filesystem-safe tag for a lambda value: 0.01 -> 'lam0p01', 100 -> 'lam100'."""
    return "lam" + ("%g" % lam).replace(".", "p").replace("-", "m")


def _build_requests(dataset: str, scenario: str, baseline: str,
                    K: int) -> list[AnchorRequest]:
    """Build the per-scenario AnchorRequest list, reusing the scenario runners."""
    if scenario == "user_centric":
        recs = load_user_recs(dataset, baseline)
        return build_user_centric(load_users(dataset), recs, K=K)
    if scenario == "item_centric":
        paths = load_item_paths(dataset, baseline)
        return build_item_centric(load_items(dataset), paths, K=K)
    if scenario == "user_group":
        recs = load_user_recs(dataset, baseline)
        return build_user_group(load_users(dataset), recs, K=K)
    if scenario == "item_group":
        paths = load_item_paths(dataset, baseline)
        return build_item_group(load_items(dataset), paths, K=K)
    raise ValueError(f"Unknown scenario: {scenario!r}")


def _item_node_set(G: nx.DiGraph) -> set[str]:
    """Item-typed node IDs, matching metrics.compute_metrics' convention."""
    return {str(n) for n, d in G.nodes(data=True) if d.get("type") == "item"}


# ---------------------------------------------------------------------------
# Core sweep
# ---------------------------------------------------------------------------

def run_ablation(dataset: str, *,
                 lambdas: list[float],
                 algorithms: list[str],
                 baselines: list[str],
                 scenarios: list[str],
                 K: int = TOP_K,
                 centrality_name: str = DEFAULT_CENTRALITY,
                 n_workers: int | None = None,
                 limit: int | None = None,
                 out_dir: Path | None = None) -> pd.DataFrame:
    """Run the full lambda sweep and write all artifacts.  Returns the tidy
    long DataFrame (one row per (scenario, baseline, algorithm, lambda))."""
    if n_workers is None:
        n_workers = NUM_WORKERS

    cfg = DATASETS[dataset]
    # Intersect requested scopes with what the dataset actually supports.
    scenarios = [s for s in scenarios if s in cfg["scenarios"]]
    baselines = [b for b in baselines if b in cfg["baselines"]]
    algorithms = [a for a in algorithms if a in ("st", "wpcst")]
    if not (scenarios and baselines and algorithms):
        raise ValueError(
            f"Nothing to run for {dataset}: scenarios={scenarios}, "
            f"baselines={baselines}, algorithms={algorithms}.")

    out_dir = out_dir or (RESULTS_ROOT / "_lambda_ablation" / dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    # One-time, shared setup.
    G = load_graph(dataset)
    all_items = _item_node_set(G)
    need_solver = "wpcst" in algorithms
    index = PcstIndex(G) if need_solver else None
    G_und = G.to_undirected(as_view=False) if need_solver else None
    centrality = None
    if need_solver:
        cache = RESULTS_ROOT / "_centrality_cache" / f"{dataset}_{centrality_name}.pkl"
        centrality = get_or_compute(G, centrality_name, cache_path=cache)

    rows: list[dict] = []

    for scenario in scenarios:
        spec = _SCENARIOS[scenario]
        if spec["needs_item_paths"] and cfg["item_paths_template"] is None:
            print(f"[lambda-ablation] skipping {scenario} on {dataset} "
                  f"(no item-paths file).")
            continue

        for baseline in baselines:
            # Requests depend on (scenario, baseline, K) only -- build once and
            # reuse across the lambda x algorithm grid.
            requests = _build_requests(dataset, scenario, baseline, K)
            if limit is not None:
                requests = requests[:limit]
            print(f"[lambda-ablation] {dataset}/{scenario}/{baseline}: "
                  f"{len(requests)} requests")
            if not requests:
                continue

            cell_workers = 1 if spec["force_serial"] else n_workers

            for alg in algorithms:
                use_index = index if alg == "wpcst" else None
                use_cent = centrality if alg == "wpcst" else None
                use_gund = G_und if alg == "wpcst" else None

                for lam in lambdas:
                    dispatch = make_dispatcher(
                        alg, G=G, index=use_index, centrality=use_cent,
                        G_und=use_gund, K=K, lam=lam, gamma=GAMMA,
                        anchor_kind=spec["anchor_kind"], use_root=spec["use_root"],
                        dataset=dataset,
                        dterm_cache_dir=RESULTS_ROOT / "_dterm_cache",
                    )
                    results = execute(requests, dispatch,
                                      anchor_kind=spec["anchor_kind"],
                                      n_workers=cell_workers)

                    tag = _lam_tag(lam)
                    stem = f"{dataset}_{scenario}_{baseline}_{alg}_{tag}"
                    write_results(out_dir / f"{stem}.jsonl", results)

                    # Score with the *main-pipeline* metric definitions so the
                    # numbers match the absolute scale of the headline tables.
                    df = compute_per_anchor(
                        results, G,
                        dataset=dataset, scenario=scenario,
                        baseline=baseline, algorithm=alg,
                        centrality=centrality_name if alg == "wpcst" else None,
                        all_items=all_items, popular_items=None,
                    )
                    df["lambda"] = lam
                    df.to_parquet(out_dir / f"{stem}.parquet", index=False)

                    n = len(df)
                    c_mean = float(df["comprehensibility"].mean()) if n else float("nan")
                    r_mean = float(df["relevance"].mean()) if n else float("nan")
                    e_mean = float(df["num_edges"].mean()) if n else float("nan")
                    # Evidence density: mean over anchors of (R / |E|).
                    with np.errstate(divide="ignore", invalid="ignore"):
                        rbar = (df["relevance"] / df["num_edges"].replace(0, np.nan))
                    rbar_mean = float(rbar.mean()) if n else float("nan")

                    rows.append({
                        "dataset": dataset, "scenario": scenario,
                        "baseline": baseline, "algorithm": alg,
                        "lambda": lam, "n_anchors": n,
                        "comprehensibility": c_mean,
                        "relevance": r_mean,
                        "evidence_density": rbar_mean,
                        "num_edges": e_mean,
                    })
                    print(f"[lambda-ablation]   {alg:<5} lam={lam:<6g} "
                          f"C={c_mean:.4f} R={r_mean:.3f} "
                          f"Rbar={rbar_mean:.3f} |E|={e_mean:.1f}")

    long_df = pd.DataFrame(rows)
    long_path = out_dir / "lambda_ablation_long.csv"
    long_df.to_csv(long_path, index=False)
    print(f"[lambda-ablation] wrote {long_path}")

    table_df = _mean_over_baselines(long_df)
    table_csv = out_dir / "lambda_ablation_table.csv"
    table_df.to_csv(table_csv, index=False)
    print(f"[lambda-ablation] wrote {table_csv}")

    tex = _to_latex(table_df, lambdas=lambdas, baselines=baselines,
                    dataset=dataset, centrality_name=centrality_name, K=K)
    table_tex = out_dir / "lambda_ablation_table.tex"
    table_tex.write_text(tex, encoding="utf-8")
    print(f"[lambda-ablation] wrote {table_tex}")

    return long_df


# ---------------------------------------------------------------------------
# Aggregation: mean over recommenders (paper Table-II layout)
# ---------------------------------------------------------------------------

def _mean_over_baselines(long_df: pd.DataFrame) -> pd.DataFrame:
    """Average the per-baseline cell means with equal weight per baseline,
    matching the paper's 'mean over PGPR and CAFE'.  Returns one row per
    (scenario, algorithm, lambda)."""
    if long_df.empty:
        return long_df.copy()
    g = (long_df
         .groupby(["scenario", "algorithm", "lambda"], as_index=False)
         .agg(comprehensibility=("comprehensibility", "mean"),
              relevance=("relevance", "mean"),
              evidence_density=("evidence_density", "mean"),
              num_edges=("num_edges", "mean"),
              n_baselines=("baseline", "nunique")))
    # Stable ordering for the table.
    g["_s"] = g["scenario"].map({s: i for i, s in enumerate(_SCENARIO_ORDER)}).fillna(99)
    g["_a"] = g["algorithm"].map({a: i for i, a in enumerate(_ALG_ORDER)}).fillna(99)
    g = g.sort_values(["_s", "_a", "lambda"]).drop(columns=["_s", "_a"])
    return g.reset_index(drop=True)


# ---------------------------------------------------------------------------
# LaTeX emission
# ---------------------------------------------------------------------------

_SCENARIO_LABEL = {
    "user_centric": "User-centric",
    "item_centric": "Item-centric",
    "user_group": "User-group",
    "item_group": "Item-group",
}
_ALG_LABEL = {"st": "ST", "wpcst": "WPCST"}


def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x))):
        return "--"
    ax = abs(x)
    if ax != 0 and (ax < 1e-2 or ax >= 1e4):
        return f"{x:.2e}"
    if ax < 1:
        return f"{x:.4f}"
    if ax < 100:
        return f"{x:.2f}"
    return f"{x:.1f}"


def _to_latex(table_df: pd.DataFrame, *, lambdas: list[float],
              baselines: list[str], dataset: str,
              centrality_name: str, K: int) -> str:
    """Render a Table-II-style LaTeX table: rows = (scenario, algorithm),
    columns grouped by lambda, each with C and R.  Mean over `baselines`."""
    lams = list(lambdas)
    base_str = ", ".join(b.upper() for b in baselines)
    ds_label = {"ml1m": "ML1M", "lfm1m": "LFM1M"}.get(dataset, dataset.upper())

    col_spec = "ll" + "".join("cc" for _ in lams)
    lam_heads = " & ".join(
        rf"\multicolumn{{2}}{{c}}{{$\lambda = {('%g' % l)}$}}" for l in lams)
    cr_heads = " & ".join("$C$ & $R$" for _ in lams)

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    cap = (rf"Ablation of the path-aware weight $\lambda$ on ST and WPCST"
           rf"(\textsc{{{centrality_name}}}) at $k={K}$ on {ds_label} "
           rf"(mean over {base_str}). $C$: comprehensibility "
           rf"$=1/(|E_S|+1)$. $R$: relevance $=\sum_{{e\in E_S}} w_M(e)$, the "
           rf"same total used in Tables~\ref{{tab:uc}}--\ref{{tab:ig}} "
           rf"(not per-edge).")
    lines.append(rf"\caption{{{cap}}}")
    lines.append(r"\label{tab:lambda-ablation}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(rf"Scenario & Alg. & {lam_heads} \\")
    lines.append(rf" &  & {cr_heads} \\")
    lines.append(r"\midrule")

    for scenario in _SCENARIO_ORDER:
        sub_s = table_df[table_df["scenario"] == scenario]
        if sub_s.empty:
            continue
        first_alg_in_block = True
        for alg in _ALG_ORDER:
            sub = sub_s[sub_s["algorithm"] == alg]
            if sub.empty:
                continue
            cells = []
            for l in lams:
                row = sub[np.isclose(sub["lambda"], l)]
                if row.empty:
                    cells += ["--", "--"]
                else:
                    cells.append(_fmt(float(row["comprehensibility"].iloc[0])))
                    cells.append(_fmt(float(row["relevance"].iloc[0])))
            scen_cell = _SCENARIO_LABEL.get(scenario, scenario) if first_alg_in_block else ""
            lines.append(rf"{scen_cell} & {_ALG_LABEL.get(alg, alg)} & "
                         + " & ".join(cells) + r" \\")
            first_alg_in_block = False
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip() != ""]


def _parse_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip() != ""]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    p.add_argument("--lambdas", default="0.01,1,100",
                   help="Comma-separated lambda values (default: 0.01,1,100).")
    p.add_argument("--algorithms", default="st,wpcst",
                   help="Subset of {st,wpcst} (default: st,wpcst).")
    p.add_argument("--baselines", default="pgpr,cafe",
                   help="Recommenders to average over (default: pgpr,cafe).")
    p.add_argument("--scenarios", default=",".join(_SCENARIO_ORDER),
                   help="Subset of scenarios (default: all four).")
    p.add_argument("--K", type=int, default=TOP_K)
    p.add_argument("--centrality", default=DEFAULT_CENTRALITY,
                   help="Centrality for WPCST (default: degree).")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap anchors per cell (quick wiring checks).")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    run_ablation(
        args.dataset,
        lambdas=_parse_floats(args.lambdas),
        algorithms=_parse_list(args.algorithms),
        baselines=_parse_list(args.baselines),
        scenarios=_parse_list(args.scenarios),
        K=args.K,
        centrality_name=args.centrality,
        n_workers=args.workers,
        limit=args.limit,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )


if __name__ == "__main__":
    main()
