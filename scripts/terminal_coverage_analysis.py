from __future__ import annotations
import json
import pickle
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd

from config import DATASETS, TOP_K, RESULTS_ROOT
from scripts.terminal_coverage import build_terminals

pd.set_option("display.width", 200)


def records(dataset, scenario, baseline, algo):
    f = RESULTS_ROOT / dataset / scenario / f"{dataset}_{scenario}_{baseline}_{algo}.jsonl"
    if not f.exists():
        return
    with open(f, "r", encoding="utf-8") as fin:
        for line in fin:
            yield json.loads(line)


def block_a():
    rows = []
    for ds, cfg in DATASETS.items():
        cent = pickle.load(open(RESULTS_ROOT / "_centrality_cache" / f"{ds}_degree.pkl", "rb"))
        for sc in cfg["scenarios"]:
            for b in cfg["baselines"]:
                terms = build_terminals(ds, sc, b, TOP_K)
                for r in records(ds, sc, b, "wpcst_cent_degree"):
                    T = terms.get(str(r["anchor_id"]))
                    if not T:
                        continue
                    V = set(map(str, r["solution_nodes"]))
                    for t in set(T):
                        rows.append((ds, sc, "retained" if t in V else "omitted", cent.get(t, np.nan)))
    d = pd.DataFrame(rows, columns=["dataset", "scenario", "status", "centrality"])
    print("\nA1. Degree centrality of the terminals (all terminals)")
    print(d.groupby(["dataset", "scenario"]).centrality.agg(["mean", "median", "count"]).round(4).to_string())
    print("\nA2. Degree centrality of retained vs omitted terminals, WPCST(degree)")
    print(d.groupby(["dataset", "scenario", "status"]).centrality.agg(["mean", "median", "count"]).round(4).to_string())


def path_stats(r):
    c = Counter()
    for p in r["top_k_summary"]:
        for e in p["edges"]:
            c[tuple(e)] += 1
    return c


def block_b_c():
    pa = pd.read_csv(RESULTS_ROOT / "terminal_coverage_per_anchor.csv")
    for ds, cfg in DATASETS.items():
        G = nx.read_graphml(cfg["graphml"], node_type=str)
        w_max = max(float(d.get("weight", 0.0)) for _, _, d in G.edges(data=True))
        share_max = np.mean([float(d.get("weight", 0.0)) == w_max for _, _, d in G.edges(data=True)])

        def base(u, v):
            if G.has_edge(u, v):
                return float(G[u][v].get("weight", 0.0))
            if G.has_edge(v, u):
                return float(G[v][u].get("weight", 0.0))
            return 0.0

        rows = []
        for b in cfg["baselines"]:
            for r in records(ds, "user_centric", b, "pcst"):
                c = path_stats(r)
                if not c:
                    continue
                m = max([w_max] + [base(u, v) * (1 + n / TOP_K) for (u, v), n in c.items()])
                rows.append((b, str(r["anchor_id"]), max(c.values()), round(m / w_max, 2)))
        a = pd.DataFrame(rows, columns=["recommender", "anchor_id", "f_max", "M_over_wmax"])
        x = pa[(pa.dataset == ds) & (pa.scenario == "user_centric") & (pa.algorithm == "pcst")]
        x = x.assign(anchor_id=x.anchor_id.astype(str)).merge(a, on=["recommender", "anchor_id"])
        x["omits"] = x.tc < 1

        print(f"\nB. {ds}: path overlap per recommender (user-centric)")
        print(x.groupby("recommender").agg(anchors=("f_max", "size"), f_max_mean=("f_max", "mean"),
                                           all_K_paths_share_an_edge=("f_max", lambda s: int((s == TOP_K).sum())),
                                           M_over_wmax_mean=("M_over_wmax", "mean")).round(2).to_string())
        print(f"\nC. {ds}: PCST omissions against M / w_max (user-centric). "
              f"w_max = {w_max}, share of edges at w_max = {share_max:.3f}")
        x["M_class"] = np.where(x.M_over_wmax >= 2.0, "M = 2 w_max", "M < 2 w_max")
        print(x.groupby("M_class").agg(anchors=("omits", "size"), anchors_with_omissions=("omits", "sum"),
                                       mean_tc=("tc", "mean")).round(3).to_string())
        print(x[x.M_class == "M = 2 w_max"].groupby("recommender").agg(
            anchors=("omits", "size"), anchors_with_omissions=("omits", "sum")).to_string())


if __name__ == "__main__":
    block_a()
    block_b_c()
