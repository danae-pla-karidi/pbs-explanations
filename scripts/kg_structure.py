from __future__ import annotations
import argparse
import collections
import pickle

import networkx as nx
import numpy as np

from config import DATASETS, RESULTS_ROOT

TYPES = ("user", "item", "external")
CENTS = ("degree", "pagerank", "betweenness_approx")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    args = p.parse_args()
    ds = args.dataset

    G = nx.read_graphml(DATASETS[ds]["graphml"], node_type=str)
    typ = {n: d.get("type") for n, d in G.nodes(data=True)}
    deg = dict(G.degree())
    print(f"== {ds}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print("degree per type (median / p90 / p99 / max):")
    for t in TYPES:
        d = np.array([deg[n] for n in G if typ[n] == t])
        print(f"  {t:9s} n={len(d):6d}  {np.median(d):7.1f} / {np.percentile(d, 90):7.1f} / "
              f"{np.percentile(d, 99):7.1f} / {d.max():6d}")

    rows = []
    for cname in CENTS:
        path = RESULTS_ROOT / "_centrality_cache" / f"{ds}_{cname}.pkl"
        if not path.exists():
            continue
        c = pickle.load(open(path, "rb"))
        if isinstance(c, dict) and "scores" in c:
            c = c["scores"]
        argmax_type = typ[max(c, key=c.get)]
        top = sorted(c, key=lambda n: -c[n])[:100]
        comp = collections.Counter(typ.get(n) for n in top)
        med = {t: float(np.median([c.get(n, 0.0) for n in G if typ[n] == t])) for t in TYPES}
        rows.append((cname, argmax_type, comp, med))
        print(f"{cname:18s} max node type: {argmax_type:9s} top-100: "
              + ", ".join(f"{t} {comp.get(t, 0) / 100:.2f}" for t in TYPES)
              + "  median c: " + ", ".join(f"{t} {med[t]:.4f}" for t in TYPES))

    # LaTeX rows: dataset, centrality, max type, top-100 item share, median c user, median c item
    print("\nLaTeX rows:")
    for cname, amt, comp, med in rows:
        label = {"degree": "degree", "pagerank": "PageRank", "betweenness_approx": "betweenness"}[cname]
        print(f"{ds.upper()} & {label} & {amt} & {comp.get('item', 0) / 100:.2f} & "
              f"{comp.get('user', 0) / 100:.2f} & {comp.get('external', 0) / 100:.2f} & "
              f"{med['user']:.3f} & {med['item']:.3f} \\\\")


if __name__ == "__main__":
    main()
