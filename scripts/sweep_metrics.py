from __future__ import annotations
import argparse
from pathlib import Path

import networkx as nx
import pandas as pd

from config import DATASETS, RESULTS_ROOT
from metrics.compute_metrics import load_records, compute_per_anchor

CENTRALITIES = ("betweenness_approx", "pagerank", "degree")
ALGS = ("pappas2017", "mst", "faces")


def parse_sweep_filename(path: Path) -> dict:
    stem = path.stem
    parts = stem.split("_")
    info = {"dataset": parts[0], "scenario": "_".join(parts[1:3]), "baseline": parts[3]}
    rest = "_".join(parts[4:])
    alg = next(a for a in ALGS if rest.startswith(a))
    rest = rest[len(alg):].lstrip("_")
    assert rest.startswith("cent_"), stem
    rest = rest[len("cent_"):]
    cent = next(c for c in CENTRALITIES if rest.startswith(c))
    tag = rest[len(cent):].lstrip("_")
    info.update(algorithm=alg, centrality=cent, tag=tag)
    if tag.startswith("bx"):
        info["sweep"], info["rho"], info["faces_key"] = "budget", float(tag[2:]), "kg_type"
    elif tag.startswith("tune"):
        info["sweep"], info["rho"] = "tuning", 1.0
        info["faces_key"] = tag[len("tune_"):] if tag.startswith("tune_") else "kg_type"
    else:
        info["sweep"], info["rho"], info["faces_key"] = tag, 1.0, "kg_type"
    return info


def item_sets(G: nx.DiGraph) -> tuple[set[str], set[str]]:
    """Same popular-item construction as metrics.compute_metrics.main."""
    all_items: set[str] = set()
    deg: dict[str, int] = {}
    for n, data in G.nodes(data=True):
        if data.get("type") == "item":
            sn = str(n)
            all_items.add(sn)
            users = sum(1 for nb in G.predecessors(n) if str(nb).startswith("u"))
            users += sum(1 for nb in G.successors(n) if str(nb).startswith("u"))
            deg[sn] = users
    popular: set[str] = set()
    if deg:
        ranked = sorted(deg.items(), key=lambda kv: kv[1], reverse=True)
        popular = {i for i, _ in ranked[: max(1, len(ranked) // 4)]}
    return all_items, popular


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    args = p.parse_args()

    cfg = DATASETS[args.dataset]
    G = nx.read_graphml(cfg["graphml"], node_type=str)
    if not isinstance(G, nx.DiGraph):
        G = nx.DiGraph(G)
    all_items, popular = item_sets(G)

    sweep_root = RESULTS_ROOT / "sweeps" / args.dataset
    out_dir = RESULTS_ROOT / "sweeps" / "per_anchor"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for jp in sorted(sweep_root.glob("*/*.jsonl")):
        info = parse_sweep_filename(jp)
        recs = load_records(jp)
        if not recs:
            continue
        df = compute_per_anchor(recs, G, dataset=info["dataset"], scenario=info["scenario"],
                                baseline=info["baseline"], algorithm=info["algorithm"],
                                centrality=info["centrality"], all_items=all_items,
                                popular_items=popular)
        if df.empty:
            continue
        for k in ("tag", "sweep", "rho", "faces_key"):
            df[k] = info[k]
        df.to_parquet(out_dir / f"{jp.stem}.parquet", index=False)
        frames.append(df)
        print(f"[sweep-metrics] {jp.name}: {len(df)} anchors")
    if not frames:
        print("[sweep-metrics] nothing found under", sweep_root)
        return
    big = pd.concat(frames, ignore_index=True)
    out = RESULTS_ROOT / "sweeps" / f"{args.dataset}_sweep_per_anchor.csv"
    big.to_csv(out, index=False)
    print(f"[sweep-metrics] wrote {out} ({len(big)} rows)")


if __name__ == "__main__":
    main()
