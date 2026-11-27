"""
Generate one static knowledge graph per (beta1, beta2) combination.

Mirrors create_initial_kg.py exactly:

    w(u, i) = beta1 * (rating / 5) + beta2 * exp(-gamma * (t0 - t_ui))
    item-external edges have weight 0.

For each recency share s in --shares it sets (beta1, beta2) = (1 - s, s) and
writes

    <out-dir>/<prefix>_b1_<beta1>_b2_<beta2>.graphml

The graph structure (nodes, item-external edges, recency per interaction) is
built once and reused; only the user-item edge weights are rewritten per beta,
so memory stays flat and generation is fast.

Inputs expected in --data-dir (same files create_initial_kg.py reads):
    i2kg_map.txt   eid<TAB>pid<TAB>name<TAB>entity
    e_map.txt      eid<TAB>name<TAB>entity
    ratings.txt    uid<TAB>pid<TAB>rating(or feedback)<TAB>timestamp
    users.txt      uid<TAB>gender<TAB>age
    kg_final.txt   entity_head<TAB>relation<TAB>entity_tail

Usage
-----
    python -m runners.generate_ablation_kgs --dataset ml1m \
        --data-dir data/ml1m --out-dir ablation_data \
        --shares 0.25,0.5,0.75,1.0
    python -m runners.generate_ablation_kgs --dataset lfm1m \
        --data-dir data/lfm1m --out-dir ablation_data \
        --shares 0.25,0.5,0.75,1.0 --prefix lmfm_kg_static
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path

import pandas as pd
import networkx as nx


def _read(data_dir: Path, name: str, names: list[str]) -> pd.DataFrame:
    df = pd.read_csv(data_dir / name, sep="\t", skiprows=1, names=names,
                     on_bad_lines="skip")
    return df


def build_base_graph(data_dir: Path, gamma: float, t0: float):
    """Build the graph once with user/item/external nodes, item-external edges
    (weight 0), and user-item edges (weight 0).  Return (G, ui) where ui is a
    list of (user_node, item_node, rating, recency) for fast per-beta reweight."""
    i2kg = _read(data_dir, "i2kg_map.txt", ["eid", "pid", "name", "entity"])
    e_map = _read(data_dir, "e_map.txt", ["eid", "name", "entity"])
    ratings = _read(data_dir, "ratings.txt", ["uid", "pid", "rating", "timestamp"])
    users = _read(data_dir, "users.txt", ["uid", "gender", "age"])
    kg_final = _read(data_dir, "kg_final.txt", ["entity_head", "relation", "entity_tail"])

    pid2eid = {int(p): int(e) for e, p in zip(i2kg["eid"], i2kg["pid"])}

    G = nx.Graph()
    for r in users.itertuples(index=False):
        G.add_node(f"u{int(r.uid)}", type="user", gender=str(r.gender), age=str(r.age))
    for r in i2kg.itertuples(index=False):
        G.add_node(int(r.eid), type="item", name=str(r.name), entity=str(r.entity))
    for eid in e_map["eid"].astype(int):
        if eid not in G:
            G.add_node(int(eid), type="external")

    ui = []
    miss = 0
    for r in ratings.itertuples(index=False):
        eid = pid2eid.get(int(r.pid))
        if eid is None:
            miss += 1
            continue
        un = f"u{int(r.uid)}"
        rec = math.exp(-gamma * (t0 - float(r.timestamp)))
        G.add_edge(un, eid, weight=0.0)
        ui.append((un, eid, float(r.rating), rec))

    for r in kg_final.itertuples(index=False):
        G.add_edge(int(r.entity_head), int(r.entity_tail), weight=0.0)

    print(f"[gen] base graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{len(ui)} user-item interactions ({miss} unmapped pids skipped)")
    return G, ui


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--out-dir", default=Path("ablation_data"), type=Path)
    p.add_argument("--prefix", default=None,
                   help="filename prefix; default '<dataset>_kg_static'")
    p.add_argument("--shares", default="0.25,0.5,0.75,1.0",
                   help="recency shares s; (beta1,beta2)=(1-s,s)")
    p.add_argument("--rating-norm", type=float, default=5.0)
    p.add_argument("--gamma", type=float, default=1e-9,
                   help="recency decay rate (create_initial_kg.py used 1e-9)")
    p.add_argument("--t0", type=float, default=None,
                   help="reference timestamp; default = now (builder convention)")
    args = p.parse_args()

    prefix = args.prefix or f"{args.dataset}_kg_static"
    t0 = args.t0 if args.t0 is not None else float(datetime.now().timestamp())
    shares = [float(x) for x in args.shares.split(",")]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    G, ui = build_base_graph(args.data_dir, args.gamma, t0)
    print(f"[gen] gamma={args.gamma:.3e}, t0={t0:.0f}, rating_norm={args.rating_norm:g}")

    for s in shares:
        b1, b2 = round(1.0 - s, 6), round(s, 6)
        for (un, eid, rating, rec) in ui:
            G[un][eid]["weight"] = b1 * (rating / args.rating_norm) + b2 * rec
        out = args.out_dir / f"{prefix}_b1_{b1:g}_b2_{b2:g}.graphml"
        nx.write_graphml(G, out)
        print(f"[gen] wrote {out}")


if __name__ == "__main__":
    main()
