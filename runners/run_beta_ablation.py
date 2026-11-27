"""
Recency-weighting (beta1/beta2) ablation for ST, PCST, and WPCST.

Sweeps the edge-weight tradeoff of Eq. (1),

    w_M(u, i) = beta1 * r_{u,i} + beta2 * exp(-delta * (t0 - t_{u,i})),

and reports comprehensibility (C), relevance (R), evidence density (Rbar),
and the quality metrics (A, D, F) on the user-centric and item-centric
scenarios.  We sweep the *recency share* s with (beta1, beta2) = (1 - s, s);
s = 0 is the paper default (beta1 = 1, beta2 = 0).  Holding beta1 + beta2 = 1
keeps the relevance scale comparable across the sweep.

Only ST, PCST, and WPCST are swept: they optimize over w_M, so their summaries
change with (beta1, beta2).  The structural baselines select edges from
recommender paths / centrality / node type, so their size is invariant to beta
and their relevance only rescales.  Pass --include-naive to add NaiveUnion as a
beta-independent reference line.

The KG is (re)weighted inside this script
------------------------------------------
The committed graphml stores only the final edge `weight` and has no rating or
timestamp, so the recency term cannot be recomputed from it.  This runner reads
the RAW ratings file (uid, pid, rating, timestamp) and rebuilds every user-item
weight in memory for each (beta1, beta2).

The ratings ids are mapped to graph node ids automatically:
  * user node = "u<uid>"  (verified: exact degree match on ML1M and LFM1M);
  * item node: the ratings `pid` (MovieLens movieId / LastFM trackId) is mapped
    to the graph's 0-indexed item node by matching, for each id, the set of
    users that interacted with it.  On both datasets this is a collision-free
    bijection, so no external mapping file is needed.  The map is cached under
    results/_beta_ablation/<dataset>/item_id_map.json.

For LFM1M the feedback column is implicit (all 1), so the rating-magnitude term
is constant and only the recency term varies across the sweep.

Mirrors run_lambda_ablation.py: reuses the per-scenario request builders, the
repo solvers, and metrics.compute_per_anchor, so C and R match the absolute
scale of the headline tables.

Outputs (under results/_beta_ablation/<dataset>/):
  <ds>_<scenario>_<baseline>_<alg>_b<tag>.jsonl     per-cell per-anchor records
  beta_ablation_long.csv                            tidy means, one row per cell

Usage
-----
    python -m runners.run_beta_ablation --dataset lfm1m \
        --ratings data/lfm1m/ratings.txt --i2kg-map data/lfm1m/i2kg_map.txt
    python -m runners.run_beta_ablation --dataset ml1m \
        --ratings data/ml1m/ratings.txt --i2kg-map data/ml1m/i2kg_map.txt
    # quick wiring check:
    python -m runners.run_beta_ablation --dataset ml1m --ratings ... \
        --i2kg-map ... --shares 0 --limit 5 --workers 1
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from config import DATASETS, TOP_K, GAMMA, LAMBDA, RESULTS_ROOT, DEFAULT_CENTRALITY, NUM_WORKERS
from algorithms import PcstIndex
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
from metrics.compute_metrics import compute_per_anchor, load_records


_SCENARIOS = {
    "user_centric": {"anchor_kind": "user", "use_root": True,
                     "needs_item_paths": False, "force_serial": False},
    "item_centric": {"anchor_kind": "item", "use_root": True,
                     "needs_item_paths": True, "force_serial": False},
    "user_group":   {"anchor_kind": "user_group", "use_root": False,
                     "needs_item_paths": False, "force_serial": True},
    "item_group":   {"anchor_kind": "item_group", "use_root": False,
                     "needs_item_paths": True, "force_serial": True},
}
_SWEEPABLE = ["st", "pcst", "wpcst"]


def _beta_tag(beta1: float, beta2: float) -> str:
    return f"b{beta1:.2f}_{beta2:.2f}".replace(".", "p")


def _item_node_set(G: nx.DiGraph) -> set[str]:
    return {str(n) for n, d in G.nodes(data=True) if d.get("type") == "item"}


def _user_node_set(G: nx.DiGraph) -> set[str]:
    return {str(n) for n, d in G.nodes(data=True) if d.get("type") == "user"}


def _build_requests(dataset, scenario, baseline, K):
    if scenario == "user_centric":
        return build_user_centric(load_users(dataset), load_user_recs(dataset, baseline), K=K)
    if scenario == "item_centric":
        return build_item_centric(load_items(dataset), load_item_paths(dataset, baseline), K=K)
    if scenario == "user_group":
        return build_user_group(load_users(dataset), load_user_recs(dataset, baseline), K=K)
    if scenario == "item_group":
        return build_item_group(load_items(dataset), load_item_paths(dataset, baseline), K=K)
    raise ValueError(f"Unknown scenario: {scenario!r}")


# ---------------------------------------------------------------------------
# Raw ratings + id mapping
# ---------------------------------------------------------------------------
def _read_ratings(path: Path):
    """Yield (uid, pid, rating, timestamp). Header auto-skipped; rating may be
    absent/implicit (LFM1M feedback), in which case rating = 1.0."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    if not text:
        return
    sep = "\t" if "\t" in text[0] else ("::" if "::" in text[0] else ",")
    start = 1 if not text[0].split(sep)[0].strip().lstrip("-").isdigit() else 0
    for line in text[start:]:
        f = line.split(sep)
        if len(f) < 4:
            continue
        uid, pid = f[0].strip(), f[1].strip()
        try:
            rating = float(f[2])
        except ValueError:
            rating = 1.0
        yield uid, pid, rating, float(f[3])


def load_i2kg_map(path: Path) -> dict[str, str]:
    """Authoritative pid -> item-node (eid) map from i2kg_map.txt
    (header: eid<TAB>pid<TAB>name<TAB>entity). The graph item node id is the
    eid (see create_initial_kg.py)."""
    df = pd.read_csv(path, sep="\t")
    return {str(int(p)): str(int(e)) for e, p in zip(df["eid"], df["pid"])}


def build_item_id_map(G: nx.DiGraph, ratings_path: Path, cache: Path) -> dict[str, str]:
    """Map raw `pid` -> graph item-node id by matching the set of users that
    interacted with each.  Cached to `cache`.  Falls back to a (uid, rating)
    fingerprint to break the (rare) user-set ties."""
    if cache.exists():
        return json.loads(cache.read_text())

    users = _user_node_set(G)
    items = _item_node_set(G)
    # graph side: per item-node, set of uids (and (uid, round(weight)) tiebreak)
    item_users: dict[str, set] = defaultdict(set)
    item_uw: dict[str, set] = defaultdict(set)
    for u, v, d in G.edges(data=True):
        if u in users and v in items:
            uu, ii = u, v
        elif v in users and u in items:
            uu, ii = v, u
        else:
            continue
        uid = uu[1:]
        item_users[ii].add(uid)
        item_uw[ii].add((uid, int(round(float(d.get("weight", 0.0))))))

    # ratings side
    pid_users: dict[str, set] = defaultdict(set)
    pid_ur: dict[str, set] = defaultdict(set)
    for uid, pid, rating, _ in _read_ratings(ratings_path):
        pid_users[pid].add(uid)
        pid_ur[pid].add((uid, int(round(rating))))

    fp_item = defaultdict(list)
    for it, us in item_users.items():
        fp_item[frozenset(us)].append(it)
    fp_pid = defaultdict(list)
    for pid, us in pid_users.items():
        fp_pid[frozenset(us)].append(pid)

    mapping: dict[str, str] = {}
    for fp, pids in fp_pid.items():
        its = fp_item.get(fp, [])
        if len(pids) == 1 and len(its) == 1:
            mapping[pids[0]] = its[0]

    # tie-break leftovers with the (uid, rating) fingerprint
    fp2_item = {frozenset(s): it for it, s in item_uw.items()}
    for pid in pid_users:
        if pid not in mapping:
            it = fp2_item.get(frozenset(pid_ur[pid]))
            if it is not None:
                mapping[pid] = it

    n_pid = len(pid_users)
    if len(mapping) != n_pid or len(set(mapping.values())) != len(mapping):
        raise RuntimeError(
            f"item id map is not a clean bijection: mapped {len(mapping)}/{n_pid}, "
            f"{len(set(mapping.values()))} distinct targets. Provide a mapping file.")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(mapping))
    print(f"[beta] built item id map: {len(mapping)} items (cached -> {cache})")
    return mapping


def build_edge_table(ratings_path: Path, item_map: dict[str, str]):
    """Return {(user_node, item_node): (rating, timestamp)} and t0 (max ts)."""
    table: dict[tuple[str, str], tuple[float, float]] = {}
    t_max = 0.0
    for uid, pid, rating, ts in _read_ratings(ratings_path):
        item = item_map.get(pid)
        if item is None:
            continue
        table[(f"u{uid}", item)] = (rating, ts)
        if ts > t_max:
            t_max = ts
    return table, t_max


def apply_weights(G: nx.DiGraph, beta1: float, beta2: float, edge_table,
                  delta: float, t0: float, rating_norm: float) -> int:
    """Overwrite user-item weights with the builder's formula
    w_M = beta1 * (rating / rating_norm) + beta2 * exp(-delta * (t0 - t));
    refresh w_base.  rating_norm = 5 reproduces create_initial_kg.py exactly
    (so (beta1, beta2) = (1, 0) matches the committed graph: ML1M rating/5,
    LFM1M feedback/5 = 0.2)."""
    n = 0
    for (user, item), (rating, ts) in edge_table.items():
        w = beta1 * (rating / rating_norm) + beta2 * math.exp(-delta * (t0 - ts))
        for a, b in ((user, item), (item, user)):
            if G.has_edge(a, b):
                G[a][b]["weight"] = w
                G[a][b]["w_base"] = w
                n += 1
    return n


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def _load_beta_graph(template, beta1, beta2):
    """Load a pre-built per-beta graphml from disk and refresh w_base.

    Mirrors runners._core.load_graph: the graphml is undirected, so convert to
    DiGraph by adding one directed edge per undirected edge (not both), to match
    the committed-graph pipeline exactly."""
    path = Path(template.format(b1=f"{beta1:g}", b2=f"{beta2:g}",
                                beta1=f"{beta1:g}", beta2=f"{beta2:g}"))
    if not path.exists():
        raise FileNotFoundError(
            f"Per-beta graph not found: {path}\n"
            f"Generate it first with runners.generate_ablation_kgs.")
    G0 = nx.read_graphml(path, node_type=str)
    if not isinstance(G0, nx.DiGraph):
        G = nx.DiGraph()
        for u, v, data in G0.edges(data=True):
            G.add_edge(str(u), str(v), **data)
    else:
        G = G0
    for u, v in G.edges():
        G[u][v]["w_base"] = float(G[u][v].get("weight", 0.0))
    print(f"[beta] loaded {path.name} ({G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges)")
    return G


_CSV_COLS = ["dataset", "scenario", "baseline", "algorithm", "beta1", "beta2",
             "recency_share", "n_anchors", "comprehensibility", "relevance",
             "evidence_density", "actionability", "diversity", "faithfulness",
             "num_edges"]


def _append_row(csv_path, row):
    """Append one cell's row to the long CSV immediately (crash-safe)."""
    import csv
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in _CSV_COLS})
        f.flush()


def run_ablation(dataset, *, shares, algorithms, baselines, scenarios,
                 graphml_template=None, ratings_path=None, delta=None, t0=None,
                 rating_norm=5.0, i2kg_map_path=None, K=TOP_K,
                 centrality_name=DEFAULT_CENTRALITY, n_workers=None,
                 limit=None, out_dir=None, resume=True):
    if n_workers is None:
        n_workers = NUM_WORKERS
    cfg = DATASETS[dataset]
    scenarios = [s for s in scenarios if s in cfg["scenarios"]]
    baselines = [b for b in baselines if b in cfg["baselines"]]
    algorithms = [a for a in algorithms if a in _SWEEPABLE + ["naive_union"]]
    if not (scenarios and baselines and algorithms):
        raise ValueError(f"Nothing to run: scenarios={scenarios}, "
                         f"baselines={baselines}, algorithms={algorithms}.")

    out_dir = out_dir or (RESULTS_ROOT / "_beta_ablation" / dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "beta_ablation_long.csv"

    disk_mode = graphml_template is not None
    edge_table = item_map = None
    G = None
    if disk_mode:
        # structure is identical across beta: load first graph for centrality.
        b1_0, b2_0 = round(1.0 - shares[0], 6), round(shares[0], 6)
        G = _load_beta_graph(graphml_template, b1_0, b2_0)
        all_items = _item_node_set(G)
    else:
        if ratings_path is None or delta is None:
            raise ValueError("In-memory mode needs --ratings and a decay; or pass "
                             "--graphml-template for the disk mode.")
        G = load_graph(dataset)
        all_items = _item_node_set(G)
        if i2kg_map_path:
            item_map = load_i2kg_map(Path(i2kg_map_path))
            print(f"[beta] loaded i2kg_map: {len(item_map)} items")
        else:
            item_map = build_item_id_map(G, Path(ratings_path), out_dir / "item_id_map.json")
        edge_table, t_max = build_edge_table(Path(ratings_path), item_map)
        if t0 is None:
            t0 = t_max
        print(f"[beta] edge table: {len(edge_table)} interactions; "
              f"t0={t0:.0f}, rating_norm={rating_norm:g}, delta={delta:.3e}")

    # Centrality is structural (weight-independent): compute once.
    need_solver = "wpcst" in algorithms
    centrality = None
    if need_solver:
        cache = RESULTS_ROOT / "_centrality_cache" / f"{dataset}_{centrality_name}.pkl"
        centrality = get_or_compute(G, centrality_name, cache_path=cache)

    rows = []
    t_start = time.time()
    for si, s in enumerate(shares):
        t_share = time.time()
        beta1, beta2 = round(1.0 - s, 6), round(s, 6)
        if disk_mode:
            G = G if si == 0 else _load_beta_graph(graphml_template, beta1, beta2)
        else:
            n = apply_weights(G, beta1, beta2, edge_table, delta, t0, rating_norm)
            print(f"[beta] beta1={beta1:.2f} beta2={beta2:.2f}: set {n} weights")
        index = PcstIndex(G) if (need_solver or "pcst" in algorithms) else None
        G_und = G.to_undirected(as_view=False)
        dterm_dir = out_dir / "_dterm_cache" / _beta_tag(beta1, beta2)

        for scenario in scenarios:
            spec = _SCENARIOS[scenario]
            if spec["needs_item_paths"] and cfg["item_paths_template"] is None:
                continue
            for baseline in baselines:
                requests = _build_requests(dataset, scenario, baseline, K)
                if limit is not None:
                    requests = requests[:limit]
                if not requests:
                    continue
                cell_workers = 1 if spec["force_serial"] else n_workers
                for alg in algorithms:
                    stem = f"{dataset}_{scenario}_{baseline}_{alg}_{_beta_tag(beta1, beta2)}"
                    jsonl = out_dir / f"{stem}.jsonl"
                    if resume and jsonl.exists() and jsonl.stat().st_size > 0:
                        results = load_records(jsonl)
                        tag = "resumed"
                    else:
                        use_index = index if alg in ("pcst", "wpcst") else None
                        use_cent = centrality if alg == "wpcst" else None
                        use_gund = G_und if alg in ("pcst", "wpcst") else None
                        dispatch = make_dispatcher(
                            alg, G=G, index=use_index, centrality=use_cent,
                            G_und=use_gund, K=K, lam=LAMBDA, gamma=GAMMA,
                            anchor_kind=spec["anchor_kind"], use_root=spec["use_root"],
                            dataset=dataset, dterm_cache_dir=dterm_dir)
                        results = execute(requests, dispatch,
                                          anchor_kind=spec["anchor_kind"],
                                          n_workers=cell_workers)
                        write_results(jsonl, results)
                        tag = "computed"

                    df = compute_per_anchor(
                        results, G, dataset=dataset, scenario=scenario,
                        baseline=baseline, algorithm=alg,
                        centrality=centrality_name if alg == "wpcst" else None,
                        all_items=all_items, popular_items=None)
                    m = len(df)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        rbar = (df["relevance"] / df["num_edges"].replace(0, np.nan)).mean()
                    row = {
                        "dataset": dataset, "scenario": scenario,
                        "baseline": baseline, "algorithm": alg,
                        "beta1": beta1, "beta2": beta2, "recency_share": beta2,
                        "n_anchors": m,
                        "comprehensibility": float(df["comprehensibility"].mean()) if m else "",
                        "relevance": float(df["relevance"].mean()) if m else "",
                        "evidence_density": float(rbar) if m else "",
                        "actionability": float(df["actionability"].mean()) if m else "",
                        "diversity": float(df["diversity"].mean()) if m else "",
                        "faithfulness": float(df["faithfulness"].mean()) if m else "",
                        "num_edges": float(df["num_edges"].mean()) if m else "",
                    }
                    rows.append(row)
                    _append_row(csv_path, row)   # crash-safe incremental write
                    cval = row["comprehensibility"]
                    rval = row["relevance"]
                    print(f"[beta]   {scenario:<12} {baseline:<4} {alg:<5} "
                          f"s={beta2:.2f}  C={cval if cval=='' else round(cval,4)} "
                          f"R={rval if rval=='' else round(rval,3)}  "
                          f"({tag}) [{time.time()-t_start:.1f}s total]")
        print(f"[beta] === share s={beta2:.2f} done in {time.time()-t_share:.1f}s "
              f"({time.time()-t_start:.1f}s total) ===")

    print(f"[beta] all {len(shares)} shares done in {time.time()-t_start:.1f}s")
    print(f"[beta] rows appended incrementally to {csv_path}")
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    p.add_argument("--shares", default="0.25,0.5,0.75,1.0",
                   help="recency shares s; (beta1,beta2)=(1-s,s)")
    p.add_argument("--algorithms", default="st,pcst,wpcst")
    p.add_argument("--baselines", default="pgpr,cafe")
    p.add_argument("--scenarios", default="user_centric,user_group")
    p.add_argument("--include-naive", action="store_true")
    p.add_argument("--centrality", default=DEFAULT_CENTRALITY)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-resume", action="store_true",
                   help="recompute cells even if their .jsonl already exists")
    # disk mode (recommended): load pre-built per-beta graphs
    p.add_argument("--graphml-template", default=None,
                   help='path pattern with {b1} {b2}, e.g. '
                        '"ablation_data/ml1m_kg_static_b1_{b1}_b2_{b2}.graphml"')
    # in-memory mode (reweights on the fly; heavier on RAM)
    p.add_argument("--ratings", default=None)
    p.add_argument("--i2kg-map", default=None)
    p.add_argument("--rating-norm", type=float, default=5.0)
    p.add_argument("--gamma", type=float, default=1e-9)
    p.add_argument("--t0", type=float, default=None)
    args = p.parse_args()

    shares = [float(x) for x in args.shares.split(",")]
    algorithms = [a.strip() for a in args.algorithms.split(",")]
    if args.include_naive and "naive_union" not in algorithms:
        algorithms.append("naive_union")
    t0 = args.t0
    if t0 is None and args.graphml_template is None:
        t0 = float(datetime.now().timestamp())

    run_ablation(
        args.dataset, shares=shares, algorithms=algorithms,
        baselines=[b.strip() for b in args.baselines.split(",")],
        scenarios=[s.strip() for s in args.scenarios.split(",")],
        graphml_template=args.graphml_template, ratings_path=args.ratings,
        delta=args.gamma, t0=t0, rating_norm=args.rating_norm,
        i2kg_map_path=args.i2kg_map, centrality_name=args.centrality,
        n_workers=args.workers, limit=args.limit, resume=not args.no_resume)


if __name__ == "__main__":
    main()
