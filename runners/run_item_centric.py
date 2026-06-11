"""
Item-centric scenario runner.

The "anchor" is a target item; terminals are users who interacted with the
item (or a top-K subset thereof, taken from the precomputed item_paths file).
Available on ML1M only (LFM-1M lacks item_paths_processed.jsonl).

Lambda is fixed to LAMBDA in config.settings (1.0).
"""

from __future__ import annotations
import argparse
from pathlib import Path

from config import (
    DATASETS, TOP_K, LAMBDA, GAMMA, RESULTS_ROOT, DEFAULT_CENTRALITY, NUM_WORKERS,
)
from sampling.make_samples import load_items
from algorithms import AnchorRequest, PcstIndex
from centralities import get_or_compute
from runners._core import (
    load_graph, load_item_paths, make_dispatcher, execute, output_path, write_results,
)


def build_requests(items_df, paths_by_item: dict[str, dict], K: int) -> list[AnchorRequest]:
    out = []
    for _, row in items_df.iterrows():
        iid = str(row["item_id"])
        rec = paths_by_item.get(iid)
        if rec is None:
            print(f"[item-centric] WARN: no item-path record for {iid}")
            continue
        # Item-centric: anchor is an ITEM, terminals are USERS.  The
        # item-paths file uses the key `recommended_users_ids` (note: users,
        # not items).  An earlier version of this runner used
        # `recommended_items_ids`, which is the user-centric key, and
        # therefore returned empty terminal lists for every anchor.
        terminals = [str(t) for t in rec.get("recommended_users_ids", [])[:K]]
        paths = rec.get("top_k_paths", [])[:K]
        out.append(AnchorRequest(
            anchor_id=iid,
            terminals=terminals,
            top_k_paths=paths,
            metadata={
                "popularity_quartile": row.get("quartile", "unknown"),
                "n_users": int(row.get("n_users", 0)),
            },
        ))
    return out


def run(dataset: str, baseline: str, algorithm: str, *, K: int,
        centrality_name: str, budget: int | None = None,
        n_workers: int | None = None, tag: str | None = None) -> Path:
    if "item_centric" not in DATASETS[dataset]["scenarios"]:
        raise RuntimeError(
            f"Item-centric scenario not supported for {dataset} "
            "(no item_paths_processed.jsonl available)."
        )
    G = load_graph(dataset)
    G_und = G.to_undirected(as_view=False) if algorithm in ("pcst", "wpcst", "pappas2017", "mst", "faces", "supernode") else None
    index = PcstIndex(G) if algorithm in ("pcst", "wpcst") else None

    centrality = None
    if algorithm in ("wpcst", "pappas2017", "mst", "faces"):
        cache = RESULTS_ROOT / "_centrality_cache" / f"{dataset}_{centrality_name}.pkl"
        centrality = get_or_compute(G, centrality_name, cache_path=cache)

    paths_by_item = load_item_paths(dataset, baseline)
    items_df = load_items(dataset)
    requests = build_requests(items_df, paths_by_item, K=K)
    print(f"[item-centric] {dataset}/{baseline}/{algorithm}: {len(requests)} requests")

    dispatch = make_dispatcher(
        algorithm, G=G, index=index, centrality=centrality, G_und=G_und,
        K=K, lam=LAMBDA, gamma=GAMMA, anchor_kind="item",
        use_root=True, budget=budget,
        dataset=dataset,
        dterm_cache_dir=RESULTS_ROOT / "_dterm_cache",
    )
    results = execute(requests, dispatch, anchor_kind="item",
                      n_workers=n_workers if n_workers is not None else NUM_WORKERS)

    out = output_path(dataset, "item_centric", baseline, algorithm,
                      centrality=centrality_name if algorithm in ("wpcst", "pappas2017", "mst", "faces", "supernode") else None,
                      tag=tag)
    write_results(out, results)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    p.add_argument("--baseline", required=True)
    p.add_argument("--algorithm", required=True)
    p.add_argument("--K", type=int, default=TOP_K)
    p.add_argument("--centrality", default=DEFAULT_CENTRALITY)
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--tag", default=None)
    args = p.parse_args()
    run(args.dataset, args.baseline, args.algorithm,
        K=args.K, centrality_name=args.centrality, budget=args.budget,
        n_workers=args.workers, tag=args.tag)


if __name__ == "__main__":
    main()
