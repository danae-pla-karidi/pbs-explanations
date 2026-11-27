"""
Item-group scenario runner.

Two groups based on popularity quartile: top quartile = popular,
bottom quartile = unpopular. Available on ML1M only.

Lambda is fixed to LAMBDA in config.settings (1.0).
"""

from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path

from config import (
    DATASETS, TOP_K, LAMBDA, GAMMA, RESULTS_ROOT, DEFAULT_CENTRALITY,
)
from sampling.make_samples import load_items
from algorithms import AnchorRequest, PcstIndex
from centralities import get_or_compute
from runners._core import (
    load_graph, load_item_paths, make_dispatcher, execute, output_path, write_results,
)


def build_group_requests(items_df, paths_by_item: dict[str, dict], K: int) -> list[AnchorRequest]:
    out = []
    for label, mask in [
        ("popular",   items_df.quartile == "q4_popular"),
        ("unpopular", items_df.quartile == "q1_unpopular"),
    ]:
        group_items = items_df[mask]["item_id"].astype(str).tolist()
        all_paths = []
        terminal_counts = Counter()
        for iid in group_items:
            r = paths_by_item.get(iid)
            if r is None:
                continue
            # See note in run_item_centric.py: terminals here are USERS who
            # interacted with the item, keyed under `recommended_users_ids`
            # in the item-paths file.
            for u in r.get("recommended_users_ids", [])[:K]:
                terminal_counts[str(u)] += 1
            all_paths.extend(r.get("top_k_paths", [])[:K])
        terminals = [u for u, _ in terminal_counts.most_common(K)]
        if not terminals:
            print(f"[item-group] WARN: empty group {label}")
            continue
        out.append(AnchorRequest(
            anchor_id=f"group_{label}",
            terminals=terminals,
            top_k_paths=all_paths,
            metadata={"group_kind": "popularity", "group_label": label,
                      "group_size": len(group_items)},
        ))
    return out


def run(dataset: str, baseline: str, algorithm: str, *, K: int,
        centrality_name: str, budget: int | None = None,
        n_workers: int | None = None, tag: str | None = None) -> Path:
    if "item_group" not in DATASETS[dataset]["scenarios"]:
        raise RuntimeError(
            f"Item-group scenario not supported for {dataset} "
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
    requests = build_group_requests(items_df, paths_by_item, K=K)
    print(f"[item-group] {dataset}/{baseline}/{algorithm}: {len(requests)} group requests")

    dispatch = make_dispatcher(
        algorithm, G=G, index=index, centrality=centrality, G_und=G_und,
        K=K, lam=LAMBDA, gamma=GAMMA, anchor_kind="item_group",
        use_root=False, budget=budget,
        dataset=dataset,
        dterm_cache_dir=RESULTS_ROOT / "_dterm_cache",
    )
    results = execute(requests, dispatch, anchor_kind="item_group", n_workers=1)

    out = output_path(dataset, "item_group", baseline, algorithm,
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
