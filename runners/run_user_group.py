"""
User-group scenario runner.

Treats each user-subgroup (e.g. male users, female users) as a single anchor.
The terminals are the union of recommended items across the group, restricted
to those that appear in the recommendation file. The aggregated top-K paths
are concatenated across the group's users.

Lambda is fixed to LAMBDA in config.settings (1.0).
"""

from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path

from config import (
    DATASETS, TOP_K, LAMBDA, GAMMA, RESULTS_ROOT, DEFAULT_CENTRALITY,
)
from sampling.make_samples import load_users
from algorithms import AnchorRequest, PcstIndex
from centralities import get_or_compute
from runners._core import (
    load_graph, load_user_recs, make_dispatcher, execute, output_path, write_results,
)


def build_group_requests(users_df, recs: dict[str, dict], K: int) -> list[AnchorRequest]:
    """One AnchorRequest per gender group (M, F)."""
    out = []
    for gender in ["M", "F"]:
        group_users = users_df[users_df.gender == gender]["user_id"].astype(str).tolist()
        item_counts = Counter()
        all_paths = []
        for uid in group_users:
            r = recs.get(uid)
            if r is None:
                continue
            for it in r.get("recommended_items_ids", [])[:K]:
                item_counts[str(it)] += 1
            all_paths.extend(r.get("top_k_paths", [])[:K])
        terminals = [it for it, _ in item_counts.most_common(K)]
        if not terminals:
            print(f"[user-group] WARN: empty group for gender={gender}")
            continue
        anchor_id = f"group_{'male' if gender == 'M' else 'female'}"
        out.append(AnchorRequest(
            anchor_id=anchor_id,
            terminals=terminals,
            top_k_paths=all_paths,
            metadata={"group_kind": "gender", "group_label": gender,
                      "group_size": len(group_users)},
        ))
    return out


def run(dataset: str, baseline: str, algorithm: str, *, K: int,
        centrality_name: str, budget: int | None = None,
        n_workers: int | None = None, tag: str | None = None) -> Path:
    G = load_graph(dataset)
    G_und = G.to_undirected(as_view=False) if algorithm in ("pcst", "wpcst", "pappas2017", "mst", "faces", "supernode") else None
    index = PcstIndex(G) if algorithm in ("pcst", "wpcst") else None

    centrality = None
    if algorithm in ("wpcst", "pappas2017", "mst", "faces"):
        cache = RESULTS_ROOT / "_centrality_cache" / f"{dataset}_{centrality_name}.pkl"
        centrality = get_or_compute(G, centrality_name, cache_path=cache)

    recs = load_user_recs(dataset, baseline)
    users_df = load_users(dataset)
    requests = build_group_requests(users_df, recs, K=K)
    print(f"[user-group] {dataset}/{baseline}/{algorithm}: {len(requests)} group requests")

    dispatch = make_dispatcher(
        algorithm, G=G, index=index, centrality=centrality, G_und=G_und,
        K=K, lam=LAMBDA, gamma=GAMMA, anchor_kind="user_group",
        use_root=False, budget=budget,
        dataset=dataset,
        dterm_cache_dir=RESULTS_ROOT / "_dterm_cache",
    )
    # Group runs are 2 anchors total - parallelism is irrelevant; use serial.
    results = execute(requests, dispatch, anchor_kind="user_group", n_workers=1)

    out = output_path(dataset, "user_group", baseline, algorithm,
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
