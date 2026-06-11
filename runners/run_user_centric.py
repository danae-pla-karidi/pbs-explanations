"""
User-centric scenario runner.

For each sampled user, build the AnchorRequest from that user's recommendation
record (top-K items + top-K explanation paths) and run the chosen algorithm.

Lambda is fixed to LAMBDA in config.settings (1.0). The ICDE'25 paper found
this value optimal; it is not swept here.
"""

from __future__ import annotations
import argparse
from pathlib import Path

from config import (
    DATASETS, TOP_K, LAMBDA, GAMMA, RESULTS_ROOT, DEFAULT_CENTRALITY, NUM_WORKERS,
)
from sampling.make_samples import load_users
from algorithms import AnchorRequest, PcstIndex
from centralities import get_or_compute
from runners._core import (
    load_graph, load_user_recs, make_dispatcher, execute, output_path, write_results,
)


def build_requests(users_df, recs: dict[str, dict], K: int) -> list[AnchorRequest]:
    out = []
    for _, row in users_df.iterrows():
        uid = str(row["user_id"])
        rec = recs.get(uid)
        if rec is None:
            print(f"[user-centric] WARN: no recommendation record for {uid}")
            continue
        terminals = [str(t) for t in rec.get("recommended_items_ids", [])[:K]]
        paths = rec.get("top_k_paths", [])[:K]
        out.append(AnchorRequest(
            anchor_id=str(rec.get("graph_user_id", uid)),
            terminals=terminals,
            top_k_paths=paths,
            metadata={"gender": row.get("gender", "unknown"),
                      "rec_user_id": uid},
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
    requests = build_requests(users_df, recs, K=K)
    print(f"[user-centric] {dataset}/{baseline}/{algorithm}: {len(requests)} requests")

    dispatch = make_dispatcher(
        algorithm, G=G, index=index, centrality=centrality, G_und=G_und,
        K=K, lam=LAMBDA, gamma=GAMMA, anchor_kind="user",
        use_root=True, budget=budget,
        dataset=dataset,
        dterm_cache_dir=RESULTS_ROOT / "_dterm_cache",
    )
    results = execute(requests, dispatch, anchor_kind="user",
                      n_workers=n_workers if n_workers is not None else NUM_WORKERS)

    out = output_path(dataset, "user_centric", baseline, algorithm,
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
    p.add_argument("--budget", type=int, default=None,
                   help="Required for pappas2017 algorithm.")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--tag", default=None)
    args = p.parse_args()
    run(args.dataset, args.baseline, args.algorithm,
        K=args.K, centrality_name=args.centrality, budget=args.budget,
        n_workers=args.workers, tag=args.tag)


if __name__ == "__main__":
    main()
