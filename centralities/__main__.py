"""
CLI entry point for precomputing centralities.

    python3 -m centralities --dataset ml1m --centrality degree
    python3 -m centralities --dataset ml1m --centrality pagerank
    python3 -m centralities --dataset ml1m --centrality betweenness_approx
    python3 -m centralities --dataset ml1m --centrality all

This loads the dataset's KG, computes the requested centrality, and writes
the result to `results/_centrality_cache/<dataset>_<centrality>.pkl`.  All
WPCST and external-baseline runners read from that cache via
`get_or_compute`, so precomputing here moves the heavy cost (PageRank,
betweenness) out of the per-anchor loop.

The previous module path `python3 -m centralities.centralities` worked
but issued a benign RuntimeWarning because the package's __init__ imports
the same submodule before the -m runner gets to it.  This package-level
entry point avoids that.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

from config import DATASETS, RESULTS_ROOT
from .centralities import CENTRALITY_FN, get_or_compute


def _load_kg(dataset: str):
    import networkx as nx
    cfg = DATASETS[dataset]
    graphml = cfg["graphml"]
    if not graphml.exists():
        raise FileNotFoundError(
            f"KG not found for dataset {dataset!r}: {graphml}.  "
            "Place the .graphml under data/<dataset>/ first."
        )
    print(f"[centralities] reading {graphml}")
    return nx.read_graphml(graphml, node_type=str)


def _cache_path(dataset: str, centrality: str) -> Path:
    return RESULTS_ROOT / "_centrality_cache" / f"{dataset}_{centrality}.pkl"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    p.add_argument("--centrality", required=True,
                   choices=list(CENTRALITY_FN.keys()) + ["all"])
    p.add_argument("--force", action="store_true",
                   help="Recompute even if a cache file exists.")
    args = p.parse_args(argv)

    targets = list(CENTRALITY_FN.keys()) if args.centrality == "all" \
              else [args.centrality]

    G = _load_kg(args.dataset)
    print(f"[centralities] graph: {len(G)} nodes, {G.number_of_edges()} edges")

    for cent in targets:
        path = _cache_path(args.dataset, cent)
        if path.exists() and not args.force:
            print(f"[centralities] '{cent}' already cached at {path}; skip "
                  "(pass --force to recompute)")
            continue
        t0 = time.time()
        get_or_compute(G, cent, cache_path=path)
        print(f"[centralities] '{cent}' done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
