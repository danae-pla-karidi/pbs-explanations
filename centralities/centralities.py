"""
Pluggable centrality measures for WPCST.

The original ICDE/TKDE-draft code hardcoded normalized degree centrality
inside the WPCST runner. The TKDE revision needs alternative centralities
(PageRank, betweenness) so we can address the reviewer's question about
whether the item-centric weakness is a centrality-choice artifact.

All centrality functions take a graph G and return a dict {node_id: float in [0,1]}
with the maximum value normalized to 1. This contract is what assign_prizes
expects.
"""

from __future__ import annotations
import time
from pathlib import Path
import pickle
from typing import Callable

import networkx as nx

from config import BETWEENNESS_K_SOURCES, SEED


# ---------------------------------------------------------------------------
# Normalization helper
# ---------------------------------------------------------------------------

def _normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    m = max(values.values())
    if m <= 0:
        return {k: 0.0 for k in values}
    return {k: v / m for k, v in values.items()}


# ---------------------------------------------------------------------------
# Concrete centralities
# ---------------------------------------------------------------------------

def degree_centrality(G: nx.Graph) -> dict[str, float]:
    """Normalized degree centrality on the undirected projection.

    Matches the ICDE / TKDE-draft default exactly so WPCST-degree numbers are
    a drop-in replacement for the existing ones.
    """
    UG = G.to_undirected() if isinstance(G, nx.DiGraph) else G
    degs = dict(UG.degree())
    return _normalize(degs)


def pagerank_centrality(G: nx.Graph, alpha: float = 0.85,
                        max_iter: int = 100, tol: float = 1e-6) -> dict[str, float]:
    """PageRank on the undirected projection. Reasonably fast even on the
    full ML1M / LFM-1M graphs (a few minutes wall-clock)."""
    UG = G.to_undirected() if isinstance(G, nx.DiGraph) else G
    print(f"[centrality] running PageRank: |V|={UG.number_of_nodes()}, |E|={UG.number_of_edges()}")
    t0 = time.time()
    pr = nx.pagerank(UG, alpha=alpha, max_iter=max_iter, tol=tol)
    print(f"[centrality] PageRank done in {time.time() - t0:.1f}s")
    return _normalize(pr)


def betweenness_approx_centrality(G: nx.Graph, k: int = BETWEENNESS_K_SOURCES,
                                  seed: int = SEED) -> dict[str, float]:
    """k-source approximate betweenness centrality.

    Exact betweenness on a graph with ~10^5 nodes is days. The k-source
    approximation samples k pivots and is the standard practical alternative;
    it is built into networkx as `betweenness_centrality(G, k=...)`.
    """
    UG = G.to_undirected() if isinstance(G, nx.DiGraph) else G
    print(f"[centrality] running betweenness approx (k={k}): |V|={UG.number_of_nodes()}")
    t0 = time.time()
    bc = nx.betweenness_centrality(UG, k=k, seed=seed, normalized=True)
    print(f"[centrality] betweenness approx done in {time.time() - t0:.1f}s")
    return _normalize(bc)


# ---------------------------------------------------------------------------
# Registry & cached loader
# ---------------------------------------------------------------------------

CENTRALITY_FN: dict[str, Callable[[nx.Graph], dict[str, float]]] = {
    "degree": degree_centrality,
    "pagerank": pagerank_centrality,
    "betweenness_approx": betweenness_approx_centrality,
}


def get_or_compute(G: nx.Graph, name: str, cache_path: Path | None = None) -> dict[str, float]:
    """Compute centrality of `name` on G, or load from cache_path if present.

    Centralities depend only on the graph structure, so we cache them once per
    (dataset, centrality) pair and reuse across all WPCST runs. Without caching,
    PageRank and betweenness would be re-computed for every anchor.
    """
    if name not in CENTRALITY_FN:
        raise ValueError(f"Unknown centrality: {name!r}; choose from {list(CENTRALITY_FN)}")

    if cache_path is not None and cache_path.exists():
        print(f"[centrality] loading cached '{name}' from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    cent = CENTRALITY_FN[name](G)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(cent, f)
        print(f"[centrality] cached '{name}' to {cache_path}")

    return cent
