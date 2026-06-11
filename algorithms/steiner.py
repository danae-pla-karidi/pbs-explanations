"""
Steiner-tree summary explanation (the ICDE algorithm).

Refactored from `lastfm_dumps/experiment_user_centric.py` and the equivalent
ML1M code, to the unified pipeline in this repo.

Key change vs. the original: the per-anchor reweighting is delegated to
`_common.apply_reweighting` so ST, PCST, and WPCST share a single source of
truth for the edge-weight update step.
"""

from __future__ import annotations
from typing import Sequence

import networkx as nx
from networkx.algorithms.approximation import steiner_tree

from ._common import (
    AnchorRequest,
    edge_frequency,
    apply_reweighting,
    PerfTracker,
    package_result,
)


def run_steiner(G: nx.DiGraph, req: AnchorRequest, *, lam: float, K: int,
                anchor_kind: str) -> dict:
    """Single-anchor Steiner-tree summary.

    The classical Steiner tree minimizes the sum of edge costs (= max_w - w_e)
    over a tree spanning all terminals. We invert weights so that a maximum
    *weight* problem becomes a minimum *cost* one, then call networkx's
    Kou-Markowsky-Berman approximation.
    """
    with PerfTracker() as perf:
        # 1. Reweight edges using top-K path frequencies.
        freq = edge_frequency(req.top_k_paths)
        max_w, _ = apply_reweighting(G, freq, lam=lam, K=K)
        if max_w == -float("inf"):
            return package_result(
                anchor_id=req.anchor_id, anchor_kind=anchor_kind, K=K,
                solution_nodes=[], solution_edges=[], sum_weight=0.0,
                perf=perf, top_k_paths=req.top_k_paths,
        metadata=req.metadata,
            )

        # 2. Project to undirected; Steiner approx requires undirected.
        UG = G.to_undirected(as_view=False)
        for u, v in UG.edges():
            # Use cost = max_w - w_e; networkx Steiner reads weight kw 'weight'
            UG[u][v]["weight"] = max_w - UG[u][v].get("w_e", 0.0)

        # 3. Restrict to terminals present in the graph.
        terms = [t for t in {req.anchor_id, *req.terminals} if t in UG]
        if len(terms) < 2:
            return package_result(
                anchor_id=req.anchor_id, anchor_kind=anchor_kind, K=K,
                solution_nodes=terms, solution_edges=[], sum_weight=0.0,
                perf=perf, top_k_paths=req.top_k_paths,
        metadata=req.metadata,
            )

        # 4. Solve.
        T = steiner_tree(UG, terms, weight="weight")

        # 5. Compute objective: total *weight* of selected edges (positive scale).
        sum_weight = 0.0
        edges_out = []
        for u, v in T.edges():
            we = G[u][v]["w_e"] if G.has_edge(u, v) else G[v][u]["w_e"]
            sum_weight += we
            edges_out.append((u, v))

    return package_result(
        anchor_id=req.anchor_id, anchor_kind=anchor_kind, K=K,
        solution_nodes=list(T.nodes()), solution_edges=edges_out,
        sum_weight=sum_weight, perf=perf, top_k_paths=req.top_k_paths,
        metadata=req.metadata,
    )
