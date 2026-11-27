from __future__ import annotations

import networkx as nx

from ._common import AnchorRequest, PerfTracker, package_result


def run_naive_union(G: nx.DiGraph, req: AnchorRequest, *, lam: float, K: int,
                    anchor_kind: str) -> dict:
    """Return the union of nodes and edges from the top-K paths.

    `lam` is unused (no reweighting is performed) but kept in the signature so
    runners can call all algorithms uniformly. Edge weights for the result are
    read from the original graph, not a reweighted copy.
    """
    with PerfTracker() as perf:
        nodes: set[str] = set()
        edges: set[tuple[str, str]] = set()

        for path in req.top_k_paths[:K]:
            seq = [str(step[2]) for step in path]
            nodes.update(seq)
            for u, v in zip(seq, seq[1:]):
                # canonicalize edge orientation if present in graph as either direction
                if G.has_edge(u, v):
                    edges.add((u, v))
                elif G.has_edge(v, u):
                    edges.add((v, u))
                else:
                    # path traverses an edge not in G (rare; e.g. self-loop encoding)
                    edges.add((u, v))

        # ensure anchor and terminals are recorded as nodes
        nodes.add(req.anchor_id)
        nodes.update(req.terminals)

        sum_weight = 0.0
        for u, v in edges:
            if G.has_edge(u, v):
                sum_weight += float(G[u][v].get("weight", 0.0))

    return package_result(
        anchor_id=req.anchor_id, anchor_kind=anchor_kind, K=K,
        solution_nodes=list(nodes), solution_edges=list(edges),
        sum_weight=sum_weight, perf=perf, top_k_paths=req.top_k_paths,
        metadata=req.metadata,
    )
