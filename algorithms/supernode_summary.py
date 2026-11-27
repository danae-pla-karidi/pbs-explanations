from __future__ import annotations
from collections import defaultdict
from typing import Iterable

import networkx as nx

from ._common import AnchorRequest, PerfTracker, package_result


def _candidate_node_set(req: AnchorRequest) -> set[str]:
    nodes: set[str] = set()
    for path in req.top_k_paths:
        for step in path:
            nodes.add(str(step[2]))
    nodes.add(str(req.anchor_id))
    nodes.update(str(t) for t in req.terminals)
    return nodes


def _supernode_id(G: nx.DiGraph, n: str, *, anchor: str,
                  terminals: set[str]) -> str:
    """Map node n to its super-node ID.  Anchor and terminals stay
    singletons, everything else collapses to its type."""
    if n == anchor:
        return n
    if n in terminals:
        return n
    data = G.nodes.get(n, {})
    t = data.get("type") or (
        "user" if str(n).startswith("u") else
        "item" if str(n).isdigit() else
        "external"
    )
    sub = data.get("ext_type") or data.get("category")
    if t == "external" and sub:
        return f"__super_{t}_{sub}"
    return f"__super_{t}"


def _candidate_edges(G: nx.DiGraph, V: set[str]) -> Iterable[tuple[str, str]]:
    """All edges of G with both endpoints in V (the explanation subgraph
    edge set, undirected/canonicalised)."""
    seen: set[tuple[str, str]] = set()
    for u in V:
        if u not in G:
            continue
        for v in G.successors(u):
            if v in V:
                a, b = sorted([str(u), str(v)])
                seen.add((a, b))
        for v in G.predecessors(u):
            if v in V:
                a, b = sorted([str(u), str(v)])
                seen.add((a, b))
    return seen


def run_supernode_summary(G: nx.DiGraph, req: AnchorRequest, *, lam: float,
                          K: int, anchor_kind: str,
                          # The signature mirrors run_pappas2017 for runner
                          # compatibility, but importance and budget are
                          # accepted and ignored (super-noding has no
                          # tunable budget; the cluster count is determined
                          # by node-type cardinality).
                          importance: dict[str, float] | None = None,
                          budget: int | None = None,
                          G_und: nx.Graph | None = None) -> dict:
    """Type-based super-node aggregation summary."""
    with PerfTracker() as perf:
        V = _candidate_node_set(req)
        anchor = str(req.anchor_id)
        terminals = {str(t) for t in req.terminals}

        # Map each candidate to its super-node
        super_of: dict[str, str] = {}
        for n in V:
            super_of[n] = _supernode_id(G, n, anchor=anchor, terminals=terminals)

        # Collect super-nodes (with size info for reporting)
        super_members: dict[str, list[str]] = defaultdict(list)
        for n, s in super_of.items():
            super_members[s].append(n)

        # Project edges into super-node space.  We preserve direction-free
        # edges between super-nodes when at least one underlying edge
        # connects their members.
        super_edges: set[tuple[str, str]] = set()
        for u, v in _candidate_edges(G, V):
            su, sv = super_of[u], super_of[v]
            if su == sv:
                continue  # intra-super-node edges collapse away
            a, b = sorted([su, sv])
            super_edges.add((a, b))

        nodes_out = sorted(super_members.keys())
        edges_out = [list(e) for e in sorted(super_edges)]

        # Sum-of-weights is undefined for super-nodes (the aggregate edge
        # has no direct weight).  Report 0.0 by convention; downstream
        # relevance metric will pick this up as 0.0, which is correct
        # for a method that throws away edge weights.
        sum_weight = 0.0

    return package_result(
        anchor_id=req.anchor_id, anchor_kind=anchor_kind, K=K,
        solution_nodes=nodes_out, solution_edges=edges_out,
        sum_weight=sum_weight, perf=perf, top_k_paths=req.top_k_paths,
        metadata={**req.metadata, "baseline": "supernode",
                  "n_supernodes": len(nodes_out),
                  "n_underlying_nodes": len(V),
                  "compression_ratio": (
                      len(nodes_out) / len(V) if V else 1.0)},
    )
