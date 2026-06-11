"""
MST-based summarization baseline (Troullinou et al. 2015).

Reference:
    G. Troullinou, H. Kondylakis, E. Daskalaki, D. Plexousakis,
    "RDF digest: Efficient summarization of RDF/S KBs",
    in ESWC 2015, Springer LNCS 9088, pp. 119-134.

This is the algorithmic predecessor of Pappas et al. (ESWC 2017): both
score nodes by importance and connect them, but they differ in *how* they
connect.  Troullinou et al. compute a maximum-cost spanning tree (MST)
on the whole graph once and then extract the paths from the MST that
link the top-K most important nodes.  Pappas et al. argue this introduces
many additional nodes since the MST commits to one path between any two
nodes and that path may pass through irrelevant intermediate nodes.

We include MST as a baseline both because it is a published, peer-reviewed
graph summarization method and because the comparison Pappas-vs-MST is
exactly the comparison Pappas et al. originally made; reproducing it on
recommendation-explanation data extends their finding to a new domain.

Adaptation to our query-driven setting
--------------------------------------
Same as Pappas: candidates are restricted to the union of the top-K
explanation paths (NaiveUnion node set).  The anchor and recommendation
terminals are forced to be retained.  We pick the top-`budget` most
important optional nodes by the supplied importance map, then extract
the MST paths connecting all chosen nodes.

Edge weight semantics
---------------------
Troullinou et al. use a *maximum-cost* spanning tree (high-weight edges
preferred).  We invert weights so a standard minimum-cost MST routine
gives the maximum-cost solution, then read out the actual edges.
"""

from __future__ import annotations
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


def _max_cost_spanning_tree(UG: nx.Graph, w_attr: str = "weight") -> nx.Graph:
    """Maximum-cost spanning tree of UG.  Implemented by inverting weights
    and running networkx's standard MST. Caches the result on UG by
    attribute `_max_mst` so we don't recompute per anchor."""
    if hasattr(UG, "_max_mst") and UG._max_mst is not None:
        return UG._max_mst
    # Negate weights for the inversion.  Add a constant so all are positive
    # (some MST implementations dislike negatives).
    w_max = max((d.get(w_attr, 0.0) for _, _, d in UG.edges(data=True)), default=1.0)
    H = UG.copy()
    for u, v in H.edges():
        H[u][v]["_neg_weight"] = w_max - H[u][v].get(w_attr, 0.0)
    T = nx.minimum_spanning_tree(H, weight="_neg_weight")
    UG._max_mst = T
    return T


def run_mst_summary(G: nx.DiGraph, req: AnchorRequest, *, lam: float, K: int,
                    anchor_kind: str, importance: dict[str, float],
                    budget: int, G_und: nx.Graph | None = None) -> dict:
    """MST-based summary at matched budget.

    Parameters mirror `run_pappas2017` so the runner can dispatch
    interchangeably.  `lam` is unused (this method does not reweight
    by path frequency).  `G_und` is the cached undirected projection
    of G; if absent, we build it.
    """
    with PerfTracker() as perf:
        Vc = _candidate_node_set(req)
        required = {str(req.anchor_id), *(str(t) for t in req.terminals)}
        optional = [n for n in Vc if n not in required and n in G]
        optional.sort(key=lambda n: -importance.get(n, 0.0))
        chosen = required | set(optional[: max(0, budget)])

        UG = G_und if G_und is not None else G.to_undirected(as_view=False)
        # Need a 'weight' attribute on UG for MST.  Inherit from G's
        # 'weight' attribute set during graph load.
        if not nx.get_edge_attributes(UG, "weight"):
            for u, v in UG.edges():
                UG[u][v]["weight"] = float(G[u][v].get("weight", 0.0)
                                           if G.has_edge(u, v) else
                                           G[v][u].get("weight", 0.0))

        T_full = _max_cost_spanning_tree(UG, w_attr="weight")

        # Extract the subtree of T_full induced by `chosen`.  In a tree,
        # the minimal subtree containing a set of vertices is the union
        # of the unique paths between them.  We build it by computing the
        # Steiner tree of the chosen set on T_full (this is exact and
        # cheap on a tree).
        chosen_in_tree = [n for n in chosen if n in T_full]
        if len(chosen_in_tree) < 2:
            return package_result(
                anchor_id=req.anchor_id, anchor_kind=anchor_kind, K=K,
                solution_nodes=list(chosen_in_tree), solution_edges=[],
                sum_weight=0.0, perf=perf, top_k_paths=req.top_k_paths,
                metadata={**req.metadata, "baseline": "mst",
                          "budget": budget},
            )
        # Use BFS-based subtree extraction: find pairwise paths in the
        # tree and collect their union.
        nodes_out: set[str] = set()
        edges_out: set[tuple[str, str]] = set()
        anchor = chosen_in_tree[0]
        for target in chosen_in_tree[1:]:
            try:
                path = nx.shortest_path(T_full, anchor, target)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            for u, v in zip(path, path[1:]):
                a, b = sorted([u, v])
                edges_out.add((a, b))
                nodes_out.update([u, v])

        sum_weight = 0.0
        for u, v in edges_out:
            if G.has_edge(u, v):
                sum_weight += float(G[u][v].get("weight", 0.0))
            elif G.has_edge(v, u):
                sum_weight += float(G[v][u].get("weight", 0.0))

    return package_result(
        anchor_id=req.anchor_id, anchor_kind=anchor_kind, K=K,
        solution_nodes=list(nodes_out), solution_edges=list(edges_out),
        sum_weight=sum_weight, perf=perf, top_k_paths=req.top_k_paths,
        metadata={**req.metadata, "baseline": "mst", "budget": budget,
                  "n_terminals": len(chosen_in_tree)},
    )
