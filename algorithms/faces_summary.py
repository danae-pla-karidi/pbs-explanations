"""
FACES-style diversity-aware summarization baseline.

Reference:
    K. Gunaratna, K. Thirunarayan, A. Sheth,
    "FACES: Diversity-aware entity summarization using incremental
     hierarchical conceptual clustering",
    in AAAI 2015, pp. 116-122.

Original FACES summarises a single entity by selecting K facts (= edges
adjacent to the entity) that maximise diversity, where diversity is
measured by clustering predicates and picking one fact per cluster.

Adaptation to multi-anchor explanation summaries
------------------------------------------------
Our problem is "summarise a subgraph induced by top-K paths around an
anchor", which generalises FACES' "summarise the facts around one entity".
We adapt by:

  1. Restrict candidates to the NaiveUnion node set (the explanation
     subgraph), as with Pappas2017 and MST.
  2. Cluster the *optional* nodes (i.e. excluding the anchor and the
     recommendation terminals, which must be retained) by their
     KG node type.  Node types in our graphs are the canonical
     ``type`` attribute (``user``, ``item``, ``external``) plus any
     finer-grained ``ext_type`` if available.  Cluster on the most
     specific type attribute present.
  3. From each cluster, pick the node with highest importance score
     (e.g. degree centrality) -- a balance between coverage (one
     representative per cluster) and salience (best representative).
  4. Distribute the budget proportionally across clusters when the
     budget is smaller than the number of clusters; spread evenly
     (round-robin by importance) when the budget is larger.
  5. Connect the chosen nodes via the same Steiner-tree machinery
     used by ST and Pappas2017, so the output is a connected
     subgraph rather than a disconnected fact list (which would not
     compare directly to the other algorithms).

Why this is a meaningful baseline
---------------------------------
This isolates a different design choice: should the summary be driven
by *importance* (Pappas, MST) or by *diversity* (FACES)?  By implementing
both at the same matched budget, we expose the importance-vs-diversity
trade-off directly.  WPCST sits on a third axis (path-frequency adapted
prizes), so the three baselines now span a recognisable design space.
"""

from __future__ import annotations
from collections import defaultdict
from typing import Iterable

import networkx as nx
from networkx.algorithms.approximation import steiner_tree

from ._common import AnchorRequest, PerfTracker, package_result


def _candidate_node_set(req: AnchorRequest) -> set[str]:
    nodes: set[str] = set()
    for path in req.top_k_paths:
        for step in path:
            nodes.add(str(step[2]))
    nodes.add(str(req.anchor_id))
    nodes.update(str(t) for t in req.terminals)
    return nodes


def _node_type(G: nx.DiGraph, n: str) -> str:
    """Return the most specific available type label for node n.

    Falls back to a coarse heuristic when the graph lacks the ``type``
    attribute: ``u``-prefixed IDs are users, purely-numeric low IDs are
    items, everything else is ``external``.
    """
    data = G.nodes.get(n, {})
    t = data.get("type")
    if t:
        # If the graph carries a finer label (e.g. dbpedia category for
        # external nodes), prefer that for clustering granularity.
        sub = data.get("ext_type") or data.get("category")
        if t == "external" and sub:
            return f"external:{sub}"
        return t
    s = str(n)
    if s.startswith("u"):
        return "user"
    if s.isdigit():
        return "item"
    return "external"


def run_faces_summary(G: nx.DiGraph, req: AnchorRequest, *, lam: float, K: int,
                      anchor_kind: str, importance: dict[str, float],
                      budget: int, G_und: nx.Graph | None = None) -> dict:
    """FACES-style diversity-aware summary at matched budget."""
    with PerfTracker() as perf:
        Vc = _candidate_node_set(req)
        required = {str(req.anchor_id), *(str(t) for t in req.terminals)}

        # 1. Cluster optional nodes by type
        clusters: dict[str, list[str]] = defaultdict(list)
        for n in Vc:
            if n in required or n not in G:
                continue
            clusters[_node_type(G, n)].append(n)

        # 2. Within each cluster, sort by importance (descending)
        for t in clusters:
            clusters[t].sort(key=lambda n: -importance.get(n, 0.0))

        # 3. Round-robin pick across clusters until budget exhausted
        chosen_optional: list[str] = []
        cluster_keys = sorted(clusters.keys())
        head = {t: 0 for t in cluster_keys}
        while len(chosen_optional) < max(0, budget):
            picked_in_round = False
            for t in cluster_keys:
                if len(chosen_optional) >= budget:
                    break
                if head[t] < len(clusters[t]):
                    chosen_optional.append(clusters[t][head[t]])
                    head[t] += 1
                    picked_in_round = True
            if not picked_in_round:
                break  # all clusters exhausted

        # 4. Final terminal set
        terms = required | set(chosen_optional)
        terms = terms & set(G.nodes())
        if len(terms) < 2:
            return package_result(
                anchor_id=req.anchor_id, anchor_kind=anchor_kind, K=K,
                solution_nodes=list(terms), solution_edges=[],
                sum_weight=0.0, perf=perf, top_k_paths=req.top_k_paths,
                metadata={**req.metadata, "baseline": "faces",
                          "budget": budget, "n_clusters": len(clusters)},
            )

        # 5. Connect via Steiner tree.  Same procedure as Pappas2017;
        # what differs is which nodes ended up in `terms`.
        UG = G_und if G_und is not None else G.to_undirected(as_view=False)
        for u, v in UG.edges():
            UG[u][v]["weight"] = 1.0  # unit-cost
        try:
            T = steiner_tree(UG, list(terms), weight="weight")
        except Exception as e:
            print(f"[faces] Steiner failed for anchor={req.anchor_id}: {e}")
            return package_result(
                anchor_id=req.anchor_id, anchor_kind=anchor_kind, K=K,
                solution_nodes=list(terms), solution_edges=[],
                sum_weight=0.0, perf=perf, top_k_paths=req.top_k_paths,
                metadata={**req.metadata, "baseline": "faces",
                          "budget": budget, "error": str(e)},
            )

        sum_weight = 0.0
        edges_out: list[tuple[str, str]] = []
        for u, v in T.edges():
            if G.has_edge(u, v):
                sum_weight += float(G[u][v].get("weight", 0.0))
            elif G.has_edge(v, u):
                sum_weight += float(G[v][u].get("weight", 0.0))
            edges_out.append((u, v))

    return package_result(
        anchor_id=req.anchor_id, anchor_kind=anchor_kind, K=K,
        solution_nodes=list(T.nodes()), solution_edges=edges_out,
        sum_weight=sum_weight, perf=perf, top_k_paths=req.top_k_paths,
        metadata={**req.metadata, "baseline": "faces", "budget": budget,
                  "n_clusters": len(clusters)},
    )
