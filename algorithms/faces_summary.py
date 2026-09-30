from __future__ import annotations
from collections import defaultdict
from typing import Iterable

import networkx as nx
from networkx.algorithms.approximation import steiner_tree

from ._common import AnchorRequest, PerfTracker, package_result

# ---------------------------------------------------------------------------
# Clustering key (revision: tuning sweep).
#   kg_type   : KG node ``type`` attribute (user / item / external).  Main tables.
#   path_type : entity type carried by the explanation paths (step[1]),
#               e.g. actor, category, genre.  A node that appears under
#               several types gets its most frequent one.
#   relation  : relation label carried by the explanation paths (step[0]),
#               e.g. starred_by_actor, belong_to_category.  Most frequent one.
# Selected through the FACES_CLUSTER_KEY environment variable so the runner
# signatures stay unchanged.  Worker processes inherit the environment.
# ---------------------------------------------------------------------------
import os
from collections import Counter

CLUSTER_KEYS = ("kg_type", "path_type", "relation")


def _cluster_key() -> str:
    key = os.environ.get("FACES_CLUSTER_KEY", "kg_type")
    if key not in CLUSTER_KEYS:
        raise ValueError(f"FACES_CLUSTER_KEY must be one of {CLUSTER_KEYS}, got {key!r}")
    return key


def _path_labels(req: AnchorRequest, idx: int) -> dict[str, str]:
    """Most frequent path label (idx=1 entity type, idx=0 relation) per node."""
    counts: dict[str, Counter] = {}
    for path in req.top_k_paths:
        for step in path:
            n = str(step[2])
            counts.setdefault(n, Counter())[str(step[idx])] += 1
    return {n: c.most_common(1)[0][0] for n, c in counts.items()}


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

        # 1. Cluster optional nodes by the selected key
        key = _cluster_key()
        labels: dict[str, str] = {}
        if key == "path_type":
            labels = _path_labels(req, 1)
        elif key == "relation":
            labels = _path_labels(req, 0)
        clusters: dict[str, list[str]] = defaultdict(list)
        n_pool = 0
        for n in Vc:
            if n in required or n not in G:
                continue
            n_pool += 1
            lab = labels.get(n, _node_type(G, n)) if key != "kg_type" else _node_type(G, n)
            clusters[lab].append(n)

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
                          "budget": budget, "n_clusters": len(clusters),
                          "cluster_key": key, "n_pool_optional": n_pool,
                          "n_admitted": len(chosen_optional)},
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
                  "n_clusters": len(clusters), "cluster_key": key,
                  "n_pool_optional": n_pool, "n_admitted": len(chosen_optional)},
    )
