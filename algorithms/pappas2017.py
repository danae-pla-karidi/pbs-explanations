"""
Pappas et al. (ESWC 2017) CentPrune baseline.

Reference:
    A. Pappas, G. Troullinou, G. Roussakis, H. Kondylakis, D. Plexousakis,
    "Exploring importance measures for summarizing RDF/S KBs",
    in The Semantic Web (ESWC 2017), Springer, pp. 387-403.

The original method has two stages:
  (1) score every node by an *adapted importance measure* (AIM); they
      evaluate six measures and identify Betweenness as the winner.
  (2) connect the top-K highest-scoring nodes via a graph Steiner tree
      (they explore three approximations and recommend CHINS).

Adaptation to our query-driven setting
--------------------------------------
The original Pappas et al. method summarises an entire schema graph.  In our
recommendation-explanation setting we summarise the explanation subgraph
*induced by the top-K paths for an anchor*.  Without this restriction the
method would degenerate to "summarise the whole KG", which is a different
problem and would yield trivially poor anchor-relevance.

Concretely:
  - candidate node set V_c = the nodes appearing in the union of the top-K
    explanation paths (i.e. the NaiveUnion node set).
  - we score V_c by a chosen AIM (Betweenness by default, matching the
    paper's recommendation).
  - we pick the top-`budget` most important nodes; the anchor and the
    K terminals are always retained regardless of score (these are the
    must-have endpoints of the explanation).
  - we run Steiner tree on the induced subgraph G[V_c] with the chosen
    nodes as terminals, using NetworkX' MST-based approximation
    (the SDISTG variant from the original paper, which is functionally
    identical to networkx.algorithms.approximation.steiner_tree).

The Steiner approximation is the same one our ST baseline uses; the
*difference* is in which nodes become terminals.  ST takes anchor +
recommended items as terminals; Pappas takes anchor + recommended items
+ top-`budget` important nodes within the explanation subgraph.

Why this is a meaningful baseline
---------------------------------
This baseline isolates one specific design choice: should the summary
include the most *structurally important* nodes within the explanation
subgraph, or the most *centrally connecting* nodes (PCST/WPCST)?  It is
a fair comparison because:
  - it operates on the same anchor-and-paths input as our methods;
  - it uses the size budget of WPCST for the same (dataset, scenario,
    recommender) cell, making the comparison size-fair;
  - it implements an externally-published, peer-reviewed graph
    summarisation method without inventing a new straw man.
"""

from __future__ import annotations
from typing import Iterable

import networkx as nx
from networkx.algorithms.approximation import steiner_tree

from ._common import AnchorRequest, PerfTracker, package_result


def _candidate_node_set(req: AnchorRequest) -> set[str]:
    """Union of nodes appearing in the top-K paths plus anchor and terminals."""
    nodes: set[str] = set()
    for path in req.top_k_paths:
        for step in path:
            nodes.add(str(step[2]))
    nodes.add(str(req.anchor_id))
    nodes.update(str(t) for t in req.terminals)
    return nodes


def run_pappas2017(G: nx.DiGraph, req: AnchorRequest, *, lam: float, K: int,
                   anchor_kind: str, importance: dict[str, float],
                   budget: int, G_und: nx.Graph | None = None) -> dict:
    """Pappas et al. 2017-style baseline at matched budget.

    Parameters
    ----------
    G : DiGraph
        The full knowledge graph.
    req : AnchorRequest
        The anchor request with anchor_id, terminals, top_k_paths.
    lam, K : kept for runner-call signature uniformity; ``lam`` is unused
        because Pappas et al. do not reweight edges (the conference method
        is path-frequency-agnostic).
    anchor_kind : 'user' | 'item' | 'user_group' | 'item_group'.
    importance : pre-computed AIM scores keyed by node id.  Default is
        normalized betweenness centrality (the paper's recommendation),
        but any importance measure with the same dict shape is accepted.
    budget : the number of *additional* nodes (beyond anchor + terminals)
        to admit as Steiner terminals.  Set per-cell to match the median
        WPCST output size, so comparison is at equal size.
    G_und : optional precomputed undirected projection of G.  Building
        it inside the function would be O(|V|+|E|) per anchor; the runner
        builds it once and passes it through.
    """
    with PerfTracker() as perf:
        # 1. Restrict candidates to the explanation subgraph (NaiveUnion nodes)
        Vc = _candidate_node_set(req)

        # 2. Required nodes: anchor + recommendation terminals (always kept)
        required: set[str] = {str(req.anchor_id), *(str(t) for t in req.terminals)}

        # 3. Rank optional nodes within Vc by AIM and take the top `budget`
        optional = [n for n in Vc if n not in required and n in G]
        optional.sort(key=lambda n: -importance.get(n, 0.0))
        chosen_optional = set(optional[: max(0, budget)])

        # 4. Final terminal set for the Steiner step
        terms = (required | chosen_optional) & set(G.nodes())
        if len(terms) < 2:
            return package_result(
                anchor_id=req.anchor_id, anchor_kind=anchor_kind, K=K,
                solution_nodes=list(terms), solution_edges=[], sum_weight=0.0,
                perf=perf, top_k_paths=req.top_k_paths,
                metadata={**req.metadata, "baseline": "pappas2017",
                          "budget": budget, "n_terminals": len(terms)},
            )

        # 5. Connect terms via Steiner tree on the *full graph* (the
        #    original Pappas paper runs on the whole schema; we follow that
        #    so the connecting paths can use any KG edge, not only those
        #    that happened to appear in the top-K paths).
        if G_und is None:
            UG = G.to_undirected(as_view=False)
        else:
            UG = G_und
        # Steiner approximation needs a 'weight' edge attribute.
        # Setting it once per call is cheap relative to the solver itself
        # and is necessary because the runner-level UG might be shared
        # with WPCST (which uses 'w_e').
        for u, v in UG.edges():
            UG[u][v]["weight"] = 1.0  # uniform; Pappas et al. minimize
                                      # number of additional nodes, which
                                      # is equivalent to unit-cost Steiner.
        try:
            T = steiner_tree(UG, list(terms), weight="weight")
        except Exception as e:
            print(f"[pappas2017] Steiner failed for anchor={req.anchor_id}: {e}")
            return package_result(
                anchor_id=req.anchor_id, anchor_kind=anchor_kind, K=K,
                solution_nodes=list(terms), solution_edges=[], sum_weight=0.0,
                perf=perf, top_k_paths=req.top_k_paths,
                metadata={**req.metadata, "baseline": "pappas2017",
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
        metadata={**req.metadata, "baseline": "pappas2017",
                  "budget": budget, "n_terminals": len(terms)},
    )
