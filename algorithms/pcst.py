"""
Prize-Collecting Steiner Tree (PCST) summary explanation.

Implements Algorithm 2 from `algorithms.tex`: PCST with static
graph-level prizes
    α = max{w(e) : e ∈ E},   β = min{w(e) : e ∈ E},
    p(v) = α for v ∈ T*, p(v) = β otherwise,
solved with the Goemans-Williamson primal-dual 2-approximation
algorithm via the `pcst_fast` library.

Solver
------
Uses pcst_fast (Hegde, Indyk, Schmidt; NeurIPS'14).  The 1.0.10
release on PyPI has a Python-binding bug that corrupts result arrays;
we ship a one-line patch in `scripts/install_pcst_fast.sh` and validate
the install via `pcst_fast_compat.py`.

WPCST (next module) extends PCST with adaptive, centrality-weighted
prizes; the solver is identical.
"""

from __future__ import annotations

import numpy as np
import networkx as nx

from ._common import (
    AnchorRequest,
    edge_frequency,
    apply_reweighting,
    costify,
    PerfTracker,
    package_result,
)
from .pcst_fast_compat import pcst_fast_solve


class PcstIndex:
    """Reusable node/edge index for PCST and WPCST.

    Built once per dataset.  Without this, each anchor would pay an
    O(|V|+|E|) overhead to build the integer-edge array pcst_fast wants
    as input.
    """
    def __init__(self, G: nx.DiGraph):
        self.nodes = list(G.nodes())
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        u_idx, v_idx, keys = [], [], []
        # pcst_fast operates on an undirected edge list.  Canonicalise
        # by taking each ordered pair from G.edges() exactly once; if
        # the graph contains both (u,v) and (v,u), the two will become
        # two parallel undirected edges in the input array, which
        # pcst_fast handles correctly (it just sees two edges with
        # potentially different costs).
        for u, v in G.edges():
            u_idx.append(self.node_to_idx[u])
            v_idx.append(self.node_to_idx[v])
            keys.append((u, v))
        self.edge_arr = np.column_stack([
            np.array(u_idx, dtype=np.int64),
            np.array(v_idx, dtype=np.int64),
        ])
        self.edge_keys = keys


def run_pcst(G: nx.DiGraph, req: AnchorRequest, *, lam: float, K: int,
             anchor_kind: str, index: PcstIndex,
             root_node: str | None = None,
             G_und: nx.Graph | None = None) -> dict:
    """Single-anchor PCST with static graph-level prizes via Goemans-Williamson.

    Per Algorithm 2 in algorithms.tex, the prize design is
        α = max{w(e) : e ∈ E},   β = min{w(e) : e ∈ E}
        p(v) = α for v ∈ T* (anchor + terminals), p(v) = β otherwise
    where w(e) are the path-aware reweighted edge weights of
    Equation (1) in the paper.

    `G_und` is accepted for API compatibility with WPCST and the other
    structural baselines but is not used here: pcst_fast operates on the
    integer edge array stored in `index`.
    """
    with PerfTracker() as perf:
        freq = edge_frequency(req.top_k_paths)
        max_w, _ = apply_reweighting(G, freq, lam=lam, K=K)
        if max_w == -float("inf"):
            return package_result(
                anchor_id=req.anchor_id, anchor_kind=anchor_kind, K=K,
                solution_nodes=[], solution_edges=[], sum_weight=0.0,
                perf=perf, top_k_paths=req.top_k_paths,
                metadata=req.metadata,
            )
        # Reset edge costs against the per-anchor reweighted graph.
        costify(G, max_w)

        # Static graph-level prizes per Algorithm 2 in algorithms.tex:
        #   α = max{w(e) : e ∈ E},   β = min{w(e) : e ∈ E}
        #   p(v) = α  for v ∈ T*  (anchor + terminals)
        #   p(v) = β  for v ∉ T*
        #
        # The high terminal prize favours including all terminals; the
        # small non-terminal prize lets non-terminals enter the tree as
        # Steiner nodes when they help connect terminals through high-
        # weight edges.  Both prizes are read from the *adjusted* weights
        # w(e) (the path-aware reweighted values stored in `w_e`), not
        # from the base weights.
        w_e_iter = (G[u][v]["w_e"] for u, v in index.edge_keys)
        first = next(w_e_iter, None)
        if first is None:
            alpha_prize = 0.0
            beta_prize = 0.0
        else:
            alpha_prize = first
            beta_prize = first
            for w in w_e_iter:
                if w > alpha_prize:
                    alpha_prize = w
                if w < beta_prize:
                    beta_prize = w
        # Guard against degenerate or negative β (the GW solver requires
        # non-negative prizes).  In practice w_base is non-negative and
        # the path-aware reweighting only scales it up, so β >= 0; this
        # is a defensive clamp.
        if beta_prize < 0.0:
            beta_prize = 0.0

        terms = {req.anchor_id, *req.terminals}
        prizes = np.full(len(index.nodes), beta_prize, dtype=np.float64)
        for t in terms:
            if t in index.node_to_idx:
                prizes[index.node_to_idx[t]] = alpha_prize


        costs = np.fromiter(
            (G[u][v]["cost"] for u, v in index.edge_keys),
            dtype=np.float64, count=len(index.edge_keys),
        )

        root_idx = index.node_to_idx.get(root_node, -1) if root_node else -1
        try:
            node_ids, edge_ids = pcst_fast_solve(
                index.edge_arr, prizes, costs,
                root=root_idx, num_clusters=1, pruning="strong",
            )
        except Exception as e:
            print(f"[pcst] solver failed for anchor={req.anchor_id}: {e}")
            return package_result(
                anchor_id=req.anchor_id, anchor_kind=anchor_kind, K=K,
                solution_nodes=[], solution_edges=[], sum_weight=0.0,
                perf=perf, top_k_paths=req.top_k_paths,
                metadata=req.metadata,
            )

        chosen_nodes = [index.nodes[int(i)] for i in node_ids]
        chosen_edges = [index.edge_keys[int(i)] for i in edge_ids]
        sum_weight = sum(G[u][v]["w_e"] for u, v in chosen_edges)

    return package_result(
        anchor_id=req.anchor_id, anchor_kind=anchor_kind, K=K,
        solution_nodes=chosen_nodes, solution_edges=chosen_edges,
        sum_weight=sum_weight, perf=perf, top_k_paths=req.top_k_paths,
        metadata={**req.metadata, "solver": "pcst_fast_gw",
                  "alpha": round(float(alpha_prize), 4),
                  "beta": round(float(beta_prize), 4)},
    )


# Kept as a compatibility shim used by wpcst.py for re-projecting cost
# attributes onto the cached G_und view.  WPCST still benefits from G_und
# for D_term computation; PCST itself does not.
def _project_costs(G: nx.DiGraph, UG: nx.Graph) -> None:
    for u, v in UG.edges():
        if G.has_edge(u, v):
            UG[u][v]["cost"] = float(G[u][v].get("cost", 0.0))
            UG[u][v]["w_e"]  = float(G[u][v].get("w_e",  0.0))
        elif G.has_edge(v, u):
            UG[u][v]["cost"] = float(G[v][u].get("cost", 0.0))
            UG[u][v]["w_e"]  = float(G[v][u].get("w_e",  0.0))
