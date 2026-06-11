"""
Shared helpers used by every summarization algorithm.

These are deliberately short and dependency-light: each algorithm imports just
what it needs. Anything specific to one algorithm lives in its own module.
"""

from __future__ import annotations
import time
import os
import resource
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import networkx as nx


# ---------------------------------------------------------------------------
# Anchor-Terminal abstraction
# ---------------------------------------------------------------------------
# An "anchor" is the input node a summary is requested for: a single user
# (user-centric), single item (item-centric), or a synthetic group node
# (user-group / item-group). "Terminals" are the nodes that must end up in the
# summary: anchor + recommended/relevant items. The original ICDE paper used
# different abstractions for each scenario; the TKDE extension unifies them.

@dataclass
class AnchorRequest:
    anchor_id: str                # user_id or item_id (or group_label)
    terminals: list[str]          # set of nodes that must be reached
    top_k_paths: list[list[list]] # each path is [[rel,type,node], ...]
    metadata: dict = field(default_factory=dict)  # gender, popularity quartile, etc.


# ---------------------------------------------------------------------------
# Edge-frequency reweighting (the WPCST step)
# ---------------------------------------------------------------------------

def edge_frequency(top_k_paths: Iterable[list[list]]) -> dict[tuple[str, str], int]:
    """Count how many of the top-K paths each (u,v) edge appears on.

    Each path is a list of triples [relation, node_type, node_id]; the edge
    sequence is implicit between consecutive nodes.
    """
    freq: dict[tuple[str, str], int] = defaultdict(int)
    for path in top_k_paths:
        nodes = [str(step[2]) for step in path]
        for u, v in zip(nodes, nodes[1:]):
            freq[(u, v)] += 1
    return freq


def apply_reweighting(G: nx.DiGraph, freq: dict[tuple[str, str], int],
                      lam: float, K: int) -> tuple[float, float]:
    """In-place edge weight update: w_e = w_base * (1 + lam * f / K).

    Returns (max_w_e, w_avg) which downstream code needs.

    This is the *unified* reweighting referenced in the TKDE extension: all
    three algorithms (ST, PCST, WPCST) operate on the same reweighted graph,
    differing only in how they pick the subset.
    """
    max_w = -float("inf")
    sum_w = 0.0
    n = 0
    for u, v in G.edges():
        base = G[u][v].get("w_base", G[u][v].get("weight", 0.0))
        f_uv = freq.get((u, v), 0) + freq.get((v, u), 0)  # tolerate direction
        w_e = base * (1.0 + lam * (f_uv / float(K)))
        G[u][v]["w_e"] = w_e
        sum_w += w_e
        n += 1
        if w_e > max_w:
            max_w = w_e
    avg = sum_w / n if n else 0.0
    return max_w, avg


def costify(G: nx.DiGraph, max_w: float) -> None:
    """Set edge cost = max_w - w_e, the convention pcst_fast / Steiner expect."""
    for u, v in G.edges():
        G[u][v]["cost"] = max_w - G[u][v]["w_e"]


# ---------------------------------------------------------------------------
# Per-anchor performance tracking
# ---------------------------------------------------------------------------

class PerfTracker:
    """Wall-clock + CPU time + RSS delta.

    Three timings are captured per anchor:
      - elapsed         : wall-clock (time.perf_counter)
      - cpu_time        : user + system CPU consumed by this process,
                          measured with time.process_time. On a 6-worker pool
                          this is the right per-anchor cost metric: wall-clock
                          can be inflated by contention, CPU time cannot.
      - rss_delta_mb    : net RSS increase from getrusage

    `getrusage` is POSIX-only but ML1M and LFM-1M experiments are run on
    Linux/Mac in practice. On Windows, RSS falls back to 0; CPU time still
    works since time.process_time is cross-platform.
    """
    def __init__(self):
        self.t0 = 0.0
        self.cpu0 = 0.0
        self.rss0 = 0
        self.peak_rss = 0
        self.elapsed = 0.0
        self.cpu_time = 0.0
        self.rss_delta_mb = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.cpu0 = time.process_time()
        try:
            self.rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except Exception:
            self.rss0 = 0
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.t0
        self.cpu_time = time.process_time() - self.cpu0
        try:
            rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # ru_maxrss is KB on Linux, bytes on macOS - we convert below
            unit = 1024.0 if _platform_rss_in_kb() else 1024.0 * 1024.0
            self.rss_delta_mb = max(0.0, (rss1 - self.rss0)) / unit
            self.peak_rss = rss1 / unit
        except Exception:
            self.rss_delta_mb = 0.0


def _platform_rss_in_kb() -> bool:
    """Linux reports ru_maxrss in KB; macOS in bytes. Best-effort detection."""
    import sys
    return sys.platform.startswith("linux")


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

def package_result(*, anchor_id: str, anchor_kind: str, K: int,
                   solution_nodes: Iterable[str], solution_edges: Iterable[Iterable[str]],
                   sum_weight: float, perf: PerfTracker, metadata: dict | None = None,
                   top_k_paths: Iterable[Iterable] | None = None) -> dict:
    """Single uniform output schema for every algorithm.

    Downstream metric code reads only this schema; runner-specific shape
    differences (user_id vs item_id) are smoothed out by anchor_kind +
    anchor_id + metadata.

    `top_k_paths` is the original top-K explanation paths; we persist a
    compact node-and-edge summary of them so the coverage/faithfulness
    metrics can be derived in compute_metrics without re-loading the
    recommendation files.
    """
    nodes = list(solution_nodes)
    edges = [list(e) for e in solution_edges]

    # Compact summary of the input paths: per-path node set + edge set.
    # Edges are unordered (canonicalized as sorted tuple) so direction
    # mismatches between graph storage and path traversal don't double-count.
    paths_summary = []
    if top_k_paths is not None:
        for path in top_k_paths:
            seq = [str(step[2]) for step in path]
            path_nodes = list(dict.fromkeys(seq))   # preserve order, dedupe
            path_edges = [sorted([u, v]) for u, v in zip(seq, seq[1:])]
            paths_summary.append({"nodes": path_nodes, "edges": path_edges})

    return {
        "anchor_kind": anchor_kind,           # 'user' | 'item' | 'user_group' | 'item_group'
        "anchor_id": str(anchor_id),
        # Backwards-compatible fields used by the existing metrics calculator:
        "user_id": str(anchor_id) if anchor_kind == "user" else None,
        "item_id": str(anchor_id) if anchor_kind == "item" else None,
        "user_group_name": str(anchor_id) if anchor_kind == "user_group" else None,
        "list_id": str(anchor_id) if anchor_kind == "item_group" else None,
        "k": K,
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "sum_weight": round(float(sum_weight), 6),
        "solution_nodes": nodes,
        "solution_edges": edges,
        "top_k_summary": paths_summary,       # for coverage/faithfulness metrics
        "metadata": metadata or {},
        "performance": {
            "execution_time": round(perf.elapsed, 6),    # wall-clock
            "cpu_time": round(perf.cpu_time, 6),         # user+system CPU
            "memory_usage": round(perf.rss_delta_mb, 4),
            "peak_memory_mb": round(perf.peak_rss, 4),
        },
    }


# ---------------------------------------------------------------------------
# Steiner-then-prune solver (pcst_fast replacement)
# ---------------------------------------------------------------------------
# pcst_fast 1.0.10 has a known bug where its result-extraction step can
# return degenerate solutions (a single vertex repeated many times) on
# graphs with zero-cost edges or large-prize/small-cost regimes.  Older
# versions (1.0.7) do not build on modern toolchains.  Rather than
# depend on an unreliable external library, we implement PCST and WPCST
# via a Steiner-tree-then-pruning heuristic:
#
#   1. Build a Steiner tree connecting the *required* terminals
#      (anchor + recommendation set) using the standard MST-based
#      approximation in NetworkX.
#   2. Iteratively prune subtrees rooted at non-required leaves whose
#      collected prize does not pay for their connecting edge cost.
#
# This loses the 2-approximation guarantee that the Goemans-Williamson
# algorithm provides for PCST, but produces non-degenerate solutions
# at predictable runtime.  The relative comparison among methods
# (uniform vs adaptive prize assignment) is preserved: PCST is solved
# with uniform prizes, WPCST with adaptive (centrality-weighted)
# prizes, and the empirical PCST-vs-WPCST distinction is governed by
# prize design rather than solver choice.
#
# References:
#   Goemans & Williamson 1995 (the original GW algorithm; not used here)
#   Hegde, Indyk, Schmidt, NeurIPS'14 (pcst_fast; not used here due to
#       the result-extraction bug in v1.0.10)
#   Kou, Markowsky, Berman 1981 (the MST-based Steiner approximation
#       which we do use for step 1, via networkx.algorithms.approximation)


def steiner_then_prune(G_und: nx.Graph,
                       required_terms: list[str],
                       prize: dict[str, float],
                       *,
                       cost_attr: str = "cost",
                       weight_attr: str = "w_e") -> tuple[list[str], list[tuple[str, str]], float]:
    """Build a Steiner tree on `required_terms`, then prune leaves whose
    prize does not pay for their connecting edge cost.

    Parameters
    ----------
    G_und : undirected NetworkX graph with `cost_attr` and `weight_attr`
        edge attributes set by `costify` (cost = max_w - w_e).
    required_terms : terminals that *must* appear in the tree (anchor +
        recommended items).  The Steiner step is rooted at these.
    prize : node->prize map.  Used during pruning: a leaf l rooted at
        edge (p, l) is dropped iff `prize.get(l, 0.0) < cost(p,l)`.
        For PCST, prize is 1.0 for terminals and 0.0 elsewhere.
        For WPCST, prize is alpha*c for terminals, beta*c otherwise.
    cost_attr, weight_attr : edge attribute names.

    Returns
    -------
    nodes : list of node ids in the final tree
    edges : list of (u, v) tuples in the final tree
    sum_weight : sum of `weight_attr` over kept edges
    """
    from networkx.algorithms.approximation import steiner_tree

    # Filter required_terms to those present in G_und (in case of group
    # scenarios with terminals from different components).
    terms = [t for t in required_terms if t in G_und]
    if len(terms) < 2:
        return list(terms), [], 0.0

    # Step 1: Steiner approximation.  Use cost_attr as edge weight here
    # (the Steiner objective minimises connecting cost; the prize-driven
    # pruning happens in step 2).
    try:
        T = steiner_tree(G_und, terms, weight=cost_attr)
    except Exception:
        return list(terms), [], 0.0

    # Step 2: prize-based pruning.  Iteratively remove a non-required
    # leaf if its prize does not exceed the cost of the edge attaching
    # it to its parent.  Required terminals are never pruned.
    required_set = set(terms)
    T_work = T.copy()
    changed = True
    while changed:
        changed = False
        # find one prunable leaf
        leaves = [n for n in T_work.nodes if T_work.degree(n) == 1
                  and n not in required_set]
        for leaf in leaves:
            # the leaf has a unique neighbour in a tree
            parent = next(iter(T_work.neighbors(leaf)))
            edge_cost = float(T_work[parent][leaf].get(cost_attr, 0.0))
            leaf_prize = float(prize.get(leaf, 0.0))
            if leaf_prize < edge_cost:
                T_work.remove_node(leaf)
                changed = True
                break  # restart leaf-finding after a structural change

    nodes = list(T_work.nodes)
    edges = list(T_work.edges)
    sum_weight = sum(float(T_work[u][v].get(weight_attr, 0.0))
                     for u, v in edges)
    return nodes, edges, sum_weight
