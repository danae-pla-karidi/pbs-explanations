from __future__ import annotations
import itertools
import hashlib
import os
import pickle
import threading
from pathlib import Path
from typing import Optional

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
from .pcst import PcstIndex, _project_costs
from .pcst_fast_compat import pcst_fast_solve


# ---------------------------------------------------------------------------
# D_term cache (per-process)
# ---------------------------------------------------------------------------

_DTERM_CACHE: dict[str, float] = {}
_DTERM_CACHE_LOCK = threading.Lock()
_DTERM_CACHE_DIR: Optional[Path] = None
_DTERM_CACHE_DATASET: Optional[str] = None
_DTERM_CACHE_DIRTY = False


def configure_dterm_cache(cache_dir: Optional[Path], dataset: str) -> None:
    """Initialise the D_term cache for this process.  See module
    docstring for the multiprocessing strategy."""
    global _DTERM_CACHE, _DTERM_CACHE_DIR, _DTERM_CACHE_DATASET, _DTERM_CACHE_DIRTY
    with _DTERM_CACHE_LOCK:
        _DTERM_CACHE_DIR = Path(cache_dir) if cache_dir is not None else None
        _DTERM_CACHE_DATASET = dataset
        _DTERM_CACHE = {}
        _DTERM_CACHE_DIRTY = False
        if _DTERM_CACHE_DIR is None:
            return
        merged = _DTERM_CACHE_DIR / f"{dataset}_dterm.pkl"
        if merged.exists():
            try:
                with open(merged, "rb") as f:
                    _DTERM_CACHE = pickle.load(f)
            except Exception as e:
                print(f"[wpcst] failed to load D_term cache {merged}: {e!r}")
                _DTERM_CACHE = {}


def flush_dterm_cache() -> None:
    """Write this process's local cache to its per-PID pickle file."""
    global _DTERM_CACHE_DIRTY
    with _DTERM_CACHE_LOCK:
        if not _DTERM_CACHE_DIRTY or _DTERM_CACHE_DIR is None or _DTERM_CACHE_DATASET is None:
            return
        _DTERM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _DTERM_CACHE_DIR / f"{_DTERM_CACHE_DATASET}_dterm_pid{os.getpid()}.pkl"
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(_DTERM_CACHE, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
        _DTERM_CACHE_DIRTY = False


def merge_dterm_caches(cache_dir: Path, dataset: str) -> None:
    """Merge per-PID pickles into the canonical `<dataset>_dterm.pkl`."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return
    merged: dict[str, float] = {}
    canonical = cache_dir / f"{dataset}_dterm.pkl"
    if canonical.exists():
        try:
            with open(canonical, "rb") as f:
                merged = pickle.load(f)
        except Exception:
            merged = {}
    pid_files = list(cache_dir.glob(f"{dataset}_dterm_pid*.pkl"))
    for p in pid_files:
        try:
            with open(p, "rb") as f:
                merged.update(pickle.load(f))
        except Exception as e:
            print(f"[wpcst] skipping malformed cache file {p}: {e!r}")
    tmp = canonical.with_suffix(canonical.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(canonical)
    for p in pid_files:
        try:
            p.unlink()
        except OSError:
            pass
    print(f"[wpcst] merged {len(pid_files)} per-PID caches into {canonical} ({len(merged)} entries)")


def _dterm_cache_key(dataset: str, anchor_id: str, terminals: list[str],
                     K: int, top_k_paths: list, lam: float) -> str:
    h = hashlib.blake2b(digest_size=8)
    for path in top_k_paths:
        for step in path:
            for elem in step:
                h.update(str(elem).encode("utf-8"))
            h.update(b"|")
        h.update(b"#")
    paths_hash = h.hexdigest()
    terminals_sig = ",".join(map(str, terminals))
    # lam is part of the key: D_term is measured under the transformed
    # costs (max_w - w~(e)), and w~(e) depends on lam (Eq. 2).  Reusing a
    # D_term across lambda values (e.g. in the lambda ablation) would
    # contaminate the adaptive prize alpha.  Main-pipeline runs all use
    # lam=1, so cache reuse across centrality variants is unaffected.
    # The dterm-v2-wprime prefix versions the definition (distances under
    # transformed costs, anchors excluded); v1 values are never reused.
    return f"dterm-v2-wprime|{dataset}|{anchor_id}|K={K}|lam={lam:g}|t={terminals_sig}|p={paths_hash}"


# ---------------------------------------------------------------------------
# Distance computation
# ---------------------------------------------------------------------------

def _avg_terminal_distance_uncached(G_und: nx.Graph, terminals: list[str]) -> float:
    """Average pairwise shortest-path distance under the transformed
    costs (`cost` = max_w - w_e), the currency the solver pays.

    Restricted to the connected component containing the first terminal;
    pairs in different components are skipped.  Returns -1.0 (sentinel)
    when no pairwise distance can be computed — the caller substitutes
    the C_avg fallback per Algorithm 3 in algorithms.tex
    (`if |T|=1 or no pair is connected, set D_term = C_avg`).  A
    sentinel is required because a legitimate D_term of exactly 0.0 is
    possible under the costs (a chain of maximum-relevance edges).
    """
    if len(terminals) < 2:
        return -1.0
    root = terminals[0]
    if root not in G_und:
        return -1.0
    cc = nx.node_connected_component(G_und, root)
    sub = G_und.subgraph(cc)
    total, n = 0.0, 0
    for a, b in itertools.combinations(terminals, 2):
        if a not in sub or b not in sub:
            continue
        try:
            total += nx.shortest_path_length(sub, a, b, weight="cost")
            n += 1
        except nx.NetworkXNoPath:
            pass
    return total / n if n else -1.0


def _avg_terminal_distance(G_und: nx.Graph, terminals: list[str], *,
                           cache_key: Optional[str] = None) -> tuple[float, bool]:
    global _DTERM_CACHE_DIRTY
    if cache_key is None:
        return _avg_terminal_distance_uncached(G_und, terminals), False
    with _DTERM_CACHE_LOCK:
        cached = _DTERM_CACHE.get(cache_key)
    if cached is not None:
        return float(cached), True
    val = _avg_terminal_distance_uncached(G_und, terminals)
    with _DTERM_CACHE_LOCK:
        _DTERM_CACHE[cache_key] = val
        _DTERM_CACHE_DIRTY = True
    return val, False


# ---------------------------------------------------------------------------
# Adaptive prize assignment
# ---------------------------------------------------------------------------

def assign_prizes(G: nx.DiGraph, terminals: set[str], centrality: dict[str, float],
                  *, gamma: float, c_avg: float, G_und: nx.Graph,
                  cache_key: Optional[str] = None
                  ) -> tuple[float, float, float, bool, bool]:
    """Compute (alpha, beta, d_term, cache_hit, fallback) per Algorithm 3.

    `terminals` is the distance set T (anchors excluded by the caller).
    Distances are measured under the transformed costs.  If no terminal
    pair admits a finite distance (singleton terminal set, or no pair
    connected), we fall back to `D_term <- C_avg`, matching the paper,
    which yields alpha = 2.
    """
    d_term, hit = _avg_terminal_distance(G_und, list(terminals), cache_key=cache_key)
    fallback = d_term < 0.0
    if fallback:
        d_term = c_avg
    alpha = 1.0 + (d_term / c_avg) if abs(c_avg) > 1e-9 else 2.0
    beta = gamma * alpha
    return alpha, beta, d_term, hit, fallback


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run_wpcst(G: nx.DiGraph, req: AnchorRequest, *, lam: float, K: int,
              anchor_kind: str, index: PcstIndex, centrality: dict[str, float],
              gamma: float, G_und: nx.Graph,
              root_node: str | None = None,
              dataset: str | None = None) -> dict:
    """Single-anchor WPCST via Goemans-Williamson with adaptive prizes."""
    with PerfTracker() as perf:
        freq = edge_frequency(req.top_k_paths)
        max_w, w_avg = apply_reweighting(G, freq, lam=lam, K=K)
        if max_w == -float("inf"):
            return package_result(
                anchor_id=req.anchor_id, anchor_kind=anchor_kind, K=K,
                solution_nodes=[], solution_edges=[], sum_weight=0.0,
                perf=perf, top_k_paths=req.top_k_paths,
                metadata=req.metadata,
            )
        costify(G, max_w)
        # Per-anchor costs onto the undirected distance graph.  G_und is a
        # copy made at runner start; without this projection its edges
        # carry no `cost` attribute and distances silently degrade to hop
        # counts (NetworkX substitutes weight 1 for missing attributes).
        _project_costs(G, G_und)
        c_avg = max_w - w_avg

        terms = {req.anchor_id, *(t for t in req.terminals if t in G)}
        if len(terms) < 2:
            return package_result(
                anchor_id=req.anchor_id, anchor_kind=anchor_kind, K=K,
                solution_nodes=list(terms), solution_edges=[], sum_weight=0.0,
                perf=perf, top_k_paths=req.top_k_paths,
                metadata=req.metadata,
            )

        # Distance set per Algorithm 3: terminals only, anchors excluded,
        # since the dispersion of the terminals drives the connection
        # cost.  The anchor keeps its terminal prize via `terms` below.
        dterm_terms = [str(t) for t in req.terminals if t in G]

        # Cache key: include dataset and the anchor's top-K paths so
        # alt-centrality re-runs hit the same value.
        cache_key = (
            _dterm_cache_key(dataset, str(req.anchor_id), dterm_terms, K, req.top_k_paths, lam)
            if dataset is not None
            else None
        )

        alpha, beta, d_term, cache_hit, dterm_fallback = assign_prizes(
            G, dterm_terms, centrality,
            gamma=gamma, c_avg=c_avg, G_und=G_und,
            cache_key=cache_key,
        )

        # Build the prize array aligned with index.nodes, then dispatch
        # to the patched pcst_fast solver.
        prizes = np.empty(len(index.nodes), dtype=np.float64)
        for i, n in enumerate(index.nodes):
            c = centrality.get(n, 0.0)
            prizes[i] = (alpha * c) if n in terms else (beta * c)

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
            print(f"[wpcst] solver failed for anchor={req.anchor_id}: {e}")
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
        sum_weight=sum_weight, perf=perf,
        top_k_paths=req.top_k_paths,
        metadata={**req.metadata, "solver": "pcst_fast_gw",
                  "alpha": round(alpha, 4), "beta": round(beta, 4),
                  "d_term": round(d_term, 4),
                  "dterm_cached": bool(cache_hit),
                  "dterm_fallback": bool(dterm_fallback)},
    )
