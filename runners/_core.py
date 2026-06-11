"""
Shared runner machinery: graph loading, recommendation parsing, parallel
execution, output writing.

A `runner` is the script that takes a (dataset, scenario, recommender_baseline,
algorithm) tuple, materializes a list of AnchorRequest, dispatches each to
the chosen algorithm in parallel, and writes results.

Per-scenario runners (run_user_centric.py, run_item_centric.py, etc.) only
have to implement the small bit that is actually scenario-specific: how to
build AnchorRequest from the recommender's output file.
"""

from __future__ import annotations
import json
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Iterable

import networkx as nx

from config import (
    DATASETS, NUM_WORKERS, CHUNK_SIZE_USER, CHUNK_SIZE_ITEM,
    LAMBDA, GAMMA, TOP_K, RESULTS_ROOT,
)
from algorithms import (
    AnchorRequest, PcstIndex,
    run_steiner, run_pcst, run_wpcst,
    run_naive_union, run_pappas2017,
    run_mst_summary, run_faces_summary, run_supernode_summary,
)
from centralities import get_or_compute


# ---------------------------------------------------------------------------
# Graph loading + cost preparation (one-time per dataset)
# ---------------------------------------------------------------------------

def load_graph(dataset: str) -> nx.DiGraph:
    """Load the GraphML and ensure it's a DiGraph with a base weight on each edge."""
    cfg = DATASETS[dataset]
    path: Path = cfg["graphml"]
    if not path.exists():
        raise FileNotFoundError(
            f"Graph file not found: {path}. Place the dataset under data/{dataset}/"
        )
    print(f"[runner] reading {path}")
    t0 = time.time()
    G0 = nx.read_graphml(path, node_type=str)
    if not isinstance(G0, nx.DiGraph):
        G = nx.DiGraph()
        for u, v, data in G0.edges(data=True):
            G.add_edge(str(u), str(v), **data)
    else:
        G = G0
    # Cache w_base (= original 'weight') on every edge.
    for u, v in G.edges():
        G[u][v]["w_base"] = float(G[u][v].get("weight", 0.0))
    print(f"[runner] graph loaded in {time.time() - t0:.1f}s "
          f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    return G


# ---------------------------------------------------------------------------
# Recommendation file parsing
# ---------------------------------------------------------------------------

def _rec_to_graph_uid_map(dataset: str) -> dict[str, str] | None:
    """Inverse of sampling._load_uid_mapping: rec-space uid -> graph uid.

    Built from the same ``uid_mapping_file`` (TSV header ``new_id<TAB>uid``;
    rec id = ``u<new_id + uid_mapping_offset>``).  Returns None when the
    dataset declares no mapping (e.g. ML1M), in which case no translation
    is applied anywhere.

    Why this exists: the sampler translates graph IDs into rec-space so the
    sample CSV matches the recommendation files, but the *requests* are then
    built entirely in rec-space, while the KG carries the real user IDs.
    Without the inverse translation (i) the anchor user is silently dropped
    from the terminal set (``t in G`` fails) and never appears in its own
    summary, and (ii) every path edge incident to a user misses the graph,
    so ``edge_frequency`` is 0 everywhere and the path-aware reweighting
    (lambda, Eq. 2) is a no-op.
    """
    cfg = DATASETS[dataset]
    map_path = cfg.get("uid_mapping_file")
    if map_path is None:
        return None
    map_path = Path(str(map_path))
    if not map_path.exists():
        print(f"[runner] WARNING: uid_mapping_file declared but not found at "
              f"{map_path}; anchors/paths stay in rec ID space.")
        return None
    offset = cfg.get("uid_mapping_offset", 1)
    inv: dict[str, str] = {}
    with open(map_path, "r", encoding="utf-8") as fin:
        header = fin.readline().strip().split("\t")
        try:
            new_id_col = header.index("new_id")
            uid_col = header.index("uid")
        except ValueError:
            print(f"[runner] WARNING: uid_mapping_file header {header!r} "
                  "missing 'new_id'/'uid' columns; no translation.")
            return None
        for line in fin:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < max(new_id_col, uid_col) + 1:
                continue
            inv[f"u{int(parts[new_id_col]) + offset}"] = f"u{parts[uid_col]}"
    print(f"[runner] {dataset}: loaded UID mapping with {len(inv)} entries "
          f"(rec -> graph)")
    return inv


def load_user_recs(dataset: str, baseline: str) -> dict[str, dict]:
    """Load the recommendations.jsonl for a recommender baseline.

    Returns: {user_id: record}, where record has keys 'recommended_items_ids'
    and 'top_k_paths' (matching the original schema).
    """
    cfg = DATASETS[dataset]
    template = cfg["user_recs_template"]
    path = Path(str(template).format(baseline=baseline))
    if not path.exists():
        raise FileNotFoundError(
            f"Recommendations file not found: {path}. "
            f"Either {baseline} is not trained on {dataset}, or the file lives elsewhere."
        )
    out: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            r = json.loads(line)
            out[str(r["user_id"])] = r
    print(f"[runner] {dataset}/{baseline}: loaded {len(out)} user records")

    # Translate to graph ID space when the dataset uses a remapped rec ID
    # space (LFM-1M).  Records stay keyed by rec-space user_id so they match
    # the sample CSV; `graph_user_id` carries the graph-space anchor, and
    # path node IDs found in the mapping (user nodes) are rewritten in
    # place so edge_frequency keys can hit graph edges.  IDs not in the
    # mapping (items, external entities) are untouched.  No-op for ML1M.
    inv = _rec_to_graph_uid_map(dataset)
    if inv:
        n_anchor = n_path = 0
        for r in out.values():
            rid = str(r["user_id"])
            gid = inv.get(rid)
            r["graph_user_id"] = gid if gid is not None else rid
            if gid is not None:
                n_anchor += 1
            for p in r.get("top_k_paths", []) or []:
                for step in p:
                    nid = str(step[2])
                    if nid in inv:
                        step[2] = inv[nid]
                        n_path += 1
        print(f"[runner] {dataset}/{baseline}: translated {n_anchor} anchors "
              f"and {n_path} path user-nodes to graph ID space")
    return out


def load_item_paths(dataset: str, baseline: str) -> dict[str, dict]:
    """Load the per-item path file used by item-centric/item-group scenarios."""
    cfg = DATASETS[dataset]
    template = cfg["item_paths_template"]
    if template is None:
        raise FileNotFoundError(
            f"Item paths not available for {dataset}. "
            "Item-centric / item-group scenarios are skipped on this dataset."
        )
    path = Path(str(template).format(baseline=baseline))
    if not path.exists():
        raise FileNotFoundError(f"Item paths file not found: {path}")
    out: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            r = json.loads(line)
            out[str(r["item_id"])] = r
    print(f"[runner] {dataset}/{baseline}: loaded {len(out)} item-path records")
    return out


# ---------------------------------------------------------------------------
# Algorithm dispatch
# ---------------------------------------------------------------------------

class _Dispatcher:
    """Picklable wrapper that binds an algorithm + its configuration.

    multiprocessing.Pool.map pickles the function it sends to workers, and
    lambdas / closures aren't picklable. This top-level class is.

    The graph G, index, centrality table, etc. are bound at __init__ and
    pickled once per worker spawn. For our 6-core pool that's an O(|V|+|E|)
    one-time cost; downstream calls just send the (small) AnchorRequest.
    """
    def __init__(self, algorithm: str, *, G: nx.DiGraph, index: PcstIndex | None,
                 centrality: dict[str, float] | None,
                 G_und: nx.Graph | None,
                 K: int, lam: float, gamma: float,
                 anchor_kind: str, use_root: bool,
                 budget: int | None,
                 dataset: str | None = None,
                 dterm_cache_dir: Path | None = None):
        self.algorithm = algorithm
        self.G = G
        self.index = index
        self.centrality = centrality
        self.G_und = G_und
        self.K = K
        self.lam = lam
        self.gamma = gamma
        self.anchor_kind = anchor_kind
        self.use_root = use_root
        self.budget = budget
        self.dataset = dataset
        self.dterm_cache_dir = dterm_cache_dir
        # `_dterm_inited` is intentionally not in __init__ so that pickling
        # the dispatcher into worker processes resets it; each worker then
        # initialises the cache lazily on its first call.

    def _ensure_dterm_cache_inited(self) -> None:
        """Idempotent per-process D_term cache init."""
        if getattr(self, "_dterm_inited", False):
            return
        if self.algorithm == "wpcst" and self.dataset is not None:
            from algorithms.wpcst import configure_dterm_cache
            configure_dterm_cache(self.dterm_cache_dir, self.dataset)
        self._dterm_inited = True

    def __call__(self, req: AnchorRequest) -> dict:
        self._ensure_dterm_cache_inited()
        a = self.algorithm
        if a == "st":
            return run_steiner(self.G, req, lam=self.lam, K=self.K,
                               anchor_kind=self.anchor_kind)
        if a == "pcst":
            return run_pcst(self.G, req, lam=self.lam, K=self.K,
                            anchor_kind=self.anchor_kind, index=self.index,
                            G_und=self.G_und,
                            root_node=req.anchor_id if self.use_root else None)
        if a == "wpcst":
            return run_wpcst(self.G, req, lam=self.lam, K=self.K,
                             anchor_kind=self.anchor_kind, index=self.index,
                             centrality=self.centrality, gamma=self.gamma,
                             G_und=self.G_und,
                             root_node=req.anchor_id if self.use_root else None,
                             dataset=self.dataset)
        if a == "naive_union":
            return run_naive_union(self.G, req, lam=self.lam, K=self.K,
                                   anchor_kind=self.anchor_kind)
        if a == "pappas2017":
            return run_pappas2017(self.G, req, lam=self.lam, K=self.K,
                                  anchor_kind=self.anchor_kind,
                                  importance=self.centrality,
                                  budget=self.budget,
                                  G_und=self.G_und)
        if a == "mst":
            return run_mst_summary(self.G, req, lam=self.lam, K=self.K,
                                   anchor_kind=self.anchor_kind,
                                   importance=self.centrality,
                                   budget=self.budget,
                                   G_und=self.G_und)
        if a == "faces":
            return run_faces_summary(self.G, req, lam=self.lam, K=self.K,
                                     anchor_kind=self.anchor_kind,
                                     importance=self.centrality,
                                     budget=self.budget,
                                     G_und=self.G_und)
        if a == "supernode":
            return run_supernode_summary(self.G, req, lam=self.lam, K=self.K,
                                         anchor_kind=self.anchor_kind,
                                         importance=self.centrality,
                                         budget=self.budget,
                                         G_und=self.G_und)
        raise ValueError(f"Unknown algorithm: {a!r}")


def make_dispatcher(algorithm: str, *, G: nx.DiGraph, index: PcstIndex,
                    centrality: dict[str, float] | None,
                    G_und: nx.Graph | None,
                    K: int, lam: float, gamma: float,
                    anchor_kind: str,
                    use_root: bool,
                    budget: int | None = None,
                    dataset: str | None = None,
                    dterm_cache_dir: Path | None = None) -> Callable[[AnchorRequest], dict]:
    """Build a picklable callable that runs `algorithm` on each AnchorRequest.

    Validates required inputs and returns a `_Dispatcher` instance.

    `dataset` and `dterm_cache_dir` are forwarded to WPCST's per-process
    D_term cache (multiprocessing-safe via per-PID pickle files).  See
    `algorithms.wpcst` for the cache mechanism.
    """
    if algorithm == "wpcst" and (centrality is None or G_und is None):
        raise ValueError("WPCST needs `centrality` and `G_und`.")
    if algorithm == "pcst" and G_und is None:
        raise ValueError("PCST needs `G_und` (steiner_then_prune solver).")
    if algorithm in ("pappas2017", "mst", "faces"):
        if centrality is None:
            raise ValueError(f"{algorithm} needs an importance map (`centrality` argument).")
        if budget is None:
            raise ValueError(f"{algorithm} needs `budget`.")
    valid = {"st", "pcst", "wpcst", "naive_union",
             "pappas2017", "mst", "faces", "supernode"}
    if algorithm not in valid:
        raise ValueError(f"Unknown algorithm: {algorithm!r}")

    return _Dispatcher(
        algorithm, G=G, index=index, centrality=centrality, G_und=G_und,
        K=K, lam=lam, gamma=gamma, anchor_kind=anchor_kind,
        use_root=use_root, budget=budget, dataset=dataset,
        dterm_cache_dir=dterm_cache_dir,
    )


# ---------------------------------------------------------------------------
# Parallel execution + output writing
# ---------------------------------------------------------------------------

def execute(requests: list[AnchorRequest], dispatch: Callable[[AnchorRequest], dict],
            *, anchor_kind: str, n_workers: int = NUM_WORKERS) -> list[dict]:
    """Map dispatch over requests in a process pool.

    Falls back to a serial loop if `n_workers <= 1`. The dispatch closure
    captures the (potentially large) graph; multiprocessing pickles it once
    per worker.

    For WPCST runs, this also handles D_term cache flush+merge: each
    worker writes its own per-PID pickle when it exits, and we then merge
    them into the canonical `<dataset>_dterm.pkl` so subsequent runs (e.g.
    alt-centrality) hit the cache.
    """
    chunksize = CHUNK_SIZE_ITEM if "item" in anchor_kind else CHUNK_SIZE_USER

    is_wpcst = isinstance(dispatch, _Dispatcher) and dispatch.algorithm == "wpcst" \
                and dispatch.dterm_cache_dir is not None and dispatch.dataset is not None

    if n_workers <= 1:
        # Serial path: cache lives in the parent process.  Make sure it's
        # initialised, run, then flush+merge.
        if is_wpcst:
            dispatch._ensure_dterm_cache_inited()
        results = [dispatch(r) for r in requests]
        if is_wpcst:
            from algorithms.wpcst import flush_dterm_cache, merge_dterm_caches
            flush_dterm_cache()
            merge_dterm_caches(dispatch.dterm_cache_dir, dispatch.dataset)
        return results

    # Parallel path.  Each worker initialises and flushes its own cache via
    # the initializer/finalizer hooks below.
    initializer = None
    initargs = ()
    if is_wpcst:
        initializer = _wpcst_worker_init
        initargs = (dispatch.dterm_cache_dir, dispatch.dataset)

    with Pool(processes=n_workers, initializer=initializer, initargs=initargs) as pool:
        results = pool.map(dispatch, requests, chunksize=chunksize)
        # Ask each worker to flush before the pool tears down.
        if is_wpcst:
            n = pool._processes
            pool.map(_wpcst_worker_flush, [None] * (n * 4), chunksize=1)
    if is_wpcst:
        from algorithms.wpcst import merge_dterm_caches
        merge_dterm_caches(dispatch.dterm_cache_dir, dispatch.dataset)
    return [r for r in results if r is not None]


def _wpcst_worker_init(cache_dir: Path, dataset: str) -> None:
    """Pool initializer hook: load D_term cache once per worker."""
    from algorithms.wpcst import configure_dterm_cache
    configure_dterm_cache(cache_dir, dataset)


def _wpcst_worker_flush(_: object) -> None:
    """Pool task hook: ask current worker to flush its D_term cache.
    The argument is ignored; we send one task per (worker x N) so each
    worker is hit at least once before the pool closes."""
    from algorithms.wpcst import flush_dterm_cache
    flush_dterm_cache()


def output_path(dataset: str, scenario: str, baseline: str, algorithm: str,
                centrality: str | None = None, tag: str | None = None) -> Path:
    """Canonical results path. The (dataset, scenario, baseline, algorithm,
    centrality) tuple is encoded in the filename so nothing is overwritten
    silently when a new variant is added."""
    parts = [dataset, scenario, baseline, algorithm]
    if centrality and algorithm in ("wpcst", "pappas2017", "mst", "faces"):
        parts.append(f"cent_{centrality}")
    if tag:
        parts.append(tag)
    fname = "_".join(parts) + ".jsonl"
    out_dir = RESULTS_ROOT / dataset / scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / fname


def write_results(path: Path, results: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fout:
        for r in results:
            fout.write(json.dumps(r) + "\n")
    print(f"[runner] wrote {path}")
