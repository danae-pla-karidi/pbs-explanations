"""
Per-anchor metrics computation.

The original `wpcst_metrics_calculator_updated.py` aggregates inside the
metric function and emits a single mean per (scenario, baseline, algorithm).
That loses per-anchor information and makes Wilcoxon impossible from saved
results.

This module computes metrics *per anchor*, writes them to
`results/per_anchor/<dataset>_<scenario>_<baseline>_<algorithm>.parquet`
(one row per anchor), and then aggregates to the canonical summary CSV.
The significance-testing script reads the parquet files.

Metric definitions match the original calculator where the original is
correct, and fix it where it isn't:

- `relevance` is now normalized to [0,1] (mean edge w_e divided by max
  edge w_e in the solution). The original implementation reported raw
  totals, which is why the ICDE table shows values in the hundreds.
- `comprehensibility_gap` is computed only at the aggregate level; it is
  NOT a per-anchor metric.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import defaultdict
import itertools

import numpy as np
import pandas as pd
import networkx as nx

from config import DATASETS, RESULTS_ROOT, METRICS, SAMPLES_DIR


# ---------------------------------------------------------------------------
# Per-anchor metric implementations
# ---------------------------------------------------------------------------

def m_comprehensibility(rec: dict) -> float:
    n_edges = rec.get("num_edges", 0)
    return 1.0 / (n_edges + 1) if n_edges >= 0 else 0.0


def m_relevance(rec: dict, G: nx.DiGraph) -> float:
    """Total $w_M$ weight of edges in the summary subgraph.

    This is the original relevance metric from~\\cite{plakaridi2025}:
        $R(S) = \\sum_{e \\in E_S} w_M(e)$
    It matches the optimization objective in the anchor-terminal
    formulation (Section~\\ref{sec:problem}, "maximize total edge
    weight") and rewards summaries that aggregate stronger user-item
    evidence.  The metric is un-normalized and unbounded above; values
    are not comparable across datasets but are directly comparable
    across algorithms on the same dataset.

    Note that larger summaries with similarly strong edges score
    higher than smaller ones; this is by design and exposes the
    comprehensibility-vs-relevance trade-off as a real Pareto problem.
    NaiveUnion's $R$ is therefore an upper bound by construction
    (it sums over every top-$K$ path edge); we annotate this in the
    headline tables.

    The implementation tolerates the directed/undirected mismatch by
    looking up each edge in either orientation; edges not found in
    $G$ contribute 0.
    """
    edges = rec.get("solution_edges", [])
    if not edges:
        return 0.0
    total = 0.0
    for pair in edges:
        if len(pair) < 2:
            continue
        u, v = str(pair[0]), str(pair[1])
        if G.has_edge(u, v):
            total += float(G[u][v].get("weight", 0.0))
        elif G.has_edge(v, u):
            total += float(G[v][u].get("weight", 0.0))
    return total


def m_actionability(rec: dict, G: nx.DiGraph,
                    all_items: set[str] | None = None) -> float:
    """Fraction of summary nodes that are item nodes.

    ICDE definition (\\cite{plakaridi2025}):
        $A(S) = |\\{v \\in V_S : v \\in I\\}| / |V_S|$
    where $I$ is the set of item nodes in the knowledge graph.
    Item nodes are actionable because the user can modify their
    ratings; user and attribute nodes are not.

    `all_items` is the set of item-typed node IDs in the KG.  If
    omitted, the function falls back to identifying item nodes by
    the lack of a `u`-prefix and the absence of letters (ML1M
    convention: items are numeric IDs); this fallback is correct
    for ML1M but should not be relied on for other datasets.
    """
    nodes = rec.get("solution_nodes", [])
    if not nodes:
        return 0.0
    if all_items is not None:
        n_items = sum(1 for n in nodes if str(n) in all_items)
    else:
        # Fallback heuristic: ML1M item nodes are bare numeric IDs.
        n_items = sum(1 for n in nodes if str(n).isdigit())
    return n_items / len(nodes)


def m_diversity(rec: dict) -> float:
    """Average pairwise (1 - Jaccard) over edge node-sets.

    ICDE definition (\\cite{plakaridi2025}):
        $D(S) = \\frac{1}{\\binom{|E_S|}{2}}
                \\sum_{e_i, e_j \\in E_S} (1 - J(e_i, e_j))$
    where $J(e_i, e_j) = |V_{e_i} \\cap V_{e_j}| / |V_{e_i} \\cup V_{e_j}|$
    is the Jaccard similarity between the two edges' endpoint sets.

    For simple graphs (no multi-edges, no self-loops) every pair of
    distinct edges shares at most one endpoint, so $J \\in \\{0, 1/3\\}$
    and the metric reduces to:
        $D(S) = 1 - \\frac{1}{3} \\cdot
                \\frac{\\sum_v \\binom{d(v)}{2}}{\\binom{|E_S|}{2}}$
    where $d(v)$ is the degree of node $v$ in the summary subgraph.
    This $O(|E_S|)$ implementation is exact for the tree summaries
    produced by all algorithms in this paper.

    For multi-edge summaries (e.g.\\ NaiveUnion if it produces
    duplicates), the simplification is conservative: identical
    edge pairs are counted as ``sharing one endpoint'' rather than
    two, slightly inflating the diversity score relative to the
    exact pairwise definition.  In practice this case does not
    arise on ML1M with the algorithms we evaluate.

    Returns NaN for summaries with $|E_S| < 2$.
    """
    edges = rec.get("solution_edges", [])
    if len(edges) < 2:
        return float("nan")
    from collections import Counter
    deg: Counter = Counter()
    n_edges = 0
    for pair in edges:
        if len(pair) < 2:
            continue
        deg[str(pair[0])] += 1
        deg[str(pair[1])] += 1
        n_edges += 1
    if n_edges < 2:
        return float("nan")
    pairs_sharing_one = sum(d * (d - 1) // 2 for d in deg.values())
    total_pairs = n_edges * (n_edges - 1) // 2
    if total_pairs == 0:
        return float("nan")
    pairs_sharing_zero = total_pairs - pairs_sharing_one
    sum_1_minus_j = pairs_sharing_one * (2.0 / 3.0) + pairs_sharing_zero * 1.0
    return sum_1_minus_j / total_pairs


def m_redundancy(rec: dict) -> float:
    return 1.0 - m_diversity(rec)


def m_privacy(rec: dict) -> float:
    nodes = rec.get("solution_nodes", [])
    if not nodes:
        return 1.0
    user_count = sum(1 for n in nodes if str(n).startswith("u") and str(n)[1:].isdigit())
    return 1.0 - (user_count / len(nodes))


def m_runtime(rec: dict) -> float:
    return float(rec.get("performance", {}).get("execution_time", 0.0))


def m_cpu_time(rec: dict) -> float:
    return float(rec.get("performance", {}).get("cpu_time", 0.0))


def m_memory(rec: dict) -> float:
    return float(rec.get("performance", {}).get("memory_usage", 0.0))


def m_pop_frac(rec: dict, all_items: set[str],
               popular_items: set[str]) -> float:
    """Fraction of item nodes in the summary that are popular.

    'Popular' is defined globally as the top-quartile of item popularity
    in the knowledge graph (computed once over all items, not just the
    sampled ones).  Returns NaN if the summary contains no item nodes;
    NaN cells are ignored in the gap aggregation.
    """
    items_in_summary = [n for n in rec.get("solution_nodes", [])
                        if str(n) in all_items]
    if not items_in_summary:
        return float("nan")
    n_pop = sum(1 for n in items_in_summary if str(n) in popular_items)
    return n_pop / len(items_in_summary)


def m_pop_frac_topk(rec: dict, all_items: set[str],
                    popular_items: set[str]) -> float:
    """Same metric, but computed on the recommender's top-K item set
    (the union of items appearing in `top_k_summary`).  Used as the
    baseline against which the summarizer's pop_frac is compared.
    Returns NaN if the top-K summary contains no items.
    """
    topk_items: set[str] = set()
    for entry in rec.get("top_k_summary", []) or []:
        for n in entry.get("nodes", []):
            if str(n) in all_items:
                topk_items.add(str(n))
    if not topk_items:
        return float("nan")
    n_pop = sum(1 for n in topk_items if n in popular_items)
    return n_pop / len(topk_items)


# ---------------------------------------------------------------------------
# Coverage and faithfulness: how much of the original explanation survives
# ---------------------------------------------------------------------------
# Both metrics use `top_k_summary` (per-path node/edge sets persisted by
# package_result), and the solution_nodes/solution_edges already on the
# record. They answer different questions:
#   - coverage:    fraction of paths preserved as a whole (a path counts as
#                  preserved iff every node and every edge appears in the
#                  summary). Binary-per-path.
#   - faithfulness: fraction of node/edge content preserved, averaged over
#                  the union of all top-K nodes and edges. Fractional.

def _canon_edge(u, v) -> tuple[str, str]:
    a, b = sorted([str(u), str(v)])
    return (a, b)


def m_coverage(rec: dict) -> float:
    """Fraction of original top-K paths fully contained in the summary."""
    paths = rec.get("top_k_summary", []) or []
    if not paths:
        return float("nan")
    sol_nodes = {str(n) for n in rec.get("solution_nodes", [])}
    sol_edges = {_canon_edge(u, v) for u, v in rec.get("solution_edges", [])}
    n_full = 0
    for p in paths:
        path_nodes = {str(n) for n in p.get("nodes", [])}
        path_edges = {_canon_edge(u, v) for u, v in p.get("edges", [])}
        if path_nodes <= sol_nodes and path_edges <= sol_edges:
            n_full += 1
    return n_full / len(paths)


def m_faithfulness(rec: dict) -> float:
    """Fraction of (node ∪ edge) content of the top-K paths preserved.

    Computed as the mean of node-faithfulness and edge-faithfulness:
        node_f = |solution_nodes ∩ path_nodes| / |path_nodes|
        edge_f = |solution_edges ∩ path_edges| / |path_edges|
    where path_nodes / path_edges are the union over all top-K paths.

    Returns the simple mean of the two, treating the case of empty path
    edges (e.g. degenerate length-1 paths) by falling back to node-only.
    """
    paths = rec.get("top_k_summary", []) or []
    if not paths:
        return float("nan")
    sol_nodes = {str(n) for n in rec.get("solution_nodes", [])}
    sol_edges = {_canon_edge(u, v) for u, v in rec.get("solution_edges", [])}

    path_nodes: set[str] = set()
    path_edges: set[tuple[str, str]] = set()
    for p in paths:
        path_nodes.update(str(n) for n in p.get("nodes", []))
        path_edges.update(_canon_edge(u, v) for u, v in p.get("edges", []))

    node_f = len(sol_nodes & path_nodes) / len(path_nodes) if path_nodes else float("nan")
    edge_f = len(sol_edges & path_edges) / len(path_edges) if path_edges else None

    if edge_f is None:
        return node_f
    return (node_f + edge_f) / 2.0


# ---------------------------------------------------------------------------
# Consistency (REMOVED)
# ---------------------------------------------------------------------------
# The original ICDE pipeline included a per-anchor "consistency" metric
# defined as the mean pairwise Jaccard between explanation node-sets that
# share an anchor (e.g. across baselines).  In our refactored pipeline,
# `compute_per_anchor` is invoked once per (scenario, baseline, algorithm,
# centrality) file, so each anchor only ever has one record and the
# Jaccard collapses to the trivial value 1.0.
#
# Rather than rewire the pipeline to read all algorithms simultaneously,
# we drop the metric.  The cross-algorithm comparison it was meant to
# capture is already produced by the CR-tradeoff figure and the Wilcoxon
# contrasts.  The hook function is preserved here only as a stub so that
# downstream code or notebooks that import it still load cleanly.


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def load_records(path: Path) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def compute_per_anchor(records: list[dict], G: nx.DiGraph,
                      *, dataset: str, scenario: str,
                      baseline: str, algorithm: str,
                      centrality: str | None = None,
                      all_items: set[str] | None = None,
                      popular_items: set[str] | None = None) -> pd.DataFrame:
    """Return a DataFrame with one row per anchor.

    `all_items` and `popular_items` are passed in so we can compute
    the popularity-bias metric `pop_frac` (fraction of summary item
    nodes that are popular) and `pop_frac_topk` (the same fraction
    on the recommender's top-K item set, used as a baseline).
    Both default to empty sets, which makes the metrics return NaN.
    """
    if all_items is None:
        all_items = set()
    if popular_items is None:
        popular_items = set()
    rows = []
    for r in records:
        # Real runner output (`package_result`) writes `anchor_id` and one
        # of {user_id, item_id, user_group_name, list_id} for backwards
        # compatibility.  Read the canonical `anchor_id` first and fall
        # back to the legacy fields if it is absent.
        a = (r.get("anchor_id")
             or r.get("user_id")
             or r.get("item_id")
             or r.get("user_group_name")
             or r.get("list_id"))
        if a is None:
            continue
        rows.append({
            "dataset": dataset,
            "scenario": scenario,
            "baseline": baseline,
            "algorithm": algorithm,
            "centrality": centrality or "",
            "anchor_id": str(a),
            "anchor_kind": r.get("anchor_kind", "unknown"),
            "metadata": json.dumps(r.get("metadata", {})),
            "num_nodes": r.get("num_nodes", 0),
            "num_edges": r.get("num_edges", 0),
            "comprehensibility": m_comprehensibility(r),
            "relevance": m_relevance(r, G),
            "actionability": m_actionability(r, G, all_items),
            "diversity": m_diversity(r),
            "redundancy": m_redundancy(r),
            "privacy": m_privacy(r),
            "coverage": m_coverage(r),
            "faithfulness": m_faithfulness(r),
            "runtime_sec": m_runtime(r),
            "cpu_time_sec": m_cpu_time(r),
            "memory_usage_mb": m_memory(r),
            "pop_frac": m_pop_frac(r, all_items, popular_items),
            "pop_frac_topk": m_pop_frac_topk(r, all_items, popular_items),
        })
    return pd.DataFrame(rows)


def aggregate(per_anchor: pd.DataFrame, *,
              user_genders: dict[str, str] | None = None,
              item_quartiles: dict[str, str] | None = None) -> pd.DataFrame:
    """Reduce the per-anchor DataFrame to one row per
    (dataset, scenario, baseline, algorithm, centrality).

    Adds comprehensibility_gap (gender-binary for user-centric;
    popular/unpopular for item-centric, NaN when not applicable) and
    per-metric standard deviations.

    `item_quartiles` is a dict ``anchor_id -> quartile_label`` where the
    labels follow the convention used by ``sampling.make_samples``:
    ``q1_unpopular``, ``q2``, ``q3``, ``q4_popular``.  For item-centric
    cells we contrast q4_popular vs q1_unpopular (25 anchors each on
    a balanced sample); q2/q3 anchors do not contribute to the gap.
    """
    metric_cols = [c for c in METRICS if c in per_anchor.columns]
    grouping = ["dataset", "scenario", "baseline", "algorithm", "centrality"]
    agg = per_anchor.groupby(grouping, dropna=False)[metric_cols].agg(["mean", "std", "count"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    agg = agg.reset_index()

    # Comprehensibility gap.
    # - user_centric: |C̄(M) - C̄(F)|
    # - item_centric: |C̄(popular) - C̄(unpopular)|, restricted to the
    #   q4_popular and q1_unpopular quartiles (25 anchors each).
    # NaN otherwise.
    #
    # We additionally compute two popularity-bias gaps for user-centric
    # cells:
    # - popular_item_gap: |mean(pop_frac | M) - mean(pop_frac | F)|,
    #   the gender disparity in the *summarizer's* popularity uptake.
    # - recommender_pop_frac_gap: same gap on `pop_frac_topk`, the
    #   recommender's own top-K popularity fraction.  This is the
    #   upstream baseline; comparing the two answers the question
    #   "does the summarizer amplify, preserve, or attenuate the
    #   recommender's gender-popularity disparity?".
    if user_genders is not None or item_quartiles is not None:
        gap_rows = []
        for keys, sub in per_anchor.groupby(grouping, dropna=False):
            scenario = keys[1] if isinstance(keys, tuple) else keys
            cg = float("nan")
            pig = float("nan")    # popular_item_gap (summarizer)
            rpg = float("nan")    # recommender_pop_frac_gap (baseline)
            if scenario == "user_centric" and user_genders is not None:
                # Comprehensibility gap on M vs F.
                pairs = []
                pop_pairs = []
                topk_pairs = []
                for _, r in sub.iterrows():
                    g = user_genders.get(r["anchor_id"])
                    if g in ("M", "F"):
                        pairs.append((g, r["comprehensibility"]))
                        if "pop_frac" in r and not pd.isna(r["pop_frac"]):
                            pop_pairs.append((g, float(r["pop_frac"])))
                        if "pop_frac_topk" in r and not pd.isna(r["pop_frac_topk"]):
                            topk_pairs.append((g, float(r["pop_frac_topk"])))
                m_vals = [c for g, c in pairs if g == "M"]
                f_vals = [c for g, c in pairs if g == "F"]
                if m_vals and f_vals:
                    cg = abs(np.mean(m_vals) - np.mean(f_vals))
                m_pop = [c for g, c in pop_pairs if g == "M"]
                f_pop = [c for g, c in pop_pairs if g == "F"]
                if m_pop and f_pop:
                    pig = abs(np.mean(m_pop) - np.mean(f_pop))
                m_topk = [c for g, c in topk_pairs if g == "M"]
                f_topk = [c for g, c in topk_pairs if g == "F"]
                if m_topk and f_topk:
                    rpg = abs(np.mean(m_topk) - np.mean(f_topk))
            elif scenario == "item_centric" and item_quartiles is not None:
                pairs = []
                for _, r in sub.iterrows():
                    q = item_quartiles.get(str(r["anchor_id"]))
                    if q in ("q4_popular", "q1_unpopular"):
                        pairs.append((q, r["comprehensibility"]))
                p_vals = [c for q, c in pairs if q == "q4_popular"]
                u_vals = [c for q, c in pairs if q == "q1_unpopular"]
                if p_vals and u_vals:
                    cg = abs(np.mean(p_vals) - np.mean(u_vals))
            gap_rows.append({**dict(zip(grouping, keys)),
                             "comprehensibility_gap": cg,
                             "popular_item_gap": pig,
                             "recommender_pop_frac_gap": rpg})
        gap_df = pd.DataFrame(gap_rows)
        if not gap_df.empty:
            agg = agg.merge(gap_df, on=grouping, how="left")

    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_KNOWN_ALGORITHMS = (
    # ordered: longer / multi-token names first so prefix matching is unambiguous
    "naive_union",
    "pappas2017", "supernode",
    "wpcst", "pcst", "st",
    "mst", "faces",
)


def parse_filename(path: Path) -> dict:
    """Recover (dataset, scenario, baseline, algorithm, centrality)
    from the canonical output filename produced by `runners/_core.output_path`.

    Filename schema (no extension):
        <dataset>_<scenario_token1>_<scenario_token2>_<baseline>_<algorithm>[_cent_<centrality>][_<tag>]

    Examples that previously confused the parser:
      ml1m_user_centric_pgpr_naive_union          (algorithm has an underscore)
      ml1m_user_centric_pgpr_wpcst_cent_pagerank  (centrality has its own field)
      ml1m_user_centric_pgpr_pappas2017_cent_betweenness_approx
                                                  (centrality has an underscore)
    """
    parts = path.stem.split("_")
    info = {"dataset": parts[0], "scenario": "", "baseline": "", "algorithm": "",
            "centrality": ""}

    # Scenarios are 2-token strings (user_centric, user_group, item_centric, item_group)
    info["scenario"] = "_".join(parts[1:3])
    info["baseline"] = parts[3]

    # The remainder may contain: algorithm (possibly multi-token like
    # 'naive_union'), an optional 'cent' marker followed by a centrality
    # name (possibly multi-token like 'betweenness_approx'), and an
    # optional trailing tag.  Match the algorithm against the known list.
    rest = parts[4:]
    rest_str = "_".join(rest)

    matched_alg = ""
    for alg in _KNOWN_ALGORITHMS:
        if rest_str == alg or rest_str.startswith(alg + "_"):
            matched_alg = alg
            break
    info["algorithm"] = matched_alg

    if matched_alg:
        tail = rest_str[len(matched_alg):].lstrip("_")
        # tail is now '' or 'cent_<centrality>[_<tag>]' or '<tag>'
        if tail.startswith("cent_"):
            cent_and_tag = tail[len("cent_"):]
            # We do not currently emit additional tags, so treat the rest
            # as the centrality string.
            info["centrality"] = cent_and_tag

    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    p.add_argument("--out-summary", default=None,
                   help="Path to write the aggregated summary CSV; "
                        "default = results/<dataset>/summary.csv")
    args = p.parse_args()

    cfg = DATASETS[args.dataset]
    G = nx.read_graphml(cfg["graphml"], node_type=str)
    if not isinstance(G, nx.DiGraph):
        G = nx.DiGraph(G)

    # User-gender lookup for user-centric comprehensibility gap.
    user_genders = {}
    for n, data in G.nodes(data=True):
        if str(n).startswith("u"):
            g = data.get("gender") or data.get("d2")
            if g in ("M", "F"):
                user_genders[str(n)] = g

    # Overlay genders from the sample CSV, translated to graph ID space.
    # Rationale: the sampler normalizes gender labels into
    # samples/<dataset>_users.csv (handling KG dumps that store gender
    # under other attributes or encodings), so for the sampled anchors the
    # CSV is the source of truth.  Its user_id column is in rec ID space
    # for remapped datasets (LFM-1M); translate via the same inverse UID
    # map the runners use so the keys join against graph-space anchor_ids.
    # No-op when the CSV is absent; identity translation for ML1M.
    users_csv = SAMPLES_DIR / f"{args.dataset}_users.csv"
    if users_csv.exists():
        try:
            from runners._core import _rec_to_graph_uid_map
            inv = _rec_to_graph_uid_map(args.dataset) or {}
            users_df = pd.read_csv(users_csv)
            n_added = 0
            for _, r in users_df.iterrows():
                if str(r.get("gender")) in ("M", "F"):
                    user_genders[inv.get(str(r["user_id"]), str(r["user_id"]))] = \
                        str(r["gender"])
                    n_added += 1
            print(f"[metrics] gender overlay from {users_csv.name}: "
                  f"{n_added} sampled users ({len(user_genders)} total)")
        except Exception as e:
            print(f"[metrics] WARNING: gender overlay failed ({e}); "
                  "falling back to graph attributes only.")

    # Item-quartile lookup for item-centric comprehensibility gap.
    # The sample CSV produced by `sampling.make_samples` records each
    # sampled item's popularity quartile.  We pass it through to
    # `aggregate` so item-centric cells can compute |C̄(popular) -
    # C̄(unpopular)| restricted to the q4_popular vs q1_unpopular split.
    item_quartiles: dict[str, str] = {}
    items_csv = SAMPLES_DIR / f"{args.dataset}_items.csv"
    if items_csv.exists():
        try:
            items_df = pd.read_csv(items_csv)
            if "quartile" in items_df.columns and "item_id" in items_df.columns:
                for _, r in items_df.iterrows():
                    item_quartiles[str(r["item_id"])] = str(r["quartile"])
                print(f"[metrics] loaded {len(item_quartiles)} item "
                      f"quartiles from {items_csv}")
        except Exception as e:
            print(f"[metrics] WARNING: could not read {items_csv}: {e}")

    # Build a global item set and the top-quartile (popular) subset
    # over *all* items in the graph (not just the sampled ones).  These
    # power the popularity-bias metrics `pop_frac` (fraction of summary
    # items that are popular) and `pop_frac_topk` (the same fraction
    # on the recommender's top-K item set, used as the upstream baseline).
    # Item nodes in ML1M are typed `type=item`; we identify them that way
    # and rank them by user-degree (the number of user nodes that link to
    # the item, which is the standard popularity proxy).
    all_items: set[str] = set()
    item_user_degree: dict[str, int] = {}
    for n, data in G.nodes(data=True):
        if data.get("type") == "item":
            sn = str(n)
            all_items.add(sn)
            # Count user neighbours (in either direction).
            users = 0
            for nb in G.predecessors(n):
                if str(nb).startswith("u"):
                    users += 1
            for nb in G.successors(n):
                if str(nb).startswith("u"):
                    users += 1
            item_user_degree[sn] = users
    popular_items: set[str] = set()
    if item_user_degree:
        sorted_items = sorted(item_user_degree.items(),
                              key=lambda kv: kv[1], reverse=True)
        cutoff = max(1, len(sorted_items) // 4)
        popular_items = {iid for iid, _ in sorted_items[:cutoff]}
        print(f"[metrics] {args.dataset}: {len(all_items)} items, "
              f"{len(popular_items)} popular (top-quartile by user degree)")

    in_root = RESULTS_ROOT / args.dataset
    per_anchor_dir = RESULTS_ROOT / "per_anchor"
    per_anchor_dir.mkdir(parents=True, exist_ok=True)

    all_per_anchor = []
    for jp in sorted(in_root.glob("*/*.jsonl")):
        info = parse_filename(jp)
        records = load_records(jp)
        if not records:
            continue
        df = compute_per_anchor(
            records, G,
            dataset=info["dataset"], scenario=info["scenario"],
            baseline=info["baseline"], algorithm=info["algorithm"],
            centrality=info["centrality"] or None,
            all_items=all_items,
            popular_items=popular_items,
        )
        if df.empty:
            continue
        out = per_anchor_dir / f"{jp.stem}.parquet"
        df.to_parquet(out, index=False)
        all_per_anchor.append(df)
        print(f"[metrics] wrote per-anchor: {out} ({len(df)} rows)")

    if not all_per_anchor:
        print("[metrics] no result files found; did you run the experiments?")
        return

    big = pd.concat(all_per_anchor, ignore_index=True)
    summary = aggregate(big, user_genders=user_genders,
                        item_quartiles=item_quartiles)
    out = Path(args.out_summary) if args.out_summary else (RESULTS_ROOT / args.dataset / "summary.csv")
    summary.to_csv(out, index=False)
    print(f"[metrics] wrote aggregated summary: {out}")

    # Also persist a human-readable text snapshot next to the CSV.  This is
    # the same content as the stdout "phase 6 printout" but on disk so it
    # survives across runs and is greppable.  One section per scenario,
    # showing the headline metrics that drive the paper's tables.
    txt_out = out.with_suffix(".txt")
    headline_cols = [
        "dataset", "scenario", "baseline", "algorithm", "centrality",
        "comprehensibility_mean", "relevance_mean",
        "actionability_mean", "diversity_mean",
        "coverage_mean", "faithfulness_mean",
        "cpu_time_sec_mean",
    ]
    avail_cols = [c for c in headline_cols if c in summary.columns]
    with open(txt_out, "w") as fh:
        fh.write(f"# Aggregated summary: {out}\n")
        fh.write(f"# Total cells: {len(summary)}\n\n")
        for sc in sorted(summary["scenario"].dropna().unique()):
            sub = summary[summary["scenario"] == sc][avail_cols]
            if sub.empty:
                continue
            fh.write(f"## scenario = {sc}  (n_cells = {len(sub)})\n")
            fh.write(sub.sort_values(["baseline", "algorithm", "centrality"]).to_string(index=False))
            fh.write("\n\n")
    print(f"[metrics] wrote summary text: {txt_out}")

    # Echo a short head() to stdout so the phase 6 log still shows progress.
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
