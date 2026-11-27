"""
End-to-end smoke test: build a tiny synthetic KG, run all five algorithms
on a single anchor, sanity-check the outputs.

This does NOT validate correctness against the real datasets; it only verifies
the pipeline wiring (imports, types, return-value schema). Run it once after
cloning the repo to confirm everything is plugged in:

    python -m tests.smoke
"""

from __future__ import annotations

import networkx as nx

from algorithms import (
    AnchorRequest, PcstIndex,
    run_steiner, run_pcst, run_wpcst, run_naive_union, run_pappas2017,
    run_mst_summary, run_faces_summary, run_supernode_summary,
)
from centralities import degree_centrality
from metrics.compute_metrics import m_coverage, m_faithfulness


def build_toy_graph() -> nx.DiGraph:
    """A 7-node graph with a clear central hub."""
    G = nx.DiGraph()
    edges = [
        ("u1", "i1", 0.8), ("u1", "i2", 0.6),
        ("i1", "h",  0.9), ("i2", "h",  0.7),
        ("h",  "i3", 0.85), ("h", "i4", 0.5),
        ("u2", "i1", 0.4), ("u2", "i3", 0.3),
    ]
    for u, v, w in edges:
        G.add_edge(u, v, weight=w, w_base=w)
    return G


def _assert_no_duplicates(label: str, r: dict) -> None:
    """Catch the pcst_fast 1.0.10 duplicate-output bug if the workaround
    in pcst.py / wpcst.py ever regresses."""
    nodes = r.get("solution_nodes", [])
    if len(nodes) != len(set(nodes)):
        raise AssertionError(
            f"[{label}] solution_nodes contains duplicates "
            f"({len(nodes)} entries, {len(set(nodes))} unique). "
            f"First 5: {nodes[:5]}"
        )


def main():
    G = build_toy_graph()
    G_und = G.to_undirected(as_view=True)
    cent = degree_centrality(G)
    index = PcstIndex(G)

    req = AnchorRequest(
        anchor_id="u1",
        terminals=["i1", "i2", "i3"],
        top_k_paths=[
            [["self", "user", "u1"], ["watched", "item", "i1"], ["has", "ext", "h"]],
            [["self", "user", "u1"], ["watched", "item", "i2"], ["has", "ext", "h"]],
            [["self", "user", "u1"], ["watched", "item", "i1"], ["has", "ext", "h"], ["leads", "item", "i3"]],
        ],
        metadata={"smoke": True},
    )

    print("=== Steiner ===")
    r = run_steiner(G, req, lam=1.0, K=3, anchor_kind="user")
    print(f"  |V|={r['num_nodes']} |E|={r['num_edges']} sum_w={r['sum_weight']:.3f} "
          f"wall={r['performance']['execution_time']:.4f}s "
          f"cpu={r['performance']['cpu_time']:.4f}s")
    print(f"  coverage={m_coverage(r):.3f} faithfulness={m_faithfulness(r):.3f}")
    _assert_no_duplicates("Steiner", r)

    print("=== PCST ===")
    r = run_pcst(G, req, lam=1.0, K=3, anchor_kind="user", index=index, root_node="u1")
    print(f"  |V|={r['num_nodes']} |E|={r['num_edges']} sum_w={r['sum_weight']:.3f}")
    _assert_no_duplicates("PCST", r)

    print("=== WPCST (degree) ===")
    r = run_wpcst(G, req, lam=1.0, K=3, anchor_kind="user", index=index,
                  centrality=cent, gamma=0.1, G_und=G_und, root_node="u1")
    print(f"  |V|={r['num_nodes']} |E|={r['num_edges']} sum_w={r['sum_weight']:.3f} "
          f"alpha={r['metadata'].get('alpha')} beta={r['metadata'].get('beta')}")
    _assert_no_duplicates("WPCST", r)

    print("=== NaiveUnion ===")
    r = run_naive_union(G, req, lam=1.0, K=3, anchor_kind="user")
    print(f"  |V|={r['num_nodes']} |E|={r['num_edges']} sum_w={r['sum_weight']:.3f}")
    _assert_no_duplicates("NaiveUnion", r)

    print("=== Pappas2017 (degree, budget=2) ===")
    r = run_pappas2017(G, req, lam=1.0, K=3, anchor_kind="user",
                       importance=cent, budget=2)
    print(f"  |V|={r['num_nodes']} |E|={r['num_edges']} sum_w={r['sum_weight']:.3f}")
    _assert_no_duplicates("Pappas2017", r)

    print("=== MST (degree, budget=2) ===")
    r = run_mst_summary(G, req, lam=1.0, K=3, anchor_kind="user",
                        importance=cent, budget=2)
    print(f"  |V|={r['num_nodes']} |E|={r['num_edges']} sum_w={r['sum_weight']:.3f}")
    _assert_no_duplicates("MST", r)

    print("=== FACES (degree, budget=2) ===")
    r = run_faces_summary(G, req, lam=1.0, K=3, anchor_kind="user",
                          importance=cent, budget=2)
    print(f"  |V|={r['num_nodes']} |E|={r['num_edges']} sum_w={r['sum_weight']:.3f}")
    _assert_no_duplicates("FACES", r)

    print("=== SuperNode ===")
    r = run_supernode_summary(G, req, lam=1.0, K=3, anchor_kind="user")
    print(f"  |V|={r['num_nodes']} |E|={r['num_edges']} "
          f"compression_ratio={r['metadata'].get('compression_ratio'):.3f}")
    _assert_no_duplicates("SuperNode", r)

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
