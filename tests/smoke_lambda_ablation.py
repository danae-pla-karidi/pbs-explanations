"""
Wiring test for runners.run_lambda_ablation on a tiny synthetic dataset.

Builds a toy KG + recommendation files + user sample, registers a 'toy'
dataset in config.DATASETS at runtime, and runs the lambda ablation for
ST and WPCST on the user-centric and user-group scenarios.  Verifies that
the sweep produces finite C/R for every (scenario, baseline, algorithm,
lambda) cell and that the table / tex artifacts are written.

    python -m tests.smoke_lambda_ablation
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

import config
from config import DATA_ROOT, SAMPLES_DIR, RESULTS_ROOT


def _build_toy_graph() -> nx.DiGraph:
    """6 users (3M/3F), 8 items, 3 attribute hubs.  Item nodes carry
    type='item'; users carry gender; every edge carries a base 'weight'."""
    G = nx.DiGraph()
    users = {"u1": "M", "u2": "M", "u3": "M", "u4": "F", "u5": "F", "u6": "F"}
    for u, g in users.items():
        G.add_node(u, type="user", gender=g)
    for i in range(1, 9):
        G.add_node(f"i{i}", type="item")
    for h in ("g1", "g2", "d1"):
        G.add_node(h, type="attr")

    # user--item interactions (base weights vary)
    ui = [
        ("u1", "i1", 0.9), ("u1", "i2", 0.6), ("u1", "i3", 0.5),
        ("u2", "i2", 0.8), ("u2", "i4", 0.4), ("u2", "i5", 0.7),
        ("u3", "i1", 0.7), ("u3", "i5", 0.6), ("u3", "i6", 0.3),
        ("u4", "i3", 0.9), ("u4", "i6", 0.5), ("u4", "i7", 0.8),
        ("u5", "i4", 0.6), ("u5", "i7", 0.7), ("u5", "i8", 0.4),
        ("u6", "i2", 0.5), ("u6", "i6", 0.8), ("u6", "i8", 0.6),
    ]
    # item--attribute edges (the hubs that summaries route through)
    ia = [
        ("i1", "g1", 0.5), ("i2", "g1", 0.55), ("i3", "g1", 0.45),
        ("i4", "g2", 0.5), ("i5", "g2", 0.6), ("i6", "g2", 0.5),
        ("i7", "d1", 0.7), ("i8", "d1", 0.65), ("i1", "d1", 0.4),
        ("i2", "d1", 0.35), ("i5", "g1", 0.3), ("i6", "g1", 0.25),
    ]
    for u, v, w in ui + ia:
        G.add_edge(u, v, weight=float(w), w_base=float(w))
    return G


def _toy_path(u: str, mid: str, item: str):
    """A 3-step explanation path [user]-[item]-[hub], schema [[rel,type,node],...]."""
    return [["self", "user", u], ["watched", "item", item], ["has", "attr", mid]]


def _write_recs(path: Path, recs: dict[str, dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for uid, rec in recs.items():
            f.write(json.dumps(rec) + "\n")


def _make_dataset() -> str:
    ds = "toy"
    ddir = DATA_ROOT / ds
    ddir.mkdir(parents=True, exist_ok=True)

    G = _build_toy_graph()
    graphml = ddir / "kg_static.graphml"
    nx.write_graphml(G, graphml)

    # Two recommenders with slightly different top-K + paths so the
    # mean-over-baselines step has something to average.
    recs_pgpr = {
        "u1": {"user_id": "u1", "recommended_items_ids": ["i1", "i2", "i3"],
               "top_k_paths": [_toy_path("u1", "g1", "i1"),
                               _toy_path("u1", "g1", "i2"),
                               _toy_path("u1", "d1", "i1")]},
        "u2": {"user_id": "u2", "recommended_items_ids": ["i2", "i5", "i4"],
               "top_k_paths": [_toy_path("u2", "g1", "i2"),
                               _toy_path("u2", "g2", "i5"),
                               _toy_path("u2", "g2", "i4")]},
        "u3": {"user_id": "u3", "recommended_items_ids": ["i1", "i5", "i6"],
               "top_k_paths": [_toy_path("u3", "d1", "i1"),
                               _toy_path("u3", "g2", "i5"),
                               _toy_path("u3", "g2", "i6")]},
        "u4": {"user_id": "u4", "recommended_items_ids": ["i3", "i7", "i6"],
               "top_k_paths": [_toy_path("u4", "g1", "i3"),
                               _toy_path("u4", "d1", "i7"),
                               _toy_path("u4", "g2", "i6")]},
        "u5": {"user_id": "u5", "recommended_items_ids": ["i7", "i4", "i8"],
               "top_k_paths": [_toy_path("u5", "d1", "i7"),
                               _toy_path("u5", "g2", "i4"),
                               _toy_path("u5", "d1", "i8")]},
        "u6": {"user_id": "u6", "recommended_items_ids": ["i6", "i2", "i8"],
               "top_k_paths": [_toy_path("u6", "g2", "i6"),
                               _toy_path("u6", "g1", "i2"),
                               _toy_path("u6", "d1", "i8")]},
    }
    # CAFE: shuffle a couple of hub choices to differ from PGPR.
    recs_cafe = {u: {**r, "top_k_paths": list(reversed(r["top_k_paths"]))}
                 for u, r in recs_pgpr.items()}

    _write_recs(ddir / "pgpr_recommendations.jsonl", recs_pgpr)
    _write_recs(ddir / "cafe_recommendations.jsonl", recs_cafe)

    # User sample CSV (the seeded sampler is bypassed for the toy set).
    users_df = pd.DataFrame(
        [("u1", "M"), ("u2", "M"), ("u3", "M"),
         ("u4", "F"), ("u5", "F"), ("u6", "F")],
        columns=["user_id", "gender"])
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    users_df.to_csv(SAMPLES_DIR / f"{ds}_users.csv", index=False)

    # Register the dataset (same dict object the runner imported).
    config.DATASETS[ds] = {
        "graphml": graphml,
        "user_recs_template": str(ddir / "{baseline}_recommendations.jsonl"),
        "item_paths_template": None,
        "baselines": ["pgpr", "cafe"],
        "scenarios": ["user_centric", "user_group"],
    }
    return ds


def main():
    ds = _make_dataset()

    # Import after the dataset is registered.
    from runners.run_lambda_ablation import run_ablation

    long_df = run_ablation(
        ds,
        lambdas=[0.01, 1.0, 100.0],
        algorithms=["st", "wpcst"],
        baselines=["pgpr", "cafe"],
        scenarios=["user_centric", "user_group"],
        K=3,
        centrality_name="degree",
        n_workers=1,
        limit=None,
    )

    # --- assertions -------------------------------------------------------
    expected_cells = 2 * 2 * 2 * 3  # scenarios x baselines x algs x lambdas
    assert len(long_df) == expected_cells, \
        f"expected {expected_cells} cells, got {len(long_df)}"
    assert long_df["comprehensibility"].notna().all(), "NaN comprehensibility"
    assert long_df["relevance"].notna().all(), "NaN relevance"
    assert (long_df["relevance"] >= 0).all(), "negative relevance"
    # ST on user-centric should connect all 3 terminals + anchor -> >=3 edges.
    uc_st = long_df[(long_df.scenario == "user_centric") & (long_df.algorithm == "st")]
    assert (uc_st["num_edges"] >= 3).all(), "ST user-centric tree too small"

    out_dir = RESULTS_ROOT / "_lambda_ablation" / ds
    for fname in ("lambda_ablation_long.csv", "lambda_ablation_table.csv",
                  "lambda_ablation_table.tex"):
        assert (out_dir / fname).exists(), f"missing artifact {fname}"

    print("\n=== mean-over-baselines table (toy) ===")
    from runners.run_lambda_ablation import _mean_over_baselines
    print(_mean_over_baselines(long_df).to_string(index=False))

    print("\n=== generated LaTeX ===")
    print((out_dir / "lambda_ablation_table.tex").read_text())

    print("Lambda-ablation smoke test passed.")


if __name__ == "__main__":
    main()
