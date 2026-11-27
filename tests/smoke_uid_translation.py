"""
Regression test for the LFM-1M UID translation fix in runners/_core.py.

Builds a toy dataset whose KG uses *real* user IDs (u1003134-style) while
the recommendation files use the recommender's sequential IDs (u1..u6),
bridged by a `user_uid_map.tsv` exactly like the LFM-1M artefacts.

Asserts the two failure modes observed on the real LFM-1M ablation outputs
are gone after translation:
  (1) the anchor user appears in its own summary (was 0/200), and
  (2) the path-aware reweighting fires, i.e. lambda changes the adjusted
      solution weight (was bit-identical across lambda in {0.01, 1, 100}).

    python -m tests.smoke_uid_translation
"""

from __future__ import annotations

import json

import networkx as nx
import pandas as pd

import config
from config import DATA_ROOT, SAMPLES_DIR


REAL_UIDS = ["1003134", "2103", "555001", "660042", "70071", "80088"]
GENDERS = ["M", "M", "M", "F", "F", "F"]


def _make_dataset() -> str:
    ds = "toylfm"
    ddir = DATA_ROOT / ds
    ddir.mkdir(parents=True, exist_ok=True)

    # KG in *graph* (real-ID) space.
    G = nx.DiGraph()
    for ruid, g in zip(REAL_UIDS, GENDERS):
        G.add_node(f"u{ruid}", type="user", gender=g)
    for i in range(1, 9):
        G.add_node(f"i{i}", type="item")
    for h in ("g1", "g2", "d1"):
        G.add_node(h, type="attr")
    ui = [(0, "i1", .9), (0, "i2", .6), (0, "i3", .5),
          (1, "i2", .8), (1, "i4", .4), (1, "i5", .7),
          (2, "i1", .7), (2, "i5", .6), (2, "i6", .3),
          (3, "i3", .9), (3, "i6", .5), (3, "i7", .8),
          (4, "i4", .6), (4, "i7", .7), (4, "i8", .4),
          (5, "i2", .5), (5, "i6", .8), (5, "i8", .6)]
    for k, it, w in ui:
        G.add_edge(f"u{REAL_UIDS[k]}", it, weight=float(w), w_base=float(w))
    ia = [("i1", "g1", .5), ("i2", "g1", .55), ("i3", "g1", .45),
          ("i4", "g2", .5), ("i5", "g2", .6), ("i6", "g2", .5),
          ("i7", "d1", .7), ("i8", "d1", .65), ("i1", "d1", .4),
          ("i2", "d1", .35), ("i5", "g1", .3), ("i6", "g1", .25)]
    for u, v, w in ia:
        G.add_edge(u, v, weight=float(w), w_base=float(w))
    nx.write_graphml(G, ddir / "kg.graphml")

    # Mapping TSV: header new_id<TAB>uid, 0-indexed; rec id = u<new_id+1>.
    with open(ddir / "user_uid_map.tsv", "w") as f:
        f.write("new_id\tuid\n")
        for i, ruid in enumerate(REAL_UIDS):
            f.write(f"{i}\t{ruid}\n")

    # Recommendations in *rec* (sequential) space, paths included.
    def path(useq, hub, item):
        return [["self", "user", f"u{useq}"], ["listened", "item", item],
                ["has", "attr", hub]]

    recs = {
        "u1": (["i1", "i2", "i3"], [path(1, "g1", "i1"), path(1, "g1", "i2"), path(1, "d1", "i1")]),
        "u2": (["i2", "i5", "i4"], [path(2, "g1", "i2"), path(2, "g2", "i5"), path(2, "g2", "i4")]),
        "u3": (["i1", "i5", "i6"], [path(3, "d1", "i1"), path(3, "g2", "i5"), path(3, "g2", "i6")]),
        "u4": (["i3", "i7", "i6"], [path(4, "g1", "i3"), path(4, "d1", "i7"), path(4, "g2", "i6")]),
        "u5": (["i7", "i4", "i8"], [path(5, "d1", "i7"), path(5, "g2", "i4"), path(5, "d1", "i8")]),
        "u6": (["i6", "i2", "i8"], [path(6, "g2", "i6"), path(6, "g1", "i2"), path(6, "d1", "i8")]),
    }
    with open(ddir / "pgpr_recommendations.jsonl", "w") as f:
        for uid, (items, paths) in recs.items():
            f.write(json.dumps({"user_id": uid, "recommended_items_ids": items,
                                "top_k_paths": paths}) + "\n")

    # Sample CSV in rec space (what the sampler would have written).
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"user_id": [f"u{i}" for i in range(1, 7)],
                  "gender": GENDERS}).to_csv(
        SAMPLES_DIR / f"{ds}_users.csv", index=False)

    config.DATASETS[ds] = {
        "graphml": ddir / "kg.graphml",
        "user_recs_template": str(ddir / "{baseline}_recommendations.jsonl"),
        "item_paths_template": None,
        "uid_mapping_file": ddir / "user_uid_map.tsv",
        "uid_mapping_offset": 1,
        "baselines": ["pgpr"],
        "scenarios": ["user_centric"],
    }
    return ds


def main():
    ds = _make_dataset()
    from runners.run_lambda_ablation import run_ablation
    from config import RESULTS_ROOT

    run_ablation(ds, lambdas=[0.01, 100.0], algorithms=["st", "wpcst"],
                 baselines=["pgpr"], scenarios=["user_centric"],
                 K=3, centrality_name="degree", n_workers=1)

    out_dir = RESULTS_ROOT / "_lambda_ablation" / ds

    def load(alg, tag):
        return [json.loads(l) for l in
                open(out_dir / f"{ds}_user_centric_pgpr_{alg}_{tag}.jsonl")]

    real_anchor_ids = {f"u{r}" for r in REAL_UIDS}
    for alg in ("st", "wpcst"):
        lo, hi = load(alg, "lam0p01"), load(alg, "lam100")

        # (0) anchors are graph-space IDs now
        assert all(r["anchor_id"] in real_anchor_ids for r in lo), \
            f"{alg}: anchor_id not translated to graph space"

        # (1) anchor present in its own summary (lambda=0.01 run)
        n_in = sum(r["anchor_id"] in r["solution_nodes"] for r in lo)
        assert n_in == len(lo), f"{alg}: anchor missing from summary ({n_in}/{len(lo)})"

        # (2) reweighting fires: adjusted weights differ across lambda
        sw_lo = {r["anchor_id"]: r["sum_weight"] for r in lo}
        sw_hi = {r["anchor_id"]: r["sum_weight"] for r in hi}
        n_diff = sum(sw_lo[a] != sw_hi[a] for a in sw_lo)
        assert n_diff > 0, f"{alg}: sum_weight identical across lambda; reweighting still inert"
        print(f"[{alg}] anchors in summary: {n_in}/{len(lo)}; "
              f"sum_weight differs across lambda for {n_diff}/{len(sw_lo)} anchors")

    print("UID-translation regression test passed.")


if __name__ == "__main__":
    main()
