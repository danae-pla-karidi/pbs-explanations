from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from config import DATASETS, TOP_K, RESULTS_ROOT, SAMPLES_DIR

SCENARIO_ORDER = ["user_centric", "item_centric", "user_group", "item_group"]


# ---------------------------------------------------------------------------
# Loaders (same files and keys as runners/_core.py)
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path, key: str) -> dict[str, dict]:
    out = {}
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            r = json.loads(line)
            out[str(r[key])] = r
    return out


def _rec_to_graph_uid(dataset: str) -> dict[str, str]:
    """rec-space user id -> graph user id (LFM1M). Empty for ML1M."""
    cfg = DATASETS[dataset]
    path = cfg.get("uid_mapping_file")
    if path is None or not Path(str(path)).exists():
        return {}
    offset = cfg.get("uid_mapping_offset", 1)
    inv = {}
    with open(path, "r", encoding="utf-8") as fin:
        header = fin.readline().strip().split("\t")
        a, b = header.index("new_id"), header.index("uid")
        for line in fin:
            p = line.rstrip("\n").split("\t")
            if len(p) > max(a, b):
                inv[f"u{int(p[a]) + offset}"] = f"u{p[b]}"
    return inv


# ---------------------------------------------------------------------------
# Terminal sets, mirroring the four runners
# ---------------------------------------------------------------------------

def terminals_user_centric(users, recs, inv, K):
    out = {}
    for uid in users["user_id"].astype(str):
        r = recs.get(uid)
        if r is None:
            continue
        out[inv.get(uid, uid)] = [str(t) for t in r.get("recommended_items_ids", [])[:K]]
    return out


def terminals_user_group(users, recs, K):
    out = {}
    for gender, name in (("M", "group_male"), ("F", "group_female")):
        counts = Counter()
        for uid in users[users.gender == gender]["user_id"].astype(str):
            r = recs.get(uid)
            if r is None:
                continue
            for it in r.get("recommended_items_ids", [])[:K]:
                counts[str(it)] += 1
        t = [it for it, _ in counts.most_common(K)]
        if t:
            out[name] = t
    return out


def terminals_item_centric(items, paths, K):
    out = {}
    for iid in items["item_id"].astype(str):
        r = paths.get(iid)
        if r is None:
            continue
        out[iid] = [str(t) for t in r.get("recommended_users_ids", [])[:K]]
    return out


def terminals_item_group(items, paths, K):
    out = {}
    for label, q in (("popular", "q4_popular"), ("unpopular", "q1_unpopular")):
        counts = Counter()
        for iid in items[items.quartile == q]["item_id"].astype(str):
            r = paths.get(iid)
            if r is None:
                continue
            for u in r.get("recommended_users_ids", [])[:K]:
                counts[str(u)] += 1
        t = [u for u, _ in counts.most_common(K)]
        if t:
            out[f"group_{label}"] = t
    return out


def build_terminals(dataset: str, scenario: str, baseline: str, K: int) -> dict[str, list[str]]:
    cfg = DATASETS[dataset]
    if scenario in ("user_centric", "user_group"):
        users = pd.read_csv(SAMPLES_DIR / f"{dataset}_users.csv")
        recs = _read_jsonl(Path(str(cfg["user_recs_template"]).format(baseline=baseline)), "user_id")
        if scenario == "user_centric":
            return terminals_user_centric(users, recs, _rec_to_graph_uid(dataset), K)
        return terminals_user_group(users, recs, K)
    items = pd.read_csv(SAMPLES_DIR / f"{dataset}_items.csv")
    items["item_id"] = items["item_id"].astype(str)
    paths = _read_jsonl(Path(str(cfg["item_paths_template"]).format(baseline=baseline)), "item_id")
    if scenario == "item_centric":
        return terminals_item_centric(items, paths, K)
    return terminals_item_group(items, paths, K)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rows, warnings = [], []
    for dataset, cfg in DATASETS.items():
        for scenario in cfg["scenarios"]:
            sdir = RESULTS_ROOT / dataset / scenario
            if not sdir.exists():
                continue
            for baseline in cfg["baselines"]:
                prefix = f"{dataset}_{scenario}_{baseline}_"
                files = sorted(sdir.glob(prefix + "*.jsonl"))
                if not files:
                    continue
                terms = build_terminals(dataset, scenario, baseline, TOP_K)
                for f in files:
                    algo = f.stem[len(prefix):]
                    n_unmatched = 0
                    with open(f, "r", encoding="utf-8") as fin:
                        for line in fin:
                            r = json.loads(line)
                            aid = str(r["anchor_id"])
                            T = terms.get(aid)
                            if not T:
                                n_unmatched += 1
                                continue
                            V = set(map(str, r["solution_nodes"]))
                            Tset = set(T)
                            cov = len(Tset & V)
                            rows.append({
                                "dataset": dataset, "scenario": scenario,
                                "recommender": baseline, "algorithm": algo,
                                "anchor_id": aid,
                                "n_terminals": len(Tset), "n_covered": cov,
                                "tc": cov / len(Tset),
                                "full": int(cov == len(Tset)),
                                "anchor_in_summary": int(aid in V),
                                "num_edges": r.get("num_edges"),
                            })
                    if n_unmatched:
                        warnings.append(f"{f.name}: {n_unmatched} records without a terminal set")

    per_anchor = pd.DataFrame(rows)
    per_anchor.to_csv(RESULTS_ROOT / "terminal_coverage_per_anchor.csv", index=False)

    cell = (per_anchor
            .groupby(["dataset", "scenario", "recommender", "algorithm"])
            .agg(n=("tc", "size"), tc=("tc", "mean"), full_rate=("full", "mean"),
                 terminals=("n_terminals", "mean"))
            .reset_index())
    cell.to_csv(RESULTS_ROOT / "terminal_coverage.csv", index=False)

    # Mean over recommenders of the per-recommender means (as for C and R).
    table = (cell.groupby(["dataset", "algorithm", "scenario"])
             .agg(tc=("tc", "mean"), full_rate=("full_rate", "mean"),
                  recommenders=("recommender", "nunique"))
             .reset_index())
    table.to_csv(RESULTS_ROOT / "terminal_coverage_table.csv", index=False)

    pd.set_option("display.width", 200)
    r3 = lambda df: (df + 1e-9).round(3)   # round half up on exact ties
    for dataset in table.dataset.unique():
        sub = table[table.dataset == dataset]
        cols = [s for s in SCENARIO_ORDER if s in set(sub.scenario)]
        print(f"\n=== {dataset}: terminal coverage TC (mean over anchors, then over recommenders)")
        print(r3(sub.pivot(index="algorithm", columns="scenario", values="tc")[cols]).to_string())
        print(f"\n=== {dataset}: fraction of summaries that contain every terminal")
        print(r3(sub.pivot(index="algorithm", columns="scenario", values="full_rate")[cols]).to_string())
        print(f"\n=== {dataset}: number of recommenders per cell")
        print(sub.pivot(index="algorithm", columns="scenario", values="recommenders")[cols].to_string())

    print("\n=== per recommender, ST / PCST / WPCST(degree)")
    core = cell[cell.algorithm.isin(["st", "pcst", "wpcst_cent_degree"])]
    print(r3(core.pivot_table(index=["dataset", "scenario", "recommender"],
                              columns="algorithm", values="tc")).to_string())

    if warnings:
        print("\nWARNINGS")
        for w in warnings:
            print("  " + w)


if __name__ == "__main__":
    main()
