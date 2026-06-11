"""
Compute the per-cell budget for the Pappas et al. 2017 baseline.

We want pappas2017 to produce subgraphs of comparable size to WPCST,
so the comparison is on quality at equal budget. This script reads the
WPCST result files for a given (dataset, scenario, baseline) and outputs
the median number of nodes, which the runner then passes to pappas2017
as `budget`.

Usage:
    python -m scripts.compute_subgraph_budgets --dataset ml1m
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import defaultdict
import statistics

from config import DATASETS, RESULTS_ROOT


def median_wpcst_size(jsonl_path: Path) -> int:
    sizes = []
    if not jsonl_path.exists():
        return 0
    with open(jsonl_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                sizes.append(r.get("num_nodes", 0))
            except json.JSONDecodeError:
                continue
    if not sizes:
        return 0
    # Subtract the 1+|terminals| we already mandate (anchor + terminals get
    # kept regardless of budget), so `budget` is the *extra* nodes to keep.
    # Approximation: floor(median - 11) to account for anchor + 10 terminals.
    return max(0, int(statistics.median(sizes)) - 11)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    cfg = DATASETS[args.dataset]
    budgets = defaultdict(dict)
    for scenario in cfg["scenarios"]:
        for baseline in cfg["baselines"]:
            wpcst_file = (
                RESULTS_ROOT / args.dataset / scenario /
                f"{args.dataset}_{scenario}_{baseline}_wpcst_cent_degree.jsonl"
            )
            b = median_wpcst_size(wpcst_file)
            budgets[scenario][baseline] = b
            print(f"[budgets] {args.dataset}/{scenario}/{baseline}: budget={b} (from {wpcst_file.name})")

    out = Path(args.out_json) if args.out_json else (RESULTS_ROOT / args.dataset / "subgraph_budgets.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(budgets, indent=2))
    print(f"[budgets] wrote {out}")


if __name__ == "__main__":
    main()
