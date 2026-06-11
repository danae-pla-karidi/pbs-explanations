"""
Run the four external graph-summarization baselines across all
(scenario, recommender) cells, using the median WPCST size for that cell
as the budget.

Baselines:
  - pappas2017  (Pappas et al. ESWC 2017): importance-rank + Steiner tree
  - mst         (Troullinou et al. ESWC 2015): MST + paths through MST
  - faces       (Gunaratna et al. AAAI 2015): diversity-aware clustering + Steiner
  - supernode   (SWeG-style Shin et al. WWW 2019): type-based node aggregation

Importance measure used by pappas/mst/faces is approximate betweenness by
default, matching Pappas et al.'s recommendation.  Override via --importance.
The supernode baseline does not use importance.

Read by `scripts/run_all_ml1m.sh` and `scripts/run_all_lfm1m.sh` (phase 4).
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

from config import DATASETS, RESULTS_ROOT
from runners.run_user_centric import run as run_uc
from runners.run_user_group import run as run_ug

RUNNERS = {"user_centric": run_uc, "user_group": run_ug}

EXTERNAL_BASELINES = ["pappas2017", "mst", "faces", "supernode"]


def _maybe_import_item_runners():
    try:
        from runners.run_item_centric import run as run_ic
        from runners.run_item_group import run as run_ig
        RUNNERS["item_centric"] = run_ic
        RUNNERS["item_group"] = run_ig
    except ImportError:
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    p.add_argument("--budgets-json", default=None,
                   help="Path to subgraph_budgets.json (default = "
                        "results/<dataset>/subgraph_budgets.json)")
    p.add_argument("--importance", default="betweenness_approx",
                   choices=["degree", "pagerank", "betweenness_approx"],
                   help="Importance measure for ranking nodes; the "
                        "Pappas paper recommends Betweenness.")
    p.add_argument("--algorithms", nargs="+", default=EXTERNAL_BASELINES,
                   choices=EXTERNAL_BASELINES,
                   help="Subset of baselines to run.")
    p.add_argument("--scenarios", nargs="+", default=None,
                   choices=["user_centric", "user_group",
                            "item_centric", "item_group"],
                   help="Subset of scenarios to run; default = all "
                        "scenarios configured for the dataset.")
    args = p.parse_args()

    _maybe_import_item_runners()

    cfg = DATASETS[args.dataset]
    budgets_path = Path(args.budgets_json) if args.budgets_json else (
        RESULTS_ROOT / args.dataset / "subgraph_budgets.json"
    )
    if not budgets_path.exists():
        raise FileNotFoundError(
            f"{budgets_path} not found. Run "
            "`python -m scripts.compute_subgraph_budgets --dataset "
            f"{args.dataset}` first."
        )
    budgets = json.loads(budgets_path.read_text())

    for algorithm in args.algorithms:
        scenario_list = args.scenarios if args.scenarios else cfg["scenarios"]
        for scenario in scenario_list:
            if scenario not in RUNNERS:
                print(f"[{algorithm}] no runner for {scenario}; skipping")
                continue
            run_fn = RUNNERS[scenario]
            for baseline in cfg["baselines"]:
                budget = budgets.get(scenario, {}).get(baseline, 0)
                if budget <= 0 and algorithm != "supernode":
                    print(f"[{algorithm}] skipping {scenario}/{baseline}: budget={budget}")
                    continue
                if algorithm == "supernode":
                    centrality_name = "degree"  # threaded through, ignored
                    print(f"--- {args.dataset}/{scenario}/{baseline}/supernode ---")
                else:
                    centrality_name = args.importance
                    print(f"--- {args.dataset}/{scenario}/{baseline}/{algorithm} "
                          f"(importance={centrality_name}, budget={budget}) ---")
                run_fn(
                    args.dataset, baseline, algorithm,
                    K=10,
                    centrality_name=centrality_name,
                    budget=budget if budget > 0 else 0,
                )


if __name__ == "__main__":
    main()
