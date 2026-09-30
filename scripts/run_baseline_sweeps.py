from __future__ import annotations
import argparse
import json
import os
import shutil
import statistics
from pathlib import Path

from config import DATASETS, RESULTS_ROOT, TOP_K
from runners.run_user_centric import run as run_uc
from runners.run_user_group import run as run_ug

RUNNERS = {"user_centric": run_uc, "user_group": run_ug}
try:
    from runners.run_item_centric import run as run_ic
    from runners.run_item_group import run as run_ig
    RUNNERS["item_centric"] = run_ic
    RUNNERS["item_group"] = run_ig
except ImportError:
    pass

SWEEP_ROOT = RESULTS_ROOT / "sweeps"
BASELINES = ["pappas2017", "mst", "faces"]
MAIN_IMPORTANCE = "betweenness_approx"
MAIN_FACES_KEY = "kg_type"


def wpcst_median_size(dataset: str, scenario: str, baseline: str) -> float:
    path = (RESULTS_ROOT / dataset / scenario /
            f"{dataset}_{scenario}_{baseline}_wpcst_cent_degree.jsonl")
    sizes = []
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    sizes.append(json.loads(line).get("num_nodes", 0))
                except json.JSONDecodeError:
                    continue
    return statistics.median(sizes) if sizes else 0.0


def budget_for(median_size: float, rho: float, K: int = TOP_K) -> int:
    return max(0, int(rho * median_size) - (1 + K))


def sweep_target(dataset: str, scenario: str, baseline: str, algorithm: str,
                 importance: str, tag: str) -> Path:
    fname = f"{dataset}_{scenario}_{baseline}_{algorithm}_cent_{importance}_{tag}.jsonl"
    return SWEEP_ROOT / dataset / scenario / fname


def run_one(dataset: str, scenario: str, baseline: str, algorithm: str, *,
            importance: str, budget: int, tag: str, faces_key: str = MAIN_FACES_KEY,
            workers: int | None) -> None:
    target = sweep_target(dataset, scenario, baseline, algorithm, importance, tag)
    if target.exists():
        print(f"[sweep] exists, skipping: {target.name}")
        return
    if budget <= 0:
        print(f"[sweep] budget=0, skipping: {target.name}")
        return
    os.environ["FACES_CLUSTER_KEY"] = faces_key if algorithm == "faces" else MAIN_FACES_KEY
    print(f"--- {dataset}/{scenario}/{baseline}/{algorithm} "
          f"importance={importance} budget={budget} tag={tag}"
          f"{' faces_key=' + faces_key if algorithm == 'faces' else ''} ---")
    out = RUNNERS[scenario](dataset, baseline, algorithm, K=TOP_K,
                            centrality_name=importance, budget=budget,
                            tag=tag, n_workers=workers)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(out), str(target))
    print(f"[sweep] moved -> {target}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    p.add_argument("--sweep", required=True, choices=["budget", "tuning"])
    p.add_argument("--rhos", nargs="+", type=float, default=[0.5, 2.0])
    p.add_argument("--scenarios", nargs="+", default=None)
    p.add_argument("--algorithms", nargs="+", default=BASELINES, choices=BASELINES)
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args()

    cfg = DATASETS[args.dataset]
    scenarios = args.scenarios or cfg["scenarios"]
    manifest = []

    for scenario in scenarios:
        if scenario not in RUNNERS:
            print(f"[sweep] no runner for {scenario}; skipping")
            continue
        for baseline in cfg["baselines"]:
            med = wpcst_median_size(args.dataset, scenario, baseline)
            if args.sweep == "budget":
                for rho in args.rhos:
                    b = budget_for(med, rho)
                    tag = f"bx{rho:g}"
                    manifest.append(dict(scenario=scenario, baseline=baseline,
                                         rho=rho, wpcst_median=med, budget=b))
                    for alg in args.algorithms:
                        run_one(args.dataset, scenario, baseline, alg,
                                importance=MAIN_IMPORTANCE, budget=b, tag=tag,
                                workers=args.workers)
            else:
                b = budget_for(med, 1.0)
                manifest.append(dict(scenario=scenario, baseline=baseline,
                                     rho=1.0, wpcst_median=med, budget=b))
                for alg in args.algorithms:
                    if alg in ("pappas2017", "mst"):
                        for imp in ("degree", "pagerank"):
                            run_one(args.dataset, scenario, baseline, alg,
                                    importance=imp, budget=b, tag="tune",
                                    workers=args.workers)
                    else:
                        for key in ("kg_type", "path_type", "relation"):
                            for imp in ("degree", "pagerank", "betweenness_approx"):
                                if key == MAIN_FACES_KEY and imp == MAIN_IMPORTANCE:
                                    continue  # main-table configuration
                                run_one(args.dataset, scenario, baseline, alg,
                                        importance=imp, budget=b,
                                        tag=f"tune_{key}", faces_key=key,
                                        workers=args.workers)

    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    mpath = SWEEP_ROOT / f"{args.dataset}_{args.sweep}_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"[sweep] wrote {mpath}")


if __name__ == "__main__":
    main()
