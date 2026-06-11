#!/usr/bin/env bash
# Master driver: run the full ML1M sweep.
# Designed to be re-runnable: each phase writes to a distinct file path,
# so partial completions can be picked up where they stopped.
#
# Usage:
#   bash scripts/run_all_ml1m.sh           # run everything in order
#   bash scripts/run_all_ml1m.sh phase2    # start from phase 2
#
# Phase guide:
#   1. Sample users + items
#   2. Core algorithms: ST, PCST, WPCST-degree across all scenarios
#   3. Naive baselines: NaiveUnion across all scenarios
#   4. External baselines (Pappas2017, MST, FACES, SuperNode); needs phase 2 done first
#      to compute matched budgets
#   5. Alternative centralities: WPCST-PageRank, WPCST-BetweennessApprox
#      on user-centric and item-centric only
#   6. Metrics + significance + figures

set -euo pipefail

DATASET=ml1m
PY=python3
START_PHASE=${1:-1}

cd "$(dirname "$0")/.."

scenarios=(item_centric item_group)
baselines=(pgpr cafe plm plmr)

# ---- Phase 1: sampling ----
if [[ "$START_PHASE" -le 1 ]]; then
  echo "============================ Phase 1: sampling ============================"
  $PY -m sampling.make_samples --dataset $DATASET
fi

# ---- Phase 2: core algorithms ----
if [[ "$START_PHASE" -le 2 ]]; then
  echo "============================ Phase 2: ST/PCST/WPCST-degree ============================"
  for scenario in "${scenarios[@]}"; do
    runner_module="runners.run_${scenario}"
    for baseline in "${baselines[@]}"; do
      for alg in st pcst wpcst; do
        echo "--- $scenario / $baseline / $alg ---"
        $PY -m $runner_module --dataset $DATASET --baseline $baseline --algorithm $alg \
          --centrality degree || echo "  (failed; continuing)"
      done
    done
  done
fi

# ---- Phase 3: NaiveUnion ----
if [[ "$START_PHASE" -le 3 ]]; then
  echo "============================ Phase 3: NaiveUnion ============================"
  for scenario in "${scenarios[@]}"; do
    runner_module="runners.run_${scenario}"
    for baseline in "${baselines[@]}"; do
      $PY -m $runner_module --dataset $DATASET --baseline $baseline --algorithm naive_union
    done
  done
fi

# ---- Phase 4: External baselines (Pappas, MST, FACES, SuperNode) (matched budget) ----
if [[ "$START_PHASE" -le 4 ]]; then
  echo "============================ Phase 4: External baselines (Pappas, MST, FACES, SuperNode) ============================"
  $PY -m scripts.compute_subgraph_budgets --dataset $DATASET
  $PY -m scripts.run_external_baselines_phase --dataset $DATASET --scenarios "${scenarios[@]}"
fi

# ---- Phase 5: Alternative centralities ----
if [[ "$START_PHASE" -le 5 ]]; then
  echo "============================ Phase 5: alt centralities ============================"
  for cent in pagerank betweenness_approx; do
    for scenario in item_centric; do
      runner_module="runners.run_${scenario}"
      for baseline in "${baselines[@]}"; do
        echo "--- $scenario / $baseline / wpcst / $cent ---"
        $PY -m $runner_module --dataset $DATASET --baseline $baseline \
          --algorithm wpcst --centrality $cent || echo "  (failed; continuing)"
      done
    done
  done
fi

# ---- Phase 6: Metrics, significance, figures ----
if [[ "$START_PHASE" -le 6 ]]; then
  echo "============================ Phase 6: metrics + significance ============================"
  $PY -m metrics.compute_metrics --dataset $DATASET
  $PY -m metrics.significance
  $PY -m plots.make_figures
fi

echo "==== ML1M sweep complete ===="
