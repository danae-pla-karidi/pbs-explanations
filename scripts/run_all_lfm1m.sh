#!/usr/bin/env bash
# Master driver for the LFM-1M sweep.
# Scope is intentionally smaller: only PGPR and CAFE recommenders, only
# user-centric and user-group scenarios. PLM/PLMR are not trained on LFM-1M
# and item_paths_processed.jsonl is not available, so item-* scenarios are
# skipped entirely.
#
# This is the "second-dataset confirmation" experiment for the TKDE submission;
# it validates the user-centric WPCST gain on a different domain (music).
#
# Usage:
#   bash scripts/run_all_lfm1m.sh
#   bash scripts/run_all_lfm1m.sh phase3   # restart from phase 3

set -euo pipefail

DATASET=lfm1m
PY=python3
START_PHASE=${1:-1}

cd "$(dirname "$0")/.."

scenarios=(user_centric user_group)
baselines=(pgpr cafe)

if [[ "$START_PHASE" -le 1 ]]; then
  echo "============================ Phase 1: sampling ============================"
  $PY -m sampling.make_samples --dataset $DATASET
fi

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

if [[ "$START_PHASE" -le 3 ]]; then
  echo "============================ Phase 3: NaiveUnion ============================"
  for scenario in "${scenarios[@]}"; do
    runner_module="runners.run_${scenario}"
    for baseline in "${baselines[@]}"; do
      $PY -m $runner_module --dataset $DATASET --baseline $baseline --algorithm naive_union
    done
  done
fi

if [[ "$START_PHASE" -le 4 ]]; then
  echo "============================ Phase 4: External baselines (Pappas, MST, FACES, SuperNode) ============================"
  $PY -m scripts.compute_subgraph_budgets --dataset $DATASET
  $PY -m scripts.run_external_baselines_phase --dataset $DATASET
fi

if [[ "$START_PHASE" -le 5 ]]; then
  echo "============================ Phase 5: alt centralities (PageRank only) ============================"
  # On LFM-1M we restrict to PageRank to fit the 2-week budget.
  # Betweenness on a graph this size with k-source approx is ~1 day per pass
  # and would not fit. We acknowledge this in the Limitations subsection.
  for scenario in "${scenarios[@]}"; do
    runner_module="runners.run_${scenario}"
    for baseline in "${baselines[@]}"; do
      echo "--- $scenario / $baseline / wpcst / pagerank ---"
      $PY -m $runner_module --dataset $DATASET --baseline $baseline \
        --algorithm wpcst --centrality pagerank || echo "  (failed; continuing)"
    done
  done
fi

if [[ "$START_PHASE" -le 6 ]]; then
  echo "============================ Phase 6: metrics + significance ============================"
  $PY -m metrics.compute_metrics --dataset $DATASET
fi

echo "==== LFM-1M sweep complete ===="
