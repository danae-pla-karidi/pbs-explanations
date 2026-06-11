"""
Central configuration for the TKDE summary-explanations experiments.

Editing this file is the only thing a user should need to do to point
the pipeline at their data. All scripts import constants from here.
"""

from __future__ import annotations
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Set REPO_ROOT to the absolute path of this repository on your machine.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
RESULTS_ROOT = REPO_ROOT / "results"
SAMPLES_DIR = REPO_ROOT / "samples"
LOGS_DIR = REPO_ROOT / "logs"

for d in (DATA_ROOT, RESULTS_ROOT, SAMPLES_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
# Each dataset declares (a) the GraphML file and (b) which recommender baselines
# have recommendation files / item-path files available.

DATASETS = {
    "ml1m": {
        "graphml": DATA_ROOT / "ml1m" / "kg_static.graphml",
        "user_recs_template": DATA_ROOT / "ml1m" / "{baseline}_recommendations.jsonl",
        "item_paths_template": DATA_ROOT / "ml1m" / "{baseline}_item_paths_processed.jsonl",
        "baselines": ["pgpr", "cafe", "plm", "plmr"],
        "scenarios": ["user_centric", "user_group", "item_centric", "item_group"],
    },
    "lfm1m": {
        "graphml": DATA_ROOT / "lfm1m" / "lmfm_kg_static_final.graphml",
        "user_recs_template": DATA_ROOT / "lfm1m" / "{baseline}_recommendations_lmfm.jsonl",
        "item_paths_template": None,  # not available for LFM-1M
        # Graph carries real Last.fm user IDs (e.g. u1003134);
        # recommenders trained on a 1-indexed sequential mapping (u1..u4817).
        # File source: PEARLM preprocessed/mapping/user.txt.
        "uid_mapping_file": DATA_ROOT / "lfm1m" / "user_uid_map.tsv",
        "uid_mapping_offset": 1,  # rec id = new_id + 1
        "baselines": ["pgpr", "cafe"],
        "scenarios": ["user_centric", "user_group"],
    },
}

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
N_USERS_PER_GENDER = 100      # 100 M + 100 F per dataset (200 total, deduplicated)
N_ITEMS = 100                  # for item-centric, stratified by popularity quartile
GROUP_SCENARIO_GROUPS = ["male", "female"]  # for user-group; popular/unpopular for item-group

# ---------------------------------------------------------------------------
# Algorithm hyperparameters
# ---------------------------------------------------------------------------
TOP_K = 10     # top-K recommended items per user (and items per group)
LAMBDA = 1.0   # WPCST reweighting strength. ICDE'25 found lam=1 optimal; we
               # use the same value across both datasets and all scenarios.
GAMMA = 0.1    # WPCST prize ratio (non-terminal vs terminal)

# Betweenness approximation: number of sampled source nodes
BETWEENNESS_K_SOURCES = 1000

# ---------------------------------------------------------------------------
# Algorithms and centralities to evaluate
# ---------------------------------------------------------------------------
ALGORITHMS = ["st", "pcst", "wpcst", "naive_union",
              "pappas2017", "mst", "faces", "supernode"]
CENTRALITIES = ["degree", "pagerank", "betweenness_approx"]
# Default centrality used by WPCST when the variant is not specified
DEFAULT_CENTRALITY = "degree"

# ---------------------------------------------------------------------------
# Runtime / parallelism
# ---------------------------------------------------------------------------
NUM_WORKERS = 6
CHUNK_SIZE_USER = 8
CHUNK_SIZE_ITEM = 4

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
# Metrics computed per anchor and aggregated downstream.
#
# Note on `consistency`: the original ICDE pipeline included a "consistency"
# metric defined as the mean pairwise Jaccard between explanation node-sets
# that share an anchor.  In our per-file aggregation pipeline (one algorithm
# per call to `compute_per_anchor`), every anchor has exactly one record,
# which makes the metric trivially 1.0.  We dropped it rather than rewire
# the pipeline; the cross-algorithm comparison the metric was meant to
# capture is already what the CR-tradeoff figure and the Wilcoxon contrasts
# do directly.
METRICS = [
    "comprehensibility",      # 1 / (num_edges + 1)
    "relevance",              # mean edge weight (normalized to [0,1])
    "actionability",          # fraction of solution nodes that neighbor the anchor
    "diversity",              # |unique nodes| / |solution nodes|
    "redundancy",             # 1 - diversity
    "privacy",                # 1 - (user nodes / total nodes)
    "coverage",               # fraction of top-K paths fully preserved in the summary
    "faithfulness",           # fraction of node+edge content of top-K paths preserved
    "runtime_sec",            # wall-clock (per-anchor)
    "cpu_time_sec",           # user+system CPU (per-anchor); cleaner than wall-clock under contention
    "memory_usage_mb",        # net RAM increase (per-anchor)
    "pop_frac",               # fraction of items in summary that are popular (q4)
    "pop_frac_topk",          # same fraction on the recommender's top-K item set (baseline)
]
# Comprehensibility-Gap is computed at the (scenario,baseline,algorithm) level only
# because it requires aggregating over a demographic group split.

# Significance test settings
SIGNIFICANCE_ALPHA = 0.05
SIGNIFICANCE_TEST = "wilcoxon_signed_rank"
MULTIPLE_COMPARISONS_CORRECTION = "holm"  # over the metric family within each cell
