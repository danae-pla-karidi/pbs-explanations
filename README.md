# Summary Explanations for Graph-Based Recommenders

Experimental pipeline for the **TKDE journal extension** of our ICDE 2025
paper *"Path-Based Summary Explanations for Graph Recommenders"* [1].

The pipeline aggregates per-recommendation explanation paths into a single
compact subgraph, formalized as a unified **anchor–terminal** bi-objective
problem (comprehensibility vs. relevance) over four scenarios: user-centric,
item-centric, user-group, item-group.

This is a **code-only** repository. Input data is not redistributed — see [DATA.md](DATA.md).

## What's new relative to the ICDE 2025 version [1]

- **WPCST** — Weighted Path-Aware Prize-Collecting Steiner Tree with adaptive,
  centrality-scaled prizes, under three centrality choices: degree, PageRank,
  approximate betweenness (`algorithms/wpcst.py`, `centralities/`).
- **Five structural baselines** — NaiveUnion, CentPrune, MST, FACES, SuperNode
  (`algorithms/`), run at a matched node budget against WPCST(degree). CentPrune (Pappas et al., ESWC 2017) is named pappas2017 in the code.
- **Two new metrics** — faithfulness and evidence density (relevance per retained edge), plus two fairness diagnostics: the Comprehensibility Gap (CG) and the Popular-Item Gap (PIG), reported with the amplification ratio of PIG against the recommender gap (metrics/).
- **Paired significance testing** — Wilcoxon signed-rank with Holm correction
  (`metrics/significance.py`).
- **Second dataset** — LastFM-1M, user-centric and user-group scenarios with PGPR and CAFE (see below).
- **Two added recommenders** — PLM and PLMR, alongside PGPR and CAFE.
- **Ablation runners** - the path-aware parameter lambda (runners/run_lambda_ablation.py) and the recency weighting over (beta1, beta2) settings (runners/generate_ablation_kgs.py, runners/run_beta_ablation.py). The WPCST centrality variants (degree, PageRank, approximate betweenness) run as a phase of the main sweeps.

## Algorithms and recommenders

Ten summarizers in total: ST, PCST, three WPCST variants (degree, PageRank, approximate betweenness), and the five structural baselines (NaiveUnion, CentPrune, MST, FACES, SuperNode). Each is applied on top of four path-based recommenders on ML1M
(**PGPR, CAFE, PLM, PLMR**) and two on LFM1M (**PGPR, CAFE**).

### LFM1M scope

LFM1M is a cross-dataset replication on the user-centric and user-group scenarios. Only PGPR and CAFE are
trained on it, and item-side path files are unavailable, so the item-centric
and item-group scenarios are skipped on LFM1M. Approximate betweenness is also
skipped on LFM1M (~1 CPU-day per pass at the ML1M sampling rate). The baselines and scenarios declared for each dataset are encoded in config/settings.py.

## Layout

```
.
├── config/        # paths, hyperparameters, seeds, per-dataset baseline/scenario lists
├── sampling/      # seeded user/item sampling
├── algorithms/    # ST, PCST, WPCST + NaiveUnion, CentPrune (pappas2017.py), MST, FACES, SuperNode
├── centralities/  # degree, PageRank, approximate betweenness
├── runners/       # one runner per scenario, plus the λ-ablation runner
├── metrics/       # per-anchor metrics, aggregation, paired Wilcoxon significance
├── plots/         # comprehensibility–relevance trade-off figure
├── scripts/       # orchestration (run_all_*.sh, budgets, external-baseline phase,
│                  #   patched pcst_fast installer, LFM1M table helper)
├── tests/         # self-contained smoke tests (no real data required)
└── samples/       # seeded anchor IDs (committed for reference)
```

`data/`, `results/`, and `logs/` are git-ignored. `data/` is populated per
[DATA.md](DATA.md); `results/` and `logs/` are produced by the run scripts.

## Quickstart

```bash
git clone https://github.com/danae-pla-karidi/pbs-explanations.git
cd pbs-explanations

# 1. Python dependencies
pip install -r requirements.txt

# 2. Patched pcst_fast (the PyPI 1.0.10 binding corrupts result arrays;
#    this builds the one-line C++ fix and installs it)
bash scripts/install_pcst_fast.sh

# 3. Smoke tests (build their own toy KG; no real data needed)
python -m tests.smoke
python -m tests.smoke_uid_translation
python -m tests.smoke_lambda_ablation
```

`tests/smoke.py` prints per-algorithm summary sizes and ends with
`Smoke test passed.` if the wiring and the patched solver are correct.

## Running the full pipeline

Populate `data/` per [DATA.md](DATA.md), then:

```bash
# Precompute centralities (betweenness on ML1M is the slow one, ~6 h;
# run it once, cached to results/_centrality_cache/)
python -m centralities --dataset ml1m  --centrality degree
python -m centralities --dataset ml1m  --centrality pagerank
python -m centralities --dataset ml1m  --centrality betweenness_approx
python -m centralities --dataset lfm1m --centrality degree
python -m centralities --dataset lfm1m --centrality pagerank

# Per-dataset sweeps (sampling -> core algorithms -> NaiveUnion ->
# matched-budget baselines -> alt-centralities -> per-dataset metrics)
bash scripts/run_all_ml1m.sh
bash scripts/run_all_lfm1m.sh

# Significance and figures
python -m metrics.significance
python -m plots.make_figures
```

Each sweep is phase-structured and re-runnable; restart from a phase by passing
its number, e.g. `bash scripts/run_all_ml1m.sh 4` to restart at the external
baselines. The set of recommenders and scenarios swept is configured at the top
of each `run_all_*.sh` and in `config/settings.py`.

### λ-ablation

```bash
python -m runners.run_lambda_ablation --dataset ml1m --baseline pgpr \
    --scenario user_centric --algorithm wpcst --centrality degree
```

Sweeps the path-aware reweighting strength lambda of Eq. (2) over {0.01, 1, 100} for ST and WPCST(degree), on the user-centric scenario, scoring outputs with the same metric functions as the main pipeline. The main runs fix lambda = 1.

## Outputs

Per-cell summaries and derived tables are written under `results/` (git-ignored):

```
results/<dataset>/<scenario>/<dataset>_<scenario>_<recommender>_<algorithm>[_cent_<X>].jsonl
results/per_anchor/<...>.parquet          # per-anchor metrics (needed for Wilcoxon)
results/<dataset>/summary.csv             # per-cell aggregation
results/<dataset>/significance*.csv|.tex  # paired Wilcoxon (Holm)
results/<dataset>/figures/                # CR-tradeoff PDFs
```

## Hyperparameters

Defaults in `config/settings.py`:

| Parameter | Value | Use |
|---|---|---|
| `SEED` | 42 | sampling and all randomized steps |
| `TOP_K` | 10 | top-K items per user/group |
| `LAMBDA` | 1.0 | path-aware edge boost, Eq. (2), applied by ST, PCST, and WPCST |
| `GAMMA` | 0.1 | WPCST non-terminal/terminal prize ratio |
| `BETWEENNESS_K_SOURCES` | 1000 | sampled sources for approximate betweenness |
| `NUM_WORKERS` | 6 | per-scenario parallelism |

`gamma = 0.1` follows the gamma ablation over {0.1, 0.3, 0.5} reported in the paper. The main runs use edge weights with beta1 = 1 and beta2 = 0 in Eq. (1), i.e. rating only, no recency term. The recency ablation regenerates the KGs under other (beta1, beta2) settings.

## Compute notes

- Reference machine: 6-core Linux workstation, 32 GB RAM.
- The dominant per-anchor cost in WPCST is the O(k^2) shortest-path queries that
  build `D_term`; it depends only on the anchor and its terminals (not on the
  centrality choice) and is cached on disk, so PageRank/betweenness variants
  reuse the warm cache.
- `pcst_fast` on PyPI (1.0.10) has a binding bug that returns corrupt arrays.
  `algorithms/pcst_fast_compat.py` self-checks at import and raises if an
  unpatched build is detected; `scripts/install_pcst_fast.sh` builds the fix.

## Citation

```bibtex
@inproceedings{plakaridi2025path,
  author    = {Pla Karidi, Danae and Pitoura, Evaggelia},
  title     = {Path-Based Summary Explanations for Graph Recommenders},
  booktitle = {2025 IEEE 41st International Conference on Data Engineering (ICDE)},
  year      = {2025},
  pages     = {973--986},
  publisher = {IEEE Computer Society},
  doi       = {10.1109/ICDE65448.2025.00078}
}
```

The extended journal version, "Summary Explanations for Graph-Based Recommenders", is under submission to IEEE TKDE. The citation will be updated on acceptance.

## License

Code is MIT (see `LICENSE`). Underlying data inherits the licenses of its
sources; see [DATA.md](DATA.md).

[1]: https://doi.ieeecomputersociety.org/10.1109/ICDE65448.2025.00078
