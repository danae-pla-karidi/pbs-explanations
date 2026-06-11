# Data

This repository contains **code only**. Input data is not redistributed and
must be placed under `data/<dataset>/` before running the pipeline. The exact
filenames the pipeline expects are defined by the templates in
`config/settings.py` (`graphml`, `user_recs_template`, `item_paths_template`,
`uid_mapping_file`).

## 1. Raw inputs — external sources

- **MovieLens-1M (ML1M)**: GroupLens, https://grouplens.org/datasets/movielens/1m/
- **LastFM-1M (LFM1M)**: subset of LastFM-1B, https://www.cp.jku.at/datasets/LFM-1b/
- **DBpedia attribute snapshots**: https://www.dbpedia.org/
- **KB4Rec mapping (ML1M ↔ DBpedia)**: https://github.com/RUCDM/KB4Rec

These are licensed for research use by their respective providers; we do not
redistribute them.

## 2. Recommender outputs — produced by external recommenders

The path-based recommenders are separate research artifacts. Working
implementations on ML1M / LFM1M are bundled in two tutorial repositories:

- **RecSys 2022 tutorial** (PGPR, CAFE):
  https://github.com/explainablerecsys/recsys2022
- **ECIR 2024 tutorial** (PGPR, CAFE, PLM, PLMR/PEARLM):
  https://github.com/explainablerecsys/ecir2024

Original method papers:
- **PGPR**: Xian et al., SIGIR 2019.
- **CAFE**: Xian et al., CIKM 2020.
- **PLM**: Geng et al., WWW 2022 — Path Language Modeling over KGs.
- **PLMR / PEARLM**: Balloccu et al., 2023 — Faithful Path Language Modelling.

Running a tutorial end-to-end produces the `*_recommendations*.jsonl` and
`*_item_paths_processed.jsonl` files the pipeline consumes. We do not
redistribute these outputs (derivative works of those codebases).

## 3. Knowledge graphs

The DBpedia-enriched KGs (`kg_static.graphml` for ML1M, `lmfm_kg_static_final.graphml`
for LFM1M) are built from the §1 sources via KB4Rec / DBpedia mapping. They are
not committed (each is ~100 MB).

## Expected layout

```
data/ml1m/
├── kg_static.graphml
├── pgpr_recommendations.jsonl
├── cafe_recommendations.jsonl
├── plm_recommendations.jsonl
├── plmr_recommendations.jsonl
├── pgpr_item_paths_processed.jsonl
├── cafe_item_paths_processed.jsonl
├── plm_item_paths_processed.jsonl
└── plmr_item_paths_processed.jsonl

data/lfm1m/
├── lmfm_kg_static_final.graphml
├── pgpr_recommendations_lmfm.jsonl
├── cafe_recommendations_lmfm.jsonl
└── user_uid_map.tsv          # rec-id ↔ graph-id mapping (see below)
                              # no item_paths files: LFM1M is user-side only
```

### LFM1M user-ID mapping

The LFM1M KG carries real LastFM user IDs (e.g. `u1003134`), while the
recommenders are trained on a 1-indexed sequential ID space (`u1..u4817`).
`user_uid_map.tsv` (header `new_id<TAB>uid`; rec id = `new_id + 1`, source:
PEARLM `preprocessed/mapping/user.txt`) lets the runners translate anchors and
user-incident path nodes back into graph-ID space. Without it the anchor is
dropped from its own summary and the path-aware reweighting becomes a no-op.
The translation is a no-op for ML1M (no `uid_mapping_file` declared).

## Sampling

Anchor sets are seeded (`SEED = 42`) and written to `samples/<dataset>_users.csv`
and `samples/<dataset>_items.csv` by `sampling/make_samples.py`, then loaded —
never re-sampled — by every runner. The committed `samples/*.csv` are the anchor
IDs used in the paper.

## File schemas

`*_recommendations*.jsonl` — one JSON object per user:
```
{"user_id": "u123", "rec_items": ["i1", "i2", ...]}
```

`*_item_paths_processed.jsonl` — one JSON object per (user, item):
```
{"user_id": "u123", "item_id": "i456", "path": ["u123", "g_genre_drama", "i456"]}
```

Per-anchor summary outputs (written under `results/`, git-ignored):
```
{"anchor_id": "u123", "vertices": [...], "edges": [[...], ...], "cpu_time_sec": 73.4, ...}
```

## Full re-run

1. Obtain raw inputs (§1) and reproduce recommender outputs (§2).
2. Build the KGs (§3) and place all files at the layout above.
3. Precompute centralities, then run `scripts/run_all_ml1m.sh` and
   `scripts/run_all_lfm1m.sh` (see [README.md](README.md)).

## License

- ML1M © GroupLens; LFM1M per its provider; both research-use only.
- DBpedia content is CC-BY-SA.
- The KG-construction, summarization, and metric code in this repo are MIT.
