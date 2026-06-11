"""
Seeded sample selection for both ML1M and LFM-1M.

Replaces the hardcoded user/item lists embedded in the original ICDE/TKDE-draft
scripts. Sample IDs are written to CSV under <repo>/samples/ and loaded by every
downstream runner.

Why this exists:
- The original code used Python literal lists baked into each runner. ML1M's
  user list contained 200 entries with 198 unique IDs (u601, u1472 duplicated);
  the item list likewise had duplicates. No random seed was set anywhere.
- Reviewers asked for explicit reproducibility; this script is the answer.

Usage:
    python -m sampling.make_samples --dataset ml1m
    python -m sampling.make_samples --dataset lfm1m
    python -m sampling.make_samples --dataset all
"""

from __future__ import annotations
import argparse
import random
from pathlib import Path
from typing import Iterable

import networkx as nx
import pandas as pd

from config import (
    DATASETS,
    SEED,
    SAMPLES_DIR,
    N_USERS_PER_GENDER,
    N_ITEMS,
)


def _eligible_items(dataset: str) -> set[str] | None:
    """Items present in *every* configured baseline's item-paths file.

    Item-centric and item-group scenarios anchor on items, with terminals
    drawn from the recommender's item-paths file (per item, "users this
    was recommended to" + their explanation paths).  Items that do not
    appear in a baseline's paths file have no terminals available for
    that baseline -- the runner would emit a "no item-path record" warning
    and skip them, leading to under-populated cells and cells of unequal
    size across baselines (which breaks the paired Wilcoxon design).

    The fix is to sample only items that appear in *all* baselines'
    paths files, so that every (baseline, algorithm) cell sees the same
    anchor set.

    Returns None if item-paths are not configured for this dataset
    (e.g. LFM-1M, where item-centric scenarios are skipped entirely).
    """
    import json

    cfg = DATASETS[dataset]
    template = cfg.get("item_paths_template")
    if template is None:
        return None
    common: set[str] | None = None
    for baseline in cfg["baselines"]:
        path = Path(str(template).format(baseline=baseline))
        if not path.exists():
            print(f"[sampling] WARNING: item-paths file missing for "
                  f"{dataset}/{baseline}: {path}.  This baseline will be "
                  "excluded from item-eligibility -- expect sample size "
                  "to be smaller than configured.")
            continue
        with open(path, "r", encoding="utf-8") as fin:
            ids = {str(json.loads(line)["item_id"]) for line in fin}
        common = ids if common is None else (common & ids)
    return common or set()


# ---------------------------------------------------------------------------
# Graph loading (read once, reuse)
# ---------------------------------------------------------------------------

def _load_graph(graphml_path: Path) -> nx.Graph:
    if not graphml_path.exists():
        raise FileNotFoundError(
            f"Graph file not found: {graphml_path}. "
            "Did you place the dataset under data/<dataset>/?"
        )
    print(f"[sampling] reading {graphml_path}")
    return nx.read_graphml(graphml_path, node_type=str)


# ---------------------------------------------------------------------------
# User sampling: stratified by gender
# ---------------------------------------------------------------------------

def _split_users_by_gender(G: nx.Graph) -> tuple[list[str], list[str], list[str]]:
    """Return (male_users, female_users, ungendered_users) found in G.

    Both ML1M and LFM-1M nominally store user gender as a node attribute,
    but the attribute key name varies by dataset and by how the graph was
    serialized:
      * ML1M (KB4Rec):     attr.name="gender"  -> data["gender"]
      * LFM-1M (older):    serialized under data-key "d2" with no
                           attr.name, so it surfaces as data["d2"];
      * Newer LFM-1M dumps may use "gender" too.
      * Some LFM-1M dumps have ``gender`` and ``age`` swapped at build
        time (the gender slot holds the age band string, the age slot
        holds the M/F letter).  We detect this by checking whether the
        canonical 'gender' field contains an obvious age-band pattern
        (e.g. "25-34") while the 'age' field is "M" or "F", and swap
        the lookup if so.
    We try several conventional keys and finally fall back to "no gender
    attribute" (third return value) so the caller can decide whether to
    sample without gender stratification.
    """
    # Detect the swapped-label case by inspecting the first user node.
    swapped = False
    for node, data in G.nodes(data=True):
        is_user = (data.get("type") == "user"
                   or (isinstance(node, str) and node.startswith("u")))
        if not is_user:
            continue
        g_val, a_val = data.get("gender"), data.get("age")
        # Heuristic: gender contains a hyphen or "+" (age-band pattern)
        # AND age is a single letter that looks like M/F.
        if (isinstance(g_val, str) and ("-" in g_val or g_val.endswith("+"))
                and isinstance(a_val, str) and a_val in ("M", "F")):
            swapped = True
            print(f"[sampling] WARNING: detected swapped gender/age "
                  f"labels in graphml (e.g. gender={g_val!r}, "
                  f"age={a_val!r}); swapping at read time.")
        break

    males, females, ungendered = [], [], []
    user_count = 0
    for node, data in G.nodes(data=True):
        is_user = (data.get("type") == "user"
                   or (isinstance(node, str) and node.startswith("u")))
        if not is_user:
            continue
        user_count += 1
        if swapped:
            gender = data.get("age")
        else:
            gender = (data.get("gender")
                      or data.get("d1")    # ML1M's serialized key
                      or data.get("d2")    # LFM-1M older dumps
                      or data.get("sex"))
        if gender == "M":
            males.append(node)
        elif gender == "F":
            females.append(node)
        else:
            ungendered.append(node)
    print(f"[sampling] found {user_count} user nodes "
          f"({len(males)} M, {len(females)} F, {len(ungendered)} ungendered)")
    if user_count > 0 and len(males) + len(females) == 0:
        for node, data in G.nodes(data=True):
            if (data.get("type") == "user"
                or (isinstance(node, str) and node.startswith("u"))):
                attrs = {k: v for k, v in data.items()}
                print(f"[sampling] no gender found.  Example user node "
                      f"{node!r}: attributes = {attrs}")
                break
    return males, females, ungendered


def _load_uid_mapping(dataset: str) -> dict[str, str] | None:
    """Load a graph-uid -> rec-uid mapping if one is configured.

    Some preprocessing pipelines (notably PEARLM's LFM-1M dump) build the
    knowledge graph using the *real* dataset user IDs (e.g. the actual
    Last.fm UIDs ``u1003134``, ``u21072247``) but train recommenders on a
    re-mapped sequential ID space (``u1``, ``u2``, ..., 1-indexed) and
    write recommendation files in that smaller space.  When the two ID
    spaces don't match, sampling on the graph and querying the
    recommendations file produces zero overlap.

    To bridge the two, the dataset config can specify
    ``uid_mapping_file``: a tab-separated file with header
    ``new_id<TAB>uid``, where ``new_id`` is 0-indexed and ``uid`` is the
    real dataset user ID.  We translate at sample time so the sample CSV
    is written in *rec-space* (the runner's native space).

    Off-by-one note: PEARLM's mapping is 0-indexed but the recommendations
    files use 1-indexed IDs (``u1`` corresponds to ``new_id=0``), so we
    add 1.  The handling matches what we observed in the LFM-1M
    artefacts; if you have a mapping that's already 1-indexed or in some
    other format, set ``uid_mapping_offset`` in the dataset config.

    Returns a dict mapping ``"u<real_uid>"`` -> ``"u<rec_uid>"``, or None
    if no mapping is configured.
    """
    cfg = DATASETS[dataset]
    map_path = cfg.get("uid_mapping_file")
    if map_path is None:
        return None
    map_path = Path(str(map_path))
    if not map_path.exists():
        print(f"[sampling] WARNING: uid_mapping_file declared but not "
              f"found at {map_path}; falling back to no translation.")
        return None
    offset = cfg.get("uid_mapping_offset", 1)
    mapping: dict[str, str] = {}
    with open(map_path, "r", encoding="utf-8") as fin:
        header = fin.readline().strip().split("\t")
        try:
            new_id_col = header.index("new_id")
            uid_col = header.index("uid")
        except ValueError:
            print(f"[sampling] WARNING: uid_mapping_file header "
                  f"{header!r} missing 'new_id' or 'uid' columns; "
                  "falling back to no translation.")
            return None
        for line in fin:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < max(new_id_col, uid_col) + 1:
                continue
            new_id = int(parts[new_id_col]) + offset
            uid = parts[uid_col]
            mapping[f"u{uid}"] = f"u{new_id}"
    print(f"[sampling] {dataset}: loaded UID mapping with "
          f"{len(mapping)} entries (graph -> rec)")
    return mapping


def _eligible_users(dataset: str) -> set[str] | None:
    """Users present in *every* configured baseline's recommendations file.

    Mirrors `_eligible_items`: user-centric and user-group scenarios
    anchor on users, and the runner reads each user's top-K
    recommendation paths from `<recommender>_recommendations.jsonl`.
    Sampling a user that no recommender has output for produces the
    "no recommendation record for u<X>" warning and a 0-anchor cell.

    The fix is to restrict the eligible user pool to users that appear
    in *every* recommender's file, so paired Wilcoxon across baselines
    is valid.  Returns None if the dataset has no recommendations files
    configured (defensive), in which case the caller should fall back
    to KG-only sampling.
    """
    import json
    cfg = DATASETS[dataset]
    template = cfg.get("user_recs_template")
    if template is None:
        return None
    common: set[str] | None = None
    for baseline in cfg["baselines"]:
        path = Path(str(template).format(baseline=baseline))
        if not path.exists():
            print(f"[sampling] WARNING: recommendations file missing "
                  f"for {dataset}/{baseline}: {path}.  This baseline "
                  "will be excluded from user-eligibility.")
            continue
        with open(path, "r", encoding="utf-8") as fin:
            ids = {str(json.loads(line)["user_id"]) for line in fin}
        common = ids if common is None else (common & ids)
    return common or set()


def sample_users(dataset: str, seed: int = SEED) -> pd.DataFrame:
    """Sample N_USERS_PER_GENDER males + N_USERS_PER_GENDER females.

    Returns a DataFrame with columns [user_id, gender]. Deduplicates by
    user_id.  Sampled users are restricted to the intersection of the
    KG user nodes (after optional uid translation) and the user IDs
    that appear in *every* recommender's recommendations file, so every
    (recommender, algorithm) cell sees the same anchor set.

    If the graph and the recommendations files use different ID spaces
    (LFM-1M case, where the graph carries real Last.fm IDs but the
    recommenders trained on sequential 1..N IDs), the optional
    ``uid_mapping_file`` config field translates graph IDs into rec
    IDs at sample time so the written CSV is directly consumable by
    the runners.

    Degrades gracefully: if the graph has no gender attribute on its
    user nodes, samples uniformly with gender='U' so user-centric still
    works (user-group will produce empty groups in that case).
    """
    cfg = DATASETS[dataset]
    G = _load_graph(cfg["graphml"])
    males, females, ungendered = _split_users_by_gender(G)
    rng = random.Random(seed)

    # If the dataset uses a separate ID space in the recommendations
    # files than in the graph, translate the graph-user ids into
    # recommendations-space before doing eligibility filtering.
    uid_map = _load_uid_mapping(dataset)
    if uid_map is not None:
        def translate(users: list[str]) -> list[str]:
            out, dropped = [], 0
            for u in users:
                t = uid_map.get(str(u))
                if t is None:
                    dropped += 1
                else:
                    out.append(t)
            if dropped:
                print(f"[sampling] {dataset}: dropped {dropped} graph "
                      f"users with no entry in the uid mapping")
            return out
        males = translate(males)
        females = translate(females)
        ungendered = translate(ungendered)

    # Restrict to users present in every recommender's recommendations file.
    eligible = _eligible_users(dataset)
    if eligible is not None:
        n_before = len(males) + len(females) + len(ungendered)
        males = [u for u in males if str(u) in eligible]
        females = [u for u in females if str(u) in eligible]
        ungendered = [u for u in ungendered if str(u) in eligible]
        n_after = len(males) + len(females) + len(ungendered)
        print(f"[sampling] {dataset}: {n_after}/{n_before} users "
              f"are common to all baselines' recommendations files")
        if n_after == 0:
            raise RuntimeError(
                f"No users are common to all configured baselines for "
                f"dataset {dataset!r}; check that the recommendations "
                "files exist and contain overlapping user IDs."
            )

    if len(males) == 0 and len(females) == 0:
        target = 2 * N_USERS_PER_GENDER
        if len(ungendered) < target:
            print(f"[sampling] WARNING: only {len(ungendered)} eligible "
                  f"user nodes available; using all of them")
        chosen = rng.sample(ungendered, min(target, len(ungendered)))
        df = pd.DataFrame([(u, "U") for u in chosen],
                          columns=["user_id", "gender"])
        df = df.drop_duplicates(subset="user_id").reset_index(drop=True)
        print(f"[sampling] {dataset}: kept {len(df)} unique users "
              f"(no gender stratification — user-group scenarios will "
              f"produce empty groups)")
        return df

    if len(males) < N_USERS_PER_GENDER:
        print(f"[sampling] WARNING: only {len(males)} eligible males in {dataset}; using all of them")
    if len(females) < N_USERS_PER_GENDER:
        print(f"[sampling] WARNING: only {len(females)} eligible females in {dataset}; using all of them")

    chosen_m = rng.sample(males, min(N_USERS_PER_GENDER, len(males)))
    chosen_f = rng.sample(females, min(N_USERS_PER_GENDER, len(females)))

    df = pd.DataFrame(
        [(u, "M") for u in chosen_m] + [(u, "F") for u in chosen_f],
        columns=["user_id", "gender"],
    )
    df = df.drop_duplicates(subset="user_id").reset_index(drop=True)
    print(f"[sampling] {dataset}: kept {len(df)} unique users "
          f"({(df.gender == 'M').sum()} M, {(df.gender == 'F').sum()} F)")
    return df


# ---------------------------------------------------------------------------
# Item sampling: stratified by popularity quartile
# ---------------------------------------------------------------------------

def _item_popularity(G: nx.Graph) -> dict[str, int]:
    """Popularity = number of users that interact with the item.

    We use the explicit ``type`` node attribute to identify items; this is
    the same attribute used by the original KG-builder scripts and the only
    reliable way to distinguish item nodes from external-entity nodes in
    the ML1M graph (both can have purely-numeric IDs: items are 0..2982,
    external entities are 5000+).  An earlier version of this function
    used ``s.isdigit()`` and silently sampled external entities as items,
    which produced empty item-path lookups for most sampled IDs.
    """
    pop: dict[str, int] = {}
    for node, data in G.nodes(data=True):
        if data.get("type") != "item":
            continue
        # Count user neighbours
        n_users = 0
        for nbr in G.neighbors(node):
            nbr_data = G.nodes[nbr]
            if nbr_data.get("type") == "user" or str(nbr).startswith("u"):
                n_users += 1
        pop[str(node)] = n_users
    return pop


def sample_items(dataset: str, seed: int = SEED) -> pd.DataFrame:
    """Sample N_ITEMS items stratified into popularity quartiles.

    25 items per quartile.  The pool is restricted to items that appear
    in *every* configured baseline's item-paths file, so that every
    (baseline, algorithm) cell sees the same anchors.  Quartiles are
    computed *after* this restriction (so the popular/unpopular split
    in item-group is meaningful relative to the eligible pool, not to
    the full KG).

    The 'quartile' column lets the item-group scenario use a popular vs
    unpopular split downstream (top vs bottom quartile).
    """
    cfg = DATASETS[dataset]
    G = _load_graph(cfg["graphml"])
    pop = _item_popularity(G)
    if not pop:
        raise RuntimeError(f"No item nodes found in {cfg['graphml']}")

    eligible = _eligible_items(dataset)
    if eligible is None:
        # Dataset has no item-paths configured (e.g. LFM-1M); item-side
        # scenarios are not run.  We still produce the CSV from the full
        # KG pool so that downstream code that calls load_items() doesn't
        # crash, but it goes unused.
        eligible_set = set(pop.keys())
    elif not eligible:
        raise RuntimeError(
            f"No items are common to all configured baselines for "
            f"dataset {dataset!r}; check that the item-paths files exist "
            "and contain overlapping item IDs."
        )
    else:
        eligible_set = eligible

    items_df = pd.DataFrame({"item_id": list(pop.keys()), "n_users": list(pop.values())})
    items_df = items_df[items_df["item_id"].isin(eligible_set)].reset_index(drop=True)
    print(f"[sampling] {dataset}: {len(items_df)} items eligible across all baselines")
    if len(items_df) < N_ITEMS:
        print(f"[sampling] WARNING: only {len(items_df)} eligible items, "
              f"requested {N_ITEMS}; will return what's available.")

    items_df["quartile"] = pd.qcut(items_df["n_users"].rank(method="first"), 4,
                                    labels=["q1_unpopular", "q2", "q3", "q4_popular"])
    rng = random.Random(seed)
    per_q = N_ITEMS // 4
    chosen_parts = []
    for q in ["q1_unpopular", "q2", "q3", "q4_popular"]:
        pool = items_df[items_df.quartile == q]["item_id"].tolist()
        if len(pool) < per_q:
            print(f"[sampling] WARNING: quartile {q} has only {len(pool)} items")
        chosen_parts.append(rng.sample(pool, min(per_q, len(pool))))
    chosen = [iid for part in chosen_parts for iid in part]
    out = items_df[items_df.item_id.isin(chosen)].reset_index(drop=True)
    out = out.drop_duplicates(subset="item_id").reset_index(drop=True)
    print(f"[sampling] {dataset}: kept {len(out)} unique items across 4 quartiles")
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _samples_path(dataset: str, kind: str) -> Path:
    return SAMPLES_DIR / f"{dataset}_{kind}.csv"


def write_samples(dataset: str) -> None:
    cfg = DATASETS[dataset]
    if "user_centric" in cfg["scenarios"]:
        df = sample_users(dataset)
        out = _samples_path(dataset, "users")
        df.to_csv(out, index=False)
        print(f"[sampling] wrote {out}")
    if "item_centric" in cfg["scenarios"]:
        df = sample_items(dataset)
        out = _samples_path(dataset, "items")
        df.to_csv(out, index=False)
        print(f"[sampling] wrote {out}")


def load_users(dataset: str) -> pd.DataFrame:
    path = _samples_path(dataset, "users")
    if not path.exists():
        raise FileNotFoundError(
            f"User sample not found: {path}. Run `python -m sampling.make_samples --dataset {dataset}` first."
        )
    return pd.read_csv(path)


def load_items(dataset: str) -> pd.DataFrame:
    path = _samples_path(dataset, "items")
    if not path.exists():
        raise FileNotFoundError(
            f"Item sample not found: {path}. Run `python -m sampling.make_samples --dataset {dataset}` first."
        )
    df = pd.read_csv(path)
    df["item_id"] = df["item_id"].astype(str)
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=list(DATASETS.keys()) + ["all"])
    args = p.parse_args(argv)
    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for ds in targets:
        write_samples(ds)


if __name__ == "__main__":
    main()
