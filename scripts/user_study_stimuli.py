from __future__ import annotations
import argparse
import json
import random
import re
import subprocess
import tempfile
import textwrap
from collections import defaultdict
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from config import DATASETS, REPO_ROOT, RESULTS_ROOT, SEED

DATASET = "ml1m"
SCENARIO = "user_centric"
ALGORITHMS = {"st": "st", "pcst": "pcst", "wpcst": "wpcst_cent_degree"}
MAX_EDGES = 50
STRATA = [(1, 20), (21, 35), (36, 50)]
PER_STRATUM = 3
LABELS = ["A", "B", "C"]
WRAP = 14                # characters per line of a movie label
OUT = REPO_ROOT / "user_study"      # tracked folder, results/ is git-ignored


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path, key: str) -> dict[str, dict]:
    out = {}
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            r = json.loads(line)
            out[str(r[key])] = r
    return out


def load_summaries() -> dict[tuple[str, str], dict[str, dict]]:
    """(recommender, algorithm) -> user -> stored result record."""
    out = {}
    base = RESULTS_ROOT / DATASET / SCENARIO
    for rec in DATASETS[DATASET]["baselines"]:
        for alg, suffix in ALGORITHMS.items():
            path = base / f"{DATASET}_{SCENARIO}_{rec}_{suffix}.jsonl"
            out[(rec, alg)] = _read_jsonl(path, "anchor_id")
    return out


def load_recommendations() -> dict[str, dict[str, list[str]]]:
    out = {}
    for rec in DATASETS[DATASET]["baselines"]:
        path = Path(str(DATASETS[DATASET]["user_recs_template"]).format(baseline=rec))
        rows = _read_jsonl(path, "user_id")
        out[rec] = {u: [str(i) for i in r["recommended_items_ids"]] for u, r in rows.items()}
    return out


def load_graph_attributes(needed_edges: set[frozenset]) -> tuple[dict, dict]:
    """Node attributes of the KG and the base weight of the needed edges."""
    ns = "{http://graphml.graphdrawing.org/xmlns}"
    keys, nodes, weights = {}, {}, {}
    for _, el in ET.iterparse(str(DATASETS[DATASET]["graphml"]), events=("end",)):
        if el.tag == ns + "key":
            keys[el.get("id")] = el.get("attr.name")
        elif el.tag == ns + "node":
            nodes[el.get("id")] = {keys[d.get("key")]: d.text for d in el.findall(ns + "data")}
            el.clear()
        elif el.tag == ns + "edge":
            e = frozenset((el.get("source"), el.get("target")))
            if e in needed_edges:
                for d in el.findall(ns + "data"):
                    if keys[d.get("key")] == "weight":
                        weights[e] = float(d.text)
            el.clear()
    # names of attribute nodes are not in the GraphML, read them from e_map
    emap = Path(DATASETS[DATASET]["graphml"]).parent / "e_map.txt"
    if emap.exists():
        with open(emap, "r", encoding="utf-8") as fin:
            fin.readline()
            for line in fin:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[0] in nodes and not nodes[parts[0]].get("name"):
                    nodes[parts[0]]["name"] = parts[1]
    return nodes, weights


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def build_pool(summaries) -> pd.DataFrame:
    rows = []
    for rec in DATASETS[DATASET]["baselines"]:
        for user in summaries[(rec, "st")]:
            sizes = {}
            for alg in ALGORITHMS:
                r = summaries[(rec, alg)].get(user)
                sizes[alg] = None if r is None else int(r["num_edges"])
            rows.append({"recommender": rec, "user_id": user, **sizes})
    pool = pd.DataFrame(rows)
    ok = pool[list(ALGORITHMS)].notna().all(axis=1)
    ok &= pool[list(ALGORITHMS)].min(axis=1) >= 1
    ok &= pool[list(ALGORITHMS)].max(axis=1) <= MAX_EDGES
    pool["eligible"] = ok
    pool["stratum"] = ""
    for lo, hi in STRATA:
        m = ok & pool["wpcst"].between(lo, hi)
        pool.loc[m, "stratum"] = f"{lo}-{hi}"
    return pool


def sample_cases(pool: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(SEED)
    chosen, used = [], set()
    for lo, hi in STRATA:
        cand = pool[pool["stratum"] == f"{lo}-{hi}"].sort_values(["recommender", "user_id"])
        idx = list(cand.index)
        rng.shuffle(idx)
        picked = 0
        for i in idx:
            if pool.at[i, "user_id"] in used:
                continue
            chosen.append(i)
            used.add(pool.at[i, "user_id"])
            picked += 1
            if picked == PER_STRATUM:
                break
    cases = pool.loc[chosen].copy()
    order = list(range(len(cases)))
    rng.shuffle(order)                       # interleave the strata in the form
    cases = cases.iloc[order].reset_index(drop=True)
    cases["case"] = [f"U{i + 1}" for i in range(len(cases))]
    # balanced labels: a cyclic Latin square over the cases, in shuffled order
    algs = list(ALGORITHMS)
    rows = [algs[k % 3:] + algs[:k % 3] for k in range(len(cases))]
    rng.shuffle(rows)
    for j, lab in enumerate(LABELS):
        cases[f"label_{lab}"] = [r[j] for r in rows]
    return cases


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def clean_name(name: str) -> str:
    name = re.sub(r"^Category:", "", name)
    name = re.sub(r"_\((\d{4}_)?film\)$", "", name)
    return name.replace("_", " ").replace('"', "'")


def _label(name: str) -> str:
    return "\\n".join(textwrap.wrap(clean_name(name), WRAP, break_long_words=False)) or name


def to_dot(record: dict, user: str, recommended: set[str], nodes: dict,
           weights: dict, alias: dict[str, str], rankdir: str) -> str:
    """Tree drawing rooted at the user. rankdir is LR or TB."""
    lines = [
        f"digraph G {{ rankdir={rankdir}; ranksep=0.2; nodesep=0.06; pad=0.1; splines=true;",
        'node [fontname="Helvetica", fontsize=11, margin="0.05,0.03", height=0.22, width=0.3];',
        'edge [color="#777777", arrowhead=none];',
    ]
    sol_nodes = [str(n) for n in record["solution_nodes"]]
    for n in sol_nodes:
        kind = nodes.get(n, {}).get("type", "external")
        if n == user:
            lines.append(f'"{n}" [label="User", shape=box, style="filled,bold", '
                         'fillcolor="#1f3a93", fontcolor=white];')
        elif kind == "user":
            lines.append(f'"{n}" [label="{alias[n]}", shape=ellipse, style=filled, fillcolor="#eeeeee"];')
        elif kind == "item":
            label = _label(nodes[n].get("name") or n)
            fill = ', style="filled,rounded", fillcolor="#ffd27f"' if n in recommended else ", style=rounded"
            lines.append(f'"{n}" [label="{label}", shape=box{fill}];')
        else:
            label = _label(nodes.get(n, {}).get("name") or n)
            lines.append(f'"{n}" [label="{label}", shape=plaintext, fontcolor="#444444"];')

    def edge(u: str, v: str, extra: str = "") -> str:
        w = weights.get(frozenset((u, v)), 0.0)
        style = f"penwidth={0.4 + 1.6 * w:.2f}" if w > 0 else "style=dashed, penwidth=0.8"
        return f'"{u}" -> "{v}" [{style}{extra}];'

    adj = defaultdict(list)
    for a, b in record["solution_edges"]:
        adj[str(a)].append(str(b))
        adj[str(b)].append(str(a))
    # breadth-first orientation from the user (or from the first node if the
    # summary omits the user), one start per connected component
    seen, drawn = set(), set()
    for root in ([user] if user in sol_nodes else []) + sol_nodes:
        if root in seen:
            continue
        seen.add(root)
        queue = [root]
        while queue:
            u = queue.pop(0)
            for v in sorted(adj[u]):
                if v not in seen:
                    seen.add(v)
                    queue.append(v)
                    drawn.add(frozenset((u, v)))
                    lines.append(edge(u, v))
    for a, b in record["solution_edges"]:          # edges that close a cycle, if any
        if frozenset((str(a), str(b))) not in drawn:
            lines.append(edge(str(a), str(b), ", constraint=false"))
    lines.append("}")
    return "\n".join(lines)


def _png_width(path: Path) -> int:
    with open(path, "rb") as fin:
        head = fin.read(24)
    return int.from_bytes(head[16:20], "big")


def render(record: dict, user: str, recommended: set[str], nodes: dict,
           weights: dict, alias: dict[str, str], path: Path) -> tuple[str, int]:
    """Draw left to right and top down, keep the narrower drawing."""
    best = None
    with tempfile.TemporaryDirectory() as tmp:
        for rankdir in ("LR", "TB"):
            out = Path(tmp) / f"{rankdir}.png"
            src = to_dot(record, user, recommended, nodes, weights, alias, rankdir)
            subprocess.run(["dot", "-Tpng", "-Gdpi=110", "-o", str(out)],
                           input=src.encode("utf-8"), check=True)
            width = _png_width(out)
            if best is None or width < best[1]:
                best = (rankdir, width, out.read_bytes())
    path.write_bytes(best[2])
    return best[0], best[1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n")[1]).parse_args()
    (OUT / "stimuli").mkdir(parents=True, exist_ok=True)

    summaries = load_summaries()
    recs = load_recommendations()
    pool = build_pool(summaries)
    pool.to_csv(OUT / "pool.csv", index=False)
    cases = sample_cases(pool)

    needed = set()
    for _, c in cases.iterrows():
        for alg in ALGORITHMS:
            for a, b in summaries[(c.recommender, alg)][c.user_id]["solution_edges"]:
                needed.add(frozenset((str(a), str(b))))
    nodes, weights = load_graph_attributes(needed)

    key_rows, case_rows = [], []
    for _, c in cases.iterrows():
        recommended = recs[c.recommender][c.user_id]
        recset = set(recommended)
        # one alias per other viewer, shared by the three summaries of the case
        others = []
        for lab in LABELS:
            r = summaries[(c.recommender, c[f"label_{lab}"])][c.user_id]
            for n in map(str, r["solution_nodes"]):
                if n != c.user_id and nodes.get(n, {}).get("type") == "user" and n not in others:
                    others.append(n)
        alias = {n: f"Viewer {i + 1}" for i, n in enumerate(others)}
        for lab in LABELS:
            alg = c[f"label_{lab}"]
            r = summaries[(c.recommender, alg)][c.user_id]
            orientation, width_px = render(r, c.user_id, recset, nodes, weights, alias,
                                           OUT / "stimuli" / f"{c.case}_{lab}.png")
            n_edges = int(r["num_edges"])
            relevance = sum(weights.get(frozenset((str(a), str(b))), 0.0) for a, b in r["solution_edges"])
            shown = recset & set(map(str, r["solution_nodes"]))
            key_rows.append({
                "case": c.case, "label": lab, "algorithm": alg,
                "recommender": c.recommender, "user_id": c.user_id, "stratum": c.stratum,
                "num_nodes": int(r["num_nodes"]), "num_edges": n_edges,
                "comprehensibility": 1.0 / (n_edges + 1), "relevance": round(relevance, 4),
                "terminal_coverage": len(shown) / len(recset),
                "orientation": orientation, "width_px": width_px,
            })
        case_rows.append({
            "case": c.case, "recommender": c.recommender, "user_id": c.user_id, "stratum": c.stratum,
            "recommended_movies": " | ".join(clean_name(nodes[i].get("name") or i) for i in recommended),
        })

    pd.DataFrame(key_rows).to_csv(OUT / "stimuli_key.csv", index=False)
    pd.DataFrame(case_rows).to_csv(OUT / "cases.csv", index=False)

    n_el = int(pool["eligible"].sum())
    print(f"Eligible pairs: {n_el} of {len(pool)} ({100 * n_el / len(pool):.1f}%)")
    print(pool[pool.eligible].groupby(["stratum", "recommender"]).size().unstack(fill_value=0).to_string())
    print(pd.DataFrame(key_rows).pivot(index=["case", "stratum", "recommender"],
                                       columns="algorithm", values="num_edges").to_string())
    print(f"Wrote {len(key_rows)} diagrams to {OUT / 'stimuli'}")


if __name__ == "__main__":
    main()
