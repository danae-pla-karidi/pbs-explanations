from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

DATASETS = [("ml1m", "ML1M"), ("lfm1m", "LFM1M")]
SCENARIOS = [("user_centric", "User-centric"), ("item_centric", "Item-centric"),
             ("user_group", "User-group"), ("item_group", "Item-group")]
RECS = [("pgpr", "PGPR"), ("cafe", "CAFE"), ("plm", "PLM"), ("plmr", "PLMR")]
CENTRALITIES = ["degree", "pagerank", "betweenness_approx"]
TOL = 5e-5


def result_file(ds: str, sc: str, rec: str, cent: str) -> Path:
    return RESULTS / ds / sc / f"{ds}_{sc}_{rec}_wpcst_cent_{cent}.jsonl"


def load_alphas(path: Path) -> dict[str, float]:
    """anchor_id -> alpha. Rows without alpha (empty or trivial runs) are skipped."""
    out = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            a = (row.get("metadata") or {}).get("alpha")
            if a is not None:
                out[str(row["anchor_id"])] = float(a)
    return out


def is_fallback(alpha: float) -> bool:
    return abs(alpha - 2.0) < TOL


def check_centralities(ds: str, sc: str, rec: str, flags: dict[str, bool], cent: str) -> None:
    for other in CENTRALITIES:
        p = result_file(ds, sc, rec, other)
        if other == cent or not p.exists():
            continue
        other_flags = {k: is_fallback(a) for k, a in load_alphas(p).items()}
        diff = [k for k in flags.keys() & other_flags.keys() if flags[k] != other_flags[k]]
        if diff:
            print(f"% WARN {ds}/{sc}/{rec}: fallback flags differ between "
                  f"{cent} and {other} for {len(diff)} anchors")


def fmt_r(r: np.ndarray, n: int) -> str:
    if r.size == 0:
        return "--"
    med = np.median(r)
    if n < 5:  # IQR is meaningless for the two-anchor group cells
        return f"{med:.2f}"
    q1, q3 = np.percentile(r, [25, 75])
    return f"{med:.2f} [{q1:.2f}, {q3:.2f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--centrality", default="degree", choices=CENTRALITIES)
    ap.add_argument("--out", default=None, help="optional .tex file to write")
    args = ap.parse_args()

    rows = []
    for ds, ds_name in DATASETS:
        for sc, sc_name in SCENARIOS:
            for rec, rec_name in RECS:
                p = result_file(ds, sc, rec, args.centrality)
                if not p.exists():
                    continue
                alphas = load_alphas(p)
                flags = {k: is_fallback(a) for k, a in alphas.items()}
                check_centralities(ds, sc, rec, flags, args.centrality)
                n = len(alphas)
                n_fb = sum(flags.values())
                r = np.array([a - 1.0 for k, a in alphas.items() if not flags[k]])
                rows.append((ds_name, sc_name, rec_name, n, n_fb, fmt_r(r, n)))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Frequency of the $D_{\mathrm{term}}$ fallback of WPCST "
        r"($D_{\mathrm{term}} \leftarrow W_{\mathrm{avg}}$, hence $\alpha = 2$) "
        r"and ratio $r = D_{\mathrm{term}}/W_{\mathrm{avg}}$ over the non-fallback anchors "
        r"(median [IQR], median only for the two-anchor group cells).}",
        r"\label{tab:dterm-fallback}",
        r"\begin{tabular}{lllrrl}",
        r"\toprule",
        r"Dataset & Scenario & Recommender & Anchors & Fallback (\%) & $r$ \\",
        r"\midrule",
    ]
    prev = None
    for ds_name, sc_name, rec_name, n, n_fb, r_str in rows:
        if prev is not None and ds_name != prev:
            lines.append(r"\midrule")
        prev = ds_name
        pct = 100.0 * n_fb / n if n else float("nan")
        lines.append(f"{ds_name} & {sc_name} & {rec_name} & {n} & "
                     f"{pct:.1f} ({n_fb}/{n}) & {r_str} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    tex = "\n".join(lines)
    print(tex)
    if args.out:
        Path(args.out).write_text(tex + "\n")


if __name__ == "__main__":
    main()
