from __future__ import annotations
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from config import REPO_ROOT, SEED

OUT = REPO_ROOT / "user_study"
ALGS = ["st", "pcst", "wpcst"]
PAIRS = [("wpcst", "st"), ("wpcst", "pcst"), ("st", "pcst")]
MEASURES = ["ease", "evidence"]
QUESTIONS = ["quick", "detail"]
B = 10_000


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    adj, running = [0.0] * len(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def parse_responses(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long tables: ratings (participant, case, label, measure, value) and
    choices (participant, case, question, label)."""
    raw = pd.read_csv(path, dtype=str)
    raw.insert(0, "participant", [f"P{i + 1}" for i in range(len(raw))])
    ratings, choices = [], []
    for col in raw.columns:
        m = re.search(r"\b(U\d+)_([ABC])_(ease|evidence)\b", col)
        g = re.search(r"\b(U\d+)_(ease|evidence)\b.*\[(?:Explanation )?([ABC])\]", col)
        c = re.search(r"\b(U\d+)_(quick|detail)\b", col)
        if m or g:
            case, label, measure = (m.group(1), m.group(2), m.group(3)) if m else (g.group(1), g.group(3), g.group(2))
            vals = raw[col].str.extract(r"^\s*(\d)")[0].astype(float)
            for pid, v in zip(raw["participant"], vals):
                ratings.append((pid, case, label, measure, v))
        elif c:
            labs = raw[col].str.extract(r"([ABC])\s*$")[0]
            for pid, lab in zip(raw["participant"], labs):
                choices.append((pid, c.group(1), c.group(2), lab))
    r = pd.DataFrame(ratings, columns=["participant", "case", "label", "measure", "value"]).dropna()
    c = pd.DataFrame(choices, columns=["participant", "case", "question", "label"]).dropna()
    return r, c


def main() -> None:
    ap = argparse.ArgumentParser(description="Analysis of the user study responses.")
    ap.add_argument("--responses", default=str(OUT / "responses.csv"))
    args = ap.parse_args()

    key = pd.read_csv(OUT / "stimuli_key.csv")
    ratings, choices = parse_responses(Path(args.responses))
    ratings = ratings.merge(key, on=["case", "label"])
    choices = choices.merge(key[["case", "label", "algorithm", "stratum"]], on=["case", "label"])
    n_part = ratings["participant"].nunique()
    print(f"Participants: {n_part}, ratings: {len(ratings)}, choices: {len(choices)}")

    # 1. ratings per algorithm
    per_part = ratings.groupby(["participant", "measure", "algorithm"])["value"].mean().unstack("algorithm")
    desc, tests = [], []
    for meas in MEASURES:
        w = per_part.xs(meas, level="measure").dropna()
        for alg in ALGS:
            desc.append({"measure": meas, "algorithm": alg, "n": len(w),
                         "mean": w[alg].mean(), "sd": w[alg].std(ddof=1), "median": w[alg].median()})
        ps = []
        for a, b in PAIRS:
            d = (w[a] - w[b]).to_numpy()
            p = 1.0 if np.allclose(d, 0) else wilcoxon(w[a], w[b], zero_method="pratt",
                                                        alternative="two-sided").pvalue
            ps.append(p)
            sd = d.std(ddof=1)
            tests.append({"measure": meas, "contrast": f"{a}_vs_{b}", "n": len(d),
                          "mean_diff": d.mean(), "d_z": d.mean() / sd if sd > 0 else np.nan, "p_value": p})
        for row, pa in zip(tests[-len(PAIRS):], holm(ps)):
            row["p_adj_holm"] = pa
    desc, tests = pd.DataFrame(desc), pd.DataFrame(tests)
    desc.to_csv(OUT / "analysis_ratings.csv", index=False)

    # 2. choices
    rng = np.random.default_rng(SEED)
    share = (choices.assign(n=1).pivot_table(index=["participant", "question"], columns="algorithm",
                                             values="n", aggfunc="sum", fill_value=0)
             .reindex(columns=ALGS, fill_value=0))
    share = share.div(share.sum(axis=1), axis=0)
    ch_rows = []
    for q in QUESTIONS:
        s = share.xs(q, level="question")
        boot = s.to_numpy()[rng.integers(0, len(s), size=(B, len(s)))].mean(axis=1)
        for j, alg in enumerate(ALGS):
            lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
            ch_rows.append({"question": q, "algorithm": alg, "n": len(s),
                            "share": s[alg].mean(), "ci_low": lo, "ci_high": hi})
    pd.DataFrame(ch_rows).to_csv(OUT / "analysis_choices.csv", index=False)
    both = share["wpcst"].unstack("question").dropna()
    d = (both["detail"] - both["quick"]).to_numpy()
    p = 1.0 if np.allclose(d, 0) else wilcoxon(both["detail"], both["quick"], zero_method="pratt",
                                               alternative="two-sided").pvalue
    sd = d.std(ddof=1)
    tests = pd.concat([tests, pd.DataFrame([{
        "measure": "wpcst_choice_share", "contrast": "detail_vs_quick", "n": len(d),
        "mean_diff": d.mean(), "d_z": d.mean() / sd if sd > 0 else np.nan, "p_value": p, "p_adj_holm": p}])])
    tests.to_csv(OUT / "analysis_tests.csv", index=False)

    # 3. proxies, over the summaries
    per_sum = (ratings.groupby(["case", "label", "measure"])["value"].mean().unstack("measure")
               .reset_index().merge(key, on=["case", "label"]))
    per_sum.to_csv(OUT / "analysis_per_summary.csv", index=False)
    rho_c, p_c = spearmanr(per_sum["comprehensibility"], per_sum["ease"])
    rho_r, p_r = spearmanr(per_sum["relevance"], per_sum["evidence"])
    rho_s, _ = spearmanr(per_sum["num_edges"], per_sum["relevance"])
    prox = pd.DataFrame([
        {"proxy": "C", "rating": "ease", "n_summaries": len(per_sum), "spearman": rho_c, "p_value": p_c},
        {"proxy": "R", "rating": "evidence", "n_summaries": len(per_sum), "spearman": rho_r, "p_value": p_r},
    ])
    prox.to_csv(OUT / "analysis_proxies.csv", index=False)

    # 4. by stratum
    by_s = ratings.groupby(["stratum", "measure", "algorithm"])["value"].mean().unstack("algorithm")[ALGS]
    ch_s = (choices.groupby(["stratum", "question"])["algorithm"].value_counts(normalize=True)
            .unstack("algorithm").reindex(columns=ALGS).fillna(0.0))
    pd.concat({"mean_rating": by_s, "choice_share": ch_s}).to_csv(OUT / "analysis_by_stratum.csv")

    pd.set_option("display.width", 200)
    for title, df in (("Ratings per algorithm (participant means)", desc), ("Tests", tests),
                      ("Choice shares", pd.DataFrame(ch_rows)), ("Proxies over summaries", prox),
                      ("Mean rating by stratum", by_s), ("Choice share by stratum", ch_s)):
        print(f"\n{title}\n{df.round(3).to_string()}")
    print(f"\nSpearman(|E_S|, R) over the {len(per_sum)} summaries: {rho_s:.3f}")


if __name__ == "__main__":
    main()
