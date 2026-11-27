from __future__ import annotations
import glob
import json

import numpy as np
import pandas as pd

from config import DATA_ROOT, RESULTS_ROOT, SEED

B = 10_000                      # bootstrap resamples and permutations
Z_ALPHA, Z_POWER = 1.959964, 0.841621   # alpha = 0.05 two-sided, power = 0.80
ALGOS = ["st", "pcst", "wpcst_degree", "naive_union", "pappas2017", "mst", "faces", "supernode"]
RECS = ["pgpr", "cafe", "plm", "plmr"]
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load() -> pd.DataFrame:
    files = sorted(glob.glob(str(RESULTS_ROOT / "per_anchor" / "*.parquet")))
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    d["alg"] = np.where(d.algorithm == "wpcst", "wpcst_" + d.centrality.fillna(""), d.algorithm)
    meta = d.metadata.apply(json.loads)
    d["gender"] = meta.apply(lambda m: m.get("gender"))
    d["quartile"] = meta.apply(lambda m: m.get("popularity_quartile"))
    # Age bands of the ML1M users (data/ml1m/users.txt: uid, gender, age).
    d["age"] = None
    users = DATA_ROOT / "ml1m" / "users.txt"
    if users.exists():
        u = pd.read_csv(users, sep="\t")
        young = {"0-18", "18-24", "25-34"}
        amap = {f"u{r.uid}": ("under35" if r.age in young else "35plus") for r in u.itertuples()}
        m = (d.dataset == "ml1m") & (d.scenario == "user_centric")
        d.loc[m, "age"] = d.loc[m, "anchor_id"].map(amap)
    return d[d.alg.isin(ALGOS)]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _boot_idx(n):
    return rng.integers(0, n, (B, n))


def _perm_diff(z, n1):
    """Differences of group means under B random relabelings (NaN-aware)."""
    pz = rng.permuted(np.tile(z, (B, 1)), axis=1)
    with np.errstate(invalid="ignore"):
        return np.nanmean(pz[:, :n1], axis=1) - np.nanmean(pz[:, n1:], axis=1)


def _p_value(perm, obs):
    return (np.sum(np.abs(perm) >= abs(obs) - 1e-15) + 1) / (B + 1)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (len(p) - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


def two_group(x, y):
    """Signed difference of means with CI, permutation p, effect size, MDE."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    n1, n2 = len(x), len(y)
    delta = x.mean() - y.mean()
    bd = x[_boot_idx(n1)].mean(1) - y[_boot_idx(n2)].mean(1)
    lo, hi = np.percentile(bd, [2.5, 97.5])
    p = _p_value(_perm_diff(np.concatenate([x, y]), n1), delta)
    s1, s2 = x.std(ddof=1), y.std(ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    se = np.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    return dict(n1=n1, n2=n2, delta=delta, gap=abs(delta), ci_lo=lo, ci_hi=hi, p=p,
                cohen_d=delta / sp if sp > 0 else np.nan, mde=(Z_ALPHA + Z_POWER) * se,
                ci_excludes_0=bool(lo > 0 or hi < 0))


def pig_cell(g1, g2):
    """Summary gap, recommender gap, amplification ratio, and change of the gap."""
    xs, xr = g1.pop_frac.to_numpy(float), g1.pop_frac_topk.to_numpy(float)
    ys, yr = g2.pop_frac.to_numpy(float), g2.pop_frac_topk.to_numpy(float)
    n1, n2 = len(xs), len(ys)
    i1, i2 = _boot_idx(n1), _boot_idx(n2)
    with np.errstate(invalid="ignore", divide="ignore"):
        dS = np.nanmean(xs) - np.nanmean(ys)
        dR = np.nanmean(xr) - np.nanmean(yr)
        bS = np.nanmean(xs[i1], 1) - np.nanmean(ys[i2], 1)
        bR = np.nanmean(xr[i1], 1) - np.nanmean(yr[i2], 1)
        ratio_b = np.abs(bS) / np.abs(bR)
        dx, dy = xs - xr, ys - yr                     # per-anchor change
        did = np.nanmean(dx) - np.nanmean(dy)
        bD = np.nanmean(dx[i1], 1) - np.nanmean(dy[i2], 1)
    out = dict(n1=n1, n2=n2,
               rec_delta=dR, rec_lo=np.percentile(bR, 2.5), rec_hi=np.percentile(bR, 97.5),
               rec_p=_p_value(_perm_diff(np.concatenate([xr, yr]), n1), dR),
               sum_delta=dS, sum_lo=np.percentile(bS, 2.5), sum_hi=np.percentile(bS, 97.5),
               sum_p=_p_value(_perm_diff(np.concatenate([xs, ys]), n1), dS),
               ratio=abs(dS) / abs(dR) if dR != 0 else np.nan,
               ratio_lo=np.nanpercentile(ratio_b, 2.5), ratio_hi=np.nanpercentile(ratio_b, 97.5),
               change=did, change_lo=np.nanpercentile(bD, 2.5), change_hi=np.nanpercentile(bD, 97.5),
               change_p=_p_value(_perm_diff(np.concatenate([dx, dy]), n1), did))
    out["change_ci_excludes_0"] = bool(out["change_lo"] > 0 or out["change_hi"] < 0)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SPLITS = [  # dataset, scenario, split name, column, group 1, group 2
    ("ml1m", "user_centric", "gender", "gender", "M", "F"),
    ("ml1m", "user_centric", "age", "age", "under35", "35plus"),
    ("ml1m", "item_centric", "popularity", "quartile", "q4_popular", "q1_unpopular"),
    ("lfm1m", "user_centric", "gender", "gender", "M", "F"),
]


def main() -> None:
    d = load()
    cg_rows, pig_rows = [], []
    for ds, sc, split, col, a, b in SPLITS:
        sub = d[(d.dataset == ds) & (d.scenario == sc)]
        for rec in [r for r in RECS if r in set(sub.baseline)]:
            for alg in ALGOS:
                s = sub[(sub.baseline == rec) & (sub.alg == alg)].sort_values("anchor_id")
                g1, g2 = s[s[col] == a], s[s[col] == b]
                if len(g1) < 2 or len(g2) < 2:
                    continue
                key = dict(dataset=ds, scenario=sc, split=split, recommender=rec, algorithm=alg)
                cg_rows.append({**key, "mean_C": s.comprehensibility.mean(),
                                **two_group(g1.comprehensibility, g2.comprehensibility)})
                if sc == "user_centric":
                    pig_rows.append({**key, **pig_cell(g1, g2)})

    cg, pig = pd.DataFrame(cg_rows), pd.DataFrame(pig_rows)
    table = ["dataset", "scenario", "split"]
    cg["p_holm"] = cg.groupby(table).p.transform(holm)
    pig["change_p_holm"] = pig.groupby(table).change_p.transform(holm)
    cg.to_csv(RESULTS_ROOT / "fairness_stats_cg.csv", index=False)
    pig.to_csv(RESULTS_ROOT / "fairness_stats_pig.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    print("\n================ COMPREHENSIBILITY GAP: summary per table")
    print(cg.groupby(table).agg(cells=("p", "size"), n1=("n1", "max"), n2=("n2", "max"),
                                ci_excludes_0=("ci_excludes_0", "sum"),
                                p_below_05=("p", lambda s: int((s < 0.05).sum())),
                                holm_below_05=("p_holm", lambda s: int((s < 0.05).sum())),
                                max_abs_d=("cohen_d", lambda s: s.abs().max())).round(2).to_string())
    for key, s in cg.groupby(table):
        print(f"\n--- CG {key}")
        print(s.set_index(["recommender", "algorithm"])[
            ["mean_C", "delta", "ci_lo", "ci_hi", "p", "p_holm", "cohen_d", "mde"]].round(4).to_string())

    print("\n================ POPULAR-ITEM GAP: recommender gap (signed, group 1 minus group 2)")
    print(pig[pig.algorithm == "naive_union"].set_index(table + ["recommender"])[
        ["rec_delta", "rec_lo", "rec_hi", "rec_p"]].round(4).to_string())
    for key, s in pig.groupby(table):
        print(f"\n--- PIG {key}")
        print(s.set_index(["recommender", "algorithm"])[
            ["sum_delta", "sum_lo", "sum_hi", "sum_p", "ratio", "ratio_lo", "ratio_hi",
             "change", "change_lo", "change_hi", "change_p", "change_p_holm"]].round(4).to_string())


if __name__ == "__main__":
    main()
