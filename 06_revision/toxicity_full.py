# -*- coding: utf-8 -*-
"""
Toxicity re-analysis on the Detoxify-scored corpus (replaces the 600-comment
Perspective sample as the primary toxicity evidence).

Recomputes every toxicity result the paper reports - platform means, stance
contrasts, the joint sentiment x toxicity overlap, and the stance-controlled
regression - and adds one analysis the sample could not support: whether Reddit
replies that cross the stance divide are more toxic than same-stance replies.

Sentiment enters with each instrument: twitter-roberta (the primary tone measure
after human validation; roberta_score_corpus.py), VADER and TextBlob.

Works on whatever shards exist, so it can be run on a partial scoring pass (the
shards are a uniform random sample by construction); coverage is reported.

Run:  python 06_revision/toxicity_full.py
Out:  06_revision/outputs/toxicity_full.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"
SHARDS = OUT / "detoxify"
STANCES = ["P", "I", "N"]
ATTRS = ["toxicity", "severe_toxicity", "identity_attack", "insult", "threat", "obscene"]
T = 0.5
N_CORPUS = {"reddit": 1_004_629, "youtube": 378_267}
res = {"instrument": "detoxify-unbiased", "threshold": T}


def rank_biserial(a, b):
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(1 - 2.0 * u / (len(a) * len(b))), float(p)


def ols(y, cols, names):
    """OLS with HC1 robust standard errors; returns coef, 95% CI and p."""
    X = np.column_stack([np.ones(len(y))] + cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    n, k = X.shape
    bread = np.linalg.inv(X.T @ X)
    meat = (X * e[:, None] ** 2).T @ X
    se = np.sqrt(np.diag(bread @ meat @ bread) * n / (n - k))
    p = 2 * stats.norm.sf(np.abs(beta / se))
    r2 = 1 - (e @ e) / ((y - y.mean()) ** 2).sum()
    return {"r2": round(float(r2), 4), "n": int(n),
            "coef": {nm: {"b": round(float(beta[i]), 5),
                          "ci95": [round(float(beta[i] - 1.96 * se[i]), 5),
                                   round(float(beta[i] + 1.96 * se[i]), 5)],
                          "p": float(p[i])}
                     for i, nm in enumerate(["intercept"] + names)}}


# ------------------------------------------------------------------ load
data = {}
for plat, f, extra in [("reddit", "reddit_window_clean", ["parent_id"]),
                       ("youtube", "youtube_window_clean", [])]:
    files = sorted(SHARDS.glob(f"{plat}_*.parquet"))
    if not files:
        raise SystemExit(f"no {plat} shards in {SHARDS}")
    sc = pd.concat([pd.read_parquet(x) for x in files], ignore_index=True)
    # 35 YouTube indexes occur twice in the corpus (same comment text, so the same
    # score); keep one score per index so the merge reproduces the corpus rows
    sc = sc.drop_duplicates("index")
    w = pd.read_parquet(OUT / f"{f}.parquet",
                        columns=["index", "Label", "vader_compound", "vader_label",
                                 "textblob_label"] + extra)
    d = sc.merge(w, on="index", how="inner")
    rb = pd.concat([pd.read_parquet(x) for x in sorted((OUT / "roberta").glob(f"{plat}_*.parquet"))],
                   ignore_index=True).drop_duplicates("index")[["index", "rb_score", "rb_label"]]
    d = d.merge(rb, on="index", how="left")
    assert d["rb_label"].notna().all(), f"{plat}: missing roberta scores"
    d.columns = [c.replace("dx_", "") for c in d.columns]
    data[plat] = d
    res.setdefault("coverage", {})[plat] = {
        "scored": int(len(d)), "corpus": N_CORPUS[plat],
        "pct": round(len(d) / N_CORPUS[plat] * 100, 2)}
    print(f"{plat}: {len(d):,} scored ({len(d)/N_CORPUS[plat]*100:.1f}% of corpus)")

# ------------------------------------------------------------------ platform
print("\n=== platform means ===")
plat_res = {}
for a in ATTRS:
    r, y = data["reddit"][a].to_numpy(), data["youtube"][a].to_numpy()
    rb, p = rank_biserial(r, y)
    plat_res[a] = {"reddit_mean": round(float(r.mean()), 4),
                   "youtube_mean": round(float(y.mean()), 4),
                   "reddit_pct_over_T": round(float((r >= T).mean() * 100), 2),
                   "youtube_pct_over_T": round(float((y >= T).mean() * 100), 2),
                   "rank_biserial": round(rb, 4), "p": p}
    print(f"  {a:16s} R={r.mean():.4f} Y={y.mean():.4f}  "
          f">=0.5: R={plat_res[a]['reddit_pct_over_T']}% Y={plat_res[a]['youtube_pct_over_T']}%  "
          f"r={rb:+.3f}")
res["by_platform"] = plat_res

# ------------------------------------------------------------------ stance
print("\n=== by stance ===")
st_res = {}
for plat, d in data.items():
    st_res[plat] = {}
    for a in ATTRS:
        groups = [d.loc[d["Label"] == s, a].to_numpy() for s in STANCES]
        h, p = stats.kruskal(*groups)
        n = sum(len(g) for g in groups)
        st_res[plat][a] = {
            "means": {s: round(float(g.mean()), 4) for s, g in zip(STANCES, groups)},
            "pct_over_T": {s: round(float((g >= T).mean() * 100), 2)
                           for s, g in zip(STANCES, groups)},
            "eps2": round(float(h / (n - 1)), 4), "p": float(p)}
        # pairwise partisan contrast as an effect size
        rb, pp = rank_biserial(groups[1], groups[0])
        st_res[plat][a]["I_vs_P_rank_biserial"] = round(rb, 4)
    t = st_res[plat]["toxicity"]
    print(f"  {plat}: toxicity means {t['means']}  eps2={t['eps2']}  "
          f"I-vs-P r={t['I_vs_P_rank_biserial']:+.3f}")
res["by_stance"] = st_res

# ------------------------------------------------------------------ joint
print("\n=== joint sentiment x toxicity ===")
joint = {}
for plat, d in data.items():
    joint[plat] = {}
    for inst, col in [("roberta", "rb_label"), ("vader", "vader_label"),
                      ("textblob", "textblob_label")]:
        tox = d["toxicity"] >= T
        neg = d[col] == "negative"
        both = int((tox & neg).sum())
        joint[plat][inst] = {
            "n": int(len(d)), "n_toxic": int(tox.sum()), "n_negative": int(neg.sum()),
            "pct_toxic_that_are_negative": round(both / tox.sum() * 100, 1),
            "pct_negative_that_are_toxic": round(both / neg.sum() * 100, 1),
        }
    rho, _ = stats.spearmanr(d["vader_compound"], d["toxicity"])
    joint[plat]["spearman_vader_toxicity"] = round(float(rho), 4)
    joint[plat]["spearman_roberta_toxicity"] = round(
        float(stats.spearmanr(d["rb_score"], d["toxicity"])[0]), 4)
    v = joint[plat]["vader"]
    print(f"  {plat}: toxic={v['n_toxic']:,}  {v['pct_toxic_that_are_negative']}% of toxic "
          f"are negative; {v['pct_negative_that_are_toxic']}% of negative are toxic; "
          f"rho={rho:+.3f}")

    # stance after controlling sentiment
    y = d["toxicity"].to_numpy(float)
    cols = [d["vader_compound"].to_numpy(float),
            (d["Label"] == "I").to_numpy(float), (d["Label"] == "N").to_numpy(float)]
    joint[plat]["ols"] = ols(y, cols, ["vader_compound", "I_vs_P", "N_vs_P"])
    raw = ols(y, cols[1:], ["I_vs_P", "N_vs_P"])
    joint[plat]["ols_no_sentiment"] = raw
    joint[plat]["ols_roberta"] = ols(y, [d["rb_score"].to_numpy(float)] + cols[1:],
                                     ["rb_score", "I_vs_P", "N_vs_P"])
    c, c0 = joint[plat]["ols"]["coef"], raw["coef"]
    print(f"    I-vs-P: raw b={c0['I_vs_P']['b']:+.4f} -> controlled b={c['I_vs_P']['b']:+.4f} "
          f"{c['I_vs_P']['ci95']}")
    print(f"    N-vs-P: raw b={c0['N_vs_P']['b']:+.4f} -> controlled b={c['N_vs_P']['b']:+.4f} "
          f"{c['N_vs_P']['ci95']}")
res["joint"] = joint

# ------------------------------------------------------------------ replies (new)
print("\n=== Reddit: are cross-stance replies more toxic? ===")
r = data["reddit"]
cid = pd.read_csv(ROOT / "data" / "reddit_labeled.csv", usecols=["index", "comment_id"])
lab = pd.read_parquet(OUT / "reddit_window.parquet", columns=["index", "Label"])
lab = lab.merge(cid, on="index", how="inner")
parent_stance = dict(zip("t1_" + lab["comment_id"].astype(str), lab["Label"]))
rep = r[r["parent_id"].astype(str).str.startswith("t1_")].copy()
rep["parent_label"] = rep["parent_id"].map(parent_stance)
rep = rep[rep["Label"].isin(["P", "I"]) & rep["parent_label"].isin(["P", "I"])]
rep["cross"] = rep["Label"] != rep["parent_label"]
g = rep.groupby("cross")["toxicity"]
same, cross = rep.loc[~rep["cross"], "toxicity"], rep.loc[rep["cross"], "toxicity"]
rb, p = rank_biserial(cross.to_numpy(), same.to_numpy())
reply = {
    "n_partisan_replies": int(len(rep)), "n_cross": int(len(cross)), "n_same": int(len(same)),
    "mean_tox_cross": round(float(cross.mean()), 4),
    "mean_tox_same": round(float(same.mean()), 4),
    "pct_toxic_cross": round(float((cross >= T).mean() * 100), 2),
    "pct_toxic_same": round(float((same >= T).mean() * 100), 2),
    "rank_biserial_cross_vs_same": round(rb, 4), "p": p,
}
y = rep["toxicity"].to_numpy(float)
reply["ols_controls_sentiment_and_stance"] = ols(
    y, [rep["cross"].to_numpy(float), rep["vader_compound"].to_numpy(float),
        (rep["Label"] == "I").to_numpy(float)],
    ["cross_stance", "vader_compound", "replier_I"])
by_dir = {}
for a_, b_ in [("P", "P"), ("P", "I"), ("I", "I"), ("I", "P")]:
    s = rep[(rep["Label"] == a_) & (rep["parent_label"] == b_)]["toxicity"]
    by_dir[f"{a_}->{b_}"] = {"n": int(len(s)), "mean": round(float(s.mean()), 4),
                             "pct_toxic": round(float((s >= T).mean() * 100), 2)}
reply["ols_controls_roberta_and_stance"] = ols(
    y, [rep["cross"].to_numpy(float), rep["rb_score"].to_numpy(float),
        (rep["Label"] == "I").to_numpy(float)],
    ["cross_stance", "rb_score", "replier_I"])
reply["pct_negative_roberta"] = {
    "cross": round(float((rep.loc[rep["cross"], "rb_label"] == "negative").mean() * 100), 2),
    "same": round(float((rep.loc[~rep["cross"], "rb_label"] == "negative").mean() * 100), 2)}
reply["by_direction"] = by_dir
res["reddit_replies"] = reply
print(f"  partisan replies: {len(rep):,} ({len(cross):,} cross, {len(same):,} same)")
print(f"  toxic: cross {reply['pct_toxic_cross']}% vs same {reply['pct_toxic_same']}%  "
      f"(mean {reply['mean_tox_cross']} vs {reply['mean_tox_same']}, r={rb:+.3f})")
cc = reply["ols_controls_sentiment_and_stance"]["coef"]["cross_stance"]
print(f"  controlled cross-stance b={cc['b']:+.4f} {cc['ci95']}")
for k, v in by_dir.items():
    print(f"    {k}: n={v['n']:,} mean={v['mean']} toxic={v['pct_toxic']}%")

with open(OUT / "toxicity_full.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
print(f"\nSaved -> {OUT / 'toxicity_full.json'}")
