# -*- coding: utf-8 -*-
"""
Stage 5: consensus sentiment (reviewer R11).

R11 suggests that instead of leaning on Cohen's kappa, we keep only the comments
where all three sentiment methods AGREE, and analyse those confident cases.
We therefore:
  * score a stratified sample with VADER, TextBlob and twitter-roberta
  * report pairwise Cohen's kappa (kept, but secondary)
  * build the 3/3 consensus subset and report its coverage
  * recompute the headline platform / stance contrasts on the consensus subset
    to show whether the conclusions survive
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
STANCES = ["P", "I", "N"]
SAMPLE_N = 20000
SEED = 42
res = {}


def rank_biserial(u, n1, n2):
    return float(1 - (2.0 * u) / (n1 * n2))


def cramers_v(ct):
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.to_numpy().sum()
    return float(np.sqrt((chi2 / n) / min(ct.shape[0] - 1, ct.shape[1] - 1)))


print("Loading ...")
cols = ["Label", "text_std", "vader_label", "vader_compound", "textblob_label"]
r = pd.read_parquet(OUT / "reddit_window.parquet")[cols]
y = pd.read_parquet(OUT / "youtube_window.parquet")[cols]


def strat(df, n):
    return df if len(df) <= n else df.groupby("Label", group_keys=False).sample(
        frac=n / len(df), random_state=SEED)


rs, ys = strat(r, SAMPLE_N), strat(y, SAMPLE_N)
print(f"  sample: Reddit {len(rs):,} | YouTube {len(ys):,}")

print("Loading transformer (downloads once) ...")
from transformers import pipeline
clf = pipeline("sentiment-analysis",
               model="cardiffnlp/twitter-roberta-base-sentiment-latest",
               truncation=True, max_length=128)
MAP = {"negative": "negative", "neutral": "neutral", "positive": "positive",
       "label_0": "negative", "label_1": "neutral", "label_2": "positive"}


def roberta(texts, batch=64):
    out = []
    for i in range(0, len(texts), batch):
        chunk = [t[:500] if isinstance(t, str) and t else "" for t in texts[i:i + batch]]
        out.extend(MAP.get(x["label"].lower(), x["label"].lower()) for x in clf(chunk))
        if (i // batch) % 40 == 0:
            print(f"    {i:,}/{len(texts):,}", flush=True)
    return out


for name, df in [("reddit", rs), ("youtube", ys)]:
    print(f"  scoring {name} ...")
    df["roberta_label"] = roberta(df["text_std"].fillna("").astype(str).tolist())

# ---------------------------------------------------------------- kappas
print("\n=== Pairwise Cohen's kappa (secondary) ===")
kap = {}
for name, df in [("reddit", rs), ("youtube", ys)]:
    kap[name] = {
        "vader_vs_roberta": round(float(cohen_kappa_score(df["vader_label"], df["roberta_label"])), 4),
        "textblob_vs_roberta": round(float(cohen_kappa_score(df["textblob_label"], df["roberta_label"])), 4),
        "vader_vs_textblob": round(float(cohen_kappa_score(df["vader_label"], df["textblob_label"])), 4),
    }
    print(f"  {name}: {kap[name]}")
res["kappa"] = kap

# ---------------------------------------------------------------- consensus
print("\n=== 3/3 consensus subset (R11) ===")
cons = {}
for name, df in [("reddit", rs), ("youtube", ys)]:
    agree = (df["vader_label"] == df["textblob_label"]) & (df["vader_label"] == df["roberta_label"])
    sub = df[agree]
    cov = len(sub) / len(df) * 100
    dist = sub["vader_label"].value_counts(normalize=True).mul(100).round(2).to_dict()
    cons[name] = {"n_sample": int(len(df)), "n_consensus": int(len(sub)),
                  "coverage_pct": round(cov, 2), "distribution_pct": dist}
    print(f"  {name}: {len(sub):,}/{len(df):,} agree ({cov:.1f}%)  dist={dist}")
res["consensus"] = cons

# platform contrast on the consensus subset
rc = rs[(rs["vader_label"] == rs["textblob_label"]) & (rs["vader_label"] == rs["roberta_label"])]
yc = ys[(ys["vader_label"] == ys["textblob_label"]) & (ys["vader_label"] == ys["roberta_label"])]
r_neg = (rc["vader_label"] == "negative").mean() * 100
y_neg = (yc["vader_label"] == "negative").mean() * 100
u, p = stats.mannwhitneyu(rc["vader_compound"], yc["vader_compound"], alternative="two-sided")
res["consensus_platform"] = {
    "reddit_pct_negative": round(float(r_neg), 2),
    "youtube_pct_negative": round(float(y_neg), 2),
    "reddit_mean_compound": round(float(rc["vader_compound"].mean()), 4),
    "youtube_mean_compound": round(float(yc["vader_compound"].mean()), 4),
    "mwu_p": float(p),
    "rank_biserial": round(rank_biserial(u, len(rc), len(yc)), 4),
}
print(f"\n  consensus platform contrast: Reddit {r_neg:.1f}% negative vs YouTube {y_neg:.1f}%")
print(f"    mean compound R={rc['vader_compound'].mean():+.4f} Y={yc['vader_compound'].mean():+.4f}"
      f"  r={res['consensus_platform']['rank_biserial']:+.4f}")

# stance association on the consensus subset
res["consensus_stance"] = {}
for name, sub in [("reddit", rc), ("youtube", yc)]:
    ct = pd.crosstab(sub["Label"], sub["vader_label"])
    v = cramers_v(ct)
    means = {s: round(float(sub[sub["Label"] == s]["vader_compound"].mean()), 4) for s in STANCES}
    res["consensus_stance"][name] = {"cramers_v": round(v, 4), "means": means}
    print(f"  {name}: consensus Cramer's V={v:.4f}  by-stance means={means}")

with open(OUT / "consensus_sentiment.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
print(f"\nSaved -> {OUT / 'consensus_sentiment.json'}")
