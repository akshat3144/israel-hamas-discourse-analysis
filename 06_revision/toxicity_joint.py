# -*- coding: utf-8 -*-
"""
Stage 3: toxicity, and the JOINT sentiment x toxicity analysis (reviewer R2).

R2 asks whether negativity and toxicity are the same phenomenon or whether
toxicity carries signal beyond negative sentiment. We therefore:
  * recover the stance labels missing from the exported Perspective files
  * join Perspective scores to VADER sentiment
  * cross-tabulate sentiment class x toxicity
  * test whether stance predicts toxicity AFTER controlling for sentiment
  * report effect sizes throughout (R5: 'toxicity', never vague 'harm')
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"
TOX = ROOT / "05_toxicity_analysis" / "outputs"
STANCES = ["P", "I", "N"]
NAMES = {"P": "Pro-Palestine", "I": "Pro-Israel", "N": "Neutral"}
ATTRS = ["TOXICITY", "SEVERE_TOXICITY", "IDENTITY_ATTACK", "INSULT", "THREAT", "PROFANITY"]
KEY = ["TOXICITY", "IDENTITY_ATTACK", "INSULT", "THREAT"]
res = {}


def rank_biserial(u, n1, n2):
    return float(1 - (2.0 * u) / (n1 * n2))


def epsilon_squared(h, n):
    return float(h / (n - 1))


# ---- load + recover labels (the exported files lost the Label column)
r_tox = pd.read_csv(TOX / "reddit_perspective.csv")
y_tox = pd.read_csv(TOX / "youtube_perspective.csv")
rl = pd.read_csv(ROOT / "data" / "reddit_labeled.csv", usecols=["index", "Label"])
yl = pd.read_csv(ROOT / "data" / "youtube_labeled.csv", usecols=["index", "Label"])
r_tox = r_tox.merge(rl, on="index", how="left")
y_tox = y_tox.merge(yl, on="index", how="left")

# ---- attach sentiment from the windowed working files
rw = pd.read_parquet(OUT / "reddit_window.parquet")[["index", "vader_compound", "vader_label"]]
yw = pd.read_parquet(OUT / "youtube_window.parquet")[["index", "vader_compound", "vader_label"]]
r_tox = r_tox.merge(rw, on="index", how="left")
y_tox = y_tox.merge(yw, on="index", how="left")

print(f"Reddit  toxicity rows: {len(r_tox)}  | in-window w/ sentiment: {r_tox['vader_compound'].notna().sum()}")
print(f"YouTube toxicity rows: {len(y_tox)}  | in-window w/ sentiment: {y_tox['vader_compound'].notna().sum()}")
res["n_reddit"] = int(len(r_tox))
res["n_youtube"] = int(len(y_tox))
res["n_reddit_in_window"] = int(r_tox["vader_compound"].notna().sum())
res["n_youtube_in_window"] = int(y_tox["vader_compound"].notna().sum())

# ---------------------------------------------------------------- platform means
print("\n=== Toxicity by platform (Perspective scores) ===")
plat = {}
for a in ATTRS:
    rm, ym = r_tox[a].dropna(), y_tox[a].dropna()
    u, p = stats.mannwhitneyu(rm, ym, alternative="two-sided")
    plat[a] = {"reddit_mean": round(float(rm.mean()), 4),
               "youtube_mean": round(float(ym.mean()), 4),
               "p": float(p),
               "rank_biserial": round(rank_biserial(u, len(rm), len(ym)), 4)}
    print(f"  {a:16s} R={plat[a]['reddit_mean']:.4f} Y={plat[a]['youtube_mean']:.4f} "
          f"p={p:.2e} r={plat[a]['rank_biserial']:+.3f}")
res["by_platform"] = plat

# ---------------------------------------------------------------- by stance
print("\n=== Toxicity by stance ===")
by_stance = {}
for name, df in [("reddit", r_tox), ("youtube", y_tox)]:
    by_stance[name] = {}
    for a in KEY:
        means = {s: round(float(df[df["Label"] == s][a].mean()), 4) for s in STANCES}
        groups = [df[df["Label"] == s][a].dropna() for s in STANCES]
        groups = [g for g in groups if len(g) > 0]
        h, p = stats.kruskal(*groups)
        n = sum(len(g) for g in groups)
        by_stance[name][a] = {"means": means, "H": round(float(h), 2),
                              "p": float(p), "eps2": round(epsilon_squared(h, n), 4)}
        print(f"  {name:8s} {a:16s} {means}  eps^2={by_stance[name][a]['eps2']:.3f} p={p:.3g}")
res["by_stance"] = by_stance

# ---------------------------------------------------------------- R2 JOINT
print("\n=== R2: JOINT sentiment x toxicity ===")
TOX_T = 0.5   # Perspective's conventional 'likely toxic' threshold
joint = {"threshold": TOX_T}
for name, df in [("reddit", r_tox), ("youtube", y_tox)]:
    d = df.dropna(subset=["TOXICITY", "vader_compound", "vader_label"]).copy()
    d["is_toxic"] = d["TOXICITY"] >= TOX_T
    d["is_negative"] = d["vader_label"] == "negative"

    # correlation between continuous sentiment and toxicity
    rho, prho = stats.spearmanr(d["vader_compound"], d["TOXICITY"])
    ct = pd.crosstab(d["is_negative"], d["is_toxic"])
    both = int(((d["is_negative"]) & (d["is_toxic"])).sum())
    neg = int(d["is_negative"].sum())
    tox = int(d["is_toxic"].sum())
    # how much of toxicity is NOT captured by negativity
    tox_not_neg = int(((~d["is_negative"]) & (d["is_toxic"])).sum())

    joint[name] = {
        "n": int(len(d)),
        "spearman_rho": round(float(rho), 4), "spearman_p": float(prho),
        "n_negative": neg, "n_toxic": tox, "n_negative_and_toxic": both,
        "n_toxic_but_not_negative": tox_not_neg,
        "pct_of_toxic_that_are_negative": round(both / tox * 100, 1) if tox else None,
        "pct_of_negative_that_are_toxic": round(both / neg * 100, 1) if neg else None,
    }
    print(f"  {name}: n={len(d)}  Spearman(compound,toxicity) rho={rho:+.3f} (p={prho:.2g})")
    print(f"    negative={neg}, toxic={tox}, both={both}, "
          f"toxic-but-not-negative={tox_not_neg}")
    if tox:
        print(f"    -> {both/tox*100:.1f}% of toxic comments are also negative-sentiment")
    if neg:
        print(f"    -> only {both/neg*100:.1f}% of negative comments are toxic")

# ---- does stance still predict toxicity AFTER controlling for sentiment?
print("\n=== Does stance predict toxicity controlling for sentiment? (OLS) ===")
for name, df in [("reddit", r_tox), ("youtube", y_tox)]:
    d = df.dropna(subset=["TOXICITY", "vader_compound", "Label"])
    X = np.column_stack([np.ones(len(d)),
                         d["vader_compound"].to_numpy(float),
                         (d["Label"] == "I").to_numpy(float),
                         (d["Label"] == "N").to_numpy(float)])
    yv = d["TOXICITY"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    dof = len(d) - X.shape[1]
    sigma2 = (resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    pvals = 2 * stats.t.sf(np.abs(t), dof)
    r2 = 1 - (resid @ resid) / ((yv - yv.mean()) ** 2).sum()
    names = ["intercept", "vader_compound", "stance_I(vs P)", "stance_N(vs P)"]
    joint[name + "_ols"] = {"r2": round(float(r2), 4),
                            "coef": {names[i]: round(float(beta[i]), 4) for i in range(4)},
                            "p": {names[i]: float(pvals[i]) for i in range(4)}}
    print(f"  {name}: R^2={r2:.4f}")
    for i in range(1, 4):
        print(f"    {names[i]:18s} b={beta[i]:+.4f}  p={pvals[i]:.3g}")
res["joint"] = joint

with open(OUT / "toxicity_joint.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
print(f"\nSaved -> {OUT / 'toxicity_joint.json'}")
