# -*- coding: utf-8 -*-
"""
Stage 2: core statistics on the matched window.

Addresses:
  R8  - bootstrapped 95% CIs on stance percentages
  R16 - controversiality demoted to a metric sanity-check; engagement re-measured
        with a NON-vote-derived signal (reply counts from parent_id)
  R17 - formal definitions of consistency and homophily (implemented as documented)
  R18 - homophily on the TRUE reply network (user->user edges from parent_id),
        alongside the thread co-membership version
  R20 - everything computed on the matched Oct-2023..May-2024 window
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"
STANCES = ["P", "I", "N"]
NAMES = {"P": "Pro-Palestine", "I": "Pro-Israel", "N": "Neutral"}
RESULTS = {}


# ---------------------------------------------------------------- helpers
def cramers_v(ct):
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.to_numpy().sum()
    r, k = ct.shape
    return float(np.sqrt((chi2 / n) / min(r - 1, k - 1)))


def rank_biserial(u, n1, n2):
    return float(1 - (2.0 * u) / (n1 * n2))


def epsilon_squared(h, n):
    return float(h / (n - 1))


def bootstrap_prop_ci(counts, B=4000, seed=42):
    """Bootstrap 95% CI for multinomial proportions (R8)."""
    rng = np.random.default_rng(seed)
    counts = np.asarray(counts, dtype=float)
    n = int(counts.sum())
    p = counts / n
    draws = rng.multinomial(n, p, size=B) / n
    lo = np.percentile(draws, 2.5, axis=0) * 100
    hi = np.percentile(draws, 97.5, axis=0) * 100
    return p * 100, lo, hi


_VOWELS = re.compile(r"[aeiouy]+")


def _syll(w):
    w = w.lower()
    n = len(_VOWELS.findall(w))
    if w.endswith("e") and n > 1:
        n -= 1
    return max(n, 1)


def flesch(text):
    if not isinstance(text, str):
        return np.nan
    words = re.findall(r"[A-Za-z]+", text)
    if len(words) < 3:
        return np.nan
    sents = max(len(re.findall(r"[.!?]+", text)), 1)
    return 206.835 - 1.015 * (len(words) / sents) - 84.6 * (sum(_syll(w) for w in words) / len(words))


def is_real_user(name):
    s = str(name)
    if s in {"[deleted]", "[removed]", "AutoModerator"}:
        return False
    return not (s.endswith("ModTeam") or s.endswith("Bot") or s.endswith("-bot"))


# ---------------------------------------------------------------- load
print("Loading windowed data ...")
r = pd.read_parquet(OUT / "reddit_window.parquet")
y = pd.read_parquet(OUT / "youtube_window.parquet")
# comment_id is needed for the reply network; join it back on `index`
cid = pd.read_csv(ROOT / "data" / "reddit_labeled.csv", usecols=["index", "comment_id"])
r = r.merge(cid, on="index", how="left")
print(f"  Reddit {len(r):,} | YouTube {len(y):,}")

RESULTS["n_reddit"] = int(len(r))
RESULTS["n_youtube"] = int(len(y))

# ---------------------------------------------------------------- R8 stance CIs
print("\n=== R8: stance distribution with bootstrap 95% CI ===")
stance_ci = {}
for plat, df in [("reddit", r), ("youtube", y)]:
    counts = df["Label"].value_counts().reindex(STANCES).to_numpy()
    pct, lo, hi = bootstrap_prop_ci(counts)
    stance_ci[plat] = {s: {"n": int(counts[i]), "pct": round(float(pct[i]), 2),
                           "lo": round(float(lo[i]), 2), "hi": round(float(hi[i]), 2)}
                       for i, s in enumerate(STANCES)}
    print(f"  {plat}:")
    for i, s in enumerate(STANCES):
        print(f"    {NAMES[s]:14s} {counts[i]:>9,}  {pct[i]:6.2f}%  [{lo[i]:.2f}, {hi[i]:.2f}]")
RESULTS["stance_ci"] = stance_ci

# ---------------------------------------------------------------- RQ1 sentiment
print("\n=== RQ1: sentiment on matched window ===")
rc, yc = r["vader_compound"].dropna(), y["vader_compound"].dropna()
u, p = stats.mannwhitneyu(rc, yc, alternative="two-sided")
sent = {
    "reddit_mean_compound": round(float(rc.mean()), 4),
    "youtube_mean_compound": round(float(yc.mean()), 4),
    "mwu_U": float(u), "mwu_p": float(p),
    "rank_biserial": round(rank_biserial(u, len(rc), len(yc)), 4),
}
print(f"  Reddit mean compound  = {sent['reddit_mean_compound']}")
print(f"  YouTube mean compound = {sent['youtube_mean_compound']}")
print(f"  Mann-Whitney U={u:,.0f}, p={p:.3e}, rank-biserial r={sent['rank_biserial']:+.4f}")

by_stance = {}
for plat, df in [("reddit", r), ("youtube", y)]:
    g = df.groupby("Label")["vader_compound"].mean().reindex(STANCES)
    by_stance[plat] = {s: round(float(g[s]), 4) for s in STANCES}
    ct = pd.crosstab(df["Label"], df["vader_label"])
    v = cramers_v(ct)
    groups = [df[df["Label"] == s]["vader_compound"].dropna() for s in STANCES]
    h, pk = stats.kruskal(*groups)
    n = sum(len(gg) for gg in groups)
    by_stance[plat + "_cramers_v"] = round(v, 4)
    by_stance[plat + "_kruskal_H"] = round(float(h), 2)
    by_stance[plat + "_kruskal_eps2"] = round(epsilon_squared(h, n), 5)
    print(f"  {plat}: by-stance means {by_stance[plat]}, Cramer's V={v:.4f}, "
          f"Kruskal eps^2={by_stance[plat + '_kruskal_eps2']:.5f}")
sent["by_stance"] = by_stance
RESULTS["sentiment"] = sent

# ---------------------------------------------------------------- length
print("\n=== Comment length (matched window) ===")
length = {}
for plat, df in [("reddit", r), ("youtube", y)]:
    length[plat] = {"mean": round(float(df["word_count"].mean()), 2),
                    "median": float(df["word_count"].median())}
    print(f"  {plat}: mean={length[plat]['mean']} median={length[plat]['median']}")
RESULTS["length"] = length

# ---------------------------------------------------------------- readability
print("\n=== Readability (Flesch, stratified sample) ===")
READ_N = 150000
read = {}
for plat, df in [("reddit", r), ("youtube", y)]:
    sub = df if len(df) <= READ_N else df.groupby("Label", group_keys=False).sample(
        frac=READ_N / len(df), random_state=42)
    sc = sub["text_std"].map(flesch)
    sub = sub.assign(readability=sc)
    sub = sub[sub["readability"].between(-100, 121)]
    groups = [sub[sub["Label"] == s]["readability"].dropna() for s in STANCES]
    h, pk = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    read[plat] = {"mean": round(float(sub["readability"].mean()), 2),
                  "n": int(len(sub)),
                  "kruskal_H": round(float(h), 2),
                  "eps2": round(epsilon_squared(h, n), 5)}
    print(f"  {plat}: mean={read[plat]['mean']} (n={read[plat]['n']:,}) eps^2={read[plat]['eps2']}")
RESULTS["readability"] = read

# ---------------------------------------------------------------- R16 engagement
print("\n=== R16: engagement (controversiality is definitional; use reply counts) ===")
amp = r[["score", "controversiality"]].dropna()
amp["controversiality"] = amp["controversiality"].astype(int)
mean_by_contro = amp.groupby("controversiality")["score"].mean()
print(f"  [sanity-check only] mean score: non-contro={mean_by_contro.get(0, float('nan')):.2f}, "
      f"contro={mean_by_contro.get(1, float('nan')):.2f}")

# NON-vote-derived engagement: number of direct replies each comment received
r["cid_full"] = "t1_" + r["comment_id"].astype(str)
reply_counts = r["parent_id"].value_counts()
r["n_replies"] = r["cid_full"].map(reply_counts).fillna(0)
eng = {
    "controversiality_sanity": {"non_controversial_mean_score": round(float(mean_by_contro.get(0, np.nan)), 3),
                                "controversial_mean_score": round(float(mean_by_contro.get(1, np.nan)), 3)},
    "replies_mean": round(float(r["n_replies"].mean()), 4),
    "replies_by_controversiality": {int(k): round(float(v), 4) for k, v in
                                    r.groupby(r["controversiality"].astype(int))["n_replies"].mean().items()},
    "replies_by_stance": {s: round(float(r[r["Label"] == s]["n_replies"].mean()), 4) for s in STANCES},
}
print(f"  mean replies received = {eng['replies_mean']}")
print(f"  replies by controversiality = {eng['replies_by_controversiality']}")
print(f"  replies by stance = {eng['replies_by_stance']}")

# Do controversial comments attract MORE discussion even though score ~0?
g0 = r[r["controversiality"] == 0]["n_replies"]
g1 = r[r["controversiality"] == 1]["n_replies"]
u2, p2 = stats.mannwhitneyu(g1, g0, alternative="two-sided")
eng["replies_mwu_p"] = float(p2)
eng["replies_rank_biserial"] = round(rank_biserial(u2, len(g1), len(g0)), 4)
print(f"  replies contro vs non: MWU p={p2:.3e}, rank-biserial r={eng['replies_rank_biserial']:+.4f}")

# OLS score ~ vader + stance (kept, but reported as weak)
d = r[["score", "vader_compound", "Label"]].dropna()
X = np.column_stack([np.ones(len(d)), d["vader_compound"].to_numpy(float),
                     (d["Label"] == "I").to_numpy(float), (d["Label"] == "N").to_numpy(float)])
yv = d["score"].to_numpy(float)
beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
resid = yv - X @ beta
r2 = 1 - (resid @ resid) / ((yv - yv.mean()) ** 2).sum()
eng["ols_r2"] = round(float(r2), 5)
eng["ols_coef"] = {"intercept": round(float(beta[0]), 4), "vader_compound": round(float(beta[1]), 4),
                   "stance_I": round(float(beta[2]), 4), "stance_N": round(float(beta[3]), 4)}
print(f"  OLS R^2={eng['ols_r2']}  coefs={eng['ols_coef']}")
RESULTS["engagement"] = eng

# ---------------------------------------------------------------- R17/R18 polarization
print("\n=== R17/R18: consistency + homophily (thread AND reply network) ===")
ru = r[r["author_name"].map(is_real_user)].copy()
print(f"  excluded {len(r) - len(ru):,} bot/deleted comments ({(1-len(ru)/len(r))*100:.2f}%)")

counts = (ru.groupby(["author_name", "Label"]).size().unstack(fill_value=0)
          .reindex(columns=STANCES, fill_value=0))
totals = counts.sum(axis=1)
active = totals[totals >= 3].index
profiles = pd.DataFrame({
    "total_comments": totals,
    "dominant_stance": counts.idxmax(axis=1),
    "consistency": counts.max(axis=1) / totals,
}).loc[active]
print(f"  active users (>=3 comments): {len(profiles):,}; mean consistency = {profiles['consistency'].mean():.4f}")

# (a) thread co-membership homophily (as in original paper)
thread_stance = ru.groupby("post_id")["Label"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "N")
pairs = ru[ru["author_name"].isin(active)].drop_duplicates(["author_name", "post_id"])[["author_name", "post_id"]]
pairs["thread_stance"] = pairs["post_id"].map(thread_stance)
pairs["user_stance"] = pairs["author_name"].map(profiles["dominant_stance"])
pairs["match"] = pairs["thread_stance"] == pairs["user_stance"]
homo_thread = pairs.groupby("author_name")["match"].mean().rename("homophily_thread").to_frame()
homo_thread["dominant_stance"] = profiles["dominant_stance"]

# (b) TRUE reply network homophily (R18)
cid2author = dict(zip("t1_" + ru["comment_id"].astype(str), ru["author_name"]))
rep = ru[ru["parent_id"].astype(str).str.startswith("t1_")].copy()
rep["parent_author"] = rep["parent_id"].map(cid2author)
rep = rep.dropna(subset=["parent_author"])
rep = rep[rep["parent_author"] != rep["author_name"]]          # drop self-replies
rep["src_stance"] = rep["author_name"].map(profiles["dominant_stance"])
rep["dst_stance"] = rep["parent_author"].map(profiles["dominant_stance"])
rep_valid = rep.dropna(subset=["src_stance", "dst_stance"])
rep_valid = rep_valid.assign(match=rep_valid["src_stance"] == rep_valid["dst_stance"])
homo_reply = rep_valid.groupby("author_name")["match"].mean().rename("homophily_reply").to_frame()
homo_reply["dominant_stance"] = profiles["dominant_stance"]

pol = {
    "n_active_users": int(len(profiles)),
    "mean_consistency": round(float(profiles["consistency"].mean()), 4),
    "dominant_counts": {s: int((profiles["dominant_stance"] == s).sum()) for s in STANCES},
    "thread": {
        "mean": round(float(homo_thread["homophily_thread"].mean()), 4),
        "pct_above_0.8": round(float((homo_thread["homophily_thread"] > 0.8).mean() * 100), 2),
        "by_stance": {s: round(float(homo_thread[homo_thread["dominant_stance"] == s]["homophily_thread"].mean()), 4)
                      for s in STANCES},
    },
    "reply_network": {
        "n_edges": int(len(rep_valid)),
        "n_users": int(homo_reply.shape[0]),
        "mean": round(float(homo_reply["homophily_reply"].mean()), 4),
        "pct_above_0.8": round(float((homo_reply["homophily_reply"] > 0.8).mean() * 100), 2),
        "by_stance": {s: round(float(homo_reply[homo_reply["dominant_stance"] == s]["homophily_reply"].mean()), 4)
                      for s in STANCES},
    },
}
for key in ["thread", "reply_network"]:
    grp = [homo_thread[homo_thread["dominant_stance"] == s]["homophily_thread"] if key == "thread"
           else homo_reply[homo_reply["dominant_stance"] == s]["homophily_reply"] for s in STANCES]
    grp = [g.dropna() for g in grp if len(g.dropna()) > 0]
    h, pk = stats.kruskal(*grp)
    n = sum(len(g) for g in grp)
    pol[key]["kruskal_H"] = round(float(h), 2)
    pol[key]["eps2"] = round(epsilon_squared(h, n), 5)

print(f"  thread homophily : mean={pol['thread']['mean']}, >0.8={pol['thread']['pct_above_0.8']}%, by stance={pol['thread']['by_stance']}")
print(f"  reply  homophily : mean={pol['reply_network']['mean']}, >0.8={pol['reply_network']['pct_above_0.8']}%, by stance={pol['reply_network']['by_stance']}")
print(f"  reply network edges = {pol['reply_network']['n_edges']:,}")
RESULTS["polarization"] = pol

profiles.to_csv(OUT / "user_profiles.csv")
homo_thread.join(homo_reply["homophily_reply"], how="outer").to_csv(OUT / "homophily.csv")

# ---------------------------------------------------------------- save
with open(OUT / "core_stats.json", "w", encoding="utf-8") as f:
    json.dump(RESULTS, f, indent=2)
print(f"\nSaved -> {OUT / 'core_stats.json'}")
