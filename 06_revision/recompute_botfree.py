# -*- coding: utf-8 -*-
"""
Stage 8: exclude automated/deleted accounts CORPUS-WIDE, not just at user level.

Motivation: AutoModerator, *-ModTeam and the [deleted] placeholder account for
5.93% of the windowed Reddit corpus, and the contamination is strongly asymmetric
(14.2% of Neutral and 6.5% of Pro-Palestine, but only 0.3% of Pro-Israel), because
subreddit rule text is itself stance-inflected. Leaving it in inflates two stance
classes, produces a spurious "community governance" topic, and biases the Neutral
baseline. This is automated text, not human discourse, so we drop it everywhere.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
STANCES = ["P", "I", "N"]
NAMES = {"P": "Pro-Palestine", "I": "Pro-Israel", "N": "Neutral"}
res = {}


def is_bot(n):
    s = str(n)
    return (s in {"[deleted]", "[removed]", "AutoModerator"}
            or s.endswith("ModTeam") or s.endswith("Bot") or s.endswith("-bot"))


def bootstrap_prop_ci(counts, B=4000, seed=42):
    rng = np.random.default_rng(seed)
    counts = np.asarray(counts, float)
    n = int(counts.sum())
    draws = rng.multinomial(n, counts / n, size=B) / n
    return counts / n * 100, np.percentile(draws, 2.5, axis=0) * 100, np.percentile(draws, 97.5, axis=0) * 100


def rank_biserial(u, n1, n2):
    return float(1 - (2.0 * u) / (n1 * n2))


def cramers_v(ct):
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.to_numpy().sum()
    return float(np.sqrt((chi2 / n) / min(ct.shape[0] - 1, ct.shape[1] - 1)))


def epsilon_squared(h, n):
    return float(h / (n - 1))


print("Loading windowed data ...")
r = pd.read_parquet(OUT / "reddit_window.parquet")
y = pd.read_parquet(OUT / "youtube_window.parquet")

r_bot = r["author_name"].map(is_bot)
print(f"Reddit  : {len(r):,} -> dropping {int(r_bot.sum()):,} bot/deleted "
      f"({r_bot.mean()*100:.2f}%)")
r = r[~r_bot].copy()
# YouTube has no equivalent automated-account problem (checked: no dominant bot author)
print(f"YouTube : {len(y):,} (no automated-account exclusion applies)")

r.to_parquet(OUT / "reddit_window_clean.parquet", index=False)
y.to_parquet(OUT / "youtube_window_clean.parquet", index=False)

res["n_reddit"] = int(len(r))
res["n_youtube"] = int(len(y))
res["n_total"] = int(len(r) + len(y))
print(f"\nFINAL ANALYTIC CORPUS: Reddit {len(r):,} + YouTube {len(y):,} = {len(r)+len(y):,}")

# ---- stance distribution with CIs
print("\n=== Stance distribution (bot-free, matched window) ===")
res["stance"] = {}
for plat, df in [("reddit", r), ("youtube", y)]:
    c = df["Label"].value_counts().reindex(STANCES).to_numpy()
    pct, lo, hi = bootstrap_prop_ci(c)
    res["stance"][plat] = {s: {"n": int(c[i]), "pct": round(float(pct[i]), 2),
                               "lo": round(float(lo[i]), 2), "hi": round(float(hi[i]), 2)}
                           for i, s in enumerate(STANCES)}
    print(f"  {plat}:")
    for i, s in enumerate(STANCES):
        print(f"    {NAMES[s]:14s} {c[i]:>9,}  {pct[i]:6.2f}%  [{lo[i]:.2f}, {hi[i]:.2f}]")

# ---- sentiment
print("\n=== Sentiment (bot-free) ===")
rc, yc = r["vader_compound"].dropna(), y["vader_compound"].dropna()
u, p = stats.mannwhitneyu(rc, yc, alternative="two-sided")
res["sentiment"] = {"reddit_mean": round(float(rc.mean()), 4),
                    "youtube_mean": round(float(yc.mean()), 4),
                    "rank_biserial": round(rank_biserial(u, len(rc), len(yc)), 4),
                    "p": float(p)}
print(f"  Reddit {res['sentiment']['reddit_mean']}  YouTube {res['sentiment']['youtube_mean']}  "
      f"r={res['sentiment']['rank_biserial']:+.4f}")
res["sentiment"]["by_stance"] = {}
for plat, df in [("reddit", r), ("youtube", y)]:
    means = {s: round(float(df[df["Label"] == s]["vader_compound"].mean()), 4) for s in STANCES}
    ct = pd.crosstab(df["Label"], df["vader_label"])
    groups = [df[df["Label"] == s]["vader_compound"].dropna() for s in STANCES]
    h, _ = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    res["sentiment"]["by_stance"][plat] = {
        "means": means, "cramers_v": round(cramers_v(ct), 4),
        "eps2": round(epsilon_squared(h, n), 5)}
    print(f"  {plat}: {means}  V={res['sentiment']['by_stance'][plat]['cramers_v']:.4f}")

# ---- length
res["length"] = {p_: {"mean": round(float(d["word_count"].mean()), 2),
                      "median": float(d["word_count"].median())}
                 for p_, d in [("reddit", r), ("youtube", y)]}
print(f"\n=== Length === reddit {res['length']['reddit']}  youtube {res['length']['youtube']}")

with open(OUT / "botfree_stats.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
print(f"\nSaved -> {OUT / 'botfree_stats.json'}")
