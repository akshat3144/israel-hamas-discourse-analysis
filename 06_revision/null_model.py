# -*- coding: utf-8 -*-
"""
Stage 10: degree-preserving null model for the reply-network assortativity.

Motivation: De Francisci Morales, Monti and Starnini (Sci. Rep. 2021) established
cross-cutting political interaction on Reddit using a *degree-preserving rewiring*
null plus a logit regression. Our marginal expected-value baseline is weaker: it
does not control for the fact that highly active repliers and highly popular reply
targets are unevenly distributed across stances. If Pro-Palestine users are simply
more numerous and more replied-to, some apparent cross-cutting is mechanical.

This script therefore rebuilds the null properly. Permuting the target column of
the edge list preserves BOTH the out-degree of every replier and the in-degree of
every target exactly (each node appears the same number of times), i.e. it is the
directed configuration model conditioned on the observed degree sequences.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"
STANCES = ["P", "I", "N"]
NAMES = {"P": "Pro-Palestine", "I": "Pro-Israel", "N": "Neutral"}
B = 500
SEED = 42


def is_real_user(name):
    s = str(name)
    if s in {"[deleted]", "[removed]", "AutoModerator"}:
        return False
    return not (s.endswith("ModTeam") or s.endswith("Bot") or s.endswith("-bot"))


def assortativity(src_codes, dst_codes, k=3):
    """Newman's categorical assortativity coefficient."""
    e = np.zeros((k, k), dtype=float)
    np.add.at(e, (src_codes, dst_codes), 1.0)
    e /= e.sum()
    a = e.sum(axis=1)
    b = e.sum(axis=0)
    tr = np.trace(e)
    ab = float((a * b).sum())
    return (tr - ab) / (1 - ab)


print("Loading reply edges ...")
r = pd.read_parquet(OUT / "reddit_window.parquet")
cid = pd.read_csv(ROOT / "data" / "reddit_labeled.csv", usecols=["index", "comment_id"])
r = r.merge(cid, on="index", how="left")
ru = r[r["author_name"].map(is_real_user)].copy()

counts = (ru.groupby(["author_name", "Label"]).size().unstack(fill_value=0)
          .reindex(columns=STANCES, fill_value=0))
totals = counts.sum(axis=1)
active = totals[totals >= 3].index
dom = counts.loc[active].idxmax(axis=1)

cid2author = dict(zip("t1_" + ru["comment_id"].astype(str), ru["author_name"]))
rep = ru[ru["parent_id"].astype(str).str.startswith("t1_")].copy()
rep["parent_author"] = rep["parent_id"].map(cid2author)
rep = rep.dropna(subset=["parent_author"])
rep = rep[rep["parent_author"] != rep["author_name"]]
rep["src"] = rep["author_name"].map(dom)
rep["dst"] = rep["parent_author"].map(dom)
edges = rep.dropna(subset=["src", "dst"])
print(f"  {len(edges):,} reply edges between profiled users")

code = {s: i for i, s in enumerate(STANCES)}
src = edges["src"].map(code).to_numpy()
dst = edges["dst"].map(code).to_numpy()

obs = assortativity(src, dst)
print(f"\nObserved Newman assortativity r = {obs:+.4f}")

# --- degree-preserving null: permute targets (keeps in- and out-degree exact)
print(f"Running degree-preserving null model ({B} permutations) ...")
rng = np.random.default_rng(SEED)
null = np.empty(B)
for i in range(B):
    null[i] = assortativity(src, rng.permutation(dst))
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{B}")

mu, sd = float(null.mean()), float(null.std(ddof=1))
z = (obs - mu) / sd if sd > 0 else float("nan")
p_emp = float((np.abs(null - mu) >= abs(obs - mu)).mean())
print(f"\nNull mean r = {mu:+.5f} (sd {sd:.5f})")
print(f"z = {z:+.1f}   empirical two-sided p {'< ' + str(1/B) if p_emp == 0 else '= ' + str(p_emp)}")

# --- per-stance cross-cutting rate vs the SAME null
print("\nPer-stance same-stance reply rate, observed vs degree-preserving null:")
per = {}
for s in STANCES:
    m = src == code[s]
    if m.sum() == 0:
        continue
    o = float((dst[m] == code[s]).mean())
    nulls = np.array([float((rng.permutation(dst)[m] == code[s]).mean()) for _ in range(200)])
    per[s] = {"observed": round(o, 4),
              "null_mean": round(float(nulls.mean()), 4),
              "excess": round(o - float(nulls.mean()), 4),
              "z": round(float((o - nulls.mean()) / nulls.std(ddof=1)), 1)}
    print(f"  {NAMES[s]:14s} obs={o:.3f}  null={nulls.mean():.3f}  "
          f"excess={o - nulls.mean():+.3f}  z={per[s]['z']:+.1f}")

res = {"n_edges": int(len(edges)), "observed_assortativity": round(obs, 4),
       "null_mean": round(mu, 5), "null_sd": round(sd, 5), "z": round(z, 1),
       "empirical_p": p_emp, "n_permutations": B, "per_stance": per}
with open(OUT / "null_model.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
print(f"\nSaved -> {OUT / 'null_model.json'}")
