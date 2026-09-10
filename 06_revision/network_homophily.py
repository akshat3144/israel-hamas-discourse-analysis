# -*- coding: utf-8 -*-
"""
Stage 2b: proper network-based homophily (reviewer R18).

Raw homophily is uninterpretable without a chance baseline: in a corpus that is
45.6% Pro-Palestine, a Pro-Palestine user replying at random already matches
~45.6% of the time. We therefore report:

  1. Observed reply homophily per stance
  2. Expected homophily under random mixing (configuration-style baseline)
  3. Chance-corrected homophily  (observed - expected)
  4. Newman's assortativity coefficient for a categorical attribute on the
     user-user reply network (the standard network-science measure)
  5. The full stance mixing matrix (who replies to whom)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"
STANCES = ["P", "I", "N"]
NAMES = {"P": "Pro-Palestine", "I": "Pro-Israel", "N": "Neutral"}


def is_real_user(name):
    s = str(name)
    if s in {"[deleted]", "[removed]", "AutoModerator"}:
        return False
    return not (s.endswith("ModTeam") or s.endswith("Bot") or s.endswith("-bot"))


print("Loading ...")
r = pd.read_parquet(OUT / "reddit_window.parquet")
cid = pd.read_csv(ROOT / "data" / "reddit_labeled.csv", usecols=["index", "comment_id"])
r = r.merge(cid, on="index", how="left")
ru = r[r["author_name"].map(is_real_user)].copy()

# user stance profiles (>=3 comments)
counts = (ru.groupby(["author_name", "Label"]).size().unstack(fill_value=0)
          .reindex(columns=STANCES, fill_value=0))
totals = counts.sum(axis=1)
active = totals[totals >= 3].index
dom = counts.loc[active].idxmax(axis=1)
print(f"active users: {len(active):,}")

# ---- build user->user reply edges from parent_id
cid2author = dict(zip("t1_" + ru["comment_id"].astype(str), ru["author_name"]))
rep = ru[ru["parent_id"].astype(str).str.startswith("t1_")].copy()
rep["parent_author"] = rep["parent_id"].map(cid2author)
rep = rep.dropna(subset=["parent_author"])
rep = rep[rep["parent_author"] != rep["author_name"]]
rep["src"] = rep["author_name"].map(dom)
rep["dst"] = rep["parent_author"].map(dom)
edges = rep.dropna(subset=["src", "dst"])
print(f"reply edges with both endpoints profiled: {len(edges):,}")

res = {"n_edges": int(len(edges)), "n_active_users": int(len(active))}

# ---- 1. mixing matrix (row = replier stance, col = target stance)
mix = pd.crosstab(edges["src"], edges["dst"]).reindex(index=STANCES, columns=STANCES, fill_value=0)
mix_prop = mix.div(mix.sum(axis=1), axis=0)
print("\n=== Stance mixing matrix (row=replier, col=who they replied to, row %) ===")
print((mix_prop * 100).round(1))
res["mixing_counts"] = mix.to_dict()
res["mixing_row_pct"] = (mix_prop * 100).round(2).to_dict()

# ---- 2. observed vs expected homophily per stance
# Expected = share of that stance among all reply TARGETS (the available pool)
target_share = edges["dst"].value_counts(normalize=True).reindex(STANCES).fillna(0)
print("\n=== Homophily vs chance baseline ===")
print(f"reply-target pool composition: " +
      ", ".join(f"{s}={target_share[s]*100:.1f}%" for s in STANCES))

per_stance = {}
for s in STANCES:
    sub = edges[edges["src"] == s]
    obs = float((sub["dst"] == s).mean()) if len(sub) else np.nan
    exp = float(target_share[s])
    # chance-corrected: how much above random mixing, normalised by headroom
    corrected = (obs - exp) / (1 - exp) if exp < 1 else np.nan
    per_stance[s] = {"n_edges": int(len(sub)), "observed": round(obs, 4),
                     "expected": round(exp, 4), "excess": round(obs - exp, 4),
                     "chance_corrected": round(corrected, 4)}
    print(f"  {NAMES[s]:14s} observed={obs:.3f}  expected={exp:.3f}  "
          f"excess={obs-exp:+.3f}  corrected={corrected:+.3f}")
res["per_stance"] = per_stance

overall_obs = float((edges["src"] == edges["dst"]).mean())
overall_exp = float(sum(
    (edges["src"] == s).mean() * target_share[s] for s in STANCES))
res["overall"] = {"observed": round(overall_obs, 4), "expected": round(overall_exp, 4),
                  "excess": round(overall_obs - overall_exp, 4)}
print(f"  OVERALL        observed={overall_obs:.3f}  expected={overall_exp:.3f}  "
      f"excess={overall_obs-overall_exp:+.3f}")

# ---- 3. Newman assortativity coefficient for categorical attribute
# r = (sum_i e_ii - sum_i a_i*b_i) / (1 - sum_i a_i*b_i)
e = mix.to_numpy(dtype=float)
e = e / e.sum()
a = e.sum(axis=1)   # fraction of edges whose source has stance i
b = e.sum(axis=0)   # fraction of edges whose target has stance i
trace = np.trace(e)
sum_ab = float((a * b).sum())
assort = (trace - sum_ab) / (1 - sum_ab)
res["newman_assortativity"] = round(float(assort), 4)
print(f"\nNewman assortativity coefficient (stance) = {assort:+.4f}")
print("  (0 = random mixing, 1 = perfectly segregated, <0 = disassortative)")

# ---- 4. significance: is the mixing matrix different from independence?
chi2, p, dof, _ = stats.chi2_contingency(mix)
n = mix.to_numpy().sum()
v = float(np.sqrt((chi2 / n) / min(mix.shape[0] - 1, mix.shape[1] - 1)))
res["mixing_chi2"] = {"chi2": round(float(chi2), 1), "p": float(p), "cramers_v": round(v, 4)}
print(f"Mixing-matrix chi-square: chi2={chi2:,.0f}, p={p:.3e}, Cramer's V={v:.4f}")

# ---- 5. cross-cutting exposure: share of replies that cross partisan lines
partisan = edges[edges["src"].isin(["P", "I"]) & edges["dst"].isin(["P", "I"])]
cross = float((partisan["src"] != partisan["dst"]).mean())
res["cross_partisan_reply_rate"] = round(cross, 4)
print(f"\nAmong P<->I reply pairs, {cross*100:.1f}% cross partisan lines.")

with open(OUT / "network_homophily.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
print(f"\nSaved -> {OUT / 'network_homophily.json'}")
