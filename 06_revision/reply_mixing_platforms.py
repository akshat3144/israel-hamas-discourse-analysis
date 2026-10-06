# -*- coding: utf-8 -*-
"""
Comment-level reply mixing on BOTH platforms.

The paper's RQ3 network result is Reddit-only and user-level, and it stated that
YouTube has no comparable reply graph. YouTube does store replies: a reply's
comment ID is `<parentID>.<replyID>`, where the parent is always the top-level
comment (YouTube threads are two levels deep). A reply that answers another
reply is stored under the top-level comment and usually opens with "@handle";
we resolve those to the mentioned author's comment in the same thread when that
is unambiguous.

For each platform, on the analysed corpus (High-confidence P/I/N), each reply
edge links the replying comment's stance to the stance of the comment it answers.
We report:
  * the stance mixing matrix and the share of partisan-to-partisan replies that
    cross the divide
  * observed same-stance rates against random mixing (the target pool)
  * comment-level Newman assortativity
  * a within-thread permutation null: target stances shuffled among the replies
    of the same post / video, which holds each thread's composition fixed - so
    "crossing" cannot come from who happens to be in the thread
  * whether cross-stance replies are more toxic, controlling sentiment (RoBERTa)
  * a user-level graph where users have >= 3 comments, as in the Reddit analysis

Run:  python 06_revision/reply_mixing_platforms.py
Out:  06_revision/outputs/reply_mixing_platforms.json
"""
import glob
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
N_PERM = 200
SEED = 20240507
T = 0.5
MENTION = re.compile(r"^[\s​﻿]*(@[^\s ,:;]+)")


def scores(plat):
    tox = pd.concat([pd.read_parquet(x) for x in glob.glob(str(OUT / "detoxify" / f"{plat}_*.parquet"))])
    rb = pd.concat([pd.read_parquet(x) for x in glob.glob(str(OUT / "roberta" / f"{plat}_*.parquet"))])
    f = {"reddit": "reddit_window_clean", "youtube": "youtube_window_clean"}[plat]
    vd = pd.read_parquet(OUT / f"{f}.parquet", columns=["index", "vader_compound"]).drop_duplicates("index")
    return (tox.drop_duplicates("index")[["index", "dx_toxicity"]]
            .merge(rb.drop_duplicates("index")[["index", "rb_score", "rb_label"]], on="index")
            .merge(vd, on="index"))


def is_real_user(name):
    s = str(name)
    return s not in {"[deleted]", "[removed]", "AutoModerator", "nan"} and not (
        s.endswith("ModTeam") or s.endswith("Bot") or s.endswith("-bot"))


# ------------------------------------------------------------------ edges
def reddit_edges():
    c = pd.read_parquet(OUT / "reddit_window_clean.parquet",
                        columns=["index", "Label", "author_name", "post_id", "parent_id"])
    ids = pd.read_csv(ROOT / "data" / "reddit_labeled.csv", usecols=["index", "comment_id"])
    c = c.merge(ids.drop_duplicates("index"), on="index", how="left")
    c = c.rename(columns={"author_name": "author", "post_id": "thread"})
    c["cid"] = c["comment_id"].astype(str)
    by_id = c.set_index("cid")
    rep = c[c["parent_id"].astype(str).str.startswith("t1_")].copy()
    rep["tid"] = rep["parent_id"].astype(str).str[3:]
    rep = rep[rep["tid"].isin(by_id.index)]
    rep["t_label"] = rep["tid"].map(by_id["Label"])
    rep["t_author"] = rep["tid"].map(by_id["author"])
    rep["resolved_by"] = "parent_id"
    return c, rep


def youtube_edges():
    c = pd.read_parquet(OUT / "youtube_window_clean.parquet",
                        columns=["index", "Label", "author", "video id"])
    c = c.drop_duplicates("index")
    meta = pd.read_csv(ROOT / "data" / "youtube_labeled.csv", usecols=["index", "id", "text"])
    c = c.merge(meta.drop_duplicates("index"), on="index", how="left")
    c = c.rename(columns={"video id": "thread"})
    c["cid"] = c["id"].astype(str)
    c["top"] = c["cid"].str.split(".").str[0]
    by_id = c.drop_duplicates("cid").set_index("cid")
    rep = c[c["cid"].str.contains(".", regex=False)].copy()
    # default target: the top-level comment of the thread
    rep["tid"] = rep["top"]
    rep["resolved_by"] = "top_level"
    # "@handle" replies answer that user's comment in the same thread
    rep["mention"] = rep["text"].astype(str).str.extract(MENTION, expand=False)
    thread_members = c[["top", "author", "cid"]].copy()
    m = rep.loc[rep["mention"].notna(), ["cid", "top", "mention"]].merge(
        thread_members.rename(columns={"cid": "cand", "author": "mention"}),
        on=["top", "mention"], how="left")
    m = m[m["cand"] != m["cid"]]
    n_cand = m.groupby("cid")["cand"].nunique()
    uniq = m[m["cid"].isin(n_cand[n_cand == 1].index)].drop_duplicates("cid")
    rep = rep.merge(uniq[["cid", "cand"]], on="cid", how="left")
    hit = rep["cand"].notna()
    rep.loc[hit, "tid"] = rep.loc[hit, "cand"]
    rep.loc[hit, "resolved_by"] = "mention"
    # a mention we could not resolve is an unknown target, not the top comment
    unresolved = rep["mention"].notna() & ~hit
    rep = rep[~unresolved]
    rep = rep[rep["tid"].isin(by_id.index)]
    rep["t_label"] = rep["tid"].map(by_id["Label"])
    rep["t_author"] = rep["tid"].map(by_id["author"])
    stats_ = {"replies_total": int(c["cid"].str.contains(".", regex=False).sum()),
              "with_mention": int(hit.sum() + unresolved.sum()),
              "mention_resolved": int(hit.sum()),
              "mention_unresolved_dropped": int(unresolved.sum())}
    return c, rep, stats_


# ------------------------------------------------------------------ measures
def newman_r(src, dst):
    m = pd.crosstab(src, dst).reindex(index=STANCES, columns=STANCES, fill_value=0).to_numpy(float)
    e = m / m.sum()
    a, b = e.sum(1), e.sum(0)
    return float((np.trace(e) - (a * b).sum()) / (1 - (a * b).sum()))


def cross_share(src, dst):
    part = np.isin(src, ["P", "I"]) & np.isin(dst, ["P", "I"])
    return float((src[part] != dst[part]).mean())


def analyse(rep, c, plat, rng):
    e = rep[rep["author"].map(is_real_user) & (rep["author"] != rep["t_author"])].copy()
    e = e[e["Label"].isin(STANCES) & e["t_label"].isin(STANCES)]
    src, dst = e["Label"].to_numpy(), e["t_label"].to_numpy()
    mix = pd.crosstab(e["Label"], e["t_label"]).reindex(index=STANCES, columns=STANCES, fill_value=0)
    pool = pd.Series(dst).value_counts(normalize=True).reindex(STANCES).fillna(0)
    per = {s: {"observed": round(float((dst[src == s] == s).mean()), 4),
               "expected": round(float(pool[s]), 4)} for s in STANCES}
    for s in STANCES:
        per[s]["excess"] = round(per[s]["observed"] - per[s]["expected"], 4)
    obs_cross, obs_r = cross_share(src, dst), newman_r(src, dst)

    # within-thread permutation null
    th = e["thread"].astype(str).to_numpy()
    order = np.argsort(th, kind="stable")
    th_s, dst_s, src_s = th[order], dst[order], src[order]
    bounds = np.flatnonzero(np.r_[True, th_s[1:] != th_s[:-1], True])
    null_cross, null_r = [], []
    for _ in range(N_PERM):
        perm = dst_s.copy()
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a > 1:
                perm[a:b] = rng.permutation(perm[a:b])
        null_cross.append(cross_share(src_s, perm))
        null_r.append(newman_r(src_s, perm))
    nc, nr = np.array(null_cross), np.array(null_r)

    res = {
        "n_edges": int(len(e)),
        "n_threads_with_edges": int(e["thread"].nunique()),
        "mixing_counts": mix.to_dict(),
        "mixing_row_pct": (mix.div(mix.sum(1), axis=0) * 100).round(2).to_dict(),
        "per_stance": per,
        "partisan_pairs": int((np.isin(src, ["P", "I"]) & np.isin(dst, ["P", "I"])).sum()),
        "cross_share_partisan": round(obs_cross, 4),
        "newman_r_comment": round(obs_r, 4),
        "within_thread_null": {
            "cross_share_mean": round(float(nc.mean()), 4), "cross_share_sd": round(float(nc.std()), 5),
            "newman_r_mean": round(float(nr.mean()), 4), "newman_r_sd": round(float(nr.std()), 5),
            "z_cross": round(float((obs_cross - nc.mean()) / nc.std()), 1) if nc.std() else None,
            "z_r": round(float((obs_r - nr.mean()) / nr.std()), 1) if nr.std() else None,
            "n_perm": N_PERM},
    }
    if "resolved_by" in e:
        res["edges_by_resolution"] = e["resolved_by"].value_counts().to_dict()

    # toxicity of cross- vs same-stance partisan replies
    sc = scores(plat)
    t = e.merge(sc, on="index", how="left")
    t = t[t["Label"].isin(["P", "I"]) & t["t_label"].isin(["P", "I"])].dropna(subset=["dx_toxicity"])
    t["cross"] = t["Label"] != t["t_label"]
    y = t["dx_toxicity"].to_numpy(float)

    def cross_coef(sent_col):
        X = np.column_stack([np.ones(len(t)), t["cross"].to_numpy(float),
                             t[sent_col].to_numpy(float), (t["Label"] == "I").to_numpy(float)])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        r_ = y - X @ b
        br = np.linalg.inv(X.T @ X)
        se_ = np.sqrt(np.diag(br @ ((X * r_[:, None] ** 2).T @ X) @ br) * len(y) / (len(y) - 4))
        return b, se_

    beta, se = cross_coef("rb_score")
    beta_v, se_v = cross_coef("vader_compound")
    res["toxicity"] = {
        "n": int(len(t)),
        "pct_toxic_cross": round(float((t.loc[t.cross, "dx_toxicity"] >= T).mean() * 100), 2),
        "pct_toxic_same": round(float((t.loc[~t.cross, "dx_toxicity"] >= T).mean() * 100), 2),
        "pct_negative_cross": round(float((t.loc[t.cross, "rb_label"] == "negative").mean() * 100), 2),
        "pct_negative_same": round(float((t.loc[~t.cross, "rb_label"] == "negative").mean() * 100), 2),
        "ols_cross_b_controlling_roberta": round(float(beta[1]), 5),
        "ols_cross_ci95": [round(float(beta[1] - 1.96 * se[1]), 5), round(float(beta[1] + 1.96 * se[1]), 5)],
        "ols_cross_p": float(2 * stats.norm.sf(abs(beta[1] / se[1]))),
        "ols_cross_b_controlling_vader": round(float(beta_v[1]), 5),
        "ols_cross_vader_ci95": [round(float(beta_v[1] - 1.96 * se_v[1]), 5),
                                 round(float(beta_v[1] + 1.96 * se_v[1]), 5)],
    }

    # user-level graph (users with >= 3 comments), as in network_homophily.py
    cu = c[c["author"].map(is_real_user)]
    counts = cu.groupby(["author", "Label"]).size().unstack(fill_value=0).reindex(columns=STANCES, fill_value=0)
    active = counts[counts.sum(1) >= 3]
    dom = active.idxmax(axis=1)
    ue = e[e["author"].isin(dom.index) & e["t_author"].isin(dom.index)]
    if len(ue):
        us, ud = ue["author"].map(dom).to_numpy(), ue["t_author"].map(dom).to_numpy()
        res["user_level"] = {"n_active_users": int(len(active)), "n_edges": int(len(ue)),
                             "newman_r": round(newman_r(us, ud), 4),
                             "cross_share_partisan": round(cross_share(us, ud), 4)}
    return res


def main():
    rng = np.random.default_rng(SEED)
    out = {}
    c, rep = reddit_edges()
    out["reddit"] = analyse(rep, c, "reddit", rng)
    c, rep, ystats = youtube_edges()
    out["youtube"] = analyse(rep, c, "youtube", rng)
    out["youtube"]["reply_resolution"] = ystats
    for p in ("reddit", "youtube"):
        r = out[p]
        print(f"\n=== {p} ===  edges {r['n_edges']:,}  partisan pairs {r['partisan_pairs']:,}")
        print(f"  cross-share {r['cross_share_partisan']}  Newman r {r['newman_r_comment']}")
        print(f"  within-thread null: {r['within_thread_null']}")
        print(f"  per stance: {r['per_stance']}")
        print(f"  toxicity: {r['toxicity']}")
        print(f"  user level: {r.get('user_level')}")
        if "edges_by_resolution" in r:
            print(f"  resolution: {r['edges_by_resolution']}")
    print("\nyoutube reply resolution:", ystats)
    with open(OUT / "reply_mixing_platforms.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
