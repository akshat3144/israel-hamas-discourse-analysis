# -*- coding: utf-8 -*-
"""
Re-run the RQ3 headline numbers on the relevance-filtered corpus.

RQ3 is the paper's strongest claim (thread co-membership says echo chamber, the
reply graph says the opposite). If it moves once off-topic comments are removed,
we need to know before submission, not after.

Recomputes reply homophily, the chance baseline, and Newman's assortativity on
the user-user reply graph, for the corpus before and after the relevance filter.

Run:  python 06_revision/rq3_on_relevant.py   (after relevance_filter.py)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"
STANCES = ["P", "I", "N"]


def is_real_user(name):
    s = str(name)
    if s in {"[deleted]", "[removed]", "AutoModerator"}:
        return False
    return not (s.endswith("ModTeam") or s.endswith("Bot") or s.endswith("-bot"))


def assortativity(src, dst, k=3):
    e = np.zeros((k, k), dtype=float)
    np.add.at(e, (src, dst), 1.0)
    e /= e.sum()
    a, b = e.sum(axis=1), e.sum(axis=0)
    tr, ab = float(np.trace(e)), float((a * b).sum())
    return (tr - ab) / (1 - ab)


def build(df, cid):
    """User-user reply edges with the stance of each endpoint."""
    d = df.merge(cid, on="index", how="left")
    d = d[d["author_name"].map(is_real_user)]
    d = d[d["Label"].isin(STANCES)]

    # dominant stance per user
    dom = (d.groupby(["author_name", "Label"]).size()
             .rename("n").reset_index()
             .sort_values("n", ascending=False)
             .drop_duplicates("author_name")
             .set_index("author_name")["Label"])

    # parent comment id -> its author
    by_cid = d.dropna(subset=["comment_id"]).set_index("comment_id")["author_name"]
    p = d["parent_id"].astype(str)
    is_comment_reply = p.str.startswith("t1_")
    tgt = p.where(is_comment_reply).str.slice(3).map(by_cid)

    edges = pd.DataFrame({"src": d["author_name"].values, "dst": tgt.values})
    edges = edges.dropna()
    edges = edges[edges["src"] != edges["dst"]]
    edges["s_st"] = edges["src"].map(dom)
    edges["d_st"] = edges["dst"].map(dom)
    return edges.dropna()


def summarise(edges, tag):
    n = len(edges)
    same = float((edges["s_st"] == edges["d_st"]).mean())
    # chance baseline: distribution of stances among reply TARGETS
    tgt_mix = edges["d_st"].value_counts(normalize=True)
    exp = float(sum(
        (edges["s_st"] == s).mean() * tgt_mix.get(s, 0.0) for s in STANCES))
    code = {s: i for i, s in enumerate(STANCES)}
    r = assortativity(edges["s_st"].map(code).to_numpy(),
                      edges["d_st"].map(code).to_numpy())
    part = edges[(edges.s_st != "N") & (edges.d_st != "N")]
    cross = float((part["s_st"] != part["d_st"]).mean())
    print(f"  {tag:<10} edges={n:,}  homophily={same:.3f}  "
          f"chance={exp:.3f}  Newman r={r:+.4f}  cross-partisan={100*cross:.1f}%")
    return {"n_edges": n, "reply_homophily": round(same, 4),
            "expected": round(exp, 4), "newman_r": round(r, 4),
            "cross_partisan_pct": round(100 * cross, 2)}


def main():
    cid = pd.read_csv(ROOT / "data" / "reddit_labeled.csv",
                      usecols=["index", "comment_id"])
    res = {}
    print("RQ3 reply-network statistics\n")
    for tag, fname in (("before", "reddit_window_clean.parquet"),
                       ("after", "reddit_window_relevant.parquet")):
        path = OUT / fname
        if not path.exists():
            print(f"  (missing {fname}; run relevance_filter.py first)")
            continue
        df = pd.read_parquet(path, columns=["index", "Label", "author_name",
                                            "parent_id"])
        res[tag] = summarise(build(df, cid), tag)

    if "before" in res and "after" in res:
        d = res["after"]["newman_r"] - res["before"]["newman_r"]
        print(f"\n  Newman r shift after removing off-topic comments: {d:+.4f}")
        res["newman_shift"] = round(d, 4)

    (OUT / "rq3_relevance_check.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'rq3_relevance_check.json'}")


if __name__ == "__main__":
    main()
