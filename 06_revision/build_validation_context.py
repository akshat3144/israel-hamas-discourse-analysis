# -*- coding: utf-8 -*-
"""
Recover reading context for the 270 validation items.

The first sentiment annotation round showed annotators only the subreddit and post
title (Reddit) or a bare video ID (YouTube), which proved too little to read many
comments correctly. This adds, for every item where it exists:
  * YouTube: the video title
  * both platforms: the text of the comment being replied to

Run:  python 06_revision/build_validation_context.py
Out:  06_revision/outputs/validation_context.parquet   (gitignored; holds comment text)
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "00_data_collection_and_labeling" / "outputs"
VAL = HERE / "validation"
OUT = HERE / "outputs"
MAXLEN = 600


def stream(path, keep):
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                o = json.loads(line)
            except Exception:
                continue
            if keep(o):
                yield o


def main():
    key = pd.read_csv(VAL / "validation_key.csv", encoding="utf-8-sig")
    blind = pd.read_csv(VAL / "validation_blind.csv", encoding="utf-8-sig").fillna("")
    key = key.merge(blind[["item_id", "context_video_id"]], on="item_id")
    ctx = {i: {"video_title": "", "parent_text": "", "is_reply": False}
           for i in key.item_id}

    # ---------------------------------------------------------------- Reddit
    rk = key[key.platform == "reddit"]
    want = set(rk.row_index)
    rows = {o["index"]: o for o in stream(RAW / "reddit_labeled_full.jsonl",
                                          lambda o: o.get("index") in want)}
    parent_of = {}
    for item, idx in zip(rk.item_id, rk.row_index):
        o = rows.get(idx)
        if o and str(o.get("parent_id", "")).startswith("t1_"):
            parent_of[item] = str(o["parent_id"])[3:]
            ctx[item]["is_reply"] = True
    need = set(parent_of.values())
    found = {}
    for chunk in pd.read_csv(RAW / "reddit_comments.csv", usecols=["id", "body"],
                             dtype=str, chunksize=500_000):
        hit = chunk[chunk["id"].isin(need)]
        found.update(zip(hit["id"], hit["body"]))
    for item, pid in parent_of.items():
        ctx[item]["parent_text"] = str(found.get(pid, ""))[:MAXLEN]
    print(f"reddit: {len(rk)} items, {len(parent_of)} replies, "
          f"{sum(1 for p in parent_of.values() if p in found)} parents recovered")

    # ---------------------------------------------------------------- YouTube
    yk = key[key.platform == "youtube"]
    meta = pd.read_csv(RAW / "youtube_video_metadata.csv", usecols=["video_id", "title"])
    titles = dict(zip(meta.video_id, meta.title))
    want = set(yk.row_index)
    rows = {o["index"]: o for o in stream(RAW / "youtube_labeled_full.jsonl",
                                          lambda o: o.get("index") in want)}
    parent_of = {}
    for item, idx, vid in zip(yk.item_id, yk.row_index, yk.context_video_id):
        ctx[item]["video_title"] = str(titles.get(vid, ""))
        o = rows.get(idx)
        cid = str(o.get("id", "")) if o else ""
        if "." in cid:
            parent_of[item] = cid.split(".")[0]
            ctx[item]["is_reply"] = True
    need = set(parent_of.values())
    found = {o["id"]: o.get("text", "") for o in
             stream(RAW / "youtube_labeled_full.jsonl", lambda o: o.get("id") in need)}
    for item, pid in parent_of.items():
        ctx[item]["parent_text"] = str(found.get(pid, ""))[:MAXLEN]
    print(f"youtube: {len(yk)} items, {sum(1 for i in yk.item_id if ctx[i]['video_title'])} "
          f"with titles, {len(parent_of)} replies, "
          f"{sum(1 for p in parent_of.values() if p in found)} parents recovered")

    df = pd.DataFrame.from_dict(ctx, orient="index").rename_axis("item_id").reset_index()
    df.to_parquet(OUT / "validation_context.parquet", index=False)
    print(f"wrote {OUT / 'validation_context.parquet'}")


if __name__ == "__main__":
    main()
