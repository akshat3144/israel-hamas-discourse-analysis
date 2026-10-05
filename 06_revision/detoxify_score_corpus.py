# -*- coding: utf-8 -*-
"""
Score every comment in the primary analysis corpus with Detoxify.

Corpus = reddit_window_clean + youtube_window_clean (the 1.38M comments the paper
analyses). Comments are processed in a fixed random order (seeded permutation)
and written in shards, so the run is resumable and - if it is stopped early - the
completed shards are themselves a uniform random sample of the corpus.

Run (GPU env):  python 06_revision/detoxify_score_corpus.py [--device cuda|cpu]
Out:            06_revision/outputs/detoxify/<platform>_<k>.parquet
                (index + Detoxify scores only; no comment text)

Remote GPU:     python detoxify_score_corpus.py --export corpus_text.parquet   (local)
                python detoxify_score_corpus.py --text corpus_text.parquet --out shards
                The export holds only platform, index and comment text - no
                usernames or other metadata - and is deleted from the remote host
                once the shards are copied back.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import detox_common as dc

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
SHARDS = OUT / "detoxify"
SHARD = 20_000
SEED = 7
CORPUS = [("reddit", "reddit_window_clean", "self_text"),
          ("youtube", "youtube_window_clean", "text")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--fp16", action="store_true",
                    help="half precision; fast on tensor-core GPUs, moves scores <0.001")
    ap.add_argument("--export", help="write the text-only extract here and exit")
    ap.add_argument("--text", help="score this extract instead of the local corpus")
    ap.add_argument("--out", default=str(SHARDS))
    args = ap.parse_args()
    shards = Path(args.out)

    if args.text:
        ext = pd.read_parquet(args.text)
        parts = [(p, ext[ext["platform"] == p][["index", "text"]], "text")
                 for p, _, _ in CORPUS]
    else:
        parts = [(p, pd.read_parquet(OUT / f"{f}.parquet", columns=["index", col]), col)
                 for p, f, col in CORPUS]

    if args.export:
        ext = pd.concat([d.rename(columns={col: "text"}).assign(platform=p)
                         [["platform", "index", "text"]] for p, d, col in parts],
                        ignore_index=True)
        ext["text"] = ext["text"].astype(str)
        ext.to_parquet(args.export, index=False, compression="zstd")
        print(f"exported {len(ext):,} rows -> {args.export} "
              f"({Path(args.export).stat().st_size / 1e6:.0f} MB)")
        return

    shards.mkdir(parents=True, exist_ok=True)
    jobs = []
    for plat, d, col in parts:
        d = d.iloc[np.random.default_rng(SEED).permutation(len(d))].reset_index(drop=True)
        for k in range(0, len(d), SHARD):
            jobs.append((plat, k // SHARD, d.iloc[k:k + SHARD], col))
    todo = [j for j in jobs if not (shards / f"{j[0]}_{j[1]:04d}.parquet").exists()]
    n_left = sum(len(j[2]) for j in todo)
    print(f"{len(jobs)} shards, {len(todo)} to do, {n_left:,} comments", flush=True)
    if not todo:
        return

    m = dc.load(args.device)
    print(f"device: {m.model.device}", flush=True)
    done, t0 = 0, time.time()
    # interleave platforms so a partial run covers both
    todo.sort(key=lambda j: (j[1], j[0]))
    for plat, k, part, col in todo:
        s = dc.score(m, part[col].tolist(), batch=args.batch, fp16=args.fp16)
        res = pd.DataFrame({"index": part["index"].to_numpy()})
        for c in m.class_names:
            res[f"dx_{c}"] = s[c]
        tmp = shards / f"{plat}_{k:04d}.parquet.tmp"
        res.to_parquet(tmp, index=False)
        tmp.replace(shards / f"{plat}_{k:04d}.parquet")
        done += len(part)
        rate = done / (time.time() - t0)
        eta = (n_left - done) / rate / 3600
        print(f"  {plat}_{k:04d}  {done:,}/{n_left:,}  {rate:.0f}/s  "
              f"eta {eta:.1f}h", flush=True)


if __name__ == "__main__":
    main()
