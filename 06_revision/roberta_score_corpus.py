# -*- coding: utf-8 -*-
"""
Score every comment in the primary corpus with twitter-roberta sentiment.

Human validation (sentiment_validation.py) found twitter-roberta the closest of the
three instruments to human tone judgements, so it becomes the primary tone measure
and needs full-corpus coverage. Settings match consensus_sentiment.py exactly:
first 500 characters, 128-token limit. Writes class probabilities, the argmax
label, and a continuous score p(positive) - p(negative) in [-1, 1].

Same sharding / resume / random-order design as detoxify_score_corpus.py.

Local:   python 06_revision/roberta_score_corpus.py [--fp16]
Remote:  python roberta_score_corpus.py --text corpus_text.parquet --out shards --fp16
Out:     06_revision/outputs/roberta/<platform>_<k>.parquet (index + scores, no text)
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
SHARDS = OUT / "roberta"
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_CHARS, MAX_LEN = 500, 128          # as in consensus_sentiment.py
SHARD, SEED = 20_000, 7
CORPUS = [("reddit", "reddit_window_clean", "self_text"),
          ("youtube", "youtube_window_clean", "text")]


def load(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device).eval()
    labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    assert sorted(labels) == ["negative", "neutral", "positive"], labels
    return tok, model, labels


@torch.no_grad()
def score(tok, model, labels, texts, batch=128, fp16=False):
    texts = [t[:MAX_CHARS] if isinstance(t, str) and t else "" for t in texts]
    order = np.argsort([len(t) for t in texts], kind="stable")
    probs = np.zeros((len(texts), len(labels)), dtype=np.float32)
    amp = fp16 and model.device.type == "cuda"
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        enc = tok([texts[i] for i in idx], return_tensors="pt", truncation=True,
                  max_length=MAX_LEN, padding=True).to(model.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(**enc).logits
        probs[idx] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    df = pd.DataFrame(probs, columns=[f"rb_p_{l}" for l in labels])
    df["rb_label"] = np.array(labels)[probs.argmax(1)]
    df["rb_score"] = df["rb_p_positive"] - df["rb_p_negative"]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="score this text-only extract instead of the local corpus")
    ap.add_argument("--out", default=str(SHARDS))
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()
    shards = Path(args.out)
    shards.mkdir(parents=True, exist_ok=True)

    if args.text:
        ext = pd.read_parquet(args.text)
        parts = [(p, ext[ext["platform"] == p][["index", "text"]], "text") for p, _, _ in CORPUS]
    else:
        parts = [(p, pd.read_parquet(OUT / f"{f}.parquet", columns=["index", c]), c)
                 for p, f, c in CORPUS]

    jobs = []
    for plat, d, col in parts:
        d = d.iloc[np.random.default_rng(SEED).permutation(len(d))].reset_index(drop=True)
        for k in range(0, len(d), SHARD):
            jobs.append((plat, k // SHARD, d.iloc[k:k + SHARD], col))
    todo = [j for j in jobs if not (shards / f"{j[0]}_{j[1]:04d}.parquet").exists()]
    todo.sort(key=lambda j: (j[1], j[0]))
    n_left = sum(len(j[2]) for j in todo)
    print(f"{len(jobs)} shards, {len(todo)} to do, {n_left:,} comments", flush=True)
    if not todo:
        return

    tok, model, labels = load()
    print(f"device: {model.device}", flush=True)
    done, t0 = 0, time.time()
    for plat, k, part, col in todo:
        res = score(tok, model, labels, part[col].tolist(), args.batch, args.fp16)
        res.insert(0, "index", part["index"].to_numpy())
        tmp = shards / f"{plat}_{k:04d}.parquet.tmp"
        res.to_parquet(tmp, index=False)
        tmp.replace(shards / f"{plat}_{k:04d}.parquet")
        done += len(part)
        rate = done / (time.time() - t0)
        print(f"  {plat}_{k:04d}  {done:,}/{n_left:,}  {rate:.0f}/s  "
              f"eta {(n_left - done) / rate / 3600:.1f}h", flush=True)


if __name__ == "__main__":
    main()
