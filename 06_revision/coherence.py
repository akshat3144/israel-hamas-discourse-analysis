# -*- coding: utf-8 -*-
"""Compute c_v coherence for the LDA/NMF topics already fitted in topics.py."""
import json
import re
from pathlib import Path

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
N_DOCS = 5000
SEED = 42


def clean(t):
    if not isinstance(t, str):
        return ""
    t = t.lower()
    t = re.sub(r"http\S+|www\S+|https\S+", "", t)
    t = re.sub(r"\@\w+|\#", "", t)
    t = re.sub(r"[^a-zA-Z\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


if __name__ == "__main__":
    topics = json.load(open(OUT / "topics_clean.json", encoding="utf-8"))
    res = {}
    for plat, f in [("reddit", "reddit_window_clean.parquet"), ("youtube", "youtube_window_clean.parquet")]:
        df = pd.read_parquet(OUT / f)[["text_std"]].sample(N_DOCS, random_state=SEED)
        toks = [c.split() for c in df["text_std"].map(clean) if len(c) > 20]
        d = Dictionary(toks)
        res[plat] = {}
        for model in ("lda", "nmf"):
            cm = CoherenceModel(topics=topics[plat][model], texts=toks,
                                dictionary=d, coherence="c_v")
            res[plat][model] = round(float(cm.get_coherence()), 4)
            print(f"{plat} {model}: c_v = {res[plat][model]}")

    with open(OUT / "coherence.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("saved coherence.json")
