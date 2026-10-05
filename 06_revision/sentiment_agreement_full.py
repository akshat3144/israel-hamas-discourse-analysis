# -*- coding: utf-8 -*-
"""
Agreement among the three sentiment instruments on the full corpus.

consensus_sentiment.py measured this on a 20k stratified sample because
twitter-roberta had only been run on that sample. With full-corpus RoBERTa scores
(roberta_score_corpus.py) it can be computed on every comment, together with the
unanimous-subset analysis the paper uses as a robustness check.

Run:  python 06_revision/sentiment_agreement_full.py
Out:  06_revision/outputs/sentiment_agreement_full.json
"""
import glob
import json
from pathlib import Path

import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

from label_noise_sensitivity import cramers_v

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
STANCES = ["P", "I", "N"]
res = {}

for plat, f in (("reddit", "reddit_window_clean"), ("youtube", "youtube_window_clean")):
    d = pd.read_parquet(OUT / f"{f}.parquet",
                        columns=["index", "Label", "vader_label", "textblob_label", "vader_compound"])
    rb = pd.concat([pd.read_parquet(x) for x in glob.glob(str(OUT / "roberta" / f"{plat}_*.parquet"))])
    d = d.merge(rb.drop_duplicates("index")[["index", "rb_label", "rb_score"]], on="index")
    d = d[d["Label"].isin(STANCES)]
    k = {"vader_vs_roberta": cohen_kappa_score(d.vader_label, d.rb_label),
         "vader_vs_textblob": cohen_kappa_score(d.vader_label, d.textblob_label),
         "textblob_vs_roberta": cohen_kappa_score(d.textblob_label, d.rb_label)}
    un = d[(d.vader_label == d.textblob_label) & (d.vader_label == d.rb_label)]
    res[plat] = {"n": int(len(d)), "kappa": {a: round(float(b), 4) for a, b in k.items()},
                 "unanimous_pct": round(len(un) / len(d) * 100, 1),
                 "unanimous_pct_negative": round(float((un.vader_label == "negative").mean() * 100), 1),
                 "unanimous_cramers_v": round(cramers_v(un.Label, un.vader_label), 4),
                 "unanimous_mean_rb_score": round(float(un.rb_score.mean()), 4)}
    res[plat]["_un"] = un
    print(plat, {a: b for a, b in res[plat].items() if a != "_un"})

ur, uy = res["reddit"].pop("_un"), res["youtube"].pop("_un")
u, p = stats.mannwhitneyu(uy.rb_score, ur.rb_score, alternative="two-sided")
res["unanimous_rank_biserial_youtube_vs_reddit"] = round(float(1 - 2 * u / (len(ur) * len(uy))), 4)
print("unanimous r:", res["unanimous_rank_biserial_youtube_vs_reddit"])
with open(OUT / "sentiment_agreement_full.json", "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2)
