# -*- coding: utf-8 -*-
"""
Stage 7: topic models on the matched window (reviewer R3, R4).

R4: the previous topic *labels* were interpretive and in one case implied a causal
    /moral attribution the model never produced. Here we emit the raw top terms and
    attach only DESCRIPTIVE labels naming the term families, never a claim about
    who did what to whom.
R3: this is presented as topic/vocabulary structure only - no narrative claims.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
STANCES = ["P", "I", "N"]
NAMES = {"P": "Pro-Palestine", "I": "Pro-Israel", "N": "Neutral"}
SAMPLE = 200000          # topic structure is stable well below full corpus
K = 5
TOPN = 10
SEED = 42
res = {}

DOMAIN_STOP = ["israel", "israeli", "palestine", "palestinian", "hamas", "gaza", "war",
               "conflict", "just", "like", "people", "know", "think", "going", "said",
               "really", "also", "would", "could", "one", "two", "even", "make", "get",
               "want", "see", "say", "tell", "much", "many", "thing", "way", "time"]
ALL_STOP = sorted(set(CountVectorizer(stop_words="english").get_stop_words()) | set(DOMAIN_STOP))


def clean(t):
    if not isinstance(t, str):
        return ""
    t = t.lower()
    t = re.sub(r"http\S+|www\S+|https\S+", "", t)
    t = re.sub(r"\@\w+|\#", "", t)
    t = re.sub(r"[^a-zA-Z\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


print("Loading ...")
data = {}
for name, f in [("reddit", "reddit_window_clean.parquet"), ("youtube", "youtube_window_clean.parquet")]:
    df = pd.read_parquet(OUT / f)[["Label", "text_std"]]
    if len(df) > SAMPLE:
        df = df.sample(SAMPLE, random_state=SEED)
    df["clean"] = df["text_std"].map(clean)
    df = df[df["clean"].str.len() > 20]
    data[name] = df
    print(f"  {name}: {len(df):,} docs for topic modelling")

for name, df in data.items():
    print(f"\n{'='*60}\n{name.upper()}\n{'='*60}")
    res[name] = {"n_docs": int(len(df))}

    # ---- top words overall and by stance
    def top_words(texts, n=20):
        v = CountVectorizer(max_features=n, stop_words=ALL_STOP)
        X = v.fit_transform(texts)
        freq = dict(zip(v.get_feature_names_out(), np.asarray(X.sum(axis=0)).ravel()))
        return sorted(freq.items(), key=lambda kv: -kv[1])[:n]

    tw = top_words(df["clean"], 20)
    res[name]["top_words"] = [(w, int(c)) for w, c in tw]
    print("Top 20 words:", ", ".join(w for w, _ in tw))

    res[name]["top_words_by_stance"] = {}
    for s in STANCES:
        sub = df[df["Label"] == s]["clean"]
        if len(sub) > 50:
            t = top_words(sub, 10)
            res[name]["top_words_by_stance"][s] = [w for w, _ in t]
            print(f"  {NAMES[s]:14s}: {', '.join(w for w, _ in t)}")

    # ---- LDA
    cv = CountVectorizer(max_df=0.95, min_df=5, max_features=1000, stop_words="english")
    tf = cv.fit_transform(df["clean"])
    lda = LatentDirichletAllocation(n_components=K, random_state=SEED, max_iter=20,
                                    learning_method="online", batch_size=4096).fit(tf)
    cvn = cv.get_feature_names_out()
    lda_topics = [[cvn[i] for i in t.argsort()[-TOPN:][::-1]] for t in lda.components_]
    res[name]["lda"] = lda_topics
    print("\nLDA topics (descriptive term lists only):")
    for i, t in enumerate(lda_topics, 1):
        print(f"  T{i}: {', '.join(t)}")

    # ---- NMF
    tv = TfidfVectorizer(max_df=0.95, min_df=5, max_features=1000, stop_words="english")
    tfidf = tv.fit_transform(df["clean"])
    nmf = NMF(n_components=K, random_state=SEED, max_iter=300, init="nndsvda").fit(tfidf)
    tvn = tv.get_feature_names_out()
    nmf_topics = [[tvn[i] for i in t.argsort()[-TOPN:][::-1]] for t in nmf.components_]
    res[name]["nmf"] = nmf_topics
    print("NMF topics:")
    for i, t in enumerate(nmf_topics, 1):
        print(f"  T{i}: {', '.join(t)}")

    # ---- TF-IDF distinctive terms per stance
    res[name]["tfidf_by_stance"] = {}
    for s in STANCES:
        sub = df[df["Label"] == s]["clean"]
        if len(sub) < 50:
            continue
        v = TfidfVectorizer(max_features=2000, stop_words=ALL_STOP)
        X = v.fit_transform(sub)
        means = np.asarray(X.mean(axis=0)).ravel()
        nm = v.get_feature_names_out()
        top = [nm[i] for i in means.argsort()[-8:][::-1]]
        res[name]["tfidf_by_stance"][s] = top
        print(f"  TF-IDF {NAMES[s]:14s}: {', '.join(top)}")

    # ---- coherence
    try:
        from gensim.corpora import Dictionary
        from gensim.models import CoherenceModel
        toks = [t.split() for t in df["clean"].sample(min(len(df), 20000), random_state=SEED)]
        d = Dictionary(toks)
        res[name]["coherence"] = {
            "lda_cv": round(float(CoherenceModel(topics=lda_topics, texts=toks, dictionary=d,
                                                 coherence="c_v").get_coherence()), 4),
            "nmf_cv": round(float(CoherenceModel(topics=nmf_topics, texts=toks, dictionary=d,
                                                 coherence="c_v").get_coherence()), 4),
        }
        print(f"  coherence c_v: LDA={res[name]['coherence']['lda_cv']}, "
              f"NMF={res[name]['coherence']['nmf_cv']}")
    except Exception as e:
        print(f"  coherence skipped: {e}")

with open(OUT / "topics_clean.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
print(f"\nSaved -> {OUT / 'topics.json'}")
