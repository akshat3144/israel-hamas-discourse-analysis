# -*- coding: utf-8 -*-
"""
Stage 4: stance classification, addressing reviewer R19.

R19 makes three criticisms:
  (a) cross-platform failure is unsurprising because comment lengths differ
      -> we re-run transfer on a LENGTH-MATCHED subsample and report whether the
         gap survives;
  (b) feature importance is taken from one sub-model while the ensemble is the
      headline model -> we compute PERMUTATION importance on the ensemble itself;
  (c) the section is weak/ill-fitting -> we reframe it as a diagnostic of lexical
      divergence, reporting both raw and length-controlled transfer.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
STANCES = ["P", "I", "N"]
NAMES = {"P": "Pro-Palestine", "I": "Pro-Israel", "N": "Neutral"}
MAX_TRAIN, MAX_EVAL = 40000, 20000
SEED = 42
res = {}

print("Loading ...")
r = pd.read_parquet(OUT / "reddit_window.parquet")[["Label", "text_std", "word_count"]]
y = pd.read_parquet(OUT / "youtube_window.parquet")[["Label", "text_std", "word_count"]]
r = r[r["text_std"].str.strip() != ""]
y = y[y["text_std"].str.strip() != ""]
print(f"  Reddit {len(r):,} | YouTube {len(y):,}")


def cap(df, n, seed=SEED):
    return df if len(df) <= n else df.groupby("Label", group_keys=False).sample(
        frac=n / len(df), random_state=seed)


def ensemble():
    return VotingClassifier([
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ("svm", CalibratedClassifierCV(LinearSVC(class_weight="balanced", max_iter=3000), cv=3)),
        ("rf", RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                      n_jobs=-1, random_state=SEED)),
    ], voting="soft")


def run(train_df, test_df, tag):
    tr, te = cap(train_df, MAX_TRAIN), cap(test_df, MAX_EVAL)
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english",
                            ngram_range=(1, 2), min_df=5)
    Xtr = tfidf.fit_transform(tr["text_std"])
    Xte = tfidf.transform(te["text_std"])
    model = ensemble().fit(Xtr, tr["Label"])
    pred = model.predict(Xte)
    acc = accuracy_score(te["Label"], pred)
    f1 = f1_score(te["Label"], pred, average="macro")
    print(f"  {tag:28s} n_tr={len(tr):>6,} n_te={len(te):>6,}  acc={acc:.4f}  macroF1={f1:.4f}")
    return {"tag": tag, "n_train": int(len(tr)), "n_test": int(len(te)),
            "accuracy": round(float(acc), 4), "macro_f1": round(float(f1), 4)}, model, tfidf, te


# ---------------------------------------------------------------- (1) raw
print("\n=== Raw (unmatched) ===")
raw = {}
r_tr, r_te = train_test_split(r, test_size=0.2, random_state=SEED, stratify=r["Label"])
y_tr, y_te = train_test_split(y, test_size=0.2, random_state=SEED, stratify=y["Label"])
raw["R->R"], red_model, red_tfidf, red_te = run(r_tr, r_te, "Reddit -> Reddit")
raw["Y->Y"], yt_model, yt_tfidf, yt_te = run(y_tr, y_te, "YouTube -> YouTube")
raw["R->Y"], *_ = run(r, y, "Reddit -> YouTube")
raw["Y->R"], *_ = run(y, r, "YouTube -> Reddit")
res["raw"] = raw

# ---------------------------------------------------------------- (2) length-matched
print("\n=== Length-matched (R19a) ===")
# Match the word-count distribution by binning and taking equal counts per bin.
BINS = [0, 5, 10, 15, 20, 30, 40, 60, 100, 10 ** 9]
r2 = r.assign(_b=pd.cut(r["word_count"], BINS, labels=False, include_lowest=True))
y2 = y.assign(_b=pd.cut(y["word_count"], BINS, labels=False, include_lowest=True))
parts_r, parts_y = [], []
rng = np.random.default_rng(SEED)
for b in sorted(set(r2["_b"].dropna()) & set(y2["_b"].dropna())):
    rb, yb = r2[r2["_b"] == b], y2[y2["_b"] == b]
    k = min(len(rb), len(yb))
    if k < 50:
        continue
    parts_r.append(rb.sample(k, random_state=SEED))
    parts_y.append(yb.sample(k, random_state=SEED))
rm = pd.concat(parts_r).drop(columns="_b")
ym = pd.concat(parts_y).drop(columns="_b")
print(f"  matched sizes: Reddit {len(rm):,} | YouTube {len(ym):,}")
print(f"  mean words after matching: Reddit {rm['word_count'].mean():.1f} | "
      f"YouTube {ym['word_count'].mean():.1f}")
res["length_matched_sizes"] = {"reddit": int(len(rm)), "youtube": int(len(ym)),
                               "reddit_mean_words": round(float(rm["word_count"].mean()), 2),
                               "youtube_mean_words": round(float(ym["word_count"].mean()), 2)}

matched = {}
rm_tr, rm_te = train_test_split(rm, test_size=0.2, random_state=SEED, stratify=rm["Label"])
ym_tr, ym_te = train_test_split(ym, test_size=0.2, random_state=SEED, stratify=ym["Label"])
matched["R->R"], *_ = run(rm_tr, rm_te, "matched Reddit -> Reddit")
matched["Y->Y"], *_ = run(ym_tr, ym_te, "matched YouTube -> YouTube")
matched["R->Y"], *_ = run(rm, ym, "matched Reddit -> YouTube")
matched["Y->R"], *_ = run(ym, rm, "matched YouTube -> Reddit")
res["length_matched"] = matched

# Does the transfer gap survive length matching?
gap_raw = ((raw["R->R"]["macro_f1"] - raw["R->Y"]["macro_f1"]) +
           (raw["Y->Y"]["macro_f1"] - raw["Y->R"]["macro_f1"])) / 2
gap_m = ((matched["R->R"]["macro_f1"] - matched["R->Y"]["macro_f1"]) +
         (matched["Y->Y"]["macro_f1"] - matched["Y->R"]["macro_f1"])) / 2
res["transfer_gap"] = {"raw": round(float(gap_raw), 4), "length_matched": round(float(gap_m), 4)}
print(f"\n  mean within-minus-cross macro-F1 gap: raw={gap_raw:.4f}  matched={gap_m:.4f}")

# ---------------------------------------------------------------- (3) CV
print("\n=== 5-fold CV macro-F1 (LR component, for speed) ===")
cv = {}
for name, df in [("reddit", r), ("youtube", y)]:
    s = cap(df, MAX_TRAIN)
    pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=5000, stop_words="english",
                                               ngram_range=(1, 2), min_df=5)),
                     ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    sc = cross_val_score(pipe, s["text_std"], s["Label"],
                         cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                         scoring="f1_macro", n_jobs=-1)
    cv[name] = {"mean": round(float(sc.mean()), 4),
                "ci95": round(float(1.96 * sc.std() / np.sqrt(5)), 4)}
    print(f"  {name}: {cv[name]['mean']:.4f} +/- {cv[name]['ci95']:.4f}")
res["cv_macro_f1"] = cv

# ---------------------------------------------------------------- (4) permutation importance on the ENSEMBLE (R19b)
print("\n=== Permutation importance on the ENSEMBLE (R19b) ===")


def perm_importance(model, tfidf, test_df, top_k=25, repeats=3, eval_n=4000):
    te = test_df if len(test_df) <= eval_n else test_df.sample(eval_n, random_state=SEED)
    X = tfidf.transform(te["text_std"]).tocsc()
    ytrue = te["Label"].to_numpy()
    base = f1_score(ytrue, model.predict(X), average="macro")
    names = tfidf.get_feature_names_out()
    lr = model.named_estimators_["lr"]
    classes = list(model.classes_)
    # candidate features: strongest LR coefficients per class
    cand = set()
    for s in STANCES:
        if s in classes:
            c = lr.coef_[classes.index(s)]
            cand.update(np.argsort(c)[-top_k:].tolist())
    cand = sorted(cand)
    rng = np.random.default_rng(SEED)
    drops = {}
    for j in cand:
        col = X[:, j].toarray().ravel()
        d = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(col).reshape(-1, 1)
            d.append(base - f1_score(ytrue, model.predict(Xp), average="macro"))
        drops[names[j]] = float(np.mean(d))
    top = sorted(drops.items(), key=lambda kv: kv[1], reverse=True)[:20]
    return {"baseline_macro_f1": round(float(base), 4), "n_candidates": len(cand),
            "top_features": [(w, round(v, 5)) for w, v in top]}


pi_r = perm_importance(red_model, red_tfidf, red_te)
print(f"  Reddit  baseline macroF1={pi_r['baseline_macro_f1']} over {pi_r['n_candidates']} features")
print("   top:", ", ".join(w for w, _ in pi_r["top_features"][:10]))
pi_y = perm_importance(yt_model, yt_tfidf, yt_te)
print(f"  YouTube baseline macroF1={pi_y['baseline_macro_f1']} over {pi_y['n_candidates']} features")
print("   top:", ", ".join(w for w, _ in pi_y["top_features"][:10]))
res["permutation_importance"] = {"reddit": pi_r, "youtube": pi_y}

with open(OUT / "ml_stance.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
print(f"\nSaved -> {OUT / 'ml_stance.json'}")
