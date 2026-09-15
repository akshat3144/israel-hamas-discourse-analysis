# -*- coding: utf-8 -*-
"""
Human validation showed that 10-16% of the analytic corpus is off-topic content
carrying a stance label, and that the model's own confidence tier does not catch
it. This builds an explicit relevance filter, validates it against the human
annotations, applies it, and re-runs the headline statistics on the cleaned
corpus so we can say whether the findings depend on the contamination.

Design note: the classifier is TRAINED on the model's own Irrelevant labels,
which are abundant, but its threshold is TUNED and SCORED on the 270 human-
labelled items, which are independent of the model. Training on model labels
alone would inherit the model's blind spots; the human set is what tells us
whether the filter actually catches what the model missed.

Run:  python 06_revision/relevance_filter.py
"""
import glob
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LABELED = ROOT / "00_data_collection_and_labeling" / "outputs"
OUT = HERE / "outputs"
VAL = HERE / "validation"

SOURCES = {
    "reddit": (LABELED / "reddit_labeled_full.jsonl", "self_text"),
    "youtube": (LABELED / "youtube_labeled_full.jsonl", "text"),
}
CAP = 120_000
SEED = 20240507


def sample_training(path, field, rng):
    """High-confidence items only: R (irrelevant) vs P/I/N (on topic)."""
    pos, neg = [], []          # pos = irrelevant
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("Confidence") != "High":
                continue
            lab = row.get("Label")
            txt = row.get(field)
            if not isinstance(txt, str):
                continue
            txt = " ".join(txt.split())
            if not (3 <= len(txt) <= 2000):
                continue
            if lab == "R":
                if len(pos) < CAP and rng.random() < 0.30:
                    pos.append(txt)
            elif lab in ("P", "I", "N"):
                if len(neg) < CAP and rng.random() < 0.12:
                    neg.append(txt)
    return pos, neg


def human_gold():
    """The 270 validation items with their majority human label."""
    key = pd.read_csv(VAL / "validation_key.csv").set_index("item_id")
    blind = pd.read_csv(VAL / "validation_blind.csv").set_index("item_id")
    ann = {}
    for f in sorted(glob.glob(str(VAL / "annotator_*.xlsx"))):
        who = os.path.basename(f).split("annotator_", 1)[-1].replace(".xlsx", "")
        d = pd.read_excel(f, sheet_name="Annotate", header=3)
        ann[who] = d.assign(
            l=d.human_label.astype(str).str.strip().str.upper()
        ).set_index("item_id")["l"]
    A = pd.DataFrame(ann)

    def maj(r):
        c = Counter(r).most_common()
        return None if c[0][1] == 1 else c[0][0]

    g = A.apply(maj, axis=1).rename("gold")
    df = key.join(g).join(blind["text"])
    df["unanimous"] = [len(set(A.loc[i])) == 1 for i in df.index]
    return df.dropna(subset=["gold"])


def cramers_v(a, b):
    tab = pd.crosstab(a, b).to_numpy(dtype=float)
    n = tab.sum()
    exp = np.outer(tab.sum(1), tab.sum(0)) / n
    chi2 = float(((tab - exp) ** 2 / np.where(exp == 0, np.nan, exp)).sum())
    return float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))


def main():
    rng = np.random.default_rng(SEED)
    gold = human_gold()
    results = {}

    for platform, (path, field) in SOURCES.items():
        print(f"\n{'='*64}\n{platform}")
        print("  sampling training data ...", flush=True)
        pos, neg = sample_training(path, field, rng)
        print(f"    irrelevant={len(pos):,}  on-topic={len(neg):,}")

        X = pos + neg
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        vec = TfidfVectorizer(max_features=120_000, ngram_range=(1, 2),
                              min_df=3, sublinear_tf=True, strip_accents="unicode")
        Z = vec.fit_transform(X)
        clf = LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced")
        clf.fit(Z, y)

        # --- score on the HUMAN set (independent of the model's labels) ---
        g = gold[gold.platform == platform]
        gx = vec.transform(g.text.astype(str).tolist())
        p = clf.predict_proba(gx)[:, 1]
        y_true = (g.gold == "R").astype(int).to_numpy()
        auc = roc_auc_score(y_true, p)
        print(f"  ROC-AUC against human 'irrelevant' judgements: {auc:.3f} (n={len(g)})")

        # pick the threshold that best separates human-R from human-stance
        best = max(
            ({"t": float(t),
              "recall": float(((p >= t) & (y_true == 1)).sum() / max(y_true.sum(), 1)),
              "fpr": float(((p >= t) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1))}
             for t in np.arange(0.30, 0.96, 0.01)),
            key=lambda d: d["recall"] - 2.0 * d["fpr"])
        t = best["t"]
        print(f"  threshold {t:.2f}: catches {100*best['recall']:.0f}% of human-irrelevant, "
              f"discards {100*best['fpr']:.0f}% of genuine stance comments")

        # --- apply to the analytic corpus ---
        src = OUT / f"{platform}_window_clean.parquet"
        txtcol = "self_text" if platform == "reddit" else "text"
        df = pd.read_parquet(src)
        probs = np.empty(len(df), dtype=float)
        step = 200_000
        texts = df[txtcol].astype(str).tolist()
        for i in range(0, len(texts), step):
            probs[i:i+step] = clf.predict_proba(
                vec.transform(texts[i:i+step]))[:, 1]
        keep = probs < t
        print(f"  corpus {len(df):,} -> {int(keep.sum()):,} "
              f"({100*(1-keep.mean()):.1f}% removed as off-topic)")

        before_v = cramers_v(df["Label"], df["vader_label"])
        after_v = cramers_v(df.loc[keep, "Label"], df.loc[keep, "vader_label"])
        before_sh = (df["Label"].value_counts(normalize=True) * 100).round(2).to_dict()
        after_sh = (df.loc[keep, "Label"].value_counts(normalize=True) * 100).round(2).to_dict()
        print(f"  Cramer's V  {before_v:.4f} -> {after_v:.4f}")
        print(f"  stance mix  {before_sh}")
        print(f"           -> {after_sh}")

        df.loc[keep].to_parquet(OUT / f"{platform}_window_relevant.parquet", index=False)
        results[platform] = {
            "auc_vs_human": round(float(auc), 4),
            "threshold": round(t, 2),
            "human_recall": round(best["recall"], 4),
            "human_fpr": round(best["fpr"], 4),
            "n_before": int(len(df)),
            "n_after": int(keep.sum()),
            "removed_pct": round(100 * float(1 - keep.mean()), 2),
            "cramers_v_before": round(before_v, 4),
            "cramers_v_after": round(after_v, 4),
            "stance_pct_before": before_sh,
            "stance_pct_after": after_sh,
        }

    (OUT / "relevance_filter.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'relevance_filter.json'}")


if __name__ == "__main__":
    main()
