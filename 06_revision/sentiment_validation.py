# -*- coding: utf-8 -*-
"""
Human sentiment validation: which automatic sentiment instrument matches people?

Two annotators labelled the 270-item validation sample (the same items as the
stance validation) as POS / NEG / NEU. This script
  1. measures their agreement (Cohen's kappa, Krippendorff's alpha, per platform),
  2. writes an adjudication workbook for the items they disagree on - a third
     annotator picks between the two disputed labels, shown unattributed and in
     random order,
  3. re-scores the 270 texts with VADER, TextBlob and twitter-roberta exactly as the
     pipeline does (raw text; VADER +-0.05, TextBlob +-0.1) and checks the result
     against the stored corpus labels where the item is in the window,
  4. scores each instrument against the human labels, and
  5. re-runs the joint sentiment x toxicity question on human sentiment, using the
     Detoxify scores.

Gold = the two annotators' label where they agree; once adjudication is returned
(validation/sentiment_adjudication.xlsx filled in) gold covers all 270.

Run:  python 06_revision/sentiment_validation.py [--round1]
Out:  06_revision/outputs/sentiment_validation[_round1].json
      06_revision/validation/sentiment_adjudication.xlsx   (if not already filled)
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from openpyxl.worksheet.datavalidation import DataValidation
from sklearn.metrics import cohen_kappa_score, f1_score

from score_validation import krippendorff_alpha_nominal

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
VAL = HERE / "validation"
A1, A2 = "Akshat", "Raghav"
ADJUDICATOR = "Arsh"
# Round 2 (annotators saw the parent comment and video title) is the reported
# round; `--round1` re-scores the superseded first round from
# validation/sentiment_round1/ so that its conclusions can be compared.
ROUND1 = "--round1" in sys.argv
BOOKS = VAL / "sentiment_round1" if ROUND1 else VAL
TAG = "_round1" if ROUND1 else ""
ADJ = BOOKS / "sentiment_adjudication.xlsx"
MAP = {"POS": "positive", "NEG": "negative", "NEU": "neutral"}
CLASSES = ["negative", "neutral", "positive"]
T = 0.5
SEED = 11
N_BOOT = 2000
res = {}


def read_book(name):
    ws = openpyxl.load_workbook(BOOKS / f"sentiment_{name}.xlsx", data_only=True)["Annotate"]
    col = [c.value for c in ws[4]].index("sentiment")
    rows = [r for r in ws.iter_rows(min_row=5, values_only=True) if r[0]]
    return pd.Series({r[0]: MAP[str(r[col]).strip()] for r in rows}, name=name)


def kappa_ci(a, b, n_boot=2000):
    rng = np.random.default_rng(SEED)
    a, b = np.asarray(a), np.asarray(b)
    ks = [cohen_kappa_score(a[i], b[i])
          for i in (rng.integers(0, len(a), len(a)) for _ in range(n_boot))]
    return [round(float(np.percentile(ks, 2.5)), 3), round(float(np.percentile(ks, 97.5)), 3)]


def agreement(a, b):
    return {"n": int(len(a)), "pct_agree": round(float((a == b).mean() * 100), 1),
            "cohen_kappa": round(float(cohen_kappa_score(a, b)), 3),
            "kappa_ci95": kappa_ci(a, b),
            "krippendorff_alpha": round(float(krippendorff_alpha_nominal(
                np.array([a.to_numpy(object), b.to_numpy(object)]))), 3)}


def score_instruments(texts):
    from textblob import TextBlob

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    vader = SentimentIntensityAnalyzer()

    def v(t):
        c = vader.polarity_scores(str(t))["compound"]
        return "positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral"

    def tb(t):
        p = TextBlob(str(t)).sentiment.polarity
        return "positive" if p > 0.1 else "negative" if p < -0.1 else "neutral"

    # same code and settings as the full-corpus run (500 chars, 128 tokens)
    import roberta_score_corpus as rsc
    rob = rsc.score(*rsc.load(), [str(t) for t in texts])["rb_label"].tolist()
    return pd.DataFrame({"vader": [v(t) for t in texts],
                         "textblob": [tb(t) for t in texts],
                         "roberta": rob}, index=texts.index)


def write_adjudication(dis, blind):
    """Round 2: same reading context as the annotators had (video title, parent)."""
    ctxd = pd.read_parquet(OUT / "validation_context.parquet").set_index("item_id")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Adjudicate"
    ws.append(["Sentiment adjudication - pick the better of the two labels"])
    ws.append(["Two annotators disagreed on these comments. Read the context (C) and the "
               "comment being replied to (D), then choose in column F the label that better "
               "describes the TONE of the comment itself (E) - not the side it takes, and "
               "not the tone of the post or parent. Same rules as the guidelines. Work alone."])
    ws.append([])
    ws.append(["item_id", "platform", "context", "replying to", "comment text",
               "your choice", "option A", "option B", "notes"])
    rng = random.Random(SEED)

    def safe(t):
        t = str(t)
        return "'" + t if t and t[0] in "=+@-" else t

    for i, (item, row) in enumerate(dis.iterrows(), start=5):
        opts = [row[A1], row[A2]]
        rng.shuffle(opts)
        opts = [k for o in opts for k, v_ in MAP.items() if v_ == o]
        b, c = blind.loc[item], ctxd.loc[item]
        if b.context_subreddit:
            ctx = f"r/{b.context_subreddit}\n\u201c{b.context_post_title}\u201d"
        else:
            ctx = f"YouTube video\n\u201c{c.video_title}\u201d"
        parent = (c.parent_text or "(reply - parent comment unavailable)") if c.is_reply else ""
        ws.append([item, b.platform, ctx, safe(parent), safe(b.text), None,
                   opts[0], opts[1], None])
        dv = DataValidation(type="list", formula1=f'"{opts[0]},{opts[1]}"', allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"F{i}")
    for col, w in zip("ABCDEFGHI", (9, 9, 26, 46, 66, 12, 10, 10, 24)):
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=5):
        for j in (2, 3, 4):
            row[j].alignment = openpyxl.styles.Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A5"
    wb.save(ADJ)


def read_adjudication():
    if not ADJ.exists():
        return {}
    ws = openpyxl.load_workbook(ADJ, data_only=True)["Adjudicate"]
    hdr = [c.value for c in ws[4]]
    col = hdr.index("your choice")
    return {r[0]: MAP[str(r[col]).strip()] for r in ws.iter_rows(min_row=5, values_only=True)
            if r[0] and r[col] and str(r[col]).strip() in MAP}


def main():
    blind = pd.read_csv(VAL / "validation_blind.csv", encoding="utf-8-sig").fillna("")
    blind = blind.set_index("item_id")
    key = pd.read_csv(VAL / "validation_key.csv", encoding="utf-8-sig").set_index("item_id")
    h = pd.concat([read_book(A1), read_book(A2)], axis=1).join(blind["platform"])
    assert len(h) == 270 and h[[A1, A2]].notna().all().all()

    # ---------------------------------------------------------------- agreement
    res["agreement"] = {"all": agreement(h[A1], h[A2])}
    for p in ("reddit", "youtube"):
        s = h[h.platform == p]
        res["agreement"][p] = agreement(s[A1], s[A2])
    res["confusion"] = pd.crosstab(h[A1], h[A2]).to_dict()
    res["distribution"] = {a: h[a].value_counts().to_dict() for a in (A1, A2)}
    print("agreement:", json.dumps(res["agreement"], indent=1))

    # ---------------------------------------------------------------- adjudication
    dis = h[h[A1] != h[A2]]
    adj = read_adjudication()
    if not adj:
        write_adjudication(dis, blind)
        print(f"wrote {ADJ.name}: {len(dis)} items to adjudicate")
    h["gold"] = np.where(h[A1] == h[A2], h[A1], None)
    for item, lab in adj.items():
        h.loc[item, "gold"] = lab
    res["annotators"], res["adjudicator"] = [A1, A2], ADJUDICATOR
    res["gold"] = {"n_agreed": int((h[A1] == h[A2]).sum()), "n_adjudicated": len(adj),
                   "n_disagreements": int(len(dis)), "n_gold": int(h["gold"].notna().sum())}

    # ---------------------------------------------------------------- instruments
    inst = score_instruments(blind.loc[h.index, "text"])
    h = h.join(inst)
    # fidelity: does re-scoring reproduce the corpus labels?
    fid = {}
    for p, f in (("reddit", "reddit_window"), ("youtube", "youtube_window")):
        w = pd.read_parquet(OUT / f"{f}.parquet", columns=["index", "vader_label", "textblob_label"])
        kk = key[key.platform == p][["row_index"]].join(h[["vader", "textblob"]])
        m = kk.merge(w, left_on="row_index", right_on="index", how="inner")
        fid[p] = {"n_in_window": int(len(m)),
                  "vader_match_pct": round(float((m.vader == m.vader_label).mean() * 100), 1),
                  "textblob_match_pct": round(float((m.textblob == m.textblob_label).mean() * 100), 1)}
    res["rescoring_fidelity"] = fid
    print("fidelity:", fid)

    g = h[h.gold.notna()]
    perf = {}
    for scope, s in (("all", g), ("reddit", g[g.platform == "reddit"]),
                     ("youtube", g[g.platform == "youtube"])):
        perf[scope] = {}
        for tool in ("vader", "textblob", "roberta"):
            perf[scope][tool] = {
                "n": int(len(s)),
                "accuracy": round(float((s[tool] == s.gold).mean()), 3),
                "macro_f1": round(float(f1_score(s.gold, s[tool], labels=CLASSES,
                                                 average="macro")), 3),
                "kappa": round(float(cohen_kappa_score(s.gold, s[tool])), 3),
                "negative_recall": round(float((s[s.gold == "negative"][tool] == "negative").mean()), 3),
                "negative_precision": round(float((s[s[tool] == "negative"].gold == "negative").mean()), 3)
                if (s[tool] == "negative").any() else None,
            }
        # human ceiling on the same items: each annotator vs gold is circular, so
        # report annotator-vs-annotator kappa on this scope for comparison
    res["instrument_vs_human"] = perf
    for scope, d in perf.items():
        print(scope, {t: (v["accuracy"], v["macro_f1"], v["kappa"]) for t, v in d.items()})

    # ---------------------------------------------------------------- joint, human
    import glob
    sc = {p: pd.concat([pd.read_parquet(x) for x in glob.glob(str(OUT / "detoxify" / f"{p}_*.parquet"))])
          .drop_duplicates("index").set_index("index")["dx_toxicity"] for p in ("reddit", "youtube")}
    h["toxicity"] = [sc[p].get(key.loc[i, "row_index"], np.nan) for i, p in zip(h.index, h.platform)]
    missing = h["toxicity"].isna()
    if missing.any():
        from detox_common import load, score
        m = load("cpu")
        h.loc[missing, "toxicity"] = score(m, blind.loc[h.index[missing], "text"].tolist())["toxicity"]
    res["n_toxicity_rescored_locally"] = int(missing.sum())
    gg = h[h.gold.notna()]
    tox = gg.toxicity >= T
    joint = {"n": int(len(gg)), "n_toxic": int(tox.sum())}
    for lab in ("gold", "vader", "textblob", "roberta"):
        neg = gg[lab] == "negative"
        joint[lab] = {"pct_toxic_that_are_negative": round(float((neg & tox).sum() / tox.sum() * 100), 1),
                      "pct_negative_that_are_toxic": round(float((neg & tox).sum() / neg.sum() * 100), 1)}
    res["joint_on_validation_sample"] = joint
    print("joint:", json.dumps(joint, indent=1))

    # ---------------------------------------------------------------- platform
    # The paper's two RQ1 tone claims, re-asked on human labels (High tier, the
    # analysed corpus). (a) overall negativity per platform, reweighted from the
    # stance-balanced sample to each platform's corpus stance mix; (b) the
    # stance x tone association with human labels for BOTH stance and tone.
    from label_noise_sensitivity import cramers_v
    from rq1_tone import validation_items
    mix = {}
    for plat, f in (("reddit", "reddit_window_clean"), ("youtube", "youtube_window_clean")):
        lab = pd.read_parquet(OUT / f"{f}.parquet", columns=["Label"])["Label"]
        mix[plat] = lab.value_counts(normalize=True).reindex(["P", "I", "N"]).to_dict()
    hk = h.join(key[["model_label", "confidence"]])
    hk = hk.join(validation_items()[["gold"]].rename(columns={"gold": "stance_gold"}))
    hi = hk[hk["confidence"] == "High"]
    rng = np.random.default_rng(SEED)

    def wneg(s, col, plat):
        return sum(mix[plat][st] * (s.loc[s.model_label == st, col] == "negative").mean()
                   for st in ("P", "I", "N"))

    plat_cmp = {"stance_mix": mix}
    for col in ("gold", "roberta", "vader", "textblob"):
        est = {p: wneg(hi[hi.platform == p], col, p) for p in ("reddit", "youtube")}
        diffs = []
        for _ in range(N_BOOT):
            b = {}
            for p in ("reddit", "youtube"):
                s_ = hi[hi.platform == p]
                s_ = pd.concat([g.sample(len(g), replace=True, random_state=int(rng.integers(1e9)))
                                for _, g in s_.groupby("model_label")])
                b[p] = wneg(s_, col, p)
            diffs.append(b["reddit"] - b["youtube"])
        plat_cmp[f"pct_negative_{col}"] = {
            "reddit": round(est["reddit"] * 100, 1), "youtube": round(est["youtube"] * 100, 1),
            "diff_pp": round((est["reddit"] - est["youtube"]) * 100, 1),
            "diff_ci95_pp": [round(float(x) * 100, 1) for x in np.percentile(diffs, [2.5, 97.5])]}
    for col in ("gold", "roberta", "vader"):
        out = {}
        for p in ("reddit", "youtube"):
            s_ = hk[hk.stance_gold.notna() & (hk.platform == p)]
            bs = []
            for _ in range(N_BOOT):
                b = s_.sample(len(s_), replace=True, random_state=int(rng.integers(1e9)))
                bs.append(cramers_v(b["stance_gold"], b[col]))
            out[p] = {"n": int(len(s_)), "v": round(cramers_v(s_["stance_gold"], s_[col]), 3),
                      "ci95": [round(float(x), 3) for x in np.percentile(bs, [2.5, 97.5])]}
        out["ratio"] = round(out["reddit"]["v"] / out["youtube"]["v"], 2)
        plat_cmp[f"stance_tone_v_{col}"] = out
    res["platform_comparison_high_tier"] = plat_cmp
    print("platform comparison:", json.dumps(plat_cmp, indent=1))

    h.drop(columns=[]).to_csv(OUT / f"sentiment_validation_items{TAG}.csv")

    # Released labels. The annotation workbooks themselves are not released:
    # round 2 shows YouTube video titles and parent comments as reading context,
    # which the YouTube Developer Policies and the platforms' terms do not allow
    # us to redistribute. Item IDs join to validation_blind.csv.
    lab = h[["platform", A1, A2]].copy()
    lab["adjudicated"] = pd.Series(adj)
    lab["gold"] = h["gold"]
    lab.rename(columns={A1: f"label_{A1}", A2: f"label_{A2}"}).rename_axis("item_id") \
        .to_csv(VAL / f"sentiment_labels{TAG}.csv")
    with open(OUT / f"sentiment_validation{TAG}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)


if __name__ == "__main__":
    main()
