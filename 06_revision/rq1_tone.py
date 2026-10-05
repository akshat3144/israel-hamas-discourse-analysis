# -*- coding: utf-8 -*-
"""
RQ1 tone results for a chosen sentiment instrument, with all three adversarial
checks on the headline stance-sentiment contrast.

The paper's RQ1 tone numbers were VADER-based. Human validation
(sentiment_validation.py) found twitter-roberta much closer to human judgement
(kappa 0.58 vs VADER 0.33), so this script recomputes every RQ1 tone result for
either instrument, so the two can be reported side by side:

  * platform contrast in tone (mean score, % negative, rank-biserial r)
  * stance-conditional means and Cramer's V (stance x sentiment label)
  * check 1: Reddit stance labels degraded to YouTube's measured accuracy
  * check 2: off-topic comments removed by the validated relevance classifier
  * check 3: both platforms disattenuated by inverting their measured stance
             confusion matrices, with a bootstrap over the validation items.
             (No script for this check existed in the repository; it is
             reconstructed here from the paper's description and verified by
             reproducing the reported VADER values.)
  * joint sentiment x toxicity with this instrument (Detoxify scores)

Run:  python 06_revision/rq1_tone.py --sent vader|roberta
Out:  06_revision/outputs/rq1_tone_<sent>.json
"""
import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from label_noise_sensitivity import (N_SIMS, REDDIT_ACC, SEED, YOUTUBE_ACC, YT_CONF,
                                     cramers_v, error_distribution)

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
VAL = HERE / "validation"
STANCES = ["P", "I", "N"]
SENTS = ["negative", "neutral", "positive"]
N_BOOT = 2000
T = 0.5


def load(plat, sent):
    f = {"reddit": "reddit_window_clean", "youtube": "youtube_window_clean"}[plat]
    d = pd.read_parquet(OUT / f"{f}.parquet",
                        columns=["index", "Label", "vader_compound", "vader_label"]
                        + (["parent_id"] if plat == "reddit" else []))
    if sent == "roberta":
        rb = pd.concat([pd.read_parquet(x) for x in
                        glob.glob(str(OUT / "roberta" / f"{plat}_*.parquet"))])
        rb = rb.drop_duplicates("index")[["index", "rb_score", "rb_label"]]
        d = d.merge(rb, on="index", how="left")
        assert d["rb_label"].notna().all(), f"{plat}: missing roberta scores"
        d["score"], d["label"] = d["rb_score"], d["rb_label"]
    else:
        d["score"], d["label"] = d["vader_compound"], d["vader_label"]
    return d[d["Label"].isin(STANCES)].reset_index(drop=True)


def rank_biserial(a, b):
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(1 - 2.0 * u / (len(a) * len(b))), float(p)


# ---------------------------------------------------------------- check 3 helpers
def majority(row):
    vals = [v for v in row if isinstance(v, str)]
    c = Counter(vals).most_common()
    if len(c) > 1 and c[0][1] == c[1][1]:
        return vals[0]
    return c[0][0]


def validation_items():
    """High-tier items with P/I/N human gold: (platform, gold, model) - as in
    validation_breakdown.py / label_noise_sensitivity.py."""
    key = pd.read_csv(VAL / "validation_key.csv").set_index("item_id")
    ann = {}
    for f in sorted(VAL.glob("annotator_*.xlsx")):
        d = pd.read_excel(f, sheet_name="Annotate", header=3)
        d["human_label"] = d["human_label"].astype(str).str.strip().str.upper()
        ann[f.stem.split("annotator_", 1)[-1]] = d.set_index("item_id")["human_label"]
    gold = pd.DataFrame(ann).apply(majority, axis=1).rename("gold")
    df = key.join(gold)
    df = df[(df["confidence"] == "High") & df["gold"].isin(STANCES)]
    return df[["platform", "gold", "model_label"]]


def confusion(items):
    c = np.zeros((3, 3))
    for g, m in zip(items["gold"], items["model_label"]):
        if m in STANCES:
            c[STANCES.index(g), STANCES.index(m)] += 1
    return c


def corrected_v(observed, conf):
    """observed[j, s] = sum_i true[i, s] * P(model j | gold i)  ->  solve for true."""
    rows = conf.sum(1, keepdims=True)
    if (rows == 0).any():
        return np.nan
    C = conf / rows
    try:
        true = np.linalg.solve(C.T, observed)
    except np.linalg.LinAlgError:
        return np.nan
    true = np.clip(true, 0, None)
    n = true.sum()
    exp = np.outer(true.sum(1), true.sum(0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum(np.where(exp > 0, (true - exp) ** 2 / exp, 0))
    return float(np.sqrt(chi2 / (n * (min(true.shape) - 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sent", choices=["vader", "roberta"], required=True)
    args = ap.parse_args()
    res = {"instrument": args.sent}
    data = {p: load(p, args.sent) for p in ("reddit", "youtube")}
    r, y = data["reddit"], data["youtube"]

    # ---- platform contrast
    rb, p = rank_biserial(y["score"].to_numpy(), r["score"].to_numpy())
    res["platform"] = {
        "reddit_mean": round(float(r["score"].mean()), 4),
        "youtube_mean": round(float(y["score"].mean()), 4),
        "reddit_pct_negative": round(float((r["label"] == "negative").mean() * 100), 1),
        "youtube_pct_negative": round(float((y["label"] == "negative").mean() * 100), 1),
        "reddit_dist": r["label"].value_counts(normalize=True).round(4).to_dict(),
        "youtube_dist": y["label"].value_counts(normalize=True).round(4).to_dict(),
        "rank_biserial_youtube_vs_reddit": round(rb, 4), "p": p}
    print("platform:", res["platform"])

    # ---- stance
    res["stance"] = {}
    for plat, d in data.items():
        res["stance"][plat] = {
            "means": {s: round(float(d.loc[d.Label == s, "score"].mean()), 4) for s in STANCES},
            "pct_negative": {s: round(float((d.loc[d.Label == s, "label"] == "negative").mean() * 100), 1)
                             for s in STANCES},
            "cramers_v": round(cramers_v(d["Label"], d["label"]), 4)}
        print(plat, res["stance"][plat])
    v_r, v_y = res["stance"]["reddit"]["cramers_v"], res["stance"]["youtube"]["cramers_v"]

    # ---- check 1: degrade Reddit labels to YouTube accuracy
    rng = np.random.default_rng(SEED)
    err = error_distribution(YT_CONF)
    q = 1.0 - YOUTUBE_ACC / REDDIT_ACC
    labels, sent = r["Label"].to_numpy(), r["label"].to_numpy()
    idx = {s: np.flatnonzero(labels == s) for s in STANCES}
    sims = []
    for _ in range(N_SIMS):
        noisy = labels.copy()
        hit = rng.random(len(labels)) < q
        for s in STANCES:
            rows = idx[s][hit[idx[s]]]
            if rows.size:
                noisy[rows] = rng.choice(STANCES, size=rows.size, p=err[s])
        sims.append(cramers_v(pd.Series(noisy), pd.Series(sent)))
    sims = np.array(sims)
    res["check_degraded"] = {"reddit_v": round(float(sims.mean()), 4),
                             "ci95": [round(float(x), 4) for x in np.percentile(sims, [2.5, 97.5])],
                             "ratio_to_youtube": round(float(sims.mean() / v_y), 2)}
    print("check 1 degraded:", res["check_degraded"])

    # ---- check 2: off-topic removed
    c2 = {}
    for plat, d in data.items():
        keep = pd.read_parquet(OUT / f"{plat}_window_relevant.parquet", columns=["index"])["index"]
        s = d[d["index"].isin(set(keep))]
        c2[plat] = {"n": int(len(s)), "v": round(cramers_v(s["Label"], s["label"]), 4),
                    "means": {st: round(float(s.loc[s.Label == st, "score"].mean()), 4)
                              for st in STANCES}}
    c2["ratio"] = round(c2["reddit"]["v"] / c2["youtube"]["v"], 2)
    res["check_offtopic"] = c2
    print("check 2 off-topic:", c2)

    # ---- check 3: disattenuation
    items = validation_items()
    c3 = {}
    for plat, d in data.items():
        it = items[items["platform"] == plat]
        obs = pd.crosstab(d["Label"], d["label"]).reindex(index=STANCES, columns=SENTS,
                                                          fill_value=0).to_numpy(float)
        point = corrected_v(obs, confusion(it))
        brng = np.random.default_rng(SEED)
        boots = [corrected_v(obs, confusion(it.iloc[brng.integers(0, len(it), len(it))]))
                 for _ in range(N_BOOT)]
        boots = np.array([b for b in boots if np.isfinite(b)])
        c3[plat] = {"n_validation_items": int(len(it)),
                    "accuracy": round(float((it.gold == it.model_label).mean()), 4),
                    "v": round(point, 4),
                    "ci95": [round(float(x), 4) for x in np.percentile(boots, [2.5, 97.5])],
                    "n_boot_valid": int(len(boots))}
    c3["ratio"] = round(c3["reddit"]["v"] / c3["youtube"]["v"], 2)
    res["check_disattenuated"] = c3
    print("check 3 corrected:", c3)

    # ---- joint sentiment x toxicity (Detoxify)
    joint = {}
    for plat, d in data.items():
        tox = pd.concat([pd.read_parquet(x) for x in
                         glob.glob(str(OUT / "detoxify" / f"{plat}_*.parquet"))])
        tox = tox.drop_duplicates("index").set_index("index")["dx_toxicity"]
        t = d["index"].map(tox)
        is_tox, is_neg = t >= T, d["label"] == "negative"
        both = int((is_tox & is_neg).sum())
        X = np.column_stack([np.ones(len(d)), d["score"].to_numpy(float),
                             (d.Label == "I").to_numpy(float), (d.Label == "N").to_numpy(float)])
        beta, *_ = np.linalg.lstsq(X, t.to_numpy(float), rcond=None)
        joint[plat] = {"pct_toxic_that_are_negative": round(both / is_tox.sum() * 100, 1),
                       "pct_negative_that_are_toxic": round(both / is_neg.sum() * 100, 1),
                       "spearman": round(float(stats.spearmanr(d["score"], t)[0]), 4),
                       "ols_b": {"score": round(float(beta[1]), 4), "I_vs_P": round(float(beta[2]), 4),
                                 "N_vs_P": round(float(beta[3]), 4)}}
    res["joint_toxicity"] = joint
    print("joint:", joint)

    with open(OUT / f"rq1_tone_{args.sent}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
