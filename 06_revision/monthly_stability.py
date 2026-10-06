# -*- coding: utf-8 -*-
"""
Are the RQ1 results stable across the seven-month window, or driven by one month?

Recomputes, for each calendar month of the matched window and each platform,
the quantities the paper's claims rest on: the stance-tone association
(Cramer's V, RoBERTa and VADER), the share of negative comments, the toxicity
rate, and the partisan-minus-neutral toxicity gap. Reddit is dated by comment
time, YouTube by video publish date (the paper's time anchors). May 2024 holds
only seven days and is flagged.

Run:  python 06_revision/monthly_stability.py
Out:  06_revision/outputs/monthly_stability.json
"""
import glob
import json
from pathlib import Path

import pandas as pd

from label_noise_sensitivity import cramers_v

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
STANCES = ["P", "I", "N"]
T = 0.5


def load(plat):
    f = {"reddit": "reddit_window_clean", "youtube": "youtube_window_clean"}[plat]
    d = pd.read_parquet(OUT / f"{f}.parquet",
                        columns=["index", "Label", "vader_label", "time_anchor"]
                        + (["video id"] if plat == "youtube" else [])).drop_duplicates("index")
    tox = pd.concat([pd.read_parquet(x) for x in glob.glob(str(OUT / "detoxify" / f"{plat}_*.parquet"))])
    rb = pd.concat([pd.read_parquet(x) for x in glob.glob(str(OUT / "roberta" / f"{plat}_*.parquet"))])
    d = d.merge(tox.drop_duplicates("index")[["index", "dx_toxicity"]], on="index")
    d = d.merge(rb.drop_duplicates("index")[["index", "rb_label"]], on="index")
    d = d[d["Label"].isin(STANCES)]
    d["month"] = pd.to_datetime(d["time_anchor"]).dt.to_period("M").astype(str)
    return d


def main():
    res = {}
    for plat in ("reddit", "youtube"):
        d = load(plat)
        rows = {}
        for m, g in d.groupby("month"):
            part = g["Label"].isin(["P", "I"])
            rows[m] = {
                "n": int(len(g)),
                "v_roberta": round(cramers_v(g["Label"], g["rb_label"]), 4),
                "v_vader": round(cramers_v(g["Label"], g["vader_label"]), 4),
                "pct_negative_roberta": round(float((g["rb_label"] == "negative").mean() * 100), 1),
                "pct_toxic": round(float((g["dx_toxicity"] >= T).mean() * 100), 2),
                "partisan_minus_neutral_toxicity": round(
                    float(g.loc[part, "dx_toxicity"].mean() - g.loc[~part, "dx_toxicity"].mean()), 4),
                "partial_month": m == "2024-05",
            }
            if "video id" in g:
                vc = g["video id"].value_counts()
                rows[m]["n_videos"] = int(len(vc))
                rows[m]["top3_video_share_pct"] = round(float(vc.head(3).sum() / vc.sum() * 100), 1)
        res[plat] = rows
    # the cross-platform comparisons, month by month
    months = sorted(set(res["reddit"]) & set(res["youtube"]))
    cmp_ = {}
    for m in months:
        r, y = res["reddit"][m], res["youtube"][m]
        cmp_[m] = {"v_ratio_roberta": round(r["v_roberta"] / y["v_roberta"], 2),
                   "v_ratio_vader": round(r["v_vader"] / y["v_vader"], 2),
                   "reddit_more_toxic": r["pct_toxic"] > y["pct_toxic"],
                   "partisan_gap_positive_both": (r["partisan_minus_neutral_toxicity"] > 0
                                                  and y["partisan_minus_neutral_toxicity"] > 0)}
    full = [m for m in months if m != "2024-05"]
    res["comparison"] = cmp_
    res["summary_full_months"] = {
        "months": full,
        "reddit_v_exceeds_youtube_roberta": sum(res["reddit"][m]["v_roberta"] > res["youtube"][m]["v_roberta"] for m in full),
        "v_ratio_roberta_min": min(cmp_[m]["v_ratio_roberta"] for m in full),
        "v_ratio_roberta_max": max(cmp_[m]["v_ratio_roberta"] for m in full),
        "reddit_more_toxic_months": sum(cmp_[m]["reddit_more_toxic"] for m in full),
        "partisan_gap_positive_both_months": sum(cmp_[m]["partisan_gap_positive_both"] for m in full),
    }
    for plat in ("reddit", "youtube"):
        print(plat)
        for m, r in res[plat].items():
            print(f"  {m}  n={r['n']:>7,}  V_rb={r['v_roberta']:.3f}  V_vader={r['v_vader']:.3f}  "
                  f"neg={r['pct_negative_roberta']:5.1f}%  toxic={r['pct_toxic']:5.2f}%  "
                  f"gap={r['partisan_minus_neutral_toxicity']:+.3f}")
    print("summary:", res["summary_full_months"])
    with open(OUT / "monthly_stability.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
