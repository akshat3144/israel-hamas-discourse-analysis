# -*- coding: utf-8 -*-
"""
Instrument check: does Detoxify reproduce Perspective on the 600 comments that
Perspective already scored?

Before Detoxify replaces Perspective for the full-corpus analysis we need to know
(1) how closely the two instruments agree, attribute by attribute, (2) whether
they flag the same comments at the 0.5 threshold, and (3) whether the paper's
joint sentiment x toxicity findings come out the same on the same 600 comments
when Detoxify is the instrument. Also measures the fp16-vs-fp32 drift.

Run (GPU env):  python 06_revision/detoxify_vs_perspective.py
Out:            06_revision/outputs/detoxify_vs_perspective.json
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, roc_auc_score

import detox_common as dc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"
TOX = ROOT / "05_toxicity_analysis" / "outputs"
T = 0.5


def joint(d, tox_col):
    """The paper's joint sentiment x toxicity numbers, for one instrument."""
    d = d.dropna(subset=[tox_col, "vader_compound", "vader_label", "Label"])
    is_tox = d[tox_col] >= T
    is_neg = d["vader_label"] == "negative"
    both = int((is_tox & is_neg).sum())
    X = np.column_stack([np.ones(len(d)), d["vader_compound"].to_numpy(float),
                         (d["Label"] == "I").to_numpy(float),
                         (d["Label"] == "N").to_numpy(float)])
    y = d[tox_col].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(d) - X.shape[1]
    se = np.sqrt(np.diag((resid @ resid) / dof * np.linalg.inv(X.T @ X)))
    p = 2 * stats.t.sf(np.abs(beta / se), dof)
    rho, _ = stats.spearmanr(d["vader_compound"], d[tox_col])
    return {
        "n": int(len(d)), "n_toxic": int(is_tox.sum()), "n_negative": int(is_neg.sum()),
        "pct_toxic_that_are_negative": round(both / is_tox.sum() * 100, 1) if is_tox.sum() else None,
        "pct_negative_that_are_toxic": round(both / is_neg.sum() * 100, 1) if is_neg.sum() else None,
        "spearman_sentiment_toxicity": round(float(rho), 3),
        "ols_b_sentiment": round(float(beta[1]), 4), "ols_p_sentiment": float(p[1]),
        "ols_b_I_vs_P": round(float(beta[2]), 4), "ols_p_I_vs_P": float(p[2]),
        "ols_b_N_vs_P": round(float(beta[3]), 4), "ols_p_N_vs_P": float(p[3]),
    }


def main():
    res = {"model": f"detoxify-{dc.MODEL}", "threshold": T}
    frames = {}
    for plat, wfile, tcol in [("reddit", "reddit_window", "self_text"),
                              ("youtube", "youtube_window", "text")]:
        d = pd.read_csv(TOX / f"{plat}_perspective.csv")
        w = pd.read_parquet(OUT / f"{wfile}.parquet",
                            columns=["index", "Label", "vader_compound", "vader_label"])
        d = d.merge(w, on="index", how="left")
        if "Label" not in d or d["Label"].isna().all():
            raise SystemExit(f"{plat}: no stance labels joined")
        txtcol = tcol if tcol in d else "text"
        d["_text"] = d[txtcol].astype(str)
        frames[plat] = d

    m = dc.load()
    print(f"device: {m.model.device}")
    for plat, d in frames.items():
        t0 = time.time()
        s32 = dc.score(m, d["_text"].tolist(), fp16=False)
        dt = time.time() - t0
        s16 = dc.score(m, d["_text"].tolist(), fp16=True)
        for c in dc.PAIRS:
            d[f"dx_{c}"] = s32[c]
        res.setdefault("fp16_vs_fp32_max_abs_diff", {})[plat] = float(
            max(np.abs(s16[c] - s32[c]).max() for c in dc.PAIRS))
        res.setdefault("throughput_per_s_fp32", {})[plat] = round(len(d) / dt, 1)

        agree = {}
        for c, p in dc.PAIRS.items():
            a, b = d[p].to_numpy(float), d[f"dx_{c}"].to_numpy(float)
            ok = ~(np.isnan(a) | np.isnan(b))
            a, b = a[ok], b[ok]
            rho, _ = stats.spearmanr(a, b)
            r, _ = stats.pearsonr(a, b)
            pa, pb = a >= T, b >= T
            entry = {"spearman": round(float(rho), 3), "pearson": round(float(r), 3),
                     "mean_perspective": round(float(a.mean()), 4),
                     "mean_detoxify": round(float(b.mean()), 4),
                     "n_flag_perspective": int(pa.sum()), "n_flag_detoxify": int(pb.sum()),
                     "n_flag_both": int((pa & pb).sum())}
            if pa.any() and (~pa).any():
                entry["kappa_at_0.5"] = round(float(cohen_kappa_score(pa, pb)), 3)
                entry["auc_detox_vs_persp_flag"] = round(float(roc_auc_score(pa, b)), 3)
            agree[c] = entry
        res.setdefault("agreement", {})[plat] = agree
        res.setdefault("joint_perspective", {})[plat] = joint(d, "TOXICITY")
        res.setdefault("joint_detoxify", {})[plat] = joint(d, "dx_toxicity")
        # scores only - no comment text or usernames leave the gitignored data
        keep = (["index", "Label", "vader_compound"] + list(dc.PAIRS.values())
                + [f"dx_{c}" for c in dc.PAIRS])
        d[keep].to_csv(OUT / f"detoxify_vs_perspective_{plat}.csv", index=False)

    with open(OUT / "detoxify_vs_perspective.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
