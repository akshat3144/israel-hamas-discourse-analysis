# -*- coding: utf-8 -*-
"""
Does the headline platform contrast survive the measured label-quality gap?

Human validation shows the stance labels are much better on Reddit (83.6%
accurate on High-tier, non-irrelevant items) than on YouTube (61.8%). The
paper's central claim is that stance and sentiment are strongly associated on
Reddit (Cramer's V = 0.149) and barely at all on YouTube (0.039). Label noise
attenuates association, so the obvious objection is that the contrast merely
reflects the contrast in label quality.

This tests it directly: degrade Reddit's labels to YouTube-grade accuracy, using
the error structure actually observed on YouTube, and recompute V. If Reddit's
association survives the degradation it cannot be an artifact of label quality;
if it collapses to YouTube's level, the finding is not safe.

Run:  python 06_revision/label_noise_sensitivity.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
STANCES = ["P", "I", "N"]
N_SIMS = 200
SEED = 20240507

# High-tier, P/I/N-gold confusion measured on the YouTube half of the validation
# sample: rows = human gold (P, I, N), cols = model label. n = 76, accuracy 0.618.
# This is the error process we transplant onto Reddit.
YT_CONF = np.array([
    [18.0,  5.0,  5.0],   # gold P -> model P, I, N
    [ 3.0, 13.0,  3.0],   # gold I
    [ 4.0,  9.0, 16.0],   # gold N
])
REDDIT_ACC = 0.8356   # n = 73
YOUTUBE_ACC = 0.6184  # n = 76


def cramers_v(a, b):
    tab = pd.crosstab(a, b).to_numpy(dtype=float)
    n = tab.sum()
    if n == 0:
        return np.nan
    exp = np.outer(tab.sum(1), tab.sum(0)) / n
    chi2 = float(((tab - exp) ** 2 / np.where(exp == 0, np.nan, exp)).sum())
    r, k = tab.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))


def error_distribution(conf):
    """P(model label | gold) restricted to the error cases, per gold class."""
    out = {}
    for i, s in enumerate(STANCES):
        row = conf[i].copy()
        row[i] = 0.0                    # errors only
        out[s] = row / row.sum()
    return out


def main():
    rng = np.random.default_rng(SEED)
    err = error_distribution(YT_CONF)
    # fraction of Reddit labels to re-draw so accuracy falls to YouTube's level
    q = 1.0 - (YOUTUBE_ACC / REDDIT_ACC)
    print(f"Reddit accuracy {REDDIT_ACC:.3f} -> YouTube {YOUTUBE_ACC:.3f}")
    print(f"corrupting {100*q:.1f}% of Reddit labels with YouTube's error structure\n")

    red = pd.read_parquet(OUT / "reddit_window_clean.parquet",
                          columns=["Label", "vader_label"])
    red = red[red["Label"].isin(STANCES)].reset_index(drop=True)
    yt = pd.read_parquet(OUT / "youtube_window_clean.parquet",
                         columns=["Label", "vader_label"])
    yt = yt[yt["Label"].isin(STANCES)].reset_index(drop=True)

    v_red = cramers_v(red["Label"], red["vader_label"])
    v_yt = cramers_v(yt["Label"], yt["vader_label"])
    print(f"observed  Reddit V = {v_red:.4f}   YouTube V = {v_yt:.4f}")

    labels = red["Label"].to_numpy()
    sent = red["vader_label"].to_numpy()
    n = len(labels)
    idx = {s: np.flatnonzero(labels == s) for s in STANCES}

    sims = []
    for _ in range(N_SIMS):
        noisy = labels.copy()
        hit = rng.random(n) < q
        for s in STANCES:
            rows = idx[s]
            rows = rows[hit[rows]]
            if rows.size:
                noisy[rows] = rng.choice(STANCES, size=rows.size, p=err[s])
        sims.append(cramers_v(pd.Series(noisy), pd.Series(sent)))

    sims = np.array(sims)
    lo, hi = np.percentile(sims, [2.5, 97.5])
    print(f"\nReddit V after YouTube-grade label noise: "
          f"{sims.mean():.4f}  [{lo:.4f}, {hi:.4f}]  ({N_SIMS} sims)")
    print(f"  retains {100*sims.mean()/v_red:.0f}% of the observed association")
    print(f"  ratio to YouTube's observed V: {sims.mean()/v_yt:.2f}x")

    verdict = ("survives" if lo > v_yt else "NOT robust")
    print(f"\n-> degraded Reddit association is {verdict} relative to YouTube")

    res = {
        "corruption_rate": round(float(q), 4),
        "observed": {"reddit_v": round(v_red, 4), "youtube_v": round(v_yt, 4)},
        "reddit_degraded_v": {
            "mean": round(float(sims.mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "n_sims": N_SIMS,
        },
        "retained_pct": round(100 * float(sims.mean()) / v_red, 1),
        "ratio_to_youtube": round(float(sims.mean()) / v_yt, 2),
        "robust": bool(lo > v_yt),
    }
    (OUT / "label_noise_sensitivity.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'label_noise_sensitivity.json'}")


if __name__ == "__main__":
    main()
