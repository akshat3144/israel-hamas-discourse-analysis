# -*- coding: utf-8 -*-
"""
Regenerate the three RQ1 figures changed by the tone/toxicity revision, in the
same style and at the same printed sizes as make_figs.py:

  f02  tone by stance        - twitter-roberta (primary tone measure), full corpus
  f03  instrument agreement  - full-corpus kappa + accuracy against human labels
  f06  negativity x toxicity - Detoxify on the full corpus, RoBERTa and VADER

Run:  python 06_revision/make_figs_revision.py
Out:  06_revision/figs/f02_*.png, f03_*.png, f06_*.png  (then copied to
      latex_script/paperfigs/)
"""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
FIGS = HERE / "figs"
PAPER = HERE.parent / "latex_script" / "paperfigs"

COL, FULL = 3.4, 7.0
DPI = 300
STANCES = ["P", "I", "N"]
SHORT = {"P": "Pro-Pal", "I": "Pro-Isr", "N": "Neutral"}
COLORS = {"P": "#2ecc71", "I": "#3498db", "N": "#95a5a6"}
PLAT = {"reddit": "#3498db", "youtube": "#e74c3c"}
PNAME = {"reddit": "Reddit", "youtube": "YouTube"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3,
})

tone = json.load(open(OUT / "rq1_tone_roberta.json", encoding="utf-8"))
agree = json.load(open(OUT / "sentiment_agreement_full.json", encoding="utf-8"))
val = json.load(open(OUT / "sentiment_validation.json", encoding="utf-8"))
tox = json.load(open(OUT / "toxicity_full.json", encoding="utf-8"))


def save(fig, name):
    fig.savefig(FIGS / name)
    plt.close(fig)
    shutil.copy2(FIGS / name, PAPER / name)
    print("  wrote", name)


# ---------------------------------------------------------------- 2. tone by stance
fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4))
fig.subplots_adjust(wspace=0.30)
for ax, plat in zip(axes, ["reddit", "youtube"]):
    st = tone["stance"][plat]
    m = st["means"]
    ax.bar([SHORT[s] for s in STANCES], [m[s] for s in STANCES],
           color=[COLORS[s] for s in STANCES], edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"{PNAME[plat]}  (Cramer's V = {st['cramers_v']:.3f})")
    ax.set_ylabel("mean RoBERTa score")
    ax.set_ylim(-0.7, 0.1)
fig.suptitle("Tone by stance: strongly coupled on Reddit, weakly on YouTube", y=1.04)
save(fig, "f02_sentiment_by_stance.png")

# ---------------------------------------------------------------- 3. instrument agreement
fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4))
fig.subplots_adjust(wspace=0.30)
pairs = ["vader_vs_roberta", "textblob_vs_roberta", "vader_vs_textblob"]
lbl = ["VADER-\nRoBERTa", "TextBlob-\nRoBERTa", "VADER-\nTextBlob"]
xx = np.arange(3)
for i, plat in enumerate(["reddit", "youtube"]):
    axes[0].bar(xx + (i - 0.5) * 0.38, [agree[plat]["kappa"][p] for p in pairs], 0.38,
                label=PNAME[plat], color=PLAT[plat], edgecolor="black", linewidth=0.4)
axes[0].set_xticks(xx); axes[0].set_xticklabels(lbl)
axes[0].set_ylabel("Cohen's kappa"); axes[0].set_title("Inter-method agreement (full corpus)")
axes[0].legend(frameon=False)
tools = ["roberta", "vader", "textblob"]
names = ["RoBERTa", "VADER", "TextBlob"]
for i, plat in enumerate(["reddit", "youtube"]):
    acc = [val["instrument_vs_human"][plat][t]["accuracy"] for t in tools]
    axes[1].bar(xx + (i - 0.5) * 0.38, acc, 0.38, label=PNAME[plat],
                color=PLAT[plat], edgecolor="black", linewidth=0.4)
axes[1].axhline(1 / 3, color="black", lw=0.8, ls=":")
axes[1].set_xticks(xx); axes[1].set_xticklabels(names)
axes[1].set_ylim(0, 1); axes[1].set_ylabel("accuracy vs human labels")
axes[1].set_title(f"Agreement with humans (n = {val['gold']['n_gold']})")
axes[1].legend(frameon=False)
save(fig, "f03_consensus.png")

# ---------------------------------------------------------------- 6. negativity x toxicity
fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4))
fig.subplots_adjust(wspace=0.30)
for ax, plat in zip(axes, ["reddit", "youtube"]):
    j = tox["joint"][plat]
    for k, (inst, name, col) in enumerate([("roberta", "RoBERTa", "#c0392b"),
                                           ("vader", "VADER", "#e6a19a")]):
        vals = [j[inst]["pct_toxic_that_are_negative"], j[inst]["pct_negative_that_are_toxic"]]
        pos = np.arange(2) + (k - 0.5) * 0.38
        ax.bar(pos, vals, 0.38, color=col, edgecolor="black", linewidth=0.4, label=name)
        for x, v in zip(pos, vals):
            ax.text(x, v + 1.5, f"{v:.0f}%", ha="center", fontsize=7, fontweight="bold")
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(["toxic that are\nalso negative", "negative that are\nalso toxic"])
    ax.set_ylim(0, 108); ax.set_ylabel("%")
    ax.set_title(f"{PNAME[plat]} (Spearman rho = {j['spearman_roberta_toxicity']:+.2f})")
    ax.legend(frameon=False, loc="upper right")
fig.suptitle("Toxicity is a subset of negativity, not a synonym (Detoxify, full corpus)", y=1.04)
save(fig, "f06_joint_toxicity.png")
