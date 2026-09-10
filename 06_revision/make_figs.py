# -*- coding: utf-8 -*-
"""
Stage 9: regenerate paper figures (reviewer R12 - unreadable labels).

Key fix: figures are generated at their FINAL printed size (3.4in for a single
column, 7.0in for a full-width figure*) so LaTeX does not downscale them and
shrink the text. Dense multi-panel figures are laid out over two rows.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)

COL, FULL = 3.4, 7.0      # inches: single column / full width
DPI = 300
STANCES = ["P", "I", "N"]
NAMES = {"P": "Pro-Palestine", "I": "Pro-Israel", "N": "Neutral"}
SHORT = {"P": "Pro-Pal", "I": "Pro-Isr", "N": "Neutral"}
COLORS = {"P": "#2ecc71", "I": "#3498db", "N": "#95a5a6"}
PLAT = {"reddit": "#3498db", "youtube": "#e74c3c"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3,
})

bf = json.load(open(OUT / "botfree_stats.json", encoding="utf-8"))
core = json.load(open(OUT / "core_stats.json", encoding="utf-8"))
net = json.load(open(OUT / "network_homophily.json", encoding="utf-8"))
tox = json.load(open(OUT / "toxicity_joint.json", encoding="utf-8"))
ml = json.load(open(OUT / "ml_stance.json", encoding="utf-8"))
cons = json.load(open(OUT / "consensus_sentiment.json", encoding="utf-8"))


def save(fig, name):
    fig.savefig(FIGS / name)
    plt.close(fig)
    print("  wrote", name)


# ---------------------------------------------------------------- 1. stance + CI (R8)
fig, ax = plt.subplots(figsize=(COL, 2.3))
w = 0.38
x = np.arange(3)
for i, (plat, off) in enumerate([("reddit", -w / 2), ("youtube", w / 2)]):
    d = bf["stance"][plat]
    pct = [d[s]["pct"] for s in STANCES]
    err = [[d[s]["pct"] - d[s]["lo"] for s in STANCES],
           [d[s]["hi"] - d[s]["pct"] for s in STANCES]]
    ax.bar(x + off, pct, w, yerr=err, capsize=2.5, label=plat.capitalize(),
           color=PLAT[plat], edgecolor="black", linewidth=0.4)
ax.set_xticks(x); ax.set_xticklabels([SHORT[s] for s in STANCES])
ax.set_ylabel("% of comments"); ax.set_title("Stance distribution (95% bootstrap CI)")
ax.legend(frameon=False)
save(fig, "f01_stance_ci.png")

# ---------------------------------------------------------------- 2. sentiment (R1/R11)
fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4))
fig.subplots_adjust(wspace=0.30)
d = bf["sentiment"]["by_stance"]
for ax, plat in zip(axes, ["reddit", "youtube"]):
    m = d[plat]["means"]
    ax.bar([SHORT[s] for s in STANCES], [m[s] for s in STANCES],
           color=[COLORS[s] for s in STANCES], edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"{plat.capitalize()}  (Cramer's V = {d[plat]['cramers_v']:.3f})")
    ax.set_ylabel("mean VADER compound")
    ax.set_ylim(-0.28, 0.16)
fig.suptitle("Sentiment by stance: politicised on Reddit, decoupled on YouTube", y=1.04)
save(fig, "f02_sentiment_by_stance.png")

# ---------------------------------------------------------------- 3. consensus (R11)
fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4))
k = cons["kappa"]
pairs = ["vader_vs_roberta", "textblob_vs_roberta", "vader_vs_textblob"]
lbl = ["VADER-\nRoBERTa", "TextBlob-\nRoBERTa", "VADER-\nTextBlob"]
xx = np.arange(3)
for i, plat in enumerate(["reddit", "youtube"]):
    axes[0].bar(xx + (i - 0.5) * 0.38, [k[plat][p] for p in pairs], 0.38,
                label=plat.capitalize(), color=PLAT[plat], edgecolor="black", linewidth=0.4)
axes[0].set_xticks(xx); axes[0].set_xticklabels(lbl)
axes[0].set_ylabel("Cohen's kappa"); axes[0].set_title("Inter-method agreement is modest")
axes[0].legend(frameon=False)
cp = cons["consensus_platform"]
axes[1].bar(["Reddit", "YouTube"], [cp["reddit_pct_negative"], cp["youtube_pct_negative"]],
            color=[PLAT["reddit"], PLAT["youtube"]], edgecolor="black", linewidth=0.4)
axes[1].set_ylabel("% negative")
axes[1].set_title(f"Unanimous subset ({cons['consensus']['reddit']['coverage_pct']:.0f}% / "
                  f"{cons['consensus']['youtube']['coverage_pct']:.0f}% of sample)")
save(fig, "f03_consensus.png")

# ---------------------------------------------------------------- 4. reply mixing (R18) KEY
fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6))
fig.subplots_adjust(wspace=0.55)
mix = pd.DataFrame(net["mixing_row_pct"]).reindex(index=STANCES, columns=STANCES)
im = axes[0].imshow(mix.to_numpy(dtype=float), cmap="RdYlBu_r", vmin=0, vmax=70)
axes[0].set_xticks(range(3)); axes[0].set_xticklabels([SHORT[s] for s in STANCES])
axes[0].set_yticks(range(3)); axes[0].set_yticklabels([SHORT[s] for s in STANCES])
axes[0].set_xlabel("replied TO"); axes[0].set_ylabel("replier")
axes[0].set_title("Reply mixing matrix (row %)")
axes[0].grid(False)
for i in range(3):
    for j in range(3):
        axes[0].text(j, i, f"{mix.iloc[i, j]:.1f}", ha="center", va="center", fontsize=8,
                     color="white" if mix.iloc[i, j] > 45 else "black")
fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

ps = net["per_stance"]
xx = np.arange(3)
axes[1].bar(xx - 0.19, [ps[s]["observed"] for s in STANCES], 0.38, label="observed",
            color="#34495e", edgecolor="black", linewidth=0.4)
axes[1].bar(xx + 0.19, [ps[s]["expected"] for s in STANCES], 0.38, label="expected (random)",
            color="#bdc3c7", edgecolor="black", linewidth=0.4)
axes[1].set_xticks(xx); axes[1].set_xticklabels([SHORT[s] for s in STANCES])
axes[1].set_ylabel("same-stance reply rate")
axes[1].set_title(f"Below chance for partisans (Newman r = {net['newman_assortativity']:+.3f})")
axes[1].legend(frameon=False)
save(fig, "f04_reply_mixing.png")

# ---------------------------------------------------------------- 5. thread vs reply homophily
fig, ax = plt.subplots(figsize=(COL, 2.3))
th = core["polarization"]["thread"]["by_stance"]
rp = core["polarization"]["reply_network"]["by_stance"]
xx = np.arange(3)
ax.bar(xx - 0.19, [th[s] for s in STANCES], 0.38, label="thread co-membership",
       color="#9b59b6", edgecolor="black", linewidth=0.4)
ax.bar(xx + 0.19, [rp[s] for s in STANCES], 0.38, label="reply network",
       color="#e67e22", edgecolor="black", linewidth=0.4)
ax.set_xticks(xx); ax.set_xticklabels([SHORT[s] for s in STANCES])
ax.set_ylabel("homophily index")
ax.set_ylim(0, max(max(th.values()), max(rp.values())) * 1.45)
ax.set_title("Co-presence vs interaction give\ndifferent answers")
ax.legend(frameon=False, loc="upper right")
save(fig, "f05_homophily_compare.png")

# ---------------------------------------------------------------- 6. joint sentiment x toxicity (R2)
fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4))
j = tox["joint"]
for ax, plat in zip(axes, ["reddit", "youtube"]):
    v = j[plat]
    vals = [v["pct_of_toxic_that_are_negative"], v["pct_of_negative_that_are_toxic"]]
    ax.bar(["toxic that are\nalso negative", "negative that are\nalso toxic"], vals,
           color=["#c0392b", "#7f8c8d"], edgecolor="black", linewidth=0.4)
    for i, val in enumerate(vals):
        ax.text(i, val + 1.5, f"{val:.1f}%", ha="center", fontsize=8, fontweight="bold")
    ax.set_ylim(0, 100); ax.set_ylabel("%")
    ax.set_title(f"{plat.capitalize()} (Spearman rho = {v['spearman_rho']:+.2f})")
fig.suptitle("Toxicity is a small subset of negativity, not a synonym", y=1.04)
save(fig, "f06_joint_toxicity.png")

# ---------------------------------------------------------------- 7. transfer gap (R19)
fig, ax = plt.subplots(figsize=(COL, 2.3))
keys = ["R->R", "Y->Y", "R->Y", "Y->R"]
raw = [ml["raw"][k]["macro_f1"] for k in keys]
mat = [ml["length_matched"][k]["macro_f1"] for k in keys]
xx = np.arange(4)
ax.bar(xx - 0.19, raw, 0.38, label="raw", color="#2c3e50", edgecolor="black", linewidth=0.4)
ax.bar(xx + 0.19, mat, 0.38, label="length-matched", color="#16a085", edgecolor="black", linewidth=0.4)
ax.set_xticks(xx); ax.set_xticklabels(keys)
ax.set_ylabel("macro-F1")
ax.set_title("Transfer gap survives length matching")
ax.legend(frameon=False)
save(fig, "f07_transfer.png")

# ---------------------------------------------------------------- 8. replies vs controversiality (R16)
fig, ax = plt.subplots(figsize=(COL, 2.3))
e = core["engagement"]
ax2 = ax.twinx()
ax.bar([0, 1], [e["controversiality_sanity"]["non_controversial_mean_score"],
                e["controversiality_sanity"]["controversial_mean_score"]],
       0.5, color="#bdc3c7", edgecolor="black", linewidth=0.4, label="score (vote-derived)")
ax2.plot([0, 1], [e["replies_by_controversiality"]["0"], e["replies_by_controversiality"]["1"]],
         "o-", color="#c0392b", lw=2, ms=6, label="replies received")
ax.set_xticks([0, 1]); ax.set_xticklabels(["non-controversial", "controversial"])
ax.set_ylabel("mean score"); ax2.set_ylabel("mean replies", color="#c0392b")
ax2.tick_params(axis="y", colors="#c0392b"); ax2.grid(False)
ax.set_title("Controversy: penalised in score,\nrewarded in replies")
save(fig, "f08_controversy.png")

print(f"\nAll figures -> {FIGS}")
