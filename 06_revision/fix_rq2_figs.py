# -*- coding: utf-8 -*-
"""
Replace the RQ2 figure blocks (f09 top words, f10 word clouds) in make_figs.py.

Problem being fixed: stripping apostrophes turned "don't" into the tokens
"don" + "t", so contraction debris (t, s, re, u, m, isn, didn) dominated the word
clouds and filler words (dont, im, thats, youre) dominated the frequency chart.
Fix: delete apostrophes WITHOUT inserting a space, enforce a minimum word length,
and add a contraction/filler stoplist.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
p = HERE / "make_figs.py"
t = p.read_text(encoding="utf-8")

marker = "# ---------------------------------------------------------------- 9. top words (RQ2)"
idx = t.index(marker)
head = t[:idx]

new = '''# ---------------------------------------------------------------- shared RQ2 setup
import re as _re
from collections import Counter as _Counter
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer as _CV

# Domain terms shared by both platforms (uninformative for contrast), plus
# contraction debris and discourse filler that would otherwise dominate.
_DOMAIN = ["israel", "israeli", "israelis", "palestine", "palestinian", "palestinians",
           "hamas", "gaza", "war", "conflict"]
_FILLER = ["dont", "doesnt", "didnt", "isnt", "wasnt", "arent", "wont", "cant",
           "couldnt", "shouldnt", "wouldnt", "youre", "theyre", "thats", "whats",
           "ive", "im", "id", "ill", "youve", "theyve", "hes", "shes", "its",
           "just", "like", "people", "know", "think", "going", "said", "say",
           "saying", "really", "actually", "also", "would", "could", "one", "two",
           "even", "make", "made", "get", "got", "want", "see", "tell", "much",
           "many", "thing", "things", "way", "time", "year", "years", "day",
           "take", "took", "good", "well", "still", "back", "look", "point",
           "mean", "lot", "part", "come", "give", "need", "let", "yes", "no",
           "ye", "lol", "yeah", "oh", "ok", "okay", "sure", "maybe", "someone",
           "something", "anything", "everything", "never", "always", "now", "go"]
_SW_WC = set(STOPWORDS) | set(_DOMAIN) | set(_FILLER)
_SW_CV = sorted(set(_CV(stop_words="english").get_stop_words()) | set(_DOMAIN) | set(_FILLER))


def _norm(text):
    """Lowercase; delete apostrophes WITHOUT splitting the word; drop other punctuation."""
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = _re.sub(r"http\\S+|www\\S+", " ", t)
    t = _re.sub(r"[\\u2019\\u02bc']", "", t)      # don't -> dont  (no space inserted)
    t = _re.sub(r"[^a-z\\s]", " ", t)
    return _re.sub(r"\\s+", " ", t).strip()


_RQ2_N = 60000
_rq2 = {}
for _plat, _f in [("reddit", "reddit_window_clean.parquet"),
                  ("youtube", "youtube_window_clean.parquet")]:
    _df = pd.read_parquet(OUT / _f, columns=["Label", "text_std"])
    if len(_df) > _RQ2_N:
        _df = _df.groupby("Label", group_keys=False).sample(frac=_RQ2_N / len(_df),
                                                            random_state=42)
    _df = _df.assign(norm=_df["text_std"].map(_norm))
    _rq2[_plat] = _df[_df["norm"].str.len() > 0]

# ---------------------------------------------------------------- 9. top words (RQ2)
fig, axes = plt.subplots(1, 2, figsize=(FULL, 3.0))
fig.subplots_adjust(wspace=0.38)
for ax, plat, color in [(axes[0], "reddit", PLAT["reddit"]),
                        (axes[1], "youtube", PLAT["youtube"])]:
    vec = _CV(max_features=15, stop_words=_SW_CV, token_pattern=r"(?u)\\b[a-z]{3,}\\b")
    X = vec.fit_transform(_rq2[plat]["norm"])
    freq = dict(zip(vec.get_feature_names_out(), np.asarray(X.sum(axis=0)).ravel()))
    tw = sorted(freq.items(), key=lambda kv: kv[1])
    ax.barh([w for w, _ in tw], [c for _, c in tw], color=color,
            edgecolor="black", linewidth=0.4, alpha=0.9)
    ax.set_title(f"{plat.capitalize()}: most frequent terms", fontsize=9, fontweight="bold")
    ax.set_xlabel("frequency")
    ax.grid(axis="y", alpha=0)
fig.suptitle("Distinct vocabularies after removing shared conflict terms", y=1.02)
save(fig, "f09_top_words.png")

# ---------------------------------------------------------------- 10. word clouds (RQ2)
fig, axes = plt.subplots(2, 3, figsize=(FULL, 4.2))
for row, (plat, label) in enumerate([("reddit", "Reddit"), ("youtube", "YouTube")]):
    df = _rq2[plat]
    for col, s in enumerate(STANCES):
        sub = df[df["Label"] == s]
        text = " ".join(sub["norm"])
        ax = axes[row, col]
        if text.strip():
            wc = WordCloud(width=700, height=440, background_color="white",
                           stopwords=_SW_WC, colormap="viridis" if row == 0 else "plasma",
                           max_words=55, relative_scaling=0.5, min_font_size=8,
                           min_word_length=3, collocations=False).generate(text)
            ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"{label} - {NAMES[s]}", fontsize=8.5, fontweight="bold")
        ax.axis("off")
        ax.grid(False)
plt.tight_layout()
save(fig, "f10_wordclouds.png")

# ---------------------------------------------------------------- 11. temporal (R20)
fig, ax = plt.subplots(figsize=(COL, 2.3))
for plat, f, lbl in [("reddit", "reddit_window_clean.parquet", "Reddit (comment date)"),
                     ("youtube", "youtube_window_clean.parquet", "YouTube (video date)")]:
    df = pd.read_parquet(OUT / f, columns=["time_anchor"])
    v = df.dropna(subset=["time_anchor"]).groupby(
        pd.to_datetime(df["time_anchor"]).dt.to_period("M")).size()
    ax.plot([str(i) for i in v.index], v.values, marker="o", ms=3.5,
            color=PLAT[plat], label=lbl, lw=1.6)
ax.set_ylabel("comments / month")
ax.set_title("Matched window: Oct 2023 - May 2024")
ax.legend(frameon=False, fontsize=6.8)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
save(fig, "f11_temporal.png")
'''

p.write_text(head + new, encoding="utf-8")
print("RQ2 figure blocks replaced")
