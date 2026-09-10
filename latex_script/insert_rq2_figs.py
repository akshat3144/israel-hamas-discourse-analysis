# -*- coding: utf-8 -*-
"""Insert the temporal figure into Sec. III and the two RQ2 figures into rq2_body."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
BS = chr(92)


def fig_block(path, caption, label, full=False):
    env = "figure*" if full else "figure"
    width = BS + "textwidth" if full else BS + "linewidth"
    pos = "[t]" if full else "[H]"
    return (f"{BS}begin{{{env}}}{pos}\n"
            f"  {BS}centering\n"
            f"  {BS}includegraphics[width={width}]{{{path}}}\n"
            f"  {BS}caption{{{caption}}}\n"
            f"  {BS}label{{{label}}}\n"
            f"{BS}end{{{env}}}\n\n")


# ---- 1. temporal figure into body_new.tex (Sec III, matched window)
b = HERE / "body_new.tex"
t = b.read_text(encoding="utf-8")
anchor = "comments (3.81" + BS + "%) and leaving YouTube unchanged."
assert anchor in t, "matched-window anchor not found"
t = t.replace(anchor, anchor + "\n\n" + fig_block(
    "f11_temporal.png",
    "Monthly comment volume over the matched window. Reddit is anchored on the "
    "comment timestamp, YouTube on the video publish date.",
    "fig:temporal"), 1)
b.write_text(t, encoding="utf-8")
print("inserted temporal figure into body_new.tex")

# ---- 2. RQ2 figures
r = HERE / "rq2_body.tex"
s = r.read_text(encoding="utf-8")

a1 = ("Pro-Palestine by " + BS + "textit{free, allah, land} and Pro-Israel\n"
      "by " + BS + "textit{god, bless, idf, stand}.")
assert a1 in s, "vocabulary anchor not found"
s = s.replace(a1, a1 + "\n\n" + fig_block(
    "f09_top_words.png",
    "Most frequent terms per platform after removing shared conflict terms, "
    "contraction fragments and discourse filler.",
    "fig:topwords", full=True), 1)

a2 = BS + "subsubsection*{Topic-model quality}"
assert a2 in s, "topic-quality anchor not found"
s = s.replace(a2, fig_block(
    "f10_wordclouds.png",
    "Stance vocabularies by platform. Reddit divides along legal and territorial "
    "terms; YouTube along devotional and solidaristic ones.",
    "fig:wordclouds", full=True) + a2, 1)
r.write_text(s, encoding="utf-8")
print("inserted 2 RQ2 figures into rq2_body.tex")
