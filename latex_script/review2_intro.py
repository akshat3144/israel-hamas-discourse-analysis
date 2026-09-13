# -*- coding: utf-8 -*-
"""
Reviewer round 2: the introduction opened directly on the paper's own argument.
Add two paragraphs of context - what the conflict was and what it did to comment
sections - before the platform-comparison motivation.

Run:  python latex_script/review2_intro.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
BODY = HERE / "body_new.tex"
NL = "\r\n"

ANCHOR = "\\subsection{Research Motivation}"

NEW = NL.join([
    "\\subsection{Research Motivation}",
    "The Hamas-led attack on southern Israel of 7~October~2023 and the Israeli military",
    "campaign in Gaza that followed became one of the most heavily discussed",
    "geopolitical events of the social-media era. Ng and Chow~\\cite{ngchow2025} index",
    "roughly 1.08~billion English-language posts across Twitter/X, Reddit and TikTok in",
    "the first four months alone, 59\\% of them negative in sentiment, and identify eight",
    "distinct volume peaks driven both by battlefield events and by offline protest and",
    "boycott movements. Public argument about the conflict was therefore not a marginal",
    "accompaniment to it; for most audiences outside the region it was the primary",
    "channel through which the conflict was encountered.",
    "",
    "That argument also changed the character of the forums carrying it. Comparing the",
    "three days either side of 7~October, the Institute for Strategic Dialogue recorded",
    "a more than fiftyfold rise in the absolute volume of antisemitic comments, and a",
    "comparable rise in anti-Muslim comments, beneath YouTube videos about the",
    "conflict~\\cite{isd2023}; the same period saw sustained pressure on platform",
    "moderation systems and a marked hardening of tone in general-audience comment",
    "sections. Comment threads that ordinarily host reaction became a venue for",
    "sustained partisan dispute, which is precisely the material this paper analyses:",
    "not what participants believe, but how the argument is organised once a platform's",
    "design decides who can reply to whom, at what length, and with what visibility.",
    "",
])


def main():
    with BODY.open("r", encoding="utf-8", newline="") as fh:
        raw = fh.read()
    if "ngchow2025" in raw:
        print("intro context already present; nothing to do")
        return
    assert raw.count(ANCHOR) == 1, "anchor not unique"
    raw = raw.replace(ANCHOR, NEW, 1)
    with BODY.open("w", encoding="utf-8", newline="") as fh:
        fh.write(raw)
    print("introduction expanded")


if __name__ == "__main__":
    main()
