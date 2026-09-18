# -*- coding: utf-8 -*-
"""Check the specific factual assertions the manuscript attaches to each source.

Existence is not enough. A real paper cited for something it does not say is
just as serious, so every number and every characterising phrase is searched
for in the full text we downloaded.
"""
import re
import sys
from pathlib import Path

import fitz

REF = Path(__file__).resolve().parent
PDF = REF / "pdf"
WEB = REF / "web"

# key -> list of (what the manuscript asserts, regex that should match the source)
CLAIMS = {
    "ngchow2025": [
        ("roughly 1.08 billion English-language posts", r"1\.08\s*billion|1,080,000,000|1\.08bn"),
        ("Twitter/X, Reddit and TikTok", r"TikTok"),
        ("59% negative in sentiment", r"59(\.\d+)?\s*%"),
        ("eight distinct volume peaks", r"\beight\b.{0,40}peak|peak.{0,40}\beight\b|\b8\b\s*(prominent\s*)?peaks"),
    ],
    "defrancisci2021": [
        ("preference for cross-cutting replies", r"cross-?cutting|opposite (side|party)"),
        ("degree-preserving rewiring null", r"rewir|degree[- ]preserv|configuration model"),
        ("Trump and Clinton supporters", r"Trump|Clinton"),
        ("asymmetric between the camps", r"asymmetr"),
    ],
    "antonakaki2025": [
        ("Telegram, Twitter/X and Reddit", r"Telegram"),
        ("LDA and BERTopic", r"BERTopic"),
        ("transformer-based emotion models", r"emotion"),
        ("propaganda amplification", r"propaganda"),
        ("October 2023 to mid-2025", r"202[35]"),
    ],
    "bail2018": [
        ("exposure to opposing views can increase polarization",
         r"increase[d]? .{0,40}polariz|more conservative|more liberal"),
        ("experimental design", r"experiment|randomi[sz]ed|field experiment"),
    ],
    "cinelli2021": [
        ("echo-chamber strength is platform-dependent",
         r"depend.{0,30}platform|platform.{0,30}depend|differen.{0,30}across (the )?(four )?platforms"),
        ("compares several platforms", r"Gab"),
    ],
    "boulianne2025": [
        ("threading architecture", r"thread"),
        ("character limits", r"character limit"),
        ("anonymity provisions", r"anonym"),
        ("voting mechanisms", r"vot(e|ing)"),
    ],
    "wulczyn2017": [
        ("personal attacks framework behind Perspective", r"personal attack"),
        ("rater judgements of rudeness or disrespect", r"attack|harass|toxic|aggress"),
    ],
    "gilardi2023": [
        ("LLMs outperform crowd workers", r"outperform"),
        ("stance or frame detection", r"stance|frame"),
    ],
    "hutto2014": [
        ("punctuation intensity", r"punctuation|exclamation"),
        ("slang", r"slang|acronym|emoticon"),
        ("negation", r"negation"),
    ],
    "newman2003": [
        ("assortativity coefficient for categorical attributes",
         r"discrete|enumerative|categor"),
        ("zero means random mixing", r"zero|random mixing"),
    ],
    "blei2003": [("documents as mixtures over latent topics", r"mixture")],
    "devlin2019": [("BERT architecture", r"bidirectional")],
    "grootendorst2022": [("clusters sentence-embedding representations",
                          r"cluster.{0,60}embedding|embedding.{0,60}cluster|HDBSCAN")],
    "pedregosa2011": [("scikit-learn as a classification toolkit", r"classification")],
}

WEB_CLAIMS = {
    "isd2023_1.html": [
        ("more than fiftyfold rise in antisemitic comments",
         r"5\d-fold|fifty-?fold|over 50-fold|50 fold"),
        ("three days either side of 7 October", r"three days"),
        ("YouTube", r"YouTube"),
    ],
    "isd2023_2.html": [
        ("comparable rise in anti-Muslim comments", r"43-fold|43 fold"),
        ("YouTube", r"YouTube"),
    ],
}


def pdf_text(p):
    doc = fitz.open(p)
    t = "\n".join(page.get_text() for page in doc)
    doc.close()
    return t


def html_text(p):
    raw = p.read_text(encoding="utf-8", errors="ignore")
    raw = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw, flags=re.S | re.I)
    return re.sub(r"<[^>]+>", " ", raw)


def main():
    problems = []
    print("=" * 70)
    for key, claims in CLAIMS.items():
        p = PDF / f"{key}.pdf"
        if not p.exists():
            print(f"\n{key}: NO PDF, claims unverified")
            problems.append((key, "no full text available"))
            continue
        text = pdf_text(p)
        print(f"\n{key}")
        for desc, pat in claims:
            ok = re.search(pat, text, re.I | re.S)
            print(f"   {'FOUND   ' if ok else 'NOT FOUND'} {desc}")
            if not ok:
                problems.append((key, desc))

    for fname, claims in WEB_CLAIMS.items():
        p = WEB / fname
        if not p.exists():
            problems.append((fname, "page not saved"))
            continue
        text = html_text(p)
        print(f"\n{fname}")
        for desc, pat in claims:
            ok = re.search(pat, text, re.I | re.S)
            print(f"   {'FOUND   ' if ok else 'NOT FOUND'} {desc}")
            if not ok:
                problems.append((fname, desc))

    print("\n" + "=" * 70)
    if problems:
        print(f"{len(problems)} claim(s) NOT located in the source text:")
        for k, d in problems:
            print(f"   {k:20} {d}")
    else:
        print("every checked claim located in its source")


if __name__ == "__main__":
    main()
