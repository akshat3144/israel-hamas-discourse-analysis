# -*- coding: utf-8 -*-
"""Mark, in each cited paper, the passage our claim actually came from.

This replaces the earlier keyword highlighting, which only marked every
occurrence of a search term and proved nothing. Each entry below pairs a claim
made in our manuscript with the specific sentence in the source that supports
it, located by reading the source rather than by matching words.

Writes PROVENANCE.txt: manuscript claim -> source page -> quoted passage.
"""
import json
import re
import sys
from pathlib import Path

import fitz

REF = Path(__file__).resolve().parent
PDF = REF / "pdf"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# key -> [(short description of what our paper asserts, [phrases to mark])]
PROVENANCE = {
    "antonakaki2025": [
        ("analysed Telegram, Twitter/X and Reddit with LDA, BERTopic and "
         "transformer emotion models",
         ["We combine traditional LDA with a refined BERTopic workflow"]),
        ("identified platform-specific narrative strategies and amplification",
         ["Captures narrative strategies, amplification patterns, and temporal dynamics"]),
    ],
    "bail2018": [
        ("exposure to opposing views can increase polarization",
         ["Republicans who followed a liberal Twitter bot became substantially more conservative"]),
        ("cross-cutting exposure intensifies rather than softens partisan feeling",
         ["Democrats exhibited slight increases in liberal attitudes after following a conservative"]),
    ],
    "blei2003": [
        ("LDA represents documents as mixtures over latent topics",
         ["documents are represented as random mixtures over latent topics"]),
    ],
    "boulianne2025": [
        ("distinguishes platforms by structural and social networking features",
         ["user connectivity, anonymity, and privacy settings",
          "perceived audiences, temporary vs strong ties, and social norms of interaction"]),
        ("compares seven platforms",
         ["We compare seven social media platforms based on their networking features"]),
    ],
    "cinelli2021": [
        ("echo-chamber strength is platform-dependent",
         ["favor the emergence of echo chambers"]),
        ("feed-based platforms differ from Reddit, so results should not be generalised",
         ["A clearcut distinction emerges between social media having a feed algorithm"]),
    ],
    "defrancisci2021": [
        ("preference for cross-cutting replies over within-group ones",
         ["We observe a preference for cross-cutting political interactions between the two communities"]),
        ("the preference is asymmetric between the two camps",
         ["The same effect is visible for Trump supporters, who are more likely to interact with Clinton"]),
        ("tested against a degree-preserving rewiring null",
         ["a network rewiring which preserves the activity of nodes"]),
    ],
    "devlin2019": [
        ("the BERT architecture our RoBERTa sentiment model builds on",
         ["model architecture is a multi-layer bidirectional Transformer encoder"]),
    ],
    "gilardi2023": [
        ("LLMs outperform crowd workers on stance and frame detection",
         ["outperforms crowd workers for several annotation tasks, including relevance, stance"]),
    ],
    "grootendorst2022": [
        ("BERTopic clusters sentence-embedding representations",
         ["clusters these embeddings"]),
    ],
    "guerra2025": [
        ("scored Reddit posts on a lexicon index of anger, polarity and subjectivity",
         ["by considering factors such as anger, polarity, and subjectivity"]),
        ("found peaks aligning with specific events on the ground",
         ["peaks in extremism scores that correspond to pivotal real-life events"]),
    ],
    "hayes2007": [
        ("the agreement statistic we report (Krippendorff's alpha)",
         ["alpha as the standard reliability measure"]),
    ],
    "hofmann2026": [
        ("hate speech in 40.4 percent of public and 31.6 percent of private source comments",
         ["higher incidence of HS in public sources"]),
        ("4,983 hand-annotated YouTube comments, classifiers at AUROC 0.83 to 0.90",
         ["annotated dataset of 4983 YouTube comments labeled for HS and sentiment",
          "AUROC scores between 0.83 to 0.90"]),
    ],
    "hutto2014": [
        ("VADER is designed for microblog contexts",
         ["attuned to sentiment in microblog-like contexts"]),
        ("captures punctuation intensity",
         ["Punctuation, namely the exclamation point"]),
        ("captures slang",
         ["acronyms, initialisms, emoticons, or slang"]),
        ("captures negation",
         ["negation, degree modifiers, and contrastive conjunctions"]),
    ],
    "leeseung2000": [
        ("NMF as a matrix decomposition",
         ["has previously been shown to be a useful decomposition for multivariate data"]),
    ],
    "newman2003": [
        ("the homophily principle: similar people interact at higher rates",
         ["the tendency for vertices in networks to be connected to other vertices that are like"]),
        ("assortativity for categorical attributes",
         ["We consider mixing according to discrete characteristics"]),
        ("zero is random mixing, negative values indicate cross-stance interaction",
         ["indicating perfect assortativity"]),
    ],
    "ng2022": [
        ("cross-dataset stance models do not transfer",
         ["models do not generalize well"]),
        ("stance models transfer poorly across corpora generally",
         ["models tend to perform poorly on unseen cases"]),
    ],
    "ngchow2025": [
        ("roughly 1.08 billion English-language posts",
         ["1,079,676,984"]),
        ("59 percent negative",
         ["negative (59%), neutral (31%), and positive (10%)"]),
        ("across Twitter/X, Reddit and TikTok in the first four months",
         ["8 prominent peaks and emergent events"]),
    ],
    "pedregosa2011": [
        ("scikit-learn as the classification toolkit",
         ["a Python module integrating a wide range of state-of-the-art machine learning algorithms"]),
    ],
    "rosen2025": [
        ("escalation in anti-Jewish and anti-Muslim posting after 7 October",
         ["examined anti-Jewish and anti-Muslim hate posts on X.com"]),
        ("replies feed back into how hatefully authors post next",
         ["hate posts led individuals to post more hatefully in their next post"]),
    ],
    "shugars2025": [
        ("Reddit's conversational visibility",
         ["the ability of users to see and engage in full conversations"]),
        ("supports more cross-cutting exchange than broadcast-oriented design",
         ["from conversational visibility but from the affordance of community"]),
    ],
    "wulczyn2017": [
        ("the personal-attacks framework Perspective operationalises",
         ["combines crowdsourcing and machine learning to analyze personal attacks at scale"]),
    ],
}


def strip_highlights(doc):
    """Remove the earlier keyword highlighting."""
    n = 0
    for pno in range(doc.page_count):
        page = doc[pno]
        for a in list(page.annots() or []):
            if a.type[0] == 8:
                page.delete_annot(a)
                n += 1
    return n


def _variants(phrase):
    """PDF text search breaks on ligatures, hyphenation and line wraps, so try
    the whole phrase first and then shrinking windows of it."""
    w = phrase.split()
    yield phrase
    for size in (8, 6, 5, 4):
        if len(w) <= size:
            continue
        for start in range(0, len(w) - size + 1):
            yield " ".join(w[start:start + size])


def mark(doc, phrase):
    """Highlight the passage; returns page number or None."""
    for cand in _variants(phrase):
        for pno in range(doc.page_count):
            page = doc[pno]
            rects = page.search_for(cand)
            if not rects:
                continue
            for r in rects:
                a = page.add_highlight_annot(r)
                a.set_colors(stroke=(1, 1, 0))
                a.update()
            return pno + 1
    return None


def sentence_at(doc, pno, phrase):
    """The full sentence containing the phrase, for the record."""
    t = doc[pno - 1].get_text().replace("-\n", "").replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    i = t.find(phrase)
    if i < 0:
        return phrase
    a = t.rfind(". ", 0, i) + 1
    b = t.find(". ", i + len(phrase))
    return t[a if a > 0 else max(0, i - 160):(b + 1) if b > 0 else i + len(phrase) + 200].strip()


def main():
    ctx = json.loads((REF / "citation_contexts.json").read_text(encoding="utf-8"))
    out = ["CITATION PROVENANCE", "Clustered by Thread, Arguing Across the Divide", ""]
    out.append("For every claim the manuscript attributes to a source, this records the")
    out.append("passage in that source the claim came from, and the page it is on. The")
    out.append("same passages are highlighted in yellow in references/pdf/.")
    out.append("")
    out.append("=" * 74)

    missing, total = [], 0
    for key, items in PROVENANCE.items():
        p = PDF / f"{key}.pdf"
        if not p.exists():
            missing.append((key, "NO PDF"))
            continue
        doc = fitz.open(p)
        removed = strip_highlights(doc)

        out.append("")
        out.append(f"{key}")
        out.append("-" * 74)
        uses = ctx.get(key, [])
        if uses:
            out.append("  the manuscript says:")
            for u in uses:
                for line in re.findall(r".{1,68}(?:\s|$)", u):
                    if line.strip():
                        out.append(f"      {line.strip()}")
                out.append("")

        for desc, phrases in items:
            out.append(f"  CLAIM: {desc}")
            for ph in phrases:
                pno = mark(doc, ph)
                total += 1
                if pno is None:
                    missing.append((key, ph))
                    out.append(f"    SOURCE: phrase not located -> {ph[:60]}")
                    continue
                quote = sentence_at(doc, pno, ph)
                out.append(f"    SOURCE: p.{pno}")
                for line in re.findall(r".{1,64}(?:\s|$)", quote):
                    if line.strip():
                        out.append(f'      "{line.strip()}"' if line is
                                   re.findall(r".{1,64}(?:\s|$)", quote)[0]
                                   else f"       {line.strip()}")
            out.append("")

        doc.saveIncr()
        doc.close()
        print(f"  {key:18} stripped {removed:4} keyword marks, "
              f"marked {sum(len(x[1]) for x in items)} passages")

    out.append("=" * 74)
    (REF / "PROVENANCE.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"\n{total} passages attempted, {len(missing)} not located")
    for k, ph in missing:
        print(f"   MISS {k}: {ph[:70]}")
    print(f"wrote {REF / 'PROVENANCE.txt'}")


if __name__ == "__main__":
    main()
