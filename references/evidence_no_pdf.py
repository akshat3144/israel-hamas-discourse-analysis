# -*- coding: utf-8 -*-
"""Write EVIDENCE.txt for the references with no downloadable full text.

Each one gets the verbatim publisher text that supports the claim we attach to
it, plus the exact reason no PDF is stored, so a reader can check the citation
without the paper in front of them.
"""
import sys
from pathlib import Path

REF = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EVIDENCE = {
    "rosen2025": {
        "why": ("Gold open access (free to read) but SAGE serves 403 to every "
                "automated client. Tried the PDF, EPUB, reader and download "
                "endpoints, and the DOAJ record, which only links back to the DOI."),
        "get": "Open https://doi.org/10.1177/20563051251383635 and save the PDF.",
        "claim": ("Rosen and Walther document a parallel escalation in anti-Jewish "
                  "and anti-Muslim posting in the month after 7 October, and show "
                  "that the replies such posts attract feed back into how hatefully "
                  "their authors post next."),
        "source": "publisher abstract, via Crossref",
        "quote": (
            "Online hate messaging targeting Muslims and Jews increased dramatically "
            "following Hamas's attack on Israelis on 7 October 2023 and Israel's "
            "military response in Gaza. This study examined anti-Jewish and "
            "anti-Muslim hate posts on X.com and the verbal replies, Likes, and "
            "reposts they acquired over the following month. ... Convergent replies "
            "to one's hate posts led individuals to post more hatefully in their "
            "next post, and more quickly."),
    },
    "santiago2025": {
        "why": ("Springer book chapter, subscription only. Unpaywall reports "
                "is_oa false with zero open locations, so no free copy exists "
                "anywhere."),
        "get": ("Plaksha or Tartu library access to Springer, or buy the chapter "
                "at EUR 29.95. The abstract below is free on the chapter page."),
        "claim": ("Santiago et al. focused on Reddit, examining 80 days of "
                  "conflict-related comments with ML-based sentiment and emotion "
                  "classifiers, and found that disgust dominated the emotional "
                  "landscape."),
        "source": "publisher abstract, link.springer.com/chapter/10.1007/978-981-96-2124-8_7",
        "quote": (
            "The research used machine learning techniques, specifically sentiment "
            "analysis and emotion analysis, to obtain insights into the thoughts and "
            "sentiments in social media. The study covered 80 days and concentrated "
            "on Reddit discussions about the Israel-Hamas conflict. ... Hugging "
            "Face's Transformers emotion analysis revealed a majority of negative "
            "feelings, with disgust ranking highest and being followed by sadness, "
            "happiness, fear, anger, and surprise."),
    },
    "flesch1948": {
        "why": ("APA PsycNet, subscription only. Unpaywall reports is_oa false "
                "with zero open locations. A 1948 paper has no preprint and no "
                "repository copy."),
        "get": ("Plaksha or Tartu library access to APA PsycNet. Not needed to "
                "check the citation: see below."),
        "claim": "readability [uses] the Flesch formula",
        "source": "our own implementation, 06_revision/core_stats.py line 75",
        "quote": (
            "return 206.835 - 1.015 * (len(words) / sents) "
            "- 84.6 * (sum(_syll(w) for w in words) / len(words))\n\n"
            "        The constants 206.835, 1.015 and 84.6 are Flesch's Reading Ease\n"
            "        coefficients, published in that 1948 paper. They can be checked\n"
            "        against any free statement of the formula. Nothing is taken from\n"
            "        the paper except the formula itself, so there is no finding here\n"
            "        that could be misreported."),
    },
}


def wrap(s, width=66, indent=8):
    out, line = [], ""
    for para in s.split("\n"):
        if not para.strip():
            out.append("")
            line = ""
            continue
        for w in para.split():
            if len(line) + len(w) + 1 > width:
                out.append(" " * indent + line)
                line = w
            else:
                line = (line + " " + w).strip()
        if line:
            out.append(" " * indent + line)
            line = ""
    return out


def main():
    out = ["EVIDENCE FOR REFERENCES WITH NO DOWNLOADABLE FULL TEXT", ""]
    out.append("Three of the 22 references could not be stored as a PDF. For each,")
    out.append("this file records why, how to obtain it, and the verbatim publisher")
    out.append("text that supports the claim the manuscript attaches to it.")
    out.append("")
    out.append("=" * 72)

    for key, e in EVIDENCE.items():
        out.append("")
        out.append(key)
        out.append("-" * 72)
        out.append("  why no PDF:")
        out += wrap(e["why"])
        out.append("  how to get it:")
        out += wrap(e["get"])
        out.append("  the manuscript claims:")
        out += wrap(e["claim"])
        out.append(f"  evidence ({e['source']}):")
        out += wrap(e["quote"])
        out.append("")

    out.append("=" * 72)
    out.append("")
    out.append("No paywall was circumvented to produce this audit. Where a paper")
    out.append("could not be obtained legitimately, that is recorded as a gap")
    out.append("rather than filled from an unauthorised source.")

    text = "\n".join(out) + "\n"
    (REF / "EVIDENCE.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
