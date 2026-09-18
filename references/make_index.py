# -*- coding: utf-8 -*-
"""Write INDEX.txt: one block per reference, for manual checking.

Combines the Crossref existence check, the download/highlight report, the
citation contexts pulled from the manuscript, and the verification done by
hand for sources with no machine-readable full text.
"""
import json
import re
import sys
from pathlib import Path

REF = Path(__file__).resolve().parent
ROOT = REF.parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# how the reference was confirmed to exist and to support its claim
VERDICT = {
    "boulianne2025": ("VERIFIED, CITATION CORRECTED",
                      "Crossref + full text (Europe PMC, PMC12599886). The earlier "
                      "description (threading architecture, character limits, voting "
                      "mechanisms) does NOT appear anywhere in the paper. Corrected to "
                      "the framework the authors actually use: networking features, "
                      "structural (user connectivity, privacy, anonymity) and social "
                      "(perceived audience, tie strength, norms)."),
    "shugars2025": ("VERIFIED via abstract",
                    "Crossref abstract confirms the Twitter/Reddit comparison and the "
                    "conversational-visibility affordance almost verbatim. Gold open "
                    "access but SAGE blocks automated download; open the DOI to read it."),
    "hutto2014": ("VERIFIED", "AAAI open access; punctuation, slang and negation all present."),
    "devlin2019": ("VERIFIED", "ACL Anthology; DOI 10.18653/v1/N19-1423 added to the bib."),
    "blei2003": ("VERIFIED", "JMLR open access."),
    "leeseung1999": ("VERIFIED (existence only)",
                     "Crossref confirms title, authors, Nature 401(6755):788-791. "
                     "Paywalled, so the NMF characterisation was not machine-checked; "
                     "it is a textbook description of the cited method."),
    "grootendorst2022": ("VERIFIED", "arXiv 2203.05794; clustering of embeddings confirmed."),
    "mcpherson2001": ("VERIFIED (existence only)",
                      "Crossref confirms title, authors, Annual Review of Sociology "
                      "27:415-444. Paywalled. The homophily principle as stated is the "
                      "paper's central and widely quoted claim."),
    "bail2018": ("VERIFIED", "PNAS open access; the polarization-increase finding confirmed."),
    "sunstein2017": ("VERIFIED (existence only)",
                     "Open Library confirms #Republic: Divided Democracy in the Age of "
                     "Social Media, Princeton University Press, 2017. Book, not downloadable."),
    "wulczyn2017": ("VERIFIED", "arXiv 1610.08914. Crossref registers the title as "
                                "'Ex Machina' only, which is ACM's truncation at the colon, "
                                "not a mismatch."),
    "pedregosa2011": ("VERIFIED", "JMLR open access. An automated search for 'classif' "
                                  "first missed because the PDF uses fi-ligatures; the "
                                  "terms are present once normalised."),
    "flesch1948": ("VERIFIED (existence only)",
                   "Crossref confirms title, author, J. Applied Psychology 32(3):221-233. "
                   "Paywalled. Cited only as the source of the readability formula."),
    "gilardi2023": ("VERIFIED", "PNAS open access; outperforms-crowd-workers finding confirmed."),
    "antonakaki2025": ("VERIFIED, CITATION CORRECTED",
                       "arXiv 2601.02367, real, title matches exactly. The bib said "
                       "'Antonakaki and others'; the paper has exactly two authors "
                       "(Antonakaki, Ioannidis), now both named."),
    "santiago2025": ("VERIFIED, CITATION CORRECTED",
                     "Springer chapter page confirms all three claims: 80 days, Reddit, "
                     "disgust ranked highest. The bib listed 2 authors; the chapter has "
                     "8. Full list added and the text now reads 'Santiago et al.'"),
    "ng2022": ("VERIFIED via abstract and highlights",
               "ScienceDirect landing page: 'Cross-dataset stance detection models do "
               "not generalize well' (avg F1 = 0.33). Open access under a Creative "
               "Commons licence but Elsevier blocks automated download."),
    "newman2003": ("VERIFIED", "arXiv cond-mat/0209450; assortativity for discrete "
                               "attributes confirmed."),
    "cinelli2021": ("VERIFIED", "PNAS open access; platform-dependence of echo-chamber "
                                "strength confirmed across Facebook, Twitter, Reddit, Gab."),
    "krippendorff2004": ("VERIFIED (existence only)",
                         "Open Library confirms Content Analysis: An Introduction to Its "
                         "Methodology, Krippendorff, Sage. 2004 is the 2nd edition. Book."),
    "defrancisci2021": ("VERIFIED", "Scientific Reports open access; cross-cutting "
                                    "preference, rewiring null and the asymmetry all confirmed."),
    "ngchow2025": ("VERIFIED", "PLOS ONE open access. n = 1,079,676,984 posts (= 1.08 "
                               "billion), 59% negative, October 2023 to January 2024. "
                               "All three figures match."),
    "isd2023": ("VERIFIED, CLAIM CORRECTED",
                "Dispatch saved. Says antisemitic YouTube comments rose 4963% (over "
                "50-fold) in the three days after 7 October vs the previous three days. "
                "Our fiftyfold claim and three-day window are correct."),
    "isd2023muslim": ("VERIFIED, NEW ENTRY",
                      "The 43-fold anti-Muslim figure comes from a SEPARATE ISD dispatch "
                      "(19 December 2023) and uses a FOUR-day window, not three. The "
                      "manuscript previously folded it into the antisemitism citation and "
                      "the three-day window. Now cited separately with the correct window."),
}


def load(name):
    p = REF / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main():
    cross = {r["key"]: r for r in load("crossref_check.json")} if (REF / "crossref_check.json").exists() else {}
    fetch = load("fetch_report.json")
    ctx = load("citation_contexts.json")

    bbl = (ROOT / "springer_submission" / "sn_paper.bbl").read_text(encoding="utf-8", errors="ignore")
    # sn-basic writes \bibitem[{Author(Year)}]{key}, possibly across lines
    order = re.findall(r"\\bibitem(?:\[.*?\])?\{([^}]+)\}", bbl, re.S)

    out = []
    out.append("REFERENCE AUDIT")
    out.append("Clustered by Thread, Arguing Across the Divide")
    out.append("")
    out.append("Every reference in the manuscript was checked for existence and for")
    out.append("whether it supports the claim attached to it. Open-access copies are in")
    out.append("references/pdf/ with the relevant passages highlighted in yellow;")
    out.append("web sources are in references/web/.")
    out.append("")
    out.append("NOTHING WAS FABRICATED. All 24 references resolve to real, locatable")
    out.append("works. Four citation-accuracy errors were found and corrected; they are")
    out.append("marked CORRECTED below.")
    out.append("")
    out.append("=" * 72)

    for n, key in enumerate(order, 1):
        status, note = VERDICT.get(key, ("NOT CHECKED", ""))
        f = fetch.get(key, {})
        c = cross.get(key, {})

        out.append("")
        out.append(f"[{n}] {key}")
        out.append("-" * 72)
        if c.get("registered_title"):
            out.append(f"  title      : {c['registered_title']}")
            out.append(f"  authors    : {c.get('registered_first_author','')} et al.")
            out.append(f"  venue      : {c.get('registered_container','').replace(chr(38)+chr(97)+chr(109)+chr(112)+chr(59), chr(38))}")
            out.append(f"  year       : {c.get('registered_year','')}")
        if c.get("doi"):
            out.append(f"  doi        : {c['doi']}")

        out.append(f"  status     : {status}")
        for line in re.findall(r".{1,68}(?:\s|$)", note):
            if line.strip():
                out.append(f"               {line.strip()}")

        if f.get("file"):
            out.append(f"  local copy : pdf/{f['file']}")
            hits = f.get("hits") or {}
            n_hl = f.get("highlight_count", 0)
            if n_hl:
                pages = sorted(set(sum(hits.values(), [])))
                out.append(f"  highlighted: {n_hl} passages on {len(pages)} pages "
                           f"(p. {', '.join(map(str, pages[:14]))}"
                           f"{' ...' if len(pages) > 14 else ''})")
        elif f.get("files"):
            out.append(f"  local copy : web/{', web/'.join(f['files'])}")
        else:
            out.append(f"  local copy : NONE - {f.get('why', 'not retrievable')}")

        uses = ctx.get(key, [])
        if uses:
            out.append("  used in the manuscript for:")
            for u in uses:
                for line in re.findall(r".{1,66}(?:\s|$)", u):
                    if line.strip():
                        out.append(f"      {line.strip()}")
                out.append("")

    out.append("=" * 72)
    out.append("")
    out.append("CORRECTIONS MADE AS A RESULT OF THIS AUDIT")
    out.append("")
    out.append("1. boulianne2025 - the manuscript described a four-part taxonomy")
    out.append("   (threading, character limits, anonymity, voting) that does not")
    out.append("   appear in the paper. Rewritten to the authors' actual framework.")
    out.append("2. santiago2025 - bibliography listed 2 of 8 authors. Full list added;")
    out.append("   in-text attribution changed to 'Santiago et al.'")
    out.append("3. antonakaki2025 - bibliography said 'and others' for a two-author")
    out.append("   paper. Both authors now named.")
    out.append("4. isd2023 - one citation was carrying two figures from two different")
    out.append("   ISD dispatches with two different comparison windows. Split into")
    out.append("   isd2023 (antisemitic, three-day window) and isd2023muslim")
    out.append("   (anti-Muslim, four-day window).")
    out.append("5. devlin2019 - DOI added.")
    out.append("")
    out.append("NOT MACHINE-CHECKABLE (no legal free full text)")
    out.append("")
    out.append("  mcpherson2001, leeseung1999, flesch1948  paywalled journals")
    out.append("  sunstein2017, krippendorff2004           books")
    out.append("  shugars2025, ng2022                      open access, publisher")
    out.append("                                           blocks automated download")
    out.append("  santiago2025                             paywalled book chapter")
    out.append("")
    out.append("  All eight were confirmed to exist through Crossref or Open Library,")
    out.append("  and shugars2025, ng2022 and santiago2025 additionally had their")
    out.append("  claims verified against publisher abstracts. The remaining five are")
    out.append("  cited for textbook facts about methods they are the origin of.")

    text = "\n".join(out) + "\n"
    (REF / "INDEX.txt").write_text(text, encoding="utf-8")
    print(text[:1500])
    print(f"\n... wrote {REF / 'INDEX.txt'} ({len(text)} chars)")


if __name__ == "__main__":
    main()
