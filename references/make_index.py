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
    "shugars2025": ("VERIFIED from full text",
                    "Downloaded by hand (SAGE blocks scripts). Abstract states the "
                    "Twitter/Reddit comparison, Twitter highly polarized, Reddit showing "
                    "promise for productive intergroup discourse, driven by conversational "
                    "visibility. Matches our sentence almost word for word. CC BY-NC."),
    "hutto2014": ("VERIFIED", "AAAI open access; punctuation, slang and negation all present."),
    "devlin2019": ("VERIFIED", "ACL Anthology; DOI 10.18653/v1/N19-1423 added to the bib."),
    "blei2003": ("VERIFIED", "JMLR open access."),
    "leeseung2000": ("VERIFIED from full text, REFERENCE SWAPPED",
                     "The paywalled Nature 1999 letter was replaced with the same "
                     "authors' open NeurIPS 2000 paper, Algorithms for Non-negative "
                     "Matrix Factorization, which is the more apt citation since we use "
                     "the algorithms rather than the parts-based argument. Free from "
                     "papers.nips.cc; multiplicative update rules highlighted."),
    "grootendorst2022": ("VERIFIED", "arXiv 2203.05794; clustering of embeddings confirmed."),
    "bail2018": ("VERIFIED", "PNAS open access; the polarization-increase finding confirmed."),
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
    "ng2022": ("VERIFIED from full text",
               "Downloaded by hand (Elsevier blocks scripts). Full text confirms "
               "'models do not generalize well (avg F1=0.33)', aggregation lifting it "
               "to 0.69, and inconsistent annotations across datasets. CC BY-NC-ND."),
    "hofmann2026": ("VERIFIED from full text, REPLACES GREY LITERATURE",
                    "Peer-reviewed: Springer, Lecture Notes in Information Systems "
                    "and Organisation, 2026, pp. 257-272, doi 10.1007/978-3-032-08489-7_18. "
                    "Free preprint on arXiv 2503.10648. Full text confirms 4,983 "
                    "hand-annotated YouTube comments, AUROC 0.83 to 0.90, hate speech "
                    "in 40.4 percent of public-source and 31.6 percent of private-source "
                    "comments, and the same three stance labels we use."),
    "rosen2025": ("VERIFIED via abstract, REPLACES GREY LITERATURE",
                  "Peer-reviewed: Social Media + Society 2025, "
                  "doi 10.1177/20563051251383635, gold open access. Abstract confirms "
                  "hate messaging targeting Muslims and Jews increased dramatically "
                  "after 7 October 2023, and the reply-dynamics finding. SAGE blocks "
                  "automated download; open the DOI to read it."),
    "newman2003": ("VERIFIED", "arXiv cond-mat/0209450; assortativity for discrete "
                               "attributes confirmed."),
    "cinelli2021": ("VERIFIED", "PNAS open access; platform-dependence of echo-chamber "
                                "strength confirmed across Facebook, Twitter, Reddit, Gab."),
    "hayes2007": ("VERIFIED from full text, REFERENCE SWAPPED",
                  "Krippendorff's 2004 book was replaced with Hayes and Krippendorff "
                  "2007, Answering the Call for a Standard Reliability Measure for "
                  "Coding Data, Communication Methods and Measures 1(1):77-89, "
                  "doi 10.1080/19312450709336664. Freely available from UPenn "
                  "Annenberg and it is the standard journal citation for alpha. "
                  "Krippendorff's own Computing Krippendorff's Alpha-Reliability "
                  "(2011) is also stored alongside it."),
    "defrancisci2021": ("VERIFIED", "Scientific Reports open access; cross-cutting "
                                    "preference, rewiring null and the asymmetry all confirmed."),
    "ngchow2025": ("VERIFIED", "PLOS ONE open access. n = 1,079,676,984 posts (= 1.08 "
                               "billion), 59% negative, October 2023 to January 2024. "
                               "All three figures match."),
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
    out.append("NOTHING WAS FABRICATED. All 22 references resolve to real, locatable")
    out.append("works. Four citation-accuracy errors were found and corrected; they are")
    out.append("marked CORRECTED below.")
    out.append("")

    # file inventory, so the counts here can be checked against the folder
    pdfs = {p.stem for p in (REF / "pdf").glob("*.pdf")}
    html = sorted(p.name for p in (REF / "web").glob("*.html"))
    web_for = {"isd2023": "isd2023_1.html", "isd2023muslim": "isd2023_2.html"}
    held = [k for k in order if k in pdfs or web_for.get(k) in html]
    absent = [k for k in order if k not in held]
    extra = sorted(pdfs - set(order))

    out.append("WHAT IS IN THIS FOLDER")
    out.append("")
    out.append(f"  references in the manuscript          {len(order):>3}")
    out.append(f"  with a local full text                {len(held):>3}")
    out.append(f"  without                               {len(absent):>3}"
               f"   ({', '.join(absent)})")
    out.append("")
    out.append(f"  PDF files in references/pdf           {len(pdfs):>3}")
    out.append("")
    out.append(f"  The PDF count is {len(pdfs)}, not {len(held)}, because "
               f"{len(extra)} file is a supporting")
    out.append("  document rather than a reference in its own right:")
    for k in extra:
        out.append(f"      {k}")
    out.append("")
    out.append(f"  So: {len(pdfs)} PDFs - {len(extra)} supporting = "
               f"{len(held)} of {len(order)} references.")
    out.append("")
    out.append("  Every source is now a peer-reviewed paper or a preprint of one.")
    out.append("  No web pages, reports or books are cited.")
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

        # the folder is the authority, not the fetch log: several PDFs arrived
        # after that log was written, some of them by hand
        if key in pdfs:
            out.append(f"  local copy : pdf/{key}.pdf")
            hits = f.get("hits") or {}
            n_hl = f.get("highlight_count", 0)
            if n_hl:
                pages = sorted(set(sum(hits.values(), [])))
                out.append(f"  highlighted: {n_hl} passages on {len(pages)} pages "
                           f"(p. {', '.join(map(str, pages[:14]))}"
                           f"{' ...' if len(pages) > 14 else ''})")
        elif web_for.get(key) in html:
            # the two ISD dispatches were saved under one fetch entry
            out.append(f"  local copy : web/{web_for[key]}")
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
    out.append("   ISD dispatches with two different comparison windows. Both have")
    out.append("   since been replaced by peer-reviewed papers; see below.")
    out.append("5. devlin2019 - DOI added.")
    out.append("")
    out.append("SOURCES CHANGED TO MAKE THE EVIDENCE CHECKABLE")
    out.append("")
    out.append("  Five references were either unreadable without a subscription or")
    out.append("  were grey literature. All five are gone.")
    out.append("")
    out.append("  isd2023 + isd2023muslim (web)  ->  hofmann2026 + rosen2025")
    out.append("      The two Institute for Strategic Dialogue dispatches were the")
    out.append("      only non-peer-reviewed sources in the paper. Replaced with")
    out.append("      Hofmann et al. 2026 (Springer LNISO 257-272, free preprint on")
    out.append("      arXiv), which annotates 4,983 YouTube comments on this exact")
    out.append("      conflict using the same three stance labels we use, and Rosen")
    out.append("      and Walther 2025 (Social Media + Society, gold open access) on")
    out.append("      the post-7-October rise in anti-Jewish and anti-Muslim posting.")
    out.append("      The 50-fold and 43-fold figures were dropped with them: those")
    out.append("      were ISD's own measurements and no peer-reviewed study reports")
    out.append("      them, so the claim now rests on what the new sources state.")
    out.append("      Hofmann was also added to Related Work, where it belonged")
    out.append("      anyway as the closest prior work to our design.")
    out.append("")
    out.append("")
    out.append("  krippendorff2004 (book)  ->  hayes2007")
    out.append("      Hayes and Krippendorff 2007, Communication Methods and Measures")
    out.append("      1(1):77-89. Free from UPenn Annenberg, and the standard journal")
    out.append("      citation for the alpha statistic we report.")
    out.append("")
    out.append("  leeseung1999 (Nature)    ->  leeseung2000")
    out.append("      Lee and Seung, Algorithms for Non-negative Matrix Factorization,")
    out.append("      NeurIPS 2000. Free from papers.nips.cc, and the better citation")
    out.append("      because we use the algorithms, not the Nature letter's argument.")
    out.append("")
    out.append("  sunstein2017 (book)      ->  removed")
    out.append("      Cited once for a framing claim that bail2018 already supports,")
    out.append("      and bail2018 is stored here in full text with the finding")
    out.append("      highlighted. Nothing in the paper now depends on it.")
    out.append("")
    out.append("  mcpherson2001 (paywalled)  ->  covered by newman2003")
    out.append("      Newman 2003 states the same principle formally and on")
    out.append("      networks: 'assortative mixing in networks, the tendency for")
    out.append("      vertices to be connected to other vertices that are like (or")
    out.append("      unlike) them'. That is the better citation here anyway,")
    out.append("      because we measure assortativity on a reply graph rather than")
    out.append("      sociological homophily. Newman is free on arXiv and held here")
    out.append("      in full text. Cinelli 2021, also held in full text,")
    out.append("      independently shows homophily underpinning echo-chamber work.")
    out.append("")
    out.append("STILL WITHOUT A DOWNLOADABLE COPY (2 of 22)")
    out.append("")
    out.append("  santiago2025    Springer book chapter, subscription. All three")
    out.append("                  claims we make from it (80 days, Reddit, disgust")
    out.append("                  ranked highest) are stated in the publisher's free")
    out.append("                  abstract, which is quoted in this file above.")
    out.append("  flesch1948      Journal of Applied Psychology, subscription.")
    out.append("                  Kept deliberately. 06_revision/core_stats.py")
    out.append("                  hardcodes his exact constants:")
    out.append("                    206.835 - 1.015*(words/sents) - 84.6*(syll/words)")
    out.append("                  Those three numbers ARE Flesch 1948. No other work")
    out.append("                  can be cited for them, and a 1948 paper has no open")
    out.append("                  version. Citing a later secondary source instead")
    out.append("                  would be weaker scholarship, not stronger.")
    out.append("")
    out.append("  Both were confirmed real through Crossref, which returns the exact")
    out.append("  title, authors, volume and pages we cite. Neither carries a finding")
    out.append("  that could be misreported: santiago2025 is described from its own")
    out.append("  abstract, and flesch1948 is cited only as the origin of a formula")
    out.append("  whose implementation is in this repository.")

    text = "\n".join(out) + "\n"
    (REF / "INDEX.txt").write_text(text, encoding="utf-8")
    print(text[:1500])
    print(f"\n... wrote {REF / 'INDEX.txt'} ({len(text)} chars)")


if __name__ == "__main__":
    main()
