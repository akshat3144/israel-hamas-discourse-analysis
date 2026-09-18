# -*- coding: utf-8 -*-
"""Highlight the exact figures the manuscript quotes, not just topic words.

These are the strings a reader should check first: if any of them were not in
the source, the manuscript would be misreporting a number.
"""
import sys
from pathlib import Path

import fitz

REF = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# key -> the literal strings the manuscript's numbers come from
NUMBERS = {
    "ngchow2025": ["1,079,676,984", "59%", "October 2023 to January 2024",
                   "prominent peaks"],
    "defrancisci2021": ["cross-cutting", "rewiring", "assortativity"],
    "bail2018": ["substantially more conservative"],
    "gilardi2023": ["outperforms crowd workers"],
    "antonakaki2025": ["BERTopic", "propaganda"],
    "cinelli2021": ["Gab"],
    "boulianne2025": ["networking features", "user connectivity",
                      "perceived audiences", "12,302"],
    "newman2003": ["assortativity coefficient"],
    "wulczyn2017": ["personal attack"],
}


def main():
    for key, terms in NUMBERS.items():
        p = REF / "pdf" / f"{key}.pdf"
        if not p.exists():
            print(f"  {key:18} no PDF")
            continue
        doc = fitz.open(p)
        added = 0
        report = []
        for term in terms:
            pages = []
            for pno in range(doc.page_count):
                page = doc[pno]          # keep the page alive while annotating
                rects = page.search_for(term)
                for r in rects[:4]:
                    a = page.add_highlight_annot(r)
                    a.set_colors(stroke=(1, 1, 0))
                    a.update()
                    added += 1
                if rects:
                    pages.append(pno + 1)
            report.append(f"{term!r} {'p. ' + ','.join(map(str, pages)) if pages else 'NOT FOUND'}")
        if added:
            doc.saveIncr()
        doc.close()
        print(f"  {key}")
        for r in report:
            print(f"      {r}")


if __name__ == "__main__":
    main()
