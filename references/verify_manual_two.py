# -*- coding: utf-8 -*-
"""Highlight and claim-check the two open-access papers downloaded by hand,
which the publishers' bot protection would not serve to a script."""
import re
import sys
from pathlib import Path

import fitz

REF = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WORK = {
    "shugars2025": dict(
        terms=["conversational visibility", "Reddit", "Twitter", "cross-cutting",
               "connective democracy", "affordance", "intergroup"],
        claims=[("compares Twitter and Reddit directly", r"compar\w*.{0,80}(Twitter|Reddit)"),
                ("Reddit enables more productive intergroup discourse",
                 r"Reddit.{0,120}(productive|promise|intergroup)"),
                ("driven by conversational visibility",
                 r"conversational visibility"),
                ("Twitter highly polarized", r"Twitter.{0,80}polariz")]),
    "ng2022": dict(
        terms=["stance detection", "generalize", "cross-dataset", "F1",
               "annotation", "Twitter"],
        claims=[("cross-dataset models do not generalize well",
                 r"do(es)? not generalize|poor\w*.{0,40}generaliz"),
                ("average F1 around 0.33", r"0\.33"),
                ("aggregating datasets improves it", r"aggregat\w+.{0,80}(improv|0\.69)"),
                ("inconsistent annotations across datasets",
                 r"annotations? (are )?(not )?(consistent|inconsistent)")]),
}


def main():
    for key, spec in WORK.items():
        p = REF / "pdf" / f"{key}.pdf"
        doc = fitz.open(p)
        text = "\n".join(pg.get_text() for pg in doc)

        added = 0
        for term in spec["terms"]:
            for pno in range(doc.page_count):
                page = doc[pno]
                for r in page.search_for(term)[:5]:
                    a = page.add_highlight_annot(r)
                    a.set_colors(stroke=(1, 1, 0))
                    a.update()
                    added += 1
        if added:
            doc.saveIncr()
        doc.close()

        print("=" * 64)
        print(f"{key}: {added} highlights added")
        for desc, pat in spec["claims"]:
            m = re.search(pat, text, re.I | re.S)
            print(f"   {'FOUND    ' if m else 'NOT FOUND'} {desc}")
            if m:
                s = " ".join(text[max(0, m.start() - 70):m.end() + 70].split())
                print(f"        ...{s}...")


if __name__ == "__main__":
    main()
