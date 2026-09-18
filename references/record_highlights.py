# -*- coding: utf-8 -*-
"""Read back the highlight annotations actually present in each stored PDF and
fold them into fetch_report.json, so the index reports what a reader will see
rather than what the fetch run happened to log."""
import json
import sys
from pathlib import Path

import fitz

REF = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def annots(path):
    """{highlighted text -> [pages]} for every highlight in the file."""
    doc = fitz.open(path)
    found = {}
    for pno in range(doc.page_count):
        page = doc[pno]
        for a in page.annots() or []:
            if a.type[0] != 8:                  # 8 = Highlight
                continue
            words = page.get_text("words", clip=a.rect)
            phrase = " ".join(w[4] for w in words).strip()
            if not phrase:
                continue
            phrase = phrase[:60]
            found.setdefault(phrase, [])
            if pno + 1 not in found[phrase]:
                found[phrase].append(pno + 1)
    doc.close()
    return found


def main():
    rp = REF / "fetch_report.json"
    report = json.loads(rp.read_text(encoding="utf-8"))

    for key, rec in report.items():
        p = REF / "pdf" / f"{key}.pdf"
        if not p.exists():
            continue
        a = annots(p)
        rec["hits"] = a
        rec["highlight_count"] = sum(len(v) for v in a.values())
        rec.pop("note", None)
        print(f"  {key:18} {rec['highlight_count']:4} highlights on "
              f"{len(set(sum(a.values(), [])))} pages")

    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nupdated {rp}")


if __name__ == "__main__":
    main()
