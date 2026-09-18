# -*- coding: utf-8 -*-
"""Check the references folder against the compiled bibliography.

Flags three kinds of drift: entries in sources.py for references the paper no
longer cites, PDFs that are not a cited reference, and any PDF still carrying
highlights that mark_provenance.py did not put there.
"""
import re
import sys
from pathlib import Path

import fitz

REF = Path(__file__).resolve().parent
ROOT = REF.parent
sys.path.insert(0, str(REF))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from mark_provenance import PROVENANCE  # noqa: E402

bbl = (ROOT / "springer_submission" / "sn_paper.bbl").read_text(
    encoding="utf-8", errors="ignore")
cited = set(re.findall(r"\\bibitem(?:\[.*?\])?\{([^}]+)\}", bbl, re.S))

src = (REF / "sources.py").read_text(encoding="utf-8")
listed = set(re.findall(r'^\s*"([A-Za-z0-9_]+)":\s*dict\(', src, re.M))

pdfs = {p.stem for p in (REF / "pdf").glob("*.pdf")}

print(f"cited in the paper   : {len(cited)}")
print(f"listed in sources.py : {len(listed)}")
print(f"PDF files            : {len(pdfs)}")
print(f"in PROVENANCE map    : {len(PROVENANCE)}")

print("\nSTALE in sources.py (no longer cited):")
for k in sorted(listed - cited) or ["   none"]:
    print(f"   {k}")

print("\nMISSING from sources.py (cited but absent):")
for k in sorted(cited - listed) or ["   none"]:
    print(f"   {k}")

extra = sorted(pdfs - cited)
print("\nPDFs that are not a cited reference:")
for k in extra or ["   none"]:
    print(f"   {k}")

print("\nPDFs carrying highlights but not in the PROVENANCE map")
print("(these would still hold the old keyword marks):")
found = False
for k in sorted(pdfs):
    if k in PROVENANCE:
        continue
    d = fitz.open(REF / "pdf" / f"{k}.pdf")
    n = sum(1 for i in range(d.page_count)
            for a in (d[i].annots() or []) if a.type[0] == 8)
    d.close()
    if n:
        print(f"   {k}: {n} highlights")
        found = True
if not found:
    print("   none")
