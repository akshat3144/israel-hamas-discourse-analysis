# -*- coding: utf-8 -*-
"""Match every reference in the compiled bibliography against what is stored
locally, so the counts in INDEX.txt can be trusted."""
import re
import sys
from pathlib import Path

REF = Path(__file__).resolve().parent
ROOT = REF.parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

bbl = (ROOT / "springer_submission" / "sn_paper.bbl").read_text(encoding="utf-8",
                                                               errors="ignore")
cited = re.findall(r"\\bibitem(?:\[.*?\])?\{([^}]+)\}", bbl, re.S)

pdfs = {p.stem for p in (REF / "pdf").glob("*.pdf")}
html = {p.name for p in (REF / "web").glob("*.html")}

# the two ISD dispatches are saved as isd2023_1 / isd2023_2
web_for = {"isd2023": "isd2023_1.html", "isd2023muslim": "isd2023_2.html"}

have, missing, extra = [], [], sorted(pdfs - set(cited))

for k in cited:
    if k in pdfs:
        have.append((k, f"pdf/{k}.pdf"))
    elif web_for.get(k) in html:
        have.append((k, f"web/{web_for[k]}"))
    else:
        missing.append(k)

print(f"references in the compiled bibliography : {len(cited)}")
print(f"  with a local full text                : {len(have)}")
print(f"  without                               : {len(missing)}")
print(f"PDF files in references/pdf             : {len(pdfs)}")
print(f"  of which NOT a cited reference        : {len(extra)}")
print()
print("WITHOUT A LOCAL COPY:")
for k in missing:
    print("   ", k)
print()
print("STORED BUT NOT A REFERENCE (supplementary):")
for k in extra:
    print("   ", k)
print()
print(f"arithmetic: {len(pdfs)} pdfs - {len(extra)} supplementary "
      f"+ {len(html)} web pages = {len(pdfs) - len(extra) + len(html)} "
      f"covering {len(have)} references")
