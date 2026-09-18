# -*- coding: utf-8 -*-
"""How much of the manuscript does the provenance record actually cover?

Counts every citation instance in the body, and checks which of them have a
claim mapped to a specific passage in the source.
"""
import json
import re
import sys
from pathlib import Path

REF = Path(__file__).resolve().parent
ROOT = REF.parent
sys.path.insert(0, str(REF))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from mark_provenance import PROVENANCE  # noqa: E402

tex = (ROOT / "springer_submission" / "sn_paper.tex").read_text(encoding="utf-8")
body = tex[tex.index(r"\section{Introduction}"):]

# every place a source is invoked, counting multi-key \cite once per key
instances = []
for m in re.finditer(r"\\cite\{([^}]*)\}", body):
    for k in (x.strip() for x in m.group(1).split(",")):
        instances.append(k)

per_key = {}
for k in instances:
    per_key[k] = per_key.get(k, 0) + 1

ctx = json.loads((REF / "citation_contexts.json").read_text(encoding="utf-8"))

print("CITATION COVERAGE")
print("=" * 66)
print(f"  citation instances in the body : {len(instances)}")
print(f"  distinct sources cited         : {len(per_key)}")
print(f"  claim->passage mappings made   : "
      f"{sum(len(v) for v in PROVENANCE.values())}")
print()
print(f"  {'source':20}{'cited':>7}{'sentences':>11}{'mapped':>8}")
print("  " + "-" * 44)
gaps = []
for k in sorted(per_key):
    n_ctx = len(ctx.get(k, []))
    n_map = len(PROVENANCE.get(k, []))
    flag = "" if n_map else "   <- NO MAPPING"
    print(f"  {k:20}{per_key[k]:>7}{n_ctx:>11}{n_map:>8}{flag}")
    if not n_map:
        gaps.append(k)

print()
if gaps:
    print(f"  sources with no passage mapped: {', '.join(gaps)}")
else:
    print("  every cited source has at least one claim mapped to a passage")

print()
print("WHAT THIS AUDIT DOES NOT COVER")
print("=" * 66)
print("  The manuscript's own results. Every number produced by our pipeline")
print("  (corpus counts, Cramer's V, Newman r, Krippendorff alpha, the human")
print("  validation, the context experiment, the robustness checks) rests on")
print("  the code in 06_revision/, not on any cited source. Nothing in this")
print("  folder verifies them.")
