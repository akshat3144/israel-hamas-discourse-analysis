# -*- coding: utf-8 -*-
"""Build a minimal Overleaf bundle: only the sources and the figures the paper
actually includes (the paperfigs folder still holds 58 superseded images)."""
import re
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "overleaf_paper.zip"

SOURCES = ["paper.tex", "refs.bib", "authblk.sty", "stfloats.sty",
           "rw_platform.tex", "rw_conflict.tex", "rq2_body.tex"]

# every \includegraphics target across the compiled sources
src = ""
for f in ["paper.tex", "rw_platform.tex", "rw_conflict.tex", "rq2_body.tex"]:
    src += (HERE / f).read_text(encoding="utf-8") + "\n"
used = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", src)))
print(f"figures referenced by the paper: {len(used)}")
for u in used:
    print("   ", u)

missing = [u for u in used if not (HERE / "paperfigs" / u).exists()]
if missing:
    raise SystemExit(f"MISSING figure files: {missing}")

if OUT.exists():
    OUT.unlink()
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for s in SOURCES:
        z.write(HERE / s, s)
    for u in used:
        z.write(HERE / "paperfigs" / u, f"paperfigs/{u}")

size_mb = OUT.stat().st_size / 1024 / 1024
print(f"\nwrote {OUT.name}: {len(SOURCES)} sources + {len(used)} figures, "
      f"{size_mb:.2f} MB")
