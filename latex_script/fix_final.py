# -*- coding: utf-8 -*-
"""Definitive repair: strip every CR byte, then restore mangled \\ref commands."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
BS = chr(92)

for name in ["body_new.tex", "rq2_body.tex", "rw_conflict.tex", "rw_platform.tex"]:
    p = HERE / name
    data = p.read_bytes()
    n_cr = data.count(b"\r")
    data = data.replace(b"\r", b"")           # remove ALL carriage returns
    t = data.decode("utf-8")
    fixes = 0
    for bad in (BS + BS + "ef{", BS + "ef{", "~ef{"):
        while bad in t:
            t = t.replace(bad, ("~" + BS + "ref{") if bad.startswith("~")
                          else (BS + "ref{"), 1)
            fixes += 1
    p.write_text(t, encoding="utf-8", newline="\n")
    print(f"  {name}: stripped {n_cr} CR, repaired {fixes} ref command(s)")

# verify
import re
src = ""
for name in ["body_new.tex", "rq2_body.tex", "rw_conflict.tex", "rw_platform.tex"]:
    src += (HERE / name).read_text(encoding="utf-8")
refs = re.findall(r"\\ref\{([^}]+)\}", src)
print(f"\ntotal \\ref found: {len(refs)}")
print("unique:", sorted(set(refs)))
