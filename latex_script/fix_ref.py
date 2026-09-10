# -*- coding: utf-8 -*-
"""Turn the mangled '\\ef{' back into '\\ref{'."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
BS = chr(92)
bad = BS + "ef{"
good = BS + "ref{"

for name in ["body_new.tex", "rq2_body.tex", "rw_conflict.tex", "rw_platform.tex"]:
    p = HERE / name
    t = p.read_text(encoding="utf-8")
    n = t.count(bad)
    if n:
        p.write_text(t.replace(bad, good), encoding="utf-8")
    print(f"  {name}: fixed {n}")
print("done")
