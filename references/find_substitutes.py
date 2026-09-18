# -*- coding: utf-8 -*-
"""Can a reference we already hold in full text carry the homophily claim,
so McPherson is no longer load-bearing?"""
import re
import sys
from pathlib import Path

import fitz

REF = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def text(key):
    d = fitz.open(REF / "pdf" / f"{key}.pdf")
    t = "\n".join(p.get_text() for p in d)
    d.close()
    return t


def show(key, pattern, width=260, limit=4):
    t = text(key)
    print(f"\n--- {key} : /{pattern}/ ---")
    seen = set()
    for m in re.finditer(pattern, t, re.I):
        a, b = max(0, m.start() - width // 2), m.end() + width // 2
        s = " ".join(t[a:b].split())
        if s in seen:
            continue
        seen.add(s)
        print("   ..." + s + "...")
        if len(seen) >= limit:
            break
    if not seen:
        print("   (no match)")


print("=" * 74)
print("Does an already-cited, already-downloaded paper define homophily?")
show("cinelli2021", r"homophil\w*")
show("defrancisci2021", r"homophil\w*")
show("newman2003", r"homophil\w*")
show("newman2003", r"assortative mixing.{0,160}")
