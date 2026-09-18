# -*- coding: utf-8 -*-
"""Pull the exact comparison windows and multipliers out of the two ISD pages."""
import re
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parent / "web"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def text(p):
    raw = p.read_text(encoding="utf-8", errors="ignore")
    raw = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw, flags=re.S | re.I)
    return " ".join(re.sub(r"<[^>]+>", " ", raw).split())


def show(label, t, pattern, width=260, limit=6):
    print(f"\n--- {label} ---")
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


for name, label in (("isd2023_1.html", "ANTISEMITISM DISPATCH"),
                    ("isd2023_2.html", "ANTI-MUSLIM DISPATCH")):
    p = WEB / name
    if not p.exists():
        print(f"{name}: missing")
        continue
    t = text(p)
    print("=" * 74)
    print(label, f"({name}, {len(t)} chars)")
    show("fold increases", t, r"\d+[\s-]*fold")
    show("comparison windows", t, r"(three|four|seven|\d+)\s+days")
    show("YouTube volume claim", t, r"YouTube.{0,200}")
