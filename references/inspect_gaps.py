# -*- coding: utf-8 -*-
"""Look at the exact wording around the claims that did not match, so the
mismatches can be judged rather than guessed at."""
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


def show(label, t, pattern, width=190, limit=6):
    print(f"\n--- {label} ---")
    hits = list(re.finditer(pattern, t, re.I))
    if not hits:
        print("   (no match)")
        return
    for m in hits[:limit]:
        a, b = max(0, m.start() - width // 2), m.end() + width // 2
        print("   ..." + " ".join(t[a:b].split()) + "...")


def main():
    t = text("ngchow2025")
    print("=" * 74)
    print("ngchow2025  title:", " ".join(t.split("\n")[0:2])[:120])
    show("any 'billion'", t, r"billion")
    show("large post counts", t, r"\b1[,.]0\d\d?\b|\b1\.\d{1,2}\s*(billion|bn|B)\b")
    show("corpus size statements", t, r"(posts|comments|documents)\D{0,40}\d[\d,\.]{5,}")
    show("59 percent", t, r".{0,60}59(\.\d+)?\s*%.{0,60}")

    t = text("boulianne2025")
    print("\n" + "=" * 74)
    print("boulianne2025  title:", " ".join(t.split("\n")[0:3])[:140])
    show("thread", t, r"thread")
    show("character", t, r"character")
    show("vote/voting", t, r"vot(e|ing|es)")
    show("affordance", t, r"affordance")

    t = text("pedregosa2011")
    print("\n" + "=" * 74)
    print("pedregosa2011  chars:", len(t))
    print("  first 300 chars:", " ".join(t[:300].split()))
    show("classif", t, r"classif")


if __name__ == "__main__":
    main()
