# -*- coding: utf-8 -*-
"""What does Boulianne et al. actually distinguish platforms by, and is the
pedregosa miss just an fi-ligature artefact?"""
import re
import sys
import unicodedata
from pathlib import Path

import fitz

REF = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def text(key, normalise=False):
    d = fitz.open(REF / "pdf" / f"{key}.pdf")
    t = "\n".join(p.get_text() for p in d)
    d.close()
    if normalise:                       # fold fi/fl ligatures into plain ASCII
        t = unicodedata.normalize("NFKD", t)
    return t


def show(label, t, pattern, width=230, limit=5):
    print(f"\n--- {label} ---")
    hits = list(re.finditer(pattern, t, re.I))
    if not hits:
        print("   (no match)")
        return
    for m in hits[:limit]:
        a, b = max(0, m.start() - width // 2), m.end() + width // 2
        print("   ..." + " ".join(t[a:b].split()) + "...")


def main():
    print("=" * 74)
    print("PEDREGOSA: ligature test")
    raw = text("pedregosa2011")
    norm = text("pedregosa2011", normalise=True)
    print("  'classif' in raw       :", bool(re.search("classif", raw, re.I)))
    print("  'classif' in normalised:", bool(re.search("classif", norm, re.I)))
    show("classification (normalised)", norm, r"classi\w*", limit=4)

    print("\n" + "=" * 74)
    print("BOULIANNE: what are platforms actually distinguished by?")
    t = text("boulianne2025", normalise=True)
    show("structural networking features", t, r"[Ss]tructural networking features")
    show("we distinguish", t, r"[Ww]e distinguish")
    show("identifiability", t, r"identifiab")
    show("network(ing) feature", t, r"networking feature")
    show("abstract", t, r"Abstract", width=900, limit=1)


if __name__ == "__main__":
    main()
