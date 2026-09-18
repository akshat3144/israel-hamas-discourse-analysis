# -*- coding: utf-8 -*-
"""For every claim the manuscript makes, rank the sentences in the cited source
that could be its origin.

This does not highlight anything. It produces candidates for a human to pick
from, because matching a claim to the passage it came from is a judgement, not
a keyword search.
"""
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import fitz

REF = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STOP = set("""a an the and or but if of to in on for with as by at from is are was
were be been being that this these those it its their there here we our us they
them he she his her which who whom whose what when where how than then so such
can could may might will would shall should must do does did not no nor also
more most other some any each both between into over under about across during""".split())


def sentences(pdf):
    """[(page, sentence)] with running text reflowed."""
    doc = fitz.open(pdf)
    out = []
    for pno in range(doc.page_count):
        raw = doc[pno].get_text()
        raw = raw.replace("-\n", "").replace("\n", " ")
        raw = re.sub(r"\s+", " ", raw)
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", raw):
            s = s.strip()
            if 40 <= len(s) <= 600:
                out.append((pno + 1, s))
    doc.close()
    return out


def toks(s):
    return [w for w in re.findall(r"[a-z]+", s.lower())
            if w not in STOP and len(w) > 2]


def score_all(claim, sents, idf):
    c = Counter(toks(claim))
    if not c:
        return []
    scored = []
    for pno, s in sents:
        t = Counter(toks(s))
        if not t:
            continue
        shared = set(c) & set(t)
        if not shared:
            continue
        # idf-weighted overlap, length-normalised so long sentences do not win
        num = sum(idf.get(w, 1.0) for w in shared)
        den = math.sqrt(len(t)) + 3
        scored.append((num / den, pno, s))
    scored.sort(reverse=True)
    return scored


def main():
    ctx = json.loads((REF / "citation_contexts.json").read_text(encoding="utf-8"))
    only = sys.argv[1:] or sorted(ctx)

    for key in only:
        pdf = REF / "pdf" / f"{key}.pdf"
        if not pdf.exists():
            print(f"\n##### {key}: NO PDF")
            continue
        sents = sentences(pdf)
        df = Counter()
        for _, s in sents:
            df.update(set(toks(s)))
        n = max(len(sents), 1)
        idf = {w: math.log(n / (1 + d)) + 1 for w, d in df.items()}

        print(f"\n{'#' * 74}\n##### {key}   ({len(sents)} sentences)\n{'#' * 74}")
        for i, claim in enumerate(ctx.get(key, []), 1):
            print(f"\n  OUR CLAIM {i}:")
            print(f"    {claim[:330]}")
            print("  CANDIDATE PASSAGES:")
            for sc, pno, s in score_all(claim, sents, idf)[:4]:
                print(f"    [{sc:5.2f}] p.{pno:<3} {s[:300]}")


if __name__ == "__main__":
    main()
