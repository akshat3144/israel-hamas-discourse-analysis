# -*- coding: utf-8 -*-
"""Trace every number the manuscript reports back to a pipeline output.

Read-only. Extracts each numeric claim from sn_paper.tex with its surrounding
sentence, flattens every JSON in 06_revision/outputs, and reports which claims
can be matched to a computed value and which cannot.

An unmatched number is not automatically wrong: it may be derived (a ratio, a
rounding, a count stated in prose) or produced by a script that prints rather
than saves. It means the number needs checking by hand.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\nitro 5\Desktop\Projects\israel_hamas_discourse_analysis")
OUT = ROOT / "06_revision" / "outputs"
TEX = ROOT / "springer_submission" / "sn_paper.tex"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def flatten(obj, prefix=""):
    """{dotted path: value} for every scalar in a nested structure."""
    flat = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            flat.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            flat.update(flatten(v, f"{prefix}[{i}]"))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        flat[prefix] = obj
    return flat


def load_outputs():
    vals = {}
    for f in sorted(OUT.glob("*.json")):
        try:
            vals.update({f"{f.stem}:{k}": v
                         for k, v in flatten(json.loads(f.read_text(encoding="utf-8"))).items()})
        except Exception as e:
            print(f"  could not read {f.name}: {e}")
    return vals


def paper_numbers():
    """(value, raw string, sentence) for each number stated in the body."""
    t = TEX.read_text(encoding="utf-8")
    t = t[t.index(r"\section{Introduction}"):]
    t = re.sub(r"%.*", "", t)
    t = re.sub(r"\\(cite|ref|label|includegraphics)\{[^}]*\}", " ", t)

    out = []
    for m in re.finditer(r"(?<![\w.])(\d{1,3}(?:\{,\}\d{3})+|\d+\.\d+|\d+)(\\?%)?", t):
        raw = m.group(0)
        num = m.group(1).replace("{,}", "")
        try:
            val = float(num)
        except ValueError:
            continue
        a = max(0, m.start() - 130)
        b = min(len(t), m.end() + 130)
        sent = " ".join(re.sub(r"[{}$\\]", " ", t[a:b]).split())
        out.append((val, raw, sent))
    return out


def close(a, b):
    """does a paper value match a computed one, allowing for rounding?"""
    if b == 0:
        return abs(a) < 1e-9
    for cand in (b, b * 100, b / 100):          # proportions vs percentages
        if abs(a - cand) < 1e-9:
            return True
        for dp in (0, 1, 2, 3, 4):
            if abs(a - round(cand, dp)) < 10 ** (-dp - 1) * 1.01:
                return True
    return False


def main():
    vals = load_outputs()
    nums = paper_numbers()

    # numbers that are structural rather than empirical
    SKIP_CTX = re.compile(
        r"Section|Table|Figure|Fig\.|RQ\d|October|May|2023|2024|2025|2026|"
        r"equation|Eq\.|volume|pp\.|vol\.|Appendix", re.I)

    matched, unmatched = [], []
    seen = set()
    for val, raw, sent in nums:
        if val in (0, 1, 2, 3, 4, 5) and not re.search(r"\d\.\d", raw):
            continue                              # bare small integers
        key = (val, sent[:60])
        if key in seen:
            continue
        seen.add(key)

        hits = [k for k, v in vals.items() if close(val, v)]
        if hits:
            matched.append((val, raw, hits[:3], sent))
        else:
            unmatched.append((val, raw, sent, bool(SKIP_CTX.search(sent))))

    print("=" * 78)
    print(f"numbers stated in the body      : {len(seen)}")
    print(f"  traced to a pipeline output   : {len(matched)}")
    print(f"  not traced                    : {len(unmatched)}")
    print("=" * 78)

    print("\nNOT TRACED, and not obviously structural")
    print("-" * 78)
    n = 0
    for val, raw, sent, structural in unmatched:
        if structural:
            continue
        n += 1
        print(f"\n  {raw}")
        print(f"     ...{sent[:210]}...")
    if not n:
        print("  none")

    print(f"\n\n({sum(1 for u in unmatched if u[3])} further unmatched numbers "
          "look structural: section/table/figure numbers, dates, page numbers)")


if __name__ == "__main__":
    main()
