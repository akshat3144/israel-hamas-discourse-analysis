# -*- coding: utf-8 -*-
"""
Replace LaTeX en dashes ('--') with plain hyphens, 'to', or 'versus'.

'--' renders as the longer en dash. Compounds become hyphens, numeric/date ranges
become 'to', and Pro-Israel--Pro-Palestine becomes 'versus' (a hyphen there would
read as Pro-Israel-Pro-Palestine, which is unparseable).
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ["body_new.tex", "rq2_body.tex", "rw_conflict.tex",
         "rw_platform.tex", "preamble_patched.tex"]

# Explicit cases first (ranges and the ambiguous compound)
EXPLICIT = [
    ("7~October~2023--7~May~2024", "7~October~2023 to 7~May~2024"),
    ("Pro-Israel--Pro-Palestine", "Pro-Israel versus Pro-Palestine"),
    ("24--35 months", "24 to 35 months"),
    ("2024--2026", "2024 to 2026"),
]

EN = "--"
total_before = total_after = 0

for name in FILES:
    p = HERE / name
    t = p.read_text(encoding="utf-8")
    # count only real en dashes, not '---' or comment rules
    before = len(re.findall(r"(?<!-)--(?!-)", t))
    total_before += before
    if before == 0:
        print(f"  {name}: none")
        continue

    orig = t
    for a, b in EXPLICIT:
        t = t.replace(a, b)

    # remaining word--word compounds -> hyphen, but never inside a % comment rule
    def repl(m):
        return f"{m.group(1)}-{m.group(2)}"

    t = re.sub(r"(?<!-)(\w)--(\w)(?!-)", repl, t)

    after = len(re.findall(r"(?<!-)--(?!-)", t))
    total_after += after
    p.write_text(t, encoding="utf-8")
    print(f"  {name}: {before} -> {after}")

    # show what changed
    for ol, nl in zip(orig.splitlines(), t.splitlines()):
        if ol != nl:
            print(f"      - {ol.strip()[:78]}")
            print(f"      + {nl.strip()[:78]}")

print(f"\ntotal en dashes: {total_before} -> {total_after}")
