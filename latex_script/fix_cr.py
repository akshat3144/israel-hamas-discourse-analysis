# -*- coding: utf-8 -*-
"""Repair stray carriage returns that ate the backslash of LaTeX commands
(e.g. '\\ref' became CR + 'ef')."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
CR = chr(13)
BS = chr(92)

for name in ["body_new.tex", "rq2_body.tex", "rw_conflict.tex", "rw_platform.tex"]:
    p = HERE / name
    raw = p.read_bytes().decode("utf-8")
    if CR not in raw:
        print(f"  clean: {name}")
        continue
    before = raw.count(CR)
    # CR immediately followed by a letter = a mangled LaTeX command
    fixed = ""
    i = 0
    repaired = 0
    while i < len(raw):
        ch = raw[i]
        if ch == CR:
            nxt = raw[i + 1] if i + 1 < len(raw) else ""
            if nxt.isalpha():          # \ref -> CR + "ef"  => restore backslash
                fixed += BS
                repaired += 1
            elif nxt == "\n":          # plain CRLF line ending -> drop the CR
                pass
            else:
                pass
        else:
            fixed += ch
        i += 1
    p.write_text(fixed, encoding="utf-8", newline="\n")
    print(f"  {name}: {before} CR found, {repaired} LaTeX commands restored")

print("done")
