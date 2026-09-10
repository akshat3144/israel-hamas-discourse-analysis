# -*- coding: utf-8 -*-
"""Repair the Table 1 block in body_new.tex: restore '\\\\' row terminators."""
from pathlib import Path

BS = chr(92)  # backslash
p = Path(__file__).resolve().parent / "body_new.tex"
t = p.read_text(encoding="utf-8")

start = t.index(BS + "begin{table*}[t]")
end = t.index(BS + "end{table*}", start) + len(BS + "end{table*}")
block = t[start:end]

fixed_lines = []
for line in block.splitlines():
    stripped = line.rstrip()
    # a tabular row ends with a single backslash -> must be a double backslash
    if stripped.endswith(BS) and not stripped.endswith(BS + BS):
        stripped = stripped[:-1] + BS + BS
    fixed_lines.append(stripped)
fixed = "\n".join(fixed_lines)

t = t[:start] + fixed + t[end:]
p.write_text(t, encoding="utf-8")

n = sum(1 for l in fixed.splitlines() if l.rstrip().endswith(BS + BS))
print(f"repaired {n} row terminators in Table 1")
